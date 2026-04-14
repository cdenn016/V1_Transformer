"""
Active Inference / Expected Free Energy Extension
==================================================

Experimental module that augments the VFE E-step with active-inference terms
from Karl Friston's Active Inference framework.  The module is isolated here
so that the core VFE file (variational_ffn.py) stays focused on the principal
gauge-theoretic free energy computation.

This file contains:

1. ``_compute_active_inference_gradient`` — EFE pragmatic + epistemic terms.
2. ``_compute_distillation_gradient`` — Bootstrap self-distillation (BYOL-style).
3. ``compute_ai_gradients`` — Dispatch wrapper that resolves refs from the FFN
   instance and calls both helpers.
4. ``configure_ffn_active_inference`` — Called from ``blocks.py`` to set the 9
   AI instance attributes on a freshly-constructed FFN.
5. ``wire_readout_references`` — Called from ``model.py`` after the full module
   hierarchy is built to plumb PriorBank / W_out fallback references.

Mathematical background
-----------------------
The Active Inference free energy augments the VFE with:

    F_AI = lambda_prag * H[p_pred(v | mu_i)]            (pragmatic: entropy min)
         - lambda_epi  * MI(v; mu | q_i)                (epistemic: BALD MI)

Both terms are differentiable in mu.  Gradients are computed via
``torch.autograd.grad`` on a freshly-detached mu leaf inside
``torch.enable_grad()``.  The graphs are local and do not perturb the
surrounding analytic VFE gradient graph.

The bootstrap self-distillation term (Sec. 3.3):

    L_distill,i = lambda * CE(sg[p_pred(mu_tilde_i)],  p_pred(mu_i))

where mu_tilde_i = sum_j sg[beta_ij] * sg[Omega_ij mu_j] (transported
attention-weighted mean, fully stop-gradient).

All three terms are gated by their respective weights.  When all weights are
zero (the default configuration), every function in this module is a no-op
that adds zero overhead to the forward pass.
"""

import math
import logging
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Active inference / Expected Free Energy gradient helper
# =============================================================================
#
# Augments the VFE E-step with the two terms that active inference prescribes
# when the agent does not have direct access to the observation:
#
#   F_AI = lambda_prag * H[p_pred(v|mu_i)]                        (pragmatic)
#        - lambda_epi  * MI(v; mu | q_i)                           (epistemic)
#
# Pragmatic term:
#   p_pred(v|mu_i) is the PriorBank readout (KL-based softmax over vocab).
#   Minimizing its entropy makes the belief commit to a confident prediction
#   without target leak.  This is the "self-observation" component: the agent
#   conditions on what its own current belief most strongly predicts.
#
# Epistemic term (BALD-style mutual information):
#   MI(v; mu | q_i) = H[E_{mu ~ q_i} p(v|mu)] - E_{mu ~ q_i}[H[p(v|mu)]]
#   Estimated by Monte Carlo: sample S values mu_s ~ N(mu_i, Sigma_i), evaluate
#   the readout at each, then compute the predictive-distribution disagreement.
#   High MI <=> the predictive distribution depends meaningfully on which
#   element of q_i you sample <=> belief uncertainty carries decision-relevant
#   information <=> updating the belief is epistemically valuable.
#   The negative sign in F_AI means we *maximize* MI in the E-step, which
#   counter-balances the pragmatic term's tendency toward self-reinforcement.
#
# Both terms are differentiable in mu_i.  We compute the gradient via
# torch.autograd.grad on a freshly-detached mu leaf, so the autograd graph is
# local to this function and does not perturb the rest of the E-step graph.
#
# CRITICAL: torch.inference_mode() (used by model.generate) marks tensors as
# inference tensors that cannot be promoted to require_grad even inside
# enable_grad().  Detect and skip the EFE term in inference-mode contexts --
# the rest of the E-step still runs normally and produces correct samples,
# the EFE bonus just isn't applied during pure decoding.

def _compute_active_inference_gradient(
    mu_current: torch.Tensor,
    sigma_current: torch.Tensor,
    prior_bank,
    pragmatic_weight: float,
    epistemic_weight: float,
    epistemic_samples: int,
    decode_tau: float,
    w_out: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    r"""Compute dF_AI/d(mu, sigma) via autograd through the readout.

    Returns Euclidean gradients for BOTH μ and Σ.  When
    ``unified_natgrad=True`` these are folded into the VFE gradient sum
    before the Fisher natural-gradient projection, so all terms share the
    same information-geometric descent on the belief manifold (Amari 1998:
    the Fisher metric is a property of the belief parameterization, not the
    loss function).

    Uses PriorBank.decode when prior_bank is provided (principled KL-based
    readout).  Falls back to a linear W_out projection followed by softmax
    when prior_bank is None.  The linear fallback is less principled (it
    ignores sigma and uses Euclidean-not-KL distance) but keeps the EFE path
    functional for configurations that don't use PriorBank.

    Memory model
    ------------
    Define U = B * N * V * 4 bytes (one diag-logits buffer in float32).  The
    naive joint-graph implementation peaks at (3S + 2) * U because it holds
    every sample's logits + softmax + log-softmax tensors and the (S, B, N, V)
    probs_stack in the autograd graph simultaneously.  At K=20, B=64, N=128,
    V=50257, S=4 that is ~14 U ≈ 23 GB.

    The two-pass refactor below uses the algebraic identity

        MI = E_s[KL(p_s || p̄)],   p̄ = (1/S) Σ_s p_s,

    so log p̄ can be precomputed as a *constant* in pass 1 (no_grad,
    streaming Welford-style accumulation), and pass 2 then builds /
    backwards / frees the autograd graph for one sample at a time.  Peak
    retained graph drops to a single sample (~5 U), saving (3S - 3) * U.

    Sigma gradient
    --------------
    Σ enters two paths:

    1. **Readout** — PriorBank.decode uses Σ in the KL-based logits
       (trace(Σ_q/Σ_p) terms).  Linear W_out ignores Σ, giving zero sigma
       gradient for the pragmatic term.

    2. **Reparameterization** (epistemic only) — samples are drawn via
       z_s = μ + √Σ · ε (diagonal) or z_s = μ + L·ε (Cholesky).  The
       diagonal case tracks Σ through √Σ for the explore-exploit gradient
       ∂MI/∂Σ.  Full covariance tracks Σ through the readout only (Cholesky
       differentiation is expensive; diagonal captures the main effect).

    Args:
        mu_current: (B, N, K) current belief mean (will be detached internally).
        sigma_current: (B, N, K) diagonal or (B, N, K, K) full covariance.
        prior_bank: PriorBank instance providing decode(mu, sigma, tau) -> logits.
        pragmatic_weight: lambda_prag.  0.0 disables the pragmatic term.
        epistemic_weight: lambda_epi.  0.0 disables the epistemic term.
        epistemic_samples: S, number of MC samples for BALD MI estimate.
        decode_tau: temperature for PriorBank.decode (1.0 = standard).
        w_out: optional (V, K) linear projection tensor used as fallback
               readout when prior_bank is None.

    Returns:
        (grad_mu, grad_sigma) — each (B, N, K) for diagonal or appropriate
        shape for full covariance.  Either may be None if all terms are
        disabled, readout is unavailable, or we are in inference-mode.
    """
    _none_pair = (None, None)
    if prior_bank is None and w_out is None:
        return _none_pair
    if pragmatic_weight <= 0.0 and epistemic_weight <= 0.0:
        return _none_pair
    if torch.is_inference_mode_enabled():
        return _none_pair

    is_diagonal = (sigma_current.dim() == mu_current.dim())
    use_prior_bank = (prior_bank is not None)

    # Guard against decode_tau=0 for BOTH readout paths.  PriorBank.decode
    # internally divides by tau; the W_out fallback does too.  Without the
    # guard, tau=0 produces NaN logits and corrupts mu via the EFE gradient.
    _tau_safe = max(float(decode_tau), 1e-8)

    def _readout(mu_input: torch.Tensor, sigma_arg: torch.Tensor) -> torch.Tensor:
        """Return (B, N, V) logits from either PriorBank or W_out."""
        if use_prior_bank:
            return prior_bank.decode(mu_input, sigma_arg, tau=_tau_safe)
        else:
            return torch.matmul(mu_input, w_out.T) / _tau_safe

    # Detached float32 copies shared across pragmatic + per-sample epistemic.
    # Both autograd passes use the same mu_f32 leaf seed, so per-sample
    # gradients are gradients w.r.t. the same mu_current point.
    # sigma_f32 is the base value; fresh sigma_var leaves are created per
    # autograd context to track ∂G/∂Σ.
    mu_f32 = mu_current.detach().to(torch.float32)
    sigma_f32 = sigma_current.detach().to(torch.float32)
    grad_mu_accum: Optional[torch.Tensor] = None
    grad_sigma_accum: Optional[torch.Tensor] = None

    # ----- Pragmatic: minimize H[p_pred(v|mu, sigma)] -----------------
    # Single decode + autograd.grad call.  retain_graph=False frees the
    # graph before the epistemic loop runs, so the two terms do not
    # coexist in memory.  Sigma is tracked through the readout: with
    # PriorBank.decode the KL-based logits depend on Σ; with linear W_out
    # the dependence is zero and autograd returns a zero gradient.
    if pragmatic_weight > 0.0:
        with torch.enable_grad():
            mu_var = mu_f32.clone().requires_grad_(True)
            sigma_var = sigma_f32.clone().requires_grad_(True)
            logits = _readout(mu_var, sigma_var)  # (B, N, V)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)  # (B, N)
            pragmatic_term = pragmatic_weight * entropy.mean()
            grad_prag_mu, grad_prag_sigma = torch.autograd.grad(
                pragmatic_term, [mu_var, sigma_var],
                create_graph=False, retain_graph=False,
            )
        grad_mu_accum = grad_prag_mu.detach()
        grad_sigma_accum = grad_prag_sigma.detach()
        del logits, log_probs, probs, entropy, pragmatic_term
        del mu_var, sigma_var, grad_prag_mu, grad_prag_sigma

    # ----- Epistemic: -MI(v; mu | q_i) via Welford-style 2-pass backward -----
    # Sigma enters two paths: (1) the readout _readout(mu_s, sigma_var) and
    # (2) the reparameterization mu_s = mu + sqrt(sigma) * noise (diagonal).
    # For full covariance, path (2) would require differentiating through
    # Cholesky — expensive.  We track sigma through the readout in all cases
    # and additionally through the reparameterization in the diagonal case.
    _full_cov_epi = not is_diagonal  # full-cov: readout-only sigma gradient
    if epistemic_weight > 0.0 and epistemic_samples > 0:
        # ---- Build the noise sampler (detached, for Pass 1) ----
        # Store RAW noise ε ~ N(0,I) so Pass 2 can apply tracked sigma.
        if is_diagonal:
            std_detached = sigma_f32.clamp(min=1e-6).sqrt()

            def _sample_scaled_noise(template: torch.Tensor) -> torch.Tensor:
                return torch.randn_like(template) * std_detached
        else:
            # Full covariance: use Cholesky for correct correlated sampling.
            # Falls back to diagonal std on numerical failure (non-SPD Sigma).
            sigma_spd = 0.5 * (sigma_f32 + sigma_f32.transpose(-1, -2))
            K_dim = sigma_spd.shape[-1]
            eye_k = torch.eye(K_dim, device=sigma_spd.device, dtype=sigma_spd.dtype)
            sigma_spd = sigma_spd + 1e-6 * eye_k
            _cholesky_ok = True
            try:
                L_chol = torch.linalg.cholesky(sigma_spd)  # (B, N, K, K)

                def _sample_scaled_noise(template: torch.Tensor) -> torch.Tensor:
                    z = torch.randn_like(template)  # (B, N, K)
                    return torch.einsum('bnkj,bnj->bnk', L_chol, z)
            except Exception:
                _cholesky_ok = False
                sigma_diag = torch.diagonal(sigma_spd, dim1=-2, dim2=-1)
                std_detached = sigma_diag.clamp(min=1e-6).sqrt()

                def _sample_scaled_noise(template: torch.Tensor) -> torch.Tensor:
                    return torch.randn_like(template) * std_detached

        # ---- Pass 1: streaming p̄ accumulation, no autograd graph ----
        # Save RAW noise ε for diagonal (so Pass 2 can apply tracked sigma)
        # or pre-scaled noise for full-cov (sigma gradient via readout only).
        raw_noise_list: List[torch.Tensor] = []
        scaled_noise_list: List[torch.Tensor] = []
        with torch.no_grad():
            probs_avg: Optional[torch.Tensor] = None
            for _s in range(epistemic_samples):
                if is_diagonal:
                    # Store raw ε, compute scaled noise for Pass 1
                    raw_z = torch.randn_like(mu_f32)
                    raw_noise_list.append(raw_z)
                    noise = raw_z * std_detached
                else:
                    # Store pre-scaled noise (sigma gradient via readout only)
                    noise = _sample_scaled_noise(mu_f32)
                    scaled_noise_list.append(noise)
                mu_s = mu_f32 + noise
                logits_s = _readout(mu_s, sigma_f32)
                # Streaming mean: avoid materializing (S, B, N, V) probs_stack.
                _contrib = F.softmax(logits_s, dim=-1) / epistemic_samples
                probs_avg = _contrib if probs_avg is None else probs_avg + _contrib
                del mu_s, logits_s, _contrib
            # log p̄ used as a stop-gradient constant in pass 2.
            log_probs_avg_const = probs_avg.clamp(min=1e-12).log()
            del probs_avg

        # ---- Pass 2: per-sample autograd, accumulate gradient manually ----
        # Each iteration: rebuild the sample's decode graph, compute
        # L_s = -(ε / S) · KL(p_s || sg(p̄_const)), backward, free.
        # Sigma is tracked through: (a) readout (always) and (b)
        # reparameterization mu_s = mu + sqrt(sigma_var) * raw_z (diagonal).
        epi_per_sample_weight = -epistemic_weight / epistemic_samples
        for _s in range(epistemic_samples):
            with torch.enable_grad():
                mu_var_s = mu_f32.clone().requires_grad_(True)
                sigma_var_s = sigma_f32.clone().requires_grad_(True)
                if is_diagonal:
                    # Reparameterization with tracked sigma
                    std_tracked = sigma_var_s.clamp(min=1e-6).sqrt()
                    mu_s_grad = mu_var_s + raw_noise_list[_s] * std_tracked
                else:
                    # Full-cov: use pre-scaled noise (sigma via readout only)
                    mu_s_grad = mu_var_s + scaled_noise_list[_s]
                logits_s_grad = _readout(mu_s_grad, sigma_var_s)
                log_probs_s_grad = F.log_softmax(logits_s_grad, dim=-1)
                probs_s_grad = log_probs_s_grad.exp()
                # KL(p_s || p̄) = Σ_v p_s · (log p_s - log p̄)
                kl_s = (probs_s_grad
                        * (log_probs_s_grad - log_probs_avg_const)).sum(dim=-1)
                F_s = epi_per_sample_weight * kl_s.mean()
                grad_s_mu, grad_s_sigma = torch.autograd.grad(
                    F_s, [mu_var_s, sigma_var_s],
                    create_graph=False, retain_graph=False,
                )
            _gs_mu = grad_s_mu.detach()
            _gs_sigma = grad_s_sigma.detach()
            grad_mu_accum = (_gs_mu if grad_mu_accum is None
                             else grad_mu_accum + _gs_mu)
            grad_sigma_accum = (_gs_sigma if grad_sigma_accum is None
                                else grad_sigma_accum + _gs_sigma)
            # Drop Python refs so the storage can be reclaimed before the
            # next sample's decode allocates.
            del mu_var_s, sigma_var_s, mu_s_grad, logits_s_grad
            del log_probs_s_grad, probs_s_grad, kl_s, F_s
            del grad_s_mu, grad_s_sigma, _gs_mu, _gs_sigma

    if grad_mu_accum is None:
        return (None, None)
    _out_dtype = mu_current.dtype
    _grad_mu = grad_mu_accum.to(_out_dtype)
    _grad_sigma = grad_sigma_accum.to(_out_dtype) if grad_sigma_accum is not None else None
    return (_grad_mu, _grad_sigma)


# =============================================================================
# Bootstrap self-distillation gradient (BYOL-style E-step term)
# =============================================================================
#
# Adds a third active-inference term to the E-step free energy:
#
#   L_distill,i  =  lambda * CE( sg[p_pred(mu_tilde_i)],  p_pred(mu_i) )
#
# where mu_tilde_i = sum_j sg[beta_ij] * sg[Omega_ij mu_j]  is the
# attention-aggregated transported belief (the same quantity the attention
# sublayer computes as its message aggregation).  BOTH stop-gradients are
# essential:
#
#   sg on the target distribution    -> BYOL-style collapse prevention
#   sg on beta_ij (and Omega_ij via phi) -> prevents "attend-to-twins" collapse
#                                        (SymPy-verified: without this,
#                                         the gradient on the attention
#                                         scores is  beta_k * (CE_k - <CE>),
#                                         which drives beta onto the
#                                         neighbours that already agree)
#
# See docs/bootstrap_self_distillation.md for the full derivation and
# docs/_bootstrap_distill_verify.py for the symbolic verification.

def _compute_distillation_gradient(
    mu_current: torch.Tensor,
    sigma_current: torch.Tensor,
    head_beta_bep: List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
    irrep_dims: List[int],
    prior_bank,
    weight: float,
    normalize_by_logv: bool,
    decode_tau: float,
    w_out: Optional[torch.Tensor] = None,
    mode: str = 'aggregated',
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    r"""Compute dL_distill/d(mu, sigma) via autograd through the readout.

    Builds the aggregated transported belief mu_tilde using per-head
    block-diagonal einsums, runs the PriorBank (or W_out fallback) decode at
    both mu and mu_tilde, and computes the cross-entropy with a stop-gradient
    on the target.  Sigma is tracked through the local readout so PriorBank
    configurations get a ∂L_distill/∂Σ contribution.

    Args:
        mu_current: (B, N, K) current belief mean.
        sigma_current: (B, N, K) or (B, N, K, K) current covariance.
        head_beta_bep: list of (beta_h, (exp_phi_h, exp_neg_phi_h)) per head.
            beta_h is (B, N, N), the block exp pairs are each (B, N, d_h, d_h).
            For single-beta configurations the same beta is repeated across heads.
        irrep_dims: list [d_1, d_2, ...] of per-head block dimensions.  The
            total K = sum(d_h).
        prior_bank: PriorBank instance (preferred readout path).
        weight: lambda_distill.
        normalize_by_logv: if True, divide the CE by log(V) so the loss
            is unit-order regardless of vocabulary size.
        decode_tau: temperature for PriorBank.decode.
        w_out: optional (V, K) linear fallback readout used when prior_bank
            is None.
        mode: 'aggregated' uses mu_tilde as the target site (single readout);
              'per_pair' not yet implemented.

    Returns:
        (grad_mu, grad_sigma) — each (B, N, K) or appropriate shape.
        Either may be None if disabled / unavailable / inference_mode.
    """
    _none_pair = (None, None)
    if weight <= 0.0:
        return _none_pair
    if prior_bank is None and w_out is None:
        return _none_pair
    if torch.is_inference_mode_enabled():
        return _none_pair
    if mode != 'aggregated':
        # Per-pair mode reserved for future work -- fall back to aggregated.
        mode = 'aggregated'

    B, N, K = mu_current.shape
    assert sum(irrep_dims) == K, f"irrep_dims sum {sum(irrep_dims)} != K {K}"
    assert len(head_beta_bep) == len(irrep_dims), (
        f"head_beta_bep length {len(head_beta_bep)} != num heads {len(irrep_dims)}"
    )

    is_diagonal = (sigma_current.dim() == mu_current.dim())

    # ----- Build mu_tilde with full stop-gradient -----
    # mu_tilde_i^(h) = sum_j beta_ij^(h) * (exp_phi[i] * exp_neg_phi[j]) * mu_j^(h)
    # Everything here is detached: beta, exp_phi, exp_neg_phi, and mu itself.  The
    # target is therefore a pure stop-gradient constant from autograd's perspective.
    _f32 = torch.float32
    mu_detached = mu_current.detach().to(_f32)
    mu_tilde = torch.zeros_like(mu_detached)

    block_start = 0
    for h, d_h in enumerate(irrep_dims):
        block_end = block_start + d_h
        mu_h = mu_detached[:, :, block_start:block_end]  # (B, N, d_h)
        beta_h, (exp_phi_h, exp_neg_phi_h) = head_beta_bep[h]
        beta_d = beta_h.detach().to(_f32)
        exp_phi_d = exp_phi_h.detach().to(_f32)
        exp_neg_phi_d = exp_neg_phi_h.detach().to(_f32)
        # Einsum contraction:
        #   exp_phi_d[b, i, k, l]   * (exp_neg_phi_d[b, j, l, m] * mu_h[b, j, m])
        #   summed over l, m,  then weighted by beta_d[b, i, j] and summed over j
        mu_tilde_h = torch.einsum(
            'bij,bikl,bjlm,bjm->bik',
            beta_d, exp_phi_d, exp_neg_phi_d, mu_h,
        )
        mu_tilde[:, :, block_start:block_end] = mu_tilde_h
        block_start = block_end

    # ----- CE loss and autograd through mu_current's readout -----
    # Sigma is tracked through the local decode so PriorBank configurations
    # get ∂L_distill/∂Σ.  With linear W_out, sigma doesn't appear in the
    # readout and autograd returns a zero gradient.
    _tau_safe = max(float(decode_tau), 1e-8)
    use_prior_bank = (prior_bank is not None)

    with torch.enable_grad():
        mu_var = mu_current.detach().to(_f32).requires_grad_(True)
        sigma_var = sigma_current.detach().to(_f32).requires_grad_(True)

        def _decode(mu_in: torch.Tensor,
                    sigma_in: torch.Tensor) -> torch.Tensor:
            if use_prior_bank:
                return prior_bank.decode(mu_in, sigma_in, tau=_tau_safe)
            else:
                return torch.matmul(mu_in, w_out.T) / _tau_safe

        # Target: decode at mu_tilde (fully detached, but run inside enable_grad
        # so inference_mode tensors don't cause issues)
        with torch.no_grad():
            sigma_detached = sigma_current.detach().to(_f32)
            target_logits = _decode(mu_tilde, sigma_detached)
            target_probs = F.softmax(target_logits, dim=-1)
        # Belt-and-suspenders: ensure target_probs has no grad_fn
        target_probs = target_probs.detach()

        # Local: decode at mu_var (grad tracked through mu AND sigma)
        local_logits = _decode(mu_var, sigma_var)
        local_log_probs = F.log_softmax(local_logits, dim=-1)

        # Cross-entropy: -sum_v target_v * log local_v  per position
        ce_per_pos = -(target_probs * local_log_probs).sum(dim=-1)  # (B, N)

        if normalize_by_logv:
            V = local_log_probs.shape[-1]
            ce_per_pos = ce_per_pos / math.log(max(V, 2))

        total_loss = weight * ce_per_pos.mean()

        grad_distill_mu, grad_distill_sigma = torch.autograd.grad(
            total_loss, [mu_var, sigma_var],
            create_graph=False, retain_graph=False,
        )

    _out_dtype = mu_current.dtype
    return (grad_distill_mu.detach().to(_out_dtype),
            grad_distill_sigma.detach().to(_out_dtype))


# =============================================================================
# Dispatch helpers — called from VariationalFFNDynamic._vfe_iteration
# =============================================================================

def compute_ai_gradients(
    ffn,
    mu_current: torch.Tensor,
    sigma_current: torch.Tensor,
    W_out: Optional[torch.Tensor],
    beta_current: torch.Tensor,
    beta_heads,
    cached_block_exp_pairs,
    irrep_dims,
    multihead_vfe: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
           Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute both the EFE and distillation gradients in one call.

    Reads ``ffn._ai_*`` attributes (set by ``configure_ffn_active_inference``)
    and resolves ``_prior_bank_ref`` / ``_ai_w_out_ref`` from ``ffn.__dict__``
    using the same list-unwrapping pattern as the original inlined code.

    Returns:
        (grad_efe_mu, grad_efe_sigma, grad_distill_mu, grad_distill_sigma)
        — any element may be None if its term is disabled, no readout is
        wired, or we are in inference_mode.
    """
    # ----- Resolve shared readout references -----
    _ai_bank_raw = ffn.__dict__.get('_prior_bank_ref', None)
    _ai_bank = _ai_bank_raw[0] if isinstance(_ai_bank_raw, list) else _ai_bank_raw

    _ai_wout_ref = ffn.__dict__.get('_ai_w_out_ref', None)
    if W_out is not None:
        _w_out_tensor: Optional[torch.Tensor] = W_out
    elif _ai_wout_ref is not None:
        _proj = _ai_wout_ref[0] if isinstance(_ai_wout_ref, list) else _ai_wout_ref
        _w_out_tensor = _proj.weight  # (V, K)
    else:
        _w_out_tensor = None

    # ----- EFE gradient (pragmatic + epistemic) -----
    grad_efe_mu: Optional[torch.Tensor] = None
    grad_efe_sigma: Optional[torch.Tensor] = None
    _ai_enabled = getattr(ffn, '_ai_enabled', False)
    _ai_prag = getattr(ffn, '_ai_pragmatic_weight', 0.0)
    _ai_epi = getattr(ffn, '_ai_epistemic_weight', 0.0)
    if _ai_enabled and (_ai_prag > 0.0 or _ai_epi > 0.0):
        _ai_samples = getattr(ffn, '_ai_epistemic_samples', 4)
        _ai_tau = getattr(ffn, '_ai_decode_tau', 1.0)
        if _ai_bank is not None or _w_out_tensor is not None:
            grad_efe_mu, grad_efe_sigma = _compute_active_inference_gradient(
                mu_current=mu_current,
                sigma_current=sigma_current,
                prior_bank=_ai_bank,
                pragmatic_weight=_ai_prag,
                epistemic_weight=_ai_epi,
                epistemic_samples=_ai_samples,
                decode_tau=_ai_tau,
                w_out=_w_out_tensor,
            )
            # Write to VFE debug dict if present
            try:
                import transformer.core.vfe_utils as _vfe_utils_mod
                if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
                    if grad_efe_mu is not None:
                        _vfe_utils_mod._VFE_GRAD_DEBUG['ai_efe_mu_grad'] = (
                            _vfe_utils_mod._grad_norm(grad_efe_mu))
                    if grad_efe_sigma is not None:
                        _vfe_utils_mod._VFE_GRAD_DEBUG['ai_efe_sigma_grad'] = (
                            _vfe_utils_mod._grad_norm(grad_efe_sigma))
            except Exception:
                pass

    # ----- Distillation gradient -----
    grad_distill_mu: Optional[torch.Tensor] = None
    grad_distill_sigma: Optional[torch.Tensor] = None
    _ai_distill_weight = getattr(ffn, '_ai_distill_weight', 0.0)
    if _ai_distill_weight > 0.0 and irrep_dims is not None:
        _bep_list = cached_block_exp_pairs
        if _bep_list is not None and (_ai_bank is not None or _w_out_tensor is not None):
            # Multi-head: use per-head betas; single-beta: replicate shared beta
            if multihead_vfe and beta_heads is not None and len(beta_heads) == len(irrep_dims):
                _head_beta_bep = [
                    (beta_heads[h], _bep_list[h]) for h in range(len(irrep_dims))
                ]
            else:
                _head_beta_bep = [
                    (beta_current, _bep_list[h]) for h in range(len(irrep_dims))
                ]
            grad_distill_mu, grad_distill_sigma = _compute_distillation_gradient(
                mu_current=mu_current,
                sigma_current=sigma_current,
                head_beta_bep=_head_beta_bep,
                irrep_dims=list(irrep_dims),
                prior_bank=_ai_bank,
                weight=_ai_distill_weight,
                normalize_by_logv=getattr(ffn, '_ai_distill_normalize', True),
                decode_tau=getattr(ffn, '_ai_decode_tau', 1.0),
                w_out=_w_out_tensor,
                mode=getattr(ffn, '_ai_distill_mode', 'aggregated'),
            )
            # Write to VFE debug dict if present
            try:
                import transformer.core.vfe_utils as _vfe_utils_mod
                if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
                    if grad_distill_mu is not None:
                        _vfe_utils_mod._VFE_GRAD_DEBUG['ai_distill_mu_grad'] = (
                            _vfe_utils_mod._grad_norm(grad_distill_mu))
                    if grad_distill_sigma is not None:
                        _vfe_utils_mod._VFE_GRAD_DEBUG['ai_distill_sigma_grad'] = (
                            _vfe_utils_mod._grad_norm(grad_distill_sigma))
            except Exception:
                pass

    return grad_efe_mu, grad_efe_sigma, grad_distill_mu, grad_distill_sigma


# =============================================================================
# Setup helpers — called from blocks.py and model.py
# =============================================================================

def configure_ffn_active_inference(ffn, cfg) -> None:
    """Read active-inference fields from BlockConfig and set as instance attributes on ffn.

    Replaces the 9-attribute assignment block in ``GaugeTransformerBlock.__init__``
    (blocks.py lines 276-291).  Also initializes ``_prior_bank_ref = None`` in
    ``ffn.__dict__`` so the later model-side wiring can bypass nn.Module
    sub-module registration.

    Args:
        ffn: A VariationalFFNDynamic instance (duck-typed).
        cfg: A BlockConfig instance (or any object with the active-inference
            attributes accessible via getattr).
    """
    # Master EFE toggle and weights
    ffn._ai_enabled = getattr(cfg, 'active_inference', False)
    ffn._ai_pragmatic_weight = getattr(cfg, 'active_inference_pragmatic_weight', 1.0)
    ffn._ai_epistemic_weight = getattr(cfg, 'active_inference_epistemic_weight', 0.5)
    ffn._ai_epistemic_samples = getattr(cfg, 'active_inference_epistemic_samples', 4)
    ffn._ai_decode_tau = getattr(cfg, 'active_inference_decode_tau', 1.0)
    # Bootstrap self-distillation (can be enabled independently of the
    # master active_inference toggle -- this is a standalone term).
    ffn._ai_distill_weight = getattr(cfg, 'active_inference_distill_weight', 0.0)
    ffn._ai_distill_normalize = getattr(cfg, 'active_inference_distill_normalize', True)
    ffn._ai_distill_mode = getattr(cfg, 'active_inference_distill_mode', 'aggregated')
    # _prior_bank_ref defaults to None; the model wires it in after the
    # full module hierarchy is constructed.
    ffn.__dict__.setdefault('_prior_bank_ref', None)


def wire_readout_references(transformer_stack, prior_bank, out_proj_module, logger=None) -> None:
    """Plumb PriorBank and W_out fallback references into each block's FFN.

    Replaces the active-inference wiring block in ``GaugeTransformerLM.__init__``
    (model.py lines 275-334).  Must be called AFTER both
    ``GaugeTransformerStack`` and ``out_proj`` are constructed.

    Uses ``__dict__`` assignment to bypass ``nn.Module.__setattr__`` so the
    already-owned PriorBank is not double-registered as a sub-module of the FFN.

    Emits:
        - INFO  when EFE is active with PriorBank readout.
        - WARNING when EFE is active but no PriorBank (fallback to W_out).
        - WARNING when active_inference is combined with closed_form_e_step.
        - WARNING when active_inference is combined with use_deq.

    Args:
        transformer_stack: A GaugeTransformerStack whose ``.blocks`` attribute
            is a list of GaugeTransformerBlock instances.
        prior_bank: The model's PriorBank (may be None).
        out_proj_module: The model's ``nn.Linear`` output projection.
        logger: Optional Python logger.  Falls back to module-level logger.
    """
    _log = logger if logger is not None else globals()['logger']

    # Plumb PriorBank reference to each FFN for EFE pragmatic/epistemic terms.
    # __dict__ assignment bypasses nn.Module sub-module auto-registration
    # (PriorBank is owned by the model, not the FFN; re-registering would
    # double-count parameters).
    if prior_bank is not None:
        for _block in transformer_stack.blocks:
            _block.ffn.__dict__['_prior_bank_ref'] = prior_bank

    # Active inference diagnostics + W_out fallback wiring.
    # Runs AFTER out_proj is created so the fallback can grab it.
    _ai_on = any(getattr(_b.ffn, '_ai_enabled', False) for _b in transformer_stack.blocks)
    if _ai_on:
        if prior_bank is not None:
            _log.info(
                "[GaugeTransformerLM] active_inference=True with PriorBank readout -- "
                "EFE E-step augmentation is active (pragmatic + epistemic terms via PriorBank.decode)"
            )
        else:
            # Fallback: wire out_proj reference so the EFE helper can use
            # the linear projection readout instead of PriorBank.  List
            # wrapper bypasses nn.Module sub-module auto-registration.
            for _block in transformer_stack.blocks:
                _block.ffn.__dict__['_ai_w_out_ref'] = [out_proj_module]
            _log.warning(
                "[GaugeTransformerLM] active_inference=True but PriorBank is disabled -- "
                "falling back to W_out linear readout for EFE pragmatic/epistemic terms. "
                "Enable use_prior_bank=True for the principled KL-based decode."
            )

        # Hard errors for configurations where EFE silently does nothing or
        # produces a biased M-step gradient.  These were previously logged
        # as warnings but the audit confirmed that the forward/backward
        # mismatch is a correctness bug, not a performance hint — raise so
        # misconfigurations fail fast instead of training incorrectly.
        _closed_form = any(
            getattr(_b.ffn, 'closed_form_e_step', False)
            for _b in transformer_stack.blocks
        )
        if _closed_form:
            raise ValueError(
                "[GaugeTransformerLM] active_inference=True is INCOMPATIBLE with "
                "closed_form_e_step=True. The closed-form E-step bypasses the "
                "iterative VFE loop where the EFE gradient is applied, so the EFE "
                "pragmatic/epistemic terms would have NO EFFECT. Disable one of "
                "the two flags before constructing the model."
            )
        _deq_on = any(getattr(_b.ffn, 'use_deq', False) for _b in transformer_stack.blocks)
        if _deq_on:
            raise ValueError(
                "[GaugeTransformerLM] active_inference=True with use_deq=True "
                "produces a biased M-step gradient: the DEQ implicit-differentiation "
                "backward pass builds its Jacobian from the VFE-only step operator, "
                "NOT the VFE+EFE composite. The forward pass includes EFE but the "
                "backward fixed-point correction is applied to the wrong operator. "
                "Disable one of the two flags before constructing the model."
            )

    # Bootstrap self-distillation diagnostics (standalone toggle, separate
    # from the active_inference master gate).  The distillation term can
    # be enabled without the pragmatic+epistemic EFE terms, but its
    # interaction with the pragmatic term in particular is a stability
    # concern, not just a tuning question.  See
    # docs/bootstrap_self_distillation.md Section 12 for the full argument.
    _distill_on = any(
        getattr(_b.ffn, '_ai_distill_weight', 0.0) > 0.0
        for _b in transformer_stack.blocks
    )
    if _distill_on:
        # Distillation gradients are folded into the VFE gradient sum
        # inside the iterative E-step loop, so it is incompatible with
        # paths that bypass the loop.  Check even when the master _ai_on
        # toggle is off, since distillation can be enabled independently.
        _closed_form_distill = any(
            getattr(_b.ffn, 'closed_form_e_step', False)
            for _b in transformer_stack.blocks
        )
        if _closed_form_distill:
            raise ValueError(
                "[GaugeTransformerLM] active_inference_distill_weight > 0 is "
                "INCOMPATIBLE with closed_form_e_step=True. The closed-form "
                "E-step bypasses the iterative loop where the distillation "
                "gradient is applied, so it would have NO EFFECT. Disable "
                "one of the two flags."
            )
        _deq_distill = any(
            getattr(_b.ffn, 'use_deq', False)
            for _b in transformer_stack.blocks
        )
        if _deq_distill:
            raise ValueError(
                "[GaugeTransformerLM] active_inference_distill_weight > 0 "
                "with use_deq=True produces a biased M-step gradient: the "
                "DEQ backward builds its Jacobian from the VFE-only step "
                "operator, not the VFE+distillation composite. Disable one "
                "of the two flags."
            )

        # Resolve W_out fallback for distillation too, even if master
        # active_inference toggle is off.  The distillation term uses the
        # same readout resolution path as the EFE helpers.
        if prior_bank is None:
            for _block in transformer_stack.blocks:
                _block.ffn.__dict__.setdefault('_ai_w_out_ref', [out_proj_module])

        _log.warning(
            "[GaugeTransformerLM] active_inference_distill_weight > 0 -- "
            "bootstrap self-distillation active in the E-step "
            "(see docs/bootstrap_self_distillation.md)."
        )

        # Pragmatic + distillation = mutual reinforcement of uniform collapse.
        # This is a stability issue, not a tuning question.  The pragmatic term
        # sharpens p_pred(mu_i) toward its local argmax; distillation
        # propagates that argmax across positions via the attention-aggregated
        # transported readout; pragmatic then sharpens the propagated peak
        # further at the neighbour; the cycle reinforces a collapsed state
        # that the epistemic MI term was originally introduced to prevent but
        # which it cannot address when the target is frozen by stop-gradient.
        _prag_on = any(
            getattr(_b.ffn, '_ai_enabled', False) and
            getattr(_b.ffn, '_ai_pragmatic_weight', 0.0) > 0.0
            for _b in transformer_stack.blocks
        )
        if _prag_on:
            _log.warning(
                "[GaugeTransformerLM] active_inference_pragmatic_weight > 0 "
                "simultaneously with active_inference_distill_weight > 0 is a "
                "STABILITY HAZARD, not a tuning question.  The two terms "
                "mutually reinforce uniform collapse: pragmatic sharpens toward "
                "the local argmax, distillation propagates the argmax across "
                "positions via transported neighbour readouts, and the cycle "
                "amplifies a collapsed prediction that the epistemic term "
                "cannot break (its target is frozen by stop-gradient).  "
                "Recommended: set active_inference_pragmatic_weight=0.0 when "
                "distillation is enabled, and rely on the distillation term's "
                "own data-dependent fixed-point structure for the E-step "
                "augmentation.  See docs/bootstrap_self_distillation.md "
                "Section 12."
            )

        # Pure-VFE / supervised-signal guard.  The distillation term has a
        # uniform-collapse attractor that is held off only by the M-step CE
        # signal against actual token targets.  If there is no such signal,
        # the term drives the model into the collapsed state rather than
        # toward a meaningful representation.  We cannot detect the absence
        # of a supervised signal at model construction time (it depends on
        # how forward() is called), so we log an informational reminder
        # that the user must ensure the term is only active in the joint
        # E-step + M-step training regime.
        _log.warning(
            "[GaugeTransformerLM] Reminder: bootstrap self-distillation has a "
            "uniform-collapse attractor that is only counteracted by the "
            "M-step cross-entropy loss against actual token targets.  Do NOT "
            "enable this term in pure-VFE configurations, pretraining warmup "
            "without a supervised loss, or any E-step-only analysis path.  "
            "See docs/bootstrap_self_distillation.md Section 5."
        )
