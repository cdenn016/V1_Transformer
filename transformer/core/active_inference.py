"""
Active Inference / Expected Free Energy Extension
==================================================

Experimental module that augments the VFE E-step with active-inference terms
from Karl Friston's Active Inference framework.  The module is isolated here
so that the core VFE file (variational_ffn.py) stays focused on the principal
gauge-theoretic free energy computation.

This file contains:

1. ``_compute_active_inference_gradient`` — EFE pragmatic + epistemic terms.
2. ``compute_ai_gradients`` — Dispatch wrapper that resolves refs from the FFN
   instance and calls the EFE helper.
3. ``configure_ffn_active_inference`` — Called from ``blocks.py`` to set the
   AI instance attributes on a freshly-constructed FFN.
4. ``wire_readout_references`` — Called from ``model.py`` after the full module
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

Both terms are gated by their respective weights.  When all weights are
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
                allow_unused=True,
            )
        grad_mu_accum = grad_prag_mu.detach() if grad_prag_mu is not None else torch.zeros_like(mu_f32)
        grad_sigma_accum = grad_prag_sigma.detach() if grad_prag_sigma is not None else torch.zeros_like(sigma_f32)
        del logits, log_probs, probs, entropy, pragmatic_term
        del mu_var, sigma_var, grad_prag_mu, grad_prag_sigma

    # ----- Epistemic: -MI(v; mu | q_i) via Welford-style 2-pass backward -----
    # Canonical BALD MI: the MC sample z_s = mu + sqrt(sigma) * eps already
    # absorbs sigma, so the readout must treat z_s as a point estimate.
    # We pass sigma=0 to _readout in BOTH passes so p_bar (pass-1) and p_s
    # (pass-2) use the same point-conditional decoder — otherwise the KL
    # trace term in decode() counts sigma twice (once via reparameterization,
    # once via the direct Gaussian-decode path) and inflates predictive
    # entropy. See transformer/vfe/efe.py:103-110 for the reference pattern.
    #
    # Consequence for full-covariance: path-2 reparameterization uses
    # pre-scaled noise (no Cholesky differentiation), so with sigma=0 in the
    # readout, the epistemic term contributes zero gradient to Sigma in the
    # full-cov case.  This matches efe.py semantics (diagonal-only BALD) and
    # is an accepted limitation documented here rather than a silent bias.
    _full_cov_epi = not is_diagonal
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
        # or pre-scaled noise for full-cov (reparameterization-only path).
        # Cache zero-sigma tensor used for the point-conditional readout in
        # both passes (see BALD MI discussion above).
        zero_sigma_f32 = torch.zeros_like(sigma_f32)
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
                    # Store pre-scaled noise (no sigma path for full-cov)
                    noise = _sample_scaled_noise(mu_f32)
                    scaled_noise_list.append(noise)
                mu_s = mu_f32 + noise
                logits_s = _readout(mu_s, zero_sigma_f32)
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
        # Sigma is tracked ONLY through reparameterization
        # mu_s = mu + sqrt(sigma_var) * raw_z (diagonal case).  The readout
        # receives sigma=0 so p_s matches p̄'s point-conditional decoder.
        # Full-cov: no Sigma path (pre-scaled noise is detached).
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
                    # Full-cov: pre-scaled noise; no Sigma gradient path
                    mu_s_grad = mu_var_s + scaled_noise_list[_s]
                # sigma=0 so z_s is treated as a point estimate (canonical
                # BALD).  Avoids double-counting via decode's KL trace term.
                logits_s_grad = _readout(mu_s_grad, torch.zeros_like(sigma_var_s))
                log_probs_s_grad = F.log_softmax(logits_s_grad, dim=-1)
                probs_s_grad = log_probs_s_grad.exp()
                # KL(p_s || p̄) = Σ_v p_s · (log p_s - log p̄)
                kl_s = (probs_s_grad
                        * (log_probs_s_grad - log_probs_avg_const)).sum(dim=-1)
                F_s = epi_per_sample_weight * kl_s.mean()
                grad_s_mu, grad_s_sigma = torch.autograd.grad(
                    F_s, [mu_var_s, sigma_var_s],
                    create_graph=False, retain_graph=False,
                    allow_unused=True,
                )
            _gs_mu = grad_s_mu.detach() if grad_s_mu is not None else torch.zeros_like(mu_f32)
            _gs_sigma = grad_s_sigma.detach() if grad_s_sigma is not None else torch.zeros_like(sigma_f32)
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
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute EFE gradients (pragmatic + epistemic).

    Reads ``ffn._ai_*`` attributes (set by ``configure_ffn_active_inference``)
    and resolves ``_prior_bank_ref`` / ``_ai_w_out_ref`` from ``ffn.__dict__``
    using the same list-unwrapping pattern as the original inlined code.

    Returns:
        (grad_efe_mu, grad_efe_sigma) — either may be None if the term is
        disabled, no readout is wired, or we are in inference_mode.
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
            except (ImportError, AttributeError):
                # Debug dict module not present or _VFE_GRAD_DEBUG not exposed:
                # this hook is best-effort instrumentation, so silent skip is OK.
                pass

    return grad_efe_mu, grad_efe_sigma


# =============================================================================
# Setup helpers — called from blocks.py and model.py
# =============================================================================

def configure_ffn_active_inference(ffn, cfg) -> None:
    """Read active-inference fields from BlockConfig and set as instance attributes on ffn.

    Sets 6 EFE attributes on the FFN instance.  Also initializes ``_prior_bank_ref = None`` in
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

    Raises:
        ValueError: If active_inference is combined with closed_form_e_step or use_deq.

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

