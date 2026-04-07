"""
Variational Feed-Forward Networks for Gauge Transformer
========================================================

VFE E-step belief evolution: iteratively updates beliefs (mu, Sigma) and
optionally gauge frames (phi) by minimizing variational free energy with
dynamic attention weights beta recomputed at each iteration.

Supports SO(3), SO(N), and GL(K) gauge groups. Generator count determines
the group: phi_dim=3 for SO(3), N(N-1)/2 for SO(N), K**2 for GL(K).

Key features:
- Dynamic beta: attention weights co-evolve with beliefs each VFE iteration
- Sigma softmax coupling: includes dBeta/dSigma term in VFE gradients
- Block-diagonal KL decomposition via irrep_dims for memory efficiency
- Fused attention+gradient paths for diagonal covariance mode
- Multi-head VFE: per-head beta_h through VFE iterations (multihead_vfe)
- Isotropic covariance: force Sigma = sigma^2 I (isotropic_covariance)
- DEQ mode: implicit differentiation for E-step fixed point (use_deq)
- Amortized inference: gradient flow through priors for learned E-step init
- Learnable alpha: Bayesian precision via Gamma-Normal conjugacy
- PriorBank: token-dependent priors via token_ids (prior_bank / use_prior_bank)
- exact_diagonal_transport: lifts diagonal sigma to full for exact transport

Mathematical Foundation:
-----------------------
Free Energy (E-STEP):
    F = alpha * Sum_i KL(q_i||p_i)                         # Prior consistency
      + lambda_belief * Sum_{i,j} beta_ij * KL(q_i||Omega_{ij}q_j)  # Belief alignment
      + lambda_softmax * Sum_{i,j} gamma_ij * KL(p_i||Omega_{ij}p_j)  # Prior alignment
      + CE(W_out * mu, targets)                             # Discrete observations

E-step: Minimize F w.r.t. mu, Sigma (with W_out frozen)
M-step: Minimize F w.r.t. W_out, embeddings (with mu frozen)

Gradient computation:
    dF/dtheta for theta = {mu_q, Sigma_q, mu_p, Sigma_p, phi}

With natural gradient projection:
    Delta_theta = -eta * F_inv(theta) * grad_F(theta)

Where F(theta) is the Fisher-Rao metric.
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

from transformer.core.gauge_utils import (
    stable_matrix_exp_pair,
    newton_schulz_orthogonalize,
    fused_block_matrix_exp_pairs,
    fused_block_diagonal_kl_diag,
    fused_block_diagonal_kl_full,
)

import transformer.core.vfe_utils as _vfe_utils_mod

from transformer.core.vfe_utils import (
    SIGMA_EPS,
    TRANSPORT_JITTER,
    KL_CEIL_BASE,
    KL_CEIL_SCALE,
    GRAD_CLIP_THRESHOLD,
    KAPPA_CLAMP_RANGE,
    _VFE_GRAD_DEBUG,
    _grad_norm,
    _per_pos_stats,
    _aggregate_multihead_vfe_debug,
    squeeze_trailing_singletons,
    _safe_spd_inv,
    _safe_eigh,
    retract_spd_torch,
    retract_spd_diagonal_torch,
    _retract_phi,
)


# =============================================================================
# Implicit EM Gradient (IFT-based M-step)
# =============================================================================
# In proper EM, the M-step gradient is dF/dθ = ∂F/∂θ|_{q=q*}. With finite
# E-step iterations, q* ≠ q_converged, so ∂F/∂q ≠ 0. The implicit function
# theorem gives the exact gradient without requiring convergence:
#
#   dq*/dθ = -(∂²F/∂q²)⁻¹ · ∂²F/(∂q∂θ)
#
# For diagonal Gaussians, this yields a per-dimension scale factor:
#   s_k = (α/σ²_p,k) / (α/σ²_p,k + Σ_j β_ij/σ²_j,k)  ∈ [0, 1]
#
# This interpolates between straight-through (s=1) and pure EM (s=0).

class ImplicitEMGradient(torch.autograd.Function):
    r"""Apply implicit differentiation scaling to M-step gradient for mu.

    Forward: returns mu_final unchanged (identity).
    Backward: scales gradient flowing to mu_embed by the implicit
    differentiation factor s_k = (α/σ²_p) / A_k, where A_k is the
    effective precision at the E-step fixed point.

    This replaces straight-through (s=1) with the information-geometrically
    correct gradient from the IFT, while still allowing CE gradients to
    reach embeddings (unlike pure EM where s=0).
    """

    @staticmethod
    def forward(
        ctx,
        mu_final: torch.Tensor,       # (B, N, K) — evolved beliefs from E-step
        mu_embed: torch.Tensor,        # (B, N, K) — embedding means (need grad)
        implicit_scale: torch.Tensor,  # (B, N, K) — IFT scale factors ∈ [0, 1]
    ) -> torch.Tensor:
        ctx.save_for_backward(implicit_scale)
        return mu_final  # forward is identity

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        implicit_scale, = ctx.saved_tensors
        # grad to mu_final: unchanged (flows to W_out via logits)
        # grad to mu_embed: scaled by IFT factor
        # grad to implicit_scale: None (detached)
        return grad_output, implicit_scale * grad_output, None


class ImplicitEMGradientSigma(torch.autograd.Function):
    r"""Apply implicit differentiation scaling to M-step gradient for sigma.

    Analogous to ImplicitEMGradient but for covariance parameters.
    The sigma fixed-point equation (supplementary Eq. B.6) gives:
        Σ_i^{-1} = (1/2)[Σ_p^{-1} + Σ_j β_ij (Ω_ij Σ_j Ω_ij^T)^{-1}]

    The implicit gradient scale is:
        s_k = (Σ_p^{-2}) / (Σ_p^{-2} + Σ_j β_ij Σ_j^{-2})

    Works for both diagonal (B, N, K) and full covariance (B, N, K, K).
    """

    @staticmethod
    def forward(
        ctx,
        sigma_final: torch.Tensor,       # Evolved covariance from E-step
        sigma_embed: torch.Tensor,        # Embedding covariance (need grad)
        implicit_scale: torch.Tensor,     # IFT scale factors
    ) -> torch.Tensor:
        ctx.save_for_backward(implicit_scale)
        return sigma_final

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        implicit_scale, = ctx.saved_tensors
        # Scale is (B, N, K) from diagonal approx; grad_output may be
        # (B, N, K, K) for full covariance.  Unsqueeze to broadcast.
        scale = implicit_scale
        if grad_output.dim() > scale.dim():
            scale = scale.unsqueeze(-1)
        return grad_output, scale * grad_output, None


def compute_implicit_em_scales(
    alpha_i: torch.Tensor,     # (B, N, K) or scalar — prior coupling strength
    sigma_p: torch.Tensor,     # (B, N, K) diagonal or (B, N, K, K) full — prior sigma
    beta: torch.Tensor,        # (B, H, N, N) or (B, N, N) — attention weights
    sigma_q: torch.Tensor,     # (B, N, K) diagonal or (B, N, K, K) full — evolved sigma
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute implicit differentiation scale factors for principled EM M-step.

    From the E-step fixed-point equation for μ:
        A_k = α/σ²_{p,k} + Σ_j β_{ij}/σ²_{j,k}
        s_k^{(μ)} = (α/σ²_{p,k}) / A_k

    For σ (from covariance fixed-point):
        s_k^{(σ)} = (α/σ⁴_{p,k}) / (α/σ⁴_{p,k} + Σ_j β_{ij}/σ⁴_{j,k})

    The attention-weighted precision Σ_j β_{ij}/σ²_{j,k} uses per-position j
    covariances weighted by the actual attention weights, rather than
    approximating all transported covariances as equal to σ²_{q,i,k}.

    Args:
        alpha_i: Prior coupling. Scalar or (B, N, K) from adaptive α.
        sigma_p: Prior covariance (diagonal or full).
        beta: Attention weights from final E-step iteration.
        sigma_q: Evolved covariance from E-step.
        eps: Numerical floor.

    Returns:
        mu_scale: Same shape as sigma_p diagonal — per-dim scale for μ gradient
        sigma_scale: Same shape as sigma_p diagonal — per-dim scale for σ gradient
    """
    is_diagonal = sigma_p.dim() == 3

    if is_diagonal:
        sigma_p_safe = sigma_p.clamp(min=eps)  # (B, N, K)
        sigma_q_safe = sigma_q.clamp(min=eps)  # (B, N, K)
    else:
        # Full covariance: extract diagonal for scale computation
        sigma_p_safe = torch.diagonal(sigma_p, dim1=-2, dim2=-1).clamp(min=eps)  # (B, N, K)
        sigma_q_safe = torch.diagonal(sigma_q, dim1=-2, dim2=-1).clamp(min=eps)  # (B, N, K)

    # Broadcast alpha_i to (B, N, K) if scalar
    if not isinstance(alpha_i, torch.Tensor):
        alpha_val = alpha_i
        alpha_i = torch.full_like(sigma_p_safe, alpha_val)
    elif alpha_i.dim() == 0:
        alpha_i = alpha_i.expand_as(sigma_p_safe)
    elif alpha_i.dim() < sigma_p_safe.dim():
        # (B, N) → (B, N, K)
        alpha_i = alpha_i.unsqueeze(-1).expand_as(sigma_p_safe)

    # Reduce beta to (B, N, N) if multihead
    if beta.dim() == 4:  # (B, H, N, N)
        beta_2d = beta.mean(dim=1)  # Average over heads → (B, N, N)
    else:
        beta_2d = beta  # Already (B, N, N)

    # === Mu scale ===
    # Prior precision contribution: α_k / σ²_{p,k}
    prior_prec_mu = alpha_i / sigma_p_safe  # (B, N, K)

    # Attention-weighted precision: Σ_j β_{ij} / σ²_{q,j,k}
    # Uses per-position j covariances rather than approximating with σ²_{q,i,k}.
    # At the fixed point, σ²_{j,transported} ≈ σ²_{q,j} (transport ≈ identity near
    # convergence). This captures variance heterogeneity across positions.
    inv_sigma_q = 1.0 / sigma_q_safe  # (B, N, K) — per-position precision
    attn_prec_mu = torch.einsum('bij,bjk->bik', beta_2d, inv_sigma_q)  # (B, N, K)

    effective_prec_mu = prior_prec_mu + attn_prec_mu
    mu_scale = (prior_prec_mu / effective_prec_mu.clamp(min=eps)).detach()

    # === Sigma scale ===
    # From the covariance fixed-point Hessian: uses precision squared
    prior_prec_sigma = alpha_i / (sigma_p_safe ** 2)  # (B, N, K)
    inv_sigma_q_sq = 1.0 / (sigma_q_safe ** 2)  # (B, N, K)
    attn_prec_sigma = torch.einsum('bij,bjk->bik', beta_2d, inv_sigma_q_sq)  # (B, N, K)

    effective_prec_sigma = prior_prec_sigma + attn_prec_sigma
    sigma_scale = (prior_prec_sigma / effective_prec_sigma.clamp(min=eps)).detach()

    return mu_scale, sigma_scale


# Import attention computation for dynamic β
from transformer.core.attention import compute_attention_weights
from transformer.core.transport_ops import compute_transport_operators

# Numerical event monitor (shared with attention.py)
from math_utils.numerical_monitor import record as _nr



# =============================================================================
# VFE Gradient Computation (moved to vfe_gradients.py)
# =============================================================================
# These functions are defined in transformer.core.vfe_gradients and imported
# here for backward compatibility.  Do not add gradient logic here.
from transformer.core.vfe_gradients import (
    _compute_vfe_gradients_block_diagonal,
    _compute_vfe_gradients_block_diagonal_diag,
    _fused_attention_and_vfe_gradients_block_diag,
    compute_vfe_gradients_gpu,
    compute_natural_gradient_gpu,
)


# retract_spd_torch and retract_spd_diagonal_torch are imported from vfe_utils above.


# =============================================================================
# DEQ Fixed-Point Implicit Differentiation
# =============================================================================

class DEQFixedPoint(torch.autograd.Function):
    """Implicit differentiation through E-step fixed point via Neumann series.

    Forward: identity (fixed point already computed by the E-step loop).
    Backward: corrects gradients using (I - J)^{-1} ≈ I + J + J² + ... (K terms)
    where J is the Jacobian of one E-step evaluated at the fixed point.
    """

    @staticmethod
    def forward(ctx, mu_star, sigma_star, e_step_fn, n_steps, neumann_terms):
        ctx.save_for_backward(mu_star.detach(), sigma_star.detach())
        ctx.e_step_fn = e_step_fn
        ctx.neumann_terms = neumann_terms
        return mu_star, sigma_star

    @staticmethod
    def backward(ctx, grad_mu, grad_sigma):
        mu_star, sigma_star = ctx.saved_tensors
        e_step_fn = ctx.e_step_fn
        K = ctx.neumann_terms

        # Neumann series: (I - J^T)^{-1} v ≈ v + J^T v + (J^T)^2 v + ...
        v_mu = grad_mu.clone()
        v_sigma = grad_sigma.clone()
        total_mu = grad_mu.clone()
        total_sigma = grad_sigma.clone()

        for _ in range(K):
            # One vjp through the E-step at the fixed point
            mu_in = mu_star.detach().requires_grad_(True)
            sigma_in = sigma_star.detach().requires_grad_(True)
            with torch.enable_grad():
                mu_out, sigma_out = e_step_fn(mu_in, sigma_in)
            jt_v = torch.autograd.grad(
                outputs=[mu_out, sigma_out],
                inputs=[mu_in, sigma_in],
                grad_outputs=[v_mu, v_sigma],
                retain_graph=False,
                allow_unused=True,
            )
            v_mu = jt_v[0] if jt_v[0] is not None else torch.zeros_like(grad_mu)
            v_sigma = jt_v[1] if jt_v[1] is not None else torch.zeros_like(grad_sigma)
            total_mu = total_mu + v_mu
            total_sigma = total_sigma + v_sigma

        return total_mu, total_sigma, None, None, None


class DEQFixedPointFull(torch.autograd.Function):
    r"""Implicit differentiation through joint (μ, Σ, φ) fixed point.

    Extends DEQFixedPoint to include gauge frame φ in the fixed-point
    variables.  At the E-step fixed point, all VFE gradients vanish:
        ∂F/∂μ = 0,  ∂F/∂Σ = 0,  ∂F/∂φ = 0.

    The IFT gives the exact M-step gradient:
        ∂z*/∂θ = −(∂²F/∂z²)⁻¹ · ∂²F/∂z∂θ
    where z = (μ, Σ, φ) and θ are model parameters (embeddings, etc.).

    We approximate (I − J)⁻¹ via Neumann series:
        (I − J^T)⁻¹ v ≈ v + J^T v + (J^T)² v + ⋯  (K terms)
    where J is the Jacobian of one full E-step (μ, Σ, φ) → (μ', Σ', φ')
    evaluated at the fixed point.

    This corrects the straight-through bias in the M-step φ gradient:
    instead of ∂φ*/∂φ_init ≈ I, we get the IFT-corrected Jacobian that
    accounts for how the E-step trajectory depends on initial conditions.
    """

    @staticmethod
    def forward(ctx, mu_star, sigma_star, phi_star, e_step_fn, n_steps, neumann_terms):
        ctx.save_for_backward(
            mu_star.detach(), sigma_star.detach(), phi_star.detach(),
        )
        ctx.e_step_fn = e_step_fn
        ctx.neumann_terms = neumann_terms
        return mu_star, sigma_star, phi_star

    @staticmethod
    def backward(ctx, grad_mu, grad_sigma, grad_phi):
        mu_star, sigma_star, phi_star = ctx.saved_tensors
        e_step_fn = ctx.e_step_fn
        K = ctx.neumann_terms

        # Neumann series: (I - J^T)^{-1} v ≈ v + J^T v + (J^T)^2 v + ...
        v_mu = grad_mu.clone()
        v_sigma = grad_sigma.clone()
        v_phi = grad_phi.clone()
        total_mu = grad_mu.clone()
        total_sigma = grad_sigma.clone()
        total_phi = grad_phi.clone()

        for _ in range(K):
            mu_in = mu_star.detach().requires_grad_(True)
            sigma_in = sigma_star.detach().requires_grad_(True)
            phi_in = phi_star.detach().requires_grad_(True)
            with torch.enable_grad():
                mu_out, sigma_out, phi_out = e_step_fn(mu_in, sigma_in, phi_in)
            jt_v = torch.autograd.grad(
                outputs=[mu_out, sigma_out, phi_out],
                inputs=[mu_in, sigma_in, phi_in],
                grad_outputs=[v_mu, v_sigma, v_phi],
                retain_graph=False,
                allow_unused=True,
            )
            v_mu = jt_v[0] if jt_v[0] is not None else torch.zeros_like(grad_mu)
            v_sigma = jt_v[1] if jt_v[1] is not None else torch.zeros_like(grad_sigma)
            v_phi = jt_v[2] if jt_v[2] is not None else torch.zeros_like(grad_phi)
            total_mu = total_mu + v_mu
            total_sigma = total_sigma + v_sigma
            total_phi = total_phi + v_phi

        return total_mu, total_sigma, total_phi, None, None, None


# =============================================================================
# Active inference / Expected Free Energy gradient helper
# =============================================================================
#
# Augments the VFE E-step with the two terms that active inference prescribes
# when the agent does not have direct access to the observation:
#
#   F_AI = λ_prag · H[p_pred(v|μ_i)]                            (pragmatic)
#        − λ_epi  · MI(v; μ | q_i)                               (epistemic)
#
# Pragmatic term:
#   p_pred(v|μ_i) is the PriorBank readout (KL-based softmax over vocab).
#   Minimizing its entropy makes the belief commit to a confident prediction
#   without target leak.  This is the "self-observation" component: the agent
#   conditions on what its own current belief most strongly predicts.
#
# Epistemic term (BALD-style mutual information):
#   MI(v; μ | q_i) = H[E_{μ ~ q_i} p(v|μ)] − E_{μ ~ q_i}[H[p(v|μ)]]
#   Estimated by Monte Carlo: sample S values μ_s ~ N(μ_i, Σ_i), evaluate the
#   readout at each, then compute the predictive-distribution disagreement.
#   High MI ⇔ the predictive distribution depends meaningfully on which
#   element of q_i you sample ⇔ belief uncertainty carries decision-relevant
#   information ⇔ updating the belief is epistemically valuable.
#   The negative sign in F_AI means we *maximize* MI in the E-step, which
#   counter-balances the pragmatic term's tendency toward self-reinforcement.
#
# Both terms are differentiable in μ_i.  We compute the gradient via
# torch.autograd.grad on a freshly-detached μ leaf, so the autograd graph is
# local to this function and does not perturb the rest of the E-step graph.
#
# CRITICAL: torch.inference_mode() (used by model.generate) marks tensors as
# inference tensors that cannot be promoted to require_grad even inside
# enable_grad().  Detect and skip the EFE term in inference-mode contexts —
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
) -> Optional[torch.Tensor]:
    r"""Compute ∂F_AI/∂μ via autograd through the readout.

    Uses PriorBank.decode when prior_bank is provided (principled KL-based
    readout).  Falls back to a linear W_out projection followed by softmax
    when prior_bank is None.  The linear fallback is less principled (it
    ignores σ and uses Euclidean-not-KL distance) but keeps the EFE path
    functional for configurations that don't use PriorBank.

    Args:
        mu_current: (B, N, K) current belief mean (will be detached internally).
        sigma_current: (B, N, K) diagonal or (B, N, K, K) full covariance.
        prior_bank: PriorBank instance providing decode(mu, sigma, tau) → logits.
        pragmatic_weight: λ_prag.  0.0 disables the pragmatic term.
        epistemic_weight: λ_epi.  0.0 disables the epistemic term.
        epistemic_samples: S, number of MC samples for BALD MI estimate.
        decode_tau: temperature for PriorBank.decode (1.0 = standard).
        w_out: optional (V, K) linear projection tensor used as fallback
               readout when prior_bank is None.

    Returns:
        (B, N, K) gradient of F_AI with respect to μ_current, or None if
        both weights are zero, both readout paths are unavailable, or we
        are inside an inference-mode context.
    """
    if prior_bank is None and w_out is None:
        return None
    if pragmatic_weight <= 0.0 and epistemic_weight <= 0.0:
        return None
    if torch.is_inference_mode_enabled():
        return None

    is_diagonal = (sigma_current.dim() == mu_current.dim())
    use_prior_bank = (prior_bank is not None)

    # Guard against decode_tau=0 for BOTH readout paths.  PriorBank.decode
    # internally divides by tau; the W_out fallback does too.  Without the
    # guard, tau=0 produces NaN logits and corrupts μ via the EFE gradient.
    _tau_safe = max(float(decode_tau), 1e-8)

    def _readout(mu_input: torch.Tensor, sigma_arg: torch.Tensor) -> torch.Tensor:
        """Return (B, N, V) logits from either PriorBank or W_out."""
        if use_prior_bank:
            return prior_bank.decode(mu_input, sigma_arg, tau=_tau_safe)
        else:
            return torch.matmul(mu_input, w_out.T) / _tau_safe

    with torch.enable_grad():
        mu_var = mu_current.detach().to(torch.float32).requires_grad_(True)
        sigma_arg = sigma_current.detach().to(torch.float32)

        total_efe = mu_var.new_zeros(())

        # ---- Pragmatic: minimize H[p_pred(v|μ)] ----
        if pragmatic_weight > 0.0:
            logits = _readout(mu_var, sigma_arg)  # (B, N, V)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)  # (B, N)
            pragmatic_term = pragmatic_weight * entropy.mean()
            total_efe = total_efe + pragmatic_term

        # ---- Epistemic: -MI(v; μ | q_i) via Monte Carlo BALD estimate ----
        if epistemic_weight > 0.0 and epistemic_samples > 0:
            if is_diagonal:
                sigma_diag = sigma_arg
                std = sigma_diag.clamp(min=1e-6).sqrt()
                _sample_noise = lambda: torch.randn_like(mu_var) * std
            else:
                # Full covariance: use Cholesky for correct correlated sampling.
                # Falls back to diagonal std on numerical failure (non-SPD Σ).
                sigma_spd = 0.5 * (sigma_arg + sigma_arg.transpose(-1, -2))
                # Add jitter for numerical stability
                K_dim = sigma_spd.shape[-1]
                eye_k = torch.eye(K_dim, device=sigma_spd.device, dtype=sigma_spd.dtype)
                sigma_spd = sigma_spd + 1e-6 * eye_k
                try:
                    L_chol = torch.linalg.cholesky(sigma_spd)  # (B, N, K, K)
                    def _sample_noise():
                        z = torch.randn_like(mu_var)  # (B, N, K)
                        # (B, N, K, K) @ (B, N, K, 1) → (B, N, K, 1) → (B, N, K)
                        return torch.einsum('bnkj,bnj->bnk', L_chol, z)
                except Exception:
                    sigma_diag = torch.diagonal(sigma_spd, dim1=-2, dim2=-1)
                    std = sigma_diag.clamp(min=1e-6).sqrt()
                    _sample_noise = lambda: torch.randn_like(mu_var) * std

            probs_samples = []
            for _s in range(epistemic_samples):
                mu_s = mu_var + _sample_noise()
                logits_s = _readout(mu_s, sigma_arg)
                probs_s = F.softmax(logits_s, dim=-1)
                probs_samples.append(probs_s)
            probs_stack = torch.stack(probs_samples, dim=0)  # (S, B, N, V)

            # H[E_q p(v|μ)] : entropy of the average predictive distribution
            probs_avg = probs_stack.mean(dim=0)  # (B, N, V)
            log_probs_avg = (probs_avg.clamp(min=1e-12)).log()
            H_avg = -(probs_avg * log_probs_avg).sum(dim=-1)  # (B, N)

            # E_q[H[p(v|μ)]] : average entropy across samples
            log_probs_stack = (probs_stack.clamp(min=1e-12)).log()
            H_each = -(probs_stack * log_probs_stack).sum(dim=-1)  # (S, B, N)
            avg_H = H_each.mean(dim=0)  # (B, N)

            mi = H_avg - avg_H  # (B, N) ≥ 0 in expectation
            # Maximize MI ⇔ subtract λ_epi · MI from F.
            epistemic_term = -epistemic_weight * mi.mean()
            total_efe = total_efe + epistemic_term

        grad_efe = torch.autograd.grad(total_efe, mu_var, create_graph=False, retain_graph=False)[0]

    # Cast back to original dtype, detach so it does not entangle with the
    # surrounding analytic-gradient graph.
    return grad_efe.detach().to(mu_current.dtype)


# =============================================================================
# Dynamic-β VFE: Full Active Inference with Evolving Attention (RECOMMENDED!)
# =============================================================================

class VariationalFFNDynamic(nn.Module):
    """
    Dynamic-beta Variational FFN: VFE E-step with beta recomputed each iteration.

    At each integration step, beliefs (mu, Sigma) and attention weights (beta)
    co-evolve:

        1. Compute beta from current beliefs: beta_ij = softmax(-KL(q_i||Omega_ij[q_j])/kappa)
        2. Compute full VFE gradient: dF/dtheta (includes dBeta/dtheta nonlinearity)
        3. Update beliefs via natural gradient descent
        4. (Optional) Update gauge frames phi via dF/dphi
        5. Repeat for n_iterations steps

    The dBeta/dmu softmax coupling is the principled nonlinearity (replaces GELU):
        dBeta_ij/dmu_i = -beta_ij * [dKL_ij/dmu_i - Sum_k beta_ik * dKL_ik/dmu_i] / kappa

    Supports SO(3), SO(N), and GL(K) gauge groups via generators (n_gen, K, K)
    and gauge frames phi (B, N, phi_dim). Additional capabilities:
        - multihead_vfe: per-head beta_h through VFE iterations (requires irrep_dims)
        - prior_bank / use_prior_bank: token-dependent priors via PriorBank and token_ids
        - learnable_alpha: Bayesian precision via Gamma-Normal conjugacy
        - isotropic_covariance: force Sigma = sigma^2 I
        - DEQ mode (use_deq): implicit differentiation for E-step fixed point
        - amortized_inference: gradient flow through priors for learned E-step init
        - exact_diagonal_transport: lift diagonal sigma to full for exact transport,
          then extract diagonal from result; disables fused diagonal paths

    Complexity: O(n_steps * N^2 * K)
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,  # (n_gen, K, K) Lie algebra generators
        alpha: float = 0.01,       # Self-coupling weight (KL(q||p)) for E-step VFE descent.
                                   # NOTE: Decoupled from the M-step loss alpha (config['alpha']).
                                   # With amortized inference, dCE/dθ flows through q*(θ) via the
                                   # VFE computation graph, so dq*/dθ already encodes the
                                   # self-coupling effect. Adding explicit KL(q||p) in the M-step
                                   # loss double-counts the coupling. Correct separation: E-step
                                   # handles belief regularization (α_E > 0), M-step handles
                                   # prediction quality (CE only, α_M = 0).
        lambda_belief: float =  1.0,  # Belief alignment weight (direct: β·∇KL)
        lambda_softmax: float = 1.0,  # Softmax coupling weight (GELU-like ∂β/∂θ · KL)
        kappa: float =          1.0,        # Attention temperature
        n_iterations: int =     1,    # VFE descent steps (more steps = deeper equilibration)
        
        mu_lr: float =          0.1,           # E-step μ step size (used when learnable_lr=False)
        sigma_lr: float =       0.001,      # E-step σ trust region scale (used when learnable_lr=False)
        phi_lr: float =         0.05,      # Learning rate for phi updates
        
        learnable_lr: bool =              True, # Learn step size?
        update_sigma: bool =              True, # Update covariances?
        diagonal_covariance: bool =       False,  # Use diagonal Σ for efficiency
        exact_diagonal_transport: bool =  False,  # Lift diagonal σ for exact transport
        compute_sigma_align_grad: bool =  True,  # Compute sigma gradient from alignment term
        
        # Phi (gauge frame) evolution via VFE gradients
        update_phi: bool = True,  # If True, update phi via ∂F/∂φ (after E-step loop)
        update_phi_per_iteration: bool =  True,  # If True, update phi during EACH E-step iteration
        
        phi_max_norm: Optional[float] =   None,  # Max phi norm; None = auto (π for SO(N), 5.0 for GL(K))
        prior_bank: Optional[nn.Module] = None,  # Token-dependent PriorBank (if provided)
        use_prior_bank: bool =            False,  # If True, use PriorBank (token-dependent) instead of position-dependent priors
        
        # Memory-efficient options (NEW!)
        irrep_dims: Optional[List[int]] = None,  # Block dimensions for principled KL decomposition
        
        # Self-attention masking (prevents attention collapse)
        mask_self_attention: bool =       False,  # If True, mask out diagonal (no self-attention)
        
        # Bayesian precision (learned prior self-coupling)
        learnable_alpha: bool =           False,  # If True, use Gamma-Normal conjugate precision
        # Multi-head VFE: maintain per-head β through iterations
        multihead_vfe: bool =             True,  # If True, compute separate β_h per irrep block
        # Phi gradient preconditioning mode
        phi_natural_gradient: str =          'killing',  # 'clip'|'cartan'|'killing'|'pullback'
        killing_center_reg: Optional[float] = None,  # Killing form center regularization (None=2K)
        
        # DEQ implicit differentiation
        use_deq: bool =          False,                # Use DEQ backward for E-step fixed point
        deq_neumann_terms: int = 5,           # Neumann series terms for DEQ backward
        
        # Gauge mode
        gauge_mode: str = 'learned',          # 'learned', 'trivial' (Ω = I), or 'constant'
        # Constant gauge: per-head learnable Ω from the attention module.
        # When gauge_mode='constant', these are used for transport in VFE iterations
        # instead of identity (which would be inconsistent with the attention module).
        constant_omega: Optional[nn.ParameterList] = None,
        
        # Isotropic covariance limit
        isotropic_covariance: bool = False,   # If True, force Σ = σ²I after each E-step update
        # Amortized inference: gradient flow through priors for learned E-step init
        amortized_inference: bool =  True,
        
        # Rotary Position Embeddings (RoPE) — must match attention sublayer setting
        use_rope: bool =             True,
        rope_base: float =           10000.0,
        
        gauge_param: str =           'phi',# 'phi' (Lie algebra) or 'omega' (direct GL(K))
        obs_sigma_gradient: bool =   True, # ∂E_q[CE]/∂σ via Hessian diagonal of expected CE
        obs_sigma_weight: float =    1.0,  # Weight for sigma observation gradient
        sigma_max: float =           5.0,  # Upper bound on σ (prevents nat_grad blowup from 2σ²·∇σ)
        
        e_step_sigma_floor: float =  0.1,  # Floor on σ_p inside E-step (caps 1/σ_p at 1/floor)
        detach_phi: bool =           False,# Detach phi from backprop in non-amortized mode
                                           # (enables fully backprop-free training with phi P-flow)
        deq_include_phi: bool =      False,# Include phi in DEQ fixed-point variables.
                                           # When True, the Neumann-series IFT correction applies
                                           # to the joint (mu, sigma, phi) fixed point, giving the
                                           # exact M-step phi gradient instead of straight-through.
                                           # Requires use_deq=True and evolve_phi=True.
        closed_form_e_step: bool =   False,# Use closed-form precision-weighted fixed point
                                           # instead of gradient descent. Diagonal path uses
                                           # the enhanced form that absorbs softmax coupling
                                           # (S, c terms); full-cov uses linear-only CF.
        implicit_em: bool =          False,# Principled M-step via implicit differentiation.
                                           # Detaches mu/sigma at E-step start (proper EM boundary)
                                           # then scales CE→embedding gradient by IFT factor
                                           # s_k = (α/σ²_p) / (α/σ²_p + Σβ/σ²_q) ∈ [0,1].
                                           # Replaces ad-hoc straight-through (s=1) and pure EM (s=0)
                                           # with info-geometrically correct value.
        learnable_head_kappa: bool =    False,# If True, learn per-head κ_h
        n_picard_steps: int =           0,  # Re-solve iterations (diagonal) or Picard steps (full-cov)
        picard_trust_region: float =    5.0,# Whitened trust region for Picard steps
        compile_vfe: bool = False,         # torch.compile the VFE iteration (Finding 25)
        gradient_checkpoint_vfe: bool = False,  # Activation checkpointing for VFE loop (Finding 26)
    ):
        """
        Initialize dynamic-beta VFE FFN.

        Args:
            embed_dim: K, dimension of belief vectors.
            generators: Lie algebra generators (n_gen, K, K). n_gen = 3 for SO(3),
                N(N-1)/2 for SO(N), K^2 for GL(K).
            alpha: Weight for KL(q||p) self-coupling (prior anchoring).
            lambda_belief: Weight for belief alignment term.
            kappa: Temperature for attention softmax (higher = softer).
            n_iterations: Number of VFE descent iterations per forward pass.
            learnable_lr: If True, step size eta is a learnable parameter.
                If False, uses fixed mu_lr and sigma_lr instead.
            mu_lr: E-step μ natural gradient step size (used when learnable_lr=False).
            sigma_lr: E-step σ trust region scale (used when learnable_lr=False).
            update_sigma: If True, also update covariance matrices Sigma.
            diagonal_covariance: Use diagonal Sigma for O(K) instead of O(K^2).
            compute_sigma_align_grad: If True, compute sigma gradient from alignment.
            update_phi: If True, update phi via dF/dphi after E-step loop.
            update_phi_per_iteration: If True, evolve phi during each E-step iteration.
            phi_lr: Learning rate for phi updates.
            phi_max_norm: Max norm for phi; None = auto-select in retraction
                (π for SO(N), 5.0 for GL(K)).
            prior_bank: Optional PriorBank module providing token-dependent priors via
                token_ids. When set with use_prior_bank=True, replaces position-dependent
                priors with token-dependent priors from the PriorBank.
            use_prior_bank: If True, use PriorBank for token-dependent priors.
                Requires prior_bank and token_ids in forward().
            irrep_dims: Block dimensions [d_1, d_2, ...] for memory-efficient
                block-diagonal KL decomposition.
            mask_self_attention: If True, mask diagonal to prevent attention collapse.
            learnable_alpha: If True, Bayesian precision via Gamma-Normal conjugacy:
                alpha_k = c0_k / (b0_k + kl_k). Per-dimension, gauge-invariant.
            multihead_vfe: If True, per-head beta_h through VFE iterations.
                Requires irrep_dims.
            phi_natural_gradient: Phi gradient preconditioning mode
                ('clip'|'cartan'|'killing'|'pullback').
            use_deq: If True, use DEQ implicit differentiation for E-step fixed point.
            deq_neumann_terms: Neumann series terms for DEQ backward pass.
            gauge_mode: 'learned', 'trivial' (Omega=I), or 'constant'.
            constant_omega: Per-head learnable Omega from the attention module for
                gauge_mode='constant'.
            isotropic_covariance: If True, force Sigma = sigma^2 I after each update.
            amortized_inference: If True, gradients flow through priors for learned
                E-step initialization.
            exact_diagonal_transport: When True with diagonal_covariance, lifts sigma
                to full covariance via diag_embed for exact gauge transport, then
                extracts the diagonal from the result. Disables fused diagonal paths.
            use_rope: Apply rotary position embeddings (must match attention sublayer).
            rope_base: RoPE base frequency.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)
        # Cache skew-symmetry flag: for SO(K), exp(-A) = exp(A)^T (saves one matrix_exp)
        self._generators_are_skew = bool(torch.allclose(
            generators + generators.transpose(-1, -2),
            torch.zeros_like(generators), atol=1e-5
        ))
        self.n_iterations = n_iterations
        self.gauge_param = gauge_param
        self.mask_self_attention = mask_self_attention
        self.update_sigma = update_sigma
        self.diagonal_covariance = diagonal_covariance
        self.exact_diagonal_transport = exact_diagonal_transport
        self.compute_sigma_align_grad = compute_sigma_align_grad
        self.gauge_mode = gauge_mode
        self.isotropic_covariance = isotropic_covariance
        self.amortized_inference = amortized_inference
        self.obs_sigma_gradient = obs_sigma_gradient
        self.obs_sigma_weight = obs_sigma_weight
        self.sigma_max = sigma_max
        self.e_step_sigma_floor = e_step_sigma_floor
        self.detach_phi = detach_phi
        self.closed_form_e_step = closed_form_e_step
        self.n_picard_steps = n_picard_steps
        self.picard_trust_region = picard_trust_region
        self.gradient_checkpoint_vfe = gradient_checkpoint_vfe
        self.implicit_em = implicit_em
        self._last_implicit_mu_scale = None   # (B, N, K) stored after E-step for model.py
        self._last_implicit_sigma_scale = None
        if implicit_em:
            logger.info(f"[VariationalFFNDynamic] Implicit EM enabled: IFT-based M-step gradient")
            logger.info(f"  → Detaches mu/sigma at E-step start, applies s_k = (α/σ²_p)/A_k scaling")
        # RoPE: DISABLED in VFE E-step iterations. The E-step is belief refinement
        # (analogous to V aggregation), not attention scoring (Q·K). Position
        # awareness enters via β_ij from the attention sublayer, which already
        # applies RoPE. Applying RoPE inside the E-step double-counts position,
        # breaks KL geometry (rotates μ but not Σ), and distorts the VFE fixed point.
        #
        # DESIGN NOTE: This means the attention sublayer's β (position-aware via
        # RoPE) and the E-step's internally-recomputed β (position-blind) are
        # INCONSISTENT. The attention sublayer uses RoPE-rotated KL for routing
        # and aggregation, while the E-step uses raw KL for belief refinement.
        # This is intentional: position information enters through the attention
        # sublayer's message aggregation, and the E-step refines beliefs based
        # on content-only alignment. With ffn_n_iterations=1, the E-step β is
        # what drives belief evolution; the attention sublayer β drives mu/sigma
        # aggregation (the "value" path).
        self._use_rope_vfe = True
        self._rope_base_vfe = rope_base
        # EXPERIMENTAL: when True, the rope KL also rotates Σ (R Σ R^T sandwich),
        # implementing the framework-consistent gauge-transport interpretation
        # of RoPE.  Default False uses the standard-transformer pattern (rotate
        # only μ).  Plumbed through from BlockConfig.rope_full_gauge.
        self._rope_full_gauge_vfe = False  # set externally by VariationalFFNDynamic.__init__
        # Constant gauge: store reference to attention module's per-head Ω parameters.
        # When gauge_mode='constant', these are used to build transport operators
        # in VFE iterations, ensuring consistency with the attention module.
        # Without this, the FFN would use Ω=I (identity), computing different
        # attention patterns than the attention module.
        self.constant_omega = constant_omega

        # Phi evolution via VFE gradients (principled approach)
        self.update_phi = update_phi
        self.update_phi_per_iteration = update_phi_per_iteration  # Dynamical gauge frames
        if update_phi_per_iteration and gauge_mode == 'constant':
            import warnings
            warnings.warn(
                "evolve_phi_e_step=True with gauge_mode='constant' is ineffective: "
                "phi is not used for transport in constant gauge mode (Ω is a direct parameter). "
                "Set evolve_phi_e_step=False to avoid wasted computation.",
                UserWarning,
            )
        if update_phi_per_iteration:
            logger.info(f"[VariationalFFNDynamic] φ will evolve DURING E-step iterations (dynamical gauge frames)")
        self.phi_lr = phi_lr
        self.phi_max_norm = phi_max_norm

        # Phi gradient preconditioning mode
        self.phi_natural_gradient = phi_natural_gradient
        # Register as buffers so they move to the correct device with the model
        self.register_buffer('_phi_preconditioner', None)
        self.register_buffer('_structure_constants', None)
        self.register_buffer('_gram', None)
        if phi_natural_gradient not in ('clip', 'cartan', 'killing', 'pullback'):
            raise ValueError(f"phi_natural_gradient must be 'clip'|'cartan'|'killing'|'pullback', got '{phi_natural_gradient}'")
        if phi_natural_gradient in ('cartan', 'killing', 'pullback'):
            from transformer.core.gauge_preconditioner import (
                build_cartan_projector, build_killing_form_preconditioner,
                build_structure_constants,
            )
            if phi_natural_gradient == 'cartan':
                self._phi_preconditioner = build_cartan_projector(generators)
                logger.info(f"[VariationalFFNDynamic] φ preconditioning: Cartan (sym_dampening=0.1)")
            elif phi_natural_gradient == 'killing':
                self._phi_preconditioner = build_killing_form_preconditioner(
                    generators, center_reg=killing_center_reg,
                )
                _cr = killing_center_reg if killing_center_reg is not None else 2.0 * generators.shape[-1]
                logger.info(f"[VariationalFFNDynamic] φ preconditioning: Killing form natural gradient (center_reg={_cr:.1f})")
            elif phi_natural_gradient == 'pullback':
                self._structure_constants = build_structure_constants(generators)
                # Frobenius inner product: <T_a, T_b> = tr(T_a^T T_b)
                self._gram = torch.einsum('aij,bij->ab', generators, generators)
                logger.info(f"[VariationalFFNDynamic] φ preconditioning: pullback natural gradient (exact)")

        # Memory-efficient options
        self.irrep_dims = irrep_dims

        # VFE hyperparameters
        self.alpha = alpha
        self.lambda_belief = lambda_belief
        self.lambda_softmax = lambda_softmax
        self.kappa = kappa
        self.learnable_head_kappa = learnable_head_kappa

        # =================================================================
        # Multi-head VFE: per-block β through iterations
        # =================================================================
        self.multihead_vfe = multihead_vfe and irrep_dims is not None
        if self.multihead_vfe:
            n_heads = len(irrep_dims)
            if learnable_head_kappa:
                logger.info(f"[VariationalFFNDynamic] Multi-head VFE: {n_heads} heads with learnable per-head κ (init from κ={kappa})")
            else:
                logger.info(f"[VariationalFFNDynamic] Multi-head VFE: {n_heads} heads with shared κ={kappa}")

        # =================================================================
        # Per-head learnable temperature κ_h (shared concept with attention)
        # =================================================================
        if learnable_head_kappa and irrep_dims is not None:
            init_kappas = torch.tensor([
                kappa for _d_h in irrep_dims
            ])
            self.log_kappa_per_head = nn.Parameter(torch.log(init_kappas))
            self.register_buffer('_kappa_init', init_kappas)
        else:
            self.log_kappa_per_head = None
            self._kappa_init = None

        # =================================================================
        # Bayesian Precision: Log-barrier form (Eq. 882-884)
        # =================================================================
        # α* = c₀ / (b₀ + KL(q‖p))
        #
        # Log-barrier regulariser: α shrinks as the full KL divergence
        # between the variational posterior q and prior p grows.
        # Gauge-invariant (KL is a gauge scalar).
        # Initialized so α ≈ alpha (the scalar value) when KL ≈ 0.
        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            # Initialize: c₀ = alpha * b₀, b₀ = 1
            # so that α = c₀ / (b₀ + 0) = alpha when KL = 0
            # Per-dimension: each belief dimension k gets its own (c₀_k, b₀_k)
            # so different irrep blocks can learn different precision curves.
            alpha_init = max(alpha, 0.01)  # avoid division by zero
            b0_init = 1.0
            c0_init = alpha_init * b0_init
            # Parameterize via softplus to ensure positivity — shape (K,)
            self.raw_c0 = nn.Parameter(torch.full((embed_dim,), self._softplus_inverse(c0_init)))
            self.raw_b0 = nn.Parameter(torch.full((embed_dim,), self._softplus_inverse(b0_init)))
            logger.info(f"[VariationalFFNDynamic] Bayesian precision enabled (per-dim): "
                        f"c₀={c0_init:.4f}, b₀={b0_init:.1f}, "
                        f"initial α≈{alpha_init} (K={embed_dim})")

        # PriorBank integration
        self.use_prior_bank = use_prior_bank
        self.prior_bank = prior_bank

        if use_prior_bank and prior_bank is not None:
            self.prior_bank = prior_bank
            logger.info(f"[VariationalFFNDynamic] Using PriorBank with token-dependent priors (vocab_size={prior_bank.vocab_size})")
        elif use_prior_bank and prior_bank is None:
            raise ValueError(
                "use_prior_bank=True requires prior_bank to be provided! "
                "Create a PriorBank and pass it to VariationalFFNDynamic."
            )

        # Per-iteration diagnostics (set externally by trainer)
        self._collect_iteration_diagnostics = False
        self._iteration_diagnostics: list = []

        # Lightweight E-step gradient norms (always stored, last iteration only)
        # 'nat_grad_mu'/'nat_grad_sigma'/'grad_phi' are RAW (pre-clip) norms.
        # 'nat_grad_mu_clipped'/'nat_grad_sigma_clipped' are post-clip norms.
        self._e_step_grad_norms: Dict[str, float] = {
            'nat_grad_mu': 0.0, 'nat_grad_sigma': 0.0, 'grad_phi': 0.0,
            'nat_grad_mu_clipped': 0.0, 'nat_grad_sigma_clipped': 0.0,
        }

        # Debug: set to True to print per-component gradient breakdown each E-step iteration.
        # Shows self-coupling, alignment, softmax, obs, Euclidean total, nat_grad amplification.
        self._debug_vfe_gradients: bool = False

        # Lightweight VFE gradient decomposition (always stored when _collect_vfe_metrics=True).
        # Populated from _VFE_GRAD_DEBUG at end of E-step; readable after forward().
        self._collect_vfe_metrics: bool = False
        self.last_vfe_debug: Optional[Dict[str, float]] = None

        # DEQ implicit differentiation
        self.use_deq = use_deq
        self.deq_neumann_terms = deq_neumann_terms
        self.deq_include_phi = deq_include_phi
        if use_deq and implicit_em:
            raise ValueError(
                "use_deq=True and implicit_em=True are mutually exclusive. "
                "Both correct the M-step gradient for E-step dynamics: DEQ via "
                "Neumann-series (I-J)^{-1}, implicit_em via per-dimension IFT "
                "scale s_k. Using both double-counts the correction."
            )

        # Learnable step size (stored in unconstrained space, apply softplus for positive LR)
        if learnable_lr:
            self.raw_lr = nn.Parameter(torch.tensor(self._softplus_inverse(0.1)))
        else:
            # Fixed per-variable E-step rates from config
            self.register_buffer('raw_lr', torch.tensor(self._softplus_inverse(mu_lr)))
            self._fixed_sigma_lr = sigma_lr

        # torch.compile the VFE iteration inner loop (Finding 25).
        # Fuses small element-wise ops and reduces kernel launch overhead.
        # Disabled by default because torch.compile adds compilation latency
        # on the first forward pass and may interact with dynamic shapes.
        if compile_vfe:
            self._vfe_iteration = torch.compile(
                self._vfe_iteration,
                mode='reduce-overhead',
                # fullgraph=False allows Python-level control flow in _vfe_iteration
                fullgraph=False,
            )
            logger.info("[VariationalFFNDynamic] torch.compile applied to _vfe_iteration")

    @property
    def lr(self) -> torch.Tensor:
        """E-step μ learning rate, constrained to (0, ∞) via softplus."""
        return F.softplus(self.raw_lr)

    def _get_kappa_h(self, head_idx: int, d_h: int):
        r"""Get per-head temperature κ_h.

        When learnable_head_kappa=True: κ_h = exp(log_kappa_per_head[h])
        When False: κ_h = self.kappa (bare; callers apply √d_h scaling)
        """
        if self.learnable_head_kappa and self.log_kappa_per_head is not None:
            kappa_h = torch.exp(self.log_kappa_per_head[head_idx])
            # Clamp to [0.5, 1.5] × init, matching attention module (attention.py:2688)
            k0 = self._kappa_init[head_idx]
            return kappa_h.clamp(min=0.5 * k0, max=1.5 * k0)
        return self.kappa

    def _get_sigma_trust(self, effective_lr: torch.Tensor) -> float:
        r"""E-step σ trust region scale.

        When learnable_lr=False, returns the user-specified sigma_lr directly,
        decoupled from the μ step size. When learnable_lr=True, falls back to
        the legacy coupled ratio effective_lr * 0.01.
        """
        if hasattr(self, '_fixed_sigma_lr'):
            return self._fixed_sigma_lr
        return effective_lr * 0.01

    @staticmethod
    def _softplus_inverse(x: float) -> float:
        """Compute inverse of softplus: log(exp(x) - 1)."""
        if x > 20.0:
            return x  # softplus ≈ identity for large x
        return float(np.log(np.expm1(x)))

    def get_bayesian_alpha(
        self,
        mu_q: torch.Tensor,      # (B, N, K)
        mu_p: torch.Tensor,      # (B, N, K)
        sigma_p: torch.Tensor,   # (B, N, K) diagonal or (B, N, K, K) full
        sigma_q: torch.Tensor,   # (B, N, K) diagonal or (B, N, K, K) full
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute per-dimension Bayesian precision via log-barrier form.

        α_k = c₀_k / (b₀_k + kl_k)

        where kl_k is the per-dimension KL contribution. Each belief
        dimension k gets its own precision, so different irrep blocks
        (compact vs non-compact) can learn different regularization curves.

        Diagonal covariance: kl_k decomposes exactly per dimension.
        Full covariance: uses diagonal elements of (Σ_p⁻¹ Σ_q) and
            per-dim Mahalanobis as proxy contributions, with the logdet
            spread uniformly across dimensions.

        Returns:
            alpha: (B, N, K) per-dimension-per-agent precision
        """
        c0 = F.softplus(self.raw_c0)  # (K,)
        b0 = F.softplus(self.raw_b0)  # (K,)

        delta_mu = mu_q - mu_p  # (B, N, K)
        K = mu_q.shape[-1]
        is_diagonal = (sigma_p.dim() == 3)

        if is_diagonal:
            sigma_p_safe = sigma_p.clamp(min=eps)
            sigma_q_safe = sigma_q.clamp(min=eps)
            # Per-dimension KL contributions (no sum — keep (B, N, K))
            trace_term = sigma_q_safe / sigma_p_safe              # (B, N, K)
            mahal_term = delta_mu ** 2 / sigma_p_safe             # (B, N, K)
            logdet_term = torch.log(sigma_p_safe) - torch.log(sigma_q_safe)  # (B, N, K)
        else:
            sigma_p_inv = _safe_spd_inv(sigma_p, eps=eps)  # (B, N, K, K)
            # Per-dim proxy: diagonal of (Σ_p⁻¹ Σ_q)
            prod = torch.matmul(sigma_p_inv, sigma_q)  # (B, N, K, K)
            trace_term = prod.diagonal(dim1=-2, dim2=-1)          # (B, N, K)
            # Per-dim Mahalanobis: δμ_k * (Σ_p⁻¹ δμ)_k
            sp_inv_delta = torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)
            mahal_term = delta_mu * sp_inv_delta                  # (B, N, K)
            # logdet can't decompose per-dim; spread uniformly
            logdet_p = torch.linalg.slogdet(sigma_p.float())[1]  # (B, N)
            logdet_q = torch.linalg.slogdet(sigma_q.float())[1]  # (B, N)
            logdet_term = ((logdet_p - logdet_q) / K).unsqueeze(-1).expand_as(delta_mu)  # (B, N, K)

        # Per-dimension KL contribution
        kl_k = 0.5 * (trace_term + mahal_term - 1 + logdet_term)  # (B, N, K)
        kl_k = kl_k.clamp(min=0.0)

        alpha = c0 / (b0 + kl_k)  # (B, N, K)

        return alpha

    def _precondition_phi_grad(
        self,
        grad_phi: torch.Tensor,   # (..., n_gen)
        phi: torch.Tensor,        # (..., n_gen)
    ) -> torch.Tensor:
        """
        Apply phi gradient preconditioning based on self.phi_natural_gradient mode.

        Modes:
            'clip': Simple norm clipping to 10.0 (no geometric awareness)
            'cartan': Cartan decomposition with fixed sym_dampening=0.1
            'killing': Killing form natural gradient (position-independent, no free params)
            'pullback': Full pullback metric through exp (position-dependent, exact)

        Args:
            grad_phi: Raw Euclidean gradient ∂F/∂φ^a
            phi: Current gauge frame coordinates (needed for 'pullback' mode)

        Returns:
            Preconditioned gradient, same shape as grad_phi
        """
        if self.phi_natural_gradient == 'cartan':
            from transformer.core.gauge_preconditioner import apply_cartan_preconditioning
            return apply_cartan_preconditioning(grad_phi, self._phi_preconditioner)

        elif self.phi_natural_gradient == 'killing':
            from transformer.core.gauge_preconditioner import apply_killing_form_natural_gradient
            return apply_killing_form_natural_gradient(grad_phi, self._phi_preconditioner)

        elif self.phi_natural_gradient == 'pullback':
            from transformer.core.gauge_preconditioner import apply_pullback_natural_gradient
            return apply_pullback_natural_gradient(
                grad_phi, phi, self.generators,
                self._structure_constants, self._gram,
            )

        else:  # 'clip' (default)
            grad_phi_norm = torch.norm(grad_phi, dim=-1, keepdim=True)
            return torch.where(
                grad_phi_norm > 10.0,
                grad_phi * 10.0 / (grad_phi_norm + 1e-6),
                grad_phi
            )

    # =================================================================
    # Direct Omega Gradient (No Lie Algebra / No matrix_exp)
    # =================================================================

    def _compute_omega_grad_direct(
        self,
        omega_current: torch.Tensor,    # (B, N, K, K) direct group elements
        mu_current: torch.Tensor,       # (B, N, K) belief means
        sigma_current: Optional[torch.Tensor],  # (B, N, K) diagonal variances
        is_diagonal: bool,
        mask: Optional[torch.Tensor],
        eps: float,
    ) -> Optional[torch.Tensor]:
        """Compute ∂F_align/∂Ω_i via autograd (vectorized) or analytic fallback.

        Mirrors the phi path: builds alignment_loss as a differentiable computation
        through compute_attention_weights with omega.requires_grad_(), then calls
        torch.autograd.grad for fully vectorized C++ backward pass.

        Falls back to analytic tiled loop when autograd is unavailable.

        Returns:
            grad_omega: (B, N, K, K) gradient ∂F_align/∂Ω_i, or None if no gradient computed.
        """
        from transformer.core.attention import compute_attention_weights

        if sigma_current is None or not is_diagonal:
            return None  # Full covariance path not yet implemented

        B, N, K = mu_current.shape
        device = mu_current.device
        irrep_dims = self.irrep_dims if self.irrep_dims is not None else [K]

        # ── Autograd path (vectorized, matches phi autograd) ──────────
        omega_for_grad = omega_current.detach().clone().requires_grad_(True)

        # Build per-block (omega_h, omega_h_inv) pairs with grad tracking
        _block_exp_pairs = []
        block_start = 0
        for d in irrep_dims:
            block_end = block_start + d
            om_blk = omega_for_grad[:, :, block_start:block_end, block_start:block_end]
            om_inv_blk = torch.linalg.inv(om_blk)
            _block_exp_pairs.append((om_blk, om_inv_blk))
            block_start = block_end

        # Compute alignment loss per head (differentiable through omega_for_grad)
        alignment_loss = torch.tensor(0.0, device=device, dtype=mu_current.dtype)
        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h
            mu_h = mu_current[:, :, block_start:block_end].detach()
            sigma_h = sigma_current[:, :, block_start:block_end].detach() if is_diagonal else None
            gen_h = self.generators[:, block_start:block_end, block_start:block_end]

            kappa_h = self._get_kappa_h(h, d_h)  # Match main multihead path scaling
            beta_kl_h = compute_attention_weights(
                mu_q=mu_h, sigma_q=sigma_h,
                phi=torch.zeros(B, N, 1, device=device),  # dummy
                generators=gen_h,
                kappa=kappa_h, epsilon=eps, mask=mask,
                return_kl=True,
                diagonal_covariance=is_diagonal,
                gauge_param='omega', omega=omega_for_grad,
                gauge_mode=self.gauge_mode,
                irrep_dims=[d_h],
                mask_self_attention=self.mask_self_attention,
                cached_block_exp_pairs=[_block_exp_pairs[h]],
            )
            if isinstance(beta_kl_h, tuple):
                beta_h, kl_h = beta_kl_h
            else:
                beta_h = beta_kl_h
                kl_h = beta_h
            alignment_loss = alignment_loss + self.lambda_belief * (beta_h * kl_h).sum()
            block_start = block_end

        if alignment_loss.grad_fn is not None:
            grad_omega = torch.autograd.grad(
                alignment_loss,
                omega_for_grad,
                create_graph=False,
                retain_graph=False,
            )[0]
            return grad_omega

        return None

    def _retract_omega(
        self,
        omega: torch.Tensor,      # (B, N, K, K)
        grad_omega: torch.Tensor,  # (B, N, K, K) Euclidean gradient
        step_size: float,
        trust_region: float = 0.3,
    ) -> torch.Tensor:
        """Retract Omega update on GL(K) via Lie algebra clipping.

        Computes the natural gradient in the Lie algebra gl(K):
          ξ = Ωᵀ · ∂F/∂Ω           (pullback to Lie algebra)
          clip ||ξ||_F ≤ trust_region  (Riemannian trust region)
          ΔΩ = Ω · ξ_clipped        (push forward)
          Ω_new = Ω - η · ΔΩ        (Euler retraction ≈ Ω·exp(-η·ξ))

        The Riemannian norm ||ξ||_F is invariant under left translation,
        so the trust region is constant in the intrinsic geometry.

        When irrep_dims is set, works block-diagonally to avoid O(K³) matmuls
        on the full K×K matrix (e.g., 5×10×10 instead of 50×50).

        Args:
            omega: Current group elements (B, N, K, K)
            grad_omega: Euclidean gradient ∂F/∂Ω (B, N, K, K)
            step_size: Learning rate
            trust_region: Max Riemannian step size ||ξ||_F

        Returns:
            omega_new: Updated group elements (B, N, K, K)
        """
        irrep_dims = self.irrep_dims if self.irrep_dims is not None else None

        if irrep_dims is not None:
            # Block-diagonal retraction: process each head block independently
            omega_new = omega.clone()
            block_start = 0
            for d in irrep_dims:
                block_end = block_start + d
                om_blk = omega[:, :, block_start:block_end, block_start:block_end]
                gr_blk = grad_omega[:, :, block_start:block_end, block_start:block_end]

                # Lie algebra element per block: ξ_h = Ω_hᵀ · ∂F/∂Ω_h
                omT = om_blk.transpose(-2, -1)
                xi_blk = omT @ gr_blk

                # Clip in Lie algebra norm (= Riemannian norm)
                xi_norm = xi_blk.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
                scale = torch.clamp(trust_region / (xi_norm + 1e-8), max=1.0)
                xi_blk = xi_blk * scale

                # Push forward and Euler retract: Ω·(I - η·ξ)
                omega_new[:, :, block_start:block_end, block_start:block_end] = (
                    om_blk - step_size * (om_blk @ xi_blk)
                )
                block_start = block_end
            return omega_new

        # Fallback: full K×K retraction (no irrep structure)
        OmegaT = omega.transpose(-2, -1)
        xi = OmegaT @ grad_omega

        # Clip in Lie algebra norm
        xi_norm = xi.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
        scale = torch.clamp(trust_region / (xi_norm + 1e-8), max=1.0)
        xi = xi * scale

        # Push forward and Euler retract
        omega_new = omega - step_size * (omega @ xi)

        return omega_new

    def _compute_phi_grad(
        self,
        phi_current: torch.Tensor,
        mu_current: torch.Tensor,
        sigma_current: Optional[torch.Tensor],
        is_diagonal: bool,
        mask: Optional[torch.Tensor],
        eps: float,
        cached_block_exp_pairs: Optional[list] = None,
    ) -> Optional[torch.Tensor]:
        """Compute ∂F_align/∂φ via autograd.

        Returns the preconditioned gradient, or None if no gradient could be computed.
        """
        phi_for_grad = phi_current.clone().requires_grad_(True)

        if self.multihead_vfe:
            alignment_loss = torch.tensor(0.0, device=mu_current.device,
                                          dtype=mu_current.dtype)
            _phi_bep = None
            if self.irrep_dims is not None:
                _phi_bep = fused_block_matrix_exp_pairs(
                    phi_for_grad, self.generators, self.irrep_dims,
                    enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                    skew_symmetric=self._generators_are_skew,
                )
            block_start = 0
            for h, d_h in enumerate(self.irrep_dims):
                block_end = block_start + d_h
                mu_h = mu_current[:, :, block_start:block_end].detach()
                if sigma_current is None:
                    sigma_h = None
                elif is_diagonal:
                    sigma_h = sigma_current[:, :, block_start:block_end].detach()
                else:
                    sigma_h = sigma_current[:, :, block_start:block_end, block_start:block_end].detach()
                gen_h = self.generators[:, block_start:block_end, block_start:block_end]
                kappa_h = self._get_kappa_h(h, d_h)  # Normalize for block dimension
                _phi_head_bep = [_phi_bep[h]] if _phi_bep is not None else None

                beta_phi_h_result = compute_attention_weights(
                    mu_q=mu_h, sigma_q=sigma_h,
                    phi=phi_for_grad, generators=gen_h,
                    kappa=kappa_h, epsilon=eps, mask=mask,
                    return_kl=True,
                    diagonal_covariance=is_diagonal,
                    irrep_dims=[d_h],
                    mask_self_attention=self.mask_self_attention,
                    gauge_mode=self.gauge_mode,
                    cached_block_exp_pairs=_phi_head_bep,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                )
                beta_phi_h, kl_h = beta_phi_h_result
                # Separate direct and softmax coupling weights for phi gradient.
                # d/dphi [sum(beta * KL)] = sum(dBeta/dphi * KL) + sum(beta * dKL/dphi)
                # Direct term (beta * dKL/dphi) gets lambda_belief.
                # Softmax coupling (dBeta/dphi * KL) gets lambda_softmax.
                # Achieved via stop-gradient: detach the factor NOT being differentiated.
                alignment_loss = alignment_loss + (
                    self.lambda_belief * (beta_phi_h.detach() * kl_h).sum()
                    + self.lambda_softmax * (beta_phi_h * kl_h.detach()).sum()
                )
                block_start = block_end
        else:
            beta_for_phi_result = compute_attention_weights(
                mu_q=mu_current.detach(),
                sigma_q=sigma_current.detach() if sigma_current is not None else None,
                phi=phi_for_grad,
                generators=self.generators,
                kappa=self.kappa,
                epsilon=eps,
                mask=mask,
                                return_kl=True,
                diagonal_covariance=is_diagonal,
                irrep_dims=self.irrep_dims,
                mask_self_attention=self.mask_self_attention,
                gauge_mode=self.gauge_mode,
                exact_diagonal_transport=self.exact_diagonal_transport,
            )
            if isinstance(beta_for_phi_result, tuple):
                beta_phi, kl_matrix = beta_for_phi_result
            else:
                beta_phi = beta_for_phi_result
                kl_matrix = beta_phi
            alignment_loss = (
                self.lambda_belief * (beta_phi.detach() * kl_matrix).sum()
                + self.lambda_softmax * (beta_phi * kl_matrix.detach()).sum()
            )

        if alignment_loss.grad_fn is not None:
            grad_phi = torch.autograd.grad(
                alignment_loss,
                phi_for_grad,
                create_graph=False,
                retain_graph=False,
            )[0]
            return self._precondition_phi_grad(grad_phi, phi_current)

        return None

    def _make_deq_step_fn(self, phi_current, mu_p_current, sigma_p,
                           mask, is_diagonal, eps, dtype):
        """Create a differentiable E-step closure for DEQ backward.

        Returns a function (mu, sigma) -> (mu', sigma') that performs one
        VFE natural gradient step with autograd-tracked operations.
        """
        def step_fn(mu_in, sigma_in):
            # Compute transport
            if self.irrep_dims is None and not self.multihead_vfe:
                cached_transport = compute_transport_operators(
                    phi=phi_current,
                    generators=self.generators,
                    enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                    gauge_mode=self.gauge_mode,
                    generators_are_skew=self._generators_are_skew,
                )
            else:
                cached_transport = None

            alpha_eff = self.get_bayesian_alpha(mu_in, mu_p_current, sigma_p, sigma_in, eps=eps) if self.learnable_alpha else self.alpha
            _alpha_c0 = F.softplus(self.raw_c0) if self.learnable_alpha else None

            if self.multihead_vfe:
                # Differentiable multihead step (no detach)
                grad_mu = torch.zeros_like(mu_in)
                grad_sigma = torch.zeros_like(sigma_in)
                block_start = 0
                for h, d_h in enumerate(self.irrep_dims):
                    block_end = block_start + d_h
                    mu_h = mu_in[:, :, block_start:block_end].contiguous()
                    mu_p_h = mu_p_current[:, :, block_start:block_end].contiguous()
                    if is_diagonal:
                        sigma_h = sigma_in[:, :, block_start:block_end].contiguous()
                        sigma_p_h = sigma_p[:, :, block_start:block_end].contiguous()
                    else:
                        sigma_h = sigma_in[:, :, block_start:block_end, block_start:block_end].contiguous()
                        sigma_p_h = sigma_p[:, :, block_start:block_end, block_start:block_end].contiguous()
                    gen_h = self.generators[:, block_start:block_end, block_start:block_end]
                    kappa_h = self._get_kappa_h(h, d_h)  # Match main path scaling

                    beta_h = compute_attention_weights(
                        mu_q=mu_h, sigma_q=sigma_h,
                        phi=phi_current, generators=gen_h,
                        kappa=kappa_h, epsilon=eps, mask=mask,
                        return_kl=False,
                        diagonal_covariance=is_diagonal,
                            mask_self_attention=self.mask_self_attention,
                        gauge_mode=self.gauge_mode,
                        exact_diagonal_transport=self.exact_diagonal_transport,
                    )
                    # Slice alpha per head block if per-dim tensor
                    alpha_h = alpha_eff[:, :, block_start:block_end] if isinstance(alpha_eff, torch.Tensor) and alpha_eff.dim() == 3 else alpha_eff
                    c0_h = _alpha_c0[block_start:block_end] if _alpha_c0 is not None else None
                    grad_mu_h, grad_sigma_h = compute_vfe_gradients_gpu(
                        mu_q=mu_h, sigma_q=sigma_h,
                        mu_p=mu_p_h, sigma_p=sigma_p_h,
                        beta=beta_h, phi=phi_current, generators=gen_h,
                        alpha=alpha_h, lambda_belief=self.lambda_belief, lambda_softmax=self.lambda_softmax,
                        kappa=kappa_h, eps=eps,
                        alpha_c0=c0_h,
                        compute_sigma_align_grad=self.compute_sigma_align_grad,
                        exact_diagonal_transport=self.exact_diagonal_transport,
                    )
                    grad_mu[:, :, block_start:block_end] = grad_mu_h
                    if is_diagonal:
                        grad_sigma[:, :, block_start:block_end] = grad_sigma_h
                    else:
                        if grad_sigma_h.dim() == 3 and d_h == 1:
                            grad_sigma_h = grad_sigma_h.unsqueeze(-1)
                        grad_sigma[:, :, block_start:block_end, block_start:block_end] = grad_sigma_h
                    block_start = block_end
            else:
                beta = compute_attention_weights(
                    mu_q=mu_in, sigma_q=sigma_in,
                    phi=phi_current, generators=self.generators,
                    kappa=self.kappa, epsilon=eps, mask=mask,
                    return_kl=False,
                    diagonal_covariance=is_diagonal,
                    cached_transport=cached_transport,
                    irrep_dims=self.irrep_dims,
                    mask_self_attention=self.mask_self_attention,
                    gauge_mode=self.gauge_mode,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                )
                grad_mu, grad_sigma = compute_vfe_gradients_gpu(
                    mu_q=mu_in, sigma_q=sigma_in,
                    mu_p=mu_p_current, sigma_p=sigma_p,
                    beta=beta, phi=phi_current, generators=self.generators,
                    alpha=alpha_eff, lambda_belief=self.lambda_belief, lambda_softmax=self.lambda_softmax,
                    kappa=self.kappa, eps=eps,
                    alpha_c0=_alpha_c0,
                    cached_transport=cached_transport,
                    compute_sigma_align_grad=self.compute_sigma_align_grad,
                    irrep_dims=self.irrep_dims,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                )

            grad_mu = torch.clamp(grad_mu, min=-1e3, max=1e3)
            grad_sigma = torch.clamp(grad_sigma, min=-1e3, max=1e3)

            nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
                grad_mu, grad_sigma, sigma_in, eps=eps,
            )

            # Norm clipping for nat_grad_mu (was missing in DEQ path)
            nat_grad_mu_norm = torch.linalg.norm(nat_grad_mu, dim=-1, keepdim=True)
            nat_grad_mu = nat_grad_mu * torch.clamp(
                500.0 / (nat_grad_mu_norm + eps), max=1.0
            )

            # Norm clipping for nat_grad_sigma (matching main path Fix 1)
            if is_diagonal:
                nat_grad_sigma_norm = torch.linalg.norm(nat_grad_sigma, dim=-1, keepdim=True)
            else:
                nat_grad_sigma_norm = torch.linalg.norm(
                    nat_grad_sigma.flatten(-2), dim=-1, keepdim=True
                ).unsqueeze(-1)
            nat_grad_sigma = nat_grad_sigma * torch.clamp(
                500.0 / (nat_grad_sigma_norm + eps), max=1.0
            )

            # mu update with trust region
            delta_mu = -self.lr * nat_grad_mu
            if is_diagonal:
                sigma_sqrt = torch.sqrt(sigma_in.float().clamp(min=eps)).to(dtype)
                whitened = delta_mu / sigma_sqrt
            else:
                sigma_diag = torch.diagonal(sigma_in, dim1=-2, dim2=-1).clone().float().clamp(min=eps)
                whitened = delta_mu / torch.sqrt(sigma_diag).to(dtype)

            w_norm = torch.linalg.norm(whitened, dim=-1, keepdim=True)
            scale = torch.clamp(2.0 / (w_norm + eps), max=1.0)
            mu_out = mu_in + scale * delta_mu

            # sigma update — calibrated step: ~0.1% max change per iter
            if self.update_sigma:
                sigma_trust = self._get_sigma_trust(self.lr)
                if is_diagonal:
                    sigma_out = retract_spd_diagonal_torch(
                        sigma_diag=sigma_in, delta_sigma=-nat_grad_sigma,
                        step_size=1.0, trust_region=sigma_trust, eps=eps,
                        sigma_max=self.sigma_max,
                    )
                else:
                    sigma_out = retract_spd_torch(
                        Sigma=sigma_in, delta_Sigma=-nat_grad_sigma,
                        step_size=1.0, trust_region=sigma_trust * 0.5, eps=eps,
                        sigma_max=self.sigma_max,
                    )
            else:
                sigma_out = sigma_in

            return mu_out, sigma_out

        return step_fn

    def _make_deq_step_fn_with_phi(
        self,
        mu_p_current: torch.Tensor,
        sigma_p: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_diagonal: bool,
        eps: float,
        dtype: torch.dtype,
    ):
        r"""Create a differentiable E-step closure for joint (μ, Σ, φ) DEQ backward.

        Returns a function (mu, sigma, phi) -> (mu', sigma', phi') that performs
        one VFE natural gradient step with full autograd tracking, including a
        differentiable phi update via Euclidean gradient descent on F_align.

        The phi update is a differentiable Euclidean step:
            φ' = φ - η_φ · ∂F_align/∂φ
        rather than the Lie group retraction used in the forward E-step.
        At the fixed point (where ∂F_align/∂φ ≈ 0), the Euclidean and
        retraction steps coincide to first order, so the IFT Jacobian
        is correct regardless of which retraction is used forward.
        """
        def step_fn(mu_in: torch.Tensor, sigma_in: torch.Tensor,
                     phi_in: torch.Tensor) -> tuple:
            # --- mu/sigma update (same as _make_deq_step_fn) ---
            if self.irrep_dims is None and not self.multihead_vfe:
                cached_transport = compute_transport_operators(
                    phi=phi_in,
                    generators=self.generators,
                    enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                    gauge_mode=self.gauge_mode,
                    generators_are_skew=self._generators_are_skew,
                )
            else:
                cached_transport = None

            alpha_eff = (
                self.get_bayesian_alpha(mu_in, mu_p_current, sigma_p, sigma_in, eps=eps)
                if self.learnable_alpha else self.alpha
            )
            _alpha_c0 = F.softplus(self.raw_c0) if self.learnable_alpha else None

            if self.multihead_vfe:
                grad_mu = torch.zeros_like(mu_in)
                grad_sigma = torch.zeros_like(sigma_in)
                block_start = 0
                for h, d_h in enumerate(self.irrep_dims):
                    block_end = block_start + d_h
                    mu_h = mu_in[:, :, block_start:block_end].contiguous()
                    mu_p_h = mu_p_current[:, :, block_start:block_end].contiguous()
                    if is_diagonal:
                        sigma_h = sigma_in[:, :, block_start:block_end].contiguous()
                        sigma_p_h = sigma_p[:, :, block_start:block_end].contiguous()
                    else:
                        sigma_h = sigma_in[:, :, block_start:block_end, block_start:block_end].contiguous()
                        sigma_p_h = sigma_p[:, :, block_start:block_end, block_start:block_end].contiguous()
                    gen_h = self.generators[:, block_start:block_end, block_start:block_end]
                    kappa_h = self._get_kappa_h(h, d_h)  # Match main path scaling

                    beta_h = compute_attention_weights(
                        mu_q=mu_h, sigma_q=sigma_h,
                        phi=phi_in, generators=gen_h,
                        kappa=kappa_h, epsilon=eps, mask=mask,
                        return_kl=False,
                        diagonal_covariance=is_diagonal,
                            mask_self_attention=self.mask_self_attention,
                        gauge_mode=self.gauge_mode,
                        exact_diagonal_transport=self.exact_diagonal_transport,
                    )
                    alpha_h = (
                        alpha_eff[:, :, block_start:block_end]
                        if isinstance(alpha_eff, torch.Tensor) and alpha_eff.dim() == 3
                        else alpha_eff
                    )
                    c0_h = _alpha_c0[block_start:block_end] if _alpha_c0 is not None else None
                    grad_mu_h, grad_sigma_h = compute_vfe_gradients_gpu(
                        mu_q=mu_h, sigma_q=sigma_h,
                        mu_p=mu_p_h, sigma_p=sigma_p_h,
                        beta=beta_h, phi=phi_in, generators=gen_h,
                        alpha=alpha_h, lambda_belief=self.lambda_belief, lambda_softmax=self.lambda_softmax,
                        kappa=kappa_h, eps=eps,
                        alpha_c0=c0_h,
                        compute_sigma_align_grad=self.compute_sigma_align_grad,
                        exact_diagonal_transport=self.exact_diagonal_transport,
                    )
                    grad_mu[:, :, block_start:block_end] = grad_mu_h
                    if is_diagonal:
                        grad_sigma[:, :, block_start:block_end] = grad_sigma_h
                    else:
                        if grad_sigma_h.dim() == 3 and d_h == 1:
                            grad_sigma_h = grad_sigma_h.unsqueeze(-1)
                        grad_sigma[:, :, block_start:block_end, block_start:block_end] = grad_sigma_h
                    block_start = block_end
            else:
                beta = compute_attention_weights(
                    mu_q=mu_in, sigma_q=sigma_in,
                    phi=phi_in, generators=self.generators,
                    kappa=self.kappa, epsilon=eps, mask=mask,
                    return_kl=False,
                    diagonal_covariance=is_diagonal,
                    cached_transport=cached_transport,
                    irrep_dims=self.irrep_dims,
                    mask_self_attention=self.mask_self_attention,
                    gauge_mode=self.gauge_mode,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                )
                grad_mu, grad_sigma = compute_vfe_gradients_gpu(
                    mu_q=mu_in, sigma_q=sigma_in,
                    mu_p=mu_p_current, sigma_p=sigma_p,
                    beta=beta, phi=phi_in, generators=self.generators,
                    alpha=alpha_eff, lambda_belief=self.lambda_belief, lambda_softmax=self.lambda_softmax,
                    kappa=self.kappa, eps=eps,
                    alpha_c0=_alpha_c0,
                    cached_transport=cached_transport,
                    compute_sigma_align_grad=self.compute_sigma_align_grad,
                    irrep_dims=self.irrep_dims,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                )

            grad_mu = torch.clamp(grad_mu, min=-1e3, max=1e3)
            grad_sigma = torch.clamp(grad_sigma, min=-1e3, max=1e3)

            nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
                grad_mu, grad_sigma, sigma_in, eps=eps,
            )

            nat_grad_mu_norm = torch.linalg.norm(nat_grad_mu, dim=-1, keepdim=True)
            nat_grad_mu = nat_grad_mu * torch.clamp(
                500.0 / (nat_grad_mu_norm + eps), max=1.0
            )

            if is_diagonal:
                nat_grad_sigma_norm = torch.linalg.norm(nat_grad_sigma, dim=-1, keepdim=True)
            else:
                nat_grad_sigma_norm = torch.linalg.norm(
                    nat_grad_sigma.flatten(-2), dim=-1, keepdim=True
                ).unsqueeze(-1)
            nat_grad_sigma = nat_grad_sigma * torch.clamp(
                500.0 / (nat_grad_sigma_norm + eps), max=1.0
            )

            # mu update with trust region
            delta_mu = -self.lr * nat_grad_mu
            if is_diagonal:
                sigma_sqrt = torch.sqrt(sigma_in.float().clamp(min=eps)).to(dtype)
                whitened = delta_mu / sigma_sqrt
            else:
                sigma_diag = torch.diagonal(sigma_in, dim1=-2, dim2=-1).clone().float().clamp(min=eps)
                whitened = delta_mu / torch.sqrt(sigma_diag).to(dtype)

            w_norm = torch.linalg.norm(whitened, dim=-1, keepdim=True)
            scale = torch.clamp(2.0 / (w_norm + eps), max=1.0)
            mu_out = mu_in + scale * delta_mu

            # sigma update — calibrated step: ~0.1% max change per iter
            if self.update_sigma:
                sigma_trust = self._get_sigma_trust(self.lr)
                if is_diagonal:
                    sigma_out = retract_spd_diagonal_torch(
                        sigma_diag=sigma_in, delta_sigma=-nat_grad_sigma,
                        step_size=1.0, trust_region=sigma_trust, eps=eps,
                        sigma_max=self.sigma_max,
                    )
                else:
                    sigma_out = retract_spd_torch(
                        Sigma=sigma_in, delta_Sigma=-nat_grad_sigma,
                        step_size=1.0, trust_region=sigma_trust * 0.5, eps=eps,
                        sigma_max=self.sigma_max,
                    )
            else:
                sigma_out = sigma_in

            # --- phi update: differentiable Euclidean descent on F_align ---
            # Compute alignment loss with autograd tracking through phi_in.
            # At the fixed point ∂F_align/∂φ ≈ 0, so the Euclidean step and
            # Lie group retraction agree to first order (both give φ' ≈ φ).
            if self.multihead_vfe:
                alignment_loss = torch.tensor(0.0, device=mu_in.device, dtype=mu_in.dtype)
                block_start = 0
                for h, d_h in enumerate(self.irrep_dims):
                    block_end = block_start + d_h
                    mu_h = mu_in[:, :, block_start:block_end].detach()
                    if sigma_in is None:
                        sigma_h = None
                    elif is_diagonal:
                        sigma_h = sigma_in[:, :, block_start:block_end].detach()
                    else:
                        sigma_h = sigma_in[:, :, block_start:block_end, block_start:block_end].detach()
                    gen_h = self.generators[:, block_start:block_end, block_start:block_end]
                    kappa_h = self._get_kappa_h(h, d_h)

                    beta_phi_h_result = compute_attention_weights(
                        mu_q=mu_h, sigma_q=sigma_h,
                        phi=phi_in, generators=gen_h,
                        kappa=kappa_h, epsilon=eps, mask=mask,
                        return_kl=True,
                        diagonal_covariance=is_diagonal,
                        irrep_dims=[d_h],
                            mask_self_attention=self.mask_self_attention,
                        gauge_mode=self.gauge_mode,
                        exact_diagonal_transport=self.exact_diagonal_transport,
                    )
                    beta_phi_h, kl_h = beta_phi_h_result
                    alignment_loss = alignment_loss + self.lambda_belief * (beta_phi_h * kl_h).sum()
                    block_start = block_end
            else:
                beta_for_phi_result = compute_attention_weights(
                    mu_q=mu_in.detach(), sigma_q=sigma_in.detach() if sigma_in is not None else None,
                    phi=phi_in, generators=self.generators,
                    kappa=self.kappa, epsilon=eps, mask=mask,
                    return_kl=True,
                    diagonal_covariance=is_diagonal,
                    irrep_dims=self.irrep_dims,
                    mask_self_attention=self.mask_self_attention,
                    gauge_mode=self.gauge_mode,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                )
                if isinstance(beta_for_phi_result, tuple):
                    beta_phi, kl_matrix = beta_for_phi_result
                else:
                    beta_phi = beta_for_phi_result
                    kl_matrix = beta_phi
                alignment_loss = self.lambda_belief * (beta_phi * kl_matrix).sum()

            # Differentiable Euclidean phi step (autograd tracks through alignment_loss)
            phi_lr_step = self.phi_lr
            if alignment_loss.grad_fn is not None:
                grad_phi_align = torch.autograd.grad(
                    alignment_loss, phi_in,
                    create_graph=True,  # Keep graph for DEQ backward VJP
                    retain_graph=False,
                )[0]
                # Norm clipping (differentiable)
                grad_phi_norm = torch.linalg.norm(grad_phi_align, dim=-1, keepdim=True)
                grad_phi_align = grad_phi_align * torch.clamp(
                    10.0 / (grad_phi_norm + 1e-6), max=1.0
                )
                phi_out = phi_in - phi_lr_step * grad_phi_align
            else:
                phi_out = phi_in

            return mu_out, sigma_out, phi_out

        return step_fn

    # =================================================================
    # Helper: Build block exp-pairs for transport operators
    # =================================================================

    def _build_block_exp_pairs(
        self,
        phi_current: torch.Tensor,        # (B, N, phi_dim)
        omega_current: Optional[torch.Tensor],  # (B, N, K, K) or None
        B: int,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[list]:
        r"""Build per-head (exp_phi_h, exp_neg_phi_h) pairs for transport operators.

        Returns a list of (Omega_h, Omega_h_inv) tuples, one per irrep block,
        or None when irrep_dims is not set.

        Branches:
            gauge_param='omega': extract per-head blocks from omega_current, invert.
            gauge_mode='trivial': return identity pairs for each block.
            gauge_mode='constant' with constant_omega: use constant per-head Omega.
            default (learned): call fused_block_matrix_exp_pairs on phi_current.
        """
        if self.irrep_dims is None:
            return None

        if omega_current is not None and self.gauge_param == 'omega':
            bep = []
            block_start = 0
            for d_h in self.irrep_dims:
                omega_h = omega_current[:, :, block_start:block_start + d_h,
                                        block_start:block_start + d_h]
                omega_h_inv = torch.linalg.inv(omega_h)
                bep.append((omega_h, omega_h_inv))
                block_start += d_h
            return bep

        if self.gauge_mode == 'trivial':
            bep = []
            for d_h in self.irrep_dims:
                eye_h = (torch.eye(d_h, device=device, dtype=dtype)
                         .unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous())
                bep.append((eye_h, eye_h))
            return bep

        if self.gauge_mode == 'constant' and self.constant_omega is not None:
            bep = []
            for h, d_h in enumerate(self.irrep_dims):
                omega_h = self.constant_omega[h].to(device=device, dtype=dtype)
                if getattr(self, 'enforce_orthogonal', False) and d_h >= 2:
                    omega_h = newton_schulz_orthogonalize(omega_h.unsqueeze(0)).squeeze(0)
                eye_h = torch.eye(d_h, device=device, dtype=dtype)
                exp_phi_h = omega_h.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()
                exp_neg_phi_h = eye_h.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()
                bep.append((exp_phi_h, exp_neg_phi_h))
            return bep

        if self.gauge_mode == 'constant':
            # constant without constant_omega: fall back to identity (legacy)
            bep = []
            for d_h in self.irrep_dims:
                eye_h = (torch.eye(d_h, device=device, dtype=dtype)
                         .unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous())
                bep.append((eye_h, eye_h))
            return bep

        # Default: learned phi
        return fused_block_matrix_exp_pairs(
            phi_current, self.generators, self.irrep_dims,
            enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
            skew_symmetric=self._generators_are_skew,
        )

    # =================================================================
    # Helper: Finalize E-step (store state for M-step)
    # =================================================================

    def _finalize_e_step(
        self,
        alpha_effective,
        sigma_p: torch.Tensor,
        sigma_current: Optional[torch.Tensor],
        beta_current: Optional[torch.Tensor],
        beta_heads: list,
        omega_current: Optional[torch.Tensor],
        eps: float,
    ) -> None:
        r"""Store post-E-step state on ``self`` for the M-step.

        Writes: ``_last_alpha_i``, ``_last_beta_for_implicit``,
        ``_last_implicit_mu_scale``, ``_last_implicit_sigma_scale``,
        ``_last_omega``.
        """
        # Alpha
        if self.learnable_alpha:
            self._last_alpha_i = alpha_effective.detach()
        else:
            self._last_alpha_i = None

        # Beta for implicit EM
        if self.implicit_em:
            if self.multihead_vfe and beta_heads:
                self._last_beta_for_implicit = torch.stack(beta_heads, dim=1).detach()
            elif beta_current is not None:
                self._last_beta_for_implicit = beta_current.detach()
            else:
                self._last_beta_for_implicit = None

        # Implicit EM gradient scales
        if self.implicit_em:
            _beta_for_scale = getattr(self, '_last_beta_for_implicit', None)
            if _beta_for_scale is not None:
                _alpha_for_scale = self.alpha
                mu_scale, sigma_scale = compute_implicit_em_scales(
                    alpha_i=_alpha_for_scale,
                    sigma_p=sigma_p,
                    beta=_beta_for_scale,
                    sigma_q=sigma_current if sigma_current is not None else sigma_p,
                    eps=eps,
                )
                self._last_implicit_mu_scale = mu_scale
                self._last_implicit_sigma_scale = sigma_scale
            else:
                self._last_implicit_mu_scale = None
                self._last_implicit_sigma_scale = None
        else:
            self._last_implicit_mu_scale = None
            self._last_implicit_sigma_scale = None

        # Omega for multi-layer propagation
        self._last_omega = omega_current

    # =================================================================
    # Helper: Prepare E-step inputs (sigma init, prior setup, alpha)
    # =================================================================

    def _prepare_e_step_inputs(
        self,
        mu: torch.Tensor,           # (B, N, K)
        sigma: Optional[torch.Tensor],
        mu_prior: torch.Tensor,     # (B, N, K)
        phi: torch.Tensor,          # (B, N, phi_dim)
        omega: Optional[torch.Tensor],
        sigma_prior: Optional[torch.Tensor],
        B: int,
        N: int,
        K: int,
        device: torch.device,
        dtype: torch.dtype,
        eps: float,
    ) -> dict:
        r"""Set up all inputs needed before E-step iterations begin.

        Performs sigma initialisation, prior (mu_p, sigma_p) extraction,
        sigma_p floor clamping, initial sigma clamping, phi/omega cloning,
        and alpha computation.  Returns a dict with keys:

            mu_p_current, sigma_p, mu_current, sigma_current,
            phi_current, omega_current, alpha_effective, _alpha_c0,
            is_diagonal, beta_history (empty list or None),
            has_observations (bool sentinel)
        """
        # ── Sigma initialisation ────────────────────────────────────────
        if sigma is None:
            if self.diagonal_covariance:
                sigma = torch.ones(B, N, K, device=device, dtype=dtype) * 0.1
            else:
                sigma = (0.1 * torch.eye(K, device=device, dtype=dtype)
                         .unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous())

        sigma = squeeze_trailing_singletons(sigma)
        is_diagonal = sigma.dim() == 3

        # ── Prior setup: mu_p, sigma_p ──────────────────────────────────
        # No .clone() needed: downstream ops (clamp, +, =) create new tensors,
        # never mutating the original. .detach() suffices for graph separation.
        if self.amortized_inference and not self.implicit_em:
            mu_p_current = mu_prior
        else:
            mu_p_current = mu_prior.detach()

        # sigma_p is ALWAYS detached in the E-step (M-step parameter)
        if sigma_prior is not None:
            sigma_p = sigma_prior.detach()
        else:
            sigma_p = sigma.detach()

        # E-step sigma_p floor
        _floor = self.e_step_sigma_floor
        if sigma_p.dim() == 3:
            sigma_p = sigma_p.clamp(min=_floor)
        else:
            diag_vals = torch.diagonal(sigma_p, dim1=-2, dim2=-1)
            diag_clamped = diag_vals.clamp(min=_floor)
            sigma_p = sigma_p + torch.diag_embed(diag_clamped - diag_vals)

        # Convert diagonal sigma_p to full covariance if needed
        if not is_diagonal and sigma_p.dim() == 3:
            sigma_p = torch.diag_embed(sigma_p)

        # ── Initial belief state ─────────────────────────────────────────
        # No .clone() needed: the E-step loop reassigns mu_current/sigma_current
        # (mu_current = mu_current + delta), never mutating in-place.
        if self.implicit_em:
            mu_current = mu.detach()
            sigma_current = sigma.detach()
        else:
            mu_current = mu
            sigma_current = sigma

        # Clamp initial sigma to [eps, sigma_max]
        if self.update_sigma:
            _eps = 1e-6
            if sigma_current.dim() == 3:
                sigma_current = sigma_current.clamp(min=_eps, max=self.sigma_max)
            else:
                eigvals, eigvecs = _safe_eigh(sigma_current, jitter=_eps)
                eigvals = eigvals.clamp(min=_eps, max=self.sigma_max)
                sigma_current = eigvecs * eigvals.unsqueeze(-2) @ eigvecs.transpose(-1, -2)

        # ── Phi / omega setup ────────────────────────────────────────────
        # No .clone() needed: phi_current is reassigned (not mutated in-place)
        # in _retract_phi calls within the iteration loop.
        if not self.amortized_inference and self.detach_phi:
            phi_current = phi.detach()
        else:
            phi_current = phi
        omega_current = omega

        # ── Alpha computation ────────────────────────────────────────────
        if self.learnable_alpha:
            alpha_effective = self.get_bayesian_alpha(
                mu_current, mu_p_current, sigma_p, sigma_current, eps=eps
            )
            _alpha_c0 = F.softplus(self.raw_c0)
        else:
            alpha_effective = self.alpha
            _alpha_c0 = None

        return {
            'mu_p_current': mu_p_current,
            'sigma_p': sigma_p,
            'mu_current': mu_current,
            'sigma_current': sigma_current,
            'phi_current': phi_current,
            'omega_current': omega_current,
            'alpha_effective': alpha_effective,
            '_alpha_c0': _alpha_c0,
            'is_diagonal': is_diagonal,
        }

    # =================================================================
    # Helper: Closed-form E-step (precision-weighted fixed point)
    # =================================================================

    def _closed_form_e_step(
        self,
        mu_current: torch.Tensor,
        sigma_current: Optional[torch.Tensor],
        phi_current: torch.Tensor,
        omega_current: Optional[torch.Tensor],
        mu_p_current: torch.Tensor,
        sigma_p: torch.Tensor,
        alpha_effective,
        _alpha_c0,
        is_diagonal: bool,
        B: int,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
        eps: float,
        mask: Optional[torch.Tensor],
        return_beta_history: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], list, Optional[list]]:
        r"""Compute the precision-weighted closed-form VFE fixed point.

        Implements:
            μ_i* = [α·μ_p/σ_p + λ·Σ_j β_ij·(Ω_ij μ_j)/σ_j] / [α/σ_p + λ·Σ_j β_ij/σ_j]
            σ_i* = 1 / [α/σ_p + λ·Σ_j β_ij/σ_j]

        Also applies Picard re-solve (n_picard_steps) and optional phi evolution.

        Returns:
            (mu_current, sigma_current, phi_current, omega_current, beta_heads, beta_history_list)
        """
        # 1. Compute block exp pairs for transport
        _cf_bep = self._build_block_exp_pairs(
            phi_current, omega_current, B, N, device, dtype,
        )

        # 2. Per-head: compute β_h, then closed-form fixed point
        beta_heads: list = []

        if is_diagonal:
            # ─────────────────────────────────────────────────────────────────
            # DIAGONAL CLOSED-FORM: element-wise precision-weighted average
            # ─────────────────────────────────────────────────────────────────
            mu_star = torch.zeros_like(mu_current)
            sigma_star = torch.zeros_like(sigma_current)
            block_start = 0

            for h, d_h in enumerate(self.irrep_dims or [self.embed_dim]):
                block_end = block_start + d_h

                mu_h = mu_current[:, :, block_start:block_end].detach().contiguous()
                mu_p_h = mu_p_current[:, :, block_start:block_end].contiguous()
                sigma_h = sigma_current[:, :, block_start:block_end].detach().contiguous()
                sigma_p_h = sigma_p[:, :, block_start:block_end].contiguous()
                gen_h = self.generators[:, block_start:block_end, block_start:block_end]

                kappa_h = self._get_kappa_h(h, d_h) if self.irrep_dims else self.kappa
                alpha_h = (alpha_effective[:, :, block_start:block_end]
                           if isinstance(alpha_effective, torch.Tensor) and alpha_effective.dim() == 3
                           else alpha_effective)
                _head_bep = [_cf_bep[h]] if _cf_bep is not None else None

                # Compute β_h AND pairwise KL (need KL for softmax coupling)
                beta_kl_result = compute_attention_weights(
                    mu_q=mu_h, sigma_q=sigma_h,
                    phi=phi_current, generators=gen_h,
                    kappa=kappa_h, epsilon=eps, mask=mask,
                    return_kl=True,
                    diagonal_covariance=True,
                    irrep_dims=[d_h] if self.irrep_dims else None,
                    cached_block_exp_pairs=_head_bep,
                    mask_self_attention=self.mask_self_attention,
                    gauge_mode=self.gauge_mode,
                    use_rope=self._use_rope_vfe,
                    rope_base=self._rope_base_vfe,
                )
                beta_h, kl_h = beta_kl_result
                beta_heads.append(beta_h)

                # Transport operators for this head
                exp_phi_h, exp_neg_phi_h = (_cf_bep[h] if _cf_bep is not None else (
                    torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                    torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                ))

                # Prior precision and information vector
                inv_sigma_p_h = 1.0 / sigma_p_h.clamp(min=eps)       # (B, N, d_h)
                prior_prec_h = alpha_h * inv_sigma_p_h                # (B, N, d_h)
                prior_info_h = alpha_h * mu_p_h * inv_sigma_p_h       # (B, N, d_h)

                # Alignment: precision-weighted transported aggregation
                Omega_h_cf = torch.einsum('bikl,bjlm->bijkm', exp_phi_h, exp_neg_phi_h)  # (B, N, N, d_h, d_h)
                sigma_j_t_diag = torch.einsum(
                    'bijkl,bijkl,bjl->bijk', Omega_h_cf, Omega_h_cf, sigma_h
                ).clamp(min=eps)  # (B, N, N, d_h)
                inv_sigma_j_t = 1.0 / sigma_j_t_diag  # (B, N, N, d_h)

                # Transported means: Omega_ij @ mu_j
                mu_j_t_cf = torch.einsum('bijkl,bjl->bijk', Omega_h_cf, mu_h)  # (B, N, N, d_h)

                # Information per pair: (Omega mu_j) / sigma_j_transported
                info_per_pair = mu_j_t_cf * inv_sigma_j_t  # (B, N, N, d_h)

                # Linear terms: attention-weighted precision and information
                align_info_h = self.lambda_belief * torch.einsum('bij,bijk->bik', beta_h, info_per_pair)  # (B, N, d_h)
                align_prec_h = self.lambda_belief * torch.einsum('bij,bijk->bik', beta_h, inv_sigma_j_t)  # (B, N, d_h)

                # ENHANCED CLOSED FORM: Include softmax coupling in fixed point
                kappa_h_val = kappa_h.item() if isinstance(kappa_h, torch.Tensor) else kappa_h
                kappa_h_scaled = max(kappa_h_val * math.sqrt(max(d_h, 1)), eps)

                if self.lambda_softmax > 0 and kappa_h_scaled > 0:
                    w_j_scalar = kl_h * beta_h  # (B, N, N)
                    w_bar = w_j_scalar.sum(dim=-1)  # (B, N)

                    kl_weighted_prec = torch.einsum('bij,bijk->bik', w_j_scalar, inv_sigma_j_t)  # (B, N, d_h)
                    avg_prec_h = align_prec_h / max(self.lambda_belief, eps)
                    S_mu_h = -(self.lambda_softmax / kappa_h_scaled) * (
                        kl_weighted_prec - w_bar.unsqueeze(-1) * avg_prec_h
                    )

                    kl_weighted_info = torch.einsum('bij,bijk->bik', w_j_scalar, info_per_pair)  # (B, N, d_h)
                    avg_info_h = align_info_h / max(self.lambda_belief, eps)
                    c_mu_h = (self.lambda_softmax / kappa_h_scaled) * (
                        kl_weighted_info - w_bar.unsqueeze(-1) * avg_info_h
                    )

                    p_bar_h = avg_prec_h
                    S_sigma_h = -(self.lambda_softmax / (2.0 * kappa_h_scaled)) * torch.einsum(
                        'bij,bijk->bik', w_j_scalar, inv_sigma_j_t - p_bar_h[:, :, None, :]
                    )
                else:
                    S_mu_h = 0.0
                    c_mu_h = 0.0
                    S_sigma_h = 0.0

                del Omega_h_cf, sigma_j_t_diag, inv_sigma_j_t, mu_j_t_cf, info_per_pair

                # Enhanced fixed point
                total_prec_h = prior_prec_h + align_prec_h + S_mu_h    # A + S
                total_info_h = prior_info_h + align_info_h - c_mu_h    # b - c
                mu_star[:, :, block_start:block_end] = total_info_h / total_prec_h.clamp(min=eps)

                entropy_scale = alpha_h + self.lambda_belief
                sigma_prec_h = prior_prec_h + align_prec_h + 2.0 * S_sigma_h
                sigma_star[:, :, block_start:block_end] = (entropy_scale / sigma_prec_h.clamp(min=eps)).clamp(max=self.sigma_max)

                block_start = block_end

            mu_current = mu_star
            if self.update_sigma:
                sigma_current = sigma_star
                if self.isotropic_covariance:
                    scalar_var = sigma_current.mean(dim=-1, keepdim=True)
                    sigma_current = scalar_var.expand_as(sigma_current)

        else:
            # ─────────────────────────────────────────────────────────────────
            # FULL-COVARIANCE CLOSED-FORM: Q_j factorization + Cholesky solve
            # ─────────────────────────────────────────────────────────────────
            K_full = self.embed_dim
            mu_star = torch.zeros_like(mu_current)
            sigma_star_full = torch.zeros(B, N, K_full, K_full, device=device, dtype=dtype)
            block_start = 0

            for h, d_h in enumerate(self.irrep_dims or [self.embed_dim]):
                block_end = block_start + d_h

                mu_h = mu_current[:, :, block_start:block_end].detach().contiguous()
                mu_p_h = mu_p_current[:, :, block_start:block_end].contiguous()
                sigma_h = sigma_current[:, :, block_start:block_end, block_start:block_end].detach().contiguous()
                sigma_p_h = sigma_p[:, :, block_start:block_end, block_start:block_end].contiguous()
                gen_h = self.generators[:, block_start:block_end, block_start:block_end]

                kappa_h = self._get_kappa_h(h, d_h) if self.irrep_dims else self.kappa
                alpha_h = (alpha_effective[:, :, block_start:block_end]
                           if isinstance(alpha_effective, torch.Tensor) and alpha_effective.dim() == 3
                           else alpha_effective)
                _head_bep = [_cf_bep[h]] if _cf_bep is not None else None

                beta_h = compute_attention_weights(
                    mu_q=mu_h, sigma_q=sigma_h,
                    phi=phi_current, generators=gen_h,
                    kappa=kappa_h, epsilon=eps, mask=mask,
                    return_kl=False,
                    diagonal_covariance=False,
                    irrep_dims=[d_h] if self.irrep_dims else None,
                    cached_block_exp_pairs=_head_bep,
                    mask_self_attention=self.mask_self_attention,
                    gauge_mode=self.gauge_mode,
                    use_rope=self._use_rope_vfe,
                    rope_base=self._rope_base_vfe,
                )
                beta_heads.append(beta_h)

                exp_phi_h, exp_neg_phi_h = (_cf_bep[h] if _cf_bep is not None else (
                    torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                    torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                ))
                E_i = exp_neg_phi_h

                sigma_p_h_safe = sigma_p_h + eps * torch.eye(d_h, device=device, dtype=dtype)
                L_p = torch.linalg.cholesky(sigma_p_h_safe)
                Sigma_p_inv_h = torch.cholesky_inverse(L_p)

                if isinstance(alpha_h, torch.Tensor) and alpha_h.dim() == 3:
                    A_prior_h = alpha_h.unsqueeze(-1) * Sigma_p_inv_h
                    b_prior_h = torch.einsum('bijk,bik->bij', A_prior_h, mu_p_h)
                else:
                    A_prior_h = alpha_h * Sigma_p_inv_h
                    b_prior_h = torch.einsum('bijk,bik->bij', A_prior_h, mu_p_h)

                sigma_h_safe = sigma_h + eps * torch.eye(d_h, device=device, dtype=dtype)
                L_j = torch.linalg.cholesky(sigma_h_safe)
                Sigma_j_inv_h = torch.cholesky_inverse(L_j)

                Q_j = torch.einsum('bjlk,bjlm,bjmn->bjkn', exp_phi_h, Sigma_j_inv_h, exp_phi_h)
                r_j = torch.einsum('bjkl,bjl->bjk', exp_neg_phi_h, mu_h)
                Q_agg = torch.einsum('bij,bjkl->bikl', beta_h, Q_j)
                Qr_j = torch.einsum('bjkl,bjl->bjk', Q_j, r_j)
                Qr_agg = torch.einsum('bij,bjk->bik', beta_h, Qr_j)

                A_align_h = self.lambda_belief * torch.einsum('bikl,bikm,bimn->biln', E_i, Q_agg, E_i)
                b_align_h = self.lambda_belief * torch.einsum('bikl,bik->bil', E_i, Qr_agg)

                A_h = A_prior_h + A_align_h + eps * torch.eye(d_h, device=device, dtype=dtype)
                b_h = b_prior_h + b_align_h

                L_A = torch.linalg.cholesky(A_h)
                mu_star_h = torch.cholesky_solve(b_h.unsqueeze(-1), L_A).squeeze(-1)
                A_inv_h = torch.cholesky_inverse(L_A)

                if isinstance(alpha_h, torch.Tensor) and alpha_h.dim() == 3:
                    entropy_scale = (alpha_h + self.lambda_belief).unsqueeze(-1)
                    Sigma_star_h = entropy_scale * A_inv_h
                else:
                    Sigma_star_h = (alpha_h + self.lambda_belief) * A_inv_h

                Sigma_star_h = Sigma_star_h.clamp(max=self.sigma_max)
                mu_star[:, :, block_start:block_end] = mu_star_h
                sigma_star_full[:, :, block_start:block_end, block_start:block_end] = Sigma_star_h

                block_start = block_end

            mu_current = mu_star
            if self.update_sigma:
                sigma_current = sigma_star_full

        beta_current = beta_heads[-1] if beta_heads else None

        # ── Iterative re-solve: Picard steps ─────────────────────────────
        if self.n_picard_steps > 0 and self.lambda_softmax > 0 and _cf_bep is not None:
            if is_diagonal:
                for _resolve_iter in range(self.n_picard_steps):
                    mu_prev = mu_current.clone()
                    mu_star = torch.zeros_like(mu_current)
                    sigma_star = torch.zeros_like(sigma_current)
                    beta_heads_new = []
                    block_start = 0

                    for h, d_h in enumerate(self.irrep_dims or [self.embed_dim]):
                        block_end = block_start + d_h

                        mu_h = mu_current[:, :, block_start:block_end].detach().contiguous()
                        mu_p_h = mu_p_current[:, :, block_start:block_end].contiguous()
                        sigma_h = sigma_current[:, :, block_start:block_end].detach().contiguous()
                        sigma_p_h = sigma_p[:, :, block_start:block_end].contiguous()
                        gen_h = self.generators[:, block_start:block_end, block_start:block_end]

                        kappa_h = self._get_kappa_h(h, d_h) if self.irrep_dims else self.kappa
                        alpha_h_iter = alpha_effective
                        if isinstance(alpha_effective, torch.Tensor) and alpha_effective.dim() == 3:
                            alpha_h_iter = alpha_effective[:, :, block_start:block_end]
                        if self.learnable_alpha and hasattr(self, 'get_bayesian_alpha'):
                            alpha_n = self.get_bayesian_alpha(
                                mu_current, mu_p_current, sigma_p, sigma_current, eps=eps
                            )
                            if isinstance(alpha_n, torch.Tensor) and alpha_n.dim() == 3:
                                alpha_h_iter = alpha_n[:, :, block_start:block_end]
                            else:
                                alpha_h_iter = alpha_n

                        _head_bep = [_cf_bep[h]] if _cf_bep is not None else None

                        beta_h, kl_h = compute_attention_weights(
                            mu_q=mu_h, sigma_q=sigma_h,
                            phi=phi_current, generators=gen_h,
                            kappa=kappa_h, epsilon=eps, mask=mask,
                            return_kl=True,
                            diagonal_covariance=True,
                            irrep_dims=[d_h] if self.irrep_dims else None,
                            cached_block_exp_pairs=_head_bep,
                            mask_self_attention=self.mask_self_attention,
                            gauge_mode=self.gauge_mode,
                            use_rope=self._use_rope_vfe,
                            rope_base=self._rope_base_vfe,
                        )
                        beta_heads_new.append(beta_h)

                        exp_phi_h, exp_neg_phi_h = (_cf_bep[h] if _cf_bep is not None else (
                            torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                            torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                        ))

                        inv_sigma_p_h = 1.0 / sigma_p_h.clamp(min=eps)
                        prior_prec_h = alpha_h_iter * inv_sigma_p_h
                        prior_info_h = alpha_h_iter * mu_p_h * inv_sigma_p_h

                        Omega_h_cf = torch.einsum('bikl,bjlm->bijkm', exp_phi_h, exp_neg_phi_h)
                        sigma_j_t_diag = torch.einsum(
                            'bijkl,bijkl,bjl->bijk', Omega_h_cf, Omega_h_cf, sigma_h
                        ).clamp(min=eps)
                        inv_sigma_j_t = 1.0 / sigma_j_t_diag
                        mu_j_t_cf = torch.einsum('bijkl,bjl->bijk', Omega_h_cf, mu_h)
                        info_per_pair = mu_j_t_cf * inv_sigma_j_t

                        align_info_h = self.lambda_belief * torch.einsum('bij,bijk->bik', beta_h, info_per_pair)
                        align_prec_h = self.lambda_belief * torch.einsum('bij,bijk->bik', beta_h, inv_sigma_j_t)

                        kappa_h_val = kappa_h.item() if isinstance(kappa_h, torch.Tensor) else kappa_h
                        kappa_h_scaled = max(kappa_h_val * math.sqrt(max(d_h, 1)), eps)

                        if self.lambda_softmax > 0 and kappa_h_scaled > 0:
                            w_j_scalar = kl_h * beta_h
                            w_bar = w_j_scalar.sum(dim=-1)
                            kl_weighted_prec = torch.einsum('bij,bijk->bik', w_j_scalar, inv_sigma_j_t)
                            avg_prec_h = align_prec_h / max(self.lambda_belief, eps)
                            S_mu_h = -(self.lambda_softmax / kappa_h_scaled) * (
                                kl_weighted_prec - w_bar.unsqueeze(-1) * avg_prec_h
                            )
                            kl_weighted_info = torch.einsum('bij,bijk->bik', w_j_scalar, info_per_pair)
                            avg_info_h = align_info_h / max(self.lambda_belief, eps)
                            c_mu_h = (self.lambda_softmax / kappa_h_scaled) * (
                                kl_weighted_info - w_bar.unsqueeze(-1) * avg_info_h
                            )
                            p_bar_h = avg_prec_h
                            S_sigma_h = -(self.lambda_softmax / (2.0 * kappa_h_scaled)) * torch.einsum(
                                'bij,bijk->bik', w_j_scalar, inv_sigma_j_t - p_bar_h[:, :, None, :]
                            )
                        else:
                            S_mu_h = 0.0
                            c_mu_h = 0.0
                            S_sigma_h = 0.0

                        del Omega_h_cf, sigma_j_t_diag, inv_sigma_j_t, mu_j_t_cf, info_per_pair

                        total_prec_h = prior_prec_h + align_prec_h + S_mu_h
                        total_info_h = prior_info_h + align_info_h - c_mu_h
                        mu_star[:, :, block_start:block_end] = total_info_h / total_prec_h.clamp(min=eps)

                        entropy_scale = alpha_h_iter + self.lambda_belief
                        sigma_prec_h = prior_prec_h + align_prec_h + 2.0 * S_sigma_h
                        sigma_star[:, :, block_start:block_end] = (
                            entropy_scale / sigma_prec_h.clamp(min=eps)
                        ).clamp(max=self.sigma_max)

                        block_start = block_end

                    mu_current = mu_star
                    if self.update_sigma:
                        sigma_current = sigma_star
                        if self.isotropic_covariance:
                            scalar_var = sigma_current.mean(dim=-1, keepdim=True)
                            sigma_current = scalar_var.expand_as(sigma_current)

                    beta_heads = beta_heads_new

                    rel_change = (mu_current - mu_prev).norm() / mu_prev.norm().clamp(min=eps)
                    if rel_change < 1e-4:
                        break

            else:
                # FULL COVARIANCE: Original Picard (linear-only CF + grad)
                mu_0 = mu_current.clone()
                sigma_0 = sigma_current

                for _picard_iter in range(self.n_picard_steps):
                    grad_softmax_full = torch.zeros_like(mu_current)
                    block_start = 0

                    for h, d_h in enumerate(self.irrep_dims or [self.embed_dim]):
                        block_end = block_start + d_h
                        beta_h = beta_heads[h]
                        exp_phi_h, exp_neg_phi_h = _cf_bep[h]

                        mu_h = mu_current[:, :, block_start:block_end]

                        Omega_h = torch.einsum('bikl,bjlm->bijkm', exp_phi_h, exp_neg_phi_h)
                        mu_j_t = torch.einsum('bijkl,bjl->bijk', Omega_h, mu_h)
                        delta = mu_h[:, :, None, :] - mu_j_t

                        sigma_h_blk = sigma_0[:, :, block_start:block_end, block_start:block_end]

                        Sigma_j_t = torch.einsum('bijkl,bjlm,bijnm->bijkn', Omega_h, sigma_h_blk, Omega_h)
                        Sigma_j_t = Sigma_j_t + eps * torch.eye(d_h, device=device, dtype=dtype)
                        Sigma_j_t_inv = torch.linalg.inv(Sigma_j_t)

                        grad_kl_pair = torch.einsum('bijkl,bijl->bijk', Sigma_j_t_inv, delta)

                        sigma_i_h = sigma_h_blk
                        trace_h = torch.einsum('bijkl,bilk->bij', Sigma_j_t_inv, sigma_i_h)
                        mahal_h = torch.einsum('bijk,bijk->bij', delta, grad_kl_pair)
                        logdet_jt = torch.linalg.slogdet(Sigma_j_t)[1]
                        logdet_i = torch.linalg.slogdet(
                            sigma_i_h + eps * torch.eye(d_h, device=device, dtype=dtype)
                        )[1]
                        logdet_h = logdet_jt - logdet_i.unsqueeze(2)
                        kl_h = (0.5 * (trace_h + mahal_h - d_h + logdet_h)).clamp(min=0.0)

                        kappa_h = self._get_kappa_h(h, d_h) if self.irrep_dims else self.kappa
                        kappa_h_scaled = max(kappa_h * math.sqrt(max(d_h, 1)), eps)
                        avg_grad_h = torch.einsum('bij,bijk->bik', beta_h, grad_kl_pair)
                        grad_dev = avg_grad_h.unsqueeze(2) - grad_kl_pair
                        d_beta_d_mu_h = beta_h.unsqueeze(-1) * grad_dev / kappa_h_scaled
                        grad_softmax_h = self.lambda_softmax * torch.einsum(
                            'bij,bijk->bik', kl_h, d_beta_d_mu_h
                        )
                        grad_softmax_full[:, :, block_start:block_end] = grad_softmax_h

                        block_start = block_end

                    if self.learnable_alpha and _alpha_c0 is not None:
                        alpha_n = self.get_bayesian_alpha(
                            mu_current, mu_p_current, sigma_p, sigma_current, eps=eps
                        )
                        sigma_p_diag = torch.diagonal(sigma_p, dim1=-2, dim2=-1).clamp(min=eps)
                        sigma_q_diag = torch.diagonal(sigma_current, dim1=-2, dim2=-1).clamp(min=eps)
                        delta_mu_p = mu_current - mu_p_current

                        alpha_mismatch = (alpha_n - alpha_effective) * delta_mu_p / sigma_p_diag
                        kl_k = 0.5 * (sigma_q_diag / sigma_p_diag + delta_mu_p ** 2 / sigma_p_diag
                                      - 1.0 + torch.log(sigma_p_diag) - torch.log(sigma_q_diag))
                        kl_k = kl_k.clamp(min=0.0)
                        product_rule = -(alpha_n ** 2 / _alpha_c0) * kl_k * delta_mu_p / sigma_p_diag
                        grad_softmax_full = grad_softmax_full + alpha_mismatch + product_rule

                    correction = torch.zeros_like(mu_current)
                    block_start = 0
                    for h, d_h in enumerate(self.irrep_dims or [self.embed_dim]):
                        block_end = block_start + d_h
                        Sigma_0_h = sigma_0[:, :, block_start:block_end, block_start:block_end]
                        grad_h = grad_softmax_full[:, :, block_start:block_end]
                        correction[:, :, block_start:block_end] = torch.einsum(
                            'bijk,bik->bij', Sigma_0_h, grad_h
                        )
                        block_start = block_end
                    w_norm = (grad_softmax_full * correction).sum(
                        dim=-1, keepdim=True
                    ).clamp(min=0.0).sqrt()

                    scale = (self.picard_trust_region / w_norm.clamp(min=eps)).clamp(max=1.0)
                    mu_current = mu_0 - scale * correction

        # ── Beta history for return ───────────────────────────────────────
        beta_history_out: Optional[list] = None
        if return_beta_history:
            beta_stacked = (torch.stack(beta_heads, dim=1) if len(beta_heads) > 1
                            else beta_heads[0].unsqueeze(1))
            beta_history_out = [beta_stacked.detach().clone()]

        # ── Phi evolution via gradient (phi enters nonlinearly) ───────────
        if (self.update_phi and torch.is_grad_enabled()
                and self.gauge_mode not in ('trivial', 'constant')):
            _use_omega = omega_current is not None and self.gauge_param == 'omega'
            if _use_omega:
                grad_omega = self._compute_omega_grad_direct(
                    omega_current, mu_current, sigma_current,
                    is_diagonal, mask, eps,
                )
                if grad_omega is not None:
                    omega_current = self._retract_omega(
                        omega_current, grad_omega, self.phi_lr,
                        trust_region=getattr(self, 'omega_trust_region', 0.3),
                    )
            else:
                _phi_bep_cf = (fused_block_matrix_exp_pairs(
                    phi_current, self.generators, self.irrep_dims,
                    enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                    skew_symmetric=self._generators_are_skew,
                ) if self.irrep_dims is not None and self.gauge_mode == 'learned' else None)
                grad_phi = self._compute_phi_grad(
                    phi_current, mu_current, sigma_current,
                    is_diagonal, mask, eps,
                    cached_block_exp_pairs=_phi_bep_cf,
                )
                if grad_phi is not None:
                    phi_current = _retract_phi(
                        phi=phi_current,
                        delta_phi=-grad_phi,
                        generators=self.generators,
                        step_size=self.phi_lr,
                        max_norm=self.phi_max_norm,
                    )

        return mu_current, sigma_current, phi_current, omega_current, beta_heads, beta_history_out


    # =================================================================
    # Helper: Single VFE iteration (extracted from forward loop)
    # =================================================================

    def _vfe_iteration(
        self,
        iteration: int,
        mu_current: torch.Tensor,
        sigma_current: Optional[torch.Tensor],
        phi_current: torch.Tensor,
        omega_current: Optional[torch.Tensor],
        mu_p_current: torch.Tensor,
        sigma_p: torch.Tensor,
        alpha_effective,
        _alpha_c0,
        is_diagonal: bool,
        B: int,
        N: int,
        eps: float,
        mask: Optional[torch.Tensor],
        has_observations: bool,
        targets: Optional[torch.Tensor],
        W_out: Optional[torch.Tensor],
        return_beta_history: bool,
        _detach_e_step: bool = True,
        _obs_cache: Optional[dict] = None,
        _precomputed_block_exp_pairs=None,
        connection_delta: Optional[torch.Tensor] = None,
        cocycle_relaxation: float = 0.0,
    ):
        r"""Execute one VFE natural-gradient iteration.

        Args:
            _precomputed_block_exp_pairs: If provided, reuse these block exp
                pairs instead of recomputing. Used when update_phi_per_iteration
                is False to avoid redundant matrix exponentials across iterations.
            connection_delta: Frozen non-flat connection δ_ij (B, N, N, n_gen).
                When provided, transport becomes Ω = exp(φ_i)·exp(α·δ·G)·exp(−φ_j).
            cocycle_relaxation: Scale factor for connection_delta.

        Returns:
            (mu_current, sigma_current, phi_current, omega_current,
             beta_current, beta_heads, alpha_effective, beta_history_entry)
            where beta_history_entry is a stacked beta (if return_beta_history)
            or None.
        """
        beta_heads = []
        beta_current = None
        beta_history_entry = None
        _is_final_iter = (iteration == self.n_iterations - 1)

        # Cosine decay: lr drops from 1.0 to 0.1 across iterations
        # Steeper than linear 0.5 decay — stabilizes later iterations where
        # natural gradients can amplify and cause oscillatory divergence
        if self.n_iterations > 1:
            progress = iteration / (self.n_iterations - 1)  # 0→1
            decay_factor = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            decay_factor = 1.0
        effective_lr = self.lr * decay_factor

        # =================================================================
        # STEP 0: Precompute transport operators ONCE per iteration
        # =================================================================
        # Both compute_attention_weights and compute_vfe_gradients_gpu need
        # the same Ω_ij = exp(φ_i)·exp(-φ_j). Computing once and passing
        # cached_transport avoids redundant matrix exponentials.
        # Skip caching when using block-diagonal or chunked paths (they
        # compute transport internally in chunks to save memory).
        _nonflat_omega = None  # Per-head sliceable non-flat Omega (multihead path)
        if self.irrep_dims is None and not self.multihead_vfe:
            if omega_current is not None and self.gauge_param == 'omega':
                # Direct omega: build full-K transport from omega blocks
                from transformer.core.transport_ops import compute_transport_operators_direct
                cached_transport = compute_transport_operators_direct(
                    omega=omega_current,
                    connection_delta=connection_delta,
                    generators=self.generators if connection_delta is not None else None,
                    cocycle_relaxation=cocycle_relaxation,
                )
            elif self.gauge_mode == 'constant' and self.constant_omega is not None:
                # Constant gauge with known Ω: build full-K transport from
                # per-head constant_omega blocks (non-block-diagonal path).
                K = mu_current.shape[-1]
                omega_full = torch.eye(K, device=mu_current.device, dtype=mu_current.dtype)
                _blk_start = 0
                for h_idx in range(len(self.constant_omega)):
                    omega_h = self.constant_omega[h_idx].to(
                        device=mu_current.device, dtype=mu_current.dtype)
                    d_h = omega_h.shape[0]
                    if getattr(self, 'enforce_orthogonal', False) and d_h >= 2:
                        omega_h = newton_schulz_orthogonalize(
                            omega_h.unsqueeze(0)).squeeze(0)
                    omega_full[_blk_start:_blk_start+d_h, _blk_start:_blk_start+d_h] = omega_h
                    _blk_start += d_h
                Omega = omega_full.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
                    B, N, N, -1, -1).contiguous()
                cached_transport = {'Omega': Omega}
            else:
                cached_transport = compute_transport_operators(
                    phi=phi_current,
                    generators=self.generators,
                    enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                    gauge_mode=self.gauge_mode,
                    generators_are_skew=self._generators_are_skew,
                    connection_delta=connection_delta,
                    cocycle_relaxation=cocycle_relaxation,
                )
        else:
            if connection_delta is not None and cocycle_relaxation > 0:
                # Non-flat multihead: compute full-K Omega once, slice per head.
                # The fused block-diagonal path doesn't support non-flat transport,
                # so we pre-compute the full Omega and inject per-head slices via
                # cached_transport in the per-head loop below.
                _nonflat_transport = compute_transport_operators(
                    phi=phi_current,
                    generators=self.generators,
                    enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                    gauge_mode=self.gauge_mode,
                    generators_are_skew=self._generators_are_skew,
                    connection_delta=connection_delta,
                    cocycle_relaxation=cocycle_relaxation,
                )
                _nonflat_omega = _nonflat_transport['Omega']  # (B, N, N, K, K)
            cached_transport = None

        # =================================================================
        # Determine alpha: Bayesian precision or fixed scalar
        # Recompute per-iteration for learnable alpha (state-dependent);
        # for fixed alpha, alpha_effective was set before the loop.
        # =================================================================
        if self.learnable_alpha:
            alpha_effective = self.get_bayesian_alpha(
                mu_current, mu_p_current, sigma_p, sigma_current, eps=eps
            )  # (B, N, K) - per-dim gauge-invariant, state-dependent

        # Initialize cached block exp pairs for phi gradient reuse
        _mh_cached_bep = None
        _cached_bep = None

        # Enable/disable VFE gradient debug dict for this iteration
        # _vfe_utils_mod._VFE_GRAD_DEBUG accessed via _vfe_utils_mod
        if self._debug_vfe_gradients or self._collect_vfe_metrics:
            _vfe_utils_mod._VFE_GRAD_DEBUG = {}
        else:
            _vfe_utils_mod._VFE_GRAD_DEBUG = None

        if self.multihead_vfe:
            # =============================================================
            # MULTI-HEAD VFE: Per-block β_h through iterations
            # =============================================================
            # Each irrep block gets its own attention pattern.
            # This maintains head diversity through all VFE iterations,
            # enabling different heads to cluster at different scales.
            grad_mu = torch.zeros_like(mu_current)
            grad_sigma = torch.zeros_like(sigma_current)
            beta_heads = []  # For history tracking

            # Precompute block_exp_pairs ONCE for all heads.
            # When update_phi_per_iteration=False, the caller passes in
            # precomputed pairs to avoid redundant matrix exp across iterations.
            if _precomputed_block_exp_pairs is not None:
                _mh_cached_bep = _precomputed_block_exp_pairs
            else:
                _mh_cached_bep = self._build_block_exp_pairs(
                    phi_current, omega_current, B, N,
                    mu_current.device, mu_current.dtype,
                )

            # =============================================================
            # FUSED MULTI-HEAD VFE: Compute β_h and gradients in single
            # Omega pass per head (eliminates redundant Omega construction).
            # Previously: compute_attention_weights + compute_vfe_gradients_gpu
            # built Omega separately → 2× Omega per head. Now: 1× per head.
            # =============================================================
            _use_fused_mh = (is_diagonal and self.irrep_dims is not None
                             and not self.exact_diagonal_transport)
            block_start = 0
            for h, d_h in enumerate(self.irrep_dims):
                block_end = block_start + d_h

                mu_h = mu_current[:, :, block_start:block_end]
                if _detach_e_step:
                    mu_h = mu_h.detach()
                mu_h = mu_h.contiguous()
                mu_p_h = mu_p_current[:, :, block_start:block_end].contiguous()
                if is_diagonal:
                    sigma_h = sigma_current[:, :, block_start:block_end]
                    if _detach_e_step:
                        sigma_h = sigma_h.detach()
                    sigma_h = sigma_h.contiguous()
                    sigma_p_h = sigma_p[:, :, block_start:block_end].contiguous()
                else:
                    sigma_h = sigma_current[:, :, block_start:block_end, block_start:block_end]
                    if _detach_e_step:
                        sigma_h = sigma_h.detach()
                    sigma_h = sigma_h.contiguous()
                    sigma_p_h = sigma_p[:, :, block_start:block_end, block_start:block_end].contiguous()
                gen_h = self.generators[:, block_start:block_end, block_start:block_end]

                # Scale kappa by sqrt(d_h) to normalize KL across different-dim
                # super-blocks. Without this, larger blocks (e.g., 12-dim from
                # cross-coupled heads) produce proportionally larger KL values,
                # causing attention sharpness imbalance vs smaller blocks.
                kappa_h = self._get_kappa_h(h, d_h)
                _head_bep = [_mh_cached_bep[h]] if _mh_cached_bep is not None else None

                alpha_h = alpha_effective[:, :, block_start:block_end] if isinstance(alpha_effective, torch.Tensor) and alpha_effective.dim() == 3 else alpha_effective
                c0_h = _alpha_c0[block_start:block_end] if _alpha_c0 is not None else None

                # Non-flat transport: extract per-head Omega slice for this block
                _head_ct = None
                if _nonflat_omega is not None:
                    _head_ct = {'Omega': _nonflat_omega[:, :, :, block_start:block_end, block_start:block_end]}

                # EXPERIMENTAL: rope_full_gauge path uses an autograd-based
                # gradient computation that lifts σ to full covariance and
                # rotates BOTH μ and Σ by R(θ).  Slower than the fused path.
                # Only diagonal σ is supported.  See vfe_gradients.py for the
                # implementation.
                #
                # CRITICAL: torch.inference_mode() (used by model.generate)
                # marks tensors as inference tensors that cannot be promoted
                # to require_grad even inside enable_grad().  Detect this and
                # fall back to the standard fused path for generation.  The
                # fallback path still has the rope chain-rule fix (Option C),
                # so it's correct, just not the experimental Option-B.
                _use_rope_full = (
                    getattr(self, '_rope_full_gauge_vfe', False)
                    and self._use_rope_vfe
                    and is_diagonal
                    and _nonflat_omega is None
                    and not torch.is_inference_mode_enabled()
                )
                if _use_rope_full:
                    from transformer.core.vfe_gradients import (
                        _compute_rope_full_gauge_gradient_per_head,
                    )
                    beta_h, grad_mu_h, grad_sigma_h = _compute_rope_full_gauge_gradient_per_head(
                        mu_h=mu_h, sigma_h=sigma_h,
                        mu_p_h=mu_p_h, sigma_p_h=sigma_p_h,
                        phi=phi_current, gen_h=gen_h,
                        alpha=alpha_h,
                        lambda_belief=self.lambda_belief,
                        lambda_softmax=self.lambda_softmax,
                        kappa=kappa_h, eps=eps,
                        rope_base=self._rope_base_vfe,
                        d_h=d_h,
                        cached_block_exp_pairs=_head_bep,
                        enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                        mask=mask,
                        mask_self_attention=self.mask_self_attention,
                    )
                elif _use_fused_mh and _nonflat_omega is None:
                    # FUSED: single pass computes β_h AND gradients (1× Omega)
                    # Not compatible with non-flat transport (fused path builds
                    # Omega from block exp pairs internally).
                    beta_h, grad_mu_h, grad_sigma_h, _ = _fused_attention_and_vfe_gradients_block_diag(
                        mu_q=mu_h, sigma_q=sigma_h,
                        mu_p=mu_p_h, sigma_p=sigma_p_h,
                        phi=phi_current, generators=gen_h,
                        alpha=alpha_h, lambda_belief=self.lambda_belief, lambda_softmax=self.lambda_softmax,
                        kappa=kappa_h, eps=eps,
                        irrep_dims=[d_h],
                        compute_sigma_align_grad=self.compute_sigma_align_grad,
                        enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                        alpha_c0=c0_h,
                        cached_block_exp_pairs=_head_bep,
                        mask=mask,
                        mask_self_attention=self.mask_self_attention,
                        use_rope=self._use_rope_vfe,
                        rope_base=self._rope_base_vfe,
                    )
                else:
                    # Fallback: separate attention + gradient (full covariance)
                    # When _head_ct has non-flat Omega, it takes priority over
                    # cached_block_exp_pairs in both functions.
                    beta_h = compute_attention_weights(
                        mu_q=mu_h, sigma_q=sigma_h,
                        phi=phi_current, generators=gen_h,
                        kappa=kappa_h, epsilon=eps, mask=mask,
                        return_kl=False,
                        diagonal_covariance=is_diagonal,
                        irrep_dims=[d_h],
                        mask_self_attention=self.mask_self_attention,
                        gauge_mode=self.gauge_mode,
                        cached_block_exp_pairs=_head_bep,
                        cached_transport=_head_ct,
                        use_rope=self._use_rope_vfe,
                        rope_base=self._rope_base_vfe,
                        exact_diagonal_transport=self.exact_diagonal_transport,
                    )
                    grad_mu_h, grad_sigma_h = compute_vfe_gradients_gpu(
                        mu_q=mu_h, sigma_q=sigma_h,
                        mu_p=mu_p_h, sigma_p=sigma_p_h,
                        beta=beta_h, phi=phi_current, generators=gen_h,
                        alpha=alpha_h, lambda_belief=self.lambda_belief, lambda_softmax=self.lambda_softmax,
                        kappa=kappa_h, eps=eps, alpha_c0=c0_h,
                        compute_sigma_align_grad=self.compute_sigma_align_grad,
                        irrep_dims=[d_h],
                        cached_block_exp_pairs=_head_bep,
                        cached_transport=_head_ct,
                        exact_diagonal_transport=self.exact_diagonal_transport,
                        use_rope=self._use_rope_vfe,
                        rope_base=self._rope_base_vfe,
                    )

                beta_heads.append(beta_h)
                grad_mu[:, :, block_start:block_end] = grad_mu_h
                if is_diagonal:
                    grad_sigma[:, :, block_start:block_end] = grad_sigma_h
                else:
                    # compute_vfe_gradients_gpu squeezes trailing singletons,
                    # so d_h=1 heads return (B,N,1) instead of (B,N,1,1).
                    # Restore full covariance shape for block assignment.
                    if grad_sigma_h.dim() == 3 and d_h == 1:
                        grad_sigma_h = grad_sigma_h.unsqueeze(-1)
                    grad_sigma[:, :, block_start:block_end, block_start:block_end] = grad_sigma_h

                # Accumulate per-head debug norms (sum of squares for proper norm aggregation)
                if _vfe_utils_mod._VFE_GRAD_DEBUG is not None and _vfe_utils_mod._VFE_GRAD_DEBUG:
                    _pfx = f'head{h}(d={d_h})'
                    for _k, _v in list(_vfe_utils_mod._VFE_GRAD_DEBUG.items()):
                        # Store per-head values with prefix, clear base keys
                        _vfe_utils_mod._VFE_GRAD_DEBUG[f'{_pfx}/{_k}'] = _v
                    # Reset base keys for next head
                    for _k in [k for k in _vfe_utils_mod._VFE_GRAD_DEBUG if '/' not in k]:
                        del _vfe_utils_mod._VFE_GRAD_DEBUG[_k]

                block_start = block_end

            if return_beta_history:
                beta_stacked = torch.stack(beta_heads, dim=1)
                beta_history_entry = beta_stacked.detach().clone()
            beta_current = beta_heads[-1]

        else:
            # =============================================================
            # SINGLE-β VFE: Original behavior (all blocks share one β)
            # =============================================================
            # SINGLE-β VFE: All blocks share one β
            # Use fused path when possible (diagonal + block-diagonal)
            # =============================================================
            _cached_bep = self._build_block_exp_pairs(
                phi_current, omega_current, B, N,
                mu_current.device, mu_current.dtype,
            )

            # Use fused path for diagonal + block-diagonal.
            # Fused path builds Omega from block exp pairs (flat only).
            # When non-flat cached_transport is available, fall back to non-fused.
            _has_nonflat_ct = (cached_transport is not None
                               and 'Omega' in cached_transport
                               and connection_delta is not None)
            _use_fused_single = (is_diagonal and self.irrep_dims is not None
                                 and not self.exact_diagonal_transport
                                 and not _has_nonflat_ct)

            # Detach beliefs for gradient computation (consistent with multihead
            # path at line 3667). Without this, analytical VFE gradients
            # participate in autograd, giving the single-β path a different
            # backward (I - lr*J) vs multihead's straight-through (I).
            _mu_for_grad = mu_current.detach() if _detach_e_step else mu_current
            _sigma_for_grad = sigma_current.detach() if _detach_e_step else sigma_current

            if _use_fused_single:
                beta_current, grad_mu, grad_sigma, _ = _fused_attention_and_vfe_gradients_block_diag(
                    mu_q=_mu_for_grad, sigma_q=_sigma_for_grad,
                    mu_p=mu_p_current, sigma_p=sigma_p,
                    phi=phi_current, generators=self.generators,
                    alpha=alpha_effective, lambda_belief=self.lambda_belief, lambda_softmax=self.lambda_softmax,
                    kappa=self.kappa, eps=eps,
                    irrep_dims=self.irrep_dims,
                    compute_sigma_align_grad=self.compute_sigma_align_grad,
                    enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                    alpha_c0=_alpha_c0,
                    cached_block_exp_pairs=_cached_bep,
                    mask=mask,
                    mask_self_attention=self.mask_self_attention,
                    use_rope=self._use_rope_vfe,
                    rope_base=self._rope_base_vfe,
                )
            else:
                # Fallback: separate attention + gradient
                beta_current = compute_attention_weights(
                    mu_q=_mu_for_grad, sigma_q=_sigma_for_grad,
                    phi=phi_current, generators=self.generators,
                    kappa=self.kappa, epsilon=eps, mask=mask,
                    return_kl=False,
                    diagonal_covariance=is_diagonal,
                    cached_transport=cached_transport,
                    irrep_dims=self.irrep_dims,
                    mask_self_attention=self.mask_self_attention,
                    gauge_mode=self.gauge_mode,
                    cached_block_exp_pairs=_cached_bep,
                    use_rope=self._use_rope_vfe,
                    rope_base=self._rope_base_vfe,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                )
                grad_mu, grad_sigma = compute_vfe_gradients_gpu(
                    mu_q=_mu_for_grad, sigma_q=_sigma_for_grad,
                    mu_p=mu_p_current, sigma_p=sigma_p,
                    beta=beta_current, phi=phi_current,
                    generators=self.generators, alpha=alpha_effective,
                    lambda_belief=self.lambda_belief, lambda_softmax=self.lambda_softmax, kappa=self.kappa,
                    eps=eps, alpha_c0=_alpha_c0,
                    cached_transport=cached_transport,
                    compute_sigma_align_grad=self.compute_sigma_align_grad,
                    irrep_dims=self.irrep_dims,
                    cached_block_exp_pairs=_cached_bep,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                    use_rope=self._use_rope_vfe,
                    rope_base=self._rope_base_vfe,
                )

            if return_beta_history:
                beta_history_entry = beta_current.detach().clone()

        # Add FRESH observation gradient (recomputed from current beliefs)
        # Use .detach() on mu_current to avoid second-order gradients through the
        # observation gradient computation. Gradients still flow through VFE dynamics
        # (the natural gradient update), just not through how the obs grad was computed.
        # This is more stable than full gradient flow while still allowing embeddings
        # to learn from VFE dynamics via the mu_current → mu_new update chain.
        if has_observations:
            logits = torch.matmul(mu_current.detach(), W_out.T)
            probs = F.softmax(logits, dim=-1)
            # Use pre-computed targets/mask/one_hot from _obs_cache (avoids
            # redundant targets.clone(), F.one_hot, and mask creation per iter)
            mask_obs = _obs_cache['mask_obs']
            one_hot = _obs_cache['one_hot']
            grad_error = (probs - one_hot) * mask_obs
            discrete_obs_grad = torch.matmul(grad_error, W_out)
            # Scale observation gradient by obs_weight (for warmup ramp)
            _obs_weight = getattr(self, '_obs_weight', 1.0)
            if _obs_weight < 1.0:
                discrete_obs_grad = discrete_obs_grad * _obs_weight
            # Debug: observation mu gradient
            if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
                _vfe_utils_mod._VFE_GRAD_DEBUG['obs_mu_grad'] = _grad_norm(discrete_obs_grad)

            grad_mu = grad_mu + discrete_obs_grad

            # Observation gradient for sigma (exact via Stein's lemma):
            #
            #   ∂/∂σ_k E_q[CE(z)] = (1/2) · E_q[∂²CE/∂z_k²]
            #
            # This is EXACT for any smooth loss, not a Taylor approximation.
            # For CE with softmax: ∂²CE/∂z_k² = Var_p[W[:,k]] ≥ 0.
            # We approximate E_q[H_kk(z)] ≈ H_kk(μ) (zeroth-order in σ).
            if self.obs_sigma_gradient:
                W_out_sq = _obs_cache['W_out_sq']                    # (V, K)
                EW2 = torch.matmul(probs, W_out_sq)                  # (B, N, K)
                EW  = torch.matmul(probs, W_out)                     # (B, N, K)
                hessian_diag = EW2 - EW ** 2                         # (B, N, K)
                # Clamp: FP rounding can violate Var ≥ 0
                _neg_mask = hessian_diag < 0
                if _neg_mask.any():
                    if _is_final_iter:
                        _nr("obs_sigma_hessian_neg_clamp", count=int(_neg_mask.sum().item()))
                    hessian_diag = hessian_diag.clamp(min=0.0)
                _sigma_obs_scale = (0.5 * self.obs_sigma_weight) * _obs_weight
                # Cap observation sigma gradient to prevent systematic upward bias
                # from dominating. Var_p[W[:,k]] >= 0 always, so this term only
                # pushes sigma upward; cap prevents runaway growth.
                obs_sigma_grad = (_sigma_obs_scale * hessian_diag * mask_obs).clamp(max=10.0)
                # Debug: observation sigma gradient (before diag_embed)
                if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
                    _vfe_utils_mod._VFE_GRAD_DEBUG['obs_sigma_grad'] = _grad_norm(obs_sigma_grad)
                # For full covariance, obs gradient is diagonal-only: embed on diagonal
                # to avoid broadcasting (B, N, K) into every row of (B, N, K, K).
                if not is_diagonal:
                    obs_sigma_grad = torch.diag_embed(obs_sigma_grad)
                grad_sigma = grad_sigma + obs_sigma_grad

        # =====================================================================
        # Active inference / EFE gradient (pragmatic + epistemic via PriorBank)
        # =====================================================================
        # Gated on the master toggle ffn._ai_enabled (set from
        # BlockConfig.active_inference).  When enabled and a readout path
        # is available, compute the EFE Euclidean gradient and STASH it on
        # self._ai_pending_grad_mu for use by the retraction step below.
        # We do NOT add it to grad_mu here, because:
        #   1. The main VFE grad_mu passes through a whitened trust region
        #      (||δμ/√σ|| ≤ 2) that is frequently saturated by the VFE
        #      terms alone.  Adding EFE to the sum means EFE gets clipped
        #      proportionally and its effective contribution collapses to
        #      a rounding-level perturbation (see session diagnostic).
        #   2. The VFE trust region is calibrated for KL-geometry terms;
        #      the EFE pragmatic+epistemic terms are not KLs and deserve
        #      their own budget.
        # Instead, we apply EFE as a SEPARATE whitened-trust-region update
        # after the VFE step below.  This gives EFE a fixed contribution
        # independent of how close the VFE gradient is to its trust region.
        #
        # Cost: O(B·N·V·K) per iteration for the pragmatic term (single
        # PriorBank.decode call) and O(S·B·N·V·K) for epistemic with S
        # MC samples.  Default S=4.  Master toggle off → zero impact.
        _ai_enabled = getattr(self, '_ai_enabled', False)
        _ai_prag = getattr(self, '_ai_pragmatic_weight', 0.0)
        _ai_epi  = getattr(self, '_ai_epistemic_weight', 0.0)
        _ai_bank = self.__dict__.get('_prior_bank_ref', None)
        _ai_wout_ref = self.__dict__.get('_ai_w_out_ref', None)
        _ai_pending_grad_mu = None  # separate EFE grad, applied after VFE step
        if _ai_enabled and (_ai_prag > 0.0 or _ai_epi > 0.0):
            _ai_samples = getattr(self, '_ai_epistemic_samples', 4)
            _ai_tau = getattr(self, '_ai_decode_tau', 1.0)
            # Unwrap list-wrapped refs (used to bypass nn.Module registration)
            _bank = _ai_bank[0] if isinstance(_ai_bank, list) else _ai_bank
            # Prefer explicit W_out arg from the current forward pass, fall
            # back to the stored out_proj reference if the caller didn't pass
            # W_out through (happens in eval/decode paths).
            if W_out is not None:
                _w_out_tensor = W_out
            elif _ai_wout_ref is not None:
                _proj = _ai_wout_ref[0] if isinstance(_ai_wout_ref, list) else _ai_wout_ref
                _w_out_tensor = _proj.weight  # (V, K)
            else:
                _w_out_tensor = None
            if _bank is not None or _w_out_tensor is not None:
                _ai_pending_grad_mu = _compute_active_inference_gradient(
                    mu_current=mu_current,
                    sigma_current=sigma_current,
                    prior_bank=_bank,
                    pragmatic_weight=_ai_prag,
                    epistemic_weight=_ai_epi,
                    epistemic_samples=_ai_samples,
                    decode_tau=_ai_tau,
                    w_out=_w_out_tensor,
                )
                if _ai_pending_grad_mu is not None and _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
                    _vfe_utils_mod._VFE_GRAD_DEBUG['ai_efe_mu_grad'] = _grad_norm(_ai_pending_grad_mu)

        # Debug: Euclidean totals (after obs, before clip)
        if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
            _vfe_utils_mod._VFE_GRAD_DEBUG['euclidean_mu_total'] = _grad_norm(grad_mu)
            _vfe_utils_mod._VFE_GRAD_DEBUG['euclidean_sigma_total'] = _grad_norm(grad_sigma)
            _ps = _per_pos_stats(grad_mu)
            _vfe_utils_mod._VFE_GRAD_DEBUG['euclidean_mu_pos_mean'] = _ps[0]
            _vfe_utils_mod._VFE_GRAD_DEBUG['euclidean_mu_pos_max'] = _ps[1]
            _ps = _per_pos_stats(grad_sigma)
            _vfe_utils_mod._VFE_GRAD_DEBUG['euclidean_sigma_pos_mean'] = _ps[0]
            _vfe_utils_mod._VFE_GRAD_DEBUG['euclidean_sigma_pos_max'] = _ps[1]

        # Clip for stability
        grad_mu = torch.clamp(grad_mu, min=-1e3, max=1e3)
        grad_sigma = torch.clamp(grad_sigma, min=-1e3, max=1e3)

        # =================================================================
        # Isotropic gradient projection: average grad_sigma across dims
        # =================================================================
        # When isotropic, all dims share one scalar σ². Average the per-dim
        # gradients so the natural gradient and retraction operate on the
        # consensus direction, rather than K independent updates collapsed
        # after the fact. This is the correct constrained gradient:
        #   ∂F/∂(σ²) = (1/K) Σ_k ∂F/∂σ_k²
        if self.isotropic_covariance:
            if is_diagonal:
                grad_sigma = grad_sigma.mean(dim=-1, keepdim=True).expand_as(grad_sigma)
            else:
                diag_grad = torch.diagonal(grad_sigma, dim1=-2, dim2=-1)
                avg_grad = diag_grad.mean(dim=-1, keepdim=True)
                K = grad_sigma.shape[-1]
                grad_sigma = avg_grad.unsqueeze(-1) * torch.eye(
                    K, device=grad_sigma.device, dtype=grad_sigma.dtype
                )

        # =================================================================
        # STEP 3: Natural gradient projection
        # =================================================================
        nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
            grad_mu, grad_sigma, sigma_current, eps=eps,

        )

        # Clamp natural gradient norm to prevent oscillatory divergence
        # in deeper layers where Sigma eigenvalues amplify gradients
        nat_grad_mu_norm = torch.linalg.norm(nat_grad_mu, dim=-1, keepdim=True)
        max_nat_grad_norm = 500.0
        nat_grad_scale = torch.clamp(
            max_nat_grad_norm / (nat_grad_mu_norm + eps), max=1.0
        )
        nat_grad_mu = nat_grad_mu * nat_grad_scale

        # Clamp nat_grad_sigma norm (analogous to nat_grad_mu clipping above).
        # The natural gradient nat_grad_sigma = 2σ²·grad_sigma squares the
        # covariance, amplifying gradients when sigma is large. Without clipping,
        # the backward pass sees unclipped gradient magnitudes even though the
        # forward retraction trust region clips the whitened step.
        if is_diagonal:
            nat_grad_sigma_norm = torch.linalg.norm(nat_grad_sigma, dim=-1, keepdim=True)
        else:
            nat_grad_sigma_norm = torch.linalg.norm(
                nat_grad_sigma.flatten(-2), dim=-1, keepdim=True
            ).unsqueeze(-1)
        max_nat_grad_sigma_norm = 500.0
        nat_grad_sigma_scale = torch.clamp(
            max_nat_grad_sigma_norm / (nat_grad_sigma_norm + eps), max=1.0
        )
        nat_grad_sigma = nat_grad_sigma * nat_grad_sigma_scale

        # Store E-step gradient norms — defer .item() to final iteration only
        # to avoid CUDA synchronization stalls on every VFE iteration.
        if _is_final_iter:
            _raw_nat_grad_norm = nat_grad_mu.detach().norm().item()
            _raw_nat_grad_sigma_norm = nat_grad_sigma.detach().norm().item()
            self._e_step_grad_norms['nat_grad_mu'] = _raw_nat_grad_norm
            self._e_step_grad_norms['nat_grad_sigma'] = _raw_nat_grad_sigma_norm
            self._e_step_grad_norms['nat_grad_mu_clipped'] = nat_grad_mu.detach().norm().item()
            self._e_step_grad_norms['nat_grad_sigma_clipped'] = nat_grad_sigma.detach().norm().item()
            self._e_step_grad_norms['mu_cap_frac'] = (
                nat_grad_mu_norm.squeeze(-1) >= max_nat_grad_norm * 0.99
            ).float().mean().item()
            if is_diagonal:
                self._e_step_grad_norms['sigma_cap_frac'] = (
                    nat_grad_sigma_norm.squeeze(-1) >= max_nat_grad_sigma_norm * 0.99
                ).float().mean().item()
            else:
                self._e_step_grad_norms['sigma_cap_frac'] = (
                    nat_grad_sigma_norm.squeeze(-1).squeeze(-1) >= max_nat_grad_sigma_norm * 0.99
                ).float().mean().item()

        # =================================================================
        # DEBUG: Print per-component gradient breakdown
        # =================================================================
        if _vfe_utils_mod._VFE_GRAD_DEBUG is not None and self._debug_vfe_gradients:
            d = _vfe_utils_mod._VFE_GRAD_DEBUG

            # Detect multihead mode: keys have 'headN(d=M)/' prefix
            _is_multihead = any('/' in k for k in d)

            # Euclidean totals computed on the full (already assembled) grad tensors
            _eu_mu = _grad_norm(grad_mu)
            _eu_sig = _grad_norm(grad_sigma)
            _ps_mu = _per_pos_stats(grad_mu)
            _ps_sig = _per_pos_stats(grad_sigma)

            # Nat_grad amplification factors (compute .item() here since debug is active)
            _raw_ng_mu = nat_grad_mu.detach().norm().item()
            _raw_ng_sig = nat_grad_sigma.detach().norm().item()
            _amp_mu = _raw_ng_mu / max(_eu_mu, 1e-12)
            _amp_sig = _raw_ng_sig / max(_eu_sig, 1e-12)
            # Fraction of positions hitting the 500 clip
            _mu_clip_frac = (nat_grad_mu_norm.squeeze(-1) >= max_nat_grad_norm * 0.99).float().mean().item()
            if is_diagonal:
                _sig_clip_frac = (nat_grad_sigma_norm.squeeze(-1) >= max_nat_grad_sigma_norm * 0.99).float().mean().item()
            else:
                _sig_clip_frac = (nat_grad_sigma_norm.squeeze(-1).squeeze(-1) >= max_nat_grad_sigma_norm * 0.99).float().mean().item()

            logger.debug(f"\n{'='*80}")
            logger.debug(
                f"  [VFE GRAD DEBUG] iter {iteration}/{self.n_iterations}"
                f"  diag={is_diagonal}  K={mu_current.shape[-1]}"
                f"  B×N={mu_current.shape[0]}×{mu_current.shape[1]}"
                f"  multihead={_is_multihead}"
            )
            logger.debug(f"{'='*80}")

            if _is_multihead:
                # Extract unique head prefixes
                _heads = sorted(set(k.split('/')[0] for k in d if '/' in k),
                                key=lambda x: int(x.split('head')[1].split('(')[0]))
                logger.debug(f"  --- Per-head breakdown ({len(_heads)} heads) ---")
                logger.debug(
                    f"  {'Head':<16} {'s_self':>8} {'s_align':>8} {'s_smx':>8}"
                    f" {'mu_self':>8} {'mu_dir':>8} {'mu_smx':>8}"
                    f" {'KL_avg':>8} {'KL_max':>8} {'sp_min':>8} {'sq_max':>8}"
                )
                logger.debug(
                    f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8}"
                    f" {'-'*8} {'-'*8} {'-'*8}"
                    f" {'-'*8} {'-'*8} {'-'*8} {'-'*8}"
                )
                for hp in _heads:
                    def _hget(key, default=0):
                        return d.get(f'{hp}/{key}', default)
                    logger.debug(
                        f"  {hp:<16}"
                        f" {_hget('grad_sigma_self'):>8.1f}"
                        f" {_hget('grad_sigma_align_direct'):>8.1f}"
                        f" {_hget('grad_sigma_softmax'):>8.1f}"
                        f" {_hget('grad_mu_self'):>8.1f}"
                        f" {_hget('grad_mu_direct'):>8.1f}"
                        f" {_hget('grad_mu_softmax'):>8.1f}"
                        f" {_hget('kl_pairwise_mean'):>8.2f}"
                        f" {_hget('kl_pairwise_max'):>8.1f}"
                        f" {_hget('sigma_p_min'):>8.4f}"
                        f" {_hget('sigma_q_eig_max'):>8.4f}"
                    )
            else:
                # Single-beta mode
                logger.debug("  --- Covariance state ---")
                logger.debug(
                    f"  sigma_p  range:  [{d.get('sigma_p_min', 0):.4f}, {d.get('sigma_p_max', 0):.4f}]"
                    f"  ->  1/sigma_p range: [{1/max(d.get('sigma_p_max', 1), 1e-12):.2f},"
                    f" {1/max(d.get('sigma_p_min', 1e-12), 1e-12):.2f}]"
                )
                logger.debug(f"  sigma_q  eig range: [{d.get('sigma_q_eig_min', 0):.4f}, {d.get('sigma_q_eig_max', 0):.4f}]")
                logger.debug("  --- Euclidean gradient components (global norms) ---")
                logger.debug(
                    f"  {'Component':<30} {'mu':>12} {'sigma':>12} {'s pos_mean':>12} {'s pos_max':>12}"
                )
                logger.debug(
                    f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*12}"
                )
                logger.debug(
                    f"  {'self-coupling (a*dKL/dtheta)':<30}"
                    f" {d.get('grad_mu_self', 0):>12.1f}"
                    f" {d.get('grad_sigma_self', 0):>12.1f}"
                    f" {d.get('grad_sigma_self_pos_mean', 0):>12.2f}"
                    f" {d.get('grad_sigma_self_pos_max', 0):>12.2f}"
                )
                logger.debug(
                    f"  {'align direct (l*beta*dKL/dtheta)':<30}"
                    f" {d.get('grad_mu_direct', 0):>12.1f}"
                    f" {d.get('grad_sigma_align_direct', 0):>12.1f}"
                    f" {d.get('grad_sigma_align_pos_mean', 0):>12.2f}"
                    f" {d.get('grad_sigma_align_pos_max', 0):>12.2f}"
                )
                logger.debug(
                    f"  {'softmax (KL*dbeta/dtheta)':<30}"
                    f" {d.get('grad_mu_softmax', 0):>12.1f}"
                    f" {d.get('grad_sigma_softmax', 0):>12.1f}"
                    f" {d.get('grad_sigma_softmax_pos_mean', 0):>12.2f}"
                    f" {d.get('grad_sigma_softmax_pos_max', 0):>12.2f}"
                )

            # Observation (shared between multihead and single-beta, computed on full tensor)
            if 'obs_mu_grad' in d:
                logger.debug(
                    f"  {'observation (CE)':<30}"
                    f" {d.get('obs_mu_grad', 0):>12.1f}"
                    f" {d.get('obs_sigma_grad', 0):>12.1f}"
                )

            logger.debug("  --- Euclidean total (assembled, after obs) ---")
            logger.debug(f"  grad_mu:    {_eu_mu:>10.1f}  (pos mean: {_ps_mu[0]:.2f}, max: {_ps_mu[1]:.2f})")
            logger.debug(f"  grad_sigma: {_eu_sig:>10.1f}  (pos mean: {_ps_sig[0]:.2f}, max: {_ps_sig[1]:.2f})")
            logger.debug("  --- Natural gradient (Fisher projection) ---")
            logger.debug(
                f"  nat_grad_mu:    {_raw_nat_grad_norm:>10.1f}  (amplification: {_amp_mu:.2f}x)"
                f"  clip: {self._e_step_grad_norms['nat_grad_mu_clipped']:.1f}"
                f"  ({_mu_clip_frac*100:.0f}% positions at cap)"
            )
            logger.debug(
                f"  nat_grad_sigma: {_raw_nat_grad_sigma_norm:>10.1f}  (amplification: {_amp_sig:.2f}x)"
                f"  clip: {self._e_step_grad_norms['nat_grad_sigma_clipped']:.1f}"
                f"  ({_sig_clip_frac*100:.0f}% positions at cap)"
            )
            logger.debug(f"{'='*80}")
            # Store before resetting (debug mode may coexist with metrics collection)
            if self._collect_vfe_metrics:
                _aggregate_multihead_vfe_debug(_vfe_utils_mod._VFE_GRAD_DEBUG, self.irrep_dims)
                self.last_vfe_debug = dict(_vfe_utils_mod._VFE_GRAD_DEBUG)
            _vfe_utils_mod._VFE_GRAD_DEBUG = None  # Reset for next iteration

        # Store lightweight copy for external consumption (no printing overhead)
        elif self._collect_vfe_metrics and _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
            _aggregate_multihead_vfe_debug(_vfe_utils_mod._VFE_GRAD_DEBUG, self.irrep_dims)
            self.last_vfe_debug = dict(_vfe_utils_mod._VFE_GRAD_DEBUG)
            _vfe_utils_mod._VFE_GRAD_DEBUG = None

        # =================================================================
        # STEP 4: Update beliefs (E-step) with WHITENED trust region
        # =================================================================
        # The natural gradient nat_grad_mu = Σ @ grad scales with σ
        # Use whitened trust region: ||δμ / √σ|| instead of raw norm
        delta_mu = -effective_lr * nat_grad_mu

        # Whitened trust region for mu (float32 for sqrt/division stability under AMP)
        if is_diagonal:
            sigma_sqrt = torch.sqrt(sigma_current.float().clamp(min=eps)).to(sigma_current.dtype)
            whitened_delta = delta_mu / sigma_sqrt
        else:
            # Use .clone() after diagonal to avoid view-related gradient issues
            sigma_diag = torch.diagonal(sigma_current, dim1=-2, dim2=-1).clone().float().clamp(min=eps)
            whitened_delta = delta_mu / torch.sqrt(sigma_diag).to(delta_mu.dtype)

        whitened_norm = torch.linalg.norm(whitened_delta, dim=-1, keepdim=True)
        mu_trust_region = 2.0  # Trust region on whitened norm
        scale = torch.clamp(mu_trust_region / (whitened_norm + eps), max=1.0)
        mu_current = mu_current + scale * delta_mu

        # Track trust region clip fraction (final iteration only to avoid CUDA sync)
        if _is_final_iter:
            self._e_step_grad_norms['mu_trust_frac'] = (
                scale.squeeze(-1) < 0.99
            ).float().mean().item()
            self._e_step_grad_norms['whitened_mu_mean'] = whitened_norm.mean().item()
            self._e_step_grad_norms['whitened_mu_max'] = whitened_norm.max().item()

        # =================================================================
        # Active inference / EFE μ-update (separate budget, Euclidean)
        # =================================================================
        # Applied AFTER the VFE whitened trust region so that the EFE
        # contribution is not diluted when the VFE gradient saturates its
        # own trust region.  Design choices:
        #
        # 1. EUCLIDEAN gradient, NOT natural gradient.  The VFE terms are
        #    KL divergences with a Fisher metric = 1/σ², so their natural
        #    gradient multiplies by σ² (Fisher^{-1}).  The EFE pragmatic
        #    and epistemic terms are entropies / mutual informations of a
        #    softmax readout, which do not have a Fisher-metric
        #    justification w.r.t. μ.  We use ∇_μ F_AI directly.
        #
        # 2. SEPARATE step size (_ai_lr, default 1.0, vs VFE's
        #    E_mu_q_lr ~ 0.1).  The EFE gradient magnitudes are much
        #    smaller than the VFE coupling gradients because the softmax
        #    readout saturates the entropy gradient near uniform init.
        #    A larger step size is needed for EFE to move μ meaningfully
        #    in the same number of iterations.
        #
        # 3. SEPARATE whitened trust region (_ai_trust_region, default 0.5)
        #    so total update stays bounded: VFE budget (2.0) + EFE budget
        #    (0.5) in whitened norm.
        #
        # 4. Applied OUTSIDE the natural-gradient / trust-region chain for
        #    VFE, so VFE's step is not modified and the two updates
        #    compose additively.
        if _ai_pending_grad_mu is not None:
            _ai_lr = getattr(self, '_ai_lr', 1.0)
            delta_efe = -_ai_lr * _ai_pending_grad_mu  # Euclidean, no σ² scaling

            # Whitened trust region: ||δμ/√σ|| ≤ _ai_trust_region
            if is_diagonal:
                sigma_sqrt_efe = torch.sqrt(
                    sigma_current.float().clamp(min=eps)
                ).to(sigma_current.dtype)
            else:
                sigma_diag_efe = torch.diagonal(
                    sigma_current, dim1=-2, dim2=-1
                ).float().clamp(min=eps)
                sigma_sqrt_efe = torch.sqrt(sigma_diag_efe).to(sigma_current.dtype)

            whitened_efe = delta_efe / sigma_sqrt_efe
            whitened_efe_norm = torch.linalg.norm(whitened_efe, dim=-1, keepdim=True)
            efe_trust_region = getattr(self, '_ai_trust_region', 0.5)
            efe_scale = torch.clamp(
                efe_trust_region / (whitened_efe_norm + eps), max=1.0
            )
            mu_current = mu_current + efe_scale * delta_efe

            if _is_final_iter:
                self._e_step_grad_norms['ai_efe_raw_grad_norm'] = _ai_pending_grad_mu.detach().norm().item()
                self._e_step_grad_norms['ai_efe_whitened_mean'] = whitened_efe_norm.mean().item()
                self._e_step_grad_norms['ai_efe_whitened_max'] = whitened_efe_norm.max().item()
                self._e_step_grad_norms['ai_efe_trust_frac'] = (
                    efe_scale.squeeze(-1) < 0.99
                ).float().mean().item()

        if self.update_sigma:
            # SPD-preserving retraction: sigma_new = sigma * exp(step * clip(delta/sigma, -trust, trust))
            # step_size=1.0 so trust_region alone controls max relative change.
            # nat_grad_sigma = 2σ²·grad → whitened = -2σ·grad, clipped by trust.
            # With effective_lr≈0.1: max_exp = 0.001 → ~0.1% per iter, ~1% over 10 iters.
            # Calibrated between frozen (pre-#768: 0.025%/iter) and overcorrected (0.5%/iter).
            sigma_trust_base = self._get_sigma_trust(effective_lr)
            sigma_trust_diag = sigma_trust_base
            sigma_trust_full = sigma_trust_base * 0.5  # Full cov more sensitive
            if is_diagonal:
                sigma_current = retract_spd_diagonal_torch(
                    sigma_diag=sigma_current,
                    delta_sigma=-nat_grad_sigma,
                    step_size=1.0,
                    trust_region=sigma_trust_diag,
                    eps=eps,
                    sigma_max=self.sigma_max,
                )
            else:
                sigma_current = retract_spd_torch(
                    Sigma=sigma_current,
                    delta_Sigma=-nat_grad_sigma,
                    step_size=1.0,
                    trust_region=sigma_trust_full,
                    eps=eps,
                    sigma_max=self.sigma_max,
                )

        # =============================================================
        # STEP 4b2: Sigma condition clamping
        # =============================================================
        # Prevent sigma anisotropy from growing unbounded. When
        # sigma_max / sigma_min > max_condition, clamp outlier
        # dimensions toward the geometric mean. This keeps the
        # natural gradient well-conditioned without forcing full
        # isotropy.
        if self.update_sigma:
            max_condition = 10.0
            if is_diagonal:
                # sigma_current: (B, N, K)
                # Apply unconditionally — torch.where handles the no-clamp case
                # without a CUDA-syncing .any() check.
                s_min = sigma_current.min(dim=-1, keepdim=True).values.clamp(min=eps)
                s_max = sigma_current.max(dim=-1, keepdim=True).values
                condition = s_max / s_min  # (B, N, 1)
                needs_clamp = condition > max_condition
                # Geometric mean preserves det(Sigma) = product of eigenvalues
                geo_mean = sigma_current.log().mean(dim=-1, keepdim=True).exp()
                lower = geo_mean / (max_condition ** 0.5)
                upper = geo_mean * (max_condition ** 0.5)
                sigma_clamped = sigma_current.clamp(min=lower, max=upper)
                sigma_current = torch.where(
                    needs_clamp.expand_as(sigma_current),
                    sigma_clamped,
                    sigma_current,
                )
            else:
                # Full covariance: clamp eigenvalue ratio
                try:
                    eigvals = torch.linalg.eigvalsh(sigma_current)  # (B, N, K)
                except (RuntimeError, torch.linalg.LinAlgError):
                    eigvals = None
                if eigvals is not None:
                    e_min = eigvals[..., 0:1].clamp(min=eps)
                    e_max = eigvals[..., -1:]
                    condition = e_max / e_min
                    # Apply unconditionally — ridge is 0 when condition is fine
                    geo_mean = eigvals.log().mean(dim=-1, keepdim=True).exp()
                    lower = geo_mean / (max_condition ** 0.5)
                    # Regularize toward isotropic: Sigma → Sigma + ridge * I
                    # ridge = 0 when all eigenvalues already satisfy condition
                    ridge = (lower - e_min).clamp(min=0.0).mean(dim=-1, keepdim=True)
                    K = sigma_current.shape[-1]
                    sigma_current = sigma_current + ridge.unsqueeze(-1) * torch.eye(
                        K, device=sigma_current.device, dtype=sigma_current.dtype
                    )

        # =============================================================
        # STEP 4c: Isotropic covariance enforcement (Limit 1)
        # =============================================================
        # After sigma update, collapse per-dimension variances to scalar σ²I.
        # This maintains the isotropic constraint through VFE dynamics.
        if self.update_sigma and self.isotropic_covariance:
            if is_diagonal:
                # sigma_current: (B, N, K) → average across K, expand back
                scalar_var = sigma_current.mean(dim=-1, keepdim=True)
                sigma_current = scalar_var.expand_as(sigma_current)
            else:
                # sigma_current: (B, N, K, K) → extract diag, average, rebuild σ²I
                diag_vals = torch.diagonal(sigma_current, dim1=-2, dim2=-1)
                scalar_var = diag_vals.mean(dim=-1, keepdim=True)  # (B, N, 1)
                K = sigma_current.shape[-1]
                sigma_current = scalar_var.unsqueeze(-1) * torch.eye(
                    K, device=sigma_current.device, dtype=sigma_current.dtype
                )

        # =============================================================
        # DIAGNOSTIC: Per-iteration convergence data
        # =============================================================
        if self._collect_iteration_diagnostics:
            # Sigma condition number for diagnostics
            if is_diagonal:
                _s_det = sigma_current.detach()
                _s_cond = (_s_det.max(dim=-1).values / _s_det.min(dim=-1).values.clamp(min=eps)).mean().item()
            else:
                _s_diag_det = torch.diagonal(sigma_current.detach(), dim1=-2, dim2=-1)
                _s_cond = (_s_diag_det.max(dim=-1).values / _s_diag_det.min(dim=-1).values.clamp(min=eps)).mean().item()
            _diag = {
                'iteration': iteration,
                'grad_mu_norm': grad_mu.detach().norm().item(),
                'grad_sigma_norm': grad_sigma.detach().norm().item(),
                'nat_grad_mu_norm': nat_grad_mu.detach().norm().item(),
                'nat_grad_mu_raw_norm': self._e_step_grad_norms.get('nat_grad_mu', 0.0),
                'nat_grad_sigma_norm': nat_grad_sigma.detach().norm().item(),
                'nat_grad_sigma_raw_norm': self._e_step_grad_norms.get('nat_grad_sigma', 0.0),
                'nat_grad_sigma_max': nat_grad_sigma.detach().abs().max().item(),
                'delta_mu_norm': delta_mu.detach().norm().item(),
                'mu_norm': mu_current.detach().norm().item(),
                'sigma_mean': sigma_current.detach().mean().item(),
                'sigma_max': sigma_current.detach().max().item(),
                'sigma_min': sigma_current.detach().min().item(),
                'sigma_std': sigma_current.detach().std().item(),
                'sigma_condition': _s_cond,
                'effective_lr': effective_lr.detach().item() if isinstance(effective_lr, torch.Tensor) else float(effective_lr),
                'scale_mean': scale.detach().mean().item(),
            }
            if mu_p_current is not None:
                _diag['mu_diff_to_prior_norm'] = (mu_current - mu_p_current).detach().norm().item()
            # Beta entropy from last computed beta
            try:
                if self.multihead_vfe and beta_heads:
                    _b_diag = beta_heads[-1].detach().clamp(min=1e-10)
                elif 'beta_current' in locals() and beta_current is not None:
                    _b_diag = beta_current.detach().clamp(min=1e-10)
                else:
                    _b_diag = None
                if _b_diag is not None:
                    _diag['beta_entropy'] = -(_b_diag * _b_diag.log()).sum(dim=-1).mean().item()
            except Exception:
                pass
            # Relative belief change from previous iteration
            if iteration == 0:
                self._diag_prev_mu = mu_current.detach().clone()
            else:
                _diag['mu_change_rel'] = (
                    (mu_current - self._diag_prev_mu).detach().norm().item()
                    / (mu_current.detach().norm().item() + 1e-8)
                )
                self._diag_prev_mu = mu_current.detach().clone()
            self._iteration_diagnostics.append(_diag)

        # =============================================================
        # STEP 4b: Optional Gauge Frame Evolution DURING E-step
        # =============================================================
        _skip_phi_update = self.gauge_mode in ('trivial', 'constant')
        _use_omega = omega_current is not None and getattr(self, 'gauge_param', 'phi') == 'omega'

        if (self.update_phi_per_iteration and torch.is_grad_enabled()
                and not _skip_phi_update):

            if _use_omega:
                # Direct Omega path: no matrix_exp, no dexp series
                grad_omega = self._compute_omega_grad_direct(
                    omega_current, mu_current, sigma_current,
                    is_diagonal, mask, eps,
                )
                if grad_omega is not None:
                    self._e_step_grad_norms['grad_phi'] = grad_omega.detach().norm().item()
                    omega_current = self._retract_omega(
                        omega_current, grad_omega, self.phi_lr,
                        trust_region=getattr(self, 'omega_trust_region', 0.3),
                    )
            else:
                # Phi path (existing): matrix_exp + dexp series
                _phi_bep = _mh_cached_bep if self.multihead_vfe else _cached_bep
                grad_phi = self._compute_phi_grad(
                    phi_current, mu_current, sigma_current,
                    is_diagonal, mask, eps,
                    cached_block_exp_pairs=_phi_bep,
                )
                if grad_phi is not None:
                    self._e_step_grad_norms['grad_phi'] = grad_phi.detach().norm().item()
                    phi_current = _retract_phi(
                        phi=phi_current,
                        delta_phi=-grad_phi,
                        generators=self.generators,
                        step_size=self.phi_lr,
                        max_norm=self.phi_max_norm,
                    )

        return (mu_current, sigma_current, phi_current, omega_current,
                beta_current, beta_heads, alpha_effective, beta_history_entry)

    def forward(
        self,
        mu: torch.Tensor,          # (B, N, K) - current beliefs
        beta: torch.Tensor = None, # (B, n_heads, N, N) - UNUSED, kept for API compat
        mu_prior: torch.Tensor = None,    # (B, N, K) - embedding priors
        phi: torch.Tensor = None,         # (B, N, phi_dim) - gauge frames
        sigma: Optional[torch.Tensor] = None,  # (B, N, K, K) or (B, N, K) if diagonal
        mask: Optional[torch.Tensor] = None,   # (B, N, N) - causal mask
        targets: Optional[torch.Tensor] = None,  # (B, N) - target token IDs
        W_out: Optional[torch.Tensor] = None,    # (V, K) - output projection
        token_ids: Optional[torch.Tensor] = None,  # (B, N) - token IDs for PriorBank lookup
        return_beta_history: bool = False,  # Return β evolution for analysis
        omega: Optional[torch.Tensor] = None,  # (B, N, K, K) direct group elements (gauge_param='omega')
        sigma_prior: Optional[torch.Tensor] = None,  # (B, N, K, K) or (B, N, K) - embedding prior covariance
        connection_delta: Optional[torch.Tensor] = None,  # (B, N, N, n_gen) frozen non-flat connection
        cocycle_relaxation: float = 0.0,  # Scale for connection_delta: 0=flat, 1=fully non-flat
        precomputed_block_exp_pairs: Optional[list] = None,  # Shared block exp pairs from blocks.py
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[list]]:
        """
        Dynamic VFE E-step descent with beta recomputation at each iteration.

        Flow at each iteration:
            1. beta = softmax(-KL(q||Omega[q])/kappa)  [recompute from current beliefs]
            2. dF/dmu = alpha*(mu-mu_p)/sigma_p + lambda*Sum_j beta*(dKL/dmu)
                        + Sum_j KL*(dBeta/dmu) + dCE/dmu
            3. mu <- mu - eta * F_inv * dF/dmu  [natural gradient descent]
            4. (Optional) sigma <- retract_spd(sigma, -eta * dF/dsigma)
            5. (Optional) phi <- retract(phi, -eta_phi * dF/dphi)

        When multihead_vfe=True, each irrep block gets its own beta_h.
        When use_prior_bank=True, priors come from PriorBank via token_ids.

        Args:
            mu: Current belief means (B, N, K).
            beta: UNUSED, kept for API compatibility.
            mu_prior: Prior means from embeddings (B, N, K).
            phi: Gauge frames (B, N, phi_dim) where phi_dim = n_gen.
            sigma: Belief covariances - (B, N, K, K) full or (B, N, K) diagonal.
                When diagonal_covariance=True, shape is (B, N, K).
            mask: Causal mask (B, N, N) where 0 = cannot attend.
            targets: Target token IDs for observation term (B, N).
            W_out: Output projection (V, K) for dCE/dmu computation.
            token_ids: Token IDs (B, N) for PriorBank lookup. Required when
                use_prior_bank=True.
            return_beta_history: If True, return list of beta at each step.
            sigma_prior: Embedding prior covariance (B, N, K, K) or (B, N, K).
                When provided, used as sigma_p in the E-step (proper prior
                reference). When None, falls back to sigma.detach() (legacy).
            connection_delta: Frozen non-flat connection δ_ij (B, N, N, n_gen).
                Computed once from initial μ by GaugeConnection in blocks.py.
                Treated as an E-step constant (like σ_p).
            cocycle_relaxation: Scale factor for connection_delta (0=flat, 1=non-flat).

        Returns:
            mu_new: Updated beliefs (B, N, K).
            sigma_new: Updated covariances (same shape as input) or None.
            phi_new: Updated gauge frames (B, N, phi_dim).
            beta_history: List of beta tensors if return_beta_history, else None.
        """
        B, N, K = mu.shape
        device = mu.device
        dtype = mu.dtype
        eps = 1e-6

        # =================================================================
        # SAFETY: Disable autocast if active. The VFE inner loop uses
        # analytical gradients with eigh, sqrt, log, exp, matrix inv —
        # all of which need float32. If caller has autocast enabled,
        # disable it and upcast inputs.
        # =================================================================
        _amp_active = torch.is_autocast_enabled('cuda')
        _amp_ctx = torch.amp.autocast('cuda', enabled=False) if _amp_active else None
        if _amp_ctx is not None:
            _amp_ctx.__enter__()
            mu = mu.float()
            if sigma is not None:
                sigma = sigma.float()
            mu_prior = mu_prior.float()
            phi = phi.float()
            if W_out is not None:
                W_out = W_out.float()

        # ── Prepare all E-step inputs ────────────────────────────────────
        _state = self._prepare_e_step_inputs(
            mu, sigma, mu_prior, phi, omega, sigma_prior,
            B, N, K, device, dtype, eps,
        )
        mu_current = _state['mu_current']
        sigma_current = _state['sigma_current']
        phi_current = _state['phi_current']
        omega_current = _state['omega_current']
        mu_p_current = _state['mu_p_current']
        sigma_p = _state['sigma_p']
        is_diagonal = _state['is_diagonal']
        alpha_effective = _state['alpha_effective']
        _alpha_c0 = _state['_alpha_c0']

        # Track β evolution if requested
        beta_history = [] if return_beta_history else None
        has_observations = targets is not None and W_out is not None
        _detach_e_step = True
        beta_current = None
        beta_heads = []

        # =====================================================================
        # CLOSED-FORM E-STEP: Precision-weighted fixed point (optional)
        # =====================================================================
        if self.closed_form_e_step:
            (mu_current, sigma_current, phi_current, omega_current,
             beta_heads, _cf_beta_history) = self._closed_form_e_step(
                mu_current=mu_current,
                sigma_current=sigma_current,
                phi_current=phi_current,
                omega_current=omega_current,
                mu_p_current=mu_p_current,
                sigma_p=sigma_p,
                alpha_effective=alpha_effective,
                _alpha_c0=_alpha_c0,
                is_diagonal=is_diagonal,
                B=B, N=N,
                device=device, dtype=dtype, eps=eps,
                mask=mask,
                return_beta_history=return_beta_history,
            )
            if return_beta_history and _cf_beta_history:
                beta_history = _cf_beta_history
            beta_current = beta_heads[-1] if beta_heads else None
        # =====================================================================
        # VFE Descent Loop with Dynamic β (runs outside AMP autocast)
        # =====================================================================
        # Skip when closed_form_e_step handled the E-step above.
        _n_iters = 0 if self.closed_form_e_step else self.n_iterations

        # Pre-compute observation constants (invariant across VFE iterations).
        # Avoids redundant targets.clone(), F.one_hot(V), and W_out**2 per iter.
        _obs_cache = None
        if has_observations and _n_iters > 0:
            _tv = targets.clone()
            _tv[_tv == -1] = 0
            _mask_obs = (targets != -1).unsqueeze(-1).float()
            _one_hot = F.one_hot(_tv, num_classes=W_out.shape[0]).float() * _mask_obs
            _obs_cache = {
                'mask_obs': _mask_obs,
                'one_hot': _one_hot,
            }
            if self.obs_sigma_gradient:
                _obs_cache['W_out_sq'] = W_out ** 2  # (V, K)

        # When phi is frozen across iterations, precompute block exp pairs
        # once to avoid redundant matrix exponentials (Finding 23: ~2-3x speedup
        # on matrix exp cost with 3 iterations).
        # precomputed_block_exp_pairs from blocks.py: shared with attention sublayer,
        # avoids redundant fused_block_matrix_exp_pairs (same phi, same generators).
        _hoisted_bep = precomputed_block_exp_pairs
        if _hoisted_bep is None and _n_iters > 1 and not self.update_phi_per_iteration and self.multihead_vfe:
            _hoisted_bep = self._build_block_exp_pairs(
                phi_current if not (getattr(self, 'amortized_inference', False) is False and self.detach_phi)
                else phi_current.detach(),
                omega_current, B, N,
                mu_current.device, mu_current.dtype,
            )

        # Gradient checkpointing (Finding 26): trade ~2x compute for ~3x
        # memory savings with 3 iterations. Only checkpoint non-final
        # iterations — the final iteration's activations are needed for the
        # backward pass regardless.
        _use_ckpt = (self.gradient_checkpoint_vfe and self.training
                     and _n_iters > 1 and torch.is_grad_enabled())

        for iteration in range(_n_iters):
            _iter_kwargs = dict(
                iteration=iteration,
                mu_current=mu_current,
                sigma_current=sigma_current,
                phi_current=phi_current,
                omega_current=omega_current,
                mu_p_current=mu_p_current,
                sigma_p=sigma_p,
                alpha_effective=alpha_effective,
                _alpha_c0=_alpha_c0,
                is_diagonal=is_diagonal,
                B=B, N=N, eps=eps,
                mask=mask,
                has_observations=has_observations,
                targets=targets,
                W_out=W_out,
                return_beta_history=return_beta_history,
                _detach_e_step=_detach_e_step,
                _obs_cache=_obs_cache,
                _precomputed_block_exp_pairs=_hoisted_bep,
                connection_delta=connection_delta,
                cocycle_relaxation=cocycle_relaxation,
            )
            _is_final = (iteration == _n_iters - 1)
            if _use_ckpt and not _is_final:
                (mu_current, sigma_current, phi_current, omega_current,
                 beta_current, beta_heads, alpha_effective,
                 _iter_beta) = torch.utils.checkpoint.checkpoint(
                    self._vfe_iteration,
                    use_reentrant=False,
                    **_iter_kwargs,
                )
            else:
                (mu_current, sigma_current, phi_current, omega_current,
                 beta_current, beta_heads, alpha_effective,
                 _iter_beta) = self._vfe_iteration(**_iter_kwargs)
            if return_beta_history and _iter_beta is not None:
                beta_history.append(_iter_beta)

        # =================================================================
        # DEQ implicit differentiation: replace straight-through backward
        # with Neumann-series approximation of (I - J)^{-1}
        # =================================================================
        if self.use_deq and self.training and torch.is_grad_enabled():
            if self.deq_include_phi and self.update_phi:
                # Joint (μ, Σ, φ) fixed-point: IFT corrects ALL three variables,
                # eliminating the straight-through bias in the M-step φ gradient.
                step_fn = self._make_deq_step_fn_with_phi(
                    mu_p_current, sigma_p,
                    mask, is_diagonal, eps, dtype,
                )
                mu_current, sigma_current, phi_current = DEQFixedPointFull.apply(
                    mu_current, sigma_current, phi_current, step_fn,
                    self.n_iterations, self.deq_neumann_terms,
                )
            else:
                # Original (μ, Σ)-only fixed point; φ gets straight-through gradient.
                step_fn = self._make_deq_step_fn(
                    phi_current, mu_p_current, sigma_p,
                    mask, is_diagonal, eps, dtype,
                )
                mu_current, sigma_current = DEQFixedPoint.apply(
                    mu_current, sigma_current, step_fn,
                    self.n_iterations, self.deq_neumann_terms,
                )

        # =================================================================
        # STEP 5: Optional Gauge Frame Evolution via VFE Gradient (after loop)
        # =================================================================
        # Skip when closed_form_e_step already handled phi evolution above.
        _use_omega = omega_current is not None and getattr(self, 'gauge_param', 'phi') == 'omega'
        _skip_phi_post = self.closed_form_e_step and is_diagonal  # Already done in closed-form path
        if (self.update_phi and not self.update_phi_per_iteration
                and torch.is_grad_enabled()
                and self.gauge_mode not in ('trivial', 'constant')
                and not _skip_phi_post):

            if _use_omega:
                # Direct Omega path
                grad_omega = self._compute_omega_grad_direct(
                    omega_current, mu_current, sigma_current,
                    is_diagonal, mask, eps,
                )
                if grad_omega is not None:
                    omega_current = self._retract_omega(
                        omega_current, grad_omega, self.phi_lr,
                        trust_region=getattr(self, 'omega_trust_region', 0.3),
                    )
            else:
                # Phi path (existing)
                _phi_bep_post = None
                if self.irrep_dims is not None and self.gauge_mode == 'learned':
                    _phi_bep_post = fused_block_matrix_exp_pairs(
                        phi_current, self.generators, self.irrep_dims,
                        enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                        skew_symmetric=self._generators_are_skew,
                    )
                grad_phi = self._compute_phi_grad(
                    phi_current, mu_current, sigma_current,
                    is_diagonal, mask, eps,
                    cached_block_exp_pairs=_phi_bep_post,
                )
                if grad_phi is not None:
                    phi_current = _retract_phi(
                        phi=phi_current,
                        delta_phi=-grad_phi,
                        generators=self.generators,
                        step_size=self.phi_lr,
                        max_norm=self.phi_max_norm,
                    )

        # Re-enable autocast context and cast results back to original dtype
        if _amp_ctx is not None:
            _amp_ctx.__exit__(None, None, None)
            mu_current = mu_current.to(dtype)
            if sigma_current is not None:
                sigma_current = sigma_current.to(dtype)
            phi_current = phi_current.to(dtype)

        # Store post-E-step state for M-step
        self._finalize_e_step(
            alpha_effective, sigma_p, sigma_current,
            beta_current, beta_heads, omega_current, eps,
        )

        # Return results
        # NOTE: Previously returned .detach() which BREAKS gradient flow!
        # The VFE descent is an "inner loop" optimization, but we still need
        # gradients to flow through the final result to train the embeddings.
        # The detach was likely added to prevent backprop through all iterations,
        # but it completely breaks learning. If memory is an issue, consider
        # gradient checkpointing instead.
        if self.update_sigma:
            return mu_current, sigma_current, phi_current, beta_history
        else:
            return mu_current, None, phi_current, beta_history

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, n_iterations={self.n_iterations}, "
            f"alpha={self.alpha}, lambda_belief={self.lambda_belief}, kappa={self.kappa}"
        )
