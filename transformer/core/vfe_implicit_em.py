"""
Implicit EM Gradient (IFT-based M-step)
========================================

Extracted from ``variational_ffn.py`` to keep the "side-quest" implicit-EM
machinery in its own focused module.  Activated by ``implicit_em=True`` on
``VariationalFFNDynamic``.

Mathematical background
-----------------------
In proper EM, the M-step gradient is ``dF/dθ = ∂F/∂θ|_{q=q*}``.  With finite
E-step iterations, ``q* ≠ q_converged``, so ``∂F/∂q ≠ 0``.  The implicit
function theorem gives the exact gradient without requiring convergence:

    ``dq*/dθ = −(∂²F/∂q²)⁻¹ · ∂²F/(∂q∂θ)``

For diagonal Gaussians this yields a per-dimension scale factor

    ``s_k = (α/σ_{p,k}) / (α/σ_{p,k} + Σ_j β_{ij}/σ_{j,k})  ∈ [0, 1]``

where σ denotes diagonal variance (not standard deviation).

which interpolates between straight-through (``s=1``) and pure EM (``s=0``).

Public API
----------
- :class:`ImplicitEMGradient` — autograd Function applying the ``s`` scale
  to the gradient of ``μ_embed``
- :class:`ImplicitEMGradientSigma` — analogous Function for ``σ_embed``
- :func:`compute_implicit_em_scales` — computes ``(μ_scale, σ_scale)`` from
  the final E-step ``(α, σ_p, β, σ_q)`` tuple

The three symbols are re-exported from ``variational_ffn`` for backward
compatibility.  External callers should import from there or from this
module directly; both paths resolve to the same objects.
"""

import torch
from typing import Tuple


__all__ = [
    "ImplicitEMGradient",
    "ImplicitEMGradientSigma",
    "compute_implicit_em_scales",
]


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

    From the E-step fixed-point equation for μ (σ denotes variance throughout):
        A_k = α/σ_{p,k} + Σ_j β_{ij}/σ_{j,k}
        s_k^{(μ)} = (α/σ_{p,k}) / A_k

    For σ (from covariance fixed-point, using squared-variance Hessian):
        s_k^{(σ)} = (α/σ_{p,k}²) / (α/σ_{p,k}² + Σ_j β_{ij}/σ_{j,k}²)

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
