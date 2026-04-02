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
      + lambda_beta * Sum_{i,j} beta_ij * KL(q_i||Omega_{ij}q_j)  # Belief alignment
      + lambda_gamma * Sum_{i,j} gamma_ij * KL(p_i||Omega_{ij}p_j)  # Prior alignment
      + CE(W_out * mu, targets)                             # Discrete observations

E-step: Minimize F w.r.t. mu, Sigma (with W_out frozen)
M-step: Minimize F w.r.t. W_out, embeddings (with mu frozen)

Gradient computation:
    dF/dtheta for theta = {mu_q, Sigma_q, mu_p, Sigma_p, phi}

With natural gradient projection:
    Delta_theta = -eta * F_inv(theta) * grad_F(theta)

Where F(theta) is the Fisher-Rao metric.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from transformer.core.gauge_utils import (
    stable_matrix_exp_pair,
    newton_schulz_orthogonalize,
    fused_block_matrix_exp_pairs,
    fused_block_diagonal_kl_diag,
    fused_block_diagonal_kl_full,
)

# =============================================================================
# VFE Gradient Debug Infrastructure
# =============================================================================
# Module-level dict populated by gradient functions when _VFE_GRAD_DEBUG is not None.
# The E-step loop sets this before calling gradient functions, then reads it after.
_VFE_GRAD_DEBUG: Optional[Dict[str, float]] = None


def _grad_norm(t: torch.Tensor) -> float:
    """Global Frobenius norm as float, detached."""
    return t.detach().norm().item()


def _per_pos_stats(t: torch.Tensor) -> Tuple[float, float, float]:
    """Per-position norm stats: (mean, max, frac_above_100).

    For (B, N, K) or (B, N, K, K) tensors, computes norm over last dim(s).
    """
    if t.dim() == 3:
        norms = torch.linalg.norm(t, dim=-1)  # (B, N)
    else:
        norms = torch.linalg.norm(t.flatten(-2), dim=-1)  # (B, N)
    return (
        norms.mean().item(),
        norms.max().item(),
        (norms > 100.0).float().mean().item(),
    )


def _aggregate_multihead_vfe_debug(d: Dict[str, float], irrep_dims) -> None:
    r"""Aggregate per-head VFE debug metrics into base keys for plotting.

    In multi-head VFE mode, gradient functions write base keys (e.g. ``grad_mu_self``)
    which are then renamed to ``headN(d=M)/grad_mu_self`` per head.  Downstream CSV
    logging and plotting expect the base keys.  This function adds them in-place:

    - Gradient norms: :math:`\sqrt{\sum_h d_h \|\nabla_h\|^2 / K}` (RMS over
      orthogonal blocks, weighted by head dimension :math:`d_h`).
    - ``kl_pairwise_mean``, ``kappa_scaled``: dimension-weighted mean across heads.
    - ``kl_pairwise_max``: max across heads.
    """
    head_keys = [k for k in d if '/' in k]
    if not head_keys:
        return
    base_names = set(k.split('/', 1)[1] for k in head_keys)
    heads = sorted(
        set(k.split('/')[0] for k in head_keys),
        key=lambda x: int(x.split('head')[1].split('(')[0]),
    )
    dims = list(irrep_dims) if irrep_dims else [1] * len(heads)
    total_dim = sum(dims)
    if total_dim == 0:
        return
    for base in base_names:
        vals = []
        for h, hp in enumerate(heads):
            v = d.get(f'{hp}/{base}')
            if v is not None and isinstance(v, (int, float)):
                vals.append((v, dims[h] if h < len(dims) else 1))
        if not vals:
            continue
        if 'grad_' in base:
            # Gradient norms: RMS weighted by head dimension (orthogonal blocks)
            d[base] = math.sqrt(sum(dim * v ** 2 for v, dim in vals) / total_dim)
        elif base == 'kl_pairwise_max':
            d[base] = max(v for v, _ in vals)
        else:
            # Dimension-weighted mean (kl_pairwise_mean, kappa_scaled, etc.)
            d[base] = sum(dim * v for v, dim in vals) / total_dim


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


def _safe_spd_inv(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Robust inversion for SPD (symmetric positive-definite) covariance matrices.

    Uses adaptive regularization: max(eps, eps * ||M||_diag_max) to handle
    both well-conditioned and ill-conditioned cases. Falls back to
    pseudoinverse if Cholesky/inv still fails (e.g., from extreme GL(K)
    transport operators corrupting covariance structure).

    Runs in float32 to survive AMP autocast contexts.

    Args:
        M: (..., K, K) covariance matrices
        eps: Base regularization floor

    Returns:
        M_inv: (..., K, K) inverse matrices
    """
    K = M.shape[-1]
    device = M.device
    orig_dtype = M.dtype

    # Force float32 for numerical stability under AMP
    with torch.amp.autocast('cuda', enabled=False):
        M = M.float()

        # Adaptive regularization: scale eps by matrix magnitude
        # This ensures regularization is proportional to the matrix scale
        diag_max = M.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1, keepdim=True).unsqueeze(-1)
        reg_scale = torch.clamp(diag_max * eps, min=eps)  # (..., 1, 1)
        M_reg = M + reg_scale * torch.eye(K, device=device, dtype=torch.float32)

        try:
            result = torch.linalg.inv(M_reg)
        except (torch.linalg.LinAlgError, RuntimeError):
            # LinAlgError for non-batched singular matrices
            # RuntimeError for batched tensors with singular elements
            # Fallback: pseudoinverse (always succeeds, handles rank-deficient)
            result = torch.linalg.pinv(M_reg)

    return result.to(orig_dtype)


def _safe_eigh(
    M: torch.Tensor,
    jitter: float = 1e-6,
    max_jitter: float = 1e-2,
    symmetrize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Robust eigendecomposition for symmetric matrices with escalating jitter.

    Retries ``torch.linalg.eigh`` with geometrically increasing jitter
    (×10 per retry) until ``max_jitter`` is reached. If all retries fail,
    falls back to SVD which uses a different algorithm (gesdd vs syev) and
    is more tolerant of ill-conditioning. For symmetric PSD matrices the
    singular values equal the eigenvalues and U provides the eigenvectors.

    Runs in float32 to survive AMP autocast contexts.

    Args:
        M: (..., K, K) symmetric matrices
        jitter: Initial diagonal regularization
        max_jitter: Maximum jitter before SVD fallback
        symmetrize: Whether to symmetrize M before decomposition

    Returns:
        (eigenvalues, eigenvectors) — same shapes as ``torch.linalg.eigh``
    """
    K = M.shape[-1]
    device = M.device
    orig_dtype = M.dtype

    with torch.amp.autocast('cuda', enabled=False):
        M = M.float()

        if symmetrize:
            M = 0.5 * (M + M.transpose(-1, -2))

        I_K = torch.eye(K, device=device, dtype=torch.float32)
        current_jitter = jitter

        while current_jitter <= max_jitter:
            try:
                M_reg = M + current_jitter * I_K
                eigvals, eigvecs = torch.linalg.eigh(M_reg)
                return eigvals.to(orig_dtype), eigvecs.to(orig_dtype)
            except (RuntimeError, torch.linalg.LinAlgError):
                current_jitter *= 10.0

        # Ultimate fallback: SVD (gesdd algorithm, more numerically stable)
        # For symmetric M: M = U @ diag(s) @ U^T, so s = eigenvalues, U = eigenvectors
        M_reg = M + max_jitter * I_K
        U, s, Vh = torch.linalg.svd(M_reg, full_matrices=False)
        return s.to(orig_dtype), U.to(orig_dtype)


# Import attention computation for dynamic β
from transformer.core.attention import compute_attention_weights, compute_transport_operators

# Numerical event monitor (shared with attention.py)
from math_utils.numerical_monitor import record as _nr

# Import SO(N) and GL(K) retraction for proper phi updates
try:
    from math_utils.generators import (
        retract_soN_torch,
        retract_glK_torch,
        is_soN_generators,
        is_glK_generators,
    )
    RETRACTION_AVAILABLE = True
except ImportError:
    RETRACTION_AVAILABLE = False


def _retract_phi(
    phi: torch.Tensor,
    delta_phi: torch.Tensor,
    generators: torch.Tensor,
    step_size: float,
    trust_region: float = None,  # None = auto-select based on gauge group
    max_norm: float = None,  # None = auto-select based on gauge group
    bch_order: int = None,  # None = auto-select based on gauge group
    eps: float = 1e-6,
    gauge_group: str = None,  # Explicit: 'GLK', 'SON', or None for auto-detect
) -> torch.Tensor:
    """
    Retract phi update using appropriate method for gauge group.

    When gauge_group is provided, uses it directly. Otherwise auto-selects
    based on n_gen:
    - n_gen = N(N-1)/2 → SO(N): compact, uses trust_region=0.3, max_norm=π, bch_order=1
    - n_gen = K²       → GL(K): non-compact, uses trust_region=0.1, max_norm=5.0, bch_order=2

    Args:
        phi: Current gauge frames (..., n_gen)
        delta_phi: Update direction (..., n_gen)
        generators: Lie algebra generators (n_gen, dim, dim)
        step_size: Learning rate
        trust_region: Maximum relative change per update (auto if None)
        max_norm: Maximum norm for phi (auto if None)
        bch_order: BCH expansion order (auto if None)
        eps: Numerical stability
        gauge_group: Explicit gauge group ('GLK' or 'SON'). Overrides
            n_gen-based auto-detection. Required when n_gen doesn't match
            standard formulas (e.g. cross-head coupled GL(K)).

    Returns:
        phi_new: Updated gauge frames
    """
    n_gen = generators.shape[0]
    if gauge_group == 'GLK':
        is_glk = RETRACTION_AVAILABLE
        is_son = False
    elif gauge_group == 'SON':
        is_glk = False
        is_son = RETRACTION_AVAILABLE
    else:
        # Auto-detect from n_gen (original heuristic).
        # When n_gen doesn't match SO(N) or GL(K) exactly, default to GL(K)
        # retraction (conservative settings). This covers cross-head coupled
        # GL(K) where n_gen = H*d² + n_cross*d² > K².
        is_son = RETRACTION_AVAILABLE and is_soN_generators(n_gen)
        is_glk = RETRACTION_AVAILABLE and (is_glK_generators(n_gen) or not is_son)

    # Auto-select defaults based on gauge group
    if trust_region is None:
        trust_region = 0.1 if is_glk else 0.3
    if max_norm is None:
        max_norm = 5.0 if is_glk else math.pi
    if bch_order is None:
        bch_order = 2 if is_glk else 1

    if not RETRACTION_AVAILABLE:
        # Fallback: gradient descent with constant trust region in Lie algebra norm
        update = step_size * delta_phi
        update_norm = torch.norm(update, dim=-1, keepdim=True)
        scale = torch.clamp(trust_region / (update_norm + eps), max=1.0)
        phi_new = phi + scale * update
        # Clamp to max norm
        phi_new_norm = torch.norm(phi_new, dim=-1, keepdim=True)
        phi_new = torch.where(
            phi_new_norm > max_norm,
            phi_new * max_norm / (phi_new_norm + eps),
            phi_new
        )
        return phi_new

    # Check if this is GL(K) (n_gen = K²) or SO(N) (n_gen = N(N-1)/2)
    if is_glk:
        # GL(K) is non-compact - needs conservative settings
        return retract_glK_torch(
            phi=phi,
            delta_phi=delta_phi,
            generators=generators,
            step_size=step_size,
            trust_region=trust_region,
            max_norm=max_norm,
            bch_order=bch_order,
            eps=eps,
        )
    elif is_son:
        # SO(N) is compact - can use standard settings
        return retract_soN_torch(
            phi=phi,
            delta_phi=delta_phi,
            generators=generators,
            step_size=step_size,
            trust_region=trust_region,
            max_norm=max_norm,
            bch_order=bch_order,
            eps=eps,
        )
    else:
        # Unknown gauge group - conservative fallback with constant trust region
        update = step_size * delta_phi
        update_norm = torch.norm(update, dim=-1, keepdim=True)
        scale = torch.clamp(trust_region / (update_norm + eps), max=1.0)
        phi_new = phi + scale * update
        phi_new_norm = torch.norm(phi_new, dim=-1, keepdim=True)
        phi_new = torch.where(
            phi_new_norm > max_norm,
            phi_new * max_norm / (phi_new_norm + eps),
            phi_new
        )
        return phi_new


# =============================================================================
# Memory-Efficient VFE Gradient Helpers
# =============================================================================

def _compute_vfe_gradients_block_diagonal(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K, K) full block-diagonal covariances
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K, K) prior covariances
    beta: torch.Tensor,        # (B, N, N) attention weights
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames
    generators: torch.Tensor,  # (n_gen, K, K) generators
    alpha: 'float | torch.Tensor',
    lambda_belief: float,
    lambda_softmax: float,
    kappa: float,
    eps: float,
    irrep_dims: List[int],
    compute_sigma_align_grad: bool,
    enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
    alpha_c0: Optional[torch.Tensor] = None,  # (K,) for product-rule correction when alpha is learnable
    cached_block_exp_pairs: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Block-diagonal VFE gradient computation for full covariance mode.

    Processes each irrep block separately to reduce memory from O(N^2 K^2) to
    O(N^2 * max(d_i^2)). Includes sigma softmax coupling
    (dBeta/dSigma) via per-block per-pair storage.

    Args:
        mu_q: Belief means (B, N, K).
        sigma_q: Full block-diagonal covariances (B, N, K, K).
        mu_p: Prior means (B, N, K).
        sigma_p: Prior covariances (B, N, K, K).
        beta: Attention weights (B, N, N).
        phi: Gauge frames (B, N, phi_dim), phi_dim = n_gen.
        generators: Lie algebra generators (n_gen, K, K).
        alpha: Self-coupling weight, scalar or (B, N, K) Bayesian precision.
        lambda_belief: Belief alignment weight.
        kappa: Temperature for softmax coupling.
        eps: Numerical stability floor.
        irrep_dims: Block dimensions [d_1, d_2, ...] for block-diagonal KL.
        compute_sigma_align_grad: Whether to compute dF/dSigma from alignment.
        enforce_orthogonal: If True, enforce Omega in SO(K) via Newton-Schulz.
        alpha_c0: (K,) softplus(raw_c0) for product-rule correction when alpha is learnable.
        cached_block_exp_pairs: Precomputed (exp_phi, exp_neg_phi) per block.

    Returns:
        grad_mu: (B, N, K) gradient w.r.t. mu.
        grad_sigma: (B, N, K, K) gradient w.r.t. Sigma.
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Initialize output gradients
    grad_mu = torch.zeros(B, N, K, device=device, dtype=dtype)
    grad_sigma = torch.zeros(B, N, K, K, device=device, dtype=dtype)

    # =================================================================
    # 1. Self-Coupling Gradient (block-wise but simpler)
    # =================================================================
    sigma_p_inv = _safe_spd_inv(sigma_p, eps=eps)
    delta_mu = mu_q - mu_p
    grad_mu_self = alpha * torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)

    sigma_q_inv = _safe_spd_inv(sigma_q, eps=eps)
    # For full covariance (4D), alpha (B,N,1) needs extra dim to broadcast with (B,N,K,K)
    alpha_4d = alpha.unsqueeze(-1) if isinstance(alpha, torch.Tensor) else alpha
    grad_sigma_self = alpha_4d * 0.5 * (sigma_p_inv - sigma_q_inv)

    # Product-rule correction for learnable alpha (full covariance):
    # ∂(α·KL)/∂θ = α·∂KL/∂θ + (∂α/∂θ)·KL
    # When α_k = c₀_k/(b₀_k + kl_k), ∂α_k/∂θ = -α_k²/c₀_k · ∂kl_k/∂θ
    if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
        # Per-dimension KL proxy from diagonal elements
        prod_qp = torch.matmul(sigma_p_inv, sigma_q)  # (B, N, K, K)
        trace_k = prod_qp.diagonal(dim1=-2, dim2=-1)  # (B, N, K)
        sp_inv_delta = torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)
        mahal_k = delta_mu * sp_inv_delta  # (B, N, K)
        logdet_p = torch.linalg.slogdet(sigma_p.float())[1]  # (B, N)
        logdet_q = torch.linalg.slogdet(sigma_q.float())[1]  # (B, N)
        logdet_k = ((logdet_p - logdet_q) / K).unsqueeze(-1).expand_as(delta_mu)  # (B, N, K)
        kl_k = 0.5 * (trace_k + mahal_k - 1 + logdet_k).clamp(min=0.0)
        # Correction to mu gradient
        grad_mu_self = grad_mu_self - (alpha ** 2 / alpha_c0) * kl_k * torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)
        # Correction to sigma gradient (4D broadcast)
        correction_scale = ((alpha ** 2 / alpha_c0) * kl_k).unsqueeze(-1)  # (B, N, K, 1)
        grad_sigma_self = grad_sigma_self - correction_scale * 0.5 * (sigma_p_inv - sigma_q_inv)

    grad_mu = grad_mu + grad_mu_self
    grad_sigma = grad_sigma + grad_sigma_self

    # Debug: self-coupling component norms
    if _VFE_GRAD_DEBUG is not None:
        _VFE_GRAD_DEBUG['grad_mu_self'] = _grad_norm(grad_mu_self)
        _VFE_GRAD_DEBUG['grad_sigma_self'] = _grad_norm(grad_sigma_self)
        _ps = _per_pos_stats(grad_sigma_self)
        _VFE_GRAD_DEBUG['grad_sigma_self_pos_mean'] = _ps[0]
        _VFE_GRAD_DEBUG['grad_sigma_self_pos_max'] = _ps[1]
        # sigma_p eigenvalue range (shows how tight priors are)
        sp_diag = torch.diagonal(sigma_p, dim1=-2, dim2=-1)
        _VFE_GRAD_DEBUG['sigma_p_min'] = sp_diag.min().item()
        _VFE_GRAD_DEBUG['sigma_p_max'] = sp_diag.max().item()
        # sigma_q eigenvalue range
        try:
            sq_eig = torch.linalg.eigvalsh(sigma_q)
            _VFE_GRAD_DEBUG['sigma_q_eig_min'] = sq_eig.min().item()
            _VFE_GRAD_DEBUG['sigma_q_eig_max'] = sq_eig.max().item()
        except (RuntimeError, torch.linalg.LinAlgError):
            _VFE_GRAD_DEBUG['sigma_q_eig_min'] = float('nan')
            _VFE_GRAD_DEBUG['sigma_q_eig_max'] = float('nan')

    # =================================================================
    # 2. Belief Alignment Gradient (block-diagonal + chunked processing)
    # =================================================================
    # Precompute matrix exponentials — FUSED by dimension group
    if cached_block_exp_pairs is not None:
        _fused_pairs = cached_block_exp_pairs
    else:
        _fused_pairs = fused_block_matrix_exp_pairs(
            phi, generators, irrep_dims, enforce_orthogonal=enforce_orthogonal
        )
    block_exp_phi = [p[0] for p in _fused_pairs]
    block_exp_neg_phi = [p[1] for p in _fused_pairs]

    # Accumulators for alignment gradients
    grad_mu_align = torch.zeros_like(mu_q)
    grad_sigma_align = torch.zeros_like(sigma_q)

    # For KL values and gradients - accumulate per-pair data for softmax coupling.
    # NOTE: Full (B, N, N) and (B, N, N, K) tensors are needed because the softmax
    # coupling term requires all pairwise KL values and gradients simultaneously.
    # Chunking over query positions (i) still saves memory on intermediate Omega,
    # transported beliefs, and inverses - just not on the final accumulators.
    kl_values = torch.zeros(B, N, N, device=device, dtype=dtype)
    grad_kl_per_pair_full = torch.zeros(B, N, N, K, device=device, dtype=dtype)

    # Per-block per-pair sigma gradients for softmax coupling (Pass 2).
    # Memory: Σ_b B*N²*d_b² instead of B*N²*K² (~82× savings for typical irrep specs).
    grad_sigma_per_pair_blocks = None
    if compute_sigma_align_grad:
        grad_sigma_per_pair_blocks = [
            torch.zeros(B, N, N, d, d, device=device, dtype=dtype)
            for d in irrep_dims
        ]

    # Process all query positions (no chunking)
    C = N
    for i_start in range(0, N, C):
        i_end = min(i_start + C, N)
        C_actual = i_end - i_start

        # Process each irrep block for this chunk of query positions
        block_start = 0
        for block_idx, d in enumerate(irrep_dims):
            block_end = block_start + d

            # Extract block beliefs - use .contiguous() to create copies and avoid
            # inplace modification errors during backward pass
            mu_block = mu_q[:, :, block_start:block_end].contiguous()  # (B, N, d)
            sigma_block = sigma_q[:, :, block_start:block_end, block_start:block_end].contiguous()  # (B, N, d, d)

            # Get chunked exponentials for query positions - use .contiguous() for same reason
            exp_phi_i = block_exp_phi[block_idx][:, i_start:i_end].contiguous()  # (B, C, d, d)
            exp_neg_phi_j = block_exp_neg_phi[block_idx].contiguous()  # (B, N, d, d)

            # Compute Omega for this chunk: (B, C, N, d, d)
            Omega_chunk = torch.einsum(
                'bikl,bjlm->bijkm',
                exp_phi_i, exp_neg_phi_j
            )  # (B, C, N, d, d)

            # Transport means and covariances for this chunk
            mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega_chunk, mu_block)  # (B, C, N, d)
            sigma_j_transported = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega_chunk, sigma_block, Omega_chunk.transpose(-1, -2)
            )  # (B, C, N, d, d)

            del Omega_chunk

            # Regularize and invert (adaptive regularization for numerical stability)
            # Use 1e-4 jitter (not eps=1e-6) — GL(K) transport can produce
            # near-singular covariances that need stronger regularization.
            I_d = torch.eye(d, device=device, dtype=dtype)
            sigma_j_transported = 0.5 * (sigma_j_transported + sigma_j_transported.transpose(-1, -2))
            sigma_j_reg = sigma_j_transported + 1e-4 * I_d
            sigma_j_inv = _safe_spd_inv(sigma_j_reg, eps=1e-4)  # (B, C, N, d, d)

            # Delta mu for this block (query chunk) - contiguous to avoid view issues
            mu_block_i = mu_block[:, i_start:i_end].contiguous()  # (B, C, d)
            delta_mu_block = mu_block_i[:, :, None, :] - mu_j_transported  # (B, C, N, d)

            # ∂KL_ij/∂μ_i for this block
            grad_kl_block = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_block)  # (B, C, N, d)
            grad_kl_per_pair_full[:, i_start:i_end, :, block_start:block_end] = grad_kl_block

            # KL terms for this block
            mahal_block = torch.einsum('bijk,bijk->bij', delta_mu_block, grad_kl_block)  # (B, C, N)

            # Use contiguous slice and clone for expand to avoid view issues
            sigma_i_block_slice = sigma_block[:, i_start:i_end].contiguous()  # (B, C, d, d)
            sigma_i_block = sigma_i_block_slice[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()  # (B, C, N, d, d)
            trace_block = torch.einsum('bijkk->bij', torch.einsum('bijkl,bijlm->bijkm', sigma_j_inv, sigma_i_block))

            try:
                L_j = torch.linalg.cholesky(sigma_j_reg)
                logdet_j = 2.0 * torch.sum(torch.log(torch.diagonal(L_j, dim1=-2, dim2=-1) + eps), dim=-1)
            except RuntimeError:
                # Fallback: use slogdet instead of zeroing (which biases KL)
                sign_j, logdet_j = torch.linalg.slogdet(sigma_j_reg)
                logdet_j = torch.where(sign_j > 0, logdet_j, torch.zeros_like(logdet_j))

            sigma_i_block_diag = sigma_i_block_slice + eps * I_d  # (B, C, d, d)
            try:
                L_i = torch.linalg.cholesky(sigma_i_block_diag)
                logdet_i = 2.0 * torch.sum(torch.log(torch.diagonal(L_i, dim1=-2, dim2=-1) + eps), dim=-1)
            except RuntimeError:
                # Fallback: use slogdet instead of zeroing (which biases KL)
                sign_i, logdet_i = torch.linalg.slogdet(sigma_i_block_diag)
                logdet_i = torch.where(sign_i > 0, logdet_i, torch.zeros_like(logdet_i))

            kl_block = 0.5 * (trace_block + mahal_block - d + logdet_j - logdet_i[:, :, None])
            # Clamp KL to [0, max] for numerical stability (scale ceiling with block dim d)
            kl_values[:, i_start:i_end, :] = kl_values[:, i_start:i_end, :] + kl_block.clamp(min=0.0, max=max(100.0, 20.0 * d))

            # Sigma alignment gradient for this block
            if compute_sigma_align_grad:
                sigma_i_inv_block = _safe_spd_inv(sigma_i_block_diag, eps=1e-4)  # (B, C, d, d)
                # Use .clone() after expand to ensure contiguous memory layout
                sigma_i_inv_exp = sigma_i_inv_block[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()
                grad_sigma_block = 0.5 * (sigma_j_inv - sigma_i_inv_exp)  # (B, C, N, d, d)
                beta_chunk = beta[:, i_start:i_end, :].contiguous()  # (B, C, N)
                grad_sigma_block_weighted = lambda_belief * torch.einsum('bij,bijkl->bikl', beta_chunk, grad_sigma_block)
                grad_sigma_align[:, i_start:i_end, block_start:block_end, block_start:block_end] = (
                    grad_sigma_align[:, i_start:i_end, block_start:block_end, block_start:block_end] + grad_sigma_block_weighted
                )
                # Store per-pair sigma gradients for softmax coupling (Pass 2)
                grad_sigma_per_pair_blocks[block_idx][:, i_start:i_end, :, :, :] = grad_sigma_block

            del sigma_j_transported, sigma_j_inv, mu_j_transported
            block_start = block_end

    # avg_grad = Σ_j β_ij · ∂KL_ij/∂μ_i (used for both direct and softmax terms)
    avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair_full)
    grad_mu_direct = lambda_belief * avg_grad

    # Softmax coupling term
    # Scale kappa by √K to match attention temperature τ = √K
    kappa_scaled = max(kappa * math.sqrt(max(K, 1)), eps)
    grad_deviation = avg_grad.unsqueeze(2) - grad_kl_per_pair_full
    d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa_scaled
    grad_mu_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

    grad_mu_align = grad_mu_direct + grad_mu_softmax
    grad_mu = grad_mu + grad_mu_align
    grad_sigma = grad_sigma + grad_sigma_align

    # Debug: alignment component norms (before softmax coupling)
    if _VFE_GRAD_DEBUG is not None:
        _VFE_GRAD_DEBUG['grad_mu_direct'] = _grad_norm(grad_mu_direct)
        _VFE_GRAD_DEBUG['grad_mu_softmax'] = _grad_norm(grad_mu_softmax)
        _VFE_GRAD_DEBUG['grad_sigma_align_direct'] = _grad_norm(grad_sigma_align)
        _ps = _per_pos_stats(grad_sigma_align)
        _VFE_GRAD_DEBUG['grad_sigma_align_pos_mean'] = _ps[0]
        _VFE_GRAD_DEBUG['grad_sigma_align_pos_max'] = _ps[1]
        # KL pairwise stats (drives softmax coupling magnitude)
        _VFE_GRAD_DEBUG['kl_pairwise_mean'] = kl_values.mean().item()
        _VFE_GRAD_DEBUG['kl_pairwise_max'] = kl_values.max().item()
        _VFE_GRAD_DEBUG['kappa_scaled'] = kappa_scaled
        # Fraction of pairs near the KL ceiling (diagnoses clamp saturation)
        _kl_ceil = max(100.0, 20.0 * K)
        _VFE_GRAD_DEBUG['kl_frac_above_90pct'] = (kl_values > 0.9 * _kl_ceil).float().mean().item()
        _VFE_GRAD_DEBUG['kl_p95'] = kl_values.quantile(0.95).item()

    # Sigma softmax coupling (Pass 2): ∂β/∂Σ term computed per-block.
    # Uses stored per-block per-pair sigma gradients to avoid (B, N, N, K, K) memory.
    _grad_sigma_before_softmax = grad_sigma.clone() if (_VFE_GRAD_DEBUG is not None) else None
    if compute_sigma_align_grad and grad_sigma_per_pair_blocks is not None:
        kappa_scaled = max(kappa * math.sqrt(max(K, 1)), eps)
        block_start = 0
        for block_idx, d in enumerate(irrep_dims):
            block_end = block_start + d
            g_per_pair = grad_sigma_per_pair_blocks[block_idx]  # (B, N, N, d, d)
            avg_g = torch.einsum('bij,bijkl->bikl', beta, g_per_pair)  # (B, N, d, d)
            g_deviation = avg_g.unsqueeze(2) - g_per_pair  # (B, N, N, d, d)
            d_beta_d_sigma = beta.unsqueeze(-1).unsqueeze(-1) * g_deviation / kappa_scaled  # (B, N, N, d, d)
            grad_sigma_softmax_block = lambda_softmax * torch.einsum('bij,bijkl->bikl', kl_values, d_beta_d_sigma)  # (B, N, d, d)
            grad_sigma[:, :, block_start:block_end, block_start:block_end] = (
                grad_sigma[:, :, block_start:block_end, block_start:block_end] + grad_sigma_softmax_block
            )
            block_start = block_end
        del grad_sigma_per_pair_blocks

    # Debug: softmax coupling contribution and final totals
    if _VFE_GRAD_DEBUG is not None:
        if _grad_sigma_before_softmax is not None:
            _sigma_softmax_contrib = grad_sigma - _grad_sigma_before_softmax
            _VFE_GRAD_DEBUG['grad_sigma_softmax'] = _grad_norm(_sigma_softmax_contrib)
            _ps = _per_pos_stats(_sigma_softmax_contrib)
            _VFE_GRAD_DEBUG['grad_sigma_softmax_pos_mean'] = _ps[0]
            _VFE_GRAD_DEBUG['grad_sigma_softmax_pos_max'] = _ps[1]
        _VFE_GRAD_DEBUG['grad_mu_total'] = _grad_norm(grad_mu)
        _VFE_GRAD_DEBUG['grad_sigma_total'] = _grad_norm(grad_sigma)
        _ps = _per_pos_stats(grad_sigma)
        _VFE_GRAD_DEBUG['grad_sigma_total_pos_mean'] = _ps[0]
        _VFE_GRAD_DEBUG['grad_sigma_total_pos_max'] = _ps[1]

    return grad_mu, grad_sigma


def _compute_vfe_gradients_block_diagonal_diag(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K) diagonal variances
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K) prior variances
    beta: torch.Tensor,        # (B, N, N) attention weights
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames
    generators: torch.Tensor,  # (n_gen, K, K) generators
    alpha: 'float | torch.Tensor',
    lambda_belief: float,
    lambda_softmax: float,
    kappa: float,
    eps: float,
    irrep_dims: List[int],
    compute_sigma_align_grad: bool,
    enforce_orthogonal: bool = False,
    alpha_c0: Optional[torch.Tensor] = None,  # (K,) for product-rule correction
    cached_block_exp_pairs: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Block-diagonal VFE gradient computation for diagonal covariance mode.

    Processes each irrep block separately with small d x d Omega tensors
    and O(d) diagonal KL formulas. No matrix inverse or Cholesky needed.
    Includes sigma softmax coupling (dBeta/dSigma term).

    Memory: O(N^2 * max(d_i^2)) instead of O(N^2 * K^2).

    Args:
        mu_q: Belief means (B, N, K).
        sigma_q: Diagonal variances (B, N, K).
        mu_p: Prior means (B, N, K).
        sigma_p: Prior diagonal variances (B, N, K).
        beta: Attention weights (B, N, N).
        phi: Gauge frames (B, N, phi_dim), phi_dim = n_gen.
        generators: Lie algebra generators (n_gen, K, K).
        alpha: Self-coupling weight, scalar or (B, N, K) Bayesian precision.
        lambda_belief: Belief alignment weight.
        kappa: Temperature for softmax coupling.
        eps: Numerical stability floor.
        irrep_dims: Block dimensions [d_1, d_2, ...] for block-diagonal KL.
        compute_sigma_align_grad: Whether to compute dF/dSigma from alignment.
        enforce_orthogonal: If True, enforce Omega in SO(K) via Newton-Schulz.
        alpha_c0: (K,) softplus(raw_c0) for product-rule correction when alpha is learnable.
        cached_block_exp_pairs: Precomputed (exp_phi, exp_neg_phi) per block.

    Returns:
        grad_mu: (B, N, K) gradient w.r.t. mu.
        grad_sigma: (B, N, K) gradient w.r.t. diagonal sigma.
    """
    # Squeeze trailing singleton dimensions for robustness
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)
    while sigma_p.dim() > 3 and sigma_p.shape[-1] == 1:
        sigma_p = sigma_p.squeeze(-1)

    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Force float32 for all sigma divisions, logs, and KL computation under AMP
    mu_q = mu_q.float()
    mu_p = mu_p.float()
    sigma_q = sigma_q.float()
    sigma_p = sigma_p.float()
    beta = beta.float()

    sigma_q_safe = sigma_q.clamp(min=eps)
    sigma_p_safe = sigma_p.clamp(min=eps)

    # =================================================================
    # 1. Self-Coupling Gradient (diagonal, no blocks needed)
    # =================================================================
    delta_mu = mu_q - mu_p
    grad_mu_self = alpha * delta_mu / sigma_p_safe
    grad_sigma_self = alpha * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)

    # Product-rule correction for learnable alpha
    if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
        kl_k = 0.5 * (sigma_q_safe / sigma_p_safe + delta_mu ** 2 / sigma_p_safe
                      - 1.0 + torch.log(sigma_p_safe) - torch.log(sigma_q_safe))
        kl_k = kl_k.clamp(min=0.0)
        grad_mu_self = grad_mu_self - (alpha ** 2 / alpha_c0) * kl_k * delta_mu / sigma_p_safe
        grad_sigma_self = grad_sigma_self - (alpha ** 2 / alpha_c0) * kl_k * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)

    # Debug: self-coupling component norms
    if _VFE_GRAD_DEBUG is not None:
        _VFE_GRAD_DEBUG['grad_mu_self'] = _grad_norm(grad_mu_self)
        _VFE_GRAD_DEBUG['grad_sigma_self'] = _grad_norm(grad_sigma_self)
        _ps = _per_pos_stats(grad_sigma_self)
        _VFE_GRAD_DEBUG['grad_sigma_self_pos_mean'] = _ps[0]
        _VFE_GRAD_DEBUG['grad_sigma_self_pos_max'] = _ps[1]
        _VFE_GRAD_DEBUG['sigma_p_min'] = sigma_p_safe.min().item()
        _VFE_GRAD_DEBUG['sigma_p_max'] = sigma_p_safe.max().item()
        _VFE_GRAD_DEBUG['sigma_q_eig_min'] = sigma_q_safe.min().item()
        _VFE_GRAD_DEBUG['sigma_q_eig_max'] = sigma_q_safe.max().item()

    # =================================================================
    # 2. Belief Alignment Gradient (block-diagonal + diagonal formulas)
    # =================================================================
    # Precompute matrix exponentials — FUSED by dimension group
    if cached_block_exp_pairs is not None:
        _fused_pairs = cached_block_exp_pairs
    else:
        _fused_pairs = fused_block_matrix_exp_pairs(
            phi, generators, irrep_dims, enforce_orthogonal=enforce_orthogonal
        )
    block_exp_phi = [p[0] for p in _fused_pairs]
    block_exp_neg_phi = [p[1] for p in _fused_pairs]

    # Accumulators for per-pair KL values and gradients across all blocks
    kl_values = torch.zeros(B, N, N, device=device, dtype=dtype)
    grad_kl_per_pair_full = torch.zeros(B, N, N, K, device=device, dtype=dtype)
    grad_sigma_align = torch.zeros_like(sigma_q)
    # Accumulator for sigma softmax coupling (same memory cost as grad_kl_per_pair_full)
    grad_sigma_per_pair_full = torch.zeros(B, N, N, K, device=device, dtype=dtype) if (
        compute_sigma_align_grad) else None

    # Process each irrep block sequentially with direct Omega construction.
    # This avoids the factored transport approach (A @ (B diag(σ) B^T) @ A^T)
    # which accumulates more float32 rounding error through intermediate matmuls.
    block_start = 0
    for block_idx, d in enumerate(irrep_dims):
        block_end = block_start + d

        mu_block = mu_q[:, :, block_start:block_end].contiguous()        # (B, N, d)
        sigma_block = sigma_q_safe[:, :, block_start:block_end].contiguous()  # (B, N, d)

        # Block Omega: (B, N, N, d, d) — direct construction for numerical precision
        Omega_block = torch.einsum(
            'bikl,bjlm->bijkm',
            block_exp_phi[block_idx], block_exp_neg_phi[block_idx]
        )

        # Transport means
        mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega_block, mu_block)  # (B, N, N, d)

        # Diagonal covariance transport: σ_t[k] = Σ_l Ω_kl² * σ[l]
        # This extracts diag(Ω @ diag(σ) @ Ω^T), discarding off-diagonal elements
        # of the transported covariance. For non-identity Ω ∈ GL(K), the full
        # transported covariance is NOT diagonal, so this is an approximation that
        # breaks strict gauge equivariance. Use exact_diagonal_transport=True for
        # exact transport (lifts to full covariance, applies sandwich product,
        # extracts diagonal). The approximation quality depends on how far Ω is
        # from identity within each irrep block.
        sigma_j_transported = torch.einsum(
            'bijkl,bijkl,bjl->bijk', Omega_block, Omega_block, sigma_block
        ).clamp(min=eps)  # (B, N, N, d)

        del Omega_block

        # Delta mu (broadcast instead of expand+clone to avoid 59M-element copy)
        mu_block_i = mu_block[:, :, None, :]  # (B, N, 1, d) - broadcasts with (B, N, N, d)
        delta_mu = mu_block_i - mu_j_transported  # (B, N, N, d)

        # ∂KL/∂μ_i = (μ_i - μ_j_t) / σ_j_t (element-wise)
        grad_kl_block = delta_mu / sigma_j_transported  # (B, N, N, d)
        grad_kl_per_pair_full[:, :, :, block_start:block_end] = grad_kl_block

        # Diagonal KL for this block (broadcast, no clone)
        sigma_i_block = sigma_block[:, :, None, :]  # (B, N, 1, d)
        trace_block = (sigma_i_block / sigma_j_transported).sum(dim=-1)
        mahal_block = (delta_mu ** 2 / sigma_j_transported).sum(dim=-1)
        logdet_block = (torch.log(sigma_j_transported.clamp(min=eps))
                        - torch.log(sigma_i_block.clamp(min=eps))).sum(dim=-1)

        kl_block = 0.5 * (trace_block + mahal_block - d + logdet_block)
        kl_block = kl_block.clamp(min=0.0, max=max(100.0, 20.0 * d))
        kl_values = kl_values + kl_block

        # Sigma alignment gradient for this block
        if compute_sigma_align_grad:
            sigma_j_inv_diag = 1.0 / sigma_j_transported  # (B, N, N, d)
            sigma_i_inv = 1.0 / sigma_block  # (B, N, d)
            grad_sigma_pair = 0.5 * (sigma_j_inv_diag - sigma_i_inv[:, :, None, :])  # broadcast
            grad_sigma_align[:, :, block_start:block_end] = (
                grad_sigma_align[:, :, block_start:block_end]
                + lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_sigma_pair)
            )
            if grad_sigma_per_pair_full is not None:
                grad_sigma_per_pair_full[:, :, :, block_start:block_end] = grad_sigma_pair

        del sigma_j_transported, mu_j_transported, delta_mu
        block_start = block_end

    # avg_grad = Σ_j β_ij · ∂KL_ij/∂μ_i (used for both direct and softmax terms)
    avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair_full)
    grad_mu_direct = lambda_belief * avg_grad

    # Softmax coupling term
    kappa_scaled = max(kappa * math.sqrt(max(K, 1)), eps)
    grad_deviation = avg_grad.unsqueeze(2) - grad_kl_per_pair_full
    d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa_scaled
    grad_mu_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

    grad_mu_align = grad_mu_direct + grad_mu_softmax
    grad_mu = grad_mu_self + grad_mu_align

    # Debug: alignment component norms (before softmax coupling for sigma)
    if _VFE_GRAD_DEBUG is not None:
        _VFE_GRAD_DEBUG['grad_mu_direct'] = _grad_norm(grad_mu_direct)
        _VFE_GRAD_DEBUG['grad_mu_softmax'] = _grad_norm(grad_mu_softmax)
        _VFE_GRAD_DEBUG['grad_sigma_align_direct'] = _grad_norm(grad_sigma_align)
        _ps = _per_pos_stats(grad_sigma_align)
        _VFE_GRAD_DEBUG['grad_sigma_align_pos_mean'] = _ps[0]
        _VFE_GRAD_DEBUG['grad_sigma_align_pos_max'] = _ps[1]
        _VFE_GRAD_DEBUG['kl_pairwise_mean'] = kl_values.mean().item()
        _VFE_GRAD_DEBUG['kl_pairwise_max'] = kl_values.max().item()
        _VFE_GRAD_DEBUG['kappa_scaled'] = kappa_scaled
        # Fraction of pairs near the KL ceiling (diagnoses clamp saturation)
        _kl_ceil = max(100.0, 20.0 * K)
        _VFE_GRAD_DEBUG['kl_frac_above_90pct'] = (kl_values > 0.9 * _kl_ceil).float().mean().item()
        _VFE_GRAD_DEBUG['kl_p95'] = kl_values.quantile(0.95).item()

    # Sigma softmax coupling: Σ_j KL_ij · ∂β_ij/∂σ_i
    grad_sigma_softmax_norm = 0.0
    if grad_sigma_per_pair_full is not None:
        avg_sigma_grad = torch.einsum('bij,bijk->bik', beta, grad_sigma_per_pair_full)
        sigma_grad_deviation = avg_sigma_grad.unsqueeze(2) - grad_sigma_per_pair_full
        d_beta_d_sigma = beta.unsqueeze(-1) * sigma_grad_deviation / kappa_scaled
        grad_sigma_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_sigma)
        if _VFE_GRAD_DEBUG is not None:
            grad_sigma_softmax_norm = _grad_norm(grad_sigma_softmax)
        grad_sigma_align = grad_sigma_align + grad_sigma_softmax

    grad_sigma = grad_sigma_self + grad_sigma_align

    # Debug: final totals
    if _VFE_GRAD_DEBUG is not None:
        _VFE_GRAD_DEBUG['grad_sigma_softmax'] = grad_sigma_softmax_norm
        _VFE_GRAD_DEBUG['grad_mu_total'] = _grad_norm(grad_mu)
        _VFE_GRAD_DEBUG['grad_sigma_total'] = _grad_norm(grad_sigma)
        _ps = _per_pos_stats(grad_sigma)
        _VFE_GRAD_DEBUG['grad_sigma_total_pos_mean'] = _ps[0]
        _VFE_GRAD_DEBUG['grad_sigma_total_pos_max'] = _ps[1]

    return grad_mu.to(dtype), grad_sigma.to(dtype)


def _fused_attention_and_vfe_gradients_block_diag(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K) diagonal variances
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K) prior variances
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames
    generators: torch.Tensor,  # (n_gen, K, K) generators
    alpha: 'float | torch.Tensor',
    lambda_belief: float,
    lambda_softmax: float,
    kappa: float,
    eps: float,
    irrep_dims: List[int],
    compute_sigma_align_grad: bool,
    enforce_orthogonal: bool = False,
    alpha_c0: Optional[torch.Tensor] = None,
    cached_block_exp_pairs: Optional[list] = None,
    mask: Optional[torch.Tensor] = None,
    mask_self_attention: bool = False,
    use_rope: bool = False,
    rope_base: float = 10000.0,
    return_kl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Fused attention + VFE gradient computation for block-diagonal diagonal mode.

    Computes beta (attention weights) AND VFE gradients in a single pass over
    irrep blocks, sharing the Omega construction. Eliminates the redundant Omega
    computation that occurs when compute_attention_weights and
    compute_vfe_gradients_gpu are called separately. Includes sigma softmax
    coupling (dBeta/dSigma). Incompatible with exact_diagonal_transport (which
    lifts to full covariance and disables fused diagonal paths).

    For a config with 5 heads x d=15, this saves 5 x O(B*N^2*d^2) Omega
    constructions per VFE iteration.

    Args:
        mu_q: Belief means (B, N, K).
        sigma_q: Diagonal variances (B, N, K).
        mu_p: Prior means (B, N, K).
        sigma_p: Prior diagonal variances (B, N, K).
        phi: Gauge frames (B, N, phi_dim), phi_dim = n_gen.
        generators: Lie algebra generators (n_gen, K, K).
        alpha: Self-coupling weight, scalar or (B, N, K) Bayesian precision.
        lambda_belief: Belief alignment weight.
        kappa: Temperature for softmax coupling.
        eps: Numerical stability floor.
        irrep_dims: Block dimensions [d_1, d_2, ...] for block-diagonal KL.
        compute_sigma_align_grad: Whether to compute dF/dSigma from alignment.
        enforce_orthogonal: If True, enforce Omega in SO(K) via Newton-Schulz.
        alpha_c0: (K,) softplus(raw_c0) for product-rule correction.
        cached_block_exp_pairs: Precomputed (exp_phi, exp_neg_phi) per block.
        mask: Causal mask (B, N, N), 0 = cannot attend.
        mask_self_attention: If True, mask diagonal (no self-attention).
        use_rope: Apply rotary position embeddings to mu for KL computation.
        rope_base: RoPE base frequency.
        return_kl: If True, also return pairwise KL matrix.

    Returns:
        beta: (B, N, N) attention weights.
        grad_mu: (B, N, K) gradient w.r.t. mu.
        grad_sigma: (B, N, K) gradient w.r.t. diagonal sigma.
        kl_matrix: (B, N, N) pairwise KL divergences, or None if return_kl=False.
    """
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)
    while sigma_p.dim() > 3 and sigma_p.shape[-1] == 1:
        sigma_p = sigma_p.squeeze(-1)

    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    mu_q = mu_q.float()
    mu_p = mu_p.float()
    sigma_q = sigma_q.float()
    sigma_p = sigma_p.float()

    sigma_q_safe = sigma_q.clamp(min=eps)
    sigma_p_safe = sigma_p.clamp(min=eps)

    # Apply RoPE to a copy of mu for KL computation (not for gradients)
    if use_rope:
        from transformer.core.attention import _apply_rope
        mu_q_rope = _apply_rope(mu_q, base=rope_base)
    else:
        mu_q_rope = mu_q

    # Self-coupling gradient
    delta_mu_sp = mu_q - mu_p
    grad_mu_self = alpha * delta_mu_sp / sigma_p_safe
    grad_sigma_self = alpha * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)

    if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
        kl_k = 0.5 * (sigma_q_safe / sigma_p_safe + delta_mu_sp ** 2 / sigma_p_safe
                      - 1.0 + torch.log(sigma_p_safe) - torch.log(sigma_q_safe))
        kl_k = kl_k.clamp(min=0.0)
        grad_mu_self = grad_mu_self - (alpha ** 2 / alpha_c0) * kl_k * delta_mu_sp / sigma_p_safe
        grad_sigma_self = grad_sigma_self - (alpha ** 2 / alpha_c0) * kl_k * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)

    # Debug: self-coupling component norms (fused path)
    if _VFE_GRAD_DEBUG is not None:
        _VFE_GRAD_DEBUG['grad_mu_self'] = _grad_norm(grad_mu_self)
        _VFE_GRAD_DEBUG['grad_sigma_self'] = _grad_norm(grad_sigma_self)
        _ps = _per_pos_stats(grad_sigma_self)
        _VFE_GRAD_DEBUG['grad_sigma_self_pos_mean'] = _ps[0]
        _VFE_GRAD_DEBUG['grad_sigma_self_pos_max'] = _ps[1]
        _VFE_GRAD_DEBUG['sigma_p_min'] = sigma_p_safe.min().item()
        _VFE_GRAD_DEBUG['sigma_p_max'] = sigma_p_safe.max().item()
        _VFE_GRAD_DEBUG['sigma_q_eig_min'] = sigma_q_safe.min().item()
        _VFE_GRAD_DEBUG['sigma_q_eig_max'] = sigma_q_safe.max().item()

    # Precompute matrix exponentials
    if cached_block_exp_pairs is not None:
        _fused_pairs = cached_block_exp_pairs
    else:
        _fused_pairs = fused_block_matrix_exp_pairs(
            phi, generators, irrep_dims, enforce_orthogonal=enforce_orthogonal
        )
    block_exp_phi = [p[0] for p in _fused_pairs]
    block_exp_neg_phi = [p[1] for p in _fused_pairs]

    # Accumulators
    kl_values = torch.zeros(B, N, N, device=device, dtype=torch.float32)
    # When RoPE is active, kl_values uses RoPE-rotated mu (for attention β) but
    # gradients use raw mu. The softmax coupling ∂β/∂μ requires KL values consistent
    # with the gradient space, so we accumulate raw-mu KLs separately.
    kl_values_raw = torch.zeros(B, N, N, device=device, dtype=torch.float32) if use_rope else None
    grad_kl_per_pair_full = torch.zeros(B, N, N, K, device=device, dtype=torch.float32)
    grad_sigma_align = torch.zeros_like(sigma_q)
    grad_sigma_per_pair_full = torch.zeros(B, N, N, K, device=device, dtype=torch.float32) if (
        compute_sigma_align_grad) else None
    # Single pass over blocks: compute Omega, KL, and gradients together
    block_start = 0
    for block_idx, d in enumerate(irrep_dims):
        block_end = block_start + d

        # Use RoPE-rotated mu for KL/attention, raw mu for gradients
        mu_block_rope = mu_q_rope[:, :, block_start:block_end].contiguous()
        mu_block = mu_q[:, :, block_start:block_end].contiguous()
        sigma_block = sigma_q_safe[:, :, block_start:block_end].contiguous()

        # Build Omega ONCE for this block
        Omega_block = torch.einsum(
            'bikl,bjlm->bijkm',
            block_exp_phi[block_idx], block_exp_neg_phi[block_idx]
        )

        # Transport (use RoPE mu for KL computation)
        mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega_block, mu_block_rope)
        sigma_j_transported = torch.einsum(
            'bijkl,bijkl,bjl->bijk', Omega_block, Omega_block, sigma_block
        ).clamp(min=1e-4)

        # KL computation (for attention weights)
        mu_block_i_rope = mu_block_rope[:, :, None, :]  # broadcast, no clone needed
        delta_mu_kl = mu_block_i_rope - mu_j_transported

        sigma_block_safe = sigma_block[:, :, None, :].clamp(min=1e-4)
        trace_block = (sigma_block_safe / sigma_j_transported).sum(dim=-1)
        mahal_block = (delta_mu_kl ** 2 / sigma_j_transported).sum(dim=-1)
        logdet_block = (torch.log(sigma_j_transported) - torch.log(sigma_block_safe)).sum(dim=-1)

        kl_block = 0.5 * (trace_block + mahal_block - d + logdet_block)
        kl_block = kl_block.clamp(min=0.0, max=max(100.0, 20.0 * d))
        kl_values = kl_values + kl_block

        # Gradient computation (use raw mu, not RoPE)
        # Re-transport with raw mu if RoPE is active
        if use_rope:
            mu_j_transported_raw = torch.einsum('bijkl,bjl->bijk', Omega_block, mu_block)
            delta_mu_grad = mu_block[:, :, None, :] - mu_j_transported_raw
        else:
            delta_mu_grad = mu_block[:, :, None, :] - mu_j_transported

        grad_kl_block = delta_mu_grad / sigma_j_transported
        grad_kl_per_pair_full[:, :, :, block_start:block_end] = grad_kl_block

        # Accumulate raw-mu KL for softmax coupling consistency when RoPE is active
        if kl_values_raw is not None:
            mahal_raw = (delta_mu_grad ** 2 / sigma_j_transported).sum(dim=-1)
            kl_block_raw = 0.5 * (trace_block + mahal_raw - d + logdet_block)
            kl_values_raw = kl_values_raw + kl_block_raw.clamp(min=0.0, max=max(100.0, 20.0 * d))

        if compute_sigma_align_grad:
            sigma_j_inv_diag = 1.0 / sigma_j_transported
            sigma_i_inv = 1.0 / sigma_block
            grad_sigma_pair = 0.5 * (sigma_j_inv_diag - sigma_i_inv[:, :, None, :])
            if grad_sigma_per_pair_full is not None:
                grad_sigma_per_pair_full[:, :, :, block_start:block_end] = grad_sigma_pair

        del Omega_block, sigma_j_transported, mu_j_transported
        block_start = block_end

    # Compute attention weights from KL values
    dim_scale = math.sqrt(max(K, 1))
    logits = -kl_values / (kappa * dim_scale)

    if mask is not None:
        logits = logits.masked_fill(mask == 0, float('-inf'))

    if mask_self_attention:
        diag_idx = torch.arange(N, device=device)
        has_other_targets = (logits != float('-inf')).sum(dim=-1) > 1
        logits = logits.clone()
        diag_vals = logits[:, diag_idx, diag_idx]
        masked_diag_vals = torch.where(
            has_other_targets,
            torch.full_like(diag_vals, float('-inf')),
            diag_vals
        )
        logits[:, diag_idx, diag_idx] = masked_diag_vals

    beta = torch.nn.functional.softmax(logits, dim=-1)
    masked_positions = (logits == float('-inf'))
    beta = torch.where(masked_positions, beta, beta.clamp(min=eps))
    beta_sum = beta.sum(dim=-1, keepdim=True).clamp(min=eps)
    beta = beta / beta_sum

    # Compute VFE gradients using the beta we just computed
    avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair_full)
    grad_mu_direct = lambda_belief * avg_grad

    kappa_scaled = max(kappa * math.sqrt(max(K, 1)), eps)
    grad_deviation = avg_grad.unsqueeze(2) - grad_kl_per_pair_full
    d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa_scaled
    # Use raw-mu KL values for the softmax coupling when RoPE is active,
    # so the chain rule ∂(β·KL)/∂μ is consistent: both KL and ∂KL/∂μ
    # are computed in the same (raw, non-rotated) space.
    kl_for_coupling = kl_values_raw if kl_values_raw is not None else kl_values
    grad_mu_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_for_coupling, d_beta_d_mu)

    grad_mu = grad_mu_self + grad_mu_direct + grad_mu_softmax

    # Sigma gradients
    grad_sigma_softmax_norm = 0.0
    if grad_sigma_per_pair_full is not None:
        # Direct sigma gradient
        block_start = 0
        for block_idx, d in enumerate(irrep_dims):
            block_end = block_start + d
            grad_sigma_pair = grad_sigma_per_pair_full[:, :, :, block_start:block_end]
            grad_sigma_align[:, :, block_start:block_end] = (
                grad_sigma_align[:, :, block_start:block_end]
                + lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_sigma_pair)
            )
            block_start = block_end

        # Debug: capture direct alignment norm before softmax coupling
        if _VFE_GRAD_DEBUG is not None:
            _VFE_GRAD_DEBUG['grad_sigma_align_direct'] = _grad_norm(grad_sigma_align)
            _ps = _per_pos_stats(grad_sigma_align)
            _VFE_GRAD_DEBUG['grad_sigma_align_pos_mean'] = _ps[0]
            _VFE_GRAD_DEBUG['grad_sigma_align_pos_max'] = _ps[1]

        # Sigma softmax coupling (use raw-mu KL for consistency, same as mu coupling)
        avg_sigma_grad = torch.einsum('bij,bijk->bik', beta, grad_sigma_per_pair_full)
        sigma_grad_deviation = avg_sigma_grad.unsqueeze(2) - grad_sigma_per_pair_full
        d_beta_d_sigma = beta.unsqueeze(-1) * sigma_grad_deviation / kappa_scaled
        grad_sigma_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_for_coupling, d_beta_d_sigma)
        if _VFE_GRAD_DEBUG is not None:
            grad_sigma_softmax_norm = _grad_norm(grad_sigma_softmax)
        grad_sigma_align = grad_sigma_align + grad_sigma_softmax

    grad_sigma = grad_sigma_self + grad_sigma_align

    # Debug: final totals (fused path)
    if _VFE_GRAD_DEBUG is not None:
        _VFE_GRAD_DEBUG['grad_mu_direct'] = _grad_norm(grad_mu_direct)
        _VFE_GRAD_DEBUG['grad_mu_softmax'] = _grad_norm(grad_mu_softmax)
        _VFE_GRAD_DEBUG['grad_sigma_softmax'] = grad_sigma_softmax_norm
        _VFE_GRAD_DEBUG['kl_pairwise_mean'] = kl_values.mean().item()
        _VFE_GRAD_DEBUG['kl_pairwise_max'] = kl_values.max().item()
        _VFE_GRAD_DEBUG['kappa_scaled'] = kappa_scaled
        # Fraction of pairs near the KL ceiling (diagnoses clamp saturation)
        _kl_ceil = max(100.0, 20.0 * K)
        _VFE_GRAD_DEBUG['kl_frac_above_90pct'] = (kl_values > 0.9 * _kl_ceil).float().mean().item()
        _VFE_GRAD_DEBUG['kl_p95'] = kl_values.quantile(0.95).item()
        _VFE_GRAD_DEBUG['grad_mu_total'] = _grad_norm(grad_mu)
        _VFE_GRAD_DEBUG['grad_sigma_total'] = _grad_norm(grad_sigma)
        _ps = _per_pos_stats(grad_sigma)
        _VFE_GRAD_DEBUG['grad_sigma_total_pos_mean'] = _ps[0]
        _VFE_GRAD_DEBUG['grad_sigma_total_pos_max'] = _ps[1]

    kl_out = kl_values if return_kl else None
    return beta.to(dtype), grad_mu.to(dtype), grad_sigma.to(dtype), kl_out


# Chunked VFE gradient path removed — block-diagonal path handles memory
# efficiency via irrep decomposition (always enabled via use_block_diagonal_kl=True).
# See _compute_vfe_gradients_block_diagonal_diag for the active path.
# =============================================================================
# GPU-Based Gradient Computation (PyTorch - FAST!)
# =============================================================================

def compute_vfe_gradients_gpu(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K) diagonal variances or (B, N, K, K) full
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K) diagonal or (B, N, K, K) full
    beta: torch.Tensor,        # (B, N, N) attention weights
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames where n_gen is # of generators
    generators: torch.Tensor,  # (n_gen, K, K) Lie algebra generators
    alpha: 'float | torch.Tensor' = 0.01,  # Self-coupling weight: scalar or (B, N, K) per-dim Bayesian precision
    lambda_belief: float = 1.0,  # Belief alignment weight (direct: β·∇KL)
    lambda_softmax: float = 1.0,  # Softmax coupling weight (GELU-like ∂β/∂θ · KL)
    kappa: float = 1.0,        # Temperature (for normalization)
    eps: float = 1e-6,
    alpha_c0: Optional[torch.Tensor] = None,  # (K,) softplus(raw_c0) for product-rule correction when alpha is learnable
    cached_transport: Optional[dict] = None,  # Precomputed transport operators
    compute_sigma_align_grad: bool = True,  # Compute sigma gradient from alignment term
    irrep_dims: Optional[List[int]] = None,  # Block dimensions for block-diagonal processing
    enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
    cached_block_exp_pairs: Optional[list] = None,  # Precomputed block exponential pairs
    exact_diagonal_transport: bool = False,  # Lift diagonal σ for exact transport
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute VFE gradients entirely on GPU using PyTorch.

    Fully vectorized. Supports SO(3), SO(N), and GL(K) gauge groups.
    The number of generators (n_gen) determines the group: 3 for SO(3),
    N(N-1)/2 for SO(N), K^2 for GL(K).

    Gradients computed:
    1. Self-coupling: d/d_mu_q [alpha * KL(q||p)]
    2. Belief alignment: d/d_mu_q [lambda * Sum_j beta_ij * KL(q_i || Omega_ij q_j)]

    The dBeta/dSigma softmax coupling term is always included:
        dF/dSigma_i = Sum_j beta_ij * dKL_ij/dSigma_i + Sum_j KL_ij * dBeta_ij/dSigma_i

    Dispatches to memory-efficient block-diagonal paths when irrep_dims is set:
    - irrep_dims + full cov  -> _compute_vfe_gradients_block_diagonal
    - irrep_dims + diagonal  -> _compute_vfe_gradients_block_diagonal_diag

    Args:
        mu_q: Belief means (B, N, K).
        sigma_q: Belief variances - diagonal (B, N, K) or full (B, N, K, K).
        mu_p: Prior means (B, N, K).
        sigma_p: Prior variances - diagonal (B, N, K) or full (B, N, K, K).
        beta: Attention weights (B, N, N), already normalized.
        phi: Gauge frames (B, N, phi_dim) where phi_dim = n_gen.
        generators: Lie algebra generators (n_gen, K, K).
        alpha: Weight for KL(q||p) self-coupling. Scalar (uniform) or (B, N, K)
            tensor from per-dimension Bayesian precision (learnable_alpha).
        lambda_belief: Weight for belief alignment term.
        kappa: Temperature parameter.
        eps: Numerical stability floor.
        alpha_c0: (K,) softplus(raw_c0) for product-rule correction when alpha
            is a learnable (B, N, K) tensor.
        cached_transport: Optional dict with precomputed 'Omega' (B, N, N, K, K)
            from compute_transport_operators(). Avoids redundant matrix exponentials.
        compute_sigma_align_grad: If True (default), include the sigma alignment
            gradient dKL/dSigma_q = 0.5 * (Sigma_transported^{-1} - Sigma_q^{-1}).
        irrep_dims: Block dimensions [d_1, ...] for block-diagonal KL decomposition.
            Reduces memory from O(N^2 K^2) to O(N^2 * max(d_i^2)).
        enforce_orthogonal: If True, enforce Omega in SO(K) via Newton-Schulz.
        cached_block_exp_pairs: Precomputed (exp_phi, exp_neg_phi) per irrep block.
        exact_diagonal_transport: When True and sigma is diagonal, lifts sigma to
            full covariance via diag_embed for exact gauge transport, then extracts
            the diagonal from the result. Disables fused diagonal paths.

    Returns:
        grad_mu: Gradient w.r.t. mu_q, shape (B, N, K).
        grad_sigma: Gradient w.r.t. sigma_q, shape (B, N, K) diagonal or
            (B, N, K, K) full, matching input.
    """
    # Squeeze trailing singleton dimensions for robustness
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)
    while sigma_p.dim() > 3 and sigma_p.shape[-1] == 1:
        sigma_p = sigma_p.squeeze(-1)

    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Detect diagonal vs full covariance
    is_diagonal = sigma_q.dim() == 3

    # Exact diagonal transport: lift to full, use full-cov path, extract diagonal
    if exact_diagonal_transport and is_diagonal:
        sigma_q_full = torch.diag_embed(sigma_q)
        sigma_p_full = torch.diag_embed(sigma_p)
        grad_mu, grad_sigma_full = compute_vfe_gradients_gpu(
            mu_q, sigma_q_full, mu_p, sigma_p_full, beta, phi, generators,
            alpha, lambda_belief, kappa, eps, alpha_c0,
            cached_transport, compute_sigma_align_grad, irrep_dims,
            enforce_orthogonal, cached_block_exp_pairs,
            exact_diagonal_transport=False,
        )
        return grad_mu, torch.diagonal(grad_sigma_full, dim1=-2, dim2=-1)

    # =================================================================
    # MEMORY-EFFICIENT PATH: Block-diagonal processing
    # =================================================================
    if irrep_dims is not None and not is_diagonal:
        return _compute_vfe_gradients_block_diagonal(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
            alpha, lambda_belief, lambda_softmax, kappa, eps, irrep_dims,
            compute_sigma_align_grad, enforce_orthogonal,
            alpha_c0=alpha_c0,
            cached_block_exp_pairs=cached_block_exp_pairs,
        )

    # =================================================================
    # MEMORY-EFFICIENT PATH: Block-diagonal for diagonal covariance
    # =================================================================
    if irrep_dims is not None and is_diagonal:
        return _compute_vfe_gradients_block_diagonal_diag(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
            alpha, lambda_belief, lambda_softmax, kappa, eps, irrep_dims,
            compute_sigma_align_grad, enforce_orthogonal,
            alpha_c0=alpha_c0,
            cached_block_exp_pairs=cached_block_exp_pairs,
        )

    # =================================================================
    # 1. Self-Coupling Gradient: ∂/∂μ_q [α · KL(q||p)]
    # =================================================================
    # For diagonal Gaussians:
    #   KL(q||p) = 0.5 * Σ_k [ σ_q[k]/σ_p[k] + (μ_p[k]-μ_q[k])²/σ_p[k] - 1 + log(σ_p[k]/σ_q[k]) ]
    #   ∂KL/∂μ_q = (μ_q - μ_p) / σ_p
    #   ∂KL/∂σ_q = 0.5 * (1/σ_p - 1/σ_q)

    if is_diagonal:
        # Force float32 for all sigma divisions, logs, and KL computation.
        # The Omega einsum and mu transport can stay in AMP dtype for speed,
        # but sigma ratios and log-det terms need float32 precision.
        _orig_dtype = sigma_q.dtype
        sigma_q = sigma_q.float()
        sigma_p = sigma_p.float()
        mu_q = mu_q.float()
        mu_p = mu_p.float()
        beta = beta.float()

        # Clamp for stability
        sigma_q_safe = sigma_q.clamp(min=eps)
        sigma_p_safe = sigma_p.clamp(min=eps)

        # Self-coupling gradient w.r.t. μ
        delta_mu = mu_q - mu_p  # (B, N, K)
        grad_mu_self = alpha * delta_mu / sigma_p_safe  # (B, N, K)

        # Self-coupling gradient w.r.t. σ (diagonal)
        grad_sigma_self = alpha * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)  # (B, N, K)

        # Product-rule correction: ∂(α·KL)/∂θ = α·∂KL/∂θ + (∂α/∂θ)·KL
        # When α_k = c₀_k/(b₀_k + kl_k), ∂α_k/∂θ = -α_k²/c₀_k · ∂kl_k/∂θ
        if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
            kl_k = 0.5 * (sigma_q_safe / sigma_p_safe + delta_mu ** 2 / sigma_p_safe
                          - 1.0 + torch.log(sigma_p_safe) - torch.log(sigma_q_safe))
            kl_k = kl_k.clamp(min=0.0)
            # ∂α/∂μ · KL = -α²/c₀ · (δμ/σ_p) · kl_k
            grad_mu_self = grad_mu_self - (alpha ** 2 / alpha_c0) * kl_k * delta_mu / sigma_p_safe
            # ∂α/∂σ_q · KL = -α²/c₀ · 0.5·(1/σ_p - 1/σ_q) · kl_k
            grad_sigma_self = grad_sigma_self - (alpha ** 2 / alpha_c0) * kl_k * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)
    else:
        # Full covariance - use matrix operations
        # ∂KL/∂μ_q = Σ_p^{-1} (μ_q - μ_p)
        sigma_p_inv = _safe_spd_inv(sigma_p, eps=eps)

        delta_mu = mu_q - mu_p  # (B, N, K)
        grad_mu_self = alpha * torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)

        # ∂KL/∂Σ_q = 0.5 * (Σ_p^{-1} - Σ_q^{-1})
        sigma_q_inv = _safe_spd_inv(sigma_q, eps=eps)
        # For full covariance (4D), alpha (B,N,K) needs extra dim to broadcast with (B,N,K,K)
        alpha_4d = alpha.unsqueeze(-1) if isinstance(alpha, torch.Tensor) else alpha
        grad_sigma_self = alpha_4d * 0.5 * (sigma_p_inv - sigma_q_inv)

        # Product-rule correction for learnable alpha (full covariance):
        # ∂(α·KL)/∂θ = α·∂KL/∂θ + (∂α/∂θ)·KL
        # Per-dimension KL via eigendecomposition of Σ_p^{-1/2} Σ_q Σ_p^{-1/2},
        # which is symmetric with eigenvalues λ_k giving per-mode contributions
        # kl_mode_k = 0.5(λ_k - 1 - log λ_k). Projected back to original basis
        # via eigenvector magnitudes.
        if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
            sp_inv_delta = torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)
            mahal_k = delta_mu * sp_inv_delta  # (B, N, K) — per-dim Mahalanobis

            # Per-mode KL from eigendecomposition of L^{-1} Σ_q L^{-T}
            # where L = cholesky(Σ_p). Eigenvalues λ_k of this symmetric matrix
            # equal eigenvalues of Σ_p^{-1} Σ_q.
            try:
                L_p = torch.linalg.cholesky(sigma_p.float())
                # M = L^{-1} Σ_q L^{-T} (symmetric, same eigenvalues as Σ_p^{-1} Σ_q)
                Lp_inv_Sq = torch.linalg.solve_triangular(L_p, sigma_q.float(), upper=False)
                M = torch.linalg.solve_triangular(
                    L_p, Lp_inv_Sq.transpose(-1, -2), upper=False
                ).transpose(-1, -2)
                eigvals, eigvecs = torch.linalg.eigh(M)  # (B,N,K), (B,N,K,K)
                eigvals = eigvals.clamp(min=eps).to(sigma_q.dtype)
                eigvecs = eigvecs.to(sigma_q.dtype)
                # Per-eigenmode KL: kl_mode_k = 0.5(λ_k - 1 - log λ_k) ≥ 0
                kl_mode = 0.5 * (eigvals - 1.0 - torch.log(eigvals))  # (B, N, K)
                # Project to original basis via L^{-T} V: kl_orig_k = Σ_m |V_km|² kl_mode_m
                # eigvecs are in the L^{-1} basis; |V_km|² distributes modes to original dims
                V_sq = eigvecs ** 2  # (B, N, K, K) — squared eigenvector components
                kl_k_trace_logdet = torch.einsum('bnkm,bnm->bnk', V_sq, kl_mode)
                kl_k = (kl_k_trace_logdet + 0.5 * mahal_k).clamp(min=0.0)
            except RuntimeError:
                # Fallback: uniform logdet distribution (original approximation)
                prod_qp = torch.matmul(sigma_p_inv, sigma_q)
                trace_k = prod_qp.diagonal(dim1=-2, dim2=-1)
                logdet_p = torch.linalg.slogdet(sigma_p.float())[1]
                logdet_q = torch.linalg.slogdet(sigma_q.float())[1]
                logdet_k = ((logdet_p - logdet_q) / K).unsqueeze(-1).expand_as(delta_mu)
                kl_k = 0.5 * (trace_k + mahal_k - 1 + logdet_k).clamp(min=0.0)

            grad_mu_self = grad_mu_self - (alpha ** 2 / alpha_c0) * kl_k * sp_inv_delta
            correction_scale = ((alpha ** 2 / alpha_c0) * kl_k).unsqueeze(-1)  # (B, N, K, 1)
            grad_sigma_self = grad_sigma_self - correction_scale * 0.5 * (sigma_p_inv - sigma_q_inv)

    # =================================================================
    # 2. Belief Alignment Gradient: ∂/∂μ_i [λ · Σ_j β_ij · KL(q_i || Ω_ij q_j)]
    # =================================================================
    # Full gradient via product rule:
    #   ∂/∂μ_i [Σ_j β_ij · KL_ij] = Σ_j β_ij · ∂KL_ij/∂μ_i + Σ_j KL_ij · ∂β_ij/∂μ_i
    #                                  ↑ direct term           ↑ SOFTMAX COUPLING (nonlinearity!)
    #
    # The softmax coupling gradient is the KEY nonlinearity (replaces GELU/ReLU):
    #   ∂β_ij/∂μ_i = -β_ij · [∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i] / κ

    if is_diagonal:
        # Get transport operators (use cached if available)
        if cached_transport is not None and 'Omega' in cached_transport:
            Omega = cached_transport['Omega']
        else:
            # Compute transport operators (vectorized)
            phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)  # (B, N, K, K)
            exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)

            # Re-orthogonalization for SO(K) if requested
            if enforce_orthogonal and K >= 16:
                exp_phi = newton_schulz_orthogonalize(exp_phi)
                exp_neg_phi = newton_schulz_orthogonalize(exp_neg_phi)

            # Transport: Ω_ij = exp(φ_i) @ exp(-φ_j)
            # For all pairs: (B, N, N, K, K)
            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

        # Transport all μ_j to frame i: μ_j_transported[i,j] = Ω_ij @ μ_j
        mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)  # (B, N, N, K)

        # Difference: μ_i - μ_j_transported (for each pair i,j)
        delta_mu_ij = mu_q.unsqueeze(2) - mu_j_transported  # (B, N, N, K)

        # =================================================================
        # DIAGONAL COVARIANCE TRANSPORT: diag(Ω @ diag(σ_j) @ Ω^T)
        # =================================================================
        # For diagonal input, the diagonal of the transported covariance is:
        #   σ_j_transported[k] = Σ_l Ω_kl² * σ_j[l]
        # This avoids materializing any (B, N, N, K, K) covariance tensors.

        sigma_q_safe = sigma_q.clamp(min=eps)  # (B, N, K)

        # Transported diagonal covariance via 3-operand einsum (no Omega² intermediate)
        sigma_j_transported_diag = torch.einsum(
            'bijkl,bijkl,bjl->bijk', Omega, Omega, sigma_q_safe
        ).clamp(min=eps)  # (B, N, N, K)

        # ∂KL_ij/∂μ_i = (μ_i - μ_j_transported) / σ_j_transported (element-wise for diagonal)
        grad_kl_per_pair = delta_mu_ij / sigma_j_transported_diag  # (B, N, N, K)

        # Compute FULL KL values using diagonal formulas (no Cholesky/inv needed!)
        # KL(N(μ_i,diag(σ_i)) || N(μ_j_t,diag(σ_j_t))) =
        #   0.5 * (Σ_k σ_i[k]/σ_j_t[k] + Σ_k (μ_i-μ_j_t)²[k]/σ_j_t[k] - K + Σ_k log(σ_j_t[k]/σ_i[k]))

        sigma_i_expanded = sigma_q_safe[:, :, None, :]  # (B, N, 1, K) - broadcasts

        # Trace term: Σ_k σ_i[k] / σ_j_transported[k]
        trace_term = (sigma_i_expanded / sigma_j_transported_diag).sum(dim=-1)  # (B, N, N)

        # Mahalanobis term: Σ_k (μ_i - μ_j_transported)² / σ_j_transported[k]
        mahal_term = (delta_mu_ij ** 2 / sigma_j_transported_diag).sum(dim=-1)  # (B, N, N)

        # Log-determinant term: Σ_k log(σ_j_transported[k]) - Σ_k log(σ_i[k])
        logdet_term = (torch.log(sigma_j_transported_diag.clamp(min=eps)) - torch.log(sigma_i_expanded.clamp(min=eps))).sum(dim=-1)  # (B, N, N)

        # Full KL divergence
        kl_values = 0.5 * (trace_term + mahal_term - K + logdet_term)
        # Clamp KL to [0, max] for numerical stability (scale ceiling with K)
        kl_ceil = max(100.0, 20.0 * K)
        kl_values = kl_values.clamp(min=0.0, max=kl_ceil)  # (B, N, N)

        # =================================================================
        # 2a. Direct term: Σ_j β_ij · ∂KL_ij/∂μ_i
        # =================================================================
        # avg_grad = Σ_k β_ik · ∂KL_ik/∂μ_i (used for both direct and softmax terms)
        avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)  # (B, N, K)
        grad_mu_direct = lambda_belief * avg_grad

        # =================================================================
        # 2b. Softmax coupling term (THE NONLINEARITY!):
        #     ∂β_ij/∂μ_i = -β_ij · [∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i] / κ
        #     grad_softmax = Σ_j KL_ij · ∂β_ij/∂μ_i
        # =================================================================
        # Deviation from average: avg_grad_i - ∂KL_ij/∂μ_i
        # β_ij = softmax(-KL/κ), so ∂β_ij/∂μ_i = (β_ij/κ)(avg_grad - ∂KL_ij/∂μ_i)
        # grad_kl_per_pair: (B, N, N, K), avg_grad: (B, N, K) -> expand to (B, N, 1, K)
        grad_deviation = avg_grad.unsqueeze(2) - grad_kl_per_pair  # (B, N, N, K)

        # Softmax coupling gradient: ∂β_ij/∂μ_i = β_ij · grad_deviation / κ
        # Scale kappa by √K to match attention temperature scaling (τ = √K)
        kappa_scaled = max(kappa * math.sqrt(max(K, 1)), eps)
        d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa_scaled  # (B, N, N, K)

        # Weight by KL values and sum: Σ_j KL_ij · ∂β_ij/∂μ_i
        grad_mu_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

        # Total alignment gradient (direct + softmax coupling)
        grad_mu_align = grad_mu_direct + grad_mu_softmax

        # =================================================================
        # Sigma gradient from alignment term (diagonal case)
        # ∂KL/∂σ_i[k] = 0.5 * (1/σ_j_transported[k] - 1/σ_i[k])
        # Weighted by attention: Σ_j β_ij * ∂KL_ij/∂σ_i
        # =================================================================
        if compute_sigma_align_grad:
            sigma_j_inv_diag = 1.0 / sigma_j_transported_diag  # (B, N, N, K)
            sigma_i_inv = 1.0 / sigma_q_safe  # (B, N, K)
            # Gradient per pair: 0.5 * (1/σ_j_transported[k] - 1/σ_i[k])
            grad_sigma_per_pair = 0.5 * (sigma_j_inv_diag - sigma_i_inv[:, :, None, :])  # broadcast

            # Direct term: Σ_j β_ij * ∂KL_ij/∂σ_i
            grad_sigma_direct = lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_sigma_per_pair)  # (B, N, K)

            # Softmax coupling: Σ_j KL_ij · ∂β_ij/∂σ_i (analogous to μ coupling)
            # ∂β_ij/∂σ_i = -β_ij/κ · [∂KL_ij/∂σ_i - Σ_k β_ik · ∂KL_ik/∂σ_i]
            #            = β_ij/κ · [avg_sigma_grad - ∂KL_ij/∂σ_i]
            avg_sigma_grad = torch.einsum('bij,bijk->bik', beta, grad_sigma_per_pair)  # (B, N, K)
            sigma_grad_deviation = avg_sigma_grad.unsqueeze(2) - grad_sigma_per_pair  # (B, N, N, K)
            d_beta_d_sigma = beta.unsqueeze(-1) * sigma_grad_deviation / kappa_scaled  # (B, N, N, K)
            grad_sigma_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_sigma)  # (B, N, K)
            grad_sigma_align = grad_sigma_direct + grad_sigma_softmax
        else:
            # Simplified: no sigma gradient from alignment (legacy behavior)
            grad_sigma_align = torch.zeros_like(sigma_q)
    else:
        # Full covariance belief alignment
        # Get transport operators (use cached if available)
        if cached_transport is not None and 'Omega' in cached_transport:
            Omega = cached_transport['Omega']
        else:
            phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
            exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)

            # Re-orthogonalization for SO(K) if requested
            if enforce_orthogonal and K >= 16:
                exp_phi = newton_schulz_orthogonalize(exp_phi)
                exp_neg_phi = newton_schulz_orthogonalize(exp_neg_phi)

            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

        # Transport means
        mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)
        delta_mu_ij = mu_q.unsqueeze(2) - mu_j_transported

        # Transport covariances: Σ_j_transported = Ω @ Σ_j @ Ω^T
        sigma_j_transported = torch.einsum(
            'bijkl,bjlm,bijmn->bijkn',
            Omega, sigma_q, Omega.transpose(-1, -2)
        )  # (B, N, N, K, K)

        # Regularize and invert
        sigma_j_reg = sigma_j_transported + max(eps, 1e-4) * torch.eye(K, device=device, dtype=dtype)
        # NaN safety: if transport produced NaN, replace with identity covariance
        if torch.isnan(sigma_j_reg).any():
            nan_mask = torch.isnan(sigma_j_reg).any(dim=-1).any(dim=-1)  # (B, N, N)
            eye_K = torch.eye(K, device=device, dtype=dtype)
            sigma_j_reg = torch.where(
                nan_mask.unsqueeze(-1).unsqueeze(-1),
                eye_K.expand_as(sigma_j_reg),
                sigma_j_reg,
            )
        sigma_j_inv = _safe_spd_inv(sigma_j_reg, eps=eps)  # (B, N, N, K, K)

        # ∂KL_ij/∂μ_i
        grad_kl_per_pair = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_ij)

        # Compute FULL KL values (not just Mahalanobis - include trace and logdet!)
        # KL(q_i || Ω_ij[q_j]) = 0.5 * (tr(Σ_j_t^{-1} Σ_i) + mahal - K + log|Σ_j_t| - log|Σ_i|)

        # Mahalanobis term: δμ^T @ Σ_j_transported^{-1} @ δμ
        mahal_term = torch.einsum('bijk,bijk->bij', delta_mu_ij, grad_kl_per_pair)  # (B, N, N)

        # Trace term: tr(Σ_j_transported^{-1} @ Σ_i)
        # Use .clone() after expand for contiguous memory layout
        sigma_i_expanded = sigma_q[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()  # (B, N, N, K, K)
        trace_term = torch.einsum('bijkk->bij', torch.einsum('bijkl,bijlm->bijkm', sigma_j_inv, sigma_i_expanded))

        # Log-determinant terms using Cholesky with fallback
        try:
            L_j_t = torch.linalg.cholesky(sigma_j_reg)  # (B, N, N, K, K)
            logdet_j_t = 2.0 * torch.sum(torch.log(torch.diagonal(L_j_t, dim1=-2, dim2=-1) + eps), dim=-1)  # (B, N, N)
        except RuntimeError:
            eigvals = torch.linalg.eigvalsh(sigma_j_reg)
            logdet_j_t = torch.sum(torch.log(eigvals.clamp(min=eps)), dim=-1)

        sigma_i_reg = sigma_q + eps * torch.eye(K, device=device, dtype=dtype)
        try:
            L_i = torch.linalg.cholesky(sigma_i_reg)  # (B, N, K, K)
            logdet_i = 2.0 * torch.sum(torch.log(torch.diagonal(L_i, dim1=-2, dim2=-1) + eps), dim=-1)  # (B, N)
        except RuntimeError:
            eigvals = torch.linalg.eigvalsh(sigma_i_reg)
            logdet_i = torch.sum(torch.log(eigvals.clamp(min=eps)), dim=-1)
        # Use .clone() after expand for contiguous memory layout
        logdet_i_expanded = logdet_i[:, :, None].expand(-1, -1, N).clone()  # (B, N, N)

        # Full KL divergence
        kl_values = 0.5 * (trace_term + mahal_term - K + logdet_j_t - logdet_i_expanded)
        # Clamp KL to [0, max] for numerical stability (scale ceiling with K)
        kl_ceil = max(100.0, 20.0 * K)
        kl_values = kl_values.clamp(min=0.0, max=kl_ceil)  # (B, N, N)

        # avg_grad = Σ_k β_ik · ∂KL_ik/∂μ_i (used for both direct and softmax terms)
        avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)
        grad_mu_direct = lambda_belief * avg_grad

        # Softmax coupling term
        # Scale kappa by √K to match attention temperature scaling (τ = √K)
        kappa_scaled = max(kappa * math.sqrt(max(K, 1)), eps)
        grad_deviation = avg_grad.unsqueeze(2) - grad_kl_per_pair
        d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa_scaled
        grad_mu_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

        grad_mu_align = grad_mu_direct + grad_mu_softmax

        # =================================================================
        # Sigma gradient from alignment term (full covariance case)
        # ∂KL/∂Σ_i = 0.5 * (Σ_j_transported^{-1} - Σ_i^{-1})
        # Weighted by attention: Σ_j β_ij * ∂KL_ij/∂Σ_i
        # =================================================================
        if compute_sigma_align_grad:
            # Use Σ_i^{-1} computed earlier in self-coupling section (sigma_q_inv)
            # Use .clone() after expand for contiguous memory layout
            sigma_i_inv_expanded = sigma_q_inv[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()  # (B, N, N, K, K)

            # Gradient per pair: 0.5 * (Σ_j_transported^{-1} - Σ_i^{-1})
            grad_sigma_per_pair = 0.5 * (sigma_j_inv - sigma_i_inv_expanded)  # (B, N, N, K, K)

            # Direct term: Σ_j β_ij * ∂KL_ij/∂Σ_i
            grad_sigma_direct = lambda_belief * torch.einsum('bij,bijkl->bikl', beta, grad_sigma_per_pair)  # (B, N, K, K)

            # Softmax coupling for full covariance (always enabled)
            avg_sigma_grad = torch.einsum('bij,bijkl->bikl', beta, grad_sigma_per_pair)  # (B, N, K, K)
            sigma_grad_deviation = avg_sigma_grad.unsqueeze(2) - grad_sigma_per_pair  # (B, N, N, K, K)
            d_beta_d_sigma = beta.unsqueeze(-1).unsqueeze(-1) * sigma_grad_deviation / kappa_scaled  # (B, N, N, K, K)
            grad_sigma_softmax = lambda_softmax * torch.einsum('bij,bijkl->bikl', kl_values, d_beta_d_sigma)  # (B, N, K, K)
            grad_sigma_align = grad_sigma_direct + grad_sigma_softmax
        else:
            # Simplified: no sigma gradient from alignment (legacy behavior)
            grad_sigma_align = torch.zeros_like(sigma_q)

    # =================================================================
    # 3. Combine Gradients
    # =================================================================
    grad_mu = grad_mu_self + grad_mu_align
    grad_sigma = grad_sigma_self + grad_sigma_align

    # Cast back from float32 (diagonal path upcasts for numerical safety under AMP)
    return grad_mu.to(dtype), grad_sigma.to(dtype)


def compute_natural_gradient_gpu(
    grad_mu: torch.Tensor,     # (B, N, K) Euclidean gradient
    grad_sigma: torch.Tensor,  # (B, N, K) or (B, N, K, K)
    sigma_q: torch.Tensor,     # (B, N, K) or (B, N, K, K)
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project Euclidean gradients to natural gradients using Fisher metric.

    For Gaussian distributions, the Fisher information metric is:
        F_μ = Σ^{-1}  →  natural_grad_μ = Σ @ euclidean_grad_μ
        F_σ = (1/2)Σ^{-2} →  natural_grad_σ = 2 * Σ² @ euclidean_grad_σ (diagonal approx)

    Derivation: The Fisher metric on the covariance Σ of a Gaussian is
    g(δΣ₁, δΣ₂) = (1/2) tr(Σ⁻¹ δΣ₁ Σ⁻¹ δΣ₂). For diagonal Σ = diag(σ),
    this gives g_{kk} = 1/(2σ_k²), so g^{kk} = 2σ_k².

    Args:
        grad_mu: Euclidean gradient w.r.t. μ
        grad_sigma: Euclidean gradient w.r.t. σ
        sigma_q: Current covariance
        eps: Numerical stability

    Returns:
        nat_grad_mu: Natural gradient for μ
        nat_grad_sigma: Natural gradient for σ
    """
    # Squeeze trailing singleton dimensions for robustness
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)

    is_diagonal = sigma_q.dim() == 3
    orig_dtype = sigma_q.dtype

    # Force float32: sigma^2 products and small sigma divisions break in float16
    with torch.amp.autocast('cuda', enabled=False):
        sigma_q = sigma_q.float()
        grad_mu = grad_mu.float()
        grad_sigma = grad_sigma.float()

        if is_diagonal:
            # Diagonal case: simple element-wise multiplication
            sigma_safe = sigma_q.clamp(min=eps)
            nat_grad_mu = sigma_safe * grad_mu  # (B, N, K)
            nat_grad_sigma = 2.0 * sigma_safe * sigma_safe * grad_sigma  # (B, N, K)
        else:
            # Full covariance: matrix multiplication
            nat_grad_mu = torch.einsum('bnij,bnj->bni', sigma_q, grad_mu)
            # Full Fisher natural gradient: δΣ = 2 * Σ @ ∇_Σ @ Σ
            nat_grad_sigma = 2.0 * torch.einsum('bnij,bnjk,bnkl->bnil', sigma_q, grad_sigma, sigma_q)

    return nat_grad_mu.to(orig_dtype), nat_grad_sigma.to(orig_dtype)


# =============================================================================
# SPD Retraction (PyTorch GPU version)
# =============================================================================

def retract_spd_torch(
    Sigma: torch.Tensor,
    delta_Sigma: torch.Tensor,
    step_size: float = 1.0,
    trust_region: float = 2.0,
    eps: float = 1e-6,
    sigma_max: float = 5.0,
) -> torch.Tensor:
    """
    SPD-preserving retraction for covariance matrices (PyTorch GPU).

    Uses affine-invariant exponential map on SPD manifold.

    Args:
        Sigma: SPD matrices, shape (B, N, K, K) or (B*N, K, K)
        delta_Sigma: Symmetric tangent vectors, same shape as Sigma
        step_size: Learning rate τ
        trust_region: Max Frobenius norm of whitened tangent
        eps: Regularization floor for numerical stability
        sigma_max: Upper bound on eigenvalues (sigma_max² for covariance eigenvalues)

    Returns:
        Sigma_new: SPD matrices, same shape as Sigma
    """
    # Handle different input shapes
    original_shape = Sigma.shape
    orig_dtype = Sigma.dtype
    if Sigma.dim() == 4:
        B, N, K, _ = Sigma.shape
        Sigma = Sigma.reshape(B * N, K, K)
        delta_Sigma = delta_Sigma.reshape(B * N, K, K)

    batch_size, K, _ = Sigma.shape
    device = Sigma.device

    # Force float32 for eigendecomposition, sqrt, exp — all break in float16
    with torch.amp.autocast('cuda', enabled=False):
        Sigma = Sigma.float()
        delta_Sigma = delta_Sigma.float()

        # Symmetrize inputs (numerical safety)
        Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))
        delta_Sigma = 0.5 * (delta_Sigma + delta_Sigma.transpose(-1, -2))

        # Exponential map retraction on SPD manifold
        eigenvalues, eigenvectors = _safe_eigh(Sigma, jitter=eps, symmetrize=False)
        eigenvalues = eigenvalues.clamp(min=eps)

        sqrt_eig = torch.sqrt(eigenvalues)
        inv_sqrt_eig = 1.0 / sqrt_eig

        Sigma_sqrt = eigenvectors * sqrt_eig.unsqueeze(-2) @ eigenvectors.transpose(-1, -2)
        Sigma_inv_sqrt = eigenvectors * inv_sqrt_eig.unsqueeze(-2) @ eigenvectors.transpose(-1, -2)

        # Whitened tangent: R = Σ^{-1/2} (η · ΔΣ) Σ^{-1/2}
        R = Sigma_inv_sqrt @ (step_size * delta_Sigma) @ Sigma_inv_sqrt
        R = 0.5 * (R + R.transpose(-1, -2))

        if trust_region is not None and trust_region > 0:
            R_norm = torch.linalg.norm(R, ord='fro', dim=(-2, -1), keepdim=True)
            scale = torch.clamp(trust_region / (R_norm + eps), max=1.0)
            R = R * scale

        # exp(R) via eigendecomposition
        R_eigenvalues, R_eigenvectors = _safe_eigh(R, jitter=eps, symmetrize=False)
        R_eigenvalues = R_eigenvalues.clamp(-50.0, 50.0)
        exp_R = R_eigenvectors * torch.exp(R_eigenvalues).unsqueeze(-2) @ R_eigenvectors.transpose(-1, -2)

        Sigma_new = Sigma_sqrt @ exp_R @ Sigma_sqrt
        Sigma_new = 0.5 * (Sigma_new + Sigma_new.transpose(-1, -2))

        # Spectral floor + ceiling: eigenvalues in [eps, sigma_max²]
        # sigma_max bounds the standard deviation; covariance eigenvalues are σ².
        eig_new, vec_new = _safe_eigh(Sigma_new, jitter=eps, symmetrize=False)
        eig_new = eig_new.clamp(min=eps, max=sigma_max * sigma_max)
        Sigma_new = vec_new * eig_new.unsqueeze(-2) @ vec_new.transpose(-1, -2)

    Sigma_new = Sigma_new.to(orig_dtype)

    # Restore original shape
    if len(original_shape) == 4:
        Sigma_new = Sigma_new.reshape(original_shape)

    return Sigma_new


def retract_spd_diagonal_torch(
    sigma_diag: torch.Tensor,
    delta_sigma: torch.Tensor,
    step_size: float = 1.0,
    trust_region: float = 5.0,
    eps: float = 1e-6,
    sigma_max: float = 5.0,
) -> torch.Tensor:
    """
    SPD retraction for diagonal covariances (much simpler).

    For diagonal matrices, the exponential map reduces to:
        σ_new = σ * exp(τ * δσ / σ)

    This ensures positivity: exp(x) > 0 for all x.

    Args:
        sigma_diag: Diagonal variances, shape (B, N, K)
        delta_sigma: Tangent in diagonal form, shape (B, N, K)
        step_size: Learning rate τ
        trust_region: Max absolute value of exponent argument
        eps: Floor for sigma values
        sigma_max: Upper bound on σ. Posterior σ should not greatly exceed the prior.
            Prevents nat_grad_sigma = 2σ²·∇σ blowup (amplification grows as σ⁴).

    Returns:
        sigma_new: Positive diagonal variances, shape (B, N, K)
    """
    orig_dtype = sigma_diag.dtype

    # Force float32: exp(50) overflows float16 (max ~65504)
    with torch.amp.autocast('cuda', enabled=False):
        sigma_safe = sigma_diag.float().clamp(min=eps)
        delta_sigma = delta_sigma.float()

        # Whitened tangent: δσ / σ (element-wise for diagonal)
        whitened = delta_sigma / sigma_safe

        # Trust region on whitened tangent
        if trust_region is not None and trust_region > 0:
            whitened = whitened.clamp(-trust_region, trust_region)

        # Exponential update: σ_new = σ * exp(τ * whitened)
        # Clip exponent to prevent overflow
        exp_arg = (step_size * whitened).clamp(-50.0, 50.0)
        sigma_new = sigma_safe * torch.exp(exp_arg)

    # Clamp to [eps, sigma_max] — posterior σ should not blow up past the prior.
    # With sigma_max=5.0 and init_sigma_scale=1.0, allows 5× expansion before clamping.
    # This bounds the natural gradient amplification: 2σ² ≤ 2·sigma_max² = 50.
    return sigma_new.clamp(min=eps, max=sigma_max).to(orig_dtype)


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
        lambda_belief: float = 1.0,  # Belief alignment weight (direct: β·∇KL)
        lambda_softmax: float = 1.0,  # Softmax coupling weight (GELU-like ∂β/∂θ · KL)
        kappa: float = 1.0,        # Attention temperature
        n_iterations: int = 1,    # VFE descent steps (more steps = deeper equilibration)
        learnable_lr: bool = True, # Learn step size?
        mu_lr: float = 0.1,           # E-step μ step size (used when learnable_lr=False)
        sigma_lr: float = 0.001,      # E-step σ trust region scale (used when learnable_lr=False)
        update_sigma: bool = True, # Update covariances?
        diagonal_covariance: bool = False,  # Use diagonal Σ for efficiency
        compute_sigma_align_grad: bool = True,  # Compute sigma gradient from alignment term
        # Phi (gauge frame) evolution via VFE gradients
        update_phi: bool = True,  # If True, update phi via ∂F/∂φ (after E-step loop)
        update_phi_per_iteration: bool = True,  # If True, update phi during EACH E-step iteration
        phi_lr: float = 0.05,      # Learning rate for phi updates
        phi_max_norm: Optional[float] = None,  # Max phi norm; None = auto (π for SO(N), 5.0 for GL(K))
        prior_bank: Optional[nn.Module] = None,  # Token-dependent PriorBank (if provided)
        use_prior_bank: bool = False,  # If True, use PriorBank (token-dependent) instead of position-dependent priors
        # Memory-efficient options (NEW!)
        irrep_dims: Optional[List[int]] = None,  # Block dimensions for principled KL decomposition
        # Self-attention masking (prevents attention collapse)
        mask_self_attention: bool = False,  # If True, mask out diagonal (no self-attention)
        # Bayesian precision (learned prior self-coupling)
        learnable_alpha: bool = False,  # If True, use Gamma-Normal conjugate precision
        # Multi-head VFE: maintain per-head β through iterations
        multihead_vfe: bool = True,  # If True, compute separate β_h per irrep block
        # Phi gradient preconditioning mode
        phi_natural_gradient: str = 'killing',  # 'clip'|'cartan'|'killing'|'pullback'
        killing_center_reg: Optional[float] = None,  # Killing form center regularization (None=2K)
        # DEQ implicit differentiation
        use_deq: bool = False,                # Use DEQ backward for E-step fixed point
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
        amortized_inference: bool = True,
        # Rotary Position Embeddings (RoPE) — must match attention sublayer setting
        use_rope: bool = True,
        rope_base: float = 10000.0,
        exact_diagonal_transport: bool = False,  # Lift diagonal σ for exact transport
        gauge_param: str = 'phi',  # 'phi' (Lie algebra) or 'omega' (direct GL(K))
        obs_sigma_gradient: bool = True,  # ∂E_q[CE]/∂σ via Hessian diagonal of expected CE
        obs_sigma_weight: float = 1.0,     # Weight for sigma observation gradient
        sigma_max: float = 5.0,            # Upper bound on σ (prevents nat_grad blowup from 2σ²·∇σ)
        e_step_sigma_floor: float = 0.1,  # Floor on σ_p inside E-step (caps 1/σ_p at 1/floor)
        detach_phi: bool = False,          # Detach phi from backprop in non-amortized mode
                                           # (enables fully backprop-free training with phi P-flow)
        deq_include_phi: bool = False,     # Include phi in DEQ fixed-point variables.
                                           # When True, the Neumann-series IFT correction applies
                                           # to the joint (mu, sigma, phi) fixed point, giving the
                                           # exact M-step phi gradient instead of straight-through.
                                           # Requires use_deq=True and evolve_phi=True.
        closed_form_e_step: bool = False,  # Use closed-form precision-weighted fixed point
                                           # instead of gradient descent. Diagonal path uses
                                           # the enhanced form that absorbs softmax coupling
                                           # (S, c terms); full-cov uses linear-only CF.
        implicit_em: bool = False,         # Principled M-step via implicit differentiation.
                                           # Detaches mu/sigma at E-step start (proper EM boundary)
                                           # then scales CE→embedding gradient by IFT factor
                                           # s_k = (α/σ²_p) / (α/σ²_p + Σβ/σ²_q) ∈ [0,1].
                                           # Replaces ad-hoc straight-through (s=1) and pure EM (s=0)
                                           # with info-geometrically correct value.
        learnable_head_kappa: bool = False,  # If True, learn per-head κ_h
        n_picard_steps: int = 0,           # Re-solve iterations (diagonal) or Picard steps (full-cov)
        picard_trust_region: float = 5.0,  # Whitened trust region for Picard steps
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
        self.implicit_em = implicit_em
        self._last_implicit_mu_scale = None   # (B, N, K) stored after E-step for model.py
        self._last_implicit_sigma_scale = None
        if implicit_em:
            print(f"[VariationalFFNDynamic] Implicit EM enabled: IFT-based M-step gradient")
            print(f"  → Detaches mu/sigma at E-step start, applies s_k = (α/σ²_p)/A_k scaling")
        # RoPE: DISABLED in VFE E-step iterations. The E-step is belief refinement
        # (analogous to V aggregation), not attention scoring (Q·K). Position
        # awareness enters via β_ij from the attention sublayer, which already
        # applies RoPE. Applying RoPE inside the E-step double-counts position,
        # breaks KL geometry (rotates μ but not Σ), and distorts the VFE fixed point.
        self._use_rope_vfe = False
        self._rope_base_vfe = rope_base
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
            print(f"[VariationalFFNDynamic] φ will evolve DURING E-step iterations (dynamical gauge frames)")
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
                print(f"[VariationalFFNDynamic] φ preconditioning: Cartan (sym_dampening=0.1)")
            elif phi_natural_gradient == 'killing':
                self._phi_preconditioner = build_killing_form_preconditioner(
                    generators, center_reg=killing_center_reg,
                )
                _cr = killing_center_reg if killing_center_reg is not None else 2.0 * generators.shape[-1]
                print(f"[VariationalFFNDynamic] φ preconditioning: Killing form natural gradient (center_reg={_cr:.1f})")
            elif phi_natural_gradient == 'pullback':
                self._structure_constants = build_structure_constants(generators)
                # Frobenius inner product: <T_a, T_b> = tr(T_a^T T_b)
                self._gram = torch.einsum('aij,bij->ab', generators, generators)
                print(f"[VariationalFFNDynamic] φ preconditioning: pullback natural gradient (exact)")

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
                print(f"[VariationalFFNDynamic] Multi-head VFE: {n_heads} heads with learnable per-head κ (init from κ={kappa})")
            else:
                print(f"[VariationalFFNDynamic] Multi-head VFE: {n_heads} heads with shared κ={kappa}")

        # =================================================================
        # Per-head learnable temperature κ_h (shared concept with attention)
        # =================================================================
        if learnable_head_kappa and irrep_dims is not None:
            init_kappas = torch.tensor([
                kappa * math.sqrt(d_h) for d_h in irrep_dims
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
            print(f"[VariationalFFNDynamic] Bayesian precision enabled (per-dim): "
                  f"c₀={c0_init:.4f}, b₀={b0_init:.1f}, "
                  f"initial α≈{alpha_init} (K={embed_dim})")

        # PriorBank integration
        self.use_prior_bank = use_prior_bank
        self.prior_bank = prior_bank

        if use_prior_bank and prior_bank is not None:
            self.prior_bank = prior_bank
            print(f"[VariationalFFNDynamic] Using PriorBank with token-dependent priors (vocab_size={prior_bank.vocab_size})")
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

    @property
    def lr(self) -> torch.Tensor:
        """E-step μ learning rate, constrained to (0, ∞) via softplus."""
        return F.softplus(self.raw_lr)

    def _get_kappa_h(self, head_idx: int, d_h: int):
        r"""Get per-head temperature κ_h.

        When learnable_head_kappa=True: κ_h = exp(log_kappa_per_head[h])
        When False: κ_h = self.kappa * √d_h (static scaling)
        """
        if self.learnable_head_kappa and self.log_kappa_per_head is not None:
            kappa_h = torch.exp(self.log_kappa_per_head[head_idx])
            # Clamp to [0.5, 1.5] × init, matching attention module (attention.py:2688)
            k0 = self._kappa_init[head_idx]
            return kappa_h.clamp(min=0.5 * k0, max=1.5 * k0)
        return self.kappa * math.sqrt(d_h)

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

        # Initialize sigma if not provided
        if sigma is None:
            if self.diagonal_covariance:
                sigma = torch.ones(B, N, K, device=device, dtype=dtype) * 0.1
            else:
                sigma = 0.1 * torch.eye(K, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()

        # Squeeze trailing singleton dimensions for robustness
        while sigma.dim() > 3 and sigma.shape[-1] == 1:
            sigma = sigma.squeeze(-1)

        is_diagonal = sigma.dim() == 3

        # =====================================================================
        # PriorBank: Use token-dependent priors for VFE dynamics
        # =====================================================================
        # =====================================================================
        # Prior Setup: mu_p and sigma_p for VFE self-coupling
        # =====================================================================
        # Use mu_prior / sigma_prior passed from model.py for BOTH PriorBank
        # and standard embedding paths.  These are already in the correct
        # coordinate frame (cross_head_perm applied).  The FFN must NOT
        # re-encode from PriorBank — that returns un-permuted priors which
        # would misalign with the permuted beliefs in the VFE loop.
        #
        # mu_p: live when amortized (gradient is well-conditioned: -α/σ_p).
        # Detached when non-amortized or implicit_em.
        # sigma_p: ALWAYS detached (M-step parameter; 1/σ_p in E-step
        # creates positive feedback).
        if self.amortized_inference and not self.implicit_em:
            # Amortized: gradient flows through mu_p → embeddings learn good E-step init.
            # mu_p gradient is well-conditioned: d(grad)/d(mu_p) = -α/σ_p, no feedback loop.
            #
            # When implicit_em=True, the IFT scale factor is the sole gradient path
            # to embeddings (via ImplicitEMGradient.apply after the E-step). Keeping
            # mu_p live here would double-count: embeddings receive BOTH the IFT-scaled
            # gradient AND the straight-through gradient through self-coupling.
            mu_p_current = mu_prior.clone()
        else:
            # Non-amortized (or implicit_em): detach priors (fixed reference)
            mu_p_current = mu_prior.detach().clone()

        # sigma_p is ALWAYS detached in the E-step: it is an M-step parameter (Level 2).
        # The E-step treats (μ_p, σ_p) as fixed while inferring q. Letting CE gradient
        # flow through the E-step's 1/σ_p terms creates positive feedback: smaller σ_p
        # → larger gradient → even smaller σ_p. The M-step loss (lambda_hyper · KL(s||h))
        # provides the correct, bounded gradient for sigma learning.
        if sigma_prior is not None:
            sigma_p = sigma_prior.detach().clone()
        else:
            sigma_p = sigma.detach().clone()

        # E-step sigma_p floor: prevent 1/σ_p blowup in self-coupling gradient.
        # PriorBank allows σ_p down to 0.01 (for sharp decode logits), but
        # the E-step gradient ∂KL(q||p)/∂σ = 0.5·(1/σ_p - 1/σ_q) needs a higher
        # floor to prevent nat_grad_sigma explosion (at σ_p=0.01, 1/σ_p=100).
        # Configurable via e_step_sigma_floor (default 0.1 caps 1/σ_p at 10.0).
        _floor = self.e_step_sigma_floor
        if sigma_p.dim() == 3:
            sigma_p = sigma_p.clamp(min=_floor)
        else:
            # Full covariance: clamp diagonal elements
            diag_vals = torch.diagonal(sigma_p, dim1=-2, dim2=-1)
            diag_clamped = diag_vals.clamp(min=_floor)
            sigma_p = sigma_p + torch.diag_embed(diag_clamped - diag_vals)

        # Convert diagonal sigma_p to full covariance if needed (PriorBank returns diagonal)
        if not is_diagonal and sigma_p.dim() == 3:
            sigma_p = torch.diag_embed(sigma_p)


        # Current state (will evolve)
        # Implicit EM: detach beliefs at E-step start (proper EM boundary).
        # The implicit gradient scale factor compensates for detachment with
        # the info-geometrically correct CE→embedding gradient.
        if self.implicit_em:
            mu_current = mu.detach().clone()
            sigma_current = sigma.detach().clone()
        else:
            mu_current = mu.clone()
            sigma_current = sigma.clone()

        # Clamp initial sigma to [eps, sigma_max] before E-step.
        # Without this, embedding/prior sigma can far exceed sigma_max (e.g., σ=16
        # vs sigma_max=5), causing nat_grad_sigma = 2σ²·∇σ to amplify by 2×16²=512
        # on the first iteration. The retraction clamps AFTER the update, but the
        # gradient was already computed on the un-clamped value.
        if self.update_sigma:
            _eps = 1e-6
            if sigma_current.dim() == 3:
                # Diagonal: element-wise clamp
                sigma_current = sigma_current.clamp(min=_eps, max=self.sigma_max)
            else:
                # Full covariance: spectral clamp on eigenvalues
                # Use _safe_eigh for robust decomposition — sigma from attention
                # aggregate_messages can be ill-conditioned or near-indefinite
                # early in training (wild transport operators, uncalibrated β).
                eigvals, eigvecs = _safe_eigh(sigma_current, jitter=_eps)
                eigvals = eigvals.clamp(min=_eps, max=self.sigma_max * self.sigma_max)
                sigma_current = eigvecs * eigvals.unsqueeze(-2) @ eigvecs.transpose(-1, -2)

        # Detach phi when detach_phi=True and non-amortized: enables fully backprop-free
        # training where phi_embed learns via phi P-flow instead of backprop.
        if not self.amortized_inference and self.detach_phi:
            phi_current = phi.detach().clone()
        else:
            phi_current = phi.clone()
        omega_current = omega.clone() if omega is not None else None  # Track omega for direct GL(K)

        # Track β evolution if requested
        beta_history = [] if return_beta_history else None

        # Store observation info for fresh gradient computation
        has_observations = targets is not None and W_out is not None
        _detach_e_step = True  # Standard path detaches; DEQ step_fn sets False
        beta_current = None  # Sentinel; set inside VFE loop for implicit EM scale computation
        beta_heads = []      # Per-head betas (multihead); populated inside VFE loop

        # =====================================================================
        # Determine alpha: Bayesian precision or fixed scalar
        # (needed by both closed-form and gradient descent paths)
        # =====================================================================
        if self.learnable_alpha:
            alpha_effective = self.get_bayesian_alpha(
                mu_current, mu_p_current, sigma_p, sigma_current, eps=eps
            )  # (B, N, K) - per-dim gauge-invariant, state-dependent
            _alpha_c0 = F.softplus(self.raw_c0)
        else:
            alpha_effective = self.alpha  # scalar (backward compatible)
            _alpha_c0 = None

        # =====================================================================
        # CLOSED-FORM E-STEP: Precision-weighted fixed point (optional)
        # =====================================================================
        # When closed_form_e_step=True, compute the exact fixed point of the
        # VFE objective (dropping the softmax coupling term KL·∂β/∂μ):
        #
        #   μ_i* = [α·μ_p/σ_p + λ·Σ_j β_ij·(Ω_ij μ_j)/σ_j] / [α/σ_p + λ·Σ_j β_ij/σ_j]
        #   σ_i* = 1 / [α/σ_p + λ·Σ_j β_ij/σ_j]
        #
        # This naturally includes aggregation and replaces the gradient descent loop.
        if self.closed_form_e_step:
            # 1. Compute block exp pairs for transport
            if self.irrep_dims is not None:
                if omega_current is not None and self.gauge_param == 'omega':
                    _cf_bep = []
                    block_start = 0
                    for d_h in self.irrep_dims:
                        omega_h = omega_current[:, :, block_start:block_start+d_h, block_start:block_start+d_h]
                        omega_h_inv = torch.linalg.inv(omega_h)
                        _cf_bep.append((omega_h, omega_h_inv))
                        block_start += d_h
                elif self.gauge_mode == 'trivial':
                    _cf_bep = []
                    for d_h in self.irrep_dims:
                        eye_h = torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()
                        _cf_bep.append((eye_h, eye_h))
                elif self.gauge_mode == 'constant' and self.constant_omega is not None:
                    _cf_bep = []
                    for h, d_h in enumerate(self.irrep_dims):
                        omega_h = self.constant_omega[h].to(device=device, dtype=dtype)
                        if getattr(self, 'enforce_orthogonal', False) and d_h >= 2:
                            omega_h = newton_schulz_orthogonalize(omega_h.unsqueeze(0)).squeeze(0)
                        eye_h = torch.eye(d_h, device=device, dtype=dtype)
                        _cf_bep.append((
                            omega_h.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous(),
                            eye_h.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous(),
                        ))
                else:
                    _cf_bep = fused_block_matrix_exp_pairs(
                        phi_current, self.generators, self.irrep_dims,
                        enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                    )
            else:
                _cf_bep = None

            # 2. Per-head: compute β_h, then closed-form fixed point
            beta_heads = []

            if is_diagonal:
                # =============================================================
                # DIAGONAL CLOSED-FORM: element-wise precision-weighted average
                # =============================================================
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
                    alpha_h = alpha_effective[:, :, block_start:block_end] if isinstance(alpha_effective, torch.Tensor) and alpha_effective.dim() == 3 else alpha_effective
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
                    exp_phi_h, exp_neg_phi_h = _cf_bep[h] if _cf_bep is not None else (
                        torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                        torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                    )

                    # Prior precision and information vector
                    inv_sigma_p_h = 1.0 / sigma_p_h.clamp(min=eps)       # (B, N, d_h)
                    prior_prec_h = alpha_h * inv_sigma_p_h                # (B, N, d_h)
                    prior_info_h = alpha_h * mu_p_h * inv_sigma_p_h       # (B, N, d_h)

                    # Alignment: precision-weighted transported aggregation
                    # Transported diagonal covariance: sigma_j_t[k] = sum_l Omega[k,l]^2 * sigma_j[l]
                    # This is diag(Omega @ diag(sigma_j) @ Omega^T), the exact diagonal transport.
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

                    # =============================================================
                    # ENHANCED CLOSED FORM: Include softmax coupling in fixed point
                    # =============================================================
                    # The softmax gradient ∂β/∂μ is LINEAR in μ_i (SymPy verified).
                    # The σ_i terms CANCEL in ∂β/∂σ (SymPy verified).
                    # This allows the full VFE fixed point (linear + softmax) to be
                    # solved in one division per dimension. See derivation:
                    # derivations/enhanced_closed_form_vfe.md
                    #
                    # mu*[k] = (b[k] - c[k]) / (A[k] + S[k])
                    # sigma*[k] = (alpha + 1) / (lam_total[k] + 2*S_sigma[k])

                    kappa_h_val = kappa_h.item() if isinstance(kappa_h, torch.Tensor) else kappa_h
                    kappa_h_scaled = kappa_h_val * math.sqrt(max(d_h, 1))
                    kappa_h_scaled = max(kappa_h_scaled, eps)

                    if self.lambda_softmax > 0 and kappa_h_scaled > 0:
                        # Per-pair softmax weights: w_j = KL_ij * beta_ij
                        w_j = kl_h.unsqueeze(-1) * beta_h.unsqueeze(-1)  # (B, N, N, 1) for broadcasting
                        # But we need (B, N, N) for scalar operations and (B, N, N, d_h) for per-dim
                        w_j_scalar = kl_h * beta_h  # (B, N, N)
                        w_bar = w_j_scalar.sum(dim=-1)  # (B, N) — expected KL

                        # S[k]: softmax coupling's contribution to precision
                        # S = -(lam_s/kappa)(sum_j w_j/sigma_jt - w_bar * avg_prec)
                        kl_weighted_prec = torch.einsum('bij,bijk->bik', w_j_scalar, inv_sigma_j_t)  # (B, N, d_h)
                        avg_prec_h = align_prec_h / max(self.lambda_belief, eps)  # = sum_m beta_im / sigma_mt
                        S_mu_h = -(self.lambda_softmax / kappa_h_scaled) * (
                            kl_weighted_prec - w_bar.unsqueeze(-1) * avg_prec_h
                        )  # (B, N, d_h)

                        # c[k]: softmax coupling's contribution to information
                        # c = (lam_s/kappa)(sum_j w_j * nu_j - w_bar * avg_info)
                        kl_weighted_info = torch.einsum('bij,bijk->bik', w_j_scalar, info_per_pair)  # (B, N, d_h)
                        avg_info_h = align_info_h / max(self.lambda_belief, eps)  # = sum_m beta_im * nu_m
                        c_mu_h = (self.lambda_softmax / kappa_h_scaled) * (
                            kl_weighted_info - w_bar.unsqueeze(-1) * avg_info_h
                        )  # (B, N, d_h)

                        # S_sigma[k]: sigma softmax coupling (σ_i terms cancel — exact!)
                        # S_sigma = -(lam_s/(2*kappa)) * sum_j w_j * (1/sigma_jt - p_bar)
                        p_bar_h = avg_prec_h  # attention-weighted transported precision
                        S_sigma_h = -(self.lambda_softmax / (2.0 * kappa_h_scaled)) * torch.einsum(
                            'bij,bijk->bik', w_j_scalar, inv_sigma_j_t - p_bar_h[:, :, None, :]
                        )  # (B, N, d_h)
                    else:
                        S_mu_h = 0.0
                        c_mu_h = 0.0
                        S_sigma_h = 0.0

                    del Omega_h_cf, sigma_j_t_diag, inv_sigma_j_t, mu_j_t_cf, info_per_pair

                    # Enhanced fixed point: (b - c) / (A + S) for mu, (alpha+1) / (lam_total + 2*S_sigma) for sigma
                    total_prec_h = prior_prec_h + align_prec_h + S_mu_h    # A + S  (B, N, d_h)
                    total_info_h = prior_info_h + align_info_h - c_mu_h    # b - c  (B, N, d_h)
                    mu_star[:, :, block_start:block_end] = total_info_h / total_prec_h.clamp(min=eps)

                    entropy_scale = alpha_h + self.lambda_belief  # alpha + 1 (when lambda_belief=1)
                    sigma_prec_h = prior_prec_h + align_prec_h + 2.0 * S_sigma_h  # lam_total + 2*S_sigma
                    sigma_star[:, :, block_start:block_end] = (entropy_scale / sigma_prec_h.clamp(min=eps)).clamp(max=self.sigma_max)

                    block_start = block_end

                mu_current = mu_star
                if self.update_sigma:
                    sigma_current = sigma_star
                    # Isotropic enforcement: closed-form produces per-dimension sigma,
                    # must project back to σ²I to maintain Limit 1 constraint.
                    if self.isotropic_covariance:
                        scalar_var = sigma_current.mean(dim=-1, keepdim=True)
                        sigma_current = scalar_var.expand_as(sigma_current)

            else:
                # =============================================================
                # FULL-COVARIANCE CLOSED-FORM: Q_j factorization + Cholesky solve
                # =============================================================
                # Uses the transported precision factorization:
                #   Sigma_{j,t}^{-1} = E_i^T @ Q_j @ E_i
                # where Q_j = exp_phi_j^T @ Sigma_j^{-1} @ exp_phi_j (i-independent)
                # See docs/closed_form_e_step_derivation.md for full derivation.
                K_full = self.embed_dim
                mu_star = torch.zeros_like(mu_current)
                # Full covariance output: (B, N, K, K) block-diagonal
                sigma_star_full = torch.zeros(B, N, K_full, K_full, device=device, dtype=dtype)
                block_start = 0

                for h, d_h in enumerate(self.irrep_dims or [self.embed_dim]):
                    block_end = block_start + d_h

                    mu_h = mu_current[:, :, block_start:block_end].detach().contiguous()
                    mu_p_h = mu_p_current[:, :, block_start:block_end].contiguous()
                    # Full covariance: (B, N, d_h, d_h)
                    sigma_h = sigma_current[:, :, block_start:block_end, block_start:block_end].detach().contiguous()
                    sigma_p_h = sigma_p[:, :, block_start:block_end, block_start:block_end].contiguous()
                    gen_h = self.generators[:, block_start:block_end, block_start:block_end]

                    kappa_h = self._get_kappa_h(h, d_h) if self.irrep_dims else self.kappa
                    alpha_h = alpha_effective[:, :, block_start:block_end] if isinstance(alpha_effective, torch.Tensor) and alpha_effective.dim() == 3 else alpha_effective
                    _head_bep = [_cf_bep[h]] if _cf_bep is not None else None

                    # Compute β_h (attention weights for this head)
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

                    # Transport operators for this head
                    exp_phi_h, exp_neg_phi_h = _cf_bep[h] if _cf_bep is not None else (
                        torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                        torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                    )
                    E_i = exp_neg_phi_h  # (B, N, d_h, d_h)

                    # --- Prior precision (matrix) ---
                    # Sigma_p_h: (B, N, d_h, d_h) → Sigma_p_inv via Cholesky
                    sigma_p_h_safe = sigma_p_h + eps * torch.eye(d_h, device=device, dtype=dtype)
                    L_p = torch.linalg.cholesky(sigma_p_h_safe)           # (B, N, d_h, d_h)
                    Sigma_p_inv_h = torch.cholesky_inverse(L_p)           # (B, N, d_h, d_h)

                    # alpha_h may be scalar or (B, N, d_h) — handle both
                    if isinstance(alpha_h, torch.Tensor) and alpha_h.dim() == 3:
                        # Per-dimension alpha: expand to diagonal matrix for multiplication
                        A_prior_h = alpha_h.unsqueeze(-1) * Sigma_p_inv_h  # broadcast (B,N,d,1)*(B,N,d,d)
                        b_prior_h = torch.einsum('bijk,bik->bij', A_prior_h, mu_p_h)
                    else:
                        A_prior_h = alpha_h * Sigma_p_inv_h                # (B, N, d_h, d_h)
                        b_prior_h = torch.einsum('bijk,bik->bij', A_prior_h, mu_p_h)  # (B, N, d_h)

                    # --- Q_j factorization ---
                    # Sigma_j^{-1} via Cholesky
                    sigma_h_safe = sigma_h + eps * torch.eye(d_h, device=device, dtype=dtype)
                    L_j = torch.linalg.cholesky(sigma_h_safe)             # (B, N, d_h, d_h)
                    Sigma_j_inv_h = torch.cholesky_inverse(L_j)           # (B, N, d_h, d_h)

                    # Q_j = exp_phi_j^T @ Sigma_j^{-1} @ exp_phi_j  (i-independent)
                    Q_j = torch.einsum(
                        'bjlk,bjlm,bjmn->bjkn', exp_phi_h, Sigma_j_inv_h, exp_phi_h
                    )  # (B, N, d_h, d_h)

                    # r_j = exp_neg_phi_j @ mu_j  (mean in flat frame)
                    r_j = torch.einsum('bjkl,bjl->bjk', exp_neg_phi_h, mu_h)  # (B, N, d_h)

                    # --- Beta-weighted aggregation ---
                    # Q_agg_i = sum_j beta_ij * Q_j
                    Q_agg = torch.einsum('bij,bjkl->bikl', beta_h, Q_j)  # (B, N, d_h, d_h)

                    # Qr_j = Q_j @ r_j, then aggregate
                    Qr_j = torch.einsum('bjkl,bjl->bjk', Q_j, r_j)      # (B, N, d_h)
                    Qr_agg = torch.einsum('bij,bjk->bik', beta_h, Qr_j)  # (B, N, d_h)

                    # --- Transform to position i's frame ---
                    # A_align = lambda * E_i^T @ Q_agg @ E_i
                    # CRITICAL: 'bikl' for E_i^T (transpose via swapped last indices)
                    A_align_h = self.lambda_belief * torch.einsum(
                        'bikl,bikm,bimn->biln', E_i, Q_agg, E_i
                    )  # (B, N, d_h, d_h)

                    # b_align = lambda * E_i^T @ Qr_agg
                    b_align_h = self.lambda_belief * torch.einsum(
                        'bikl,bik->bil', E_i, Qr_agg
                    )  # (B, N, d_h)

                    # --- Total precision and solve ---
                    A_h = A_prior_h + A_align_h  # (B, N, d_h, d_h)
                    b_h = b_prior_h + b_align_h  # (B, N, d_h)

                    # Regularize A_h for numerical stability
                    A_h = A_h + eps * torch.eye(d_h, device=device, dtype=dtype)

                    # Cholesky solve: A_h @ mu* = b_h, Sigma* = c * A^{-1}
                    # Entropy scaling: c = alpha + lambda (from ∂F/∂Σ = 0)
                    L_A = torch.linalg.cholesky(A_h)                     # (B, N, d_h, d_h)
                    mu_star_h = torch.cholesky_solve(
                        b_h.unsqueeze(-1), L_A
                    ).squeeze(-1)                                         # (B, N, d_h)
                    A_inv_h = torch.cholesky_inverse(L_A)                # (B, N, d_h, d_h)

                    # Exact posterior covariance: Sigma* = (alpha + lambda) * A^{-1}
                    if isinstance(alpha_h, torch.Tensor) and alpha_h.dim() == 3:
                        # Per-dimension alpha: c_k = alpha_k + lambda, scale each row/col
                        entropy_scale = (alpha_h + self.lambda_belief).unsqueeze(-1)  # (B, N, d_h, 1)
                        Sigma_star_h = entropy_scale * A_inv_h  # broadcast (B,N,d,1)*(B,N,d,d)
                    else:
                        entropy_scale = alpha_h + self.lambda_belief  # scalar
                        Sigma_star_h = entropy_scale * A_inv_h       # (B, N, d_h, d_h)

                    # Clamp Sigma eigenvalues to [eps, sigma_max]
                    Sigma_star_h = Sigma_star_h.clamp(max=self.sigma_max)

                    mu_star[:, :, block_start:block_end] = mu_star_h
                    sigma_star_full[:, :, block_start:block_end, block_start:block_end] = Sigma_star_h

                    block_start = block_end

                mu_current = mu_star
                if self.update_sigma:
                    sigma_current = sigma_star_full

            beta_current = beta_heads[-1] if beta_heads else None

            # =================================================================
            # Iterative re-solve: converge the beta <-> (mu, sigma) loop
            # =================================================================
            # The enhanced closed form (diagonal path) absorbs softmax coupling
            # into the precision-weighted solve: mu* = (b-c)/(A+S). But S, c,
            # S_sigma depend on beta and KL computed from INITIAL beliefs.
            # Re-solving with updated beta/KL converges the self-consistency.
            #
            # For full-covariance (which uses linear-only CF), we keep the
            # original Picard correction: mu^(n+1) = mu_0 - Sigma_0 @ grad.
            if self.n_picard_steps > 0 and self.lambda_softmax > 0 and _cf_bep is not None:
                if is_diagonal:
                    # ---------------------------------------------------------
                    # DIAGONAL: Iterative re-solve with updated beta/KL
                    # ---------------------------------------------------------
                    # Each step recomputes beta, KL from current beliefs, then
                    # re-solves the full enhanced CF. This correctly converges
                    # the beta <-> (mu, sigma) loop without double-counting.
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
                            # Recompute alpha from current beliefs if learnable
                            if self.learnable_alpha and hasattr(self, 'get_bayesian_alpha'):
                                alpha_n = self.get_bayesian_alpha(
                                    mu_current, mu_p_current, sigma_p, sigma_current, eps=eps
                                )
                                if isinstance(alpha_n, torch.Tensor) and alpha_n.dim() == 3:
                                    alpha_h_iter = alpha_n[:, :, block_start:block_end]
                                else:
                                    alpha_h_iter = alpha_n

                            _head_bep = [_cf_bep[h]] if _cf_bep is not None else None

                            # Recompute beta and KL from current beliefs
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

                            # Transport operators
                            exp_phi_h, exp_neg_phi_h = _cf_bep[h] if _cf_bep is not None else (
                                torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                                torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                            )

                            # Prior precision and information
                            inv_sigma_p_h = 1.0 / sigma_p_h.clamp(min=eps)
                            prior_prec_h = alpha_h_iter * inv_sigma_p_h
                            prior_info_h = alpha_h_iter * mu_p_h * inv_sigma_p_h

                            # Transport and alignment (same as initial CF)
                            Omega_h_cf = torch.einsum('bikl,bjlm->bijkm', exp_phi_h, exp_neg_phi_h)
                            sigma_j_t_diag = torch.einsum(
                                'bijkl,bijkl,bjl->bijk', Omega_h_cf, Omega_h_cf, sigma_h
                            ).clamp(min=eps)
                            inv_sigma_j_t = 1.0 / sigma_j_t_diag
                            mu_j_t_cf = torch.einsum('bijkl,bjl->bijk', Omega_h_cf, mu_h)
                            info_per_pair = mu_j_t_cf * inv_sigma_j_t

                            align_info_h = self.lambda_belief * torch.einsum('bij,bijk->bik', beta_h, info_per_pair)
                            align_prec_h = self.lambda_belief * torch.einsum('bij,bijk->bik', beta_h, inv_sigma_j_t)

                            # Enhanced softmax coupling with FRESH beta/KL
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

                            # Re-solve enhanced fixed point
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

                        # Update beta_heads for downstream use
                        beta_heads = beta_heads_new

                        # Convergence check
                        rel_change = (mu_current - mu_prev).norm() / mu_prev.norm().clamp(min=eps)
                        if rel_change < 1e-4:
                            break

                else:
                    # ---------------------------------------------------------
                    # FULL COVARIANCE: Original Picard (linear-only CF + grad)
                    # ---------------------------------------------------------
                    # The full-cov path uses linear-only CF, so the softmax
                    # gradient IS the full residual. Picard is correct here.
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

                            Omega_h = torch.einsum(
                                'bikl,bjlm->bijkm', exp_phi_h, exp_neg_phi_h
                            )
                            mu_j_t = torch.einsum(
                                'bijkl,bjl->bijk', Omega_h, mu_h
                            )
                            delta = mu_h[:, :, None, :] - mu_j_t

                            sigma_h_blk = sigma_0[:, :, block_start:block_end, block_start:block_end]

                            Sigma_j_t = torch.einsum(
                                'bijkl,bjlm,bijnm->bijkn', Omega_h, sigma_h_blk, Omega_h
                            )
                            Sigma_j_t = Sigma_j_t + eps * torch.eye(d_h, device=device, dtype=dtype)
                            Sigma_j_t_inv = torch.linalg.inv(Sigma_j_t)

                            grad_kl_pair = torch.einsum(
                                'bijkl,bijl->bijk', Sigma_j_t_inv, delta
                            )

                            sigma_i_h = sigma_h_blk
                            trace_h = torch.einsum(
                                'bijkl,bilk->bij', Sigma_j_t_inv, sigma_i_h
                            )
                            mahal_h = torch.einsum(
                                'bijk,bijk->bij', delta, grad_kl_pair
                            )
                            logdet_jt = torch.linalg.slogdet(Sigma_j_t)[1]
                            logdet_i = torch.linalg.slogdet(sigma_i_h + eps * torch.eye(d_h, device=device, dtype=dtype))[1]
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

                        # Alpha correction (full-cov path)
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

                        # Picard update: mu^(n+1) = mu_0 - Sigma_0 @ ∇F_softmax(mu^(n))
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

            # Store beta for implicit EM (if needed)
            if return_beta_history:
                beta_stacked = torch.stack(beta_heads, dim=1) if len(beta_heads) > 1 else beta_heads[0].unsqueeze(1)
                beta_history = [beta_stacked.detach().clone()]

            # Phi evolution via gradient (phi enters nonlinearly, no closed form)
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
                    _phi_bep_cf = fused_block_matrix_exp_pairs(
                        phi_current, self.generators, self.irrep_dims,
                        enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                    ) if self.irrep_dims is not None and self.gauge_mode == 'learned' else None
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

        # =====================================================================
        # VFE Descent Loop with Dynamic β (runs outside AMP autocast)
        # =====================================================================
        # Skip when closed_form_e_step handled the E-step above.
        _n_iters = 0 if self.closed_form_e_step else self.n_iterations
        for iteration in range(_n_iters):
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
            if self.irrep_dims is None and not self.multihead_vfe:
                if omega_current is not None and self.gauge_param == 'omega':
                    # Direct omega: build full-K transport from omega blocks
                    from transformer.core.attention import compute_transport_operators_direct
                    cached_transport = compute_transport_operators_direct(
                        omega=omega_current,
                        irrep_dims=self.irrep_dims if self.irrep_dims is not None else [self.embed_dim],
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
                    )
            else:
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
            global _VFE_GRAD_DEBUG
            if self._debug_vfe_gradients or self._collect_vfe_metrics:
                _VFE_GRAD_DEBUG = {}
            else:
                _VFE_GRAD_DEBUG = None

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
                # Without this, each head builds full Omega (B,N,N,d,d) twice
                # (once in compute_attention_weights, once in compute_vfe_gradients_gpu),
                # causing 2×n_heads redundant matrix_exp calls per VFE iteration.
                _mh_cached_bep = None
                if self.irrep_dims is not None:
                    if omega_current is not None and self.gauge_param == 'omega':
                        # Direct omega path: build (Omega_h, Omega_h_inv) per head
                        # from the block-diagonal omega matrix. No matrix_exp needed.
                        _mh_cached_bep = []
                        block_start = 0
                        for d_h in self.irrep_dims:
                            omega_h = omega_current[:, :, block_start:block_start+d_h, block_start:block_start+d_h]
                            omega_h_inv = torch.linalg.inv(omega_h)
                            _mh_cached_bep.append((omega_h, omega_h_inv))
                            block_start += d_h
                    elif self.gauge_mode == 'trivial':
                        # No matrix exponentials needed: Ω = I for all blocks.
                        # 'trivial': global frame (no transport).
                        _mh_cached_bep = []
                        for d_h in self.irrep_dims:
                            eye_h = torch.eye(d_h, device=mu_current.device,
                                              dtype=mu_current.dtype)
                            eye_h = eye_h.unsqueeze(0).unsqueeze(0).expand(
                                B, N, -1, -1).contiguous()
                            _mh_cached_bep.append((eye_h, eye_h))
                    elif self.gauge_mode == 'constant' and self.constant_omega is not None:
                        # Constant gauge: use per-head Ω from the attention module.
                        # exp_phi = Ω (broadcast to all positions), exp_neg_phi = I.
                        # This produces Ω_ij = Ω @ I = Ω for all pairs, consistent
                        # with the attention module's transport.
                        _mh_cached_bep = []
                        for h, d_h in enumerate(self.irrep_dims):
                            omega_h = self.constant_omega[h].to(
                                device=mu_current.device, dtype=mu_current.dtype)
                            if getattr(self, 'enforce_orthogonal', False) and d_h >= 2:
                                omega_h = newton_schulz_orthogonalize(
                                    omega_h.unsqueeze(0)).squeeze(0)
                            eye_h = torch.eye(d_h, device=mu_current.device,
                                              dtype=mu_current.dtype)
                            exp_phi_h = omega_h.unsqueeze(0).unsqueeze(0).expand(
                                B, N, -1, -1).contiguous()
                            exp_neg_phi_h = eye_h.unsqueeze(0).unsqueeze(0).expand(
                                B, N, -1, -1).contiguous()
                            _mh_cached_bep.append((exp_phi_h, exp_neg_phi_h))
                    elif self.gauge_mode == 'constant':
                        # Constant gauge without constant_omega: fall back to identity
                        # (legacy behavior for backward compatibility)
                        _mh_cached_bep = []
                        for d_h in self.irrep_dims:
                            eye_h = torch.eye(d_h, device=mu_current.device,
                                              dtype=mu_current.dtype)
                            eye_h = eye_h.unsqueeze(0).unsqueeze(0).expand(
                                B, N, -1, -1).contiguous()
                            _mh_cached_bep.append((eye_h, eye_h))
                    else:
                        _mh_cached_bep = fused_block_matrix_exp_pairs(
                            phi_current, self.generators, self.irrep_dims,
                            enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
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

                    if _use_fused_mh:
                        # FUSED: single pass computes β_h AND gradients (1× Omega)
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
                            exact_diagonal_transport=self.exact_diagonal_transport,
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
                    if _VFE_GRAD_DEBUG is not None and _VFE_GRAD_DEBUG:
                        _pfx = f'head{h}(d={d_h})'
                        for _k, _v in list(_VFE_GRAD_DEBUG.items()):
                            # Store per-head values with prefix, clear base keys
                            _VFE_GRAD_DEBUG[f'{_pfx}/{_k}'] = _v
                        # Reset base keys for next head
                        for _k in [k for k in _VFE_GRAD_DEBUG if '/' not in k]:
                            del _VFE_GRAD_DEBUG[_k]

                    block_start = block_end

                if return_beta_history:
                    beta_stacked = torch.stack(beta_heads, dim=1)
                    beta_history.append(beta_stacked.detach().clone())
                beta_current = beta_heads[-1]

            else:
                # =============================================================
                # SINGLE-β VFE: Original behavior (all blocks share one β)
                # =============================================================
                # SINGLE-β VFE: All blocks share one β
                # Use fused path when possible (diagonal + block-diagonal)
                # =============================================================
                _cached_bep = None
                if self.irrep_dims is not None:
                    if omega_current is not None and self.gauge_param == 'omega':
                        # Direct omega path: build (Omega_h, Omega_h_inv) per head
                        _cached_bep = []
                        block_start = 0
                        for d_h in self.irrep_dims:
                            omega_h = omega_current[:, :, block_start:block_start+d_h, block_start:block_start+d_h]
                            omega_h_inv = torch.linalg.inv(omega_h)
                            _cached_bep.append((omega_h, omega_h_inv))
                            block_start += d_h
                    elif self.gauge_mode == 'trivial':
                        _cached_bep = []
                        for d_h in self.irrep_dims:
                            eye_h = torch.eye(d_h, device=mu_current.device,
                                              dtype=mu_current.dtype)
                            eye_h = eye_h.unsqueeze(0).unsqueeze(0).expand(
                                B, N, -1, -1).contiguous()
                            _cached_bep.append((eye_h, eye_h))
                    elif self.gauge_mode == 'constant' and self.constant_omega is not None:
                        _cached_bep = []
                        for h, d_h in enumerate(self.irrep_dims):
                            omega_h = self.constant_omega[h].to(
                                device=mu_current.device, dtype=mu_current.dtype)
                            if getattr(self, 'enforce_orthogonal', False) and d_h >= 2:
                                omega_h = newton_schulz_orthogonalize(
                                    omega_h.unsqueeze(0)).squeeze(0)
                            eye_h = torch.eye(d_h, device=mu_current.device,
                                              dtype=mu_current.dtype)
                            exp_phi_h = omega_h.unsqueeze(0).unsqueeze(0).expand(
                                B, N, -1, -1).contiguous()
                            exp_neg_phi_h = eye_h.unsqueeze(0).unsqueeze(0).expand(
                                B, N, -1, -1).contiguous()
                            _cached_bep.append((exp_phi_h, exp_neg_phi_h))
                    elif self.gauge_mode == 'constant':
                        _cached_bep = []
                        for d_h in self.irrep_dims:
                            eye_h = torch.eye(d_h, device=mu_current.device,
                                              dtype=mu_current.dtype)
                            eye_h = eye_h.unsqueeze(0).unsqueeze(0).expand(
                                B, N, -1, -1).contiguous()
                            _cached_bep.append((eye_h, eye_h))
                    else:
                        _cached_bep = fused_block_matrix_exp_pairs(
                            phi_current, self.generators, self.irrep_dims,
                            enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                        )

                # Use fused path for diagonal + block-diagonal
                _use_fused_single = (is_diagonal and self.irrep_dims is not None
                                     and not self.exact_diagonal_transport)

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
                    )

                if return_beta_history:
                    beta_history.append(beta_current.detach().clone())

            # Add FRESH observation gradient (recomputed from current beliefs)
            # Use .detach() on mu_current to avoid second-order gradients through the
            # observation gradient computation. Gradients still flow through VFE dynamics
            # (the natural gradient update), just not through how the obs grad was computed.
            # This is more stable than full gradient flow while still allowing embeddings
            # to learn from VFE dynamics via the mu_current → mu_new update chain.
            if has_observations:
                logits = torch.matmul(mu_current.detach(), W_out.T)
                probs = F.softmax(logits, dim=-1)
                targets_valid = targets.clone()
                targets_valid[targets == -1] = 0
                one_hot = F.one_hot(targets_valid, num_classes=W_out.shape[0]).float()
                mask_obs = (targets != -1).unsqueeze(-1).float()
                one_hot = one_hot * mask_obs
                grad_error = (probs - one_hot) * mask_obs
                discrete_obs_grad = torch.matmul(grad_error, W_out)
                # Scale observation gradient by obs_weight (for warmup ramp)
                _obs_weight = getattr(self, '_obs_weight', 1.0)
                if _obs_weight < 1.0:
                    discrete_obs_grad = discrete_obs_grad * _obs_weight
                # Debug: observation mu gradient
                if _VFE_GRAD_DEBUG is not None:
                    _VFE_GRAD_DEBUG['obs_mu_grad'] = _grad_norm(discrete_obs_grad)

                grad_mu = grad_mu + discrete_obs_grad

                # Observation gradient for sigma (exact via Stein's lemma):
                #
                #   ∂/∂σ_k E_q[CE(z)] = (1/2) · E_q[∂²CE/∂z_k²]
                #
                # This is EXACT for any smooth loss, not a Taylor approximation.
                # For CE with softmax: ∂²CE/∂z_k² = Var_p[W[:,k]] ≥ 0.
                # We approximate E_q[H_kk(z)] ≈ H_kk(μ) (zeroth-order in σ).
                if self.obs_sigma_gradient:
                    W_out_sq = W_out ** 2                                # (V, K)
                    EW2 = torch.matmul(probs, W_out_sq)                  # (B, N, K)
                    EW  = torch.matmul(probs, W_out)                     # (B, N, K)
                    hessian_diag = EW2 - EW ** 2                         # (B, N, K)
                    # Clamp: FP rounding can violate Var ≥ 0
                    _neg_mask = hessian_diag < 0
                    if _neg_mask.any():
                        _nr("obs_sigma_hessian_neg_clamp", count=int(_neg_mask.sum().item()))
                        hessian_diag = hessian_diag.clamp(min=0.0)
                    _sigma_obs_scale = (0.5 * self.obs_sigma_weight) * _obs_weight
                    # Cap observation sigma gradient to prevent systematic upward bias
                    # from dominating. Var_p[W[:,k]] >= 0 always, so this term only
                    # pushes sigma upward; cap prevents runaway growth.
                    obs_sigma_grad = (_sigma_obs_scale * hessian_diag * mask_obs).clamp(max=10.0)
                    # Debug: observation sigma gradient (before diag_embed)
                    if _VFE_GRAD_DEBUG is not None:
                        _VFE_GRAD_DEBUG['obs_sigma_grad'] = _grad_norm(obs_sigma_grad)
                    # For full covariance, obs gradient is diagonal-only: embed on diagonal
                    # to avoid broadcasting (B, N, K) into every row of (B, N, K, K).
                    if not is_diagonal:
                        obs_sigma_grad = torch.diag_embed(obs_sigma_grad)
                    grad_sigma = grad_sigma + obs_sigma_grad

            # Debug: Euclidean totals (after obs, before clip)
            if _VFE_GRAD_DEBUG is not None:
                _VFE_GRAD_DEBUG['euclidean_mu_total'] = _grad_norm(grad_mu)
                _VFE_GRAD_DEBUG['euclidean_sigma_total'] = _grad_norm(grad_sigma)
                _ps = _per_pos_stats(grad_mu)
                _VFE_GRAD_DEBUG['euclidean_mu_pos_mean'] = _ps[0]
                _VFE_GRAD_DEBUG['euclidean_mu_pos_max'] = _ps[1]
                _ps = _per_pos_stats(grad_sigma)
                _VFE_GRAD_DEBUG['euclidean_sigma_pos_mean'] = _ps[0]
                _VFE_GRAD_DEBUG['euclidean_sigma_pos_max'] = _ps[1]

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
            _raw_nat_grad_norm = nat_grad_mu.detach().norm().item()  # pre-clamp for diagnostics
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
            _raw_nat_grad_sigma_norm = nat_grad_sigma.detach().norm().item()
            max_nat_grad_sigma_norm = 500.0
            nat_grad_sigma_scale = torch.clamp(
                max_nat_grad_sigma_norm / (nat_grad_sigma_norm + eps), max=1.0
            )
            nat_grad_sigma = nat_grad_sigma * nat_grad_sigma_scale

            # Store E-step gradient norms (overwritten each iteration; final = last iter)
            self._e_step_grad_norms['nat_grad_mu'] = _raw_nat_grad_norm
            self._e_step_grad_norms['nat_grad_sigma'] = _raw_nat_grad_sigma_norm
            self._e_step_grad_norms['nat_grad_mu_clipped'] = nat_grad_mu.detach().norm().item()
            self._e_step_grad_norms['nat_grad_sigma_clipped'] = nat_grad_sigma.detach().norm().item()
            # Per-position clip fraction for the 500-norm cap
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
            if _VFE_GRAD_DEBUG is not None and self._debug_vfe_gradients:
                d = _VFE_GRAD_DEBUG

                # Detect multihead mode: keys have 'headN(d=M)/' prefix
                _is_multihead = any('/' in k for k in d)

                # Euclidean totals computed on the full (already assembled) grad tensors
                _eu_mu = _grad_norm(grad_mu)
                _eu_sig = _grad_norm(grad_sigma)
                _ps_mu = _per_pos_stats(grad_mu)
                _ps_sig = _per_pos_stats(grad_sigma)

                # Nat_grad amplification factors
                _amp_mu = _raw_nat_grad_norm / max(_eu_mu, 1e-12)
                _amp_sig = _raw_nat_grad_sigma_norm / max(_eu_sig, 1e-12)
                # Fraction of positions hitting the 500 clip
                _mu_clip_frac = (nat_grad_mu_norm.squeeze(-1) >= max_nat_grad_norm * 0.99).float().mean().item()
                if is_diagonal:
                    _sig_clip_frac = (nat_grad_sigma_norm.squeeze(-1) >= max_nat_grad_sigma_norm * 0.99).float().mean().item()
                else:
                    _sig_clip_frac = (nat_grad_sigma_norm.squeeze(-1).squeeze(-1) >= max_nat_grad_sigma_norm * 0.99).float().mean().item()

                print(f"\n{'='*80}")
                print(f"  [VFE GRAD DEBUG] iter {iteration}/{self.n_iterations}"
                      f"  diag={is_diagonal}  K={mu_current.shape[-1]}"
                      f"  B×N={mu_current.shape[0]}×{mu_current.shape[1]}"
                      f"  multihead={_is_multihead}")
                print(f"{'='*80}")

                if _is_multihead:
                    # Extract unique head prefixes
                    _heads = sorted(set(k.split('/')[0] for k in d if '/' in k),
                                    key=lambda x: int(x.split('head')[1].split('(')[0]))
                    print(f"  --- Per-head breakdown ({len(_heads)} heads) ---")
                    print(f"  {'Head':<16} {'σ_self':>8} {'σ_align':>8} {'σ_smx':>8}"
                          f" {'μ_self':>8} {'μ_dir':>8} {'μ_smx':>8}"
                          f" {'KL_avg':>8} {'KL_max':>8} {'σ_p_min':>8} {'σ_q_max':>8}")
                    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8}"
                          f" {'─'*8} {'─'*8} {'─'*8}"
                          f" {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
                    for hp in _heads:
                        def _hget(key, default=0):
                            return d.get(f'{hp}/{key}', default)
                        print(f"  {hp:<16}"
                              f" {_hget('grad_sigma_self'):>8.1f}"
                              f" {_hget('grad_sigma_align_direct'):>8.1f}"
                              f" {_hget('grad_sigma_softmax'):>8.1f}"
                              f" {_hget('grad_mu_self'):>8.1f}"
                              f" {_hget('grad_mu_direct'):>8.1f}"
                              f" {_hget('grad_mu_softmax'):>8.1f}"
                              f" {_hget('kl_pairwise_mean'):>8.2f}"
                              f" {_hget('kl_pairwise_max'):>8.1f}"
                              f" {_hget('sigma_p_min'):>8.4f}"
                              f" {_hget('sigma_q_eig_max'):>8.4f}")
                else:
                    # Single-β mode
                    print(f"  --- Covariance state ---")
                    print(f"  σ_p  range:  [{d.get('sigma_p_min', 0):.4f}, {d.get('sigma_p_max', 0):.4f}]"
                          f"  →  1/σ_p range: [{1/max(d.get('sigma_p_max', 1), 1e-12):.2f},"
                          f" {1/max(d.get('sigma_p_min', 1e-12), 1e-12):.2f}]")
                    print(f"  σ_q  eig range: [{d.get('sigma_q_eig_min', 0):.4f}, {d.get('sigma_q_eig_max', 0):.4f}]")
                    print()
                    print(f"  --- Euclidean gradient components (global norms) ---")
                    print(f"  {'Component':<30} {'μ':>12} {'σ':>12} {'σ pos_mean':>12} {'σ pos_max':>12}")
                    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
                    print(f"  {'self-coupling (α·∂KL/∂θ)':<30}"
                          f" {d.get('grad_mu_self', 0):>12.1f}"
                          f" {d.get('grad_sigma_self', 0):>12.1f}"
                          f" {d.get('grad_sigma_self_pos_mean', 0):>12.2f}"
                          f" {d.get('grad_sigma_self_pos_max', 0):>12.2f}")
                    print(f"  {'align direct (λ·β·∂KL/∂θ)':<30}"
                          f" {d.get('grad_mu_direct', 0):>12.1f}"
                          f" {d.get('grad_sigma_align_direct', 0):>12.1f}"
                          f" {d.get('grad_sigma_align_pos_mean', 0):>12.2f}"
                          f" {d.get('grad_sigma_align_pos_max', 0):>12.2f}")
                    print(f"  {'softmax (KL·∂β/∂θ)':<30}"
                          f" {d.get('grad_mu_softmax', 0):>12.1f}"
                          f" {d.get('grad_sigma_softmax', 0):>12.1f}"
                          f" {d.get('grad_sigma_softmax_pos_mean', 0):>12.2f}"
                          f" {d.get('grad_sigma_softmax_pos_max', 0):>12.2f}")

                # Observation (shared between multihead and single-β, computed on full tensor)
                if 'obs_mu_grad' in d:
                    print(f"  {'observation (CE)':<30}"
                          f" {d.get('obs_mu_grad', 0):>12.1f}"
                          f" {d.get('obs_sigma_grad', 0):>12.1f}")

                print()
                print(f"  --- Euclidean total (assembled, after obs) ---")
                print(f"  grad_mu:    {_eu_mu:>10.1f}  (pos mean: {_ps_mu[0]:.2f}, max: {_ps_mu[1]:.2f})")
                print(f"  grad_sigma: {_eu_sig:>10.1f}  (pos mean: {_ps_sig[0]:.2f}, max: {_ps_sig[1]:.2f})")
                print()
                print(f"  --- Natural gradient (Fisher projection) ---")
                print(f"  nat_grad_mu:    {_raw_nat_grad_norm:>10.1f}  (amplification: {_amp_mu:.2f}x)"
                      f"  clip: {self._e_step_grad_norms['nat_grad_mu_clipped']:.1f}"
                      f"  ({_mu_clip_frac*100:.0f}% positions at cap)")
                print(f"  nat_grad_sigma: {_raw_nat_grad_sigma_norm:>10.1f}  (amplification: {_amp_sig:.2f}x)"
                      f"  clip: {self._e_step_grad_norms['nat_grad_sigma_clipped']:.1f}"
                      f"  ({_sig_clip_frac*100:.0f}% positions at cap)")
                print(f"{'='*80}\n")
                # Store before resetting (debug mode may coexist with metrics collection)
                if self._collect_vfe_metrics:
                    _aggregate_multihead_vfe_debug(_VFE_GRAD_DEBUG, self.irrep_dims)
                    self.last_vfe_debug = dict(_VFE_GRAD_DEBUG)
                _VFE_GRAD_DEBUG = None  # Reset for next iteration

            # Store lightweight copy for external consumption (no printing overhead)
            elif self._collect_vfe_metrics and _VFE_GRAD_DEBUG is not None:
                _aggregate_multihead_vfe_debug(_VFE_GRAD_DEBUG, self.irrep_dims)
                self.last_vfe_debug = dict(_VFE_GRAD_DEBUG)
                _VFE_GRAD_DEBUG = None

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

            # Track trust region clip fraction
            self._e_step_grad_norms['mu_trust_frac'] = (
                scale.squeeze(-1) < 0.99
            ).float().mean().item()
            self._e_step_grad_norms['whitened_mu_mean'] = whitened_norm.mean().item()
            self._e_step_grad_norms['whitened_mu_max'] = whitened_norm.max().item()

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
                    s_min = sigma_current.min(dim=-1, keepdim=True).values.clamp(min=eps)
                    s_max = sigma_current.max(dim=-1, keepdim=True).values
                    condition = s_max / s_min  # (B, N, 1)
                    needs_clamp = condition > max_condition
                    if needs_clamp.any():
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
                        if (condition > max_condition).any():
                            geo_mean = eigvals.log().mean(dim=-1, keepdim=True).exp()
                            lower = geo_mean / (max_condition ** 0.5)
                            # Regularize toward isotropic: Sigma → Sigma + ridge * I
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
                    'nat_grad_mu_raw_norm': _raw_nat_grad_norm,
                    'nat_grad_sigma_norm': nat_grad_sigma.detach().norm().item(),
                    'nat_grad_sigma_raw_norm': _raw_nat_grad_sigma_norm,
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

        # Store final alpha_i for M-step loss (avoids changing return signatures)
        # alpha_effective is (B, N, K) if learnable_alpha, else scalar
        if self.learnable_alpha:
            self._last_alpha_i = alpha_effective.detach()  # (B, N, K)
        else:
            self._last_alpha_i = None

        # Store final beta for implicit EM scale computation
        # beta_current holds the last iteration's attention weights
        # (multihead: last head's beta; single-head: full beta)
        if self.implicit_em:
            if self.multihead_vfe and beta_heads:
                # Multihead: stack per-head betas into (B, H, N, N)
                self._last_beta_for_implicit = torch.stack(beta_heads, dim=1).detach()
            elif beta_current is not None:
                self._last_beta_for_implicit = beta_current.detach()
            else:
                self._last_beta_for_implicit = None

        # Compute and store implicit EM gradient scales (Phase 3/4)
        if self.implicit_em:
            # Get last beta — use _last_beta stored during the VFE loop
            _beta_for_scale = getattr(self, '_last_beta_for_implicit', None)

            if _beta_for_scale is not None:
                # Always use fixed ffn_alpha for IFT scale, NOT adaptive alpha_i.
                # Adaptive α_i gates E-step dynamics (shrinks as KL grows), but using
                # it for IFT creates a death spiral: α_i↓ → scale↓ → weak CE signal
                # → embeddings don't learn → more smoothing → KL↑ → α_i↓ further.
                # The IFT scale should be a stable structural property.
                _alpha_for_scale = self.alpha
                mu_scale, sigma_scale = compute_implicit_em_scales(
                    alpha_i=_alpha_for_scale,
                    sigma_p=sigma_p,
                    beta=_beta_for_scale,
                    sigma_q=sigma_current if sigma_current is not None else sigma_p,
                    eps=eps,
                )
                self._last_implicit_mu_scale = mu_scale      # (B, N, K)
                self._last_implicit_sigma_scale = sigma_scale  # (B, N, K)
            else:
                self._last_implicit_mu_scale = None
                self._last_implicit_sigma_scale = None
        else:
            self._last_implicit_mu_scale = None
            self._last_implicit_sigma_scale = None

        # Store evolved omega for multi-layer propagation (gauge_param='omega').
        # Without this, omega evolution from E-step iterations is lost between
        # layers — each layer would receive the original embedding omega.
        self._last_omega = omega_current

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
