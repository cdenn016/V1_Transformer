"""
VFE Utility Functions
=====================

Utility functions and constants extracted from variational_ffn.py.  These are
pure mathematical helpers -- no nn.Module, no learned parameters.

Provides:
- Module-level debug dict ``_VFE_GRAD_DEBUG``
- Numerical constants (SIGMA_EPS, TRANSPORT_JITTER, ...)
- squeeze_trailing_singletons -- collapse spurious trailing singleton dims
- _safe_spd_inv  -- robust SPD matrix inversion with adaptive regularization
- _safe_eigh     -- eigendecomposition with escalating jitter + SVD fallback
- retract_spd_torch          -- affine-invariant SPD exponential map (full matrix)
- retract_spd_diagonal_torch -- exponential retraction for diagonal covariances
- _retract_phi   -- phi retraction to Lie group (SO(N) or GL(K))
- _grad_norm      -- global Frobenius norm helper for debug logging
- _per_pos_stats  -- per-position norm statistics for debug logging
- _aggregate_multihead_vfe_debug -- aggregate per-head VFE debug metrics
"""

import math
import torch
from typing import Dict, Optional, Tuple

# =============================================================================
# Module-level constants
# =============================================================================
SIGMA_EPS: float = 1e-6
TRANSPORT_JITTER: float = 1e-4
KL_CEIL_BASE: float = 100.0
KL_CEIL_SCALE: float = 20.0
GRAD_CLIP_THRESHOLD: float = 10.0
KAPPA_CLAMP_RANGE: Tuple[float, float] = (0.5, 1.5)

# =============================================================================
# VFE Gradient Debug Infrastructure
# =============================================================================
# Module-level dict populated by gradient functions when _VFE_GRAD_DEBUG is not None.
# The E-step loop sets this before calling gradient functions, then reads it after.
_VFE_GRAD_DEBUG: Optional[Dict[str, float]] = None


# =============================================================================
# Debug helpers
# =============================================================================

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
# Squeeze utility
# =============================================================================

def squeeze_trailing_singletons(tensor: torch.Tensor, max_dim: int = 3) -> torch.Tensor:
    """Remove trailing singleton dimensions until ``tensor.dim() <= max_dim``.

    Replaces the recurring inline pattern::

        while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
            sigma_q = sigma_q.squeeze(-1)

    Args:
        tensor: Input tensor, any shape.
        max_dim: Target maximum dimensionality.  Only trailing dimensions of
            size 1 are removed; the loop stops as soon as the last dimension
            is not 1 or ``tensor.dim() <= max_dim``.

    Returns:
        Tensor with trailing singleton dimensions removed, shape unchanged if
        the last dimension is already > 1 or ``tensor.dim() <= max_dim``.
    """
    while tensor.dim() > max_dim and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    return tensor


# =============================================================================
# Robust SPD linear-algebra primitives
# =============================================================================

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
    (x10 per retry) until ``max_jitter`` is reached. If all retries fail,
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
        (eigenvalues, eigenvectors) -- same shapes as ``torch.linalg.eigh``
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


# =============================================================================
# SPD retraction (PyTorch GPU versions)
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
        sigma_max: Upper bound on eigenvalues (sigma_max^2 for covariance eigenvalues)

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

    # Force float32 for eigendecomposition, sqrt, exp -- all break in float16
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

        # Spectral floor + ceiling: eigenvalues in [eps, sigma_max^2]
        # sigma_max bounds the standard deviation; covariance eigenvalues are σ^2.
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
            Prevents nat_grad_sigma = 2σ^2·∇σ blowup (amplification grows as σ⁴).

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

    # Clamp to [eps, sigma_max] -- posterior σ should not blow up past the prior.
    # With sigma_max=5.0 and init_sigma_scale=1.0, allows 5x expansion before clamping.
    # This bounds the natural gradient amplification: 2σ^2 ≤ 2·sigma_max^2 = 50.
    return sigma_new.clamp(min=eps, max=sigma_max).to(orig_dtype)


# =============================================================================
# Phi retraction (Lie group / gauge frame update)
# =============================================================================

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
    - n_gen = N(N-1)/2 -> SO(N): compact, uses trust_region=0.3, max_norm=π, bch_order=1
    - n_gen = K^2       -> GL(K): non-compact, uses trust_region=0.1, max_norm=5.0, bch_order=2

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
        # GL(K) where n_gen = H*d^2 + n_cross*d^2 > K^2.
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

    # Check if this is GL(K) (n_gen = K^2) or SO(N) (n_gen = N(N-1)/2)
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
