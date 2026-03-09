# -*- coding: utf-8 -*-
"""
Shared Gauge-Geometry Utilities
================================

Consolidated geometry primitives used across attention, variational_ffn,
and embeddings. Extracted from the legacy codebase where these were
duplicated or scattered.

Contents:
    - stable_matrix_exp_pair: Numerically stable matrix exponential
    - safe_spd_inv: Robust SPD matrix inversion
    - retract_spd: Full-covariance SPD manifold retraction
    - retract_spd_diagonal: Diagonal-covariance retraction
    - retract_phi: Gauge frame retraction (SO(N) / GL(K))
    - so3_log: SO(3) → so(3) logarithm map
    - so3_compose_bch: Baker-Campbell-Hausdorff composition in so(3)
"""

import math
import torch
from typing import Tuple, Optional

# Import SO(N) and GL(K) retraction for proper phi updates
try:
    from .generators import (
        retract_soN_torch,
        retract_glK_torch,
        is_soN_generators,
        is_glK_generators,
    )
    RETRACTION_AVAILABLE = True
except ImportError:
    RETRACTION_AVAILABLE = False

# Import SO(N) BCH composition for proper Lie group operations
try:
    from .generators import soN_compose_bch_torch
    SON_BCH_AVAILABLE = True
except ImportError:
    SON_BCH_AVAILABLE = False


# =============================================================================
# Matrix Exponential
# =============================================================================

def stable_matrix_exp_pair(
    matrix: torch.Tensor,
    dim_threshold: int = 8,
    max_norm: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute exp(M) and exp(-M) with norm clamping and float64 upcasting.

    Two stability measures:
    1. Frobenius norm clamping: caps ‖M‖_F at max_norm to prevent the
       Padé scaling-squaring algorithm from overflowing.
    2. Float64 upcasting for K >= dim_threshold.

    Note on surjectivity:
        exp(M) always has det > 0 (since det(exp(M)) = exp(tr(M))), so the
        outputs live in GL⁺(K), the identity component.
        Ω_ij = exp(M_i)·exp(-M_j) covers all of GL⁺(K) (by polar decomposition).
        For SO(K), exp: so(K) → SO(K) is surjective.

    Args:
        matrix: (..., d, d) matrix to exponentiate.
        dim_threshold: Upcast to float64 when d >= this value.
        max_norm: Maximum Frobenius norm for input matrix.

    Returns:
        (exp_pos, exp_neg): Tuple of exp(M) and exp(-M).
    """
    # Clamp Frobenius norm — gradient flows through scaling factor
    mat_norm = matrix.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    scale = (max_norm / mat_norm).clamp(max=1.0)
    matrix = matrix * scale

    d = matrix.shape[-1]
    orig_dtype = matrix.dtype

    if d >= dim_threshold:
        matrix_up = matrix.double()
        exp_pos = torch.linalg.matrix_exp(matrix_up).to(orig_dtype)
        exp_neg = torch.linalg.matrix_exp(-matrix_up).to(orig_dtype)
    else:
        matrix_f32 = matrix.float()
        exp_pos = torch.linalg.matrix_exp(matrix_f32).to(orig_dtype)
        exp_neg = torch.linalg.matrix_exp(-matrix_f32).to(orig_dtype)

    return exp_pos, exp_neg


# =============================================================================
# SPD Inverse
# =============================================================================

def safe_spd_inv(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Robust inversion for SPD covariance matrices.

    Uses adaptive regularization: max(eps, eps * ||M||_diag_max).
    Falls back to pseudoinverse if standard inverse fails.

    Args:
        M: (..., K, K) covariance matrices
        eps: Base regularization floor

    Returns:
        M_inv: (..., K, K) inverse matrices
    """
    K = M.shape[-1]
    device = M.device
    orig_dtype = M.dtype
    M = M.float()

    diag_max = M.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1, keepdim=True).unsqueeze(-1)
    reg_scale = torch.clamp(diag_max * eps, min=eps)
    M_reg = M + reg_scale * torch.eye(K, device=device, dtype=torch.float32)

    try:
        result = torch.linalg.inv(M_reg)
    except torch.linalg.LinAlgError:
        result = torch.linalg.pinv(M_reg)

    return result.to(orig_dtype)


# =============================================================================
# SPD Retraction (Full Covariance)
# =============================================================================

def retract_spd(
    Sigma: torch.Tensor,
    delta_Sigma: torch.Tensor,
    step_size: float = 1.0,
    trust_region: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """SPD-preserving retraction for covariance matrices.

    Uses the affine-invariant exponential map on the SPD manifold:
        R = Σ^{-1/2} (η · ΔΣ) Σ^{-1/2}     (whitened tangent)
        Σ_{k+1} = Σ^{1/2} exp(R) Σ^{1/2}     (exponential map)

    This is gauge-invariant: for any invertible A, the retraction of
    A Σ A^T along A ΔΣ A^T equals A Σ_{new} A^T.

    Args:
        Sigma: SPD matrices, shape (B, N, K, K) or (B*N, K, K)
        delta_Sigma: Symmetric tangent vectors, same shape
        step_size: Learning rate τ
        trust_region: Max Frobenius norm of whitened tangent R
        eps: Regularization floor

    Returns:
        Sigma_new: SPD matrices, same shape as Sigma
    """
    original_shape = Sigma.shape
    orig_dtype = Sigma.dtype

    if Sigma.dim() == 4:
        B, N, K, _ = Sigma.shape
        Sigma = Sigma.reshape(B * N, K, K)
        delta_Sigma = delta_Sigma.reshape(B * N, K, K)

    Sigma = Sigma.float()
    delta_Sigma = delta_Sigma.float()
    K = Sigma.shape[-1]
    device = Sigma.device

    # Symmetrize inputs
    Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))
    delta_Sigma = 0.5 * (delta_Sigma + delta_Sigma.transpose(-1, -2))

    # Eigendecompose Σ → Σ^{1/2} and Σ^{-1/2}
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
    eigenvalues = eigenvalues.clamp(min=eps)

    sqrt_eig = torch.sqrt(eigenvalues)
    inv_sqrt_eig = 1.0 / sqrt_eig

    Sigma_sqrt = eigenvectors * sqrt_eig.unsqueeze(-2) @ eigenvectors.transpose(-1, -2)
    Sigma_inv_sqrt = eigenvectors * inv_sqrt_eig.unsqueeze(-2) @ eigenvectors.transpose(-1, -2)

    # Whitened tangent: R = Σ^{-1/2} (η · ΔΣ) Σ^{-1/2}
    R = Sigma_inv_sqrt @ (step_size * delta_Sigma) @ Sigma_inv_sqrt
    R = 0.5 * (R + R.transpose(-1, -2))

    # Trust region
    if trust_region is not None and trust_region > 0:
        R_norm = torch.linalg.norm(R, ord='fro', dim=(-2, -1), keepdim=True)
        scale = torch.clamp(trust_region / (R_norm + eps), max=1.0)
        R = R * scale

    # Exponential map via eigendecomposition
    R_eigenvalues, R_eigenvectors = torch.linalg.eigh(R)
    R_eigenvalues = R_eigenvalues.clamp(-50.0, 50.0)
    exp_R = R_eigenvectors * torch.exp(R_eigenvalues).unsqueeze(-2) @ R_eigenvectors.transpose(-1, -2)

    # Retraction: Σ_{new} = Σ^{1/2} exp(R) Σ^{1/2}
    Sigma_new = Sigma_sqrt @ exp_R @ Sigma_sqrt
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.transpose(-1, -2))

    # Post-retraction spectral floor
    eig_new, vec_new = torch.linalg.eigh(Sigma_new)
    eig_new = eig_new.clamp(min=eps)
    Sigma_new = vec_new * eig_new.unsqueeze(-2) @ vec_new.transpose(-1, -2)

    Sigma_new = Sigma_new.to(orig_dtype)
    if len(original_shape) == 4:
        Sigma_new = Sigma_new.reshape(original_shape)

    return Sigma_new


# =============================================================================
# SPD Retraction (Diagonal Covariance)
# =============================================================================

def retract_spd_diagonal(
    sigma_diag: torch.Tensor,
    delta_sigma: torch.Tensor,
    step_size: float = 1.0,
    trust_region: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """SPD retraction for diagonal covariances.

    For diagonal matrices, the exponential map reduces to:
        σ_new = σ * exp(τ * δσ / σ)

    This ensures positivity: exp(x) > 0 for all x.

    Args:
        sigma_diag: Diagonal variances, shape (B, N, K)
        delta_sigma: Tangent in diagonal form, shape (B, N, K)
        step_size: Learning rate τ
        trust_region: Max absolute value of exponent argument
        eps: Floor for sigma values

    Returns:
        sigma_new: Positive diagonal variances, shape (B, N, K)
    """
    orig_dtype = sigma_diag.dtype
    sigma_safe = sigma_diag.float().clamp(min=eps)
    delta_sigma = delta_sigma.float()

    # Whitened tangent: δσ / σ
    whitened = delta_sigma / sigma_safe

    if trust_region is not None and trust_region > 0:
        whitened = whitened.clamp(-trust_region, trust_region)

    # Exponential update: σ_new = σ * exp(τ * whitened)
    exp_arg = (step_size * whitened).clamp(-50.0, 50.0)
    sigma_new = sigma_safe * torch.exp(exp_arg)

    return sigma_new.clamp(min=eps, max=100.0).to(orig_dtype)


# =============================================================================
# Phi Retraction (Gauge Frame Updates)
# =============================================================================

def retract_phi(
    phi: torch.Tensor,
    delta_phi: torch.Tensor,
    generators: torch.Tensor,
    step_size: float,
    trust_region: Optional[float] = None,
    max_norm: Optional[float] = None,
    bch_order: Optional[int] = None,
    eps: float = 1e-6,
    gauge_group: Optional[str] = None,
) -> torch.Tensor:
    """Retract phi update using appropriate method for gauge group.

    Auto-selects defaults based on gauge group:
    - SO(N): compact, trust_region=0.3, max_norm=π, bch_order=1
    - GL(K): non-compact, trust_region=0.1, max_norm=1.0, bch_order=0

    Args:
        phi: Current gauge frames (..., n_gen)
        delta_phi: Update direction (..., n_gen)
        generators: Lie algebra generators (n_gen, dim, dim)
        step_size: Learning rate
        trust_region: Maximum relative change per update (auto if None)
        max_norm: Maximum norm for phi (auto if None)
        bch_order: BCH expansion order (auto if None)
        eps: Numerical stability
        gauge_group: 'GLK', 'SON', or None for auto-detect

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
        is_son = RETRACTION_AVAILABLE and is_soN_generators(n_gen)
        is_glk = RETRACTION_AVAILABLE and (is_glK_generators(n_gen) or not is_son)

    # Auto-select defaults
    if trust_region is None:
        trust_region = 0.1 if is_glk else 0.3
    if max_norm is None:
        max_norm = 1.0 if is_glk else math.pi
    if bch_order is None:
        bch_order = 0 if is_glk else 1

    if not RETRACTION_AVAILABLE:
        return _retract_phi_fallback(phi, delta_phi, step_size, trust_region, max_norm, eps)

    if is_glk:
        return retract_glK_torch(
            phi=phi, delta_phi=delta_phi, generators=generators,
            step_size=step_size, trust_region=trust_region,
            max_norm=max_norm, bch_order=bch_order, eps=eps, grad_clip=10.0,
        )
    elif is_son:
        return retract_soN_torch(
            phi=phi, delta_phi=delta_phi, generators=generators,
            step_size=step_size, trust_region=trust_region,
            max_norm=max_norm, bch_order=bch_order, eps=eps,
        )
    else:
        return _retract_phi_fallback(phi, delta_phi, step_size, trust_region, max_norm, eps)


def _retract_phi_fallback(phi, delta_phi, step_size, trust_region, max_norm, eps):
    """Simple gradient descent fallback for phi retraction."""
    update = step_size * delta_phi
    phi_norm = torch.norm(phi, dim=-1, keepdim=True).clamp(min=0.1)
    update_norm = torch.norm(update, dim=-1, keepdim=True)
    scale = torch.clamp(trust_region * phi_norm / (update_norm + eps), max=1.0)
    phi_new = phi + scale * update
    phi_new_norm = torch.norm(phi_new, dim=-1, keepdim=True)
    phi_new = torch.where(
        phi_new_norm > max_norm,
        phi_new * max_norm / (phi_new_norm + eps),
        phi_new
    )
    return phi_new


# =============================================================================
# SO(3) Lie Algebra Utilities
# =============================================================================

def so3_log(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Logarithm map from SO(3) → so(3).

    Given R ∈ SO(3), find φ ∈ ℝ³ such that exp([φ]_×) = R.

    Args:
        R: Rotation matrices, shape (..., 3, 3)
        eps: Threshold for small angle approximation

    Returns:
        phi: Lie algebra elements, shape (..., 3)
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)

    skew = R - R.transpose(-1, -2)
    v_x = (skew[..., 2, 1] - skew[..., 1, 2]) / 2.0
    v_y = (skew[..., 0, 2] - skew[..., 2, 0]) / 2.0
    v_z = (skew[..., 1, 0] - skew[..., 0, 1]) / 2.0
    vex_skew = torch.stack([v_x, v_y, v_z], dim=-1)

    sin_theta = torch.sin(theta)
    small_angle = theta < eps
    coeff = torch.where(
        small_angle,
        0.5 + theta**2 / 12.0,
        theta / (2.0 * sin_theta + eps)
    )

    return coeff.unsqueeze(-1) * vex_skew


def so3_compose_bch(
    phi1: torch.Tensor,
    phi2: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """Compose two so(3) elements using Baker-Campbell-Hausdorff formula.

    log(exp(φ₁)·exp(φ₂)) = φ₁ + φ₂ + ½[φ₁,φ₂] + ...
    For so(3), the Lie bracket is: [X, Y] = X × Y (cross product)

    Args:
        phi1: First so(3) element, shape (..., 3)
        phi2: Second so(3) element, shape (..., 3)
        order: BCH expansion order (0=addition, 1=first correction, 2=second)

    Returns:
        phi_composed: Composed element in so(3), shape (..., 3)
    """
    if order == 0:
        return phi1 + phi2

    bracket_12 = torch.cross(phi1, phi2, dim=-1)
    result = phi1 + phi2 + 0.5 * bracket_12

    if order >= 2:
        bracket_1_12 = torch.cross(phi1, bracket_12, dim=-1)
        bracket_2_12 = torch.cross(phi2, bracket_12, dim=-1)
        result = result + (1.0 / 12.0) * bracket_1_12 - (1.0 / 12.0) * bracket_2_12

    return result
