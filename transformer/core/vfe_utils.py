"""
VFE Utility Functions
=====================

Utility functions and constants extracted from variational_ffn.py.  These are
pure mathematical helpers — no nn.Module, no learned parameters.

Provides:
- Module-level debug dict ``_VFE_GRAD_DEBUG``
- Numerical constants (SIGMA_EPS, TRANSPORT_JITTER, …)
- squeeze_trailing_singletons — collapse spurious trailing singleton dims
- _safe_spd_inv  — robust SPD matrix inversion with adaptive regularization
- _safe_eigh     — eigendecomposition with escalating jitter + SVD fallback
- retract_spd_torch          — affine-invariant SPD exponential map (full matrix)
- retract_spd_diagonal_torch — exponential retraction for diagonal covariances
- _retract_phi   — phi retraction to Lie group (SO(N) or GL(K))
- _grad_norm      — global Frobenius norm helper for debug logging
- _per_pos_stats  — per-position norm statistics for debug logging
- _aggregate_multihead_vfe_debug — aggregate per-head VFE debug metrics
"""

import logging
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable
from math_utils.numerical_monitor import record as _nr

logger = logging.getLogger(__name__)

# =============================================================================
# Diagnostics toggle
# =============================================================================
# Gates host-syncing instrumentation across the gradient/transport hot path:
#  - safe_kl_clamp's `_nr("kl_nonfinite")` / `_nr("kl_saturated")`
#  - _fused_attention_and_vfe_gradients_block_diag's `_nr("fused_vfe_*_nan")`
#  - stable_matrix_exp_pair's `_nr("matexp_norm_clamp")`
# Off by default — each emit-site otherwise fires a `.any()` + `.sum().item()`
# host sync per call. Set env-var VFE_KL_DIAGNOSTICS=1 or call
# enable_kl_diagnostics() to turn the whole observability layer on for a
# diagnostic sweep.
_KL_DIAGNOSTICS_ENABLED: bool = bool(int(os.environ.get("VFE_KL_DIAGNOSTICS", "0")))


def enable_kl_diagnostics(enabled: bool = True) -> None:
    """Toggle host-sync diagnostic counters at runtime."""
    global _KL_DIAGNOSTICS_ENABLED
    _KL_DIAGNOSTICS_ENABLED = bool(enabled)


def kl_diagnostics_enabled() -> bool:
    """Return whether host-sync diagnostic counters are currently enabled."""
    return _KL_DIAGNOSTICS_ENABLED

# =============================================================================
# Module-level constants
# =============================================================================
SIGMA_EPS: float = 1e-6
TRANSPORT_JITTER: float = 1e-4
KL_CEIL_BASE: float = 100.0
KL_CEIL_SCALE: float = 5.0

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


def _aggregate_multihead_vfe_debug(d: Dict[str, float], irrep_dims: Optional[Sequence[int]]) -> None:
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

def _safe_spd_inv(M: torch.Tensor, eps: float = 1e-6,
                  exp_phi: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Robust inversion for SPD (symmetric positive-definite) covariance matrices.

    Uses Cholesky decomposition as the primary path (via
    ``torch.linalg.cholesky_ex`` + ``torch.cholesky_inverse``), with
    adaptive regularization: max(eps, eps * ||M||_diag_max).  On failure,
    escalates the jitter by 10x (up to 3 rounds) before falling back to
    pseudoinverse.

    **Gauge invariance**: The Cholesky factor L is NOT gauge covariant
    (the sandwich product ``Ω @ Σ @ Ω.T`` does not commute with Cholesky).
    However, ``Σ^{-1} = L^{-T} L^{-1}`` IS gauge covariant as a precision:
    under a global gauge ``g``, ``Σ → g Σ g.T`` implies
    ``Σ^{-1} → g^{-T} Σ^{-1} g^{-1}``, identically to the output of
    ``torch.linalg.inv``.  So Cholesky is safe here because ``L`` is never
    stored, transported, or compared across frames — only the final
    inverse is returned.

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

        # Symmetrize before Cholesky — roundoff asymmetry breaks Cholesky on
        # otherwise-SPD matrices (sandwich products can leave O(eps_machine)
        # off-diagonal asymmetry).
        M_sym = 0.5 * (M + M.transpose(-1, -2))

        # Adaptive regularization: scale eps by matrix magnitude so the
        # regularization is proportional to the matrix scale.
        diag_max = M_sym.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1, keepdim=True).unsqueeze(-1)
        reg_scale = torch.clamp(diag_max * eps, min=eps)  # (..., 1, 1)
        I_K = torch.eye(K, device=device, dtype=torch.float32)
        if exp_phi is not None:
            # Gauge-covariant ridge: transforms as Σ under h·Σ·h^T.
            _gf = exp_phi.to(dtype=torch.float32)
            R_unit = _gf @ _gf.transpose(-1, -2)
        else:
            R_unit = I_K
        M_reg = M_sym + reg_scale * R_unit

        # Primary path: Cholesky.  cholesky_ex returns (L, info) where
        # info[b] != 0 indicates batch element b failed.  We do not let
        # it raise — we inspect info and escalate regularization.
        L, info = torch.linalg.cholesky_ex(M_reg)
        if not info.any():
            result = torch.cholesky_inverse(L)
        else:
            # Secondary path: escalating jitter applied ONLY to failed
            # batch elements.  We keep the already-successful results from
            # the first pass and retry only the failures with increasing
            # regularization, avoiding unnecessary perturbation to
            # well-conditioned matrices.
            _nr("spd_inv_chol_escalate")
            result = torch.cholesky_inverse(L)  # partial: good where info==0
            failed = (info != 0)  # boolean mask over batch dims

            current_reg = reg_scale
            success = False
            for _ in range(3):
                current_reg = current_reg * 10.0
                M_reg_more = M_sym + current_reg * R_unit
                L_retry, info_retry = torch.linalg.cholesky_ex(M_reg_more)
                # Only overwrite elements that were still failing
                still_failed = failed & (info_retry != 0)
                newly_fixed = failed & (info_retry == 0)
                if newly_fixed.any():
                    inv_retry = torch.cholesky_inverse(L_retry)
                    # Expand failed mask to matrix dims for where()
                    mask_expand = newly_fixed
                    while mask_expand.dim() < result.dim():
                        mask_expand = mask_expand.unsqueeze(-1)
                    mask_expand = mask_expand.expand_as(result)
                    result = torch.where(mask_expand, inv_retry, result)
                failed = still_failed
                if not failed.any():
                    success = True
                    break
            if failed.any():
                # Last resort for remaining failures: pseudoinverse.
                _nr("spd_inv_pseudoinverse")
                try:
                    inv_last = torch.linalg.inv(M_reg)
                except (torch.linalg.LinAlgError, RuntimeError):
                    inv_last = torch.linalg.pinv(M_reg)
                mask_expand = failed
                while mask_expand.dim() < result.dim():
                    mask_expand = mask_expand.unsqueeze(-1)
                mask_expand = mask_expand.expand_as(result)
                result = torch.where(mask_expand, inv_last, result)

    return result.to(orig_dtype)


def _safe_eigh(
    M: torch.Tensor,
    jitter: float = 1e-6,
    max_jitter: float = 1e-2,
    symmetrize: bool = True,
    exp_phi: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Robust eigendecomposition for symmetric matrices with escalating jitter.

    Retries ``torch.linalg.eigh`` with geometrically increasing jitter
    (×10 per retry) until ``max_jitter`` is reached. If all retries fail,
    falls back to SVD which uses a different algorithm (gesdd vs syev) and
    is more tolerant of ill-conditioning. For symmetric PSD matrices the
    singular values equal the eigenvalues and U provides the eigenvectors.

    Adds a small linearly-spaced diagonal perturbation to break exact
    eigenvalue degeneracies, preventing NaN in the backward pass (the
    ``eigh`` gradient involves ``1/(λ_i - λ_j)`` terms that diverge
    when eigenvalues coincide).

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
        if exp_phi is not None:
            _gf = exp_phi.to(dtype=torch.float32)
            R_unit = _gf @ _gf.transpose(-1, -2)
        else:
            R_unit = I_K
        # Break eigenvalue degeneracy: add linearly-spaced diagonal perturbation
        # so that eigenvalues of identical magnitude are separated by ~jitter/K.
        # This prevents NaN in the eigh backward pass (1/(λ_i - λ_j) terms).
        degeneracy_breaker = torch.linspace(
            0, jitter, K, device=device, dtype=torch.float32
        ).diag()
        current_jitter = jitter

        while current_jitter <= max_jitter:
            try:
                M_reg = M + current_jitter * R_unit + degeneracy_breaker
                eigvals, eigvecs = torch.linalg.eigh(M_reg)
                return eigvals.to(orig_dtype), eigvecs.to(orig_dtype)
            except (RuntimeError, torch.linalg.LinAlgError):
                _nr("eigh_jitter_escalate")
                current_jitter *= 10.0
                # Scale degeneracy breaker with jitter
                degeneracy_breaker = torch.linspace(
                    0, current_jitter, K, device=device, dtype=torch.float32
                ).diag()

        _nr("eigh_svd_fallback")
        # Ultimate fallback: SVD (gesdd algorithm, more numerically stable).
        # For symmetric PSD M: singular values = eigenvalues ≥ 0, U = V.
        # For symmetric indefinite M (after insufficient jitter): SVD returns
        # |λ_i| not λ_i.  We recover signs from the diagonal of U^T M U,
        # which equals diag(λ_i) for eigenvectors U of a symmetric matrix.
        M_reg = M + max_jitter * R_unit + degeneracy_breaker
        U, s, Vh = torch.linalg.svd(M_reg, full_matrices=False)
        # Recover eigenvalue signs: diag(U^T M_reg U) = eigenvalues
        diag_check = torch.einsum('...ki,...kj,...ij->...i', U, M_reg, U)
        # Where the diagonal is negative, the true eigenvalue is -s
        s_signed = torch.where(diag_check < 0, -s, s)
        return s_signed.to(orig_dtype), U.to(orig_dtype)


# =============================================================================
# SPD eigenvalue floor
# =============================================================================

def spd_eigfloor(
    Sigma: torch.Tensor,
    floor: float,
    sigma_max: Optional[float] = None,
    exp_phi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Enforce :math:`\lambda_{\min}(\Sigma) \geq \mathrm{floor}` via spectral clamp.

    Two paths:

    **Frame-dependent (default, ``exp_phi=None``).**
    Computes :math:`\Sigma = V \Lambda V^\top` via ``_safe_eigh`` and returns
    :math:`V \, \mathrm{clamp}(\Lambda, \mathrm{floor}, \sigma_{\max}^2) \, V^\top`.
    Bounds the condition number
    :math:`\kappa(\Sigma) \leq \sigma_{\max}^2 / \mathrm{floor}`.
    Not gauge-covariant under GL(K): a transported :math:`h \Sigma h^\top`
    has eigenvectors that differ from :math:`h V`, so the clamp produces
    numerically different matrices in different frames.

    **Gauge-covariant (``exp_phi=A`` provided).**
    Whitens via :math:`W = A^{-1} \Sigma A^{-\top}`, clamps eigenvalues of
    :math:`W`, then de-whitens: :math:`\Sigma' = A W_{\mathrm{clamped}} A^\top`.
    Under the transformation
    :math:`\Sigma \mapsto h \Sigma h^\top, \; A \mapsto h A`,
    :math:`A^{-1} \to A^{-1} h^{-1}`, so
    :math:`W \to A^{-1} h^{-1} (h \Sigma h^\top) h^{-\top} A^{-\top} = W`
    is invariant; the clamp preserves this invariance, and de-whitening
    with :math:`hA` reintroduces the correct scaling. Therefore
    :math:`\mathrm{spd\_eigfloor}(h \Sigma h^\top, \mathrm{exp\_phi}=hA)
    = h \cdot \mathrm{spd\_eigfloor}(\Sigma, \mathrm{exp\_phi}=A) \cdot h^\top`
    holds exactly (up to numerical roundoff in the matrix inverse).

    Args:
        Sigma: (..., K, K) symmetric matrices.
        floor: Minimum eigenvalue; must be positive.
        sigma_max: If set, also clamp eigenvalues to
            :math:`\lambda \leq \sigma_{\max}^2` (upper bound on variance).
        exp_phi: Optional (..., K, K) local frame :math:`A \in GL(K)`.
            Must be invertible; for :math:`A = \exp(\phi \cdot G)` this is
            automatic. When provided, the strict gauge-covariant path is
            taken (one matrix inverse + two matmuls on top of the eigh cost).

    Returns:
        Clamped SPD matrix with same shape and dtype as ``Sigma``.
    """
    orig_dtype = Sigma.dtype

    if exp_phi is None:
        # Frame-dependent path — cheap, frame-anchored clamp.
        eig, vec = _safe_eigh(Sigma, jitter=floor)
        if sigma_max is not None:
            eig = eig.clamp(min=floor, max=sigma_max * sigma_max)
        else:
            eig = eig.clamp(min=floor)
        out = (vec * eig.unsqueeze(-2)) @ vec.transpose(-1, -2)
        out = 0.5 * (out + out.transpose(-1, -2))
        return out.to(orig_dtype)

    # Gauge-covariant path: whiten via A^{-1}, clamp in the whitened frame,
    # de-whiten via A. Force float32: inverting near-singular A is the
    # critical numerical step and breaks under fp16.
    with torch.amp.autocast('cuda', enabled=False):
        A = exp_phi.to(dtype=torch.float32)
        Sigma_f = Sigma.to(dtype=torch.float32)
        # Compute W = A^{-1} Σ A^{-T} via two lu_solves against a shared
        # factorization. Avoids materializing A^{-1} explicitly, which
        # would lose precision in fp32 when cond(A) is large under
        # compounded matrix_exp factors.
        LU, pivots = torch.linalg.lu_factor(A)
        left = torch.linalg.lu_solve(LU, pivots, Sigma_f)              # A^{-1} Σ
        W = torch.linalg.lu_solve(LU, pivots, left.transpose(-1, -2))  # A^{-1} (A^{-1} Σ)^T
        W = W.transpose(-1, -2)                                        # = A^{-1} Σ A^{-T}
        W = 0.5 * (W + W.transpose(-1, -2))

        # Eigh in the whitened frame: W is SPD under gauge action, so the
        # standard jitter·I perturbation is already frame-covariant here.
        eig, vec = _safe_eigh(W, jitter=floor)
        if sigma_max is not None:
            eig = eig.clamp(min=floor, max=sigma_max * sigma_max)
        else:
            eig = eig.clamp(min=floor)
        W_clamped = (vec * eig.unsqueeze(-2)) @ vec.transpose(-1, -2)
        W_clamped = 0.5 * (W_clamped + W_clamped.transpose(-1, -2))

        out = A @ W_clamped @ A.transpose(-1, -2)
        out = 0.5 * (out + out.transpose(-1, -2))
    return out.to(orig_dtype)


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
        _sigma_max_sq = sigma_max * sigma_max
        _oob = (eig_new < eps) | (eig_new > _sigma_max_sq)
        if _oob.any():
            _nr("spd_retract_eigenclamp", count=int(_oob.sum().item()))
        eig_new = eig_new.clamp(min=eps, max=_sigma_max_sq)
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
        _raw_exp_arg = step_size * whitened
        _exp_clip = (_raw_exp_arg < -50.0) | (_raw_exp_arg > 50.0)
        if _exp_clip.any():
            _nr("diag_retract_exp_clip", count=int(_exp_clip.sum().item()))
        exp_arg = _raw_exp_arg.clamp(-50.0, 50.0)
        sigma_new = sigma_safe * torch.exp(exp_arg)

    # Clamp to [eps, sigma_max] — posterior σ should not blow up past the prior.
    # With sigma_max=5.0 and init_sigma_scale=1.0, allows 5× expansion before clamping.
    # This bounds the natural gradient amplification: 2σ² ≤ 2·sigma_max² = 50.
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
    trust_region: Optional[float] = None,  # None = auto-select based on gauge group
    max_norm: Optional[float] = None,  # None = auto-select based on gauge group
    bch_order: Optional[int] = None,  # None = auto-select based on gauge group
    eps: float = 1e-6,
    gauge_group: Optional[str] = None,  # Explicit: 'GLK', 'SON', or None for auto-detect
    project_slk: bool = False,  # Hard project φ → sl(K) per block (det Ω_h = 1)
    trace_clamp: Optional[float] = None,  # Soft per-block cap |tr(φ·G_h)| ≤ T
    irrep_dims: Optional[List[int]] = None,  # Required for project_slk/trace_clamp on multi-block GL(K)
) -> torch.Tensor:
    """
    Retract phi update using appropriate method for gauge group.

    When gauge_group is provided, uses it directly. Otherwise auto-selects
    based on n_gen:
    - n_gen = N(N-1)/2 → SO(N): compact, uses trust_region=0.3, max_norm=π, bch_order=4
    - n_gen = K²       → GL(K): non-compact, uses trust_region=0.1, max_norm=5.0, bch_order=4

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
        bch_order = 4  # Degree-5 BCH (all 6 Dynkin terms; O(ε^6) error)

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
        return _apply_det_control(phi_new, generators, is_glk, project_slk, trace_clamp, irrep_dims, eps)

    # Check if this is GL(K) (n_gen = K²) or SO(N) (n_gen = N(N-1)/2)
    if is_glk:
        # GL(K) is non-compact - needs conservative settings
        phi_new = retract_glK_torch(
            phi=phi,
            delta_phi=delta_phi,
            generators=generators,
            step_size=step_size,
            trust_region=trust_region,
            max_norm=max_norm,
            bch_order=bch_order,
            eps=eps,
        )
        return _apply_det_control(phi_new, generators, True, project_slk, trace_clamp, irrep_dims, eps)
    elif is_son:
        # SO(N) is compact - can use standard settings; tr(G_a)=0 already so
        # no determinant control needed (det=1 automatically).
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
        return _apply_det_control(phi_new, generators, is_glk, project_slk, trace_clamp, irrep_dims, eps)


def _apply_det_control(
    phi: torch.Tensor,
    generators: torch.Tensor,
    is_glk: bool,
    project_slk: bool,
    trace_clamp: Optional[float],
    irrep_dims: Optional[List[int]] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""Constrain the per-block determinant direction(s) of φ for GL(K) groups.

    For a block-diagonal gauge group :math:`GL(K_1) \oplus \cdots \oplus GL(K_H)`,
    each block carries an independent trace functional
    :math:`s_h = \operatorname{tr}(\phi \cdot G^{(h)})` and an independent
    determinant. L2-norm clamping does not bound any of them, so
    :math:`\det(\Omega_h)` can blow up per block. Killing only the summed
    direction :math:`\sum_h s_h` is **insufficient** — compensating signs
    across blocks defeat the constraint.

    Behavior:
      * No-op if not is_glk (SO(N) generators are skew, tr=0 by construction).
      * No-op if both ``project_slk=False`` and ``trace_clamp is None``.
      * If ``project_slk`` set: per-block projection so :math:`s_h \equiv 0`
        for every block ⇒ :math:`\det(\Omega_h) \equiv 1`. Drops one DOF per
        block. Block-diagonal trace vectors are mutually orthogonal so the
        projection collapses to one matmul.
      * Else (``trace_clamp=T``): rescale only the trace component per block
        so :math:`|s_h| \le T`. Soft cap; preserves trace direction within bounds.
      * If both set, ``project_slk`` wins.
      * If ``irrep_dims is None``, falls back to single-block treatment
        (entire K-dim space treated as one GL(K) block).
    """
    if not is_glk:
        return phi
    if not project_slk and trace_clamp is None:
        return phi
    # Build per-block trace vectors V[h, a] = tr(G_a restricted to block h).
    # For block-diagonal generators these vectors have disjoint supports, so
    # they are automatically orthogonal and a single matmul handles all blocks.
    n_gen = generators.shape[0]
    K = generators.shape[-1]
    if irrep_dims is None:
        irrep_dims = [K]  # treat as single GL(K) block
    H = len(irrep_dims)
    V_blocks = torch.zeros(H, n_gen, dtype=phi.dtype, device=phi.device)
    start = 0
    for h, d_h in enumerate(irrep_dims):
        end = start + d_h
        V_blocks[h] = generators[:, start:end, start:end].diagonal(
            dim1=-2, dim2=-1
        ).sum(dim=-1).to(phi.dtype)
        start = end
    v_norm_sq = (V_blocks * V_blocks).sum(dim=-1).clamp(min=eps)  # (H,)
    s = phi @ V_blocks.transpose(-2, -1)                          # (..., H)
    if project_slk:
        coeffs = s / v_norm_sq                                    # (..., H)
        return phi - torch.einsum('...h,hg->...g', coeffs, V_blocks)
    # Soft per-block clamp on s_h
    T = float(trace_clamp)
    s_clamped = s.clamp(min=-T, max=T)
    delta_coeffs = (s_clamped - s) / v_norm_sq                    # (..., H)
    return phi + torch.einsum('...h,hg->...g', delta_coeffs, V_blocks)


def build_slk_basis(
    generators: torch.Tensor,
    irrep_dims: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Construct per-block trace vectors and an orthonormal basis of the
    traceless subalgebra for a block-diagonal gauge group.

    For :math:`GL(K_1) \oplus \cdots \oplus GL(K_H)` with ``n_gen`` generators,
    each block contributes a trace functional :math:`V_h[a] = \operatorname{tr}
    (G_a|_{\text{block }h})`. The traceless subalgebra
    :math:`sl(K_1) \oplus \cdots \oplus sl(K_H)` is the orthogonal complement of
    :math:`\operatorname{span}\{V_h\}`, dimension :math:`n_{\text{gen}} - H`.

    Returns:
        V_blocks: ``(H, n_gen)`` per-block trace vectors (disjoint supports,
            mutually orthogonal for block-diagonal generators).
        P: ``(n_gen, n_gen - H)`` orthonormal basis whose columns span the
            traceless complement. ``c @ P.T`` maps free coordinates of length
            ``n_gen - H`` to a traceless element of the full algebra.
    """
    n_gen = generators.shape[0]
    K = generators.shape[-1]
    if irrep_dims is None:
        irrep_dims = [K]
    H = len(irrep_dims)

    V_blocks = torch.zeros(H, n_gen, dtype=generators.dtype, device=generators.device)
    start = 0
    for h, d_h in enumerate(irrep_dims):
        end = start + d_h
        V_blocks[h] = generators[:, start:end, start:end].diagonal(
            dim1=-2, dim2=-1
        ).sum(dim=-1)
        start = end

    # Project out span{V_h} and pick an orthonormal basis of the complement.
    # Compute numerically in float32 for stability, then cast back.
    V32 = V_blocks.to(torch.float32)
    v_norm_sq = (V32 * V32).sum(dim=-1).clamp(min=1e-12)           # (H,)
    # I - Σ_h V_h V_h^T / ||V_h||²  ∈ R^{n_gen × n_gen}
    I = torch.eye(n_gen, dtype=torch.float32, device=generators.device)
    proj = I - (V32.T * (1.0 / v_norm_sq)) @ V32                   # (n_gen, n_gen)
    # Eigendecompose the projector; eigenvectors with eigenvalue ≈ 1 span sl.
    eigvals, eigvecs = torch.linalg.eigh(proj)
    # Sort descending so the first n_gen - H columns are the sl basis.
    order = torch.argsort(eigvals, descending=True)
    P32 = eigvecs[:, order[: n_gen - H]].contiguous()              # (n_gen, n_gen - H)
    P = P32.to(generators.dtype)
    return V_blocks, P


# =============================================================================
# Trajectory recording protocol (core-side)
# =============================================================================
# These live in core/ so that blocks.py, model.py, and variational_ffn.py
# can record trajectories without importing from analysis/.  The analysis
# layer registers its TrajectoryRecorder via set_global_recorder().

@dataclass
class IterationSnapshot:
    r"""Per-iteration belief state within a single layer's VFE E-step.

    Records the state of beliefs at one VFE iteration for a subset of
    token positions. Used for intra-fiber trajectory analysis: tracking
    how $(\mu, \Sigma)$ evolve through the Gaussian belief manifold
    $\mathcal{G}_K$ equipped with the Fisher-Rao metric.

    Attributes:
        iteration: E-step iteration index (0-based).
        mu: Belief means for recorded tokens, shape ``(N_recorded, K)``.
        sigma_diag: Diagonal covariance for recorded tokens, shape ``(N_recorded, K)``.
        beta_entropy: Mean attention entropy (scalar) at this iteration.
        grad_mu_norm: L2 norm of the Euclidean mu gradient (scalar).
        grad_sigma_norm: L2 norm of the Euclidean sigma gradient (scalar).
    """
    iteration: int
    mu: np.ndarray                      # (N_recorded, K)
    sigma_diag: np.ndarray              # (N_recorded, K)
    beta_entropy: float = 0.0
    grad_mu_norm: float = 0.0
    grad_sigma_norm: float = 0.0


@runtime_checkable
class RecorderProtocol(Protocol):
    """Structural type for a trajectory recorder.

    Conformers (`TrajectoryRecorder` in `analysis/trajectory.py`,
    `PublicationMetrics` in `analysis/publication_metrics.py`) implement
    these with concrete shared-prefix signatures; the trailing `**kwargs`
    absorbs conformer-specific optional arguments.
    """
    def record_step(self, step: int, epoch: float,
                    train_metrics: Dict[str, float], **kwargs: Any) -> None: ...
    def start_forward(self, batch_size: int, seq_len: int,
                      **kwargs: Any) -> None: ...


_global_recorder: Optional[RecorderProtocol] = None


def get_global_recorder() -> Optional[RecorderProtocol]:
    """Return the global trajectory recorder, or None if not set."""
    return _global_recorder


def set_global_recorder(recorder: Optional[RecorderProtocol]) -> None:
    """Register a trajectory recorder from the analysis layer."""
    global _global_recorder
    _global_recorder = recorder


# =============================================================================
# Observation gradient computation (extracted from VariationalFFNDynamic)
# =============================================================================

# =============================================================================
# Sigma SPD retraction (extracted from VariationalFFNDynamic)
# =============================================================================

def apply_natural_gradient_step(
    grad_mu: torch.Tensor,
    grad_sigma: torch.Tensor,
    sigma_current: torch.Tensor,
    is_diagonal: bool,
    eps: float,
    effective_lr: float,
    isotropic_covariance: bool,
    compute_natural_gradient_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, float],
        Tuple[torch.Tensor, torch.Tensor],
    ],
) -> Dict[str, torch.Tensor]:
    r"""Compute natural gradients, clamp, apply trust region, and return update.

    This is the core STEP 3 + STEP 4 of the VFE E-step: Fisher projection,
    norm clamping, whitened trust-region mu update, and diagnostic scalars.

    Args:
        grad_mu: Euclidean mu gradient, ``(B, N, K)``.
        grad_sigma: Euclidean sigma gradient, ``(B, N, K)`` or ``(B, N, K, K)``.
        sigma_current: Current covariance, ``(B, N, K)`` or ``(B, N, K, K)``.
        is_diagonal: Whether covariance is diagonal.
        eps: Numerical stability epsilon.
        effective_lr: Decayed learning rate for this iteration.
        isotropic_covariance: Whether to project gradients isotropically.
        compute_natural_gradient_fn: Callable for Fisher projection.

    Returns:
        dict with keys:
            ``mu_new``, ``nat_grad_mu``, ``nat_grad_sigma``,
            ``nat_grad_mu_norm``, ``nat_grad_sigma_norm``,
            ``max_nat_grad_norm``, ``max_nat_grad_sigma_norm``,
            ``delta_mu``, ``scale``, ``whitened_norm``.
    """
    # Clip for stability
    grad_mu = torch.clamp(grad_mu, min=-1e3, max=1e3)
    grad_sigma = torch.clamp(grad_sigma, min=-1e3, max=1e3)

    # Isotropic gradient projection: average grad_sigma across dims
    if isotropic_covariance:
        if is_diagonal:
            grad_sigma = grad_sigma.mean(dim=-1, keepdim=True).expand_as(grad_sigma)
        else:
            diag_grad = torch.diagonal(grad_sigma, dim1=-2, dim2=-1)
            avg_grad = diag_grad.mean(dim=-1, keepdim=True)
            K = grad_sigma.shape[-1]
            grad_sigma = avg_grad.unsqueeze(-1) * torch.eye(
                K, device=grad_sigma.device, dtype=grad_sigma.dtype
            )

    # Natural gradient projection
    nat_grad_mu, nat_grad_sigma = compute_natural_gradient_fn(
        grad_mu, grad_sigma, sigma_current, eps=eps,
    )

    # Clamp mu natural gradient norm
    nat_grad_mu_norm = torch.linalg.norm(nat_grad_mu, dim=-1, keepdim=True)
    max_nat_grad_norm = 500.0
    nat_grad_scale = torch.clamp(
        max_nat_grad_norm / (nat_grad_mu_norm + eps), max=1.0
    )
    nat_grad_mu = nat_grad_mu * nat_grad_scale

    # Clamp sigma natural gradient norm
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

    # Trust region mu update
    delta_mu = -effective_lr * nat_grad_mu
    if is_diagonal:
        sigma_sqrt = torch.sqrt(sigma_current.float().clamp(min=eps)).to(sigma_current.dtype)
        whitened_delta = delta_mu / sigma_sqrt
    else:
        sigma_diag = torch.diagonal(sigma_current, dim1=-2, dim2=-1).clone().float().clamp(min=eps)
        whitened_delta = delta_mu / torch.sqrt(sigma_diag).to(delta_mu.dtype)

    whitened_norm = torch.linalg.norm(whitened_delta, dim=-1, keepdim=True)
    mu_trust_region = 2.0
    scale = torch.clamp(mu_trust_region / (whitened_norm + eps), max=1.0)
    mu_new = scale * delta_mu  # Caller adds to mu_current

    return {
        'mu_delta': mu_new,
        'nat_grad_mu': nat_grad_mu,
        'nat_grad_sigma': nat_grad_sigma,
        'nat_grad_mu_norm': nat_grad_mu_norm,
        'nat_grad_sigma_norm': nat_grad_sigma_norm,
        'max_nat_grad_norm': max_nat_grad_norm,
        'max_nat_grad_sigma_norm': max_nat_grad_sigma_norm,
        'delta_mu': delta_mu,
        'scale': scale,
        'whitened_norm': whitened_norm,
        'grad_mu_clipped': grad_mu,
        'grad_sigma_clipped': grad_sigma,
    }


def retract_sigma_e_step(
    sigma_current: torch.Tensor,
    nat_grad_sigma: torch.Tensor,
    effective_lr: float,
    is_diagonal: bool,
    eps: float,
    update_sigma: bool,
    sigma_trust: float,
    sigma_max: float,
    isotropic_covariance: bool,
) -> torch.Tensor:
    r"""Apply the SPD-preserving retraction to ``sigma_current``.

    Three sequential operations:

    1. **Retraction** — ``retract_spd_diagonal_torch`` (diagonal) or
       ``retract_spd_torch`` (full), with trust region ``sigma_trust``.
    2. **Condition clamping** — prevent anisotropy ratio
       ``sigma_max / sigma_min > 10`` by nudging extreme dimensions
       toward the geometric mean.
    3. **Isotropic enforcement** (when ``isotropic_covariance``) —
       collapse to :math:`\sigma^2 I` after each update.

    Only runs when ``update_sigma`` is ``True``.

    Returns the (possibly unchanged) ``sigma_current``.
    """
    if update_sigma:
        sigma_trust_diag = sigma_trust
        sigma_trust_full = sigma_trust * 0.5  # Full cov more sensitive
        if is_diagonal:
            sigma_current = retract_spd_diagonal_torch(
                sigma_diag=sigma_current,
                delta_sigma=-nat_grad_sigma,
                step_size=effective_lr,
                trust_region=sigma_trust_diag,
                eps=eps,
                sigma_max=sigma_max,
            )
        else:
            sigma_current = retract_spd_torch(
                Sigma=sigma_current,
                delta_Sigma=-nat_grad_sigma,
                step_size=effective_lr,
                trust_region=sigma_trust_full,
                eps=eps,
                sigma_max=sigma_max,
            )

    # Sigma condition clamping
    if update_sigma:
        max_condition = 10.0
        if is_diagonal:
            s_min = sigma_current.min(dim=-1, keepdim=True).values.clamp(min=eps)
            s_max = sigma_current.max(dim=-1, keepdim=True).values
            condition = s_max / s_min
            needs_clamp = condition > max_condition
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
            try:
                eigvals = torch.linalg.eigvalsh(sigma_current)
            except (RuntimeError, torch.linalg.LinAlgError) as _e:
                # Surface the failure mode so an ill-conditioned σ propagating
                # downstream is not invisible. Apply a defensive eps·I ridge so
                # the next KL/inv call has a fighting chance against the same
                # near-singular block.
                import logging as _logging
                _logging.getLogger(__name__).debug(
                    "retract_sigma_e_step condition-clamp eigvalsh failed: %s: %s",
                    type(_e).__name__, _e,
                )
                eigvals = None
                K = sigma_current.shape[-1]
                sigma_current = sigma_current + eps * torch.eye(
                    K, device=sigma_current.device, dtype=sigma_current.dtype
                )
            if eigvals is not None:
                e_min = eigvals[..., 0:1].clamp(min=eps)
                e_max = eigvals[..., -1:]
                condition = e_max / e_min
                geo_mean = eigvals.log().mean(dim=-1, keepdim=True).exp()
                lower = geo_mean / (max_condition ** 0.5)
                ridge = (lower - e_min).clamp(min=0.0).mean(dim=-1, keepdim=True)
                K = sigma_current.shape[-1]
                sigma_current = sigma_current + ridge.unsqueeze(-1) * torch.eye(
                    K, device=sigma_current.device, dtype=sigma_current.dtype
                )

    # Isotropic covariance enforcement
    if update_sigma and isotropic_covariance:
        if is_diagonal:
            scalar_var = sigma_current.mean(dim=-1, keepdim=True)
            sigma_current = scalar_var.expand_as(sigma_current)
        else:
            diag_vals = torch.diagonal(sigma_current, dim1=-2, dim2=-1)
            scalar_var = diag_vals.mean(dim=-1, keepdim=True)
            K = sigma_current.shape[-1]
            sigma_current = scalar_var.unsqueeze(-1) * torch.eye(
                K, device=sigma_current.device, dtype=sigma_current.dtype
            )

    return sigma_current
