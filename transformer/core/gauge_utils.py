"""
Shared Gauge-Geometry Utilities
================================

Shared utilities for gauge transport computations used across
attention.py, variational_ffn.py, and embeddings.py.

Consolidates duplicated matrix exponential and KL divergence patterns.
"""

import torch
from typing import Tuple


def stable_matrix_exp_pair(
    matrix: torch.Tensor,
    dim_threshold: int = 8,
    max_norm: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute exp(M) and exp(-M) with norm clamping and float64 upcasting.

    Two stability measures:
    1. Frobenius norm clamping: caps ‖M‖_F at max_norm to prevent the
       Padé scaling-squaring algorithm from overflowing. For GL⁺(K),
       exp(M) with ‖M‖ >> 1 produces extreme condition numbers that
       make downstream Ω Σ Ω^T numerically non-positive-definite.
    2. Float64 upcasting for K >= dim_threshold.

    Note on surjectivity:
        exp(M) always has det > 0 (since det(exp(M)) = exp(tr(M))), so the
        outputs live in GL⁺(K), the identity component.

        Even within GL⁺(K), a single exp(M) is NOT surjective for K > 1.
        By Culver (1966), A ∈ GL(K,ℝ) has a real log iff for each negative
        real eigenvalue, the number of Jordan blocks of each size is even.
        E.g. diag(-2, -3) has det = 6 > 0 but no real logarithm.

        This does not limit transport: Ω_ij = exp(M_i)·exp(-M_j) is a free
        product of two exponentials, which covers all of GL⁺(K) (by polar
        decomposition: A = exp(log P)·exp(log O) where P sym.pos.def., O ∈ SO).
        For SO(K), exp: so(K) → SO(K) is surjective — no issues.

    Args:
        matrix: (..., d, d) matrix to exponentiate.
        dim_threshold: Upcast to float64 when d >= this value. Default 8.
        max_norm: Maximum Frobenius norm for input matrix. Default 10.0.

    Returns:
        (exp_pos, exp_neg): Tuple of exp(M) and exp(-M), both same dtype as input.
    """
    # Clamp Frobenius norm to prevent overflow in matrix_exp.
    # Gradient flows through the scaling factor, so φ still gets
    # signal to shrink when it exceeds the cap.
    mat_norm = matrix.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    scale = (max_norm / mat_norm).clamp(max=1.0)
    matrix = matrix * scale

    d = matrix.shape[-1]
    # Always compute matrix_exp in at least float32 to avoid AMP float16 disasters.
    # For large K (>= dim_threshold), use float64 as before.
    orig_dtype = matrix.dtype
    with torch.amp.autocast('cuda', enabled=False):
        if d >= dim_threshold:
            matrix_f64 = matrix.double()
            exp_pos = torch.linalg.matrix_exp(matrix_f64).to(orig_dtype)
            exp_neg = torch.linalg.matrix_exp(-matrix_f64).to(orig_dtype)
        else:
            matrix_f32 = matrix.float()
            exp_pos = torch.linalg.matrix_exp(matrix_f32).to(orig_dtype)
            exp_neg = torch.linalg.matrix_exp(-matrix_f32).to(orig_dtype)
    return exp_pos, exp_neg


def newton_schulz_orthogonalize(
    X: torch.Tensor,
    n_iters: int = 5,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Project a matrix to SO(K) via Newton-Schulz iteration.

    The iteration X_{k+1} = X_k @ (3I - X_k^T X_k) / 2 converges to the
    nearest orthogonal matrix when all singular values of X_0 are in (0, √3).

    For matrices with singular values outside this basin, we first rescale
    by the largest singular value estimate (Frobenius norm / √K) to bring
    the iterate into the convergence basin.

    Args:
        X: (..., K, K) matrices to orthogonalize.
        n_iters: Maximum Newton-Schulz iterations. Default 5.
        tol: Early stopping tolerance on ||X^T X - I||_F. Default 1e-6.

    Returns:
        Orthogonalized matrices (..., K, K), approximately in O(K).
    """
    K = X.shape[-1]
    eye = torch.eye(K, device=X.device, dtype=X.dtype)

    # Rescale to bring singular values near 1 (convergence basin: (0, √3)).
    # Frobenius norm / sqrt(K) estimates the RMS singular value.
    frob = X.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    rms_sv = frob / (K ** 0.5)
    # Only rescale if RMS singular value > 1.5 (well inside basin if near 1)
    needs_rescale = (rms_sv > 1.5).squeeze(-1).squeeze(-1)
    if needs_rescale.any():
        X = torch.where(
            needs_rescale[..., None, None],
            X / rms_sv,
            X,
        )

    for _ in range(n_iters):
        XtX = X.transpose(-1, -2) @ X
        deviation = XtX - eye
        dev_norm = deviation.norm(dim=(-2, -1))
        if (dev_norm < tol).all():
            break
        X = X @ ((3.0 * eye - XtX) / 2.0)

    return X
