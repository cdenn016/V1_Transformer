"""
KL Divergence Computation — Unified Module
===========================================

Consolidates 9 previously scattered KL matrix computation variants into a
single parametric entry point with three focused kernel functions.

Three covariance modes:
    DENSE          — full (B, N, K, K) covariance; Cholesky-based KL
    DIAGONAL       — diagonal (B, N, K) covariance; closed-form, O(N²K)
    BLOCK_DIAGONAL — block-diagonal structure; delegates to fused kernels
                     in gauge_utils.py, reducing CUDA launches to O(unique dims)

Chunking is handled entirely inside ``compute_kl_matrix``; the three kernel
functions contain only the core mathematics.

Gauge equivariance invariant: transported covariance always uses the sandwich
product ``Omega @ Sigma @ Omega.T``.  This module never touches that transport
itself — the kernels receive already-transported tensors (mu_t, sigma_t).

Usage
-----
>>> from transformer.core.kl_computation import compute_kl_matrix, KLMode, safe_kl_clamp
>>> kl = compute_kl_matrix(mu_q, sigma_q, mu_t, sigma_t, mode=KLMode.DIAGONAL)
"""

from enum import Enum
from typing import List, Optional

import torch
from math_utils.numerical_monitor import record as _nr

from transformer.core.gauge_utils import (
    fused_block_diagonal_kl_diag,
    fused_block_diagonal_kl_full,
    fused_block_matrix_exp_pairs,
    stable_matrix_exp_pair,
    newton_schulz_orthogonalize,
)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "KLMode",
    "compute_kl_matrix",
    "safe_kl_clamp",
]


class KLMode(Enum):
    """Covariance representation used when computing the KL matrix."""
    DENSE = "dense"
    DIAGONAL = "diagonal"
    BLOCK_DIAGONAL = "block_diagonal"


# =============================================================================
# Utility
# =============================================================================

def safe_kl_clamp(kl: torch.Tensor, kl_max: float = 100.0) -> torch.Tensor:
    r"""Clamp a KL tensor to a finite, non-negative range.

    Applies ``clamp(0, kl_max)`` then replaces NaN/+inf with *kl_max* and
    -inf with 0.  Using ``nan=kl_max`` (repulsive) rather than ``nan=0.0``
    (attractive) ensures that numerically degenerate pairs are ignored by
    the downstream softmax rather than attended to.

    Args:
        kl: Tensor of (possibly un-clamped) KL values.
        kl_max: Upper bound.  Default 100.0.

    Returns:
        Clamped tensor, same shape and device as *kl*.
    """
    kl = kl.clamp(min=0.0, max=kl_max)
    return kl.nan_to_num(nan=kl_max, posinf=kl_max, neginf=0.0)


# =============================================================================
# Kernel Functions — pure math, no chunking logic
# =============================================================================

def _kl_kernel_dense(
    mu_q: torch.Tensor,      # (..., K) query means
    sigma_q: torch.Tensor,   # (..., K, K) query covariances
    mu_t: torch.Tensor,      # (..., K) transported key means
    sigma_t: torch.Tensor,   # (..., K, K) transported key covariances
    kl_max: float,
    eps: float,
) -> torch.Tensor:
    r"""Full-covariance KL divergence: KL(N(mu_q, Sigma_q) || N(mu_t, Sigma_t)).

    Formula:
        KL = 0.5 * (tr(Sigma_t^{-1} Sigma_q)
                    + (mu_t - mu_q)^T Sigma_t^{-1} (mu_t - mu_q)
                    - K
                    + log|Sigma_t| - log|Sigma_q|)

    Implemented via Cholesky for numerical stability.  Falls back to
    progressive diagonal regularisation on near-singular inputs.

    Args:
        mu_q: (..., K) query means.
        sigma_q: (..., K, K) query covariances.
        mu_t: (..., K) transported key means.
        sigma_t: (..., K, K) transported key covariances.
        kl_max: Clamp ceiling (typically ``max(100, 20*K)``).
        eps: Regularisation floor.

    Returns:
        kl: (...,) non-negative KL values.
    """
    K = mu_q.shape[-1]
    device = mu_q.device
    orig_dtype = mu_q.dtype

    # Force float32: Cholesky, solve_triangular, and log-det all break in fp16.
    mu_q = mu_q.float()
    sigma_q = sigma_q.float()
    mu_t = mu_t.float()
    sigma_t = sigma_t.float()

    I = torch.eye(K, device=device, dtype=torch.float32)
    sigma_q_reg = sigma_q + eps * I
    sigma_t_reg = sigma_t + eps * I

    # NaN guard: transported covariances can contain NaN when phi is very large.
    nan_mask = torch.isnan(sigma_t_reg).any(dim=-1).any(dim=-1)
    if nan_mask.any():
        _nr("nan_replace")
        sigma_t_reg = torch.where(
            nan_mask.unsqueeze(-1).unsqueeze(-1),
            I.expand_as(sigma_t_reg),
            sigma_t_reg,
        )

    def _cholesky_with_fallback(mat: torch.Tensor) -> torch.Tensor:
        try:
            return torch.linalg.cholesky(mat)
        except RuntimeError:
            reg = eps
            for _ in range(5):
                reg *= 10.0
                mat_reg = mat + (reg - eps) * I
                mat_reg = 0.5 * (mat_reg + mat_reg.transpose(-1, -2))
                try:
                    L = torch.linalg.cholesky(mat_reg)
                    _nr("chol_recover")
                    return L
                except RuntimeError:
                    continue
            _nr("chol_fail")
            return torch.linalg.cholesky(I.expand_as(mat) + eps * I)

    try:
        L_p = _cholesky_with_fallback(sigma_t_reg)

        # Trace term: tr(Sigma_t^{-1} Sigma_q)
        Y = torch.linalg.solve_triangular(L_p, sigma_q_reg, upper=False)
        Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
        trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)

        # Mahalanobis term
        delta_mu = mu_t - mu_q
        v = torch.linalg.solve_triangular(
            L_p, delta_mu.unsqueeze(-1), upper=False
        ).squeeze(-1)
        mahal_term = torch.sum(v ** 2, dim=-1)

        # Log-det terms
        logdet_p = 2.0 * torch.sum(
            torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1).clamp(min=1e-12)), dim=-1
        )
        L_q = _cholesky_with_fallback(sigma_q_reg)
        logdet_q = 2.0 * torch.sum(
            torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1).clamp(min=1e-12)), dim=-1
        )

        kl = 0.5 * (trace_term + mahal_term - K + logdet_p - logdet_q)
        kl = safe_kl_clamp(kl, kl_max)
        return kl.to(orig_dtype)

    except RuntimeError:
        # Scalar fallback — preserves autograd graph
        raise  # Let callers handle if needed; chunked path always has this try/except


def _kl_kernel_diagonal(
    mu_q: torch.Tensor,    # (..., K) query means
    sigma_q: torch.Tensor, # (..., K) query diagonal variances
    mu_t: torch.Tensor,    # (..., K) transported key means
    sigma_t: torch.Tensor, # (..., K) transported key diagonal variances
    kl_max: float,
    eps: float,
) -> torch.Tensor:
    r"""Diagonal-covariance KL divergence.

    Closed-form simplification:
        KL = 0.5 * (sum(sigma_q / sigma_t)
                    + sum((mu_t - mu_q)^2 / sigma_t)
                    - K
                    + sum(log sigma_t - log sigma_q))

    O(K) per pair (no Cholesky, no matrix inversion).

    Args:
        mu_q: (..., K) query means.
        sigma_q: (..., K) query diagonal variances (positive).
        mu_t: (..., K) transported key means.
        sigma_t: (..., K) transported key diagonal variances (positive).
        kl_max: Clamp ceiling.
        eps: Floor for sigma before division/log.

    Returns:
        kl: (...,) non-negative KL values.
    """
    K = mu_q.shape[-1]
    orig_dtype = mu_q.dtype

    # Force float32 for sigma divisions and logs to survive AMP float16.
    with torch.amp.autocast('cuda', enabled=False):
        mu_q = mu_q.float()
        sigma_q = sigma_q.float().clamp(min=eps)
        mu_t = mu_t.float()
        sigma_t = sigma_t.float().clamp(min=eps)

        trace_term = (sigma_q / sigma_t).sum(dim=-1)
        delta = mu_t - mu_q
        mahal_term = ((delta ** 2) / sigma_t).sum(dim=-1)
        logdet_term = (torch.log(sigma_t) - torch.log(sigma_q)).sum(dim=-1)

        kl = 0.5 * (trace_term + mahal_term - K + logdet_term)
        kl = safe_kl_clamp(kl, kl_max)

    return kl.to(orig_dtype)


def _kl_kernel_block_diagonal(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    block_exp_pairs: list,
    irrep_dims: List[int],
    diagonal_sigma: bool,
    kl_max: float,
    eps: float,
) -> torch.Tensor:
    r"""Block-diagonal KL divergence — delegates to fused gauge_utils kernels.

    Exploits block-diagonal structure of the gauge group representation:
        KL(q || p) = sum_b KL(q_b || p_b)

    Delegates to the fused batch kernels in ``gauge_utils.py`` which group
    same-sized blocks together and process them with a single matrix_exp call,
    reducing CUDA kernel launches from O(num_blocks) to O(num_unique_dims).

    Args:
        mu_q: (B, N, K) belief means.
        sigma_q: (B, N, K) diagonal variances or (B, N, K, K) full covariance.
        block_exp_pairs: list of ``(exp_phi, exp_neg_phi)`` per block as returned
            by ``fused_block_matrix_exp_pairs``.
        irrep_dims: Block dimension list [d_1, d_2, ...] summing to K.
        diagonal_sigma: If True, ``sigma_q`` is (B, N, K); if False, (B, N, K, K).
        kl_max: Not used directly; the fused kernels use their own internal ceiling
            derived from K.  Kept for API symmetry.
        eps: Numerical stability floor passed to the fused kernels.

    Returns:
        kl: (B, N, N) total KL divergence across all blocks.
    """
    if diagonal_sigma:
        return fused_block_diagonal_kl_diag(
            mu_q, sigma_q, block_exp_pairs, irrep_dims, eps=eps
        )
    else:
        return fused_block_diagonal_kl_full(
            mu_q, sigma_q, block_exp_pairs, irrep_dims, eps=eps
        )


# =============================================================================
# Public Entry Point
# =============================================================================

def compute_kl_matrix(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_transported: torch.Tensor,
    sigma_transported: torch.Tensor,
    mode: KLMode = KLMode.DENSE,
    chunk_size: Optional[int] = None,
    # Block-diagonal specific
    block_exp_pairs: Optional[list] = None,
    irrep_dims: Optional[List[int]] = None,
    # Shared options
    kl_max: float = 100.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Compute pairwise KL divergence matrix: KL(q_i || Omega_ij[q_j]).

    This is the single parametric entry point for all KL matrix computations.
    Callers are responsible for:
      1. Building transport operators (Omega, block_exp_pairs, or exp_phi/exp_neg_phi).
      2. Transporting key means and covariances: mu_transported, sigma_transported.
      3. Selecting the appropriate mode and passing compatible sigma shapes.

    For BLOCK_DIAGONAL mode, ``mu_transported`` and ``sigma_transported`` are
    *not* used; the kernel re-derives transport internally from ``block_exp_pairs``
    via the fused gauge_utils kernels.  This is intentional: the fused path
    avoids materialising the full (B, N, N, d, d) Omega tensor.

    Args:
        mu_q: (B, N, K) query belief means.
        sigma_q: Query covariances.
            DENSE:          (B, N, K, K) full covariance.
            DIAGONAL:       (B, N, K) diagonal variances.
            BLOCK_DIAGONAL: (B, N, K) diagonal or (B, N, K, K) full.
        mu_transported: (B, N, N, K) transported key means.
            Ignored for BLOCK_DIAGONAL mode.
        sigma_transported: Transported key covariances.
            DENSE:    (B, N, N, K, K).
            DIAGONAL: (B, N, N, K).
            Ignored for BLOCK_DIAGONAL mode.
        mode: KLMode enum selecting the computation kernel.
        chunk_size: If set, the (N, N) KL matrix is assembled in
            ``chunk_size x chunk_size`` tiles to bound peak memory.
            None = no chunking (all pairs in one vectorised pass).
            Ignored for BLOCK_DIAGONAL mode (fused kernels handle tiling
            internally with their own ``_tile_size`` parameter).
        block_exp_pairs: List of ``(exp_phi, exp_neg_phi)`` per irrep block.
            Required for BLOCK_DIAGONAL mode.
        irrep_dims: Block dimensions [d_1, ...] summing to K.
            Required for BLOCK_DIAGONAL mode.
        kl_max: Clamp ceiling for KL values.  Scaled automatically inside the
            fused block-diagonal kernels; passed through for DENSE/DIAGONAL.
        eps: Numerical stability floor.

    Returns:
        kl_matrix: (B, N, N) pairwise KL divergence matrix.

    Raises:
        ValueError: If BLOCK_DIAGONAL mode is requested without block_exp_pairs
            or irrep_dims.
    """
    if mode is KLMode.BLOCK_DIAGONAL:
        if block_exp_pairs is None or irrep_dims is None:
            raise ValueError(
                "KLMode.BLOCK_DIAGONAL requires both block_exp_pairs and irrep_dims."
            )
        diagonal_sigma = (sigma_q.dim() == 3)
        return _kl_kernel_block_diagonal(
            mu_q, sigma_q, block_exp_pairs, irrep_dims,
            diagonal_sigma=diagonal_sigma,
            kl_max=kl_max,
            eps=eps,
        )

    # DENSE and DIAGONAL modes: select kernel, optionally chunk.
    if chunk_size is None:
        return _compute_unchunked(
            mu_q, sigma_q, mu_transported, sigma_transported, mode, kl_max, eps
        )
    else:
        return _compute_chunked(
            mu_q, sigma_q, mu_transported, sigma_transported,
            mode, chunk_size, kl_max, eps
        )


# =============================================================================
# Internal: unchunked and chunked assembly
# =============================================================================

def _compute_unchunked(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_transported: torch.Tensor,   # (B, N, N, K)
    sigma_transported: torch.Tensor,
    mode: KLMode,
    kl_max: float,
    eps: float,
) -> torch.Tensor:
    """Compute full (B, N, N) KL matrix in one vectorised pass."""
    B, N, K = mu_q.shape
    # Expand query beliefs over all key positions
    mu_i = mu_q[:, :, None, :].expand(-1, -1, N, -1).clone()   # (B, N, N, K)

    if mode is KLMode.DIAGONAL:
        sigma_i = sigma_q[:, :, None, :].expand(-1, -1, N, -1).clone()
        return _kl_kernel_diagonal(mu_i, sigma_i, mu_transported, sigma_transported,
                                   kl_max=kl_max, eps=eps)
    else:
        # DENSE
        sigma_i = sigma_q[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()
        return _kl_kernel_dense(mu_i, sigma_i, mu_transported, sigma_transported,
                                kl_max=kl_max, eps=eps)


def _compute_chunked(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_transported: torch.Tensor,   # (B, N, N, K)
    sigma_transported: torch.Tensor,
    mode: KLMode,
    chunk_size: int,
    kl_max: float,
    eps: float,
) -> torch.Tensor:
    """Assemble (B, N, N) KL matrix by processing chunk_size x chunk_size tiles."""
    B, N, K = mu_q.shape
    row_chunks: list = []

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        n_i = i_end - i_start

        mu_i_chunk = mu_q[:, i_start:i_end].contiguous()

        col_chunks: list = []
        for j_start in range(0, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            n_j = j_end - j_start

            # Slice pre-transported quantities for this chunk
            mu_t_chunk = mu_transported[:, i_start:i_end, j_start:j_end].contiguous()
            sigma_t_chunk = sigma_transported[:, i_start:i_end, j_start:j_end].contiguous()

            # Expand query beliefs to match chunk shape
            if mode is KLMode.DIAGONAL:
                sigma_i_chunk = sigma_q[:, i_start:i_end].contiguous()
                mu_i_exp = mu_i_chunk[:, :, None, :].expand(-1, -1, n_j, -1).clone()
                sigma_i_exp = sigma_i_chunk[:, :, None, :].expand(-1, -1, n_j, -1).clone()
                kl_chunk = _kl_kernel_diagonal(
                    mu_i_exp, sigma_i_exp, mu_t_chunk, sigma_t_chunk,
                    kl_max=kl_max, eps=eps,
                )
            else:
                # DENSE
                sigma_i_chunk = sigma_q[:, i_start:i_end].contiguous()
                mu_i_exp = mu_i_chunk[:, :, None, :].expand(-1, -1, n_j, -1).clone()
                sigma_i_exp = sigma_i_chunk[:, :, None, :, :].expand(-1, -1, n_j, -1, -1).clone()
                kl_chunk = _kl_kernel_dense(
                    mu_i_exp, sigma_i_exp, mu_t_chunk, sigma_t_chunk,
                    kl_max=kl_max, eps=eps,
                )

            col_chunks.append(kl_chunk)

        row_chunks.append(torch.cat(col_chunks, dim=2))  # (B, n_i, N)

    return torch.cat(row_chunks, dim=1)  # (B, N, N)
