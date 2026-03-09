# -*- coding: utf-8 -*-
"""
Unified KL Divergence Operations
==================================

Replaces 8 separate _compute_kl_matrix_* functions from the legacy
attention.py with a single dispatch that handles all covariance types
and memory strategies.

Dispatch axes:
    covariance: 'full' | 'diagonal'
    structure:  'dense' | 'block_diagonal'
    memory:     chunk_size=None (full) | chunk_size=int (chunked)

The block-diagonal + chunked combinations are handled by nesting the
block loop inside the chunk loop.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from transformer_v2.gauge_utils import stable_matrix_exp_pair


# =============================================================================
# Public API
# =============================================================================

def compute_kl_matrix(
    mu_q: torch.Tensor,           # (B, N, K) belief means
    sigma_q: torch.Tensor,        # (B, N, K, K) or (B, N, K) if diagonal
    phi: torch.Tensor,            # (B, N, n_gen) gauge frames
    generators: torch.Tensor,     # (n_gen, K, K) Lie algebra generators
    diagonal_covariance: bool = False,
    irrep_dims: Optional[List[int]] = None,
    chunk_size: Optional[int] = None,
    use_identity_transport: bool = False,
    enforce_orthogonal: bool = False,
    cached_transport: Optional[dict] = None,
) -> torch.Tensor:
    """Compute pairwise KL divergence matrix: KL(q_i || Ω_ij[q_j]).

    Single entry point replacing 8 legacy functions. Dispatches based on
    covariance type, block structure, and chunk size.

    Args:
        mu_q: (B, N, K) belief means
        sigma_q: (B, N, K, K) full or (B, N, K) diagonal variances
        phi: (B, N, n_gen) gauge frames
        generators: (n_gen, K, K) Lie algebra generators
        diagonal_covariance: True if sigma_q is (B, N, K) diagonal
        irrep_dims: Block dims [d₁, d₂, ...] for block-diagonal mode
        chunk_size: Process N×N in C×C chunks for memory efficiency
        use_identity_transport: If True, Ω_ij = I (skip transport)
        enforce_orthogonal: If True, project Ω to SO(K) via Newton-Schulz
        cached_transport: Optional precomputed {'exp_phi', 'exp_neg_phi', 'Omega'}

    Returns:
        kl_matrix: (B, N, N) pairwise KL divergences
    """
    if irrep_dims is not None:
        return _kl_matrix_block_diagonal(
            mu_q, sigma_q, phi, generators, irrep_dims,
            diagonal_covariance=diagonal_covariance,
            chunk_size=chunk_size,
            enforce_orthogonal=enforce_orthogonal,
        )

    if chunk_size is not None:
        return _kl_matrix_chunked(
            mu_q, sigma_q, phi, generators, chunk_size,
            diagonal_covariance=diagonal_covariance,
            enforce_orthogonal=enforce_orthogonal,
        )

    if diagonal_covariance:
        return _kl_matrix_diagonal(
            mu_q, sigma_q, phi, generators,
            cached_transport=cached_transport,
            enforce_orthogonal=enforce_orthogonal,
        )

    return _kl_matrix_full(
        mu_q, sigma_q, phi, generators,
        cached_transport=cached_transport,
        use_identity_transport=use_identity_transport,
        enforce_orthogonal=enforce_orthogonal,
    )


# =============================================================================
# Transport Operator Computation (shared by all paths)
# =============================================================================

def compute_transport_operators(
    phi: torch.Tensor,
    generators: torch.Tensor,
    enforce_orthogonal: bool = False,
    gauge_mode: str = 'learned',
) -> dict:
    """Precompute transport operators for all token pairs.

    Ω_ij = exp(φ_i · G) · exp(-φ_j · G)

    Args:
        phi: (B, N, n_gen) gauge frames
        generators: (n_gen, K, K) Lie algebra generators
        enforce_orthogonal: Project to SO(K) via Newton-Schulz
        gauge_mode: 'learned' or 'trivial' (identity transport)

    Returns:
        dict with 'exp_phi', 'exp_neg_phi', 'Omega'
    """
    B, N, _ = phi.shape
    K = generators.shape[1]
    dtype = phi.dtype
    device = phi.device

    if gauge_mode == 'trivial':
        eye_K = torch.eye(K, device=device, dtype=dtype)
        return {
            'exp_phi': eye_K.expand(B, N, K, K).contiguous(),
            'exp_neg_phi': eye_K.expand(B, N, K, K).contiguous(),
            'Omega': eye_K.expand(B, N, N, K, K).contiguous(),
        }

    phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)

    # For skew-symmetric generators (SO(K)): exp(-A) = exp(A)^T
    _is_skew = torch.allclose(
        generators + generators.transpose(-1, -2),
        torch.zeros_like(generators), atol=1e-5
    )
    if _is_skew:
        exp_neg_phi = exp_phi.transpose(-1, -2)

    if enforce_orthogonal and K >= 16:
        eye_K = torch.eye(K, device=device, dtype=dtype)
        exp_phi = exp_phi @ ((3.0 * eye_K - exp_phi.transpose(-1, -2) @ exp_phi) / 2.0)
        exp_neg_phi = exp_neg_phi @ ((3.0 * eye_K - exp_neg_phi.transpose(-1, -2) @ exp_neg_phi) / 2.0)

    Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

    return {
        'exp_phi': exp_phi,
        'exp_neg_phi': exp_neg_phi,
        'Omega': Omega,
    }


def _compute_block_transport(
    phi: torch.Tensor,
    gen_block: torch.Tensor,
    enforce_orthogonal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute transport for a single irrep block.

    Returns (exp_phi_block, exp_neg_phi_block, Omega_block).
    """
    d = gen_block.shape[1]
    dtype = phi.dtype
    device = phi.device

    phi_matrix = torch.einsum('bna,aij->bnij', phi, gen_block)
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)

    if enforce_orthogonal and d >= 16:
        eye_d = torch.eye(d, device=device, dtype=dtype)
        exp_phi = exp_phi @ ((3.0 * eye_d - exp_phi.transpose(-1, -2) @ exp_phi) / 2.0)
        exp_neg_phi = exp_neg_phi @ ((3.0 * eye_d - exp_neg_phi.transpose(-1, -2) @ exp_neg_phi) / 2.0)

    Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)
    return exp_phi, exp_neg_phi, Omega


# =============================================================================
# KL Divergence Primitives
# =============================================================================

def _kl_full_cov(
    mu_i: torch.Tensor,         # (..., K) query means
    sigma_i: torch.Tensor,      # (..., K, K) query covariances
    mu_p: torch.Tensor,         # (..., K) transported means
    sigma_p: torch.Tensor,      # (..., K, K) transported covariances
    K: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """KL(N(μ_i, Σ_i) || N(μ_p, Σ_p)) for full covariance matrices.

    Uses Cholesky-based computation with progressive regularization fallback.
    """
    I = torch.eye(K, device=mu_i.device, dtype=torch.float32)

    mu_i = mu_i.float()
    sigma_i = sigma_i.float()
    mu_p = mu_p.float()
    sigma_p = sigma_p.float()

    sigma_i_reg = sigma_i + eps * I
    sigma_p_reg = sigma_p + eps * I

    # Symmetrize transported covariance
    sigma_p_reg = 0.5 * (sigma_p_reg + sigma_p_reg.transpose(-1, -2))

    # NaN guard
    nan_mask = torch.isnan(sigma_p_reg).any(dim=-1).any(dim=-1)
    if nan_mask.any():
        sigma_p_reg = torch.where(
            nan_mask.unsqueeze(-1).unsqueeze(-1),
            I.expand_as(sigma_p_reg),
            sigma_p_reg,
        )

    # Cholesky with progressive fallback
    L_p = _safe_cholesky(sigma_p_reg, eps, I)
    L_q = _safe_cholesky(sigma_i_reg, eps, I)

    # Trace term: tr(Σ_p⁻¹ Σ_q)
    Y = torch.linalg.solve_triangular(L_p, sigma_i_reg, upper=False)
    Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
    trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)

    # Mahalanobis term: (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q)
    delta_mu = mu_p - mu_i
    v = torch.linalg.solve_triangular(
        L_p, delta_mu.unsqueeze(-1), upper=False
    ).squeeze(-1)
    mahal_term = torch.sum(v ** 2, dim=-1)

    # Log determinant terms
    logdet_p = 2.0 * torch.sum(
        torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1).clamp(min=1e-12)), dim=-1
    )
    logdet_q = 2.0 * torch.sum(
        torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1).clamp(min=1e-12)), dim=-1
    )

    kl = 0.5 * (trace_term + mahal_term - K + logdet_p - logdet_q)
    kl_ceil = max(100.0, 5.0 * K)
    kl = torch.clamp(kl, min=0.0, max=kl_ceil)
    kl = kl.nan_to_num(nan=0.0, posinf=kl_ceil, neginf=0.0)

    return kl


def _kl_diagonal(
    mu_i: torch.Tensor,           # (..., K) query means
    sigma_i: torch.Tensor,        # (..., K) query diagonal variances
    mu_p: torch.Tensor,           # (..., K) transported means
    sigma_p: torch.Tensor,        # (..., K) transported diagonal variances
    K: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """KL(N(μ_i, diag(σ_i)) || N(μ_p, diag(σ_p))) for diagonal covariances.

    No Cholesky needed — O(K) per pair instead of O(K³).
    """
    sigma_i = sigma_i.float().clamp(min=eps)
    sigma_p = sigma_p.float().clamp(min=eps)
    mu_i = mu_i.float()
    mu_p = mu_p.float()

    trace_term = (sigma_i / sigma_p).sum(dim=-1)
    delta_mu = mu_p - mu_i
    mahal_term = (delta_mu ** 2 / sigma_p).sum(dim=-1)
    logdet_term = (torch.log(sigma_p) - torch.log(sigma_i)).sum(dim=-1)

    kl = 0.5 * (trace_term + mahal_term - K + logdet_term)
    kl_ceil = max(100.0, 5.0 * K)
    return torch.clamp(kl, min=0.0, max=kl_ceil)


def _safe_cholesky(
    M: torch.Tensor,
    eps: float,
    I: torch.Tensor,
) -> torch.Tensor:
    """Cholesky decomposition with progressive regularization fallback."""
    try:
        return torch.linalg.cholesky(M)
    except RuntimeError:
        reg = eps
        for _ in range(5):
            reg *= 10.0
            M_reg = M + reg * I
            M_reg = 0.5 * (M_reg + M_reg.transpose(-1, -2))
            try:
                return torch.linalg.cholesky(M_reg)
            except RuntimeError:
                continue
        return torch.linalg.cholesky(I.expand_as(M) + eps * I)


# =============================================================================
# Full Matrix KL (standard path)
# =============================================================================

def _kl_matrix_full(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    phi: torch.Tensor,
    generators: torch.Tensor,
    cached_transport: Optional[dict] = None,
    use_identity_transport: bool = False,
    enforce_orthogonal: bool = False,
) -> torch.Tensor:
    """Full-covariance KL matrix computation. O(N²K²) memory."""
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Get transport operators
    if use_identity_transport:
        mu_transported = mu_q[:, None, :, :].expand(-1, N, -1, -1).clone()
        Sigma_transported = sigma_q[:, None, :, :, :].expand(-1, N, -1, -1, -1).clone()
    else:
        if cached_transport is not None and 'Omega' in cached_transport:
            Omega = cached_transport['Omega']
        else:
            transport = compute_transport_operators(phi, generators, enforce_orthogonal)
            Omega = transport['Omega']

        mu_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)
        Sigma_transported = torch.einsum(
            'bijkl,bjlm,bijmn->bijkn',
            Omega, sigma_q, Omega.transpose(-1, -2)
        )
        Sigma_transported = 0.5 * (Sigma_transported + Sigma_transported.transpose(-1, -2))

    # Expand query beliefs for pairwise comparison
    mu_i = mu_q[:, :, None, :].expand(-1, -1, N, -1).clone()
    Sigma_i = sigma_q[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()

    kl = _kl_full_cov(mu_i, Sigma_i, mu_transported, Sigma_transported, K)
    return kl.to(dtype)


# =============================================================================
# Diagonal KL (O(N²K) memory)
# =============================================================================

def _kl_matrix_diagonal(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    phi: torch.Tensor,
    generators: torch.Tensor,
    cached_transport: Optional[dict] = None,
    enforce_orthogonal: bool = False,
) -> torch.Tensor:
    """Diagonal-covariance KL. O(N²K) instead of O(N²K²)."""
    # Squeeze trailing singleton dimensions for robustness
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)

    B, N, K = mu_q.shape
    dtype = mu_q.dtype
    eps = 1e-6
    sigma_q = sigma_q.clamp(min=eps)

    # Get transport operators
    if cached_transport is not None and 'Omega' in cached_transport:
        Omega = cached_transport['Omega']
    else:
        transport = compute_transport_operators(phi, generators, enforce_orthogonal)
        Omega = transport['Omega']

    # Transport means
    mu_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)

    # Diagonal of transported covariance: diag(Ω @ diag(σ_j) @ Ω^T)_k = Σ_l Ω_kl² * σ_j[l]
    sigma_j_orig = sigma_q[:, None, :, :].expand(-1, N, -1, -1)
    sigma_j_transported = torch.einsum(
        'bijkl,bijkl,bijl->bijk', Omega, Omega, sigma_j_orig
    ).clamp(min=eps)

    # Expand query beliefs
    sigma_i = sigma_q[:, :, None, :].expand(-1, -1, N, -1)
    mu_i = mu_q[:, :, None, :].expand(-1, -1, N, -1)

    kl = _kl_diagonal(mu_i, sigma_i, mu_transported, sigma_j_transported, K)
    return kl.to(dtype)


# =============================================================================
# Block-Diagonal KL (exploits irrep structure)
# =============================================================================

def _kl_matrix_block_diagonal(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    phi: torch.Tensor,
    generators: torch.Tensor,
    irrep_dims: List[int],
    diagonal_covariance: bool = False,
    chunk_size: Optional[int] = None,
    enforce_orthogonal: bool = False,
) -> torch.Tensor:
    """Block-diagonal KL: process each irrep block separately.

    Since generators and covariances are block-diagonal:
    - Ω_ij is block-diagonal
    - KL = Σ_blocks KL_block (additive decomposition)

    Memory: O(N² × max(dᵢ²)) instead of O(N² × K²)
    """
    # Squeeze trailing singletons for diagonal mode
    if diagonal_covariance:
        while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
            sigma_q = sigma_q.squeeze(-1)

    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    eps = 1e-6

    assert sum(irrep_dims) == K, f"irrep_dims sum {sum(irrep_dims)} != K={K}"

    if diagonal_covariance:
        sigma_q = sigma_q.clamp(min=eps)

    kl_total = torch.zeros(B, N, N, device=device, dtype=dtype)

    block_start = 0
    for d in irrep_dims:
        block_end = block_start + d

        mu_block = mu_q[:, :, block_start:block_end].contiguous()
        gen_block = generators[:, block_start:block_end, block_start:block_end].contiguous()

        if chunk_size is not None:
            kl_block = _kl_block_chunked(
                mu_block, sigma_q, phi, gen_block, d, K,
                block_start, block_end,
                diagonal_covariance, chunk_size, enforce_orthogonal,
            )
        else:
            # Compute block transport
            _, _, Omega_block = _compute_block_transport(phi, gen_block, enforce_orthogonal)

            # Transport means
            mu_transported = torch.einsum('bijkl,bjl->bijk', Omega_block, mu_block)

            if diagonal_covariance:
                sigma_block = sigma_q[:, :, block_start:block_end].contiguous()
                sigma_j_transported = torch.einsum(
                    'bijkl,bijkl,bjl->bijk', Omega_block, Omega_block, sigma_block
                ).clamp(min=eps)

                sigma_i = sigma_block[:, :, None, :].expand(-1, -1, N, -1)
                mu_i = mu_block[:, :, None, :].expand(-1, -1, N, -1)

                kl_block = _kl_diagonal(mu_i, sigma_i, mu_transported, sigma_j_transported, d)
            else:
                sigma_block = sigma_q[:, :, block_start:block_end, block_start:block_end].contiguous()
                sigma_transported = torch.einsum(
                    'bijkl,bjlm,bijmn->bijkn',
                    Omega_block, sigma_block, Omega_block.transpose(-1, -2)
                )

                mu_i = mu_block[:, :, None, :].expand(-1, -1, N, -1).clone()
                sigma_i = sigma_block[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()

                kl_block = _kl_full_cov(mu_i, sigma_i, mu_transported, sigma_transported, d)

            del Omega_block

        kl_block = torch.clamp(kl_block, min=0.0, max=max(100.0, 5.0 * K))
        kl_total = kl_total + kl_block
        block_start = block_end

    return kl_total


# =============================================================================
# Chunked KL (trades compute for memory)
# =============================================================================

def _kl_matrix_chunked(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    phi: torch.Tensor,
    generators: torch.Tensor,
    chunk_size: int,
    diagonal_covariance: bool = False,
    enforce_orthogonal: bool = False,
) -> torch.Tensor:
    """Chunked KL: O(C²K²) memory instead of O(N²K²)."""
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    eps = 1e-6

    # Precompute per-token matrix exponentials: O(N×K²) memory
    phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)
    del phi_matrix

    if enforce_orthogonal and K >= 16:
        eye_K = torch.eye(K, device=device, dtype=dtype)
        exp_phi = exp_phi @ ((3.0 * eye_K - exp_phi.transpose(-1, -2) @ exp_phi) / 2.0)
        exp_neg_phi = exp_neg_phi @ ((3.0 * eye_K - exp_neg_phi.transpose(-1, -2) @ exp_neg_phi) / 2.0)

    if diagonal_covariance:
        while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
            sigma_q = sigma_q.squeeze(-1)
        sigma_q = sigma_q.clamp(min=eps)

    # Precompute log-determinants for query covariances
    if not diagonal_covariance:
        I = torch.eye(K, device=device, dtype=torch.float32)
        sigma_q_f32 = sigma_q.float()
        L_q_all = _safe_cholesky(sigma_q_f32 + eps * I, eps, I)
        logdet_q_all = 2.0 * torch.sum(
            torch.log(torch.diagonal(L_q_all, dim1=-2, dim2=-1) + eps), dim=-1
        )

    # Process in chunks
    row_chunks = []
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)

        exp_phi_i = exp_phi[:, i_start:i_end].contiguous()
        mu_i = mu_q[:, i_start:i_end].contiguous()

        col_chunks = []
        for j_start in range(0, N, chunk_size):
            j_end = min(j_start + chunk_size, N)

            exp_neg_phi_j = exp_neg_phi[:, j_start:j_end].contiguous()
            mu_j = mu_q[:, j_start:j_end].contiguous()

            # Chunk transport: (B, n_i, n_j, K, K)
            Omega_chunk = torch.einsum('bikl,bjlm->bijkm', exp_phi_i, exp_neg_phi_j)

            # Transport means
            mu_transported = torch.einsum('bijkl,bjl->bijk', Omega_chunk, mu_j)
            n_i = i_end - i_start
            n_j = j_end - j_start

            if diagonal_covariance:
                sigma_j = sigma_q[:, j_start:j_end].contiguous()
                sigma_j_exp = sigma_j[:, None, :, :].expand(-1, n_i, -1, -1)
                sigma_j_transported = torch.einsum(
                    'bijkl,bijkl,bijl->bijk', Omega_chunk, Omega_chunk, sigma_j_exp
                ).clamp(min=eps)

                sigma_i_exp = sigma_q[:, i_start:i_end, None, :].expand(-1, -1, n_j, -1)
                mu_i_exp = mu_i[:, :, None, :].expand(-1, -1, n_j, -1)

                kl_chunk = _kl_diagonal(mu_i_exp, sigma_i_exp, mu_transported, sigma_j_transported, K)
            else:
                sigma_j = sigma_q[:, j_start:j_end].contiguous()
                sigma_transported = torch.einsum(
                    'bijkl,bjlm,bijmn->bijkn',
                    Omega_chunk, sigma_j, Omega_chunk.transpose(-1, -2)
                )
                sigma_transported = 0.5 * (sigma_transported + sigma_transported.transpose(-1, -2))

                sigma_i = sigma_q[:, i_start:i_end].contiguous()
                sigma_i_exp = sigma_i[:, :, None, :, :].expand(-1, -1, n_j, -1, -1).clone()
                mu_i_exp = mu_i[:, :, None, :].expand(-1, -1, n_j, -1).clone()

                kl_chunk = _kl_full_cov(mu_i_exp, sigma_i_exp, mu_transported, sigma_transported, K)

            kl_ceil = max(100.0, 5.0 * K)
            kl_chunk = torch.clamp(kl_chunk, min=0.0, max=kl_ceil)
            kl_chunk = kl_chunk.nan_to_num(nan=0.0, posinf=kl_ceil, neginf=0.0)
            col_chunks.append(kl_chunk)

            del Omega_chunk

        row_chunks.append(torch.cat(col_chunks, dim=2))

    return torch.cat(row_chunks, dim=1).to(dtype)


def _kl_block_chunked(
    mu_block, sigma_q, phi, gen_block, d, K,
    block_start, block_end,
    diagonal_covariance, chunk_size, enforce_orthogonal,
):
    """Chunked processing for a single irrep block."""
    B, N, _ = mu_block.shape
    device = mu_block.device
    dtype = mu_block.dtype
    eps = 1e-6

    phi_matrix = torch.einsum('bna,aij->bnij', phi, gen_block)
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)
    del phi_matrix

    if enforce_orthogonal and d >= 16:
        eye_d = torch.eye(d, device=device, dtype=dtype)
        exp_phi = exp_phi @ ((3.0 * eye_d - exp_phi.transpose(-1, -2) @ exp_phi) / 2.0)
        exp_neg_phi = exp_neg_phi @ ((3.0 * eye_d - exp_neg_phi.transpose(-1, -2) @ exp_neg_phi) / 2.0)

    row_chunks = []
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        n_i = i_end - i_start

        exp_phi_i = exp_phi[:, i_start:i_end].contiguous()
        mu_i = mu_block[:, i_start:i_end].contiguous()

        col_chunks = []
        for j_start in range(0, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            n_j = j_end - j_start

            exp_neg_phi_j = exp_neg_phi[:, j_start:j_end].contiguous()
            mu_j = mu_block[:, j_start:j_end].contiguous()

            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi_i, exp_neg_phi_j)
            mu_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_j)

            if diagonal_covariance:
                sigma_j = sigma_q[:, j_start:j_end, block_start:block_end].contiguous()
                sigma_j_exp = sigma_j[:, None, :, :].expand(-1, n_i, -1, -1)
                sigma_j_t = torch.einsum(
                    'bijkl,bijkl,bijl->bijk', Omega, Omega, sigma_j_exp
                ).clamp(min=eps)

                sigma_i = sigma_q[:, i_start:i_end, block_start:block_end].contiguous()
                sigma_i_exp = sigma_i[:, :, None, :].expand(-1, -1, n_j, -1)
                mu_i_exp = mu_i[:, :, None, :].expand(-1, -1, n_j, -1)

                kl_chunk = _kl_diagonal(mu_i_exp, sigma_i_exp, mu_transported, sigma_j_t, d)
            else:
                sigma_j = sigma_q[:, j_start:j_end, block_start:block_end, block_start:block_end].contiguous()
                sigma_t = torch.einsum(
                    'bijkl,bjlm,bijmn->bijkn',
                    Omega, sigma_j, Omega.transpose(-1, -2)
                )

                sigma_i = sigma_q[:, i_start:i_end, block_start:block_end, block_start:block_end].contiguous()
                sigma_i_exp = sigma_i[:, :, None, :, :].expand(-1, -1, n_j, -1, -1).clone()
                mu_i_exp = mu_i[:, :, None, :].expand(-1, -1, n_j, -1).clone()

                kl_chunk = _kl_full_cov(mu_i_exp, sigma_i_exp, mu_transported, sigma_t, d)

            col_chunks.append(kl_chunk)
            del Omega

        row_chunks.append(torch.cat(col_chunks, dim=2))

    return torch.cat(row_chunks, dim=1).to(dtype)
