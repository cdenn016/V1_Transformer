"""
Shared Gauge-Geometry Utilities
================================

Shared utilities for gauge transport computations used across
attention.py, variational_ffn.py, and embeddings.py.

Consolidates duplicated matrix exponential and KL divergence patterns.
"""

import torch
from collections import defaultdict
from typing import List, Optional, Tuple



def stable_matrix_exp_pair(
    matrix: torch.Tensor,
    dim_threshold: int = 20,
    max_norm: float = 10.0,
    skew_symmetric: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute exp(M) and exp(-M) with norm clamping and float64 upcasting.

    Two stability measures:
    1. Frobenius norm clamping: caps ‖M‖_F at max_norm to prevent the
       Padé scaling-squaring algorithm from overflowing. For GL⁺(K),
       exp(M) with ‖M‖ >> 1 produces extreme condition numbers that
       make downstream Ω Σ Ω^T numerically non-positive-definite.
    2. Float64 upcasting for K >= dim_threshold (default 20, raised from 8
       since float32 + norm clamping is sufficient for typical head dims ≤16).

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
        For SO(K), exp: so(K) -> SO(K) is surjective -- no issues.

    Args:
        matrix: (..., d, d) matrix to exponentiate.
        dim_threshold: Upcast to float64 when d >= this value. Default 8.
        max_norm: Maximum Frobenius norm for input matrix. Default 10.0.
        skew_symmetric: If True, skip computing exp(-M) and use exp(M).mT
            instead. For skew-symmetric M, exp(-M) = exp(M)^T exactly.
            Cache this flag at init rather than checking every forward pass.

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
            matrix_up = matrix.double().contiguous()
        else:
            matrix_up = matrix.float().contiguous()
        exp_pos = torch.linalg.matrix_exp(matrix_up).to(orig_dtype)
        if skew_symmetric:
            # For skew-symmetric M: exp(-M) = exp(M)^T (free transpose)
            exp_neg = exp_pos.transpose(-1, -2)
        else:
            exp_neg = torch.linalg.matrix_exp(-matrix_up).to(orig_dtype)
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


# =============================================================================
# Fused Block-Diagonal Kernels
# =============================================================================
# These functions process ALL irrep blocks in grouped batches instead of
# launching separate matrix_exp + KL kernels per block.  For typical configs
# (e.g. 75xℓ₀ + 30xℓ₁ + 18xℓ₂ = 123 blocks, 3 unique dims), this reduces
# CUDA kernel launches from O(num_blocks) to O(num_unique_dims).


def fused_block_matrix_exp_pairs(
    phi: torch.Tensor,
    generators: torch.Tensor,
    irrep_dims: List[int],
    enforce_orthogonal: bool = False,
    skew_symmetric: bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Compute matrix exponential pairs for all irrep blocks in fused batches.

    Groups blocks by dimension and computes all blocks of each size via a
    single ``stable_matrix_exp_pair`` call.  Reduces kernel launches from
    O(num_blocks) to O(num_unique_dims).

    For K=255 with [1]*75 + [3]*30 + [5]*18 = 123 blocks:
      Old: 123 stable_matrix_exp_pair calls (246 CUDA kernels)
      New:   3 stable_matrix_exp_pair calls (  6 CUDA kernels)

    Args:
        phi: (B, N, n_gen) gauge field coefficients.
        generators: (n_gen, K, K) block-diagonal Lie algebra generators.
        irrep_dims: list of block dimensions [d₁, d₂, ...].
        enforce_orthogonal: if True, apply Newton-Schulz for blocks with d >= 16.

    Returns:
        List of (exp_phi_block, exp_neg_phi_block) tuples, one per block in
        the same order as *irrep_dims*.  Each tensor has shape (B, N, d, d).
    """
    B, N, _ = phi.shape

    # Build (index, start, end, dim) for each block
    block_info: List[Tuple[int, int, int, int]] = []
    start = 0
    for idx, d in enumerate(irrep_dims):
        block_info.append((idx, start, start + d, d))
        start += d

    # Group by dimension
    dim_groups: dict = defaultdict(list)
    for idx, s, e, d in block_info:
        dim_groups[d].append((idx, s, e))

    results: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(irrep_dims)

    for d, group in dim_groups.items():
        n_blocks = len(group)

        # Stack generator sub-blocks: (n_blocks, n_gen, d, d)
        gen_stack = torch.stack(
            [generators[:, s:e, s:e] for _, s, e in group], dim=0
        )

        # Batched Lie-algebra element: phi · G per block
        # phi: (B, N, n_gen), gen_stack: (n_blocks, n_gen, d, d)
        #  -> (n_blocks, B, N, d, d)
        phi_matrices = torch.einsum('bna,gaij->gbnij', phi, gen_stack)

        # Merge block-batch and batch dims for a single matrix_exp call
        phi_flat = phi_matrices.reshape(n_blocks * B, N, d, d)
        exp_phi_flat, exp_neg_phi_flat = stable_matrix_exp_pair(
            phi_flat, skew_symmetric=skew_symmetric
        )

        # Reshape back: (n_blocks, B, N, d, d)
        exp_phi_all = exp_phi_flat.reshape(n_blocks, B, N, d, d)
        exp_neg_phi_all = exp_neg_phi_flat.reshape(n_blocks, B, N, d, d)

        if enforce_orthogonal and d >= 16:
            shape = exp_phi_all.shape
            exp_phi_all = newton_schulz_orthogonalize(
                exp_phi_all.reshape(-1, d, d)
            ).reshape(shape)
            exp_neg_phi_all = newton_schulz_orthogonalize(
                exp_neg_phi_all.reshape(-1, d, d)
            ).reshape(shape)

        # Scatter results back to per-block list
        for local_idx, (global_idx, _, _) in enumerate(group):
            results[global_idx] = (
                exp_phi_all[local_idx].contiguous(),
                exp_neg_phi_all[local_idx].contiguous(),
            )

    return results  # type: ignore[return-value]


def fused_block_diagonal_kl_diag(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    block_exp_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    irrep_dims: List[int],
    eps: float = 1e-6,
    _tile_size: int = 16,
) -> torch.Tensor:
    """Fused block-diagonal KL divergence for diagonal covariance mode.

    Memory-efficient implementation with three dispatch paths:

    1. **Triton kernels** (d=1,3,5 on CUDA): zero intermediate memory -- the
       entire matrix_exp product, transport, and KL computation stays in
       GPU registers.  Eliminates ~400-600 MB of (n_blocks, B, N, N, d, d)
       Omega intermediates.

    2. **Scalar fast path** (d=1 PyTorch fallback): pure element-wise ops
       with no matrix operations.  For 75 scalar blocks this avoids all
       einsum overhead.

    3. **Row-tiled path** (d>1 PyTorch fallback): processes ``_tile_size``
       query rows at a time, reducing peak Omega memory by a factor of
       ``N / _tile_size``.  For N=64, _tile_size=16 gives 4x memory
       reduction.

    Args:
        mu_q: (B, N, K) belief means.
        sigma_q: (B, N, K) diagonal variances (already clamped to >= eps).
        block_exp_pairs: list of (exp_phi, exp_neg_phi) per block.
        irrep_dims: block dimensions [d₁, d₂, ...].
        eps: numerical stability floor.
        _tile_size: number of query rows per tile for d>1 blocks.

    Returns:
        kl_total: (B, N, N) total KL divergence across all blocks.
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    kl_max = max(100.0, 20.0 * K)

    kl_total = torch.zeros(B, N, N, device=device, dtype=dtype)

    # Build block ranges and group by dimension
    block_ranges: List[Tuple[int, int, int, int]] = []
    start = 0
    for idx, d in enumerate(irrep_dims):
        block_ranges.append((idx, start, start + d, d))
        start += d

    dim_groups: dict = defaultdict(list)
    for idx, s, e, d in block_ranges:
        dim_groups[d].append((idx, s, e))

    # ── PyTorch path (tiled for memory efficiency) ──────────────────────
    for d, group in dim_groups.items():
        n_blocks = len(group)

        # Stack beliefs: (n_blocks, B, N, d)
        mu_stack = torch.stack([mu_q[:, :, s:e] for _, s, e in group], dim=0)
        sigma_stack = torch.stack([sigma_q[:, :, s:e] for _, s, e in group], dim=0)

        # Stack exp pairs: (n_blocks, B, N, d, d)
        exp_phi_stack = torch.stack(
            [block_exp_pairs[idx][0] for idx, _, _ in group], dim=0)
        exp_neg_phi_stack = torch.stack(
            [block_exp_pairs[idx][1] for idx, _, _ in group], dim=0)

        if d == 1:
            # ── Scalar fast path: element-wise, no matrix ops ───────────
            # AMP guard: sigma division and log must stay float32
            with torch.amp.autocast('cuda', enabled=False):
                _f32 = torch.float32
                # Squeeze out trivial 1x1 matrix dims -> (n_blocks, B, N)
                _ep = exp_phi_stack if exp_phi_stack.dtype == _f32 else exp_phi_stack.float()
                _en = exp_neg_phi_stack if exp_neg_phi_stack.dtype == _f32 else exp_neg_phi_stack.float()
                ep = _ep.squeeze(-1).squeeze(-1)
                en = _en.squeeze(-1).squeeze(-1)
                mu_s = (mu_stack if mu_stack.dtype == _f32 else mu_stack.float()).squeeze(-1)
                sig_s = (sigma_stack if sigma_stack.dtype == _f32 else sigma_stack.float()).squeeze(-1)

                # Omega_ij = exp_phi_i * exp_neg_phi_j  (scalar product)
                omega = ep[:, :, :, None] * en[:, :, None, :]  # (g, B, N, N)

                # Transport mean and variance
                mu_t = omega * mu_s[:, :, None, :]
                sig_t = (omega * omega * sig_s[:, :, None, :]).clamp(min=eps)

                mu_i = mu_s[:, :, :, None]
                sig_i = sig_s[:, :, :, None].clamp(min=eps)
                delta = mu_i - mu_t

                kl = 0.5 * (sig_i / sig_t + delta * delta / sig_t - 1.0
                            + torch.log(sig_t) - torch.log(sig_i))
                kl = kl.clamp(min=0.0, max=kl_max)
                kl = kl.nan_to_num(nan=kl_max, posinf=kl_max, neginf=0.0)

            kl_total = kl_total + kl.sum(dim=0)  # sum over blocks -> (B,N,N)

        else:
            # ── Row-tiled path: peak memory reduced by N/_tile_size ─────
            # AMP guard: sigma transport, division, and log must stay float32
            with torch.amp.autocast('cuda', enabled=False):
                _f32 = torch.float32
                _mu_f32 = mu_stack if mu_stack.dtype == _f32 else mu_stack.float()
                _sig_f32 = sigma_stack if sigma_stack.dtype == _f32 else sigma_stack.float()
                _ep_f32 = exp_phi_stack if exp_phi_stack.dtype == _f32 else exp_phi_stack.float()
                _en_f32 = exp_neg_phi_stack if exp_neg_phi_stack.dtype == _f32 else exp_neg_phi_stack.float()
                for i_start in range(0, N, _tile_size):
                    i_end = min(i_start + _tile_size, N)

                    # Omega tile: (n_blocks, B, tile, N, d, d)
                    ep_tile = _ep_f32[:, :, i_start:i_end]
                    Omega_tile = torch.einsum(
                        'gbikl,gbjlm->gbijkm', ep_tile, _en_f32)

                    # Transport mean and diagonal variance
                    mu_t = torch.einsum(
                        'gbijkl,gbjl->gbijk', Omega_tile, _mu_f32)
                    sig_t = torch.einsum(
                        'gbijkl,gbijkl,gbjl->gbijk',
                        Omega_tile, Omega_tile, _sig_f32
                    ).clamp(min=eps)
                    del Omega_tile

                    # KL for this tile of query rows
                    mu_i = _mu_f32[:, :, i_start:i_end, None, :].expand(
                        -1, -1, -1, N, -1)
                    sig_i = _sig_f32[:, :, i_start:i_end, None, :].expand(
                        -1, -1, -1, N, -1)
                    delta = mu_t - mu_i

                    trace = (sig_i / sig_t).sum(dim=-1)
                    mahal = (delta * delta / sig_t).sum(dim=-1)
                    logdet = (torch.log(sig_t) - torch.log(sig_i)).sum(dim=-1)

                    kl_tile = 0.5 * (trace + mahal - d + logdet)
                    kl_tile = kl_tile.clamp(min=0.0, max=kl_max)
                    kl_tile = kl_tile.nan_to_num(
                        nan=kl_max, posinf=kl_max, neginf=0.0)

                    # Sum over blocks and accumulate into output rows
                    kl_total[:, i_start:i_end, :] = (
                        kl_total[:, i_start:i_end, :] + kl_tile.sum(dim=0))

    return kl_total


def fused_block_diagonal_kl_full(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    block_exp_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    irrep_dims: List[int],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused block-diagonal KL divergence for full covariance mode.

    Groups same-sized irrep blocks and computes transport + KL in batched
    passes.  Falls back to diagonal approximation on Cholesky failure.

    Args:
        mu_q: (B, N, K) belief means.
        sigma_q: (B, N, K, K) block-diagonal covariances.
        block_exp_pairs: list of (exp_phi, exp_neg_phi) per block.
        irrep_dims: block dimensions [d₁, d₂, ...].
        eps: numerical stability floor.

    Returns:
        kl_total: (B, N, N) total KL divergence across all blocks.
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    kl_max = max(100.0, 20.0 * K)

    kl_total = torch.zeros(B, N, N, device=device, dtype=dtype)

    # Build block ranges and group by dimension
    block_ranges: List[Tuple[int, int, int, int]] = []
    start = 0
    for idx, d in enumerate(irrep_dims):
        block_ranges.append((idx, start, start + d, d))
        start += d

    dim_groups: dict = defaultdict(list)
    for idx, s, e, d in block_ranges:
        dim_groups[d].append((idx, s, e))

    for d, group in dim_groups.items():
        n_blocks = len(group)

        # Stack beliefs: (n_blocks, B, N, d) and (n_blocks, B, N, d, d)
        mu_stack = torch.stack([mu_q[:, :, s:e] for _, s, e in group], dim=0)
        sigma_stack = torch.stack(
            [sigma_q[:, :, s:e, s:e] for _, s, e in group], dim=0
        )

        # Stack exp pairs: (n_blocks, B, N, d, d)
        exp_phi_stack = torch.stack([block_exp_pairs[idx][0] for idx, _, _ in group], dim=0)
        exp_neg_phi_stack = torch.stack([block_exp_pairs[idx][1] for idx, _, _ in group], dim=0)

        # AMP guard: sandwich product, Cholesky, solve, log-det must stay float32
        with torch.amp.autocast('cuda', enabled=False):
            _f32 = torch.float32
            _mu_f32 = mu_stack if mu_stack.dtype == _f32 else mu_stack.float()
            _sig_f32 = sigma_stack if sigma_stack.dtype == _f32 else sigma_stack.float()
            _ep_f32 = exp_phi_stack if exp_phi_stack.dtype == _f32 else exp_phi_stack.float()
            _en_f32 = exp_neg_phi_stack if exp_neg_phi_stack.dtype == _f32 else exp_neg_phi_stack.float()

            # Batched Omega: (n_blocks, B, N, N, d, d)
            Omega = torch.einsum('gbikl,gbjlm->gbijkm', _ep_f32, _en_f32)
            del exp_phi_stack, exp_neg_phi_stack

            # Batched transport
            mu_transported = torch.einsum('gbijkl,gbjl->gbijk', Omega, _mu_f32)
            sigma_transported = torch.einsum(
                'gbijkl,gbjlm,gbijmn->gbijkn',
                Omega, _sig_f32, Omega.transpose(-1, -2)
            )
            del Omega

            I_d = torch.eye(d, device=device, dtype=torch.float32)
            mu_i = _mu_f32[:, :, :, None, :].expand(-1, -1, -1, N, -1)
            sigma_i = _sig_f32[:, :, :, None, :, :].expand(-1, -1, -1, N, -1, -1)

            sigma_i_reg = sigma_i + eps * I_d
            sigma_t_reg = sigma_transported + eps * I_d

            try:
                L_p = torch.linalg.cholesky(sigma_t_reg)
                L_q = torch.linalg.cholesky(sigma_i_reg)

                Y = torch.linalg.solve_triangular(L_p, sigma_i_reg, upper=False)
                Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
                trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)

                delta_mu = mu_transported - mu_i
                v = torch.linalg.solve_triangular(
                    L_p, delta_mu.unsqueeze(-1), upper=False
                ).squeeze(-1)
                mahal_term = torch.sum(v ** 2, dim=-1)

                logdet_p = 2.0 * torch.sum(
                    torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1) + eps), dim=-1
                )
                logdet_q = 2.0 * torch.sum(
                    torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1) + eps), dim=-1
                )

                kl_group = 0.5 * (trace_term + mahal_term - d + logdet_p - logdet_q)
            except RuntimeError:
                # Cholesky failed -- fall back to diagonal approximation
                sigma_diag_t = torch.diagonal(sigma_t_reg, dim1=-2, dim2=-1).clamp(min=eps)
                sigma_diag_i = torch.diagonal(sigma_i_reg, dim1=-2, dim2=-1).clamp(min=eps)
                delta_mu = mu_transported - mu_i

                trace_term = (sigma_diag_i / sigma_diag_t).sum(dim=-1)
                mahal_term = ((delta_mu ** 2) / sigma_diag_t).sum(dim=-1)
                logdet_term = (torch.log(sigma_diag_t) - torch.log(sigma_diag_i)).sum(dim=-1)

                kl_group = 0.5 * (trace_term + mahal_term - d + logdet_term)

            kl_group = kl_group.clamp(min=0.0, max=kl_max)
            kl_group = kl_group.nan_to_num(nan=kl_max, posinf=kl_max, neginf=0.0)

        kl_total = kl_total + kl_group.sum(dim=0)

        del mu_transported, sigma_transported

    return kl_total


def _compute_dkl_domega_diag(
    mu_i: torch.Tensor,     # (..., d) query means
    sig_i: torch.Tensor,    # (..., d) query variances
    mu_t: torch.Tensor,     # (..., d) transported means
    sig_t: torch.Tensor,    # (..., d) transported diagonal variances
    mu_j: torch.Tensor,     # (..., d) key means (for the outer product term)
    sig_j: torch.Tensor,    # (..., d) key variances
    Omega: torch.Tensor,    # (..., d, d) transport operator
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute ∂KL/∂Ω for diagonal covariance KL.

    Returns: (..., d, d) gradient matrix.

    Formula (sympy-verified):
        ∂KL/∂Ω[r,s] = c_r · Ω[r,s] · σ_j[s] / σ_t[r] - δ_r · μ_j[s] / σ_t[r]
        where c_r = 1 - σ_i[r]/σ_t[r] - δ_r^2/σ_t[r], δ_r = μ_i[r] - μ_t[r]
    """
    sig_t = sig_t.clamp(min=eps)
    delta = mu_i - mu_t                                         # (..., d)
    c = 1.0 - sig_i / sig_t - delta ** 2 / sig_t               # (..., d)

    # Term 1: diag(c/σ_t) @ Ω @ diag(σ_j)
    c_over_st = (c / sig_t).unsqueeze(-1)                       # (..., d, 1)
    sig_j_diag = sig_j.unsqueeze(-2)                            # (..., 1, d)
    term1 = c_over_st * Omega * sig_j_diag                     # (..., d, d)

    # Term 2: -outer(δ/σ_t, μ_j)
    delta_over_st = (delta / sig_t).unsqueeze(-1)               # (..., d, 1)
    mu_j_row = mu_j.unsqueeze(-2)                               # (..., 1, d)
    term2 = -delta_over_st * mu_j_row                           # (..., d, d)

    return term1 + term2
