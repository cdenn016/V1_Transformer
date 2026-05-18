"""
Shared Gauge-Geometry Utilities
================================

Shared utilities for gauge transport computations used across
attention.py, variational_ffn.py, and embeddings.py.

Consolidates duplicated matrix exponential and KL divergence patterns.
"""

import torch
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from math_utils.numerical_monitor import record as _nr
from transformer.core.vfe_utils import KL_CEIL_BASE, KL_CEIL_SCALE, spd_eigfloor

# KL divergence ceiling: kl_max = max(KL_CEIL_BASE, KL_CEIL_SCALE * dim).
# Single source of truth is vfe_utils.py.
KL_CEIL_MULT = KL_CEIL_SCALE  # alias for backward compat within this module

# Bounded LRU cache for identity matrices to avoid re-allocating I_K every call
# in Newton-Schulz orthogonalization. Keys are (K, device, dtype). Cap at 16
# entries — well above the typical (K, device, dtype) cardinality of a single
# training run (usually 1-3).
_EYE_CACHE: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
_EYE_CACHE_MAX = 16


def _cached_eye(K: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a cached ``torch.eye(K, device=device, dtype=dtype)`` tensor.

    The returned tensor is a SHARED view; callers must not mutate it in place.
    """
    key = (K, device, dtype)
    eye = _EYE_CACHE.get(key)
    if eye is None:
        if len(_EYE_CACHE) >= _EYE_CACHE_MAX:
            _EYE_CACHE.pop(next(iter(_EYE_CACHE)))
        eye = torch.eye(K, device=device, dtype=dtype)
        _EYE_CACHE[key] = eye
    return eye



def stable_matrix_exp_pair(
    matrix: torch.Tensor,
    dim_threshold: int = 20,
    max_norm: float = 10.0,
    skew_symmetric: bool = False,
    only_forward: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute exp(M) and optionally exp(-M) with norm clamping and float64 upcasting.

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
        For SO(K), exp: so(K) → SO(K) is surjective — no issues.

    Args:
        matrix: (..., d, d) matrix to exponentiate.
        dim_threshold: Upcast to float64 when d >= this value. Default 8.
        max_norm: Maximum Frobenius norm for input matrix. Default 20.0.
        skew_symmetric: If True, skip computing exp(-M) and use exp(M).mT
            instead. For skew-symmetric M, exp(-M) = exp(M)^T exactly.
            Cache this flag at init rather than checking every forward pass.
        only_forward: If True, only compute exp(M) and return None for exp(-M).
            Use when only the forward exponential is needed (e.g., PriorBank
            decode where μ_v = A_v @ μ_0 doesn't require A_v⁻¹). Saves one
            full matrix_exp call for GL(K) where exp(-M) ≠ exp(M)^T.

    Returns:
        (exp_pos, exp_neg): Tuple of exp(M) and exp(-M). exp_neg is None
        when only_forward=True. Both same dtype as input.
    """
    # Clamp Frobenius norm to prevent overflow in matrix_exp.
    # Gradient flows through the scaling factor, so φ still gets
    # signal to shrink when it exceeds the cap.
    mat_norm = matrix.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    scale = (max_norm / mat_norm).clamp(max=1.0)
    if (scale < 1.0).any():
        _nr("matexp_norm_clamp", count=int((scale < 1.0).sum().item()))
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
        if only_forward:
            exp_neg = None
        elif skew_symmetric:
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
    eye = _cached_eye(K, X.device, X.dtype)

    # Rescale to bring singular values into convergence basin (0, √3).
    # Use Frobenius norm as an UPPER BOUND on the largest singular value
    # (since σ_max ≤ ‖X‖_F), and rescale so ‖X‖_F / scale ≤ √3 - margin.
    # This guarantees σ_max < √3, unlike the previous RMS-based heuristic
    # which could leave outlier singular values above √3 for ill-conditioned
    # matrices.
    frob = X.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    sqrt3_safe = 1.7  # √3 ≈ 1.732, leave margin for safety
    needs_rescale = (frob > sqrt3_safe).squeeze(-1).squeeze(-1)
    if needs_rescale.any():
        X = torch.where(
            needs_rescale[..., None, None],
            X * (sqrt3_safe / frob),
            X,
        )

    # Quadratic convergence: error is O(eps^{2^k}), so 5 iterations always
    # converges to machine precision. Skip per-iteration convergence check
    # to avoid GPU-CPU sync from .all().
    for _ in range(n_iters):
        XtX = X.transpose(-1, -2) @ X
        X = X @ ((3.0 * eye - XtX) / 2.0)

    return X


# =============================================================================
# Fused Block-Diagonal Kernels
# =============================================================================
# These functions process ALL irrep blocks in grouped batches instead of
# launching separate matrix_exp + KL kernels per block.  For typical configs
# (e.g. 75×ℓ₀ + 30×ℓ₁ + 18×ℓ₂ = 123 blocks, 3 unique dims), this reduces
# CUDA kernel launches from O(num_blocks) to O(num_unique_dims).

# Cache for generator sub-block stacks.  Keyed by (data_ptr, shape, irrep_dims)
# for stability across forward passes within a process.  data_ptr() is stable
# for registered buffers (they don't move unless .to() is called).  Bounded to
# _GEN_STACK_CACHE_MAX entries to prevent unbounded growth in multi-model runs.
_gen_stack_cache: Dict[Tuple, Dict[int, torch.Tensor]] = {}
_GEN_STACK_CACHE_MAX: int = 8


def _gen_cache_key(generators: torch.Tensor, irrep_dims: List[int]) -> Tuple:
    """Content-stable cache key for generator sub-block stacks.

    Uses data_ptr + shape + device as a fast proxy for content identity.
    Falls back to full content hash if data_ptr is 0 (e.g., meta tensors).
    """
    ptr = generators.data_ptr()
    if ptr == 0:
        # Meta tensor or unusual case — hash content
        return (generators.shape, generators.device, tuple(irrep_dims),
                hash(generators.cpu().numpy().tobytes()))
    return (ptr, generators.shape, generators.device, tuple(irrep_dims))


def fused_block_matrix_exp_pairs(
    phi: torch.Tensor,
    generators: torch.Tensor,
    irrep_dims: List[int],
    enforce_orthogonal: bool = False,
    skew_symmetric: bool = False,
    only_forward: bool = False,
) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
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
        enforce_orthogonal: if True, apply Newton-Schulz for blocks with d >= 2.
        only_forward: if True, skip computing exp(-M) per block. Use for
            PriorBank decode where only A (not A⁻¹) is needed. Saves one
            matrix_exp call per unique block dim and halves memory.

    Returns:
        List of (exp_phi_block, exp_neg_phi_block) tuples, one per block in
        the same order as *irrep_dims*.  Each tensor has shape (B, N, d, d).
        exp_neg_phi_block is None when only_forward=True.
    """
    B, N, _ = phi.shape

    # Silent dimension-mismatch guard.  Without this, an irrep_dims list
    # that does not partition the generator space correctly would slice
    # generators at invalid ranges (generators[:, s:e, s:e] with s >= K
    # returns an empty tensor, and the downstream matrix_exp silently
    # produces nonsense).  Mirrors the assertion in
    # fused_block_diagonal_kl_diag / fused_block_diagonal_kl_full.
    _K_gen = generators.shape[-1]
    _id_sum = sum(irrep_dims)
    if _id_sum != _K_gen:
        raise ValueError(
            f"fused_block_matrix_exp_pairs: sum(irrep_dims)={_id_sum} does "
            f"not equal generators.shape[-1]={_K_gen}. "
            f"irrep_dims={irrep_dims}, generators.shape={tuple(generators.shape)}."
        )
    if phi.shape[-1] != generators.shape[0]:
        raise ValueError(
            f"fused_block_matrix_exp_pairs: phi.shape[-1]={phi.shape[-1]} "
            f"does not equal generators.shape[0]={generators.shape[0]} (n_gen). "
            f"phi.shape={tuple(phi.shape)}, generators.shape={tuple(generators.shape)}."
        )

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

    # Content-stable cache key (survives checkpoint reload, multiprocessing)
    _cache_key = _gen_cache_key(generators, irrep_dims)

    for d, group in dim_groups.items():
        n_blocks = len(group)

        # Reuse cached generator stacks when possible (generators are fixed
        # architectural constants registered as buffers).
        if _cache_key in _gen_stack_cache and d in _gen_stack_cache[_cache_key]:
            gen_stack = _gen_stack_cache[_cache_key][d]
        else:
            gen_stack = torch.stack(
                [generators[:, s:e, s:e] for _, s, e in group], dim=0
            )
            # Evict oldest entry if cache is full
            if len(_gen_stack_cache) >= _GEN_STACK_CACHE_MAX:
                _gen_stack_cache.pop(next(iter(_gen_stack_cache)))
            if _cache_key not in _gen_stack_cache:
                _gen_stack_cache[_cache_key] = {}
            _gen_stack_cache[_cache_key][d] = gen_stack

        # Batched Lie-algebra element: phi · G per block
        # phi: (B, N, n_gen), gen_stack: (n_blocks, n_gen, d, d)
        #  → (n_blocks, B, N, d, d)
        phi_matrices = torch.einsum('bna,gaij->gbnij', phi, gen_stack)

        # Merge block-batch and batch dims for a single matrix_exp call
        phi_flat = phi_matrices.reshape(n_blocks * B, N, d, d)
        exp_phi_flat, exp_neg_phi_flat = stable_matrix_exp_pair(
            phi_flat, skew_symmetric=skew_symmetric, only_forward=only_forward,
        )

        # Reshape back: (n_blocks, B, N, d, d)
        exp_phi_all = exp_phi_flat.reshape(n_blocks, B, N, d, d)
        exp_neg_phi_all = (
            exp_neg_phi_flat.reshape(n_blocks, B, N, d, d)
            if exp_neg_phi_flat is not None else None
        )

        if enforce_orthogonal and d >= 2:
            shape = exp_phi_all.shape
            exp_phi_all = newton_schulz_orthogonalize(
                exp_phi_all.reshape(-1, d, d)
            ).reshape(shape)
            if exp_neg_phi_all is not None:
                exp_neg_phi_all = newton_schulz_orthogonalize(
                    exp_neg_phi_all.reshape(-1, d, d)
                ).reshape(shape)

        # Scatter results back to per-block list
        for local_idx, (global_idx, _, _) in enumerate(group):
            results[global_idx] = (
                exp_phi_all[local_idx].contiguous(),
                exp_neg_phi_all[local_idx].contiguous() if exp_neg_phi_all is not None else None,
            )

    return results  # type: ignore[return-value]


def fused_block_diagonal_kl_diag(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    block_exp_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    irrep_dims: List[int],
    eps: float = 1e-6,
    _tile_size: int = 16,
    alpha_div: float = 1.0,
) -> torch.Tensor:
    r"""Fused block-diagonal KL or Rényi :math:`\alpha`-divergence for diagonal covariance.

    Memory-efficient implementation with three dispatch paths:

    1. **Triton kernels** (d=1,3,5 on CUDA): zero intermediate memory — the
       entire matrix_exp product, transport, and KL computation stays in
       GPU registers.  Eliminates ~400–600 MB of (n_blocks, B, N, N, d, d)
       Omega intermediates.

    2. **Scalar fast path** (d=1 PyTorch fallback): pure element-wise ops
       with no matrix operations.  For 75 scalar blocks this avoids all
       einsum overhead.

    3. **Row-tiled path** (d>1 PyTorch fallback): processes ``_tile_size``
       query rows at a time, reducing peak Omega memory by a factor of
       ``N / _tile_size``.  For N=64, _tile_size=16 gives 4× memory
       reduction.

    When ``alpha_div != 1.0``, the Rényi :math:`\alpha`-divergence is used:

    .. math::
        D_\alpha(q \| p) = \tfrac{1}{2}\!\left[
            \alpha \sum_k \frac{\Delta\mu_k^2}{\tilde\sigma_k}
            + \frac{1}{\alpha-1}\sum_k
              \bigl((1-\alpha)\log s_k + \alpha\log t_k - \log\tilde\sigma_k\bigr)
        \right]

    where :math:`\tilde\sigma_k = (1-\alpha)s_k + \alpha t_k`.

    Args:
        mu_q: (B, N, K) belief means.
        sigma_q: (B, N, K) diagonal variances (already clamped to >= eps).
        block_exp_pairs: list of (exp_phi, exp_neg_phi) per block.
        irrep_dims: block dimensions [d₁, d₂, ...].
        eps: numerical stability floor.
        _tile_size: number of query rows per tile for d>1 blocks.
        alpha_div: Rényi divergence order (default 1.0 = KL).

    Returns:
        kl_total: (B, N, N) total divergence across all blocks.
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    kl_max = max(KL_CEIL_BASE, KL_CEIL_MULT * K)

    # Silent dimension-mismatch guard.  Without this, irrep_dims that don't
    # sum to K cause the block loop to accumulate KL only over the first
    # sum(irrep_dims) channels and silently drop the tail, producing an
    # under-sum that biases attention toward false-negative divergence.
    _id_sum = sum(irrep_dims)
    if _id_sum != K:
        raise ValueError(
            f"fused_block_diagonal_kl_diag: sum(irrep_dims)={_id_sum} does "
            f"not equal K={K}. irrep_dims={irrep_dims}, mu_q.shape={tuple(mu_q.shape)}."
        )
    if len(block_exp_pairs) != len(irrep_dims):
        raise ValueError(
            f"fused_block_diagonal_kl_diag: len(block_exp_pairs)="
            f"{len(block_exp_pairs)} does not match len(irrep_dims)="
            f"{len(irrep_dims)}."
        )

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
                # Squeeze out trivial 1×1 matrix dims → (n_blocks, B, N)
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

                if abs(alpha_div - 1.0) < 1e-6:
                    # Standard KL
                    kl = 0.5 * (sig_i / sig_t + delta * delta / sig_t - 1.0
                                + torch.log(sig_t) - torch.log(sig_i))
                else:
                    # Rényi α-divergence
                    sig_blend = ((1.0 - alpha_div) * sig_i
                                 + alpha_div * sig_t).clamp(min=eps)
                    mahal = alpha_div * delta * delta / sig_blend
                    logdet = ((1.0 - alpha_div) * torch.log(sig_i)
                              + alpha_div * torch.log(sig_t)
                              - torch.log(sig_blend)) / (alpha_div - 1.0)
                    kl = 0.5 * (mahal + logdet)
                kl = kl.clamp(min=0.0, max=kl_max)
                if torch.isnan(kl).any():
                    _nr("fused_kl_nan_replace", count=int(torch.isnan(kl).sum().item()))
                kl = kl.nan_to_num(nan=kl_max, posinf=kl_max, neginf=0.0)

            kl_total += kl.sum(dim=0)  # sum over blocks → (B,N,N)

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

                    if abs(alpha_div - 1.0) < 1e-6:
                        # Standard KL
                        trace = (sig_i / sig_t).sum(dim=-1)
                        mahal = (delta * delta / sig_t).sum(dim=-1)
                        logdet = (torch.log(sig_t) - torch.log(sig_i)).sum(dim=-1)
                        kl_tile = 0.5 * (trace + mahal - d + logdet)
                    else:
                        # Rényi α-divergence
                        sig_blend = ((1.0 - alpha_div) * sig_i
                                     + alpha_div * sig_t).clamp(min=eps)
                        mahal = (alpha_div * delta * delta / sig_blend).sum(dim=-1)
                        logdet = (
                            (1.0 - alpha_div) * torch.log(sig_i.clamp(min=eps))
                            + alpha_div * torch.log(sig_t)
                            - torch.log(sig_blend)
                        ).sum(dim=-1) / (alpha_div - 1.0)
                        kl_tile = 0.5 * (mahal + logdet)
                    kl_tile = kl_tile.clamp(min=0.0, max=kl_max)
                    if torch.isnan(kl_tile).any():
                        _nr("fused_kl_nan_replace", count=int(torch.isnan(kl_tile).sum().item()))
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
    alpha_div: float = 1.0,
    sigma_floor: Optional[float] = None,
) -> torch.Tensor:
    r"""Fused block-diagonal KL or Rényi :math:`\alpha`-divergence for full covariance.

    Groups same-sized irrep blocks and computes transport + divergence in
    batched passes.  Falls back to diagonal approximation on Cholesky failure.

    When ``alpha_div != 1.0``, uses the blended covariance
    :math:`\tilde{\Sigma} = (1-\alpha)\Sigma_q + \alpha\Sigma_t` and computes:

    .. math::
        D_\alpha(q \| p) = \tfrac{1}{2}\Bigl[
            \alpha \cdot \delta\mu^\top \tilde{\Sigma}^{-1} \delta\mu
            + \tfrac{1}{\alpha-1}\bigl(
                (1-\alpha)\log|\Sigma_q| + \alpha\log|\Sigma_t|
                - \log|\tilde{\Sigma}|
            \bigr)
        \Bigr]

    Args:
        mu_q: (B, N, K) belief means.
        sigma_q: (B, N, K, K) block-diagonal covariances.
        block_exp_pairs: list of (exp_phi, exp_neg_phi) per block.
        irrep_dims: block dimensions [d₁, d₂, ...].
        eps: numerical stability floor.
        alpha_div: Rényi divergence order (default 1.0 = standard KL).
        sigma_floor: If set, use ``spd_eigfloor(Σ, floor=sigma_floor)`` to
            regularize Σ_q and Σ_t before Cholesky. Bounds the condition
            number and matches the E-step floor. When ``None`` falls back to
            the legacy ``+ eps · I_d`` jitter (may fail Cholesky on
            block-diagonal Σ with collapsing per-block eigenvalues).

    Returns:
        kl_total: (B, N, N) total divergence across all blocks.
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    kl_max = max(KL_CEIL_BASE, KL_CEIL_MULT * K)

    # Silent dimension-mismatch guard (see fused_block_diagonal_kl_diag).
    _id_sum = sum(irrep_dims)
    if _id_sum != K:
        raise ValueError(
            f"fused_block_diagonal_kl_full: sum(irrep_dims)={_id_sum} does "
            f"not equal K={K}. irrep_dims={irrep_dims}, mu_q.shape={tuple(mu_q.shape)}."
        )
    if len(block_exp_pairs) != len(irrep_dims):
        raise ValueError(
            f"fused_block_diagonal_kl_full: len(block_exp_pairs)="
            f"{len(block_exp_pairs)} does not match len(irrep_dims)="
            f"{len(irrep_dims)}."
        )

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

            if sigma_floor is not None:
                # Bound λ_min(Σ) ≥ sigma_floor via spectral clamp. Respects
                # the E-step floor, prevents Cholesky failure on block-diagonal
                # Σ with collapsing per-block eigenvalues (see
                # `gauge_kl_full_chol_fallback_diag` pathology under
                # `use_output_projection=False`).
                sigma_i_reg = spd_eigfloor(sigma_i, floor=sigma_floor)
                sigma_t_reg = spd_eigfloor(sigma_transported, floor=sigma_floor)
            else:
                sigma_i_reg = sigma_i + eps * I_d
                sigma_t_reg = sigma_transported + eps * I_d

            try:
                if abs(alpha_div - 1.0) < 1e-6:
                    # Standard KL divergence.
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
                else:
                    # Rényi α-divergence for full covariance.
                    # Blended covariance: Σ̃ = (1-α)Σ_q + α·Σ_t
                    sigma_blend = (1.0 - alpha_div) * sigma_i_reg + alpha_div * sigma_t_reg
                    sigma_blend = 0.5 * (sigma_blend + sigma_blend.transpose(-1, -2))

                    L_blend = torch.linalg.cholesky(sigma_blend)
                    L_q = torch.linalg.cholesky(sigma_i_reg)
                    L_t = torch.linalg.cholesky(sigma_t_reg)

                    # Mahalanobis: α · δμᵀ Σ̃⁻¹ δμ
                    delta_mu = mu_transported - mu_i
                    v = torch.linalg.solve_triangular(
                        L_blend, delta_mu.unsqueeze(-1), upper=False
                    ).squeeze(-1)
                    mahal_term = alpha_div * torch.sum(v ** 2, dim=-1)

                    # Log-det terms
                    logdet_q = 2.0 * torch.sum(
                        torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1) + eps), dim=-1
                    )
                    logdet_t = 2.0 * torch.sum(
                        torch.log(torch.diagonal(L_t, dim1=-2, dim2=-1) + eps), dim=-1
                    )
                    logdet_blend = 2.0 * torch.sum(
                        torch.log(torch.diagonal(L_blend, dim1=-2, dim2=-1) + eps), dim=-1
                    )

                    logdet_term = (
                        (1.0 - alpha_div) * logdet_q + alpha_div * logdet_t - logdet_blend
                    ) / (alpha_div - 1.0)

                    kl_group = 0.5 * (mahal_term + logdet_term)

            except RuntimeError:
                # Cholesky failed — fall back to diagonal approximation.
                # Clamp diagonals at sigma_floor when provided so fallback
                # gradients (∂logdet/∂σ = 1/σ) stay bounded; otherwise use eps.
                _nr("gauge_kl_full_chol_fallback_diag")
                _fallback_floor = sigma_floor if sigma_floor is not None else eps
                sigma_diag_t = torch.diagonal(sigma_t_reg, dim1=-2, dim2=-1).clamp(min=_fallback_floor)
                sigma_diag_i = torch.diagonal(sigma_i_reg, dim1=-2, dim2=-1).clamp(min=_fallback_floor)
                delta_mu = mu_transported - mu_i

                if abs(alpha_div - 1.0) < 1e-6:
                    trace_term = (sigma_diag_i / sigma_diag_t).sum(dim=-1)
                    mahal_term = ((delta_mu ** 2) / sigma_diag_t).sum(dim=-1)
                    logdet_term = (torch.log(sigma_diag_t) - torch.log(sigma_diag_i)).sum(dim=-1)
                    kl_group = 0.5 * (trace_term + mahal_term - d + logdet_term)
                else:
                    # Diagonal α-divergence fallback
                    sigma_blend_diag = (
                        (1.0 - alpha_div) * sigma_diag_i + alpha_div * sigma_diag_t
                    ).clamp(min=eps)
                    mahal_term = (alpha_div * (delta_mu ** 2) / sigma_blend_diag).sum(dim=-1)
                    logdet_per_dim = (
                        (1.0 - alpha_div) * torch.log(sigma_diag_i)
                        + alpha_div * torch.log(sigma_diag_t)
                        - torch.log(sigma_blend_diag)
                    )
                    logdet_term = logdet_per_dim.sum(dim=-1) / (alpha_div - 1.0)
                    kl_group = 0.5 * (mahal_term + logdet_term)

            kl_group = kl_group.clamp(min=0.0, max=kl_max)
            if torch.isnan(kl_group).any():
                _nr("fused_kl_nan_replace", count=int(torch.isnan(kl_group).sum().item()))
            kl_group = kl_group.nan_to_num(nan=kl_max, posinf=kl_max, neginf=0.0)

        kl_total += kl_group.sum(dim=0)

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
        where c_r = 1 - σ_i[r]/σ_t[r] - δ_r²/σ_t[r], δ_r = μ_i[r] - μ_t[r]
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
