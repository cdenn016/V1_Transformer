"""
Custom Triton Kernels for Fused Pairwise KL Divergence
======================================================

Fuses matrix_exp product (Omega), Gaussian transport, and diagonal KL
divergence into a single kernel per (b, i, j) pair.  All d×d intermediates
(Omega, transported mu/sigma) stay in registers — no global memory allocation.

For a typical config with 75 scalar (d=1), 30 vector (d=3), and 18 tensor
(d=5) irrep blocks:
  - Eliminates ~400 MB of intermediate (B,N,N,d,d) Omega tensors
  - Memory traffic drops from ~5 round-trips × globals to 1 read + 1 write
  - Each program: O(n_blocks × d²) register ops, writes 1 scalar to output

Usage::

    from transformer.core.triton_kernels import pairwise_kl_diag_triton

    # block_exp_pairs from fused_block_matrix_exp_pairs()
    kl = pairwise_kl_diag_triton(mu_q, sigma_q, block_exp_pairs, irrep_dims)

Falls back gracefully when Triton is not available (import guarded).
"""

import torch
from collections import defaultdict
from typing import List, Tuple

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _pairwise_kl_d1_kernel(
        kl_out_ptr,
        ep_ptr, en_ptr, mu_ptr, sig_ptr,
        N, n_blocks,
        # Strides (element counts, not bytes)
        ep_stride_blk,   # = B * N  (between blocks in flat scalar arrays)
        mu_stride_blk,   # = B * N  (same layout)
        eps: tl.constexpr,
        NBLK: tl.constexpr,   # padded n_blocks (next power of 2)
    ):
        """Fused KL for all scalar (d=1) irrep blocks.

        Each program handles one (b, i, j) pair, vectorises over all
        n_blocks scalar blocks.  Reads 4 × n_blocks values, writes 1 float.
        """
        pid = tl.program_id(0)
        j = pid % N
        i = (pid // N) % N
        b = pid // (N * N)

        blk_range = tl.arange(0, NBLK)
        mask = blk_range < n_blocks

        base_i = b * N + i
        base_j = b * N + j

        epi = tl.load(ep_ptr + blk_range * ep_stride_blk + base_i,
                       mask=mask, other=1.0)
        enj = tl.load(en_ptr + blk_range * ep_stride_blk + base_j,
                       mask=mask, other=1.0)
        mi = tl.load(mu_ptr + blk_range * mu_stride_blk + base_i,
                      mask=mask, other=0.0)
        mj = tl.load(mu_ptr + blk_range * mu_stride_blk + base_j,
                      mask=mask, other=0.0)
        si = tl.load(sig_ptr + blk_range * mu_stride_blk + base_i,
                      mask=mask, other=1.0)
        sj = tl.load(sig_ptr + blk_range * mu_stride_blk + base_j,
                      mask=mask, other=1.0)

        omega = epi * enj
        mu_t = omega * mj
        sig_t = tl.maximum(omega * omega * sj, eps)
        si_safe = tl.maximum(si, eps)

        delta = mi - mu_t
        kl = 0.5 * (si_safe / sig_t
                     + delta * delta / sig_t
                     - 1.0
                     + tl.log(sig_t) - tl.log(si_safe))
        kl = tl.maximum(kl, 0.0)
        kl = tl.where(mask, kl, 0.0)

        tl.atomic_add(kl_out_ptr + b * N * N + i * N + j, tl.sum(kl))


    @triton.jit
    def _pairwise_kl_d3_kernel(
        kl_out_ptr,
        ep_ptr, en_ptr, mu_ptr, sig_ptr,
        N, n_blocks,
        ep_stride_blk,   # = B * N * 9  (flat d×d per position)
        ep_stride_pos,    # = 9          (elements per position)
        mu_stride_blk,    # = B * N * 3
        mu_stride_pos,    # = 3
        eps: tl.constexpr,
        NBLK: tl.constexpr,
    ):
        """Fused KL for all d=3 irrep blocks.

        3×3 matmul + transport + KL fully in registers.
        """
        pid = tl.program_id(0)
        j = pid % N
        i = (pid // N) % N
        b = pid // (N * N)

        blk_range = tl.arange(0, NBLK)
        mask = blk_range < n_blocks

        base_i = b * N + i
        base_j = b * N + j

        ep_base_i = blk_range * ep_stride_blk + base_i * ep_stride_pos
        en_base_j = blk_range * ep_stride_blk + base_j * ep_stride_pos

        # Load 3×3 exp_phi[b,i] and exp_neg_phi[b,j] for all blocks
        a0 = tl.load(ep_ptr + ep_base_i + 0, mask=mask, other=0.0)
        a1 = tl.load(ep_ptr + ep_base_i + 1, mask=mask, other=0.0)
        a2 = tl.load(ep_ptr + ep_base_i + 2, mask=mask, other=0.0)
        a3 = tl.load(ep_ptr + ep_base_i + 3, mask=mask, other=0.0)
        a4 = tl.load(ep_ptr + ep_base_i + 4, mask=mask, other=0.0)
        a5 = tl.load(ep_ptr + ep_base_i + 5, mask=mask, other=0.0)
        a6 = tl.load(ep_ptr + ep_base_i + 6, mask=mask, other=0.0)
        a7 = tl.load(ep_ptr + ep_base_i + 7, mask=mask, other=0.0)
        a8 = tl.load(ep_ptr + ep_base_i + 8, mask=mask, other=0.0)

        b0 = tl.load(en_ptr + en_base_j + 0, mask=mask, other=0.0)
        b1 = tl.load(en_ptr + en_base_j + 1, mask=mask, other=0.0)
        b2 = tl.load(en_ptr + en_base_j + 2, mask=mask, other=0.0)
        b3 = tl.load(en_ptr + en_base_j + 3, mask=mask, other=0.0)
        b4 = tl.load(en_ptr + en_base_j + 4, mask=mask, other=0.0)
        b5 = tl.load(en_ptr + en_base_j + 5, mask=mask, other=0.0)
        b6 = tl.load(en_ptr + en_base_j + 6, mask=mask, other=0.0)
        b7 = tl.load(en_ptr + en_base_j + 7, mask=mask, other=0.0)
        b8 = tl.load(en_ptr + en_base_j + 8, mask=mask, other=0.0)

        # 3×3 matmul: Omega = A @ B  (row-major: A[r,k] = a[r*3+k])
        c00 = a0*b0 + a1*b3 + a2*b6
        c01 = a0*b1 + a1*b4 + a2*b7
        c02 = a0*b2 + a1*b5 + a2*b8
        c10 = a3*b0 + a4*b3 + a5*b6
        c11 = a3*b1 + a4*b4 + a5*b7
        c12 = a3*b2 + a4*b5 + a5*b8
        c20 = a6*b0 + a7*b3 + a8*b6
        c21 = a6*b1 + a7*b4 + a8*b7
        c22 = a6*b2 + a7*b5 + a8*b8

        # Load beliefs
        mu_base_i = blk_range * mu_stride_blk + base_i * mu_stride_pos
        mu_base_j = blk_range * mu_stride_blk + base_j * mu_stride_pos

        mi0 = tl.load(mu_ptr + mu_base_i + 0, mask=mask, other=0.0)
        mi1 = tl.load(mu_ptr + mu_base_i + 1, mask=mask, other=0.0)
        mi2 = tl.load(mu_ptr + mu_base_i + 2, mask=mask, other=0.0)

        mj0 = tl.load(mu_ptr + mu_base_j + 0, mask=mask, other=0.0)
        mj1 = tl.load(mu_ptr + mu_base_j + 1, mask=mask, other=0.0)
        mj2 = tl.load(mu_ptr + mu_base_j + 2, mask=mask, other=0.0)

        si0 = tl.maximum(tl.load(sig_ptr + mu_base_i + 0, mask=mask, other=1.0), eps)
        si1 = tl.maximum(tl.load(sig_ptr + mu_base_i + 1, mask=mask, other=1.0), eps)
        si2 = tl.maximum(tl.load(sig_ptr + mu_base_i + 2, mask=mask, other=1.0), eps)

        sj0 = tl.load(sig_ptr + mu_base_j + 0, mask=mask, other=1.0)
        sj1 = tl.load(sig_ptr + mu_base_j + 1, mask=mask, other=1.0)
        sj2 = tl.load(sig_ptr + mu_base_j + 2, mask=mask, other=1.0)

        # Transport mean: mu_t = Omega @ mu_j
        mt0 = c00*mj0 + c01*mj1 + c02*mj2
        mt1 = c10*mj0 + c11*mj1 + c12*mj2
        mt2 = c20*mj0 + c21*mj1 + c22*mj2

        # Transport sigma (diagonal): st[r] = sum_c Omega[r,c]² * sigma_j[c]
        st0 = tl.maximum(c00*c00*sj0 + c01*c01*sj1 + c02*c02*sj2, eps)
        st1 = tl.maximum(c10*c10*sj0 + c11*c11*sj1 + c12*c12*sj2, eps)
        st2 = tl.maximum(c20*c20*sj0 + c21*c21*sj1 + c22*c22*sj2, eps)

        # Diagonal KL
        d0 = mi0 - mt0
        d1 = mi1 - mt1
        d2 = mi2 - mt2

        kl = 0.5 * (
            (si0/st0 + si1/st1 + si2/st2)
            + (d0*d0/st0 + d1*d1/st1 + d2*d2/st2)
            - 3.0
            + (tl.log(st0) - tl.log(si0))
            + (tl.log(st1) - tl.log(si1))
            + (tl.log(st2) - tl.log(si2))
        )
        kl = tl.maximum(kl, 0.0)
        kl = tl.where(mask, kl, 0.0)

        tl.atomic_add(kl_out_ptr + b * N * N + i * N + j, tl.sum(kl))


    @triton.jit
    def _pairwise_kl_d5_kernel(
        kl_out_ptr,
        ep_ptr, en_ptr, mu_ptr, sig_ptr,
        N, n_blocks,
        ep_stride_blk, ep_stride_pos,
        mu_stride_blk, mu_stride_pos,
        eps: tl.constexpr,
        NBLK: tl.constexpr,
    ):
        """Fused KL for all d=5 irrep blocks.  5×5 matmul in registers."""
        pid = tl.program_id(0)
        j = pid % N
        i = (pid // N) % N
        b = pid // (N * N)

        blk_range = tl.arange(0, NBLK)
        mask = blk_range < n_blocks

        base_i = b * N + i
        base_j = b * N + j

        ep_off_i = blk_range * ep_stride_blk + base_i * ep_stride_pos
        en_off_j = blk_range * ep_stride_blk + base_j * ep_stride_pos

        # Load 5×5 matrices A (exp_phi[b,i]) and B (exp_neg_phi[b,j])
        # a[r*5+k], b[k*5+c]  — row-major
        a = [tl.load(ep_ptr + ep_off_i + idx, mask=mask, other=(1.0 if idx % 6 == 0 else 0.0))
             for idx in range(25)]
        bb = [tl.load(en_ptr + en_off_j + idx, mask=mask, other=(1.0 if idx % 6 == 0 else 0.0))
              for idx in range(25)]

        # 5×5 matmul: c[r*5+c] = sum_k a[r*5+k] * bb[k*5+c]
        c = []
        for r in range(5):
            for col in range(5):
                val = a[r*5+0] * bb[0*5+col]
                for k in range(1, 5):
                    val = val + a[r*5+k] * bb[k*5+col]
                c.append(val)

        # Load beliefs
        mu_off_i = blk_range * mu_stride_blk + base_i * mu_stride_pos
        mu_off_j = blk_range * mu_stride_blk + base_j * mu_stride_pos

        mi = [tl.load(mu_ptr + mu_off_i + dim, mask=mask, other=0.0) for dim in range(5)]
        mj = [tl.load(mu_ptr + mu_off_j + dim, mask=mask, other=0.0) for dim in range(5)]
        si = [tl.maximum(tl.load(sig_ptr + mu_off_i + dim, mask=mask, other=1.0), eps)
              for dim in range(5)]
        sj = [tl.load(sig_ptr + mu_off_j + dim, mask=mask, other=1.0) for dim in range(5)]

        # Transport mean: mt[r] = sum_c Omega[r,c] * mu_j[c]
        mt = []
        for r in range(5):
            val = c[r*5+0] * mj[0]
            for col in range(1, 5):
                val = val + c[r*5+col] * mj[col]
            mt.append(val)

        # Transport sigma (diagonal): st[r] = sum_c Omega[r,c]² * sigma_j[c]
        st = []
        for r in range(5):
            val = c[r*5+0]*c[r*5+0]*sj[0]
            for col in range(1, 5):
                val = val + c[r*5+col]*c[r*5+col]*sj[col]
            st.append(tl.maximum(val, eps))

        # Diagonal KL
        kl = tl.zeros_like(mi[0])
        for dim in range(5):
            delta = mi[dim] - mt[dim]
            kl = kl + si[dim]/st[dim] + delta*delta/st[dim] + tl.log(st[dim]) - tl.log(si[dim])
        kl = 0.5 * (kl - 5.0)
        kl = tl.maximum(kl, 0.0)
        kl = tl.where(mask, kl, 0.0)

        tl.atomic_add(kl_out_ptr + b * N * N + i * N + j, tl.sum(kl))


def _next_power_of_2(n: int) -> int:
    """Round up to the next power of 2 (minimum 1)."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def _pack_scalar_blocks(
    block_exp_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    group: list,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack d=1 blocks into contiguous (n_blocks, B, N) tensors for the Triton kernel."""
    ep_list = [block_exp_pairs[idx][0].squeeze(-1).squeeze(-1) for idx, _, _ in group]
    en_list = [block_exp_pairs[idx][1].squeeze(-1).squeeze(-1) for idx, _, _ in group]
    mu_list = [mu_q[:, :, s] for _, s, _ in group]
    sig_list = [sigma_q[:, :, s] for _, s, _ in group]
    return (
        torch.stack(ep_list, dim=0).contiguous(),   # (n_blocks, B, N)
        torch.stack(en_list, dim=0).contiguous(),
        torch.stack(mu_list, dim=0).contiguous(),
        torch.stack(sig_list, dim=0).contiguous(),
    )


def _pack_matrix_blocks(
    block_exp_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    group: list,
    d: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack d>1 blocks into contiguous (n_blocks, B, N, d*d) / (n_blocks, B, N, d) tensors."""
    B, N, _ = mu_q.shape
    ep_list = [block_exp_pairs[idx][0].reshape(B, N, d*d) for idx, _, _ in group]
    en_list = [block_exp_pairs[idx][1].reshape(B, N, d*d) for idx, _, _ in group]
    mu_list = [mu_q[:, :, s:e] for _, s, e in group]
    sig_list = [sigma_q[:, :, s:e] for _, s, e in group]
    return (
        torch.stack(ep_list, dim=0).contiguous(),   # (n_blocks, B, N, d*d)
        torch.stack(en_list, dim=0).contiguous(),
        torch.stack(mu_list, dim=0).contiguous(),    # (n_blocks, B, N, d)
        torch.stack(sig_list, dim=0).contiguous(),
    )


def pairwise_kl_diag_triton(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    block_exp_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    irrep_dims: List[int],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute pairwise KL via fused Triton kernels — zero intermediate memory.

    Dispatches to specialised kernels for d=1, d=3, d=5.  Falls back to
    the tiled PyTorch path for unsupported d values.

    Args:
        mu_q: (B, N, K) belief means.
        sigma_q: (B, N, K) diagonal variances.
        block_exp_pairs: list of (exp_phi, exp_neg_phi) per block.
        irrep_dims: block dimensions [d₁, d₂, ...].
        eps: numerical stability floor.

    Returns:
        kl_total: (B, N, N) pairwise KL.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")

    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    kl_out = torch.zeros(B, N, N, device=device, dtype=dtype)
    grid = (B * N * N,)

    # Group blocks by dimension
    block_ranges: list = []
    start = 0
    for idx, d in enumerate(irrep_dims):
        block_ranges.append((idx, start, start + d, d))
        start += d

    dim_groups: dict = defaultdict(list)
    for idx, s, e, d in block_ranges:
        dim_groups[d].append((idx, s, e))

    # Track which dim groups fall back to PyTorch (unsupported d)
    fallback_groups: dict = {}

    for d, group in dim_groups.items():
        n_blocks = len(group)
        NBLK = _next_power_of_2(n_blocks)

        if d == 1:
            ep, en, mu, sig = _pack_scalar_blocks(block_exp_pairs, mu_q, sigma_q, group)
            stride_blk = B * N  # elements between blocks
            _pairwise_kl_d1_kernel[grid](
                kl_out,
                ep, en, mu, sig,
                N, n_blocks,
                stride_blk, stride_blk,
                eps=eps,
                NBLK=NBLK,
            )

        elif d == 3:
            ep, en, mu, sig = _pack_matrix_blocks(
                block_exp_pairs, mu_q, sigma_q, group, d)
            _pairwise_kl_d3_kernel[grid](
                kl_out,
                ep, en, mu, sig,
                N, n_blocks,
                B * N * 9, 9,   # ep strides
                B * N * 3, 3,   # mu strides
                eps=eps,
                NBLK=NBLK,
            )

        elif d == 5:
            ep, en, mu, sig = _pack_matrix_blocks(
                block_exp_pairs, mu_q, sigma_q, group, d)
            _pairwise_kl_d5_kernel[grid](
                kl_out,
                ep, en, mu, sig,
                N, n_blocks,
                B * N * 25, 25,  # ep strides
                B * N * 5, 5,    # mu strides
                eps=eps,
                NBLK=NBLK,
            )

        else:
            # Unsupported d — collect for PyTorch fallback
            fallback_groups[d] = group

    return kl_out, fallback_groups
