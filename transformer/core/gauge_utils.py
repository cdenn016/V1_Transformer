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

try:
    from .triton_kernels import pairwise_kl_diag_triton as _pairwise_kl_diag_triton
    from .triton_kernels import TRITON_AVAILABLE as _TRITON_AVAILABLE
except ImportError:
    _TRITON_AVAILABLE = False
    _pairwise_kl_diag_triton = None


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
            matrix_f64 = matrix.double().contiguous()
            exp_pos = torch.linalg.matrix_exp(matrix_f64).to(orig_dtype)
            exp_neg = torch.linalg.matrix_exp(-matrix_f64).to(orig_dtype)
        else:
            matrix_f32 = matrix.float().contiguous()
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


# =============================================================================
# Fused Block-Diagonal Kernels
# =============================================================================
# These functions process ALL irrep blocks in grouped batches instead of
# launching separate matrix_exp + KL kernels per block.  For typical configs
# (e.g. 75×ℓ₀ + 30×ℓ₁ + 18×ℓ₂ = 123 blocks, 3 unique dims), this reduces
# CUDA kernel launches from O(num_blocks) to O(num_unique_dims).


def fused_block_matrix_exp_pairs(
    phi: torch.Tensor,
    generators: torch.Tensor,
    irrep_dims: List[int],
    enforce_orthogonal: bool = False,
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
        #  → (n_blocks, B, N, d, d)
        phi_matrices = torch.einsum('bna,gaij->gbnij', phi, gen_stack)

        # Merge block-batch and batch dims for a single matrix_exp call
        phi_flat = phi_matrices.reshape(n_blocks * B, N, d, d)
        exp_phi_flat, exp_neg_phi_flat = stable_matrix_exp_pair(phi_flat)

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
    kl_max = max(100.0, 5.0 * K)

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

    # ── Triton dispatch (zero intermediate memory) ──────────────────────
    groups_to_process = dim_groups
    if _TRITON_AVAILABLE and device.type == 'cuda':
        try:
            kl_triton, fallback_groups = _pairwise_kl_diag_triton(
                mu_q, sigma_q, block_exp_pairs, irrep_dims, eps
            )
            kl_total = kl_total + kl_triton.clamp(max=kl_max).nan_to_num(
                nan=0.0, posinf=kl_max, neginf=0.0
            )
            groups_to_process = fallback_groups  # empty if all dims handled
        except Exception:
            pass  # fall through to PyTorch path

    # ── PyTorch path (tiled for memory efficiency) ──────────────────────
    for d, group in groups_to_process.items():
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
            # Squeeze out trivial 1×1 matrix dims → (n_blocks, B, N)
            ep = exp_phi_stack.squeeze(-1).squeeze(-1)
            en = exp_neg_phi_stack.squeeze(-1).squeeze(-1)
            mu_s = mu_stack.squeeze(-1)
            sig_s = sigma_stack.squeeze(-1)

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
            kl = kl.nan_to_num(nan=0.0, posinf=kl_max, neginf=0.0)

            kl_total = kl_total + kl.sum(dim=0)  # sum over blocks → (B,N,N)

        else:
            # ── Row-tiled path: peak memory reduced by N/_tile_size ─────
            for i_start in range(0, N, _tile_size):
                i_end = min(i_start + _tile_size, N)

                # Omega tile: (n_blocks, B, tile, N, d, d)
                ep_tile = exp_phi_stack[:, :, i_start:i_end]
                Omega_tile = torch.einsum(
                    'gbikl,gbjlm->gbijkm', ep_tile, exp_neg_phi_stack)

                # Transport mean and diagonal variance
                mu_t = torch.einsum(
                    'gbijkl,gbjl->gbijk', Omega_tile, mu_stack)
                sig_t = torch.einsum(
                    'gbijkl,gbijkl,gbjl->gbijk',
                    Omega_tile, Omega_tile, sigma_stack
                ).clamp(min=eps)
                del Omega_tile

                # KL for this tile of query rows
                mu_i = mu_stack[:, :, i_start:i_end, None, :].expand(
                    -1, -1, -1, N, -1)
                sig_i = sigma_stack[:, :, i_start:i_end, None, :].expand(
                    -1, -1, -1, N, -1)
                delta = mu_t - mu_i

                trace = (sig_i / sig_t).sum(dim=-1)
                mahal = (delta * delta / sig_t).sum(dim=-1)
                logdet = (torch.log(sig_t) - torch.log(sig_i)).sum(dim=-1)

                kl_tile = 0.5 * (trace + mahal - d + logdet)
                kl_tile = kl_tile.clamp(min=0.0, max=kl_max)
                kl_tile = kl_tile.nan_to_num(
                    nan=0.0, posinf=kl_max, neginf=0.0)

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
    kl_max = max(100.0, 5.0 * K)

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

        # Batched Omega: (n_blocks, B, N, N, d, d)
        Omega = torch.einsum('gbikl,gbjlm->gbijkm', exp_phi_stack, exp_neg_phi_stack)
        del exp_phi_stack, exp_neg_phi_stack

        # Batched transport
        mu_transported = torch.einsum('gbijkl,gbjl->gbijk', Omega, mu_stack)
        sigma_transported = torch.einsum(
            'gbijkl,gbjlm,gbijmn->gbijkn',
            Omega, sigma_stack, Omega.transpose(-1, -2)
        )
        del Omega

        I_d = torch.eye(d, device=device, dtype=dtype)
        mu_i = mu_stack[:, :, :, None, :].expand(-1, -1, -1, N, -1)
        sigma_i = sigma_stack[:, :, :, None, :, :].expand(-1, -1, -1, N, -1, -1)

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
            # Cholesky failed — fall back to diagonal approximation
            sigma_diag_t = torch.diagonal(sigma_t_reg, dim1=-2, dim2=-1).clamp(min=eps)
            sigma_diag_i = torch.diagonal(sigma_i_reg, dim1=-2, dim2=-1).clamp(min=eps)
            delta_mu = mu_transported - mu_i

            trace_term = (sigma_diag_i / sigma_diag_t).sum(dim=-1)
            mahal_term = ((delta_mu ** 2) / sigma_diag_t).sum(dim=-1)
            logdet_term = (torch.log(sigma_diag_t) - torch.log(sigma_diag_i)).sum(dim=-1)

            kl_group = 0.5 * (trace_term + mahal_term - d + logdet_term)

        kl_group = kl_group.clamp(min=0.0, max=kl_max)
        kl_group = kl_group.nan_to_num(nan=0.0, posinf=kl_max, neginf=0.0)

        kl_total = kl_total + kl_group.sum(dim=0)

        del mu_transported, sigma_transported

    return kl_total


# =============================================================================
# Analytic Phi Gradient (Custom Backward — No Autograd Graph)
# =============================================================================
# Replaces torch.autograd.grad(alignment_loss, phi) with a hand-coded backward
# pass. Eliminates the ~250MB autograd memory spike from recording the graph
# through matrix_exp → KL → softmax for each phi update.
#
# Mathematical derivation (verified symbolically in derivations/analytic_phi_grad_derivation.py):
#
#   F_align = λ Σ_{i,j} β_ij KL(q_i || Ω_ij q_j)
#
#   ∂F_align/∂φ^a_i = λ Σ_j β_ij [1 + (w_i - KL_ij)/τ] ∂KL_ij/∂φ^a_i
#
#   where w_i = Σ_j β_ij KL_ij,  τ = κ√K  (softmax temperature)
#
# The ∂KL/∂φ chain:
#   ∂KL_ij/∂φ^a_i = tr(∂KL/∂Ω_ij · ∂Ω_ij/∂φ^a_i)
#
# For diagonal covariance:
#   ∂KL/∂Ω[r,s] = c_r · Ω[r,s] · σ_j[s] / σ_t[r]  -  δ_r · μ_j[s] / σ_t[r]
#   where c_r = 1 - σ_i[r]/σ_t[r] - δ_r²/σ_t[r],  δ_r = μ_i[r] - μ_t[r]
#
# The dexp chain for ∂Ω_ij/∂φ^a_i uses the BCH-like series:
#   ∂Ω_ij/∂φ^a_i = dexp_{X_i}(G_a) · exp(-X_j)
#                 = exp(X_i) · Ψ(ad_{X_i})(G_a) · exp(-X_j)
#   where Ψ(z) = (e^z - 1)/z = Σ_{k=0}^∞ z^k/(k+1)!
#
# Efficient computation via commutator iteration:
#   Q^{(0)} = exp(-X_j) · (∂KL/∂Ω)^T · exp(X_i)
#   Q^{(k)} = [Q^{(k-1)}, X_i]  (commutator)
#   ∂KL_ij/∂φ^a_i = Σ_{k=0}^p 1/(k+1)! · tr(Q^{(k)} · G_a)


def analytic_phi_gradient_block_diag(
    mu_q: torch.Tensor,         # (B, N, K) belief means
    sigma_q: torch.Tensor,      # (B, N, K) diagonal variances
    beta: torch.Tensor,         # (B, N, N) attention weights
    kl_matrix: torch.Tensor,    # (B, N, N) KL divergence values
    phi: torch.Tensor,          # (B, N, n_gen) gauge frame coefficients
    generators: torch.Tensor,   # (n_gen, K, K) Lie algebra generators
    irrep_dims: List[int],      # block dimensions [d₁, d₂, ...]
    lambda_belief: float,       # belief alignment weight
    kappa: float,               # attention temperature
    eps: float = 1e-6,
    dexp_order: int = 4,        # truncation order for dexp series
    tile_size: int = 16,        # query tile size for memory efficiency
    enforce_orthogonal: bool = False,
    cached_block_exp_pairs: Optional[list] = None,
) -> torch.Tensor:
    """Compute ∂F_align/∂φ analytically without building an autograd graph.

    This is the memory-efficient replacement for:
        phi_for_grad = phi.clone().requires_grad_(True)
        ... compute alignment_loss through full KL pipeline ...
        grad_phi = torch.autograd.grad(alignment_loss, phi_for_grad)[0]

    Memory: O(tile_size × N × max(d²)) per tile vs O(N² × K²) for autograd.

    Args:
        mu_q: (B, N, K) belief means (detached)
        sigma_q: (B, N, K) diagonal variances (detached, clamped)
        beta: (B, N, N) attention weights β_ij
        kl_matrix: (B, N, N) KL divergence values KL(q_i || Ω_ij q_j)
        phi: (B, N, n_gen) current gauge frame coefficients
        generators: (n_gen, K, K) Lie algebra generators
        irrep_dims: block dimensions for multi-head structure
        lambda_belief: weight for belief alignment term
        kappa: softmax temperature parameter
        eps: numerical stability floor
        dexp_order: truncation order for dexp Bernoulli series (4-8 typical)
        tile_size: number of query rows per tile
        enforce_orthogonal: whether to apply Newton-Schulz orthogonalization
        cached_block_exp_pairs: precomputed (exp_phi, exp_neg_phi) per block

    Returns:
        grad_phi: (B, N, n_gen) analytic gradient ∂F_align/∂φ
    """
    import math

    B, N, K = mu_q.shape
    n_gen = generators.shape[0]
    device = mu_q.device
    dtype = mu_q.dtype

    # Work in float32 for numerical stability
    mu_q = mu_q.float()
    sigma_q = sigma_q.float().clamp(min=eps)
    beta = beta.float()
    kl_matrix = kl_matrix.float()
    phi = phi.float()

    # Softmax coupling coefficient:
    # ∂F/∂φ^a_i = λ Σ_j β_ij [1 + (w_i - KL_ij)/τ] ∂KL_ij/∂φ^a_i
    # where w_i = Σ_j β_ij KL_ij,  τ = κ√K
    tau = max(kappa * math.sqrt(max(K, 1)), eps)
    w_i = (beta * kl_matrix).sum(dim=-1, keepdim=True)  # (B, N, 1)
    coupling_weight = 1.0 + (w_i - kl_matrix) / tau     # (B, N, N)
    weight = lambda_belief * beta * coupling_weight       # (B, N, N)

    # Compute matrix exponentials (reuse cached if available)
    if cached_block_exp_pairs is not None:
        block_exp_pairs = cached_block_exp_pairs
    else:
        block_exp_pairs = fused_block_matrix_exp_pairs(
            phi, generators, irrep_dims,
            enforce_orthogonal=enforce_orthogonal,
        )

    # Build block ranges
    block_ranges = []
    start = 0
    for idx, d in enumerate(irrep_dims):
        block_ranges.append((idx, start, start + d, d))
        start += d

    # Accumulator for phi gradient
    grad_phi = torch.zeros(B, N, n_gen, device=device, dtype=torch.float32)

    # Process each block independently
    for block_idx, blk_start, blk_end, d in block_ranges:
        exp_phi_blk = block_exp_pairs[block_idx][0]      # (B, N, d, d)
        exp_neg_phi_blk = block_exp_pairs[block_idx][1]  # (B, N, d, d)

        # Extract block-level beliefs
        mu_blk = mu_q[:, :, blk_start:blk_end]           # (B, N, d)
        sig_blk = sigma_q[:, :, blk_start:blk_end]       # (B, N, d)

        # Block generators: (n_gen, d, d)
        gen_blk = generators[:, blk_start:blk_end, blk_start:blk_end]

        # Lie algebra element X_i = phi · G for this block: (B, N, d, d)
        X_blk = torch.einsum('bna,aij->bnij', phi, gen_blk)

        # Precompute w_j = exp(-X_j) @ mu_j for factored mean transport
        w_j = torch.einsum('bjkl,bjl->bjk', exp_neg_phi_blk, mu_blk)  # (B, N, d)

        # Precompute G_mat_j = exp(-X_j) diag(σ_j) exp(-X_j)^T for variance transport
        B_sigma = exp_neg_phi_blk * sig_blk[:, :, None, :]  # scale columns
        G_mat_j = B_sigma @ exp_neg_phi_blk.transpose(-1, -2)  # (B, N, d, d)
        del B_sigma

        gen_blk_t = gen_blk.transpose(-1, -2)  # (n_gen, d, d)

        # Tile over query positions i
        for i_start in range(0, N, tile_size):
            i_end = min(i_start + tile_size, N)

            exp_phi_i = exp_phi_blk[:, i_start:i_end]     # (B, tile, d, d)
            mu_i = mu_blk[:, i_start:i_end]                # (B, tile, d)
            sig_i = sig_blk[:, i_start:i_end]              # (B, tile, d)
            X_i = X_blk[:, i_start:i_end]                  # (B, tile, d, d)
            weight_tile = weight[:, i_start:i_end, :]      # (B, tile, N)

            # Transport mean: μ_t[i,j] = exp(X_i) @ w_j
            mu_t = torch.einsum('bikl,bjl->bijk', exp_phi_i, w_j)  # (B, tile, N, d)

            # Transport diagonal variance: σ_t[i,j,k] = diag(exp(X_i) G_mat_j exp(X_i)^T)[k]
            AG = torch.einsum('bikm,bjmn->bijkn', exp_phi_i, G_mat_j)  # (B, tile, N, d, d)
            sig_t = torch.einsum('bijkn,bikn->bijk', AG, exp_phi_i).clamp(min=eps)
            del AG

            # ∂KL/∂Ω coefficients
            delta = mu_i[:, :, None, :] - mu_t             # (B, tile, N, d)
            c = 1.0 - sig_i[:, :, None, :] / sig_t - delta ** 2 / sig_t
            v1 = c / sig_t                                 # (B, tile, N, d)
            v2 = delta / sig_t                             # (B, tile, N, d)

            # Build Q^{(0)} = exp(-X_j) @ (∂KL/∂Ω)^T @ exp(X_i)
            # Term1: G_mat_j @ [exp(X_i)^T @ diag(v1) @ exp(X_i)]
            exp_phi_i_t = exp_phi_i.transpose(-1, -2)      # (B, tile, d, d)
            V1_scaled = exp_phi_i_t[:, :, None, :, :] * v1[:, :, :, None, :]
            V1_mat = V1_scaled @ exp_phi_i[:, :, None, :, :]
            del V1_scaled
            Q0_term1 = torch.einsum('bjkl,bijlm->bijkm', G_mat_j, V1_mat)
            del V1_mat

            # Term2: -outer(w_j, v2 @ exp(X_i))
            v2_exp = torch.einsum('bijk,bikm->bijm', v2, exp_phi_i)
            Q0_term2 = -w_j[:, None, :, :, None] * v2_exp[:, :, :, None, :]
            del v2_exp

            Q0 = Q0_term1 + Q0_term2
            del Q0_term1, Q0_term2, v1, v2, c, delta, mu_t, sig_t

            # dexp series: ∂KL_ij/∂φ^a_i = Σ_{k=0}^p 1/(k+1)! tr(Q^{(k)} G_a)
            # k=0 term
            dkl_dphi = torch.einsum('bijkl,akl->bija', Q0, gen_blk_t)

            # Higher order terms via commutator iteration
            Q_k = Q0
            factorial_inv = 1.0
            for k in range(1, dexp_order + 1):
                factorial_inv /= (k + 1)
                comm = Q_k @ X_i[:, :, None, :, :] - X_i[:, :, None, :, :] @ Q_k
                Q_k = comm
                contrib = torch.einsum('bijkl,akl->bija', Q_k, gen_blk_t)
                dkl_dphi = dkl_dphi + factorial_inv * contrib

            del Q0, Q_k

            # Weight by β and softmax coupling, sum over j
            grad_phi_tile = torch.einsum('bij,bija->bia', weight_tile, dkl_dphi)
            del dkl_dphi

            grad_phi[:, i_start:i_end, :] = grad_phi[:, i_start:i_end, :] + grad_phi_tile

    return grad_phi.to(dtype)
