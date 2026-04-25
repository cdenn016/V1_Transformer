"""
Regression tests for the `use_output_projection=False` full-covariance path.

Context: with `diagonal_covariance=False`, `em_mode='em_phi_p'`, and
`use_prior_bank=True`, flipping `use_output_projection` from True → False
exposes a latent numerical failure. The attention-KL path
(`fused_block_diagonal_kl_full` in `gauge_utils.py`) Cholesky-decomposes
`sigma_i_reg = sigma_q + eps * I_d` and `sigma_t_reg = Omega_ij @ sigma_q @
Omega_ij^T + eps * I_d`. Without W_O mixing, the block-diagonal Σ drifts into
a regime where per-block eigenvalues fall below `eps=1e-6`, Cholesky fails,
the diagonal-fallback counter `gauge_kl_full_chol_fallback_diag` climbs, and
the `1/sigma_diag` gradient through that branch blows up over ~150 steps
(observed: steps 400 → 552 → NaN).

Fix (B1): thread `sigma_floor` through `fused_block_diagonal_kl_full` and
replace the `+ eps * I_d` jitter with `spd_eigfloor(Σ, floor=sigma_floor)`
so the condition number is bounded by `sigma_max² / sigma_floor` and
Cholesky is guaranteed to succeed.

These tests verify:
    T1. Without sigma_floor, the Cholesky fallback path fires on a
        deliberately collapsed block-diagonal Σ.
    T2. With sigma_floor set, the primary Cholesky path succeeds and
        gradients stay finite.
    T3. The plumbing from `compute_kl_matrix` → `fused_block_diagonal_kl_full`
        passes sigma_floor correctly.
    T4. The plumbing from `compute_attention_weights` → `compute_kl_matrix` →
        `fused_block_diagonal_kl_full` passes sigma_floor correctly in
        BLOCK_DIAGONAL mode (both diagonal_covariance=False branches at
        attention.py:610 and 701).
"""

from __future__ import annotations

import torch

from math_utils import numerical_monitor
from transformer.core.gauge_utils import (
    fused_block_diagonal_kl_full,
    fused_block_matrix_exp_pairs,
)


def _counter(name: str) -> int:
    """Read the current value of a numerical_monitor counter without resetting."""
    return numerical_monitor._counts.get(name, 0)


# =============================================================================
# Helpers
# =============================================================================

def _collapsed_block_sigma(B: int, N: int, irrep_dims: list[int],
                           collapse_block: int = 0, collapse_eig: float = 1e-10,
                           dtype=torch.float32) -> torch.Tensor:
    """Build block-diagonal Σ where one head-block has a near-zero eigenvalue.

    Mimics the end-state of a ~500-step run with `use_output_projection=False`
    where a particular head's belief variance collapses in one direction.
    """
    K = sum(irrep_dims)
    sigma = torch.zeros(B, N, K, K, dtype=dtype)
    start = 0
    for idx, d in enumerate(irrep_dims):
        if idx == collapse_block:
            eigs = torch.tensor([collapse_eig] + [1.0] * (d - 1), dtype=dtype)
            Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=dtype))
            block = Q @ torch.diag(eigs) @ Q.t()
        else:
            A = torch.randn(d, d, dtype=dtype) * 0.3
            block = A @ A.t() + 0.1 * torch.eye(d, dtype=dtype)
        sigma[..., start:start + d, start:start + d] = block
        start += d
    return sigma


def _block_exp_pairs(phi: torch.Tensor, generators: torch.Tensor,
                     irrep_dims: list[int]) -> list:
    """Build `block_exp_pairs` for `fused_block_diagonal_kl_full` via the
    same fused path attention.py uses. Returns list of
    (exp_phi_block, exp_neg_phi_block) per block."""
    return fused_block_matrix_exp_pairs(phi, generators, irrep_dims)


# =============================================================================
# T1: Without sigma_floor, Cholesky fallback fires on collapsed block Σ.
# =============================================================================

def test_fused_kl_full_cholesky_fallback_without_floor():
    """Precondition: the bug reproduces — with default eps=1e-6 and a
    deeply collapsed block-diagonal Σ transported by a high-norm Ω, the
    Cholesky fails and falls back to diagonal. This mirrors the mechanism
    observed in the user's step-400→552 run: `phi_trace_clamp=0.75` permits
    ||phi||_F in the 0.5-1.5 range, which under `exp(phi·G)` yields Ω with
    eigenvalue spread of several orders of magnitude, amplifying a
    small-eigenvalue block-diagonal Σ_q into a Cholesky-failing Σ_t."""
    torch.manual_seed(3)
    B, N = 1, 4
    irrep_dims = [10, 10]

    K = sum(irrep_dims)
    mu_q = torch.randn(B, N, K, dtype=torch.float32)
    # Block 0 has eigenvalue 0 (exactly singular after float32 roundoff).
    sigma_q = _collapsed_block_sigma(B, N, irrep_dims, collapse_eig=0.0)

    # Inflate phi well above the trust region so exp(phi·G) is poorly
    # conditioned, amplifying the collapsed eigenvalue under sandwich.
    n_gen = K * K
    phi = torch.randn(B, N, n_gen, dtype=torch.float32) * 2.0
    generators = torch.randn(n_gen, K, K, dtype=torch.float32) * 0.5
    pairs = _block_exp_pairs(phi, generators, irrep_dims)

    before = _counter("gauge_kl_full_chol_fallback_diag")
    kl = fused_block_diagonal_kl_full(
        mu_q, sigma_q, pairs, irrep_dims, eps=1e-6, sigma_floor=None,
    )
    after = _counter("gauge_kl_full_chol_fallback_diag")

    assert after > before, (
        f"Expected Cholesky fallback to fire on collapsed block Σ "
        f"under high-norm Ω; counter went {before} → {after}. "
        f"If this test passes, the precondition for the B1 fix may no "
        f"longer hold — re-check that eps=1e-6 jitter is still the only "
        f"regularization in the non-sigma_floor branch."
    )
    assert torch.isfinite(kl).all(), "Fallback KL is not finite"


# =============================================================================
# T2: With sigma_floor set, the primary Cholesky path succeeds.
# =============================================================================

def test_fused_kl_full_eigfloor_prevents_fallback():
    """With `sigma_floor=0.01`, the spectral clamp upstream of Cholesky
    guarantees λ_min ≥ 0.01, so the Cholesky never fails and the counter
    does not increment."""
    torch.manual_seed(0)
    B, N = 1, 4
    irrep_dims = [10, 10]

    mu_q = torch.randn(B, N, sum(irrep_dims), dtype=torch.float32, requires_grad=True)
    sigma_q = _collapsed_block_sigma(B, N, irrep_dims, collapse_eig=1e-14)

    K = sum(irrep_dims)
    n_gen = K * K
    phi = torch.randn(B, N, n_gen, dtype=torch.float32) * 0.05
    # Build block-diagonal generators: n_gen basis matrices of shape (K, K)
    # that slice cleanly into irrep_dims blocks under generators[:, s:e, s:e].
    generators = torch.randn(n_gen, K, K, dtype=torch.float32) * 0.3

    pairs = _block_exp_pairs(phi, generators, irrep_dims)

    before = _counter("gauge_kl_full_chol_fallback_diag")
    kl = fused_block_diagonal_kl_full(
        mu_q, sigma_q, pairs, irrep_dims, eps=1e-6, sigma_floor=0.01,
    )
    after = _counter("gauge_kl_full_chol_fallback_diag")

    assert after == before, (
        f"Cholesky fallback fired even with sigma_floor=0.01; "
        f"counter {before} → {after}. spd_eigfloor should have prevented this."
    )
    assert torch.isfinite(kl).all(), "KL non-finite with sigma_floor set"

    loss = kl.sum()
    loss.backward()
    assert torch.isfinite(mu_q.grad).all(), "mu_q gradient non-finite with sigma_floor"


# =============================================================================
# T3: Plumbing through compute_kl_matrix (BLOCK_DIAGONAL + full sigma).
# =============================================================================

def test_compute_kl_matrix_passes_sigma_floor_to_block_full():
    """Regression for the kl_computation.py:513 call site: sigma_floor must
    reach fused_block_diagonal_kl_full. Under a collapsed Σ, presence of the
    floor is observable via the absence of the Cholesky fallback counter
    increment."""
    from transformer.core.kl_computation import compute_kl_matrix, KLMode

    torch.manual_seed(1)
    B, N = 1, 4
    irrep_dims = [10, 10]

    K = sum(irrep_dims)
    mu_q = torch.randn(B, N, K, dtype=torch.float32)
    sigma_q = _collapsed_block_sigma(B, N, irrep_dims, collapse_eig=1e-14)

    n_gen = K * K
    phi = torch.randn(B, N, n_gen, dtype=torch.float32) * 0.05
    generators = torch.randn(n_gen, K, K, dtype=torch.float32) * 0.3
    pairs = _block_exp_pairs(phi, generators, irrep_dims)

    stats_before = _counter("gauge_kl_full_chol_fallback_diag")
    kl = compute_kl_matrix(
        mu_q=mu_q, sigma_q=sigma_q,
        mu_transported=None, sigma_transported=None,
        mode=KLMode.BLOCK_DIAGONAL,
        block_exp_pairs=pairs, irrep_dims=irrep_dims,
        sigma_floor=0.01,
    )
    stats_after = _counter("gauge_kl_full_chol_fallback_diag")

    assert stats_after == stats_before, (
        "compute_kl_matrix(mode=BLOCK_DIAGONAL, full Σ) did not forward "
        "sigma_floor to the fused kernel"
    )
    assert torch.isfinite(kl).all()


# =============================================================================
# T4: End-to-end through compute_attention_weights.
# =============================================================================

def test_compute_attention_weights_block_full_forwards_sigma_floor():
    """Mimics the attention call issued by variational_ffn.py:1207-1211
    where sigma_floor=self.e_step_sigma_floor is passed. Verifies that
    under an ill-conditioned Σ_q, the fallback counter does not fire."""
    from transformer.core.attention import compute_attention_weights

    torch.manual_seed(2)
    B, N = 1, 4
    irrep_dims = [10, 10]
    K = sum(irrep_dims)

    mu_q = torch.randn(B, N, K, dtype=torch.float32)
    sigma_q = _collapsed_block_sigma(B, N, irrep_dims, collapse_eig=1e-14)

    # GL(K) generator basis (n_gen = K*K). The fused exp-pair kernel slices
    # generators[:, s:e, s:e] per head-block, so random (n_gen, K, K)
    # generators yield non-trivial per-head transports.
    n_gen = K * K
    phi = torch.randn(B, N, n_gen, dtype=torch.float32) * 0.05
    generators = torch.randn(n_gen, K, K, dtype=torch.float32) * 0.3

    stats_before = _counter("gauge_kl_full_chol_fallback_diag")
    beta = compute_attention_weights(
        mu_q=mu_q, sigma_q=sigma_q, phi=phi, generators=generators,
        kappa=1.0, diagonal_covariance=False, irrep_dims=irrep_dims,
        sigma_floor=0.01, spd_floor_mode='eigclamp',
    )
    stats_after = _counter("gauge_kl_full_chol_fallback_diag")

    assert stats_after == stats_before, (
        "compute_attention_weights BLOCK_DIAGONAL full-sigma path did not "
        "forward sigma_floor through compute_kl_matrix"
    )
    assert torch.isfinite(beta).all()
