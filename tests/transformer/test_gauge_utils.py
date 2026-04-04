"""
Tests for transformer/core/gauge_utils.py
==========================================

Tests the mathematical backbone: matrix exponentials, Newton-Schulz
orthogonalization, fused block-diagonal KL divergences, and Omega gradients.

These functions are on the critical path of every forward pass.
"""

import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_skew_symmetric(K, *batch_shape):
    """Generate random skew-symmetric matrices."""
    A = torch.randn(*batch_shape, K, K)
    return 0.5 * (A - A.transpose(-1, -2))


def _random_gl_matrix(K, *batch_shape):
    """Generate random matrices in gl(K)."""
    return torch.randn(*batch_shape, K, K) * 0.5


def _make_so3_generators_torch(K=3):
    """Small SO(3) generators for K=3."""
    from math_utils.generators import generate_so3_generators
    G = generate_so3_generators(K)
    return torch.from_numpy(G).float()


def _make_block_test_setup(B=2, N=4):
    """Create minimal block-diagonal test setup.

    Returns phi, generators, irrep_dims, and derived quantities.
    Uses irrep_spec: 2 scalar blocks (d=1) + 1 vector block (d=3) -> K=5.
    """
    irrep_dims = [1, 1, 3]
    K = sum(irrep_dims)
    n_gen = 3  # SO(3)-like

    # Build block-diagonal generators
    G = torch.zeros(n_gen, K, K)
    # scalar blocks: generators are zero (trivial representation)
    # vector block: use SO(3) generators for the 3x3 sub-block
    so3 = _make_so3_generators_torch(3)
    G[:, 2:5, 2:5] = so3

    phi = torch.randn(B, N, n_gen) * 0.3
    return phi, G, irrep_dims, B, N, K


# ===========================================================================
# TestStableMatrixExpPair
# ===========================================================================

class TestStableMatrixExpPair:
    """Tests for stable_matrix_exp_pair()."""

    def test_exp_inverse_identity(self):
        """exp(M) · exp(-M) ~= I."""
        from transformer.core.gauge_utils import stable_matrix_exp_pair
        M = _random_gl_matrix(4, 3)
        exp_pos, exp_neg = stable_matrix_exp_pair(M)
        product = exp_pos @ exp_neg
        I = torch.eye(4).expand_as(product)
        assert torch.allclose(product, I, atol=1e-4), \
            f"max deviation: {(product - I).abs().max():.2e}"

    def test_det_positive(self):
        """det(exp(M)) > 0 for any M."""
        from transformer.core.gauge_utils import stable_matrix_exp_pair
        M = _random_gl_matrix(5, 8)
        exp_pos, _ = stable_matrix_exp_pair(M)
        dets = torch.linalg.det(exp_pos)
        assert (dets > 0).all(), f"Found non-positive dets: {dets[dets <= 0]}"

    def test_det_equals_exp_trace(self):
        """det(exp(M)) ~= exp(tr(M)) -- Jacobi's formula."""
        from transformer.core.gauge_utils import stable_matrix_exp_pair
        M = _random_gl_matrix(4, 6)
        exp_pos, _ = stable_matrix_exp_pair(M)
        det_exp = torch.linalg.det(exp_pos)
        exp_trace = torch.exp(torch.diagonal(M, dim1=-2, dim2=-1).sum(-1))
        assert torch.allclose(det_exp, exp_trace, rtol=1e-3), \
            f"max relative error: {((det_exp - exp_trace) / exp_trace).abs().max():.2e}"

    def test_norm_clamping_activates(self):
        """Large ||M|| gets clamped; output is still finite."""
        from transformer.core.gauge_utils import stable_matrix_exp_pair
        M = torch.randn(3, 3) * 100.0  # way beyond max_norm=10
        exp_pos, exp_neg = stable_matrix_exp_pair(M, max_norm=10.0)
        assert torch.isfinite(exp_pos).all()
        assert torch.isfinite(exp_neg).all()

    def test_gradient_flows_through_clamping(self):
        """Gradient w.r.t. M exists even when norm is clamped."""
        from transformer.core.gauge_utils import stable_matrix_exp_pair
        M = torch.randn(3, 3) * 20.0
        M = M.detach().requires_grad_(True)
        exp_pos, _ = stable_matrix_exp_pair(M, max_norm=10.0)
        loss = exp_pos.sum()
        loss.backward()
        assert M.grad is not None
        assert torch.isfinite(M.grad).all()

    def test_skew_symmetric_gives_orthogonal(self):
        """exp(skew-symmetric) ∈ SO(K): R^T R = I."""
        from transformer.core.gauge_utils import stable_matrix_exp_pair
        S = _random_skew_symmetric(5, 4)
        exp_pos, _ = stable_matrix_exp_pair(S)
        RtR = exp_pos.transpose(-1, -2) @ exp_pos
        I = torch.eye(5).expand_as(RtR)
        assert torch.allclose(RtR, I, atol=1e-4), \
            f"max deviation from orthogonality: {(RtR - I).abs().max():.2e}"

    def test_output_dtype_preserved(self):
        """Output dtype matches input dtype."""
        from transformer.core.gauge_utils import stable_matrix_exp_pair
        M = torch.randn(3, 3, dtype=torch.float32)
        exp_pos, exp_neg = stable_matrix_exp_pair(M)
        assert exp_pos.dtype == torch.float32
        assert exp_neg.dtype == torch.float32


# ===========================================================================
# TestNewtonSchulzOrthogonalize
# ===========================================================================

class TestNewtonSchulzOrthogonalize:
    """Tests for newton_schulz_orthogonalize()."""

    def test_output_orthogonal(self):
        """X^T X ~= I after projection."""
        from transformer.core.gauge_utils import newton_schulz_orthogonalize
        # Start close to orthogonal (small perturbation from I)
        X = torch.eye(4) + 0.1 * torch.randn(4, 4)
        Q = newton_schulz_orthogonalize(X, n_iters=10)
        I = torch.eye(4)
        assert torch.allclose(Q.T @ Q, I, atol=1e-3)

    def test_identity_input_unchanged(self):
        """NS(I) = I."""
        from transformer.core.gauge_utils import newton_schulz_orthogonalize
        I = torch.eye(5)
        Q = newton_schulz_orthogonalize(I)
        assert torch.allclose(Q, I, atol=1e-6)

    def test_singular_values_near_one(self):
        """All singular values ∈ [1-ε, 1+ε] after projection."""
        from transformer.core.gauge_utils import newton_schulz_orthogonalize
        # Start close to orthogonal so NS converges within its iteration budget
        X = torch.eye(6) + 0.1 * torch.randn(6, 6)
        Q = newton_schulz_orthogonalize(X, n_iters=15)
        svs = torch.linalg.svdvals(Q)
        assert (svs > 0.95).all() and (svs < 1.05).all(), \
            f"Singular values: {svs}"

    def test_det_unit_magnitude(self):
        """|det(NS(X))| ~= 1 -- projects to O(K)."""
        from transformer.core.gauge_utils import newton_schulz_orthogonalize
        # Start with det > 0 input close to identity
        X = torch.eye(4) + 0.1 * torch.randn(4, 4)
        Q = newton_schulz_orthogonalize(X, n_iters=10)
        assert torch.allclose(torch.linalg.det(Q).abs(), torch.tensor(1.0), atol=0.05)

    def test_batched_orthogonalization(self):
        """Batched input (..., K, K) works correctly."""
        from transformer.core.gauge_utils import newton_schulz_orthogonalize
        X = torch.eye(4).unsqueeze(0).expand(3, -1, -1) + 0.1 * torch.randn(3, 4, 4)
        Q = newton_schulz_orthogonalize(X, n_iters=10)
        I = torch.eye(4)
        for i in range(3):
            assert torch.allclose(Q[i].T @ Q[i], I, atol=1e-2), \
                f"Batch {i}: deviation {(Q[i].T @ Q[i] - I).abs().max():.4e}"


# ===========================================================================
# TestFusedBlockMatrixExpPairs
# ===========================================================================

class TestFusedBlockMatrixExpPairs:
    """Tests for fused_block_matrix_exp_pairs()."""

    def test_output_shapes(self):
        """Each block pair has shape (B, N, d, d)."""
        from transformer.core.gauge_utils import fused_block_matrix_exp_pairs
        phi, G, irrep_dims, B, N, K = _make_block_test_setup()
        results = fused_block_matrix_exp_pairs(phi, G, irrep_dims)
        assert len(results) == len(irrep_dims)
        for idx, d in enumerate(irrep_dims):
            exp_pos, exp_neg = results[idx]
            assert exp_pos.shape == (B, N, d, d)
            assert exp_neg.shape == (B, N, d, d)

    def test_scalar_blocks_are_scalar_exp(self):
        """For d=1 blocks, exp should be scalar exponentials."""
        from transformer.core.gauge_utils import fused_block_matrix_exp_pairs
        phi, G, irrep_dims, B, N, K = _make_block_test_setup()
        results = fused_block_matrix_exp_pairs(phi, G, irrep_dims)
        # Scalar blocks (d=1) with zero generators -> exp(0) = 1
        for idx, d in enumerate(irrep_dims):
            if d == 1:
                exp_pos, exp_neg = results[idx]
                # Generators are zero for scalar blocks -> identity
                assert torch.allclose(exp_pos, torch.ones_like(exp_pos), atol=1e-4)

    def test_inverse_property(self):
        """exp_pos · exp_neg ~= I per block."""
        from transformer.core.gauge_utils import fused_block_matrix_exp_pairs
        phi, G, irrep_dims, B, N, K = _make_block_test_setup()
        results = fused_block_matrix_exp_pairs(phi, G, irrep_dims)
        for idx, d in enumerate(irrep_dims):
            exp_pos, exp_neg = results[idx]
            product = exp_pos @ exp_neg
            I = torch.eye(d).expand_as(product)
            assert torch.allclose(product, I, atol=1e-4), \
                f"Block {idx} (d={d}): max deviation {(product - I).abs().max():.2e}"

    def test_all_results_finite(self):
        """No NaN/Inf in any block output."""
        from transformer.core.gauge_utils import fused_block_matrix_exp_pairs
        phi, G, irrep_dims, B, N, K = _make_block_test_setup()
        results = fused_block_matrix_exp_pairs(phi, G, irrep_dims)
        for idx, (exp_pos, exp_neg) in enumerate(results):
            assert torch.isfinite(exp_pos).all(), f"Block {idx} exp_pos has non-finite"
            assert torch.isfinite(exp_neg).all(), f"Block {idx} exp_neg has non-finite"


# ===========================================================================
# TestFusedBlockDiagonalKL
# ===========================================================================

class TestFusedBlockDiagonalKLDiag:
    """Tests for fused_block_diagonal_kl_diag()."""

    def _setup(self, B=2, N=4):
        from transformer.core.gauge_utils import (
            fused_block_matrix_exp_pairs, fused_block_diagonal_kl_diag,
        )
        phi, G, irrep_dims, B, N, K = _make_block_test_setup(B, N)
        block_exp_pairs = fused_block_matrix_exp_pairs(phi, G, irrep_dims)
        mu_q = torch.randn(B, N, K)
        sigma_q = torch.rand(B, N, K).clamp(min=0.1)
        return mu_q, sigma_q, block_exp_pairs, irrep_dims, B, N, K

    def test_kl_non_negative(self):
        """KL(q||p) ≥ 0 for all pairs."""
        from transformer.core.gauge_utils import fused_block_diagonal_kl_diag
        mu_q, sigma_q, bep, irrep_dims, B, N, K = self._setup()
        kl = fused_block_diagonal_kl_diag(mu_q, sigma_q, bep, irrep_dims)
        assert (kl >= -1e-5).all(), f"Found negative KL: {kl.min():.4e}"

    def test_kl_zero_self(self):
        """KL(q_i || Ω_ii q_i) ~= 0 on diagonal (identity transport)."""
        from transformer.core.gauge_utils import (
            fused_block_matrix_exp_pairs, fused_block_diagonal_kl_diag,
        )
        irrep_dims = [1, 1, 3]
        K = sum(irrep_dims)
        B, N = 2, 4
        n_gen = 3
        G = torch.zeros(n_gen, K, K)
        so3 = _make_so3_generators_torch(3)
        G[:, 2:5, 2:5] = so3
        # Use SAME phi for all positions -> Ω_ii = I
        phi_single = torch.randn(1, 1, n_gen) * 0.3
        phi = phi_single.expand(B, N, n_gen).contiguous()
        bep = fused_block_matrix_exp_pairs(phi, G, irrep_dims)
        mu_q = torch.randn(B, N, K)
        sigma_q = torch.rand(B, N, K).clamp(min=0.1)
        kl = fused_block_diagonal_kl_diag(mu_q, sigma_q, bep, irrep_dims)
        # Diagonal should be zero (self-transport)
        diag_kl = torch.diagonal(kl, dim1=-2, dim2=-1)
        assert torch.allclose(diag_kl, torch.zeros_like(diag_kl), atol=1e-3), \
            f"Diagonal KL not zero: max={diag_kl.abs().max():.4e}"

    def test_kl_finite(self):
        """No NaN/Inf in KL output."""
        from transformer.core.gauge_utils import fused_block_diagonal_kl_diag
        mu_q, sigma_q, bep, irrep_dims, B, N, K = self._setup()
        kl = fused_block_diagonal_kl_diag(mu_q, sigma_q, bep, irrep_dims)
        assert torch.isfinite(kl).all()

    def test_kl_shape(self):
        """Output shape is (B, N, N)."""
        from transformer.core.gauge_utils import fused_block_diagonal_kl_diag
        mu_q, sigma_q, bep, irrep_dims, B, N, K = self._setup()
        kl = fused_block_diagonal_kl_diag(mu_q, sigma_q, bep, irrep_dims)
        assert kl.shape == (B, N, N)

    def test_gradient_finite(self):
        """Backward through KL produces finite gradients."""
        from transformer.core.gauge_utils import (
            fused_block_matrix_exp_pairs, fused_block_diagonal_kl_diag,
        )
        phi, G, irrep_dims, B, N, K = _make_block_test_setup()
        phi = phi.clone().requires_grad_(True)
        bep = fused_block_matrix_exp_pairs(phi, G, irrep_dims)
        mu_q = torch.randn(B, N, K, requires_grad=True)
        sigma_q = torch.rand(B, N, K).clamp(min=0.1).requires_grad_(True)
        kl = fused_block_diagonal_kl_diag(mu_q, sigma_q, bep, irrep_dims)
        kl.sum().backward()
        assert phi.grad is not None and torch.isfinite(phi.grad).all()
        assert mu_q.grad is not None and torch.isfinite(mu_q.grad).all()


class TestFusedBlockDiagonalKLFull:
    """Tests for fused_block_diagonal_kl_full()."""

    def test_kl_non_negative(self):
        """KL ≥ 0 for full covariance mode."""
        from transformer.core.gauge_utils import (
            fused_block_matrix_exp_pairs, fused_block_diagonal_kl_full,
        )
        irrep_dims = [1, 3]
        K = sum(irrep_dims)
        B, N = 2, 4
        n_gen = 3
        G = torch.zeros(n_gen, K, K)
        so3 = _make_so3_generators_torch(3)
        G[:, 1:4, 1:4] = so3
        phi = torch.randn(B, N, n_gen) * 0.3
        bep = fused_block_matrix_exp_pairs(phi, G, irrep_dims)
        mu_q = torch.randn(B, N, K)
        # Full covariance: (B, N, K, K) block-diagonal SPD
        sigma_q = torch.zeros(B, N, K, K)
        for b in range(B):
            for n in range(N):
                A = torch.randn(K, K) * 0.3
                sigma_q[b, n] = A @ A.T + 0.5 * torch.eye(K)
        kl = fused_block_diagonal_kl_full(mu_q, sigma_q, bep, irrep_dims)
        assert (kl >= -1e-3).all(), f"Found negative KL: {kl.min():.4e}"

    def test_kl_finite(self):
        """No NaN/Inf in full-cov KL output."""
        from transformer.core.gauge_utils import (
            fused_block_matrix_exp_pairs, fused_block_diagonal_kl_full,
        )
        irrep_dims = [1, 3]
        K = sum(irrep_dims)
        B, N = 2, 3
        n_gen = 3
        G = torch.zeros(n_gen, K, K)
        so3 = _make_so3_generators_torch(3)
        G[:, 1:4, 1:4] = so3
        phi = torch.randn(B, N, n_gen) * 0.3
        bep = fused_block_matrix_exp_pairs(phi, G, irrep_dims)
        mu_q = torch.randn(B, N, K)
        sigma_q = torch.zeros(B, N, K, K)
        for b in range(B):
            for n in range(N):
                A = torch.randn(K, K) * 0.3
                sigma_q[b, n] = A @ A.T + 0.5 * torch.eye(K)
        kl = fused_block_diagonal_kl_full(mu_q, sigma_q, bep, irrep_dims)
        assert torch.isfinite(kl).all()


# ===========================================================================
# TestDklDomegaDiag
# ===========================================================================

class TestDklDomegaDiag:
    """Tests for _compute_dkl_domega_diag()."""

    def test_gradient_matches_autograd(self):
        """Manual ∂KL/∂Ω ~= autograd gradient."""
        from transformer.core.gauge_utils import _compute_dkl_domega_diag
        d = 3
        torch.manual_seed(42)
        mu_i = torch.randn(d)
        sig_i = torch.rand(d).clamp(min=0.1)
        mu_j = torch.randn(d)
        sig_j = torch.rand(d).clamp(min=0.1)
        Omega = (torch.eye(d) + 0.1 * torch.randn(d, d)).requires_grad_(True)

        # Compute transported quantities
        mu_t = Omega @ mu_j
        sig_t = (Omega ** 2) @ sig_j  # diagonal approximation

        # Manual gradient
        grad_manual = _compute_dkl_domega_diag(
            mu_i, sig_i, mu_t.detach(), sig_t.detach().clamp(min=1e-6),
            mu_j, sig_j, Omega.detach(),
        )

        # Autograd gradient
        sig_t_ag = ((Omega ** 2) @ sig_j).clamp(min=1e-6)
        mu_t_ag = Omega @ mu_j
        delta = mu_i - mu_t_ag
        kl = 0.5 * (sig_i / sig_t_ag + delta ** 2 / sig_t_ag - 1.0
                     + torch.log(sig_t_ag) - torch.log(sig_i)).sum()
        kl.backward()
        grad_autograd = Omega.grad

        assert torch.allclose(grad_manual, grad_autograd, atol=1e-3), \
            f"max diff: {(grad_manual - grad_autograd).abs().max():.2e}"
