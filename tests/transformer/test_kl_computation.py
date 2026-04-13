"""
Tests for transformer.core.kl_computation
==========================================

Validates the unified KL divergence computation module: the three kernel
functions (diagonal, dense, block-diagonal), the safe_kl_clamp utility,
and the compute_kl_matrix entry point with chunked/unchunked paths.

Mathematical invariants tested:
    - KL(q || p) >= 0 for all valid Gaussians
    - KL(q || q) = 0
    - KL is asymmetric in general
    - Diagonal kernel matches closed-form for K=1
    - Dense kernel matches diagonal when Sigma = diag(sigma)
    - Chunked and unchunked produce identical results
"""

import pytest
import torch

from transformer.core.kl_computation import (
    KLMode,
    compute_kl_matrix,
    safe_kl_clamp,
    _kl_kernel_diagonal,
    _kl_kernel_dense,
)


# =============================================================================
# Helpers
# =============================================================================

def _random_diagonal_beliefs(B, N, K, device='cpu'):
    """Create (mu, sigma_diag) with sigma > 0."""
    mu = torch.randn(B, N, K, device=device)
    sigma = torch.rand(B, N, K, device=device).clamp(min=0.1) + 0.1
    return mu, sigma


def _random_dense_beliefs(B, N, K, device='cpu'):
    """Create (mu, Sigma_full) with Sigma SPD."""
    mu = torch.randn(B, N, K, device=device)
    A = torch.randn(B, N, K, K, device=device) * 0.3
    Sigma = A @ A.transpose(-1, -2) + 0.1 * torch.eye(K, device=device)
    return mu, Sigma


def _transport_identity(mu, sigma):
    """Build transported tensors for identity transport (Omega=I).

    For diagonal: mu_t has shape (B, N, N, K), sigma_t has shape (B, N, N, K).
    For dense:    mu_t has shape (B, N, N, K), sigma_t has shape (B, N, N, K, K).
    """
    B, N = mu.shape[:2]
    mu_t = mu[:, None, :, :].expand(B, N, N, -1).contiguous()  # (B, N, N, K)
    if sigma.dim() == 3:
        sigma_t = sigma[:, None, :, :].expand(B, N, N, -1).contiguous()
    else:
        sigma_t = sigma[:, None, :, :, :].expand(B, N, N, -1, -1).contiguous()
    return mu_t, sigma_t


# =============================================================================
# TestKLMode
# =============================================================================

class TestKLMode:
    """Enum membership and string values for KLMode."""

    def test_enum_values(self):
        """KLMode has three members with expected string values."""
        assert KLMode.DENSE.value == "dense"
        assert KLMode.DIAGONAL.value == "diagonal"
        assert KLMode.BLOCK_DIAGONAL.value == "block_diagonal"

    def test_enum_count(self):
        """Exactly three modes."""
        assert len(KLMode) == 3


# =============================================================================
# TestSafeKlClamp
# =============================================================================

class TestSafeKlClamp:
    """safe_kl_clamp: clamp to [0, kl_max] with NaN/Inf replacement."""

    def test_non_negativity(self):
        """Output is always >= 0."""
        kl = torch.tensor([-5.0, -0.1, 0.0, 1.0, 50.0])
        result = safe_kl_clamp(kl)
        assert (result >= 0).all(), f"Negative values found: {result[result < 0]}"

    def test_upper_bound_default(self):
        """Output never exceeds default kl_max=100."""
        kl = torch.tensor([0.0, 50.0, 99.9, 100.0, 100.1, 1000.0])
        result = safe_kl_clamp(kl)
        assert (result <= 100.0).all(), f"Values above 100: {result[result > 100]}"

    def test_nan_replacement(self):
        """NaN values are replaced with kl_max (repulsive)."""
        kl = torch.tensor([1.0, float('nan'), 3.0])
        result = safe_kl_clamp(kl, kl_max=50.0)
        assert torch.isfinite(result).all(), "NaN not replaced"
        assert result[1].item() == 50.0, f"NaN should become kl_max=50, got {result[1]}"

    def test_posinf_replacement(self):
        """+inf replaced with kl_max."""
        kl = torch.tensor([1.0, float('inf')])
        result = safe_kl_clamp(kl, kl_max=42.0)
        assert result[1].item() == 42.0

    def test_neginf_replacement(self):
        """-inf replaced with 0."""
        kl = torch.tensor([1.0, float('-inf')])
        result = safe_kl_clamp(kl)
        assert result[1].item() == 0.0

    def test_passthrough_valid(self):
        """Values already in [0, kl_max] pass through unchanged."""
        kl = torch.tensor([0.0, 5.0, 50.0, 99.9])
        result = safe_kl_clamp(kl, kl_max=100.0)
        assert torch.allclose(result, kl, atol=1e-6)

    def test_custom_kl_max(self):
        """Custom kl_max is respected."""
        kl = torch.tensor([0.0, 5.0, 10.0, 20.0])
        result = safe_kl_clamp(kl, kl_max=10.0)
        assert (result <= 10.0).all()
        assert result[3].item() == 10.0


# =============================================================================
# TestKLKernelDiagonal
# =============================================================================

class TestKLKernelDiagonal:
    r"""Diagonal-covariance KL kernel: KL(N(mu_q, diag(sigma_q)) || N(mu_t, diag(sigma_t))).

    Tests the closed-form O(K) diagonal kernel for mathematical correctness.
    """

    def test_non_negativity(self, cpu_device):
        """KL(q || p) >= 0 for random diagonal Gaussians."""
        B, N, K = 2, 4, 6
        mu_q, sigma_q = _random_diagonal_beliefs(B, N, K, cpu_device)
        mu_t, sigma_t = _random_diagonal_beliefs(B, N, K, cpu_device)
        kl = _kl_kernel_diagonal(mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-6)
        assert (kl >= -1e-6).all(), f"Negative KL: min={kl.min():.2e}"

    def test_zero_for_identical(self, cpu_device):
        """KL(q || q) = 0 when distributions are identical."""
        B, N, K = 2, 4, 6
        mu, sigma = _random_diagonal_beliefs(B, N, K, cpu_device)
        kl = _kl_kernel_diagonal(mu, sigma, mu, sigma, kl_max=100.0, eps=1e-6)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-4), \
            f"Self-KL should be 0, max={kl.abs().max():.2e}"

    def test_asymmetry(self, cpu_device):
        """KL(q || p) != KL(p || q) in general."""
        B, N, K = 1, 2, 4
        mu_q, sigma_q = _random_diagonal_beliefs(B, N, K, cpu_device)
        mu_t, sigma_t = _random_diagonal_beliefs(B, N, K, cpu_device)
        kl_forward = _kl_kernel_diagonal(mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-6)
        kl_reverse = _kl_kernel_diagonal(mu_t, sigma_t, mu_q, sigma_q, kl_max=100.0, eps=1e-6)
        # With random inputs, exact equality would be astronomical coincidence
        assert not torch.allclose(kl_forward, kl_reverse, atol=1e-4), \
            "KL should be asymmetric for different distributions"

    def test_closed_form_k1(self, cpu_device):
        r"""For K=1, verify against hand-computed KL.

        KL(N(mu_q, s_q) || N(mu_t, s_t)) = 0.5 * (s_q/s_t + (mu_t-mu_q)^2/s_t - 1 + ln(s_t/s_q))
        """
        mu_q = torch.tensor([[[2.0]]])   # (1,1,1)
        sigma_q = torch.tensor([[[1.0]]])
        mu_t = torch.tensor([[[0.0]]])
        sigma_t = torch.tensor([[[2.0]]])
        # Expected: 0.5 * (1/2 + 4/2 - 1 + ln(2/1)) = 0.5 * (0.5 + 2 - 1 + 0.6931) = 0.5 * 2.1931 = 1.0966
        expected = 0.5 * (1.0/2.0 + 4.0/2.0 - 1.0 + torch.log(torch.tensor(2.0/1.0)))
        kl = _kl_kernel_diagonal(mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-8)
        assert torch.allclose(kl.squeeze(), expected, atol=1e-4), \
            f"Expected {expected:.4f}, got {kl.squeeze():.4f}"

    def test_batch_shape(self, cpu_device):
        """Output shape is (...) when input is (..., K)."""
        B, N, K = 3, 5, 8
        mu_q, sigma_q = _random_diagonal_beliefs(B, N, K, cpu_device)
        mu_t, sigma_t = _random_diagonal_beliefs(B, N, K, cpu_device)
        kl = _kl_kernel_diagonal(mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-6)
        assert kl.shape == (B, N), f"Expected ({B}, {N}), got {kl.shape}"

    def test_output_finite(self, cpu_device):
        """No NaN or Inf in output for random inputs."""
        B, N, K = 2, 4, 6
        mu_q, sigma_q = _random_diagonal_beliefs(B, N, K, cpu_device)
        mu_t, sigma_t = _random_diagonal_beliefs(B, N, K, cpu_device)
        kl = _kl_kernel_diagonal(mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-6)
        assert torch.isfinite(kl).all(), "Output contains NaN or Inf"

    def test_large_variance_stability(self, cpu_device):
        """No NaN when sigma spans several orders of magnitude."""
        mu_q = torch.zeros(1, 1, 4)
        sigma_q = torch.tensor([[[1e-3, 1e-1, 1e1, 1e3]]])
        mu_t = torch.zeros(1, 1, 4)
        sigma_t = torch.tensor([[[1e-1, 1e0, 1e2, 1e4]]])
        kl = _kl_kernel_diagonal(mu_q, sigma_q, mu_t, sigma_t, kl_max=200.0, eps=1e-8)
        assert torch.isfinite(kl).all(), f"NaN with large variance range: {kl}"

    def test_gradient_flow(self, cpu_device):
        """Autograd works through the kernel."""
        B, N, K = 1, 2, 4
        mu_q = torch.randn(B, N, K, requires_grad=True)
        sigma_q = torch.rand(B, N, K).clamp(min=0.2) + 0.1
        sigma_q = sigma_q.requires_grad_(True)
        mu_t = torch.randn(B, N, K)
        sigma_t = torch.rand(B, N, K).clamp(min=0.2) + 0.1
        kl = _kl_kernel_diagonal(mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-6)
        loss = kl.sum()
        loss.backward()
        assert mu_q.grad is not None, "No gradient for mu_q"
        assert sigma_q.grad is not None, "No gradient for sigma_q"
        assert torch.isfinite(mu_q.grad).all(), "NaN in mu_q gradient"
        assert torch.isfinite(sigma_q.grad).all(), "NaN in sigma_q gradient"


# =============================================================================
# TestKLKernelDense
# =============================================================================

class TestKLKernelDense:
    r"""Full-covariance KL kernel with Cholesky-based computation.

    Tests the dense kernel including numerical stability via progressive
    regularization and Cholesky fallbacks.
    """

    def test_non_negativity(self, cpu_device):
        """KL(q || p) >= 0 for random SPD covariances."""
        B, N, K = 2, 3, 4
        mu_q, sigma_q = _random_dense_beliefs(B, N, K, cpu_device)
        mu_t, sigma_t = _random_dense_beliefs(B, N, K, cpu_device)
        kl = _kl_kernel_dense(mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-6)
        assert (kl >= -1e-4).all(), f"Negative KL: min={kl.min():.2e}"

    def test_zero_for_identical(self, cpu_device):
        """KL(q || q) = 0 when distributions are identical."""
        B, N, K = 2, 3, 4
        mu, Sigma = _random_dense_beliefs(B, N, K, cpu_device)
        kl = _kl_kernel_dense(mu, Sigma, mu, Sigma, kl_max=100.0, eps=1e-6)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-3), \
            f"Self-KL should be 0, max={kl.abs().max():.2e}"

    def test_consistency_with_diagonal(self, cpu_device):
        r"""Full Sigma = diag(sigma) gives same KL as diagonal kernel.

        This validates that both kernels implement the same mathematical
        formula, just with different computational paths.
        """
        B, N, K = 2, 3, 5
        mu_q, sigma_diag_q = _random_diagonal_beliefs(B, N, K, cpu_device)
        mu_t, sigma_diag_t = _random_diagonal_beliefs(B, N, K, cpu_device)

        # Diagonal kernel
        kl_diag = _kl_kernel_diagonal(mu_q, sigma_diag_q, mu_t, sigma_diag_t,
                                       kl_max=100.0, eps=1e-6)

        # Dense kernel with diag_embed
        Sigma_q = torch.diag_embed(sigma_diag_q)
        Sigma_t = torch.diag_embed(sigma_diag_t)
        kl_dense = _kl_kernel_dense(mu_q, Sigma_q, mu_t, Sigma_t,
                                     kl_max=100.0, eps=1e-6)

        assert torch.allclose(kl_diag, kl_dense, atol=1e-3), \
            f"max deviation: {(kl_diag - kl_dense).abs().max():.2e}"

    def test_known_k2(self, cpu_device):
        r"""Manually computed KL for K=2 with known matrices.

        q = N([0,0], I), p = N([1,0], 2I)
        KL = 0.5 * (tr(0.5*I) + 0.5*1 - 2 + ln(4)) = 0.5*(1 + 0.5 - 2 + 1.386) = 0.443
        """
        mu_q = torch.zeros(1, 1, 2)
        Sigma_q = torch.eye(2).unsqueeze(0).unsqueeze(0)
        mu_t = torch.tensor([[[1.0, 0.0]]])
        Sigma_t = 2.0 * torch.eye(2).unsqueeze(0).unsqueeze(0)

        expected = 0.5 * (0.5 + 0.5 + 0.5 - 2 + torch.log(torch.tensor(4.0)))
        kl = _kl_kernel_dense(mu_q, Sigma_q, mu_t, Sigma_t, kl_max=100.0, eps=1e-8)
        assert torch.allclose(kl.squeeze(), expected, atol=1e-3), \
            f"Expected {expected:.4f}, got {kl.squeeze():.4f}"

    def test_nan_guard(self, cpu_device):
        """NaN in sigma_t gets replaced, output is finite."""
        B, N, K = 1, 2, 3
        mu_q, Sigma_q = _random_dense_beliefs(B, N, K, cpu_device)
        mu_t = torch.randn(B, N, K)
        Sigma_t = _random_dense_beliefs(B, N, K, cpu_device)[1]
        # Inject NaN into one entry
        Sigma_t[0, 0, 0, 0] = float('nan')
        kl = _kl_kernel_dense(mu_q, Sigma_q, mu_t, Sigma_t, kl_max=100.0, eps=1e-6)
        assert torch.isfinite(kl).all(), "NaN propagated to output"

    def test_gradient_flow(self, cpu_device):
        """Autograd through dense Cholesky path."""
        B, N, K = 1, 2, 3
        mu_q = torch.randn(B, N, K, requires_grad=True)
        A = torch.randn(B, N, K, K) * 0.3
        Sigma_q = (A @ A.transpose(-1, -2) + 0.2 * torch.eye(K))
        Sigma_q = Sigma_q.detach().requires_grad_(True)
        mu_t = torch.randn(B, N, K)
        Sigma_t = _random_dense_beliefs(B, N, K, cpu_device)[1]
        kl = _kl_kernel_dense(mu_q, Sigma_q, mu_t, Sigma_t, kl_max=100.0, eps=1e-6)
        kl.sum().backward()
        assert mu_q.grad is not None and torch.isfinite(mu_q.grad).all()

    def test_output_shape(self, cpu_device):
        """Output is (...) when input is (..., K) / (..., K, K)."""
        B, N, K = 2, 5, 4
        mu_q, Sigma_q = _random_dense_beliefs(B, N, K, cpu_device)
        mu_t, Sigma_t = _random_dense_beliefs(B, N, K, cpu_device)
        kl = _kl_kernel_dense(mu_q, Sigma_q, mu_t, Sigma_t, kl_max=100.0, eps=1e-6)
        assert kl.shape == (B, N)


# =============================================================================
# TestComputeKLMatrix
# =============================================================================

class TestComputeKLMatrix:
    """Entry-point compute_kl_matrix: mode dispatch, chunking, block-diagonal."""

    def test_diagonal_shape(self, cpu_device):
        """DIAGONAL mode returns (B, N, N) matrix."""
        B, N, K = 2, 6, 4
        mu_q, sigma_q = _random_diagonal_beliefs(B, N, K, cpu_device)
        mu_t, sigma_t = _transport_identity(mu_q, sigma_q)
        kl = compute_kl_matrix(mu_q, sigma_q, mu_t, sigma_t, mode=KLMode.DIAGONAL)
        assert kl.shape == (B, N, N), f"Expected ({B},{N},{N}), got {kl.shape}"

    def test_dense_shape(self, cpu_device):
        """DENSE mode returns (B, N, N) matrix."""
        B, N, K = 2, 4, 3
        mu_q, Sigma_q = _random_dense_beliefs(B, N, K, cpu_device)
        mu_t, Sigma_t = _transport_identity(mu_q, Sigma_q)
        kl = compute_kl_matrix(mu_q, Sigma_q, mu_t, Sigma_t, mode=KLMode.DENSE)
        assert kl.shape == (B, N, N)

    def test_self_kl_diagonal_is_zero(self, cpu_device):
        """Diagonal of KL matrix should be ~0 (self-KL) for identity transport."""
        B, N, K = 2, 6, 4
        mu_q, sigma_q = _random_diagonal_beliefs(B, N, K, cpu_device)
        mu_t, sigma_t = _transport_identity(mu_q, sigma_q)
        kl = compute_kl_matrix(mu_q, sigma_q, mu_t, sigma_t, mode=KLMode.DIAGONAL)
        diag = torch.diagonal(kl, dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-3), \
            f"Self-KL max: {diag.abs().max():.2e}"

    def test_chunked_equals_unchunked_diagonal(self, cpu_device):
        """Chunked and unchunked paths give identical results for DIAGONAL mode."""
        B, N, K = 2, 8, 4
        mu_q, sigma_q = _random_diagonal_beliefs(B, N, K, cpu_device)
        mu_t, sigma_t = _transport_identity(mu_q, sigma_q)

        kl_full = compute_kl_matrix(mu_q, sigma_q, mu_t, sigma_t,
                                     mode=KLMode.DIAGONAL, chunk_size=None)
        kl_chunked = compute_kl_matrix(mu_q, sigma_q, mu_t, sigma_t,
                                        mode=KLMode.DIAGONAL, chunk_size=3)
        assert torch.allclose(kl_full, kl_chunked, atol=1e-5), \
            f"Chunked vs unchunked max deviation: {(kl_full - kl_chunked).abs().max():.2e}"

    def test_chunked_equals_unchunked_dense(self, cpu_device):
        """Chunked and unchunked paths give identical results for DENSE mode."""
        B, N, K = 1, 6, 3
        mu_q, Sigma_q = _random_dense_beliefs(B, N, K, cpu_device)
        mu_t, Sigma_t = _transport_identity(mu_q, Sigma_q)

        kl_full = compute_kl_matrix(mu_q, Sigma_q, mu_t, Sigma_t,
                                     mode=KLMode.DENSE, chunk_size=None)
        kl_chunked = compute_kl_matrix(mu_q, Sigma_q, mu_t, Sigma_t,
                                        mode=KLMode.DENSE, chunk_size=2)
        assert torch.allclose(kl_full, kl_chunked, atol=1e-4), \
            f"Chunked vs unchunked max deviation: {(kl_full - kl_chunked).abs().max():.2e}"

    def test_block_diagonal_requires_args(self):
        """BLOCK_DIAGONAL mode raises ValueError without required arguments."""
        B, N, K = 1, 4, 6
        mu_q = torch.randn(B, N, K)
        sigma_q = torch.rand(B, N, K).clamp(min=0.1)
        mu_t = torch.randn(B, N, N, K)
        sigma_t = torch.rand(B, N, N, K).clamp(min=0.1)

        with pytest.raises(ValueError, match="block_exp_pairs"):
            compute_kl_matrix(mu_q, sigma_q, mu_t, sigma_t,
                              mode=KLMode.BLOCK_DIAGONAL)

    def test_non_negativity_matrix(self, cpu_device):
        """Full KL matrix is non-negative everywhere."""
        B, N, K = 2, 6, 4
        mu_q, sigma_q = _random_diagonal_beliefs(B, N, K, cpu_device)
        mu_t, sigma_t = _transport_identity(mu_q, sigma_q)
        kl = compute_kl_matrix(mu_q, sigma_q, mu_t, sigma_t, mode=KLMode.DIAGONAL)
        assert (kl >= -1e-4).all(), f"Negative KL in matrix: min={kl.min():.2e}"

    def test_output_finite(self, cpu_device):
        """No NaN/Inf in KL matrix output."""
        B, N, K = 2, 6, 5
        mu_q, sigma_q = _random_diagonal_beliefs(B, N, K, cpu_device)
        mu_t, sigma_t = _transport_identity(mu_q, sigma_q)
        kl = compute_kl_matrix(mu_q, sigma_q, mu_t, sigma_t, mode=KLMode.DIAGONAL)
        assert torch.isfinite(kl).all(), "KL matrix contains NaN or Inf"

    def test_kl_matrix_not_symmetric(self, cpu_device):
        """KL matrix is NOT symmetric (KL is asymmetric)."""
        B, N, K = 1, 4, 3
        mu_q, sigma_q = _random_diagonal_beliefs(B, N, K, cpu_device)
        # Create non-identity transport so off-diagonals differ
        mu_t = torch.randn(B, N, N, K, device=cpu_device)
        sigma_t = torch.rand(B, N, N, K, device=cpu_device).clamp(min=0.1) + 0.1
        kl = compute_kl_matrix(mu_q, sigma_q, mu_t, sigma_t, mode=KLMode.DIAGONAL)
        # Check KL[i,j] != KL[j,i] for at least one pair
        diff = (kl - kl.transpose(-1, -2)).abs()
        assert diff.max() > 1e-3, "KL matrix should not be symmetric"

    def test_custom_kl_max(self, cpu_device):
        """kl_max parameter is respected in output."""
        B, N, K = 1, 4, 3
        mu_q = torch.zeros(B, N, K)
        sigma_q = torch.ones(B, N, K) * 0.01  # very narrow
        # Transport to very different means -> large KL
        mu_t = torch.ones(B, N, N, K) * 100.0
        sigma_t = torch.ones(B, N, N, K) * 0.01
        kl = compute_kl_matrix(mu_q, sigma_q, mu_t, sigma_t,
                               mode=KLMode.DIAGONAL, kl_max=25.0)
        assert (kl <= 25.0 + 1e-4).all(), f"kl_max=25 violated: max={kl.max():.2e}"


# =============================================================================
# TestNumericalMonitor
# =============================================================================

class TestNumericalMonitor:
    """Tests for math_utils.numerical_monitor record/flush counter."""

    def test_record_and_flush(self):
        """record() increments, flush() returns and resets."""
        from math_utils.numerical_monitor import record, flush, _counts
        _counts.clear()  # ensure clean state
        record("test_event")
        record("test_event")
        result = flush()
        assert result == {"test_event": 2}
        assert flush() == {}, "flush should reset counters"

    def test_multiple_events(self):
        """Independent counters for different events."""
        from math_utils.numerical_monitor import record, flush, _counts
        _counts.clear()
        record("chol_recover")
        record("nan_replace")
        record("chol_recover")
        result = flush()
        assert result["chol_recover"] == 2
        assert result["nan_replace"] == 1

    def test_count_parameter(self):
        """record(event, count=N) increments by N."""
        from math_utils.numerical_monitor import record, flush, _counts
        _counts.clear()
        record("neg_clamp", count=5)
        record("neg_clamp", count=3)
        result = flush()
        assert result["neg_clamp"] == 8

    def test_empty_flush(self):
        """flush() on empty returns empty dict."""
        from math_utils.numerical_monitor import flush, _counts
        _counts.clear()
        assert flush() == {}
