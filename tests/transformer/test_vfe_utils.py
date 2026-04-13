"""
Tests for transformer.core.vfe_utils
=====================================

Validates the pure mathematical utility functions used by the VFE E-step:
SPD inversion, eigendecomposition, SPD retraction (full and diagonal),
squeeze utility, debug helpers, and multi-head aggregation.

Mathematical invariants tested:
    - M_inv @ M approx I (inverse correctness)
    - Eigendecomposition round-trip: V @ diag(lam) @ V^T approx M
    - SPD retraction preserves positive-definiteness
    - Diagonal retraction preserves positivity
    - squeeze_trailing_singletons contracts only trailing size-1 dims
"""

import math
import pytest
import torch

from transformer.core.vfe_utils import (
    squeeze_trailing_singletons,
    _safe_spd_inv,
    _safe_eigh,
    retract_spd_torch,
    retract_spd_diagonal_torch,
    _retract_phi,
    _grad_norm,
    _per_pos_stats,
    _aggregate_multihead_vfe_debug,
    SIGMA_EPS,
)


# =============================================================================
# Helpers
# =============================================================================

def _random_spd(K, batch_shape=(), device='cpu', scale=0.3):
    """Generate random SPD matrix: A A^T + 0.1 I."""
    A = torch.randn(*batch_shape, K, K, device=device) * scale
    return A @ A.transpose(-1, -2) + 0.1 * torch.eye(K, device=device)


# =============================================================================
# TestSqueezeTrailingSingletons
# =============================================================================

class TestSqueezeTrailingSingletons:
    """squeeze_trailing_singletons: remove trailing size-1 dims until max_dim."""

    def test_squeeze_to_3d(self):
        """(B, N, K, 1, 1) -> (B, N, K) when max_dim=3."""
        t = torch.randn(2, 4, 6, 1, 1)
        result = squeeze_trailing_singletons(t, max_dim=3)
        assert result.shape == (2, 4, 6)

    def test_noop_at_max_dim(self):
        """No-op when already at max_dim."""
        t = torch.randn(2, 4, 6)
        result = squeeze_trailing_singletons(t, max_dim=3)
        assert result.shape == (2, 4, 6)

    def test_noop_below_max_dim(self):
        """No-op when below max_dim."""
        t = torch.randn(2, 4)
        result = squeeze_trailing_singletons(t, max_dim=3)
        assert result.shape == (2, 4)

    def test_non_singleton_preserved(self):
        """Non-singleton trailing dims are not removed."""
        t = torch.randn(2, 4, 6, 3)
        result = squeeze_trailing_singletons(t, max_dim=3)
        assert result.shape == (2, 4, 6, 3), "Should not remove dim of size 3"

    def test_partial_squeeze(self):
        """(B, N, K, K, 1) -> (B, N, K, K) when max_dim=4."""
        t = torch.randn(2, 4, 3, 3, 1)
        result = squeeze_trailing_singletons(t, max_dim=4)
        assert result.shape == (2, 4, 3, 3)

    def test_data_unchanged(self):
        """Values are preserved through squeeze."""
        t = torch.randn(2, 3, 4, 1, 1)
        result = squeeze_trailing_singletons(t, max_dim=3)
        assert torch.equal(result, t.squeeze(-1).squeeze(-1))


# =============================================================================
# TestSafeSpdInv
# =============================================================================

class TestSafeSpdInv:
    r"""_safe_spd_inv: robust SPD inversion with adaptive regularization.

    Key invariant: M_inv @ M \approx I for well-conditioned inputs.
    """

    def test_inverse_identity(self, cpu_device):
        """M_inv @ M approx I for well-conditioned SPD."""
        K = 4
        M = _random_spd(K, batch_shape=(2, 3), device=cpu_device)
        M_inv = _safe_spd_inv(M)
        product = M_inv @ M
        I = torch.eye(K, device=cpu_device)
        assert torch.allclose(product, I.expand_as(product), atol=1e-3), \
            f"max deviation from I: {(product - I).abs().max():.2e}"

    def test_spd_preservation(self, cpu_device):
        """Inverse of SPD is SPD (symmetric, positive eigenvalues)."""
        K = 4
        M = _random_spd(K, device=cpu_device)
        M_inv = _safe_spd_inv(M)
        # Symmetry
        assert torch.allclose(M_inv, M_inv.transpose(-1, -2), atol=1e-5), \
            "Inverse should be symmetric"
        # Positive eigenvalues
        eigvals = torch.linalg.eigvalsh(M_inv)
        assert (eigvals > 0).all(), f"Non-positive eigenvalues in inverse: {eigvals}"

    def test_batch_support(self, cpu_device):
        """Works with batched (..., K, K) inputs."""
        K = 3
        M = _random_spd(K, batch_shape=(2, 5), device=cpu_device)
        M_inv = _safe_spd_inv(M)
        assert M_inv.shape == (2, 5, K, K)
        product = M_inv @ M
        I = torch.eye(K, device=cpu_device)
        assert torch.allclose(product, I.expand_as(product), atol=1e-3)

    def test_pseudoinverse_fallback(self, cpu_device):
        """Rank-deficient matrix doesn't crash (falls back to pinv)."""
        K = 4
        # Rank-1 matrix: singular, but _safe_spd_inv should handle it
        v = torch.randn(K)
        M = v.unsqueeze(-1) @ v.unsqueeze(-2)  # rank 1
        M_inv = _safe_spd_inv(M, eps=1e-4)
        assert torch.isfinite(M_inv).all(), "Pseudoinverse fallback produced NaN/Inf"

    def test_output_dtype_matches_input(self, cpu_device):
        """Output dtype matches input dtype."""
        K = 3
        M = _random_spd(K, device=cpu_device)
        M_inv = _safe_spd_inv(M)
        assert M_inv.dtype == M.dtype

    def test_identity_gives_identity(self, cpu_device):
        """inv(I) = I."""
        K = 4
        I = torch.eye(K, device=cpu_device).unsqueeze(0)
        I_inv = _safe_spd_inv(I)
        assert torch.allclose(I_inv.squeeze(0), torch.eye(K), atol=1e-4)


# =============================================================================
# TestSafeEigh
# =============================================================================

class TestSafeEigh:
    r"""_safe_eigh: robust eigendecomposition with escalating jitter + SVD fallback.

    Key invariant: V @ diag(lam) @ V^T \approx M.
    """

    def test_round_trip(self, cpu_device):
        """V @ diag(lam) @ V^T reconstructs M."""
        K = 5
        M = _random_spd(K, device=cpu_device)
        eigvals, eigvecs = _safe_eigh(M)
        reconstructed = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
        # The jitter may shift eigenvalues slightly, so relax tolerance
        assert torch.allclose(reconstructed, M, atol=1e-3), \
            f"max reconstruction error: {(reconstructed - M).abs().max():.2e}"

    def test_eigenvalues_real_and_ordered(self, cpu_device):
        """Eigenvalues are real (all finite) and in ascending order."""
        K = 4
        M = _random_spd(K, batch_shape=(2,), device=cpu_device)
        eigvals, _ = _safe_eigh(M)
        assert torch.isfinite(eigvals).all()
        # Ascending order (torch.linalg.eigh convention)
        diffs = eigvals[..., 1:] - eigvals[..., :-1]
        assert (diffs >= -1e-5).all(), "Eigenvalues not in ascending order"

    def test_orthogonal_eigenvectors(self, cpu_device):
        """V^T V = I (orthonormal eigenvectors)."""
        K = 4
        M = _random_spd(K, device=cpu_device)
        _, V = _safe_eigh(M)
        product = V.transpose(-1, -2) @ V
        I = torch.eye(K, device=cpu_device)
        assert torch.allclose(product, I, atol=1e-4), \
            f"max deviation from orthogonality: {(product - I).abs().max():.2e}"

    def test_symmetrize_flag(self, cpu_device):
        """Asymmetric input gets symmetrized when symmetrize=True."""
        K = 3
        M = torch.randn(K, K, device=cpu_device)
        M_sym = 0.5 * (M + M.T)
        M_sym = M_sym @ M_sym + 0.1 * torch.eye(K)  # make SPD
        # Add asymmetry
        M_asym = M_sym + 0.01 * torch.randn(K, K)
        # Should not crash with symmetrize=True (default)
        eigvals, eigvecs = _safe_eigh(M_asym, symmetrize=True)
        assert torch.isfinite(eigvals).all()

    def test_batch_support(self, cpu_device):
        """Works with (..., K, K) batch shapes."""
        K = 3
        M = _random_spd(K, batch_shape=(2, 4), device=cpu_device)
        eigvals, eigvecs = _safe_eigh(M)
        assert eigvals.shape == (2, 4, K)
        assert eigvecs.shape == (2, 4, K, K)

    def test_output_dtype_matches(self, cpu_device):
        """Output dtype matches input dtype."""
        K = 3
        M = _random_spd(K, device=cpu_device)
        eigvals, eigvecs = _safe_eigh(M)
        assert eigvals.dtype == M.dtype
        assert eigvecs.dtype == M.dtype

    def test_positive_eigenvalues_for_spd(self, cpu_device):
        """SPD input has all positive eigenvalues (after jitter)."""
        K = 5
        M = _random_spd(K, device=cpu_device)
        eigvals, _ = _safe_eigh(M)
        assert (eigvals > 0).all(), f"Non-positive eigenvalues: {eigvals[eigvals <= 0]}"


# =============================================================================
# TestRetractSPDTorch
# =============================================================================

class TestRetractSPDTorch:
    """retract_spd_torch: affine-invariant exponential map on SPD manifold.

    Key invariant: output is always SPD (positive-definite, symmetric).
    """

    def test_spd_preservation(self, cpu_device):
        """Output eigenvalues are all > 0 (positive-definite)."""
        B, N, K = 2, 3, 4
        Sigma = _random_spd(K, batch_shape=(B, N), device=cpu_device)
        delta = torch.randn(B, N, K, K, device=cpu_device) * 0.1
        delta = 0.5 * (delta + delta.transpose(-1, -2))  # symmetrize tangent
        result = retract_spd_torch(Sigma, delta, step_size=0.5)
        eigvals = torch.linalg.eigvalsh(result.reshape(-1, K, K))
        assert (eigvals > 0).all(), f"Non-positive eigenvalues: {eigvals.min():.2e}"

    def test_symmetry(self, cpu_device):
        """Output is symmetric."""
        B, N, K = 2, 3, 4
        Sigma = _random_spd(K, batch_shape=(B, N), device=cpu_device)
        delta = torch.randn(B, N, K, K, device=cpu_device) * 0.1
        delta = 0.5 * (delta + delta.transpose(-1, -2))
        result = retract_spd_torch(Sigma, delta, step_size=0.5)
        assert torch.allclose(result, result.transpose(-1, -2), atol=1e-5), \
            "Output not symmetric"

    def test_zero_step_identity(self, cpu_device):
        """step_size=0 returns original Sigma (up to numerical noise)."""
        B, N, K = 1, 2, 3
        Sigma = _random_spd(K, batch_shape=(B, N), device=cpu_device)
        delta = torch.randn(B, N, K, K, device=cpu_device)
        delta = 0.5 * (delta + delta.transpose(-1, -2))
        result = retract_spd_torch(Sigma, delta, step_size=0.0)
        assert torch.allclose(result, Sigma, atol=1e-3), \
            f"max deviation at step_size=0: {(result - Sigma).abs().max():.2e}"

    def test_sigma_max_eigenvalue_bound(self, cpu_device):
        """No eigenvalue exceeds sigma_max^2."""
        B, N, K = 1, 2, 3
        sigma_max = 3.0
        Sigma = _random_spd(K, batch_shape=(B, N), device=cpu_device)
        # Large delta to push eigenvalues up
        delta = 10.0 * torch.eye(K, device=cpu_device).expand(B, N, K, K)
        result = retract_spd_torch(Sigma, delta, step_size=1.0, sigma_max=sigma_max)
        eigvals = torch.linalg.eigvalsh(result.reshape(-1, K, K))
        assert (eigvals <= sigma_max ** 2 + 1e-3).all(), \
            f"Eigenvalue {eigvals.max():.4f} exceeds sigma_max^2={sigma_max**2}"

    def test_output_shape_4d(self, cpu_device):
        """4D input (B, N, K, K) returns 4D output."""
        B, N, K = 2, 3, 4
        Sigma = _random_spd(K, batch_shape=(B, N), device=cpu_device)
        delta = torch.randn(B, N, K, K, device=cpu_device) * 0.1
        delta = 0.5 * (delta + delta.transpose(-1, -2))
        result = retract_spd_torch(Sigma, delta)
        assert result.shape == (B, N, K, K)

    def test_output_shape_3d(self, cpu_device):
        """3D input (BN, K, K) returns 3D output."""
        BN, K = 6, 4
        Sigma = _random_spd(K, batch_shape=(BN,), device=cpu_device)
        delta = torch.randn(BN, K, K, device=cpu_device) * 0.1
        delta = 0.5 * (delta + delta.transpose(-1, -2))
        result = retract_spd_torch(Sigma, delta)
        assert result.shape == (BN, K, K)

    def test_output_finite(self, cpu_device):
        """No NaN/Inf in output."""
        B, N, K = 2, 3, 4
        Sigma = _random_spd(K, batch_shape=(B, N), device=cpu_device)
        delta = torch.randn(B, N, K, K, device=cpu_device) * 0.5
        delta = 0.5 * (delta + delta.transpose(-1, -2))
        result = retract_spd_torch(Sigma, delta, step_size=1.0)
        assert torch.isfinite(result).all(), "Output contains NaN or Inf"


# =============================================================================
# TestRetractSPDDiagonalTorch
# =============================================================================

class TestRetractSPDDiagonalTorch:
    """retract_spd_diagonal_torch: exponential retraction for diagonal covariances.

    Key invariant: output is always positive (sigma_new > 0).
    """

    def test_positive_output(self, cpu_device):
        """All diagonal entries > 0."""
        B, N, K = 2, 4, 6
        sigma = torch.rand(B, N, K, device=cpu_device).clamp(min=0.1) + 0.1
        delta = torch.randn(B, N, K, device=cpu_device) * 0.5
        result = retract_spd_diagonal_torch(sigma, delta, step_size=1.0)
        assert (result > 0).all(), f"Non-positive values: min={result.min():.2e}"

    def test_zero_step_identity(self, cpu_device):
        """step_size=0 returns original sigma."""
        B, N, K = 1, 2, 4
        sigma = torch.rand(B, N, K, device=cpu_device).clamp(min=0.1) + 0.5
        delta = torch.randn(B, N, K, device=cpu_device)
        result = retract_spd_diagonal_torch(sigma, delta, step_size=0.0)
        # exp(0) = 1, so sigma_new = sigma * 1 = sigma (clamped to [eps, sigma_max])
        assert torch.allclose(result, sigma.clamp(min=SIGMA_EPS, max=5.0), atol=1e-4), \
            f"max deviation at step_size=0: {(result - sigma).abs().max():.2e}"

    def test_sigma_max_bound(self, cpu_device):
        """Output never exceeds sigma_max."""
        sigma_max = 3.0
        sigma = torch.ones(1, 1, 4) * 2.0
        delta = torch.ones(1, 1, 4) * 100.0  # large positive delta
        result = retract_spd_diagonal_torch(sigma, delta, step_size=1.0, sigma_max=sigma_max)
        assert (result <= sigma_max + 1e-6).all(), f"max={result.max():.4f} > sigma_max={sigma_max}"

    def test_sigma_eps_floor(self, cpu_device):
        """Output never drops below eps."""
        sigma = torch.ones(1, 1, 4) * 0.5
        delta = torch.ones(1, 1, 4) * -100.0  # large negative delta
        result = retract_spd_diagonal_torch(sigma, delta, step_size=1.0, eps=1e-6)
        assert (result >= 1e-6).all(), f"min={result.min():.2e} < eps"

    def test_monotonicity_positive_delta(self, cpu_device):
        """Positive gradient direction increases sigma."""
        B, N, K = 1, 1, 4
        sigma = torch.ones(B, N, K) * 1.0
        delta = torch.ones(B, N, K) * 0.5  # positive direction
        result = retract_spd_diagonal_torch(sigma, delta, step_size=1.0)
        # exp(0.5) > 1, so sigma_new > sigma
        assert (result > sigma).all(), "Positive delta should increase sigma"

    def test_output_shape(self, cpu_device):
        """Output shape matches input."""
        B, N, K = 2, 5, 8
        sigma = torch.rand(B, N, K).clamp(min=0.1)
        delta = torch.randn(B, N, K)
        result = retract_spd_diagonal_torch(sigma, delta)
        assert result.shape == (B, N, K)

    def test_output_finite(self, cpu_device):
        """No NaN/Inf for random inputs."""
        B, N, K = 2, 4, 6
        sigma = torch.rand(B, N, K).clamp(min=0.1) + 0.1
        delta = torch.randn(B, N, K) * 2.0
        result = retract_spd_diagonal_torch(sigma, delta, step_size=1.0)
        assert torch.isfinite(result).all()


# =============================================================================
# TestRetractPhi
# =============================================================================

class TestRetractPhi:
    """_retract_phi: gauge frame retraction to Lie group."""

    def test_output_shape(self, cpu_device, so3_generators):
        """Output has same shape as input phi."""
        n_gen = so3_generators.shape[0]
        phi = torch.randn(2, 4, n_gen, device=cpu_device)
        delta = torch.randn(2, 4, n_gen, device=cpu_device)
        result = _retract_phi(phi, delta, so3_generators, step_size=0.1)
        assert result.shape == phi.shape

    def test_output_finite(self, cpu_device, so3_generators):
        """No NaN/Inf in output."""
        n_gen = so3_generators.shape[0]
        phi = torch.randn(2, 4, n_gen, device=cpu_device)
        delta = torch.randn(2, 4, n_gen, device=cpu_device)
        result = _retract_phi(phi, delta, so3_generators, step_size=0.1)
        assert torch.isfinite(result).all()

    def test_zero_delta(self, cpu_device, so3_generators):
        """Zero delta returns phi close to original (within trust region clamp)."""
        n_gen = so3_generators.shape[0]
        phi = torch.randn(2, 4, n_gen, device=cpu_device) * 0.1
        delta = torch.zeros(2, 4, n_gen, device=cpu_device)
        result = _retract_phi(phi, delta, so3_generators, step_size=0.1)
        assert torch.allclose(result, phi, atol=1e-5), \
            f"Zero delta should leave phi unchanged, max dev: {(result - phi).abs().max():.2e}"

    def test_max_norm_bound_son(self, cpu_device, so3_generators):
        """SO(N) phi is bounded by max_norm=pi."""
        n_gen = so3_generators.shape[0]
        phi = torch.randn(2, 4, n_gen, device=cpu_device) * 0.1
        # Large delta to push norm past pi
        delta = torch.randn(2, 4, n_gen, device=cpu_device) * 100.0
        result = _retract_phi(
            phi, delta, so3_generators, step_size=1.0,
            gauge_group='SON'
        )
        norms = torch.norm(result, dim=-1)
        assert (norms <= math.pi + 0.1).all(), \
            f"SO(N) phi norm {norms.max():.4f} exceeds pi"


# =============================================================================
# TestGradNorm
# =============================================================================

class TestGradNorm:
    """_grad_norm: global Frobenius norm helper."""

    def test_zero_tensor(self):
        """Norm of zeros is 0."""
        assert _grad_norm(torch.zeros(3, 4)) == 0.0

    def test_known_norm(self):
        """Known Frobenius norm matches."""
        t = torch.tensor([[3.0, 4.0]])  # norm = 5
        assert abs(_grad_norm(t) - 5.0) < 1e-5


# =============================================================================
# TestPerPosStats
# =============================================================================

class TestPerPosStats:
    """_per_pos_stats: per-position norm statistics."""

    def test_3d_tensor(self):
        """Works with (B, N, K) input."""
        t = torch.ones(2, 3, 4)  # each position has norm = 2.0
        mean_n, max_n, frac = _per_pos_stats(t)
        assert abs(mean_n - 2.0) < 1e-4
        assert abs(max_n - 2.0) < 1e-4
        assert frac == 0.0  # no norms above 100

    def test_4d_tensor(self):
        """Works with (B, N, K, K) input."""
        t = torch.ones(1, 2, 3, 3)
        mean_n, max_n, frac = _per_pos_stats(t)
        assert mean_n > 0
        assert max_n > 0
        assert frac == 0.0

    def test_large_norms_fraction(self):
        """Fraction above 100 is computed correctly."""
        t = torch.zeros(1, 4, 3)
        t[0, 0, :] = 200.0  # norm = 200*sqrt(3) >> 100
        t[0, 1, :] = 200.0
        _, _, frac = _per_pos_stats(t)
        assert frac == 0.5  # 2 out of 4 positions above 100


# =============================================================================
# TestAggregateMultiheadVfeDebug
# =============================================================================

class TestAggregateMultiheadVfeDebug:
    """_aggregate_multihead_vfe_debug: per-head metric aggregation."""

    def test_gradient_rms_aggregation(self):
        """Gradient norms aggregate as RMS weighted by head dimension."""
        d = {
            'head0(d=3)/grad_mu_self': 1.0,
            'head1(d=2)/grad_mu_self': 2.0,
        }
        irrep_dims = [3, 2]
        _aggregate_multihead_vfe_debug(d, irrep_dims)
        # RMS: sqrt((3*1^2 + 2*2^2) / 5) = sqrt((3+8)/5) = sqrt(2.2)
        expected = math.sqrt((3 * 1.0 + 2 * 4.0) / 5)
        assert abs(d['grad_mu_self'] - expected) < 1e-6

    def test_kl_max_aggregation(self):
        """kl_pairwise_max takes max across heads."""
        d = {
            'head0(d=3)/kl_pairwise_max': 5.0,
            'head1(d=2)/kl_pairwise_max': 10.0,
        }
        _aggregate_multihead_vfe_debug(d, [3, 2])
        assert d['kl_pairwise_max'] == 10.0

    def test_weighted_mean_aggregation(self):
        """kl_pairwise_mean uses dimension-weighted mean."""
        d = {
            'head0(d=3)/kl_pairwise_mean': 6.0,
            'head1(d=2)/kl_pairwise_mean': 4.0,
        }
        _aggregate_multihead_vfe_debug(d, [3, 2])
        # (3*6 + 2*4) / 5 = 26/5 = 5.2
        assert abs(d['kl_pairwise_mean'] - 5.2) < 1e-6

    def test_no_head_keys_noop(self):
        """No head keys means no aggregation."""
        d = {'some_key': 42.0}
        _aggregate_multihead_vfe_debug(d, [3, 2])
        assert d == {'some_key': 42.0}

    def test_empty_dict(self):
        """Empty dict is a no-op."""
        d = {}
        _aggregate_multihead_vfe_debug(d, [3, 2])
        assert d == {}
