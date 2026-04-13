"""
Tests for gauge.py: transport operators, Lie algebra, trust regions, initialization.

Covers:
  - phi_to_omega / retract_phi (Lie algebra path)
  - lie_algebra_clip_grad (Riemannian trust region)
  - regularize_omega_conditioning (polar factor regularization)
  - init_omega (GL(K) initialization)
"""

import math
import torch
import pytest

from .conftest import random_spd, random_gl, K, EPS, REL_TOL

DEVICE = 'cpu'


class TestPhiToOmega:
    """Test Lie algebra → group element conversion."""

    def test_phi_zero_gives_identity(self):
        """phi = 0 => Omega = exp(0) = I."""
        from ..gauge import phi_to_omega, make_gl_generators

        generators = make_gl_generators(K)
        phi = torch.zeros(K * K)
        Omega = phi_to_omega(phi, generators)

        diff = (Omega - torch.eye(K)).norm()
        assert diff < 1e-6, f"phi=0 should give I, got diff = {diff:.6e}"

    def test_phi_roundtrip_consistency(self):
        """Small phi: exp(phi*G) should be near I + phi*G (first-order Taylor)."""
        from ..gauge import phi_to_omega, make_gl_generators

        generators = make_gl_generators(K)
        torch.manual_seed(42)
        phi = 0.01 * torch.randn(K * K)  # Small perturbation

        Omega = phi_to_omega(phi, generators)
        # First-order: exp(X) ≈ I + X for small X
        X = torch.einsum('a,aij->ij', phi, generators)
        I_plus_X = torch.eye(K) + X

        diff = (Omega - I_plus_X).norm()
        assert diff < 1e-3, f"First-order Taylor error too large: {diff:.6e}"

    def test_phi_to_omega_invertible(self):
        """exp(phi*G) should always be invertible."""
        from ..gauge import phi_to_omega, make_gl_generators

        generators = make_gl_generators(K)
        torch.manual_seed(42)
        phi = torch.randn(5, K * K)  # Batch of random phi

        Omega = phi_to_omega(phi, generators)
        dets = torch.linalg.det(Omega)
        assert (dets.abs() > 1e-6).all(), f"Some Omega not invertible: min |det| = {dets.abs().min():.6e}"


class TestRetractPhi:
    """Test Lie algebra retraction with norm capping."""

    def test_retract_phi_norm_cap(self):
        """Large update gets capped to max_norm."""
        from ..gauge import retract_phi

        phi = torch.zeros(K * K)
        delta_phi = torch.randn(K * K) * 100.0  # Very large update
        max_norm = 2.0

        phi_new = retract_phi(phi, delta_phi, max_norm=max_norm)
        assert phi_new.norm() <= max_norm + 1e-5, \
            f"Norm cap violated: ||phi|| = {phi_new.norm():.4f} > {max_norm}"

    def test_retract_phi_small_update_passthrough(self):
        """Small update within norm passes through unchanged."""
        from ..gauge import retract_phi

        phi = torch.zeros(K * K)
        delta_phi = 0.01 * torch.randn(K * K)
        max_norm = 5.0

        phi_new = retract_phi(phi, delta_phi, max_norm=max_norm)
        expected = phi + delta_phi
        diff = (phi_new - expected).norm()
        assert diff < 1e-6, f"Small update should pass through: diff = {diff:.6e}"

    def test_retract_phi_auto_norm_detection(self):
        """GL(K) detection (n_gen = K^2) gives max_norm=5.0."""
        from ..gauge import retract_phi

        # K=4, so n_gen = 16 = 4^2 → detected as GL(K), max_norm = 5.0
        phi = torch.zeros(K * K)
        delta_phi = torch.randn(K * K) * 100.0

        phi_new = retract_phi(phi, delta_phi, max_norm=None)
        assert phi_new.norm() <= 5.0 + 1e-5, \
            f"Auto GL(K) norm should be 5.0, got ||phi|| = {phi_new.norm():.4f}"


class TestLieAlgebraClipGrad:
    """Test Lie algebra trust region: xi = Omega^T @ grad, clip ||xi||_F."""

    def test_lie_clip_riemannian_norm_bounded(self):
        """Output satisfies ||Omega^{-1} * result||_F <= trust_radius."""
        from ..gauge import lie_algebra_clip_grad

        torch.manual_seed(42)
        Omega = random_gl(K, scale=0.2)
        grad = torch.randn(K, K) * 10.0  # Large gradient
        trust_radius = 0.3

        nat_grad = lie_algebra_clip_grad(grad, Omega, trust_radius)

        # The Lie algebra element is xi = Omega^T @ grad (before clipping)
        # After clipping, the result is Omega @ xi_clipped
        # So Omega^{-1} @ result = xi_clipped, and ||xi_clipped||_F <= trust_radius
        xi_recovered = torch.linalg.inv(Omega) @ nat_grad
        xi_norm = xi_recovered.norm()
        assert xi_norm <= trust_radius + 1e-5, \
            f"Riemannian norm {xi_norm:.4f} exceeds trust_radius {trust_radius}"

    def test_lie_clip_zero_grad_gives_zero(self):
        """Zero gradient produces zero update."""
        from ..gauge import lie_algebra_clip_grad

        Omega = random_gl(K)
        grad = torch.zeros(K, K)

        nat_grad = lie_algebra_clip_grad(grad, Omega, trust_radius=0.3)
        assert nat_grad.norm() < 1e-10, f"Zero grad should give zero: norm = {nat_grad.norm():.6e}"

    def test_lie_clip_small_grad_passthrough(self):
        """Small gradient within trust region is not clipped."""
        from ..gauge import lie_algebra_clip_grad

        torch.manual_seed(42)
        Omega = random_gl(K, scale=0.1)
        grad = torch.randn(K, K) * 0.001  # Very small gradient
        trust_radius = 10.0  # Very large trust region

        nat_grad = lie_algebra_clip_grad(grad, Omega, trust_radius)
        # Should be Omega @ (Omega^T @ grad) = Omega @ Omega^T @ grad
        expected = Omega @ (Omega.T @ grad)
        diff = (nat_grad - expected).norm() / expected.norm().clamp(min=1e-8)
        assert diff < 1e-4, f"Small grad should not be clipped: diff = {diff:.6e}"


class TestRegularizeOmegaConditioning:
    """Test progressive polar factor regularization."""

    def test_wellconditioned_unchanged(self):
        """Omega with cond < cond_max passes through unchanged."""
        from ..gauge import regularize_omega_conditioning

        torch.manual_seed(42)
        Omega = random_gl(K, (3,), scale=0.1)  # Near identity, cond ~1
        result = regularize_omega_conditioning(Omega, cond_max=50.0)

        diff = (result - Omega).norm()
        assert diff < 1e-8, f"Well-conditioned should be unchanged: diff = {diff:.6e}"

    def test_illconditioned_reduced(self):
        """Omega with cond >> cond_max has lower cond after regularization."""
        from ..gauge import regularize_omega_conditioning

        # Create ill-conditioned Omega
        U, _, Vh = torch.linalg.svd(random_gl(K))
        S = torch.diag(torch.tensor([100.0, 1.0, 0.5, 0.1]))
        Omega = (U @ S @ Vh).unsqueeze(0)  # cond = 1000

        result = regularize_omega_conditioning(Omega, cond_max=10.0)

        cond_before = torch.linalg.svdvals(Omega).max() / torch.linalg.svdvals(Omega).min()
        cond_after = torch.linalg.svdvals(result).max() / torch.linalg.svdvals(result).min()
        assert cond_after < cond_before, \
            f"Conditioning should improve: {cond_before:.1f} -> {cond_after:.1f}"

    def test_preserves_det_sign(self):
        """Regularization preserves sign(det(Omega))."""
        from ..gauge import regularize_omega_conditioning

        # Positive det
        Omega_pos = random_gl(K, (5,), scale=0.5)
        # Force positive det
        dets = torch.linalg.det(Omega_pos)
        neg_mask = dets < 0
        Omega_pos[neg_mask, :, 0] *= -1

        # Negative det
        Omega_neg = Omega_pos.clone()
        Omega_neg[:, :, 0] *= -1  # Flip to negative det

        # Stretch to make ill-conditioned
        Omega_pos[:, 0, :] *= 100.0
        Omega_neg[:, 0, :] *= 100.0

        result_pos = regularize_omega_conditioning(Omega_pos, cond_max=5.0)
        result_neg = regularize_omega_conditioning(Omega_neg, cond_max=5.0)

        dets_pos = torch.linalg.det(result_pos)
        dets_neg = torch.linalg.det(result_neg)

        assert (dets_pos > 0).all(), f"Positive det not preserved: dets = {dets_pos}"
        assert (dets_neg < 0).all(), f"Negative det not preserved: dets = {dets_neg}"


class TestInitOmega:
    """Test GL(K) initialization."""

    def test_init_omega_default_positive_det(self):
        """All frames have det > 0 by default (negative_det_fraction=0)."""
        from ..gauge import init_omega

        torch.manual_seed(42)
        Omega = init_omega((20, 2, K, K), scale=0.1, device='cpu', negative_det_fraction=0.0)
        dets = torch.linalg.det(Omega)
        assert (dets > 0).all(), f"Some frames have negative det: min det = {dets.min():.6e}"

    def test_init_omega_mixed_det(self):
        """negative_det_fraction=0.5 produces ~50% negative dets."""
        from ..gauge import init_omega

        torch.manual_seed(42)
        Omega = init_omega((100, 1, K, K), scale=0.01, device='cpu', negative_det_fraction=0.5)
        dets = torch.linalg.det(Omega)
        neg_frac = (dets < 0).float().mean().item()
        assert 0.3 < neg_frac < 0.7, f"Expected ~50% negative, got {neg_frac:.1%}"

    def test_init_omega_near_identity(self):
        """Small scale => frames close to I."""
        from ..gauge import init_omega

        torch.manual_seed(42)
        scale = 0.01
        Omega = init_omega((10, 1, K, K), scale=scale, device='cpu')

        I = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(10, 1, K, K)
        diff = (Omega - I).norm(dim=(-2, -1)).mean()
        assert diff < 10 * scale * K, f"Not near identity: mean diff = {diff:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
