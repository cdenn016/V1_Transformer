"""
Tests for transformer.core.transport_ops
==========================================

Validates parallel transport operators, RoPE position encoding,
and omega-to-block-exp-pairs conversion.

Mathematical invariants tested:
    - Cocycle property: Omega_ij @ Omega_jk approx Omega_ik (flat connection)
    - exp(phi) @ exp(-phi) = I
    - SO(K) orthogonality when generators are skew-symmetric
    - GL(K) positive determinant: det(exp(X)) > 0
    - RoPE norm preservation: |mu_rotated| = |mu| per pair
    - omega_to_block_exp_pairs round-trip
"""

import math
import pytest
import torch

from transformer.core.transport_ops import (
    compute_transport_operators,
    compute_transport_operators_direct,
    omega_to_block_exp_pairs,
    _apply_rope,
    _build_rope_freqs,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_so3_generators(device='cpu'):
    """SO(3) generators (3, 3, 3) — skew-symmetric."""
    from math_utils.generators import generate_so3_generators
    G = generate_so3_generators(3)
    return torch.from_numpy(G).float().to(device)


def _make_glk_generators(K, device='cpu'):
    """GL(K) generators (K^2, K, K) — basis for gl(K)."""
    n_gen = K * K
    G = torch.zeros(n_gen, K, K, device=device)
    idx = 0
    for i in range(K):
        for j in range(K):
            G[idx, i, j] = 1.0
            idx += 1
    return G


# =============================================================================
# TestComputeTransportOperators
# =============================================================================

class TestComputeTransportOperators:
    """compute_transport_operators: phi-based transport Omega_ij = exp(phi_i)exp(-phi_j)."""

    def test_trivial_gauge_returns_identity(self, cpu_device):
        """gauge_mode='trivial' returns Omega=I for all pairs."""
        G = _make_so3_generators(cpu_device)
        phi = torch.randn(2, 4, 3, device=cpu_device)
        result = compute_transport_operators(phi, G, gauge_mode='trivial')
        I = torch.eye(3, device=cpu_device)
        assert torch.allclose(result['Omega'], I.expand(2, 4, 4, 3, 3), atol=1e-6)
        assert torch.allclose(result['exp_phi'], I.expand(2, 4, 3, 3), atol=1e-6)

    def test_constant_gauge_returns_identity(self, cpu_device):
        """gauge_mode='constant' returns Omega=I (actual Omega injected by attention module)."""
        G = _make_so3_generators(cpu_device)
        phi = torch.randn(2, 4, 3, device=cpu_device)
        result = compute_transport_operators(phi, G, gauge_mode='constant')
        I = torch.eye(3, device=cpu_device)
        assert torch.allclose(result['Omega'], I.expand(2, 4, 4, 3, 3), atol=1e-6)

    def test_self_transport_is_identity(self, cpu_device):
        """Omega_ii = exp(phi_i) exp(-phi_i) = I."""
        G = _make_so3_generators(cpu_device)
        B, N = 2, 4
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.3
        result = compute_transport_operators(phi, G, gauge_mode='learned')
        Omega = result['Omega']
        I = torch.eye(3, device=cpu_device)
        for b in range(B):
            for i in range(N):
                assert torch.allclose(Omega[b, i, i], I, atol=1e-4), \
                    f"Omega[{b},{i},{i}] not identity: max dev {(Omega[b,i,i] - I).abs().max():.2e}"

    def test_exp_phi_times_exp_neg_phi_is_identity(self, cpu_device):
        """exp(phi) @ exp(-phi) = I for each token."""
        G = _make_so3_generators(cpu_device)
        phi = torch.randn(1, 3, 3, device=cpu_device) * 0.5
        result = compute_transport_operators(phi, G, gauge_mode='learned')
        product = result['exp_phi'] @ result['exp_neg_phi']
        I = torch.eye(3, device=cpu_device)
        assert torch.allclose(product, I.expand_as(product), atol=1e-4), \
            f"exp(phi)exp(-phi) not I: max dev {(product - I).abs().max():.2e}"

    def test_so3_orthogonality(self, cpu_device):
        """For skew-symmetric generators, exp(phi) is orthogonal: R^T R = I."""
        G = _make_so3_generators(cpu_device)
        phi = torch.randn(2, 4, 3, device=cpu_device) * 0.5
        result = compute_transport_operators(phi, G, gauge_mode='learned')
        exp_phi = result['exp_phi']
        product = exp_phi.transpose(-1, -2) @ exp_phi
        I = torch.eye(3, device=cpu_device)
        assert torch.allclose(product, I.expand_as(product), atol=1e-4), \
            f"Not orthogonal: max dev {(product - I).abs().max():.2e}"

    def test_glk_positive_determinant(self, cpu_device):
        """For GL(K) generators, det(exp(phi)) > 0."""
        K = 3
        G = _make_glk_generators(K, cpu_device)
        phi = torch.randn(2, 4, K * K, device=cpu_device) * 0.2
        result = compute_transport_operators(phi, G, gauge_mode='learned')
        dets = torch.linalg.det(result['exp_phi'])
        assert (dets > 0).all(), f"Negative determinants: {dets[dets <= 0]}"

    def test_cocycle_property_flat(self, cpu_device):
        """Omega_ij @ Omega_jk approx Omega_ik for flat connection."""
        G = _make_so3_generators(cpu_device)
        phi = torch.randn(1, 4, 3, device=cpu_device) * 0.3
        result = compute_transport_operators(phi, G, gauge_mode='learned')
        Omega = result['Omega']
        # Pick i=0, j=1, k=2
        lhs = Omega[0, 0, 1] @ Omega[0, 1, 2]  # Omega_01 @ Omega_12
        rhs = Omega[0, 0, 2]  # Omega_02
        assert torch.allclose(lhs, rhs, atol=1e-3), \
            f"Cocycle violation: max dev {(lhs - rhs).abs().max():.2e}"

    def test_output_shapes(self, cpu_device):
        """Output dict has correct shapes."""
        G = _make_so3_generators(cpu_device)
        B, N, K = 2, 5, 3
        phi = torch.randn(B, N, 3, device=cpu_device)
        result = compute_transport_operators(phi, G, gauge_mode='learned')
        assert result['exp_phi'].shape == (B, N, K, K)
        assert result['exp_neg_phi'].shape == (B, N, K, K)
        assert result['Omega'].shape == (B, N, N, K, K)

    def test_output_finite(self, cpu_device):
        """No NaN/Inf in output."""
        G = _make_so3_generators(cpu_device)
        phi = torch.randn(2, 4, 3, device=cpu_device) * 0.5
        result = compute_transport_operators(phi, G, gauge_mode='learned')
        assert torch.isfinite(result['Omega']).all()
        assert torch.isfinite(result['exp_phi']).all()

    def test_non_flat_transport_differs(self, cpu_device):
        """With connection_delta, Omega differs from flat transport."""
        G = _make_so3_generators(cpu_device)
        B, N = 1, 3
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.3
        delta = torch.randn(B, N, N, 3, device=cpu_device) * 0.5

        flat = compute_transport_operators(phi, G, gauge_mode='learned')
        non_flat = compute_transport_operators(
            phi, G, gauge_mode='learned',
            connection_delta=delta, cocycle_relaxation=1.0
        )
        # Should differ (non-zero delta breaks cocycle)
        diff = (flat['Omega'] - non_flat['Omega']).abs().max()
        assert diff > 1e-3, "Non-flat should differ from flat transport"


# =============================================================================
# TestComputeTransportOperatorsDirect
# =============================================================================

class TestComputeTransportOperatorsDirect:
    """compute_transport_operators_direct: direct GL(K) group elements (no matrix_exp)."""

    def test_trivial_gauge(self, cpu_device):
        """Trivial gauge returns identity."""
        K = 3
        omega = torch.randn(2, 4, K, K, device=cpu_device)
        result = compute_transport_operators_direct(omega, gauge_mode='trivial')
        I = torch.eye(K, device=cpu_device)
        assert torch.allclose(result['Omega'], I.expand(2, 4, 4, K, K), atol=1e-6)

    def test_flat_cocycle(self, cpu_device):
        """Flat: Omega_ij = omega_i @ omega_j^{-1} satisfies cocycle."""
        K = 3
        B, N = 1, 4
        A = torch.randn(B, N, K, K, device=cpu_device) * 0.3
        omega = torch.eye(K, device=cpu_device) + A  # near identity for well-conditioned inv
        result = compute_transport_operators_direct(omega, gauge_mode='learned')
        Omega = result['Omega']
        # Cocycle: Omega_01 @ Omega_12 = Omega_02
        lhs = Omega[0, 0, 1] @ Omega[0, 1, 2]
        rhs = Omega[0, 0, 2]
        assert torch.allclose(lhs, rhs, atol=1e-3), \
            f"Direct cocycle violation: {(lhs - rhs).abs().max():.2e}"

    def test_self_transport_identity(self, cpu_device):
        """Omega_ii = omega_i @ omega_i^{-1} = I."""
        K = 3
        A = torch.randn(1, 3, K, K, device=cpu_device) * 0.3
        omega = torch.eye(K, device=cpu_device) + A
        result = compute_transport_operators_direct(omega, gauge_mode='learned')
        I = torch.eye(K, device=cpu_device)
        for i in range(3):
            assert torch.allclose(result['Omega'][0, i, i], I, atol=1e-4)

    def test_output_shapes(self, cpu_device):
        """Correct output shapes."""
        K = 4
        B, N = 2, 5
        omega = torch.eye(K, device=cpu_device).expand(B, N, K, K).clone()
        result = compute_transport_operators_direct(omega, gauge_mode='learned')
        assert result['Omega'].shape == (B, N, N, K, K)
        assert result['omega_i'].shape == (B, N, K, K)
        assert result['omega_j_inv'].shape == (B, N, K, K)


# =============================================================================
# TestOmegaToBlockExpPairs
# =============================================================================

class TestOmegaToBlockExpPairs:
    """omega_to_block_exp_pairs: convert direct Omega to per-block (fwd, inv) pairs."""

    def test_correct_block_count(self, cpu_device):
        """Returns one pair per irrep block."""
        K = 6
        omega = torch.eye(K, device=cpu_device).unsqueeze(0).unsqueeze(0).expand(2, 4, K, K).contiguous()
        irrep_dims = [3, 2, 1]
        pairs = omega_to_block_exp_pairs(omega, irrep_dims)
        assert len(pairs) == 3

    def test_block_shapes(self, cpu_device):
        """Each pair has shape (B, N, d, d) for its block dimension."""
        K = 6
        B, N = 2, 4
        omega = torch.eye(K, device=cpu_device).expand(B, N, K, K).contiguous()
        irrep_dims = [3, 2, 1]
        pairs = omega_to_block_exp_pairs(omega, irrep_dims)
        assert pairs[0][0].shape == (B, N, 3, 3)
        assert pairs[1][0].shape == (B, N, 2, 2)
        assert pairs[2][0].shape == (B, N, 1, 1)

    def test_identity_blocks(self, cpu_device):
        """Identity omega gives identity blocks."""
        K = 5
        omega = torch.eye(K, device=cpu_device).unsqueeze(0).unsqueeze(0).expand(1, 2, K, K).contiguous()
        irrep_dims = [3, 2]
        pairs = omega_to_block_exp_pairs(omega, irrep_dims)
        I3 = torch.eye(3, device=cpu_device)
        I2 = torch.eye(2, device=cpu_device)
        assert torch.allclose(pairs[0][0][0, 0], I3, atol=1e-5)
        assert torch.allclose(pairs[1][0][0, 0], I2, atol=1e-5)

    def test_inverse_correctness(self, cpu_device):
        """omega_block @ omega_block_inv = I."""
        K = 4
        A = torch.randn(1, 2, K, K, device=cpu_device) * 0.2
        omega = torch.eye(K, device=cpu_device) + A
        irrep_dims = [2, 2]
        pairs = omega_to_block_exp_pairs(omega, irrep_dims)
        for fwd, inv in pairs:
            product = fwd @ inv
            I = torch.eye(fwd.shape[-1], device=cpu_device)
            assert torch.allclose(product, I.expand_as(product), atol=1e-3), \
                f"Block inverse error: {(product - I).abs().max():.2e}"


# =============================================================================
# TestRoPE
# =============================================================================

class TestRoPE:
    """RoPE (Rotary Position Embeddings) for belief means."""

    def test_shape_preservation(self, cpu_device):
        """Output has same shape as input."""
        mu = torch.randn(2, 8, 6, device=cpu_device)
        result = _apply_rope(mu)
        assert result.shape == mu.shape

    def test_norm_preservation(self, cpu_device):
        """RoPE preserves per-pair L2 norm: SO(2) rotation on each pair."""
        mu = torch.randn(2, 8, 6, device=cpu_device)
        result = _apply_rope(mu)
        # Compute norm per pair of dimensions
        K = 6
        half_K = K // 2
        for p in range(half_K):
            pair_orig = mu[:, :, 2*p:2*p+2]
            pair_rot = result[:, :, 2*p:2*p+2]
            norm_orig = torch.norm(pair_orig, dim=-1)
            norm_rot = torch.norm(pair_rot, dim=-1)
            assert torch.allclose(norm_orig, norm_rot, atol=1e-5), \
                f"Pair {p}: norm not preserved, max dev {(norm_orig - norm_rot).abs().max():.2e}"

    def test_position_sensitivity(self, cpu_device):
        """Different positions produce different rotations."""
        # Same content at different positions should differ after RoPE
        mu = torch.ones(1, 4, 6, device=cpu_device)
        result = _apply_rope(mu)
        # Position 0 and position 3 should differ
        assert not torch.allclose(result[0, 0], result[0, 3], atol=1e-3), \
            "Different positions should produce different RoPE rotations"

    def test_output_finite(self, cpu_device):
        """No NaN/Inf."""
        mu = torch.randn(2, 16, 8, device=cpu_device)
        result = _apply_rope(mu)
        assert torch.isfinite(result).all()

    def test_odd_k_last_dim_unchanged(self, cpu_device):
        """For odd K, the last dimension is unchanged (no pair partner)."""
        mu = torch.randn(1, 4, 5, device=cpu_device)  # K=5 (odd)
        result = _apply_rope(mu)
        # Last dimension (index 4) should be unchanged
        assert torch.allclose(result[:, :, 4], mu[:, :, 4], atol=1e-6), \
            "Odd K: last dimension should be unchanged by RoPE"


# =============================================================================
# TestBuildRopeFreqs
# =============================================================================

class TestBuildRopeFreqs:
    """_build_rope_freqs: frequency band computation for RoPE."""

    def test_shape(self, cpu_device):
        """Returns (K//2,) tensor."""
        freqs = _build_rope_freqs(8, device=cpu_device)
        assert freqs.shape == (4,)

    def test_decreasing(self, cpu_device):
        """Frequencies decrease (higher dims rotate slower)."""
        freqs = _build_rope_freqs(8, device=cpu_device)
        diffs = freqs[1:] - freqs[:-1]
        assert (diffs <= 0).all(), "Frequencies should be decreasing"

    def test_first_freq(self, cpu_device):
        """First frequency is 1.0 (1/base^0 = 1)."""
        freqs = _build_rope_freqs(8, base=10000.0, device=cpu_device)
        assert abs(freqs[0].item() - 1.0) < 1e-5
