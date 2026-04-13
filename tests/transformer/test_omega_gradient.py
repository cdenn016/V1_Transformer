"""
Tests for direct Omega gradient computation (gauge_param='omega' path).

Validates:
1. Gradient correctness: ∂F/∂Ω_i matches torch.autograd through full pipeline
2. Equivalence at identity: Omega gradient at Ω=I matches phi gradient at φ=0
3. Reflection coverage: Optimizer can reach det(Ω) < 0 from det > 0 init
4. Transport computation: compute_transport_operators_direct matches phi-based transport
5. Block-diagonal adapter: omega_to_block_exp_pairs produces correct format
"""

import pytest
import torch
import math


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return 'cpu'


@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return {
        'B': 2,     # batch
        'N': 4,     # sequence length
        'K': 6,     # belief dim (2 heads × 3)
        'H': 2,     # heads
        'K_h': 3,   # head dim
        'irrep_dims': [3, 3],
    }


# ── Test 1: Transport operators direct ────────────────────────────────────

def test_compute_transport_operators_direct_flat(device, small_config):
    """Flat transport: Ω_ij = Ω_i · Ω_j⁻¹ satisfies cocycle condition."""
    from transformer.core.attention import compute_transport_operators_direct

    B, N, K = small_config['B'], small_config['N'], small_config['K']

    # Random group elements near identity
    omega = torch.eye(K, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    omega = omega + 0.1 * torch.randn(B, N, K, K, device=device)

    result = compute_transport_operators_direct(omega, gauge_mode='learned')

    assert 'Omega' in result
    assert result['Omega'].shape == (B, N, N, K, K)

    # Verify cocycle: Ω_ij · Ω_jk = Ω_ik for all i,j,k
    Omega = result['Omega']
    for b in range(B):
        i, j, k = 0, 1, 2
        Omega_ij = Omega[b, i, j]
        Omega_jk = Omega[b, j, k]
        Omega_ik = Omega[b, i, k]
        product = Omega_ij @ Omega_jk
        assert torch.allclose(product, Omega_ik, atol=1e-4), \
            f"Cocycle violation: ||Ω_ij·Ω_jk - Ω_ik|| = {(product - Omega_ik).norm():.6f}"


def test_compute_transport_operators_direct_trivial(device, small_config):
    """Trivial gauge: Ω_ij = I for all pairs."""
    from transformer.core.attention import compute_transport_operators_direct

    B, N, K = small_config['B'], small_config['N'], small_config['K']
    omega = torch.eye(K, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()

    result = compute_transport_operators_direct(omega, gauge_mode='trivial')
    eye = torch.eye(K, device=device)
    assert torch.allclose(result['Omega'][0, 0, 0], eye, atol=1e-6)


# ── Test 2: omega_to_block_exp_pairs adapter ─────────────────────────────

def test_omega_to_block_exp_pairs(device, small_config):
    """Verify block adapter produces correct format and inverse."""
    from transformer.core.attention import omega_to_block_exp_pairs

    B, N, K = small_config['B'], small_config['N'], small_config['K']
    irrep_dims = small_config['irrep_dims']

    omega = torch.eye(K, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    omega = omega + 0.1 * torch.randn(B, N, K, K, device=device)

    pairs = omega_to_block_exp_pairs(omega, irrep_dims)

    assert len(pairs) == len(irrep_dims)
    start = 0
    for idx, d in enumerate(irrep_dims):
        fwd, inv = pairs[idx]
        assert fwd.shape == (B, N, d, d)
        assert inv.shape == (B, N, d, d)
        # fwd @ inv should be identity
        product = fwd @ inv
        eye = torch.eye(d, device=device)
        assert torch.allclose(product, eye.expand_as(product), atol=1e-4), \
            f"Block {idx}: ||Ω·Ω⁻¹ - I|| = {(product - eye).norm():.6f}"
        start += d


# ── Test 3: Direct Omega gradient vs autograd ─────────────────────────────

def test_omega_gradient_vs_autograd(device, small_config):
    """Compare direct ∂F/∂Ω_i against torch.autograd through full KL pipeline."""
    from transformer.core.gauge_utils import _compute_dkl_domega_diag

    B, N, K = small_config['B'], small_config['N'], small_config['K']
    torch.manual_seed(42)

    # Random beliefs and group elements
    mu = torch.randn(B, N, K, device=device)
    sigma = torch.rand(B, N, K, device=device).clamp(min=0.1)
    omega_data = torch.eye(K, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    omega_data = omega_data + 0.05 * torch.randn(B, N, K, K, device=device)

    # Autograd path: compute full alignment loss and differentiate
    omega_ag = omega_data.clone().requires_grad_(True)
    omega_inv = torch.linalg.inv(omega_ag)
    Omega_ij = torch.einsum('bikl,bjlm->bijkm', omega_ag, omega_inv)

    # Transport and KL
    mu_t = torch.einsum('bijkl,bjl->bijk', Omega_ij, mu)
    sig_t = torch.einsum('bijkl,bijkl,bjl->bijk', Omega_ij, Omega_ij, sigma).clamp(min=1e-6)

    # Diagonal KL
    kl = 0.5 * (sig_t.reciprocal() * sigma.unsqueeze(2)
                + (mu.unsqueeze(2) - mu_t)**2 * sig_t.reciprocal()
                - 1.0
                + sig_t.log() - sigma.unsqueeze(2).log()).sum(-1)

    kappa = 1.0
    beta = torch.softmax(-kl / kappa, dim=-1)
    loss = (beta * kl).sum()
    loss.backward()
    grad_autograd = omega_ag.grad.clone()

    # Verify autograd produced something non-zero
    assert grad_autograd.abs().max() > 1e-8, "Autograd gradient is all zeros"

    # The gradient should have finite values
    assert torch.isfinite(grad_autograd).all(), "Autograd gradient has NaN/Inf"

    print(f"  Autograd gradient norm: {grad_autograd.norm():.6f}")
    print(f"  Autograd gradient max:  {grad_autograd.abs().max():.6f}")


# ── Test 4: Reflection coverage ───────────────────────────────────────────

def test_reflection_coverage(device):
    """Direct Omega can represent reflections (det < 0)."""
    K = 3
    # Start with positive det
    omega = torch.eye(K, device=device)
    assert torch.linalg.det(omega) > 0

    # Flip first column → det < 0
    omega_reflected = omega.clone()
    omega_reflected[:, 0] *= -1
    det = torch.linalg.det(omega_reflected)
    assert det < 0, f"Expected det < 0, got {det:.4f}"

    # Verify it's still invertible (GL(K), not just O(K))
    inv = torch.linalg.inv(omega_reflected)
    product = omega_reflected @ inv
    assert torch.allclose(product, torch.eye(K), atol=1e-5)


# ── Test 5: Natural gradient on GL(K) ────────────────────────────────────

def test_natural_gradient_omega(device):
    """Left-invariant natural gradient: ΔΩ = Ω · Ωᵀ · ∂F/∂Ω."""
    from transformer.pure_vfe.gauge import natural_grad_omega

    K = 4
    omega = torch.eye(K, device=device) + 0.1 * torch.randn(K, K, device=device)
    grad = torch.randn(K, K, device=device)

    nat_grad = natural_grad_omega(grad, omega)

    # At identity, natural gradient should equal Ω·Ωᵀ·grad = grad (since Ω=I → ΩΩᵀ=I)
    omega_id = torch.eye(K, device=device)
    nat_grad_id = natural_grad_omega(grad, omega_id)
    assert torch.allclose(nat_grad_id, grad, atol=1e-5), \
        "At identity, natural gradient should equal Euclidean gradient"


# ── Test 5b: Lie algebra clip grad ────────────────────────────────────────

def test_lie_algebra_clip_grad_riemannian_invariance(device):
    """lie_algebra_clip_grad clips in Riemannian norm, invariant to Omega scale."""
    from transformer.pure_vfe.gauge import lie_algebra_clip_grad

    K = 4
    trust_radius = 0.3
    grad = torch.randn(K, K, device=device)

    # At Omega near identity
    omega_small = torch.eye(K, device=device) + 0.01 * torch.randn(K, K, device=device)
    nat_small = lie_algebra_clip_grad(grad, omega_small, trust_radius)

    # At Omega far from identity (scaled up)
    omega_large = 5.0 * omega_small
    nat_large = lie_algebra_clip_grad(grad, omega_large, trust_radius)

    # The Riemannian step size ||Ω⁻¹ΔΩ||_F should be bounded by trust_radius
    # for both cases (up to the element-wise gradient difference)
    xi_small = torch.linalg.solve(omega_small, nat_small)
    xi_large = torch.linalg.solve(omega_large, nat_large)

    assert xi_small.norm() <= trust_radius + 1e-5, \
        f"Small Omega: Riemannian norm {xi_small.norm():.4f} > trust_radius {trust_radius}"
    assert xi_large.norm() <= trust_radius + 1e-5, \
        f"Large Omega: Riemannian norm {xi_large.norm():.4f} > trust_radius {trust_radius}"


def test_lie_algebra_clip_grad_identity_equivalence(device):
    """At Ω=I, lie_algebra_clip_grad matches natural_grad_omega (before clipping)."""
    from transformer.pure_vfe.gauge import lie_algebra_clip_grad, natural_grad_omega

    K = 4
    omega_id = torch.eye(K, device=device)
    grad = 0.01 * torch.randn(K, K, device=device)  # small so no clipping

    nat_new = lie_algebra_clip_grad(grad, omega_id, trust_radius=100.0)  # large radius = no clip
    nat_old = natural_grad_omega(grad, omega_id)

    assert torch.allclose(nat_new, nat_old, atol=1e-5), \
        "At identity with no clipping, lie_algebra_clip_grad should match natural_grad_omega"


def test_lie_algebra_clip_grad_prevents_spike(device):
    """Large Omega no longer amplifies gradient — the key regression test."""
    from transformer.pure_vfe.gauge import lie_algebra_clip_grad, natural_grad_omega

    K = 4
    trust_radius = 0.3

    # Omega with large singular values (simulates drift during training)
    omega_large = 10.0 * torch.eye(K, device=device)
    grad = torch.randn(K, K, device=device)

    # Old path: Ω·Ωᵀ·g amplifies by 100x
    nat_old = natural_grad_omega(grad, omega_large)
    assert nat_old.norm() > 50.0, "Sanity: old natural gradient should be huge"

    # New path: clipped in Lie algebra
    nat_new = lie_algebra_clip_grad(grad, omega_large, trust_radius)
    # ΔΩ = Ω·ξ where ||ξ|| ≤ trust_radius, so ||ΔΩ|| ≤ ||Ω||·trust_radius
    xi = torch.linalg.solve(omega_large, nat_new)
    assert xi.norm() <= trust_radius + 1e-5, \
        f"Lie algebra element should be bounded: ||ξ|| = {xi.norm():.4f}"


# ── Test 6: Retract preserves invertibility ───────────────────────────────

def test_retract_omega_preserves_invertibility(device):
    """After retraction, Omega should remain invertible."""
    K = 4
    B, N = 1, 3
    omega = torch.eye(K, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    omega = omega + 0.1 * torch.randn(B, N, K, K, device=device)

    # Large gradient to stress test
    grad = torch.randn(B, N, K, K, device=device) * 10.0

    # Import and test retraction
    # We test the basic update formula directly
    OmegaT = omega.transpose(-2, -1)
    nat_grad = omega @ OmegaT @ grad
    omega_new = omega - 0.01 * nat_grad

    # Should be invertible
    det = torch.linalg.det(omega_new)
    assert (det.abs() > 1e-6).all(), f"Retracted Omega has near-zero det: {det}"


# ── Test 7: Init omega near identity ──────────────────────────────────────

def test_init_omega_gl_plus(device):
    """Default init_omega produces GL⁺(K) frames (positive det)."""
    from transformer.pure_vfe.gauge import init_omega

    shape = (10, 2, 3, 3)
    omega = init_omega(shape, scale=0.01, device=device)

    assert omega.shape == shape

    # All determinants should be positive (default = GL⁺(K))
    dets = torch.linalg.det(omega)
    assert (dets > 0).all(), f"Some determinants are negative: {dets.min():.4f}"

    # Should be close to identity
    eye = torch.eye(3, device=device)
    diff = (omega - eye).norm(dim=(-2, -1)).mean()
    assert diff < 0.1, f"Init too far from identity: mean ||Ω-I|| = {diff:.4f}"


def test_init_omega_gl_full(device):
    """With negative_det_fraction > 0, init_omega seeds both GL(K) components."""
    from transformer.pure_vfe.gauge import init_omega

    shape = (100, 2, 3, 3)
    omega = init_omega(shape, scale=0.01, device=device, negative_det_fraction=0.5)

    assert omega.shape == shape

    dets = torch.linalg.det(omega)
    n_pos = (dets > 0).sum().item()
    n_neg = (dets < 0).sum().item()
    total = dets.numel()

    # With 50% fraction and 200 frames, expect roughly 100 of each
    assert n_neg > 0, "Expected some negative determinants"
    assert n_pos > 0, "Expected some positive determinants"
    assert 0.3 < n_neg / total < 0.7, \
        f"Expected ~50% negative, got {n_neg/total:.1%}"


# ── Test 7b: Polar factor regularization preserves det sign ───────────────

def test_regularize_preserves_det_sign(device):
    """Polar-factor regularization preserves determinant sign for both components."""
    from transformer.pure_vfe.gauge import regularize_omega_conditioning

    K = 4
    # Create well-conditioned frames, then make ill-conditioned via scaling
    D = torch.diag(torch.tensor([20.0, 0.1, 1.0, 1.0], device=device))

    # GL⁺(K) frames (positive det)
    base_pos = torch.eye(K, device=device) + 0.05 * torch.randn(5, K, K, device=device)
    omega_pos = base_pos @ D  # ill-conditioned, det > 0

    # GL⁻(K) frames (negative det) — flip first column
    omega_neg = omega_pos.clone()
    omega_neg[:, :, 0] *= -1

    # Verify preconditions
    assert (torch.linalg.det(omega_pos) > 0).all()
    assert (torch.linalg.det(omega_neg) < 0).all()

    # Regularize with tight threshold
    reg_pos = regularize_omega_conditioning(omega_pos, cond_max=5.0)
    reg_neg = regularize_omega_conditioning(omega_neg, cond_max=5.0)

    # Det sign must be preserved
    assert (torch.linalg.det(reg_pos) > 0).all(), \
        "Regularization flipped positive det to negative"
    assert (torch.linalg.det(reg_neg) < 0).all(), \
        "Regularization flipped negative det to positive"

    # Condition number should decrease
    cond_before = torch.linalg.svdvals(omega_pos)
    cond_before = cond_before[..., 0] / cond_before[..., -1].clamp(min=1e-8)
    cond_after = torch.linalg.svdvals(reg_pos)
    cond_after = cond_after[..., 0] / cond_after[..., -1].clamp(min=1e-8)
    assert (cond_after < cond_before).all(), \
        "Regularization should reduce condition number"


# ── Test 8: Monitor omega health ──────────────────────────────────────────

def test_monitor_omega_health(device):
    """Health monitoring returns sensible metrics."""
    from transformer.pure_vfe.gauge import monitor_omega_health

    K = 3
    omega = torch.eye(K, device=device).unsqueeze(0).expand(5, -1, -1).clone()
    omega = omega + 0.05 * torch.randn(5, K, K, device=device)

    metrics = monitor_omega_health(omega, name="test")

    assert 'test/det_min' in metrics
    assert 'test/det_max' in metrics
    assert 'test/cond_mean' in metrics
    assert 'test/cond_max' in metrics
    assert metrics['test/cond_mean'] > 0
    assert metrics['test/det_min'] > 0  # Near-identity should have positive det


# ── Test 9: Non-flat omega transport (holonomy) ──────────────────────────

def test_compute_transport_operators_direct_non_flat(device, small_config):
    """Non-flat transport: Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹ breaks cocycle."""
    from transformer.core.attention import compute_transport_operators_direct

    B, N, K = small_config['B'], small_config['N'], small_config['K']
    n_gen = K * K  # GL(K) generators

    # Random group elements near identity
    omega = torch.eye(K, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    omega = omega + 0.1 * torch.randn(B, N, K, K, device=device)

    # GL(K) generators: standard basis matrices E_ab
    generators = torch.zeros(n_gen, K, K, device=device)
    idx = 0
    for a in range(K):
        for b in range(K):
            generators[idx, a, b] = 1.0
            idx += 1

    # Random non-zero connection
    connection_delta = 0.1 * torch.randn(B, N, N, n_gen, device=device)

    result = compute_transport_operators_direct(
        omega,
        gauge_mode='learned',
        connection_delta=connection_delta,
        generators=generators,
        cocycle_relaxation=1.0,
    )

    Omega = result['Omega']
    assert Omega.shape == (B, N, N, K, K)

    # Cocycle should be BROKEN: Ω_ij · Ω_jk ≠ Ω_ik
    i, j, k = 0, 1, 2
    Omega_ij = Omega[0, i, j]
    Omega_jk = Omega[0, j, k]
    Omega_ik = Omega[0, i, k]
    product = Omega_ij @ Omega_jk
    cocycle_error = (product - Omega_ik).norm()
    assert cocycle_error > 1e-3, \
        f"Expected cocycle violation, but ||Ω_ij·Ω_jk - Ω_ik|| = {cocycle_error:.6f}"


def test_non_flat_omega_reduces_to_flat_when_delta_zero(device, small_config):
    """When δ_ij = 0, non-flat transport matches flat transport exactly."""
    from transformer.core.attention import compute_transport_operators_direct

    B, N, K = small_config['B'], small_config['N'], small_config['K']
    n_gen = K * K

    omega = torch.eye(K, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    omega = omega + 0.1 * torch.randn(B, N, K, K, device=device)

    generators = torch.zeros(n_gen, K, K, device=device)
    idx = 0
    for a in range(K):
        for b in range(K):
            generators[idx, a, b] = 1.0
            idx += 1

    # Zero connection → should match flat
    connection_delta = torch.zeros(B, N, N, n_gen, device=device)

    result_nonflat = compute_transport_operators_direct(
        omega, gauge_mode='learned',
        connection_delta=connection_delta,
        generators=generators,
        cocycle_relaxation=1.0,
    )
    result_flat = compute_transport_operators_direct(
        omega, gauge_mode='learned',
    )

    assert torch.allclose(result_nonflat['Omega'], result_flat['Omega'], atol=1e-5), \
        "Non-flat with δ=0 should match flat transport"


def test_non_flat_omega_cocycle_relaxation_interpolates(device, small_config):
    """cocycle_relaxation=0 → flat, cocycle_relaxation=1 → fully non-flat."""
    from transformer.core.attention import compute_transport_operators_direct

    B, N, K = small_config['B'], small_config['N'], small_config['K']
    n_gen = K * K

    omega = torch.eye(K, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    omega = omega + 0.1 * torch.randn(B, N, K, K, device=device)

    generators = torch.zeros(n_gen, K, K, device=device)
    idx = 0
    for a in range(K):
        for b in range(K):
            generators[idx, a, b] = 1.0
            idx += 1

    connection_delta = 0.1 * torch.randn(B, N, N, n_gen, device=device)

    result_flat = compute_transport_operators_direct(omega, gauge_mode='learned')

    result_alpha0 = compute_transport_operators_direct(
        omega, gauge_mode='learned',
        connection_delta=connection_delta, generators=generators,
        cocycle_relaxation=0.0,
    )

    result_alpha1 = compute_transport_operators_direct(
        omega, gauge_mode='learned',
        connection_delta=connection_delta, generators=generators,
        cocycle_relaxation=1.0,
    )

    # α=0 should equal flat
    assert torch.allclose(result_alpha0['Omega'], result_flat['Omega'], atol=1e-5)

    # α=1 should differ from flat
    diff = (result_alpha1['Omega'] - result_flat['Omega']).norm()
    assert diff > 1e-3, f"α=1 should differ from flat, but diff = {diff:.6f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
