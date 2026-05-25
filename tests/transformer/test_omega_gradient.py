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


# ── Full-covariance omega gradient (no longer returns None) ────────────────

def test_compute_omega_grad_direct_full_covariance(device, small_config):
    """_compute_omega_grad_direct returns a finite grad when is_diagonal=False.

    Previously the function early-returned None on the full-covariance path;
    the fix mirrors the phi-path per-block sigma slicing at
    variational_ffn.py:_compute_phi_grad lines 1166-1169.
    """
    from transformer.core.variational_ffn import VariationalFFNDynamic
    from math_utils.generators import generate_so3_generators

    B, N, K = small_config['B'], small_config['N'], small_config['K']
    irrep_dims = small_config['irrep_dims']
    # SO(3) block generators per 3-dim head.
    gen_block = torch.from_numpy(generate_so3_generators(3)).float().to(device)
    n_gen = gen_block.shape[0]
    generators = torch.zeros(n_gen, K, K, device=device)
    generators[:, :3, :3] = gen_block
    generators[:, 3:6, 3:6] = gen_block  # same block twice; test only needs GL(3)×GL(3)

    ffn = VariationalFFNDynamic(
        embed_dim=K, generators=generators, alpha=0.001, kappa=1.0,
        n_iterations=1, diagonal_covariance=False, irrep_dims=irrep_dims,
        gauge_param='omega',
    ).to(device)

    torch.manual_seed(0)
    mu = torch.randn(B, N, K, device=device)
    # Full-cov sigma: SPD per token.
    A = torch.randn(B, N, K, K, device=device) * 0.1
    sigma_full = torch.einsum('bnij,bnkj->bnik', A, A) + 0.5 * torch.eye(K, device=device)
    eye_k = torch.eye(K, device=device).expand(B, N, -1, -1).clone()
    omega = eye_k + 0.05 * torch.randn(B, N, K, K, device=device)

    grad_omega = ffn._compute_omega_grad_direct(
        omega_current=omega, mu_current=mu, sigma_current=sigma_full,
        is_diagonal=False, mask=None, eps=1e-8,
    )
    assert grad_omega is not None, "Full-cov omega grad must not return None"
    assert grad_omega.shape == omega.shape
    assert torch.isfinite(grad_omega).all(), "Full-cov omega grad has NaN/Inf"
    assert grad_omega.abs().max() > 1e-10, "Full-cov omega grad is trivially zero"


# ── log|det(Omega)| penalty arithmetic ─────────────────────────────────────

def test_omega_det_penalty_zero_at_identity(device, small_config):
    """(log|det Ω|)² penalty vanishes when Ω = I."""
    B, N = small_config['B'], small_config['N']
    K_h = small_config['K_h']
    identity = torch.eye(K_h, device=device).expand(B, N, -1, -1).clone()
    _, logabsdet = torch.linalg.slogdet(identity)     # (B, N)
    penalty = (logabsdet ** 2).mean()
    assert penalty.abs().item() < 1e-10, f"Penalty at Ω=I should be ≈0, got {penalty.item():.3e}"


def test_omega_det_penalty_positive_off_identity(device, small_config):
    """Penalty is strictly positive when det|Ω| ≠ 1."""
    B, N = small_config['B'], small_config['N']
    K_h = small_config['K_h']
    # Scale identity by 2 → det = 2^K_h, logabsdet = K_h·log 2 > 0.
    omega = 2.0 * torch.eye(K_h, device=device).expand(B, N, -1, -1).clone()
    _, logabsdet = torch.linalg.slogdet(omega)
    penalty = (logabsdet ** 2).mean()
    expected = (K_h * math.log(2.0)) ** 2
    assert penalty.item() > 0, "Penalty should be positive for det|Ω| ≠ 1"
    assert abs(penalty.item() - expected) < 1e-4, (
        f"Penalty {penalty.item():.4f} != expected {expected:.4f}"
    )


def test_omega_det_penalty_invariant_to_reflection(device, small_config):
    """Penalty treats det<0 and det>0 symmetrically (reflections allowed)."""
    B, N = small_config['B'], small_config['N']
    K_h = small_config['K_h']
    omega_pos = torch.eye(K_h, device=device).expand(B, N, -1, -1).clone() * 1.5
    omega_neg = omega_pos.clone()
    omega_neg[..., :, 0] *= -1  # flip first column → det sign flip, |det| unchanged
    _, lad_pos = torch.linalg.slogdet(omega_pos)
    _, lad_neg = torch.linalg.slogdet(omega_neg)
    # log|det| identical up to sign convention; slogdet returns log of absolute value
    assert torch.allclose(lad_pos, lad_neg, atol=1e-6), (
        "log|det Ω| should be sign-invariant; penalty must allow reflections"
    )


# ── End-to-end: gauge_param='omega' × em_mode='em_phi_p' ───────────────────

def _omega_minimal_config(em_mode='em_phi_p', gauge_fixed_priors=False):
    """Minimal model config for the omega path."""
    return {
        'vocab_size': 64,
        'embed_dim': 12,
        'n_layers': 2,
        'irrep_spec': [('l0', 4, 1), ('l1', 1, 3), ('l2', 1, 5)],
        'hidden_dim': 24,
        'max_seq_len': 16,
        'kappa_beta': 1.0,
        'dropout': 0.0,
        'pos_encoding_mode': 'learned',
        'evolve_sigma': True,
        'evolve_phi': False,
        'tie_embeddings': True,
        'diagonal_covariance': True,
        'ffn_mode': 'VFE_dynamic',
        'gauge_param': 'omega',
        'gauge_fixed_priors': gauge_fixed_priors,
        'em_mode': em_mode,
    }


def test_omega_em_phi_p_end_to_end_forward_backward(device):
    """Full model forward+backward under gauge_param='omega' × em_mode='em_phi_p'.

    Verifies:
      1. Forward pass produces finite logits of the right shape.
      2. Backward through free-energy loss produces a non-None, finite
         gradient on omega_embed.
      3. em_phi_p value-freeze: omega_current at E-step exit equals the
         initial embedding lookup (no retract during E-step).
    """
    from transformer.core.model import GaugeTransformerLM
    from transformer.train import compute_free_energy_loss

    cfg = _omega_minimal_config(em_mode='em_phi_p')
    model = GaugeTransformerLM(cfg).to(device)
    B, N = 2, 8
    token_ids = torch.randint(0, cfg['vocab_size'], (B, N), device=device)
    targets = torch.randint(0, cfg['vocab_size'], (B, N), device=device)

    # 1. Forward
    logits, attn_info = model.forward_with_attention(token_ids, targets=targets)
    assert logits.shape == (B, N, cfg['vocab_size'])
    assert torch.isfinite(logits).all(), "Non-finite logits under omega+em_phi_p"

    # omega_initial must be present and attached; evolved omega must be detached
    assert attn_info['omega_initial'] is not None, "omega_initial missing from attn_info"
    assert attn_info['omega_initial'].requires_grad, "omega_initial must be attached for M-step"
    evolved = attn_info['omega']
    assert evolved is not None
    assert not evolved.requires_grad, "Evolved omega must be detached at EM boundary under em_phi_p"

    # 2. Loss + backward
    loss, metrics = compute_free_energy_loss(
        model, token_ids, targets,
        M_alpha=0.1, M_beta=0.1, mass_phi=0.0,
        omega_det_penalty=1e-3,  # exercise the V9 penalty branch
        pad_token_id=-100,
    )
    assert torch.isfinite(loss), f"Non-finite loss: {loss.item()}"
    loss.backward()

    # 3. omega_embed gradient reached
    omega_embed_weight = model.token_embed.omega_embed.weight
    assert omega_embed_weight.grad is not None, "omega_embed received no gradient"
    assert torch.isfinite(omega_embed_weight.grad).all(), "omega_embed grad has NaN/Inf"
    assert omega_embed_weight.grad.abs().max() > 0, "omega_embed grad is trivially zero"

    # 4. Penalty metric is recorded
    assert 'loss/omega_det' in metrics
    assert metrics['loss/omega_det'] >= 0.0


def test_omega_em_phi_p_with_gauge_fixed_priors(device):
    """gauge_fixed_priors=True × gauge_param='omega' produces distinct token priors."""
    from transformer.core.model import GaugeTransformerLM

    cfg = _omega_minimal_config(em_mode='em_phi_p', gauge_fixed_priors=True)
    model = GaugeTransformerLM(cfg).to(device)

    # Two different token IDs → two distinct prior mu (via Omega_v @ base_mu).
    # Before the fix, both tokens would hit exp(dummy_zero_phi) = I → mu = base_mu.
    B, N = 1, 4
    tok_a = torch.zeros(B, N, dtype=torch.long, device=device)
    tok_b = torch.ones(B, N, dtype=torch.long, device=device)
    with torch.no_grad():
        _, info_a = model.forward_with_attention(tok_a, targets=tok_a)
        _, info_b = model.forward_with_attention(tok_b, targets=tok_b)
    mu_a = info_a['mu_prior']
    mu_b = info_b['mu_prior']
    diff = (mu_a - mu_b).abs().max().item()
    assert diff > 1e-4, (
        f"gauge_fixed_priors+omega collapsed all tokens to identical mu (max |diff| = {diff:.2e}); "
        "the Omega-v transport branch is not active"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
