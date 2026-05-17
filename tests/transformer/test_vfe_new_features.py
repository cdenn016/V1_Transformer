"""
Regression tests for the four new /vfe features added 2026-05-17:

  Q3: use_prior_bank toggle
  Q4: equivariant head mixer (Schur-commutant)
  Q7: non-flat parallel transport (built from scratch, not ported)
  Q8: pure-omega gauge parameterization

The tests focus on the invariants that must hold for each feature:
  - Q3: with toggle off the model still trains; logits shape correct.
  - Q4: mixer at identity is a true no-op (bitwise) on (mu, sigma).
  - Q7: at strength=0 the path is mathematically equivalent to flat (small
        numerical drift is allowed); a non-zero strength produces non-trivial
        triangle holonomy; backward pass produces finite gradients.
  - Q8: omega-direct forward produces (Omega, Omega^{-1}) pairs with
        Omega · Omega^{-1} = I; backward produces finite gradients for the
        primary M-step parameters; project_slk renormalizes det to 1.
"""

import math

import pytest
import torch

from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel
from transformer.vfe.head_mixer import VFEHeadMixer
from transformer.vfe.non_flat import (
    VFENonFlatConnection,
    compute_pairwise_omega_with_delta,
    compute_kl_attention_pairwise,
    triangle_holonomy_norm,
)
from transformer.vfe.omega_direct import (
    init_omega_from_phi,
    compute_pairwise_omega_from_endpoints,
    project_omega_to_slk,
)


CFG_KWARGS = dict(
    vocab_size=256,
    embed_dim=20,
    irrep_spec=[('fund', 2, 10)],
    n_layers=1,
    max_seq_len=32,
    n_e_steps=1,
    diagonal_covariance=True,
    gauge_group='GLK',
    norm_type='mahalnorm',
    use_rope=False,
    mask_self_attention=False,
)


def _shared_state_copy(src: VFEModel, dst: VFEModel) -> None:
    """Copy bytewise the intersection of state dicts (skip new params)."""
    sa, sb = src.state_dict(), dst.state_dict()
    for k, v in sa.items():
        if k in sb and sb[k].shape == v.shape:
            sb[k].copy_(v)
    dst.load_state_dict(sb)


# ----------------------------------------------------------------------------
# Q3 — use_prior_bank toggle
# ----------------------------------------------------------------------------


def test_q3_prior_bank_off_has_output_proj():
    cfg = VFEConfig(**CFG_KWARGS, use_prior_bank=False)
    model = VFEModel(cfg)
    assert model.output_proj is not None
    assert isinstance(model.output_proj, torch.nn.Linear)
    assert model.output_proj.weight.shape == (cfg.vocab_size, cfg.embed_dim)
    assert model.output_proj.bias is None  # bias=False per design


def test_q3_prior_bank_on_has_no_output_proj():
    cfg = VFEConfig(**CFG_KWARGS, use_prior_bank=True)
    model = VFEModel(cfg)
    assert model.output_proj is None


def test_q3_prior_bank_off_forward_works():
    cfg = VFEConfig(**CFG_KWARGS, use_prior_bank=False)
    model = VFEModel(cfg).eval()
    tok = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits = model(tok)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()


# ----------------------------------------------------------------------------
# Q4 — Equivariant head mixer
# ----------------------------------------------------------------------------


def test_q4_mixer_at_identity_is_noop_bitwise():
    torch.manual_seed(0)
    mixer = VFEHeadMixer([('fund', 2, 10)], embed_dim=20)
    assert mixer.is_identity()

    mu = torch.randn(2, 8, 20)
    sigma = torch.rand(2, 8, 20).clamp(min=0.1)
    mu_out, sigma_out = mixer(mu, sigma)
    assert torch.equal(mu_out, mu)
    assert torch.equal(sigma_out, sigma)


def test_q4_mixer_active_changes_output():
    torch.manual_seed(0)
    mixer = VFEHeadMixer([('fund', 2, 10)], embed_dim=20)
    # Add a non-zero off-diagonal so the heads actually mix.
    with torch.no_grad():
        mixer.mixer_delta['fund'].add_(torch.tensor([[0.0, 0.3], [0.3, 0.0]]))
    assert not mixer.is_identity()
    mu = torch.randn(2, 8, 20)
    sigma = torch.rand(2, 8, 20).clamp(min=0.1)
    mu_out, sigma_out = mixer(mu, sigma)
    assert not torch.equal(mu_out, mu)
    assert not torch.equal(sigma_out, sigma)


def test_q4_mixer_in_block_at_identity_matches_no_mixer():
    """With shared params + mixer at identity, the model output is bitwise
    equal to the no-mixer model. Confirms the mixer is autograd-clean and
    truly identity at init."""
    torch.manual_seed(0)
    base = VFEModel(VFEConfig(**CFG_KWARGS)).eval()
    torch.manual_seed(0)
    with_mixer = VFEModel(VFEConfig(**CFG_KWARGS, use_equivariant_head_mixer=True)).eval()
    _shared_state_copy(base, with_mixer)

    tok = torch.randint(0, CFG_KWARGS['vocab_size'], (2, 16))
    with torch.no_grad():
        l_b = base(tok)
        l_m = with_mixer(tok)
    assert torch.equal(l_b, l_m), \
        f"Mixer-at-identity should be bitwise no-op; got max |Δ| = {(l_b - l_m).abs().max().item()}"


# ----------------------------------------------------------------------------
# Q7 — Non-flat parallel transport
# ----------------------------------------------------------------------------


def test_q7_connection_init_is_zero():
    """Strength and W start at zero so delta is exactly zero."""
    from math_utils.generators import generate_glK_multihead_generators
    g = generate_glK_multihead_generators(20, 2)
    gens = torch.from_numpy(g).float() if not isinstance(g, torch.Tensor) else g
    nfc = VFENonFlatConnection(generators=gens, irrep_dims=[10, 10])
    assert nfc.raw_strength.item() == 0.0
    assert nfc.W_raw.abs().max().item() == 0.0
    assert nfc.strength.item() == 0.0
    mu = torch.randn(1, 4, 20)
    delta = nfc(mu)
    assert delta.abs().max().item() == 0.0


def test_q7_pairwise_omega_at_delta_zero_is_block_diag_product():
    """At delta=0, Omega_ij = exp(phi_i)·exp(-phi_j) per block."""
    from transformer.vfe.attention import compute_gauge_transport
    from math_utils.generators import generate_glK_multihead_generators
    torch.manual_seed(0)
    g = generate_glK_multihead_generators(20, 2)
    gens = torch.from_numpy(g).float() if not isinstance(g, torch.Tensor) else g
    phi = torch.randn(1, 4, gens.shape[0]) * 0.05
    bep = compute_gauge_transport(phi, gens, [10, 10], enforce_orthogonal=False)
    delta = torch.zeros(1, 4, 4, gens.shape[0])
    op = compute_pairwise_omega_with_delta(phi, delta, gens, [10, 10], cached_block_exp_pairs=bep)
    # For each block check Omega_ij matches exp_phi_i @ exp_neg_phi_j.
    for h, d_h in enumerate([10, 10]):
        Omega, _ = op[h]
        exp_phi_i = bep[h][0]                   # (1, 4, d_h, d_h)
        exp_neg_phi_j = bep[h][1]
        expected = exp_phi_i.unsqueeze(2) @ exp_neg_phi_j.unsqueeze(1)
        assert torch.allclose(Omega, expected, atol=1e-5), \
            f"block {h}: pairwise omega mismatch with flat construction"


def test_q7_triangle_holonomy_zero_when_flat_active_when_not():
    """Holonomy norm is ~0 with strength=0 and >0 when activated."""
    from transformer.vfe.attention import compute_gauge_transport
    from math_utils.generators import generate_glK_multihead_generators
    torch.manual_seed(0)
    g = generate_glK_multihead_generators(20, 2)
    gens = torch.from_numpy(g).float() if not isinstance(g, torch.Tensor) else g
    phi = torch.randn(1, 8, gens.shape[0]) * 0.05
    mu = torch.randn(1, 8, 20)
    bep = compute_gauge_transport(phi, gens, [10, 10], enforce_orthogonal=False)
    nfc = VFENonFlatConnection(generators=gens, irrep_dims=[10, 10])

    # At init (delta=0)
    delta = nfc(mu)
    op = compute_pairwise_omega_with_delta(phi, delta, gens, [10, 10], cached_block_exp_pairs=bep)
    h_flat = triangle_holonomy_norm(op, [10, 10], n_samples=32)
    assert h_flat < 1e-4, f"flat holonomy should be ~0 got {h_flat}"

    # Activate
    with torch.no_grad():
        nfc.raw_strength.fill_(2.0)
        nfc.W_raw[0, 0, 1] = 0.5
        nfc.W_raw[0, 1, 0] = -0.5
    delta = nfc(mu)
    op = compute_pairwise_omega_with_delta(phi, delta, gens, [10, 10], cached_block_exp_pairs=bep)
    h_active = triangle_holonomy_norm(op, [10, 10], n_samples=32)
    assert h_active > h_flat, \
        f"active holonomy should exceed flat baseline (got {h_active} vs {h_flat})"


def test_q7_model_forward_and_backward():
    cfg = VFEConfig(**CFG_KWARGS, use_non_flat_transport=True)
    model = VFEModel(cfg)
    tok = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss, ce = model(tok, tok)
    assert torch.isfinite(loss)
    loss.backward()
    # W_raw should be in the parameter list; its grad must be finite (or zero)
    W_grad = model.stack.blocks[0].e_step.non_flat_connection.W_raw.grad
    assert W_grad is None or torch.isfinite(W_grad).all()


# ----------------------------------------------------------------------------
# Q8 — Pure-omega gauge parameterization
# ----------------------------------------------------------------------------


def test_q8_init_omega_from_phi_pair_consistency():
    """init_omega_from_phi returns (Omega, Omega^{-1}) pairs with Ω·Ω^{-1}=I."""
    from math_utils.generators import generate_glK_multihead_generators
    torch.manual_seed(0)
    g = generate_glK_multihead_generators(20, 2)
    gens = torch.from_numpy(g).float() if not isinstance(g, torch.Tensor) else g
    phi = torch.randn(2, 8, gens.shape[0]) * 0.05
    op = init_omega_from_phi(phi, gens, [10, 10])
    for h, d_h in enumerate([10, 10]):
        O, Oi = op[h]
        prod = O @ Oi
        eye = torch.eye(d_h).expand_as(prod)
        assert torch.allclose(prod, eye, atol=1e-5), \
            f"block {h}: Ω·Ω^{{-1}} should equal I (max |Δ|={(prod - eye).abs().max().item()})"


def test_q8_project_slk_normalizes_det_to_one():
    """project_omega_to_slk rescales each block to det=1."""
    torch.manual_seed(0)
    B, N, d = 2, 4, 10
    # Random Omega with non-unit det (random GL+(d) matrices)
    O = torch.eye(d).expand(B, N, d, d).clone()
    O = O + 0.1 * torch.randn(B, N, d, d)  # perturb
    Oi = torch.linalg.inv(O)
    op = project_omega_to_slk([(O, Oi)], [d])
    O_new, Oi_new = op[0]
    det = torch.linalg.det(O_new.float())
    assert torch.allclose(det, torch.ones_like(det), atol=1e-4), \
        f"det should be 1 after project_slk; got range [{det.min()}, {det.max()}]"
    # And the pair stays consistent
    prod = O_new @ Oi_new
    eye = torch.eye(d).expand_as(prod)
    assert torch.allclose(prod, eye, atol=1e-4)


def test_q8_omega_direct_model_forward_and_backward():
    cfg = VFEConfig(**CFG_KWARGS, gauge_parameterization='omega_direct')
    model = VFEModel(cfg)
    tok = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss, ce = model(tok, tok)
    assert torch.isfinite(loss)
    # Backward should flow to phi_embed, base_mu, base_log_sigma at minimum.
    loss.backward()
    important = ['prior_bank.phi_embed.weight', 'prior_bank.base_mu', 'prior_bank.base_log_sigma']
    for n, p in model.named_parameters():
        if n in important:
            assert p.grad is not None, f"{n} should have a gradient"
            assert torch.isfinite(p.grad).all(), f"{n}: non-finite grad"


def test_q8_compute_pairwise_omega_from_endpoints_flat():
    """Flat omega-direct transport: Omega_ij = Omega_i · Omega_j^{-1}."""
    torch.manual_seed(0)
    B, N, d = 1, 4, 10
    O = torch.eye(d).expand(B, N, d, d).clone() + 0.05 * torch.randn(B, N, d, d)
    Oi = torch.linalg.inv(O)
    op = compute_pairwise_omega_from_endpoints([(O, Oi)], [d])
    Omega_ij, Omega_ij_inv = op[0]
    # Spot-check (i=0, j=1)
    expected = O[:, 0] @ Oi[:, 1]
    assert torch.allclose(Omega_ij[:, 0, 1], expected, atol=1e-5)
    # And pair consistency
    prod = Omega_ij @ Omega_ij_inv
    eye = torch.eye(d).expand_as(prod)
    assert torch.allclose(prod, eye, atol=1e-4)
