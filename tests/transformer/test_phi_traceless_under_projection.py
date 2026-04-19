"""
Traceless-phi invariant under phi_project_slk=True.

Covers the fix documented in 2026-04-19_edits.md: positional phi is
re-parameterized onto sl(K) so that tr(composed_phi) = 0 survives BCH
composition with the prior-bank token phi (both VFE and legacy paths).
"""

import pytest
import torch

from transformer.core.vfe_utils import build_slk_basis
from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel
from transformer.vfe.positional import VFEPositionalEncoding
from transformer.vfe.prior_bank import VFEPriorBank


def _per_block_trace(phi: torch.Tensor, V_blocks: torch.Tensor) -> torch.Tensor:
    return phi @ V_blocks.transpose(-2, -1)


@pytest.fixture
def slk_cfg():
    cfg = VFEConfig(
        vocab_size=32,
        embed_dim=12,
        irrep_spec=[('l0', 3, 4)],  # 3 heads x 4 = K=12
        n_layers=1,
        max_seq_len=16,
        n_e_steps=1,
        diagonal_covariance=True,
        gauge_group='GLK',
        phi_project_slk=True,
    )
    return cfg


class TestBuildSlkBasis:

    def test_basis_shape_and_orthonormality(self, slk_cfg):
        model = VFEModel(slk_cfg)
        V_blocks, P = build_slk_basis(model.generators, slk_cfg.irrep_dims)
        H = len(slk_cfg.irrep_dims)
        n_gen = model.generators.shape[0]
        assert V_blocks.shape == (H, n_gen)
        assert P.shape == (n_gen, n_gen - H)
        I = torch.eye(P.shape[-1], dtype=P.dtype)
        assert torch.allclose(P.T @ P, I, atol=1e-5)

    def test_basis_is_traceless(self, slk_cfg):
        model = VFEModel(slk_cfg)
        V_blocks, P = build_slk_basis(model.generators, slk_cfg.irrep_dims)
        # Every column of P must lie in the traceless subalgebra.
        residual = V_blocks @ P  # (H, n_gen - H)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)


@pytest.mark.parametrize("bch_order", [1, 2])
class TestVfePositionalTraceless:

    def test_pos_phi_traceless_after_projection(self, slk_cfg, bch_order):
        slk_cfg.bch_order = bch_order
        model = VFEModel(slk_cfg)
        V_blocks, _ = build_slk_basis(model.generators, slk_cfg.irrep_dims)
        # Raw positional parameter expanded through basis.
        if model.pos_enc.pos_phi_basis is not None:
            pos = model.pos_enc.pos_phi_free @ model.pos_enc.pos_phi_basis.T
        else:
            pos = model.pos_enc.pos_phi_free
        s = _per_block_trace(pos, V_blocks)
        assert torch.allclose(s, torch.zeros_like(s), atol=1e-5)

    def test_composed_phi_traceless_after_forward(self, slk_cfg, bch_order):
        slk_cfg.bch_order = bch_order
        model = VFEModel(slk_cfg)
        V_blocks, _ = build_slk_basis(model.generators, slk_cfg.irrep_dims)
        token_ids = torch.randint(0, slk_cfg.vocab_size, (2, 8))
        beliefs = model.prior_bank.encode(token_ids)
        composed = model.pos_enc(beliefs.phi, seq_len=token_ids.shape[1])
        s = _per_block_trace(composed, V_blocks)
        assert torch.allclose(s, torch.zeros_like(s), atol=1e-5), (
            f"Composed phi not traceless: max|s| = {s.abs().max().item():.3e}"
        )


class TestGradientStaysInSlk:

    def test_pos_phi_free_grad_does_not_populate_trace(self, slk_cfg):
        slk_cfg.bch_order = 2
        model = VFEModel(slk_cfg)
        V_blocks, P = build_slk_basis(model.generators, slk_cfg.irrep_dims)

        token_ids = torch.randint(0, slk_cfg.vocab_size, (2, 8))
        targets = torch.randint(0, slk_cfg.vocab_size, (2, 8))
        _, loss = model(token_ids, targets=targets)
        loss.backward()

        # Expand pos_phi_free.grad through the basis to full n_gen coords,
        # then project onto V_blocks — this must be zero since pos_phi_free
        # only lives in the traceless subspace by construction.
        grad_free = model.pos_enc.pos_phi_free.grad
        assert grad_free is not None
        grad_full = grad_free @ P.T        # (N, n_gen)
        s = grad_full @ V_blocks.T         # (N, H)
        assert torch.allclose(s, torch.zeros_like(s), atol=1e-5)


class TestLegacyPriorBank:

    def _make_bank(self, phi_project_slk: bool):
        from transformer.core.prior_bank import PriorBank
        from math_utils.generators import generate_glK_multihead_generators

        K, n_heads = 12, 3
        irrep_dims = [K // n_heads] * n_heads
        gens = generate_glK_multihead_generators(K, n_heads)
        if not isinstance(gens, torch.Tensor):
            gens = torch.from_numpy(gens).float()
        bank = PriorBank(
            vocab_size=32,
            embed_dim=K,
            gauge_fixed_priors=True,
            generators=gens,
            phi_dim=gens.shape[0],
            phi_scale=0.3,
            irrep_dims=irrep_dims,
            phi_project_slk=phi_project_slk,
        )
        return bank, gens, irrep_dims

    def test_encode_phi_traceless_under_projection(self):
        bank, gens, irrep_dims = self._make_bank(phi_project_slk=True)
        V_blocks, _ = build_slk_basis(gens, irrep_dims)
        token_ids = torch.randint(0, 32, (2, 8))
        mu, sigma, phi = bank.encode(token_ids)[:3]
        s = phi @ V_blocks.T
        assert torch.allclose(s, torch.zeros_like(s), atol=1e-5)

    def test_encode_phi_not_projected_by_default(self):
        # Sanity: without the flag, raw phi has non-zero per-block trace.
        bank, gens, irrep_dims = self._make_bank(phi_project_slk=False)
        V_blocks, _ = build_slk_basis(gens, irrep_dims)
        token_ids = torch.randint(0, 32, (2, 8))
        mu, sigma, phi = bank.encode(token_ids)[:3]
        s = (phi @ V_blocks.T).abs().max().item()
        assert s > 1e-4, f"Expected non-zero per-block trace without projection, got {s}"


class TestGaugeTransformerLMEndToEnd:
    """Matches the user's EM_CONFIG: use_prior_bank=True + gauge_fixed_priors=False.

    The failure that prompted this test: _embed_irrep_dims was gated on
    gauge_fixed_priors, so per-head block structure never reached the bank's
    sl(K) projection and the projector collapsed to a single-block treatment
    that only killed the full-K trace while leaving per-head traces alive.
    """

    def _build_model(self, phi_project_slk: bool):
        from transformer.core.model import GaugeTransformerLM

        config = {
            'vocab_size': 128, 'embed_dim': 20, 'max_seq_len': 16, 'n_layers': 1,
            'irrep_spec': [('fund', 2, 10)], 'gauge_group': 'GLK',
            'gauge_mode': 'learned', 'gauge_param': 'phi',
            'use_prior_bank': True, 'gauge_fixed_priors': False,
            'phi_project_slk': phi_project_slk, 'phi_trace_clamp': None,
            'pos_encoding_mode': 'none', 'bch_order': 1,
            'diagonal_covariance': True, 'evolve_phi': True,
            'em_mode': 'em_phi_p', 'ffn_n_iterations': 1, 'kappa_beta': 1.0,
            'mask_self_attention': False,
        }
        torch.manual_seed(0)
        return GaugeTransformerLM(config)

    def test_attn_info_phi_traceless_per_head(self):
        model = self._build_model(phi_project_slk=True)
        assert model.prior_bank.irrep_dims == [10, 10]
        assert model.prior_bank._phi_trace_vec is not None
        assert model.prior_bank._phi_trace_vec.shape == (2, 200)

        token_ids = torch.randint(0, 128, (2, 8))
        model.eval()
        with torch.no_grad():
            _, attn_info = model.forward_with_attention(token_ids, targets=token_ids)
        phi = attn_info['phi']
        V = model.prior_bank._phi_trace_vec
        s = (phi @ V.T).abs().max().item()
        assert s < 1e-5, f"Per-head trace of attn_info['phi'] not zero: max|s_h|={s}"

    def test_det_omega_is_unity_per_block(self):
        model = self._build_model(phi_project_slk=True)
        token_ids = torch.randint(0, 128, (2, 8))
        model.eval()
        with torch.no_grad():
            _, attn_info = model.forward_with_attention(token_ids, targets=token_ids)
        phi = attn_info['phi']
        gens = model.generators
        A = torch.einsum('...g,gij->...ij', phi, gens)
        Omega = torch.linalg.matrix_exp(A)
        start = 0
        for d_h in [10, 10]:
            end = start + d_h
            det_h = torch.linalg.det(Omega[..., start:end, start:end])
            assert torch.allclose(det_h, torch.ones_like(det_h), atol=1e-4), (
                f"det(Omega_h) not unit for block [{start}:{end}]: "
                f"min={det_h.min().item()}, max={det_h.max().item()}"
            )
            start = end

    def test_without_projection_det_is_not_unit(self):
        # Negative control: flag off, det should spread around 1 (log-normal).
        model = self._build_model(phi_project_slk=False)
        assert model.prior_bank._phi_trace_vec is None
        token_ids = torch.randint(0, 128, (2, 8))
        model.eval()
        with torch.no_grad():
            _, attn_info = model.forward_with_attention(token_ids, targets=token_ids)
        phi = attn_info['phi']
        gens = model.generators
        A = torch.einsum('...g,gij->...ij', phi, gens)
        Omega = torch.linalg.matrix_exp(A)
        det_h = torch.linalg.det(Omega[..., 0:10, 0:10])
        spread = det_h.std().item()
        assert spread > 1e-3, (
            f"Expected non-unit det(Omega) spread without projection, got std={spread}"
        )


class TestLegacyGaugePositionalEncoding:

    def test_legacy_learned_mode_traceless(self):
        from transformer.core.embeddings import GaugePositionalEncoding
        from math_utils.generators import generate_glK_multihead_generators

        K = 12
        n_heads = 3
        irrep_dims = [K // n_heads] * n_heads
        gens = generate_glK_multihead_generators(K, n_heads)
        if not isinstance(gens, torch.Tensor):
            gens = torch.from_numpy(gens).float()
        phi_dim = gens.shape[0]

        enc = GaugePositionalEncoding(
            max_seq_len=16, mode='learned', scale=0.1,
            composition='bch2', phi_dim=phi_dim, generators=gens,
            gauge_group='GLK', irrep_dims=irrep_dims, phi_project_slk=True,
        )
        V_blocks, _ = build_slk_basis(gens, irrep_dims)
        pos = enc(num_agents=8)                  # (8, phi_dim)
        s = pos @ V_blocks.T
        assert torch.allclose(s, torch.zeros_like(s), atol=1e-5)

    def test_legacy_sinusoidal_projected_at_init(self):
        from transformer.core.embeddings import GaugePositionalEncoding
        from math_utils.generators import generate_glK_multihead_generators

        K = 12
        n_heads = 3
        irrep_dims = [K // n_heads] * n_heads
        gens = generate_glK_multihead_generators(K, n_heads)
        if not isinstance(gens, torch.Tensor):
            gens = torch.from_numpy(gens).float()
        phi_dim = gens.shape[0]

        enc = GaugePositionalEncoding(
            max_seq_len=16, mode='sinusoidal', scale=0.1,
            composition='bch2', phi_dim=phi_dim, generators=gens,
            gauge_group='GLK', irrep_dims=irrep_dims, phi_project_slk=True,
        )
        V_blocks, _ = build_slk_basis(gens, irrep_dims)
        pos = enc(num_agents=8)
        s = pos @ V_blocks.T
        assert torch.allclose(s, torch.zeros_like(s), atol=1e-5)
