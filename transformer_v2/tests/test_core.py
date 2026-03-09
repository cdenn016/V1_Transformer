# -*- coding: utf-8 -*-
"""
Core tests for transformer_v2.

Extracted from legacy __main__ blocks in attention.py, blocks.py,
embeddings.py, and model.py. Run with: python -m pytest transformer_v2/tests/test_core.py
"""

import torch
import torch.nn as nn
import pytest

from transformer_v2.config import GaugeTransformerConfig
from transformer_v2.model import GaugeTransformerLM
from transformer_v2.blocks import GaugeTransformerBlock, GaugeTransformerStack
from transformer_v2.attention import (
    IrrepMultiHeadAttention,
    create_attention_mask,
    compute_attention_weights,
)
from transformer_v2.kl_ops import compute_kl_matrix, compute_transport_operators
from transformer_v2.embeddings import GaugeTokenEmbedding, GaugePositionalEncoding
from transformer_v2.prior_bank import PriorBank


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def so3_config():
    """Minimal SO(3) config for testing."""
    return GaugeTransformerConfig(
        vocab_size=100,
        embed_dim=16,
        n_layers=2,
        hidden_dim=64,
        max_seq_len=16,
        kappa=1.0,
        dropout=0.0,
        evolve_sigma=False,
        evolve_phi=False,
        tie_embeddings=True,
        use_layernorm=True,
        use_dropout=False,
        use_residual=True,
        gauge_group='SO3',
        irrep_spec=[('ℓ0', 4, 1), ('ℓ1', 2, 3), ('ℓ2', 1, 5)],
    )


@pytest.fixture
def glk_config():
    """GL(K) multi-head config for testing."""
    return GaugeTransformerConfig(
        vocab_size=100,
        embed_dim=16,
        n_layers=2,
        hidden_dim=64,
        max_seq_len=16,
        kappa=1.0,
        dropout=0.0,
        evolve_sigma=False,
        evolve_phi=False,
        tie_embeddings=True,
        use_layernorm=False,
        use_dropout=False,
        use_residual=False,
        gauge_group='GLK',
        gauge_dim=16,
        use_multi_irrep=True,
        irrep_spec=[('fund', 4, 4)],
    )


@pytest.fixture
def diagonal_config():
    """Config with diagonal covariance for memory-efficient testing."""
    return GaugeTransformerConfig(
        vocab_size=100,
        embed_dim=16,
        n_layers=1,
        hidden_dim=64,
        max_seq_len=16,
        kappa=1.0,
        dropout=0.0,
        evolve_sigma=True,
        evolve_phi=False,
        diagonal_covariance=True,
        gauge_group='SO3',
        irrep_spec=[('ℓ0', 4, 1), ('ℓ1', 2, 3), ('ℓ2', 1, 5)],
    )


@pytest.fixture
def so3_generators():
    """Random skew-symmetric SO(3) generators for embed_dim=16."""
    G = torch.randn(3, 16, 16)
    return 0.5 * (G - G.transpose(-1, -2))


# ── Config Tests ────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        cfg = GaugeTransformerConfig(vocab_size=256, embed_dim=32)
        assert cfg.phi_dim == 3  # SO3 default
        assert cfg.n_layers == 6

    def test_trivial_gauge_overrides(self):
        cfg = GaugeTransformerConfig(
            vocab_size=256, embed_dim=32, gauge_mode='trivial'
        )
        assert cfg.use_identity_transport is True
        assert cfg.evolve_phi is False

    def test_pure_fep_unties_embeddings(self):
        cfg = GaugeTransformerConfig(
            vocab_size=256, embed_dim=32,
            pure_fep_mode=True, tie_embeddings=True,
        )
        assert cfg.tie_embeddings is False

    def test_irrep_dims_computed(self):
        cfg = GaugeTransformerConfig(
            vocab_size=256, embed_dim=15,
            irrep_spec=[('ℓ0', 3, 1), ('ℓ1', 2, 3), ('ℓ2', 1, 3)],
            use_block_diagonal_kl=True,
        )
        assert cfg.irrep_dims == [1, 1, 1, 3, 3, 3]

    def test_from_legacy_dict(self):
        legacy = {
            'vocab_size': 128,
            'embed_dim': 32,
            'kappa_beta': 2.0,
            'ffn_alpha': 0.01,
        }
        cfg = GaugeTransformerConfig.from_legacy_dict(legacy)
        assert cfg.vocab_size == 128
        assert cfg.kappa == 2.0
        assert cfg.alpha == 0.01


# ── Attention Tests ─────────────────────────────────────────────────────

class TestAttention:
    def test_create_mask_causal(self):
        mask = create_attention_mask(num_agents=8, causal=True)
        assert mask.shape == (8, 8)
        assert mask[0, 7] == 0.0  # Upper triangle blocked
        assert mask[7, 0] == 1.0  # Lower triangle open

    def test_create_mask_full(self):
        mask = create_attention_mask(num_agents=8, causal=False)
        assert mask.shape == (8, 8)
        assert (mask == 1.0).all()

    def test_kl_matrix_shapes(self, so3_generators):
        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K)
        sigma = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1) * 0.5
        phi = torch.randn(B, N, 3) * 0.1

        kl = compute_kl_matrix(
            mu, sigma, phi, so3_generators,
            diagonal_covariance=False,
        )
        assert kl.shape == (B, N, N)

    def test_kl_matrix_diagonal(self, so3_generators):
        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K)
        sigma = torch.ones(B, N, K) * 0.5
        phi = torch.randn(B, N, 3) * 0.1

        kl = compute_kl_matrix(
            mu, sigma, phi, so3_generators,
            diagonal_covariance=True,
        )
        assert kl.shape == (B, N, N)

    def test_attention_module(self, so3_config, so3_generators):
        B, N = 2, 8
        K = so3_config.embed_dim

        attn = IrrepMultiHeadAttention(so3_config, so3_generators)
        mu = torch.randn(B, N, K)
        sigma = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1) * 0.5
        phi = torch.randn(B, N, 3) * 0.1

        mu_out, sigma_out, beta, kl = attn(
            mu, sigma, phi, so3_generators,
            return_attention=True,
        )
        assert mu_out.shape == (B, N, K)
        assert beta is not None


# ── Embeddings Tests ────────────────────────────────────────────────────

class TestEmbeddings:
    def test_token_embedding(self, so3_config, so3_generators):
        embed = GaugeTokenEmbedding(so3_config, so3_generators)
        token_ids = torch.randint(0, 100, (2, 8))
        mu, sigma, phi = embed(token_ids)
        assert mu.shape == (2, 8, 16)
        assert phi.shape == (2, 8, 3)

    def test_positional_encoding(self, so3_config, so3_generators):
        pos_enc = GaugePositionalEncoding(so3_config, so3_generators)
        phi = torch.randn(2, 8, 3) * 0.1
        phi_out = pos_enc.compose(phi, num_agents=8, device=phi.device)
        assert phi_out.shape == phi.shape


# ── Block Tests ─────────────────────────────────────────────────────────

class TestBlocks:
    def test_single_block(self, so3_config, so3_generators):
        B, N, K = 2, 8, 16
        block = GaugeTransformerBlock(so3_config, so3_generators)

        mu = torch.randn(B, N, K)
        sigma = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1) * 0.5
        phi = torch.randn(B, N, 3) * 0.1
        mu_prior = mu.clone() * 0.5

        mu_out, sigma_out, phi_out = block(
            mu, sigma, phi, so3_generators,
            mu_prior=mu_prior,
        )
        assert mu_out.shape == (B, N, K)

    def test_stack(self, so3_config, so3_generators):
        B, N, K = 2, 8, 16
        stack = GaugeTransformerStack(so3_config, so3_generators)
        assert len(stack.blocks) == so3_config.n_layers

        mu = torch.randn(B, N, K)
        sigma = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1) * 0.5
        phi = torch.randn(B, N, 3) * 0.1
        mu_prior = mu.clone() * 0.5

        mu_out, sigma_out, phi_out, intermediates = stack(
            mu, sigma, phi, so3_generators,
            mu_prior=mu_prior,
            return_intermediates=True,
        )
        assert mu_out.shape == (B, N, K)
        assert len(intermediates) == so3_config.n_layers


# ── Model Tests ─────────────────────────────────────────────────────────

class TestModel:
    def test_forward(self, so3_config):
        model = GaugeTransformerLM(so3_config)
        token_ids = torch.randint(0, 100, (2, 8))
        logits = model(token_ids)
        assert logits.shape == (2, 8, 100)

    def test_forward_with_agents(self, so3_config):
        model = GaugeTransformerLM(so3_config)
        token_ids = torch.randint(0, 100, (2, 8))
        logits, agents = model(token_ids, return_agents=True)
        assert logits.shape == (2, 8, 100)
        assert 'mu' in agents
        assert agents['intermediates'] is not None

    def test_forward_with_attention(self, so3_config):
        model = GaugeTransformerLM(so3_config)
        token_ids = torch.randint(0, 100, (2, 8))
        logits, attn_info = model.forward_with_attention(token_ids)
        assert logits.shape == (2, 8, 100)
        assert 'beta' in attn_info
        assert attn_info['n_layers'] == 2

    def test_generate(self, so3_config):
        model = GaugeTransformerLM(so3_config)
        prompt = torch.randint(0, 100, (1, 4))
        generated = model.generate(prompt, max_new_tokens=4, temperature=1.0, top_k=10)
        assert generated.shape == (1, 8)

    def test_param_count(self, so3_config):
        model = GaugeTransformerLM(so3_config)
        total = model.get_num_params(non_embedding=False)
        non_embed = model.get_num_params(non_embedding=True)
        assert total > non_embed > 0


# ── Prior Bank Tests ────────────────────────────────────────────────────

class TestPriorBank:
    def test_encode_decode(self):
        bank = PriorBank(vocab_size=100, embed_dim=16)
        token_ids = torch.randint(0, 100, (2, 8))
        mu, sigma, phi = bank.encode(token_ids)
        assert mu.shape == (2, 8, 16)

        logits = bank.decode(mu, sigma)
        assert logits.shape == (2, 8, 100)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
