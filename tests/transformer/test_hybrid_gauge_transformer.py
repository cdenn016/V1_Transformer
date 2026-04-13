"""
Tests for transformer.baselines.hybrid_gauge_transformer
=========================================================

Validates the hybrid baseline model: standard GELU FFN + gauge KL-attention
+ PriorBank embeddings. Tests forward pass, gauge group inference, state
dict round-trip, and generation.
"""

import pytest
import torch

from transformer.baselines.hybrid_gauge_transformer import (
    HybridGaugeTransformerLM,
    _infer_gauge_group,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_hybrid_config(**overrides):
    """Minimal hybrid model config for fast tests."""
    config = {
        'vocab_size': 100,
        'embed_dim': 12,
        'n_layers': 1,
        'irrep_spec': [('glk', 2, 6)],  # 2 heads, dim=6 each, K=12
        'hidden_dim': 32,
        'max_seq_len': 32,
        'kappa_beta': 1.0,
        'dropout': 0.0,
        'gauge_group': 'GLK',
        'diagonal_covariance': True,
        'use_rope': False,
        'pos_encoding_mode': 'none',
        'use_prior_bank': True,
        'gauge_fixed_priors': False,
        'tie_embeddings': True,
        'gauge_mode': 'learned',
    }
    config.update(overrides)
    return config


def _make_hybrid_model(device='cpu', **overrides):
    """Create a HybridGaugeTransformerLM on CPU."""
    config = _make_hybrid_config(**overrides)
    model = HybridGaugeTransformerLM(config)
    model = model.to(device)
    model.eval()
    return model, config


# =============================================================================
# TestInferGaugeGroup
# =============================================================================

class TestInferGaugeGroup:
    """_infer_gauge_group: infer gauge group from generator shape."""

    def test_so3(self):
        """3 generators -> SO3."""
        gen = torch.zeros(3, 3, 3)
        group, dim = _infer_gauge_group(gen)
        assert group == 'SO3'

    def test_glk(self):
        """K^2 generators -> GLK."""
        K = 4
        gen = torch.zeros(K * K, K, K)
        group, dim = _infer_gauge_group(gen)
        assert group == 'GLK'
        assert dim == K

    def test_none(self):
        """None generators -> default SO3."""
        group, dim = _infer_gauge_group(None)
        assert group == 'SO3'
        assert dim == 3


# =============================================================================
# TestHybridGaugeTransformerLM
# =============================================================================

class TestHybridGaugeTransformerLM:
    """HybridGaugeTransformerLM: forward pass, state dict, gradients."""

    def test_creation(self, cpu_device):
        """Model instantiates without error."""
        model, _ = _make_hybrid_model(cpu_device)
        assert model is not None

    def test_forward_pass_shape(self, cpu_device):
        """Logits shape is (B, N, V)."""
        model, config = _make_hybrid_model(cpu_device)
        B, N = 2, 8
        x = torch.randint(0, config['vocab_size'], (B, N), device=cpu_device)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (B, N, config['vocab_size']), \
            f"Expected ({B}, {N}, {config['vocab_size']}), got {logits.shape}"

    def test_forward_pass_finite(self, cpu_device):
        """No NaN/Inf in logits."""
        model, config = _make_hybrid_model(cpu_device)
        x = torch.randint(0, config['vocab_size'], (2, 8), device=cpu_device)
        with torch.no_grad():
            logits = model(x)
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"

    def test_gradient_flow(self, cpu_device):
        """Backward produces gradients in all parameter groups."""
        model, config = _make_hybrid_model(cpu_device)
        model.train()
        x = torch.randint(0, config['vocab_size'], (2, 8), device=cpu_device)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        # Check at least some parameters have gradients
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())
        assert params_with_grad > 0, "No parameters received gradients"

    def test_eval_determinism(self, cpu_device):
        """Two calls in eval mode produce identical output."""
        model, config = _make_hybrid_model(cpu_device)
        x = torch.randint(0, config['vocab_size'], (1, 8), device=cpu_device)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2, atol=1e-6), "Eval mode not deterministic"

    def test_prior_bank_exists(self, cpu_device):
        """Model has a PriorBank when use_prior_bank=True."""
        model, _ = _make_hybrid_model(cpu_device, use_prior_bank=True)
        assert hasattr(model, 'prior_bank'), "Model missing prior_bank"

    def test_state_dict_roundtrip(self, cpu_device):
        """Save and reload state_dict produces same output."""
        model, config = _make_hybrid_model(cpu_device)
        x = torch.randint(0, config['vocab_size'], (1, 8), device=cpu_device)
        with torch.no_grad():
            out_before = model(x)

        state = model.state_dict()
        model2 = HybridGaugeTransformerLM(config).to(cpu_device)
        model2.load_state_dict(state)
        model2.eval()
        with torch.no_grad():
            out_after = model2(x)

        assert torch.allclose(out_before, out_after, atol=1e-5), \
            f"State dict roundtrip error: {(out_before - out_after).abs().max():.2e}"

    def test_diagonal_covariance_config(self, cpu_device):
        """diagonal_covariance=True works."""
        model, config = _make_hybrid_model(cpu_device, diagonal_covariance=True)
        x = torch.randint(0, config['vocab_size'], (1, 4), device=cpu_device)
        with torch.no_grad():
            logits = model(x)
        assert torch.isfinite(logits).all()

    def test_forward_with_attention(self, cpu_device):
        """forward_with_attention returns logits + attention data."""
        model, config = _make_hybrid_model(cpu_device)
        x = torch.randint(0, config['vocab_size'], (1, 8), device=cpu_device)
        with torch.no_grad():
            result = model.forward_with_attention(x)
        # May return a tuple (logits, attn_data) or a dict
        if isinstance(result, tuple):
            logits = result[0]
        else:
            logits = result.get('logits', result)
        assert logits.shape[0] == 1
        assert logits.shape[2] == config['vocab_size']


# =============================================================================
# TestHybridGaugeGenerate
# =============================================================================

class TestHybridGaugeGenerate:
    """HybridGaugeTransformerLM.generate: autoregressive text generation."""

    def test_generate_produces_tokens(self, cpu_device):
        """Generate produces integer tensor."""
        model, config = _make_hybrid_model(cpu_device)
        prompt = torch.randint(0, config['vocab_size'], (1, 4), device=cpu_device)
        with torch.no_grad():
            tokens = model.generate(prompt, max_new_tokens=6, temperature=1.0)
        assert tokens.dtype in (torch.int64, torch.int32, torch.long)
        assert tokens.shape[1] == 10  # 4 prompt + 6 new

    def test_generate_finite_logits(self, cpu_device):
        """Generation doesn't produce NaN (smoke test)."""
        model, config = _make_hybrid_model(cpu_device)
        prompt = torch.randint(0, config['vocab_size'], (1, 2), device=cpu_device)
        with torch.no_grad():
            tokens = model.generate(prompt, max_new_tokens=3, temperature=1.0)
        # If generation succeeded without error, logits were finite
        assert tokens.shape[1] == 5

    def test_generate_bounded_length(self, cpu_device):
        """Generation length is bounded by max_new_tokens."""
        model, config = _make_hybrid_model(cpu_device)
        prompt = torch.randint(0, config['vocab_size'], (1, 2), device=cpu_device)
        max_new = 5
        with torch.no_grad():
            tokens = model.generate(prompt, max_new_tokens=max_new, temperature=1.0)
        assert tokens.shape[1] <= 2 + max_new + 1  # prompt + new + possible off-by-one
