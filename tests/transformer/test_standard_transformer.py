"""
Tests for transformer/baselines/standard_transformer.py
========================================================

Tests the baseline transformer: forward pass, RoPE, causal masking,
and parameter counts for fair comparison.
"""

import pytest
import torch
import warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_standard_config(**overrides):
    """Minimal standard transformer config."""
    config = {
        'vocab_size': 100,
        'embed_dim': 32,
        'n_layers': 1,
        'n_heads': 2,
        'hidden_dim': 64,
        'max_seq_len': 32,
        'dropout': 0.0,
        'tie_embeddings': True,
    }
    config.update(overrides)
    return config


def _make_standard_model(**overrides):
    """Create a StandardTransformerLM."""
    from transformer.baselines.standard_transformer import StandardTransformerLM
    config = _make_standard_config(**overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return StandardTransformerLM(config), config


# ===========================================================================
# TestStandardTransformerForward
# ===========================================================================

class TestStandardTransformerForward:
    """Tests for StandardTransformerLM.forward()."""

    def test_output_shape(self):
        """Output logits are (B, N, V)."""
        model, config = _make_standard_model()
        model.eval()
        x = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            output = model(x)
        logits = output['logits']
        assert logits.shape == (2, 8, 100)

    def test_output_finite(self):
        """No NaN/Inf in output logits."""
        model, _ = _make_standard_model()
        model.eval()
        x = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            output = model(x)
        assert torch.isfinite(output['logits']).all()

    def test_gradient_flow(self):
        """Backward produces finite gradients."""
        model, _ = _make_standard_model()
        model.train()
        x = torch.randint(0, 100, (1, 8))
        targets = torch.randint(0, 100, (1, 8))
        output = model(x, labels=targets)
        loss = output['loss']
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), \
                    f"Non-finite gradient in {name}"

    def test_causal_masking(self):
        """Future tokens don't influence past: changing future doesn't change past logits."""
        model, _ = _make_standard_model()
        model.eval()
        x1 = torch.tensor([[1, 2, 3, 4, 5]])
        x2 = torch.tensor([[1, 2, 3, 99, 50]])  # changed tokens 3,4
        with torch.no_grad():
            logits1 = model(x1)['logits']
            logits2 = model(x2)['logits']
        # Positions 0,1,2 should be identical (causal → future doesn't affect past)
        assert torch.allclose(logits1[0, :3], logits2[0, :3], atol=1e-4), \
            f"Causal mask violated: max diff {(logits1[0, :3] - logits2[0, :3]).abs().max():.2e}"

    def test_rope_mode(self):
        """Model with use_rope=True produces valid output."""
        model, _ = _make_standard_model(use_rope=True)
        model.eval()
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            output = model(x)
        assert output['logits'].shape == (1, 8, 100)
        assert torch.isfinite(output['logits']).all()

    def test_attention_only_mode(self):
        """disable_ffn=True works (attention-only ablation)."""
        model, _ = _make_standard_model(disable_ffn=True)
        model.eval()
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            output = model(x)
        assert output['logits'].shape == (1, 8, 100)
        assert torch.isfinite(output['logits']).all()

    def test_loss_computed_with_labels(self):
        """Passing labels returns loss."""
        model, _ = _make_standard_model()
        model.train()
        x = torch.randint(0, 100, (1, 8))
        labels = torch.randint(0, 100, (1, 8))
        output = model(x, labels=labels)
        assert 'loss' in output
        assert output['loss'].dim() == 0  # scalar
        assert torch.isfinite(output['loss'])


# ===========================================================================
# TestRoPE
# ===========================================================================

class TestRoPE:
    """Tests for RoPE functions."""

    def test_rope_preserves_norm(self):
        """||Q_rotated|| ≈ ||Q|| — RoPE is a rotation."""
        from transformer.baselines.standard_transformer import apply_rope_to_qk
        B, H, N, D = 1, 1, 8, 16
        Q = torch.randn(B, H, N, D)
        K = torch.randn(B, H, N, D)
        Q_rot, K_rot = apply_rope_to_qk(Q, K)
        q_norms_before = Q.norm(dim=-1)
        q_norms_after = Q_rot.norm(dim=-1)
        assert torch.allclose(q_norms_before, q_norms_after, atol=1e-5), \
            f"Norm changed: max diff {(q_norms_before - q_norms_after).abs().max():.4e}"

    def test_rope_position_dependent(self):
        """Different positions → different rotations."""
        from transformer.baselines.standard_transformer import apply_rope_to_qk
        B, H, N, D = 1, 1, 8, 16
        Q = torch.ones(B, H, N, D)
        K = torch.ones(B, H, N, D)
        Q_rot, _ = apply_rope_to_qk(Q, K)
        # Position 0 and position 7 should differ
        assert not torch.allclose(Q_rot[0, 0, 0], Q_rot[0, 0, 7], atol=1e-4)


# ===========================================================================
# TestParameterCount
# ===========================================================================

class TestParameterCount:
    """Tests for parameter count comparison."""

    def test_attention_only_fewer_params(self):
        """Disabling FFN reduces parameter count."""
        model_full, _ = _make_standard_model(disable_ffn=False)
        model_attn, _ = _make_standard_model(disable_ffn=True)
        n_full = sum(p.numel() for p in model_full.parameters())
        n_attn = sum(p.numel() for p in model_attn.parameters())
        assert n_attn < n_full, \
            f"Attention-only ({n_attn}) should have fewer params than full ({n_full})"
