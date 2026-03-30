"""
Tests for transformer/baselines/flops_counter.py
==================================================

Tests analytical FLOPs estimation for both standard and gauge transformers.
Peer review will scrutinize these claims.
"""

import pytest


# ===========================================================================
# TestStandardFLOPs
# ===========================================================================

class TestStandardFLOPs:
    """Tests for count_standard_transformer_flops()."""

    def test_flops_positive(self):
        """Standard transformer FLOPs > 0."""
        from transformer.baselines.flops_counter import count_standard_transformer_flops
        result = count_standard_transformer_flops(
            vocab_size=50000, embed_dim=256, n_layers=6,
            n_heads=4, hidden_dim=512, seq_len=128,
        )
        assert result['forward_total'] > 0
        assert result['step_total'] > 0

    def test_flops_scale_with_seq_len(self):
        """2× seq_len → total FLOPs increase significantly (quadratic attention)."""
        from transformer.baselines.flops_counter import count_standard_transformer_flops
        kwargs = dict(vocab_size=50000, embed_dim=256, n_layers=6,
                      n_heads=4, hidden_dim=512)
        r1 = count_standard_transformer_flops(seq_len=64, **kwargs)
        r2 = count_standard_transformer_flops(seq_len=128, **kwargs)
        ratio = r2['forward_total'] / r1['forward_total']
        assert ratio > 1.5, f"FLOPs ratio {ratio:.2f} too low for 2× seq_len"

    def test_no_ffn_reduces_flops(self):
        """disable_ffn=True reduces total FLOPs."""
        from transformer.baselines.flops_counter import count_standard_transformer_flops
        kwargs = dict(vocab_size=50000, embed_dim=256, n_layers=6,
                      n_heads=4, hidden_dim=512, seq_len=128)
        r_full = count_standard_transformer_flops(disable_ffn=False, **kwargs)
        r_attn = count_standard_transformer_flops(disable_ffn=True, **kwargs)
        assert r_attn['forward_total'] < r_full['forward_total']

    def test_backward_is_2x_forward(self):
        """Backward FLOPs ≈ 2× forward FLOPs."""
        from transformer.baselines.flops_counter import count_standard_transformer_flops
        result = count_standard_transformer_flops(
            vocab_size=50000, embed_dim=256, n_layers=6,
            n_heads=4, hidden_dim=512, seq_len=128,
        )
        assert result['backward_total'] == 2 * result['forward_total']


# ===========================================================================
# TestGaugeFLOPs
# ===========================================================================

class TestGaugeFLOPs:
    """Tests for count_gauge_transformer_flops()."""

    def test_flops_positive(self):
        """Gauge transformer FLOPs > 0."""
        from transformer.baselines.flops_counter import count_gauge_transformer_flops
        result = count_gauge_transformer_flops(
            vocab_size=50000, embed_dim=25, n_layers=2,
            n_heads=6, head_dim=5, seq_len=64,
            phi_dim=100, ffn_n_iterations=3,
        )
        assert result['forward_total'] > 0

    def test_more_iterations_more_flops(self):
        """More VFE iterations → more FLOPs."""
        from transformer.baselines.flops_counter import count_gauge_transformer_flops
        kwargs = dict(vocab_size=50000, embed_dim=25, n_layers=2,
                      n_heads=6, head_dim=5, seq_len=64, phi_dim=100)
        r1 = count_gauge_transformer_flops(ffn_n_iterations=1, **kwargs)
        r3 = count_gauge_transformer_flops(ffn_n_iterations=3, **kwargs)
        assert r3['forward_total'] > r1['forward_total']


# ===========================================================================
# TestFLOPsComparison
# ===========================================================================

class TestFLOPsComparison:
    """Tests for compare_flops()."""

    def test_comparison_returns_both(self):
        """compare_flops returns dict with both model types."""
        from transformer.baselines.flops_counter import compare_flops
        gauge_config = {
            'vocab_size': 50000, 'embed_dim': 25, 'n_layers': 2,
            'irrep_spec': [('fund', 6, 5)],
            'ffn_n_iterations': 3,
        }
        standard_config = {
            'vocab_size': 50000, 'embed_dim': 256, 'n_layers': 6,
            'n_heads': 4, 'hidden_dim': 512,
        }
        result = compare_flops(gauge_config, standard_config, seq_len=64)
        assert 'gauge' in result
        assert 'standard' in result


# ===========================================================================
# TestFormatFlops
# ===========================================================================

class TestFormatFlops:
    """Tests for format_flops()."""

    def test_format_readable(self):
        """format_flops produces human-readable string."""
        from transformer.baselines.flops_counter import format_flops
        s = format_flops(1.5e12)
        assert 'TFLOPs' in s
        s = format_flops(1.5e9)
        assert 'GFLOPs' in s
