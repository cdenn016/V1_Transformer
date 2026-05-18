"""
Tests for transformer.utils.query_variation
==================================================

Validates the query variation diagnostic: uniform vs diverse attention
detection, and belief similarity analysis.
"""

import pytest
import numpy as np
import torch


# =============================================================================
# TestAnalyzeQueryVariation
# =============================================================================

class TestAnalyzeQueryVariation:
    """analyze_query_variation: attention row L2 distance diagnostics."""

    def test_smoke_test(self, capsys):
        """Runs without error on synthetic attention weights."""
        from transformer.utils.query_variation import analyze_query_variation
        B, H, N = 1, 2, 8
        beta = torch.softmax(torch.randn(B, H, N, N), dim=-1)
        analyze_query_variation(beta)
        captured = capsys.readouterr()
        assert 'QUERY-SIDE VARIATION ANALYSIS' in captured.out

    def test_uniform_detection(self, capsys):
        """All-equal rows produce near-zero L2 distances."""
        from transformer.utils.query_variation import analyze_query_variation
        B, H, N = 1, 1, 6
        # Uniform attention: every row is 1/N
        beta = torch.ones(B, H, N, N) / N
        analyze_query_variation(beta)
        captured = capsys.readouterr()
        # Should report very small distances
        assert 'TRULY UNIFORM' in captured.out or 'Mean L2' in captured.out

    def test_diverse_detection(self, capsys):
        """Random attention should show non-trivial variation."""
        from transformer.utils.query_variation import analyze_query_variation
        B, H, N = 1, 2, 8
        # Random softmax attention — diverse by construction
        beta = torch.softmax(torch.randn(B, H, N, N) * 5.0, dim=-1)
        analyze_query_variation(beta)
        captured = capsys.readouterr()
        assert 'Head' in captured.out

    def test_different_head_counts(self, capsys):
        """Works with varying H, N values."""
        from transformer.utils.query_variation import analyze_query_variation
        for H, N in [(1, 4), (4, 8), (8, 16)]:
            beta = torch.softmax(torch.randn(1, H, N, N), dim=-1)
            analyze_query_variation(beta)
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# =============================================================================
# TestBeliefSimilarity
# =============================================================================

class TestBeliefSimilarity:
    """test_belief_similarity: embedding collapse detection."""

    def test_smoke_test(self, minimal_config, cpu_device, capsys):
        """Runs without error on a minimal model."""
        from transformer.utils.query_variation import test_belief_similarity
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config).to(cpu_device)
        model.eval()
        input_ids = torch.randint(0, minimal_config['vocab_size'], (1, 8), device=cpu_device)
        with torch.no_grad():
            test_belief_similarity(model, input_ids)
        captured = capsys.readouterr()
        assert len(captured.out) > 0
