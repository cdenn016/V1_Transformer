"""
Tests for semantic analysis module.

Tests cover:
- Token categorization
- Clustering metrics on synthetic embeddings
- Semantic field coherence computation
- Sigma covariance analysis
- Word pair analysis
- Omega extraction and metrics
- SemanticTrajectoryTracker save/load
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# Check if tiktoken tokenizer is available (requires network on first use)
_TIKTOKEN_AVAILABLE = False
try:
    import tiktoken
    _tok = tiktoken.get_encoding("gpt2")
    _TIKTOKEN_AVAILABLE = True
except Exception:
    pass

requires_tiktoken = pytest.mark.skipif(
    not _TIKTOKEN_AVAILABLE,
    reason="tiktoken GPT-2 encoding not available (requires network)"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def random_mu_embed():
    """Random mu embedding (vocab=200, dim=16)."""
    torch.manual_seed(42)
    return torch.randn(200, 16)


@pytest.fixture
def random_phi_embed():
    """Random phi embedding (vocab=200, dim=3) for SO(3)."""
    torch.manual_seed(43)
    return torch.randn(200, 3)


@pytest.fixture
def clustered_embed():
    """Embedding with known cluster structure for 3 categories."""
    torch.manual_seed(44)
    n_per_class = 50
    dim = 8
    # Class 0: centered at (2, 0, 0, ...)
    c0 = torch.randn(n_per_class, dim) * 0.1 + torch.tensor([2.0] + [0.0] * (dim - 1))
    # Class 1: centered at (0, 2, 0, ...)
    c1 = torch.randn(n_per_class, dim) * 0.1 + torch.tensor([0.0, 2.0] + [0.0] * (dim - 2))
    # Class 2: centered at (0, 0, 2, ...)
    c2 = torch.randn(n_per_class, dim) * 0.1 + torch.tensor([0.0, 0.0, 2.0] + [0.0] * (dim - 3))
    return torch.cat([c0, c1, c2], dim=0)


@pytest.fixture
def random_sigma_diag():
    """Random diagonal Sigma (vocab=200, K=16)."""
    torch.manual_seed(45)
    return torch.rand(200, 16) * 0.5 + 0.1  # positive values


@pytest.fixture
def random_sigma_full():
    """Random full Sigma (vocab=50, K=4, K=4) -- SPD matrices."""
    torch.manual_seed(46)
    n, K = 50, 4
    # Generate SPD matrices: A @ A^T + eps*I
    A = torch.randn(n, K, K) * 0.3
    sigma = torch.bmm(A, A.transpose(-1, -2)) + 0.1 * torch.eye(K).unsqueeze(0)
    return sigma


# =============================================================================
# Token Categorization Tests
# =============================================================================

@requires_tiktoken
class TestTokenCategorization:
    def test_categorize_token_returns_valid_category(self):
        from transformer.analysis.semantics import categorize_token
        cat = categorize_token(0)
        assert cat in ('letter', 'digit', 'punct', 'function', 'content', 'other')

    def test_categorize_digit(self):
        from transformer.analysis.semantics import categorize_token, get_token_id
        # Token for "0" should be a digit
        tid = get_token_id("0")
        if tid is not None:
            assert categorize_token(tid) == 'digit'

    def test_categorize_function_word(self):
        from transformer.analysis.semantics import categorize_token, get_token_id
        tid = get_token_id("the")
        if tid is not None:
            assert categorize_token(tid) == 'function'

    def test_get_token_id_single_token(self):
        from transformer.analysis.semantics import get_token_id
        # "cat" should resolve to a single token
        tid = get_token_id("cat")
        assert tid is not None
        assert isinstance(tid, int)

    def test_get_token_id_multi_token_returns_none(self):
        from transformer.analysis.semantics import get_token_id
        # Long word likely splits into multiple tokens
        tid = get_token_id("antidisestablishmentarianism")
        assert tid is None


# =============================================================================
# Clustering Metrics Tests
# =============================================================================

@requires_tiktoken
class TestClusteringMetrics:
    def test_compute_clustering_metrics_returns_dict(self, random_mu_embed):
        from transformer.analysis.semantics import compute_clustering_metrics
        result = compute_clustering_metrics(random_mu_embed, n_tokens=100, embed_name='test')
        assert isinstance(result, dict)

    def test_silhouette_in_range(self, random_mu_embed):
        from transformer.analysis.semantics import compute_clustering_metrics
        result = compute_clustering_metrics(random_mu_embed, n_tokens=100, embed_name='test')
        sil = result.get('test_silhouette_score')
        if isinstance(sil, float):
            assert -1.0 <= sil <= 1.0

    def test_few_tokens_graceful(self):
        from transformer.analysis.semantics import compute_clustering_metrics
        small = torch.randn(5, 4)
        result = compute_clustering_metrics(small, n_tokens=5, embed_name='tiny')
        assert isinstance(result, dict)


# =============================================================================
# Semantic Field Coherence Tests
# =============================================================================

@requires_tiktoken
class TestSemanticFieldCoherence:
    def test_compute_field_coherence_returns_dict(self, random_mu_embed):
        from transformer.analysis.semantics import compute_semantic_field_coherence
        result = compute_semantic_field_coherence(random_mu_embed, embed_name='mu')
        assert isinstance(result, dict)
        assert 'mu_field_coherence_ratio' in result
        assert 'mu_n_fields_resolved' in result

    def test_custom_fields(self, random_mu_embed):
        from transformer.analysis.semantics import compute_semantic_field_coherence
        custom = {'group_a': ['cat', 'dog'], 'group_b': ['the', 'is']}
        result = compute_semantic_field_coherence(random_mu_embed, fields=custom, embed_name='test')
        assert isinstance(result, dict)

    def test_field_coherence_ratio_positive(self, random_mu_embed):
        from transformer.analysis.semantics import compute_semantic_field_coherence
        result = compute_semantic_field_coherence(random_mu_embed, embed_name='mu')
        ratio = result.get('mu_field_coherence_ratio', 0)
        assert ratio >= 0


# =============================================================================
# Sigma Covariance Tests
# =============================================================================

@requires_tiktoken
class TestSigmaSemantics:
    def test_diagonal_sigma(self, random_sigma_diag):
        from transformer.analysis.semantics import analyze_sigma_semantics
        result = analyze_sigma_semantics(random_sigma_diag, n_tokens=100, verbose=False)
        assert result['sigma_mode'] == 'diagonal'
        assert 'sigma_diag_silhouette_score' in result or 'sigma_diag_clustering_error' in result

    def test_full_sigma(self, random_sigma_full):
        from transformer.analysis.semantics import analyze_sigma_semantics
        result = analyze_sigma_semantics(random_sigma_full, n_tokens=50, verbose=False)
        assert result['sigma_mode'] == 'full'
        assert 'sigma_effective_rank_mean' in result or 'sigma_effective_rank_error' in result

    def test_effective_rank_bounded(self, random_sigma_full):
        from transformer.analysis.semantics import analyze_sigma_semantics
        result = analyze_sigma_semantics(random_sigma_full, n_tokens=50, verbose=False)
        eff_rank = result.get('sigma_effective_rank_mean')
        if eff_rank is not None:
            K = random_sigma_full.shape[1]
            assert 1.0 <= eff_rank <= K

    def test_trace_by_category(self, random_sigma_diag):
        from transformer.analysis.semantics import analyze_sigma_semantics
        result = analyze_sigma_semantics(random_sigma_diag, n_tokens=100, verbose=False)
        # Should have at least one category trace
        trace_keys = [k for k in result if k.startswith('sigma_trace_') and k.endswith('_mean')]
        assert len(trace_keys) > 0


# =============================================================================
# Word Pair Analysis Tests
# =============================================================================

@requires_tiktoken
class TestWordPairAnalysis:
    def test_analyze_word_pairs_returns_dict(self, random_mu_embed):
        from transformer.analysis.semantics import analyze_word_pairs
        result = analyze_word_pairs(random_mu_embed)
        assert isinstance(result, dict)
        assert 'pairs' in result

    def test_pairs_list_has_entries(self, random_mu_embed):
        from transformer.analysis.semantics import analyze_word_pairs
        result = analyze_word_pairs(random_mu_embed)
        # Should have at least some resolved pairs
        assert len(result['pairs']) > 0

    def test_expanded_pairs(self, random_mu_embed):
        from transformer.analysis.semantics import analyze_word_pairs
        result = analyze_word_pairs(random_mu_embed)
        # We expanded to 14 pairs, should resolve most of them
        assert len(result['pairs']) >= 6


# =============================================================================
# Gauge Group Identification Tests
# =============================================================================

class TestGaugeGroupIdent:
    def test_so3(self):
        from transformer.analysis.semantics import identify_gauge_group
        assert identify_gauge_group(3) == "SO(3)"

    def test_so2(self):
        from transformer.analysis.semantics import identify_gauge_group
        assert identify_gauge_group(1) == "SO(2)"

    def test_glk(self):
        from transformer.analysis.semantics import identify_gauge_group
        # 9 generators = GL(3) since 3^2 = 9
        assert identify_gauge_group(9) == "GL(3)"

    def test_son(self):
        from transformer.analysis.semantics import identify_gauge_group
        # SO(5) has 10 generators: 5*4/2 = 10
        assert identify_gauge_group(10) == "SO(5)"


# =============================================================================
# Semantic Trajectory Tracker Tests
# =============================================================================

class TestSemanticTrajectoryTracker:
    def test_should_record(self):
        from transformer.analysis.semantics import SemanticTrajectoryTracker
        tracker = SemanticTrajectoryTracker(interval=100)
        assert tracker.should_record(0)
        assert tracker.should_record(100)
        assert not tracker.should_record(50)

    def test_save_load_roundtrip(self, tmp_path):
        from transformer.analysis.semantics import SemanticTrajectoryTracker
        tracker = SemanticTrajectoryTracker()
        tracker.snapshots = [
            {'step': 0, 'mu_silhouette': 0.1, 'mu_field_coherence': 1.2},
            {'step': 5000, 'mu_silhouette': 0.3, 'mu_field_coherence': 1.5},
        ]
        path = tmp_path / "semantic_trajectory.json"
        tracker.save(path)
        assert path.exists()

        loaded = SemanticTrajectoryTracker.load(path)
        assert len(loaded.snapshots) == 2
        assert loaded.snapshots[0]['step'] == 0
        assert loaded.snapshots[1]['mu_silhouette'] == 0.3

    def test_summarize_empty(self):
        from transformer.analysis.semantics import SemanticTrajectoryTracker
        tracker = SemanticTrajectoryTracker()
        summary = tracker.summarize()
        assert 'error' in summary

    def test_summarize_with_data(self):
        from transformer.analysis.semantics import SemanticTrajectoryTracker
        tracker = SemanticTrajectoryTracker()
        tracker.snapshots = [
            {'step': 0, 'mu_silhouette': -0.1, 'mu_field_coherence': 0.8},
            {'step': 5000, 'mu_silhouette': 0.05, 'mu_field_coherence': 1.1},
            {'step': 10000, 'mu_silhouette': 0.2, 'mu_field_coherence': 1.4},
        ]
        summary = tracker.summarize()
        assert summary['n_snapshots'] == 3
        assert summary['mu_silhouette_initial'] == -0.1
        assert summary['mu_silhouette_final'] == 0.2
        assert 'mu_silhouette_positive_at_step' in summary
        assert summary['mu_silhouette_positive_at_step'] == 5000


# =============================================================================
# Integration: analyze_gauge_semantics
# =============================================================================

@requires_tiktoken
class TestAnalyzeGaugeSemantics:
    def test_basic_call(self, random_mu_embed, random_phi_embed):
        from transformer.analysis.semantics import analyze_gauge_semantics
        result = analyze_gauge_semantics(
            mu_embed=random_mu_embed,
            phi_embed=random_phi_embed,
            save_plots=False,
            verbose=False,
        )
        assert 'mu_clustering' in result
        assert 'mu_field_coherence' in result
        assert 'phi_clustering' in result
        assert 'phi_field_coherence' in result

    def test_mu_only(self, random_mu_embed):
        from transformer.analysis.semantics import analyze_gauge_semantics
        result = analyze_gauge_semantics(
            mu_embed=random_mu_embed,
            save_plots=False,
            verbose=False,
        )
        assert 'mu_clustering' in result
        assert 'error' not in result


class TestAnalyzeGaugeSemanticsNoToken:
    """Tests that don't require tiktoken."""

    def test_no_embeddings_returns_error(self):
        from transformer.analysis.semantics import analyze_gauge_semantics
        result = analyze_gauge_semantics(save_plots=False, verbose=False)
        assert 'error' in result


# =============================================================================
# Omega Metrics Tests (no model needed)
# =============================================================================

class TestOmegaMetrics:
    def test_frobenius_dist(self):
        from transformer.analysis.semantics import omega_frobenius_dist
        omega = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
        omega[1, 0, 1] = 0.5
        d = omega_frobenius_dist(omega, 0, 1)
        assert d > 0

    def test_geodesic_dist_identity(self):
        from transformer.analysis.semantics import omega_geodesic_dist
        omega = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
        d = omega_geodesic_dist(omega, 0, 1)
        assert abs(d) < 1e-4  # Same matrix => distance ~0

    @requires_tiktoken
    def test_compute_omega_clustering(self):
        from transformer.analysis.semantics import compute_omega_clustering_metrics
        # Small random group elements
        torch.manual_seed(99)
        omega = torch.eye(3).unsqueeze(0).repeat(50, 1, 1) + torch.randn(50, 3, 3) * 0.05
        result = compute_omega_clustering_metrics(omega, n_tokens=50)
        assert 'omega_det_mean' in result
        assert 'omega_identity_dist_mean' in result
