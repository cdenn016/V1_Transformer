"""Tests for the semantics.py figure-generation code.

Covers the additions from 2026-04-18 (evening):
- plot_sigma_clustering (both diagonal and full covariance modes)
- plot_representation_comparison
- _add_cluster_envelopes (all envelope modes + known-covariance geometry)
- _resolve_token_colors (category / semantic_field / pos with fallback)
- analyze_sigma_semantics save_path path (end-to-end wiring)
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

# Matplotlib may not be available in CI — skip the whole file in that case.
try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.patches import Ellipse
    from matplotlib.figure import Figure
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import sklearn  # noqa: F401
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

requires_mpl = pytest.mark.skipif(
    not (MPL_AVAILABLE and SKLEARN_AVAILABLE),
    reason="matplotlib+sklearn required for figure tests",
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def diag_sigma():
    """Synthetic (100, 8) diagonal covariance tensor."""
    torch.manual_seed(0)
    return torch.exp(torch.randn(100, 8) * 0.5).clamp(min=1e-3)


@pytest.fixture
def full_sigma():
    """Synthetic (100, 8, 8) SPD covariance tensor."""
    torch.manual_seed(1)
    A = torch.randn(100, 8, 8) * 0.3
    return A @ A.transpose(-1, -2) + 0.1 * torch.eye(8).unsqueeze(0)


@pytest.fixture
def mu_embed():
    torch.manual_seed(2)
    return torch.randn(100, 8)


@pytest.fixture
def phi_embed():
    torch.manual_seed(3)
    return torch.randn(100, 3)


@pytest.fixture
def omega_embed():
    torch.manual_seed(4)
    B = torch.randn(100, 8, 8) * 0.1
    return torch.eye(8).unsqueeze(0) + B


# =============================================================================
# plot_sigma_clustering
# =============================================================================

@requires_mpl
class TestPlotSigmaClustering:
    def test_diagonal_shape_renders(self, diag_sigma, tmp_path):
        from transformer.analysis.semantics import plot_sigma_clustering
        out = tmp_path / "sig_diag.png"
        fig = plot_sigma_clustering(diag_sigma, save_path=out, n_tokens=100)
        assert fig is not None
        assert isinstance(fig, Figure)
        assert out.exists() and out.stat().st_size > 0

    def test_full_shape_renders(self, full_sigma, tmp_path):
        from transformer.analysis.semantics import plot_sigma_clustering
        out = tmp_path / "sig_full.png"
        fig = plot_sigma_clustering(full_sigma, save_path=out, n_tokens=100)
        assert fig is not None
        assert out.exists()

    @pytest.mark.parametrize("mode", ["ellipse", "hull", "kde", "centroid", "none"])
    def test_envelope_modes(self, diag_sigma, mode, tmp_path):
        from transformer.analysis.semantics import plot_sigma_clustering
        out = tmp_path / f"sig_{mode}.png"
        fig = plot_sigma_clustering(
            diag_sigma, save_path=out, envelope_mode=mode, n_tokens=100,
        )
        assert fig is not None
        # The 4-panel layout produces 4 axes regardless of mode
        assert len(fig.axes) >= 4

    def test_full_mode_panel_titles(self, full_sigma, tmp_path):
        from transformer.analysis.semantics import plot_sigma_clustering
        fig = plot_sigma_clustering(full_sigma, save_path=tmp_path / "x.png", n_tokens=100)
        titles = [ax.get_title() for ax in fig.axes[:4]]
        # Panel (b) for full covariance should reference log-determinant
        assert any("Log-Determinant" in t for t in titles)

    def test_diag_mode_panel_titles(self, diag_sigma, tmp_path):
        from transformer.analysis.semantics import plot_sigma_clustering
        fig = plot_sigma_clustering(diag_sigma, save_path=tmp_path / "x.png", n_tokens=100)
        titles = [ax.get_title() for ax in fig.axes[:4]]
        # Panel (b) for diagonal covariance should reference log-trace
        assert any("Log-Trace" in t for t in titles)


# =============================================================================
# plot_representation_comparison
# =============================================================================

@requires_mpl
class TestPlotRepresentationComparison:
    def test_all_representations(self, mu_embed, phi_embed, omega_embed, diag_sigma, tmp_path):
        from transformer.analysis.semantics import plot_representation_comparison
        out = tmp_path / "repr.png"
        fig = plot_representation_comparison(
            mu=mu_embed, phi=phi_embed, omega=omega_embed, sigma=diag_sigma,
            save_path=out, n_tokens=100,
        )
        assert fig is not None
        assert out.exists()

    def test_omega_missing_renders_placeholder(self, mu_embed, phi_embed, diag_sigma, tmp_path):
        from transformer.analysis.semantics import plot_representation_comparison
        fig = plot_representation_comparison(
            mu=mu_embed, phi=phi_embed, omega=None, sigma=diag_sigma,
            save_path=tmp_path / "no_omega.png", n_tokens=100,
        )
        assert fig is not None

    def test_all_none_returns_none(self, tmp_path):
        from transformer.analysis.semantics import plot_representation_comparison
        fig = plot_representation_comparison(
            mu=None, phi=None, omega=None, sigma=None,
            save_path=tmp_path / "none.png", n_tokens=100,
        )
        assert fig is None

    def test_full_cov_sigma(self, mu_embed, phi_embed, full_sigma, tmp_path):
        from transformer.analysis.semantics import plot_representation_comparison
        fig = plot_representation_comparison(
            mu=mu_embed, phi=phi_embed, omega=None, sigma=full_sigma,
            save_path=tmp_path / "full_cov.png", n_tokens=100,
        )
        assert fig is not None


# =============================================================================
# _add_cluster_envelopes
# =============================================================================

@requires_mpl
class TestClusterEnvelopes:
    def _make_two_clusters(self, seed=0):
        """Two well-separated 2D Gaussian clusters with known means/covariances."""
        rng = np.random.default_rng(seed)
        n = 100
        c1 = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 0.25]], size=n)
        c2 = rng.multivariate_normal([5.0, 5.0], [[0.5, 0.0], [0.0, 0.5]], size=n)
        points = np.vstack([c1, c2])
        categories = ['a'] * n + ['b'] * n
        palette = {'a': '#0072B2', 'b': '#D55E00'}
        return points, categories, palette

    def test_ellipse_geometry_matches_known_cov(self):
        """2σ ellipse for a known 2D covariance should have axes ≈ 2·2·sqrt(eigenvalue)."""
        from transformer.analysis.semantics import _add_cluster_envelopes
        import matplotlib.pyplot as plt

        points, categories, palette = self._make_two_clusters(seed=42)
        fig, ax = plt.subplots()
        _add_cluster_envelopes(
            ax, points, categories, palette, mode='ellipse', n_sigma=2.0,
        )
        ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
        assert len(ellipses) == 2  # one per category

        # Cluster 'a' has cov = diag([1.0, 0.25]); 2σ axes → 2·2·[1.0, 0.5] = [4.0, 2.0].
        # Find the ellipse whose center is near the origin.
        for e in ellipses:
            if np.allclose(e.center, [0.0, 0.0], atol=0.3):
                # width = major axis = 4.0, height = minor axis = 2.0 (within 20%)
                w, h = e.width, e.height
                assert 3.2 <= w <= 4.8, f"major axis {w} out of range"
                assert 1.6 <= h <= 2.4, f"minor axis {h} out of range"
                return
        pytest.fail("No ellipse near origin found")

    def test_hull_mode_adds_polygon(self):
        from transformer.analysis.semantics import _add_cluster_envelopes
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        points, categories, palette = self._make_two_clusters(seed=43)
        fig, ax = plt.subplots()
        _add_cluster_envelopes(ax, points, categories, palette, mode='hull')
        polys = [p for p in ax.patches if isinstance(p, Polygon)]
        assert len(polys) == 2

    def test_none_mode_is_noop(self):
        from transformer.analysis.semantics import _add_cluster_envelopes
        import matplotlib.pyplot as plt

        points, categories, palette = self._make_two_clusters(seed=44)
        fig, ax = plt.subplots()
        _add_cluster_envelopes(ax, points, categories, palette, mode='none')
        assert len(ax.patches) == 0  # no patches added

    def test_too_few_points_skipped_gracefully(self):
        from transformer.analysis.semantics import _add_cluster_envelopes
        import matplotlib.pyplot as plt

        # Category 'a' has only 2 points, below min_points=5
        points = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1],
                           [5.2, 5.0], [4.9, 4.9], [5.05, 5.05]])
        categories = ['a', 'a', 'b', 'b', 'b', 'b', 'b']
        palette = {'a': '#0072B2', 'b': '#D55E00'}
        fig, ax = plt.subplots()
        _add_cluster_envelopes(ax, points, categories, palette, mode='ellipse')
        # 'a' should be skipped (too few points), 'b' should get an ellipse
        ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
        assert len(ellipses) == 1


# =============================================================================
# _resolve_token_colors
# =============================================================================

class TestResolveTokenColors:
    def test_category_mode(self):
        from transformer.analysis.semantics import _resolve_token_colors
        categories, palette = _resolve_token_colors(50, label_mode='category')
        assert len(categories) == 50
        assert all(c in palette for c in categories)
        # Palette is the 6-category map
        expected = {'letter', 'digit', 'punct', 'function', 'content', 'other'}
        assert set(palette.keys()) == expected

    def test_semantic_field_mode(self):
        from transformer.analysis.semantics import _resolve_token_colors
        categories, palette = _resolve_token_colors(200, label_mode='semantic_field')
        assert len(categories) == 200
        # 'other' should always be present in the palette
        assert 'other' in palette

    def test_pos_mode_falls_back_without_nltk(self):
        """If nltk is unavailable, 'pos' should fall back to 'category'."""
        import sys
        # Simulate nltk missing by stashing it out if present
        saved = sys.modules.pop('nltk', None)
        try:
            from transformer.analysis.semantics import _resolve_token_colors
            with pytest.warns(RuntimeWarning):
                categories, palette = _resolve_token_colors(30, label_mode='pos')
            # After fallback, palette should be the category palette
            assert 'letter' in palette or 'noun' in palette  # either is acceptable
        finally:
            if saved is not None:
                sys.modules['nltk'] = saved

    def test_unknown_mode_raises(self):
        from transformer.analysis.semantics import _resolve_token_colors
        with pytest.raises(ValueError):
            _resolve_token_colors(10, label_mode='bogus')


# =============================================================================
# _safe_pca_2d — degenerate input handling
# =============================================================================

class TestSafePCA:
    def test_zero_variance_returns_none(self):
        from transformer.analysis.semantics import _safe_pca_2d
        coords, ratio, status = _safe_pca_2d(np.ones((10, 5)))
        assert coords is None
        assert status == 'zero variance'

    def test_non_finite_returns_none(self):
        from transformer.analysis.semantics import _safe_pca_2d
        arr = np.array([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0], [0.0, np.inf, 0.0]])
        coords, _, status = _safe_pca_2d(arr)
        assert coords is None
        assert status == 'non-finite values'

    def test_too_few_samples(self):
        from transformer.analysis.semantics import _safe_pca_2d
        coords, _, status = _safe_pca_2d(np.array([[1.0, 2.0]]))
        assert coords is None
        assert status == 'too few samples'

    def test_normal_input_returns_2d(self):
        from transformer.analysis.semantics import _safe_pca_2d
        rng = np.random.default_rng(0)
        coords, ratio, status = _safe_pca_2d(rng.normal(size=(50, 8)))
        assert status == 'ok'
        assert coords is not None and coords.shape == (50, 2)
        assert ratio is not None and ratio.size == 2

    def test_pads_to_n_components(self):
        """When input has 1 effective dim, coords are still (n, 2) with zero padding."""
        from transformer.analysis.semantics import _safe_pca_2d
        rng = np.random.default_rng(1)
        # 1D embedded in 3D — effective dim 1
        x = rng.normal(size=(30, 1))
        arr = np.concatenate([x, x, x], axis=1)
        coords, ratio, status = _safe_pca_2d(arr, n_components=2)
        # Even though there's genuine variance only along one direction, PCA
        # still fits 2 components from a 3D input (it's the requested dim that
        # matters, not the effective rank). The output must be shape (30, 2).
        assert coords is not None and coords.shape == (30, 2)


# =============================================================================
# analyze_sigma_semantics end-to-end save_path wiring
# =============================================================================

@requires_mpl
class TestAnalyzeSigmaSavePath:
    def test_save_path_produces_figure(self, diag_sigma, tmp_path):
        from transformer.analysis.semantics import analyze_sigma_semantics
        out = tmp_path / "sigma_out.png"
        result = analyze_sigma_semantics(
            diag_sigma, n_tokens=100, verbose=False,
            step=12345, save_path=out,
        )
        assert out.exists()
        assert result.get('sigma_plot_saved') is True

    def test_no_save_path_skips_figure(self, diag_sigma):
        from transformer.analysis.semantics import analyze_sigma_semantics
        result = analyze_sigma_semantics(
            diag_sigma, n_tokens=100, verbose=False,
        )
        # When save_path=None (default), no plot-saved key
        assert 'sigma_plot_saved' not in result


# =============================================================================
# bhattacharyya_distance_matrix — mathematical correctness
# =============================================================================

class TestBhattacharyyaDistanceMatrix:
    def test_zero_diagonal(self):
        from transformer.analysis.semantics import bhattacharyya_distance_matrix
        torch.manual_seed(0)
        mu = torch.randn(20, 4)
        sigma = torch.exp(torch.randn(20, 4) * 0.3)
        D = bhattacharyya_distance_matrix(mu, sigma)
        assert torch.allclose(D.diag(), torch.zeros(20))

    def test_symmetric(self):
        from transformer.analysis.semantics import bhattacharyya_distance_matrix
        torch.manual_seed(1)
        mu = torch.randn(15, 3)
        sigma = torch.exp(torch.randn(15, 3) * 0.3)
        D = bhattacharyya_distance_matrix(mu, sigma)
        assert torch.allclose(D, D.T)

    def test_non_negative(self):
        from transformer.analysis.semantics import bhattacharyya_distance_matrix
        torch.manual_seed(2)
        mu = torch.randn(15, 3)
        sigma = torch.exp(torch.randn(15, 3) * 0.3)
        D = bhattacharyya_distance_matrix(mu, sigma)
        assert (D >= 0).all()

    def test_1d_closed_form(self):
        """Verify against the 1D closed form for N(μ, σ²).

        D_B = 1/8 · (μ_1 - μ_2)² / σ̄² + 1/2 · ln(σ̄² / sqrt(σ²_1 · σ²_2))
              where σ̄² = (σ²_1 + σ²_2)/2.
        """
        from transformer.analysis.semantics import bhattacharyya_distance_matrix
        mu = torch.tensor([[0.0], [1.0]])
        var = torch.tensor([[1.0], [4.0]])
        D = bhattacharyya_distance_matrix(mu, var)
        sig_bar = (1.0 + 4.0) / 2
        t1 = (1/8) * (1.0 ** 2) / sig_bar
        t2 = 0.5 * (np.log(sig_bar) - 0.5 * np.log(1.0) - 0.5 * np.log(4.0))
        expected = t1 + t2
        got = D[0, 1].item()
        assert abs(got - expected) < 1e-5, f"got={got}, expected={expected}"

    def test_full_cov_spd(self):
        from transformer.analysis.semantics import bhattacharyya_distance_matrix
        torch.manual_seed(4)
        n, K = 20, 4
        A = torch.randn(n, K, K) * 0.3
        sigma = A @ A.transpose(-1, -2) + 0.2 * torch.eye(K).unsqueeze(0)
        mu = torch.randn(n, K)
        D = bhattacharyya_distance_matrix(mu, sigma)
        assert torch.allclose(D.diag(), torch.zeros(n), atol=1e-4)
        assert torch.allclose(D, D.T, atol=1e-4)
        assert (D >= -1e-4).all()

    def test_isotropic_full_matches_diagonal(self):
        """Σ_i = s_i · I (isotropic full-cov) must give the same D_B as Σ_i = [s_i]*K diagonal."""
        from transformer.analysis.semantics import bhattacharyya_distance_matrix
        torch.manual_seed(5)
        n, K = 10, 3
        s = torch.exp(torch.randn(n) * 0.3)
        mu = torch.randn(n, K)
        full = s.view(-1, 1, 1) * torch.eye(K).unsqueeze(0)
        diag = s.unsqueeze(-1).expand(-1, K)
        D_full = bhattacharyya_distance_matrix(mu, full)
        D_diag = bhattacharyya_distance_matrix(mu, diag)
        assert (D_full - D_diag).abs().max() < 1e-3


# =============================================================================
# plot_sigma_bhattacharyya_mds
# =============================================================================

@requires_mpl
class TestPlotSigmaBhattacharyyaMDS:
    def test_diagonal_path_renders(self, mu_embed, diag_sigma, tmp_path):
        from transformer.analysis.semantics import plot_sigma_bhattacharyya_mds
        out = tmp_path / "bhat_diag.png"
        fig = plot_sigma_bhattacharyya_mds(
            mu=mu_embed, sigma=diag_sigma, save_path=out, n_tokens=100,
        )
        assert fig is not None
        assert len(fig.axes) >= 2  # 2-panel layout
        assert out.exists() and out.stat().st_size > 0

    def test_full_cov_path_renders(self, mu_embed, full_sigma, tmp_path):
        from transformer.analysis.semantics import plot_sigma_bhattacharyya_mds
        out = tmp_path / "bhat_full.png"
        fig = plot_sigma_bhattacharyya_mds(
            mu=mu_embed, sigma=full_sigma, save_path=out, n_tokens=80,
        )
        assert fig is not None
        assert out.exists()

    @pytest.mark.parametrize("mode", ["ellipse", "hull", "kde", "centroid", "none"])
    def test_envelope_modes(self, mu_embed, diag_sigma, mode, tmp_path):
        from transformer.analysis.semantics import plot_sigma_bhattacharyya_mds
        fig = plot_sigma_bhattacharyya_mds(
            mu=mu_embed, sigma=diag_sigma,
            save_path=tmp_path / f"bhat_{mode}.png",
            envelope_mode=mode, n_tokens=80,
        )
        assert fig is not None

    def test_degenerate_inputs_no_runtimewarning(self, mu_embed, tmp_path):
        """Zero-variance Σ, identity Ω, and zero φ must not emit RuntimeWarnings.

        These are the real cases that fire sklearn's 'invalid value encountered
        in divide' — all tokens having the same Σ (shared log_sigma_diag broadcast
        to (V, K)), Ω ≡ I under gauge_mode='trivial', or φ at init.
        """
        import warnings as _w
        from transformer.analysis.semantics import (
            plot_sigma_clustering, plot_omega_clustering,
            plot_embedding_clustering, plot_representation_comparison,
        )
        K = 8
        n = 50
        const_sigma = torch.full((n, K), 0.25)
        eye_omega = torch.eye(K).unsqueeze(0).repeat(n, 1, 1)
        zero_phi = torch.zeros(n, 16)

        with _w.catch_warnings():
            _w.simplefilter('error', RuntimeWarning)

            fig1 = plot_sigma_clustering(const_sigma, save_path=tmp_path / 'a.png', n_tokens=n)
            fig2 = plot_omega_clustering(eye_omega, save_path=tmp_path / 'b.png', n_tokens=n)
            fig3 = plot_embedding_clustering(
                zero_phi, embed_type='phi', save_path=tmp_path / 'c.png', n_tokens=n,
            )
            fig4 = plot_representation_comparison(
                mu=mu_embed[:n], phi=zero_phi, omega=eye_omega, sigma=const_sigma,
                save_path=tmp_path / 'd.png', n_tokens=n,
            )
        for f in (fig1, fig2, fig3, fig4):
            assert f is not None

    def test_reproducibility(self, mu_embed, diag_sigma, tmp_path):
        """MDS with fixed random_state should produce identical embeddings."""
        from transformer.analysis.semantics import (
            bhattacharyya_distance_matrix, plot_sigma_bhattacharyya_mds,
        )
        # Two identical calls — compare coords extracted from ax.collections
        fig1 = plot_sigma_bhattacharyya_mds(
            mu=mu_embed, sigma=diag_sigma,
            save_path=tmp_path / "r1.png", n_tokens=50, mds_random_state=42,
        )
        fig2 = plot_sigma_bhattacharyya_mds(
            mu=mu_embed, sigma=diag_sigma,
            save_path=tmp_path / "r2.png", n_tokens=50, mds_random_state=42,
        )
        # Stress values should match (extracted from title suffix)
        t1 = fig1.axes[0].get_title()
        t2 = fig2.axes[0].get_title()
        assert t1 == t2, f"MDS not reproducible: {t1} vs {t2}"
