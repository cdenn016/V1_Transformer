"""
Tests for holonomy metrics and visualization.

Covers:
    - HolonomySnapshot / HolonomyProfile data classes
    - compute_holonomy_snapshot metric computation
    - compute_curvature_by_distance binning
    - compute_flatness_trajectory extraction
    - Visualization figure generation (smoke tests)
"""

import pytest
import torch
import numpy as np


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def flat_exp_delta():
    """Identity exp_delta (flat transport) — shape (B, N, N, K, K)."""
    B, N, K = 2, 12, 3
    return torch.eye(K).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, N, K, K).clone()


@pytest.fixture
def curved_exp_delta():
    """Perturbed exp_delta (non-flat transport)."""
    B, N, K = 2, 12, 3
    exp_delta = torch.eye(K).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, N, K, K).clone()
    torch.manual_seed(42)
    exp_delta = exp_delta + 0.1 * torch.randn(B, N, N, K, K)
    return exp_delta


# =============================================================================
# HolonomySnapshot Tests
# =============================================================================

class TestHolonomySnapshot:

    def test_flat_transport_gives_zero_metrics(self, flat_exp_delta):
        from transformer.analysis.holonomy_metrics import compute_holonomy_snapshot

        snap = compute_holonomy_snapshot(flat_exp_delta, step=0, layer=0, head=0, sample_size=100)
        assert snap.mean_norm < 1e-5
        assert snap.max_norm < 1e-5
        assert snap.frac_gt_001 == 0.0
        assert snap.frac_gt_01 == 0.0

    def test_curved_transport_gives_nonzero_metrics(self, curved_exp_delta):
        from transformer.analysis.holonomy_metrics import compute_holonomy_snapshot

        snap = compute_holonomy_snapshot(curved_exp_delta, step=100, layer=0, head=0, sample_size=200)
        assert snap.mean_norm > 0.01
        assert snap.max_norm > snap.mean_norm
        assert snap.frac_gt_001 > 0

    def test_spectral_gap_computed(self, curved_exp_delta):
        from transformer.analysis.holonomy_metrics import compute_holonomy_snapshot

        snap = compute_holonomy_snapshot(curved_exp_delta, step=0, layer=0, head=0, sample_size=50)
        # Spectral gap should be non-negative
        assert snap.mean_spectral_gap >= 0

    def test_wilson_trace_flat(self, flat_exp_delta):
        from transformer.analysis.holonomy_metrics import compute_holonomy_snapshot

        snap = compute_holonomy_snapshot(flat_exp_delta, step=0, layer=0, head=0, sample_size=50)
        # |tr(I)| / K = K / K = 1.0 for flat (identity) transport
        assert abs(snap.mean_wilson_trace - 1.0) < 1e-5

    def test_wilson_trace_curved(self, curved_exp_delta):
        from transformer.analysis.holonomy_metrics import compute_holonomy_snapshot

        snap = compute_holonomy_snapshot(curved_exp_delta, step=0, layer=0, head=0, sample_size=50)
        assert snap.mean_wilson_trace > 0

    def test_to_log_dict_format(self, flat_exp_delta):
        from transformer.analysis.holonomy_metrics import compute_holonomy_snapshot

        snap = compute_holonomy_snapshot(flat_exp_delta, step=0, layer=2, head=1, sample_size=50)
        log_dict = snap.to_log_dict(prefix='test')
        assert 'test/L2_H1/mean_norm' in log_dict
        assert 'test/L2_H1/spectral_gap' in log_dict
        assert all(isinstance(v, float) for v in log_dict.values())


# =============================================================================
# HolonomyProfile Tests
# =============================================================================

class TestHolonomyProfile:

    def test_profile_aggregation(self, curved_exp_delta):
        from transformer.analysis.holonomy_metrics import compute_holonomy_snapshot, HolonomyProfile

        snaps = []
        for layer in range(3):
            snap = compute_holonomy_snapshot(curved_exp_delta, step=100, layer=layer, head=0, sample_size=50)
            snaps.append(snap)

        profile = HolonomyProfile(
            step=100,
            snapshots=snaps,
            global_mean_norm=float(np.mean([s.mean_norm for s in snaps])),
            global_max_norm=float(np.max([s.max_norm for s in snaps])),
        )

        assert profile.global_mean_norm > 0
        assert profile.global_max_norm >= profile.global_mean_norm

        log_dict = profile.to_log_dict()
        assert 'holonomy/global_mean_norm' in log_dict
        assert len(log_dict) == 2 + 3 * 9  # 2 global + 3 layers * 9 metrics each


# =============================================================================
# Curvature by Distance Tests
# =============================================================================

class TestCurvatureByDistance:

    def test_output_shapes(self, curved_exp_delta):
        from transformer.analysis.holonomy_metrics import compute_curvature_by_distance

        result = compute_curvature_by_distance(curved_exp_delta, max_distance=5, samples_per_distance=200)
        assert len(result['distances']) > 0
        assert result['distances'].shape == result['mean_norms'].shape
        assert result['distances'].shape == result['std_norms'].shape

    def test_distances_are_sequential(self, curved_exp_delta):
        from transformer.analysis.holonomy_metrics import compute_curvature_by_distance

        result = compute_curvature_by_distance(curved_exp_delta, max_distance=5, samples_per_distance=200)
        # Distances should be monotonically increasing
        assert all(result['distances'][i] < result['distances'][i + 1]
                    for i in range(len(result['distances']) - 1))

    def test_flat_transport_low_norms(self, flat_exp_delta):
        from transformer.analysis.holonomy_metrics import compute_curvature_by_distance

        result = compute_curvature_by_distance(flat_exp_delta, max_distance=5, samples_per_distance=100)
        assert result['mean_norms'].max() < 1e-5


# =============================================================================
# Flatness Trajectory Tests
# =============================================================================

class TestFlatnessTrajectory:

    def test_trajectory_extraction(self, curved_exp_delta):
        from transformer.analysis.holonomy_metrics import (
            compute_holonomy_snapshot, compute_flatness_trajectory, HolonomyProfile,
        )

        history = []
        for step in [100, 200, 300]:
            snap = compute_holonomy_snapshot(curved_exp_delta, step=step, layer=0, head=0, sample_size=50)
            profile = HolonomyProfile(
                step=step, snapshots=[snap],
                global_mean_norm=snap.mean_norm, global_max_norm=snap.max_norm,
            )
            history.append(profile)

        traj = compute_flatness_trajectory(history)
        assert np.array_equal(traj['steps'], [100, 200, 300])
        assert len(traj['global_mean']) == 3
        # per_layer_mean is a dict mapping layer index -> array of means
        assert 0 in traj['per_layer_mean']
        assert traj['per_layer_mean'][0].shape == (3,)  # 3 steps for layer 0


# =============================================================================
# Visualization Smoke Tests
# =============================================================================

class TestVisualization:
    """Smoke tests: ensure figures are created without errors."""

    @pytest.fixture(autouse=True)
    def _require_matplotlib(self):
        pytest.importorskip('matplotlib')

    def test_plot_holonomy_distribution(self):
        from transformer.visualization.holonomy_plots import plot_holonomy_distribution

        norms = np.random.exponential(0.1, size=500)
        fig = plot_holonomy_distribution(norms, title='Test Distribution')
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_holonomy_evolution(self):
        from transformer.visualization.holonomy_plots import plot_holonomy_evolution

        steps = np.arange(0, 1000, 100)
        mean = np.exp(-steps / 500) * 0.5
        maxx = mean * 3
        fig = plot_holonomy_evolution(steps, mean, maxx)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_layer_profile(self):
        from transformer.visualization.holonomy_plots import plot_layer_holonomy_profile

        layer_means = np.array([0.05, 0.12, 0.08, 0.15])
        layer_stds = np.array([0.01, 0.03, 0.02, 0.04])
        fig = plot_layer_holonomy_profile(layer_means, layer_stds)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_curvature_vs_distance(self):
        from transformer.visualization.holonomy_plots import plot_curvature_vs_distance

        bins = np.arange(1, 11, dtype=float)
        means = 0.1 * np.exp(-bins / 5)
        stds = means * 0.2
        counts = np.full(10, 50)
        fig = plot_curvature_vs_distance(bins, means, stds, counts)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_wilson_spectrum(self):
        from transformer.visualization.holonomy_plots import plot_wilson_spectrum

        K = 3
        C = np.eye(K)[None].repeat(100, axis=0) + np.random.randn(100, K, K) * 0.05
        fig = plot_wilson_spectrum(C)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_flat_vs_nonflat(self):
        from transformer.visualization.holonomy_plots import plot_flat_vs_nonflat_comparison

        steps = np.arange(0, 1000, 10)
        flat_ppl = 100 * np.exp(-steps / 300) + 20
        nonflat_ppl = 100 * np.exp(-steps / 250) + 18
        fig = plot_flat_vs_nonflat_comparison(steps, flat_ppl, steps, nonflat_ppl)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_summary(self):
        from transformer.visualization.holonomy_plots import plot_holonomy_summary

        norms = np.random.exponential(0.1, size=500)
        steps = np.arange(0, 500, 50)
        mean = np.exp(-steps / 200) * 0.3
        maxx = mean * 2.5
        layer_means = np.array([0.05, 0.12, 0.08, 0.15])
        dist = {
            'bin_centers': np.arange(1, 6, dtype=float),
            'mean_norms': np.array([0.1, 0.08, 0.06, 0.04, 0.02]),
            'std_norms': np.array([0.02] * 5),
            'counts': np.array([100] * 5),
        }
        fig = plot_holonomy_summary(norms, steps, mean, maxx, layer_means, dist)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


