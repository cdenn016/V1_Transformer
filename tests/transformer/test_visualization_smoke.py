"""
Smoke tests for transformer.visualization modules
====================================================

Verifies that each visualization module imports without error and that
key plotting functions run without crashing using synthetic data and
the Agg (non-interactive) matplotlib backend.

These are smoke tests only — they verify no exceptions are raised,
not visual correctness of the figures.
"""

import pytest
import numpy as np
import torch

# Gate matplotlib — viz tests require the optional 'viz' dependency group
matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# TestVisualizationImports
# =============================================================================

class TestVisualizationImports:
    """All visualization modules importable without error."""

    def test_all_modules_importable(self):
        """Import every visualization module."""
        modules = [
            'transformer.visualization.ablation_plots',
            'transformer.visualization.belief_space_viz',
            'transformer.visualization.holonomy_plots',
            'transformer.visualization.interactive_belief_viz',
            'transformer.visualization.pub_style',
            'transformer.visualization.training_plots',
            'transformer.visualization.vfe_dynamics_plots',
        ]
        import importlib
        for mod_name in modules:
            mod = importlib.import_module(mod_name)
            assert mod is not None, f"Failed to import {mod_name}"


# =============================================================================
# TestPubStyle
# =============================================================================

class TestPubStyle:
    """pub_style.py: publication style utilities."""

    def test_set_pub_style(self):
        """set_pub_style runs without error."""
        from transformer.visualization.pub_style import set_pub_style
        set_pub_style()
        plt.close('all')


# =============================================================================
# TestHolonomyPlots
# =============================================================================

class TestHolonomyPlotsSmoke:
    """holonomy_plots.py: holonomy visualization functions."""

    def test_plot_holonomy_distribution(self):
        """plot_holonomy_distribution with synthetic data."""
        from transformer.visualization.holonomy_plots import plot_holonomy_distribution
        norms = np.random.exponential(0.1, size=100)
        fig = plot_holonomy_distribution(norms)
        plt.close('all')

    def test_plot_holonomy_evolution(self):
        """plot_holonomy_evolution with synthetic arrays."""
        from transformer.visualization.holonomy_plots import plot_holonomy_evolution
        n = 10
        steps = np.arange(n)
        global_mean = np.random.exponential(0.05, size=n)
        global_max = np.random.exponential(0.1, size=n)
        fig = plot_holonomy_evolution(steps, global_mean, global_max)
        plt.close('all')


# =============================================================================
# TestTrainingPlots
# =============================================================================

class TestTrainingPlotsSmoke:
    """training_plots.py: training curve visualization."""

    def test_load_metrics_csv_missing_file(self):
        """load_metrics_csv raises FileNotFoundError on missing file."""
        from transformer.visualization.training_plots import load_metrics_csv
        from pathlib import Path
        with pytest.raises(FileNotFoundError):
            load_metrics_csv(Path('/nonexistent/metrics.csv'))

    def test_load_metrics_csv_valid(self):
        """load_metrics_csv loads a valid CSV."""
        from transformer.visualization.training_plots import load_metrics_csv
        from pathlib import Path
        import tempfile, csv
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'train_loss_ce', 'val_ce'])
            writer.writerow([1, 4.5, 4.0])
            writer.writerow([2, 4.0, 3.5])
            path = f.name
        result = load_metrics_csv(Path(path))
        assert isinstance(result, dict)
        import os
        os.unlink(path)


# =============================================================================
# TestVFEDynamicsPlots
# =============================================================================

class TestVFEDynamicsPlotsSmoke:
    """vfe_dynamics_plots.py: VFE gradient dynamics visualization."""

    def test_load_vfe_metrics_missing_file(self):
        """load_vfe_metrics handles missing file."""
        from transformer.visualization.vfe_dynamics_plots import load_vfe_metrics
        from pathlib import Path
        try:
            result = load_vfe_metrics(Path('/nonexistent/metrics.csv'))
            assert result is None or isinstance(result, dict)
        except (FileNotFoundError, OSError):
            pass  # Also acceptable


# =============================================================================
# TestBeliefSpaceViz
# =============================================================================

class TestBeliefSpaceVizSmoke:
    """belief_space_viz.py: belief space visualization."""

    def test_visualize_belief_space(self):
        """visualize_belief_space with synthetic embeddings."""
        from transformer.visualization.belief_space_viz import visualize_belief_space
        K = 8
        n_tokens = 50
        mu = np.random.randn(n_tokens, K)
        # valid_tokens: list of token strings
        valid = [f'tok_{i}' for i in range(n_tokens)]
        # token_categories: list of category labels matching CATEGORY_COLORS keys
        cat_names = ['animals', 'food', 'objects', 'places', 'actions']
        categories = [cat_names[i % len(cat_names)] for i in range(n_tokens)]
        fig = visualize_belief_space(mu, valid, categories)
        plt.close('all')
