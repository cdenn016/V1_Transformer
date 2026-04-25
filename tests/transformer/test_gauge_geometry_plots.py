"""
Tests for Yang-Mills plot log-scale guard and discrete holonomy YM computation.

Covers the fix for the end-of-publication-run UserWarning emitted when
`plot_yang_mills_evolution` log-scaled a placeholder all-zero series.

- Flat bilinear transport: YM energy is exactly zero by the cocycle identity,
  so the plotter must skip the YM axis rather than log-scaling zeros.
- Positive YM: the plotter must emit the two-axis layout and guard the log
  scale on both axes.
"""

import warnings

import numpy as np
import pytest
import torch

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformer.visualization.gauge_geometry_plots import (
    plot_yang_mills_evolution,
)
from transformer.analysis.gauge_geometry import (
    compute_yang_mills_energy_from_omega,
    _build_phi_matrix,
)
from transformer.core.gauge_utils import stable_matrix_exp_pair


def _assert_no_user_warning(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return fn()


class TestPlotYangMillsEvolutionGuard:
    """Log-scale guards on the Yang-Mills evolution plot."""

    def test_flat_transport_no_user_warning(self):
        """All-zero YM + positive Dirichlet must not trigger a log-scale warning."""
        steps = [0, 2000, 4000, 6000, 8000]
        energies = [0.0] * len(steps)
        dirichlet = [595.6, 743.4, 833.9, 870.0, 884.8]

        fig = _assert_no_user_warning(
            lambda: plot_yang_mills_evolution(
                steps=steps, energies=energies, dirichlet_energies=dirichlet,
            )
        )
        assert fig is not None
        # Flat-transport collapse: exactly one axis (no twin YM axis).
        assert len(fig.axes) == 1, (
            f"Expected single-axis layout when YM is all zero, "
            f"got {len(fig.axes)} axes"
        )
        # Dirichlet is strictly positive, so the single axis is log-scaled.
        assert fig.axes[0].get_yscale() == "log"
        plt.close(fig)

    def test_nonflat_ym_two_axis_no_warning(self):
        """Positive YM + positive Dirichlet: two-axis layout, both log-scaled."""
        steps = [0, 1000, 2000, 3000]
        energies = [1e-3, 5e-3, 8e-3, 1.2e-2]
        dirichlet = [2.0, 5.0, 8.0, 10.0]

        fig = _assert_no_user_warning(
            lambda: plot_yang_mills_evolution(
                steps=steps, energies=energies, dirichlet_energies=dirichlet,
            )
        )
        assert fig is not None
        assert len(fig.axes) == 2, (
            f"Expected two axes (YM + Dirichlet twin), got {len(fig.axes)}"
        )
        for ax in fig.axes:
            assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_nonpositive_dirichlet_skips_log_scale(self):
        """Zero-or-negative Dirichlet must not be log-scaled."""
        steps = [0, 1000, 2000]
        energies = [0.0, 0.0, 0.0]
        dirichlet = [0.0, 0.0, 0.0]

        fig = _assert_no_user_warning(
            lambda: plot_yang_mills_evolution(
                steps=steps, energies=energies, dirichlet_energies=dirichlet,
            )
        )
        assert fig is not None
        # When Dirichlet hits zero, its axis stays linear.
        for ax in fig.axes:
            assert ax.get_yscale() == "linear"
        plt.close(fig)

    def test_no_dirichlet_series(self):
        """Plot must not crash when dirichlet_energies is None."""
        steps = [0, 1000, 2000]
        energies = [0.0, 0.0, 0.0]

        fig = _assert_no_user_warning(
            lambda: plot_yang_mills_evolution(
                steps=steps, energies=energies, dirichlet_energies=None,
            )
        )
        assert fig is not None
        plt.close(fig)


class TestYangMillsFromOmegaBilinear:
    """Verify YM energy vanishes on bilinear transport up to float noise."""

    def test_bilinear_transport_yields_zero_ym(self):
        """Omega_ij = omega_i · omega_j^{-1} ⇒ C_ijk = I ⇒ E_YM ≈ 0."""
        torch.manual_seed(0)
        B, N, K = 1, 16, 6

        # Build random gauge frames from random phi; any invertible omega works.
        # Use moderate scale so matrices stay well-conditioned.
        n_gen = K * K
        generators = torch.randn(n_gen, K, K) * 0.3
        phi = torch.randn(B, N, n_gen) * 0.05

        phi_matrix = _build_phi_matrix(phi, generators)  # (B, N, K, K)
        flat = phi_matrix.reshape(B * N, K, K)
        exp_phi_flat, _ = stable_matrix_exp_pair(flat, only_forward=True)
        omega_per_tok = exp_phi_flat.reshape(B, N, K, K)

        energy, diag = compute_yang_mills_energy_from_omega(
            omega_per_tok, sample_size=200, seed=0,
        )
        assert energy.shape == (B,)
        # Floating-point noise on K=6 logm should be well under 1e-4.
        assert float(energy.max().item()) < 1e-4, (
            f"E_YM for bilinear transport should be ~0, got {energy.max().item():.3e}"
        )
        assert diag["mean_F_norm"] < 1e-2
        assert diag["max_F_norm"] < 1e-2

    def test_too_few_tokens_returns_zero(self):
        """N < 3: no triples possible; must return zero without error."""
        omega = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 2, 4, 4).contiguous()
        energy, diag = compute_yang_mills_energy_from_omega(omega, sample_size=10)
        assert energy.shape == (1,)
        assert float(energy.item()) == 0.0
        assert diag["mean_F_norm"] == 0.0
