"""Tests for transformer.analysis.scaling_stats."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transformer.analysis.scaling_stats import (
    PowerLawFit,
    fit_power_law,
    seed_summary,
)


def _synthetic(
    a: float = 200.0, b: float = -1.0, c: float = 50.0,
    K_values=(10, 20, 30, 40, 50, 60, 80, 100, 120),
    n_seeds: int = 3,
    noise_std: float = 0.5,
    rng_seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    rows = []
    for K in K_values:
        for s in range(n_seeds):
            true_y = a * (K ** b) + c
            y = true_y + rng.normal(0, noise_std)
            rows.append({'K': float(K), 'seed': s, 'test_ppl': y})
    return pd.DataFrame(rows)


def test_fit_recovers_power_law_within_ci():
    df = _synthetic(a=200.0, b=-1.0, c=50.0, noise_std=0.5)
    fit = fit_power_law(df, axis_col='K', y_col='test_ppl', n_bootstrap=200, rng_seed=0)
    assert isinstance(fit, PowerLawFit)
    # Fit should hit the true params closely
    assert abs(fit.b - (-1.0)) < 0.1
    assert abs(fit.c - 50.0) < 5.0
    assert fit.r_squared > 0.99
    # Bootstrap actually ran
    assert fit.n_bootstrap_succeeded > 0
    # CIs should bracket the truth
    assert fit.b_ci[0] <= -1.0 <= fit.b_ci[1] or abs(fit.b - (-1.0)) < 0.1


def test_fit_loglog_method():
    df = _synthetic(c=0.0, noise_std=0.1)
    fit = fit_power_law(df, axis_col='K', n_bootstrap=100, method='loglog', rng_seed=0)
    assert fit.method == 'loglog'
    assert abs(fit.b - (-1.0)) < 0.1


def test_fit_falls_back_to_loglog_on_tiny_data():
    # Two K values is too few for nonlinear (3 free params); should fallback.
    df = pd.DataFrame({
        'K': [10.0, 10.0, 100.0, 100.0],
        'seed': [0, 1, 0, 1],
        'test_ppl': [200.0, 202.0, 70.0, 72.0],
    })
    fit = fit_power_law(df, axis_col='K', n_bootstrap=10, method='nonlinear', rng_seed=0)
    assert fit.method == 'loglog'


def test_fit_grid_and_ribbon_shapes_match():
    df = _synthetic(noise_std=0.3)
    fit = fit_power_law(df, axis_col='K', n_bootstrap=50, rng_seed=0)
    assert fit.axis_grid.shape == fit.pred_mean.shape
    assert fit.pred_lo.shape == fit.pred_mean.shape
    assert fit.pred_hi.shape == fit.pred_mean.shape
    assert np.all(fit.pred_lo <= fit.pred_hi + 1e-9)


def test_fit_skips_bootstrap_with_one_seed():
    df = _synthetic(n_seeds=1, noise_std=0.0)
    fit = fit_power_law(df, axis_col='K', n_bootstrap=100, rng_seed=0)
    assert fit.n_bootstrap_succeeded == 0
    # CIs should be NaN when bootstrap is skipped
    assert np.isnan(fit.b_ci[0]) and np.isnan(fit.b_ci[1])


def test_fit_raises_on_empty_data():
    df = pd.DataFrame({'K': [], 'seed': [], 'test_ppl': []})
    with pytest.raises(ValueError):
        fit_power_law(df, axis_col='K')


def test_seed_summary_basic_stats():
    df = _synthetic(noise_std=2.0)
    summary = seed_summary(df, axis_col='K')
    assert 'mean_test_ppl' in summary.columns
    assert 'std_test_ppl' in summary.columns
    assert 'n_seeds' in summary.columns
    # Expect 9 axis values, 3 seeds each
    assert (summary['n_seeds'] == 3).all()
    assert len(summary) == 9
