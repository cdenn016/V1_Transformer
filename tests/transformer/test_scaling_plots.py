"""Tests for transformer.analysis.scaling_plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from transformer.analysis.scaling_plots import (
    fig_compute_frontier,
    fig_gauge_vs_K,
    fig_scaling_main,
    fig_trajectory_by_K,
    render_all,
)
from transformer.analysis.scaling_stats import PowerLawFit


def _stub_terminal() -> pd.DataFrame:
    rows = []
    for K in (10, 40, 80, 120):
        for seed in (6, 23, 111):
            rows.append({
                'K': float(K),
                'seed': seed,
                'test_ppl': 200.0 / (K ** 0.5) + 50.0 + 0.1 * seed,
                'val_ppl': 220.0 / (K ** 0.5) + 50.0,
                'total_params': K * 1_000_000,
                'flops_per_step': K * 1e9,
                'total_flops': K * 1e9 * 30000,
            })
    return pd.DataFrame(rows)


def _stub_trajectory() -> pd.DataFrame:
    rows = []
    for K in (10, 40, 80, 120):
        for seed in (6, 23, 111):
            for step in range(0, 30000, 1000):
                rows.append({
                    'K': float(K),
                    'seed': seed,
                    'step': step,
                    'tokens_seen': step * 32 * 128,
                    'train_ppl': 1000 / (1 + step / 1000),
                    'val_ppl': 1100 / (1 + step / 1000),
                    'gauge_field_energy': 100.0 + K * 10,
                    'attention_entropy': 2.5 + 0.01 * K,
                    'gauge_orbit_dim': 10.0 * K,
                })
    return pd.DataFrame(rows)


def _stub_fit() -> PowerLawFit:
    grid = np.logspace(1, np.log10(120), 50)
    pred = 200.0 / (grid ** 0.5) + 50.0
    return PowerLawFit(
        a=200.0, b=-0.5, c=50.0,
        a_ci=(180.0, 220.0), b_ci=(-0.55, -0.45), c_ci=(48.0, 52.0),
        r_squared=0.99,
        n_bootstrap=100, n_bootstrap_succeeded=100,
        seeds_used=[6, 23, 111],
        axis_grid=grid, pred_mean=pred,
        pred_lo=pred * 0.95, pred_hi=pred * 1.05,
        method='nonlinear',
    )


def test_fig_scaling_main_writes_files(tmp_path: Path):
    out = fig_scaling_main(_stub_terminal(), _stub_fit(), tmp_path)
    assert out.exists() and out.stat().st_size > 0
    assert (tmp_path / 'fig_scaling_main.pdf').exists()


def test_fig_compute_frontier_writes_files(tmp_path: Path):
    out = fig_compute_frontier(_stub_terminal(), tmp_path)
    assert out.exists() and out.stat().st_size > 0
    assert (tmp_path / 'fig_compute_frontier.pdf').exists()


def test_fig_trajectory_by_K_writes_files(tmp_path: Path):
    out = fig_trajectory_by_K(_stub_trajectory(), tmp_path, smoothing_window=3)
    assert out.exists() and out.stat().st_size > 0
    assert (tmp_path / 'fig_trajectory_by_K.pdf').exists()


def test_fig_gauge_vs_K_writes_files(tmp_path: Path):
    out = fig_gauge_vs_K(_stub_terminal(), _stub_trajectory(), tmp_path)
    assert out.exists() and out.stat().st_size > 0
    assert (tmp_path / 'fig_gauge_vs_K.pdf').exists()


def test_fig_trajectory_handles_empty(tmp_path: Path):
    empty = pd.DataFrame(columns=['K', 'seed', 'step', 'tokens_seen'])
    out = fig_trajectory_by_K(empty, tmp_path)
    assert out.exists() and out.stat().st_size > 0  # placeholder still rendered


def test_fig_compute_frontier_missing_flops_no_crash(tmp_path: Path):
    df = _stub_terminal().drop(columns=['total_flops'])
    out = fig_compute_frontier(df, tmp_path)
    assert out.exists() and out.stat().st_size > 0


def test_render_all_returns_dict_of_paths(tmp_path: Path):
    paths = render_all(_stub_terminal(), _stub_trajectory(), _stub_fit(), tmp_path)
    expected = {'fig_scaling_main', 'fig_compute_frontier',
                'fig_trajectory_by_K', 'fig_gauge_vs_K'}
    assert set(paths.keys()) == expected
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
