"""NaN-robustness tests for holonomy diagnostics.

Closes the 2026-04-19 audit finding that `compute_holonomy_snapshot`
propagated NaN through `.mean()/.std()/.median()/.max()/wilson_trace`
while `frac_gt_*` and `spectral_gap` silently absorbed it.  These tests
lock in the new contract:

- Reductions run over the non-NaN subset only.
- `nan_fraction` exposes the failure rate.
- All-NaN input returns NaN stats + `nan_fraction == 1.0` without raising.
- The dedicated `holonomy_metrics.csv` is written by `PublicationMetrics`
  on every invocation and is independent of the main training-metrics CSV.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from transformer.analysis.holonomy_metrics import compute_holonomy_snapshot


def _make_exp_delta(N: int, K: int, nan_fraction_target: float) -> torch.Tensor:
    """Construct a (1, N, N, K, K) exp_delta tensor with the requested NaN
    density on a per-edge basis so that holonomy sampling hits both clean
    and poisoned triples.
    """
    torch.manual_seed(0)
    exp_delta = torch.eye(K).reshape(1, 1, 1, K, K).expand(1, N, N, K, K).clone()
    # Perturb toward non-identity so mean_norm is nonzero on clean triples.
    exp_delta = exp_delta + 0.01 * torch.randn_like(exp_delta)
    # Poison a deterministic subset of edges.
    n_edges = N * N
    n_poisoned = int(round(nan_fraction_target * n_edges))
    idx = torch.randperm(n_edges)[:n_poisoned]
    flat = exp_delta.reshape(1, n_edges, K, K)
    flat[0, idx] = float('nan')
    return flat.reshape(1, N, N, K, K)


def test_mixed_nan_reductions_use_valid_subset():
    """Valid stats computed over non-NaN triples only; nan_fraction reports
    the contamination rate."""
    exp_delta = _make_exp_delta(N=8, K=4, nan_fraction_target=0.25)
    snap = compute_holonomy_snapshot(exp_delta, sample_size=100, seed=123)

    # nan_fraction must be finite and within [0, 1]
    assert 0.0 <= snap.nan_fraction <= 1.0
    # Reductions on the non-NaN subset must be finite
    if snap.nan_fraction < 1.0:
        assert math.isfinite(snap.mean_norm), f"mean_norm={snap.mean_norm}"
        assert math.isfinite(snap.std_norm), f"std_norm={snap.std_norm}"
        assert math.isfinite(snap.median_norm), f"median_norm={snap.median_norm}"
        assert math.isfinite(snap.max_norm), f"max_norm={snap.max_norm}"
        assert math.isfinite(snap.mean_wilson_trace), f"wilson_trace={snap.mean_wilson_trace}"
    # frac_gt_* must be in [0, 1]
    assert 0.0 <= snap.frac_gt_001 <= 1.0
    assert 0.0 <= snap.frac_gt_01 <= 1.0
    assert snap.sample_size == 100


def test_clean_input_gives_zero_nan_fraction():
    """A fully valid exp_delta must produce nan_fraction == 0 and all-finite stats."""
    exp_delta = _make_exp_delta(N=8, K=4, nan_fraction_target=0.0)
    snap = compute_holonomy_snapshot(exp_delta, sample_size=50, seed=7)

    assert snap.nan_fraction == 0.0
    assert math.isfinite(snap.mean_norm)
    assert math.isfinite(snap.std_norm)
    assert math.isfinite(snap.median_norm)
    assert math.isfinite(snap.max_norm)
    assert math.isfinite(snap.mean_wilson_trace)
    assert math.isfinite(snap.mean_spectral_gap)


def test_all_nan_input_returns_nan_stats_without_raising():
    """All-NaN input: reductions return NaN, nan_fraction==1, no exception."""
    B, N, K = 1, 6, 3
    exp_delta = torch.full((B, N, N, K, K), float('nan'))
    snap = compute_holonomy_snapshot(exp_delta, sample_size=20, seed=1)

    assert snap.nan_fraction == pytest.approx(1.0)
    assert math.isnan(snap.mean_norm)
    assert math.isnan(snap.std_norm)
    assert math.isnan(snap.median_norm)
    assert math.isnan(snap.max_norm)
    # spectral_gap and wilson_trace degrade to NaN when C_clean is empty
    # (no SVD possible). They must not raise.
    assert math.isnan(snap.mean_spectral_gap) or snap.mean_spectral_gap == 0.0
    assert math.isnan(snap.mean_wilson_trace)


def test_to_log_dict_exposes_nan_fraction_and_variant():
    """`nan_fraction` and the variant tag must be emitted to the logging
    dict so downstream CSVs/TensorBoard can track the failure rate and
    distinguish raw vs cocycle-scaled diagnostics."""
    exp_delta = _make_exp_delta(N=6, K=3, nan_fraction_target=0.1)
    snap = compute_holonomy_snapshot(exp_delta, sample_size=30, seed=42)

    d = snap.to_log_dict(prefix='holonomy')
    assert any(k.endswith('/nan_fraction') for k in d)
    # Default variant is 'raw'; key format: holonomy/L{layer}_H{head}_{variant}/nan_fraction
    key = f'holonomy/L{snap.layer}_H{snap.head}_{snap.variant}/nan_fraction'
    assert d[key] == snap.nan_fraction
    # delta spectral-norm stats must be present too
    assert any(k.endswith('/delta_max_spec') for k in d)


def test_scaled_variant_changes_log_keys():
    """Snapshots with variant='scaled' produce distinct log keys from
    'raw' so both can coexist in one CSV row."""
    exp_delta = _make_exp_delta(N=6, K=3, nan_fraction_target=0.0)
    snap_raw = compute_holonomy_snapshot(
        exp_delta, sample_size=30, seed=1, variant='raw',
        delta_stats={'delta_max_spec': 1.0, 'delta_p95_spec': 0.5, 'delta_mean_spec': 0.3},
    )
    snap_scaled = compute_holonomy_snapshot(
        exp_delta, sample_size=30, seed=1, variant='scaled',
        delta_stats={'delta_max_spec': 0.5, 'delta_p95_spec': 0.25, 'delta_mean_spec': 0.15},
    )
    d_raw = snap_raw.to_log_dict(prefix='holonomy')
    d_scaled = snap_scaled.to_log_dict(prefix='holonomy')
    assert set(d_raw.keys()).isdisjoint(d_scaled.keys()), "raw and scaled keys must not collide"
    assert snap_raw.delta_max_spec == 1.0
    assert snap_scaled.delta_max_spec == 0.5


def test_holonomy_csv_written_independently(tmp_path: Path):
    """`PublicationMetrics.compute_holonomy_diagnostics` must append rows
    to its own CSV regardless of whether the main training CSV has an
    entry at the same step."""
    from transformer.analysis.publication_metrics import PublicationMetrics, HolonomyProfile

    pm = PublicationMetrics(experiment_name='test_holonomy_csv', base_dir=tmp_path)
    assert pm.holonomy_csv_path.parent.exists()

    # Construct two synthetic profiles and push them through the writer
    # directly; we don't need a real model to exercise the CSV path.
    from transformer.analysis.holonomy_metrics import HolonomySnapshot
    snap1 = HolonomySnapshot(
        step=500, layer=0, head=0,
        mean_norm=0.5, std_norm=0.1, median_norm=0.5, max_norm=0.9,
        frac_gt_001=0.8, frac_gt_01=0.6, mean_spectral_gap=0.2,
        mean_wilson_trace=0.3, nan_fraction=0.0, sample_size=500,
        variant='raw', delta_max_spec=1.2, delta_p95_spec=0.8, delta_mean_spec=0.5,
    )
    snap2 = HolonomySnapshot(
        step=1000, layer=0, head=0,
        mean_norm=float('nan'), std_norm=float('nan'),
        median_norm=float('nan'), max_norm=float('nan'),
        frac_gt_001=0.5, frac_gt_01=0.2, mean_spectral_gap=0.0,
        mean_wilson_trace=float('nan'), nan_fraction=0.7, sample_size=500,
        variant='raw', delta_max_spec=200.0, delta_p95_spec=45.0, delta_mean_spec=12.0,
    )
    pm._append_holonomy_csv_row(HolonomyProfile(step=500, snapshots=[snap1], global_mean_norm=0.5, global_max_norm=0.9))
    pm._append_holonomy_csv_row(HolonomyProfile(step=1000, snapshots=[snap2], global_mean_norm=float('nan'), global_max_norm=float('nan')))

    # File exists with header + 2 data rows.
    assert pm.holonomy_csv_path.exists(), "holonomy_metrics.csv not created"
    content = pm.holonomy_csv_path.read_text()
    lines = content.strip().splitlines()
    assert len(lines) == 3, f"expected header + 2 rows, got {len(lines)}"
    header = lines[0].split(',')
    assert 'step' in header
    assert 'global_mean_norm' in header
    assert any('nan_fraction' in h for h in header), "nan_fraction column missing"
