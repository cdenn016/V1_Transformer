"""Tests for transformer.analysis.scaling_ingest."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from transformer.analysis.scaling_ingest import (
    _compute_tokens_seen,
    _parse_axis_from_name,
    _parse_seed_from_name,
    aggregate_sweep,
    discover_runs,
)


def _write_run(
    run_dir: Path,
    *,
    final_test_ppl: float = 50.0,
    total_params: int = 1_000_000,
    total_steps: int = 30000,
    batch_size: int = 32,
    seq_len: int = 128,
    metric_rows: int = 5,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'experiment_config.json').write_text(json.dumps({
        'batch_size': batch_size,
        'max_seq_len': seq_len,
    }))
    (run_dir / 'result_em.json').write_text(json.dumps({
        'test_ppl': final_test_ppl,
        'final_ppl': final_test_ppl * 1.1,
        'test_loss': 3.9,
        'final_loss': 4.0,
        'total_params': total_params,
        'total_steps': total_steps,
        'tokens_seen': total_steps * batch_size * seq_len,
        'flops_per_step': 2e10,
        'total_training_flops': 2e10 * total_steps,
        'batch_size': batch_size,
        'seq_len': seq_len,
    }))
    header = (
        'step,timestamp,train_loss_total,train_loss_ce,val_loss,val_ce,'
        'train_ppl,val_ppl\n'
    )
    rows = [
        f'{i * 200},0.0,4.0,4.0,4.5,4.5,{50 + i},{60 + i}'
        for i in range(metric_rows)
    ]
    (run_dir / 'metrics.csv').write_text(header + '\n'.join(rows) + '\n')


def test_parse_axis_from_name():
    assert _parse_axis_from_name('100_K=40_GL(10)_N=128', r'(?:^|_)K=(\d+)(?:$|_)') == 40.0
    assert _parse_axis_from_name('K=120_seed=23', r'(?:^|_)K=(\d+)(?:$|_)') == 120.0
    assert _parse_axis_from_name('foo', r'K=(\d+)') is None


def test_parse_axis_does_not_match_K_in_N():
    # Anchored regex must not match K in 'N=128'-like substrings.
    assert _parse_axis_from_name('foo_N=128', r'(?:^|_)K=(\d+)(?:$|_)') is None


def test_parse_seed_from_name():
    assert _parse_seed_from_name('1_seed=23') == 23
    assert _parse_seed_from_name('seed=111') == 111
    assert _parse_seed_from_name('foo') is None


def test_compute_tokens_seen():
    assert _compute_tokens_seen([0, 100, 200], 32, 128) == [0, 409600, 819200]


def test_discover_runs_basic(tmp_path: Path):
    seed_dir = tmp_path / '1_seed=23'
    _write_run(seed_dir / '100_K=40_GL(10)_N=128')
    _write_run(seed_dir / '200_K=80_GL(10)_N=128')
    records = discover_runs(tmp_path)
    assert len(records) == 2
    axis_values = sorted(r.sweep_axis_value for r in records)
    assert axis_values == [40.0, 80.0]
    assert all(r.seed == 23 for r in records)


def test_discover_runs_skips_unparseable(tmp_path: Path, caplog):
    seed_dir = tmp_path / '1_seed=23'
    _write_run(seed_dir / '100_K=40_GL(10)_N=128')
    # A run with no K= in the name and no callback fallback — should be skipped.
    _write_run(seed_dir / 'mystery_run')
    records = discover_runs(tmp_path)
    assert len(records) == 1
    assert records[0].sweep_axis_value == 40.0


def test_axis_from_config_callback(tmp_path: Path):
    seed_dir = tmp_path / '1_seed=6'
    run = seed_dir / 'mystery_run'
    _write_run(run)
    # Embed irrep_dim into config so the callback can pull it.
    cfg = json.loads((run / 'experiment_config.json').read_text())
    cfg['irrep_dim'] = 12
    (run / 'experiment_config.json').write_text(json.dumps(cfg))
    records = discover_runs(
        tmp_path,
        sweep_axis_regex=r'NEVER_MATCHES',
        sweep_axis_from_config=lambda c: c['irrep_dim'],
    )
    assert len(records) == 1
    assert records[0].sweep_axis_value == 12.0


def test_aggregate_sweep_returns_terminal_and_trajectory(tmp_path: Path):
    for seed in (6, 23, 111):
        sd = tmp_path / f'1_seed={seed}'
        _write_run(sd / '100_K=40_GL(10)_N=128')
        _write_run(sd / '200_K=80_GL(10)_N=128', metric_rows=8)

    terminal_df, traj_df = aggregate_sweep(tmp_path, sweep_axis_name='K')
    assert isinstance(terminal_df, pd.DataFrame)
    assert isinstance(traj_df, pd.DataFrame)
    assert set(terminal_df['K'].unique()) == {40.0, 80.0}
    assert set(terminal_df['seed'].unique()) == {6, 23, 111}
    # 6 runs, K=40 has 5 rows each, K=80 has 8 rows each
    assert len(traj_df) == 3 * 5 + 3 * 8
    assert {'K', 'seed', 'step', 'tokens_seen'} <= set(traj_df.columns)
    # Tokens seen must reflect bs*seq_len from each run's config (32 * 128 = 4096)
    assert traj_df.loc[traj_df['step'] == 200, 'tokens_seen'].iloc[0] == 200 * 32 * 128


def test_aggregate_handles_missing_run(tmp_path: Path):
    # Mimic the K=90 missing in seed=6 case from the real sweep.
    for seed in (6, 23, 111):
        sd = tmp_path / f'1_seed={seed}'
        _write_run(sd / '100_K=40_GL(10)_N=128')
        if seed != 6:
            _write_run(sd / '200_K=90_GL(10)_N=128')
    terminal_df, _ = aggregate_sweep(tmp_path, sweep_axis_name='K')
    counts = terminal_df.groupby('K').size().to_dict()
    assert counts == {40.0: 3, 90.0: 2}


def test_aggregate_raises_on_empty_root(tmp_path: Path):
    with pytest.raises(RuntimeError):
        aggregate_sweep(tmp_path, sweep_axis_name='K')
