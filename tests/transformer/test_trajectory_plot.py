"""Tests for scripts/trajectory_plot.py (single/multi-run trajectory figure)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent.parent / 'scripts'
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import trajectory_plot as tp  # noqa: E402


def _write_run(run_dir: Path, batch_size: int = 16, seq_len: int = 128, n_rows: int = 10):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'experiment_config.json').write_text(json.dumps({
        'batch_size': batch_size, 'max_seq_len': seq_len,
    }))
    (run_dir / 'result_em.json').write_text(json.dumps({
        'test_ppl': 50.0, 'final_ppl': 60.0,
        'total_params': 1_000_000, 'tokens_seen': n_rows * 200 * batch_size * seq_len,
        'flops_per_step': 2e10, 'total_training_flops': 2e10 * n_rows * 200,
        'batch_size': batch_size, 'seq_len': seq_len,
    }))
    header = (
        'step,timestamp,train_loss_total,train_loss_ce,val_loss,val_ce,'
        'train_ppl,val_ppl\n'
    )
    rows = []
    for i in range(n_rows):
        step = (i + 1) * 200
        rows.append(f'{step},0.0,4.0,4.0,4.5,4.5,{100 - i * 5},{120 - i * 5}')
    (run_dir / 'metrics.csv').write_text(header + '\n'.join(rows) + '\n')


def test_single_run_writes_files(tmp_path: Path):
    run_dir = tmp_path / 'run_a'
    _write_run(run_dir)
    cfg = {
        'runs': [{'run_dir': run_dir, 'label': 'A', 'color': None, 'linestyle': '-'}],
        'output_dir': tmp_path / 'out',
        'output_name': 'traj',
        'smoothing_window': 0,
        'x_log': True, 'y_log': True, 'title': None,
    }
    out = tp.render_trajectory_figure(cfg)
    assert out.exists() and out.stat().st_size > 0
    assert (cfg['output_dir'] / 'traj.pdf').exists()


def test_multi_run_overlay(tmp_path: Path):
    run_a = tmp_path / 'run_a'
    run_b = tmp_path / 'run_b'
    _write_run(run_a, batch_size=32)
    _write_run(run_b, batch_size=16)
    cfg = {
        'runs': [
            {'run_dir': run_a, 'label': 'A bs=32', 'color': None, 'linestyle': '-'},
            {'run_dir': run_b, 'label': 'B bs=16', 'color': None, 'linestyle': '--'},
        ],
        'output_dir': tmp_path / 'out',
        'output_name': 'overlay',
        'smoothing_window': 3,
        'x_log': True, 'y_log': True, 'title': 'overlay test',
    }
    out = tp.render_trajectory_figure(cfg)
    assert out.exists() and out.stat().st_size > 0


def test_missing_run_dir_skipped_not_crash(tmp_path: Path, caplog):
    cfg = {
        'runs': [
            {'run_dir': tmp_path / 'does_not_exist', 'label': 'X',
             'color': None, 'linestyle': '-'},
        ],
        'output_dir': tmp_path / 'out',
        'output_name': 'x',
        'smoothing_window': 0,
        'x_log': True, 'y_log': True, 'title': None,
    }
    # Should not raise; figure is empty placeholder.
    out = tp.render_trajectory_figure(cfg)
    assert out.exists()


def test_empty_runs_raises(tmp_path: Path):
    cfg = {
        'runs': [],
        'output_dir': tmp_path / 'out',
        'output_name': 'x',
        'smoothing_window': 0,
        'x_log': True, 'y_log': True, 'title': None,
    }
    with pytest.raises(ValueError):
        tp.render_trajectory_figure(cfg)


def test_record_from_dir_uses_config_for_bs(tmp_path: Path):
    run_dir = tmp_path / 'run'
    _write_run(run_dir, batch_size=16, seq_len=128, n_rows=5)
    record = tp._record_from_dir(run_dir)
    assert record.config.get('batch_size') == 16
    assert record.config.get('max_seq_len') == 128
