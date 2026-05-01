"""
Scaling-Sweep Data Ingestion
============================

Walks a multi-seed scaling sweep directory and produces tidy long-format
DataFrames suitable for downstream statistics and plotting. Parameterized by
sweep axis name (``K``, ``irrep_dim``, ...) so the same machinery serves the
embed-dim sweep at ``checkpoints_publication/0_scaling_data/`` and the
upcoming K=48 / variable-irrep-dim sweep without modification.

Layout assumed:

    <sweep_root>/<seed_dir>/<run_dir>/
        result_em.json
        metrics.csv
        experiment_config.json

Where ``<seed_dir>`` matches ``seed_regex`` (default captures the integer in
``seed=NN``) and ``<run_dir>`` matches ``sweep_axis_regex`` (default captures
``K=NN``). When the swept variable is not in the directory name (e.g. the
irrep-dim sweep where all dirs share ``K=48``), pass
``sweep_axis_from_config`` to derive the value from the loaded
``experiment_config.json`` instead.

The trajectory loader computes per-row ``tokens_seen = step * batch_size *
seq_len`` from each run's own config, so sweeps mixing batch sizes (the K
sweep uses bs=32 for K<=80 and bs=16 for K>=90) collapse cleanly onto a
common x-axis.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from transformer.visualization.training_plots import load_metrics_csv

logger = logging.getLogger(__name__)

_DEFAULT_AXIS_REGEX = r'(?:^|_)K=(\d+)(?:$|_)'
_DEFAULT_SEED_REGEX = r'seed=(\d+)'

# Trajectory columns we care about for downstream plots. Anything not in this
# allowlist is silently dropped (we have no use for the other 90+ metrics
# columns). One warning per never-seen column is emitted via
# _warn_unknown_columns to surface schema drift early.
_TRAJECTORY_COLUMNS_ALLOWLIST = (
    'steps',
    'train_loss_ce', 'train_loss_ce_raw', 'train_loss_total',
    'val_loss', 'val_ce', 'val_ppl', 'val_bpc',
    'train_ppl', 'train_bpc',
    'attention_entropy', 'attention_concentration',
    'gauge_field_energy', 'gauge_yang_mills_energy',
    'gauge_orbit_dim', 'gauge_abelian_fraction', 'gauge_mean_F_norm',
    'gauge_det_omega_mean', 'gauge_spectrum_mean',
    'holonomy_mean_norm', 'holonomy_max_norm',
    'phi_effective_rank', 'phi_rank_ratio',
)


@dataclass(frozen=True)
class RunRecord:
    """A single run's location and parsed metadata."""

    sweep_axis_value: float
    seed: int
    run_dir: Path
    config: Dict[str, Any]
    result_em: Dict[str, Any]
    metrics_path: Path


def _parse_axis_from_name(
    name: str,
    regex: str,
) -> Optional[float]:
    """Return the numeric axis value parsed from a directory name, or None."""
    match = re.search(regex, name)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except (IndexError, ValueError):
        return None


def _parse_seed_from_name(name: str, regex: str = _DEFAULT_SEED_REGEX) -> Optional[int]:
    match = re.search(regex, name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (IndexError, ValueError):
        return None


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def discover_runs(
    sweep_root: Path,
    sweep_axis_regex: str = _DEFAULT_AXIS_REGEX,
    sweep_axis_from_config: Optional[Callable[[Dict[str, Any]], float]] = None,
    seed_regex: str = _DEFAULT_SEED_REGEX,
) -> List[RunRecord]:
    """Walk ``sweep_root`` and return one RunRecord per discovered run.

    Two-stage axis parse: regex on the run dir name first, then fall back
    to ``sweep_axis_from_config`` against the loaded ``experiment_config.json``.
    Runs missing both ``result_em.json`` and ``metrics.csv`` are skipped with
    a warning. Runs missing one or the other are kept (downstream code handles
    the missing piece).
    """
    sweep_root = Path(sweep_root)
    if not sweep_root.is_dir():
        raise FileNotFoundError(f"Sweep root does not exist: {sweep_root}")

    records: List[RunRecord] = []
    seed_dirs = sorted(p for p in sweep_root.iterdir() if p.is_dir())
    for seed_dir in seed_dirs:
        seed = _parse_seed_from_name(seed_dir.name, seed_regex)
        if seed is None:
            logger.debug("Skipping non-seed dir %s", seed_dir.name)
            continue

        run_dirs = sorted(p for p in seed_dir.iterdir() if p.is_dir())
        for run_dir in run_dirs:
            config = _safe_load_json(run_dir / 'experiment_config.json') or {}
            result_em = _safe_load_json(run_dir / 'result_em.json') or {}
            metrics_path = run_dir / 'metrics.csv'

            if not config and not result_em and not metrics_path.exists():
                logger.warning("Run dir has no parseable artifacts: %s", run_dir)
                continue

            axis_value: Optional[float] = _parse_axis_from_name(
                run_dir.name, sweep_axis_regex
            )
            if axis_value is None and sweep_axis_from_config is not None and config:
                try:
                    axis_value = float(sweep_axis_from_config(config))
                except Exception as exc:
                    logger.warning(
                        "sweep_axis_from_config failed on %s: %s", run_dir, exc,
                    )

            if axis_value is None:
                logger.warning(
                    "Could not determine sweep-axis value for %s (regex=%r); skipping",
                    run_dir, sweep_axis_regex,
                )
                continue

            records.append(RunRecord(
                sweep_axis_value=axis_value,
                seed=seed,
                run_dir=run_dir,
                config=config,
                result_em=result_em,
                metrics_path=metrics_path,
            ))

    logger.info(
        "Discovered %d runs across %d seed dirs under %s",
        len(records), len(seed_dirs), sweep_root,
    )
    return records


# Track unknown trajectory columns so each is reported only once.
_UNKNOWN_COLUMNS_REPORTED: set = set()


def _warn_unknown_columns(seen: set) -> None:
    novel = (seen - set(_TRAJECTORY_COLUMNS_ALLOWLIST)) - _UNKNOWN_COLUMNS_REPORTED
    if novel:
        for col in sorted(novel):
            logger.debug("Trajectory column not in allowlist (dropping): %s", col)
        _UNKNOWN_COLUMNS_REPORTED.update(novel)


def _compute_tokens_seen(
    steps: List[int],
    batch_size: int,
    seq_len: int,
) -> List[int]:
    return [int(s) * int(batch_size) * int(seq_len) for s in steps]


def load_trajectory(record: RunRecord) -> pd.DataFrame:
    """Load metrics.csv as a DataFrame with computed tokens_seen.

    Returns an empty DataFrame (with the canonical column set) when
    metrics.csv is missing — keeps the downstream concat shape stable.
    """
    if not record.metrics_path.exists():
        logger.warning("metrics.csv missing for %s", record.run_dir)
        return pd.DataFrame(columns=['sweep_axis', 'seed', 'step', 'tokens_seen'])

    metrics = load_metrics_csv(record.metrics_path)
    if 'steps' not in metrics or not metrics['steps']:
        logger.warning("metrics.csv has no steps column for %s", record.run_dir)
        return pd.DataFrame(columns=['sweep_axis', 'seed', 'step', 'tokens_seen'])

    _warn_unknown_columns(set(metrics.keys()))

    bs = record.config.get('batch_size') or record.result_em.get('batch_size') or 32
    seq_len = record.config.get('max_seq_len') or record.result_em.get('seq_len') or 128

    n = len(metrics['steps'])
    data: Dict[str, Any] = {
        'sweep_axis': [record.sweep_axis_value] * n,
        'seed': [record.seed] * n,
        'step': metrics['steps'],
        'tokens_seen': _compute_tokens_seen(metrics['steps'], bs, seq_len),
    }
    for col in _TRAJECTORY_COLUMNS_ALLOWLIST:
        if col == 'steps':
            continue
        if col in metrics and len(metrics[col]) == n:
            data[col] = metrics[col]
    return pd.DataFrame(data)


def _build_terminal_row(record: RunRecord) -> Optional[Dict[str, Any]]:
    """Pull final-test/val/params/FLOPs metrics for one run."""
    if not record.result_em:
        logger.warning("result_em.json missing for %s", record.run_dir)
        return None
    r = record.result_em
    test_ppl = r.get('test_ppl')
    val_ppl = r.get('final_ppl')
    return {
        'sweep_axis': record.sweep_axis_value,
        'seed': record.seed,
        'test_ppl': test_ppl,
        'test_loss': r.get('test_loss'),
        'val_ppl': val_ppl,
        'val_loss': r.get('final_loss'),
        'total_params': r.get('total_params'),
        'total_steps': r.get('total_steps'),
        'tokens_seen': r.get('tokens_seen'),
        'flops_per_step': r.get('flops_per_step'),
        'total_flops': r.get('total_training_flops'),
        'batch_size': r.get('batch_size'),
        'seq_len': r.get('seq_len'),
        'run_dir': str(record.run_dir),
    }


def aggregate_sweep(
    sweep_root: Path,
    sweep_axis_name: str = 'K',
    sweep_axis_regex: str = _DEFAULT_AXIS_REGEX,
    sweep_axis_from_config: Optional[Callable[[Dict[str, Any]], float]] = None,
    seed_regex: str = _DEFAULT_SEED_REGEX,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Walk the sweep root and return ``(terminal_df, trajectory_df)``.

    ``terminal_df`` has one row per run with final-test, val, parameter, and
    FLOPs columns plus the swept axis renamed to ``sweep_axis_name`` for
    readability. ``trajectory_df`` is long-format with one row per
    (run, step) and a computed ``tokens_seen`` column.
    """
    records = discover_runs(
        sweep_root,
        sweep_axis_regex=sweep_axis_regex,
        sweep_axis_from_config=sweep_axis_from_config,
        seed_regex=seed_regex,
    )
    if not records:
        raise RuntimeError(
            f"No runs discovered under {sweep_root}. Check sweep_axis_regex={sweep_axis_regex!r}."
        )

    terminal_rows = [row for row in (_build_terminal_row(r) for r in records) if row]
    terminal_df = pd.DataFrame(terminal_rows).rename(columns={'sweep_axis': sweep_axis_name})

    traj_frames = [load_trajectory(r) for r in records]
    traj_frames = [f for f in traj_frames if not f.empty]
    if traj_frames:
        trajectory_df = pd.concat(traj_frames, ignore_index=True).rename(
            columns={'sweep_axis': sweep_axis_name},
        )
    else:
        trajectory_df = pd.DataFrame(
            columns=[sweep_axis_name, 'seed', 'step', 'tokens_seen'],
        )

    logger.info(
        "Aggregated sweep: %d runs (terminal), %d trajectory rows. "
        "%s values: %s",
        len(terminal_df),
        len(trajectory_df),
        sweep_axis_name,
        sorted(terminal_df[sweep_axis_name].unique().tolist())
            if not terminal_df.empty else [],
    )
    return terminal_df, trajectory_df
