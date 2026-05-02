r"""
Single- and Multi-Run Trajectory Plot
=====================================

Click-to-run. Edit the CONFIG dict below and press Run.

Renders training and validation PPL vs ``tokens_seen`` on log-log axes for
one or more runs, overlaying multiple runs on the same axes for direct
sample-efficiency comparison (e.g. VFE vs standard transformer baseline at
matched K, depth, batch, iso-token).

Each entry in CONFIG['runs'] is a dict with:

    {
        'run_dir': Path(...),       # one run's checkpoint dir (must contain
                                    # metrics.csv and experiment_config.json)
        'label':   'VFE K=90',      # legend label
        'color':   None | str,      # None -> next from PUB_CYCLE
        'linestyle': '-',           # any matplotlib linestyle
    }

Output: a single 2-panel PNG/PDF (train PPL on the left, val PPL on the
right) saved to ``CONFIG['output_dir'] / CONFIG['output_name'].png`` and
``.pdf``. Reuses ``set_pub_style`` and the trajectory loader from
``scaling_ingest`` so styling matches the existing scaling figures.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from transformer.analysis.scaling_ingest import RunRecord, load_trajectory, _safe_load_json
from transformer.visualization.pub_style import PUB_COLORS, PUB_CYCLE, set_pub_style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG — edit and press Run
# ============================================================================

CONFIG = {
    # One or more runs to overlay on the same axes. Add more dicts to compare.
    'runs': [
        {
            'run_dir': Path(
                r'C:\Users\chris and christine\Desktop\V13_Gauge_Transformer'
                r'\transformer\checkpoints_publication\ffn_VFE_dynamic'
            ),
            'label': 'VFE wiki-ja',
            'color': None,        # None => next color from PUB_CYCLE
            'linestyle': '-',
        },
        # Example second-run entry to compare a baseline:
        # {
        #     'run_dir': Path(r'.../baseline_K=40_wiki-ja_seed23'),
        #     'label':   'Std QKV K=40 wiki-ja',
        #     'color':   None,
        #     'linestyle': '--',
        # },
    ],
    'output_dir': Path(
        r'C:\Users\chris and christine\Desktop\V13_Gauge_Transformer'
        r'\publication_outputs\trajectory'
    ),
    'output_name': 'trajectory',
    'smoothing_window': 25,        # rolling mean window on the trajectory; 0 disables
    'x_log': True,                 # tokens_seen axis log-scale
    'y_log': True,                 # PPL axis log-scale
    'title': None,                 # auto-generated when None
}


def _record_from_dir(run_dir: Path, label_seed: int = 0) -> RunRecord:
    """Build a synthetic RunRecord from a single run directory.

    The trajectory loader needs RunRecord (config, result_em, metrics_path) to
    compute ``tokens_seen``. We don't need a sweep-axis value here, so we set
    it to a sentinel float; downstream plotting ignores it.
    """
    cfg = _safe_load_json(run_dir / 'experiment_config.json') or {}
    res = _safe_load_json(run_dir / 'result_em.json') or {}
    return RunRecord(
        sweep_axis_value=float('nan'),
        seed=int(label_seed),
        run_dir=run_dir,
        config=cfg,
        result_em=res,
        metrics_path=run_dir / 'metrics.csv',
    )


def _smooth(s: pd.Series, window: int) -> pd.Series:
    if window and window > 1:
        return s.rolling(window=window, min_periods=1).mean()
    return s


def _plot_one(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    label: str,
    color: str,
    linestyle: str,
    smoothing_window: int,
) -> None:
    sub = df[df[y_col].notna()][['tokens_seen', y_col]].sort_values('tokens_seen')
    if sub.empty:
        return
    y = _smooth(sub[y_col], smoothing_window)
    ax.plot(sub['tokens_seen'], y, color=color, linewidth=1.5,
            linestyle=linestyle, label=label)


def render_trajectory_figure(cfg: dict) -> Path:
    set_pub_style()
    runs: List[Dict] = cfg['runs']
    if not runs:
        raise ValueError("CONFIG['runs'] must contain at least one entry")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    color_idx = 0
    for entry in runs:
        run_dir = Path(entry['run_dir'])
        if not run_dir.is_dir():
            logger.warning("Run dir does not exist, skipping: %s", run_dir)
            continue
        record = _record_from_dir(run_dir)
        df = load_trajectory(record)
        if df.empty:
            logger.warning("No trajectory rows for %s, skipping", run_dir)
            continue

        color = entry.get('color') or PUB_CYCLE[color_idx % len(PUB_CYCLE)]
        color_idx += 1
        linestyle = entry.get('linestyle', '-')
        label = entry.get('label') or run_dir.name

        _plot_one(axes[0], df, 'train_ppl', label, color, linestyle,
                  cfg.get('smoothing_window', 25))
        _plot_one(axes[1], df, 'val_ppl', label, color, linestyle,
                  cfg.get('smoothing_window', 25))

        terminal = df.dropna(subset=['val_ppl']).tail(1)
        if not terminal.empty:
            tk = int(terminal['tokens_seen'].iloc[0])
            ppl = float(terminal['val_ppl'].iloc[0])
            logger.info(
                "  %s: terminal val_ppl=%.2f at tokens_seen=%s",
                label, ppl, f'{tk:,}',
            )

    for ax, title in zip(axes, ('Training PPL', 'Validation PPL')):
        ax.set_xlabel('Tokens seen')
        ax.set_ylabel(title)
        if cfg.get('x_log', True):
            ax.set_xscale('log')
        if cfg.get('y_log', True):
            ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(labels), 4), fontsize=8, frameon=False,
        )

    title = cfg.get('title') or 'Training trajectory: PPL vs tokens seen'
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    name = cfg.get('output_name', 'trajectory')
    png_path = output_dir / f'{name}.png'
    pdf_path = output_dir / f'{name}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote: %s', png_path)
    logger.info('Wrote: %s', pdf_path)
    return png_path


def main(cfg: dict) -> None:
    render_trajectory_figure(cfg)
    logger.info('Done.')


if __name__ == '__main__':
    main(CONFIG)
