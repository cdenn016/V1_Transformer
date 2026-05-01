r"""
Publication-Quality Scaling Figures
===================================

Four figures for a scaling-sweep paper section:

* :func:`fig_scaling_main` — test/val PPL vs swept axis with multi-seed
  error bars and a power-law fit ribbon. Two panels: log-log and linear-y.
* :func:`fig_compute_frontier` — PPL vs total FLOPs and PPL vs total
  parameters (Pareto-style), per-K mean markers annotated.
* :func:`fig_trajectory_by_K` — training and validation PPL vs
  ``tokens_seen``, one curve per axis value, multi-seed mean and band.
* :func:`fig_gauge_vs_K` — terminal-step gauge / attention diagnostics
  vs swept axis (2x2: gauge field energy, holonomy, attention entropy,
  gauge orbit dim).

All figures call :func:`set_pub_style` at entry, save as PNG and PDF at
300 DPI, and avoid Claude-isms in any text drawn into the figure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transformer.analysis.scaling_stats import PowerLawFit, _power_law
from transformer.visualization.pub_style import (
    PUB_COLORS,
    PUB_CYCLE,
    set_pub_style,
)

logger = logging.getLogger(__name__)


def _save_dual(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    """Save fig as both PNG and PDF; return the PNG path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f'{name}.png'
    pdf = output_dir / f'{name}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    return png


def _color_for(value: float, all_values: np.ndarray) -> str:
    """Pick a stable color for one axis value from PUB_CYCLE."""
    sorted_vals = np.sort(np.unique(all_values))
    idx = int(np.searchsorted(sorted_vals, value)) % len(PUB_CYCLE)
    return PUB_CYCLE[idx]


def _agg_per_axis(
    df: pd.DataFrame, axis_col: str, y_col: str,
) -> pd.DataFrame:
    grp = df.groupby(axis_col)[y_col]
    out = pd.DataFrame({
        axis_col: list(grp.groups.keys()),
        'mean': grp.mean().values,
        'std': grp.std(ddof=1).values,
        'n': grp.count().values,
    })
    out['sem'] = out['std'] / np.sqrt(out['n']).replace(0, 1)
    return out.sort_values(axis_col).reset_index(drop=True)


def fig_scaling_main(
    terminal_df: pd.DataFrame,
    fit: PowerLawFit,
    output_dir: Path,
    axis_col: str = 'K',
    axis_label: str = r'$K$',
) -> Path:
    """Two-panel scaling figure: log-log on the left, linear-y on the right.

    Test PPL with seed error bars (closed markers) plus the fitted power
    law and its bootstrap ribbon. Val PPL overlaid with open markers.
    """
    set_pub_style()
    test_agg = _agg_per_axis(terminal_df, axis_col, 'test_ppl')
    val_agg = (_agg_per_axis(terminal_df, axis_col, 'val_ppl')
               if 'val_ppl' in terminal_df.columns else None)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, scale in zip(axes, ('loglog', 'linear')):
        ax.fill_between(
            fit.axis_grid, fit.pred_lo, fit.pred_hi,
            alpha=0.18, color=PUB_COLORS['blue'], linewidth=0,
            label='95% bootstrap CI',
        )
        ax.plot(
            fit.axis_grid, fit.pred_mean,
            color=PUB_COLORS['blue'], linewidth=1.5,
            label=fr'fit: $a\,K^{{b}}+c$, $b={fit.b:.2f}$',
        )
        ax.errorbar(
            test_agg[axis_col], test_agg['mean'], yerr=test_agg['std'],
            fmt='o', color=PUB_COLORS['black'], capsize=3, markersize=5,
            linewidth=1.2, label='test PPL (seed mean)',
        )
        if val_agg is not None and val_agg['mean'].notna().any():
            ax.errorbar(
                val_agg[axis_col], val_agg['mean'], yerr=val_agg['std'],
                fmt='s', color=PUB_COLORS['orange'], capsize=3, markersize=4,
                linewidth=1.0, alpha=0.7, label='val PPL (seed mean)',
                markerfacecolor='none',
            )
        if scale == 'loglog':
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel(axis_label)
        ax.set_ylabel('Perplexity')
        ax.grid(True, which='both', alpha=0.25)

    axes[0].set_title('log–log')
    axes[1].set_title('linear–y')
    axes[1].legend(loc='upper right', frameon=True, fontsize=7)
    fig.suptitle(
        f'WikiText-103 scaling: PPL vs {axis_label}    '
        f'(R$^2$={fit.r_squared:.3f}, n_seeds={len(fit.seeds_used)})',
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    return _save_dual(fig, output_dir, 'fig_scaling_main')


def fig_compute_frontier(
    terminal_df: pd.DataFrame,
    output_dir: Path,
    axis_col: str = 'K',
    axis_label: str = r'$K$',
) -> Path:
    """PPL vs total FLOPs and PPL vs total parameters (Pareto-style)."""
    set_pub_style()
    df = terminal_df.dropna(subset=['test_ppl']).copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, x_col, x_label, x_log in [
        (axes[0], 'total_flops', 'Total training FLOPs', True),
        (axes[1], 'total_params', 'Total parameters', True),
    ]:
        if x_col not in df.columns or df[x_col].isna().all():
            ax.text(0.5, 0.5, f'{x_col} unavailable',
                    transform=ax.transAxes, ha='center', va='center',
                    color=PUB_COLORS['gray'])
            continue
        sub = df.dropna(subset=[x_col])
        ax.scatter(
            sub[x_col], sub['test_ppl'],
            color=PUB_COLORS['gray'], s=22, alpha=0.45,
            label='per-seed', zorder=2,
        )
        agg = sub.groupby(axis_col).agg(
            x=(x_col, 'mean'), y=('test_ppl', 'mean'), s=('test_ppl', 'std'),
        ).reset_index().sort_values('x')
        ax.errorbar(
            agg['x'], agg['y'], yerr=agg['s'],
            fmt='o-', color=PUB_COLORS['blue'], capsize=3, markersize=6,
            linewidth=1.2, label=f'mean over seeds, by {axis_label}', zorder=3,
        )
        for _, row in agg.iterrows():
            ax.annotate(
                f"{int(row[axis_col])}",
                xy=(row['x'], row['y']),
                xytext=(4, 4), textcoords='offset points',
                fontsize=7, color=PUB_COLORS['black'],
            )
        if x_log:
            ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Test perplexity')
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(loc='upper right', frameon=True, fontsize=7)

    fig.suptitle(
        f'Compute frontier: test PPL vs FLOPs and parameters    (axis: {axis_label})',
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    return _save_dual(fig, output_dir, 'fig_compute_frontier')


def fig_trajectory_by_K(
    trajectory_df: pd.DataFrame,
    output_dir: Path,
    axis_col: str = 'K',
    axis_label: str = r'$K$',
    smoothing_window: Optional[int] = 50,
) -> Path:
    """Training and val PPL vs ``tokens_seen``, one curve per axis value."""
    set_pub_style()
    if trajectory_df.empty:
        logger.warning("Empty trajectory_df; skipping fig_trajectory_by_K")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.5, 0.5, 'no trajectory data', transform=ax.transAxes,
                ha='center', va='center', color=PUB_COLORS['gray'])
        return _save_dual(fig, output_dir, 'fig_trajectory_by_K')

    axis_values = np.sort(trajectory_df[axis_col].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)

    for ax, y_col, title in [
        (axes[0], 'train_ppl', 'Training PPL'),
        (axes[1], 'val_ppl', 'Validation PPL'),
    ]:
        if y_col not in trajectory_df.columns or trajectory_df[y_col].isna().all():
            ax.text(0.5, 0.5, f'{y_col} unavailable',
                    transform=ax.transAxes, ha='center', va='center',
                    color=PUB_COLORS['gray'])
            continue
        for v in axis_values:
            color = _color_for(v, axis_values)
            sub = trajectory_df[
                (trajectory_df[axis_col] == v) & trajectory_df[y_col].notna()
            ][['tokens_seen', y_col, 'seed']].sort_values('tokens_seen')
            if sub.empty:
                continue
            seed_curves = sub.pivot_table(
                index='tokens_seen', columns='seed', values=y_col, aggfunc='mean',
            ).sort_index()
            if smoothing_window and smoothing_window > 1:
                seed_curves = seed_curves.rolling(
                    window=smoothing_window, min_periods=1,
                ).mean()
            mean_curve = seed_curves.mean(axis=1)
            std_curve = seed_curves.std(axis=1, ddof=1)
            ax.plot(
                mean_curve.index, mean_curve.values,
                color=color, linewidth=1.4,
                label=f'{axis_label[1:-1]}={int(v)}',
            )
            ax.fill_between(
                mean_curve.index,
                (mean_curve - std_curve).values,
                (mean_curve + std_curve).values,
                color=color, alpha=0.15, linewidth=0,
            )
        ax.set_xlabel('Tokens seen')
        ax.set_ylabel(title)
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, -0.02), ncol=min(len(labels), 6),
                   fontsize=7, frameon=False)
    fig.suptitle(
        f'Training trajectories by {axis_label}    (mean±std over seeds)',
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    return _save_dual(fig, output_dir, 'fig_trajectory_by_K')


def fig_gauge_vs_K(
    terminal_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    output_dir: Path,
    axis_col: str = 'K',
    axis_label: str = r'$K$',
) -> Path:
    """2x2: gauge field energy, holonomy norm, attention entropy, orbit dim — at terminal."""
    set_pub_style()

    panels = [
        ('gauge_field_energy', 'Gauge field energy'),
        ('holonomy_mean_norm', r'Mean $\Vert C - I \Vert_F$'),
        ('attention_entropy', 'Attention entropy'),
        ('gauge_orbit_dim', 'Gauge orbit dim'),
    ]

    if not trajectory_df.empty:
        terminal_per_run = (
            trajectory_df.sort_values('step')
            .groupby([axis_col, 'seed'], as_index=False)
            .last()
        )
    else:
        terminal_per_run = pd.DataFrame(columns=[axis_col, 'seed'])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (col, ylab) in zip(axes.flat, panels):
        if col not in terminal_per_run.columns or terminal_per_run[col].isna().all():
            ax.text(0.5, 0.5, f'{col} unavailable',
                    transform=ax.transAxes, ha='center', va='center',
                    color=PUB_COLORS['gray'])
            ax.set_title(ylab, fontsize=9)
            continue
        agg = _agg_per_axis(
            terminal_per_run.dropna(subset=[col]), axis_col, col,
        )
        ax.errorbar(
            agg[axis_col], agg['mean'], yerr=agg['std'],
            fmt='o-', color=PUB_COLORS['blue'], capsize=3, markersize=5,
            linewidth=1.2,
        )
        ax.set_xlabel(axis_label)
        ax.set_ylabel(ylab)
        ax.grid(True, which='both', alpha=0.25)
    fig.suptitle(
        f'Geometric and attention diagnostics at terminal step (vs {axis_label})',
        fontsize=10, y=1.0,
    )
    fig.tight_layout()
    return _save_dual(fig, output_dir, 'fig_gauge_vs_K')


def render_all(
    terminal_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    fit: PowerLawFit,
    output_dir: Path,
    axis_col: str = 'K',
    axis_label: str = r'$K$',
    smoothing_window: Optional[int] = 50,
) -> Dict[str, Path]:
    """Render all four figures and return a dict of name -> PNG path."""
    return {
        'fig_scaling_main': fig_scaling_main(
            terminal_df, fit, output_dir, axis_col, axis_label,
        ),
        'fig_compute_frontier': fig_compute_frontier(
            terminal_df, output_dir, axis_col, axis_label,
        ),
        'fig_trajectory_by_K': fig_trajectory_by_K(
            trajectory_df, output_dir, axis_col, axis_label,
            smoothing_window=smoothing_window,
        ),
        'fig_gauge_vs_K': fig_gauge_vs_K(
            terminal_df, trajectory_df, output_dir, axis_col, axis_label,
        ),
    }
