"""
Shared publication-quality style settings for all visualization modules.

Provides a single source of truth for matplotlib rcParams and colorblind-safe
palettes. Every plot function in transformer/visualization/ should call
set_pub_style() before creating figures.

Usage:
    from transformer.visualization.pub_style import set_pub_style, PUB_COLORS
    set_pub_style()
    fig, ax = plt.subplots(...)
"""

from __future__ import annotations

import warnings

try:
    from matplotlib import rcParams
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# Colorblind-safe palette (Okabe-Ito, reordered for contrast)
# =============================================================================

PUB_COLORS = {
    'blue':       '#0072B2',
    'orange':     '#E69F00',
    'green':      '#009E73',
    'red':        '#D55E00',
    'purple':     '#CC79A7',
    'cyan':       '#56B4E9',
    'yellow':     '#F0E442',
    'black':      '#000000',
    'gray':       '#999999',
}

# Ordered list for sequential use (maximally distinct neighbors)
PUB_CYCLE = [
    PUB_COLORS['blue'],
    PUB_COLORS['orange'],
    PUB_COLORS['green'],
    PUB_COLORS['red'],
    PUB_COLORS['purple'],
    PUB_COLORS['cyan'],
    PUB_COLORS['yellow'],
    PUB_COLORS['black'],
]


def set_pub_style() -> None:
    """Apply publication-quality matplotlib defaults.

    Enforces:
        - Serif font family (Computer Modern / Times)
        - Axis labels 11pt, titles 12pt, ticks 9pt, legend 8pt
        - 300 DPI for saved figures
        - Top/right spines removed
        - Subtle grid (alpha 0.3)
        - Tight bounding box on save
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    rcParams.update({
        # Typography
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',

        # Figure
        'figure.dpi': 150,
        'figure.figsize': (8, 5),

        # Saving
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Axes
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.prop_cycle': __import__('cycler').cycler('color', PUB_CYCLE),

        # Grid
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,

        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 5,

        # Ticks
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })


def _safe_legend(ax: "plt.Axes", *args, **kwargs) -> None:
    """Call ax.legend() only if labeled artists exist, suppressing UserWarning on empty legend."""
    if args:
        ax.legend(*args, **kwargs)
        return
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(**kwargs)
