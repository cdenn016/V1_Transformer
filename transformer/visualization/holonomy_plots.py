"""
Holonomy Visualization: Publication-quality figures for curvature analysis.
===========================================================================

Produces figures for the key question: "Does language have curvature?"

Figures:
    1. Holonomy norm distribution -- histogram + KDE of ‖C_ijk - I‖_F
    2. Holonomy evolution -- mean/max norm over training steps
    3. Per-layer holonomy profile -- bar chart across layers
    4. Curvature vs distance -- holonomy as function of triangle size
    5. Wilson loop spectrum -- eigenvalue distribution of C_ijk
    6. Flat vs non-flat comparison -- side-by-side PPL curves with holonomy overlay
    7. Per-head holonomy heatmap -- which heads learn curvature?
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.ticker as ticker
    from matplotlib.colors import Normalize
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    import torch
except ImportError:
    torch = None

from transformer.visualization.pub_style import set_pub_style, PUB_COLORS


# =============================================================================
# Publication Style
# =============================================================================

HOLONOMY_STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'cyan': '#56B4E9',
    'yellow': '#F0E442',
    'black': '#000000',
}


def _apply_style():
    if MATPLOTLIB_AVAILABLE:
        set_pub_style()
        plt.rcParams.update(HOLONOMY_STYLE)


def _format_step_axis(ax):
    """Format x-axis: 150000 -> '150k'."""
    def fmt(x, pos):
        return f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))


def _require_matplotlib():
    """Raise ImportError if matplotlib is not available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for holonomy visualization")


# =============================================================================
# Figure 1: Holonomy Norm Distribution
# =============================================================================

def plot_holonomy_distribution(
    norms: np.ndarray,
    title: str = 'Holonomy Norm Distribution',
    thresholds: Tuple[float, ...] = (0.01, 0.1, 1.0),
    log_scale: bool = True,
    ax=None,
    output_path: Optional[Path] = None,
):
    """Histogram + KDE of ‖C_ijk - I‖_F with threshold annotations."""
    _require_matplotlib()
    _apply_style()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Histogram
    if log_scale and norms.min() > 0:
        bins = np.logspace(np.log10(max(norms.min(), 1e-6)), np.log10(norms.max() + 1e-8), 50)
        ax.set_xscale('log')
    else:
        bins = 50

    ax.hist(norms, bins=bins, density=True, alpha=0.7, color=COLORS['blue'],
            edgecolor='white', linewidth=0.5, label='Sampled triples')

    # Threshold lines
    colors_th = [COLORS['green'], COLORS['orange'], COLORS['red']]
    labels_th = ['Flat regime', 'Moderate curvature', 'Strong curvature']
    for th, c, lab in zip(thresholds, colors_th, labels_th):
        ax.axvline(th, color=c, linestyle='--', linewidth=1.2, alpha=0.8, label=f'{lab} ({th})')

    ax.set_xlabel(r'$\|C_{ijk} - I\|_F$')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(loc='upper right', framealpha=0.9)

    if own_fig:
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path)
            plt.close(fig)
        return fig
    return None


# =============================================================================
# Figure 2: Holonomy Evolution Over Training
# =============================================================================

def plot_holonomy_evolution(
    steps: np.ndarray,
    global_mean: np.ndarray,
    global_max: np.ndarray,
    per_layer_mean: Optional[np.ndarray] = None,
    layer_indices: Optional[List[int]] = None,
    title: str = 'Holonomy Evolution',
    output_path: Optional[Path] = None,
):
    """Training curves: global mean/max + optional per-layer decomposition."""
    _require_matplotlib()
    _apply_style()
    n_panels = 2 if per_layer_mean is not None and per_layer_mean.shape[1] > 0 else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: Global mean + max
    ax = axes[0]
    ax.plot(steps, global_mean, color=COLORS['blue'], label='Mean $\\|C - I\\|_F$')
    ax.fill_between(steps, 0, global_mean, color=COLORS['blue'], alpha=0.15)
    ax.plot(steps, global_max, color=COLORS['red'], linestyle='--', alpha=0.7,
            label='Max $\\|C - I\\|_F$')
    ax.set_xlabel('Training Step')
    ax.set_ylabel(r'$\|C_{ijk} - I\|_F$')
    ax.set_title('Global Holonomy')
    ax.legend()
    _format_step_axis(ax)

    # Panel 2: Per-layer decomposition
    if n_panels > 1:
        ax = axes[1]
        n_layers = per_layer_mean.shape[1]
        cmap = plt.cm.viridis(np.linspace(0.2, 0.9, n_layers))
        for li in range(n_layers):
            label = f'Layer {layer_indices[li]}' if layer_indices else f'Layer {li}'
            ax.plot(steps, per_layer_mean[:, li], color=cmap[li],
                    label=label, linewidth=1.2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel(r'Mean $\|C - I\|_F$')
        ax.set_title('Per-Layer Holonomy')
        ax.legend(fontsize=8, ncol=2)
        _format_step_axis(ax)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


# =============================================================================
# Figure 3: Per-Layer Holonomy Profile (Bar Chart)
# =============================================================================

def plot_layer_holonomy_profile(
    layer_means: np.ndarray,
    layer_stds: Optional[np.ndarray] = None,
    head_means: Optional[np.ndarray] = None,
    title: str = 'Per-Layer Holonomy Profile',
    output_path: Optional[Path] = None,
):
    """Bar chart of per-layer holonomy, optionally with per-head heatmap."""
    _require_matplotlib()
    _apply_style()
    n_layers = len(layer_means)

    if head_means is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    # Bar chart: per-layer
    bars = ax1.bar(range(n_layers), layer_means, yerr=layer_stds,
                   color=COLORS['blue'], alpha=0.8, capsize=3,
                   edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel(r'Mean $\|C_{ijk} - I\|_F$')
    ax1.set_title(title)
    ax1.set_xticks(range(n_layers))

    # Heatmap: per-layer x per-head
    if head_means is not None:
        im = ax2.imshow(head_means.T, aspect='auto', cmap='YlOrRd',
                        interpolation='nearest')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Head')
        ax2.set_title('Holonomy by Layer x Head')
        ax2.set_xticks(range(n_layers))
        ax2.set_yticks(range(head_means.shape[1]))
        fig.colorbar(im, ax=ax2, label=r'$\|C - I\|_F$', shrink=0.8)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


# =============================================================================
# Figure 4: Curvature vs Triangle Size (Distance)
# =============================================================================

def plot_curvature_vs_distance(
    bin_centers: np.ndarray,
    mean_norms: np.ndarray,
    std_norms: np.ndarray,
    counts: np.ndarray,
    title: str = 'Curvature vs Token Distance',
    output_path: Optional[Path] = None,
):
    """Holonomy norm as a function of mean pairwise token distance."""
    _require_matplotlib()
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), height_ratios=[3, 1],
                                    sharex=True)

    # Top: curvature vs distance
    valid = counts > 0
    ax1.errorbar(bin_centers[valid], mean_norms[valid], yerr=std_norms[valid],
                 fmt='o-', color=COLORS['blue'], capsize=3, markersize=5,
                 linewidth=1.5, label='Mean holonomy')
    ax1.set_ylabel(r'$\|C_{ijk} - I\|_F$')
    ax1.set_title(title)
    ax1.legend()

    # Bottom: sample counts per bin
    ax2.bar(bin_centers[valid], counts[valid], width=np.diff(bin_centers[:2])[0] * 0.8
            if len(bin_centers) > 1 else 1.0,
            color=COLORS['cyan'], alpha=0.6, edgecolor='white')
    ax2.set_xlabel('Mean Pairwise Token Distance')
    ax2.set_ylabel('Count')
    ax2.set_yscale('log')

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


# =============================================================================
# Figure 5: Wilson Loop Spectrum
# =============================================================================

def plot_wilson_spectrum(
    C_matrices,
    max_samples: int = 500,
    title: str = 'Wilson Loop Eigenvalue Spectrum',
    output_path: Optional[Path] = None,
):
    """Eigenvalue scatter in complex plane + |lambda| histogram."""
    _require_matplotlib()
    _apply_style()

    n = min(len(C_matrices), max_samples)
    C_sub = C_matrices[:n]

    if torch is not None and isinstance(C_sub, torch.Tensor):
        eigvals = torch.linalg.eigvals(C_sub).cpu().numpy()
    else:
        eigvals = np.linalg.eigvals(C_sub)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: complex plane scatter
    ax1.scatter(eigvals.real.flatten(), eigvals.imag.flatten(),
                s=3, alpha=0.3, color=COLORS['blue'])
    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=0.8, alpha=0.4)
    ax1.plot(1, 0, 'r+', markersize=12, markeredgewidth=2, label='Identity')
    ax1.set_xlabel(r'Re($\lambda$)')
    ax1.set_ylabel(r'Im($\lambda$)')
    ax1.set_title('Eigenvalues in Complex Plane')
    ax1.set_aspect('equal')
    ax1.legend()

    # Right: |λ| distribution
    eigvals_abs = np.abs(eigvals).flatten()
    ax2.hist(eigvals_abs, bins=50, density=True, alpha=0.7,
             color=COLORS['orange'], edgecolor='white', linewidth=0.5)
    ax2.axvline(1.0, color=COLORS['red'], linestyle='--', linewidth=1.2,
                label='$|\\lambda| = 1$ (flat)')
    ax2.set_xlabel(r'$|\lambda|$')
    ax2.set_ylabel('Density')
    ax2.set_title('Eigenvalue Magnitude Distribution')
    ax2.legend()

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


# =============================================================================
# Figure 6: Flat vs Non-Flat PPL Comparison
# =============================================================================

def plot_flat_vs_nonflat_comparison(
    flat_steps: np.ndarray,
    flat_ppl: np.ndarray,
    nonflat_steps: np.ndarray,
    nonflat_ppl: np.ndarray,
    holonomy_steps: Optional[np.ndarray] = None,
    holonomy_mean: Optional[np.ndarray] = None,
    title: str = 'Flat vs Non-Flat Transport',
    output_path: Optional[Path] = None,
):
    """Side-by-side PPL curves with optional holonomy overlay on twin axis."""
    _require_matplotlib()
    _apply_style()
    has_holonomy = holonomy_steps is not None and holonomy_mean is not None
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 4.5))

    # PPL curves
    ax1.plot(flat_steps, flat_ppl, color=COLORS['blue'], linewidth=2,
             label=r'Flat ($\delta_{ij} = 0$)')
    ax1.plot(nonflat_steps, nonflat_ppl, color=COLORS['red'], linewidth=2,
             label=r'Non-flat ($\delta_{ij} \neq 0$)')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Validation Perplexity')
    ax1.set_title(title)
    _format_step_axis(ax1)

    # PPL gap annotation
    if len(flat_ppl) > 0 and len(nonflat_ppl) > 0:
        final_flat = flat_ppl[-1]
        final_nonflat = nonflat_ppl[-1]
        gap = final_flat - final_nonflat
        pct = 100 * gap / final_flat if final_flat > 0 else 0
        sign = '+' if gap > 0 else ''
        ax1.annotate(
            f'$\\Delta$PPL = {sign}{gap:.2f} ({sign}{pct:.1f}%)',
            xy=(0.98, 0.95), xycoords='axes fraction',
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8),
        )

    # Holonomy overlay on twin axis
    if has_holonomy:
        ax2 = ax1.twinx()
        ax2.plot(holonomy_steps, holonomy_mean, color=COLORS['green'],
                 linestyle=':', linewidth=1.5, alpha=0.8,
                 label=r'Mean $\|C - I\|_F$')
        ax2.set_ylabel(r'Mean Holonomy $\|C - I\|_F$', color=COLORS['green'])
        ax2.tick_params(axis='y', labelcolor=COLORS['green'])
        ax2.legend(loc='center right')

    ax1.legend(loc='upper right')
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


# =============================================================================
# Figure 7: Multi-Panel Summary (the "money figure")
# =============================================================================

def plot_holonomy_summary(
    norms: np.ndarray,
    steps: np.ndarray,
    global_mean: np.ndarray,
    global_max: np.ndarray,
    layer_means: np.ndarray,
    C_matrices=None,
    title: str = 'Holonomy Summary',
    output_path: Optional[Path] = None,
):
    """Multi-panel overview: distribution, evolution, per-layer, and spectrum."""
    _require_matplotlib()
    _apply_style()
    n_cols = 3 if C_matrices is None else 4
    fig = plt.figure(figsize=(4.5 * n_cols, 4))
    gs = GridSpec(1, n_cols, figure=fig, wspace=0.35)

    # Panel A: Distribution
    ax_a = fig.add_subplot(gs[0, 0])
    plot_holonomy_distribution(norms, title='(a) Distribution', ax=ax_a)

    # Panel B: Evolution
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(steps, global_mean, color=COLORS['blue'], label='Mean')
    ax_b.fill_between(steps, 0, global_mean, color=COLORS['blue'], alpha=0.15)
    ax_b.plot(steps, global_max, color=COLORS['red'], linestyle='--', alpha=0.7, label='Max')
    ax_b.set_xlabel('Training Step')
    ax_b.set_ylabel(r'$\|C - I\|_F$')
    ax_b.set_title('(b) Training Evolution')
    ax_b.legend(fontsize=8)
    _format_step_axis(ax_b)

    # Panel C: Per-layer profile
    ax_c = fig.add_subplot(gs[0, 2])
    n_layers = len(layer_means)
    ax_c.bar(range(n_layers), layer_means, color=COLORS['blue'], alpha=0.8,
             edgecolor='white', linewidth=0.5)
    ax_c.set_xlabel('Layer')
    ax_c.set_ylabel(r'Mean $\|C - I\|_F$')
    ax_c.set_title('(c) Per-Layer Profile')
    ax_c.set_xticks(range(n_layers))

    # Panel D: Wilson spectrum (if holonomy matrices provided as array)
    if C_matrices is not None and isinstance(C_matrices, (np.ndarray,) + ((torch.Tensor,) if torch else ())):
        ax_d = fig.add_subplot(gs[0, 3])
        n = min(len(C_matrices), 300)
        if torch is not None and isinstance(C_matrices, torch.Tensor):
            eigvals = torch.linalg.eigvals(C_matrices[:n]).cpu().numpy()
        else:
            eigvals = np.linalg.eigvals(C_matrices[:n])
        ax_d.scatter(eigvals.real.flatten(), eigvals.imag.flatten(),
                     s=2, alpha=0.3, color=COLORS['purple'])
        theta = np.linspace(0, 2 * np.pi, 100)
        ax_d.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=0.5, alpha=0.3)
        ax_d.plot(1, 0, 'r+', markersize=10, markeredgewidth=2)
        ax_d.set_xlabel(r'Re($\lambda$)')
        ax_d.set_ylabel(r'Im($\lambda$)')
        ax_d.set_title('(d) Wilson Spectrum')
        ax_d.set_aspect('equal')

    fig.suptitle(title, fontsize=14, y=1.04)
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig
