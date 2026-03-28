from __future__ import annotations

"""
VFE Dynamics Visualization
===========================

Publication-quality figures for understanding variational free energy dynamics
in the gauge-theoretic transformer. Visualizes:

1. VFE gradient decomposition (self-coupling vs alignment vs softmax coupling)
2. Covariance health (eigenvalue evolution, condition numbers, prior-belief gap)
3. Transport operator statistics (phi norms, pairwise distances)
4. Attention information-theoretic profile (per-head entropy, head diversity)
5. Combined dynamics dashboard (multi-panel summary)

Input: metrics.csv from train_publication.py containing VFE dynamics columns.

Usage:
    python -m transformer.visualization.vfe_dynamics_plots --file metrics.csv
    python -m transformer.visualization.vfe_dynamics_plots --file metrics.csv --mode dashboard
"""

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from typing import TYPE_CHECKING

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    pass  # MATPLOTLIB_AVAILABLE checked via pub_style

if TYPE_CHECKING:
    from matplotlib.figure import Figure

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from transformer.visualization.pub_style import set_pub_style, PUB_COLORS, MATPLOTLIB_AVAILABLE


def _smooth(data: list, window: int = 15) -> np.ndarray:
    """Savitzky-Golay smoothing with fallback to raw data."""
    arr = np.array(data, dtype=float)
    if SCIPY_AVAILABLE and len(arr) > window and window >= 3:
        w = min(window, len(arr) - 1)
        if w % 2 == 0:
            w -= 1
        if w >= 3:
            return savgol_filter(arr, w, min(3, w - 1))
    return arr


# =============================================================================
# CSV Loading
# =============================================================================

def load_vfe_metrics(csv_path: Path) -> Dict[str, list]:
    r"""Load VFE dynamics metrics from publication CSV.

    Returns dict mapping column name to list of float values.
    Skips rows where the column is empty/None.
    """
    data: Dict[str, list] = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                if val is not None and val.strip() != '':
                    try:
                        data[key].append(float(val))
                    except (ValueError, TypeError):
                        data[key].append(None)
                else:
                    data[key].append(None)
    return data


def _get_valid(data: Dict, key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (steps, values) arrays where values are not None."""
    if key not in data or 'step' not in data:
        return np.array([]), np.array([])
    steps = data['step']
    vals = data[key]
    s_out, v_out = [], []
    for s, v in zip(steps, vals):
        if s is not None and v is not None:
            s_out.append(float(s))
            v_out.append(float(v))
    return np.array(s_out), np.array(v_out)


# =============================================================================
# Figure 1: VFE Gradient Decomposition
# =============================================================================

def plot_vfe_gradient_decomposition(
    data: Dict[str, list],
    save_path: Optional[Path] = None,
    smooth_window: int = 15,
) -> Optional["Figure"]:
    r"""Plot VFE gradient component decomposition over training.

    Shows how the E-step gradient $\nabla_\mu F$ and $\nabla_\Sigma F$ are
    composed of self-coupling (prior KL), alignment (attention-weighted KL),
    and softmax coupling (higher-order $\partial\beta/\partial\theta$) terms.

    This is the single most informative diagnostic for understanding what
    drives belief evolution in the VFE transformer.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    set_pub_style()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # --- Panel A: mu gradient components ---
    ax = axes[0]
    components_mu = [
        ('vfe_grad_mu_self', 'Self-coupling', '#2196F3'),
        ('vfe_grad_mu_direct', 'Alignment (direct)', '#FF9800'),
        ('vfe_grad_mu_softmax', 'Softmax coupling', '#4CAF50'),
        ('vfe_grad_mu_total', 'Total', '#212121'),
    ]
    _has_data = False
    for col, label, color in components_mu:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            _has_data = True
            ls = '-' if 'total' not in col.lower() else '-'
            lw = 2.0 if 'total' in col.lower() else 1.2
            ax.plot(steps, _smooth(vals, smooth_window), label=label,
                    color=color, linewidth=lw, linestyle=ls)
            # Light raw data
            ax.plot(steps, vals, color=color, alpha=0.15, linewidth=0.5)
    if not _has_data:
        ax.text(0.5, 0.5, 'No VFE gradient data available',
                transform=ax.transAxes, ha='center', va='center', fontsize=12, color='gray')
    ax.set_ylabel(r'$\|\nabla_\mu F\|$')
    ax.set_yscale('log')
    ax.set_title(r'(a) Mean gradient $\nabla_\mu F$ decomposition')
    ax.legend(loc='upper right', framealpha=0.9)

    # --- Panel B: sigma gradient components ---
    ax = axes[1]
    components_sigma = [
        ('vfe_grad_sigma_self', 'Self-coupling', '#2196F3'),
        ('vfe_grad_sigma_align_direct', 'Alignment (direct)', '#FF9800'),
        ('vfe_grad_sigma_softmax', 'Softmax coupling', '#4CAF50'),
        ('vfe_grad_sigma_total', 'Total', '#212121'),
    ]
    _has_data = False
    for col, label, color in components_sigma:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            _has_data = True
            lw = 2.0 if 'total' in col.lower() else 1.2
            ax.plot(steps, _smooth(vals, smooth_window), label=label,
                    color=color, linewidth=lw)
            ax.plot(steps, vals, color=color, alpha=0.15, linewidth=0.5)
    if not _has_data:
        ax.text(0.5, 0.5, 'No VFE gradient data available',
                transform=ax.transAxes, ha='center', va='center', fontsize=12, color='gray')
    ax.set_ylabel(r'$\|\nabla_\Sigma F\|$')
    ax.set_xlabel('Training step')
    ax.set_yscale('log')
    ax.set_title(r'(b) Covariance gradient $\nabla_\Sigma F$ decomposition')
    ax.legend(loc='upper right', framealpha=0.9)

    fig.suptitle('VFE Gradient Component Decomposition', fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# Figure 2: Covariance Health
# =============================================================================

def plot_covariance_health(
    data: Dict[str, list],
    save_path: Optional[Path] = None,
    smooth_window: int = 15,
) -> Optional["Figure"]:
    r"""Plot covariance health metrics over training.

    Panel (a): $\Sigma_q$ statistics (mean, min, max diagonal values)
    Panel (b): Condition number $\kappa(\Sigma_q)$ = max/min eigenvalue
    Panel (c): Prior-belief gap $KL(q^* \| p)$ distribution statistics
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    set_pub_style()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # --- Panel A: Sigma_q statistics ---
    ax = axes[0]
    for col, label, color in [
        ('sigma_q_mean', r'$\bar{\sigma}_q$', '#2196F3'),
        ('sigma_q_min', r'$\sigma_q^{\min}$', '#F44336'),
        ('sigma_q_max', r'$\sigma_q^{\max}$', '#4CAF50'),
    ]:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            ax.plot(steps, _smooth(vals, smooth_window), label=label, color=color, linewidth=1.5)
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.5)
    # Prior sigma overlay
    for col, label, ls in [
        ('sigma_p_mean', r'$\bar{\sigma}_p$ (prior)', '--'),
    ]:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            ax.plot(steps, _smooth(vals, smooth_window), label=label,
                    color='#9E9E9E', linewidth=1.2, linestyle=ls)
    ax.set_ylabel('Diagonal covariance')
    ax.set_xlabel('Training step')
    ax.set_title(r'(a) Belief covariance $\Sigma_q$')
    ax.legend(loc='best', framealpha=0.9)

    # --- Panel B: Condition number ---
    ax = axes[1]
    for col, label, color in [
        ('sigma_q_cond_mean', r'$\bar{\kappa}(\Sigma_q)$', '#9C27B0'),
        ('sigma_q_cond_max', r'$\kappa_{\max}(\Sigma_q)$', '#FF5722'),
    ]:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            ax.plot(steps, _smooth(vals, smooth_window), label=label, color=color, linewidth=1.5)
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.5)
    ax.set_ylabel(r'Condition number $\kappa$')
    ax.set_xlabel('Training step')
    ax.set_yscale('log')
    ax.set_title(r'(b) Covariance conditioning')
    ax.legend(loc='best', framealpha=0.9)

    # --- Panel C: Prior-belief gap ---
    ax = axes[2]
    for col, label, color in [
        ('prior_belief_kl_mean', r'$\overline{KL}(q^*\|p)$', '#E91E63'),
        ('prior_belief_kl_max', r'$KL_{\max}(q^*\|p)$', '#FF9800'),
    ]:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            ax.plot(steps, _smooth(vals, smooth_window), label=label, color=color, linewidth=1.5)
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.5)
    ax.set_ylabel(r'$KL(q^* \| p)$ (nats)')
    ax.set_xlabel('Training step')
    ax.set_title(r'(c) Prior-belief divergence')
    ax.legend(loc='best', framealpha=0.9)

    fig.suptitle('Covariance Health & Prior-Belief Gap', fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# Figure 3: Transport & Attention Structure
# =============================================================================

def plot_transport_attention(
    data: Dict[str, list],
    save_path: Optional[Path] = None,
    smooth_window: int = 15,
) -> Optional["Figure"]:
    r"""Plot transport operator and attention information-theoretic metrics.

    Panel (a): Gauge frame $\phi$ norms and pairwise distances (transport proxy)
    Panel (b): Per-head attention entropy and head diversity (correlation)
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    set_pub_style()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel A: Transport / phi statistics ---
    ax = axes[0]
    ax2 = ax.twinx()
    for col, label, color in [
        ('phi_norm_mean', r'$\overline{\|\phi\|}$', '#1565C0'),
        ('phi_norm_max', r'$\|\phi\|_{\max}$', '#42A5F5'),
    ]:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            ax.plot(steps, _smooth(vals, smooth_window), label=label, color=color, linewidth=1.5)
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.5)
    ax.set_ylabel(r'Gauge frame $\|\phi\|$', color='#1565C0')
    ax.tick_params(axis='y', labelcolor='#1565C0')

    for col, label, color in [
        ('phi_pairwise_dist_mean', r'$\overline{\|\phi_i - \phi_j\|}$', '#C62828'),
        ('phi_pairwise_dist_max', r'$\|\phi_i - \phi_j\|_{\max}$', '#EF5350'),
    ]:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            ax2.plot(steps, _smooth(vals, smooth_window), label=label,
                     color=color, linewidth=1.5, linestyle='--')
            ax2.plot(steps, vals, color=color, alpha=0.12, linewidth=0.5)
    ax2.set_ylabel(r'Pairwise $\|\phi_i - \phi_j\|$', color='#C62828')
    ax2.tick_params(axis='y', labelcolor='#C62828')

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', framealpha=0.9)
    ax.set_xlabel('Training step')
    ax.set_title(r'(a) Transport geometry ($\phi$ statistics)')

    # --- Panel B: Attention entropy per head ---
    ax = axes[1]
    for col, label, color in [
        ('attn_entropy_per_head_mean', r'$\bar{H}(\beta_h)$', '#2E7D32'),
        ('attn_entropy_per_head_min', r'$H_{\min}(\beta_h)$', '#A5D6A7'),
        ('attn_entropy_per_head_max', r'$H_{\max}(\beta_h)$', '#1B5E20'),
    ]:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            ax.plot(steps, _smooth(vals, smooth_window), label=label, color=color, linewidth=1.5)
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.5)
    # Head correlation on secondary axis
    ax2 = ax.twinx()
    steps, vals = _get_valid(data, 'head_correlation_mean')
    if len(vals) > 0:
        ax2.plot(steps, _smooth(vals, smooth_window), label=r'$\bar{r}_{hh}$ (head corr)',
                 color='#FF6F00', linewidth=1.5, linestyle=':')
        ax2.set_ylabel(r'Inter-head correlation $\bar{r}_{hh}$', color='#FF6F00')
        ax2.tick_params(axis='y', labelcolor='#FF6F00')
        ax2.set_ylim(-0.2, 1.0)

    ax.set_ylabel(r'Attention entropy $H(\beta)$')
    ax.set_xlabel('Training step')
    ax.set_title(r'(b) Attention entropy & head diversity')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', framealpha=0.9)

    fig.suptitle('Transport Geometry & Attention Structure', fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# Figure 4: VFE KL Landscape
# =============================================================================

def plot_kl_landscape(
    data: Dict[str, list],
    save_path: Optional[Path] = None,
    smooth_window: int = 15,
) -> Optional["Figure"]:
    r"""Plot pairwise KL divergence statistics and effective temperature.

    Shows how the KL divergences $KL(q_i \| \Omega_{ij} q_j)$ that drive
    attention evolve, along with the effective temperature $\kappa \sqrt{K}$.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    set_pub_style()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel A: Pairwise KL ---
    ax = axes[0]
    for col, label, color in [
        ('vfe_kl_pairwise_mean', r'$\overline{KL}_{ij}$', '#7B1FA2'),
        ('vfe_kl_pairwise_max', r'$KL_{\max}$', '#E040FB'),
    ]:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            ax.plot(steps, _smooth(vals, smooth_window), label=label, color=color, linewidth=1.5)
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.5)
    ax.set_ylabel(r'Pairwise $KL(q_i \| \Omega_{ij} q_j)$')
    ax.set_xlabel('Training step')
    ax.set_title(r'(a) Pairwise KL divergence (attention driver)')
    ax.legend(loc='best', framealpha=0.9)

    # --- Panel B: Effective temperature ---
    ax = axes[1]
    steps, vals = _get_valid(data, 'vfe_kappa_scaled')
    if len(vals) > 0:
        ax.plot(steps, _smooth(vals, smooth_window), label=r'$\kappa_{\mathrm{eff}} = \kappa\sqrt{K}$',
                color='#F57F17', linewidth=2.0)
        ax.plot(steps, vals, color='#F57F17', alpha=0.12, linewidth=0.5)
    # Also show attention entropy for context
    steps2, vals2 = _get_valid(data, 'attention_entropy')
    if len(vals2) > 0:
        ax2 = ax.twinx()
        ax2.plot(steps2, _smooth(vals2, smooth_window), label=r'$H(\beta)$',
                 color='#2E7D32', linewidth=1.5, linestyle='--')
        ax2.set_ylabel(r'Attention entropy $H(\beta)$', color='#2E7D32')
        ax2.tick_params(axis='y', labelcolor='#2E7D32')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='best', framealpha=0.9)
    else:
        ax.legend(loc='best', framealpha=0.9)
    ax.set_ylabel(r'Effective temperature $\kappa_{\mathrm{eff}}$')
    ax.set_xlabel('Training step')
    ax.set_title(r'(b) Softmax temperature & attention sharpness')

    fig.suptitle('KL Divergence Landscape', fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# Figure 5: Combined Dashboard
# =============================================================================

def plot_vfe_dynamics_dashboard(
    data: Dict[str, list],
    save_path: Optional[Path] = None,
    smooth_window: int = 15,
) -> Optional["Figure"]:
    r"""Six-panel dashboard summarizing all VFE dynamics in one figure.

    Layout (3x2):
        (a) Training loss + PPL          (b) VFE gradient decomposition (mu)
        (c) Covariance health            (d) Prior-belief gap
        (e) Transport phi statistics     (f) Attention entropy + head diversity
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    set_pub_style()

    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    # --- (a) Training loss + PPL ---
    ax = fig.add_subplot(gs[0, 0])
    for col, label, color in [
        ('train_loss_total', 'Train loss', '#1976D2'),
        ('val_loss', 'Val loss', '#D32F2F'),
    ]:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            ax.plot(steps, _smooth(vals, smooth_window), label=label, color=color, linewidth=1.5)
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.5)
    ax.set_ylabel('Loss')
    ax.set_title('(a) Training & validation loss')
    ax.legend(loc='upper right', framealpha=0.9)

    # PPL on secondary axis
    ax2 = ax.twinx()
    steps, vals = _get_valid(data, 'val_ppl')
    if len(vals) > 0:
        ax2.plot(steps, _smooth(vals, smooth_window), label='Val PPL',
                 color='#FF6F00', linewidth=1.5, linestyle='--')
        ax2.set_ylabel('Perplexity', color='#FF6F00')
        ax2.tick_params(axis='y', labelcolor='#FF6F00')

    # --- (b) VFE gradient decomposition (mu) ---
    ax = fig.add_subplot(gs[0, 1])
    for col, label, color in [
        ('vfe_grad_mu_self', 'Self-coupling', '#2196F3'),
        ('vfe_grad_mu_direct', 'Alignment', '#FF9800'),
        ('vfe_grad_mu_softmax', 'Softmax', '#4CAF50'),
        ('vfe_grad_mu_total', 'Total', '#212121'),
    ]:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            lw = 2.0 if 'total' in col else 1.2
            ax.plot(steps, _smooth(vals, smooth_window), label=label, color=color, linewidth=lw)
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.5)
    ax.set_ylabel(r'$\|\nabla_\mu F\|$')
    ax.set_yscale('log')
    ax.set_title(r'(b) VFE $\nabla_\mu$ decomposition')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=7)

    # --- (c) Covariance health ---
    ax = fig.add_subplot(gs[1, 0])
    for col, label, color in [
        ('sigma_q_mean', r'$\bar{\sigma}_q$', '#2196F3'),
        ('sigma_q_min', r'$\sigma_q^{\min}$', '#F44336'),
        ('sigma_q_max', r'$\sigma_q^{\max}$', '#4CAF50'),
        ('sigma_p_mean', r'$\bar{\sigma}_p$', '#9E9E9E'),
    ]:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            ls = '--' if 'sigma_p' in col else '-'
            ax.plot(steps, _smooth(vals, smooth_window), label=label,
                    color=color, linewidth=1.5, linestyle=ls)
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.5)
    ax.set_ylabel('Diagonal covariance')
    ax.set_title(r'(c) Belief covariance $\Sigma_q$ vs prior $\Sigma_p$')
    ax.legend(loc='best', framealpha=0.9, fontsize=7)

    # --- (d) Prior-belief gap ---
    ax = fig.add_subplot(gs[1, 1])
    for col, label, color in [
        ('prior_belief_kl_mean', r'$\overline{KL}(q^*\|p)$', '#E91E63'),
        ('prior_belief_kl_max', r'$KL_{\max}(q^*\|p)$', '#FF9800'),
        ('prior_belief_kl_std', r'$\mathrm{std}[KL]$', '#9C27B0'),
    ]:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            ax.plot(steps, _smooth(vals, smooth_window), label=label, color=color, linewidth=1.5)
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.5)
    ax.set_ylabel(r'$KL(q^* \| p)$ (nats)')
    ax.set_title('(d) Prior-belief divergence')
    ax.legend(loc='best', framealpha=0.9, fontsize=7)

    # --- (e) Transport phi statistics ---
    ax = fig.add_subplot(gs[2, 0])
    for col, label, color in [
        ('phi_norm_mean', r'$\overline{\|\phi\|}$', '#1565C0'),
        ('phi_pairwise_dist_mean', r'$\overline{\|\phi_i-\phi_j\|}$', '#C62828'),
    ]:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            ax.plot(steps, _smooth(vals, smooth_window), label=label, color=color, linewidth=1.5)
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.5)
    ax.set_ylabel('Norm')
    ax.set_xlabel('Training step')
    ax.set_title(r'(e) Transport geometry ($\phi$ norms)')
    ax.legend(loc='best', framealpha=0.9)

    # --- (f) Attention entropy + head diversity ---
    ax = fig.add_subplot(gs[2, 1])
    for col, label, color in [
        ('attn_entropy_per_head_mean', r'$\bar{H}(\beta_h)$', '#2E7D32'),
        ('attention_entropy', r'$H(\beta)$ (global)', '#81C784'),
    ]:
        steps, vals = _get_valid(data, col)
        if len(vals) > 0:
            ax.plot(steps, _smooth(vals, smooth_window), label=label, color=color, linewidth=1.5)
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.5)
    ax2 = ax.twinx()
    steps, vals = _get_valid(data, 'head_correlation_mean')
    if len(vals) > 0:
        ax2.plot(steps, _smooth(vals, smooth_window), label=r'$\bar{r}_{hh}$',
                 color='#FF6F00', linewidth=1.5, linestyle=':')
        ax2.set_ylabel(r'Head correlation', color='#FF6F00')
        ax2.tick_params(axis='y', labelcolor='#FF6F00')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='best', framealpha=0.9, fontsize=7)
    else:
        ax.legend(loc='best', framealpha=0.9)
    ax.set_ylabel(r'Attention entropy $H(\beta)$')
    ax.set_xlabel('Training step')
    ax.set_title('(f) Attention entropy & head diversity')

    fig.suptitle('VFE Dynamics Dashboard', fontsize=16, y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# Convenience: Generate all figures from a metrics CSV
# =============================================================================

def generate_all_vfe_figures(
    csv_path: Path,
    output_dir: Optional[Path] = None,
    smooth_window: int = 15,
) -> Dict[str, Path]:
    r"""Generate all VFE dynamics figures from a single metrics CSV.

    Args:
        csv_path: Path to metrics.csv from train_publication.py
        output_dir: Directory for output figures (default: same dir as CSV)
        smooth_window: Savitzky-Golay window for smoothing

    Returns:
        Dict mapping figure name to saved file path.
    """
    if output_dir is None:
        output_dir = csv_path.parent / 'vfe_dynamics_figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_vfe_metrics(csv_path)
    saved = {}

    figures = [
        ('vfe_gradient_decomposition', plot_vfe_gradient_decomposition),
        ('covariance_health', plot_covariance_health),
        ('transport_attention', plot_transport_attention),
        ('kl_landscape', plot_kl_landscape),
        ('vfe_dynamics_dashboard', plot_vfe_dynamics_dashboard),
    ]

    for name, plot_fn in figures:
        path = output_dir / f'{name}.png'
        try:
            fig = plot_fn(data, save_path=path, smooth_window=smooth_window)
            if fig is not None:
                plt.close(fig)
                saved[name] = path
                print(f"  Saved: {path}")
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")

    # Also save PDF versions for publication
    for name, plot_fn in figures:
        pdf_path = output_dir / f'{name}.pdf'
        try:
            fig = plot_fn(data, save_path=pdf_path, smooth_window=smooth_window)
            if fig is not None:
                plt.close(fig)
                saved[f'{name}_pdf'] = pdf_path
        except Exception:
            pass

    return saved


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='VFE Dynamics Visualization')
    parser.add_argument('--file', type=str, required=True, help='Path to metrics.csv')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'gradient', 'covariance', 'transport', 'kl', 'dashboard'],
                        help='Which figure to generate')
    parser.add_argument('--smooth', type=int, default=15, help='Smoothing window')
    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        exit(1)

    output_dir = Path(args.output) if args.output else csv_path.parent / 'vfe_dynamics_figures'

    if args.mode == 'all':
        saved = generate_all_vfe_figures(csv_path, output_dir, args.smooth)
        print(f"\nGenerated {len(saved)} figures in {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        data = load_vfe_metrics(csv_path)
        mode_map = {
            'gradient': ('vfe_gradient_decomposition', plot_vfe_gradient_decomposition),
            'covariance': ('covariance_health', plot_covariance_health),
            'transport': ('transport_attention', plot_transport_attention),
            'kl': ('kl_landscape', plot_kl_landscape),
            'dashboard': ('vfe_dynamics_dashboard', plot_vfe_dynamics_dashboard),
        }
        name, fn = mode_map[args.mode]
        path = output_dir / f'{name}.png'
        fig = fn(data, save_path=path, smooth_window=args.smooth)
        if fig is not None:
            plt.close(fig)
            print(f"Saved: {path}")
