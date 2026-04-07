# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 19:40:08 2025

@author: chris and christine
"""

"""
Unified Training Visualization for Gauge-Theoretic Transformer
================================================================

Plots VFE training metrics including total loss, free energy components
(belief alignment, self-consistency, model coupling), BPC, gradient norms
for mu/sigma/phi parameter groups, and learning rate schedules.

Modes:
    basic: Quick 3x2 grid of core training metrics.
    pub:   Publication-quality 2x2 panel with training dynamics.
    paper: Compact 2x2 panel + LaTeX table from ablation study directories.

Usage:
    python -m transformer.visualization.training_plots
    python -m transformer.visualization.training_plots --mode pub --file metrics.csv
    python -m transformer.visualization.training_plots --mode paper --ablation_dir checkpoints_publication
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import rcParams
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - install with: pip install matplotlib")

try:
    import seaborn as sns
    from scipy.signal import savgol_filter
    from scipy import stats
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from transformer.visualization.pub_style import set_pub_style, PUB_COLORS


# =============================================================================
# CSV Loading (Shared)
# =============================================================================

def load_metrics_csv(csv_path: Path) -> Dict:
    """Load training metrics from CSV file.

    Args:
        csv_path: Path to a metrics CSV with columns like step, train_loss_total,
            val_loss, train_bpc, val_bpc, val_ppl, grad_norm_mu, etc.

    Returns:
        Dict mapping column names to lists of parsed values (float or None).
    """
    metrics = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse step
            if 'step' in row:
                metrics['steps'].append(int(row['step']))

            # Parse all numeric fields
            for key, value in row.items():
                if key == 'step':
                    continue

                # Try to parse as float
                try:
                    if value and value not in ['', 'None', 'nan']:
                        metrics[key].append(float(value))
                    else:
                        metrics[key].append(None)
                except (ValueError, TypeError):
                    metrics[key].append(None)

    return dict(metrics)


def find_most_recent_metrics() -> Optional[Path]:
    """Auto-detect most recent metrics file."""
    search_paths = [
        Path('checkpoints_publication'),
        Path('checkpoints'),
        Path('.'),
    ]

    all_metrics = []
    for base_path in search_paths:
        if base_path.exists():
            all_metrics.extend(base_path.rglob('metrics.csv'))

    if not all_metrics:
        return None

    # Return most recently modified
    return max(all_metrics, key=lambda p: p.stat().st_mtime)


# =============================================================================
# Basic Mode - Quick 3x2 Grid
# =============================================================================

def create_basic_plots(metrics: Dict, output_path: Path, start_step: int = 5):
    """Create basic 3x2 grid visualization.

    Args:
        metrics: Dictionary of training metrics
        output_path: Path to save the figure
        start_step: Skip initial steps to avoid transient spikes (default: 5)
    """
    set_pub_style()
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Cannot create plots - matplotlib not installed")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle('Training History', fontsize=16)

    # Filter metrics to skip initial transient
    all_steps = metrics.get('steps', [])
    start_idx = next((i for i, s in enumerate(all_steps) if s >= start_step), 0)
    steps = all_steps[start_idx:]

    # Create filtered metrics dict for this plot
    filtered_metrics = {}
    for key, values in metrics.items():
        if key == 'steps':
            filtered_metrics[key] = steps
        elif isinstance(values, list) and len(values) == len(all_steps):
            filtered_metrics[key] = values[start_idx:]
        else:
            filtered_metrics[key] = values

    def filter_vals(vals):
        return [(s, v) for s, v in zip(steps, vals) if v is not None]

    # Use filtered metrics for all subsequent lookups
    m = filtered_metrics

    # 1. Loss curves
    ax = axes[0, 0]
    train_loss = m.get('train_loss_total', m.get('train_loss', []))
    val_loss = m.get('val_loss', [])

    ax.plot(steps, train_loss, label='Train Loss', alpha=0.7)

    val_data = filter_vals(val_loss)
    if val_data:
        val_steps, val_values = zip(*val_data)
        ax.plot(val_steps, val_values, 'o-', label='Val Loss', linewidth=2)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Free Energy Components
    ax = axes[0, 1]
    belief_align = m.get('train_loss_belief_align', [])
    self_consistency = m.get('train_loss_self_consistency', [])
    model_align = m.get('train_loss_model_coupling', [])

    belief_data = filter_vals(belief_align)
    self_data = filter_vals(self_consistency)
    model_data = filter_vals(model_align)

    if belief_data:
        b_steps, b_vals = zip(*belief_data)
        ax.plot(b_steps, b_vals, label='β (Belief Align)', alpha=0.7)
    if self_data:
        s_steps, s_vals = zip(*self_data)
        ax.plot(s_steps, s_vals, label='α (Self-Consistency)', alpha=0.7)
    if model_data:
        m_steps, m_vals = zip(*model_data)
        ax.plot(m_steps, m_vals, label='γ (Model Align)', alpha=0.7)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss Component')
    ax.set_title('Free Energy Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. BPC
    ax = axes[1, 0]
    train_bpc = m.get('train_bpc', [])
    val_bpc = m.get('val_bpc', [])

    train_bpc_data = filter_vals(train_bpc)
    val_bpc_data = filter_vals(val_bpc)

    if train_bpc_data:
        t_steps, t_vals = zip(*train_bpc_data)
        ax.plot(t_steps, t_vals, label='Train BPC', alpha=0.7)
    if val_bpc_data:
        v_steps, v_vals = zip(*val_bpc_data)
        ax.plot(v_steps, v_vals, 'o-', label='Val BPC', linewidth=2)

    ax.set_xlabel('Step')
    ax.set_ylabel('Bits-Per-Character')
    ax.set_title('Bits-Per-Character')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Gradient Norms
    ax = axes[1, 1]
    grad_groups = [
        ('grad_norm_total', 'Total', 'k-', 0.8),
        ('grad_norm_mu', 'μ', 'b--', 0.6),
        ('grad_norm_sigma', 'Σ', 'g--', 0.6),
        ('grad_norm_phi', 'φ', 'r--', 0.6),
        ('grad_norm_other', 'Other', 'c:', 0.5),
    ]
    for key, label, style, alpha in grad_groups:
        data = filter_vals(m.get(key, []))
        if data:
            steps_g, vals_g = zip(*data)
            ax.plot(steps_g, vals_g, style, label=label, alpha=alpha)

    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 5. Learning Rates
    ax = axes[2, 0]
    mu_lr = m.get('mu_lr', [])
    sigma_lr = m.get('sigma_lr', [])
    phi_lr = m.get('phi_lr', [])
    ffn_lr = m.get('ffn_lr', [])

    mu_lr_data = filter_vals(mu_lr)
    sigma_lr_data = filter_vals(sigma_lr)
    phi_lr_data = filter_vals(phi_lr)
    ffn_lr_data = filter_vals(ffn_lr)

    if mu_lr_data:
        m_steps, m_vals = zip(*mu_lr_data)
        ax.plot(m_steps, m_vals, label='μ LR', alpha=0.7)
    if sigma_lr_data:
        s_steps, s_vals = zip(*sigma_lr_data)
        ax.plot(s_steps, s_vals, label='σ LR', alpha=0.7)
    if phi_lr_data:
        p_steps, p_vals = zip(*phi_lr_data)
        ax.plot(p_steps, p_vals, label='φ LR', alpha=0.7)
    if ffn_lr_data:
        f_steps, f_vals = zip(*ffn_lr_data)
        ax.plot(f_steps, f_vals, label='FFN LR', alpha=0.7)

    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedules')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Performance
    ax = axes[2, 1]
    tokens_per_sec = m.get('tokens_per_sec', [])

    tokens_data = filter_vals(tokens_per_sec)
    if tokens_data:
        t_steps, t_vals = zip(*tokens_data)
        ax.plot(t_steps, t_vals, alpha=0.5, label='Tokens/sec')

        # Add moving average
        window = min(10, len(t_vals) // 10)
        if window > 1:
            moving_avg = []
            for i in range(len(t_vals)):
                start = max(0, i - window + 1)
                moving_avg.append(sum(t_vals[start:i+1]) / (i - start + 1))
            ax.plot(t_steps, moving_avg, linewidth=2, label=f'{window}-step MA')

        ax.set_xlabel('Step')
        ax.set_ylabel('Tokens/second')
        ax.set_title('Training Throughput')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Basic plots saved to: {output_path}")
    plt.close()


def plot_head_kappas(metrics: Dict, output_path: Path, start_step: int = 5):
    """Plot per-head learned temperatures over training.

    Creates a single figure with:
    - Individual κ_h traces per head (colored lines)
    - Shaded min/max range
    - Mean κ as bold line

    Args:
        metrics: Dictionary of training metrics (from CSV or live)
        output_path: Path to save the figure
        start_step: Skip initial steps to avoid transient spikes
    """
    set_pub_style()
    if not MATPLOTLIB_AVAILABLE:
        return

    all_steps = metrics.get('steps', [])
    start_idx = next((i for i, s in enumerate(all_steps) if s >= start_step), 0)
    steps = all_steps[start_idx:]

    # Collect per-head kappa traces: look for kappa/head_0, kappa/head_1, ...
    head_traces = {}
    for key, values in metrics.items():
        if key.startswith('kappa_head_') or key.startswith('kappa/head_'):
            # Extract head index
            h_idx = key.split('_')[-1] if 'kappa_head_' in key else key.split('head_')[-1]
            try:
                h_idx = int(h_idx)
            except ValueError:
                continue
            vals = values[start_idx:] if isinstance(values, list) and len(values) == len(all_steps) else values
            head_traces[h_idx] = vals

    # Also check summary stats
    kappa_mean = metrics.get('kappa_mean', metrics.get('kappa/per_head_mean', []))
    kappa_min = metrics.get('kappa_min', metrics.get('kappa/per_head_min', []))
    kappa_max = metrics.get('kappa_max', metrics.get('kappa/per_head_max', []))

    if isinstance(kappa_mean, list) and len(kappa_mean) == len(all_steps):
        kappa_mean = kappa_mean[start_idx:]
    if isinstance(kappa_min, list) and len(kappa_min) == len(all_steps):
        kappa_min = kappa_min[start_idx:]
    if isinstance(kappa_max, list) and len(kappa_max) == len(all_steps):
        kappa_max = kappa_max[start_idx:]

    if not head_traces and not kappa_mean:
        return  # No kappa data to plot

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = PUB_COLORS if len(head_traces) <= len(PUB_COLORS) else plt.cm.tab20.colors

    # Plot individual head traces
    for h_idx in sorted(head_traces.keys()):
        vals = head_traces[h_idx]
        valid = [(s, v) for s, v in zip(steps, vals) if v is not None]
        if valid:
            h_steps, h_vals = zip(*valid)
            ax.plot(h_steps, h_vals, alpha=0.7, linewidth=1.2,
                    color=colors[h_idx % len(colors)],
                    label=f'Head {h_idx}')

    # Plot mean as bold line
    if kappa_mean:
        valid_mean = [(s, v) for s, v in zip(steps, kappa_mean) if v is not None]
        if valid_mean:
            m_steps, m_vals = zip(*valid_mean)
            ax.plot(m_steps, m_vals, 'k-', linewidth=2.5, alpha=0.9, label='Mean')

    # Shaded min/max range
    if kappa_min and kappa_max:
        valid_range = [(s, lo, hi) for s, lo, hi in zip(steps, kappa_min, kappa_max)
                       if lo is not None and hi is not None]
        if valid_range:
            r_steps, r_lo, r_hi = zip(*valid_range)
            ax.fill_between(r_steps, r_lo, r_hi, alpha=0.15, color='gray')

    ax.set_xlabel('Step')
    ax.set_ylabel(r'$\kappa_h$')
    ax.set_title('Per-Head Learned Temperature')
    ax.legend(fontsize=8, ncol=min(4, max(1, len(head_traces) + 1)))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[plot] Per-head kappa saved to: {output_path}")


def print_basic_summary(metrics: Dict):
    """Print text summary of training metrics (loss, PPL, BPC, timing)."""
    print("="*70)
    print("TRAINING SUMMARY")
    print("="*70)

    steps = metrics.get('steps', [])
    train_loss = [x for x in metrics.get('train_loss_total', metrics.get('train_loss', [])) if x is not None]
    val_loss = [x for x in metrics.get('val_loss', []) if x is not None]
    val_ppl = [x for x in metrics.get('val_ppl', metrics.get('val_perplexity', [])) if x is not None]
    val_bpc = [x for x in metrics.get('val_bpc', []) if x is not None]
    step_times = [x for x in metrics.get('step_time', []) if x is not None]

    print(f"Total steps: {len(steps)}")
    print(f"Final step: {steps[-1] if steps else 0}")
    print()

    if train_loss:
        print(f"Training Loss:")
        print(f"  Initial: {train_loss[0]:.4f}")
        print(f"  Final:   {train_loss[-1]:.4f}")
        print(f"  Best:    {min(train_loss):.4f}")
        print()

    if val_loss:
        print(f"Validation Loss:")
        print(f"  Best: {min(val_loss):.4f}")
        print()

    if val_ppl:
        print(f"Validation Perplexity:")
        print(f"  Best: {min(val_ppl):.2f}")
        print()

    if val_bpc:
        print(f"Validation BPC:")
        print(f"  Best: {min(val_bpc):.3f}")
        print()

    if step_times:
        avg_time = sum(step_times) / len(step_times)
        print(f"Step Time:")
        print(f"  Average: {avg_time:.2f} seconds")
        print(f"  Total: {sum(step_times)/60:.1f} minutes")
        print()

    print("="*70)


# =============================================================================
# Publication Mode - Comprehensive Figures (Simplified)
# =============================================================================

def create_publication_figures(metrics: Dict, output_dir: Path, start_step: int = 5):
    """Create publication-quality figures (simplified from plot_training_history_pub.py).

    Args:
        metrics: Dictionary of training metrics
        output_dir: Directory to save figures
        start_step: Skip initial steps to avoid transient spikes (default: 5)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Cannot create figures - matplotlib not installed")
        return

    # Setup publication style
    set_pub_style()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter metrics to skip initial transient
    all_steps = metrics.get('steps', [])
    start_idx = next((i for i, s in enumerate(all_steps) if s >= start_step), 0)
    steps = all_steps[start_idx:]

    # Create filtered metrics dict
    m = {}
    for key, values in metrics.items():
        if key == 'steps':
            m[key] = steps
        elif isinstance(values, list) and len(values) == len(all_steps):
            m[key] = values[start_idx:]
        else:
            m[key] = values

    # Figure 1: Training Dynamics (2x2 panel)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Loss
    ax = axes[0, 0]
    train_loss = m.get('train_loss_total', m.get('train_loss', []))
    val_loss = m.get('val_loss', [])
    ax.plot(steps, train_loss, label='Train', alpha=0.7)
    val_mask = [v is not None for v in val_loss]
    if any(val_mask):
        val_steps = [s for s, mask in zip(steps, val_mask) if mask]
        val_vals = [v for v, mask in zip(val_loss, val_mask) if mask]
        ax.scatter(val_steps, val_vals, label='Val', s=30, alpha=0.8)
    ax.set_title('(a) Total Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # BPC
    ax = axes[0, 1]
    train_bpc = m.get('train_bpc', [])
    val_bpc = m.get('val_bpc', [])
    train_bpc_clean = [(s, v) for s, v in zip(steps, train_bpc) if v is not None]
    val_bpc_clean = [(s, v) for s, v in zip(steps, val_bpc) if v is not None]
    if train_bpc_clean:
        t_steps, t_vals = zip(*train_bpc_clean)
        ax.plot(t_steps, t_vals, label='Train', alpha=0.7)
    if val_bpc_clean:
        v_steps, v_vals = zip(*val_bpc_clean)
        ax.scatter(v_steps, v_vals, label='Val', s=30, alpha=0.8)
    ax.set_title('(b) Bits-Per-Character')
    ax.set_xlabel('Step')
    ax.set_ylabel('BPC')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Perplexity
    ax = axes[1, 0]
    val_ppl = m.get('val_ppl', m.get('val_perplexity', []))
    val_ppl_clean = [(s, v) for s, v in zip(steps, val_ppl) if v is not None and v > 0]
    if val_ppl_clean:
        v_steps, v_vals = zip(*val_ppl_clean)
        ax.semilogy(v_steps, v_vals, 'o-', alpha=0.8)
    ax.set_title('(c) Validation Perplexity')
    ax.set_xlabel('Step')
    ax.set_ylabel('Perplexity (log)')
    ax.grid(True, alpha=0.3)

    # Free Energy Components
    ax = axes[1, 1]
    belief = m.get('train_loss_belief_align', [])
    alpha_loss = m.get('train_loss_self_consistency', [])
    gamma = m.get('train_loss_model_coupling', [])

    belief_clean = [(s, v) for s, v in zip(steps, belief) if v is not None and v > 1e-6]
    alpha_clean = [(s, v) for s, v in zip(steps, alpha_loss) if v is not None and v > 1e-6]
    gamma_clean = [(s, v) for s, v in zip(steps, gamma) if v is not None and v > 1e-6]

    if belief_clean:
        b_steps, b_vals = zip(*belief_clean)
        ax.plot(b_steps, b_vals, label='β', alpha=0.7)
    if alpha_clean:
        a_steps, a_vals = zip(*alpha_clean)
        ax.plot(a_steps, a_vals, label='α', alpha=0.7)
    if gamma_clean:
        g_steps, g_vals = zip(*gamma_clean)
        ax.plot(g_steps, g_vals, label='γ', alpha=0.7)

    ax.set_title('(d) Free Energy Components')
    ax.set_xlabel('Step')
    ax.set_ylabel('Component Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    fig_path = output_dir / 'fig1_training_dynamics.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path}")
    plt.close()

    print(f"\n📊 Publication figures saved to: {output_dir}/")


# =============================================================================
# Paper Mode - Compact 2x2 + LaTeX Table
# =============================================================================

def create_paper_outputs(ablation_dir: Path, output_dir: Path):
    """Create paper-ready 2x2 panel and LaTeX table from ablation study.

    Scans ablation_dir for ffn_* subdirectories, each containing a
    metrics.csv. Generates a 4-panel figure (val BPC, training loss,
    val PPL, free energy components) and a LaTeX results table.

    Args:
        ablation_dir: Directory containing ffn_<mode>/ subdirectories.
        output_dir: Directory to write PDF, PNG, LaTeX, and JSON outputs.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Cannot create figures - matplotlib not installed")
        return

    set_pub_style()

    # Find all FFN mode metrics
    ffn_modes = {}
    for subdir in ablation_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith('ffn_'):
            metrics_file = subdir / 'metrics.csv'
            if metrics_file.exists():
                mode = subdir.name.replace('ffn_', '')
                ffn_modes[mode] = load_metrics_csv(metrics_file)
                print(f"  ✓ Loaded {mode}")

    if len(ffn_modes) == 0:
        print("❌ No FFN mode metrics found in ablation directory")
        return

    # Create 2x2 panel figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    mode_styles = {
        'learned': {'color': '#2E86AB', 'linestyle': '-', 'marker': 'o'},
        'variational_approx': {'color': '#A23B72', 'linestyle': '--', 'marker': 's'},
        'variational_full': {'color': '#F18F01', 'linestyle': '-.', 'marker': '^'},
        'variational_gradient_engine': {'color': '#C73E1D', 'linestyle': '-', 'marker': 'D'},
    }

    # Panel A: Validation BPC
    ax = axes[0, 0]
    for mode, metrics in ffn_modes.items():
        style = mode_styles.get(mode, {})
        steps = metrics.get('steps', [])
        val_bpc = metrics.get('val_bpc', [])
        valid_data = [(s, v) for s, v in zip(steps, val_bpc) if v is not None]
        if valid_data:
            val_steps, val_values = zip(*valid_data)
            ax.plot(val_steps, val_values, label=mode.replace('_', ' ').title(),
                   color=style.get('color', 'gray'), linestyle=style.get('linestyle', '-'),
                   marker=style.get('marker', 'o'), markevery=max(1, len(val_steps)//10))
    ax.set_title('A) Validation Performance', fontweight='bold', loc='left')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Bits Per Character')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Training Loss
    ax = axes[0, 1]
    for mode, metrics in ffn_modes.items():
        style = mode_styles.get(mode, {})
        steps = metrics.get('steps', [])
        train_loss = metrics.get('train_loss_total', [])
        valid_data = [(s, v) for s, v in zip(steps, train_loss) if v is not None]
        if valid_data:
            train_steps, train_values = zip(*valid_data)
            ax.plot(train_steps, train_values, label=mode.replace('_', ' ').title(),
                   color=style.get('color', 'gray'), linestyle=style.get('linestyle', '-'))
    ax.set_title('B) Training Convergence', fontweight='bold', loc='left')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Total Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Validation Perplexity
    ax = axes[1, 0]
    for mode, metrics in ffn_modes.items():
        style = mode_styles.get(mode, {})
        steps = metrics.get('steps', [])
        val_ppl = metrics.get('val_ppl', metrics.get('val_perplexity', []))
        valid_data = [(s, v) for s, v in zip(steps, val_ppl) if v is not None]
        if valid_data:
            val_steps, val_values = zip(*valid_data)
            ax.plot(val_steps, val_values, label=mode.replace('_', ' ').title(),
                   color=style.get('color', 'gray'), linestyle=style.get('linestyle', '-'),
                   marker=style.get('marker', 'o'), markevery=max(1, len(val_steps)//10))
    ax.set_title('C) Validation Perplexity', fontweight='bold', loc='left')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Perplexity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: Free Energy (for first variational mode found)
    ax = axes[1, 1]
    demo_mode = None
    for mode in ['variational_gradient_engine', 'variational_full', 'variational_approx']:
        if mode in ffn_modes:
            demo_mode = mode
            break

    if demo_mode:
        metrics = ffn_modes[demo_mode]
        steps = metrics.get('steps', [])
        ce = metrics.get('train_loss_ce', [])
        belief = metrics.get('train_loss_belief_align', [])
        alpha_loss = metrics.get('train_loss_self_consistency', [])

        ce_clean = [(s, v) for s, v in zip(steps, ce) if v is not None]
        belief_clean = [(s, v) for s, v in zip(steps, belief) if v is not None and v > 1e-6]
        alpha_clean = [(s, v) for s, v in zip(steps, alpha_loss) if v is not None and v > 1e-6]

        if ce_clean:
            c_steps, c_vals = zip(*ce_clean)
            ax.plot(c_steps, c_vals, label='CE', alpha=0.7)
        if belief_clean:
            b_steps, b_vals = zip(*belief_clean)
            ax.plot(b_steps, b_vals, label='β', alpha=0.7)
        if alpha_clean:
            a_steps, a_vals = zip(*alpha_clean)
            ax.plot(a_steps, a_vals, label='α', alpha=0.7)

        ax.set_title(f'D) Free Energy: {demo_mode.replace("_", " ").title()}', fontweight='bold', loc='left')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss Component')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / 'publication_figures.pdf'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\n📊 Paper figure saved: {fig_path}")

    png_path = output_dir / 'publication_figures.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
    print(f"📊 PNG version saved: {png_path}")
    plt.close()

    # Create LaTeX table
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Performance comparison of FFN modes.}")
    latex.append(r"\label{tab:results}")
    latex.append(r"\begin{tabular}{lccc}")
    latex.append(r"\hline")
    latex.append(r"FFN Mode & Val BPC $\downarrow$ & Val PPL $\downarrow$ & $\beta$ (final) \\")
    latex.append(r"\hline")

    for mode in ['learned', 'variational_approx', 'variational_full', 'variational_gradient_engine']:
        if mode not in ffn_modes:
            continue

        metrics = ffn_modes[mode]

        # Get best values
        val_bpc = [v for v in metrics.get('val_bpc', []) if v is not None]
        val_ppl = [v for v in metrics.get('val_ppl', []) if v is not None]
        belief_align = [v for v in metrics.get('train_loss_belief_align', []) if v is not None]

        bpc_str = f"{min(val_bpc):.3f}" if val_bpc else "---"
        ppl_str = f"{min(val_ppl):.2f}" if val_ppl else "---"
        beta_str = f"{belief_align[-1]:.4f}" if belief_align else "0.0000"

        mode_name = mode.replace('_', ' ').title()
        latex.append(f"{mode_name} & {bpc_str} & {ppl_str} & {beta_str} \\\\")

    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    table_path = output_dir / 'results_table.tex'
    with open(table_path, 'w') as f:
        f.write('\n'.join(latex))

    print(f"📄 LaTeX table saved: {table_path}")

    # Create JSON summary
    summary = {}
    for mode, metrics in ffn_modes.items():
        val_bpc = [v for v in metrics.get('val_bpc', []) if v is not None]
        val_ppl = [v for v in metrics.get('val_ppl', []) if v is not None]

        summary[mode] = {
            'best_val_bpc': min(val_bpc) if val_bpc else None,
            'best_val_ppl': min(val_ppl) if val_ppl else None,
        }

    json_path = output_dir / 'publication_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"📊 JSON summary saved: {json_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified Training Visualization Tool')
    parser.add_argument('--mode', type=str, default='basic',
                       choices=['basic', 'pub', 'paper'],
                       help='Visualization mode: basic (quick plots), pub (comprehensive), paper (2x2+LaTeX)')
    parser.add_argument('--file', type=str, default=None,
                       help='Path to metrics.csv (auto-detects if not provided)')
    parser.add_argument('--ablation_dir', type=str, default=None,
                       help='Directory with FFN mode subdirectories (for paper mode)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file/directory path')
    parser.add_argument('--start-step', type=int, default=5,
                       help='Skip initial steps to avoid transient spikes (default: 5)')

    args = parser.parse_args()

    print("="*70)
    print("UNIFIED TRAINING VISUALIZATION TOOL")
    print("="*70)
    print(f"\nMode: {args.mode}")

    # Handle paper mode separately (requires ablation directory)
    if args.mode == 'paper':
        if args.ablation_dir is None:
            print("\n❌ Paper mode requires --ablation_dir")
            print("   Example: python plot_training.py --mode paper --ablation_dir checkpoints_publication")
            return

        ablation_dir = Path(args.ablation_dir)
        if not ablation_dir.exists():
            print(f"\n❌ Ablation directory not found: {ablation_dir}")
            return

        output_dir = Path(args.output) if args.output else Path('.')
        create_paper_outputs(ablation_dir, output_dir)
        return

    # Determine input file
    if args.file:
        input_path = Path(args.file)
        if not input_path.exists():
            print(f"\n❌ File not found: {input_path}")
            return
    else:
        print("\n🔍 Auto-detecting most recent metrics.csv...")
        input_path = find_most_recent_metrics()
        if input_path is None:
            print("❌ No metrics.csv files found")
            print("   Run training first or specify path with --file")
            return
        print(f"✓ Found: {input_path}")

    # Load metrics
    print(f"\n📊 Loading metrics from: {input_path}")
    metrics = load_metrics_csv(input_path)
    print(f"   Loaded {len(metrics.get('steps', []))} steps")

    # Create visualizations
    start_step = args.start_step
    print(f"   Skipping first {start_step} steps (use --start-step to change)")

    if args.mode == 'basic':
        output_path = Path(args.output) if args.output else input_path.parent / 'training_plots.png'
        print_basic_summary(metrics)
        create_basic_plots(metrics, output_path, start_step=start_step)

    elif args.mode == 'pub':
        output_dir = Path(args.output) if args.output else input_path.parent / 'figures'
        create_publication_figures(metrics, output_dir, start_step=start_step)

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()