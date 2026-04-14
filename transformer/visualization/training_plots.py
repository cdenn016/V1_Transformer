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
"""

import csv
from pathlib import Path
from typing import Dict
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("  matplotlib not available - install with: pip install matplotlib")

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

    _pub_color_list = list(PUB_COLORS.values())
    colors = _pub_color_list if len(head_traces) <= len(_pub_color_list) else plt.cm.tab20.colors

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
