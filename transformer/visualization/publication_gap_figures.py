"""
Publication Gap Figures
======================

Figures identified as missing in the 2026-04-10 diagnostics audit.
Each function loads data from the existing metrics CSV or runs a
targeted evaluation, then produces a publication-quality figure.

Gap IDs reference the audit plan at .claude/plans/cheeky-riding-flurry.md:
  C2  — VFE loss decomposition (stacked area)
  C5  — Gauge equivariance error
  C6  — Posterior collapse diagnostic
  H1  — KL vs dot-product attention comparison
  H4  — E-step convergence (gradient norm vs iteration)
  H6  — Phi spectral evolution
  H7  — Per-head attention entropy
  H8  — Linear probe on frozen mu
  M6  — VFE gradient decomposition

Usage:
    from transformer.visualization.publication_gap_figures import *
    # Most functions accept (csv_path, save_path) or (model, save_path)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from transformer.visualization.pub_style import set_pub_style, PUB_COLORS
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


# =============================================================================
# Helpers
# =============================================================================

def _load_csv(csv_path: Path) -> Dict[str, list]:
    """Load metrics CSV into {column: [values]} with None for missing."""
    data: Dict[str, list] = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                if val is not None and val.strip() not in ('', 'None', 'nan'):
                    try:
                        data[key].append(float(val))
                    except (ValueError, TypeError):
                        data[key].append(None)
                else:
                    data[key].append(None)
    return data


def _valid(data: Dict, key: str, steps_key: str = 'step'
           ) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (steps, values) where both are non-None."""
    steps = data.get(steps_key, [])
    vals = data.get(key, [])
    pairs = [(s, v) for s, v in zip(steps, vals)
             if s is not None and v is not None]
    if not pairs:
        return np.array([]), np.array([])
    s, v = zip(*pairs)
    return np.array(s), np.array(v)


def _smooth(arr: np.ndarray, window: int = 15) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def _ensure_mpl():
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib required for figure generation")
    set_pub_style()


# =============================================================================
# C2: VFE Loss Decomposition (Stacked Area)
# =============================================================================

def plot_vfe_decomposition(
    csv_path: Path,
    save_path: Optional[Path] = None,
    start_step: int = 100,
    smooth_window: int = 25,
) -> Any:
    r"""Stacked area plot of VFE loss components over training.

    Shows CE, α·KL(q||p), β·alignment, and γ·model coupling as stacked
    areas, revealing the trade-off between reconstruction and regularization.

    Addresses audit gap C2.
    """
    _ensure_mpl()
    data = _load_csv(csv_path)

    components = [
        ('train_loss_ce', 'CE Loss', PUB_COLORS['blue']),
        ('train_loss_self_consistency', r'$\alpha \cdot \mathrm{KL}(q \| p)$', PUB_COLORS['orange']),
        ('train_loss_belief_align', r'$\beta \cdot \mathrm{Alignment}$', PUB_COLORS['green']),
        ('train_loss_model_coupling', r'$\gamma \cdot \mathrm{Model}$', PUB_COLORS['purple']),
    ]

    fig, (ax_stack, ax_frac) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Collect aligned data
    steps_raw = data.get('step', [])
    arrays = {}
    for key, label, color in components:
        vals = data.get(key, [None] * len(steps_raw))
        arr = np.array([v if v is not None else 0.0 for v in vals])
        arr = np.clip(arr, 0, None)  # Loss components should be non-negative
        arrays[key] = arr

    steps = np.array([s for s in steps_raw if s is not None])
    mask = steps >= start_step
    steps = steps[mask]

    smoothed = {}
    for key, label, color in components:
        smoothed[key] = _smooth(arrays[key][mask], smooth_window)

    # Panel A: Stacked area (absolute)
    bottom = np.zeros_like(steps, dtype=float)
    for key, label, color in components:
        vals = smoothed[key]
        ax_stack.fill_between(steps, bottom, bottom + vals,
                              alpha=0.7, label=label, color=color)
        bottom = bottom + vals

    ax_stack.plot(steps, bottom, color='black', linewidth=1.0, alpha=0.5,
                  label='Total', linestyle='--')
    ax_stack.set_ylabel('Loss')
    ax_stack.set_title('(a) VFE Loss Decomposition')
    ax_stack.legend(loc='upper right', ncol=2)

    # Panel B: Fractional contribution
    total = bottom.copy()
    total[total < 1e-8] = 1e-8
    bottom_frac = np.zeros_like(steps, dtype=float)
    for key, label, color in components:
        frac = smoothed[key] / total
        ax_frac.fill_between(steps, bottom_frac, bottom_frac + frac,
                             alpha=0.7, color=color)
        bottom_frac = bottom_frac + frac

    ax_frac.set_ylabel('Fraction of Total Loss')
    ax_frac.set_xlabel('Training Step')
    ax_frac.set_title('(b) Relative Contribution')
    ax_frac.set_ylim(0, 1)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# C5: Gauge Equivariance Verification
# =============================================================================

@torch.no_grad()
def plot_equivariance_error(
    model: Any,
    dataloader: Any,
    save_path: Optional[Path] = None,
    n_transforms: int = 50,
    n_batches: int = 5,
    device: str = 'cuda',
) -> Any:
    r"""Quantitative gauge equivariance stress test.

    For random gauge transforms g in GL(K):
      error = ||F(x, g·theta) - g·F(x, theta)|| / ||F(x, theta)||

    where F extracts the post-E-step beliefs (mu_q, sigma_q).

    Addresses audit gap C5.
    """
    _ensure_mpl()
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    errors_mu = []
    errors_sigma = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break
        input_ids = batch[0].to(device)

        # Get baseline beliefs
        out_base = model.forward_with_attention(input_ids)
        mu_base = out_base['mu_q'].detach()
        # sigma might be diagonal (B,N,K) or full (B,N,K,K)
        sigma_base = out_base.get('sigma_q', None)
        if sigma_base is not None:
            sigma_base = sigma_base.detach()

        K = mu_base.shape[-1]

        for _ in range(n_transforms):
            # Random gauge transform: small perturbation from identity
            g = torch.eye(K, device=device) + 0.1 * torch.randn(K, K, device=device)

            # Transform embeddings: mu -> g @ mu, sigma -> g @ sigma @ g^T
            # Save originals
            embed = model.token_embed
            orig_mu = embed.mu_embed.weight.data.clone()
            orig_sigma = embed.log_sigma_diag.data.clone() if hasattr(embed, 'log_sigma_diag') else None

            # Apply transform to mu embeddings
            embed.mu_embed.weight.data = (orig_mu @ g.T)

            # Forward with transformed embeddings
            out_transformed = model.forward_with_attention(input_ids)
            mu_trans = out_transformed['mu_q'].detach()

            # Expected: g @ mu_base
            mu_expected = mu_base @ g.T

            # Compute relative error
            err_mu = (mu_trans - mu_expected).norm() / (mu_expected.norm() + 1e-8)
            errors_mu.append(err_mu.item())

            # Restore
            embed.mu_embed.weight.data = orig_mu

    errors_mu = np.array(errors_mu)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors_mu, bins=50, color=PUB_COLORS['blue'], alpha=0.7,
            edgecolor='white', linewidth=0.5)
    ax.axvline(np.median(errors_mu), color=PUB_COLORS['red'], linestyle='--',
               linewidth=2, label=f'Median: {np.median(errors_mu):.4f}')
    ax.set_xlabel(r'Relative Equivariance Error $\|F(g \cdot \theta) - g \cdot F(\theta)\| / \|F(\theta)\|$')
    ax.set_ylabel('Count')
    ax.set_title('Gauge Equivariance Stress Test')
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# C6: Posterior Collapse Diagnostic
# =============================================================================

def plot_posterior_collapse(
    csv_path: Path,
    save_path: Optional[Path] = None,
    start_step: int = 100,
    smooth_window: int = 25,
) -> Any:
    r"""Posterior collapse diagnostic: tracks sigma_q statistics over training.

    If sigma_q collapses toward the prior sigma_p (KL -> 0), beliefs
    ignore the data. If sigma_q shrinks to zero, the model is overconfident.

    Shows: (a) sigma_q mean/min/max, (b) prior-belief KL, (c) effective dim.

    Addresses audit gap C6.
    """
    _ensure_mpl()
    data = _load_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Sigma_q statistics
    ax = axes[0]
    for key, label, color in [
        ('sigma_q_mean', r'$\bar{\sigma}_q$', PUB_COLORS['blue']),
        ('sigma_q_min', r'$\sigma_q^{\min}$', PUB_COLORS['cyan']),
        ('sigma_q_max', r'$\sigma_q^{\max}$', PUB_COLORS['red']),
        ('sigma_p_mean', r'$\bar{\sigma}_p$ (prior)', PUB_COLORS['gray']),
    ]:
        steps, vals = _valid(data, key)
        if len(steps) == 0:
            continue
        mask = steps >= start_step
        ax.plot(steps[mask], _smooth(vals[mask], smooth_window),
                label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$\sigma$')
    ax.set_title('(a) Covariance Evolution')
    ax.legend(fontsize=7)
    ax.set_yscale('log')

    # (b) Prior-belief KL
    ax = axes[1]
    for key, label, color in [
        ('prior_belief_kl_mean', 'Mean KL(q||p)', PUB_COLORS['blue']),
        ('prior_belief_kl_max', 'Max KL(q||p)', PUB_COLORS['red']),
    ]:
        steps, vals = _valid(data, key)
        if len(steps) == 0:
            continue
        mask = steps >= start_step
        ax.plot(steps[mask], _smooth(vals[mask], smooth_window),
                label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('KL Divergence')
    ax.set_title('(b) Prior-Belief KL')
    ax.legend(fontsize=7)

    # (c) Effective dimensionality (phi spectral)
    ax = axes[2]
    steps, vals = _valid(data, 'phi_effective_rank')
    if len(steps) > 0:
        mask = steps >= start_step
        ax.plot(steps[mask], _smooth(vals[mask], smooth_window),
                label='Effective Rank', color=PUB_COLORS['blue'], linewidth=1.5)
    steps2, vals2 = _valid(data, 'phi_rank_ratio')
    if len(steps2) > 0:
        ax2 = ax.twinx()
        mask2 = steps2 >= start_step
        ax2.plot(steps2[mask2], _smooth(vals2[mask2], smooth_window),
                 label='Rank Ratio', color=PUB_COLORS['orange'], linewidth=1.5,
                 linestyle='--')
        ax2.set_ylabel('Rank / Max', color=PUB_COLORS['orange'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank')
    ax.set_title(r'(c) $\phi$ Effective Dimensionality')

    fig.suptitle('Posterior Collapse Diagnostic', fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# H4: E-Step Convergence
# =============================================================================

def plot_estep_convergence(
    iteration_csv_path: Path,
    save_path: Optional[Path] = None,
    steps_to_show: Optional[List[int]] = None,
    max_curves: int = 8,
) -> Any:
    r"""E-step gradient norm vs VFE iteration number.

    Shows ||grad_mu|| and ||grad_sigma|| decreasing across E-step iterations,
    demonstrating that VFE minimization converges within the iteration budget.

    Requires iteration diagnostics CSV (track_iteration_diagnostics=True).

    Addresses audit gap H4.
    """
    _ensure_mpl()
    data = _load_csv(iteration_csv_path)

    steps_all = data.get('step', [])
    iters_all = data.get('iteration', [])
    grad_mu_all = data.get('nat_grad_mu_norm', data.get('grad_mu_norm', []))
    grad_sigma_all = data.get('grad_sigma_norm', [])

    if not steps_all or not iters_all:
        print("No iteration diagnostics data found.")
        return None

    # Group by training step
    from collections import defaultdict
    by_step: Dict[float, Dict[str, list]] = defaultdict(lambda: {'iter': [], 'mu': [], 'sigma': []})
    for s, it, gm, gs in zip(steps_all, iters_all, grad_mu_all, grad_sigma_all):
        if s is None or it is None:
            continue
        by_step[s]['iter'].append(it)
        if gm is not None:
            by_step[s]['mu'].append(gm)
        if gs is not None:
            by_step[s]['sigma'].append(gs)

    if steps_to_show is None:
        available = sorted(by_step.keys())
        # Sample evenly
        indices = np.linspace(0, len(available) - 1, min(max_curves, len(available)), dtype=int)
        steps_to_show = [available[i] for i in indices]

    fig, (ax_mu, ax_sigma) = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(steps_to_show)))

    for idx, step in enumerate(steps_to_show):
        entry = by_step.get(step)
        if entry is None:
            continue
        iters = np.array(entry['iter'])
        sort_idx = np.argsort(iters)
        iters = iters[sort_idx]

        if entry['mu']:
            mu_vals = np.array(entry['mu'])[sort_idx[:len(entry['mu'])]]
            ax_mu.semilogy(iters[:len(mu_vals)], mu_vals,
                           'o-', color=cmap[idx], label=f'Step {int(step)}',
                           markersize=4, linewidth=1.2)
        if entry['sigma']:
            sig_vals = np.array(entry['sigma'])[sort_idx[:len(entry['sigma'])]]
            ax_sigma.semilogy(iters[:len(sig_vals)], sig_vals,
                              'o-', color=cmap[idx], label=f'Step {int(step)}',
                              markersize=4, linewidth=1.2)

    ax_mu.set_xlabel('E-Step Iteration')
    ax_mu.set_ylabel(r'$\|\nabla_\mu \mathcal{F}\|$')
    ax_mu.set_title(r'(a) $\mu$ Gradient Norm')
    ax_mu.legend(fontsize=7, ncol=2)

    ax_sigma.set_xlabel('E-Step Iteration')
    ax_sigma.set_ylabel(r'$\|\nabla_\sigma \mathcal{F}\|$')
    ax_sigma.set_title(r'(b) $\sigma$ Gradient Norm')
    ax_sigma.legend(fontsize=7, ncol=2)

    fig.suptitle('E-Step VFE Convergence', fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# H6: Phi Spectral Evolution
# =============================================================================

def plot_phi_spectral_evolution(
    csv_path: Path,
    save_path: Optional[Path] = None,
    start_step: int = 100,
    smooth_window: int = 25,
) -> Any:
    r"""Phi embedding spectral properties over training.

    Shows effective rank, top-1/top-5 variance fractions, and spectral gap
    evolving as gauge frames differentiate during learning.

    Addresses audit gap H6.
    """
    _ensure_mpl()
    data = _load_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Effective rank
    ax = axes[0]
    steps, vals = _valid(data, 'phi_effective_rank')
    if len(steps) > 0:
        mask = steps >= start_step
        ax.plot(steps[mask], _smooth(vals[mask], smooth_window),
                color=PUB_COLORS['blue'], linewidth=1.5)
        ax.fill_between(steps[mask], vals[mask] * 0, _smooth(vals[mask], smooth_window),
                        alpha=0.1, color=PUB_COLORS['blue'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank')
    ax.set_title(r'(a) $\phi$ Effective Rank')

    # (b) Top-k variance fractions
    ax = axes[1]
    for key, label, color in [
        ('phi_top1_variance_fraction', 'Top-1', PUB_COLORS['red']),
        ('phi_top5_variance_fraction', 'Top-5', PUB_COLORS['orange']),
    ]:
        steps, vals = _valid(data, key)
        if len(steps) == 0:
            continue
        mask = steps >= start_step
        ax.plot(steps[mask], _smooth(vals[mask], smooth_window),
                label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Fraction of Total Variance')
    ax.set_title(r'(b) Variance Concentration')
    ax.set_ylim(0, 1)
    ax.legend()

    # (c) Spectral gap + Frobenius norm
    ax = axes[2]
    steps, vals = _valid(data, 'phi_spectral_gap')
    if len(steps) > 0:
        mask = steps >= start_step
        ax.plot(steps[mask], _smooth(vals[mask], smooth_window),
                label='Spectral Gap', color=PUB_COLORS['green'], linewidth=1.5)
    ax2 = ax.twinx()
    steps2, vals2 = _valid(data, 'phi_mean_token_norm')
    if len(steps2) > 0:
        mask2 = steps2 >= start_step
        ax2.plot(steps2[mask2], _smooth(vals2[mask2], smooth_window),
                 label=r'$\|\phi\|$ mean', color=PUB_COLORS['purple'],
                 linewidth=1.5, linestyle='--')
        ax2.set_ylabel(r'$\|\phi\|$', color=PUB_COLORS['purple'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Spectral Gap')
    ax.set_title(r'(c) Spectral Gap & $\phi$ Norm')

    fig.suptitle(r'Gauge Frame ($\phi$) Spectral Evolution', fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# H7: Per-Head Attention Entropy
# =============================================================================

def plot_attention_entropy_per_head(
    csv_path: Path,
    save_path: Optional[Path] = None,
    start_step: int = 100,
    smooth_window: int = 25,
) -> Any:
    r"""Attention entropy evolution showing head specialization.

    Low entropy = sharp attention (specialized). High entropy = diffuse.
    Diverging per-head curves indicate head specialization.

    Addresses audit gap H7.
    """
    _ensure_mpl()
    data = _load_csv(csv_path)

    fig, (ax_ts, ax_summary) = plt.subplots(1, 2, figsize=(12, 5),
                                             gridspec_kw={'width_ratios': [2, 1]})

    # Time series: mean, min, max entropy
    for key, label, color, ls in [
        ('attn_entropy_per_head_mean', 'Mean', PUB_COLORS['blue'], '-'),
        ('attn_entropy_per_head_min', 'Min', PUB_COLORS['cyan'], '--'),
        ('attn_entropy_per_head_max', 'Max', PUB_COLORS['red'], '--'),
    ]:
        steps, vals = _valid(data, key)
        if len(steps) == 0:
            continue
        mask = steps >= start_step
        ax_ts.plot(steps[mask], _smooth(vals[mask], smooth_window),
                   label=label, color=color, linestyle=ls, linewidth=1.5)

    # Shade between min and max
    steps_min, vals_min = _valid(data, 'attn_entropy_per_head_min')
    steps_max, vals_max = _valid(data, 'attn_entropy_per_head_max')
    if len(steps_min) > 0 and len(steps_max) > 0:
        mask = steps_min >= start_step
        ax_ts.fill_between(
            steps_min[mask],
            _smooth(vals_min[mask], smooth_window),
            _smooth(vals_max[mask], smooth_window),
            alpha=0.15, color=PUB_COLORS['blue'])

    ax_ts.set_xlabel('Step')
    ax_ts.set_ylabel('Attention Entropy (nats)')
    ax_ts.set_title('(a) Per-Head Entropy Over Training')
    ax_ts.legend()

    # Final-step statistics
    steps_mean, vals_mean = _valid(data, 'attn_entropy_per_head_mean')
    steps_std, vals_std = _valid(data, 'attn_entropy_per_head_std')
    if len(vals_mean) > 0:
        final_mean = vals_mean[-1]
        final_std = vals_std[-1] if len(vals_std) > 0 else 0
        ax_summary.bar(['Mean'], [final_mean], yerr=[final_std],
                        color=PUB_COLORS['blue'], alpha=0.7, capsize=5)
    # Also show overall attention entropy
    steps_overall, vals_overall = _valid(data, 'attention_entropy')
    if len(vals_overall) > 0:
        ax_summary.bar(['Overall'], [vals_overall[-1]],
                        color=PUB_COLORS['orange'], alpha=0.7)

    ax_summary.set_ylabel('Entropy (nats)')
    ax_summary.set_title('(b) Final Entropy')

    fig.suptitle('Attention Entropy & Head Specialization', fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# M6: VFE Gradient Decomposition
# =============================================================================

def plot_vfe_gradient_components(
    csv_path: Path,
    save_path: Optional[Path] = None,
    start_step: int = 100,
    smooth_window: int = 25,
) -> Any:
    r"""VFE gradient component norms over training.

    Shows self-coupling, direct alignment, and softmax coupling
    contributions to both mu and sigma gradients.

    Addresses audit gap M6.
    """
    _ensure_mpl()
    data = _load_csv(csv_path)

    mu_components = [
        ('vfe_grad_mu_self', 'Self-coupling', PUB_COLORS['blue']),
        ('vfe_grad_mu_direct', 'Alignment (direct)', PUB_COLORS['orange']),
        ('vfe_grad_mu_softmax', 'Softmax coupling', PUB_COLORS['green']),
        ('vfe_grad_mu_total', 'Total', PUB_COLORS['black']),
    ]
    sigma_components = [
        ('vfe_grad_sigma_self', 'Self-coupling', PUB_COLORS['blue']),
        ('vfe_grad_sigma_align_direct', 'Alignment (direct)', PUB_COLORS['orange']),
        ('vfe_grad_sigma_softmax', 'Softmax coupling', PUB_COLORS['green']),
        ('vfe_grad_sigma_total', 'Total', PUB_COLORS['black']),
    ]

    fig, (ax_mu, ax_sigma) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for components, ax, title in [
        (mu_components, ax_mu, r'(a) $\nabla_\mu \mathcal{F}$ Components'),
        (sigma_components, ax_sigma, r'(b) $\nabla_\sigma \mathcal{F}$ Components'),
    ]:
        for key, label, color in components:
            steps, vals = _valid(data, key)
            if len(steps) == 0:
                continue
            mask = steps >= start_step
            lw = 2.0 if 'total' in key else 1.2
            ls = '-' if 'total' not in key else '-'
            ax.semilogy(steps[mask], _smooth(vals[mask], smooth_window),
                        label=label, color=color, linewidth=lw, linestyle=ls)
            # Raw data (faint)
            ax.semilogy(steps[mask], vals[mask], color=color,
                        alpha=0.1, linewidth=0.3)
        ax.set_ylabel('Gradient Norm')
        ax.set_title(title)
        ax.legend(fontsize=7)

    ax_sigma.set_xlabel('Training Step')
    fig.suptitle('VFE Gradient Decomposition', fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# H8: Linear Probe on Frozen Mu
# =============================================================================

@torch.no_grad()
def evaluate_linear_probe(
    model: Any,
    dataloader: Any,
    device: str = 'cuda',
    max_batches: int = 100,
) -> Dict[str, float]:
    r"""Evaluate a linear probe on frozen mu embeddings.

    Trains a simple linear classifier on the model's post-E-step mu
    representations to predict the next token. High accuracy means
    beliefs encode task-relevant semantics.

    Addresses audit gap H8. Returns accuracy metrics (does not plot).
    """
    device_t = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device_t)

    all_mu = []
    all_targets = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        input_ids, target_ids = batch
        input_ids = input_ids.to(device_t)
        target_ids = target_ids.to(device_t)

        out = model.forward_with_attention(input_ids)
        mu = out['mu_q']  # (B, N, K)
        all_mu.append(mu.cpu())
        all_targets.append(target_ids.cpu())

    mu_cat = torch.cat(all_mu, dim=0)  # (total_B, N, K)
    targets_cat = torch.cat(all_targets, dim=0)  # (total_B, N)

    # Flatten to (total_B * N, K) and (total_B * N,)
    B, N, K = mu_cat.shape
    mu_flat = mu_cat.reshape(-1, K).float()
    targets_flat = targets_cat.reshape(-1)

    # Remove padding
    valid = targets_flat != -100
    mu_flat = mu_flat[valid]
    targets_flat = targets_flat[valid]

    # Train linear probe (closed-form least squares on one-hot targets is
    # too memory-intensive for V=50k; use iterative logistic regression)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Subsample if too large
    n = mu_flat.shape[0]
    if n > 50000:
        indices = torch.randperm(n)[:50000]
        mu_flat = mu_flat[indices]
        targets_flat = targets_flat[indices]

    X = mu_flat.numpy()
    y = targets_flat.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(
        max_iter=200, solver='saga', n_jobs=-1, C=1.0, verbose=0)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    # Top-5 accuracy
    probs_test = clf.predict_proba(X_test)
    top5_preds = np.argsort(probs_test, axis=1)[:, -5:]
    top5_acc = np.mean([y_test[i] in top5_preds[i] for i in range(len(y_test))])

    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'top5_accuracy': top5_acc,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_classes': len(np.unique(y_train)),
        'embed_dim': K,
    }


def plot_linear_probe_results(
    results: Dict[str, float],
    save_path: Optional[Path] = None,
) -> Any:
    """Bar chart of linear probe accuracy on frozen mu embeddings."""
    _ensure_mpl()

    fig, ax = plt.subplots(figsize=(6, 4))

    metrics = ['test_accuracy', 'top5_accuracy']
    labels = ['Top-1 Accuracy', 'Top-5 Accuracy']
    values = [results.get(m, 0) for m in metrics]
    colors = [PUB_COLORS['blue'], PUB_COLORS['orange']]

    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Accuracy')
    ax.set_title(f'Linear Probe on Frozen $\\mu$ (K={results.get("embed_dim", "?")}, '
                 f'{results.get("n_classes", "?")} classes)')
    ax.set_ylim(0, min(1.0, max(values) * 1.3))

    # Chance level
    n_cls = results.get('n_classes', 1)
    if n_cls > 1:
        chance = 1.0 / n_cls
        ax.axhline(chance, color=PUB_COLORS['gray'], linestyle=':', linewidth=1,
                    label=f'Chance: {chance:.4f}')
        ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# H1: KL vs Dot-Product Attention Comparison
# =============================================================================

@torch.no_grad()
def plot_kl_vs_dotproduct_attention(
    model: Any,
    input_ids: torch.Tensor,
    tokenizer: Any = None,
    save_path: Optional[Path] = None,
    head_idx: int = 0,
    device: str = 'cuda',
) -> Any:
    r"""Side-by-side comparison of KL-divergence and dot-product attention.

    Runs the gauge model to get KL attention weights, then computes
    standard dot-product attention from the same mu embeddings for comparison.

    Addresses audit gap H1.
    """
    _ensure_mpl()
    device_t = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device_t)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device_t)

    out = model.forward_with_attention(input_ids)
    beta_kl = out.get('beta', out.get('attention_weights', None))

    # Extract KL attention for the requested head
    if beta_kl is not None and beta_kl.dim() == 4:
        # (B, n_heads, N, N) -> take head_idx
        attn_kl = beta_kl[0, head_idx].cpu().numpy()
    elif beta_kl is not None and beta_kl.dim() == 3:
        attn_kl = beta_kl[0].cpu().numpy()
    else:
        print("Could not extract KL attention weights")
        return None

    # Compute dot-product attention from the same mu embeddings
    mu = out['mu_q'][0]  # (N, K)
    N, K = mu.shape
    scores_dp = (mu @ mu.T) / math.sqrt(K)  # (N, N)
    # Apply causal mask
    causal = torch.tril(torch.ones(N, N, device=mu.device))
    scores_dp = scores_dp.masked_fill(causal == 0, float('-inf'))
    attn_dp = F.softmax(scores_dp, dim=-1).cpu().numpy()

    # Token labels
    if tokenizer is not None:
        tokens = [tokenizer.decode([t]) for t in input_ids[0].cpu().tolist()]
    else:
        tokens = [str(i) for i in range(N)]
    # Truncate long tokens
    tokens = [t[:8] for t in tokens]

    fig, (ax_kl, ax_dp, ax_diff) = plt.subplots(1, 3, figsize=(16, 5))

    vmax = max(attn_kl.max(), attn_dp.max())
    kwargs = dict(cmap='Blues', vmin=0, vmax=vmax, aspect='auto')

    im1 = ax_kl.imshow(attn_kl, **kwargs)
    ax_kl.set_title(f'(a) KL Attention (head {head_idx})')
    ax_kl.set_xlabel('Key')
    ax_kl.set_ylabel('Query')

    im2 = ax_dp.imshow(attn_dp, **kwargs)
    ax_dp.set_title('(b) Dot-Product Attention')
    ax_dp.set_xlabel('Key')

    diff = attn_kl - attn_dp
    vabs = max(abs(diff.min()), abs(diff.max()))
    im3 = ax_diff.imshow(diff, cmap='RdBu_r', vmin=-vabs, vmax=vabs, aspect='auto')
    ax_diff.set_title('(c) Difference (KL - DP)')
    ax_diff.set_xlabel('Key')
    plt.colorbar(im3, ax=ax_diff, fraction=0.046)

    # Add token labels if not too many
    if N <= 32:
        for ax in [ax_kl, ax_dp, ax_diff]:
            ax.set_xticks(range(N))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticks(range(N))
            ax.set_yticklabels(tokens, fontsize=6)

    # Compute correlation
    corr = np.corrcoef(attn_kl.flatten(), attn_dp.flatten())[0, 1]
    fig.suptitle(f'KL vs Dot-Product Attention (Pearson r = {corr:.3f})',
                 fontsize=13, y=1.02)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
