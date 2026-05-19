r"""
Publication-quality visualizations for cross-head coupling in the /vfe path.

All figures use matplotlib only (no plotly / seaborn) so they are equally
usable inside notebooks and in headless training. Per CLAUDE.md, defaults
target publication quality: tight layout, 300 dpi save, colorblind-safe
palettes, and axis labels in math mode where applicable.

Functions accept the same artifacts the metrics module consumes plus a
``VFEConfig`` for structural context. Each returns a ``matplotlib.figure.Figure``
so the caller decides whether to save, ``plt.show()``, or push to TensorBoard.

Figure index
============

1. :func:`plot_generator_sparsity` — :math:`(n_{\mathrm{gen}}, K, K)` sparsity
   heatmap. Diagonal generators show as block-diagonal squares; cross
   generators show as off-diagonal patches inside super-blocks.
2. :func:`plot_omega_block_strength` — :math:`(n_{\mathrm{heads}},
   n_{\mathrm{heads}})` heatmap of mean Frobenius mass per head-pair sub-
   block of :math:`\Omega`. Off-diagonal entries inside merged super-blocks
   are the cross-coupling signal.
3. :func:`plot_super_block_graph` — node-link diagram of the super-block
   partition. Nodes are heads, edges are coupling pairs; coupled-component
   color matches super-block ID.
4. :func:`plot_phi_energy_partition` — bar chart of
   :math:`E_{\mathrm{diag}}` vs :math:`E_{\mathrm{cross}}` per layer.
5. :func:`plot_kl_share` — stacked bar of merged-vs-singleton super-block
   KL contributions per layer.

Returned figure objects can be combined with the standard matplotlib
``savefig(path, dpi=300, bbox_inches='tight')`` invocation. None of these
functions write to disk by themselves.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch


# Colorblind-safe palettes. Wong (2011), 8-color cycle.
_WONG = [
    '#000000', '#E69F00', '#56B4E9', '#009E73',
    '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
]


def _apply_publication_style(ax) -> None:
    """In-place: minor publication-quality polish on a matplotlib Axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=4, width=1.0)


def _super_block_extents(cfg) -> List[Tuple[int, int]]:
    """List of (start, end) ranges in the K dim for each super-block."""
    extents = []
    cursor = 0
    for d in cfg.effective_block_dims:
        extents.append((cursor, cursor + d))
        cursor += d
    return extents


# -----------------------------------------------------------------------------
# 1. Generator sparsity heatmap
# -----------------------------------------------------------------------------


def plot_generator_sparsity(
    generators: torch.Tensor,
    cfg,
    title: Optional[str] = None,
) -> mpl.figure.Figure:
    r"""Sparsity heatmap of the (combined) generator basis on the K dimension.

    Plots :math:`\sum_a |G_a|` so every non-zero entry across the basis shows
    up. The figure annotates the super-block partition lines (white dashed)
    and the per-head dashed lines (gray dotted) inside merged super-blocks.

    Args:
        generators: ``(n_gen, K, K)`` tensor or numpy array.
        cfg: VFEConfig with active block structure.
        title: Optional figure title. Defaults to a structural summary.
    """
    if isinstance(generators, torch.Tensor):
        G = generators.detach().to(torch.float64).cpu().numpy()
    else:
        G = np.asarray(generators, dtype=np.float64)
    K = G.shape[-1]
    sparsity = np.abs(G).sum(axis=0)                 # (K, K)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(
        sparsity, cmap='magma', origin='upper',
        interpolation='nearest', aspect='equal',
    )
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04,
                 label=r'$\sum_a |G_a[i,j]|$')

    # Super-block boundaries (solid white).
    for s, e in _super_block_extents(cfg):
        ax.axhline(s - 0.5, color='white', lw=0.8, linestyle='--', alpha=0.7)
        ax.axvline(s - 0.5, color='white', lw=0.8, linestyle='--', alpha=0.7)
    ax.axhline(K - 0.5, color='white', lw=0.8, linestyle='--', alpha=0.7)
    ax.axvline(K - 0.5, color='white', lw=0.8, linestyle='--', alpha=0.7)

    # Per-head boundaries inside merged super-blocks (dotted gray).
    if cfg.is_cross_coupled:
        _, _n_heads, d_head = cfg.irrep_spec[0]
        cursor = 0
        for s, e in _super_block_extents(cfg):
            d = e - s
            if d > d_head:
                # interior head boundaries
                for k in range(1, d // d_head):
                    pos = s + k * d_head
                    ax.axhline(pos - 0.5, color='lightgray', lw=0.6,
                               linestyle=':', alpha=0.7)
                    ax.axvline(pos - 0.5, color='lightgray', lw=0.6,
                               linestyle=':', alpha=0.7)

    if title is None:
        title = (
            f"Generator sparsity, K={K}, n_gen={G.shape[0]}"
            + (
                f", cross_couplings={len(cfg.cross_couplings)}"
                if cfg.is_cross_coupled else ""
            )
        )
    ax.set_title(title)
    ax.set_xlabel(r'column index $j$')
    ax.set_ylabel(r'row index $i$')
    _apply_publication_style(ax)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 2. Omega block-strength heatmap (per-head-pair Frobenius mass)
# -----------------------------------------------------------------------------


def plot_omega_block_strength(
    block_strength: np.ndarray,
    cfg,
    title: Optional[str] = None,
    log_scale: bool = True,
) -> mpl.figure.Figure:
    r"""Heatmap of mean Frobenius mass per (head_a, head_b) sub-block of
    :math:`\Omega` (output of
    :func:`vfe.cross_coupling_metrics.omega_block_strength`).

    Off-diagonal entries inside a merged super-block visualize the cross-
    coupling contribution. Singleton super-blocks have all mass concentrated
    on the diagonal entry.

    Args:
        block_strength: ``(n_heads, n_heads)`` numpy matrix.
        cfg: VFEConfig.
        title: Optional figure title.
        log_scale: If True, plot ``log10(strength + eps)``; else linear.
    """
    _, n_heads, _ = cfg.irrep_spec[0]
    M = np.asarray(block_strength, dtype=np.float64)
    if log_scale:
        eps = max(1e-12, np.percentile(M[M > 0], 1) * 1e-3 if (M > 0).any() else 1e-12)
        Mdisp = np.log10(M + eps)
        cbar_label = r'$\log_{10}\,\overline{\|\Omega_{ab}\|_F^2}$'
    else:
        Mdisp = M
        cbar_label = r'$\overline{\|\Omega_{ab}\|_F^2}$'

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(
        Mdisp, cmap='viridis', origin='upper',
        interpolation='nearest', aspect='equal',
    )
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label=cbar_label)
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_heads))
    ax.set_xticklabels(range(n_heads))
    ax.set_yticklabels(range(n_heads))
    ax.set_xlabel('head $b$')
    ax.set_ylabel('head $a$')

    # Outline super-block groups in red dashed.
    if cfg.is_cross_coupled and cfg.super_block_head_groups is not None:
        for group in cfg.super_block_head_groups:
            if len(group) <= 1:
                continue
            heads = sorted(group)
            for a in heads:
                for b in heads:
                    rect = plt.Rectangle(
                        (b - 0.5, a - 0.5), 1, 1, fill=False,
                        edgecolor=_WONG[6], linewidth=1.2, linestyle='--',
                    )
                    ax.add_patch(rect)

    if title is None:
        title = (
            r'Mean per-head-pair $\|\Omega_{ab}\|_F^2$'
            + ('  (merged super-blocks dashed red)' if cfg.is_cross_coupled else '')
        )
    ax.set_title(title)
    _apply_publication_style(ax)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 3. Super-block adjacency graph
# -----------------------------------------------------------------------------


def plot_super_block_graph(
    cfg,
    title: Optional[str] = None,
) -> mpl.figure.Figure:
    r"""Node-link diagram of the cross-coupling pattern.

    Each node is an original head. Each edge is a cross-coupling pair
    ``(a, b)``. Nodes are colored by super-block ID. Singleton super-blocks
    get a desaturated color; merged super-blocks pick saturated colors from
    the Wong palette.

    Layout: heads arranged on a circle for ``n_heads <= 16``; otherwise grid.
    """
    _, n_heads, _ = cfg.irrep_spec[0]
    if cfg.is_cross_coupled and cfg.super_block_head_groups is not None:
        groups = cfg.super_block_head_groups
        couplings = cfg.cross_couplings
    else:
        groups = [[h] for h in range(n_heads)]
        couplings = []

    head_to_super = {h: i for i, g in enumerate(groups) for h in g}

    # Layout
    if n_heads <= 16:
        angles = np.linspace(0, 2 * math.pi, n_heads, endpoint=False)
        positions = {
            h: (math.cos(a), math.sin(a)) for h, a in zip(range(n_heads), angles)
        }
    else:
        cols = int(math.ceil(math.sqrt(n_heads)))
        positions = {
            h: ((h % cols) - cols / 2, (h // cols) - cols / 2)
            for h in range(n_heads)
        }

    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    # Draw edges (couplings).
    for (a, b) in couplings:
        xa, ya = positions[a]
        xb, yb = positions[b]
        ax.annotate(
            '',
            xy=(xb, yb), xycoords='data',
            xytext=(xa, ya), textcoords='data',
            arrowprops=dict(
                arrowstyle='-|>',
                color=_WONG[6],
                lw=1.2,
                alpha=0.8,
                connectionstyle='arc3,rad=0.10',
            ),
        )

    # Draw nodes.
    for h in range(n_heads):
        x, y = positions[h]
        super_id = head_to_super[h]
        # Singletons in light gray; merged super-blocks in saturated colors.
        size_h = len(groups[super_id])
        if size_h <= 1:
            color = '#cccccc'
            edge = '#666666'
        else:
            color = _WONG[1 + (super_id % (len(_WONG) - 1))]
            edge = 'black'
        circle = plt.Circle((x, y), 0.10, facecolor=color, edgecolor=edge, lw=1.2, zorder=5)
        ax.add_patch(circle)
        ax.text(
            x, y, str(h),
            ha='center', va='center', fontsize=10, fontweight='bold', zorder=6,
        )

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    if title is None:
        title = (
            f'Super-block partition  '
            f'({len(groups)} super-blocks, '
            f'{len(couplings)} cross-couplings)'
        )
    ax.set_title(title)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 4. Phi-energy partition bar chart
# -----------------------------------------------------------------------------


def plot_phi_energy_partition(
    energy_per_layer: List[Dict[str, float]],
    cfg=None,
    title: Optional[str] = None,
) -> mpl.figure.Figure:
    r"""Stacked bar of ``phi_energy_diag`` vs ``phi_energy_cross`` per layer.

    Args:
        energy_per_layer: List of dicts (one per layer) as returned by
            :func:`vfe.cross_coupling_metrics.phi_energy_partition`.
        cfg: VFEConfig (unused except for the optional title).
        title: Optional figure title.
    """
    L = len(energy_per_layer)
    diag = np.array([d['phi_energy_diag'] for d in energy_per_layer])
    cross = np.array([d['phi_energy_cross'] for d in energy_per_layer])
    xs = np.arange(L)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.bar(xs, diag, color=_WONG[2], edgecolor='black', linewidth=0.5, label='diagonal generators')
    ax.bar(xs, cross, bottom=diag, color=_WONG[6], edgecolor='black', linewidth=0.5, label='cross generators')
    ax.set_xlabel('layer')
    ax.set_ylabel(r'$\overline{\phi^2}$')
    ax.set_xticks(xs)
    ax.legend(frameon=False, loc='best', fontsize=9)
    if title is None:
        title = r'$\phi$ energy partition: diag vs cross generators'
    ax.set_title(title)
    _apply_publication_style(ax)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 5. KL share stacked bar
# -----------------------------------------------------------------------------


def plot_kl_share(
    kl_share_per_layer: List[Dict[str, float]],
    cfg=None,
    title: Optional[str] = None,
) -> mpl.figure.Figure:
    r"""Stacked bar of singleton-vs-merged super-block KL per layer.

    Args:
        kl_share_per_layer: List of dicts (one per layer) as returned by
            :func:`vfe.cross_coupling_metrics.cross_block_kl_share`.
        cfg: VFEConfig (unused except for the optional title).
        title: Optional figure title.
    """
    L = len(kl_share_per_layer)
    singleton = np.array([d['kl_singleton'] for d in kl_share_per_layer])
    merged = np.array([d['kl_merged'] for d in kl_share_per_layer])
    xs = np.arange(L)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.bar(xs, singleton, color=_WONG[2], edgecolor='black', linewidth=0.5,
           label='singleton super-blocks')
    ax.bar(xs, merged, bottom=singleton, color=_WONG[6], edgecolor='black',
           linewidth=0.5, label='merged (cross-coupled) super-blocks')
    ax.set_xlabel('layer')
    ax.set_ylabel(r'$\sum$ KL$_{ij}$')
    ax.set_xticks(xs)
    ax.legend(frameon=False, loc='best', fontsize=9)
    if title is None:
        title = 'Per-super-block KL share (singleton vs merged)'
    ax.set_title(title)
    _apply_publication_style(ax)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 6. Per-super-block effective rank / entropy line plot
# -----------------------------------------------------------------------------


def plot_super_block_diagnostics(
    *,
    effective_rank: Optional[np.ndarray] = None,
    attention_entropy: Optional[np.ndarray] = None,
    cfg=None,
    title: Optional[str] = None,
) -> mpl.figure.Figure:
    r"""Side-by-side bar charts of per-super-block effective rank and attention
    entropy.

    Singleton super-blocks are colored gray; merged super-blocks are colored
    by the Wong palette so the cross-coupled blocks are visually distinct.

    Either ``effective_rank`` or ``attention_entropy`` (or both) must be
    given. When only one is supplied the figure has a single subplot.
    """
    if effective_rank is None and attention_entropy is None:
        raise ValueError(
            "plot_super_block_diagnostics requires at least one of "
            "effective_rank or attention_entropy."
        )

    n_plots = int(effective_rank is not None) + int(attention_entropy is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.0 * n_plots, 4.0), squeeze=False)
    panels = axes[0]

    if cfg is not None and cfg.is_cross_coupled:
        _, _n_heads, d_head = cfg.irrep_spec[0]
        merged_mask = np.array(
            [d > d_head for d in cfg.effective_block_dims], dtype=bool,
        )
    else:
        merged_mask = None

    next_panel = 0
    if effective_rank is not None:
        ax = panels[next_panel]
        next_panel += 1
        xs = np.arange(len(effective_rank))
        colors = (
            [_WONG[6] if merged_mask[i] else '#bbbbbb' for i in xs]
            if merged_mask is not None else _WONG[2]
        )
        ax.bar(xs, effective_rank, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('super-block index')
        ax.set_ylabel(r'effective rank $\exp(H(\sigma))$')
        ax.set_xticks(xs)
        ax.set_title('Effective rank per super-block')
        _apply_publication_style(ax)

    if attention_entropy is not None:
        ax = panels[next_panel]
        xs = np.arange(len(attention_entropy))
        colors = (
            [_WONG[6] if merged_mask[i] else '#bbbbbb' for i in xs]
            if merged_mask is not None else _WONG[2]
        )
        ax.bar(xs, attention_entropy, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('super-block index')
        ax.set_ylabel(r'$\overline{H(\beta_{ij})}$  (nats)')
        ax.set_xticks(xs)
        ax.set_title('Attention entropy per super-block')
        _apply_publication_style(ax)

    if title is not None:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig
