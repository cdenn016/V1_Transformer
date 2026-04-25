"""
Gauge Geometry Visualization
==============================

Publication-quality figures for Yang-Mills energy, curvature tensor
decomposition, gauge field Dirichlet energy, and gauge orbit analysis.

Mathematical background
-----------------------
The Yang-Mills functional over a principal bundle with connection A is

    E_YM[A] = (1/2) ||F_A||² = (1/2) ∫ tr(F ∧ *F)

where the curvature two-form is F_A = dA + A ∧ A.  In the discrete,
finite-dimensional setting used here, the curvature tensor for an ordered
triple of tokens (i, j, k) is

    F_{ijk} = Omega_{ij} @ Omega_{jk} @ Omega_{ki} - I_K

i.e. the failure of parallel transport around the triangle (i,j,k) to return
to the identity.  The abelian (diagonal/trace) and non-abelian (off-diagonal
commutator) parts decompose as

    F = F_ab + F_nonab,    tr(F_ab) = tr(F),    F_nonab = F - F_ab.

The Dirichlet / gauge-field energy for a pair (i,j) is

    E_D(i,j) = ||phi_i - phi_j||²

which enters the Yang-Mills energy through the transport operators
Omega_ij = exp(phi_i) exp(-phi_j).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None          # type: ignore[assignment]
    Figure = None       # type: ignore[assignment]
    Patch = None        # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False

from transformer.visualization.pub_style import set_pub_style, PUB_COLORS, PUB_CYCLE, _safe_legend


# =============================================================================
# Internal helpers
# =============================================================================

def _save(fig: "Figure", save_path: Optional[Path]) -> None:
    """Save *fig* to *save_path* at 300 DPI with tight bounding box."""
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


def _no_data_axes(ax, message: str = "No data available") -> None:
    """Render a centered placeholder message on *ax*."""
    ax.text(
        0.5, 0.5, message,
        transform=ax.transAxes,
        ha='center', va='center',
        fontsize=10, color=PUB_COLORS['gray'],
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _unit_circle(ax) -> None:
    r"""Overlay the unit circle on a complex-plane axes.

    Used to indicate where eigenvalues of unitary / orthogonal matrices lie.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta),
            color=PUB_COLORS['gray'], linewidth=0.8, linestyle='--',
            alpha=0.6, zorder=1, label='Unit circle')


# =============================================================================
# Figure 1 — Yang-Mills Energy Evolution
# =============================================================================

def plot_yang_mills_evolution(
    steps: List[int],
    energies: List[float],
    dirichlet_energies: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 5),
) -> Optional["Figure"]:
    r"""Plot Yang-Mills energy over the course of training.

    The Yang-Mills functional is approximated in the discrete setting as

    .. math::

        E_{\mathrm{YM}} = \frac{1}{|\mathcal{T}|}
            \sum_{(i,j,k)\in\mathcal{T}} \|F_{ijk}\|_F^2

    where :math:`\mathcal{T}` is the set of sampled token triples and
    :math:`F_{ijk} = \Omega_{ij}\Omega_{jk}\Omega_{ki} - I`.

    When *dirichlet_energies* is provided, the Dirichlet / gauge-field energy

    .. math::

        E_D = \frac{1}{N^2}\sum_{i,j} \|\phi_i - \phi_j\|^2

    is plotted on a secondary y-axis.

    Args:
        steps: Training step indices, length T.
        energies: Yang-Mills energy values, length T.
        dirichlet_energies: Optional Dirichlet energy values, length T.
        save_path: If given, save the figure here.
        figsize: Figure dimensions in inches.

    Returns:
        The :class:`matplotlib.figure.Figure` or ``None`` if matplotlib is
        unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    set_pub_style()

    steps_arr = np.asarray(steps, dtype=float)
    energies_arr = np.asarray(energies, dtype=float)

    # Flat (bilinear / cocycle) transport gives E_YM = 0 exactly; in
    # fp32 the reported value is bounded by matrix-log roundoff, which
    # empirically sits near 1e-8 at K=8 and scales mildly with K.
    # 1e-4 leaves two orders of magnitude of headroom above that noise
    # floor while still admitting any physically meaningful non-flat
    # signal, which tracks O(phi_scale^2) = O(1e-3) or larger.
    ym_tol = 1e-4
    finite_ym = energies_arr[np.isfinite(energies_arr)]
    ym_has_signal = finite_ym.size > 0 and float(np.max(finite_ym)) > ym_tol

    dir_arr: Optional[np.ndarray] = None
    dir_valid = np.zeros(0, dtype=bool)
    if dirichlet_energies is not None:
        dir_arr = np.asarray(dirichlet_energies, dtype=float)
        dir_valid = np.isfinite(dir_arr)

    # --------------------------------------------------------------
    # Case 1: YM has no signal (flat transport). Collapse to a single
    # Dirichlet-only axis with an annotation documenting E_YM ≡ 0.
    # --------------------------------------------------------------
    if not ym_has_signal:
        fig, ax = plt.subplots(figsize=figsize)
        if dir_arr is not None and dir_valid.any():
            ax.plot(
                steps_arr[dir_valid], dir_arr[dir_valid],
                color=PUB_COLORS['orange'], linewidth=1.8,
                label=r'$E_D$', zorder=3,
            )
            ax.fill_between(
                steps_arr[dir_valid], 0, dir_arr[dir_valid],
                color=PUB_COLORS['orange'], alpha=0.08,
            )
            ax.set_ylabel(
                r'Dirichlet energy $E_D = \overline{\|\phi_i - \phi_j\|^2}$'
            )
            dir_min = float(dir_arr[dir_valid].min())
            if dir_min > 0.0:
                ax.set_yscale('log')
            ax.text(
                0.02, 0.02,
                r'$E_{\mathrm{YM}} \equiv 0$ (flat bilinear transport)',
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=8, color=PUB_COLORS['gray'],
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75),
            )
            _safe_legend(ax, loc='upper right', framealpha=0.9)
        else:
            _no_data_axes(ax, 'No Yang-Mills or Dirichlet data')
        ax.set_xlabel('Training step')
        ax.set_title('Gauge-Field Energy Evolution')
        fig.tight_layout()
        _save(fig, save_path)
        return fig

    # --------------------------------------------------------------
    # Case 2: YM has signal. Two-axis plot with guarded log scales.
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    valid = np.isfinite(energies_arr)
    ax.plot(
        steps_arr[valid], energies_arr[valid],
        color=PUB_COLORS['blue'], linewidth=1.8,
        label=r'$E_{\mathrm{YM}}$', zorder=3,
    )
    ax.fill_between(
        steps_arr[valid], 0, energies_arr[valid],
        color=PUB_COLORS['blue'], alpha=0.08,
    )
    ax.set_xlabel('Training step')
    ax.set_ylabel(r'$E_{\mathrm{YM}}$')
    ym_min = float(energies_arr[valid].min())
    if ym_min > 0.0:
        ax.set_yscale('log')
    ax.set_title('Yang-Mills Energy Evolution')
    _safe_legend(ax, loc='upper right', framealpha=0.9)

    if dir_arr is not None and dir_valid.any():
        ax2 = ax.twinx()
        ax2.plot(
            steps_arr[dir_valid], dir_arr[dir_valid],
            color=PUB_COLORS['orange'], linewidth=1.5,
            linestyle='--', label=r'$E_D$', zorder=2,
        )
        ax2.set_ylabel(
            r'Dirichlet energy $E_D = \overline{\|\phi_i - \phi_j\|^2}$',
            color=PUB_COLORS['orange'],
        )
        ax2.tick_params(axis='y', labelcolor=PUB_COLORS['orange'])
        dir_min = float(dir_arr[dir_valid].min())
        if dir_min > 0.0:
            ax2.set_yscale('log')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        _safe_legend(ax, h1 + h2, l1 + l2, loc='upper right', framealpha=0.9)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =============================================================================
# Figure 2 — Curvature Tensor Eigenvalue Spectrum
# =============================================================================

def plot_curvature_spectrum(
    F: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
) -> Optional["Figure"]:
    r"""Scatter eigenvalues of curvature tensors in the complex plane.

    Each curvature matrix :math:`F_t \in \mathbb{R}^{K \times K}` (one per
    token triple) is eigendecomposed.  Because :math:`F_t` is generally
    non-symmetric, eigenvalues may be complex; they are plotted in the complex
    plane and coloured by triple index.  The unit circle is overlaid as a
    reference: eigenvalues near the unit circle indicate near-unitary curvature
    (pure rotation), while magnitudes significantly less than one indicate
    contraction.

    Args:
        F: Curvature tensors of shape ``(n_triples, K, K)``.
        save_path: If given, save the figure here.
        figsize: Figure dimensions in inches.

    Returns:
        The :class:`matplotlib.figure.Figure` or ``None`` if matplotlib is
        unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    set_pub_style()

    if F is None or F.ndim != 3 or F.shape[0] == 0:
        fig, ax = plt.subplots(figsize=figsize)
        _no_data_axes(ax, 'No curvature data (expected shape (n_triples, K, K))')
        ax.set_title('Curvature Tensor Eigenvalue Spectrum')
        fig.tight_layout()
        _save(fig, save_path)
        return fig

    n_triples, K, _ = F.shape
    fig, ax = plt.subplots(figsize=figsize)

    # Colourmap: cycle through PUB_CYCLE, wrapping for large n_triples
    cmap = plt.cm.get_cmap('viridis', n_triples)  # type: ignore[attr-defined]

    for t in range(n_triples):
        try:
            eigvals = np.linalg.eigvals(F[t])
        except np.linalg.LinAlgError:
            continue
        color = cmap(t / max(n_triples - 1, 1))
        ax.scatter(
            eigvals.real, eigvals.imag,
            s=18, alpha=0.7, color=color,
            zorder=3, linewidths=0.0,
        )

    _unit_circle(ax)
    # Mark the identity eigenvalue (1, 0): a zero-curvature tensor has all
    # eigenvalues equal to zero (since F = Omega - I); we mark the origin too.
    ax.axhline(0.0, color=PUB_COLORS['gray'], linewidth=0.5, alpha=0.4)
    ax.axvline(0.0, color=PUB_COLORS['gray'], linewidth=0.5, alpha=0.4)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap,  # type: ignore[attr-defined]
                                norm=plt.Normalize(0, n_triples - 1))  # type: ignore[attr-defined]
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Triple index $t$', fontsize=9)

    ax.set_xlabel(r'$\mathrm{Re}(\lambda)$')
    ax.set_ylabel(r'$\mathrm{Im}(\lambda)$')
    ax.set_title('Curvature Tensor Eigenvalue Spectrum')
    ax.set_aspect('equal')
    _safe_legend(ax, loc='upper right', framealpha=0.9)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =============================================================================
# Figure 3 — Abelian / Non-Abelian Curvature Decomposition
# =============================================================================

def plot_curvature_decomposition(
    F: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 5),
) -> Optional["Figure"]:
    r"""Stacked bar chart decomposing curvature into abelian and non-abelian parts.

    For each curvature tensor :math:`F_t` the Frobenius norm is split as

    .. math::

        \|F_t\|_F^2
          = \underbrace{\sum_{k} |F_t{}_{kk}|^2}_{\text{abelian}}
          + \underbrace{\sum_{k \neq l} |F_t{}_{kl}|^2}_{\text{non-abelian}}

    The abelian fraction :math:`r^{\mathrm{ab}}_t = \|F_t^{\mathrm{diag}}\|_F^2 /
    \|F_t\|_F^2` and non-abelian fraction :math:`r^{\mathrm{nab}}_t = 1 -
    r^{\mathrm{ab}}_t` are plotted as a stacked bar chart.  A purely U(1) /
    diagonal gauge field would have :math:`r^{\mathrm{ab}} = 1`; a fully
    non-commutative field would have :math:`r^{\mathrm{ab}} \to 0`.

    Args:
        F: Curvature tensors of shape ``(n_triples, K, K)``.
        save_path: If given, save the figure here.
        figsize: Figure dimensions in inches.

    Returns:
        The :class:`matplotlib.figure.Figure` or ``None`` if matplotlib is
        unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    set_pub_style()

    if F is None or F.ndim != 3 or F.shape[0] == 0:
        fig, ax = plt.subplots(figsize=figsize)
        _no_data_axes(ax, 'No curvature data (expected shape (n_triples, K, K))')
        ax.set_title('Abelian vs Non-Abelian Curvature')
        fig.tight_layout()
        _save(fig, save_path)
        return fig

    n_triples = F.shape[0]

    # Compute per-triple decomposition
    total_sq = np.sum(F ** 2, axis=(1, 2))  # (n_triples,)
    diag_sq = np.array([np.sum(np.diag(F[t]) ** 2) for t in range(n_triples)])
    offdiag_sq = total_sq - diag_sq

    # Guard division by zero
    safe_total = np.where(total_sq > 0, total_sq, 1.0)
    frac_ab = diag_sq / safe_total       # (n_triples,)
    frac_nab = offdiag_sq / safe_total   # (n_triples,)

    # Clamp numerical noise
    frac_ab = np.clip(frac_ab, 0.0, 1.0)
    frac_nab = np.clip(frac_nab, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(n_triples)

    ax.bar(
        xs, frac_ab,
        color=PUB_COLORS['blue'], alpha=0.85,
        label=r'Abelian $\|F^{\mathrm{diag}}\|_F^2 / \|F\|_F^2$',
    )
    ax.bar(
        xs, frac_nab,
        bottom=frac_ab,
        color=PUB_COLORS['orange'], alpha=0.85,
        label=r'Non-abelian $\|F^{\mathrm{off}}\|_F^2 / \|F\|_F^2$',
    )

    # Mean fraction annotations
    mean_ab = float(np.mean(frac_ab))
    ax.axhline(mean_ab, color=PUB_COLORS['blue'], linewidth=1.0,
               linestyle=':', alpha=0.6)
    ax.text(
        n_triples - 0.5, mean_ab + 0.02,
        rf'$\bar{{r}}^{{\mathrm{{ab}}}} = {mean_ab:.2f}$',
        ha='right', va='bottom', fontsize=8, color=PUB_COLORS['blue'],
    )

    ax.set_xlabel('Triple index $t$')
    ax.set_ylabel('Squared Frobenius fraction')
    ax.set_ylim(0.0, 1.05)
    ax.set_title('Abelian vs Non-Abelian Curvature')
    _safe_legend(ax, loc='lower right', framealpha=0.9)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =============================================================================
# Figure 4 — Gauge Field Energy Map
# =============================================================================

def plot_gauge_field_energy_map(
    phi_diff_sq: np.ndarray,
    beta: np.ndarray,
    token_labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 7),
) -> Optional["Figure"]:
    r"""Heatmap of attention-weighted gauge-field Dirichlet energy.

    For each token pair :math:`(i, j)` the entry

    .. math::

        w_{ij} = \beta_{ij} \cdot \|\phi_i - \phi_j\|^2

    measures how much the attention weight allocates to pairs with large
    gauge-frame separation.  Small :math:`w_{ij}` means either the pair is
    nearly gauge-aligned or its attention weight is negligible; large
    :math:`w_{ij}` signals both high attention *and* a large transport.

    Args:
        phi_diff_sq: ``(N, N)`` matrix of squared gauge-frame distances
            :math:`\|\phi_i - \phi_j\|^2`.
        beta: ``(N, N)`` attention weight matrix :math:`\beta_{ij}`.
        token_labels: Optional length-N list of token strings for tick labels.
        save_path: If given, save the figure here.
        figsize: Figure dimensions in inches.

    Returns:
        The :class:`matplotlib.figure.Figure` or ``None`` if matplotlib is
        unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    set_pub_style()

    if phi_diff_sq is None or beta is None:
        fig, ax = plt.subplots(figsize=figsize)
        _no_data_axes(ax, 'No phi_diff_sq / beta data')
        ax.set_title('Gauge Field Energy Map')
        fig.tight_layout()
        _save(fig, save_path)
        return fig

    phi_diff_sq = np.asarray(phi_diff_sq, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if phi_diff_sq.ndim != 2 or beta.ndim != 2:
        fig, ax = plt.subplots(figsize=figsize)
        _no_data_axes(ax, 'phi_diff_sq and beta must be 2-D arrays')
        ax.set_title('Gauge Field Energy Map')
        fig.tight_layout()
        _save(fig, save_path)
        return fig

    energy_map = beta * phi_diff_sq  # (N, N)
    N = energy_map.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        energy_map,
        cmap='YlOrRd',
        aspect='equal',
        interpolation='nearest',
        origin='upper',
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\beta_{ij} \cdot \|\Delta\phi\|^2$', fontsize=10)

    if token_labels is not None and len(token_labels) == N:
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(token_labels, fontsize=7)
    else:
        # Show only a sparse subset of integer ticks for readability
        tick_step = max(1, N // 10)
        ticks = list(range(0, N, tick_step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    ax.set_xlabel('Token $j$')
    ax.set_ylabel('Token $i$')
    ax.set_title('Gauge Field Energy Map')

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =============================================================================
# Figure 5 — Gauge-Invariant Diagnostics
# =============================================================================

def plot_gauge_invariant_scatter(
    invariants: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 4),
) -> Optional["Figure"]:
    r"""Three-panel diagnostic figure for gauge-invariant quantities.

    Panel (a): Histogram of :math:`\det(\Omega_{ij})`, the determinant of
    transport operators.  For :math:`GL^+(K)` transport the determinant is
    always positive; values close to 1 indicate near-volume-preserving transport.

    Panel (b): Distribution of pairwise KL divergences
    :math:`KL(q_i \| \Omega_{ij} q_j)` that drive attention.

    Panel (c): Per-token-pair Dirichlet energy
    :math:`\|\phi_i - \phi_j\|^2` (scalar gauge-field energy).

    Args:
        invariants: Dictionary with optional keys ``'det_omega'``,
            ``'kl_values'``, ``'field_energy'``.  Each value is a 1-D
            numpy array.
        save_path: If given, save the figure here.
        figsize: Figure dimensions in inches.

    Returns:
        The :class:`matplotlib.figure.Figure` or ``None`` if matplotlib is
        unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    set_pub_style()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ---- (a) det(Omega) histogram ----
    ax = axes[0]
    det_data = invariants.get('det_omega') if invariants else None
    if det_data is not None and len(det_data) > 0:
        det_arr = np.asarray(det_data, dtype=float)
        det_arr = det_arr[np.isfinite(det_arr)]
        if len(det_arr) > 0:
            ax.hist(
                det_arr, bins=40, density=True,
                color=PUB_COLORS['blue'], alpha=0.8,
                edgecolor='white', linewidth=0.4,
            )
            ax.axvline(1.0, color=PUB_COLORS['red'], linewidth=1.2,
                       linestyle='--', label=r'$\det = 1$')
            _safe_legend(ax, loc='upper right', framealpha=0.9)
        else:
            _no_data_axes(ax, 'No finite det(Ω) values')
    else:
        _no_data_axes(ax, r'No $\det(\Omega)$ data')
    ax.set_xlabel(r'$\det(\Omega_{ij})$')
    ax.set_ylabel('Density')
    ax.set_title(r'(a) $\det(\Omega_{ij})$ distribution')

    # ---- (b) KL value distribution ----
    ax = axes[1]
    kl_data = invariants.get('kl_values') if invariants else None
    if kl_data is not None and len(kl_data) > 0:
        kl_arr = np.asarray(kl_data, dtype=float)
        kl_arr = kl_arr[np.isfinite(kl_arr) & (kl_arr >= 0)]
        if len(kl_arr) > 0:
            ax.hist(
                kl_arr, bins=40, density=True,
                color=PUB_COLORS['orange'], alpha=0.8,
                edgecolor='white', linewidth=0.4,
            )
            mean_kl = float(np.mean(kl_arr))
            ax.axvline(mean_kl, color=PUB_COLORS['red'], linewidth=1.2,
                       linestyle='--',
                       label=rf'Mean = {mean_kl:.3f}')
            _safe_legend(ax, loc='upper right', framealpha=0.9)
        else:
            _no_data_axes(ax, 'No finite non-negative KL values')
    else:
        _no_data_axes(ax, r'No $KL$ data')
    ax.set_xlabel(r'$KL(q_i \| \Omega_{ij} q_j)$')
    ax.set_ylabel('Density')
    ax.set_title(r'(b) KL divergence distribution')

    # ---- (c) Dirichlet energy per token pair ----
    ax = axes[2]
    fe_data = invariants.get('field_energy') if invariants else None
    if fe_data is not None and len(fe_data) > 0:
        fe_arr = np.asarray(fe_data, dtype=float)
        fe_arr = fe_arr[np.isfinite(fe_arr) & (fe_arr >= 0)]
        if len(fe_arr) > 0:
            ax.hist(
                fe_arr, bins=40, density=True,
                color=PUB_COLORS['green'], alpha=0.8,
                edgecolor='white', linewidth=0.4,
            )
            mean_fe = float(np.mean(fe_arr))
            ax.axvline(mean_fe, color=PUB_COLORS['red'], linewidth=1.2,
                       linestyle='--',
                       label=rf'Mean = {mean_fe:.3f}')
            _safe_legend(ax, loc='upper right', framealpha=0.9)
        else:
            _no_data_axes(ax, 'No finite non-negative field energy values')
    else:
        _no_data_axes(ax, r'No field energy data')
    ax.set_xlabel(r'$\|\phi_i - \phi_j\|^2$')
    ax.set_ylabel('Density')
    ax.set_title('(c) Dirichlet energy per token pair')

    fig.suptitle('Gauge-Invariant Diagnostics', fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =============================================================================
# Figure 6 — Gauge Orbit Structure (PCA Projection)
# =============================================================================

def plot_gauge_orbit_pca(
    orbit_samples: List[Dict[str, np.ndarray]],
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
) -> Optional["Figure"]:
    r"""PCA projection of gauge orbit samples in the phi parameter space.

    A gauge orbit is the set of models related by a gauge transformation
    :math:`\phi \mapsto \phi + \xi` for some :math:`\xi \in \mathfrak{g}`.
    Sampling around the trained model and projecting the stacked
    :math:`\phi` matrices (flattened to vectors) into 2D via PCA visualises
    the local orbit geometry.

    The original model (first sample or sample marked ``'is_original': True``)
    is highlighted with a star marker.  Each sample is coloured sequentially
    using the publication colour cycle.

    Args:
        orbit_samples: List of dicts.  Each dict must contain:
            - ``'phi'``: ``(N, n_gen)`` array of gauge frame parameters.
            - Optionally ``'mu'``: ``(N, K)`` mean vectors (not used for PCA
              but preserved for future extensions).
            - Optionally ``'is_original'``: bool flag.
        save_path: If given, save the figure here.
        figsize: Figure dimensions in inches.

    Returns:
        The :class:`matplotlib.figure.Figure` or ``None`` if matplotlib is
        unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    set_pub_style()

    if not orbit_samples:
        fig, ax = plt.subplots(figsize=figsize)
        _no_data_axes(ax, 'No orbit samples provided')
        ax.set_title('Gauge Orbit Structure (PCA Projection)')
        fig.tight_layout()
        _save(fig, save_path)
        return fig

    # Flatten each sample's phi matrix to a single feature vector
    vectors: List[np.ndarray] = []
    original_idx: Optional[int] = None
    for idx, sample in enumerate(orbit_samples):
        phi = sample.get('phi')
        if phi is None:
            continue
        phi_arr = np.asarray(phi, dtype=float)
        vectors.append(phi_arr.flatten())
        if sample.get('is_original', False) or (original_idx is None and idx == 0):
            original_idx = len(vectors) - 1

    if len(vectors) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        _no_data_axes(ax, "No 'phi' arrays found in orbit_samples")
        ax.set_title('Gauge Orbit Structure (PCA Projection)')
        fig.tight_layout()
        _save(fig, save_path)
        return fig

    # Pad to equal length (in case samples have different N)
    max_len = max(v.shape[0] for v in vectors)
    padded = np.zeros((len(vectors), max_len), dtype=float)
    for i, v in enumerate(vectors):
        padded[i, : v.shape[0]] = v

    # Manual PCA (avoids sklearn dependency)
    X = padded - padded.mean(axis=0, keepdims=True)
    if X.shape[0] < 2 or X.shape[1] < 2:
        # Degenerate: cannot do PCA, fall back to plotting first two dims
        coords_2d = padded[:, :2] if padded.shape[1] >= 2 else np.hstack(
            [padded, np.zeros((padded.shape[0], 2 - padded.shape[1]))]
        )
        pc_var = np.array([np.nan, np.nan])
    else:
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            coords_2d = U[:, :2] * S[:2]  # (n_samples, 2)
            explained = S ** 2 / (S ** 2).sum()
            pc_var = explained[:2]
        except np.linalg.LinAlgError:
            coords_2d = X[:, :2]
            pc_var = np.array([np.nan, np.nan])

    fig, ax = plt.subplots(figsize=figsize)

    n_samples = coords_2d.shape[0]
    cmap = plt.cm.get_cmap('viridis', max(n_samples, 2))  # type: ignore[attr-defined]

    for i in range(n_samples):
        color = cmap(i / max(n_samples - 1, 1))
        if i == original_idx:
            continue  # Plot original on top after the loop
        ax.scatter(
            coords_2d[i, 0], coords_2d[i, 1],
            s=50, color=color, alpha=0.75, zorder=3,
            linewidths=0.3, edgecolors='white',
        )

    # Overlay the original model with a large star
    if original_idx is not None:
        ax.scatter(
            coords_2d[original_idx, 0], coords_2d[original_idx, 1],
            s=220, marker='*', color=PUB_COLORS['red'],
            zorder=5, label='Original model',
            linewidths=0.5, edgecolors='white',
        )

    # Colourbar for sample index
    sm = plt.cm.ScalarMappable(cmap=cmap,  # type: ignore[attr-defined]
                                norm=plt.Normalize(0, n_samples - 1))  # type: ignore[attr-defined]
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Sample index', fontsize=9)

    xlab = (
        rf'PC 1 ({100*pc_var[0]:.1f}% var.)'
        if np.isfinite(pc_var[0])
        else 'PC 1'
    )
    ylab = (
        rf'PC 2 ({100*pc_var[1]:.1f}% var.)'
        if len(pc_var) > 1 and np.isfinite(pc_var[1])
        else 'PC 2'
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title('Gauge Orbit Structure (PCA Projection)')
    _safe_legend(ax, loc='upper right', framealpha=0.9)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =============================================================================
# Batch generator
# =============================================================================

def generate_all_gauge_geometry_figures(
    data: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    r"""Generate and save all gauge geometry figures from a data dictionary.

    Expected keys in *data* (all optional; missing keys produce placeholder
    figures rather than errors):

    - ``'ym_steps'``: list of int — training step indices.
    - ``'ym_energies'``: list of float — Yang-Mills energy per step.
    - ``'ym_dirichlet_energies'``: list of float — Dirichlet energy per step
      (optional overlay on the Yang-Mills figure).
    - ``'curvature_F'``: ``(n_triples, K, K)`` numpy array of curvature
      tensors.
    - ``'phi_diff_sq'``: ``(N, N)`` numpy array of squared gauge-frame
      distances.
    - ``'beta'``: ``(N, N)`` numpy array of attention weights.
    - ``'token_labels'``: optional list of strings for axis tick labels.
    - ``'invariants'``: dict with optional keys ``'det_omega'``,
      ``'kl_values'``, ``'field_energy'`` (each a 1-D numpy array).
    - ``'orbit_samples'``: list of dicts (see :func:`plot_gauge_orbit_pca`).

    Args:
        data: Dictionary of pre-computed gauge geometry quantities.
        output_dir: Directory in which to write ``*.png`` and ``*.pdf``
            figures.  Created automatically if it does not exist.

    Returns:
        Dict mapping figure stem name to the saved file path (PNG).  PDF
        variants are also saved and appear in the returned dict under the
        key ``'<name>_pdf'``.
    """
    if not MATPLOTLIB_AVAILABLE:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if data is None:
        data = {}

    figures = [
        (
            'yang_mills_evolution',
            lambda p: plot_yang_mills_evolution(
                steps=data.get('ym_steps', []),
                energies=data.get('ym_energies', []),
                dirichlet_energies=data.get('ym_dirichlet_energies'),
                save_path=p,
            ),
        ),
        (
            'curvature_spectrum',
            lambda p: plot_curvature_spectrum(
                F=data.get('curvature_F', np.empty((0, 1, 1))),
                save_path=p,
            ),
        ),
        (
            'curvature_decomposition',
            lambda p: plot_curvature_decomposition(
                F=data.get('curvature_F', np.empty((0, 1, 1))),
                save_path=p,
            ),
        ),
        (
            'gauge_field_energy_map',
            lambda p: plot_gauge_field_energy_map(
                phi_diff_sq=data.get('phi_diff_sq'),
                beta=data.get('beta'),
                token_labels=data.get('token_labels'),
                save_path=p,
            ),
        ),
        (
            'gauge_invariant_scatter',
            lambda p: plot_gauge_invariant_scatter(
                invariants=data.get('invariants', {}),
                save_path=p,
            ),
        ),
        (
            'gauge_orbit_pca',
            lambda p: plot_gauge_orbit_pca(
                orbit_samples=data.get('orbit_samples', []),
                save_path=p,
            ),
        ),
    ]

    saved: Dict[str, Path] = {}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for name, plot_fn in figures:
            for ext in ('png', 'pdf'):
                out_path = output_dir / f'{name}.{ext}'
                key = name if ext == 'png' else f'{name}_pdf'
                try:
                    fig = plot_fn(out_path)
                    if fig is not None:
                        plt.close(fig)
                        saved[key] = out_path
                except (ValueError, TypeError, np.linalg.LinAlgError):
                    pass  # Missing data is expected; silently skip

    return saved
