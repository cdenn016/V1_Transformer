"""
Fiber Trajectory Visualization
================================

Publication-quality figures for intra-fiber VFE trajectory analysis.
Visualizes how beliefs evolve through the Gaussian manifold during
E-step iterations, using Fisher-Rao geometry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    gridspec = None
    Figure = None
    LineCollection = None
    MATPLOTLIB_AVAILABLE = False

from transformer.visualization.pub_style import set_pub_style, PUB_COLORS, PUB_CYCLE, _safe_legend


# ---------------------------------------------------------------------------
# Type aliases (avoid torch dependency — analysis types carry numpy arrays)
# ---------------------------------------------------------------------------

try:
    from transformer.analysis.trajectory import IterationSnapshot
    from transformer.analysis.fiber_trajectory import FiberTrajectoryStats
except ImportError:
    # Graceful degradation: allow the module to be imported without the
    # analysis sub-package installed (e.g., in lightweight environments).
    IterationSnapshot = None  # type: ignore[assignment,misc]
    FiberTrajectoryStats = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _fisher_rao_coords(
    mu: np.ndarray,
    sigma_diag: np.ndarray,
) -> np.ndarray:
    r"""Map a Gaussian belief state to Fisher-Rao-scaled natural coordinates.

    For each dimension :math:`k` constructs the two-component embedding

    .. math::

        v_k = \left[
            \frac{\mu_k}{\sigma_k},\;
            \sqrt{2}\,\log\sigma_k
        \right]

    where :math:`\sigma_k = \sqrt{\text{sigma\_diag}_k}` is the standard
    deviation.  The full vector is of length :math:`2K` and encodes both the
    mean displacement in units of standard deviation and the log-scale of the
    covariance.

    Args:
        mu: (K,) mean vector for one token at one snapshot.
        sigma_diag: (K,) diagonal variance vector (not std dev).

    Returns:
        (2K,) Fisher-Rao-scaled coordinate vector.
    """
    std = np.sqrt(np.maximum(sigma_diag, 1e-24))  # (K,)
    coord_mu = mu / std                            # (K,) — mean in std-dev units
    coord_log_sig = np.sqrt(2.0) * np.log(std)    # (K,) — log-scale
    return np.concatenate([coord_mu, coord_log_sig])  # (2K,)


def _build_trajectory_matrix(
    snapshots: List,
    token_row: int,
) -> np.ndarray:
    r"""Construct the (T, 2K) Fisher-Rao coordinate matrix for one token.

    Args:
        snapshots: Ordered list of :class:`IterationSnapshot` objects.
        token_row: Row index into each snapshot's ``mu`` / ``sigma_diag``
            arrays.

    Returns:
        (T, 2K) array of Fisher-Rao coordinates, one row per snapshot.
    """
    rows = []
    for snap in snapshots:
        mu = snap.mu[token_row]           # (K,)
        sig = snap.sigma_diag[token_row]  # (K,)
        rows.append(_fisher_rao_coords(mu, sig))
    return np.array(rows)  # (T, 2K)


def _pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""PCA projection of a data matrix to 2 components.

    Centers the data, computes the top-2 principal components via SVD, and
    returns the projected coordinates together with the component vectors.

    Args:
        X: (N, D) data matrix.

    Returns:
        Tuple of:
            - (N, 2) projected coordinates
            - (2, D) principal component vectors (rows)
            - (2,) explained variance (eigenvalues)
    """
    X_c = X - X.mean(axis=0, keepdims=True)
    if X_c.shape[0] < 2:
        # Single point — return zeros
        return np.zeros((X_c.shape[0], 2)), np.zeros((2, X_c.shape[1])), np.zeros(2)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    n_comp = min(2, Vt.shape[0])
    proj = X_c @ Vt[:n_comp].T  # (N, 2)
    variance = (S[:n_comp] ** 2) / max(X_c.shape[0] - 1, 1)
    # Pad to (N, 2) if rank < 2
    if proj.shape[1] < 2:
        pad = np.zeros((proj.shape[0], 2 - proj.shape[1]))
        proj = np.concatenate([proj, pad], axis=1)
        variance = np.concatenate([variance, np.zeros(2 - len(variance))])
        Vt_pad = np.zeros((2, Vt.shape[1]))
        Vt_pad[: Vt.shape[0]] = Vt[: min(2, Vt.shape[0])]
        Vt = Vt_pad
    return proj, Vt[:2], variance


def _compute_velocities(
    snapshots: List,
    token_row: int,
) -> np.ndarray:
    r"""Per-step Fisher-Rao velocity for one token using the midpoint metric.

    Computes

    .. math::

        v_t = \sqrt{\sum_k \left[
            \frac{(\Delta\mu_k)^2}{{\bar\sigma_k^2}}
            + \frac{(\Delta\sigma_k^2)^2}{2\,\bar\sigma_k^4}
        \right]}

    where :math:`\bar\sigma_k = \tfrac12(\sigma_{t,k} + \sigma_{t+1,k})` is
    the midpoint standard deviation and :math:`\Delta\sigma_k^2` is the step
    change in variance.

    Args:
        snapshots: Ordered list of :class:`IterationSnapshot`.
        token_row: Token row index.

    Returns:
        (T-1,) array of velocities.  Empty array for single-snapshot input.
    """
    T = len(snapshots)
    if T < 2:
        return np.array([], dtype=np.float64)
    velocities = np.empty(T - 1)
    for t in range(T - 1):
        mu1 = snapshots[t].mu[token_row]             # (K,)
        sig1 = snapshots[t].sigma_diag[token_row]    # (K,) variances
        mu2 = snapshots[t + 1].mu[token_row]
        sig2 = snapshots[t + 1].sigma_diag[token_row]
        std1 = np.sqrt(np.maximum(sig1, 1e-24))
        std2 = np.sqrt(np.maximum(sig2, 1e-24))
        std_mid = 0.5 * (std1 + std2)
        std_mid = np.maximum(std_mid, 1e-12)
        d_mu = mu2 - mu1
        d_var = sig2 - sig1
        term_mu = (d_mu ** 2) / (std_mid ** 2)
        term_var = (d_var ** 2) / (2.0 * std_mid ** 4)
        velocities[t] = np.sqrt(np.sum(term_mu + term_var))
    return velocities


def _compute_convergence_curve(
    snapshots: List,
    token_row: int,
) -> np.ndarray:
    r"""Fisher-Rao distance from each snapshot state to the final state.

    Args:
        snapshots: Ordered list of :class:`IterationSnapshot`.
        token_row: Token row index.

    Returns:
        (T,) distances; last entry is 0 by definition.
    """
    T = len(snapshots)
    if T == 1:
        return np.array([0.0])
    mu_f = snapshots[-1].mu[token_row]
    sig_f = snapshots[-1].sigma_diag[token_row]
    curve = np.empty(T)
    for t in range(T):
        mu_t = snapshots[t].mu[token_row]
        sig_t = snapshots[t].sigma_diag[token_row]
        # Closed-form Fisher-Rao (Atkinson & Mitchell 1981) per dimension
        std_t = np.sqrt(np.maximum(sig_t, 1e-24))
        std_f = np.sqrt(np.maximum(sig_f, 1e-24))
        d_mu = mu_f - mu_t
        d_std = std_f - std_t
        denom = 2.0 * std_t * std_f
        denom = np.maximum(denom, 1e-24)
        arg = 1.0 + (d_mu ** 2 + d_std ** 2) / denom
        arg = np.maximum(arg, 1.0)
        d_k = np.sqrt(2.0) * np.arccosh(arg)
        curve[t] = np.sqrt(np.sum(d_k ** 2))
    return curve


def _fit_log_linear(
    y: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    r"""Ordinary least-squares log-linear fit :math:`\log y \approx a - \lambda t`.

    Args:
        y: (T,) positive-valued convergence curve.

    Returns:
        Tuple of:
            - (T,) fitted curve values (in original space)
            - :math:`\lambda` (decay rate, clipped to :math:`\ge 0`)
            - :math:`a` (log-intercept)
    """
    T = len(y)
    t = np.arange(T, dtype=np.float64)
    mask = y > 0.0
    if mask.sum() < 2:
        return np.full(T, np.nan), 0.0, 0.0
    t_fit = t[mask]
    log_y = np.log(y[mask])
    A = np.column_stack([np.ones_like(t_fit), t_fit])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    log_a, neg_lam = coeffs
    lam = max(-neg_lam, 0.0)
    fit_vals = np.exp(log_a) * np.exp(-lam * t)
    return fit_vals, lam, float(log_a)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_fiber_phase_portrait(
    snapshots: List,
    token_indices: List[int],
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> Optional["Figure"]:
    r"""PCA phase portrait of VFE fiber trajectories in Fisher-Rao coordinates.

    Projects each snapshot's belief state into the Fisher-Rao-scaled coordinate
    space :math:`[{\mu_k}/{\sigma_k},\, \sqrt{2}\log\sigma_k]` over all
    dimensions :math:`k`, then reduces to 2D via PCA computed jointly across
    all requested token trajectories.  This gives a single shared embedding
    space so that trajectories can be compared geometrically.

    Direction-of-travel arrows are added every few steps.  Velocity is encoded
    as line opacity: fast steps are more opaque, slow steps are more transparent.

    Args:
        snapshots: Ordered list of :class:`IterationSnapshot` objects from
            consecutive VFE E-step iterations.
        token_indices: Indices into the N_recorded axis to plot.
        save_path: If provided, save figure to this path at 300 DPI.
        figsize: Figure dimensions in inches.

    Returns:
        :class:`matplotlib.figure.Figure`, or ``None`` if matplotlib is
        unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    if not snapshots or not token_indices:
        return None
    set_pub_style()

    T = len(snapshots)

    # --- Build joint PCA ---
    all_coords: List[np.ndarray] = []
    for tok in token_indices:
        try:
            traj = _build_trajectory_matrix(snapshots, tok)  # (T, 2K)
            all_coords.append(traj)
        except (IndexError, Exception):
            continue
    if not all_coords:
        return None

    joint = np.concatenate(all_coords, axis=0)  # (len(tokens)*T, 2K)
    _, pca_vt, _ = _pca_2d(joint)
    joint_c = joint - joint.mean(axis=0, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)

    for tok_i, tok in enumerate(token_indices):
        try:
            traj = _build_trajectory_matrix(snapshots, tok)  # (T, 2K)
        except (IndexError, Exception):
            continue

        traj_c = traj - joint.mean(axis=0, keepdims=True)
        proj = traj_c @ pca_vt.T  # (T, 2)

        color = PUB_CYCLE[tok_i % len(PUB_CYCLE)]
        label = f"Token {tok}"

        # Compute per-step velocities for alpha encoding
        vels = _compute_velocities(snapshots, tok)  # (T-1,)
        if len(vels) > 0 and vels.max() > 0.0:
            alpha_vals = 0.25 + 0.75 * (vels / vels.max())
        else:
            alpha_vals = np.full(max(T - 1, 1), 0.7)

        # Draw segments with velocity-encoded alpha
        for s in range(T - 1):
            seg_alpha = float(np.clip(alpha_vals[s] if s < len(alpha_vals) else 0.7, 0.15, 1.0))
            ax.plot(
                proj[s : s + 2, 0],
                proj[s : s + 2, 1],
                color=color,
                alpha=seg_alpha,
                linewidth=1.5,
                solid_capstyle="round",
            )

        # Direction arrows (every ~T//5 steps, minimum 1)
        arrow_step = max(1, T // 5)
        for s in range(0, T - 1, arrow_step):
            dx = proj[s + 1, 0] - proj[s, 0]
            dy = proj[s + 1, 1] - proj[s, 1]
            if abs(dx) + abs(dy) > 1e-10:
                ax.annotate(
                    "",
                    xy=(proj[s, 0] + 0.6 * dx, proj[s, 1] + 0.6 * dy),
                    xytext=(proj[s, 0] + 0.4 * dx, proj[s, 1] + 0.4 * dy),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        lw=1.2,
                    ),
                )

        # Start: circle; End: star
        ax.plot(proj[0, 0], proj[0, 1], "o", color=color, markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5,
                label=label)
        ax.plot(proj[-1, 0], proj[-1, 1], "*", color=color, markersize=10,
                markeredgecolor="white", markeredgewidth=0.6, zorder=5)

    ax.set_xlabel("PC1 (Fisher-Rao scaled)")
    ax.set_ylabel("PC2 (Fisher-Rao scaled)")
    ax.set_title("VFE Fiber Trajectory Phase Portrait")
    _safe_legend(ax, loc="best", framealpha=0.9)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_velocity_profiles(
    snapshots: List,
    token_indices: List[int],
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Optional["Figure"]:
    r"""Fisher-Rao velocity profile over VFE iterations.

    Plots the per-step Fisher-Rao speed :math:`\|ds/dt\|` as defined by the
    midpoint metric approximation:

    .. math::

        v_t = \sqrt{\sum_k \left[
            \frac{(\Delta\mu_k)^2}{\bar\sigma_k^2}
            + \frac{(\Delta\sigma_k^2)^2}{2\,\bar\sigma_k^4}
        \right]}

    on a log-scaled Y axis so that exponential deceleration (the typical
    convergence signature) appears as a straight line.

    Args:
        snapshots: Ordered list of :class:`IterationSnapshot` objects.
        token_indices: Token row indices to plot.
        save_path: Save path for PNG/PDF output.
        figsize: Figure dimensions in inches.

    Returns:
        :class:`matplotlib.figure.Figure`, or ``None`` if matplotlib is
        unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    if not snapshots or not token_indices:
        return None
    set_pub_style()

    fig, ax = plt.subplots(figsize=figsize)

    any_data = False
    for tok_i, tok in enumerate(token_indices):
        try:
            vels = _compute_velocities(snapshots, tok)
        except (IndexError, Exception):
            continue
        if len(vels) == 0:
            continue
        any_data = True
        color = PUB_CYCLE[tok_i % len(PUB_CYCLE)]
        iters = np.arange(1, len(vels) + 1)
        ax.plot(iters, vels, color=color, linewidth=1.5, label=f"Token {tok}")

    if not any_data:
        ax.text(0.5, 0.5, "No velocity data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fisher-Rao velocity")
    ax.set_yscale("log")
    ax.set_title("Fisher-Rao Velocity Profile")
    _safe_legend(ax, loc="best", framealpha=0.9)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_convergence_curves(
    snapshots: List,
    token_indices: List[int],
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Optional["Figure"]:
    r"""Fisher-Rao distance to fixed point over VFE iterations.

    Plots :math:`c(t) = d_{\mathrm{FR}}(q_t, q_T)` on a log scale and
    overlays a dashed exponential fit :math:`A e^{-\lambda t}` for each
    token.  The fitted decay rate :math:`\lambda` appears in the legend.

    Args:
        snapshots: Ordered list of :class:`IterationSnapshot` objects.
        token_indices: Token row indices to plot.
        save_path: Save path for PNG/PDF output.
        figsize: Figure dimensions in inches.

    Returns:
        :class:`matplotlib.figure.Figure`, or ``None`` if matplotlib is
        unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    if not snapshots or not token_indices:
        return None
    set_pub_style()

    T = len(snapshots)
    iters = np.arange(T)

    fig, ax = plt.subplots(figsize=figsize)

    any_data = False
    for tok_i, tok in enumerate(token_indices):
        try:
            curve = _compute_convergence_curve(snapshots, tok)
        except (IndexError, Exception):
            continue
        if len(curve) == 0:
            continue
        any_data = True
        color = PUB_CYCLE[tok_i % len(PUB_CYCLE)]
        # Exclude the final point: d_FR(q_T, q_T) = 0 by definition,
        # which is -inf on log scale and creates a visual spike artifact.
        n_plot = len(curve) - 1 if len(curve) > 1 else len(curve)
        t_plot = np.arange(n_plot)
        curve_plot = curve[:n_plot]
        ax.plot(t_plot, curve_plot, color=color, linewidth=1.5,
                label=f"Token {tok}")

        # Dashed exponential fit overlay (also exclude final zero)
        try:
            fit_vals, lam, _ = _fit_log_linear(curve_plot)
            if np.any(np.isfinite(fit_vals)):
                ax.plot(t_plot, fit_vals, color=color, linewidth=1.0,
                        linestyle="--", alpha=0.7,
                        label=rf"$\lambda={lam:.2f}$ (fit)")
        except (ValueError, np.linalg.LinAlgError):
            pass

    if not any_data:
        ax.text(0.5, 0.5, "No convergence data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$d_{\mathrm{FR}}(q_t,\,q_T)$")
    ax.set_yscale("log")
    ax.set_title("Convergence to Fixed Point")
    _safe_legend(ax, loc="best", framealpha=0.9)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_geodesic_deviation(
    stats_list: List,
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Optional["Figure"]:
    r"""Bar chart of geodesic deviation ratios :math:`\rho = L(T) / d_{\mathrm{FR}}`.

    Colors bars by deviation magnitude:

    - green for :math:`\rho < 1.2` (near-geodesic)
    - orange for :math:`1.2 \le \rho \le 2.0` (moderate detour)
    - red for :math:`\rho > 2.0` (heavy oscillation / overshoot)

    A horizontal dashed reference line at :math:`\rho = 1.0` marks the
    geodesic bound.

    Args:
        stats_list: List of :class:`FiberTrajectoryStats` objects, one per
            token.
        save_path: Save path for PNG/PDF output.
        figsize: Figure dimensions in inches.

    Returns:
        :class:`matplotlib.figure.Figure`, or ``None`` if matplotlib is
        unavailable or the stats list is empty.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    if not stats_list:
        return None
    set_pub_style()

    token_positions = [s.token_idx for s in stats_list]
    ratios = [s.deviation_ratio for s in stats_list]

    bar_colors = []
    for r in ratios:
        if r < 1.2:
            bar_colors.append(PUB_COLORS["green"])
        elif r <= 2.0:
            bar_colors.append(PUB_COLORS["orange"])
        else:
            bar_colors.append(PUB_COLORS["red"])

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(stats_list))
    ax.bar(x, ratios, color=bar_colors, width=0.7, edgecolor="white",
           linewidth=0.5, zorder=2)
    ax.axhline(1.0, color=PUB_COLORS["black"], linewidth=1.2,
               linestyle="--", label="Geodesic reference ($\\rho=1$)", zorder=3)

    # Color-coded legend proxies
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PUB_COLORS["green"],  label=r"$\rho < 1.2$"),
        Patch(facecolor=PUB_COLORS["orange"], label=r"$1.2 \leq \rho \leq 2.0$"),
        Patch(facecolor=PUB_COLORS["red"],    label=r"$\rho > 2.0$"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in token_positions], rotation=45, ha="right")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Arc length / Geodesic distance")
    ax.set_title("Geodesic Deviation Ratio")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_arc_length_heatmap(
    arc_lengths_by_layer: np.ndarray,
    token_labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 4),
) -> Optional["Figure"]:
    r"""Heatmap of Fisher-Rao arc lengths per token per layer.

    Provides a compact view of how much "inferential work" the VFE E-step
    performs at each token position across transformer layers.  Longer arc
    lengths indicate larger belief updates in Fisher-Rao geometry.

    Args:
        arc_lengths_by_layer: (n_layers, n_tokens) array of arc lengths.
        token_labels: Optional list of token label strings for the X axis.
            If ``None``, integer positions are used.
        save_path: Save path for PNG/PDF output.
        figsize: Figure dimensions in inches.

    Returns:
        :class:`matplotlib.figure.Figure`, or ``None`` if matplotlib is
        unavailable or the array is empty.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    arc_lengths_by_layer = np.asarray(arc_lengths_by_layer)
    if arc_lengths_by_layer.ndim != 2 or arc_lengths_by_layer.size == 0:
        return None
    set_pub_style()

    n_layers, n_tokens = arc_lengths_by_layer.shape

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        arc_lengths_by_layer,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Fisher-Rao arc length", fontsize=10)

    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([str(i) for i in range(n_layers)], fontsize=8)
    ax.set_ylabel("Layer index")

    if token_labels is not None and len(token_labels) == n_tokens:
        ax.set_xticks(np.arange(n_tokens))
        ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
    else:
        tick_step = max(1, n_tokens // 16)
        ax.set_xticks(np.arange(0, n_tokens, tick_step))
        ax.set_xticklabels(
            [str(i) for i in range(0, n_tokens, tick_step)], fontsize=8
        )
    ax.set_xlabel("Token position")
    ax.set_title("Inferential Work per Token per Layer")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_fiber_trajectory_dashboard(
    snapshots: List,
    stats_list: List,
    token_indices: List[int],
    save_path: Optional[Path] = None,
) -> Optional["Figure"]:
    r"""Six-panel dashboard summarizing fiber trajectory analysis.

    Panel layout (2 rows, 3 columns):

    +--------------------+--------------------+--------------------+
    | Phase portrait     | Velocity profiles  | Convergence curves |
    +--------------------+--------------------+--------------------+
    | Arc length bars    | Geodesic deviation | mu disp vs sig     |
    |                    | bar chart          | ratio scatter      |
    +--------------------+--------------------+--------------------+

    Args:
        snapshots: Ordered list of :class:`IterationSnapshot` objects.
        stats_list: List of :class:`FiberTrajectoryStats` objects for the
            token positions corresponding to ``token_indices``.
        token_indices: Token row indices recorded in the snapshots.
        save_path: Save path for PNG/PDF output.

    Returns:
        :class:`matplotlib.figure.Figure`, or ``None`` if matplotlib is
        unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    if not snapshots:
        return None
    set_pub_style()

    T = len(snapshots)
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35, figure=fig)

    # ---- (0,0) Phase portrait ------------------------------------------------
    ax00 = fig.add_subplot(gs[0, 0])
    _draw_phase_portrait_on_ax(ax00, snapshots, token_indices)

    # ---- (0,1) Velocity profiles ---------------------------------------------
    ax01 = fig.add_subplot(gs[0, 1])
    any_vel = False
    for tok_i, tok in enumerate(token_indices):
        try:
            vels = _compute_velocities(snapshots, tok)
        except (IndexError, Exception):
            continue
        if len(vels) == 0:
            continue
        any_vel = True
        color = PUB_CYCLE[tok_i % len(PUB_CYCLE)]
        iters = np.arange(1, len(vels) + 1)
        ax01.plot(iters, vels, color=color, linewidth=1.5, label=f"Token {tok}")
    if any_vel:
        ax01.set_yscale("log")
        _safe_legend(ax01, loc="best", framealpha=0.9, fontsize=7)
    else:
        ax01.text(0.5, 0.5, "No data", transform=ax01.transAxes,
                  ha="center", va="center", color="gray")
    ax01.set_xlabel("Iteration")
    ax01.set_ylabel("Fisher-Rao velocity")
    ax01.set_title("Velocity Profiles")

    # ---- (0,2) Convergence curves --------------------------------------------
    ax02 = fig.add_subplot(gs[0, 2])
    any_conv = False
    for tok_i, tok in enumerate(token_indices):
        try:
            curve = _compute_convergence_curve(snapshots, tok)
        except (IndexError, Exception):
            continue
        if len(curve) == 0:
            continue
        any_conv = True
        color = PUB_CYCLE[tok_i % len(PUB_CYCLE)]
        # Exclude final point (d_FR(q_T,q_T)=0 → -inf on log scale)
        n_plot = len(curve) - 1 if len(curve) > 1 else len(curve)
        t_plot = np.arange(n_plot)
        curve_plot = curve[:n_plot]
        ax02.plot(t_plot, curve_plot, color=color, linewidth=1.5, label=f"Token {tok}")
        try:
            fit_vals, lam, _ = _fit_log_linear(curve_plot)
            if np.any(np.isfinite(fit_vals)):
                ax02.plot(t_plot, fit_vals, color=color, linewidth=0.9,
                          linestyle="--", alpha=0.7)
        except (ValueError, np.linalg.LinAlgError):
            pass
    if any_conv:
        ax02.set_yscale("log")
        _safe_legend(ax02, loc="best", framealpha=0.9, fontsize=7)
    else:
        ax02.text(0.5, 0.5, "No data", transform=ax02.transAxes,
                  ha="center", va="center", color="gray")
    ax02.set_xlabel("Iteration")
    ax02.set_ylabel(r"$d_{\mathrm{FR}}(q_t,\,q_T)$")
    ax02.set_title("Convergence to Fixed Point")

    # ---- (1,0) Arc length bar chart ------------------------------------------
    ax10 = fig.add_subplot(gs[1, 0])
    if stats_list:
        toks = [s.token_idx for s in stats_list]
        arcs = [s.arc_length for s in stats_list]
        x = np.arange(len(stats_list))
        colors_arc = [PUB_CYCLE[i % len(PUB_CYCLE)] for i in range(len(stats_list))]
        ax10.bar(x, arcs, color=colors_arc, width=0.7, edgecolor="white", linewidth=0.5)
        ax10.set_xticks(x)
        ax10.set_xticklabels([str(t) for t in toks], rotation=45, ha="right", fontsize=7)
        ax10.set_ylabel("Fisher-Rao arc length")
        ax10.set_title("Arc Length per Token")
    else:
        ax10.text(0.5, 0.5, "No stats", transform=ax10.transAxes,
                  ha="center", va="center", color="gray")
    ax10.set_xlabel("Token position")

    # ---- (1,1) Geodesic deviation bars ---------------------------------------
    ax11 = fig.add_subplot(gs[1, 1])
    if stats_list:
        toks_dev = [s.token_idx for s in stats_list]
        ratios = [s.deviation_ratio for s in stats_list]
        bar_colors = []
        for r in ratios:
            if r < 1.2:
                bar_colors.append(PUB_COLORS["green"])
            elif r <= 2.0:
                bar_colors.append(PUB_COLORS["orange"])
            else:
                bar_colors.append(PUB_COLORS["red"])
        x = np.arange(len(stats_list))
        ax11.bar(x, ratios, color=bar_colors, width=0.7, edgecolor="white",
                 linewidth=0.5, zorder=2)
        ax11.axhline(1.0, color=PUB_COLORS["black"], linewidth=1.0,
                     linestyle="--", zorder=3)
        ax11.set_xticks(x)
        ax11.set_xticklabels([str(t) for t in toks_dev], rotation=45, ha="right",
                              fontsize=7)
        ax11.set_ylabel("Arc length / Geodesic distance")
        ax11.set_title("Geodesic Deviation Ratio")
    else:
        ax11.text(0.5, 0.5, "No stats", transform=ax11.transAxes,
                  ha="center", va="center", color="gray")
    ax11.set_xlabel("Token position")

    # ---- (1,2) mu displacement vs sigma ratio scatter ------------------------
    ax12 = fig.add_subplot(gs[1, 2])
    if stats_list:
        mu_disps = np.array([s.mu_displacement for s in stats_list])
        sig_ratios = np.array([s.sigma_ratio for s in stats_list])
        arc_sizes = np.array([s.arc_length for s in stats_list])
        arc_norm = arc_sizes / (arc_sizes.max() + 1e-12)
        scatter_colors = [PUB_CYCLE[i % len(PUB_CYCLE)] for i in range(len(stats_list))]
        sc = ax12.scatter(
            mu_disps,
            sig_ratios,
            c=scatter_colors,
            s=30 + 100 * arc_norm,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax12.axhline(1.0, color=PUB_COLORS["gray"], linewidth=0.8,
                     linestyle=":", zorder=2)
        ax12.set_xlabel(r"$\|\mu_T - \mu_0\|_2$ (Euclidean displacement)")
        ax12.set_ylabel(r"Geometric mean $\sigma$ ratio $(\sigma_T/\sigma_0)$")
        ax12.set_title(r"$\mu$ Displacement vs $\sigma$ Ratio")
    else:
        ax12.text(0.5, 0.5, "No stats", transform=ax12.transAxes,
                  ha="center", va="center", color="gray")

    fig.suptitle("Fiber Trajectory Analysis Dashboard", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Internal helper: draw phase portrait content onto a given Axes
# ---------------------------------------------------------------------------

def _draw_phase_portrait_on_ax(
    ax: "plt.Axes",
    snapshots: List,
    token_indices: List[int],
) -> None:
    r"""Draw the Fisher-Rao PCA phase portrait into an existing Axes object.

    Factored out so that it can be embedded inside dashboard panels without
    creating a redundant figure.  Arguments mirror
    :func:`plot_fiber_phase_portrait`.

    Args:
        ax: The target :class:`matplotlib.axes.Axes`.
        snapshots: Ordered list of :class:`IterationSnapshot` objects.
        token_indices: Token row indices to plot.
    """
    if not snapshots or not token_indices:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        return

    T = len(snapshots)
    all_coords: List[np.ndarray] = []
    valid_tokens: List[int] = []
    for tok in token_indices:
        try:
            traj = _build_trajectory_matrix(snapshots, tok)
            all_coords.append(traj)
            valid_tokens.append(tok)
        except (IndexError, Exception):
            continue

    if not all_coords:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        return

    joint = np.concatenate(all_coords, axis=0)
    _, pca_vt, _ = _pca_2d(joint)
    joint_mean = joint.mean(axis=0, keepdims=True)

    for tok_i, (tok, traj) in enumerate(zip(valid_tokens, all_coords)):
        traj_c = traj - joint_mean
        proj = traj_c @ pca_vt.T  # (T, 2)
        color = PUB_CYCLE[tok_i % len(PUB_CYCLE)]

        vels = _compute_velocities(snapshots, tok)
        if len(vels) > 0 and vels.max() > 0.0:
            alpha_vals = 0.25 + 0.75 * (vels / vels.max())
        else:
            alpha_vals = np.full(max(T - 1, 1), 0.7)

        for s in range(T - 1):
            seg_alpha = float(np.clip(
                alpha_vals[s] if s < len(alpha_vals) else 0.7, 0.15, 1.0
            ))
            ax.plot(proj[s : s + 2, 0], proj[s : s + 2, 1],
                    color=color, alpha=seg_alpha, linewidth=1.2)

        ax.plot(proj[0, 0], proj[0, 1], "o", color=color, markersize=5,
                markeredgecolor="white", markeredgewidth=0.6, zorder=5,
                label=f"Token {tok}")
        ax.plot(proj[-1, 0], proj[-1, 1], "*", color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)

    ax.set_xlabel("PC1 (Fisher-Rao scaled)", fontsize=9)
    ax.set_ylabel("PC2 (Fisher-Rao scaled)", fontsize=9)
    ax.set_title("Phase Portrait", fontsize=10)
    _safe_legend(ax, loc="best", framealpha=0.9, fontsize=7)


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------

def generate_all_fiber_figures(
    snapshots: List,
    stats_list: List,
    token_indices: List[int],
    output_dir: Path,
) -> Dict[str, Path]:
    r"""Batch-generate all fiber trajectory figures and save to ``output_dir``.

    Wraps each individual plot function in a ``try``/``except`` block so that
    failures in one figure do not prevent the others from being generated.
    Both PNG (screen quality, 300 DPI) and PDF (vector, for manuscripts) are
    produced for each figure.

    Args:
        snapshots: Ordered list of :class:`IterationSnapshot` objects.
        stats_list: List of :class:`FiberTrajectoryStats` objects.
        token_indices: Token row indices to include in all trajectory figures.
        output_dir: Directory where figures are written.  Created if it does
            not already exist.

    Returns:
        Dict mapping figure name (e.g. ``"phase_portrait"``) to the saved
        :class:`pathlib.Path`.  Both ``"phase_portrait"`` and
        ``"phase_portrait_pdf"`` entries are included for successfully saved
        figures.
    """
    if not MATPLOTLIB_AVAILABLE:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}

    # Arc-length-by-layer placeholder (derive from stats_list if available)
    arc_by_layer: Optional[np.ndarray] = None
    if stats_list:
        arcs = np.array([s.arc_length for s in stats_list])
        arc_by_layer = arcs[np.newaxis, :]  # (1, n_tokens) — single layer

    figure_jobs = [
        (
            "phase_portrait",
            lambda p: plot_fiber_phase_portrait(snapshots, token_indices, p),
        ),
        (
            "velocity_profiles",
            lambda p: plot_velocity_profiles(snapshots, token_indices, p),
        ),
        (
            "convergence_curves",
            lambda p: plot_convergence_curves(snapshots, token_indices, p),
        ),
        (
            "geodesic_deviation",
            lambda p: plot_geodesic_deviation(stats_list, p),
        ),
        (
            "fiber_dashboard",
            lambda p: plot_fiber_trajectory_dashboard(
                snapshots, stats_list, token_indices, p
            ),
        ),
    ]

    # Arc length heatmap only makes sense with multi-layer data; include when
    # possible.
    if arc_by_layer is not None:
        _arc = arc_by_layer  # capture for lambda closure
        figure_jobs.append(
            (
                "arc_length_heatmap",
                lambda p, a=_arc: plot_arc_length_heatmap(a, save_path=p),
            )
        )

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        for name, plot_fn in figure_jobs:
            for ext in ("png", "pdf"):
                path = output_dir / f"{name}.{ext}"
                try:
                    fig = plot_fn(path)
                    if fig is not None:
                        plt.close(fig)
                        key = name if ext == "png" else f"{name}_pdf"
                        saved[key] = path
                except (ValueError, TypeError, OSError, np.linalg.LinAlgError):
                    pass

    return saved
