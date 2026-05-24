"""Publication-quality cluster-scatter figures for VFE belief geometry.

Each public function projects a 2D embedding (already computed upstream by the
projection step), colors points by their integer cluster label, overlays light
cluster-centroid markers, and renders a small text box with the headline
metrics for that quantity. Every figure is saved as both a vector PDF and a
300-dpi PNG.

The module deliberately reuses ``transformer/visualization/pub_style.py`` (a
shared house-style utility, not legacy semantics code) when available, and
falls back to a minimal serif/300-dpi style otherwise. The Agg backend is
selected lazily inside the plot functions so that merely importing this module
never mutates the global matplotlib backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

# Importing transformer.visualization.pub_style pulls in the visualization
# package __init__, which transitively calls matplotlib.use("Agg") at import
# time. We must not let merely importing THIS module mutate the caller's
# backend, so we snapshot the active backend and restore it afterwards. The
# snapshot is best-effort: if matplotlib is not importable we simply proceed
# to the fallback style.
try:
    import matplotlib as _mpl

    _backend_before = _mpl.get_backend()
except Exception:  # pragma: no cover - matplotlib always present in this env
    _mpl = None
    _backend_before = None

try:
    from transformer.visualization.pub_style import (
        set_pub_style,
        PUB_COLORS,
        PUB_CYCLE,
    )
    _HAVE_PUB_STYLE = True
except ImportError:  # pragma: no cover - exercised only without pub_style
    _HAVE_PUB_STYLE = False

    PUB_COLORS = {
        "blue": "#0072B2",
        "orange": "#E69F00",
        "green": "#009E73",
        "red": "#D55E00",
        "purple": "#CC79A7",
        "cyan": "#56B4E9",
        "yellow": "#F0E442",
        "black": "#000000",
        "gray": "#999999",
    }
    PUB_CYCLE = [
        PUB_COLORS["blue"],
        PUB_COLORS["orange"],
        PUB_COLORS["green"],
        PUB_COLORS["red"],
        PUB_COLORS["purple"],
        PUB_COLORS["cyan"],
        PUB_COLORS["yellow"],
        PUB_COLORS["black"],
    ]

    def set_pub_style() -> None:
        """Minimal fallback style: serif fonts, 300-dpi saves.

        Used only when ``transformer.visualization.pub_style`` cannot be
        imported. Mirrors the essential typography/resolution settings of the
        shared house style.
        """
        from matplotlib import rcParams

        rcParams.update(
            {
                "font.family": "serif",
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "savefig.dpi": 300,
                "figure.dpi": 150,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )


# Restore whatever backend was active before the pub_style import, so that
# importing this module is a no-op on the global matplotlib backend.
if _mpl is not None and _backend_before is not None:
    try:
        if _mpl.get_backend() != _backend_before:
            _mpl.use(_backend_before, force=False)
    except Exception:  # pragma: no cover - defensive; never fail at import
        pass


# Maximum number of token strings to annotate before the scatter gets cluttered.
_MAX_ANNOTATIONS = 30


def _color_for_label(label: int) -> str:
    """Return a palette color for an integer cluster label (cycled)."""
    return PUB_CYCLE[int(label) % len(PUB_CYCLE)]


def _metric_lines(metrics: dict, headline_key: str, headline_label: str) -> list[str]:
    """Build the text-box lines: silhouette, n_clusters, plus one headline.

    All key lookups are guarded so that a missing entry simply omits that
    line rather than raising.
    """
    lines: list[str] = []

    sil = metrics.get("silhouette")
    if sil is not None:
        try:
            lines.append(f"silhouette = {float(sil):.3f}")
        except (TypeError, ValueError):
            pass

    n_clusters = metrics.get("n_clusters")
    if n_clusters is not None:
        lines.append(f"n_clusters = {int(n_clusters)}")

    headline_val = metrics.get(headline_key)
    if headline_val is not None:
        try:
            lines.append(f"{headline_label} = {float(headline_val):.3f}")
        except (TypeError, ValueError):
            pass

    return lines


def _scatter_figure(
    coords: np.ndarray,
    labels: np.ndarray,
    metrics: dict,
    outdir: Union[str, Path],
    name: str,
    title: str,
    headline_key: str,
    headline_label: str,
    token_strings: Optional[list[str]] = None,
) -> tuple[str, str]:
    """Render and save one cluster scatter as PDF + PNG.

    Parameters
    ----------
    coords : (n, 2) array
        2D embedding coordinates for each token.
    labels : (n,) integer array
        Cluster assignment per token.
    metrics : dict
        Headline metrics dict. ``silhouette`` and ``n_clusters`` are common;
        ``headline_key`` selects the quantity-specific headline (guarded if
        absent).
    outdir : path-like
        Output directory; created with ``parents=True, exist_ok=True``.
    name : str
        Basename for the saved files (``<name>.pdf`` / ``<name>.png``).
    title : str
        Axis title.
    headline_key, headline_label : str
        Metrics-dict key and its human-readable label for the text box.
    token_strings : optional list[str]
        Per-token labels; at most ``_MAX_ANNOTATIONS`` are annotated.

    Returns
    -------
    (pdf_path, png_path) : tuple[str, str]
        Absolute paths to the two saved files.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_pub_style()

    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    uniq = np.unique(labels)
    for lab in uniq:
        mask = labels == lab
        pts = coords[mask]
        color = _color_for_label(lab)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=28,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.85,
            label=f"cluster {int(lab)}",
            zorder=2,
        )
        # Light cluster-centroid marker.
        centroid = pts.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=160,
            marker="X",
            color=color,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.45,
            zorder=3,
        )

    # Annotate a bounded number of token strings.
    if token_strings is not None:
        n_annot = min(len(token_strings), coords.shape[0], _MAX_ANNOTATIONS)
        for i in range(n_annot):
            ax.annotate(
                str(token_strings[i]),
                (coords[i, 0], coords[i, 1]),
                fontsize=6,
                alpha=0.7,
                xytext=(2, 2),
                textcoords="offset points",
            )

    # Headline-metrics text box.
    lines = _metric_lines(metrics, headline_key, headline_label)
    if lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=PUB_COLORS["gray"],
                alpha=0.85,
            ),
            zorder=4,
        )

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    if len(uniq) > 1:
        ax.legend(loc="best", fontsize=7, framealpha=0.9)

    pdf_path = outdir / f"{name}.pdf"
    png_path = outdir / f"{name}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(pdf_path), str(png_path)


def plot_mu_clustering(
    coords: np.ndarray,
    labels: np.ndarray,
    metrics: dict,
    outdir: Union[str, Path],
    token_strings: Optional[list[str]] = None,
) -> tuple[str, str]:
    r"""Plot the per-token mean (:math:`\mu`) clustering scatter.

    Colors the 2D embedding by cluster label, overlays cluster centroids, and
    renders a text box with ``silhouette`` and ``n_clusters``.

    Returns the ``(pdf_path, png_path)`` of the saved ``mu_clustering`` figures.
    """
    return _scatter_figure(
        coords,
        labels,
        metrics,
        outdir,
        name="mu_clustering",
        title=r"Token-mean ($\mu$) clustering",
        headline_key="norm_mean",
        headline_label="mean ||mu||",
        token_strings=token_strings,
    )


def plot_sigma_clustering(
    coords: np.ndarray,
    labels: np.ndarray,
    metrics: dict,
    outdir: Union[str, Path],
    token_strings: Optional[list[str]] = None,
) -> tuple[str, str]:
    r"""Plot the per-token covariance (:math:`\Sigma`) clustering scatter.

    Quantity-specific headline: ``effective_rank_mean``.

    Returns the ``(pdf_path, png_path)`` of the saved ``sigma_clustering`` figures.
    """
    return _scatter_figure(
        coords,
        labels,
        metrics,
        outdir,
        name="sigma_clustering",
        title=r"Covariance ($\Sigma$) clustering",
        headline_key="effective_rank_mean",
        headline_label="eff. rank (mean)",
        token_strings=token_strings,
    )


def plot_phi_vector_clustering(
    coords: np.ndarray,
    labels: np.ndarray,
    metrics: dict,
    outdir: Union[str, Path],
    token_strings: Optional[list[str]] = None,
) -> tuple[str, str]:
    r"""Plot the per-token Lie-algebra coefficient (:math:`\phi`) clustering scatter.

    Quantity-specific headline: ``energy_frac_offdiag`` (off-diagonal
    cross-coupling energy fraction).

    Returns the ``(pdf_path, png_path)`` of the saved ``phi_vector_clustering``
    figures.
    """
    return _scatter_figure(
        coords,
        labels,
        metrics,
        outdir,
        name="phi_vector_clustering",
        title=r"Lie-algebra ($\phi$) vector clustering",
        headline_key="energy_frac_offdiag",
        headline_label="off-diag energy frac",
        token_strings=token_strings,
    )


def plot_omega_clustering(
    coords: np.ndarray,
    labels: np.ndarray,
    metrics: dict,
    outdir: Union[str, Path],
    token_strings: Optional[list[str]] = None,
) -> tuple[str, str]:
    r"""Plot the per-token transport (:math:`\Omega = \exp(\phi\cdot G)`) clustering scatter.

    Quantity-specific headline: ``omega_dist_from_identity_mean``
    (mean :math:`\lVert \Omega - I \rVert_F`).

    Returns the ``(pdf_path, png_path)`` of the saved ``omega_clustering`` figures.
    """
    return _scatter_figure(
        coords,
        labels,
        metrics,
        outdir,
        name="omega_clustering",
        title=r"Transport ($\Omega$) geodesic clustering",
        headline_key="omega_dist_from_identity_mean",
        headline_label="mean ||Omega - I||_F",
        token_strings=token_strings,
    )
