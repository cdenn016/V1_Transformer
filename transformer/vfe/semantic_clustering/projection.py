# transformer/vfe/semantic_clustering/projection.py
r"""Low-dimensional projection of belief-geometry distance/feature matrices.

The semantic-clustering pipeline builds geometry-faithful dissimilarity
matrices (Euclidean, Mahalanobis, Bhattacharyya, log-Euclidean, Omega
geodesic) and needs to embed them into 2D/3D for plotting. This module
wraps the projector selection with a robust fallback chain.

Primary projector is UMAP with ``metric="precomputed"``. If umap-learn is
not importable we fall back to scikit-learn's
``TSNE(metric="precomputed")`` and, failing that, to
``MDS(dissimilarity="precomputed")``.

Non-metric caveat
-----------------
The Bhattacharyya covariance dissimilarity is not a metric (it violates the
triangle inequality). UMAP ``metric="precomputed"`` and
``MDS(dissimilarity="precomputed")`` both accept arbitrary non-negative
dissimilarity matrices. ``TSNE(metric="precomputed")`` can raise on
non-metric input in some scikit-learn versions, so the t-SNE fallback branch
is wrapped in ``try/except`` and degrades to MDS on failure. The distance
input is clamped to ``>= 0`` defensively before any projector sees it.
"""
from __future__ import annotations

import warnings

import numpy as np


def project(
    matrix: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    precomputed: bool = True,
    random_state: int = 0,
) -> np.ndarray:
    r"""Project a distance or feature matrix into ``n_components`` dimensions.

    Parameters
    ----------
    matrix : np.ndarray
        If ``precomputed`` is True, an ``(n, n)`` symmetric, non-negative
        dissimilarity matrix. Otherwise an ``(n, d)`` feature matrix.
    method : str
        ``"umap"`` (default) selects the UMAP-primary fallback chain
        (UMAP -> t-SNE -> MDS). ``"pca"`` selects scikit-learn PCA and is
        only valid for feature matrices (``precomputed=False``).
    n_components : int
        Target embedding dimensionality (2 or 3).
    precomputed : bool
        Whether ``matrix`` is a precomputed dissimilarity matrix.
    random_state : int
        Seed forwarded to the backend projector.

    Returns
    -------
    np.ndarray
        An ``(n, n_components)`` float array of embedded coordinates.

    Notes
    -----
    For tiny inputs (``n <= n_components + 1``) no projector can produce a
    well-defined embedding, so a zero-padded ``(n, n_components)`` array is
    returned. A :func:`warnings.warn` names the backend actually used.
    """
    X = np.asarray(matrix, dtype=np.float64)
    n = int(X.shape[0])

    # Tiny-n guard: cannot meaningfully embed; return zero-padded coords.
    if n <= n_components + 1:
        warnings.warn(
            f"project: n={n} <= n_components+1={n_components + 1}; "
            f"returning zero-padded coordinates (backend=none).",
            stacklevel=2,
        )
        return np.zeros((n, n_components), dtype=np.float64)

    if method == "pca":
        if precomputed:
            raise ValueError(
                "method='pca' requires a feature matrix (precomputed=False); "
                "PCA cannot operate on a precomputed dissimilarity matrix."
            )
        from sklearn.decomposition import PCA

        warnings.warn("project: using backend=PCA.", stacklevel=2)
        coords = PCA(
            n_components=n_components, random_state=random_state
        ).fit_transform(X)
        return np.asarray(coords, dtype=np.float64)

    if method != "umap":
        raise ValueError(
            f"Unknown method={method!r}; expected 'umap' or 'pca'."
        )

    if not precomputed:
        raise NotImplementedError(
            "method='umap' with precomputed=False is not supported; "
            "use method='pca' for feature matrices."
        )

    # Clamp defensively: precomputed dissimilarities must be >= 0.
    D = np.clip(X, 0.0, None)
    return _project_precomputed(D, n_components, random_state)


def _project_precomputed(
    D: np.ndarray, n_components: int, random_state: int
) -> np.ndarray:
    """Embed a precomputed dissimilarity matrix via UMAP -> t-SNE -> MDS."""
    # 1) UMAP (primary).
    try:
        import umap  # type: ignore
    except ImportError:
        pass
    else:
        warnings.warn("project: using backend=UMAP (precomputed).", stacklevel=2)
        reducer = umap.UMAP(
            metric="precomputed",
            n_components=n_components,
            random_state=random_state,
        )
        coords = reducer.fit_transform(D)
        return np.asarray(coords, dtype=np.float64)

    # 2) t-SNE fallback. Bhattacharyya dissimilarities are non-metric and some
    #    scikit-learn versions raise on precomputed non-metric input; degrade
    #    to MDS on any failure.
    from sklearn.manifold import TSNE

    try:
        warnings.warn(
            "project: umap-learn unavailable; using backend=TSNE (precomputed).",
            stacklevel=2,
        )
        coords = TSNE(
            n_components=n_components,
            metric="precomputed",
            init="random",
            random_state=random_state,
        ).fit_transform(D)
        return np.asarray(coords, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - degrade on any TSNE failure
        warnings.warn(
            f"project: TSNE(precomputed) failed ({exc!r}); "
            f"using backend=MDS (precomputed).",
            stacklevel=2,
        )

    # 3) MDS fallback (accepts arbitrary non-negative dissimilarities).
    from sklearn.manifold import MDS

    coords = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=random_state,
        normalized_stress="auto",
    ).fit_transform(D)
    return np.asarray(coords, dtype=np.float64)
