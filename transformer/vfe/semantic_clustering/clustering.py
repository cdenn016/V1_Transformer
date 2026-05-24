"""Unsupervised clustering of belief geometry with silhouette-swept auto-k.

This module turns a precomputed dissimilarity matrix (e.g. one of the
geometry-faithful distance matrices produced by
``transformer.vfe.semantic_clustering.geometry``) into a flat integer cluster
labeling. The primary backend is average-linkage agglomerative clustering,
which accepts non-metric dissimilarities (the Bhattacharyya Sigma distance is
not a true metric). The number of clusters can be selected automatically by
sweeping a range of candidate ``k`` values and maximizing the silhouette score
computed directly on the precomputed distances.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.metrics import silhouette_score


def _best_k_agglomerative(
    matrix: np.ndarray,
    k_range: range,
    n: int,
) -> np.ndarray:
    r"""Sweep candidate ``k`` values and return the silhouette-optimal labeling.

    For each feasible :math:`k \in` ``k_range`` (with :math:`2 \le k \le n-1`),
    fit average-linkage agglomerative clustering on the precomputed
    dissimilarity ``matrix`` and score the labeling with the silhouette
    coefficient :math:`s = (b - a) / \max(a, b)` averaged over points, using
    ``metric="precomputed"``. The labeling with the largest mean silhouette is
    returned.

    Parameters
    ----------
    matrix : (n, n) np.ndarray
        Symmetric, non-negative, zero-diagonal precomputed distance matrix.
    k_range : range
        Candidate cluster counts to try.
    n : int
        Number of samples (``matrix.shape[0]``).

    Returns
    -------
    np.ndarray
        Integer labels of shape ``(n,)`` for the best scoring ``k``.
    """
    best_labels: np.ndarray | None = None
    best_score = -np.inf
    for k in k_range:
        # Silhouette is undefined unless 2 <= n_labels <= n_samples - 1.
        if k < 2 or k >= n:
            continue
        model = AgglomerativeClustering(
            metric="precomputed", linkage="average", n_clusters=k
        )
        labels = model.fit_predict(matrix)
        # Degenerate solutions (a single realized cluster) can't be scored.
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels, metric="precomputed")
        if score > best_score:
            best_score = score
            best_labels = labels
    if best_labels is None:
        # No feasible/scoreable k: fall back to a single cluster.
        return np.zeros(n, dtype=int)
    return best_labels.astype(int)


def cluster(
    matrix: np.ndarray,
    method: str = "agglomerative",
    precomputed: bool = True,
    k: Union[int, str] = "auto",
    k_range: range = range(2, 9),
    random_state: int = 0,
) -> np.ndarray:
    r"""Cluster a precomputed dissimilarity matrix into integer labels.

    Parameters
    ----------
    matrix : (n, n) np.ndarray
        Precomputed dissimilarity matrix (symmetric, non-negative, zero
        diagonal). Average-linkage agglomerative clustering and HDBSCAN both
        accept non-metric dissimilarities, so the non-metric Bhattacharyya
        Sigma distance is admissible here.
    method : str, default "agglomerative"
        ``"agglomerative"`` uses
        ``AgglomerativeClustering(metric="precomputed", linkage="average")``.
        ``"hdbscan"`` uses ``HDBSCAN(metric="precomputed")`` (auto-k; noise
        points receive label ``-1``).
    precomputed : bool, default True
        Whether ``matrix`` is a precomputed distance matrix. Currently only the
        precomputed path is supported; ``False`` raises ``ValueError``.
    k : int or "auto", default "auto"
        Fixed cluster count, or ``"auto"`` to sweep ``k_range`` and pick the
        silhouette-maximizing labeling. Ignored when ``method="hdbscan"``.
    k_range : range, default range(2, 9)
        Candidate cluster counts swept when ``k == "auto"``.
    random_state : int, default 0
        Accepted for interface symmetry; the agglomerative and silhouette paths
        are deterministic and do not consume it.

    Returns
    -------
    np.ndarray
        Integer cluster labels of shape ``(n,)``.

    Notes
    -----
    Small-``n`` guard: when ``n < 4`` the function returns a single all-zeros
    cluster, because silhouette selection is ill-posed for so few points.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"expected a square (n, n) distance matrix, got shape {matrix.shape}"
        )
    if not precomputed:
        raise ValueError(
            "cluster() currently only supports precomputed=True "
            "(matrix must be an (n, n) distance matrix)"
        )

    n = matrix.shape[0]

    # Small-n guard: a single trivial cluster.
    if n < 4:
        return np.zeros(n, dtype=int)

    if method == "hdbscan":
        model = HDBSCAN(metric="precomputed")
        labels = model.fit_predict(matrix)
        return labels.astype(int)

    if method != "agglomerative":
        raise ValueError(
            f"unknown method {method!r}; expected 'agglomerative' or 'hdbscan'"
        )

    if k == "auto":
        return _best_k_agglomerative(matrix, k_range, n)

    if not isinstance(k, (int, np.integer)):
        raise ValueError(f"k must be an int or 'auto', got {k!r}")
    k_int = int(k)
    if k_int < 1:
        raise ValueError(f"k must be >= 1, got {k_int}")
    if k_int >= n:
        # Degenerate: more clusters than separable points; collapse to one.
        return np.zeros(n, dtype=int)

    model = AgglomerativeClustering(
        metric="precomputed", linkage="average", n_clusters=k_int
    )
    labels = model.fit_predict(matrix)
    return labels.astype(int)
