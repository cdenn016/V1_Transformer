r"""Cluster-quality and per-quantity geometry metrics for semantic clustering.

Every function returns a plain-keyed dict of JSON-serializable values
(``float``, ``int``, ``None``, or lists thereof). No torch tensors or numpy
arrays leak into the returned dicts so the results can be written straight to
``metrics.json``.

Four families of metric:

* :func:`common_metrics` — unsupervised cluster-validity indices given a
  precomputed distance matrix and integer labels (silhouette, Calinski-Harabasz,
  Davies-Bouldin, and an inter/intra distance ratio read off ``D``).
* :func:`sigma_metrics` — per-token covariance geometry: exponential-entropy
  effective rank, log-determinant, trace, and anisotropy (max/min eigenvalue
  ratio).
* :func:`phi_metrics` — Lie-algebra energy partition between the per-head
  *diagonal* generators (:math:`E_{aa}`) and the *off-diagonal* generators
  (:math:`E_{ab}, a \neq b`) within each ``gl(d)`` block, plus optional
  transport summaries when ``Omega`` is supplied.
* :func:`mu_metrics` — token-mean norm statistics and mean pairwise distance.

Note on ``phi_metrics`` and ``cross_coupling_metrics.phi_energy_partition``:
the existing ``transformer/vfe/cross_coupling_metrics.py::phi_energy_partition``
(line ~90) was read before writing this module. It answers a *different*
question — it splits ``||phi||^2`` between the per-head GL blocks ("diag") and
the inter-head cross-coupling blocks ("cross"), keyed off ``cfg.cross_couplings``,
and takes a ``VFEConfig`` rather than ``irrep_dims``. The plan's ``phi_metrics``
instead splits *within* each ``gl(d)`` block between the diagonal generators
(:math:`E_{aa}`) and the off-diagonal generators (:math:`E_{ab}, a \neq b`).
The two conventions are not the same partition, and the signatures differ, so
the function is NOT reused. Only its style is matched: the L2 energy is the
mean of ``phi**2`` over the leading (token) axes, and the return value is a
plain-float dict.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def _to_numpy(t) -> np.ndarray:
    """Detach a torch tensor to a CPU float64 numpy array; pass arrays through."""
    if isinstance(t, torch.Tensor):
        return t.detach().to(torch.float64).cpu().numpy()
    return np.asarray(t, dtype=np.float64)


# -----------------------------------------------------------------------------
# Common cluster-validity metrics
# -----------------------------------------------------------------------------


def common_metrics(
    D: np.ndarray,
    labels: np.ndarray,
    precomputed: bool = True,
    X: Optional[np.ndarray] = None,
) -> dict:
    r"""Unsupervised cluster-validity indices.

    Args:
        D: ``(n, n)`` precomputed distance/dissimilarity matrix.
        labels: ``(n,)`` integer cluster labels.
        precomputed: If True, the silhouette uses ``metric="precomputed"`` on
            ``D``; otherwise ``D`` is treated as a feature matrix.
        X: Optional ``(n, d)`` feature matrix. Calinski-Harabasz and
            Davies-Bouldin require features; when ``X is None`` both are
            reported as ``None``.

    Returns:
        Dict with ``silhouette``, ``calinski_harabasz``, ``davies_bouldin``,
        ``inter_intra_ratio``, and ``n_clusters``. ``silhouette`` is ``None``
        when fewer than two clusters are present (the score is undefined).
    """
    D = np.asarray(D, dtype=np.float64)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    n_clusters = int(unique.shape[0])

    silhouette: Optional[float]
    if n_clusters < 2:
        silhouette = None
    else:
        metric = "precomputed" if precomputed else "euclidean"
        silhouette = float(silhouette_score(D, labels, metric=metric))

    calinski: Optional[float] = None
    davies: Optional[float] = None
    if X is not None and n_clusters >= 2:
        X_arr = _to_numpy(X)
        calinski = float(calinski_harabasz_score(X_arr, labels))
        davies = float(davies_bouldin_score(X_arr, labels))

    inter_intra = _inter_intra_ratio(D, labels)

    return {
        "silhouette": silhouette,
        "calinski_harabasz": calinski,
        "davies_bouldin": davies,
        "inter_intra_ratio": inter_intra,
        "n_clusters": n_clusters,
    }


def _inter_intra_ratio(D: np.ndarray, labels: np.ndarray) -> Optional[float]:
    r"""Mean between-cluster distance divided by mean within-cluster distance.

    Reads pair distances directly off the precomputed matrix ``D``. Only the
    strict upper triangle is used to avoid double-counting and the zero
    diagonal. Returns ``None`` when either the within- or between-cluster pair
    set is empty (e.g. a single cluster, or every cluster a singleton).
    """
    n = D.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    if iu.size == 0:
        return None
    same = labels[iu] == labels[ju]
    within = D[iu, ju][same]
    between = D[iu, ju][~same]
    if within.size == 0 or between.size == 0:
        return None
    within_mean = float(within.mean())
    if within_mean == 0.0:
        return None
    return float(between.mean()) / within_mean


# -----------------------------------------------------------------------------
# Sigma (covariance) geometry metrics
# -----------------------------------------------------------------------------


def sigma_metrics(sigma, diagonal: bool = True) -> dict:
    r"""Per-token covariance geometry, averaged over tokens.

    For each token the eigenvalues :math:`\lambda` are the diagonal entries
    when ``diagonal`` (``sigma`` shape ``(n, K)``) or the eigenvalues of the
    SPD matrix when full (``sigma`` shape ``(n, K, K)``). The effective rank is
    the exponential entropy of the normalized spectrum,

    .. math::
        R = \exp\!\Big(-\sum_k p_k \log p_k\Big), \qquad
        p_k = \frac{\lambda_k}{\sum_{k'} \lambda_{k'}},

    bounded in ``[1, K]``.

    Returns:
        Dict with ``effective_rank_mean``/``effective_rank_std`` (both in
        ``[1, K]``), ``logdet_mean``, ``trace_mean``, and ``anisotropy_mean``
        (mean of max/min eigenvalue ratio across tokens).
    """
    eps = 1e-12
    s = _to_numpy(sigma)

    if diagonal:
        lam = s  # (n, K)
    else:
        # Symmetric eigenvalues of each SPD block; clamp to be safe.
        lam = np.linalg.eigvalsh(s)  # (n, K)
    lam = np.clip(lam, eps, None)

    p = lam / lam.sum(axis=-1, keepdims=True)
    entropy = -(p * np.log(np.clip(p, eps, None))).sum(axis=-1)  # (n,)
    eff_rank = np.exp(entropy)  # (n,) in [1, K]

    logdet = np.log(lam).sum(axis=-1)  # (n,)
    trace = lam.sum(axis=-1)  # (n,)
    anisotropy = lam.max(axis=-1) / lam.min(axis=-1)  # (n,)

    return {
        "effective_rank_mean": float(eff_rank.mean()),
        "effective_rank_std": float(eff_rank.std()),
        "logdet_mean": float(logdet.mean()),
        "trace_mean": float(trace.mean()),
        "anisotropy_mean": float(anisotropy.mean()),
    }


# -----------------------------------------------------------------------------
# Phi (Lie-algebra) energy partition + optional Omega summaries
# -----------------------------------------------------------------------------


def _diagonal_generator_indices(irrep_dims: List[int]) -> List[int]:
    r"""Indices of the diagonal generators :math:`E_{aa}` in the multi-head basis.

    The generator bank (``math_utils.generators.generate_glK_multihead_generators``)
    lays out, per head ``h`` of dim ``d_h``, the ``d_h^2`` basis matrices
    :math:`E_{ij}` in row-major local order ``c = i * d_h + j``. The diagonal
    generators are therefore the local indices ``a * d_h + a`` for
    ``a in range(d_h)``, offset by the cumulative ``sum(d_k^2)`` of preceding
    heads.

    Example: ``irrep_dims=[2, 2]`` gives ``{0, 3, 4, 7}`` (E_00, E_11 in each
    of the two heads).
    """
    diag: List[int] = []
    offset = 0
    for d in irrep_dims:
        for a in range(d):
            diag.append(offset + a * d + a)
        offset += d * d
    return diag


def phi_metrics(
    phi,
    irrep_dims: List[int],
    omega: Optional[torch.Tensor] = None,
) -> dict:
    r"""Energy partition of ``phi`` between diagonal and off-diagonal generators.

    Splits the mean L2 mass of ``phi`` (over the leading token axes) between the
    per-head diagonal generators :math:`E_{aa}` and the off-diagonal generators
    :math:`E_{ab}, a \neq b` within each ``gl(d)`` block:

    .. math::
        E_{\text{diag}} = \sum_{c \in I_{\text{diag}}} \overline{\phi_c^2},
        \qquad
        E_{\text{off}} = \sum_{c \notin I_{\text{diag}}} \overline{\phi_c^2}.

    The fractions ``energy_frac_diagonal`` and ``energy_frac_offdiag`` sum to
    ``1.0`` (both ``0.0`` when the total energy is zero).

    Args:
        phi: ``(n, n_gen)`` Lie-algebra coefficients (also accepts a leading
            batch axis ``(B, N, n_gen)``; the mean is taken over all leading
            axes).
        irrep_dims: Per-head irrep dims; ``sum(d^2)`` must equal ``n_gen``.
        omega: Optional ``(n, K, K)`` per-token transport. When given, adds
            ``omega_dist_from_identity_mean`` (mean ``||Omega - I||_F``) and
            ``omega_det_mean`` (mean ``det(Omega)``).

    Returns:
        Dict with ``energy_frac_diagonal``, ``energy_frac_offdiag``,
        ``energy_diagonal``, ``energy_offdiag``, ``energy_total``, and (when
        ``omega`` is supplied) the two Omega summaries.
    """
    p = _to_numpy(phi)
    n_gen = p.shape[-1]
    expected = int(sum(d * d for d in irrep_dims))
    if expected != n_gen:
        raise ValueError(
            f"sum(d^2 for d in irrep_dims) = {expected} does not match "
            f"phi's generator dimension {n_gen} (irrep_dims={irrep_dims})."
        )

    # Mean phi^2 over every leading (token / batch / seq) axis -> (n_gen,).
    e_per_gen = (p ** 2).reshape(-1, n_gen).mean(axis=0)

    diag_idx = _diagonal_generator_indices(irrep_dims)
    mask = np.zeros(n_gen, dtype=bool)
    mask[diag_idx] = True

    e_diag = float(e_per_gen[mask].sum())
    e_off = float(e_per_gen[~mask].sum())
    e_total = e_diag + e_off

    if e_total > 0.0:
        frac_diag = e_diag / e_total
        frac_off = e_off / e_total
    else:
        frac_diag = 0.0
        frac_off = 0.0

    out = {
        "energy_frac_diagonal": frac_diag,
        "energy_frac_offdiag": frac_off,
        "energy_diagonal": e_diag,
        "energy_offdiag": e_off,
        "energy_total": e_total,
    }

    if omega is not None:
        om = _to_numpy(omega)  # (n, K, K)
        K = om.shape[-1]
        eye = np.eye(K, dtype=om.dtype)
        dist = np.linalg.norm(om - eye, ord="fro", axis=(-2, -1))  # (n,)
        det = np.linalg.det(om)  # (n,)
        out["omega_dist_from_identity_mean"] = float(dist.mean())
        out["omega_det_mean"] = float(det.mean())

    return out


# -----------------------------------------------------------------------------
# Mu (token mean) metrics
# -----------------------------------------------------------------------------


def mu_metrics(mu) -> dict:
    r"""Token-mean norm statistics and mean pairwise distance.

    Returns:
        Dict with ``norm_mean``/``norm_std`` (statistics of ``||mu_i||_2``) and
        ``pairwise_distance_mean`` (mean of ``||mu_i - mu_j||_2`` over distinct
        pairs; ``0.0`` for fewer than two tokens).
    """
    m = _to_numpy(mu)  # (n, K)
    norms = np.linalg.norm(m, axis=-1)  # (n,)

    n = m.shape[0]
    if n < 2:
        pairwise_mean = 0.0
    else:
        diff = m[:, None, :] - m[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)  # (n, n)
        iu, ju = np.triu_indices(n, k=1)
        pairwise_mean = float(dists[iu, ju].mean())

    return {
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "pairwise_distance_mean": pairwise_mean,
    }
