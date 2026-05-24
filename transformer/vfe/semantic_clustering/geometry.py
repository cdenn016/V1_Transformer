r"""Geometry-faithful per-token distance matrices for VFE semantic clustering.

Each public function returns a symmetric ``(n, n)`` :class:`numpy.ndarray` with a
zero diagonal and non-negative entries, suitable as a precomputed dissimilarity
matrix for downstream projection (UMAP / MDS / t-SNE) and clustering.

The four quantities mirror the components of a Gaussian belief tuple
:math:`(\mu, \Sigma, \phi)` and the induced transport
:math:`\Omega = \exp(\sum_c \phi_c G_c)`:

* :func:`mu_distances` — Euclidean / Mahalanobis distance between means.
* :func:`sigma_distances` — Bhattacharyya / log-Euclidean distance between
  covariances.
* :func:`phi_vector_distances` — PCA-whitened Euclidean distance between the raw
  Lie-algebra coefficient vectors.
* :func:`omega_geodesic_distances` — block-wise (per-head) left-invariant
  geodesic between transport elements on the matrix Lie group GL+(K).

No neural-network components; pure NumPy / SciPy linear algebra.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.linalg import expm, logm

__all__ = [
    "mu_distances",
    "sigma_distances",
    "phi_vector_distances",
    "omega_geodesic_distances",
]


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Detach a tensor and return a contiguous float64 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def _finalize(D: np.ndarray) -> np.ndarray:
    """Force exact symmetry, zero diagonal, and non-negativity (clean float noise)."""
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def mu_distances(
    mu: torch.Tensor | np.ndarray,
    sigma: Optional[torch.Tensor | np.ndarray] = None,
    metric: str = "euclidean",
    diagonal: bool = True,
) -> np.ndarray:
    r"""Pairwise distances between token means :math:`\mu_i`.

    For ``metric='euclidean'``,

    .. math::
        d_{ij} = \lVert \mu_i - \mu_j \rVert_2 .

    For ``metric='mahalanobis'`` the means are whitened by the average belief
    covariance :math:`\bar{S} = \tfrac{1}{n}\sum_k \Sigma_k`,

    .. math::
        d_{ij} = \sqrt{(\mu_i - \mu_j)^\top \bar{S}^{-1} (\mu_i - \mu_j)} .

    Parameters
    ----------
    mu : (n, K) means.
    sigma : (n, K) diagonal variances or (n, K, K) full covariances; required for
        ``metric='mahalanobis'``.
    metric : ``'euclidean'`` or ``'mahalanobis'``.
    diagonal : whether ``sigma`` is diagonal ``(n, K)`` (else full ``(n, K, K)``).

    Returns
    -------
    (n, n) symmetric distance matrix.
    """
    mu_np = _to_numpy(mu)
    diff = mu_np[:, None, :] - mu_np[None, :, :]  # (n, n, K)

    if metric == "euclidean":
        D = np.linalg.norm(diff, axis=-1)
        return _finalize(D)

    if metric == "mahalanobis":
        if sigma is None:
            raise ValueError("metric='mahalanobis' requires sigma.")
        sig = _to_numpy(sigma)
        if diagonal:
            sbar = sig.mean(axis=0)  # (K,)
            sbar = np.maximum(sbar, 1e-12)
            sq = (diff ** 2 / sbar[None, None, :]).sum(axis=-1)  # (n, n)
        else:
            sbar = sig.mean(axis=0)  # (K, K)
            sbar_inv = np.linalg.inv(sbar)
            # (n, n, K) @ (K, K) -> contract: (diff S^-1 diff)
            tmp = diff @ sbar_inv  # (n, n, K)
            sq = (tmp * diff).sum(axis=-1)  # (n, n)
        D = np.sqrt(np.maximum(sq, 0.0))
        return _finalize(D)

    raise ValueError(f"Unknown mu metric: {metric!r}")


def sigma_distances(
    sigma: torch.Tensor | np.ndarray,
    mu: Optional[torch.Tensor | np.ndarray] = None,
    metric: str = "bhattacharyya",
    diagonal: bool = True,
) -> np.ndarray:
    r"""Pairwise distances between token covariances :math:`\Sigma_i`.

    For ``metric='bhattacharyya'`` between :math:`N(\mu_i, S_i)` and
    :math:`N(\mu_j, S_j)`, with :math:`S = \tfrac{1}{2}(S_i + S_j)`,

    .. math::
        D_B = \tfrac{1}{8}(\mu_i - \mu_j)^\top S^{-1}(\mu_i - \mu_j)
              + \tfrac{1}{2}\ln\frac{\det S}{\sqrt{\det S_i\,\det S_j}} .

    For ``metric='logeuclidean'`` the distance is the log-Euclidean metric on the
    SPD manifold,

    .. math::
        d_{ij} = \lVert \log S_i - \log S_j \rVert_F ,

    which for diagonal covariances reduces to
    :math:`\lVert \log \sigma_i - \log \sigma_j \rVert_2`.

    Parameters
    ----------
    sigma : (n, K) diagonal variances or (n, K, K) full covariances.
    mu : (n, K) means; required for the Bhattacharyya mean term.
    metric : ``'bhattacharyya'`` or ``'logeuclidean'``.
    diagonal : whether ``sigma`` is diagonal ``(n, K)``.

    Returns
    -------
    (n, n) symmetric distance matrix.
    """
    sig = _to_numpy(sigma)

    if metric == "bhattacharyya":
        if diagonal:
            return _bhattacharyya_diag(sig, mu)
        return _bhattacharyya_full(sig, mu)

    if metric == "logeuclidean":
        if diagonal:
            log_sig = np.log(np.maximum(sig, 1e-12))  # (n, K)
            diff = log_sig[:, None, :] - log_sig[None, :, :]
            D = np.linalg.norm(diff, axis=-1)
            return _finalize(D)
        # full: precompute logm once per token, then Frobenius of differences
        n = sig.shape[0]
        logs = np.stack([_safe_logm(sig[k]) for k in range(n)])  # (n, K, K)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = np.linalg.norm(logs[i] - logs[j], "fro")
        return _finalize(D)

    raise ValueError(f"Unknown sigma metric: {metric!r}")


def _bhattacharyya_diag(
    sig: np.ndarray, mu: Optional[torch.Tensor | np.ndarray]
) -> np.ndarray:
    r"""Vectorized diagonal-covariance Bhattacharyya dissimilarity, shape (n, n)."""
    n, K = sig.shape
    sig = np.maximum(sig, 1e-12)
    S_ij = 0.5 * (sig[:, None, :] + sig[None, :, :])  # (n, n, K)

    if mu is not None:
        mu_np = _to_numpy(mu)
        mu_diff = mu_np[:, None, :] - mu_np[None, :, :]  # (n, n, K)
        mahal = (mu_diff ** 2 / S_ij).sum(axis=-1)  # (n, n)
    else:
        mahal = np.zeros((n, n))

    log_det_S = np.log(S_ij).sum(axis=-1)  # (n, n)
    log_det_i = np.log(sig).sum(axis=-1)  # (n,)
    det_term = log_det_S - 0.5 * (log_det_i[:, None] + log_det_i[None, :])  # (n, n)

    D = mahal / 8.0 + 0.5 * det_term
    return _finalize(D)


def _bhattacharyya_full(
    sig: np.ndarray, mu: Optional[torch.Tensor | np.ndarray]
) -> np.ndarray:
    r"""Full-covariance Bhattacharyya dissimilarity via slogdet + solve, shape (n, n)."""
    n = sig.shape[0]
    mu_np = _to_numpy(mu) if mu is not None else None
    slogdet_i = np.array([np.linalg.slogdet(sig[k])[1] for k in range(n)])  # (n,)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            S = 0.5 * (sig[i] + sig[j])
            _, logdet_S = np.linalg.slogdet(S)
            if mu_np is not None:
                d = mu_np[i] - mu_np[j]
                mahal = float(d @ np.linalg.solve(S, d))
            else:
                mahal = 0.0
            det_term = logdet_S - 0.5 * (slogdet_i[i] + slogdet_i[j])
            D[i, j] = D[j, i] = mahal / 8.0 + 0.5 * det_term
    return _finalize(D)


def phi_vector_distances(
    phi: torch.Tensor | np.ndarray,
    whiten: bool = True,
    max_comps: int = 50,
) -> np.ndarray:
    r"""PCA-whitened Euclidean distance between Lie-algebra coefficient vectors.

    The raw coefficient vectors :math:`\phi_i \in \mathbb{R}^{n_\text{gen}}` are
    centered and (optionally) whitened onto their top
    :math:`m = \min(n-1, \text{max\_comps})` principal components before computing

    .. math::
        d_{ij} = \lVert W \phi_i - W \phi_j \rVert_2 ,

    where :math:`W` whitens by the singular values. Whitening makes the distance
    invariant to anisotropic scaling of the coefficient space; identical rows map
    to identical projections and hence zero distance.

    Parameters
    ----------
    phi : (n, n_gen) coefficient vectors.
    whiten : divide each retained component by its singular value when True.
    max_comps : cap on the number of retained principal components.

    Returns
    -------
    (n, n) symmetric distance matrix.
    """
    X = _to_numpy(phi)  # (n, n_gen)
    n = X.shape[0]

    if n < 2:
        return np.zeros((n, n))

    Xc = X - X.mean(axis=0, keepdims=True)

    if n == 2:
        # only one effective direction; PCA is degenerate — use centered Euclidean
        diff = Xc[:, None, :] - Xc[None, :, :]
        return _finalize(np.linalg.norm(diff, axis=-1))

    m = min(n - 1, max_comps)
    # SVD of the centered data matrix is more stable than eigh of a built covariance
    U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = (U * S)[:, :m]  # (n, m) — principal-component scores

    if whiten:
        s = S[:m].copy()
        nonzero = s > 1e-12
        scale = np.zeros_like(s)
        scale[nonzero] = 1.0 / s[nonzero]
        comps = comps * scale[None, :]

    diff = comps[:, None, :] - comps[None, :, :]
    D = np.linalg.norm(diff, axis=-1)
    return _finalize(D)


def _safe_logm(M: np.ndarray) -> np.ndarray:
    """Real matrix logarithm; returns a NaN-filled array on non-finite results."""
    try:
        L = logm(M.astype(np.float64))
        L = np.asarray(L).real
        if not np.isfinite(L).all():
            return np.full_like(M, np.nan, dtype=np.float64)
        return L
    except Exception:
        return np.full_like(M, np.nan, dtype=np.float64)


def omega_geodesic_distances(
    phi: torch.Tensor | np.ndarray,
    generators: torch.Tensor | np.ndarray,
    irrep_dims: list[int],
) -> np.ndarray:
    r"""Block-wise left-invariant geodesic distance between transport elements.

    Each token's algebra element is :math:`A = \sum_c \phi_c G_c`, restricted per
    head to its irrep block, and exponentiated to the group element
    :math:`\Omega_h = \exp(A_h)`. The per-head distance is the left-invariant
    Frobenius geodesic on the matrix Lie group :math:`GL^+(d_h)`,

    .. math::
        d_{ij,h} = \lVert \log\!\big(\Omega_{h,i}^{-1}\,\Omega_{h,j}\big) \rVert_F ,

    which is invariant under left translation :math:`\Omega \mapsto A\,\Omega`
    (it is *not* the affine-invariant SPD-cone distance, since :math:`\Omega` is a
    general group element, not a symmetric positive-definite matrix). The total
    distance is the quadrature (Pythagorean sum) over the block-independent heads,

    .. math::
        d_{ij} = \sqrt{\sum_h d_{ij,h}^2} .

    When :func:`scipy.linalg.logm` returns a non-finite result for a head, that
    head falls back to the chordal Frobenius distance
    :math:`\lVert \Omega_{h,i} - \Omega_{h,j} \rVert_F`.

    Parameters
    ----------
    phi : (n, n_gen) coefficient vectors.
    generators : (n_gen, K, K) generator bank :math:`G`.
    irrep_dims : per-head irrep block dimensions; ``sum(irrep_dims) == K``.

    Returns
    -------
    (n, n) symmetric distance matrix.
    """
    phi_np = _to_numpy(phi)  # (n, n_gen)
    G = _to_numpy(generators)  # (n_gen, K, K)
    n = phi_np.shape[0]
    n_heads = len(irrep_dims)
    K = G.shape[-1]

    # head K-slices
    slices: list[tuple[int, int]] = []
    start = 0
    for d in irrep_dims:
        slices.append((start, start + d))
        start += d

    # Guard: restricting A_full to its diagonal blocks (A_h = A_full[a:b, a:b])
    # and exponentiating per block recovers the true per-head transport only when
    # the generator bank is block-diagonal in the irrep_dims partition — then
    # A_full = sum_c phi_c G_c carries no off-block algebra for any phi, and
    # exp(A_full) = blockdiag(exp(A_h)). This holds for the standard multihead
    # glK bank and for super-block-contiguous cross-coupled banks. A bank with
    # generators spanning blocks (e.g. auto_close_cross_head_basis=True closing
    # under brackets across super-blocks) would make the slice silently drop the
    # off-block components and return a wrong geodesic. Detect it once on the
    # bank and fail loudly rather than computing a wrong-but-finite distance.
    block_mask = np.zeros((K, K), dtype=bool)
    for a, b in slices:
        block_mask[a:b, a:b] = True
    offblock = np.abs(G[:, ~block_mask])
    if offblock.size and float(offblock.max()) > 1e-9:
        raise ValueError(
            "omega_geodesic_distances: the generator bank has off-block support "
            f"(max |off-block entry| = {float(offblock.max()):.3e}), so the "
            "per-head block restriction A_full[a:b, a:b] would silently drop "
            "cross-block algebra. The per-head left-invariant geodesic requires a "
            "block-diagonal bank in the irrep_dims partition "
            f"(irrep_dims={list(irrep_dims)}, K={K})."
        )

    # Precompute per-head Omega for every token: A_full = sum_c phi_c G_c, then
    # restrict to each head's K-block and exponentiate. Block-diagonal banks make
    # the restricted block equal the exact per-head algebra sum.
    # omegas[h] -> list of (d_h, d_h) arrays of length n
    omegas: list[list[np.ndarray]] = [[] for _ in range(n_heads)]
    for k in range(n):
        A_full = np.einsum("c,cij->ij", phi_np[k], G)  # (K, K)
        for h, (a, b) in enumerate(slices):
            A_h = A_full[a:b, a:b]
            omegas[h].append(expm(A_h).real if np.iscomplexobj(A_full) else expm(A_h))

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sq_sum = 0.0
            for h in range(n_heads):
                Oi = omegas[h][i]
                Oj = omegas[h][j]
                M = np.linalg.solve(Oi, Oj)  # Oi^{-1} Oj
                L = _safe_logm(M)
                if np.isfinite(L).all():
                    d_h = np.linalg.norm(L, "fro")
                else:
                    d_h = np.linalg.norm(Oi - Oj, "fro")  # chordal fallback
                sq_sum += d_h ** 2
            D[i, j] = D[j, i] = np.sqrt(max(sq_sum, 0.0))
    return _finalize(D)
