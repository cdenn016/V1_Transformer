r"""End-to-end orchestration for VFE semantic clustering (one view).

:func:`run_clustering` ties the package units together for a single belief view
(``'contextual'`` or ``'vocab'``): extract a :class:`BeliefBundle`, optionally
subsample to a tractable token count, then for each of the four quantities
(``mu``, ``Sigma``, the ``phi`` coefficient vector, and the transport
``Omega = exp(phi.G)``) build a geometry-faithful distance matrix, project it to
2D, cluster it, compute metrics, and save a separate publication-quality figure.
A ``metrics.json`` and flat ``metrics.csv`` sidecar are written to ``outdir``.

The function is importable and side-effect-light (it only writes inside
``outdir``), so it is unit-testable without the CONFIG-driven entry point in
``transformer/vfe/run_semantic_clustering.py``.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from scipy.linalg import expm

if TYPE_CHECKING:
    from transformer.vfe.model import VFEModel

from transformer.vfe.semantic_clustering import geometry as geo
from transformer.vfe.semantic_clustering import metrics as met
from transformer.vfe.semantic_clustering.bundle import BeliefBundle
from transformer.vfe.semantic_clustering.clustering import cluster
from transformer.vfe.semantic_clustering.extract import (
    extract_contextual,
    extract_vocab,
)
from transformer.vfe.semantic_clustering.plotting import (
    plot_mu_clustering,
    plot_omega_clustering,
    plot_phi_vector_clustering,
    plot_sigma_clustering,
)
from transformer.vfe.semantic_clustering.projection import project

_DEFAULT_METHODS = {"projection": "umap", "clustering": "agglomerative"}


def _subsample(bundle: BeliefBundle, max_points: int, seed: int) -> BeliefBundle:
    """Randomly subsample a bundle's tokens to at most ``max_points`` rows.

    The geodesic and Bhattacharyya distance matrices are O(n^2) (and the Omega
    path additionally exponentiates a matrix per token), so an unbounded vocab
    or long context must be capped. Generators/irrep_dims are shared and left
    untouched.
    """
    n = bundle.n
    if n <= max_points:
        return bundle
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    strings = (
        [bundle.token_strings[i] for i in idx]
        if bundle.token_strings is not None
        else None
    )
    return BeliefBundle(
        mu=bundle.mu[idx_t],
        sigma=bundle.sigma[idx_t],
        phi=bundle.phi[idx_t],
        token_ids=bundle.token_ids[idx_t],
        token_strings=strings,
        generators=bundle.generators,
        irrep_dims=bundle.irrep_dims,
        source=bundle.source,
        layer=bundle.layer,
        diagonal=bundle.diagonal,
    )


# C0 control characters (U+0000–U+001F), DEL (U+007F), and the Unicode
# replacement character U+FFFD. A byte-level BPE tokenizer (e.g. tiktoken
# cl100k_base, whose first 256 ids are single-byte fallbacks) decodes an
# id that spans only part of a multi-byte UTF-8 sequence to U+FFFD '�', and a
# control-byte id to a C0 char; matplotlib renders these as the .notdef box or
# warns "Glyph 0 (\x00) missing from font". Stripping them collapses a
# byte-fallback label to the empty string, which the annotation selector then
# skips as non-informative rather than printing a row of identical '�' markers.
_CTRL_CHARS = {chr(c) for c in range(0x20)} | {chr(0x7F)} | {"�"}


def _sanitize_label(s: str) -> str:
    """Drop unrenderable C0/DEL/U+FFFD characters from a decoded token label."""
    return "".join(ch for ch in s if ch not in _CTRL_CHARS)


def _decode_strings(token_ids: torch.Tensor, dataset: Optional[Any]) -> Optional[list[str]]:
    """Best-effort token-id -> string decode via a dataset/tokenizer, else None."""
    if dataset is None:
        return None
    ids = token_ids.tolist()
    try:
        if hasattr(dataset, "decode"):
            return [_sanitize_label(dataset.decode([i])) for i in ids]
        if hasattr(dataset, "tokenizer") and hasattr(dataset.tokenizer, "decode"):
            return [_sanitize_label(dataset.tokenizer.decode([i])) for i in ids]
    except Exception as exc:  # noqa: BLE001 - decoding is best-effort only
        warnings.warn(f"token decode failed ({exc!r}); proceeding without labels.")
    return None


def _per_token_omega(
    phi: torch.Tensor, generators: torch.Tensor
) -> np.ndarray:
    r"""Per-token full transport ``Omega_k = exp(sum_c phi[k,c] G_c)``, shape (n, K, K)."""
    phi_np = phi.detach().to(torch.float64).cpu().numpy()  # (n, n_gen)
    G = generators.detach().to(torch.float64).cpu().numpy()  # (n_gen, K, K)
    n, K = phi_np.shape[0], G.shape[-1]
    out = np.empty((n, K, K), dtype=np.float64)
    for k in range(n):
        A = np.einsum("c,cij->ij", phi_np[k], G)  # (K, K)
        out[k] = np.asarray(expm(A)).real
    return out


def _project_and_cluster(
    D: np.ndarray, proj_method: str, clust_method: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Project a distance matrix to 2D and cluster it; return (coords, labels)."""
    coords = project(D, method=proj_method, n_components=2, precomputed=True,
                     random_state=seed)
    labels = cluster(D, method=clust_method, precomputed=True, k="auto",
                     random_state=seed)
    return coords, labels


def run_clustering(
    model: "VFEModel",
    *,
    source: str,
    layer: Union[str, int] = "final",
    token_ids: Optional[torch.Tensor] = None,
    dataset: Optional[Any] = None,
    methods: Optional[dict[str, str]] = None,
    outdir: Union[str, Path],
    max_points: int = 200,
    seed: int = 0,
) -> dict:
    r"""Run the full clustering pipeline for one belief view and write artifacts.

    Args:
        model: A ``VFEModel``.
        source: ``'contextual'`` or ``'vocab'``.
        layer: Layer for contextual extraction (``'final'`` or an int).
        token_ids: For ``'contextual'`` a ``(B, N)`` batch; for ``'vocab'`` a
            ``(n,)`` set of token types (None -> a default vocab range).
        dataset: Optional object exposing ``decode``/``tokenizer`` for labels.
        methods: ``{'projection': ..., 'clustering': ...}`` (defaults to
            UMAP + agglomerative).
        outdir: Output directory for the four figures + metrics sidecars.
        max_points: Cap on the number of tokens clustered (O(n^2) distances).
        seed: Seed for subsampling and the projection backend.

    Returns:
        The nested metrics dict (also written to ``metrics.json``).
    """
    methods = {**_DEFAULT_METHODS, **(methods or {})}
    proj_method = methods["projection"]
    clust_method = methods["clustering"]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if source == "contextual":
        if token_ids is None:
            raise ValueError("source='contextual' requires token_ids (B, N).")
        bundle = extract_contextual(model, token_ids, layer=layer)
    elif source == "vocab":
        bundle = extract_vocab(model, token_ids=token_ids)
    else:
        raise ValueError(f"source must be 'contextual' or 'vocab', got {source!r}")

    bundle = _subsample(bundle, max_points=max_points, seed=seed)
    strings = _decode_strings(bundle.token_ids, dataset)

    diagonal = bundle.diagonal
    results: dict = {}

    # ---- mu: Mahalanobis (geometry-faithful) -------------------------------
    D_mu = geo.mu_distances(bundle.mu, sigma=bundle.sigma,
                            metric="mahalanobis", diagonal=diagonal)
    coords, labels = _project_and_cluster(D_mu, proj_method, clust_method, seed)
    mu_metrics = {**met.common_metrics(D_mu, labels),
                  **met.mu_metrics(bundle.mu)}
    plot_mu_clustering(coords, labels, mu_metrics, outdir, token_strings=strings)
    results["mu"] = mu_metrics

    # ---- Sigma: Bhattacharyya (geometry-faithful) --------------------------
    D_sig = geo.sigma_distances(bundle.sigma, mu=bundle.mu,
                                metric="bhattacharyya", diagonal=diagonal)
    coords, labels = _project_and_cluster(D_sig, proj_method, clust_method, seed)
    sig_metrics = {**met.common_metrics(D_sig, labels),
                   **met.sigma_metrics(bundle.sigma, diagonal=diagonal)}
    plot_sigma_clustering(coords, labels, sig_metrics, outdir, token_strings=strings)
    results["sigma"] = sig_metrics

    # ---- phi vector: PCA-whitened Euclidean --------------------------------
    D_phi = geo.phi_vector_distances(bundle.phi)
    coords, labels = _project_and_cluster(D_phi, proj_method, clust_method, seed)
    phi_metrics = {**met.common_metrics(D_phi, labels),
                   **met.phi_metrics(bundle.phi, bundle.irrep_dims)}
    plot_phi_vector_clustering(coords, labels, phi_metrics, outdir,
                               token_strings=strings)
    results["phi_vector"] = phi_metrics

    # ---- Omega: per-head left-invariant GL+(K) geodesic --------------------
    if bundle.generators is not None:
        D_om = geo.omega_geodesic_distances(bundle.phi, bundle.generators,
                                            bundle.irrep_dims)
        coords, labels = _project_and_cluster(D_om, proj_method, clust_method, seed)
        omega = _per_token_omega(bundle.phi, bundle.generators)
        om_metrics = {**met.common_metrics(D_om, labels),
                      **met.phi_metrics(bundle.phi, bundle.irrep_dims, omega=omega)}
        plot_omega_clustering(coords, labels, om_metrics, outdir,
                              token_strings=strings)
        results["omega"] = om_metrics
    else:
        warnings.warn("model.generators is None; skipping the Omega view.")

    results["meta"] = {
        "source": bundle.source,
        "layer": bundle.layer,
        "n_tokens": int(bundle.n),
        "K": int(bundle.K),
        "diagonal_covariance": bool(diagonal),
        "projection": proj_method,
        "clustering": clust_method,
        "irrep_dims": list(bundle.irrep_dims),
    }

    _write_metrics(results, outdir)
    return results


def _write_metrics(results: dict, outdir: Path) -> None:
    """Write ``metrics.json`` (nested) and ``metrics.csv`` (flat) to ``outdir``."""
    (outdir / "metrics.json").write_text(json.dumps(results, indent=2))

    with (outdir / "metrics.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["quantity", "metric", "value"])
        for quantity, block in results.items():
            if not isinstance(block, dict):
                continue
            for key, value in block.items():
                writer.writerow([quantity, key, value])
