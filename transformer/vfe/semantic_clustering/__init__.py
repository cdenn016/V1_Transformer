"""Standalone semantic-clustering visualization for the VFE package.

Clusters per-token belief geometry (``mu``, ``Sigma``, ``phi`` and the group
element ``Omega = exp(phi.G)``) with geometry-faithful distances and emits
separate publication-quality figures plus a metrics sidecar, for both
contextual and vocab-level views.

This package is intentionally decoupled from the legacy
``transformer/analysis/semantics.py``: it reads ``transformer/vfe`` attributes
directly and re-derives the geometric primitives it needs.
"""

from transformer.vfe.semantic_clustering.bundle import BeliefBundle
from transformer.vfe.semantic_clustering.extract import (
    extract_contextual,
    extract_vocab,
)
from transformer.vfe.semantic_clustering.pipeline import run_clustering
from transformer.vfe.semantic_clustering.plotting import (
    plot_mu_clustering,
    plot_omega_clustering,
    plot_phi_vector_clustering,
    plot_sigma_clustering,
)

__all__ = [
    "BeliefBundle",
    "extract_contextual",
    "extract_vocab",
    "run_clustering",
    "plot_mu_clustering",
    "plot_sigma_clustering",
    "plot_phi_vector_clustering",
    "plot_omega_clustering",
]
