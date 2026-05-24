# Memo — ml-engineer (blue, Phase 2)

Authored in-role by the coordinator (dispatch tool unavailable); synthesized into `02_blue_opening.md`.

## Position
The φ/Ω *visual* divergence is a projection artifact, not a math bug. UMAP with `metric="precomputed"` (`projection.py:125-137`) optimizes a fuzzy-topological neighborhood cross-entropy with no constraint fixing global orientation or preserving inter-cluster distance; its embeddings carry free rotation/reflection and are not distance-faithful [McInnes2018 arXiv:1802.03426 §2-3]. Feeding two rank-correlated-but-not-identical dissimilarity matrices (whitened-Euclidean φ vs Ω geodesic) into an orientation-free embedder produces two pictures that look different while certifying the same partition. The matrix-level agreement is the right grain: D_φ vs D_Ω Spearman 0.952–0.972, both panels k=2 in both views (verified in vocab/contextual metrics.json). Silhouette on a precomputed non-metric dissimilarity is a heuristic comparison [Rousseeuw1987], appropriate for "both pick k=2 with comparable silhouette."

## Caveat for the rebuttal
A 0.96 matrix-level Spearman does NOT license a side-by-side *visual* equivalence claim — two UMAP layouts are not related by a rigid transform. The blue claim is "same cluster structure," not "same picture." Hold that line precisely.

## Primary-source citation
[McInnes2018 arXiv:1802.03426] — UMAP non-distance/orientation preservation; [Rousseeuw1987] — silhouette as heuristic.
