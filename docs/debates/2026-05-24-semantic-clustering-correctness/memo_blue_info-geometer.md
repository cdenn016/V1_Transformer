# Memo — info-geometer (blue, Phase 2)

Authored in-role by the coordinator (dispatch tool unavailable); synthesized into `02_blue_opening.md`.

## Position
The non-φ metrics are defensible. Mahalanobis-μ whitening by the average belief covariance (`geometry.py:90-105`) is a standard pooled-covariance whitening. Bhattacharyya on Σ (`geometry.py:147-150`) is a recognized Gaussian dissimilarity; it is non-metric (no triangle inequality), and the pipeline correctly routes it only through projectors/clusterers that accept non-metric input (UMAP/MDS precomputed, average-linkage agglomerative) — `projection.py:16-22` and `clustering.py:8-10` document and respect this.

## Concession (load-bearing)
PCA-whitening on φ (`geometry.py:261-266`, default) is NOT a canonical Lie-algebra metric: not the Killing form, not an Ad-invariant inner product, not the Fisher metric on the induced family. Whitening rescales each empirical principal direction by its singular value — data-dependent, no Cencov-invariance [Cencov1972; Bishop2006 §12.1]. Defensible only as a *visualization* preconditioning, justified empirically by the ~0.005 Spearman cost vs unwhitened φ and the 0.952–0.972 tracking of the canonical Ω geodesic. Do not defend whitening as principled.

## Primary-source citation
[Cencov1972] — uniqueness of the Fisher metric under sufficient statistics; [Bishop2006 PRML §12.1] — whitening is a non-distance-preserving metric re-choice.
