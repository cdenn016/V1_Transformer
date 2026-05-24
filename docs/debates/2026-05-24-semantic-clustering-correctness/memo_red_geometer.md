# Memo — geometer (red, opening)

## Position

The φ panel and the Ω panel are not two views of one gauge object. `phi_vector_distances`
(geometry.py:215-270) builds a PCA-**whitened** Euclidean distance on the raw Lie-algebra
coefficients; `omega_geodesic_distances` (geometry.py:285-357) builds an **un-whitened**
affine-invariant group geodesic `||log(Ω_i⁻¹Ω_j)||_F`. These are different Riemannian
structures on different spaces. Presenting them side-by-side as "the same cluster structure
seen two ways" is a category conflation the 0.96 Spearman correlation does not repair.

## Argument

The affine-invariant Riemannian distance on the SPD/positive-definite cone — and the
analogous bi-invariant group distance — is `d(P,Q) = ||log(P^{-1/2} Q P^{-1/2})||_F`, which is
invariant under the congruence action `P → A P Aᵀ` [Pennec, Fillard, Ayache 2006, IJCV 66:41-66,
"A Riemannian Framework for Tensor Computing"]. The defining property is precisely that the
metric does NOT change under anisotropic linear rescaling of the underlying space; the geodesic
between two elements and the Fréchet mean are uniquely defined by this invariance. The Ω geodesic
in geometry.py inherits exactly this character: it uses `||log(Ω_i⁻¹Ω_j)||_F`, an un-rescaled,
basis-respecting distance.

PCA whitening does the opposite. Whitening rotates to the PCA basis, **rescales each retained axis
to unit variance**, and (here) drops the inverse rotation — geometry.py:258-266 keeps the
PC scores `U·S` and divides each by its singular value, `comps * (1/s)`. That is an explicit
anisotropic rescaling of the coefficient space. Whitening = "rotate to PCA basis, rescale each
axis to unit norm" — it manufactures isotropy that the raw φ distribution does not have and is
not distance-preserving with respect to the un-whitened geometry. Whitening is a metric (re)choice,
not a faithful representation [Bishop, *Pattern Recognition and Machine Learning* 2006, §12.1, on
whitening as `y = L^{-1/2}Uᵀ(x-μ̄)` producing unit covariance]. The Ω geodesic carries no such
rescaling. So the φ matrix is a whitened Mahalanobis-on-PCs object and the Ω matrix is an
affine-invariant geodesic object; they answer different distance questions.

The 0.96 Spearman is a rank-order summary statistic over 19,900 off-diagonal pairs. The figure's
stated purpose (run_clustering plots φ and Ω as separate publication scatters, pipeline.py:210/222)
is **visual** comparison of 2D layouts, not rank correlation of distance matrices. Each scatter is a
separate UMAP fit on a different precomputed matrix; UMAP optimizes a topological cross-entropy and
its output coordinates are not metrically interpretable [McInnes, Healy, Melville 2018,
arXiv:1802.03426 — embedding dimensions have no specific meaning, the method preserves topological
neighborhood structure rather than exact distances]. A 0.96 matrix-level rank correlation does not
imply two UMAP layouts will look the same; it cannot, because UMAP is not distance- or
orientation-faithful. The defense "the divergence is just UMAP reflection freedom plus a benign
metric choice" inverts the burden: the metric choice (whiten one, not the other) is exactly what
breaks the claim that the two panels show the same object.

## What would falsify my position

If the module whitened φ and Ω identically (or neither), and the residual visual divergence were
demonstrably only a rigid reflection/rotation of an otherwise-identical layout, the panels would be
the same object up to UMAP gauge freedom and my attack fails. It does not: only φ is whitened
(geometry.py:261 `whiten=True` default), Ω is not, and UMAP layouts under different precomputed
matrices are not related by a rigid transform.

## Newly-discovered canon

- [Pennec, Fillard, Ayache 2006, IJCV 66(1):41-66, "A Riemannian Framework for Tensor Computing"]
  — affine-invariant Riemannian metric on the SPD cone; geodesic distance via matrix log; invariance
  under `P → A P Aᵀ`; unique geodesic and Fréchet mean. Verified via Springer Link / CIS-UPenn PDF
  (https://link.springer.com/article/10.1007/s11263-005-3222-z).
- [McInnes, Healy, Melville 2018, arXiv:1802.03426, "UMAP"] — embedding dimensions are not
  interpretable; the method targets topological/neighborhood structure, not metric distance
  preservation. Verified via arXiv abstract and secondary analyses.
- Whitening as anisotropic rescale-to-unit-variance (not distance-preserving): standard treatment,
  [Bishop PRML 2006 §12.1]; corroborated by whitening-transform references (rotate to PCA basis,
  rescale each axis, optional inverse-rotate).
