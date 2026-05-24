# Memo — info-geometer (red, opening)

## Position

The PCA whitening applied to φ (geometry.py:215-270) is a sample-covariance preconditioning that
has no information-geometric justification as a distance on the Lie-algebra coefficient space, and
it is applied to only one of the two panels the figure asks the viewer to compare. The whitening is
neither the Fisher metric of the φ family nor any invariant inner product on the algebra; it is the
empirical second-moment of the 200 sampled φ vectors, which makes the resulting "distance" depend on
the subsample rather than on the geometry.

## Argument

A principled metric on a parameter space is reparameterization-invariant — the natural gradient
`∇̃L = g(θ)⁻¹∇L` with `g` the Fisher matrix is the canonical example, and invariance under
reparameterization is its *defining* property [Amari 1998, "Natural gradient works efficiently in
learning," Neural Computation 10(2):251-276]. PCA whitening preconditions with the empirical
second-moment of the sample, not with the Fisher of the φ family — the same category error the
standard literature flags when Adam/RMSProp are mistaken for natural gradient [Amari1998 §4;
external_canon_math.md §3 pitfall 5]. The whitener here is `W = diag(1/s)` on the top-m principal
components of the *centered sample matrix* (geometry.py:249, 258-266). Its eigenbasis and eigenvalues
are properties of which 200 tokens were subsampled (pipeline.py:62 `rng.choice`), not of the
coefficient geometry. Change the subsample seed and the whitener — hence every φ distance — changes.

For Lie-algebra-valued parameters the appropriate inner product is not Euclidean and must be argued
from structure (Killing form, an invariant inner product, or a task Fisher) [external_canon_math.md §1,
closing paragraph: "do not assume Euclidean"]. The module assumes Euclidean-after-whitening, which is
the worst of both: not the raw Euclidean coordinate metric, and not an invariant metric. The Ω panel,
by contrast, uses `||log(Ω_i⁻¹Ω_j)||_F`, which IS an invariant group distance. So the two panels do
not even share a metric philosophy: one is sample-conditioned anisotropic Euclidean, the other is
intrinsic and invariant. The claim that they "encode the same cluster structure" is asserting an
agreement between two objects that were built under incompatible metric assumptions, and the only
evidence offered is a rank correlation. Spearman is a rank-correlation summary: it constrains the
ordering of distance values, not the geometric structure of the underlying matrices, and it does not
predict 2D-layout similarity under a manifold-learning embedding whose objective is built from the
actual neighbor distances, not their ranks.

The defense's own decomposition concedes the mechanism: D_φ_whitened vs D_φ_unwhitened = 0.985
(evidence pack), i.e. whitening alone moves the rank correlation, and the 2D UMAP layout is far more
sensitive to small metric changes than the Spearman is. A 1.5% rank-correlation cost can produce a
visibly different 2D embedding because UMAP's neighbor graph is built from the actual distances, not
their ranks.

## What would falsify my position

If the whitener were the Fisher information of the φ-induced Gaussian family (or a fixed
data-independent invariant inner product on gl(K)), the φ distance would be a principled,
subsample-stable information metric and my attack collapses. The code uses neither: it whitens by the
empirical SVD of the subsampled, centered φ matrix (geometry.py:258), which is data- and
seed-dependent.

## Newly-discovered canon

- [Amari 1998, "Natural gradient works efficiently in learning," Neural Computation 10(2):251-276]
  — natural gradient's defining property is reparameterization invariance; empirical second-moment
  preconditioning (Adam/RMSProp) is not the Fisher and not invariant. Registered in
  external_bibliography.md.
- Spearman is a rank-correlation summary over distance-value orderings; it does not constrain the
  geometric structure of the underlying matrices nor predict 2D-layout similarity under a
  manifold-learning embedding (UMAP optimizes a cross-entropy on actual neighbor distances, not their
  ranks [McInnes2018]).
