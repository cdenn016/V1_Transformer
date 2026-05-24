# memo_red_gauge-theorist (rebuttal) — semantic-clustering-correctness

## Concession forced by canon-cop (and correct on the merits)

The opening's Pennec attachment is withdrawn. Pennec, Fillard, Ayache 2006 defines the affine-invariant geodesic on the SPD cone Sym⁺(K), invariant under the *congruence* action P → A P Aᵀ. The code computes M = Ωᵢ⁻¹Ωⱼ then ‖log M‖_F (`geometry.py:349-352`, verified read), which is a distance on GL⁺(K) *group elements*, invariant under *left translation* Ω → A Ω. Different manifold, different symmetry, different invariance. Pennec is the wrong domain and I do not cite it for the Ω geodesic. I also do not call the Ω distance "bi-invariant" or "affine-invariant" — there is no bi-invariant Riemannian metric on non-compact GL⁺(K) (no compact real form), as blue states correctly at `02_blue_opening.md:25`.

Note for judges: the docstring at `geometry.py:290` itself names the distance "affine-invariant." That is a docstring; the code at 349-352 is canonical and computes the left-invariant group distance. The docstring carries the same wrong-domain error I just conceded — drift, zero weight, but worth flagging as a code-hygiene tell.

## What survives: the two panels are different geometric objects

Blue's Vector 1 leans on the φ-panel and the Ω-panel encoding "the same cluster structure." Concede the per-head quadrature is the exact product-manifold metric on GL(10) × GL(10) ⊂ GL(20) [Lee 2013, product-manifold construction: d² = Σ_h d_h²], and concede block-diagonal exp factorization [Hall 2015 §2-3; Nakahara 2003 §5]. Those make the *Ω-panel* internally sound. They say nothing about the φ-panel.

The φ-panel distance (`geometry.py:215-270`, `whiten=True`) is PCA-whitened Euclidean on the raw algebra coordinates φ: it rotates to the empirical PCA basis of the φ-sample and divides each retained principal direction by its singular value. That is a *sample-conditioned anisotropic rescaling of the chart*, not a metric pulled back from any group or algebra structure. The left-invariant Frobenius metric on the algebra is the un-whitened Killing-orthogonal Frobenius form; whitening replaces it with a per-run, per-sample reweighting that has no left-invariance and changes under any reparameterization of φ [Hall 2015 §3 on the Lie-algebra inner product]. So the φ-panel and the Ω-panel are not two reductions of one metric object; they are two different metrics, one principled-on-the-group and one chart-dependent. The side-by-side visual comparison invites the reader to treat them as the same object, which they are not.

## Newly-discovered canon

- [Hall 2015, *Lie Groups, Lie Algebras, and Representations*, §3] — the inner product on the Lie algebra inducing a left-invariant metric is the Frobenius/Killing form at the identity; rescaling algebra coordinates by empirical sample variances is not such an inner product.
- (Already in pack: Nakahara 2003 §5-6, Lee 2013, McInnes 2018.)
