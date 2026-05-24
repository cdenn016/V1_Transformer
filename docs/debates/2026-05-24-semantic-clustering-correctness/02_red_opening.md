# Red Opening — semantic-clustering-correctness

## Steelman (opposing position)

The distance formulas compute what their docstrings claim, the index alignment `strings[i] ↔
coords[i] ↔ token_ids[i]` is sound, the per-head quadrature geodesic equals the full-matrix geodesic
to ~4.4e-16 under the block-diagonal generator bank, and the φ/Ω matrices are ~0.96 Spearman-correlated
— so the module's geometry and bookkeeping are right, and its remaining weaknesses are presentational
completeness rather than correctness bugs.

## Position

The claim survives only by equivocating on the word "correct." Two of its load-bearing moves fail
under the canon: first, the φ panel and the Ω panel are built under incompatible metric assumptions —
a sample-conditioned, anisotropically rescaled whitened-Euclidean distance versus an intrinsic,
invariant group geodesic — so presenting them as "the same cluster structure seen two ways" is a
geometric category error that a 0.96 rank correlation cannot repair. Second, the partition of the
figure's failures into "presentation/completeness, not correctness" relabels present inaccuracies as
missing features to keep the verdict "correct" unfalsifiable. The module is not correct-by-design on
the φ/Ω comparison, and its vocab figure is an accuracy defect, not a completeness gap. Red concedes
the per-head quadrature and the index alignment; the red case does not need them.

## Evidence

The φ distance and the Ω distance are not the same kind of object. `omega_geodesic_distances`
(geometry.py:285-357) computes `||log(Ω_i⁻¹Ω_j)||_F`, a member of the affine-invariant /
bi-invariant distance class whose defining property is invariance under the congruence action
`P → A P Aᵀ` — it does not change under anisotropic linear rescaling of the underlying space
[Pennec, Fillard, Ayache 2006, IJCV 66(1):41-66, "A Riemannian Framework for Tensor Computing"].
`phi_vector_distances` (geometry.py:215-270) does the opposite: it rotates to the PCA basis and
divides each retained component by its singular value (`comps * (1/s)`, geometry.py:261-266), which is
exactly the whitening transform `y = L^{-1/2}Uᵀ(x−x̄)` — an anisotropic rescale to unit variance that
is not distance-preserving with respect to the un-whitened geometry [Bishop, *Pattern Recognition and
Machine Learning* 2006 §12.1]. So one panel is intrinsic and invariant; the other is a
sample-conditioned Mahalanobis-on-principal-components. They answer different distance questions on
different spaces, and only φ is whitened (`whiten=True` default, geometry.py:218). That is the metric
mismatch the claim files under "metric choice (whitened Euclidean vs un-whitened affine-invariant)
costing <0.05 correlation" — but the cost is not measured in correlation; it is measured in whether
the two panels are the same object, and they are not.

The whitener is worse than a fixed metric rechoice: it has no information-geometric standing and it
couples the geometry to the subsample. The principled metric on a parameter manifold is the one
invariant under reparameterization — that is the *defining* property of the Fisher/natural-gradient
construction [Amari 1998, "Natural gradient works efficiently in learning," Neural Computation
10(2):251-276]. The φ whitener is none of these: it is the empirical SVD of the centered,
*subsampled* φ matrix (geometry.py:249, 258), so the distance any token pair receives depends on which
other 199 tokens `_subsample` happened to draw (pipeline.py:62). Mistaking an empirical second-moment
preconditioner for a metric is the same category error the canon flags when Adam/RMSProp are called
natural gradient [Amari1998 §4]. The Ω geodesic has no such coupling — it is pairwise-intrinsic.

The 0.96 Spearman does not rescue the comparison. Spearman is a rank-correlation summary; it
constrains the ordering of distance values, not the geometric structure of the underlying matrices,
and it does not predict layout similarity under a manifold-learning embedding. The figure's stated
purpose is visual: `run_clustering` emits φ and Ω as separate publication scatters (pipeline.py:210,
222), each a separate UMAP fit on a different precomputed matrix. UMAP coordinates are not metrically
interpretable and the method preserves topological neighborhood structure rather than exact distance
[McInnes, Healy, Melville 2018, arXiv:1802.03426], so two UMAP layouts from two different matrices are
not related by a rigid transform. A 0.96 matrix-level rank correlation cannot make two
non-distance-faithful 2D layouts look like the same picture — which is exactly the comparison the
side-by-side figure invites. "The divergence is just UMAP reflection freedom plus a benign metric
choice" inverts the burden: the metric choice — whiten one panel, not the other — is what makes them
different objects in the first place.

"Correct-by-design" is being used to immunize the figure against its own illegibility. The claim uses
a narrow internalist sense of "correct" (the formulas are faithful to their inputs; no off-by-one) to
license a broad externalist conclusion (the module is right). For a scientific visualization these are
different predicates: a figure can be faithful to its inputs and still license a false inference in the
reader. The vocab view is exactly this. The entry point extracts token ids 0..255 (run_semantic_
clustering.py:152-153) and decodes each individually under cl100k_base (pipeline.py:103), whose first
256 ids are byte fallbacks; the verifier reproduced 107 → U+FFFD, 23 → empty, only 72/200 unique. `_
sanitize_label` strips C0/DEL but not U+FFFD or empty (pipeline.py:88-93), so the "vocabulary
semantics" scatter ships two-thirds placeholder glyphs. The 30 annotated points are chosen by **array
order** — `range(min(len(token_strings), coords.shape[0], 30))` (plotting.py:224-235) — which after the
sorted subsample are the lowest ids, i.e. punctuation/digit fragments. A reader cannot tell "these are
cluster representatives" from "these are the first 30 indices," and the figure invites the former while
delivering the latter. Displaying mostly unrenderable labels on a figure titled as a semantics result,
and annotating the least-meaningful tokens by index, are *present inaccuracies in a shipped artifact*,
not missing features. Relabeling them "completeness" is the equivocation: incompleteness is a missing
view; this is an inaccurate one. The same applies to the undisclosed per-occurrence duplicates
(extract.py:98 flattens `(B,N)→(B·N)`; all four plots share one `strings` list, pipeline.py:193-222):
the design may be defensible, but a figure whose correct reading is never communicated and whose
default reading is "duplication bug" has a communication-correctness hole, and the user — who flagged
exactly this — is the existence proof. To call this "correct by design," one must redefine the design's
purpose after the fact as "place points faithfully regardless of legibility," a purpose no one would
state in advance; that redefinition makes "correct" unlosable and therefore unfalsifiable.

There is also a latent correctness defect in the geometry itself, currently masked by the active
toggle. `omega_geodesic_distances` computes the full `A_full = einsum(phi, G)` and then discards the
off-diagonal blocks, exponentiating only `A_h = A_full[a:b, a:b]` (geometry.py:337-340) with no
assertion that the discarded blocks are zero. Under the active config (`cross_couplings=[]`, the
multihead bank is exactly block-diagonal — verifier off-block max = 0.0; `||expm(A_full) −
blockdiag(expm(A_h))|| ≈ 4.4e-16`) this is correct. Under `auto_close_cross_head_basis=True`, which
model.py:447-454 warns can add generators spanning super-blocks, `A_full` acquires off-block support
and geometry.py:339 silently truncates it; `expm` of a truncated block is not the block of `expm` of
the full matrix once off-block entries are nonzero, so the Ω geodesic would be computed on a different
group element than the model's own transport (`_per_token_omega`, pipeline.py:111-122, exponentiates
the full `A`). "Only breaks under a non-default toggle" is the latent-bug signature — correct under the
tested config, silently wrong under a reachable one — which is a weaker property than "correct."

## Falsification conditions

This position is wrong if any of the following hold. (1) The φ and Ω panels are shown to be the same
geometric object: if the module whitened both identically (or neither), and the residual visual
divergence were a rigid reflection/rotation of an otherwise-identical UMAP layout — but only φ is
whitened (geometry.py:218) and UMAP layouts from different matrices are not rigidly related
[McInnes2018]. (2) The φ whitener is in fact a principled metric: if it were the Fisher information of
the φ family or a fixed data-independent invariant inner product on gl(K) — but it is the SVD of the
subsampled centered matrix (geometry.py:258), so it is data- and seed-dependent [Amari1998]. (3) The
vocab/annotation artifacts are genuine incompleteness: if the figures disclosed per-occurrence
semantics and annotated salient/representative tokens — but annotation is array-order
(plotting.py:227), fallbacks are unstripped (pipeline.py:88-93), and duplicates are undisclosed. (4)
"Correctness" were explicitly scoped in the claim to numerical faithfulness and index alignment, with
figure legibility adjudicated on the separate accuracy axis the user requested ("correctness, accuracy,
AND completeness", claim §User context) — but the claim folds accuracy failures into "completeness" and
concludes "none of which is a correctness bug," which is the equivocation Red attacks.

Every panel memo is reflected above: geometer (φ-vs-Ω metric mismatch, Pennec, UMAP), info-geometer
(whitening is not a metric, Amari, Spearman-invariance), code-quality (array-order annotation,
fallback labels, undisclosed duplicates), philosophy-of-science ("correct-by-design" equivocation and
the accuracy/completeness relabeling), implementation-engineer (config trace to the byte-fallback
vocab path; latent unguarded block-restriction). No memo is discounted.
