# memo_red_info-geometer (rebuttal) — semantic-clustering-correctness

## Target: blue's "0.952 Spearman rescues the comparison" (Vector 1, `02_blue_opening.md:17,19`)

Blue's Vector 1 rests on D_φ(whitened) vs D_Ω Spearman 0.952–0.972 plus both panels picking k=2. The number is real (concede it). The inference from it is the error: rank-agreement of the two *input* distance matrices is a necessary-but-not-sufficient condition for the two *output* layouts to be visually comparable.

Two independent canon facts close this. First, whitening is not a principled metric: the unique reparameterization-invariant metric on a statistical/parametric manifold is the Fisher–Rao metric [Cencov 1972, *Statistical Decision Rules and Optimal Inference*; Amari 1998, Neural Computation 10(2):251-276]. Sample-covariance preconditioning — exactly PCA-whitening, the same family as Adam/RMSProp axis rescaling — is not Fisher and is not invariant under reparameterization of φ [Amari 1998]. Blue concedes this verbatim (`02_blue_opening.md:39`): whitened φ "is not a canonical metric ... no Cencov-invariance standing." So the φ-panel's metric is admitted non-canonical by both sides.

Second, the Spearman value is precisely the wrong statistic to rescue a *visual* comparison. Spearman constrains the ordering of distance values. UMAP's objective is a fuzzy-topological cross-entropy built from actual neighbor *distances*, not their ranks, and its output is explicitly not distance-preserving with free global rotation/reflection [McInnes, Healy, Melville 2018, arXiv:1802.03426, §2-3] — blue's own citation. A high rank-correlation between two input matrices is therefore consistent with two visibly different UMAP layouts. The measured decomposition makes this concrete: whitening alone moves the φ-distance by ~0.005 Spearman away from un-whitened φ (D_φ_whitened vs D_φ_unwhitened = 0.9849, evidence pack). That the rank-agreement survives whitening is the *expected* behavior of a near-rank-preserving linear rescale on this sample — it is a property of the rescale, not independent evidence that the two panels depict the same geometric object. The 0.952 confirms the orderings nearly agree; it does not certify that the side-by-side picture is faithful, and blue's McInnes citation forbids that inference.

(Precise statement, to avoid overclaim: whitening is a linear map; the induced pairwise distance is not in general a monotone function of the un-whitened distance — different pairs can swap order. The 0.9849 Spearman says it nearly preserves ranks *on this sample*, not that it must.)

## Newly-discovered canon

- [Amari 1998, Neural Computation 10(2):251-276] — natural gradient / reparameterization invariance is the defining property of a principled metric; sample-covariance preconditioning is not the Fisher metric. (In `external_bibliography.md`.)
- [Cencov 1972] — Fisher–Rao is the unique invariant metric on a statistical manifold; whitening has no such standing. (Already cited by blue at `02_blue_opening.md:39`.)
