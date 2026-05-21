# Blue Memo — Gauge Theorist (Phase 3 Rebuttal)

## (i) Concession from red's opening

Red's gauge-theorist memo (per the harvested canon trail) argues that the edge-mode and quantum-reference-frame literature [Donnelly-Freidel 2016; Bartlett-Rudolph-Spekkens 2007; Vanrietvelde 2020; Witten 2018; Carrozza-Hoehn 2022] supplies *constructive symplectic machinery* — boundary actions, von Neumann subalgebras, post-selection — that the manuscript's appeal at line 569 invokes by name without exhibiting. This is correct as stated. The manuscript does not construct an explicit boundary symplectic form, nor a von Neumann subalgebra, nor a post-selection formula for its statistical-manifold gauge frames.

I concede: the edge-mode citation at line 569 is a *combined-license appeal*, not a transfer of the constructive apparatus.

## (ii) Strongest defense against red's core attack

The combined-license appeal is *legitimate* if its scope is the dual-role labeling of $\phi_i$ (gauge-redundant + physical-state) rather than the inheritance of constructive content. Lines 555–571 of the manuscript define $\phi_i$ in two roles: **Role A** as gauge-redundant transport encoded in $\Omega_{ij} = e^{\phi_i}e^{-\phi_j}$, and **Role B** as a physical state object on the model fiber. The edge-mode literature licenses this *type* of dual treatment in gauge-theoretic settings — at the boundary of a subregion, would-be-gauge degrees of freedom become physical [Donnelly-Freidel 2016, *JHEP* 09: 102]. The manuscript appeals to this licensing for the type of move, not for the specific symplectic-boundary-action machinery the QFT literature constructs.

This is a *philosophy-of-citation* question, not a *math-content* question. A citation that licenses a *type of move* is standard in physics literature — citing Goldstone's theorem to license labeling some mode as "Goldstone-analog" is the usual pattern even in papers that don't construct the SSB potential explicitly. The Witten 2018 / Carrozza-Hoehn 2022 references red supplies are *deeper* references for the QFT machinery; they do not invalidate the type-level appeal. The reviewer test is whether the dual role is *self-consistent within §Theory's setting*, not whether the manuscript inherits the full QFT apparatus.

The cocycle structure $\Omega_{ij}\Omega_{jk} = \Omega_{ik}$ that line 1209 establishes (with its 2-line proof delegated to the companion) is the mathematical content of Role A. It is correct as stated under $\Omega_{ij} = e^{\phi_i}e^{-\phi_j}$. The lattice gauge / Wilson-line argument [Wilson 1974, *Phys. Rev. D* 10] supplies the standard invariance argument: $\mathrm{tr}(\Omega_{ij}\Omega_{jk}\Omega_{ki}) = \mathrm{tr}(I) = K$ trivially under the vertex-frame parameterization, and trace cyclicity gives invariance under the residual constant-per-agent subgroup of $\mathrm{GL}(K)$. This is the holonomy content the manuscript needs at lines 1201–1224, and it is *standard*; the companion paper does not supply anything beyond the Wilson-1974 lattice-gauge baseline.

## (iii) Counter-attack on red's weakest evidence

Red's per-citation count of `Dennis2025trans` at 14 includes **four citations specifically tied to the multi-head / RoPE block-diagonal gauge identification** (lines 1623, 1818, 1875, and the multi-head reference at 1702). These four are *not* derivational gaps inside §Theory's core scope — they are *extensions* to the multi-head case after the single-block GL($K$) construction is complete. The single-block construction is fully derived in §Theory; the multi-head extension is the natural consequence of "the gauge group of the multi-head architecture is the block-diagonal subgroup $\mathrm{GL}(d_{\text{head}})^H \subset \mathrm{GL}(d_{\text{model}})$" which is a one-line linear-algebra observation, not a derivational step requiring delegation.

The actual content delegated to the companion paper in these four citations is the **engineering treatment of how multi-head attention with rectangular $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}$ relates to the thin-SVD decomposition $M_h = U_Q A_Q (U_K A_K)^\top$**. This is implementation-side content that belongs in a companion paper *as the organizational choice*. The §Theory section's job is to identify the gauge group; the companion's job is to implement the projections that realize it. The reviewer who insists that *every* extension of the theory be derived inside §Theory is not applying a reasonable publication standard — they are demanding a 200-page paper.

Red's falsification condition (ii) requires that the companion-citation pattern occur "at load-bearing reduction steps inside its theory section." The four multi-head citations are at extension steps *after* the core reduction completes. The honest count under the strict load-bearing criterion (see the variational memo for the per-citation reclassification) is 5, not 14; the threshold red set (≥10) is not crossed by the operative citation pattern.

## Newly-discovered canon

- `[Wilson 1974]` is already in the Phase 2 blue harvest (the lattice-gauge invariance argument). No new canon.
- For the dual-role gauge-frame discussion, `[CarrozzaHoehn 2022]` (red harvest) and `[Witten 2018]` (red harvest) are the modern bridges. Both are *symplectic-construction* references; if blue wants to claim full constructive transfer to the statistical-manifold setting, blue would have to construct the symplectic form. Blue does not so claim — the appeal is at the type-of-move level only.
