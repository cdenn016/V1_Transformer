# Blue Memo — Information Geometer (Phase 3 Rebuttal)

## (i) Concession from red's opening

Red's distinction between Cencov-uniqueness-of-the-Fisher-metric [Chentsov 1982; Bauer-Bruveris-Michor 2016, *Bull. London Math. Soc.* 48: 499–506] and uniqueness-of-the-divergence is correct. The Fisher–Rao metric is unique up to a positive scalar on a statistical manifold; the f-divergence class (KL, Hellinger, Rényi-$\alpha$, $\chi^2$, total variation) is consistent with the Fisher metric and KL is one member among many [Liese-Vajda 1987]. Sinkhorn-style entropy-regularized soft assignment recovers softmax from optimal-transport functionals with no Fisher input [Cuturi 2013, "Sinkhorn distances," NeurIPS]. Calling the resulting attention a "geometric necessity" without scope qualification *is* a uniqueness-trade. The figure caption at line 1411 trades.

I concede: the figure caption is overstrong.

## (ii) Strongest defense against red's core attack

The body text immediately surrounding the figure caption *makes the correct distinction red is using against the caption*. Line 1252 states: "Within the f-divergence class satisfying assumptions (i)–(iii) of Appendix~\ref{app:conditional_uniqueness}, the forward KL divergence $D_{\mathrm{KL}}(q_i \| \Omega_{ij}q_j)$ is the unique f-divergence that preserves exponential-family closure under linear coupling and yields a consistent dual interpretation for the attention weights." This is **conditional uniqueness within the f-divergence class under three displayed assumptions**, not Cencov-style metric uniqueness. The conditional theorem is the correct shape — it has the form Csiszár/Liese-Vajda uniqueness theorems take [Liese-Vajda 1987; Pacheco-Sasai 2024, arXiv:2402.05014]: uniqueness within a divergence class under assumption-list constraints.

The body text *also* states the assumptions are post-hoc: line 1044 says "The choice of forward KL is justified post-hoc by the conditional uniqueness theorem." Calling the justification "post-hoc" is exactly the disclosure red's Popper-style demarcation demands. A construction that names its own uniqueness theorem as a post-hoc justification rather than a derivation is *not* equivocating; it is disclosing.

Where red's strike has bite is the gap between body and caption. The honest path for blue is: concede caption-level revision needed; defend that the body text at lines 1029, 1044, 1064 (within-framework labeling), 1252 (conditional uniqueness statement), and the appendix referenced at line 4259 (`app:conditional_uniqueness`) jointly carry the operationalization sub-claim 4 (falsifiability and scope) at the section level. Sub-claim 4 reads "Empirical/analogy claims (...) are labeled as analogy where they are analogy and as derivation where derivation is supplied." The body labels the ansatz as ansatz; only the figure caption mislabels. A figure-caption revision is a minor revision, not a section-level rejection.

## (iii) Counter-attack on red's weakest evidence

Red invokes Cuturi 2013 to argue softmax-from-energy is recoverable without information-geometric input — therefore the "geometric necessity" framing trades on a stronger uniqueness than the canon supplies. This argument carries against the figure caption. It does *not* carry against the actual derivation in §1.13. The manuscript's derivation does not claim that softmax-from-energy is uniquely Fisher-geometric in general; it claims, under the appendix-conditional assumptions, that forward KL is the unique f-divergence with the displayed closure properties. Cuturi 2013 derives softmax from entropy-regularized OT, which is *also* a closure property of a specific energy functional — it does not contradict the Csiszár-style conditional uniqueness inside the f-divergence class. The two results are compatible: the appendix-conditional theorem is uniqueness within f-divergences under linear-coupling closure; Cuturi-Sinkhorn is uniqueness within OT under entropy regularization. They are different uniqueness statements on different domains.

A reviewer who reads the body text and the appendix-conditional theorem will see a defensible Csiszár-style conditional uniqueness statement. A reviewer who reads only the figure caption will see overreach. The remedy is to revise the caption.

## Newly-discovered canon

- `[Csiszár-Shields 2004]` "Information Theory and Statistics: A Tutorial," *Foundations and Trends in Communications and Information Theory* 1(4): 417–528. Canonical modern tutorial on the f-divergence class and its uniqueness theorems under assumption-list constraints. Already in `external_canon_inference.md` via the LieseVajda1987 lineage but the Csiszár-Shields entry is the cleaner pedagogical reference for the appendix-conditional-uniqueness shape.

Append candidate for `external_canon_inference.md` §10 (pitfalls): "softmax-from-energy is recoverable from multiple energy functionals (KL-of-Gaussians under f-divergence assumptions; entropy-regularized OT [Cuturi 2013]); uniqueness claims must specify the closure class."
