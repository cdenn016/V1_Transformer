# Verdict — semantic-clustering-correctness (binding, chief reconciliation)

## First-pass verdicts

| Judge | Outcome | Decisive evidence |
|-------|---------|-------------------|
| canon-strict | REMAND | `03_blue_rebuttal.md:7` (blue concedes the vocab view is a present accuracy defect, not a completeness gap) crossed against `01b_extended_evidence.md:5` + `external_bibliography.md:26,30` (Lee2013 product-manifold metric, Nakahara2003/Hall2015 left-invariant GL⁺(K) geodesic) — verified external canon splits across the two conjuncts. |
| code-truth   | BLUE_WINS | Executed under the re-traced active config with the exact builder `generate_glK_multihead_generators(20,2)` at `model.py:485`: bank off-block max = 0.0 (exact); `max\|expm(A_full) − blockdiag(expm(A_h))\| = 4.44e-16`; index alignment `extract.py:98,106` + `pipeline.py:62-80,182,193-223` verified off-by-one-free. The correctness sub-claims are true at the code level; the vocab defect is correctly computed and deferred to chief/scope as a frame question. |
| scope        | REMAND | `03_blue_rebuttal.md:7` — the defender's own concession that the claim's universal quantifier "none of which is a correctness bug" is false on the vocab figure; the compound claim packs a true geometry/semantics proposition with a false universal-quantifier proposition and must be split. |

## Reconciliation rule applied

**Rule 2 — scope override for REMAND on equivocation.** The scope judge declared REMAND because the claim packs multiple independently-truth-evaluable propositions under one "correct-by-design" headline whose load-bearing term "correct" equivocates between internalist math/code-correctness (true) and externalist figure-as-scientific-artifact correctness (false on the vocab panel) (`04_judge_scope.md:10,23,39`). Rule 2 fires first in priority order and stops there; the binding outcome is REMAND and I adopt the scope judge's sub-claim list (A/B/C/D) as the spawn list.

## Decisive evidence (binding)

`03_blue_rebuttal.md:7` — the defender's own concession: "On the vocab figure that wording is too strong, and Red is right to press it... The honest partition is: this is an accuracy defect on the vocab figure's *presentation*, not a completeness gap." When the defender concedes that the claim's universal quantifier ("none of which is a correctness bug") is false as written on one figure while the geometry and contextual-semantics conjuncts stand, the conjunction is non-atomic and cannot be ruled simply true or simply false. It must be split.

## Outcome (binding)

**REMAND**

Spawn the following four sub-claims (adopted verbatim from `04_judge_scope.md:53-59`):

- **Sub-claim A (geometry — BLUE-leaning, the user's first flagged phenomenon):** Under the active config (K=20, irrep_dims=[10,10], cross_couplings=[], block-diagonal bank), the φ-distance matrix and the Ω-geodesic distance matrix encode the same cluster structure (Spearman 0.952–0.972, both panels independently pick k=2), and the residual *visual* divergence between the `phi_vector_clustering` and `omega_clustering` panels is a non-isometric-projection (UMAP) layout artifact, not a geometry or index bug. The per-head quadrature `d=√(Σ_h d_h²)` is the exact product-manifold metric on GL(10)×GL(10), and the two Ω code paths agree to 4.4e-16. (PCA-whitening on φ is conceded non-canonical but rank-cost ≈0.005, so it does not break structure-agreement.)

- **Sub-claim B (contextual semantics — BLUE-leaning, the user's second flagged phenomenon):** The duplicate token labels (`the`, `of`, `Atlantic`) in the *contextual* figures are the defined per-occurrence semantics of the `(B,N)->(B·N)` flatten in `extract_contextual` (`extract.py:98`), with verified off-by-one-free index alignment `strings[i] ↔ coords[i] ↔ token_ids[i]`. This is correct-by-design data, not corruption.

- **Sub-claim C (vocab figure accuracy — RED-leaning, conceded by blue; NOT a phenomenon the user flagged):** The `vocab` figure is an *accuracy* defect, not a completeness gap: under cl100k_base the first-256-id extraction yields 107/200 U+FFFD, 23/200 empty, 72/200 unique (`01_evidence.md:54`), `_sanitize_label` does not strip the fallbacks (`pipeline.py:88-93`), and `_MAX_ANNOTATIONS=30` annotates the lowest (anti-salient) array-order ids (`plotting.py:224-235`). The claim's "none of which is a correctness bug" is false on this axis.

- **Sub-claim D (block-restriction guard — latent, agreed by both; completeness/robustness):** The unguarded block-restriction at `geometry.py:337-340` is exact under the active config (off-block max = 0.0) and is a *latent* correctness exposure that surfaces only under the opt-in `auto_close_cross_head_basis=True` toggle. A guard belongs there; this is a robustness gap, not a present bug.

## Reasoning

Rule 2 fired because the scope judge — who holds special standing for REMAND — declared REMAND on the ground of equivocation, supplying a concrete decomposition. The canon-strict judge independently reached the same REMAND on the same trigger (verified external canon splits cleanly across two conjuncts). The code-truth BLUE_WINS is not overridden so much as scoped: that judge adjudicated only the three correctness sub-claims (geometry math, contextual per-occurrence semantics, index alignment), found them true at the code level, and explicitly deferred the vocab-figure frame question to the chief and scope judge (`04_judge_code_truth.md:84,87`). Its BLUE verdict is therefore preserved within sub-claims A and B, which the REMAND records as blue-leaning. The binding evidence is the defender's own concession at `03_blue_rebuttal.md:7`: once blue grants that the universal quantifier is false on the vocab figure while the geometry and contextual-semantics conjuncts stand, the claim is no longer a single atomic proposition that can win or lose as a unit — it is a conjunction of a true part and a false part, which is the textbook REMAND condition.

## Action

The two phenomena you originally flagged are **correct-by-design — not bugs**:

- **φ/Ω visual divergence** (`phi_vector_clustering` vs `omega_clustering` looking different despite `Ω = exp(Σ_c φ_c G_c)`): correct-by-design. The two distance matrices agree at Spearman 0.952–0.972 and both independently pick k=2; the visual gap is a UMAP non-isometric-projection layout artifact (UMAP carries free global rotation/reflection and is not distance-preserving). The per-head quadrature is the exact product-manifold metric on GL(10)×GL(10), and the two Ω code paths agree to 4.4e-16. No geometry or index bug.
- **Duplicate `Atlantic`/`the`/`of` labels in the contextual figures**: correct-by-design. `extract_contextual` flattens `(B,N)->(B·N)` (`extract.py:98`), one row per occurrence; a recurring token type legitimately yields multiple identical-labeled rows. Index alignment is verified off-by-one-free. This is per-occurrence semantics, not corruption.

**(i) CONFIRMED CORRECT — no action.** Sub-claims A and B (the two phenomena you flagged). The geometry math, the per-head geodesic quadrature, the two-Ω-path agreement, the contextual per-occurrence flatten, and the full index-alignment chain are correct under your active config.

**(ii) REAL DEFECT to FIX — sub-claim C, vocab figure. Severity: presentation/accuracy defect (shipped-figure integrity), not a math or alignment bug.** The arithmetic on each row is correct, but the figure licenses a false reader inference (≈65% unrenderable glyphs; the 30 annotated points are the lowest, least-meaningful array-order ids). Three concrete fixes:
1. In `_sanitize_label` (`pipeline.py:88-93`), strip or flag U+FFFD and empty strings so byte-fallback rows do not reach the figure as semantic labels.
2. In the annotation path (`plotting.py:224-235`), replace the array-order `range(min(...,30))` selection with salience/cluster-representative point selection.
3. In the entry point (`run_semantic_clustering.py:66`, the cl100k_base id range), choose an informative id range that skips the byte-fallback band — or document the byte-fallback limitation in the figure title/caption and scope it honestly.

**(iii) LATENT GUARD — optional, sub-claim D.** The block-restriction at `geometry.py:337-340` is numerically exact under your active config (off-block = 0.0, so block-restrict-then-exp equals full-exp to 4.44e-16). It is unguarded: before anyone sets `auto_close_cross_head_basis=True` (which can add super-block-spanning generators), add an assertion that the discarded off-block entries of `A_full` vanish, or fall back to full-matrix exp. Cheap to close; not a present bug.

Separately and outside the binding sub-claim set: the docstring at `geometry.py:290` carries a wrong-domain "affine-invariant" label (the code computes a left-invariant GL⁺(K) group geodesic, not the SPD-cone congruence distance). Optional cleanup, not load-bearing for any sub-claim.
