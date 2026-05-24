# Verdict (canon-strict) — semantic-clustering-correctness

## Evidence audit

Citations weighted per the canon-strict rubric: verified external canon ×3, unverified external ×1, executed verification ×2, `path:line` ×1, canon-cop strike −1 each. Manuscript/CLAUDE.md-as-authority: none cited by either side (canon-cop confirms, `02b_canon_cop_openings.md:18,22`).

| Side | External citations (verified) | External citations (unverified) | sympy/FD / executed | path:line | Canon-cop strikes |
|------|------------------------------|--------------------------------|----------|-----------|-------------------|
| Red  | 3 — Cencov1972 (Fisher uniqueness, `external_canon_math.md:24`), Amari1998 (Adam/RMSProp ≠ natural gradient, `external_canon_math.md:61,135`), Bishop2006 §12.1 (whitening not distance-preserving) | 3 — Popper1959 (falsifiability), Tufte1983 (graphical integrity), Wilson2017 (figures-as-deliverable); real + in-domain for the framing axis, not in the math bibliography | ~4 (uses the shared executed pack: byte-fallback decode, array-order annotation, off-block=0.0) | ~6 (geometry.py:337-340,349-352; pipeline.py:88-93,111-122; plotting.py:224-235; model.py:447-454) | **2** (Pennec wrong-domain, `02b_canon_cop_openings.md:10-12`) — conceded + withdrawn `01b_extended_evidence.md:69-73`, `03_red_rebuttal.md:9` |
| Blue | 4 — Lee2013 (product-manifold metric, `external_bibliography.md:30`, `01b_extended_evidence.md:5`), Hall2015 (block-diag exp / BCH, `external_canon_math.md:108-112`), Nakahara2003 §5-6/§10.3 (left-invariant matrix-Lie-group geodesic, `external_bibliography.md:26`), McInnes2018 (UMAP non-isometry) | 1 — Rousseeuw1987 (silhouette heuristic, used correctly) | ~6 (Spearman 0.952/0.972, exp/BCH decomposition 0.9696, whitening 0.9849, 4.4e-16 path agreement, off-block 0.0, k=2 both panels) | ~6 (geometry.py:339,349-356; pipeline.py:62-80,119-122,182,193-222; extract.py:98-106; config.py:893-902) | **0** |

## Concessions made

- **Red conceded:** Pennec/Fillard/Ayache 2006 withdrawn as the canon for `omega_geodesic_distances` (wrong manifold — SPD cone Sym⁺(K) under congruence `P→APAᵀ`, not GL⁺(K) group elements under left translation `Ω→AΩ`); the per-head quadrature `d=√Σ d_h²` is the exact product-manifold metric [Lee2013]; block-diagonal exp factors exactly [Hall2015 §2-3]; the two Ω code paths agree at 4.4e-16 structurally; index alignment is verified off-by-one-free, so the contextual duplicate labels are per-occurrence semantics, not corruption (falsification condition (b) dead). Red explicitly conceded the quadrature and the index alignment in its own opening (`02_red_opening.md:21`).
- **Blue conceded:** PCA-whitening on φ (`geometry.py:215-270`, `whiten=True`) is not a canonical Lie-algebra metric — not the Killing/Frobenius form at the identity, not Cencov-invariant, not Fisher (`02_blue_opening.md:39`, anchored to `external_canon_math.md:24,61`). Separately and substantively: the vocab figure is "an accuracy defect on the vocab figure's *presentation*, not a completeness gap" — the claim's blanket "none of which is a correctness bug" is "too strong" on the vocab view (`03_blue_rebuttal.md:7`).

## Decisive evidence

`03_blue_rebuttal.md:7` (blue's own concession that the vocab view is an accuracy defect, not a completeness gap) crossed against `01b_extended_evidence.md:5,15-21` + `external_bibliography.md:26,30` (Lee2013 product-manifold metric, Nakahara2003/Hall2015 left-invariant GL⁺(K) geodesic — verified, in-domain canon for the geometric portion that red could not dislodge after the Pennec strike). Both are verified external-canon-grade; they bear on **different conjuncts** of the load-bearing claim.

## My weighted scores

Scoring the two conjuncts separately because that is how the canon splits.

**Geometric portion** (the two flagged phenomena — φ/Ω divergence is a projection artifact; contextual duplicates are per-occurrence semantics):
- Blue: 4 verified external ×3 = 12; executed ×2 (Spearman + path agreement, the load-bearing two) = 4; path:line ×1 ≈ 4. Subtotal ≈ **20**.
- Red: surviving external on this portion = Cencov/Amari/Bishop (whitening-not-canonical) ×3 = 9, but this is the point blue *conceded*, so it scores red zero marginal — it does not establish that the comparison is *invalid*, only that whitening is non-canonical (blue's measured 0.9849 whitening cost bounds the damage at ~0.005); minus the −2 Pennec strikes. The only named target-distance class for Ω was wrong-domain and withdrawn. Subtotal ≈ **2 net** (executed evidence red shares, minus strikes).
- On the geometric portion canon is one-sided: **Blue carries it decisively.** Red has no surviving external canon establishing the Ω geodesic is mis-derived or in an incompatible class (its falsification condition (a) requires this; it fails to meet it).

**Completeness-framing portion** ("the genuine defects are confined to presentation/completeness … none of which is a correctness bug"):
- Red: Tufte1983 + Wilson2017 (figure-as-deliverable, graphical integrity) + Popper1959 (unfalsifiability of relabel-as-completeness) — in-domain framing canon ×1 each ≈ 3; plus the executed vocab evidence (107/200 U+FFFD, array-order anti-salient annotation) ×2 ≈ 4. Subtotal ≈ **7**.
- Blue: concedes this conjunct (`03_blue_rebuttal.md:7`): the vocab view is a present accuracy defect. Blue does not defend "none of which is a correctness bug" for the vocab figure. Subtotal on this conjunct ≈ **0 defended**.
- On the framing portion: **Red carries it**, by blue's own concession plus in-domain figure-integrity canon.

## Outcome (this judge)

**REMAND**

## Reasoning

The load-bearing claim is a conjunction — "(i) correct-by-design on the two flagged phenomena AND (ii) the genuine defects are confined to completeness, none of which is a correctness bug." Verified external canon splits cleanly across the two conjuncts, which is the exact trigger for REMAND in my rubric ("weighted totals near tie AND both sides cite verified external canon on different parts of the claim"). On conjunct (i), blue holds the only surviving canon: Lee2013 makes the per-head quadrature the *exact* product metric, Nakahara2003/Hall2015 place `‖log(Ωᵢ⁻¹Ωⱼ)‖_F` as a principled left-invariant GL⁺(K) geodesic, and McInnes2018 supplies the non-isometry that makes the φ/Ω visual divergence a genuine projection artifact while the Spearman 0.952–0.972 + dual-k=2 certifies the structure agrees — and red's only attempt to name a target distance class for Ω (Pennec, congruence-invariant SPD-cone) was wrong-domain, drew 2 canon-cop strikes, and was withdrawn (`03_red_rebuttal.md:9`). The two flagged phenomena are therefore correct-by-design and survive canon scrutiny. On conjunct (ii), the claim's blanket "none of which is a correctness bug" is defeated by blue's own concession (`03_blue_rebuttal.md:7`) that the vocab view is a present accuracy defect, not a completeness gap, backed by red's in-domain figure-as-deliverable canon [Tufte1983, Wilson2017] and the executed evidence that the figure annotates anti-salient byte-fallback glyphs (107/200 U+FFFD, array-order selection). I do not round this up to BLUE_WINS: that would silently drop the broken conjunct, which my rubric forbids, and the task's tie-break instruction ("on close calls err toward RED_WINS or REMAND") points the same way. I do not award RED_WINS either: red's surviving canon does not touch the geometric portion at all (its whitening attack was conceded as non-canonical-but-bounded, not invalidating), so it has not won the substantive correctness question — it has only shown the *framing* over-reaches on one figure. The honest disposition is REMAND on the conjunction.

Focused follow-up question for re-dispatch: **Is the load-bearing claim severable — geometric correctness of the two flagged phenomena (canon-supported, BLUE) decoupled from the framing assertion "none of which is a correctness bug" (canon-defeated by blue's own vocab concession, RED) — or does the conceded vocab-view accuracy defect defeat the claim as a single atomic proposition?** If severable, the geometric sub-claim should be recorded BLUE and the framing sub-claim recorded RED with the required fix.

## Action

Split the claim into two adjudicated sub-claims. (1) Accept the geometric sub-claim: the φ/Ω visual divergence is a projection-layout artifact [Lee2013, Nakahara2003, Hall2015, McInnes2018] and the contextual duplicate labels are per-occurrence semantics — both canon-correct under the active config. (2) Reject the framing sub-claim "none of which is a correctness bug": amend the claim to record the vocab-view label/annotation defect as a presentation *accuracy* defect (blue's concession, `03_blue_rebuttal.md:7`), and fix the code — strip/flag U+FFFD and empty labels in `_sanitize_label` (`pipeline.py:88-93`), replace array-order annotation with salience/cluster-representative selection (`plotting.py:224-235`), and choose an informative vocab id range or document the byte-fallback limitation in the figure. Separately, add the unguarded-block-restriction runtime guard at `geometry.py:337-340` (assert off-block ≈ 0 or fall back to full-matrix exp) before `auto_close_cross_head_basis=True` is exercised — latent, not present, but cheap to close. Correct the wrong-domain "affine-invariant" docstring label at `geometry.py:290` to "left-invariant GL⁺(K) group geodesic."
```
