# Verdict (canon-strict) — pifb-signature-problem

> Filename note for the chief: the dispatch instruction named this file `04_judge_canon_strict.md`; the agent template names it `04_verdict_canon.md`. This is the canon-strict first-pass verdict regardless of filename.

## Evidence audit

| Side | External citations (verified) | External citations (unverified) | sympy/FD | path:line | Canon-cop strikes |
|------|------------------------------|--------------------------------|----------|-----------|-------------------|
| Red  | 4 — [Lee2013 Ch. 13]✓(biblio), [AmariNagaoka2000 Ch. 2]✓, [Cencov1972]✓, [Popper §6/§31–33] (standard, treated verified) | 4 — [O'Neill 1983 Ch. 2–3], [Knapp 2002 Ch. VI], [Evans 2010 §2.3–2.4], [Forster&Sober 1994], [John PDE Ch.7] | 5 (mixed-real generators; compact-generator (−,−); real-only (+,+); non-separable sign flip; complex-G degeneracy — `01b` red panel) | 0 code; manuscript-line refs used as fact-of-disclosure | 0 |
| Blue | 4 — [Horn&Johnson §4.5]✓(standard), [AmariNagaoka2000 Ch. 2]✓, [Cencov1972]✓, [Hall 2015 Ch. 5] (in canon_math) | 3 — [Knapp 2002 Ch. I], [Evans 2010 §2.3], [Streater-Wightman/Folland], [Popper §15] | 5 (trace algebra; rank/det; Lorentz invariance; Sylvester count; compact-generator tr=−2 — `01b` blue panel) | 0 code; manuscript-line refs used as fact-of-disclosure | 0 |

No canon-cop reports were filed for this debate (none present in the working dir). Neither side cited `Attention/*.tex` or `CLAUDE.md` as *authority for a canonical form*; both cite manuscript `.tex` lines only as evidence of the *fact of disclosure* (textual evidence, weight 1) — no −2 manuscript-as-authority strike applies.

## Concessions made

- **Red conceded:** the worked-example trace algebra (G_ττ=−2(∂_τψ_τ)², G_xx=+2(∂_xψ_x)², G_τx∈iℝ, complex det=0 rank-1 → Re(·) rank-2 (−,+)); the causal-cone Sylvester count [Horn&Johnson §4.5]; the standard group facts (SL(2,ℂ)≅Spin⁺(1,3) [Hall Ch.5], SO⁺(1,1) frame group with ΛᵀηΛ=η); the "existence demonstration, not derivation" framing as the manuscript's own honest self-description; and **4 of 5** disclosure items it had initially flagged (convention :2868, single-generator collapse :2872/:2902, mixed-real-generator route :2950 — all confirmed disclosed). Retracted its own gauge-theorist/philosophy "undisclosed" framing on those four.
- **Blue conceded:** (1) "structurally compatible with Lorentzian signature," taken as the bare unqualified phrase, "is stronger than the construction earns" — the correct statement is conditional on the imaginary-φ_τ postulate and the unmotivated rank-changing real-part projection (:2892); (2) the separability sign-flip consequence "is not spelled out at :2877" — "a genuine granularity gap in the completeness sub-claim ... and blue grants it"; (3) the causal-cone route "does not apply to the framework's implemented dynamics" (:2929 parabolic / infinite speed).

## Decisive evidence

**[Lee 2013, *Introduction to Smooth Manifolds* (2nd ed.), Ch. 13]** (verified in `external_bibliography.md`): a pseudo-Riemannian metric is by definition a smooth, symmetric, **non-degenerate, real** (0,2)-tensor. The object the framework's construction natively produces, G_μν=tr(A_μ A_ν), is complex-valued and rank-one (det=0 — the manuscript's own arithmetic at :2887). It fails the definition on two independent counts (degeneracy and non-reality). The Lorentzian form of Eq. `lorentzian_metric` (:2889) exists only after the real-part projection that the manuscript itself certifies at :2892 has "no physical principle in the construction that mandates" it. External canon (Lee) defines the target object; the manuscript admits the framework does not produce it without an analyst-applied, unmotivated step.

Supporting (weight 1, unverified-but-standard): [O'Neill 1983 Ch. 2–3] corroborates the metric definition. The completeness-clause defeater is **blue's own rebuttal text** ("a genuine granularity gap ... and blue grants it"), backed by the matching sympy on both panels (the non-separable Re(G_ττ) is sign-indefinite, weight 2).

## My weighted scores

Scored on the operative clauses of the compound claim (math-correctness clause; "structurally compatible with Lorentzian signature" framing clause; "accurately and completely disclosed" clause).

- **Math-correctness clause:** both sides + verified canon ([Horn&Johnson §4.5], [Hall Ch.5], [AmariNagaoka2000], [Cencov1972]) agree it is correct. Not contested. Neutral — does not move the verdict; it is an existence demonstration, conceded by both.
- **Framing clause + completeness clause** (the contested core):
  - Red weighted total: **+11**. Decisive [Lee2013]✓ ×3; [AmariNagaoka2000]✓ + [Cencov1972]✓ on the signature-decoupled-from-statistics corollary ×3 (counted once as the info-geometric pair, +3); non-separable sign-flip sympy ×2; [O'Neill]/[Knapp]/[Evans] unverified-standard support ×1 each (+3, capped at the load-bearing two = +3).
  - Blue weighted total on the contested clauses: **−2 net**. Blue scores positively only on the conceded math-correctness clause (already credited as neutral/shared); on the two contested clauses blue does not defend — it concedes both. Its rebuttal's surviving positive content ("every sensitivity is disclosed") is true for the generator/convention axis but is conceded-around by red and does not rescue the separability-sign axis, where blue grants the gap. Net contribution to the contested clauses is negative (forfeiture).

The asymmetry is not a tally artifact: blue forfeits two of the three operative clauses outright. Red carries them on verified external canon ([Lee2013]) plus blue's own concessions.

## Outcome (this judge)

**RED_WINS**

## Reasoning

The compound claim is conjunctive: the treatment is sound *as a correct existence demonstration* AND "structurally compatible with Lorentzian signature" AND every gap "accurately and **completely** disclosed." The math-correctness conjunct holds — verified against [Horn&Johnson §4.5] (Sylvester), [Hall Ch.5] (SL(2,ℂ)≅Spin⁺(1,3)), and reproduced by both panels' sympy — and red concedes it without reservation. But two conjuncts fail under verified external canon and blue's own concessions. The framing conjunct fails on [Lee2013 Ch. 13]: a metric is by definition real, symmetric, and non-degenerate; the framework natively yields a complex rank-1 (det=0) form, and the manuscript admits at :2892 that the rank-changing real-part projection delivering the Lorentzian form has "no physical principle ... that mandates" it — so the framework does not produce a Lorentzian metric, the analyst's projection does. The completeness conjunct fails on a single surviving red strike that blue explicitly grants: the separability ansatz at :2877 is disclosed as a display simplification but its sign consequence is not — dropping separability gives Re(G_ττ)=2((∂_τψ_x)²−(∂_τψ_τ)²) (sign-indefinite; sign-indefinite under either i-placement, so the conclusion is invariant to the manuscript's :2877 transcription quirk), so the temporal direction can become spacelike, which the gap-list omits while it does disclose the analogous sign consequence for the single-generator collapse at :2872. A conjunction with two failed conjuncts is false; the rubric forbids splitting the difference when one side carries verified external canon (Lee, in the bibliography) and the other concedes. This is not REMAND: the disagreement is same-part (both sides argue the framing and completeness clauses), not different-part, and blue forfeits rather than contests. The defensible residue — the bare existence-demonstration core blue itself flagged as "the live one" — survives, but the compound claim *as written* does not.

## Action

Manuscript edit, two surgical changes inside `Attention/Participatory_it_from_bit.tex:2863–2952`, no structural rework:
1. Narrow the framing wherever the section asserts "structurally compatible with Lorentzian signature" (e.g., :2907, :2932, :2952) to the conditional form the manuscript's own :2892/:2907 already support: compatible *conditional on the imaginary-φ_τ postulate and the unmotivated rank-changing real-part projection*. The existence-demonstration core is sound; the unqualified phrase is the overclaim.
2. Add one sentence at :2877 stating that dropping separability can render the temporal direction spacelike (Re(G_ττ)=2((∂_τψ_x)²−(∂_τψ_τ)²) is sign-indefinite), mirroring the sign-consequence disclosure already present for the single-generator collapse at :2872. This closes the one completeness gap blue conceded.

For the chief: clause-A/B (math/group facts) is undisputed and canon-confirmed; the verdict turns on clauses C (framing) and E (completeness), both of which blue concedes and red carries on [Lee2013]✓ + matching sympy. The code-truth judge has no jurisdiction (no code implements this section); expect the scope judge to flag the claim as a compound/conjunctive proposition — if scope declares REMAND on equivocation grounds (the claim packs "sound existence demonstration" together with "completely disclosed" and "structurally compatible"), that is a defensible alternative reconciliation, but on canon weighting the conjunction is false and red wins.
