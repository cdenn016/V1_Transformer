# Verdict (canon-strict) — pifb-consensus-gauge-invariance

## Evidence audit

Verified = key present in `external_bibliography.md` or fact stated in `external_canon_*.md`.
Unverified = named in `01_evidence.md`/`01b_extended_evidence.md` as canon-to-confirm but NOT
tagged in `external_bibliography.md` (the bibliography lists Peskin & Schroeder and Weinberg only
under "Coverage gaps — extend on demand"; Folland and Knapp are not listed at all). Per the
citation-hygiene note in `01b_extended_evidence.md:37–43`, the P&S/Folland/Knapp facts were
confirmed at source level via web search but carry no bibliography tag.

| Side | External citations (verified) | External citations (unverified) | sympy/FD | path:line | Canon-cop strikes |
|------|------------------------------|--------------------------------|----------|-----------|-------------------|
| Red  | 4 — [Cencov1972] (`external_canon_math.md:24`), [AmariNagaoka2000], [Nakahara2003 §10.4], [Friston2010] | 4 — [PeskinSchroeder1995 §9.4], [Folland], [Knapp], Ay–Jost–Lê–Schwachhöfer 2015 arXiv:1207.6736; + Popper/SEP and Wikipedia (orientation-only) | 1 — closed-form Gaussian KL invariant under μ→gμ, Σ→gΣgᵀ on K=4 to 4×10⁻¹⁵ | 0 | 0 (no canoncop report exists) |
| Blue | 5 — [Cencov1972] (`external_canon_math.md:24`), [AmariNagaoka2000], [Nakahara2003 §10.1–10.4], [KobayashiNomizu Vol. I §III.2], [Friston2010] | 3 — [PeskinSchroeder1995 §9.4], standard Haar-measure theorem (Folland-type), arXiv:hep-th/0103160 (Map(C,G) not locally compact) | 0 | 0 | 0 (no canoncop report exists) |

No `Attention/*.tex`-as-authority or `CLAUDE.md`-as-authority strikes on either side: both teams
cite `:2954–2992` as the claim under evaluation (weight 0, neutral), not as canonical authority.
Grep of both openings for authority-citation patterns returned no matches.

## Concessions made
- Red conceded: the three geometric obstructions are correct [Nakahara2003 §10.1–10.4;
  PeskinSchroeder1995 §9.4; Folland/Knapp] (`03_red_rebuttal.md:4–20`); the consensus-metric
  downgrade to "heuristic target … conditional on a regulator" is honest (:2986); the manuscript
  *intends* and states the non-derivational reading ("rather than a derivation," :2992) is on the
  page (`03_red_rebuttal.md:29–33`); the "objective reality" prose is genuinely conditionalized at
  :2986 (`03_red_rebuttal.md:99–101`).
- Blue conceded: the production verbs "arises" (:2990) and "emerges" (:2992) overclaim and require
  a one-word-per-sentence trim to "is consistent with" (`03_blue_rebuttal.md:8–22`); the
  U(1)/SU(2)/SU(3)/SO(1,3) enumeration overclaims by suggestion and should be trimmed to the bare
  containment fact or deleted (`03_blue_rebuttal.md:24–32`); an inline hedge is needed at the
  earlier "objective reality" prose location (`03_blue_rebuttal.md:98–102`).

## Decisive evidence
**[Cencov1972]**, verified at `external_canon_math.md:24` ("the Fisher metric is the unique (up to
scalar) Riemannian metric on a statistical manifold invariant under sufficient statistics") and
`external_bibliography.md:20`. This is the only load-bearing external canon entry on the live
sub-point (sub-point 4). It establishes that the GL(K,ℝ)-invariance of Gaussian KL is intrinsic to
the divergence, present at N=1 before any consensus. That fact CONFIRMS the manuscript's own
categorical self-label at :2992 ("a metaphysical interpretation **rather than a derivation**") —
it does not contradict it. The manuscript line :2992, read directly, contains both "rather than a
derivation" and "may not be falsifiable" in the same sentence; the claim in `00_claim.md` asserts
exactly the conjunction of those two phrases.

## My weighted scores
Scoring the LIVE sub-point only (sub-points 1–3 are conceded by both sides as correctly stated and
score equally for both; they do not break the tie). Weights: verified external 3×, unverified 1×,
sympy/FD 2×.

- Red weighted total (live sub-point): Cencov 3 + AmariNagaoka 3 + arXiv:1207.6736 (unverified) 1
  + KL-invariance FD check 2 = **9**, all of which establish "not a derivation" — i.e., they
  support the manuscript's own flag rather than contradict the claim. Red's class-(a)/class-(b)
  Popper distinction rests on Popper/SEP, which is orientation-only (Wikipedia/SEP, not
  bibliography-tagged) and adjudicates the manuscript's *word choice*, not an external canonical
  form. As canon-contradiction of the claim: **0**.
- Blue weighted total (live sub-point): Cencov 3 + AmariNagaoka 3 + Friston2010 3 = **9** in direct
  support of "the invariance is intrinsic, therefore the non-derivational flag is the correct
  characterization, and the manuscript states it."

## Outcome (this judge)
BLUE_WINS

## Reasoning
The RED_WINS gate in my rubric requires "at least one verified external canon citation that
contradicts the claim." Red's strongest and only verified external canon on the live sub-point is
Cencov/Chentsov [Cencov1972, `external_canon_math.md:24`] plus the corroborating
[AmariNagaoka2000] and the FD check — and every one of these establishes that GL(K)-invariance is
intrinsic to the divergence, which is precisely the content of "rather than a derivation." That
canon supports the claim; it does not contradict it. Red's actual wedge — that the production verbs
"arises/emerges" perform a derivation the flag disowns, and that "may not be falsifiable" names a
class-(a) defect when the real defect is class-(b) circularity — is a textual-consistency and
word-choice argument grounded in Popper/SEP, which is orientation-only and not bibliography-tagged
canon; it is a lexical-hermeneutic dispute about the manuscript's own wording, not an external-canon
contradiction. That dispute is the scope judge's lane, not mine. The BLUE_WINS gate requires a
verified external canon citation supporting the claim: blue supplies five (Cencov, AmariNagaoka,
Nakahara §10.1–10.4, KobayashiNomizu, Friston2010), the manuscript line :2992 literally contains
"a metaphysical interpretation rather than a derivation … may not be falsifiable," and blue
concedes the verb trim and the SM-enumeration trim that red presses. On uncompromising external-
canon adherence the canon points one way: the obstructions are canonically correct and the
non-derivational flag is canonically vindicated by Cencov. The residue red identifies is real but
lexical (verb choice, defect-naming), discharged by local edits blue already concedes, not a
canon-level defeat of the claim.

## Action
Adopt the claim as substantially correct (the three obstructions are canonically correct; the
non-derivational flag is canonically vindicated by [Cencov1972]). Three local manuscript trims
survive as required follow-up, all conceded by blue and none altering the canon verdict:
1. :2990 "gauge invariance **arises** as a consistency requirement" and :2992 "it **emerges** from
   the informational requirements of consensus formation" → soften the production verbs to "is
   consistent with" / "requires," so the verbs match the same-sentence "rather than a derivation"
   disclaimer.
2. :2992 — trim the U(1)/SU(2)/SU(3)/SO(1,3) "available as subgroups" enumeration to the bare
   containment fact (every compact Lie group embeds in some GL(n); containment selects nothing) or
   delete it.
3. Add an inline conditional hedge or forward cross-reference at the earlier "closest analog to
   objective reality" prose, which is conditionalized only retroactively at :2986.

Note for the chief: red's class-(a)/class-(b) Popper distinction (circularity vs unfalsifiability)
is a well-posed frame-check on the *precision* of the manuscript's self-flag and belongs to the
scope judge's adjudication. If scope reads `00_claim.md` as requiring the flag to name the defect
*correctly* (circularity, not unfalsifiability) rather than merely to label the thesis
non-derivational, scope may push REMAND on that hermeneutic axis. On the external-canon axis this
judge owns, the canon vindicates the claim and BLUE_WINS narrow.
