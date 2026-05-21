# Canon-cop report — pifb-theory-rock-solid — rebuttal (Phase 3) — red

## Summary

Total strikes: 0
Action: RECORD (debate continues)

## Grep pass (canon_cop_validator.py)

```json
{
  "target": "docs/debates/2026-05-21-pifb-theory-rock-solid/03_red_rebuttal.md",
  "total_strikes": 0,
  "action": "RECORD",
  "manuscript_authority_hits": [],
  "citation_checks": [],
  "attention_citation_count": 0,
  "claude_md_citation_count": 0,
  "external_citation_count": 0
}
```

The grep validator's `attention_citation_count` is 0 because the rebuttal cites manuscript line numbers in plain prose (`"manuscript line 1458"`, `"line 1894"`, `"line 552"`) rather than `Attention/<file>.tex` paths. The single explicit `Attention/*.tex` reference in the citation summary at line 45 is meta-commentary explicitly disclaiming any manuscript-as-authority usage. All manuscript line references in the body are locating statements being attacked, not citing canonical forms — consistent with the operator's stated design.

## LLM pass — subtle patterns

| Pattern | Line | Strikes | Note |
|---------|------|---------|------|
| Manuscript-as-authority for canonical form | — | 0 | All canonical forms (sandwich rule, Gaussian KL, envelope theorem, generalized eigenvalue problem) are cited from external canon (Nakahara, Blei et al., Kingma-Welling, Milgrom-Segal, Arnold 1989, Goldstein-Poole-Safko, Vaswani). Manuscript lines appear only as the *location* of statements being attacked. |
| Implicit "our framework establishes" | — | 0 | The single "by construction" occurrence at line 29 is the rebuttal *quoting* manuscript line 2064 admitting the definitional circularity — i.e., red is attacking the construction-circularity, not committing it. |
| Fabricated `[Author Year §X]` | — | 0 | Citations not in `external_bibliography.md` (Arnold1989, Popper1959, GoldsteinPooleSafko2002, Weinberg1995, Sonderby2016, DonnellyFreidel2016, BartlettRudolphSpekkens2007, Vanrietvelde2020, Rovelli1996, Witten2018, CarrozzaHoehn2022, RanganathTranBlei2016, Milgrom-Segal 2002) are real, well-known external references in the appropriate domains, and the bibliography's "Coverage gaps — extend on demand" section explicitly sanctions on-demand extension to standard references not pre-listed. The Popper §15 attribution at line 17 is a slightly loose section pointer (Popper §15 is "Strictly Universal and Numerically Universal Statements" — the post-hoc-revision argument is implicit, not headlined) but the cite is real and the spirit of the claim is supported by Popper's broader thesis. Not strike-worthy under the rubric. |
| Wrong-domain citation | — | 0 | Sønderby 2016 (line 27) is cited *to attack* blue's misuse — red argues correctly that Ladder VAE uses finite-variance Gaussians, not the σ²→0 deterministic-decoder limit. This is the right paper for the claim. Witten 2018 / Donnelly-Freidel 2016 / Carrozza-Höhn 2022 at line 35 are the standard edge-modes references and the use (citing them for the symplectic/conjugate-variable/boundary-action machinery red claims is missing from blue's combined-license cite) is appropriate. |
| Reasoning-by-construction circularity | — | 0 | No instance. |
| Hand-wave-with-citation | — | 0 | Every cite is load-bearing for the specific claim. The Arnold 1989 GTM 60 Ch. 5 §22–25 cite for the generalized eigenvalue problem with independent T and V is the standard reference. Goldstein-Poole-Safko Ch. 6 is the small-oscillations chapter — correct for the mass-analogy attack. |

## Banned-phrase scan

| Phrase | Line | Note |
|---|---|---|
| (none found) | — | The red rebuttal contains no instances of `key insight`, `crucially`, `critically`, `notably`, `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores`, `perhaps`, `it could be argued`, `one might suggest`, or `both sides have a point`. |

## Decision

Soft-cap status: **0 strikes — record, debate continues.** No mandatory rewrite. The judges may weight the rebuttal's citation discipline as a positive factor; the canon-cop has no further constraint to impose.
