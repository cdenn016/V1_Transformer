# Canon-cop report — pifb-theory-rock-solid — opening (Phase 2.5) — blue

## Summary

Total strikes: 0
Action: RECORD (debate continues)

## Panel-staffing verification (per user's special note)

Both 5-memo + panel_choice deliverables exist on disk:

- `02_blue_panel_choice.md` (5 expert tags + one-sentence justifications + rejection rationale)
- `02_blue_memo_geometer.md`
- `02_blue_memo_info-geometer.md`
- `02_blue_memo_variational.md`
- `02_blue_memo_gauge-theorist.md`
- `02_blue_memo_philosophy-of-science.md`

`philosophy-of-science` is included as required. As on the red side, the coordinator authored the memos directly because the Agent dispatch tool was unavailable in its roster, but the methodology's substantive requirement — five memo files with one-sentence panel-choice justifications logged — is satisfied. Staffing observation only; does not bear on canon-cop strike counting.

## Grep pass (canon_cop_validator.py)

```json
{
  "target": "docs\\debates\\2026-05-21-pifb-theory-rock-solid\\02_blue_opening.md",
  "total_strikes": 0,
  "action": "RECORD",
  "manuscript_authority_hits": [],
  "citation_checks": [],
  "attention_citation_count": 2,
  "claude_md_citation_count": 0,
  "external_citation_count": 0
}
```

The two `Attention/*.tex` tokens are the title-line slug and the closing-paragraph negation at line 47 ("No `Attention/*.tex` line cited as authority for canonical form — manuscript lines are cited only for the location of statements being defended, not for the canonical form itself"). Both are bookkeeping references, not citations of the manuscript as authority.

The validator's CITATION_RE expects bracketed `[Author Year §X]` with a space; blue uses concatenated `[Author2003]` form, so the regex returns no citation_checks. The LLM pass below cross-checks every external reference against `external_bibliography.md` and its "Coverage gaps — extend on demand" clause.

## LLM pass — subtle patterns

| Pattern | Line | Strikes | Note |
|---------|------|---------|------|
| Manuscript-as-authority for canonical form | n/a | 0 | Blue explicitly disclaims this at line 47: "No `Attention/*.tex` line cited as authority for canonical form — manuscript lines are cited only for the location of statements being defended, not for the canonical form itself." Spot-check of evidence items 1, 2, 3 confirms: each evidence item names a manuscript line as the *location* of a stated form, then matches it against external canon (`[Nakahara2003 §10.3]`, `[BleiKuckelbirgJordan2017]`, `[Milgrom Segal 2002]`, etc.). The canonical authority is external in every case. |
| CLAUDE.md cited as authority | n/a | 0 | No appeal to CLAUDE.md. |
| "Our framework establishes" / "by construction in this work" | n/a | 0 | Blue uses "the load-bearing geometric primitives ... match the canonical forms of [Nakahara2003]" — *match the canonical form*, not *the framework establishes the canonical form*. No by-construction circularity. |
| Fabricated `[Author Year]` | n/a | 0 | Citations not in `external_bibliography.md` (Milgrom Segal 2002, Sønderby2016, DonnellyFreidel2016, BartlettRudolphSpekkens2007, Vanrietvelde2020, Weinberg1995 Vol. II §19, Peskin Schroeder Ch. 11, Friston2017Graphical) are real, well-known papers. The bibliography's "Coverage gaps — extend on demand" clause at lines 74–87 explicitly authorizes this extension pattern and at line 79 explicitly names "Weinberg *Quantum Theory of Fields* Vol. II; Peskin & Schroeder *Introduction to QFT*" as the standard refs to fetch on demand for symmetry-breaking material — which is exactly what blue cites them for in evidence item 5 (Goldstone analogy caveat). The methodology at line 180 makes "Merge harvested canon" a required step; blue's citation summary closes with new canon harvested to `01b_extended_evidence.md` per that workflow. Friston2017Graphical and ParrPezzuloFriston2022 ARE in the bibliography. |
| Wrong-domain citation | n/a | 0 | Nakahara §10.3, Ch. 9–10 for fiber-bundle (2,0)-tensor transport under GL(K) — correct domain. Blei-Kucukelbir-Jordan 2017 / Kingma-Welling 2014 App. B for closed-form Gaussian KL — correct domain. Milgrom & Segal 2002 for the envelope-theorem covariance correction — correct domain (their theorem covers exactly the parametric-optimum gradient setting). Sønderby 2016 for ladder-VAE rigid-link prior limit — correct domain. Donnelly-Freidel 2016 for edge-mode / dual-role gauge frame — correct domain. Bartlett-Rudolph-Spekkens 2007 and Vanrietvelde 2020 for quantum reference frames / residual subgroup splits — correct domain. Weinberg Vol. II §19 and Peskin-Schroeder Ch. 11 for explicit-vs-spontaneous symmetry-breaking — correct domain. |
| Reasoning-by-construction circularity | n/a | 0 | Evidence items 1–6 all run from external canon → match → manuscript. No appeal to the construction as proof of the construction. The honest concession at lines 41–43 explicitly recognizes the sub-claim 5/6 vulnerability under strict literal reading rather than papering over it via by-construction defense. |
| Hand-wave-with-citation | n/a | 0 | Every external citation is load-bearing for the specific claim. Milgrom-Segal 2002 grounds the envelope-theorem covariance term verbatim. Sønderby 2016 grounds the rigid-link σ²→0 reduction of the cross-scale shadow. Donnelly-Freidel 2016 grounds the residual-subgroup edge-mode separation. None is window dressing. |
| Banned phrases (Claude-isms) | n/a | 0 | No instance of `key insight`, `crucially`, `critically`, `notably`, `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores`. |
| Banned phrases (debate hedges) | n/a | 0 | No instance of `perhaps`, `it could be argued`, `one might suggest`, `both sides have a point`. |

## Action

RECORD. Debate continues. Soft cap not approached.
