# Canon-cop report — pifb-theory-rock-solid — opening (Phase 2.5) — red

## Summary

Total strikes: 0
Action: RECORD (debate continues)

## Panel-staffing verification (per user's special note)

Both 5-memo + panel_choice deliverables exist on disk:

- `02_red_panel_choice.md` (5 expert tags + one-sentence justifications + rationale + lead attack vectors)
- `02_red_memo_geometer.md`
- `02_red_memo_info-geometer.md`
- `02_red_memo_variational.md`
- `02_red_memo_gauge-theorist.md`
- `02_red_memo_philosophy-of-science.md`

`philosophy-of-science` is included as required. The coordinator authored the memos directly (not via Agent tool dispatch) because the Agent dispatch tool was unavailable in its roster, but the methodology's substantive requirement — five memo files exist with one-sentence panel-choice justifications logged — is satisfied. This is a staffing observation only and does not bear on canon-cop strike counting.

## Grep pass (canon_cop_validator.py)

```json
{
  "target": "docs\\debates\\2026-05-21-pifb-theory-rock-solid\\02_red_opening.md",
  "total_strikes": 0,
  "action": "RECORD",
  "manuscript_authority_hits": [],
  "citation_checks": [],
  "attention_citation_count": 1,
  "claude_md_citation_count": 0,
  "external_citation_count": 0
}
```

The one `Attention/*.tex` token detected by the file-path regex is in the title-line slug `Attention/Participatory_it_from_bit.tex` identifying the target manuscript under attack — not a citation of `Attention/*.tex` as authority for canonical form.

## LLM pass — subtle patterns

| Pattern | Line | Strikes | Note |
|---------|------|---------|------|
| Manuscript-as-authority for canonical form | n/a | 0 | All manuscript line citations (1880, 1882, 2064, 1252, 1411, 1406, 1029, 1044, 1064, 547, 552, 632, 634, 683, 1233, 1419, 1582, 1991, 636, 943, 1042, 1209, 1294, 1352, 1365, 1607, 1615, 1623, 1676, 1702, 1818, 1875, 2052, 1458–1459, 1507, 1518, 1529) quote the manuscript against itself — the manuscript is the claim under evaluation, not authority for canonical form. The canonical form is supplied by Arnold 1989 (Mathematical Methods, Ch. 5 §22–25), Goldstein-Poole-Safko 2002 Ch. 6, Vaswani 2017, Chentsov 1982 / Ay-Jost-Lê-Schwachhöfer 2017 / Bauer-Bruveris-Michor 2016, Cuturi 2013, Sønderby et al. 2016, and Ranganath-Tran-Blei 2016 — all real papers with primary-source URLs supplied in the opening. |
| CLAUDE.md cited as authority | n/a | 0 | No appeal to CLAUDE.md. |
| "Our framework establishes" / "by construction in this work" | n/a | 0 | Red's whole posture is attacking the framework; no by-construction circularity. |
| Fabricated `[Author Year]` | n/a | 0 | All non-bibliography citations (Arnold 1989, Goldstein-Poole-Safko 2002, Cuturi 2013, Ay-Jost-Lê-Schwachhöfer 2017, Bauer-Bruveris-Michor 2016, Sønderby 2016, Ranganath-Tran-Blei 2016) are real papers with primary-source URLs in the text. The bibliography's "Coverage gaps — extend on demand" clause at lines 74–87 authorizes this extension pattern; harvest is logged via the coordinator's evidence files per methodology line 180. |
| Wrong-domain citation | n/a | 0 | Arnold Ch. 5 §22–25 is the small-oscillations / generalized-eigenvalue chapter — correct domain for the inertia-tensor vs potential-Hessian independence argument. Vaswani 2017 for asymmetric softmax-attention — correct domain. Cuturi 2013 Sinkhorn — correct domain for softmax-from-optimal-transport. Cencov for Fisher uniqueness — correct domain. Sønderby for ladder VAE — correct domain. Ranganath for hierarchical variational models — correct domain. |
| Reasoning-by-construction circularity | n/a | 0 | None — red's argument runs from external canon to falsification of manuscript construction, not the reverse. |
| Hand-wave-with-citation | n/a | 0 | Every external citation is load-bearing for the specific claim it supports (e.g., Arnold's generalized-eigenvalue requirement directly grounds the tautology charge; Sinkhorn/OT directly grounds the multi-derivation non-uniqueness charge). |
| Banned phrases (Claude-isms) | n/a | 0 | No instance of `key insight`, `crucially`, `critically`, `notably`, `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores`. |
| Banned phrases (debate hedges) | n/a | 0 | No instance of `perhaps`, `it could be argued`, `one might suggest`, `both sides have a point`. |

## Action

RECORD. Debate continues. Soft cap not approached.
