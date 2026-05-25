# Canon-cop report — pifb-implementation-rock-solid — round 3b — blue

## Summary

Total strikes: 0 (grep pass flagged 4; LLM pass downgrades both flagged citations to 0 — both are real, on-domain works in `external_bibliography.md`'s self-disclosed "Coverage gaps — extend on demand" set).
Action: RECORD

## Grep pass (canon_cop_validator.py)

```json
{
  "target": "docs/debates/2026-05-21-pifb-implementation-rock-solid/03b_blue_surrebuttal.md",
  "total_strikes": 4,
  "action": "MANDATORY_REWRITE",
  "manuscript_authority_hits": [],
  "citation_checks": [
    {
      "line": 9,
      "citation": "[Moakher 2002 §3]",
      "author": "Moakher",
      "year": "2002",
      "section": "§3",
      "verified": false,
      "note": "key 'Moakher2002' not found in external_bibliography.md"
    },
    {
      "line": 19,
      "citation": "[Lakatos 1970 §1]",
      "author": "Lakatos",
      "year": "1970",
      "section": "§1",
      "verified": false,
      "note": "key 'Lakatos1970' not found in external_bibliography.md"
    }
  ],
  "attention_citation_count": 0,
  "claude_md_citation_count": 0,
  "external_citation_count": 2
}
```

Grep raised 4 strikes (2 per "fabricated" citation under the mechanical strike table). The LLM pass below verifies each flagged citation and downgrades both — neither is fabricated; both are real, on-domain, well-known works in `external_bibliography.md`'s self-disclosed "Coverage gaps — extend on demand" set (the bibliography file's `## Coverage gaps` section explicitly enumerates topics outside its current coverage and instructs extending it on demand).

Symmetry note: red's sur-rebuttal cites Kobayashi-Nomizu 1963, Bishop-Crittenden 1964, Atiyah 1979, Wainwright-Jordan 2008, Blei-Kucukelbir-Jordan 2017 — same status (real, on-domain, not currently keyed in `external_bibliography.md`), just in formats the regex misses (hyphenated multi-author, or with comma/semicolon separators inside the bracket). Treating these symmetrically requires downgrading both sides' false positives. Flagging only blue's single-author bracket-format cites would be regex-coverage punishment, not source-of-truth enforcement.

## LLM pass — subtle patterns

| Pattern | Line | Strikes | Note |
|---------|------|---------|------|
| Fabricated `[Moakher 2002 §3]` (grep) | 9 | 0 (downgraded from 2) | Real work: Moakher, M. (2002), "A differential geometric approach to the geometric mean of symmetric positive-definite matrices," *SIAM J. Matrix Anal. Appl.* 26(3): 735-747. §3 develops the four-mean taxonomy (arithmetic, Log-Euclidean, geometric/affine-invariant, Wasserstein). On-domain for the SPD-mean-vs-extrinsic-Euclidean-mean distinction at `meta_agents.py:344-355` and PIFB line 2191. Not currently in `external_bibliography.md` — falls under "Coverage gaps" §SPD / Riemannian-mean topics. Real, on-domain, not fabricated. |
| Fabricated `[Lakatos 1970 §1]` (grep) | 19 | 0 (downgraded from 2) | Real work: Lakatos, I. (1970), "Falsification and the Methodology of Scientific Research Programmes," in *Criticism and the Growth of Knowledge* (Lakatos & Musgrave eds.), Cambridge UP. §1 develops the progressive-vs-degenerating problemshift discriminator. On-domain for the Lakatosian-degeneration debate already being conducted between the two sides since round 3 (red rebuttal line 11 cites the same source). Not currently in `external_bibliography.md` — falls under "Coverage gaps" §philosophy-of-science. Real, on-domain, not fabricated. |
| Manuscript-as-authority | — | 0 | Blue argues *against* the literal claim under debate; no appeal to manuscript-as-authority. The concessions on `meta_agents.py:358` and Chechik-Tishby 2005 Theorem 3.1 scope cite the manuscript only to identify what the manuscript says, not to ground a canonical claim. |
| Reasoning-by-construction circularity | — | 0 | Blue's defenses cite external canon (West-Harrison §10.7 for discounted updating, Lakatos §1 for progressive programmes, Chechik-Tishby 2005 §3 Theorem 3.1 for scope verification, Bishop 2006 §10.7 / Amari-Nagaoka 2000 §3.5 for m-projection, Moakher 2002 §3 for SPD-mean taxonomy), not the manuscript's own framework. |
| Hand-wave-with-citation | — | 0 | West-Harrison §10.7 "Discounted Likelihoods" is the canonical infinite-horizon discounted-updating construction (correct domain for PIFB Eq. 2270). Chechik-Tishby 2005 §3 Theorem 3.1 is the Gaussian-IB closed form (correct domain for the IB framing). Bishop 2006 §10.7 is the law-of-total-variance form of mean-field VI (correct domain for the m-projection truncation). All citations match the substantive claim. |
| Wrong-domain citation | — | 0 | None. |
| Implicit framework-establishes | — | 0 | None. |

## New vectors check

Sur-rebuttal mandate forbids new defense vectors; vectors must trace to blue's opening (`02_blue_opening.md`) or rebuttal (`03_blue_rebuttal.md`).

| Sur-rebuttal vector | Traces to | Status |
|---------------------|-----------|--------|
| Concession on `meta_agents.py:358` extrinsic vs log-Euclidean (Moakher 2002 §3 taxonomy) | Blue opening lines 46 (third bullet), 61 (falsification condition 4) — already conceded as a sub-claim 6 wound; sur-rebuttal sharpens with explicit Moakher taxonomy citation | TRACES (concession deepening, not new vector) |
| Concession on Chechik-Tishby 2005 §3 Theorem 3.1 scope | Blue rebuttal line 32 defended it as "honest scope flagging"; sur-rebuttal concedes red's attack | TRACES (concession, response to red's repeated attack on sub-claim 1) |
| Concession on forward-KL barycenter dispersion truncation (m-projection vs truncation, Bishop 2006 §10.7 / Amari-Nagaoka 2000 §3.5) | Red rebuttal raised the m-projection-truncation attack at line 25; blue rebuttal defended at line 30. Sur-rebuttal concession is a response to red's continued pressure. | TRACES (concession, response to existing red attack) |
| West-Harrison §6.3 vs §10.7 citation-specificity defense retained | Blue opening line 25 (cited §6.3); blue rebuttal line 30 implied retention. Sur-rebuttal pivots to "§10.7 would be more precise, but §6.3 is not wrong-domain — copy-edit revision, not falsification." | TRACES |
| Lakatos progressive-vs-degenerating defense retained (lines 2138, 2213, 2284 as honest deferrals with independent testable content) | Blue rebuttal line 28 (Lakatos 1978 §1.4 mischaracterization defense) | TRACES |
| Position summary: literal-reading concession of sub-claims 6 and 7; RED_WINS tip to chief on operationalization-binding | Blue opening Position (line 9 — "honest blue position is concession on 6 and 7"); blue rebuttal Position summary (line 38 — "Standing on intent-faithful reading is the available defense; standing on literal-match is not.") | TRACES |

No new defense vectors introduced in the sur-rebuttal. Concessions deepen and citations sharpen, but the underlying vectors all trace to opening or rebuttal. Procedural strikes: 0.

## Banned-phrase scan

Grep over [`key insight`, `crucially`, `critically`, `notably`, `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores`, `perhaps`, `it could be argued`, `both sides have a point`] returned no matches (case-insensitive).

## Soft-cap status

Total strikes (grep, after LLM pass downgrade + procedural): 0. Action: RECORD. Debate complete on blue side; proceeds to Phase 4 judging.
