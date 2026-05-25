# Canon-cop report — pifb-implementation-rock-solid — Phase 2 (opening) — blue

## Summary

Total strikes: 2
Action: RECORD (debate continues; 2 < 3 threshold for mandatory rewrite)

Grep pass: 2 strikes (one citation key not in `external_bibliography.md`). LLM pass: 0 strikes.

## Grep pass (canon_cop_validator.py)

```json
{
  "target": "02_blue_opening.md",
  "total_strikes": 2,
  "action": "RECORD",
  "manuscript_authority_hits": [],
  "citation_checks": [
    {
      "line": 36,
      "citation": "[Hall 2015 §5.3]",
      "author": "Hall",
      "year": "2015",
      "section": "§5.3",
      "verified": false,
      "note": "key 'Hall2015' not found in external_bibliography.md"
    }
  ],
  "attention_citation_count": 0,
  "claude_md_citation_count": 0,
  "external_citation_count": 1
}
```

## LLM pass — subtle patterns

| Pattern | Line | Strikes | Note |
|---------|------|---------|------|
| Attention/*.tex as authority for canonical form | — | 0 | All PIFB line citations (2123-2125, 2129, 2138, 2141-2152, 2160, 2172, 2174, 2179, 2191, 2197, 2213, 2219, 2222, 2275, 2284) cite the manuscript as the claim under evaluation, not as authority for canonical forms. Strike-free. |
| CLAUDE.md as authority | — | 0 | No CLAUDE.md cite. |
| user_theory_summary.md as authority | — | 0 | No user_theory_summary.md cite. |
| Implicit "our framework establishes" / "by construction in this work" | — | 0 | Blue defends sub-claims by external canonical reference; no "the work proves itself" pattern. |
| Fabricated `[Author Year §X]` (LLM verification of grep hit) | 36 | (already counted in grep) | The grep flagged "Hall 2015 §5.3" as not in the bibliography. LLM verification: Hall, B. (2015) *Lie Groups, Lie Algebras, and Representations* (2nd ed., Springer GTM 222) is a real textbook; §5.3 is "The Baker-Campbell-Hausdorff Formula and Its Consequences." Blue cites it to back the first-order BCH approximation $\phi_I = \sum w_i \phi_i$ with $\mathcal{O}(\|\phi_i\|^2)$ error — which is precisely what Hall §5.3 establishes. The citation is substantively correct; the strike reflects a coverage gap in `external_bibliography.md` (its own footer says "extend on demand"), not a fabricated source. The 2 grep strikes stand under the mechanical rule but the LLM pass adds zero additional strikes. |
| Wrong-domain citation | — | 0 | Spot-checks: Friston2010 Eq. 2.2 for the FE form — correct domain (line 15). BleiKuckelbirgJordan2017 §3 for variational mean-field — correct domain (line 15). Bishop2006 §10.7 for the moment-matched mean / law-of-total-variance Gaussian barycenter — §10.7 is "Exponential Family Distributions" but the variational-Gaussian moment-matching is in §10.2 ("Illustration: Variational Mixture of Gaussians") and §10.7 covers exponential-family conjugate variational families. Edge case: the §-number is slightly off-target for the precise law-of-total-variance step (better aligned with §10.1.2 or §10.2). This is borderline; the substantive claim is correct and the textbook is correct, so under the rule "right paper, wrong claim" (wrong-domain, 2 strikes) does not fire — the paper is right and the claim is in the same domain. Recording as a soft observation, not a strike. Hinton 2002 *Neural Computation* §2 for PoE — correct (line 23). Genest-Zidek 1986 *Statistical Science* §3.2 for log-linear pool — correct (line 24). West-Harrison 1997 *Bayesian Forecasting* §6.3 for dynamic discount — correct (line 25). Bissiri-Holmes-Walker 2016 *JRSSB* §2 for tempered Bayes — correct domain; whether the *scale-distance* mapping is faithful is the substantive debate question, not a wrong-domain canon-cop strike. Tishby-Pereira-Bialek 1999 §2 / Chechik-Tishby 2005 §3.1 for IB — correct (line 28). Karcher 1977 / Nakahara §5.5 / Helgason 1978 §III.6 for Karcher mean on Lie groups — correct (line 36). Pennec 2009 / Bonnabel-Sepulchre 2009 for SPD manifold means / left-invariant metrics — correct (line 36). Moakher 2002 for SO(3) Karcher mean uniqueness — correct (line 36). Cover-Thomas §2.3 for mutual-information disclaimer — correct (line 38). |
| Reasoning-by-construction circularity | — | 0 | Blue argues from external canon to "the manuscript correctly invokes X" — direction is canon → manuscript, not manuscript → manuscript. |
| Hand-wave-with-citation | — | 0 | Each external citation has a concrete connection to the specific claim it backs. The closest borderline is Bishop2006 §10.7 (see above) and BHW 2016 §2 for the manuscript's scale-distance mapping — but in both cases the paper is in the correct domain and the substantive question is what red attacks under Vector 3, not a canon-cop strike. |

## Banned-phrase scan

No banned-phrase hits. Blue uses "explicitly" (substantive adjective, not the Claude-ism "explicitly stated"), "exactly" (substantive), "correctly" (judgment, substantive). No `key insight`, no `crucially`, no `notably`, no `importantly`, no `it's worth noting`, no `interestingly`, no `fundamentally`, no `in particular`, no `leverages`, no `underscores`, no `perhaps`, no `it could be argued`, no `one might suggest`, no `both sides have a point`. Clean.

## Soft-cap status

2 strikes < 3 → RECORD, no rewrite. Debate continues.

## Notes for judges

1. The 2 grep strikes against blue come from a single citation ("Hall 2015 §5.3") that is substantively correct — Hall's textbook is a real source and §5.3 is the BCH chapter blue uses. The strikes reflect a coverage gap in `external_bibliography.md` (which itself documents that gap in its footer), not a fabricated citation or wrong-domain misapplication. The canon-strict judge's −2-per-fabricated-cite weighting should treat this as a recoverable bibliographic-completeness issue: blue could close the wound by appending "Hall, B. (2015) *Lie Groups, Lie Algebras, and Representations*, 2nd ed., Springer GTM 222" to the bibliography file or by switching the cite to one of the in-bibliography Lie-group references that also covers BCH (none of the current bibliography entries does — Nakahara2003 has the matrix exponential at §5.5 but BCH proper is not in the catalogued chapter list).

2. Blue's concession posture on sub-claims 6 and 7 is honest by the methodology's standards (the rebuttal-round rules in `debate_methodology.md` say "the judge will weight a concession highly, and the user is better served by an honest debate"). The canon-cop does not weight in on the strategic merit; this report only records that the concession is not itself a source-of-truth violation.

3. Blue's defense citations on sub-claims 1-5 are unusually well-covered (Friston2010, Beal2003, BleiKuckelbirgJordan2017, Bishop2006, JordanGhahramaniJaakkolaSaul1999, Hinton 2002, Genest-Zidek 1986, West-Harrison 1997, BHW 2016, Tishby-Pereira-Bialek 1999, Chechik-Tishby 2005, Karcher 1977, Nakahara2003, Helgason 1978, Pennec 2009, Bonnabel-Sepulchre 2009, Moakher 2002, Hall 2015, Cover-Thomas 2006, Achille-Soatto 2018, Popper 1963). Source-of-truth grounding is dense; the canon-strict judge has substantial canonical surface to verify against.
