# Canon-cop report — pifb-pullback-mechanism — Phase 2 openings — both sides

## Summary

| Side | Grep strikes | LLM strikes | Total | Action |
|------|-------------|-------------|-------|--------|
| Red  | 0 | 0 | **0** | RECORD |
| Blue | 0 | 0 | **0** | RECORD |

Neither side reaches the ≥3 soft-cap threshold. No mandatory rewrite. The debate proceeds to Phase 3 rebuttals.

## Contested canonical fact — verified against a primary source

The binding question both openings turn on: in the standard fiber-bundle metric (Kaluza–Klein / Sasaki), does the connection live in the **vertical/fiber** block or the **horizontal** block?

**Verified answer (primary source, WebFetch of the Kaluza–Klein line element):** the connection enters the **vertical/fiber** block as a covariantizing shift; the **horizontal** block is the pulled-back base metric. The canonical 5D line element is

```
ds² = g_μν dx^μ dx^ν + φ²(A_ν dx^ν + dx^5)²
```

with the base metric `g_μν` in the horizontal slot and the gauge potential `A_μ` appearing inside the fiber shift `(A_ν dx^ν + dx^5)`. The full block form `g̃ = [g_μν + φ²A_μA_ν | φ²A_μ ; φ²A_ν | φ²]` confirms `A` never stands alone as a horizontal block. The Sasaki metric on `TM` follows the same pattern: the H/V split comes from the (Levi-Civita) connection, and both blocks are built from the pulled-back base metric, not from a standalone quadratic in the connection.

**Both openings state this fact correctly and identically.** Red: "the horizontal block is the pullback of a base metric `g_ab(x)`, and the connection one-form enters only the vertical/fiber block" (02_red_opening.md:13). Blue: "the connection one-form `ω` enters the vertical/fiber block ... and the horizontal block is the pulled-back base metric `π* g_C`" (02_blue_opening.md:7), and Blue concedes the textbook-match outright (02_blue_opening.md:11, :26). Canon-cop records that **the canon supports both sides' shared statement of the standard form**; the disagreement between Red and Blue is over the *consequence* (Red: the substitution is a fatal defect; Blue: it is an admissible labeled-novel substitution forced by the "it from bit" setting), which is a matter for the judges, not for canon-cop.

The manuscript's construction — `κ(A_μ,A_ν)` in the horizontal block, Fisher-Rao in the vertical — **inverts** the canonical form. This is a true statement of fact, recorded here as canon-verified.

## Grep pass (canon_cop_validator.py)

### Red — `02_red_opening.md`

```json
{
  "target": "docs\\debates\\2026-05-24-pifb-pullback-mechanism\\02_red_opening.md",
  "total_strikes": 0,
  "action": "RECORD",
  "manuscript_authority_hits": [],
  "citation_checks": [],
  "attention_citation_count": 0,
  "claude_md_citation_count": 0,
  "external_citation_count": 0
}
```

### Blue — `02_blue_opening.md`

```json
{
  "target": "docs\\debates\\2026-05-24-pifb-pullback-mechanism\\02_blue_opening.md",
  "total_strikes": 0,
  "action": "RECORD",
  "manuscript_authority_hits": [],
  "citation_checks": [],
  "attention_citation_count": 0,
  "claude_md_citation_count": 0,
  "external_citation_count": 0
}
```

Note on grep coverage: the validator scans for the literal `Attention/*.tex` / `CLAUDE.md` / `user_theory_summary.md` patterns. Both openings cite the manuscript by bare `.tex` line number (`:2726`, `:2768`, etc.) rather than the file path, so the grep pass sees nothing. The manuscript-as-authority question therefore falls entirely to the LLM pass below, which read each `:NNNN` reference in context.

## LLM pass — subtle patterns

### Red — `02_red_opening.md`

| Pattern checked | Lines | Strikes | Note |
|-----------------|-------|---------|------|
| Manuscript line refs as authority | :5, :9, :13, :15, :17, :19, :21 | 0 | Every `:NNNN` identifies *what the manuscript claims* ("the manuscript names it" :13; "is admitted gauge-noninvariant" :15; "settled by stipulation" :21), never as the standard. Canonical authority is routed to Nakahara Ch. 10–11, Kobayashi–Nomizu Vol. I §II–III, KK/Sasaki. Correct usage. |
| Reasoning-by-construction circularity | :13, :21 | 0 | The circularity ("canonical horizontal block requires a base metric as input; the manuscript wants it as output") is alleged *against the manuscript* as a finding — it is not Red reasoning circularly. Not a self-justification strike. |
| Fabricated `[Author Year §X]` | :13 ("Sasaki metric, Albuquerque 2018"), :19 (Da Costa et al. 2020) | 0 | The load-bearing canonical cite for the horizontal/vertical fact is the KK line element (verified) plus Nakahara / Kobayashi–Nomizu (textbook, source-level, in bibliography). "Albuquerque 2018" and the HandWiki/Wikipedia cites are corroborating, non-load-bearing pointers; the bibliography permits wiki for orientation and the textbook carries the claim. Not strikable. |
| Wrong-domain citation | :19 (FristonEtAl2017, ParrPezzuloFriston2022) | 0 | Used to argue active inference does *not* localize "perceived space" in the parameter tier — applied to the correct domain (active-inference hierarchy). In-bibliography. Defensible. |
| Hand-wave-with-citation | :19 (Popper 1959) | 0 | Popper-falsifiability invoked for a falsifiability claim — on-point, not window dressing. |

### Blue — `02_blue_opening.md`

| Pattern checked | Lines | Strikes | Note |
|-----------------|-------|---------|------|
| Manuscript line refs as authority | :7, :13, :15, :17, :21, :22, :23 | 0 | Every `:NNNN` is claim-identification ("the manuscript proposes it, 'We propose...'" :17; "labels it as such: 'tw' not 'YM'" :15; "the non-invariance computed explicitly" :22). Canonical authority routed to Cencov, Amari–Nagaoka, Nakahara §10.4, Kobayashi–Nomizu. Correct usage. |
| Implicit "our framework establishes" | :15 (it-from-bit "precludes a pre-existing base metric") | 0 | Uses the manuscript's *premise* to justify the *novelty* of the substitution under external_canon_math.md §4 (labeled-and-justified novel construction is admissible). This is defending a novel-label, not asserting canonical status from the manuscript. Not a strike. |
| Reasoning-by-construction circularity | :24, :33 | 0 | Blue explicitly disclaims circularity (:24: "The manuscript is cited only to identify what is claimed; canon establishes that it holds") and correctly labels the cross-term vanishing at :33 as *definitional, to be read as a definition* rather than a theorem. This is the anti-circular reading, not a circular one. |
| Fabricated `[Author Year §X]` | :23 (FristonEtAl2017 *Neural Computation* 29:1–49), :23/:21 (Da Costa 2020, AmariNagaoka2000, Nielsen2020) | 0 | FristonEtAl2017 and AmariNagaoka2000 and Cencov1972 and Nielsen2020 are all in `external_bibliography.md` with matching venue/volume. Da Costa et al. 2020 is not in the bibliography but is a real active-inference paper used in-domain; per bibliography policy on coverage gaps this is a good-faith on-domain cite, not a fabrication. The Fisher/Cencov load-bearing cites verify. |
| Wrong-domain citation | :21 (Cencov for vertical block), :22 (Nakahara §10.4 for connection transformation law) | 0 | Cencov→Fisher-uniqueness and Nakahara→connection-transformation-law are both right-paper-right-claim. Verified against external_canon_math.md §1 and §2. |

## Even-handedness check

Both sides use manuscript `:NNNN` references in exactly the same way (claim-identification, not authority). The rule requires that if one side is struck for this, the other must be struck identically — neither is struck, symmetrically. Both route canonical authority to the same external textbooks (Nakahara, Kobayashi–Nomizu) and the same information-geometry canon (Cencov, Amari). Both correctly state the KK/Sasaki canonical block structure, which the primary source confirms. The openings are clean under the source-of-truth precedence rule.

## Verdict

- **Red: 0 strikes — RECORD.**
- **Blue: 0 strikes — RECORD.**

No mandatory rewrite on either side. The contested canonical fact (connection lives in the vertical block; horizontal block is the pulled-back base metric) is verified and recorded as canon-supported; both openings state it correctly. The substantive disagreement over whether the manuscript's inversion is a fatal defect (Red) or an admissible labeled-novel substitution (Blue) is preserved for the Phase 3 rebuttals and the judges.
