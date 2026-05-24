# memo_red_philosophy-of-science (rebuttal) — semantic-clustering-correctness

## Target: blue's frame defense (`02_blue_opening.md:35`, Vector 4) and the (c) admission (`02_blue_opening.md:51`)

Blue claims the label "correct-by-design" does not equivocate because it separates "correct" (geometry + alignment) from "complete/well-communicated" (the four conceded items). That separation is exactly the equivocation, not its absence. The claim defines its own scoring rule: every property of the artifact that is *present and accurate* counts toward "correctness," and every property that is *present and inaccurate* is reassigned to a "completeness" bucket and excused. Under that partition no observable defect can ever falsify "correct," because any inaccuracy a critic produces is, by the claim's own construction, relabeled a completeness gap. A predictive statement that cannot be contradicted by any observation is not a scientific claim about the artifact; it is a definition dressed as a finding [Popper 1959, *The Logic of Scientific Discovery*, §§4, 6, 19–21, the falsifiability criterion].

Blue's own opening hands over the ground. The last line of `02_blue_opening.md:51` states that condition (c) — whether a completeness item is actually a correctness/scientific-integrity defect — is "the live question on which the debate most plausibly turns," and concedes the judges must weigh it. So even blue does not treat (c) as settled by the numbers; it treats it as the open question. The 0.952 Spearman, the 4.4e-16 path agreement, and the exact block-diagonality settle sub-proposition 1's *geometry*, but they say nothing about whether a figure that miscommunicates by construction is "correct."

## The vocab figure is a correctness defect, not a completeness gap

The deliverable is a scientific figure titled around vocabulary semantics. Executed (evidence pack, investigator A + verifier): of the 200 plotted ids, 107 decode to U+FFFD (the Unicode replacement character), 23 to empty string, only 72 are unique, 130 are duplicates. A figure in which 65% of the points are unrenderable byte-fallbacks presented under a "vocabulary" framing does not have a *missing* feature; it has a *present, false* feature — it asserts a semantic layout over symbols that carry no recoverable type identity. Calling that "completeness" is the relabeling move above. The correct grain of evaluation for a visualization is whether a competent reader draws a true conclusion [Tufte 1983, *The Visual Display of Quantitative Information*, on graphical integrity — the representation must not state more than the data support]. Here the figure states more than the data support.

## The contextual duplicates: concede the data, press the disclosure

The per-occurrence flatten is structural (`extract.py:98-106`) — there is no data-corruption bug, and I concede that. But "correct-by-design" is asserted of the *figure*, and the figure omits the disclosure that rows are per-occurrence. A figure whose intended reading requires undisclosed semantics is one a reader predictably misreads. "The design is internally consistent" is not the same proposition as "the figure is correct"; the claim slides between them. That slide is the equivocation a scope judge should flag.

## Newly-discovered canon

- [Popper 1959, *The Logic of Scientific Discovery*, §§4, 6, 19–21] — demarcation by falsifiability; a claim immunized against every possible observation is non-empirical. (In `external_bibliography.md`? confirm key; concept is standard.)
- [Tufte 1983, *The Visual Display of Quantitative Information*, ch. on graphical integrity] — a graphic has integrity only if the visual conclusion matches the data; representing non-data (byte-fallback junk) as semantic structure violates it.
