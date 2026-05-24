# Memo — philosophy-of-science (red, opening)

## Position

The claim's central move — partitioning the module's defects into "geometry/index correct" versus
"presentation completeness-only" — is a framing error. For a scientific visualization, the artifact's
*legibility* (does the figure mean what a competent viewer takes it to mean?) is part of its
correctness, not a separate axis below it. "Correct-by-design" is incoherent when the design is
undisclosed and the predictable reading is wrong. The claim quarantines the figure's communicative
failures into a non-correctness bucket precisely so the headline "the module is correct" survives;
that is the equivocation to flag.

## Argument

The claim equivocates on "correct." It uses a narrow, internalist sense (the distance formulas
compute what their docstrings say; `strings[i] ↔ coords[i] ↔ token_ids[i]` has no off-by-one) to
license a broad, externalist conclusion (the module is right, defects are merely cosmetic). These are
different predicates. A figure can be internally faithful to its inputs and still be a wrong
scientific artifact if it systematically licenses a false inference in the reader. The vocab figure
is exactly this: every line of code is faithful to its inputs, and the output is a "vocabulary
semantics" scatter that is two-thirds placeholder glyphs with array-order-selected labels. Internal
faithfulness does not transfer to scientific correctness; asserting that it does is the
theory-ladenness trap — the designer who knows the per-occurrence convention reads the figure
correctly, but the figure is not for the designer.

"Correct-by-design" carries an implicit success condition: a design is correct if it does what it is
*for*. The module is for producing publication-quality figures that communicate belief geometry
(pipeline.py docstring; the run lives under a semantic-clustering output dir intended for a
manuscript). Judged against that purpose, a figure that (a) predictably reads as a duplication bug
and (b) displays mostly unrenderable labels fails its purpose. To call it "correct by design" one
must redefine the design's purpose as "compute and place points faithfully regardless of whether the
reader can interpret them" — a purpose no one would state in advance. That redefinition-after-the-fact
to preserve the correctness verdict is unfalsifiable-by-construction: any miscommunication can be
relabeled "completeness," so "correct" can never be lost.

Two of the three "completeness" items are demonstrable inaccuracy, not incompleteness. Incompleteness
is a missing feature (e.g., "no 3D view"). Displaying 130/200 placeholder labels on a figure titled as
a semantics result, and annotating the least-meaningful 30 tokens by array order, are *present
inaccuracies* in a shipped artifact. The claim's taxonomy ("never disclosed," "near-meaningless," "array
order rather than salience") describes accuracy defects in completeness language to keep them off the
correctness ledger.

The genuinely defensible parts of the claim — the per-head quadrature geodesic equals the full-matrix
geodesic under the block-diagonal bank (verified ~4.4e-16), and the index alignment is sound — should
be conceded plainly. The red case does not need them. What it needs is the recognition that
"correct-by-design" is being used to immunize the figure against its own illegibility.

## What would falsify my position

If "correctness" were explicitly scoped in the claim to "numerical faithfulness of the distance
matrices and index alignment" — and the figures' legibility were adjudicated on a separate, equally
weighted accuracy axis the user actually asked for ("correctness, accuracy, AND completeness", claim
§User context) — then the partition would be coherent. The claim does not scope it that way: it folds
accuracy failures into "completeness" and concludes "none of which is a correctness bug," which is the
equivocation.

## Newly-discovered canon

- No external citation; this is a frame-analysis of the claim's use of "correct," "completeness," and
  "by design." Grounded in the claim text (§Load-bearing sub-propositions 3, §User context: the user
  explicitly asked for correctness AND accuracy AND completeness) and the code artifacts the other
  memos cite. The standing methodological point — internal faithfulness to inputs does not entail
  correctness of the scientific claim the artifact makes — is the falsifiability/scope discipline this
  lens enforces.
