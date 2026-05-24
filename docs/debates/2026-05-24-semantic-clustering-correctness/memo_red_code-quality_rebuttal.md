# memo_red_code-quality (rebuttal) — semantic-clustering-correctness

## Target: blue's "completeness/UX, not correctness" reclassification (Vector 4, `02_blue_opening.md:33`)

Blue's move is to grant every defect red names and then file it under "completeness/UX/latent." For most software that triage is reasonable. For a *scientific-visualization* deliverable it is not, because the deliverable's contract is exactly that the rendered figure communicates the truth of the underlying data. The unit of correctness for a figure is not "the arithmetic on each row is right" — blue's framing at `02_blue_opening.md:33` ("the geometry computed on those rows is still correct, it is the selection of rows that is poor"). The unit of correctness is "a competent reader reaches a true conclusion from the picture." By that contract:

- A vocab figure where 130/200 points are unrenderable (107 U+FFFD, 23 empty) and 72/200 unique, annotated on its 30 least-meaningful points (`plotting.py:224-235`, first-30 array-order = lowest ids = ASCII fragments), communicates a false impression of vocabulary semantic structure. The arithmetic being correct per row does not make the figure correct; a correct computation rendered into a misleading graphic is a defective deliverable.
- A contextual figure with undisclosed per-occurrence duplicate labels is one a reader misreads as a labeling bug. "Internally consistent design" and "correct figure" are different predicates.

The `_sanitize_label` design (`pipeline.py:88-93`) strips C0+DEL but passes U+FFFD and empty strings straight to the plot — a sanitizer that does not sanitize the dominant failure mode of the actual tokenizer in use. That is a design smell, not a missing feature: the function exists precisely to clean labels and does not clean the labels that need cleaning under cl100k_base.

## The standard

The criterion that a graphic must not induce conclusions the data do not support is the standard for graphical integrity [Tufte 1983, *The Visual Display of Quantitative Information*]. Reproducible-research practice holds that a figure shipped as a result is part of the result and is judged as the result, not as decoration [Wilson et al. 2017, PLOS Comput Biol, "Good enough practices in scientific computing," on treating analysis outputs as products]. Under either standard, "the math on the rows is right, the figure just selects bad rows" does not earn the label "correct" for the figure.

## Concession

I concede the four items are not bugs in the geometry math or the index alignment, and that under the active config the block-restriction result is numerically exact. The disagreement is purely about whether "correct" is an honest label for the *deliverable* given those four present, reachable defects. It is not.

## Newly-discovered canon

- [Wilson et al. 2017, PLOS Computational Biology 13(6):e1005510, "Good enough practices in scientific computing"] — analysis outputs (figures, tables) are products subject to the same correctness scrutiny as code.
- [Tufte 1983] — graphical integrity (already invoked by philosophy-of-science memo).
