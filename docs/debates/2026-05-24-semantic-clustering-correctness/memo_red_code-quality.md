# Memo — code-quality (red, opening)

## Position

The vocab figure and the label-annotation logic produce artifacts that are inaccurate as scientific
communication, and "completeness" understates the defect. A figure titled "Token-mean (μ) clustering"
/ "vocabulary semantics" whose visible labels are 130/200 placeholder/empty glyphs and whose 30
annotated points are chosen by array order is not an incomplete-but-correct figure; it is a figure
that misrepresents what it shows.

## Argument

**The annotation selection is array-order, not salience (plotting.py:224-235).** The loop annotates
`range(min(len(token_strings), coords.shape[0], 30))` — the **first 30 rows of the array**, with a
fixed `xytext=(2,2)` offset. For the vocab view those first 30 rows (after the sorted subsample,
pipeline.py:62 `np.sort`) are the lowest token ids — the byte-fallback punctuation/digit fragments
(`! " # $ % ( * + , - 0 1 …`). So the 30 labels a reader actually sees on the "vocabulary semantics"
figure are the least semantically meaningful tokens in the entire vocabulary, selected by nothing
more than their integer id. No cluster-representative, no centroid-nearest, no salience criterion is
applied. A reader cannot distinguish "these are the representative tokens of the clusters" from
"these are the first 30 array indices." The figure invites the first reading and delivers the second.

**The vocab labels are dominated by non-renderable / empty fallbacks.** `_sanitize_label`
(pipeline.py:88-93) strips only C0 + DEL; it does not strip U+FFFD or whitespace. So a byte-fallback
id that decodes to '�' is annotated as '�', and an id that decodes to '' is annotated as nothing.
The verifier reproduced 107 U+FFFD, 23 empty, 72/200 unique. A publication-quality figure (the
module's stated standard, plotting.py docstring; CLAUDE.md "Figures should be publication quality by
default" — cited as project intent, not as correctness authority) that is two-thirds placeholder
glyphs does not meet a scientific-accuracy bar. The defense classifies this as "the vocab view is
near-meaningless" and files it under completeness; a near-meaningless figure that is nonetheless
shipped and titled as a semantics result is an accuracy defect.

**The contextual duplicates are undisclosed (pipeline.py:193-222; extract.py:98).** Per-occurrence
flattening `(B,N)→(B·N)` is a legitimate design — one row per occurrence is defensible. But the four
plot calls pass the same `strings` list (pipeline.py:193/202/210/222) and the figure carries no
indication that `the`, `of`, `Atlantic` appearing three times are three occurrences rather than a
duplication bug. The single most likely reading by a viewer — including the user, who flagged it — is
"why is this token plotted three times?" A design whose correct interpretation is never communicated
and whose default interpretation is "bug" is not "correct by design"; the design has a
communication-correctness hole. Correctness of a visualization includes that the artifact means what
it appears to mean.

## What would falsify my position

If the figures disclosed per-occurrence semantics (e.g., a caption or a `#k` suffix on repeated
labels) and annotated salient/cluster-representative tokens rather than array-order ids, these would
be cosmetic completeness items. As shipped, neither holds: array-order annotation
(plotting.py:227 `range(...30)`), no occurrence disclosure, no fallback stripping.

## Newly-discovered canon

- No external canon required; this is a software-and-scientific-communication-quality memo grounded in
  `path:line` (plotting.py:224-235 array-order annotation, pipeline.py:88-93 fallback non-stripping)
  plus the verifier's reproduced label statistics (107 U+FFFD, 23 empty, 72/200 unique). The accuracy
  standard invoked is the project's own "publication quality by default" (cited as stated intent, not
  as a correctness oracle).
