# Red panel choice — semantic-clustering-correctness (Phase 2 opening)

Mode is `code` with math/theory sub-axes on geometric soundness. The `code` mode
default (transformer-ml, implementation-engineer, code-quality, numerical-analyst,
philosophy-of-science) is pruned: there is no attention-form question (transformer-ml
dropped) and the logm/expm stability is already shown clean in the evidence pack
(numerical-analyst dropped). The geometric sub-axis pulls in geometer and
info-geometer. Final panel of 5:

1. **geometer** — The strongest math vector is that PCA-whitening on phi
   (geometry.py:261-266) makes the phi panel a *different metric object* than the
   Omega panel (un-whitened affine-invariant geodesic), so presenting them as two
   views of the same gauge object is misleading. Pennec/Arsigny canonical SPD/group
   metric and whitening-as-metric-rechoice are this lens.

2. **info-geometer** — Whether sample-wide PCA whitening of the phi coefficient space
   is a defensible Fisher-like preconditioning, or an arbitrary anisotropic rescaling
   that destroys the algebra-image correspondence the figure claims to show.

3. **implementation-engineer** — Trace the active K=20 / irrep_dims=[10,10] config from
   the entry point; establish whether the unguarded block-restriction (geometry.py:337-340)
   and the byte-fallback decode (pipeline.py:103) are reached, and whether the
   block-restriction is a latent vs active correctness defect.

4. **code-quality** — Whether shipping a vocab figure where 130/200 labels are U+FFFD/empty
   with first-30-array-order annotation, and undisclosed per-occurrence duplicates, is a
   scientific-communication / accuracy defect rather than mere "completeness."

5. **philosophy-of-science** (mandatory) — Frame-check "correct-by-design": whether the
   claim's partition of defects into "math correct, presentation completeness-only" is
   coherent when the design is undisclosed and predictably misread by a viewer.
