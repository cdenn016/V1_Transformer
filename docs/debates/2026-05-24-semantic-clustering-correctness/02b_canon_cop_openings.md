# Canon-cop — Phase 2 openings

| Side | Strikes | Action |
|------|---------|--------|
| Red (`02_red_opening.md`) | **2** | RECORD — no rewrite |
| Blue (`02_blue_opening.md`) | **0** | RECORD — no rewrite |

Soft cap for mandatory rewrite is ≥3. Neither side reaches it; debate proceeds to Phase 3 with both openings intact.

## Red — 2 strikes (LLM pass; grep pass clean)

**`02_red_opening.md:25-29` — wrong-domain citation (2 strikes).** Red attaches Pennec, Fillard, Ayache 2006 ("A Riemannian Framework for Tensor Computing") to `omega_geodesic_distances` and characterizes it via invariance under the congruence action `P → A P Aᵀ`. That congruence action is the symmetry of the SPD/tensor cone Sym⁺(K) (Pennec's domain). The code (`geometry.py:349-352`, verified) computes `M = Ω_i⁻¹Ω_j` then `‖log M‖_F` — a distance on GL⁺(K) **group elements**, left-invariant under `Ω_k → A Ω_k`, with no SPD square-root symmetrization. Real paper, wrong manifold/symmetry group.

Content note for judges (not a separate strike): red:27-28 also calls the distance "affine-invariant / bi-invariant." There is no bi-invariant Riemannian metric on non-compact GL⁺(K); blue states this correctly at `02_blue_opening.md:25`. Adjudicated on the geometry axis.

## Blue — 0 strikes

All citations (Nakahara 2003, Hall 2015, Lee 2013, McInnes 2018, Rousseeuw 1987, Bishop 2006, Cencov 1972, plus `external_canon_math.md` = the canon itself) are real, in-domain, used correctly. The PCA-whitening concession (whitened φ is not a canonical/Čencov-invariant metric) is honest. No manuscript/CLAUDE.md/docstring cited as authority for what is correct.

## Both sides

In-repo `.py` paths are cited as evidence for *what the code does* (legitimate in code mode, non-strike). No reasoning-by-construction circularity.

**Judges: weight red's Pennec wrong-domain citation (`02_red_opening.md:25-29`) negatively per your rubrics.**
