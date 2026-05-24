# Red Panel Choice — pifb-signature-problem (Phase 2, opening)

No `Agent`/`Task` tool is available in this environment, so the coordinator embodies the five lenses directly. Each lens is logged below as a standalone memo file (`memo_red_<tag>.md`); this file records the selection and the one-sentence justification per the binding rule in `debate_methodology.md` ("Coordinator–consultant protocol", rule 3).

| Tag | Justification (why this lens attacks this claim) |
|---|---|
| `philosophy-of-science` (mandatory) | Frame-checks whether "structurally compatible with Lorentzian signature" is a content-bearing claim or an unfalsifiable one once every input knob is a free postulate; catches manuscript-as-authority circularity. |
| `gauge-theorist` | The signature is read off the Lie-algebra trace form $\mathrm{tr}(A_\mu A_\nu)$; the $+\mathrm{tr}$ vs $-\mathrm{tr}$ convention and the compact-vs-non-compact generator choice are the actual sign drivers — this is the Killing-form-signature question on real forms of $\mathfrak{gl}(K,\mathbb{C})$. |
| `geometer` | The genuine object $G_{\mu\nu}$ is complex and degenerate (rank-1); the "metric" appears only after a rank-changing real-part projection. Whether a projected non-metric is a metric the framework produces is a differential-geometry question. |
| `info-geometer` | The claim confines complexity to the connection while the Fisher fiber stays real positive-definite; whether the resulting base signature is connected to the statistical (information) content or free-floating is exactly the Fisher-metric / dual-structure question. |
| `numerical-analyst` | Rank-1 $\to$ rank-2 via $\mathrm{Re}(\cdot)$ is a discontinuous (conditioning-pathological) operation; the conformal-class ambiguity is a scale indeterminacy; the parabolic-vs-hyperbolic continuum-limit tension is a stability-of-the-continuum-limit question. |

The dispatch prompt's strongly-indicated panel (philosophy-of-science mandatory + gauge-theorist, geometer, info-geometer, numerical-analyst) is adopted unchanged: each of the four attack surfaces in the dispatch maps cleanly onto one of these lenses, and the variational / transformer-ml lenses add little here because no E-step or attention form is under evaluation (the causal-cone first-order-flow point is covered by numerical-analyst's continuum-limit competence).
