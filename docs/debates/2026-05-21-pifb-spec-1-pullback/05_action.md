# Action — pifb-spec-1-pullback

**From verdict:** RED_WINS

## Recommended action

Two line-edit-scale fixes are required before the section is publication-ready. The underlying mathematical content (Fisher-Rao fiber metric, group-level natural-gradient retraction, bundle-metric construction, gauge non-invariance disclosure, refusal of the `tr(FF)` escape) does not need structural revision.

### Edit 1 — BCH order at PIFB:2649

The chart-coordinate truncation error term is currently written as `O(||eta * ~∇_phi F||^2)`. The leading BCH correction `(1/2)[phi, eta * ~∇F]` is bilinear in phi and (eta * ~∇F); at fixed phi = O(1) (the obvious reading of `U_i = exp(phi_i)` at line 2647), this term is **first-order** in (eta * ~∇F), not second-order. Sympy verification: residual decreases linearly with eta (10x per decade), not quadratically (would be 100x per decade).

Two acceptable repairs:
- Restate the error as `O(||eta * ~∇_phi F||)` with the `[phi, eta * ~∇F]` leading commutator named explicitly.
- Reframe the comparison as the additive-chart-step `phi(t+1) = phi(t) - eta * ~∇F` versus the exact group retraction `U(t+1) = U(t) exp(-eta * ~∇F)` — the BCH discrepancy is first-order in (eta * ~∇F) and zero in the abelian sector.

The current line 2651 statement "exact in the abelian sector" survives both repairs (in the abelian sector, all commutators vanish, so the discrepancy is identically zero).

### Edit 2 — Hoffman 2019 citation at PIFB:2751

The sentence "The Interface-Theory reading~\cite{Hoffman2019} sharpens this further: perceived space is an evolved interface optimized for fitness rather than a faithful depiction of an agent-independent geometry, and the carrier of that interface is precisely the slow-timescale generative model rather than the moment-to-moment posterior" attributes a slow-parameter-vs-fast-state architectural identification to Hoffman's Interface Theory of Perception. Hoffman's actual argument is an evolutionary fitness-beats-truth claim at the species-genome level; the slow-vs-fast architectural distinction is from active-inference literature (Friston 2017 hierarchical inference), not from Hoffman.

Suggested rewrite: keep the Hoffman citation for "perceived space is an evolved interface optimized for fitness rather than a faithful depiction of an agent-independent geometry"; replace the second clause with "Within the framework's hierarchical structure, the carrier of this interface is identified with the slow-timescale generative model rather than the moment-to-moment posterior (an internal framework move; see Friston 2017 for the canonical slow-parameter vs fast-state separation in hierarchical active inference)."

## Follow-up debates (if any)

None. After applying Edits 1-2 the claim under adjudication becomes "publication-ready after the two line-edits" — a different claim from the one the verdict rejected, but one the manuscript can durably satisfy.
