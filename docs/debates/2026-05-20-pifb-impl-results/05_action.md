# Action — pifb-impl-results

Three micro-edits to §Implementation and §Results. No structural changes; no claim-level revisions.

## Edit 1 — Fig. 4 caption (line 2308) and Phase II paragraph (line 2349)

Replace the inferential "confirmed" framing with the dependency-aware "consistent with" framing, since the composite NE score is defined as the mean of the other three diagnostics (line 2244) and is not an independent observation.

- Line 2308: change "as confirmed by simultaneous increases in all non-equilibrium diagnostics" to "consistent with simultaneous increases in three independent diagnostics (energy variance, gradient variance, energy flux) and the composite score that aggregates them."
- Line 2349 (Fig. 6 caption final sentence): change "The simultaneous spike in all four indicators at step 150 distinguishes this from a numerical artifact" to "The simultaneous spike across the three independent diagnostics and their composite is consistent with collective reorganization rather than numerical artifact."
- Line 2354 (Phase II "Reorganization Fluctuations" paragraph): apply the same softening from "distinguishes" to "is consistent with."

## Edit 2 — Emergent Properties at Higher Scales (lines 2169-2173)

The subsubsection currently asserts per-scale covariance trends ("$\Sigma_I^{(s+1)}$ are typically smaller") and "emergent coordination patterns not reducible to constituent actions" without measurement support, in a section that otherwise polices its scope. Two options:

- **Option A (recommended):** Mark the subsubsection as theoretical expectation rather than reported observation. Insert at the start of paragraph 2171: "We expect, though do not directly measure in the single-seed run of Section~\ref{sec:results}, that meta-agent beliefs..."
- **Option B (stronger):** Delete the closing rhetorical sentence "This can be poetically interpreted as the universe coming to 'know thyself'" (final clause of 2171), which is the kind of inflated rhetoric the calibrated sections avoid. The "Banned patterns" list in CLAUDE.md does not include this exact phrase, but the spirit of the project style guide rules out poetic flourishes inside technical subsubsections.

Both A and B are minimal; Option B alone leaves the descriptive claims uncalibrated, so the recommended fix is A plus B.

## Edit 3 — $\alpha \approx 1.8$ provenance (line 2428)

Add the fitting window and $t_c$ treatment in one parenthetical sentence. Suggested insertion after the existing $\alpha \approx 1.8$ statement: "The fit uses [N] points in the window $t \in [t_1, t_2]$ with $t_c$ [fixed at 150 / fit as a free parameter]; details and reproduction code are in Section~\ref{sec:methods_metagent}." If the fitting details are already in the methods section, replace with a back-reference to that section. If they are not, either add them or delete the bare number — an unreproducible diagnostic that is also explicitly disclaimed as not-a-critical-exponent earns little and risks more than it delivers.

## Scope of action

These are wording- and provenance-level edits to a §Results section that is otherwise calibrated. None of the three changes touches the claim structure of the section; none requires re-running experiments; none affects the §Implementation theoretical scaffolding. Total edit footprint: under twenty lines.
