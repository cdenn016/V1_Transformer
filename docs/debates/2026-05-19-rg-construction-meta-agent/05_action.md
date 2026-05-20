# Action — rg-construction-meta-agent

**From verdict:** BLUE_WINS (integrated, with calibrating manuscript edits)

Per-sub-claim: A BLUE; B BLUE (with calibration); C BLUE (with calibration); D BLUE.

## Recommended action

Three concrete manuscript revisions follow from the calibrations both teams converged on. None invalidates the section; all sharpen its labeling.

1. **Theorem 4 wording (line 4508-4510).** Rename the theorem from "Gaussian closure with local-potential correction" to "Augmented-class closure with explicit log-determinant correction" to remove the suggestion that strict closure is the substantive content. Add a one-sentence remark after the proof at line 4523 stating that for Gaussian belief networks with parent-dependent precisions the strict-closure special case is empty, and that the substantive content of the theorem is the augmented-class form. Cite `[Cardy1996 §3.3]` and `[Wilson1974 §V]` for the canonical analogue in effective-action RG (fluctuation-determinant correction at the saddle).

2. **Proposition 1 paragraph heading (line 4541).** Change "Controlled approximate closure" to "Structural decomposition of the closure residual" to match what Proposition 1 actually delivers — a taxonomy of residual sources, not a quantitative bound with pinned constants. The body labeling at line 4554 (the explicit "stated as a proposition rather than a theorem because constants ... are not pinned down") is already correct; only the paragraph heading overpromises.

3. **Theorem 5 hypothesis (line 4561).** Add a one-paragraph remark before the theorem statement that the local Lipschitz bound is locally derivable under closure conditions (i)-(v) by first-order Taylor expansion of the parent free energy in the high-coherence regime, with $L_q, L_p$ bounded by operator norms of the parent Fisher information and $A_I$ identified with the constrained gap $\tau \cdot m_I$ from Eq.~\ref{eq:rg_constrained_gap}. Cite `[AmariNagaoka2000 §3.2]`. The conditional theorem form at line 4561 is then explicitly grounded rather than free-standing.

## Follow-up debates

No sub-claim is REMAND. None required.

The user may, at their discretion, schedule a follow-up `math`-mode debate on a tighter quantitative version of Proposition 1 — specifying the closure norm $\|\cdot\|_\mathcal{B}$, the regularity class of the parent ansatz, and Laplace-method-derived values for at least $C_1$ (dispersion) and $C_5$ (anharmonic). The honest labeling at line 4554 is sufficient to make the proposition mathematically respectable under the structural-decomposition reading, so this is genuinely optional rather than blocking.

## Open issues surfaced (no follow-up debate required)

- Theorem 4's strict-closure special case ($A_{\xi\xi}$ independent of $Y$) is empty for any Gaussian belief network with parent-dependent precisions. Blue conceded this. The augmented-class form is what carries the section's content; Edit 1 above relabels the theorem to reflect this.
- Theorem 5's Lipschitz constants $L_q, L_p, A_I$ are introduced as free parameters in the theorem statement. Edit 3 above adds the local derivation; tighter explicit bounds remain a future-work opportunity.
- The noncompact $\mathrm{GL}^+(K)$ extension of the construction is registered at lines 4368 and 4580 as not provided by the present manuscript. The verdict's BLUE_WINS holds for the working compact-$G$ regime; the noncompact construction is a separate research program.
