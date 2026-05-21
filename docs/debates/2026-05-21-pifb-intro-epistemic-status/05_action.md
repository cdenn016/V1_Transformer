# Action — pifb-intro-epistemic-status

**From verdict:** RED_WINS

## Recommended action

Three textual edits to the introduction. After application, re-spawn the debate with the same evidence pack to confirm §1.6 / §1.7 framing matches the body audit.

1. **Expand the §1.7 line 142 Lorentzian-postulate enumeration** to surface all six body-flagged postulates that `Attention/Participatory_it_from_bit.tex:2735, :2774, :2783` identify. The current §1.7 entry names three (constant generator, imaginary temporal gauge component, real-part projection); §sec:open_problems line 3499 names a different three. The missing items are:
   - Regime I / $F_{\mu\nu} \equiv 0$ pure-gauge restriction (line 2735).
   - $+\mathrm{tr}(AB)$ bilinear-form sign convention with the non-compact generator choice (line 2774).
   - Separability ansatz $\psi_\tau(\tau), \psi_x(x)$ (line 2783).

   Bring §sec:open_problems line 3499 into postulate-count alignment with the resulting §1.7 list.

2. **Replace the §1.6 line 125 "recovered" framing** with the body's actual epistemic claim. Current: "Standard scaled dot-product attention is recovered as a zero-dimensional isotropic-Gaussian limit of the KL-consensus construction up to a separately introduced learned bilinear compatibility $M$ and the standard normalization and bias assumptions." Replace with the §sec:transformers line 1601 phrasing:
   > "Standard scaled dot-product attention is consistent with a zero-dimensional isotropic-Gaussian limit of the KL-consensus construction under three explicit reductions (isotropic covariances, trivial-frame transport, learned bilinear $M$)."

   Then add an explicit sentence in §1.6 separating the analytic limit (derived in §sec:transformers) from the empirical sweep (in §sec:scaling_validation, which trains the general gauge model rather than the trivial-frame reduction; see line 1609).

3. **Add a §1.7 entry recording the consensus-metric regulator gap.** The consensus metric is the framework's only candidate gauge-invariant geometric observable. §sec:consensus_metric line 2880 and §sec:open_problems line 3509 admit it is "regulator-dependent and is presented there as a heuristic rather than a finite gauge-invariant observable." §1.7 currently has no entry for this status; the §1.7 "No quantitative physics predictions" entry covers numerical absence, not structural-observable-not-finitely-defined status.

## Follow-up debates (if any)

After the three edits, **re-spawn this debate with the same evidence pack** to confirm the framing matches the body audit. If the re-spawn returns BLUE_WINS, the introduction is calibrated; if it returns REMAND or RED_WINS, additional iterations are needed.

The two sibling debates in `2026-05-21-pifb-intro-prior-art/` (REMAND) and `2026-05-21-pifb-intro-pan-agentic/` (REMAND) proceed independently.
