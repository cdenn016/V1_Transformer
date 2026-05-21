# Action — wilson-observable-scope

**From verdict:** RED_WINS (literal trigger; manuscript-side remediation)

## Recommended action

PIFB:868 ("the form actually implemented as the optional `holonomy_penalty` regularizer") and PIFB:874 ("The implementation exposes α as the `cocycle_relaxation` configuration parameter") are direct manuscript claims about implementation features that are absent from `transformer/vfe/` (grep returns zero matches) but present in the broader codebase (`transformer/core/`, `transformer/analysis/`, `transformer/pure_vfe/`).

The remediation routes through the manuscript, not the /vfe code, because:
- Wilson 1974, Kogut-Susskind 1975, Creutz 1983 define Wilson loops on closed cycles.
- PIFB:876 itself observes that autoregressive (causal) transformer DAGs have no closed loops, making the Wilson observable structurally degenerate in /vfe's default LM path.
- PIFB:824, 826, 878, 880 explicitly label the Wilson constructions as Regime II content.
- Adding Wilson machinery to /vfe would import a term PIFB:880 declares degenerate in the Regime I limit.

Applied in this round: two-line manuscript scope qualification at PIFB:868 (line for `holonomy_penalty`) and PIFB:874 (line for `cocycle_relaxation`), making explicit that "the implementation" of those features refers to the Regime II research branches in `transformer/core/` and `transformer/analysis/`, not the Regime I LM pure path at `transformer/vfe/`.

## Follow-up debates (if any)

None directly. If the user later wants Wilson regularization in a bidirectional `/vfe` configuration (the language-modeling experimental test of the gauge-curvature linguistic conjecture per PIFB:876), that is a research-track code addition warranting its own debate and dedicated implementation.
