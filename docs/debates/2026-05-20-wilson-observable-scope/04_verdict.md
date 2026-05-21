# Verdict — wilson-observable-scope

## Outcome

RED_WINS

## Decisive evidence

Blue's own concession in `03_blue_rebuttal.md`:5–7 grants every element of the claim's blue-loses falsification trigger. The concession reads verbatim: "PIFB:868 (`the form actually implemented as the optional holonomy_penalty regularizer`) and PIFB:874 (`The implementation exposes α as the cocycle_relaxation configuration parameter`) are direct declarative statements about an implementation identifier, and `grep -r 'holonomy_penalty\|cocycle_relaxation\|wilson' transformer/vfe/` returns zero matches (per `01_evidence.md` lines 8–12). Those two manuscript lines, taken at face value against the user's nomination of `transformer/vfe/` as the canonical pure path, are false. This is a real manuscript-vs-code consistency gap." That concession satisfies the claim's clauses (i), (ii), and (iii) verbatim: `non_flat.py` is reachable in /vfe under `use_non_flat_transport=True` (per `01_evidence.md` and `transformer/vfe/config.py:677-705`), PIFB:868 and PIFB:874 declaratively name the two identifiers, and the /vfe grep is empty.

## Reasoning

The claim's `00_claim.md` falsification structure makes the verdict mechanical once the three named conditions are admitted. Blue concedes all three. The remaining dispute is whether the construction belongs to Regime II — but the claim was not "Wilson is Regime I"; it was "the manuscript's own description of the implementation names identifiers absent from the canonical pure path." On that narrower disposition, blue's defense reduces to the wider-tree reading of "the implementation" in PIFB:868 and PIFB:874, and red's rebuttal correctly observes that the definite article in those sentences carries no scope qualifier locating "the implementation" outside /vfe. Blue's Regime II citations (PIFB:824, 826, 878, 880) are well-grounded and establish that the construction is not language-modeling-core; they do not erase the consistency gap, they explain its origin.

The source-of-truth precedence routes the remediation. Wilson 1974 §3, Kogut-Susskind 1975 §II, and Creutz 1983 §5.1 all define the Wilson loop as the gauge-invariant observable on a closed path in a lattice. PIFB:876 itself states that causal autoregressive attention defines a directed acyclic graph with no closed cycles. The /vfe module's default runtime is causal lower-triangle masking (`transformer/vfe/train_vfe.py:38`), so the Wilson observable is canonically degenerate on the language-modeling path /vfe realises. Adding Wilson machinery to /vfe would import a regularizer that PIFB:880 itself states is "degenerate in that limit." The gap is therefore correctly closed by manuscript scope clarification at PIFB:868 and PIFB:874 — not by Wilson code in /vfe.

This is RED_WINS on the literal falsification trigger with a manuscript-side action; blue's Regime II classification is preserved as the rationale for routing the fix to the text rather than the code.

## Action

Edit the manuscript at PIFB:868 and PIFB:874 to scope-qualify the implementation referent. Specifically:

1. At `Attention/Participatory_it_from_bit.tex:868`, replace "is the form actually implemented as the optional `holonomy_penalty` regularizer" with a scope-explicit form such as "is exposed as the optional `holonomy_penalty` regularizer in the diagnostic and Regime II research branches of the codebase (`transformer/core/connection.py`, `transformer/analysis/holonomy_metrics.py`); the Regime I language-modeling path in `transformer/vfe/` does not include this term because the holonomy diagnostics are degenerate in the pure-gauge limit (PIFB:880)."

2. At `Attention/Participatory_it_from_bit.tex:874`, replace "The implementation exposes α as the `cocycle_relaxation` configuration parameter" with a scope-explicit form such as "The Regime II research branch exposes α as the `cocycle_relaxation` configuration parameter (in `transformer/core/connection.py`); the Regime I `/vfe` path realises the α = 0, δ_ij ≡ 0 limit by construction at initialization (`transformer/vfe/non_flat.py:228-234`) and provides no α-knob because the homotopy is collapsed."

Do not add Wilson observable, `holonomy_penalty`, or `cocycle_relaxation` machinery to `transformer/vfe/`. The external canon ([Wilson1974 §3], [KogutSusskind1975 §II], [Creutz1983 §5.1]) plus PIFB:876 and PIFB:880 jointly establish that the construction is structurally trivial on the causal autoregressive DAG `/vfe` realises, and PIFB:824, 826, 878, 880 explicitly label the section as Regime II content outside the language-modeling-only restriction. The two-line manuscript edit closes the consistency gap red identified while preserving the Regime I purity of `/vfe`.
