# Claim — wilson-observable-scope

**Mode:** implementation
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (transformer/vfe/non_flat.py, PIFB:828-880, PIFB:847-867)
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge\

## Claim

PIFB §"Discrete Regime II via an Edge-Relaxed Cocycle" (PIFB:828-880) is part of the language-modeling core because (a) `transformer/vfe/non_flat.py` implements its central construction `Ω_ij = U_i · exp(δ_ij · G) · U_j^{-1}` (PIFB:eq:edge_relaxed_omega at line 836-837), reachable under the `use_non_flat_transport=True` toggle, and (b) PIFB:874 itself states "The implementation exposes α as the `cocycle_relaxation` configuration parameter." The manuscript's prescribed Wilson observable `W_ijk = Re Tr[exp(δ_ij·G) exp(δ_jk·G) exp(δ_ki·G)]` (PIFB:eq:wilson_observable, line 858) and the Wilson holonomy-penalty regularizer `S_Wilson[δ] = β Σ_{(i,j,k)} (1 - W_ijk/K)` (PIFB:eq:wilson_action, line 862-867) plus the homotopy parameter α at PIFB:871 are *missing* from `transformer/vfe/` (no occurrences of `cocycle_relaxation`, `holonomy_penalty`, or `wilson` in the `/vfe` subtree), and are not reachable under any toggle. This is a falsifying gap in the /vfe module's realization of the PIFB framework for language modeling.

## User context

Follow-up to `docs/debates/2026-05-20-vfe-module-purity-for-pifb/04_verdict.md`. The original verdict ruled Wilson constructions are Regime-II content placed in the framework's "gravitational and signature-related extrapolations" (per PIFB:824 and PIFB:826) and therefore out of scope for the language-modeling-only restriction. This debate reopens that ruling on the basis that `non_flat.py` IS reachable in the /vfe language model, and the manuscript itself claims the Wilson regularizer is implemented (PIFB:868: "The squared Frobenius variant ... is the form actually implemented as the optional `holonomy_penalty` regularizer.").

## Falsification conditions

- **Blue (defender) loses** if red shows (i) `non_flat.py` is reachable in the /vfe language model (use_non_flat_transport=True is a legitimate toggle, not a vestigial config), (ii) PIFB:868 and PIFB:874 explicitly claim `holonomy_penalty` and `cocycle_relaxation` are implemented, and (iii) grep of `transformer/vfe/` for those identifiers returns zero matches.
- **Red (attacker) loses** if blue shows either (a) PIFB:824 / PIFB:826 / PIFB:880 places the Wilson observable strictly out of the language-modeling Regime I context (Regime II is the "gravitational and signature-related extrapolations" by the manuscript's own labeling), and the non_flat.py path is a Regime II *option* that does not invalidate the Regime I default, OR (b) the Wilson observable is reachable in /vfe under some toggle the red team missed.
