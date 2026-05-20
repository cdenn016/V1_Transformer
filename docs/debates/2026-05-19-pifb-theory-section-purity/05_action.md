# Action — pifb-theory-section-purity

**From verdict:** RED_WINS (omnibus), with granular per-sub-claim breakdown (blue wins 6 of 8 line items; red wins on C4, C6, and the omnibus framing).

## Recommended action

Reject the omnibus claim as literally stated, accept the granular finding. The Theory section of `Attention/Participatory_it_from_bit.tex` is not uniformly "theoretically pure and mathematically correct against the standard literature"; it is a *mostly* literature-compliant construction with two flagged framing problems and several honestly labeled extensions. The two specific manuscript fixes that would convert "RED_WINS on omnibus" to "BLUE_WINS on omnibus" are:

### Fix 1 — C6 (Regime II Wilson link variable), highest priority

Edit `Attention/Participatory_it_from_bit.tex` line 845. Current text claims `$\delta_{ij}$` is "gauge-invariant in this parameterization." Replace with language that distinguishes invariance-under-a-lifted-gauge-action from invariance-as-a-physical-observable. Suggested replacement:

> The connection coefficients $\delta_{ij}$ are invariant under the chosen lift of the vertex-local gauge transformation, in which the action $U_i \mapsto g_i U_i$ is absorbed entirely into the vertex factors and the central $\exp(\delta_{ij}\cdot G)$ is left untouched. The underlying link variable $\Omega_{ij}$ remains gauge-covariant in the standard lattice gauge theory sense [WilsonConfinement1974, eq.~12; Creutz1983, ch.~5 eq.~5.1], transforming as $\Omega_{ij} \mapsto g_i \Omega_{ij} g_j^{-1}$. Gauge-invariant observables of the connection require closed-loop traces, supplied below by the Wilson observable $W_{ijk}$ of Eq.~\eqref{eq:wilson_observable}.

This change makes the gauge-invariance claim literature-compatible.

### Fix 2 — C4 (Lahav–Neemeh CFR identification), framing only

Edit `Attention/Participatory_it_from_bit.tex` lines 768–776 to either:

a) **Weaken** the identification to a structural analogy by replacing "The construction in this section supplies it" at line 770 with "The construction in this section supplies one mathematically explicit realization of such a law; the choice of $\mathrm{GL}(K)$ as the structure group is a modeling commitment justified by Section~\ref{sec:gauge_group_choice} rather than by Lahav and Neemeh themselves."

b) **Strengthen** the identification by citing Lahav–Neemeh 2025 (already in the bibliography at line 770) and pointing to whatever transformation-law content the 2025 follow-up provides.

Either edit makes the C4 claim literature-compatible without retracting the Lahav–Neemeh reference.

### Acceptable as-is (no edit required)

- C1, C2, C3 (group-level), C5 (with downstream `sec:meta_agent_rg`), C7a (with downstream `sec:meta_agent_variational` bound), C7b — all literature-compliant.

- The cross-scale shadow construction (Eq. cross_scale_shadow, line 540), the multi-agent VFE coupling (line 1021), the culture closure inequality (line 711), and the Regime II promotion of `$\delta_{ij}$` as an independent field — all honestly labeled as extensions beyond standard FEP / lattice-gauge canon. Both teams agreed on this point.

## Follow-up debates (if user wishes to push further)

1. **C6 focused sub-debate:** "A vertex-trivialized parameterization $\Omega_{ij} = U_i \exp(\delta_{ij}\cdot G) U_j^{-1}$ with $\delta_{ij}$ inert under the lifted vertex-local gauge action constitutes the standard Wilson lattice gauge theory link variable in the sense of Creutz 1983 Ch. 5."

   Expected outcome: this is the genuinely contested mathematical question; the verdict above gives RED the point on the literal "gauge-invariant" framing at line 845 but does not adjudicate whether a more careful parameterization could rescue the construction. A focused debate could settle it.

2. **C4 focused sub-debate:** "Lahav and Neemeh 2025 (the follow-up cited at manuscript line 770) supplies the Lie-group transformation law between cognitive reference frames that the 2022 paper omits, and the manuscript's $\mathrm{GL}(K)$ identification matches that supply."

   Expected outcome: depends on whether L–N 2025 actually contains the relevant content. The judge did not have access to that paper during this debate. If L–N 2025 supplies the law, blue can flip C4 to a literature-match.

3. **Cross-scale shadow legitimacy sub-debate:** "The cross-scale shadow construction $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ (Eq. cross_scale_shadow, line 540) is a legitimate refinement of Friston 2017 / Parr–Pezzulo–Friston 2022 hierarchical variational inference, or an undeclared model substitution."

   The manuscript flags this distinction explicitly at line 546 ("This is a structural commitment of the framework rather than a theorem of standard hierarchical variational inference") but the user might want a focused adjudication of whether the substitution is principled. Not adjudicated in this omnibus debate.

## Per-sub-claim verdict map (for the user's reference)

| Sub-claim | Topic | Outcome | Decisive citation |
|---|---|---|---|
| C1 | Principal/associated bundle skeleton | BLUE | Kobayashi–Nomizu Vol. I Ch. II §5; Nakahara §10.1.1; Steenrod §3 |
| C2 | Gaussian group action + GL(K) KL invariance | BLUE | Amari 2016 §2.4; Csiszár 1963; Liese–Vajda 2006 §1 |
| C3 | Multi-agent overlap + epistemic collapse (group-level) | BLUE | Hall 2015 §3.5; manuscript line 795 BCH caveat |
| C4 | Lahav–Neemeh CFR identification | **RED** | Manuscript line 772 "working correspondence rather than an identity"; L–N 2022 supplies no Lie-group law |
| C5 | Culture closure as RG block-spin | BLUE | Wilson 1971 §III; Cardy 1996 ch.~3; manuscript line 4351–4427 supplies the construction red claimed was missing |
| C6 | Regime II Wilson link variable | **RED** | Creutz 1983 ch.~5 Eq.~5.1; Wilson 1974 §II Eq.~12; manuscript line 878 implementation admission |
| C7a | Lie-algebra weighted average vs Karcher mean | BLUE | Karcher 1977 §1; Pennec 2006 §3; manuscript line 2100 supplies the bound |
| C7b | State vs model fiber representations | BLUE | Nakahara §10.1 Eq.~10.7; manuscript line 596 two-representation definition |
| Omnibus | Entire Theory section is literature-pure | **RED** | Blue concession at `03_blue_rebuttal.md` lines 5–13 |

## Status

This debate is closed. Two short manuscript edits (Fix 1 at line 845; Fix 2 at lines 770–772) would convert the omnibus outcome to BLUE_WINS without retracting any substantive theoretical content.
