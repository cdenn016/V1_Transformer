# Evidence Pack — pifb-consensus-gauge-invariance (Debate 3)

## Manuscript references (Attention/Participatory_it_from_bit.tex)

- `:2958–2966` — naive average $\bar G_{\mu\nu}=\tfrac1N\sum w_i G_{i,\mu\nu}$ depends on each agent's arbitrary gauge frame; "critical flaw"; physical geometry should be gauge-invariant (gauge = redundancy of description).
- `:2971–2977` (`sec:consensus_metric`) — proposed gauge-averaged metric $\langle G_i\rangle_{\mu\nu}=\int_G dg\, G_{i,\mu\nu}(c;U_i\mapsto U_i g)$, Haar measure. Then the substantive limitations, stated in place: for CONSTANT $g$, $A\to g^{-1}Ag$ and $\mathrm{tr}(A_\mu A_\nu)$ is "already invariant by cyclicity of the trace; the Haar average over a single copy of $G$ is then either trivial or unnecessary, and does not by itself rescue the non-invariance." The non-invariance concerns LOCAL $g(c)$: $A\to g^{-1}Ag+g^{-1}dg$ (Maurer–Cartan), so an honest gauge-orbit average integrates over all maps $g:\mathcal{C}\to G$ — "an infinite-dimensional functional integral ... requires a gauge fixing or a regulator to be well-defined; in general no finite gauge-orbit average over local $g$ exists without such a choice." Non-compact $\mathrm{SO}(1,3)$: "Haar measure is infinite even for constant $g$." Retained as a HEURISTIC, explicitly NOT claimed to produce a finite regulator-free gauge-invariant metric.
- `:2981–2986` (Eq. `consensus_metric`) — $\bar G^{\text{consensus}}_{\mu\nu}=\sum_i w_i\langle G_i\rangle_{\mu\nu}$; "a heuristic target rather than a completed observable"; gauge-invariance "conditional on a regulator whose construction is left to future work"; transfers to the model fiber; prose elsewhere calling it "the closest analog to objective reality" should be read in this conditional sense.
- `:2988–2992` — "Connection to Physical Gauge Invariance": proposes gauge invariance in physics "arises as a consistency requirement for multi-agent consensus"; lists $\mathrm{U}(1),\mathrm{SU}(2),\mathrm{SU}(3),\mathrm{SO}(1,3)$ as available subgroups of the connection-sector symmetry; "metaphysical interpretation rather than a derivation, and the hypothesis may not be falsifiable from within the framework."

## Canon facts to verify (experts: WebFetch)

- Trace cyclicity: $\mathrm{tr}(g^{-1}A_\mu g\, g^{-1}A_\nu g)=\mathrm{tr}(A_\mu A_\nu)$ for constant $g$ — elementary; confirms the constant-$g$ invariance claim.
- Maurer–Cartan inhomogeneous term in the gauge transformation of a connection $A\to g^{-1}Ag+g^{-1}dg$ — Nakahara §10.1–10.4; Kobayashi–Nomizu. Confirms local-$g$ non-invariance.
- Gauge-orbit averaging / Faddeev–Popov: integrating over the gauge group of maps requires gauge-fixing; the orbit space is infinite-dimensional — Peskin & Schroeder §9.4; Weinberg Vol. 2. Confirms the functional-integral obstruction.
- Haar measure: a locally compact group has finite Haar measure iff it is compact; $\mathrm{SO}(1,3)$ non-compact → infinite total Haar measure — Knapp; Folland *A Course in Abstract Harmonic Analysis*. Confirms the non-compact obstruction.
- `external_canon_math.md` / `external_canon_inference.md` for any consensus/averaging canon.

## What this evidence does NOT settle

- Whether listing $\mathrm{U}(1),\mathrm{SU}(2),\mathrm{SU}(3),\mathrm{SO}(1,3)$ as "available as subgroups of the connection-sector gauge symmetry" (:2992) is a harmless structural observation or an overclaim that gestures at deriving the Standard Model gauge group (the manuscript does hedge "whether specific subgroups are dynamically selected ... remains open").
- Whether "gauge invariance emerges from consensus requirements" is (a) a coherent if-unfalsifiable interpretation, (b) circular/backwards (gauge invariance is assumed in the construction — KL is GL(K)-invariant by setup — so deriving it from consensus may be assuming what it concludes), or (c) vacuous. Philosophy-of-science lens central.
- Whether calling the construction a "heuristic target" fully discharges the earlier prose's "closest analog to objective reality" framing, or whether that earlier prose (referenced at :2986) still overclaims at its own location.
