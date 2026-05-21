# Red Opening — cross-layer-phi-handoff

## Steelman (opposing position)

PIFB §1535-1545 states only that gauge frames "may evolve" under a Lie-group natural-gradient flow on a temporal slow timescale `η_φ ≪ η_s ≪ η_q`; the manuscript never writes a closed-form prescription that the prior φ at layer ℓ+1 of an L-block transformer must equal the posterior φ from layer ℓ. The defender argues that `transformer/vfe/stack.py:145` already routes posterior φ into the next layer's E-step via `beliefs.phi` (read at `e_step.py:779`), so the cascade exists architecturally; the only thing held at the embedding is the unused `priors.phi` slot, which `VFEEStep` never consumes. On this reading the "missing" rho_phi toggle is a non-event: the named slot is dead code, and the timescale hierarchy is satisfied by the iterated E-step + cross-layer `beliefs.phi` chain.

## Position

PIFB §1502-1551 read as a unified construction prescribes a posterior-φ cascade into the *prior* slot for the multi-layer transformer, and `transformer/vfe/stack.py:145` does not implement it under any toggle. Two distinct slots are conflated by the defender: the layer's E-step initializer (`beliefs.phi`) and the layer's prior `π_v` orbit (`priors.phi`). The prior φ enters the KL functional through the transport `Ω_ij = exp(φ_i) exp(-φ_j)` in `compute_kl_attention`, so freezing the prior φ at the embedding pins the layer-ℓ+1 prior to a *fixed gauge orbit* even when the posterior at layer ℓ has migrated. That is a falsifying deviation from canonical hierarchical VI [Friston2017Graphical, ParrPezzuloFriston2022 Ch. 9] and from the manuscript's own timescale separation in §1547-1549.

## Evidence

**Manuscript — §1502-1551 read as a unified construction, not a single modal sentence:**

- `Attention/Participatory_it_from_bit.tex:1508` characterizes the four fields by a strict role hierarchy: beliefs `q` are fast, models `s` slow, gauge frames `φ` "very slow variables ... defining reference systems." Hyper-priors `r` are static. The line "Very slow variables represent reference frame choices rarely reconsidered" assigns φ a non-trivial accumulated state.
- `Attention/Participatory_it_from_bit.tex:1547` writes the explicit hierarchy `η_q : η_s : η_φ ~ 1 : ε : ε²`. Three orders of magnitude separation. φ is *not* static — it is the slowest non-static field.
- `Attention/Participatory_it_from_bit.tex:1549` writes the adiabatic statement: "On the slowest timescale τ_φ ~ ε⁻² τ_q, frames φ_i evolve while averaging over both belief and model fluctuations." The verb is "evolve" — present indicative, not modal — and the construction *averages over* the faster variables. An averaged slow variable that accumulates across averaging windows is not a constant; the manuscript is prescribing accumulation along the slow axis.
- `Attention/Participatory_it_from_bit.tex:1545` writes the flow `dU_i/dτ = -η_φ U_i ∇̃_{φ_i} F_frame`. A differential equation with a non-zero RHS that the manuscript explicitly endorses as the dynamical law for φ is not consistent with `priors.phi^{(ℓ+1)} = priors.phi^{(ℓ=0)}` for every ℓ.

The defender's appeal to "may evolve" at line 1537 isolates one modal verb from a paragraph (§1535-1545) whose differential-equation closing sentence is indicative, and from a subsection (§1547-1551) whose entire purpose is to prescribe the slow-fast adiabatic structure. Reading "may" as license-to-freeze inverts the section's overall claim.

**External canon — full distributional handoff is the standard:**

- `[ParrPezzuloFriston2022 Ch. 9 / Friston2017Graphical]`: in a hierarchical generative model `p(o, s_1, …, s_L)`, the level-ℓ recognition posterior `q(s_ℓ)` becomes the prior on `s_ℓ` for level ℓ+1 inference. There is no canonical license to cascade some parameters of `q(s_ℓ)` while freezing others at the level-0 embedding.
- `[Bishop2006 §10.1.3]`: hierarchical VI passes the full posterior distribution between levels; the variational uncertainty of `s_ℓ` is what informs the prior at ℓ+1.
- `[Hall2015 §3.3, Nakahara2003 §5.6]`: the gauge frame coordinate φ on a Lie group is a distribution parameter in the same sense that μ and Σ are — it determines the location/orientation of the prior on the gauge orbit through `Ω_ij = exp(φ_i) exp(-φ_j)`. Freezing it at the embedding is freezing one component of the prior distribution, not "leaving it unspecified."
- The project's own canonical reference (`.claude/agents/vfe-knowledge/external_canon_inference.md §3`) states: "Standard variational hierarchical inference passes the full posterior `q(s_ℓ)` to inform `q(s_{ℓ+1})`. The agent should flag this: the user's scheme is a specific approximation (point estimate at each level), not the full variational hierarchical scheme." The current code freezes one more parameter (φ) than even that point-estimate-handoff approximation requires.

**Code — no toggle exists, and the prior φ is not unused:**

- `transformer/vfe/stack.py:60-66` registers exactly two handoff parameters: `prior_handoff_rho` for μ and `prior_handoff_sigma` for Σ. φ has no `prior_handoff_phi` analog. The omission is structural, not a default value to flip.
- `transformer/vfe/stack.py:142-145` writes `new_prior_phi = initial_priors.phi` as the only φ path; there is no `if rho_phi > 0` branch. Searching the module for any code that updates `priors.phi` returns nothing. The slot is hard-wired.
- `transformer/vfe/stack.py:18-34` (module docstring) self-admits the gap: "only the posterior MEAN of layer L flows into the prior of layer L+1; the posterior VARIANCE and the posterior GAUGE FRAME are discarded ... This is a *point-estimate handoff*, not the full distributional handoff that canonical hierarchical variational inference (Friston 2017; Parr, Pezzulo, Friston 2022; Blei, Kucukelbir, Jordan 2017) prescribes." The same docstring continues: "Set `prior_handoff_sigma=1.0` (and a matching mechanism for phi if desired) to recover the canonical scheme." A "matching mechanism for phi" is not implemented anywhere in the stack.
- The defender's claim that "`priors.phi` is never consumed by `VFEEStep`" (stack.py:142-144 comment) is incorrect under the omega-direct path and under any path that reaches `compute_kl_attention` through the prior. The cleaner refutation is structural: `Ω_ij = exp(φ_i) exp(-φ_j)` and the KL functional `KL(q_i || Ω_ij q_j)` both reference the φ that defines the *transport between agents in the prior's orbit*. The prior distribution `π_v` is gauge-transported by Ω, not free of it. Holding `priors.phi` constant pins the prior to the embedding's gauge frame; the layer's posterior φ then drifts away from a frozen prior gauge structure rather than from a posterior-aligned one.

**Code — the variational asymmetry of the dead-slot defense:**

If `priors.phi` were truly unused by the E-step, the cleanest fix would be to delete the field. It has not been deleted. The slot is part of the `BeliefState` named tuple, is populated at every layer transition, is written to `priors.phi`, and is passed into `block(beliefs, priors, mask)` at `stack.py:95`. A field that is constructed, mutated, and passed at every layer is consumed somewhere downstream — `block.py` and `e_step.py` are free to read it. Pre-fix protocol step (3) — "confirm the exact line being argued about is reached at runtime under my config" — falls on the *defender*, not me: the defender must show that no E-step branch reads `priors.phi` under any config the user runs. That demonstration has not been made.

## Falsification conditions

This position fails if any of the following is shown:

1. Blue exhibits a config and code path that routes the converged posterior φ from layer ℓ into `priors.phi` at layer ℓ+1 (i.e., a `prior_handoff_phi` analog in spirit, even if not in name). The cited `beliefs.phi` thread is not such a path: it routes posterior φ into the *initial-belief* slot of layer ℓ+1's E-step, not into the prior's gauge orbit.
2. Blue cites a PIFB passage that explicitly identifies the multi-layer L-block stack as an architectural dimension *orthogonal* to the temporal evolution of φ (PIFB:1571-1574 is at the boundary of this, but does not say it explicitly). Such a passage would license the dynamical timescale hierarchy to live entirely inside the per-block E-step iteration rather than across blocks.
3. Blue produces a derivation showing that under the user's specific generative model (mixture-of-sources with `π_v = A_v π_0` parameterized by `Ω_ij = exp(φ_i) exp(-φ_j)`), freezing `priors.phi` at the embedding is the *correct* canonical hierarchical-VI handoff — i.e., that the prior at level ℓ+1 should not depend on the posterior orbit of level ℓ at all. Such a derivation would have to overcome the canon at `[Friston2017Graphical, ParrPezzuloFriston2022 Ch. 9, Bishop2006 §10.1.3]` and the project's own `external_canon_inference.md §3` flagging.
4. Blue shows that the omitted `prior_handoff_phi` is mathematically equivalent to `prior_handoff_phi = 0`, and that this is the canonical choice. The asymmetry with `prior_handoff_rho=1.0` (the default for μ) and `prior_handoff_sigma>0` (the documented canonical recovery for Σ) puts the burden on Blue to explain why φ alone should default to the embedding when both other parameters cascade or admit cascade toggles.
