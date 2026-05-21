# Evidence Pack — cross-layer-phi-handoff

## Code references

- `transformer/vfe/stack.py:60-66` — VFEStack `__init__`: stores `prior_handoff_rho`, `prior_handoff_sigma`, builds `n_layers` × VFEBlock.
- `transformer/vfe/stack.py:94-152` — `forward()` loop body. Per layer ℓ:
  - μ: `new_prior_mu = (1-ρ_μ)·priors.mu + ρ_μ·beliefs.mu` (full cascade when `ρ_μ=1.0`, default).
  - Σ: `rho_sigma == 0.0` → `new_prior_sigma = initial_priors.sigma` (frozen at embedding); `rho_sigma > 0` → SPD eigenvalue floor blend; `rho_sigma == 1.0` → full posterior σ cascade.
  - φ: `new_prior_phi = initial_priors.phi` — **hard-wired to the embedding value with no rho_phi toggle, line 145**.
- `transformer/vfe/stack.py:142-144` (comment block): "priors.phi is never consumed by VFEEStep (phi is initialised from beliefs.phi, which already flows posterior -> next-layer input), so the prior phi stays at the embedding value."
- `transformer/vfe/e_step.py:779` — `phi = beliefs.phi`: the inner E-step *reads* posterior φ from the incoming `beliefs.phi`, NOT from `priors.phi`. The next layer's incoming `beliefs.phi` is set by `VFEBlock.forward` to the converged posterior φ from the previous layer's E-step. So posterior φ DOES cascade — just not via the named `priors.phi` slot.
- `transformer/vfe/block.py:172-192` — `VFEBlock.forward`: `beliefs = self.e_step(beliefs, priors, mask)`; head_mixer then norm; returns updated beliefs (mu, sigma, phi all from E-step output). The phi the next layer sees in `beliefs.phi` IS the posterior φ.
- `transformer/vfe/stack.py:1-34` — module docstring: "Implementation note — mean-only cascade vs canonical hierarchical VI." Acknowledges that under the default `(ρ_μ=1.0, ρ_σ=0.0)` only the posterior mean cascades into the prior, with σ and φ frozen at embedding values in the *prior* slot. Says: "Set `prior_handoff_sigma=1.0` (and a matching mechanism for phi if desired) to recover the canonical scheme."

## Manuscript references

- `Attention/Participatory_it_from_bit.tex:1502-1569` — §"Dynamical Structure and Emergent Timescales". Timescale hierarchy `η_q : η_s : η_φ ~ 1 : ε : ε²`. Belief `q` is fast; model `s` is medium; gauge frame `φ` is very slow.
- `Attention/Participatory_it_from_bit.tex:1535-1545` — §"Gauge Frame Evolution":
  > "Gauge frames `φ_i(c)` may evolve to minimize free energy with respect to frame choices. ... Gauge frames evolve under the Lie-group natural-gradient flow `dU_i = -η_φ U_i ∇̃_{φ_i} F_frame` (Eq:gauge_natural_gradient) with learning rate η_φ ≪ η_s ≪ η_q, corresponding to slow updates on coordinate system recalibration timescales."
- `Attention/Participatory_it_from_bit.tex:1547-1551` — §"Timescale Hierarchy and Adiabatic Approximation":
  > "On the slowest timescale τ_φ ~ ε⁻² τ_q, frames φ_i evolve while averaging over both belief and model fluctuations."
- `Attention/Participatory_it_from_bit.tex:1572-1574` (transformer-limit setup) — does not explicitly prescribe a cross-layer φ-cascade pattern for the L-block stacking; the L blocks are treated as an architectural depth dimension separate from the temporal evolution of φ.

## Canon excerpts (external sources of truth)

- **Friston 2017 (active inference, hierarchical generative models)**, **Parr/Pezzulo/Friston 2022 Ch. 4** — canonical hierarchical variational inference: at each level, the posterior of level ℓ becomes the prior of level ℓ+1 for ALL distribution parameters. There is no canonical license to freeze one parameter at the embedding value while cascading another.
- **Bishop 2006 §10.1.3** (hierarchical VI) — full distributional handoff is the standard.
- **Hall 2015 §3.3 / Nakahara 2003 §5.6** — the gauge frame parameter (Lie-algebra coordinate φ) IS a distribution parameter in the sense that it determines the location/orientation of the prior on the gauge orbit `π_v = A_v π_0`; freezing it at the embedding is therefore freezing one component of the prior, not "leaving it unspecified."

## What this evidence does NOT settle

1. The transformer limit at PIFB:1571-1842 does not explicitly write down a multi-layer cascade — the manuscript treats the multi-layer transformer as a depth-dimension extension of the per-position dynamics, and the natural-gradient flow `dU/dt` of PIFB:2547 is a temporal flow, not a depth flow. The mapping from "temporal evolution of φ at rate η_φ" to "L-block φ cascade" is the *user's architectural choice*, not the manuscript's prescription.
2. The `stack.py:142-144` comment claims "phi already flows posterior → next-layer input" via `beliefs.phi`. If this is true, then the apparent gap is only a naming/slot issue — the cascade IS happening, just not via `priors.phi`. Verify whether `e_step.py` actually reads φ from `beliefs.phi` and uses the posterior value, or whether it re-initializes from somewhere else.
3. The user's `CLAUDE.md` states `EM_CONFIG Best Results` for layernorm + no residual + skip_attention + n_layers=1 — i.e., the user's best empirical results use only a single layer where the cross-layer φ handoff question is moot. The architectural depth dimension is itself somewhat secondary in the user's empirical work.
