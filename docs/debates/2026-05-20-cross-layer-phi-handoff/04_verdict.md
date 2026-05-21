# Verdict — cross-layer-phi-handoff

## Outcome

BLUE_WINS

## Decisive evidence

`transformer/vfe/stack.py:142-145` plus the grep-verified absence of any read of `priors.phi` across the `transformer/vfe/` tree. The three hits returned are `stack.py:14` (docstring), `stack.py:142` (comment), and `stack.py:145` (the write itself). Zero reads in `e_step.py`, `attention.py`, or `block.py`. The operative cascade is at `transformer/vfe/e_step.py:779` (`phi = beliefs.phi`) — the next block's E-step initialises φ from the previous block's posterior φ returned via `BeliefState.phi` from `transformer/vfe/block.py:172` and re-passed by `transformer/vfe/stack.py:94-95`. The transport operator built at `transformer/vfe/e_step.py:835-838` (`compute_gauge_transport(phi, ...)`) reads the cascaded belief-side φ, not the prior slot. Red conceded all of this in `03_red_rebuttal.md` lines 4-12.

## Reasoning

The claim asserts a "falsifying deviation from the canonical hierarchical-VI structure" caused by `stack.py:145` hard-wiring `new_prior_phi = initial_priors.phi` with no toggle. The claim's own falsification condition (b) at `00_claim.md:20` is met: stack.py and e_step.py do in fact route the posterior φ into the next layer's E-step under every config, via `beliefs.phi`, even though the named `priors.phi` slot stays at the embedding. Red conceded this empirical finding in the rebuttal and explicitly withdrew the "no φ cascade" framing.

Red's pivoted attack — that the σ/φ asymmetry (σ has a `prior_handoff_sigma` toggle; φ has no `prior_handoff_phi`) is a structural gap under CLAUDE.md's "theoretically pure path must exist under appropriate toggles" requirement — is the weaker form Red itself flags as defensible only "if the judge weights canonical-handoff symmetry above operational equivalence" (`03_red_rebuttal.md:34`). The weighting goes against Red here on two grounds. First, the σ-slot toggle does real work because `priors.sigma` has an actual operational consumer at `transformer/vfe/e_step.py:785, 537-538, 850` (the self-KL term `α · KL(q_i || p_i)` and `get_bayesian_alpha`). The φ-slot has none. A `prior_handoff_phi` toggle would have nothing to toggle; Blue's defense at `03_blue_rebuttal.md:21` is correct that introducing a meaningful prior-φ would require a new term in F (e.g., a separate prior-φ-dependent Ω endpoint), which is a manuscript-level construction, not a `stack.py` damping flag. Second, PIFB:1545's natural-gradient flow `\dot U_i = -η_φ U_i \tilde{\nabla}_{\phi_i}\mathcal{F}_{\text{frame}}` is a temporal flow on a single agent index i, not a depth flow on the L-block index ℓ. Red's falsification condition (2) in the opening (`02_red_opening.md:45`) anticipated and conceded that PIFB does not equate ℓ with τ_φ. Without that identification, there is no manuscript prescription that the cross-layer prior-φ slot must be the posterior φ; the canonical hierarchical-VI requirement that the posterior of level ℓ inform the prior at level ℓ+1 is on the variational family's parameters, not on every named field in the implementation's BeliefState, and the cascade is already satisfied via the belief channel.

The canonical hierarchical-VI canon Red invokes (Friston 2017; Parr-Pezzulo-Friston 2022 Ch. 4; Bishop 2006 §10.1.3) prescribes that the posterior `q(s_ℓ)` informs the prior at level ℓ+1. The implementation routes that information through `beliefs.phi`; nothing in the canon requires it to route through a field named `priors.phi`. Treating the named slot as the load-bearing object is nominalism, not mathematics. Red's structural premise — that the prior π_v is gauge-transported by Ω in the KL — does not survive `transformer/vfe/attention.py:124-198` where `compute_kl_attention` takes a single `phi` argument (the belief φ) on both endpoints of Ω.

## Action

Documentation cleanup, not a code falsifier. Two acceptable paths.

First, the lightweight path. Update the module docstring at `transformer/vfe/stack.py:18-34` to remove the "and a matching mechanism for phi if desired" hedge and replace it with an explicit statement: the φ slot in `priors` is intentionally unused because the KL functional reads φ from `beliefs` on both transport endpoints, and the posterior-φ cascade is structurally implemented via `beliefs.phi` returned by `VFEBlock.forward`. Cross-reference `transformer/vfe/e_step.py:779` and `transformer/vfe/attention.py:165-181`. This converts the current text (which invites the cascade-gap reading Red executed in the opening) into a positive statement of the design.

Second, the structural path. Delete the `phi` field from `BeliefState` entirely when used as a prior, or split `BeliefState` into a `Posterior` namedtuple (μ, σ, φ) and a `Prior` namedtuple (μ, σ). This is a refactor with no semantic effect under the current call graph and would remove the asymmetry Red exploited. It is a larger change and would touch every E-step call site, so the docstring rewrite is the proportionate fix.

A `prior_handoff_phi` toggle should NOT be added until and unless a manuscript-level construction introduces an operational consumer of `priors.phi` distinct from `beliefs.phi`. Adding a toggle for an inert field would create the appearance of a configurable canonical mode while changing no computation, which is the inverse of the CLAUDE.md "theoretically pure path under appropriate toggles" requirement.
