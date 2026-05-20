# Option (a) — Joint Generative Model Route (deferred)

**Status:** deferred alternative to the soft-assignment-Lagrangian framing currently adopted in `GL(K)_attention.tex` §4.6. This document preserves the scope of work required to upgrade the framing if the user wishes to claim the alignment free energy is *derived* from a joint Bayesian generative model rather than *posited* as an engineered soft-assignment functional.

**Context:** the red-blue debate on softmax-β stationarity (`docs/debates/2026-05-19-softmax-beta-stationary-point/`) returned RED_WINS on the headline claim that "every intermediate step is mathematically valid" because the manuscript at line 697 invoked an "augmented joint generative model" verbally but never wrote it down. The verdict offered two manuscript edits: (a) write the joint model and derive the row coupling from its mean-field ELBO, or (b) adopt the soft-assignment-Lagrangian framing explicitly. The user chose (b). This file captures (a) for future reference.

## What option (a) requires

The row-Lagrangian `F_align^(τ) = Σ_j [β_ij E_ij + τ β_ij log(β_ij/π_j)]` with `E_ij = D_KL(q_i ‖ Ω_ij q_j)` must emerge as the variational coupling produced by mean-field decomposition of a joint ELBO over an explicitly written joint generative model `p(o_{1:N}, k_{1:N}, z_{1:N})` whose parameters do *not* depend on the variational quantities `{q_i}`.

The standard variational mean-field derivation pattern, applied to a joint model `p(o_{1:N}, k_{1:N}, z_{1:N})` with mean-field recognition `Q = Π_i [q_i(k_i) β_i(z_i)]`, produces an ELBO of the form
`L = Σ_i E_{q_i β_i}[log p_i(o_i, k_i, z_i | k_{-i}, z_{-i}, θ)] - Σ_i [H(q_i) + H(β_i)] + (cross-token coupling terms)`
where the cross-token coupling terms come from the conditional dependence `p(k_i | k_{-i}, z_i, θ)` and similar. To reproduce the manuscript's `Σ_j β_ij D_KL(q_i ‖ Ω_ij q_j)`, the joint conditional `p(k_i | k_{-i}, z_i = j, θ)` must be precisely a Gaussian centered on `Ω_ij μ_j` with covariance `Ω_ij Σ_j Ω_ij^T`. This is a strong specific assumption on the joint model: the conditional of one agent's latent given another's is exactly a transported version of the other's posterior parameters, which is an unusual structural form for a Bayesian model (it makes the joint a function of the recognition parameters, which is what option (a) is trying to escape).

### The candidate joint model

A clean way to write a joint model with the required form is to introduce hyper-parameters `θ_i = (μ_i^p, Σ_i^p, φ_i)` (the priors and gauge frames, which are M-step parameters) and treat the joint as
```
p(o_{1:N}, k_{1:N}, z_{1:N} | θ_{1:N}) = Π_i p(o_i | k_i) · p(k_i | z_i, k_{-i}, θ_i, θ_{-i}) · p(z_i | π_i)
```
with the conditional `p(k_i | z_i = j, k_{-i}, θ_i, θ_j) = N(k_i; Ω_ij(φ_i, φ_j) μ_j^p, Ω_ij Σ_j^p Ω_ij^T)`. This makes the joint a function of the *priors* and *gauge frames* (M-step parameters), not of the *posteriors* `{q_i}`. The mean-field recognition `Q = Π_i q_i β_i` then produces an ELBO whose β-row terms reduce to the manuscript's `Σ_j β_ij D_KL(q_i ‖ Ω_ij q_j)` *under the additional substitution* that the conditional `p(k_i | z_i = j, ...)` uses the M-step prior parameters `(μ_j^p, Σ_j^p)` rather than the E-step posterior parameters `(μ_j, Σ_j)`. That substitution corresponds to the "fixed-prior" form of the joint, which the manuscript's current row-Lagrangian does *not* use (the manuscript uses `Ω_ij q_j` where `q_j = N(μ_j, Σ_j)` is the recognition posterior, not the prior).

### The gap to bridge

The cleanest version of option (a) replaces the `q_j` in `D_KL(q_i ‖ Ω_ij q_j)` with the prior `p_j` (or some fixed reference distribution). This is a *different* row-Lagrangian. It would compute attention against fixed neighbors rather than against currently-believed neighbors, which would lose the consensus-energy interpretation and the iterative E-step belief-propagation structure. The user would need to choose: (i) keep the recognition-dependent row coupling and accept that option (a) cannot recover it from a fixed-parameter joint, or (ii) accept the prior-dependent row coupling and lose the iterative belief-propagation structure.

A second, more sophisticated option is to use a coordinate-ascent VI argument: the recognition-dependent coupling emerges if the joint model is written with the variational parameters `{q_j}` of other agents treated as conditioning variables (block coordinate descent over `{q_i}` one block at a time, with the joint conditional `p(k_i | z_i = j, {q_j}_{j ≠ i})` taking the *currently inferred* parameters of `q_j`). This is the "consensus-VI" or "loopy mean-field" reading. It is mathematically defensible as a coordinate-descent fixed-point scheme on a consistency functional, but it does not correspond to standard Bayesian variational inference on a fixed-parameter generative model — and the canon at `external_canon_inference.md` §1 flags this as "novel construction requiring its own justification."

### The literature this would need to cite

To make option (a) rigorous, the manuscript should locate the construction in the multi-agent variational ecology / collective active inference literature:

- **Friston et al. (2017), "Active inference: a process theory."** Hierarchical and multi-agent extensions of FEP. Not specifically the consensus-VI form, but the canonical citation for multi-agent active inference.
- **Ramstead, Constant, Badcock, Friston (2020), "Variational ecology and the physics of sentient systems."** Multi-agent FEP with explicit ecological coupling. Cited in `external_canon_inference.md` §1 as the relevant multi-agent generalization in the standard FEP corpus.
- **Friston, Parr, de Vries (2017), "The graphical brain: belief propagation and active inference."** Belief-propagation framing of inter-agent message passing.
- **Heins, Millidge, Demekas, Klein, Friston, Fields, Da Costa (2024), "pymdp: a Python library for active inference in discrete state spaces."** Recent computational treatment of multi-agent active inference with explicit message-passing structure.
- **Ueltzhöffer (2022), "On the thermodynamics of prediction under dissipative adaptation"** or similar — formal treatment of the consensus-functional reading.

None of these papers (to the present author's knowledge) write the *specific* gauge-transport-coupled joint generative model that the manuscript's row-Lagrangian would derive from. The user would either need to (i) propose this specific joint as a novel construction within the multi-agent FEP framework and provide the missing derivation, or (ii) cite an existing paper that already uses this specific construction.

### Scope of work to upgrade

If the user wishes to upgrade framing (b) to framing (a) in the future, the required additions to `GL(K)_attention.tex` §4 are:

1. **One subsection (~half a page)** introducing the joint generative model `p(o_{1:N}, k_{1:N}, z_{1:N} | θ)` with explicit conditional `p(k_i | z_i = j, k_{-i}, θ_i, θ_j) = N(Ω_ij μ_j^p, Ω_ij Σ_j^p Ω_ij^T)`, distinguishing the M-step prior parameters from the E-step posterior parameters, and noting that the joint depends on M-step parameters only.
2. **One subsection (~half a page)** writing the mean-field ELBO and showing the row-coupling term reduces to `Σ_j β_ij D_KL(q_i ‖ Ω_ij p_j)` (with `p_j` the prior) or alternatively to `Σ_j β_ij D_KL(q_i ‖ Ω_ij q_j)` (with `q_j` the current posterior) under the consensus-VI reading. Whichever choice is made, the manuscript must commit to it.
3. **Citation chain (~one paragraph)** locating the construction in the multi-agent FEP literature (Friston2017Graphical, Ramstead2020, etc.) and labeling any novel structural choices.
4. **Bib entries** for Friston2017Graphical, Ramstead2020, and any other cited sources (currently absent from `references.bib`).

### When option (a) would be worth doing

Option (a) is worth the effort if:
- The user wishes to claim the framework is FEP-derived rather than soft-assignment-engineered. This is a theoretical positioning choice that affects how the manuscript is read in the active-inference / FEP community.
- The user is targeting a journal or audience where "engineered Lagrangian" is read as less rigorous than "variational mean-field." JMLR readers are unlikely to make this distinction (Cuturi 2013 / Sinkhorn is standard there); NeurIPS / active-inference audiences may.
- A primary source (Friston2017Graphical, Ramstead2020, etc.) can be located that uses the specific gauge-transport-coupled multi-agent joint, removing the burden of proposing it as a novel construction.

Option (a) is NOT worth doing if:
- The user is content with the supplementary's own framing at line 183 ("the canonical free energy of the main text adds the τβ log(β/π) entropy term to make the softmax form of β a stationary point") — this is the additive / engineered framing, already adopted.
- The user does not want to commit to a specific multi-agent FEP joint model and would prefer to leave the construction as an engineered functional with its own internal logic.
- The Cuturi 2013 / Boyd-Vandenberghe citations already in place after the framing-(b) edit are read as sufficient by the target audience.

## Pointer to the soft-assignment-Lagrangian framing (current §4.6)

The manuscript at `Attention/GL(K)_attention.tex` §4.6 (lines 679–769 after the 2026-05-19 framing-(b) edits) now states explicitly that `F_align^(τ)` is an engineered soft-assignment functional in the sense of Cuturi 2013, with the entropy term added to make the softmax the exact KKT stationary point on the simplex per Boyd-Vandenberghe §5.5. The strict-convexity statement after the Lagrangian establishes uniqueness. The "+const" wording at the energy-entropy decomposition has been corrected to `⟨-log π⟩_β` with the uniform-π reduction stated explicitly.

This is the current framing. Option (a) is the alternative the user may upgrade to if and when the multi-agent FEP literature furnishes a primary source for the specific joint construction.
