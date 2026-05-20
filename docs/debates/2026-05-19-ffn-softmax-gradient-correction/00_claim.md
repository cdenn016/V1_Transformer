# Claim — ffn-softmax-gradient-correction

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (`Attention/GL(K)_attention.tex` §5.3 "Feed forward Networks and Non-linear Activations" lines 1878–1950; envelope-theorem treatment at §4.7 lines 859–874; belief-dynamics restatement at line 967; supplementary at `Attention/GL(K)_supplementary.tex:1190, :1300`)
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge

## Claim

§5.3 of `Attention/GL(K)_attention.tex` derives the standard transformer FFN structure (gated linear unit with Boltzmann gate over KL energies, SiLU as the binary limit, GELU as a same-family relative, ReLU as the zero-temperature limit, FFN depth as VFE-iteration count) as the inner-loop μ-update dynamics of the VFE objective. The "softmax-gradient correction" paragraph (lines 1936–1947) labels the term `∂β_ij/∂μ_i = -(β_ij/τ)[∂E_ij/∂μ_i - ⟨∂E/∂μ_i⟩_β]` as absent from `∇F_red` by the envelope theorem but present in autograd. The composite claim being debated:

> The envelope-theorem justification for treating the GLU-with-Boltzmann-gate form as the "carrier" of the FFN nonlinearity is **exact only at the joint stationary point** `(μ*, β*(μ*))`. Inside the inner E-step loop, where `β_t = softmax(-KL(μ_t)/τ)` is recomputed at every iteration `t < n_e_steps` *before* `μ` has reached its fixed point, the covariance correction `-τ⁻¹ Cov_{β_t}(E_{ij}, ∂E_{ij}/∂μ_i)` is generically non-zero. The §5.3 derivation therefore presents an *envelope-form* FFN whose gradient field differs from the *autograd-form* belief-update dynamics actually used in the trainer by the same `Cov_β(KL, ∇KL)/τ` quantity quantified at Eq. (eq:autograd_envelope_gap, line 870). A more careful formulation of the FFN belief dynamics — one that does not assume `β` has converged before `μ` updates — must either (a) acknowledge the GLU form is the leading-order term of a series whose remainder is the covariance correction, (b) restrict the FFN derivation to the converged-`β` regime explicitly, or (c) include the correction as a structural second-order term in the FFN claim.

## Sub-claims (compound — note for the judge)

This claim has two load-bearing pieces:

1. **Sub-claim A (envelope geometry).** The envelope absence of `∂β/∂μ` from `∇F_red` requires `β = β*(μ)`, i.e. `∂F/∂β = 0`. At any intermediate inner-loop iterate `t < n_e_steps`, this holds *trivially* because `β_t := softmax(-KL_t/τ)` is constructed at each step to satisfy the row-Lagrangian KKT condition for *the current `μ_t`*, not for the converged `μ*`. The relevant question is therefore not "is β at its stationary point as a function of fixed μ?" — it is. The question is whether the *autograd-form* gradient `∇⟨E⟩_β*(μ_t)` differs from the *reduced-form* `∇F_red(μ_t)` in a way that matters off the joint fixed point.

2. **Sub-claim B (FFN derivation completeness).** The §5.3 boxed result Eq. (eq:vfe_glu, line 1903) presents the per-edge update as a GLU `e_ij · gate(e_ij)`. The "softmax-gradient correction" paragraph adds the centered covariance term as a separate non-linearity but does not integrate it into the boxed GLU form. The claim is that this is an editorial separation; the *actual* per-iteration μ-update used by the trainer combines both, and any "more careful formulation" of FFN-from-VFE-iterations would unify them.

## User context

The user invoked this debate noting:
> "'Softmax-gradient correction' may not be applicable given the envelope theorem derivation...however, it may wind up being an important term for more careful formulations?"

The user's intuition: the §5.3 derivation buys its clean GLU-with-Boltzmann-gate form via the envelope theorem, but the envelope theorem is a *boundary statement* about gradients at stationary points. A "more careful formulation" — e.g. one that derives the FFN as the gradient flow of `⟨E⟩_β*` rather than `F_red`, or one that tracks `μ` and `β` jointly off-equilibrium — would carry the covariance correction as a real second-order term.

The load-bearing question for the judge: **does the §5.3 derivation depend on a stationarity assumption that the inner E-step loop, by construction, does not satisfy at iterates `t < n_e_steps - 1`?** If so, the claim that the GLU-with-Boltzmann-gate "is" the FFN nonlinearity needs the qualifier "to leading order in `Cov_β(E, ∇E)/τ`" attached.
