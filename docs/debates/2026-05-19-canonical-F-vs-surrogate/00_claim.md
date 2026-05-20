# Claim — canonical-F-vs-surrogate

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (`Attention/GL(K)_attention.tex` §4.7 lines 838–874 with the envelope-gap derivation; supplementary `Attention/GL(K)_supplementary.tex` §B.1 line 181–195 with the "entropy-suppressed surrogate" admission; the autograd-vs-reduced-F distinction at line 868–874)
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge

## Claim

The canonical free energy `F_red = Σ_i D_KL(q_i ‖ p_i) - τ Σ_i log Z_i - E_q[log p(o|k)]` with the `τβ_{ij} log(β_{ij}/π_{ij})` attention-entropy term and the entropy-suppressed surrogate `F_surr = Σ_j β_{ij} E_{ij}` (β held fixed) have **non-equivalent gradients** off the joint stationary point. The gradient gap, derived at `Attention/GL(K)_attention.tex:868–872`, is

```
∇_x ⟨E⟩_{β*} − ∇_x F_red = -τ⁻¹ Cov_{β*}(E_{ij}, ∂E_{ij}/∂x)
```

with `x ∈ {μ_i, Σ_i, φ_i}`. The covariance vanishes only at the joint stationary point of F_red in x or in the high-temperature limit τ → ∞, but not generically. The manuscript at `:874` adopts the autograd convention (differentiating through the softmax, computing the surrogate gradient) for the framework's iterative belief updates, and the supplementary §B.1 at `:183` confirms the surrogate is the working form. **The framework's E-step gradient flow therefore descends on the surrogate, not on the canonical F it nominally defines, with the two objectives having coincident stationary points only at joint equilibrium or in the τ → ∞ limit.**

## User context

Queued follow-up debate from the first debate's `05_action.md`. The user requested it after seeing the four sub-claim verdicts return BLUE_WINS. The debate tests:

1. **Mathematical correctness** of the envelope-gap formula at lines 870–872 (the `-τ⁻¹ Cov_β*` form).
2. **Whether the autograd-convention adoption is mathematically clean** (blue: it matches standard transformer training; red: the framework's stationary-point analysis uses F's gradient but the implementation uses the surrogate's, so the analysis does not apply to the implementation off-equilibrium).
3. **Whether the supplementary §B.1 retreat to the surrogate** is a benign brevity choice (blue: the Σ_i gradient is identical under both forms because the attention entropy does not depend on Σ_i) or a load-bearing admission that the working form is the surrogate.

## Sub-claims

The headline is compound. Load-bearing sub-claims:

1. **Sub-claim α (gradient-gap formula).** The covariance form `∇_x ⟨E⟩_{β*} − ∇_x F_red = -τ⁻¹ Cov_{β*}(E_{ij}, ∂E_{ij}/∂x)` is the correct derivative of the difference. Verifiable by direct calculation.
2. **Sub-claim β (vanishing locus).** The gap vanishes only at (i) joint stationary points where `∇_x F_red = 0` (and the cov vanishes for any x-independent reason) or (ii) in the τ → ∞ limit (uniform β). It does not vanish generically off-equilibrium.
3. **Sub-claim γ (autograd convention adoption).** The manuscript declares at line 874 that "the gradient expressions below are gradients of the attention-weighted energy `Σ β E_{ij}`, not of the reduced free energy F_red." This is an explicit choice of the surrogate gradient as the implementation form.
4. **Sub-claim δ (Σ_i gradient identity under both forms).** Per supplementary §B.1 at line 183, the Σ_i gradient is identical under canonical F and surrogate because the attention entropy term does not depend on Σ_i. The gap is in the μ_i and φ_i gradients only.
5. **Sub-claim ε (training-consequence).** The E-step gradient flow under the autograd / surrogate convention does not descend on F_red off-equilibrium. The manuscript's stationary-point analysis (e.g., the softmax-β derivation at §4.6) uses F's gradient and applies only at the joint stationary point.

Sub-claims α, β, γ, δ are factual (derivable from the manuscript or by direct calculation). Sub-claim ε is the load-bearing one for the debate: does the framework's gradient flow do what the framework claims it does?
