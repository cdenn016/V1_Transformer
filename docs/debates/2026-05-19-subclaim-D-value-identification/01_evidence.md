# Evidence Pack — subclaim-D-value-identification

Neutral fact pack. Both teams work from this file.

## Manuscript references

### Mixture-aggregation (`Attention/GL(K)_attention.tex:1296`, label `eq:glk_mixture_aggregation`)

> "The mixture-of-sources generative model (Section §[mixture_derivation]) specifies that the component distribution for source j in agent i's frame has mean Ω_{ij}μ_j. The posterior mean under mixture responsibilities β_{ij} is therefore:
> `μ̂_i = Σ_j β_{ij} Ω_{ij} μ_j`
> which is the standard mixture-of-Gaussians expectation."

### Distinction between attention geometry and value geometry (`:1302`)

> "A separate geometric object arises in the attention weight computation, where the Mahalanobis identity introduces Ω⁻ᵀ as the natural transform for the dual (covector) structure of the inner product. Specifically, Ω⁻ᵀ appears in the logit μ_i^T Ω⁻ᵀ μ_j because the KL divergence measures distance in the transported covariance metric (ΩΩ^T)⁻¹, and Ω⁻ᵀ is the correct transform for covectors (dual vectors) under a linear map. To be clear, this dual transform belongs to the attention scoring geometry, not to the value aggregation."

### Value projection (`:1308`)

> "Next, we define the value projection:
> `V_j ≡ W_V^T μ_j, W_V ∈ ℝ^{d_k × d_k},`
> where we absorb the gauge transport Ω into the learned matrix W_V."

The value projection is defined with `W_V ∈ ℝ^{d_k × d_k}` (square, full-dimensional).

### Boxed reduction (`:1316`)

> "`μ̂_i = Σ_j β_{ij} V_j`,
> identical to the standard transformer attention update `z_i = Σ_j α_{ij} V_j`."

### Attention vs value gauge factorization (`:1863`, in RoPE section)

> "An important distinction is that RoPE applies the position-dependent gauge only to the attention computation (queries and keys) and not to the value aggregation. In our framework, the full gauge transport Ω_{ij} mediates both the attention score `D_KL(q_i ‖ Ω_{ij} q_j)` and the value aggregation `μ̂_i = Σ_j β_{ij} Ω_{ij} μ_j`. The asymmetry in RoPE with position-dependent attention but position-independent values corresponds to factoring the gauge transport into an attention gauge and a value gauge that need not coincide. This decomposition is supported (but not required) by the full framework."

## Canon excerpts — `external_canon_transformers.md`

### §1 — Standard form (verbatim)

> "Attention(Q, K, V) = softmax(Q K^T / √d_k) V
> where ... `V ∈ ℝ^{N × d_v}` values from input X via learned `W_V ∈ ℝ^{d_{\text{model}} × d_v}`."

Standard `W_V` is **rectangular** `d_{\text{model}} × d_v`, where `d_v` is the per-head value dimension (typically `d_v = d_{\text{model}}/h`).

### §1 — Multi-head

> "MHA(X) = Concat(head_1, ..., head_h) W_O
> head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)
> where each head uses its own `W_Q^i, W_K^i, W_V^i ∈ ℝ^{d_{\text{model}} × d_k}`..."

Standard per-head `W_V^i` is rectangular.

### §1 — KQV interpretation

> "- Q ↔ 'what am I looking for'
> - K ↔ 'what do I offer'
> - V ↔ 'what I contribute if matched'
> - Softmax(QK^T/√d_k) ↔ assignment matrix (rows sum to 1)."

The KQV interpretation has Q, K, V as parallel objects, each via its own learned projection.

## What this evidence does NOT settle

1. **Square vs rectangular `W_V`.** The manuscript writes `W_V ∈ ℝ^{d_k × d_k}` (square). Standard Vaswani is `W_V ∈ ℝ^{d_{\text{model}} × d_v}` (rectangular). Two readings:
   - **Blue.** The manuscript is at the per-head level: `d_k` here means the head dimension (which is `d_v` in Vaswani notation), and `d_k × d_k` matches the per-head invertible structure paralleling the `M_h^a` identification for Q K^T. Or: `W_V` is the FULL `d_{\text{model}} × d_{\text{model}}` aggregation kernel including the output projection `W_O`; in that case it is also square. Either reading recovers standard attention.
   - **Red.** The manuscript's notation `W_V ∈ ℝ^{d_k × d_k}` does not match standard `W_V ∈ ℝ^{d_{\text{model}} × d_v}`. If `d_k` is the per-head dimension, the framework's per-head value transport is `GL(d_{\text{head}})`; the lift to `d_{\text{model}}` requires the same thin-SVD argument as for Q K^T (issue from sub-claim C). The reduction is therefore to a head-space sub-factor of standard value aggregation, not to the standard's rectangular per-head `W_V`.

2. **Constant-gauge pullout.** Under `Ω_{ij} = Ω`, `Σ_j β_{ij} Ω μ_j = Ω · Σ_j β_{ij} μ_j`. This is exact by linearity. The pulled-out `Ω` is then identified with `W_V^T`. Whether this identification matches the standard transformer's `W_V` depends on (1) above. Under pair-dependent `Ω_{ij}`, the pullout fails (`Σ_j β_{ij} Ω_{ij} μ_j ≠ Ω · Σ_j β_{ij} μ_j` because `Ω_{ij}` depends on j). The constant-gauge specialization is therefore necessary for the absorption.

3. **Attention gauge vs value gauge collapse.** Manuscript line 1863 says the framework supports factoring `Ω` into separate attention and value gauges that need not coincide; the §5.2.3 reduction collapses them to the same `Ω`. Standard transformers do not have a notion of "attention gauge" vs "value gauge" — they have independently learned `W_Q, W_K, W_V`. The gauge framework's collapse of attention-gauge and value-gauge to the same `Ω` is a stronger constraint than standard transformers impose. This is consistent with the *reduction direction* (gauge has more structure, standard has less; collapse loses structure), but it is a *pre-imposed coupling* of attention and value that the standard does not have. Whether this is a structural fact (the gauge framework reveals a hidden coupling in standard transformers) or a coincidence of the specific reduction taken (other reductions might give different `W_Q, W_K, W_V`) is open.

4. **Implications for "exact reduction."** Sub-claim D claims the reduction is "exact" under the constant-gauge pullout plus the W_V absorption. The pullout is exact (linearity). The absorption is by definition (`W_V^T` is defined to make the identification hold). The composition is therefore exact under both reductions. The remaining question is whether the resulting `Σ β_{ij} V_j` is "the standard transformer aggregation" — which depends on the square/rectangular question (1) above.

5. **No empirical residue.** Even if the reduction is mathematically exact, it has no empirical consequence: the gauge framework can be replaced by standard transformer attention with appropriate `W_V`, and standard transformers can be claimed to implement gauge value aggregation in disguise. The debate is over which framing is more accurate, not over which produces different results.

6. **Independence from sub-claim C.** Sub-claim D inherits the rectangular-projection issue from sub-claim C (the standard's `W_V` is rectangular, the framework's `W_V` is square in the manuscript's notation). If sub-claim C goes RED on the rectangular issue, sub-claim D is likely to go the same way for the same reason. The two are not logically independent.
