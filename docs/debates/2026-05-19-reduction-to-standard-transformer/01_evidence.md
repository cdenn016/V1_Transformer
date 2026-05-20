# Evidence Pack — reduction-to-standard-transformer

Neutral fact pack. Both teams work from this file.

## Manuscript references — `Attention/GL(K)_attention.tex` §5

The reduction lives in §5 "Reduction to Transformer Attention" (line 990) and §5.2 "Connection to Standard Transformers" (line 1020). Two derivation routes are presented:

### Headline framing (line 992)

> "In this section we demonstrate that standard transformer self-attention emerges as a set of limiting cases of our gauge-theoretic framework."

The opening uses "emerges as a set of limiting cases" — not "is mathematically identical to."

### The two routes (line 1022)

> "The gauge-covariant attention β_ij = softmax(-D_KL(q_i || Ω_ij q_j)/τ) derived in the preceding section is already a complete attention mechanism. Standard transformer attention is recovered as a special case along either of two equivalent routes. The first route (§5.2.1) retains Ω_ij = U_i U_j^{-1} non-trivially throughout and exposes untied query-key projections Q_i = U_i^{-1} μ_i, K_j = U_j^T Σ_j^{-1} μ_j directly from the transported KL ... The second route (§5.2.2) takes three successive limits: isotropic covariances, shared frames giving trivial pairwise transport Ω_ij = I, and a learned shared bilinear compatibility M = W_Q W_K^T acting on the means."

### Limit chain (Route 2, §5.2.2)

The route 2 derivation chains four logical moves:

| Step | Line | Operation |
|---|---|---|
| 1 | 1028–1036 | Naive `Σ → 0` gives `KL = +∞` between distinct Diracs. Remedied by "joint scaling limit where σ² remains finite but is absorbed into learned parameters." |
| 2 | 1038 | Assume isotropic `Σ_i = σ² I`. Log-det cancels; trace term becomes `d_k`. |
| 3 | 1043 | Full KL under GL(K) transport: `½[log det(ΩΩ^T) + tr((ΩΩ^T)^{-1}) - d_k] + (1/2σ²)(μ_i - Ω_{ij}μ_j)^T (ΩΩ^T)^{-1}(μ_i - Ω_{ij}μ_j)`. |
| 4 | 1049 | Mahalanobis identity: `(μ_i - Ω_{ij}μ_j)^T (Ω_{ij}Ω_{ij}^T)^{-1}(μ_i - Ω_{ij}μ_j) = ‖Ω_{ij}^{-1}μ_i - μ_j‖²`. Verified "symbolically against the direct Gaussian KL to machine precision" at line 1133. |
| 5 | 1080–1089 | Isotropic-with-general-Ω form: `D_KL = S(Ω_{ij}) + (1/2σ²)‖Ω_{ij}^{-1}μ_i - μ_j‖²` where `S(Ω) = ½[log det(ΩΩ^T) + tr((ΩΩ^T)^{-1}) - d_k] ≥ 0` with equality iff `Ω ∈ O(K)`. |
| 6 | 1115–1117 | Constant gauge: `Ω_{ij} = Ω` for all pairs. Then `S(Ω)` becomes pair-independent and "cancels under softmax." |
| 7 | 1167–1174 | Expand squared norm: `s_{ij} = (1/2σ²)‖Ω^{-1}μ_i‖² + (1/2σ²)‖μ_j‖² - (1/σ²)μ_i^T Ω^{-T}μ_j + C`. |
| 8 | 1184–1188 | i-terms `(1/2σ²)‖Ω^{-1}μ_i‖² + C` cancel under softmax over j. |
| 9 | 1196–1225 | j-bias `-(1/2σ²)‖μ_j‖²` "approximately cancels" via either (a) high-dim concentration (`‖μ_j‖² = d_k σ_0² ± O(σ_0² √d_k)`) or (b) layer normalization. |
| 10 | 1237–1241 | Identify `W_Q W_K^T = σ^{-2} Ω^{-T} ∈ GL(d_k)`. Existence of factorization argued via SVD. |
| 11 | 1243–1250 | Rectangular-projection caveat: standard `W_Q^a W_K^{aT} ∈ ℝ^{d_model × d_model}` is rank ≤ d_head, hence "not an element of GL(d_model)." Identification operates on the invertible head-space factor `M_h^a = A_Q^a A_K^{aT} ∈ GL(d_head)`. |
| 12 | 1252 | `σ⁻²` and `Ω^{-T}` "always appear together"; they are absorbed jointly into `W_Q, W_K` so the full `σ → 0` limit need not be taken. |
| 13 | 1273–1281 | Temperature `τ = √d_k` derived from the dimensional-variance argument: dot products `Q_i K_j^T` have std `O(√d_k)`, normalizing to `O(1)` gives `τ = √d_k`. |
| 14 | 1286 | Final box: `β_{ij} = softmax_j(Q_i K_j^T / √d_k)`. |
| 15 | 1316 | Value aggregation `μ̂_i = Σ_j β_{ij} V_j` with `V_j = W_V^T μ_j`, where `W_V` "absorbs the gauge transport Ω." |
| 16 | 1331 | Complete formula: `Attention(Q,K,V) = softmax(QK^T/√d_k) V`. |
| 17 | 1337–1339 | Disclosed character of the limits: "The limits are deliberately aggressive. They collapse the statistical manifold into a basic Euclidean space and absorb the gauge parameters into the learned matrices. Each limit discards a specific geometric degree of freedom and each can be independently relaxed to produce a generalization of standard attention." |

### Route 1 (untied carving) — §5.2.1, line 1119

> "Eq. (full_kl_general) is the unique inhomogeneous form taken by the gauge-covariant Gaussian KL when the covariance field is unconstrained and the frames are general GL(d_k) elements; it has been verified symbolically against the direct Gaussian KL to machine precision."  (line 1133)

The untied carving (line 1142) defines `Q_i = U_i^{-1} μ_i`, `K_j = U_j^T Σ_j^{-1} μ_j`, and produces the cross term `x_i^T k_j = Q_i^T K_j` (line 1137). The remaining terms in eq. `full_kl_general` partition into:
- j-only terms `r_j, log det Σ_j, log|det U_j|` — absorbed into `log π_{ij}` (line 1154).
- Belief-covariance coupling terms `x_i^T H_j x_i`, `tr(H_j P_i)` — described as "gauge-theoretic uncertainty corrections beyond vanilla attention; they vanish under the closure `Σ_j = U_j C U_j^T` for shared SPD C, which collapses H_j to the constant C^{-1}" (line 1155).

Route 1 therefore also requires a covariance closure to reduce.

### Supplementary acknowledgement — §B.1 (line 183)

> "For brevity we work with the entropy-suppressed surrogate `Σ_j β_{ij} D_KL(q_i || Ω_{ij}q_j)` (i.e. holding the attention distribution β fixed and treating the alignment energy at this β); the canonical free energy of the main text adds the τβ_{ij}log(β_{ij}/π_{ij}) entropy term to make the softmax form of β a stationary point."

This confirms the canonical-F vs surrogate distinction in CLAUDE.md is honestly disclosed at the supplementary opening.

### Main-text envelope-gap acknowledgement (line 866)

The autograd-vs-reduced-F gradient gap is `∇⟨E⟩_β* − ∇F_red = -τ^{-1} Cov_β*(E_{ij}, ∂E_{ij}/∂x)`. The manuscript states: "Standard transformer training follows the autograd convention (differentiating through the softmax), and we adopt the same convention in the gradient expressions and algorithm below." The reduction of §5 is about the *forward* attention rule, not the gradient; this gradient gap is therefore orthogonal to the headline reduction claim (but it could be relevant to sub-claim E about gradient-equivalence if anyone reads the reduction that way).

## Canon excerpts — `external_canon_transformers.md`

### §1 — Standard scaled dot-product attention [Vaswani2017 §3.2.1]

> `Attention(Q, K, V) = softmax(Q K^T / √d_k) V`
>
> Q ∈ ℝ^{N × d_k} via learned `W_Q ∈ ℝ^{d_model × d_k}`. K ∈ ℝ^{N × d_k} via learned `W_K ∈ ℝ^{d_model × d_k}`. V ∈ ℝ^{N × d_v} via learned `W_V ∈ ℝ^{d_model × d_v}`. √d_k is the dimension-scaling factor — "standard justification is that for unit-variance Q, K, the dot product Q_i · K_j has variance d_k, so dividing by √d_k restores unit variance and prevents the softmax from saturating."

### §2 — τ vs √d_k

> Standard: 1/√d_k from [Vaswani2017]. User's framework: `τ = κ √K` where κ is learnable. "When a manuscript writes τ = κ √K, it should label κ as the user's addition." Manuscript discloses this at line 855: "in the working implementation the temperature is factorised as τ = κ√K, with κ > 0 a learnable scalar ... the standard-transformer recovery corresponds to the special case κ = 1."

### §10 — Pitfalls explicitly enumerated by the canon

The canon's pitfall list flags exactly the issues at stake:

1. "Standard transformer is a special case." — must check derivation is tight or has hidden assumptions.
2. `W_Q W_K^T = σ^{-2} Ω^{-T}` — "The agent should check whether the identification is unique (it isn't — `W_Q W_K^T` is rank-deficient in general; many Ω satisfy this)."
3. κ as "the same as" 1/√d_k — "They are related (κ multiplies the standard scaling); they are not equivalent."
10. "Conflating learned attention with derived attention. The user's GL(K) gauge transformer has *no* learned QKV projections — attention emerges from the KL on gauge-transported beliefs. Standard transformers have learned QKV. When comparing, ensure the comparison is apples-to-apples."

### Multi-head, RoPE, masking — for context

- Multi-head: §5.4 (line 1696). Manuscript identifies `Ω = block-diag(Ω^a)`, `Ω^a = (σ² W_K^a W_Q^{aT})^{-1} ∈ GL(d_head)`. Disclosed rectangular-projection caveat (line 1720).
- RoPE: §5.5 (line 1796). Position-dependent gauge `Ω_{ij} = exp(-φ_i)^T exp(φ_j)^T` restricted to SO(2)^{d_k/2} (line 1856).
- Causal masking and ALiBi: §4.7 (lines 776–836) — recovered as non-uniform attention priors `π_j`.

## What this evidence does NOT settle

1. **Exact vs approximate.** The manuscript itself says the limits are "deliberately aggressive" (line 1337) and that the key-bias cancellation is "approximate" (line 1198). The blue side may argue that the headline boxed result (line 1331) is an exact statement *under the stated limits*; the red side may argue that "exact under approximate cancellations" is a contradiction and the reduction is therefore an approximation.

2. **Uniqueness of W_Q W_K^T = σ^{-2} Ω^{-T}.** The factorization in line 1241 is non-unique (any SVD-based pair works; many gauge transports map to the same logit kernel). Whether non-uniqueness affects the *reduction direction* (gauge → transformer is well-defined) or only the *recovery direction* (transformer → gauge has ambiguity) is open.

3. **Rectangular-projection rank gap.** The manuscript at lines 1243–1250 explicitly states `W_Q^a W_K^{aT} ∈ ℝ^{d_model × d_model}` is rank ≤ d_head, so the GL(d_k) identification operates on the invertible head-space factor M_h^a, not on the ambient kernel. Whether this counts as "exact reduction to standard attention" or as "reduction to a sub-form" is the central interpretive question.

4. **Constant-gauge cancellation.** At line 1117, `S(Ω)` becomes pair-independent under constant gauge and "cancels under softmax." This is a softmax shift-invariance fact and is exact *given* constant gauge. The status of the prior step (specializing to constant gauge while claiming the framework is gauge-equivariant) is a structural question.

5. **σ → 0 not actually taken.** Lines 1036 and 1252 explicitly say the σ → 0 limit is NOT taken; instead σ^{-2} is "absorbed into learned parameters." This means the boxed `Attention = softmax(QK^T/√d_k) V` (line 1331) holds for ANY finite σ once the absorption is done. The red side may argue this makes the "deterministic-belief" framing misleading; the blue side may argue the absorption is precisely the standard reparameterization and changes nothing operationally.

6. **τ = √d_k vs τ = κ √K.** The manuscript reduction (line 1280) derives `τ = √d_k`. The full framework uses `τ = κ √K` with learnable κ (line 855). The reduction therefore corresponds to `κ = 1`, as the manuscript explicitly notes at line 855.

7. **Value aggregation `μ̂_i = Σ_j β_{ij} Ω_{ij} μ_j` (line 1296)** has an `Ω_{ij}` factor inside the sum. The manuscript collapses this to `Σ_j β_{ij} V_j` (line 1316) by "absorbing the gauge transport Ω into the learned matrix W_V." Under constant gauge (`Ω_{ij} = Ω`) the Ω can be pulled out of the sum and absorbed; this is clean. Under pair-dependent Ω_{ij} the absorption is not literal — only the constant-gauge specialization recovers the standard `V_j = W_V^T μ_j` form. This is consistent with the explicit constant-gauge assumption.
