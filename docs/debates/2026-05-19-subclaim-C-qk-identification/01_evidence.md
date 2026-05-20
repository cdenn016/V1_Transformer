# Evidence Pack — subclaim-C-qk-identification

Neutral fact pack. Both teams work from this file.

## Manuscript references

### The headline identification (`Attention/GL(K)_attention.tex:1237`)

> "We identify the product of W_Q, W_K with the gauge transport:
> `W_Q W_K^T = (1/σ²) Ω^{-T} ∈ GL(d_k).`
> Such a factorization always exists for any invertible Ω: any M ∈ GL(d_k) can be written as M = AB^T for suitable A, B ∈ GL(d_k) (e.g., via SVD: M = UΛV^T = (UΛ^{1/2})(VΛ^{1/2})^T, where invertibility of M ensures all singular values are positive, so A and B are themselves invertible)."

### The rectangular-projection disclosure (`:1243`–`:1250`)

> "This identification concerns the bilinear form `M = W_Q W_K^T` that determines the attention logits, not the individual matrices W_Q and W_K separately. In standard transformer implementations, the per-head projection matrices `W_Q^a, W_K^a ∈ ℝ^{d_{\text{model}} × d_{\text{head}}}` are rectangular (with `d_{\text{model}} ≫ d_{\text{head}}`), and neither is individually invertible in the `d_{\text{model}}`-dimensional space. The ambient-space logit kernel for head a is the low-rank matrix `W_Q^a (W_K^a)^T ∈ ℝ^{d_{\text{model}} × d_{\text{model}}}` with rank ≤ d_{\text{head}}, which is not an element of `GL(d_{\text{model}})`. To identify the gauge-theoretic `GL(d_{\text{head}})` structure, one factors each rectangular map via thin SVD as `W_Q^a = U_Q^a A_Q^a` and `W_K^a = U_K^a A_K^a`, where `U_Q^a, U_K^a ∈ ℝ^{d_{\text{model}} × d_{\text{head}}}` have orthonormal columns (isometric subspace embeddings) and `A_Q^a, A_K^a ∈ GL(d_{\text{head}})` are invertible head-space maps."

### Head-space kernel (`:1247`, label `eq:head_space_kernel`)

> "The invertible head-space kernel is then
> `M_h^a := A_Q^a (A_K^a)^T ∈ GL(d_{\text{head}})`
> and the actual transformer logit kernel is its low-rank lift `W_Q^a (W_K^a)^T = U_Q^a M_h^a (U_K^a)^T`. The gauge-theoretic identification `σ⁻²Ω^{-T} ↔ M_h^a` operates at the level of the invertible head-space factor, not the ambient low-rank kernel."

### Multi-head section (`:1709`–`:1720`)

> "Recall from Section §[glk_invariance] that, in the isotropic flat-bundle limit, standard transformer attention can be interpreted as `GL(d_k)` gauge-theoretic attention with the effective gauge transport identified as `Ω = (σ² W_K W_Q^T)^{-1} ∈ GL(d_k)`. Multi-head attention restricts this to a block-diagonal subgroup ... `Ω^a = (σ² W_K^a (W_Q^a)^T)^{-1} ∈ GL(d_{\text{head}})` ... The rectangular shape of the standard per-head projection matrices `W_Q^a ∈ ℝ^{d_k × d_{\text{head}}}` can be factored via thin SVD as `W_Q^a = U_Q^a A_Q^a`, where `U_Q^a` is an isometric subspace embedding and `A_Q^a ∈ GL(d_{\text{head}})` is the invertible head-space map (cf. Eq.~ref{eq:head_space_kernel}). In the gauge-theoretic interpretation, the isometric factor selects the head subspace (analogous to the block projection), while the invertible factor carries the `GL(d_{\text{head}})` gauge transport."

### Untied carving (Route 1, `:1142`, label `eq:gauge_qk`)

> "`Q_i = U_i^{-1} μ_i, K_j = U_j^T Σ_j^{-1} μ_j.`
> For the same token, the two projections satisfy `W_Q^(i) = U_i^{-1}` and `W_K^(i) = U_i^T Σ_i^{-1}`, with the inverse-transpose tying broken by the precision factor `Σ_i^{-1}`. The construction is therefore genuinely untied: W_Q and W_K are different functions of the per-token belief, not the same projection up to symmetry."

## Canon excerpts

### `external_canon_transformers.md` §1 — Standard form

> "`Attention(Q, K, V) = softmax(Q K^T / √d_k) V` where:
> - `Q ∈ ℝ^{N × d_k}` queries from input X via learned `W_Q ∈ ℝ^{d_{\text{model}} × d_k}`.
> - `K ∈ ℝ^{N × d_k}` keys from input X via learned `W_K ∈ ℝ^{d_{\text{model}} × d_k}`.
> - `V ∈ ℝ^{N × d_v}` values from input X via learned `W_V ∈ ℝ^{d_{\text{model}} × d_v}`.
> - √d_k is the dimension-scaling factor in the denominator..."

### `external_canon_transformers.md` §1 — Multi-head

> "MHA(X) = Concat(head_1, ..., head_h) W_O
> head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)
> where each head uses its own `W_Q^i, W_K^i, W_V^i ∈ ℝ^{d_{\text{model}} × d_k}` with `d_k = d_{\text{model}} / h`."

Vaswani's multi-head per-head W matrices are rectangular `d_{\text{model}} × d_k` (where `d_k` is the per-head dimension, equal to `d_{\text{model}}/h`).

### `external_canon_transformers.md` §10 pitfall #2 (verbatim)

> "**`W_Q W_K^T = σ⁻² Ω^{-T}`** (a specific claim in `GL(K)_attention.tex`). This is an *identification* between learned projections and gauge transformations. The agent should check whether the identification is unique (it isn't — `W_Q W_K^T` is rank-deficient in general; many Ω satisfy this)."

### `external_canon_transformers.md` §10 pitfall #10 (verbatim)

> "**Conflating learned attention with derived attention.** The user's GL(K) gauge transformer has *no* learned QKV projections — attention emerges from the KL on gauge-transported beliefs. Standard transformers have learned QKV. When comparing, ensure the comparison is apples-to-apples."

### Vaswani 2017 §3.2.1 — primary source ground truth

The original transformer construction defines per-head projections as learned rectangular matrices. There is no factorization `W_Q^a = U_Q^a A_Q^a` in [Vaswani2017]; the W matrices are atomic learnable parameters. The thin-SVD factorization is a *mathematical identity* that exists for any matrix (every matrix has an SVD), but it is not part of the standard's construction.

## What this evidence does NOT settle

1. **Equivalent forms vs structural identification.** Every matrix has an SVD. The gauge framework's `M_h^a = A_Q^a (A_K^a)^T` identification operates on the invertible factor of an SVD of the standard `W_Q^a, W_K^a`. Two readings:
   - **Blue.** Since the SVD is an identity, identifying with the invertible factor IS identifying with the rectangular matrix. The lift `W_Q^a = U_Q^a A_Q^a` is just a different parameterization of the same object.
   - **Red.** The thin-SVD factorization is a derived object (not a learned parameter). Standard transformer training updates `W_Q^a` directly, not `(U_Q^a, A_Q^a)` separately. The gauge identification therefore operates on a quantity derived post-hoc from the standard's learned parameter, not on the parameter itself. Reduction to `M_h^a` is reduction to a sub-form.

2. **Implications for training dynamics.** Even if the bilinear form `μ_i^T M_h^a μ_j` equals `μ_i^T W_Q^a (W_K^a)^T μ_j` (which it does for any thin SVD), the *gradients* through these two parameterizations are not identical. Standard transformer training learns the full `W_Q^a, W_K^a`; the gauge framework's natural-gradient descent on `M_h^a` (if implemented) would be a different optimization. The first-debate verdict noted the §5 reduction is about forward attention, not about gradients; this debate inherits that scope.

3. **Untied carving (Route 1) alternative.** The manuscript at line 1142 gives an alternative identification `Q_i = U_i^{-1} μ_i, K_j = U_j^T Σ_j^{-1} μ_j` that does NOT pass through the σ⁻²Ω⁻ᵀ form. Route 1 avoids the rectangular-projection issue entirely by using the per-token gauge frame `U_i` directly. This is the strongest blue defense in the first debate; it is also the subject of a queued follow-up debate (Route 1 alone reduces to Vaswani §3.2.1). For sub-claim C as currently formulated, Route 1 is the alternative.

4. **Non-uniqueness of the recovery.** Canon pitfall #2 notes `W_Q W_K^T = σ⁻²Ω⁻ᵀ` is rank-deficient in general; many `(σ, Ω)` pairs give the same `M_h^a`. This means the gauge identification is non-injective going from standard transformer to gauge framework, but injective (well-defined) going from gauge to standard. The forward direction (gauge → transformer) is what the claim asserts. The non-uniqueness of the recovery (transformer → gauge) is a separate concern.

5. **Whether Vaswani's `W_Q^a` is "the same" as the gauge framework's `A_Q^a`.** Operationally, `W_Q^a = U_Q^a A_Q^a` with `U_Q^a` orthonormal columns; the standard transformer learns `W_Q^a` as a single parameter, and `(U_Q^a, A_Q^a)` are derived via SVD. Are they "the same" entity? Mathematically related by a deterministic decomposition; structurally different (one is the learned parameter, the other is a derived factor).
