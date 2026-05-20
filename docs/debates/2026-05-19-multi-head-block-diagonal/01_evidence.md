# Evidence Pack — multi-head-block-diagonal

Neutral fact pack. Both teams work from this file.

## Manuscript references — `Attention/GL(K)_attention.tex` §5.4 (line 1703–1799)

### Section opening (`:1706`)

> "Standard transformer architectures employ multi-head attention, partitioning the `d_k`-dimensional embedding space into `H` independent heads:
>
> `μ_i = [h_i^1, h_i^2, ..., h_i^H], h_i^a ∈ ℝ^{d_head}, d_k = H × d_head.`
>
> Each head computes attention independently using separate projection matrices `(W_Q^a, W_K^a, W_V^a)`, and the results are concatenated. Under the gauge-theoretic identification, this corresponds to a block-diagonal restriction within the full `GL(d_k)` gauge group."

### Block-diagonal Ω (`:1716–1727`)

> "Multi-head attention restricts this to a block-diagonal subgroup:
>
> `Ω = diag(Ω^1, Ω^2, ..., Ω^H), Ω^a = (σ² W_K^a (W_Q^a)^T)^{-1} ∈ GL(d_head)`
>
> where each head `a` learns its own `GL(d_head)` gauge transformation. ... The rectangular shape of the standard per-head projection matrices `W_Q^a ∈ ℝ^{d_k × d_head}` can be factored via thin SVD as `W_Q^a = U_Q^a A_Q^a`, where `U_Q^a` is an isometric subspace embedding and `A_Q^a ∈ GL(d_head)` is the invertible head-space map (cf. Eq. `eq:head_space_kernel`). In the gauge-theoretic interpretation, the isometric factor selects the head subspace (analogous to the block projection), while the invertible factor carries the `GL(d_head)` gauge transport."

### Direct-product group (`:1731–1735`)

> "The full multi-head gauge group is thus the direct product:
>
> `G_multi-head = GL(d_head)^H ⊂ GL(d_k)`
>
> a `(H · d_head²)`-dimensional subgroup of the `d_k²`-dimensional `GL(d_k)`."

### Off-diagonal discard quantification (`:1781`)

> "the complement `m` has dimension `d_k² - H · d_head²`. For typical architectures (e.g., `d_k = 512, H = 8, d_head = 64`), this is `512² - 8 × 64² = 229,376` generators out of `262,144` total implying that *87.5% of the gauge algebra is discarded* by the multi-head factorization!"

### Before-vs-after head mixing (`:1797`)

> "This is a qualitatively different mechanism from the standard output projection `W_O ∈ ℝ^{d_k × d_k}` in transformers. The output projection mixes head outputs after attention. After each head has independently computed its attention-weighted aggregation `Σ_j β_{ij}^a V_j^a`, the results are concatenated and linearly mixed. Off-diagonal gauge transport, however, mixes heads before attention: the cross-head blocks `Ω_{ij}^{ab}` enter the KL divergence that determines the attention scores themselves. This makes the attention weights cross-head aware whereby head `a`'s attention pattern can depend on the content of head `b` through the transport, before any aggregation occurs at all. The full-`GL(d_k)` transport is therefore more expressive than the block-diagonal-plus-output-projection factorization, as it couples the attention computation across heads rather than only the output computation."

### Per-head temperature (`:1745–1757`)

> "each head possesses its own belief covariance `Σ^{(a)}` which controls the scale of the KL divergence and thereby the effective attention temperature for that head. In the isotropic limit `Σ^{(a)} = σ_a² I`, the KL between beliefs in head `a` reduces to `D_KL^{(a)} = (1/(2σ_a²))‖·‖²`, so the per-head covariance `σ_a²` *is* the per-head temperature. ... In standard transformers, `σ_a^{-2}` is absorbed into the learned projections `W_Q^{(a)}, W_K^{(a)}`, making the per-head temperature implicit in the weight scales `‖W_Q^{(a)}‖, ‖W_K^{(a)}‖`."

### Per-head holonomy decomposition (`:1737–1743`)

> "Under the discrete Regime II construction of Lemma `vanishing_holonomy`, the edge-relaxed transport factorises across heads: each head `a` carries its own connection field `δ_{ij}^{(a)}` acting in its own irrep block, and the holonomy decomposes as a direct sum
>
> `H_{ijk} = ⊕_{a=1}^H H_{ijk}^{(a)}`
>
> with the Wilson observable factorising additively, `W_{ijk} = Σ_{a=1}^H W_{ijk}^{(a)}`."

## Canon excerpts — `external_canon_transformers.md`

### §1 — Multi-head attention standard form (verbatim)

> "Multi-head attention
> `MHA(X) = Concat(head_1, ..., head_h) W_O`
> `head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)`
> where each head uses its own `W_Q^i, W_K^i, W_V^i ∈ ℝ^{d_model × d_k}` with `d_k = d_model / h`."

Standard form. Three independently learned rectangular `d_model × d_head` matrices per head, plus a single `W_O ∈ ℝ^{d_model × d_model}` output projection after concatenation. No SVD factorization in the construction.

### §10 pitfall #10 (verbatim)

> "Conflating learned attention with derived attention. The user's GL(K) gauge transformer has *no* learned QKV projections — attention emerges from the KL on gauge-transported beliefs. Standard transformers have learned QKV. When comparing, ensure the comparison is apples-to-apples."

### Vaswani 2017 §3.2.2 — primary source ground truth

The original transformer multi-head construction:
1. Linear-project Q, K, V via `h` separate learned rectangular matrices into `h` parallel head subspaces.
2. Compute attention independently in each head: `Attention(Q^i, K^i, V^i) = softmax(Q^i (K^i)^T / √d_k) V^i`.
3. Concatenate the `h` head outputs.
4. Linear-project the concatenation via `W_O` back to the model dimension.

The standard's "head subspace" is selected by the learned `W_Q^i, W_K^i, W_V^i` projections (rectangular). The output mixing is the single `W_O` matrix applied after concatenation.

## Direct algebraic facts

### The block-diagonal Ω from per-head W_Q, W_K

If `W_Q = block-diag(W_Q^1, ..., W_Q^H)` and `W_K = block-diag(W_K^1, ..., W_K^H)` (interpreting per-head W matrices as living in disjoint block subspaces of the d_k embedding), then:

`W_K W_Q^T = block-diag(W_K^1 (W_Q^1)^T, ..., W_K^H (W_Q^H)^T)`,

and so `Ω = (σ² W_K W_Q^T)^{-1}` is block-diagonal with `Ω^a = (σ² W_K^a (W_Q^a)^T)^{-1} ∈ GL(d_head)`. This is the manuscript's `:1716` identification, and it requires interpreting the rectangular per-head W matrices as block-diagonal embeddings into `d_model`.

### Off-diagonal dimensionality

Generator count of `gl(d_k)` is `d_k²`. Generator count of `⊕_a gl(d_head)` is `H · d_head²`. Off-diagonal complement is `d_k² - H · d_head²`. For `d_k = 512, H = 8, d_head = 64`: `262144 - 32768 = 229376`, exactly the `87.5%` figure at line 1781. The arithmetic is correct.

### Per-head holonomy product structure

For the edge-relaxed `Ω_{ij} = exp(φ_i) · exp(δ_{ij} · G) · exp(-φ_j)` in Regime II, if `φ_i, G, δ_{ij}` are all block-diagonal in the same head structure, then the matrix exponential preserves block-diagonal structure (since `exp(block-diag(X)) = block-diag(exp(X))` for any block-diagonal X). The triangle holonomy therefore factors as a direct sum across heads. The factorization is correct.

## What this evidence does NOT settle

1. **Block-diagonal interpretation of rectangular W matrices.** The manuscript at `:1716` writes `Ω^a = (σ² W_K^a (W_Q^a)^T)^{-1} ∈ GL(d_head)` with `W_K^a, W_Q^a ∈ ℝ^{d_k × d_head}` (rectangular). The inverse `(W_K^a (W_Q^a)^T)^{-1}` is taken on a `d_k × d_k` matrix that is rank-deficient (rank ≤ d_head). The inverse exists only on the rank-d_head image. The standard transformer's per-head W matrices ARE rectangular, but the gauge identification requires the thin-SVD lift documented at `:1727` to land on `GL(d_head)`. This is the same caveat as sub-claim C; the multi-head debate inherits it.

2. **"87.5% discarded" framing.** The dimensional arithmetic is correct, but whether "discarded" is the right characterization is contested. Blue: the off-diagonal generators are projected out by the multi-head architectural choice, representing a structural constraint that standard transformers impose. Red: the off-diagonal generators are not "discarded" — they were never present in the standard transformer's parameter manifold. The standard transformer's parameter space is `H × 3 × (d_model × d_head) + (d_model × d_model)` = roughly `3·d_model·d_k + d_model²` parameters (Q, K, V per head plus W_O); the gauge framework's full-`GL(d_k)` has `d_k²` parameters. These are different parameter spaces; the framework's "discard" language treats the gauge framework as the canonical reference, which is a framing choice.

3. **Before-vs-after head mixing.** The manuscript at `:1797` claims off-diagonal `Ω_{ij}^{ab}` makes head `a`'s attention pattern "cross-head aware" before aggregation, while standard `W_O` mixes only after. This is the "more expressive" claim. But standard transformers across multiple LAYERS mix heads before subsequent attention computations (a layer-`ℓ+1` head's `W_Q^a` operates on the concatenated-and-mixed output of layer `ℓ`). The "before vs after" distinction is at the single-layer level; multi-layer transformers already mix heads across layers. Whether the framework's single-layer cross-head mixing is genuinely more expressive than multi-layer cross-head mixing is an empirical question.

4. **Per-head temperature claim.** The manuscript at `:1757` says standard transformers absorb `σ_a^{-2}` into `‖W_Q^a‖, ‖W_K^a‖`, but standard transformers also have a *single* `1/√d_k` temperature applied uniformly across all heads (Vaswani §3.2.1). The per-head variation in standard transformers is implicit in the learned weight magnitudes; the framework's explicit `σ_a²` makes this variation explicit. Whether this is a *retention* of structure standard transformers already implement implicitly, or a *novel* additional degree of freedom that standard transformers don't have, is interpretive.

5. **Per-head holonomy decomposition.** The decomposition `H_{ijk} = ⊕_a H_{ijk}^{(a)}` at `:1739` is correct under the block-diagonal structure assumption. Whether standard transformers have *any* holonomy structure (they don't — they have no notion of parallel transport or curvature) is a separate question. The manuscript's claim is about the framework's structure under the multi-head restriction, not about standard transformers having holonomy.

6. **Relationship to sub-claim C verdict.** Sub-claim C's BLUE_WINS verdict accepted the value-level identification on the head-space factor `M_h^a`. The multi-head debate's sub-claims α and β are the same identification at the §5.4 level. Sub-claims γ, δ, ε are the novel content of this debate that was not adjudicated by sub-claim C.
