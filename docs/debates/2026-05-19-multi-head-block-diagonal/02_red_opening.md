# Red Opening — multi-head-block-diagonal

## Steelman (opposing position)

When the gauge framework is taken as the canonical reference geometry of attention, standard multi-head attention is the architectural restriction obtained by projecting the full `GL(d_k)` transport onto its block-diagonal subgroup `GL(d_head)^H`; the rectangular per-head `W_Q^a, W_K^a` admit a thin-SVD factorization whose isometric factor selects the head subspace and whose invertible factor `A^a ∈ GL(d_head)` carries the gauge transport, the off-diagonal generators of `gl(d_k)` are projected out (`d_k² − H·d_head²` of them, 87.5% at typical sizes), the per-head covariance `σ_a²` is retained where standard transformers absorb it into weight norms, and full-`GL(d_k)` transport with non-zero off-diagonal blocks `Ω_{ij}^{ab}` would make attention scores cross-head aware before aggregation in a way the post-attention `W_O` cannot reproduce within a single layer.

## Position

Sub-claims α (block-diagonal structure) and β (thin-SVD lift) are inherited from sub-claim C's BLUE_WINS verdict; I concede them at the value level on the head-space factor. The novel content — γ ("87.5% discarded"), δ ("more expressive" via before-vs-after head mixing), ε (per-head temperature as a retention) — is **false as stated**. Each rests on a framing in which the gauge framework's parameterization is treated as primary and the standard transformer's parameterization is treated as a restriction of it. That framing is not a theorem; it is a choice, and reversing it produces equally defensible statements that the manuscript does not acknowledge. Specifically:

1. The "87.5% discarded" language presupposes that `gl(d_k)` is the canonical ambient algebra. The standard transformer's parameter manifold is `3·H·d_model·d_head + d_model²` (Q, K, V per head plus `W_O`), not `d_k²`. The off-diagonal generators are not "discarded" — they are not in the standard's parameter space to begin with.

2. The single-layer "before vs after" expressiveness claim ignores layered alternation: layer `ℓ+1`'s `W_Q^a, W_K^a` operate on the `W_O`-mixed output of layer `ℓ`, so cross-head awareness enters subsequent attention computations across layers. The multi-layer standard transformer is not block-diagonal in the relevant sense.

3. The per-head `σ_a²` is presented as a structure standard transformers "absorb." Vaswani §3.2.1 specifies a **single uniform `1/√d_k` factor** shared across all heads; per-head variation in standard transformers is purely a consequence of independent weight learning, not a per-head explicit temperature parameter. The framework's `κ_a` is an additional parameter slot, not a retention.

4. The block-diagonal `Ω = (σ² W_K W_Q^T)^{-1}` identification at line 1716 requires interpreting the rectangular per-head matrices as living in **disjoint block subspaces** of `d_model`. Vaswani §3.2.2 does not impose this; the per-head `W_Q^a, W_K^a, W_V^a ∈ ℝ^{d_model × d_head}` are independently learned rectangular projections whose column spaces in `d_model` are unconstrained and generically overlap.

## Evidence

**Citation 1 — Standard multi-head construction (primary source).** Vaswani 2017 §3.2.2, reproduced in `external_canon_transformers.md` §1:

> `MHA(X) = Concat(head_1, ..., head_h) W_O`
> `head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)`
> where each head uses its own `W_Q^i, W_K^i, W_V^i ∈ ℝ^{d_model × d_k}` with `d_k = d_model / h`.

The standard construction does not specify any embedding of the per-head matrices into a block-diagonal `d_model × d_model` matrix. The per-head rectangular projections are learned independently with no orthogonality constraint between heads' column spaces. The block-diagonal structure required to make `(σ² W_K W_Q^T)^{-1}` factor as `block-diag((σ² W_K^a (W_Q^a)^T)^{-1})` is an additional structural reading imposed by the framework. The evidence pack §"Direct algebraic facts" makes the requirement explicit: "*interpreting the rectangular per-head W matrices as living in disjoint block subspaces of the d_k embedding*." That interpretation is not in Vaswani.

**Citation 2 — Standard `1/√d_k` is uniform across heads (primary source).** Vaswani 2017 §3.2.1 and `external_canon_transformers.md` §1:

> `Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V`
> "`√d_k` is the dimension-scaling factor in the denominator; the standard justification is that for unit-variance Q, K, the dot product `Q_i · K_j` has variance `d_k`, so dividing by `√d_k` restores unit variance."

The denominator `√d_k` is dimensional, fixed at construction time, and identical for every head. The standard transformer has **no per-head explicit temperature parameter**. Per-head variation in attention sharpness arises implicitly from `‖W_Q^a‖, ‖W_K^a‖` because Q, K are learned outputs; it is not a separate degree of freedom that the framework "retains." The framework's `κ_a` at line 1751 is an additional learnable scalar that the standard transformer does not have. Calling `κ_a` a "retention of structure standard transformers absorb" reverses the relationship: the framework adds an explicit per-head parameter; the standard absorbs nothing because there is nothing per-head to absorb.

**Citation 3 — Canon pitfall #10, apples-to-apples.** `external_canon_transformers.md` §10 pitfall 10 (verbatim):

> "Conflating learned attention with derived attention. The user's GL(K) gauge transformer has no learned QKV projections — attention emerges from the KL on gauge-transported beliefs. Standard transformers have learned QKV. When comparing, ensure the comparison is apples-to-apples."

The "87.5% discarded" framing at `Attention/GL(K)_attention.tex:1781` violates this pitfall. The standard transformer's parameter manifold is, as the evidence pack §"What this evidence does NOT settle" item 2 states, "`H × 3 × (d_model × d_head) + (d_model × d_model)` parameters." For `d_model = d_k = 512, H = 8`: that is `8 · 3 · 512 · 64 + 512² = 786,432 + 262,144 = 1,048,576` parameters. The framework's full `GL(d_k)` (per edge) has `d_k² = 262,144` parameters. These are **different parameter manifolds with different dimensions**. The claim "87.5% of the gauge algebra is discarded" is true only on the framework's own algebra; the framing implicit in the exclamation mark — that the standard is a lossy restriction of the framework — is a choice, and the reverse framing (the framework introduces a different parameterization that does not contain the standard's `W_O` and `V`-projection structure) is equally defensible.

**Citation 4 — Multi-layer cross-head mixing in standard transformers.** Vaswani 2017 §3.1 (model architecture): the encoder and decoder are stacks of `N = 6` identical layers; each layer's multi-head sub-layer takes as input the output of the previous layer. The output of layer `ℓ` is `LayerNorm(x + MHA_ℓ(x))` where `MHA_ℓ` ends in the `W_O^{(ℓ)}` mixing. Layer `ℓ+1`'s `W_Q^a` then acts on a `d_model`-vector that has already been mixed across heads by `W_O^{(ℓ)}`. Layer `ℓ+1` head `a`'s attention scores therefore depend on what was in layer `ℓ` head `b` before layer `ℓ+1`'s aggregation. The "more expressive" claim at line 1797 — that off-diagonal `Ω_{ij}^{ab}` makes attention "cross-head aware before aggregation" in a way that `W_O` cannot — collapses outside the single-layer setting. A multi-layer stack with `W_O` mixing achieves cross-head awareness in the next layer's attention; the framework's full-`GL(d_k)` does it within one layer. That is an architectural difference, not a strict expressiveness ordering. The evidence pack §"What this evidence does NOT settle" item 3 explicitly flags this.

**Citation 5 — Manuscript line that overstates the per-head temperature claim.** `Attention/GL(K)_attention.tex:1757`:

> "In standard transformers, `σ_a^{-2}` is absorbed into the learned projections `W_Q^{(a)}, W_K^{(a)}`, making the per-head temperature implicit in the weight scales `‖W_Q^{(a)}‖, ‖W_K^{(a)}‖`. ... The gauge framework retains the per-head covariance structure that standard transformers absorb."

The verb "retain" presupposes a per-head temperature exists in the standard form to be retained. The standard form has one scalar `1/√d_k` shared across heads (Vaswani §3.2.1). The variation across heads arises from learning, not from an explicit per-head parameter. The framework's `κ_a` (line 1751) and `Σ^{(a)}` (line 1745) are explicit per-head parameters added by the framework; this is novel structure, not retained structure.

## Falsification conditions

This position is wrong if any of the following hold:

1. **(γ rebuttal target)** Blue produces a passage in Vaswani 2017 or in standard transformer pedagogy that frames `gl(d_k)` (the full `d_k × d_k` matrix algebra) as the canonical parameter manifold of multi-head attention, such that the block-diagonal `H · d_head²` count is recognised as a restriction within the standard's own construction rather than within the framework's reconstruction. Absent such a passage, "87.5% discarded" is framework-internal language and the manuscript should label it as such.

2. **(δ rebuttal target)** Blue establishes that the full-`GL(d_k)` single-layer transport with off-diagonal `Ω_{ij}^{ab}` is strictly more expressive than any multi-layer block-diagonal-plus-`W_O` stack at the function-class level — that is, that there exists a family of attention patterns realizable in single-layer full-`GL(d_k)` that no multi-layer stack of standard MHA with `W_O` can realize. Without such a separation, the "more expressive" claim at line 1797 must be qualified as single-layer expressiveness.

3. **(ε rebuttal target)** Blue points to a per-head explicit temperature parameter in standard transformer constructions — for example, a per-head learnable temperature `τ_a` distinct from `1/√d_k` — that predates the framework and is what the framework "retains." Without this, `κ_a` and `Σ^{(a)}` are framework-novel additions and the verb "retain" at line 1757 mischaracterizes the relationship.

4. **(block-diagonal embedding)** Blue produces a primary source (Vaswani 2017 or canonical transformer reference) that specifies the per-head rectangular `W_Q^a, W_K^a, W_V^a` as living in disjoint block subspaces of `d_model`, such that `W_K W_Q^T` is naturally block-diagonal in `ℝ^{d_k × d_k}` without thin-SVD lifting. Without that, the identification at line 1716 is a structural reading, and the manuscript should label it as a chosen factorization rather than a direct correspondence.

If none of (1)–(4) is established, the multi-head section's novel claims γ, δ, ε must be qualified: the framework provides a reframing of standard multi-head attention as a structured restriction within its own algebra, but the rhetorical claims of "discard," "more expressive," and "retention" overstate the asymmetry between the two parameterizations.
