# Action — subclaim-D-value-identification

**From verdict:** BLUE_WINS

## Recommended action

**Accept sub-claim D as written.** Under the explicit constant-gauge precondition `Ω_{ij} = Ω`, the value-aggregation reduction is exact: the pullout `Σ_j β_{ij} Ω μ_j = Ω · Σ_j β_{ij} μ_j` is linearity, and the absorption `V_j ≡ W_V^T μ_j` with `W_V ≡ Ω^T` is a definitional identification analogous to the `W_Q W_K^T = σ⁻²Ω⁻ᵀ` identification accepted in sub-claim C. The thin-SVD lift `W_V^a = U_V^a A_V^a` applies in parallel to V, with the isometric `U_V^a` absorbed into the multi-head output projection `W_O`.

The verdict notes that the *image* of the reduction is the parameter sub-family of standard transformers in which `W_Q W_K^T` and `W_V` share a common underlying `Ω`. The reduction is exact onto this sub-family; it does not exhaust the full standard parameter manifold. This is a feature of the reduction direction (gauge → transformer is a specialization), not a defect.

**Recommended edits at `Attention/GL(K)_attention.tex:1310`:**

Two clarifications inside §5.2.3 "Value Aggregation":

1. State explicitly that `d_k` in `W_V ∈ ℝ^{d_k × d_k}` denotes the per-head dimension `d_head` under the multi-head construction (parallel to the sub-claim C recommendation at `:1240`).

2. Add a one-sentence statement that the thin-SVD lift documented for Q/K at `:1245-1250` applies to V by parallel construction: `W_V^a = U_V^a A_V^a` with `U_V^a` an isometric subspace embedding and `A_V^a ∈ GL(d_head)` the invertible head-space map. The isometric factor is absorbed into the per-head slice of the multi-head output projection `W_O`.

**Recommended note in §5.7 summary at `:1958` or thereabouts:** mention that the reduction's image is a sub-family of the standard transformer's parameter manifold (the sub-family with common-`Ω` coupling between attention and value), not the full manifold. This is consistent with the manuscript's already-stated "constant gauge specialization" framing but makes explicit that the reduction is exact onto a sub-family, not onto the unrestricted standard transformer.

## Follow-up debates (if any)

None for sub-claim D itself. Same two related open queue items as sub-claim C apply: multi-head block-diagonal structure (§5.4) and Route 1 untied carving (§5.2.1) — both remain queued from the first debate.

## Cross-claim observations

The four-debate sweep over sub-claims A, B, C, D returned BLUE_WINS in all four, confirming the first debate's orchestrator-level "defensible" tags. The common pattern across the four verdicts: each sub-claim's *mathematical content* is correct as written; each verdict identifies an *editorial cleanup* at the manuscript level (section title, summary recap, dimension notation, or operational scope statement). None of the four require substantive mathematical revision.

The first debate's overall RED_WINS verdict on the headline compound claim ("every intermediate step mathematically valid as an exact identity") stands: sub-claim E (the approximate key-norm cancellation) is what carried the headline failure. Sub-claims A-D individually pass; the compound headline fails on E alone.
