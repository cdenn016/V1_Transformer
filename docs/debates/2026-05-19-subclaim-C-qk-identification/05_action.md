# Action — subclaim-C-qk-identification

**From verdict:** BLUE_WINS

## Recommended action

**Accept sub-claim C as written.** The identification operates at the level of the invertible head-space factor `M_h^a := A_Q^a (A_K^a)^T ∈ GL(d_head)`, which is explicitly stated at `Attention/GL(K)_attention.tex:1252`. The value-level scalar logit produced by `μ_i^T M_h^a μ_j` matches the standard transformer's attention logit on the head subspace. Both sides agree the bilinear-form value equality holds exactly via the SVD identity.

**Notation cleanup recommended:** the verdict notes a dimension-notation inconsistency. Line `:1240` writes `W_Q W_K^T = σ⁻²Ω⁻ᵀ ∈ GL(d_k)` while the descent paragraph at `:1247` uses `d_head`. Both readings are present in the text but the relationship between `d_k` and `d_head` should be made explicit at first use.

**Recommended edit at `Attention/GL(K)_attention.tex:1240`:**

State whether `d_k` at this line denotes the per-head dimension (`d_head`) or the full embedding dimension (`d_model`). The §5.4 multi-head section makes clear it is the head dimension under the multi-head construction; under single-head attention the two coincide. A one-line clarification at `:1240` ("here `d_k` denotes the per-head dimension `d_head` under the multi-head construction of §5.4") removes the ambiguity.

**Recommended edit at `Attention/GL(K)_attention.tex:1245-1250`:**

The rectangular-projection paragraph already disclosed the descent to `M_h^a`. Adding a one-sentence summary at the close — "The gauge-theoretic identification is therefore value-level on the head subspace; it does not assert parameter-level identity between the gauge framework's `(σ, Ω)` and the standard's atomically-learned `W_Q^a, W_K^a`" — would close the parameterization-vs-operation ambiguity flagged by red.

## Follow-up debates (if any)

None for sub-claim C itself. The two related open queue items remain:
1. **Multi-head = block-diagonal GL(K)** (§5.4) — the rectangular-projection caveat at `:1720` intersects this debate; a focused sub-debate would test whether the thin-SVD lift is a structural correspondence or a chosen factorization at the multi-head level.
2. **Route 1 untied carving alone reduces to Vaswani §3.2.1** — the manuscript's alternative route at §5.2.1; remains open from the first debate.
