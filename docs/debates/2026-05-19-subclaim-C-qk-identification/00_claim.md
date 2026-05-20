# Claim — subclaim-C-qk-identification

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (`Attention/GL(K)_attention.tex` §5.2.2 lines 1227–1271, with focus on the rectangular-projection paragraph at lines 1243–1250 and `\label{eq:head_space_kernel}` at line 1247; external canon on standard multi-head attention [Vaswani2017 §3.2.1])
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge

## Claim

The identification `W_Q W_K^T = σ^{-2} Ω^{-T} ∈ GL(d_k)` at `Attention/GL(K)_attention.tex:1237` recovers the standard inner-product score `Q_i^T K_j = μ_i^T Ω^{-T} μ_j` up to the joint scale `σ^{-2}` and the dimension factor `1/√d_k` from the temperature, **at the level of the invertible head-space factor `M_h^a := A_Q^a (A_K^a)^T ∈ GL(d_{\text{head}})`** defined at `:1247`. The ambient transformer kernel `W_Q^a (W_K^a)^T ∈ ℝ^{d_{\text{model}} × d_{\text{model}}}` is the low-rank lift of `M_h^a` through the isometric subspace embeddings `U_Q^a, U_K^a` (thin SVD), and this lift is consistent with the standard multi-head construction of Vaswani et al. (2017) §3.2.1.

## User context

Sub-claim C of the first debate's compound claim (`docs/debates/2026-05-19-reduction-to-standard-transformer/00_claim.md`). Marked "defensible on `M_h^a`, not on the ambient kernel" by the orchestrator; the first debate's verdict cited the rectangular-projection rank gap as one of the failures of the headline's "exact" claim.

This debate isolates the structural question from the headline-level "exact" question. Two competing readings:
- **Blue reading.** The thin-SVD decomposition `W_Q^a = U_Q^a A_Q^a` is a *structural* feature of multi-head attention — the rectangular `W_Q^a ∈ ℝ^{d_{\text{model}} × d_{\text{head}}}` IS a composition "select head subspace then transform in head space," and the gauge framework matches the head-space transformation `A_Q^a`. The lift through `U_Q^a` is the trivial part both frameworks share.
- **Red reading.** Vaswani §3.2.1 defines per-head `W_Q^a ∈ ℝ^{d_{\text{model}} × d_{\text{head}}}` as a *single* learned rectangular projection, not as a composition `U_Q^a A_Q^a`. Forcing the SVD decomposition is a post-hoc identification, and the gauge identification therefore operates on a derived object (the invertible factor of an SVD) rather than on the standard's actual learned parameter `W_Q^a`. The claim "recovers standard inner-product score" is technically correct *at the level of the bilinear form `μ_i^T M_h^a μ_j`*, but the standard form is `μ_i^T W_Q^a (W_K^a)^T μ_j` with `W_Q^a (W_K^a)^T` rank-deficient and not in `GL(d_{\text{model}})`.

The judge should determine which reading is supported by the canon ([Vaswani2017 §3.2.1] as primary source) and whether the manuscript's rectangular-projection paragraph at lines 1243–1250 correctly characterizes the relationship.
