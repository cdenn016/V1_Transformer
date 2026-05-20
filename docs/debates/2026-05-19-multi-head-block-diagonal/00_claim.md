# Claim — multi-head-block-diagonal

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (`Attention/GL(K)_attention.tex` §5.4 lines 1703–1799 "Multi-Head Attention as Block-Diagonal GL(K) Structure"; external canon for standard multi-head attention [Vaswani2017 §3.2.2])
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge

## Claim

Standard transformer multi-head attention is **exactly** the gauge-theoretic block-diagonal restriction `G_multi-head = GL(d_head)^H ⊂ GL(d_k)` (line 1732), with the rectangular per-head projection matrices factorizing via thin SVD into an isometric subspace embedding `U_Q^a ∈ ℝ^{d_model × d_head}` and an invertible head-space map `A_Q^a ∈ GL(d_head)` (line 1727), and the off-diagonal blocks `Ω_{ij}^{ab}` of the full `GL(d_k)` transport projected out (`87.5%` of generators discarded for typical `d_k = 512, H = 8, d_head = 64`, line 1781). The standard transformer's output projection `W_O` captures only after-attention head mixing; the framework's full-`GL(d_k)` off-diagonal transport would implement before-attention head mixing, a strictly more expressive structure not present in standard multi-head (line 1797).

## Sub-claims (compound)

1. **Sub-claim α (block-diagonal structure).** Standard multi-head attention partitions the `d_k`-dimensional embedding into `H` independent heads operating on separate `d_head = d_k/H` subspaces. The gauge transport `Ω = block-diag(Ω^a)` with `Ω^a ∈ GL(d_head)` is the natural structural identification.

2. **Sub-claim β (thin-SVD lift).** Per-head projections `W_Q^a, W_K^a, W_V^a ∈ ℝ^{d_model × d_head}` factor as `W^a = U^a A^a` with `U^a` isometric (orthonormal columns) and `A^a ∈ GL(d_head)`. The gauge identification operates on `A^a`; `U^a` selects the head subspace.

3. **Sub-claim γ (off-diagonal discard).** The full `GL(d_k)` gauge algebra `gl(d_k)` decomposes as `⊕_a gl(d_head) ⊕ m` where `m` consists of off-diagonal blocks. Multi-head attention discards `m`. For typical sizes, `m` has `87.5%` of the generators of `gl(d_k)`.

4. **Sub-claim δ (before-vs-after head mixing).** Off-diagonal `Ω_{ij}^{ab}` would mix heads BEFORE the attention computation (in the KL divergence that determines β); the standard output projection `W_O` mixes heads AFTER attention. The framework claims this asymmetry makes full-`GL(d_k)` "more expressive" than standard multi-head plus `W_O`.

5. **Sub-claim ε (per-head temperature).** Per-head covariance `Σ^{(a)} = σ_a² I` IS the per-head attention temperature; in standard transformers this is implicit in `‖W_Q^a‖, ‖W_K^a‖`; the framework retains the per-head covariance structure that standard transformers absorb (line 1757).

## User context

Queued follow-up debate from the first debate's `05_action.md` and reinforced by sub-claim C's verdict. The rectangular-projection caveat at `:1727` is the same caveat as in §5.2.2 (lines 1245–1252), where sub-claim C went BLUE on value-level identification on the head-space factor. Sub-claim α and β here inherit that structure. The novel content of this debate is sub-claims γ, δ, ε (the off-diagonal-discard, before-vs-after head mixing, and per-head temperature claims), which are specific to the multi-head section.

The load-bearing question for the judge: is the multi-head correspondence a *direct architectural identification* (standard multi-head IS block-diagonal gauge) or a *structural reading* requiring post-hoc SVD factorization that the standard does not provide?
