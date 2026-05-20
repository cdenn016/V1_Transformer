# Claim — subclaim-A-flat-bundle

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (`Attention/GL(K)_attention.tex` §5 lines 990–1342, with focus on the constant-gauge specialization at line 1115; `Attention/GL(K)_supplementary.tex` §A on principal-bundle structure; external canon for gauge equivariance and the trivial-bundle limit)
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge

## Claim

Setting `Ω_ij = Ω` for all agent pairs (the constant-gauge / flat-bundle specialization at `Attention/GL(K)_attention.tex:1115`, then further taking `Ω = I` or absorbing `Ω^{-T}` into learned `W_Q W_K^T`) is a **well-defined limit** of the GL(K) gauge-covariant attention framework that **does not destroy** the upstream gauge-equivariance arguments used in the manuscript (the GL(K) invariance theorem at `\ref{thm:glk_invariance}`, the covariance transport sandwich `Σ → Ω Σ Ω^T`, and the natural-gradient preconditioning on `\mathfrak{gl}(K)`).

## User context

Sub-claim A of the first debate's compound claim (`docs/debates/2026-05-19-reduction-to-standard-transformer/00_claim.md`). Marked "defensible" by the orchestrator without formal adjudication. The user requested explicit debate.

The load-bearing distinction is between two readings:
- (i) The flat-bundle limit is a *specialization* of the general framework: a particular choice of `Ω ∈ GL(K)` (namely the identity, or a constant element), within a framework that is generally gauge-equivariant. Specialization does not destroy the upstream argument; it instantiates it at a particular point.
- (ii) The flat-bundle limit is a *gauge fixing*: by fixing `Ω = I` you have chosen a particular gauge, and gauge-equivariance arguments (which require invariance under ALL `Ω ∈ GL(K)`) no longer apply to the fixed-gauge object. The fixed-gauge object is just an ordinary Euclidean construction.

The judge should determine whether the manuscript's framework actually preserves gauge equivariance after taking the flat-bundle limit, or whether the limit is a gauge fixing whose downstream consequences are not gauge-equivariant.
