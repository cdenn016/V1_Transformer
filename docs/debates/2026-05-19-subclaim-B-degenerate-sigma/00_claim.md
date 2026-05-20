# Claim — subclaim-B-degenerate-sigma

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (`Attention/GL(K)_attention.tex` §5.2 Deterministic Beliefs paragraph at line 1024 onwards, with focus on lines 1036 and 1252; external canon on KL between Gaussians and the absolute-continuity requirement)
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge

## Claim

The "degenerate-Σ limit" used in `Attention/GL(K)_attention.tex` §5.2.2 to reduce the Gaussian KL to a quadratic Mahalanobis form is **not** an analytic `Σ → 0` limit but a **literal change of parameterization** in which `σ^{-2}` and `Ω^{-T}` always appear together as `σ^{-2}Ω^{-T}` and are jointly absorbed into the learned projections `W_Q W_K^T`. Under this reparameterization, the reduction is exact for any positive finite `σ` and the manuscript's claim at line 1252 ("the full limit need not be taken") correctly characterizes the operation.

## User context

Sub-claim B of the first debate's compound claim (`docs/debates/2026-05-19-reduction-to-standard-transformer/00_claim.md`). Marked "defensible" by the orchestrator; first debate's verdict explicitly endorsed this reading at the manuscript's line 1252.

Load-bearing question for the judge: does "degenerate-Σ limit" as a section title (line 1024 paragraph header "Deterministic Beliefs via Scaled Limit") match what is mathematically done in the derivation? The manuscript at line 1036 admits the literal `Σ → 0` limit is ill-defined (`KL = +∞` between distinct Diracs) and switches to a "joint scaling limit where σ² remains finite but is absorbed into learned parameters." This is a reparameterization, not a limit in the analytic sense. Is the naming a stylistic infelicity, or does it misrepresent the operation in a way that affects downstream claims (e.g., "deterministic beliefs" being read as actually deterministic rather than as a finite-σ reparameterization)?
