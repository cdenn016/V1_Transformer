# Claim — route-1-untied-carving

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (`Attention/GL(K)_attention.tex` §5.2.1 lines 1122–1161 "Untied Query-Key Carving from Per-Token Frames"; external canon for standard scaled dot-product attention [Vaswani2017 §3.2.1])
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge

## Claim

Route 1 (untied carving from per-token frames) at `Attention/GL(K)_attention.tex` §5.2.1 reduces **exactly** to Vaswani et al. (2017) §3.2.1 scaled dot-product attention under the covariance closure `Σ_j = U_j C U_j^T` for shared SPD `C`. The carving `Q_i = C^{-1/2} U_i^{-1} μ_i`, `K_j = C^{-1/2} U_j^{-1} μ_j` (line 1158) directly recovers the standard `Q^T K` inner product, with the j-only terms (`r_j`, `log det Σ_j`, `log|det U_j|`) absorbed into the standard prior slot `log π_{ij}`.

## Sub-claims (compound)

1. **Sub-claim α (KL decomposition exactness).** The transported Gaussian KL admits the exact decomposition at Eq. `eq:full_kl_general` (line 1129), verified symbolically to machine precision (line 1136).

2. **Sub-claim β (cross-term carving).** The cross term `-2 x_i^T k_j` factors as `Q_i^T K_j` with `Q_i = U_i^{-1} μ_i, K_j = U_j^T Σ_j^{-1} μ_j` (line 1145). This is the untied carving directly from the gauge-covariant KL, with `W_Q^(i) = U_i^{-1}` and `W_K^(i) = U_i^T Σ_i^{-1}` differing by the precision factor (line 1148).

3. **Sub-claim γ (j-only absorption).** Terms depending on j alone (`r_j`, `log det Σ_j`, `log|det U_j|`) are absorbed into `log π_{ij}` (line 1158). This is mathematically valid (softmax shift-invariance: `softmax_j(x_j + c_j) = π_j' · exp(x_j) / Σ_k π_k' · exp(x_k)` with `π_j' = exp(c_j)`).

4. **Sub-claim δ (covariance closure).** Under the closure `Σ_j = U_j C U_j^T` for shared SPD `C`, the belief-covariance coupling terms `x_i^T H_j x_i, tr(H_j P_i)` "vanish" by collapsing `H_j = U_j^T Σ_j^{-1} U_j` to the constant `C^{-1}` (line 1158). The carving simplifies to the symmetric form `Q_i = C^{-1/2} U_i^{-1} μ_i, K_j = C^{-1/2} U_j^{-1} μ_j` with shared bilinear `C^{-1}`.

5. **Sub-claim ε (Vaswani §3.2.1 recovery).** The resulting form `μ_i^T M_{ij} μ_j` factorizes through per-token vectors exactly as in Vaswani §3.2.1, with the additive prior `log π_{ij}` matching the standard's prior/masking slot.

## User context

Final open queue item from the first debate's compound headline. The first debate's verdict noted Route 1 as "blue's strongest unrefuted move" — but with the rectangular-projection caveat and the manuscript's own "approximate" language at line 1198 (which was Route 2's failure point), it remained open whether Route 1 has its own analogous failure point.

The load-bearing question for the judge: under the closure `Σ_j = U_j C U_j^T`, does the j-only term `r_j = μ_j^T Σ_j^{-1} μ_j` evaluate to a *constant* (uniform π_{ij}, matching Vaswani §3.2.1 exactly) or to a *content-dependent function of the j-th token* (e.g., `‖K_j‖²` under the symmetric carving), which would make the recovery to "Vaswani + content-dependent additive prior" rather than to Vaswani §3.2.1 itself?

This is the same key-norm-bias question that drove the first debate's RED_WINS on Route 2 — relocated from "approximate cancellation under LN" to "exact absorption into log π_{ij}." Whether the relocation makes the recovery exact (blue) or simply re-labels the problem (red) is the central question.
