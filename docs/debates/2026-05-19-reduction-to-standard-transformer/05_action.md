# Action — reduction-to-standard-transformer

**From verdict:** RED_WINS

## Recommended action

1. **Soften the headline claim to match what the manuscript actually supports.** The defensible statement is:

   > "Standard scaled dot-product attention emerges as a limiting case of the GL(K) framework under three aggressive limits (isotropic covariance, constant gauge, σ⁻²Ω⁻ᵀ absorbed into learned projections), with one approximate cancellation conditioned on layer normalization or high-dimensional concentration, and with the GL(d_head) identification operating on the invertible head-space sub-factor of standard multi-head attention."

   This is consistent with the manuscript's own framing at line 992 ("emerges as a set of limiting cases") and line 1337 ("the limits are deliberately aggressive"), and avoids the compound "exactly (not approximately, every intermediate step mathematically valid)" that the verdict found unsupported.

2. **Manuscript edit at `Attention/GL(K)_attention.tex` line 1331 (boxed result).** Precede the boxed `Attention(Q,K,V) = softmax(QK^T/√d_k) V` with an explicit preconditions sentence:

   > "Under constant gauge, isotropic covariance with σ⁻²Ω⁻ᵀ absorbed into `W_Q W_K`, and key-norm constancy supplied by layer normalization or high-dimensional concentration, the gauge attention rule reduces to:"

   This makes clear that the equality holds modulo named preconditions rather than as an unconditional identity, and removes the gap between the boxed result and the disclosures at lines 1198, 1243–1250, 1252, and 1337.

3. **Sub-claim status from `00_claim.md`** (compound decomposition):
   - **Sub-claim A** (flat-bundle limit `Ω_ij = I` is well-defined and preserves gauge-equivariance upstream): defensible.
   - **Sub-claim B** (degenerate Σ as a literal change of parameterization, not as the analytic `σ → 0` limit): defensible — line 1252 makes this explicit.
   - **Sub-claim C** (`Q K^T` identification recovers standard inner-product score up to sign and `1/√K`): defensible **on the invertible head-space sub-factor `M_h^a`**; not defensible on the rank-deficient ambient kernel without thin-SVD lift through isometric `U_Q^a, U_K^a`.
   - **Sub-claim D** (value identification `V_j = W_V^T μ_j` under constant gauge): defensible — exact under `Ω_ij = Ω` pullout.
   - **Sub-claim E** (every intermediate step mathematically valid as an exact identity): **fails** — the key-norm cancellation is "approximate" by the manuscript's own wording.

## Follow-up debates (if any)

The compound nature of the original input surfaced four candidate load-bearing propositions during Phase 0. Three remain as potential follow-up debates:

1. **Softmax β as stationary point** — claim: `β_ij = softmax(-KL/τ)` is the exact stationary point of F with the `τβ log(β/π)` entropy term (`GL(K)_attention.tex` §4 "Deriving Attention", line 679; supplementary §B opening at line 183). Tests the load-bearing Lagrangian derivation behind the attention rule.

2. **Canonical F vs entropy-suppressed surrogate** — claim: the canonical F and the surrogate `Σ β · KL` are NOT gradient-equivalent; their gradients differ by `-τ⁻¹ Cov_β(KL, ∇KL)` (lines 866–871, 855, 1354; supplementary §B.1 at line 183). The manuscript discloses this gap honestly; the debate would test whether the standard-transformer-training analogy (line 866 "Standard transformer training follows the autograd convention") is mathematically clean or rests on an implicit envelope-theorem invocation that is then violated by the autograd path.

3. **Multi-head = block-diagonal GL(K)** — claim: multi-head attention is exactly the block-diagonal GL(K) structure with H independent GL(K/H) frames (§5.4, line 1696). The verdict's rectangular-projection concern intersects this debate: the manuscript at line 1720 ("the rectangular shape of the standard per-head projection matrices ... can be factored via thin SVD") is the same caveat as in §5.2.2. A focused sub-debate would test whether the thin-SVD lift is a structural correspondence or a chosen factorization.

4. **Route 1 (untied carving) alone reduces to Vaswani §3.2.1** — claim: under the closure `Σ_j = U_j C U_j^T`, the untied carving `Q_i = C^{-1/2} U_i^{-1} μ_i`, `K_j = C^{-1/2} U_j^{-1} μ_j` (line 1142, 1155) recovers exactly Vaswani §3.2.1 attention with the additive prior `log π_{ij}` absorbed into the standard's masking slot. The verdict raised this as the strongest blue defense; the debate would test whether the additive-prior absorption constitutes equivalence with Vaswani §3.2.1 (no additive bias) or a generalization on top of it (Vaswani + ALiBi-style bias).
