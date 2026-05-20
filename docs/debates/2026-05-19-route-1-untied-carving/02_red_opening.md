# Red Opening — route-1-untied-carving

## Steelman (opposing position)

Route 1 carves the gauge-covariant Gaussian KL into an exact decomposition (Eq. `eq:full_kl_general`, line 1129) verified symbolically to machine precision, factors the only cross-term `-2 x_i^T k_j` as a clean dot product `Q_i^T K_j` (Eq. `eq:gauge_qk`, line 1145), and absorbs every remaining j-only structural term into the softmax prior slot `log π_{ij}` by the standard shift-invariance identity `softmax_j(s_{ij} + c_j) ∝ exp(c_j) · exp(s_{ij})`, with the residual i-j belief-covariance couplings `tr(H_j P_i)` and `x_i^T H_j x_i` collapsing to constants under the structural closure `Σ_j = U_j C U_j^T`, so that under the closure the carving simplifies to `Q_i = C^{-1/2} U_i^{-1} μ_i`, `K_j = C^{-1/2} U_j^{-1} μ_j` and standard scaled-dot-product attention is recovered without insertion of any external bilinear `M`.

## Position

The recovery is to **Vaswani §3.2.1 plus a content-dependent additive logit bias of the form `-‖K_j^{sym}‖² / (2τ)`**, not to Vaswani §3.2.1 itself. The headline word "exactly" is false under the strict reading of the destination architecture, and is the same key-norm-bias failure relocated from Route 2's "approximate cancellation" (first debate, RED_WINS) to Route 1's "absorption into `log π_{ij}`." Relabeling the term as "prior" does not eliminate its `μ_j`-dependence; Vaswani §3.2.1 has no such bias slot in the base form.

## Evidence

1. **Symbolic verification of the load-bearing algebra** (the j-only term that gets "absorbed"). Under the closure `Σ_j = U_j C U_j^T`,
   - `r_j = μ_j^T Σ_j^{-1} μ_j = μ_j^T U_j^{-T} C^{-1} U_j^{-1} μ_j = ‖C^{-1/2} U_j^{-1} μ_j‖² = ‖K_j^{sym}‖²`.

   Executed sympy session (d=2 symbolic, d=3 numeric, both seed-fixed):
   ```
   r_j - ||K_j^sym||^2 (symbolic) = 0
   r_j (numeric)            = 19.38715744993426
   ||K_j^sym||^2 (numeric)  = 19.38715744993426
   diff                     = 0.0
   d(r_j)/d(μ_j)            = [4.22, 0.90, -14.97]   (nonzero — content-dependent)
   ```
   The j-only "prior bias" absorbed into `log π_{ij}` *is* the squared key norm in the symmetric-carving frame. Its gradient with respect to `μ_j` is `2 Σ_j^{-1} μ_j ≠ 0`. It is therefore a content-dependent function of the j-th token's representation, not a constant and not a position-only bias.

2. **Vaswani 2017 §3.2.1 — primary source, no additive bias on logits.** `external_canon_transformers.md` §1 (verbatim): `Attention(Q, K, V) = softmax(Q K^T / √d_k) V`, with Q, K from input X via learned rectangular `W_Q, W_K ∈ ℝ^{d_model × d_k}`. The standard form contains **no additive `log π_{ij}` term inside the softmax.** Canon §6 (Causal masking) and the introductory paragraph of §3.2 in [Vaswani2017] reserve the additive logit slot for *extensions* of the base form — causal mask, ALiBi, T5 relative bias — all introduced *on top of* §3.2.1. Identification of the gauge reduction's `log π_{ij}` slot with the base form's logit requires that `log π_{ij}` be either absent, constant in j, or position-only (mask-style). The Route 1 absorption produces a `μ_j`-dependent term, which canon §10 pitfall 1 ("the derivation must show that under the stated limits, the user's β reduces to softmax(QK^T/√d_k)") explicitly does not permit.

3. **First debate's RED_WINS verdict on the identical quantity** (`docs/debates/2026-05-19-reduction-to-standard-transformer/04_verdict.md`, lines 9, 19). The verdict's decisive evidence was the manuscript's own "approximate" language at lines 1198, 1213, 1271, 1289 for cancellation of `‖μ_j‖²/(2σ²)`. The Route-1 quantity `r_j / (2τ) = ‖K_j^{sym}‖² / (2τ)` is the *same key-norm-squared family* in a rotated frame (`C^{-1/2} U_j^{-1}` rather than identity). The first debate's Action item 3 explicitly anticipated this very sub-debate: "Optionally, run a focused sub-debate on Route 1 §5.2.1 alone: ... 'with the additive prior `log π_{ij}` absorbed into the standard's masking slot.' That sub-debate would test whether the additive-prior absorption is a legitimate equivalence with Vaswani §3.2.1 or a generalization on top of it." The structural relocation does not change the destination test.

4. **Manuscript line 1158 self-characterization** (`Attention/GL(K)_attention.tex`, verbatim): "Terms depending on j alone, namely `r_j`, `log det Σ_j`, and `log|det U_j|`, contribute a **key-side prior bias** that is absorbed into `log π_{ij}`." The manuscript itself names this a "bias" — i.e., an additive logit term — and locates it in the prior slot. Per canon §6, the prior slot is the extension slot, not part of `softmax(QK^T/√d_k)`. The manuscript's own naming concedes that the destination is `softmax(QK^T/√d_k + bias)` rather than `softmax(QK^T/√d_k)`.

5. **The "genuinely untied" structural feature is *destroyed* by the closure that recovers standard attention** (line 1148 vs line 1158). The pre-closure carving at line 1145 has `W_Q^{(i)} = U_i^{-1}` and `W_K^{(i)} = U_i^T Σ_i^{-1}` — different functions of the per-token belief, "with the inverse-transpose tying broken by the precision factor `Σ_i^{-1}`." Under the closure `Σ_j = U_j C U_j^T`, this becomes `W_Q^{(i)} = U_i^{-1}` and `W_K^{(i)} = U_i^T (U_i C U_i^T)^{-1} = U_i^T U_i^{-T} C^{-1} U_i^{-1} = C^{-1} U_i^{-1}`, so up to the constant `C^{-1}` factor the query and key are the *same* per-token projection `U_i^{-1} μ_i`. The line-1148 claim that the construction is "genuinely untied" is true of the *general* gauge framework but false of the closure that performs the reduction. The closure recovers a *symmetric-bilinear* attention rule, not Vaswani's `W_Q W_K^T`-style untied form.

6. **Rotary identification at line 1160 is recovery of RoPE, not Vaswani §3.2.1** (canon §5: `[Su2024RoPE]` is an extension of standard transformer attention, introduced years after §3.2.1). The line-1160 paragraph explicitly identifies `U_i ∈ O(d_k)` as RoPE rotations; the resulting `Q_i = U_i^T μ_i`, `K_j = U_j^T μ_j` is the RoPE construction. The headline destination "Vaswani §3.2.1" is the base form with sinusoidal positional encoding ([Vaswani2017] §3.5); RoPE is a *different* positional encoding *and* a different inner-product structure. Calling RoPE recovery "Vaswani §3.2.1 recovery" misnames the destination.

7. **The trivial-frame fallback at line 1161 dead-ends in the first debate's failed route.** Line 1161: "The trivial-frame specialisation `U_i = U` for all `i` ... the analysis of Section `dot_product_derivation` below applies." Section `dot_product_derivation` (line 1163 onward) is Route 2 — exactly the route the first debate ruled RED_WINS on. The trivial-frame route's `r_j` becomes `‖C^{-1/2} U^{-1} μ_j‖²`, still `μ_j`-dependent. Route 1 cannot rescue Route 2 by handing off to it; it inherits Route 2's failure mode.

## Falsification conditions

This position is wrong if any of the following is shown with primary-source evidence:

- **(F1)** Vaswani 2017 §3.2.1, read as primary source, does include an additive content-dependent logit bias in the base form `Attention(Q, K, V) = softmax(QK^T/√d_k) V`, such that "Vaswani §3.2.1" already encompasses `softmax(QK^T/√d_k + f(K_j)) V` for arbitrary `f`. (Canon §1 contradicts this; would require a direct quote from [Vaswani2017] showing the bias slot is part of the base form rather than the masking/positional extension.)

- **(F2)** The standard reading of "recovery of Vaswani §3.2.1" in the gauge-attention literature treats an additive content-dependent key-side bias as part of §3.2.1 itself rather than an extension. (Canon §6 says the opposite; would require a counter-cite from a peer-reviewed treatment that identifies key-side content bias with §3.2.1.)

- **(F3)** The closure `Σ_j = U_j C U_j^T` forces `r_j = μ_j^T Σ_j^{-1} μ_j` to a constant in `j`. (Refuted symbolically and numerically in Evidence 1: `r_j = ‖K_j^{sym}‖²` with nonzero gradient in `μ_j` for generic `μ_j`. Falsifying this would require showing `μ_j` itself is constrained to a sphere or to a fixed norm, which Route 1 does not assume; the manuscript at line 1127 takes `Σ_i` as "arbitrary SPD" but says nothing constraining `μ_j`.)

- **(F4)** The "absorption into `log π_{ij}`" is mathematically equivalent to producing no logit bias at all (i.e., the absorbed term cancels under softmax row-normalization). (Refuted by softmax algebra: `softmax_j(s_{ij} + c_j)` is *not* `softmax_j(s_{ij})` when `c_j` varies in `j`; that is precisely why the term is named a "bias" at line 1158.)

- **(F5)** The verdict on Route 2 (first debate) was about a *different* quantity than `‖K_j^{sym}‖²`, so the cross-reference is illegitimate. (The first debate's decisive evidence was the `‖μ_j‖²/(2σ²)` key-norm bias under Route 2's isotropic closure; under Route 1's closure `Σ_j = U_j C U_j^T`, the same role is played by `‖C^{-1/2} U_j^{-1} μ_j‖²`. These are the same quantity in different frames — both are the squared norm of the j-th key vector in the recovered attention.)
