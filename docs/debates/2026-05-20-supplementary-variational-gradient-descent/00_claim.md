# Claim — supplementary-variational-gradient-descent

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (`Attention/GL(K)_supplementary.tex` §Variational Gradient Descent lines 617–670; canonical sources [Amari 2016 Ch. 2-4; Pennec 2006 "Intrinsic statistics on Riemannian manifolds"; Bhatia 2007 "Positive Definite Matrices"; Hall 2015 §2.7; Higham 2008 §10])
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge

## Claim

The `\section{Variational Gradient Descent: Implementation and Numerical Methods}` of `Attention/GL(K)_supplementary.tex` (lines 617–670, with `\label{app:variational_descent}`), comprising §D.1 Natural Gradient Descent on the Gaussian Manifold, §D.2 Manifold Retraction for Covariance Matrices, and §D.3 Gauge Frame Dynamics on GL(K), is complete and mathematically/theoretically pure as a self-contained supplementary chapter. The natural-gradient projections `δμ = -Σ ∇_μ` and `δΣ = -2 Σ sym(∇_Σ) Σ` are correctly derived from the Fisher-Rao metric on the Gaussian SPD manifold; the affine-invariant SPD retraction `Σ_{k+1} = V Λ^{1/2} U exp(τ Λ_B) U^⊤ Λ^{1/2} V^⊤` is numerically sound; the gauge-frame dynamics use the left-trivialised differential `Ψ_L(ad_X)(T_a) = T_a - c_1(θ) ad_X(T_a) + c_2(θ) ad_X²(T_a)` for SO(3) and the right-vs-left trivialization difference is acknowledged with cross-reference to §C. The chapter does not contain residual theoretical-purity issues comparable in magnitude to those identified and corrected in the fourteen-debate audit series.

## Sub-claims (compound)

The claim asserts SIX properties:

1. **Sub-claim α (Fisher-Rao natural-gradient projections, §D.1 lines 621–633).** Eq. (623–626) `δμ = -Σ ∇_μ, δΣ = -2 Σ sym(∇_Σ) Σ` matches the canonical Gaussian natural-gradient form derived from the Fisher metric `g(δΣ_1, δΣ_2) = (1/2) tr(Σ^{-1} δΣ_1 Σ^{-1} δΣ_2)` per [Amari 2016 §2.3, §4.3].

2. **Sub-claim β (whitening and trust-region).** The whitening step `B = Λ^{-1/2} (V^⊤ δΣ V) Λ^{-1/2}` at Eq. (631) correctly transports the tangent into the affine-invariant eigenbasis per [Pennec 2006; Bhatia 2007 Ch. 4]. The Frobenius-norm trust-region clipping is a standard numerical safety.

3. **Sub-claim γ (affine-invariant SPD retraction, §D.2 line 637).** `Σ_{k+1} = V Λ^{1/2} U exp(τ Λ_B) U^⊤ Λ^{1/2} V^⊤` with `B = U Λ_B U^⊤` is the affine-invariant exponential map on SPD at Σ_k = V Λ V^⊤. SPD-preserving by construction; matches the [Pennec 2006] / [Bhatia 2007 §4] canonical retraction.

4. **Sub-claim δ (left-trivialised differential for SO(3), §D.3 Eq. 661).** `Ψ_L(ad_X)(T_a) = T_a - c_1(θ) ad_X(T_a) + c_2(θ) ad_X²(T_a)` matches the canonical Hall 2015 §2.7 / Theorem 5.4 series expansion for the LEFT-trivialised differential `(I - e^{-ad_X})/ad_X` applied to T_a, with the SO(3) `ad_X³ = -θ² ad_X` truncation. The sign of the c_1 term flips from the §C.2.1 right-trivialised form, and the chapter explicitly acknowledges this convention difference at line 663.

5. **Sub-claim ε (trivialization relation Hall 2015).** Line 659: `Q_a^L = Ad_{exp(-φ)}(Q_a^R)` equivalently `Q_a^R · exp(φ) = exp(φ) · Q_a^L`. Verified via the standard identity `Ad_{exp(X)} = e^{ad_X}`: `Ad_{exp(-X)} · (e^{ad_X}-I)/ad_X = (I - e^{-ad_X})/ad_X`. Correct.

6. **Sub-claim ζ (preconditioner cross-references, §D.3 line 665).** The three preconditioner modes referenced as "(i) Cartan-involution-modified", "(ii) Cartan decomposition", "(iii) full pullback natural gradient" point to the §C.5 preconditioning section.

## Concrete potential issues already identified during evidence assembly

These textual issues were found during pre-debate reading:

### Issue 1: §D.3 line 665 still uses OLD pullback metric formula

The text at line 665 says:

> "(iii) the full *pullback* natural gradient `G_{ab}(φ) = ⟨Ψ(ad_X)(T_a), Ψ(ad_X)(T_b)⟩`, where `Ψ(z) = (e^z - 1)/z`, which captures the position-dependent curvature of the exponential map."

This is the OLD (uncorrected) pullback metric formula — it omits the `exp(φ) exp(φ)^⊤` factor that was added at supplementary §C.5 line 606 in the previous debate (commit `e4481f7c`). The formula here is now INCONSISTENT with the corrected §C.5 version.

Concrete fix needed: update line 665 to include the `exp(φ) exp(φ)^⊤` factor, or remove the formula and reference §C.5 directly.

### Issue 2: Cross-reference "App.~C.3" at line 665 is wrong

Line 665 says: "(positive-definite on `sl(K)` as derived in App.~C.3)".

§C subsection structure (verified via Grep):
- §C.1 Structure of the Gauge Frame Gradient
- §C.2 Differential of the Matrix Exponential
- §C.3 KL Gradient Through Transport
- §C.4 Retraction and Numerical Considerations
- §C.5 Gauge Frame Preconditioning for GL(K)

The Cartan-involution-modified metric / Killing form derivation is in §C.5, NOT §C.3. The reference "App.~C.3" should be "App.~C.5". Same pattern of cross-reference drift identified in §B line 337 ("Section 3.6" → §3.7) and elsewhere.

### Issue 3: Missing citation for affine-invariant SPD retraction

Eq. (637–639) presents the affine-invariant SPD exponential map without citing [Pennec 2006] or [Bhatia 2007 Ch. 4]. This is a canonical result on SPD-manifold geometry.

### Issue 4: Missing reference for "symmetric quadrature" K>3 path

Line 663 says: "For higher-dimensional representations (K > 3), where the Cayley-Hamilton truncation requires K terms, we instead compute `Q_a^L` via the Fréchet derivative of the matrix exponential using symmetric quadrature."

"Symmetric quadrature" is mentioned without a specific reference; this is the [Higham 2008] block-matrix algorithm or a Padé approximation method. Citation gap.

### Issue 5: Gradient-validation claim at line 670

"All gauge frame gradient implementations are validated against finite-difference approximations ... with relative error < 1e-5."

No code reference (test file path / function name) is given. Could point to `tests/transformer/test_gauge_preconditioner.py` or similar.

### Issue 6: `|τ λ_B^(j)| ≤ 50` clipping bound at line 640

The hard-coded numerical safety bound `|τ λ_B^(j)| ≤ 50` is presented without justification (why 50, why not 30 or 100?). Editorial.

## User context

Fifteenth debate in the audit series, fourth in the supplementary chapter sweep (§A, §B both RED_WINS-narrow on editorial gaps; §C RED_WINS substantive on FD-verified formula errors). The previous debate (§C) corrected three formula errors and the codebase. The post-§C state of the manuscript still has the OLD pullback formula at §D.3 line 665 — propagating the bug forward.

Load-bearing questions for the judge:

1. Do sub-claims α through ε hold under primary-source verification?
2. Sub-claim ζ explicitly relies on §C.5 cross-references — does it survive given the broken cross-reference at line 665 and the now-stale pullback formula at line 665?
3. Is the aggregate of editorial / cross-reference issues comparable to prior narrow verdicts, or does the inherited pullback bug push this debate into the substantive scope (like §C)?

A compound verdict should reflect the worst load-bearing sub-claim. Sub-claim ζ is partly broken by Issue 1 (stale pullback formula) and Issue 2 (broken cross-reference); the other sub-claims appear to hold.
