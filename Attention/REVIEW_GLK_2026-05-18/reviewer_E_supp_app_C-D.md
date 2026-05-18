# Reviewer E — GL(K)_supplementary.tex Appendix C–D (lines 387–663)

**Date:** 2026-05-18
**Scope:** Appendix C (lines 387–607) and Appendix D (lines 609–662) of `GL(K)_supplementary.tex`. Sister file `GL(K)_attention.tex` consulted for cross-references only.
**Standards used:**
- Hall, *Lie Groups, Lie Algebras, and Representations* (matrix exponential, dexp series).
- Higham, *Functions of Matrices* (Fréchet derivative, block-matrix exp identity).
- Helgason, *Differential Geometry, Lie Groups, and Symmetric Spaces* (Cartan decomposition, Killing form on real gl(K)).
- Amari & Nagaoka, *Methods of Information Geometry* (Fisher metric on Gaussian, natural gradient).
- Bhatia, *Positive Definite Matrices*, and Pennec et al. (2006) (affine-invariant SPD retraction).
- Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix Manifolds*, and Boumal, *Introduction to Optimization on Smooth Manifolds* (retractions).

## Summary

The appendix range is mathematically uneven. The high-level structure is correct — gauge-frame gradient splits into self and cross sums, the chain rule through transport assembles three KL terms, retraction on SPD is the affine-invariant exponential map. Within these scaffolds I find five mathematical errors load-bearing enough to block acceptance in current form: (i) Eqs. 530 and 534 use a definition of `dexp_φ(T_a)` that conflicts with Eq. 441 and yields the wrong derivative under the reading the manuscript itself sets up; numerical verification shows the formulas as written give a different answer from the actual derivative of `Ω_ij μ_j` and `Ω_ij Σ_j Ω_ij^T`. (ii) Eqs. 441, 469, and 593 attach three different objects to the same symbol `dexp_φ(ξ)`, an internal notational inconsistency in a 95-line subsection. (iii) Appendix C Eq. 452 and Appendix D Eq. 652 give opposite signs for the c₁ term in the so(3) Rodrigues dexp; only Eq. 452 is correct. (iv) Eq. 581 names the bilinear form `2K tr(XY) − 2 tr(X) tr(Y)` "the Killing form of gl(K)" and asserts positive definiteness on sl(K); the form that is positive definite is the **Cartan-involution-modified** Killing form `2K tr(X^T Y) − 2 tr(X) tr(Y)` used silently in Eq. 584 and explicitly in App. D Eq. 656. (v) Eq. 629 omits the `V V^T` "interface" factors and as written is not equivalent to the retraction the code performs. Additionally, the appendix mislabels the Ω-direct retraction (left-trivialization presented as "right-invariant"), drops the `exp(φ)^T exp(φ)` middle factor in the pullback Frobenius metric, and provides two undefined `\citep` keys (`rossmann2002lie` missing from `references.bib`, `gallier2020differential` not the actual key). Style hygiene is also poor: 30+ spacing-macro violations in the range.

The good news: the natural-gradient projections in D.1 match the code; the `Sum_l ∂β/∂φ` softmax derivative at Eq. 414 is correct; the high-level architecture of the chain rule (mean, trace, logdet terms) is sound; the Cartan-decomposition preconditioner and the structural decomposition `gl(K) = so(K) ⊕ Sym(K) ⊕ ℝ` are correct. Most of the issues are repairable with notational discipline and a careful pass over the dexp algebra.

## Major findings

### M-E-1. The symbol `dexp_φ(ξ)` is used for three different objects within Appendix C

**Claim kind:** (S) — standard differential of matrix exp; classification turns on which convention.

**Locations:**
- Eq. 441 (line 441): `dexp_φ(ξ) = Sum_{n≥0} 1/(n+1)! ad_φ^n(ξ) = (e^{ad_φ} − I)/ad_φ (ξ)`. This is the **right-trivialized** differential (call it `Φ(ad_φ)`).
- Eq. 445 (line 445): "`D_φ(exp)[ξ] = dexp_φ(ξ) · e^φ`" — consistent with the right-trivialized convention.
- Eq. 469 (line 469): "`dexp_φ(ξ) = ∫_0^1 e^{tφ} ξ e^{(1−t)φ} dt`" — this is the **Fréchet derivative** `D_φ(exp)[ξ]`, which differs from the right-trivialized object by a right factor of `exp(φ)`.
- Eq. 593 (line 593): "`d exp_φ(T_a) = exp(φ) · Ψ(ad_φ)(T_a)`" — this is again the Fréchet form (factor of `exp(φ)` present).

**Standard treatment:** Two conventions are common in the literature. Hall (*Lie Groups…*, Ch. 3) reserves `dexp_x(Y) = (1 − e^{−ad_x})/ad_x · Y` and writes the Fréchet as `D_x(exp)[Y] = exp(x) · dexp_{−x}(Y) = ∫_0^1 e^{tx} Y e^{(1−t)x} dt`. Either choice is fine; using the same symbol for both within one subsection is not.

**Problem:** Numerical verification shows the three definitions are genuinely different objects (Eq. 441 ≠ Eq. 469 ≠ Eq. 593 unless `[φ, ξ] = 0`). This propagates into Eqs. 530, 534, and 537 (see M-E-2).

**Required revision:** Pick one convention. Recommendation: keep `dexp_φ` as the right-trivialized series at Eq. 441, and write `D_φ(exp)[ξ] = ∫_0^1 e^{tφ} ξ e^{(1−t)φ} dt` at Eq. 469. Then in Eq. 593 write `D_φ(exp)[T_a] = exp(φ) · dexp_φ(T_a) = exp(φ) · Ψ(ad_φ)(T_a)`.

### M-E-2. Equations 530 and 534 give wrong derivatives under the manuscript's Eq. 441 reading

**Claim kind:** (S) — calculus of `∂Ω/∂φ` for `Ω = exp(φ_i) exp(−φ_j)`.

**Locations:** Eqs. 529–537 (lines 528–537).

**Claim:**
```
∂(Ω_ij μ_j)/∂φ_i^a = dexp_{φ_i}(T_a) · exp(−φ_j) · μ_j ≡ Q_a^{(i)} · exp(−φ_j) · μ_j
```
and the analogous Eq. 534 for `∂Σ̃/∂φ_i^a`.

**Standard treatment.** With `Ω = exp(φ_i) · exp(−φ_j)`, the chain rule gives `∂Ω/∂φ_i^a = D_{φ_i}(exp)[T_a] · exp(−φ_j)`. Using the right-trivialized identity `D_φ(exp)[ξ] = dexp_φ(ξ) · exp(φ)`, we get `∂Ω/∂φ_i^a = dexp_{φ_i}(T_a) · exp(φ_i) · exp(−φ_j) = dexp_{φ_i}(T_a) · Ω_ij`. So:
```
∂(Ω_ij μ_j)/∂φ_i^a = dexp_{φ_i}(T_a) · Ω_ij · μ_j        (right-trivialized convention)
                   = D_{φ_i}(exp)[T_a] · exp(−φ_j) · μ_j   (Fréchet convention)
```

The manuscript Eq. 530 writes `Q_a^{(i)} · exp(−φ_j) · μ_j`. **This is correct only if `Q_a` is the Fréchet derivative.** Under Eq. 441's reading (right-trivialized), the expression is missing an inner `exp(φ_i)` factor.

**Numerical verification.** With `φ_i, φ_j, T_a, μ_j` as a 2×2 example:
- Numerical `d(Ω μ_j)/dφ_i^a`: `[0.984, −0.142]`.
- Eq. 530 with `Q_a` = right-trivialized series: `[1.009, 0.208]` (wrong).
- Correct: `Q_a^{right-triv} · Ω · μ_j` = `[0.984, −0.142]`.

Equation 534 has a second error: extra `exp(−φ_j)^T` factors. The form written is `Q_a · exp(−φ_j) · Σ_j · exp(−φ_j)^T · Ω^T + sym.`. Computing: with the same numerical example, this gives `[[2.19, −1.81], [−1.81, −4.87]]`, while the correct numerical derivative is `[[3.47, −0.51], [−0.51, −4.64]]`. The error is the spurious `exp(−φ_j)^T` insertion between `Σ_j` and `Ω^T`.

**Required revision.** Pick the Fréchet or right-trivialized convention for `dexp` (see M-E-1) and rewrite Eqs. 530, 533–534, 537 consistently. The correct expression for the σ channel:

```
∂Σ̃_ij/∂φ_i^a = (∂Ω_ij/∂φ_i^a) · Σ_j · Ω_ij^T  +  Ω_ij · Σ_j · (∂Ω_ij/∂φ_i^a)^T
              = D_{φ_i}(exp)[T_a] · exp(−φ_j) · Σ_j · Ω_ij^T  +  Ω_ij · Σ_j · exp(−φ_j)^T · D_{φ_i}(exp)[T_a]^T   (Fréchet)
              = dexp_{φ_i}(T_a) · Ω_ij · Σ_j · Ω_ij^T  +  Ω_ij · Σ_j · Ω_ij^T · dexp_{φ_i}(T_a)^T                  (right-triv)
```

### M-E-3. Appendix C Eq. 452 and Appendix D Eq. 652 disagree on the sign of c₁ in the so(3) dexp Rodrigues

**Claim kind:** (S) — Rodrigues form of `dexp` on so(3).

**Claim (App C, line 452):** `dexp_φ(T_a) = T_a + c_1(θ) [φ, T_a] + c_2(θ) [φ, [φ, T_a]]`.
**Claim (App D, line 652):** `Q_a = T_a − c_1(θ) ad_X(T_a) + c_2(θ) ad_X^2(T_a)`.

**Standard treatment.** The right-trivialized `dexp_φ = Sum_{n≥0} 1/(n+1)! ad_φ^n`. On so(3) with `||φ|| = θ`, the closed form is `I + ((1−cos θ)/θ²) ad_φ + ((θ − sin θ)/θ³) ad_φ²` — i.e., the `+c_1` sign matches Eq. 452.

**Numerical verification.** With `φ = θ J_3`, `θ = 0.7`, `T_a = J_1`:
- Series sum (App C, +c_1): exact match to 5e-17.
- App D (−c_1): max abs error 0.67. Off by sign.

**Diagnosis.** Eq. 652 likely intends `dexp_{−X}(T_a) = T_a − c_1 ad_X(T_a) + c_2 ad_X^2(T_a)`, but uses the symbol `Q_a` previously defined (in Eq. 537 / Eq. 593) as `dexp_X(T_a)`. The two appendices are inconsistent.

**Required revision.** Pick the App C sign and propagate to App D.

### M-E-4. Eq. 581 mislabels the Cartan-involution-modified Killing form as "the Killing form"

**Claim kind:** (S) — Killing form on gl(K, ℝ).

**Claim (line 581):** "The Killing form of gl(K) is `κ(X, Y) = 2K tr(XY) − 2 tr(X) tr(Y)`."
**Eq. 584 (line 584):** `g̃_ab = 2K tr(T_a^T T_b) − 2 tr(T_a) tr(T_b)`.

**Standard treatment.** On a real Lie algebra `g`, the Killing form is `B(X, Y) = tr(ad_X ad_Y)`. For `sl(K, ℝ)`, `B(X, Y) = 2K tr(XY)` (no transpose). On the **real** Lie algebra `sl(K, ℝ)` the Killing form is **sign-indefinite** (positive on traceless symmetric, negative on traceless skew, by the Cartan decomposition `sl(K, ℝ) = so(K) ⊕ Sym_0(K)`). The form `2K tr(X^T Y) − 2 tr(X) tr(Y)` is the **Cartan-involution-modified** form `−B(X, θ(Y))` with `θ(X) = −X^T`; this version *is* positive definite on `sl(K)`. The code at `transformer/core/gauge_preconditioner.py::build_killing_form_preconditioner` correctly implements the modified form (line 210: `g̃_ab = 2K · tr(T_a^T T_b) − 2 · tr(T_a) · tr(T_b)`).

**Problem.** Eq. 581 writes the unmodified bilinear form (no transpose) and calls it "the Killing form"; Eq. 584 silently substitutes `T_a → T_a^T` to make it positive definite without explanation. App. D Eq. 656 explicitly calls the same expression "Cartan-involution-modified Killing form" — App. C and App. D are inconsistent on naming.

**Required revision.** Either (i) write Eq. 581 as the Cartan-modified form `κ̃(X, Y) := −B(X, θ(Y)) = 2K tr(X^T Y) − 2 tr(X) tr(Y)` and explain the modification, or (ii) keep the raw Killing form in Eq. 581 and acknowledge it is sign-indefinite on sl(K, ℝ), then introduce the modification when needed for positive-definite preconditioning. Either way, name match App. D.

### M-E-5. Eq. 629 retraction formula drops the `V V^T` interface factors

**Claim kind:** (S) — affine-invariant SPD retraction.

**Claim (Eq. 629, line 629):** `Σ_{k+1} = V Λ^{1/2} U exp(τ Λ_B) U^T Λ^{1/2} V^T`, with `Σ_k = V Λ V^T` and `B = U Λ_B U^T` the whitened tangent.

**Standard treatment.** The affine-invariant exponential retraction on SPD(K) [Pennec et al. 2006; Bhatia 2007 Ch. 6] is `R_Σ(B) = Σ^{1/2} exp(B) Σ^{1/2}` when `B` is the **whitened** tangent (i.e., `B = Σ^{−1/2} V_raw Σ^{−1/2}`). Substituting `Σ^{1/2} = V Λ^{1/2} V^T`:

```
Σ_new = (V Λ^{1/2} V^T) · U exp(τ Λ_B) U^T · (V Λ^{1/2} V^T)
```

This is what `retract_spd_torch` (lines 489–506 of `transformer/core/vfe_utils.py`) computes: `Σ_sqrt @ exp_R @ Σ_sqrt` where `Σ_sqrt = eigvecs · sqrt_eig · eigvecs^T = V Λ^{1/2} V^T`. The code is correct.

**Problem.** Eq. 629 writes `V Λ^{1/2} U exp(τ Λ_B) U^T Λ^{1/2} V^T`, dropping the inner `V^T V = I` between `Λ^{1/2}` and `U` (and `U^T` and `Λ^{1/2}`). As written, Eq. 629 is `V Λ^{1/2} (V^T V)^{−1?} · U exp · U^T · (V V^T)^{−1?} Λ^{1/2} V^T` — the manuscript expression is only equal to the correct form if one interprets `V Λ^{1/2}` as standing in for `Σ^{1/2}`, but with `Λ^{1/2}` being treated as a diagonal applied in the eigenbasis. The notation is ambiguous and, read literally as concatenation of matrices, is not the standard retraction.

**Required revision.** Write either `Σ_new = Σ^{1/2} U exp(τ Λ_B) U^T Σ^{1/2}` with `Σ^{1/2} = V Λ^{1/2} V^T`, or fully expand `Σ_new = V Λ^{1/2} V^T · U exp(τ Λ_B) U^T · V Λ^{1/2} V^T`.

### M-E-6. The "right-invariant" Ω-direct retraction is left-invariant under the standard convention

**Claim kind:** (S) — left- vs. right-invariant retractions on a Lie group.

**Locations:** App. C does not directly discuss; the appendix range covers only `φ`-mode updates. However, the corresponding code section (`transformer/vfe/e_step.py::_forward_omega_direct`, lines 1107–1367) is referenced indirectly via App. C.5 and is consistent with the App. C.5 preconditioning discussion. The code docstring (line 1122) describes `Ω → Ω · exp(−η X)`, `X = proj(Ω^{−1} · dF/dΩ)` as "right-invariant". This is a naming mismatch.

**Standard treatment.** For a Lie group `G` with a left-invariant metric, `⟨X, Y⟩_Ω = ⟨Ω^{−1} X, Ω^{−1} Y⟩_e`, the Riemannian gradient at Ω is `grad F(Ω) = Ω · (Ω^{−1} · dF/dΩ)_{♯_e}` (left-trivialization), and the retraction is `Ω → Ω · exp(−η · X_{♯_e})`. Right-invariant gives the opposite arrangement: `grad F(Ω) = ((dF/dΩ) · Ω^{−1})_{♯_e} · Ω` and step `Ω → exp(−η · ...) · Ω`. The code does `X = Ω^{-1} dF/dΩ` followed by `Ω · exp(−η X)` — this is the **left-invariant** convention. References: Boumal, *Optimization on Smooth Manifolds*, §7.5; Absil-Mahony-Sepulchre §4.1.

**Problem.** The label is wrong; the math is fine. Since the appendix range does not include the docstring text, this is informational; flag if it appears in the body of App. C.5 or App. D in future revisions.

**Required revision.** Code-side rename or comment correction. Not in manuscript scope unless explicitly stated.

## Minor findings

### m-E-1. Eq. 600 pullback metric formula drops `exp(φ)^T exp(φ)` middle factor

**Claim (line 599–600):** "`G_ab(φ) = ⟨Ψ(ad_φ)(T_a), Ψ(ad_φ)(T_b)⟩_G`, where `⟨X, Y⟩_G = tr(X^T Y)`."

**Issue.** If we pull back the Frobenius inner product on `gl(K)` (as a vector space, at the point `exp(φ) ∈ GL(K)`) via the Fréchet derivative `D_φ(exp)[T_a] = exp(φ) · Ψ(ad_φ)(T_a)`, the pullback is

```
G_ab(φ) = ⟨exp(φ) Ψ(ad_φ)(T_a), exp(φ) Ψ(ad_φ)(T_b)⟩ = tr(Ψ(ad_φ)(T_a)^T · exp(φ)^T exp(φ) · Ψ(ad_φ)(T_b))
```

This has the inner factor `exp(φ)^T exp(φ)`, which on non-compact GL(K) is **not** the identity. Eq. 600 drops it. What Eq. 600 actually computes is the pullback of the Frobenius metric through the **right-trivialized** differential, which is a left-invariant metric on the group — not the pullback through `D_φ(exp)` of a fixed metric on the group.

These are different metrics. Either is a valid choice, but the manuscript should pick one explicitly. The code (`gauge_preconditioner.py::build_pullback_metric_tensor`, line 438) describes it as "pullback of the bi-invariant Frobenius metric on GL(K)" — bi-invariance of Frobenius on non-compact GL(K) is false. Both manuscript and code prose need correction.

**Required revision.** Reword Eq. 600 vicinity: "the metric induced by the right-trivialized differential", and drop the "bi-invariant" claim. If the intent is genuinely the pullback of the constant-coefficient Frobenius metric, include the `exp(φ)^T exp(φ)` factor.

### m-E-2. App. D Eq. 648 reads `∂/∂φ^a exp(Sum_b φ^b T_b) = exp(X) · Q_a`

The form is correct when `Q_a = dexp_X(T_a)` is the right-trivialized differential (so `D_X(exp)[T_a] = dexp_X(T_a) · exp(X)`). However, Eq. 648 writes the product as `exp(X) · Q_a` (left), not `Q_a · exp(X)` (right). These differ unless `[Q_a, exp(X)] = 0`. The standard identity (Hall §3.4, Higham §10.2) is `D_X(exp)[T_a] = ∫_0^1 e^{tX} T_a e^{(1−t)X} dt`, which factors as `e^X · ∫_0^1 e^{-(1-t)X} T_a e^{(1-t)X} dt · ... ` Wait — let me re-derive: `D_X(exp)[T_a] = e^X · dexp_{−X}(T_a) = dexp_X(T_a) · e^X`. Both factorizations are valid; in particular, `e^X · dexp_{−X}(T_a)` (left) equals `dexp_X(T_a) · e^X` (right). So Eq. 648 is correct **if** `Q_a` is interpreted as `dexp_{−X}(T_a)` — but Eq. 650 defines `Q_a = dexp_X(T_a)`. The two are inconsistent. Combined with M-E-3, App. D's notation here is shaky.

**Required revision.** Define `Q_a := dexp_{−X}(T_a)` and use the left factorization, or `Q_a := dexp_X(T_a)` with right factorization `exp(X) → exp(X)` on the **right**: `D_X(exp)[T_a] = Q_a · exp(X)`.

### m-E-3. Block-matrix identity (Eq. 476) is correct but pinned to a missing citation

The Mathias / Najfeld-Havel block identity `exp([[φ ξ];[0 φ]]) = [[exp(φ), dexp_φ(ξ)];[0, exp(φ)]]` is standard (Higham 2008, Functions of Matrices, Theorem 10.13). Here `dexp_φ` in this identity is the **Fréchet** derivative `D_φ(exp)[ξ]`. Adds to the M-E-1 confusion: the same symbol on the LHS of Eqs. 469 and 476 refers to the Fréchet object, while Eq. 441 defines it as the right-trivialized object.

### m-E-4. Eq. 514, 522 chain-rule expressions are correct symbolically but require Σ̃ symmetric

The trace and logdet derivatives are standard matrix-calculus facts and they hold because `Σ̃_ij = Ω_ij Σ_j Ω_ij^T` is symmetric. The appendix should note this (or refer back to the sandwich-product identity from App. B). The sandwich `Ω Σ Ω^T` is the standard rule for (2,0)-tensor parallel transport (Nakahara, *Geometry, Topology and Physics*, §10.3), and the code preserves it (verified in `compute_kl_attention` and `compute_pairwise_omega_with_delta`).

### m-E-5. Eq. 462 / line 462 Taylor coefficients for the c₁, c₂ small-angle limits

`c_1(θ) = (1 − cos θ)/θ² ≈ 1/2 − θ²/24 + ...` ✓ (next term `+θ⁴/720`).
`c_2(θ) = (θ − sin θ)/θ³ ≈ 1/6 − θ²/120 + ...` ✓ (next term `+θ⁴/5040`).
Small-angle limits correct.

### m-E-6. Eq. 442 series identity

`Sum_{n≥0} 1/(n+1)! z^n = (e^z − 1)/z`. ✓ This is the entire-function `Ψ(z)` Eq. 594. Convergence: `Ψ` is entire, so the series for `Ψ(ad_φ)` converges in operator norm for any bounded `ad_φ`. The appendix asserts convergence implicitly; for a finite-dimensional Lie algebra `gl(K)` this is fine, but it would be worth a sentence acknowledging that **on non-compact groups, `exp` itself is not a global diffeomorphism**, so even though `Ψ(ad_φ)` converges, the **invertibility** of `Ψ(ad_φ)` fails at conjugate points where the eigenvalues of `ad_φ` are in `2πi ℤ \ {0}`. This blocks the right-trivialized differential from being a diffeomorphism on a neighborhood. The pullback metric is degenerate at these conjugate points. The appendix should at least note this caveat.

### m-E-7. Line 661 finite-difference validation

"All gauge frame gradient implementations are validated against finite-difference approximations [...] with relative error < 1e-5." This is healthy practice, but I cannot find a corresponding test file in the audit map. `tests\test_math_utils.py` exists but I have not verified that it tests the `dexp` Rodrigues forms or the `_update_phi` gradient chain. If this validation is real, please point to the test path in the manuscript or add `\citep` / footnote.

### m-E-8. Line 564 normbound claim

"`||d exp_φ(T_a)|| = O(1) uniformly in ‖φ‖`" for compact directions, and `~exp(‖φ‖)` for symmetric directions. The first is correct for the spectrum of `ad_φ` purely imaginary (compact case); the second uses a heuristic norm argument. Strictly speaking, on a symmetric direction `φ = X` (X symmetric), `ad_X` has real eigenvalues, and `(e^{ad_X} − I)/ad_X` has operator norm bounded by `(e^{‖ad_X‖} − 1)/‖ad_X‖` for non-singular `ad_X`. The "~exp(‖φ‖)" rate is only roughly correct; the precise bound depends on the spectral structure of `ad_X` and on whether `‖ad_X‖` is the operator norm of the adjoint action (which on `gl(K)` is `≤ 2 ‖X‖_op` since `ad_X(Y) = XY − YX`). The order-of-magnitude argument is fine for motivating preconditioning, but the appendix should label it as a bound, not an equality.

### m-E-9. Implementation claim line 552

"automatic differentiation through PyTorch's `matrix_exp` handles the GL(K) differential implicitly." Verified: the code path `transformer/vfe/attention.py:65 → core/gauge_utils.py:111` uses `torch.linalg.matrix_exp`. The autograd backward for `torch.linalg.matrix_exp` uses Pade approximant scaling-and-squaring with cached intermediate exponentials; the gradient is the Fréchet derivative computed via the block-triangular formula or equivalent. PyTorch 2.x ships this. ✓.

### m-E-10. `η_φ` value discrepancy

Line 658: "we apply a step with learning rate `η_φ` (typically 10^{-1})". The actual default in `transformer/vfe/config.py:45` is `e_phi_lr: float = 0.05`, i.e., 5e-2 not 1e-1. Either the manuscript should say "5e-2 to 1e-1" (range) or the default value should match. Minor.

### m-E-11. CLAUDE.md trust-region vs. retraction discrepancy not addressed

CLAUDE.md (project policy) says `σ_new = σ · exp(E_sigma_q_lr · decay_t · clamp(δσ/σ, ±E_sigma_q_trust))`. The actual `retract_spd_diagonal_torch` (vfe_utils.py:559–577) does not apply a `decay_t` factor inside the retraction; if the user's intent is cosine-decayed σ LR, the decay is presumably applied externally (multiplying `effective_lr` before calling the retract function). I do not see this decay in `transformer/vfe/e_step.py::forward` (lines 672–685). The appendix is silent on the decay schedule. Either the CLAUDE.md description is aspirational (and should be aligned with the code), or the code is missing the cosine-decay multiplier — beyond the scope of this manuscript review, but worth surfacing for the auditor.

### m-E-12. Eq. 568 norm-clipping baseline definition

`g̃ = g · min(1, c/‖g‖)`. Standard; matches `phi_evolution.py::precondition_phi_gradient` mode `'clip'` (calls the threshold 10.0). Minor: appendix could cite the threshold value.

### m-E-13. Center-regularization of the Killing metric

Eq. 588 mentions "regularized by `g̃ → g̃ + ε I` in practice". The code does this differently — `build_killing_form_preconditioner` (gauge_preconditioner.py:226–229) describes a `center_reg` parameter and discusses condition-number tradeoffs explicitly. The manuscript should reference the code's regularization parameter by name and value, or at least note that center regularization induces an O(1/ε) condition number on the center direction.

## Equation verification log

| Eq. line | Object | Verification | Verdict |
|---|---|---|---|
| 397–407 | `∂F/∂φ_i` decomposition | Algebraic verification of product rule, sums over (i aligning) and (others aligning to i) | ✓ Correct structure |
| 411–415 | `Sum_l ∂β_kl/∂φ_i · K_kl` reduction | Direct softmax derivative `∂β_kl/∂s_kl = β_kl(δ_{lk} − β_km)`; only `s_ki` depends on `φ_i` | ✓ Correct |
| 422–427 | `∂β_ij/∂φ_i` | Same as 414, with `K_ij` instead of `K_ki` | ✓ Correct |
| 441 | `dexp_φ = (e^{ad_φ}−I)/ad_φ` (right-trivialized) | Hall §3.4 | ✓ as right-triv. |
| 452 | `dexp_φ(T_a)` Rodrigues SO(3) | Numerical match to 5e-17 (verified `θ=0.7, T_a=J_1`) | ✓ |
| 469 | `dexp_φ = ∫₀¹ e^{tφ} ξ e^{(1-t)φ} dt` | Higham 2008 Thm 10.13 (Fréchet derivative). | **Symbol conflict with Eq. 441** |
| 476 | Block-matrix identity | Higham 2008 Thm 10.13 / Najfeld-Havel | ✓ structurally (`dexp` here is Fréchet) |
| 487–497 | KL chain rule template | Standard Gaussian KL gradient | ✓ |
| 504–508 | Mean term derivative | Mahalanobis derivative correct, but "+ (terms from `∂Σ̃^{-1}/∂φ`)" is a vague placeholder — manuscript does not write out this contribution to the mean term. | △ Incomplete |
| 513–516 | Trace term derivative | `∂/∂φ tr(A^{-1} B) = -tr(A^{-1} dA/dφ A^{-1} B)` is the standard matrix-calculus identity. ✓ |
| 521–524 | Logdet term derivative | `∂/∂φ log|A| = tr(A^{-1} dA/dφ)` ✓ |
| 530–531 | `∂μ̃/∂φ_i^a` | **Wrong** as written under right-triv `Q_a`; correct only if `Q_a` is Fréchet (M-E-2) |
| 533–534 | `∂Σ̃/∂φ_i^a` | **Wrong** as written; spurious `exp(-φ_j)^T` factor (M-E-2) |
| 544 | `φ ← φ − η ∂F/∂φ` | Standard | ✓ |
| 581 | Killing form formula | **Mislabeled** — this is `−B(X, θ(Y))` not `B(X, Y)` (M-E-4) |
| 584 | `g̃_ab = 2K tr(T_a^T T_b) − 2 tr(T_a) tr(T_b)` | Matches code | ✓ (assuming Cartan-modified) |
| 593–594 | `dexp_φ(T_a) = exp(φ) · Ψ(ad_φ)(T_a)` (Fréchet form) | **Symbol conflict with Eq. 441** |
| 600 | Pullback metric | Inner `exp(φ)^T exp(φ)` factor dropped (m-E-1) |
| 615–616 | Natural gradient projections | Fisher metric on Gaussian: `Σ ∇_μ`, `2 Σ ∇_Σ Σ` | ✓ Matches `vfe_gradients.py:1996-2022` |
| 622 | Whitened tangent `B = Λ^{-1/2} (V^T δΣ V) Λ^{-1/2}` | Standard SPD whitening | ✓ Matches `vfe_utils.py:493` |
| 629 | Affine-invariant retraction | **Drops `V V^T` interface factors** (M-E-5) |
| 648 | `∂exp(X)/∂φ^a = exp(X) · Q_a` | Convention conflict with Eq. 537/593 (m-E-2) |
| 652 | App. D Rodrigues with `−c_1` | **Wrong sign on c_1** (M-E-3) |
| 656 | Cartan-modified Killing form metric | Matches code, naming consistent with App. D | ✓ |

## Style scan

The 387–663 range contains heavy use of banned spacing macros `\,`, `\;`, `\!`. From the grep output:

- `\,\|\,` separator inside KL: lines 187, 188, 189, 197, 234, 241, 263, 322, 324, 327, 329, 374, 1070-1315 (out of scope but pattern persists).
- `\!\big`, `\!\left`, `\!\Bigr`, `\!\begin`: lines 200, 209, 322, 327, 374, 469, 476, 492, 514, 522, 648.
- `\;\oplus\;` in Cartan decomposition (line 560).
- `\,`, `\;` spacing in display equations: lines 322, 327, 469, 600, 615, 616, 622, 629, 648, 652.

Total: ~30 spacing-macro instances within lines 387–663. All should be stripped (project policy `style_constraints.md`).

**Banned Claude-isms in range 387–663:** I did not see `key insight`, `crucially`, `notably`, etc. in this range. ✓.

**Horizontal rules:** none in range. ✓.

**Equation punctuation:** Most display equations end with `,` or `.` ✓. Eq. 411–414 ends with `.` ✓. Eq. 427 ends with `.` ✓. Eq. 537 ends without punctuation — minor.

**Self-referential drafting language:** None observed. ✓.

## Citations checked

- `\citep{gallier2020differential,rossmann2002lie}` (line 438): **Both keys absent from `references.bib`**. The bib has `Gallier2020` (capital G) and **no Rossmann entry at all**. Will fail to compile cleanly; LaTeX will emit `?` placeholders. The reference at `references.bib:1113` is `Gallier2020`, *Differential Geometry and Lie Groups: A Computational Perspective*. Either rename the cite key in the manuscript to `Gallier2020`, or add a `gallier2020differential` alias to the bib. For Rossmann, add a proper entry — likely `Rossmann2002`, *Lie Groups: An Introduction Through Linear Groups* (Oxford), or replace with `Hall2015`.
- `\citep{higham2008functions}` (line 473): bib has `Higham2008` (capital H), not the lowercase `higham2008functions`. Same issue.
- `\citep{culver1966existence}` (line 552): bib has the key `culver1966existence` ✓ — only one of the four in the range that matches.

These four citation keys must be reconciled before submission.

External citations checked (against the standard literature):
- Hall, *Lie Groups, Lie Algebras, and Representations* §3 — `dexp` conventions verified against the right-trivialized form.
- Higham, *Functions of Matrices*, §10.2-10.6 — Fréchet derivative integral and block-matrix identity verified.
- Helgason, *Differential Geometry, Lie Groups, and Symmetric Spaces*, Ch. III — Cartan decomposition / Killing form sign-indefiniteness on real semisimple Lie algebras verified at the textbook-section level (cannot verify specific page without the book in front of me).
- Bhatia, *Positive Definite Matrices*, Ch. 6 — affine-invariant SPD retraction.
- Pennec et al. 2006 (affine-invariant metric on SPD covariance matrices) — same.
- Amari, *Information Geometry and Its Applications*, Ch. 6 — Fisher metric on Gaussian and natural gradient.
- Boumal, *Introduction to Optimization on Smooth Manifolds*, §7.5 — left/right-invariant retraction conventions.

## Code cross-references checked

**`transformer/vfe/e_step.py::VFEEStep` (the inner E-step loop):**

- The φ retraction call is `_retract_phi(phi, grad_phi, ...)` at line 790, invoked from `_update_phi` (line 706). The function lives in `transformer/core/vfe_utils.py:602`. ✓ exists.
- The σ retraction goes through `retract_sigma_e_step` at line 675, defined at `transformer/core/vfe_utils.py:993`. Inside, `retract_spd_diagonal_torch` (line 528) does `σ_new = σ · exp(τ · clamp(δσ/σ, ±trust))` — matches the diagonal-σ retraction the appendix describes (line 631 doesn't explicitly write the diagonal form; the appendix focuses on the full-cov case). However, the appendix gives no formula for diagonal-σ retraction — App. D.2 is full-cov only. **The manuscript should at minimum state that diagonal-σ uses `σ_new = σ · exp(τ · δσ/σ)` after clamping, since this is the production path** (`diagonal_covariance=True` per `BlockConfig.__post_init__`).
- `_update_phi` (e_step.py:706) builds Ω from a fresh detached φ leaf (line 730: `phi_for_grad = phi.detach().requires_grad_(True)`) and computes `grad_phi` via autograd through `compute_kl_attention` (line 776: `torch.autograd.grad(alignment_loss, phi_for_grad, ...)`). The appendix's description of the φ gradient as "automatic differentiation through `torch.matrix_exp`" (line 552) is accurate. ✓.
- Natural gradient projection: `compute_natural_gradient_gpu` (vfe_gradients.py:1966) returns `(σ · ∇_μ, 2σ² · ∇_σ)` for diagonal-σ, which matches Eq. 615–616 for both `δμ` and `δΣ`. ✓.

**`transformer/vfe/config.py::VFEConfig`:**

- `e_mu_lr: float = 0.1` (line 40), `e_sigma_lr: float = 0.001` (line 41), `e_sigma_q_trust: float = 5.0` (line 42), `e_phi_lr: float = 0.05` (line 45). The appendix (line 658) says `η_φ` is "typically 10^{-1}" — actually default is 5×10^{-2}. m-E-10 above.
- The CLAUDE.md naming `E_mu_q_lr`, `E_sigma_q_lr`, `E_sigma_q_trust` does NOT match the actual config field names `e_mu_lr`, `e_sigma_lr`, `e_sigma_q_trust`. The appendix range does not reference these names by code-side identifier, so this is a CLAUDE.md ↔ code drift, not a manuscript issue.

**`transformer/vfe/block.py::MahalanobisNorm(μ, σ)`:** Outside the range of the appendix; the manuscript does not reference it in 387–663. CLAUDE.md flags a known gap with `RoPE × MahalanobisNorm`; the appendix does not mention this. The known limitation should be acknowledged elsewhere in the supplementary, not necessarily here.

**`transformer/core/connection.py` (optional non-flat transport):**

- The appendix range does not discuss the non-flat connection MLP option. Out of scope for App. C–D.

**`transformer/core/gauge_preconditioner.py`:**

- `build_cartan_projector` (line 58) implements `P_sym = (1/2) G^{-1} (G + S)` with `S_{ab} = tr(T_a T_b)`. The manuscript Eq. 575 matches: `P_sym = (1/2) G^{-1} (G + S)`. ✓.
- `build_killing_form_preconditioner` (line 199) computes `g̃_ab = 2K tr(T_a^T T_b) − 2 tr(T_a) tr(T_b)`. The manuscript Eq. 584 matches. ✓.
- `build_pullback_metric_tensor` (line 424) computes `G_ab(φ) = Ψ^T · gram · Ψ` where `Ψ = Sum_k 1/(k+1)! ad_φ^k`. The manuscript Eq. 600 matches **the right-trivialized pullback**, not the Fréchet pullback (see m-E-1). The docstring (line 438) claims "bi-invariant Frobenius metric" — false for non-compact GL(K).

**`transformer/core/phi_evolution.py::precondition_phi_gradient`:**

- Mode strings `'clip'`, `'cartan'`, `'killing'`, `'pullback'` match the four App. C.5 strategies. ✓.

**`transformer/vfe/omega_direct.py::omega_natural_grad_step`:**

- Update rule `Ω → Ω · exp(−η · proj(Ω^{−1} dF/dΩ))` (lines 290–360). The docstring calls it "right-invariant Riemannian step"; this is the left-invariant convention under Boumal/Absil-Mahony-Sepulchre (M-E-6). The math is fine; only the label is wrong.
- `_build_killing_matrix_per_block` (line 51, referenced at e_step.py:264) is used per block when `gauge_parameterization='omega_direct'`. Out of scope for App. C/D, which focus on `φ`-mode.

**Tests:**

- Did not find a dedicated unit test for the Eq. 530/534 transport derivative chain. `transformer/pure_vfe/tests/test_gradients.py` (per `codebase_map.md`) tests gradients via finite differences; whether it covers φ gradients with non-trivial `φ_i ≠ 0`, `φ_j ≠ 0`, transported Σ is something the codebase auditor should verify in a separate pass.

## Overall verdict

Major revision required. The high-level structure is publishable, but five mathematical errors (M-E-1 through M-E-5) in load-bearing equations break the chain of computation as written. The errors are repairable with a notational/disciplined pass: pick one `dexp` convention and propagate; fix the c_1 sign in App. D; rewrite Eq. 629 as `Σ_new = Σ^{1/2} U exp(τΛ_B) U^T Σ^{1/2}` with `Σ^{1/2} = V Λ^{1/2} V^T` expanded once; relabel the bilinear form in Eq. 581 as Cartan-modified Killing; and add the inner `exp(φ)^T exp(φ)` factor (or relabel the pullback as right-trivialized) in Eq. 600.

Additionally, the four citation-key mismatches (`gallier2020differential`, `rossmann2002lie`, `higham2008functions`, mostly resolved by `culver1966existence`) must be fixed before submission.

The manuscript-to-code consistency on the natural gradient (D.1) and the SPD retraction core (D.2) is strong; the gauge-frame Lie-algebra gradient and the preconditioning options (C.5) match the code modulo the naming issues (m-E-1, M-E-4). The empirical chain is sound; the algebra needs another careful pass.
