# Blue Rebuttal — supplementary-gauge-frame-gradients

## Concession

I concede the three mathematical errors red identifies as load-bearing, on independent finite-difference and trace verification. The compound claim as written — that §C "does not contain residual theoretical-purity issues comparable in magnitude to those identified and corrected in the thirteen-debate audit series" — cannot be defended at lines 599, 590, and 556. Specifically:

1. **Line 599 is wrong by O(1) relative error.** Re-run of the finite-difference test red cites (seed 0, `phi, T ~ N(0, 0.5^2) ∈ R^{3×3}`, central `eps = 1e-7`) reproduces red's numbers to the digit:

   ```
   ||D_true||                                           = 3.3553
   ||D_true - psi_R(ad_phi)(T) · exp(phi)|| (right)     = 3.66e-9   matches
   ||D_true - exp(phi) · psi_L(ad_phi)(T)|| (left)      = 3.66e-9   matches
   ||D_true - exp(phi) · psi_R(ad_phi)(T)|| (line 599)  = 7.018     FAILS
   relative error of line 599 form: 2.0915
   ```

   The manuscript's `exp(φ) · Ψ(ad_φ)(T_a)` with `Ψ(z) = (e^z - 1)/z` is neither left- nor right-trivialized; the canonical identity `exp(φ) · ψ_L(ad_φ)(T) = ψ_R(ad_φ)(T) · exp(φ)` requires the `exp(φ)` factor and the choice of generator (`ψ_L` vs `ψ_R`) to be paired consistently. The manuscript pairs them inconsistently.

2. **Line 590 is exactly 2× the value derived from line 587.** Direct trace computation on random `X, Y ∈ gl(3)` (seed 1) reproduces red's ratio:

   ```
   B(X,Y) via tr(ad_X ad_Y)                         = -26.7234
   B(X,Y) via 2K tr(XY) - 2 tr(X) tr(Y)            = -26.7234   matches
   g(X,Y) = -(1/2) B(X, theta Y)                    = -10.3806
   g_simplified = K tr(X Y^T) - tr(X) tr(Y)        = -10.3806
   g_line590 = 2K tr(T_a^T T_b) - 2 tr(T_a) tr(T_b) = -20.7612
   ratio (line 590 / derivation) = 2.0000
   ```

   The line-589 word "equivalently" is false by a factor of two. This is not editorial.

3. **Line 556 cross-reference resolves to the wrong section.** `\ref{sec:glk_lm}` lands on `GL(K)_attention.tex:2080` (§4.2 experimental WikiText-103 setup), which contains no polar-decomposition argument. The surjectivity claim `exp(φ_i) exp(-φ_j)` covers `GL^+(K)` is asserted without derivation anywhere in the manuscripts; per Culver 1966 the image of `exp: gl(K) → GL^+(K)` is not all of `GL^+(K)` for K ≥ 2, so the product-of-two-exponentials surjectivity is non-trivial and cannot be hand-waved.

The compound claim's sub-claim ζ fails. Red's primary verification stands.

## Core attack

Red overreaches by recommending a verdict of RED_WINS rather than RED_WINS-narrow. The pattern of the previous two supplementary debates (§A General Mathematical Framework, §B Covariance Dynamics — both RED_WINS-narrow) is the right calibration here. Red's own evidence demonstrates that sub-claims α (autograd assembly, line 416 verified exactly), β (Rodrigues SO(3) specialization, Taylor expansions verified), δ (KL transport derivatives, sandwich-product chain rule), and the retraction-strategy structure of §C.4 all survive primary-source verification. Red concedes this explicitly at `02_red_opening.md:84`: "The chapter's sub-claims α, β (SO(N) Rodrigues), δ, ε (modulo the polar-decomposition cross-reference) hold under primary-source verification."

Red's finite-difference falsification at line 599 is decisive for sub-claim ζ. It is not decisive for sub-claims α, β, δ, and the SO(3) half of sub-claim ε, which red does not contest. A RED_WINS verdict on a compound claim where four of six sub-claims are uncontested under verification, and the remaining two contain specific equation-level errors that admit specific equation-level fixes, conflates "the claim as stated is overconfident" with "the chapter must be rewritten." The first is true; the second is not what red's evidence supports.

The action scope is bounded. Per `02_red_opening.md:33`: "the pullback construction in §C.5 ¶4 inherits a quantitatively wrong differential." The §C.5 ¶4 pullback construction is one of four preconditioner modes in the ladder; the other three (norm clipping, Cartan projector, Killing-form metric — modulo the line-590 factor of 2) are independent and unaffected by the line-599 error. Fixing line 599 and line 606 leaves three of four modes intact. Fixing line 590 by either dropping the `-(1/2)` from line 587 or dividing line 590 by 2 restores the "equivalently" claim with a single character change. These are surgical fixes, not a chapter rewrite.

## Defense

Independent finite-difference verification of the Frobenius pullback (seed 2, `phi ~ N(0, 0.3^2) ∈ R^{3×3}`, full E_{ij} basis of gl(3)) confirms the exact form of the corrected metric:

```
G^pull_ab = tr(psi_R(ad_phi)(T_a)^T · psi_R(ad_phi)(T_b) · exp(phi) exp(phi)^T)
```

reproduces the true Frobenius-pulled-back metric at machine precision (relative error 2.77e-16). The manuscript's line-606 formula drops the right-side factor `exp(φ) exp(φ)^T`, giving relative error 1.35 against the true pullback at this scale of φ. The fix is one factor of `exp(φ) exp(φ)^T` inserted between `Ψ(ad_φ)(T_a)^T` and `Ψ(ad_φ)(T_b)` (or equivalently `exp(φ)^T exp(φ)` on the other side for the left-trivialized form). The fix is local to one equation and does not propagate further into the chapter.

The chain-rule structure that runs from §C.1 through §C.3 — the autograd β-coupled product rule (Eq. eq:phi_grad_complete), the softmax derivative simplifications (Eqs. eq:reverse_beta_grad_phi, eq:beta_grad_phi), the mean-trace-logdet decomposition of `K_{ij}`, the transport derivatives `∂μ̃/∂φ^a = Q_a Ω μ` and `∂Σ̃/∂φ^a = Q_a Ω Σ Ω^T + Ω Σ Ω^T Q_a^T` — is the operational content the codebase consumes for the gauge-frame gradient `∂F/∂φ_i`. Per `01_evidence.md:200–217`, these derivatives use the right-trivialized convention `D_φ exp[T_a] = Q_a · exp(φ)` and are internally consistent: `∂Ω_ij/∂φ_i^a = Q_a Ω_ij` follows directly from `Ω_ij = exp(φ_i) exp(-φ_j)` under the right convention, and the covariance derivative uses the sandwich product `Σ → Ω Σ Ω^T` correctly per [Nakahara2003 §10.3] and CLAUDE.md's hard constraint. The line-599 error in §C.5 does not propagate backward into §C.3's transport derivatives because §C.3 derives them from `D_φ exp` directly, not from the line-599 formula. The autograd path is mathematically pure; the line-599 error is local to §C.5's preconditioner motivation.

The Cartan-decomposition ladder structure `gl(K) = so(K) ⊕ Sym(K) ⊕ R` with four preconditioner modes (norm clipping → Cartan projector → Killing metric → pullback natural gradient) is canonical per [Iserles-Nørsett-Munthe-Kaas Acta Numerica 2000 §2.3] and [Helgason 1978 Ch. III]. The Cartan-decomposition equation at line 563–565 is correct; the Cartan projector formula `P_sym = (1/2) G^{-1}(G + S)` at line 578–580 is the standard symmetric-part projector and is unaffected by the line-590 factor or the line-599 error. The conceptual ladder — the asymmetric growth of `‖dexp_φ‖` along compact-skew vs symmetric directions justifying position-dependent preconditioning — is the load-bearing motivation for §C.5 and is correct per Hall 2015 §5.3 and Higham 2008 §10.7. The line-599 and line-606 corrections preserve this ladder; they sharpen the formula at the top of it.

Recommended verdict: **RED_WINS-narrow**, with the action being five specific corrections:

1. Replace line 599 `Ψ(z) = (e^z - 1)/z` paired with left-multiplication by `exp(φ)` with one of the canonical forms — either left-trivialized `D_φ exp[T_a] = exp(φ) · ((1 - e^{-ad_φ})/ad_φ)(T_a)`, or right-trivialized `D_φ exp[T_a] = ((e^{ad_φ} - I)/ad_φ)(T_a) · exp(φ)`. Cite [Hall 2015 Theorem 5.4].
2. Reconcile line 590 with line 587 by either dropping the `-(1/2)` from line 587 (so `g̃ = B(X, Y^T)` matches line 590) or dividing line 590 by 2 (so the "equivalently" claim holds as written). Cite [Knapp 2002 Prop 1.93].
3. Add the position-dependent factor `exp(φ) exp(φ)^T` to Eq. eq:pullback_metric at line 606, or restrict the formula's scope to the compact subalgebra `so(K) ⊆ gl(K)` explicitly. Cite [Hall 2015 §5.4].
4. Replace `\ref{sec:glk_lm}` at line 556 with a direct citation to [Higham 2008 §8] for the polar-decomposition surjectivity argument, and either add a brief surjectivity derivation in §C.4 or cite [Culver 1966] for the surjectivity of `exp(φ_i) exp(-φ_j): gl(K) × gl(K) → GL^+(K)`.
5. Unify the `dexp_φ` notation between §C.2 (right-trivialized generator), §C.2.2 (full Fréchet derivative via integral / block-matrix form), and §C.5 (line 599) by picking one convention — most likely the full Fréchet `D_φ exp[ξ]` since that is what the codebase actually computes — and renaming the right-trivialized object explicitly as `ψ_R(ad_φ)(ξ)` or `Q_φ(ξ)` to avoid the symbol collision. The current text reads line 446 as defining one object and lines 474, 481 as using the same symbol for an object differing by an `e^φ` factor.

The compound verdict the claim sought (no residual purity issues) is overconfident and fails at sub-claim ζ. The corrected verdict (chapter is salvageable with surgical formula fixes; α, β, δ, the compact-case half of ε, and the ladder structure of ζ all survive) is RED_WINS-narrow, the same verdict the previous two supplementary debates produced.
