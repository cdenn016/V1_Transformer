# Claim — supplementary-gauge-frame-gradients

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (`Attention/GL(K)_supplementary.tex` §Gauge Frame Gradients lines 392–612; canonical sources [Hall 2015 §2.7, §5.4; Higham 2008 §10.2, Alg. 10.27; Gallier 2020; Iserles-Nørsett-Munthe-Kaas Acta Numerica 2000; Knapp 2002 §1.5])
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge

## Claim

The `\section{Gauge Frame Gradients}` of `Attention/GL(K)_supplementary.tex` (lines 392–612, comprising §C.1 Structure of the Gauge Frame Gradient, §C.2 Differential of the Matrix Exponential with §C.2.1 SO(N) and §C.2.2 GL(K) specializations, §C.3 KL Gradient Through Transport, §C.4 Retraction and Numerical Considerations, §C.5 Gauge Frame Preconditioning for GL(K) with four preconditioner modes) is complete and mathematically/theoretically pure as a self-contained supplementary chapter. The chain-rule decomposition `φ_i → Ω_ij → KL_ij` with the autograd β-coupled product-rule structure is correctly assembled; the differential of the matrix exponential `dexp_φ(ξ) = Σ ad_φ^n(ξ)/(n+1)! = (e^{ad_φ}-I)/ad_φ (ξ)` and its SO(N) Rodrigues form and GL(K) integral / block-matrix form are correctly stated; the KL gradient through transport with the sandwich-product covariance derivative is correctly computed; the retraction strategies for compact SO(N) and non-compact GL(K) are numerically sound; the Cartan decomposition `gl(K) = so(K) ⊕ Sym(K) ⊕ R` with its four preconditioning strategies (norm clipping, Cartan projector, Killing-form metric, pullback natural gradient) is correctly motivated. The chapter does not contain residual theoretical-purity issues comparable in magnitude to those identified and corrected in the thirteen-debate audit series.

## Sub-claims (compound)

The claim asserts SIX properties:

1. **Sub-claim α (autograd gradient assembly, §C.1 lines 397–432).** Eq. eq:phi_grad_complete correctly captures both directions of `φ_i` influence (agent `i` attending to others, others attending to `i`); the simplification at Eq. eq:reverse_beta_grad_phi and Eq. eq:beta_grad_phi correctly applies the softmax derivative.

2. **Sub-claim β (dexp series and trivialization, §C.2 lines 434–468).** The Hall/Gallier series form `dexp_φ(ξ) = (e^{ad_φ}-I)/ad_φ (ξ)` and its SO(3) Rodrigues specialization with `c_1(θ) = (1-cos θ)/θ²`, `c_2(θ) = (θ-sin θ)/θ³` are correctly stated and grounded in [Hall 2015 Theorem 5.4 / §2.7; Gallier 2020].

3. **Sub-claim γ (GL(K) integral / block-matrix form, §C.2.2 lines 469–486).** The integral form `dexp_φ(ξ) = ∫_0^1 e^{tφ} ξ e^{(1-t)φ} dt` and the (2K × 2K) block-matrix identity are canonical per [Higham 2008 §10.2, Algorithm 10.27].

4. **Sub-claim δ (KL gradient through transport, §C.3 lines 488–541).** The mean-term, trace-term, log-determinant-term decomposition and the transport derivatives `∂μ̃_ij/∂φ_i = Q_a^(i) Ω_ij μ_j`, `∂Σ̃_ij/∂φ_i = Q_a^(i) Ω_ij Σ_j Ω_ij^⊤ + Ω_ij Σ_j Ω_ij^⊤ (Q_a^(i))^⊤` are correctly derived using the sandwich product per CLAUDE.md hard constraint.

5. **Sub-claim ε (retraction, §C.4 lines 543–557).** Lie-algebra updates in the unstructured-`g` space are valid; the SO(N) retraction to the principal ball `‖φ‖ < π - ε` handles the periodicity of the exponential map; GL(K) requires no retraction beyond gradient clipping per the polar-decomposition surjectivity argument.

6. **Sub-claim ζ (gauge-frame preconditioning, §C.5 lines 558–612).** The Cartan decomposition `gl(K) = so(K) ⊕ Sym(K) ⊕ R` is correctly stated; the Cartan projector `P_sym = (1/2) G^{-1}(G + S)`, the Cartan-involution-modified bilinear form `g(X,Y) = -(1/2) B(X, θ(Y))`, the Killing form on `sl(K)`, and the pullback natural-gradient construction with `Ψ(z) = (e^z-1)/z` are correctly motivated and computed.

## Concrete potential issues already identified during evidence assembly

These are concrete textual issues identified before dispatching openings:

### Issue 1: dexp_φ notation inconsistency between §C.2 and §C.2.2 and §C.5

The chapter uses the symbol `dexp_φ` in three distinct senses:

- **Lines 446–450 (§C.2)**: `dexp_φ(ξ) = (e^{ad_φ}-I)/ad_φ (ξ)`, with the explicit statement at line 450: "Equation (eq:dexp_series) defines the right-trivialized differential, related to the Fréchet derivative (directional derivative in matrix space) by `D_φ(exp)[ξ] = dexp_φ(ξ) · e^φ`."

- **Lines 471–476 (§C.2.2)**: Eq. eq:dexp_integral writes `dexp_φ(ξ) = ∫_0^1 e^{tφ} ξ e^{(1-t)φ} dt`. The introductory sentence at line 471 explicitly identifies this with "the Fréchet derivative `D_φ(exp)[ξ]`." But per Higham 2008 §10.2, the integral form gives `D_φ exp[ξ]`, not `dexp_φ(ξ)`. These differ by a factor of `e^φ`.

- **Lines 480–484 (§C.2.2)**: Eq. eq:dexp_block writes `exp([[φ, ξ], [0, φ]]) = [[e^φ, dexp_φ(ξ)], [0, e^φ]]`. This is Higham 2008 Algorithm 10.27, which gives the off-diagonal block `L_exp(φ, ξ) = D_φ exp[ξ]` (the full Fréchet derivative), not `(e^{ad_φ}-I)/ad_φ (ξ)`.

- **Line 599 (§C.5)**: `d exp_φ(T_a) = exp(φ) · Ψ(ad_φ)(T_a)` with `Ψ(z) = (e^z-1)/z`. This mixes the right-trivialized generator `(e^z-1)/z` with a left-multiplication by `exp(φ)`. The canonical left-trivialized form is `D_φ exp[ξ] = exp(φ) · ((1-e^{-ad_φ})/ad_φ)(ξ)` (with `(1-e^{-z})/z`, not `(e^z-1)/z`). The right-trivialized form is `D_φ exp[ξ] = ((e^{ad_φ}-I)/ad_φ)(ξ) · exp(φ)`. These two forms are equivalent via the operator identity `(e^z-1)/z = e^z · (1-e^{-z})/z`. The chapter's mixed convention at line 599 is non-standard.

This is a substantive notational inconsistency that could mislead implementations.

### Issue 2: Killing-form normalization at Eq. eq:killing_metric (line 590)

The chapter defines the Cartan-involution-modified bilinear form as `g(X, Y) = -(1/2) B(X, θ(Y))` at line 587. The Killing form on gl(K) is `B(X, Y) = 2K tr(XY) - 2 tr(X) tr(Y)` per [Knapp 2002 Prop 1.93]. Direct computation:

```
g(X, Y) = -(1/2) B(X, θ(Y)) = -(1/2) B(X, -Y^⊤) = (1/2) B(X, Y^⊤)
       = (1/2)[2K tr(X Y^⊤) - 2 tr(X) tr(Y^⊤)]
       = K tr(X Y^⊤) - tr(X) tr(Y)
```

But Eq. eq:killing_metric at line 590 states `tilde_g_{ab} = 2K tr(T_a^⊤ T_b) - 2 tr(T_a) tr(T_b)`, which is `B(T_a, T_b^⊤)`, not `(1/2) B(T_a, T_b^⊤)`. Either the `g̃` symbol denotes `2g` (i.e., dropping the `-(1/2)` prefactor from the definition), or the coefficient is off by a factor of 2.

### Issue 3: Cross-reference at line 556 to `sec:glk_lm`

The supplementary at line 556 says "but the pairwise transport `Ω_ij = exp(φ_i) exp(-φ_j)` covers all of `GL^+(K)` via polar decomposition (see main text, Section~\ref{sec:glk_lm})." The label `sec:glk_lm` resolves to main paper §4.2 "GL(K) Language Modeling: The Full General Model" at `GL(K)_attention.tex:2080`, which is an EXPERIMENTAL design subsection, not a polar-decomposition derivation. The cited content does not exist at the cited location.

### Issue 4: Pullback metric assumes compact subgroup

Eq. eq:pullback_metric (line 606) writes `G_{ab}(φ) = ⟨Ψ(ad_φ)(T_a), Ψ(ad_φ)(T_b)⟩_G` with `⟨X, Y⟩_G = tr(X^⊤ Y)` (Frobenius). If `d exp_φ(T_a) = exp(φ) · Ψ(ad_φ)(T_a)` per line 599, then the true Frobenius-pulled-back metric is:

```
G^{pullback}(φ)_{ab} = ⟨D_φ exp[T_a], D_φ exp[T_b]⟩_F
                    = tr((exp(φ) Ψ(ad_φ)(T_a))^⊤ (exp(φ) Ψ(ad_φ)(T_b)))
                    = tr(Ψ(ad_φ)(T_a)^⊤ exp(φ)^⊤ exp(φ) Ψ(ad_φ)(T_b))
```

For φ ∈ so(K) (skew, compact), `exp(φ) ∈ SO(K)` and `exp(φ)^⊤ exp(φ) = I`, so the formula reduces to the chapter's `⟨Ψ(ad_φ)(T_a), Ψ(ad_φ)(T_b)⟩_G`. For general φ ∈ gl(K) (non-compact), `exp(φ)^⊤ exp(φ)` does not simplify, and the formula at line 606 omits this position-dependent factor.

### Issue 5: Equation eq:reverse_beta_grad_phi at line 416

I verified the simplification by direct computation: for agent k's softmax over neighbors with only `K_{ki}` depending on `φ_i`:

```
∂β_{kl}/∂φ_i = β_{kl} · (δ_{li} - β_{ki}) · (-∂K_{ki}/∂φ_i / τ)
Σ_l (∂β_{kl}/∂φ_i) K_{kl} = -(∂K_{ki}/∂φ_i / τ) · Σ_l β_{kl}(δ_{li} - β_{ki}) K_{kl}
                          = -(β_{ki} ∂K_{ki}/∂φ_i / τ) · (K_{ki} - Σ_l β_{kl} K_{kl})
```

The chapter's formula matches this exactly. ✓ No issue.

## User context

The user is conducting a holistic structural audit of all supplementary chapters. This is the fourteenth debate in the gauge-transformer audit series, the third in the supplementary chapter sweep (after §A General Mathematical Framework and §B Covariance Dynamics, both RED_WINS-narrow). The previous two supplementary debates verified the math content and identified editorial / cross-reference / citation gaps. The pattern suggests the same standard applies here.

Load-bearing questions for the judge:

1. Do sub-claims α through ζ each hold under primary-source verification?
2. Are the identified issues (dexp notation inconsistency, Killing-form normalization, cross-reference drift, pullback-metric assumption) editorial-only or do any rise to derivation-incorrect status?
3. Are there additional mathematical or theoretical-purity issues beyond those flagged here that red can identify?

A compound verdict should reflect the worst load-bearing sub-claim. The dexp notation inconsistency (Issue 1) is the most serious — it could affect implementations that copy the integral form expecting the right-trivialized object.
