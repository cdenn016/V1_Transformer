# Red Opening — supplementary-gauge-frame-gradients

## Steelman (opposing position)

Section C of `GL(K)_supplementary.tex` (lines 392–612) is a complete and mathematically pure self-contained derivation of the gauge-frame gradient: §C.1 correctly applies the softmax product rule, §C.2 cites Hall and Gallier for the right-trivialized `dexp_φ` series, §C.2.2 quotes the canonical integral and Higham block-matrix forms, §C.3 chains through the sandwich-product covariance with correct transport derivatives `Q_a Ω μ` and `Q_a Ω Σ Ω^T + Ω Σ Ω^T Q_a^T`, §C.4 gives a defensible compact-vs-non-compact retraction story, and §C.5 lays out a four-stage preconditioning ladder (clip, Cartan project, Cartan-modified Killing metric, position-dependent pullback) that is correctly motivated by the asymmetric growth of `‖dexp_φ‖` along compact vs symmetric directions. The textual mismatches flagged in the evidence pack are editorial — they do not invalidate any derivation that the codebase actually consumes.

## Position

The chapter contains at least three primary-source contradictions in load-bearing equations, plus one broken cross-reference, that elevate its status beyond editorial drift. The most serious is Eq. eq:pullback_metric and its preceding line 599: the formula `d exp_φ(T_a) = exp(φ) · Ψ(ad_φ)(T_a)` with `Ψ(z) = (e^z − 1)/z` is *neither* the left-trivialized *nor* the right-trivialized Fréchet derivative and is numerically wrong by O(1) relative error. The pullback metric Eq. eq:pullback_metric (line 606) is derived from this wrong differential and therefore inherits the error. Compound verdict: the claim that §C is "mathematically and theoretically pure" fails on the §C.5 ¶4 pullback construction (sub-claim ζ).

## Evidence

**Citation A — Hall 2015 §2.7 / Theorem 5.4 and Gallier 2020:**
The canonical Fréchet derivative of `exp: g → G` is
- left-trivialized: `D_φ exp[ξ] = e^φ · ((1 − e^{−ad_φ})/ad_φ)(ξ)`, with generator `ψ_L(z) = (1 − e^{−z})/z = 1 − z/2 + z²/6 − ...`
- right-trivialized: `D_φ exp[ξ] = ((e^{ad_φ} − I)/ad_φ)(ξ) · e^φ`, with generator `ψ_R(z) = (e^z − 1)/z = 1 + z/2 + z²/6 + ...`

These two forms are equivalent because `e^φ` (as a left-multiplication operator) commutes with any analytic function of `ad_φ`, giving the identity `exp(φ) · ψ_L(ad_φ)(ξ) = ψ_R(ad_φ)(ξ) · exp(φ)`. Verified symbolically:
```
sympy: simplify( exp(z)*(1-exp(-z))/z − (exp(z)-1)/z ) = 0
```

**Citation B — `GL(K)_supplementary.tex:599`:**
The chapter writes `d exp_φ(T_a) = exp(φ) · Ψ(ad_φ)(T_a)` with `Ψ(z) = (e^z − 1)/z`. This combines the *right-trivialized* generator `ψ_R` with a *left*-multiplication by `exp(φ)`. It is not the canonical left-trivialized form (which would require `ψ_L`, with alternating signs); it is not the canonical right-trivialized form (which would put `exp(φ)` on the right). Finite-difference verification with `phi, T ~ N(0, 0.5²) ∈ R^{3×3}`, seed 0, central-difference `eps = 1e-7`:

```
||D_true||                                   = 3.3553
||D_true − ψ_R(ad_φ)(T) · exp(φ)||           = 3.66e-9   (right canonical: matches)
||D_true − exp(φ) · ψ_L(ad_φ)(T)||           = 3.66e-9   (left canonical: matches)
||D_true − exp(φ) · ψ_R(ad_φ)(T)||           = 7.018     (manuscript l599: FAILS)
```

The manuscript form is off by O(1) relative to `‖D_true‖`. This is a derivation error, not an editorial drift. Since Eq. eq:pullback_metric (line 606) defines the metric via the Frobenius pullback of this differential, the pullback construction in §C.5 ¶4 inherits a quantitatively wrong differential. The chapter calls this "the most geometrically principled approach" — it is the highest-fidelity preconditioner in the four-stage ladder and the only one labeled "exact (up to series truncation of Ψ)".

**Citation C — Knapp 2002 Proposition 1.93 (Killing form on `sl(K,R)`):**
Direct trace-of-ad computation for `K = 3` (basis `E_{ij}` of `gl(3)`, seed 1):
```
B(X, Y) = tr(ad_X · ad_Y) = −26.723
2K tr(XY) − 2 tr(X) tr(Y) = −26.723   (matches)
```
This confirms the line-585 form of `B` on `gl(K)`. Now compute the Cartan-modified metric `g(X, Y) = −(1/2) B(X, θ(Y))` with `θ(Y) = −Y^T`:
```
g_derived = (1/2) B(X, Y^T) = K tr(X Y^T) − tr(X) tr(Y)        = −10.381
g_line590 = 2K tr(T_a^T T_b) − 2 tr(T_a) tr(T_b)              = −20.761
ratio (line 590 / derivation) = 2.000
```
Eq. eq:killing_metric (line 590) is twice the value derived from the line-587 definition `g(X,Y) = −(1/2) B(X, θ(Y))`. The text at line 589 says "Equivalently, in the generator basis"; the equivalence is broken by a factor of two. The natural gradient `g̃^{−1} ∇F` is scaled by 1/2 relative to the derivation — for an optimizer, this is a learning-rate redefinition, not catastrophic, but the chapter's "equivalently" claim is false.

**Citation D — Higham 2008 §10.2 Eq. 10.15 and Algorithm 10.27:**
Higham's integral and block-matrix identities both give the *full* Fréchet derivative:
```
L_exp(A, E) = ∫_0^1 e^{(1-s)A} E e^{sA} ds = D_A exp[E]
exp([[A, E], [0, A]]) = [[e^A, L_exp(A, E)], [0, e^A]]   ; (1,2)-block is D_A exp[E]
```
Finite-difference confirms the integral equals `D_true` (~1.2e-5 trapezoid error with 200 nodes). But Eq. eq:dexp_integral (line 474) and Eq. eq:dexp_block (line 481) both label this object `dexp_φ(ξ)`, which line 446 defined as the right-trivialized generator `(e^{ad_φ} − I)/ad_φ (ξ)`. These two objects differ by an `e^φ` factor:
```
D_φ exp[ξ]  =  ((e^{ad_φ} − I)/ad_φ)(ξ) · e^φ
```
Reading line 446 then implementing line 474 yields `D_φ exp[ξ]`, not the right-trivialized `dexp_φ` of line 446. The transport derivatives at Eq. eq:dtilde_mu / eq:dtilde_Sigma (lines 535, 537) are written assuming the right-trivialized convention with `D_{φ} exp[T_a] = Q_a · exp(φ)` (line 541), so an implementer who computes `dexp_φ` from the integral and substitutes will absorb an extra `e^φ` into `Q_a` and propagate the error through every transport derivative.

**Citation E — broken cross-reference at line 556:**
The text states `Ω_ij = exp(φ_i) exp(−φ_j)` covers `GL^+(K)` "via polar decomposition (see main text, Section~\ref{sec:glk_lm})." The label `sec:glk_lm` resolves to `GL(K)_attention.tex:2080`, §4.2 "GL(K) Language Modeling: The Full General Model" — an experimental WikiText-103 training section. It contains no polar-decomposition argument. The surjectivity claim (which is non-obvious: while `exp: Sym → SPD` is bijective and `exp: so(K) → SO(K)` is surjective, the *product* `exp(φ_i) exp(−φ_j)` covering all `GL^+(K)` requires a derivation that for any `A = UP ∈ GL^+(K)` one can choose `φ_j` such that `exp(φ_i) = A exp(φ_j)` factors through the image of `exp` from `gl(K)`, which is itself not all of `GL^+(K)` per Culver 1966) is asserted but undefended.

**Citation F — pullback metric (Eq. eq:pullback_metric, line 606) silently restricts to a compact subgroup:**
The true Frobenius-pulled-back metric is
```
G^{pullback}(φ)_{ab} = ⟨D_φ exp[T_a], D_φ exp[T_b]⟩_F
                    = tr( Ψ(ad_φ)(T_a)^T · exp(φ)^T exp(φ) · Ψ(ad_φ)(T_b) )
```
(using the right-trivialized form `D_φ exp[T_a] = Ψ(ad_φ)(T_a) · exp(φ)`, the canonical form; the manuscript's left-multiplication form at line 599 would give the same factor `exp(φ) exp(φ)^T` from the other side). The chapter at line 606 writes `G_{ab}(φ) = ⟨Ψ(ad_φ)(T_a), Ψ(ad_φ)(T_b)⟩_G`, dropping the position-dependent factor `exp(φ)^T exp(φ)`. For `φ ∈ so(K)`, this factor is `I` and the formula is correct. For general `φ ∈ gl(K)` — the entire stated scope of §C.5 — it is not. The natural-gradient preconditioner derived from Eq. eq:pullback_metric therefore lacks the very factor that "automatically compensates the exponential amplification of dexp in the non-compact sector" (line 610). The compensation works only for the compact subalgebra.

## Falsification conditions

This position is wrong if any of the following holds.

1. **Line 599 is a notational shorthand**, not a substantive claim: the chapter intends `exp(φ) · Ψ(ad_φ)(T_a)` to be read as `Ψ(ad_φ)(T_a) · exp(φ)` (i.e., the order is decorative). Refutation: this would require the chapter to state the convention explicitly, and the manuscript at line 597 calls this "the differential of `exp` at `φ` in direction `T_a`" — a definition, not a notational shorthand. The codebase, if it implements `Ψ(ad_φ)(T_a) · exp(φ)` (the correct right-trivialized form), would not match the manuscript formula as written.

2. **The Killing-form factor of 2 is absorbed into an unstated overall scale** of the natural-gradient preconditioner. Refutation: the chapter at line 587 explicitly says "Equivalently, in the generator basis", and the derivation `g(X,Y) = −(1/2) B(X, θ(Y))` is incompatible with the line-590 formula by exactly the missing `1/2`. The claim of equivalence is the load-bearing word.

3. **The cross-reference to `sec:glk_lm` is a typo** for a different label (e.g., a non-existent `sec:polar_decomposition`). Refutation: searching the manuscripts (`grep \label{sec:glk_lm}`) returns exactly one definition, in §4.2 of `GL(K)_attention.tex`. There is no polar-decomposition section elsewhere; the surjectivity argument is simply not in the manuscripts.

4. **The pullback metric formula is intentionally `⟨Ψ T_a, Ψ T_b⟩_F`** as a *coordinate-dependent* approximation, not the true Frobenius pullback. Refutation: the chapter at line 596–597 calls this "the most geometrically principled approach" and "the pullback of the Gram metric through the exponential map" — these phrases promise the *exact* Frobenius pullback of the Gram metric, which includes the `exp(φ)^T exp(φ)` factor for `φ ∈ gl(K)`. Calling it "principled" while silently dropping the factor that handles the non-compact case is what makes the §C.5 ¶4 claim load-bearing.

The chapter's sub-claims α, β (SO(N) Rodrigues), δ, ε (modulo the polar-decomposition cross-reference) hold under primary-source verification. Sub-claims γ (GL(K) integral / block), ζ (Cartan-modified metric, line-599 differential, pullback metric) fail in at least one load-bearing equation each.

Compound verdict recommendation: the worst load-bearing sub-claim is ζ (line 599 + line 606); the verdict on the compound claim is therefore RED_WINS, with the line-599 finite-difference falsification as the single decisive citation.
