# Blue Opening — supplementary-gauge-frame-gradients

## Steelman (opposing position)

§C.5 of `GL(K)_supplementary.tex` contains a numerically-falsifiable formula at line 599 (`d exp_φ(T_a) = exp(φ) · ((e^{ad_φ} − I)/ad_φ)(T_a)`) that disagrees with the canonical Hall/Higham Fréchet derivative by an O(1) error on generic inputs, together with a factor-of-2 inconsistency at line 590, an omitted `exp(φ)^⊤ exp(φ)` factor at line 606, a notational collision in the symbol `dexp_φ` across §C.2 / §C.2.2, and a broken cross-reference at line 556 — so the compound claim of "mathematically/theoretically pure" fails on sub-claim ζ.

## Position

I cannot defend the compound claim as written. The strongest available defense is the **partial** one: sub-claims α (autograd assembly, §C.1), β (dexp series and SO(3) Rodrigues, §C.2 / §C.2.1), γ-integral (GL(K) integral and Higham block-matrix forms, §C.2.2), δ (KL gradient through transport, §C.3), and ε (Lie-algebra updates and SO(N) principal-ball retraction, §C.4) survive primary-source verification. Sub-claim ζ (Cartan decomposition and the four preconditioning modes, §C.5) does not survive: one formula at line 599 is empirically wrong, one at line 590 is off by a factor of 2 from its own line-587 definition, and one at line 606 omits a non-compact factor. The supporting issues (Issue 1 notation drift, Issue 3 broken xref) are editorial. The verdict the evidence supports is RED_WINS scoped to specific §C.5 formula fixes, not a chapter-wide rewrite.

## Evidence

### Sub-claim α — verified [sympy]

The reverse-direction softmax simplification at Eq. `eq:reverse_beta_grad_phi` (lines 416–421) reduces to zero residual under direct algebraic expansion using `∂β_kl/∂z = β_kl(δ_kl − β_kk)` per [Bishop2006 §4.3.4 / §5.4.2]:

```
LHS - RHS = 0   (sympy.simplify on N=4 generic symbols, K_l, β_l, dK_{ki}, τ)
```

The chapter's formula correctly captures both the diagonal product `−β_{ki} K_{ki}` and the centering term `β_{ki} ⟨K_k⟩_β`. No issue.

### Sub-claim β — verified [sympy + canon]

The series `dexp_φ(ξ) = Σ ad_φⁿ(ξ)/(n+1)! = (e^{ad_φ} − I)/ad_φ (ξ)` at Eq. `eq:dexp_series` (line 446) matches the right-trivialized form in [Hall2015 §2.7 / Theorem 5.4 p. 110; Gallier2020 Vol. 2 §1.5; Iserles-Nørsett-Munthe-Kaas Acta Numerica 2000 §2.3]. The SO(3) Rodrigues specialization at Eq. `eq:dexp_so3` (lines 456–465) follows from `ad_φ³ = −θ² ad_φ` for skew-symmetric φ with norm θ, verified symbolically against a generic skew tangent X:

```
ad_φ(ad_φ(ad_φ(X))) + θ² ad_φ(X)
  = sympy.zeros(3, 3)   (verified on (a, b, c) parameterization of so(3))
```

The Taylor expansions of `c_1(θ) = (1 − cos θ)/θ²` and `c_2(θ) = (θ − sin θ)/θ³` at line 467 are textbook. No issue.

### Sub-claim γ-integral — verified [Higham2008]

The integral form `dexp_φ(ξ) = ∫₀¹ e^{tφ} ξ e^{(1−t)φ} dt` at Eq. `eq:dexp_integral` (line 473) is the Fréchet derivative `L_exp(φ, ξ) = D_φ exp[ξ]` of [Higham2008 §10.2 Eq. 10.15, Theorem 10.13]. The (2K × 2K) block-matrix identity at Eq. `eq:dexp_block` (lines 480–484) is [Higham2008 Algorithm 10.27]. **Both formulas are individually correct**; the issue is the symbol `dexp_φ` being applied to objects that differ by a factor of `e^φ` from line 446 (see Concession A below).

### Sub-claim δ — verified [finite-difference, K=3]

With `Q_a^(i) ≡ dexp_{φ_i}(T_a)` defined at line 541 by the right-trivialized identity `D_{φ_i} exp[T_a] = Q_a^(i) · exp(φ_i)`, the chain rule applied to `Ω_ij = exp(φ_i) exp(−φ_j)` gives `∂Ω_ij/∂φ_i^a = Q_a^(i) Ω_ij`. Finite-difference verification on K=3 random `φ_i`, `φ_j`, `T_a`, `Σ_j`:

```
||Q_a^(i) Ω_ij − FD(∂Ω/∂φ_i^a)||                            = 1.20e-10
||Q_a^(i) Ω_ij Σ_j Ω_ij^⊤ + Ω_ij Σ_j Ω_ij^⊤ (Q_a^(i))^⊤
   − FD(∂(Ω Σ_j Ω^⊤)/∂φ_i^a)||                              = 2.08e-9
```

Eqs. `eq:dtilde_mu` and `eq:dtilde_Sigma` (lines 533–539) are internally consistent with the right-trivialized convention. The sandwich-product covariance derivative obeys `(AB)^⊤ = B^⊤ A^⊤` as required by the CLAUDE.md hard constraint `Σ_transported = Ω Σ Ω^⊤`.

### Sub-claim ε — verified [canon]

Lie-algebra updates in g (a vector space) at Eq. `eq:phi_update` (lines 547–550) are standard. The SO(N) principal-ball retraction `‖φ‖ < π − ε` at line 554 handles the periodicity of `exp: so(N) → SO(N)` via the standard antipodal identification `exp(φ) = exp(φ − 2π n̂)` for SO(3) [Hall2015 §5.1]. For GL(K), the chapter cites `culver1966existence` (line 556) to acknowledge that `exp: gl(K) → GL⁺(K)` is not surjective, and argues that the two-exponential product `Ω_ij = exp(φ_i) exp(−φ_j)` extends coverage. The two-exponential extension is principled, though the cited support for the surjectivity claim is broken (Issue 3 below).

### Sub-claim ζ — partially refuted [finite-difference, sympy]

Three §C.5 formulas do not survive verification.

**Line 599 (most serious).** The chapter writes `d exp_φ(T_a) = exp(φ) · Ψ(ad_φ)(T_a)` with `Ψ(z) = (e^z − 1)/z`. The canonical Fréchet-derivative forms are [Hall2015 Theorem 5.4; Iserles-Nørsett-Munthe-Kaas 2000 §2.3]:

```
LEFT-trivialized:   D_φ exp[ξ] = exp(φ) · ((1 − e^{−ad_φ})/ad_φ)(ξ)
RIGHT-trivialized:  D_φ exp[ξ] = ((e^{ad_φ} − I)/ad_φ)(ξ) · exp(φ)
```

The chapter combines the LEFT placement `exp(φ) · (·)` with the RIGHT generator `(e^z − 1)/z`. This is neither standard form. Finite-difference verification on K=3, ‖φ‖ ≈ 0.7:

```
||exp(φ)·((e^{ad_φ}−I)/ad_φ)(T_a) − FD(D_φ exp[T_a])||  = 1.46    (line 599)
||((e^{ad_φ}−I)/ad_φ)(T_a)·exp(φ) − FD(D_φ exp[T_a])||  = 1.02e-10 (canonical right)
||exp(φ)·((1−e^{−ad_φ})/ad_φ)(T_a) − FD(D_φ exp[T_a])|| = 1.02e-10 (canonical left)
```

The line-599 form is off by an O(1) factor; the fix is to replace `Ψ(z) = (e^z − 1)/z` with `Ψ(z) = (1 − e^{−z})/z`, or to move `exp(φ)` to the right of `Ψ(ad_φ)(T_a)`.

**Line 590 (factor-of-2 normalization).** The chapter defines `g(X, Y) = −(1/2) B(X, θ(Y))` at line 587 with `θ(X) = −X^⊤` and `B(X, Y) = 2K tr(XY) − 2 tr(X) tr(Y)` [Knapp2002 Prop. 1.93 for B on sl(K); chapter extends to gl(K)]. Direct sympy computation:

```
g(X, Y) = −(1/2) B(X, −Y^⊤) = (1/2)[2K tr(X Y^⊤) − 2 tr(X) tr(Y^⊤)]
       = K tr(X Y^⊤) − tr(X) tr(Y)
sympy.simplify(g − (K tr(X Y^⊤) − tr(X) tr(Y))) = 0
```

Eq. `eq:killing_metric` (line 590) writes `g̃_{ab} = 2K tr(T_a^⊤ T_b) − 2 tr(T_a) tr(T_b)`, which is exactly `2g`. Either the symbol `g̃` denotes `2g` (and the chapter omits the line-587 prefactor `−1/2` when going to the generator basis), or the formula at line 590 is wrong by a factor of 2.

**Line 606 (omitted non-compact factor).** The Frobenius pullback through `D_φ exp` is

```
G^{pullback}_{ab}(φ) = ⟨D_φ exp[T_a], D_φ exp[T_b]⟩_F
                    = tr(Ψ(ad_φ)(T_a)^⊤ · exp(φ)^⊤ exp(φ) · Ψ(ad_φ)(T_b))
```

Line 606 writes `G_{ab}(φ) = ⟨Ψ(ad_φ)(T_a), Ψ(ad_φ)(T_b)⟩_G`, omitting the `exp(φ)^⊤ exp(φ)` factor. For φ ∈ so(K), this factor is the identity and the formula is exact; for φ ∈ gl(K) (the §C.5 target), `exp(φ)^⊤ exp(φ) = exp(φ + φ^⊤ + O([φ, φ^⊤]))` does not simplify, and the formula is only the pullback of the metric `⟨·, ·⟩_G` along the **right-trivialized** generator `Ψ(ad_φ)`, not the Frobenius pullback along `D_φ exp`. The chapter does not disambiguate which metric is being pulled back.

### Cartan decomposition itself — verified [canon]

`gl(K) = so(K) ⊕ Sym(K) ⊕ R` at Eq. `eq:cartan_decomposition` (lines 563–565) is the standard decomposition with respect to the Cartan involution `θ(X) = −X^⊤` per [Helgason1978 Ch. III; Knapp2002 §VI.2]. The Cartan projector `P_sym = (1/2) G⁻¹(G + S)` at line 579 follows from `T_a = (1/2)(T_a − T_a^⊤) + (1/2)(T_a + T_a^⊤)` projected onto the symmetric component. The four-mode ladder (norm clipping, Cartan projector, Killing-form metric, pullback natural gradient) is a principled progression of geometric fidelity per [Amari1998 §4; Iserles-Nørsett-Munthe-Kaas 2000].

The ζ refutation is **scoped**: the Cartan decomposition, the projector, the involution, and the four-mode structure are correct. The three failing items are the specific formulas at lines 590, 599, and 606. These are formula errors, not conceptual ones — the fix for each is a single equation rewrite.

## Concessions (to be granted to red regardless of opening)

**Concession A (Issue 1 — dexp notation drift, lines 446 / 450 / 471 / 473 / 480).** The symbol `dexp_φ` denotes the right-trivialized generator `(e^{ad_φ} − I)/ad_φ (·)` at line 446, then is reused at line 473 for the Fréchet integral `∫₀¹ e^{tφ} ξ e^{(1−t)φ} dt` and at line 480 for the Higham block-matrix entry `L_exp(φ, ξ)`. These two objects differ by a factor of `exp(φ)`. The chapter never reconciles. This is substantive editorial drift that could mislead an implementer copying the integral form expecting the right-trivialized object. Fix: pick one convention and rewrite all four equation labels.

**Concession B (Issue 3 — cross-reference to `sec:glk_lm`, line 556).** The label `sec:glk_lm` resolves to `GL(K)_attention.tex:2080` "§4.2 GL(K) Language Modeling: The Full General Model" — an experimental section, not a polar-decomposition argument. The polar-decomposition surjectivity claim is true ([Higham2008 §8] + `exp: so(K) → SO(K)` surjective + `exp: Sym → Sym_{++}` bijective ⇒ `exp(φ_i) exp(−φ_j)` covers `O(K) · Sym_{++} = GL⁺(K)` via polar decomposition `A = UP`), but the chapter does not derive it and the cited support is broken. Fix: either replace the xref with an inline derivation in §C.4, or cite `[Higham2008 §8]` directly.

## Falsification conditions

This blue position is wrong if any of the following is shown:

1. **Falsifies sub-claim α.** If red shows the softmax simplification at Eq. `eq:reverse_beta_grad_phi` does not reduce to zero residual under direct expansion (e.g., a counterexample with explicit `β`, `K`, `dK`, `τ`), or shows that the chapter's autograd assembly misses a term that the envelope-theorem treatment of main paper §3.5 actually preserves.

2. **Falsifies sub-claim β.** If red shows the right-trivialized form `(e^{ad_φ} − I)/ad_φ (ξ)` does not match the SO(3) Rodrigues coefficients `c_1(θ) = (1 − cos θ)/θ²`, `c_2(θ) = (θ − sin θ)/θ³` (e.g., by deriving different coefficients from `ad_φ³ = −θ² ad_φ`), or shows that the Taylor expansion `c_1 ≈ 1/2 − θ²/24` at line 467 is wrong.

3. **Falsifies sub-claim δ.** If red shows the transport derivatives `∂μ̃_ij/∂φ_i^a = Q_a^(i) Ω_ij μ_j`, `∂Σ̃_ij/∂φ_i^a = Q_a^(i) Ω_ij Σ_j Ω_ij^⊤ + Ω_ij Σ_j Ω_ij^⊤ (Q_a^(i))^⊤` are not the right-trivialized chain-rule derivatives (e.g., shows they require an `exp(φ_i)` factor that the chapter has absorbed wrongly), or that the sandwich-product covariance derivative is incorrect under the project's CLAUDE.md convention.

4. **Falsifies sub-claim ε.** If red shows the principal-ball retraction `‖φ‖ < π − ε` does not handle SO(N) periodicity, or that the two-exponential product `Ω_ij = exp(φ_i) exp(−φ_j)` provably does NOT cover `GL⁺(K)`. The standard polar-decomposition argument supports coverage; if red cites a counterexample in `GL⁺(K)` not reachable as a product of two `exp` images, sub-claim ε fails.

5. **Strengthens ζ refutation beyond the three formulas.** If red shows that sub-claim ζ fails in a way **broader** than the three line-590 / line-599 / line-606 formula errors — e.g., that the Cartan decomposition itself is misstated, or the Cartan projector `P_sym = (1/2) G⁻¹(G + S)` is derivation-incorrect, or the four-mode structure is internally inconsistent — then the scoped action ladder proposed below is insufficient and a wider §C.5 rewrite is warranted.

6. **Refutes Concession A.** If red shows the chapter intends `dexp_φ` to denote a single consistent object (e.g., always the Fréchet derivative throughout §C.2 and §C.2.2, with line 446 stating an identity rather than a definition), then Issue 1 reduces to a typesetting clarification rather than substantive drift. I do not see this reading in the current text — line 446 introduces `dexp_φ` as the **definition** of a single symbol via `=` — but I grant the falsification path.

7. **Refutes the polar-decomposition fix for Concession B.** If red shows the polar-decomposition surjectivity argument `exp · exp` over gl(K) does NOT cover `GL⁺(K)` (e.g., shows that the Sym_{++} × SO(K) product structure does not lift through the two-exponential product), then sub-claim ε fails along with Concession B's fix.

## Proposed scoped action ladder (if the judge rules RED_WINS on ζ)

1. Replace line 599 with one of the canonical forms:
   - `d exp_φ(T_a) = exp(φ) · ((1 − e^{−ad_φ})/ad_φ)(T_a)` (left-trivialized; replaces Ψ), OR
   - `d exp_φ(T_a) = ((e^{ad_φ} − I)/ad_φ)(T_a) · exp(φ)` (right-trivialized; moves exp(φ)).

2. Update the line-606 pullback metric to include the non-compact factor:
   `G_{ab}(φ) = tr(Ψ(ad_φ)(T_a)^⊤ · exp(φ)^⊤ exp(φ) · Ψ(ad_φ)(T_b))`,
   or explicitly restrict §C.5 mode 4 to the so(K) directions where `exp(φ)^⊤ exp(φ) = I`.

3. Reconcile line 590 with line 587 — either drop the `−1/2` prefactor in the definition or add the corresponding factor of `1/2` to `g̃_{ab}`.

4. Replace the broken `sec:glk_lm` xref at line 556 with `[Higham2008 §8]` plus a one-paragraph polar-decomposition argument.

5. Unify the `dexp_φ` symbol across §C.2 and §C.2.2: choose either the right-trivialized form (Eqs. eq:dexp_series, eq:dtilde_mu, eq:dtilde_Sigma) or the Fréchet form (Eqs. eq:dexp_integral, eq:dexp_block); rewrite the inconsistent equations and label clearly which trivialization is in use.

These are **five scoped equation/citation fixes**, not a chapter rewrite. The math content of α, β, γ-integral, δ, ε, the Cartan decomposition, and the four-mode ladder all survive.
