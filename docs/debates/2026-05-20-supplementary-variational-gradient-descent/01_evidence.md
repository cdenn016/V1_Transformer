# Evidence Pack — supplementary-variational-gradient-descent

## Section structure

`Attention/GL(K)_supplementary.tex` §Variational Gradient Descent spans lines 617–670, with `\label{app:variational_descent}` at line 618.

- §D.1 Natural Gradient Descent on the Gaussian Manifold (lines 621–633).
- §D.2 Manifold Retraction for Covariance Matrices (lines 635–650; label `sec:retraction` at line 635).
- §D.3 Gauge Frame Dynamics on GL(K) (lines 652–670).

## Key equations

### §D.1 — natural-gradient projections

Eq. (623–626):
```
δμ = -Σ ∇_μ,    δΣ = -2 Σ sym(∇_Σ) Σ
```
with `sym(M) = (1/2)(M + M^⊤)`. Citations: `\citep{Cencov1982,amari2016information}`.

Whitening Eq. (631):
```
B = Λ^{-1/2} (V^⊤ δΣ V) Λ^{-1/2}
```
where `Σ = V Λ V^⊤`. Frobenius-norm trust-region clip with max relative step `ρ` (no specific value given in §D, but the BlockConfig in code uses `e_sigma_q_trust = 5.0` per CLAUDE.md).

### §D.2 — affine-invariant SPD retraction

Eq. (637–639):
```
Σ_{k+1} = V Λ^{1/2} U exp(τ Λ_B) U^⊤ Λ^{1/2} V^⊤
```
where `B = U Λ_B U^⊤` is the eigendecomposition of the whitened tangent. Component-wise clipping `|τ λ_B^{(j)}| ≤ 50`.

Post-retraction safeguards (lines 642–650):
- `sanitize_sigma` symmetrizes and checks for NaNs.
- Spectral floor `ε_SPD = 1e-4`.
- Condition cap `κ_max = 1e4`.

### §D.3 — gauge-frame differential

Eq. (657) left-trivialised:
```
∂/∂φ^a exp(Σ_b φ^b T_b) = exp(X) · Q_a^L
```
with `X = Σ_b φ^b T_b` and `Q_a^L = dexp^L_X(T_a)`.

Trivialization relation (line 659):
```
Q_a^R · exp(φ) = exp(φ) · Q_a^L,    Q_a^L = Ad_{exp(-φ)}(Q_a^R)
```
Citation: `\citep{Hall2015}`.

Eq. (661) SO(3) closed form:
```
Q_a^L = T_a - c_1(θ) ad_X(T_a) + c_2(θ) ad_X²(T_a)
```
with `θ = ‖φ‖`, `c_1(θ) = (1-cos θ)/θ²`, `c_2(θ) = (θ-sin θ)/θ³`. Taylor-expanded small-angle limits for `θ < 1e-4`.

Cross-reference at line 663: "The right-trivialised form used in Appendix~C carries +c_1(θ) ad_X in place of -c_1(θ) ad_X; both formulas reproduce the numerical Fréchet derivative to machine precision under their own conventions."

### §D.3 preconditioner cross-reference at line 665

```
(i) Cartan-involution-modified natural gradient — uses g̃_ab = 2K tr(T_a^⊤ T_b) - 2 tr(T_a) tr(T_b);
    (positive-definite on sl(K) as derived in App.~C.3)
(ii) Cartan decomposition preconditioner — dampens non-compact symmetric directions;
(iii) full pullback natural gradient G_ab(φ) = ⟨Ψ(ad_X)(T_a), Ψ(ad_X)(T_b)⟩, Ψ(z) = (e^z-1)/z.
```

## Canonical-reference verification

### Sub-claim α: Fisher-Rao natural-gradient projections

Derivation from standard sources:

- **[Amari 2016 "Information Geometry and Its Applications" §2.3]**: Fisher information matrix for Gaussian `N(μ, Σ)` has block-diagonal structure `F = blockdiag(F_μ, F_Σ)` with `F_μ = Σ^{-1}` and `F_Σ` (on the tangent space of SPD) such that `g(δΣ_1, δΣ_2) = (1/2) tr(Σ^{-1} δΣ_1 Σ^{-1} δΣ_2)`.

- **Natural gradient for μ**: `nat_grad_μ = (F_μ)^{-1} ∇_μ = Σ ∇_μ`. For descent: `δμ = -Σ ∇_μ`. ✓ matches Eq. (623).

- **Natural gradient for Σ**: solve `g(nat_grad_Σ, v) = ⟨∇_Σ, v⟩` for all symmetric `v`. Setting `nat_grad_Σ = 2 Σ sym(∇_Σ) Σ`:
  `g(2 Σ sym(∇_Σ) Σ, v) = (1/2) tr(Σ^{-1} · 2 Σ sym(∇_Σ) Σ · Σ^{-1} v) = tr(sym(∇_Σ) v) = tr(∇_Σ v)` (when v is symmetric).
  For descent: `δΣ = -2 Σ sym(∇_Σ) Σ`. ✓ matches Eq. (623).

  Standard refs: [Amari 2016 §4.3 "Riemannian gradient on Gaussian"; Smith 2005 "Covariance, subspace, and intrinsic Cramér-Rao bounds"].

### Sub-claim β: whitening and trust region

The affine-invariant whitening at Eq. (631) is the standard transport into the eigenbasis. Verification: `B` represents `Σ^{-1/2} δΣ Σ^{-1/2}` in the eigenbasis, which is the canonical "horizontal" tangent at Σ ∈ SPD(K) per [Pennec 2006]. Affine invariance: under `Σ → A Σ A^⊤` for invertible `A`, the whitened tangent transforms as `B → B` (invariant).

The trust-region Frobenius-norm clip on `B` is a standard SPD-manifold safety per [Absil-Mahony-Sepulchre 2008 "Optimization Algorithms on Matrix Manifolds"].

### Sub-claim γ: affine-invariant SPD retraction

Derivation: starting from `Σ_k = V Λ V^⊤` and the whitened tangent `B = U Λ_B U^⊤`, the affine-invariant exponential map at Σ_k is:

```
exp_Σ_k(δΣ) = Σ_k^{1/2} · exp(Σ_k^{-1/2} δΣ Σ_k^{-1/2}) · Σ_k^{1/2}
```

with `Σ_k^{1/2} = V Λ^{1/2} V^⊤` and `Σ_k^{-1/2} δΣ Σ_k^{-1/2} = V B V^⊤` (where B is the whitened tangent at line 631):

```
exp_Σ_k(τ δΣ) = V Λ^{1/2} V^⊤ · V exp(τ B) V^⊤ · V Λ^{1/2} V^⊤
             = V Λ^{1/2} exp(τ B) Λ^{1/2} V^⊤
             = V Λ^{1/2} U exp(τ Λ_B) U^⊤ Λ^{1/2} V^⊤    (using exp(U Λ_B U^⊤) = U exp(Λ_B) U^⊤)
```

Matches Eq. (637–639) exactly. ✓ Canonical refs:
- **[Pennec 2006 "Intrinsic statistics on Riemannian manifolds"]**: affine-invariant metric on SPD.
- **[Bhatia 2007 "Positive Definite Matrices" §6.1 "The Riemannian metric"]**: explicit retraction formula.
- **[Absil-Mahony-Sepulchre 2008 §5.4.6 "The set of symmetric positive-definite matrices"]**: SPD exponential map.

SPD-preserving by construction since `exp(τ Λ_B) ≻ 0` (positive eigenvalues) and the sandwich `V Λ^{1/2} · (·) · Λ^{1/2} V^⊤` preserves positive-definiteness.

### Sub-claim δ: SO(3) closed-form Ψ_L

Derivation from `Ψ_L(z) = (1 - e^{-z})/z = Σ_k (-z)^k / (k+1)!`:

```
Ψ_L(ad_X) = I - ad_X/2 + ad_X²/6 - ad_X³/24 + ad_X⁴/120 - ...
```

For SO(3) with `‖φ‖ = θ` and `ad_X³ = -θ² ad_X` (verified by direct computation on so(3) generators):

```
ad_X³ = -θ² ad_X
ad_X⁴ = ad_X · ad_X³ = -θ² ad_X²
ad_X⁵ = ad_X · ad_X⁴ = -θ² ad_X³ = θ⁴ ad_X
...
```

So the series truncates:
```
Ψ_L(ad_X) = I + ad_X · (-1/2 + θ²/24 - θ⁴/720 + ...) + ad_X² · (1/6 - θ²/120 + θ⁴/5040 - ...)
          = I - c_1(θ) ad_X + c_2(θ) ad_X²
```

with `c_1(θ) = 1/2 - θ²/24 + θ⁴/720 - ... = (1 - cos θ)/θ²` ✓
and `c_2(θ) = 1/6 - θ²/120 + θ⁴/5040 - ... = (θ - sin θ)/θ³` ✓

So `Ψ_L(ad_X)(T_a) = T_a - c_1(θ) ad_X(T_a) + c_2(θ) ad_X²(T_a)`. ✓ Matches Eq. (661). Canonical ref: [Hall 2015 §2.7].

### Sub-claim ε: trivialization relation

Identity `Ad_{exp(X)} = e^{ad_X}` per [Hall 2015 Proposition 2.25]. Then:
```
Ad_{exp(-X)} · (e^{ad_X} - I)/ad_X = e^{-ad_X} · (e^{ad_X} - I)/ad_X
                                   = (I - e^{-ad_X})/ad_X = Ψ_L(ad_X)
```
✓ Matches line 659.

### Sub-claim ζ: preconditioner cross-references

The three modes at line 665:

- **(i) Cartan-involution-modified**: `g̃_ab = 2K tr(T_a^⊤ T_b) - 2 tr(T_a) tr(T_b)`. This matches the §C.5 line 590 formula (post my Edit 3 from the §C debate). The reference "App.~C.3" is wrong; correct location is `App.~C.5` (label `sec:glk_preconditioning` at line 559).
- **(ii) Cartan decomposition**: dampens non-compact directions. Matches §C.5 mode 2 at lines 575–582.
- **(iii) Pullback natural gradient**: stated formula `G_ab(φ) = ⟨Ψ(ad_X)(T_a), Ψ(ad_X)(T_b)⟩` is the OLD uncorrected formula. The corrected formula at §C.5 line 606 (post my Edit 2 from the §C debate) is `G_ab(φ) = tr(Ψ(ad_φ)(T_a)^⊤ · exp(φ) exp(φ)^⊤ · Ψ(ad_φ)(T_b))`, including the position-dependent `exp(φ) exp(φ)^⊤` factor that is essential for non-compact GL(K). §D.3 line 665 still propagates the old form.

## Bib verification

- `Cencov1982` exists at `references.bib:1833`. ✓
- `amari2016information`: verify presence.
- `Hall2015` exists. ✓
- `Higham2008` exists. ✓
- `Pennec2006`: not yet verified.
- `Bhatia2007`: not yet verified.

## Canon excerpts — external standards

### Natural gradient on Gaussian

- [Amari 2016 §2.3]: Fisher metric on Gaussian.
- [Amari 2016 §4.3]: natural gradient on parametric statistical manifolds.
- [Bishop 2006 §10.7]: natural gradient.
- [Sun-Marchand-Reilly 2014 "On the natural gradient" — clear derivation of `2 Σ sym(∇) Σ`].

### SPD manifold geometry

- [Pennec 2006 "Intrinsic statistics on Riemannian manifolds"]: affine-invariant metric, exponential map, inverse Christoffel symbols.
- [Bhatia 2007 "Positive Definite Matrices" §4.1, §6.1]: SPD Riemannian geometry.
- [Absil-Mahony-Sepulchre 2008 "Optimization Algorithms on Matrix Manifolds" §5.4.6]: SPD retractions.
- [Smith 2005 "Covariance, subspace, and intrinsic Cramér-Rao bounds"]: SPD Hessian forms.

### Lie group / dexp

- [Hall 2015 §2.7 / Theorem 5.4]: left/right trivializations of D exp.
- [Iserles-Nørsett-Munthe-Kaas Acta Numerica 2000 §2.3]: Lie-group methods, explicit dexp series.
- [Higham 2008 §10.2, Algorithm 10.27]: Fréchet derivative computation.

## What this evidence does NOT settle

1. **Whether the stale pullback formula at line 665 (Issue 1) is sufficient to fail the compound claim.** It is a real correctness issue — the formula is now inconsistent with the corrected §C.5 form. Whether this rises above the editorial threshold depends on whether the formula is just a forward reference (label-only consistency issue) or whether it's load-bearing on its own (e.g., implementations consuming the §D.3 formula directly would inherit the same non-compact bug we just fixed in the codebase).

2. **Whether the cross-reference "App.~C.3" (Issue 2) is sufficient to fail the compound claim.** Same pattern as prior narrow verdicts.

3. **Whether the absence of [Pennec 2006] / [Bhatia 2007] / [Absil-Mahony-Sepulchre 2008] citations (Issue 3) for the SPD retraction is an editorial gap comparable to prior §B/§C citation gaps.**

4. **Whether the cumulative editorial scope of Issues 1-6 is comparable to the prior narrow verdicts (3-5 gaps) or substantive verdicts.** Issue 1 (stale pullback formula) is the most consequential — it's a forward-propagated bug, not just a citation gap.

5. **Whether the math content of sub-claims α-ε survives. Pre-debate verification suggests yes, but the agents should verify independently.**
