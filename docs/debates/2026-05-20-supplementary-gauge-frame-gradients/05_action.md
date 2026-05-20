# Action — supplementary-gauge-frame-gradients

**From verdict:** RED_WINS (substantive, scoped to §C.5)

## Summary of verdict

The compound claim that `\section{Gauge Frame Gradients}` of `Attention/GL(K)_supplementary.tex` (lines 392–612) is "complete and mathematically/theoretically pure" fails on sub-claim ζ (§C.5 Gauge Frame Preconditioning for GL(K)). This debate's verdict differs in KIND from the prior two supplementary debates (§A General Mathematical Framework, §B Covariance Dynamics, both RED_WINS-narrow on editorial gaps): here, three mathematical formula errors are independently FD-verified by both teams.

**Math sub-claims verified by both sides:**
- α (autograd gradient assembly, §C.1): symbolic residual = 0.
- β (dexp series + SO(3) Rodrigues, §C.2 / §C.2.1): `ad_φ³ + θ² ad_φ = 0` confirmed; series truncation matches the Rodrigues coefficients exactly.
- γ-integral (Higham forms, §C.2.2): canonical per [Higham 2008 §10.2, Algorithm 10.27] — though the symbol `dexp_φ` is misapplied across §C.2 / §C.2.2.
- δ (KL gradient through transport, §C.3): FD-verified at 1e-10. Transport derivatives consistent with right-trivialized convention.
- ε (retractions, §C.4): principled; cross-reference at line 556 to `sec:glk_lm` is broken.

**Sub-claim ζ fails on three FD-verified formula errors:**

1. **Line 599** `d exp_φ(T_a) = exp(φ) · Ψ(ad_φ)(T_a)` with `Ψ(z) = (e^z-1)/z` is **mathematically wrong**. The relative error against the true Fréchet derivative is 2.09. The canonical Hall 2015 forms (left-trivialized with `(1-e^{-z})/z`, or right-trivialized with `exp(φ)` placed AFTER `Ψ(ad_φ)(T_a)`) match to 3.66e-9. The chapter combines left placement with right-trivialized generator — not either canonical form.

2. **Line 590** Killing-metric formula `g̃_{ab} = 2K tr(T_a^⊤ T_b) - 2 tr(T_a) tr(T_b)` is off by exactly 2× relative to the value derived from line 587's definition `g(X, Y) = -(1/2) B(X, θ(Y))` with `B(X, Y) = 2K tr(XY) - 2 tr(X) tr(Y)` per [Knapp 2002 Prop 1.93]. The mode-3 Killing-metric preconditioner output is therefore off by a factor of 1/2 if the line-590 formula is used directly.

3. **Line 606** pullback metric omits the `exp(φ)^⊤ exp(φ)` factor that survives for non-compact `φ ∈ gl(K)`. Blue's FD verification: the corrected form `tr(Ψ(ad_φ)(T_a)^⊤ · exp(φ) exp(φ)^⊤ · Ψ(ad_φ)(T_b))` matches the true Frobenius pullback at relative error 2.77e-16; the line-606 formula (which omits the factor) has relative error 1.35 at `‖φ‖ ≈ 0.3`. Since §C.5 is titled "Gauge Frame Preconditioning for GL(K)" (non-compact), the omission is wrong in the chapter's stated scope.

The line-599 and line-606 errors cascade: line 606 is constructed from line 599's wrong differential, so fixing line 599 changes what line 606 must say.

## Recommended action

Three mathematical revisions and two editorial unifications to `Attention/GL(K)_supplementary.tex` §C. No equation changes outside §C.5 and the §C.2 / §C.2.2 / §C.4 cross-references.

### Edit 1 — line 599 (mathematical revision)

Replace `d exp_φ(T_a) = exp(φ) · Ψ(ad_φ)(T_a)` with the right-trivialized canonical form (matching §C.3's convention via Q_a^(i) at line 541):

```
D_{φ} exp[T_a] = Q_a^{(φ)} · exp(φ),  where Q_a^{(φ)} = ((e^{ad_φ} - I)/ad_φ)(T_a)
```

Cite [Hall 2015 §2.7 / Theorem 5.4].

### Edit 2 — line 590 (mathematical revision)

Reconcile the formula with the line-587 definition. Two options:

(a) Keep line 587's `g(X, Y) = -(1/2) B(X, θ(Y))` definition; replace line 590 with `g_{ab} = K tr(T_a^⊤ T_b) - tr(T_a) tr(T_b)` (dropping the factor of 2).

(b) Drop the `-(1/2)` prefactor from line 587 to make it `g(X, Y) = -B(X, θ(Y))`; keep line 590 as `g̃_{ab} = 2K tr(T_a^⊤ T_b) - 2 tr(T_a) tr(T_b)`.

Either option resolves the inconsistency. Cite [Knapp 2002 Prop 1.93].

### Edit 3 — line 606 (mathematical revision)

Either (a) restore the missing factor for the non-compact case:

```
G_{ab}(φ) = tr(Ψ(ad_φ)(T_a)^⊤ · exp(φ)^⊤ exp(φ) · Ψ(ad_φ)(T_b))
```

or (b) restrict the formula to the compact subgroup (skew-symmetric φ ∈ so(K)) and rename §C.5 to flag the restriction. Option (a) preserves the chapter's stated GL(K) scope.

### Edit 4 — §C.2 / §C.2.2 dexp_φ notation unification (editorial)

The symbol `dexp_φ` is used for two distinct objects:
- Lines 446–450 (§C.2): the right-trivialized generator `(e^{ad_φ} - I)/ad_φ (·)`.
- Lines 473, 480 (§C.2.2): the full Fréchet derivative `D_φ exp[·]` via Higham integral and block-matrix forms.

These differ by a factor of `exp(φ)`. Unify by selecting one convention — most naturally the right-trivialized form at line 446 — and rename the §C.2.2 integral and block-matrix forms with the canonical [Higham 2008] notation `L_exp(φ, ξ)` or `D_φ exp[ξ]`. The relation `D_φ exp[ξ] = dexp_φ(ξ) · exp(φ)` should be stated once and used consistently.

### Edit 5 — line 556 cross-reference (editorial)

Replace `\ref{sec:glk_lm}` (which resolves to main paper §4.2 experimental section at `GL(K)_attention.tex:2080`, NOT a polar-decomposition derivation) with a direct citation to [Higham 2008 §8] plus a short inline argument:

> For any `A ∈ GL^+(K)` with polar decomposition `A = UP` (`U ∈ SO(K)`, `P ∈ Sym_{++}`), choose `φ_i ∈ gl(K)` with `exp(φ_i) = UP^{1/2}` and `φ_j` with `exp(-φ_j) = P^{1/2}` (equivalently `exp(φ_j) = P^{-1/2}`); both factorizations exist because `exp` is surjective onto `SO(K)` from `so(K)` and bijective from `Sym(K)` to `Sym_{++}` per [Higham 2008 §8].

## Optional code-mode follow-up

The verdict identifies a potential code-implementation question: whether `transformer/aif/` or `transformer/vfe/` consumes the line-606 formula directly without the `exp(φ)^⊤ exp(φ)` correction. If so, the natural-gradient preconditioner in mode-4 inherits the relative error 1.35 at moderate `‖φ‖`. This is a code-mode follow-up debate / audit, not a theory remand. The user may wish to verify whether the implementation uses the verbatim line-606 formula or derives the preconditioner separately.

## Cumulative debate-series state

Fourteenth debate in the gauge-transformer audit series. Closed queue:

1. §5 transformer reduction (RED_WINS).
2. Softmax-β stationarity (RED_WINS).
3. Sub-claim A flat bundle (BLUE_WINS — §5 reduction sub-claim).
4. Sub-claim B degenerate Σ (BLUE_WINS).
5. Sub-claim C QK^T identification (BLUE_WINS).
6. Sub-claim D V identification (BLUE_WINS).
7. Canonical F vs surrogate (RED_WINS).
8. Multi-head block-diagonal (BLUE_WINS).
9. Route 1 untied carving (RED_WINS).
10. FFN softmax-gradient correction (RED_WINS).
11. §3 Gauge-Covariant VFE (RED_WINS narrow).
12. Supplementary §A General Mathematical Framework (RED_WINS narrow).
13. Supplementary §B Covariance Dynamics (RED_WINS narrow).
14. **Supplementary §C Gauge Frame Gradients (RED_WINS, substantive, this debate).**

This is the first debate in the series where actual MATHEMATICAL ERRORS (FD-verified wrong formulas) are identified, not editorial gaps. The remedy is bounded to specific equation fixes in §C.5 plus the cross-document notation unification.

Optional follow-ups remain: supplementary §D Variational Gradient Descent (lines 611–665) and the Participatory_it_from_bit.tex §Theory.

## Follow-up debates

- **Code-mode audit of mode-4 pullback preconditioner** (suggested by judge): does `transformer/aif/` or `transformer/vfe/` use the verbatim line-606 formula, or is the preconditioner implemented from first principles? This determines whether the math error has propagated into running code.
