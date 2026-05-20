# Evidence Pack — supplementary-gauge-frame-gradients

## Section structure

`Attention/GL(K)_supplementary.tex` §Gauge Frame Gradients spans lines 392–612.

- §C.1 Structure of the Gauge Frame Gradient (lines 397–432).
- §C.2 Differential of the Matrix Exponential (lines 434–450).
  - §C.2.1 SO(N) specialization (lines 452–467, Rodrigues form).
  - §C.2.2 GL(K) specialization (lines 469–486, integral form + block-matrix form).
- §C.3 KL Gradient Through Transport (lines 488–541).
- §C.4 Retraction and Numerical Considerations (lines 543–557).
- §C.5 Gauge Frame Preconditioning for GL(K) (lines 558–612), with four modes: norm clipping, Cartan projector, Killing-form metric, pullback natural gradient.

## Key equations

### §C.1 — autograd gradient assembly

Full autograd gradient Eq. eq:phi_grad_complete (lines 401–411):

```
∂F/∂φ_i = Σ_j [(∂β_{ij}/∂φ_i) K_{ij} + β_{ij} (∂K_{ij}/∂φ_i)]
        + Σ_k [β_{ki} (∂K_{ki}/∂φ_i) + Σ_l (∂β_{kl}/∂φ_i) K_{kl}]
```

Reverse-direction softmax simplification Eq. eq:reverse_beta_grad_phi (lines 416–421):

```
Σ_l (∂β_{kl}/∂φ_i) K_{kl} = -(β_{ki}/τ) (∂K_{ki}/∂φ_i) (K_{ki} - Σ_l β_{kl} K_{kl})
```

Forward-direction softmax Eq. eq:beta_grad_phi (lines 427–432):

```
∂β_{ij}/∂φ_i = -(β_{ij}/τ) [∂K_{ij}/∂φ_i - Σ_k β_{ik} ∂K_{ik}/∂φ_i]
```

### §C.2 — matrix-exponential differential (right-trivialized)

Eq. eq:dexp_series (line 446):

```
dexp_φ(ξ) = Σ_{n=0}^∞ ad_φ^n(ξ)/(n+1)! = (e^{ad_φ}-I)/ad_φ (ξ)
```

Line 450 statement: "Equation eq:dexp_series defines the right-trivialized differential, related to the Fréchet derivative (directional derivative in matrix space) by `D_φ(exp)[ξ] = dexp_φ(ξ) · e^φ`."

### §C.2.1 — SO(3) Rodrigues

Eq. eq:dexp_so3 (lines 456–465):

```
dexp_φ(T_a) = T_a + c_1(θ)[φ, T_a] + c_2(θ)[φ, [φ, T_a]]
c_1(θ) = (1-cos θ)/θ²,  c_2(θ) = (θ-sin θ)/θ³
```

Taylor expansions for θ < ε at line 467: `c_1 ≈ 1/2 - θ²/24`, `c_2 ≈ 1/6 - θ²/120`.

### §C.2.2 — GL(K) integral and block-matrix

Eq. eq:dexp_integral (lines 473–475):

```
dexp_φ(ξ) = ∫_0^1 e^{tφ} ξ e^{(1-t)φ} dt
```

Eq. eq:dexp_block (lines 480–484), citing Higham 2008:

```
exp([[φ, ξ], [0, φ]]) = [[e^φ, dexp_φ(ξ)], [0, e^φ]]
```

### §C.3 — KL gradient through transport

Mean-term, trace-term, log-determinant-term decompositions at lines 506–530. Transport derivatives (Eqs. eq:dtilde_mu, eq:dtilde_Sigma, lines 533–539):

```
∂μ̃_{ij}/∂φ_i^a = Q_a^(i) Ω_{ij} μ_j
∂Σ̃_{ij}/∂φ_i^a = Q_a^(i) Ω_{ij} Σ_j Ω_{ij}^⊤ + Ω_{ij} Σ_j Ω_{ij}^⊤ (Q_a^(i))^⊤
```

with `Q_a^(i) ≡ dexp_{φ_i}(T_a)` defined via `D_{φ_i}(exp)[T_a] = Q_a^(i) exp(φ_i)`.

### §C.4 — retraction

Lie-algebra update Eq. eq:phi_update (lines 547–550):

```
φ_i ← φ_i - η_φ (∂F/∂φ_i)
```

SO(N) retraction to principal ball `‖φ‖ < π - ε` at line 554. GL(K) note at line 556: "the pairwise transport Ω_ij = exp(φ_i) exp(-φ_j) covers all of GL^+(K) via polar decomposition (see main text, Section~\ref{sec:glk_lm})."

### §C.5 — gauge-frame preconditioning

Eq. eq:cartan_decomposition (line 563–565):

```
gl(K) = so(K) ⊕ Sym(K) ⊕ R
```

Cartan projector at line 578–580:

```
P_sym = (1/2) G^{-1} (G + S)
```

with `G_{ab} = tr(T_a^⊤ T_b)` (Frobenius Gram), `S_{ab} = tr(T_a T_b)`. Preconditioner `M = I - (1 - λ_sym) P_sym`.

Cartan-involution-modified metric Eq. eq:killing_metric (line 590):

```
g̃_{ab} = 2K tr(T_a^⊤ T_b) - 2 tr(T_a) tr(T_b)
```

defined at line 587 as `g(X, Y) = -(1/2) B(X, θ(Y))` with `B(X,Y) = 2K tr(XY) - 2 tr(X) tr(Y)` (Killing form), `θ(X) = -X^⊤` (Cartan involution).

Pullback metric Eq. eq:pullback_metric (line 606):

```
G_{ab}(φ) = ⟨Ψ(ad_φ)(T_a), Ψ(ad_φ)(T_b)⟩_G
```

with `⟨X, Y⟩_G = tr(X^⊤ Y)` (Frobenius). The differential at line 599:

```
d exp_φ(T_a) = exp(φ) · Ψ(ad_φ)(T_a),  Ψ(z) = (e^z-1)/z
```

## Canonical-reference verification

### Sub-claim α (autograd gradient)

The softmax derivative `∂β_kl/∂z = β_kl(δ_{kl} - β_kk)` is standard [Bishop 2006 §4.3.4 / §5.4.2]. Direct computation:

For agent k's softmax with only `K_{ki}` depending on φ_i:
```
∂β_{kl}/∂φ_i = β_{kl}(δ_{li} - β_{ki}) · (-∂K_{ki}/∂φ_i / τ)
Σ_l (∂β_{kl}/∂φ_i) K_{kl} = -(∂K_{ki}/∂φ_i / τ) [β_{ki} K_{ki} - β_{ki} Σ_l β_{kl} K_{kl}]
                          = -(β_{ki} ∂K_{ki}/∂φ_i / τ) (K_{ki} - ⟨K_k⟩_β)
```

Matches Eq. eq:reverse_beta_grad_phi ✓.

### Sub-claim β (dexp series)

Canonical form per [Hall 2015 "Lie Groups, Lie Algebras, and Representations" Theorem 5.4 (page 110) or §2.7; Gallier 2020]:

```
D_X(exp)[Y] = e^X · Σ_{n=0}^∞ (-1)^n ad_X^n(Y)/(n+1)!  (left-trivialized)
            = (Σ_{n=0}^∞ ad_X^n(Y)/(n+1)!) · e^X        (right-trivialized)
```

The "right-trivialized differential" `(e^{ad_X}-I)/ad_X (Y) = Σ ad_X^n(Y)/(n+1)!` matches Eq. eq:dexp_series ✓.

### Sub-claim β SO(3) Rodrigues

For skew-symmetric `φ` with `‖φ‖ = θ`, `ad_φ³ = -θ² ad_φ` (verified via direct computation for SO(3) generators). Series truncates:

```
Σ ad_φ^n/(n+1)! = I + ad_φ/2 + ad_φ²/6 + (-θ² ad_φ)/24 + (-θ² ad_φ²)/120 + ...
                = I + ad_φ(1/2 - θ²/24 + θ⁴/720 - ...) + ad_φ²(1/6 - θ²/120 + θ⁴/5040 - ...)
                = I + ad_φ · (1-cos θ)/θ² + ad_φ² · (θ-sin θ)/θ³
```

Matches Eq. eq:dexp_so3 ✓.

### Sub-claim γ (GL(K) integral form)

[Higham 2008 "Functions of Matrices" §10.2 Eq. 10.15]:

```
L_exp(A, E) = D_A exp[E] = ∫_0^1 e^{(1-s)A} E e^{sA} ds
```

Substituting `t = 1-s`: `= ∫_0^1 e^{tA} E e^{(1-t)A} dt`.

This is the FULL Fréchet derivative `D_φ exp[ξ]`, NOT the right-trivialized form `(e^{ad_φ}-I)/ad_φ (ξ)`. The two differ by a factor of `e^φ`:

```
D_φ exp[ξ] = ((e^{ad_φ}-I)/ad_φ)(ξ) · e^φ
           = ∫_0^1 e^{tφ} ξ e^{(1-t)φ} dt
```

The chapter at Eq. eq:dexp_integral labels the integral as `dexp_φ(ξ)`, but line 446 defined `dexp_φ(ξ) = (e^{ad_φ}-I)/ad_φ (ξ)`. **These are not the same object.** The integral is `D_φ exp[ξ] = dexp_φ(ξ) · e^φ`, not `dexp_φ(ξ)` alone.

This is a substantive notational inconsistency.

### Sub-claim γ (block-matrix identity)

[Higham 2008 "Functions of Matrices" Algorithm 10.27]:

```
exp([[A, E], [0, A]]) = [[e^A, L_exp(A, E)], [0, e^A]]
```

where `L_exp(A, E) = D_A exp[E]` (full Fréchet derivative). The block (1,2) entry is the FULL Fréchet derivative, not the right-trivialized form. The chapter at Eq. eq:dexp_block uses `dexp_φ(ξ)` for this entry, again inconsistent with line 446.

### Sub-claim δ (KL gradient through transport)

The chain rule for `K_{ij} = D_{KL}(q_i ‖ Ω_{ij} q_j)` through `Ω_{ij}` is standard. The transport derivatives Eqs. eq:dtilde_mu, eq:dtilde_Sigma use the chain rule:

```
∂(Ω_{ij} μ_j)/∂φ_i^a = (∂Ω_{ij}/∂φ_i^a) μ_j
```

With `Ω_{ij} = exp(φ_i) exp(-φ_j)` and using the right-trivialized convention `Q_a^(i) ≡ dexp_{φ_i}(T_a)` and `D_{φ_i} exp[T_a] = Q_a^(i) exp(φ_i)`:

```
∂Ω_{ij}/∂φ_i^a = D_{φ_i} exp[T_a] · exp(-φ_j) = Q_a^(i) exp(φ_i) exp(-φ_j) = Q_a^(i) Ω_{ij}
```

So `∂(Ω_{ij} μ_j)/∂φ_i^a = Q_a^(i) Ω_{ij} μ_j` ✓ matches Eq. eq:dtilde_mu.

For the covariance, `∂(Ω_{ij} Σ_j Ω_{ij}^⊤)/∂φ_i^a = (Q_a^(i) Ω_{ij}) Σ_j Ω_{ij}^⊤ + Ω_{ij} Σ_j (Ω_{ij}^⊤ (Q_a^(i))^⊤) = Q_a^(i) Ω_{ij} Σ_j Ω_{ij}^⊤ + Ω_{ij} Σ_j Ω_{ij}^⊤ (Q_a^(i))^⊤` ✓ matches Eq. eq:dtilde_Sigma. (Note: this uses `(AB)^⊤ = B^⊤ A^⊤` correctly.)

The transport derivatives are internally consistent with the RIGHT-trivialized convention.

### Sub-claim ζ (Killing form on gl(K))

[Knapp 2002 "Lie Groups Beyond an Introduction" Proposition 1.93]: For `sl(K, R)`, the Killing form is `B(X, Y) = 2K tr(XY)`. Extending to `gl(K)` with the center contribution:

```
B(X, Y) = 2K tr(XY) - 2 tr(X) tr(Y)
```

Direct computation of `g(X, Y) = -(1/2) B(X, θ(Y))` with `θ(Y) = -Y^⊤`:

```
g(X, Y) = -(1/2) B(X, -Y^⊤) = (1/2) B(X, Y^⊤)
        = (1/2) [2K tr(X Y^⊤) - 2 tr(X) tr(Y^⊤)]
        = K tr(X Y^⊤) - tr(X) tr(Y)
```

In the generator basis: `g(T_a, T_b) = K tr(T_a T_b^⊤) - tr(T_a) tr(T_b)`.

But Eq. eq:killing_metric (line 590) writes `g̃_{ab} = 2K tr(T_a^⊤ T_b) - 2 tr(T_a) tr(T_b)`, off by a factor of 2 from the derivation.

**Note**: The chapter may be defining `g̃ = 2g` or omitting the `-(1/2)` prefactor from the line 587 definition when going to the generator basis. Either way, this is a normalization discrepancy that should be made explicit.

### Sub-claim ζ (pullback metric assumes compact subgroup)

For φ ∈ so(K) (skew), `exp(φ) ∈ SO(K)` so `exp(φ)^⊤ exp(φ) = I`. The Frobenius-pullback metric reduces to `⟨Ψ(ad_φ)(T_a), Ψ(ad_φ)(T_b)⟩_F`.

For general φ ∈ gl(K), `exp(φ)^⊤ exp(φ) ≠ I` and the metric should include this factor. The chapter's Eq. eq:pullback_metric omits it.

## Cross-reference verification

### Reference at line 556 to `sec:glk_lm`

The label `sec:glk_lm` resolves to `Attention/GL(K)_attention.tex:2080` "§4.2 GL(K) Language Modeling: The Full General Model". This section is an experimental design section about training the GL(K) gauge transformer on WikiText-103, NOT a polar-decomposition derivation. The supplementary at line 556 references this label claiming "polar decomposition" content, which does not exist at the cited location.

### Reference at line 399 to "main text Section 3.5"

The supplementary at line 399 references "main text Section~3.5 for the distinction between autograd and reduced-free-energy gradients." Main paper §3.5 is "Full Variational Free Energy" (line 840, label `sec:final_free_energy`), which contains the envelope-theorem treatment at lines 859–874. ✓ Reference is correct (plain text but section number matches current state of main paper).

## Bib verification

- `Hall2015` exists at `references.bib:1046`.
- `Higham2008` exists at `references.bib:1138`.
- `Gallier2020` exists at `references.bib:1146`.
- `culver1966existence` exists at `references.bib:2719`.

## Canon excerpts — external standards

### dexp series and trivializations

- [Hall 2015 §2.7 / Theorem 5.4] — `D_X exp[Y] = exp(X) · (1-e^{-ad_X})/ad_X (Y)` (left-trivialized) = `((e^{ad_X}-I)/ad_X)(Y) · exp(X)` (right-trivialized).
- [Gallier 2020 "Differential Geometry and Lie Groups" Vol. 2] — same formulas.
- [Iserles-Nørsett-Munthe-Kaas "Lie-group methods" Acta Numerica 2000 §2.3] — explicit discussion of right- vs left-trivialization conventions.

### Fréchet derivative integral form

- [Higham 2008 "Functions of Matrices" Theorem 10.13 / §10.2 Eq. 10.15]: `L_exp(A, E) = ∫_0^1 e^{(1-s)A} E e^{sA} ds`.
- [Higham 2008 Algorithm 10.27]: block-matrix construction for computing `L_exp`.

### Killing form on gl(K)

- [Knapp 2002 "Lie Groups Beyond an Introduction" Proposition 1.93]: Killing form on `sl(K, R)` is `B(X, Y) = 2K tr(XY)`.
- [Helgason 1978 "Differential Geometry, Lie Groups, and Symmetric Spaces" Ch. III]: Cartan decomposition and involutions.

### Polar decomposition of GL+(K)

- [Higham 2008 §8]: polar decomposition `A = UP` with `U ∈ O(K)` orthogonal and `P ∈ Sym_{++}` SPD. For `A ∈ GL^+(K)`, `U ∈ SO(K)`. The supplementary's claim that `Ω_ij = exp(φ_i) exp(-φ_j)` covers `GL^+(K)` via polar decomposition can be defended by noting `exp` is surjective onto `GL^+(K)` for the product of an SO(K) factor and an SPD factor (`exp` maps `Sym → SPD` bijectively, and `exp` maps `so(K) → SO(K)` surjectively); this would need a derivation, and the cited section `sec:glk_lm` does not provide it.

## What this evidence does NOT settle

1. **Whether the dexp_φ notation inconsistency at Issue 1 rises above the editorial threshold.** It is a substantive textual error that could mislead implementations, but no derivation in §C actually depends on the inconsistency (the transport derivatives at §C.3 are consistent with the right-trivialized convention; the integral/block forms at §C.2.2 are correct for `D_φ exp` even if the symbol is misapplied).

2. **Whether the Killing-form normalization at Issue 2 is editorial or derivation-incorrect.** The factor of 2 affects the actual numerical preconditioner if implemented from the manuscript formula directly; if the codebase derives the metric from `g(X, Y) = -(1/2) B(X, θ(Y))` directly, the implementation may differ from the line-590 formula.

3. **Whether the pullback metric at Issue 4 is intended only for compact subgroups.** The chapter at §C.5 specifically targets GL(K) (non-compact). If the pullback formula at line 606 only holds for compact subgroups, this is a derivation gap.

4. **Whether the polar-decomposition surjectivity argument at line 556 is derived anywhere in the manuscripts** or simply asserted via the (incorrect) cross-reference to `sec:glk_lm`.

5. **Whether the chapter cites sufficient canonical references.** §C does cite [Hall 2015, Gallier 2020, Higham 2008, culver 1966existence] — better than §B's zero internal citations — but [Knapp 2002] or [Helgason 1978] for the Killing form / Cartan involution machinery is absent.
