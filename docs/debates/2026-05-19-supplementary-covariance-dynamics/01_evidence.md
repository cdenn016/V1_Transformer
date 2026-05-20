# Evidence Pack — supplementary-covariance-dynamics

## Section structure

`Attention/GL(K)_supplementary.tex` §Covariance Dynamics and Equilibrium Analysis spans lines 180–387.

- §B.1 Covariance Gradient of the Generalized Free Energy (lines 183–275).
  - §B.1.1 Gaussian KL Divergence under GL(K) Gauge Transport (lines 199–273).
  - Boxed Eq. eq:Sigma_gradient_final at lines 256–271.
- §B.2 Fixed-Point Equation and Symmetric Solution (lines 277–292).
  - §B.2.1 Homogeneous limit (lines 294–317).
  - §B.2.2 Alignment-dominated regime (lines 319–362).
  - §B.2.3 Gradient flow dynamics (lines 364–387).

## Key equations

### Surrogate free energy used as starting point (line 187–195)

```
F_i = KL(q_i ‖ p_i) + Σ_{j≠i} β_{ij} KL(q_i ‖ Ω_{ij} q_j) - E_{q_i}[log p(o_i | k_i)]
```

with `q_i = N(μ_i, Σ_i)`, `p_i = N(μ_{p,i}, Σ_{p,i})`, `Σ_j β_{ij} = 1`.

**Note (line 184–185 disclaimer):** "For brevity we work with the entropy-suppressed surrogate ... the canonical free energy of the main text adds the `τβ_{ij} log(β_{ij}/π_{ij})` entropy term to make the softmax form of β a stationary point. The covariance gradient is identical under both forms because the attention entropy does not depend on Σ_i."

### Eq. eq:gaussian_KL (lines 203–221)

```
KL(N(μ_1, Σ_1) ‖ N(μ_2, Σ_2)) = (1/2)[log(|Σ_2|/|Σ_1|) + tr(Σ_2^{-1} Σ_1) + (μ_2-μ_1)^⊤ Σ_2^{-1} (μ_2-μ_1) - d]
```

### Eq. ∂KL/∂Σ_1 (lines 227–232)

```
∂KL/∂Σ_1 = (1/2)[-Σ_1^{-1} + Σ_2^{-1}]
```

### Eq. eq:Sigma_gradient_final boxed (lines 256–271)

```
∂F_i/∂Σ_i = (1/2)[-2 Σ_i^{-1} + Σ_{p,i}^{-1} + Σ_j β_{ij} (Ω_{ij} Σ_j Ω_{ij}^⊤)^{-1}]
          + Σ_j (∂β_{ij}/∂Σ_i) KL(q_i ‖ Ω_{ij} q_j)
```

### Eq. eq:sigma_fixed_point_beta (lines 282–290)

```
Σ_i^{-1} = (1/2)[Σ_{p,i}^{-1} + Σ_j β_{ij} (Ω_{ij} Σ_j Ω_{ij}^⊤)^{-1}]
```

(setting `∂F_i/∂Σ_i = 0` under `Σ_j ∂β_{ij}/∂Σ_i = 0`).

### Homogeneous limit (line 306–315)

```
Σ_∞^{-1} = (1/2)[Σ_0^{-1} + Σ_∞^{-1}] ⇒ Σ_∞ = Σ_0
```

### Eq. eq:beta_weighted_precision (lines 339–348)

```
Σ_i^{-1} ≈ Σ_j β_{ij} (Ω_{ij} Σ_j Ω_{ij}^⊤)^{-1} = ⟨(Ω_{ij} Σ_j Ω_{ij}^⊤)^{-1}⟩_β
```

(alignment-dominated regime, both `α_i ≪ 1` AND `τ` small).

### Hessian (lines 377–381)

```
∂²KL/∂Σ_1∂Σ_1 = (1/2) Σ_1^{-1} ⊠_{sym} Σ_1^{-1}
```

positive definite for `Σ_1 ≻ 0`.

## Canonical-reference verification (primary sources)

### Sub-claim α: Gaussian KL closed form

The closed-form Gaussian KL is a standard textbook result.

- **[Bishop 2006 §2.3.6 "The Gaussian Distribution," around Eq. 2.121]**: same form modulo notation.
- **[Murphy 2012 §2.3.2 "MVN: KL divergence"]**: same form.
- **[Cover-Thomas 1991 §8.5 "KL Divergence between Gaussians"]**: same form (in the second edition Cover-Thomas 2006, this is §8.6).
- **[Cover-Thomas 2006 §8.6 Eq. 8.69]**: confirms `D(N(μ_1, K_1) ‖ N(μ_2, K_2)) = (1/2)[log(|K_2|/|K_1|) + tr(K_2^{-1} K_1) - d + (μ_1-μ_2)^⊤ K_2^{-1} (μ_1-μ_2)]`.

The supplementary's formula at lines 203–221 matches canon **exactly** (modulo the sign convention on the quadratic term `(μ_2 - μ_1)^⊤ Σ_2^{-1} (μ_2 - μ_1)`, which is symmetric under `μ_1 ↔ μ_2` so the two conventions coincide).

### Sub-claim β: ∂KL/∂Σ_1

The matrix-calculus derivative.

- **[Petersen-Pedersen Matrix Cookbook (v. 2012) §9.1 Eq. 75]**: `∂ log|X|/∂X = X^{-T}`, with `X^{-T} = X^{-1}` for symmetric `X`.
- **[Petersen-Pedersen Matrix Cookbook §9.4 Eq. 100]**: `∂ tr(AX)/∂X = A^⊤`.

Applying to the Gaussian KL Hess-as-function-of-Σ_1 (with Σ_2 fixed):

- `-(1/2) ∂ log|Σ_1|/∂Σ_1 = -(1/2) Σ_1^{-1}` (using symmetric Σ_1).
- `+(1/2) ∂ tr(Σ_2^{-1} Σ_1)/∂Σ_1 = (1/2) Σ_2^{-T} = (1/2) Σ_2^{-1}`.
- All other terms (`log|Σ_2|`, quadratic-μ, `-d`) have zero ∂/∂Σ_1.

Sum: `∂KL/∂Σ_1 = (1/2)[-Σ_1^{-1} + Σ_2^{-1}]`. ✓ Matches supplementary line 227–232 **exactly**.

### Sub-claim γ: sandwich-product transport

`Ω q_j = N(Ω μ_j, Ω Σ_j Ω^⊤)` is the standard push-forward of a Gaussian under a linear map `Ω ∈ GL(K)`.

- **[Bishop 2006 §2.3.3 "Marginal Gaussian distributions" + linear transformation property]**: if `x ~ N(μ, Σ)` and `y = Ax + b`, then `y ~ N(Aμ + b, A Σ A^⊤)`.
- This is the sandwich product per CLAUDE.md hard constraint: "Covariance transport must always use the sandwich product: `Σ_transported = Ω @ Σ @ Ω.T`."

The supplementary at line 234 applies this canonically.

### Sub-claim δ: coefficient -2 verification

Direct calculation:
```
∂F_i/∂Σ_i = ∂/∂Σ_i KL(q_i ‖ p_i) + Σ_j β_{ij} ∂/∂Σ_i KL(q_i ‖ Ω_{ij} q_j) [autograd β-fixed part]
          = (1/2)[-Σ_i^{-1} + Σ_{p,i}^{-1}] + (1/2) Σ_j β_{ij} [-Σ_i^{-1} + (Ω Σ_j Ω^⊤)^{-1}]
          = (1/2)[-(1 + Σ_j β_{ij}) Σ_i^{-1} + Σ_{p,i}^{-1} + Σ_j β_{ij} (Ω Σ_j Ω^⊤)^{-1}]
          = (1/2)[-2 Σ_i^{-1} + Σ_{p,i}^{-1} + Σ_j β_{ij} (Ω Σ_j Ω^⊤)^{-1}]
```
using `Σ_j β_{ij} = 1`. The coefficient `-2` is exact. ✓

### Sub-claim ε: softmax identity `Σ_j ∂β/∂Σ = 0`

From `β_{ij} = exp(-E_{ij}/τ)/Σ_k exp(-E_{ik}/τ)` with `Σ_j β_{ij} = 1`:
```
∂(Σ_j β_{ij})/∂Σ_i = Σ_j ∂β_{ij}/∂Σ_i = ∂(1)/∂Σ_i = 0
```
The identity is automatic from the simplex constraint. ✓

The fixed-point equation Eq. eq:sigma_fixed_point_beta correctly drops the attention-weight term when "the KL divergences D_KL(q_i ‖ Ω_{ij} q_j) are approximately uniform across neighbors" or in the τ→0 or τ→∞ limits. This is consistent with `Σ_j ∂β/∂Σ_i = 0` constraining the *sum* of corrections; the full attention-weight term `Σ_j (∂β_{ij}/∂Σ_i) KL_{ij}` need not vanish term-by-term, only when KLs are uniform or β is sharp/flat.

### Sub-claim ζ: Hessian positive-definiteness

The Hessian of `-(1/2) log|Σ_1|` is a positive-definite quartic on `Sym(K)`:
```
Hess[H, H] = (1/2) tr(Σ_1^{-1} H Σ_1^{-1} H) = (1/2) ‖Σ_1^{-1/2} H Σ_1^{-1/2}‖_F²
```
which is `≥ 0` with equality iff `H = 0`. The linear part `(1/2) tr(Σ_2^{-1} Σ_1)` contributes zero Hessian. ✓

The `Σ_1^{-1} ⊠_{sym} Σ_1^{-1}` notation is shorthand for the symmetrized Kronecker product acting on `Sym(K)`; equivalent forms appear in [Smith 2005 "Covariance, subspace, and intrinsic Cramér-Rao bounds"] and the SPD-manifold literature.

## Concrete editorial / correctness issues identified during evidence assembly

### Issue 1: Section-number drift at supplementary line 337

The supplementary at line 337 says: "*To obtain a coefficient of 1 on the alignment term, one requires the state-dependent prior coupling `α_i ≪ 1` (cf. main text, Section~3.6)*"

Main paper section structure (verified via Grep on `\subsection{}` headers in `GL(K)_attention.tex`):
- §3.5 Full Variational Free Energy (line 840, label `sec:final_free_energy`)
- §3.6 Interpretation (line 877)
- §3.7 State-Dependent Prior Precision (line 895, where α_i is derived)

The state-dependent prior coupling `α_i` lives at §3.7, NOT §3.6. The reference is incorrect.

Supplementary line 273 says "see main text, Section~3.5" for the envelope theorem — this correctly points to §3.5 Full Variational Free Energy (envelope-theorem treatment is at lines 859–874 within that section). ✓

### Issue 2: `Λ_o` referenced at line 385 without definition

The stability disclaimer at line 385 reads: "*Hence the covariance alignment fixed-point is an attractor of the variational dynamics under the standing assumptions `Σ_j β_{ij} = 1`, `Ω_{ij}` invertible, and `Λ_o` negligible relative to the inter-agent coupling.*"

`Λ_o` is nowhere defined in §B or earlier in the supplementary. Likely interpretation: observation precision (the inverse of the observation noise covariance `R` that appears at line 253). The cross-reference at line 253 mentions `R^{-1}` as the observation-precision contribution; line 385's `Λ_o` is the same quantity but with a different symbol. Editorial gap.

### Issue 3: Plain-text section reference style

Lines 273 ("see main text, Section~3.5") and 337 ("cf. main text, Section~3.6") use plain-text section numbers rather than LaTeX `\ref{}` to a label. This is the source of drift Issue 1. Best practice would be to use `\ref{sec:final_free_energy}` and `\ref{sec:state_dependent_precision}`.

### Issue 4: Section heading "Symmetric Solution" potentially overpromises

§B.2 is titled "Fixed-Point Equation and Symmetric Solution." The fixed-point equation is established but a unique "symmetric solution" is only obtained in the homogeneous limit (§B.2.1). The line 387 disclaimer ("we do not claim a contractive proof of global uniqueness here") is honest but the section title might mislead a reader expecting a uniqueness theorem.

## Canon excerpts — external standards

### KL of Gaussians (textbook)

- [Bishop 2006 §2.3.6 "The Gaussian Distribution"]: derivation of KL(N(μ_1,Σ_1) ‖ N(μ_2,Σ_2)).
- [Murphy 2012 §2.3.2 "MVN: KL divergence"]: same formula.
- [Cover-Thomas 2006 §8.6 Eq. 8.69]: information-theoretic version.

### Matrix calculus

- [Petersen-Pedersen Matrix Cookbook §9.1, §9.4]: derivatives of log determinants and traces.
- [Magnus-Neudecker 2019 "Matrix Differential Calculus"]: comprehensive treatment, including symmetric-matrix derivatives.

### Gaussian belief propagation / collaborative inference

- [Wainwright-Jordan 2008 §3 "Mean-field methods"]: belief-propagation fixed points; the supplementary's coefficient `-(1+α_i)` is consistent with the variational coupling structure.
- [Bishop 2006 §10.1]: variational message passing on Gaussian graphs.

### SPD manifold geometry

- [Pennec 2006 "Intrinsic statistics on Riemannian manifolds"]: log-Euclidean and affine-invariant metrics on SPD.
- [Smith 2005 "Covariance, subspace, and intrinsic Cramér-Rao bounds"]: SPD Hessian forms; the `⊠_{sym}` notation appears in similar contexts.

## What this evidence does NOT settle

1. **Whether the section-number drift at line 337 is sufficient on its own to fail the compound claim.** It is a concrete cross-reference error introducing reader confusion; whether it rises above the editorial threshold depends on judge calibration relative to the prior debates' editorial standard.

2. **Whether `Λ_o` undefined at line 385 is sufficient on its own to fail the compound claim.** Editorial gap; clear from context but unrigorous.

3. **Whether the gauge-equivariance of the σ-gradient `∂F_i/∂Σ_i` is verified or merely asserted.** The chapter doesn't explicitly show that the right-hand-side of Eq. eq:Sigma_gradient_final transforms covariantly under simultaneous push-forward `Σ_i → Ω Σ_i Ω^⊤`. The framework's hard constraint (CLAUDE.md "Preserve gauge equivariance") would benefit from this verification.

4. **Whether the symmetry-product Hessian notation `⊠_{sym}` is defined in the manuscript.** It is not defined in §B. Editorial — readers familiar with SPD-manifold notation will infer the meaning; others may not.

5. **Whether the "alignment-dominated regime" (both `α_i ≪ 1` and `τ` small) ever obtains in the framework's empirical experiments.** Per CLAUDE.md the framework uses `α_i = 1` (constant) as default, with `α_i*` derived only when `E_learnable_alpha=True`. The §B.2.2 regime analysis describes a theoretical limit that may not be exercised in practice.

6. **Whether the chapter cites the relevant external canonical references.** §B does NOT cite Bishop 2006, Murphy 2012, Cover-Thomas 1991, or Petersen-Pedersen Matrix Cookbook for the Gaussian KL or its derivatives — even though these are textbook results. The supplementary's general bibliography at §A includes some of these, but §B has no citations within its own text. Editorial gap.
