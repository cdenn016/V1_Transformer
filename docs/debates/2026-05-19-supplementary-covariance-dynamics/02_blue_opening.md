# Blue Opening — supplementary-covariance-dynamics

## Steelman (opposing position)

Section B of `GL(K)_supplementary.tex` is not theoretically pure because (a) the line-337 cross-reference to "main text, Section 3.6" is wrong (the cited material lives at §3.7), (b) the line-385 symbol `Λ_o` appears with no antecedent definition anywhere in the chapter, and (c) §B treats textbook results (Gaussian KL, its `∂/∂Σ_1`, the sandwich-product push-forward, the SPD Hessian) as self-evident with zero in-text citations, which the prior debate series has already flagged as below the editorial threshold for "self-contained" supplementary material.

## Position

All six load-bearing sub-claims (α Gaussian KL closed form, β `∂KL/∂Σ_1`, γ sandwich-product transport, δ assembly with coefficient `-2`, ε fixed-point and regime analysis, ζ Hessian positive-definiteness) are mathematically/theoretically pure under primary-source verification against `external_canon_math.md` and `external_canon_inference.md`. The three editorial issues red can raise (`3.6` → `3.7` cross-reference drift, undefined `Λ_o`, missing inline citations) are real but scoped to single-word, single-symbol, and citation-list fixes; none of them touches the derivation chain. The verdict should be BLUE_WINS with three editorial action items.

## Evidence

### Sub-claim α — Gaussian KL closed form (lines 203–221)

The supplementary states

```
KL(N(μ_1, Σ_1) ‖ N(μ_2, Σ_2))
  = (1/2)[log(|Σ_2|/|Σ_1|) + tr(Σ_2^{-1} Σ_1) + (μ_2-μ_1)^⊤ Σ_2^{-1} (μ_2-μ_1) - d]
```

This matches `external_canon_inference.md §1` "Closed-form KL between Gaussians":

```
KL(q ‖ p) = ½ [ tr(Σ_p⁻¹ Σ_q) + (μ_p − μ_q)ᵀ Σ_p⁻¹ (μ_p − μ_q) − K + log(|Σ_p|/|Σ_q|) ]
```

with the identification `(q, p) ↔ (1, 2)`, `K ↔ d`. The quadratic term is symmetric under `μ_1 ↔ μ_2` so the sign convention is identical to canon. Primary sources: [Bishop2006 §2.3.6 Eq. 2.121], [Murphy2012 §2.3.2], [CoverThomas2006 §8.6 Eq. 8.69]. No derivation gap.

### Sub-claim β — `∂KL/∂Σ_1` (lines 227–232)

The supplementary states `∂KL/∂Σ_1 = (1/2)[-Σ_1^{-1} + Σ_2^{-1}]`. Direct verification using matrix-calculus canon:

- [PetersenPedersen MatrixCookbook §9.1 Eq. 75]: `∂ log|X|/∂X = X^{-T}`. For symmetric `X`, `X^{-T} = X^{-1}`. So `-(1/2) ∂ log|Σ_1|/∂Σ_1 = -(1/2) Σ_1^{-1}`.
- [PetersenPedersen MatrixCookbook §9.4 Eq. 100]: `∂ tr(AX)/∂X = A^⊤`. So `(1/2) ∂ tr(Σ_2^{-1} Σ_1)/∂Σ_1 = (1/2) Σ_2^{-T} = (1/2) Σ_2^{-1}`.
- `log|Σ_2|`, `-d`, and the quadratic-μ term all have zero `∂/∂Σ_1`.

Sum: `(1/2)[-Σ_1^{-1} + Σ_2^{-1}]`. Matches the supplementary exactly. Also reproduced verbatim in [Magnus-Neudecker 2019 Matrix Differential Calculus Ch. on Gaussian likelihood derivatives].

### Sub-claim γ — sandwich-product transport (line 234)

The supplementary writes `Ω_{ij} q_j = N(Ω_{ij}μ_j, Ω_{ij}Σ_jΩ_{ij}^⊤)`. This is the standard Gaussian linear-transformation rule: if `x ~ N(μ, Σ)` and `y = Ax + b`, then `y ~ N(Aμ + b, AΣA^⊤)` [Bishop2006 §2.3.3]. It is also the load-bearing transport identity from `external_canon_math.md §2` ("Transport of tensors — the sandwich identity (THIS IS THE STANDARD)"), case (2,0)-tensor: `T → ρ(g) T ρ(g)^⊤`. The supplementary applies it canonically.

### Sub-claim δ — assembly with coefficient `-2` (lines 256–271)

Direct algebra from sub-claims β and γ:

```
∂F_i/∂Σ_i = ∂_{Σ_i} KL(q_i ‖ p_i) + Σ_j β_{ij} ∂_{Σ_i} KL(q_i ‖ Ω_{ij} q_j) + [∂β/∂Σ correction]
          = (1/2)[-Σ_i^{-1} + Σ_{p,i}^{-1}]
            + (1/2) Σ_j β_{ij} [-Σ_i^{-1} + (Ω_{ij}Σ_jΩ_{ij}^⊤)^{-1}]
            + Σ_j (∂β_{ij}/∂Σ_i) KL_{ij}
          = (1/2)[-(1 + Σ_j β_{ij}) Σ_i^{-1} + Σ_{p,i}^{-1} + Σ_j β_{ij} (Ω_{ij}Σ_jΩ_{ij}^⊤)^{-1}] + [correction]
          = (1/2)[-2 Σ_i^{-1} + Σ_{p,i}^{-1} + Σ_j β_{ij} (Ω_{ij}Σ_jΩ_{ij}^⊤)^{-1}] + [correction]
```

using `Σ_j β_{ij} = 1` from line 197. This matches the boxed Eq. (eq:Sigma_gradient_final) at lines 256–271 exactly. Line 275 of the supplementary states the same identity in prose: "The coefficient of `Σ_i^{-1}` is `-(1 + Σ_j β_{ij}) = -2`". No gap. The "softmax-gradient correction" term `Σ_j (∂β_{ij}/∂Σ_i) D_KL` is the same envelope-theorem versus autograd distinction made in the main paper for the µ-gradient, consistent with `external_canon_inference.md §1 "Form-1 vs Form-2 vs Form-3 conflation"` pitfall checks.

### Sub-claim ε — fixed-point and regime analysis (lines 277–362)

The softmax constraint `Σ_j β_{ij} = 1` implies `Σ_j ∂β_{ij}/∂Σ_i = ∂(1)/∂Σ_i = 0` automatically. This is the simplex-constraint identity. Under the additional condition that the KL divergences `D_KL(q_i ‖ Ω_{ij} q_j)` are approximately uniform across `j` (or in the `τ→0`/`τ→∞` limits, both of which the supplementary calls out at line 273 and line 279), the full sum `Σ_j (∂β_{ij}/∂Σ_i) D_{KL,ij}` vanishes. Then Eq. (eq:sigma_fixed_point_beta) follows by setting `∂F_i/∂Σ_i = 0`.

The homogeneous-limit reduction `Σ_∞^{-1} = (1/2)[Σ_0^{-1} + Σ_∞^{-1}] ⇒ Σ_∞ = Σ_0` (line 306–315) is correct by direct substitution.

The alignment-dominated regime (§B.2.2) at line 337 explicitly states the standing assumption: "one requires the state-dependent prior coupling `α_i ≪ 1` (cf. main text, Section~3.6), which changes the coefficient from `-(1+α_i)` to approximately `-1`. In the alignment-dominated regime where both `α_i ≪ 1` and `τ` is small, the fixed point becomes ...". Both standing assumptions are made explicit before the regime equation is written. The structural argument matches `external_canon_inference.md §3` "Hierarchical / nested formulations" coupling-strength scaling and is internally consistent.

### Sub-claim ζ — Hessian positive-definiteness (lines 375–387)

The Hessian of `-(1/2) log|Σ_1|` on `Sym(K)` is the standard positive-definite quartic form:

```
Hess[H, H] = (1/2) tr(Σ_1^{-1} H Σ_1^{-1} H) = (1/2) ‖Σ_1^{-1/2} H Σ_1^{-1/2}‖_F²
```

with equality iff `H = 0`. The linear term `(1/2) tr(Σ_2^{-1} Σ_1)` contributes zero second derivative. So the total Hessian is exactly `(1/2) Σ_1^{-1} ⊠_{sym} Σ_1^{-1}` as the supplementary states. Equivalent forms appear in [Smith 2005 "Covariance, subspace, and intrinsic Cramér-Rao bounds"] and [Pennec 2006 "Intrinsic statistics on Riemannian manifolds"], i.e. the SPD-manifold canon listed at `01_evidence.md §Canon excerpts`. The line-387 disclaimer "we do not claim a contractive proof of global uniqueness here" honestly bounds the result to local stability, which is exactly what the positive-definiteness establishes. Honest scoping is not a derivation gap.

### Editorial issues are scoped fixes, not load-bearing failures

The three editorial items in `01_evidence.md §Concrete editorial issues`:

1. **Line-337 `3.6` → `3.7`** is a one-word cross-reference drift. The mathematical content at line 337 is correct: it correctly identifies `α_i ≪ 1` as the state-dependent prior coupling condition, and `α_i` is correctly identified as the coefficient `-(1+α_i)` on `Σ_i^{-1}`. The drift is the section-number label only. Fix: replace `3.6` with `3.7` or, better, use `\ref{sec:state_dependent_precision}`. Single-token edit. No downstream propagation into any derivation.

2. **`Λ_o` undefined at line 385**. Editorial gap. The intended interpretation is observation precision `R^{-1}` from the noise covariance `R` introduced at line 253. Fix: either define `Λ_o := R^{-1}` at line 385, or replace `Λ_o` with `R^{-1}` to match line-253 notation. Single-symbol edit. The substantive content of the disclaimer ("observation-likelihood contribution is negligible compared to inter-agent coupling") is mathematically defensible from line-253's prose "in the high-precision regime where `Σ_i^{-1} ≫ R^{-1}`, this `O(1)` correction is negligible compared to the `O(Σ_i^{-1})` alignment and prior terms".

3. **Missing inline citations**. §B presents Gaussian KL and its derivatives as standard textbook results without inline citations to [Bishop2006], [Murphy2012], [PetersenPedersen]. This is consistent with treating these as common knowledge in the matrix-calculus and Gaussian-information-theory communities. The supplementary's bibliography at §A already includes the canonical sources. Adding three to four inline `\cite{}` calls (`\cite{Bishop2006}` at line 201, `\cite{PetersenPedersen}` at line 224, `\cite{Pennec2006}` at line 383) would close the gap without altering any derivation. Editorial action item.

None of (1)–(3) propagates into a derivation error in sub-claims α–ζ.

## Falsification conditions

This position is wrong if any of the following can be demonstrated against the primary canon:

1. **Derivation breakage in any of α–ζ.** If red shows the supplementary's KL formula, `∂KL/∂Σ_1`, sandwich-product application, coefficient-`-2` assembly, fixed-point reduction, or Hessian formula disagrees with [Bishop2006], [Murphy2012], [CoverThomas2006], [PetersenPedersen], [Smith 2005], or [Pennec2006] under a verifiable derivation, blue concedes the affected sub-claim.

2. **Section-number drift propagates downstream.** If red shows that the `3.6` vs `3.7` confusion at line 337 actually corresponds to a substantively different coupling mechanism (i.e., §3.6 "Interpretation" of the main paper introduces an `α_i` definition that disagrees with §3.7's "State-Dependent Prior Precision"), and that the supplementary's regime equation `(eq:beta_weighted_precision)` therefore relies on the wrong mechanism, blue concedes Sub-claim ε beyond editorial.

3. **Gauge-equivariance failure of the σ-gradient.** If red shows that the right-hand side of Eq. (eq:Sigma_gradient_final) fails to transform covariantly under simultaneous push-forward `Σ_i → Ω Σ_i Ω^⊤` and `Σ_j → Ω Σ_j Ω^⊤` (with the standard `Ω_{ij}` transformation rule), blue concedes a load-bearing CLAUDE.md hard-constraint gap. (The `01_evidence.md §What this evidence does NOT settle` item 3 flags this as not explicitly verified in the chapter.)

4. **`Λ_o` is not `R^{-1}`.** If red shows that `Λ_o` at line 385 cannot be plausibly identified with the observation-noise precision `R^{-1}` (e.g., the supplementary uses `Λ_o` elsewhere in the manuscript with a different meaning), the editorial gap escalates to a substantive ambiguity in the stability disclaimer.

5. **Surrogate-vs-canonical σ-gradient.** If red shows that adding the attention-entropy term `τβ log(β/π)` to the surrogate (i.e., moving to the canonical F of the main text) changes the σ-gradient in any way — contradicting the line-184–185 disclaimer that "the covariance gradient is identical under both forms because the attention entropy does not depend on `Σ_i`" — blue concedes Sub-claim δ.

6. **Missing-citation threshold exceeded.** If red shows that the missing-citation pattern in §B is materially worse than (or comparable to) the citation-related grounds on which the prior debate series ruled adversely, the editorial issue may escalate to a structural failure rather than an action item. Blue's prior position is that the §B citation pattern (zero inline citations to canonical texts) is materially less severe than the prior `D Gamma prior` debate (which involved labeling a canonical Bayesian construction as a "natural choice" without identification) — because §B presents its calculations as standard and does not claim novelty for them, whereas the Gamma-prior case under-labeled the canon.
