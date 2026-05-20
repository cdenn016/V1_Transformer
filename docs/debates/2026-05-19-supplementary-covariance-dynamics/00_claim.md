# Claim — supplementary-covariance-dynamics

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (`Attention/GL(K)_supplementary.tex` §Covariance Dynamics and Equilibrium Analysis lines 180–387; main paper §3.4–§3.7 cross-references at lines 840–946)
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge

## Claim

The `\section{Covariance Dynamics and Equilibrium Analysis}` of `Attention/GL(K)_supplementary.tex` (lines 180–387, comprising §B.1 Covariance Gradient of the Generalized Free Energy with §B.1.1 Gaussian KL Divergence under GL(K) Gauge Transport, §B.2 Fixed-Point Equation and Symmetric Solution with §B.2.1 Homogeneous limit, §B.2.2 Alignment-dominated regime, §B.2.3 Gradient flow dynamics) is complete and mathematically/theoretically pure as a self-contained supplementary chapter. The Gaussian KL closed form and its `∂/∂Σ_1` are correctly stated, the application to the gauge-transported pair `(q_i, Ω_{ij}q_j)` via the sandwich product `Σ_{transported} = Ω_{ij}Σ_jΩ_{ij}^⊤` is correctly evaluated, the boxed Σ-gradient Eq. eq:Sigma_gradient_final assembles these terms with the correct coefficient `-2` on `Σ_i^{-1}`, the fixed-point equation Eq. eq:sigma_fixed_point_beta drops the attention-weight term under the `Σ_j ∂β_ij/∂Σ_i = 0` softmax identity, the three regimes (homogeneous, alignment-dominated, gradient flow) are analyzed honestly with explicit standing assumptions, and the Hessian positive-definiteness at the equilibrium is correctly identified. The chapter contains no residual theoretical-purity issues comparable in magnitude to those identified and corrected in the twelve-debate audit series.

## Sub-claims (compound)

The claim asserts six independent properties of §B:

1. **Sub-claim α (Gaussian KL formula).** Eq. (eq:gaussian_KL) at lines 203–221 matches the canonical Gaussian KL closed form per [Bishop 2006 §2.3; Murphy 2012 §2.3; Cover-Thomas 1991 §8.5].

2. **Sub-claim β (Σ-derivative of Gaussian KL).** The formula `∂KL/∂Σ_1 = (1/2)[-Σ_1^{-1} + Σ_2^{-1}]` at lines 227–232 matches the standard matrix-calculus result for the derivative of the Gaussian KL with respect to the first covariance.

3. **Sub-claim γ (sandwich-product transport application).** The application of `∂KL/∂Σ_1` to `(q_i, Ω_{ij}q_j)` with `Ω_{ij}q_j = N(Ω_{ij}μ_j, Ω_{ij}Σ_jΩ_{ij}^⊤)` yields `(1/2)[-Σ_i^{-1} + (Ω_{ij}Σ_jΩ_{ij}^⊤)^{-1}]` at line 244–251. The sandwich product `Ω_{ij}Σ_jΩ_{ij}^⊤` is the load-bearing transport identity (CLAUDE.md hard constraint).

4. **Sub-claim δ (boxed Σ-gradient assembly).** The boxed Eq. (eq:Sigma_gradient_final) at lines 256–271 correctly assembles the prior-KL contribution, the β-weighted alignment-KL contribution (with the coefficient `-(1 + Σ_j β_{ij}) = -2` on `Σ_i^{-1}`), and the autograd `∂β/∂Σ` correction. The "softmax-gradient correction" treatment matches the canonical-F-vs-surrogate verdict applied to the main paper.

5. **Sub-claim ε (fixed-point equation and regime analysis).** Eq. (eq:sigma_fixed_point_beta) `Σ_i^{-1} = (1/2)[Σ_{p,i}^{-1} + Σ_j β_{ij}(Ω_{ij}Σ_jΩ_{ij}^⊤)^{-1}]` correctly drops the attention-weight correction term under `Σ_j ∂β_{ij}/∂Σ_i = 0`; the homogeneous-limit reduction to `Σ_∞ = Σ_0` is correct; the alignment-dominated regime requires both `α_i ≪ 1` (main paper §3.7 mechanism) AND `τ` small; the regime analyses are honest about scope.

6. **Sub-claim ζ (Hessian and stability).** The Hessian `∂²KL/∂Σ_1∂Σ_1 = (1/2) Σ_1^{-1} ⊠_{sym} Σ_1^{-1}` at lines 377–381 is positive definite, justifying local stability of the equilibrium. The line-387 disclaimer "we do not claim a contractive proof of global uniqueness here" honestly bounds the result.

## Concrete potential issues already identified during evidence assembly

- **Section-number drift at line 337.** Supplementary line 337 plain-text references "main text, Section 3.6" for the state-dependent prior coupling `α_i`. The current main paper places α_i at §3.7 "State-Dependent Prior Precision" (line 895 of `GL(K)_attention.tex`), not §3.6 "Interpretation" (line 877). This is a hard cross-reference error.

- **`Λ_o` referenced at line 385 without definition.** The stability disclaimer at line 385 uses "Λ_o negligible relative to the inter-agent coupling," but `Λ_o` is nowhere defined in §B. Editorial gap.

- **Surrogate-vs-canonical treatment.** §B operates on the entropy-suppressed surrogate `Σ_j β_{ij} D_{KL}(q_i ‖ Ω_{ij}q_j)` per the disclaimer at line 184–185. The chapter notes the σ-gradient is identical under both forms because the attention entropy does not depend on Σ_i. This claim should be verified against the §3-§5 series's canonical-F-vs-surrogate verdict — there the σ-gradient claim is that the cancellation produces -τ log Z reduction, consistent with the supplementary's claim.

## User context

This is the thirteenth debate in the gauge-transformer manuscript audit series, the second in the supplementary chapter sweep following §General Mathematical Framework. The user is conducting a holistic structural audit of all supplementary chapters.

Load-bearing questions for the judge:

1. Do sub-claims α through ζ each hold under primary-source verification?
2. Are the identified editorial issues (section-number drift, undefined `Λ_o`) comparable in magnitude to the prior debate verdicts, or do they fall below the editorial threshold?
3. Are there additional mathematical or theoretical-purity issues beyond those flagged here that red can identify?

A compound verdict (RED_WINS, BLUE_WINS, REMAND, OUT_OF_SCOPE) should reflect the worst load-bearing sub-claim. If all six sub-claims hold under verification and only the editorial issues remain, the verdict should be BLUE_WINS with minor editorial action items.
