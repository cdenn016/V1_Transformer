# Red Opening — supplementary-covariance-dynamics

## Steelman (opposing position)

§B of `Attention/GL(K)_supplementary.tex` correctly derives the Gaussian KL closed form (lines 203-221), its `∂/∂Σ_1` (lines 227-232), the sandwich-transport application (line 234), the assembled boxed σ-gradient with coefficient `-2 Σ_i^{-1}` (Eq. eq:Sigma_gradient_final, lines 256-271), the fixed-point equation after dropping the softmax-summed correction (Eq. eq:sigma_fixed_point_beta, lines 282-290), and the SPD Hessian (lines 377-381) — and explicitly disclaims the global-uniqueness gap (line 387) so that nothing is overstated mathematically.

## Position

§B's six mathematical sub-claims (α through ζ) hold against primary-source canon, but the chapter is **not** "complete and self-contained" as the compound claim asserts. Three concrete defects undermine the self-containment prong, and a fourth gap weakens the local-stability argument beyond what the line-387 disclaimer covers. The claim as written should not stand without remediation.

## Evidence

### What survives canon verification (granted)

The math layer of §B is correct against canon:

- **Sub-claim α (Gaussian KL).** Eq. eq:gaussian_KL lines 203-221 matches `D(N(μ_1,K_1) ‖ N(μ_2,K_2)) = (1/2)[log(|K_2|/|K_1|) + tr(K_2^{-1}K_1) + (μ_1-μ_2)^⊤ K_2^{-1}(μ_1-μ_2) - d]` per [Cover-Thomas 2006 §8.6 Eq. 8.69], [Bishop 2006 §2.3.6 around Eq. 2.121], [Murphy 2012 §2.3.2]. The `(μ_2 - μ_1)` vs `(μ_1 - μ_2)` convention is symmetric.
- **Sub-claim β (∂KL/∂Σ_1).** Direct calculation via [Petersen-Pedersen Matrix Cookbook §9.1 Eq. 75] (`∂log|X|/∂X = X^{-⊤}`) and [§9.4 Eq. 100] (`∂tr(AX)/∂X = A^⊤`) gives `∂KL/∂Σ_1 = (1/2)[-Σ_1^{-1} + Σ_2^{-1}]` for symmetric Σ_1. Lines 227-232 match exactly.
- **Sub-claim γ (sandwich transport).** `Ω_{ij}q_j = N(Ω_{ij}μ_j, Ω_{ij}Σ_jΩ_{ij}^⊤)` is the standard push-forward of a Gaussian under a linear map per [Bishop 2006 §2.3.3], consistent with the project's hard constraint per CLAUDE.md.
- **Sub-claim δ (coefficient -2).** The arithmetic `(1/2)[-Σ_i^{-1} + Σ_p,i^{-1}] + (1/2)Σ_j β_{ij}[-Σ_i^{-1} + (Ω_{ij}Σ_jΩ_{ij}^⊤)^{-1}] = (1/2)[-(1+Σ_j β_{ij})Σ_i^{-1} + Σ_p,i^{-1} + Σ_j β_{ij}(Ω_{ij}Σ_jΩ_{ij}^⊤)^{-1}]` collapses to `(1/2)[-2Σ_i^{-1} + ...]` via `Σ_j β_{ij} = 1`. Lines 256-271 are exact.
- **Sub-claim ε (softmax simplex).** `Σ_j ∂β_{ij}/∂Σ_i = ∂(Σ_j β_{ij})/∂Σ_i = ∂(1)/∂Σ_i = 0` follows from the simplex constraint, as the chapter states. The fixed-point equation eq:sigma_fixed_point_beta correctly drops the `Σ_j (∂β_{ij}/∂Σ_i) K_{ij}` term under uniform-KL or τ→0/τ→∞ regimes (where the *individual* `∂β_{ij}/∂Σ_i` terms either align with uniform K_{ij} so the product vanishes by the simplex identity, or the β distribution collapses).
- **Sub-claim ζ (Hessian PD for a single KL).** `Hess[H,H] = (1/2)tr(Σ_1^{-1} H Σ_1^{-1} H) = (1/2)‖Σ_1^{-1/2} H Σ_1^{-1/2}‖_F²` is the affine-invariant SPD-manifold Hessian per [Smith 2005 "Covariance, subspace, and intrinsic Cramér-Rao bounds"]; positive definite for Σ_1 ≻ 0. Lines 377-381 are correct *for a single KL term*.

I grant the math. The four defects below operate on the *self-containment* and *completeness* prongs of the compound claim.

### Defect 1 — concrete cross-reference error at line 337

Supplementary line 337 (`Attention/GL(K)_supplementary.tex`) reads:

> "To obtain a coefficient of 1 on the alignment term, one requires the state-dependent prior coupling `α_i ≪ 1` (cf.\ main text, Section~3.6)"

Main-paper subsection structure (verified via grep on `\subsection{}` in `Attention/GL(K)_attention.tex`):
- §3.5 line 840 "Full Variational Free Energy" (`\label{sec:final_free_energy}`)
- §3.6 line 877 "Interpretation"
- §3.7 line 895 "State-Dependent Prior Precision" (`\label{sec:state_dependent_precision}`)

The state-dependent prior coupling `α_i` is derived at §3.7 lines 895-922, not §3.6. The supplementary cross-reference is wrong by one subsection.

This is a hard primary-source contradiction: the manuscript references its own §3.6 for a derivation that lives at §3.7. A reader following the cross-reference lands at §3.6 "Interpretation" and finds no α_i derivation. This is exactly the kind of self-referential inconsistency the prior twelve-debate audit series treated as a load-bearing defect when discovered.

### Defect 2 — `Λ_o` undefined at line 385

Supplementary line 385:

> "Hence the covariance alignment fixed-point is an attractor of the variational dynamics under the standing assumptions `Σ_j β_{ij}=1`, `Ω_{ij}` invertible, and `Λ_o` negligible relative to the inter-agent coupling."

Verified via grep `\Lambda_o|Lambda_o|\Lambda_\{o\}` across `GL(K)_supplementary.tex`: the symbol `Λ_o` appears exactly once in the entire file, at line 385. It is never defined. The closest semantic match is `R^{-1}` (Gaussian observation precision) at line 253, but the chapter does not state the identification `Λ_o := R^{-1}` or any equivalent. The stability disclaimer hinges on an undefined quantity, which compromises its rigor.

### Defect 3 — §B contains zero internal citations to its canonical sources

Verified via grep `\cite|\citep|\citet` across `GL(K)_supplementary.tex`, restricted to lines 180-387:

The §B section has **no `\cite`, `\citep`, or `\citet` calls anywhere within its 207 lines**. The closest citations are at line 49 (§A general references to Nakahara/Frankel/Blei/Amari) and line 441 (§C citing Gallier/Hall for the dexp differential). §B itself stands on uncited assertions of:

- The Gaussian KL closed form (textbook: [Bishop 2006 §2.3.6, Murphy 2012 §2.3.2, Cover-Thomas 2006 §8.6]).
- The matrix-calculus identities `∂log|X|/∂X = X^{-⊤}` and `∂tr(AX)/∂X = A^⊤` (textbook: [Petersen-Pedersen Matrix Cookbook §9.1, §9.4; Magnus-Neudecker 2019]).
- The Gaussian push-forward `y = Ax + b ⇒ y ~ N(Aμ+b, AΣA^⊤)` (textbook: [Bishop 2006 §2.3.3]).
- The symmetrized Kronecker product notation `⊠_{sym}` (literature: [Smith 2005], SPD-manifold treatments).

The editorial standard set by the prior debate on §3.7's Gamma-prior regularizer required adding `\citep{bishop2006pattern,murphy2012machine}` (visible at line 921 of `GL(K)_attention.tex`). The same standard applied to §B would require analogous citations. §B currently fails this standard.

### Defect 4 — Hessian PD argument does not cover F_i, only a single KL term

Supplementary line 376:

> "Local stability of the equilibrium [eq:sigma_fixed_point_beta] follows from the positive-definiteness of the Hessian. For the Gaussian KL terms `D_KL(N(μ_1,Σ_1) ‖ N(μ_2,Σ_2))` viewed as a function of `Σ_1` at fixed `Σ_2`,

then exhibits Eq. line 377-381 which is the Hessian of a *single* Gaussian KL with respect to Σ_1 at fixed Σ_2.

The fixed point being analyzed, however, is for `F_i = D_KL(q_i ‖ p_i) + Σ_j β_{ij}(Σ_i) D_KL(q_i ‖ Ω_{ij}q_j) - obs_term`. The full Hessian of F_i with respect to Σ_i contains two terms the chapter does not display:

1. A cross term `2 Σ_j (∂β_{ij}/∂Σ_i)(∂K_{ij}/∂Σ_i)` from `∂²(β_{ij}K_{ij})/∂Σ_i² = 2(∂β/∂Σ)(∂K/∂Σ) + β(∂²K/∂Σ²) + K(∂²β/∂Σ²)`.
2. A `Σ_j (∂²β_{ij}/∂Σ_i²) K_{ij}` term from the softmax curvature.

The simplex identity `Σ_j ∂β_{ij}/∂Σ_i = 0` constrains only the *sum* of the linear ∂β/∂Σ_i, not its quadratic interaction with ∂K_{ij}/∂Σ_i (which can be non-zero even when individual K_{ij} are uniform, since the gradients ∂K_{ij}/∂Σ_i need not be uniform). These contributions can be of either sign per [Wainwright-Jordan 2008 §3], where softmax-coupled mean-field fixed points generically have indefinite local Hessians off-equilibrium.

The chapter's argument therefore exhibits Hessian positive-definiteness for a *single uncoupled KL*, then asserts local stability for the *coupled F_i*. The line-387 disclaimer ("we do not claim a contractive proof of global uniqueness here") covers *global* uniqueness but not the *local* completeness gap. Either the softmax-induced contributions need a separate argument, or the chapter should explicitly invoke the envelope theorem at the fixed point (where `∂β/∂Σ_i` vanishes term-by-term per the chapter's uniform-KL standing assumption) to drop them.

### Citations

- [Bishop 2006 §2.3.6 Eq. 2.121] — Gaussian KL closed form. Standard form matches Eq. eq:gaussian_KL.
- [Murphy 2012 §2.3.2] — Gaussian KL closed form, alternative derivation.
- [Cover-Thomas 2006 §8.6 Eq. 8.69] — information-theoretic Gaussian KL.
- [Petersen-Pedersen Matrix Cookbook §9.1 Eq. 75, §9.4 Eq. 100] — `∂log|X|/∂X = X^{-⊤}` and `∂tr(AX)/∂X = A^⊤`.
- [Magnus-Neudecker 2019 "Matrix Differential Calculus"] — symmetric-matrix derivatives.
- [Smith 2005 "Covariance, subspace, and intrinsic Cramér-Rao bounds"] — SPD Hessian and the symmetrized Kronecker product.
- [Wainwright-Jordan 2008 §3 "Mean-field methods"] — softmax-coupled mean-field fixed-point Hessians.
- `Attention/GL(K)_supplementary.tex:337` — section-number drift (cites §3.6, should cite §3.7).
- `Attention/GL(K)_supplementary.tex:385` — `Λ_o` undefined.
- `Attention/GL(K)_supplementary.tex:180-387` — §B contains zero `\cite/\citep/\citet` calls (verified by grep).
- `Attention/GL(K)_attention.tex:895` — §3.7 "State-Dependent Prior Precision," labeled `sec:state_dependent_precision`, is the actual location of the α_i derivation.
- `Attention/GL(K)_attention.tex:921` — §3.7's Gamma-prior regularizer cites `bishop2006pattern, murphy2012machine`, establishing the editorial standard §B does not meet.

## Falsification conditions

This red attack fails (claim survives) if all four conditions hold:

1. The cross-reference at supplementary line 337 is shown to actually point to §3.7 (e.g., the user has a more recent version where §3.6 is "State-Dependent Prior Precision" — verified by re-checking the .tex line numbers).
2. `Λ_o` at supplementary line 385 is defined somewhere in §B or earlier in the supplementary that I missed (verifiable by exhaustive grep).
3. The editorial standard for §B does not require internal citations to Bishop/Murphy/Cover-Thomas/Petersen-Pedersen — for example, if §A's general reference at line 49 is judged to license uncited use of standard textbook results throughout §B.
4. The Hessian argument at lines 375-385 is judged sufficient for local stability under the chapter's standing assumptions (`Σ_j β_{ij} = 1`, `Ω_{ij}` invertible, `Λ_o` negligible) when read together with the envelope theorem at §3.5 — i.e., the chapter is read as implicitly invoking `Σ_j ∂β_{ij}/∂Σ_i = 0` at the uniform-KL fixed point so that the softmax-induced second-order terms vanish term-by-term there.

Conditions (1) and (2) are factual and verifiable. Condition (3) is an editorial-threshold judgment. Condition (4) is a charitable-reading judgment.

If conditions (1), (2), and (3) fail (which the evidence supports), the compound claim's "self-contained" prong is materially compromised — independently of whether the math sub-claims α-ζ survive (they do).
