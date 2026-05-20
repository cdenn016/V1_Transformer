# Red Rebuttal — supplementary-covariance-dynamics

## Concession

Red concedes the mathematical-purity prong of the compound claim — sub-claims α (Gaussian KL closed form, line 203–221), β (`∂KL/∂Σ_1 = (1/2)[-Σ_1^{-1} + Σ_2^{-1}]`, lines 227–232), γ (sandwich-product transport `Ω q_j = N(Ωμ_j, ΩΣ_jΩ^⊤)`, line 234), δ (the `-(1 + Σ_j β_{ij}) = -2` assembly, lines 256–271), ε (simplex identity `Σ_j ∂β_{ij}/∂Σ_i = 0` and the homogeneous reduction `Σ_∞ = Σ_0`, lines 279–315), and ζ (positive-definite Hessian `(1/2) Σ_1^{-1} ⊠_{sym} Σ_1^{-1}`, lines 377–381) — all six survive primary-source verification. Bishop2006 §2.3.6 Eq. 2.121, Murphy2012 §2.3.2, CoverThomas2006 §8.6 Eq. 8.69, PetersenPedersen MatrixCookbook §9.1 Eq. 75 and §9.4 Eq. 100, and the Smith2005 / Pennec2006 SPD Hessian canon match the derivation chain term by term. The math is canon.

Red also concedes blue's gauge-equivariance defense in spirit: under the global diagonal gauge action `g_i = g` for all `i`, the precision `Σ_i^{-1}` transforms as `g^{-⊤} Σ_i^{-1} g^{-1}`, the prior precision transforms identically, and the transported alignment term `(Ω_{ij}Σ_jΩ_{ij}^⊤)^{-1}` becomes `g^{-⊤}(Ω_{ij}Σ_jΩ_{ij}^⊤)^{-1} g^{-1}` because `Ω_{ij} → gΩ_{ij}g^{-1}` under the same action. All three terms in the boxed Eq. (eq:Sigma_gradient_final) get conjugated by `g^{-⊤}` on the left and `g^{-1}` on the right — covariant transformation of the σ-gradient as a (2,0)-tensor density. The hard-constraint CLAUDE.md test passes, even though the chapter does not state it.

The compound claim's mathematical-purity prong holds.

## Core attack

The compound claim has two prongs, and blue's defense addresses only one. The literal language of `00_claim.md` is "complete and mathematically/theoretically pure as a **self-contained** supplementary chapter." The self-containment prong fails on four concrete, citeable load-bearing breaks — three of which blue characterizes as "single-token editorial fixes" and one of which blue's defense does not address at all.

**Break 1 — `α_i` is used in §B at line 275 before §B defines it.** Line 275 of `GL(K)_supplementary.tex` reads:

> "with unit coefficient `α_i = 1`."

This is the first use of `α_i` in §B. The chapter has not previously introduced `α_i`; the surrogate free energy at line 187–195 has unit prior weight, not `α_i`. The next use at line 337 references the symbol as if it had been introduced ("the state-dependent prior coupling `α_i ≪ 1`"). A reader of §B in isolation encounters an undefined symbol at line 275 and a back-pointer to the wrong main-paper section at line 337 — the symbol is never grounded inside §B itself.

The main paper at `GL(K)_attention.tex` line 408 settles the canonical location:

> `$\alpha$ ($\alpha_i$) & $\mathbb{R}^+$ & Prior precision; state-dependent variant (Sec.~\ref{sec:state_dependent_precision})`

`sec:state_dependent_precision` is §3.7 (the table at line 408 labels it that way and the section heading at line 895 confirms). So the supplementary line 337 reference to "§3.6" is wrong, AND the §B-internal introduction of `α_i` at line 275 is missing. These are not the same fix: line 337 is a `3.6 → 3.7` edit, but line 275 needs either a definition of `α_i` or a forward-cross-reference to §3.7 explaining where the symbol comes from.

**Break 2 — `Λ_o` at line 385 is out of step with §B's own notation at line 253.** §B introduces the observation-noise contribution at line 253 using `R^{-1}` ("e.g., `(1/2)R^{-1}` for Gaussian observations with noise covariance `R`"). Line 385 then writes:

> "`Λ_o` negligible relative to the inter-agent coupling"

`Λ_o` is nowhere defined in §B. The most charitable reading — blue's reading — is `Λ_o := R^{-1}`. But §B uses `R^{-1}` at line 253 and `Λ_o` at line 385 for the same physical quantity, 130 lines apart, with zero glossary cross-reference. This is not a single-symbol fix as blue claims; it is a notational inconsistency within §B between the equation where the observation-precision term is introduced and the stability disclaimer that invokes it.

**Break 3 — `⊠_{sym}` is used at line 380 without definition.** §B writes:

> `∂²D_{KL}/∂Σ_1∂Σ_1 = (1/2) Σ_1^{-1} ⊠_{sym} Σ_1^{-1}`

with the gloss "the symmetrized Kronecker product acting on Sym(K)" but no formula, no citation to Smith2005 or to the matrix-Kronecker calculus where `⊠_{sym}` is defined, and no equation showing how the Hessian acts on a tangent direction `H ∈ Sym(K)`. The form `Hess[H, H] = (1/2) tr(Σ_1^{-1} H Σ_1^{-1} H)` that blue cites in its sub-claim ζ defense is the standard one, but §B does not write it. A reader familiar with the SPD-manifold canon will infer the meaning; a reader who is not has no recourse inside the chapter.

**Break 4 — Zero `\cite{}` calls in 207 lines of §B.** I ran the count: `awk 'NR>=180 && NR<=387' GL(K)_supplementary.tex | grep -c '\\cite{'` returns `0`. §B presents four canonical results — Gaussian KL closed form (line 203–221), the matrix-calculus derivative `∂KL/∂Σ_1` (line 227–232), the Gaussian linear pushforward `Ωq_j = N(Ωμ_j, ΩΣ_jΩ^⊤)` (line 234), and the SPD Hessian `Σ_1^{-1} ⊠_{sym} Σ_1^{-1}` (line 380) — without a single inline citation to Bishop2006, Murphy2012, CoverThomas2006, PetersenPedersen, Smith2005, or Pennec2006. The opening sentence at line 185 ("a well known result in information geometry") names no source.

Blue's defense at lines 80–86 of `02_blue_opening.md` argues that this pattern is "materially less severe" than the §3 `D Gamma prior` debate because §B does not claim novelty for these calculations. The standard the prior debate series established is different. In the `D Gamma prior` debate the verdict against blue was that the manuscript identified the regularizer `R(α_i) = b_0 α_i - c_0 log α_i` as "the negative log-density of a Gamma(α_i; c_0+1, b_0) distribution — the conjugate prior for the precision parameter of a Gaussian likelihood" only after citation pressure; the canonical Bayesian construction needed to be named. §B has the same problem at greater multiplicity. Four canonical correspondences (Bishop §2.3.6 / Murphy §2.3.2 / CoverThomas §8.6 for the Gaussian KL; PetersenPedersen §9.1, §9.4 for the matrix-calculus derivatives; Bishop §2.3.3 for the linear pushforward; Smith2005 / Pennec2006 for the SPD Hessian) are each presented as standard without identification. The aggregate citation gap in §B is **four uncited canonical correspondences in 207 lines** versus the Gamma-prior debate's single under-labeled construction. Larger, not smaller.

**Break 5 — The §B.2.2 alignment-dominated regime is gated by `α_i ≪ 1`, but the canonical framework default is `α_i = 1`.** Line 337 states the standing assumption explicitly: "one requires the state-dependent prior coupling `α_i ≪ 1`." Per the main paper at `GL(K)_attention.tex` line 948 ("In our experiments we study both `R(α_i) ≠ 0` and `R(α_i) = 0` with `α_i = α = 1`") and per the CLAUDE.md project policy that the framework defaults to `α_i = 1`, the §B.2.2 regime is theoretical-only — it is never exercised in the framework's empirical setup unless `R(α_i) ≠ 0` is enabled and additionally `α_i^*` happens to fall into the `≪ 1` range. §B writes the regime equation as if it were a substantive prediction; it should be labeled as a theoretical limit not exercised by the canonical configuration.

## Defense

Blue's strongest move is the falsification-condition concession at line 96 of `02_blue_opening.md`: "If red shows that the `3.6` vs `3.7` confusion at line 337 actually corresponds to a substantively different coupling mechanism ... blue concedes Sub-claim ε beyond editorial." Red does not need to reach beyond editorial to defend the compound claim's failure. The §B.2.2 regime equation (eq:beta_weighted_precision) at line 339–348 depends on the standing assumption that the coefficient of `Σ_i^{-1}` reduces from `-(1+α_i)` to approximately `-1`. This standing assumption is a substantive mathematical requirement of the regime analysis. The chapter places the canonical home of `α_i` at the wrong main-paper section AND does not introduce `α_i` inside §B at all (Break 1 above). The drift is therefore not a single-token editorial fix as blue argues — it is the failure of self-containment for the load-bearing variable of §B.2.2's regime analysis.

Blue's second-strongest move is the surrogate-vs-canonical defense at line 184–185 of the supplementary: "the covariance gradient is identical under both forms because the attention entropy does not depend on `Σ_i`." Red grants this is correct. The σ-gradient of the surrogate `Σ_j β KL` and of the canonical F (which adds `τ β log(β/π)`) differ by `-τ^{-1} Cov_β(KL, ∇_Σ KL)` — but `∇_Σ` of the entropy term is zero because the entropy depends on `β` (which depends on `Σ_i` through KL) but the entropy itself does not depend on `Σ_i` directly. By the simplex identity `Σ_j ∂β/∂Σ_i = 0` and the cancellation in the entropy's derivative, the surrogate and canonical σ-gradients agree. Sub-claim δ stands.

Where red's position holds: the compound claim asserts §B is "complete and mathematically/theoretically pure as a **self-contained** supplementary chapter." Self-containment requires every symbol used in §B be defined in §B or be a primitive of the canon explicitly cited. §B fails this on five counts (`α_i` introduced without §B-internal definition; `Λ_o` introduced as a third notation for `R^{-1}`; `⊠_{sym}` used without a formula; four canonical correspondences with zero `\cite{}`; regime analysis gated by a parameter the canonical configuration does not exercise). The aggregate is at least as severe as the §3 `D Gamma prior` verdict.

The mathematical-purity prong of the compound claim holds (concession). The self-containment prong fails on five concrete, citeable load-bearing breaks. By the worst-load-bearing-sub-claim rule stated in `00_claim.md` line 47 ("A compound verdict ... should reflect the worst load-bearing sub-claim"), the verdict should be **RED_WINS-narrow** with five editorial action items:

1. Define `α_i` at first use in §B (line 275) or insert a forward-cross-reference using `\ref{sec:state_dependent_precision}`.
2. Fix the line 337 cross-reference from `Section~3.6` to `\ref{sec:state_dependent_precision}` (which is §3.7).
3. Resolve the `R^{-1}` vs `Λ_o` notational inconsistency between line 253 and line 385.
4. Either write the explicit Hessian formula `Hess[H,H] = (1/2) tr(Σ_1^{-1} H Σ_1^{-1} H)` at line 380 or cite Smith2005 / Pennec2006 for the `⊠_{sym}` notation.
5. Add four `\cite{}` calls: `\cite{bishop2006pattern}` at line 201 for the Gaussian KL, `\cite{petersen2008matrix}` at line 224 for the matrix-calculus derivatives, `\cite{bishop2006pattern}` at line 234 for the linear pushforward, `\cite{pennec2006riemannian}` at line 383 for the SPD Hessian.
