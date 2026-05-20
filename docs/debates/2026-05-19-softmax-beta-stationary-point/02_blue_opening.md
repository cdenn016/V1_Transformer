# Blue Opening — softmax-beta-stationary-point

## Steelman (opposing position)

The strongest red attack runs as follows. Equation \eqref{eq:mixture_joint} at line 688 defines `P(k|z=j) = N(k; Ω_ij μ_j, Ω_ij Σ_j Ω_ij^T)`, i.e. the prior component itself depends on `q_j`, which is another variational quantity. Plugging a recognition object back into the generative model and then "deriving" the recognition update is not a derivation — it is a fixed-point definition. The manuscript's own line 697 admits this: the components "depend on the variational posteriors $q_j$ of other agents, making the generative model itself a function of variational quantities." Once you accept that, the row-Lagrangian at line 741 is best read as a *consensus-energy ansatz* (the second framing the manuscript offers), in which case the word "derived" overstates what is shown. A secondary attack: the manuscript writes only the equality Lagrangian (line 741) without the inequality multipliers `β_j ≥ 0`, never verifies second-order conditions, and at line 734 writes "energy minus entropy plus const" without flagging that "const" is `−Σ β log π`, which is β-dependent unless `π` is uniform. Strict reading: the proof shown is partial.

## Position

Under the canonical functional `F_align^(τ) = Σ_j [β_ij E_ij + τ β_ij log(β_ij/π_j)]` with `E_ij` treated as fixed in the inner optimization, `β_ij = π_j exp(−E_ij/τ)/Z_i` is the *exact, unique* interior stationary point and the *unique global minimum*. The row-Lagrangian shown at `Attention/GL(K)_attention.tex` lines 741–753 is mathematically valid; every intermediate step (the `+1` absorption, the simplex normalization, the τ-rescaling, the uniform-π specialization) survives symbolic verification. The self-referential structure flagged at line 697 is the standard coordinate-ascent variational EM picture and does not invalidate the inner optimization; it relocates the open question to *joint* fixed-point consistency, which is a separate claim from the headline.

## Evidence

**(1) Symbolic verification of line 747 → line 753 (the `+1` absorption).** A sympy session with `N=3`, symbolic `E_k`, `π_k > 0`, `β_k > 0`, `λ ∈ ℝ`, applied to the manuscript's Lagrangian

```
L = Σ_j β_j (E_j + log β_j − log π_j) − λ(Σ_j β_j − 1)
```

produces stationarity `∂L/∂β_k = E_k − λ + log(β_k/π_k) + 1 = 0`, parametric solution `β_k(λ) = π_k exp(−E_k + λ − 1)`, and after enforcing `Σ_k β_k = 1`,

```
β_1 = π_1 exp(E_2 + E_3) / (π_1 exp(E_2 + E_3) + π_2 exp(E_1 + E_3) + π_3 exp(E_1 + E_2))
    = π_1 exp(−E_1) / Σ_m π_m exp(−E_m)
```

with `sp.simplify(β_k − π_k exp(−E_k)/Z) = 0` for each `k`. This is identical to Eq. \eqref{eq:mixture_softmax_general} at line 753. The `+1` does not propagate to the result — it is absorbed into the dual variable `λ` and reabsorbed by the simplex normalization, exactly as the manuscript states implicitly. The intermediate step is mathematically valid.

**(2) Symbolic verification of the τ-rescaled form (lines 764–769).** Repeating the sympy session with the canonical functional `F_align^(τ) = Σ_j [β_j E_j + τ β_j log(β_j/π_j)]` yields stationarity `E_k − λ + τ + τ log(β_k/π_k) = 0` and, after normalization,

```
β_k = π_k exp(−E_k/τ) / Σ_m π_m exp(−E_m/τ)
```

with `sp.simplify(β_k − π_k exp(−E_k/τ)/Z(τ)) = 0`. The substituted minimum is

```
F_align^(τ)*(β*) = −τ log Σ_m π_m exp(−E_m/τ) = −τ log Z_i
```

matching the manuscript's claim at line 769 exactly (sympy returned `F* − (−τ log Z) = 0`). Sub-claim E is symbolically tight.

**(3) Strict convexity and uniqueness.** The Hessian of `f(β) = Σ_j β_j (E_j + log β_j − log π_j)` is diagonal with `∂²f/∂β_k² = 1/β_k > 0` on the open simplex (sympy computed `H = diag(1/β_1, 1/β_2, 1/β_3)`, eigenvalues `{1/β_k}`). `f` is strictly convex, and the equality constraint `Σ β = 1` is affine, so the constrained problem has a unique global minimum [Boyd & Vandenberghe §3.1.4, §5.5.1]. The manuscript does not state this explicitly, but it is a one-line fact and is what makes "exact stationary point" coincide with "unique global minimum" under the claim's hypotheses.

**(4) KKT inequality constraints are non-binding.** The full KKT system for `β ∈ Δ^N` carries `ν_j ≥ 0` multipliers for `β_j ≥ 0` with complementary slackness `ν_j β_j = 0`. Because `∂f/∂β_k = E_k + log β_k + 1 − log π_k` diverges to `−∞` as `β_k → 0⁺`, no interior stationary point can satisfy `β_k = 0`; the inequality constraints are strictly inactive and the equality-only Lagrangian at line 741 is sufficient. This is the standard treatment for entropy-regularized soft assignment [Cuturi 2013 *Sinkhorn Distances* §4]. The manuscript's choice to write only the equality Lagrangian is convention, not error.

**(5) The "augmented joint" framing at line 697 is standard coordinate-ascent VI.** The functional `F_align = KL[Q(k,z) ‖ P(k,z)]` at line 715 is exactly Form-1 of variational free energy `F[q] = E_q[log q − log p]` [Friston2010; BleiKuckelbirgJordan2017 §3, Eq. 14] applied to the joint over `(k, z)` with mean-field factorization `Q = q_i(k) β(z)`. Coordinate-ascent VI (CAVI) [BleiKuckelbirgJordan2017 §3.2] optimizes one factor at a time with the others held fixed; the line 741 Lagrangian *is* the β-block CAVI update, with `q_j` and `q_i` fixed. The block update is exact (not approximate) within each iteration. Joint fixed-point convergence is a separate property of CAVI — it is guaranteed monotone non-increase of F by the standard CAVI proof, with convergence to a local stationary point. The "self-referential" character flagged by the manuscript at line 697 is the textbook structure of mean-field VI, not a circularity in the inner-block derivation.

**(6) Forward-KL choice is the variational direction.** `KL(q ‖ p)` is the standard variational direction [BleiKuckelbirgJordan2017 §2; canon `external_canon_inference.md` §4]. The manuscript's `KL(q_i ‖ Ω_ij q_j)` matches this. Supplementary Appendix H (`Attention/GL(K)_supplementary.tex` line 1091) gives an independent uniqueness argument for the forward-KL choice within the exponential-family closure constraint; this is supporting evidence but not load-bearing for the present claim.

**(7) Structural identity with entropy-regularized soft assignment.** The functional `Σ_j β_j E_j − τ H(β) + τ KL(β ‖ π)` (with `H(β) = −Σ β log β`) is exactly the row-block of an entropy-regularized optimal-transport problem; its stationary point `β = π ⊙ exp(−E/τ)/Z` is the row-update of the Sinkhorn iteration [Cuturi 2013 §4, Eq. 2]. The manuscript's derivation is one block of this standard construction. This is the citation that resolves the "novel vs standard" question raised in `external_canon_inference.md` §1: the τβ log(β/π) term is *standard maxent regularization* in the soft-assignment literature, and the manuscript presents it as a Lagrangian (line 715, `F = KL[Q‖P]`, decomposed and minimized), not as a consequence of bare FEP. Within this framing, the derivation is canonical.

## Falsification conditions

This position is wrong if any of the following holds.

**(F1) The algebra at line 747 → line 753 is incorrect.** Specifically, if the symbolic identity `π_k exp(−E_k)/Σ_m π_m exp(−E_m) ≡ unique interior solution of L`'s stationarity equations under `Σ β = 1` fails. The sympy session in evidence (1) returns `sp.simplify(β_k − π_k exp(−E_k)/Z) = 0`; if a counter-computation contradicts this, the defense collapses on sub-claim C.

**(F2) The functional `f(β)` is not strictly convex on the open simplex.** If a counter-Hessian computation shows a non-positive eigenvalue, uniqueness fails and "exact stationary point" weakens to "one of multiple stationary points." Evidence (3) shows the Hessian eigenvalues are `{1/β_k}` on the open simplex; refuting this refutes uniqueness.

**(F3) The Form-1 free energy `F = KL[Q ‖ P]` at line 715 is not the standard variational free energy for the joint `(k, z)` problem.** If a primary source [Friston2010, BleiKuckelbirgJordan2017, ParrPezzuloFriston2022 Ch. 2] shows the canonical form for joint factorization differs from line 715, the structural-identity argument fails. This is the *outer* claim — that line 715 is a valid VFE; if it isn't, the whole derivation is unfounded.

**(F4) The "augmented joint" framing at line 697 collapses into vacuous self-reference.** If a primary source shows that block-CAVI updates require the prior components to be *parameter-free* in the variational quantities (i.e., cannot depend on `q_j`), the block β-update at line 741 ceases to be a valid CAVI block. Standard CAVI permits prior dependence on other variational factors via the mean-field decomposition of a *fixed* joint generative model (cf. [BleiKuckelbirgJordan2017 §3.2, Eq. 18]); if this is shown to be a misreading, sub-claim A collapses.

**(F5) The temperature substitution at lines 764–769 changes the stationary point.** If a counter-computation shows the τ-extended Lagrangian yields a different `β*` than `π exp(−E/τ)/Z`, sub-claim E fails. Evidence (2) refutes this symbolically; a contradicting computation would falsify the defense.

**(F6) The boundary `β_k = 0` is reachable at a stationary point.** If a counter-example shows an interior–exterior solution `β_k = 0` for some k with `ν_k > 0` such that the KKT system is satisfied, the equality-only Lagrangian at line 741 is insufficient. The `−log β_k → ∞` blow-up at the boundary forecloses this for the entropy-regularized functional; if `τ → 0` is taken before the optimization is performed, the boundary becomes reachable and the softmax form does not apply (it limits to the argmax — see [Cuturi 2013 §3]). For finite `τ > 0`, the defense stands.

## Concession in advance

The manuscript at line 697 hand-waves the "self-referential" question with two informal framings ("augmented joint", "consensus energy") and a one-sentence assurance that "in either framing, the resulting update equations are the same." This is the weakest paragraph in the derivation. The block-CAVI reading I gave under evidence (5) is the correct rigorous resolution, and the manuscript should state it explicitly (cite [BleiKuckelbirgJordan2017 §3.2]) rather than offer two non-equivalent informal pictures. The headline claim — that `β_ij = softmax(−KL/τ)` is the exact stationary point of `F_align^(τ)` with `E_ij` fixed — does not depend on this paragraph being tightened, but the manuscript's exposition does need the tightening. This is editorial, not substantive.

Additionally, the manuscript at line 734 writes "energy minus entropy plus const" without flagging that `const = −Σ β log π = ⟨−log π⟩_β` is β-dependent unless `π = 1/N`. The decomposition at line 730 itself is correct (`−log π` enters the softmax as a prior bias at line 753); the "+ const" framing at line 734 is uniform-π shorthand. The claim explicitly assumes uniform π, so this is consistent with the headline, but the manuscript should drop the "+ const" framing or add a one-line qualifier. Also editorial.
