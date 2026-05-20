# Verdict — subclaim-D-value-identification

## Outcome

BLUE_WINS

## Decisive evidence

`Attention/GL(K)_attention.tex:1339` (inside the same §5.2.3 "Complete Attention Formula" subsection as the value-aggregation reduction at `:1316`):

> "The gauge identification operates on the invertible GL(d_head) head-space factor M_h^a of the rectangular per-head projections (Eq. eq:head_space_kernel); the ambient transformer kernel W_Q^a (W_K^a)^T, of rank at most d_head, is the low-rank lift of this head-space kernel through the isometric subspace embeddings U_Q^a, U_K^a."

Combined with the general thin-SVD construction at `Attention/GL(K)_attention.tex:1245`:

> "...one factors each rectangular map via thin SVD as W_Q^a = U_Q^a A_Q^a and W_K^a = U_K^a A_K^a, where U_Q^a, U_K^a ∈ R^{d_model × d_head} have orthonormal columns (isometric subspace embeddings) and A_Q^a, A_K^a ∈ GL(d_head) are invertible head-space maps."

The thin-SVD factorization at `:1245` is stated as a property of any rectangular `d_model × d_head` projection of rank `d_head`; nothing in the construction is specific to Q or K, so it applies to V by the same argument (`W_V^a = U_V^a A_V^a`). The manuscript's `W_V ∈ R^{d_k × d_k}` at `:1310` is the invertible head-space factor `A_V^a`. The lift to Vaswani's rectangular `W_V ∈ R^{d_model × d_v}` in `external_canon_transformers.md §1` is the same `U_V^a · A_V^a` construction; the isometric `U_V^a` is absorbed into the per-head slice of `W_O`, which is precisely the standard concat-then-`W_O` multi-head structure of [Vaswani2017 §3.2.1].

## Reasoning

Sub-claim D is conditional: "Under the constant-gauge specialization `Ω_{ij} = Ω`." Red conceded the pullout step in its rebuttal (`03_red_rebuttal.md:5`): "Under the explicit constant-gauge specialization `Ω_{ij} = Ω` declared at `:1325`, the pullout `Σ_j β_{ij} Ω μ_j = Ω · Σ_j β_{ij} μ_j` is an algebraic identity by linearity of the finite weighted sum ... The pullout step is exact." That concession removes the algebraic-identity attack and leaves three residual attacks: (a) the square/rectangular shape mismatch, (b) the absorption-is-renaming critique, and (c) the parameter-space-sub-class observation.

On (a), red's claim that "the manuscript does not exhibit that lift" for V is contradicted by `:1339` directly. The manuscript itself, in the same `§5.2.3` Complete Attention Formula subsection that contains the value-aggregation reduction, explicitly states the lift mechanism and identifies the gauge-theoretic `W_V` with the invertible head-space factor `A_V^a`, with the ambient rectangular projection produced by the low-rank lift through an isometric embedding. The thin-SVD argument at `:1245` is general for any rank-`d_head` rectangular `d_model × d_head` projection; V satisfies that condition under [Vaswani2017 §3.2.1]'s multi-head construction, so the same lift applies verbatim. Red's residual concern that `U_V^a` introduces structure not pinned down by the gauge identification is correct, but the manuscript closes that gap by absorbing `U_V^a` into `W_O`, which is the standard multi-head concat-then-`W_O` operation; this is not a separate, unjustified architectural addition.

On (b), red's "renaming, not derivation" framing is rebutted by blue's explicit four-step algebraic chain in `03_blue_rebuttal.md:19–25`: each step is an identity (linearity of the sum, linearity of `Ω`, definition of `V_j`, regrouping to standard form). The introduction of a symbol `V_j` for the quantity `Ω μ_j` inside a chain of equalities is how reductions are written; calling that "renaming" would invalidate every specialization-reduction in physics. The standard meaning of "exact reduction" in this context — used in the GR→Newton analogy red did not contest — is equation-level identity under a stated specialization, not bijection between full parameter manifolds.

On (c), blue made a substantial concession (`03_blue_rebuttal.md:5`) that the gauge framework's collapse of attention transport and value transport onto a single `Ω` is a strictly stronger structural condition than standard transformers' independent learning of `W_Q, W_K, W_V`. Blue accepted that the reduction does not recover the full standard parameter manifold and identified its image as a sub-family on which the Q–K kernel and V projection are tied to a common gauge object. The remaining question is whether this concession invalidates the "exact" qualifier in sub-claim D. It does not. The claim is explicitly conditional on `Ω_{ij} = Ω`; under that precondition the value-aggregation equation reduces algebraically to standard form. The parameter-space-sub-class observation is a statement about which standard transformers admit a gauge representation, not about whether the algebraic reduction is exact. Red's stronger reading — that "exact reduction" must mean a bijection on the full parameter manifold — is not the meaning in physics or mathematics and would, as blue noted, classify every specialization-reduction as non-exact.

The shape mismatch attack — red's load-bearing strike — is resolved by `:1339` directly. The remaining attacks fail under blue's algebraic chain and the conditional-precondition reading of the claim.

## Action

Accept sub-claim D as defended: under the explicit constant-gauge specialization `Ω_{ij} = Ω`, the value-aggregation reduction `μ̂_i = Σ_j β_{ij} Ω μ_j → Σ_j β_{ij} V_j` is exact at the equation level, with the manuscript's square `W_V ∈ R^{d_k × d_k}` identified with the invertible head-space factor `A_V^a` and lifted to Vaswani's rectangular `W_V ∈ R^{d_model × d_v}` via the thin-SVD construction supplied at `Attention/GL(K)_attention.tex:1245` and `:1339`.

Two clarifications should be folded into the manuscript at `:1310` to forestall future challenges:

1. State explicitly that `d_k` at `:1310` denotes the per-head dimension `d_head` (matching the usage at `:1336` and `:1339`), not the ambient model dimension. The current notation is silent and red's rectangular attack hinges on this ambiguity.

2. State explicitly that the thin-SVD lift at `:1245` applies to `W_V` by parallel construction, with the isometric `U_V^a` absorbed into the per-head slice of `W_O` under the standard concat-then-`W_O` multi-head structure. The manuscript currently states the lift only for Q/K and lets the V parallel be inferred.

The image of the reduction is a parameter sub-family of standard transformers — the family in which `W_Q W_K^T` and `W_V` share a common underlying gauge object `Ω` modulo the bilinear and thin-SVD factorization freedoms. The manuscript should be explicit that the reduction is exact onto this sub-family but does not exhaust the full standard parameter manifold, in line with blue's bounded concession at `03_blue_rebuttal.md:5–7`.
