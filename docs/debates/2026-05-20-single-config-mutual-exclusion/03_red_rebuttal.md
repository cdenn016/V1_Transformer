# Red Rebuttal — single-config-mutual-exclusion

## Concession

I grant the geometric identity on which blue's witness rests. For any orthogonal `Ω ∈ O(d_h)` and any isotropic per-block covariance `Σ_h = σ² · I_{d_h}`, the (0,2)-tensor sandwich evaluates to
`Ω · (σ² I) · Ωᵀ = σ² · ΩΩᵀ = σ² · I`,
so the diagonal-σ kernel at `transformer/vfe/non_flat.py:539` and the full sandwich at `transformer/vfe/prior_bank.py:329` agree bitwise on the isotropic fiber. The SO(N) `exp(skew)` exponentials produced by `fused_block_matrix_exp_pairs` are orthogonal to fp32 round-off (blue's measurement: 2.38e-7), and `transformer/core/vfe_utils.py:1131-1141` does enforce isotropic projection on `σ_q` after each E-step retraction — so the q-side invariant is real for as long as the witness lives.

This concession is narrow and exact. It is the *geometric* fact. The disputed question is whether that fact, on the chosen fiber, constitutes "realizing the exact (0,2)-tensor sandwich transport" in the sense the claim demands. I argue below it does not — for two independent reasons that blue's own opening flags as falsification conditions and then leaves un-defended.

## Core attack

Blue's witness satisfies the literal text of the existential by making the sandwich transport into the identity map on Σ. On the chosen fiber `{σ² · 1_K : σ² > 0}` — a 1-dimensional subspace of the K-dimensional σ-space — the transformation rule `Σ → Ω Σ Ωᵀ` evaluates to `Σ → Σ` for every admissible `Ω` and every admissible Σ. Transport never transforms the covariance. The witness "realizes" the sandwich the way "f(x) = x" realizes "any linear map": at the cost of being the trivial linear map.

Blue's falsification condition #6 names this exactly:

> "The user intended the claim to exclude degenerate witnesses — i.e., to require that the transport actually transport (`Ω σ²I Ωᵀ ≠ σ²I` for the chosen Ω). The literal text of the claim does not exclude fixed points; if the judge reads the existential as implicitly demanding non-vacuous transport, the witness above does not survive."

This is not a defense. It is a concession that under the non-vacuous reading the witness fails. The non-vacuous reading is the one the claim demands: PIFB:1619-1626 invokes the sandwich form as the operation that propagates non-trivial Σ-information across the gauge orbit; Nakahara 2003 §10.3 derives the sandwich `T → ρ(g)ᵀ T ρ(g)` as the rule by which (0,2)-tensors *transform* — not the rule under which a chosen (0,2)-tensor stays fixed. A witness that lives on the orbit's fixed-point locus does not exercise the rule; it sidesteps it.

The drift attack closes the residual ambiguity. Blue's E-step isotropy enforcement at `vfe_utils.py:1131-1141` projects `σ_q` to its block-mean after each retraction. The encode prior σ — `base_sigma` derived from `base_log_sigma`, the `nn.Parameter(torch.full((K,), sigma_init_log))` at `transformer/vfe/prior_bank.py:194` — has no analogous projector. A grep across `transformer/vfe/prior_bank.py` for `isotropic_covariance` returns zero matches; the prior bank's sandwich call at `prior_bank.py:329` (`exp_h_f32 @ sigma_diag @ exp_h_f32.transpose(-2, -1)`) reads `base_sigma` raw. The M-step gradient `∂F/∂base_log_sigma` is anisotropic by construction (the cross-entropy term, the KL-to-q term, and the hyper-prior KL term all contribute different per-coordinate gradients), so after step 1 of training `base_sigma` leaves the isotropic subspace. From that step onward, `Ω diag(σ_anisotropic) Ωᵀ` has off-diagonals of order `(σ_max - σ_min) · |Ω_ij · Ω_ik|`, and blue's own finite-difference check on the anisotropic case (`Full-sandwich off-diagonals max: 1.83`) shows these are O(1), not fp32 noise. The diagonal-σ kernel drops them. Blue's falsification condition #7 already concedes:

> "If red argues the claim implicitly requires the equivalence to persist through training, the witness does not survive that stronger reading either."

The asymmetry is structural: the codebase has built an isotropy projector for `σ_q` but not for `σ_p`. The witness lives only at initialization, and only on the literal-text reading that admits identity transport as "transport."

The geometric content of the canonical sandwich, per [Nakahara 2003 §10.3], is the transformation rule for non-trivially-anisotropic (0,2)-tensors under a non-trivially-non-orthogonal group action. Blue's witness restricts the group to its compact orthogonal subgroup and the covariance to the scalar-multiple-of-identity ray. Blue's pre-emptive concession ("the defense narrows the gauge group from the manuscript's general GL(K) to its compact subgroup SO(N), and the covariance to its scalar-multiple-of-identity subspace") understates what this narrowing does: it removes the entire dynamical content of the rule the claim is supposed to exhibit. The PIFB manuscript identifies the gauge group of the language-modeling reduction as `GL(d_head)^H ⊂ GL(K)` (PIFB:1785) — neither SO(N) nor the scalar-σ ray is part of that reduction.

## Defense

My opening's load-bearing point was that `transformer/vfe/config.py:566-573` and `transformer/vfe/config.py:484-507` together force a binary choice at construction: either `gauge_parameterization='omega_direct'` and `diagonal_covariance=True` (the omega-direct sandwich-free path), or `exact_full_cov_decode=True` and `diagonal_covariance=False` (the full-sandwich path). The two guards target opposite truth values of the same flag.

Blue does not contest the existence of this gate. Blue's defense is that on a measure-zero fiber of the configuration space the two paths coincide because the sandwich becomes the identity. That defense does not survive any reading of "realizes the exact sandwich transport" that requires the transport to actually transform Σ:

- *Literal-text reading, one forward pass, vacuous transport admitted.* Blue's witness survives. The witness exhibits the sandwich form `Σ → Ω Σ Ωᵀ` on inputs where the form evaluates to the identity. Whether this counts as "realizing" the form is a question for the judge; blue's own falsification #6 admits the negative reading is open.

- *Non-vacuous-transport reading.* Blue's witness fails by blue's falsification #6. The diagonal subspace is *not* invariant under non-orthogonal `Ω`, and the isotropic ray is invariant under *every* orthogonal `Ω` — so the witness realizes "exact sandwich" only by selecting the intersection where the sandwich is trivial.

- *Persistence-through-training reading.* Blue's witness fails by blue's falsification #7. The prior-bank `base_log_sigma` parameter at `prior_bank.py:194` carries no isotropy projector (zero `isotropic_covariance` references in `prior_bank.py`); the M-step gradient on cross-entropy is generically anisotropic; after one optimizer step the witness's invariant breaks; `prior_bank.py:329` then computes a sandwich with O(1) off-diagonals that the omega-direct path's diagonal-σ kernel at `non_flat.py:539` drops. The "exact sandwich" claim fails on the second forward pass.

The canonical sandwich rule of [Nakahara 2003 §10.3] and PIFB:1619-1626 has dynamical content only when Σ is genuinely anisotropic and `Ω` is genuinely non-orthogonal. The PIFB manuscript at line 1785 fixes the gauge group of the language-modeling reduction as `GL(d_head)^H`, which contains the orthogonal subgroup as a measure-zero submanifold but is not generated by it. The encode prior bank at `prior_bank.py:316-329` walks `irrep_dims` and applies `A_h diag(s_h) A_h^T` per block on a `base_sigma` that the codebase makes no effort to keep isotropic. Blue's witness asks the judge to accept that the canonical sandwich form is "realized" by a configuration in which the sandwich is the constant-Σ identity map at initialization and a strict approximation thereafter — on a gauge group blue itself concedes is narrower than the manuscript's. Under every reading of "realizes" except the most permissive, the mutual exclusion stands.

The claim as stated requires a single `VFEConfig` that simultaneously realizes both canonical forms. Blue has produced a `VFEConfig` that realizes form (a) (the omega-direct retraction at `omega_direct.py:345, 390`, unconested) and exhibits form (b) only as the identity. Realizing the identity is not realizing the transport.
