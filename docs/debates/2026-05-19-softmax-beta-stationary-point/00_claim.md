# Claim — softmax-beta-stationary-point

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (manuscript `Attention/GL(K)_attention.tex` §4 lines 679–769, supplementary `Attention/GL(K)_supplementary.tex` §B opening at line 183; external canon for KKT/Lagrange-multiplier optimization on the simplex and the variational mean-field maxent derivation)
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge

## Claim

`β_ij = softmax_j(-D_KL(q_i ‖ Ω_ij q_j)/τ)` is the **exact** stationary point of the canonical alignment free energy
`F_align^(τ) = Σ_j [β_ij E_ij + τ β_ij log(β_ij/π_j)]`
with `E_ij = D_KL(q_i ‖ Ω_ij q_j)` and uniform prior `π_j = 1/N`, derived via the row-Lagrangian shown in `Attention/GL(K)_attention.tex` §4.6 "Deriving Attention from a Mixture-of-Sources Model" (lines 679–769), with every intermediate step (mixture-of-sources generative model → mean-field variational posterior → energy-entropy decomposition → KKT/Lagrange-multiplier optimization → softmax solution → temperature rescaling) mathematically valid.

## Sub-claims (compound — flagged for possible separate debates)

The headline is compound. Five load-bearing sub-claims:

1. **Sub-claim A (variational construction):** The mixture-of-sources generative model `P(k,z) = P(k|z=j)P(z=j) = (Ω_ij q_j)(k) · π_j` is a well-defined Bayesian generative model whose `KL[Q‖P]` with `Q(k,z) = q_i(k)·β(z)` mean-field factorization yields the form at Eq.~\eqref{eq:mixture_free_energy} (line 721).
2. **Sub-claim B (energy-entropy decomposition):** The decomposition `F_align = ⟨E⟩_β - H(β) + const` (line 734) is exact, with the "const" term properly identified (uniform-π case) or absorbed into a shifted energy (non-uniform-π case).
3. **Sub-claim C (KKT stationarity):** The Lagrangian-stationarity equation `∂L/∂β_ik = E_ik + log β_ik + 1 - log π_k - λ = 0` (line 747) yields the softmax form `β_ik = π_k exp(-E_ik)/Σ_m π_m exp(-E_im)` (line 753) via algebraic solving and the simplex normalization constraint.
4. **Sub-claim D (uniform-π specialization):** Under `π_k = 1/N`, the boxed result `β_ik = softmax_k(-E_ik)` (line 760) follows by direct cancellation of the prior factors.
5. **Sub-claim E (temperature factor τ):** The temperature-rescaled canonical form `F_align^(τ) = Σ_j [β E + τ β log(β/π)]` (line 766) has the same stationary point structure `β* = π exp(-E/τ)/Z`, with the substituted value `-τ log Z` matching the reduced free energy used downstream.

The judge should be alert to "exact stationary point" being weaker than "unique global minimum" (the Lagrangian is convex on the open simplex, so stationary ≡ minimum, but the manuscript does not explicitly verify second-order conditions). The judge should also weigh whether the mixture-of-sources construction at line 697 ("the component distributions P(k|z=j) = Ω_ij q_j depend on the variational posteriors q_j of other agents, making the generative model itself a function of variational quantities") is a clean fixed-point coupling or a circularity in the derivation.

## User context

Second debate in the `/red-blue-debate` series on the `GL(K)_attention.tex` manuscript. The first debate (`docs/debates/2026-05-19-reduction-to-standard-transformer/`) returned RED_WINS on the §5 reduction; the manuscript edit was applied. The user requested this follow-up on the §4 derivation that produces the attention rule itself, which the §5 reduction takes as input.
