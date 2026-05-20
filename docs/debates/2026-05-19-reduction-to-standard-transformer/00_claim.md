# Claim — reduction-to-standard-transformer

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (manuscripts `Attention/GL(K)_attention.tex` §5, `Attention/GL(K)_supplementary.tex` §B; external canon for standard transformer attention)
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge

## Claim

Standard scaled dot-product transformer attention, as defined in Vaswani et al. (2017) §3.2.1, is recovered **exactly** (not approximately, and with every intermediate step mathematically valid) as the flat-bundle, degenerate-covariance, infinite-precision-prior limit of the GL(K) gauge-covariant variational free energy attention derived in `Attention/GL(K)_attention.tex` §5 ("Reduction to Transformer Attention", lines 990–1956), with the load-bearing identification `Q K^T / sqrt(K) ↔ - KL(q_i || Ω_ij q_j) / (κ sqrt(K))` and `V ↔ μ_q` made in §5.2 (line 1020 onwards).

## User context

The user requested a `/red-blue-debate` on "the attention and transformer derivations in GL(K)_attention.tex and the corresponding supplementary manuscripts." From four candidate load-bearing propositions surfaced from the manuscript section structure (softmax-β stationarity, transformer reduction, canonical-F vs surrogate, multi-head block-diagonal), the user selected the transformer reduction. The other three remain as candidate follow-up debates.

## Sub-claims (compound — flagged for possible separate debates)

The headline claim above is compound. The load-bearing sub-claims that the debate may surface:

1. **Sub-claim A (flat bundle):** Setting `Ω_ij = I` (the flat-connection / trivial-holonomy limit) is a well-defined limit that does not destroy gauge-equivariance arguments used upstream.
2. **Sub-claim B (degenerate Σ):** The covariance `Σ_i → 0` (or → some fixed Σ_∞) limit converts the Mahalanobis-form `KL(q_i || Ω_ij q_j)` into a quadratic `||μ_i - μ_j||^2 / σ^2` that is then identified with `Q K^T`.
3. **Sub-claim C (Q K^T identification):** The identification `Q ↔ μ_q` and `K ↔ μ_q` (same vector under "self-key" gauge) recovers the standard inner-product score up to a sign and the `1/sqrt(K)` scaling.
4. **Sub-claim D (V identification):** The aggregated value `V ↔ μ_q` (or `Ω μ_q`) reproduces the standard `softmax(QK^T/sqrt(K)) V` output.
5. **Sub-claim E (validity of intermediate steps):** Every algebraic and limit step in §5.2 (line 1020–1342) holds without missing assumptions or covert linearization.

The judge should be alert to "the reduction is exact" being weaker than "the reduction holds up to a controlled, named approximation." If a step is a documented approximation rather than an exact identity, that affects the verdict.
