# Action — subclaim-A-flat-bundle

**From verdict:** BLUE_WINS

## Recommended action

**Accept sub-claim A as written.** The flat-bundle limit is well-defined and the upstream GL(K)-invariance theorem (`Attention/GL(K)_attention.tex:520-528`) holds for every `Ω ∈ GL(K)` including the trivial element. The verdict confirms what the first debate's orchestrator marked "defensible."

**Separate audit item:** the §5.7 summary at `Attention/GL(K)_attention.tex:1974-1978` (and the "gauge-equivariant" descriptor at `:1962`) conflates two distinct invariances:

1. The theorem's single-Ω simultaneous push-forward invariance under `(μ_i, μ_j) → (Ω μ_i, Ω μ_j)` with both arguments transformed by the same Ω. This is the textbook f-divergence covariance property.
2. The summary's invocation of `W_Q, W_K` "as valid gauge transformations" acting independently per agent — a per-agent local equivariance not delivered by the single-Ω theorem.

Both sides conceded this conflation is real. The manuscript should either (a) restate the §5.7 summary to drop the per-agent equivariance claim and keep only the codomain statement, or (b) prove the per-agent local equivariance separately if it is intended as a substantive claim.

**Recommended edit at `Attention/GL(K)_attention.tex:1962`:**
- "natural-gradient descent on the **gauge-equivariant** free energy" → "natural-gradient descent on the free energy" (or qualify "gauge-equivariant" with explicit reference to the universally-quantified theorem at `\ref{thm:glk_invariance}`, distinguishing the framework's equivariance from any equivariance of the gauge-fixed standard-transformer limit).

**Recommended edit at `Attention/GL(K)_attention.tex:1974-1978`:**
- Replace "the learned projections W_Q, W_K are valid gauge transformations" with a more precise statement: the W matrices are elements of GL(d_k) under the framework's identification, and the theorem at `\ref{thm:glk_invariance}` applies to the framework as a whole (not to the standard-transformer limit's W matrices acting independently).

## Follow-up debates (if any)

None for sub-claim A itself. The verdict raises the §5.7 conflation as a separate audit item that does not require a debate (both sides agree on the fact); it requires a manuscript edit.
