# Action — supplementary-covariance-dynamics

**From verdict:** RED_WINS-narrow

## Summary of verdict

The compound claim that §Covariance Dynamics and Equilibrium Analysis of `Attention/GL(K)_supplementary.tex` (lines 180–387) is "complete and mathematically/theoretically pure" fails on the self-containment prong. The "mathematically/theoretically pure" prong survives decisively — all six sub-claims (α Gaussian KL, β Σ-derivative, γ sandwich-product transport, δ coefficient -2 assembly, ε fixed-point and regime analysis, ζ Hessian PD) are verified against external canon ([Bishop 2006 §2.3.6, Murphy 2012 §2.3.2, Cover-Thomas 2006 §8.6, Petersen-Pedersen §9.1/§9.4, Smith 2005 / Pennec 2006]). Red conceded all six sub-claims.

The "complete and self-contained" prong fails on five-to-six concrete editorial breaks that exceed the three-gap (Debate 3 §3 Gauge-Covariant VFE) and four-gap (Debate 12 §A General Mathematical Framework) editorial-threshold precedent established by the prior audit series. The breaks are: undefined `α_i` at first use in §B (line 275); section-number drift at line 337 (currently "3.6", should resolve to "3.7"); undefined `Λ_o` at line 385; undefined `⊠_{sym}` notation at line 380; zero internal citations across 207 lines of §B covering four canonical correspondences; and §B.2.2 regime gated by `α_i ≪ 1` against the framework default `α_i = 1` per main paper line 948.

## Recommended action

Six scoped editorial corrections to `Attention/GL(K)_supplementary.tex`. No equation changes, no derivation revisions, no structural rewrites.

### Edit 1 — define `α_i` at first use in §B (around line 275)

The current text at line 275 reads: "The coefficient of `Σ_i^{-1}` is `-(1 + Σ_j β_{ij}) = -2`, since the entropy of `q_i` contributes `-(1/2) Σ_i^{-1}` to both the prior KL (once, with unit coefficient `α_i = 1`) and the attention-weighted alignment KL (once, weighted by `Σ_j β_{ij} = 1`)."

Add either an inline cross-reference: "...with unit coefficient `α_i = 1` (see main paper Section~\ref{sec:state_dependent_precision} for the state-dependent generalization)" — or a footnote at first use defining `α_i` as the self-coupling weight on the prior KL.

### Edit 2 — replace plain-text section reference at line 337

The current text reads: "...one requires the state-dependent prior coupling `α_i ≪ 1` (cf.\ main text, Section~3.6)..."

Replace with: "(cf. main text, Section~\ref{sec:state_dependent_precision})" so the cross-reference resolves to the correct section regardless of future section renumbering.

### Edit 3 — define `Λ_o` at line 385

The current text reads: "Hence the covariance alignment fixed-point is an attractor of the variational dynamics under the standing assumptions `Σ_j β_{ij} = 1`, `Ω_{ij}` invertible, and `Λ_o` negligible relative to the inter-agent coupling."

Either define inline: "...and `Λ_o := R^{-1}` (the observation precision, cf. line~253) negligible..."; or directly substitute `R^{-1}` for `Λ_o` to match the line-253 notation.

### Edit 4 — ground the `⊠_{sym}` notation at line 380

The current Hessian display uses `Σ_1^{-1} ⊠_{sym} Σ_1^{-1}` without defining `⊠_{sym}` or citing the SPD-manifold canon.

Either write the explicit Hessian action:

```latex
\frac{\partial^2 D_{\mathrm{KL}}}{\partial \Sigma_1 \partial \Sigma_1}[H, H]
= \frac{1}{2} \mathrm{tr}(\Sigma_1^{-1} H \Sigma_1^{-1} H)
```

(which is manifestly positive definite as `(1/2) ‖Σ_1^{-1/2} H Σ_1^{-1/2}‖_F²`), or append a citation `\citep{smith2005covariance,pennec2006riemannian}` to ground the `⊠_{sym}` notation in the SPD-manifold canon.

### Edit 5 — inline citations to canonical correspondences

Three citation additions:

a. **Gaussian KL closed form (line 221)**: append `\citep{bishop2006pattern,murphy2012machine,coverthomas2006elements}` to the displayed equation. Verify `coverthomas2006elements` exists in `Attention/references.bib`; if not, use the existing closest canonical citation (Bishop / Murphy already verified at `references.bib:2516, 2523` from the §3.7 Gamma-prior edit).

b. **Matrix-calculus derivative (line 232)**: append `\citep{petersen2012matrix}` to the boxed gradient expression.

c. **Linear pushforward (line 234)**: append `\citep{bishop2006pattern}` (§2.3.3 covers linear Gaussian transformations).

### Edit 6 — flag the alignment-dominated regime as theoretical-only

At §B.2.2 (line 337–352), after the boxed Eq. eq:beta_weighted_precision, append a single sentence:

> "This regime is studied as a theoretical limit; the canonical configuration uses `α_i = 1` per main paper Section~\ref{sec:state_dependent_precision}, in which case the prior term remains present at coefficient `-1/2`."

This calibrates reader expectations: the §B.2.2 analysis is a limit, not a default operating point.

## Bib additions

Verify in `Attention/references.bib`:
- `bishop2006pattern` — EXISTS at line 2516 (from §3.7 Gamma-prior fix).
- `murphy2012machine` — EXISTS at line 2523 (from §3.7 Gamma-prior fix).
- `petersen2012matrix` — verify; if absent, add the Matrix Cookbook entry.
- `coverthomas2006elements` — verify; if absent, add or use Bishop/Murphy alone.
- `smith2005covariance`, `pennec2006riemannian` — verify; if absent, add or write out the explicit Hessian instead (Edit 4 alternative).

## Cumulative debate-series state

Thirteenth debate in the gauge-transformer audit series. Closed queue:

1. §5 transformer reduction (RED_WINS).
2. Softmax-β stationarity (RED_WINS).
3. Sub-claim A flat bundle (BLUE_WINS — §5 reduction sub-claim).
4. Sub-claim B degenerate Σ (BLUE_WINS).
5. Sub-claim C QK^T identification (BLUE_WINS).
6. Sub-claim D V identification (BLUE_WINS).
7. Canonical F vs surrogate (RED_WINS).
8. Multi-head block-diagonal (BLUE_WINS).
9. Route 1 untied carving (RED_WINS).
10. FFN softmax-gradient correction (RED_WINS).
11. §3 Gauge-Covariant VFE (RED_WINS narrow).
12. Supplementary §A General Mathematical Framework (RED_WINS narrow).
13. **Supplementary §B Covariance Dynamics (RED_WINS narrow, this debate).**

Optional follow-ups: supplementary §C Gauge Frame Gradients and §D Variational Gradient Descent (next in the chapter sweep); Participatory_it_from_bit.tex §Theory as the companion paper's foundational chapter.

## Follow-up debates

None required from this verdict. The math is canonical; the edits are scoped. Optional continuation of the supplementary chapter sweep: §C (lines 388–610), §D (lines 611–665).
