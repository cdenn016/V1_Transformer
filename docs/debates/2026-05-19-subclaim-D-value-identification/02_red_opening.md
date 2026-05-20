# Red Opening — subclaim-D-value-identification

## Steelman (opposing position)

Under the constant-gauge specialization `Ω_{ij} = Ω` for all `(i,j)`, the sum `Σ_j β_{ij} Ω μ_j` factors by linearity to `Ω · Σ_j β_{ij} μ_j`, and the matrix `Ω ∈ GL(d_k)` can be renamed `W_V^T` to produce the standard form `μ̂_i = Σ_j β_{ij} V_j` with `V_j = W_V^T μ_j`; the algebraic step is trivially correct.

## Position

The reduction is **not exact to the standard transformer's value aggregation**, because (i) the standard `W_V` is rectangular `d_model × d_v` while the manuscript's `W_V` is square `d_k × d_k`, so the identification recovers at most a head-space sub-factor of the standard projection rather than the standard projection itself; (ii) the constant-gauge step is a definitional absorption rather than a derivation — the manuscript's own line 1867 admits the framework *generically* allows distinct attention and value gauges, so collapsing them to a single `Ω` is a pre-imposed coupling not present in standard transformers; and (iii) sub-claim D inherits the rectangular-projection failure mode of sub-claim C, since the same `d_k × d_k` notation that breaks the Q–K reduction breaks the V identification by the same rank/shape mismatch.

## Evidence

- **Manuscript `Attention/GL(K)_attention.tex:1310`**: `V_j ≡ W_V^T μ_j, W_V ∈ ℝ^{d_k × d_k}`. The projection is **square**, not rectangular.

- **External canon `external_canon_transformers.md` §1** (verbatim, citing [Vaswani2017 §3.2.1]): `V ∈ ℝ^{N × d_v}` via learned `W_V ∈ ℝ^{d_model × d_v}`, and in MHA each head uses `W_V^i ∈ ℝ^{d_model × d_k}` with `d_k = d_model / h`. The standard `W_V` is **rectangular**: it maps the full model dimension `d_model` down to the per-head value dimension `d_v`. A square `d_k × d_k` matrix cannot perform this projection — it lives entirely in the head subspace and presupposes that the lift from `d_model` to per-head `d_k` has already happened off-stage. The manuscript does not exhibit that lift.

- **Manuscript `Attention/GL(K)_attention.tex:1313`**: "we **absorb** the gauge transport Ω into the learned matrix W_V." This is a renaming (`W_V^T := Ω`), not a derivation. The standard transformer's `W_V` is trained by SGD on a downstream loss against an *independent* `W_Q, W_K`; the gauge framework's `Ω = exp(φ_i) exp(−φ_j)` is determined by `φ` updates inside the E-step (CLAUDE.md "Transport" definition). Calling these the same matrix is a notational identification, not an equality of objects.

- **Manuscript `Attention/GL(K)_attention.tex:1867`** (in the RoPE subsection): "the asymmetry in RoPE with position-dependent attention but position-independent values corresponds to factoring the gauge transport into an attention gauge and a value gauge that need not coincide. This decomposition is supported (but not required) by the full framework." The manuscript itself admits the framework *natively* supports distinct attention and value gauges. The constant-gauge specialization at §5.2.3 collapses both to the *same* `Ω`, which is one specific point in the framework's design space, not the only one.

- **External canon §1 — KQV interpretation**: "each head uses its own `W_Q^i, W_K^i, W_V^i`." Standard transformers have **three independently learned projections** per head. The gauge framework collapses Q, K, and V to **one shared transport `Ω`** (the attention logit `μ_i^T Ω^{-T} μ_j` and the value aggregation `Σ β Ω μ_j` are built from the same `Ω`). This is a structural difference: three parameters per head versus one. The reduction loses this independence; the claim that the reduction is "exact" is therefore exact only to a *constrained* sub-family of standard attention (the family in which `W_Q W_K^T` and `W_V` are tied to the same underlying matrix), not to the standard transformer's full parameterization.

- **Manuscript `Attention/GL(K)_attention.tex:1325`** (Complete Attention Formula): the constant-gauge reduction simultaneously imposes `W_Q W_K^T = σ⁻²Ω⁻ᵀ` on the attention side **and** absorbs the same `Ω` into `W_V` on the value side. The standard transformer's three projections are not constrained to share a common `Ω`. The gauge framework's reduction is a measure-zero slice of standard parameter space.

- **Cross-link to sub-claim C** (evidence pack §1, Red reading): the rectangular issue in sub-claim C (the manuscript's `d_k × d_k` versus Vaswani's `d_model × d_v`) is the same shape mismatch here. If sub-claim C resolves Red on the rectangular argument, sub-claim D fails by the same evidence.

## Falsification conditions

This position is wrong if any of the following can be cited:

1. The manuscript explicitly disambiguates `d_k` in line 1310 as `d_model` (so the square `d_k × d_k` notation actually denotes `d_model × d_model`, which would then admit `W_V ∈ ℝ^{d_model × d_v}` as a sub-block). Searching `Attention/GL(K)_attention.tex` for the definition of `d_k` near §5.2.3 shows no such disambiguation; the symbol is used throughout §5 as the per-head dimension.

2. The manuscript supplies an explicit thin-SVD-style lift from per-head `d_k` to `d_model` for the value path, paralleling whatever lift sub-claim C may invoke for Q–K. The §5.2.3 text contains no such lift — the boxed reduction at line 1318 jumps directly from `Σ_j β_{ij} Ω μ_j` to `Σ_j β_{ij} V_j` via the renaming at 1313, with no dimension-changing step.

3. Blue produces a citation to a standard-transformer reference in which `W_V` is constrained to be `d_k × d_k` (square, full-head-dimension) and is tied to the attention projections `W_Q, W_K`. Canon §1 contradicts this: `W_V` is independently learned and rectangular.

4. Blue argues that "exact" means "exact up to a definitional rename of the matrix that appears" rather than "exact to the standard transformer's parameterization." If that is the intended reading, the claim is trivially true but trivially weak — every linear map can be renamed to look like another linear map, and the reduction has no content beyond the linearity of `Σ`. The judge should weigh whether this is the "exact reduction" the manuscript is claiming.
