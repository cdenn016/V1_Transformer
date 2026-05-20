# P2 Investigation — full-covariance decode projection

## Summary

The diagonal projection at `transformer/vfe/prior_bank.py:494` (decode) discards information **only** when `gauge_fixed_priors=True` AND `diagonal_covariance=False`. In every other supported configuration the projection is structurally a no-op or trivially exact. An opt-in exact-per-block decode is implementable at ~20× the diagonal cost; recommendation is to ship it gated by a new `exact_full_cov_decode` flag, default `False`, so the pure path exists per the CLAUDE.md "pure path under appropriate toggles" doctrine.

## What sigma_p and sigma_q actually contain at decode

The covariance tensor passed to `decode(mu_q, sigma_q, tau)` is one of two shapes:

1. `(B, N, K)` — strictly diagonal. Stored as a per-coordinate variance vector. Used when `diagonal_covariance=True` (the active config, `train_vfe.py:79`).
2. `(B, N, K, K)` — block-diagonal SPD matrix, with off-block elements *exactly zero* by construction (`prior_bank.py:339-348`, `_apply_gauge_transform` builds `Σ_p` block-by-block; `_diag_to_full_block` at `prior_bank.py:254-266` embeds a diagonal into a block-diagonal frame in direct mode).

Within each block `h` of size `d_h × d_h`, the content depends on mode:

| Mode | Within-block content |
|---|---|
| `gauge_fixed_priors=True`, `diagonal_covariance=True` | scalar `Σ_j A_{ij}^2 s_j` per coordinate (diagonal-of-sandwich) — stored as `(K,)`. |
| `gauge_fixed_priors=True`, `diagonal_covariance=False` | **dense `A_h · diag(s_h) · A_hᵀ`** — has nonzero off-diagonal mass within the block. |
| `gauge_fixed_priors=False`, `diagonal_covariance=True` | per-token `σ_v_k` lookup — stored as `(K,)`. |
| `gauge_fixed_priors=False`, `diagonal_covariance=False` | diagonal of the per-token σ embedded into a block-diagonal `(K, K)` tensor — all off-diagonal entries zero. |

The E-step preserves the same shape and the block-diagonal structure (gauge transport `Ω` is block-diagonal, so the sandwich `Ω Σ Ωᵀ` cannot generate cross-block coupling). Within each block, the E-step propagates the *dense* (`d_h × d_h`) sigma when `diagonal_covariance=False`.

## What the diagonal projection actually discards

`prior_bank.py:494`:
```python
sigma_q_diag = torch.diagonal(sigma_q, dim1=-2, dim2=-1) if is_full_cov else sigma_q
sigma_p_diag = torch.diagonal(sigma_p, dim1=-2, dim2=-1) if sigma_p.dim() >= 3 and sigma_p.shape[-1] == sigma_p.shape[-2] else sigma_p
```

This extracts the diagonal of each `(K, K)` tensor. The off-block elements are zero anyway (block-diagonal), so the projection only loses **within-block off-diagonal mass**.

That mass is zero in three of the four mode combinations above. It is nonzero only for `(gauge_fixed_priors=True, diagonal_covariance=False)` — the dense-sandwich-within-block case.

In that case, the exact within-block KL between two `d_h × d_h` dense Gaussians is:
```
2·KL_h(q || p_v)  =  tr(Σ_p_v_h⁻¹ Σ_q_h)
                  +  (μ_q_h - μ_p_v_h)ᵀ Σ_p_v_h⁻¹ (μ_q_h - μ_p_v_h)
                  -  d_h  +  log|Σ_p_v_h| / |Σ_q_h|
```
and the total decode KL is `Σ_h KL_h` (block-diagonal log-det and trace decompose). The diagonal projection replaces this with `KL_diag` computed coordinate-wise, which equals the exact KL only when both `Σ_q_h` and `Σ_p_v_h` are themselves diagonal — which they are not in this case.

## Approximation magnitude

For two block Gaussians with within-block correlation coefficient `ρ`, the diagonal-vs-full KL gap scales roughly as `-0.5 · d_h · log(1 - ρ²)` in the unit-variance, equal-mean reference. At `ρ = 0.5`, this is `~0.14 · d_h` per block; at `ρ = 0.9`, `~0.83 · d_h`. With `d_h = 20`, this is `~3` to `~17` nats per block — non-trivial relative to typical CE values for moderate vocabularies.

Initially after construction `Σ_q = A diag(σ_0) Aᵀ` with `A = exp(Σ_a φ^a G_a)`. With `phi_scale = 0.001` (active config), `φ` is small and `A ≈ I`, so within-block off-diagonal mass starts near zero. As `φ` is learned upward the off-diagonals grow. The approximation is therefore tight at initialization and degrades as training learns nontrivial gauge frames.

## Cost of an exact-full-cov decode

Per position `(b, n)` and per token `v`, an exact block-decomposed KL needs:
- Per block: invert `Σ_p_v_h` (`O(d_h³)`), compute log-det (`O(d_h³)`), trace `tr(Σ_p_v_h⁻¹ Σ_q_h)` (`O(d_h²)`), quadratic form (`O(d_h²)`).
- Σ_p_v_h⁻¹ and log|Σ_p_v_h| can be cached across positions (they depend only on `v`): one Cholesky per (block, vocab token) per forward, total `V · Σ_h d_h³`.
- Per-position cost is then `B · N · V · Σ_h d_h²`.

For the user's active scale (`K = 100`, 5 blocks of `d_h = 20`, `V = 10000`, `B = 16`, `N = 128`):
- Cache build: `V · Σ_h d_h³ = 10⁴ · 5 · 8000 ≈ 4·10⁸` FLOPs once per forward.
- Per-position: `B · N · V · Σ_h d_h² = 16 · 128 · 10⁴ · 2000 ≈ 4·10¹⁰` FLOPs per forward.
- Compared to the current diagonal decode: `B · N · V · K = 16 · 128 · 10⁴ · 100 ≈ 2·10⁹`.

So an exact block decode is **~20× more expensive** than the diagonal projection at these sizes. On an RTX 5090 (32 GB) this is tractable for K ≤ 100, V ≤ 50000; for larger configurations the diagonal projection remains the practical default.

## Recommendation

Add an opt-in `exact_full_cov_decode` flag to `VFEConfig`, default `False`, gated to the path `(gauge_fixed_priors=True, diagonal_covariance=False)`. When the flag is on, `decode` decomposes by block and computes the per-block full-cov KL exactly; when off, the diagonal projection is retained (current behavior).

This satisfies the CLAUDE.md "pure path under appropriate toggles" doctrine: a user wanting strict Law 3 sets `exact_full_cov_decode=True`; the default keeps the O(V·K) decode for compute-budget reasons.

Documentation strike (separate): qualify the Law-3 statement in `transformer/vfe/config.py:338-348` (already partially done in today's edit pass) and `transformer/vfe/prior_bank.py` decode docstring to make explicit which configurations satisfy Law 3 strictly vs by approximation.

## Out of scope for this investigation

- Whether the diagonal projection is the *right* approximation. There are alternatives (e.g., a tied-rank approximation of the block off-diagonals) but they introduce new approximation classes and are not standard.
- Whether the gauge-fixed-priors=False mode should be promoted to first-class (it is currently the user's active mode, but the diagonal-cov-within-block-tensor structure makes Law 3 trivially satisfied — the decode reads the same diagonal that encode writes).
- Whether the cross-coupling super-block path (`cross_couplings` non-empty) changes the cost analysis. It does — super-blocks can be much larger than per-head `d_h`, raising `d_h³` cost dramatically. An exact-full-cov-decode flag should be advertised as cost-quadratic in the largest super-block dimension.
