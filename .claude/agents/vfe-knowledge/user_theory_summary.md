# User Claims — Summary of the Gauge-Theoretic VFE Transformer

**This file is a summary of the user's claims and constructions.** It is *not* the source of truth for the agents — the source-of-truth files are `external_canon_*.md`, which cite standard literature in information geometry, differential geometry, gauge theory, FEP / active inference, variational inference, and transformer attention.

The agents use this file to know *what the user is claiming*. They then evaluate those claims against the external canon. Findings cite standard sources (e.g., `[Nakahara2003 §10.3]`, `[Friston2010]`, `[Vaswani2017]`), not this file.

Equations below are paraphrased from CLAUDE.md and the manuscripts in `Attention/` as of 2026-05-18. The manuscripts evolve; read the current `.tex` for authoritative claims.

## Variable conventions

| Symbol | Meaning |
|---|---|
| K | Belief-space dimension (per-head) |
| N | Sequence length |
| μ, Σ | Belief mean and covariance (Gaussian) |
| φ | Gauge-frame parameter (Lie-algebra element); `φ_i ∈ gl(K)` |
| Ω_ij | Transport from site j to site i: `Ω_ij = exp(φ_i) · exp(-φ_j)` |
| κ | Learnable inverse-temperature scalar |
| τ | Effective softmax temperature: `τ = κ √K` |
| β_ij | Attention weight from j to i |
| γ_ij | Meta-attention weight (model-to-model coupling) |
| α | Self-coupling weight (belief-to-prior) |
| λ_h | Hyper-prior weight |
| q, p, s, h | Beliefs / priors / models / hyper-prior (VFE hierarchy: h → s → p → q → observations) |

## The free energy functional (canonical form)

From CLAUDE.md, `\label{eq:free_energy_functional_final}`:

```
F = α · KL(q_i ‖ p_i)                                                    # self-coupling
  + λ_h · KL(s_i ‖ h)                                                    # hyper-prior
  + Σ_ij [ β_ij  · KL(q_i ‖ Ω_ij q_j) + τ · β_ij  · log(β_ij  / π_ij) ]   # belief coupling + attention entropy
  + Σ_ij [ γ_ij · KL(s_i ‖ Ω_ij s_j) + τ · γ_ij · log(γ_ij / π^(s)_ij) ]  # model coupling + meta entropy
  − E_q[ log p(o | x) ]                                                  # observation likelihood
```

with `π_ij = 1/N` (uniform attention prior).

### Why the entropy term is non-negotiable

Without `τ · β · log(β / π)`, the row-Lagrangian for β yields a delta function, not softmax. Manuscript line 1261 explicitly distinguishes the canonical F above from the **entropy-suppressed surrogate** `Σ β · KL` (no entropy term). Their gradients differ by `−τ⁻¹ Cov_β(KL, ∇KL)`. If code computes `Σ β KL` but cites the canonical F, that is a major theoretical drift.

**Audit check:** wherever F is assembled, look for the `β · log(β / π)` term. Common location: `transformer/core/vfe_utils.py`, `transformer/vfe/block.py`, `transformer/vfe/e_step.py`.

## Attention

```
β_ij = softmax_j ( −KL(q_i ‖ Ω_ij q_j) / (κ √K) )
```

- `κ` is a learnable hyperparameter.
- `√K` is intentional dimension scaling (analogous to scaled dot-product attention's `√d_k`).
- The negative sign matters: low KL ⇒ high attention.

**Audit check:** verify the sign, the `κ √K` denominator (not just `κ`, not just `√K`), and that the softmax is over the correct axis (sites j for site i).

## Transport

```
Ω_ij = exp(φ_i) · exp(−φ_j)
```

This factorization covers GL⁺(K) via a product of two matrix exponentials. It is *not* simply `exp(φ_i − φ_j)` — the matrix exponentials don't commute unless `φ_i` and `φ_j` commute. If code computes `exp(φ_i − φ_j)`, that is a major bug.

## Covariance transport — the sandwich

```
Σ_transported = Ω · Σ · Ωᵀ
```

This is the single most common correctness bug. Anywhere covariance is transported, the conjugation must be `Ω … Ωᵀ`. Variants to flag:

- `Ω · Σ` (missing right multiplication) — **Critical**
- `Σ · Ωᵀ` (missing left multiplication) — **Critical**
- `Ω · Σ · Ω` (missing transpose) — **Critical**
- Diagonal approximation `σ_t = |Ω| σ` or similar — **acceptable** when `diagonal_covariance=True` is the active mode.

**Audit check:** grep for `Sigma` near `Omega` and verify the sandwich. Files to check first: `transformer/core/transport_ops.py`, `transformer/vfe/block.py`, `transformer/vfe/e_step.py`, `transformer/pure_vfe/gaussians.py`.

## Timescales

- **Fast E-step**: belief `q` inference per forward pass. Iterative, gradient-based or fixed-point.
- **Slow M-step**: prior/model `s, p` parameter learning via backprop.
- **Static hyper-prior `h`**: frozen at init, never learned.

`σ_p` is an M-step parameter. The E-step *reads* it but must not write gradients to it (detached in VFE iterations). `sigma_ce_scale` controls the residual CE→σ_p gradient in decode (`0.0` = fully detached).

## E-step learning rates (post 2026-05-13)

- `E_mu_q_lr` — μ retraction step size.
- `E_sigma_q_lr` — σ retraction step size.
- Both modulated by the same cosine decay factor across iterations, but **independent in magnitude**.
- `E_sigma_q_trust` (default 5.0) — separate trust-region clamp on whitened tangent `δσ/σ`.

σ retraction: `σ_new = σ · exp(E_sigma_q_lr · decay_t · clamp(δσ/σ, ±E_sigma_q_trust))`.

**Pre-2026-05-13** code used `sigma_lr` only as the trust-region clamp; sweeping `E_sigma_q_lr` produced no visible effect when the clamp wasn't binding. If an audit finds code using the pre-2026-05-13 convention, that is a regression.

## KL divergence between Gaussians

For diagonal Σ:
```
KL(q ‖ p) = 0.5 · Σ_k [ log(σ_p²/σ_q²) + (σ_q² + (μ_q − μ_p)²) / σ_p² − 1 ]
```

For full Σ:
```
KL(q ‖ p) = 0.5 · [ tr(Σ_p⁻¹ Σ_q) + (μ_p − μ_q)ᵀ Σ_p⁻¹ (μ_p − μ_q) − K + log(|Σ_p|/|Σ_q|) ]
```

When the prior is the transport of another belief `p ← Ω q_j`, then `μ_p = Ω μ_j` and `Σ_p = Ω Σ_j Ωᵀ` (the sandwich).

## Stationarity of softmax β under F

Setting `∂F/∂β_ij = 0` with row constraint `Σ_j β_ij = 1`:

```
KL(q_i ‖ Ω_ij q_j) + τ (log β_ij − log π_ij) + τ + λ_i = 0
⇒ β_ij ∝ π_ij · exp( −KL(q_i ‖ Ω_ij q_j) / τ )
⇒ β_ij = softmax_j( −KL / τ )    [with uniform π]
```

This derivation requires the `τ β log(β/π)` entropy term. Without it, ∂F/∂β = KL_ij + λ_i, which has no β-dependence and the row Lagrangian collapses to picking the smallest KL deterministically. **The surrogate `Σ β KL` does not have softmax as a stationary point.** Any manuscript that derives softmax from F without the entropy term is wrong.

## Symbolic verification recipes

Where the auditor can verify with `sympy`:

1. **Softmax stationarity:** symbolically set ∂/∂β of `Σ_j (β KL + τ β log(β/π))` to zero, solve. Confirm softmax form.
2. **Sandwich invariance:** for a generated symmetric Σ and arbitrary Ω, confirm `Ω Σ Ωᵀ` is symmetric and PSD when Σ is.
3. **Ω factorization:** confirm `exp(φ_i) exp(−φ_j) ≠ exp(φ_i − φ_j)` in general by picking non-commuting 2×2 matrices.
4. **KL gradient:** symbolically differentiate KL(q‖p) w.r.t. μ_q, σ_q and confirm code matches.

See `audit_methodology.md` for when to actually invoke `sympy`.
