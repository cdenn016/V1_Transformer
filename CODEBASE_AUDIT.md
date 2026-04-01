# Codebase Audit: V10.0 Gauge Transformer

Comprehensive review across five dimensions: bugs, performance, mathematical correctness, training pipeline, and architecture. Compiled from parallel deep-review agents on 2026-03-31.

---

## Critical Findings

### BUG: Pullback Gram Matrix Uses Wrong Inner Product
**Files:** `gauge_preconditioner.py:384-385`, `variational_ffn.py:2401-2402`
**Severity:** HIGH (affects `phi_natural_gradient='pullback'` mode only)
**Status:** FIXED

The pullback natural gradient computed `tr(T_a T_b)` instead of `tr(T_a^T T_b)` (Frobenius inner product). For GL(K) with mixed symmetric/antisymmetric generators, `tr(T_a T_b)` is indefinite on the antisymmetric block, producing an incorrect metric and wrong natural gradient directions. The Killing form and Cartan modes were unaffected (they used the correct Frobenius product). Fix applied: removed the erroneous `.transpose(-2, -1)`.

### BUG: Closed-Form E-Step Uses Untransported Covariance
**File:** `variational_ffn.py:~3598-3609`
**Severity:** HIGH (affects `closed_form_e_step=True` mode only)

The closed-form alignment precision computes `sum_j beta_ij / sigma_j` using the *untransported* covariance of position j. The gradient descent path correctly uses `sigma_j_transported = diag(Omega @ diag(sigma_j) @ Omega^T)`. For non-identity transport, the closed-form solution is inconsistent with the gradient it claims to solve. Current config uses `closed_form_e_step=False`, so this does not affect production runs.

### ARCHITECTURE: VFE FFN Returns Full State, Not Delta
**File:** `blocks.py:347-350, 417-419`
**Severity:** MEDIUM-HIGH (design issue, present in all runs including PPL=71)

The VFE FFN receives `norm2(mu_q)`, evolves it to `norm2(mu_q) + delta`, and returns the full evolved state. The block then adds this as a residual: `mu_q = mu_q + (norm2(mu_q) + delta)`. The VFE correction `delta` is buried under the `norm2(mu_q)` copy of the input. A standard transformer's MLP returns an arbitrary learned function, not `input + small_correction`. This means the VFE contribution to the residual stream is proportionally small.

**Possible fix:** Have the FFN return `mu_current - mu` (the delta only). The residual then becomes `mu_q + delta`, amplifying the VFE correction. This is a one-line change but would require retuning learning rates.

### PERFORMANCE: Redundant matrix_exp Across E-Step Iterations
**File:** `variational_ffn.py:3810, 3981`
**Severity:** CRITICAL (15-30% of forward pass time)

When `evolve_phi_e_step=False` (phi static within E-step), `fused_block_matrix_exp_pairs` is called every iteration despite producing identical results. With `n_iterations=3` and 5 heads, this wastes 2/3 of matrix_exp compute.

**Fix:** Hoist the computation out of the iteration loop. Only recompute when phi actually changes.

### PERFORMANCE: Blanket AMP Autocast Disable
**File:** `variational_ffn.py:3363-3373`
**Severity:** CRITICAL (20-40% throughput loss)

The entire E-step method disables AMP autocast, forcing all operations to float32. Only sigma arithmetic (divisions, logs, inversions) needs float32. The mu transport (`Omega @ mu`), attention computation, and observation gradient could run in float16.

**Fix:** Use fine-grained `torch.amp.autocast('cuda', enabled=False)` only around sigma operations.

---

## Mathematical Correctness (Code vs Manuscript)

### Verified Correct
- **Mean gradient** `dF/dmu` (Eq. 21): Self-coupling, product-rule correction for adaptive alpha, direct alignment, softmax coupling all match.
- **Covariance gradient** `dF/dSigma` (Eq. 23): The `-(1+alpha)Sigma^{-1}` coefficient emerges correctly from summing self-coupling and alignment entropy contributions.
- **KL divergence** (Eq. 4-5): Formula correct for both diagonal and full covariance. Transport `Omega @ Sigma @ Omega^T` correctly applied.
- **Attention** `beta = softmax(-KL/(kappa*sqrt(K)))` (Eq. 7): Correct with intentional dimension normalization.
- **Transport** `Omega_ij = exp(phi_i)exp(-phi_j)` (Eq. 8): Correct.
- **Killing form metric** (Appendix C): Correct implementation of `2K tr(T_a^T T_b) - 2 tr(T_a)tr(T_b)`.
- **Cartan decomposition** (Appendix C): Correct projection onto symmetric/antisymmetric subspaces.
- **KL-decode fused matmul** (Prior Bank): Correct derivation; softmax-invariant terms properly dropped.

### Discrepancies (Non-Critical)
| # | Location | Manuscript | Code | Severity |
|---|----------|-----------|------|----------|
| 1 | VFE gradient weights | Unit coefficients | `lambda_belief`, `lambda_softmax` prefactors | Low (configurable) |
| 2 | Adaptive alpha | Scalar per agent | Per-dimension vector `(B, N, K)` | Low (generalization) |
| 3 | Fisher projection (full cov) | `sym(grad_Sigma)` before sandwich | No explicit symmetrization | Low (compensated downstream) |
| 4 | sigma_ce_scale | Not in manuscript | Engineering gradient scaling trick | Low (correct but undocumented) |

---

## Bug Inventory

### HIGH Severity
| Bug | File:Line | Description | Affects Current Config? |
|-----|-----------|-------------|------------------------|
| Pullback Gram | gauge_preconditioner.py:384 | Wrong inner product `tr(T_a T_b)` vs `tr(T_a^T T_b)` | No (`phi_natural_gradient='killing'`) — **FIXED** |
| Closed-form untransported sigma | variational_ffn.py:~3598 | Uses raw sigma_j instead of transported | No (`closed_form_e_step=False`) |

### MEDIUM Severity
| Bug | File:Line | Description | Impact |
|-----|-----------|-------------|--------|
| Obs sigma upward bias | variational_ffn.py:4081 | Stein's lemma observation gradient for sigma is always >= 0, pushing sigma up | Gradual sigma inflation when `obs_sigma_gradient=True` |
| Block-diagonal full-cov logdet | variational_ffn.py:604 | Crude uniform logdet-per-dim approximation in product-rule correction | Slightly wrong alpha gradient in full-cov mode |
| Newton-Schulz lower bound | gauge_utils.py:102 | Convergence basin check only validates upper bound, not near-zero singular values | Possible convergence to rank-deficient orthogonal factor |
| Post-softmax clamp+renorm | attention.py:707 | Clamping then renormalizing slightly modifies gradient structure | Negligible for typical epsilon=1e-8 |
| Implicit EM sigma docstring | variational_ffn.py:270 | Docstring says `sigma^{-4}` but code correctly uses `sigma^{-2}` | Misleading documentation only |

### LOW Severity
| Bug | File:Line | Description |
|-----|-----------|-------------|
| `_get_sigma_trust` type inconsistency | variational_ffn.py:2531 | Returns tensor when learnable_lr=True, float when False |
| Double norm clamping | attention.py:1209 | Phi norm clamped before stable_matrix_exp_pair which also clamps |
| Stale docstring dim_threshold | gauge_utils.py:48 | Says default=8, actual=20 |
| eigh fallback may fail | attention.py:1116 | Eigendecomposition fallback lacks its own try/except |
| RoPE odd-K dimension | attention.py:122 | Last dimension position-invariant when K is odd |

---

## Performance Bottlenecks

### Critical (>15% forward pass time)
| Finding | File | Impact | Fix |
|---------|------|--------|-----|
| Redundant matrix_exp in E-step loop | variational_ffn.py:3810 | 15-30% time | Hoist out of loop when phi static |
| Blanket AMP disable | variational_ffn.py:3363 | 20-40% throughput | Fine-grained autocast |
| Full-vocab matrix_exp in gauge_fixed decode | prior_bank.py:413 | 20-40% time | Only when `gauge_fixed_priors=True` (not current config) |

### High (5-15%)
| Finding | File | Impact | Fix |
|---------|------|--------|-----|
| Double Omega construction (non-fused) | variational_ffn.py:4014 | 10-30% in fallback | Ensure fused path fires; cache Omega |
| Full (B,N,N,K,K) Omega in non-block paths | attention.py:894 | Dominates memory | Always use block-diagonal |
| gauge_fixed decode 2.26 GB peak | prior_bank.py:287 | OOM risk | Chunk vocabulary |

### Medium (1-5%)
| Finding | File | Impact | Fix |
|---------|------|--------|-----|
| Obs gradient 50k softmax every E-step iter | variational_ffn.py:4055 | 5-10% time | Skip when `use_obs_in_vfe=False` (current) |
| O(B*N^2*K) tensors when lambda_softmax=0 | variational_ffn.py:948 | 300 MB wasted | Guard on lambda_softmax > 0 |
| Sequential per-head Python loop | attention.py:2670 | 5-15% overhead | Batch same-dim heads |
| Pullback O(n_gen^3) per token | gauge_preconditioner.py:339 | 5-15% for K>=15 | Use 'killing' mode (default) |

---

## Architecture Findings (PPL Plateau Analysis)

### Why PPL Plateaus at ~71 Regardless of K=80-120

**1. VFE FFN residual buries the correction** (see Critical Findings above). The E-step delta is added on top of `norm(mu_q)`, making the VFE contribution ~O(delta/||norm(mu_q)||) of the output. With 1 iteration and a trust region of 2.0, delta is small.

**2. LayerNorm before decode destroys magnitude information.** The final `ln_final(mu_q)` centers and scales mu to zero mean, unit variance before PriorBank decode. The KL-decode depends on actual magnitudes for discrimination. LayerNorm forces all positions to the same norm, removing per-position confidence signals.

**3. Rank-2K decode bottleneck.** The fused KL matmul `(B,N,2K) @ (2K,V)` has effective rank 2K. At K=90, rank=180. Increasing K to 120 gives rank=240 but the improvement diminishes — the marginal value of more linear dimensions drops off. The bottleneck is nonlinear discrimination capacity, not linear rank.

**4. Insufficient nonlinearity from single Boltzmann gate.** With 1 VFE iteration, the model has one softmax coupling application per head. The gate's competitive normalization redistributes gradient strength but cannot inject new features. More heads (not larger K) adds independent gates.

**5. sigma_p learns slowly.** With `sigma_ce_scale=0.1`, the CE gradient to prior covariance is 10x dampened. Combined with `M_alpha=0.0` (no VFE sigma gradient in M-step) and `lambda_hyper=0.0`, sigma_p adapts slowly. Per-token precision profiles that control decode sharpness are undertrained.

### Recommended Experiments (Priority Order)

1. **More heads, smaller d_head:** GL(10)x12 at K=120 instead of GL(15)x6 at K=90. Doubles Boltzmann gate count.
2. **Return delta from VFE FFN:** Change return to `mu_current - mu` so residual is `mu_q + delta` instead of `mu_q + norm(mu_q) + delta`.
3. **RMSNorm instead of LayerNorm before decode:** Preserves magnitude information for KL-decode.
4. **Increase sigma_ce_scale to 0.2-0.5:** Let sigma_p learn faster from CE signal.
5. **Small M_alpha > 0 (0.01-0.1):** Provides VFE-based sigma gradient and rewards good E-step initialization.
6. **Rebalance lambda_belief=2, lambda_softmax=4:** Increase nonlinear fraction from 25% to 57%.
7. **Enable exact_diagonal_transport:** Cheap fix for GL(d_h) transport correctness.
8. **Train 2 epochs:** The Boltzmann gate activates at ~1/4 epoch; more epochs give more effective nonlinear training time.

---

## Training Pipeline Issues

### ISSUE (HIGH): Per-Group Gradient Clipping Is Too Aggressive
**File:** `experiment_runner.py:1029-1039`

When `use_param_groups=True` (default), each of ~7 parameter groups (mu_embed, sigma_embed, phi_embed, attention, ffn, no_decay, output) is independently clipped to `grad_clip=1.0`. This is far more aggressive than global clipping: if 7 groups each clip to 1.0, the total gradient norm is at most 7.0, but more importantly, each group is independently constrained regardless of how much signal it carries.

With `M_mu_p_lr=0.05`, the maximum effective step for mu_embed is `0.05 * 1.0 = 0.05` per step. For phi_embed with `M_phi_lr=0.0075` and clip at 1.0, the max step is `0.0075`. These caps may prevent the model from making sufficiently large updates to escape local minima, directly contributing to a PPL plateau.

**Fix:** Increase `grad_clip` to 3.0-5.0 when using per-group clipping, or use different thresholds per group, or switch to global clipping.

### ISSUE (HIGH): Gradient Accumulation Silently Ignored
**File:** `experiment_runner.py:996, 1045, 1054`

`PublicationTrainer.train_step()` calls `loss.backward()` WITHOUT dividing by `grad_accumulation_steps`, and calls `optimizer.step()` on EVERY step. `FastTrainer.train_step()` correctly implements accumulation (divide loss, step every N). Since PublicationTrainer overrides the method entirely, `grad_accumulation_steps > 1` has no effect.

**Fix:** Implement accumulation in PublicationTrainer or document that it is not supported.

### ISSUE (MEDIUM): Double Regularization on Phi
**File:** `optimizer.py:476-482` (weight_decay=0.05 on phi) + `train.py:527-535` (mass_phi=0.01 loss term)

Both embed_weight_decay (0.05) and mass_phi (0.01) pull phi toward zero. This double penalty may over-regularize gauge frames, collapsing transport diversity and limiting attention expressiveness.

**Fix:** Set `embed_weight_decay=0.0` for phi when mass_phi > 0, or create a separate parameter group for phi with no weight decay.

### ISSUE (MEDIUM): Sliding Window Overlap
**File:** `datasets.py:772-779`

Consecutive training samples overlap by `max_seq_len - 1` tokens. With seq_len=128, sample 0 and sample 1 share 127/128 tokens. Shuffling helps but within a mini-batch, overlapping sequences reduce effective data diversity.

### Verified Correct
- Loss composition (CE + VFE terms) is additive, no double-counting
- Envelope theorem beta detach is correct
- Input/target alignment (shifted by 1) is correct
- Data shuffling is appropriate
- SL(K) projection timing is correct
- Token-weighted validation CE averaging is correct

---

## Files Modified During This Audit

| File | Change |
|------|--------|
| `gauge_preconditioner.py:384` | Fixed pullback Gram: removed `.transpose(-2, -1)` |
| `variational_ffn.py:2401` | Fixed pullback Gram: removed `.transpose(-2, -1)` |
| `training/experiment_runner.py:1865+` | Added hybrid model branch |
| `training/experiment_runner.py:577+` | Added `_model_blocks` property for hybrid compat |
| `training/experiment_runner.py` | Replaced 5x `self.model.transformer.blocks` with `self._model_blocks` |
| `train.py:589-635` | Replaced `model.transformer.blocks` with `_blocks` helper |
| `train_publication.py:995` | Added 'hybrid' to valid modes list |
| `baselines/hybrid_gauge_transformer.py:558+` | Fixed `forward_with_attention` to stack beta/kl across layers |
