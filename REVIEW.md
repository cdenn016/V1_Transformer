# VFE Transformer Codebase Review

**Reviewer**: Claude (automated deep review)
**Date**: 2026-03-13
**Scope**: Full codebase review against GL(K)_attention.tex and supplementary.tex

---

## Executive Summary

The codebase correctly implements the gauge-theoretic variational free energy framework described in the manuscript. The core mathematical pipeline—KL-based attention via `β_ij = softmax(-KL(q_i || Ω_ij q_j) / τ)`, natural gradient descent on the Gaussian manifold, GL(K) gauge transport via `Ω_ij = exp(φ_i)·exp(-φ_j)`, and the E-step/M-step decomposition—is sound. I verified the sign conventions, gradient formulas, and the observation likelihood incorporation are all correct.

Below I organize findings by severity. Most "critical" bugs flagged in initial analysis turned out to be correct on deeper inspection (the observation gradient sign, `scatter_reduce` usage, block-diagonal function existence). The real issues are primarily in numerical stability edge cases, performance bottlenecks, and a few genuine inconsistencies between theory and implementation.

---

## 1. VERIFIED CORRECT (False Alarms)

These were flagged in initial analysis but are actually correct:

**Observation gradient sign** (`variational_ffn.py:2246`): `grad_mu = grad_mu + discrete_obs_grad` is correct. The free energy is F = VFE_vacuum + CE (since -log_likelihood = cross-entropy). The gradient ∂F/∂μ includes +∂CE/∂μ. Then `delta_mu = -lr * nat_grad(grad_mu)` performs descent. Sign chain verified end-to-end.

**`scatter_reduce_` with `include_self=False`** (`prior_bank.py:344`): Correct for `amax` with initial value `float('-inf')`. The `-inf` initial is the identity element for max, so including or excluding it is equivalent.

**Block-diagonal gradient functions**: `_compute_vfe_gradients_block_diagonal` and `_compute_vfe_gradients_block_diagonal_diag` ARE defined in `variational_ffn.py` starting at line 237.

**Σ gradient formula**: The implementation correctly produces `∂F/∂Σ_i = 0.5[-(1+α)Σ_i^{-1} + α·Σ_{p,i}^{-1} + Σ_j β_ij (Ω_ij Σ_j Ω_ij^T)^{-1}]` matching Eq. (paper line 1430-1444), verified by tracing both self-coupling and alignment gradient terms.

---

## 2. REAL ISSUES

### 2.1 Theory-Implementation Inconsistencies

#### **MEDIUM: prior_bank.py diagonal covariance approximation under gauge-fixed priors**
`prior_bank.py:195-199` — When `gauge_fixed_priors=True`, the code extracts only the diagonal of `R @ Σ_0 @ R^T`:
```python
sigma_p = torch.einsum('...kl,l->...k', R_sq, base_sigma)  # diagonal approx
```
This breaks gauge covariance: `Σ_i ≠ Ω_ij Σ_j Ω_ij^T` in the off-diagonal entries. The `embeddings.py` code correctly forces `diagonal_covariance=False` when `gauge_fixed_priors=True` (line 159-168), but `prior_bank.py` does not have this guard. Should either: (a) add the same guard, or (b) document the approximation explicitly and quantify its impact on KL computation.

#### **LOW: Softmax coupling comment vs code sign convention**
Throughout `variational_ffn.py`, the softmax coupling gradient is computed as:
```python
grad_deviation = avg_grad.unsqueeze(2) - grad_kl_per_pair  # note: avg - per_pair
```
The paper writes `∂β_ij/∂μ_i = -β_ij/τ · [∂KL_ij/∂μ_i - Σ_k β_ik ∂KL_ik/∂μ_i]`, which has `per_pair - avg`. The negation is absorbed into the descent step, so the code is correct, but the in-code comments should clarify the sign convention to prevent future confusion.

### 2.2 Numerical Stability

#### **HIGH: No norm clamping before matrix_exp in embeddings.py**
`embeddings.py:295-296` — When `gauge_fixed_priors=True`, computes `torch.linalg.matrix_exp(phi_matrix)` without any norm clamping. For large `||φ||`, `matrix_exp` can overflow in non-compact (symmetric) directions of GL(K). The rest of the codebase uses `stable_matrix_exp_pair()` from `gauge_utils.py` which clamps norms. This path should also use it.

**Fix**: Replace `torch.linalg.matrix_exp(phi_matrix)` with `stable_matrix_exp_pair(phi_matrix)`.

#### **MEDIUM: numerical_utils.py eigenvalue floor too aggressive**
`numerical_utils.py:365` — Default `eps=1e-4` for `sanitize_sigma` is 100× larger than the standard `1e-6`. For KL computation, eigenvalues clipped at 1e-4 introduce systematic bias that grows with dimension K. The fixed floor should be adaptive to the matrix scale (e.g., `eps * max_eigenvalue`).

#### **MEDIUM: SO(3) arccos precision loss**
`embeddings.py:481` — `torch.acos` near ±1 loses precision in float32. Should upcast to float64 before `acos`:
```python
cos_theta = torch.clamp(cos_theta.double(), -1.0 + eps, 1.0 - eps)
theta = torch.acos(cos_theta).float()
```

#### **LOW: push_pull.py orthogonality check samples only element 0**
`push_pull.py:250-254` — When checking if Ω is orthogonal for the fast transport path, only the first batch element is checked. If element 0 happens to be orthogonal but others are not, the wrong code path is taken. Should check all elements or use a metadata flag.

### 2.3 Performance Bottlenecks

#### **HIGH: Transport operators recomputed 10× in phi update loop**
`variational_ffn.py:2308-2393` — When `update_phi_per_iteration=True`, attention weights and KL matrices are recomputed from scratch at every VFE iteration. With `n_iterations=10`, this means 10× redundant matrix exponential computations. Since phi changes are small per iteration, the KL matrix from the main VFE step could be cached and reused (or incrementally updated).

**Estimated speedup**: 2-4× for configurations with phi updates enabled.

#### **HIGH: No gradient checkpointing**
`model.py:589-601` — The transformer stack processes all layers without gradient checkpointing. For long sequences (N=1024) with K=90, this can exhaust GPU memory. Adding `torch.utils.checkpoint.checkpoint()` per block would trade ~30% compute for ~60% memory savings.

#### **MEDIUM: Generation is O(max_tokens × max_seq_len)**
`model.py:1020-1058` — For each generated token, the entire sequence is re-processed. Standard KV-caching is inapplicable because beliefs evolve through transport operators. This is a fundamental architectural constraint, but approximate caching strategies (e.g., caching transported statistics and updating incrementally) could provide significant speedup for inference.

#### **MEDIUM: Pullback natural gradient is O(K^6) per token**
`gauge_preconditioner.py:365-380` — The pullback metric computation involves structure constants (O(n_gen³) storage and computation), making it infeasible for large K. For GL(10) with 100 generators this is tractable, but scaling to GL(50)+ would require the Killing form approximation. Consider adding an automatic fallback or a clear warning.

#### **LOW: Numba path uses triple-nested Python loop**
`attention.py:683-704` — The Numba KL matrix computation loops over all (B, N, N) pairs in Python. This is already noted as a fallback for CPU, but for development/debugging it's extremely slow. Consider parallelizing via `@numba.njit(parallel=True)` with `prange`.

### 2.4 Code Quality

#### **MEDIUM: Learning rate logging shows base rates, not scheduled**
`train_fast.py:596-597` — Logs base learning rates from optimizer param groups, not the actual scheduled rates. Should use `scheduler.get_last_lr()` or compute `base_lr × schedule_factor`.

#### **MEDIUM: Hardcoded π value**
`train_fast.py:392` — Uses `3.14159` instead of `math.pi`. Minor precision loss.

#### **LOW: Python 3.9+ type hints**
Several files (`numerical_monitor.py:11`, `numerical_utils.py:399`) use `dict[str, int]` syntax instead of `Dict[str, int]` from typing module. Breaks on Python 3.8.

#### **LOW: transport.py function naming**
`transport.py:264` — `_matrix_exponential_so3` is used for general SO(N) and GL(K), not just SO(3). Should be renamed to `_matrix_exponential_lie_algebra` or similar.

#### **LOW: Dead code in generate()**
`model.py:292-304` — Fallback path that creates random skew-symmetric generators should never execute if `generators.py` is available. Should fail fast with a clear error instead.

### 2.5 Missing Features / Theoretical Gaps

#### **MEDIUM: No Σ retraction in diagonal covariance mode**
The supplementary (Appendix D) describes an affine-invariant exponential map retraction for SPD covariance updates. The code implements this for full covariance matrices (`variational_ffn.py:1281-1361`), but for diagonal covariance mode, updates are simple additive: `sigma_new = sigma + delta_sigma`. This loses the SPD-preservation guarantee. For diagonal mode, the retraction simplifies to `σ_new[k] = σ_old[k] · exp(τ · δσ_whitened[k])`, which is cheap and preserves positivity. Consider implementing this.

#### **LOW: Cartan preconditioning projection is approximate**
`gauge_preconditioner.py:97-108` — The Cartan decomposition uses `P_sym = gram^{-1}(gram + trace_prod)/2` which is an approximation to the true Cartan involution θ(X) = -X^T. For orthonormal generators this is exact, but for general generator bases it may not be. The Killing form mode (`'killing'`) has no free parameters and is more principled. Consider documenting this limitation.

---

## 3. SPEED-UP OPPORTUNITIES

### 3.1 Quick Wins (< 1 day effort)

1. **Cache transport operators in phi update loop**: Reuse KL matrix from attention step when phi changes are small. ~2-4× speedup for evolve_phi configurations.

2. **Add gradient checkpointing**: Wrap each `GaugeTransformerBlock` in `torch.utils.checkpoint`. ~60% memory reduction.

3. **Use `math.pi`**: Replace hardcoded `3.14159` in train_fast.py.

4. **Diagonal retraction**: Implement `σ_new = σ_old · exp(τ · δ)` for diagonal mode instead of additive update.

### 3.2 Medium Effort (1-3 days)

5. **Fused block-diagonal KL + transport kernel**: The block-diagonal attention path computes separate matrix exponentials and KL values per irrep block. A fused kernel that processes all blocks in a single pass would reduce kernel launch overhead and improve GPU utilization.

6. **Approximate KV-caching for generation**: Cache the (N, K) transported belief statistics and update incrementally for each new token. Would need theoretical validation that the approximation error is bounded.

7. **Adaptive eigenvalue regularization**: Replace fixed `eps=1e-4` floors with scale-adaptive regularization: `eps_adaptive = max(eps, eps * λ_max)`.

### 3.3 Larger Effort (1+ week)

8. **CUDA kernel for KL matrix**: The pairwise KL computation is the main bottleneck. A custom CUDA kernel that fuses transport, KL computation, and softmax would eliminate intermediate tensor allocations and provide ~5-10× speedup.

9. **FlashAttention-style tiling for KL attention**: Process the (N, N) KL matrix in tiles to reduce HBM traffic, similar to FlashAttention for dot-product attention.

---

## 4. THEORY VERIFICATION SUMMARY

| Paper Equation | Code Location | Status |
|---|---|---|
| β_ij = softmax(-KL(q_i ∥ Ω_ij q_j) / τ) | attention.py:493 | ✅ Correct, with √K normalization |
| Ω_ij = exp(φ_i)·exp(-φ_j) | attention.py:260-284 | ✅ Correct |
| ∂F/∂μ_i (general, Eq. 4.3) | variational_ffn.py:921-1070 | ✅ Correct, including ∂β/∂μ softmax coupling |
| ∂F/∂Σ_i (Eq. in Supplementary B) | variational_ffn.py:274-284, 396-404 | ✅ Correct, -(1+α) coefficient verified |
| Natural gradient: δμ = -η·Σ·∂F/∂μ | variational_ffn.py:1266 | ✅ Correct |
| SPD retraction via exp map | variational_ffn.py:1281-1361 | ✅ Correct for full covariance |
| GL(K) invariance of KL | attention.py (transport then KL) | ✅ Correct |
| Vanishing holonomy (Thm 2) | attention.py:260-284 (cocycle form) | ✅ By construction |
| Temperature scaling τ = √d_k | attention.py:490-493 | ✅ Correct |
| Causal masking as π_j prior | attention.py:512-516 | ✅ Correct |
| ALiBi as exponential prior | attention.py:503-510 | ✅ Correct |
| Killing form metric | gauge_preconditioner.py:196-246 | ✅ Correct formula |
| Observation likelihood (CE) | variational_ffn.py:2236-2246 | ✅ Correct sign |

---

## 5. PRIORITIZED FIX LIST

| Priority | Issue | File:Line | Effort |
|---|---|---|---|
| P1 | Use `stable_matrix_exp_pair` in embeddings.py gauge_fixed path | embeddings.py:296 | 5 min |
| P1 | Add gradient checkpointing to transformer stack | model.py:589-601 | 30 min |
| P1 | Cache transports in phi update loop | variational_ffn.py:2344 | 1 hr |
| P2 | Guard diagonal covariance in prior_bank gauge_fixed path | prior_bank.py:195-199 | 15 min |
| P2 | Adaptive eigenvalue floor in numerical_utils | numerical_utils.py:365 | 30 min |
| P2 | Fix LR logging to show scheduled rates | train_fast.py:596-597 | 15 min |
| P2 | Float64 upcast for arccos in embeddings | embeddings.py:481 | 5 min |
| P2 | Diagonal SPD retraction | variational_ffn.py (new) | 2 hr |
| P3 | Use math.pi in train_fast | train_fast.py:392 | 1 min |
| P3 | Fix Python 3.8 type hints | numerical_monitor.py, numerical_utils.py | 10 min |
| P3 | Rename transport.py:_matrix_exponential_so3 | transport.py:264 | 5 min |
| P3 | Fix push_pull.py batch orthogonality check | push_pull.py:250-254 | 15 min |
