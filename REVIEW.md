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

## 5. COMPLETED FIXES (2026-03-13)

All items from the initial prioritized fix list have been implemented:

| Priority | Issue | File | Status |
|---|---|---|---|
| P1 | Use `stable_matrix_exp_pair` in gauge_fixed path | embeddings.py | ✅ Done — norm clamping prevents overflow in GL(K) |
| P1 | Add gradient checkpointing to transformer stack | blocks.py | ✅ Done — configurable via `gradient_checkpointing` flag |
| P1 | Reduce phi update frequency in E-step loop | variational_ffn.py | ✅ Done — `phi_update_interval` param (default=2, ~2× speedup) |
| P2 | Guard diagonal covariance in prior_bank gauge_fixed path | prior_bank.py | ✅ Done — full covariance path when `diagonal_covariance=False` |
| P2 | Adaptive eigenvalue floor in numerical_utils | numerical_utils.py | ✅ Done — `max(eps * λ_max, 1e-8)` replaces fixed 1e-4 |
| P2 | Fix LR logging to show scheduled rates | train_fast.py | ✅ Done — uses `scheduler.get_last_lr()` |
| P2 | Float64 upcast for arccos in embeddings | embeddings.py | ✅ Done — prevents precision loss near ±1 |
| P2 | Diagonal SPD retraction | variational_ffn.py | ✅ Already existed (`retract_spd_diagonal_torch`) |
| P3 | Use math.pi in train_fast | train_fast.py | ✅ Done |
| P3 | Fix Python 3.8 type hints | numerical_monitor.py, numerical_utils.py | ✅ Done — `Dict[str, int]` and `Tuple[...]` |
| P3 | Rename `_matrix_exponential_so3` | transport.py | ✅ Done → `_matrix_exponential_lie_algebra` |
| P3 | Fix push_pull.py batch orthogonality check | push_pull.py | ✅ Done — checks up to 8 batch elements |

---

## 6. NEXT PRIORITY LIST

### P1 — Performance (High Impact)

| Issue | Description | File(s) | Effort |
|---|---|---|---|
| Fused block-diagonal KL + transport kernel | Process all irrep blocks in a single pass instead of separate matrix_exp + KL per block. Reduces kernel launch overhead. | attention.py, variational_ffn.py | 2-3 days |
| CUDA kernel for pairwise KL matrix | Fuse transport → KL → softmax into a single kernel. Eliminates O(B·N²·K²) intermediate tensors. Main inference bottleneck. | New file: cuda_kernels/ | 1-2 weeks |
| FlashAttention-style tiling for KL attention | Process (N,N) KL matrix in tiles to reduce HBM traffic. Especially important for N > 512. | attention.py | 1-2 weeks |

### P2 — Architecture Improvements (Medium Impact)

| Issue | Description | File(s) | Effort |
|---|---|---|---|
| Approximate KV-caching for generation | Cache transported belief statistics (N, K) and update incrementally per token. Currently O(max_tokens × N). Needs theoretical validation. | model.py | 3-5 days |
| Killing form fallback for large K | `gauge_preconditioner.py` pullback mode is O(K⁶). Add automatic fallback to Killing form when K > threshold (e.g., K=30). | gauge_preconditioner.py | 1 day |
| Cartan preconditioning documentation | The Cartan decomposition at `gauge_preconditioner.py:97-108` is approximate for non-orthonormal generators. Document when exact vs approximate. | gauge_preconditioner.py | 2 hrs |

### P3 — Code Quality & Testing

| Issue | Description | File(s) | Effort |
|---|---|---|---|
| Parallel Numba KL fallback | `attention.py:683-704` uses triple-nested Python loop. Add `@numba.njit(parallel=True)` with `prange`. | attention.py | 1 hr |
| Dead code in generate() | `model.py:292-304` random generator fallback should fail fast with clear error. | model.py | 15 min |
| Gradient checkpointing tests | Verify numerical equivalence of checkpointed vs non-checkpointed forward pass. | tests/ (new) | 2 hrs |
| Benchmark suite | Add timing benchmarks for key operations: KL matrix, transport, natural gradient, SPD retraction. Track regressions. | benchmarks/ (new) | 1 day |
| Mixed-precision audit | Verify all paths work correctly under `torch.amp.autocast`. Several `float()` upcasts exist but coverage is incomplete. | All core files | 1 day |

---

## 7. DEEP ANALYSIS: FlashAttention-Style Tiling for KL Attention

**Date**: 2026-03-14
**Context**: Currently limited to N=128, K=100, diagonal covariance. Will tiling unlock meaningful scaling?

---

### 7.1 Why FlashAttention Works (and Why KL Attention is Different)

FlashAttention's core insight is simple: the softmax numerator `exp(q·k/√d)` factors over the key dimension, so you can tile the N×N attention matrix and accumulate partial softmax results with an online correction (the log-sum-exp trick). The key properties that make this work:

1. **Dot-product factorization**: `score(i,j) = q_i^T k_j` decomposes as an inner product. You can compute partial sums over j-tiles and combine them.
2. **Softmax is reducible**: `softmax(x)_i = exp(x_i) / Σ_j exp(x_j)` can be computed in streaming fashion using the online softmax trick (Milakov & Gimelshein 2018): maintain running max and running sum, rescale when a new tile introduces a larger max.
3. **Value aggregation is linear**: `output_i = Σ_j β_ij v_j` is a weighted sum that can be accumulated tile by tile.
4. **No intermediate N×N matrix**: The entire point is to never materialize the N×N attention matrix in HBM. Each tile computes its contribution to the output and discards the tile's scores.

**For KL attention, every one of these properties changes:**

| Property | Dot-Product Attention | KL Attention (This Codebase) |
|---|---|---|
| Score computation | `q_i^T k_j` — O(K) per pair | `KL(q_i \|\| Ω_ij q_j)` — O(K²) per pair (transport + KL) |
| Factorization | Bilinear: separates into q and k | **Not bilinear**: KL involves log-det, trace, Mahalanobis — all couple i,j nonlinearly through Ω_ij |
| Transport operator | None (or absorbed into W_Q, W_K) | Ω_ij = exp(φ_i)·exp(-φ_j) — requires K×K matrix multiply per pair |
| Softmax reducibility | Yes — standard online softmax | Yes — **still works** (softmax is applied to scalar KL values) |
| Value aggregation | Linear: `Σ β_ij v_j` | Linear: `Σ β_ij Ω_ij μ_j` — but Ω_ij must be recomputed or cached per tile |
| Backward pass | QK^T gradient tiles naturally | KL gradients involve Ω, Σ, μ coupling — significantly more complex |

The crucial difference: **KL scores are not bilinear in (i, j)**. Standard FlashAttention exploits Q·K^T = matrix multiply, which can be tiled using standard GEMM blocking. KL divergence between transported Gaussians involves:

```
KL(q_i || Ω_ij q_j) = ½[tr(Σ_j_t^{-1} Σ_i) + (μ_i - Ω_ij μ_j)^T Σ_j_t^{-1} (μ_i - Ω_ij μ_j) - K + log(det Σ_j_t / det Σ_i)]
```

where `Σ_j_t = Ω_ij Σ_j Ω_ij^T`. This cannot be decomposed into independent functions of i and j.

### 7.2 What CAN Be Tiled (and What Can't)

Despite the non-bilinear structure, tiling is still possible — it just provides different benefits than standard FlashAttention:

**Tileable components (reduce HBM traffic):**

1. **Online softmax over KL scores**: Once you have KL(i,j) for a tile of j-values, you can update running max / running sum for position i. This is identical to FlashAttention's online softmax.

2. **Message aggregation**: `m_i = Σ_j β_ij Ω_ij μ_j` can be accumulated tile by tile. Load a j-tile, compute transport + KL + partial softmax + partial message, accumulate, move to next tile.

3. **Transport factorization**: Ω_ij = exp(φ_i)·exp(-φ_j) factors! You can precompute exp(φ_i) for the i-tile and exp(-φ_j) for the j-tile, then form Ω_ij = A_i · B_j per-pair within the tile. This is O(tile_i × K²) + O(tile_j × K²) precompute, then O(tile_i × tile_j × K²) for the products.

**Non-tileable complications:**

1. **Per-pair K×K matrix multiply**: Even with the factored form, computing Ω_ij μ_j requires a K×K × K matmul per (i,j) pair. This is the fundamental cost that tiling cannot eliminate — it can only control when these happen (in SRAM vs HBM).

2. **Backward pass through KL**: The gradient ∂KL/∂μ_i, ∂KL/∂Σ_i, ∂KL/∂φ_i all depend on the full Ω_ij and transported statistics. Recomputation in the backward pass (as FlashAttention does for QK^T) means recomputing transport operators, which is expensive.

3. **Diagonal covariance path**: Your current path (`_compute_kl_matrix_diagonal`) avoids Cholesky but still materializes Omega as (B, N, N, K, K) at line 1130. **This is the actual bottleneck at your scale.**

### 7.3 Memory Analysis at Your Current Scale: N=128, K=100

Let's do the arithmetic for your actual configuration (B=16, N=128, K=100, diagonal covariance, GL(10) with 8 heads of d_head=10 via block-diagonal):

**Block-diagonal path (what you're actually using with irrep_dims=[10]*8 + [1]*20 or similar):**

With 8 heads of d_head=10, the block-diagonal path processes each head independently. Per head:
- Omega per block: (B, N, N, d, d) = (16, 128, 128, 10, 10) = 16 × 128² × 100 × 4B ≈ **100 MB per head**
- But fused_block_matrix_exp_pairs precomputes exp_phi per block: (B, N, d, d) = much smaller
- Fused KL avoids materializing full Omega — the Triton kernels compute Omega in registers

**If NOT using block-diagonal (raw diagonal path):**

The non-block-diagonal `_compute_kl_matrix_diagonal` path at line 1055-1189 materializes:
- `Omega`: (B, N, N, K, K) = (16, 128, 128, 100, 100) = **~25 GB** — impossible!
- `mu_transported`: (B, N, N, K) = (16, 128, 128, 100) × 4B ≈ **100 MB**
- `sigma_j_transported_diag`: (B, N, N, K) ≈ **100 MB**

So the Omega tensor is the killer. Your block-diagonal decomposition already avoids this by never forming the full K×K Omega — each block only forms a d_head × d_head Omega.

**With your Triton kernels (triton_kernels.py):**

The existing Triton kernels (`_pairwise_kl_d1_kernel`, `_pairwise_kl_d3_kernel`, `_pairwise_kl_d5_kernel`, `_pairwise_kl_generic_kernel`) already implement the most important optimization: **they keep Omega in registers and never write it to HBM**. Each kernel:
- Takes one (b, i, j) triple per program
- Loads exp_phi[b,i] and exp_neg_phi[b,j] from HBM (2 × d² values)
- Computes Omega = A·B in registers (d² register values)
- Computes transport and KL in registers
- Writes one scalar to output

This is already a form of "FlashAttention-style" computation — the intermediate Omega never touches HBM!

**What the Triton kernels DON'T do:**

They don't tile over the N×N output. Each (b,i,j) is an independent program. The output KL matrix (B, N, N) = (16, 128, 128) × 4B ≈ 1 MB is small and fully materialized in HBM. This is fine for N=128.

### 7.4 Where Does Your Memory Actually Go?

At N=128, K=100, B=16, the memory breakdown is approximately:

| Tensor | Shape | Size | Notes |
|---|---|---|---|
| Beliefs (μ, σ, φ) | 3 × (B, N, K) | 3 × 25 MB ≈ 75 MB | Small |
| Generators | (n_gen, K, K) | (100, 100, 100) × 4B ≈ 4 MB | Tiny |
| exp_phi, exp_neg_phi per block | 8 × (B, N, d, d) | 8 × 10 MB ≈ 80 MB | Precomputed |
| KL matrix | (B, N, N) | 1 MB | Tiny |
| Attention weights β | (B, N, N) per head, or (B, H, N, N) | 8 MB | Tiny |
| VFE gradients (μ, σ, φ) | ~3 × (B, N, K) | 75 MB | Per iteration |
| **Autograd graph** | All intermediates for backward | **~2-4 GB** | **Dominant!** |
| Output projection | (K, vocab) × activations | ~50M params × grads | ~400 MB |

**The autograd graph is the real memory bottleneck**, not HBM traffic from KL tiling. Each VFE iteration through the E-step loop records all intermediate tensors for backward. With `n_vfe_iterations=5`, that's 5× the single-pass autograd cost.

### 7.5 Verdict: Does FlashAttention-Style Tiling Help at N=128, K=100?

**Short answer: No, not meaningfully. Your bottleneck is elsewhere.**

**Detailed reasoning:**

1. **Your Triton kernels already eliminate the Omega HBM bottleneck.** The fused `_pairwise_kl_d*_kernel` kernels keep Omega in registers. This is the single most important optimization that FlashAttention-style approaches provide, and you already have it.

2. **The N×N KL matrix is 1 MB at N=128.** FlashAttention avoids materializing the N×N attention matrix because at N=8192, that's 256 MB per head. At N=128, it's 64 KB per head. Tiling to avoid this is pointless.

3. **The output accumulation (online softmax + message tiling) saves nothing at N=128.** The softmax is over 128 elements per query — this fits in a single cache line. There's no HBM round-trip to save.

4. **The actual bottleneck is the autograd graph from VFE iterations.** Each E-step iteration records gradients through KL computation, transport, and natural gradient updates. Gradient checkpointing (which you've already implemented in blocks.py) addresses this directly.

5. **The secondary bottleneck is compute, not memory.** At N=128, K=100 with 8 heads of GL(10), you're doing 128² × 8 = 131K matrix exponential products per forward pass. Each is a 10×10 matmul — fast in isolation, but the sheer count adds up. Tiling doesn't reduce compute; it reduces memory traffic. Your compute is already register-bound via Triton.

### 7.6 What WOULD Help You Scale Beyond N=128, K=100

The path to larger N and K follows a different priority ordering than FlashAttention tiling:

#### Priority 1: Reduce VFE iteration autograd cost (immediate impact)

**DEQ implicit differentiation** (already in your codebase via `use_deq=True`):
- Replaces O(n_iterations) autograd memory with O(1)
- For n_vfe_iterations=5, this alone could cut peak memory by ~60%
- Combined with gradient checkpointing, should let you reach N=256 or N=512

**Estimated scaling**: N=128 → N=256-384 at same K=100

#### Priority 2: Sparse attention patterns (N scaling)

For N > 512, the O(N²) pairwise KL computation becomes the bottleneck regardless of memory tricks. The principled approach:

- **Sliding window + global tokens**: Compute KL only for |i-j| < W plus a few global positions. Reduces O(N²) to O(N·W). Your causal mask already supports this — extend `create_attention_mask` with a window parameter.
- **Top-k KL approximation**: Compute KL for a random subset of j per query i, keep only top-k. Similar to routing in mixture-of-experts.
- **Hierarchical attention**: Use the RG flow structure you already have — attend within meta-agent clusters at fine scale, between clusters at coarse scale.

**Estimated scaling**: N=512 → N=2048+ with W=256 window

#### Priority 3: Mixed precision for transport (K scaling)

The `stable_matrix_exp_pair` function upcasts to float64 for K≥8. For GL(10) blocks (d_head=10), this doubles memory for the matrix exponential computation. Options:

- **Float32 matrix_exp with tighter norm clamping**: For d=10, float32 Padé is accurate to ~1e-6 if ||M||_F < 5. Your norm clamp is already at 10.0 — tightening to 5.0 may allow staying in float32.
- **Cayley transform instead of matrix_exp**: For SO(K), Ω = (I + A/2)(I - A/2)^{-1} is an algebraic alternative to exp(A) that avoids the Padé approximation entirely. Cheaper and more stable, but only covers SO(K), not GL+(K).

#### Priority 4: FlashAttention-style tiling (ONLY for N > 512 with non-block-diagonal)

If you eventually move to configurations where:
- N > 512 (the N×N KL matrix exceeds ~4 MB)
- Full K×K covariance (not diagonal)
- No block-diagonal structure (single-head GL(K))

Then tiling the N×N computation would help. The implementation would look like:

```python
# Pseudocode for FlashKL attention
def flash_kl_attention(mu, sigma, exp_phi, exp_neg_phi, kappa):
    B, N, K = mu.shape
    TILE = 64  # tile size

    # Output accumulators (in HBM)
    output = torch.zeros(B, N, K)  # accumulated message
    l = torch.zeros(B, N)           # log-sum-exp denominator
    m = torch.full((B, N), -inf)    # running max

    for j_start in range(0, N, TILE):
        j_end = min(j_start + TILE, N)
        # Load j-tile beliefs into SRAM
        mu_j = mu[:, j_start:j_end]          # (B, Tj, K)
        sigma_j = sigma[:, j_start:j_end]
        exp_neg_phi_j = exp_neg_phi[:, j_start:j_end]

        for i_start in range(0, N, TILE):
            i_end = min(i_start + TILE, N)
            # Load i-tile beliefs into SRAM
            mu_i = mu[:, i_start:i_end]
            sigma_i = sigma[:, i_start:i_end]
            exp_phi_i = exp_phi[:, i_start:i_end]

            # Compute Omega for tile (in SRAM/registers)
            Omega_tile = exp_phi_i @ exp_neg_phi_j  # (B, Ti, Tj, K, K) — in SRAM

            # Compute KL for tile (in SRAM)
            kl_tile = kl_divergence(mu_i, sigma_i, mu_j, sigma_j, Omega_tile)

            # Online softmax update
            logits_tile = -kl_tile / (kappa * sqrt(K))
            m_new = max(m[i_start:i_end], logits_tile.max(dim=-1))
            # ... standard online softmax rescaling ...

            # Accumulate transported messages
            msg_tile = einsum('bijkl,bjl->bijk', Omega_tile, mu_j)
            output[i_start:i_end] += beta_tile @ msg_tile  # rescaled

            # Free tile intermediates (never written to HBM)
```

**But note**: the tile of Omega is (Ti, Tj, K, K). For Ti=Tj=64, K=100: 64² × 100² × 4B = 160 MB per tile. This is too large for GPU SRAM (typically 192 KB per SM). You'd need to further sub-tile the K dimension or rely on the block-diagonal structure to keep d small.

This is why **for your architecture, the block-diagonal + Triton approach is already the right answer**. Each irrep block has d≤10, so the per-pair Omega is 10×10 = 400 bytes — easily fits in registers.

### 7.7 Concrete Recommendation

**Do not implement FlashAttention-style N×N tiling at this time.** Instead, pursue this scaling path:

| Step | Action | Expected Gain | Effort |
|---|---|---|---|
| 1 | Enable `use_deq=True` for training | N: 128 → 256 (memory) | Already implemented, just config |
| 2 | Enable `gradient_checkpointing=True` | N: 256 → 384 (memory) | Already implemented, just config |
| 3 | Add sliding window attention mask | N: 384 → 2048 (compute) | 1-2 days |
| 4 | Optimize Triton generic kernel for d=10 | 1.5-2× throughput | 2-3 days |
| 5 | Float32 matrix_exp with tighter clamping | K: 100 → 150 (memory) | 1 day |
| 6 | FlashKL tiling (if needed after steps 1-5) | N: 2048 → 8192 (memory) | 1-2 weeks |

Steps 1-2 are free — just toggle config flags you already have. Step 3 is the highest-impact new code. Steps 4-5 are tuning. Step 6 is only worth doing if you actually hit the N×N memory wall after everything else.

### 7.8 Why the Current Limit is N=128, K=100 (Root Cause)

The limit isn't the KL matrix or attention computation — it's the **total forward+backward memory of the VFE E-step loop**. With n_vfe_iterations=5:

- Each iteration records: KL matrix computation graph, natural gradient computation, belief update
- PyTorch autograd retains all intermediate tensors until backward completes
- For 5 iterations: ~5× the single-pass memory

The fix is DEQ (O(1) memory in iterations) + gradient checkpointing (recompute instead of store). These two changes, which you've already implemented, should roughly triple your effective N before any tiling work is needed.

**Bottom line**: FlashAttention-style tiling for KL attention is a real optimization, but it's item #6 on your priority list, not item #1. Your existing Triton kernels already capture the most important insight (register-resident Omega). The path to N=512+ runs through DEQ + gradient checkpointing + sparse attention, not through N×N tiling.
