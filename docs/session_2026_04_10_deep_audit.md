# Deep Audit — 2026-04-10

New findings NOT addressed by the prior 7-round audit (2026-04-07). Three parallel agents audited: (1) training numerics and gradient flow, (2) performance bottlenecks, (3) mathematical consistency across implementations.

## CRITICAL / HIGH Findings

### Finding 1 — Double preconditioning of phi M-step gradients (LIVE IN PRODUCTION)

**Files:** `experiment_runner.py:1188-1196`, `optimizer.py:116-126`

When `use_killing_form=True` AND `optimizer_type='riemannian_adam'` — which is the EM_CONFIG default — phi gradients are preconditioned twice:

1. `experiment_runner.py:1194`: Cartan projection `grad → grad @ C^T` (dampens symmetric directions by `sym_dampening`)
2. `optimizer.py:126`: Killing inverse `grad → grad @ K_inv` (full Riemannian natural gradient)

The effective transform `grad → C^T @ K_inv` is geometrically meaningless — it is neither the Cartan-projected gradient nor the Killing natural gradient. The Cartan projector is a cheaper approximation of what the Killing inverse already does precisely (weight compact vs non-compact directions). Applying both distorts the gradient in a way that doesn't correspond to any known metric.

**Impact:** Every training run using the default EM_CONFIG has been running with distorted phi M-step gradients. The model still trains because AdamW's adaptive moments partially compensate, but the optimization trajectory on GL(K) is suboptimal.

**Fix:** When `optimizer_type='riemannian_adam'`, skip the Cartan preconditioning in `experiment_runner.py` (the optimizer already handles it with the proper metric). Add a mutual-exclusion guard.

### Finding 2 — KL ceiling mismatch: `train.py` uses `20*K`, everything else uses `5*K`

**Files:** `train.py:185` vs `gauge_utils.py:17`, `vfe_utils.py:32`

The M-step loss function (`gaussian_kl_divergence`) clamps KL at `max(100, 20*K)`. All E-step paths (attention, VFE gradients) clamp at `max(100, 5*K)`. For K=20: M-step ceiling = 400, E-step ceiling = 100.

This creates an E-step/M-step asymmetry: the M-step loss tries to minimize KL values that the E-step clips away. The stale comment at `gauge_utils.py:16` ("current default is 20.0") confirms this was a leftover from an earlier constant.

**Fix:** Change `train.py:185` from `20.0 * K` to `KL_CEIL_SCALE * K` (import from `vfe_utils`), or hardcode `5.0 * K` to match. Fix the stale comment.

### Finding 3 — Per-block KL ceiling uses block dim `d` in gradients but full dim `K` in attention

**Files:** `vfe_gradients.py:382,695,1021` vs `gauge_utils.py:343`

Gradient paths clamp per-block KL at `max(100, 5*d)` where `d` is the irrep block dimension. Attention paths clamp at `max(100, 5*K)` where `K` is the full embedding dim.

At the current config (K=20, d=10), both equal 100 (the 100 floor dominates). For K>20, the asymmetry becomes significant: attention allows much higher per-block KL than the gradient softmax-coupling multiplier.

**Fix for K>20 configs:** Use `max(100, KL_CEIL_SCALE * K)` in the gradient paths, matching attention.

## MEDIUM Findings

### Finding 4 — Natural gradient sigma stall at small variance

**File:** `vfe_gradients.py:1708-1712`

`nat_grad_sigma = 2 * sigma^2 * grad_sigma` is the correct Fisher metric inverse, but creates a positive feedback trap: when sigma collapses to small values (e.g., 1e-3), the natural gradient is suppressed by `sigma^2 = 1e-6`, preventing recovery. The floor `eps=1e-6` only guards against zero, not against the quadratic suppression.

**Impact:** Collapsed covariances cannot recover through the E-step natural gradient alone. The M-step sigma_p gradient (which uses a different path) must compensate.

### Finding 5 — `use_obs_in_vfe=True` makes training loss unreliable

**File:** `variational_ffn.py:1914-1931`

When `use_obs_in_vfe=True`, the E-step steers beliefs toward the correct answer before CE is computed. Training CE is evaluated on beliefs that already "saw" the targets. The `detach()` prevents gradient leakage to parameters, but train/val loss comparison becomes misleading. Currently `use_obs_in_vfe=False` in EM_CONFIG, so this is latent.

### Finding 6 — `belief_align_loss` scales with sequence length

**File:** `train.py:384-388`

`weighted_kl.sum(dim=(-2, -1))` sums over N×N pairs. CE loss is O(1) in N (mean over tokens). The effective weight of belief alignment relative to CE grows linearly with context length. `M_beta` is not sequence-length-invariant. Currently `M_beta=0.0` in EM_CONFIG, so this is latent.

## PERFORMANCE Findings

### Perf 1 — `num_workers=10` on Windows is likely slower than `num_workers=0`

**File:** `train_publication.py:327`

Windows uses `spawn` for multiprocessing. Each of the 10 workers re-imports the full Python environment. With tokenized data cached to disk, in-process loading is faster. Setting `num_workers=0` eliminates 2-10 second startup overhead and may improve steady-state throughput. Zero-risk config change.

### Perf 2 — `_shared_bep` ignored when `update_phi_per_iteration=True`

**File:** `variational_ffn.py:2615`

The matrix exponentials computed in `blocks.py` (`_shared_bep`) are discarded when `update_phi_per_iteration=True` (EM_CONFIG default). The first VFE iteration could reuse them since phi hasn't changed yet. Saves one `fused_block_matrix_exp_pairs` call per forward pass (~2-5 ms).

### Perf 3 — `zero_grad(set_to_none=True)` missing in `PublicationTrainer`

**File:** `experiment_runner.py:1249`

Uses `self.optimizer.zero_grad()` (fills with zeros) instead of `zero_grad(set_to_none=True)` (deallocates). `FastTrainer` already uses the efficient version. One-line fix.

### Perf 4 — Redundant matrix exp in `_compute_phi_grad`

**File:** `variational_ffn.py:1035-1042`

`phi_current.clone().requires_grad_(True)` forces a fresh `fused_block_matrix_exp_pairs` call to build an autograd graph, even though the result is numerically identical to the cached BEPs. Saves ~2-5 ms per forward pass but requires careful validation. High implementation risk.

### Perf 5 — Lazy imports in `vfe_gradients.py` block `torch.compile`

**File:** `vfe_gradients.py:603`

`from transformer.core.transport_ops import _apply_rope, _un_apply_rope_pair_outer` inside a function body causes graph breaks. Should be hoisted to module level for future torch.compile compatibility.

## LOW / Informational

- **Stale comment** at `gauge_utils.py:16`: says "current default is 20.0" but value is 5.0
- **kappa_warmup prints misleading message** when `learnable_head_kappa=False` (the warmup is a no-op)
- **VFE terms lack padding mask** (`train.py:406-443`): KL computed over padding positions. Low impact since datasets don't produce padded batches in practice
- **sigma_ce_scale documentation inconsistency**: default 0.1 in code, docstring says 0.01

## Summary

| # | Finding | Severity | Live in prod? | Fix risk |
|---|---------|----------|---------------|----------|
| 1 | Double phi preconditioning | HIGH | YES | Low |
| 2 | KL ceiling 20*K vs 5*K | HIGH | YES | Low |
| 3 | Per-block KL ceiling d vs K | MEDIUM | No (K=20) | Low |
| 4 | Sigma natural gradient stall | MEDIUM | Latent | Medium |
| 5 | use_obs_in_vfe loss unreliability | MEDIUM | No (off) | Docs only |
| 6 | belief_align_loss scales with N | MEDIUM | No (M_beta=0) | Low |
| P1 | num_workers=10 on Windows | PERF | YES | None (config) |
| P2 | _shared_bep ignored first iter | PERF | YES | Low |
| P3 | zero_grad inefficient | PERF | YES | None (1 line) |
| P4 | Redundant matrix exp for phi grad | PERF | YES | High |
| P5 | Lazy imports block compile | PERF | Latent | None |
