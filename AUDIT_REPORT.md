# Gauge-Transformer Deep Code Audit Report

**Date**: 2026-03-31
**Scope**: Full codebase (118 Python files, ~62,580 LOC)
**Audited by**: 5 parallel code-reviewer agents covering core architecture, math utilities, training pipeline, pure VFE, and hard constraint sweep

---

## Executive Summary

| Category | Status | Issues |
|----------|--------|--------|
| Gauge Equivariance (sandwich product) | **CLEAN** | 0 violations across entire codebase |
| Timescale Separation (sigma_p detach) | **CLEAN** | Correctly detached everywhere |
| No Neural Networks | **2 VIOLATIONS** | connection.py MLP path, attention.py W_O projection |
| No CLI Arguments | **16 VIOLATIONS** | train_publication.py (critical), 15 auxiliary files |
| Correctness Bugs | **4 CRITICAL** | Missing arg in variational_ffn.py, NameError in inference.py, broken grad accumulation, CUDA/Python divergence |

**Total findings**: 9 Critical, 16 Warnings, 23 Suggestions

---

## CRITICAL Findings

### C1. Missing `lambda_softmax` argument causes parameter shift
**File**: `transformer/core/variational_ffn.py:1483-1489`
**Impact**: Completely wrong gradients if non-fused diagonal block-diagonal path is exercised

The call to `_compute_vfe_gradients_block_diagonal_diag` is missing `lambda_softmax`, causing all subsequent positional arguments to shift: `kappa` is passed as `lambda_softmax`, `eps` (1e-6) as `kappa`, and `irrep_dims` (a list!) as `eps`.

**Fix**: Add `lambda_softmax,` between `lambda_belief,` and `kappa,` at line 1485.

### C2. `inference.py` uses `Path` before importing it
**File**: `inference.py:23`
**Impact**: `NameError` crash on every import/execution

```python
CHECKPOINT_PATH = str(Path("checkpoints_publication") / ...)  # line 23
# ...
from pathlib import Path  # line 38
```

**Fix**: Move `from pathlib import Path` to top of file before config section.

### C3. `PublicationTrainer.train_step` ignores gradient accumulation
**File**: `transformer/training/experiment_runner.py:936-1046`
**Impact**: Wrong training dynamics when `grad_accumulation_steps > 1`

Loss is never divided by `grad_accumulation_steps`, and `optimizer.step()` is called every micro-batch. Compare with `FastTrainer.train_step` which correctly implements accumulation.

**Fix**: Add loss scaling and conditional optimizer step matching FastTrainer's pattern.

### C4. CUDA/Python softmax correction divergence
**File**: `transformer/pure_vfe/csrc/pairwise_kl.cu:195`
**Impact**: CUDA and Python paths produce different gradients for extreme correction values

Python path clamps correction to `[-1, 2]`:
```python
correction = ((e_kl - kl_ij) / tau).clamp(-1.0, 2.0)
```
CUDA kernel applies no clamp:
```c
float w = b_ij * (1.0f + (e_kl - kl_ij) / tau);
```

**Fix**: Add clamp to CUDA kernel.

### C5. MLP connection path violates NO NEURAL NETWORKS constraint
**File**: `transformer/core/connection.py:72-79`
**Impact**: `nn.Linear` + `nn.GELU()` in `connection_type='mlp'` path

Not active by default (default is `'bilinear'`), but code exists and is tested.

**Fix**: Remove the `'mlp'` option or clearly mark as experimental ablation.

### C6. `attention.py` output projection is a borderline violation
**File**: `transformer/core/attention.py:2550`
**Impact**: Second `nn.Linear(embed_dim, embed_dim)` beyond the single allowed K-to-vocab projection

Gated by `use_output_projection` flag. CLAUDE.md allows only "a linear output projection from K dimensions to vocabulary size."

### C7. `train_publication.py` uses argparse
**File**: `transformer/train_publication.py:752-795`
**Impact**: Direct violation of NO CLI ARGUMENTS constraint

The primary training entry point has a full `argparse.ArgumentParser`.

**Fix**: Replace with module-level config constants (which already exist).

### C8. `--use_fast_math` in CUDA compilation
**File**: `transformer/pure_vfe/cuda_ext.py:53`
**Impact**: Reduced precision in KL divergence and log-determinant computations

`--use_fast_math` enables flush-to-zero and approximate math operations. Risky for a system where catastrophic cancellation is possible in `trace + mahal - K + logdet`.

### C9. SO(3) dexp formula applied to GL(3) generators
**File**: `math_utils/transport.py:537`
**Impact**: Wrong transport differentials for GL(3) gauge groups

Gates on `K == 3` (matrix dimension) instead of checking generator count. GL(3) has K=3 but 9 generators; the SO(3) Rodrigues formula requires skew-symmetry.

**Fix**: Check `K == 3 and n_generators == 3` or verify generators are skew-symmetric.

---

## WARNING Findings

### Architecture

| ID | File | Issue |
|----|------|-------|
| W1 | `gauge_utils.py:308` vs `attention.py:1025` | Inconsistent NaN replacement: `nan=0.0` (attracts to broken pairs) vs `nan=kl_ceil` (repels). Opposite safety strategies. |
| W2 | `attention.py:1207-1213` | Redundant phi norm clamping before `stable_matrix_exp_pair()` which clamps internally |
| W3 | `model.py:491` | `out_proj` parameters wasted when PriorBank is active — receives weight decay but never used |

### Math Utilities

| ID | File | Issue |
|----|------|-------|
| W4 | `push_pull.py:194` | Eigenvalue floor 1e-4 (hard absolute) inconsistent with `sanitize_sigma` floor 1e-6*lambda_max (relative) |
| W5 | `numerical_utils.py:202-213` | `safe_inv` silently corrupts with up to 1e-4 diagonal perturbation — no warning logged |
| W6 | `push_pull.py:277` | Orthogonality check on ALL batch elements is O(batch * K^3); outer function already checks first 8 |
| W7 | `transport.py` | No NaN/Inf check on NumPy matrix exponential output |

### Training Pipeline

| ID | File | Issue |
|----|------|-------|
| W8 | `train_fast.py` vs `optimizer.py` | Dual LR scheduler implementations with different floor semantics |
| W9 | `optimizer.py:108-126` | `RiemannianAdamW._precondition_mu` silently no-ops on shape mismatch |
| W10 | `train_fast.py:231` | Perplexity overflow — no clamp on `exp(loss)` unlike PublicationTrainer |
| W11 | `train_fast.py:456` | `weights_only=False` in `torch.load` — arbitrary code execution risk |
| W12 | `config.py:201-204` | No validation: warmup > max_steps, negative LR, checkpoint_dir=None crashes |
| W13 | `experiment_runner.py:546-557` | Duplicate CSV headers (`delta_mu_norm`, `mu_norm`, `sigma_mean` each appear twice) |
| W14 | `experiment_runner.py:1418` | LR logging reads base LR, not scheduler-modulated LR — misleading metrics |

### Pure VFE

| ID | File | Issue |
|----|------|-------|
| W15 | `pairwise_kl.cu:531` | CUDA gradient kernels only support K=8; K=4 or K=16 causes hard crash |
| W16 | `pairwise_kl.cu` | All CUDA kernels assume float32; float16/bfloat16 inputs produce garbage |

---

## Hard Constraint Status

### Gauge Equivariance: CLEAN

All 10+ instances of covariance transport verified correct:
- `attention.py:903-905, 1401-1403, 1860-1862, 2203-2205` — einsum `Omega @ sigma @ Omega.T`
- `attention.py:1079` — direct `Omega @ sigma @ Omega.T`
- `attention.py:2189-2191` — diagonal: `Omega_{k,l}^2 * sigma_l` (correct diagonal of sandwich)
- `gauge_utils.py:413-415` — einsum sandwich
- `pure_vfe/gauge.py:69` — `Omega @ Sigma @ Omega.transpose(-2, -1)`
- `math_utils/push_pull.py` — documented and implemented correctly

### Timescale Separation: CLEAN

- `variational_ffn.py:3423` — `sigma_p = sigma_prior.detach().clone()`
- `prior_bank.py:436` — sigma_ce_scale controls residual gradient (default 0.01)
- `model.py:1258,1263,1274` — `sigma_prior.detach()` in implicit EM paths
- `pure_vfe/inference.py` — `@torch.no_grad()` on E-step; M-step in learning.py only

### CLI Arguments: 16 VIOLATIONS

**Primary entry point**: `train_publication.py` (argparse)

**Auxiliary files**: `scripts/gauge_frame_spectral_analysis.py` (click), `scripts/run_ablation_suite.py`, `scripts/generate_publication_figures.py`, `scripts/kn5_baseline.py`, `transformer/utils/evaluate_test_set.py`, `transformer/utils/test_query_variation.py`, `transformer/utils/evaluation.py`, `transformer/training/experiment_runner.py`, `transformer/visualization/belief_space_viz.py`, `transformer/visualization/attention_context.py`, `transformer/visualization/vfe_dynamics_plots.py`, `transformer/visualization/interactive_belief_viz.py`, `transformer/analysis/bayesian_validation.py`, `transformer/visualization/training_plots.py`, `transformer/visualization/belief_space_frequent.py`

---

## Priority Fixes

### P0 — Fix Immediately
1. **C1**: Add missing `lambda_softmax` argument in `variational_ffn.py:1485`
2. **C2**: Move `from pathlib import Path` to top of `inference.py`
3. **C3**: Implement gradient accumulation in `PublicationTrainer.train_step`
4. **C4**: Add correction clamp to CUDA kernel `pairwise_kl.cu:195`

### P1 — Fix Soon
5. **C7**: Remove argparse from `train_publication.py`
6. **C9**: Fix SO(3)/GL(3) gate in `transport.py:537`
7. **W1**: Standardize NaN replacement strategy (use `kl_ceil` everywhere)
8. **W4**: Reconcile eigenvalue floor strategies

### P2 — Address When Convenient
9. **C5**: Remove or quarantine MLP connection path
10. **C8**: Remove `--use_fast_math` from CUDA compilation
11. **W12**: Add TrainingConfig validation
12. **W13**: Fix duplicate CSV headers
13. **W14**: Log scheduled LR instead of base LR

---

## Positive Observations

The codebase demonstrates strong adherence to the core mathematical framework:

- **Gauge equivariance is perfectly maintained** — every single covariance transport uses the correct sandwich product across all code paths (nn.Module, pure VFE, NumPy, and PyTorch)
- **Timescale separation is correctly implemented** — sigma_p is consistently detached in E-step contexts with the sigma_ce_scale mechanism providing controlled gradient flow
- **Mathematical documentation is thorough** — LaTeX in docstrings, shape comments at critical points, and variable names matching paper notation
- **Test coverage for core primitives is solid** — finite-difference gradient validation, GL(K) invariance, cocycle condition, SPD retraction, E-step monotonicity
