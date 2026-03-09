# Gauge-Transformer Refactor Plan

## Overview

The current `transformer/core/` implementation is **9,622 LOC across 10 files**. The architecture is correct and working but suffers from massive code duplication, parameter forwarding bloat, and accumulated legacy code. This plan targets a **~55-60% LOC reduction** while preserving identical behavior.

**Strategy**: Build a new `transformer_v2/` directory with the refactored code. The legacy `transformer/` stays untouched.

---

## Current Problems (by severity)

### 1. KL Matrix Code Duplication (~1,500 lines → ~350 lines)
`attention.py` has **8 separate `_compute_kl_matrix_*` functions** that share ~80% identical logic:
- `_compute_kl_matrix_numba` (dead code — numba not used)
- `_compute_kl_matrix_torch` (full cov, standard)
- `_compute_kl_matrix_diagonal` (diag cov, standard)
- `_compute_kl_matrix_chunked` (full cov, chunked)
- `_compute_kl_matrix_diagonal_chunked` (diag cov, chunked)
- `_compute_kl_matrix_block_diagonal` (full cov, block-diag)
- `_compute_kl_matrix_block_diagonal_diag` (diag cov, block-diag)
- `_compute_kl_matrix_block_diagonal_chunked` (diag cov, block-diag, chunked)

Every variant repeats: transport operator application, the KL formula (trace + mahal + logdet terms), Cholesky fallback, NaN guards, and clamping.

### 2. VFE Gradient Code Duplication (~750 lines → ~250 lines)
`variational_ffn.py` has **4 gradient computation paths** with ~70% shared logic:
- `compute_vfe_gradients_gpu` inline path (full cov, standard)
- `_compute_vfe_gradients_block_diagonal` (full cov, block-diag)
- `_compute_vfe_gradients_block_diagonal_diag` (diag cov, block-diag)
- `_compute_vfe_gradients_chunked` (diag cov, chunked)

Each repeats: self-coupling gradient, transport computation, alignment gradient, softmax coupling term.

### 3. Parameter Forwarding Explosion (~300 lines → ~30 lines)
`GaugeTransformerLM.__init__` has **50+ parameters** that get forwarded:
- `model.py` → `GaugeTransformerStack` → `GaugeTransformerBlock` → `GaugeFFN` → `VariationalFFNDynamic`
- Every parameter appears in 4 different `__init__` signatures verbatim

### 4. Duplicated Phi Update Code (~150 lines → ~50 lines)
`VariationalFFNDynamic.forward()` has nearly identical phi update blocks for:
- Per-iteration phi evolution (inside VFE loop)
- Post-loop phi evolution (after VFE loop)

### 5. Legacy/Dead Code (~350 lines → 0 lines)
- `MockMultiAgentSystem` class (lines 1425-1523 in variational_ffn.py) — deprecated adapter
- `convert_torch_to_numpy_system()` (lines 1526-1589) — deprecated converter
- `_compute_kl_matrix_numba` — dead code path
- `if __name__` test blocks in attention.py, blocks.py, embeddings.py, model.py (~350 lines total)
- `so3_log_torch()` and `so3_compose_bch()` in embeddings.py (belong in gauge_utils)

### 6. Thin Wrapper Module (255 lines → 0 lines)
`ffn.py` (`GaugeFFN`) is a pure pass-through wrapper around `VariationalFFNDynamic` with no added logic. Every parameter and method call is forwarded verbatim.

---

## Target Architecture

```
transformer_v2/
├── __init__.py              # Public API
├── config.py                # ~80 lines  — GaugeTransformerConfig dataclass
├── model.py                 # ~250 lines — GaugeTransformerLM (simplified)
├── blocks.py                # ~200 lines — Block + Stack (config-driven)
├── attention.py             # ~500 lines — Unified KL attention
├── variational_ffn.py       # ~600 lines — VFE belief evolution
├── embeddings.py            # ~350 lines — Token + positional embeddings
├── prior_bank.py            # ~350 lines — Token-dependent priors (mostly unchanged)
├── gauge_utils.py           # ~150 lines — Transport operators, matrix exp, retractions
├── kl_ops.py                # ~200 lines — KL divergence primitives
├── gauge_preconditioner.py  # ~350 lines — Phi preconditioning (mostly unchanged)
└── tests/
    └── test_core.py         # Extracted from __main__ blocks
```

**Estimated total: ~3,030 lines** (vs 9,622 current = **68% reduction**)

---

## Refactor Steps

### Step 1: Create `config.py` — Config Dataclass

Replace the 50+ parameter forwarding chain with a single `GaugeTransformerConfig` dataclass:

```python
@dataclass
class GaugeTransformerConfig:
    # Architecture
    vocab_size: int
    embed_dim: int
    n_layers: int = 6
    n_heads: int = 1
    max_seq_len: int = 512

    # Gauge group
    gauge_group: str = 'SO3'          # 'SO3' | 'SON' | 'GLK'
    phi_dim: int = 3
    irrep_dims: Optional[List[int]] = None

    # VFE parameters
    alpha: float = 0.001              # Prior coupling
    kappa: float = 1.0                # Softmax temperature
    n_vfe_iterations: int = 1         # E-step iterations
    learnable_lr: bool = True
    lambda_belief: float = 1.0
    update_sigma: bool = True
    diagonal_covariance: bool = False

    # Phi evolution
    update_phi: bool = False
    update_phi_per_iteration: bool = False
    phi_lr: float = 0.05
    phi_max_norm: float = 3.14159
    phi_natural_gradient: str = 'clip'

    # Memory efficiency
    chunk_size: Optional[int] = None

    # Features
    mask_self_attention: bool = False
    learnable_alpha: bool = False
    multihead_vfe: bool = False
    per_head_kappa: bool = False
    position_encoding: str = 'gauge'  # 'gauge' | 'rope' | 'alibi'

    # Pure FEP mode
    pure_fep_mode: bool = False
    use_prior_bank: bool = False
    prior_lr: float = 0.01
    gauge_fixed_priors: bool = False

    # Training
    dropout: float = 0.1
    tie_weights: bool = True
```

Every module receives the single `config` object. No more 4-level parameter forwarding.

### Step 2: Create `kl_ops.py` — Unified KL Primitives

Extract the shared math from all 8 KL functions into composable primitives:

```python
def kl_divergence_gaussian(
    mu_q, sigma_q,       # Query beliefs
    mu_p, sigma_p,       # Prior/transported beliefs
    covariance_type: str, # 'full' | 'diagonal' | 'block_diagonal'
    block_dims: Optional[List[int]] = None,
) -> torch.Tensor:
    """Unified KL(q || p) for all covariance types."""

def transport_beliefs(
    mu, sigma, phi,
    exp_pos, exp_neg,
    covariance_type: str,
    block_dims: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply gauge transport Ω_ij to beliefs."""

def kl_matrix(
    mu, sigma, phi, generators,
    covariance_type: str,
    block_dims: Optional[List[int]] = None,
    chunk_size: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute full N×N KL matrix — dispatches chunking internally."""
```

The key insight: `covariance_type` + `block_dims` + `chunk_size` replace 8 separate functions. Chunking is handled as an **inner loop** detail, not a separate code path.

For block-diagonal mode, a single loop over blocks replaces dedicated functions:
```python
if covariance_type == 'block_diagonal':
    kl = torch.zeros(B, N, N)
    offset = 0
    for d in block_dims:
        kl += kl_divergence_gaussian(
            mu_q[..., offset:offset+d], sigma_q_block,
            mu_p[..., offset:offset+d], sigma_p_block,
            covariance_type='full',  # Each block is full-cov
        )
        offset += d
```

### Step 3: Refactor `attention.py` — Unified KL Attention

The current 2,795-line file becomes ~500 lines by:

1. **Delete** all 8 `_compute_kl_matrix_*` functions → replaced by `kl_ops.kl_matrix()`
2. **Delete** `_compute_kl_matrix_numba` (dead code)
3. **Simplify** `compute_attention_weights()` — it currently has complex dispatch logic that selects between the 8 variants; this becomes a single call to `kl_ops.kl_matrix()`
4. **Simplify** `IrrepMultiHeadAttention.__init__` — takes `config` instead of 50+ params
5. **Delete** `if __name__` block (90+ lines)
6. **Keep** `aggregate_messages()`, `compute_transport_operators()` (already clean)
7. **Move** RoPE helpers to `gauge_utils.py` or keep inline (they're small)

### Step 4: Refactor `variational_ffn.py` — Unified VFE

The current 2,685-line file becomes ~600 lines by:

1. **Unify gradient computation**: All 4 gradient paths become one function using `kl_ops` primitives:
   ```python
   def compute_vfe_gradients(
       mu, sigma, phi, mu_prior, beta, generators, config,
   ) -> Tuple[torch.Tensor, ...]:
       """Single gradient function for all covariance/memory modes."""
   ```
   The function uses `config.diagonal_covariance`, `config.irrep_dims`, and `config.chunk_size` to select the right computation path internally, but the shared logic (self-coupling, transport, alignment) is written once.

2. **Extract phi update** into a single helper:
   ```python
   def _update_phi(phi, grad_phi, config, generators, preconditioner):
       """Unified phi update with retraction. Used in both per-iteration and post-loop."""
   ```
   Called from two sites with different gradients, eliminating the ~100 lines of duplication.

3. **Delete** `MockMultiAgentSystem` and `convert_torch_to_numpy_system()` (~165 lines)

4. **Simplify** `VariationalFFNDynamic.__init__` — takes `config` instead of 30+ params

5. **Move** `_retract_phi()` and `retract_spd_*()` to `gauge_utils.py`

### Step 5: Expand `gauge_utils.py` — Shared Geometry Utilities

Current `gauge_utils.py` is only 71 lines (just `stable_matrix_exp_pair`). Expand to ~150 lines:

- `stable_matrix_exp_pair()` (existing)
- `retract_spd()` — extracted from variational_ffn.py (handles both full and diagonal)
- `retract_phi()` — extracted from variational_ffn.py
- `safe_spd_inv()` — extracted from variational_ffn.py
- `so3_log()` — extracted from embeddings.py
- `so3_compose_bch()` — extracted from embeddings.py
- `compute_transport_operators()` — possibly extract from attention.py

### Step 6: Simplify `blocks.py`

The current 795 lines (with 90-line `__main__` block) become ~200 lines:

- `GaugeTransformerBlock.__init__` takes `config` instead of 50+ params
- `GaugeTransformerStack.__init__` takes `config` — no more duplicating the entire parameter list
- Delete `if __name__` block
- The forward pass logic is already clean — keep it

### Step 7: Eliminate `ffn.py` Wrapper

`GaugeFFN` adds zero value — it's a pure pass-through. In the refactored code, `GaugeTransformerBlock` directly instantiates `VariationalFFNDynamic`. The `create_ffn()` factory is also unnecessary with config-driven construction.

### Step 8: Simplify `model.py`

Current 1,302 lines → ~250 lines:
- Takes `GaugeTransformerConfig` instead of 50+ constructor params
- Generator creation logic stays (it's model-level responsibility)
- Delete `if __name__` block (~100 lines)
- Simplify the long parameter forwarding in `__init__`

### Step 9: Clean `embeddings.py`

Current 875 lines → ~350 lines:
- Move `so3_log_torch()`, `so3_compose_bch()` to `gauge_utils.py`
- Takes `config` instead of many params
- Delete `if __name__` block (~80 lines)
- `update_embeddings_from_beliefs()` stays (it's core functionality)

### Step 10: Extract Tests

Collect all `if __name__` blocks into `tests/test_core.py`:
- attention.py test block → `test_kl_attention()`
- blocks.py test block → `test_transformer_block()`
- embeddings.py test block → `test_embeddings()`
- model.py test block → `test_model()`

---

## LOC Budget

| Module | Current | Refactored | Reduction |
|--------|---------|------------|-----------|
| config.py | 0 | 80 | (new) |
| attention.py | 2,795 | 500 | -82% |
| variational_ffn.py | 2,685 | 600 | -78% |
| model.py | 1,302 | 250 | -81% |
| blocks.py | 795 | 200 | -75% |
| embeddings.py | 875 | 350 | -60% |
| ffn.py | 255 | 0 | -100% |
| prior_bank.py | 393 | 350 | -11% |
| gauge_utils.py | 71 | 150 | +111% |
| gauge_preconditioner.py | 392 | 350 | -11% |
| kl_ops.py | 0 | 200 | (new) |
| __init__.py | 59 | 40 | -32% |
| tests/test_core.py | 0 | 200 | (new) |
| **Total** | **9,622** | **3,270** | **-66%** |

---

## Implementation Order

The dependencies dictate this order:

1. **`config.py`** — no dependencies, needed by everything
2. **`gauge_utils.py`** — no dependencies, shared geometry primitives
3. **`kl_ops.py`** — depends on gauge_utils, needed by attention + VFE
4. **`gauge_preconditioner.py`** — mostly copy, minor cleanup
5. **`prior_bank.py`** — mostly copy, add config support
6. **`embeddings.py`** — depends on gauge_utils, config
7. **`attention.py`** — depends on kl_ops, gauge_utils, config
8. **`variational_ffn.py`** — depends on kl_ops, gauge_utils, attention, config
9. **`blocks.py`** — depends on attention, variational_ffn, config
10. **`model.py`** — depends on everything
11. **`tests/test_core.py`** — depends on model
12. **`__init__.py`** — public API surface

---

## Validation Strategy

After each module is written, verify:
1. **Numerical equivalence**: Same inputs → same outputs (within float tolerance) as legacy code
2. **Config equivalence**: `GaugeTransformerConfig(...)` produces identical model to legacy `GaugeTransformerLM(...)` with the same params
3. **End-to-end**: Load legacy weights into refactored model, confirm identical forward pass
4. **Training**: Run a short training loop and verify loss curves match

---

## What NOT to Change

- **Mathematical formulas**: All KL, VFE, retraction math stays identical
- **Generator construction** in `math_utils/generators.py`: Out of scope (works fine, just large)
- **Agent/meta/gradients** directories: Out of scope
- **Legacy `transformer/`**: Left completely untouched
- **`simulation_runner.py`** and study scripts: Out of scope (separate cleanup task)
