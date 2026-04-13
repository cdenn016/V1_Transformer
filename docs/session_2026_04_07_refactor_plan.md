# Deep Refactor Plan — 2026-04-07

This document captures two pieces of work:
1. **Completed**: drift hazard #29 fix — `blocks.GaugeTransformerBlock.forward` and the
   inline per-layer body in `model.GaugeTransformerLM.forward_with_attention` have been
   consolidated. The two paths now go through a single implementation.
2. **Planned**: extraction of the "side-quest" (opt-in, research-variant) code paths
   from `variational_ffn.py` into dedicated helper modules, and extraction of the
   Hebbian / P-flow / delta-rule learning path into its own module.

## Part 1: Drift hazard #29 — DONE

### What changed

**`transformer/core/blocks.py`**
- `GaugeTransformerBlock.forward` gained a keyword-only `return_attention: bool = False`
  parameter. When `False` (default), it returns the original 3-tuple
  `(mu_q, sigma_q, phi)`. When `True`, it returns a 5-tuple
  `(mu_q, sigma_q, phi, beta, kl)` where `beta` and `kl` are the per-head attention
  weights and KL matrices from the attention sublayer (or `None` when
  `skip_attention=True`).
- The block now stores `self._last_mu_attn` and `self._last_mu_ffn` as transient
  attributes so the model-level diagnostics code (per-layer norms, residual ratios)
  can read them without duplicating the attention/FFN computation.
- The docstring DRIFT HAZARD warning introduced in round 3 is removed — the duplication
  no longer exists.

**`transformer/core/model.py`**
- The inline per-layer body inside `forward_with_attention` (previously ~140 lines
  duplicating the block forward) is replaced by a single
  `block(..., return_attention=True)` call that unpacks the 5-tuple.
- The `_w_out_fwa` computation (cross-head permuted W_out for the final layer) stays
  in `forward_with_attention` as model-level bookkeeping before the block call.
- Per-layer diagnostics read `block._last_mu_attn` / `block._last_mu_ffn` instead of
  inline variables.
- The DRIFT HAZARD comment introduced in round 3 is removed.
- The omega-mode branch that builds per-head cached transports from omega blocks
  (present in `block.forward` but missing from the inline training copy before the
  refactor) now runs correctly during training because training goes through
  `block.forward`. Training was silently missing this branch for `gauge_param='omega'`
  configs; the consolidated path fixes that too.

### Verification

1. Both files parse cleanly (`ast.parse`).
2. No instance of `mu_attn, sigma_attn, beta, kl = block.attention(...)` remains in
   `model.py` (the duplicate pattern is gone).
3. A runtime smoke test builds a 2-layer SO(3) model with `embed_dim=7`,
   `irrep_spec=[('l0', 4, 1), ('l1', 1, 3)]`, `implicit_em=False`, and runs both
   `model(tokens)` (inference) and `model.forward_with_attention(tokens)` (training)
   on the same input. The logits are **bitwise identical**
   (`max |logits_inf − logits_train| = 0.00e+00`).
4. Existing test suite: 33 passed, 1 deselected (`test_model_config_stored`, a
   pre-existing failure unrelated to the refactor — it checks `model.config ==
   minimal_config` but the model injects `gauge_param` into the stored dict).

---

## Part 2: Side-quest extraction — PLANNED

### Current state of `variational_ffn.py`

At 3,786 lines, `variational_ffn.py` is the largest file in the core package and
contains a mix of concerns:

| Line range  | Content                                             | ~Lines |
|------------:|-----------------------------------------------------|-------:|
| 1–103       | Imports, module-level constants, recorder helpers   |    103 |
| 104–280     | `ImplicitEMGradient`, `ImplicitEMGradientSigma`, `compute_implicit_em_scales` |    176 |
| 283–441     | `DEQFixedPoint`, `DEQFixedPointFull`                 |    158 |
| 448–874     | `VariationalFFNDynamic.__init__`                    |    396 |
| 876–1018    | `lr` property, `_get_kappa_h`, `_get_sigma_trust`, `get_bayesian_alpha`, `_precondition_phi_grad` |    142 |
| 1019–1128   | `_compute_omega_grad_direct`                        |    109 |
| 1129–1223   | `_retract_omega`                                    |     94 |
| 1224–1322   | `_compute_phi_grad`                                 |     98 |
| **1324–1480** | **`_make_deq_step_fn` — DEQ closure (μ, Σ only)** |  **156** |
| **1481–1743** | **`_make_deq_step_fn_with_phi` — DEQ closure with φ** |  **262** |
| 1744–1828   | `_build_block_exp_pairs`                            |     84 |
| 1829–1887   | `_finalize_e_step`                                  |     58 |
| 1888–2006   | `_prepare_e_step_inputs`                            |    118 |
| **2007–2515** | **`_closed_form_e_step` — diag + full-cov + Picard** |  **508** |
| 2516–3475   | `_vfe_iteration` (main per-iteration body)          |    959 |
| 3476–3782   | `forward` (top-level orchestration)                 |    306 |
| 3783–       | `extra_repr`                                        |      3 |
| **Total**   |                                                     | **3,786** |

The **bold rows** are side-quest code paths that are opt-in and not exercised by the
default configuration. Together they account for **~1,260 lines**, about 33% of the
file.

### Side-quest inventory

#### Side quest A: **Implicit EM** (`implicit_em=True`)
- **What it does**: Replaces the straight-through M-step gradient (`∂L/∂θ ≈
  ∂L/∂z*`) with the information-geometrically correct IFT scale
  `s_k = (α/σ²_p) / (α/σ²_p + Σ β/σ²_q) ∈ [0, 1]` computed analytically from the
  fixed-point values.
- **Code**:
  - `variational_ffn.py:104-280` — `ImplicitEMGradient`, `ImplicitEMGradientSigma`,
    `compute_implicit_em_scales` — 176 lines
  - Called from `_finalize_e_step` (stores `_last_implicit_mu_scale` /
    `_last_implicit_sigma_scale`)
  - Consumed by `model.forward_with_attention:1357-1359` — applies the IFT scale
    via `ImplicitEMGradient.apply(mu_q.detach(), mu_prior, implicit_mu_scale)`
- **Dependencies**: None on `_vfe_iteration`. The two autograd Functions are pure
  (no `self`, no FFN internals).
- **Consumers**: `variational_ffn._finalize_e_step`, `model.forward_with_attention`
- **Mutual exclusions**: `implicit_em=True + use_deq=True` already raises at init
  (`variational_ffn.py:831-837`).
- **Extraction complexity**: **LOW**. Self-contained. Pure autograd Functions + one
  helper.

#### Side quest B: **DEQ (Deep Equilibrium)** (`use_deq=True`)
- **What it does**: Replaces the straight-through backward through the iterative
  E-step with a Neumann-series approximation of `(I − J_T)^{−1}`, where `J_T` is the
  Jacobian of one E-step iteration at the fixed point. Gives the correct
  implicit-function-theorem gradient for the M-step instead of the I-approximation.
- **Code**:
  - `variational_ffn.py:283-441` — `DEQFixedPoint`, `DEQFixedPointFull` (autograd
    Functions with Neumann loop) — 158 lines
  - `variational_ffn.py:1324-1480` — `_make_deq_step_fn` (μ, Σ closure) — 156 lines
  - `variational_ffn.py:1481-1743` — `_make_deq_step_fn_with_phi` (joint closure,
    includes a differentiable phi retraction) — 262 lines
  - Called from `variational_ffn.forward:3614-3648` (the `self.use_deq and
    self.training and torch.is_grad_enabled()` branch)
- **Dependencies**: The step functions need `self.irrep_dims`, `self.gauge_mode`,
  `self.multihead_vfe`, `self.lambda_belief`, `self.lambda_softmax`, `self.kappa`,
  `self._get_kappa_h`, `self._generators_are_skew`, `self.generators`,
  `self.get_bayesian_alpha`, `self.learnable_alpha`, `self.raw_c0`,
  `self.compute_sigma_align_grad`, `self.exact_diagonal_transport`,
  `self.mask_self_attention`, `self.update_sigma`, `self.sigma_max`, `self.lr`,
  `self.phi_lr`, `self._get_sigma_trust`.
- **Consumers**: `variational_ffn.forward`
- **Mutual exclusions**:
  - `use_deq=True + implicit_em=True` → ValueError at init
  - `use_deq=True + active_inference=True` → ValueError in `wire_readout_references`
  - `use_deq=True + closed_form_e_step=True` → undefined (probably broken — DEQ
    requires an iterative E-step; closed-form bypasses the loop)
- **Extraction complexity**: **MEDIUM**. The autograd Functions extract cleanly; the
  step-function factories need `ffn` as a first argument to access attributes.

#### Side quest C: **Closed-form E-step** (`closed_form_e_step=True`)
- **What it does**: Instead of running gradient descent in the E-step, compute
  the precision-weighted fixed point analytically by solving a linear system
  `A μ* = b` where `A = α Σ_p^{−1} + λ Σ_j β_ij Σ_j_t^{−1}` and
  `b = α Σ_p^{−1} μ_p + λ Σ_j β_ij Σ_j_t^{−1} Ω_ij μ_j`.
- **Code**:
  - `variational_ffn.py:2007-2515` — `_closed_form_e_step` method — 508 lines
    - Lines 2030-2150: diagonal covariance branch with full softmax-coupling
      correction
    - Lines 2157-2247: full-covariance branch via Cholesky solve (no softmax
      coupling; documented simplification)
    - Lines 2251-2515: **Picard resolve loop** nested inside the closed-form branch
      (two separate sub-loops for diagonal and full-cov)
  - Called from `variational_ffn.forward:3579-3595` (the `self.closed_form_e_step`
    branch)
- **Dependencies**: Very heavy. Uses `self.irrep_dims`, `self._build_block_exp_pairs`,
  `self._get_kappa_h`, `self.kappa`, `self.lambda_belief`, `self.lambda_softmax`,
  `self.n_picard_steps`, `self.update_sigma`, `self.isotropic_covariance`,
  `self.sigma_max`, `self.mask_self_attention`, `self.gauge_mode`,
  `self._use_rope_vfe`, `self._rope_base_vfe`, `self.learnable_alpha`,
  `self.get_bayesian_alpha`, `self.embed_dim`.
- **Consumers**: `variational_ffn.forward` only.
- **Mutual exclusions**:
  - `closed_form_e_step=True + active_inference=True` → ValueError in
    `wire_readout_references`
  - `n_picard_steps > 0` requires `closed_form_e_step=True` (raised in
    `BlockConfig.__post_init__` after round-6 Fix #41)
- **Extraction complexity**: **HIGH**. The method is 500 lines. Picard is nested
  inside and reuses intermediates. The extraction should flatten this into a
  dispatcher plus four sub-functions (diag, full, picard-diag, picard-full).

#### Side quest D: **Picard resolve** (`n_picard_steps > 0`)
- **What it does**: Iterative correction of the closed-form solution to account
  for the softmax coupling nonlinearity that the linear fixed point omits.
- **Code**: Nested inside `_closed_form_e_step` at `variational_ffn.py:2251-2515`
  (two loops, one for diagonal and one for full covariance).
- **Dependencies**: Same as closed-form plus `self.picard_trust_region`.
- **Consumers**: Only inside `_closed_form_e_step`.
- **Mutual exclusions**: Requires `closed_form_e_step=True`.
- **Extraction complexity**: **MEDIUM**. Extracts together with the closed-form
  branch. Two sub-functions.

#### Side quest E: **Hebbian / P-flow / delta rule** (`use_p_flow=True`,
`use_delta_rule_w_out=True`, `detach_phi=True`)
- **What it does**: Backprop-free learning. After the backward pass, update token
  embeddings via an EMA toward successful beliefs (P-flow), update gauge frames via
  an EMA toward VFE-evolved values (φ P-flow), and update the output projection via
  the Widrow-Hoff delta rule instead of gradient descent.
- **Code** is spread across five files:
  - `model.py:1662-1830` — `p_flow_update`, `phi_flow_update`,
    `delta_rule_update_w_out` dispatch methods (~170 lines)
  - `embeddings.py:608-786` — `_compute_pflow_weights`,
    `update_embeddings_from_beliefs`, `update_phi_from_beliefs` (~180 lines)
  - `prior_bank.py:564-742` — `update_from_beliefs`, `_update_gauge_fixed_base_prior`
    (~180 lines)
  - `training/experiment_runner.py:1095-1144` —
    `_apply_p_flow_and_delta_rule` (~50 lines)
  - Configuration flags in `block_config.py`, `train_publication.py`
- **Dependencies**: Accesses `model.token_embed`, `model.prior_bank`,
  `model.out_proj`, and `block.ffn._last_beta_for_implicit` in places.
- **Consumers**: `experiment_runner.PublicationTrainer.train_step` post-backward.
- **Mutual exclusions**: None explicit; runs independently of DEQ / implicit_em /
  closed_form.
- **Extraction complexity**: **HIGH**. Spans five files and doesn't live in
  `variational_ffn.py` at all. Extraction is orthogonal to the VFE refactor.

### Proposed target structure

```
transformer/core/
├── variational_ffn.py                (~2,500 lines, down from 3,786)
│   └── VariationalFFNDynamic
│       ├── __init__
│       ├── _vfe_iteration
│       ├── _prepare_e_step_inputs
│       ├── _finalize_e_step
│       ├── forward (dispatches to closed-form / iterative / DEQ via helpers)
│       ├── get_bayesian_alpha
│       ├── _precondition_phi_grad
│       ├── _compute_phi_grad
│       ├── _compute_omega_grad_direct
│       ├── _retract_omega
│       ├── _build_block_exp_pairs
│       └── small helpers (_get_kappa_h, _get_sigma_trust, etc.)
│
├── active_inference.py               (EXISTING — already extracted)
│
├── vfe_implicit_em.py                (NEW, ~180 lines)
│   ├── ImplicitEMGradient            (autograd Function)
│   ├── ImplicitEMGradientSigma       (autograd Function)
│   └── compute_implicit_em_scales(α, sigma_p, beta, sigma_q, eps) -> (mu_scale, sigma_scale)
│
├── vfe_deq.py                        (NEW, ~600 lines)
│   ├── DEQFixedPoint                 (autograd Function, μ/Σ only)
│   ├── DEQFixedPointFull             (autograd Function, joint μ/Σ/φ)
│   ├── _DEQ_VJP_NORM_CAP             (constant)
│   ├── make_deq_step_fn(ffn, phi_current, mu_p_current, sigma_p, mask, is_diagonal, eps, dtype)
│   │     -> step_fn closure           (takes ffn as first arg)
│   └── make_deq_step_fn_with_phi(ffn, mu_p_current, sigma_p, mask, is_diagonal, eps, dtype)
│         -> step_fn closure           (takes ffn as first arg)
│
└── vfe_closed_form.py                (NEW, ~550 lines)
    ├── run_closed_form_e_step(ffn, state, mask, return_beta_history)
    │     -> (mu, sigma, phi, omega, beta_heads, beta_history)
    │     dispatches to diagonal or full-cov, then optionally Picard
    ├── _cf_diagonal(ffn, state, cf_bep, mask)
    ├── _cf_full_cov(ffn, state, cf_bep, mask)
    ├── _picard_resolve_diagonal(ffn, state, cf_bep, mask)
    └── _picard_resolve_full_cov(ffn, state, cf_bep, mask)

transformer/learning/                 (NEW DIRECTORY)
└── hebbian.py                        (~550 lines)
    ├── apply_p_flow_and_delta_rule(trainer, input_ids, target_ids,
    │                                full_metrics, is_standard, use_delta_rule)
    │     (moved from experiment_runner._apply_p_flow_and_delta_rule)
    ├── p_flow_update(model, token_ids, mu_beliefs, sigma_beliefs,
    │                 prediction_errors, ema_decay, pad_token_id=-100)
    │     (moved from GaugeTransformerLM.p_flow_update)
    ├── phi_flow_update(model, token_ids, phi_evolved,
    │                   prediction_errors, ema_decay, pad_token_id=-100)
    │     (moved from GaugeTransformerLM.phi_flow_update)
    ├── delta_rule_update_w_out(model, mu_beliefs, targets, lr, pad_token_id=-100)
    │     (moved from GaugeTransformerLM.delta_rule_update_w_out)
    └── _hebbian EMA helpers
          (moved from PriorBank.update_from_beliefs / _update_gauge_fixed_base_prior
           and GaugeTokenEmbedding.update_embeddings_from_beliefs /
           update_phi_from_beliefs / _compute_pflow_weights)
```

### Extraction pattern (applies to every side quest)

1. **Create the new module file** with a focused docstring describing the side-quest.
2. **Copy the code verbatim** from the source file(s).
3. **Convert instance methods to free functions** that take the relevant object
   (FFN, model, trainer) as the first argument. This is the same pattern
   `active_inference.py` uses for `compute_ai_gradients(ffn, ...)`.
4. **Update the source file** to import the free functions and replace the method
   body with a thin delegation:
   ```python
   # variational_ffn.py
   from transformer.core.vfe_closed_form import run_closed_form_e_step

   def forward(self, mu, ...):
       ...
       if self.closed_form_e_step:
           (mu_current, sigma_current, phi_current, omega_current,
            beta_heads, _cf_beta_history) = run_closed_form_e_step(
               self, state, mask, return_beta_history,
           )
           ...
   ```
5. **Preserve backward compatibility** by re-exporting moved symbols from their
   original location:
   ```python
   # variational_ffn.py
   from transformer.core.vfe_implicit_em import (
       ImplicitEMGradient,
       ImplicitEMGradientSigma,
       compute_implicit_em_scales,
   )
   __all__ = [..., 'ImplicitEMGradient', 'ImplicitEMGradientSigma',
              'compute_implicit_em_scales']
   ```
   This matters because `model.forward_with_attention` already imports
   `ImplicitEMGradient` from `variational_ffn`, and any external code that does
   the same would break without the re-export.
6. **Run syntax + runtime smoke test** on each extraction.
7. **Commit as a single atomic change** per side quest so bisecting is possible.

### Ordering

Extract in this order (smallest-to-largest, most-independent-first):

1. **Implicit EM** (176 lines, no FFN dependencies, pure autograd Functions).
   Lowest risk, highest confidence. Do first.
2. **DEQ** (576 lines across two step-function factories + two autograd Functions).
   Medium risk. The step-function factories need `ffn` as first arg to access
   ~20 instance attributes. Requires a careful signature.
3. **Closed-form E-step + Picard** (508 lines, one method). Largest single
   extraction. The method body is already well-structured with separate diagonal
   and full-cov branches; flatten these into sub-functions.
4. **Hebbian / P-flow / delta rule** (~550 lines spread across five files).
   Highest risk because it touches five files and doesn't live in
   `variational_ffn.py` at all. Do last, treat as a separate refactor.

### Dependency constraints to preserve

All the mutual-exclusion rules established by earlier audit rounds must continue
to be enforced in the extracted modules:

| Combination                                    | Enforcement                                       |
|:-----------------------------------------------|:--------------------------------------------------|
| `implicit_em + use_deq`                        | ValueError in `variational_ffn.__init__` (line 831) |
| `active_inference + use_deq`                   | ValueError in `wire_readout_references`            |
| `active_inference + closed_form_e_step`        | ValueError in `wire_readout_references`            |
| `n_picard_steps > 0` without `closed_form_e_step` | ValueError in `BlockConfig.__post_init__` (round 6) |
| `learnable_reflection + use_prior_bank`        | ValueError in `_resolve_gauge_mode`                |

After the refactor each extracted module should re-raise these same errors (or rely
on the caller's pre-check) and include a comment pointing at the enforcement site.

### Interface design for `run_closed_form_e_step`

The current `_closed_form_e_step` method takes 14 arguments:
```python
def _closed_form_e_step(
    self, mu_current, sigma_current, phi_current, omega_current,
    mu_p_current, sigma_p, alpha_effective, _alpha_c0,
    is_diagonal, B, N, device, dtype, eps, mask, return_beta_history,
) -> Tuple[...]:
```

This is ugly. After extraction, bundle the state into a single dict (matching the
pattern `_prepare_e_step_inputs` already uses):

```python
def run_closed_form_e_step(
    ffn: "VariationalFFNDynamic",
    state: dict,              # output of ffn._prepare_e_step_inputs
    mask: Optional[torch.Tensor],
    return_beta_history: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor,
           Optional[torch.Tensor], list, Optional[list]]:
    """..."""
    mu_current = state['mu_current']
    sigma_current = state['sigma_current']
    phi_current = state['phi_current']
    omega_current = state['omega_current']
    mu_p_current = state['mu_p_current']
    sigma_p = state['sigma_p']
    alpha_effective = state['alpha_effective']
    _alpha_c0 = state['_alpha_c0']
    is_diagonal = state['is_diagonal']
    B, N = state['B'], state['N']   # add these to _prepare_e_step_inputs
    device = ffn.generators.device
    dtype = mu_current.dtype
    eps = 1e-6

    # ... dispatch to _cf_diagonal or _cf_full_cov ...
```

This reduces the call site in `variational_ffn.forward` from:
```python
(mu_current, sigma_current, phi_current, omega_current,
 beta_heads, _cf_beta_history) = self._closed_form_e_step(
    mu_current=mu_current,
    sigma_current=sigma_current,
    phi_current=phi_current,
    omega_current=omega_current,
    mu_p_current=mu_p_current,
    sigma_p=sigma_p,
    alpha_effective=alpha_effective,
    _alpha_c0=_alpha_c0,
    is_diagonal=is_diagonal,
    B=B, N=N,
    device=device, dtype=dtype, eps=eps,
    mask=mask,
    return_beta_history=return_beta_history,
)
```
to:
```python
(mu_current, sigma_current, phi_current, omega_current,
 beta_heads, _cf_beta_history) = run_closed_form_e_step(
    self, _state, mask, return_beta_history,
)
```

### Interface design for DEQ step-function factories

Current:
```python
def _make_deq_step_fn(self, phi_current, mu_p_current, sigma_p,
                      mask, is_diagonal, eps, dtype):
    def step_fn(mu_in, sigma_in):
        # uses self.irrep_dims, self.gauge_mode, self.multihead_vfe, ...
        ...
    return step_fn
```

After extraction:
```python
# vfe_deq.py
def make_deq_step_fn(
    ffn: "VariationalFFNDynamic",
    phi_current: torch.Tensor,
    mu_p_current: torch.Tensor,
    sigma_p: torch.Tensor,
    mask: Optional[torch.Tensor],
    is_diagonal: bool,
    eps: float,
    dtype: torch.dtype,
):
    def step_fn(mu_in: torch.Tensor, sigma_in: torch.Tensor):
        # uses ffn.irrep_dims, ffn.gauge_mode, ffn.multihead_vfe, ...
        ...
    return step_fn
```

The closure captures `ffn` by reference, so every attribute access becomes
`ffn.attr` instead of `self.attr`. This is the same pattern `active_inference.py`
uses for `_compute_active_inference_gradient(mu_current, sigma_current, prior_bank,
pragmatic_weight, ...)` where state is passed explicitly.

Call site in `variational_ffn.forward` changes from:
```python
step_fn = self._make_deq_step_fn(phi_current, mu_p_current, sigma_p,
                                  mask, is_diagonal, eps, dtype)
```
to:
```python
from transformer.core.vfe_deq import make_deq_step_fn
step_fn = make_deq_step_fn(self, phi_current, mu_p_current, sigma_p,
                           mask, is_diagonal, eps, dtype)
```

### Risks and mitigations

**Risk 1 — Cyclic imports**. `vfe_closed_form.py` and `vfe_deq.py` need a
type-annotated reference to `VariationalFFNDynamic`, but `variational_ffn.py`
imports from them. Solve with `TYPE_CHECKING` guards:
```python
# vfe_closed_form.py
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from transformer.core.variational_ffn import VariationalFFNDynamic
```
and type-annotate as `"VariationalFFNDynamic"` (string forward reference).

**Risk 2 — Checkpoint compatibility**. `ImplicitEMGradient` and `DEQFixedPoint` are
`torch.autograd.Function` subclasses. PyTorch does not serialize them in
`state_dict()`, so moving them to a new module will not break checkpoints.
However, if any code imports `from variational_ffn import ImplicitEMGradient`
(which `model.py` does at line 40-ish), that import must continue to work via
the re-export pattern.

**Risk 3 — Circular method-dependency on `_get_kappa_h`, `_build_block_exp_pairs`,
`get_bayesian_alpha`**. The extracted functions will need to call these. Since they
take `ffn` as the first argument and these are instance methods on the FFN,
the calls become `ffn._get_kappa_h(...)`, `ffn._build_block_exp_pairs(...)`,
`ffn.get_bayesian_alpha(...)`. This works but creates a coupling: the helper
module is tied to the specific method names on the FFN. Document this dependency
in each helper's docstring.

**Risk 4 — The Picard resolve loop is deeply nested and reuses intermediates**.
The diagonal Picard branch at `variational_ffn.py:2253+` reuses `mu_p_current`,
`sigma_p`, `phi_current`, `_cf_bep`, `is_diagonal`, `alpha_effective`, etc. from
the enclosing scope. When extracting, bundle these into the `state` dict or pass
explicitly. A naive extraction risks shadowing bugs if any variable name is
misspelled.

**Risk 5 — The `_ffn_implicit_em_state` checkpoint save/restore** (round 3 Fix #24)
saves `_last_alpha_i`, `_last_beta_for_implicit`, `_last_implicit_mu_scale`,
`_last_implicit_sigma_scale` as transient attributes on the FFN. These stay where
they are; only `compute_implicit_em_scales` (the helper function) moves.

**Risk 6 — Performance regression from function-call overhead**. Closed-form E-step
is called once per forward pass; DEQ backward is called once per backward. Neither
is a hot loop. The refactor adds one extra function call per E-step, which is
negligible. The iterative E-step (hot loop in `_vfe_iteration`) is not being
changed, so the ~959-line main body keeps running as-is.

### Testing strategy (per extraction)

After each extraction:

```bash
# 1. Syntax check
python -c "
import ast
for f in ['transformer/core/variational_ffn.py',
          'transformer/core/vfe_implicit_em.py',
          'transformer/core/vfe_deq.py',
          'transformer/core/vfe_closed_form.py']:
    ast.parse(open(f, encoding='utf-8').read())
    print(f'OK  {f}')
"

# 2. Import check (catches cyclic imports and missing re-exports)
python -c "
from transformer.core.variational_ffn import (
    VariationalFFNDynamic,
    ImplicitEMGradient, ImplicitEMGradientSigma, compute_implicit_em_scales,
    DEQFixedPoint, DEQFixedPointFull,
)
from transformer.core.vfe_implicit_em import ImplicitEMGradient as IEM
from transformer.core.vfe_deq import make_deq_step_fn
from transformer.core.vfe_closed_form import run_closed_form_e_step
print('all imports resolve')
"

# 3. Default-config forward pass (regression test for the core path — no side
#    quests enabled, so none of the extracted code runs)
python -c "
import torch
from transformer.core.model import GaugeTransformerLM
config = {'vocab_size': 50, 'embed_dim': 7, 'n_layers': 2,
          'irrep_spec': [('l0', 4, 1), ('l1', 1, 3)],
          'hidden_dim': 14, 'max_seq_len': 16, 'kappa_beta': 1.0,
          'gauge_group': 'SO3', 'use_rope': False}
torch.manual_seed(0)
m = GaugeTransformerLM(config)
m.eval()
tok = torch.randint(0, 50, (2, 8))
with torch.no_grad():
    logits = m(tok)
print(f'forward OK, logits shape: {logits.shape}')
"

# 4. Each-side-quest config forward pass (regression test for the extracted code)
python -c "
import torch
from transformer.core.model import GaugeTransformerLM
base = {'vocab_size': 50, 'embed_dim': 7, 'n_layers': 2,
        'irrep_spec': [('l0', 4, 1), ('l1', 1, 3)],
        'hidden_dim': 14, 'max_seq_len': 16, 'kappa_beta': 1.0,
        'gauge_group': 'SO3', 'use_rope': False}
torch.manual_seed(0)

for name, extra in [
    ('default', {}),
    ('implicit_em', {'implicit_em': True}),
    ('closed_form', {'closed_form_e_step': True}),
    ('closed_form+picard', {'closed_form_e_step': True, 'n_picard_steps': 2}),
    # use_deq needs training mode and backward; skip in smoke test
]:
    cfg = {**base, **extra}
    m = GaugeTransformerLM(cfg)
    m.eval()
    tok = torch.randint(0, 50, (2, 8))
    with torch.no_grad():
        logits = m(tok)
    print(f'  {name}: logits shape {logits.shape} OK')
"
```

Each extraction should be followed by these checks *before* committing. The
`default` config test is the critical regression check — the core path must
remain bitwise-identical after the refactor.

### Differential validation (optional but recommended)

For each extraction, run the pre-extraction and post-extraction versions on the
same input and verify the outputs are numerically identical:

```bash
# pre-extraction: record logits at commit A
python -c "..." > before.json

# post-extraction: record logits at commit B
python -c "..." > after.json

python -c "
import json
a = json.load(open('before.json'))
b = json.load(open('after.json'))
for k in a:
    assert a[k] == b[k], f'{k}: {a[k]} vs {b[k]}'
print('all numerics match')
"
```

This catches any silent algorithmic change that slipped in during the extraction.

### Out of scope for this refactor

The following were considered and deliberately excluded:

- **Splitting `_vfe_iteration` into sub-phases**. The 959-line body has a clear
  structure (natural gradient → trust region → retraction → phi update → sigma
  clamping → etc.) and could be split, but that's a different refactor from
  side-quest extraction. Leave for a follow-up.
- **Moving `get_bayesian_alpha`, `_precondition_phi_grad`, `_compute_phi_grad`,
  `_compute_omega_grad_direct`, `_retract_omega` out of the FFN**. These are used
  every iteration in the main loop and are part of the "core VFE" contract, not
  side quests.
- **Extracting the gauge-preconditioner modes** (Cartan, Killing, Pullback). Those
  already live in `gauge_preconditioner.py` and are called via `_precondition_phi_grad`.
  No change needed.
- **Refactoring `attention.py` shape helpers** into a separate module. They were
  audited in round 7 and are correct; moving them adds no safety.

### Success criteria

After all four extractions:

1. `variational_ffn.py` is ~2,500 lines (down from 3,786, a ~33% reduction).
2. Each side-quest has its own file with a clear public API.
3. The default config (no side quests enabled) produces **bitwise-identical** output
   to the pre-refactor version on a fixed seed.
4. Each side-quest config (`implicit_em=True`, `closed_form_e_step=True`,
   `n_picard_steps > 0`, `use_deq=True`) produces numerically-equivalent output
   (tolerance: 0 for pure code movement, < 1e-6 if any floating-point reordering
   creeps in).
5. No circular imports, no stale references, no broken re-exports.
6. All pre-existing tests still pass.
7. Existing CLI / import patterns continue to work:
   - `from transformer.core.variational_ffn import ImplicitEMGradient` still resolves
   - `from transformer.core.variational_ffn import DEQFixedPoint` still resolves
   - `model.p_flow_update(...)` still callable (even after hebbian extraction)

### Estimated effort

- **Implicit EM extraction**: 30 minutes. Pure extraction + one re-export line.
- **DEQ extraction**: 1-2 hours. Need to convert two step-function factories and
  two autograd Functions, plus update one call site in `variational_ffn.forward`.
- **Closed-form + Picard extraction**: 2-3 hours. Largest single extraction, most
  attribute references to rewrite.
- **Hebbian extraction**: 2-3 hours. Touches five files; highest coordination cost.
- **Testing and verification**: 1 hour per extraction, plus end-to-end integration
  test at the end.

**Total**: 8-12 hours of focused work. Recommended to do over multiple sessions,
one side quest per session, with commits between each.

### Open questions for the user

Before starting the implementation:

1. **Hebbian extraction scope**: should the P-flow / delta-rule code be extracted
   to `transformer/learning/hebbian.py` (new directory) or kept inside
   `transformer/core/` alongside `active_inference.py`? The core/ location is
   more consistent with the existing module layout, but semantically
   P-flow / delta-rule is a *learning rule*, not a core VFE primitive.

2. **Naming**: `vfe_deq.py` / `vfe_implicit_em.py` / `vfe_closed_form.py` or
   drop the `vfe_` prefix (`deq.py`, `implicit_em.py`, `closed_form.py`)? The
   prefix distinguishes them from any unrelated "deq"/"closed_form" that might
   exist elsewhere, but adds clutter.

3. **Atomic commits per side quest** vs. **one big refactor commit**? Atomic is
   safer for bisecting and review, but one big commit makes the final state of
   `variational_ffn.py` easier to grok. Recommendation: atomic.

4. **Checkpoint compatibility**: are there any pickled checkpoints in the field
   that contain `ImplicitEMGradient` references that would break if the class
   is re-homed? Autograd Functions are not serialized in `state_dict()` but
   can appear in `pickle`d optimizer state in rare cases. Recommendation: verify
   with a representative checkpoint before merging.

5. **Do we want to tie the attention and VFE `log_kappa_per_head` parameters**
   (round 6 design concern)? This is orthogonal to the extraction but could be
   decided at the same time. If yes, the FFN would import the attention sublayer's
   parameter instead of having its own.

Answers to these questions determine the final shape of the refactor. I recommend
answering them before starting, then executing the extractions one at a time
following the plan above.
