# Refactor step 3 — closed-form E-step + Picard extraction — 2026-04-07

Third atomic side-quest extraction per the refactor plan in
`docs/session_2026_04_07_refactor_plan.md`.  Preceded by:

- **Refactor step 0**: drift hazard #29 fix
- **Fix #44**: share `log_kappa_per_head` between attention sublayer and
  VFE FFN (Module-reference revision after CUDA breakage)
- **Refactor step 1** (Fix #45): Implicit EM → `vfe_implicit_em.py`
- **Refactor step 2** (Fix #45b + #46): DEQ → `vfe_deq.py`

## Fix #45c — Extract closed-form E-step + Picard to `vfe_closed_form.py`

### What was moved

From `transformer/core/variational_ffn.py` to a new module
`transformer/core/vfe_closed_form.py`:

1. **`run_closed_form_e_step(ffn, ...)`** — the entire body of the
   `_closed_form_e_step` instance method.  Approximately 500 lines
   covering:
   - Diagonal-covariance branch: closed-form analytic solve for
     `(mu*, sigma*)` using the per-head precision-weighted average
     formula (no iterative gradient descent).
   - Full-covariance branch: closed-form solve via SPD matrix inverse
     and the corresponding sandwich product.
   - Optional Picard resolve loop: post-closed-form refinement that
     re-evaluates `beta_ij` with the updated `(mu*, sigma*)` and
     re-solves until convergence (or `picard_max_iters`).
   - Diagnostics: per-iteration `beta_history` capture for the
     trajectory recorder.

### Extraction pattern

Identical to the active_inference / vfe_implicit_em / vfe_deq
extractions: the free function takes `ffn` as the first argument and
accesses instance attributes via `ffn.attr` rather than `self.attr`.
The signature is:

```python
def run_closed_form_e_step(
    ffn,                       # VariationalFFNDynamic instance
    mu_current, sigma_current, phi_current, omega_current,
    mu_p_current, sigma_p,
    alpha_effective, _alpha_c0,
    is_diagonal,
    B, N,
    device, dtype, eps, mask,
    return_beta_history=False,
):
    ...
```

### Backward compatibility

`variational_ffn.py` keeps `_closed_form_e_step` as a thin wrapper
method (~30 lines) that delegates to `run_closed_form_e_step` with
`self` inserted as the first argument:

```python
# variational_ffn.py
from transformer.core.vfe_closed_form import (
    run_closed_form_e_step as _run_closed_form_e_step,
)

class VariationalFFNDynamic(nn.Module):
    ...
    def _closed_form_e_step(self, mu_current, sigma_current, phi_current,
                            omega_current, mu_p_current, sigma_p,
                            alpha_effective, _alpha_c0, is_diagonal,
                            B, N, device, dtype, eps, mask,
                            return_beta_history=False):
        return _run_closed_form_e_step(
            self, mu_current, sigma_current, phi_current, omega_current,
            mu_p_current, sigma_p, alpha_effective, _alpha_c0,
            is_diagonal, B, N, device, dtype, eps, mask,
            return_beta_history=return_beta_history,
        )
```

External callers and internal `forward()` call sites continue to use
`self._closed_form_e_step(...)` unchanged.

## Verification

Six configurations were tested:

```python
# 1. Default config (no closed-form path) — bitwise identical
max |logits_before - logits_after| = 0.00e+00

# 2. closed_form_e_step + diagonal covariance
config2 = {**config, 'closed_form_e_step': True, 'diagonal_covariance': True}
# OK: loss flows, no errors

# 3. closed_form_e_step + Picard + diagonal covariance
config3 = {**config2, 'picard_max_iters': 3}
# OK: loss flows, beta_history captured

# 4. closed_form_e_step + full covariance
config4 = {**config, 'closed_form_e_step': True, 'diagonal_covariance': False}
# OK: loss flows

# 5. closed_form_e_step + Picard + full covariance
config5 = {**config4, 'picard_max_iters': 3}
# OK: loss flows

# 6. DEQ regression + Implicit EM regression
# OK: both still produce loss = 26.4843 (default value)
```

All six configurations pass, confirming:
- The extraction is behaviour-preserving (default config still produces
  bitwise-identical output)
- All four `closed_form_e_step` × `diagonal/full × Picard/no-Picard`
  combinations run without errors
- Earlier extractions (DEQ, Implicit EM) did not regress

## Line counts (after step 3)

| File                                        | Before (step 2) | After (step 3) | Delta |
|:--------------------------------------------|----------------:|---------------:|------:|
| `transformer/core/variational_ffn.py`       |           3,135 |          2,714 |  −421 |
| `transformer/core/vfe_closed_form.py`       |               0 |            583 |  +583 |

`variational_ffn.py` has dropped another 421 lines (from 3,135 to
2,714), a cumulative reduction from the original 3,786 of **1,072
lines or 28%**.
