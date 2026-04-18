# Refactor step 2 — DEQ extraction — 2026-04-07

Second atomic side-quest extraction per the refactor plan in
`docs/session_2026_04_07_refactor_plan.md`.  Preceded by:

- **Refactor step 0**: drift hazard #29 fix (documented in
  `session_2026_04_07_refactor_plan.md` Part 1)
- **Fix #44**: share `log_kappa_per_head` between attention sublayer and
  VFE FFN (documented in `session_2026_04_07_refactor_fix44_and_implicit_em.md`)
- **Refactor step 1** (Fix #45): Implicit EM → `vfe_implicit_em.py`
  (documented in `session_2026_04_07_refactor_fix44_and_implicit_em.md`)

## Fix #45b — Extract DEQ to `vfe_deq.py`

### What was moved

From `transformer/core/variational_ffn.py` (approximately 580 lines total)
to a new module `transformer/core/vfe_deq.py`:

1. **`DEQFixedPoint`** — autograd Function for the (μ, Σ) fixed-point
   Neumann-series backward pass. Activated by `use_deq=True` when
   `deq_include_phi=False`.
2. **`DEQFixedPointFull`** — autograd Function for the joint (μ, Σ, φ)
   fixed-point Neumann-series backward pass. Activated by `use_deq=True`
   and `deq_include_phi=True`.
3. **`_DEQ_VJP_NORM_CAP`** — the divergence-cap constant (`1e4`) shared
   between both Functions, now a module-level constant. Still accessible
   as `DEQFixedPoint._DEQ_VJP_NORM_CAP` via a class attribute for
   backward compatibility.
4. **`make_deq_step_fn(ffn, phi_current, mu_p_current, sigma_p, mask,
   is_diagonal, eps, dtype)`** — free function that builds the (μ, Σ)
   E-step closure. Previously an instance method `_make_deq_step_fn` on
   `VariationalFFNDynamic` at ~156 lines.
5. **`make_deq_step_fn_with_phi(ffn, mu_p_current, sigma_p, mask,
   is_diagonal, eps, dtype)`** — free function that builds the joint
   (μ, Σ, φ) E-step closure including the differentiable phi retraction.
   Previously an instance method `_make_deq_step_fn_with_phi` at ~262 lines.

### Extraction pattern

Same as the existing `active_inference.py` pattern: the free functions
take `ffn` as the first argument and access instance attributes via
`ffn.attr` rather than `self.attr`. The closure captures `ffn` by
reference so the attribute lookups are resolved at call time (not at
factory time), which matches the original method-on-class semantics.

### Backward compatibility

`variational_ffn.py` now imports and re-exports the two autograd
Functions, and keeps thin wrapper methods (`_make_deq_step_fn`,
`_make_deq_step_fn_with_phi`) on `VariationalFFNDynamic` that delegate
to the free functions:

```python
# variational_ffn.py
from transformer.core.vfe_deq import (
    DEQFixedPoint,
    DEQFixedPointFull,
    make_deq_step_fn as _make_deq_step_fn_free,
    make_deq_step_fn_with_phi as _make_deq_step_fn_with_phi_free,
)

class VariationalFFNDynamic(nn.Module):
    ...
    def _make_deq_step_fn(self, phi_current, mu_p_current, sigma_p,
                          mask, is_diagonal, eps, dtype):
        return _make_deq_step_fn_free(
            self, phi_current, mu_p_current, sigma_p,
            mask, is_diagonal, eps, dtype,
        )

    def _make_deq_step_fn_with_phi(self, mu_p_current, sigma_p,
                                    mask, is_diagonal, eps, dtype):
        return _make_deq_step_fn_with_phi_free(
            self, mu_p_current, sigma_p,
            mask, is_diagonal, eps, dtype,
        )
```

External consumers that do `from transformer.core.variational_ffn import
DEQFixedPoint` continue to work. Internal call sites in
`variational_ffn.forward()` that call `self._make_deq_step_fn(...)` or
`self._make_deq_step_fn_with_phi(...)` also continue to work unchanged —
the wrapper methods forward to the free functions with `self` inserted
as the first argument.

## Fix #46 — Pre-existing bug in DEQ joint-phi backward

### The bug

While extracting the DEQ code I discovered a latent bug in the joint
(μ, Σ, φ) backward path that had survived all seven audit rounds because
no test exercises `deq_include_phi=True` with `deq_neumann_terms > 1`.

Inside the joint step closure, the phi sub-gradient is computed via a
nested `torch.autograd.grad` call with `create_graph=True` so the
second-order graph is available to the outer DEQ VJP:

```python
grad_phi_align = torch.autograd.grad(
    alignment_loss, phi_in,
    create_graph=True,
    retain_graph=False,   # ← BUG
)[0]
phi_out = phi_in - phi_lr_step * grad_phi_align
```

The explicit `retain_graph=False` tells PyTorch to free the graph of
`alignment_loss` immediately after computing `grad_phi_align`. But the
outer `DEQFixedPointFull.backward` runs a Neumann-series loop over `K`
iterations, and on each iteration it calls
`autograd.grad(outputs=[mu_out, sigma_out, phi_out], ..., retain_graph=False)`.

The intermediate values computed by the attention/KL operations inside
`alignment_loss` are **shared** with the intermediate values used by
`mu_out` and `sigma_out` (all three outputs come from the same per-head
`compute_attention_weights` and `compute_vfe_gradients_gpu` calls). When
the inner `autograd.grad` freed the `alignment_loss` graph, it also
freed those shared intermediates. On the FIRST Neumann iteration this
still worked because the outer VJP ran immediately after the freeing
and could use the then-current saved state. But on iteration K=1, PyTorch
tried to compute the VJP through the same graph a second time and
crashed with:

    RuntimeError: Trying to backward through the graph a second time
    (or directly access saved tensors after they have already been freed).

### The fix

Changed `retain_graph=False` to `retain_graph=True` in the inner
`autograd.grad` call. Now the intermediates are preserved until the
DEQ backward loop completes, allowing each Neumann iteration to reuse
them correctly. Added a long comment explaining the trap for future
maintainers. This is finding #46 in the audit log.

### Why it survived seven rounds of audit

The existing test suite does not exercise `deq_include_phi=True` with
`deq_neumann_terms > 1`. Round 5 (test coverage audit) flagged the DEQ
backward as one of five "critical gaps — no finite-difference tests
comparing the IFT-corrected ∂z*/∂θ against the truth". This fix
partially addresses that gap: the joint-phi path now at least runs
without crashing on K > 1 iterations. A full correctness check
(differential test against a finite-difference reference) is still
recommended as a follow-up test addition.

## Verification

```python
# 1. Syntax check
import ast
for f in ['variational_ffn.py', 'vfe_deq.py', 'vfe_implicit_em.py']:
    ast.parse(open(f).read())  # all clean

# 2. Re-exports identical
from transformer.core.variational_ffn import DEQFixedPoint, DEQFixedPointFull
from transformer.core.vfe_deq import DEQFixedPoint as DEQ2, DEQFixedPointFull as DEQF2
assert DEQFixedPoint is DEQ2 and DEQFixedPointFull is DEQF2

# 3. DEQ (mu, sigma) path
model.train()
logits, _ = model.forward_with_attention(tok, targets=tgt)
loss.backward()
# OK: loss=3.5207, grads flow

# 4. DEQ joint (mu, sigma, phi) path — the one that triggered the
#    pre-existing bug #46
config2 = {**config, 'deq_include_phi': True, 'evolve_phi': True}
model2 = GaugeTransformerLM(config2)
model2.train()
logits2, _ = model2.forward_with_attention(tok, targets=tgt)
loss2.backward()
# OK: loss=3.5207, grads flow

# 5. Default config (no DEQ) bitwise identical
max |logits_inf - logits_train| = 0.00e+00

# 6. Implicit EM path (Fix #45 regression) still works
# OK: loss=3.5207
```

All four configurations pass, confirming that:
- The extraction is behaviour-preserving (default config still produces
  bitwise-identical output)
- Neither DEQ path nor the Implicit EM path regressed
- The pre-existing bug #46 is now fixed — the joint-phi DEQ path runs
  successfully with `deq_neumann_terms > 1` for the first time

## Line counts

| File                                        | Before | After | Delta |
|:--------------------------------------------|-------:|------:|------:|
| `transformer/core/variational_ffn.py`       |  3,665 | 3,135 |  −530 |
| `transformer/core/vfe_deq.py`               |      0 |   721 |  +721 |
| `transformer/core/vfe_implicit_em.py`       |    193 |   193 |    +0 |
| **Total**                                   |  3,858 | 4,049 |  +191 |

`variational_ffn.py` has dropped another 530 lines (from 3,665 to 3,135),
a cumulative reduction from the original 3,786 of **651 lines or 17%**.
The total across the three files grew by 191 lines due to module
docstrings, Fix #46 comments, and the `TYPE_CHECKING` forward-reference
import pattern.

## Cumulative refactor progress

| # | Side quest                       | Target file               | Status      | Lines |
|---|----------------------------------|---------------------------|-------------|------:|
| ✅ | Implicit EM (Fix #45)            | `vfe_implicit_em.py`      | **DONE**    |   193 |
| ✅ | DEQ (this commit, Fix #45b + #46) | `vfe_deq.py`              | **DONE**    |   721 |
| 3 | Closed-form E-step + Picard      | `vfe_closed_form.py`      | PLANNED     |  ~550 |
| 4 | Hebbian / P-flow / delta rule    | `core/hebbian.py`         | PLANNED     |  ~550 |

Plus the earlier pre-refactor work:
- Drift hazard #29 consolidation (block forward unified)
- Fix #44 (kappa parameter sharing between attention and VFE FFN)

Ready for step 3 whenever you are.
