# Refactor step 1 — 2026-04-07

## Fix #44 — Share `log_kappa_per_head` between attention sublayer and VFE FFN

### Rationale

κ is the temperature in `β_ij = softmax(−KL(q_i ‖ Ω_ij q_j) / (κ·√d_h))`.
Under the manuscript's framework β is interpreted as the posterior `p(j | i)`
— "the probability that token j is the relevant context for token i". That
posterior is a single physical quantity derived from the pairwise KL on the
belief manifold. There is not a "β for message aggregation" and a separate
"β for belief refinement" — there is one β, computed once per (i, j) pair.

The attention sublayer uses β for message aggregation
`μ_out = Σ_j β_ij · Ω_ij · μ_j` (the V-path), and the VFE FFN uses the same
β in its alignment term `F_align = Σ_ij β_ij · KL_ij`. Both are reading
the same posterior, so they should use the same temperature parameter.

Historically each sublayer owned its own `nn.Parameter` of the same shape,
initialised to the same value. Gradient descent was free to make them
drift apart — the attention's β and the E-step's internal β would then
describe *different* posteriors even though they're ostensibly the same
quantity. This was flagged in round 6 as a "design concern (not a bug)"
and deferred for user decision.

### Implementation

**Pattern**: same as `_prior_bank_ref` in `active_inference.py` —
`__dict__` assignment to bypass `nn.Module.__setattr__` and avoid
double-registration.

**`transformer/core/variational_ffn.py`** — `VariationalFFNDynamic.__init__`
still creates a *safety-net* local parameter so the FFN remains functional
if it's ever instantiated standalone (e.g. in a unit test). The docstring
explains this.

**`transformer/core/blocks.py`** — `GaugeTransformerBlock.__init__`, after
both the attention sublayer and the FFN are constructed, deletes the
FFN's placeholder parameter and buffer from `self.ffn._parameters` /
`self.ffn._buffers` and installs direct references via `__dict__`:

```python
if cfg.learnable_head_kappa and getattr(self.attention, 'log_kappa_per_head', None) is not None:
    if 'log_kappa_per_head' in self.ffn._parameters:
        del self.ffn._parameters['log_kappa_per_head']
    if '_kappa_init' in self.ffn._buffers:
        del self.ffn._buffers['_kappa_init']
    self.ffn.__dict__['log_kappa_per_head'] = self.attention.log_kappa_per_head
    self.ffn.__dict__['_kappa_init'] = self.attention._kappa_init
```

### Verification

Runtime smoke test on a 2-layer SO(3) model with `learnable_head_kappa=True`
confirms:

- `block.attention.log_kappa_per_head is block.ffn.log_kappa_per_head` (tensor identity, not copy)
- `block.attention._kappa_init is block.ffn._kappa_init` (buffer identity)
- `model.named_parameters()` contains the kappa parameter **exactly twice**
  (once per layer, under `transformer.blocks.N.attention.log_kappa_per_head`) —
  not four times
- `model.named_buffers()` contains `_kappa_init` exactly twice (same pattern)
- Mutating `block.attention.log_kappa_per_head` is immediately visible via
  `block.ffn.log_kappa_per_head` (reference semantics)
- `state_dict()` has exactly 2 kappa keys (no duplication)
- Forward pass produces bitwise-identical logits between
  `model(tok)` (inference) and `model.forward_with_attention(tok)` (training)
  — `max |diff| = 0.00e+00`
- The `learnable_head_kappa=False` path still works (both sublayers hold
  `None`, and `_get_kappa_h` falls through to the scalar `self.kappa`)

### Backward compatibility

Old checkpoints saved before Fix #44 contain both
`attention.log_kappa_per_head` and `ffn.log_kappa_per_head` in their state
dict. After the fix, only the attention key is registered. Loading such a
checkpoint with default `strict=True` would fail with
*unexpected key `ffn.log_kappa_per_head`*. Users with old checkpoints should
either load with `strict=False` or drop the redundant keys manually before
loading. No automatic migration is provided.

---

## Fix #45 — Extract Implicit EM to `vfe_implicit_em.py`

First side-quest extraction from the refactor plan in
`docs/session_2026_04_07_refactor_plan.md`. Implicit EM was selected as the
starting point because it's the smallest and most self-contained side quest
(no coupling to `_vfe_iteration`, no cyclic imports, pure autograd
Functions).

### What was moved

From `transformer/core/variational_ffn.py:90-250` (161 lines) to a new
module `transformer/core/vfe_implicit_em.py`:

- `ImplicitEMGradient` — autograd Function for the μ gradient scale
- `ImplicitEMGradientSigma` — autograd Function for the σ gradient scale
- `compute_implicit_em_scales(alpha_i, sigma_p, beta, sigma_q, eps)` —
  helper computing the IFT scale factors from the E-step fixed-point
  values

### Backward compatibility

`variational_ffn.py` now re-exports the three symbols via a simple import:

```python
from transformer.core.vfe_implicit_em import (
    ImplicitEMGradient,
    ImplicitEMGradientSigma,
    compute_implicit_em_scales,
)
```

External consumers that import from `variational_ffn` continue to work
unchanged. The re-export uses the exact same underlying objects
(`ImplicitEMGradient is vfe_implicit_em.ImplicitEMGradient`), so `isinstance`
checks and `torch.autograd.Function` subclass dispatch continue to function.

`model.py:39` already imports `ImplicitEMGradient` and `ImplicitEMGradientSigma`
from `variational_ffn`. No change needed — the re-export is transparent.

### Verification

- Both files parse cleanly (`ast.parse`)
- Backward-compat import from `variational_ffn` works
- Direct import from `vfe_implicit_em` works
- `ImplicitEMGradient is vfe_implicit_em.ImplicitEMGradient` (symbol identity)
- `model.py` imports still resolve
- Runtime forward+backward with `implicit_em=True` produces a finite loss
  (3.52), gradients flow to parameters
- Line counts: `variational_ffn.py` = 3,665 (down from 3,786);
  `vfe_implicit_em.py` = 193 (total = 3,858, up 72 lines from docstrings
  and module header)

### Scope

This extraction preserves all existing behaviour. No algorithmic changes,
no parameter renames, no new flags. The atomic commit strategy means the
next side quest (DEQ) can proceed from a known-good baseline.

## Next steps

Per the refactor plan (`docs/session_2026_04_07_refactor_plan.md`), the
remaining extractions are:

2. **DEQ** → `vfe_deq.py` (~600 lines, 1-2 hours)
3. **Closed-form E-step + Picard** → `vfe_closed_form.py` (~550 lines, 2-3 hours)
4. **Hebbian / P-flow / delta rule** → `core/hebbian.py` (~550 lines, 2-3 hours)

Each is an independent atomic commit. Together they should bring
`variational_ffn.py` from the current 3,665 lines down to ~2,500 lines
(a ~32% reduction from the 3,786 pre-refactor baseline).
