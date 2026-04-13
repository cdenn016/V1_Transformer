# Deep Audit Round 4 Fixes — 2026-04-07

Continuation of the three-round deep audit recorded in
`session_2026_04_07_deep_audit_fixes.md`. Round 4 covers the remaining
critical code paths: model construction, the `generate` method,
`_compute_omega_grad_direct`, and the post-backward training step.

## Summary of round 4 fixes

| # | Finding | File:Line | Severity |
|---|---------|-----------|----------|
| 30 | `generate` crashes on temperature=0 (div-by-zero → NaN → multinomial error) | `model.py:1554` | HIGH |
| 31 | `_compute_omega_grad_direct` unsafe `torch.linalg.inv` at two sites | `variational_ffn.py:1057,1759` | MEDIUM |
| 32 | `_compute_omega_grad_direct` silently ignores `lambda_softmax` | `variational_ffn.py:1089` | MEDIUM |
| 33 | `self.config = config` stores live reference; `config['gauge_param']` mutates caller's dict | `model.py:120,254-255` | MEDIUM |
| 34 | Fallback random generators on `math_utils.generators` import failure | `model.py:678-687` | MEDIUM |

### False positive from round 4 subagent (withdrawn)

- **Top-p scatter "bug" (`generate` line 1573-1575)**: a subagent flagged the
  non-in-place `sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)`
  call as incorrect because the same tensor is used as both `self` and
  `src`. Direct verification with a concrete test showed that because
  `sorted_indices` is a full permutation of `[0, V-1]`, every output
  position is overwritten by the scatter and the initial tensor value is
  irrelevant. The result is identical to the `torch.zeros_like(...).scatter(...)`
  alternative. Withdrew the bug but rewrote the line to use `zeros_like`
  for clarity — that's a readability improvement, not a correctness fix.

## Detailed descriptions

### Fix #30 — `generate` temperature=0 greedy short-circuit

`GaugeTransformerLM.generate` divides `logits / temperature` at the top
of the sampling loop. When a caller passes `temperature=0` (the
conventional signal for "greedy decoding"), the division produces
`inf` for every logit, then `F.softmax(inf, ...)` produces `NaN`
probabilities, and `torch.multinomial` raises
`"probability tensor contains either inf, nan or element < 0"`. The
generation loop crashes on the first iteration.

**Fix**: added a `temperature <= 0` short-circuit at the top of the
sampling loop that does `argmax(logits[:, -1, :])` directly and
`continue`s, bypassing the temperature scaling, top-k/top-p filtering,
and multinomial sampling entirely. Also rewrote the top-p scatter to
use `torch.zeros_like(logits_next, dtype=torch.bool).scatter(...)` for
clarity (the old form was mathematically equivalent but confusing).

### Fix #31 — `_compute_omega_grad_direct` and `_build_block_exp_pairs` unsafe inverses

`variational_ffn.py:1057` computed `torch.linalg.inv(om_blk)` on the
per-head omega block directly, and `variational_ffn.py:1759` did the
same inside `_build_block_exp_pairs`. Both sites are analogous to the
unsafe inverse in `transport_ops.compute_transport_operators_direct`
and `omega_to_block_exp_pairs` that was fixed in round 2 (Fix #16).
During training, `omega` can drift toward low rank (the GL(K) direct
parameterisation permits any determinant sign), and a raw inverse on a
near-singular matrix poisons the attention graph with NaN.

**Fix**: added ridge-regularised inverse with a pinv fallback at both
sites:
```python
omega_h_reg = omega_h + 1e-6 * I
try:
    omega_h_inv = torch.linalg.inv(omega_h_reg)
except (torch.linalg.LinAlgError, RuntimeError):
    omega_h_inv = torch.linalg.pinv(omega_h_reg)
```
Matches the pattern already used in `transport_ops.py`.

### Fix #32 — `_compute_omega_grad_direct` missing `lambda_softmax`

`_compute_omega_grad_direct` at `variational_ffn.py:1089` computed the
alignment loss as

```python
alignment_loss += self.lambda_belief * (beta_h * kl_h).sum()
```

which uses a SINGLE weight `lambda_belief` applied to the full product
`β · KL`. Autograd's chain rule automatically produces
`d/dω [β·KL] = β·dKL/dω + KL·dβ/dω`, so the combined gradient is
`lambda_belief · (direct + softmax_coupling)` — the two contributions
are weighted equally.

The sibling `_compute_phi_grad` at `variational_ffn.py:1171-1174` splits
the product rule into the two terms with SEPARATE weights:
```python
alignment_loss += (
    self.lambda_belief * (beta_h.detach() * kl_h).sum()   # direct term
    + self.lambda_softmax * (beta_h * kl_h.detach()).sum() # softmax coupling
)
```

The phi path honours `lambda_softmax`; the omega path silently ignored
it. Whenever `lambda_belief != lambda_softmax`, the two gradient paths
compute different gradients for what should be the same quantity.

**Fix**: rewrote the omega alignment loss to use the same double
stop-gradient trick as the phi path. When `lambda_belief == lambda_softmax`
the result is unchanged (`d/dω [(sg[β])·KL + β·(sg[KL])]
= β·dKL/dω + KL·dβ/dω = d/dω [β·KL]`). When they differ, the omega path
now honours both weights symmetrically with the phi path.

### Fix #33 — `self.config` live reference and caller mutation

At `model.py:120`, `GaugeTransformerLM.__init__` stored the user's
config dict by reference (`self.config = config`). At `model.py:254`,
the init then mutated this dict with `config['gauge_param'] = gauge_param`
before passing it to `BlockConfig.from_config`. Two problems:

1. **Live reference**: later edits to the user's config dict silently
   change model behaviour (e.g., `model.config['kappa_beta'] = 0.5`
   would affect the next forward pass if `self.config['kappa_beta']` is
   read anywhere).
2. **Caller mutation**: constructing the model inserts `gauge_param`
   into the user's original dict. If the user re-uses the same dict
   for a second model, the second construction sees the mutation. If
   the user introspects their "original" config later, they see a
   modified version.

**Fix**: `config = copy.deepcopy(config)` at the top of `__init__` so
all subsequent mutations and storage operate on a private snapshot.
Added `import copy` to the top-of-file imports. The fix is a pure
isolation improvement — no runtime behaviour changes when the user
does not later mutate the original config.

### Fix #34 — Random generator fallback on import failure

`_build_generators` at `model.py:678-687` had a silent fallback that
activated when `GENERATORS_AVAILABLE=False` (meaning
`math_utils.generators` failed to import). The fallback constructed
random skew-symmetric matrices as the "Lie algebra generators" and
emitted a `RuntimeWarning`. Training would then proceed with random
meaningless generators — no Casimir relations, no block-diagonal
structure, no correct commutation relations. The model would silently
fail to implement the gauge transformer thesis at all.

**Fix**: replaced the warning+fallback with `raise RuntimeError(...)`
that explicitly tells the user their installation is broken and
refuses to construct the model. Training with random generators is
worse than not training at all; failing loudly is the correct
behaviour.

## Files touched in round 4

- `transformer/core/model.py` — Fix #30, #33, #34
- `transformer/core/variational_ffn.py` — Fix #31 (×2), Fix #32

All files parse cleanly via `ast.parse`. Smoke tests for the generate
temperature=0 path and the top-p scatter equivalence both pass.

## What round 4 verified as correct

- `train.py:gaussian_kl_divergence` — both diagonal and full covariance
  paths compute the correct Gaussian KL formula with Cholesky-based
  triangular solves, logdet via 2·sum(log(diag(L))), and a NaN guard
  that replaces non-finite values with the `kl_ceil` (repulsive, not
  attractive) default.
- `GaugeTransformerStack.forward` (`blocks.py:619-738`) — sequential
  block application with `hierarchical_priors` correctly detaching
  `mu_prior = mu_q.detach()` between layers, gradient checkpointing
  gated on `self.gradient_checkpointing and self.training and not
  is_final`, and omega evolution propagated to the next layer via
  `_last_evolved_omega`.
- `experiment_runner.PublicationTrainer._run_forward_and_backward`
  and `train_step` — correct delegation to `compute_free_energy_loss`,
  gradient accumulation scaling by `1/accum_steps`, tied-weights delta
  rule handling, Cartan preconditioning of phi gradients before the
  optimizer step.
- **Cross-head permutation balance**: every forward permutation at
  `_embed_and_prepare:416` is paired with an inverse permutation
  before the vocabulary projection (at `model.py:997` for `forward`,
  and `model.py:1381` for `forward_with_attention`). Intermediate
  per-layer probes at `model.py:1274,1332` apply the inverse locally
  without feeding back into the main pipeline.
- **Data pipeline** (via subagent): target shift correct, padding
  consistent (input:0, target:-100, CE:ignore_index=-100), vocab
  construction deterministic across splits, deterministic worker
  seeding.
- **Model construction** (via subagent and direct spot-check):
  generators registered as buffers (move with `.to(device)`),
  `out_proj` has `bias=False`, embedding tying happens AFTER
  `_init_weights` to preserve calibrated sigma, `_causal_mask`
  registered as non-persistent buffer.

## Remaining unread / not-in-scope

- **`VariationalFFNDynamic.__init__` past line 860** — briefly scanned;
  no obvious issues in parameter registration. Full line-by-line
  review deferred.
- **`_make_deq_step_fn` / `_make_deq_step_fn_with_phi` internals**
  (`variational_ffn.py:1213-1618`) — the factory functions that build
  the DEQ step closures. Not read in detail; the DEQ backward
  infrastructure (round 2 findings #12, #13) was already fixed.
- **`prior_bank.__init__`, `embeddings.__init__`, `attention.__init__`**
  — parameter registration and initialisation code; spot-checked but
  not line-by-line reviewed.
- **`training/metrics.py`, `training/config.py`** — metric
  computation and config dataclass; the subagent verified the FastTrainer
  loss/metrics flow end-to-end.
- **Test suite (`tests/`)** — not audited. Would be useful to see
  what invariants are already protected by regression tests.
- **Scripts (`scripts/`)** — not audited. Ablation runners and
  publication figure generation.

## Cumulative audit summary

Across four rounds, the audit found and fixed **17 distinct actionable
issues** plus withdrew **2 false positives**. The most consequential
fix was round 3 Fix #20 (attention residual bug live in the training
forward path), which was silently degrading training dynamics at every
layer and was not caught by the round 1 fix because the training loop
uses an independently-written code path (`forward_with_attention`)
that drifts from the inference path (`forward`).

The core math of the VFE pipeline — KL attention, Fisher-metric
natural gradient, sandwich-product covariance transport, per-block
fused kernels, active inference EFE/distillation, implicit-EM
gradient scaling — is correct. Remaining concerns are concentrated
in opt-in configurations (direct omega parameterisation, non-flat
transport, DEQ + active inference, closed-form E-step) and in the
drift hazard between the two block-forward implementations that
underlies Fix #29.
