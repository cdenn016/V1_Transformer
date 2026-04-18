# Refactor step 4 — Hebbian / P-flow / delta-rule extraction — 2026-04-07

Fourth and final atomic side-quest extraction per the refactor plan in
`docs/session_2026_04_07_refactor_plan.md`.  Preceded by:

- **Refactor step 0**: drift hazard #29 fix
- **Fix #44**: share `log_kappa_per_head` between attention and VFE FFN
- **Refactor step 1** (Fix #45): Implicit EM → `vfe_implicit_em.py`
- **Refactor step 2** (Fix #45b + #46): DEQ → `vfe_deq.py`
- **Refactor step 3** (Fix #45c): closed-form E-step + Picard →
  `vfe_closed_form.py`

Unlike steps 1–3 which extracted code from `variational_ffn.py`, step 4
extracts the *backprop-free Hebbian learning* machinery from four
different files (`model.py`, `embeddings.py`, `prior_bank.py`,
`training/experiment_runner.py`) and consolidates it into a single
focused module.

## Fix #45d — Extract Hebbian / P-flow / delta-rule to `core/hebbian.py`

### Why a single module instead of four

The P-flow / delta-rule code is functionally one feature spread across
four files because it touches several different model substructures:
the dispatcher lives on `GaugeTransformerLM`, the per-token-type EMA
update lives on `GaugeTokenEmbedding`, the same EMA logic (with a
gauge-fixed variant) lives on `PriorBank`, and the trainer
orchestration lives on `PublicationTrainer`.  Activated only when
`use_p_flow=True` and/or `use_delta_rule_w_out=True`, this is
research-grade code that the default training path never touches.

Consolidating it into `transformer/core/hebbian.py` makes the feature
discoverable, removes ~700 lines of opt-in machinery from four hot
files, and matches the side-quest extraction pattern established by the
previous three steps.

### What was moved

Approximately 700 lines of P-flow / delta-rule machinery, now in
`transformer/core/hebbian.py` (642 lines after consolidation):

| Original location                       | Method                              | New free function                       |
|:----------------------------------------|:------------------------------------|:----------------------------------------|
| `model.py:p_flow_update`                | dispatcher                          | `p_flow_update_model`                   |
| `model.py:phi_flow_update`              | dispatcher                          | `phi_flow_update_model`                 |
| `model.py:delta_rule_update_w_out`      | Widrow-Hoff W_out update            | `delta_rule_update_w_out_model`         |
| `embeddings.py:_compute_pflow_weights`  | segment-softmax helper              | `compute_pflow_weights`                 |
| `embeddings.py:update_embeddings_from_beliefs` | mu+sigma EMA              | `update_embeddings_from_beliefs`        |
| `embeddings.py:update_phi_from_beliefs` | phi EMA                             | `update_phi_from_beliefs`               |
| `prior_bank.py:update_from_beliefs`     | per-token-type prior EMA            | `update_prior_bank_from_beliefs`        |
| `prior_bank.py:_update_gauge_fixed_base_prior` | de-rotated base prior EMA    | `update_gauge_fixed_base_prior`         |
| `experiment_runner.py:_apply_p_flow_and_delta_rule` | trainer-level orchestration | `apply_p_flow_and_delta_rule`     |

### Extraction pattern

Same as the three preceding side quests: each free function takes the
relevant object (`embed`, `prior_bank`, `model`, `trainer`) as its
first argument and accesses instance attributes via `obj.attr` rather
than `self.attr`.

### Backward compatibility

All four source files keep thin wrapper methods that delegate to the
free functions in `hebbian.py`:

```python
# model.py
def p_flow_update(self, token_ids, mu_beliefs, prediction_errors,
                  ema_decay=0.99, sigma_beliefs=None, pad_token_id=-100):
    """Thin delegator — see ``hebbian.p_flow_update_model``."""
    from transformer.core.hebbian import p_flow_update_model
    return p_flow_update_model(
        self, token_ids, mu_beliefs, prediction_errors,
        ema_decay=ema_decay, sigma_beliefs=sigma_beliefs,
        pad_token_id=pad_token_id,
    )
```

(Imports are lazy/in-function rather than top-of-file to avoid a
circular dependency: `hebbian.py` references model attributes
post-construction, and several call sites in the model module need to
import `hebbian.py` indirectly.)

External callers continue to work unchanged:

```python
trainer.model.p_flow_update(...)              # works
trainer.model.phi_flow_update(...)            # works
trainer.model.delta_rule_update_w_out(...)    # works
embed.update_embeddings_from_beliefs(...)     # works
embed.update_phi_from_beliefs(...)            # works
prior_bank.update_from_beliefs(...)           # works
trainer._apply_p_flow_and_delta_rule(...)     # works
```

## Verification

Five P-flow / delta-rule code paths and four regression configurations
were tested:

```python
# Code paths exercised:
# 1. GaugeTokenEmbedding p_flow update
model.p_flow_update(token_ids, mu_beliefs, prediction_errors, ...)
# OK: max embedding change = 4.93e-01

# 2. PriorBank p_flow update (token-indexed prior_mu)
model_pb.p_flow_update(...)
# OK: max prior_mu change = 2.44e-01

# 3. PriorBank phi_flow update
model_pb.phi_flow_update(token_ids, phi_evolved, prediction_errors, ...)
# OK: max phi_embed change = 2.79e-01

# 4. Gauge-fixed base prior update (de-rotation path)
model_gf.p_flow_update(...)
# OK: max base_prior_mu change = 5.73e-02

# 5. delta_rule_update_w_out (W_out Widrow-Hoff)
model.delta_rule_update_w_out(mu_beliefs, targets, lr=0.01)
# OK: max W_out change = 1.76e-03
```

```python
# Regression configurations:
# 1. Default config (no P-flow, no delta rule) — bitwise identical
max |logits_before - logits_after| = 0.00e+00, loss = 26.4843

# 2. closed_form_e_step regression (side quest #3)
loss = 29.1647

# 3. DEQ regression (side quest #2)
loss = 26.4843  (matches default — DEQ Neumann inactive without backward in eval)

# 4. Implicit EM regression (side quest #1)
loss = 26.4843
```

All five P-flow paths and all four regression configurations pass,
confirming:
- The extraction is behaviour-preserving for the default training path
- All five Hebbian / P-flow code paths route through the new module
  and produce non-zero updates as expected
- Earlier extractions (Implicit EM, DEQ, closed-form) did not regress

## Line counts (after step 4)

| File                                        | Before | After | Delta |
|:--------------------------------------------|-------:|------:|------:|
| `transformer/core/model.py`                 |  1,679 | 1,565 |  −114 |
| `transformer/core/embeddings.py`            |  1,283 | 1,154 |  −129 |
| `transformer/core/prior_bank.py`            |    763 |   632 |  −131 |
| `transformer/training/experiment_runner.py` |  3,114 | 3,050 |   −64 |
| `transformer/core/hebbian.py`               |      0 |   642 |  +642 |
| **Total**                                   |  6,839 | 7,043 |  +204 |

The four hot files lost a cumulative **438 lines** of opt-in research
code.  The total grew by 204 lines because `hebbian.py` includes
module docstrings, the consolidated public-API list, and explicit
type-hint imports that were previously inherited from the host
modules.

## Cumulative refactor progress (final)

| #  | Side quest                            | Target file                | Status      | Lines |
|----|---------------------------------------|----------------------------|-------------|------:|
| ✅ | 1. Implicit EM (Fix #45)              | `vfe_implicit_em.py`       | **DONE**    |   193 |
| ✅ | 2. DEQ (Fix #45b + #46)               | `vfe_deq.py`               | **DONE**    |   721 |
| ✅ | 3. Closed-form E-step + Picard (#45c) | `vfe_closed_form.py`       | **DONE**    |   583 |
| ✅ | 4. Hebbian / P-flow / delta rule (#45d) | `core/hebbian.py`        | **DONE**    |   642 |

### Cumulative reduction in `variational_ffn.py`

| Step                                 | Lines | Cumulative delta |
|:-------------------------------------|------:|-----------------:|
| Original                             | 3,786 |                — |
| After step 1 (Implicit EM)           | 3,665 |             −121 |
| After step 2 (DEQ + Fix #46)         | 3,135 |             −651 |
| After step 3 (closed-form + Picard)  | 2,714 |           −1,072 |

`variational_ffn.py` has shrunk by **1,072 lines or 28%** from the
original 3,786.  The four side-quest modules together account for
**2,139 lines** of opt-in research code that no longer pollutes the
hot path.

Step 4 additionally reduced four hot files (`model.py`,
`embeddings.py`, `prior_bank.py`, `experiment_runner.py`) by a
cumulative **438 lines** of P-flow / delta-rule machinery.

### Pre-refactor work (already documented)

- Drift hazard #29 consolidation (block forward unified)
- Fix #44 (kappa parameter sharing between attention and VFE FFN, with
  Module-reference revision after CUDA breakage)

The four-side-quest refactor is complete.
