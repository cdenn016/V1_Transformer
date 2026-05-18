# Codebase Map — Where Each Construct Lives

**Navigation aid.** This file tells the auditor where in the codebase to look for each construct. It is descriptive — not theoretical canon. The agent evaluates the constructs against the external canon files (`external_canon_*.md`), not against this file.

Verify these paths exist before relying on them (memory can drift; code moves).

## Top-level layout

```
transformer/
  core/          # legacy / mixed-paradigm modules (variational_ffn, attention, blocks, ...)
  vfe/           # the pure VFE package (block, e_step, stack, model, attention, ...)
  pure_vfe/      # standalone pure VFE implementation with own tests
  baselines/     # standard_transformer.py and hybrid_gauge_transformer.py for comparison
  analysis/     # holonomy, fiber trajectory, gauge geometry, scaling reports
  training/      # train_fast, optimizer, metrics, config (TrainingConfig)
  visualization/ # publication plots
```

## `transformer/core/` vs `transformer/vfe/` — three structural differences

(From the project memory file `project_vfe_vs_ift_phi_multilayer.md`.)

1. **Prior μ cascade.** `vfe/` propagates the previous layer's posterior `μ_q` as the next layer's `μ_p`. `core/` does not — it re-uses the embedding-side `μ_p` per layer. Manuscripts often describe one or the other ambiguously; flag if the manuscript says "the prior is the previous layer's posterior" but the code being benchmarked is `core/`.
2. **Residual + LayerNorm placement.** `core/` uses transformer-style pre-norm residual around the FFN; `vfe/` uses a different placement (check `vfe/block.py::forward` for the exact arrangement).
3. **Evolved Ω vs propagated φ.** `core/` propagates φ between layers and reconstructs Ω at each site; `vfe/` evolves Ω directly under the alignment loss in `_update_phi`.

These differences mean **per-layer behavior is genuinely different between the two packages**. An audit that uses the wrong package will produce wrong findings.

## Key files by construct

### Free energy assembly
- `transformer/core/vfe_utils.py` — KL helpers, F components
- `transformer/vfe/block.py::GaugeTransformerBlock.forward` — block-level F assembly
- `transformer/vfe/e_step.py::EStep` — E-step iteration loop
- `transformer/core/variational_ffn.py` — legacy IFT-style FFN

### Attention (β)
- `transformer/core/attention.py` — legacy attention
- `transformer/vfe/attention.py` — pure VFE attention computing softmax over `−KL/τ`
- `transformer/vfe/block.py` — where β is consumed for residual aggregation

### Transport (Ω, sandwich)
- `transformer/core/transport_ops.py` — Ω construction, transport ops
- `transformer/core/connection.py` — connection forms (incl. opt-in MLP mode)
- `transformer/pure_vfe/gaussians.py` — Gaussian transport with sandwich
- `transformer/vfe/e_step.py` — uses Ω during the E-step

### φ updates
- `transformer/core/phi_evolution.py` — gradient-based φ updates
- `transformer/vfe/e_step.py::_update_phi` — `vfe/`-specific φ retraction with fresh detached leaf

### EM modes
- `transformer/core/em_modes.py` — single source of truth for `em_mode` string semantics
- `transformer/core/block_config.py::BlockConfig.__post_init__` — warning for the silent-freeze trap

### Configs
- `transformer/vfe/config.py` — VFE-package config (BlockConfig, related)
- `transformer/training/config.py` — TrainingConfig
- `transformer/pure_vfe/config.py` — pure VFE config

### Closed-form / diagnostics
- `transformer/core/vfe_closed_form.py` — closed-form analytical references
- `transformer/core/vfe_diagnostics.py` — runtime invariant checks
- `transformer/core/vfe_gradients.py` — gradient utilities

### Active inference / EFE
- `transformer/core/active_inference.py`, `transformer/vfe/active_inference.py`
- `transformer/core/expected_free_energy.py`, `transformer/vfe/efe.py`

### Tests
- `transformer/pure_vfe/tests/test_gauge_transport.py` — sandwich and equivariance tests
- `transformer/pure_vfe/tests/test_mathematical_invariants.py` — invariant-preservation tests
- `transformer/pure_vfe/tests/test_gradients.py` — finite-difference gradient checks
- `transformer/pure_vfe/tests/test_learning.py` — end-to-end learning tests

## Active config locations

The user's "active config" is typically the dict at the top of the entry-point file they're running. Common entry points:

- `transformer/train.py` — top-level training
- `transformer/training/train_fast.py` — fast trainer
- `transformer/vfe/train_vfe.py` — VFE-specific training (currently dirty per `git status`)
- `transformer/vfe/vfe_ablation_suite.py` — ablation runner (currently dirty per `git status`)

The user's CLAUDE.md memory `project_em_config_best_results.md` documents their best legacy config: `layernorm + no residual + skip_attention + n_layers=1; gauge-blind boundary, equivariant interior`.

**Pre-fix protocol reminder:** always open the entry-point file the user is actually running, trace the config dict through any override logic, confirm the modified line is reached. Default-value assumptions are not acceptable.

## Tests to run after audit

```
pytest transformer/pure_vfe/tests/test_gauge_transport.py -v
pytest transformer/pure_vfe/tests/test_mathematical_invariants.py -v
```

These are the fastest theoretical-purity checks. Failure of either is strong evidence of a sandwich bug or an equivariance violation.
