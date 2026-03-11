# Plan: Separate Agent Simulation from Transformer Codebase

## Goal
Split `Gauge-Transformer` into two independent repositories:
1. **`Gauge-Transformer`** — The transformer language model only
2. **`Gauge-Agents`** (or similar) — The multi-agent active inference simulation only

Both repos share a substantial mathematical foundation (`math_utils/`, `geometry/`, `gradients/`, `meta/`). The key decision is how to handle this shared code.

---

## Phase 0: Extract Shared Math into a Library Package

The biggest entanglement is the shared mathematical foundation used by both systems. This must be resolved first.

### Create `gauge-math` (shared library package)

**Files to include:**
- `math_utils/` — transport, push_pull, generators, sigma, fisher_metric, so3_frechet, numba_kernels, torch_backend, cuda_kernels, batched_ops, migration, transport_cache
- `geometry/` — geometry_base, connection, lie_algebra, geodesic_corrections, pullback_metrics, phase_space_tracker, multi_agent_mass_matrix, gauge_consensus
- `gradients/` — free_energy_clean, gradient_engine, gradient_terms, torch_gradients, torch_energy, softmax_grads, retraction, update_engine
- `meta/` — emergence, consensus, gradient_adapter, hierarchical_evolution, energy_visualization

**Package structure:**
```
gauge-math/
├── pyproject.toml
├── gauge_math/
│   ├── __init__.py
│   ├── transport.py          (from math_utils/)
│   ├── push_pull.py
│   ├── generators.py
│   ├── sigma.py
│   ├── fisher_metric.py
│   ├── so3_frechet.py
│   ├── numba_kernels.py
│   ├── torch_backend.py
│   ├── cuda_kernels.py
│   ├── batched_ops.py
│   ├── migration.py
│   ├── transport_cache.py
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── geometry_base.py
│   │   ├── connection.py
│   │   ├── lie_algebra.py
│   │   ├── geodesic_corrections.py
│   │   ├── pullback_metrics.py
│   │   ├── phase_space_tracker.py
│   │   ├── multi_agent_mass_matrix.py
│   │   └── gauge_consensus.py
│   ├── gradients/
│   │   ├── __init__.py
│   │   ├── free_energy_clean.py
│   │   ├── gradient_engine.py
│   │   ├── gradient_terms.py
│   │   ├── torch_gradients.py
│   │   ├── torch_energy.py
│   │   ├── softmax_grads.py
│   │   ├── retraction.py
│   │   └── update_engine.py
│   └── meta/
│       ├── __init__.py
│       ├── emergence.py
│       ├── consensus.py
│       ├── gradient_adapter.py
│       ├── hierarchical_evolution.py
│       └── energy_visualization.py
```

**Decision point:** You can either:
- **(A) Publish `gauge-math` as a pip-installable package** (private PyPI, GitHub Packages, or just `pip install -e .` from a local clone) — cleanest long-term
- **(B) Use git submodules** — both repos include `gauge-math` as a submodule — simpler to start but submodules are annoying
- **(C) Copy the shared code into both repos** — simplest but creates divergence risk

**Recommendation:** Option A. Create a third repo `gauge-math`, publish it (even privately), and have both consumer repos depend on it via `pyproject.toml`.

---

## Phase 1: Extract `config.py`

`config.py` contains `AgentConfig`, `SystemConfig`, and `TrainingConfig` used by both systems.

### Steps:
1. Split `config.py` into:
   - **Shared configs** → move to `gauge-math` (e.g., `AgentConfig`, `SystemConfig` — the mathematical/structural params)
   - **Transformer-specific configs** → stays in transformer repo (e.g., model architecture, vocab size, dataset config)
   - **Simulation-specific configs** → goes to agent repo (e.g., `SimulationConfig`, grid topology, preset configs)
2. Move `simulation_config.py` entirely to agent repo

---

## Phase 2: Create the Transformer Repo

### Files to keep in `Gauge-Transformer`:
```
Gauge-Transformer/
├── pyproject.toml             (depends on gauge-math)
├── transformer/
│   ├── core/
│   │   ├── model.py
│   │   ├── attention.py
│   │   ├── blocks.py
│   │   ├── ffn.py
│   │   ├── variational_ffn.py
│   │   ├── embeddings.py
│   │   └── prior_bank.py
│   ├── analysis/
│   │   ├── rg_metrics.py
│   │   └── rg_flow_enhanced.py
│   ├── data/                  (dataset loading)
│   ├── training/              (training config & metrics)
│   ├── visualization/
│   ├── train.py
│   └── train_publication.py
├── generate.py
├── inference.py
├── config.py                  (transformer-specific only)
└── tests/transformer/
```

### Import rewrites needed:
- `from math_utils.transport import ...` → `from gauge_math.transport import ...`
- `from geometry.geometry_base import ...` → `from gauge_math.geometry.geometry_base import ...`
- `from gradients.free_energy_clean import ...` → `from gauge_math.gradients.free_energy_clean import ...`
- `from meta.emergence import ...` → `from gauge_math.meta.emergence import ...`

---

## Phase 3: Create the Agent Simulation Repo

### Files to move to `Gauge-Agents`:
```
Gauge-Agents/
├── pyproject.toml             (depends on gauge-math)
├── agent/
│   ├── agents.py
│   ├── system.py
│   ├── trainer.py
│   ├── hamiltonian_trainer.py
│   ├── tensor_agent.py
│   ├── tensor_system.py
│   ├── tensor_trainer.py
│   └── masking.py
├── simulation_runner.py
├── simulation_config.py
├── config.py                  (agent-specific only)
├── analysis/
│   └── holonomy_study/
├── run_holonomy_study.py      (*)
├── run_idiom_study.py         (*)
├── run_metaphor_study.py      (*)
├── run_synthesis.py           (*)
└── tests/agent/
```

**(*)** Note: The `run_*_study.py` scripts may import from `transformer/` to run language analysis. These scripts are a coupling point — they use the transformer as a trained model to analyze linguistic phenomena via agent-based holonomy metrics. You'll need to decide:
- Move them to the agent repo and have them load a pre-trained transformer checkpoint (model weights only, no transformer code dependency)
- Move them to a separate `experiments/` repo that depends on both
- Keep them in whichever repo makes more sense for your workflow

### Import rewrites needed:
- Same pattern as transformer: `math_utils.*` → `gauge_math.*`, `geometry.*` → `gauge_math.geometry.*`, etc.

---

## Phase 4: Update Tests

1. **`gauge-math`**: Move/create tests for the shared mathematical functions (VFE convergence, transport correctness, gradient verification)
2. **`Gauge-Transformer`**: Keep transformer-specific tests
3. **`Gauge-Agents`**: Keep agent simulation tests
4. Verify each repo's test suite passes independently

---

## Phase 5: Clean Up Cross-References

1. Update all `README.md` files to reference the new repo structure
2. Update `claude.md` architecture guide
3. Update any CI/CD configurations
4. Archive or redirect the original monorepo

---

## Execution Order (Recommended)

| Step | Action | Risk |
|------|--------|------|
| 1 | Create `gauge-math` package with shared code, add `pyproject.toml`, verify imports work | Low — additive only |
| 2 | Split `config.py` into shared / transformer / agent configs | Medium — touches many imports |
| 3 | Fork repo → strip agent code → rewrite imports → `Gauge-Transformer` | Medium — many import changes |
| 4 | Fork repo → strip transformer code → rewrite imports → `Gauge-Agents` | Medium — many import changes |
| 5 | Decide on `run_*_study.py` scripts (coupling point) | Decision needed |
| 6 | Run full test suites on all three repos | Verification |
| 7 | Update docs, CI, archive monorepo | Low |

---

## Key Risks & Mitigations

1. **Import breakage**: Use `sed` or a script to bulk-rename imports. Write a test that imports every module to catch missing rewrites.
2. **Diverging shared code**: If using option A (pip package), version pin `gauge-math` in both consumer repos. Use semver.
3. **`run_*_study.py` coupling**: These scripts bridge both worlds. Best handled as a separate experiments repo or by making them load pre-trained checkpoints without importing transformer code directly.
4. **Config entanglement**: `config.py` is the messiest file to split. Do it carefully — audit every field's usage before deciding where it goes.
