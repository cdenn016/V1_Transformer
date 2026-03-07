# Codebase Separation Plan: Gauge-Transformer → Two Repositories

## Current State

### The Monorepo (Gauge-Transformer)

The monorepo contains **~170 Python files** across two semi-independent applications unified by a shared gauge-theoretic / variational free energy (VFE) mathematical foundation:

| Codebase | Core Files | Description |
|----------|-----------|-------------|
| **Transformer** | ~60 files | Gauge-theoretic transformer for language modeling |
| **Agent-Based Model** | ~15 files | Multi-agent active inference simulation |
| **Shared Math** | ~40 files | Lie algebra, transport, geometry, gradients, free energy |

### The Existing Fork (VFE-Transformer)

**A transformer-only fork already exists** at `cdenn016/VFE-Transformer`. It was created on **February 26, 2026** by bulk-uploading the Gauge-Transformer codebase and stripping agent-related code (PR #1: "Remove agent-based simulation code"). Since then:

- **VFE-Transformer** has accumulated ~14 PRs: added `SanitizationTracker`, removed positional encoding, removed single-irrep fallback, added `MANUSCRIPT_REVIEW.md`, ran new experiments (`K=70`, `K=60 non-diagonal`).
- **Gauge-Transformer** has accumulated PRs #415-428: added `gauge_preconditioner.py` (Cartan decomposition preconditioning), fixed KL normalization (`2*sqrt(K)` → `sqrt(K)`), fixed VFE sign error, pad masking fix, dead code cleanup.

**The two repos have diverged in both directions.** Neither is a strict subset of the other.

---

## Divergence Inventory

### Gauge-Transformer has, VFE-Transformer does not:
| Item | Significance |
|------|-------------|
| `transformer/core/gauge_preconditioner.py` | **High** — Cartan decomposition preconditioning for gl(K) gradients |
| KL normalization fix (`sqrt(K)` not `2*sqrt(K)`) | **High** — mathematical bug fix |
| VFE dynamic sign error fix | **Medium** — correctness fix |
| Pad masking fix in attention | **Medium** — correctness fix |
| `math_utils/numerical_monitor.py` | Low — debugging utility |
| Dead code cleanup across `variational_ffn.py`, baselines, viz | Low — housekeeping |
| `tests/conftest.py`, `test_vfe_convergence.py` | Low — test infrastructure |
| Legacy compatibility wrappers in `transformer/` root | Low — backward compat |
| `transformer/_archive/` | Low — archived scripts |
| Entire agent system (`agent/`, `meta/`, `geometry/`, `gradients/`, simulation_runner, etc.) | N/A for transformer repo |

### VFE-Transformer has, Gauge-Transformer does not:
| Item | Significance |
|------|-------------|
| `transformer/core/sanitization.py` (`SanitizationTracker`) | **Medium** — tracks numerical sanitization events |
| `Attention/MANUSCRIPT_REVIEW.md` | Low — review document |
| Positional encoding removed (implicit via causal mask) | **Medium** — architectural decision |
| Single-irrep fallback removed (hardcoded `use_multi_irrep=True`) | Low — simplification |
| Early stopping / `use_amp` / `epochs` config removed | Low — simplification |
| New experiment results (`K=70`, `K=60 non-diagonal`) | **Medium** — unique data |

---

## The Core Challenge

Both codebases import heavily from three shared packages in Gauge-Transformer:
- `math_utils/` — Lie generators, transport operators, push-pull, numerical utilities
- `geometry/` — manifolds, connections, curvature, gauge consensus
- `gradients/` — VFE computation, gradient engine, retraction maps

VFE-Transformer kept `math_utils/` but dropped `geometry/` and `gradients/` entirely (the transformer doesn't use them directly — those computations are embedded in the transformer modules).

---

## Target Repositories

| Repository | Name | Contents |
|------------|------|----------|
| **Repo A** | `VFE-Transformer` (existing) | Transformer model, training, analysis, holonomy studies, publications |
| **Repo B** | TBD (`active-inference-agents` or similar) | Multi-agent system, emergence, simulation runner, hierarchical evolution |

---

## Phase 0: Reconcile the Fork

**Goal:** Bring VFE-Transformer up to date with bug fixes and features from Gauge-Transformer before completing the separation.

### 0a. Port Bug Fixes: Gauge-Transformer → VFE-Transformer

These are correctness fixes that VFE-Transformer is missing:

1. **KL normalization fix** — change `2*sqrt(K)` to `sqrt(K)` in attention weight computation. This is a mathematical bug.
2. **VFE dynamic sign error fix** — correct the sign in VFE dynamic computation.
3. **Pad masking fix** — fix pad token masking in attention.
4. **Duplicate dict key in `test_model.py`** — minor test fix.

### 0b. Port New Features: Gauge-Transformer → VFE-Transformer

1. **`gauge_preconditioner.py`** — Cartan decomposition-based gradient preconditioning. Uses Killing form on gl(K) to separate compact (so(K)) vs. non-compact (sym(K)) gradient directions. Mathematically significant for training stability.

### 0c. Port New Features: VFE-Transformer → Gauge-Transformer

1. **`sanitization.py` (`SanitizationTracker`)** — Counts numerical sanitization events (sigma clamping, Cholesky fallbacks, KL clamping). Useful for both codebases.
2. **Evaluate positional encoding removal** — VFE-Transformer dropped positional encoding entirely (position encoded implicitly via causal mask). Decide if this should be the default or remain optional.

### 0d. Decide on Architectural Divergences

| Decision | Options |
|----------|---------|
| Positional encoding | Keep as optional (Gauge-Transformer style) vs. remove entirely (VFE-Transformer style) |
| Single-irrep fallback | Keep for backward compat vs. remove (always multi-irrep) |
| Config simplification | Keep full config vs. strip early-stopping/AMP/epochs |

**Deliverable:** A synchronized VFE-Transformer that has all bug fixes and the gauge preconditioner.

---

## Phase 1: Clean the Monorepo (Gauge-Transformer)

**Goal:** Remove transformer code from Gauge-Transformer, leaving only the agent-based system.

### 1a. Verify What the Agent System Actually Needs

The agent system imports from these shared packages:

| Agent File | Imports From |
|------------|-------------|
| `agent/agents.py` | `geometry.geometry_base`, `math_utils.push_pull`, `math_utils.generators`, `math_utils.sigma` |
| `agent/system.py` | `math_utils.transport`, `math_utils.transport_cache`, `geometry.connection`, `config` |
| `agent/trainer.py` | `gradients.free_energy_clean`, `gradients.gradient_engine`, `gradients.update_engine`, `math_utils.transport_cache`, `config`, `analysis.core.mu_tracking` |
| `agent/hamiltonian_trainer.py` | `gradients.free_energy_clean`, `config`, `geometry.phase_space_tracker`, `geometry.geodesic_corrections`, `geometry.multi_agent_mass_matrix` |
| `meta/emergence.py` | `agent.agents`, `geometry.geometry_base`, `math_utils.transport`, `math_utils.generators`, `math_utils.so3_frechet`, `config`, `agent.masking` |
| `meta/hierarchical_evolution.py` | `meta.emergence`, `gradients.update_engine` |
| `meta/consensus.py` | `math_utils.numerical_utils`, `math_utils.transport` |

### 1b. Remove Transformer-Only Code from Gauge-Transformer

Delete everything the agent system doesn't need:
- `transformer/` — entire directory
- `analysis/holonomy_study/` — transformer-specific linguistic studies
- `run_holonomy_study.py`, `run_idiom_study.py`, `run_metaphor_study.py`, `run_synthesis.py`
- `generate.py`, `inference.py`, `transformer_test.py`
- `analysis_suite.py`
- `Attention/` — manuscript and experiment data (already in VFE-Transformer)
- `Transformer Manuscript/` — same

### 1c. Consolidate Config

- **Merge** root `config.py` and `simulation_config.py` into `agent/config.py` (or keep both under `agent/`).
- Remove any transformer-specific config classes.

### 1d. Prune Shared Math for Agent Use Only

Remove math_utils/geometry/gradients files the agent system doesn't use:

**math_utils/ — keep:**
- `generators.py`, `transport.py`, `transport_cache.py`, `push_pull.py`, `sigma.py`
- `so3_frechet.py`, `numerical_utils.py`, `backend.py`, `torch_backend.py`
- `fisher_metric.py` (if used by gradient engine)
- `numba_kernels.py`, `cuda_kernels.py` (performance kernels)

**math_utils/ — likely remove:**
- `batched_ops.py` (verify if agent uses it)
- `migration.py` (verify)
- `numerical_monitor.py` (verify)

**geometry/ — keep all** (agent uses most of it)

**gradients/ — keep:**
- `free_energy_clean.py`, `gradient_engine.py`, `gradient_terms.py`, `update_engine.py`
- `retraction.py`, `gauge_fields.py`

**gradients/ — likely remove:**
- `softmax_grads.py` (transformer attention-specific)
- `torch_energy.py`, `torch_gradients.py` (verify — may be transformer-only)

### 1e. Verify Agent System Runs Independently

- `simulation_runner.py` must work with the pruned codebase.
- All agent-related tests must pass.

---

## Phase 2: Rename & Restructure Agent Repo

**Goal:** Transform the pruned Gauge-Transformer into a proper agent-based modeling repository.

### 2a. Rename the Repository

Options:
- `active-inference-agents`
- `gauge-agents`
- `vfe-multi-agent`

### 2b. Restructure Directory Layout

```
active-inference-agents/
├── src/
│   └── active_inference/
│       ├── agent/
│       │   ├── agents.py
│       │   ├── system.py
│       │   ├── trainer.py
│       │   ├── hamiltonian_trainer.py
│       │   ├── masking.py
│       │   ├── tensor_agent.py
│       │   ├── tensor_system.py
│       │   └── tensor_trainer.py
│       ├── meta/
│       │   ├── emergence.py
│       │   ├── consensus.py
│       │   ├── spatial_emergence.py
│       │   ├── hierarchical_evolution.py
│       │   └── visualization.py
│       ├── math_core/
│       │   ├── generators.py
│       │   ├── transport.py
│       │   ├── push_pull.py
│       │   └── ...
│       ├── geometry/
│       │   ├── geometry_base.py
│       │   ├── connection.py
│       │   ├── gauge_consensus.py
│       │   └── ...
│       ├── gradients/
│       │   ├── free_energy_clean.py
│       │   ├── gradient_engine.py
│       │   └── ...
│       ├── analysis/
│       │   ├── critical_dynamics.py
│       │   ├── phase_diagrams.py
│       │   └── plots/
│       └── config.py
├── experiments/
│   └── belief_inertia/
├── scripts/
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

### 2c. Packaging

```toml
[project]
name = "active-inference-agents"
dependencies = ["torch>=2.0", "numpy", "scipy", "numba", "matplotlib", "scikit-learn"]
```

### 2d. Documentation

- New README focused on multi-agent active inference.
- Move relevant manuscripts (`Participatory_it_from_bit`, sociology/psychology papers) here.

---

## Phase 3: Archive the Monorepo

**Goal:** Preserve the original Gauge-Transformer as a historical archive.

### Tasks
1. Add a README notice: "This repository has been split into two independent projects: [VFE-Transformer](link) and [active-inference-agents](link)."
2. Mark the repository as archived on GitHub.
3. Do NOT delete it — it preserves the unified git history.

---

## Phase 4: Post-Split Maintenance

**Goal:** Let each codebase evolve independently.

### Tasks
1. **VFE-Transformer:** Optimize math_utils for batched tensor ops, remove NumPy/Numba paths if unused.
2. **Agent repo:** Keep NumPy/Numba paths, add spatial discretization, expand emergence analysis.
3. **Update paper references** to point to the appropriate repo.
4. **CI/CD** for both repos independently.

---

## Execution Order & Risk Assessment

| Phase | Effort | Risk | Notes |
|-------|--------|------|-------|
| 0a. Port bug fixes to VFE-Transformer | **Low** | **Low** | 4 targeted fixes |
| 0b. Port gauge_preconditioner | Low | Low | Single file + integration |
| 0c. Backport sanitization tracker | Low | Low | Single file |
| 0d. Architectural decisions | Low | None | Discussion only |
| 1a. Audit agent dependencies | Low | None | Analysis only |
| 1b. Remove transformer from monorepo | Medium | Low | Deletion is safe with git history |
| 1c. Consolidate config | Low | Low | Straightforward |
| 1d. Prune shared math | Medium | Medium | Must verify nothing breaks |
| 1e. Verify agent system | Medium | None | Testing only |
| 2. Rename & restructure | Medium | Low | Mechanical moves + import updates |
| 3. Archive monorepo | Low | None | GitHub settings |
| 4. Post-split maintenance | Ongoing | Low | Normal development |

**Critical path:** Phase 0 (reconciliation) → Phase 1b-1d (pruning) → Phase 2 (restructure).

The existence of VFE-Transformer **simplifies the plan significantly** — we don't need to fork shared math into two copies within the monorepo. Instead, we fix up VFE-Transformer (Phase 0) and then strip the monorepo down to agent-only (Phase 1).

---

## Open Questions

1. **Repository name for the agent codebase?** — `active-inference-agents`, `gauge-agents`, `vfe-multi-agent`, or something else?
2. **Positional encoding in VFE-Transformer** — keep as optional or remove permanently?
3. **Shared `SanitizationTracker`** — backport to agent system or transformer-only?
4. **New experiment data** in VFE-Transformer (`K=70`, `K=60 non-diagonal`) — does the monorepo need these results archived?
5. **Docs/ directory** was recently deleted from Gauge-Transformer — should manuscripts be restored in the agent repo?
6. **Git history strategy for agent repo** — fresh repo (clean) vs. `git filter-repo` (preserve history)?
