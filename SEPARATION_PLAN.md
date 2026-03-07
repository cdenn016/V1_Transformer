# Codebase Separation Plan: Gauge-Transformer → Two Repositories

## Current State

The monorepo contains **~170 Python files** across two semi-independent applications unified by a shared gauge-theoretic / variational free energy (VFE) mathematical foundation:

| Codebase | Core Files | Description |
|----------|-----------|-------------|
| **Transformer** | ~60 files | Gauge-theoretic transformer for language modeling |
| **Agent-Based Model** | ~15 files | Multi-agent active inference simulation |
| **Shared Math** | ~40 files | Lie algebra, transport, geometry, gradients, free energy |

### The Core Challenge

Both codebases import heavily from three shared packages:
- `math_utils/` — Lie generators, transport operators, push-pull, numerical utilities
- `geometry/` — manifolds, connections, curvature, gauge consensus
- `gradients/` — VFE computation, gradient engine, retraction maps

These aren't thin wrappers — they are the mathematical heart of both systems. The separation strategy must handle this shared dependency cleanly.

---

## Target Repositories

| Repository | Working Name | Contents |
|------------|-------------|----------|
| **Repo A** | `gauge-transformer` | Transformer model, training, analysis, holonomy studies, publications |
| **Repo B** | `active-inference-agents` | Multi-agent system, emergence, simulation runner, hierarchical evolution |

---

## Phase 0: Audit & Dependency Mapping (Pre-work)

**Goal:** Build a precise import graph so nothing is missed.

### Tasks
1. **Generate a full import graph** — script that walks every `.py` file and records all internal imports (e.g., `from math_utils.transport import ...`). Output an adjacency list.
2. **Classify every file** into one of four buckets:
   - `T` — Transformer only
   - `A` — Agent-based model only
   - `S` — Shared (used by both T and A)
   - `D` — Dead code (imported by neither)
3. **Identify cross-contamination** — any file in `T` that imports from `A` or vice versa (these are the coupling points that must be severed).
4. **Catalog external dependencies** — which PyPI packages each codebase actually uses (to split `requirements.txt`).

### Deliverable
A `dependency_report.json` mapping every file to its bucket and listing all inbound/outbound internal imports.

---

## Phase 1: Internal Decoupling (Still One Repo)

**Goal:** Make the two codebases independently runnable within the monorepo before physically splitting.

### 1a. Isolate `config.py`

Currently `config.py` contains `SystemConfig`, `AgentConfig`, `TrainingConfig` — used by both codebases.

- **Create** `transformer/config.py` containing transformer-specific config (training hyperparams, model architecture).
- **Create** `agent/config.py` containing agent/simulation-specific config (`AgentConfig`, `SimulationConfig`).
- **Keep** a minimal shared config if truly needed (e.g., `math_utils/config.py` for numerical precision settings), otherwise eliminate it.
- **Migrate** `simulation_config.py` → `agent/simulation_config.py`.
- **Update** all imports accordingly.

### 1b. Resolve Cross-Imports Between Transformer and Agent

Identify and eliminate any direct imports between `transformer/` and `agent/`. Likely candidates:
- `transformer/analysis/rg_metrics.py` — RG analysis is used by both codebases. Fork it: one copy for transformer layer analysis, one for agent emergence analysis.
- `meta/` emergence code — if the transformer ever references meta-agent concepts, break that link.

### 1c. Fork the Shared Math into Two Copies

This is the most significant step. For each file in `math_utils/`, `geometry/`, and `gradients/`:

1. Determine if it's used by **T only**, **A only**, or **both**.
2. Files used by only one codebase → move into that codebase's directory.
3. Files used by both → **duplicate** into both codebases (one copy each).

**Proposed new structure within the monorepo:**

```
transformer/
  math_core/          # ← transformer's copy of shared math
    generators.py
    transport.py
    push_pull.py
    ...
  geometry/           # ← transformer's copy
    connection.py
    ...
  gradients/          # ← transformer's copy
    free_energy_clean.py
    ...

agent/
  math_core/          # ← agent's copy of shared math
    generators.py
    transport.py
    push_pull.py
    ...
  geometry/           # ← agent's copy
    connection.py
    geometry_base.py
    gauge_consensus.py
    ...
  gradients/          # ← agent's copy
    free_energy_clean.py
    gradient_engine.py
    ...
```

**Why duplicate instead of a shared package?**
- A shared pip-installable package adds release/versioning overhead for a research codebase.
- The two codebases will likely diverge — the transformer doesn't need spatial manifolds; the agent system doesn't need token embeddings or attention heads.
- Duplication now, prune later. Each repo can delete the parts it doesn't use.

### 1d. Verify Independence

- **Test A:** Run `transformer/train.py` with **only** `transformer/` on `sys.path`. It must work.
- **Test B:** Run `simulation_runner.py` with **only** `agent/` on `sys.path`. It must work.
- All existing tests must pass in both configurations.

---

## Phase 2: Prune Each Codebase

**Goal:** Remove dead weight from each copy of the shared math.

### Transformer Side — Remove Agent-Only Code
- `geometry/geometry_base.py` — remove spatial manifold support (periodic, sphere topologies). Transformer uses a trivial 0D "point manifold."
- `geometry/gauge_consensus.py` — multi-agent consensus not needed.
- `geometry/multi_agent_mass_matrix.py` — not needed.
- `geometry/phase_space_tracker.py` — agent-specific.
- `gradients/update_engine.py` — agent gradient application; transformer uses PyTorch autograd.
- `math_utils/numba_kernels.py` — only used by NumPy-based agent trainer.
- `math_utils/cuda_kernels.py` — same.
- `meta/` — entire directory (hierarchical emergence is agent-only).

### Agent Side — Remove Transformer-Only Code
- `transformer/` — entire directory.
- `analysis/holonomy_study/` — transformer-specific.
- `run_holonomy_study.py`, `run_idiom_study.py`, `run_metaphor_study.py` — transformer-specific.
- `generate.py`, `inference.py` — transformer-specific.
- `transformer/data/datasets.py` — WikiText loading, not needed.
- `transformer/visualization/` — attention heatmaps, not needed.

### Both Sides
- Remove any dead code identified in Phase 0.
- Clean up `__init__.py` files to only export what's actually used.

---

## Phase 3: Repository Setup

**Goal:** Create two independent Git repositories with proper packaging.

### 3a. Create New Repositories

```
gauge-transformer/              active-inference-agents/
├── src/                        ├── src/
│   └── gauge_transformer/      │   └── active_inference/
│       ├── core/               │       ├── agent/
│       │   ├── model.py        │       │   ├── agents.py
│       │   ├── attention.py    │       │   ├── system.py
│       │   ├── blocks.py       │       │   ├── trainer.py
│       │   ├── embeddings.py   │       │   ├── hamiltonian_trainer.py
│       │   └── ffn.py          │       │   └── masking.py
│       ├── math_core/          │       ├── math_core/
│       ├── geometry/           │       ├── geometry/
│       ├── gradients/          │       ├── gradients/
│       ├── analysis/           │       ├── meta/
│       ├── visualization/      │       ├── analysis/
│       ├── data/               │       └── visualization/
│       ├── experimental/       ├── configs/
│       └── baselines/          ├── scripts/
├── studies/                    ├── experiments/
│   ├── holonomy/               ├── tests/
│   ├── idiom/                  ├── pyproject.toml
│   └── metaphor/               ├── requirements.txt
├── docs/                       └── README.md
│   └── manuscripts/
├── tests/
├── scripts/
├── pyproject.toml
├── requirements.txt
└── README.md
```

### 3b. Packaging

Each repo gets its own `pyproject.toml`:

**gauge-transformer:**
```toml
[project]
name = "gauge-transformer"
dependencies = ["torch>=2.0", "tiktoken", "datasets", "scipy", "matplotlib", "scikit-learn"]
```

**active-inference-agents:**
```toml
[project]
name = "active-inference-agents"
dependencies = ["torch>=2.0", "numpy", "scipy", "numba", "matplotlib", "scikit-learn"]
```

### 3c. Git History

Two options:

| Option | Pros | Cons |
|--------|------|------|
| **A. Fresh repos** (copy files, new git init) | Clean history, small repos | Lose blame/history |
| **B. `git filter-repo`** (split history) | Preserve full history per file | Complex, large repos |

**Recommendation:** Option A (fresh repos) for a research codebase. The original monorepo remains as the historical archive.

### 3d. Documentation

- Each repo gets its own README with installation, usage, and theory overview.
- Move relevant manuscripts/docs to the appropriate repo.
- `Docs/attention manuscript/` → gauge-transformer
- `Docs/Participatory_it_from_bit/` → active-inference-agents (or both)

### 3e. CI/CD

- Each repo gets its own test suite and CI pipeline.
- Transformer: `pytest tests/` should cover model, training, analysis.
- Agent: `pytest tests/` should cover agent dynamics, emergence, simulation.

---

## Phase 4: Post-Split Cleanup

**Goal:** Let each codebase evolve independently.

### Tasks
1. **Simplify imports** — now that there's no shared root, flatten import paths where it makes sense.
2. **Diverge the math** — each repo can now specialize its copy of the math utilities:
   - Transformer: optimize for batched tensor ops, remove NumPy paths.
   - Agent: keep NumPy/Numba paths, add spatial discretization.
3. **Update references** — if papers or docs reference the monorepo, add notes pointing to the two new repos.
4. **Archive the monorepo** — mark it read-only with a README pointing to the two successors.

---

## Execution Order & Risk Assessment

| Phase | Effort | Risk | Notes |
|-------|--------|------|-------|
| 0. Audit | Low | None | Pure analysis, no code changes |
| 1a. Config split | Low | Low | Straightforward refactor |
| 1b. Cross-imports | Medium | Medium | Must not break either codebase |
| 1c. Fork shared math | **High** | **High** | Most files, most imports to update |
| 1d. Verify | Medium | None | Testing only |
| 2. Prune | Medium | Low | Deleting code is safe if tests pass |
| 3. Repo setup | Medium | Low | Mechanical file moves |
| 4. Cleanup | Low | Low | Polish |

**Total estimated phases:** 4 phases, with Phase 1c being the critical path.

---

## Open Questions

1. **Shared math package vs. duplication?** — If you anticipate wanting synchronized math updates across both repos, a third `gauge-math` pip package may be worth the overhead. For research velocity, duplication is simpler.
2. **Which repo keeps the original name `Gauge-Transformer`?** — Recommend the transformer repo keeps it.
3. **Manuscript/docs split** — Some documents span both codebases. Decide per-document.
4. **Results directories** (`Transformer Manuscript/`, `Attention/`, etc.) — Archive in the transformer repo or a separate data archive?
