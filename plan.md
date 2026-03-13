# Streamlining Plan for Gauge-Transformer Codebase
 
## Current State: 73,384 lines across ~80 Python files
 
The codebase has two distinct systems that are architecturally isolated:
1. **Multi-Agent FEP System** (agent/, geometry/, gradients/, math_utils/, config.py)
2. **Transformer LM** (transformer/, run_*.py, inference.py, generate.py)
 
They share only a thin interface: `variational_ffn.py` imports from `gradients.gradient_engine` and `geometry.geometry_base`.
 
---
 
## Phase 1: Delete Dead Code (~4,500 lines)
**Risk: ZERO** — nothing references this code.
 
### 1A. Remove backward-compatibility shim files (~200 lines)
Only used by `tests/transformer/test_imports.py` and one archived file.
- `transformer/model.py` (12 lines)
- `transformer/attention.py` (32 lines)
- `transformer/embeddings.py` (20 lines)
- `transformer/variational_ffn.py` (20 lines)
- `transformer/transformer_block.py` (15 lines)
- `transformer/ffn.py` (12 lines — NOT core/ffn.py!)
- `transformer/standard_transformer.py` (12 lines)
- `transformer/prior_bank.py` (12 lines)
- `transformer/rg_metrics.py` (20 lines)
- `transformer/publication_metrics.py` (18 lines)
- `transformer/trajectory_tracking.py` (24 lines)
- Update `test_imports.py` to import from `transformer.core.*` directly, then delete the shims.
 
### 1B. Remove dead math_utils files (~1,750 lines)
- `math_utils/so3_frechet.py` (506 lines) — never imported anywhere
- `math_utils/torch_backend.py` (745 lines) — never imported, incomplete class
- `math_utils/fisher_metric.py` (255 lines) — no external callers found
- Dead generators in `math_utils/generators.py`: `generate_soN_generators()`, `generate_glK_generators()` are only used internally; but keep them — they ARE used via `__init__.py` exports. **Actually keep generators.py intact.**
 
### 1C. Remove archived training scripts (~1,500 lines)
- `transformer/_archive/train_fast.py` (773 lines) — archived, uses old import paths
- `transformer/_archive/train_standard_baseline.py` (719 lines) — archived
- `transformer/_archive/` directory entirely
 
### 1D. Remove inline `__main__` test blocks (~300 lines)
- `transformer/core/blocks.py` lines 672-779 — duplicated by proper tests
- Similar blocks in other core files if present
 
### 1E. Remove stale experimental code if unused (~1,900 lines)
- Check if `transformer/experimental/fep_transformer.py` (1,408 lines) and `transformer/experimental/train_fep.py` (465 lines) are referenced anywhere outside `experimental/`. If not → delete.
 
---
 
## Phase 2: Consolidate Parameter Passing (~net -500 lines, massive readability win)
**Risk: LOW** — behavioral equivalence via dataclass.
 
### The Problem
The `GaugeTransformerBlock.__init__()` takes **40+ keyword arguments**. `GaugeTransformerStack` takes the **exact same 40+ arguments** and just passes them through. `GaugeTransformerLM` (model.py) takes them too and passes them to the Stack. This creates a 3-level waterfall of identical parameter lists (~180 lines of pure parameter forwarding).
 
### The Fix
Create a `@dataclass` (e.g., `BlockConfig`) that bundles all block-level parameters:
 
```python
@dataclass
class BlockConfig:
    embed_dim: int
    hidden_dim: int
    kappa_beta: float
    dropout: float = 0.1
    evolve_sigma: bool = False
    evolve_phi: bool = False
    # ... all 40+ params with defaults
```
 
Then:
- `GaugeTransformerBlock.__init__(self, config: BlockConfig, irrep_spec, generators)`
- `GaugeTransformerStack.__init__(self, n_layers: int, config: BlockConfig, irrep_spec, generators)`
- `GaugeTransformerLM.__init__(self, ...)` constructs `BlockConfig` once, passes it down
 
This eliminates ~120 lines of parameter forwarding and makes adding new features trivial (add to BlockConfig, done).
 
---
 
## Phase 3: Eliminate the GaugeFFN Wrapper (~230 lines → 0)
**Risk: LOW** — it's a pure pass-through.
 
`transformer/core/ffn.py` (230 lines) defines `GaugeFFN` which does nothing but forward all parameters to `VariationalFFNDynamic` and forward all calls. Since `VFE_dynamic` is the only mode:
- Delete `GaugeFFN` class entirely
- Have `blocks.py` instantiate `VariationalFFNDynamic` directly
- Delete `transformer/core/ffn.py`
 
---
 
## Phase 4: Deduplicate run_*.py Scripts (~600 lines saved)
**Risk: LOW** — extract shared utilities.
 
`run_holonomy_study.py`, `run_idiom_study.py`, `run_metaphor_study.py` share ~600-700 lines of identical code:
- Dependency checking (`check_deps()`)
- Helper functions (`hbar()`, `tokenize()`, `fmt_p()`)
- Benjamini-Hochberg FDR correction
- `find_phrase_positions()`, `cross_boundary_positions()`
- Plotting functions (`plot_distributions()`, `plot_layer_profiles()`, etc.)
- Statistical functions (`run_stats()`, `run_permutation_test()`)
 
### The Fix
Create `scripts/study_utils.py` with all shared code. Each `run_*.py` imports from it, keeping only the domain-specific dataset loading and main logic.
 
---
 
## Phase 5: Merge train.py and train_publication.py (or clarify boundary)
**Risk: MEDIUM** — needs careful testing.
 
`train_publication.py` (2,166 lines) imports functions from `train.py` (1,484 lines) and adds:
- CLI argument parsing
- CSV metrics output
- Publication-specific formatting
- Ablation comparison logic
 
These could be unified into one training script with a `--publication` flag, or `train_publication.py` could be slimmed to only the CLI/reporting layer (removing duplicated training logic).
 
**Recommendation**: Keep both files but ensure `train_publication.py` is a thin CLI wrapper that delegates ALL training logic to `train.py`. Remove any duplicated training functions from `train_publication.py`.
 
---
 
## Phase 6: Trim analysis module (~1,000 lines saveable)
**Risk: LOW-MEDIUM** — verify what's actually called.
 
The `transformer/analysis/` directory has 5 large files (~5,000 lines total):
- `rg_metrics.py` (1,232 lines)
- `rg_flow_analysis.py` (1,217 lines)
- `publication_metrics.py` (1,175 lines)
- `semantics.py` (1,082 lines)
- `rg_flow_enhanced.py` (845 lines)
- `rg_dynamic_vs_static.py` (477 lines)
 
`rg_flow_analysis.py` and `rg_flow_enhanced.py` likely overlap heavily with `rg_metrics.py`. Check which functions are actually called from training code vs. being standalone analysis scripts, and consolidate.
 
---
 
## Phase 7: Clean up geometry module for transformer use
**Risk: LOW** — the transformer only uses 2 things from geometry.
 
The transformer LM only imports:
- `geometry.geometry_base.BaseManifold, TopologyType` (used in `variational_ffn.py`)
- `gradients.gradient_engine._compute_agent_euclidean_gradients, project_to_natural_gradients`
- `gradients.retraction.retract_spd`
 
The rest of geometry/ (6,500 lines) and most of gradients/ (4,750 lines) serve the multi-agent system only. No action needed unless you want to split into separate packages.
 
---
 
## Summary
 
| Phase | Lines Removed | Risk | Effort |
|-------|--------------|------|--------|
| 1: Delete dead code | ~4,500 | Zero | Low |
| 2: BlockConfig dataclass | ~500 net | Low | Medium |
| 3: Remove GaugeFFN wrapper | ~230 | Low | Low |
| 4: Deduplicate run_*.py | ~600 | Low | Medium |
| 5: Unify training scripts | ~500 | Medium | High |
| 6: Trim analysis | ~1,000 | Low-Med | Medium |
| 7: Geometry cleanup | 0 (docs) | Low | Low |
| **Total** | **~7,300** | | |
 
**Net result: ~73,400 → ~66,100 lines (10% reduction)** with dramatically improved readability, especially in the core transformer module where the 40-argument waterfall becomes a single config object.
 
### Priority Order
1. Phase 1 (dead code) — immediate, zero risk
2. Phase 3 (GaugeFFN wrapper) — quick win
3. Phase 2 (BlockConfig) — biggest readability improvement
4. Phase 4 (run_*.py dedup) — cleanup
5. Phases 5-6 as time allows
