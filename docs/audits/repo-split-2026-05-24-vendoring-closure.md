# Repo Split — `transformer.core` Vendoring Closure for `vfe/` + `aif/`

Date: 2026-05-24
Scope: Exact transitive set of `transformer.core` modules (and symbols) that
`transformer/vfe/**` and `transformer/aif/**` need to run standalone, classified
by extractability. Method: read the actual code (not docstrings/comments), trace
every module-level, function-local, and `TYPE_CHECKING`-gated `from transformer.core`
import to a fixed point.

All claims cite `file:line`. No source file was modified.

---

## 1. Closure summary table

Legacy-stack classes treated as "baggage" (must NOT come along): `GaugeTransformerLM`,
`GaugeTransformerBlock`, `GaugeTransformerStack`, `IrrepMultiHeadAttention`,
`VariationalFFNDynamic`, the legacy `PriorBank`, `GaugeTokenEmbedding`/`GaugePositionalEncoding`,
`connection`, `active_inference`, `vfe_deq`, `vfe_closed_form`, `hebbian`, `embeddings`, `model`.

| core module | symbols vfe/aif need | class | LOC | drags legacy at module-import? |
|---|---|---|---|---|
| `types.py` | `BeliefState` | **A** | 33 | No (zero `transformer.core` imports) |
| `gauge_utils.py` | `fused_block_matrix_exp_pairs`, `stable_matrix_exp_pair` | **A** | 813 | No (only `vfe_utils` + `math_utils`) |
| `vfe_utils.py` | `build_slk_basis`, `safe_eigh_backward`, `retract_sigma_e_step`, `_retract_phi`, (+ `spd_eigfloor`, `_safe_spd_inv`, ceil consts pulled transitively) | **A** | 1283 | No (no `transformer.core` imports; `math_utils` only) |
| `kl_computation.py` | `_kl_kernel_diagonal` (+ kernels used by attention) | **A** | 763 | No (only `gauge_utils`, `vfe_utils`, `math_utils`) |
| `transport_ops.py` | `_apply_rope` (+ rope/cov helpers used by vfe_gradients) | **A** | 584 | No (only `gauge_utils`, `math_utils`) |
| `gauge_preconditioner.py` | `build_cartan_projector`, `build_killing_form_preconditioner`, `build_killing_form_preconditioner_per_block` (+ apply_* used by phi_evolution) | **A** | 683 | No (only `vfe_utils`) |
| `phi_evolution.py` | `precondition_phi_gradient` | **A** | 163 | No (only `gauge_preconditioner`) |
| `vfe_gradients.py` | `compute_vfe_gradients_gpu`, `compute_natural_gradient_gpu`, `_compute_rope_full_gauge_gradient_per_head`, `_fused_attention_and_vfe_gradients_block_diag` | **A** | 2520 | No (only `gauge_utils`, `transport_ops`, `vfe_utils`, `math_utils`) |
| `attention.py` | `compute_attention_weights` (free fn at line 147) | **B** | 2241 | No new legacy *imports*, but the file **defines** `IrrepMultiHeadAttention` (line 1305) at module scope |
| `expected_free_energy.py` | `compute_risk` (line 48) | **B** | 536 | No — legacy `GaugeTransformerLM` is `TYPE_CHECKING`-gated (line 38–39) |
| `blocks.py` | `MahalanobisNorm`, `CenteredMahalanobisNorm`, `RMSNorm` (lines 48–365) | **C** | 1249 | **YES** — module top imports the full legacy stack |
| `block_config.py` | `RopeFullGaugeMode`, `_ROPE_FULL_GAUGE_VALUES` (lines 26–27) | **C** | 682 | **YES** — defines legacy `BlockConfig`, imports `em_modes` |
| `em_modes.py` | (transitive via `block_config`) `EM_MODE_TABLE` | **A** | 30 | No (zero imports; pure dict leaf) |

### Non-core sibling package also in the closure (`math_utils/`)

`math_utils/` is a top-level sibling package (NOT under `transformer.core`) reached
both by closure core modules and directly by vfe. It must be vendored too.

| math_utils module | reached from | LOC | drags anything? |
|---|---|---|---|
| `numerical_monitor.py` | `gauge_utils:15`, `vfe_utils:30`, `kl_computation:31`, `transport_ops:507/513/575/580`, `vfe_gradients:34` | 30 | No (pure leaf) |
| `generators.py` | `attention:69`, `vfe_utils:754`, `vfe/config.py:870`, `vfe/model.py:390`, `vfe/positional.py:73` | 2598 | No (pure leaf) |

Not reached (do NOT vendor): `math_utils/numerical_utils.py`, `math_utils/push_pull.py`,
`math_utils/transport.py` — verified zero references from the closure or from `vfe/`/`aif/`.

---

## 2. Transitive closure list (minimal set of files to vendor)

**`transformer.core` files (13):**
```
transformer/core/types.py
transformer/core/gauge_utils.py
transformer/core/vfe_utils.py
transformer/core/kl_computation.py
transformer/core/transport_ops.py
transformer/core/gauge_preconditioner.py
transformer/core/phi_evolution.py
transformer/core/vfe_gradients.py
transformer/core/attention.py            (B — extract free fns only)
transformer/core/expected_free_energy.py (B — extract compute_risk only)
transformer/core/blocks.py               (C — extract 3 norm classes only)
transformer/core/block_config.py         (C — extract 2 constants only)
transformer/core/em_modes.py             (only needed if block_config copied whole)
```

**`math_utils` files (2):**
```
math_utils/numerical_monitor.py
math_utils/generators.py
```

**If the (B)/(C) extractions are done (recommended cut, see §5):**
the files copied *whole* drop to 9 core + 2 math_utils:
`types, gauge_utils, vfe_utils, kl_computation, transport_ops,
gauge_preconditioner, phi_evolution, vfe_gradients` + a slimmed `attention`
(free-function region only) + `gauge_ridge.py` (99 LOC, pulled in only if the
norm classes are extracted — see §4) + `numerical_monitor, generators`.
`blocks.py`, `block_config.py`, `em_modes.py`, and the legacy half of
`expected_free_energy.py`/`attention.py` are then NOT in the standalone repo.

**Total LOC if copied whole (13 core + 2 math_utils): ≈ 14,210 LOC**
(`11,580` core closure − but note the table sums the *full* files; the three
(B)/(C) files contribute mostly baggage). Core-only whole-file sum = 11,580;
+ math_utils 2,628 = **≈ 14,208 LOC**.

**Estimated LOC after surgical extraction (recommended): ≈ 9,000–9,500 LOC**
(drops `blocks.py` 1249→~320 norms, `block_config.py` 682→2 lines,
`em_modes.py` 30→0, `attention.py` 2241→~1300, `expected_free_energy.py` 536→~65;
adds `gauge_ridge.py` 99).

---

## 3. Direct (level-0) import sites in vfe/aif

Every `from transformer.core` reach, with file:line and exact symbols:

```
vfe/__init__.py:10        types                  BeliefState
vfe/types(model.py:42)    types                  BeliefState
vfe/block.py:20           types                  BeliefState
vfe/block.py:21           blocks                 MahalanobisNorm, CenteredMahalanobisNorm, RMSNorm
vfe/model.py:42           types                  BeliefState
vfe/model.py:43           blocks                 MahalanobisNorm, RMSNorm
vfe/prior_bank.py:56      types                  BeliefState
vfe/prior_bank.py:57      vfe_utils               safe_eigh_backward
vfe/prior_bank.py:58-60   gauge_utils             stable_matrix_exp_pair, fused_block_matrix_exp_pairs
vfe/stack.py:66           types                  BeliefState
vfe/stack.py:67           vfe_utils               safe_eigh_backward
vfe/e_step.py:82          types                  BeliefState
vfe/e_step.py:83-87       vfe_gradients           compute_vfe_gradients_gpu, compute_natural_gradient_gpu,
                                                  _compute_rope_full_gauge_gradient_per_head,
                                                  _fused_attention_and_vfe_gradients_block_diag
vfe/e_step.py:89-91       vfe_utils               retract_sigma_e_step, _retract_phi
vfe/e_step.py:93-97       gauge_preconditioner    build_cartan_projector, build_killing_form_preconditioner,
                                                  build_killing_form_preconditioner_per_block
vfe/e_step.py:98          phi_evolution           precondition_phi_gradient
vfe/e_step.py:194 (defer) kl_computation          _kl_kernel_diagonal
vfe/attention.py:37       gauge_utils             fused_block_matrix_exp_pairs
vfe/attention.py:38       attention               compute_attention_weights
vfe/omega_direct.py:63    gauge_utils             fused_block_matrix_exp_pairs
vfe/non_flat.py:382(defer) gauge_utils            fused_block_matrix_exp_pairs
vfe/non_flat.py:517(defer) transport_ops          _apply_rope
vfe/positional.py:90(defer) vfe_utils             build_slk_basis
vfe/config.py:12-15       block_config            RopeFullGaugeMode, _ROPE_FULL_GAUGE_VALUES
vfe/efe.py:25             expected_free_energy    compute_risk
vfe/trainer.py:640(defer) transport_ops          _apply_rope
aif/tree_search.py:53     types                  BeliefState
aif/training_loss.py:50   types                  BeliefState
aif/efe_score.py:26       types                  BeliefState
aif/belief_cache.py:29    types                  BeliefState
```

`vfe/semantic_clustering/**` has **zero** `transformer.core` imports — it is fully
self-contained inside `transformer.vfe` (verified across all 9 files in the package).

---

## 4. Hard extractions (B-class) — keep vs drop with evidence

### `attention.py` (2241 LOC) — keep free functions, drop the legacy class

vfe needs exactly one symbol: `compute_attention_weights` (`attention.py:147`).
The file's module-level `transformer.core` imports are all primitives already in
the closure (no legacy):
- `attention.py:41` `gauge_utils`
- `attention.py:46` `kl_computation`
- `attention.py:50` `vfe_utils` (`_safe_spd_inv`, `safe_eigh_backward`)
- `attention.py:53` `transport_ops`
- `attention.py:65` `vfe_utils` (`KL_CEIL_BASE`, `KL_CEIL_SCALE`)
- `attention.py:583` (deferred) `kl_computation`

**KEEP** (pure free-function region, lines 104–1304):
`create_attention_mask` (104), `compute_attention_weights` (147),
`compute_kl_matrix_from_phi` (452), `_dispatch_kl_matrix` (535),
`aggregate_messages` (929), and the transport/omega helpers
`compute_transport_operators_direct`, `omega_to_block_exp_pairs` that
`compute_attention_weights` calls internally.

**DROP** legacy baggage: `class IrrepMultiHeadAttention(nn.Module)` at
`attention.py:1305` through end of file (≈ 936 LOC). It is a legacy-model class.
It is referenced only in `__all__` (`attention.py:89`) and comments/docstrings
(`308`, `1418`, `1600`, `1733`, `1741`) — no free function in 104–1304 depends on it.
Note `attention.py:69` has a deferred `from math_utils.generators import generate_so3_generators`
inside the class/docstring example region — already covered by the math_utils closure.

Risk: copying `attention.py` *whole* is safe at runtime (the class def only uses
`nn.Module` + the primitive closure, importing nothing new legacy). The class is
pure dead weight, not a hazard — so this can be left as a "copy whole, prune later"
if surgical extraction is deemed risky.

### `expected_free_energy.py` (536 LOC) — keep `compute_risk`, drop model-coupled fns

vfe needs `compute_risk` (`expected_free_energy.py:48`), which is pure torch on
probability tensors (read lines 48–110: only `torch`/`F`, no `model` arg).

The legacy coupling is `TYPE_CHECKING`-gated and therefore does NOT execute at
import:
- `expected_free_energy.py:38-39`: `if TYPE_CHECKING: from transformer.core.model import GaugeTransformerLM`
- Used only as a string-quoted annotation in `compute_ambiguity`/`compute_epistemic_value`/`compute_efe`
  (`143`, `228`, `304`, `412`) — functions vfe does NOT import.

**KEEP** `compute_risk` (48–110). **DROP** `compute_ambiguity` (112),
`compute_epistemic_value` (140), `compute_efe` (303), and any other `model`-typed
functions, plus the `TYPE_CHECKING` block. Cleanest extraction in the set: copy a
~65-line file containing just `compute_risk` + its imports (`torch`, `torch.nn.functional`).

---

## 5. Clean cut points (C-class) — trivially decoupled

### `blocks.py` (1249 LOC) — DO NOT vendor; re-home 3 norm classes into vfe

This is the single most important cut. `from transformer.core.blocks import MahalanobisNorm`
**executes the full legacy stack at import time** because `blocks.py` module-top does:
- `blocks.py:26` `from transformer.core.attention import IrrepMultiHeadAttention`
- `blocks.py:29` `from transformer.core.variational_ffn import VariationalFFNDynamic`
- `blocks.py:31` `from transformer.core.active_inference import configure_ffn_active_inference`
- `blocks.py:34` `from transformer.core.connection import (...)`
- `blocks.py:23` `from transformer.core.block_config import BlockConfig`

But the three norm classes vfe needs sit ABOVE all legacy class defs and use only
`nn.Module` + two helpers (verified by reading lines 48–365):
- `RMSNorm` (`blocks.py:48`) — pure, `nn.Module` + torch only.
- `MahalanobisNorm` (`blocks.py:74`) — uses `spd_eigfloor` (from `vfe_utils`, in closure;
  `blocks.py:44`) and `make_ridge` (from `gauge_ridge`; `blocks.py:45`). No `BlockConfig`,
  no legacy class refs.
- `CenteredMahalanobisNorm` (`blocks.py:238`) — same two helpers, nothing else.

**Recommended cut:** create `transformer/vfe/_norms.py` containing these three classes
(≈ 320 LOC), importing `spd_eigfloor` from the vendored `vfe_utils` and `make_ridge`
from a vendored `gauge_ridge.py` (99 LOC, pure leaf — `gauge_ridge.py:30-31` imports
only `torch`). Then `blocks.py`, and its entire legacy transitive subtree, leaves the
standalone repo. This removes the biggest entanglement entirely.

### `block_config.py` (682 LOC) — DO NOT vendor; inline 2 constants

vfe needs only `RopeFullGaugeMode` and `_ROPE_FULL_GAUGE_VALUES`
(`vfe/config.py:12-15`), which are defined at `block_config.py:26-27`:
```python
RopeFullGaugeMode = Literal['off', 'vfe_only', 'both']
_ROPE_FULL_GAUGE_VALUES = ('off', 'vfe_only', 'both')
```
Importing them currently drags the legacy `BlockConfig` dataclass (`block_config.py:40`)
and `em_modes` (`block_config.py:33`). Both are pure data (no legacy-model classes),
but they are unnecessary. **Cut:** copy these two lines directly into `vfe/config.py`.
This removes `block_config.py` AND `em_modes.py` from the closure.

---

## 6. Package `__init__.py` side-effect hazard (current state, not a vendoring blocker)

`transformer/core/__init__.py` eagerly imports the ENTIRE legacy stack at package
import:
- `__init__.py:15` `from transformer.core.model import GaugeTransformerLM`
- `:17` `from transformer.core.blocks import GaugeTransformerBlock, GaugeTransformerStack`
- `:18-25` attention (incl. `IrrepMultiHeadAttention`)
- `:26-29` `embeddings`, `prior_bank` (legacy `PriorBank`), `variational_ffn`

So under the CURRENT layout, every `from transformer.core.types import BeliefState`
(the most common vfe/aif import — 6 sites incl. `vfe/__init__.py:10` and all 4 aif files)
first runs `core/__init__.py` and loads the whole legacy stack. `transformer/__init__.py`
similarly eager-imports `GaugeTransformerLM` + training + data (`transformer/__init__.py:32`),
guarded by a broad `try/except ImportError`.

This coupling is **eager-`__init__` coupling, not file-level coupling** — the vendored
standalone package gets its own clean `__init__.py` that imports nothing legacy, so the
hazard evaporates on copy. It is recorded here because it makes the *current* in-tree
import graph look maximally entangled (it is, at runtime) while the *file-level* closure
is clean. Do not be misled by the eager `__init__` when assessing per-file extractability.

---

## 7. Out-of-scope but split-relevant (training/data/plotting reach-through)

The task scoped to `transformer.core`. For completeness: the vfe/aif TRAINING and
ABLATION drivers reach into other `transformer` subpackages that are NOT core
primitives and are NOT part of this closure:
- `vfe/train_vfe.py:17` `transformer.data.datasets`; `:269` `transformer.baselines.flops_counter`
- `vfe/trainer.py:32-33,733,1296,1374-1390,1482,1555,1571` `transformer.training.*`,
  `transformer.analysis.publication_metrics`, `transformer.visualization.*`
- `vfe/vfe_ablation_suite.py:72,573` `transformer.data`, `transformer.baselines`
- `aif/train_aif_augmented.py:35` `transformer.data.datasets`

These are data-loading, metrics, BPC, and figure infrastructure — orthogonal to the
core math closure. The model/E-step/attention path (the part this audit covers) does
NOT touch them. They are a separate vendoring decision (likely "keep the trainer thin
or vendor a small data/metrics shim").

---

## 8. Honest assessment

The vendoring is **clean** at the file level. Eight of thirteen core modules are
**(A) pure primitives** copyable verbatim — the entire numerics/geometry/gradient
spine (`types`, `gauge_utils`, `vfe_utils`, `kl_computation`, `transport_ops`,
`gauge_preconditioner`, `phi_evolution`, `vfe_gradients`) has zero legacy-model
dependence; their only non-core dependency is the already-pure `math_utils`
(`numerical_monitor`, `generators`), both leaves. The closure is sealed: it pulls in
no `transformer.training`/`data`/`analysis`/`visualization` and no other `math_utils`
files.

The two **(B)** files are favorable: `compute_risk`'s only legacy tie is
`TYPE_CHECKING`-gated (never executes), and `attention.py`'s `IrrepMultiHeadAttention`
is inert dead weight that imports nothing new legacy. The two **(C)** files are
trivial: two constants to inline, and three self-contained norm classes to re-home
(needing only `spd_eigfloor` + the 99-LOC pure `gauge_ridge`).

**Single biggest entanglement risk:** `from transformer.core.blocks import MahalanobisNorm`
in `vfe/block.py:21` and `vfe/model.py:43`. `blocks.py` module-top eager-imports the
full legacy stack (`blocks.py:23,26,29,31,34`: `BlockConfig`, `IrrepMultiHeadAttention`,
`VariationalFFNDynamic`, `configure_ffn_active_inference`, `connection`). Copying
`blocks.py` whole would re-import the entire legacy subtree into the "living" repo —
defeating the split. The mitigation is mandatory and easy: extract the three norm
classes (`blocks.py:48-365`) into `vfe/_norms.py` and drop `blocks.py`. This is the one
extraction that, if skipped, silently re-entangles the standalone repo with the frozen
legacy code.
