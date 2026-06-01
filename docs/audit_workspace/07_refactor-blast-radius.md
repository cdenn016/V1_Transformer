# Audit memo 07 — Attention-removal blast radius + parameter-ownership map

LENS: hardcode `skip_attention=True` and remove the `IrrepMultiHeadAttention`
sublayer instantiation from `GaugeTransformerBlock`. This memo is the refactor
plan input. Every claim is grounded in `file:line` read from the actual code
(not docstrings) and, where it settles a question, an empirical CPU check.

Date: 2026-06-01. Env: torch CPU only. Repo root
`C:/Users/chris and christine/Desktop/VFE_1.0`.

## Executive framing

Under the live production config (EM_CONFIG, `transformer/train_publication.py`
lines ~118-318) the attention sublayer contributes **zero optimizer-visible
parameters** — empirically `NUM attention requires_grad params = 0` with
`gauge_mode='learned'`, `use_output_projection=False`,
`use_equivariant_head_mixer=False`, `learnable_head_kappa=False`. So removing the
sublayer under the live config loses nothing from the optimizer and the forward
still runs (verified). That fact makes removal *trivially safe for the live
config*, which is the least interesting part of this memo.

The value of the memo is the four code paths that are dead under the live config
but **load-bearing under other in-repo configs**. "Dead under live config" means
"must be explicitly re-homed or desupported", NOT "safe to delete":

1. `constant_omega` ownership (only registered under `gauge_mode='constant'`).
2. per-head `log_kappa` sharing (only under `learnable_head_kappa=True`).
3. `attention.irrep_dims` consumed by `model.py` omega/`non_flat` branches.
4. `attention.precompute_head_transports` (the `evolve_phi=False` cache path).

One finding changes a conclusion: re-homing `constant_omega` from the attention
module to the FFN **silently changes its optimizer learning-rate group** (it is
not a no-op). Details in BR-1.

Do NOT delete `transformer/core/attention.py`. Its module-level functions
(`compute_attention_weights`, `compute_kl_matrix_from_phi`, `aggregate_messages`,
`compute_transport_operators`) are consumed by the VFE loss and by the FFN. Only
the `IrrepMultiHeadAttention` *sublayer instantiation* in `blocks.py` is in
scope, plus the class itself iff nothing else builds it (tests still do — see
BR-9).

## Where the sublayer is referenced (the full reference set)

`blocks.py`:
- `__init__`: instantiation `self.attention = IrrepMultiHeadAttention(...)`
  (blocks.py:483-516).
- `__init__`: `constant_omega=self.attention.constant_omega` passed to FFN
  (blocks.py:566).
- `__init__`: `self.attention.rope_full_gauge_mode = _rope_mode` (blocks.py:595).
- `__init__` kappa-sharing block: reads `self.attention.log_kappa_per_head`,
  deletes FFN's own params, installs `ffn.__dict__['_kappa_attn_ref'] =
  self.attention` (blocks.py:650-666).
- `__init__` non_flat per-head connection: `self.attention.irrep_dims`
  (blocks.py:690, 693).
- `forward` non_flat E-step transport build: `self.attention.irrep_dims`
  (blocks.py:827).
- `forward` direct-omega cached transports build: `self.attention.irrep_dims`
  (blocks.py:854).
- `forward` shared-BEP build / attention dispatch:
  `self.attention.gauge_mode`, `self.attention.irrep_dims`,
  `self.attention.enforce_orthogonal`,
  `self.attention._generators_are_skew` (blocks.py:877-884), and the actual
  `self.attention(...)` call (blocks.py:913-921). All of this is inside
  `if not self.skip_attention:` (blocks.py:795) and is dead when skip is on.

`model.py`:
- `_embed_and_prepare`, omega branch: `irrep_dims =
  self.transformer.blocks[0].attention.irrep_dims` (model.py:573).
- `_embed_and_prepare`, `not evolve_phi` branch: `first_attention =
  self.transformer.blocks[0].attention;
  first_attention.precompute_head_transports(...)` (model.py:594-595).

`variational_ffn.py`:
- `self.__dict__['constant_omega'] = constant_omega` — FFN holds a `__dict__`
  reference (not a registered child) to the attention module's ParameterList
  (variational_ffn.py:435).
- `_get_kappa_h` reads through `self.__dict__['_kappa_attn_ref']`
  (variational_ffn.py:716-719).

`train.py` (the VFE-loss module, NOT the optimizer):
- imports `compute_attention_weights` from `attention` and calls it for the
  gamma / model-coupling term (train.py:30, 513). This is module-level and must
  survive.
- kappa diagnostic reads `vffn.log_kappa_per_head` first, falls back to
  `vffn._kappa_attn_ref` (train.py:760-766).

`transformer/training/optimizer.py`:
- `create_param_groups` routes by name substring; `...attention...` →
  `'attention'` group @ `config.M_attention_lr` (optimizer.py:826-834).

`transformer/training/config.py`:
- `M_attention_lr` default and doc "(W_O, constant_omega)" (config.py:60),
  `non_embed_weight_decay` (config.py:95). These are the real owners of the LR
  the task attributed to "train.py".

## Live-config consequence verification (empirical)

Built `GaugeTransformerLM` with the live EM_CONFIG values on CPU
(`gauge_group='GLK'`, `gauge_dim=10`, `irrep_spec=[('fund',2,10)]`, n_layers=1,
skip_attention=True, em_mode='ift_phi', gauge_mode='learned', gauge_param='phi',
diagonal_covariance=True, learnable_head_kappa=False, use_output_projection=False,
use_equivariant_head_mixer=False). Observed:

- `attn.constant_omega == None` (gauge_mode='learned'). Verifies the task's
  "gauge_mode=learned makes constant_omega=None".
- `attn.output_proj == None` (use_output_projection=False).
- `attn.irrep_dims == ffn.irrep_dims == [10, 10]` (True).
- `ffn._kappa_attn_ref not in ffn.__dict__` (learnable_head_kappa=False → the
  kappa-sharing block at blocks.py:650-666 never runs). `_get_kappa_h` returns
  `self.kappa` immediately (variational_ffn.py:709-710).
- `NUM attention requires_grad params = 0`. Forward produces logits `(2,16,128)`.

Under `gauge_mode='constant'` (NOT live, but the dangerous case):
- `named_parameters()` contains ONLY
  `transformer.blocks.0.attention.constant_omega.0/.1` (10×10 each).
- `ffn.constant_omega is attention.constant_omega` is True (same object via
  `__dict__`), but the FFN copy is invisible to `named_parameters()`.

Under `use_block_diagonal_kl=False`:
- `ffn.irrep_dims == None` while `attn.irrep_dims == [10,10]`, and the VFE FFN
  RAISES `RuntimeError: VariationalFFNDynamic requires irrep_dims to be set`
  (variational_ffn.py:2201-2203). So no *running* VFE config has
  `ffn.irrep_dims=None`, but `attention.irrep_dims` and `ffn.irrep_dims` are NOT
  guaranteed equal in general.

## Re-homing decisions (the refactor plan)

### BR-1 (CRITICAL) — `constant_omega`: who owns it after removal

Today the per-head `nn.ParameterList` `constant_omega` is **created inside the
attention module** (`attention.py:1623-1633`) and only when
`gauge_mode=='constant'`. The FFN borrows it via `__dict__` to avoid
double-registration (variational_ffn.py:435; comment at 431-434). The attention
module is the sole *registered* owner — under `gauge_mode='constant'`,
`named_parameters()` shows only `...attention.constant_omega.*`.

Naive removal failure mode: delete the attention sublayer and
`self.attention.constant_omega` no longer exists; the FFN's `__dict__` reference
becomes `None` (blocks.py:566 would pass `None`). Result: the FFN's transport
falls back to identity (variational_ffn.py:1458-1460, "constant without
constant_omega: fall back to identity"), and `constant_omega` **vanishes from the
optimizer — frozen at the identity init forever**. This is a silent
mathematically-wrong outcome (Ω≡I, no learned transport), not a crash.

Decision: the FFN must become the real *registered* owner. Promote
variational_ffn.py:435 from a `__dict__` reference to a real `nn.ParameterList`
constructed inside the FFN when `gauge_mode=='constant'` (it already has
`irrep_dims`, `enforce_orthogonal`, device handling). Then blocks.py:566 is
deleted (no attention to source from). Two consequences that MUST be carried:

(a) **Optimizer LR group silently changes.** `create_param_groups` routes by
name (optimizer.py:711-760). Empirically verified routing:
`transformer.blocks.0.attention.constant_omega.0` → `'attention'` group @
`M_attention_lr` (=0.013, the catch-all `'attention' in name` at optimizer.py:750;
the `'omega_embed'` rule at 736 does NOT match `'constant_omega'`).
`transformer.blocks.0.ffn.constant_omega.0` → falls through to the final `else`
(optimizer.py:759) → `'ffn'` group @ `M_vfe_hyperparam_lr`. So the re-home
**silently moves the learning rate** from `M_attention_lr` to
`M_vfe_hyperparam_lr` (weight decay stays `non_embed_wd` either way). Fix: add an
explicit routing rule for `constant_omega` (mirror the `log_kappa` rule at
optimizer.py:747, placed before the `'attention'` catch-all), or name the new
parameter so it hits the `omega_embed` rule. Pick one deliberately and document
the intended LR.

(b) **state_dict key migration.** Keys move from
`...attention.constant_omega.*` to `...ffn.constant_omega.*`. Any checkpoint
load with `strict=True` breaks (`tests/transformer/test_checkpoint.py`). Provide
a key-remap on load.

Confidence: high (sole-ownership and the identity fallback are read directly;
routing and ownership both verified empirically).

### BR-2 — per-head `log_kappa` sharing: who registers it exactly once

Today, when `learnable_head_kappa=True`, both the attention module
(attention.py:560-565 region, `log_kappa_per_head` nn.Parameter +
`_kappa_init` buffer) and the FFN (variational_ffn.py:560-568) create their own.
blocks.py:650-666 then DELETES the FFN's copy from `self.ffn._parameters` /
`self.ffn._buffers` and installs `ffn.__dict__['_kappa_attn_ref'] =
self.attention`, so the attention module's parameter is the single source of
truth (the `__dict__` bypass at blocks.py:661 is what prevents
double-registration — without it the attention module would auto-register as a
child of the FFN and its params would be double-counted).

Under the live config `learnable_head_kappa=False`, this whole block is dead
(verified: `_kappa_attn_ref` absent). But the refactor must keep
`learnable_head_kappa=True` working.

Decision: delete the entire kappa-redirect block (blocks.py:650-666). The FFN's
own `log_kappa_per_head` (variational_ffn.py:564) becomes the sole registered
owner — exactly once, no `__dict__` bypass needed because there is no longer a
second module to dedup against. `_get_kappa_h` already falls back to
`self.log_kappa_per_head` when `_kappa_attn_ref` is absent
(variational_ffn.py:716-722), so it keeps working unchanged. Routing is
unaffected: `ffn.log_kappa_per_head` hits the `'log_kappa'` rule
(optimizer.py:747) BEFORE the `'attention'` catch-all and lands in `no_decay`,
exactly as `attention.log_kappa_per_head` does today (verified empirically — both
route to `no_decay`). This is the clean asymmetry vs BR-1: kappa re-homes
routing-neutral, `constant_omega` does not.

Cleanup: train.py:760-766 reads `vffn.log_kappa_per_head` first and only falls
back to `_kappa_attn_ref` — the fallback branch (train.py:763-766) goes
permanently dead and can be dropped.

Confidence: high.

### BR-3 — `attention.irrep_dims` re-home (blocks.py:690,693,827,854 + model.py:573)

`irrep_dims` (the per-head block dimension list, e.g. `[10,10]`) is computed
inside `IrrepMultiHeadAttention` from `irrep_spec`. The FFN ALSO has its own
`self.irrep_dims` (variational_ffn.py:510, set from `cfg.ffn_irrep_dims`).

Caveat (verified): they are equal in any *running* VFE config because
`use_block_diagonal_kl=False` makes `ffn.irrep_dims=None` and the VFE FFN then
RAISES at variational_ffn.py:2201. But `attention.irrep_dims` stays populated
independently. To stay robust and independent of `use_block_diagonal_kl`, re-home
to a **block-level value derived from `cfg.irrep_spec`** (expand each
`(label, mult, dim)` to `mult` copies of `dim`), stored once on the block in
`__init__`. Then:
- blocks.py:690,693 (non_flat PerHeadGaugeConnection) → block `self.irrep_dims`.
- blocks.py:827 (non_flat E-step transport split) → block `self.irrep_dims`.
- blocks.py:854 (direct-omega cached transports) → block `self.irrep_dims`.
- model.py:573 (omega branch) → `self.transformer.blocks[0].irrep_dims` (or a
  model-level value from `irrep_spec`).

All five are dead under the live config (`gauge_param='phi'` kills the omega
branches; `non_flat_transport=False` kills the connection branch) but live under
`gauge_param='omega'` / `non_flat_transport=True`.

Confidence: high.

### BR-4 — `precompute_head_transports` (model.py:594-595): new home

The `evolve_phi=False` cache path calls
`blocks[0].attention.precompute_head_transports(phi, device, dtype)`
(model.py:594-595; method at attention.py:2182-2233). It builds per-head
transport dicts; for `gauge_mode=='constant'` it reads
`self.constant_omega[head_idx]` (attention.py:2212), otherwise it calls
`compute_transport_operators` on per-head generators
(attention.py:2228-2231). It needs: `n_heads`, `irrep_dims`, `gauge_mode`,
`enforce_orthogonal`, `constant_omega`, and per-head `head_generators`.

Decision: re-home as a method on the FFN (or a free function in `transport_ops`
taking explicit args). The FFN has `irrep_dims`, `gauge_mode`,
`enforce_orthogonal`, and (after BR-1) `constant_omega`; the only missing piece
is `head_generators` (the per-head block-partitioned generators). The FFN already
partitions generators on the fly via `fused_block_matrix_exp_pairs(phi,
generators, irrep_dims)` in its main loop (variational_ffn.py:2742-2744), so the
re-homed precompute can do the same — partition `generators` by `irrep_dims` and
build per-head BEP, rather than depending on the attention module's pre-stored
`head_generators` buffers.

Dead under live config (`evolve_phi=True` → model.py:604 `else` branch sets
`cached_head_transports=None`, verified by reading the dispatch at
model.py:572-604). Live under `evolve_phi=False`.

Confidence: high for the requirement; medium on the cleanest implementation
surface (free function vs FFN method) — both work.

### BR-5 — `rope_full_gauge_mode` (blocks.py:595)

`self.attention.rope_full_gauge_mode = _rope_mode` is set so the attention-side
σ rotation fires for mode `'both'`. With the sublayer gone this assignment has no
target. Under the live config `rope_full_gauge='off'`, so nothing fires anyway.
Decision: drop the assignment. The FFN side
(`self.ffn._rope_full_gauge_vfe = _rope_mode`, blocks.py:594) is the path that
matters for the E-step and is retained. For mode `'both'`, the attention-side σ
rotation is simply not available once the sublayer is removed — note this as a
documented behavior reduction (the FFN-side VFE σ rotation still applies for
`{'vfe_only','both'}`).

Confidence: high.

## Config-field disposition

Truly attention-only → become dead once the sublayer is gone:
- `use_output_projection` (W_O linear; attention.py:1638-1642). DEAD. Also a
  CLAUDE.md NN-constraint surface (nn.Linear) — its removal is a constraint
  *improvement*.
- `use_equivariant_head_mixer` (attention.py:1661-1675; force-disabled under
  skip_attention already, blocks.py:505-508). DEAD.
- `attention_pattern` (block_config.py:77; "'full' only supported"). DEAD.
- `attention_window` (block_config.py:78; doc says "unused, kept for API
  compat"). ALREADY dead. 
- `alibi_slope` (block_config.py:461,663; passed only to the attention module).
  DEAD for transport once sublayer removed.

NOT attention-only (do NOT treat as dead — correction to the task's list):
- `mask_self_attention`: consumed by the FFN E-step (`mask_self_attention=
  cfg.mask_self_attention` at blocks.py:553) and by `model._build...`. Keep.
- `kappa_beta`: flows to `ffn_kappa=kappa_beta` (block_config.py:622), the FFN
  softmax temperature. Keep.

`block_config.py __post_init__` skip_attention-combination warnings
(block_config.py:287-330) become vestigial once `skip_attention` is hardcoded
True — the `_detaching_modes` × `skip_attention` warning and the
`use_equivariant_head_mixer` × `skip_attention` warning can be pruned in a
follow-up.

## Param-group / optimizer wiring (correction to the task pointer)

The "M_attention_lr param group" lives in
`transformer/training/optimizer.py::create_param_groups` (optimizer.py:826-834),
NOT in train.py. It is gated by `if attention_params:` (optimizer.py:826), so
with the sublayer gone the list is empty and the group is simply skipped — no
crash. `M_attention_lr` (training/config.py:60) and the EM_CONFIG key
(train_publication.py:240) become unused for this model (still referenced by
`experiment_runner.py:1963` and `resume_training.py:476` as config plumbing —
harmless). The one real consequence is BR-1(a): `constant_omega`'s LR group
changes when it re-homes from `attention.*` to `ffn.*`.

`create_simple_param_groups` (optimizer.py:869-903, used when
`use_param_groups=False`) routes by `'embed'` vs other and is unaffected.

## Scope decision — configs that set skip_attention=False (they BREAK)

These run the attention sublayer and would break if the sublayer is removed
without a compatibility shim. List them as the scope boundary; the audit TARGET
is the `skip_attention=True` production path:
- `scripts/run_optuna_hpo.py:103` — `skip_attention: False`.
- `scripts/verify_active_inference.py:103` — `skip_attention: False`.
(`scripts/run_ablation_suite.py:101` uses `skip_attention: True` — compatible.)

Decision options: (i) hard-remove the sublayer and explicitly desupport these two
scripts (update them to `skip_attention=True` or delete the
attention-comparison runs); or (ii) keep `IrrepMultiHeadAttention` as a class and
only remove the *block instantiation* under a now-hardcoded skip, gating the old
behavior behind a legacy flag. Given CLAUDE.md ("there must ALWAYS exist a pure
path; extreme paths are opt-in") the attention sublayer is the *non*-pure
engineering heuristic, so removing it from the default block and desupporting
those two attention-comparison scripts is consistent with the thesis.

## Test blast radius (16 files touch attention symbols)

Split by failure mode:

(a) Access `block.attention.*` through a built model → BREAK on sublayer
removal:
- `tests/transformer/test_cross_head_coupling.py:202,214,376`
  (`model.transformer.blocks[0].attention`).
- `tests/transformer/test_block_equivariant_mixer.py:252`
  (`block.attention._mixer_active`).
- `tests/transformer/test_equivariant_head_mixer.py:393-399`
  (`block.attention._mixer_active`, `block.attention.mixer_params`).

(b) Construct `IrrepMultiHeadAttention(...)` standalone → SURVIVE iff the class
is kept:
- `tests/transformer/test_attention.py:289,304` and the module-function tests
  throughout that file.
- `tests/transformer/test_equivariant_head_mixer.py:54,214,272,309`.

(c) Reference attention-only config fields (`use_output_projection`) in
config-shape assertions → audit individually:
- `tests/transformer/test_equivariant_head_mixer.py:236-244` asserts the default
  `use_output_projection is False` — survives.
- `tests/transformer/test_use_output_projection_false_fullcov.py`,
  `tests/transformer/test_use_output_projection_false.py`,
  `scripts/smoke_use_output_projection_false.py` — these assert the
  *skip_attention=True / W_O-off* path already; should survive and in fact
  become the canonical path.

Memo recommendation: keep the `IrrepMultiHeadAttention` class and its
module-level functions; remove only the sublayer instantiation + the five
re-home sites above. That keeps category (b) green and isolates breakage to the
small category (a) set, which should be updated or deleted alongside the two
`skip_attention=False` scripts.

## active_inference confirmation

`configure_ffn_active_inference(self.ffn, cfg)` (blocks.py:614) is called
unconditionally but its EFE machinery is gated on `active_inference=True`. Live
config has `active_inference=False`, so it sets the `_ai_*` attributes inert and
does not touch the attention sublayer. Not in scope; confirmed independent of
this refactor.

## Naive-removal failure checklist (the spots that silently break)

1. (BR-1) `constant_omega` vanishes from the optimizer under
   `gauge_mode='constant'` → Ω frozen at identity, silent wrong math. MUST
   re-register on the FFN.
2. (BR-1a) Re-homing `constant_omega` to `ffn.*` silently moves its LR from
   `M_attention_lr` to `M_vfe_hyperparam_lr` via name-routing. MUST add an
   explicit routing rule or rename.
3. (BR-1b) state_dict keys move `attention.constant_omega.*` →
   `ffn.constant_omega.*`; strict checkpoint load breaks.
4. (BR-4) `precompute_head_transports` has no home → `evolve_phi=False` configs
   lose the transport cache (AttributeError on `blocks[0].attention`).
5. (BR-3) `model.py:573` omega branch dereferences
   `blocks[0].attention.irrep_dims` → AttributeError for `gauge_param='omega'`.
6. Two `skip_attention=False` scripts and category-(a) tests crash unless the
   class is retained or they are updated.

None of (1)-(6) fire under the live config; all fire under at least one in-repo
config. The double-counting risk the task asked about (kappa) is currently
prevented by the `__dict__` bypass (blocks.py:661); after BR-2 there is no second
module so no dedup is needed and double-counting cannot occur.
