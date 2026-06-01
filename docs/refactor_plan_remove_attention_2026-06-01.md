# Refactor plan — hardcode `skip_attention=True`, remove the attention sublayer

Date: 2026-06-01
Backbone: audit memo `07_refactor-blast-radius.md` + verifier `B_deadcode_refactor.md`,
with the BR-1 hazard escalated by verifier `A_correctness_baseline.md`.
Companion audit: `docs/audit_skip_attention_ift_phi_2026-06-01.md`.

Goal: make `skip_attention=True` the only behavior and remove the
`IrrepMultiHeadAttention` sublayer instantiation from `GaugeTransformerBlock`,
without changing the training behavior of any `skip_attention=True` config.

## What "remove the attention sublayer" does and does not mean

In scope: the `self.attention = IrrepMultiHeadAttention(...)` instantiation in
`GaugeTransformerBlock`, the `if not self.skip_attention:` branch in
`GaugeTransformerBlock.forward`, the block-level wiring that reads attributes off
`self.attention`, and the `model.py` cross-layer transport-cache machinery that
only `attention.forward` consumes.

Out of scope (do NOT delete): `transformer/core/attention.py` as a module. Its
module-level functions (`compute_attention_weights`, `compute_kl_matrix_from_phi`,
`aggregate_messages`, `compute_transport_operators`) are imported and used by the
VFE loss (`train.py:30,513`) and by the FFN E-step. The `IrrepMultiHeadAttention`
class itself is still constructed standalone by tests
(`test_attention.py`, `test_equivariant_head_mixer.py`) and by
`transformer/baselines/hybrid_gauge_transformer.py`. Removing the *sublayer
instantiation* is not the same as deleting the *class*.

## Scope decision (needs user sign-off)

Three levels, increasing aggressiveness:

- Level A — hardcode the default + assert. Set `skip_attention=True` as the
  hardcoded behavior, keep the attention instantiation guarded so non-skip configs
  still run, add an assertion/deprecation. Smallest change; leaves all dead
  buffers (PERF-4) and the whole non-skip branch in place. Does not satisfy "get
  rid of the attention sub-layer completely."

- Level B (recommended) — stop instantiating the sublayer in the block, re-home
  the handful of live dependencies, delete the `if not self.skip_attention:`
  branch and the `model.py` cache machinery, keep the `IrrepMultiHeadAttention`
  class and all `attention.py` module functions. The block becomes a pure VFE
  E-step block. This is the natural reading of "remove the sub-layer completely"
  and keeps the test/baseline surface (category-b tests, hybrid baseline, VFE-loss
  imports) intact.

- Level C — Level B plus deleting the `IrrepMultiHeadAttention` class, the
  attention-only config fields, and desupporting the two `skip_attention=False`
  scripts. Largest blast radius; breaks `hybrid_gauge_transformer.py` and the
  standalone attention tests. Only worth it if the attention comparison is truly
  abandoned.

Recommendation: Level B. The two `skip_attention=False` entry points
(`scripts/run_optuna_hpo.py:103`, `scripts/verify_active_inference.py:103`) are
attention-comparison runs; under Level B they would need `skip_attention=True` or
explicit desupport. Per CLAUDE.md ("there must always exist a pure path; extreme
paths are opt-in"), the attention sublayer is the non-pure engineering heuristic,
so removing it from the default block is consistent with the thesis — but the
attention *class* is the opt-in comparison artifact and should survive.

## Re-homing table

| ID | What lives on `self.attention` today | Live under skip? | Decision |
|----|--------------------------------------|------------------|----------|
| BR-1 | `constant_omega` (`nn.ParameterList`, registered on attention, borrowed by FFN via `__dict__`) | Yes, under `gauge_mode='constant'` — verified trained (grad 7.8e-3/5.2e-3 through the FFN E-step even with skip) | FFN becomes the registered owner; add explicit optimizer routing; checkpoint key remap |
| BR-2 | per-head `log_kappa` sharing (`blocks.py:650-666`) | Only under `learnable_head_kappa=True` | Delete the redirect block; FFN's own `log_kappa_per_head` is sole owner (routing-neutral) |
| BR-3 | `irrep_dims` (`blocks.py:690,693,827,854`; `model.py:573`) | `:690,693` in `__init__` under `non_flat_transport`; rest dead-output under skip | Compute block-level `self.irrep_dims` from `cfg.irrep_spec`; re-point `:690,693`; delete the skip-block and `model.py:573` sites |
| BR-4 | `precompute_head_transports` (`model.py:594`) | Dead-output under skip | Delete the `model.py:593-597` call; keep the method on the class (used by `hybrid_gauge_transformer.py:489,567`) |
| BR-5 | `rope_full_gauge_mode` assignment (`blocks.py:595`) | `'off'` live; matters only for mode `'both'` | Drop the assignment; keep the FFN side (`blocks.py:594`); document the mode-`'both'` attention-side σ-rotation loss |

### BR-1 detail (the one dangerous re-home)

Today `constant_omega` is created inside `IrrepMultiHeadAttention`
(`attention.py:1623-1633`) only when `gauge_mode='constant'`, and the FFN borrows
the same object via `self.__dict__['constant_omega'] = constant_omega`
(`variational_ffn.py:435`) to avoid double-registration. The attention module is
the sole registered owner.

Naive removal failure mode (verified): with the attention module gone,
`blocks.py:566` passes `None`, the FFN falls back to identity transport
(`variational_ffn.py:1457-1464`), and `constant_omega` vanishes from the optimizer
— Ω frozen at identity, silent wrong math (not a crash). Verifier A confirmed the
parameter is genuinely trained under `skip_attention=True` + `gauge_mode='constant'`
(nonzero grad through the FFN E-step), so this is a real behavior change, not a
harmless curiosity.

Fix:
1. Promote `variational_ffn.py:435` from a `__dict__` borrow to a real
   `nn.ParameterList` constructed inside the FFN when `gauge_mode=='constant'`
   (the FFN already has `irrep_dims`, `enforce_orthogonal`, device handling). Mirror
   the per-head 10×10 init from `attention.py:1623-1633`.
2. Delete `blocks.py:566` (no attention to source from).
3. Optimizer routing (`transformer/training/optimizer.py::create_param_groups`):
   today `attention.constant_omega.*` matches the `'attention'` catch-all
   (`:750`) → `M_attention_lr=0.013`. The re-homed `ffn.constant_omega.*` matches
   none of the earlier rules → final `else` (`:759`) → `ffn` group @
   `M_vfe_hyperparam_lr=0.095` — a silent ~7.3× LR change. Add an explicit
   `constant_omega` rule (mirror the `log_kappa` rule, placed before the
   `'attention'` catch-all) and decide the intended LR deliberately.
4. Checkpoint key migration: keys move `...attention.constant_omega.*` →
   `...ffn.constant_omega.*`; a `strict=True` load breaks. Provide a load-time
   key remap for `gauge_mode='constant'` checkpoints.

If the user only ever runs `gauge_mode='learned'`, BR-1 is moot for their config —
but the fix is cheap insurance and is required for Level B to be behavior-preserving
across the multi-config surface.

## Edits, in dependency order

1. `transformer/core/variational_ffn.py`
   - Register `constant_omega` as a real `nn.ParameterList` in `__init__` when
     `gauge_mode=='constant'` (BR-1.1). Keep the `__dict__` borrow path removable.
   - Remove the dead `cached_block_exp_pairs` parameter from `_compute_phi_grad`
     (`:1191`, PERF-1b cleanup) and its pass-site (`:2390`).

2. `transformer/core/blocks.py`
   - Delete the `self.attention = IrrepMultiHeadAttention(...)` instantiation
     (`:483-516`).
   - Delete `constant_omega=self.attention.constant_omega` (`:566`); the FFN now
     owns it.
   - Delete the κ-redirect block (`:650-666`, BR-2); the FFN's `log_kappa_per_head`
     stays registered.
   - Delete `self.attention.rope_full_gauge_mode = _rope_mode` (`:595`, BR-5); keep
     `self.ffn._rope_full_gauge_vfe = _rope_mode` (`:594`).
   - Add `self.irrep_dims` computed from `cfg.irrep_spec` (expand each
     `(label, mult, dim)` to `mult` copies of `dim`) in `__init__` (BR-3); re-point
     the `PerHeadGaugeConnection` ctor reads (`:690,693`) to it.
   - Delete `self.norm2` allocation (`:668`, EM-2) and collapse the norm handling so
     the single VFE sublayer always uses `norm1`.
   - Delete the entire `if not self.skip_attention:` branch in `forward`
     (`:795-975`, DS-1). The forward becomes: norm1 → FFN E-step → (optional block
     mixer) → σ update → optional residual.
   - Remove the `self.skip_attention` attribute and the `skip_attention` branching;
     hardcode the pure-VFE path.
   - Fix DS-4: when `non_flat_transport=True`, compute `delta_ij` on the (now sole)
     pre-FFN path and pass `connection_delta=delta_ij`, OR add a `__post_init__`
     warning if non-flat-under-pure-VFE is intentionally unsupported. Do not break
     the offline `publication_metrics.py:2201` direct call to the connection module.

3. `transformer/core/model.py`
   - Delete the cross-layer cache machinery in `_embed_and_prepare` (`:571-604`,
     DS-3 / C-2): the omega branch (`:572-592`) and the `precompute_head_transports`
     branch (`:593-597`) are dead-output under skip (the block forward never
     forwards `cached_head_transports` to the FFN). Set `cached_head_transports=None`
     unconditionally (or remove the threading entirely). `gauge_param='omega'` still
     works because `omega` threads to the FFN independently.
   - Remove the `blocks[0].attention.*` dereferences (`:573,594`).

4. `transformer/core/block_config.py`
   - Hardcode/remove `skip_attention` (Level B: drop the field, or default True +
     assert). Prune the now-vestigial `__post_init__` warnings (`:287-330`: the
     detaching-mode × skip warning and the `use_equivariant_head_mixer` × skip
     warning).
   - Mark attention-only fields dead: `use_output_projection`,
     `use_equivariant_head_mixer`, `attention_pattern`, `attention_window`,
     `alibi_slope`. Removing `use_output_projection` (the `nn.Linear` W_O) is a
     CLAUDE.md NN-constraint improvement. Keep `mask_self_attention` and
     `kappa_beta` — they are consumed by the FFN E-step, not attention-only.

5. `transformer/training/optimizer.py` + `transformer/training/config.py`
   - Add the explicit `constant_omega` routing rule (BR-1.3). `M_attention_lr`
     becomes unused for this model (still referenced as harmless config plumbing by
     `experiment_runner.py`, `resume_training.py`); leave or annotate.

6. Entry-point configs
   - `scripts/run_optuna_hpo.py:103` and `scripts/verify_active_inference.py:103`
     set `skip_attention: False` → set to True or desupport (Level B/C boundary).
   - `train_publication.py`, `run_ablation_suite.py` already `skip_attention=True`
     — the hardcoded `skip_attention` key becomes a no-op; leave or remove.

7. Tests
   - Category (a) break on sublayer removal (access `block.attention.*` on a built
     model): `test_cross_head_coupling.py:202,214,376`;
     `test_equivariant_head_mixer.py:395,397,399`;
     `test_per_head_gauge_connection.py:313,390,398,407,436`;
     `test_block_equivariant_mixer.py:252`. Update or remove these.
   - Category (b) survive iff the class is kept (Level B): `test_attention.py`
     module-function tests, `test_equivariant_head_mixer.py:54,214,272,309`.

## Behavior-preserving equivalence gate (the success criterion)

The refactor is a no-op for `skip_attention=True`. The gate is concrete:

1. Primary (learned gauge): `python docs/audit_workspace/equivalence_harness.py --gate`
   must pass (`atol=1e-6, rtol=1e-5` on loss + every per-param grad norm, pinned
   weights). Caveat from verifier A: under `gauge_mode='learned'` the attention
   sublayer is never called, so this gate passes trivially for the removal and does
   not protect BR-1.

2. Companion (constant gauge — the true BR-1 gate, must be added before the
   `constant_omega` re-home): capture a `gauge_mode='constant'` + `skip_attention=True`
   baseline (extend the harness: `cfg['gauge_mode']='constant'`), then after the
   re-home assert (a) `constant_omega.{0,1}.grad` stay nonzero, (b) the loss + grad
   norms match the baseline to tolerance, and (c) the optimizer assigns
   `constant_omega` the intended LR group.

3. Structural diffs: `named_parameters()` set and `optimizer.param_groups` (names +
   LRs) identical before/after for both `gauge_mode='learned'` and
   `gauge_mode='constant'`, except the intended `constant_omega` key migration
   (which the explicit routing rule must keep at its intended LR).

4. `pytest tests/transformer/` green (after updating category-a tests), including
   the gradient finite-difference smoke tests.

## Naive-removal failure checklist (the silent breakers)

1. `constant_omega` vanishes from the optimizer under `gauge_mode='constant'` → Ω
   frozen at identity (BR-1). Re-register on the FFN.
2. Re-homing `constant_omega` silently moves its LR 0.013 → 0.095 (BR-1.3). Add the
   routing rule.
3. `strict=True` checkpoint load breaks on the key migration (BR-1.4). Provide a
   remap.
4. `model.py:573` / `:594` dereference `blocks[0].attention` → AttributeError under
   `gauge_param='omega'` / `evolve_phi=False`. Delete those sites (dead-output under
   skip) and confirm the omega path still threads `omega` to the FFN.
5. `non_flat_transport=True` + pure-VFE silently no-ops (DS-4) unless δ_ij is routed
   onto the skip path or a warning is added.
6. Category-a tests and the two `skip_attention=False` scripts crash unless updated.

## Follow-on opportunities (separate PRs, not part of the no-op refactor)

- PERF-1: analytic ∂F/∂φ via the envelope identity to remove the 2nd
  `fused_block_matrix_exp_pairs` and β rebuild. Requires a `dexp`/Fréchet
  implementation and an analytic-vs-autograd ~1e-5 check before adoption. Hedge the
  win below the memo's 55% headline.
- PERF-3: enable `causal_lower_triangle=True` after verifying bit-equivalence on the
  live config.
- E-2: implement `use_autograd_mu_sigma` (the manuscript's pure ∇F_red path) or
  amend the manuscript.
- F01-D / F01-G / NUM-1 / E-1: documentation and a `gauge_group` hardening one-liner.
