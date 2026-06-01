# Dead / unreachable code under `skip_attention=True` (LIVE production config)

Audit date: 2026-06-01. Lens: dead/unreachable code when `skip_attention=True`.
Entry point: `transformer/train_publication.py` `EM_CONFIG`. Cross-checked
empirically on torch CPU with a tiny config patterned on EM_CONFIG.

## Classification rule (the discriminating axis)

A path is **STRUCTURALLY-DEAD-under-skip** only if it is reachable *solely*
through the `if not self.skip_attention:` guard in
`GaugeTransformerBlock.forward` (blocks.py:795). These are deletable by a
skip-only refactor: no `skip_attention=True` config can ever execute them.

A path is **DEAD-under-live-only** if it is gated on a *different* knob
(`gauge_mode`, `gauge_param`, `evolve_phi`, `learnable_head_kappa`,
`rope_full_gauge`, `closed_form_e_step`, `active_inference`, `non_flat_transport`)
that merely happens to be off in the live config. Flip that knob with
`skip_attention=True` still set and the path runs — so the refactor cannot
delete it.

## Empirical confirmation (live config, CPU)

A full `model(token_ids)` forward under the live values gives:
`attention.forward` calls = **0**; `precompute_head_transports` calls = **0**;
`attention` param count = **0**; `constant_omega=None`, `output_proj=None`,
`log_kappa_per_head=None`, `gauge_connection=None`, `block_mixer=None`,
`ffn.constant_omega=None`, `ffn._rope_full_gauge_vfe='off'`,
`ffn._ai_enabled=False`, `_kappa_attn_ref` not wired.

Verifying greps (blocks.py):
- `self.norm2` → only 668 (alloc), 983, 989 (both under `not skip_attention`).
- `cached_head_transports` executable refs in `GaugeTransformerBlock.forward`
  (740–1085): 813, 828, 836, 845, 855, 866, 876–905, 920 — ALL inside the skip
  block. Signature param 749 received but never consumed under skip.
- `self.gauge_connection` → call site is ONLY 815 (inside the skip block).

## Findings table

| ID | path:line | What | Classification | Deletable by skip-only refactor? |
|----|-----------|------|----------------|----------------------------------|
| DS-1 | blocks.py:795–975 | Entire `if not self.skip_attention:` block (norm1 pre-norm 800–805, non_flat delta_ij+cached_head_transports build 813–840, omega-path cached build 845–870, shared-BEP build 876–905, attention call 913–921, recorder 924–925, mu residual 958–964, sigma_attn update 974–975) | STRUCTURALLY-DEAD-under-skip | YES |
| DS-2 | blocks.py:668 / refs 983,989 | `self.norm2` module — referenced only under `not skip_attention`. `self.norm1` is the LIVE FFN pre-norm (989 skip-branch). | STRUCTURALLY-DEAD-under-skip (the module) | YES (delete norm2; keep norm1) |
| DS-3 | model.py:572–604 | `cached_head_transports` computation in `_embed_and_prepare`. Output is threaded to `block(...)` (stack fwd 1194/1209) but NEVER consumed under skip — all consumers are inside the skip block. Cross-layer transport cache is fully defeated under skip. | STRUCTURALLY-DEAD-under-skip (the OUTPUT) | The output is dead under skip; the branches inside (572 omega, 593 precompute) are DEAD-under-live (gated on gauge_param/evolve_phi). |
| DS-4 | blocks.py:813,815 + gauge_connection alloc 692/723 | `self.gauge_connection` call site is ONLY 815 (inside skip block). Under skip + `non_flat_transport=True` the connection is allocated but never called → `delta_ij`/`connection_delta` stays None → **non_flat_transport silently no-ops under skip**. Latent bug, not just dead code. | STRUCTURALLY-DEAD-under-skip (call) + latent bug | YES (call). The alloc 676–732 is reachable under non-skip. |
| DS-5 | blocks.py:650–666 | kappa-sharing wiring (`_kappa_attn_ref`, drop FFN log_kappa). Guard `cfg.learnable_head_kappa and attention.log_kappa_per_head is not None` = False under live (`learnable_head_kappa=False` → attention.log_kappa=None, attn.py:1539). | DEAD-under-live-only (gated on `learnable_head_kappa`) | NO — runs under skip + learnable_head_kappa=True |
| DS-6 | blocks.py:566 | `constant_omega=self.attention.constant_omega`. Executes in `__init__`, yields None under live (`gauge_mode='learned'` → attn.py:1633). NOT dead code; deadness is downstream. | DEAD-under-live-only (yields None; FFN consumer 1445 dead) | NO — wiring line runs every config |
| DS-7 | blocks.py:593–595 | `rope_full_gauge_mode` wiring + `ffn._rope_full_gauge_vfe`. Executes in `__init__`; value `'off'` makes downstream branches no-op. | DEAD-under-live-only (gated on `rope_full_gauge`) | NO |
| DS-8 | variational_ffn.py:1980–1995 | `_compute_rope_full_gauge_gradient_per_head` branch. `_use_rope_full` (1965) requires `_rope_full_gauge_vfe in ('vfe_only','both')`; `'off'` → False → fused path 1996 runs. | DEAD-under-live-only (gated on `rope_full_gauge`) | NO |
| DS-9 | variational_ffn.py:1445–1455 | `gauge_mode=='constant' and constant_omega is not None` branch in `_build_block_exp_pairs`. Under live (`learned`, omega=None) dispatch falls to default 1466 (`fused_block_matrix_exp_pairs`). | DEAD-under-live-only (gated on `gauge_mode`) | NO |
| DS-10 | variational_ffn.py:1415–1435 | `omega_current is not None and gauge_param=='omega'` BEP branch. Dead because `gauge_param='phi'` → omega=None. | DEAD-under-live-only (gated on `gauge_param`) | NO |
| DS-11 | variational_ffn.py:2580–2599 | `closed_form_e_step` E-step dispatch. `closed_form_e_step=False` → `_n_iters=self.n_iterations` (2604). | DEAD-under-live-only (gated on `closed_form_e_step`) | NO |
| DS-12 | vfe_closed_form.py:391,521,647 | Picard re-solve (`n_picard_steps`). Dead via TWO independent routes: `n_picard_steps=0` AND requires `closed_form_e_step=True` (file docstring line 57). | DEAD-under-live-only | NO |
| DS-13 | active_inference.py:395–429 (call at variational_ffn.py:2214) | EFE pragmatic/epistemic gradient body. Guard `_ai_enabled and (prag>0 or epi>0)` = False (`active_inference=False` → `_ai_enabled=False`, line 444). `compute_ai_gradients` returns `(None, None)`. Call still executes as a no-op every forward. | DEAD-under-live-only (gated on `active_inference`) | NO |
| DS-14 | model.py:572–592 | omega-path `cached_head_transports` build in `_embed_and_prepare`. `gauge_param='phi'` → omega=None → branch skipped, `else` at 598 → `cached_head_transports=None`. | DEAD-under-live-only (gated on `gauge_param`) | NO |
| DS-15 | model.py:593–597 + attention.precompute_head_transports | `not self.evolve_phi` branch. `evolve_phi=True` → not entered → `precompute_head_transports` NEVER called. | DEAD-under-live-only (gated on `evolve_phi`) | NO; and output is dead under skip anyway (DS-3) |
| DS-16 | attention.py whole `IrrepMultiHeadAttention.forward` + aggregate_messages/transport-build | `.forward()` never called under skip (empirical: 0 calls). Under LIVE the module has 0 params, fully inert. | forward = STRUCTURALLY-DEAD-under-skip; module *allocation* = conditionally live (feeds constant_omega@566 when gauge_mode='constant', log_kappa@650 when learnable_head_kappa=True) | forward YES. Wholesale module deletion only safe if gauge_mode='constant' + learnable_head_kappa support are dropped/relocated. |

## Deletable set (skip-only refactor)

STRUCTURALLY-DEAD (safe to delete for ALL skip_attention=True configs):
DS-1 (the whole `if not skip_attention` block), DS-2 (norm2 module),
DS-4 (gauge_connection call at 815 — but FIX the latent no-op bug, do not just
delete), the `IrrepMultiHeadAttention.forward` body + all message-aggregation
/ transport-build code (DS-16). The model.py `cached_head_transports`
plumbing (DS-3) is dead-output under skip and can be elided on the skip path.

NOT deletable (DEAD-under-live-only — re-enabled by flipping a non-skip knob):
DS-5..DS-15. These belong to other toggles and must stay for the multi-config
surface. Note 566/593–595 are not dead code at all — they run in `__init__`
on every config and merely yield None / set a mode under live values.

## Notes / latent bug

DS-4: `non_flat_transport=True` combined with `skip_attention=True` is a
silent no-op — `gauge_connection` is built but its only call site (815) is in
the bypassed attention block, so `delta_ij` stays None and the FFN receives
`connection_delta=None`. There is no `__post_init__` warning for this
combination (unlike the mixer / detaching-EM warnings). Either route the
connection computation onto the skip path or add a `__post_init__` warning.
