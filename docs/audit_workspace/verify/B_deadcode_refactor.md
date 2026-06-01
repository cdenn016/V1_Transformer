# Verifier B — dead-code + refactor-blast-radius adversarial confirmation

Scope: independently confirm/refute memo `06_dead-code-under-skip.md` and
`07_refactor-blast-radius.md` by reading actual control flow (path:line) under
the LIVE production config and grounding with a tiny CPU model. torch 2.11.0,
CPU only, device forced cpu. Tiny config required `gauge_group='GLK',
gauge_dim=10` in addition to the verifier brief's dict (else
`generate_multi_irrep_generators` rejects even-dim 'fund' as non-SO(3)).

Date: 2026-06-01.

## Bottom line

Both memos are substantially CORRECT. Every structural-deadness and re-home
claim I checked holds at the cited path:line and reproduces empirically. Four
overstatements / wording corrections below; none invalidates a deletion or a
re-home decision. The headline correction is to the BR-3/BR-4 framing: under a
skip-hardcoded refactor, the model.py cache machinery is **dead-output and
should be DELETED, not re-homed**, because the block forward never forwards
`cached_head_transports` to the FFN.

## (1) STRUCTURALLY-DEAD-under-skip set — CONFIRMED

The `if not self.skip_attention:` guard is `blocks.py:795`; the block ends at
the FFN call boundary `blocks.py:977` (the `# 2. VFE E-step` comment). I read
the whole 795–975 span.

- Entire `if not self.skip_attention:` block (DS-1): CONFIRMED dead under skip.
  norm1 pre-norm 800–805, non_flat δ_ij + cached build 813–840, omega-path
  cached build 845–870, shared-BEP build 876–905, recorder 907–908, attention
  call 913–921, mu residual 958–964, sigma_attn update 974–975 — all inside the
  guard. Empirically `attn.forward calls = 0` over a full `model(ids)`.
- `self.norm2` (DS-2): independent grep in blocks.py returns EXACTLY 668 (alloc),
  983, 989 — both refs gated on `not self.skip_attention` (983 ternary, 989
  ternary). Under skip the active norm is `self.norm1` (983/989 else-branch).
  CONFIRMED: norm2 deletable, norm1 kept.
- `IrrepMultiHeadAttention.forward` body (DS-16): reachable only via the
  `self.attention(...)` call at blocks.py:913. Empirically 0 calls. CONFIRMED.
- gauge_connection CALL at 815: see (4) — the call at 815 is the sole call in
  the block forward, but it is NOT the sole call site project-wide.

Empirical live-config probe (tiny GLK model, skip_attention=True):
`attn.forward=0`, `precompute=0`, `attn.constant_omega=None`,
`attn.output_proj=None`, `attn.log_kappa_per_head=None`, `ffn.constant_omega=None`,
`ffn._rope_full_gauge_vfe='off'`, `_kappa_attn_ref` absent, `gauge_connection=None`,
`block_mixer=None`, `_ai_enabled=False`, `attn.irrep_dims==ffn.irrep_dims==[10,10]`,
attention requires_grad params = 0, logits (2,16,128). Matches memo 06 line 23–28
exactly.

## (2) NOT-deletable (DEAD-under-live-only) set — CONFIRMED

Each is gated on a knob other than skip_attention; flipping that knob with
skip_attention=True still set runs the path. Read sites + empirical flips:

- kappa-sharing (DS-5, blocks.py:650–666): gated on `learnable_head_kappa`.
  Empirical: `learnable_head_kappa=True` + skip → attn.log_kappa is a Param and
  `ffn._kappa_attn_ref` IS installed. NOT structurally dead.
- constant_omega wiring (DS-6, blocks.py:566): runs in `__init__` every config;
  yields None only because `gauge_mode='learned'`. CONFIRMED not dead code.
- rope_full_gauge (DS-7 blocks.py:593–595; DS-8 variational_ffn.py:1965–1995):
  `_use_rope_full` requires `_rope_full_gauge_vfe in ('vfe_only','both')` AND
  `_use_rope_vfe` AND `_nonflat_omega is None` AND not inference-mode
  (variational_ffn.py:1965–1970). `'off'` → False → fused path 1996/2000.
  Gated on `rope_full_gauge`, not skip. CONFIRMED.
- gauge_mode=='constant' BEP branch (DS-9, variational_ffn.py:1445–1455):
  reached when constant_omega present; live falls through to fused 1467.
  Empirical: `gauge_mode='constant'` + skip RUNS. CONFIRMED.
- gauge_param=='omega' branch (DS-10, variational_ffn.py:1415–1435): empirical
  `gauge_param='omega'` + skip RUNS. CONFIRMED.
- closed_form / picard (DS-11 variational_ffn.py:2580; DS-12 vfe_closed_form.py):
  `_n_iters = 0 if closed_form_e_step else n_iterations` (2604). Gated on
  `closed_form_e_step`. CONFIRMED.
- active_inference (DS-13, call variational_ffn.py:2214): `compute_ai_gradients`
  returns (None,None) when off; `_ai_enabled=False` under live. Gated on
  `active_inference`. CONFIRMED — call still executes as no-op every forward.

## (3) Re-home requirements BR-1..BR-5 — CONFIRMED with one framing correction

BR-1 (constant_omega ownership) — CONFIRMED, including the LR-group hazard.
- Sole REGISTERED owner is the attention module: empirically under
  `gauge_mode='constant'`, `named_parameters()` shows ONLY
  `...attention.constant_omega.0/.1`; the FFN holds the SAME object via
  `__dict__` (`ffn.constant_omega is attn.constant_omega` → True,
  variational_ffn.py:435). Identity fallback when constant_omega is None:
  variational_ffn.py:1457–1464 (`constant without constant_omega: fall back to
  identity`). So naive removal → Ω≡I, frozen, silent wrong math. CONFIRMED.
- BR-1(a) LR-group move — CONFIRMED by reading optimizer.py:711–760 and
  reproducing the name-router: `omega_embed`@736 does NOT match `constant_omega`
  (substring 'omega_embed' absent); `attention`@750 catch-all matches
  `attention.constant_omega.*` → `attention` group @ M_attention_lr (829); the
  FFN-owned `ffn.constant_omega.*` matches NONE of the earlier rules → final
  `else`@759 → `ffn` group @ M_vfe_hyperparam_lr (839). The re-home silently
  moves the LR. MUST add an explicit routing rule. CONFIRMED.

BR-2 (log_kappa sole registration) — CONFIRMED and routing-neutral.
- attention.py log_kappa alloc is at 1533–1540 (NOT "560–565" as memo 07 states;
  see correction C-1). FFN's own copy at variational_ffn.py:560–568.
  `_get_kappa_h` falls back to `self.log_kappa_per_head` when `_kappa_attn_ref`
  absent (variational_ffn.py:716–722). Routing: both `attention.log_kappa_per_head`
  and `ffn.log_kappa_per_head` hit `log_kappa`@747 → no_decay (checked BEFORE the
  `attention` catch-all). Re-home is routing-neutral. CONFIRMED.

BR-3 (irrep_dims → block-level from irrep_spec) — CONFIRMED but SPLIT (see
correction C-2). The five cited sites read `self.attention.irrep_dims` /
`blocks[0].attention.irrep_dims`:
- blocks.py:690,693 (PerHeadGaugeConnection ctor, in `__init__`, gated on
  `non_flat_transport`): runs in `__init__` REGARDLESS of skip — genuinely needs
  the block-level re-home.
- blocks.py:827 (non_flat E-step transport split) and 854 (omega cached build):
  inside the skip block → dead-output under skip, but live under non-skip.
- model.py:573 (omega branch): dead-output under skip — see C-2, this site is
  DELETABLE under a skip-hardcoded refactor, not a required re-home.
Empirically `attn.irrep_dims == ffn.irrep_dims == [10,10]`.

BR-4 (precompute_head_transports new home) — PARTIALLY OVERSTATED (see C-2).
The model.py:594–595 call is dead-OUTPUT under skip (the output never reaches
the FFN). Under a skip-hardcoded refactor, model.py:593–597 is DELETABLE.
HOWEVER `precompute_head_transports` has TWO additional callers the memo did not
flag: `transformer/baselines/hybrid_gauge_transformer.py:489` and `:567`. So the
METHOD itself cannot simply vanish — it must survive on the class (which the memo
already recommends keeping) or those baselines break. The "new home on the FFN"
is therefore optional, not required, for the in-scope model.

BR-5 (rope_full_gauge_mode drop, blocks.py:595) — CONFIRMED. The assignment
`self.attention.rope_full_gauge_mode = _rope_mode` has no target once the
sublayer is gone; the FFN side (blocks.py:594 `_rope_full_gauge_vfe`) is the path
that matters. Drop is sound; mode 'both' loses the attention-side σ rotation
(documented reduction).

## (4) DS-4 latent bug — CONFIRMED, with one wording correction

Empirically, `non_flat_transport=True` + `skip_attention=True` allocates a
`PerHeadGaugeConnection` (blocks.py:692) but its `forward` is called 0 times over
a full `model(ids)`. The only call in the block forward is blocks.py:815, inside
the bypassed `if not self.skip_attention:` block; `delta_ij` stays None and the
FFN receives `connection_delta=None` (blocks.py:1004). So non_flat_transport is a
silent no-op under skip. CONFIRMED as a real latent bug with no `__post_init__`
warning.

WORDING CORRECTION (C-3): memo 06 calls 815 the "ONLY" call site of
`self.gauge_connection`. Project-wide, `block.gauge_connection(mu, mu)` is ALSO
called at `transformer/analysis/publication_metrics.py:2201` (a holonomy/connection
diagnostic) and the module is read at publication_metrics.py:1444 and
holonomy_metrics.py:32. So 815 is the only call in the TRAINING/INFERENCE forward,
not the only call site of the module. The no-op-under-skip conclusion is
unaffected (the diagnostic path is offline analysis), but the "ONLY" claim is
overstated. Important for the DS-4 fix: routing the connection onto the skip path
must not break the analysis-time direct call.

Note (cross-ref): the BR-3 irrep_dims re-home ALONE does not fix DS-4. After the
re-home, non_flat+skip would construct without AttributeError but still no-op
unless the δ_ij computation is also routed onto the skip path (or a
`__post_init__` warning is added). The DS-4 fix and the BR-3 re-home are
independent.

## (5) Scope-boundary — CONFIRMED

- `scripts/verify_active_inference.py:103` → `skip_attention: False`. CONFIRMED.
- `scripts/run_optuna_hpo.py:103` → `skip_attention: False`. CONFIRMED.
- `scripts/run_ablation_suite.py:101` → `skip_attention: True` (compatible).
  CONFIRMED.

Test categories:
- (a) BREAK on sublayer removal (access `block.attention.*` on a built model):
  `test_cross_head_coupling.py:202,214,376`
  (`model.transformer.blocks[0].attention`),
  `test_equivariant_head_mixer.py:395,397,399`
  (`block.attention._mixer_active`, `.mixer_params`),
  `test_per_head_gauge_connection.py:313,390,398,407,436`
  (`block.attention.register_forward_pre_hook`, `block.attention.rope_full_gauge_mode`).
  CONFIRMED. NOTE (C-4): memo 07 lists `test_block_equivariant_mixer.py:252` as
  category (a); that line uses `getattr(block.attention, '_mixer_active', False)`
  — but the attribute access `block.attention` STILL raises AttributeError if the
  sublayer attribute is removed entirely (the getattr default only guards the
  inner `_mixer_active`, not the outer `.attention`). So it DOES break on full
  removal — memo is right, but for the reason that `block.attention` itself is
  gone, not `_mixer_active`. Net: still category (a).
- (b) SURVIVE iff `IrrepMultiHeadAttention` class is kept (standalone ctor):
  `test_attention.py:289,304`, `test_equivariant_head_mixer.py:54,214,272,309`.
  CONFIRMED.

## Corrections to the memos (none change a deletion/re-home decision)

- C-1 (memo 07 BR-2): attention.py log_kappa alloc is at **1533–1540**, not
  "560–565". Lines 560–568 are the FFN's copy in variational_ffn.py. Conflated
  files; the logic is correct.
- C-2 (memo 07 BR-3/BR-4, HEADLINE): the cache output is consumed at exactly ONE
  terminal site — `attention.py:1925–1927`
  (`head_cached_transport = cached_head_transports[head_idx]`) inside
  `attention.forward`, which is dead under skip. The intermediate plumbing
  confirms this: both stack forwards thread `cached_head_transports` to
  `block(...)` (forward_with_attention model.py:1236; plain forward
  blocks.py:1194 checkpoint path / 1209 non-checkpoint path), but the block
  forward never forwards it to the FFN — the FFN call (blocks.py:994–1007) passes
  only `connection_delta=delta_ij` and `precomputed_block_exp_pairs=_shared_bep`
  (both None under skip), and every executable read of `cached_head_transports`
  in the block forward (813,828,836,845,855,866,876,886,890,898,920) is inside
  the skip guard. So the model.py:571–604 build machinery feeds a value that only
  `attention.forward` can turn into computation. `gauge_param='omega'` still works
  under skip because `omega` threads to the FFN independently via `omega=omega`
  (model.py:1237 → blocks.py:1002 → FFN `_build_block_exp_pairs`; empirically
  `gauge_param='omega'` + skip RUNS).
  Recommendation, scoped to the refactor option chosen (memo 07's two options):
  under option (i) hard-remove the non-skip path (CLAUDE.md "pure path / opt-in
  extreme paths" favors this) — DELETE model.py:573 (omega build) and
  model.py:593–597 (precompute call); they are then unconditionally dead.
  Under option (ii) keep `IrrepMultiHeadAttention` behind a legacy flag — a
  `skip_attention=False` + `evolve_phi=False` config still CONSUMES the cache via
  attention.forward:1927, so the build sites must be GUARDED by the legacy flag,
  not deleted. Either way BR-4's "re-home precompute onto the FFN" is OPTIONAL for
  the in-scope skip model; the `precompute_head_transports` METHOD must merely
  survive on the class for `hybrid_gauge_transformer.py:489,567` (BR-4 missed
  those two callers).
- C-3 (memo 06 DS-4): "ONLY call site 815" is overstated — see (4); add
  publication_metrics.py:2201 as a diagnostic caller.
- C-4 (memo 07 category split): `test_block_equivariant_mixer.py:252` correctly
  classified as (a)-break, but the proximate cause is the missing `.attention`
  attribute, not the getattr-guarded `_mixer_active`.

## Confidence

High on all path:line confirmations (read directly + empirically reproduced).
High on the optimizer routing (independently reproduced the name-router).
High on the C-2 headline correction (the FFN call signature at blocks.py:994–1007
is unambiguous and the omega-independent thread is empirically verified).
