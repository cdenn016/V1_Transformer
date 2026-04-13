# Deep Audit Round 6 — 2026-04-07

Continuation of the deep audit. Rounds 1-3 recorded in
`session_2026_04_07_deep_audit_fixes.md`; round 4 in
`session_2026_04_07_deep_audit_round4.md`; round 5 in
`session_2026_04_07_deep_audit_round5.md`. Round 6 focused on the
remaining unread areas: `block_config.py` (dataclass + from_config
routing), `gauge_utils.py` matrix primitives, and
`attention.IrrepMultiHeadAttention.__init__`.

## Round 6 fixes

### Fix #36 — `fused_block_matrix_exp_pairs` missing dimension assertions (`gauge_utils.py:206`)

Unlike `fused_block_diagonal_kl_diag` / `fused_block_diagonal_kl_full`
(which got dimension assertions in round 2 Fix #15),
`fused_block_matrix_exp_pairs` had none. A caller passing mismatched
`irrep_dims` (e.g. `[1, 2]` with a `generators` tensor of shape
`(3, 4, 4)`) would silently slice `generators[:, 0:1, 0:1]`,
`generators[:, 1:3, 1:3]` and leave index 3 unused, producing nonsense
exp pairs downstream with no error.

**Fix**: added two `ValueError` checks at entry:
1. `sum(irrep_dims) == generators.shape[-1]`
2. `phi.shape[-1] == generators.shape[0]` (phi length matches n_gen)

Verified via runtime smoke test that a mismatched call raises
`ValueError` with a clear message.

### Fix #37 — `sigma_max` dataclass/from_config default mismatch (`block_config.py:131`)

- Dataclass default: `10.0`
- `from_config` default: `5.0`

Direct `BlockConfig(...)` construction produced `sigma_max=10.0`;
`BlockConfig.from_config(config)` with the key omitted produced
`sigma_max=5.0`. Violates the single-source-of-truth principle.
Verified the production `variational_ffn.py` constructor also defaults
to `5.0`.

**Fix**: aligned the dataclass default to `5.0` (matching `from_config`
and the VFE module default). Production training configs explicitly
override this value, so the default is hit only when a caller
constructs a `BlockConfig` with no `sigma_max`.

### Fix #38 — `rope_base` dataclass/from_config default mismatch (`block_config.py:198`)

- Dataclass default: `1000.0`
- `from_config` default: `10000.0`
- Industry standard (GPT-style RoPE): `10000.0`

**Fix**: aligned the dataclass default to `10000.0` (matching
`from_config` and the industry standard). Production configs in
`train_publication.py` use a range of values (50, 5000, 10000), all
explicit overrides.

### Fix #39 — `E_learnable_alpha` boolean reversal (`block_config.py:381`)

- Dataclass default: `True`
- `from_config` default: `config.get('E_learnable_alpha',
  config.get('ffn_learnable_alpha', config.get('learnable_alpha',
  **False**)))`

A **boolean reversal**: direct `BlockConfig(...)` construction enabled
Bayesian precision `α_k = c₀/(b₀ + KL)`, but
`BlockConfig.from_config(config)` with the key omitted silently
disabled it. Production configs set `E_learnable_alpha=True` explicitly
(`train_publication.py:208`), matching the dataclass.

**Fix**: changed the `from_config` fallback chain to end at `True`
instead of `False`, matching the dataclass and production behavior.

### Fix #40 — `evolve_phi_e_step` default mismatch (`block_config.py:358`)

- Dataclass default: `True`
- `from_config` default: `False`

Production configs explicitly set `evolve_phi_e_step=True`
(`train_publication.py:195,413`), so the from_config default of
`False` was silently opposite the documented production intent.

**Fix**: changed the `from_config` default to `True`, matching the
dataclass and production. `__post_init__` at lines 170-172 still
correctly forces both `evolve_phi` and `evolve_phi_e_step` to `False`
when `gauge_mode` is `'trivial'` or `'constant'`.

### Fix #41 — `n_picard_steps > 0` requires `closed_form_e_step=True` (`block_config.py:__post_init__`)

The Picard re-solve loop in `variational_ffn._closed_form_e_step`
(line 2252) is nested inside the `closed_form_e_step` branch. Setting
`n_picard_steps=2` with `closed_form_e_step=False` is a silent no-op:
the Picard branch is never entered, and the setting has no effect on
training. A user who thinks "I'm running Picard corrections on my
iterative E-step" is actually running vanilla iterative E-step.

**Fix**: added a `ValueError` check in `__post_init__` that rejects
the combination with a clear error message pointing the user at the
two valid resolutions (either enable `closed_form_e_step` or set
`n_picard_steps=0`). Verified via runtime smoke test that the
mismatched combination raises; the `n_picard_steps=0` case still
works.

## Also touched in round 6 (round 5 continuation)

### Fix #35 final sweep (`embeddings.py:615,656,744`)

Round 5 fixed three `pad_token_id = -1` defaults in the public and
internal embedding helpers. Verified via regex that no
`pad_token_id = -1` default remains anywhere in the `transformer/`
package (positive results for `-100` only).

## What round 6 verified correct (no new bugs)

### `gauge_utils.stable_matrix_exp_pair` (via subagent + spot check)

- Frobenius norm clamping at `max_norm=20.0` before Padé scaling-
  squaring prevents overflow
- Skew-symmetric shortcut `exp(−M) = exp(M)^T` saves one matrix_exp
  call when `skew_symmetric=True`
- Dtype upcasting to float64 for `d ≥ 20` prevents precision loss in
  fp16/bf16
- Gradient flows through the norm-clamp scaling factor, allowing
  backprop to shrink oversized φ

### `gauge_utils.newton_schulz_orthogonalize` (via subagent)

- Classical `X ← 1.5·X − 0.5·X·X^T·X` iteration
- Pre-scales input by `frob/√K` with threshold `1.2` to bring
  singular values into the `(0, √3)` convergence basin
- Fixed 5-iteration loop with early termination on
  `||X^T X − I||_F < 1e-6`
- Minor concerns (subagent): the rescaling threshold is heuristic and
  could miss matrices with σ_max ∈ [1.2, √3); no explicit NaN guard
  on the input. Not fixing because this path is only used when
  `enforce_orthogonal=True`, and the upstream matrix_exp stability
  layer clamps phi before computing `exp(M)`.

### `IrrepMultiHeadAttention.__init__` (direct read, `attention.py:1234-1577`)

- Irrep block structure correctly constructed from `irrep_spec` for
  SO(3), SO(N), and GL(K) modes
- GL(K) multi-head: `n_heads · d_head == embed_dim` validated at
  line 1371
- Per-head generators correctly sliced: `global_generators[:,
  cum_dim:cum_dim+dim, cum_dim:cum_dim+dim]`, with `cum_dim` advanced
  by the per-head dim
- Scalar (ℓ=0) heads correctly use a zero generator (gauge-invariant)
- SO(3) even-dim rejection: raises `ValueError` for invalid irrep
  dimensions (must be odd)
- `use_output_projection=False` default preserves the "no W_O"
  constraint; when enabled, uses `nn.Linear(K, K, bias=False)`
- `constant_omega` for gauge_mode='constant' initialized to identity
- Skew-symmetry flag cached at construction via `torch.allclose`

**Design concern (not a bug)**: When `learnable_head_kappa=True`,
the attention sublayer and the VFE FFN have *independent*
`log_kappa_per_head` parameters. Both get their own gradient from
their own β computation, so the two kappas can drift apart during
training. This is either intentional (allowing different
temperatures at the two stages) or a missed opportunity to tie the
parameter (single kappa shared between attention and E-step β).
Flagging for user decision — not fixing because either interpretation
is defensible.

### `attention.compute_kl_matrix` (direct read, `attention.py:432-504`)

Thin wrapper around `_dispatch_kl_matrix` — no separate
implementation, delegates to the unified kernel in `kl_computation.py`.
Correct.

## Files touched in round 6

- `transformer/core/gauge_utils.py` — Fix #36
- `transformer/core/block_config.py` — Fix #37, #38, #40, #41
  - Both dataclass defaults (`sigma_max`, `rope_base`) and from_config
    defaults (`evolve_phi_e_step`, `E_learnable_alpha`) aligned
  - `__post_init__` validation for n_picard_steps + closed_form_e_step

All edited files parse cleanly via `ast.parse`. Runtime smoke tests
verify: (a) dataclass defaults produce the expected values, (b)
`from_config` with missing keys produces the same values as the
dataclass, (c) `fused_block_matrix_exp_pairs` raises on mismatched
dimensions, (d) `BlockConfig(n_picard_steps=2, closed_form_e_step=False)`
raises `ValueError`, (e) `n_picard_steps=0` with
`closed_form_e_step=False` still works.

## Cumulative audit stats (rounds 1-6)

**24 distinct actionable fixes applied + 2 false positives
withdrawn.**

### Fix count by round
- Round 1: #1-4 (attention residual, mask_self_attention default,
  closed-form docstring, CLAUDE.md map)
- Round 2: #10, #12-16 (_retract_omega pullback, DEQ divergence
  guards, fused kernel NaN guards)
- Round 3: #20-29 (**attention residual bug live in training path**,
  holonomy penalty no-op, _use_rope_vfe stale comment, etc.)
- Round 4: #30-34 (generate temperature=0, config reference, random
  generator fallback)
- Round 5: #35 (last `pad_token_id=-1` defaults)
- Round 6: #36-41 (fused_block_matrix_exp_pairs assertion,
  BlockConfig default mismatches, n_picard_steps validation)

### Most consequential single fix
Round 3 Fix #20 remains the most important: the attention residual
bug was live in the training forward path `forward_with_attention`,
and the round 1 "fix" to `blocks.py` was dead code because
the training loop uses an independently-written code path.

### Most architecturally important finding
Drift hazard #29 (the two independently-written block-forward
implementations in `blocks.py` and `model.py:forward_with_attention`)
remains unresolved architecturally. Round 3 added matching DRIFT
HAZARD comments; a proper refactor has both paths go through a single
implementation.

### Most structurally consequential fixes of round 6
The BlockConfig default mismatches (#37-40) are the most likely to
have silently affected users who omitted the keys from their config
dicts. In particular, the `E_learnable_alpha` boolean reversal (#39)
meant that anyone relying on the dataclass documentation (which said
`True`) would have silently gotten `False` via `from_config` —
disabling Bayesian precision without any warning. Production configs
override this explicitly so no production run was affected, but any
"minimal config" would have been wrong.

### Core VFE pipeline correctness
Unchanged: the mathematical core (KL attention, Fisher-metric
natural gradient, sandwich-product covariance transport, per-block
fused kernels, active inference EFE/distillation, implicit-EM
gradient scaling, left-invariant Omega retraction, full-covariance
closed-form Cholesky solver, SO(3) log and BCH, RiemannianAdamW)
is correct across all audited paths. The round 6 fixes are
configuration-hygiene issues and a defensive assertion, not
correctness bugs in the math.
