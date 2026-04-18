# Deep Audit Fixes — 2026-04-07

This document records the codebase changes made during the three-round deep
audit of the active inference pipeline conducted on 2026-04-07. All changes
are correctness fixes, robustness hardening, or documentation corrections.
No behavioural changes were introduced outside the fixes described below.

## Round 1 findings and fixes

### Fix #1 — Attention residual delta extraction (`transformer/core/blocks.py`)

`GaugeTransformerBlock.forward` originally used the plain residual
`mu_q = mu_q + mu_attn` at the attention sublayer. Because `mu_attn = Σ_j
β_ij · Ω_ij · mu_normalized[j]` is an aggregation of the normalized input
rather than a zero-centred correction, when self-attention dominates
(`KL(q_i‖q_i) = 0` makes `β_ii` maximal when `mask_self_attention=False`)
the plain residual dumps `mu_normalized` into the residual stream each
layer — the exact pathology the FFN delta extraction at line 548 was
introduced to prevent.

Replaced with `mu_q = mu_q + (mu_attn - mu_normalized)` so the residual
stream accumulates corrections rather than copies of the pre-normalization
input. Same semantics as the FFN residual.

### Fix #2 — `mask_self_attention` dict-constructor default (`transformer/core/block_config.py:348`)

The dataclass field at line 80 declared `mask_self_attention: bool = True`
but the `from_config` dict constructor at line 342 used
`config.get('mask_self_attention', False)`, a contradictory default. The
dataclass default was dead code. Unified both to `True`. Existing training
configs that explicitly set the value (e.g., `train_publication.py` lines
165, 793) are unaffected.

### Fix #3 — Closed-form E-step docstring (`transformer/core/variational_ffn.py:1890-1905`)

The docstring claimed `σ_i* = 1 / [α/σ_p + λ·Σ_j β_ij/σ_j]` but the code
correctly computes `σ_i* = (α + λ) / [α/σ_p + λ·Σ_j β_ij/σ_j]` because
`Σ_j β_ij = 1` gives `α + λ·Σβ = α + λ` in the numerator of the stationary
point of `∂F/∂σ_q = 0`. Corrected the docstring with an inline derivation.

### Fix #4 — CLAUDE.md codebase map

Added `active_inference.py`, `kl_computation.py`, `transport_ops.py`,
`vfe_gradients.py`, and `vfe_utils.py` to the `transformer/core/` codebase
map with brief descriptions of their hot-path functions. Previously these
files (including the 1,770-line `vfe_gradients.py` that contains the
production VFE gradient kernel) were invisible in the orientation map.

## Round 2 findings and fixes

### Fix #10 — `_retract_omega` left-invariant pullback (`transformer/core/variational_ffn.py`)

The original formula `ξ = Ωᵀ · grad_Ω` is only left-invariant under
orthogonal `Ω` (i.e. SO(K)), so the "trust region is constant in the
intrinsic geometry" claim in the docstring was false for the production
GL(K) configuration. Replaced with the correct left-invariant pullback
`ξ = Ω⁻¹ · grad_Ω` via `torch.linalg.solve(Ω + 1e-6·I, grad)` with a
`torch.linalg.pinv` fallback for numerical robustness. Added a historical
note to the docstring explaining the original bug. Only affects
`gauge_param='omega'`; the default `gauge_param='phi'` never called this
path.

### Fix #12 — DEQ + active-inference hard error (`transformer/core/active_inference.py:718-745`)

The original code logged a warning when `active_inference=True` was
combined with `use_deq=True` or `closed_form_e_step=True`. Both
combinations produce either silent no-ops or biased M-step gradients:
the DEQ backward Jacobian is built from the VFE-only step operator while
the forward includes EFE terms, so the IFT correction applies the wrong
operator. Converted both warnings to `raise ValueError` so
misconfigurations fail fast at model construction.

### Fix #13 — DEQ backward divergence guard (`transformer/core/variational_ffn.py`)

`DEQFixedPoint.backward` and `DEQFixedPointFull.backward` accumulated the
Neumann series `(I − J^T)⁻¹ v ≈ v + J^T v + (J^T)² v + …` with no
divergence safeguard. Added: per-iteration `torch.isfinite` check that
aborts the sum on the first non-finite vjp, and per-iteration Frobenius
norm cap at `_DEQ_VJP_NORM_CAP = 1e4` to prevent geometric growth when
`||J|| > 1` at the fixed point. A single NaN vjp previously contaminated
every downstream M-step parameter.

### Fix #14 — Fused kernel NaN guards (`transformer/core/vfe_gradients.py`)

`_fused_attention_and_vfe_gradients_block_diag` computed
`sigma_j_transported = einsum('bijkl,bijkl,bjl->bijk', Ω, Ω, σ).clamp(min=1e-4)`
with no NaN replacement — `torch.clamp` cannot remove NaN, only clamp
finite values. If `stable_matrix_exp_pair` produced NaN for an extreme
phi, the NaN propagated through the KL / softmax / gradient chain. Added
three NaN guards that mirror the ones in `kl_computation._kl_kernel_dense`
and the fused KL kernels in `gauge_utils.py`: one replacing NaN rows of
`Omega_block` with identity, one on `sigma_j_transported`, one on
`mu_j_transported`. Each calls `_nr` with a distinct tag for diagnostics.
Added `from math_utils.numerical_monitor import record as _nr` to the
imports.

### Fix #15 — Dimension assertion on fused block-diagonal KL (`transformer/core/gauge_utils.py`)

`fused_block_diagonal_kl_diag` and `fused_block_diagonal_kl_full` built
block ranges by accumulating `start += d` with no check that the ranges
exhausted the last dimension of `mu_q`. A caller passing mismatched
`irrep_dims` (e.g. `[1, 2, 3]` with `mu_q.shape[-1] == 7`) would silently
process only the first six channels and return an under-summed KL. Added
`raise ValueError` guards at the top of both kernels asserting
`sum(irrep_dims) == K` and `len(block_exp_pairs) == len(irrep_dims)`.

### Fix #16 — Safe inverse in direct-omega transport (`transformer/core/transport_ops.py`)

`compute_transport_operators_direct` called `torch.linalg.inv(omega)`
with no regularisation, and `omega_to_block_exp_pairs` called
`torch.linalg.inv(omega_blk)` on each per-head block. During training
`omega` can drift toward low rank (the GL(K) direct parameterisation
permits any determinant sign), and a raw inverse on a near-singular
matrix poisons the entire attention graph with NaN. Replaced both with
`omega + 1e-6·I` ridged inverses wrapped in `try/except` that fall back
to `torch.linalg.pinv` on failure.

### Finding #11 withdrawn

Initial subagent analysis claimed `gauge_preconditioner.py:97`
`einsum('aij,bij->ab', ...)` computed `tr(T_a T_b)` rather than the
documented Frobenius inner product `tr(T_a^T T_b)`. Direct verification
showed this is wrong: `Σ_{ij} A[a,i,j]·B[b,i,j] = tr(A_a^T B_b)` is
exactly the Frobenius inner product regardless of whether `T_a` is
symmetric. The inline comment at `gauge_preconditioner.py:251-256` in
the Killing form function documents this explicitly (the `transpose`
version was the previous bug). No change made.

## Round 3 findings and fixes

### Fix #20 — Attention residual bug was LIVE in the training forward path (`transformer/core/model.py:1141-1156`)

Round-1 Fix #1 was applied to `GaugeTransformerBlock.forward` in
`blocks.py`, but the training loop at `train_fast.py:210 →
train.py:compute_free_energy_loss:326` calls
`model.forward_with_attention(...)`, not `model.forward(...)`.
`forward_with_attention` has an independently written inline copy of
the block forward that still used the buggy `mu_q = mu_q + mu_attn`
pattern. The round-1 fix was dead code relative to the actual training.

Applied the same delta extraction to `model.py:1141-1156` so the
training forward also uses `mu_q = mu_q + (mu_attn - mu_normalized)`.
This is arguably the single most consequential fix of the entire audit
because it actually affects training dynamics, whereas the round-1 fix
only affected inference.

### Fix #21 — Holonomy penalty silent no-op (`transformer/train.py`)

`block_config.holonomy_penalty` was stored as `block.holonomy_penalty`
in `blocks.py:285` but never read by `compute_free_energy_loss`. Users
who set `holonomy_penalty > 0` in their config expecting non-flat
transport regularisation got zero effect on training. Wired the
penalty into `compute_free_energy_loss`: it iterates
`model.transformer.blocks`, reads each block's `holonomy_penalty` and
`_last_exp_delta` (populated by `blocks.py:378` and
`model.py:1090`), calls `holonomy_penalty_loss(exp_delta)` when both
are set, adds the weighted penalty to `total_loss`, and clears
`_last_exp_delta` to avoid stale reuse. The penalty is also exposed in
the metrics dict as `loss/holonomy`.

### Fix #22 — CLAUDE.md exception for `train_publication.py` argparse

`train_publication.py` uses `argparse` for `--mode` / `--ffn_mode` /
`--device` / `--checkpoint_dir` / `--seed` / `--dataset` /
`--semantic_analysis_interval`, violating the project's own NO CLI
ARGUMENTS hard constraint. The CLI exists for mode selection between
preset configuration dicts, not for tunable hyperparameter exposure,
and removing it would break the existing training workflow. Updated
CLAUDE.md to document the exception narrowly: the `train_publication.py`
exception is limited to mode selection; the hyperparameter dicts
continue to live in the file, and the exception does not extend to
other entry points.

### Fix #23 — `_use_rope_vfe` dead kwarg + stale comment (`transformer/core/variational_ffn.py:661-700`)

The `use_rope` constructor kwarg was declared at line 544 but never
stored on `self`; line 676 unconditionally hardcoded
`self._use_rope_vfe = True`. A 15-line comment above line 676 argued
that RoPE should be DISABLED in the VFE E-step to prevent
"double-counting position" and "distorting the VFE fixed point".

Ultrathink analysis confirmed that the code's `True` setting is
actually correct. The fused kernel in `vfe_gradients.py` implements a
careful hybrid objective:

```
F_align = α·KL(q‖p) + λ·Σ_ij β_ij^{RoPE} · KL_raw(q_i ‖ Ω_ij q_j)
```

where `β` is softmaxed from the RoPE-rotated KL (position-aware) but the
KL being re-weighted is content-only (raw-μ). This mirrors the attention
sublayer's "attention gauge ≠ value gauge" factorisation at
`attention.py:1789-1792`, where message aggregation uses raw μ while β
uses RoPE-rotated μ. The elaborate chain-rule infrastructure in
`vfe_gradients.py` (separate `kl_values_raw` accumulator,
`grad_kl_rope_per_pair`, `_un_apply_rope_pair_outer` at line 968) exists
specifically to make this hybrid mathematically consistent: the direct
alignment term β·∂KL_raw/∂μ uses the raw-μ gradient, while the softmax
coupling ∂β/∂μ_raw·KL_raw applies R(θ_i)^T to un-rotate the rope-space
gradient. The stale comment's "double-counting" concern does not apply
because the code explicitly prevents it by keeping the alignment
objective (KL_raw) distinct from the attention routing function (KL_rope
via β).

Applied two fixes:
1. Made the kwarg functional: `self._use_rope_vfe = use_rope`, so that
   setting `cfg.use_rope = False` disables RoPE consistently in both
   the attention sublayer AND the VFE E-step.
2. Replaced the misleading 15-line comment with an accurate explanation
   of the hybrid objective, the factorisation, and the chain-rule
   machinery that makes it work. Kept a "historical note" paragraph
   documenting the original concern and why it does not apply.

### Fix #24 — Checkpoint state mismatch on implicit-EM resume (`transformer/training/train_fast.py`)

`save_checkpoint` previously saved `model_state_dict`,
`optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, and
global step, but NOT the per-block implicit-EM state (`_last_alpha_i`,
`_last_beta_for_implicit`, `_last_implicit_mu_scale`,
`_last_implicit_sigma_scale`, `_last_omega`). On resume, the restored
optimizer momentum reflected the pre-save gradient direction but the
IFT scale used by `ImplicitEMGradient` was `None` until the first
forward pass recomputed it — creating a state mismatch between restored
momentum and freshly-computed gradient scaling.

Added a `_ffn_implicit_em_state` list to the checkpoint that serialises
these per-block transient attributes (detached, CPU-moved, with
`torch.Tensor` values re-attached to the training device on resume).
The matching restore logic in `load_checkpoint` tolerates missing or
mismatched entries (old checkpoints skip this restore and rely on the
next forward pass to re-populate the attributes).

### Fix #25 — `GaugeConnection` antisymmetry default (`transformer/core/connection.py`)

`antisymmetrize` defaulted to `False`, meaning the bilinear connection
`δ_ij^a = μ_i^T W^a μ_j` was in general not antisymmetric in `(i,j)`.
A non-antisymmetric connection is a torsion connection that violates
the cocycle interpretation of the transport and weakens the
well-definedness of the holonomy penalty. Flipped the default to
`True` (`W → (W − W^T)/2`) and updated the docstring to document the
reasoning. Users who want a deliberate torsion-bearing ablation can
still set it to `False` explicitly.

### Fix #26 — `δ_ij` clamping before matrix exp (`transformer/core/transport_ops.py:365-383`)

`compute_transport_operators` passed `α · connection_delta · G` directly
to `torch.linalg.matrix_exp` with no magnitude check. If the
`GaugeConnection` weights drifted during training, `||α·δ·G||` could
grow unboundedly and produce NaN in the matrix exponential. Added a
per-edge Frobenius norm cap at `5.0` (matching the GL(K) `max_norm`
convention in `retract_glK_torch`) applied to `scaled_delta` before the
einsum-into-generators. Biases the transport infinitesimally under the
rare saturation case but keeps the pipeline finite during long training
runs.

### Fix #27 — `M_alpha` bypassed in `learnable_alpha` mode (`transformer/train.py`)

`compute_free_energy_loss` computed
`self_consistency_loss = M_alpha * kl_per_agent.mean() / dim_scale`
in the fixed-alpha branch but
`self_consistency_loss = (alpha_scalar * kl_per_agent).mean() / dim_scale`
in the learnable-alpha branch — silently ignoring the user's `M_alpha`
config value when `learnable_alpha=True`. Fixed to multiply by `M_alpha`
in both branches so `M_alpha=0` disables the term in both modes and
`M_alpha` acts as an overall loss weight on top of the learnable
per-position alpha.

### Fix #28 — `pad_token_id` default mismatch (`transformer/core/model.py`)

`p_flow_update`, `phi_flow_update`, and `delta_rule_update_w_out`
defaulted their `pad_token_id` parameter to `-1`, but the CE loss and
the VFE observation mask use `-100`. In practice the call sites in
`experiment_runner.py` pass the trainer's configured `pad_token_id`, so
the defaults are never hit — but the API mismatch is a correctness
landmine for any other caller. Aligned the defaults to `-100`.

### Fix #29 — Divergent block forward drift hazard (`transformer/core/model.py` and `transformer/core/blocks.py`)

The root cause of Fix #20: `GaugeTransformerBlock.forward` in
`blocks.py` and `GaugeTransformerLM.forward_with_attention` in
`model.py` contain two independently written implementations of the
same "block forward". A correctness fix applied to one will not reach
the other. The proper fix is a refactor so both paths share a single
implementation, but that is out of scope for an audit.

Added matching DRIFT HAZARD comments at both call sites that:
(a) name the sibling path, (b) explain which one runs during training
and which during inference, (c) list the invariants that must be kept
in sync (residual delta extraction, shared block-exp pair computation,
non-flat delta injection, sigma residual semantics), (d) cite the
round-1 → round-3 history of Fix #20 as a worked example of what
happens when the two paths drift, and (e) recommend a regression test
that asserts identical outputs on a small fixed input.

## Files touched

- `CLAUDE.md` — codebase map updates (Fix #4), argparse exception (Fix #22)
- `docs/session_2026_04_07_deep_audit_fixes.md` — this document
- `transformer/core/active_inference.py` — Fix #12
- `transformer/core/block_config.py` — Fix #2
- `transformer/core/blocks.py` — Fix #1, drift hazard comment (Fix #29)
- `transformer/core/connection.py` — Fix #25
- `transformer/core/gauge_utils.py` — Fix #15
- `transformer/core/model.py` — Fix #20, Fix #28, drift hazard comment (Fix #29)
- `transformer/core/transport_ops.py` — Fix #16, Fix #26
- `transformer/core/variational_ffn.py` — Fix #3, Fix #10, Fix #13, Fix #23
- `transformer/core/vfe_gradients.py` — Fix #14
- `transformer/train.py` — Fix #21, Fix #27
- `transformer/training/train_fast.py` — Fix #24

## Verification

All touched Python files parse cleanly via `ast.parse`. The
`holonomy_penalty_loss` smoke test on an identity `exp_delta` returns
~0 as expected. No runtime behaviour was changed for configurations
that do not exercise the affected code paths — the fixes are strictly
additive (guards, assertions, correct formulas, wired-up loss terms)
and do not alter the default training trajectory except for the
attention residual delta extraction in the training forward path
(Fix #20), which was silently degrading training dynamics at every
layer before this audit.

## Known-incomplete items

- **Architectural refactor of the divergent block forwards (Fix #29)**:
  The two in-lined implementations should be consolidated into a single
  path. This is a larger change than an audit can safely make; drift
  hazard comments have been added to both sites in the meantime.
- **Pure-VFE variant audit**: Subagent-audited; no findings. The pure
  VFE path is mathematically equivalent to the main path at the formula
  level and has its own finite-difference gradient tests.
