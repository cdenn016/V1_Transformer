# Deep Audit Round 7 — 2026-04-07

Continuation of the deep audit. Rounds 1-3 recorded in
`session_2026_04_07_deep_audit_fixes.md`; round 4 in
`session_2026_04_07_deep_audit_round4.md`; round 5 in
`session_2026_04_07_deep_audit_round5.md`; round 6 in
`session_2026_04_07_deep_audit_round6.md`. Round 7 focused on the
joint (μ, Σ, φ) DEQ step, `GaugePositionalEncoding.compose`, the
attention shape-manipulation helpers, the pure VFE variant (direct
re-audit), and the model's gauge-resolution methods.

## Round 7 fixes

### Fix #42 — `_make_deq_step_fn_with_phi` silently ignores `lambda_softmax` (`variational_ffn.py:1684, 1703`)

The DEQ closure's phi-update branch computed the alignment loss as
`lambda_belief * (beta * kl).sum()` in both the multi-head and
single-beta branches. Autograd's chain rule automatically produces
`d/dφ[β·KL] = β·dKL/dφ + KL·dβ/dφ`, so the combined gradient ends up
weighted uniformly by `lambda_belief` — silently ignoring the
`lambda_softmax` split that the forward-path `_compute_phi_grad`
(lines 1171-1174) applies to the same two terms.

When `lambda_belief == lambda_softmax`, the two implementations are
equivalent. When they differ, the DEQ closure's phi Jacobian no
longer matches the forward E-step's Jacobian, producing an incorrect
IFT correction to the M-step phi gradient. This is the same class of
bug as round 4 Fix #32 (`_compute_omega_grad_direct`) and the
pre-existing fix in `_compute_phi_grad`.

**Fix**: rewrote the DEQ closure's phi alignment loss with the
double stop-gradient pattern:
```python
alignment_loss = (
    self.lambda_belief * (beta_phi.detach() * kl_matrix).sum()   # direct
    + self.lambda_softmax * (beta_phi * kl_matrix.detach()).sum()  # softmax coupling
)
```
Applied to both the multi-head (line 1684) and single-beta (line
1703) branches. When `lambda_belief == lambda_softmax`, the result
is identical to the original; when they differ, the DEQ closure now
correctly mirrors the forward E-step.

### Fix #43 — `GaugePositionalEncoding.compose` lacks `phi_dim` validation

`compose(phi, num_agents, device)` assumed that the caller's token
phi had the same `phi_dim` as the positional encoder's internal
`self.phi_dim`. A mismatch (e.g., token embedding with `phi_dim=3`
but encoder constructed with `phi_dim=9` for SO(4)) would crash
deep inside the BCH composition or silently broadcast wrong shapes,
depending on how far off the sizes were.

**Fix**: added a `ValueError` guard at the top of `compose` that
raises a clear diagnostic when `phi.shape[-1] != self.phi_dim`.
Verified via runtime smoke test that a mismatched phi raises with
a clear message, and that a matching phi still passes through
normally.

## What round 7 verified correct (no new bugs)

### Attention shape-manipulation helpers (via subagent)

- `_split_irreps(mu)`: `(B, N, K) → list of (B, N, d_h)`. Uses
  `.contiguous()` to avoid gradient issues. Roundtrip verified by
  the forward pass (`torch.cat(head_outputs_mu)` at line 1811).
- `_split_irreps_sigma(sigma)`: handles both diagonal `(B, N, K)`
  and full `(B, N, K, K)` modes with automatic format conversion on
  shape mismatch.
- `_block_diag_sigma(sigma_blocks)`: correctly reassembles
  block-diagonal covariance with zero off-diagonals (via direct
  index assignment, not left-over zeros).
- `precompute_head_transports`: per-head slicing of phi and
  generators is correct, returns list of dicts with
  `(exp_phi, exp_neg_phi, Omega)` keys.
- `_dispatch_kl_matrix`: routes DENSE / DIAGONAL / BLOCK_DIAGONAL
  correctly based on `irrep_dims` and sigma shape.
- `create_attention_mask`: standard `torch.tril((N, N))` causal
  mask, size-correct.

No shape or indexing bugs found.

### Pure VFE variant (direct re-audit)

- Self-coupling μ gradient: **matches main path** (identical formula,
  product-rule correction for learnable alpha)
- Alignment μ gradient including softmax coupling: **matches main
  path** (`w_ij = β_ij·[1 + (E_β[KL] - KL_ij)/τ]` with clamping
  equivalent to `∂β/∂μ · KL` plus `β · ∂KL/∂μ`)
- Sandwich product transport: strict `Ω·Σ·Ω^T` everywhere
- Fisher metric natural gradient: identical to main path
  (`-2·Σ·sym(∇)·Σ` for full covariance)
- Whitened trust region on μ: matches main path
- M-step analytic formulas for `μ_p`, `σ_p`: matches main path
- Observation gradient: `softmax(logits) - one_hot` matches main path

**Caveats (not bugs, documented):**
- Pure VFE uses a *single autograd pass* for Omega M-step gradients
  (`learning.py:51-112`), contradicting the "pure, no autograd"
  framing. Documented but the naming is misleading.
- RoPE asymmetry (rope KL for β, raw KL for μ gradient) matches the
  main path — both are consistent with the factorisation rule.
- `decode_tau` can differ silently if the pure VFE config and the
  main path config are not kept in sync.
- Omega inversions in `gauge.py` use `safe_inverse()` but late
  clamping on the output can miss cascading near-singularities.

### `GaugePositionalEncoding` (beyond compose)

- `_make_sinusoidal` at `embeddings.py:1030-1064` uses non-standard
  per-coordinate unique frequencies (not alternating sin/cos pairs
  as in standard transformers). Documented as intentional for Lie
  algebra coordinates. Mathematically reasonable for SO(3)
  (phi_dim=3) but semantically murky for SO(N) and GL(K) where
  `phi_dim` is large. The sinusoidal mode for large phi_dim is
  a known limitation, not a bug.
- `forward(num_agents, device)` correctly raises `ValueError` when
  `num_agents > max_seq_len`.
- `compose` in `exact` mode uses true SO(3) exponentiation and
  `so3_log_torch` for the BCH composition. `bch1`/`bch2` modes
  use `so3_compose_bch` for SO(3), `soN_compose_bch_torch` for
  SO(N), and `lie_compose_bch_general_torch` for GL(K). All three
  backend dispatch branches present and correct.

### `_make_deq_step_fn_with_phi` (beyond Fix #42)

The joint (μ, Σ, φ) DEQ closure correctly:
- Computes μ/Σ gradients via `compute_vfe_gradients_gpu` with
  both `lambda_belief` and `lambda_softmax` passed through (so the
  split is already correct in the μ/Σ branch)
- Clamps raw gradient to ±1e3 and natural gradient norms to 500
- Applies whitened trust region on μ
- Uses SPD retraction for σ with the calibrated trust region
- Uses autograd through the alignment loss for the phi gradient
  with `create_graph=True` (required for DEQ backward VJP)
- Clamps phi gradient norm to 10.0 in a differentiable way
- Does a Euclidean Euler step `φ' = φ - η_φ · grad_phi_align`
  rather than Lie group retraction, justified by the docstring
  comment that "at the fixed point ∂F_align/∂φ ≈ 0, so the
  Euclidean step and Lie group retraction agree to first order"

### `model._resolve_gauge_mode` and `_compute_phi_dim`

- `_resolve_gauge_mode`: validates `gauge_mode in ('learned',
  'trivial', 'constant')`; correctly forces `evolve_phi = False` and
  `evolve_phi_e_step = False` for constant/trivial modes; rejects
  `learnable_reflection=True + use_prior_bank=True` with a clear
  error message.
- `_compute_phi_dim`: correct for SO(3) (`3`), SO(N)
  (`N(N-1)/2`), GL(K) single-head (`K²`), and GL(K) multi-head with
  cross-couplings (`n_heads·d² + len(cross_couplings)·d²`).

### `PriorBank.__init__`

- Parameter registration is correct: `base_prior_mu` as
  `nn.Parameter`, `base_log_prior_sigma` as Parameter (when
  `learnable_sigma`) or buffer (otherwise), `phi_embed` as
  `nn.Embedding` with random init scaled by `phi_scale / √phi_dim`.
- `sigma_target` buffer registered in the non-`gauge_fixed_priors`
  branch (lines 202-205). **Subtle concern**: the
  `gauge_fixed_priors=True` branch does NOT register `sigma_target`,
  so if both `gauge_fixed_priors=True` and `use_prior_bank=True`,
  `_get_sigma_target` in `train.py` falls through to its
  "broadened centroid" fallback (documented as the stale branch).
  Low priority — the gauge-fixed path has its own base prior
  structure, so the hyper-prior anchor matters less.

## Files touched in round 7

- `transformer/core/variational_ffn.py` — Fix #42
- `transformer/core/embeddings.py` — Fix #43

All files parse cleanly. Runtime smoke tests verify:
1. `compose` with matching phi_dim works
2. `compose` with mismatched phi_dim raises `ValueError`

## Cumulative audit stats (rounds 1-7)

**26 distinct actionable fixes applied + 2 false positives
withdrawn across seven rounds.**

### Fix count by round
- Round 1: #1-4 (attention residual, mask_self_attention default,
  closed-form docstring, CLAUDE.md map)
- Round 2: #10, #12-16 (omega pullback, DEQ guards, NaN guards)
- Round 3: #20-29 (**training-path attention residual**, holonomy
  no-op, use_rope_vfe comment, etc.)
- Round 4: #30-34 (generate temp=0, config reference, random
  fallback)
- Round 5: #35 (last `pad_token_id=-1` defaults)
- Round 6: #36-41 (fused_block_matrix_exp_pairs assertion,
  BlockConfig default mismatches, n_picard_steps validation)
- Round 7: #42-43 (DEQ phi lambda_softmax split, positional
  encoding phi_dim validation)

### Most consequential findings
- Round 3 Fix #20 (attention residual in the training forward path)
  remains the single most important discovery.
- Round 6 #37-40 (BlockConfig default contradictions, especially the
  `E_learnable_alpha` boolean reversal) is the most consequential
  class of config-hygiene fixes.

### Core VFE pipeline correctness
Unchanged across all seven rounds: the core math is correct (KL
attention, Fisher-metric natural gradient, sandwich-product
covariance transport, per-block fused kernels, active inference
EFE/distillation, implicit-EM gradient scaling, left-invariant
Omega retraction, full-covariance closed-form Cholesky solver, SO(3)
log and BCH, RiemannianAdamW, joint (μ,Σ,φ) DEQ step). The round 7
fixes are a missing weight split (#42, mirroring round 4 #32) and
a defensive validation (#43).

### Three unresolved architectural concerns
1. **Drift hazard #29**: two independently-written block-forward
   implementations in `blocks.py` and `model.forward_with_attention`.
2. **Test coverage gaps** (from round 5): no regression tests for
   implicit-EM IFT scale, DEQ Neumann backward, closed-form vs
   iterative agreement, residual delta extraction, or
   sandwich-under-Ω-perturbation.
3. **Learnable head kappa duplication** (from round 6): attention
   sublayer and VFE FFN have independent `log_kappa_per_head`
   parameters; they can drift during training. Either intentional
   or a missed parameter-sharing opportunity.

---

## OOM Memory Analysis — PriorBank + Active Inference at B=64, N=128, V=50257, K=20, S=4

**Context**: User runs with `use_prior_bank=True`, `batch_size=64`,
`active_inference_epistemic_samples=4`, `active_inference_distill_weight=0.5`,
and `gauge_fixed_priors=True`. The in-repo `EM_CONFIG` shows
`use_prior_bank=False` and `batch_size=16`; the user is overriding these.
This section records the full quantitative analysis (no code was changed).

### Important config note

The repo `EM_CONFIG` at `train_publication.py:163` has `use_prior_bank=False`
and `batch_size=16`. The user is running with both overridden to `True` / `64`.
At `batch_size=16` the (B,N,V) tensor is 0.41 GB and the problem is much more
manageable. At `batch_size=64` all figures below apply.

### Fundamental unit

The base allocation is one (B,N,V) float32 tensor:

    U = B * N * V * 4 bytes  =  64 * 128 * 50257 * 4  ≈  1.645 GB

All estimates below are expressed as multiples of U.

### Quantitative breakdown

**1. PriorBank.decode at model output** (`prior_bank.py:543`, `model.py:491`)

Called once in the final decode path. The forward tensor is `(B,N,V)` = 1 U.
The autograd graph retains: `combined` (B,N,V) via the matmul (1 U),
`lhs` (B,N,2K) ≈ negligible at K=20, `rhs` (V,2K) ≈ negligible,
`prior_bias` (V,) negligible, and the `logits` output (1 U consumed by CE
but gradient buffer retained). Gradient checkpointing via
`torch.utils.checkpoint.checkpoint` at `prior_bank.py:507-511` recomputes
the V×K×K gauge-frame matrix-exps during backward, so those intermediates
are NOT held; only the (B,N,2K) @ (2K,V) matmul activation is retained.

Peak forward+backward from final decode: **~2 U ≈ 3.3 GB**.
Formula: `2 * B * N * V * 4`.

**2. EFE epistemic loop** (`active_inference.py:190-195`, S=4 samples)

Inside `torch.enable_grad()`, the loop runs S=4 calls to `_readout(mu_s, sigma_arg)`.
Each call is a full `prior_bank.decode`, producing `(B,N,V)` logits (1 U)
and then `F.softmax` probs (another 1 U, retained for gradient). All S=4
`probs_s` are appended to `probs_samples` and then `torch.stack`-ed into
`probs_stack` of shape `(S,B,N,V)` = S * U = 4 U. Both `probs_avg` (1 U)
and `log_probs_stack` (S * U = 4 U) must coexist during the entropy
computation at lines 199-204. The autograd graph for `total_efe` must retain
all S logit tensors until `torch.autograd.grad` is called at line 212.

Peak from epistemic term: **~(3S+2) U = 14 U ≈ 23 GB** at S=4.
Formula: `(3*S + 2) * B * N * V * 4`. At S=1: 5 U ≈ 8.2 GB. At S=2: 8 U ≈ 13 GB.

The pragmatic weight is 0 in the user's config, so that branch is skipped
(confirmed by the guard at `active_inference.py:156`).

**3. Distillation term** (`active_inference.py:344-352`)

`_compute_distillation_gradient` makes two decode calls:
- `target_logits = _decode(mu_tilde)` inside `torch.no_grad()` (line 345):
  1 U, but immediately consumed and detached at line 348; does NOT enter
  the autograd graph. `target_probs` (1 U) is retained as the CE target.
- `local_logits = _decode(mu_var)` with grad tracking (line 351): 1 U in
  the autograd graph. `local_log_probs` (1 U) also retained for backward.

Peak from distillation: **~3 U ≈ 4.9 GB**.
Formula: `3 * B * N * V * 4`.

**4. Hierarchical-prior KL(s||h)** (`train.py:498-527`)

At `lambda_hyper=0.0` (which is what `EM_CONFIG` sets), this term is
entirely skipped by the `if lambda_hyper > 0.0:` guard at `train.py:498`.
The user's config also has `lambda_hyper=0.0` at line 246. So this term
contributes **zero** peak memory in the user's scenario, regardless of
`hierarchical_priors=True`. The `hierarchical_priors` flag only propagates
`mu_prior = mu_q.detach()` between layers (`model.py:1124`), which is
O(B*N*K) = negligible.

**5. Per-token gauge-frame matrix-exp (V,K,K) for gauge_fixed_priors=True**

The block-diagonal path (`irrep_spec = [('fund', 8, 10)]`, so 8 heads of
dimension 10) avoids the full V×K×K matrix. Each `decode` call runs
`fused_block_matrix_exp_pairs` on `(1, V, phi_dim)` phi, producing per-head
exp matrices of shape `(V, d_h, d_h)` = `(50257, 10, 10)` per head,
8 heads total: `8 * 50257 * 10 * 10 * 4 bytes ≈ 160 MB`. The gradient
checkpoint at `prior_bank.py:507-511` discards these during forward and
recomputes during backward, so at any given moment at most one copy is live.

With `only_forward=True` (decode path, `prior_bank.py:502`), the inverse
`exp(-phi)` is not computed, halving this to ≈80 MB. This is the
**already-mitigated** path as noted in the existing comments.

Peak from gauge-frame computation: **~80-160 MB** (checkpointed,
not accumulating across decode calls).
Formula: `n_heads * V * d_h^2 * 4` (single copy, recomputed during backward).

### Simultaneous peak

The EFE epistemic loop and the distillation term are called sequentially
from `compute_ai_gradients` (`active_inference.py:431-474`). The EFE call
uses `retain_graph=False` at line 212 and returns a detached gradient, so
its autograd graph is freed before distillation starts. However, all S=4
logit/prob tensors within the epistemic loop coexist simultaneously during
the `probs_stack` construction.

Peak simultaneous memory from active inference alone: EFE epistemic =
**14 U ≈ 23 GB** (dominant term). The final model-output decode and
backward graph adds another ~2 U. Combined peak ≈ **16 U ≈ 26 GB**.

At B=64, N=128, V=50257, K=20, S=4, this overwhelms any consumer GPU
and many datacenter GPUs. The root cause is the `probs_stack` accumulation
pattern at `active_inference.py:195`.

### Concrete optimizations, ranked

**Rank 1: Reduce epistemic_samples from 4 to 1.**
This is a configuration change requiring no code. Saves (3S+2 - 5) U = 9 U
≈ 14.8 GB. At S=1 the BALD MI estimate is high-variance but unbiased in
expectation. The trade-off is that the epistemic gradient will be noisier,
but the E-step applies a whitened trust region that already clips extreme
updates. Expected saving: **9 U** (exact, no approximation, full
correctness preserved).

**Rank 2: Rewrite the epistemic loop to avoid probs_stack accumulation.**
Instead of appending S (B,N,V) prob tensors and stacking, compute H_avg
and avg_H incrementally using Welford's online algorithm. Each sample's
logits and probs can be freed immediately after contributing to the running
mean. This eliminates the S * U `probs_stack` and the S * U
`log_probs_stack` tensors. The autograd graph still needs to retain the
per-sample logit tensors through to `torch.autograd.grad`, but only one at
a time is needed if the gradient is accumulated incrementally via
`create_graph=False` + manual sum. Net saving vs. current code: **(2S) U**
= 8 U at S=4. This is an exact computation, fully preserving VFE gradients
and gauge equivariance. Development effort: medium (requires restructuring
the loop and verifying the incremental entropy formulas).

**Rank 3: Vocab-chunked decode inside the epistemic and distillation loops.**
Split V into shards of size C (e.g., C=5000) and process them sequentially,
freeing each shard's (B,N,C) intermediates before the next. For the
epistemic entropy only the (B,N) running sums are needed, not the full
(B,N,V) probs. For the distillation CE, `F.cross_entropy` reduction can
be replaced by a manual chunked log-sum. The autograd graph for the
gradient w.r.t. mu_var then involves only (B,N,C) activations per shard.
Net saving from the epistemic loop: **~(3S+1) U** (retains only 1 U of
running statistics plus one chunk). Saving from distillation: **~2 U**.
This is an exact computation. The existing `torch.amp.autocast(enabled=False)`
guard at `prior_bank.py:517` must be preserved inside each chunk; chunking
does not affect the float32 requirement. Development effort: high (requires
refactoring `decode` to accept a vocab slice or exporting the (B,N,2K)
fused-matmul lhs and accumulating partial logits outside).

**Rank 4: Gradient checkpointing of the entire EFE/distillation subroutine.**
Wrap the `_compute_active_inference_gradient` call in a
`torch.utils.checkpoint.checkpoint`. The local autograd graphs for the
EFE term are already closed (detached return at line 216), so the
checkpoint would recompute the S decode calls during the outer backward.
However, because the function already calls `torch.autograd.grad` and
returns a detached tensor, there is no outer autograd graph to replay
through — the checkpoint would recompute the whole function, recovering the
forward activations but adding a second set of S decode calls at backward
time. Net saving: eliminates S * U from forward memory at the cost of
doubling the compute. This is exact and preserves all correctness properties.
Development effort: low, but the benefit depends on whether forward or
peak-across-time memory is the bottleneck.

**Rank 5: Share the decode call between distillation target and any
concurrent pragmatic decode.**
With `pragmatic_weight=0` (user's config), the pragmatic decode is already
skipped. The distillation `target_logits` at `active_inference.py:345` is
fully detached and run inside `torch.no_grad()`. No sharing opportunity
exists in the current user configuration because the pragmatic term is off.
This optimization applies only when `pragmatic_weight > 0`.

**Rank 6: bf16/fp16 for the (B,N,V) decode buffer.**
Casting `combined` to bf16 immediately after the matmul at
`prior_bank.py:543` halves the buffer from 1 U to 0.5 U. However, the
existing `torch.amp.autocast('cuda', enabled=False)` guard forces the entire
decode KL to float32 because `log_softmax` over V=50257 in bf16 is
numerically unsafe: the log-sum-exp accumulates 50k terms with only ~3
significant decimal digits, producing errors in the low-probability tail
that corrupt the entropy and MI estimates. If applied, it would need an
explicit upcast to float32 only for the `F.log_softmax` call. This is an
approximation (the gradient w.r.t. mu via the bf16 matmul has reduced
precision); the impact on learning is unknown. Development effort: low,
but correctness risk is moderate and should be validated empirically.

**Rank 7: Reduce batch_size from 64 to 16.**
The repo's own `EM_CONFIG` uses `batch_size=16`. At B=16 the base unit U
drops to 0.41 GB and the peak epistemic memory at S=4 is 5.7 GB, which
is feasible on a 16 GB GPU. If B=64 is required for training dynamics,
gradient accumulation over 4 micro-batches of B=16 achieves the same
effective batch while keeping peak memory at the B=16 level. This is
exact, requires no code changes to the model, and is the single highest
leverage knob available without any algorithm modification.

### Correctness classification

The following optimizations are **exact** (preserve VFE gradients and
gauge equivariance perfectly): reducing S, vocab-chunked decode (if
the chunked sums are computed in float32), full-subroutine gradient
checkpointing, batch size reduction, and the incremental Welford entropy
accumulation.

The following are **approximations**: sub-vocab sampling for the MI
estimate (biased; variance scales as 1/C_sample), bf16 decode buffer
(precision loss in low-probability logits).

The `torch.amp.autocast(enabled=False)` guard at `prior_bank.py:517`
must be respected in all chunked or bf16 variants. The sandwich product
`Sigma_transported = Omega @ Sigma @ Omega.T` is not involved in the
decode path and is unaffected by any of these optimizations.

### Single recommended action sequence

1. Set `batch_size=16` with gradient accumulation 4 in the config
   (immediate, no code change, saves 4× across all tensors).
2. Set `active_inference_epistemic_samples=1` (saves 9 U at current B=64,
   2.25 U at B=16; single config line change).
3. If S=1 MI variance is too high, implement the incremental Welford
   entropy accumulation in `active_inference.py:189-210` to allow S=2 or
   S=4 without the probs_stack accumulation (medium effort, exact).
4. If further headroom is needed, implement vocab-chunked decode as a
   `decode_chunked` method on `PriorBank` and use it inside the epistemic
   and distillation loops (high effort, exact, largest savings).

---

## Python Idioms Audit — amortized_inference=True path

Scope: `variational_ffn.py`, `active_inference.py`, `prior_bank.py`,
`model.py`, `vfe_gradients.py`. Focus: typing, tensor lifecycle, torch
idioms, API surface. Math correctness, detach discipline, and memory
peaks are covered by other audits and are excluded here.

### (a) TYPING AND SIGNATURES

**A1 — `PriorBank.__init__` mutable-default `None` for non-Optional float
(`prior_bank.py:71,78`)**

`init_std: float = None` and `phi_dim: int = None` are annotated as
`float` and `int` but their actual type is `Optional[float]` and
`Optional[int]`. This is a latent mypy error: downstream code assigns
`if init_std is None: init_std = ...` (line 157), confirming they are
truly Optional. Fix: `init_std: Optional[float] = None`,
`phi_dim: Optional[int] = None`.

**A2 — `alpha: 'float | torch.Tensor'` string annotation
(`vfe_gradients.py:64,490,789,1169,1711`)**

The string form is a workaround for Python 3.9 `|` union syntax, but
`torch>=2.1` targets 3.8+ and the project is on 3.11+. The correct
modern form is `Union[float, torch.Tensor]` (already imported) or, on
3.10+, `float | torch.Tensor` without quotes. The string keeps mypy
silent but prevents IDE type-narrowing from working correctly, and it
is inconsistent with the rest of the file which uses proper
`Optional[X]` forms. Fix: replace with `Union[float, torch.Tensor]`
throughout `vfe_gradients.py`.

**A3 — `prior_bank` arguments untyped in `active_inference.py`
(`active_inference.py:96,307`)**

`_compute_active_inference_gradient(prior_bank, ...)` and
`_compute_distillation_gradient(prior_bank, ...)` have `prior_bank`
typed as bare (no annotation). The correct annotation is
`Optional[nn.Module]` or, more precisely, `Optional[PriorBank]`. The
duck-typing rationale is valid (the function just calls `.decode`), but
a `Protocol` with a single `decode` method would make the contract
explicit without introducing a circular import. At minimum, add
`Optional[Any]` so mypy does not infer `object`. Fix: annotate as
`Optional[Any]` or introduce a `Decodable` Protocol.

**A4 — `compute_ai_gradients` arguments untyped (`active_inference.py:439-441`)**

`beta_heads`, `cached_block_exp_pairs`, and `irrep_dims` all lack type
annotations. Given they are called in a tight hot-path loop these are
`Optional[List[torch.Tensor]]`, `Optional[List[Tuple[torch.Tensor,
torch.Tensor]]]`, and `Optional[List[int]]` respectively. Fix: add
annotations; the current signatures break the CLAUDE.md "type hints on
all function signatures" contract.

**A5 — `_vfe_iteration` alpha arguments untyped
(`variational_ffn.py:1472-1473`)**

`alpha_effective` and `_alpha_c0` have no type annotations in the
method signature. They are `Union[float, torch.Tensor]` and
`Optional[torch.Tensor]`. The leading underscore on `_alpha_c0` signals
internal use but the convention is not applied consistently (no
underscore on `alpha_effective` even though it is also an internal
derived quantity). Fix: annotate both; harmonize underscore convention.

**A6 — `_prepare_e_step_inputs` returns `dict` instead of `TypedDict`
(`variational_ffn.py:1289`)**

The return type is annotated as `dict` in the docstring but no
`TypedDict` exists. The returned dict has 9 keys with heterogeneous
value types (`torch.Tensor`, `bool`, `float`, `Optional[...]`). Since
this dict is immediately unpacked at every call site by key string
lookup, a `TypedDict` would catch stale key names at static-analysis
time. The current bare `dict` annotation defeats the purpose of typing.
Fix: define a `_EStepInputs` TypedDict or return a dataclass; at
minimum annotate as `Dict[str, Any]` to signal the intention.

### (b) TENSOR LIFECYCLE

**B1 — Redundant `.detach()` inside `torch.no_grad()` block
(`active_inference.py:198-203`)**

In Pass 1 of the Welford two-pass, `_contrib = F.softmax(logits_s,
dim=-1) / epistemic_samples` is computed inside `with torch.no_grad():`
(line 232). The `del` statements at line 242 correctly free Python refs.
However, `probs_avg` accumulated at line 241 is not explicitly deleted
after `log_probs_avg_const = probs_avg.clamp(...)` is assigned (line
244). The `del probs_avg` at line 245 handles this correctly. No issue.

What IS redundant: `grad_accum = grad_prag.detach()` at line 197
following `torch.autograd.grad(..., create_graph=False)` — the result of
`autograd.grad` with `create_graph=False` is already detached; the
`.detach()` is a no-op costing an object allocation. Same pattern
repeats at line 266: `grad_s.detach()` after `autograd.grad(...,
create_graph=False, retain_graph=False)`. Fix: drop the redundant
`.detach()` calls; document that `autograd.grad` with
`create_graph=False` returns detached tensors by construction.

**B2 — `.clone().requires_grad_(True)` pattern used where a
`torch.empty_like` leaf would suffice (`active_inference.py:187,253`)**

`mu_var = mu_f32.clone().requires_grad_(True)` creates a full copy of
the (B,N,K) tensor just to serve as an autograd leaf. Since the forward
pass replaces its value immediately (via `_readout(mu_var, ...)`), the
`.clone()` data is read by `_readout` rather than discarded. This
pattern is actually correct — the clone IS the leaf; you need its
values passed through `_readout`. No bug, but the comment at line 174
("Detached float32 copies shared across pragmatic + per-sample
epistemic") partially justifies why `sigma_f32` uses `.detach()` while
`mu_var` needs a fresh leaf. The asymmetry is subtle and worth a
one-line inline comment at each call site: `# clone() is the leaf;
mu_f32 stays fixed as the expansion point`.

**B3 — In-place slice assignment on a tensor with live graph
(`vfe_gradients.py:336,348,391`)**

`grad_kl_per_pair_full[:, i_start:i_end, :, block_start:block_end] = grad_kl_block`
is an in-place operation on a freshly-allocated zero tensor. The
accumulator `grad_kl_per_pair_full` was created with `torch.zeros(...)`,
has no grad_fn, and the slice assignment targets non-overlapping regions
per loop iteration. This is safe. However, if future refactoring adds a
`.requires_grad_(True)` to the accumulator for any reason, the in-place
assignment would break autograd silently. A minor robustness note rather
than a live bug.

**B4 — `sigma_i_block = sigma_i_block_slice[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()`
unnecessary `.clone()` after `.contiguous()` exists one line above
(`vfe_gradients.py:354-355`)**

`sigma_i_block_slice` is already `.contiguous()` at line 354. The
`.expand(...).clone()` at line 355 is needed to make the expanded
tensor writable for subsequent einsum indexing, so the `.clone()` is
not redundant here — `.expand` returns a view, and einsum requires
contiguous input. This is correct. No issue.

**B5 — `torch.norm` deprecated API (`variational_ffn.py:776`,
`vfe_utils.py:460,464,500,503`)**

`torch.norm(grad_phi, dim=-1, keepdim=True)` and similar calls use the
deprecated `torch.norm` API. The PyTorch deprecation note (since 1.7)
recommends `torch.linalg.norm` or `torch.linalg.vector_norm` for
explicit norm type. As of torch>=2.1 `torch.norm` still works but emits
a deprecation warning in some contexts. Fix: replace all five instances
with `torch.linalg.vector_norm(..., dim=-1, keepdim=True)` (p=2 Frobenius
for matrix cases, or `ord=2` for vector norm). This is the version that
matches the existing `torch.linalg.norm` calls already used in
`active_inference.py:625,663`.

### (c) TORCH IDIOMS

**C1 — `torch.enable_grad()` context manager may be unnecessary in
pragmatic branch when outer context is already grad-enabled
(`active_inference.py:186`)**

The docstring correctly notes that `torch.inference_mode()` is guarded
against at line 156. However, inside normal training the outer context
is `torch.enable_grad()` by default. The `with torch.enable_grad():`
block is only necessary when the EFE helper is called from inside a
`torch.no_grad()` context (e.g., evaluation). During forward training
there is no outer `no_grad`, so the context manager is a no-op but
harmless. The epistemic Pass 2 loop has the same pattern. This is not
a bug but worth a comment explaining when the `enable_grad()` is
actually doing work versus being defensive boilerplate.

**C2 — Import inside hot path (`active_inference.py:502,537`)**

`import transformer.core.vfe_utils as _vfe_utils_mod` appears inside
the body of `compute_ai_gradients` twice (lines 502 and 537), once in
each try/except block that writes to `_VFE_GRAD_DEBUG`. Python caches
module imports after the first load (O(1) dict lookup), so this is not a
performance catastrophe. However, the module is already imported at the
top of `variational_ffn.py` which is the caller of this function, and it
is accessible via the module's own import graph. The cleaner approach:
import once at the top of `active_inference.py` and reference the module
directly, or pass the debug dict as an argument. The current try/except
swallows any `ImportError` silently, which could hide a broken import
path in a test environment. Fix: hoist the import to module level.

**C3 — `warnings.warn` called inside a tight loop via a local
`import warnings` (`variational_ffn.py:425`)**

`import warnings` appears inside `VariationalFFNDynamic.__init__` (not
a hot path — constructor runs once) to issue a `UserWarning` about
`evolve_phi_e_step=True + gauge_mode='constant'`. This is fine for a
constructor. No action needed.

**C4 — `torch.utils.checkpoint.checkpoint` with `use_reentrant=False`
is correct but not documented (`prior_bank.py:561-565`)**

The `use_reentrant=False` flag is the recommended modern form (removes
the hooks-based re-entrance issue). This is correct. The lambda wrapper
`lambda ids: self._get_prior_for_tokens(ids, only_forward=True)` is
also correct. The comment at line 559 explains why: "Lambda wrapper
avoids passing `only_forward` through checkpoint's **kwargs". This is
accurate — `torch.utils.checkpoint.checkpoint` passes positional args
to the function but does not support kwargs. No issue.

**C5 — `torch.linalg.inv` with fallback to `torch.linalg.pinv` pattern
repeated in seven locations across `variational_ffn.py`,
`vfe_gradients.py`, and `active_inference.py`**

The pattern `try: inv = torch.linalg.inv(X); except: inv = torch.linalg.pinv(X)`
is copy-pasted across the codebase with minor variations. This is a
candidate for a single `_safe_inv(X, ridge=1e-6)` helper, which already
exists as `_safe_spd_inv` in `vfe_utils.py` for SPD matrices. The
non-SPD cases (omega blocks) are not covered by `_safe_spd_inv`. A
`_safe_general_inv(X, ridge=1e-6)` helper in `vfe_utils.py` would
eliminate the repetition. Low priority but contributes to maintenance
surface.

### (d) API SURFACE

**D1 — `_compute_active_inference_gradient` is module-private (`_`
prefix) but is only called from `compute_ai_gradients` in the same
file. Similarly, `_compute_distillation_gradient` is only called from
`compute_ai_gradients`. Both are correctly private. No issue.**

**D2 — `compute_ai_gradients` reads `ffn.__dict__` directly rather than
using `getattr` (`active_inference.py:469`)**

`_ai_bank_raw = ffn.__dict__.get('_prior_bank_ref', None)` bypasses
`nn.Module.__getattr__` intentionally (to avoid registering sub-modules).
This is a documented design choice. However, the pattern mixes two
attribute access strategies: some attributes use `getattr(ffn, '_ai_enabled', False)`
(correct) while the bank/wout references use `ffn.__dict__.get(...)`.
The asymmetry is justified by the list-wrapper convention (to bypass
sub-module registration), but it would be cleaner to add a private
helper like `_resolve_ref(raw)` that handles both the `None` and `list`
unwrapping in one place. Currently the unwrapping logic is duplicated at
lines 470 and 476-477. Fix: extract `_ai_bank_raw[0] if isinstance(..., list) else _ai_bank_raw`
into a module-level `_unwrap_ref` helper.

**D3 — `PriorBank.forward()` dispatches on a `mode: str` argument
(`prior_bank.py:656-678`)**

The `forward(mode='encode'/'decode')` dispatch string is an anti-pattern
for `nn.Module`: it conflates two fundamentally different operations into
one method. Callers already call `prior_bank.encode(...)` or
`prior_bank.decode(...)` directly at all call sites in the codebase (grep
confirms no callers use `prior_bank(token_ids, mode='...')`). The
`forward` dispatch method is dead code in practice. It also contains bare
`assert` statements (lines 674, 677) that are removed by Python's
`-O` flag, which is a correctness concern if the model is ever run
optimized. Fix: either remove `forward()` and only expose `encode`/`decode`,
or replace the `assert` with `if ... raise ValueError(...)`.

**D4 — `wire_readout_references` uses `globals()['logger']` as a
fallback (`active_inference.py:746`)**

`_log = logger if logger is not None else globals()['logger']`
This works because `logger = logging.getLogger(__name__)` is defined at
module scope. However, `globals()` is fragile (fails if the function is
ever moved to a closure or a different module context), and the idiomatic
fallback is simply to define the module logger at the top and use it
directly when the argument is None. Fix:
`_log = logger or logging.getLogger(__name__)`.

**D5 — `_get_prior_for_tokens` is `_`-private but called from
`decode()`, `encode()`, and `_compute_gauge_transform()` within the
same class. No external callers found. Correctly private.**

### (e) OTHER

**E1 — `_VFE_GRAD_DEBUG` is set to `None` or `{}` per-iteration via
a module-level assignment (`variational_ffn.py:1604-1606`)**

Setting a module-level dict reference to `None` as the "disabled" state
works but is not thread-safe. In a DataParallel or multi-process
context, two forward passes running concurrently would race on the
`_vfe_utils_mod._VFE_GRAD_DEBUG` assignment. For single-GPU training
(the primary use case) this is harmless. For documentation: the debug
mechanism is not safe to use with `DistributedDataParallel` unless the
per-process module-level state is acceptable. Worth a one-line comment.

**E2 — `assert sum(irrep_dims) == K` in `_compute_distillation_gradient`
(`active_inference.py:354`)**

This `assert` is a runtime correctness check that is stripped by `-O`.
Since this is a geometry invariant (not a user-input validation), it
should be either left as-is (acceptable given the project never runs with
`-O`) or replaced with `if sum(irrep_dims) != K: raise AssertionError(...)`.
The same pattern exists in `vfe_gradients.py` and other files. Low
priority but consistent with the `-O` concern in D3.

**E3 — `import warnings` inside `VariationalFFNDynamic.__init__`
(`variational_ffn.py:425`)**

Already noted in C3. Not a hot path; constructor-time import. Fine as-is.

**E4 — Missing `__all__` in `active_inference.py`**

The module exports `configure_ffn_active_inference`, `wire_readout_references`,
`compute_ai_gradients`, `apply_ai_mu_updates` as public API (no underscore),
plus two private functions. Without `__all__`, `from active_inference import *`
would expose the private helpers. Low priority but worth defining.

### Top 5 quick wins (ordered by payoff/effort)

1. **A1 — Fix `Optional` annotations on `PriorBank.__init__`** (`prior_bank.py:71,78`):
   two-character change per parameter, prevents mypy false negatives, and
   documents the None-check logic that already exists at line 157.

2. **A2 — Replace string union annotations with `Union[float, torch.Tensor]`**
   (`vfe_gradients.py:64,490,789,1169,1711`): five find-replace
   operations, restores IDE type-narrowing, consistent with the rest of the
   type system.

3. **B5 — Replace deprecated `torch.norm` with `torch.linalg.vector_norm`**
   (`variational_ffn.py:776`, `vfe_utils.py:460,464,500,503`): five
   find-replace operations, eliminates deprecation warnings that will become
   errors in a future torch release.

4. **D3 — Add `raise ValueError` guards in `PriorBank.forward()`** replacing
   the bare `assert` statements at lines 674 and 677: two-line change,
   ensures correctness under `-O`, aligns with D3 concern.

5. **D2 — Extract `_unwrap_ref` helper for the list-wrapped reference
   convention** (`active_inference.py:469-477`): six lines of code,
   eliminates the duplicated unwrapping logic and makes the list-wrapper
   bypass convention explicit and testable.
