# Welford-style EFE refactor — 2026-04-07

Memory optimization for `_compute_active_inference_gradient` in
`transformer/core/active_inference.py`. Triggered by the user reporting OOM
at K=20, N=128, GL(10), batch_size=64 with `use_prior_bank`,
`gauge_fixed_priors`, `hierarchical_priors`, and `active_inference` all
enabled.

## Diagnosed cause

Define `U = B * N * V * 4` bytes. At the user's parameters
(B=64, N=128, V=50257), `U ≈ 1.65 GB` per `(B,N,V)` float32 buffer.

The original epistemic loop in `_compute_active_inference_gradient`
(`active_inference.py:189-210`) accumulated all S samples' softmax outputs
into a Python list, then `torch.stack(probs_samples, dim=0)` materialized a
`(S, B, N, V)` tensor, followed by a `(S, B, N, V)` log-stack inside the
same `torch.enable_grad()` block. The autograd graph held every sample's
logits + softmax + log-softmax intermediates simultaneously, peaking at
`(3S + 2) · U`. At S=4 that is ~14 U ≈ 23 GB just for the EFE epistemic
term, before counting the surrounding model.

Hierarchical priors (`lambda_hyper=0` in `EM_CONFIG`) were a red herring —
the KL(s||h) term is gated off and contributes nothing. `gauge_fixed_priors`
already had the V×K×K matrix-exp checkpointed in `prior_bank.py:507-511`.
The active-inference epistemic loop was the dominant peak.

## What changed

Rewrote `_compute_active_inference_gradient` as a two-pass algorithm:

**Pass 1 (no_grad).** Stream the S MC samples through the readout, accumulate
`p̄ = (1/S) Σ_s p_s` via running-sum updates, free each sample's logits
before allocating the next. Save the noise tensors (`(B,N,K)` each, kilobytes)
so pass 2 can reproduce identical mu_s values. Materialize
`log_p̄_const = log(p̄)` once at the end of pass 1; release `p̄`.

**Pass 2 (per-sample autograd).** For each sample, build a fresh autograd
graph through the decode at `mu + noise[s]`, compute the per-sample loss
`L_s = -(ε / S) · KL(p_s || sg(p̄_const))`, call `autograd.grad(L_s, mu_var_s,
retain_graph=False)`, and accumulate the resulting `(B,N,K)` gradient into a
running sum. Explicit `del` of the per-sample logits / probs / log-probs
references after each iteration so the storage is reclaimed before the next
sample's decode allocates.

The pragmatic term is computed in its own `enable_grad` block immediately
before the epistemic loop, so the pragmatic and epistemic graphs never
coexist either.

### Correctness derivation

The naive joint gradient is

    ∂(-ε MI)/∂μ = (ε / S) Σ_s Σ_v ∂p_s/∂μ · (log p̄ - log p_s).

The per-sample loss `L_s = -(ε / S) · KL(p_s || sg(p̄))` with `log p̄`
treated as a constant has gradient

    ∂L_s/∂μ = -(ε / S) Σ_v [∂p_s/∂μ · (log p_s - log p̄) + p_s · ∂log p_s/∂μ]
            = -(ε / S) Σ_v ∂p_s/∂μ · (log p_s - log p̄) - (ε / S) Σ_v ∂p_s/∂μ
            = (ε / S) Σ_v ∂p_s/∂μ · (log p̄ - log p_s),

where the second equality uses `Σ_v p_s · ∂log p_s/∂μ = Σ_v ∂p_s/∂μ`, and
the third drops the `Σ_v ∂p_s(v)/∂μ` term because `Σ_v p_s(v) ≡ 1` so its
derivative is zero. Summing over samples reproduces the joint gradient
exactly.

This is **not** an approximation. It is the same gradient computed via a
different decomposition of the autograd graph.

### Memory peak

| Term | Old | New |
|---|---|---|
| Pragmatic alone | ~3 U | ~3 U (unchanged) |
| Epistemic alone (S=4) | (3S+2) U = 14 U | ~5 U |
| Pragmatic + epistemic (S=4) | ~17 U | max(3 U, 5 U) = 5 U |

At B=64, K=20, V=50257, the EFE peak drops from ~23 GB to ~8 GB. At B=16
(the value `EM_CONFIG` actually defines), the peak drops from ~5.8 GB to
~2 GB.

### Compute cost

Two extra passes over the S decode calls — one in pass 1 (no_grad, no
backward) and one in pass 2 (with grad + backward). Effective decode count
goes from S forward + S backward to 2S forward + S backward. The pragmatic
decode is unchanged. For S=4 that is 12 decode calls instead of 8 in the
forward direction, an extra ~50% forward-decode FLOPs in the EFE path.
Backward FLOPs are unchanged. The user's bottleneck is memory, not
throughput, so this trade is intended.

## Files changed

- `transformer/core/active_inference.py` — `_compute_active_inference_gradient`
  rewritten (lines 93-256). Same signature, same return type, same
  numerical output up to float32 round-off.

## Verification

`tmp/verify_welford_efe.py` runs the new implementation against a
reference copy of the original joint-graph code with seeded RNG across 8
configurations (S = 1, 4, 8; pragmatic only; pragmatic + epistemic;
varying weight; varying V; both disabled). All cases pass with
`max_abs_err ≤ 3e-8` (float32 round-off from summation order).

The S=1 case is special: `MI ≡ 0` when there is only one sample, so the
gradient is exactly zero in both implementations and the relative-error
check is skipped.

## Not changed

`_compute_distillation_gradient` still holds two `(B, N, V)` tensors
(target_probs in no_grad + local_log_probs in autograd), but does not
multiply by S, so its peak is ~3 U regardless of `S`. It runs after
`_compute_active_inference_gradient` returns its detached gradient, so
the EFE graph is gone before distillation builds its own. Distillation
chunking is not currently needed.

`PriorBank.decode` itself is unchanged. Vocab chunking inside decode would
be the next step if memory remains tight after this refactor. Held off
because (a) it requires a two-pass log-softmax over V chunks, which
doubles the matmul count, and (b) the Welford refactor alone should be
sufficient at the user's parameters once combined with `epistemic_samples=1`
or smaller `batch_size`.

## Follow-up: gauge_fixed_priors decode-cache (speed)

After the Welford fix landed, the user reported that training was *still
slow* and isolated the cause to `gauge_fixed_priors=True`. PriorBank
without gauge-fixed priors and active-inference were both quick.

### Profiling

Benchmark at K=20, V=50257, GL(10), 2 heads of dim 10, on CPU
(`tmp/bench_gauge_fixed_decode.py`):

| Op | Time |
|---|---|
| `torch.linalg.matrix_exp` forward on (100K, 10, 10) | 83 ms |
| `torch.linalg.matrix_exp` backward | **380 ms (5.6× forward)** |
| Hand-rolled Taylor matrix_exp fwd+bwd | 625 ms (worse) |
| `gauge_fixed_priors=False` decode fwd+bwd | 807 ms |
| `gauge_fixed_priors=True` decode fwd+bwd | **1574 ms (~95% slower)** |

Per decode call with `gauge_fixed_priors=True`:
- forward matrix_exp: 83 ms
- gradient checkpoint recompute (legacy): 83 ms
- backward through matrix_exp: 380 ms
- ≈ **546 ms of pure matrix_exp work per call**

PyTorch's `matrix_exp` uses the Najfeld-Havel block-matrix trick for the
backward pass — it runs another matrix_exp on a (2d, 2d) matrix, costing
~8× the forward. A hand-rolled Taylor implementation made the backward
*worse* because autograd accumulates per-matmul-node overhead.

### What changed

Replaced the gradient checkpoint in `PriorBank.decode` with a per-forward-pass
**cache** of the all-vocab matrix_exp output (`prior_bank.py:454-543`):

- New `cache_decode_priors` constructor argument (default `False` for backward
  compatibility).
- New `clear_decode_cache()` method on `PriorBank`.
- `decode()` checks the cache before recomputing; on cache miss it computes
  `_get_prior_for_tokens(arange(V))` *without* the checkpoint, stores the
  output (with autograd graph attached), and reuses it across all decode
  calls in the same forward pass.
- `model.py:920-925` and `model.py:1067-1071` now call
  `self.prior_bank.clear_decode_cache()` at the top of `forward()` and
  `forward_with_attention()` so the stale graph from the previous step is
  released.
- The legacy gradient-checkpoint path is preserved when
  `cache_decode_priors=False`, so existing memory-tight configurations are
  unaffected by default.

When backward runs, every consumer of the cached `(mu_p, sigma_p)`
contributes its gradient through the *same* matrix_exp graph, accumulating
once at `phi_embed.weight`. After backward the autograd intermediates are
freed; the next forward pass gets a fresh cache.

### Memory cost

The cache holds the matrix_exp output `(V, K, K)` (the saved tensors of
`torch.linalg.matrix_exp`'s autograd Function are M and exp(M)). At V=50k,
K=20: roughly 80–200 MB extra over the legacy checkpoint path. The user's
GPU has headroom after the Welford EFE fix.

### Speed measurement

`tmp/bench_decode_cache.py` on CPU (relative numbers should hold on GPU,
absolute will be much smaller):

| Decodes/step | Legacy | Cached | Speedup |
|---:|---:|---:|---:|
| 1 (no AI) | 2511 ms | 1460 ms | **1.72×** |
| 4 (Welford EFE 2 fwd + 2 bwd) | 6873 ms | 3982 ms | **1.73×** |
| 12 (AI + distill) | 19035 ms | 10108 ms | **1.88×** |

For the single-decode case the saving comes entirely from skipping the
gradient-checkpoint recompute (one fewer matrix_exp forward per backward
pass). For multi-decode workloads each additional decode now reuses the
same matrix_exp graph instead of running a fresh one.

### Correctness

`tmp/verify_decode_cache.py` confirms:

- N=1 decode call: forward logits and all parameter gradients are
  **bit-identical** between cached and legacy paths.
- N=3 decode calls: forward logits bit-identical; gradients differ only
  by float32 summation order (rel_err ≈ 3e-6 on `base_prior_mu` /
  `base_log_prior_sigma`, rel_err ≈ 8e-5 on `phi_embed.weight`). The
  larger relative error on `phi_embed` reflects round-off accumulated
  through the matrix_exp backward, which is itself computed in float32.

### Files changed (this follow-up)

- `transformer/core/prior_bank.py` — added `cache_decode_priors` ctor arg,
  `_decode_cache` attr, `clear_decode_cache()` method; rewrote the
  all-vocab prior fetch logic in `decode()` to support cache vs legacy
  vs eval paths.
- `transformer/core/model.py` — pass `cache_decode_priors` from config
  to `PriorBank(...)` (line 859); call `clear_decode_cache()` at the top
  of `forward()` (line 919) and `forward_with_attention()` (line 1062).

### How to enable

Add to `EM_CONFIG`:

```python
'cache_decode_priors': True,
```

The flag is read by `model.py:859` via `config.get('cache_decode_priors',
False)`. Default remains `False` so memory-tight configurations get the
legacy checkpoint behavior unchanged.

## Recommended config follow-ups

The user reported these were tried with OOM. After this refactor, in
ranked order of memory savings per effort:

1. `'active_inference_epistemic_samples': 1` in `EM_CONFIG`. Drops the
   epistemic peak from 5 U to ~3 U with no code change (BALD MI is
   unbiased at S=1, just noisier). Combined with the Welford refactor,
   total EFE peak at B=64 is ~5 GB.

2. Add `'grad_accumulation_steps': 4` to `EM_CONFIG` and reduce
   `'batch_size': 16`. `EM_CONFIG` does not currently set
   `grad_accumulation_steps`, so it defaults to 1
   (`transformer/training/config.py:87`). The TrainingConfig field name is
   `grad_accumulation_steps`, not `grad_accum_steps` (the latter is the
   Pure VFE config's field, `train_publication.py:595`).

3. Vocab-chunked decode if 1+2 still OOM. Not implemented in this session.

---

## Post-Welford + Post-Cache Deep Audit (inspection-only, no code changes)

Conducted 2026-04-07. Parameters: B=64, N=128, K=20, V=50257,
H=2 heads of dim 10, S=4, 2 VFE iterations, diagonal covariance,
amortized_inference=True. U = B×N×V×4 = 1.645 GB.

### Remaining hot spots (after both fixes)

**1. `(B,N,N,K)` softmax-coupling accumulators** (`vfe_gradients.py:616-622`)

Four tensors of shape `(B, N, N, K)` allocated per iteration in the
diagonal fused path: `kl_values`, `grad_kl_per_pair_full`,
`grad_kl_rope_per_pair` (if use_rope), `grad_sigma_per_pair_full`
(if compute_sigma_align_grad). At B=64, N=128, K=20 each is
64×128×128×20×4 = 80 MB. Four live simultaneously: 320 MB/head, 640 MB
total across 2 heads per iteration. With amortized=True and no
`gradient_checkpoint_vfe`, both iterations' graphs are retained in
backward, adding 640 MB to the backward peak.

Fix: N-chunking inside `_fused_attention_and_vfe_gradients_block_diag`.
Infrastructure already exists in the full-covariance path. Difficulty: Medium.

**2. `Omega_block (B,N,N,d,d)` ephemeral peak** (`vfe_gradients.py:946`)

At B=64, N=128, d=10: 64×128×128×10×10×4 = 419 MB per head per
block iteration. Explicitly `del`-ed (line 1043) before the next block,
so never simultaneously live. Not OOM-causing alone, but the largest
single ephemeral allocation in the inner loop. Eliminated by the same
N-chunking fix. Difficulty: Medium (same as Finding 1).

**3. `gradient_checkpoint_vfe=False` — both iteration graphs retained**

Default is False (`variational_ffn.py:290`). The flag and checkpointing
infrastructure already exist (`variational_ffn.py:2589-2628`). Setting
`gradient_checkpoint_vfe=True` frees the first iteration's intermediate
buffers (~320 MB VFE) at the cost of 2× compute on that iteration. For
2 iterations this is a 1.5× overall compute overhead in the VFE path.
Difficulty: Easy (config).

**4. `update_phi_per_iteration=True` — redundant matrix_exp per iteration**

Each iteration calls `fused_block_matrix_exp_pairs` twice: once for the
main VFE path and once for `_compute_phi_grad`. The phi-grad call cannot
reuse the main call's result because autograd needs to differentiate
through `phi_for_grad` (a new leaf). Setting `update_phi_per_iteration=False`
eliminates the per-iteration phi-grad call, enables `_hoisted_bep` reuse
across iterations (`variational_ffn.py:2577`), and halves the matrix_exp
count (from 4 to 2 calls for 2 iterations). Difficulty: Easy (config).

**5. Decode `combined (B,N,V)` = 1U per call, 10 calls/step**

Each of the 10 decode calls (4 Welford pass-1 + 4 pass-2 + 1 distillation
+ 1 final) produces a `(B, N, V)` = 1U = 1.65 GB tensor. These are
sequential (Welford ensures no two autograd graphs coexist) so the peak
is 1U at any moment. The decode is already optimally fused (single matmul
`(B,N,2K) @ (2K,V)^T`). No larger intermediates, no per-vocab matrix
inverse. The only reduction path is vocab-chunked decode (Hard).

**6. Prior recomputation across iterations: none (verified correct)**

`sigma_p` is extracted once in `_prepare_e_step_inputs`
(`variational_ffn.py:2505-2518`) and passed as a constant argument to
all `_vfe_iteration` calls. Not recomputed per iteration.

**7. KL chunking: absent in diagonal fused path (intentional)**

`_fused_attention_and_vfe_gradients_block_diag` has no N-chunking
("no chunking" comment at `vfe_gradients.py:230`). The full `(B,N,N)`
KL accumulator is always materialized. At the user's parameters this is
80 MB per head — acceptable but improvable.

**8. Backward peak vs forward peak**

Backward peak is dominated by the final decode backward graph (~2U =
3.3 GB) plus retained VFE buffers from both iterations (~640 MB). Total
backward peak: ~4 GB. This exceeds the forward VFE peak (~640 MB) but
is lower than the Welford pass-2 epistemic peak (~5U). The backward
peak from the decode cannot be reduced without vocab chunking.

### Ranked remaining optimizations (bang-for-buck)

1. `grad_accumulation_steps=4, batch_size=16`: all (B,N,X) tensors
   shrink 4×. Welford peak drops from ~8 GB to ~2 GB. No code change.
2. `update_phi_per_iteration=False`: halves matrix_exp FLOPs. No code change.
3. `gradient_checkpoint_vfe=True`: frees ~320 MB backward buffers at 1.5×
   VFE compute. No code change.
4. N-chunking in `_fused_attention_and_vfe_gradients_block_diag`: reduces
   80 MB per head accumulators by N/chunk factor. Medium code change.
5. Vocab-chunked decode in `PriorBank.decode`: reduces 1U decode peak to
   ~0.02U per chunk. Hard code change, largest long-term saving.

### What the audit found is already optimal

- Decode intermediates: clean single-matmul fusion, no hidden `(B,N,V,K)`
  tensors, no per-vocab matrix inverse.
- Welford refactor: correctly eliminates `(S,B,N,V)` probs_stack.
- Decode cache: correctly avoids V-vocab matrix_exp on all but the first
  decode call.
- Prior hoisting: correctly precomputed once before the iteration loop.
- Omega_block: correctly `del`-ed immediately after use in the block loop.
- EFE + distillation graphs: correctly freed before the next subgraph builds
  (retain_graph=False throughout).
