# Performance Audit — Live `skip_attention=True` Production Path

Date: 2026-06-01
Lens: Performance / speedups for the live `skip_attention=True` path
Target config: `train_publication.py::EM_CONFIG` (n_layers=1, ffn_n_iterations=1,
diagonal_covariance=True, GL(10) 2 heads d=10, embed_dim=20, batch=32, seq=128,
em_mode='ift_phi', gauge_param='phi', gauge_mode='learned', evolve_phi=True,
evolve_phi_e_step=True, use_rope=True, include_attention_entropy=True,
track_iteration_diagnostics=True, learnable_head_kappa=False).

Empirical measurement environment: torch CPU, B=32, N=128 (matches live shapes).
Absolute ms are CPU; the user runs CUDA RTX5090 so absolute numbers differ, but the
relative split and the host-sync / dead-work findings transfer.

## Steelman of the current code (what is already good)

1. No double matrix-exp between attention and FFN under skip_attention. The block's
   shared-BEP code (`blocks.py` 872-905) lives inside `if not self.skip_attention:`,
   so under skip_attention it never runs; `_shared_bep` stays None and is passed as
   `precomputed_block_exp_pairs=None` to the FFN (`blocks.py:1006`). The FFN computes
   its own transports. No wasted attention-side exp. Confirmed.
2. `cached_head_transports=None` for the live config: `gauge_param='phi'` +
   `evolve_phi=True` routes `model._embed_and_prepare` to the `else` branch
   (`model.py:598-604`), so `precompute_head_transports` is NOT called. Correct.
3. The diagonal fused E-step kernel `_fused_attention_and_vfe_gradients_block_diag`
   (`vfe_gradients.py:968`) does NO full (K,K) covariance lift under
   `diagonal_covariance=True` — self-coupling, KL, natgrad, and retraction all stay in
   (B,N,K) / (B,N,N,d) space. `get_bayesian_alpha` takes the diagonal Rényi branch
   (`variational_ffn.py:894-912`), no inverse, no slogdet, no eye. Good.
4. The NaN-guard identity in the fused kernel uses module-level `_cached_eye`
   (`vfe_gradients.py:1353`, `gauge_utils.py:35`). `_phi_preconditioner` is built once
   at `__init__` (`variational_ffn.py:479`), not per call. `_get_eye` is used in the
   omega / trivial / constant BEP branches. No per-call `torch.eye` on the live diagonal
   learned-phi hot path.
5. Attention module has ZERO trainable params under this config (use_output_projection
   False, learnable_head_kappa False, gauge_mode='learned' → no constant_omega Param,
   mixer force-disabled). Confirmed empirically: `attention total params: 0`. The
   optimizer's `attention_params` list is therefore empty and the `M_attention_lr` group
   is never created (`optimizer.py:826` guard). So the "dead attention params in the
   optimizer" concern does NOT apply here.
6. `_last_beta` is detached (`variational_ffn.py:1504/1506`), so `return_attention=True`
   under skip_attention does not retain the E-step graph. β stacking is one detached
   layer — negligible.
7. Debug paths are correctly gated off: `_VFE_GRAD_DEBUG` is set to None when both debug
   flags are off (`variational_ffn.py:2177`), so the per-component debug writes in the
   fused kernel (1131-1140) are skipped. `print_vfe_grad_debug` early-returns under
   `_debug_vfe_gradients=False`. `record_iteration_diagnostics` is gated by
   `_collect_iteration_diagnostics`, which the trainer only enables on a separate
   diagnostic forward every `diagnostics_interval` steps (`experiment_runner.py:1075`),
   then disables — NOT on the main training forward.

## Findings (ordered by measured magnitude)

### PERF-1 (HIGH) — φ-gradient path is ~55% of the forward; recomputes β a second time via autograd
`variational_ffn.py:2386-2406` (per-iter φ evolve) → `_compute_phi_grad`
(`variational_ffn.py:1183-1351`).

Measured: disabling per-iteration φ evolution (`update_phi=False`) drops the
forward from ~1288 ms to ~577 ms on CPU — the φ-grad path is ~711 ms (~55%).

Why: `_compute_phi_grad` clones φ to a fresh leaf `phi_for_grad` (1211), calls
`fused_block_matrix_exp_pairs` AGAIN (1218), then loops per head calling
`compute_attention_weights(..., return_kl=True)` (1244) to rebuild β and KL inside an
autograd graph, then `torch.autograd.grad` (1343). This is a full second β computation
plus its backward, on top of the analytic β/μ/σ pass in
`_compute_multihead_vfe_gradients` (`variational_ffn.py:1882-2057`).

Confirmed: `fused_block_matrix_exp_pairs` is called exactly 2× per forward (instrumented
count = 2). One for the analytic β/μ/σ grads, one inside `_compute_phi_grad`.

Note the `cached_block_exp_pairs` parameter of `_compute_phi_grad` (1191) is accepted but
NEVER read in the body — the caller passes `_mh_cached_bep` (2390) and it is silently
discarded because the autograd path needs a bep tied to the fresh `phi_for_grad` leaf,
not the value-only cached one. Dead parameter; remove it or document it.

Path to speedup (largest lever, but a real change, not a one-liner):
- The μ and σ gradients already use the envelope identity analytically (no autograd).
  The φ gradient is the only quantity still computed by autograd. An analytic
  ∂F/∂φ = Σ_j β_ij ∂KL_ij/∂φ (envelope identity makes the softmax-coupling term cancel
  against the entropy gradient under `include_attention_entropy=True`, exactly as the
  comment at 1971-1979 states for μ/σ) would remove the second `fused_block_matrix_exp_pairs`,
  the second per-head `compute_attention_weights`, AND the autograd backward. ∂KL/∂φ has a
  closed form through ∂Ω/∂φ (the dexp series already used elsewhere). Estimated upside:
  large fraction of the 55% — realistically a 1.3–1.8× end-to-end forward speedup if the
  analytic φ-grad matches the autograd value. Requires a finite-difference gradient check
  (CLAUDE.md mandate) before trusting it.
- Lower-effort partial win: even keeping autograd, the second `fused_block_matrix_exp_pairs`
  and the per-head β rebuild compute KL on the FULL N×N pair grid. Combine with PERF-3
  (causal lower-triangle) to roughly halve it.

### PERF-2 (MEDIUM) — `track_iteration_diagnostics=True` forces ~4 host syncs every forward for data read only every `log_interval` (200) steps
`variational_ffn.py:2289-2314` and the `grad_phi` writes at 2379 / 2396.

Live config sets `track_iteration_diagnostics: True` (EM_CONFIG line 309). With
ffn_n_iterations=1 every iteration is the final iteration, so the block at 2289 runs on
EVERY forward. It does `torch.stack([_mu_norm,_sig_norm,_mu_cap,_cap_frac]).cpu().tolist()`
(one device→host sync) plus three `.item()` calls (2312-2314) — about 4 forced CUDA
stream syncs per training step. The per-iter φ block adds one more `.item()` for
`grad_phi` (2396).

But `self._e_step_grad_norms` is CONSUMED only on `is_log_step` (`experiment_runner.py:973`
→ `_collect_e_step_grad_norms`, which reads via `.get(key, 0.0)`). So 199 of every 200
forwards do these syncs for a dict nobody reads. On CUDA each sync serializes the stream
and blocks the launch pipeline; measured CPU cost of the diagnostic block alone was ~84 ms
(~6.5%) here, and on CUDA the serialization penalty is typically worse per-sync.

Path to speedup:
- Cheapest: set `track_iteration_diagnostics: False` in EM_CONFIG. `_collect_e_step_grad_norms`
  degrades gracefully to 0.0 for the E-step norm rows; only the per-200-step console
  E-STEP line loses real numbers.
- Better: gate the write block by a dynamic per-step flag (e.g. `ffn._log_this_step`,
  set by the trainer only on log steps) instead of the static constructor flag, so the
  norms are still real on log steps with zero per-step sync otherwise. ~4 syncs/step
  removed.

### PERF-3 (MEDIUM, opt-in already exists) — dense N×N pair grid computed under a causal mask; `causal_lower_triangle=True` halves it but is OFF
`vfe_gradients.py:1337-1409` (dense branch) vs `1186-1335` (packed lower-tri branch).
`EM_CONFIG['causal_lower_triangle'] = False`.

The dense path builds `Omega_block` (B,N,N,d,d), `mu_j_transported` (B,N,N,d),
`sigma_j_transported` (B,N,N,d), and the KL/grad pair tensors over ALL N² pairs, then the
softmax masks out j>i. With a strict causal mask (the live mask), ~half of that pair work
is discarded. The `causal_lower_triangle=True` fast path packs to M=N(N+1)/2 pairs and is
documented bit-identical for β/grad_mu/grad_sigma under a strict lower-triangular mask
(`vfe_gradients.py:1029-1042`).

For N=128 this is a ~2× reduction on the dominant (B,N,N,K) allocations
(`grad_kl_per_pair_full` alone is 32×128×128×20×4 ≈ 42 MB) and the pair einsums. The
opt-in exists; it is simply not enabled. Estimated ~1.3–1.6× on the analytic E-step
kernel cost (it does not touch the φ-grad autograd recompute, which has its own dense
β rebuild — enabling there too compounds with PERF-1).

Caveat: this is a numerical-equivalence fast path. Enable, then verify β/μ/σ outputs are
bit-identical against the dense path on the live config before adopting (the file claims
identity but CLAUDE.md says verify, not trust comments).

### PERF-4 (LOW) — `IrrepMultiHeadAttention` is fully constructed under skip_attention; ~40k dead buffer elems moved to CUDA
`blocks.py:483-516` always instantiates `self.attention`. Under skip_attention its forward
is never called. Confirmed it holds 0 trainable params but 40000 buffer elements
(`attention.head_generators.{0,1}.gen`, each 200×10×10) — duplicates of structure already
present in `ffn.generators` (200×20×20). These buffers are pushed to the GPU by
`model.to('cuda')` (VRAM + a one-time copy) and walked on every `.to()` / state-dict op.

Magnitude is small (~160 KB), so this is a cleanliness/footprint item, not a hot-loop win.
The block still needs a few attributes it currently reads off `self.attention`
(`irrep_dims`, `constant_omega`, `gauge_mode`, `enforce_orthogonal`, `_generators_are_skew`).
Path: under skip_attention, construct a lightweight attribute holder (or compute those few
values directly from cfg) instead of a full `IrrepMultiHeadAttention`, OR free the unused
`head_generators` buffers post-init. Do NOT remove the attention construction wholesale
without re-homing those reads (they are used by the shared-BEP and kappa-sharing code in
the non-skip path).

### PERF-5 (LOW, opt-in) — `torch.compile` of `_vfe_iteration` is wired but off
`variational_ffn.py:669-676`, `compile_vfe=False` default (not set in EM_CONFIG).

The VFE iteration is dominated by many small element-wise kernels (self-coupling, natgrad
clamp/whiten, σ retraction). On CUDA, `torch.compile(mode='default', fullgraph=False)`
would fuse these and cut launch overhead — most useful with n_iterations=1 where Python
overhead is a larger fraction. The `torch.autograd.grad` call inside `_compute_phi_grad`
forces `fullgraph=False` and likely a graph break, so gains are partial. Worth an A/B on
the 5090; estimated 1.05–1.2× if it composes, but unproven and behind first-forward
compile latency. Strictly opt-in, consistent with the "computationally extreme paths are
opt-in" rule.

## Non-findings / things that looked suspicious but are fine

- `_get_kappa_h` with `learnable_head_kappa=False` returns `self.kappa` (a Python scalar) —
  no host sync, no exp/clamp. The kappa-sharing block (`blocks.py:650-666`) is dead
  (gated by `learnable_head_kappa`), harmless.
- The post-loop φ block (`variational_ffn.py:2722-2765`) is dead under live config
  (requires `update_phi_per_iteration=False`; live has it True). No double φ update.
- `constant_omega=None` (gauge_mode='learned'), `gauge_connection=None`
  (non_flat_transport=False), `configure_ffn_active_inference` AI terms all off
  (active_inference=False) — `compute_ai_gradients` returns (None, None). All confirmed
  dead-but-cheap.
- `_hoisted_bep=None` for the live config (update_phi_per_iteration=True) — the hoist at
  2614 is correctly skipped; with 1 iteration there is nothing to hoist anyway.
