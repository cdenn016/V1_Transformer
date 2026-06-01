# Verifier C — Performance / Numerics — Adversarial Confirmation

Date: 2026-06-01
Role: Independent adversarial verifier (perf + numerics) for the live
`skip_attention=True` production path (`train_publication.py::EM_CONFIG`).
Environment: torch CPU (cuda unavailable), tiny live-patterned model
(embed_dim=20, GL(10) 2 heads d=10, n_layers=1, ffn_n_iterations=1, seq=16, batch=2,
em_mode='ift_phi', diagonal_covariance=True, alpha_divergence=0.3, sigma_max=12.0).
Probe: `docs/audit_workspace/verify/_perf_numerics_probe.py` (reproducible).

All claims were checked against the ACTUAL code paths (not docstrings) and an
instrumented forward. Verdicts: CONFIRMED / REFUTED / CORRECTED.

---

## PERF-1 (HIGH) — second matrix_exp via the φ-grad autograd recompute — **CONFIRMED (2×)**

Instrumented count of `fused_block_matrix_exp_pairs` in one training forward
(`m.train()`, grad enabled), monkeypatched counter:

| config | matexp calls | path |
|--------|-------------|------|
| LIVE `evolve_phi=T, evolve_phi_e_step=T` | **2** | per-iteration φ block |
| `evolve_phi=T, evolve_phi_e_step=F` | 3 | post-loop φ block |
| `evolve_phi=F` (`update_phi=False`) | 1 | analytic pass only |

The two live calls are:
1. Analytic β/μ/σ pass — `_build_block_exp_pairs` → `fused_block_matrix_exp_pairs`
   at `variational_ffn.py:1467` (called from `_compute_multihead_vfe_gradients`,
   `variational_ffn.py:1885`).
2. φ-grad autograd recompute — `fused_block_matrix_exp_pairs` at
   `variational_ffn.py:1218`, inside `_compute_phi_grad`, on a fresh leaf
   `phi_for_grad = phi_current.clone().requires_grad_(True)` (line 1211). It then
   loops per head calling `compute_attention_weights(..., return_kl=True)`
   (`variational_ffn.py:1244`) to rebuild β + KL inside an autograd graph, then
   `torch.autograd.grad(alignment_loss, phi_for_grad)` (line 1343).

So the live training forward computes β twice and runs the block matrix-exp twice:
once analytically (for grad_μ, grad_σ) and once through autograd (for grad_φ).
The memo's "2× per forward" and "the φ-grad path is the only quantity still computed
by autograd" are both confirmed.

**Magnitude — count is environment-independent; the % is inherited.** The robust,
reproducible fact is the call-count (2→1). The memo's "~55% of forward / 1.3–1.8×
end-to-end" is its own seq=128 CPU measurement; the tiny seq=16 CPU model here cannot
reproduce that meaningfully and CUDA differs again. I did NOT independently re-measure
the 55%; I confirm it is *consistent* (the 2nd matexp + a full 2nd β/KL build + an
autograd backward is unambiguously a large constant-factor fraction of the forward,
and the per-iteration φ block is the only added work that distinguishes the 2-call
case from the 1-call case). Treat 55%/1.3–1.8× as plausible-but-CPU-seq128-specific,
not independently confirmed on the live (CUDA, seq128, batch32) shape.

**Toggle correction (memo over-implied):** the memo said "`update_phi=False` removes
the second one." That maps to `evolve_phi=False`, which disables M-step φ learning
entirely — not a free perf knob. Setting only `evolve_phi_e_step=False` does NOT
remove the autograd recompute; it MOVES it to the post-loop block
(`variational_ffn.py:2722-2754`), which calls `fused_block_matrix_exp_pairs` a THIRD
time (`variational_ffn.py:2743`) before `_compute_phi_grad` adds its own. So the
"remove the 2nd matexp" lever is the analytic-φ rewrite below, not a config flag.

### Analytic ∂F/∂φ feasibility — **FEASIBLE, with a hedged magnitude**

The envelope identity that the μ/σ gradients already use applies verbatim to φ. The
manuscript F at the softmax stationary point β = softmax(−KL(φ)/τ) gives
dF/dφ = Σ_j β_ij ∂KL_ij/∂φ; the softmax-coupling term Σ_j KL ∂β/∂φ cancels exactly
against the entropy-term gradient τ Σ_j log(β) ∂β/∂φ. This is the cancellation the
code comment at `variational_ffn.py:1971-1979` states for μ/σ, and the live μ/σ path
already exploits it (`_lambda_softmax_eff = 0.0` when `include_attention_entropy=True`,
line 1979). The φ gradient is structurally identical — the only new piece is the inner
factor ∂KL_ij/∂φ = (∂KL/∂μ_j^transported, ∂KL/∂σ_j^transported) · ∂(Ω_ij μ_j, Ω_ij Σ_ij Ωᵀ)/∂φ,
i.e. the Fréchet derivative (dexp) of the block matrix-exponential Ω_ij = exp(φ_i G)·exp(−φ_j G).

Closed-form sketch (diagonal σ, per head h, block d_h):
- Ω_ij = exp(φ_i·G) exp(−φ_j·G); ∂Ω_ij/∂φ_i^a = dexp_{φ_i}(G_a) exp(−φ_j·G),
  ∂Ω_ij/∂φ_j^a = −exp(φ_i·G) dexp_{−φ_j}(G_a), where dexp_X(V) is the Fréchet
  derivative of matrix-exp (left/right-trivialized: dexp_X(V) = exp(X)·φ_dexp(ad_X)[V]).
- The fused kernel already materializes the per-pair belief gradients
  ∂KL/∂(transported μ, σ) (the `grad_kl_per_pair` / `grad_kl_block` tensors,
  `vfe_gradients.py:828, 843, 861`). Contract those with ∂(transport)/∂φ and weight
  by β_ij (already available). No new β build, no second matrix_exp — reuse the
  Ω/exp(φ) pairs already cached as `_mh_cached_bep` (this is exactly the hook the dead
  `cached_block_exp_pairs` parameter was meant to be; see PERF-1b — the autograd path
  can't use it because it needs a fresh leaf, but an analytic path CAN).

**Honest magnitude of the win.** The analytic rewrite removes (a) the 2nd
`fused_block_matrix_exp_pairs`, (b) the per-head 2nd `compute_attention_weights`
β/KL rebuild, and (c) the autograd graph + backward bookkeeping. It does NOT remove
the Fréchet (dexp) work itself — that cost is roughly what autograd's backward already
pays. So the realistic saving is "second forward matrix_exp + second β/KL build +
autograd overhead", not the full φ-grad cost. I would hedge BELOW the memo's "large
fraction of 55%": plausibly a meaningful but sub-55% reduction. No off-the-shelf dexp
helper exists in the tree — `_retract_phi` (`vfe_utils.py:765`) works in Lie-algebra
coordinates and consumes a precomputed `delta_phi`; it does not produce ∂Ω/∂φ. The
Fréchet derivative would have to be implemented (the BCH/`dexp` series machinery is
referenced in comments but not exposed as a reusable ∂F/∂φ producer).

**Required verification before adoption (CLAUDE.md mandate).** Strongest check is
analytic-vs-autograd agreement to ~1e-5 on the tiny model (autograd is ground truth
for this exact quantity). A finite-difference cross-check on total F is secondary and
also validates the envelope claim: perturb φ, recompute β = softmax(−KL(φ)/τ) at each
perturbed φ, and confirm dF/dφ ≈ Σ_j β ∂KL/∂φ to first order (central differences,
step ~1e-4, float64). Both must pass before trusting the analytic path.

## PERF-1b — dead parameter `cached_block_exp_pairs` — **CONFIRMED**

`_compute_phi_grad` (`variational_ffn.py:1183-1351`) declares
`cached_block_exp_pairs: Optional[list] = None` (line 1191). AST analysis of the
function body finds **zero rvalue reads** of that parameter. The only other textual
occurrence (line 1253, `cached_block_exp_pairs=_phi_head_bep`) is the keyword-argument
NAME of a downstream `compute_attention_weights` call whose value is `_phi_head_bep`,
locally derived from `_phi_bep` built at line 1218 — not the parameter. The caller
passes `cached_block_exp_pairs=_mh_cached_bep` (`variational_ffn.py:2390`); the value is
silently discarded because the autograd path needs a BEP tied to the fresh
`phi_for_grad` leaf, not the value-only cached one. Dead parameter, confirmed.

## PERF-2 — `track_iteration_diagnostics` host syncs — **REFUTED on the live path (config flag never reaches the FFN)**

The memo claims `track_iteration_diagnostics=True` forces ~4 host syncs
(`.cpu()/.item()`) every forward via the gated block at
`variational_ffn.py:2289-2314`. The block itself is real and IS gated only by the
static `self.track_iteration_diagnostics` flag (plus `_is_final_iter`, always True at
ffn_n_iterations=1). BUT the flag never reaches the FFN on the live path:

- `BlockConfig` (`block_config.py`) has **no `track_iteration_diagnostics` field**;
  `BlockConfig.from_config` (`block_config.py:555`) is an explicit `cls(...)` with no
  such kwarg.
- `blocks.py:586` sets the FFN flag via
  `track_iteration_diagnostics=getattr(cfg, 'track_iteration_diagnostics', False)`
  where `cfg` is a `BlockConfig` — so the `getattr` ALWAYS falls through to `False`.
- The EM_CONFIG key (`train_publication.py:309`) lands on `TrainingConfig`
  (`training/config.py:257`) and drives the trainer-side `IterationDiagnosticsTracker`
  (`experiment_runner.py:337-340`), NOT the FFN module.

**Only-writer argument (closes the "maybe experiment_runner sets it differently"
objection).** A grep across the whole `transformer/` tree shows `blocks.py:586` is the
SOLE site that writes `ffn.track_iteration_diagnostics`. `variational_ffn.py:371` is
just the constructor assignment from the (defaulted-False) kwarg. No runtime path —
trainer, model, or stack — mutates the FFN flag after construction. So regardless of
how the live model is built, nothing can enable it via `EM_CONFIG`.

Empirical confirmation (probe), three-way:
- cfg `track_iteration_diagnostics=True`  → **FFN.track_iteration_diagnostics = False**.
- cfg flag True forward syncs:  `cpu=0, item=1, tolist=0`.
- cfg flag False forward syncs: `cpu=0, item=1, tolist=0` (delta = 0 — the block never runs).
- **FFN flag FORCED True on the module** (`ffn.track_iteration_diagnostics=True` by hand)
  → `cpu=1, item=4, tolist=1`, i.e. +1 device transfer and +3 `.item()` over baseline.
  This is exactly the "~4 host syncs" the block issues at lines 2303 (`.cpu().tolist()`)
  and 2310/2313/2314 (three `.item()`).

**Reconciliation with memo 05's "~84 ms (~6.5%)" measurement.** Memo 05's measured
*cost* is real — but it must have been produced with the FFN flag set directly (e.g.
constructing `VariationalFFNDynamic(..., track_iteration_diagnostics=True)` or setting
the module attribute), which bypasses the `BlockConfig.from_config` plumbing that the
full-model `EM_CONFIG` path goes through. The forced-flag probe above reproduces the
4-sync block, confirming its cost *when enabled*. What memo 05 got WRONG is the premise
that `EM_CONFIG['track_iteration_diagnostics']=True` enables it on the live model — it
does not. (The single `.item()` in the baseline is the `_is_final_iter`-gated
`grad_phi.norm().item()` at line 2396, which fires regardless of the diagnostic flag
and on CPU is not a device sync anyway.)

**Verdict: the PERF-2 per-forward sync cost is ZERO on the live config — the memo's
premise (that EM_CONFIG activates the block) is refuted; its cost-when-enabled is
confirmed.** There is a genuine but DIFFERENT finding here: a config-plumbing gap.
`EM_CONFIG['track_iteration_diagnostics']=True` silently does NOT enable the in-FFN
per-iteration grad-norm collection (`_e_step_grad_norms` for the E-STEP console line);
those rows degrade to the `.get(key, 0.0)` defaults. If the user wants real E-step
grad norms, the flag must be added to `BlockConfig` + `from_config` and threaded to the
FFN. As a perf item, PERF-2 should be dropped; as a correctness/wiring item it stands.

## PERF-4 — `IrrepMultiHeadAttention` dead buffers under skip — **CONFIRMED (exact)**

`blocks.py:483` always constructs `self.attention` (no skip guard). Under the live
config the attention module holds **0 trainable params, 0 total params, 40000 buffer
elements**: `attention.head_generators.0.gen` and `.1.gen`, each shape (200, 10, 10)
= 20000 elems → 40000 total. These duplicate the structure in `ffn.generators` shape
(200, 20, 20) (the per-head 10×10 blocks are the diagonal sub-blocks of the 20×20
generators). Numbers match the memo exactly (~40k elems, ~160 KB float32). Low
magnitude: a VRAM-footprint / `.to()` / state-dict cleanliness item, not a hot-loop
win. The block reads a few attributes off `self.attention` in the non-skip path, so a
fix must re-home those reads (or free the `head_generators` buffers post-init under
skip), not delete the construction wholesale.

## Numerics (memo 04) — `alpha_divergence=0.3` NaN concern — **CONFIRMED DEAD**

Live α-divergence dispatch is the inline blend in
`_fused_attention_and_vfe_gradients_block_diag` (`vfe_gradients.py:837-859`). It uses
ONLY `log`, division, and `**2` (integer square) — **no fractional powers of negative
or zero bases**. `sigma_blend_align = (1−α)σ_i + α σ_j` is `.clamp(min=eps)` (line 842);
its inputs `sigma_i_block` and `sigma_j_transported` are floored and clamped inside the
logs (lines 854-856). For α=0.3<1 the blend is a positive convex combination of positive
variances and cannot go non-positive; the `(α−1)=−0.7` divisor (line 857) is nonzero.
The fractional-power NaN risk does not exist on the live path. (NUM-3's `sig_i` clamp
gap at `gauge_utils.py:537` is in the NON-live fallback KL kernel — confirmed not on the
`_use_fused_mh=True` live dispatch — agreed, no live impact.)

NaN probe (forward + CE backward, 0 nonfinite grads):
- nominal: loss=4.84, logits finite, **0 nonfinite-grad params**.
- φ×30 stress (all φ params ×30, drives the matrix_exp Frobenius clamp):
  loss=4.85, logits finite, **0 nonfinite-grad params**.

The live forward+backward is finite nominal and under φ stress. Caveat (same as memo):
this shows OUTPUTS stayed finite; it does not separately confirm WHICH guard fired.

---

## Summary table

| Claim | Verdict | Evidence (path:line) |
|-------|---------|----------------------|
| PERF-1 2× matrix_exp/forward (live) | CONFIRMED (2; →1 only if evolve_phi=False) | `variational_ffn.py:1218` + `1467`/`1885` |
| PERF-1 55% / 1.3–1.8× magnitude | NOT independently re-measured; consistent | (CPU seq128 memo number) |
| PERF-1 analytic ∂F/∂φ feasible | FEASIBLE, win hedged below 55%; needs dexp impl + 1e-5 autograd check | `variational_ffn.py:1971-1979`; `vfe_gradients.py:828,843,861` |
| PERF-1 toggle "update_phi=False removes 2nd" | CORRECTED — that = evolve_phi=False (disables M-step φ); evolve_phi_e_step=False gives 3 not 1 | `blocks.py:543-544`; probe |
| PERF-1b dead `cached_block_exp_pairs` | CONFIRMED (0 rvalue reads) | `variational_ffn.py:1191` |
| PERF-2 ~4 host syncs/forward (live) | REFUTED — flag never reaches FFN; 0 live syncs | `block_config.py:555`; `blocks.py:586`; probe |
| PERF-2 block costs 4 syncs WHEN enabled | CONFIRMED (forced-flag probe: cpu 0→1, item 1→4, tolist 0→1) | `variational_ffn.py:2303,2310,2313,2314`; probe |
| PERF-2 (recast) config-plumbing gap | CONFIRMED as a wiring bug, not a perf win | `blocks.py:586` getattr fallthrough; sole writer |
| PERF-4 ~40k dead attn buffers | CONFIRMED exactly (40000, 0 params) | `blocks.py:483`; probe |
| NUM α=0.3 fractional-power NaN | CONFIRMED DEAD (log+div+square only) | `vfe_gradients.py:837-859` |
| NUM live fwd+bwd finite (nominal + φ stress) | CONFIRMED (0 nonfinite grads) | probe |

## Net

Highest-value confirmed lever: PERF-1 (the 2× block matrix-exp + 2nd β build via
autograd in the φ-grad path) is real and is the largest single perf target in the live
forward; an analytic envelope-identity ∂F/∂φ is feasible and would eliminate the second
matrix_exp and β rebuild, but the dexp/Fréchet term must be implemented (no helper
exists) and the win should be hedged below the memo's 55% headline. PERF-2 is the one
material refutation: its claimed per-forward sync cost does not exist on the live path
because the config flag is dropped before reaching the FFN; it survives only as a
config-wiring correctness note. PERF-1b and PERF-4 confirmed exactly. Numerics for the
live α=0.3 path are clean.
