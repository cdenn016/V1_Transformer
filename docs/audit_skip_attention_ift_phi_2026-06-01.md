# Audit — `ift_phi` + `skip_attention=True` production path

Date: 2026-06-01
Target: the live production path, `transformer/train_publication.py::EM_CONFIG`.
Method: 7 parallel expert investigators (gauge-equivariance, EM-boundary autograd,
E-step math, numerics, performance, dead-code, refactor blast-radius) followed by
3 adversarial verifiers that re-ran real CPU forward/backward checks. All raw
memos and verifier verdicts live under `docs/audit_workspace/`; the reproducible
equivalence baseline is `docs/audit_workspace/equivalence_harness.py` +
`baseline_skip_attention.json`.

Live config audited (not dataclass defaults): `skip_attention=True`,
`em_mode='ift_phi'`, `gauge_mode='learned'`, `gauge_param='phi'`,
`diagonal_covariance=True`, `evolve_sigma/phi/phi_e_step=True`,
`learnable_head_kappa=False`, `use_residual=False`, `norm_type='layernorm'`,
`use_rope=True`, `rope_base=100`, `rope_full_gauge='off'`,
`non_flat_transport=False`, `closed_form_e_step=False`, `n_layers=1`,
`ffn_n_iterations=1`, `embed_dim=20`, `irrep_spec=[('fund',2,10)]` (GL(10)×2
heads d=10), `alpha_divergence=0.3`, `E_alpha=1`, `E_lambda_belief=10`,
`E_lambda_softmax=0`, `phi_trace_clamp=0.75`, `e_step_sigma_floor=0.01`,
`sigma_max=12.0`.

## Verdict

The `ift_phi` + `skip_attention=True` production path is correct. The central
correctness claim holds and was confirmed by two independent forward+CE-backward
runs (different seeds): all three embedding parameters receive nonzero gradient,
so there is no silent-freeze gap analogous to the documented detaching-mode
(`em_phi_p`/`em_phi_q`) pathology. In `ift_phi` the FFN E-step is itself the
autograd path (`em_phi_mode='amortized'`, no detach at the EM boundary, priors
stay attached).

| param | grad norm (memo 02, seed A) | grad norm (verifier A, seed 0) | alive |
|-------|------------------------------|--------------------------------|-------|
| `token_embed.mu_embed.weight`  | 1.4678 | 0.008331 | yes |
| `token_embed.log_sigma_diag`   | 0.2363 | 0.002282 | yes |
| `token_embed.phi_embed.weight` | 1.3064 | 0.011201 | yes |

(Magnitudes differ by seed and input construction; the qualitative aliveness is
robust. Loss ≈ ln(128) = 4.85 at init confirms a healthy decode graph.)

Gauge transport is correct under the explicitly-allowed diagonal approximation:
the diagonal of the sandwich `Ω diag(σ) Ωᵀ` is used consistently in the KL score,
the β softmax, and both belief gradients; Ω is strictly block-diagonal per head
(GL(10)×2, never full K=20). The fused kernel is exactly gauge-covariant on the
monomial subgroup (signed-permutation × positive-diagonal), which is the
stabilizer of the diagonal-covariance manifold; general GL(10) breaks only by the
permitted diagonal projection. A mathematically-pure equivariant path remains
reachable under toggles (`norm_type='mahalnorm'` + full covariance +
`rope_full_gauge` + `phi_project_slk`), satisfying the CLAUDE.md pure-path rule.

Numerics are clean: the live `alpha_divergence=0.3` path uses only log, division,
and integer squares (no fractional powers of possibly-non-positive bases), the KL
safe-clamp and Killing preconditioner are well-conditioned, and a real
forward+backward produced zero non-finite gradients both nominally and under a
φ×30 stress perturbation.

## Findings

Severity reflects the verifier-adjusted assessment. "Confirmed" marks findings a
verifier independently reproduced.

### High

`PERF-1` — the φ-gradient path recomputes β and the block matrix-exponential a
second time per forward (Confirmed, 2×). The analytic β/μ/σ pass calls
`fused_block_matrix_exp_pairs` once (`variational_ffn.py:1467`, via
`_compute_multihead_vfe_gradients` at `:1885`); `_compute_phi_grad` then clones φ
to a fresh leaf and calls it again (`variational_ffn.py:1218`), rebuilds β+KL per
head via `compute_attention_weights(..., return_kl=True)` (`:1244`), and runs
`torch.autograd.grad` (`:1343`). An instrumented counter showed exactly 2 calls
on the live path (3 if `evolve_phi_e_step=False`, 1 only if `evolve_phi=False`).
The memo's "~55% of forward / 1.3–1.8× end-to-end" is a CPU/seq=128 measurement
and was not independently re-measured; treat the call-count (the robust fact) as
confirmed and the percentage as plausible-but-shape-specific. Fix: an analytic
∂F/∂φ via the same envelope identity the μ/σ path already uses
(`∂F/∂φ = Σ_j β_ij ∂KL_ij/∂φ`, the softmax-coupling term cancelling the entropy
gradient). Verifier C judged this feasible but hedged the win below 55% because
the Fréchet/dexp term still has to be computed and no `dexp` helper exists in the
tree; it requires implementation plus an analytic-vs-autograd agreement check to
~1e-5 (CLAUDE.md mandate) before adoption.

`E-2` — the manuscript-promised pure path `use_autograd_mu_sigma` is not
implemented (from investigator memo 03; not independently re-verified). The live
μ/σ kernel computes only the query-side filtering gradient
`α∇D_self + Σ_j β_ij ∂E_ij/∂μ_i` and omits the key-side smoothing column term
`Σ_{m>i} β_mi ∂E_mi/∂μ_i`, which finite differences show is of comparable
magnitude (query 1.09 vs key-side 1.13). The filtering default is correct and
explicitly sanctioned by the manuscript (`Manuscripts-Theory/GL(K)_attention.tex`
§filtering_free_energy, lines 883–897) as the standard mean-field coordinate-ascent
update for the autoregressive task, so this is not a math error. The gap is that
the manuscript names a reachable pure alternative (`use_autograd_mu_sigma`, which
descends the full ∇F_red) and `grep -rn use_autograd_mu_sigma transformer/`
returns zero hits. CLAUDE.md requires a reachable pure path; here it is doc-only.
Fix: implement the toggle (autograd through the full coupling sum without
detaching the transported key μ_j) or amend the manuscript claim.

### Medium

`F01-D` — the live `norm_type='layernorm'` avoids the documented RoPE ×
MahalanobisNorm gap but introduces a larger LayerNorm break (Confirmed,
documentation finding). Under `skip_attention=True` + `layernorm`, the block uses
`self.norm1 = nn.LayerNorm(K)` and never instantiates `MahalanobisNorm`, so the
CLAUDE.md "rotated μ divided by un-rotated σ" gap is structurally absent. But
`nn.LayerNorm` operates against the global K=20 mean/variance and is itself not
gauge-covariant: it commutes with an unsigned coordinate permutation (err 3.6e-7)
but breaks under diagonal scaling (0.39) and sign flips (0.86), reducing the
block's exact symmetry from the kernel's monomial subgroup to unsigned
permutations; live RoPE on top reduces it to the trivial frame. This is a
deliberate, allowed speed/accuracy trade (the pure path uses `mahalnorm`). Fix
(doc only): extend the CLAUDE.md KNOWN GAP note to state that
`norm_type='layernorm'` is the non-equivariant fast norm and `mahalnorm` is the
equivariant analog.

`PERF-2` (recast) — `track_iteration_diagnostics` is a config-plumbing gap, not a
per-forward cost (memo 05's perf premise Refuted; the wiring bug stands). The
memo claimed `track_iteration_diagnostics=True` forces ~4 host syncs every
forward via `variational_ffn.py:2289-2314`. Verifier C refuted this on the live
path: `BlockConfig` has no `track_iteration_diagnostics` field, so
`blocks.py:586`'s `getattr(cfg, 'track_iteration_diagnostics', False)` always
falls through to `False` (it is the sole writer of the FFN flag). The EM_CONFIG
key instead drives the trainer-side `IterationDiagnosticsTracker`. The cost when
the flag is forced on the module is real (probe: +1 `.cpu()`, +3 `.item()`), but
it never activates from EM_CONFIG. The genuine finding is the wiring gap:
`EM_CONFIG['track_iteration_diagnostics']=True` silently does not enable in-FFN
per-iteration grad-norm collection, so those console rows degrade to 0.0. Fix (if
the user wants real E-step grad norms): add the field to `BlockConfig` +
`from_config` and thread it to the FFN.

`PERF-3` — the dense N×N pair grid is computed under a causal mask;
`causal_lower_triangle=True` packs to M=N(N+1)/2 pairs and is documented
bit-identical for β/grad_μ/grad_σ, but is off in EM_CONFIG. For N=128 this is a
~2× reduction on the dominant `(B,N,N,K)` allocations and pair einsums
(`grad_kl_per_pair_full` alone ≈ 42 MB). Opt-in; verify bit-equivalence on the
live config before adopting (`vfe_gradients.py:1029-1042` claims identity).

`DS-4` — `non_flat_transport=True` combined with `skip_attention=True` is a silent
no-op (Confirmed latent bug). The `gauge_connection` is allocated
(`blocks.py:692`) but its only training-forward call site (`blocks.py:815`) lives
inside the bypassed `if not self.skip_attention:` block, so `delta_ij` stays None
and the FFN receives `connection_delta=None`. There is no `__post_init__` warning
for this combination (unlike the mixer / detaching-EM warnings). Note: the
connection module is also called at `transformer/analysis/publication_metrics.py:2201`
(offline diagnostic), so any fix must not break that path. Fix: route the δ_ij
computation onto the skip path, or add a `__post_init__` warning.

### Low

`EM-1` — `raw_sigma_lr` receives no gradient under the live config (Confirmed by
control). Its only output is the per-token belief σ_q, which never reaches the
mu-only linear decode (`use_prior_bank=False`, `use_output_projection=False`), and
with `n_layers=1` there is no later layer to consume it. Control: flipping
`use_prior_bank=True` makes the grad nonzero (1.3e-9). Allocated-but-ignored dead
weight, not a correctness bug — σ genuinely cannot affect mu-only predictions.

`EM-2` — `norm2.{weight,bias}` receive no gradient under `skip_attention=True`
(Confirmed by control). The single VFE sublayer uses `norm1` (`blocks.py:983-989`
else-branch); `norm2` is never invoked. `norm1.{weight,bias}` are alive
(0.0164/0.0159), proving this is a skip artifact, not a broken norm.

`PERF-1b` — `_compute_phi_grad` declares `cached_block_exp_pairs` (`:1191`) but
never reads it (Confirmed, 0 rvalue reads via AST). The caller passes
`_mh_cached_bep` (`:2390`); it is silently discarded because the autograd path
needs a BEP tied to the fresh `phi_for_grad` leaf. Dead parameter — remove or
document. (An analytic ∂F/∂φ could actually use it.)

`PERF-4` — `IrrepMultiHeadAttention` is fully constructed under skip_attention
(Confirmed exactly). It holds 0 trainable params but 40000 buffer elements
(`head_generators.{0,1}.gen`, each 200×10×10) duplicating the diagonal sub-blocks
of `ffn.generators` (200×20×20). ~160 KB; a footprint/`.to()`/state-dict
cleanliness item, resolved by the attention-removal refactor.

`NUM-1` — `sigma_max` default (5.0) ≠ live (12.0); docstrings/comments compute the
worst-case Fisher amplification against the default. The retraction is correct and
triple-bounded; with `ffn_n_iterations=1` the 2σ² growth is not even iterated.
Comment/default mismatch only.

`E-1` — the √K temperature factor is realized as per-head d_h=10, not embed_dim
K=20 (`vfe_gradients.py:1499`, μ_q sliced per head at `variational_ffn.py:2000`).
This is the scaled-dot-product convention (scale by √d_head); defensible and
internally consistent. The manuscript τ=κ√K should read K=d_h.

`F01-G` — `_retract_phi` selects the GL(K) branch (and hence det control) via
`n_gen` auto-detect rather than an explicit `gauge_group`. For the live n_gen=200
it resolves correctly, but a future irrep layout whose n_gen collides with
N(N-1)/2 would be misclassified as SO(N), silently disabling
`phi_trace_clamp`/`phi_project_slk`. Fix (hardening): pass `gauge_group` explicitly
into the `_retract_phi` call (`variational_ffn.py:2397`).

### Info / pass

`F01-A` diagonal σ transport equals the diagonal of the sandwich and is used
consistently across KL, β, and both gradients. `F01-B` per-head block-diagonal Ω.
`F01-E` det(Ω) bounded to [0.22, 4.48] under `phi_trace_clamp=0.75`, re-applied to
the evolved φ each iteration. `F01-F` the pure equivariant path is reachable.
`E-3`/`E-4` the query-side envelope handling and the α-divergence application are
exact (rel err 0.0 vs independent autograd). KL safe-clamp, Killing preconditioner,
softmax max-subtraction, and the autocast-disabled float32 hot spots are all sound.

## Appendix — refuted or corrected claims (do not re-investigate)

- `PERF-2` as a per-forward perf cost: refuted. The EM_CONFIG flag never reaches
  the FFN; live per-forward sync cost is zero. Survives only as a config-wiring
  note (recast above).
- Memo 07 `BR-4` "re-home `precompute_head_transports` onto the FFN": overstated.
  Under a skip-hardcoded refactor the `model.py:593-597` call is dead-output and is
  deleted, not re-homed; the method must merely survive on the
  `IrrepMultiHeadAttention` class because `transformer/baselines/hybrid_gauge_transformer.py:489,567`
  also call it (memo 07 missed those two callers).
- Memo 06 `DS-4` "ONLY call site 815": overstated. `gauge_connection` is also
  called at `publication_metrics.py:2201` (offline diagnostic).
- Memo 07 `BR-2` line reference "560–565" for the attention `log_kappa` allocation:
  the attention-side allocation is `attention.py:1533-1540`; lines 560–568 are the
  FFN's own copy in `variational_ffn.py` (files conflated; logic correct).
- Memo 07 `BR-3`/`BR-4` framing that the `model.py` cache machinery must be
  re-homed: corrected — under skip it is dead-output (`cached_head_transports` is
  never forwarded to the FFN; consumed only inside `attention.forward:1925-1927`)
  and should be deleted. `gauge_param='omega'` still works under skip because
  `omega` threads to the FFN independently.

## Equivalence baseline (for the refactor gate)

`docs/audit_workspace/equivalence_harness.py` captures loss + a full
(param → grad-norm) table on the tiny live-patterned skip model with pinned
weights (`baseline_skip_attention.weights.pt`) and construction-RNG-independent
inputs; `--gate` reproduces to `atol=1e-6, rtol=1e-5` (passed twice). Scope
limitation flagged by verifier A: this baseline is under `gauge_mode='learned'`,
where the attention sublayer is never called, so it passes trivially for the
attention-removal refactor and does not exercise the `constant_omega` path where
the BR-1 hazard bites. The refactor plan therefore requires a companion
`gauge_mode='constant'` baseline.
