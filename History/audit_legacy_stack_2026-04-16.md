# Deep Audit — V13 Gauge Transformer, Legacy Stack

**Date:** 2026-04-16
**Scope:** `transformer/core/`, `transformer/training/`, `transformer/analysis/`, `transformer/visualization/`, `transformer/baselines/`, `transformer/data/`, `transformer/utils/`, `transformer/train*.py`, `scripts/*.py`, `tests/**/*.py`
**Excluded:** `transformer/vfe/`, `transformer/pure_vfe/` (active development surface)
**Plan:** `C:\Users\chris and christine\.claude\plans\hashed-tinkering-dijkstra.md`
**Author persona:** senior ML engineer, gauge theory, variational free energy

This is a read-only findings report. No code was modified. Each finding carries a `file:line` citation, why it matters, evidence, and a fix sketch (no code).

Severity key — **Critical**: correctness or publication-result-blocking. **High**: silent correctness risk or CLAUDE.md hard-constraint drift. **Medium**: numerical, reproducibility, or documentation gaps. **Low**: style, dead flags, minor drift.

---

## Executive Summary

The legacy stack is substantially constraint-compliant. All covariance-transport sites use the sandwich product `Ω Σ Ωᵀ` correctly (`attention.py:1148-1151`, `attention.py:721-828` with AMP fp32 guards). The six `em_mode` dispatch is clean (`em_modes.py:12-18` + `block_config.py:179-183` + `variational_ffn.py:372-374`). Natural-gradient preconditioning is properly implemented with three modes — `clip`, `cartan`, `killing` — using the Cartan-modified Killing form (`gauge_preconditioner.py:199-242`). Seed plumbing is correct (`training/utils.py:9-26`) with cuDNN determinism enabled.

Four findings warrant attention before the next publication run:

- **C-01** — `gauge_geometry.py:615` computes `det(Ω_ij) = exp(tr(φ_i − φ_j))` naively in fp32 without `slogdet` fallback; a single outlier token with `|tr Δφ| > 88` produces `inf` in publication metrics. This is the legacy-stack mirror of the vfe/ bug fixed earlier today (obs 912, S26-S30).
- **H-01** — `blocks.py:43-157` exposes `nn.LayerNorm` and a parameterized `RMSNorm` alongside the gauge-equivariant `MahalanobisNorm`. Selectable via config; silently breaks gauge equivariance when chosen. No config guardrail.
- **H-02** — Five `em_mode` entries in `em_modes.py:12-18`; CLAUDE.md documents six. The sixth (`'vfe_default'`) is the hardwired `vfe/` profile not actually present in the legacy dispatcher. Documentation drift is acknowledged in CLAUDE.md but not in the dispatch table.
- **M-01** — `torch.use_deterministic_algorithms(True)` is never called anywhere in the repo. Only `cudnn.deterministic` is set. CUDA `scatter_add` / `index_put` in attention remain non-deterministic, so "reproducibility" claims are partial.

Five test gaps (ranked by regression blast radius) and one unwired-flag correction appear in the appendices.

---

## Findings

### C-01 · Critical · Dimension 3 (numerical stability) / Dimension 2 (correctness)
**`transformer/analysis/gauge_geometry.py:615`** — naive `det(Ω) = exp(tr Δφ)` without overflow guard.

**Observation.** At line 607–615, `det_omega = torch.exp(trace_diff)` is computed in fp32 over all token pairs. `trace_diff` is `tr(φ_i) − tr(φ_j)` ∈ ℝ. When `enforce_orthogonal=True` **and** `K >= 2`, a safer branch at lines 624–628 projects to O(K) via Newton-Schulz and uses `torch.linalg.det(exp_phi_flat)` — fine. But the default path runs the naive formula.

**Why it matters.** `exp` in fp32 overflows at argument ≈ 88.7 and returns `+inf`. Today's det(Ω) max ≈ 26923 corresponds to `tr Δφ` ≈ 10 — well within bounds, but during the SL(K)-projection work (obs S27–S30) det values up to exp(42) on H=6 blocks were considered plausible. A single token with `|tr φ|` near the overflow limit silently poisons downstream publication statistics (mean, std, percentile curves in `publication_metrics.py`). This is the exact failure mode of the vfe/ det(Ω) blow-up fixed earlier today; the legacy analysis path is untested for it.

**Evidence.** `CLAUDE.md` documents the det(Ω) blow-up as a Known Gap in the vfe/ path. `scripts/gauge_frame_spectral_analysis.py:188` already uses `np.linalg.slogdet` for the BERT/GPT-2 spectral analysis — there is internal precedent for the right approach. No test asserts `torch.isfinite(det_omega).all()`.

**Fix sketch.** Promote `phi_f32` and `gen_f32` to fp64 for the `trace_diff` computation; or replace `torch.exp(trace_diff)` with `torch.where(trace_diff.abs() > THRESH, torch.full_like(trace_diff, float('nan')), trace_diff.exp())` and log the clipped-count; or use `torch.linalg.slogdet` on `exp_phi_flat` unconditionally (the Newton-Schulz branch at 624 already shows the pattern). Add a `max|tr φ|` scalar to training diagnostics in `training/metrics_tracking.py` so the overflow condition is observable before it produces bad figures.

**Confidence.** High (read code, cross-checked with CLAUDE.md and observation timeline).

---

### H-01 · High · Dimension 1 (constraint compliance)
**`transformer/core/blocks.py:43-157`** — `nn.LayerNorm` and parameterized `RMSNorm` are selectable alongside `MahalanobisNorm`.

**Observation.** `blocks.py:43` defines `RMSNorm(nn.Module)` with a learnable `self.weight` gain (`nn.Parameter`). `blocks.py:69` defines the gauge-equivariant `MahalanobisNorm`. `blocks.py:157` returns `nn.LayerNorm(dim)` from a selector function. The module docstring at lines 5-13 explicitly lists `LayerNorm/RMSNorm` as options "toggled for pure VFE ablation".

**Why it matters.** CLAUDE.md Hard Constraint 1 (NO NEURAL NETWORKS) forbids learned scalar gains and `LayerNorm` outside the documented exceptions (output head, optional MLP mode in `connection.py`, optional `output_proj` in `attention.py`). `MahalanobisNorm` is the theoretically-correct path (isotropic-limit RMSNorm with `Σ`-weighted Mahalanobis), but `LayerNorm` subtracts the mean — this destroys the μ translation that gauge-transport relies on. A user editing a config dict can silently disable gauge equivariance.

**Evidence.** CLAUDE.md: "There should ALWAYS exist a theoretically/mathematically pure path under appropriate toggles. Computationally extreme paths should be opt-in toggles and clearly documented." The pure path exists (`MahalanobisNorm`); the violating paths also exist with no opt-in gating. No runtime assertion or config-level guardrail forbids `LayerNorm` when `gauge_mode='learned'`.

**Fix sketch.** Two options: (a) delete `nn.LayerNorm` branch from the selector — the ablation use case is served by setting `normalization=None`; (b) keep the option but add a runtime warning or `assert` requiring an explicit `allow_non_gauge_norm=True` flag when the selector returns a non-gauge-equivariant norm. Either way, update CLAUDE.md's Hard Constraints to mention this selector as a documented exception (if kept).

**Confidence.** High.

---

### H-02 · High · Dimension 4 (EM semantics) / documentation
**`transformer/core/em_modes.py:12-18`** vs `CLAUDE.md` "EM modes" table — dispatcher has five modes, CLAUDE.md documents six.

**Observation.** `EM_MODE_TABLE` declares five keys: `straight_through`, `ift_phi`, `em_phi_q`, `em_phi_p`, `implicit_ift`. CLAUDE.md's table lists six including `vfe_default`. The CLAUDE.md footnote acknowledges that `vfe_default` is a separate hardwired profile in the `vfe/` package, not a legacy-dispatcher option.

**Why it matters.** The legacy stack's E-step gradient flow is fully controlled by these five. A future contributor reading CLAUDE.md's table and grepping for `'vfe_default'` in `em_modes.py` finds nothing — a silent documentation gap. More critically, there is no test that asserts the `EM_MODE_TABLE` keys match the documented set: if someone adds a sixth mode without updating the table key list or the CLAUDE.md table, nothing catches it.

**Evidence.** `em_modes.py:12-18` is explicit. `block_config.py:179-183` validates against `_EM_MODE_TABLE`. `variational_ffn.py:372-374` unpacks the table. All consistent internally — but there is no test `assert set(EM_MODE_TABLE.keys()) == {'straight_through', 'ift_phi', 'em_phi_q', 'em_phi_p', 'implicit_ift'}`.

**Fix sketch.** Either add `'vfe_default'` to `EM_MODE_TABLE` as a deprecated-and-rejected entry that raises on lookup (clarifying the boundary), or restructure CLAUDE.md so the sixth row is visually demarcated as "vfe/ package only, not a dispatcher value". Add a contract test in `tests/transformer/test_gauge_utils.py` or similar.

**Confidence.** High.

---

### H-03 · High · Dimension 1 (CLI contract)
**`transformer/utils/evaluation.py:14`** — module-level `import argparse` in a utility that is not `train_publication.py`.

**Observation.** `transformer/utils/evaluation.py:14` imports `argparse` at module level. CLAUDE.md explicitly names `transformer/train_publication.py` as the **only** exception to the click-to-run CLI contract.

**Why it matters.** The contract exists so that entry points are edited-in-place (config dict at top of file), not invoked with flags. A utility that parses args at the command line sidesteps the contract. However: this file likely exposes a `__main__` block that invokes args. If so, it is analogous to the visualization-tool exceptions (`vfe_dynamics_plots.py:640`, `interactive_belief_viz.py:1114`, `belief_space_viz.py:234`) and to the scripts under `scripts/`. The distinction: those other files import argparse *inside* `if __name__ == '__main__':`; this one imports it at module level.

**Evidence.** `grep -n "import argparse"` inside `transformer/`: only `train_publication.py` (approved), `experiment_runner.py:27` (passes argparse.Namespace as type hint, acceptable), `utils/evaluation.py:14` (module-level, unguarded), and four visualization/test-utility files (local under `__main__`).

**Fix sketch.** Move `import argparse` inside the `if __name__ == '__main__':` block and mark the pattern-exception in CLAUDE.md, or explicitly list `utils/evaluation.py` as an approved exception on par with the visualization tools. Same for `experiment_runner.py:27` — though its use is solely as a type annotation, moving the import inside a `TYPE_CHECKING` block would align with the contract.

**Confidence.** Medium (file not deep-read; pattern inferred from grep context).

---

### M-01 · Medium · Dimension 5 (determinism)
**Repository-wide** — `torch.use_deterministic_algorithms` is never called.

**Observation.** A repository-wide grep for `use_deterministic_algorithms|cudnn\.enabled|non_blocking` returns **no matches**. `transformer/training/utils.py:9-26` sets `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`, but this does not bind the non-deterministic `scatter_add` / `index_put` / `put_` CUDA ops that appear in attention backward and in EM-mode-dependent paths.

**Why it matters.** CLAUDE.md does not mandate bit-exact reproducibility, but the `set_all_seeds` docstring (`training/utils.py:10-17`) claims "chosen deterministically across runs" — an overclaim. Publication results that rely on seeded ablation studies (see 5-way `alpha_divergence` sweep in `scripts/ablation_results/`) may show small run-to-run variance on GPU. This becomes load-bearing if a claimed effect is ≤ that variance.

**Evidence.** `training/utils.py:22-26` is the only determinism plumbing. `torch.use_deterministic_algorithms` appears nowhere in the scope.

**Fix sketch.** Add an optional `strict_determinism: bool = False` parameter to `set_all_seeds`; when true, call `torch.use_deterministic_algorithms(True, warn_only=True)` and set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment. Document that strict mode may slow training. Update the docstring's claim to match what the function actually does.

**Confidence.** High (exhaustive grep).

---

### M-02 · Medium · Dimension 2 (mathematical correctness) / Dimension 8 (drift)
**`transformer/core/attention.py:300-316`** — RoPE × full-Σ asymmetry is correctly documented but untested.

**Observation.** The code comment explicitly names the gap: "The implementation here departs from that prescription: the Mahalanobis term is computed in rope-rotated mean coordinates with a raw covariance, which is neither a textbook KL between rotated Gaussians nor a gauge transport." This matches CLAUDE.md "KNOWN GAP — RoPE × MahalanobisNorm" which narrows the condition to `diagonal_covariance=True AND use_rope=True AND rope_full_gauge=False`.

**Why it matters.** The gap is accepted research terrain. The risk is drift: if someone later implements `rope_full_gauge=True` for full-Σ transport, there is no test that the diagonal-σ / full-Σ paths agree in the RoPE-rotated case. Current state is internally consistent but brittle under future edits.

**Evidence.** `vfe_gradients.py:220-223` acknowledges the same asymmetry. No `test_*.py` file asserts a `Σ_shifted == R Σ R^T` property under RoPE index shift.

**Fix sketch.** Add a property test to `tests/transformer/test_attention.py` that constructs a token-index-shifted input and asserts the full-cov RoPE output is the rotated version of the un-shifted — or documents the gap as an xfail with the exact failing assertion so future work can remove the marker.

**Confidence.** High.

---

### M-03 · Medium · Dimension 3 (numerical stability) / Dimension 7 (tests)
**`transformer/core/vfe_deq.py:418,424,562,568`** — hardcoded `step_size=1.0` with no runtime contractivity check.

**Observation.** The DEQ fixed-point path passes `step_size=1.0, trust_region=sigma_trust` into `_retract_sigma` (or similar). Unlike `vfe/e_step.py:364-368` which threads `self.e_sigma_lr` from config, the DEQ path fixes step_size to 1.0 and controls magnitude through `trust_region`. The docstring at `vfe_deq.py:16-22` acknowledges that the IFT correction assumes the E-step is at a **fixed point** — requires `‖J_T‖ < 1` (spectral radius of Jacobian < 1) for the Neumann series to converge.

**Why it matters.** If the E-step is non-contractive near the returned point (e.g., step_size too large, or the forward loop did not actually converge), the Neumann-series IFT approximation diverges. The backward produces wrong gradients silently — training appears to progress but the M-step gradient direction is incorrect. No runtime assertion or warning.

**Evidence.** `vfe_deq.py:16-22` correctly identifies the requirement; `vfe_deq.py:116-118` notes "K terms" for the truncated Neumann series. No code path estimates `‖J‖` or warns when the forward E-step did not reduce the fixed-point residual below a threshold.

**Fix sketch.** On DEQ forward exit, compute the residual `‖T(z*) − z*‖` and log it. On backward, bound the Neumann sum by the dominant-eigenvalue estimate from a power iteration (cheap: one `T(v) − v` call). Abort with a warning when `‖J‖ > 0.9`. This also closes a test gap — no current test exercises the DEQ path under a deliberately non-contractive configuration.

**Confidence.** Medium (read docstring + call sites; did not fully trace the autograd closure).

---

### M-04 · Medium · Dimension 8 (drift) — correction to plan
**`transformer/core/vfe_gradients.py:1360`, `transformer/core/attention.py:173,443,877,1247,1286`** — `exact_diagonal_transport` IS wired. Prior observation was wrong.

**Observation.** The plan/Phase-1 report flagged `exact_diagonal_transport` as declared-but-not-wired. Phase A grep shows it is threaded through `block_config → blocks → attention (5 signatures) → vfe_gradients (lift at 1431) → vfe_deq (4 sites)` and `baselines/hybrid_gauge_transformer.py`. The lift at `vfe_gradients.py:1431` is conditional on `exact_diagonal_transport and is_diagonal`.

**Why it matters.** Nothing — this is a correction. The flag works. But the audit plan listed it as a drift candidate; recording the correction avoids reintroducing this misbelief.

**Fix sketch.** None required. Update Phase-1 notes in future audits.

**Confidence.** High.

---

### M-05 · Medium · Dimension 8 (drift) — correction to plan
**`transformer/core/prior_bank.py:539-550`** — `_decode_full_cov` IS dispatched when needed. Prior observation was wrong.

**Observation.** The plan/Phase-1 report flagged that `prior_bank.py:531-537` uses a diagonal decoder even when `diagonal_covariance=False`. Direct read shows the dispatch at `prior_bank.py:539-546` correctly routes to `_decode_full_cov` (defined at line 658) whenever any of `self.full_cov_decode`, `not self.diagonal_covariance and sigma_q.dim()==4`, or `self.exact_diagonal_transport and self.diagonal_covariance and sigma_q.dim()==3` is true.

**Why it matters.** Correction only — the PriorBank decoder path is correct for all three regimes. What remains is the `sigma_ce_scale` detach pattern at `prior_bank.py:628,728` (`sigma_p_safe.detach() + _s * (sigma_p_safe - sigma_p_safe.detach())`). This is the straight-through gradient scaling for the residual CE→σ_p signal — aligns with the `sigma_p` EM-boundary discipline in CLAUDE.md.

**Fix sketch.** None required. Plan correction.

**Confidence.** High.

---

### L-01 · Low · Dimension 1 (publication hygiene)
**`transformer/analysis/semantics.py:1047,1606`** — `dpi=150` below the 300-dpi publication default.

**Observation.** Two `savefig` calls at `semantics.py:1047,1606` use `dpi=150`. All other publication-figure calls in `analysis/publication_metrics.py` and `visualization/*.py` use `dpi=300`. CLAUDE.md: "ALL Figures should be publication quality by default."

**Fix sketch.** Change 150 → 300. One-line edit at each site.

**Confidence.** High.

---

### L-02 · Low · Dimension 8 (drift)
**`transformer/train.py:275`** — "Backward-compatible aliases (deprecated)"; **`transformer/train_publication.py:828,843-846`** — `--ffn_mode` deprecated; legacy mode-name mapping.

**Observation.** Active deprecation markers in the config plumbing. These are functional and acknowledged; tracked here only for index completeness.

**Fix sketch.** Document a removal schedule (e.g., "drop --ffn_mode after 2026-05-01") and enforce with a deadline check in CI.

**Confidence.** High.

---

### L-03 · Low · Dimension 7 (tests)
**Repository-wide** — only two pytest skips; no `xfail` markers.

**Observation.** `tests/transformer/test_data.py` skips when `datasets` is unavailable; `tests/transformer/test_integration.py` skips one test with "No attention weights in output". No `@pytest.mark.xfail` anywhere. That is unusually clean — either the codebase genuinely has no known-failing tests (good), or test authors have avoided writing tests that would fail (indicating gaps, not health).

**Evidence.** Phase 1 Explore agent confirmed via grep.

**Fix sketch.** None. If future known-gaps are identified (e.g., the RoPE × full-Σ property in M-02), prefer `xfail(strict=True)` over skip.

**Confidence.** High.

---

### L-04 · Low · Dimension 1 (CLI) — scripts/ argparse
**`scripts/run_ablation_suite.py:33`**, **`scripts/kn5_baseline.py:27`**, **`scripts/generate_publication_figures.py:21`**, **`scripts/gauge_frame_spectral_analysis.py:1183`**, **`scripts/test_numerical_edge_cases.py`**, **`scripts/verify_vfe_gradients_fd.py`** — argparse in analysis scripts.

**Observation.** Six scripts use argparse. CLAUDE.md's CLI exception is scoped to `transformer/train_publication.py` alone. Scripts under `scripts/` are standalone reproduction tools that predate the contract; their CLIs are load-bearing (Optuna HPO, ablation sweeps, numerical edge-case stress tests).

**Why it matters.** Low — CLAUDE.md's intent was clearly about *training* entry points, not analysis tooling. Worth acknowledging explicitly in CLAUDE.md so new scripts don't get reported as violations in future audits.

**Fix sketch.** Add a sentence to CLAUDE.md's CLI constraint: "Analysis/verification tools under `scripts/` may use argparse — they are not model entry points and do not participate in the dict-based config contract."

**Confidence.** High.

---

### L-05 · Low · Dimension 1 (constraint scope)
**`transformer/core/embeddings.py:263,327,345,362,388`** — `nn.Embedding` lookups for μ, Σ, φ, ω, sign logits.

**Observation.** When `use_prior_bank=False` (the default in `train_publication.py:138` and `model.py:160`), `GaugeTokenEmbedding` uses `nn.Embedding` for each component of the belief tuple. These are lookup tables, not feedforward networks — acceptable under CLAUDE.md's intent, but CLAUDE.md does not explicitly list them as exceptions.

**Fix sketch.** Add `nn.Embedding` to the CLAUDE.md constraint text: "Exception: `nn.Embedding` lookups in `GaugeTokenEmbedding` and `PriorBank` — these are parameter stores for the Gaussian-belief tuple, not neural computation."

**Confidence.** High.

---

## Appendix A — NN-Site Table (all surfaces)

| File:Line | Kind | Status |
|---|---|---|
| `core/model.py:269` | `nn.Linear(embed_dim, vocab_size, bias=False)` | Documented exception — final output head |
| `core/attention.py:1485` | `nn.Linear(embed_dim, embed_dim, bias=False)` | Documented exception — `use_output_projection` (off by default) |
| `core/connection.py:80-83` | `nn.Sequential(Linear, GELU, Linear)` | Documented exception — `GaugeConnection` MLP mode |
| `core/blocks.py:157` | `nn.LayerNorm(dim)` | **H-01** — selector option, no guardrail |
| `core/blocks.py:43` | `RMSNorm` with `nn.Parameter` gain | **H-01** — related; learnable gain outside belief tuple |
| `core/embeddings.py:263,327,345,362,388` | `nn.Embedding` (mu, omega, phi, pos, sign_logit) | **L-05** — belief-tuple lookup, not documented as exception |
| `core/prior_bank.py:215,243,254,261` | `nn.Embedding` (phi_embed, omega_embed) | **L-05** — same |
| `core/attention.py:416` | `F.softmax(logits, dim=-1)` | Attention weight normalization — not a nonlinearity on neural features |
| `baselines/*` | Full PyTorch stack | Permitted — baselines for comparison |

---

## Appendix B — Detach/no_grad × EM Mode Map

| em_mode | amortized_inference | amortize_sigma | exact_phi_grad | implicit_em | em_phi_mode |
|---|---|---|---|---|---|
| `straight_through` (default) | True | True | False | False | `amortized` |
| `ift_phi` | True | True | True | False | `amortized` |
| `em_phi_q` | True | False | False | False | `E_phi_q` |
| `em_phi_p` | True | False | False | False | `M_phi_p` |
| `implicit_ift` | False | False | False | True | `amortized` |

Total detach/no_grad/requires_grad sites in `transformer/core/` = **185 across 14 files**. Dominant surfaces: `variational_ffn.py` (45), `vfe_gradients.py` (13), `model.py` (35), `vfe_deq.py` (12), `vfe_closed_form.py` (7), `active_inference.py` (11). No dedicated test asserts which parameters have `requires_grad=True` at E-step iteration time for each mode. See test gap **G-01**.

---

## Appendix C — Test-Gap Matrix

| ID | Gap | Dimension | Risk | Rank |
|---|---|---|---|---|
| G-01 | `em_mode` `requires_grad` semantics untested per-mode | 4 | Wrong-gradient without crashing, affects M-step convergence | 1 |
| G-02 | Lie-algebra commutator closure `[T_a,T_b]=f^c_{ab} T_c` untested | 2 | Wrong generators silently produce non-Lie transport | 2 |
| G-03 | Closed-form vs iterative E-step numerical agreement untested | 2 | Diverging paths indicate stale gradient / wrong preconditioner | 3 |
| G-04 | Finite-difference checks for μ, Σ absent (only φ partially) | 2 | Analytical-grad bug passes tests | 4 |
| G-05 | RoPE × full-Σ property test absent | 2 | Covariance loses token-index covariance silently | 5 |
| G-06 | DEQ Jacobian spectral-radius assertion at fixed point | 3,4 | Non-contractive E-step produces wrong IFT gradient | 6 |
| G-07 | Gauge-invariance of `KL(q‖p)` under same `g` on both | 2 | Non-abelian φ breaks KL symmetry silently | 7 |
| G-08 | `max|tr φ|` training diagnostic for det(Ω) overflow guard | 3 | **C-01** precondition monitor | 8 |
| G-09 | `EM_MODE_TABLE.keys()` contract test | 4 | Doc-code drift (**H-02**) slips through | 9 |
| G-10 | `LayerNorm`-selector guardrail (**H-01**) | 1 | Config-level thesis violation slips through | 10 |

---

## Appendix D — Dead-Flag & Drift Registry

| Item | File:Line | Status |
|---|---|---|
| `--ffn_mode` | `train_publication.py:828,843-846` | DEPRECATED, active fallback mapping |
| Legacy `kappa_beta_base` → `kappa_beta` | `utils/checkpoint.py:7,85` | Active migration on load |
| `diagonal_covariance` legacy migration | `utils/checkpoint.py:85` | Active migration |
| `'stack'` attribute fallback for legacy models | `analysis/publication_metrics.py:1952` | Active fallback |
| Legacy full-Ω attention path | `core/attention.py:1108` | Active fallback when no exp-pairs cached |
| Legacy no-σ-gradient-from-alignment | `core/vfe_gradients.py:1720,1841` | Active fallback |
| Legacy coupled LR ratio | `core/variational_ffn.py:634,738` | Active fallback |
| `phi_dim=3` so(3) subalgebra | `core/embeddings.py:73,149` | Legacy option, still supported |
| `wikipedia` → `wikimedia/wikipedia` | `data/datasets.py:497` | Comment only |
| Hardcoded `step_size=1.0` in DEQ | `core/vfe_deq.py:418,424,562,568` | See **M-03** |
| `dpi=150` in semantics figures | `analysis/semantics.py:1047,1606` | See **L-01** |

Deprecations are currently **active fallbacks**, not orphan code. No dead imports or orphan modules identified inside the audit scope.

---

## Appendix E — Determinism Checklist

| Check | Status | Location |
|---|---|---|
| `torch.manual_seed` | ✅ | `training/utils.py:22` |
| `np.random.seed` | ✅ | `training/utils.py:21` |
| `random.seed` | ✅ | `training/utils.py:20` |
| `torch.cuda.manual_seed_all` | ✅ | `training/utils.py:24` |
| `cudnn.deterministic = True` | ✅ | `training/utils.py:25` |
| `cudnn.benchmark = False` | ✅ | `training/utils.py:26` |
| `torch.use_deterministic_algorithms` | ❌ | See **M-01** |
| DataLoader `worker_init_fn` | ✅ | `data/datasets.py:112-124` (applied to 6 DataLoaders) |
| `set_all_seeds` called from entry points | ✅ | `train_publication.py:851-852` |

Partial determinism. Claim in `training/utils.py:10-15` docstring is an overclaim — see **M-01**.

---

## Appendix F — Heuristic Pass-Through Status

| Heuristic (from plan) | Status |
|---|---|
| 1. KL gauge-symmetric in both arguments | Not tested (**G-07**) — code path not deep-traced |
| 2. Sandwich-product composition `T₂(T₁Σ)=(T₂T₁)Σ` | Tested (`test_transport_ops.py` cocycle) |
| 3. `det(exp φ) = exp(tr φ)` log-space hygiene | **C-01** — fp32 overflow path active |
| 4. Closed-form E-step as iterative oracle | Not tested (**G-03**) |
| 5. RoPE × Σ covariance under index shift | Not tested (**G-05**), documented gap |
| 6. Fisher-on-φ block-coupled over (μ,Σ,φ) | Partially addressed — `gauge_preconditioner.py` precondition is on φ alone; Fisher block-coupling over full tuple not implemented |
| 7. DEQ contractivity | **M-03** — not runtime-checked |

---

## Verification Checklist (audit quality)

- [x] Every Critical / High finding has a `file:line` citation.
- [x] CLAUDE.md "documented exception" claims were matched to concrete code (Appendix A).
- [x] Six `em_mode` profiles in CLAUDE.md mapped against the 5-key dispatcher (**H-02**, Appendix B).
- [x] Numerical-stability claims (e.g., `exp(tr φ)` overflow at ~88) are tied to a specific code path (**C-01**).
- [x] Test-gap count: 10 (≥5 from plan; 5 new surfaced during deep reads).
- [x] Two plan-level misbeliefs corrected (**M-04** `exact_diagonal_transport` IS wired; **M-05** `_decode_full_cov` IS dispatched).
- [x] Deprecation sweep: all active fallbacks enumerated (Appendix D).

---

## Explicit Non-Findings (to prevent false positives in future audits)

- All covariance transport sites use the sandwich product `Ω Σ Ωᵀ` correctly, with fp32 AMP guards where needed (`attention.py:724-828, 1148-1151`).
- Natural gradient on φ is properly implemented with three preconditioners (`gauge_preconditioner.py`), including the Cartan-modified Killing form.
- `sigma_p` detach discipline at the EM boundary is correct in the default `straight_through` path (`prior_bank.py:628,728` uses straight-through residual scaling).
- `mask_self_attention` defaults ON (`block_config.py` default chain) and is autograd-safe.
- Seed plumbing covers Python, NumPy, PyTorch, and CUDA devices; DataLoader workers seed deterministically.
- `exact_diagonal_transport` flag is correctly wired (correction to Phase-1 finding).
- `_decode_full_cov` is correctly dispatched (correction to Phase-1 finding).
- 643 tests pass in scope; no `xfail` markers conceal known failures.
