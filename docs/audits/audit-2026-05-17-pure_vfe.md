# Deep Audit — `transformer/pure_vfe/` — 2026-05-17

Branch: `fix/holonomy-numerics-2026-05-05`. Five parallel investigators
swept the `transformer/pure_vfe/` package; one independent verifier
re-read every cited line and confirmed or refuted each claim.

The user originally referred to the package as "pure_fep"; the actual
directory is `transformer/pure_vfe/` (the pure variational-free-energy
rewrite, distinct from `transformer/vfe/`).

## Scope

Audit target: the `transformer/pure_vfe/` package (8 modules, ~3.3k LOC).
Out of scope: tests under `transformer/pure_vfe/tests/`, all other
packages.

Files audited:
- `__init__.py` (9), `config.py` (135), `cuda_ext.py` (64),
  `gauge.py` (615), `gaussians.py` (744), `inference.py` (507),
  `learning.py` (924), `model.py` (315).

## Investigators

1. **code-reviewer** — quality, security, idioms
2. **debugger** — bugs, gradient flow, theory-correctness vs CLAUDE.md
3. **refactoring-specialist** — dead code, duplication, orphans
4. **performance-engineer** — hot paths, allocations, sync stalls
5. **python-pro** — type safety, contracts, Python idioms

A sixth verifier (`general-purpose`) received all five summaries blind
and re-read every cited line.

## Verifier verdicts (50 findings)

### code-reviewer

| # | Title | Cited | Verdict |
|---|---|---|---|
| 1.1 | Unsafe `torch.load` default permits arbitrary code execution | model.py:240,255 | **CONFIRMED** — default `trusted=True` → `weights_only=False` → pickle RCE |
| 1.2 | Broken return-type annotation `"PureFEPModel"` | model.py:240 | **CONFIRMED** — actual class is `PureVFETransformer` |
| 1.3 | Pervasive missing type hints (~40 functions) | gaussians.py + others | **CONFIRMED** |
| 1.4 | `tau: float = None` type mismatch | config.py:19 | **CONFIRMED** |
| 1.5 | Brittle string dispatch on `gauge_param` | config.py:66; model.py:57,301 | **CONFIRMED** |
| 1.6 | `print()` for status logging in library module | cuda_ext.py:59,61 | **CONFIRMED** |
| 1.7 | `safe_inverse` final fallback unguarded | gaussians.py:62-69 | **CONFIRMED** |
| 1.8 | `retract_spd exp_clip` default inconsistency E vs M step | gaussians.py:541; learning.py:446-449,853-856 | **CONFIRMED** — E-step passes `config.spd_exp_clip` (=5.0); M-step uses fn default (50.0) |
| 1.9 | Dead `beta_ji` variable + misleading comments | gauge.py:332-335 | **CONFIRMED** |
| 1.10 | `init_omega` defaults to `device='cuda'` | gauge.py:554 | **CONFIRMED** — `init_phi` defaults to `'cpu'`; disagreement |

### debugger

| # | Title | Cited | Verdict |
|---|---|---|---|
| 2.1 | `vfe_grad_Omega_full` backward uses wrong beta index | gauge.py:335 | **CONFIRMED (structural)** — `beta_ji` computed but unused; einsum reads `beta_h` |
| 2.2 | Missing ½ factor in analytic KL gradient w.r.t. Ω | gauge.py:129-131 | **CONFIRMED (structural)** — no ½ multiplier in return; math verification deferred |
| 2.3 | `set_block_diag` in-place mutation | inference.py:125-137 | **CONFIRMED (mixed pattern, no actual aliasing bug observed)** |
| 2.4 | `pairwise_kl_torch` in-place `masked_fill_` on autograd tensor | gaussians.py:276 | **INCONCLUSIVE** — `test_analytical_omega_grad_finite` exercises this path and passes; PyTorch's clamp_min backward uses input not output, may be safe |
| 2.5 | `log(0)` via `slogdet(Om)` unguarded | gaussians.py:200 | **CONFIRMED** |
| 2.6 | E-step gauge gradient uses rotated `kl_ij` not `kl_ij_raw` | inference.py:422 | **CONFIRMED but MOOT** — `vfe_grad_Omega` never reads `kl_ij` (phantom param, see 3.3) |
| 2.7 | CUDA `vfe_grad_Sigma_alignment` path skips softmax correction | gaussians.py:466-468 | **REFUTED** — CUDA branch only fires when `kl_ij=None`, where both paths use `w=beta` identically |
| 2.8 | `safe_inverse` inside `enable_grad` loses gradient through fallback | learning.py:89-108 | **REFUTED** — Omega is inverted via raw `torch.linalg.inv` (gaussians.py:181), not `safe_inverse`; the safe_inverse call acts on detached Sigma |
| 2.9 | `prior_Sigma` indexed w/o `.clone()` — storage aliasing | inference.py:224-225 | **REFUTED** — advanced indexing with long tensor copies in PyTorch; no aliasing |
| 2.10 | `grad_kl_Omega_ij` shape check (non-issue) | gauge.py:118-119 | **N/A** — investigator self-declared non-issue |

### refactoring-specialist

| # | Title | Cited | Verdict |
|---|---|---|---|
| 3.1 | `relative_trust_clip` unused | gauge.py:414 | **CONFIRMED** — 1 hit repo-wide (the definition) |
| 3.2 | `vfe_grad_Omega_full` unused in production | gauge.py:246 | **CONFIRMED** — only `test_gradients.py` references it |
| 3.3 | `kl_ij` and `precomp` phantom params in `vfe_grad_Omega(_full)` | gauge.py:161,246 | **CONFIRMED** — never read in function bodies |
| 3.4 | `beta_ji` assigned, never used | gauge.py:332 | **CONFIRMED** |
| 3.5 | `vfe_grad_Sigma_prior` exported, never called in production | gaussians.py:508 | **PARTIALLY CONFIRMED** — only `test_gradients.py` references |
| 3.6 | `grad_kl_Omega_ij` / `grad_kl_Omega_i` test-only | gauge.py:77,134 | **PARTIALLY REFUTED** — `grad_kl_Omega_ij` called by `grad_kl_Omega_i` (gauge.py:150); `grad_kl_Omega_i` itself has zero production callers → chain is test-only |
| 3.7 | `_precompute_obs_gradient` always computes Sigma quantities | learning.py:506 | **CONFIRMED** — wasted O(T·K²) compute when `sigma_obs_grad='none'` (default) |
| 3.8 | `Omega_star_sum` unconditional accumulation | learning.py:250 | **CONFIRMED but size overstated by ~10×** — at V=50257, H=4, K_h=8 it's ~51MB not 512MB |
| 3.9 | Wrong return-type annotation `PureFEPModel` | model.py:240 | **CONFIRMED** (duplicate of 1.2) |
| 3.10 | `natural_grad_omega` test-only | gauge.py:346 | **CONFIRMED** — multiple test refs in `test_omega_gradient.py`; 0 production callers |

### performance-engineer

| # | Title | Cited | Verdict |
|---|---|---|---|
| 4.1 | `.item()` sync storms in every E-step iteration | inference.py:155,358,396,427 | **CONFIRMED** — 4 syncs/iter × n_esteps=12 = 48 syncs/forward |
| 4.2 | Duplicate prior-KL computation every E-step iter | gaussians.py:642 + inference.py:149 | **CONFIRMED** — both call `kl_divergence` with identical args |
| 4.3 | `kl_decode_logits` recomputes safe_inverse + safe_logdet for prior bank every forward | gaussians.py:683-684 | **CONFIRMED** — V=50257 inversions every forward |
| 4.4 | `vfe_grad_Omega` Python H-loop materializing O(B·N²·K_h²) tensors | gauge.py:191-241 | **CONFIRMED** |
| 4.5 | Double `precompute_tokens` + double `pairwise_kl` when `use_rope=True` | inference.py:268,273,295,312 | **CONFIRMED** |
| 4.6 | `retract_spd` 3× `linalg.eigh` per call | gaussians.py:563,575,592 | **CONFIRMED** |
| 4.7 | `extract_block_diag`/`set_block_diag` Python loops | inference.py:118-122,133-136 | **CONFIRMED** — vectorizable via reshape |
| 4.8 | `safe_inverse` uses `linalg.inv` for SPD matrices | gaussians.py:39 | **CONFIRMED** — Cholesky would be ~2× faster |
| 4.9 | `_apply_rope` full `mu.clone()` every E-step iter | inference.py:87 | **CONFIRMED** |
| 4.10 | `_compute_holonomy` per-triangle `.item()` syncs | inference.py:493-504 | **CONFIRMED** — only fires when `use_holonomy=True` on final iter |

### python-pro

| # | Title | Cited | Verdict |
|---|---|---|---|
| 5.1 | Wrong class name in `load()` return annotation | model.py:240 | **CONFIRMED** (dup) |
| 5.2 | `tau: float = None` | config.py:19 | **CONFIRMED** (dup) |
| 5.3 | `assert` for runtime validation (stripped under `-O`) | config.py:132-134; gauge.py:579 | **CONFIRMED** |
| 5.4 | Pervasive missing type hints (30+ functions) | gauge.py:24, gaussians.py:116, inference.py:110 | **CONFIRMED** |
| 5.5 | Stringly-typed config fields | config.py:66,76,117,129 | **CONFIRMED** — `gauge_param`, `sigma_obs_grad`, `lr_schedule`, `device` |
| 5.6 | `get_effective_lrs()` returns unparameterized `dict` | model.py:130 | **CONFIRMED** |
| 5.7 | `to()` mutates caller's config dataclass | model.py:277 | **CONFIRMED** |
| 5.8 | `make_gl_generators device='cpu'` vs `init_omega device='cuda'` | gauge.py:478 vs 554 | **CONFIRMED** |
| 5.9 | `set_block_diag` mixes in-place mutation + return | inference.py:125-137 | **CONFIRMED** |
| 5.10 | `_compute_holonomy` early-exit missing `n_triangles` key | inference.py:468 | **CONFIRMED** — schema inconsistency |

**Tallies:**
- **CONFIRMED**: 42 (39 unambiguous + 3 "confirmed but moot/structural")
- **PARTIALLY CONFIRMED / PARTIALLY REFUTED**: 3 (3.5, 3.6, 3.8)
- **REFUTED**: 3 (2.7, 2.8, 2.9)
- **INCONCLUSIVE**: 1 (2.4 — masked_fill_ in autograd)
- **N/A**: 1 (2.10 — investigator marked non-issue)

## Verifier-only observations

1. **`beta_ji` at `gauge.py:332` is the single most consequential line in the audit.** Three investigators flagged it from different angles. The structural facts are unambiguous (variable computed and unused; einsum uses `beta_h`); the math correctness depends on whether the einsum subscript pattern with `beta_h` is or is not equivalent to the missing transpose. Maintainer should independently re-derive. The function `vfe_grad_Omega_full` is itself test-only (no production callers per 3.2), so the risk is bounded: nothing in `inference.py`/`learning.py` consumes its output today.

2. **`vfe_grad_Omega` ignores its `kl_ij` and `precomp` parameters entirely** (finding 3.3). This makes the call-site inconsistency at `inference.py:422` (passing rotated `kl_ij` instead of `kl_ij_raw`) cosmetic — no behavior change — but still worth fixing for clarity.

3. **`Omega_star_sum` size claim overstated**: investigator #4's "512 MB" should read ~51 MB at the default config (V·H·K_h² = 50257·4·64 floats ≈ 12.9M ≈ 51MB at fp32). Still worth gating behind `use_analytical_omega_grad=False`.

4. **`pairwise_kl_torch` in-place `masked_fill_` (2.4)**: cannot be resolved statically. PyTorch's `clamp_min` backward uses the input tensor, not the output, so the in-place modification of the output post-clamp may not raise. Existing tests pass. Recommend changing to out-of-place `masked_fill(...)` defensively (one-line fix) rather than relying on undocumented backward internals.

5. **Security-relevant finding worth fixing first**: `torch.load` with `trusted=True` default (1.1). Single-line change, real RCE exposure if any third party ever distributes a `pure_vfe` checkpoint. The `safetensors` migration would be a cleaner long-term fix.

## Confirmed punch list (critical → high)

### Critical (5)

1. **[CRITICAL] Unsafe `torch.load` default** — `model.py:240,255`. Flip the
   default to `trusted=False` and require explicit opt-in for non-tensor
   fields; ideally migrate to `safetensors` for state-dict-only artifacts.
2. **[CRITICAL] `.item()` sync storms in E-step** — `inference.py:155,358,
   396,427`. 4 GPU→CPU syncs per E-step iteration (48 per forward at
   `n_esteps=12`). Accumulate as GPU tensors, call `.item()` once after
   the loop.
3. **[CRITICAL] `kl_decode_logits` recomputes V=50257 inverses every
   forward** — `gaussians.py:683-684`. Cache `prior_Sigma_inv` and
   `prior_logdet` on the model; invalidate only after M-step.
4. **[CRITICAL] Duplicate prior-KL per E-step iter** —
   `gaussians.py:642 + inference.py:149`. Compute `kl_prior` once, pass as
   optional kwarg into `state_dependent_alpha`.
5. **[CRITICAL] `vfe_grad_Omega` Python H-loop with quadratic-N
   materializations** — `gauge.py:191-241`. Reuse `precomp['Omega_inv']`
   to avoid re-inverting Om_ij; lift the H-loop into a batched einsum.

### High (12)

6. **[HIGH] Wrong return-type annotation** — `model.py:240` annotates
   `-> "PureFEPModel"` but class is `PureVFETransformer`.
7. **[HIGH] Missing type hints on ~40 public functions** — across
   `gauge.py`, `gaussians.py`, `inference.py`, `learning.py`. CLAUDE.md
   mandates type hints on all signatures.
8. **[HIGH] `assert` used for runtime validation** —
   `config.py:132-134`, `gauge.py:579`. Stripped under `python -O`.
   Replace with `if … raise ValueError(...)`.
9. **[HIGH] `tau: float = None` type contradiction** — `config.py:19`.
   Change to `Optional[float]`.
10. **[HIGH] `retract_spd exp_clip` default mismatch E vs M step** —
    `learning.py:446-449, 853-856` use the fn default (50.0); E-step at
    `inference.py:407` correctly passes `config.spd_exp_clip` (5.0).
    User-configured exp_clip is silently ignored by the M-step.
11. **[HIGH] Dead `vfe_grad_Omega_full`** (`gauge.py:246`) and its
    suspect `beta_ji` (`:332-335`). Either integrate it into the M-step
    (and verify the transpose / ½ factors first) or delete it together
    with its single test.
12. **[HIGH] Dead `relative_trust_clip`** — `gauge.py:414`. 0 callers
    repo-wide. Delete.
13. **[HIGH] Phantom `kl_ij` and `precomp` parameters in
    `vfe_grad_Omega` / `vfe_grad_Omega_full`** — `gauge.py:161,246`.
    Remove from signatures or actually use `precomp` to skip
    recomputed inverses (this is also the route to fix critical #5).
14. **[HIGH] Double `precompute_tokens` + double `pairwise_kl` when
    `use_rope=True`** — `inference.py:268,273,295,312`. Compute only
    `precomp_rope`; derive `kl_ij_raw` for monitoring from the rope
    version with a constant correction.
15. **[HIGH] `retract_spd` runs 3× `linalg.eigh` per call** —
    `gaussians.py:563,575,592`. Reuse first eigh's structure to validate
    retraction analytically; only branch to a third eigh on failure.
16. **[HIGH] `extract_block_diag` / `set_block_diag` Python loops** —
    `inference.py:118-122,133-136`. Replace with a single
    `reshape`/`diagonal`-based view; eliminates the loop.
17. **[HIGH] Brittle string dispatch on `gauge_param`** —
    `config.py:66`, `model.py:57,301`. Promote to `Literal['omega','phi']`
    with `__post_init__` validation.

### Medium (15)

18. **[MED] Dead `vfe_grad_Sigma_prior`** — `gaussians.py:508`. Test-only.
19. **[MED] Dead `grad_kl_Omega_i` chain** — `gauge.py:77,134`. Test-only.
20. **[MED] Dead `natural_grad_omega`** — `gauge.py:346`. Test-only oracle;
    move to a test-support module.
21. **[MED] `_precompute_obs_gradient` always computes Sigma quantities** —
    `learning.py:506`. Gate by `sigma_obs_grad != 'none'`.
22. **[MED] `Omega_star_sum` unconditional accumulation** —
    `learning.py:250`. Gate by `use_analytical_omega_grad=False`.
23. **[MED] In-place `masked_fill_` in autograd path** —
    `gaussians.py:276`. Change to out-of-place defensively (one-line fix).
24. **[MED] `slogdet(Om)` unguarded against singular Omega** —
    `gaussians.py:200`. Clamp `ln_det_Om` floor or use a safe-logdet
    pattern matching `Sig`.
25. **[MED] `safe_inverse` uses LU for SPD matrices** — `gaussians.py:39`.
    Add `spd=True` path with Cholesky.
26. **[MED] `_apply_rope` full `mu.clone()` every iter** —
    `inference.py:87`. Use `torch.empty_like` and fill both stride-2 slices.
27. **[MED] `safe_inverse` final fallback unguarded** —
    `gaussians.py:62-69`. Raise `RuntimeError` after all retries fail.
28. **[MED] `print()` for status logging** — `cuda_ext.py:59,61`. Use
    `warnings.warn` / `logging`.
29. **[MED] `to()` mutates caller's config** — `model.py:277`. Use
    `dataclasses.replace`.
30. **[MED] `make_gl_generators` vs `init_omega` device default
    inconsistency** — `gauge.py:478` vs `:554`.
31. **[MED] `set_block_diag` mixed in-place + return semantics** —
    `inference.py:125-137`. Document or split.
32. **[MED] Stringly-typed `sigma_obs_grad`, `lr_schedule`, `device`** —
    `config.py:76,117,129`. Promote to `Literal[...]`.

### Low (5)

33. `get_effective_lrs()` returns bare `dict` (model.py:130). Parameterize.
34. `_compute_holonomy` early-exit dict missing `'n_triangles'` key
    (inference.py:468).
35. `_compute_holonomy` per-triangle `.item()` syncs (inference.py:493-504).
36. `init_omega` GPU-default (gauge.py:554).
37. Dead `beta_ji` variable + misleading inline comment (gauge.py:332).

## Test suite (baseline before any fixes)

- **Command:** `pytest transformer/pure_vfe/tests tests/transformer -q -k "pure_vfe or pure_fep"`
- **Result:** **117 passed, 0 failed** (860 deselected, 3 RuntimeWarnings
  about early E-step termination — expected for single-token edge cases).

The package is in good shape test-wise; the audit findings are all
about latent quality, performance, and correctness risks rather than
currently-breaking bugs.
