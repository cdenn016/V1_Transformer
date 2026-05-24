# Deep Audit — 2026-05-24 — full-covariance & exact-diagonal-transport paths

Focused continuation of the `transformer/vfe/` deep audit, scoped to the full-covariance (`diagonal_covariance=False`) and exact-diagonal-transport (`exact_diagonal_transport`) paths. Staffed with four domain experts (math-purity auditor, numerical analyst, differential geometer, runtime-reachability engineer) dispatched in parallel, then a single independent `general-purpose` verifier who re-read every cited location and reconciled cross-expert tensions. Several decisive findings live in `transformer/core/` (the full-cov KL and live sandwich kernels), which the /vfe layer delegates to.

## Scope
`transformer/vfe/{e_step,prior_bank,non_flat,stack,head_mixer,efe,attention,config}.py` plus the delegated `transformer/core/{gauge_utils,vfe_gradients,attention}.py` kernels; `train_vfe.py` / `vfe_ablation_suite.py` configs.

## Headline
- **The central worry is rebutted.** The full-covariance belief/attention KL is the *true* full Gaussian KL (sandwich `Ω Σ Ωᵀ` + Cholesky trace + logdet via triangular solve), not a silent diagonal collapse. Two experts converged on this independently and the verifier confirmed by reading `core/gauge_utils.py:559-777` and by runtime instrumentation (16 Cholesky calls, 0 diagonal-fallback hits on a full-cov forward). The `torch.diagonal`-on-4-D-Σ at `e_step.py:499-511` that prompted the hypothesis is confined to `get_bayesian_alpha`'s per-dimension α-precision proxy, not the F-driving KL.
- **The full-cov path is dormant-by-config.** No shipped CONFIG dict or ablation sweep sets `diagonal_covariance=False` or `exact_diagonal_transport=True`, and reaching full-cov requires ≥2 coupled manual edits (also disabling RoPE or setting `rope_full_gauge`). So every full-cov defect below is an untested/latent research-path issue, not a live training hazard — with one exception (D1).
- **One live correctness cliff (D1):** `use_autograd_mu_sigma=True` + `diagonal_covariance=False` constructs cleanly then raises `NotImplementedError` mid-forward. Reachable only by manual flag-setting; cheap `__post_init__` fix.
- The covariance sandwiches that exist (`prior_bank.py:328`, `head_mixer.py:258-277`, `stack.py:149-156`) are genuine congruences (Sylvester's law ⇒ SPD preserved); `stack.py` symmetrizes-before-`eigh` correctly. `exact_diagonal_transport=True` is genuinely exact (lifts to full-cov, keeps off-diagonal sandwich mass), not a misnamed approximation.

## Expert findings (raw, condensed)

### Math-purity auditor (vfe-codebase-auditor)
- Full-cov belief KL, pairwise attention KL, self-coupling gradient, natural-gradient pullback, and `exact_full_cov_decode` all implement the exact full-cov forms (verified vs Cover&Thomas / Amari / Nakahara). Clean.
- [medium] Default `decode` (`prior_bank.py:615-616`) projects full Σ to its diagonal before the KL readout when `exact_full_cov_decode=False` — drops off-diagonal trace/Mahalanobis/logdet mass. Documented approximation with an opt-in exact escape (`exact_full_cov_decode=True`), not silent.
- [low] Per-K Bayesian-α correction uses a non-canonical `logdet_k = (logdet_p−logdet_q)/K` per-dim proxy (`core/vfe_gradients.py:161-164`); full-cov KL does not decompose per-coordinate. Reachable only under `E_learnable_alpha=True & diagonal_covariance=False`.
- [note] Stale docstring `core/vfe_gradients.py:1684-1688` claims an "inline full-cov path ignores alpha_div" — that path is unreachable with `irrep_dims` set (always true in /vfe); full-cov α-divergence is implemented in the block-diagonal kernel.

### Numerical analyst
- [critical→PARTIAL] `prior_bank.py:328` sandwich is **not symmetrized**; `:366-367` default SPD floor is an **absolute** diagonal clamp (`eps=0.01`) that cannot lift a negative eigenvalue; `:504-508` (`_decode_exact_full_cov`, gated by `exact_full_cov_decode=True`) uses `eps_small=1e-6·I` absolute floor + `cholesky_inverse` (forms an explicit inverse; Higham §14.1 says solve instead). Hazard real but threshold overstated — see verifier reconciliation.
- [checked, no hazard] `stack.py:149-159` cross-layer Σ handoff symmetrizes before `eigh`, clamps eigenvalues `[1e-4, σ_max]`, reconstructs — canonical nearest-SPD projection.
- [low] `omega_direct.py:451` `det` is contained (sign/scale only); `semantic_clustering/geometry.py:196-210` correctly uses `slogdet`+`solve`.

### Differential geometer
- The exact prior sandwich `A diag(s) Aᵀ` (`prior_bank.py:328`), `gauge_covariant_ridge` floor `ε·(A Aᵀ)` (`:359`), head-mixer `M Σ Mᵀ` batched + per-pair fallback (`head_mixer.py:258-277`), and the `stack.py` SPD projection are all genuine congruences / valid SPD projections (Sylvester's law of inertia; Higham nearest-SPD).
- [medium] Default diagonal clamp (`prior_bank.py:366-367`) is **gauge-breaking in principle** (a diagonal floor is not a congruence; does not commute with `Ω Σ Ωᵀ`). Recommends `gauge_covariant_ridge=True` as default. Practically latent.
- [note] `head_mixer` `M = I+Δ` not guaranteed invertible ⇒ PSD not strict PD. Non-flat diagonal-of-sandwich (`non_flat.py:521-523`) non-equivariant under non-orthogonal Ω (documented, full-cov-gated). The live transport sandwich is delegated to `core/` (verified clean by the auditor).

### Runtime-reachability engineer
- [high, LIVE] `use_autograd_mu_sigma=True` + `diagonal_covariance=False`: constructs without error (no `__post_init__` guard), raises `NotImplementedError` at `e_step.py:1740-1748` mid-forward. Reachable only by manual flag-setting (auto-promotion paths all require diagonal).
- [medium, informational] Full-cov + exact-diagonal dormant-by-config in both entry points and all sweeps.
- [medium] `exact_diagonal_transport=True` genuinely exact: `core/attention.py:289-291` lifts via `diag_embed` and routes through the full-cov sandwich KL (keeps off-diagonal mass), vs `non_flat.py:521-523` which drops it.
- [low] Full-cov belief KL is the true Cholesky-based full KL (`core/gauge_utils.py:660-705`), not a diagonal collapse; the `e_step.py:499-511` diagonal is the α-proxy.

## Verifier verdicts

| Claim | Verdict | Source |
|---|---|---|
| A1 `e_step.py:499-511` is the α-precision proxy, not belief KL | CONFIRMED | `e_step.py:469-513` (`get_bayesian_alpha`, returns `c0/(b0+kl_k)`) |
| A2 full-cov belief KL is the true sandwich Gaussian KL | CONFIRMED | `core/gauge_utils.py:660-663` (sandwich), `688-690` (trace via L_p solve), `698-705` (logdet, −d) |
| A3 self-coupling grads use full inverse | CONFIRMED | `core/vfe_gradients.py:146-150` |
| **Central: full-cov KL correct, no silent diagonal collapse** | CONFIRMED | A1+A2+A3 |
| B1 sandwich not symmetrized | CONFIRMED | `prior_bank.py:328` |
| B2 default SPD floor is diagonal clamp at `self.eps=0.01` | CONFIRMED | `prior_bank.py:366-367`, `eps` at `:121` |
| B3 `cholesky_inverse` + `eps_small=1e-6`, gated by `exact_full_cov_decode` | CONFIRMED | `prior_bank.py:504-508`, `eps_small` at `:122`, gate at `:610-611`, default False `:112` |
| B4 sandwich indefinite at φ-scale≈1.0 (absolute floor can't lift) | PARTIAL | verifier probe: SPD at scale 1.0 (min-eig +4.4e-6, 0/3000 fail); failures at scale ≥2.0. Direction right, threshold overstated |
| C1 diagonal clamp is gauge-breaking | CONFIRMED | `prior_bank.py:366-367` |
| C2 "inert" (geom) vs "indefinite downstream" (num) | RECONCILED | different lines: `:367` clamp (inert, never indefinite) vs `:504` Cholesky floor (hazard at high φ) |
| C3 `gauge_covariant_ridge` is proper congruence floor, default False | CONFIRMED | `prior_bank.py:354-364`, default `:113` |
| D1 autograd+fullcov constructs, raises mid-forward (no guard) | CONFIRMED | constructs OK (probe); raise at `e_step.py:1740-1748` |
| D2 cliff only reachable manually | CONFIRMED | `config.py:634,739` force autograd but `:610-616,712-718` require diagonal |
| E1 full-cov/exact-diag dormant in configs + sweeps | CONFIRMED | `train_vfe.py:79,81`, `vfe_ablation_suite.py:137,139`, SWEEPS 220-470 |
| E2 full-cov needs ≥2 coupled edits (rope guard) | CONFIRMED | `config.py:596-601`, both ship `use_rope=True` |
| F `exact_diagonal_transport` genuinely exact vs non-flat diag-drop | CONFIRMED | `core/attention.py:289-291` vs `non_flat.py:521-523` |

## Confirmed punch list (ranked)

**Live (actionable now)**
1. **[high] Unguarded `use_autograd_mu_sigma=True` + `diagonal_covariance=False` config cliff.** Constructs cleanly, raises `NotImplementedError` mid-forward (`e_step.py:1740-1748`). Fix: add a `__post_init__` guard rejecting the pair so the failure surfaces at config time, not after a training step. Cheap, eliminates a real footgun. (Auto-promotion paths already require diagonal, so this is the only way to hit it.)

**Dormant / latent (only bite if full-cov is manually enabled — fix opportunistically or document)**
2. **[medium] Full-cov decode numerical conditioning** (`_decode_exact_full_cov`, gated by `exact_full_cov_decode=True`): (a) symmetrize the sandwich at `prior_bank.py:328` (`0.5(S+Sᵀ)`) before any Cholesky/eigh consumer; (b) replace the absolute `eps_small=1e-6·I` floor (`:504`) with a spectrum-relative floor (e.g. `eps_small·λ_max·I` or eigen-clamp like `stack.py`); (c) replace `cholesky_inverse`+matmul (`:508`) with `cholesky_solve` (Higham §14.1 — never form an explicit inverse). Hazard is real only at ‖φ‖ ≳ 2 (verifier probe) and `phi_trace_clamp=None` by default, so it is latent.
3. **[medium] Default diagonal SPD clamp is gauge-breaking** (`prior_bank.py:366-367`). It does not commute with `Ω Σ Ωᵀ`. Make `gauge_covariant_ridge=True` (the congruence floor `ε·A Aᵀ` at `:359`) the default for non-diagonal priors, or document the clamp as a gauge-breaking convenience. Practically inert today (rarely engages, never makes a block indefinite).

**Cleanup / documentation**
4. **[low] Stale docstring** `core/vfe_gradients.py:1684-1688` describes an unreachable "inline full-cov path"; full-cov α-divergence is in the block-diagonal kernel. Delete/correct.
5. **[low] Default full-cov decode** projects Σ to diagonal unless `exact_full_cov_decode=True` (`prior_bank.py:615-616`) — already documented with an opt-in escape; note it where the full-cov experiment config is set.
6. **[low] Per-K Bayesian-α `logdet/K` proxy** (`core/vfe_gradients.py:161-164`) is non-canonical; label it as a heuristic or restrict the learnable-α product-rule correction to the diagonal path. Reachable only under `E_learnable_alpha=True & diagonal_covariance=False`.

## Verified clean (do not "fix")
- Full-cov belief/attention KL = true Gaussian KL (`core/gauge_utils.py:660-705`); the diagonal at `e_step.py:499-511` is the α-proxy.
- Sandwiches at `prior_bank.py:328`, `head_mixer.py:258-277`, the `gauge_covariant_ridge` floor, and the `stack.py:149-156` SPD projection are genuine congruences / valid SPD projections.
- `exact_diagonal_transport=True` keeps off-diagonal sandwich mass through the KL — genuinely exact, flag name accurate.

## Test suite
No code was modified by this audit. The vfe suite was green at the last run (`pytest tests/transformer/test_vfe_*.py tests/transformer/vfe/` → 221 passed). Note: none of the full-cov findings are covered by existing tests (the path is dormant-by-config) — any fix to items 1-3 should ship with a regression test that constructs a full-cov config.
