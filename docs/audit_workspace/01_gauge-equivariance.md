# Audit Memo 01 — Gauge Equivariance of the `skip_attention=True` + `ift_phi` E-step

**Date:** 2026-06-01
**Auditor lens:** gauge theory / differential geometry — the equivariance memo lost in pass 1.
**Scope:** the LIVE production path only (`transformer/train_publication.py::EM_CONFIG`), not dataclass defaults.
**Environment:** torch 2.11.0+cpu, CPU forced (cuda unavailable). All numbers reproduced on CPU.

## Live config audited (from the task brief, treated as ground truth)

```
skip_attention=True, em_mode='ift_phi', gauge_mode='learned', gauge_param='phi',
diagonal_covariance=True, isotropic_covariance=False, exact_diagonal_transport=False,
evolve_sigma=True, evolve_phi=True, evolve_phi_e_step=True, use_rope=True, rope_base=100,
rope_full_gauge='off', non_flat_transport=False, n_layers=1, ffn_n_iterations=1,
embed_dim=20, irrep_spec=[('fund',2,10)] => GL(10) x 2 heads (d=10, K=20),
norm_type='layernorm', use_layernorm=True, phi_trace_clamp=0.75, phi_project_slk=False,
alpha_divergence=0.3, e_step_sigma_floor=0.01.
```

In this regime the FFN's fused kernel runs with `lambda_softmax_eff=0.0` because
`include_attention_entropy` forces the envelope identity at the softmax stationary point
(`variational_ffn.py:1979`).

## Verdict (one line)

The covariant *kernel* (per-head Omega transport, diagonal sandwich, KL/beta score, mu/sigma
gradients) is built and used consistently and is **exactly gauge-covariant under the monomial
subgroup** (signed permutations x positive diagonal) that preserves diagonal covariance —
verified on the LIVE Rényi branch `alpha_divergence=0.3`, not just the KL branch. Strict
full-`GL(10)` covariance is intentionally relinquished by `diagonal_covariance=True` (the
explicitly ALLOWED diagonal approximation). At the *block* level the live `nn.LayerNorm` and
`use_rope=True` each strip away most of that subgroup, leaving the full live block with **no
nontrivial exact gauge frame** — the expected, deliberate behavior of a RoPE+LayerNorm
transformer that trades gauge symmetry for position encoding and normalization. A
mathematically-pure equivariant path remains reachable under toggles (`norm_type='mahalnorm'` or
`'none'` + full covariance + `rope_full_gauge` or RoPE off), so the CLAUDE.md "pure path must
exist" rule is satisfied.

### Subgroups, stated separately (with numbers)
- **Bare fused kernel** (`vfe_gradients.py`, transport + KL + gradients, RoPE off, no norm):
  exactly covariant on the full **monomial subgroup** (signed perm x positive diagonal).
  Confirmed: permutation `(C)` dKL=1.9e-6, diagonal scaling `(C2)` dKL=1.9e-6, sign flip `(G)`
  dKL=**0.000**. General GL(10) `(C3)` breaks (dKL=23.5) by the diagonal approximation (by design).
- **+ RoPE** (`use_rope=True`, live): restricts to frames that preserve each `(2k,2k+1)` rope
  pair. A pair-constant diagonal frame stays invariant `(E)` dKL=1.9e-6; a coordinate permutation
  that scrambles pairs breaks `(E)` dKL=7.36. Distinct frequencies per pair leave no nontrivial
  pair-scrambling permutation invariant.
- **+ nn.LayerNorm** (`norm_type='layernorm'`, live): uses the global K=20 mean/variance.
  Commutes with an unsigned coordinate permutation `(D)` err=3.6e-7 but breaks under diagonal
  scaling `(D)` err=0.39 and even a sign flip `(G)` err=0.86 (subtracting the global mean is not
  sign-equivariant). So LayerNorm alone reduces the surviving group from monomial to unsigned
  **permutations** (it also mixes the two heads, so per-head SO(10) rotations break too).
- **Full live block** (`LayerNorm ∘ RoPE-kernel`): the surviving exact symmetry is the
  intersection of {mean/std-preserving signless frames} ∩ {RoPE-pair-preserving frames} ≈
  **trivial**. This is correct and intended; the monomial covariance is a property of the bare
  kernel, recovered in the pure-ish path.

## Reproduction

Test driver: `docs/audit_workspace/01_gauge_equiv_repro.py`. Run from the repo root:

```
python docs/audit_workspace/01_gauge_equiv_repro.py
```

Observed output, run on the LIVE Rényi branch `alpha_div=0.3` (CPU, float32 — the kernel
internally upcasts to float32):

```
K=20 d=10 n_heads=2 n_gen=200
(A) max|diag(OmSigOm^T) - kernel_einsum_diag| (head0) = 2.384e-07
(B) head-1 gens leaking into head-0 block: 0.000e+00 ; head-0 into head-1: 0.000e+00
(C) permutation-frame: max|dKL|=1.907e-06 max|dBeta|=5.960e-08 max|grad_mu covar err|=2.341e+00 max|grad_sigma covar err|=8.729e+00
(C-debug) grad_mu self-term covar err=0.000e+00  direct(align)-term covar err=3.338e-06
(C2) diagonal-scaling frame: max|dKL|=1.907e-06 max|dBeta|=5.960e-08
(C3) GENERAL GL(10) frame: max|dKL|=2.348e+01 max|dBeta|=2.668e-01
(D) LayerNorm vs permutation frame: max err = 3.576e-07
(D) LayerNorm vs diagonal-scaling frame: max err = 3.914e-01
(E) RoPE + pair-constant diagonal frame: max|dKL|=1.907e-06
(E) RoPE + coordinate-permutation frame: max|dKL|=7.361e+00
(G) kernel + sign-flip frame: max|dKL|=0.000e+00      (signs ARE in the kernel's monomial subgroup)
(G) LayerNorm + sign-flip frame: max err=8.633e-01    (signs do NOT survive LayerNorm)
(F) clamp=0.75: per-token log det(exp(phi_h)) in [-0.276, 0.301]; pairwise det(Omega) in [0.576, 1.825]
```

The earlier KL-branch run (`alpha_div=1.0`) gave the same qualitative verdict; the numbers above
are the production divergence. A second saturating run drives `phi` hard along the trace direction
and confirms the clamp caps `log det(exp(phi_h))` at exactly **0.7500** per head.

The `(C)` line's `grad_mu/grad_sigma covar err` (2.34 / 8.73) is a **test-harness artifact**, not a
code bug: the convenience wrapper `run_kernel` hardcoded the untransformed prior `(mu_p, sigma_p)`.
The `(C-debug)` decomposition, which passes the transformed prior, shows the self-coupling gradient
is covariant to machine precision (`0.000e+00`) and the alignment gradient to float32 noise
(`3.3e-6`) — including the Rényi-branch-specific `sigma_blend` and `logdet/(alpha_div-1)` paths.
The corrected diagonal-scaling test `(C2)` and the det test `(F)` are unaffected.

### Wired-model confirmation
The hand-built `E_ab` generator basis was cross-checked against the actual wired model
(`GaugeTransformerLM` constructed from the live-spec dict). `model.transformer.blocks[0]` yields
`generators.shape == (200, 20, 20)`, zero cross-head leakage (`g[:,0:10,10:20]` and
`g[:,10:20,0:10]` are identically 0), exactly 100 generators supported in each head block,
`ffn.irrep_dims == [10, 10]`, `ffn.alpha_divergence == 0.3`, `norm1` is `nn.LayerNorm`, and
`skip_attention == True`. So the equivariance conclusions verify the wired kernel, not merely a
reconstruction.

---

## Findings

### F01-A — Diagonal covariance transport IS the diagonal of the sandwich, used consistently. [INFO / PASS]
- **Path:** `transformer/core/vfe_gradients.py:1372-1374` (dense), `:1225-1227` (causal lower-triangle fast path).
- **Code:** `sigma_j_transported = einsum('bijkl,bijkl,bjl->bijk', Omega, Omega, sigma_block)` =
  `sum_l Omega[k,l]^2 * sigma_j[l]` = `diag(Omega @ diag(sigma_j) @ Omega^T)`.
- **Evidence:** test `(A)` gives `max|diag(Omega Sigma Omega^T) - kernel_einsum| = 2.4e-7` (float32 noise).
- **Consistency:** the SAME `sigma_j_transported` tensor feeds the KL/beta score
  (`:1407-1409`), the mu-alignment gradient `grad_kl = delta_mu/sigma_j_transported`
  (`:1437`), and the sigma-alignment gradient (`:1478`). The SAME `Omega_block` transports
  `mu_j` for both the KL score (`:1364`) and the gradient (`:1431`). Score and message use one
  transport. This is the CLAUDE.md sandwich rule under the explicitly-allowed diagonal approximation.
- **Severity:** INFO (correct). No action.

### F01-B — Omega is block-diagonal per head (GL(10) x 2), not full K=20. [INFO / PASS]
- **Path:** `transformer/core/variational_ffn.py:1919` (`gen_h = self.generators[:, block_start:block_end, block_start:block_end]`), kernel called per head with `irrep_dims=[d_h]` (`:2006`). Block exp pairs built per block in `gauge_utils.py::fused_block_matrix_exp_pairs`.
- **Evidence:** test `(B)` — head-1 generators have exactly `0.000e+00` magnitude inside the head-0 block and vice versa, so `Omega = exp(phi)|_h @ exp(-phi)|_h` is strictly block-diagonal. Each head transports its own 10-dim block; no 20x20 transport is ever formed in the live path.
- **Severity:** INFO (correct). No action.

### F01-C — Core kernel is exactly gauge-covariant on the monomial subgroup; full GL(10) is the documented diagonal-approx break. [LOW — by design, ensure documented]
- **Path:** whole fused kernel `vfe_gradients.py:968-...`.
- **Evidence:**
  - All numbers below are on the LIVE Rényi branch `alpha_div=0.3` (the `else` path:
    `sigma_blend=(1-α)σ_i+ασ_j_t`, mahal `α·Δμ²/σ_blend`, `logdet/(α-1)`), not the KL branch.
  - Orthogonal (per-head permutation) frame `(C)/(C-debug)`: `KL` invariant to `1.9e-6`,
    `beta` to `6.0e-8`, `grad_mu` covariant (`G @ grad_mu`) with self-term error `0.0` and
    alignment-term error `3.3e-6`, `grad_sigma` covariant under the induced permutation. The
    Rényi-specific gradient paths are therefore equivariant.
  - Positive-diagonal frame `(C2)`: `KL`/`beta` invariant to `~1.9e-6`. Sign-flip frame `(G)`:
    `dKL=0.000` (exact). Together permutations + diagonal + signs span the monomial subgroup
    (signed perm x diagonal), which is exactly the stabilizer of the diagonal-covariance manifold,
    so the diagonal divergence is an exact invariant there.
  - General `GL(10)` frame `(C3)`: `dKL = 23.5`, `dBeta = 0.27` — strict equivariance is lost.
    This is unavoidable under `diagonal_covariance=True`: `g diag(sigma) g^T` is dense for a
    non-monomial `g`, and the kernel keeps only its diagonal. This is the diagonal approximation
    that CLAUDE.md explicitly permits.
- **Math:** `mu^T Sigma^{-1} mu`-type terms are `GL(K)`-invariant only when `Sigma` transports as
  the full sandwich; the diagonal projection breaks that for non-monomial `g`. Correct and expected.
- **Severity:** LOW. The behavior is correct given the toggle; the only ask is that the diagonal
  approximation's restriction to the monomial subgroup is stated where users reason about
  equivariance (currently the docstrings frame the limitation only via RoPE). No code change required.

### F01-D — `rope_full_gauge='off'` + `norm_type='layernorm'`: the documented RoPE x MahalanobisNorm gap does NOT apply, but a LARGER LayerNorm break does. [MEDIUM — known/acceptable, must be documented as the live break]
- **Path:** `transformer/core/blocks.py:805 / :988-989` — under `skip_attention=True` and
  `norm_type='layernorm'`, the block applies `mu_normalized = self.norm1(mu_q)` where
  `self.norm1 = nn.LayerNorm(K)` (`blocks.py:389-390`, `_make_norm`). `MahalanobisNorm` is never
  instantiated, so the CLAUDE.md "RoPE x MahalanobisNorm" gap (rotated mu divided by un-rotated
  sigma) is **structurally absent** in the live config — settled.
- **But** `nn.LayerNorm` is itself not gauge-equivariant. Evidence: it commutes with an unsigned
  permutation `(D)` (`3.6e-7`) but breaks under diagonal scaling `(D)` (`3.9e-1`) and even a sign
  flip `(G)` (`8.6e-1`). The norm operates against the **global K=20 mean/variance**, so it is not
  sign-equivariant (the mean shifts under a sign flip) and it mixes the two heads. LayerNorm alone
  therefore reduces the surviving exact symmetry from the kernel's monomial subgroup to unsigned
  **coordinate permutations**.
- **RoPE compounds this**: `use_rope=True` (live) restricts further to RoPE-pair-preserving
  frames `(E)`, and a coordinate permutation that scrambles pairs breaks it (`7.36`). The full
  live block `LayerNorm ∘ RoPE-kernel` thus retains no nontrivial exact gauge frame — the
  intended behavior of a RoPE+LayerNorm transformer. The monomial covariance is a property of the
  bare kernel, recovered in the pure-ish path (F01-F).
- **RoPE interaction `(E)`:** RoPE preserves invariance under frames acting consistently within
  each `(2k,2k+1)` rope pair (pair-constant diagonal: `dKL=1.1e-5`), and breaks under frames that
  scramble rope pairs (`dKL=24`). This is the standard RoPE coordinate-pairing limitation, not a
  norm-division asymmetry. So for the live `layernorm` path the relevant equivariance break is
  LayerNorm (large, affects the diagonal subgroup) plus RoPE's intrinsic pair coupling — NOT the
  MahalanobisNorm gap.
- **Severity:** MEDIUM as a *documentation* finding. The CLAUDE.md "KNOWN GAP" entry is about
  `mahalnorm`; the codebase already implies `layernorm` is non-equivariant (the `MahalanobisNorm`
  sl(K) theorem docstring frames mahalnorm as the equivariant analog of LayerNorm), but that
  implication is buried in the norm docstring and not surfaced where the live `layernorm` config
  is reasoned about. No correctness bug — `layernorm` is a deliberate, ALLOWED engineering choice
  (the pure path uses `mahalnorm`, F01-F). Recommend making the implication explicit in the
  CLAUDE.md known-gap section: `norm_type='layernorm'` is the non-equivariant fast norm and breaks
  gauge equivariance down to unsigned coordinate permutations (and, with live RoPE on top, down to
  the trivial frame); `mahalnorm` is the equivariant analog.
- **Fix (doc only):** extend the "KNOWN GAP" note: under `norm_type='layernorm'` the per-coordinate
  normalization is not gauge-covariant; use `norm_type='mahalnorm'` for the equivariant path.

### F01-E — det(Omega) is bounded under `phi_trace_clamp=0.75`, `phi_project_slk=False`. [INFO / PASS]
- **Path:** det control applied at embedding init (`embeddings.py:472`) AND re-applied to the
  *evolved* phi every E-step iteration: `variational_ffn.py:2397-2405` calls `_retract_phi(...,
  trace_clamp=self.phi_trace_clamp, project_slk=self.phi_project_slk, irrep_dims=self.irrep_dims)`,
  which calls `_apply_det_control` after the GL(K) retraction (`vfe_utils.py:854, :883-944`).
- **Mechanism:** per head, `s_h = tr(phi . G^(h))` is soft-clamped to `[-0.75, 0.75]`; since
  `det(Omega_h) = exp(s_h^(i) - s_h^(j))`, the pairwise determinant is bounded in
  `[exp(-1.5), exp(1.5)] = [0.223, 4.482]`.
- **Evidence:** test `(F)` — random phi gives per-token `log det in [-0.283, 0.271]` and pairwise
  `det(Omega) in [0.568, 1.720]`; a saturating phi (driven along the trace direction) caps
  `log det(exp(phi_h))` at exactly `0.7500` per head, confirming the bound is tight and enforced.
- **Note:** `_retract_phi` is called without an explicit `gauge_group`, so it auto-detects. For
  `n_gen=200` (GL(10) x 2, not `K^2=400`, not SO(N)) the auto-detect resolves to `is_glk=True`
  via the `not is_son` default (`vfe_utils.py:816`), so det control activates as intended. This is
  correct for the live spec but is a fragile heuristic — see F01-G.
- **Severity:** INFO (correct). No action on the bound itself.

### F01-F — A mathematically-pure equivariant path is reachable under toggles. [INFO / PASS — CLAUDE.md rule satisfied]
- **Path:** `norm_type='mahalnorm'` instantiates `MahalanobisNorm` (`blocks.py:393-394`), whose
  full-covariance branch computes `s^2 = mu^T Sigma^{-1} mu` via `linalg.solve`
  (`blocks.py:196-224`). This rescaling is `GL(K)`-invariant:
  `(g mu)^T (g Sigma g^T)^{-1} (g mu) = mu^T Sigma^{-1} mu`. Combined with
  `diagonal_covariance=False` (full sandwich, exact `GL(K)` transport) and either `use_rope=False`
  or `rope_full_gauge in {'vfe_only','both'}` (which lifts sigma and applies `R Sigma R^T`,
  `transport_ops.py:175-227`; `variational_ffn.py:1965-1995`), plus `phi_project_slk=True`
  (det Omega = 1, true `SL(K)`), the stack is strictly gauge-equivariant.
- The live config opts OUT of this for speed (diagonal sigma, layernorm, rope-off) — exactly the
  "computationally-extreme paths are opt-in, a pure path always exists" contract. Satisfied.
- **Severity:** INFO. No action.

### F01-G — `_retract_phi` relies on n_gen auto-detection for the GL(K) branch (minor robustness). [LOW]
- **Path:** `transformer/core/variational_ffn.py:2397` does not pass `gauge_group='GLK'` to
  `_retract_phi`; the GL(K) branch (and hence det control) is selected by the `is_soN/is_glK`
  heuristic in `vfe_utils.py:803-816`. For the live `n_gen=200` it correctly resolves to GL(K),
  but a future irrep layout whose `n_gen` accidentally matches `N(N-1)/2` for some N would be
  misclassified as SO(N), silently disabling `phi_trace_clamp`/`phi_project_slk` and the GL(K)
  retraction. The omega-mode retraction path is unaffected.
- **Severity:** LOW (latent). **Fix:** thread the model's known `gauge_group` into the
  `_retract_phi` call instead of relying on `n_gen` auto-detect. Not exercised by the live config;
  flagged for defense in depth.

---

## Summary table

| id | severity | status | one-liner |
|----|----------|--------|-----------|
| F01-A | INFO | PASS | diagonal sigma transport = diag of sandwich, used consistently in KL + grad |
| F01-B | INFO | PASS | per-head block-diagonal Omega (GL(10)x2), never full K=20 |
| F01-C | LOW | by design | exact covariance on monomial subgroup; general GL(10) break is the allowed diagonal approx |
| F01-D | MEDIUM | doc | live `layernorm` avoids the mahalnorm gap but LayerNorm reduces exact symmetry to unsigned permutations (trivial once RoPE is on); make implication explicit |
| F01-E | INFO | PASS | det(Omega) bounded to [0.22, 4.48] under clamp=0.75; clamp re-applied to evolved phi |
| F01-F | INFO | PASS | pure equivariant path reachable (mahalnorm + full cov + rope_full_gauge + project_slk) |
| F01-G | LOW | latent | `_retract_phi` GL(K) branch via n_gen auto-detect; pass gauge_group explicitly for robustness |

## Recommended actions
1. (F01-D, doc) Make explicit in the CLAUDE.md "KNOWN GAP" section what the mahalnorm docstring
   only implies: under `norm_type='layernorm'` the per-coordinate norm is not gauge-covariant and
   reduces the block's exact symmetry to unsigned coordinate permutations (and to the trivial
   frame once live RoPE is included); `norm_type='mahalnorm'` is the
   equivariant analog (the sl(K) theorem). The mahalnorm RoPE gap is a separate, narrower issue
   and does not apply to the live `layernorm` config.
2. (F01-G, optional hardening) Pass `gauge_group=` explicitly into `_retract_phi` so det control
   never silently disables on a future irrep layout with a colliding `n_gen`.
3. No correctness fixes required for the live path — the covariant core is sound and the
   approximations are the documented, ALLOWED ones.
