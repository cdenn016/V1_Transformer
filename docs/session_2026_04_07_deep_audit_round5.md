# Deep Audit Round 5 — 2026-04-07

Continuation of the deep audit. Rounds 1-3 recorded in
`session_2026_04_07_deep_audit_fixes.md`; round 4 in
`session_2026_04_07_deep_audit_round4.md`. Round 5 focused on the
remaining unread critical paths: the optimizer, full-covariance
closed-form E-step, DEQ step-function internals, Lie group operations
(SO(3) log and BCH composition), `training/metrics.py`,
`embeddings.__init__` and its EMA update helpers, and the test suite.

## Round 5 fixes

### Fix #35 — `pad_token_id = -1` default remained in three internal helpers (`embeddings.py:615,656,744`)

Round-3 Fix #28 changed the defaults on the public wrappers
(`model.p_flow_update`, `model.phi_flow_update`,
`model.delta_rule_update_w_out`) from `-1` to `-100` to match the
CE loss `ignore_index`. But the internal helpers those wrappers call
(`GaugeTokenEmbedding._compute_pflow_weights`,
`GaugeTokenEmbedding.update_embeddings_from_beliefs`,
`GaugeTokenEmbedding.update_phi_from_beliefs`) still had
`pad_token_id: int = -1`. The wrappers pass `pad_token_id=...`
explicitly, so the internal default is dead code in practice, but any
direct caller of the internal methods would get the wrong default.

**Fix**: aligned all three internal defaults to `-100` and added a
comment noting consistency with the CE loss ignore_index. Verified
via regex that no `pad_token_id = -1` default remains anywhere in
the `transformer/` package.

## Round 5 verification — no new bugs found

The following critical paths were directly audited in round 5 and
confirmed correct:

### Full-covariance closed-form E-step (`variational_ffn.py:2157-2247`)

Walked through the Cholesky-based fixed-point solver by hand. The
key identity is that for `Ω_ij = exp(φ_i) · exp(−φ_j)`, we have
`Ω_ij^{−1} = exp(φ_j) · exp(−φ_i)` and the transported precision is
`Σ_j_t^{−1} = Ω_ij^{−T} · Σ_j^{−1} · Ω_ij^{−1}`. The code computes

```
Q_j = exp(φ_j)^T · Σ_j^{−1} · exp(φ_j)           # per-j, O(N·K³)
A_align = exp(−φ_i)^T · (Σ_j β_ij Q_j) · exp(−φ_i)
        = λ · Σ_j β_ij · Σ_j_t^{−1}
```

which matches the textbook precision-weighted fixed point
`A μ* = b` with `A = α·Σ_p^{−1} + λ·Σ_j β_ij Σ_j_t^{−1}` and
`b = α·Σ_p^{−1}·μ_p + λ·Σ_j β_ij Σ_j_t^{−1}·Ω_ij μ_j`. I verified
the einsum indices and the sandwich factorisation match. The
entropy-scaled covariance `Σ* = (α + λ) · A^{−1}` at line 2234
matches the corrected docstring from round-1 Fix #3.

**Note**: the full-cov closed-form branch *does not* include the
softmax-coupling correction terms (`S_mu_h`, `c_mu_h`, `S_sigma_h`)
that the diagonal branch at lines 2108-2131 includes. The comment at
line 1894 documents this as an intentional simplification
("Diagonal path uses the enhanced form that absorbs softmax coupling
(S, c terms); full-cov uses linear-only CF"). When `lambda_softmax > 0`,
the full-cov and diagonal closed-form fixed points differ — not a
bug but an asymmetry the user should be aware of. **Test coverage
for closed-form agreement is limited to K=1 (see test suite
audit)**, so this asymmetry is regression-unprotected.

### DEQ step function closure (`variational_ffn.py:1324-1447`)

`_make_deq_step_fn` builds a closure that performs one VFE iteration
with autograd-tracked operations. The closure correctly:
- Computes β via `compute_attention_weights`
- Computes VFE gradients via `compute_vfe_gradients_gpu` (including
  softmax coupling)
- Projects to natural gradient via `compute_natural_gradient_gpu`
- Clamps raw gradient to ±1e3 and natural gradient norms to 500
- Applies retraction with trust region

The AI/distillation gradients are NOT in the closure — this was
finding #12 from round 2, already fixed by raising `ValueError` when
`active_inference=True` is combined with `use_deq=True` in
`wire_readout_references`. The DEQ backward divergence guards were
added in round 2 Fix #13.

### SO(3) log and BCH composition (`embeddings.py:804-907`)

**`so3_log_torch`**: computes `θ = arccos((tr(R) − 1)/2)` and then
`log(R) = (θ/(2·sin θ)) · (R − R^T)`. The axial vector extraction
via `v_x = (skew[2,1] − skew[1,2])/2 = R[2,1] − R[1,2]` correctly
picks the skew-symmetric components. Small-angle approximation
`θ/(2·sin θ) ≈ 1/2 + θ²/12` handles `θ → 0`. Near-π handling
extracts the rotation axis from the column of `R + I` with the
largest norm (rank-1 structure when `R = −I + 2nn^T`). Sign is
resolved via dot product with the standard vex. All correct.

**`so3_compose_bch`** at first order: `φ₁ + φ₂ + (1/2)·[φ₁, φ₂]`
with `[X, Y] = X × Y` (cross product) in so(3). Correct.

**`so3_compose_bch`** at second order: adds
`(1/12)·([φ₁,[φ₁,φ₂]] − [φ₂,[φ₁,φ₂]])`. Verified via the identity
`−[Y,[X,Y]] = [Y,[Y,X]]` (antisymmetry of the inner bracket in so(3)
via the vector triple product `Y × (X × Y) = X(Y·Y) − Y(X·Y)`),
which makes this mathematically equivalent to the standard BCH
second-order term `(1/12)·([X,[X,Y]] + [Y,[Y,X]])`. Correct.

### `RiemannianAdamW` optimizer (via subagent + spot check)

- AdamW math correct (decoupled weight decay, bias-correction via
  parent class `torch.optim.AdamW`)
- Natural gradient preconditioning applied correctly for
  mu_embed (Fisher metric Σ), sigma_embed (constant 2× rescaling),
  phi_embed (Killing metric inverse)
- Riemannian trust-region clipping per group in metric-aware norms
- AMP/GradScaler interaction correct (unscale → precondition → clip
  → step)
- Parameter group routing exhaustive with fall-through to 'ffn'
- No bugs found; minor cosmetic concern that gradient reassignment
  (`p.grad = p.grad @ K_inv`) could theoretically confuse GradScaler
  if it held the old reference, but this is safe in the current call
  order (unscale happens before preconditioning)

### Closed-form mu update Cholesky math

The full-covariance path uses `torch.linalg.cholesky(A_h)` and
`torch.cholesky_solve(b_h, L_A)` to solve `A μ = b`, and
`torch.cholesky_inverse(L_A)` to compute `A^{−1}` for the covariance
update. Both Cholesky calls are applied to `A_h = A_prior + A_align +
ε·I` (line 2226), which is SPD by construction when `α > 0` and
`Σ_p^{−1}`, `Σ_j_t^{−1}` are SPD. Regularisation by `ε·I` guarantees
positive-definiteness even when numerical rounding makes a summand
marginal. Correct.

### `_compute_gauge_transform` in PriorBank (`prior_bank.py:265-294`)

Simple and correct: computes `φ·G = Σ_a φ_a · G_a` via einsum and
calls `stable_matrix_exp_pair(phi_dot_G, only_forward=only_forward)`.
The `only_forward` flag propagates correctly.

### `training/metrics.py`

Pure CSV logger, no math. No concerns.

## Test coverage findings (via subagent)

The test suite covers ~70% of the audit invariants. **Critical
gaps** (these are places where an audit bug could survive because
no regression test catches it):

1. **Implicit EM IFT scale formula** (`_last_implicit_mu_scale`) —
   ZERO tests validate the `s_k = (α/σ²_p) / (α/σ²_p + Σ β/σ²_q)`
   formula at the E-step fixed point.
2. **DEQ Neumann-series backward correction** — ZERO finite-difference
   tests comparing the IFT-corrected `∂z*/∂θ` against the truth.
3. **Closed-form E-step vs iterative agreement** — Only a K=1 test
   exists. The full-covariance closed-form branch (which lacks the
   softmax coupling term) has no regression test for K > 1 or for
   multi-head configurations.
4. **Attention residual delta extraction** — The round-3 Fix #20 bug
   (the training forward path using the wrong residual pattern) would
   not have been caught by any existing test because no test
   distinguishes the two code paths (`model.forward` vs
   `model.forward_with_attention`).
5. **Sandwich product under perturbation** — Only a static numpy
   test exists; no PyTorch autograd test that perturbing Ω gives the
   correct derivative of `Ω·Σ·Ω^T`.

The test suite has strong coverage for: KL formula, attention
softmax normalisation, gauge covariance under `(μ, Σ)`, transport
cocycle `Ω_ij · Ω_jk = Ω_ik` for flat transport, SPD preservation,
and VFE monotonicity.

**Recommendation for the user**: add regression tests for the five
gaps above. The closed-form agreement test in particular is low-cost
and would immediately flag the softmax-coupling asymmetry between
the diagonal and full-covariance branches.

## Files touched in round 5

- `transformer/core/embeddings.py` — Fix #35 (three `pad_token_id` defaults)

## Cumulative audit total

Across five rounds the audit has found and fixed **18 distinct
actionable issues** (rounds 1-5 combined) plus withdrawn **2 false
positives** (the first-round generator-inversion concern, withdrawn
after direct verification; and the round-4 top-p scatter concern,
withdrawn after a concrete test showed the scatter works because
`sorted_indices` is a full permutation).

### Fix count by round
- Round 1: Fixes 1, 2, 3, 4 (attention residual, mask_self_attention default, closed-form docstring, CLAUDE.md map)
- Round 2: Fixes 10, 12, 13, 14, 15, 16 (withdrew #11)
- Round 3: Fixes 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 (plus drift hazard comments)
- Round 4: Fixes 30, 31, 32, 33, 34 (withdrew top-p scatter)
- Round 5: Fix 35

### Most consequential fix
Round 3 Fix #20 remains the single most important discovery of the
audit: the attention residual bug was live in the training forward
path `forward_with_attention` but the round-1 fix only reached the
inference path `forward` because the two paths are independently
implemented (drift hazard #29).

### Most structurally important finding
Finding #29 (the divergent block-forward implementations between
`blocks.GaugeTransformerBlock.forward` and
`model.forward_with_attention`) remains unresolved architecturally.
It is mitigated by DRIFT HAZARD comments added to both sites in
round 3, but the proper fix is a refactor that has both paths go
through a single implementation.

### Test coverage recommendations
The audit found five invariants with no regression test coverage
(listed above). These are the highest-priority additions for
preventing future audit-style findings.
