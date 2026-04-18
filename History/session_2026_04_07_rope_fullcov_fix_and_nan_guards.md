# RoPE full-covariance chain-rule fix + NaN guards — 2026-04-07

Two related fixes to `transformer/core/vfe_gradients.py`, both touching the
same code path: `_compute_vfe_gradients_block_diagonal` (full covariance,
block-diagonal VFE gradient).

## Fix 1 — RoPE chain rule for the full-covariance gradient

### Problem

The user reported a `UserWarning` from `vfe_gradients.py:1148`:

```
UserWarning: use_rope=True is not yet implemented for the full-covariance
block-diagonal gradient path; falling back to raw chain rule (introduces a
bias in the softmax coupling term). Use the diagonal-covariance block-
diagonal path for rope-correct gradients.
```

`compute_attention_weights(use_rope=True)` softmaxes β from `KL_RoPE`
(rope-rotated μ).  The chain rule for the softmax coupling term in
`grad_mu` therefore needs to go through

    ∂β/∂μ_raw_i = -(β/κ_eff)(δ - β) · R(θ_i)^T · ∂KL_RoPE/∂(R μ_i)

The diagonal-cov helper (`_compute_vfe_gradients_block_diagonal_diag`) and
the fused multi-head helper (`_fused_attention_and_vfe_gradients_block_diag`)
already implement this.  The full-cov helper did not, so it fell back to
the raw chain rule `∂KL_raw/∂μ` and emitted the warning.  Quantitatively
this introduced a softmax-coupling bias of order **5× the gradient magnitude**
on a small finite-difference test (max relative error ≈ 5.19, i.e. 518%).

### Fix

`_compute_vfe_gradients_block_diagonal` now accepts `use_rope` and `rope_base`
parameters and mirrors the diagonal helper's pattern:

1. Compute `mu_q_rope = _apply_rope(mu_q, base=rope_base)` once per call.
2. Inside the per-block loop, transport `mu_block_rope` with the same Ω
   used for the raw transport: `mu_j_transported_rope = Ω · mu_block_rope`.
   The covariance transport `Σ_j_t = Ω Σ_j Ωᵀ` is unchanged — Σ is left
   raw, matching the standard-transformer convention and the diagonal
   helper's σ asymmetry.  See `BlockConfig.rope_full_gauge` for the
   experimental flag that rotates Σ as well.
3. Compute `delta_mu_block_rope = μ_i_rope - μ_j_t_rope` and the rope-
   space gradient `grad_kl_rope_block = Σ_j_t⁻¹ · delta_mu_block_rope`,
   accumulated into a new `grad_kl_rope_per_pair` tensor (only allocated
   when `use_rope=True`).
4. After the block loop, un-rotate per query position via
   `_un_apply_rope_pair_outer(grad_kl_rope_per_pair, base=rope_base)` and
   use the result for the softmax coupling chain rule.  The direct
   alignment term still uses the raw `∂KL_raw/∂μ` (the alignment
   *objective* is the raw KL).
5. The `grad_mu_softmax` multiplier remains the raw KL (`kl_values`),
   matching the diagonal helper.

The dispatcher `compute_vfe_gradients_gpu` now forwards `use_rope` and
`rope_base` to the full-cov helper and the warning is removed.  The
recursive `exact_diagonal_transport` call (which lifts diagonal σ to full
and recurses through the full-cov branch) also forwards both parameters,
so the exact-diagonal-transport path is now rope-correct as well.

### Verification — finite-difference check

A finite-difference test with `B=1, N=4, K=3, irrep_dims=[3]`:

| Path                                       | max abs diff vs autograd | max rel diff |
|:-------------------------------------------|-------------------------:|-------------:|
| **Old** (raw chain rule, biased)           |                 1.57e-01 |     5.19e+00 |
| **New** (rope-corrected, matching jitter)  |                 4.50e-05 |     2.04e-03 |

The new path matches the autograd reference to the float32 noise floor —
a roughly **3500× reduction** in absolute error and removal of the 518%
relative bias.  The reference objective was

    F_align = lambda_belief · Σ_ij β_ij(KL_RoPE) · KL_raw_ij

with the i-side-only convention used by the E-step (μ_j held fixed during
differentiation w.r.t. μ_i).

### Bitwise check (use_rope=False)

The non-rope code path was restructured but is mathematically identical:
`grad_kl_for_coupling` aliases `grad_kl_per_pair_full` when `use_rope=False`,
and the recomputed `avg_grad_for_coupling` is the same einsum as the
original `avg_grad`.  Smoke tests confirm the no-rope path produces
bitwise-identical sigma gradients (`max abs diff = 0`) and only the rope
path's μ softmax-coupling term differs from the no-rope baseline.

## Fix 2 — NaN guards on Omega / sigma_t / mu_t in the full-cov path

### Problem

The user reported NaN training loss at step 100 of the EM_CONFIG:

```
Step 100/60000 | Loss: nan | CE: nan | β: 0.0000 | PPL: nan
[NUM] nan_replace: 48
```

The failing config:
- `gauge_param='phi'`, `evolve_phi=True`, `evolve_phi_e_step=True`
- `diagonal_covariance=False` (full covariance)
- `use_rope=True`, `rope_base=50`
- `phi_natural_gradient='killing'`, `killing_form_sym_dampening=0.5`
- `gauge_dim=10`, `irrep_spec=[('fund', 2, 10)]`

This routes through `_compute_vfe_gradients_block_diagonal` — exactly the
helper modified in Fix 1.  Although Fix 1 does not introduce new NaN
sources (it reuses the same `Omega_chunk` and `sigma_j_inv`), it exposed a
*pre-existing* gap: the full-cov helper had **no NaN guards** at all,
while the diagonal-cov fused helper (`_fused_attention_and_vfe_gradients_block_diag`)
already had three (`fused_vfe_omega_nan`, `fused_vfe_sigma_t_nan`,
`fused_vfe_mu_t_nan`) at lines ~905-875.

When φ drifts to extreme values (which happens in `gauge_param='phi'` mode
because φ is a learnable parameter updated each step), `stable_matrix_exp_pair`'s
Frobenius-norm clamp helps for finite extreme values but does not catch
NaN φ.  An unguarded NaN in Ω propagates into `mu_j_transported`,
`sigma_j_transported`, KL, softmax, and gradients — poisoning the entire
batch and producing the `Loss: nan, β: 0.0` symptom.

### Fix

Three guards added inside the block loop, mirroring the diagonal helper:

```python
# 1. Omega NaN guard — replace bad pairs with identity
if torch.isnan(Omega_chunk).any():
    _nr("vfe_full_omega_nan")
    _nan_mask = torch.isnan(Omega_chunk).any(dim=-1).any(dim=-1)
    Omega_chunk = torch.where(
        _nan_mask.unsqueeze(-1).unsqueeze(-1),
        I_d.expand_as(Omega_chunk),
        Omega_chunk,
    )

# 2. Sigma transport NaN guard
if torch.isnan(sigma_j_transported).any():
    _nr("vfe_full_sigma_t_nan")
    _nan_mask_s = torch.isnan(sigma_j_transported).any(dim=-1).any(dim=-1)
    sigma_j_transported = torch.where(
        _nan_mask_s.unsqueeze(-1).unsqueeze(-1),
        I_d.expand_as(sigma_j_transported),
        sigma_j_transported,
    )

# 3. Mu transport NaN guard (raw and rope variants)
if torch.isnan(mu_j_transported).any():
    _nr("vfe_full_mu_t_nan")
    mu_j_transported = torch.where(
        torch.isnan(mu_j_transported),
        torch.zeros_like(mu_j_transported),
        mu_j_transported,
    )
if use_rope and torch.isnan(mu_j_transported_rope).any():
    _nr("vfe_full_mu_t_rope_nan")
    mu_j_transported_rope = torch.where(
        torch.isnan(mu_j_transported_rope),
        torch.zeros_like(mu_j_transported_rope),
        mu_j_transported_rope,
    )
```

The numerical-monitor event names (`vfe_full_*_nan`) are distinct from the
fused diagonal helper's events (`fused_vfe_*_nan`) so dashboards can
differentiate which path triggered.

### Verification

Three smoke tests:

| Case | Expectation | Result |
|:-----|:------------|:-------|
| Clean φ | No nan_replace events | `events: {}` ✓ |
| φ with NaN at one position | `vfe_full_omega_nan` fires; gradients still finite for the rest of the batch | `vfe_full_omega_nan: 2`, all gradients finite ✓ |
| φ with extreme magnitude (~100) | norm-clamp handles it without NaN | No events, all gradients finite ✓ |

End-to-end check with the user's actual EM_CONFIG settings (gauge_param=phi
+ full cov + use_rope + killing preconditioner): forward + CE backward
produces a finite loss with zero NaN gradients and zero NaN-recovery
events on the first step.

### What this fix does NOT do

- Does **not** prevent the underlying φ instability that produces the NaN
  in the first place.  If φ keeps drifting catastrophically, the guards
  will fire on more and more pairs and effectively zero out the gradient.
  Investigate `M_phi_lr`, `phi_natural_gradient`, `killing_form_sym_dampening`,
  and `mass_phi` upstream regularizers if the events fire repeatedly.
- Does **not** preserve the gradient signal from the corrupted pairs — they
  contribute zero KL and zero gradient, biasing β toward uniform on the
  affected rows.  Treat sustained `vfe_full_*_nan` events in the numerical
  monitor as a debugging signal that something is wrong upstream.

## Fix 3 — Full-covariance Σ_p eigenvalue floor (root cause)

### The asymmetry between full-cov and diagonal-cov self-coupling

Despite the NaN guards in Fix 2, the user's training still NaN'd at step
75 (full-cov, gauge_param=phi).  Subsequent debugging revealed the *actual*
root cause: a long-standing asymmetry between the diagonal-cov and
full-cov sigma-prior floor.

The E-step self-coupling gradient is

    grad_mu_self = α · Σ_p⁻¹ · (μ_q − μ_p)

In **diagonal-cov** mode, this is `α · (μ_q − μ_p) / σ_p[k]` element-wise.
The user's `e_step_sigma_floor = 0.01` clamps each `σ_p[k] ≥ 0.01`, so
each element of the self-coupling gradient is bounded by `α/0.01 = 100·α`.

In **full-cov** mode, the previous floor (`variational_ffn.py:1331-1333`)
clamped only the *diagonal* of Σ_p:

```python
diag_vals = torch.diagonal(sigma_p, dim1=-2, dim2=-1)
diag_clamped = diag_vals.clamp(min=_floor)
sigma_p = sigma_p + torch.diag_embed(diag_clamped - diag_vals)
```

But the matrix inverse `Σ_p⁻¹` has eigenvalues `1/λ_k`, not `1/diag_k`.
A full-cov Σ_p like

    [[1.0,  0.999, 0.999],
     [0.999, 1.0,  0.999],
     [0.999, 0.999, 1.0]]

has all diagonals = 1 (≥ floor) but eigenvalues `(0.001, 0.001, 2.998)`,
so `1/λ_min ≈ 1000`.  The self-coupling gradient is then amplified by
**1000×** instead of the intended **100×**.

This explains every observed symptom:

1. **Full-cov explodes but diagonal-cov doesn't.**  Same config, only
   the cov mode differs, only the full-cov path lacks the eigenvalue
   floor.
2. **Smaller M-step LR makes the explosion happen earlier.**  With small
   M-step LR the model can't fit the data, so the residual `(μ_q − μ_p)`
   stays large, the E-step self-coupling stays at maximum amplification,
   and the slow drift toward singular Σ_p is *not* compensated by faster
   prior updates.  Faster M-step LR randomly knocks Σ_p out of the
   singular regime.
3. **`grad_mu_self = 6.36e+01` at step 25 vs total `7.79e+01`.**  Self-
   coupling dominates the gradient — already approaching the 100×
   amplification cap from a `λ_min ≈ 0.016` Σ_p.  After 50 more steps it
   crosses into the unbounded regime.
4. **`sigma_q cond = 1.7` is fine.**  The σ_q condition clamp at
   `variational_ffn.py:2266-2305` enforces `cond(Σ_q) ≤ 10`, which works.
   The issue is *Σ_p* (the *prior*), not Σ_q (the *belief*).

### Fix

Replace the diagonal-only clamp with an additive eigenvalue shift:

```python
# E-step sigma_p floor
_floor = self.e_step_sigma_floor
if sigma_p.dim() == 3:
    sigma_p = sigma_p.clamp(min=_floor)
else:
    # Full covariance: enforce λ_min(Σ_p) ≥ _floor.
    # Adding _floor·I shifts every eigenvalue by _floor, so
    # λ_min(new) = λ_min(old) + _floor ≥ _floor for any PSD input.
    K_sigma = sigma_p.shape[-1]
    I_K = torch.eye(K_sigma, device=sigma_p.device, dtype=sigma_p.dtype)
    sigma_p = sigma_p + _floor * I_K
```

`O(K²)` per token, no eigendecomposition.

### Verification

Unit test with K=3, ill-conditioned Σ_p (correlation 0.999, λ_min ≈ 0.001):

| Path                                  | `\|grad_mu\|` |
|:--------------------------------------|------------:|
| Old (diagonal floor only)             |    1640.65  |
| New (eigenvalue floor via I shift)    |     150.30  |

**11× reduction** on this case, and on more pathological correlations
(0.99999, λ_min ≈ 0.00001) the reduction is 1000×+.  The well-conditioned
case (Σ_p = I) is perturbed by **0.54%**, which is negligible.

End-to-end test with the user's EM_CONFIG settings (full cov, gauge_param=phi,
use_rope, killing form): the model now survives 13+ training steps before
hitting other instabilities, vs ~step 7 without this fix.  The remaining
instability is a separate dynamic-coupling issue between the killing-form
phi natural gradient and prior_bank base parameters that this fix does
not address.

### What this fix does NOT do

- Does **not** make the EM_CONFIG with `gauge_param='phi'` fully stable.
  The Σ_p inversion was the dominant amplification source but not the only
  one — the killing-form natural gradient on `phi` and the prior_bank
  base parameter coupling also contribute.  For a fully stable run with
  full-cov + phi gauge param, additional config changes are typically
  needed: raise `e_step_sigma_floor` (0.05 or 0.1 cap the amplification
  at 20× / 10×), lower `E_alpha` (0.5 or 0.1 reduces the self-coupling
  weight), or switch `phi_natural_gradient` to `'cartan'` to avoid the
  GL(K) center-direction degeneracy in the Killing form.

## Files modified

| File | Change |
|:-----|:-------|
| `transformer/core/vfe_gradients.py` | (a) `_compute_vfe_gradients_block_diagonal` now accepts `use_rope`/`rope_base` and computes rope-space gradients un-rotated for the softmax coupling chain rule. (b) Added Omega / sigma_t / mu_t NaN guards inside the block loop. (c) Dispatcher `compute_vfe_gradients_gpu` forwards `use_rope`/`rope_base` to the full-cov helper and to the `exact_diagonal_transport` recursive call. Removed the now-stale warning at the full-cov branch. |
| `transformer/core/variational_ffn.py` | `_prepare_e_step_inputs` (line 1326) now floors *eigenvalues* of full-cov Σ_p via `Σ_p ← Σ_p + _floor·I`, replacing the previous diagonal-only clamp.  This bounds `Σ_p⁻¹` eigenvalues to `≤ 1/_floor`, matching the diagonal-cov amplification cap. |
