# Numerics Audit — Live skip_attention=True E-step

Date: 2026-06-01
Lens: Numerical stability of the live E-step under the production EM_CONFIG.

## Scope and live config reconciliation

Audited the live production path in `transformer/train_publication.py` (`EM_CONFIG`),
skip_attention=True. Important config reconciliation vs the task brief:

- **`sigma_max` is 12.0 in EM_CONFIG (line 322), NOT 5.0.** The task header said 5.0.
  CLAUDE.md mandates auditing the live value, so all sigma-bound analysis uses 12.0.
- All other listed live values confirmed in EM_CONFIG: diagonal_covariance=True,
  alpha_divergence=0.3, e_step_sigma_floor=0.01, phi_trace_clamp=0.75,
  phi_natural_gradient='killing', em_mode='ift_phi', irrep_spec=[('fund',2,10)] (GL(10)).

## Live dispatch (settled, not assumed)

`VariationalFFNDynamic._compute_multihead_vfe_gradients` (`variational_ffn.py:1896`):
`_use_fused_mh = is_diagonal and irrep_dims is not None and not exact_diagonal_transport`
→ True for the live config. `rope_full_gauge='off'` ⇒ `_use_rope_full=False`,
`non_flat_transport=False` ⇒ `_nonflat_omega=None`. So the live grad kernel is
**`_fused_attention_and_vfe_gradients_block_diag`** (`vfe_gradients.py:2000`).

Full live sigma path:
1. fused kernel → Euclidean `grad_sigma`
2. `_apply_natgrad_step` (`vfe_utils.py:1066`) → `compute_natural_gradient_gpu`
   (`vfe_gradients.py:2209`, `nat_grad_sigma = 2σ²·grad`) → 500-norm clip
   (`vfe_utils.py:1137-1141`) — **the 500 clip IS in the live path.**
3. `retract_sigma_e_step` (`vfe_utils.py:1173`) → `retract_spd_diagonal_torch`
   (sigma_max=12, whitened δσ/σ trust ±5) → condition clamp (κ≤10).
4. final clamp `[1e-4, 12]` at `blocks.py:1024`.

The live softmax is `_fused_attention_and_vfe_gradients_block_diag`'s
`F.softmax(logits)` at `vfe_gradients.py:1516`, logits = -kl/(κ√K).

## Empirical CPU check (passed)

`docs/audit_workspace/_numerics_probe.py` builds a tiny live-patterned model
(embed_dim=20, GL(10), skip_attention, ift_phi, diagonal, alpha_div=0.3, sigma_max=12,
killing, RoPE off-gauge) and runs forward + backward on CPU. With VFE_KL_DIAGNOSTICS=1:

- nominal: logits finite, 0 params with nonfinite grad.
- stress (σ params +2.0, μ embeddings ×10): logits finite, 0 nonfinite grad.
- stress (φ params ×30 to drive matrix_exp Frobenius clamp): logits finite, 0 nonfinite grad.

No NaN/Inf observed in forward or backward, nominal or perturbed. NOTE: the counters
(`matexp_norm_clamp` etc.) were not read back, so this shows the outputs stayed finite,
not that a specific guard fired. The φ×30 case (‖φ·G‖_F ≈ 15 > max_norm 10) very likely
engaged the Frobenius clamp, but that was not separately verified.

## Findings

### NUM-1 (low) sigma_max default (5.0) ≠ live config (12.0) — doc comments describe the default
`retract_spd_diagonal_torch` docstring (`vfe_utils.py:713-714,743-744`) and
`variational_ffn.py:253` compute "2σ² ≤ 2·sigma_max² = 50" against the function's own
*default* sigma_max=5.0. The formula is correct for the default. The live EM_CONFIG sets
sigma_max=12.0, so the actual worst-case Fisher amplification is 2·12² = 288 (5.8×
larger) and the "5× expansion before clamping" comment becomes 12×. The retraction
itself is correct and still bounded (sigma_max clamp + ±5 whitened trust + 500-norm
clip), and with ffn_n_iterations=1 the live E-step does a *single* retraction per
forward from a floored prior, so the 2σ² growth is not even iterated. Default-vs-live
mismatch in the comments, not a live blowup.

### NUM-2 (info) alpha_divergence=0.3 does NOT trigger the fractional-power NaN risk
The task flagged "fractional powers of possibly-negative or zero quantities → NaN".
The diagonal Rényi closed form (`kl_computation.py:453-476`, fused at
`gauge_utils.py:535-545`) uses only `log` and division — **no fractional powers**.
For α=0.3<1, `sig_blend = (1-α)·s + α·t = 0.7·s + 0.3·t` is a positive convex
combination of positive variances, so it cannot go non-positive; the `.clamp(min=eps)`
is defensive only. (The α>1 regime, which the docstrings warn about, is not the live
value.) Concern is dead for the live config.

### NUM-3 (info) NON-LIVE: `sig_blend` `sig_i` clamp gap is in the fallback KL kernel only
`gauge_utils.py:537` (`fused_block_diagonal_kl_diag`, row-tiled branch) builds
`sig_blend = (1-α)·sig_i + α·sig_t` where `sig_t` is clamped (line 519) but `sig_i`
(line 525) is only clamped inside the `log` (line 541), not in the blend itself.
**This kernel is NOT on the live path.** Its only caller is `kl_computation.py:520`,
reached via `compute_attention_weights`, which is the `else` fallback at
`vfe_gradients.py:2018-2042` that the live config (`_use_fused_mh=True`) does not take.
For contrast, the LIVE inline α-divergence blend in
`_fused_attention_and_vfe_gradients_block_diag` (`vfe_gradients.py:840-842`) DOES apply
`.clamp(min=eps)` to `sigma_blend_align`, and its inputs `sigma_block` (=`sigma_q_safe`,
clamped at line 1065) and `sigma_j_transported` (clamped at line 810) are both floored —
so the live path is self-consistent and safe. Flag kept only as a fallback-path
inconsistency for parity; a one-line `.clamp(min=eps)` on `sig_i` at gauge_utils.py:537
would match the live kernel. No live impact.

### NUM-4 (info) GL(10) matrix_exp runs in float32, not float64
`stable_matrix_exp_pair` (`gauge_utils.py:51`) upcasts to float64 only when
`d >= dim_threshold=20`. Live head dim is 10, so `matrix_exp` for GL(10) runs in
float32 with Frobenius-norm clamp at max_norm=10.0 (default, passed at line 316).
Worst case ‖M‖_F=10 with eigenvalues ±10 gives cond(exp M) ~ e²⁰ ≈ 5e8, and the
sandwich Ω Σ Ωᵀ compounds to ~e⁴⁰ — but phi_trace_clamp=0.75 + phi_scale=0.05 init
keep φ small in practice, and the empirical φ×30 stress stayed finite. Acceptable for
the live config; flagged because the float32/float64 threshold is a silent behavior the
config does not surface. (Mathematically the norm-clamp lets gradient flow through the
scale factor so φ still gets shrink signal.)

## Steelman / confirmed-safe (no finding)

- **KL safe-clamp**: `safe_kl_clamp` (`kl_computation.py:70`) with
  propagate_kl_nonfinite=False (live) clamps to [0, kl_max] then `nan_to_num(nan=kl_max,
  posinf=kl_max, neginf=0)`. NaN→kl_max is repulsive (down-weighted by softmax), correct.
- **Killing preconditioner conditioning**: `build_killing_form_preconditioner`
  (`gauge_preconditioner.py:203`) defaults center_reg=2K to give condition number ~1,
  uses center_only projection to preserve sl(K) geometry, and inverts via `_safe_spd_inv`
  (Cholesky + escalating jitter). Well-conditioned by construction. No finding.
- **softmax numerics**: live `F.softmax` (`vfe_gradients.py:1516`) has internal
  max-subtraction; logits ∈ [-kl_max/(κ√K), 0] bounded; κ=kappa_beta=1 frozen,
  `_get_kappa_h` returns positive constant (no zero-divide); beta floored + renormalized.
- **phi_trace_clamp=0.75**: `_apply_det_control` (`vfe_utils.py:940-944`) soft-clamps the
  per-block trace component with `v_norm_sq.clamp(min=eps)` guarding the division. Safe.
- **nat_grad_sigma = 2σ²·grad**: mathematically the diagonal Fisher inverse (g^kk=2σ_k²),
  correct; triple-bounded as above.
- **autocast-disabled regions**: every σ-division/log/matrix_exp hot spot wraps in
  `torch.amp.autocast('cuda', enabled=False)` and forces float32 (KL kernels, retraction,
  nat-grad). On the CPU test env autocast is a no-op anyway. Coverage is appropriate.
