# Audit memo: E-step gradient math vs canonical free energy F

Lens: does the implemented E-step (μ, σ, φ natural-gradient updates and the β
softmax) match the canonical F in CLAUDE.md, under the LIVE production config?

Live config audited (from `transformer/train_publication.py::EM_CONFIG`, lines 118-329):
`skip_attention=True, em_mode='ift_phi', gauge_param='phi', gauge_mode='learned',
diagonal_covariance=True, alpha_divergence=0.3, E_alpha=1, E_lambda_belief=10,
E_lambda_softmax=0, include_attention_entropy=True, E_learnable_alpha=True,
E_learnable_lr=True, ffn_n_iterations=1, kappa_beta=1, learnable_head_kappa=False,
embed_dim=20, irrep_spec=[('fund',2,10)] (2 heads, d_h=10), use_rope=True,
rope_base=100, rope_full_gauge='off', evolve_phi_e_step=True,
phi_natural_gradient='killing', e_step_sigma_floor=0.01`.

Live μ/σ/β kernel = `_fused_attention_and_vfe_gradients_block_diag`
(`transformer/core/vfe_gradients.py:968`), invoked PER HEAD with `irrep_dims=[d_h]`
at `transformer/core/variational_ffn.py:2000` inside `_compute_multihead_vfe_gradients`.
Path selection: `_use_fused_mh` true (diagonal, no non-flat, no rope_full_gauge),
`_nonflat_omega is None` (non_flat_transport=False), so the fused per-head path runs.

## Summary of conclusions

The implemented μ/σ E-step computes the QUERY-SIDE-ONLY local belief gradient
`α∇D_self + Σ_j β_ij ∂E_ij/∂μ_i` and OMITS the key-side (smoothing) column term
`Σ_{m>i} β_mi ∂E_mi/∂μ_i`. The omitted term is ~equal magnitude to the kept term
(FD: query 1.09, key 1.13). This is NOT the gradient of the canonical/reduced free
energy `F_red = Σ_ij β_ij E_ij + ...` as written (which contains both terms) — it
is off by roughly half.

The manuscript (`Manuscripts-Theory/GL(K)_attention.tex`, §filtering_free_energy,
lines 883-897) RESOLVES this explicitly and in favour of the code: it derives BOTH
the query-side partial (eq:queryside_partial, line 885) and the key-side partial
(eq:keyside_partial, line 892), states that "the default reference implementation
applies only the query-side (filtering) update; the column term is omitted," cites
this as "the standard coordinate-ascent belief update of mean-field variational
inference [Beal2003, bishop2006]," and argues the filtering (no-future-coupling)
scheme is the correct default for the autoregressive next-token task. So the code
matches the manuscript's DEFINED belief dynamics (eq:belief_dynamics, line 994).

The genuine deviation is a CONSTRAINT/DOC-DRIFT issue, not a math error: CLAUDE.md
mandates "there must ALWAYS exist a theoretically/mathematically pure path under
appropriate toggles," and the manuscript itself names that pure path —
`use_autograd_mu_sigma`, which "instead descends the full gradient ∇F_red" adding
the key-side term (lines 895, 897). That toggle DOES NOT EXIST in the codebase
(`grep -rn use_autograd_mu_sigma transformer/` → zero hits; no autograd μ/σ path in
`variational_ffn.py`). So the mathematically-pure (global ∇F_red / smoothing) path
is unreachable in code — see finding E-2.

Apart from that: the β softmax, the α-divergence substitution (verified exact vs an
independently-written D_α via autograd, rel err 0.0), the entropy-term handling (the
query-side envelope identity holds to machine precision), and the Fisher
natural-gradient retractions for μ and σ are all implemented correctly and
consistently. With `ffn_n_iterations=1` exactly one local filtering sweep is taken.

## (1) β softmax and the √K factor — VERIFIED, uses per-head d_h=10 not K=20

`logits = -kl_values / (kappa * dim_scale)`, `dim_scale = math.sqrt(max(K,1))`
(`vfe_gradients.py:1499-1500`). Inside the kernel `B,N,K = mu_q.shape` and the
caller passes `mu_q=mu_h` (per-head slice) with `irrep_dims=[d_h]`
(`variational_ffn.py:2000-2006`), so **K here is d_h = 10, NOT embed_dim = 20**.
`kappa` comes from `_get_kappa_h(h,d_h)` which, with `learnable_head_kappa=False`,
returns bare `self.kappa = 1` (`variational_ffn.py:709-710`; `self.kappa` traces to
`ffn_kappa = kappa_beta = 1`, `block_config.py:622`). Hence the effective
temperature is `τ = κ·√d_h = 1·√10` per head. This is the scaled-dot-product
convention (scale by √d_head, not √d_model) and the docstring at
`variational_ffn.py:1921-1924` states it as the design intent. CLAUDE.md writes
`τ = κ·√K`; the implemented K is the per-head block dim. Defensible and internally
consistent, but worth recording that "K" in the manuscript τ is realised as d_h.

## (2) α-divergence consistency (alpha_divergence=0.3) — VERIFIED consistent

`alpha_div != 1.0` selects the Rényi α-divergence branch with blended variance
`σ_blend = (1-α_d)σ_i + α_d σ_j_t`. The SAME `alpha_div` drives:
- the β-side divergence used for the softmax (`vfe_gradients.py:1412-1424`),
- the direct μ alignment gradient `α_d·Δμ/σ_blend` (`1436-1444`),
- the σ alignment gradient (`1481-1491`),
- the self-coupling term and its learnable-α product-rule correction (`1103-1128`),
- the raw-KL multiplier accumulator (`1449-1465`).
There is NO place where β uses α-divergence while the gradient uses KL or vice
versa. The same consistency holds in the non-fused
`_compute_vfe_gradients_block_diagonal_diag` (`587-965`) and in `get_bayesian_alpha`
(`variational_ffn.py:847-911`, the learnable-α divergence selector). FD check below
shows the α=0.3 and α=1.0 paths behave identically in structure (same envelope
property). No inconsistency found.

## (3) E_lambda_softmax=0 + include_attention_entropy=True — VERIFIED correct

Two independent mechanisms, both correct:

μ/σ path: `_lambda_softmax_eff = 0.0 if self.include_attention_entropy else
self.lambda_softmax` (`variational_ffn.py:1979`). So the `dβ/dθ·KL` softmax-coupling
branch (`vfe_gradients.py:1528-1547`, gated by `lambda_softmax`) is multiplied by 0
and contributes nothing. The canonical entropy term `τ·β·log(β/π)` is NOT added
explicitly to the μ/σ gradient; instead it is accounted for by the ENVELOPE
IDENTITY: at the softmax stationary point β*=softmax(-KL/τ), the gradient of
`Σβ·KL + τΣβ·log(β/π)` through β vanishes, leaving only the direct term
`Σ_j β_ij ∂KL_ij/∂μ_i`. Empirically verified to MACHINE PRECISION (rel err = 0.0,
float64) that autograd of the entropy-augmented F with detached neighbours equals
the kernel's direct term. So forcing lambda_softmax→0 here is exactly right, and
`E_lambda_softmax=0` in config is doubly redundant (it would be overridden anyway).

φ path: the entropy term IS added explicitly and autograd-differentiated
(`variational_ffn.py:1267-1272`: `_F_h = (β·KL).sum() + κ_h·√d_h·(β·logβ).sum()`),
then `lambda_softmax` is ignored. Note it uses `log(β)` not `log(β/π)`; since
π=1/N uniform, the `−log N` shift is constant per row and `Σβ=1`, so its gradient
is zero — dropping `/π` is harmless. Consistent.

## (4) μ, σ, φ natural-gradient updates vs manuscript — VERIFIED

μ: Fisher natural gradient `∇̃μ = Σ·∇μ` (diagonal: `σ·∇μ`,
`vfe_gradients.py:2208`), then descent `μ_new = μ − lr·scale·∇̃μ` with a WHITENED
trust region `‖δμ/√σ‖ ≤ 2.0` (`vfe_utils.py:1144-1155`). Correct Fisher-preconditioned step.

σ: Fisher `∇̃σ = 2σ²·∇σ` (`vfe_gradients.py:2209`), SPD exponential-map retraction
`σ_new = σ·exp(step·clamp(δσ/σ, ±trust))` with `delta_sigma = −∇̃σ`
(`vfe_utils.py:740`, called from `retract_sigma_e_step:1204-1211`). The σ step size
is `sigma_lr·decay` (decoupled from μ LR, `variational_ffn.py:1784-1790`), matching
the CLAUDE.md "decoupled σ_lr / raw_sigma_lr" spec. Condition-number clamp
(ratio ≤ 10) and `[eps, sigma_max]` clamp applied after. Correct SPD retraction.

φ: autograd grad of the entropy-augmented alignment loss → Killing-form
preconditioner `_precondition_phi_grad` (phi_natural_gradient='killing') →
`_retract_phi` GL(K) retraction with `trace_clamp=0.75`
(`variational_ffn.py:2387-2404`). Consistent with the manuscript φ M-step / Killing
natural gradient.

Fisher metric for diagonal Gaussian: g_μ=Σ⁻¹ ⇒ g⁻¹=Σ; g_σσ=1/(2σ²) ⇒ g⁻¹=2σ².
Both match the kernel (`vfe_gradients.py:2176-2181`). Verified.

## (5) n_iterations=1 — single local E-step sweep; key-side term dropped

With `ffn_n_iterations=1`, `decay_factor=1.0` (`variational_ffn.py:2141-2146`) and
exactly ONE belief update is performed: β computed from the prior-initialised
beliefs, one μ/σ natural-gradient step, one φ retraction. This is a single
coordinate (Jacobi) sweep of the filtering E-step. It does NOT reach an E-step
fixed point; it is one amortised step. The dropped KEY-SIDE cross-term (μ_i entering
other rows' E_mi as the transported key, `eq:keyside_partial` line 892) is ~equal in
magnitude to the kept query-side term (FD: query 1.09 vs key 1.13), so the
implemented update is ~half of the global ∇_μ F_red. Per the manuscript
(§filtering_free_energy) this is the INTENDED filtering default for the
autoregressive task; the global ∇F_red ("smoothing") path is the manuscript's
opt-in alternative `use_autograd_mu_sigma` — which is unimplemented (see E-2). So
the single iteration is, by design, a local-belief filtering update rather than a
step on the globally-coupled F, and the documented pure path to the global F is not
reachable in code.

## Empirical checks (CPU, this audit)

1. Detached-neighbour autograd of `Σβ·KL + τΣβ·log β` == kernel direct term:
   rel err 0.0 (float64) for both α_div=1.0 and α_div=0.3. (envelope identity exact)
2. Full FD of the global entropy-augmented F vs kernel μ-grad: rel err ~0.42-0.50,
   fully explained by the omitted key-side cross-term (decomposition: direct 1.09,
   key-side 1.13, full FD 2.14).
3. √K factor: confirmed K=d_h=10 by tracing mu_q shape through the per-head call.

## Dead-path consequences (confirmed against task statement)

- `gauge_param='phi'` ⇒ `_use_omega=False` (`variational_ffn.py:2361`); the direct-Ω
  retraction branch (`2371-2383`) is dead; phi branch runs.
- `learnable_head_kappa=False` ⇒ `_get_kappa_h` short-circuits to `self.kappa`
  (`variational_ffn.py:709-710`); the κ-sharing / per-head log_kappa block is dead.
- `active_inference=False` ⇒ `compute_ai_gradients` returns (None,None); EFE folding
  at `variational_ffn.py:2243-2246` is a no-op. `configure_ffn_active_inference`
  (`active_inference.py:431`) effectively inert.
- `non_flat_transport=False` ⇒ `_nonflat_omega=None`; the connection-δ branch dead.
- `rope_full_gauge='off'` ⇒ `_use_rope_full=False` (`variational_ffn.py:1965-1970`);
  the experimental σ-rotating autograd path dead; the analytic fused path runs with
  the RoPE-on-μ-only convention (documented KNOWN GAP in CLAUDE.md).

## Findings (severity)

- E-2 (constraint-violation, high): the manuscript-promised mathematically-pure
  path `use_autograd_mu_sigma` (descends the full ∇F_red incl. key-side smoothing
  term, GL(K)_attention.tex lines 895/897) is NOT implemented anywhere
  (`grep -rn use_autograd_mu_sigma transformer/` = 0 hits; no autograd μ/σ path in
  variational_ffn.py). The live μ/σ kernel computes ONLY the query-side filtering
  gradient (`vfe_gradients.py:1523-1524`, key-side column term never formed). The
  filtering default is correct and manuscript-sanctioned, but CLAUDE.md requires a
  reachable pure path; here it is doc-only. Fix: implement the toggle (autograd
  through the full coupling sum without detaching the transported key μ_j), or amend
  the manuscript to drop the claim. n_iterations=1 also means no fixed-point
  equilibration of even the filtering objective.
- E-1 (observation, low): √K realised as per-head d_h=10, not embed_dim K=20
  (`vfe_gradients.py:1499`, mu_q sliced per head at `variational_ffn.py:2000`).
  Defensible scaled-dot-product convention; manuscript τ=κ√K should say K=d_h.
- E-3 (correctness, info): query-side envelope handling of the entropy term for μ/σ
  is exact (verified rel err 0.0); `E_lambda_softmax=0` is redundant given the
  `include_attention_entropy` override at `variational_ffn.py:1979`. No action.
- E-4 (correctness, info): α-divergence (0.3) applied consistently across β and the
  μ/σ/self gradients; self-coupling formulas verified exact vs independent autograd
  of the code's D_α (rel err 0.0). The D_α convention (logdet `/(α-1)`, blended
  variance) is the manuscript's, not textbook Rényi `1/(α(α-1))` — consistent. No
  inconsistency.
