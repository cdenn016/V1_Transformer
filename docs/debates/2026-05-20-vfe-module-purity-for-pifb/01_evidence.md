# Evidence Pack — vfe-module-purity-for-pifb

## Active config (`transformer/vfe/train_vfe.py`)

Edit-as-of `2026-05-20`. Values resolve through `VFEConfig.__post_init__` (`transformer/vfe/config.py:457`). Relevant resolved values for this debate:

- `vocab_size`: populated from dataloader (wikitext-103, ~50K)
- `embed_dim = 20`, `irrep_spec = [('fund', 2, 10)]` → K=20, two heads of d_head=10
- `n_layers = 3`, `n_e_steps = 1`
- `gauge_group = 'GLK'`, `gauge_parameterization = 'phi'` (default)
- `phi_preconditioner = 'killing'`, `phi_project_slk = False`, `phi_trace_clamp = 0.75`
- `diagonal_covariance = True`, `isotropic_covariance = False`, `exact_diagonal_transport = False`
- `use_rope = True`, `rope_full_gauge = 'off'`, `rope_base = 150`, `bch_order = 3`
- `gauge_fixed_priors = False` (direct per-token priors; *not* the gauge-orbit form)
- `use_prior_bank = False` (decode is `nn.Linear`; *not* Law-3 KL-to-prior)
- `mask_self_attention = False`, `causal_lower_triangle = True`
- `use_equivariant_head_mixer = True`, `gauge_covariant_ridge = True`
- `E_learnable_alpha = True`, `learnable_kappa = False`, `kappa = 1.0`, `alpha = 1.0`
- `include_attention_entropy = True`, `alpha_divergence = 1.0` (standard KL)
- `lambda_align = 2.45`, `lambda_soft = 0.0`, `mass_phi = 0.0`
- `prior_handoff_rho = 1.0`, `prior_handoff_sigma = 0.1` (partial σ blend, not full handoff)
- `use_non_flat_transport = False`, `use_autograd_mu_sigma = False`
- `cross_couplings = []`
- `norm_type = 'layernorm'` (gauge-blind ablation; user excluded from this debate)
- `e_mu_lr = 0.5`, `e_sigma_lr = 0.015`, `e_phi_lr = 0.00`

`__post_init__` orphan warning is NOT triggered because `n_layers=3 > 1` lifts both σ and φ via cross-layer cascade.

Per the user's framing, this debate is about whether a *pure* path exists under appropriate toggles — not whether the active config is itself pure. So `gauge_fixed_priors=False`, `use_prior_bank=False`, and `norm_type='layernorm'` here do not by themselves settle the debate.

## Code references — `transformer/vfe/`

Module layout (sans temp/cache/run dirs):

```
__init__.py
_numerics.py        — NaN sentinel + trust-region clamps
attention.py        — compute_kl_attention, compute_gauge_transport (200 LOC)
block.py            — VFEBlock: E-step → head_mixer? → norm (192 LOC)
config.py           — VFEConfig + __post_init__ (886 LOC)
cross_coupling_metrics.py / cross_coupling_viz.py — diagnostics only
e_step.py           — VFEEStep main inner loop (2,321 LOC)
efe.py              — Expected Free Energy generation policy (252 LOC)
head_mixer.py       — Schur-commutant per-irrep-type mixer
model.py            — VFEModel: encode → pos → stack → norm → decode (492 LOC)
non_flat.py         — Edge-relaxed cocycle transport Ω_ij = exp(φ_i)·exp(δ_ij)·exp(-φ_j)
omega_direct.py     — Group-level Ω state (instead of φ); GL+(K) retraction
positional.py       — BCH-based positional gauge composition
prior_bank.py       — Per-token encode (gauge orbit) + decode (KL-to-prior) (646 LOC)
stack.py            — L-layer stack + cross-layer handoff (153 LOC)
trainer.py / train_vfe.py — outer M-step (CE + aux + mass_φ)
vfe_ablation_suite.py — sweep harness
```

### Free-energy functional realization (E-step)

- `transformer/vfe/e_step.py:13-67` — module docstring listing the FIVE manuscript terms and stating that the E-step inner loop assembles only THREE: `α·KL(q‖p)`, `Σ_ij β_ij·KL(q_i‖Ω_ij q_j)`, `τ·β·log(β/π)` (when `include_attention_entropy=True`). The `λ_h·KL(s‖h)` hyper-prior term and the `γ·KL(s‖s)+entropy` model-coupling term are **explicitly stated as NOT implemented** ("there is no gamma-attention parameter anywhere in the package, and `s` does not exist as a separate distribution from `q`"). The observation term `-E_q[log p(o|x)]` is realized at the model level via cross-entropy, not inside the E-step.
- `transformer/vfe/model.py:11-32` — outer-M-step docstring confirming the outer loop minimises `CE + 0.5·mass_φ·‖φ‖² + Σ aux_hyperparam_loss`, NOT the manuscript F. Aux terms route gradients to `raw_c0`, `raw_b0`, `log_κ` only.
- `transformer/vfe/e_step.py:1574-1593` (`_update_phi`, entropy-on branch) — direct φ-loss assembly: `_F = (β·KL).sum() + κ·√K·(β_safe·log β_safe).sum()`; `λ_align` scales `_F`. Uniform prior `π=1/N`; the additive `log N` constant is dropped.
- `transformer/vfe/e_step.py:1043-1073` — fused single-softmax dispatch passing `lambda_softmax=0.0` when `include_attention_entropy=True` (envelope identity at β fixed point).
- `transformer/vfe/e_step.py:954-1042` — per-head softmax dispatch with one fused-kernel call per head at temperature `τ_h = κ·√d_head` (matches GL(K)_attention.tex eq:per_head_temperature).
- `transformer/vfe/e_step.py:566-722` — `_auxiliary_hyperparam_loss`: detached-belief scalar F with hyperparameters live; the only autograd path from `raw_c0`/`raw_b0`/`log_κ` to CE.
- `transformer/vfe/e_step.py:1346-1370` — per-iter retraction order: μ-update (with optional `e_mu_q_trust` sigma-whitened clamp) → σ-retract (SPD diagonal or full path) → φ-retract (Killing-preconditioned, optional sl(K) projection or trace-clamp).
- `transformer/vfe/e_step.py:1159-1162` — `compute_natural_gradient_gpu`: produces `nat_grad_mu = Σ·∇_μ F` (Fisher-Rao on the mean block, `g_μμ = Σ^{-1}`) and the corresponding σ natural gradient.

### Attention

- `transformer/vfe/attention.py:124-198` — `compute_kl_attention`: `β_ij = softmax(-KL(q_i‖Ω_ij q_j)/τ)` via the unified `compute_attention_weights` kernel; calls return `(β, kl_matrix)`.
- `transformer/vfe/attention.py:11-30` — docstring noting the structural self-attractor at `i=j`: `Ω_ii = exp(φ_i)·exp(-φ_i) = I` ⇒ `KL(q_i‖Ω_ii q_i)=0` saturates the diagonal under softmax unless `mask_self_attention=True`.
- `transformer/vfe/attention.py:82-121` — `compute_gauge_transport`: block-diagonal matrix-exp pairs `(exp(φ·G_h), exp(-φ·G_h))`; cached skew-symmetry detection for SO(N) fast path.

### Encode / Decode (PriorBank, Law 3)

- `transformer/vfe/prior_bank.py:291-372` — `_apply_gauge_transform`: `μ_v = A_v μ_0`, `Σ_v = A_v diag(σ_0) A_v^T` (full-cov path is the exact sandwich product per block; diagonal-cov path is `diag(A diag(s) A^T)` approximation per block).
- `transformer/vfe/prior_bank.py:408-433` — `encode`: gauge-fixed branch (`A_v ▷ π_0`) vs direct branch (per-token (μ_v, σ_v) lookup with φ retained for transport only).
- `transformer/vfe/prior_bank.py` `decode` (called from `model.py:188-190`): `logits = -KL(q‖π_v) / τ` with `τ = decode_tau · exp(-decode_log_scale)`, `decode_log_scale ∈ [-3, 3]` learnable.
- `transformer/vfe/prior_bank.py` `_decode_exact_full_cov`: optional exact per-block KL between block-diagonal Gaussians for Law-3 pure decode (gate: `exact_full_cov_decode=True` requires `gauge_fixed_priors=True`, `diagonal_covariance=False`, `use_prior_bank=True` — validated in `config.py:484-507`).

### Gauge frame / connection / transport

- `transformer/vfe/positional.py:23-100` — `VFEPositionalEncoding`: `φ_i^(0) = BCH(φ_token, p_i)` with `bch_order` configurable; `phi_project_slk=True` route enforces `sl(K)` traceless subspace.
- `transformer/vfe/non_flat.py` — edge-relaxed cocycle: `Ω_ij = exp(φ_i·G)·exp(δ_ij·G)·exp(-φ_j·G)`, with `δ_ij` an antisymmetric bilinear of `(μ_i, μ_j)`; gates `use_non_flat_transport`. Compatible with the PIFB §"Discrete Regime II via an Edge-Relaxed Cocycle" construction (PIFB:828-870).
- `transformer/vfe/omega_direct.py` — group-level Ω parameterization (right-invariant retraction `Ω ← Ω·exp(-η·X_proj)`, matches `eq:gauge_group_retraction` PIFB:2566-2570); φ chart-coordinate path is the first-order truncation (PIFB:2572-2576 explicitly labels chart-coordinate form as the truncation, group-level as canonical).

### Multi-block / multi-head / cross-coupling

- `transformer/vfe/model.py:377-492` (`_build_generators`) — gauge group dispatch (SO3 / SON / GLK / GLK multi-head / GLK cross-head-coupled). `irrep_dims = [d_head]*H` for the multi-head GL(d_head)^H reduction (PIFB:1785, GL(K)_attention.tex line 1744).
- `transformer/vfe/config.py:455-887` — `super_block_dims` / `super_block_head_groups` derived properties for cross-coupling; `effective_block_dims` is the partition that every block-iteration site walks.

### Cross-layer prior handoff (timescale separation)

- `transformer/vfe/stack.py:1-153` — VFEStack with `prior_handoff_rho` (μ damping) and `prior_handoff_sigma` (σ blend). Module docstring lines 18-34 acknowledges the default (`rho=1.0, sigma=0.0`) is a **mean-only point-estimate handoff**, not the full distributional handoff of canonical hierarchical VI (Friston 2017; Parr/Pezzulo/Friston 2022; Blei/Kucukelbir/Jordan 2017), and states "Set `prior_handoff_sigma=1.0` (and a matching mechanism for φ if desired) to recover the canonical scheme." The φ handoff at line 145 is hard-wired to the embedding (no toggle).

### Renyi α-divergence

- `transformer/vfe/config.py:711-731` — `alpha_divergence != 1.0` triggers a warning when combined with autograd/non-flat/omega-direct paths (those paths reconstruct the standard KL and silently ignore the Rényi exponent). The analytic kernel honors `alpha_divergence` per PIFB:1632-1643 `eq:renyi_attention_itfb`.

### Outer loss / observation operator

- `transformer/vfe/model.py:227-306` — `forward(token_ids, targets)`: `F.cross_entropy(logits, targets)` plus optional `mass_φ` plus `Σ aux_hyperparam_loss`. The cross-entropy term is the observation operator in the PIFB §"Recasting External Observations" §1394+ "cross-entropy resolution" sense (manuscript line 1443: `F_agent^xent := -E_q[log q_e]` is "structurally identical to term 5 of the canonical free energy at Eq.~\eqref{eq:free_energy_functional_final}"). The E-step does not see `targets` (Law 1).

## Manuscript references — `Attention/Participatory_it_from_bit.tex`

In scope (per user's "language modeling aspects only" constraint):

- `tex:1247-1392` — §"The Complete Free Energy Functional". Eq:free_energy_functional_final (boxed) and Eq:beta_optimal. Five terms enumerated above. PIFB:1282 explicitly distinguishes the canonical F (with entropy) from the entropy-suppressed surrogate `Σ β·KL` and states their gradients differ by `-τ⁻¹·Cov_β(KL, ∇KL)`.
- `tex:1285-1333` — §"State-Dependent Prior Precision". Eq:state_dependent_alpha: `α_i*(c) = c₀/(b₀ + KL(q_i‖p_i))`. PIFB:1322 derives the per-θ autograd correction `(α*)² b₀/c₀ · ∂KL/∂θ`. Scalar per-agent — NOT per-dim.
- `tex:1334-1392` — §"Envelope Theorem and Reduced Free Energy". Eq:free_energy_reduced: `F_red = Σ_i KL(q_i‖p_i) - τ Σ_i log Z_i - E_q[log p(o|·)]`. Envelope identity Eq:envelope_gradient: receiver + sender sums needed for any gradient descent of x_i.
- `tex:1394-1467` — §"Recasting External Observations". Mean-gradient equivalence between observation-likelihood and environmental-agent KL coupling; full variational equivalence requires the cross-entropy substitution `-E_q[log q_e]` (PIFB:1442-1446 eq:xent_resolution).
- `tex:1502-1569` — §"Dynamical Structure and Emergent Timescales". Timescale separation `η_q : η_s : η_φ ∼ 1 : ε : ε²`. Per-component Fisher: `g_μμ = Σ⁻¹`, `g_ΣΣ[V,W] = ½ tr(Σ⁻¹ V Σ⁻¹ W)`.
- `tex:1571-1842` — §"Transformer Architectures as the Zero-Dimensional Limit". Untied query-key carving (Eq:gauge_qk: `Q_i = U_i⁻¹ μ_i`, `K_j = U_j^T Σ_j⁻¹ μ_j`); trivial-frame route to `softmax(QK^T/(κ√d_k))V`. Multi-head as `GL(d_head)^H ⊂ GL(K)` (PIFB:1785). Value aggregation `μ̂_i = Σ_j β_ij Ω_ij μ_j` (Eq:glk_mixture_aggregation_itfb at PIFB:1816).
- `tex:1632-1643` — Renyi α-divergence generalization, Eq:renyi_attention_itfb.
- `tex:2538-2580` — §"Natural Gradient Dynamics". Eq:gauge_natural_gradient: `dμ/dt = -η_μ Σ ∇_μ F`, `dΣ/dt = -2 η_Σ Σ ∇_Σ F Σ`, `dU/dt = -η_φ U G_κ⁻¹ ∇_φ F`. Eq:gauge_group_retraction `U^{t+1} = U^t exp(-η · ∇̃_φ F)`. Chart-coordinate `dφ/dt = -η_φ ∇̃_φ F + O(‖·‖²)` is the first-order truncation; canonical is the group-level form.
- `tex:2544-2570` — Killing-form preconditioner `κ(X,Y) = c_G tr(XY) + c'_G tr(X)tr(Y)`; non-compact G needs pullback through `Ψ(ad_φ) = (e^z-1)/z` (Eq:pullback_metric).
- `tex:1119-1180` — §"Non-Uniform Attention Priors": causal masking, sliding window, ALiBi, relative position biases as `log π_ij` substitutions in eq:beta_optimal.

Out of scope (per user's language-modeling-only restriction):

- §"Meta-agent Formation", §"It From Bit: pullback construction", §"Lorentzian Signature Problem", §"Statistical Precision as Configuration-Space Stiffness" (mass analogy / Hessian), §"Consciousness", §"Multi-Agent Simulation", §"RG Construction" (PIFB §4408+), §"Meta-Agent Emergence Simulations".

## Canon excerpts (external sources of truth)

- **Variational inference / Free Energy Principle**: Friston 2010 (free energy formulation; Eq. 2.2 the canonical FEP F = E_q[ln q - ln p(o,x)]); Beal 2003 §2.3 (mean-field VMP gradient); Bishop 2006 Ch. 10 (variational inference; CAVI sequential updates monotone, parallel not guaranteed); Blei/Kucukelbir/Jordan 2017 (VI review).
- **Information geometry**: Amari 2016 Ch. 2-3 (Fisher metric for Gaussians: block-diagonal `g_μμ = Σ⁻¹`, `g_ΣΣ`); natural gradient `∇̃ F = G⁻¹ ∇F`; for the mean block this gives `Σ · ∇_μ F`, for the covariance block `2Σ · ∇_Σ F · Σ`.
- **Differential geometry / Lie theory**: Nakahara 2003 §5.6, §10.1-10.3 (matrix Lie groups, exponential map, principal bundles, parallel transport `Σ → Ω Σ Ω^T` as the conjugation action of the structure group on the associated bundle); Hall 2015 Ch. 2-3 (matrix exponential, BCH formula); Baez & Muniain (Gauge fields, knots, gravity) — cocycle `Ω_ij Ω_jk = Ω_ik` and `Ω_ii = I`.
- **Gauge theory**: cocycle / right-invariant retraction `g^{t+1} = g^t exp(η X)` is the standard Lie-group natural-gradient update (Amari 2016 §3.4, Absil/Mahony/Sepulchre *Optimization on Manifolds*).
- **Killing form**: Hall 2015 §7.3 — Killing form `B(X,Y) = tr(ad_X ad_Y)`; for `gl(K)`, `B(X,Y) = 2K tr(XY) - 2 tr(X)tr(Y)` (matches PIFB:2558 schema up to scalar). Bi-invariant on semisimple groups, indefinite for non-compact `gl(K)` — pullback metric (Eq:pullback_metric, PIFB:2559-2564) is needed for positive-definiteness.
- **Transformer attention**: Vaswani et al. 2017 (Eq:3 scaled dot-product `Attention(Q,K,V) = softmax(QK^T/√d_k) V`); RoPE: Su et al. 2021; ALiBi: Press et al. 2021.

## What this evidence does NOT settle

1. **Whether the manuscript "claims to implement" the canonical 5-term F is binding.** The /vfe e_step.py docstring (lines 13-67) admits dropping two terms (`λ_h·KL(s‖h)`, `γ·KL(s‖s)+entropy`). The user's claim asks whether **any toggle setting** lifts the dropped terms into an active code path. *No flag in `VFEConfig` adds a `γ`-coupled model term or a `λ_h` hyper-prior term.* The question for the debate is whether these terms are required for the **language-modeling-only** restriction or whether the manuscript's own §"Dynamical Structure" §1502+ adiabatic argument (`η_s ≪ η_q`, with `s` frozen at the perception timescale) makes them legitimately absent in the zero-dimensional / per-position transformer limit.
2. **Whether the cross-entropy term at the model level satisfies the manuscript's observation operator.** PIFB:1441-1467 explicitly treats `-E_q[log q_e]` (cross-entropy) as the canonical form of the observation operator and identifies it as "structurally identical to term 5" of the canonical F. So this is *not* a violation per the manuscript itself.
3. **Whether the per-dimension Bayesian α (E-step docstring lines 50-66) is "the manuscript's α" or a research deviation.** PIFB:1308-1313 derives scalar `α_i*`; code has per-K-dim α. /vfe's own docstring admits this is "a stronger generalisation... not currently derived in the manuscript". Red may attack on novelty-without-derivation; blue may argue scalar is the `K=1` reduction of the per-K form and the closed-form Eq:state_dependent_alpha holds dimension-by-dimension under diagonal Gaussian assumptions.
4. **Whether `prior_handoff_sigma=1.0` actually recovers the canonical hierarchical VI handoff that PIFB §1502+ prescribes.** `stack.py:120-140` blends posterior σ with embedding σ and applies a `clamp(1e-4, sigma_max)` floor. The φ-handoff is hard-wired (`new_prior_phi = initial_priors.phi`) with NO toggle to pass posterior φ as next-layer prior φ. PIFB §1535-1545 says φ "may evolve" under natural-gradient flow on `F_frame` but does not prescribe a specific cross-layer cascade in the L-block transformer limit. The unilateral φ freeze is a structural choice the debate must adjudicate against the manuscript.
5. **Whether the diagonal-σ transport approximation (`diagonal_covariance=True, exact_diagonal_transport=False`) is "in the pure path" the user is asking about.** PIFB:1619-1626 and Nakahara §10.3 prescribe the exact sandwich. Under `(diagonal_covariance=False, use_prior_bank=True, gauge_fixed_priors=True, exact_full_cov_decode=True, norm_type ∈ {mahalnorm, centered_mahalnorm}, rope_full_gauge ∈ {vfe_only, both})`, the full-covariance path is reached end-to-end. Whether *that specific* toggle combination is the "pure path" red and blue should be debating, vs. the active-config diagonal path, is itself a question.
6. **Whether `bch_order=1` (additive φ composition) is the chart-coordinate truncation PIFB:2572-2576 calls a first-order approximation, and whether `bch_order>=2` recovers exactness.** Active config uses `bch_order=3`; default is `bch_order=4`. The truncated BCH series is only exact for abelian generators; nonabelian generators leave a residual at every fixed order.
7. **Whether the `phi_preconditioner='pullback'` option (the only positive-definite metric for non-compact `gl(K)`, PIFB:2559-2564 Eq:pullback_metric) is reachable in /vfe.** `config.py:597-609` actively rejects `'pullback'` at runtime in the inner E-step path because `structure_constants` is not threaded through; the metric math was corrected on 2026-05-20 and is only available via the outer optimizer `RiemannianAdamW(metric='pullback')`. **Inside the E-step, no pullback preconditioner is reachable today.**
