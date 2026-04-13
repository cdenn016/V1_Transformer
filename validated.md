# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:14:45 2026

@author: chris and christine
"""

## Verified Mathematical Correctness (Audited 2026-04-11)

The following have been line-by-line verified against first principles by deep audit.

**Core Formulas (all PASS):**
- KL divergence: dense and diagonal formulas, sign conventions, factor-of-1/2 — `kl_computation.py`
- Self-coupling gradient: `∂KL(q∥p)/∂μ = Σ_p^{-1}(μ_q - μ_p)` — `vfe_gradients.py`
- Belief alignment gradient: `∂KL(q_i∥Ωq_j)/∂μ_i = Σ_t^{-1}(μ_i - Ωμ_j)` — `vfe_gradients.py`
- Dynamic β (softmax coupling) gradient: `∂β_j/∂θ = -(β_j/κ)(∂KL_j/∂θ - Σ_k β_k·∂KL_k/∂θ)` — `vfe_gradients.py`
- Natural gradient μ: `Δμ = -η·Σ·∇F` (Fisher metric) — `vfe_gradients.py`
- Natural gradient σ (diagonal): `Δσ = -η·2σ²·∇F` (Fisher metric) — `vfe_gradients.py`
- Natural gradient σ (full): `ΔΣ = -η·2Σ·∇Σ·Σ` (Fisher sandwich) — `vfe_gradients.py`
- Closed-form Σ equilibrium: `Σ^{-1} = (α+λ)^{-1}(α/Σ_p + λ·Σ_j β_ij/Σ_t)` — `vfe_closed_form.py`
- Closed-form μ equilibrium: precision-weighted combination of prior + transported neighbors — `vfe_closed_form.py`
- SPD retraction: affine-invariant exponential map `Σ_new = Σ^{1/2}·exp(Σ^{-1/2}·ΔΣ·Σ^{-1/2})·Σ^{1/2}` — `vfe_utils.py`
- Diagonal SPD retraction: `σ_new = σ·exp(τ·Δσ/σ)` — `vfe_utils.py`
- DEQ Neumann series implicit differentiation — `vfe_deq.py`
- Implicit EM scaling: `s_k = (α/σ_p²)/(α/σ_p² + Σβ/σ_q²)` in [0,1] — `vfe_implicit_em.py`
- Product-rule correction for learnable α — `vfe_gradients.py`

**Gauge Geometry (all PASS):**
- Covariance sandwich product `Ω·Σ·Ω^T` enforced consistently (20+ occurrences, 0 violations)
- Flat transport cocycle: `Ω_ij·Ω_jk = Ω_ik` verified to machine precision
- GL(K) gauge invariance chain: KL(h·q ∥ h·Ω·q') = KL(q ∥ Ω·q')
- SO(3) generators: skew-symmetry, cyclic commutation, Casimir C₂ = ℓ(ℓ+1)
- SO(N) generators: N(N-1)/2 basis, correct commutation
- GL(K) generators: K² basis (E_{ij}), correct normalization
- Killing form: `g̃_ab = 2K·tr(T_a^T T_b) - 2·tr(T_a)·tr(T_b)`, null on center, regularized
- Cartan projector: idempotent, correct eigenvalue structure
- Pullback metric at φ=0 reduces to Frobenius inner product (correct)
- Structure constants: antisymmetry f^c_{ab} = -f^c_{ba} verified
- RoPE: rotation correct, norm-preserving, covariance sandwich R·Σ·R^T verified
- Connection antisymmetrization: δ_ij = -δ_ji for bilinear mode

**Invariants (all PASS):**
- σ_p detachment in ALL E-step paths (iterative, closed-form, DEQ)
- Hyper-prior h frozen (registered buffer, .detach())
- No-NN constraint (only W_out is nn.Linear; connection MLP is documented exception)
- Three-limit reduction to standard QK^T/√d_k attention (verified end-to-end)
- Forward KL uniqueness (conditional on f-divergence class)
- Softmax from constrained VFE minimization (Lagrange multipliers → Gibbs form)
- Kappa scaling κ·√K consistent across all 7 occurrences

**Numerical Stability (all PASS):**
- Matrix exp norm clamp ≤ 20.0 (safe for float32)
- Cholesky progressive regularization (3 rounds + pseudoinverse fallback)
- AMP float32 guards on all sensitive ops
- NaN → kl_max (repulsive) handling
- Newton-Schulz convergence basin (σ_max ≤ 1.7 < √3)
- _safe_eigh SVD fallback with sign recovery + eigenvalue-gap degeneracy breaker (prevents backward NaN)

**Numerical Gradient Verification (finite-difference, PASS):**
- grad_mu: max relative error 8.95e-06 (vs central differences, float64, ε=1e-5)
- grad_sigma: max relative error 3.13e-06
- Script: `scripts/verify_vfe_gradients_fd.py`

**Advanced Features (all PASS):**
- Learnable α: log-barrier `α_k = c0/(b0+KL_k)`, product-rule in all 4 gradient paths
- Prior bank: gauge-fixed priors `p_v = A_v ▷ p_0`, sandwich product correct, σ_p detached
- Cross-head coupling: super-block merging via union-find, KL decomposes additively
- VFE nonlinearities: direct term (λ_belief · β·∇KL) is **Boltzmann GLU** (GELU/SiLU analog); softmax coupling (λ_softmax · ∂β/∂μ · KL) is **attention-variance coupling** (second-order, structurally distinct from GELU)
- BCH composition: all 6 Dynkin degree-5 terms implemented, convergence rate O(ε^6) confirmed
- MahalanobisNorm: sigma now passed in all forward_with_attention paths (was missing → RMSNorm fallback)
- Pure VFE reference implementation: mathematically consistent with main (11 comparison items verified)
- Training gradient flow: all 11 detach sites correct, no gradient leakage found
- Manuscript equations: all 8 unverified groups cross-referenced and confirmed (value aggregation, multi-head, Bayesian α, temperature, BERT, symmetry breaking, weight decay, LayerNorm)
- Information geometry: Fisher-Rao metric consistent across natural gradients, SPD retraction, and observation Hessian

**Constants (centralized 2026-04-11):**
- `KL_CEIL_BASE=100.0`, `KL_CEIL_SCALE=5.0`: single source of truth in `vfe_utils.py`
- BCH default order: 4 (degree-5, all 6 Dynkin terms, O(ε^6) error)

**Known Manuscript Issues (for peer review):**
- belief_inertia_unified.tex: relaxation time τ inconsistency (2M/γ vs M/γ at lines 433/485)
- belief_inertia_unified.tex: GL(d) notation `Λ̃ = ΩΛΩ^T` wrong for non-orthogonal Ω (correct: `Ω^{-T}ΛΩ^{-1}`)
- belief_inertia_unified.tex: exponential family claim at line 2229 wrong in general (missing ∇³A correction)
- GL(K)_supplementary.tex: forward KL uniqueness is conditional on f-divergences (stated but could be more prominent)
- Code comments: "GELU-like" label was on the wrong term — FIXED: direct term (λ_belief) is the Boltzmann GLU, softmax coupling (λ_softmax) is attention-variance coupling
- Learnable α: precision regularizer R(α) from manuscript not implemented (design gap, not bug)
- Learnable α: "Gamma-Normal conjugacy" label is interpretive, not mathematically exact
- belief_inertia_unified.tex line 472: resonance amplitude formula is standard oscillator, not proven for curved belief manifold
- belief_inertia_unified.tex line 547: momentum transfer Ω_ki^T vs Ω_ki^{-1} ambiguity for non-orthogonal GL(K)
- belief_inertia_unified.tex line 538: linearization regime has no quantitative radius of validity
- Participatory_it_from_bit.tex abstract: "resolve Lorentzian signature" is overstated — 1+1 pathway demonstrated, not dynamical selection
- GL(K)_attention.tex: κ·√K scaling is structural/empirical (status S), not derived from VFE first principles

**Phase 6 Audit (2026-04-12):**

*Bugs Fixed:*
- Gradient clipping mismatch (HIGH): experiment_runner.py applied Euclidean clip_grad_norm_ to Killing-preconditioned phi gradients when optimizer was AdamW. Fixed: phi groups now use Riemannian norm clipping in the Killing metric.
- Mu Riemannian norm error (MEDIUM): optimizer.py _riemannian_clip computed `(ξ/σ²)² = g²` instead of `ξ²/σ² = σ²g²` for Fisher norm clipping. Missing factor of σ² per dimension. Fixed: `(p.grad/sigma)**2` → `p.grad**2/sigma`.
- Eigenvalue floor in gauge_geometry.py logm raised from 1e-7 to 1e-6 (above float32 machine epsilon)
- fit_convergence_rate() now returns NaN for oscillatory convergence curves instead of silently clamping to 0
- PriorBank logs warning when defaulting phi_dim=3 without generators
- BlockConfig default inconsistencies: e_step_sigma_floor (0.01→0.1) and rope_base (50→10000.0) dataclass defaults now match from_config() defaults
- Fisher-Rao distance formula (HIGH): `(Δμ²+Δσ²)/(2σ₁σ₂)` → `(Δμ²+2Δσ²)/(4σ₁σ₂)` — old formula inflated mean-displacement distances by √2 (verified via Poincaré half-plane isometry + infinitesimal check)
- E-step early exit: optional `e_step_early_exit_tol` for convergence-based early termination (default None = disabled)
- Learnable LR clamped: `softplus(raw_lr).clamp(max=0.5)` prevents E-step step size explosion
- Debug .item() batching: 4 GPU→CPU syncs per iteration → 1 batched transfer

*New FD Validation (all PASS):*
- Implicit EM IFT scale factors: compute_implicit_em_scales() formula verified (6 tests) — vfe_implicit_em.py
- DEQ Neumann backward: Neumann series (I-J^T)^{-1} converges to exact inverse for linear contraction (2 tests) — vfe_deq.py
- Sandwich product derivative: ∂(Ω·Σ·Ω^T)/∂φ autograd matches central FD to < 1e-4 relative error (3 tests) — transport_ops.py
- Connection MLP equivariance: bilinear mode is equivariant under SO(d_head); MLP mode is NOT (documented exception confirmed, 3 tests) — connection.py
- Oscillatory convergence detection: fit_convergence_rate correctly returns NaN for oscillatory, positive for monotonic (4 tests) — fiber_trajectory.py

*Deep Code Reviews (3 agents, ~10K lines, all PASS):*
- Training pipeline: loss computation, 11 detach sites, optimizer preconditioning order, LR schedule interaction — all correct
- Transport ops + attention: factored transport, 20+ sandwich product sites, RoPE, value aggregation, AMP guards, edge cases — all correct
- Prior bank + cross-head: gauge-fixed priors, sigma detachment, union-find, fused decode KL, MahalanobisNorm passing — all correct
- Q_j factorization for full-cov closed-form E-step: verified to machine precision (1e-14)
- Closed-form Σ* = (α+λ)A^{-1} equilibrium: VFE gradient = 0 to 1e-15

*Phase 9 Verifications (_vfe_iteration body, model paths, edge cases):*
- Full-cov SPD retraction eigenvalue bounds: trust region + spectral clamp = safe
- forward_with_attention: return_attention kwarg, detach logic, logits path all correct
- Sigma retraction (both diagonal and full-cov): verified correct
- Omega retraction: left-invariant pullback ξ = Ω⁻¹·∂F/∂Ω correct
- Weight initialization: KL≈0 at init confirmed
- Kappa warmup: exists in experiment_runner.py, freeze/unfreeze correct
- VFE descent test: natural gradient step produces F(q_new) < F(q_old) (new FD test)
- 15 additional items verified correct (conditioning, isotropic enforcement, observation Hessian, etc.)

*Phase 7 Verifications (probability/statistics, differential geometry):*
- Fisher-Rao geodesic distance: fixed √2 error in mu coefficient (Poincaré half-plane derivation)
- Rényi α-divergence (α=0.275): all 4 gradient paths FD-verified (self-coupling mu/sigma, alignment mu/sigma)
- Active inference BALD MI: two-pass Welford estimation mathematically proven correct
- Spectral analysis: effective_rank, entropy, asymmetry formulas all standard and correct
- Fisher-Rao infinitesimal distance: already correct (midpoint metric evaluation)

*Prior Audit Items Verified:*
- Double phi preconditioning: FIXED (conditional guard in experiment_runner.py)
- obs_in_vfe: CORRECT (gated via use_obs_in_vfe config)
- Belief alignment scaling: CORRECT (β-weighted sum scales with neighbors per VFE math)
- Holonomy computation: CORRECT (triple product in holonomy.py)
- DEQ implementation: mathematically correct (Neumann series, divergence guards, mutual exclusions)
- Implicit EM: correct formulas, FD validated
- Connection MLP gauge equivariance: NOT equivariant (negative test confirms known limitation)
