# Evidence Pack — section-3-gauge-covariant-vfe

## Antecedent material referenced by §3

- `Attention/GL(K)_attention.tex:515-566` — §2.1.3 Theorem 1 "GL(K) Gauge Invariance of KL Divergence." The foundational invariance result `D_KL(Ω_* P ‖ Ω_* Q) = D_KL(P ‖ Q)` for `Ω ∈ GL(K)`, proven by direct computation on the trace, quadratic, and log-det terms. The proof at lines 539–555 is correct (`(det Ω)²` factors cancel; sandwich product `Σ_{transported} = Ω Σ Ω^T` is the load-bearing identity throughout the framework).

- `Attention/GL(K)_attention.tex:564` — Honest caveat: "our exponential parameterization `e^φ` restricts to `det > 0` (since `det(exp(φ)) = exp(tr(φ)) > 0`), but this is merely the identity component."

## §3.1 Agent State Space (lines 576–608)

- Lines 580–597: dual-latent setup `k_i ∈ R^{K_q}` (belief) and `m_i ∈ R^{K_p}` (model) with separate Gaussian state functions `q_i, s_i` and separate Gaussian base priors `p_i, r_i`. Variances positive-definite.
- Line 608: "These priors are local in that they live within each agent's own gauge frame `φ` and need not be related across agents until transported via the gauge connection."

## §3.2 Gauge Transport (lines 610–633)

- Eq. eq:gauge_frame_rotation at line 615: `Ω_ij(c) = e^{φ_i(c)} · e^{-φ_j(c)} ∈ GL⁺(K)`.
- Line 619: `φ_i: U_i → gl(K)` is "agent i's gauge frame field (an open local section of the Lie algebra bundle)." `GL⁺(K) = {A ∈ GL(K) : det A > 0}` is the identity component.
- Eq. eq:gauge_action_on_vectors at lines 621–631: `Ω_ij k_j := ρ_q(Ω_ij) k_j ∈ R^{K_q}` and similarly for `m_j` via `ρ_p(Ω̃_ij)`. Two representations of the same group element.

## §3.2.1 Vanishing Holonomy and Flat Bundle (lines 635–666)

- Lemma 1 at line 640–650: for `Ω_ij = g_i g_j⁻¹` with vertex-local `g_i ∈ G`, the holonomy `H_ijk = Ω_ij Ω_jk Ω_ki = I` identically.
- Line 656: "This is a defining property of the vertex-frame parameterization, not a derived dynamical statement: the choice `Ω_ij = g_i g_j⁻¹` encodes a globally trivial principal G-bundle, and Lemma 1 states the cocycle identity that any such trivialization satisfies."
- Eq. eq:edge_relaxed_omega_glk at line 658: edge-relaxed extension `Ω_ij = exp(φ_i) exp(δ_ij·G) exp(-φ_j)` for non-flat structure. "The companion paper [Dennis2025it] develops this discrete Regime II structure in detail."

## §3.3 Single-Agent FEP (lines 668–677)

- Eq. eq:single_agent_fep at lines 672–675: `F_i^{single} = KL(q_i ‖ p_i) - E_{q_i}[log p(o_i | k_i)]`.
- Citations [friston2010free, parr2022active].
- Line 677: deferral of symmetric two-channel form to companion paper; cross-fiber morphisms `Φ_i, Φ̃_i` from Table 1 explicitly *not* invoked in Eq. eq:free_energy_final.

## §3.4 Deriving Attention from Mixture-of-Sources (lines 679–771)

**This subsection was substantially edited by the softmax-β-stationarity debate.** Current state:
- Line 682 (post-edit): engineered soft-assignment Lagrangian framing per Cuturi2013 + Boyd2004.
- Line 697 (post-edit): explicit "components held fixed during inner minimization; alignment FE is an engineered consensus functional; entropy term added to make softmax stationary."
- Line 744 (post-edit): strict-convexity statement with Hessian `diag(1/β_k)`, log-barrier non-binding at interior, citations Boyd2004 + Wainwright2008.
- Lines 754–771: softmax derivation `β_ik* = π_k exp(-E_ik/τ) / Σ π_m exp(-E_im/τ)`, with τ rescaling and the canonical row-Lagrangian form Eq. eq:F_align_canonical_tau.

**Out of scope for this debate.** Adjudicated 2026-05-19 by debate `softmax-beta-stationary-point` (RED_WINS).

## §3.5 Full Free Energy (lines 840–874)

**Lines 868–874 were edited by the canonical-F-vs-surrogate debate.** Eq. eq:free_energy_final at lines 845–855:
```
F_red = Σ_i KL(q_i ‖ p_i) - τ Σ_i log Z_i - E_q[log p(o | {k_i})]
```
with `Z_i = Σ_j π_j exp(-E_ij/τ)`.

The "Autograd versus reduced-free-energy gradients" paragraph at line 868 contains the gap formula Eq. eq:autograd_envelope_gap at line 870 and the adopted-convention statement at line 874.

**Out of scope.** Adjudicated by debate `canonical-F-vs-surrogate` (RED_WINS).

## §3.6 Interpretation (lines 877–893)

Three terms: belief prior `KL(q_i ‖ p_i)`, belief alignment `β_ij KL(q_i ‖ Ω_ij q_j)`, observation likelihood `-E_q[log p(o | {k_i})]`. Plain-language gloss on each.

## §3.7 State-Dependent Prior Precision (lines 895–962)

- Eq. eq:free_energy_adaptive at lines 906–911: `F_i = α_i KL(q_i ‖ p_i) + R(α_i) + Σ_j β_ij KL(q_i ‖ Ω_ij q_j) - E_{q_i}[log p(o_i | k_i)]`.
- Eq. eq:precision_regularizer at lines 916–918: `R(α_i) = b₀ α_i - c₀ log α_i` with `b₀, c₀ > 0`.
- Eq. eq:state_dependent_alpha at lines 933–940: `α_i* = c₀ / (b₀ + KL(q_i ‖ p_i))`.
- Eq. eq:alpha_chain_rule at line 952: `∂α_i*/∂θ = -(α_i*)²/c₀ · ∂KL/∂θ`.
- Eq. eq:alpha_product_rule at line 957: `∂(α_i* KL)/∂θ = (α_i*)² (b₀/c₀) ∂KL/∂θ`.
- Lines 961–962 (post-canonical-F-debate edit): envelope-theorem reference and reference-implementation pointer to `transformer/core/vfe_gradients.py`.

## §3.8 Belief Dynamics (lines 965–977)

**Lines 967–973 were edited by the canonical-F-vs-surrogate debate.** Belief dynamics `∂q_i/∂t = -η_q ∇_{q_i}[Σ_j β_ij* E_ij + α_i KL(q_i ‖ p_i) - E_{q_i}[log p(o_i | k_i)]]` on the entropy-suppressed surrogate (autograd convention).

Line 977 parenthetical: "in the full theory with general, possibly curved, base manifolds we would necessarily integrate over agent supports and overlaps in order to obtain the complete variational free energy (which we needn't consider for our present transformer derivations). This then introduces extensive and rich geometry including Riemann curvature, gauge curvature, Fischer curvature, induced gauge connections, belief/model transport between and within agents and fibers, non-trivial vertical and horizontal holonomies, and much, much more."

## §3.9 Symmetry Breaking (lines 979–985)

- Line 981: vacuum state `q_i = Ω_ij q_j` and `p_i = Ω̃_ij p_j` for all pairs; degenerate manifold of ground states parameterized by gauge orbit.
- Line 983 Elitzur disclaimer: "local gauge symmetries are more precisely understood as redundancies of description rather than physical symmetries ... Elitzur's theorem forbids spontaneous breaking of local gauge symmetries in lattice gauge theory [weinberg1995quantum]. Our gauge group acts as a reparameterization symmetry of the KL divergence (Theorem 1), and the degeneracy of the vacuum manifold is a consequence of this reparameterization freedom rather than of a global physical symmetry."
- Line 985: explicit symmetry breaking by observation/likelihood: "The observation/likelihood term `-E_q[log p(o|k)]` is manifestly not gauge-invariant, and it is this term that forces agents into distinct configurations." Open question: "Whether the vacuum manifold supports approximate zero modes of the Hessian (corresponding to directions of near-zero curvature along the gauge orbit) is a question we leave to future work."

## §3.10 Summary (lines 987–992)

Two-principle summary: (i) standard single-agent FEP; (ii) mixture-of-sources for inter-agent communication. Forward KL is the unique f-divergence preserving exponential-family closure (referencing Appendix A uniqueness theorem at `Attention/GL(K)_supplementary.tex:1203` from the prior softmax-β debate).

## Canon excerpts — external standards for the unaudited sub-claims

### Sub-claim A (Ω parameterization)

- **Lattice gauge theory canonical form [Wilson 1974; Kogut-Susskind 1975].** A standard lattice gauge connection is parameterized by edge variables `U_e ∈ G` (one per oriented edge), with the gauge transformation law `U_e → g_s U_e g_t⁻¹` where `s, t` are the source/target vertices. The cocycle condition `U_{ij} U_{jk} = U_{ik}` for all triples is equivalent to the connection being a *pure gauge* (flat). Wilson loops `W_C = Tr ∏_{e ∈ C} U_e` are the gauge-invariant order parameters; pure gauge gives `W_C = K` for all loops.
- The manuscript's parameterization `Ω_ij = exp(φ_i) exp(-φ_j)` is the vertex-frame parameterization, which equivalent to `U_{ij} = g_i g_j⁻¹` for `g_i = exp(φ_i)`. This *is* the cocycle / pure-gauge specialization; honestly labeled at line 656.

### Sub-claim D (log-barrier regularizer)

- **Conjugate prior on precision [Bishop 2006 §2.3.6 "The Gaussian Distribution"; Murphy 2012 §4.6.1].** The Gamma distribution `Gamma(α; c, b) = b^c / Γ(c) · α^{c-1} exp(-b α)` for `α > 0`, `c, b > 0` is the conjugate prior for the precision parameter (inverse variance) of a Gaussian. Its negative log-density is `-log p(α) = b α - (c - 1) log α + const`.
- The manuscript's `R(α_i) = b₀ α_i - c₀ log α_i` matches `-log Gamma(α_i; c₀ + 1, b₀)` (up to additive constants). So `R(α_i)` IS the negative log-density of a Gamma hyperprior on `α_i`, and `α_i* = argmin_α [α · KL + R(α)]` IS the MAP estimate of `α` under a Gamma prior given a quadratic-in-`α` likelihood-substitute. The "natural choice" framing at line 914 is theoretically motivated.

### Sub-claim F (symmetry breaking)

- **Elitzur's theorem [Elitzur 1975; Weinberg 1995 vol. 2 §21.6].** In a lattice gauge theory, the expectation value of any non-gauge-invariant local operator is zero in the absence of explicit gauge-fixing. Equivalently, local gauge symmetries cannot be spontaneously broken — there are no Goldstone modes from local gauge redundancies. The "spontaneous breaking" of gauge symmetry in the Higgs mechanism is, more precisely, the gauging of a global symmetry whose Goldstone modes are eaten by the gauge fields.
- The manuscript's correct interpretation: the GL(K) action on `(μ, Σ)` is a *reparameterization* (Theorem 1 says KL is invariant under simultaneous push-forward), and the vacuum manifold's degeneracy is a redundancy-of-description issue. The Elitzur disclaimer at line 983 is the correct way to handle this.

## Code references (for sub-claims that touch implementation)

- `transformer/core/vfe_gradients.py` — referenced at §3.7 line 962 as the "reference implementation" that evaluates the corrected form Eq. eq:alpha_product_rule. The closed-form is wired up only when `E_learnable_alpha=True` (cf. `transformer/vfe/e_step.py:848-852`).

## What this evidence does NOT settle

1. **Whether `Ω_ij = exp(φ_i) exp(-φ_j)` derivations elsewhere in the manuscript or supplementary quietly assume the more general edge-relaxed form.** The judge should consider whether the edge-relaxed `Ω_ij = exp(φ_i) exp(δ_ij·G) exp(-φ_j)` mentioned at line 658 is ever invoked in §3 or §4 derivations *as if* it were the default.

2. **Whether the dual-fiber state space at §3.1 introduces unstated constraints on the framework.** The cross-fiber morphisms `Φ_i, Φ̃_i` are mentioned in Table 1 but not used; if they reappear silently in §3.4 (mixture model) or §3.5 (full F), the dual-fiber framing is load-bearing rather than honest scaffolding.

3. **Whether the log-barrier regularizer is theoretically motivated or merely "convenient."** The Gamma-conjugate-prior derivation above suggests it IS theoretically motivated, but the manuscript at line 914 only says "a natural choice," not "the negative log-density of a Gamma hyperprior." A red-team strike: the manuscript should make this explicit (editorial clarification, not structural correction).

4. **Whether the §3.8 "curved base manifolds" parenthetical at line 977 is honest future-work signaling or load-bearing scaffolding for the framework.** If any §3 equation depends on overlap integrals, gauge curvature, or Fisher curvature, the parenthetical is misleading.

5. **Whether the symmetry-breaking analysis at §3.9 is sufficient.** The "approximate zero modes of the Hessian" question at line 985 is explicitly labeled future work — but if such zero modes are present and observable, they may affect the framework's empirical predictions in ways not currently audited.
