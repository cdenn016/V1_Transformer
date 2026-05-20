# Blue Opening — section-3-gauge-covariant-vfe

## Steelman (opposing position)

§3 of `Attention/GL(K)_attention.tex` is decorated with structural restrictions and forward-looking parentheticals (vertex-frame flat bundle at line 656, dual-fiber morphisms declared dormant at line 677, curved-base machinery deferred at line 977) which red will read as concealed load-bearing dependencies that quietly enter the §3 derivations (mixture-of-sources, alignment energy, log-barrier precision, vacuum manifold) and undermine the claim of theoretical purity.

## Position

The unaudited foundational content of §3 — sub-claims A through F — is internally consistent and externally compatible with primary-source canon. Each restriction is explicitly named at the manuscript line where the specialization is introduced, none of the §3 equations rely on the deferred machinery, the FEP citation is verbatim Form 3 of `[Friston2010, ParrPezzuloFriston2022 Ch. 2]`, the log-barrier regularizer matches the negative log density of a Gamma conjugate prior on the precision parameter `[Bishop2006 §2.3.6, Murphy2012 §4.6.1]`, and the Elitzur disclaimer at line 983 directly cites `[Weinberg1995 vol. 2 §21.6]` and resolves the redundancy-vs-physical-symmetry distinction at the right level of rigor. The §3 derivation survives primary-source adversarial pressure on all six sub-claims.

## Evidence

### Sub-claim A — Ω parameterization is honestly labeled, not hidden

`Attention/GL(K)_attention.tex:656` reads in full: "This is a defining property of the vertex-frame parameterization, not a derived dynamical statement: the choice $\Omega_{ij} = g_i g_j^{-1}$ encodes a globally trivial principal $G$-bundle, and Lemma~\ref{thm:vanishing_holonomy} states the cocycle identity that any such trivialization satisfies." This is the highest standard of mathematical disclosure: the structural specialization is named at the point of introduction, the lemma is recast as definitional rather than dynamical, and the edge-relaxed extension `Ω_ij = exp(φ_i) exp(δ_ij·G) exp(-φ_j)` is given immediately at `Attention/GL(K)_attention.tex:658` with citations `[WilsonConfinement1974, KogutSusskind1975, Creutz1983]` and a deferral to `[Dennis2025it]`. This matches the lattice gauge canon: `external_canon_math.md:88` records that `U_{ij} = g_i g_j^{-1}` is precisely the pure-gauge / cocycle specialization, and the manuscript labels it as such.

No equation downstream in §3 references the edge-relaxed `δ_ij` form. The mixture-of-sources construction at `Attention/GL(K)_attention.tex:693` uses `P(k|z=j) = N(k; Ω_ij μ_j, Ω_ij Σ_j Ω_ij^T)` with the flat `Ω_ij = exp(φ_i) exp(-φ_j)` parameterization; the alignment energy at line 727 reads `E_ij = D_KL[q_i ‖ Ω_ij q_j]` with the same Ω. The reduced free energy at lines 845–855 uses the same Ω throughout. The flat-bundle restriction is therefore the operating assumption of §3 — not a buried dependency.

### Sub-claim B — Dual-fiber state space is honest scaffolding

`Attention/GL(K)_attention.tex:677` reads: "We declare here that the cross-fiber morphisms $\Phi_i, \tilde{\Phi}_i$ named in Table~\ref{tab:notation} do not enter \eqref{eq:free_energy_final}: belief and model channels couple only through (i) the shared structure group $G$ (with $\Omega_{ij}$ and $\tilde{\Omega}_{ij}$ acting as the same group element in two representations) and (ii) the iterative VFE updates; the cross-bundle morphisms are anticipated for future development and are not instantiated in the present implementation." This is a direct, in-text declaration of non-load-bearing scaffolding.

Inspection of the reduced free energy `Attention/GL(K)_attention.tex:845-855` confirms this: the right-hand side contains only `D_KL(q_i ‖ p_i)`, `-τ log Z_i`, and `-E_q[log p(o|{k_i})]`. No `Φ_i` or `Φ̃_i` appears. The model-channel variables `m_i, s_i, r_i, γ_ij, λ_h` are declared deferred to Supplementary Appendix G at line 677 and to the companion paper at the same line. §3 operates entirely on the belief channel with `(q_i, p_i, β_ij, Ω_ij)`.

### Sub-claim C — Single-agent FEP citation matches the canon

The manuscript equation at `Attention/GL(K)_attention.tex:672-675`:
```
F_i^single = D_KL(q_i ‖ p_i) - E_{q_i}[log p(o_i | k_i)]
```
is verbatim Form 3 of the canonical FEP per `external_canon_inference.md:17`:
```
F[q] = E_q[-log p(o|s)] + KL(q(s) ‖ p(s))
```
with `s → k_i` and the term ordering swapped (algebraically identical). The citations `[friston2010free, parr2022active]` at line 670 are the canonical FEP references. The manuscript explicitly disclaims active-inference policy machinery: line 670 introduces this as "the standard variational free energy for a single agent" — perceptual inference, not policy selection. There is no `G(π)` term and no claim that this form implies active inference; it is the canonical single-agent state-estimation form.

### Sub-claim D — Log-barrier regularizer is theoretically motivated (Gamma conjugate prior)

`R(α_i) = b_0 α_i - c_0 log α_i` at `Attention/GL(K)_attention.tex:917` matches the negative log density of `Gamma(α; c_0 + 1, b_0)`, the standard conjugate prior on the precision of a Gaussian likelihood per `[Bishop2006 §2.3.6, Murphy2012 §4.6.1]` (recorded at `external_canon_inference.md` and at `01_evidence.md:92-93`).

Symbolic verification (sympy):
```
Input:  R(α) = b0*α - c0*log(α); F(α) = α*KL + R(α); solve(dF/dα = 0, α)
Output: α* = c0 / (KL + b0)
Check:  dF/dα at α*: 0
```
This matches `Attention/GL(K)_attention.tex:937` exactly: `α_i* = c_0/(b_0 + D_KL(q_i ‖ p_i))`.

The "natural choice" framing at line 914 is honest: there are at least three compatible derivations — (i) the Gamma-prior MAP on the precision parameter `[Bishop2006 §2.3.6]`, (ii) the standard log-barrier convex relaxation of `α > 0` `[Boyd2004 §11.2]`, and (iii) the unique smooth regularizer that gives a closed-form rational solution. The manuscript picks the simplest functional form and labels it as one choice. A red strike that the manuscript should explicitly cite `[Bishop2006]` is an editorial improvement, not a structural correctness issue.

### Sub-claim E — Curved-base parenthetical is dependence-free

`Attention/GL(K)_attention.tex:574` opens §3 with the operating restriction: "All probability distributions are defined at a specific base manifold point $c = c^*$ unless otherwise noted." Every equation in §3 is evaluated at this single base manifold point. The parenthetical at line 977 is explicit: "in the full theory with general, possibly curved, base manifolds we would necessarily integrate over agent supports and overlaps in order to obtain the complete variational free energy (which we needn't consider for our present transformer derivations)."

Inspection of every numbered equation in §3 — eq:dual_latents, eq:gaussian_states, eq:base_priors, eq:gauge_frame_rotation, eq:gauge_action_on_vectors, eq:holonomy, eq:edge_relaxed_omega_glk, eq:single_agent_fep, eq:mixture_joint, eq:mixture_posterior, eq:mixture_free_energy, eq:mixture_energy_entropy, eq:mixture_softmax_general, eq:mixture_softmax, eq:F_align_canonical_tau, eq:causal_prior, eq:causal_attention, eq:window_prior, eq:alibi_prior, eq:alibi_attention, eq:relative_bias_prior, eq:relative_bias_attention, eq:free_energy_final, eq:autograd_envelope_gap, eq:free_energy_adaptive, eq:precision_regularizer, eq:state_dependent_alpha, eq:alpha_chain_rule, eq:alpha_product_rule, eq:belief_dynamics — produces no integral over `c`, no Riemann curvature `R^a_bcd`, no Fisher curvature, no overlap integral, no induced connection. The parenthetical is honest future-work signaling.

### Sub-claim F — Elitzur disclaimer is canon-aligned

`Attention/GL(K)_attention.tex:983` reads: "In physics, local gauge symmetries are more precisely understood as redundancies of description rather than physical symmetries, and there are well-known subtleties in applying the language of spontaneous symmetry breaking to gauge redundancies (Elitzur's theorem forbids spontaneous breaking of local gauge symmetries in lattice gauge theory \citep{weinberg1995quantum}). Our gauge group acts as a reparameterization symmetry of the KL divergence (Theorem~\ref{thm:glk_invariance}), and the degeneracy of the vacuum manifold is a consequence of this reparameterization freedom rather than of a global physical symmetry."

This matches the canonical interpretation per `[Weinberg1995 vol. 2 §21.6]` recorded at `01_evidence.md:97-98`: Elitzur's theorem says non-gauge-invariant local operators have zero expectation value absent explicit gauge fixing, so "spontaneous breaking" of local gauge symmetry is a misnomer. The manuscript's stance — the symmetry under discussion is a *reparameterization* of the KL divergence (Theorem 1, `Attention/GL(K)_attention.tex:515-566`), not a physical local gauge symmetry — is precisely the correct disclaimer. The honest explicit-symmetry-breaking statement at line 985, "the observation/likelihood term $-E_q[\log p(o|k)]$ is manifestly not gauge-invariant," correctly identifies the source of the symmetry-breaking effect as explicit external coupling, which is well-defined regardless of Elitzur.

### Antecedent foundation — Theorem 1 invariance

The Phase-2 defense rests on `Attention/GL(K)_attention.tex:515-566`, Theorem 1 (GL(K) Gauge Invariance of KL Divergence). The proof at lines 539–555 confirms `D_KL(Ω_* P ‖ Ω_* Q) = D_KL(P ‖ Q)` by direct computation: the `(det Ω)^2` Jacobian factors cancel, the trace and quadratic terms invert through the sandwich `Σ → Ω Σ Ω^T`, and the log-det term cancels via `log|Ω Σ Ω^T| - log|Σ| = 2 log|det Ω|`. This is the foundational invariance result that makes the gauge group a reparameterization of KL — the structural fact the §3.9 disclaimer at line 983 invokes.

## Falsification conditions

This position is wrong if any of the following hold:

1. **Sub-claim A falls** if any §3 equation downstream of `Attention/GL(K)_attention.tex:658` invokes the edge-relaxed `Ω_ij = exp(φ_i) exp(δ_ij·G) exp(-φ_j)` form as the operating definition (rather than the vertex-frame `Ω_ij = exp(φ_i) exp(-φ_j)`), or if any equation in §4 or §5 silently extends from §3 using the non-flat form.

2. **Sub-claim B falls** if the cross-fiber morphisms `Φ_i, Φ̃_i` from Table 1 appear in any §3 derivation (mixture model at `Attention/GL(K)_attention.tex:684-697`, alignment FE at lines 710–734, full F at lines 840–874, state-dependent precision at lines 895–962), or if the dual-fiber state space introduces a constraint (e.g., a relationship between `K_q` and `K_p`, or a forced morphism between `q_i` and `s_i`) that is invoked but not declared.

3. **Sub-claim C falls** if the form `F_i^single = D_KL(q_i ‖ p_i) - E_{q_i}[log p(o_i | k_i)]` at line 673 contradicts the canonical FEP per `[Friston2010, ParrPezzuloFriston2022 Ch. 2]` (e.g., a missing `-log p(o)` term that is implicit in the standard derivation, or a sign error, or an incompatible interpretation of the conditional `p(o|k)`).

4. **Sub-claim D falls** if (i) the sympy-verified closed-form `α_i* = c_0/(b_0 + KL)` is shown to depend on a hidden constraint, (ii) the `R(α) = b_0 α - c_0 log α` form is shown to be incompatible with the Gamma negative log density in `[Bishop2006 §2.3.6]`, or (iii) red produces a primary-source canon claim that "log-barrier" cannot mean what the manuscript means by it at line 914.

5. **Sub-claim E falls** if any equation in §3 explicitly invokes overlap integrals over agent supports, Riemann curvature, gauge curvature, Fisher curvature, induced connections, vertical or horizontal holonomies, or non-trivial base-manifold geometry — that is, if line 977's parenthetical is load-bearing rather than informational.

6. **Sub-claim F falls** if the manuscript's vacuum-manifold analysis at lines 981–985 contradicts `[Weinberg1995 vol. 2 §21.6]` or `[Elitzur1975]` — for instance, if the manuscript claims the vacuum manifold's degeneracy *is* spontaneous breaking of a local gauge symmetry (the precise claim Elitzur forbids), rather than what it does claim (the degeneracy is a reparameterization-orbit redundancy resolved by the non-gauge-invariant observation term).

If red shows that the compound F (Eq. eq:free_energy_final) silently requires the curved-base machinery, the dual-fiber morphisms, or the edge-relaxed Ω that the manuscript explicitly declared non-load-bearing, the corresponding sub-claim falls and the compound claim falls with it. If red can only show that the manuscript should cite `[Bishop2006]` more explicitly at line 914 or that the disclaimer at line 977 is wordy, the compound claim survives as a robust derivation requiring at most editorial polish.
