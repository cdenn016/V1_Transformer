# Evidence Pack — supplementary-general-mathematical-framework

## Supplementary §General Mathematical Framework — full structure

The section spans `Attention/GL(K)_supplementary.tex` lines 46–177, comprising four subsections:

### Subsection 1 — Principal Bundle and Associated Bundles (lines 47–64)

- Line 49: "The following geometric and probabilistic constructions comprise standard methods in differential geometry and gauge theory. See references for details, proofs, and examples \citep{nakahara2003geometry, frankel2011geometry, blei2017variational, amari2016information}."
- Line 51: Principal G-bundle `π: N → C`. Standard setup.
- Line 53 ("Bundle triviality" paragraph): Explicit disclosure that the vertex-frame parameterization encodes a "globally trivial principal G-bundle" with vanishing curvature 2-form (Lemma 1 of main text). Edge-relaxed extension `Ω_ij = exp(φ_i) exp(δ_ij·G) exp(-φ_j)` promotes to non-trivial principal G-bundle for Regime II.
- Line 55: "Let `ρ_q: G → Aut(B_q)` and `ρ_p: G → Aut(B_p)` be representations of G on smooth statistical manifolds `B_q` (belief/recognition fiber) and `B_p` (model/prior fiber). These fibers are typically K-dimensional probability simplices `Δ^K` (for categorical distributions) or statistical manifolds with information-geometric structure (e.g., Gaussian manifolds, exponential families)."
- Lines 58–62: Associated bundles `E_q := N ×_{ρ_q} B_q`, `E_p := N ×_{ρ_p} B_p` with equivalence relation `(n·g, b) ∼_u (n, ρ_u(g) b)`.

### Subsection 2 — Agents and Multi-Agent Systems (lines 66–130)

- Lines 67–74: Agent definition `A^i = (σ^i_q, σ^i_p)` — pair of local sections over domain `U_i ⊂ C`.
- Line 76: Notation `q_i(c) := σ^i_q(c) ∈ B_q(c)`, `p_i(c) := σ^i_p(c) ∈ B_p(c)`.
- Lines 78–85: Multi-agent system `M = {A^i}_{i ∈ I}`.
- Lines 87–94: Meta-agent and epistemic death — agents satisfying `q_i(c) = q_j(c), p_i(c) = p_j(c)` on overlap. Note: pointwise equality without gauge transport.
- Lines 96–130: Hierarchical Meta-Agent Emergence and Cross-Scale Coupling subsubsection.
  - Line 102: Belief consensus `q_i = Ω_{ij} q_j ∀ i, j ∈ I_M`.
  - Line 103: Model consensus `s_i = Ω̃_{ij} s_j` — note this introduces `s_i` as a NEW field not previously defined.
  - Line 122–127: Renormalized coupling constants `β_ij^(ζ+1), γ_ij^(ζ+1)`, effective gauge frames `φ_M^(ζ+1) = average({φ_i^(ζ)})`.

### Subsection 3 — Bundle Morphisms and Transport Operators (lines 132–137)

- Six families of operators briefly named without derivation:
  - Intra-bundle: `Ω^(q)_ij: Γ(B_q) → Γ(B_q)`, `Ω^(p)_ij: Γ(B_p) → Γ(B_p)`.
  - Cross-scale: `Λ^s_{s'}: Γ^s(B_q) → Γ^{s'}(B_q)`, `Λ̃^s_{s'}: Γ^s(B_p) → Γ^{s'}(B_p)`.
  - Inter-bundle: `Θ^i_j: Γ(B_q) → Γ(B_p)`, `Θ̃^i_j: Γ(B_p) → Γ(B_q)`.
  - Global bundle: `Φ: E_p → E_q`, `Φ̃: E_q → E_p`.

### Subsection 4 — Gauge Frames and Connections (lines 139–174)

- Line 142: `φ_i: U_i → g = Lie(G)`.
- Line 148: Connection 1-form `A^(i)_μ(c) = U_i^{-1}(c) ∂_μ U_i(c)` (Maurer-Cartan-type).
- Line 154: Field strength (gauge curvature) `F^(i)_{μν} = ∂_μ A^(i)_ν - ∂_ν A^(i)_μ + [A^(i)_μ, A^(i)_ν]`.
- Line 160: Inter-agent gauge transformation `Ω_ij(c) = exp(φ_i(c)) exp(-φ_j(c)) ∈ G`.
- Line 166: Transport action `q_j(c) → Ω_ij(c) · q_j(c) := ρ(Ω_ij(c)) q_j(c)`.
- Line 172: Overlap connection law `A^(i)_μ = Ω_ij A^(j)_μ Ω_ij^{-1} + Ω_ij ∂_μ Ω_ij^{-1}`.

## What §General Mathematical Framework does NOT contain

**Note: these are concrete absences from lines 46–177. The supplementary's later sections may contain them; the load-bearing question is whether the §General Mathematical Framework chapter, AS WRITTEN, is "complete."**

### Information geometry primitives — absent

- No explicit definition of the Fisher-Rao metric `g_B(∂_a, ∂_b) = E[(∂_a log ρ)(∂_b log ρ)]`. The Fisher-Rao metric is the foundational Riemannian structure on probability spaces and is unique up to scale [Čencov 1982].
- No statement of the Gaussian KL closed form `KL(N(μ_1, Σ_1) ‖ N(μ_2, Σ_2)) = (1/2)[log(|Σ_2|/|Σ_1|) + tr(Σ_2^{-1} Σ_1) + (μ_2 - μ_1)^⊤ Σ_2^{-1} (μ_2 - μ_1) - K]`.
- No definition of the natural gradient `∇̃_q f = g_B^{-1} ∇_q f` — required for the framework's natural-gradient descent claim.
- No KL invariance theorem under GL(K) push-forward (this is Theorem 1 of the main paper at lines 520–556, NOT in the supplementary).

### Variational EM machinery — absent

- No statement of the variational free energy functional `F = KL(q ‖ p) - E_q[log p(o|s)]`.
- No E-step / M-step decomposition.
- No mention of softmax-β attention weights, mixture-of-sources model, or any of the §3.4 machinery from the main paper.
- No mention of priors `p_i` separate from the model fiber `B_p` — but line 76 introduces `p_i(c) = σ^i_p(c) ∈ B_p(c)`, which puts `p_i` on the MODEL fiber, not the belief fiber. This is inconsistent with the main paper §3.1 where `p_i(k_i) = N(μ_0,i^(q), Σ_0,i^(q))` lives in the BELIEF fiber `E_q` alongside `q_i`.

### Cross-scale shadow construction — absent

- The Participatory paper at lines 536–548 derives `p_i^(s) = Ω_{i,I}[q_I^(s+1)]` — priors are cross-scale shadows of meta-agent posteriors, not independent primitives. The supplementary's `p_i` is introduced as a section without derivation.

### Two roles for the gauge frame — absent

- Participatory at lines 557–565 develops Role A (transport, gauge-redundant) vs Role B (state, gauge-covariant) with citations [DonnellyFreidel2016, BartlettRudolph2007, Rovelli1996]. The supplementary has no analogous discussion; line 142's `φ_i` could be read either way without clarification.

### Local trivialization caveats — absent

- Participatory at lines 577–581 develops the Čech-cocycle treatment with `exp` map locally-bijective caveat and the polar/Cartan decomposition for non-compact `GL^+(K)`. The supplementary at line 142 introduces `φ_i: U_i → g` without these caveats.

## Notation inconsistency between supplementary and main paper

- **Supplementary line 55**: `B_p` = "model/prior fiber"; `p_i` at line 76 is a section of `B_p`.
- **Main paper §3.1 line 583**: `m_i ∈ R^{K_p}` is the "model latent in E_p"; `k_i ∈ R^{K_q}` is the "belief latent in E_q".
- **Main paper §3.1 line 602**: `p_i(k_i) = N(k_i; μ_0,i^(q), Σ_0,i^(q))` — the BELIEF prior, with mean and covariance superscripted `(q)`, indicating it lives in the BELIEF fiber `E_q`, NOT in the model fiber `E_p`.
- **Conclusion**: the supplementary's `B_p (model/prior fiber)` is ambiguous. If "prior" means "belief prior `p_i`", then `p_i` should live in `B_q` (belief fiber) per the main paper, not `B_p`. If "prior" means "model hyper-prior `r_i`", then `B_p` is the model fiber alone. The supplementary line 55 conflates these.

## Supplementary's later content (where the "absent" material appears)

- §B Covariance Dynamics and Equilibrium Analysis (lines 178–387): contains the Gaussian KL under `GL(K)` gauge transport at §B.1, fixed-point equations, gradient flow dynamics.
- §C Gauge Frame Gradients (lines 388–610): differential of matrix exponential, KL gradient through transport.
- §D Variational Gradient Descent: Implementation and Numerical Methods (lines 611–665): natural gradient descent on Gaussian manifold (line 614), manifold retraction for covariance matrices, gauge frame dynamics on `GL(K)`.

If sub-claims δ and ε are interpreted as "the supplementary as a whole establishes these" (not "§General Mathematical Framework establishes these"), the gaps reduce to forward references that don't exist in §General Mathematical Framework but the content exists later in the supplementary.

## Canon excerpts — external standards

- **Fisher-Rao metric uniqueness [Čencov 1982].** The Fisher-Rao metric is the unique (up to scale) Riemannian metric on probability spaces invariant under sufficient statistics. The Participatory paper at line 510 cites this; the supplementary does not.
- **Principal bundles and connections [Nakahara 2003 §10; Frankel 2011]** — both papers cite these for the bundle scaffold. Standard references.
- **Bundle triviality theorem [Nakahara 2003 §9.5; Steenrod 1951].** A principal G-bundle is trivial (admits a global section) iff transition functions are coboundaries: `g_ij = g_i g_j^{-1}` for some `g_i: U_i → G`. The supplementary's `Ω_ij = exp(φ_i) exp(-φ_j)` directly satisfies this coboundary condition with `g_i = exp(φ_i)`, hence the bundle is trivial — consistent with the supplementary's line 53 disclosure.
- **Variational EM [Wainwright-Jordan 2008 §3; Bishop 2006 §10].** The E-step / M-step decomposition is the canonical iteration for variational inference. The supplementary defines no E-step / M-step in §General Mathematical Framework; the EM-mode taxonomy lives in the main paper (per CLAUDE.md) and the implementation lives in `transformer/vfe/e_step.py`.

## Code references (incidental, for sub-claim α/γ verification)

- `transformer/vfe/non_flat.py` — implements edge-relaxed `Ω_ij = exp(φ_i) exp(δ_ij·G) exp(-φ_j)` non-flat transport per the supplementary line 53 / main paper Eq. eq:edge_relaxed_omega_glk forward reference.
- `transformer/vfe/connection.py` — bilinear connection field per the non-flat extension.

## What this evidence does NOT settle

1. **Whether sub-claims δ and ε refer to "§General Mathematical Framework specifically" or "the supplementary as a whole".** The judge will need to decide what the user means by "the entire section." Reading 1 (strict): §General Mathematical Framework is the four subsections at lines 46–177; sub-claims δ, ε require Fisher-Rao, KL closed form, natural gradient, VFE functional to appear within those lines. Reading 2 (charitable): "the entire section" means the supplementary chapter §A, and sub-claims δ, ε are deferred to later supplementary sections (§B, §C, §D) with adequate forward references.

2. **Whether the `B_p (model/prior fiber)` notation at supplementary line 55 is a genuine inconsistency with the main paper §3.1 or a translation issue.** If the supplementary's `B_p` is intended to host the hyper-prior `r_i` and the model `s_i` (which are at the same place mathematically), and "prior" at line 55 means the hyper-prior `r_i` rather than the belief prior `p_i`, then the notation is internally consistent — but inconsistent with the line 76 statement `p_i(c) = σ^i_p(c) ∈ B_p(c)` which uses `p_i` for what reads as the belief prior in the main paper.

3. **Whether the absence of the cross-scale shadow construction in the supplementary is a load-bearing gap.** The Participatory paper makes the cross-scale shadow construction load-bearing for its theoretical exposition. The supplementary and main paper of the gauge-transformer treat priors `p_i` as primitive boundary data (the PriorBank per CLAUDE.md). Whether the cross-scale shadow construction is required for the supplementary's "completeness" claim depends on what role priors play in the main paper's derivations.

4. **Whether the absence of the two-role gauge frame discussion (Role A vs Role B) is a load-bearing gap.** The supplementary uses `φ_i` consistently in the transport role (Role A) throughout, without invoking it as a state variable (Role B) the way the Participatory paper does. If the main paper of the gauge-transformer also uses `φ_i` only in Role A, the absence of Role B discussion is justified by scope.

5. **Whether the §General Mathematical Framework title commits the chapter to comprehensive coverage.** The user's claim asserts the section is "complete and mathematically/theoretically pure." "Complete" can mean (a) complete relative to its stated scope or (b) complete as a comprehensive foundational chapter. The title "General Mathematical Framework" suggests (b); the four-subsection structure suggests (a) with the rest deferred.
