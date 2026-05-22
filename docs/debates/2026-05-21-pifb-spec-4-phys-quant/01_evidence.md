# Evidence Pack — pifb-spec-4-phys-quant

## Manuscript references (read the actual TeX, not this summary)

Source file: `C:/Users/chris and christine/Desktop/V13_Gauge_Transformer/Attention/Participatory_it_from_bit.tex`

### §Physical Quantities and Dimensionless Constants (lines 3051-3122)

- **3053-3060 (`sec:phenomenological_interpretation` — Information as Primal)** — Central commitment: information (VFE, KL in nats) is dimensionally fundamental. Physical dimensions (kg, m, s) are emergent phenomenological labels constructed via consensus formation. "Like color terms labeling electromagnetic frequencies." **3060 — limitations paragraph:** has NOT established dimensional conversion bits→kg, derived numerical values of constants, or specified how to measure "beliefs" of physical systems. May be unfalsifiable.
- **3062-3071 (`Structural Parallels to Physical Concepts`)** —
  - **Time** (3066): subjective `tau_i` advances through information updates, `Delta I_i = KL(q_i^new || q_i^old)`. Different update rates → different subjective flows, "structurally analogous to proper time in relativity"; "quantitative correspondence with relativistic time dilation and the Lorentz transformation is not established here and the parallel remains qualitative."
  - **Mass** (3068-3070): Fisher information matrix `M_ij[q] = E_q[(∂_i log q)(∂_j log q)]` in natural gradient `dtheta/dt = -M^-1 ∇F`. Large Fisher → small M^-1 → slow updates → "confident beliefs resist change". "Formally analogous to inertial mass." Identification `tr(M)` with effective mass. **3070 — Critical disclaimer paragraph:**
    > "We do not claim a computational test of $M_{\mathrm{eff}} \propto \Sigma_p^{-1}$ in which $\omega^2$ and $M_{\mathrm{eff}}$ are measured as operationally independent quantities; the within-framework reading above identifies the Fisher mass with the Hessian by construction on this slice (Section~\ref{sec:mass}, Section~\ref{sec:velocity_quadratic}), so the scaling is a definitional consequence of that identification rather than an empirical dispersion law, and no operationally independent measurement is reported in this manuscript."
    
    **This is the load-bearing disavowal.** Project memory `project_pifb_mass_todo_plan.md` says all prior PIFB R²/ω data is placeholder pending the 4-step plan; this manuscript paragraph is the in-text honest acknowledgement.
- **3072-3074 (`Energy and free energy`)** — VFE decreases monotonically along natural-gradient flow. "Mathematically analogous to energy dissipation in physical systems." Disclaimer: "we have not established that F has energy dimensions, derived thermodynamic relations, or connected to measured energies of physical systems."
- **3076-3078 (`Action principle structure`)** — Variational principle: agents extremize F. Mathematically analogous to Hamilton's principle. Euler-Lagrange equations from `delta F = 0` are the natural-gradient flow equations: Fisher-Rao natural gradient on (mu, Sigma), Lie-group natural gradient on G via Eq. `gauge_group_retraction`. Disclaimer: "we have not derived Lagrangians for specific physical systems, recovered Newton's laws quantitatively, connected to quantum path integrals beyond formal similarity, or predicted measurable physical quantities."
- **3080-3084 (`Targeted Research Program: Dimensionless Constants`)** — One testable prediction: dimensionless ratios (alpha ≈ 1/137, m_e/m_p ≈ 1/1836) should be derivable from pure information geometry. **3084 — Symmetric success-failure criterion:** "Success would substantively support the phenomenological interpretation. Failure of the derivation under a fully developed information-geometric construction --- one in which the relevant statistical manifolds, comparison structure, and Cencov scale fixings are all specified --- would constitute substantive evidence against the phenomenological interpretation. Failure under an incomplete construction is non-diagnostic and does not bear on the interpretation; the burden is on the program to develop the construction to the point where the test is symmetric, not to indefinitely defer the test by appeal to incompleteness." **This is the falsifiability commitment.**
- **3086-3092 (`The Kantian Philosophical Framework`)** — Phenomena/noumena = physical/information-geometric. **3092 — Important softening:** "The framework recasts this in the structural-realism family — Cassirer's neo-Kantian variant, Worrall's epistemic structural realism~\cite{Worrall1989}, and the ontic structural realism of Ladyman \& Ross~\cite{Ladyman2007} (remaining agnostic among these variants) — rather than adopting Kant's stronger constitutive-a-priori commitment, which Section~\ref{sec:scope_limitations} explicitly disavows."
- **3094-3102 (`Philosophical Status and Research Program`)** — Phenomenological interpretation is "a metaphysical postulate, not a derived result". Advantages: resolves dimensional paradoxes, aligns with Wheeler's "it from bit", internally consistent. Research program: (1) derive dimensionless constants, (2) resolve signature problem, (3) emergence of matter fields, (4) rigorous dimensional analysis. **3102 — Acknowledged risk:** "the framework may ultimately function as philosophical interpretation rather than scientific theory. If it makes no falsifiable predictions beyond qualitative structural analogies, it occupies a different epistemic category than physics proper."
- **3104-3121 (Table `tab:physical_correspondences`)** — Status column: Time → "Formal analogy", Mass → "Speculative", Energy → "Dimensional gap", Entropy → "Structural similarity", Action → "Mathematical parallel", hbar → "Undefined", c → "Unmeasured".

### §Summary and Implications (lines 3123-3157)

- **3125 — Concession paragraph.** "The Lorentzian signature is not derived, the dimensionality count is not derived, and matter fields and interactions are absent. The construction is a proof of concept that agent-frame-dependent induced geometry can carry spacetime-like features within a fixed gauge fixing, not a derivation of physical spacetime."
- **3127-3145 (Boxed: Belief Tangent-Space Decomposition)** —
  - **What is derived:** Gram-Schmidt decomposition `T_q B = R qdot + V_obs + V_subthresh + V_internal` under assumed eigenvalue hierarchy.
  - **What is postulated:** (i) eigenvalue hierarchy + three large eigenvalues in V_obs; (ii) Lorentzian (-,+,+,+) for `R qdot + V_obs` under SO(3) restriction as a separate postulate; (iii) under GL(K,C) extension, imaginary frame component + real-part projection; (iv) under 4D extension, the three spatial generators T_x, T_y, T_z in sl(K,R) chosen mutually trace-orthogonal `tr(T_a T_b) = 0`. "None of (i)–(iv) is derived from variational free-energy minimization in the present formulation."
  - **An interpretive reading:** "We do not claim to have derived physical spacetime, the Lorentz signature, or the dimensionality count from the framework; the construction shows that the framework is compatible with such a reading under the postulates listed above."
- **3147-3149 (`Pan-Agentic Ontology and Plural Time`)** — Network of coupled agents as substrate. Time plural: each agent's Fisher arc length is its own clock. "Relativistic proper time is not reproduced under the present compact-group construction — Fisher arc length is Riemannian and positive-definite while proper time is Lorentzian, and the two run in opposite directions with motion."
- **3151-3157 (`Sustained Non-Equilibrium under the Threshold-Detector Dynamics`)** — Participatory loop via cross-scale feedback. Threshold-detector meta-agent formation at scale s+1. Wheeler's "self-excited circuit" structurally analogous. **3157 — Scope qualifier:** "we make no claim about thermodynamic perpetual motion or about cosmological complexity growth, both of which are outside the scope of the present construction."

## External canon to verify

Canon files at `C:/Users/chris and christine/Desktop/V13_Gauge_Transformer/.claude/agents/vfe-knowledge/`.

- **[Amari1998]** Natural gradient `~∇F = G^-1 ∇F` where G is Fisher information. Verify 3068 invocation is canonical.
- **[Friston2010]** FEP precision: high posterior precision → resistant to update, low precision → readily updated. **Verify 3068 mass-precision identification is canonically supported by Friston.**
- **[FristonEtAl2017]** Process theory; precision as inverse variance.
- **[AmariNagaoka2000]** Cencov uniqueness theorem — Fisher is the unique invariant metric.
- **[Cencov1972]** Original Cencov.
- **[Worrall1989]** Worrall, J. (1989). "Structural realism: The best of both worlds?" *Dialectica* 43(1-2): 99-124. **Epistemic structural realism — verify 3092 citation is correctly attributed.**
- **[LadymanRoss2007]** Ladyman, J. & Ross, D. (2007). *Every Thing Must Go: Metaphysics Naturalized*. Oxford. **Ontic structural realism — verify 3092 citation.**
- **[Wheeler1990]** Wheeler primary "It from Bit" — already cited extensively. Verify 3098 invocation.
- **Cencov scale fixing** — used at 3084 in the dimensionless-constants program. Verify Cencov 1972 supports the specific "scale fixing" usage; this is non-trivial — Cencov's theorem is about invariance, not scale fixing per se.

## Cross-reference checks the agents should perform

1. **Boxed postulate list (3141) vs actual constructions:**
   - Postulate (i) eigenvalue hierarchy — appears at §`sec:observable_sectors` 2972-3013. Match.
   - Postulate (ii) Lorentzian signature under SO(3) — comes from §`sec:signature_resolution`. Match.
   - Postulate (iii) imaginary frame + real-part — from `sec:worked_signature` 2820, 2840-2846. Match.
   - Postulate (iv) spatial generators trace-orthogonal — does the worked example actually USE this? The 2D worked example uses only one generator T = diag(1,-1) along both temporal and spatial directions, not three distinct spatial generators. Check whether (iv) is an extension postulate (4D) or a current postulate (2D). Read 2854: "For four-dimensional base manifolds with one imaginary and three real gauge components, the local frame group at each non-degenerate point is the full proper orthochronous Lorentz group SO+(1,3); the question of which global metrics with this local frame group can be obtained, and whether the resulting geometry is Minkowski or merely Lorentzian-signatured, is open." So (iv) is an extension postulate — verify the box's phrasing matches.
2. **Mass-Fisher claim consistency:** is the disavowal at 3070 contradicted by language at §sec:mass (line ~1901, *outside* the speculative section but cross-referenced)? If §sec:mass makes the empirical claim that 3070 disavows, that is internal inconsistency.
3. **Time = Fisher arc length consistency:** 3149 says Fisher arc length is Riemannian positive-definite, NOT proper time. Earlier §`sec:fisher_arc_length` (line ~2609) makes the same distinction. Match.
4. **Action principle:** §3076 invokes the Lie-group natural gradient via `gauge_group_retraction` (Eq. at 2641). Verify the Euler-Lagrange / variational invocation is correct: are the natural-gradient flow equations of motion actually the Euler-Lagrange equations of `delta F = 0`, or are they preconditioned gradient descent? Amari 1998 establishes natural gradient = E-L for the appropriate Lagrangian on the statistical manifold; verify.

## What this evidence does NOT settle

1. Whether `tr(M)` is the canonical Fisher-mass identification or whether other contractions (eigenvalues, determinants) would be more defensible.
2. Whether the symmetric success-failure criterion at 3084 truly imposes falsifiability, or whether "fully developed construction" is a movable goalpost.
3. Whether structural realism (Worrall, Ladyman & Ross) actually applies here, or whether the framework is closer to Kant despite the disavowal.
4. Whether the Wheeler "self-excited circuit" analogy at 3155 is correctly attributed to Wheeler 1983 or whether the citation refers to a different Wheeler paper.
5. Whether the manuscript's Cencov-scale-fixing usage at 3084 is a coherent application of Cencov's uniqueness theorem.

## Project context the agents should know

Memory note `project_pifb_mass_todo_plan.md`: "4-step plan for operationally-independent omega^2 vs Sigma_p^-1 test; all prior PIFB R^2/omega data is placeholder pending these experiments". The manuscript text at 3070 correctly disavows the empirical mass-dispersion claim. The debate tests whether the disavowal is *complete and not contradicted elsewhere in the section*, NOT whether the empirical experiment has been performed.
