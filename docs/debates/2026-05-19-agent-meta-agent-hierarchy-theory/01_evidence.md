# Evidence Pack — agent-meta-agent-hierarchy-theory

Neutral fact pack. Both teams use this as the shared starting point.

## Manuscript references (Attention/Participatory_it_from_bit.tex)

### Sub-claim A — Agent definition

- `Attention/Participatory_it_from_bit.tex:180` — `\section{Theory}` opens.
- `Attention/Participatory_it_from_bit.tex:186` — Symbol conventions: $q_i$ belief on state fiber, $p_i$ prior on same fiber (cross-scale shadow), $s_i$ generative model on model fiber, $r_i$ hyper-prior on model fiber.
- `Attention/Participatory_it_from_bit.tex:483-489` — Definition of base manifold $\mathcal{C}$ as smooth $n$-dimensional manifold representing the "noumenal substrate."
- `Attention/Participatory_it_from_bit.tex:495-534` — Definition of statistical manifolds $\mathcal{B}_{\mathrm{state}}, \mathcal{B}_{\mathrm{model}}$ with Fisher-Rao metric (Eq. line 502) and Gaussian-family realization (line 514).
- `Attention/Participatory_it_from_bit.tex:539-546` — Cross-scale shadow construction (Eq.~\ref{eq:cross_scale_shadow}): $p_i^{(s)}(c) = \Omega_{i,I}[q_I^{(s+1)}](c)$, $r_i^{(s)}(c) = \tilde\Omega_{i,I}[s_I^{(s+1)}](c)$. The manuscript explicitly states "This is a structural commitment of the framework rather than a theorem of standard hierarchical variational inference: in the standard scheme [Friston2017, ParrPezzulo2022] the level-$\ell$ prior is derived from a generative-model conditional $p(s_\ell | s_{\ell+1})$, not posited as a transported posterior."
- `Attention/Participatory_it_from_bit.tex:550-587` — Principal bundle $\pi: \mathcal{N} \to \mathcal{C}$ with structure group $G$ (typically $\mathrm{SO}(N)$, natural $\mathrm{GL}(K,\mathbb{R})$). Local-trivialization caveat at line 577-581 (no global section for non-trivial bundles; Čech cocycle is the actual mathematical object). Exponential parametrization caveat at line 580: $\exp$ surjective for connected compact $G$, polar/Cartan decomposition needed for $\mathrm{GL}^+(K)$.
- `Attention/Participatory_it_from_bit.tex:557-565` — "Two roles for the gauge frame" — Role A (transport, gauge-redundant under global $U_i \mapsto U_i g$) and Role B (state, cognitive variable contributing to ontology). Manuscript cites `[DonnellyFreidel2016]` (edge modes), `[BartlettRudolphSpekkens2007, Vanrietvelde2020]` (quantum reference frames), `[Rovelli1996]` (relational interpretation) as antecedents for the dual role.
- `Attention/Participatory_it_from_bit.tex:589-610` — Associated bundles $\mathcal{E}_\mathrm{state} = \mathcal{N} \times_{\rho_\mathrm{state}} \mathcal{B}_\mathrm{state}$ via representation $\rho_\mathrm{state}: G \to \mathrm{Aut}(\mathcal{B}_\mathrm{state})$. Gaussian congruence representation $\rho(g)\mathcal{N}(\mu,\Sigma) = \mathcal{N}(g\mu, g\Sigma g^\top)$ at line 608.
- `Attention/Participatory_it_from_bit.tex:617-631` — **Definition of Agent** (Definition box). $\mathcal{A}^i$ at scale $s$ with support $\mathcal{U}_i \subseteq \mathcal{C}$ consists of two primitive sections $q_i, s_i$, two derived sections $p_i, r_i$ (from cross-scale shadow), and gauge frame field $\phi_i: \mathcal{U}_i \to \mathfrak{g}$. The variational hierarchy $r_i \to s_i \to p_i \to q_i \to$ observations is stated as realized cross-scale.

### Sub-claim B — Meta-agent definition

- `Attention/Participatory_it_from_bit.tex:650-666` — Multi-agent system definition. Domains $\mathcal{U}_i$ overlap; intersections enable belief comparison through gauge transport.
- `Attention/Participatory_it_from_bit.tex:668-682` — Perfect Consensus definition (Definition box at line 672): $\Omega_{ij}[q_j](c) = q_i(c)$ on the shared region. Manuscript explicitly disclaims this as the criterion for meta-agent formation (line 682): "Perfect consensus is an idealization that is not the criterion for meta-agent formation: imposed pointwise it is the epistemic-death limit."
- `Attention/Participatory_it_from_bit.tex:684-691` — **Definition of Meta-Agent** (Definition box at line 684). Cluster $I = \{i,j,\ldots\}$ forms a meta-agent at scale $(s+1)$ when (1) coherence $C_\mathrm{belief} \cdot C_\mathrm{model} > \Gamma_\min$, (2) non-empty multi-child support $U_I = \mathrm{ConnComp}(\{c : \sum_i \chi_i(c) \ge 2\}) \neq \varnothing$, (3) minimum cluster size $|I| \ge N_\min$ (typically 2). Constructed via gauge-covariant averaging.
- `Attention/Participatory_it_from_bit.tex:697-718` — Culture as coarse-grained slow-channel identity. Graph-weighted internal-coherence criterion (Eq.~\ref{eq:culture_internal_coherence}, line 704) and closure criterion (Eq.~\ref{eq:culture_closure}, line 712).
- `Attention/Participatory_it_from_bit.tex:720-750` — Epistemic Death (Definition at line 725); "heat death" of informational universe; gauge-invariant under global diagonal $U_i \mapsto U_i g$.
- `Attention/Participatory_it_from_bit.tex:2056-2069` — Variational principle for meta-agent formation (Eq.~\ref{eq:meta_agent_FE_criterion}, line 2063): $\mathcal{F}^*[q_I,\ldots] + C(I) < \mathcal{F}^*[\{q_i,\ldots\}]$. Quadratic savings $|I|(|I|-1)\varepsilon$ vs. sub-linear MDL overhead $C(I)$.
- `Attention/Participatory_it_from_bit.tex:2071-2078` — Information-bottleneck refinement (Eq.~\ref{eq:meta_agent_IB}, line 2073) flagged as "research direction"; closed-form Gaussian-IB cited `[chechik2005information]`.
- `Attention/Participatory_it_from_bit.tex:2080-2093` — Gauge-covariant variational barycenter (Eq.~\ref{eq:meta_agent_barycenter}, line 2081-2083): $q_I^* = \arg\min_{q_I} \sum_i w_i^I \mathrm{KL}(\Omega_{Ii}q_i \| q_I) + \lambda \mathrm{KL}(q_I \| p_I)$. Gaussian closed form for $\mu_I^*, \Sigma_I^*$ at lines 2087-2091.
- `Attention/Participatory_it_from_bit.tex:2095-2100` — Karcher/Fréchet mean for parent frame (Eq.~\ref{eq:meta_agent_frame_barycenter}, line 2097). Existence on convex normal balls of radius $< \pi/2$ for compact $G$ (`[Karcher1977]` is the canonical citation, not in manuscript — would be a gap). Non-compact $\mathrm{GL}^+(K)$ caveat explicitly registered.
- `Attention/Participatory_it_from_bit.tex:2102-2114` — Threshold-based detector. Bounded exponential coherence proxies $C_q, C_s \in [0,1]$, presence $P \in [0,1]$, consensus score $\Gamma = P C_q C_s$. The relation between detector and variational criterion is "heuristic and partial rather than a derivation."
- `Attention/Participatory_it_from_bit.tex:2116-2127` — Working barycenter formulae (Eqs.~\ref{eq:meta_agent_mu_impl}, \ref{eq:meta_agent_sigma_impl}, lines 2121-2125) drop the dispersion term as $\mathcal{O}(\varepsilon)$. Manuscript states: "The implementation formula~\eqref{eq:meta_agent_sigma_impl} corresponds to neither [forward-KL barycenter nor moment-matched mixture nor precision-weighted product]: it is an implementation heuristic that agrees with the variational forward-KL barycenter at $\mathcal{O}(\varepsilon)$ when constituent post-transport means coincide."

### Sub-claim C — Hierarchical emergence and RG

- `Attention/Participatory_it_from_bit.tex:752-763` — Hierarchical structure: scale-0 (individuals) → scale-1 (groups) → ... → $s_\max$. Bidirectional flow (consensus up, prior propagation down). Manuscript invokes "Wheeler's participatory loop."
- `Attention/Participatory_it_from_bit.tex:2044-2046` — Section "The Participatory Universe" opens (`sec:participatory`). Manuscript states: "We demonstrate that our gauge-theoretic framework supports structural parallels to this vision" — *parallels*, not identification.
- `Attention/Participatory_it_from_bit.tex:2050-2054` — Scale tower; hard cap $s_\max = 25$ as "computational constraint not theoretical principle."
- `Attention/Participatory_it_from_bit.tex:2105` — Manuscript states: "The threshold-based detector developed in the remainder of this subsubsection is a candidate-selection surrogate for the closure-ansatz conditions of that construction rather than of the exact pushforward step."
- `Attention/Participatory_it_from_bit.tex:2135-2153` — Bottom-up emergence and RG (`sec:bottom_up_emergence` block). Manuscript states: "the present construction is RG-inspired rather than a literal RG analysis: we do not exhibit a $\beta$-function, locate fixed points, or demonstrate scale invariance beyond the parametric form."
- `Attention/Participatory_it_from_bit.tex:2143-2151` — Scale-invariant free-energy functional form $S^{(s+1)}$ with renormalized couplings $\beta_{IJ}^{(s+1)} = f_\beta(\{\beta_{ij}^{(s)}\})$. The functions $f_\beta, f_\gamma$ are *not derived* — line 2139 says they "depend on how informational coupling aggregates across scales" without specifying the aggregation rule.
- `Attention/Participatory_it_from_bit.tex:2155-2164` — Cross-scale information flow $\mathcal{I}_{s \to s+1}$ as transported KL (Eq.~\ref{eq:cross_scale_information_flow}, line 2159).
- `Attention/Participatory_it_from_bit.tex:2166-2170` — "Emergent properties at higher scales": longer coherence lengths, smaller covariances via information pooling, "emergent coordination patterns not reducible to constituent actions, arising from the nonlinear coupling through the free energy functional."
- `Attention/Participatory_it_from_bit.tex:2172-2181` — Emergent timescale hierarchy across scales; analogies to thermodynamics, fluid dynamics, social systems.
- `Attention/Participatory_it_from_bit.tex:2183-2226` — Top-down participation; Ouroboros tower with exponentially decaying ancestral weights $\lambda_k = \lambda_0 \rho^k$; self-referential closure at the top scale.
- `Attention/Participatory_it_from_bit.tex:2217` — Implementation Note: "Whether the released simulator code realizes the full transport $\Omega_{i,I}$ or a frame-trivial substitute is not independently verified in this manuscript."

## Canon excerpts (from `.claude/agents/vfe-knowledge/external_canon_inference.md`)

- **§1 — Standard variational free energy form** (Friston2010, ParrPezzuloFriston2022 Ch. 2). Three equivalent forms: (1) expected log-ratio, (2) KL-to-true-posterior + log-evidence, (3) accuracy + complexity.
- **§1 — "What is *not* in this standard form":** "Multi-agent coupling terms of the form $\sum_{ij} \beta_{ij} \mathrm{KL}(q_i \| \Omega_{ij}q_j)$. These are user-introduced. The standard FEP is single-agent (or hierarchical with a single ancestral generative model). Multi-agent generalizations exist in the FEP literature (e.g., [Friston2017Graphical] for graphical brain, [Ramstead2020] for variational ecology), but the *specific* coupling form via gauge transport $\Omega_{ij}$ is a user construction, not the field standard. The agent should label this as a novel construction requiring its own justification."
- **§3 — Hierarchical / nested formulations** (Friston2017Graphical, ParrPezzuloFriston2022 Ch. 9). Recognition distribution factorizes mean-field across levels. Cross-level couplings come from generative model $p(s_\ell | s_{\ell+1})$. Canon explicitly flags: "The user's multi-layer cascade (where the previous layer's posterior $\mu_q$ becomes the next layer's prior $\mu_p$) is **not standard mean-field across the hierarchy**. It is a deterministic point-passing scheme that loses the variational uncertainty about $s_\ell$ when passed to $\ell+1$."
- **§4 — KL direction (mode-seeking vs mean-seeking).** Forward KL $\mathrm{KL}(q\|p)$ is zero-forcing/mode-seeking; reverse KL is mean-seeking. The framework uses forward KL with the transported neighbor in the second slot, i.e., the variational direction.
- **§7 — Markov blankets, NESS, the "physics-y" formulation.** "Modern FEP statements ground free energy minimization in non-equilibrium steady-state (NESS) systems with Markov blankets. ... It is *contested* in the literature — e.g., debates around whether the Markov-blanket / NESS argument is tautological or has empirical content (van Es 2021, Aguilera et al. 2022, etc.). When a manuscript invokes this framing, the agent should not treat NESS-FEP as uncontested."
- **§10 — Pitfalls:** Single-agent FEP extended to multi-agent is *novel* construction; hierarchical mean-field vs. point-passing must be distinguished.

## Relevant primary sources both teams should cite

**Differential geometry / fiber bundles:**
- `[Nakahara2003 §10.3]` — Principal/associated bundles, local trivializations, Čech cocycle condition for transition functions.
- `[KobayashiNomizu1963]` — *Foundations of Differential Geometry*, canonical reference for connections on principal bundles.

**Information geometry:**
- `[AmariNagaoka2000]` — *Methods of Information Geometry*. Fisher-Rao metric; KL as the canonical divergence; Gaussian KL closed form.
- `[Karcher1977]` — Existence/uniqueness of the Riemannian center of mass (Karcher/Fréchet mean) on convex normal balls of radius $< \pi/2$. This is the actual citation needed for the manuscript's Eq.~\ref{eq:meta_agent_frame_barycenter}.
- `[Cuturi2014, Agueh2011]` — Wasserstein barycenters; the forward-KL barycenter is a parallel construction in information geometry.

**FEP / active inference:**
- `[Friston2010]` — "The free-energy principle: a unified brain theory?" Single-agent VFE.
- `[Friston2017Graphical]` — "Active inference: a process theory" — hierarchical FEP with generative-model conditional $p(s_\ell | s_{\ell+1})$.
- `[ParrPezzuloFriston2022]` — *Active Inference* textbook, Ch. 9 hierarchical models.
- `[Ramstead2020]` — Variational ecology / multi-agent FEP extensions.
- `[Kirchhoff2018, FristonLevin2020]` — Markov blankets at multiple scales.

**Gauge theory / quantum reference frames:**
- `[DonnellyFreidel2016]` — "Local subsystems in gauge theory and gravity" — edge modes.
- `[BartlettRudolphSpekkens2007]` — "Reference frames, superselection rules, and quantum information."
- `[Rovelli1996]` — Relational interpretation of quantum mechanics.

**Renormalization group:**
- `[Wilson1971, Wilson1974]` — RG and critical phenomena.
- `[Cardy1996]` — *Scaling and Renormalization in Statistical Physics*.

**Information bottleneck:**
- `[Tishby1999]` — Original IB paper.
- `[Chechik2005]` — Gaussian information bottleneck.

## What this evidence does NOT settle

- Whether the cross-scale shadow construction $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ admits a *derivation* (rather than a posit) from any standard hierarchical scheme. The manuscript explicitly labels it as a structural commitment.
- Whether the threshold-based detector $\Gamma = P C_q C_s > \Gamma_\min$ produces the *same hierarchical organization* as the variational FE-improvement criterion in the high-coherence regime. The manuscript explicitly labels this as open.
- Whether the "scale-invariant functional form" of $S^{(s+1)}$ is closed under the actual coarse-graining map, i.e., whether the renormalized couplings $\beta_{IJ}^{(s+1)} = f_\beta(\{\beta_{ij}^{(s)}\})$ can be exhibited concretely with $f_\beta$ specified.
- Whether the Karcher mean of frames is well-defined when constituents straddle the cut locus of the gauge group (manuscript invokes the convex-normal-ball condition implicitly without specifying when it is satisfied).
- Whether the manuscript's identification of $\phi_i$ as a *cognitive reference frame* (Lahav-Neemeh correspondence, Section `sec:cognitive_reference_frames`) is mathematically licensed or merely interpretive — the manuscript itself flags a known gap ("gauge invariance places no constraint on which gauge frame produces which phenomenal quale").
- Whether the Ouroboros-tower exponentially-decaying ancestral-weight scheme (Section "Top-Down Participation") corresponds to any standard hierarchical Bayesian object or is a free-form posit.

These are the cracks the red team should probe.
