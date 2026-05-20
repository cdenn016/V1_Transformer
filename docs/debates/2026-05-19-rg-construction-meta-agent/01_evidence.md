# Evidence Pack — rg-construction-meta-agent

Neutral fact pack. Both teams use this as the shared starting point.

## Manuscript references (Attention/Participatory_it_from_bit.tex)

### Sub-claim A — Exact pushforward RG and gauge covariance

- `Attention/Participatory_it_from_bit.tex:4351-4352` — `\section{Renormalization-Group Construction for Meta-Agent Formation}`, `\label{sec:meta_agent_rg}`.
- `Attention/Participatory_it_from_bit.tex:4354` — Opening paragraph distinguishes two claims: "The pushforward step is exact and structural ... The closure step is approximate and conditional."
- `Attention/Participatory_it_from_bit.tex:4356-4368` — Paragraph "Exact pushforward RG."
  - Microscopic state $X_s = \{q_i, p_i, U_i, \chi_i\}_{i\in I_s}$ with Gibbs measure $d\mathbb{P}_s = Z_s^{-1}\exp[-\mathcal{F}_s/\tau]d\nu_s$.
  - Coarse-graining map $\mathcal{R}_s: X_s \mapsto \{Y_I\}$ via the M-projection barycenter and Karcher frame.
  - Eq.~\ref{eq:rg_exact_free_energy} (line 4358): $\exp[-\mathcal{F}_{s+1}(Y)/\tau] = d\widetilde{\rho}_{s+1}/d\nu_{s+1}(Y)$ with $\widetilde{\rho}_{s+1} = (\mathcal{R}_s)_*(e^{-\mathcal{F}_s/\tau}d\nu_s)$.
  - Two gauge actions distinguished: (a) global diagonal right-translation leaves $\Omega_{ij}$ invariant; (b) base-local diagonal action $U_i(c) \mapsto h(c)U_i(c)$ conjugates $\Omega_{ij}$ tensorially.
  - Noncompact $\mathrm{GL}^+(K)$ caveat: "a gauge slice or Radon-Nikodym correction is required because the Lebesgue density transforms by $|\det h|^{\dim}$ and the action is not unimodular."
- `Attention/Participatory_it_from_bit.tex:4458-4465` — **Theorem 1 (Partition function and observable preservation)** with proof. Statement: $Z_{s+1} = Z_s$ and $\mathbb{E}_{s+1}[A(Y)] = \mathbb{E}_s[A(\mathcal{R}_s(X))]$.
- `Attention/Participatory_it_from_bit.tex:4467-4481` — **Theorem 2 (Discrete semigroup composition)** with proof. Statement: $(\mathcal{R}_{s+1})_*(\mathcal{R}_s)_*\mathbb{P}_s = (\mathcal{R}_{s+1}\circ\mathcal{R}_s)_*\mathbb{P}_s$.
- `Attention/Participatory_it_from_bit.tex:4485-4495` — **Theorem 3 (Gauge covariance of $\mathcal{R}_s$)** with proof. Statement: $\mathcal{R}_s(h\cdot X_s) = h\cdot\mathcal{R}_s(X_s)$ for compact $G$ with bi-invariant Karcher means and gauge-invariant weights.

### Sub-claim B — Closure ansatz, Schur, adiabatic elimination

- `Attention/Participatory_it_from_bit.tex:4370-4376` — Paragraph "Closure ansatz on the multi-agent class."
  - Eq.~\ref{eq:rg_closure} (line 4372): $\varepsilon_I = \inf_{c}\|\mathcal{F}^{\mathrm{exact}}_{s+1} - \mathcal{F}^{\mathrm{agent}}_{s+1} - c\|_{L^\infty(\mathcal{N}_I)} / (\tau + \|\mathcal{F}^{\mathrm{exact}}_{s+1}\|_{L^\infty(\mathcal{N}_I)})$.
  - Sufficient conditions (i)-(v) for $\varepsilon_I \ll 1$: Gaussian children, coherent cluster, positive constrained weighted gap, edge-marginal compatibility, compact gauge group.
- `Attention/Participatory_it_from_bit.tex:4378-4392` — Linearized barycentric expansion, internal Hessian $H_I^\perp \approx L_I \otimes F(q_I)$, softmax-eliminated correction.
  - Eq.~\ref{eq:rg_internal_hessian} (line 4380): $\mathcal{F}_I^{\mathrm{int}} = \tfrac{1}{2}\xi^\top H_I^\perp(q_I)\xi + O(\|\xi\|^3)$ with $H_I^\perp \approx L_I \otimes F(q_I)$.
  - Eq.~\ref{eq:rg_constrained_gap} (line 4389): $\lambda_{I,w} = \inf_{\xi: \sum w_i\xi_i = 0, \xi\neq 0} \xi^\top L_I \xi / \xi^\top\xi$, $m_I = \lambda_{I,w}\lambda_{\min}(F)$.
- `Attention/Participatory_it_from_bit.tex:4394-4399` — Laplace approximation Eq.~\ref{eq:rg_laplace}: $\mathcal{F}_{s+1}(Y_I) = \mathcal{F}_{\mathrm{eff}}(Y_I) + \tfrac{\tau}{2}\log\det'H_I^\perp(Y_I) + \mathrm{const} + \mathcal{E}_{\mathrm{anh}}$.
- `Attention/Participatory_it_from_bit.tex:4401-4406` — Renormalized inter-cluster coupling at raw-conductance level. Eq.~\ref{eq:rg_inter_cluster} (line 4402): $\kappa_{IJ}^R = \sum_{i\in I}\sum_{j\in J}w_i^I w_j^J \kappa_{ij}$; edge-marginal compatibility $w_i^I = r_i/\sum r_{i'}$ vanishes the first-order residual.
- `Attention/Participatory_it_from_bit.tex:4408-4419` — Paragraph "Adiabatic elimination." Four hypotheses (a)-(c) and (e). Eq.~\ref{eq:rg_schur} (line 4410): $G_{\mathrm{eff}} = G_{YY} - G_{Y\xi}G_{\xi\xi}^{-1}G_{\xi Y}$. Schur-complement flow Eq.~\ref{eq:rg_schur_flow} with error terms.
- `Attention/Participatory_it_from_bit.tex:4500-4522` — **Theorem 4 (Gaussian closure with local-potential correction)** with proof. Statement: under conditions (i)-(v), exact renormalized free energy lies in Gaussian quadratic KL class augmented by smooth scalar potential $V(Y) = \tfrac{\tau}{2}\log\det'A_{\xi\xi}(Y)$; strict Gaussian-KL closure when $A_{\xi\xi}$ is $Y$-independent. Schur-complement form Eq.~\ref{eq:rg_app_schur_complement} (line 4511).
- `Attention/Participatory_it_from_bit.tex:4527-4539` — Renormalized inter-cluster transport $\Omega_{IJ}^R$ as weighted Karcher mean (line 4530); holonomy obstruction $\mathcal{H}_{IJ}$ (line 4534) as "one of the principal closure-error terms"; edge-marginal mismatch $\delta_I^{\mathrm{marg}}$ (line 4539).
- `Attention/Participatory_it_from_bit.tex:4541-4554` — **Proposition 1 (Schematic closure-residual bound)**. The schematic inequality with constants $C_1, \ldots, C_6$ "depending on the closure norm $\|\cdot\|_\mathcal{B}$ and the regularity of the parent ansatz." Manuscript states: "This is stated as a proposition rather than a theorem because the constants $C_k$, the closure norm $\|\cdot\|_\mathcal{B}$, and the regularity class of the parent ansatz are not pinned down here, and tightness is not established; a theorem-grade statement would require specifying each."
- `Attention/Participatory_it_from_bit.tex:4573-4574` — Adiabatic elimination cited from `[fenichel1979geometric]`.

### Sub-claim C — Retention rules and detector-as-surrogate

- `Attention/Participatory_it_from_bit.tex:4421-4427` — Paragraph "Retention rules." Two distinct principles: Wilsonian relevance criterion at fixed point classifying eigendirections of $DR_{\theta^*}$ by multipliers `[Wilson1971, wilson1975renormalization]`, and MDL/Bayesian retention `[Schwarz1978, Rissanen1978]`. Eq.~\ref{eq:rg_retention_gain} (line 4424): $\Delta_I = \mathcal{L}_{\mathrm{micro}}(I) - \mathcal{L}_{\mathrm{parent}}(I)$ with complexity penalty $C(I)$ chosen as BIC, Bayesian-evidence prior, or variational complexity.
- `Attention/Participatory_it_from_bit.tex:4427` — Manuscript notes: "RG-relevant directions need not minimize description length, and MDL-preferred coarsenings need not be RG-relevant; the two criteria coincide only in favorable regimes."
- `Attention/Participatory_it_from_bit.tex:4429-4430` — Paragraph "Detector as candidate-selection surrogate." Manuscript states: "tightness of this residual bound is not established here."
- `Attention/Participatory_it_from_bit.tex:4556-4557` — Detector $\Gamma_I = P_I C_q(I) C_p(I) \in [0,1]$ with bounded exponentials. Conditions: $\Gamma_I > \Gamma_{\min}$, $|I| \ge N_{\min}$, $m_I > m_{\min}$.
- `Attention/Participatory_it_from_bit.tex:4559-4571` — **Theorem 5 (Detector implies positive retention gain)** with proof. Statement: under the local-Lipschitz bound $\mathcal{F}_{\mathrm{parent}}^*(I) \le \mathcal{F}_{\mathrm{micro}}^*(I) - A_I + L_q V_I^{(q)} + L_p V_I^{(p)} + \varepsilon_I$, the condition at Eq.~\ref{eq:rg_app_detector_retention} (line 4563) implies $\Delta_I > 0$ when $\Gamma_I > \Gamma_{\min}$.

### Sub-claim D — RG framing and Wilson1971/Cardy1996

- `Attention/Participatory_it_from_bit.tex:4354` — Manuscript opens with citations to `[Wilson1971, Cardy1996]` for the RG construction template.
- `Attention/Participatory_it_from_bit.tex:2137` — Prior section already disclaimed: "the present construction is RG-inspired rather than a literal RG analysis: we do not exhibit a $\beta$-function, locate fixed points, or demonstrate scale invariance beyond the parametric form."
- `Attention/Participatory_it_from_bit.tex:4576-4577` — Paragraph "Finite-size scaling." Manuscript states: "A stronger phase-transition or universality claim than the body subsection asserts would require finite-size scaling tests... we register this as a future-work program rather than as a delivered result, and the body subsection accordingly speaks of 'reorganization event' or 'phase-transition-like event' rather than 'phase transition' in the statistical-mechanical sense."
- `Attention/Participatory_it_from_bit.tex:4579-4580` — Summary: "The construction is rigorous on a compact gauge group; the noncompact case requires a gauge slice or regulator that the present manuscript does not provide."

## Canon excerpts (from `.claude/agents/vfe-knowledge/external_canon_inference.md` and external_canon_math.md)

- **§3 — Hierarchical / nested formulations.** Standard hierarchical FEP `[Friston2017Graphical, ParrPezzuloFriston2022 Ch. 9]` is Markovian-in-scale with cross-level couplings from the generative-model conditional $p(s_\ell | s_{\ell+1})$. The manuscript's RG construction extends beyond this — pushforward of a full Gibbs measure under a coarse-graining map, not Markov-blanket-mediated information flow.
- **§10 Pitfall 6 — Single-agent FEP extended to multi-agent.** "The standard FEP literature does not contain this specific functional with gauge-transport-coupled KL terms... Flag the user's coupling as novel." The RG section operates within the user's multi-agent functional class; the "RG closure" therefore claims closure within a novel class.

## Standard references the teams should cite

**Renormalization group (textbook canon):**
- `[Wilson1971]` — "Renormalization group and critical phenomena. I."; classic. The Wilsonian RG flow on couplings with $\beta$-function $dg/dl = \beta(g)$.
- `[Wilson1974]` — "The renormalization group and the $\epsilon$ expansion." Lattice methods.
- `[wilson1975renormalization]` — "The renormalization group: critical phenomena and the Kondo problem."
- `[Cardy1996]` — *Scaling and Renormalization in Statistical Physics*, Cambridge University Press. The standard pedagogical reference for the structural form of an RG step (coarse-grain, rescale, renormalize).
- `[Goldenfeld1992]` — *Lectures on Phase Transitions and the Renormalization Group*. Block-spin RG and Migdal-Kadanoff treatments.
- `[Kadanoff1966]` — Block-spin scaling, the original coarse-graining idea.
- `[FisherBerkerWegner1980]` — Phenomenological RG and effective Hamiltonians.

**Gibbs measures and pushforward:**
- `[Bogachev2007]` — *Measure Theory* (Vol I, II), Springer. Pushforward of Borel measures, Radon-Nikodym derivatives.
- `[Georgii2011]` — *Gibbs Measures and Phase Transitions*, de Gruyter. Standard reference for Gibbs measures on infinite-volume lattice systems.

**Riemannian center of mass:**
- `[Karcher1977]` — "Riemannian center of mass and mollifier smoothing." Existence and uniqueness on convex normal balls of radius $< r_{\mathrm{cx}}/2$ where $r_{\mathrm{cx}}$ is the convexity radius.
- `[Afsari2011]` — "Riemannian $L^p$ center of mass: existence, uniqueness, and convexity." Modern refinement.
- `[Pennec2006]` — "Intrinsic statistics on Riemannian manifolds." The forward-KL barycenter (M-projection) on statistical manifolds.

**Slow-manifold reduction / normal hyperbolicity:**
- `[Fenichel1979]` — "Geometric singular perturbation theory for ordinary differential equations." The classical reference for invariant slow manifolds; the manuscript cites this at line 4574.
- `[Jones1995]` — "Geometric singular perturbation theory," LNM 1609. Modernization of Fenichel.
- `[Kuehn2015]` — *Multiple Time Scale Dynamics*. Comprehensive modern treatment.

**Schur complements and block-matrix theory:**
- `[HornJohnson2013]` §0.8.5 — Schur complement formula for block-decomposed matrices. The manuscript cites this for the effective metric Eq.~\ref{eq:rg_schur}.
- `[Zhang2005]` — *The Schur Complement and Its Applications*, Springer.

**Laplace approximation and saddle-point asymptotics:**
- `[Wong2001]` — *Asymptotic Approximations of Integrals*, SIAM. The manuscript cites this for the Laplace integration of internal modes.
- `[BenderOrszag1999]` — *Advanced Mathematical Methods for Scientists and Engineers*, Springer. Saddle-point / steepest-descent.

**MDL / BIC model selection:**
- `[Schwarz1978]` — "Estimating the dimension of a model." Defines BIC = $-2\log L + k\log n$.
- `[Rissanen1978]` — "Modeling by shortest data description." Original MDL formulation.
- `[Grunwald2007]` — *The Minimum Description Length Principle*. Modern textbook.

**Information geometry:**
- `[AmariNagaoka2000]` — *Methods of Information Geometry*. M-projection / forward-KL projection onto exponential families.
- `[Amari2016]` — *Information Geometry and Its Applications*.

**Gauge theory pushforward:**
- `[Nakahara2003]` §10.3 — Principal bundles, associated bundles, gauge transformations.
- `[KobayashiNomizu1963]` Vol I — Foundations of Differential Geometry; canonical reference.
- `[BlauThompson1991]` — Lectures on 2D gauge theories; pushforward / Faddeev-Popov gauge-fixing.

**Variational inference and free-energy correspondences:**
- `[BleiKuckelbirgJordan2017]` — Variational inference primer; the additive-KL ↔ product-prior correspondence.
- `[Friston2010]` — Standard variational free-energy form.

## What this evidence does NOT settle

- Whether the manuscript's "exact pushforward RG" framing — which is structurally just the change-of-variables for pushforward measures `[Bogachev2007 §3]` — earns the additional baggage of the term "renormalization group," which in `[Wilson1971, Cardy1996]` is bound up with $\beta$-function flow, fixed points, and critical exponents. The manuscript itself notes that finite-size scaling and universality are "future work."
- Whether the "linear-Gaussian" hypothesis in Theorem 4 is satisfied by any nontrivial empirical regime of the manuscript's simulator. The constrained internal Hessian $A_{\xi\xi}$ depends on the parent state $Y$ through the precision matrices $\Sigma_i^{-1}$; the "strict Gaussian-KL closure when $A_{\xi\xi}$ is $Y$-independent" condition may be empty in practice.
- Whether Proposition 1's schematic closure-residual bound is mathematically meaningful given that the constants $C_1, \ldots, C_6$, the closure norm $\|\cdot\|_\mathcal{B}$, and the regularity class of the parent ansatz are explicitly not specified. The manuscript labels this as a proposition rather than a theorem precisely for this reason; whether the labeling is sufficient or whether the underspecification renders the bound vacuous is the central question.
- Whether the local-Lipschitz bound assumed in Theorem 5 ($\mathcal{F}_{\mathrm{parent}}^*(I) \le \mathcal{F}_{\mathrm{micro}}^*(I) - A_I + L_q V_I^{(q)} + L_p V_I^{(p)} + \varepsilon_I$) has been derived or is asserted. If asserted without derivation, the theorem is conditional on a hypothesis that is itself unverified.
- Whether the Karcher mean for the renormalized inter-cluster transport $\Omega_{IJ}^R$ (line 4530) is well-defined when constituent transports $\Theta_{ij}^{IJ}$ straddle the convex normal ball cut locus of $G$. The cut-locus issue from the prior debate's verdict at `docs/debates/2026-05-19-agent-meta-agent-hierarchy-theory/04_verdict.md` extends naturally to this object.
- Whether the noncompact $\mathrm{GL}^+(K)$ caveat (lines 4368, 4435) actually leaves the construction without a rigorous foundation in the framework's natural gauge group, since the manuscript explicitly states "the noncompact case requires a gauge slice or regulator that the present manuscript does not provide."
- Whether Theorem 4's "local-potential correction" $V(Y) = \tfrac{\tau}{2}\log\det'A_{\xi\xi}(Y)$ is genuinely additive and smooth in $Y$ across the full simulator regime, or whether it introduces singularities at points where the constrained internal Hessian becomes degenerate (e.g., near phase-transition-like points where the soft mode condition (e) of the adiabatic elimination paragraph is violated).

These are the structural cracks the red team should probe and the blue team must address.
