# Evidence Pack — pifb-discussion-kinetic-term

## Manuscript references

### The Kinetic Term subsection under debate (Discussion §3085-3095)

- `Participatory_it_from_bit.tex:3085` — subsection header "The Kinetic Term: Biological vs. Fundamental Readings"
- `Participatory_it_from_bit.tex:3087` — "The kinetic term $\frac{1}{2}d\mu^T \Sigma^{-1} d\mu$ is second-order in the belief update. Standard active-inference treatments employ first-order natural-gradient descent on the variational free energy because synaptic time constants dominate over any putative inertial term on cortical timescales; the overdamped reduction is not an oversight but the appropriate biophysical regime."
- `Participatory_it_from_bit.tex:3089-3091` — the overdamped-limit equation: $m\ddot{q} + \gamma\dot{q} = -\nabla V \xrightarrow{m/\gamma \to 0} \gamma\dot{q} = -\nabla V$
- `Participatory_it_from_bit.tex:3093` — "Furthermore, the framework allows agent priors to evolve dynamically. In traditional approaches predictive models are considered fixed."
- `Participatory_it_from_bit.tex:3095` — the load-bearing hedge: "Where the present framework reintroduces the second-order structure is in its reading of the variational principle as a candidate fundamental scaffolding rather than a phenomenological model of neural inference; under that reading, the precision-induced configuration-space stiffness developed in Section~\ref{sec:mass} acquires a kinetic-energy counterpart through the postulated velocity-quadratic metric form. This reintroduction is contingent on the fundamental-physics reading and inherits the postulate that the stiffness matrix $\Sigma_p^{-1}$ doubles as the inertia tensor; it is not a derivation of inertia from the variational principle, and the biological-timescale omission remains entirely correct for its intended target."

### Mass analogy machinery the subsection cross-references (sec:mass)

- `Participatory_it_from_bit.tex:1847` — sec:mass intro: "postulate-dependent mass-like scaling $\omega^2 \propto m_{\text{eff}}^{-1}$ in the isolated-agent harmonic limit, once the kinetic-metric postulate of Section sec:velocity_quadratic is in place...not as a derivation of physical inertial mass from statistical precision."
- `Participatory_it_from_bit.tex:1849` — "What the Hessian gives, and what is added": explicit statement that "$M_i$ is only a stiffness on belief configuration space" without the kinetic-metric postulate; "The identification of $\Sigma_p^{-1}$ with an effective mass requires a separate kinetic-metric postulate" and "Under standard Lagrangian mechanics~\cite{arnold1989mathematical} a non-degenerate harmonic identification requires the inertia tensor and the potential Hessian to be operationally independent $(0,2)$-tensors at the equilibrium configuration; the present construction reuses the same matrix $M_{\mu\mu}$ for both roles and therefore does not supply such an independent test."
- `Participatory_it_from_bit.tex:1958-1961` — "Off-diagonal block caveats" (sec:mass_block_caveats): "What can fail under asymmetric attention is not block-transpose symmetry of the Hessian but the existence of a global potential whose second variation matches the running update rule: if attention weights $\beta_{ij}$ are treated as instantaneous and frozen-asymmetric ($\beta_{ik}\neq\beta_{ki}$) within the dynamics, the resulting flow is no longer the gradient flow of $\mathcal{F}$ alone, and the bilinear form $M_{\mu\mu}$ does not function as the inertia tensor of a conservative Hamiltonian." The Newtonian reading is recovered only "in the symmetric-attention or isolated-agent limits."
- `Participatory_it_from_bit.tex:2013-2024` — "Within-Framework Interpretation: Stiffness as Precision (Mass Analogy)": "the identification of the Hessian sector $[M_{\mu\mu}]$ with an 'effective mass' is interpretive within the framework rather than a derivation of physical inertial mass...we reuse $M_i$ in Eq. eq:effective_mass as both the stiffness (Hessian of $\mathcal{F}$ at the consensus point) and the inertia (kinetic-metric coefficient, postulated separately)."
- `Participatory_it_from_bit.tex:2026-2034` — Section sec:velocity_quadratic explicitly headed "This is a postulate, not a consequence of $\mathcal{F}$." "Under this identification the harmonic-oscillator scaling $\omega^2 \propto k/m$ is a definitional consequence of the postulate rather than an independent dynamical scaling: when $k$ and $m$ are both equal to $M_{\mu\mu}$ by construction, $\omega^2$ reduces to a per-direction unit relation and the analogy is structural, not empirical."

### Bibliographic resource

- `references.bib` — `arnold1989mathematical` (Arnold, *Mathematical Methods of Classical Mechanics*, 2nd ed., Springer 1989) is the canonical reference cited at line 1849; teams should consult §13–15 (Lagrangian mechanics, Legendre transform) and §22–24 (small oscillations, normal modes) for the standard form of the kinetic-energy reading.

## Canon excerpts (teams should expand via WebFetch / literature-review)

### Active inference / FEP canon (relevant to first-order overdamped status)

- **Friston, K. (2010)**, "The free-energy principle: a unified brain theory?", *Nature Reviews Neuroscience* 11(2), 127–138, Eq. 2.2: $\dot{\mu} = -\partial_\mu F$ (first-order natural-gradient descent on the free energy). This is the canonical overdamped form.
- **Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., Pezzulo, G. (2017)**, "Active inference: a process theory," *Neural Computation* 29(1), 1–49 — re-stating the first-order descent as the operative update rule.
- **Buckley, C. L., Kim, C. S., McGregor, S., Seth, A. K. (2017)**, "The free energy principle for action and perception: A mathematical review," *Journal of Mathematical Psychology* 81, 55–79 — explicit derivation that synaptic time constants reduce the dynamics to first-order on neural-process timescales. Confirms the "overdamped reduction is the appropriate biophysical regime" claim.

### Classical mechanics canon (relevant to the kinetic-term reintroduction)

- **Arnold, V. I. (1989)**, *Mathematical Methods of Classical Mechanics*, 2nd ed., Springer (Graduate Texts in Mathematics 60). Already in `references.bib`. The canonical Lagrangian/Hamiltonian formalism that the manuscript's Section sec:mass invokes. §22.2 makes explicit that a kinetic energy is a Riemannian metric on configuration space, and §15.A states the small-oscillation result $\omega^2 = K/m$ where K is the Hessian of the potential at equilibrium and m is the inertia tensor from the kinetic-energy quadratic form, *as independent objects*.
- **Goldstein, H., Poole, C., Safko, J. (2002)**, *Classical Mechanics*, 3rd ed., Addison-Wesley. §1.4 (D'Alembert and Lagrange equations), §6.2 (small oscillations) — independent textbook source for the same canonical form.
- **Marsden, J. E. and Ratiu, T. S. (1999)**, *Introduction to Mechanics and Symmetry*, 2nd ed., Springer (TAM 17). §3.3 (Lagrange-d'Alembert) and §7 (Hamiltonian mechanics on Lie groups) — for the gauge-frame / Lie-group case the manuscript needs.

### Free-energy-as-action canon (relevant to "fundamental-physics reading")

- **Friston, K., Sengupta, B., Auletta, G. (2014)**, "Cognitive dynamics: from attractors to active inference," *Proceedings of the IEEE* 102(4), 427–445. Discusses second-order extensions of free-energy dynamics through generalized motion.
- **Ramstead, M. J. D., Friston, K. J., et al. (2018)**, "Answering Schrödinger's question: A free-energy formulation," *Physics of Life Reviews* 24, 1–16. Reads FEP as an action principle and asks the kinetic-term question explicitly.
- **Friston, K., Da Costa, L., Sajid, N., Heins, C., Ueltzhöffer, K., Pavliotis, G. A., Parr, T. (2023)**, "The free energy principle made simpler but not too simple," *Physics Reports* — explicit discussion of stochastic-versus-deterministic dynamics and the role of fluctuating versus dissipative parts.

### Information geometry canon (relevant to the "natural gradient" reading)

- **Amari, S. (1998)**, "Natural gradient works efficiently in learning," *Neural Computation* 10(2), 251–276. The natural-gradient flow $\dot{\theta} = -F(\theta)^{-1} \nabla L(\theta)$ is explicitly first-order; F here is the Fisher information matrix, playing the role of a Riemannian metric on the statistical manifold, NOT an inertia tensor.
- **Amari, S. and Nagaoka, H. (2000)**, *Methods of Information Geometry*, AMS. §3.4 — the natural gradient is a Riemannian metric construction on a statistical manifold, no second-order kinetic term enters.

## What this evidence does NOT settle

1. Whether the equivalence-of-stiffness-and-inertia postulate the manuscript registers at 3095 and 2026-2034 is a substantive new physical postulate (deserving a Discussion paragraph dedicated to its specific empirical consequences) or a definitional convention (the mathematics doesn't care what we call $M_{\mu\mu}$).
2. Whether the manuscript's hedge is **too weak** because it does not name the equivalence-principle-style condition between stiffness and inertia as a *physical postulate analogous to the equivalence of inertial and gravitational mass*, which the manuscript ITSELF identifies as analogous at lines 3114-3116 of the Pullback Gravity subsection.
3. Whether the manuscript's hedge is **too strong** because the cited Arnold 1989 §22.2 actually does NOT require strict operational independence of kinetic-metric and potential-Hessian — the standard treatment allows the kinetic energy to be $\frac{1}{2}\dot{q}^T g_{ij}(q) \dot{q}$ where $g_{ij}$ is any Riemannian metric on configuration space, and one may freely choose $g_{ij} = $ Hessian of V at equilibrium without contradiction; the "independence" is between the metric *as a function on configuration space* and the potential's behavior at the equilibrium point.
4. Whether the manuscript's invocation of the overdamped reduction $m\ddot{q} + \gamma\dot{q} = -\nabla V \to \gamma\dot{q} = -\nabla V$ at line 3089-3091 maps correctly onto the active-inference natural-gradient flow, given that the natural-gradient flow's "metric" is the Fisher information (which corresponds to $\gamma$, the damping coefficient, not $m$) — Amari 1998's natural gradient is explicitly a *gradient flow on a Riemannian manifold*, not a damped Newtonian limit. This may be an analogy error.
5. Whether the "framework allows agent priors to evolve dynamically; in traditional approaches predictive models are considered fixed" claim at line 3093 is accurate — this is at best an oversimplification; standard active-inference allows learning of model parameters (the M-step in expectation-maximization-based active inference; Friston 2017 §3). Teams should verify against the canon.

Teams are required to address points 1–4 with primary-source citations; point 5 is a factual accuracy check.
