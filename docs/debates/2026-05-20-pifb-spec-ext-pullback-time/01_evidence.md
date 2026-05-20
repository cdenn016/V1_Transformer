# Evidence Pack — pifb-spec-ext-pullback-time

## Manuscript references (lines 2504-2695)

### §Speculative Extensions opening (2504-2507)
- `:2504-2505` — section header with `\label{sec:speculative_extensions}` and title "From Information to Physical and Phenomenal Reality"
- `:2507` — Outlook framing: "the framework no longer makes empirically falsifiable predictions in the sense set out in Section sec:scope_limitations... Readers seeking the empirically anchored content of the framework should read this section as Outlook rather than as Results."

### §Time as Information Flow (2509-2535)
- `:2509-2517` — Bit-Counting Time subsubsection
  - 2513: **Dimensional disclaimer.** "$\tau_i$ is a *dimensionless* count parameter labeling positions on belief-update trajectories; it has no SI units of seconds." Operational definition $\Delta\tau_i = \Delta I_i / (1\text{ bit})$ with $\Delta I_i = \mathrm{KL}(q_i^{\mathrm{new}} \| q_i^{\mathrm{old}})$.
  - 2517: comparison literature — Wheeler 1990, Lloyd 2002, Rovelli 2004 relational time, Page-Wootters 1983, Connes-Rovelli 1994 thermal time, Jacobson 1995, Van Raamsdonk 2010
- `:2519-2523` — Minimum Time from Minimum Information subsubsection
  - 2521: Planck time $t_P = \sqrt{\hbar G/c^5} \sim 10^{-43}$s
  - 2523: bit-Planck connection $t_P \sim \hbar/E_P \sim 1\text{ bit}/(\text{max info rate})$ explicitly labelled "highly speculative and requires substantial development to make rigorous"
- `:2525-2534` — Fisher Arc Length subsubsection with `\label{sec:fisher_arc_length}`
  - 2528-2533: Fisher-Rao arc length $\Delta\tau = \int \sqrt{g_{\mathcal B}(\dot q, \dot q)} d\tau$
  - 2534: "We do not identify this quantity with relativistic proper time: as discussed in detail in Section sec:fisher_arc_length, Fisher arc length and special-relativistic proper time have opposite character (positive-definite Riemannian versus indefinite Lorentzian) and run in opposite directions with motion."
- `:2536 (inline!)` — `\subsection{Natural Gradient Dynamics on Statistical Manifolds}` jammed inline with preceding paragraph. **Presentation defect.**

### §Natural Gradient Dynamics on Statistical Manifolds (2536-2578)
- `:2538-2540` — Fisher Information for Gaussian Distributions subsubsection
- `:2542-2578` — Gauge-Covariant Update Equations subsubsection
  - Eq.~eq:gauge_natural_gradient: $\dot\mu = -\eta_\mu \tilde\nabla_\mu \mathcal F$, $\dot\Sigma = -2\eta_\Sigma \Sigma(\nabla_\Sigma \mathcal F)\Sigma$, $\dot U = -\eta_\phi U \tilde\nabla_{\phi} \mathcal F$
  - Eq.~eq:pullback_metric: position-dependent right-invariant metric $\mathcal G_{ab}(\phi) = \langle\Psi(\mathrm{ad}_\phi)T_a, \Psi(\mathrm{ad}_\phi)T_b\rangle_G$ with $\Psi(z) = (e^z-1)/z$
  - Eq.~eq:gauge_group_retraction: discrete-time retraction $U^{t+1} = U^t \exp(-\eta_\phi \tilde\nabla_{\phi^t} \mathcal F)$
  - Killing form for $\mathfrak{so}(K)$ with $K > 2$: $B(X,Y) = (K-2)\mathrm{tr}(XY)$, restricted to skew-symmetric is proportional to standard $-\mathrm{tr}(XY)$

### §It From Bit: The Pullback Construction (2580-2585)
- `:2580-2581` — subsection header with `\label{sec:pullback}` and Wheeler "it from bit" cite
- `:2585` — toy-model disclaimer: "We emphasize this is a toy model demonstration that spacetime-like geometry can emerge from information, not a claim that this is how physical spacetime actually arises... The signature problem is not a minor technical detail but the central unsolved challenge preventing this framework from genuinely explaining relativistic spacetime rather than merely producing toy models with spacetime-like features."

### §The Pullback Mechanism: From Information to Geometry (2587-2695)
- `:2587-2599` — Agent Sections as Smooth Fields subsubsection
  - Fisher-Rao metric Eq for Gaussians at 2595-2597
- `:2601-2624` — A Bundle Metric on the Associated Bundle subsubsection
  - Eq.~eq:bundle_metric: $g_{E_q}\big|_{(c,q)}(X_H + X_V, Y_H + Y_V) := g_\mathcal{C}^{\mathrm{tw}}(\pi_* X_H, \pi_* Y_H) + g_\mathcal{B}(X_V, Y_V)$
  - Eq.~eq:horizontal_metric: $g_{\mathcal{C},\mu\nu}^{\mathrm{tw}}(c) := \kappa(A^{(i)}_\mu(c), A^{(i)}_\nu(c))$
  - **Piecewise κ convention** at 2620-2623: $\kappa(A,B) = -\mathrm{tr}(AB)$ for compact (positive-definite); $\kappa(A,B) = +\mathrm{tr}(AB)$ for $\mathfrak{gl}(K,\mathbb{C})$ or non-compact (indefinite)
  - Eq.~eq:induced_metric_full (boxed) at 2640: $G^{(q)}_{i,\mu\nu}(c) = \kappa(A^{(i)}_\mu, A^{(i)}_\nu) + \mathbb{E}_{q_i(c)}[(\nabla^{(i)}_\mu \log q_i)(\nabla^{(i)}_\nu \log q_i)]$
- `:2645-2646` — **Gauge-invariance disclosure** paragraph: under local gauge transformation $U_i \to U_i g(c)$, the connection $A^{(i)} = U_i^{-1} dU_i$ acquires Maurer-Cartan piece; $\kappa(A_\mu, A_\nu)$ is therefore NOT invariant; the horizontal block is agent-frame-dependent. Gauge-invariant content is routed through sec:consensus_metric.
- `:2648-2658` — same construction for prior pullback $G^{(p)}$ (boxed), generative-model $G^{(s)}$ (boxed), hyper-prior $G^{(r)}$
- `:2663-2676` — Three Tiers of Induced Geometry subsubsection with `\label{sec:three_tiers}`
  - epistemic ($G^{(q)}$), expectational ($G^{(p)}$), structural ($G^{(s)}, G^{(r)}$)
  - 2672: "Different agents with different generative models therefore perceive different structural geometries on the same underlying noumenal base $\mathcal{C}$, and within the manuscript's interpretive frame and conditional on the regulator caveat of Section sec:consensus_metric, there is no agent-independent metric on $\mathcal{C}$, only the family of agent-dependent induced metrics arising from their slow-timescale model parameters."
- `:2678-2694` — Gaussian R^2 example with conformal-factor reduction $G^{(q)}_{i,\mu\nu}(c) = (1/\sigma^2)(\partial_\mu\mu_i)\cdot(\partial_\nu\mu_i)$ when isotropic constant covariance

## Canon excerpts (teams should expand)

### Time-from-information canon
- **Wheeler, J. A. (1990)**, "Information, Physics, Quantum: The Search for Links" — already cited
- **Lloyd, S. (2002)**, "Computational capacity of the universe," *Phys. Rev. Lett.* 88, 237901
- **Rovelli, C. (2004)**, *Quantum Gravity*, Cambridge — relational time
- **Page, D. N., Wootters, W. K. (1983)**, "Evolution without evolution," *Phys. Rev. D* 27, 2885
- **Connes, A., Rovelli, C. (1994)**, "Von Neumann algebra automorphisms and time-thermodynamics relation," *Class. Quant. Grav.* 11, 2899
- **Jacobson, T. (1995)**, "Thermodynamics of spacetime: The Einstein equation of state," *PRL* 75, 1260
- **Van Raamsdonk, M. (2010)**, "Building up spacetime with quantum entanglement," *Gen. Rel. Grav.* 42, 2323

### Fisher-Rao / information geometry canon
- **Amari, S., Nagaoka, H. (2000)**, *Methods of Information Geometry*, AMS — Fisher-Rao metric is positive-definite Riemannian
- **Rao, C. R. (1945)**, "Information and accuracy attainable in the estimation of statistical parameters"

### Bundle metric / connection canon
- **Nakahara, M. (2003)**, *Geometry, Topology and Physics*, IoP — bundle metrics on principal/associated bundles
- **Baez, J., Muniain, J. (1994)**, *Gauge Fields, Knots and Gravity*, World Scientific
- **Kobayashi, S., Nomizu, K. (1963)**, *Foundations of Differential Geometry*, Vol. 1

### Killing form / Lie algebra inner product canon
- **Helgason, S. (1978)**, *Differential Geometry, Lie Groups, and Symmetric Spaces*, AMS — Killing form theory; bi-invariance; indefinite on non-compact groups

## What this evidence does NOT settle

1. **Presentation defect at line 2536.** The "Natural Gradient Dynamics on Statistical Manifolds" subsection header is inlined with the preceding paragraph rather than starting on a new line. Either a missing blank line or a missing `\par`. Verify whether this renders correctly in the compiled PDF.

2. **Bit-Planck connection rigor.** The 2523 paragraph "highly speculative and requires substantial development to make rigorous" is honest about the gap. The question is whether the framework should connect bit-counting time to Planck time even as a "speculative avenue" — Lloyd 2002 makes a related claim (computational capacity bounds), which the manuscript cites; but the specific identification $t_P \sim 1\text{ bit}/(\text{max info rate})$ requires more than dimensional analysis.

3. **Fisher arc length / proper time direction-of-time mismatch.** Line 2534 notes "run in opposite directions with motion." A fast-moving (relativistic) clock has SHORTER proper time (time dilation); a fast-updating (high Fisher arc length) agent has MORE bit-time. This opposite-direction observation is correct but the framework's implication that this opposition is "consistent with relational and emergent approaches to time" needs verification — most emergent-time proposals (Page-Wootters, Rovelli) do NOT have this opposite-direction feature.

4. **Bundle metric gauge non-invariance scope.** The 2645-2646 disclosure correctly registers that the horizontal block is gauge-non-invariant. But Eq.~eq:bundle_metric is presented at 2611 as a metric on the associated bundle $E_q$. If the metric itself is not gauge-invariant, calling it "a bundle metric" without qualifier is technically misleading — it is a bundle metric within an agent's gauge fixing. The 2624 paragraph says "fiber-respecting metric" and "gauge-orthogonal by construction" but the 2645 disclosure clarifies this is only within a fixed gauge. The two paragraphs should be more tightly linked.

5. **Three-tier identification $G^{(s)}$ as carrier of perceived geometry.** The sec:three_tiers reading at 2672 is a substantive interpretive identification — "the structural tier $G_i^{(s)}$ rather than the expectational tier $G_i^{(p)}$ or the epistemic tier $G_i^{(q)}$" carries phenomenal spatial structure. This was the move in the Pullback Gravity Discussion debate (debate 2026-05-20-pifb-discussion-pullback-gravity) that established the decoupling of $\Sigma_p^{-1}$ (inertia) from $\Sigma_s^{-1}$ (gravity-curvature). The forward-reference from sec:three_tiers to that Discussion subsection should exist; verify.

6. **Conformal-factor reduction interpretation at 2694.** "High certainty (small σ) magnifies distances. Regions where the agent is confident appear 'larger' in information geometry." Is this the canonical Fisher-Rao reading or framework-specific? Standard Fisher-Rao information geometry says high Fisher information = short statistical distances, which agrees, but the phenomenological "appear larger" framing is interpretive.

Teams should verify points 1-3 against the cited canon. Points 4-6 are within-manuscript interpretive consistency checks.
