# The Grand Scheme: Gauge-Theoretic Multi-Agent VFE as a Fundamental Theory

**Status:** Working theoretical framework. Core mathematical constructions established; several critical gaps remain between the pieces.

---

## 1. The Ontology: There Are Only Agents

The theory posits a single primitive: **agents**. Every entity in the framework --- internal states, observations, Markov blankets, environments, meta-agents --- reduces to agents carrying Gaussian beliefs on a statistical fiber bundle over a structureless base manifold.

### 1.1 Agent Definition

An agent $\mathcal{A}^i$ is a smooth local section of two independent associated fiber bundles:

$$\mathcal{A}^i = \bigl(q_i,\; p_i,\; s_i,\; r_i,\; \Omega_i,\; \widetilde{\Omega}_i,\; b_{0,i},\; c_{0,i},\; \chi_i\bigr)$$

| Symbol | Space | Interpretation | Timescale |
|---|---|---|---|
| $q_i = \mathcal{N}(\mu_{q,i},\, \Sigma_{q,i})$ | Belief fiber $\mathcal{B}_q$ | What agent thinks is happening now | Fast ($\eta_q$) |
| $p_i = \mathcal{N}(\mu_{p,i},\, \Sigma_{p,i})$ | Belief fiber $\mathcal{B}_q$ | What agent expects to happen | Medium ($\eta_p$) |
| $s_i = \mathcal{N}(\mu_{s,i},\, \Sigma_{s,i})$ | Model fiber $\mathcal{B}_p$ | Agent's model of reality | Slow ($\epsilon \eta_s$) |
| $r_i = \mathcal{N}(\mu_{r,i},\, \Sigma_{r,i})$ | Model fiber $\mathcal{B}_p$ | Prior on the model (from meta-agent above) | Slow ($\epsilon \eta_r$) |
| $\Omega_i \in \mathrm{GL}(K)$ | Belief gauge frame | Internal coordinate system for beliefs | Medium ($\eta_\Omega$) |
| $\widetilde{\Omega}_i \in \mathrm{GL}(K)$ | Model gauge frame | Internal coordinate system for models | Slow ($\epsilon \eta_{\tilde\Omega}$) |
| $b_{0,i},\, c_{0,i} > 0$ | Precision hyperparameters | Adaptive prior coupling $\alpha_i = c_0/(b_0 + D)$ | Slow ($\epsilon$) |
| $\chi_i: \mathcal{C} \to [0,1]$ | Support function | Where the agent exists on the base manifold | Fixed or very slow |

The belief fiber $(q_i, p_i, \Omega_i)$ handles perception and inference. The model fiber $(s_i, r_i, \widetilde{\Omega}_i)$ handles ontological structure and generative models. These are independent: $q_i$ is "what I think is happening"; $s_i$ is "what I think my model of the world IS." Aligning $s_i$ across agents creates shared ontologies.

### 1.2 Observations as Environmental Agents

The observation likelihood term
$$-\sum_i \int_{\mathcal{C}} \chi_i(c)\, \mathbb{E}_{q_i}\!\bigl[\log p(o_i \mid c)\bigr]\, dc$$
is formally equivalent to coupling with **environmental agents** $e_k$ carrying sharp beliefs:

$$q_{e_k}(c) = \delta(c - c_k), \qquad \beta_{i,e_k} \propto \log p_i(o_k \mid c)$$

The alignment term $\sum_{i,k} \beta_{i,e_k}\, D_{\mathrm{KL}}\!\bigl(q_i \,\|\, \Omega_{i,e_k}[q_{e_k}]\bigr)$ produces identical VFE gradients. Environmental agents are simply agents with certainty $\Sigma \to 0$ --- infinite precision, infinite mass. They anchor the system without being ontologically distinct.

### 1.3 Markov Blankets as Boundary Agents

In the hierarchical framework, a meta-agent $\alpha$ at scale $\zeta+1$ is composed of agents at scale $\zeta$ with soft membership $W_{i\alpha}(x) = S_{i\alpha}(x) \cdot C_{i\alpha}(x)$, where $S$ is the species gate (model alignment) and $C$ is coalition membership (belief alignment).

The **Markov blanket** of $\alpha$ is the set of agents with intermediate membership that also couple to agents outside $\alpha$:
$$\mathcal{B}_\alpha = \bigl\{i : 0 < W_{i\alpha} < 1,\; \exists\, j \text{ with } W_{j\alpha} \approx 0 \text{ and } \beta_{ij} > 0\bigr\}$$

These boundary agents carry the full $(q, p, s, r, \Omega, \widetilde{\Omega})$ tuple. They are not a separate ontological category but a derived geometric property of the membership field. The "blanket" is a collection of agents mediating between interior and exterior --- itself an agent-composed structure.

### 1.4 The Resulting Monism

With observations dissolved into environmental agents and blankets dissolved into boundary agents, the VFE functional becomes purely agent-agent coupling:

$$S = \sum_i \alpha_i\, D_{\mathrm{KL}}(q_i \| p_i) + \sum_{i,j} \beta_{ij}\, D_{\mathrm{KL}}\bigl(q_i \| \Omega_{ij}[q_j]\bigr) + \sum_{i,j} \gamma_{ij}\, D_{\mathrm{KL}}\bigl(s_i \| \widetilde{\Omega}_{ij}[s_j]\bigr)$$

No environment. No observations. No blankets. Only agents coupled through gauge-covariant KL divergences at multiple scales.

---

## 2. The Gauge Structure

### 2.1 Why GL(K)

The KL divergence between Gaussian distributions is invariant under any invertible linear transformation of the latent space:

$$D_{\mathrm{KL}}(\Omega_* P \,\|\, \Omega_* Q) = D_{\mathrm{KL}}(P \| Q), \qquad \forall\, \Omega \in \mathrm{GL}(K)$$

This is a theorem, not a design choice. The full $f$-divergence family shares this invariance because density ratios cancel Jacobians. Agents can rotate, scale, and shear their internal coordinate systems without affecting information-geometric quantities. Only relative frames $\Omega_{ij} = \Omega_i \Omega_j^{-1}$ are physical.

### 2.2 Transport

The gauge transport operator between agents $i$ and $j$ is:

$$\Omega_{ij}(c) = \exp\!\bigl(\phi_i(c)\bigr)\, \exp\!\bigl(-\phi_j(c)\bigr) \in \mathrm{GL}^+(K)$$

Transported beliefs:
$$\Omega_{ij}[q_j] = \mathcal{N}\!\bigl(\Omega_{ij}\mu_j,\; \Omega_{ij}\,\Sigma_j\,\Omega_{ij}^\top\bigr)$$

The covariance always transports via the sandwich product $\Sigma' = \Omega\,\Sigma\,\Omega^\top$. This is the single most important correctness constraint.

### 2.3 Flat Bundle (Vertex-Local Frames)

For vertex-local frames, the cocycle condition holds identically:

$$\Omega_{ij}\,\Omega_{jk} = \Omega_i\Omega_j^{-1}\,\Omega_j\Omega_k^{-1} = \Omega_i\Omega_k^{-1} = \Omega_{ik}$$

Holonomy vanishes: $H_{ijk} = \Omega_{ij}\Omega_{jk}\Omega_{ki} = I$. The gauge bundle is flat and transport is path-independent. This is the regime appropriate for compositional semantics in natural language and, hypothetically, for the compositional structure of everyday physical reasoning.

### 2.4 Non-Flat Extension (Lattice Gauge)

For non-flat transport, edge-local twist variables $V_{ij} \in \mathrm{GL}(K)$ break the cocycle condition:

$$\hat{\Omega}_{ij} = \Omega_i \cdot V_{ij} \cdot \Omega_j^{-1}$$

The plaquette (elementary Wilson loop) measures curvature:
$$W(\square) = V_{12}\, V_{23}\, V_{34}^{-1}\, V_{41}^{-1} \neq I$$

Regularized by the Yang-Mills action $S_{\mathrm{YM}} = (\beta/K) \sum_\square [K - \mathrm{Re}\,\mathrm{tr}(W)]$.

### 2.5 Lorentzian Signature from GL(K, $\mathbb{C}$)

Complex gauge frames with imaginary temporal components produce Lorentzian signature through the Yang-Mills kinetic metric. Given $\phi(\tau, x) = i\,\psi_\tau \cdot T + \psi_x \cdot T$, the connection one-forms yield:

$$G_{\tau\tau} = \mathrm{tr}(A_\tau^2) = i^2 (\partial_\tau \psi_\tau)^2\,\mathrm{tr}(T^2) = -2(\partial_\tau \psi_\tau)^2 < 0$$

$$G_{xx} = \mathrm{tr}(A_x^2) = (\partial_x \psi_x)^2\,\mathrm{tr}(T^2) = +2(\partial_x \psi_x)^2 > 0$$

Lorentzian signature $(-,+)$ emerges from $i^2 = -1$. The metric-preserving subgroup is the Lorentz group $\mathrm{SO}(1,1)$; in four dimensions, $\mathrm{SO}(1,3)$ is a subgroup of $\mathrm{GL}(4, \mathbb{C})$. Beliefs remain real Gaussians throughout --- the imaginary structure lives in the gauge connection, not in the probability distributions.

---

## 3. The Variational Free Energy

### 3.1 Complete Functional

$$\boxed{S[\{q_i\}, \{p_i\}, \{s_i\}, \{\Omega_i\}, \{\widetilde{\Omega}_i\}] = \underbrace{\sum_i \alpha_i\, D_{\mathrm{KL}}(q_i \| p_i)}_{\text{T1: self-consistency}} + \underbrace{\sum_{ij} \beta_{ij}\, D_{\mathrm{KL}}(q_i \| \Omega_{ij}[q_j])}_{\text{T2: belief alignment}} + \underbrace{\sum_{ij} \gamma_{ij}\, D_{\mathrm{KL}}(s_i \| \widetilde{\Omega}_{ij}[s_j])}_{\text{T3: model alignment}}}$$

All integrals carry support-weighted volume forms $\int \chi_i(c) \cdots \sqrt{|g|}\, dc$ when the base manifold is non-trivial. The observation term is absorbed into T2 via environmental agents.

### 3.2 Attention from Mixture-of-Sources

Each agent $i$ posits its belief was drawn from one of $N$ source agents via categorical latent $z$:

$$P(k, z) = P(k \mid z)\, P(z), \qquad P(k \mid z{=}j) = \Omega_{ij}[q_j]$$

Mean-field factorization $Q(k,z) = q_i(k)\,\beta(z)$ and minimizing the alignment free energy yields:

$$\beta_{ik} = \frac{\pi_k \exp(-E_{ik}/\tau)}{\sum_m \pi_m \exp(-E_{im}/\tau)}, \qquad E_{ij} = D_{\mathrm{KL}}\bigl(q_i \| \Omega_{ij}[q_j]\bigr)$$

This is not an architectural choice. It is the unique solution to constrained free energy minimization. Non-uniform attention priors $\pi_j$ recover causal masking ($\pi_j = 0$ for $j > i$), ALiBi ($\pi_j \propto e^{-m|i-j|}$), and sliding-window attention as special cases.

### 3.3 Adaptive Precision

The self-coupling weight $\alpha_i$ is promoted to a variational parameter with log-barrier regularization:

$$\alpha_i^* = \frac{c_0}{b_0 + D_{\mathrm{KL}}(q_i \| p_i)}$$

When beliefs match priors ($D \approx 0$): $\alpha_i \approx c_0/b_0$ (trusts prior). When beliefs are distant from priors ($D$ large): $\alpha_i \to 0$ (prior released, attention dominates). This implements context-dependent precision gating without neural network components.

### 3.4 Uniqueness of Forward KL

Among the class of local $f$-divergences with linear coupling and exponential-family closure, the forward KL is uniquely determined. The reverse KL breaks log-linearity of the stationary solution; symmetric divergences mix both modes. This is a conditional uniqueness result: given the structural constraints of the multi-agent VFE framework, forward KL is the only consistent choice.

---

## 4. Mass and Inertia

### 4.1 The Mass Matrix

The mass matrix is the Hessian of the free energy with respect to belief means:

$$\boxed{M_i = \underbrace{\bar{\Lambda}_{p_i}}_{\text{bare mass}} + \underbrace{\Lambda_{o_i}}_{\text{sensory mass}} + \underbrace{\sum_k \beta_{ik}\,\widetilde{\Lambda}_{q_k}}_{\text{incoming relational mass}} + \underbrace{\sum_j \beta_{ji}\, \Lambda_{q_i}}_{\text{outgoing recoil mass}}}$$

where $\Lambda = \Sigma^{-1}$ denotes precision, $\widetilde{\Lambda}_{q_k} = \Omega_{ik}\Lambda_{q_k}\Omega_{ik}^\top$ is gauge-transported precision, and $\bar{\Lambda}_{p_i} = \Sigma_{p_i}^{-1}$ is the prior precision.

The off-diagonal blocks encode inter-agent mass coupling:

$$[M^\mu]_{ik} = -\beta_{ik}\,\Omega_{ik}\Lambda_{q_k} - \beta_{ki}\,\Lambda_{q_i}\,\Omega_{ki}^\top \qquad (i \neq k)$$

The covariance sector:

$$[M^\Sigma]_{ii} = \tfrac{1}{2}(\Lambda_{q_i} \otimes \Lambda_{q_i})\,\bigl(1 + \textstyle\sum_k \beta_{ik} + \sum_j \beta_{ji}\bigr)$$

### 4.2 Physical Interpretation

Mass is precision. Confident agents are massive and resist perturbation. Uncertain agents are light and responsive. This maps onto physical intuition: a rock (spatial position known to $\sim 10^{-10}$ m, precision $\sim 10^{20}$) is massive; a quantum particle in superposition (large $\Sigma$) is light.

The four mass components have distinct origins:

- **Bare mass** ($\bar{\Lambda}_p$): resistance from deep prior expectations, present even in isolation
- **Sensory mass** ($\Lambda_o$): anchoring from observations / environmental agents
- **Incoming relational mass** ($\sum_k \beta_{ik}\widetilde{\Lambda}_{q_k}$): inertia from coupling to confident neighbors
- **Outgoing recoil mass** ($\sum_j \beta_{ji}\Lambda_{q_i}$): resistance from being attended-to by others (Newton's third law analog: influencing others rigidifies the influencer)

### 4.3 Computational Validation

The mass-precision relationship $M_{\mathrm{eff}} = (0.23/\sigma^2) + 0.02$ was validated with $R^2 = 0.9998$. Harmonic oscillator frequency scaling $\omega^2 = k/M$ confirmed. Energy conservation under symplectic integration held to $\pm 0.01\%$.

---

## 5. Dynamics: First and Second Order

### 5.1 First-Order (Overdamped) Dynamics

The standard VFE gradient flow (natural gradient descent on the product manifold):

$$\dot{\mu}_{q,i} = -\eta_\mu\, \Sigma_{q,i}\, \nabla_{\mu_{q,i}} S \qquad \text{(perception)}$$

$$\dot{\mu}_{p,i} = -\eta_p\, \Sigma_{p,i}\, \nabla_{\mu_{p,i}} S \qquad \text{(expectation learning)}$$

$$\dot{\Omega}_i = -\eta_\Omega\, \nabla_{\Omega_i} S \qquad \text{(gauge frame adjustment = action?)}$$

$$\dot{s}_i = -\epsilon\,\eta_s\, \Sigma_{s,i}\, \nabla_{\mu_{s,i}} S \qquad \text{(model evolution, slow)}$$

Covariance dynamics handled via Cholesky parameterization: autograd through $L_q$ captures the full Wishart-geometry gradient.

This is standard active inference without the "active" part. Agents respond to the current VFE landscape via steepest descent. No policies, no counterfactual evaluation, no planning horizon.

### 5.2 Second-Order (Hamiltonian/Underdamped) Dynamics

The Hamiltonian formulation introduces momenta conjugate to belief parameters:

$$H = \underbrace{\frac{1}{2}\dot{\mu}^\top M_{\mu\mu}\, \dot{\mu} + \frac{1}{2}\mathrm{tr}\bigl[M_{\Sigma\Sigma}[\dot{\Sigma}, \dot{\Sigma}]\bigr] + \frac{1}{2}\langle\dot{\phi}, \dot{\phi}\rangle_{\mathfrak{g}}}_{\text{kinetic energy } T} + \underbrace{S[\mu, \Sigma, \phi]}_{\text{potential energy } V = \text{VFE}}$$

Hamilton's equations:

$$\dot{\mu}_i = \sum_k [M^{-1}]_{ik}^{\mu\mu}\, \pi_k^\mu$$

$$\dot{\pi}_i^\mu = -\frac{\partial S}{\partial \mu_i} - \frac{1}{2}\pi^\top \frac{\partial M^{-1}}{\partial \mu_i}\pi$$

The force decomposes into four terms:

$$-\frac{\partial S}{\partial \mu_i} = \underbrace{-\bar{\Lambda}_{p_i}(\mu_i - \bar{\mu}_i)}_{\text{prior restoring}} \underbrace{- \sum_k \beta_{ik}\widetilde{\Lambda}_{q_k}(\mu_i - \widetilde{\mu}_k)}_{\text{consensus}} \underbrace{- \sum_j \beta_{ji}\Lambda_{q_i}\,\Omega_{ji}^\top(\widetilde{\mu}_i^{(j)} - \mu_j)}_{\text{reciprocal}} \underbrace{- \Lambda_{o_i}(\mu_i - o_i)}_{\text{sensory}}$$

With damping $\gamma_i$, the equation of motion becomes:

$$M_i\,\ddot{\mu}_i + \gamma_i\,\dot{\mu}_i + \nabla_{\mu_i} S = 0$$

Three dynamical regimes determined by the discriminant $\Delta = \gamma_i^2 - 4 K_i M_i$ (where $K_i = \nabla^2 S|_{\mu^*}$ is the local curvature):

| Regime | Condition | Behavior |
|---|---|---|
| Overdamped | $\Delta > 0$ | Monotonic relaxation. Standard Bayesian updating. |
| Critically damped | $\Delta = 0$ | Fastest equilibration without oscillation. |
| Underdamped | $\Delta < 0$ | Oscillatory approach with overshoot. Belief perseverance, resonance. |

### 5.3 The Choice Between First and Second Order

**Open question.** First-order dynamics (overdamped VFE) is computationally tractable and recovers standard variational inference. Second-order dynamics (Hamiltonian) produces richer phenomena --- oscillation, resonance, momentum transfer, belief perseverance --- and provides the mass interpretation. The physical world appears to be second-order (Newton's laws, not Aristotelian friction).

The relationship between the two: first-order is the $\gamma \to \infty$ (infinite damping) limit of second-order. Standard active inference and standard transformers operate in this limit. The second-order theory is strictly more general.

**What determines the damping?** In the belief inertia manuscript, $\gamma_i$ is a free parameter (the "learning rate" in inverse). A deeper theory would derive $\gamma$ from the agent's coupling structure or from the model fiber. One possibility: $\gamma_i$ is the entropy production rate of agent $i$, connecting damping to thermodynamic irreversibility. This is not yet established.

---

## 6. Agent-Dependent Time

### 6.1 Proper Time as Information Distance

Each agent experiences its own proper time, measured by the Fisher-Rao arc length along its belief trajectory:

$$d\tau_i = \sqrt{d\mu_i^\top\, \Sigma_{q,i}^{-1}\, d\mu_i + \tfrac{1}{2}\,\mathrm{tr}\bigl(\Sigma_{q,i}^{-1}\, d\Sigma_{q,i}\, \Sigma_{q,i}^{-1}\, d\Sigma_{q,i}\bigr)}$$

This is the second-order approximation to $D_{\mathrm{KL}}(q_i^{\text{new}} \| q_i^{\text{old}})$.

### 6.2 Scale-Dependent Temporal Resolution

A scale-$\zeta$ agent "ticks" when its belief has changed by a scale-appropriate amount of information:

$$\Delta\tau^{(\zeta)} \geq \epsilon_\zeta$$

where $\epsilon_\zeta$ is a threshold that grows with scale. Scale-0 agents (elementary) update per bit. A human-scale meta-agent requires vastly more information to register a "tick" --- a macroscopic "difference that makes a difference."

This creates the timescale hierarchy:

$$\tau_{\text{belief}}^{(0)} < \tau_{\text{prior}}^{(0)} < \tau_{\text{belief}}^{(1)} < \tau_{\text{prior}}^{(1)} < \tau_{\text{belief}}^{(2)} < \cdots$$

The ratio between consecutive scales is controlled by the timescale separation parameter $\epsilon$: $\tau^{(\zeta+1)} \sim \epsilon^{-1} \tau^{(\zeta)}$.

### 6.3 Time Dilation from Precision

For a Gaussian agent with prior precision $\Lambda_p = \Sigma_p^{-1}$:

$$d\tau = \sqrt{d\mu^\top\, \Lambda_p\, d\mu} = \|d\mu\|_{\Lambda_p}$$

High-precision agents (large $\Lambda_p$, large mass) experience more proper time per coordinate displacement $d\mu$. A small belief change for a confident agent corresponds to a large information distance. This is an information-geometric analog of gravitational time dilation: massive objects (high precision) experience time differently from light objects (low precision).

### 6.4 Open Problem: Connecting Agent Time to the Hamiltonian

The Hamiltonian uses coordinate time $t$ as the evolution parameter. Agent-dependent proper time $\tau_i$ is measured along trajectories in belief space. The relationship between these is:

$$\frac{d\tau_i}{dt} = \sqrt{\dot{\mu}_i^\top\, \Lambda_{p,i}\, \dot{\mu}_i}$$

For the Hamiltonian $H = T + V$, the kinetic energy $T = \frac{1}{2}\dot{\mu}^\top M \dot{\mu}$ where $M \approx \Lambda_p$ (the dominant mass term). So $d\tau_i/dt \approx \sqrt{2T_i/M_i}$ --- the rate of proper time depends on kinetic energy relative to mass.

**The missing piece:** A fully covariant formulation where agent-dependent proper time replaces coordinate time as the evolution parameter. This would require a reparameterization-invariant action principle:

$$S_{\text{worldline}} = \int \sqrt{g_{\mathcal{B}}(\dot{q}_i, \dot{q}_i)}\, d\lambda$$

where $\lambda$ is an affine parameter and $g_{\mathcal{B}}$ is the Fisher-Rao metric. The Euler-Lagrange equations of this action would give geodesic motion on the statistical manifold, with the VFE acting as an external potential. This connects to the Hamiltonian formulation but makes the agent-dependence of time manifest.

---

## 7. Induced Geometry: It From Bit

### 7.1 The Pullback Construction

Each agent's belief field $q_i: \mathcal{C} \to \mathcal{B}$ induces a Riemannian metric on the base manifold $\mathcal{C}$ by pulling back the Fisher-Rao metric:

$$G^{(q)}_{i,\mu\nu}(c) = \bigl(\partial_\mu \mu_i\bigr)^\top \Sigma_{q,i}^{-1}\, \bigl(\partial_\nu \mu_i\bigr) + \frac{1}{2}\,\mathrm{tr}\!\bigl(\Sigma_{q,i}^{-1}\, \partial_\mu \Sigma_{q,i}\, \Sigma_{q,i}^{-1}\, \partial_\nu \Sigma_{q,i}\bigr)$$

Similarly for priors: $G^{(p)}_{i,\mu\nu}$ from the prior field $p_i$.

The epistemic metric $G^{(q)}$ fluctuates rapidly (beliefs change fast). The ontological metric $G^{(p)}$ evolves slowly (generative models are stable). Phenomenal spacetime is the ontological metric.

### 7.2 Observer Dependence

Different agents induce different metrics: $G_i \neq G_j$ in general. There is no privileged geometry. Consensus geometry emerges only after gauge-orbit averaging:

$$\bar{G}_{\mu\nu}^{\text{consensus}}(c) = \sum_i w_i(c)\, G_{i,\mu\nu}(c)$$

### 7.3 Dimensional Sectors

Eigenvalue decomposition of the induced metric $G_{i,\mu\nu}$ reveals three sectors:

$$\lambda_{\text{obs}} \gg \lambda_{\text{dark}} \gg \lambda_{\text{internal}} \approx 0$$

The **observable sector** ($\sim$4 eigenvalues for physical spacetime) carries most information flux. The **internal sector** ($\sim K - 4$ dimensions) has near-zero eigenvalues and is imperceptible. The vast statistical manifold remains invisible; phenomenal spacetime is a thin slice.

### 7.4 Lorentzian Structure

With real gauge frames, induced metrics are Riemannian (positive-definite). The $\mathrm{GL}(K, \mathbb{C})$ extension resolves this: complex temporal gauge components produce negative eigenvalues via $i^2 = -1$, yielding Lorentzian signature without imposing it (Section 2.5).

---

## 8. Hierarchical Structure

### 8.1 Species and Coalitions

Two independent structures govern meta-agent formation:

**Species** (model fiber, slow): Agents sharing generative models form species groups. The species gate is:
$$S_{i\sigma}(x) = \sigma\!\bigl(-D_{\mathrm{KL}}(s_i(x) \| \widetilde{\Omega}_{i\sigma}[s_\sigma(x)]) / \tau_{\text{species}}\bigr) \cdot \chi_i(x)$$

**Coalitions** (belief fiber, fast): Agents agreeing on current state form dynamic coalitions:
$$C_{i\alpha}(x) = \sigma\!\bigl(-D_{\mathrm{KL}}(q_i(x) \| \Omega_{i\alpha}[q_\alpha(x)]) / \tau_{\text{belief}}\bigr) \cdot \chi_i(x)$$

**Selection rule:** $W_{i\alpha}(x) = S_{i\alpha}(x) \cdot C_{i\alpha}(x)$. You can only coordinate with agents sharing your model. Species gates meta-agent formation.

### 8.2 Precision Pooling

Meta-agent beliefs form via gauge-covariant precision-weighted averaging:

$$\Lambda_\alpha = \sum_i w_i\, \Omega_{\alpha i}\, \Lambda_i\, \Omega_{\alpha i}^\top \qquad \text{(precisions add)}$$

$$\mu_\alpha = \Lambda_\alpha^{-1} \sum_i w_i\, \Omega_{\alpha i}\, \Lambda_i\, \mu_i \qquad \text{(precision-weighted mean)}$$

More certain agents dominate. This is the information-geometrically correct aggregation.

### 8.3 The Ouroboros Tower

Bidirectional information flow across scales:

- **Bottom-up:** Constituent beliefs $\to$ precision pooling $\to$ meta-agent state
- **Top-down:** Meta-agent belief $\to$ prior for constituents: $p_i^{(\zeta)} \leftarrow \Omega_{i,I}[q_I^{(\zeta+1)}]$
- **Multi-generation:** Hyperprior propagation with exponential decay: weight $= \gamma^{\Delta\zeta}$ for ancestor $\Delta\zeta$ scales above
- **Self-referential closure:** Top-scale agents observe entire system, forming self-consistent priors

This creates Wheeler's "self-excited circuit": the system observes itself, forms collective priors, which shape individual beliefs, which change the collective state, which gets re-observed.

### 8.4 Renormalization Group

Meta-agent hierarchy IS real-space RG. At each scale $\zeta$, effective coupling constants are measured:

$$g^{(\zeta)} = \bigl(\langle D_{\mathrm{KL}}(q_i \| p_i)\rangle,\; \langle E_{ij}\rangle,\; H[\beta],\; \xi,\; \ldots\bigr)$$

The beta function $\beta(g) = dg/d(\ln \zeta)$ governs flow. Fixed points $\beta(g^*) = 0$ define scale-invariant theories. Linearization around fixed points yields critical exponents determining universality class.

---

## 9. Emergent Active Inference (The Super-Deterministic Hypothesis)

### 9.1 The Claim

Active inference --- with policies, expected free energy, epistemic and pragmatic value --- emerges as a coarse-grained description of deterministic VFE gradient flow at lower scales. No explicit planning mechanism is needed.

### 9.2 The Argument

At the constituent level, dynamics are fully determined by initial conditions plus $S$:

$$\dot{\xi}_i = -\eta\, G^{-1}\, \nabla_{\xi_i} S$$

where $\xi = (\mu, \Sigma, \phi, s, r, \ldots)$ and $G$ is the Fisher information metric. No stochasticity, no choices, no counterfactuals.

From the meta-agent's perspective, the Ouroboros loop creates what looks like perception-action cycles:

1. Constituents update beliefs $\to$ meta-agent state changes (precision pooling = "perception")
2. Meta-agent state propagates priors to constituents (top-down feedback = "action")
3. Constituents respond $\to$ meta-agent changes $\to$ repeat

When you coarse-grain over the constituents and describe only the meta-agent's effective state, the dynamics can satisfy the active inference equations even though the micro-level dynamics have no such structure.

### 9.3 How the Standard Active Inference Components Emerge

**Pragmatic value** (achieving preferences): The prior $p_i$ encodes preferred states. Self-coupling $\alpha_i D_{\mathrm{KL}}(q_i \| p_i)$ acts as an attractor in belief space. When gauge frame updates $\dot{\phi}_i$ change how agent $i$ appears to neighbors, the agent effectively "acts on the world" to reduce the discrepancy between beliefs and preferences. The gradient points toward the preferred state without explicit goal representation.

**Epistemic value** (uncertainty reduction): Covariance dynamics drive $\Sigma_i$ down. High-uncertainty agents have low mass and are pulled toward informative neighbors via the alignment coupling. This resembles information-seeking behavior without an explicit epistemic drive.

**Policy selection**: In continuous time with infinitesimal actions, policy evaluation collapses to gradient computation. If the action space is the tangent space $\{d\phi_i, d\mu_i, d\Sigma_i\}$ and the objective is $S$, the gradient selects the unique locally optimal action. No counterfactual branching needed.

**Temporal depth from timescale hierarchy**: Scale $\ell$ effectively averages over $\epsilon^{-\ell}$ fast-timescale steps. The top-level meta-agent's dynamics implicitly integrate over an exponentially long horizon, providing the planning depth that local gradient descent lacks.

### 9.4 The Super-Determinism Connection

In Bell's theorem, super-determinism says measurement "choices" are correlated with hidden variables through shared past causes. The analog: an agent's gauge frame update is correlated with all other agents' states through the shared VFE potential. There are no independent decisions.

The apparent purposiveness at the meta-agent level is the projection of high-dimensional deterministic flow onto a low-dimensional coarse-grained description. "Choice" is an artifact of coarse-graining, not a fundamental feature.

### 9.5 Where This Might Fall Short

**Local minima.** Gradient descent is myopic. An agent needing to temporarily increase free energy to reach a much lower-energy state cannot do so via gradient flow alone. The Hamiltonian (second-order) formulation partially addresses this: momentum can carry beliefs through energy barriers (belief overshoot). But it is unclear whether this fully substitutes for explicit planning.

**Genuine novelty.** Gradient flow on a fixed VFE landscape cannot anticipate qualitatively new types of interactions. Meta-agent formation via consensus detection is reactive (detecting existing alignment), not proactive (creating alignment that doesn't yet exist).

**Formal derivation.** The claim that active inference emerges from VFE gradient flow is currently an argument, not a theorem. The formal result would be: coarse-graining the hierarchical VFE dynamics to meta-agent level produces equations satisfying the active inference schema with identifiable expected free energy $G(\pi)$. This has not been proven.

---

## 10. Connection to Transformers

The V10.0 Gauge Transformer implements the 0-dimensional, single-fiber, flat-bundle, overdamped, adiabatic limit of this framework:

| Full Theory | Transformer Limit |
|---|---|
| Base manifold $\mathcal{C}$ with geometry | Single point $c^*$ (0D) |
| Two fibers (belief + model) | One fiber (belief only; model = embeddings) |
| Non-flat gauge connection | Flat (cocycle holds) |
| Hamiltonian dynamics | Overdamped (natural gradient E-step) |
| Agent-dependent time | Coordinate time (training steps) |
| Induced spacetime metric | Not computed |
| Hierarchical meta-agents | Single scale |
| $\mathrm{GL}(K, \mathbb{C})$ | $\mathrm{GL}(K, \mathbb{R})$ |
| Environmental agents | Explicit observation likelihood |

Three successive limits recover standard dot-product attention from the VFE:

1. **Isotropic covariances** $\Sigma_i = \sigma^2 I$: KL reduces to scaled Euclidean distance + geometric bias $S(\Omega)$
2. **Constant gauge** $\Omega_{ij} \to \Omega$ (position-independent): no curvature; $S(\Omega)$ becomes shared constant
3. **Absorption into projections** $W_Q W_K^\top = \sigma^{-2}\Omega^{-\top}$: recovers $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$

The Boltzmann gate (softmax coupling term $\partial\beta/\partial\theta$) provides the nonlinearity that replaces MLP activations. It is the derivative of attention with respect to beliefs --- the unique nonlinearity that the VFE framework generates.

---

## 11. What Is Established vs What Is Missing

### Established (mathematical proofs or computational validation)

- GL(K) invariance of KL divergence (algebraic identity)
- Softmax attention from mixture-of-sources VFE minimization (variational calculus)
- Forward KL uniqueness under exponential-family closure (conditional uniqueness theorem)
- Mass = Hessian of free energy = sum of four precision terms (direct computation)
- Mass-precision correlation $R^2 = 0.9998$ (simulation)
- Harmonic oscillator frequency from epistemic mass (simulation)
- Energy conservation under symplectic integration to $\pm 0.01\%$ (simulation)
- BERT 144-head KL-attention correlation $\bar{r} = 0.821$ (empirical validation on pretrained model)
- GL(K) language model PPL = 71.6 on WikiText-103 (training experiment)
- Environmental agent equivalence (change of variables preserving gradients)
- Lorentzian signature from GL(K, $\mathbb{C}$) (worked example, not yet trained)
- Hierarchical emergence via consensus detection (simulation up to 25 scales)

### Missing: Critical Gaps

**Gap 1: Dimensional analysis.** The framework measures everything in bits (nats). Converting to physical units (kilograms, meters, seconds) requires identifying dimensional anchors. The relationship $M \propto \Sigma^{-1}$ gives mass in units of inverse-variance, not kilograms. What sets the conversion factor? This is the central unsolved problem for any physical interpretation.

**Gap 2: Deriving the Hamiltonian from agent dynamics.** The mass matrix $M = \nabla^2 S$ is well-defined, but the kinetic energy $T = \frac{1}{2}\dot{\mu}^\top M \dot{\mu}$ is postulated by analogy with classical mechanics, not derived from the agent ontology. Why should agents have momentum? One route: the Ouroboros feedback loop between scales creates effective inertia --- meta-agent priors resist change, and constituents responding to those priors generate momentum-like behavior. This needs to be made precise.

**Gap 3: The damping coefficient.** In $M\ddot{\mu} + \gamma\dot{\mu} + \nabla S = 0$, what determines $\gamma$? Is it a free parameter, or does it emerge from the coupling structure? Possible answers: (a) $\gamma$ is the rate of entropy production, connecting to thermodynamic irreversibility; (b) $\gamma$ emerges from averaging over fast degrees of freedom (adiabatic elimination of fast-fiber dynamics); (c) $\gamma$ is determined by the model fiber through the timescale separation parameter $\epsilon$. None of these is worked out.

**Gap 4: Agent-dependent time in the Hamiltonian.** The proper time $d\tau_i = \sqrt{d\mu^\top \Lambda_p\, d\mu}$ and the Hamiltonian coordinate time $t$ coexist but are not unified. A covariant formulation would use a worldline action $S = \int \sqrt{g_{\mathcal{B}}(\dot{q}, \dot{q})}\, d\lambda$ with the VFE as an external potential. The Euler-Lagrange equations would give geodesic motion perturbed by the VFE gradient, with proper time emerging as the natural parameter. This has not been derived.

**Gap 5: Formal derivation of emergent active inference.** The argument that active inference emerges from VFE gradient flow is heuristic. A formal result would show that coarse-graining the hierarchical VFE dynamics reproduces the expected free energy functional $G(\pi)$ at the meta-agent level. The ingredients (models of others via model fiber, preferences via priors, actions via gauge frame updates, temporal depth via timescale hierarchy) are present, but the derivation is absent.

**Gap 6: Quantum extension.** The framework uses classical probability distributions (Gaussians). Quantum mechanics requires density matrices, which live on a different statistical manifold (the space of positive semidefinite trace-one operators with the Bures-Wasserstein metric). The gauge group would need to act on density matrices rather than probability distributions. The KL divergence would be replaced by quantum relative entropy $S(\rho \| \sigma) = \mathrm{tr}(\rho \log \rho - \rho \log \sigma)$.

**Gap 7: Falsifiability.** The framework is compatible with any measurement by reinterpreting it as information geometry. This is both a strength (generality) and a weakness (unfalsifiability). Specific quantitative predictions that could be wrong are needed. The belief inertia manuscript provides some (overshoot scaling, resonance frequency), but these are for the social dynamics application, not the physical theory.

**Gap 8: Computational scaling.** The framework has been tested up to $\sim$200 agents and 25 scales. Physical systems involve $\sim 10^{80}$ particles. How the RG flow connects micro-scale agent dynamics to macro-scale physics across 80+ orders of magnitude is entirely open.

---

## 12. The Path Forward: Connecting the Pieces

The following sequence of results would, if established, connect the existing pieces into a coherent theory:

**Step 1.** Derive the Hamiltonian structure from the multi-scale VFE. Show that the Ouroboros feedback loop between scales $\zeta$ and $\zeta+1$ generates effective second-order dynamics at scale $\zeta$ when scale $\zeta+1$ is integrated out. The meta-agent prior acts as an inertial term because it resists change on the slow timescale. This would derive the mass matrix rather than postulating it.

**Step 2.** Derive the damping coefficient $\gamma$ from the model fiber. Show that the timescale separation $\epsilon$ between belief and model dynamics produces an effective friction $\gamma \propto \epsilon^{-1}$ when the model fiber is adiabatically eliminated. This would connect the overdamped/underdamped distinction to the speed of model learning.

**Step 3.** Formulate the covariant worldline action. Replace coordinate time with proper time via the reparameterization-invariant action $S_{\text{wl}} = \int \bigl[\frac{1}{2}g_{\mathcal{B}}(\dot{q}, \dot{q}) + V(q)\bigr] d\tau$ where $V$ is the VFE potential. Derive geodesic equations perturbed by VFE forces. Show that the scale-dependent temporal resolution $\epsilon_\zeta$ emerges from the geodesic length per RG scale.

**Step 4.** Derive active inference from coarse-graining. Take the full hierarchical VFE dynamics (first- or second-order), coarse-grain to meta-agent level, and show that the effective dynamics satisfy the active inference equations with identifiable expected free energy, pragmatic value, and epistemic value. The gauge frame update should map onto the "action" in active inference; the model fiber should map onto the "generative model."

**Step 5.** Connect the dimensional analysis. Identify a physical system (possibly a simple harmonic oscillator or hydrogen atom) where the information-geometric quantities (precision, Fisher distance, VFE) can be mapped to measured physical quantities (mass, length, energy) via specific dimensional anchors. This would fix the conversion factors and make the theory falsifiable for that system.

Each of these is a substantial theoretical project. Together they would transform the framework from a collection of compelling analogies and partial results into a unified predictive theory.
