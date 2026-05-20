# Evidence Pack — rock-example-meta-agent

Neutral fact pack. No editorial framing.

## Manuscript anchor

Primary anchor: `Attention/Participatory_it_from_bit.tex` lines 3837–3851.

### Section content (line by line)

| Line | Element |
|---|---|
| 3837 | `\subsection{Macroscopic Objects as Meta-Agents: The Rock Example}` |
| 3839 | Framing: "In our formalism, a rock is not a primitive entity but an emergent meta-agent with hierarchical structure." |
| 3841 | Atomic constituents: "$\sim 10^{23}$ atomic constituents. Each atom functions as a scale-0 agent maintaining beliefs about its local environment - primarily the positions and momenta of neighboring atoms. Atoms in a solid are strongly coupled to their neighbors through electromagnetic interactions, which in our framework manifests as large attention weights $\beta_{\text{atom,neighbor}} \approx 1$. This strong coupling drives belief alignment: atoms in a crystalline lattice maintain highly coherent beliefs about their relative spatial configuration." |
| 3843 | Meta-agent emergence: "When these atomic agents achieve sufficient coherence (small KL divergences between transported beliefs), they form a scale-1 meta-agent - the rock itself. This meta-agent possesses collective belief $q_{\text{rock}}$ representing the consensus configuration of constituent atoms, collective prior $p_{\text{rock}}$ encoding the stable lattice structure characteristic of the material, and collective gauge frame $\phi_{\text{rock}}$ representing the rock's overall internal orientation. The rock's phenomenal geometry - what it would 'experience' if it possessed the information integration necessary for experience - is the metric $G_{\text{rock}}$ induced through pullback from its collective belief field." |
| 3845 | Inertia / slow dynamics: "The rock exhibits slow dynamics characteristic of massive objects. Its Fisher information is large (sharp, confident beliefs about spatial configuration), making belief updates resistant to perturbation. The rock's proper time $\tau_{\text{rock}}$ advances slowly because information updates occur infrequently - the system is near equilibrium. This formalizes the intuition that massive objects have inertia and resist changes in state." |
| 3847 | Photon mediation: "When a human observes the rock, the coupling $\beta_{\text{human,rock}}$ becomes non-zero. This coupling is not direct but mediated by photon agents - electromagnetic quanta that couple strongly to both rock-atoms (scattering) and human-photoreceptors (absorption). The photons function as information carriers, transporting beliefs about the rock's configuration to the human observer. The human agent's beliefs $q_{\text{human}}$ update to align with the transported rock beliefs $\Omega_{\text{human,rock}}[q_{\text{rock}}]$ through free energy minimization. This process is phenomenologically experienced as 'seeing the rock,' but it is simply agent-agent information coupling mediated by intermediate agents." |
| 3849 | Bidirectionality and quantum analog: "observation is bidirectional. The rock's state also updates slightly in response to coupling with the human - or more precisely, in response to the photons that scattered from it. For macroscopic rocks, this perturbation is negligible: $\Delta q_{\text{rock}} \approx 0$ because the rock's large Fisher information (high mass) resists updates. For microscopic quantum particles with broad wavefunctions (low Fisher information, high uncertainty), the perturbation can be significant: $\Delta q_{\text{particle}}$ may be large, corresponding to quantum measurement back-action. The classical-quantum distinction emerges from the magnitude of Fisher information rather than from qualitatively different dynamics." |
| 3851 | Four principles: "(1) observation is not a special process but ordinary agent-agent coupling. (2) the observer-observed distinction is conventional rather than ontological - both are agents coupled through information exchange. (3) macroscopic objects emerge as meta-agents from microscopic constituents through consensus formation. (4) classical behavior (definite states, negligible back-action) and quantum behavior (superposition, significant back-action) reflect different regimes of the same information-geometric dynamics." |

### Cross-document anchors

- §`sec:cognitive_first` at line 103: the framework's pan-agentic commitment. Line 119: "Every system at every scale is treated as an agent with its own gauge frame: an electron is a scale-0 agent, a molecule a higher-scale meta-agent assembled from atomic constituents, a brain a yet-higher-scale meta-agent assembled from cellular constituents, and so on through ecosystems and societies. There are no non-agent physical things in the framework." Plus: "the framework does not require panpsychism in the metaphysical sense: phenomenal properties are not ascribed to scale-0 electrons. What is ascribed is gauge-frame structure plus participation in variational dynamics, with the question of how phenomenal properties arise upon aggregation across scales left open as a research item."

- §`sec:agent_definition` at line 613: agent definition. Each agent needs primitive $q_i$, $s_i$, derived $p_i = \Omega_{i,I}[q_I^{(s+1)}]$, derived $r_i$, gauge frame $\phi_i$. Hierarchy $r_i \to s_i \to p_i \to q_i \to \text{observations}$.

- §`sec:meta_agent_variational` at line 2060: variational criterion for meta-agent formation, Eq. `eq:meta_agent_FE_criterion`: $\mathcal{F}^*[q_I, p_I, s_I, r_I, \{q_i, s_i\}] + C(I) < \mathcal{F}^*[\{q_i, p_i, s_i, r_i\}]$.

- §`sec:meta_agent_threshold`: threshold-based detector (approximates the variational rule).

- §`sec:mass` at line 1843: precision-as-stiffness "Mass Analogy" subsection, where (after this morning's RED_WINS verdict edits) the framing is "postulate-dependent mass-like scaling" with TODO for the empirical test. The rock example at line 3845 says "Fisher information is large … making belief updates resistant to perturbation … This formalizes the intuition that massive objects have inertia," which evokes §`sec:mass`.

- §`sec:scope_limitations` at line 134, esp. line 146: "**No quantum extension.** A rigorous quantum version of the framework does not currently exist; the structural parallels with QBism and other relational interpretations are suggestive rather than derived."

- §`sec:pullback`: pullback-induced geometry from belief fields. The "metric $G_{\text{rock}}$ induced through pullback" reference at line 3843 inherits the pullback's caveats. Line 142 (scope-limitations): "the gauge-orbit-averaged consensus metric used as the framework's candidate gauge-invariant geometry is also conditional on a regulator that we do not construct."

- §`app:fisher_gaussian` at line 3866: the Fisher block-diagonal form $F = \mathrm{diag}(\Sigma^{-1}, \tfrac12 \Sigma^{-1}\otimes\Sigma^{-1})$ and the natural-gradient formulas $\tilde\nabla_\mu \mathcal{F} = \Sigma\nabla_\mu\mathcal{F}$, $\tilde\nabla_\Sigma\mathcal{F} = 2\Sigma(\nabla_\Sigma\mathcal{F})\Sigma$.

- Earlier rock illustration at line 2023 (in §`sec:mass`): "A macroscopic object maintains extraordinarily precise self-localization: $\bar\Sigma_{\text{rock}} \approx \epsilon I$ with $\epsilon \ll 1$ a small dimensionless precision (we make no claim that $\epsilon$ corresponds to specific physical units; cf. the dimensional caveat in §`sec:scope_limitations`) yields a large effective stiffness $M_{\text{rock}} \sim \epsilon^{-1}$ … Rocks are certain, thus exhibit large second-variation rigidity, thus are hard to move under the dynamics. We do not extend this analogy to quantum-mechanical scenarios: spatial delocalization of a quantum particle does not imply lower inertial mass in standard quantum mechanics, and the framework's effective mass is a configuration-space stiffness that we are not equating with physical inertial mass here." Note: line 3849 in the Rock Example DOES extend the analogy to quantum-mechanical scenarios — direct tension with line 2023.

- §3160 (in the phenomenological interpretation section, line ~3155): "Each observer assigns a different scalar effective mass via $m_{\text{eff}} = \mathrm{tr}(\Sigma_{p,i}^{-1})/K$ (proportional to the trace of their prior precision in the isolated-agent limit). The particle has no observer-independent mass prior to consensus formation."

- "Measurement is the dynamical process by which an apparatus (itself a high-precision macroscopic system) couples to the particle and dominates the free energy landscape, forcing all observers into agreement. What physics calls 'wavefunction collapse' is modeled by the transition from pre-consensus (perspectival, observer-dependent) to post-consensus (shared, objective) physics." (line ~3163). Plus disclaimer at line 3166: "The analogy is suggestive only. The framework contains no quantum-mechanical formalism — no Hilbert space, no Born rule, no superposition states — so it cannot derive the measurement problem's resolution."

### Sympy-tractable claims for blue derivation

For Gaussian agents with natural-gradient flow $\dot\mu = -\eta_q \Sigma_i \nabla_{\mu_i}\mathcal{F}$ and bare-mass term plus single-coupling test:

**B1 (inertia from precision).** $\mathcal{F}_i = \mathrm{KL}(q_i\|p_i) + \beta_{ij}\mathrm{KL}(q_i\|\Omega_{ij}q_j)$ with Gaussian $q_i = \mathcal{N}(\mu_i, \Sigma_i)$, $p_i = \mathcal{N}(\bar\mu_i, \bar\Sigma_i)$. Gradient $\nabla_{\mu_i}\mathcal{F} = \bar\Lambda_i(\mu_i - \bar\mu_i) + \beta_{ij}\tilde\Lambda_{q_j}(\mu_i - \tilde\mu_j)$ (canonical, matches Eq. `eq:total_gradient_mu`). Natural step:
$$
\dot\mu_i = -\eta_q \Sigma_i \nabla_{\mu_i}\mathcal{F} = -\eta_q \Sigma_i [\bar\Lambda_i(\mu_i - \bar\mu_i) + \beta_{ij}\tilde\Lambda_{q_j}(\mu_i - \tilde\mu_j)].
$$
For rock with $\bar\Sigma_i = \Sigma_i = \epsilon I$ small, $\bar\Lambda_i = \Lambda_i = \epsilon^{-1}I$, and $\beta_{ij}\tilde\Lambda_{q_j} = O(1)$:
$$
\dot\mu_i = -\eta_q \epsilon \cdot [\epsilon^{-1}(\mu_i - \bar\mu_i) + O(1)(\mu_i - \tilde\mu_j)] = -\eta_q [(\mu_i - \bar\mu_i) + \epsilon \cdot O(1)(\mu_i - \tilde\mu_j)].
$$
Pull from coupling is suppressed by $\epsilon$; only the prior anchoring survives. As $\epsilon \to 0$, rock pinned to its own prior $\bar\mu_i$ — does not update toward a low-precision observer's belief. **This is the inertia result.**

**B2 (cross-observer consensus via photon mediation).** Symbolically: human $h_1, h_2$ at low precision $\Lambda_h$ couple to photon agents $\gamma$ which were emitted/scattered from rock atoms. Photon emission/scattering imprints rock's mean configuration onto photon belief: $\mu_\gamma \approx \mu_{\text{rock}}$ (under gauge transport $\Omega_{\gamma,\text{rock}}$). Human $h_i$ updates: $\dot\mu_{h_i} = -\eta_q \Sigma_{h_i} \beta_{h_i,\gamma}\tilde\Lambda_\gamma (\mu_{h_i} - \tilde\mu_\gamma)$. At equilibrium $\mu_{h_i} \to \tilde\mu_\gamma \to \tilde\mu_{\text{rock}}$. Both observers reach the same equilibrium up to gauge frame: $\mu_{h_1} = \Omega_{h_1,\text{rock}}\mu_{\text{rock}}$, $\mu_{h_2} = \Omega_{h_2,\text{rock}}\mu_{\text{rock}}$, both observing the same noumenal $\mu_{\text{rock}}$. **This is the cross-observer agreement result.**

**B3 (back-action vanishing).** Rock $i$ in coupling to a single human $h$: $\dot\mu_i = -\eta_q \Sigma_i \beta_{i,h}\tilde\Lambda_h(\mu_i - \tilde\mu_h) = O(\Sigma_i) = O(\epsilon)$ as $\Sigma_i \to 0$. Covariance: $\dot\Sigma_i \propto \Sigma_i(\Lambda_i - \tilde\Lambda_h)\Sigma_i$ which for $\Lambda_i \gg \tilde\Lambda_h$ scales as $-\Sigma_i \Lambda_i \Sigma_i + O(\Sigma_i^2 \Lambda_h) = O(\Sigma_i) + O(\Sigma_i^2)$, both vanishing as $\epsilon \to 0$. **This is the back-action result.**

These three derivations are within reach of sympy or hand computation and constitute the explicit derivation the user asked the blue team to produce.

## Canon excerpts

From `external_canon_inference.md`:
- Standard FEP [Friston2010]: natural gradient on F, precision-weighted updates. The Fisher metric as preconditioner. The framework's "high precision ⇒ slow update" is canonical [Amari 1998].

From `external_canon_math.md`:
- Natural gradient: $\tilde\nabla = F^{-1}\nabla$. For Gaussian with parametrization $(\mu, \Sigma)$, $F = \mathrm{diag}(\Sigma^{-1}, \tfrac12 \Sigma^{-1}\otimes\Sigma^{-1})$, $\tilde\nabla_\mu = \Sigma\nabla_\mu$. Steepest descent in Fisher-Riemannian sense [Amari1998 §4].
- Cencov uniqueness: Fisher metric is the unique (up to scalar) Riemannian metric on a statistical manifold invariant under sufficient statistics [Cencov 1972].

External canonical references not in repo:
- Standard QM measurement back-action: the observer-induced perturbation on a quantum state is a statement about the Born-rule projector / continuous monitoring [Wiseman & Milburn 2009]. It is not a Fisher-information statement in standard QM, and the framework's claim that "the classical-quantum distinction emerges from the magnitude of Fisher information" is not a statement of standard QM but a within-framework reading.
- Crystalline lattice / phonon mean-field: in solid-state physics, atoms in a lattice are described by harmonic oscillation around equilibrium positions [Ashcroft & Mermin 1976 §22]. There is no standard physics claim that atoms "have beliefs" about their neighbors' positions; the agent interpretation is a within-framework reading.
- Photon as agent: standard QED treats photons as quantum field excitations with definite polarization and momentum but no internal belief / prior structure. The framework's "photon agent" is a within-framework extension of the pan-agentic commitment.

## What this evidence does NOT settle

1. Whether the blue-team derivations B1–B3 above are actually carried out in the section's text (they are NOT — the section only states the intuitions in prose).
2. Whether "phenomenal geometry — what it would 'experience'" at line 3843 is consistent with §`sec:cognitive_first`'s explicit non-ascription of phenomenal properties at scale 0 (or scale 1).
3. Whether "the classical-quantum distinction emerges from the magnitude of Fisher information" at line 3849 contradicts §`sec:scope_limitations` line 146 ("No quantum extension. A rigorous quantum version of the framework does not currently exist") and line 2023's "We do not extend this analogy to quantum-mechanical scenarios."
4. Whether photons-as-agents is consistent with §`sec:agent_definition` and §`sec:cognitive_first` — what scale are photons at, what are their $q, p, s, r, \phi$, do they have meta-agent indices?
5. Whether "proper time $\tau_{\text{rock}}$ advances slowly because information updates occur infrequently" conflates the framework's information-update rate with relativistic proper time (the framework's "time" needs the pullback story, which is speculative and conditional on a regulator per §`sec:scope_limitations` line 142).
6. Whether the section's rock-as-meta-agent claim is licensed by the variational criterion at Eq. `eq:meta_agent_FE_criterion`, the threshold detector at §`sec:meta_agent_threshold`, or the renormalization-group reading at §`sec:meta_agent_rg`. The section asserts "achieve sufficient coherence" without naming which criterion.
