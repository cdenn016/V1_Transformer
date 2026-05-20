# Evidence Pack — pifb-discussion-macro-consensus

## Manuscript references

### The Macroscopic Objects subsection under debate (Discussion §3136-3153)

- `Participatory_it_from_bit.tex:3136` — subsection header "Macroscopic Objects as Consensus Enforcers"
- `Participatory_it_from_bit.tex:3138` — opening disclaimer paragraph: "The following is an interpretive extension of the formalism beyond what the simulations of Section sec:meta_agent_emergence demonstrate. Macroscopic objects are not modeled directly as agents in those simulations; their description here as 'agents with sharp prior covariance' is a stipulation extending the multi-scale meta-agent construction (Appendix app:examples) by analogy. Treat the following as a framework-internal interpretation rather than a derivation."
- `Participatory_it_from_bit.tex:3140` — rhetorical setup: "If our perceptions and models of reality are tentative and contingent why then does everyday physics appear objective? Our answer is that macroscopic objects enforce consensus through precision dominance."
- `Participatory_it_from_bit.tex:3142` — load-bearing math sentence: "A rock maintains extraordinarily precise self-localization with prior covariance Σ_rock ≈ εI where ε << 1 is a small dimensionless precision in the framework's representational space. We do not assign ε a physical-units value (we are not claiming ε ~ ℓ_P^2); the dimensional bridge between informational precision and physical length scales is among the open problems of Section sec:scope_limitations."
- `Participatory_it_from_bit.tex:3144-3146` — coupling equation: $\beta_{i,\mathrm{rock}}\, \mathrm{KL}(q_i \| \Omega_{i,\mathrm{rock}}[q_{\mathrm{rock}}])$
- `Participatory_it_from_bit.tex:3148` — math claim: "contains the rock's prior precision $\Sigma_{\mathrm{rock}}^{-1}$ as the sandwich operator inside the KL, so the alignment penalty scales with this very large precision and dominates $i$'s total free energy. Agent $i$'s beliefs are forced into alignment; there is no freedom to maintain alternative priors when coupled to such a high-precision neighbor."
- `Participatory_it_from_bit.tex:3150-3153` — conclusion: "The apparent objectivity of macroscopic physics thus arises not from pre-existing external reality but from epistemic anchoring: high-precision systems constrain all observers into consensus. Rocks appear 'certainly there' because their extreme self-precision leaves\nNo room for disagreement."
  - NOTE: Capital "N" in "No room" mid-sentence is a typographic defect.

### Cross-referenced machinery

- `Participatory_it_from_bit.tex:sec:framework` — the full multi-agent free energy where the coupling at 3144-3146 lives:
  $F = \sum_i \alpha_i \mathrm{KL}(q_i\|p_i) + \sum_{ij} \beta_{ij}\mathrm{KL}(q_i \| \Omega_{ij} q_j) + \ldots$
  Each agent has its own prior coupling AND its consensus couplings; the math claim at 3148 is about which term dominates.
- `Participatory_it_from_bit.tex:sec:meta_agent_emergence` (around 2248) — the simulations that the 3138 disclaimer says do NOT include macroscopic objects.
- `Participatory_it_from_bit.tex:app:examples` (around 3837 — the "Macroscopic Objects as Meta-Agents: The Rock Example" appendix) — extended worked example. Teams should consult this appendix because it does additional load-bearing work that the Discussion subsection summarizes.

### Gaussian KL closed form (for math correctness check)

The KL divergence between two Gaussians $q = \mathcal{N}(\mu_q, \Sigma_q)$ and $p = \mathcal{N}(\mu_p, \Sigma_p)$ is:
$$2\,\mathrm{KL}(q \| p) = \mathrm{tr}(\Sigma_p^{-1} \Sigma_q) + (\mu_p - \mu_q)^\top \Sigma_p^{-1} (\mu_p - \mu_q) - K + \ln(|\Sigma_p|/|\Sigma_q|)$$

So $\mathrm{KL}(q_i \| q_{\mathrm{rock}})$ scales as $\sim d^\top \Sigma_{\mathrm{rock}}^{-1} d$ in the Mahalanobis term, which grows like $\epsilon^{-1}$ for $\Sigma_{\mathrm{rock}} = \epsilon I$. The "dominates total free energy" claim is mathematically correct in the limit $\beta_{i,\mathrm{rock}}/\epsilon \to \infty$, conditional on $\beta_{i,\mathrm{rock}}$ remaining of order unity (or not being suppressed by softmax normalization).

## What this evidence does NOT settle

1. Whether the disclaimer at 3138 ("stipulation extending the multi-scale meta-agent construction by analogy") is a sufficient disclaimer, given that the entire substantive content of the subsection rests on the stipulation. If the stipulation is doing all the work, the disclaimer either rescues the section as honest speculation OR concedes that there is no content.
2. Whether the title "Macroscopic Objects as Consensus Enforcers" is misleading: the mechanism described is NOT multi-agent consensus formation (sec:meta_agent_emergence), it is a single-coupling KL dominance. "Consensus" in the framework's technical sense involves agents reaching agreement; "enforcement" of a perspective by a high-precision neighbor is a different mechanism.
3. Whether the "dominates total free energy" claim at 3148 needs a regime condition. The math is correct when the rock's precision dominates over agent $i$'s own prior precision $\Sigma_{p,i}^{-1}$ AND over agent $i$'s sensory precision $\Lambda_{o_i}$ AND when $\beta_{i,\mathrm{rock}}$ is not driven to zero by softmax normalization across competing neighbors. The subsection asserts dominance without these conditions.
4. Whether the rock has a generative model $s_{\mathrm{rock}}$ and a gauge frame $\phi_{\mathrm{rock}}$. The framework requires every agent to carry the full tuple $(q, s, \phi)$ (sec:scope_limitations, pan-agentic ontology). If rocks have generative models and gauge frames, what are they? The subsection does not address this even though the pan-agentic commitment requires it.
5. Whether the rock-as-agent stipulation conflicts with sec:meta_agent_emergence's explicit claim that meta-agents form via consensus among constituent agents. A rock is not a meta-agent of its constituent atoms in the simulation sense, since the constituents would have their own beliefs and the consensus dynamics would need to be modeled. The subsection elides this.
6. Whether the typographic defect at 3152-3153 ("leaves\nNo room") is a paragraph-break issue (rare in this manuscript) or a sentence-internal capitalization error.

Teams should verify points 1-5 against the actual math and the appendix app:examples. Point 6 is a presentation defect both sides can stipulate.
