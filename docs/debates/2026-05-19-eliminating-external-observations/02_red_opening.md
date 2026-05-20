# Red Opening — eliminating-external-observations

## Steelman (opposing position)

The section delivers a mathematically correct identification (verified at the level of mean gradients, with a transparent and isolated covariance discrepancy resolved by a single textbook identity) of observation likelihoods with a coupling to environmental belief densities, and the resulting reformulation eliminates the only ontological asymmetry in the framework by treating every term in the free energy as an agent-to-agent coupling.

## Position

C1 holds; C2 fails. The math at lines 1415--1445 is correct under the explicit identification of the noumenal location $c_k$ with the observation $o_k$, but the construction at lines 1426--1431 does not satisfy the manuscript's own definition of an agent, the cross-entropy resolution at line 1442 introduces a structurally different coupling operator that breaks the homogeneous all-agent KL-coupling ontology, and the cross-scale reading the user wants is asserted in prose at line 1447 but is not constructed anywhere in the section.

## Evidence

### C1 is conceded (with one noted assumption)

Sympy verification of the three load-bearing identities in 1D (the multivariate case reduces by the same algebra):
- $\partial_{\mu_i}\mathrm{KL}(q_i \| q_{e_k}) = \Lambda_o(\mu_i - c_k)$ matches $\partial_{\mu_i}[-\mathbb{E}_{q_i}\log\mathcal{N}(o_k; \mu_i, \Sigma_o)] = \Lambda_o(\mu_i - o_k)$ under the identification $c_k = o_k$. The identification is intentional (line 1425: "noumenal location $c_k$ corresponding to observation $o_k$") but presupposes an identity emission map $g(c) = c$; standard active inference allows arbitrary $g$ [ParrPezzuloFriston2022 §4.3], so the equivalence is to a restricted sub-class of observation models, not to the general likelihood term written at line 1396. Worth one sentence in revision; not a falsification.
- $\partial_{\Sigma_i}\mathrm{KL}(q_i \| q_{e_k}) - \partial_{\Sigma_i}[-\mathbb{E}_{q_i}\log p(o_k\mid c)] = -\tfrac12\Sigma_i^{-1}$. Sign and magnitude match line 1439.
- $-\mathbb{E}_{q_i}[\log q_{e_k}] = \mathrm{KL}(q_i \| q_{e_k}) + H(q_i)$ with $H(q_i) = \tfrac12\log|2\pi e\Sigma_i|$. Standard cross-entropy decomposition [CoverThomas2006 §2.5]; verified.

The mathematical content of C1 stands.

### C2 fails on three independent grounds

**Strike 1: Internal-definition violation.** The manuscript's own agent definition at lines 617--631 (§`sec:agent_definition`) requires every agent to consist of two primitive sections $(q_i, s_i)$, two derived sections $(p_i, r_i)$ realized as cross-scale shadows $p_i = \Omega_{i,I}[q_I^{(s+1)}]$, $r_i = \tilde\Omega_{i,I}[s_I^{(s+1)}]$ via Eq. eq:cross_scale_shadow (line 540), a gauge-frame field $\phi_i: \mathcal{U}_i \to \mathfrak{g}$, and an embedding in the variational hierarchy $r \to s \to p \to q \to$ observations across scales. The environmental agents at lines 1426--1431 are specified as: $q_{e_k} = \mathcal{N}(c_k, \Sigma_o)$, $p_{e_k} = q_{e_k}$, $\beta_{i,e_k} = 1$, $\Omega_{i,e_k} = I$. There is no model section $s_{e_k}$, no hyper-prior $r_{e_k}$, no scale label, no meta-agent index $I$, no gauge-frame field $\phi_{e_k}$ (only the implicit $\Omega = I$ relation to every receiving agent simultaneously), and $p_{e_k}$ is not derived as a cross-scale shadow but set equal to $q_{e_k}$ by hand. By the manuscript's own definition, environmental "agents" are not agents.

The consequence is sharper than missing-by-omission. With $p_{e_k} = q_{e_k}$, the self-coupling term $\alpha\,\mathrm{KL}(q_{e_k} \| p_{e_k}) = 0$ identically and the env agent has no internal dynamics in the fast channel; with no $s_{e_k}, r_{e_k}$, no slow-channel dynamics at all. This contradicts the section's own claim at line 1409 that env agents are "subject to the same information-geometric dynamics as all other agents." They are not subject to the same dynamics; they are subject to no dynamics, and they participate in the F functional only as fixed posterior carriers on the receiving agent's side of the KL.

This matches the standard FEP partition [Friston2013, Pearl1988 §3.2; cf. `external_canon_inference.md` §7] in which sensory states are passive carriers of external information, not autonomous agents. The construction relabels FEP sensory states as agents without giving them agent structure.

**Strike 2: The cross-entropy resolution breaks the coupling ontology.** The canonical free energy at Eq. eq:free_energy_functional_final (lines 1252--1263) couples every agent pair via the same operator $\mathrm{KL}(q_i \| \Omega_{ij} q_j)$ for beliefs and $\mathrm{KL}(s_i \| \tilde\Omega_{ij} s_j)$ for models. The cross-entropy resolution at line 1442 replaces the env-agent KL by $-\mathbb{E}_{q_i}[\log q_{e_k}] = \mathrm{KL}(q_i \| q_{e_k}) + H(q_i)$. This is a structurally different operator from the KL used everywhere else in the functional: it differs from KL by the receiving agent's own entropy.

The section therefore presents a trilemma it does not acknowledge. (a) Keep the KL form and accept the $-\tfrac12\Sigma_i^{-1}$ covariance discrepancy: then env-agent dynamics differ from observation-likelihood dynamics in the covariance sector, contradicting the "self-contained framework that reproduces observation dynamics" framing. (b) Use the cross-entropy form: then env agents are coupled through a different operator from the one used between agents elsewhere in the functional, breaking the homogeneous all-agents-couple-via-KL ontology that motivates the elimination of observation terms in the first place. (c) Restrict to fixed-covariance dynamics: then the framework's $\Sigma_i$ E-step (documented throughout the codebase via the canonical retraction $\sigma_{new} = \sigma \cdot \exp(e\_sigma\_lr \cdot \delta\sigma/\sigma)$, CLAUDE.md `E-step LRs are decoupled` section) is turned off for env-coupled agents only, again breaking homogeneity. The motivation in C2 ("environmental stimuli treated as additional agents coupled through information exchange rather than as privileged external inputs," line 1398) requires homogeneity of the coupling operator; the resolutions on offer destroy it.

A clean resolution would derive the cross-entropy term as the genuine inter-agent coupling, and demote the closed-form Gaussian KL used elsewhere to a special case. The section does not do this; it offers cross-entropy as a local patch on observation terms only.

**Strike 3: The cross-scale reading is evoked, not constructed.** The user's intended reading -- "observations are then communication between agents potentially across various scales" -- requires a construction in which environmental agents sit at a scale $s' \neq s$ (the receiver's scale) and the coupling $i \leftarrow e_k$ is mediated by the cross-scale shadow relation Eq. eq:cross_scale_shadow (line 540). The manuscript evokes this at line 1447 ("Markov blankets composed of sensory agents (cells, organs, etc) are themselves composed of sensory Markov blankets (receptors, proteins, molecules, etc), and onward down to single bits"). The formal construction at lines 1426--1431 contains no scale label for $e_k$, no meta-agent index relating $e_k$ to a constituent at a different scale, and no transport across scales. The receiving agent $i$ and the environmental agent $e_k$ are treated as occupying the same scale and being coupled by the same in-scale belief KL that couples in-scale agents elsewhere in the functional.

The user's reading is therefore not delivered by the construction; it is named in the closing paragraph but no equation realizes it. A genuine cross-scale construction would assign $e_k$ a scale $s'$, identify a meta-agent that contains both $i$ and $e_k$ as constituents (or place $e_k$ at scale $s+1$ with $i$ as a constituent), and couple $i$ to $e_k$ via $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ with $I = e_k$ -- which would make observations literally the cross-scale shadows of meta-agent beliefs. None of this appears in the section.

### Summary of citations

- Manuscript: `Attention/Participatory_it_from_bit.tex:617-631` (agent definition); `:540` (cross-scale shadow eq); `:1252-1263` (canonical F functional, homogeneous KL coupling); `:1409` (env agents claimed subject to same dynamics); `:1426-1431` (env-agent construction); `:1442` (cross-entropy substitution); `:1447` (cross-scale prose).
- External canon: [Friston2013 §3] / `external_canon_inference.md` §7 (Markov blanket: sensory states are passive carriers); [ParrPezzuloFriston2022 §4.3] (observation likelihood with arbitrary emission $g$); [CoverThomas2006 §2.5] (cross-entropy / KL / entropy identity, verified by sympy).
- Sympy verification of K1, K2, K3 in 1D with explicit derivatives (mean, covariance, cross-entropy identity all match the manuscript's claims under the $c_k = o_k$ identification).

## Falsification conditions

This position is wrong if blue produces any of the following from the manuscript:

1. A construction (in §`Eliminating External Observations` or elsewhere) that endows environmental agents with $s_{e_k}, r_{e_k}, \phi_{e_k}$ and a scale label, and realizes $p_{e_k}$ as a cross-scale shadow via Eq. eq:cross_scale_shadow rather than by hand-setting $p_{e_k} = q_{e_k}$.
2. A line where the cross-entropy substitution is presented either as the canonical inter-agent coupling operator throughout the framework (with the Gaussian-KL form demoted to a special case), or where the non-homogeneity it introduces is acknowledged and accepted as a feature.
3. A line where a non-prose, equation-level cross-scale construction places $e_k$ at a scale $s' \neq s$ and derives the $i$-to-$e_k$ coupling from the cross-scale shadow relation rather than from same-scale KL coupling.

If any of (1)--(3) is produced, C2 holds and the compound claim survives.
