# Action — eliminating-external-observations

**From verdict:** RED_WINS (on the compound claim, at the current text)

## Sub-claim status

- **C1 (algebraic correctness of mean-gradient match, covariance-gradient discrepancy, cross-entropy identity).** Holds. Sympy verified by both teams under the identification $c_k = o_k$ and identity emission map.
- **C2 (motivation: self-contained framework with observations as agent communication).** Fails on the current text on three independent grounds: agent-definition deficit, cross-entropy substitution recovers the standard FEP observation operator under a renamed wrapper, and the unconditional bookends (title at 1394, parsimony at 1447) overclaim relative to the body's conditional language and the externalism named at 1484.

The compound claim therefore fails on C2. The defects are repairable by five mandatory line-level revisions.

## Recommended action

Mandatory repairs to `Attention/Participatory_it_from_bit.tex`. After items A1–A5 are applied, the verdict on the repaired text flips to BLUE_WINS.

### A1. Repair the subsection title at line 1394

Replace the unconditional "Eliminating External Observations: A Self-Contained Framework" with a title that names what the section actually delivers. Candidate:

> Recasting External Observations as Environmental-Agent Couplings: Mean-Gradient Equivalence and a Cross-Entropy Resolution

The current title advertises an elimination that the body explicitly conditions on the cross-entropy substitution or fixed-covariance regime, and the cross-referenced §`sec:symmetry_breaking` (line 1484) names env agents as external source fields.

### A2. Repair the "same dynamics" claim at line 1409

Soften "subject to the same information-geometric dynamics as all other agents" to specify the env-agent sub-class as a boundary class with fixed dynamics. Candidate:

> Environmental agents enter the free energy as a boundary class of constituent agents whose fast-channel and slow-channel dynamics are fixed by construction ($q_{e_k} = p_{e_k}$, $\beta_{i,e_k} = 1$); they participate in the functional as carriers of sensory information rather than as autonomously F-minimizing agents.

This brings the claim into agreement with the explicit construction at lines 1426–1430 where env agents are stipulated with $q = p$ and no $s$, $r$, or full $\phi$ dynamics.

### A3. Repair the construction at line 1428

State explicitly that $p_{e_k} = q_{e_k}$ is a boundary-condition stipulation and not a cross-scale shadow per Eq. `eq:cross_scale_shadow`. Add a sentence such as:

> The equality $p_{e_k} = q_{e_k}$ is stipulated as a boundary condition: env agents do not inherit a prior through the cross-scale shadow relation of Eq.~\eqref{eq:cross_scale_shadow}, and we do not specify primitive generative-model or hyper-prior sections ($s_{e_k}$, $r_{e_k}$) or a meta-agent index for them. The env-agent sub-class is therefore a degenerate special case of the agent definition at §\ref{sec:agent_definition}, retained for the role of carrying sensory information into the functional rather than for full participation in the variational hierarchy.

### A4. Repair the cross-entropy framing at line 1442

State that the cross-entropy substitution recovers term 5 of the canonical functional (line 1260), which is the standard FEP observation operator. Candidate addition after the existing cross-entropy identity:

> The cross-entropy form $-\mathbb{E}_{q_i}[\log q_{e_k}]$ is exactly term 5 of the canonical free energy at Eq.~\eqref{eq:free_energy_functional_final}: the standard FEP observation operator~\cite{Friston2010, parr2022active}. The substitution is therefore identification of the env-agent coupling with the existing observation slot, and full variational equivalence is achieved by routing the env-agent density into the standard observation operator. This recovers — rather than eliminates — the standard observation term, now sourced from an env-agent density $q_{e_k}$ rather than directly from data $o_k$.

### A5. Repair the parsimony sentence at line 1447

Qualify the unconditional "no external reality providing special inputs" claim. Candidate replacement:

> Under the cross-entropy substitution above, the formulation is more parsimonious notationally: every term in the free energy is written as an agent-to-agent or agent-to-environmental-agent coupling, with no explicit observation symbol $o$ appearing in the functional. The dependence on external information is not eliminated; the env-agent's belief $q_{e_k}$ carries the noumenal location $c_k$ in exactly the role the observation $o_k$ played, and §\ref{sec:symmetry_breaking} (line 1484) identifies env agents as external source fields that explicitly break gauge symmetry. The Markov-blanket-of-sensors picture (cells, organs, receptors, proteins, molecules, bits) is a heuristic composition with the cross-scale machinery at Eq.~\eqref{eq:cross_scale_shadow}; an equation-level cross-scale construction is not given here.

## Optional addition

### A6 (optional). Equation-level cross-scale construction

To deliver the user's intended reading — "observations are then communication between agents potentially across various scales" — extend the env-agent construction to carry a scale label $s' < s$ and route the $i$-to-$e_k$ coupling through Eq. `eq:cross_scale_shadow`:

> Let $e_k$ be a constituent of $i$ at scale $s' = s - 1$, with belief $q_{e_k}^{(s-1)}$ on the state fiber at scale $s'$. The cross-scale shadow $p_{e_k}^{(s-1)} = \Omega_{e_k, I'}[q_{I'}^{(s)}]$ with $I' = i$ assigns the env-agent's prior as the gauge-transported posterior of its containing agent at the next scale up. The fast-channel coupling
> \begin{equation}
> \beta_{i,e_k}\,\mathrm{KL}\big(q_i \|\Omega_{i, e_k}[q_{e_k}]\big)
> \end{equation}
> then realizes "observations from level $s-1$" as the gauge-transported belief of a lower-scale constituent acting on the receiver at level $s$. Composition over scales $s' \in \{s-1, s-2, \ldots, 0\}$ recovers the Markov-blanket-of-sensors picture of cells composed of receptors composed of proteins composed of molecules composed of bits, as the iterated cross-scale shadow of a chain of constituent env-agents.

Items A1–A5 are mandatory for the verdict to flip to BLUE_WINS on the repaired text. Item A6 is optional but is the only one that delivers the user's stated intended reading.

## What survives without modification

After A1–A5, the following content of the section stands unmodified:

- The explicit algebra at lines 1431–1445 (mean-gradient identity, covariance-gradient discrepancy of $-\tfrac12 \Sigma_i^{-1}$, cross-entropy identity).
- The Dirac caveat at line 1425.
- The gauge-fixing $\Omega_{i,e_k} = I$ at line 1431, now correctly named as an explicit-symmetry-breaking move per §`sec:symmetry_breaking`.
- The structural-analogy table at lines 1451–1463 (which is appropriately scoped already).

## Follow-up debates (if any)

One natural follow-up — separate debate, separate slug:

- **Cross-scale agent communication.** Does the framework's hierarchical agent structure at §`sec:agent_definition` and the cross-scale shadow at Eq. `eq:cross_scale_shadow` actually realize an "observations as cross-scale agent communication" picture for the bottom of the hierarchy, or does the bottom-scale boundary (line 548) require external sensory inputs that fall outside the framework's all-agent ontology? This is the user's intended reading and would be settled by either the A6 extension above or a separate construction at the bottom of the hierarchy.

No other sub-claims of the present debate need separate adjudication.
