# Claim — eliminating-external-observations

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (anchored at `Attention/Participatory_it_from_bit.tex` lines 1394–1467 plus the cross-references to §`sec:symmetry_breaking` at 1469 and Eq. `eq:dirac_kl` at 1612)
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge\

## Claim

The §`subsection{Eliminating External Observations: A Self-Contained Framework}` (lines 1394–1467 of `Attention/Participatory_it_from_bit.tex`) is both **mathematically correct** in the explicit gradient computations (mean-gradient agreement, covariance-gradient discrepancy by $-\tfrac12\Sigma_i^{-1}$, cross-entropy resolution) **and appropriately motivates** the eliminative move of representing external observations as environmental agents within the framework's all-agent ontology, supporting the user's intended reading that "in the theory the only things in the theory are agents; observations are then communication between agents potentially across various scales."

## User context

User: "perform a /red-blue-debate on \subsection{Eliminating External Observations: A Self-Contained Framework} ... teams might consider /sympy derivations using the $-\mathbb{E}_q \log p(o|c)$ term ... recall in the theory the only things in the theory are agents ... observations are then communication between agents potentially across various scales (hopefully?)".

The user notes they previously had a "'derivation'/construction of observations as environmental agents" and want the debate to evaluate whether the current text successfully realizes that derivation.

## Sub-propositions (for reference, not separate debates)

The compound claim has two load-bearing parts:

- **C1 (Correctness).** The explicit derivation in lines 1415–1445 is mathematically correct:
  - The mean-gradient identity $\partial \mathrm{KL}(q_i\|q_{e_k})/\partial\mu_i = \Lambda_o(\mu_i - c_k) = \partial[-\mathbb{E}_{q_i}\log p(o_k|c)]/\partial\mu_i$ for any $\Sigma_o > 0$.
  - The covariance-gradient discrepancy: $\partial_{\Sigma_i}\mathrm{KL}(q_i\|q_{e_k}) = \tfrac12(\Lambda_o - \Sigma_i^{-1})$ vs. $\partial_{\Sigma_i}[-\mathbb{E}_{q_i}\log p(o_k|c)] = \tfrac12\Lambda_o$.
  - The cross-entropy identity $-\mathbb{E}_{q_i}[\log q_{e_k}] = \mathrm{KL}(q_i\|q_{e_k}) + H(q_i)$, with $H(q_i) = \tfrac12\log|2\pi e\Sigma_i|$.
  - The Dirac-limit caveat: $\mathrm{KL}(q_i\|\delta(c - c_k)) = +\infty$ for non-degenerate $q_i$, requiring finite sensory precision $\Sigma_o > 0$.
  - The construction's gauge-fixing $\Omega_{i,e_k} = I$ (line 1431) is consistent with the §`sec:symmetry_breaking` reading of environmental agents as entering the free energy with fixed frames.

- **C2 (Motivation).** The construction motivates the "self-contained framework" claim:
  - Environmental agents are agents in the framework's sense (have $q$, $p$, $\phi$; couple via KL).
  - The substitution eliminates the asymmetry between "agents internal" and "observations external" (line 1396).
  - The cross-entropy resolution preserves the standard active-inference observation dynamics in the covariance sector under full variational equivalence.
  - The fixed-covariance alternative is a coherent regime where the discrepancy vanishes.
  - The cross-scale reading — that environmental agents may be agents at a different scale (cells composed of receptors composed of proteins composed of molecules composed of bits, as stated at line 1447) — is well-supported by the framework's hierarchical-agent structure.

A win for BLUE requires both C1 and C2 to hold. A win for RED requires demonstration of a derivation error in C1, OR a motivational defect: the substitution is partial/conditional in a way the section's framing does not adequately disclose, the env-agent construction violates the framework's agent-definition (e.g., env agents have $q=p$ and frozen dynamics, breaking the standard agent role), or the cross-scale reading is asserted but not supported by an explicit construction.

## Specific questions the teams should address

1. Is the gauge-fixing $\Omega_{i,e_k} = I$ (line 1431) compatible with the framework's gauge-covariance hard constraint, or is it a privileged choice that re-introduces the asymmetry the section claims to eliminate?
2. Env agents are set with $q_{e_k} = p_{e_k}$ (Eq. line 1428) and $\beta_{i,e_k} = 1$ (line 1429). Are these agents in the same dynamical sense as the framework's other agents — do they minimize their own free energy contribution, or are they static carriers?
3. The cross-entropy substitution loses the $-\tfrac12 \Sigma_i^{-1}$ "pull toward larger covariance." Under standard active-inference / Friston canon, is the cross-entropy form the canonical observation likelihood, and does the KL form (with the additional pull) constitute a different theory?
4. The user's intended reading is "observations as cross-scale agent communication." Does the section actually deliver a cross-scale construction (env agents at a different scale than the receiving agent), or does it deliver a same-scale agent substitution with frozen dynamics?
5. Sympy verification: derive $\partial \mathrm{KL}(q_i \| \mathcal{N}(c_k, \Sigma_o))/\partial \mu_i$ and $\partial/\partial \Sigma_i$ explicitly, then derive $\partial[-\mathbb{E}_{q_i}\log \mathcal{N}(o_k; \mu_i, \Sigma_o)]/\partial\mu_i$ and $\partial/\partial\Sigma_i$, and verify the claimed mean-gradient agreement and covariance-gradient discrepancy.
