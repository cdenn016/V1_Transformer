# Evidence Pack — pifb-electron-scale-0-loadbearing

## Manuscript references (post-2026-05-21 edits)

- `Attention/Participatory_it_from_bit.tex:119` (edited §1.5 paragraph):
  > "Every system at every scale is treated as an agent with its own gauge frame: an electron is a scale-0 agent, a molecule a higher-scale meta-agent assembled from atomic constituents, a brain a yet-higher-scale meta-agent assembled from cellular constituents, and so on through ecosystems and societies. ... The statistical manifold associated with an agent is whichever is appropriate to its scale and nature -- multivariate Gaussian on $\mathbb{R}^K$ in the working implementation of Sections~\ref{sec:framework}--\ref{sec:transformers}, but more generally any exponential family, mixture family, or in the quantum case the manifold of density operators with quantum relative entropy as the divergence (the manuscript does not develop the quantum case; see Section~\ref{sec:open_problems}). The Gaussian belief tuple is a simplification of the working architecture, not a commitment of the ontology."

- `Attention/Participatory_it_from_bit.tex:119` (panprotopsychist positioning paragraph): "the framework is a *panprotopsychist* position in the sense of Chalmers~\cite{Chalmers2013consciousness, Chalmers2016}: scale-0 agents are ascribed gauge-frame structure plus participation in variational dynamics, not phenomenal properties".

- `Attention/Participatory_it_from_bit.tex:146` (§1.7 "No quantum extension"): "A rigorous quantum version of the framework does not currently exist; the structural parallels with QBism and other relational interpretations are suggestive rather than derived."

- `Attention/Participatory_it_from_bit.tex:489-499` (edited §sec:base_manifold): base manifold $\mathcal{C}$ reclassified as "parameter / index space of contexts" rather than "noumenal substrate" — consistent with the §1.5 pan-agentic claim that there are no non-agent physical things.

- `Attention/Participatory_it_from_bit.tex:617-625` (§sec:agents_as_sections): formal definition of agent as smooth section of principal G-bundle carrying $(q, p, \phi, s, r)$ fields with Gaussian beliefs on $\mathbb{R}^K$ in the working implementation.

## Canonical literature

### Classical statistical-manifold / information geometry
- Amari & Nagaoka 2000, *Methods of Information Geometry* (AMS) — exponential and mixture families, dual flat structure, Fisher–Rao metric, e-/m-connections.
- Amari 2016, *Information Geometry and Its Applications* — modern textbook treatment.

### Quantum information geometry
- Petz 1996, "Monotone metrics on matrix spaces" (Linear Algebra Appl.) — characterizes the Riemannian metrics on the density-operator manifold $\mathcal{D}(\mathcal{H})$ invariant under quantum coarse-graining. The Bures metric and the Kubo–Mori metric are canonical choices.
- Petz 2008, *Quantum Information Theory and Quantum Statistics* (Springer) — textbook on density-operator-manifold geometry.
- Bengtsson & Życzkowski 2017, *Geometry of Quantum States* (2nd ed., Cambridge) — full treatment of quantum statistical manifolds.
- Amari & Nagaoka 2000 ch. 7 — quantum extension of information geometry.

### Hudson's theorem (red-team's prior attack)
- Hudson 1974, "When is the Wigner quasi-probability density non-negative?" Rep. Math. Phys. 6:249–252 — pure quantum states with non-negative Wigner functions are exactly the Gaussian states. This is what blocks the *bare* classical-Gaussian-on-$\mathbb{R}^K$ mapping of generic quantum states; it does NOT block the density-operator-manifold mapping (which is a separate statistical manifold with quantum relative entropy as the divergence).

### Quantum relative entropy as gauge-invariant divergence candidate
- Umegaki 1962, "Conditional expectation in an operator algebra IV: Entropy and information" — defines quantum relative entropy $S(\rho \| \sigma) = \mathrm{tr}(\rho \log \rho - \rho \log \sigma)$.
- Vedral 2002, "The role of relative entropy in quantum information theory" — properties and applications.
- Quantum relative entropy is monotone under CPTP maps (data-processing inequality, Lindblad–Uhlmann), which is the analog of the classical KL monotonicity under coarse-graining.

### Friston-side multi-scale extension
- Friston, Wiese & Hobson 2020, "Sentience and the Origins of Consciousness: From Cartesian Duality to Markovian Monism" (Entropy 22:516) — Markovian monism ascribes gradient-flow dynamics on a variational free-energy to every Markov-blanket-bearing system, ranging from particles up. The pan-agentic / FEP extension of this is straightforward at the structural level; the gauge-frame structure on top is the manuscript's specific addition.
- Sengupta, Tozzi, Cooray, Douglas & Friston 2016, "Towards a Neuronal Gauge Theory" (PLoS Biology) — earliest FEP+gauge formulation, now cited in §1.8.

### Panprotopsychism / Russellian monism literature
- Chalmers 2013/2015, "Panpsychism and Panprotopsychism" — defines panprotopsychism as the position that fundamental physical entities have proto-phenomenal properties (or that consciousness arises from the structural primitives without those primitives themselves being phenomenal). The §1.5 edited paragraph now positions the framework explicitly here.
- Chalmers 2016, "The Combination Problem for Panpsychism" — discusses how micro-properties combine into macro-experience. The framework's meta-agent variational principle is the candidate combination rule.
- Stanford Encyclopedia of Philosophy "Panpsychism" §2.3 — "Suspending judgment about consciousness at fundamental levels is the textbook panprotopsychist position."

## What this evidence does NOT settle

1. **Whether a constructive density-operator-manifold version of the framework exists in any depth**: the §1.5 edit *names* the density-operator manifold as the appropriate quantum statistical manifold but the manuscript does not develop the construction. Red can argue that an aspirational mention is not the same as in-principle coherence.

2. **Whether the gauge frame $\phi_i \in \mathfrak{g}$ has a well-defined analog on the density-operator manifold**: classical $\mathrm{GL}(K, \mathbb{R})$ acts on real Gaussians via $\rho(\Omega) \cdot (\mu, \Sigma) = (\Omega \mu, \Omega \Sigma \Omega^\top)$. On the density-operator manifold, the natural symmetry group is the unitary group $U(\mathcal{H})$ acting via $\rho \mapsto U \rho U^\dagger$; whether $\mathrm{GL}(K, \mathbb{C})$ or a broader group gives a coherent gauge-bundle structure with inter-agent transport is open.

3. **Whether "electron is a scale-0 agent" is illustrative or load-bearing**: if illustrative, the claim survives the no-quantum-extension disclaimer trivially. If load-bearing (e.g., the framework's predictions depend on electrons being agents in the same gauge-theoretic sense as molecules and brains), then the absence of a constructive scale-0 instantiation is a more serious gap. Red and blue need to agree on which reading is in play.

4. **Whether structural realism / Russellian-monism literature accepts "any-statistical-manifold" as an admissible primitive set**: the panprotopsychist position requires *some* fundamental structural property at the micro-level. The §1.5 commitment is "gauge-frame structure + variational dynamics + statistical manifold (family unspecified)". Whether this counts as a *positive* commitment in the SEP §2.3 sense, or whether leaving the family unspecified makes the position vacuous, is the deeper philosophical question.
