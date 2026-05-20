# Evidence Pack — pifb-theory-section-purity

**Manuscript:** `Attention/Participatory_it_from_bit.tex`
**Section scope:** \section{Theory} (lines 180–2040)
**Canon location:** `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge\`
**Authoritative directive (user):** Teams must consult the external literature as source of truth. The project's own canon files may be consulted *only* for citation forms and for redirection to primary external references; theoretical purity and correctness are to be adjudicated against Kobayashi–Nomizu, Nakahara, Amari, Chentsov, Friston, Wilson, Kogut–Susskind, Creutz, Lahav–Neemeh, etc.

## Manuscript references — subsection-level map

| Subsection | Line | Topic |
|---|---|---|
| Notation conventions | 183 | Symbol overloading map |
| **The Base Manifold** | 483 | Smooth $n$-manifold $\mathcal{C}$; "noumenal" reading |
| **Statistical Manifolds** | 495 | $\mathcal{B}_{\mathrm{state}}, \mathcal{B}_{\mathrm{model}}$ with Fisher–Rao metric; Chentsov |
| Cross-Scale Shadows | 536 | Priors / hyper-priors as transports of meta-agent posteriors |
| **Principal Bundles & Gauge Freedom** | 550 | $\pi: \mathcal{N} \to \mathcal{C}$, $G$-bundle; Role A vs Role B; local trivializations |
| **Associated Bundles** | 589 | $\mathcal{E}_{\mathrm{state}} = \mathcal{N} \times_{\rho} \mathcal{B}$ |
| **Agents as Smooth Sections** | 612 | Definition: $q_i, s_i$ primitive sections; $p_i, r_i$ derived; $\phi_i \in \mathfrak{g}$ |
| **Multi-Agent Systems** | 650 | Overlap $\mathcal{U}_i \cap \mathcal{U}_j$, meta-agent formation |
| Consensus & Meta-Agent | 668 | Pointwise perfect-consensus definition |
| **Culture / RG closure** | 698 | Within-cluster vs cross-cluster KL inequality |
| **Epistemic Collapse / Death** | 720 | $\mathrm{KL}(q_i \| \Omega_{ij}q_j) = 0$ everywhere |
| Hierarchical Cross-Scale | 752 | Bottom-up and top-down flows |
| **Cognitive Reference Frames** | 765 | Lahav–Neemeh CFR identification with $\phi_i$ |
| Transport Operators | 778 | $\Omega_{ij} = \exp(\phi_i)\exp(-\phi_j)$ |
| Gauge Covariance | 791 | Global diagonal vs full local invariance |
| **Connection Forms** | 797 | Regime I ($\theta = U^{-1}dU$) vs Regime II ($A_\mu$ independent) |
| Field Strength | 816 | $F_{\mu\nu}$ defined; Regime II only |
| **Discrete Regime II** | 828 | Edge-relaxed cocycle, Wilson lattice link variable |
| **Hierarchy of Operators** | 882 | $\Omega_{ij}, \tilde\Omega_{ij}, \Omega_{i,I}, \Lambda^{s\to s+1}$ |
| Curvature: four geometries | 897 | Fiber + group + connection + base |
| Working Framework | 911 | Simplifications |
| Matched Bundles | 922 | $\mathcal{E}_{\mathrm{state}} = \mathcal{E}_{\mathrm{model}}$ by Eq. cross-scale-shadow |
| Gaussian Fiber | 928 | $K + K(K+1)/2$ params |
| Gauge Group $\mathrm{GL}(K)$ / $\mathrm{SO}(3)$ | 934 | Thm. 1 ($\mathrm{GL}(K)$ KL invariance proved) |
| Flat Base | 983 | $\mathcal{C} = \mathbb{R}^2$ |
| Zero-Dim Transformer Limit | 993 | $\dim \mathcal{C} = 0$ reduction |
| VFE Functional intro | 1005 | Action principle motivation |
| Multi-Agent Extension | 1021 | Acknowledged as engineered consensus energy, not FEP-derived |
| **Mixture-of-Sources** | 1033 | Source-selection construction; softmax derivation |

## Specific equation locations to weigh

- **Eq. transport_def** (line 782): $\Omega_{ij}(c) = \exp[\phi_i(c)] \exp[-\phi_j(c)]$
- **Eq. cross_scale_shadow** (line 540): $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$
- **Definition Principal Bundle** (line 567): right action, $\pi(n\cdot g) = \pi(n)$
- **Definition Associated Bundle** (line 596): $\mathcal{E} = \mathcal{N} \times_\rho \mathcal{B}$, equivalence $(n\cdot g, b) \sim (n, \rho(g)b)$
- **Definition Agent** (line 617): five fields per agent (two primitive, two derived, one $\phi$)
- **Eq. culture_internal_coherence** (line 703): $\sum_{i,j\in A}\gamma_{ij}\mathrm{KL}(s_i \| \tilde\Omega_{ij}s_j)/\sum\gamma_{ij} \le \varepsilon_A$
- **Eq. culture_closure** (line 711): internal $\ll$ external slow-channel coupling
- **Definition Epistemic Death** (line 725): $\mathrm{KL}(q_i \| \Omega_{ij}q_j) = 0$ AND $\mathrm{KL}(s_i \| \tilde\Omega_{ij}s_j) = 0$
- **Eq. edge_relaxed_omega** (line 832): $\Omega_{ij} = U_i \exp(\delta_{ij}\cdot G) U_j^{-1}$ — Regime II Wilson link variable
- **Eq. omega_gauge_law** (line 842): $\Omega_{ij} \to g_i \Omega_{ij} g_j^{-1}$ — lattice gauge transformation
- **Eq. wilson_observable** (line 856): $W_{ijk} = \mathrm{Re}\,\mathrm{Tr}[\exp(\delta_{ij}G)\exp(\delta_{jk}G)\exp(\delta_{ki}G)]$
- **Eq. wilson_action** (line 862): $S_{\mathrm{Wilson}} = \beta \sum (1 - W_{ijk}/K)$
- **Theorem $\mathrm{GL}(K)$ invariance** (line 939): KL invariant under simultaneous pushforward (full proof at 949)

## External-literature waypoints (primary sources to consult)

### Bundle theory (C1, C2, C7)
- **Kobayashi & Nomizu**, *Foundations of Differential Geometry*, Vol. I, Ch. II — definition of principal bundles, right action, transition functions.
- **Nakahara**, *Geometry, Topology and Physics*, 2003, Ch. 10 — principal/associated bundles, sections, connections, Maurer–Cartan, Yang–Mills curvature.
- **Steenrod**, *The Topology of Fibre Bundles*, 1951 — classical reference for cocycle / associated bundle construction.

### Information geometry (C1 statistical manifold, C7 Fisher metric, GL(K) theorem)
- **Amari**, *Information Geometry and Its Applications*, 2016 — Fisher–Rao metric, KL closed form, exponential families.
- **Chentsov (Čencov)**, *Statistical Decision Rules and Optimal Inference*, 1982 — uniqueness of Fisher metric under sufficient statistics (cited in manuscript at line 510).

### Lattice gauge theory (C6, Regime II Wilson observable)
- **Wilson**, "Confinement of quarks", *Phys. Rev. D* 10:2445 (1974) — original Wilson plaquette action.
- **Kogut & Susskind**, "Hamiltonian formulation of Wilson's lattice gauge theories", *Phys. Rev. D* 11:395 (1975).
- **Creutz**, *Quarks, Gluons and Lattices*, Cambridge 1983 — textbook treatment of link variables, plaquette action, gauge transformations $U_l \to g_x U_l g_{x+\hat\mu}^{-1}$.

### Variational Inference / FEP (C2 transport actions, multi-agent VFE)
- **Friston**, "The free-energy principle: a unified brain theory?", *Nat. Rev. Neurosci.* 11:127 (2010).
- **Parr, Pezzulo, Friston**, *Active Inference*, MIT Press 2022 — hierarchical generative models, conditional priors.
- **Wainwright & Jordan**, *Graphical Models, Exponential Families and Variational Inference*, FnT ML 2008 — mean-field factorization conditions.

### Quantum reference frames / edge modes (Role A vs Role B; manuscript line 561 cites these)
- **Donnelly & Freidel**, "Local subsystems in gauge theory and gravity", JHEP 2016:102.
- **Bartlett, Rudolph, Spekkens**, "Reference frames, superselection rules, and quantum information", *Rev. Mod. Phys.* 79:555 (2007).
- **Rovelli**, "Relational quantum mechanics", *Int. J. Theor. Phys.* 35:1637 (1996).

### Cognitive reference frames (C4)
- **Lahav & Neemeh**, "A relativistic theory of consciousness", *Front. Psychol.* 12:704270 (2022); 2025 follow-up cited at manuscript line 770.

### Renormalization group (C5 culture closure)
- **Wilson**, "Renormalization group and critical phenomena", *Phys. Rev. B* 4:3174 (1971).
- **Cardy**, *Scaling and Renormalization in Statistical Physics*, 1996 — block-spin / coarse-graining methodology.

## Sub-claims to weigh (lifted from 00_claim.md)

C1 — Bundle constructions match Kobayashi–Nomizu / Nakahara.
C2 — Group actions on state/model fibers are well-defined left actions paired with right principal action; convention (line 604) is internally consistent.
C3 — Multi-agent overlap + epistemic-death definition is mathematically distinct from geometric identity and uses transport correctly.
C4 — Identification of $\phi_i$ with Lahav–Neemeh CFR is supported or interpretive.
C5 — Culture closure (Eqs. culture_internal_coherence, culture_closure) constitutes a valid RG block-spin condition.
C6 — Regime II edge-relaxed cocycle is a genuine Wilson lattice link variable (Wilson 1974, KS 1975, Creutz 1983); $\delta_{ij}=0$ recovers a flat bundle.
C7 — Hierarchy of operators ($\Omega_{ij}, \tilde\Omega_{ij}, \Omega_{i,I}, \Lambda^{s\to s+1}, F_{\mu\nu}, \Phi$) is bundle-compatible without structural overloading.

## What this evidence does NOT settle

- Whether the framework's structural commitment that priors are *transports of meta-agent posteriors* (Eq. cross_scale_shadow) constitutes a legitimate refinement of standard hierarchical VI (Friston 2017) or an undeclared model substitution.
- Whether the Regime I "vanishing holonomy" cocycle theorem (Lemma referenced at line 579) holds with the strength implied or only on simply-connected patches.
- Whether the dual gauge action — "global diagonal right-translation" for the redundancy reading vs "vertex-local left-translation" for the Regime II lattice gauge reading (line 793) — is non-conjugate (as the manuscript claims at line 793) and whether observables (KL pairings, traces of closed loops) are genuinely invariant under both.
- Whether "epistemic death" being gauge-invariant under the simultaneous transformation (line 739) holds non-abelianly via the group-level form rather than the Lie-algebra additive form.
- Whether the cluster-closure inequality of Eq. culture_closure is a proven sufficient condition for adiabatic elimination in the RG construction, or whether it is being asserted as such.
- Whether the cross-scale operator $\Omega_{i,I}$ legitimately lives in the *same* group $G$ when constructed from a weighted Lie-algebra average $\phi_I = \sum_i w_i \phi_i$ (line 891), given BCH non-commutativity.

## Operational notes for the two teams

- The user has explicitly demanded literature-as-source-of-truth. The judge will count strikes only when the cited source can be checked by name+year (textbook + section, or paper + equation/page number).
- Treat each sub-claim C1–C7 as an independently arguable strike. Both opening and rebuttal should cover all seven, but in proportion to the strength of the attack/defense available.
- Direct quotations from manuscript lines must be exact and line-numbered. Direct quotations from external sources must include section / chapter / equation number.
