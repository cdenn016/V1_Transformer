# Evidence Pack — pifb-theory-rock-solid

Mode: theory. Canon location: `.claude/agents/vfe-knowledge/`. Working directory: `docs/debates/2026-05-21-pifb-theory-rock-solid/`.

## Scope

§Theory of `Attention/Participatory_it_from_bit.tex` runs **lines 180–2070**, with 19 subsections:

| Line | Subsection |
|------|------------|
| 180 | `\section{Theory}` (notation conventions) |
| 483 | The Base Manifold: Noumenal Space |
| 495 | Statistical Manifolds: Beliefs and Models |
| 536 | Cross-Scale Shadows |
| 550 | Principal Bundles and Gauge Freedom |
| 589 | Associated Bundles |
| 612 | Agents as Smooth Sections |
| 650 | Multi-Agent Systems and Overlapping Perspectives |
| 765 | Cognitive Reference Frames as Gauge Frames |
| 882 | The Hierarchy of Transport Operators |
| 897 | Curvature Structure: Four Interacting Geometries |
| 911 | Working Framework: Simplifications and Scope |
| 1005 | The Variational Free Energy Functional: Dynamics from Information Geometry |
| 1247 | The Complete Free Energy Functional |
| 1421 | Recasting External Observations as Environmental-Agent Couplings |
| 1496 | Explicit Symmetry Breaking via Observations |
| 1529 | Dynamical Structure and Emergent Timescales |
| 1598 | Transformer Architectures as the Zero-Dimensional Limit |
| 1871 | Statistical Precision as Configuration-Space Stiffness: A Mass Analogy |

The Theory section concludes at line 2070, immediately before `\section{Implementation}` at line 2071.

## Key manuscript references

- **Notation overload acknowledged (lines 183–194).** The "Notation and symbol conventions" paragraph at the head of §Theory enumerates symbol overloads (`s`, `κ`, `τ`, `α`, `β`, `γ`, `χ`, `Ω`, `π`) and labels each.
- **Canonical free energy functional (line 1252, `\label{eq:free_energy_functional_final}`).** The boxed functional with KL self-coupling, hyper-prior coupling, belief alignment with attention entropy, model coupling with meta-attention entropy, and observation likelihood.
- **Softmax-from-Lagrangian derivation (lines 1266–1281).** Solves the row-Lagrangian for $\beta_{ij}$ subject to $\sum_j\beta_{ij}=1$ and recovers the softmax form Eq.~\eqref{eq:beta_optimal}.
- **Envelope theorem and reduced free energy (lines 1361–1398).** Substituting $\beta^*$ into $\mathcal{F}_{\text{align}}$ gives $-\tau\log Z_i$; states that autograd-on-surrogate $\nabla\langle E\rangle_\beta$ differs from $\nabla\mathcal{F}_{\text{red}}$ by $-\tau^{-1}\mathrm{Cov}_{\beta^*}(E,\nabla E)$.
- **State-dependent precision (lines 1285–1359).** Per-agent variational parameter $\alpha_i$ with closed-form $\alpha_i^* = c_0/(b_0 + D_{\mathrm{KL}}(q_i\|p_i))$; product-rule chain at autograd $(\alpha_i^*)^2 b_0/c_0 \cdot \partial D_{\mathrm{KL}}/\partial\theta$.
- **Environmental agents / mean-gradient equivalence proposition (lines 1438–1475).** Proves mean-gradient equivalence between observation-likelihood and environmental-agent formulations; covariance equivalence requires cross-entropy substitution. Cross-entropy resolution: $-\mathbb{E}_{q_i}[\log q_{e_k}] = \mathrm{KL}(q_i\|q_{e_k}) + H(q_i)$.
- **Mixture-of-sources construction (lines 1033–1113).** Variational consensus-energy derivation of $\beta_{ij}\propto\pi_j\exp(-D_{\mathrm{KL}}/\tau)$ with explicit status caveat at lines 1038, 1058 (source-independence is a structural assumption, not tractability).
- **Untied query-key carving (lines 1702–1742).** Exact decomposition $D_{\mathrm{KL}}(q_i\|\Omega_{ij\#}q_j) = \tfrac12[\dots\log\det\Sigma_j - \log\det\Sigma_i + 2\log|\det U_i|\dots]$ with $Q_i=U_i^{-1}\mu_i$, $K_j=U_j^\top\Sigma_j^{-1}\mu_j$. Stated to be verified symbolically against direct Gaussian KL "to machine precision."
- **Precision-as-mass analogy (lines 1871–2069).** Hessian $M_{\mu\mu}$ derivations with $\Omega^{-T}$ GL-correct transport; explicit acknowledgement that "mass" terminology requires a separate kinetic-metric postulate, with $\omega^2\propto m_{\text{eff}}^{-1}$ being definitional rather than empirical under that postulate.
- **TODO marker at line 1874.** Inside §Theory: `\textbf{TODO:}` for an empirical test of $\omega^2 \propto m_{\text{eff}}^{-1}$ "deferred to future work; no such study is reported in this manuscript."
- **Goldstone / SSB caveat (lines 1499–1527).** Section claims symmetry breaking is "explicit rather than spontaneous"; "A genuine spontaneous-symmetry-breaking story would require an internal mechanism… neither of which we establish here. The Goldstone parallel is suggestive but is taken here as analogy rather than derivation."
- **Thermodynamic analogy (line 1409, "Thermodynamic Analogy").** Free energy compared to Helmholtz $F = U - TS$ / grand potential $\Omega = F - \mu N$ — labeled as analogy.
- **Structural-analogy table with physical actions (lines 1479–1491).** Compares the framework to classical mechanics, GR, gauge theory actions; explicitly labeled "Structural analogy, not dimensional equivalence."
- **Dual role for gauge frame Role A / Role B (lines 555–565).** Explicit treatment of $\phi_i$ as both gauge-redundant (transport role) and physical state (state role), with cited justification from edge modes [DonnellyFreidel2016], quantum reference frames [BartlettRudolphSpekkens2007, Vanrietvelde2020], Rovelli relational interpretation.
- **Bundle-section caveat (lines 577–581).** Local trivialization caveat: global section requires trivial bundle; $\exp:\mathfrak{g}\to G$ only locally bijective for non-compact $\mathrm{GL}^+(K)$.

## Companion-paper dependence flags

Recurrent in-text references to `\cite{Dennis2025trans}` (gauge-theoretic transformer companion paper):
- Line 1036 (mixture-of-sources construction "adapted from" companion).
- Line 1288 (per-agent variational precision construction "discusses precision optimization as a learning problem").
- Line 1346 (per-coordinate precision form "adopted" by companion paper).
- Line 1359 (chain-rule slot referenced in companion).
- Line 1609 (transformer-limit reduction "follows the development in" companion).
- Line 1617 (multi-head lift "treated in" companion).
- Line 1631 (untied query-key reduction).
- Line 1670 (Rényi $\alpha$-divergence configuration in companion).
- Line 1696 (thin-SVD decomposition equations referenced in companion).
- Line 630 (multi-agent / `gamma_{ij}=0` convention follows companion).

The theory section's load-bearing reductions (transformer limit, per-coordinate $\alpha$, multi-head lift) cite the companion paper as the locus of full development.

## Canon excerpts (external, from `vfe-knowledge/external_canon_*.md`)

- **Closed-form KL between Gaussians** [BleiKuckelbirgJordan2017, KingmaWelling2014 App. B]:
  $\mathrm{KL}(q\|p) = \tfrac12\big[\mathrm{tr}(\Sigma_p^{-1}\Sigma_q) + (\mu_p-\mu_q)^\top\Sigma_p^{-1}(\mu_p-\mu_q) - K + \log(|\Sigma_p|/|\Sigma_q|)\big]$.
- **Standard variational free energy** [Friston2010, ParrPezzuloFriston2022 Ch. 2]: $F = \mathbb{E}_q[\log q - \log p(o,s)]$; three equivalent decompositions. Multi-agent coupling $\sum_{ij}\beta_{ij}\mathrm{KL}(q_i\|\Omega_{ij}q_j)$ is **not** in the standard form — it is user-introduced.
- **Attention entropy term $\tau\beta\log(\beta/\pi)$** — standard if presented as Lagrangian for soft-assignment; novel if claimed to follow from FEP alone (`external_canon_inference.md:30`).
- **Hierarchical mean-field** [Friston2017Graphical, ParrPezzuloFriston2022 Ch. 9]: $q(s_1,\dots,s_L) = \prod_\ell q(s_\ell)$. Cross-level couplings come from the generative model $p(s_\ell|s_{\ell+1})$, not from a deterministic point-passing scheme.
- **Principal bundles** [Nakahara2003 Ch. 9–10, KobayashiNomizu Vol. I Ch. I–II, Frankel2011].
- **Covariance transport (sandwich)** [Nakahara2003 §10.3]: $\Sigma \mapsto \Omega\Sigma\Omega^\top$ for a vector / covariant 2-tensor under transformation $\Omega$.
- **Goldstone theorem (canonical)** [Weinberg1995 Vol. II §19, Peskin Schroeder Ch. 11]: spontaneous breaking of a continuous global symmetry yields a massless boson per broken generator. The Goldstone mode comes from internal dynamics, not from an external source field; explicit symmetry breaking (e.g., Zeeman term) does not give Goldstone modes.
- **Cencov's theorem** [Cencov1972]: Fisher metric is the unique invariant Riemannian metric on a statistical manifold up to scalar.
- **Rényi divergence (closed form for Gaussians)** [vanErven Harremoës 2014, Burbea Rao 1982].
- **f-divergence class** [LieseVajda1987, Csiszár Shields 2004]. KL ($\alpha=1$), Hellinger ($\alpha=1/2$), and other Rényi/α members.
- **EM and variational EM** [DempsterLairdRubin1977, BleiKuckelbirgJordan2017, Beal2003 thesis].
- **Edge modes** [DonnellyFreidel2016]; **quantum reference frames** [BartlettRudolphSpekkens2007, Vanrietvelde2020]; **relational interpretation** [Rovelli1996].

## What this evidence does NOT settle

1. Whether every displayed equation in §Theory has been verified against canonical Gaussian KL / Fisher / KL-gradient formulas at machine precision. The manuscript asserts symbolic verification "to machine precision" at line 1717 for one specific equation; no global verification status is given.
2. Whether the conditional uniqueness theorem in Appendix~\ref{app:conditional_uniqueness} (referenced at line 1038, 1112) actually closes the gap the mixture-of-sources construction acknowledges. (Verifying requires reading the appendix.)
3. Whether the multi-layer cascade where "the previous layer's posterior $\mu_q$ becomes the next layer's prior $\mu_p$" (referenced via the companion paper, e.g., line 630) constitutes a non-standard variational hierarchy in the sense flagged by `external_canon_inference.md:60`.
4. Whether the cross-scale shadow relation at Eq.~\eqref{eq:cross_scale_shadow} produces a true variational hierarchy with the full posterior passed up/down, or a deterministic point-passing scheme. Both teams should look at the cross-scale shadow definition.
5. Whether the per-coordinate precision derivation at lines 1330–1346 rigorously follows from the diagonal-Gaussian mean-field factorization, or whether it requires the additional Gamma-Normal conjugacy assumption flagged by Bishop §10.2.
6. Whether the TODO at line 1874 (deferring empirical test of $\omega^2\propto m_{\text{eff}}^{-1}$ to future work) violates sub-claim 6 of the operationalization (no unresolved gaps blocking reviewer acceptance), or whether it is the kind of "future work" stub that reviewers routinely accept.
7. Whether the Goldstone caveat at line 1512 ("suggestive but… analogy rather than derivation") removes the load-bearing nature of the SSB language used earlier in the section, or whether the section still relies on the analogy load-bearingly elsewhere.
8. Whether the dual-role gauge-frame treatment (Role A redundant / Role B physical, lines 557–565) is a clean theoretical move backed by the cited edge-modes / quantum-reference-frame literature, or whether it constitutes a covert "have your cake and eat it" device that papers over a contradiction.

The opening teams should pick the strongest 2–3 vectors per side from this list (and beyond) and synthesize.
