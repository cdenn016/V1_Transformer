# Evidence Pack — pifb-theory-foundations

## Manuscript references (lines 483-911)

Key structural definitions and load-bearing claims:

- `:483-493` §Base Manifold $\mathcal{C}$ definition with "Conceptual point: not physical spacetime"
- `:495-534` §Statistical Manifolds — Gaussian families with Fisher-Rao metric; Cencov 1982 uniqueness citation at 510
- `:536-548` §Cross-Scale Shadows — Eq. eq:cross_scale_shadow $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$; "structural commitment, not a theorem of standard hierarchical variational inference"; Appendix A reduction cited (Lemmas aug_joint_welldefined and shadow_mf_optimum)
- `:550-587` §Principal Bundles — Two roles framing at 557; Dual-role rigor citing Donnelly-Freidel 2016 (edge modes), Bartlett-Rudolph-Spekkens 2007 / Vanrietvelde 2020 (quantum reference frames), Rovelli 1996 (relational) at 561; Local trivialization caveats at 577; exp-map non-compactness caveat at 581
- `:589-610` §Associated Bundles — Convention statement at 604 explicitly registering $(n\cdot g, b) \sim (n, \rho(g)b)$ vs alternative
- `:612-648` §Agents as Smooth Sections — Definition with primitive $(q_i, s_i)$ and derived $(p_i, r_i)$ sections; gauge frame field; variational hierarchy $r \to s \to p \to q \to \text{observations}$
- `:650-764` §Multi-Agent Systems with subsections: Multi-Agent Configuration (652-666), Consensus and Meta-Agent Formation (668-695), Culture as Coarse-Grained Slow-Channel Identity (697-720), Epistemic Collapse and Information Death (720-752), Hierarchical Structure (752+)
- `:765-880` §Cognitive Reference Frames as Gauge Frames — Transport Operators (778), Gauge Covariance (791), Connection Forms and Parallel Transport (797), Gauge Field Strength (816), Discrete Regime II via edge-relaxed cocycle (828)
- `:882-895` §Hierarchy of Transport Operators
- `:897-910` §Curvature Structure: Four Interacting Geometries
- `:911-1004` §Working Framework: Simplifications and Scope

## Canon excerpts (teams should expand)

### Differential geometry / bundle canon
- **Nakahara, M. (2003)**, *Geometry, Topology and Physics*, 2nd ed., IoP — already cited at 591
- **Frankel, T. (2011)**, *The Geometry of Physics*, 3rd ed., Cambridge — already cited at 591
- **Kobayashi, S., Nomizu, K. (1963)**, *Foundations of Differential Geometry* Vol. 1

### Information geometry canon
- **Amari, S., Nagaoka, H. (2000)**, *Methods of Information Geometry*, AMS
- **Cencov, N. N. (1982)**, *Statistical Decision Rules and Optimal Inference* — already cited at 510 for Fisher metric uniqueness

### Edge modes / quantum reference frames canon
- **Donnelly, W., Freidel, L. (2016)**, "Local subsystems in gauge theory and gravity," JHEP — already cited at 561
- **Bartlett, S. D., Rudolph, T., Spekkens, R. W. (2007)**, "Reference frames, superselection rules, and quantum information" — already cited at 561
- **Vanrietvelde, A., et al. (2020)**, "A change of perspective: switching quantum reference frames" — already cited at 561
- **Rovelli, C. (1996)**, "Relational quantum mechanics" — already cited at 561

### Hierarchical VI canon
- **Friston, K. (2017)**, "Active inference: a process theory" — already cited at 546
- **Parr, T., Pezzulo, G., Friston, K. (2022)**, *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*, MIT Press — already cited at 546
- **Sønderby, C. K., et al. (2016)**, "Ladder Variational Autoencoders" — already cited at 546

## What this evidence does NOT settle

1. **Cencov 1982 uniqueness in 1D.** The statement at 510 that the Fisher-Rao metric is "unique up to scaling as the only Riemannian metric on probability spaces invariant under sufficient statistics" has nuance: Cencov's theorem holds for finite probability spaces and extends to certain infinite-dimensional cases; the exact scope should be verified.

2. **Cross-Scale Shadow Appendix A reduction.** Lemmas aug_joint_welldefined and shadow_mf_optimum are referenced at 546 but their actual statements were debated in earlier debates (commit b6c7b71d — "new appendix discharges cross-scale shadow reduction obligation"). Whether the current text accurately summarizes those lemmas.

3. **Two-roles framing scope.** The Role A (gauge-redundant transport) / Role B (frame-as-state ontology) split is honest, but the cited antecedents (edge modes, quantum reference frames, Rovelli) handle related but distinct ambiguities. Whether the analogy is load-bearing or just suggestive.

4. **Definition of Meta-Agent stability.** The definition at 684-691 specifies three conditions (coherence threshold, multi-child support, minimum cluster size). Whether these jointly capture "what is ordinarily called a culture" as the Culture subsection at 697 attempts to argue.

5. **Connection forms and gauge field strength at 797-816.** Standard Yang-Mills setup; verify the framework's use is consistent.
