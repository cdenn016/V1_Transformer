# Evidence Pack — pifb-pullback-mechanism (Debate 1)

Neutral fact pack. Coordinators/experts read the manuscript lines and canon directly.

## Manuscript references (Attention/Participatory_it_from_bit.tex)

- `:2716–2720` — Fisher-Rao metric on the Gaussian fiber $g_{\mathcal{B}}(\delta q,\delta q)=\delta\mu^\top\Sigma^{-1}\delta\mu+\tfrac12\mathrm{tr}(\Sigma^{-1}\delta\Sigma\Sigma^{-1}\delta\Sigma)$.
- `:2726` — claims the score-only $L^2$ pullback "does not earn the name pullback of a fiber-bundle metric because no metric on the total space $E$ has been specified"; sets up the bundle-metric construction.
- `:2728–2732` — tangent-space split $T_{(c,q)}E_q = H\oplus V$ via connection one-form $A^{(i)}=U_i^{-1}dU_i$; $V\cong T_q\mathcal{B}$.
- `:2734` (Eq. `bundle_metric`) — $g_{E_q}(X_H+X_V,Y_H+Y_V):=g^{\mathrm{tw}}_{\mathcal{C}}(\pi_*X_H,\pi_*Y_H)+g_{\mathcal{B}}(X_V,Y_V)$.
- `:2739` (Eq. `horizontal_metric`) — $g^{\mathrm{tw}}_{\mathcal{C},\mu\nu}:=\kappa(A^{(i)}_\mu,A^{(i)}_\nu)$, with piecewise $\kappa=-\mathrm{tr}$ (compact, pos-def) or $+\mathrm{tr}$ (non-compact, indefinite) at :2744–2745.
- `:2747` — labels it "tw" not "YM"; states $g^{\mathrm{tw}}$ is connection-dependent and generally not gauge-invariant; blocks "gauge-orthogonal by construction".
- `:2753–2763` — section tangent-map decomposition $\sigma_{i,*}\partial_\mu=(\partial_\mu)^H+\nabla^{(i)}_\mu q_i$; pullback $G^{(q)}_{i,\mu\nu}=g^{\mathrm{tw}}_{\mu\nu}+g_{\mathcal{B}}(\nabla_\mu q,\nabla_\nu q)$; vertical piece = score outer product; boxed Eq. `induced_metric_full`.
- `:2768` — gauge-invariance disclosure: under $U_i\to U_i g(c)$, $A\to g^{-1}Ag+g^{-1}dg$, $\kappa(A,A)$ not invariant; $F_{\mu\nu}=0$ in Regime I so no YM escape; invariance routed to consensus metric (Section consensus_metric).
- `:2771–2782` — score-only is the $A=0$ special case; same construction gives $G^{(p)},G^{(s)},G^{(r)}$; four tensors "coexistent rather than alternative".
- `:2786–2799` (`sec:three_tiers`) — epistemic $G^{(q)}$, expectational $G^{(p)}$, structural $G^{(s)}/G^{(r)}$; proposes perceived space $=G^{(s)}$; Kant/Wheeler/Clark/Seth/Hoffman/Friston citations; honest caveat that Hoffman operates at species-evolutionary not within-lifetime level.
- `:2801–2817` — Gaussian-on-$\mathbb{R}^2$ worked example; isotropic case → conformal metric $G=\sigma^{-2}\partial_\mu\mu\cdot\partial_\nu\mu$.

## Canon excerpts (.claude/agents/vfe-knowledge/external_canon_math.md)

- `:16–24` — Fisher information metric; Cencov uniqueness (the unique invariant metric on a statistical manifold up to scale).
- `:35` — KL ≈ ½ Fisher second-order.
- `:51` — dual e-/m-connections on a statistical manifold (Amari–Nagaoka Ch. 3).

## What this evidence does NOT settle (experts should fetch external canon)

- Whether a fiber-bundle metric of the form "horizontal (base) ⊕ vertical (fiber)" via a connection is a standard, well-defined object (canonical refs: Kobayashi–Nomizu *Foundations of Differential Geometry*; Nakahara *Geometry, Topology and Physics* ch. 9–10 on connections and associated bundles; the "bundle metric / Kaluza–Klein metric / Sasaki metric" construction). Experts should WebFetch / cite a textbook for the canonical form and check whether the manuscript's $g_{E_q}$ matches it.
- Whether the horizontal block being a *connection-dependent, gauge-noninvariant* quadratic form disqualifies it from being part of a genuine bundle metric, or whether bundle metrics are routinely connection-dependent (the canonical answer: a metric on the total space built from a connection + base metric + fiber metric is standard, e.g. Kaluza–Klein; gauge-noninvariance of the horizontal block under a change of section/trivialization is expected, not a defect — but the experts must confirm).
- Whether the "$L^2$ pullback of scores vs bona-fide bundle-metric pullback" distinction at :2726 is substantive (does specifying $g_{E_q}$ add genuine content, or is the resulting $G^{(q)}$ formula the same object relabeled, given the cross term vanishes and the score term is unchanged?).
- Whether "perceived space = structural tier $G^{(s)}$" is a falsifiable / well-posed identification or a metaphysical preference; whether the active-inference citation [Friston2017] supports locating structural perception in parameter-learning vs state-inference.
