# Expert Memo — Red / Info-Geometer — pifb-pullback-mechanism (opening)

## Steelman

The vertical block is exactly the Fisher-Rao metric in score form, $\mathbb E_{q}[(\nabla_\mu\log q)(\nabla_\nu\log q)]$ (:2759), which for the Gaussian fiber reduces to $\delta\mu^\top\Sigma^{-1}\delta\mu + \tfrac12\mathrm{tr}(\Sigma^{-1}\delta\Sigma\,\Sigma^{-1}\delta\Sigma)$ (:2718). This is textbook and Cencov-canonical; the score-outer-product identity is standard. The isotropic worked example $G=\sigma^{-2}\,\partial_\mu\mu\cdot\partial_\nu\mu$ (:2814) is the correct conformal pullback. So the *vertical* content of $G^{(q)}$ is a legitimate information-geometric object, and the manuscript's claim that the pullback carries Fisher information is sound for that block.

## Strongest falsification

The boxed $G^{(q)} = \kappa(A_\mu,A_\nu) + \mathbb E_q[(\nabla_\mu\log q)(\nabla_\nu\log q)]$ (:2763) is presented as a single induced metric of information-geometric pedigree. It is not an information metric. Cencov's (Chentsov's) theorem states that the Fisher metric is the *unique* Riemannian metric on a statistical manifold invariant under sufficient statistics, up to scale [Cencov1972; Fisher information metric, Wikipedia]. The Fisher piece earns its invariance from this theorem. The $\kappa(A_\mu,A_\nu)$ piece is constructed from the connection one-form $A^{(i)}=U_i^{-1}dU_i$, which is a *frame* object living on the principal bundle, not a function of the statistical family $\{q(c)\}$ at all. Adding it to the Fisher metric produces a quadratic form that is no longer characterized by Cencov invariance: it depends on a choice (the gauge frame $U_i$) external to the statistical manifold, and is admitted to change under re-gauging (:2768). A sum "Fisher + frame-dependent term" is not invariant under sufficient statistics, so by Cencov it is not *the* information metric; it is the Fisher metric plus a non-information additive piece.

This sharpens the $L^2$-vs-bundle distinction at :2726. The manuscript says the score-only expression is "merely $L^2$" while $G^{(q)}$ is a "bundle-metric pullback." Information-geometrically, the *only* part with information content is the vertical Fisher block — and that is identical in both readings. The difference is solely the additive $\kappa(A,A)$, which carries zero statistical-distinguishability content (it does not depend on $q$). So the upgrade from "$L^2$ score pullback" to "bundle-metric pullback" adds no information geometry; it adds a frame-rotation form. Calling the sum an information metric on $\mathcal C$ conflates a Cencov-canonical object with a frame artifact.

## External citation

[Fisher information metric, Wikipedia, quoting Chentsov's theorem]: "the Fisher information metric on statistical models is the only Riemannian metric (up to rescaling) that is invariant under sufficient statistics." [AmariNagaoka2000 Ch. 2; Cencov1972]: the Fisher metric $g_{jk}=\mathbb E[\partial_j\log p\,\partial_k\log p]$ is uniquely determined on a statistical manifold by invariance; any additive term that depends on structure external to the family $\{p(\cdot;\theta)\}$ breaks this uniqueness and is not part of the information metric. [external_canon_math.md §1, "Cencov's uniqueness theorem"]: "Any metric that fails this invariance is not a valid information metric."

## Falsification condition

My falsification is wrong if $\kappa(A_\mu,A_\nu)$ can be shown to depend on the statistical family $\{q_i(c)\}$ (so that it carries statistical-distinguishability content and the sum remains Cencov-invariant). It cannot: $A^{(i)}=U_i^{-1}dU_i$ is a function of the gauge frame $\phi_i$ alone (:2732), independent of $q$ — the manuscript itself recovers the score-only metric by setting $A=0$ at constant frame (:2771), confirming $\kappa(A,A)$ is orthogonal to the statistical content.

## Newly-discovered canon

- Wikipedia, "Fisher information metric": $g_{jk}(\theta)=\mathbb E_{x\sim p(x|\theta)}[\partial_j\log p\,\partial_k\log p]$; "By Chentsov's theorem, the Fisher information metric on statistical models is the only Riemannian metric (up to rescaling) that is invariant under sufficient statistics." URL: https://en.wikipedia.org/wiki/Fisher_information_metric
- Reinforces [external_canon_math.md §1] / [Cencov1972]: a metric failing sufficient-statistic invariance is not a valid information metric.
