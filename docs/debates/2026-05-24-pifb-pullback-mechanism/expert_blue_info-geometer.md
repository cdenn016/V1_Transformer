# Expert Memo (Blue) — Information Geometer

## Lens

Fisher-Rao metric, score functions, Cencov uniqueness. The vertical block of $g_{E_q}$ — the Fisher-Rao metric on the Gaussian fiber (:2716) and its appearance as the score outer product in the pullback (:2759) — is the piece I assess, and it is the strongest-pedigree component of the entire construction.

## Steelman of the claim

The vertical block is not a modeling choice that needs defending; it is the *forced* choice. On a statistical manifold, the Fisher-Rao metric is the unique Riemannian metric (up to scale) invariant under reparameterization by sufficient statistics. Any metric placed on the belief fiber that is *not* Fisher-Rao would fail Cencov invariance and would not deserve the name "information metric." So the manuscript's use of $g_{\mathcal{B}}$ as the vertical block is the canonically mandated object, and the identification of the vertical pullback with the score outer product is a textbook identity, not a derivation that could go wrong.

## Derivation from external canon

The Fisher information metric is defined as the score outer product:
$$g_{ij}(\theta) = \mathbb{E}_{p(x;\theta)}\!\big[\partial_i\log p\,\partial_j\log p\big] = -\mathbb{E}_{p(x;\theta)}\!\big[\partial_i\partial_j\log p\big]$$
under regularity [AmariNagaoka2000 Ch. 2; external_canon_math.md §"Fisher information metric"]. This is exactly the manuscript's vertical-block identity at :2759, $g_{\mathcal{B}}(\nabla_\mu q,\nabla_\nu q) = \mathbb{E}_q[(\nabla_\mu\log q)(\nabla_\nu\log q)]$: when the section's covariant derivative is expressed in fiber coordinates and paired in the Fisher-Rao metric, the result is the score outer product contracted with the parameter derivatives $\partial_\mu\theta_i\,\partial_\nu\theta_j$. The chain rule through the section is what turns the abstract $g_{\mathcal{B}}(\nabla q,\nabla q)$ into the concrete score expression. This is the standard "pullback of the Fisher metric along a parameterized family" and it is correct.

For Gaussians specifically, the closed form at :2716,
$$g_{\mathcal{B}}(\delta q,\delta q) = \delta\mu^\top\Sigma^{-1}\delta\mu + \tfrac12\mathrm{tr}(\Sigma^{-1}\delta\Sigma\,\Sigma^{-1}\delta\Sigma),$$
is the textbook Gaussian Fisher-Rao metric [AmariNagaoka2000; the mean block is $\Sigma^{-1}$, the covariance block is the $\tfrac12\mathrm{tr}(\Sigma^{-1}d\Sigma\,\Sigma^{-1}d\Sigma)$ form]. The worked example at :2806 reproduces this exactly, and the isotropic reduction to the conformal metric $G = \sigma^{-2}\,\partial_\mu\mu\cdot\partial_\nu\mu$ (:2814) is the correct specialization when $\Sigma=\sigma^2 I$ is constant (covariance-gradient term drops, mean block becomes $\sigma^{-2}I$). The "high certainty magnifies distance" reading is the correct qualitative consequence of $\sigma^{-2}$ scaling.

Cencov's theorem [Cencov1972; external_canon_math.md §"Cencov's uniqueness theorem"] is the load-bearing external authority: the Fisher metric is the *unique* invariant Riemannian metric on a statistical manifold up to scale. This means the manuscript could not have chosen a different vertical block without abandoning information-geometric meaning. The vertical block is therefore not merely *a* legitimate metric — it is *the* canonical one, and its appearance is mandatory once "information geometry" is the stated game.

## On the "$L^2$ vs bundle-metric" distinction from the info-geometry side

The score outer product $\mathbb{E}[(\nabla\log q)(\nabla\log q)]$ is *already* a bona-fide Riemannian metric pulled back from the statistical manifold — it is the pullback of the Fisher-Rao metric along the section, full stop. So from a pure information-geometry standpoint, calling it "merely $L^2$" undersells it: the $L^2$ inner product on log-density variations *is* the Fisher-Rao metric, by definition of Fisher-Rao [AmariNagaoka2000 Ch. 2]. The manuscript's distinction is therefore not "score pullback is sub-standard"; it is the narrower point that the score pullback metrizes the *fiber*, and the manuscript additionally wants a metric on the *total space* $E$ to include the horizontal frame-twist. That is a legitimate refinement, but I concede it adds nothing to the *vertical* block — the vertical block was already a genuine Fisher-Rao pullback before the bundle-metric framing was introduced. The bundle framing's only additive content is the horizontal $\kappa(A,A)$ piece (see gauge-theorist memo), not any change to the information-geometric vertical part.

## Falsification condition

The vertical-block claim fails if: (a) the manuscript's Gaussian Fisher-Rao form (:2716) differed from the canonical $\delta\mu^\top\Sigma^{-1}\delta\mu + \tfrac12\mathrm{tr}(\Sigma^{-1}\delta\Sigma\,\Sigma^{-1}\delta\Sigma)$ — it does not, it matches [AmariNagaoka2000]; or (b) the score-form identity at :2759 were not the Fisher metric — it is, by the definition of Fisher information as the score outer product [external_canon_math.md §"Fisher information metric"]; or (c) the manuscript symmetrized or otherwise broke Cencov invariance in defining the fiber metric — it does not. The information-geometric core is unassailable: the vertical block is the Cencov-unique metric and the pullback identity is textbook. The defensible weakness lives entirely in the horizontal block and the tier interpretation, not here.

## Newly-discovered canon

- [AmariNagaoka2000 Ch. 2] — Fisher metric as score outer product; Gaussian closed form; pullback of Fisher metric along a parameterized family.
- [Cencov1972] — uniqueness of the Fisher metric (up to scale) among reparameterization-invariant metrics on a statistical manifold; forces the vertical-block choice.
- [Nielsen2020 "An elementary introduction to information geometry," *Entropy* 22(10):1100] — modern restatement of the Fisher-Rao pullback and the score-form identity, freely available for verification.
