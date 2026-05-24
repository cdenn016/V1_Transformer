# Expert Memo (Blue) — Differential Geometer

## Lens

Bundle metrics, horizontal/vertical splits, pullback by a section. The object under dispute, $g_{E_q}$ (Eq. bundle_metric, :2734), is a metric on the total space of a fiber bundle built from a base form, a connection-induced horizontal/vertical split, and a fiber metric. I judge it against the standard differential-geometry constructions of this exact shape.

## Steelman of the claim

The construction $g_{E_q}(X_H+X_V,Y_H+Y_V) := g^{\mathrm{tw}}_{\mathcal{C}}(\pi_*X_H,\pi_*Y_H) + g_{\mathcal{B}}(X_V,Y_V)$ is not an idiosyncratic invention. It is the textbook recipe for putting a metric on the total space of a bundle once a connection is chosen. The two canonical instances are the Sasaki metric on a tangent bundle and the Kaluza–Klein metric on a principal bundle. Both build the total-space metric as a block sum: horizontal block pulled from a base metric via the bundle projection, vertical block from a fiber metric, with the two blocks declared orthogonal by the connection-determined splitting. The manuscript's $g_{E_q}$ is structurally identical: it is a Sasaki/Kaluza–Klein-type bundle metric with the Fisher-Rao metric playing the role of the fiber metric and $\kappa(A,A)$ playing the role of the base/horizontal block.

The pullback $G_i^{(q)} = (\sigma_i^{(q)})^* g_{E_q}$ is then the ordinary pullback of a (0,2)-tensor by a smooth map (the section). The tangent-map decomposition $\sigma_{i,*}\partial_\mu = (\partial_\mu)^H + \nabla^{(i)}_\mu q_i$ is just the splitting of $d\sigma$ into horizontal and vertical parts; the vertical part of a section's differential is by definition the covariant derivative of the section [KobayashiNomizu Vol. I §III, associated-bundle covariant derivative]. Feeding this through the block-diagonal $g_{E_q}$ gives $G^{(q)}_{\mu\nu} = \kappa(A_\mu,A_\nu) + g_{\mathcal{B}}(\nabla_\mu q,\nabla_\nu q)$ with no cross term, exactly :2755.

## Derivation from external canon

The Sasaki metric (Sasaki 1958; standard exposition in [KobayashiNomizu Vol. II], Yano–Ishihara *Tangent and Cotangent Bundles*) takes a Riemannian $(M,g)$ and metrizes $TM$ by: declaring the projection $\tau:TM\to M$ a Riemannian submersion, giving each fiber the induced Euclidean metric, and making horizontal and vertical distributions orthogonal. The horizontal distribution is determined by the Levi-Civita connection. The total-space metric is then $g^S(X,Y) = g(\tau_*X^H,\tau_*Y^H) + g(K X^V, K Y^V)$ where $K$ is the connection map. This is precisely a block-diagonal $g_{\mathcal{C}}(\text{horizontal}) \oplus g_{\text{fiber}}(\text{vertical})$ form. The WebFetch of the Sasaki construction confirms the three defining properties: submersion, fiber-induced metric, orthogonality of parallel/horizontal to fiber directions.

The Kaluza–Klein bundle metric [Bleecker1981 *Gauge Theory and Variational Principles*; Frankel2011 Ch. 20] does the same on a principal $G$-bundle $P\to M$ with connection $A$: $g_P = \pi^* g_M \oplus \langle\cdot,\cdot\rangle_{\mathfrak{g}}\circ(A\otimes A)$, i.e. the vertical block is an Ad-invariant inner product on $\mathfrak{g}$ pulled back through the connection one-form, and the horizontal block is the base metric pulled up. The manuscript's $\kappa:\mathfrak{g}\times\mathfrak{g}\to\mathbb{R}$ with $\kappa = -\mathrm{tr}(\cdot\,\cdot)$ on a compact form is exactly the standard negative-Killing/negative-trace inner product that makes $\mathfrak{so}(N)$ inner products positive-definite [KobayashiNomizu Vol. II, on bi-invariant metrics from the Killing form]. The sign convention $-\mathrm{tr}$ for compact, $+\mathrm{tr}$ for non-compact is the standard one for the trace form on matrix Lie algebras, since $\mathrm{tr}(A^2)\le 0$ for skew-symmetric $A$.

So both blocks of $g_{E_q}$ have an exact canonical pedigree: the vertical block is a fiber metric (Fisher-Rao), the horizontal block is a connection-pulled inner-product form on the base, and the orthogonal direct sum is the Sasaki/Kaluza–Klein recipe.

## The "$L^2$ vs bundle-metric" distinction — what is real and what is bookkeeping

The :2726 distinction is mathematically real but its operational content is narrow, and the honest defense states this exactly. Real content: the $L^2$ score pullback $\mathbb{E}[(\nabla\log q)(\nabla\log q)]$ pulls back a metric on the *fiber only* — it never specifies a metric on the total space $E$, so calling it "the pullback of a bundle metric" is, strictly, a category error. Specifying $g_{E_q}$ on $E$ repairs the category: the score expression is then literally the vertical block of a genuine total-space metric, and the full pullback is a pullback of that total-space metric. That is a legitimate upgrade in mathematical standing.

But the upgrade adds *numerical* content only through the horizontal block. When $A^{(i)}=0$ (gauge frame locally constant, :2771), $\kappa(A,A)$ vanishes and $G^{(q)}$ collapses to the bare score outer product — numerically identical to the $L^2$ pullback. The bundle-metric framing earns its name as a *construction* (a metric on $E$ is specified), not as a *new formula* in the flat-frame case. I concede this precisely: the formal upgrade is real; its extra numerical content is exactly the $\kappa(A,A)$ piece and nothing else.

## A point Red will attack — pre-empted

The orthogonality of the two blocks (and hence the vanishing cross term at :2755) is **by definition**, not by theorem. $g_{E_q}$ is *declared* block-diagonal at :2734. So "gauge-orthogonal by construction" (:2747) should be read as a defining choice, not a derived identity. This is not a defect: Sasaki and Kaluza–Klein make the identical definitional choice — the connection's job is precisely to define which directions are horizontal so that the block split is well-posed. The cross term vanishes because the metric was built to make it vanish, which is standard and legitimate, but the manuscript's language must be read as definitional.

## Falsification condition

The geometric claim fails if either: (a) the horizontal distribution $H_{(c,q)}$ is not a well-defined complement to $V$ — i.e. if the connection one-form $A^{(i)}=U_i^{-1}dU_i$ does not satisfy the two connection axioms [external_canon_math.md §"Connection on a principal bundle": $A(\xi^*)=\xi$ and $R_g^*A=\mathrm{Ad}(g^{-1})A$], so that $T_{(c,q)}E = H\oplus V$ does not hold; or (b) the vertical block is not a genuine Riemannian metric on the fiber. Neither holds: $A=U^{-1}dU$ is the Maurer–Cartan-pulled connection of a gauge frame, satisfying the axioms by construction [Nakahara2003 §10.5], and the Fisher-Rao metric is positive-definite on the open cone of non-degenerate Gaussians. The block sum of a positive-definite horizontal form (compact case, $-\mathrm{tr}$) and a positive-definite vertical form is positive-definite, so $g_{E_q}$ is Riemannian in the body regime. The construction stands as a Sasaki/Kaluza–Klein-class bundle metric.

## Newly-discovered canon

- Sasaki, S. (1958). "On the differential geometry of tangent bundles of Riemannian manifolds." *Tohoku Math. J.* 10. — Canonical block-diagonal total-space metric from a connection-induced H/V split.
- Yano, K. & Ishihara, S. (1973). *Tangent and Cotangent Bundles: Differential Geometry*. Marcel Dekker. — Standard exposition of the Sasaki construction and the connection map $K$.
- [Bleecker1981 Ch. on Kaluza–Klein] and [Frankel2011 Ch. 20] — Kaluza–Klein bundle metric as base $\oplus$ Ad-invariant-fiber via the connection one-form; direct structural analogue of $g_{E_q}$.
