# Blue Panel Choice — pifb-dimensional-sectors (Phase 2, opening)

Mode: theory + formal math (spectral / linear algebra, dimension counting). Panel=full, 5-of-10.

No Agent dispatch tool is available in this environment; the five lenses are embodied by the coordinator and logged as per-lens memos in `02_blue_lens_memos.md`. Selection follows the `theory`/`math` mode defaults with the claim-keyword overrides applied (eigenvalue → numerical-analyst; the section is literally a bundle-section pullback → gauge-theorist replaces the default `variational`, which has no ELBO/EM purchase on this passage).

| Tag | Why this lens defends this claim |
|---|---|
| `philosophy-of-science` (mandatory) | Polices the speculative-vs-asserted boundary at :3067/:3071 and the "we may suppose" hedge at :3065; the section's own falsifiability self-labeling ("no quantitative predictions ... that could be tested") is exactly this lens's domain. |
| `geometer` | The load-bearing claim is a pullback fact: $G_i(c)=\sigma_i^* g_{\mathcal B}$ is an $n\times n$ tensor on $T_c\mathcal C$ with rank $\le n=\dim\mathcal C$ and is PSD. Lee Ch. 11 (pullbacks) / Ch. 13 (Riemannian) settles the base-vs-fiber bookkeeping. |
| `info-geometer` | $\dim(\mathcal B)=K+K(K+1)/2$ is the Gaussian statistical-manifold dimension (mean + symmetric covariance); the metric being pulled back is the Fisher–Rao metric, PSD by construction (Amari–Nagaoka; Cencov uniqueness). Confirms the fiber dimension and the PSD source. |
| `numerical-analyst` | $\lambda_a\ge 0$ with an orthonormal eigenbasis is the spectral theorem for symmetric PSD matrices; the eigenvalue-threshold partition is a well-defined Borel partition of the spectrum. This lens also flags the genuine vulnerability: thresholds are free, so the sector-size hierarchy is a posit, not a theorem. |
| `gauge-theorist` | The object is a section $\sigma_i:\mathcal C\to\mathcal B$ of a bundle and its differential $d\sigma_i:T_c\mathcal C\to T_{q_i}\mathcal B$; the "vast majority of dimensions invisible" claim is about the image of $d\sigma_i$ inside the fiber tangent space, not about eigen-directions. Nakahara Ch. 10 / Kobayashi–Nomizu §III settles that $d\sigma_i$ has rank $\le\dim\mathcal C$ and that unsampled fiber directions contribute nothing to the pullback. |

`variational` discounted: the passage contains no ELBO, EM-boundary, mean-field, or FEP content — it is pure spectral linear algebra on a pullback metric. Its lens would add no citation the other four do not already cover better.
