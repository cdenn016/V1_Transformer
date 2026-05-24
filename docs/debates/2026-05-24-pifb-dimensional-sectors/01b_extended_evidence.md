# Extended Evidence — pifb-dimensional-sectors

Harvested canon appended by coordinators during the debate. Source-of-truth: external canon; the manuscript is the claim under evaluation.

## Blue panel (Phase 2 opening)

- [Lee2013, Ch. 11] — pullback of a covariant $(0,2)$-tensor by a smooth map $\sigma:\mathcal C\to\mathcal B$: $(\sigma^* g)_c(u,v)=g_{\sigma(c)}(d\sigma_c u, d\sigma_c v)$. Rank bound: in coordinates $\sigma^* g=(d\sigma)^\top g\,(d\sigma)$, so $\operatorname{rank}(\sigma^* g)\le\operatorname{rank}(d\sigma)\le\dim\mathcal C$. [Lee2013, Ch. 13] — pullback of a PSD/Riemannian metric is PSD (the quadratic form $u\mapsto g(d\sigma u,d\sigma u)\ge0$).
- [AmariNagaoka2000, Ch. 2] — multivariate Gaussian as a statistical manifold of dimension $K+K(K+1)/2=K(K+3)/2$ (mean + symmetric covariance); Fisher metric is PSD as a score-Gram matrix. Arithmetic verified: $768+\tfrac{768\cdot769}{2}=768+295{,}296=296{,}064=\tfrac{768\cdot771}{2}$ (both forms in Eq. :3014 agree).
- [Cencov1972] — Fisher metric is the unique (up to scale) metric on a statistical manifold invariant under sufficient statistics; canonical choice of $g_{\mathcal B}$.
- [doCarmo1992] / [BhatiaJainLim2019] — spectral theorem for symmetric PSD operators: real orthonormal eigenbasis, $\lambda_a\ge0$; orthogonal complete eigenprojectors, so the three-sector sum (:3050–3052) reconstructs $G_i$ exactly for any threshold pair.
- [Nakahara2003, Ch. 10] / [KobayashiNomizu Vol. I §III] — differential of a bundle section $d\sigma_i:T_c\mathcal C\to T_{q_i}\mathcal B$ has rank $\le\dim\mathcal C$; the pullback metric depends only on $g_{\mathcal B}$ restricted to $\operatorname{im}(d\sigma_i)$; fiber directions outside that image contribute nothing.
- [Popper1959] (*The Logic of Scientific Discovery*) — falsifiability / demarcation; standard for judging whether a self-labeled "(Speculative)" passage that disclaims testable predictions is honestly labeled.

## Red panel (Phase 2 opening)

New canon beyond the blue harvest above (deduplicated; Lee pullback-rank, Amari–Nagaoka Gaussian dimension, Čencov, and Popper are already listed and not repeated).

### Differential geometry — pullback is Riemannian iff immersion (degeneracy otherwise)
- Lee, *Introduction to Smooth Manifolds* (2nd ed., GTM 218), Ch. 13: the pullback $\sigma^* g$ is a genuine (positive-definite) Riemannian metric **iff** $\sigma$ is an immersion; otherwise it is degenerate (rank $<\dim\mathcal C$). When $\dim\mathcal C=0$ the pullback is the empty $0\times 0$ tensor with no eigenvalues. ProofWiki restatement: https://proofwiki.org/wiki/Pullback_of_Riemannian_Metric_by_Smooth_Mapping_is_Riemannian_Metric_iff_Mapping_is_Immersion

### Information geometry — eigenvalue magnitudes are coordinate-dependent
- Čencov 1972 / Amari–Nagaoka 2000 Ch. 2: Fisher invariance is invariance of the metric *tensor* under sufficient statistics, **not** invariance of the *numerical eigenvalues* of a pullback under arbitrary base reparameterization. Under $c\mapsto\psi(c)$ with Jacobian $J$, $\tilde G=J^{-\top}GJ^{-1}$, so eigenvalues rescale by the squared singular values of $J$ — an absolute threshold $\lambda_a>\Lambda_{\rm obs}$ is coordinate-dependent.

### Numerical analysis — a stable cluster partition requires a spectral gap
- Davis & Kahan 1970, "The rotation of eigenvectors by a perturbation. III," *SIAM J. Numer. Anal.* 7(1):1–46: the $\sin\theta$ theorem — cluster/subspace separability requires a nonzero eigengap $\delta$; subspace perturbation $\le\|E\|/\delta$, diverging as $\delta\to0$. Restatement: https://www.cs.columbia.edu/~djhsu/coms4772-f16/lectures/davis-kahan.pdf
- Stewart & Sun 1990, *Matrix Perturbation Theory*, Ch. V: gap-dependent invariant-subspace bounds.
- Bhatia 1997, *Matrix Analysis* (GTM 169), Ch. III: Weyl's inequality $|\lambda_k(A+E)-\lambda_k(A)|\le\|E\|$; eigenvalues in a dense region are not separably clusterable.

### Gauge theory / signature — a PSD form cannot be Lorentzian
- Nakahara 2003, *Geometry, Topology and Physics* (2nd ed.), §7.1–7.2: Lorentzian metrics are indefinite with $\ge1$ negative eigenvalue; signature is a basis-independent invariant. A PSD pullback ($\lambda_a\ge0$) cannot be Lorentzian.
- Sylvester's law of inertia (Horn & Johnson 2013, *Matrix Analysis* 2nd ed., Thm 4.5.8): positive/negative/zero eigenvalue counts of a symmetric form are invariant under real congruence; a PSD pullback has zero negative eigenvalues and cannot be made indefinite by a real base-coordinate change.

### Philosophy of science — explanatory direction
- Hempel & Oppenheim 1948, "Studies in the Logic of Explanation," *Philosophy of Science* 15(2):135–175: deductive-nomological adequacy — the explanandum must be a logical consequence of the explanans, not assumed by it (the (1+3) identification assumes perceived spacetime to *define* the observable sector).

## Blue panel (Phase 3 rebuttal)

In-manuscript downstream-leakage check of the :3065 supposition (no external canon; textual evidence that the "honestly labeled" falsification condition is unmet within the manuscript). All four downstream references to `sec:observable_sectors` re-hedge rather than asserting the (1+3) identification as established:
- `:3188–3190` — the eigenvalue hierarchy and "the choice of three large eigenvalues in $V_{\rm obs}$" are listed under "What is postulated"; closes "None of (i)–(iv) is derived from variational free-energy minimization."
- `:3192` — "We do not claim to have derived physical spacetime, the Lorentz signature, or the dimensionality count from the framework; the construction shows that the framework is compatible with such a reading under the postulates listed above."
- `:2858` — the 1+3 split "is fixed by the choice of a single imaginary frame direction rather than derived from free-energy dynamics, and the construction does not currently distinguish 1+3 from 2+2 or other indefinite splits on dynamical grounds."
- `:3598`/`:3600` — signature work filed under open questions; closes "it does not show that the framework derives it." `:3596` invokes Sylvester's law of inertia to explain why non-compactness alone does not flip the signature (the manuscript itself uses the gauge-theorist's Sylvester argument).

- [Lee2013, Ch. 13] — pullback $\sigma^* g$ is Riemannian (positive-definite) iff $\sigma$ is an immersion; degenerate but still a well-defined bilinear form otherwise, with $n=0$ the trivial empty form. Used by blue to distinguish "sound construction for all $n$" (claimed) from "non-trivially instantiated at $n\ge5$" (not claimed; bracketed at :3021).

## Red panel (Phase 3 rebuttal)

In-manuscript evidence (no new external canon; the external citations the rebuttal relies on — Čencov/Amari–Nagaoka reparameterization, Davis–Kahan/Bhatia gaps, Nakahara/Sylvester signature, Popper/Hempel — are all listed above under "Red panel (Phase 2 opening)" and are not repeated). The rebuttal turns the downstream concession back on the in-span soundness claim and pins the realized-construction dimensions:
- In-span vs downstream tension. Eqs. :3057–3058 ($|\mathcal D_{\rm obs}|\ll|\mathcal D_{\rm subthresh}|\ll|\mathcal D_{\rm internal}|$) and :3061–3062 ($\lambda_{\rm obs}\gg\lambda_{\rm subthresh}\gg\lambda_{\rm internal}\approx0$) are written *unconditionally* as properties the spectrum satisfies; the :3021 "when $\dim(\mathcal C)$ is large enough" clause governs sector *existence*, not the *ordering*. Downstream :3190 lists "the eigenvalue hierarchy itself, including the choice of three large eigenvalues in $V_{\rm obs}$" under "What is postulated" — so the manuscript itself concedes the in-span unconditional assertion is an imposed posit, not a spectral property.
- Realized-construction $\dim(\mathcal C)$. The only computationally validated construction is the transformer limit at $\dim(\mathcal C)=0$ (:482 "For transformers ($\dim\mathcal C=0$)"; :1001, :1613 — empty $0\times0$ $G_i$, no spectrum). The one concrete geometric example is $\mathcal C=\mathbb R^2$ (:2803), $\dim=2$ — cannot fill $|\mathcal D_{\rm obs}|=4$, let alone three nested sectors. No construction in the manuscript instantiates $\dim(\mathcal C)$ large enough to exhibit the three-tier hierarchy; the separation has no witness in any realized case.
- :3029 in-span overread. "the 4-dimensional sector admits a $(1+3)$ Lorentzian decomposition" is inside the debated span; the PSD spectrum admits only a Euclidean 4-subspace ($\lambda_a\ge0$). The Lorentzian sign is imported from the signature postulates (Debate 2 REMAND). Sylvester invariance [Nakahara §7.1–7.2; Horn–Johnson Thm 4.5.8] forbids obtaining the indefinite signature from the PSD pullback by any real base-coordinate change.
