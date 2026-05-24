# Blue Lens Memos — pifb-dimensional-sectors (Phase 2)

Five embodied lenses. Each: steelman the opposing attack, derive the defense from external canon, give ≥1 external citation, and argue the relevant falsification condition is unmet. Source-of-truth precedence: the manuscript (`Attention/Participatory_it_from_bit.tex:2994–3074`) is the claim under evaluation; the defense is grounded in external canon, and the memo concedes wherever canon contradicts.

---

## Memo 1 — geometer

**Target sub-point:** Base-vs-fiber bookkeeping (:3019, :3045, :3073).

**Steelman of the attack.** A critic could say the section confuses two spectra: the eigenvalues of an object on the $\dim(\mathcal B)$-dimensional fiber versus an object on the $\dim(\mathcal C)$-dimensional base, and the "vast majority of dimensions are invisible" rhetoric is the kind of language that usually accompanies that confusion.

**Defense from canon.** The manuscript does not make that confusion; it pre-empts it. $G_i(c)=\sigma_i^* g_{\mathcal B}|_{q_i(c)}$ is a pullback of a $(0,2)$-tensor along the smooth section $\sigma_i:\mathcal C\to\mathcal B$. For any smooth map $\sigma:\mathcal C\to\mathcal B$ and any $(0,2)$-tensor $g$ on $\mathcal B$, the pullback is defined pointwise by $(\sigma^* g)_c(u,v)=g_{\sigma(c)}(d\sigma_c\,u,\,d\sigma_c\,v)$ for $u,v\in T_c\mathcal C$ [Lee2013, Ch. 11]. Three facts follow immediately and are exactly what the manuscript asserts:

1. $\sigma^* g$ is a tensor on $\mathcal C$, hence an $n\times n$ object with $n=\dim\mathcal C$ — matching :3019 verbatim ("$n\times n$ matrix with $n:=\dim(\mathcal C)$, and has at most $n$ eigenvalues").
2. $\operatorname{rank}(\sigma^* g)\le\operatorname{rank}(d\sigma_c)\le\min(\dim\mathcal C,\dim\mathcal B)=\dim\mathcal C$, so at most $n$ nonzero eigenvalues. This is the rank bound for a Gram-type pullback: $\sigma^* g=(d\sigma_c)^{\!\top} g_{\sigma(c)}(d\sigma_c)$ in coordinates, and $\operatorname{rank}(A^\top M A)\le\operatorname{rank}(A)$.
3. If $g$ is PSD then $\sigma^* g$ is PSD: $(\sigma^* g)_c(u,u)=g_{\sigma(c)}(d\sigma_c u, d\sigma_c u)\ge 0$ [Lee2013 Ch. 13 — pullback of a Riemannian/PSD metric].

The "invisible" rhetoric at :3045/:3073 is then explicitly relocated to the correct object: it is a statement about the image of $d\sigma_i$ inside $T_{q_i}\mathcal B$, i.e. that $d\sigma_i$ samples at most $\dim\mathcal C$ of the $\dim\mathcal B$ fiber directions. :3073 names the two senses and forbids conflating them. That is the disambiguation a careful differential geometer would demand, and it is present.

**External citation.** [Lee2013, Ch. 11 (pullbacks of covariant tensors), Ch. 13 (Riemannian metrics and induced/pullback metrics)].

**Falsification condition argued unmet.** The claim fails if $\sigma^* g_{\mathcal B}$ could have rank $>\dim\mathcal C$ or fail to be PSD. The pullback rank bound and PSD-preservation are theorems, not modeling choices; the condition is unmet.

---

## Memo 2 — info-geometer

**Target sub-point:** Fiber dimension $\dim(\mathcal B)=K(K+3)/2$ and the PSD source of $g_{\mathcal B}$.

**Steelman of the attack.** A critic could question whether $296{,}064$ is the right count and whether the metric being pulled back is genuinely PSD (if it were indefinite, the $\lambda_a\ge0$ claim would be unsupported).

**Defense from canon.** The Gaussian family $\{N(\mu,\Sigma):\mu\in\mathbb R^K,\ \Sigma\in\mathrm{Sym}^{++}_K\}$ is a smooth statistical manifold of dimension $K$ (mean) plus $K(K+1)/2$ (symmetric covariance) $=K(K+3)/2$ [AmariNagaoka2000, Ch. 2 — the multivariate-Gaussian model is a standard worked example]. The arithmetic is exact (verified): $768+\tfrac{768\cdot769}{2}=768+295{,}296=296{,}064=\tfrac{768\cdot771}{2}$. Both expressions in Eq. (:3014) evaluate identically.

The Fisher–Rao metric $g_{ij}(\theta)=E_{p_\theta}[\partial_i\log p\,\partial_j\log p]$ is positive semidefinite by construction as a Gram matrix of score functions, and positive definite where the score components are linearly independent [AmariNagaoka2000, Ch. 2]. By Cencov's uniqueness theorem it is the canonical metric to put on a statistical manifold [Cencov1972]. So $g_{\mathcal B}$ is PSD at the source, and by Memo 1 the pullback $G_i(c)$ inherits PSD — the spectral claim $\lambda_a\ge0$ at :3005 rests on a Gram-of-scores fact, not on an assertion.

**External citation.** [AmariNagaoka2000, Ch. 2]; [Cencov1972] (uniqueness of the Fisher metric).

**Falsification condition argued unmet.** The claim fails if $\dim(\mathcal B)\ne K(K+3)/2$ or if $K=768\not\Rightarrow296{,}064$, or if $g_{\mathcal B}$ were not PSD. The dimension count is the standard Gaussian-family count, the arithmetic checks, and the Fisher metric is PSD as a score-Gram matrix. Unmet.

---

## Memo 3 — numerical-analyst

**Target sub-point:** Spectral decomposition well-definedness; the three-sector partition.

**Steelman of the attack — and the genuine concession.** The strongest attack: the two thresholds $\Lambda_{\text{obs}},\Lambda_{\text{subthresh}}$ are free parameters, so the sector-size hierarchy $|\mathcal D_{\text{obs}}|\ll|\mathcal D_{\text{subthresh}}|\ll|\mathcal D_{\text{internal}}|$ (:3058) and the eigenvalue hierarchy (:3062) do not follow from anything — they are tuned, not derived. This attack lands, and the honest position concedes it: the orderings at :3058 and :3062 are a modeling posit, not a theorem of the spectral decomposition.

**Where the defense holds.** Concede the hierarchy; defend the decomposition's well-definedness. The spectral theorem for a real symmetric PSD matrix gives a real orthonormal eigenbasis with $\lambda_a\ge0$ [doCarmo1992 — symmetric operators; standard linear algebra, also Bhatia–Jain–Lim for the SPD setting]. Given any two cutoffs $\Lambda_{\text{subthresh}}\le\Lambda_{\text{obs}}$, the partition of the eigen-directions into $\{\lambda_a>\Lambda_{\text{obs}}\}$, $\{\Lambda_{\text{subthresh}}<\lambda_a\le\Lambda_{\text{obs}}\}$, $\{\lambda_a\le\Lambda_{\text{subthresh}}\}$ is a well-defined disjoint partition of the spectrum, and the three-term sum (:3050–3052) reconstructs $G_i$ exactly because the eigenprojectors are orthogonal and complete. So the decomposition is mathematically well-defined for any threshold choice; what is not derived is that real agents land in the $\ll$ regime. The manuscript's own framing supports the narrower reading: :3023–3042 introduce the sectors with "We define" — i.e. as definitions/partition, not as a derived hierarchy. The overclaim risk is confined to whether the $\ll$ ordering is asserted as fact, and it is hedged ("may comprise a large number of dimensions, depending on ...", :3037).

**External citation.** [doCarmo1992 — spectral theorem for symmetric operators]; [BhatiaJainLim2019 — PSD-matrix spectral structure].

**Falsification condition argued unmet (for the well-definedness claim).** The well-definedness claim fails if a symmetric PSD $G_i$ could lack a real orthonormal eigenbasis or non-negative eigenvalues, or if threshold partitioning were ill-defined. None holds. The *separate* claim that the hierarchy is derived is NOT defended — it is conceded as a posit.

---

## Memo 4 — gauge-theorist

**Target sub-point:** The differential $d\sigma_i$ and the "unsampled fiber directions" claim (:3019, :3045, :3073).

**Steelman of the attack.** A critic could argue that calling $\sim3\times10^5$ fiber directions "invisible" is hand-waving unless one can say precisely what is and is not in the image of the section's differential.

**Defense from canon.** A section $\sigma_i:\mathcal C\to\mathcal B$ of a bundle has a differential $d\sigma_i|_c:T_c\mathcal C\to T_{q_i(c)}\mathcal B$ that is a linear map between tangent spaces; its image is a subspace of dimension at most $\dim\mathcal C$ [Nakahara2003, Ch. 10 — sections and bundle maps; KobayashiNomizu Vol. I §III — differentials of bundle maps]. The pullback metric $G_i(c)$ depends only on $g_{\mathcal B}$ restricted to $\operatorname{im}(d\sigma_i)$: any fiber tangent vector outside that image never enters $(\sigma^* g)_c(u,v)=g(d\sigma u, d\sigma v)$ because $d\sigma u,d\sigma v\in\operatorname{im}(d\sigma_i)$ by definition. So the manuscript's statement at :3045 — "only $\dim(\mathcal C)$ directions in $T_{q_i(c)}\mathcal B$ are in the image of $d\sigma_i$, and the rest are unobserved by $G_i(c)$" — is exactly the image-of-the-differential fact, correctly stated. :3073 keeps the base-level reduction (which eigen-directions of $G_i$ are above threshold) cleanly separate from the fiber-level reduction (which fiber directions $d\sigma_i$ samples). These are two genuinely different reductions and the section refuses to conflate them.

**External citation.** [Nakahara2003, Ch. 10]; [KobayashiNomizu Vol. I, §III].

**Falsification condition argued unmet.** Fails if $\operatorname{rank}(d\sigma_i)$ could exceed $\dim\mathcal C$, or if fiber directions outside $\operatorname{im}(d\sigma_i)$ could contribute to the pullback. Neither is possible for a linear differential and a pullback metric. Unmet. (Scope caveat handed to philosophy-of-science: whether $\dim\mathcal C$ is itself large enough for the decomposition to be non-vacuous is a separate question the section explicitly brackets at :3021, "when $\dim(\mathcal C)$ is large enough to accommodate them.")

---

## Memo 5 — philosophy-of-science (mandatory)

**Target sub-point:** Speculative labeling (:3067–3071), the "we may suppose" hedge (:3065), and circularity of the spacetime identification.

**Steelman of the attack.** Two real pressure points. (a) "We may suppose that the observable sector corresponds to the $(1+3)$ dimensions we perceive as spacetime" (:3065) imports the explanandum — it identifies the large-eigenvalue sector with perceived spacetime, which is the very thing a theory of emergent spacetime is supposed to explain. (b) :3065 sits one subsection *before* the "(Speculative)" header, so a reader could take it as established before reaching the disclaimer.

**Defense from canon — and the cushioned concession.** Concede (a) as interpretive: the identification of the observable sector with perceived spacetime is a reading, not a derivation, and the section says so by mood — "we may suppose" (:3065), "we conjecture" (:3029), "we hypothesize" (:3069). On falsifiability grounds [Popper, *The Logic of Scientific Discovery*], a passage that (i) labels its central physical identification "(Speculative)" in the subsection header (:3067), (ii) enumerates three candidate mechanisms as alternatives rather than asserting one, and (iii) states in plain text "it makes no quantitative predictions about measured spacetime properties that could be tested experimentally" (:3071) is not making a covert empirical claim — it is explicitly flagging an unfalsifiable conjecture as such. That is the honest move, not the dishonest one: the failure mode philosophy-of-science polices is the reverse, dressing speculation as established result. Here the labeling runs the other direction.

On the temporal sign: :3029 states the sign "is not intrinsic to the positive semi-definite spectral decomposition ... it is imported from the postulate set of Section [signature_resolution] and inherits the open-problem status of that section." A PSD spectrum gives $\lambda_a\ge0$ only; Lorentzian signature requires an external indefinite-form postulate. The section correctly attributes the sign to an external, separately-flagged postulate rather than claiming the spectral decomposition delivers it. This is the correct division of labor and inherits Debate 2's REMAND honestly rather than smuggling its conclusion.

**The :3065 placement** is the residual weakness. The honest defense: the supposition is hedged in the subjunctive ("may suppose") and is immediately followed by the explicitly-labeled speculative subsection, so a charitable reading treats :3065 as the lead-in to the disclaimer, not a free-standing assertion. This is a concession-cushioned point, not a clean win — if the supposition leaks into *later* sections as established, that would falsify the "honestly labeled" claim, and that is the condition to watch.

**External citation.** [Popper, *The Logic of Scientific Discovery*, 1959 — demarcation / falsifiability]; the manuscript's own self-labeling at :3067/:3071 is the textual evidence, not a canonical authority.

**Falsification condition argued unmet.** The "honestly labeled" claim fails if the $(3+1)$ structure were asserted rather than hypothesized, or if a testable quantitative prediction were claimed, or if the supposition were used downstream as established. The header is "(Speculative)", the text disclaims testable predictions, and the mood is subjunctive throughout. Within :2994–3074 the condition is unmet; the only live risk (downstream leakage) is outside this span and is flagged for the judges.

---

## Newly-discovered canon (for merge into 01b_extended_evidence.md)

- [Lee2013, Ch. 11] — pullback of a covariant $(0,2)$-tensor by a smooth map $\sigma:\mathcal C\to\mathcal B$: $(\sigma^* g)_c(u,v)=g_{\sigma(c)}(d\sigma_c u, d\sigma_c v)$; rank $\le\dim\mathcal C$; in coordinates $\sigma^* g=(d\sigma)^\top g\,(d\sigma)$ so $\operatorname{rank}(\sigma^* g)\le\operatorname{rank}(d\sigma)$. [Lee2013, Ch. 13] — pullback of a PSD/Riemannian metric is PSD.
- [AmariNagaoka2000, Ch. 2] — multivariate Gaussian as a statistical manifold of dimension $K+K(K+1)/2$; Fisher metric as a PSD score-Gram matrix. Arithmetic verified: $768+295{,}296=296{,}064=768\cdot771/2$.
- [doCarmo1992] / [BhatiaJainLim2019] — spectral theorem for symmetric PSD operators: real orthonormal eigenbasis, $\lambda_a\ge0$; eigenprojectors orthogonal and complete (three-sector sum reconstructs $G_i$ exactly).
- [Nakahara2003, Ch. 10] / [KobayashiNomizu Vol. I §III] — differential of a bundle section $d\sigma_i:T_c\mathcal C\to T_{q_i}\mathcal B$ has rank $\le\dim\mathcal C$; pullback metric depends only on $g_{\mathcal B}|_{\operatorname{im}(d\sigma_i)}$.
- [Popper1959] — falsifiability / demarcation; the relevant standard for assessing whether a self-labeled "(Speculative)" passage that disclaims testable predictions is honestly labeled.
