# Extended Evidence — pifb-spec-ext-rock-solid

Harvested external canon from the round-2 (opening) expert memos. Concatenated and deduplicated. The judges should read this alongside `01_evidence.md`.

## Round 2 — Red panel additions

### Philosophy of science

- **Popper, *The Logic of Scientific Discovery* (1959)** — falsifiability as demarcation. Quoted from [SEP "Karl Popper" §3, "The Problem of Demarcation"]: "if a theory is incompatible with possible empirical observations it is scientific; conversely, a theory which is compatible with all such observations [...] is unscientific." [SEP §4]: a theory must be *logically structured* to permit empirical refutation in principle. Operative point: falsifiability requires refutation by *possible observations* in the present logical structure of the theory, not by hypothetical future-construction tests. Applies to the 3084 dimensionless-constants research program criterion.

- **Lakatos, *Methodology of Scientific Research Programmes* (1978) FMSRP:33–34** — quoted from [SEP "Imre Lakatos" §2.2]: progressive programmes are *theoretically progressive* ("each successive theory must exhibit excess empirical substance by forecasting previously unknown phenomena") and *empirically progressive* ("some of those novel predictions must be validated through observation and testing"). Degenerative programmes "either fail to generate new predictions or produce novel forecasts that are systematically falsified." Operative point: 3084 is structured as the *promise* of a progressive shift, not a progressive shift itself; the construction the criterion conditions on does not exist in the manuscript.

- **SEP "Structural Realism" §§3, 3.2 [Worrall1989, Ladyman1998]** — ESR is "a purely epistemological modification of scientific realism"; OSR makes metaphysical claims. Ladyman: "structural realism should be thought of as metaphysically rather than merely epistemically revisionary"; specifically that "the problem of ontological discontinuity is arguably left untouched by simply adopting Ramsification." Held in reserve: the 3092 "agnostic among these variants" claim is contestable but registered correctly in the present round.

### Information geometry

- **Cencov1972 / [AmariNagaoka2000 §2.3]** — the Fisher metric is the unique Riemannian metric on a statistical manifold invariant under sufficient statistics, *up to a positive scalar multiple*. Operative point: Cencov is a scale-removing theorem, not a scale-fixing one. The 3084 reference to "Cencov scale fixings" misnames the canonical primitive — Cencov does not provide scale fixings; the load-bearing primitive in the deferred construction is not delivered by the named theorem.

- **[AmariNagaoka2000 §2.5; KullbackLeibler1951]** — the Fisher metric in score-function form $g_{ij}(\theta) = \mathbb{E}[(\partial_i \log p)(\partial_j \log p)]$. The bundle metric vertical block at eq:induced_metric_full (PIFB:2717) correctly uses this form. The horizontal block $\kappa(A^{(i)}_\mu, A^{(i)}_\nu)$ is a stipulated direct-summand, not a pullback in the same sense.

### Gauge theory

- **Gribov 1978 *Nucl. Phys. B* 139, 1; Singer 1978 *Comm. Math. Phys.* 60, 7 ("Some Remarks on the Gribov Ambiguity"); Wikipedia "Gribov ambiguity" for the modern statement** — in non-abelian gauge theory, no globally well-defined gauge-fixing condition exists; "a gauge fixing submanifold may not intersect a gauge orbit at all or it may intersect it more than once." The Faddeev-Popov procedure is locally well-defined but globally ambiguous; the standard remedy is restriction to the first Gribov region, which does not solve the problem globally. Operative point: the consensus-metric construction at PIFB:2922–2937 sits on top of this 50-year-old open problem in lattice gauge theory; the manuscript's "needs a regulator" at 2928 understates the difficulty.

- **[Folland1995 *A Course in Abstract Harmonic Analysis* Ch. 2]** — a non-compact locally compact group has a left-invariant Haar measure but it is infinite. For $\mathrm{SO}(1,3)$ specifically, the Haar measure is infinite even for constant elements. Manuscript 2928 cites this correctly. Operative point: this is an *additional* obstruction on top of the local-$g(c)$ functional-integral / Gribov obstruction; both are present in the construction the manuscript names "consensus metric."

- **[Zinn-Justin 2002 *Quantum Field Theory and Critical Phenomena* §3; Hartle-Hawking 1983 *Phys. Rev. D* 28, 2960]** — Wick rotation canonically continues a *real spacetime coordinate* and produces a *real* metric throughout. The PIFB:2820–2846 construction continues a *Lie-algebra component* and produces a complex-valued bilinear form whose real part is projected. Operative point: the second of the two structural disanalogies (which space gets continued) is not retracted by the manuscript's "performed inside the gauge frame rather than on the base coordinates" framing.

- **[Nakahara2003 §10.3; KobayashiNomizu Vol. I §III.2]** — the Maurer-Cartan piece $g^{-1} dg$ is the standard inhomogeneous term in the local-gauge transformation of a connection. Manuscript 2722–2723 identifies it correctly.

### Variational inference / FEP

- **[external_canon_inference.md §1]** — multi-agent coupling terms of the form $\Sigma_{ij} \beta_{ij} \mathrm{KL}(q_i \| \Omega_{ij} q_j)$ are user-introduced; the standard FEP literature is single-agent (or hierarchical with a single ancestral generative model). Multi-agent extensions exist (variational ecology [Ramstead2020], graphical brain [Friston2017Graphical]) but use different couplings. Operative point: §Speculative Extensions rests on a user-novel scaffold; its only empirical validation in the present manuscript is via §Results data the user has flagged as placeholder.

- **[BleiKuckelbirgJordan2017; ParrPezzuloFriston2022 Ch. 2]** — variational free-energy minimization is canonically a perception-action-learning principle. It does not provide a canonical mechanism for deriving dimensionless physical constants from a statistical-manifold geometry. The 3082 commitment lives outside the canonical scope of variational inference; sub-claim 7 is weakened from this second canon direction (in addition to philosophy-of-science's deferred-falsifiability attack and the info-geometer's Cencov-not-a-scale-fixing attack).

### Differential geometry

- **[Lee2013 Ch. 11]** — pullback of a (0,2)-tensor by a smooth section is a well-defined tensor on the base manifold. The construction at PIFB:2705 is geometrically correct as far as it goes; the issue is the rhetorical lift at 2738, not the math.

- **[Nakahara2003 Ch. 10.3, §11.1; KobayashiNomizu Vol. I Ch. III; Frankel2011 Ch. 17]** — the associated-bundle horizontal-vertical decomposition with horizontal subspaces specified by the connection is the standard construction. PIFB:2682–2685 cites this correctly.

## Round 2 — Blue panel additions

### Philosophy of science

- **Popper, *Conjectures and Refutations* (1963), Ch. 1 §I [SEP "Karl Popper" §3]** — positive characterization of bold conjectures as the engine of science; the demarcation rule applies at the *programme* level, not at the level of every individual sentence within a paper. Popper distinguishes "heuristic content" (need not be presently testable) from "empirical content" (must be) explicitly. Operative point: a manuscript that registers its speculative content as outside present empirical reach is in canonical compliance with the Popperian standard provided the boundary between heuristic and empirical content is honestly drawn. The 2582 Outlook bracket and the 3127–3145 boxed enumeration constitute exactly this drawing of the boundary.

- **Lakatos, *Methodology of Scientific Research Programmes* (1978), pp. 47–49** — hard core / protective belt distinction. Research programmes have a methodologically untestable hard core and a protective belt of auxiliary hypotheses through which empirical tests are routed. Lakatos: "the hard core ... is irrefutable by the methodological decision of its protagonists." Operative point: explicit-postulate enumeration (PIFB:3141 four-postulate list with closing "None of (i)–(iv) is derived from variational free-energy minimization in the present formulation") is the canonical Lakatosian move — protective-belt identification without disguising it as hard-core content.

- **Friedman, *Dynamics of Reason* (2001), Ch. III §3** — relativized constitutive a priori. A physical theory comes equipped with a stratum of conceptual postulates that are not themselves empirically tested but that *constitute* the framework within which empirical tests acquire meaning. Operative point: the 3094–3102 "metaphysical postulate, not a derived result" and "framework may ultimately function as philosophical interpretation rather than scientific theory" admissions are canonically appropriate under Friedman's relativized-constitutive register; the 3056 "information ... is the dimensionally fundamental quantity" commitment reads as a framework-constitutive choice in this sense, which is canonical practice.

### Variational inference / FEP

- **[ParrPezzuloFriston2022 *Active Inference* Ch. 14 + closing "Open Questions"]** — canonical Outlook-mode extensions in the FEP textbook literature; speculative chapters on consciousness, embodiment, social inference, evolution are explicitly flagged as research-programme commitments rather than as derived consequences of the FEP itself. Operative point: Outlook-mode chapters of the kind under debate are canonical practice in the active-inference literature itself.

- **[Friston 2010 *Nature Reviews Neuroscience* 11, 127–138, Box 4]** — the original FEP paper itself includes substantial interpretive material flagged as such (origin of life, evolutionary selection). The Outlook register is canonical to the foundational FEP literature.

- **[BleiKucukelbirJordan 2017 *J. Amer. Statist. Assoc.* 112, 859–877 §5]** — variational inference frameworks come equipped with interpretive choices (variational family, KL direction, mean-field factorization) that are not themselves empirically tested but constitute the framework within which empirical tests acquire meaning. Operative point: structural-existence claims (a variational family exists; an approximation is well-defined) and empirical-fit claims (the approximation has small KL on a specific dataset) are conceptually separate; the former does not inherit empirical weight from the latter. Grounds the placeholder-isolation reading (sub-claim 8) in canonical variational-inference practice.

### Information geometry

- **[AmariNagaoka 2000 §2.1, §7.2, §7.4]** — Fisher information matrix in score-function form; block-Fisher form for `K`-dimensional Gaussian family (`g_B(δq, δq) = δμ^⊤ Σ⁻¹ δμ + ½ tr(Σ⁻¹ δΣ Σ⁻¹ δΣ)`); natural-gradient direction on the SPD covariance sector (`d Σ/dt = -2 η_Σ Σ (∇_Σ F) Σ`, the `2Σ · Σ` sandwich that preserves positive-definiteness automatically). PIFB:2670–2676 and PIFB:2623–2627 match these canonical forms.

- **[Amari 1998 *Neural Computation* 10, 251–276]** — natural gradient on a Riemannian manifold; the gauge-frame Lie-group natural gradient at PIFB:2625 with right-trivialized retraction is canonical [Amari 1998 §2 + extension to Lie groups via the Killing-form preconditioner of [Nakahara 2003 §11.1]].

- **[Bishop *Pattern Recognition and Machine Learning* (2006) §10.1.1; Skovgaard 1984 *Scand. J. Statist.* 11, 211–223]** — variational free-energy Hessian and posterior Fisher precision coincide for quadratic potentials. The PIFB:3070 within-framework structural identification of `M_eff` with `Σ_p⁻¹` is canonical under this Hessian/Fisher coincidence.

### Differential geometry

- **[Lee 2013 *Introduction to Smooth Manifolds* (2nd ed.) Proposition 11.25]** — pullback of a covariant tensor field by a smooth map is a covariant tensor field on the source manifold. PIFB:2705 invokes this canonically. Already harvested by red as `[Lee2013 Ch. 11]`; here pinned to the specific proposition.

- **[O'Neill 1983 *Semi-Riemannian Geometry* §3.3 "The Index"]** — Sylvester's law of inertia for symmetric bilinear forms; signature is a basis-invariant of a non-degenerate symmetric form. PIFB:2791 and PIFB:2874 invocations are canonically correct.

- **[Hawking-Ellis 1973 *The Large Scale Structure of Space-Time* §4.1]** — the conformal class of a Lorentzian metric is determined by its null cone; two metrics with the same null cone differ by a positive conformal factor. The causal-cone construction at PIFB:2860–2877 inherits this canonical conformal-class ambiguity and registers it at PIFB:2876.

### Gauge theory

- **[Nakahara2003 §10.4 "Curvature"]** — pure-gauge connections have vanishing curvature; `F = dA + ½ [A, A] = 0` when `A = U⁻¹ dU`. PIFB:2722 and PIFB:2783 Yang-Mills escape-hatch denial is the canonical standard result, not a special move of this manuscript.

- **[Henneaux & Teitelboim 1992 *Quantization of Gauge Systems* Ch. 19 "BRST Quantization of Gauge Theories with Open Algebras"]** — canonical treatment of gauge-orbit averaging in non-abelian gauge theory; Faddeev-Popov is locally well-defined, globally ambiguous; the standard remedy is restriction to the first Gribov region. The PIFB:2928 regulator-conditional flag is the canonical hygiene move for an honest treatment.

- **[Wheeler in Misner-Thorne-Wheeler 1973 *Gravitation* §40 "Pre-Geometry, Pre-Pre-Geometry"]** — Wheeler's own treatment of the "it from bit" vision explicitly registers it as a speculative Outlook commitment, not as a derived theorem. The manuscript's invocation of this register at PIFB:2657–2662 is canonical to the Wheeler literature itself.

## Round 3 — Red panel additions (rebuttal)

### Philosophy of science

- **Lakatos 1978 *MSRP* pp. 33–34** [via SEP "Imre Lakatos" §2.2 at `https://plato.stanford.edu/entries/lakatos/`] — diagnostic for theoretical vs. empirical progressiveness: progressive programmes generate excess empirical substance and have novel predictions validated; degenerative programmes either fail to generate new predictions or produce novel forecasts that are systematically falsified. Operative point: PIFB:3084 announces a research program but does not yet generate any novel prediction in the present manuscript; the "burden-of-construction clause" is a promise to discharge the burden later, which is the textbook degenerative-by-promise pattern.

- **Popper 1959 *Logic of Scientific Discovery* §6** [via SEP "Karl Popper" §3 at `https://plato.stanford.edu/entries/popper/`] — "a theory which is compatible with all such observations [...] is unscientific." Operative point: PIFB:3060 admits "any measurement can be reinterpreted as labeling information geometry, making the framework compatible with any result by construction"; this trips Popper's demarcation rule by the manuscript's own internal logic.

- **Friedman 2001 *Dynamics of Reason* Ch. III §3, pp. 71–80** — relativized constitutive a priori requires the constitutive principle to constitute the meaning of empirical tests within a working scientific theory; principles compatible with any result by construction have no empirical tests to constitute. Operative point: blue's invocation of Friedman to legitimize PIFB:3056 as a "framework-constitutive choice" misapplies Friedman's structure, since PIFB:3060 admits the commitment has no empirical content.

### Information geometry (compound on Cencov)

- **AmariNagaoka 2000 §2.3** [also in canon as `external_canon_math.md §1`] — Cencov 1972 uniqueness theorem: the Fisher metric is the unique Riemannian metric on a statistical manifold invariant under sufficient statistics, *up to a positive scalar multiple*. Operative point: Cencov is a scale-removing theorem, not a scale-fixing one. The PIFB:3084 reference to "Cencov scale fixings" names a primitive the cited theorem cannot deliver; the falsifiability criterion conditions on a fictitious foundational element.

- **Amari 1998 *Neural Computation* 10, 251–276 §2** — natural gradient on Riemannian manifolds normalized by the operational invariance principle, not by external scale-fixing. Confirms no mechanism for fixing physical constants is introduced in the canonical information-geometric literature.

### Variational inference (compound on placeholder-isolation)

- **ParrPezzuloFriston 2022 *Active Inference* §3.3, §3.4** — variational-inference treatments require local registration of approximation status; the canonical practice is paragraph-level hedge attachment, not section-level cross-reference. Operative point: PIFB:2807 forward-references a placeholder-flagged §Results section without a within-paragraph or within-subsection hedge; this trips blue's own Falsification Condition 1 by direct reading of the manuscript text.

- **BleiKucukelbirJordan 2017 §5** — structural-vs-empirical content separation requires the structural content to be locally bound to its empirical-status registration; the canonical practice is paragraph-level binding. Confirms that PIFB:3070 delivers this binding correctly (the placeholder admission is within the same paragraph as the identification claim) but PIFB:2807 does not.

### Differential geometry (3056 ontological-lift attack)

- **Lee 2013 *Introduction to Smooth Manifolds* (2nd ed.) Prop. 11.25** — pullback of a covariant tensor by a smooth map is a covariant tensor on the source manifold. Confirms PIFB:2705 mathematics is correct. The structural failure is not in the math at 2705; it is in the rhetorical lift at PIFB:3056 (no same-paragraph hedge on the ontological commitment).

- **Hawking-Ellis 1973 *The Large Scale Structure of Space-Time* §4.1, Prop. 4.1.1** — confirms conformal class is determined by null cone; PIFB:2876–2877 invokes this canonically. The structural failure does not lie at 2876.

### Gauge theory (cross-pillar consistency attack)

- **Singer 1978 *Comm. Math. Phys.* 60, 7** ("Some remarks on the Gribov ambiguity") — no globally well-defined non-abelian gauge-fixing condition exists. Confirms PIFB:2928 hygiene is canonical, but the regulator-conditional registration does not propagate to PIFB:3056 in the manuscript text; the cross-pillar dependency runs from ontology to regulator to "the regulator does not yet exist."

- **Skovgaard 1984 *Scand. J. Statist.* 11, 211–223; Bishop 2006 *PRML* §10.1.1** — variational free-energy Hessian and posterior Fisher precision coincide for quadratic potentials; PIFB:3070 invokes this canonically. The "agree by construction in the limit" admission is the manuscript's own concession that the mass-from-Fisher identification is definitional rather than empirical — supporting the placeholder-isolation attack at sub-claim 8 by structural concession in the text.

## Round 3 — Blue panel additions (rebuttal)

### Philosophy of science (rebuttal)

- **Popper 1963 *Conjectures and Refutations* Ch. 1 §I.** Positive characterization of bold conjectures; the heuristic-content / empirical-content distinction. A manuscript that honestly draws the boundary between heuristic content (not presently testable) and empirical content (presently testable) is in canonical compliance with the Popperian demarcation rule even when heuristic content is large. The demarcation rule applies to theories; PIFB:2582 Outlook bracket and PIFB:3127–3145 boxed enumeration are precisely this boundary-drawing.

- **Lakatos 1978 *FMSRP* pp. 47–49 (hard core / protective belt).** Hard-core commitments "irrefutable by the methodological decision of [their] protagonists"; empirical tests run through the protective belt of auxiliary hypotheses. The PIFB:3141 four-postulate enumeration with closing "None of (i)–(iv) is derived from variational free-energy minimization in the present formulation" is the canonical Lakatosian protective-belt identification, canonically authorizing the move PIFB makes.

- **Friedman 2001 *Dynamics of Reason* Ch. III §3.** Relativized constitutive a priori. The PIFB:3056 information-as-primal commitment reads as a Friedman-style constitutive choice that constitutes the framework within which empirical content acquires meaning; PIFB:3094–3102 registers the philosophical status appropriately.

### Information geometry (rebuttal)

- **Amari–Nagaoka 2000 §2.3 (Cencov uniqueness up to positive scalar) and §3.2 (α-connection family).** Cencov is scale-removing, not scale-fixing; conceded as a category-error in PIFB:3084 prose wording. The α-connection canonical one-parameter family on statistical manifolds supplies the canonical structure the 3082 dimensionless-ratios programme can canonically invoke, after the 3084 wording is editorially repaired.

- **Lauritzen 1987 *Differential Geometry in Statistical Inference* IMS Lecture Notes 10.** Canonical α-connection and dual-affine treatment in statistical inference; further canonical structure available to the 3082 programme.

- **Bishop 2006 *PRML* §10.1.1; Skovgaard 1984 *Scand. J. Statist.* 11:211.** Variational free-energy Hessian and posterior Fisher precision coincide for quadratic potentials. Canonically grounds the PIFB:3070 within-framework mass-from-Fisher identification as a structural-identification claim (not an empirical-fit claim).

### Gauge theory (rebuttal)

- **Faddeev–Popov 1967 *Phys. Lett. B* 25:29.** Original gauge-orbit-averaging functional integral; labeled, named, downstream-referenced for fifty years despite the [Singer 1978] global ambiguity. Canonical practice of labeling regulator-conditional constructions. PIFB:eq:consensus_metric at 2934 is in this canonical register.

- **Becchi–Rouet–Stora 1976 *Ann. Phys.* 98:287; Tyutin 1975.** BRST symmetry; labeled, named, conditional construction in non-abelian gauge theory.

- **DeWitt 1967 *Phys. Rev.* 160:1113 ("Quantum Theory of Gravity I").** Wheeler-DeWitt equation as labeled, named, downstream-referenced object in canonical quantum gravity, conditional on operator-ordering, the problem of time, and constraint algebra closure.

- **Peskin–Schroeder 1995 *An Introduction to QFT* §16.2; Weinberg 1996 *QTF II* §15.4.** Canonical textbook treatment of Faddeev-Popov as labeled-conditional.

### Variational inference (rebuttal)

- **Blei–Kucukelbir–Jordan 2017 *J. Amer. Statist. Assoc.* 112:859 §5.** Structural-existence and empirical-fit content separation in variational inference.

- **Friston 2010 *Nat. Rev. Neurosci.* 11:127–138 Box 4.** Outlook-mode interpretive content in foundational FEP paper itself.

- **Parr–Pezzulo–Friston 2022 *Active Inference* Ch. 14 + "Open Questions."** Outlook-mode extensions of FEP in textbook practice.

- **Ramstead, Friston, Hipólito 2020 *Synthese* 199:6271.** Variational ecology FEP framing; canonical extension to multi-agent settings.

- **Friston, Parr, de Vries 2017 *Network Neuroscience* 1:381.** Hierarchical FEP with single ancestral generative model.

### Differential geometry (rebuttal)

- **Lee 2013 *Introduction to Smooth Manifolds* (2nd ed.) Prop. 11.25.** Pullback of covariant tensor field by smooth map is canonically well-defined. PIFB:2738 "precise sense" is a structural-mathematical claim about pullback well-definedness, with agent-dependence carried within the same sentence ("agent-dependent because both the section and the connection $A^{(i)}$ are") and the surrounding 2722–2724 paragraphs.

- **O'Neill 1983 *Semi-Riemannian Geometry* §3.3 ("The Index").** Sylvester's law of inertia; signature as basis-invariant.

- **Hawking–Ellis 1973 *Large Scale Structure of Space-Time* §4.1.** Conformal-class determination by null cone.

- **Frankel 2011 *Geometry of Physics* (3rd ed.) Ch. 17.** Associated-bundle horizontal-vertical decomposition.
