# Peer Review — Participatory_it_from_bit.tex — 2026-05-18

## Summary

The manuscript proposes a gauge-theoretic variational free energy framework in which agents are smooth sections of a principal G-bundle carrying Gaussian belief and generative-model fields plus a Lie-algebra-valued gauge frame, develops a mixture-of-sources derivation of softmax attention as the alignment-energy minimiser, validates a single-layer non-MLP transformer on WikiText-103, and then advances a sequence of "speculative extensions" toward Wheeler's "it from bit", Kant's phenomenal/noumenal distinction, Lahav-Neemeh cognitive reference frames, and a Lorentzian-signature mechanism via complexified gauge frames. The strongest contributions are the carefully scoped abstract (which honestly delimits exponent / floor / iso-token caveats on the scaling claim) and the consistent labelling of the indefinite-signature construction, the consensus metric, the pullback-curvature gravity reading, the qualia identification, and the cognitive-shareability thesis as conditional, interpretive, or as worked-example existence statements rather than derivations — together with the explicit pan-agentic ontology committed to up front. The weakest claims are concentrated in the philosophical centerpiece: the manuscript's central thesis ("how 'it' computationally emerges from 'bit'") is in places asserted as a derivation when, by the manuscript's own scoping language elsewhere, only a structural correspondence has been exhibited; several novel mathematical constructions (the pointwise free-energy functional with attention-entropy term, the cross-scale shadow relation, the gauge-orbit-averaged consensus metric, the GL(K,ℂ) frame-twist signature mechanism) carry no internal proof of equivalence to the standard FEP and need to be labelled as novel extensions rather than as instances of "Friston's free energy principle". The manuscript-to-codebase divergence is generally minor and self-disclosed. Style discipline is high. Verdict: major revisions — the math content is mostly correct, but the conceptual framing intermittently overstates what is derived versus what is postulated, and several (S)-tagged invocations of standard FEP / standard transformer attention are actually (N)-tagged novel constructions or (R) reductions with the reduction not displayed.

## Standards against which the manuscript was reviewed

- [Friston2010] — standard form of variational free energy `F = E_q[log q - log p(o,s)] = KL(q‖p(s|o)) - log p(o)`.
- [ParrPezzuloFriston2022], [FristonEtAl2017] — active-inference process theory; hierarchical / nested formulations.
- [BleiKuckelbirgJordan2017], [KingmaWelling2014] — variational inference, ELBO, Gaussian KL closed form.
- [DempsterLairdRubin1977] — EM separation; expectation step does not see targets.
- [AmariNagaoka2000], [Amari1998], [Amari2016] — Fisher-Rao metric, natural gradient, statistical-manifold geometry.
- [Cencov1972] — uniqueness of Fisher metric up to scaling under sufficient-statistic invariance.
- [Nakahara2003], [Frankel2011], [KobayashiNomizu] — principal bundles, associated bundles, connections, holonomy, Maurer-Cartan form, parallel transport.
- [Vaswani2017] — standard scaled dot-product attention `softmax(QKᵀ/√d_k) V`.
- [Bronstein2021] — geometric deep learning, gauge-equivariant networks.
- [Tsai2019, Ramsauer2021, Millidge2021] — alternative interpretations of attention (kernel, Hopfield, predictive coding); none uniquely "is" attention.
- [BaiKolterKoltun2019] — IFT-style gradients through fixed points.
- [Wheeler1990] "Information, Physics, Quantum" — primary source for "it from bit" (verified in the manuscript's own `references.bib` line 469).
- [Wheeler1983] "Law without law" — verified (`references.bib` line 298).
- [Kant1781] — phenomenal/noumenal distinction (verified in bibliography).
- [LahavNeemeh2022, LahavNeemeh2025] — Relativistic Theory of Consciousness, cognitive reference frames; user's representation of their position checked against published abstracts only (full text not retrieved — see citation verification).
- [Kretschmann1917], [Norton1993] — general-covariance / gauge-invariance distinction; engagement appears competent.
- [DonnellyFreidel2016], [BartlettRudolphSpekkens2007] — edge modes / quantum reference frames; cited as analogues for the dual-role-of-φ argument.
- [HoffmannChinchilla2022] — Chinchilla scaling exponent (b ≈ −0.34), used in the scaling comparison.
- [Wilson1971, Cardy1996], [karcher1977riemannian], [fenichel1979geometric], [tishby1999information] — for the RG appendix.

## Major Issues

### M1. The pointwise free energy with attention-entropy `τ β log(β/π)` term is not Friston's free energy; the manuscript should distinguish.

**Claim (manuscript):** §1011–1024 "The variational free energy principle [Friston2010, Parr2022] provides a tractable approximation to intractable Bayesian inference. ... A single agent minimizing F[q] performs standard variational inference. Our framework extends this to multiple interacting agents with different reference frames." This is followed at §1232 by Eq. (eq:pointwise_free_energy), which contains the alignment terms `Σ β_ij KL(q_i ‖ Ω_ij q_j) + τ β_ij log(β_ij/π_ij)` and the model-channel analog with γ.

**Claim kind:** (S)→(N). The manuscript presents Eq. (eq:pointwise_free_energy) and the boxed Eq. (eq:free_energy_functional_final) as the multi-agent extension of Friston's F.

**Standard treatment:** The standard single-agent variational free energy [Friston2010] is `F = E_q[log q − log p(o,s)] = KL(q‖p(s|o)) − log p(o)`, equivalent to `accuracy + complexity = −E_q[log p(o|s)] + KL(q‖p(s))`. There are no inter-agent KL coupling terms, no `Σ_ij β_ij KL(q_i ‖ Ω_ij q_j)` term, and no `τ β log(β/π)` attention-entropy term in the standard FEP. Standard FEP extensions to multi-agent / variational-ecology settings [Ramstead2020, Friston2017Graphical] do not contain the specific gauge-transport-coupled form `Ω_ij = exp(φ_i)exp(−φ_j)` with row-Lagrangian-derived softmax β.

**Problem:** The manuscript cites [Friston2010, Parr2022] as the backbone but then writes a functional that is neither Friston's F nor any other published F. The pointwise functional Eq. (eq:pointwise_free_energy) is a *novel construction*, internally self-consistent (the τβlog(β/π) term is the standard Lagrangian-with-entropy-regularization for the constrained softmax problem, which the manuscript correctly derives), but does not follow from FEP alone. The status note at line 1034 acknowledges this for the inter-agent KL coupling — "the mixture-of-sources construction below is best read as a consensus-energy ansatz rather than as a generative-model derivation: the component distributions ... depend on the variational posteriors $q_j$ of other agents" — and is good. But the framing at §1011 still says the variational free energy "provides" the tractable approximation, conflating the standard single-agent F with the manuscript's multi-agent functional.

**Required revision:** Add a one-sentence disclosure at the start of §1019 ("Multi-Agent Extension") saying: "The multi-agent functional Eq. (eq:pointwise_free_energy) is a novel extension of Friston's single-agent F; the inter-agent coupling terms `Σ_ij β_ij KL(q_i ‖ Ω_ij q_j)`, the attention-entropy term `τ β log(β/π)`, and the model-channel analogue with γ are not present in the standard FEP literature." The existing §1034 status note covers part of this but is buried inside §1029 (Mixture-of-Sources Consensus Energy); promote a single-sentence disclosure to the head of §1019, where the manuscript first claims to "extend" Friston's framework.

### M2. "Standard scaled dot-product attention is recovered as a gauge-fixed and isotropic-Gaussian limit" — the reduction is presented in two routes; the manuscript should explicitly state which is the canonical reduction and which is heuristic.

**Claim (manuscript):** Abstract: "standard scaled dot-product attention is recovered as a gauge-fixed and isotropic-Gaussian limit of the KL-consensus construction, up to a separately introduced learned bilinear compatibility M and the standard normalisation and bias assumptions"; expanded in §1566–1810 with two routes (trivial-frame route §1711+, untied-QK-from-per-token-frames route §1670+).

**Claim kind:** (R) — reduction claim.

**Standard treatment:** Standard scaled dot-product attention [Vaswani2017 §3.2.1] is `softmax(QKᵀ/√d_k) V`, with Q, K, V linear projections of the input. The standard form contains a `1/√d_k` scaling factor.

**Problem:** The manuscript's Eq. (line 1665) recovers `μᵢᵀ M μⱼ` with `M = W_Q W_Kᵀ`. This is essentially an *identification* (writing a bilinear `μᵢᵀ M μⱼ` as a product of two linear maps applied to μ), not a derivation: any invertible bilinear M is a product of W_Q W_Kᵀ in many ways. The manuscript's §1660–1668 acknowledges that the rectangular `W_Q W_Kᵀ` is rank-deficient and routes through a thin-SVD lift, which is good. The remaining issue is that the `1/√d_k` factor of standard attention is not derived here from anything specific to gauge theory: the manuscript uses `τ = κ√K` (line 1242) where the `√K` factor is identified with `√d_k`, but the `κ` is a learnable scalar that does not appear in [Vaswani2017]. So strictly, the gauge-theoretic construction recovers `softmax(μᵢᵀMμⱼ / κ√K)` with a learnable `κ`, not `softmax(QKᵀ/√d_k)` with `d_k` fixed by dimension. The manuscript is currently correct to call this "recovery up to a separately introduced learned bilinear compatibility M and the standard normalisation and bias assumptions"; what it should also say is "up to a learnable temperature κ on top of the dimensional `√K` scaling".

**Required revision:** In Abstract and §1597 ("Connection to Standard Transformers"), add: "The standard `1/√d_k` factor is recovered as the `√K` factor of `τ = κ√K`; the additional learnable `κ` is a free hyperparameter of the gauge-theoretic temperature not present in [Vaswani2017]." Otherwise the abstract reads as if `τ = √K` exactly recovers the standard form, which is slightly stronger than what the construction actually delivers.

### M3. "Standard transformers are degenerate gauge-theoretic systems" / "Standard transformers represent the zero-dimensional, gauge-fixed, single-scale limit of this richer geometric structure" — qualify as one interpretive lens among several.

**Claim (manuscript):** §3247: "The transformer derivation reveals that standard transformers are degenerate gauge-theoretic systems where spatial structure has collapsed to a single point." §3253: "Standard transformers represent the zero-dimensional, gauge-fixed, single-scale limit of this richer geometric structure. They succeed because they capture the essential information-theoretic core..." §109: "Standard transformer attention is recovered as the zero-dimensional limit, providing empirical validation".

**Claim kind:** (I) — interpretive identification claim presented as (R) or even (S).

**Standard treatment:** Multiple equally rigorous interpretations of attention exist in the standard literature: kernel-method view [Tsai2019], modern-Hopfield view [Ramsauer2021], predictive-coding view [Millidge2021], and the gauge-theoretic view the manuscript advances. None of these is uniquely "the" derivation of standard transformers. [Vaswani2017] derives nothing — it specifies an architecture and trains it.

**Problem:** "Are" in "are degenerate gauge-theoretic systems" is a (S)/(R)-flavored verb but the underlying content is (I): the gauge-theoretic framework provides *one* mathematical reading of attention as soft mixture-of-sources inference. The empirical success of standard transformers does not select one interpretation over the others — kernel methods, Hopfield networks, and predictive coding all also "explain" why attention works, under their own assumptions. The manuscript's own related-work paragraph (§154–166) acknowledges that QBism, Rovelli RQM, etc. are different interpretations of relational structure; the same epistemic courtesy is owed to alternative attention interpretations.

**Required revision:** Soften "are degenerate gauge-theoretic systems" (and parallel constructions) to "admit a degenerate-gauge-theoretic reading" or "can be cast as the zero-dimensional limit of the present gauge-theoretic construction". Add a one-sentence acknowledgement in §3245 that kernel-method, Hopfield, and predictive-coding interpretations of attention exist and that the gauge-theoretic reading is complementary rather than exclusive. The Discussion already does this carefully for QBism vs RQM (§3237); apply the same discipline here.

### M4. The cross-scale shadow relation `p_i = Ω_{i,I}[q_I^{(s+1)}]` is presented as a theorem when it is a definitional move that imposes a strong structural constraint not standard in hierarchical FEP.

**Claim (manuscript):** §531–544 introduces `p_i^{(s)}(c) = Ω_{i,I}[q_I^{(s+1)}](c)`, `r_i^{(s)}(c) = Ω̃_{i,I}[s_I^{(s+1)}](c)` (Eq. cross_scale_shadow). §916: "Beliefs and priors share the latent-state manifold ... as theorems following from the cross-scale shadow relation rather than as separate assumptions."

**Claim kind:** (S)→(N). The manuscript labels the matched-bundle property as a "theorem".

**Standard treatment:** Hierarchical variational inference [BleiKuckelbirgJordan2017, Friston2017Graphical, ParrPezzuloFriston2022 Ch. 9] passes the full recognition distribution `q(s_ℓ)` up to inform `q(s_{ℓ+1})`. The standard prior at level ℓ comes from the generative model `p(s_ℓ | s_{ℓ+1})`, not from the meta-level posterior. The user's scheme is a *deterministic point-passing scheme* that fixes the level-ℓ prior to be a transported copy of the level-(ℓ+1) posterior. This is a strong structural assumption.

**Problem:** The cross-scale shadow Eq. (eq:cross_scale_shadow) is a *definitional choice* that fixes the level-ℓ prior. The manuscript calls the consequence ("p and q live on the same statistical manifold") a "theorem rather than a simplifying assumption" — but that theorem only follows from the prior definition Eq. (eq:cross_scale_shadow), which is itself a strong assumption. This is internally consistent but the rhetorical move is misleading: the "theorem" is reading off a definition.

**Required revision:** Re-label §916 from "as theorems following from the cross-scale shadow relation" to "as consequences of the cross-scale shadow definition Eq. (eq:cross_scale_shadow)". Add an explicit remark at §531 that the cross-scale shadow is a structural commitment, not derivable from FEP — and contrast it with the standard hierarchical FEP scheme in which `p(s_ℓ | s_{ℓ+1})` is part of the generative model and the level-ℓ posterior is computed from observations through that generative model. The reduction of one to the other (or the demonstration that one approximates the other) is not currently in the manuscript.

### M5. Gauge-orbit-averaged consensus metric — the regulator gap is well-flagged, but the construction is invoked downstream as if finite.

**Claim (manuscript):** Eq. (eq:consensus_metric) §2885; §2880 acknowledges that "an honest gauge-orbit average would have to integrate over all maps g: C → G, not over a single copy of G. That is an infinite-dimensional functional integral over a space of gauge-group-valued fields and requires a gauge fixing or a regulator to be well-defined." Then §3083: "A candidate gauge-invariant collective metric can be defined through gauge averaging (Section consensus_metric), conditional on a regulator that the present manuscript does not construct"; §3166: "the structural pullback metrics ... converge $G^{(s)}_i \to G^{(s)}_j$ at the slow timescale ... and its identification with the consensus metric $\bar{G}_{\mu\nu}^{\text{consensus}}$ of Section consensus_metric is conditional on the regulator caveat stated there."

**Claim kind:** (N) labelled honestly; the issue is downstream invocation.

**Standard treatment:** Gauge-invariant observables in gauge theory require either (a) gauge-fixing with a Faddeev-Popov determinant correction, (b) projection onto gauge-invariant subspaces (Wilson loops, etc.), or (c) functional integration with a chosen regulator that is itself a structural input.

**Problem:** The manuscript's regulator caveat is correctly stated at §2880 and §2889, and the consensus metric is correctly labelled as a heuristic target. But the central physics-interpretation moves (§3166 within-species pullback agreement, §3175 "what is modelled as objectivity is the gauge-invariant structure that persists at equilibrium when agents minimize their collective free energy", §3171 "shared spacetime") all *operationally depend on having a regulator that makes the gauge-orbit average finite*. The manuscript flags "conditional on the regulator caveat" at the end of these passages, which is good, but a reader following the physical reading will repeatedly construct an argument that depends on a quantity the manuscript does not show is finite. The current flagging is enough for honesty but the cumulative load on a single conditional is heavy.

**Required revision:** Add one sentence to §3083 / §3171 / §3175 saying: "Because the regulator that would make the gauge-orbit average finite has not been constructed, the within-species and intersubjective consensus arguments below are conditional on the existence of a successful regulator — they identify what such structure would look like under one, not finite gauge-invariant observables the present manuscript exhibits." This is a one-sentence repetition of the §2880 disclaimer at each location where it applies; the manuscript's "Pristine Codebase" principle (per CLAUDE.md) translated to manuscripts would benefit from a similar disposition.

### M6. The GL(K,ℂ) Lorentzian-signature mechanism imposes three postulates beyond the gauge structure; the construction is now well-labelled, but the wording at §777 / §3175 should not call this a "Lorentzian-preserving subgroup is available" without immediately recalling that availability is necessary but not sufficient.

**Claim (manuscript):** §2745–2810 ("Worked example: algebraic compatibility of Lorentzian signature with GL(2,ℂ) gauge frames under two postulates"). Three postulates are explicitly listed: (i) imaginary temporal generator (`φ_τ = iψ_τ T`), (ii) non-compact generator with `tr(T²) > 0`, (iii) real-part projection of the resulting complex bilinear form. §2731 also adds (iv) the implicit choice of `+tr(AB)` vs `-tr(AB)` as the Killing form sign.

**Claim kind:** (N) — worked example. Honestly labelled at §2767 ("Each step in this pathway is mathematically well-defined; the dynamical content (whether free-energy minimisation actually selects step 2 over real-valued GL(K), and whether step 3 picks out SO(1,3) over other SO(p,q) subgroups) is unresolved.").

**Standard treatment:** Real `GL(K,ℝ)` acting by congruence `Σ → Ω Σ Ωᵀ` on positive-definite `Σ` preserves positive-definiteness by Sylvester's law of inertia [HornJohnson]. Compact `SO(N)` similarly preserves positive-definiteness. Indefinite signature on a Lie-algebra bilinear requires either complexification or restriction to a non-compact real form with indefinite Killing form. [Wheeler1990] does not advance this construction; the standard physics route to Lorentzian signature from Wick rotation continues `τ → iτ` on coordinates, not `φ → iφ` on the Lie algebra.

**Problem:** Minor: the manuscript at §3175 says "lifting it to GL(K) or GL(K,ℂ) is a necessary first step: GL(K,ℂ) contains the Lorentz group SO(1,3) as a subgroup, so a Lorentz-preserving structure is at least available. It is not by itself sufficient: the linearised 2D worked example of Section worked_signature obtains an indefinite bilinear form only after additionally imposing an imaginary frame component along a designated temporal direction and discarding the resulting complex off-diagonal terms by a real-part projection." This is well-worded. Earlier at §777 in Sec. cognitive_reference_frames, the GL(K) gauge group is introduced with a "naturally larger choice" framing, and the reader who arrives at §3175 without having seen the postulates of §2767 might still take "Lorentz-preserving subgroup is available" too strongly. The signature section §2725 already opens with a strong disclosure paragraph that handles this. Recommendation: at the first appearance of `GL(K,ℂ)` in the body (line 124 in Epistemic Status), insert a forward-reference: "(necessary but not sufficient for Lorentzian signature; see §sec:signature_resolution for the additional postulates)".

**Required revision:** Add the one-clause forward reference at §124. Otherwise the construction is well-scoped and the disclaimers are exemplary.

### M7. "Natural gradient descent on beliefs follows ..." — should not be called natural gradient without making clear what is being preconditioned.

**Claim (manuscript):** §1517: "Natural gradient descent on beliefs follows `∂q_i/∂τ = -η_q ~∇_{q_i} F_fast` where `~∇_{q_i} = g_B^{-1} ∇_{q_i}` is the natural gradient with respect to the Fisher-Rao metric g_B on B_state. For Gaussian beliefs q_i = N(μ_i, Σ_i), the Fisher metric in the natural parameters is block-diagonal with `g_μμ = Σ_i^{-1}` on the mean block and `g_ΣΣ[V,W] = (1/2)tr(Σ_i^{-1} V Σ_i^{-1} W)` on the covariance block." Cited to [Amari2016].

**Claim kind:** (S).

**Standard treatment:** Natural gradient [Amari1998] preconditions Euclidean gradients by the inverse Fisher information matrix. For Gaussians parameterised by `(μ, Σ)` (mean-covariance / moment parameterisation) the Fisher matrix is block-diagonal in `μ, Σ` with `F_μμ = Σ^{-1}` and a specific Σ-block; for natural-parameter parameterisation `(Σ^{-1}μ, -½Σ^{-1})` the Fisher is the second derivative of the log-partition. The statement in §1517 is consistent with the moment-parameterisation Fisher.

**Problem:** Minor consistency: the manuscript says "Fisher metric in the natural parameters" but then gives `g_μμ = Σ^{-1}`, which is the *moment*-parameterisation Fisher block (μ is a moment parameter, not a natural parameter — the natural parameters of the Gaussian are `θ_1 = Σ^{-1}μ`, `θ_2 = -½Σ^{-1}`). The two parameterisations have different Fisher matrices; calling `Σ^{-1}` the "Fisher in the natural parameters" misnames the parameterisation. The natural-gradient flow itself is correctly written.

**Required revision:** Change "in the natural parameters" to "in the moment parameters" or "in the (μ, Σ) parameterisation" at §1517. The text immediately after is correct as written.

### M8. "The closest historical neighbours are Whitehead's process ontology and a hierarchical structural panpsychism, though the framework does not require panpsychism in the metaphysical sense: phenomenal properties are not ascribed to scale-0 electrons" — internally consistent but in tension with §3454 which acknowledges the open problem of why scale-0 frames carry phenomenal character.

**Claim (manuscript):** §114 "Pan-agentic multi-scale ontology" paragraph: "What is ascribed is gauge-frame structure plus participation in variational dynamics, with the question of how phenomenal properties arise upon aggregation across scales left open as a research item." §3454 ("The central open question"): "[the framework] must say why scale-0 electrons have the same kind of belief-and-model-bearing structure (q, s, φ) as scale-N humans, and why aggregating low-scale agents into high-scale meta-agents preserves rather than eliminates phenomenal properties. ... nothing in the formal construction settles whether the meta-agent's pullback metric G_I inherits anything experiential from the constituent metrics {G_i}, or whether the inheritance fails at certain scale jumps, or whether scale-0 frames carry phenomenal character at all."

**Claim kind:** (I) — interpretive. The internal tension is local-rhetorical, not logical.

**Standard treatment:** Pan-agentic / panpsychist positions in the philosophy-of-mind literature [Goff2017] differ on this exact question. The manuscript correctly identifies its position as not requiring metaphysical panpsychism (no ascription of phenomenology to electrons), only structural panpsychism (every agent carries `(q, s, φ)`).

**Problem:** Minor: a reader could leave §114 thinking the framework is non-committal on whether scale-0 agents have phenomenal properties (correct), and then arrive at §3454 expecting to see the question still open — which it is. The two passages are consistent. The remaining work: §114 could foreshadow that scale-0 agents *carry the structural ingredients of* the phenomenological identification but the question of whether that identification holds at scale 0 is open. Currently §114 just says "is left open as a research item" which is the right phrase; the load is on the cross-reference to §3454 reading carefully.

**Required revision:** None — the two passages are consistent. Optionally tighten §114 to forward-reference §3454 explicitly.

### M9. The "no external observations" / environmental-agents formal equivalence is mean-gradient only; the manuscript discloses this but should also disclose that the cross-entropy substitution is non-standard in active inference.

**Claim (manuscript):** §1390–1444 ("Eliminating External Observations"). The construction is shown to give *mean-gradient* equivalence but to differ from the standard FEP observation likelihood by `-½ Σ_i^{-1}` in the covariance gradient. To restore full equivalence the manuscript proposes either (a) replacing the environmental-agent KL by a cross-entropy `-E_q[log q_e]`, or (b) restricting to fixed-covariance dynamics.

**Claim kind:** (R) for the mean-gradient match; (N) for the cross-entropy substitution.

**Standard treatment:** Standard active inference [Friston2010, ParrPezzuloFriston2022 §2] treats observations as a likelihood `p(o|s)` contributing `-E_q[log p(o|s)]` to F. Cross-entropy `-E_q[log q_e]` is *not* the standard active-inference observation term — it is the negative log-likelihood under a Gaussian sensor density, which agrees with the active-inference term only when `q_e` is the literal Gaussian sensor.

**Problem:** The manuscript at §1437–1441 correctly observes that the cross-entropy substitution recovers the negative log-likelihood, but does not flag that this requires `q_e` to be interpreted as the sensor density rather than as the "belief of an environmental agent". The substitution is internally consistent but the philosophical move (treating sensor likelihoods as agent beliefs) is non-standard. A reader could conclude that observation-likelihood and environmental-agent formulations are equivalent in full generality, which they are not.

**Required revision:** At §1442, add: "The cross-entropy substitution `-E_q[log q_e]` recovers the standard active-inference observation term only under the interpretation of `q_e` as the sensor likelihood density, not as a belief distribution being optimised by an environmental agent. The 'fully internal' framing of environmental agents is therefore consistent with standard active-inference at the level of mean dynamics but interprets the sensor density as a frozen environmental-agent belief rather than as a likelihood." Otherwise the structural-equivalence summary table at §1462 (which calls the framework one of multiple variational stationarity principles) reads as if both formulations are interchangeable.

### M10. The Fisher-Rao Gaussian-manifold sectional-curvature claim at §895 is now correct but the older "constant negative curvature" reading is referenced elsewhere — verify consistency.

**Claim (manuscript):** §895: "The manifold of Gaussian distributions N(μ,Σ) under the Fisher-Rao metric has nonpositive sectional curvature: the SPD covariance sector S^+_K is a Riemannian symmetric space of nonpositive curvature, while certain low-dimensional Gaussian families -- the univariate case and the location-scale family with fixed correlation, for example -- reduce to hyperbolic geometry of constant negative curvature. The general multivariate Gaussian family does not have constant sectional curvature across all tangent planes; the qualitative consequence -- that uncertainty grows along geodesics and they diverge -- follows from nonpositivity."

**Claim kind:** (S) — corrected.

**Standard treatment:** The univariate Gaussian Fisher-Rao geometry is hyperbolic with constant curvature `-1/2` [AmariNagaoka2000 Ch. 5; Skovgaard 1984]. The multivariate Gaussian Fisher-Rao geometry is *not* of constant curvature in general — the SPD cone with the affine-invariant metric is a symmetric space of nonpositive sectional curvature but with varying eigenvalues.

**Problem:** §895 is now correct. Cross-check: §924 (Gaussian Probability Manifold simplification subsection) says "Under the Fisher-Rao metric, the Gaussian manifold has nonpositive sectional curvature [Amari2016] (the SPD covariance factor is a symmetric space of nonpositive curvature; specific low-dimensional Gaussian families reduce to hyperbolic geometry but the general case does not have constant sectional curvature across all tangent planes)". Both passages are consistent.

**Required revision:** None — verify no lingering "constant negative curvature" reading from earlier drafts survives elsewhere (a search of the manuscript for "constant negative" returned no hits).

### M11. The kinetic-metric postulate (Section velocity_quadratic) is the load-bearing assumption for the "mass = precision" reading; the manuscript discloses this at §1847 and §2027 but the gravity-reading section (§3146-3171) still uses the consequence as if derived.

**Claim (manuscript):** §1845: "The construction is best read as a precision-induced configuration-space metric and a Newtonian-shaped harmonic analogy under that postulate, not as a derivation of physical inertial mass from statistical precision." §2027: "This is a postulate, not a consequence of F." §3146: "We have shown that within the framework the Fisher-Rao precision on the state fiber plays the role of effective mass: M_eff = κ · tr(Σ_p^{-1})." §3164: "Inertial and gravitational mass therefore do not automatically share a single origin in this framework: the equivalence principle would hold only under an additional dynamical link between Σ_{p,i}^{-1} and Σ_{s,i}^{-1} that the present construction does not derive."

**Claim kind:** (N), well-flagged.

**Standard treatment:** Effective mass / inertia in a Lagrangian system is the coefficient of the kinetic term in `L = T − V`. The Hessian of `V` is the stiffness, not the mass. Identifying the two requires a separate kinetic-metric postulate.

**Problem:** §1845 and §2027 are exemplary disclosures. §3146 introduces "Fisher-Rao precision ... plays the role of effective mass" without re-stating the kinetic-metric postulate. A reader who arrives at §3146 from the abstract or from §1846 might think this is now an established intra-framework fact, when it is only the consequence of the postulate. §3164 then conditions the equivalence-principle reading on "an additional dynamical link", which is honest.

**Required revision:** At §3146 first sentence, replace "We have shown that within the framework the Fisher-Rao precision on the state fiber plays the role of effective mass" with "Within the framework, the precision-as-mass reading of §sec:mass — under the kinetic-metric postulate of §sec:velocity_quadratic — gives M_eff = κ · tr(Σ_p^{-1})". This is a one-sentence change that propagates the postulate-not-derivation flag from §1845 / §2027 forward to the gravity-reading section.

### M12. The renormalization-group construction (§4454–4683) is now structurally honest about the pushforward-vs-closure distinction, but the body §2136 still describes "RG-inspired" coarse-graining without forwarding the closure caveat to readers who skip the appendix.

**Claim (manuscript):** §2136: "...the present construction is RG-inspired rather than a literal RG analysis: we do not exhibit a β-function, locate fixed points, or demonstrate scale invariance beyond the parametric form, and the analogy is structural rather than computational." §4459 ("Exact pushforward RG"): "The pushforward step is exact and structural ... The closure step is approximate and conditional: the exact renormalized free energy lies near the multi-agent functional class only under explicit hypotheses on cluster coherence, edge-marginal compatibility, gauge-group regularity, and timescale separation."

**Claim kind:** (N), well-scoped. Theorems in §4565–4682 are correctly stated as theorems for the pushforward step and as a proposition for the closure-residual bound.

**Standard treatment:** Wilsonian RG [Wilson1971, Cardy1996] requires exhibition of a coarse-graining map, a β-function, fixed points, and ideally finite-size-scaling collapse for the universality claim. Pushforward of a Gibbs measure under any measurable map preserves the partition function trivially (Theorem rg_pushforward); this is structural and not the load-bearing RG content.

**Problem:** §4574 ("Discrete semigroup composition") is correct as stated but the semigroup property holds *only* for the exact pushforward. The body §2136 says "scale invariance of the functional form suggests the framework may exhibit critical phenomena, fixed points, or universal behavior" — this is conditional on closure, which the manuscript correctly defers to the appendix. A reader of §2136 who does not read §4459 might come away thinking the RG construction is more delivered than it is. The appendix (Proposition rg_residual) is properly hedged ("stated as a proposition rather than a theorem because the constants ... are not pinned down here").

**Required revision:** At §2136 first paragraph, after "RG-inspired rather than a literal RG analysis", add: "Specifically, the partition-function preservation and semigroup composition properties (Appendix rigorous_rg, Theorems rg_pushforward–rg_semigroup) are exact under any measurable coarse-graining and do not by themselves deliver an RG flow; the load-bearing closure step — whether the exact renormalized free energy lies in the multi-agent functional class — is approximate and depends on the closure-residual bound of Proposition rg_residual under conditions (i)–(v) of §4480." Mention also that finite-size scaling / universality has been deferred to future multi-seed work (which the manuscript already does in the appendix §4679, but a body-side mention strengthens the scoping).

### M13. "From Bit to It to Bit Again" recursive update equation `q_i^(n+1) = T_Ω[Π_g({q_j^(n)})]` (§2467) is a schematic correspondence, not a working update; the manuscript should not label the corresponding fast/slow cycle as the formal correlate of Wheeler's self-excited circuit.

**Claim (manuscript):** §2466: "The complete fast-channel cycle, as a schematic update, is `(beliefs) →^{pullback} (induced epistemic geometry) →^{transport} (updated beliefs)`, or formally `q_i^(n+1) = T_Ω[Π_g({q_j^(n)})]` where Π_g is the pullback of the Fisher metric onto the base manifold and T_Ω is gauge-covariant transport. We do not box this expression: it is a structural sketch of the participatory loop on the fast channel G^(q), not a derived update equation". Then §2480 (Table) lists "It from bit" → "Structural pullback metric induced from generative-model sections"; "Participatory universe" → "Cross-scale meta-agent feedback loops"; "Observer-dependent reality" → "Gauge-frame-dependent pullback metrics"; "Self-excited circuit" → "Sustained non-equilibrium under threshold detection".

**Claim kind:** (I), flagged correctly. The "we do not box this expression" disclaimer is exemplary.

**Standard treatment:** [Wheeler1990] "Information, Physics, Quantum: The Search for Links" is a position paper. It does not contain a mathematical formalism; Wheeler's "self-excited circuit" and "law without law" are evocative figures, not equations. Any "derivation" or "realization" of Wheeler's program is therefore a structural mapping rather than a proof.

**Problem:** The manuscript is already careful at §2470 ("structural sketch of the participatory loop ... not a derived update equation") and at the Table caption §2486 ("We do not claim to have derived mass, spacetime geometry, or quantum mechanics from the framework; this table is a structural correspondence, not a derivation summary"). The remaining issue is that the §2607 section is titled *"It From Bit: The Pullback Construction"* — a title that, taken at face value, claims a derivation. The actual content of §2607–2723 is the pullback mechanism with the regulator caveat propagated forward correctly, but the section title is stronger than the section content.

**Required revision:** Retitle §2607 to "It-from-Bit: An Induced-Geometry Reading via the Pullback Construction" or similar. The section content is fine; only the title overstates.

### M14. Lahav-Neemeh "supplies a transformation law" claim — the recovered Alice/Bob composition is a derived consequence of the manuscript's transport law, but the structural overlap with Lahav-Neemeh's stated proposal needs verification against their actual papers.

**Claim (manuscript):** §163 (Related Work): "Their account asserts that a transformation law between cognitive frames exists but does not write one down; the gauge-frame fields φ_i ∈ g and transport operators Ω_ij defined in (eq:transport_def) and developed in Section cognitive_reference_frames supply such a law, with KL(q_i ‖ Ω_{ij}[q_j]) as the frame-invariant scalar". §3417–3454 (Sec. Gauge-Theoretic Realisation of the Cognitive-Frame Transformation Law): extended treatment, with §3438 noting "We do not claim our gauge frames *are* Lahav and Neemeh's cognitive frames in any stronger sense than that the two notions play the same conceptual role."

**Claim kind:** (I) — structural-function correspondence.

**Standard treatment:** Lahav-Neemeh's two papers ([LahavNeemeh2022], [LahavNeemeh2025]) are not part of the standard FEP / gauge-theory canon; they are recent philosophy-of-mind work I have not personally verified.

**Problem:** Without retrieving the actual Lahav-Neemeh papers, I cannot verify the central claim that they "assert that a transformation law between cognitive frames exists but does not write one down". The manuscript's framing of their position is internally consistent (and §3438 honestly acknowledges that the match is functional rather than identity-level), but a reviewer of this manuscript would want to check the primary source. The manuscript's Alice/Bob construction at Eq. (eq:alice_bob_composition) is a valid composition of transports within the present framework, but its identification as "recovering" Lahav-Neemeh's intended isomorphism depends on what their isomorphism actually is.

**Required revision:** None on the math (the gauge-theoretic realization is correct as a composition of transports within the framework). Recommendation for the reviewer / editor: independently verify the Lahav-Neemeh representation against the cited papers before accepting the "supplies a transformation law" claim. Mark this as a citation to verify — see Citation Verification below.

### M15. The qualia-as-gauge-frame-dependent-phenomenology subsection §3377–3402 is honest about its (I) status but the load-bearing identification "G_i = σ_i* g_B is the formal correlate of phenomenal character" is presented in a sentence that elides the identification.

**Claim (manuscript):** §3382: "The framework suggests a formal and mathematical approach to qualia; the subjective, qualitative aspects of conscious experience that constitute 'what it's like' to see red, taste coffee, or feel pain. Different gauge frames φ_i induce different metrics G_i = σ_i* g_B through pullback from the same noumenal substrate C. On the interpretive reading explored here, the formal correlate of phenomenal character is the geometric structure of this induced metric."

**Claim kind:** (I), labelled clearly.

**Standard treatment:** "Phenomenal character" is a contested philosophy-of-mind concept; no mathematical structure has been shown to be its "formal correlate" in any uncontroversial sense.

**Problem:** §3384 is good ("We do not assert identity between the pullback metric and the phenomenology itself; the framework supplies a structural correlate of phenomenal variation, not a derivation of phenomenal content"). The remaining issue is the framing of "Qualia indeterminacy: a feature on relational readings, an open problem here" (§3388) — the discussion is honest and the "open research question" framing is good. But the central interpretive move ("phenomenal character = geometric structure of the induced metric") is repeated several times under different sub-headings (§3382, §3397, §3401) without re-flagging the (I) status at each. The "information-geometric structuralism" framing at §3400 helps.

**Required revision:** None. The section's repeated structuralism framing is internally consistent and the §3367 scope-of-section disclaimer is good. The qualia discussion is appropriately conditional throughout.

## Minor Issues

- m1. §163, line 165 — "Lahav and Neemeh cite predictive-processing and active-inference work approvingly without committing to a gauge-theoretic formalisation, and the present framework can be read as one such formalisation." This sentence is correct but understates the originality of the move; consider strengthening to "and the present framework offers one such formalisation" or similar (not load-bearing).

- m2. §895 "the SPD covariance sector S^+_K is a Riemannian symmetric space of nonpositive curvature": correct. Cross-check: the SPD manifold under the affine-invariant metric is a Hadamard manifold (simply connected, nonpositive sectional curvature). [BhatiaJainLim2019] is the standard reference and would strengthen the citation here in addition to [Amari2016].

- m3. §1517 "g_ΣΣ[V, W] = (1/2) tr(Σ_i^{-1} V Σ_i^{-1} W)" — this is the *moment-parameter* Fisher metric on the Σ block, in agreement with [AmariNagaoka2000] Theorem 7.1 and standard treatments. Notation: the `[V, W]` for the bilinear-form evaluation is non-standard; consider writing as `g_ΣΣ(V, W)` with parentheses, or as a Riesz-representation expression `g_ΣΣ : T_Σ S^+_K × T_Σ S^+_K → ℝ`.

- m4. §1234 "τ Σ β log(β/π)" form differs by a factor of τ from §1066 "β log(β) − β log(π)" — the manuscript correctly says at §1108 that this is achieved by rescaling `E → E/τ`. The reader is not always reminded that the τ in §1234 corresponds to the κ in `τ = κ √K` of the working implementation; a brief recall of `τ = κ √K` at §1234 would help.

- m5. §2099 "first-order Baker--Campbell--Hausdorff approximation": correct usage; for completeness, "exact for abelian gauge groups or for commuting φ_i" should also note that the approximation has error O(‖[φ_i, φ_j]‖) at second order, not just O(‖φ_i‖²) (which understates).

- m6. §3144 "Where the present framework reintroduces the second-order structure is in its reading of the variational principle as a candidate fundamental scaffolding": "fundamental scaffolding" is awkward; consider "candidate substrate" or "candidate first-principles scaffolding".

- m7. §2152 "Recently this scale invariance of the functional form suggests the framework may exhibit critical phenomena, fixed points, or universal behavior analogous to phase transitions in statistical mechanics [Wilson1974]. The investigation of RG fixed points and critical exponents in this information-theoretic setting constitutes an important direction for future work we are currently engaged in." — the "we are currently engaged in" phrasing is forward-looking and slightly weakens the manuscript's claim discipline; consider "constitutes an important direction for future work" without the second clause.

- m8. §2419 "Near the reorganisation point (step 150) in this single-seed run, the energy variance grows sharply and a power-law fit ΔE² ∝ |t − t_c|^{−α} to the rising portion gives α ≈ 1.8" — good honest reporting. The `NE ≈ 0.63` value at §2335 is the composite-score value (the diagnostic that crosses the detection threshold), not an exponent, so there is no contradiction with the §2419 α ≈ 1.8 exponent. No action needed.

- m9. §2532 "Time becomes discrete, quantized in bits rather than continuous." — overstated relative to §2530's dimensional disclaimer. Consider softening to "If τ_i is identified with subjective time, it is discrete and quantized in bits..."

- m10. §3270 ("The Gauge Curvature Conjecture"): the Regime-II conditional opening paragraph is exemplary. The §3272 "We propose a potentially falsifiable conjecture: in the Regime~II extension, language is a gauge theory" — the verb "is" here is again a (S)-flavored verb for an (I) claim. Consider "language admits a Regime~II gauge-theoretic reading" or "...is amenable to a Regime~II gauge-theoretic reading".

## Math Reviewer Items

### MR-1. α_i disambiguation (continued from commit `89e7982d` "pass 8 - alpha_i disambiguation (math reviewer MR-1 partial)").

The notation paragraph §189 introduces `α_i` (per-agent variational precision parameter, weights `α_i KL(q_i‖p_i)`) and `w_i^I` (cluster-aggregation weight) as disjoint symbols. §1287 and §1303 use α_i consistently in the state-dependent precision section. §1313 "λ_{p,i} ≡ α_i/σ_p^2" introduces a derived weight-decay coefficient. §2067 (variational principle for meta-agent formation) uses `w_i^I` for cluster weights with `Σ w_i^I = 1` and explicitly says "this is a distinct object from the per-agent precision parameter α_i of Section state_dependent_precision". §2316 (Phase I component decomposition) writes E_self = Σ_i ∫ α_i KL(q_i‖p_i) dx. **Verified:** α_i and w_i^I are now disambiguated consistently in all body sections I traced.

**Remaining item from MR-1:** at §2867 ("Consensus Metrics") the weights `w_i(c)` appear without disambiguation against either α_i or w_i^I. The context (consensus metric averaging) suggests these are presence-or-coherence weights, not variational precision. Cross-reference: §2884 (Eq. consensus_metric) and §3084 ("conditional on a regulator") use `w_i(c)`. Add a one-clause definition at §2867: "where w_i(c) are presence-or-coherence weights (the cluster-aggregation w_i^I of §sec:meta_agent_variational specialised to a single cluster, distinct from the per-agent variational precision α_i of §sec:state_dependent_precision)."

### MR-2. Eq. (eq:gaussian_kl) at §525 — Gaussian KL closed form.

Form: `KL(N(μ_1, Σ_1) ‖ N(μ_2, Σ_2)) = (1/2)[log(|Σ_2|/|Σ_1|) + tr(Σ_2^{-1} Σ_1) + (μ_2 − μ_1)ᵀ Σ_2^{-1} (μ_2 − μ_1) − K]`. **Verified** against [KingmaWelling2014 Appendix B] / [BleiKuckelbirgJordan2017] standard form. The dimension constant `−K` is correctly included. No issues.

### MR-3. Eq. (eq:precision_transport) at §1859 — GL(d) precision transport.

`Λ̃_{q_k} := (Ω_{ik} Σ_k Ω_{ik}^T)^{-1} = Ω_{ik}^{-T} Λ_{q_k} Ω_{ik}^{-1}`. This is correct for invertible Ω, by the inverse-of-product identity `(ABC)^{-1} = C^{-1} B^{-1} A^{-1}` with `Σ_k = Λ_{q_k}^{-1}`. The disclosed special-case identity at §1862 — that under O(d) where Ω^{-T} = Ω, precision transport coincides with covariance transport — is correct. **Verified.**

### MR-4. Eq. (eq:mass_mu_diagonal) at §1938 — diagonal mass-matrix block. (Off-diagonal verified algebraically; covariance off-diagonal and cross blocks NOT independently verified.)

`[M_μμ]_ii = Λ̄_{p_i} + Σ_k β_{ik} Λ̃_{q_k} + Σ_j β_{ji} Λ_{q_i} + Λ_{o_i}`. The "sender contribution from `D_KL(q_j ‖ q̃_i)` giving `Ω_{ji}^T Λ̃_{q_i}^{(j)} Ω_{ji} = Λ_{q_i}` by GL-cancellation" is correct via `Ω_{ji}^T Ω_{ji}^{-T} Λ_{q_i} Ω_{ji}^{-1} Ω_{ji} = Λ_{q_i}` (the `Ω^T Ω^{-T} = I` and `Ω^{-1} Ω = I` algebraic cancellations). **Diagonal block verified.** The §1862 caveat that the partial cancellations `Λ̃_k Ω_{ik} = Ω_{ik} Λ_k` hold only under O(d) is also correct.

**Off-diagonal mean block** Eq. (eq:mass_mu_offdiagonal) at §1952: `[M_μμ]_ik = -β_{ik} Ω_{ik}^{-T} Λ_{q_k} - β_{ki} Λ_{q_i} Ω_{ki}^{-1}` for i ≠ k. Algebraic verification: from `∂² KL(q_i ‖ q̃_k)/∂μ_i ∂μ_k^T`, with the gradient `Λ̃_{q_k}(μ_i - μ̃_k) = Λ̃_{q_k}(μ_i - Ω_{ik}μ_k)`, the cross-derivative with respect to μ_k is `-Λ̃_{q_k} Ω_{ik} = -Ω_{ik}^{-T} Λ_{q_k} Ω_{ik}^{-1} Ω_{ik} = -Ω_{ik}^{-T} Λ_{q_k}`. **Off-diagonal mean block verified algebraically.**

**Covariance off-diagonal block** Eq. (eq:mass_sigma_offdiagonal) at §1985: not independently verified — the two-term symmetric-from-i-and-k structure with the `Λ̃ Σ Λ̃ ⊗ Λ̃ + Λ̃ ⊗ Λ̃ Σ Λ̃` Kronecker form involves a Kronecker-product chain rule through `Λ̃ = (Ω Σ Ω^T)^{-1}` that is error-prone; a sympy spot-check is recommended.

**Cross mean-covariance block** Eq. (eq:cross_block) at §2007: the result `[C^{μΣ}]_ik = 0` at consensus follows from the rank-one structure `-Ω^{-T} Λ V Λ Ω^{-1} (μ_i - μ̃_k)` vanishing when `μ_i = μ̃_k`. The vanishing-at-consensus claim is verified; the explicit form away from consensus was not checked symbolically.

### MR-5. The Karcher / variational barycenter at §2080–2092.

For Gaussian children with transported moments `μ̃_i = Ω_{Ii} μ_i`, `Σ̃_i = Ω_{Ii} Σ_i Ω_{Ii}^T`, the forward-KL barycenter is
`μ_I* = Σ_i w_i^I μ̃_i / W_I`,
`Σ_I* = Σ_i w_i^I [Σ̃_i + (μ_I* − μ̃_i)(μ_I* − μ̃_i)^T] / W_I` (with W_I = Σ w_i^I).
The covariance formula is the moment-matched mixture covariance (E[XX^T] − E[X]E[X]^T = Σ_i w_i Σ̃_i + Σ_i w_i (μ̃_i − μ_I)(μ̃_i − μ_I)^T after the cross terms cancel). **Verified.** The drop of the dispersion term at §2120–2128 in implementation is correctly labelled as an O(ε) high-coherence approximation.

### MR-6. Karcher frame on non-compact GL^+(K) (§2099, §2132, §4548).

The §2099 disclosure that no bi-invariant Riemannian metric exists on GL^+(K) (the Killing form on gl(K) is indefinite, as the bi-invariant existence-and-uniqueness theorem requires positive-definiteness of the Killing form) is correct [Helgason 1978 / KobayashiNomizu Vol II]. The §4540 statement that the rigorous theorems are proved on compact G = SO(K) and require a gauge slice or Radon-Nikodym correction for noncompact GL^+(K) is the correct hedge. **Verified.** Recommendation: at §4540 cite Helgason for the bi-invariant nonexistence; the current "as flagged in the body" reference is internal.

### MR-7. The mixture-of-sources free energy reduction at §1066–1100.

`F_align = Σ_j β_{ij} ∫ q_i(k) log[q_i(k) β_{ij} / (P(k|z=j) π_j)] dk` decomposes to `Σ_j β_{ij} [KL(q_i ‖ Ω_{ij} q_j) + log β_{ij} − log π_j]`. **Verified.** The Lagrangian `L = Σ_j β_{ij}(E_{ij} + log β_{ij} − log π_j) − λ(Σ β_{ij} − 1)` with `∂L/∂β_{ik} = E_{ik} + log β_{ik} + 1 − log π_k − λ = 0` gives `β_{ik} = π_k exp(−E_{ik})/Z` with Z = Σ_m π_m exp(−E_{im}). **Verified.** This is the standard maximum-entropy / softmax derivation [Jaynes 1957 / standard ML treatments] applied correctly.

### MR-8. The reduced free energy `−τ log Z_i` at §1336–1346.

Substituting β_{ij}* = π_{ij} exp(−E_{ij}/τ)/Z_i into the full functional F_align = Σ_j β_{ij}(E_{ij} + τ log(β_{ij}/π_{ij})) gives F_align* = −τ log Z_i. **Verified** by direct substitution: `β_{ij}* E_{ij} + τ β_{ij}* log(β_{ij}*/π_{ij}) = β_{ij}* E_{ij} + τ β_{ij}* [−E_{ij}/τ − log Z_i] = −τ β_{ij}* log Z_i`, summed over j with Σ_j β_{ij}* = 1 gives `−τ log Z_i`. The envelope-theorem subsection §1330–1367 correctly handles the autograd-vs-reduced-free-energy distinction.

### MR-9. The §1659 cocycle obstruction to pair-independent transport.

"Under Ω_{ij} = U_i U_j^{-1} the self-pair gives Ω_{ii} = I, and the cocycle Ω_{ij} Ω_{jk} = Ω_{ik} at i = k forces Ω_{ij} Ω_{ji} = I; pair-independence Ω_{ij} = Ω then yields Ω² = Ω, and invertibility forces Ω = I." **Verified** — this is correct and is a clean derivation of why the "constant gauge limit Ω_{ij} = Ω for all i,j" used informally in some standard-attention reductions is structurally inconsistent with the vertex-frame parameterisation. The manuscript correctly uses this to motivate the trivial-frame route (single shared frame U_i = U, so Ω_{ij} = I) and the learned-bilinear-compatibility separation.

### MR-10. Lemma (Vanishing Holonomy) at §1201–1215.

`H_{ijk} = Ω_{ij} Ω_{jk} Ω_{ki} = g_i g_j^{-1} g_j g_k^{-1} g_k g_i^{-1} = I`. **Verified** by direct algebraic cancellation. The §1217 disclosure that this expresses flatness of the Regime~I connection (the Maurer-Cartan identity forces `F_μν^(i) ≡ 0` for `A_μ = U^{-1} ∂_μ U`) is correct.

### MR-11. State-dependent precision closed form at §1303–1313.

`α_i^*(c) = c_0 / (b_0 + KL(q_i ‖ p_i))`. Derivation: stationarity of `α_i KL(q_i‖p_i) + b_0 α_i − c_0 log α_i` with respect to `α_i` gives `KL + b_0 − c_0/α_i = 0`, solved as `α_i^* = c_0/(b_0 + KL)`. **Verified.** The product-rule chain at Eq. (eq:alpha_chain_rule_itfb) and Eq. (eq:alpha_product_rule_itfb) is correct: `∂α^*/∂θ = -(α^*)² / c_0 · ∂KL/∂θ` (chain through α_i^* = c_0/(b_0 + KL)), and applying the product rule to `α^* KL` gives `α^* ∂KL/∂θ − (α^*)² KL/c_0 · ∂KL/∂θ = (α^*)² b_0/c_0 · ∂KL/∂θ` via `α^*/c_0 = 1/(b_0+KL)` and `1 − KL/(b_0+KL) = b_0/(b_0+KL)`. **Verified.**

### MR-12. Pushforward RG theorems at §4565–4622 — exact structural results verified; gauge-covariance and exact-closure theorems NOT independently verified.

**Theorem rg_pushforward** (§4565, partition function and observable preservation): correct. The pushforward of a measure preserves total mass by definition (`ρ_{s+1}(X_{s+1}) = ρ_s(R_s^{-1}(X_{s+1})) = ρ_s(X_s)`) and the observable identity is standard change-of-variables. **Verified.**

**Theorem rg_semigroup** (§4574, discrete semigroup composition): correct. The composition `(R_{s+1})_*(R_s)_* = (R_{s+1} ∘ R_s)_*` follows from the pre-image identity `(R_{s+1} ∘ R_s)^{-1}(A) = R_s^{-1}(R_{s+1}^{-1}(A))`. **Verified.**

**Theorem rg_covariance** (§4588, gauge covariance of R_s under base-local diagonal gauge action): the proof at §4593 invokes bi-invariance of the gauge-group metric (`d_G(hU, hU_i) = d_G(U, U_i)`), pushforward-invariance of forward KL (`KL(h_* a ‖ h_* b) = KL(a ‖ b)`), and equivariance of the Karcher minimization argument. Each step is plausible and is standard for compact G with bi-invariant metric. **Not independently verified at the symbolic level.** The non-compact GL^+(K) case is flagged separately at §4540 as requiring a gauge slice or Radon-Nikodym correction; that flag is correct.

**Theorem rg_exact_closure** (§4607, Gaussian closure with local-potential correction): the §4620 proof reduces to Gaussian integration of `exp(-[Y;ξ]^T A [Y;ξ]/2)` over ξ, yielding the Schur complement `A_eff = A_YY - A_Yξ A_ξξ^{-1} A_ξY` plus `(τ/2) log det A_ξξ`. This is standard Gaussian-Laplace integration. **Verified at the structural level** (the Schur-complement form is standard); the manuscript's hedging that strict Gaussian-KL closure obtains only when `A_ξξ` is independent of Y is correct.

**Proposition rg_residual** (§4646, schematic closure-residual bound): correctly stated as a proposition (not a theorem) because the constants C_1,...,C_6, the closure norm `‖·‖_B`, and the regularity class are not pinned down. The qualitative content (small dispersion + small holonomy spread + compatible weights + large internal gap + weak anharmonicity + approximate Gaussianity imply small closure residual) is plausible; tightness is not established.

**Theorem rg_detector_retention** (§4661): the proof at §4671 is correct algebra. Given `Γ_I > Γ_min` with `Γ_I = P_I C_q C_p ≤ C_q C_p = exp(-V_q/τ_q - V_p/τ_p)`, one has `V_q/τ_q + V_p/τ_p < log(1/Γ_min)`. Substituting into the Lipschitz bound gives `Δ_I > 0` under the stated condition (eq:rg_app_detector_retention). **Verified.**

### MR-13. Conditional Representation Theorem for Forward KL (Appendix H, §4265–4446) — NOT independently verified.

This theorem is load-bearing for the manuscript's invocation at §1108 and §1242 that "within the f-divergence class satisfying assumptions (i)--(iii)..., the forward KL divergence is the unique f-divergence that yields a consistent dual interpretation for the attention weights." The full assumptions (i)--(iii) at §4281, the geometric-mean solution at §4319, the envelope-theorem dual relation at §4348, and the conditional theorem statement at §4359 were not verified within this review's time budget. **Action:** if the framework's KL-direction choice is to be justified as a uniqueness consequence rather than as a free choice, an independent math-reviewer pass on Appendix H is needed; if the appendix-H assumptions are weaker than the body's invocation suggests, M1 and the related forward-KL justification are affected.

## Editorial / Style

- §1262 "Differentiating the row-Lagrangian L_i = ∫ Σ_j[β_{ij} KL(q_i ‖ Ω_{ij} q_j) + τ β_{ij} log(β_{ij}/π̃_{ij})] − λ_i(c)(Σ_j β_{ij} − 1) and solving the row-normalisation constraint Σ_j β_{ij}(c) = 1 yields the closed-form softmax" — long sentence; consider splitting into two for readability.

- §699 and §707 contain `\;\big\|\,` and `\,` LaTeX spacing macros (the only three instances of banned spacing macros in the file). Strip these per project style.

- Banned-phrase scan: no instances of "key insight", "crucially", "critically", "notably", "importantly", "leverages", "underscores", "it's worth noting", "interestingly", "fundamentally", "in particular" were found. (Note: "in particular" appears at line 3164 and 3321 in academic-prose contexts, e.g., "the equivalence principle in particular", which is a legitimate restrictive use rather than a Claude-ism transition. These should still be reviewed and replaced where they function as transitions; at line 3164 it is restrictive ("the equivalence principle, in particular, is now an explicitly conditional claim") and acceptable; at line 3321 "in particular Norton's analysis" introduces a specific reference and is also acceptable.) Style discipline overall is high.

- §2531 "Time becomes discrete, quantized in bits rather than continuous. Different agents experience different time flows, with time dilation arising from different update rates along belief trajectories. There exists no absolute time, only relative temporal ordering through causal information flow. An agent processing information rapidly experiences more subjective time than an agent processing slowly, even if both occupy the same region." — three consecutive sentences using "Time", "Different", "An". Consider varying sentence-opener structure.

- §2466 "complete fast-channel cycle" is hyphenated but §2495 uses "fast channel" unhyphenated. Standardise.

- §3253 "Standard transformers represent the zero-dimensional, gauge-fixed, single-scale limit of this richer geometric structure. They succeed because they capture the essential information-theoretic core, but they omit the geometric content that could enable more sophisticated spatial reasoning, hierarchical abstraction, and emergent geometric structure." — "They succeed because they capture the essential information-theoretic core" is a strong causal claim about why transformers work. Recommended softening: "They are empirically successful at sequence modelling, which on this reading is consistent with their capturing the essential information-theoretic core under the gauge-fixed limit; whether this captures the *reason* for their success is interpretive."

- §3247 "The transformer derivation reveals that standard transformers are degenerate gauge-theoretic systems" — "reveals" is a (S)-flavored verb for an (I) claim (per M3). Replace with "casts standard transformers as a degenerate gauge-theoretic system".

- §3271 "We propose a potentially falsifiable conjecture: in the Regime II extension, language is a gauge theory and linguistic evolution is driven by minimization of gauge field curvature." — "is" should be softened, per M3 / m10. Also "potentially falsifiable" is hedged correctly; consider "we propose a falsifiable conjecture in the Regime~II extension: that linguistic evolution is driven by minimization of gauge field curvature" — strengthens the conjecture's identity as a conjecture and is grammatically smoother.

## Citation Verification

**Scope of verification.** Within the time budget I verified citation *presence* in the manuscript's `references.bib` for the items below; the structural appropriateness of each citation was judged against standard knowledge of the cited work. *Content verification against the actual cited papers* (i.e., that the cited claim is actually in the source's text) was performed only where explicitly noted as `[content-verified]`. WebFetch was not invoked for this review.

- [✓ bib present; standardly used] [Wheeler1990] — `references.bib` line 469, "Information, Physics, Quantum: The Search for Links", in *Complexity, Entropy, and the Physics of Information* (ed. Zurek), Addison-Wesley 1990, pp. 3–28. This is the standard "it from bit" essay. Manuscript uses it for the broad participatory-universe / it-from-bit framing throughout; the manuscript does not attribute any specific equation or theorem to Wheeler1990, only the thematic vocabulary. Appropriate.

- [✓ bib present; standardly used] [Wheeler1983] — `references.bib` line 298, "Law without law", in *Quantum Theory and Measurement* (ed. Wheeler & Zurek), Princeton 1983, pp. 182–213. Used for "no phenomenon is a phenomenon until it is an observed phenomenon" at §65 (standard Wheeler quote) and for the self-excited circuit metaphor at §2045; both are appropriate usages of Wheeler's well-known position statements.

- [✓ bib present; standardly used] [Friston2010] — `references.bib` line 562, "The free-energy principle: a unified brain theory?", *Nature Reviews Neuroscience* 11:127–138. Standard canonical citation for the single-agent variational free-energy principle.

- [✓ bib present; standardly used] [Friston2017] — `references.bib` line 571, "Active inference: a process theory", *Neural Computation* 29:1–49. Standard reference for the active-inference process theory.

- [✓ bib present; standardly used] [Parr2022] — `references.bib` line 580, *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*, MIT Press. Standard textbook reference.

- [✓ bib present] [Kant1781] — `references.bib` line 1584. Used for the phenomenal/noumenal distinction; standard philosophical citation.

- [✓ bib present; engagement appears competent] [Kretschmann1917] / [Norton1993] — `references.bib` lines 1596, 1608. The §3321 engagement with the general-covariance / gauge-invariance distinction reads as a competent representation of the philosophy-of-physics literature on the Kretschmann objection, but I have not verified the specific Norton 1993 claims against the actual paper.

- [✓ bib present; standardly used] [HoffmannChinchilla2022] — `references.bib` line 2708. The §2519 comparison of the manuscript's `b ≈ −1.05` exponent to the Chinchilla `b ≈ −0.34` exponent is framed as a comparison between *different parameter classes and different training regimes*, not as a quality claim. The exponent `b ≈ −0.34` is the well-known Chinchilla cross-entropy scaling exponent. Appropriate.

- [?] [LahavNeemeh2022] / [LahavNeemeh2025] — present in `references.bib` lines 2720, 2731 but I have not retrieved the actual papers to verify the manuscript's representation of their position. The central claim — that they "assert that a transformation law between cognitive frames exists but does not write one down" — needs primary-source verification before the §163 / §3417–3454 development can be accepted at face value. **Reviewer recommendation:** verify against the published papers; the manuscript's §3438 honest acknowledgement that the match is at structural function rather than ontological identity provides a safety net.

- [?] [DonnellyFreidel2016] / [BartlettRudolphSpekkens2007] / [Vanrietvelde2020] — present in bibliography; cited at §558 as analogues for the dual-role-of-φ argument (edge modes in gauge theory; quantum reference frames). The structural parallel claimed is plausible but I have not verified the cited papers' precise statements. Recommendation: light spot-check by the area editor.

- [?] [tishby1999information] / [chechik2005information] — present at §2070, §2077; the information-bottleneck refinement section invokes these for the IB Lagrangian and the Gaussian-IB closed-form solution. Standard references, appropriately cited; specific equation-level verification would require retrieving the papers.

- [✓] [Vaswani2017] — standard transformer paper; cited correctly in the context of "standard scaled dot-product attention" recovery.

- [✓] [Amari2016] — used as the [Amari1998 / AmariNagaoka2000] standard reference; the citations are at the source level and appropriate.

- [✓] [Wilson1971], [Cardy1996] — RG references at §887, §2138, §2154, §4454, §4458. Standard citations, appropriate.

- [✓] [karcher1977riemannian] — at §4551 for the Karcher mean. Standard reference. Used correctly.

- [✓] [fenichel1979geometric] — at §4676 for the normal-hyperbolicity argument. Standard reference, used correctly.

- [✓] [Cencov1982] — at §505 ("The metric is unique up to scaling as the only Riemannian metric on probability spaces invariant under sufficient statistics"). The Čencov uniqueness theorem [Cencov1972 / 1982 English translation] is correctly invoked.

## Manuscript ↔ Code Consistency

I traced the manuscript's load-bearing implementation claims against the project knowledge files in `.claude/agents/vfe-knowledge/codebase_map.md` and `notation_dictionary.md` and the project's `CLAUDE.md`. Full code paths were not opened (codebase audit is out of scope for the manuscript reviewer).

- §1241 "In the working implementation the temperature is factorised as τ = κ√K, with κ a learnable scalar and the √K factor the dimension scaling familiar from scaled dot-product attention" — matches `CLAUDE.md` "Attention" section ("kappa is a learnable hyperparameter; the sqrt(K) factor is intentional dimension scaling on top of kappa"). **Consistent.**

- §778 "Ω_{ij}(c) = exp[φ_i(c)] exp[-φ_j(c)] ∈ G" — matches `CLAUDE.md` "Transport: Omega_ij = exp(phi_i) * exp(-phi_j)" pattern. **Consistent.**

- §784 "Σ_j ↦ R_{ij} Σ_j R_{ij}^⊤" — matches `CLAUDE.md` "Covariance transport: Sigma_transported = Omega @ Sigma @ Omega.T — the sandwich product". **Consistent.**

- §1259 boxed Eq. (eq:free_energy_functional_final) — matches `CLAUDE.md` "Free energy (canonical form, manuscript `\label{eq:free_energy_functional_final}`)" listing: `α * KL(q_i || p_i) + λ_h * KL(s_i || h) + Σ_ij [β KL(q‖Ωq) + τ β log(β/π)] + Σ_ij [γ KL(s‖Ωs) + τ γ log(γ/π)] − E_q[log p(o|x)]`. **Consistent.**

- §1242 "minimising over β subject to Σ_j β_{ij} = 1 recovers the softmax solution of (eq:softmax_attention_general) and substituting back yields the reduced alignment free energy `−τ Σ_i log Z_i`" — matches the CLAUDE.md note that "Manuscript line 1261 explicitly distinguishes the canonical F from the 'entropy-suppressed surrogate'". **Consistent.** The §1278 and §1366 paragraphs explicitly carry the autograd-vs-envelope distinction.

- §1571 "0-dimensional gauge theory" / §1577 "the unreduced GL(K) gauge theory on a single point is validated as a working language model in Section sec:transformers, trained without learned attention projections, MLPs, or pointwise activation functions" — matches `CLAUDE.md` "Hard Constraints: NO NEURAL NETWORKS". **Consistent.**

- §2247 "Throughout the simulations of this section, the slow subsystem (s_i, r_i) is frozen with γ_{ij} = 0" — matches the manuscript_index.md note that the multi-agent simulation is "single-seed, slow subsystem frozen (γ_ij = 0), threshold-based meta-agent formation". **Consistent.** §2243 ("Multi-Agent Simulation Procedure") and §2271 (Table) carry the configuration honestly.

- §2519 "for this architecture the total parameter count N is exactly linear in K across the sweep (mean ratio N/K ≈ 6.53×10^5 to four significant figures across all eleven K values)" — this is a precise numerical claim about the codebase parameter count that would need cross-checking against the actual configs in `publication_outputs/scaling_analysis/`. The configuration check is out of scope here.

- The dimension `K = 13` in the multi-agent simulation (§2275 Table) versus `K ∈ [10, 120]` in the transformer scaling (§120, §149) is correctly distinguished by context and is not a contradiction.

- I did not detect any manuscript ↔ code divergence in the dimensions checked. The most fragile coupling — manuscript Eq. (eq:free_energy_functional_final) ↔ code free energy assembly — is correctly aligned with the project's documented canonical form per CLAUDE.md and would benefit from a direct line-level audit by `vfe-codebase-auditor`.

## Novel-construction inventory

The following constructions in this manuscript are not part of the standard FEP / gauge-theory / transformer literature and require independent justification in the manuscript:

1. **Multi-agent F with `Σ_{ij} β_{ij} KL(q_i ‖ Ω_{ij} q_j)` and attention-entropy `τ β log(β/π)` terms.** Novel; correctly derived as the variational solution of an entropy-regularized mixture-of-sources problem at §1066–1100. Not standard FEP. (See M1.)

2. **Cross-scale shadow relation `p_i^{(s)} = Ω_{i,I}[q_I^{(s+1)}]`, `r_i^{(s)} = Ω̃_{i,I}[s_I^{(s+1)}]`.** Novel structural commitment; not standard hierarchical FEP (which would pass the full posterior, not a transported point). (See M4.)

3. **GL(K,ℂ) frame-twist signature mechanism with imaginary temporal generator and real-part projection.** Novel; honestly labelled as a worked-example existence statement requiring three independent postulates. (See M6.)

4. **Gauge-orbit-averaged consensus metric `Ḡ_μν^{consensus}`.** Novel; the regulator-dependence is correctly flagged. (See M5.)

5. **Pan-agentic structuralism as ontological commitment** (§114). Novel ontological position; honestly distinguished from standard panpsychism, neutral monism, and Lahav-Neemeh's relational physicalism. (See M8.)

6. **Three-tier pullback geometry (`G^{(q)}`, `G^{(p)}`, `G^{(s)}`) with the perceived-spatial-geometry identification at the structural tier `G^{(s)}`.** Novel three-way decomposition consistent with hierarchical predictive processing but not derived from it. (See §2692–2705.)

7. **State-dependent prior precision α_i(c) with log-barrier regulariser.** Novel intra-framework construction; the closed-form optimum α_i* = c_0 / (b_0 + KL(q_i ‖ p_i)) is correctly derived at §1303.

8. **The "no external observations" / environmental-agent equivalence at the mean-gradient level.** Novel; the cross-entropy substitution required for full variational equivalence is non-standard and should be flagged. (See M9.)

9. **The kinetic-metric postulate `M_μμ = Σ_p^{-1}` used as both stiffness and mass.** Novel postulate; honestly labelled at §1845, §2027. (See M11.)

10. **The Regime-II edge-relaxed cocycle `Ω_{ij} = U_i exp(δ_{ij} · G) U_j^{-1}` with data-dependent `δ_{ij}`.** Novel concrete realisation of lattice gauge theory on the agent graph; the Wilson observable / Wilson action constructions are standard once the Regime-II promotion is granted.

11. **The renormalization-group construction with pushforward-then-closure separation (Sec. meta_agent_rg / app:rigorous_rg).** Novel; the pushforward theorems are correct (and trivial); the closure-residual bound is a Proposition rather than a Theorem, which is appropriate scoping.

All novel constructions above are labelled as such or are accompanied by appropriate scoping disclaimers. The principal residual issues are: (M1) presenting the multi-agent F as if it were Friston's F (one-sentence fix at §1019), (M3) the "are" / "is" verbs for interpretive identifications of standard transformers with the gauge-theoretic framework, and (M4) the "theorem" framing of a definitional consequence of the cross-scale shadow.

## Open questions

- The Lahav-Neemeh representation (§163 and §3417–3454) depends on what their published transformation law actually is. I could not retrieve the papers within the time budget. **Action:** verify against [LahavNeemeh2022, LahavNeemeh2025] before final acceptance of the "supplies a transformation law" claim.

- The §3438 disclosure ("We do not claim our gauge frames *are* Lahav and Neemeh's cognitive frames in any stronger sense than that the two notions play the same conceptual role") provides a safety net if the Lahav-Neemeh primary-source verification shows the manuscript has overstated their position. With the safety net in place, even a Lahav-Neemeh-overstatement finding would only require strengthening §163 to match the §3438 hedge, not a structural rewrite.

- The §2247 frozen-slow-subsystem disclosure interacts with the Section 5 ("Speculative Extensions") slow-channel γ_{ij}-coupled phenomena (gravity reading, signature mechanism, consciousness reading). The §2247 disclosure says the simulations report `prior alignment` as the operational proxy for what would otherwise be a model-channel quantity, and §2258 explains that "model coherence C_s as deployed in clustering reduces to a constant under frozen s (we instead detect clusters through the dynamic prior shadow)". This is honest, but it means the empirically validated content of the simulations is entirely on the fast channel, while the speculative-extension content depends in part on the slow channel that has not been exercised. A reader might benefit from an explicit statement in Sec. 5 introduction that "the slow-channel constructions developed here have not been exercised in the simulations of §sec:results, which freeze the slow channel".

- **Appendix theorems and proofs not independently verified.** Three load-bearing appendix results were not verified within this review's time budget:
  - **Conditional Representation Theorem for the Forward KL Divergence via Variational Duality** (Appendix H, §4265–4446). This theorem is load-bearing for the manuscript's claim (§1108, §1242) that "within the f-divergence class satisfying assumptions (i)--(iii)..., the forward KL divergence is the unique f-divergence that yields a consistent dual interpretation for the attention weights." If the theorem is wrong or its assumptions (i)--(iii) are weaker than the manuscript invokes, the "we use forward KL because it is canonically picked out" justification collapses to a choice rather than a uniqueness result. I traced the construction at §4281 (coupled variational problem) and §4319 (forward KL and geometric-mean solution) at a high level only; full symbolic verification of the dual-cost identification at §4435 and the conditional theorem statement at §4359 was not performed. **Action:** a dedicated math-reviewer pass on Appendix H is recommended; the result is the formal justification for the framework's KL-direction choice.
  - **RG appendix Theorems rg_pushforward / rg_semigroup** (§4565–4582). These are correct as standard measure-theoretic identities: pushforward of a measure preserves total mass and composes as a semigroup. **Content-verified at the algebraic level** (the proofs reduce to one-line change-of-variables identities), as noted in MR-12 above.
  - **RG appendix Theorem rg_covariance / Theorem rg_exact_closure** (§4588 / §4607). I did not independently verify the gauge-covariance theorem for the coarse-graining map under base-local diagonal gauge action, nor the Gaussian-closure theorem with the local-potential correction. The proofs sketched at §4593 and §4620 are plausible but I have not symbolically checked the bi-invariance argument or the Laplace-integration log-determinant decomposition. **Action:** spot-check by an independent math reviewer recommended.

- **Off-diagonal mass-matrix blocks not independently verified.** MR-4 verified the diagonal block `[M_μμ]_ii` (Eq. mass_mu_diagonal) and MR-5 verified the barycenter (Eq. meta_agent_mu/sigma_barycenter). The off-diagonal mean block (Eq. mass_mu_offdiagonal at §1952), the covariance off-diagonal block (Eq. mass_sigma_offdiagonal at §1985), and the cross mean-covariance block (Eq. cross_block at §2007) involve tensor expressions with `Ω` vs `Ω^{-T}` and `Ω^T` factors where sign and inversion errors are easy. I checked the §1944–1952 derivation of `[M_μμ]_ik = -β_{ik} Ω_{ik}^{-T} Λ_{q_k} - β_{ki} Λ_{q_i} Ω_{ki}^{-1}` algebraically and it is correct under non-orthogonal Ω. The covariance off-diagonal block (Eq. mass_sigma_offdiagonal) and the cross block were *not* independently verified. **Action:** the off-diagonal covariance block under GL(d) is the most error-prone piece in §1842–2039; a sympy spot-check is recommended.

## Overall Verdict

**Minor revisions, required.** The manuscript's mathematical content is largely correct (subject to the appendix-theorem verification caveats above), the empirical scaling claim is honestly scoped, the simulation results are reported with appropriate single-seed disclaimers, and the philosophical extensions are mostly labelled at the right epistemic register (worked example / structural correspondence / interpretive reading / conditional on regulator). The 4686-line manuscript contains very few banned style patterns and exhibits high discipline in distinguishing what is derived from what is postulated. The required revisions are concentrated in rhetorical positioning rather than mathematical content: (a) M1 — present the multi-agent F honestly as a novel extension of Friston's F (one-paragraph fix at §1019); (b) M3 — soften the "are" / "is" / "reveals" verbs at §1597, §3247, §3271 for what are interpretive identifications between gauge-theoretic constructs and standard machine-learning constructs; (c) M4 — re-label the "theorem" at §916 to "consequence of definition Eq. (eq:cross_scale_shadow)"; (d) M5 — repeat the regulator caveat at §3083 / §3171 / §3175 where the conditional argument lives; (e) M9 — flag that the cross-entropy substitution is non-standard; (f) M11 — re-flag the kinetic-metric postulate at §3146; (g) M12 — forward the pushforward-vs-closure separation from the appendix to the body §2136; (h) M13 — retitle §2607 to a less load-bearing form. None require new mathematics; each can be addressed in a single revision pass.

The pan-agentic ontology (§114), the three-tier pullback geometry (§2692–2705), the consensus-energy mixture-of-sources derivation (§1029–1108), the cross-scale shadow construction (§531–544), the GL(K,ℂ) signature mechanism (§2725–2810), the RG construction (§4454–4683), the kinetic / mass-analogy postulate (§1842–2039), and the Lahav-Neemeh structural correspondence (§3414–3454) are all internally consistent. The honest delimitation of speculative content from empirically validated content is a strength of this manuscript and should be preserved through the revisions above.

**Verdict for the participatory IFB framing specifically:** the IFB framing depends most heavily on M1 (presenting the multi-agent F as Friston's), M3 (the "are" / "reveals" identifications), M5 (the regulator gap in the consensus-metric construction that underwrites the "objective-spacetime-from-consensus" reading), M11 (the precision-as-mass / gravity reading), and M13 (the §2607 section title "It From Bit: The Pullback Construction"). With those revisions applied, the IFB framing is recoverable as **a mathematically scaffolded structural correspondence to Wheeler's vision rather than a derivation of it** — which is what the manuscript's own §2486 ("structural correspondence, not a derivation summary") and §116 ("Level 3: Speculative Physical Interpretation") explicitly claim. The framework genuinely *exhibits* a pullback construction, a cross-scale feedback loop, observer-dependent pullback metrics, and a sustained-non-equilibrium dynamics under threshold detection, all of which are mathematical-vocabulary realisations of Wheelerian themes that have previously remained informal. **Verdict on the IFB framing: minor revisions, required** — the framing is honest enough in many local passages and the residual issues are rhetorical (eliminate "are degenerate gauge-theoretic systems", "reveals", "supplies", "derived from", in favour of "admits a reading as", "is consistent with", "structurally corresponds to"). Reject would be appropriate only if the framing claimed to *derive* Wheeler's participatory universe rather than to provide a mathematical scaffold for it — and the manuscript's epistemic-status section §116 and scope-and-limitations section §128 already explicitly disavow the derivation reading. The remaining work is to propagate that disavowal consistently into the body sections where the IFB language is deployed.

The manuscript is closer to acceptance than the volume of items above suggests; the issues are concentrated in rhetorical positioning rather than mathematical content, and each can be addressed in a single revision pass. The appendix-theorem verification gaps flagged in Open Questions are *recommended* second-reviewer items rather than blocking, since the body construction stands or falls primarily on M1–M13 above; if the Conditional Uniqueness Theorem (Appendix H) is wrong, the manuscript's KL-direction choice becomes a free choice rather than a uniqueness consequence, but the rest of the construction still goes through.
