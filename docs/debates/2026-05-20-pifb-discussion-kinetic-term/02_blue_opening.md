# Blue Opening — pifb-discussion-kinetic-term

## Steelman (opposing position)

The Discussion subsection at line 3095 hedges the wrong axis: it concedes that stiffness-as-inertia is a postulate but does not register the prior "damping = stiffness" identification implicit in mapping the overdamped reduction `mq̈ + γq̇ = -∇V → γq̇ = -∇V` onto the natural-gradient flow `μ̇ = -Σ_p^{-1}∇F`, where Σ_p^{-1} would play the role of inverse damping (mobility), not inertia; the hedge is therefore selective, naming only one identification of a triple, and the calibration is too weak.

## Position

The Discussion subsection at lines 3085-3095 is correctly calibrated. The claim has three load-bearing parts and each is anchored: (i) standard active inference is first-order on cortical timescales (canon-supported), (ii) the framework's second-order structure is a contingent reading aligned with Friston's own action-principle line of work (canon-supported by three primary sources from the FEP literature), and (iii) the hedge at 3095 registers the stiffness-as-inertia postulate in exactly the language the manuscript body uses four times, so the Discussion compresses an already-registered caveat rather than over- or under-claiming. The judge would have to identify a substantive gap not already named in the body for the calibration to fail.

## Evidence

**1. The first-order overdamped status of standard active inference is the canon.** Friston 2010 *Nat. Rev. Neurosci.* 11(2):127-138 writes the variational update as `\dot\mu = -\partial_\mu F`, a first-order gradient flow; this is summarized in the project's own external canon at [external_canon_inference.md §1 / Friston2010]. Friston et al. 2017 *Neural Computation* 29(1):1-49 re-states the dynamics as "gradient descent on variational free energy" (paper abstract and §2). Buckley, Kim, McGregor, Seth 2017 *J. Math. Psych.* 81:55-79 derives the first-order reduction under synaptic time-constant dominance. The manuscript line 3087 ("Standard active-inference treatments employ first-order natural-gradient descent...the overdamped reduction is not an oversight but the appropriate biophysical regime") matches this canon verbatim and the hedge at 3095 ("the biological-timescale omission remains entirely correct for its intended target") explicitly defers to it.

**2. The "fundamental-physics reading" at 3095 is anchored in Friston's own action-principle work, not a manuscript fiction.** Three primary sources from the FEP literature already treat the FEP as a variational principle of least action:

- *Friston 2008 "DEM: a variational treatment of dynamic systems" NeuroImage 41(3):849-885* explicitly formulates free-energy minimisation in "generalized coordinates of motion" (μ, μ̇, μ̈, ...) and writes the optimisation as a "variational action" / "path-integral of free-energy." (Source: published abstract and DEM scheme description, verified via canonical citation entry and web search.)
- *Friston et al. 2022 "Path integrals, particular kinds, and strange things"* (arXiv:2210.12761) abstract: "describes a path integral formulation of the free energy principle" invoking "a method or principle of least action." (Source: arXiv abstract retrieved 2026-05-20.)
- *Friston, Da Costa, Sajid, Heins, Ueltzhöffer, Pavliotis, Parr 2023 "The free energy principle made simpler but not too simple" Physics Reports* (arXiv:2201.06387) decomposes the flow via the Helmholtz decomposition into a solenoidal (conservative, divergence-free) part and a gradient (dissipative, irrotational) part on isocontours of the steady-state density, the standard mechanical-action signature. (Source: paper abstract and Helmholtz-decomposition section, verified via web search 2026-05-20.)

The manuscript's line at 3095 ("its reading of the variational principle as a candidate fundamental scaffolding rather than a phenomenological model of neural inference") is therefore not an idiosyncratic Gauge-Transformer construct but the same reading the principal author of FEP has been advancing since 2008. The hedge correctly identifies it as one reading among several, neither claiming sole legitimacy nor overstating its status.

**3. The stiffness-as-inertia postulate is registered four times in the body, not just once at 3095.** The manuscript layers the caveat:

- `Participatory_it_from_bit.tex:1849`: "Under standard Lagrangian mechanics [arnold1989mathematical] a non-degenerate harmonic identification requires the inertia tensor and the potential Hessian to be operationally independent (0,2)-tensors at the equilibrium configuration; the present construction reuses the same matrix $M_{\mu\mu}$ for both roles and therefore does not supply such an independent test."
- `Participatory_it_from_bit.tex:2013-2015`: "the identification of the Hessian sector $[M_{\mu\mu}]$ with an 'effective mass' is interpretive within the framework rather than a derivation of physical inertial mass."
- `Participatory_it_from_bit.tex:2026` subsection heading: "This is a postulate, not a consequence of $\mathcal{F}$."
- `Participatory_it_from_bit.tex:2029-2030`: "when $k$ and $m$ are both equal to $M_{\mu\mu}$ by construction, $\omega^2$ reduces to a per-direction unit relation and the analogy is structural, not empirical."

The Discussion subsection at 3095 inherits all four by cross-reference to Section sec:mass. Demanding the Discussion duplicate registrations made four times in the body would conflict with editorial parsimony and the project's prose-style mandate (CLAUDE.md §Style: "Remove content that doesn't earn its place through rigorous derivation").

**4. The internal degeneracy of the analogy is self-flagged at line 2029-2030.** The manuscript itself states that $\omega^2 \propto k/m$ reduces to a "per-direction unit relation" because $k$ and $m$ are the same matrix by construction. This is the exact calibration signal a reader needs: the analogy is structural, not predictive. The Discussion's hedge at 3095 ("not a derivation of inertia from the variational principle") is the matching summary phrase.

**5. Anticipating the "triple-identification" attack.** A natural-gradient flow `μ̇ = -G(μ)^{-1}∇F` with Riemannian metric $G$ is, formally, a *gradient flow on a Riemannian manifold* [Amari1998 §2], not a damped Newtonian limit. The metric $G$ supplies the geometry of steepest descent, not a damping coefficient. When the manuscript writes the overdamped reduction at 3089-3091 it invokes the *mechanical analogy*; it is not claiming that $\Sigma_p^{-1}$ plays both the damping and inertia roles of the same Newtonian system. The overdamped reduction is the limit `m/γ → 0` of a Newtonian system; the natural-gradient flow is a separate object (a gradient flow on the statistical manifold). The manuscript's hedge at 3095 registers the stiffness-as-inertia postulate explicitly; the damping role is structurally absent from the natural-gradient flow per [Amari1998 §2-3], so there is no third identification to register. If red argues the overdamped equation maps to natural gradient with $\Sigma_p^{-1}$ in the damping slot, that is a category error: natural-gradient flow has no damping coefficient as a separate object from its metric.

## Falsification conditions

The defense fails if any of the following can be established with primary-source citations:

1. **Action-principle reading is foreign to FEP.** If Friston 2008 DEM, Friston et al. 2022 path-integrals, and Friston et al. 2023 Phys Reports collectively do *not* read the FEP as a variational/action principle with generalized motion and Helmholtz-decomposed flow, then the manuscript's "fundamental-physics reading" at 3095 is idiosyncratic and the hedge cannot anchor in standard literature. Verification status: all three primary sources confirm the action-principle reading via published abstracts and section content (web search 2026-05-20). Falsification path not realised.

2. **The body fails to register the stiffness-as-inertia postulate.** If `sec:mass` (lines 1844-2043) does not name $\Sigma_p^{-1}$ doubling as inertia as a postulate distinct from the Hessian-as-stiffness identification, the Discussion's compressed restatement at 3095 would be load-bearing and the calibration would need to be stronger. Verification status: four explicit registrations at lines 1849, 2013-2015, 2026, 2029-2030. Falsification path not realised.

3. **The Discussion over-claims a derivation.** If the language at 3095 read "we derive the kinetic structure from $\mathcal{F}$" or "inertia follows from the variational principle," the hedge would be falsified. Verification status: the actual language is "not a derivation of inertia from the variational principle, and the biological-timescale omission remains entirely correct for its intended target." Falsification path not realised.

4. **The first-order overdamped reduction is not the canonical active-inference dynamics.** If [Friston2010, FristonEtAl2017, Buckley2017] write the dynamics as second-order or as something other than first-order gradient descent on free energy, the manuscript's framing of standard active inference at 3087 would be wrong. Verification status: all three sources confirm first-order gradient descent on $\mathcal{F}$ as the canonical form. Falsification path not realised.

The defense is contingent: it stands on the body's four registrations and on the three-paper FEP-as-action-principle line. A demonstration that any of those is misread would collapse the defense.

## Concession in advance (Phase 3 candidate)

The claim at line 3093 ("the framework allows agent priors to evolve dynamically. In traditional approaches predictive models are considered fixed") is imprecise. Standard active inference includes parameter learning via the M-step at slow timescales [FristonEtAl2017 §3]. The accurate statement is that within-inference (E-step) timescales standardly hold the prior fixed, and the framework relaxes that. If red attacks this line, blue grants the imprecision and proposes a one-line tightening. This concession is logically independent of the main hedge at 3095 and does not affect the calibration verdict.
