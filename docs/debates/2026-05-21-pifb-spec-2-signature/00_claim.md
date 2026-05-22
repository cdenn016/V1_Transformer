# Claim — pifb-spec-2-signature

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** Attention/Participatory_it_from_bit.tex §2773-2903 (Temporal Structure and the Signature Problem, including Diagnosis, Compact-Group restriction, GL(K) and GL(K,C) Resolution, Postulates Required for an Indefinite Pullback, Worked Example, Alternative Causal-Cone Route, Temporal Direction from Belief Trajectories)
**Canon location:** C:/Users/chris and christine/Desktop/V13_Gauge_Transformer/.claude/agents/vfe-knowledge/

## Claim

The subsection "Temporal Structure and the Signature Problem" (PIFB lines 2773-2903), including the GL(K,C) worked example with imaginary phi_tau and real-part projection, the causal-cone alternative route, and the temporal-direction-from-belief-trajectories subsection, is publication-ready and rock-solid.

## Operational reading (binding for both teams and the judge)

The section openly labels itself as not deriving Lorentzian signature from variational dynamics. The interesting test is:

1. **The math is correct.** The frame-twist quadratic form computation tr(A_tau^2) = -2(d_tau psi_tau)^2 < 0 for T=diag(1,-1) is correct under the stated +tr(AB) convention; the real-part projection yields a well-defined real metric; Sylvester's law of inertia and the SO+(1,1) / SO+(1,3) local frame group claims hold under the construction. The sector split between real Gaussian fiber and complex frame field is internally consistent.
2. **Each postulate is explicitly flagged as a postulate.** (a) Sector split, (b) imaginary phi_tau assignment, (c) real-part projection of the complex bilinear form, (d) +tr(AB) versus -tr(AB) sign convention, (e) single-generator + separable-ansatz simplifications, (f) the 1+3 split being input rather than dynamically selected. These must not silently slip into a derivation.
3. **The causal-cone alternative route is independently rigorous.** The Lorentzian conformal class from postulated finite information speed c_I works algebraically. The tension with first-order natural-gradient dynamics (parabolic infinite-speed continuum limit) is acknowledged in-section. The conformal-factor / dimension-count openness is preserved.
4. **The Wick-rotation analogy is correctly distinguished from standard Wick rotation.** Standard Wick continues tau -> i*tau on the base manifold; the construction here continues phi_tau -> i*phi_tau in the Lie algebra and adds a real-part projection without Wick counterpart. This non-equivalence must be stated explicitly.
5. **Worked example self-consistency.** T = diag(1,-1) is in sl(2,R), NOT in so(1,1). The construction's claim that the local frame group at non-degenerate points is SO+(1,1) must either (a) follow from the worked computation or (b) be flagged as an independent input. Verify whether the SO+(1,1) statement smuggles in the result via the real-part projection or whether it falls out cleanly.

A red strike lands if any postulate is implicit, if the Wick analogy is overstated, or if the algebra is incorrect under the stated conventions. A blue defense requires external canon: Nakahara on principal bundles, Wick rotation literature (Schlingemann or Visser for analytic continuation), Sylvester's law as standard linear algebra.

## User context

One of four parallel debates on the §Speculative Extensions section. This is the most contentious subsection — isolating it preserves the value of the other three debates from being dragged down by the signature problem.
