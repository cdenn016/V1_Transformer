# Evidence Pack — pifb-spec-ext-signature-collective

## Manuscript references (lines 2698-2868)

### §Temporal Structure and the Signature Problem (2698-2828) with `\label{sec:signature_resolution}`

- `:2701-2702` Sector split paragraph: gauge group enlarged to GL(K,C) on connection sector only; transports of beliefs restricted to GL(K,R); numerical verification at K=2,3 shows complex Ω produces complex KL with negative real part. Real Gaussian probability theory, Fisher-Rao geometry, KL nonnegativity, and F ≥ 0 live on GL(K,R) sector.
- `:2704` Complexification confined to gauge frame field. Real Lorentzian metric recovered via real-part projection.
- `:2706` Stronger quantum extension (density matrices on Hilbert space, Born rule, signed measures) is "a separate program that this manuscript does not adopt."
- `:2708` Regime I status: F_{μν} ≡ 0 throughout; signature mechanism is via tr(A_μ A_ν), which is gauge-noninvariant; NOT Yang-Mills.

- `:2710-2712` Signature Problem: Diagnosis — induced metrics G^(q), G^(p) are positive semi-definite from score outer products. Spacetime needs Lorentzian (-,+,+,+).

- `:2714-2716` Compact gauge groups force Riemannian via similarity transform Ω Σ Ω^T preserving eigenvalues. Non-compactness necessary but NOT sufficient (Sylvester's law).

- `:2718-2726` GL(K) and GL(K,C) Resolution — three paragraphs:
  - Non-compact real forms can produce indefinite consensus metric
  - Complexification GL(K,C) contains SO(1,3) via vector representation SO+(1,3) ⊂ GL(4,R) ⊂ GL(4,C) and spinor SL(2,C) ≅ Spin+(1,3) ⊂ GL(2,C)
  - Frame-twist metric: tr(A_μ A_ν) on non-compact form with imaginary temporal component gives G_ττ < 0

- `:2728-2740` Postulates Required: three-step pathway from SO(3) to Lorentzian — GL(K,R), complexified frames, SO(1,3) subgroup restriction. "Each step is mathematically well-defined; the dynamical content... is unresolved."

- `:2742-2783` Worked Example: GL(2,C) Gauge Frames... with `\label{sec:worked_signature}`. Boxed Eq. eq:lorentzian_metric at 2768: ds² = -2(∂_τ ψ_τ)² dτ² + 2(∂_x ψ_x)² dx².
  - 2745: "Wick rotation in the Lie algebra rather than on the base coordinates"
  - 2747: T = diag(1,-1) in sl(2,R), explicit "T is NOT in compact so(2)"
  - 2753: postulated imaginary frame Eq. eq:complex_gauge_frame
  - 2756: separability ansatz ψ_τ(τ), ψ_x(x) registered
  - 2771: "real-part projection... has no Wick counterpart"; conformal class observation (±2 → ∓1 under normalized T)
  - 2773-2779: local Lorentz transformations Λ(ξ) verified Λ^T η Λ = η; 1+3 vs 2+2 distinction not derived
  - 2783: open question — free-energy mechanism for imaginary φ_τ selection, 1+3 vs 2+2 split

- `:2785-2808` Alternative Route: Lorentzian Conformal Class from Finite-Speed Epistemic Causality with `\label{sec:causal_cone_route}`
  - Setup: M ≅ R_τ × Σ, shared τ postulate (substantive), positive-definite spatial h, finite max speed c_I, inner-product not Finsler
  - Eq. eq:causal_cone_metric at 2796: g = -c_I² dτ² + h_ab dx^a dx^b with one negative eigenvalue, Lorentzian by Sylvester
  - Conformal class ambiguity registered; dimension count not selected
  - 2804: tension with first-order natural-gradient dynamics. Three potential resolution routes named (telegraph-type continuum, hyperbolic dynamics, architectural finite-speed constraint), "None of these is currently realized."
  - 2808: status — two routes "have disjoint open problems"; strengthen each other as parallel existence

- `:2810-2828` Temporal Direction from Belief Trajectories — Gram-Schmidt tangent decomposition; Eq. eq:lor_belief_metric at 2823 with explicit "leading minus sign is imposed by ansatz, not derived." Alternative real GL(K,R) route with mixed compact+non-compact generators noted.

### §Collective Geometry and Gauge Invariance (2830-2868)

- `:2832-2842` Consensus Metrics subsubsection — naive average flaw (gauge-frame-dependent)
- `:2844-2862` Gauge-Invariant Metric Construction with `\label{sec:consensus_metric}`
  - Eq at 2850: gauge-averaged metric ⟨G_i⟩(c) over Haar measure
  - 2853: critical paragraph — Haar averaging over local g(c) is infinite-dimensional functional integral; requires gauge fixing or regulator; SO(1,3) Haar measure infinite even for constant g; "non-compact ... carries the additional obstruction"; consensus metric "a heuristic target rather than a completed observable... gauge-invariance is conditional on a regulator whose construction is left to future work."
  - Eq. eq:consensus_metric at 2858
  - 2862: extensive epistemic-register registration — "those statements should be read in this conditional sense"

- `:2864-2868` Connection to Physical Gauge Invariance — "cognitive-first perspective offers an alternative hypothesis"; closing "Whether specific subgroups are dynamically selected by free energy minimization remains an open question."

## Canon excerpts

### Wick rotation canon
- **Wick, G.-C. (1954)**, "Properties of Bethe-Salpeter wave functions," *Phys. Rev.* 96, 1124 — original
- **Zinn-Justin, J. (2002)**, *Quantum Field Theory and Critical Phenomena*, OUP §3 — Wick rotation on base coordinates, real-valued throughout
- **Hartle, J. B., Hawking, S. W. (1983)**, "Wave function of the Universe," *Phys. Rev. D* 28, 2960 — Wick rotation on path-integral measure

### Sylvester's law canon
- **Sylvester, J. J. (1852)** — original; signature invariance under non-singular real transformations
- **Horn, R. A., Johnson, C. R. (2013)**, *Matrix Analysis*, Cambridge §4.5.8 — modern statement

### Lorentz / SL(2,C) representation canon
- **Penrose, R., Rindler, W. (1984)**, *Spinors and Space-Time*, Vol. 1 — spinor double cover
- **Weinberg, S. (1995)**, *The Quantum Theory of Fields* Vol. 1 §2.5-2.7 — vector vs spinor representations of Lorentz group

### Gauge orbit averaging / Haar measure canon
- **Gribov, V. (1978)**, "Quantization of non-abelian gauge theories," *Nucl. Phys. B* 139, 1 — Gribov copies; gauge fixing
- **Faddeev, L. D., Popov, V. N. (1967)**, "Feynman diagrams for the Yang-Mills field," *Phys. Lett. B* 25, 29 — Faddeev-Popov procedure; infinite gauge volume issue
- **Helgason, S. (1978)**, *Differential Geometry, Lie Groups, and Symmetric Spaces*, AMS — Haar measure on compact vs non-compact groups; infinite Haar measure on non-compact

### Causal structure / conformal canon
- **Hawking, S. W., Ellis, G. F. R. (1973)**, *The Large Scale Structure of Space-Time*, Cambridge — causal cones determine conformal class
- **Penrose, R. (1972)**, *Techniques of Differential Topology in Relativity*, SIAM — conformal class from null cone

## What this evidence does NOT settle

1. **Wick rotation analogy stretch at line 2745.** The phrase "Wick rotation in the Lie algebra rather than on the base coordinates" is a structural analogy. Standard Wick rotation continues a real coordinate τ → iτ and the resulting metric is real (Hartle-Hawking 1983, Zinn-Justin 2002 §3). The framework's construction continues a Lie-algebra element to imaginary and produces a complex-valued bilinear form, then takes real part. The 2771 paragraph explicitly admits "no Wick counterpart" for the real-part projection. The question is whether calling this a "Wick rotation in the Lie algebra" is the right rhetorical move or whether a different label ("imaginary-frame postulate plus real-projection," "frame-Wick") would be clearer.

2. **1+3 vs 2+2 selection.** Line 2779 admits "The signature split 1+3 is selected by the input choice (one imaginary direction); the construction does not currently distinguish 1+3 from 2+2 on dynamical grounds." This is honest but the broader signature subsection's title and framing imply Lorentzian (1+3) as the target. The construction reaches a Lorentzian conformal class with arbitrary signature (1, n-1); 2+2 is available with two imaginary directions.

3. **Sylvester's law direction at 2799.** The 2799 sentence "Since c_I^2 > 0 and h_ab is positive definite, g has one negative eigenvalue and dim Σ positive eigenvalues, hence Lorentzian signature (-,+,...,+) by Sylvester's law" — Sylvester's law of inertia says signature is invariant under non-singular real transformations, so the signature of g is determined by its definition (-c_I^2, h_ab). The "by Sylvester's law" attribution is a courtesy citation rather than a substantive invocation.

4. **Shared τ postulate at 2791.** The "shared update parameter τ" is explicitly admitted as a substantive postulate at odds with the per-agent Fisher arc length elsewhere in the manuscript ("the interpretation of τ as a single global parameter is itself a substantive postulate"). Whether this is enough hedging given that the entire causal-cone route hinges on it.

5. **Connection to Physical Gauge Invariance at 2868.** The framing "gauge invariance arises as a consistency requirement for multi-agent consensus" — already covered in the Gauge Invariance as Consensus Discussion subsection (debated; edits at commit 68ebfec8). Whether the 2864-2868 paragraph should cross-reference the Discussion debate edits.

Teams should verify points 1-3 against the canon; points 4-5 are within-manuscript consistency.
