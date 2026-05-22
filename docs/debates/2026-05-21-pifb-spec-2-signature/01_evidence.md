# Evidence Pack — pifb-spec-2-signature

## Manuscript references (read the actual TeX, not this summary)

Source file: `C:/Users/chris and christine/Desktop/V13_Gauge_Transformer/Attention/Participatory_it_from_bit.tex`

### §Temporal Structure and the Signature Problem (lines 2773-2903)

- **2776-2781 — Sector split paragraph.** Gauge group enlarged from `GL(K,R)` to `GL(K,C)` on connection sector only; belief fiber remains real Gaussians; transport on the fiber restricted to `Omega in GL(K,R)`. Numerical verification claim at `K=2,3`: complex `Omega` produces non-Hermitian sandwich and complex Gaussian KL with negative real part and nonzero imaginary part. Explicit statement: real Gaussian KL, Fisher-Rao geometry, KL non-negativity, `F >= 0` live on `GL(K,R)` sector.
- **2783 — Regime status paragraph.** Construction stays inside Regime I: `A = U^-1 dU`, `F_{mu nu} = 0` identically. Signature mechanism via frame-twist `tr(A_mu A_nu)` not via Yang-Mills `tr(FF)`. Bilinear form on g chosen as `<A,B> := tr(AB)` on `gl(K,C)` (indefinite); compact form would use `-tr(AB)`.
- **2785-2787 (`Diagnosis`)** — `G_i^(q)`, `G_i^(p)` are positive *semi*-definite (expectations of outer products of scores). Zero eigenvalues where section is locally constant.
- **2789-2791 (`Why Compact Gauge Groups Force Riemannian`)** — `SO(N)` preserves positive-definiteness (similarity transform, eigenvalues preserved). Non-compactness necessary but not sufficient — Sylvester's law preserves positivity under `GL(K,R)` real similarity. **The framework's restriction to compact SO(3) IS what currently forces Riemannian signature.**
- **2793-2801 (`The GL(K) and GL(K,C) Resolution`)** — Three paragraphs:
  - 1. Non-compact real forms `GL(K,R)`: consensus metric over non-compact subgroup can acquire indefinite character.
  - 2. Complexification `GL(K,C)`: contains SO+(1,3) vector rep in GL(4,R) ⊂ GL(4,C); spinor double cover SL(2,C) ≅ Spin+(1,3) in GL(2,C). **Verify both claims against standard Lie-group references.**
  - 3. Frame-twist metric: with imaginary `phi_tau = i psi_tau · T` and non-compact `T` with `tr(T^2) > 0`, the `i^2 = -1` flips sign of `G^tw_{tau tau} = -(d_tau psi_tau)^2 tr(T^2) < 0`.
- **2803-2815 (`Postulates Required for Indefinite Pullback`)** — Three-step pathway: (i) `GL(K,R)` with real Gaussians (validated empirically); (ii) Complexified gauge frames acting on real beliefs; (iii) Subgroup restriction to `SO(p,q)` with `SO(1,3)` selected by one-imaginary-direction input. **The 1+3 split is fixed by the input choice, not derived from FEP dynamics.** Each step "mathematically well-defined" but dynamical content unresolved.
- **2817-2858 (`sec:worked_signature`) — Worked Example.** GL(2,C) on 2D base C with coordinates (tau, x). Generator `T = diag(1, -1) in sl(2,R)` with `tr(T^2) = 2`. Postulated `phi(tau,x) = i psi_tau(tau,x) · T + psi_x(tau,x) · T` (Eq. `complex_gauge_frame`). Separable ansatz `psi_tau = psi_tau(tau)`, `psi_x = psi_x(x)`. Linearized connection `A_mu = ∂_mu phi`. Result:
  - `G_{tau tau} = -2(∂_tau psi_tau)^2 < 0`
  - `G_{xx} = +2(∂_x psi_x)^2 > 0`
  - `G_{tau x} = i(∂_tau psi_tau)(∂_x psi_x) tr(T^2) in iR`
  - Real-part projection: `G^Lor_{tau x} = 0`, `ds^2 = -2(∂_tau psi_tau)^2 dtau^2 + 2(∂_x psi_x)^2 dx^2` (Eq. `lorentzian_metric`).
- **2841-2846 — Real-part projection paragraph.** Explicit acknowledgement that real-part projection is "an additional choice, separate from Eq. complex_gauge_frame". Wick-rotation distinction: standard Wick continues `tau -> i tau` on base, the construction here continues `phi_tau -> i phi_tau` in Lie algebra and adds real-part projection without Wick counterpart. **Derivation gap flagged.** Note about `+/- 2` coefficients being artefact of unnormalized `T`.
- **2848-2854 (Local SO+(1,1) frame group)** — Lorentz transformations `Lambda(xi) = ((cosh, sinh), (sinh, cosh))`, `Lambda^T eta Lambda = eta`. Tetrad rescaling absorbs `|∂psi|` factors. 4D extension to `SO+(1,3)` with three real + one imaginary direction. Spinor SL(2,C) vs vector rep distinguished.
- **2856 — Three features.** Indefinite signature from connection, not fiber. Structural existence proof, not dynamical derivation. Total metric `G^total = G^tw + G^Fisher` indefinite when frame-twist dominates temporal direction.
- **2858 — Central open question.** Whether free-energy mechanism selects imaginary `phi_tau` over real one, and whether same mechanism singles out 1+3 vs 2+2.
- **2860-2883 (`sec:causal_cone_route`) — Alternative Route.** Postulates: M = R_tau x Sigma product, spatial info-metric `h_ab`, finite max speed `c_I`, inner-product (not Finsler) influence boundary. Result Eq. `causal_cone_metric`: `g = -c_I^2 dtau^2 + h_ab dx^a dx^b`, signature (-,+,...,+) by Sylvester. Conformal-class ambiguity explicit. Dimension count NOT selected. **Tension with first-order dynamics paragraph (2879-2880):** natural-gradient flow is parabolic, infinite-speed in naive continuum limit; three potential routes flagged (telegraph-type continuum, second-order hyperbolic dynamics, architectural finite-speed constraint).
- **2885-2903 (`Temporal Direction from Belief Trajectories`)** — Tangent decomposition `T_q B = R qdot + qdot^perp`. Singles out direction but NOT sign. Eq. `lor_belief_metric` ds^2 = -g_B(qdot,qdot) dtau^2 + sum g_B(e_i,e_i) dx_i^2 — minus sign "imposed by ansatz, not derived". Sylvester's law rules out sign flip under SO(K) alone. Mixed compact/non-compact alternative route flagged.

## Canon excerpts and external sources to verify

Canon files at `C:/Users/chris and christine/Desktop/V13_Gauge_Transformer/.claude/agents/vfe-knowledge/`.

Specific claims to verify against external canon:
- **Sylvester's law of inertia** — under real similarity `S^T A S` with S invertible real, the number of positive/negative/zero eigenvalues of symmetric A is preserved. **Verify the 2789-2791 claim that this rules out a sign flip under SO(K).**
- **SL(2,C) ≅ Spin+(1,3) and vector rep in GL(4,R)** — standard from Nakahara 2003 §10, or Wikipedia "Lorentz group" + "Spin group". **Verify claims at 2799 and 2854.**
- **Wick rotation** — `tau -> i tau` on base manifold produces real Euclidean metric. References: Schlingemann "From Euclidean Field Theory To Quantum Field Theory" or Wightman & Streater. **The construction's distinction from standard Wick at 2820, 2846 must be honest — verify whether some literature DOES perform a Lie-algebra Wick-like continuation with similar real-part projection.**
- **Killing form on gl(K,C)** — `B(X,Y) = 2K tr(XY) - 2 tr(X) tr(Y)`. Indefinite (positive on Hermitian, negative on anti-Hermitian). **Compare to the +/- tr conventions at 2698-2699.**
- **SO+(1,1) as local frame group** — `O(p,q)` preserves diagonal indefinite form. SO+(1,1) is connected 1D Lorentz boost group, parametrized by rapidity. Verify the construction's claim at 2848 that the tetrad rescaling absorbs `|∂psi|` factors so `Lambda` acts on tetrad indices.
- **Causal-cone construction** — `g_{munu} dc^mu dc^nu = -c^2 dtau^2 + h_ab dx^a dx^b` is standard ADM-type decomposition; Lorentzian signature by Sylvester. Verify at 2870-2874.

## Specific math check the agents should perform (per advisor)

The text claims `T = diag(1,-1) in sl(2,R)` (line 2822), and then claims the resulting local frame group at non-degenerate points is `SO+(1,1)` (line 2848). Verify:
1. Is `T = diag(1,-1)` actually in `sl(2,R)` and `not` in any compact form? (It is symmetric and traceless, so yes — `sl(2,R)` not `so(2)`.)
2. Does the local frame group `SO+(1,1)` follow from the worked metric `ds^2 = -2(∂_tau psi_tau)^2 dtau^2 + 2(∂_x psi_x)^2 dx^2`, or is it asserted independently? The frame-group claim is a statement about the tangent-space orthonormal-frame transformations preserving the local metric. After tetrad rescaling `e_0 propto 1/|∂_tau psi_tau|`, `e_1 propto 1/|∂_x psi_x|`, the metric in tetrad indices is `eta = diag(-1,+1)`, and SO+(1,1) is the connected component of O(1,1) preserving this. **Check whether this follows cleanly or whether the real-part projection plays a non-trivial role in establishing it.**
3. Does the construction need `|∂_tau psi_tau| != 0` and `|∂_x psi_x| != 0` (non-degeneracy) — does the text flag this?

## What this evidence does NOT settle

1. Whether a free-energy mechanism actually selects the imaginary `phi_tau` over the real one of the same magnitude under any refinement of the variational principle. The text concedes this is open at 2858, 2903.
2. Whether the causal-cone route's reconciliation with first-order natural-gradient dynamics is achievable via telegraph-type continuum limit, second-order dynamics, or architectural constraint. Text concedes openness at 2880.
3. Whether the GL(K,C) sector split is fully consistent across all consumers in the manuscript: belief fiber stays real, connection complexified. Cross-check needed against §pullback construction in Debate 1 territory.
4. Whether the +tr(AB) vs -tr(AB) convention is consistently applied across all subsections that use it (this section, the pullback section, and the consensus metric section).
5. Whether the manuscript's claim at 2854 that "the construction here uses the vector representation on the four-dimensional base" is consistent with the 2D worked example actually presented (which is in `sl(2,R)`, not the vector rep of SO(1,1) which is 2D and not in sl(2,R) in any natural sense — sl(2,R) is 3D as a Lie algebra and its 2D defining rep is on R^2, which IS what the construction uses; verify language carefully).
