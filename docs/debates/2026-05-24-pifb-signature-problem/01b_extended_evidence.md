# Extended Evidence — pifb-signature-problem (harvested canon)

Concatenated, deduplicated canon discovered by the coordinators' expert panels. For the judges.

## Blue panel (Phase 2, opening)

### Verified symbolic computations (sympy, executed)
- Trace algebra: with T=diag(1,−1), tr(T²)=2; A_τ=i(∂ψ_τ)T, A_x=(∂ψ_x)T ⇒ G_ττ=−2(∂ψ_τ)², G_xx=+2(∂ψ_x)², G_τx=2i(∂ψ_τ)(∂ψ_x)∈iℝ. Matches manuscript Eqs. at :2883–2885 exactly.
- Rank: complex bilinear form det = 0 (rank one); Re(·) = diag(−2(∂ψ_τ)², +2(∂ψ_x)²), det = −4(∂ψ_τ)²(∂ψ_x)² < 0, eigenvalues {−2(∂ψ_τ)², +2(∂ψ_x)²} (rank two, signature (−,+)). Confirms the manuscript's "rank-one complex → rank-two real" claim at :2887.
- Lorentz invariance: Λ(ξ)=[[cosh ξ, sinh ξ],[sinh ξ, cosh ξ]], η=diag(−1,+1) ⇒ ΛᵀηΛ−η = 0 for all ξ. Confirms :2897–2900.
- Causal-cone Sylvester count: g=diag(−c², h₁, h₂, h₃), c>0, hᵢ>0 ⇒ eigenvalues {−c², h₁, h₂, h₃} (one negative, three positive), signature (−,+,+,+). Confirms :2923.
- Compact-generator sensitivity: T_c=[[0,1],[−1,0]] ⇒ tr(T_c²)=−2 (sign flip), confirming the disclosed single-generator-collapse sensitivity at :2872, :2902.

### External canon citations introduced
- [Horn & Johnson, *Matrix Analysis* (2nd ed.), §4.5] — Sylvester's law of inertia: signature of a real symmetric form is a congruence invariant. Grounds both the GL(K,ℝ) positive-definiteness no-go (:2837, :2950) and the causal-cone count (:2923).
- [Knapp, *Lie Groups Beyond an Introduction*, ch. I] — Killing/trace form: negative-definite on a compact real form, indefinite on a split/non-compact real form; tr(AB) indefinite on 𝔤𝔩. Grounds the +tr vs −tr convention dependence (:2829, :2868).
- [Hall, *Lie Groups, Lie Algebras, and Representations*, ch. 5] — SL(2,ℂ) ≅ Spin⁺(1,3) spin double cover; vector rep of SO⁺(1,3). Confirmed independently via Lorentz-group canon (SL(2,ℂ) double-covers the restricted Lorentz group; SO⁺(1,3) is the identity component of O(1,3)).
- [Streater-Wightman / Folland, *Quantum Field Theory*] — Wick rotation relates real forms of the complexified rotation group SO(4,ℂ). Grounds :2845.
- [AmariNagaoka2000, *Methods of Information Geometry*, Ch. 2] — closed-form Gaussian KL with log-determinant term; the term goes complex under non-Hermitian Σ. Grounds the sector-split necessity at :2823.
- [Cencov1972] — Fisher metric is the unique sufficient-statistic-invariant Riemannian metric, hence positive-definite; complexifying the fiber would break this. Grounds :2856.
- [Evans, *Partial Differential Equations* (2nd ed.), §2.3, §2.4] — heat (parabolic, infinite propagation speed) vs wave (hyperbolic, finite domain of dependence). Grounds the first-order-dynamics tension at :2929.
- [Popper, *The Logic of Scientific Discovery*, §15] — existential vs universal statement asymmetry; an "it can exhibit X" claim is the appropriate falsifiable existential form, distinct from a lawlike derivation claim. Grounds the existence-demonstration framing at :2825, :2902, :2907.

## Red panel (Phase 2, opening)

### External canon citations introduced (deduped against the blue list above)
- [Popper, *The Logic of Scientific Discovery*, §6, §31–33] — falsifiability as demarcation: a statement excluding no outcome in its domain is empirically contentless. Bears on "compatible with signature S" when the same construction is equally compatible with ¬S (Riemannian, (−,−), (−,+)).
- [Forster & Sober (1994), *BJPS* 45(1)] — ad-hoc parameter fitting; an existence demonstration requiring ≥5 tuned postulates to reach the target signature sits at the ad-hoc-fit limit.
- [Lee, *Introduction to Smooth Manifolds* (2nd ed.), Ch. 13] — a (pseudo-)metric is by definition a smooth symmetric *non-degenerate real* (0,2)-tensor; degenerate or complex-valued symmetric forms are not metrics.
- [O'Neill, *Semi-Riemannian Geometry with Applications to Relativity* (1983), Ch. 2–3] — a Lorentzian metric is a non-degenerate real symmetric (0,2)-tensor of signature (−,+,…,+); the canonical reference for the exact object the claim must produce.
- [John, *Partial Differential Equations* (4th ed.), Ch. 7] — classification of second-order PDE; domain-of-dependence / characteristic-cone distinction between parabolic and hyperbolic types. Reinforces the Evans citation on the causal-cone first-order tension.

### Executed sympy verifications (red — additional to the blue list)
- **Mixed real generators, NO complexification:** one compact T_c=[[0,1],[−1,0]] (tr=−2) and one non-compact T=diag(1,−1) (tr=+2) under +tr ⇒ Re(G) eigenvalues {−2(∂ψ)², +2(∂ψ)²} — indefinite signature with NO i and NO real-part projection. Reproduces the manuscript's own :2950 disclosure; confirms the complex apparatus of §sec:worked_signature is inessential to the signature mechanism.
- **Compact-generator sign flip (full Re(G)):** T_x∈𝔰𝔬(2) with T_τ=diag(1,−1) ⇒ Re(G) eigenvalues {−2(∂ψ)², −2(∂ψ)²} — signature (−,−), not Lorentzian. The generator choice, not the framework, selects the signature.
- **Real-only single generator (no i):** both diagonal entries +2(∂ψ)² — Riemannian. In the collapsed case the imaginary postulate does all the sign work.
- **Non-separable single generator:** Re(G_ττ)=2((∂_τψ_x)²−(∂_τψ_τ)²), Re(G_xx)=2((∂_xψ_x)²−(∂_xψ_τ)²). With cross-derivatives present the temporal sign can be positive (timelike→spacelike). The separability ansatz at :2877 is load-bearing for the signature, beyond the "display simplification" framing the manuscript gives it.
- **Complex G eigenvalues (degeneracy):** {0, −2(∂_τψ−∂_xψ)(∂_τψ+∂_xψ)} — the genuine object has a zero eigenvalue (degenerate, not a metric); Re(·) creates the non-degeneracy.
