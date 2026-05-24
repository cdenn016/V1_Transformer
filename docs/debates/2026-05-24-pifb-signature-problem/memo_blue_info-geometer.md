# Blue Memo — info-geometer

## Steelman of the attack
The sector split is incoherent: confining complexification to the connection while the Fisher fiber stays real-positive-definite means the indefinite tr(A_μ A_ν) is doing all the work, decoupled from the framework's actual statistical content (KL between real Gaussians), so the "signature" never touches the information geometry the framework is built on.

## Defense from canon
The sector split is *coherent and conservative*, and its central claim is an analytic fact, not a numerical one:

1. **Why the fiber must stay real.** The Fisher-Rao metric is the unique (up to scale) metric on a statistical manifold invariant under sufficient statistics — Cencov's theorem [Cencov1972; external_canon_math.md §1]. It is positive-(semi)definite by construction as an expectation of score outer products [AmariNagaoka2000 Ch. 2]. A genuinely complex transport Ω would break exactly this: the closed-form Gaussian KL `½[tr(Σ_p⁻¹Σ_q) + (Δμ)⊤Σ_p⁻¹Δμ − K + log(|Σ_p|/|Σ_q|)]` [AmariNagaoka2000 Ch. 2; KingmaWelling2014 App. B] acquires a complex value the moment Σ = ΩΣ₀Ω⊤ is non-Hermitian — the log-determinant term alone goes complex. This is an *analytic* consequence of the closed form, so the manuscript's conservative choice to keep KL≥0 and ℱ≥0 on the real GL(K,ℝ) sector (2823, 2856) is the mathematically forced move, not an arbitrary restriction. The manuscript's "numerically checked at K=2,3" framing (2823) understates a fact that follows from the closed form directly.

2. **The decoupling is acknowledged, not denied.** The manuscript states the signature "is a property of how gauge frames vary over the base manifold, not of the statistical geometry within a single fiber" (2847) and locates it in the bilinear form κ on the Lie algebra, "not the Fisher-Rao metric on the fiber" (2829). The base-vs-fiber separation is standard bundle geometry: the connection lives on the principal bundle, the Fisher metric on the associated statistical fiber, and there is no theorem requiring a single object to be simultaneously the source of both [standard associated-bundle structure, external_canon_math.md §2]. The total metric G^total = G^tw + G^Fisher (2902) is the honest combined object; the indefinite part is the connection contribution, the positive-semidefinite part is Fisher.

## Honest concession (info-geometry side)
Whether the connection-sector signature is *physically tied* to the statistical content, rather than a parallel structure on the same base, is genuinely unsettled (evidence pack, line 30). The claim does not assert that link — it asserts the construction is correct and disclosed. The defended content survives; the deeper coupling question is the manuscript's own open problem.

## External primary-source citation
[AmariNagaoka2000, *Methods of Information Geometry*, Ch. 2 — closed-form Gaussian KL with log-determinant term]; [Cencov1972 — Fisher metric is the unique sufficient-statistic-invariant metric, hence positive-definite].

## Falsification condition (argued unmet)
Fails if the Gaussian KL stays real under a generic complex Ω (it does not — the log|Σ| term goes complex, an analytic fact) or if Cencov-invariance permitted a complex Fisher metric (it does not). Neither holds.
