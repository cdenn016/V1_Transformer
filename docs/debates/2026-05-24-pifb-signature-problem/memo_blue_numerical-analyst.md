# Blue Memo — numerical-analyst

## Steelman of the attack
The causal-cone route postulates a finite maximal epistemic speed c_ℐ, but the framework's actual dynamics are first-order natural-gradient flow, which is parabolic and propagates signals at infinite speed; the route therefore contradicts the framework it claims to extend, and the contradiction is fatal to "structural compatibility."

## Defense from canon
The parabolic/hyperbolic tension is real — and the manuscript states it itself, in the correct technical terms, as an *open problem*, not a solved one:

1. **The PDE fact is correctly stated.** First-order-in-time gradient flow ∂_τ μ = −Σ∇ℱ is the heat-type (parabolic) equation; parabolic equations have infinite propagation speed (the fundamental solution of the heat equation is strictly positive everywhere for any τ>0) [Evans, *Partial Differential Equations*, §2.3]. Hyperbolic (wave-type) equations have a finite domain of dependence / light cone [Evans §2.4]. The manuscript says exactly this: "Naive continuum limits of such flows are parabolic and yield infinite signal speed, so the causal-cone route does not apply directly to the current implementation" (2929). Stating the obstruction in the correct PDE vocabulary, and listing telegraph-limit / second-order-hyperbolic / architectural-constraint as the three candidate fixes (2929) with "None of these is currently realized," is the honest disclosure, not an overclaim.

2. **The signature count itself is dynamics-independent.** Whether or not finite-speed propagation is realized, *if* the postulate holds the Sylvester count g = diag(−c², h) → (−,+,…,+) is correct (sympy verified). The route is explicitly "conditional on the postulates" (2929). An existence demonstration conditional on a stated postulate is not falsified by the postulate being currently unrealized — it is falsified only by the conditional being wrong, which it is not.

3. **Sector-split numerics are reproducible.** The "numerically checked at K=2,3" claim (2823) — complex Ω gives non-Hermitian ΩΣΩ⊤ and complex/negative Gaussian KL — is reproducible and follows analytically from the log-determinant term of the closed-form KL going complex. No finite-precision artifact is needed; the claim is robust.

## External primary-source citation
[Evans, *Partial Differential Equations* (2nd ed.), §2.3 (heat equation, infinite propagation speed) and §2.4 (wave equation, finite domain of dependence)] — the canonical parabolic-vs-hyperbolic signal-speed distinction.

## Falsification condition (argued unmet)
Fails if (i) the manuscript claimed finite-speed propagation *follows from* the existing first-order dynamics — it explicitly denies this (2929); or (ii) the Sylvester count for the causal-cone metric were wrong — sympy confirms it. Neither holds. The first-order tension is disclosed, correctly named, and flagged unresolved — which is what the completeness sub-claim requires.
