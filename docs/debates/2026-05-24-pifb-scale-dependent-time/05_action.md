# Action — pifb-scale-dependent-time

**From verdict:** REMAND

## Recommended action

Three surgical manuscript edits in `Attention/Participatory_it_from_bit.tex`, none touching the physics or the four sound components. After them, the "rigorous formalization" billing is defensible across components A, B, C. No further debate round needed — this is a copy-edit pass against a verified itemization.

1. **Line ~2649 (stated mechanism for $Z_I = 1/N$).** Replace the causal clause "incoherent injection, with power spread uniformly over all $N$ eigenmodes, projects onto this single mode with weight $1/N$." The $1/N$ is variance-of-the-mean under independence: the barycenter is the projection onto the constant zero mode $\mathbf{1}/\sqrt N$ of $L$, and independent injection of any (not necessarily isotropic) per-constituent covariance gives the zero mode a $1/N$ share of total power. Equipartition over all $N$ modes is the special isotropic case and is not what delivers the result, so it must not be stated as the reason. Soften "the spectral origin of $Z_I = 1/N$" / "exactly" accordingly.

2. **Line ~2651 (Eq. `Ilow_eq_dispersion`, first equality).** Change the first `=` (between $\mathcal{I}_{s\to s+1}$ and the pure-mean dispersion) to `≈`, and state the condition: leading order in the post-transport mean difference with $\Omega_{i,I}\approx I$ and $\Sigma_i\approx\Sigma_I$, inheriting the "to leading order" qualifier already carried at the Laplacian-reduction step (`:2642`). The full definition of $\mathcal{I}_{s\to s+1}$ at `:2233` is a sum of transported Gaussian KLs (carrying $\Omega_{i,I}=\exp(\phi_i)\exp(-\phi_I)$ and covariance terms) that collapses to the pure-mean dispersion only on the gauge-aligned, equal-covariance locus. The **second** equality (dispersion $=\tfrac12\sum_{a\ge2}\|\hat\mu_a\|_\Lambda^2$) is exact Parseval on the orthonormal eigenbasis and stays `=`.

3. **Line ~2656 (renormalization verdict, first sentence).** Drop or repair the step "because the bit is reparametrization-invariant, … the slowdown cannot reside in a rescaling of the unit." It conflates Cencov coordinate-invariance on the belief manifold with Wilsonian RG invariance across scales — distinct transformations. The load-bearing second argument (the $1/N$ is a law-of-large-numbers ratio with no critical point; promotion to a dynamical critical exponent $z$ defined by $\tau\sim\xi^z$ is the open fixed-point problem flagged at `:2227`) already carries the conclusion and should be left to do so.

## What survives unchanged (do not edit)

- **(B)** $\mathrm{KL} = \tfrac12 ds^2$ (`:2593`, Eq. `kl_arclength`) — exact pure-mean equal-covariance specialization of the second-order Fisher expansion [`external_canon_math.md:35,:42`; AmariNagaoka2000 Ch. 2].
- **(A) trace-ratio** $Z_I \to 1/N$ (Eq. `bit_production_ratio`, `Z_decomposition`) — theorem under the independence + common-$\Lambda$ assumptions stated at `:2633`.
- **Eq. `Ilow_eq_dispersion` second equality** — exact Parseval.
- **(C) conclusion** ($Z^{(s)}$ is a comoving-units choice, not a dynamical critical exponent) — correct on the LLN / no-critical-point ground [Hohenberg & Halperin 1977; open fixed point at `:2227`].

## Follow-up debates (if any)

None required. The three edits are a copy-edit pass, not a re-derivation. Optional: a separate debate on the upstream §Bit-Counting Time Planck-time speculation (`:2596`) was not in scope here.
