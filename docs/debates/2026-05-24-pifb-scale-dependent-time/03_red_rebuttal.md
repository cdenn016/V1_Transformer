# Red Rebuttal — pifb-scale-dependent-time

## Concession

I grant blue's tiered defense on the components where it cites correctly, and I grant more than blue does on one point the dispatching framing told me to attack.

1. **(B) is exact, not an analogy.** Blue is right. The Gaussian KL closed form [`external_canon_math.md:42`; KingmaWelling2014 App. B] specializes at $\Sigma_q = \Sigma_p = \Sigma$ and a pure-mean step to $\tfrac12\delta\mu^\top\Lambda\delta\mu$ with the trace, log-det, and dimension terms cancelling identically; the Amari–Nagaoka second-order expansion [`external_canon_math.md:35`] carries no residual because the covariance sector does not move. Blue's executed check (rel err $2\times10^{-11}$) confirms it. Eq. `kl_arclength` (`:2593`) writing `=` is the textbook specialization, and "one JND unit $ds=1$ is $\tfrac12$ bit" (`:2596`) is arithmetic on that identity. (B) is rigorous.

2. **The (A) trace-ratio limit $Z_I \to 1/N$ needs no isotropy.** Blue is right by linearity of the trace: $\mathrm{tr}(\Lambda\, v_\xi/N)/\mathrm{tr}(\Lambda\, v_\xi) = 1/N$ for any common $\Lambda$ and any $v_\xi$. The decomposition $\mathrm{Cov}(\Delta\mu_I) = \mathrm{Cov}(c) + v_\xi/N$ (Eq. `Z_decomposition`, `:2633`) is the variance-of-the-mean under independence, with both required assumptions — independence of $\xi_i$ across constituents and common $v_\xi$ — stated at the point of use (`:2633`). I do not contest (A)'s trace-ratio.

3. **(C) is the conservative, canon-consistent move.** Blue's Hohenberg–Halperin citation [Rev. Mod. Phys. 49, 435 (1977)] is correct: the dynamic critical exponent $z$ is defined at a critical point through $\tau_{\text{char}} \sim \xi^z$. $Z^{(s)}$ is a law-of-large-numbers ratio defined for every finite $N$ with no critical point and no diverging $\xi$; declining to call it $z$ and deferring to the open RG fixed point (`:2654`) matches the canonical definition. (C) is rigorous.

4. **I withdraw any attack on Eq. `Ilow_eq_dispersion` (`:2651`) carrying a strict `=`.** I verified the second equality symbolically. The decomposition $\sum_i \tfrac12(\mu_i-\mu_I)^\top\Lambda(\mu_i-\mu_I) = \tfrac12\sum_{a\ge2}\|\hat\mu_a\|_\Lambda^2$ is exact to machine precision (rel diff $1.4\times10^{-16}$, $N=6$, $K=3$, random SPD $\Lambda$, random symmetric $L$). It is Parseval on the orthonormal Laplacian eigenbasis with the constant zero-mode equal to the barycenter, on a tensor product where $\Lambda$ acts on features and the eigenbasis on nodes — no commutation of $\Lambda$ with $L$ is needed. Blue called this "the quadratic-regime statement"; it is in fact exact within the equal-covariance Gaussian mean sector the subsection already restricts to. Neither side should attack it.

## Core attack

Blue's concession #1 ("the spectral uniform-power step is a stated regime, not a derived consequence") is too charitable to the manuscript and conceals the real defect. The manuscript prose at `:2649` does not merely *state* uniform power as a regime — it offers it as the **causal mechanism** for the result:

> "...incoherent injection, with power spread uniformly over all $N$ eigenmodes, projects onto this single mode with weight $1/N$, **which is the spectral origin of $Z_I = 1/N$**" (`Attention/Participatory_it_from_bit.tex:2649`).

The clause "with power spread uniformly over all $N$ eigenmodes" is non-restrictive — set off by commas, it asserts that incoherent injection *has* uniform mode power, and "which is the spectral origin of" makes that uniformity the *reason* the projection weight is $1/N$. This is a derivational claim, and it is false under the manuscript's own definition of incoherence.

Executed counterexample ($N=8$, $4\times10^5$ draws, scalar feature for clarity; random symmetric weighted Laplacian $L$):

```
Case A independent isotropic (common scalar variance):
  per-mode power = [0.999 1.001 1.002 0.998 1.001 0.995 0.997 1.000]   <- uniform
  zero-mode fraction of total = 0.1250   (1/N = 0.125)

Case B independent heteroscedastic (per-node variance 0.2..3.0):
  per-mode power = [1.60 2.34 1.17 1.40 2.21 1.04 1.07 1.98]            <- NOT uniform
  zero-mode fraction of total = 0.12497  (1/N = 0.125)
```

Case B injection is independent across constituents — incoherent in the manuscript's exact sense (`:2633`: "$\xi_i$ independent across constituents"). The only thing dropped is the *common* $v_\xi$, replaced by per-node variances. The per-mode power is then decisively non-uniform (spanning $1.04$ to $2.34$), so the stated mechanism "power spread uniformly over all $N$ eigenmodes" does not obtain. Yet the zero-mode share of total power is still exactly $1/N$. The correct mechanism is the one two paragraphs earlier: the barycenter is $\mathbf{1}/\sqrt N$, so its variance is the variance-of-the-mean, $N\cdot\mathrm{Var}(\mu_I) = N \cdot (\text{avg node variance})/N$, which holds for any independent injection regardless of mode-power distribution.

So the prose at `:2649` derives a correct result ($Z_I = 1/N$) from a mechanism (uniform mode power) that the manuscript's own incoherence assumption does not deliver. A right conclusion reached through a stated-but-false mechanism is not a "rigorous formalization" of that conclusion; it is the trace-ratio result of `:2633` dressed in a spectral story that does not track. Blue's "stated regime, not derived" framing implies the manuscript is silent on *why* the $1/N$ holds spectrally; the prose is not silent — it asserts a specific and incorrect why ("which is the spectral origin of").

The asymmetry sharpens the point. Where node-correlation is present (genuine coherence beyond the $c+\xi_i$ split), the zero-mode fraction departs from $1/N$ — I measured $0.087 \ne 0.125$ under a random SPD node-covariance ($N=8$). The $1/N$ is governed entirely by the *independence* structure, never by the *mode-power* distribution. The manuscript pinned it on the wrong variable.

## Defense

My opening attacked the word "rigorous" on the most novel component, the spectral realization of (A). Blue's rescue is that independence delivers the trace-ratio $1/N$ and that the uniform-power step is "named rather than derived." That rescue saves the *number* and concedes the *spectral narrative is under-discharged*. It does not save the manuscript's *as-written* prose, and the claim under debate is explicitly about the prose: "whether the manuscript prose, as written, is a rigorous formalization" (`00_claim.md:27`). Three reinforcements.

First, blue's own concession #1 grants that "incoherence does not by itself force the injection covariance to be isotropic in the Laplacian eigenbasis." Case B is the concrete instance: incoherent (independent) yet anisotropic-in-modes. Blue stops at "the manuscript names the regime." But `:2649` does not *name* uniform power as a sufficient condition it is assuming — it *predicates* it of incoherent injection ("incoherent injection, with power spread uniformly...") and then makes it the cause. Blue conceded the milder defect (under-derivation) and missed the load-bearing one (a false predication serving as the offered derivation).

Second, the manuscript itself distinguishes `≈` from `=` deliberately elsewhere in the same subsection — Eq. `coupling_laplacian` (`:2644`) carries `≈`, Eq. `Ilow_eq_dispersion` (`:2651`) carries `=`, and $\Lambda_I \approx \Lambda$ (`:2640`) carries `≈` exactly because, as blue concedes, equality holds only on the gauge-aligned locus ($\|\Omega\Sigma\Omega^\top - \Sigma\| = 44.5 \ne 0$ off it). The author is fluent in marking what is exact versus approximate at the equation level. That fluency does not extend to the *prose* derivation at `:2649`, where a false causal clause is stated in the declarative. The standard for "rigorous formalization" is set by the manuscript's own equation-level discipline; the prose falls short of it.

Third, against the canon. The result the prose claims to derive — uniform equipartition of variance over Laplacian eigenmodes — is the special isotropic case [Fiedler1973; Chung 1997 §1 on the constant zero-eigenvector], not a consequence of independence. Independence gives variance-of-the-mean (a first-moment fact about $\mathbf{1}^\top\mu/N$), not equipartition (a statement about all $N$ modes). The manuscript conflates the two: it invokes the all-mode equipartition picture to explain a single-mode result that follows from independence alone. The canon supports the zero-eigenvector identity and the algebraic-connectivity relaxation rate $2\eta_\mu\lambda_2$ (`:2654`, correctly cited), but it does not license "incoherent $\Rightarrow$ uniform mode power." That inference is the manuscript's, and Case B refutes it.

I do not claim the subsection is worthless or that $Z_I = 1/N$ is wrong. (B), (C), and the (A) trace-ratio survive. I claim the specific word "rigorous" in the compound claim fails on the (A) spectral component as written, because the prose at `:2649` offers a false mechanism as the derivation of a correct number — and the claim is adjudicated on the prose, not on the author's private notes (`00_claim.md:27`).
