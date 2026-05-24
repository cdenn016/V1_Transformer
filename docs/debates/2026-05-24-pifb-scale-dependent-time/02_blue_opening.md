# Blue Opening — pifb-scale-dependent-time

## Steelman (opposing position)

The subsection dresses three distinct objects in shared notation — an exact information-geometric identity, a law-of-large-numbers ratio, and a negative claim about renormalization — and the two novel pieces (the $Z_I = 1/N$ coherence filter and its spectral realization) rest on independence, common-precision, and uniform-spectral-power assumptions that the manuscript invokes selectively rather than discharging, so the "rigorous formalization" billing overreaches on exactly the components the claim marks as most novel.

## Position

The claim is defensible as a tiered statement, and the tiers must be kept separate. Component (B), $\mathrm{KL} = \tfrac12 ds^2$, is an exact canonical identity, not an analogy. Component (A), the $Z_I = 1/N$ incoherent / $Z_I = 1$ coherent dichotomy, follows from the law of large numbers applied to the barycenter under independence and a common precision — both of which the manuscript states at the point of use — with the trace-ratio limit holding unconditionally and the *spectral* reading carrying one regime assumption (uniform power over the Laplacian eigenbasis) that the manuscript names rather than derives. Component (C), the refusal to call $Z^{(s)}$ a dynamical critical exponent, is correct against the canonical definition of $z$ and is the most conservative move in the subsection. The subsection is a rigorous formalization of (B) and (C) and a correctly-hedged-but-not-fully-discharged formalization of (A); the only honest defect is that the spectral $1/N$ projection and the $\Lambda_I \approx \Lambda$ step are stated as regimes, not theorems, which the manuscript discloses but does not prove.

## Evidence

**(B) is the canonical second-order Fisher identity, exact for the stated step — not an analogy.** Amari–Nagaoka establish that KL is the second-order expansion of the Fisher metric, $\mathrm{KL} \approx \tfrac12(\Delta\theta)^\top g(\theta)(\Delta\theta) + O(\Delta\theta^3)$ [AmariNagaoka2000 Ch. 2; `external_canon_math.md:35`]. For a Gaussian belief, the closed-form Gaussian KL [`external_canon_math.md:42`; KingmaWelling2014 App. B] specializes, at equal covariance $\Sigma_q = \Sigma_p = \Sigma$ and a pure-mean step, to $\tfrac12\,\delta\mu^\top \Sigma^{-1}\delta\mu$, with the trace, log-det, and dimension terms cancelling exactly — the higher-order remainder in the Amari–Nagaoka expansion vanishes identically because the covariance sector does not move. This is not an infinitesimal approximation in the Gaussian mean sector; it is exact. Executed verification (K=3, random SPD $\Sigma$):

```
B: full KL  = 9.806446030768967e-07
B: 1/2 dmuLam dmu = 9.806446030965024e-07
B: exact equal-cov match? rel err = 1.9992601084508816e-11
```

The relative error of $2\times10^{-11}$ is round-off, not truncation. The manuscript's Eq. `kl_arclength` (`:2593`) writing $\mathrm{KL} = \tfrac12\delta\mu^\top\Lambda\delta\mu = \tfrac12 ds^2$ with $ds$ the Fisher-Rao line element is therefore the textbook specialization, and the consequence the manuscript draws — that one JND unit $ds=1$ is $\tfrac12$ bit, the two clocks differing by a square (`:2596`, `:2621`) — is the correct bookkeeping of a quadratic-versus-linear functional of the same Fisher length. A cluster slowing by $1/N$ in squared length slows by $1/\sqrt N$ in length; that is arithmetic on the identity, not a second postulate.

**(A) trace-ratio limits follow from the law of large numbers under independence and common precision, both stated at the point of use.** The barycenter $\mu_I = N^{-1}\sum_i \mu_i$ (Eq. `meta_agent_mu_impl`, `:2192`) gives $\Delta\mu_I = N^{-1}\sum_i \Delta\mu_i$. With the decomposition $\Delta\mu_i = c + \xi_i$, $\xi_i$ independent across constituents with common covariance $v_\xi$ (stated at `:2633`), the variance of the mean is the standard $\mathrm{Cov}(\Delta\mu_I) = \mathrm{Cov}(c) + v_\xi/N$ — the $1/N$ variance-of-the-mean reduction, which is the law of large numbers for the empirical mean. Executed Monte Carlo (N=50, K=4, $4\times10^5$ draws) matches to $3\times10^{-3}$ sampling error, and the trace ratio at zero coherence is the canonical $1/N$ independent of the structure of $v_\xi$ or the common $\Lambda$:

```
A: max abs diff emp vs Cc + v_xi/N = 0.003242447513849611
A incoherent Z_I (tr Lam Cov_I)/(tr Lam Cov_i) = 0.02  vs 1/N= 0.02
A coherent Z_I = 1.0
```

The trace-ratio limit $Z_I \to 1/N$ does not require isotropy: $\mathrm{tr}(\Lambda\, v_\xi/N)/\mathrm{tr}(\Lambda\, v_\xi) = 1/N$ for any common $\Lambda$ and any $v_\xi$, by linearity of the trace. The manuscript's Eq. `Z_decomposition` (`:2634`) and its two limits are this fact.

**(A) spectral realization is exact in the uniform-power regime the manuscript names.** For a connected weighted Laplacian $L$, the zero eigenvector is the constant direction $\mathbf 1/\sqrt N$ exactly [Fiedler1973; Chung 1997 §1], and the algebraic connectivity $\lambda_2$ sets the consensus relaxation rate, which the manuscript correctly cites at `:2654`. Executed (N=30, random symmetric weights):

```
smallest eigenvalue (should be ~0): -3.97e-15
zero-mode is constant 1/sqrt(N)? |v0| entries equal: True
Var(barycenter) = 0.03341  predicted 1/N = 0.03333
per-mode power E[a_k^2] ~ 1 (uniform): 1.0007 +- 0.0025
```

Under isotropic injection the barycenter variance is $1/N$ and per-mode power is uniform, so the zero mode holds one of $N$ equal shares and the projection weight is $1/N$. The manuscript states this as a regime — "incoherent injection, with power spread uniformly over all $N$ eigenmodes, projects onto this single mode with weight $1/N$" (`:2649`) — naming the uniform-power condition explicitly rather than claiming it follows from incoherence alone.

**(C) the refusal to call $Z^{(s)}$ a dynamical critical exponent is correct against the canonical definition.** The dynamic critical exponent $z$ is defined at a critical point through the divergence of the relaxation time with the correlation length, $\tau_{\text{char}} \sim \xi^{z}$, characterizing critical slowing down [Hohenberg & Halperin, Rev. Mod. Phys. 49, 435 (1977); the $\tau \sim \xi^z$ relation is the defining content of dynamic scaling]. The manuscript's $Z^{(s)}$ is a law-of-large-numbers ratio defined for every $N$, with no critical point invoked and no diverging correlation length; calling it a comoving-units choice $\hat\tau^{(s+1)} = \hat\tau^{(s)}/Z^{(s)}$ and deferring the $z$-identification to the open RG fixed point at `sec:meta_agent_rg` (`:2654`) is the conservative, canon-consistent reading. The KL/bit unit is dimensionless and reparametrization-invariant — a direct consequence of KL being a divergence with the Cencov sufficiency-invariance property [`external_canon_math.md:24`; Cencov1972] — so the bit cannot renormalize as a *unit*, and the slowdown must live in the production *rate*, which is exactly where the manuscript places it.

## Concessions stated up front

Four points are granted before they are attacked, because the claim survives all four.

The spectral "uniform power over all $N$ eigenmodes" (`:2649`) is a stated regime, not a derived consequence of incoherence. Incoherence (independence across constituents) does not by itself force the injection covariance to be isotropic in the Laplacian eigenbasis; the uniform-mode-power step additionally requires the noise covariance to commute with $L$ (the scalar isotropic case verified above). The manuscript names the regime; it does not prove that incoherence implies it. The trace-ratio $1/N$ (Tier-2 above) is unaffected, but the *spectral picture* of where that $1/N$ comes from carries this assumption.

$\Lambda_I \approx \Lambda$ (`:2640`) is exact only on the gauge-aligned locus and approximate otherwise. Under the averaging convention $\bar\Sigma_I = \sum_i w_i\,\Omega\Sigma_i\Omega^\top/\sum w_i$ (Eq. `meta_agent_sigma_impl`, `:2195`), common $\Sigma_i = \Sigma$ gives $\Lambda_I = \Lambda$ exactly only when $\Omega\Sigma\Omega^\top = \Sigma$ (aligned frames, or $\Omega$ orthogonal with respect to $\Sigma$). For general $\Omega \in \mathrm{GL}(K)$ this fails (verified: $\|\Omega\Sigma\Omega^\top - \Sigma\| = 44.5 \ne 0$), so the manuscript's "$\approx$" is the honest symbol.

The Laplacian reduction uses symmetrized weights $(\beta_{ij}+\beta_{ji})/2$ (`:2642`). The attention weights $\beta_{ij} = \mathrm{softmax}(\cdot)$ are generically asymmetric, so the quadratic form $\mu^\top(L\otimes\Lambda)\mu$ with a symmetric Laplacian is the symmetrized surrogate, which the manuscript flags rather than hides.

The entire construction is restricted to the Gaussian mean sector at equal covariance (`:2628`). The identification of $\mathcal I_{s\to s+1}$ with nonzero-mode energy (Eq. `Ilow_eq_dispersion`, `:2651`) is the quadratic-regime statement and is not claimed beyond it.

None of these four touches (B), (C), or the trace-ratio limit of (A). They bound the *spectral narrative* and the equal-covariance step, both of which the manuscript discloses.

## Falsification conditions

This position is wrong if any of the following holds.

1. **(B) falls** if the equal-covariance pure-mean Gaussian KL deviates from $\tfrac12\delta\mu^\top\Lambda\delta\mu$ by more than round-off — i.e. if the Amari–Nagaoka second-order expansion carries a non-vanishing higher-order term in the mean sector at fixed covariance. Direct expansion and the executed check (rel err $2\times10^{-11}$) say it does not; produce a step where it does and (B) is broken.

2. **(A) trace-ratio falls** if the manuscript's $\mathrm{Cov}(\Delta\mu_I) = \mathrm{Cov}(c) + v_\xi/N$ omits a *required* assumption that is *not* stated at `:2633`. The two it needs — independence of $\xi_i$ across constituents and a common $v_\xi$ — are both written there. Identify a third assumption that is load-bearing and absent, and (A) downgrades from theorem to illustration.

3. **(A) spectral falls** if line `:2649` claims the $1/N$ projection weight for incoherent injection *without* naming the uniform-power regime. It names it ("with power spread uniformly over all $N$ eigenmodes"). Show that the prose asserts $1/N$ as a consequence of incoherence alone, with no regime qualifier, and the spectral claim is overstated.

4. **(C) falls** if $Z^{(s)}$ admits a genuine $\tau \sim \xi^z$ dynamical-exponent reading at a critical point that the manuscript should have asserted but suppressed. The manuscript instead defers $z$ to the open RG fixed point [`sec:meta_agent_rg`] and matches the Hohenberg–Halperin definition by declining the identification. Exhibit a critical point and a diverging correlation length at which $Z^{(s)}$ scales as $\xi^z$, and the hedge becomes an evasion.

I cannot defend the subsection as a fully discharged derivation of the spectral $1/N$ origin under arbitrary incoherent noise, nor the exactness of $\Lambda_I = \Lambda$ off the gauge-aligned locus. The strongest honest defense is that (B) and (C) are rigorous against the canon, the (A) trace-ratio limits are a clean law-of-large-numbers consequence, and the two residual gaps are regimes the manuscript discloses rather than errors it commits.
