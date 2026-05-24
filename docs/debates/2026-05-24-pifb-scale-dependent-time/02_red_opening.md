# Red Opening — pifb-scale-dependent-time

## Steelman (opposing position)

Read charitably as a sequence of clearly-labeled speculative constructions, the subsection assembles three internally-consistent pieces: under common precision and mean-only injections the law of large numbers gives $Z_I \in [1/N, 1]$ as the meta-to-constituent bit-production ratio; the relation $\mathrm{KL} = \tfrac12 ds^2$ is the standard second-order Fisher expansion (canon_math.md:35, [AmariNagaoka2000 Ch. 2]) correctly specialized to the pure-mean Gaussian sector; and the renormalization paragraph commendably refuses to inflate $Z^{(s)}$ into a dynamical critical exponent, naming the RG fixed-point work as open.

## Position

The subsection is **not** a "rigorous formalization" as the claim asserts, because two of its load-bearing equations are stated as exact identities when they are only leading-order approximations under assumptions the manuscript does not satisfy in its own framework, and the renormalization verdict rests on a non-sequitur even though its conclusion lands in a defensible place. Specifically: (1) Eq. `Ilow_eq_dispersion` (2651) is written with an equality sign that is false under the framework's own non-trivial gauge transport $\Omega_{i,I}$; (2) the verdict at 2654 derives "the bit does not renormalize" from "the bit is reparametrization-invariant," conflating coordinate invariance on the belief manifold with RG invariance across scales.

## Evidence

**Vector 1 — the $\mathcal{I}_{s\to s+1}$ identity (Eq. `Ilow_eq_dispersion`, line 2651) is an overclaimed equality.**

The manuscript defines the cross-scale flow at line 2233 (Eq. `cross_scale_information_flow`) as a sum of *full transported Gaussian KLs*:
$$\mathcal{I}_{s\to s+1} = \sum_i \mathrm{KL}\big(q_i^{(s)} \,\big\|\, \Omega_{i,I}[q_I^{(s+1)}]\big),$$
with explicit gauge transport $\Omega_{i,I}$ and Gaussian beliefs carrying covariance. Line 2651 then sets this **equal** (not $\approx$) to the pure-mean Mahalanobis dispersion
$$\mathcal{I}_{s\to s+1} = \sum_i \tfrac12(\mu_i - \mu_I)^\top \Lambda (\mu_i - \mu_I),$$
and the prose at 2649 reinforces it: the dispersion is "*exactly* the energy these modes hold."

The closed-form Gaussian KL (canon_math.md:42, [KingmaWelling2014 App. B]) is
$$\mathrm{KL}(q\|p) = \tfrac12\big[\mathrm{tr}(\Sigma_p^{-1}\Sigma_q) + (\mu_p-\mu_q)^\top\Sigma_p^{-1}(\mu_p-\mu_q) - K + \log(|\Sigma_p|/|\Sigma_q|)\big].$$
For the transported meta-belief $\Omega_{i,I}[q_I] = \mathcal{N}(\Omega\mu_I,\, \Omega\Sigma_I\Omega^\top)$ this collapses to the pure-mean form $\tfrac12(\mu_i-\mu_I)^\top\Lambda(\mu_i-\mu_I)$ only when **both** $\Sigma_i = \Sigma_I$ **and** $\Omega_{i,I} = I$. The common-Σ stipulation (2628) buys the first. Nothing buys the second: the framework's transport is $\Omega_{i,I} = \exp(\phi_i)\exp(-\phi_I)$ (CLAUDE.md transport rule; canon_math.md:107–112), which is the identity only when $\phi_i = \phi_I$, i.e., only when the constituent and meta frames coincide — the degenerate flat-frame case the gauge construction exists to avoid.

Executed numerical check (common Σ enforced, $K=3$, generic $\Omega=\exp(B)\in\mathrm{GL}^+(3)$):
```
full transported Gaussian KL  = 1.65018
pure-mean dispersion (Eq.2651)= 0.17181
difference                    = 1.47837
Omega=I, common Sigma: full KL= 0.17181   dispersion= 0.17181   -> equal: True
```
The transport-induced terms (the $\mathrm{tr}(\Omega^{-\top}\Lambda\Omega^{-1}\Sigma) - K$ and $\log|\Omega\Sigma\Omega^\top|/|\Sigma|$ pieces of the full KL) dominate and survive the common-Σ assumption. The upstream Laplacian reduction at line 2642 honestly carries the qualifier "*to leading order in the post-transport mean difference*"; that qualifier must be inherited at 2651. The `=` should be `≈`, conditioned on $\Omega_{i,I}\approx I$. As written, the identity that "the injected information partitions cleanly" into zero-mode (meta-clock) plus nonzero-mode (dispersion) shares is false in the framework's own non-flat regime.

**Vector 2 — the renormalization verdict (line 2654) is a non-sequitur.**

The verdict reasons: "*Because the bit is reparametrization-invariant, it is a scale-invariant unit, and the slowdown cannot reside in a rescaling of the unit; it resides in the rate at which bits are produced.*" The premise (KL/bit is reparametrization-invariant) is the Cencov property: KL is invariant under smooth coordinate changes $\theta \to f(\theta)$ on the *belief parameter manifold* because the underlying distributions are unchanged (canon_math.md:24, [Cencov1972]; canon_math.md:60, the defining property of the natural gradient). That is invariance under *relabeling coordinates at a fixed scale*.

The conclusion is about behavior under *RG coarse-graining across scales* $s \to s+1$ — integrating out constituent (fast) modes to obtain a meta-agent (effective) theory. These are unrelated transformations. Wilsonian coarse-graining can endow same-named quantities with anomalous dimensions even when each scale's quantities are individually reparametrization-invariant; coordinate-invariance of an object at scale $s$ places no constraint on how that object maps to scale $s+1$ under the blocking transformation of Section `sec:meta_agent_rg`. The dynamical critical exponent $z$ is defined by $\tau \sim \xi^z$ at a critical point, governing critical slowing down as the correlation length diverges (dynamic-scaling hypothesis, [Hohenberg & Halperin, Rev. Mod. Phys. 49, 435 (1977)]; see also the dynamic-scaling relation $\tau \simeq \xi^z$). Whether $Z^{(s)}$ is or is not such an exponent is settled by the analytic-fixed-point analysis the manuscript itself flags as open at 2227 — not by the reparametrization-invariance of the bit. The conclusion at 2654 ("$Z^{(s)}$ is a choice of units, not a renormalization") may be defensible, but it does not follow from the stated premise; it is smuggled in. A "rigorous formalization" cannot rest a verdict on a premise that does not entail it.

**Vector 3 (minor flag, not load-bearing) — the Laplacian symmetrization.** The reduction at 2642–2644 requires the symmetrized weights $(\beta_{ij}+\beta_{ji})/2$ to obtain a symmetric $L$ with a real spectrum and the zero-mode $\mathbf{1}/\sqrt N$. The manuscript states the symmetrization openly, so this is honest, but softmax $\beta_{ij}$ is generically asymmetric and the consensus dynamics $\dot\mu = -2\eta_\mu(L\otimes I)\mu$ driven by the *un*symmetrized $L$ need not have $\mathbf{1}$ as a left eigenvector with the clean projection geometry the $Z_I=1/N$ argument reads off. This weakens "rigorous" but does not by itself falsify; I raise it as a flag.

**What I concede up front (do not attack):** The $Z_I = 1/N$ trace-ratio at 2633–2636 follows from $\xi_i$-independence alone — $\mathrm{Cov}(\Delta\mu_I) = v_\xi/N$ holds for arbitrary $v_\xi$, so the ratio needs no isotropy in the $K$-dimensional belief space, and the "uniform power over $N$ eigenmodes" phrasing refers correctly to the trivially-isotropic $I_N$ in the constituent-index space. Eq. `kl_arclength` (2593) is the correct pure-mean specialization of the Fisher second-order expansion (canon_math.md:35) and the manuscript explicitly conditions it on "a pure mean step at fixed covariance" — component (B) of the claim is sound as written. These pieces I do not contest.

## Falsification conditions

My position is wrong if any of the following holds.

1. **Vector 1 fails** if the framework guarantees $\Omega_{i,I} = I$ at the scale-transition where $\mathcal{I}_{s\to s+1}$ is evaluated (e.g., a gauge-fixing convention that aligns constituent and meta frames before the transported KL is taken). If transport is identity at that step, the full Gaussian KL reduces to the pure-mean dispersion under common-Σ and Eq. `Ilow_eq_dispersion` is an exact identity, not an approximation. Blue must exhibit that convention in the manuscript or the framework; absent it, the `=` is false and the numerical check above stands.

2. **Vector 2 fails** if "reparametrization-invariant" in the manuscript means invariance under the *RG coarse-graining map itself* (not coordinate relabeling on the belief manifold), in which case the premise would directly entail the conclusion. The text at 2596 grounds the invariance in "reparametrization of the belief coordinates" — coordinate relabeling — which is the Cencov sense, not the RG sense. Blue must show the manuscript uses the RG sense; the cited line uses the coordinate sense.

3. The whole position fails if the claim is read so weakly that "rigorous formalization" tolerates approximations-stated-as-identities and verdicts-from-non-entailing-premises, provided the conclusions happen to be defensible. Under the methodology's distinction between a labeled approximation and a claimed-exact result (canon_math.md:144–151), I read "rigorous formalization" to forbid exactly this slippage; if the judge reads it more permissively, the claim survives on its conclusions while failing on its derivations.
