# Memo — debate-expert-variational — blue — opening — pifb-implementation-rock-solid

## Lens

Variational inference and FEP — ELBO, EM separation, mean-field factorization, forward-KL barycenter, variational vs IFT, information bottleneck (Tishby), Gaussian-IB closed form (Chechik-Tishby).

## Steelman of the opposing position

The IB Lagrangian at Eq. 2133 is presented with a complete mathematical apparatus (the Tishby form $\mathcal{L} = I(T;X) - \beta_{\mathrm{IB}} I(T;Y)$, the Chechik-Tishby Gaussian-IB closed form claimed at line 2138, the interpretation of $C(I)$ in Eq. 2123 as the IB rate term) yet the manuscript at line 2138 lists three concrete ingredients still unsupplied (natural-gradient transition kernel $X \to Y$ in the gauge-covariant setting; gauge-frame component of $T$ under encoder noise; empirical comparison of IB-optimal coarse-graining with the threshold-detector hierarchy), which means the IB Lagrangian appears in §Implementation as a paragraph of unconsummated mathematical machinery in a section reviewers expect to describe what was implemented.

## My position (in service of blue)

The IB framing at lines 2131-2138 is explicitly bracketed as "Information-bottleneck refinement (research direction)" in the paragraph header at line 2131; the closing sentence at line 2138 ("the present manuscript stands on the FE-improvement criterion") explicitly declines to base the existing construction on IB. The variational FE-improvement criterion at Eq. 2123-2125 and the forward-KL barycenter at Eq. 2141-2152 are the actual variational content of the section, and both are mathematically standard. The forward-KL barycenter $q_I^* = \arg\min_{q_I} \sum_i w_i^I \mathrm{KL}(\Omega_{Ii}q_i \| q_I) + \lambda \mathrm{KL}(q_I \| p_I)$ is the standard variational mean-field problem with prior regularization [Beal 2003 §2.2.2], and its Gaussian closed form at Eq. 2147-2152 is the standard moment-matched solution [BleiKuckelbirgJordan2017 §2.4]. The dispersion term that is dropped to obtain Eq. 2184 is explicitly labeled "leading-order approximation in the high-coherence regime" at line 2179, with the dropped term being $\mathcal{O}(\varepsilon)$ in the post-transport pairwise dispersion $\varepsilon$ — this is a textbook saddle-point approximation argument, not a hidden simplification.

The cross-scale information flow at Eq. 2219 is explicitly disclaimed as not being a mutual information at line 2222 ("we avoid writing this quantity as a mutual information, since mutual information requires a joint distribution on $(k_i, k_I)$ rather than a pair of marginals") — exactly the variational-inference standard that mutual information requires a joint, while transported-KL between two marginals is a divergence between distributions on the same fiber. This is reviewer-grade epistemic discipline.

## Evidence

- **[Friston2010] Eq. 2.2 and the canonical accuracy-plus-complexity decomposition**: $\mathcal{F} = \mathrm{E}_q[-\log p(o|s)] + \mathrm{KL}(q(s) \| p(s))$. The FE-improvement criterion at Eq. 2123-2125 of §Implementation reduces to the Friston form when summed over constituents and the parent: the pairwise alignment terms collapse into child-parent KL terms exactly as line 2129 derives. The variational principle is standard.
- **[Tishby-Pereira-Bialek 1999] "The information bottleneck method," Allerton**: $\min_{p(T|X)} I(T;X) - \beta I(T;Y)$. Sign convention is rate minus $\beta$ times predictive information, matching the manuscript at line 2133. The Tishby form is the canonical one; the manuscript cites correctly.
- **[Chechik-Globerson-Tishby-Weiss 2005] "Information bottleneck for Gaussian variables," JMLR 6: 165-188, Theorem 3.1**: for jointly Gaussian $(X, Y)$ with covariances $\Sigma_x, \Sigma_y, \Sigma_{xy}$, the IB-optimal $T = AX + \xi$ has projection matrix $A$ aligned with the top canonical-correlation eigenvectors of $\Sigma_x^{-1/2}\Sigma_{xy}\Sigma_y^{-1}\Sigma_{yx}\Sigma_x^{-1/2}$. The manuscript at line 2138 states "the optimal $T$ is a precision-weighted projection along the top canonical-correlation directions between $X$ and $Y$" — exactly the Chechik-Tishby form. The citation is correct.
- **[Beal 2003] §2.2.2 and [Bishop 2006] §10.1**: the standard variational mean-field minimization of $\mathrm{KL}(q \| p)$ over a tractable family yields a fixed-point iteration $q_i \propto \exp(\mathrm{E}_{q_{-i}}[\log p])$; for Gaussian families with prior regularization this gives the closed-form barycenter at Eq. 2147-2152. The dispersion term in $\Sigma_I^*$ (Eq. 2150-2152) is the law-of-total-variance expression for the moment-matched mixture covariance; dropping it under the high-coherence regime is the standard leading-order saddle-point approximation.
- **[Cover & Thomas 2006] §2.3**: mutual information $I(X; Y) = \sum p(x, y) \log [p(x,y)/p(x)p(y)]$ requires a joint distribution. The transported-KL at Eq. 2219 is between two marginals $q_i^{(s)}$ and $\Omega_{i,I}[q_I^{(s+1)}]$ on the same fiber — this is a divergence, not a mutual information, and the manuscript at line 2222 disclaims the confusion explicitly.

## Newly-discovered canon (for 01b_extended_evidence.md)

- **Slonim & Tishby, "Document clustering using word clusters via the information bottleneck method," *SIGIR* (2000): 208-215.** Provides the agglomerative IB algorithm, which is the canonical clustering procedure most closely resembling the manuscript's threshold-detector hierarchy; the IB-optimal coarse-graining the manuscript at line 2138 says it does *not* settle against the threshold detector has an established canonical comparison procedure available.
- **Achille & Soatto, "Information dropout: Learning optimal representations through noisy computation," *IEEE TPAMI* 40(12) (2018): 2897-2905.** Derives the IB Lagrangian as a variational bound on representation quality with the same sign convention as the manuscript at line 2133. Independent corroboration that the Tishby sign convention is standard.
- **[Beal 2003] §3.3 (Variational EM as coordinate ascent on F)**: the alternating maximization of $\mathcal{F}$ with respect to $q$ (E-step) and the model parameters (M-step) is the canonical variational EM procedure; the manuscript's FE-improvement criterion is consistent with this framework.

## Falsification conditions

This variational defense is wrong if:

1. The forward-KL barycenter closed form at Eq. 2147-2152 has an error in the dispersion term derivation — verify by direct expansion of $\sum_i w_i^I \mathrm{KL}(\Omega_{Ii}q_i \| q_I)$ for Gaussian $q_i, q_I$.
2. The IB Lagrangian sign convention at Eq. 2133 disagrees with [Tishby 1999] — verify against the original paper.
3. The leading-order-approximation argument for dropping the dispersion term at line 2179 is not actually $\mathcal{O}(\varepsilon)$ but a higher-order error in the high-coherence regime.
4. The transported-KL at Eq. 2219 is later in §Implementation referred to as a mutual information, undoing the disclaimer at line 2222.

I verified (4) is not violated in §Implementation lines 2101-2304 (the section does not re-label Eq. 2219 as mutual information anywhere after line 2222). (1), (2), (3) require symbolic verification I have not run; the manuscript's claims are textbook-consistent under standard derivations, but a sympy-driven verification would close the loop.

## Confidence

HIGH on the IB framing being correctly disclaimed as a research direction (lines 2131-2138); HIGH on the forward-KL barycenter being the standard variational object [Beal 2003, BleiKuckelbirgJordan2017]; HIGH on the cross-scale information flow disclaimer at line 2222 being reviewer-grade. MEDIUM on the closed-form Eq. 2147-2152 absent symbolic verification, but the form matches the moment-matched mixture covariance standard.
