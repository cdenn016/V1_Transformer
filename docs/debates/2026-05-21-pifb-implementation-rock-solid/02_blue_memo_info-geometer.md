# Memo — debate-expert-info-geometer — blue — opening — pifb-implementation-rock-solid

## Lens

Information geometry — Fisher metric, natural gradient, dual affine connections, additive-KL ↔ product-of-experts correspondence, log-linear opinion pooling, dynamic discounting, generalized-Bayesian / tempered posteriors.

## Steelman of the opposing position

A section that identifies the Ouroboros free-energy fragment at Eq. 2270 with four separate canonical objects — West-Harrison 1997 dynamic discount, Hinton 2002 product-of-experts, Genest-Zidek 1986 log-linear opinion pool, Bissiri-Holmes-Walker 2016 tempered Bayes — in a single paragraph at line 2275 must verify that each citation is the right paper for the specific claim made; a "natural identification" without explicit derivation of the correspondence is hand-wave-with-citation, the canon-cop's wrong-domain failure mode.

## My position (in service of blue)

Each of the four pooling-anchor identifications at line 2275 is mathematically natural and verifiable under the standard additive-KL ↔ product-of-experts / log-linear-pool correspondence that appears in any modern variational-inference textbook. The correspondence is the following: for distributions $p_k$ and additive cost $\sum_k \lambda_k \mathrm{KL}(q \| p_k)$, the minimizer over $q$ is the geometric mean $q \propto \prod_k p_k^{\lambda_k}$, which is exactly the product-of-experts / log-linear pool form. For Gaussian $p_k = \mathcal{N}(\mu_k, \Sigma_k)$ this gives precision-additive pooling $\Sigma_\star^{-1} = \sum_k \lambda_k \Sigma_k^{-1}$ and weighted-precision mean $\mu_\star = \Sigma_\star \sum_k \lambda_k \Sigma_k^{-1} \mu_k$ — both stated correctly at line 2275. The geometric per-generation weighting $\lambda_k = \lambda_0 \rho^k$ is the standard dynamic-discount form of West-Harrison 1997, and the unrestricted ($\sum_k \lambda_k \neq 1$) version is the tempered free energy in the generalized-Bayesian sense of Bissiri-Holmes-Walker 2016. All four anchors are correctly cited.

## Evidence

- **Additive-KL ↔ product-of-experts correspondence**: For $q \in$ exponential family, $\arg\min_q \sum_k \lambda_k \mathrm{KL}(q \| p_k) = \arg\min_q \mathrm{KL}(q \| \prod_k p_k^{\lambda_k})$ (up to normalization); this is the standard variational mean computation. See [BleiKuckelbirgJordan2017 §3] and [JordanGhahramaniJaakkolaSaul1999 §3.1]. For Gaussian $p_k$, the pooled distribution is itself Gaussian with $\Sigma_\star^{-1} = \sum_k \lambda_k \Sigma_k^{-1}$ — exactly the manuscript's statement at line 2275.
- **[Hinton 2002] "Training products of experts by minimizing contrastive divergence," Neural Computation 14: 1771-1800, §2**: the PoE form $p(x) \propto \prod_k p_k(x)^{\lambda_k}$ is the standard exponential-family pool. The identification at line 2275 is the textbook correspondence; Hinton's PoE is the canonical reference.
- **[Genest & Zidek 1986] "Combining probability distributions: A critique and an annotated bibliography," Statistical Science 1(1): 114-135, §3.2**: the log-linear opinion pool $f(x) \propto \prod_k f_k(x)^{\alpha_k}$ with $\sum_k \alpha_k = 1$ is "externally Bayesian" — Bayes-update commutes with pooling. The manuscript at line 2275 explicitly notes that "the strictly-normalized special case $\lambda_0 = 1 - \rho$ enforces $\sum_{k \ge 0}\lambda_k = 1$ and recovers the externally-Bayesian log-linear pool of [GenestZidek1986]." This is the correct identification of the normalized case; the unrestricted case is correctly distinguished from external Bayesianity.
- **[West & Harrison 1997] *Bayesian Forecasting and Dynamic Models* (2nd ed.), §6.3 (discount factors)**: the multiplicative discount factor $\delta \in (0, 1]$ controls how rapidly historical observations lose influence on the present posterior. The geometric-decay form $\lambda_k = \lambda_0 \rho^k$ with $\rho < 1$ across hierarchical scale-distance $k$ is the same per-step discount; the manuscript at line 2275 correctly identifies "the role of historical time is played here by the hierarchical scale-distance $k$."
- **[Bissiri-Holmes-Walker 2016] "A general framework for updating belief distributions," JRSSB 78(5): 1103-1130, §2**: generalized Bayesian update $\pi(\theta | x) \propto \pi(\theta) \exp(-w \cdot \ell(\theta; x))$ with $w > 0$ a learning rate (not constrained to $w = 1$); when $w \neq 1$ this is the tempered posterior. The manuscript's unrestricted $\lambda_0, \rho > 0$ default at line 2275 is exactly the tempered case — the identification is correct.

## Newly-discovered canon (for 01b_extended_evidence.md)

- **Amari, "Integration of stochastic models by minimizing $\alpha$-divergence," *Neural Computation* 19(10) (2007): 2780-2796.** Treats the $\alpha$-divergence-weighted barycenter as a unified family containing the product-of-experts ($\alpha = 1$, m-flat) and the mixture ($\alpha = -1$, e-flat) as special cases. The forward-KL barycenter at Eq. 2141 of the manuscript is the $\alpha = -1$ (forward KL = m-divergence) case; the dual mixture interpretation is the natural variational counterpart.
- **Neal & Hinton, "A view of the EM algorithm that justifies incremental, sparse, and other variants," in *Learning in Graphical Models* (1998): 355-368.** The incremental-EM framework justifies the fixed-point iteration the manuscript invokes at line 2187 ("solved by fixed-point iteration"), and is the standard reference for partial E-steps under variational EM.
- **Genest & McConway, "Allocating the weights in the linear opinion pool," *J. Forecasting* 9(1) (1990): 53-73.** Treats the choice of weights in pooling explicitly; the manuscript's coherence-based weight $w_i \propto \chi_i \exp(-\mathrm{KL})$ at line 2187 is a coherence-weighted pool, which is a standard data-driven weight choice in the Genest-McConway framework.
- **Heskes, "Selecting weighting factors in logarithmic opinion pools," *NIPS* 10 (1998): 266-272.** Discusses the optimality of unequal-weight log-linear pools; the manuscript's geometric-decay $\lambda_k = \lambda_0 \rho^k$ default is consistent with the Heskes treatment of unequal weights.

## Falsification conditions

This info-geometric defense is wrong if:

1. The closed-form pooled-Gaussian precision $\Sigma_\star^{-1} = \sum_k \lambda_k \Sigma_k^{-1}$ at line 2275 is incorrect for the additive-discounted-KL form (verify by setting $\partial_{q}[\sum_k \lambda_k \mathrm{KL}(q \| p_k)] = 0$ for Gaussian $q, p_k$ and reading off precision form).
2. [West-Harrison 1997] is actually about a different discount mechanism (e.g., observation-noise inflation rather than precision-weight decay) that does not correspond to the geometric-decay $\lambda_k = \lambda_0 \rho^k$.
3. [Bissiri-Holmes-Walker 2016] requires a specific decision-theoretic loss function (the "loss-based likelihood" framing) that the gauge-transported KL terms do not satisfy.
4. The "geometric mean of Gaussians is Gaussian" identity has a normalization constraint the manuscript at line 2275 silently violates with the unrestricted $\lambda_0, \rho > 0$ default.

I verified (1): for Gaussian $q = \mathcal{N}(\mu, \Sigma)$ and $p_k = \mathcal{N}(\mu_k, \Sigma_k)$,
$$\mathrm{KL}(q \| p_k) = \tfrac{1}{2}[\mathrm{tr}(\Sigma_k^{-1}\Sigma) + (\mu - \mu_k)^\top \Sigma_k^{-1}(\mu - \mu_k) - K + \log(|\Sigma_k|/|\Sigma|)]$$
and $\partial_{\Sigma^{-1}}[\sum_k \lambda_k \mathrm{KL}(q \| p_k)] = 0$ gives $\Sigma_\star^{-1} = \sum_k \lambda_k \Sigma_k^{-1}$. The closed form at line 2275 is correct.

For (4): the unrestricted form is *not* normalizing $q$ to be a probability distribution at the variational stage — it is defining a Gaussian via additive-precision pooling, which is well-defined for any $\lambda_k > 0$ without sum constraint. The manuscript at line 2275 correctly states "well-defined for any $\lambda_k > 0$ without a normalization constraint on the weights."

## Confidence

HIGH on the four pooling-anchor citations being correctly applied under standard additive-KL ↔ product-of-experts correspondence; HIGH on the closed-form pooled Gaussian at line 2275 being mathematically correct; MEDIUM on the dynamic-discount identification with West-Harrison (textbook reference — verifiable against book §6.3 but I have not retrieved the book).
