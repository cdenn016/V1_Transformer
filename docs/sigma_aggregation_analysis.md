# Covariance Aggregation in Attention: Mixture vs. Precision

## The Two Formulas

The attention sublayer in the gauge-theoretic transformer aggregates messages from neighboring agents, each transported into a common frame via the gauge connection $\Omega_{ij} = \exp(\phi_i)\exp(-\phi_j) \in \mathrm{GL}^+(K)$. For agent $i$, the transported belief from neighbor $j$ is the Gaussian $\mathcal{N}(\Omega_{ij}\mu_j,\; \Omega_{ij}\Sigma_j\Omega_{ij}^\top)$, weighted by attention coefficient $\beta_{ij} = \mathrm{softmax}(-D_{\mathrm{KL}}(q_i \| \Omega_{ij}q_j)/\tau)$.

The mean aggregation is unambiguous:

$$m_i = \sum_j \beta_{ij}\,\Omega_{ij}\mu_j$$

This is the standard mixture-of-Gaussians expectation, consistent with the mixture-of-sources generative model (GL(K)\_attention.tex, Section 5.2.1). The question is what happens to the covariance.

**Mixture covariance** (moment matching). The law of total variance decomposes the covariance of the mixture distribution into two terms:

$$\Sigma_i^{\mathrm{mix}} = \underbrace{\sum_j \beta_{ij}\,\Sigma_j^{\to i}}_{\text{expected within-component variance}} + \underbrace{\sum_j \beta_{ij}\,\mu_j^{\to i}(\mu_j^{\to i})^\top - m_i m_i^\top}_{\text{between-component variance}}$$

where $\Sigma_j^{\to i} = \Omega_{ij}\Sigma_j\Omega_{ij}^\top$ and $\mu_j^{\to i} = \Omega_{ij}\mu_j$. The first term is the $\beta$-weighted average of transported covariances. The second is the covariance of the transported means under the mixing distribution $\beta$. Equivalently, writing $\mathrm{E}_Z[\cdot]$ for expectation over the categorical source variable $z$ with probabilities $\beta_{ij}$:

$$\mathrm{Cov}(X) = \mathrm{E}_Z[\mathrm{Cov}(X|Z)] + \mathrm{Cov}_Z(\mathrm{E}[X|Z])$$

**Precision aggregation** (VFE fixed point). Setting $\partial\mathcal{F}_i/\partial\Sigma_i = 0$ in the free energy gradient (GL(K)\_supplementary.tex, Appendix B, eq. B.6) yields the equilibrium condition:

$$(\Sigma_i^{\mathrm{eq}})^{-1} = \frac{1}{2}\left[\Sigma_{p,i}^{-1} + \sum_j \beta_{ij}\,(\Sigma_j^{\to i})^{-1}\right]$$

Dropping the prior term (which the E-step handles separately), the attention sublayer's contribution is:

$$(\Sigma_i^{\mathrm{prec}})^{-1} = \sum_j \beta_{ij}\,(\Sigma_j^{\to i})^{-1}$$

This is the $\beta$-weighted harmonic mean of the transported covariances.

## Why They Compute Different Things

The two formulas answer different questions, and the distinction is not merely numerical but conceptual.

The mixture covariance answers: "If I draw a source $j$ with probability $\beta_{ij}$ and then draw $x$ from that source's transported belief, what is the marginal covariance of $x$?" This is the standard posterior covariance of the mixture-of-sources generative model that underlies the attention mechanism (GL(K)\_attention.tex, Section 5.2.1, eq. 23). It is the Bayesian answer for a single observation step.

The precision aggregation answers: "What covariance makes the VFE gradient vanish?" This is the fixed point of an iterative process. It describes what $\Sigma_i$ should converge to after repeated E-step updates under the free energy $\mathcal{F}_i = D_{\mathrm{KL}}(q_i \| p_i) + \sum_j \beta_{ij}D_{\mathrm{KL}}(q_i \| \Omega_{ij}q_j)$, assuming $\beta_{ij}$ and neighbor beliefs are held constant.

The formulas coincide only in degenerate cases. When all transported means agree ($\mu_j^{\to i} = m_i$ for all $j$), the between-component variance vanishes, and the mixture covariance reduces to the weighted average of transported covariances $\sum_j \beta_{ij}\Sigma_j^{\to i}$. Even then, this differs from the precision aggregation, which computes the harmonic mean $(\sum_j \beta_{ij}(\Sigma_j^{\to i})^{-1})^{-1}$. The arithmetic mean of covariances is always at least as large (in the Loewner order) as the harmonic mean of covariances, with equality only when all $\Sigma_j^{\to i}$ are identical.

## The Var[E] Term

The between-component variance $\mathrm{Cov}_Z(\mathrm{E}[X|Z])$ captures an important piece of information that the precision formula discards entirely: the disagreement among neighbors about where the mean should be.

Consider $N$ agents whose transported means $\mu_j^{\to i}$ differ but whose transported covariances $\Sigma_j^{\to i}$ are identical. The precision aggregation returns a covariance equal to $\Sigma_j^{\to i}$ (the harmonic mean of identical matrices is the matrix itself). The mixture covariance returns $\Sigma_j^{\to i} + \mathrm{Cov}_\beta(\mu_j^{\to i})$, which is strictly larger. The additional term reflects genuine posterior uncertainty: when neighbors disagree about the mean, the aggregated belief should be wider to account for the ambiguity, not as narrow as if a single neighbor had spoken.

This difference is largest when the mixture is multimodal or widely spread. In the gauge transformer, wild GL$(K)$ transport operators early in training can produce transported means that span a large region of belief space. The mixture covariance correctly inflates the posterior to reflect this uncertainty. The precision formula, blind to mean disagreement, produces an unjustifiably tight covariance.

The between-component variance also maintains gradient information. In the mixture formula, $\mathrm{Cov}_\beta(\mu^{\to i})$ depends on $\beta_{ij}$ and on the transported means, both of which carry gradients with respect to the gauge frames $\phi$ and belief parameters. The precision formula, involving only $(\Sigma_j^{\to i})^{-1}$, couples the covariance output exclusively to the neighbor covariances. This impoverishes the gradient signal available for learning the gauge connection.

## Why Mixture Is Correct for Message Passing

The `aggregate_messages()` function computes a single message-passing update within the attention sublayer. The E-step in the subsequent FFN block then iterates beliefs toward the VFE equilibrium. These are two distinct operations on two different timescales.

The attention sublayer computes the posterior of agent $i$ under the mixture-of-sources generative model: agent $i$ assumes its state was drawn from one of $N$ neighbors, selected by $z$ with probabilities $\beta_{ij}$. The posterior mean is $m_i = \sum_j \beta_{ij}\mu_j^{\to i}$ and the posterior covariance is $\Sigma_i^{\mathrm{mix}}$. This is a single Bayesian update, not an iterative fixed-point computation.

The E-step then takes $\Sigma_i^{\mathrm{mix}}$ as an initialization and refines it via natural gradient descent on the VFE. Over multiple iterations, $\Sigma_i$ moves toward $\Sigma_i^{\mathrm{eq}}$. The mixture covariance provides a warm start that is conservative (wider than the equilibrium), letting the E-step tighten beliefs as appropriate. The precision formula preempts the E-step by jumping directly to the equilibrium, removing the iterative refinement that the VFE loop is designed to perform.

Empirically, this distinction matters. Training with the precision formula as the attention output produced worse results than the mixture formula, consistent with the analysis above: the precision formula is too aggressive as a single-step initialization, collapsing uncertainty prematurely before the E-step has had a chance to incorporate the prior and observation terms.

## Numerical Stability

The two formulas have qualitatively different numerical behavior.

The mixture covariance computes $\mathrm{E}[XX^\top] - \mathrm{E}[X]\mathrm{E}[X]^\top$, a subtraction of two positive semidefinite matrices. In exact arithmetic this is always PSD (by the law of total variance). In finite precision, catastrophic cancellation can occur when the second-moment matrix $\mathrm{E}[XX^\top]$ is much larger than the result $\Sigma_i^{\mathrm{mix}}$. This happens when transported means are large relative to transported covariances. The standard mitigation is a spectral floor: symmetrize the result and clamp eigenvalues to a minimum of $\epsilon = 10^{-4}$, which is sufficient in practice.

The precision aggregation computes $\sum_j \beta_{ij}(\Sigma_j^{\to i})^{-1}$, a sum of positive definite matrices. The sum itself is numerically stable and guaranteed PD. However, the formula requires inverting each transported covariance and then inverting the accumulated precision. With $\mathrm{GL}(K)$ transport, $\Sigma_j^{\to i} = \Omega_{ij}\Sigma_j\Omega_{ij}^\top$ can have a very large condition number (the condition number of the product is bounded by $\kappa(\Omega_{ij})^2\kappa(\Sigma_j)$). Inverting an ill-conditioned matrix amplifies errors: if $\kappa(\Sigma_j^{\to i}) \sim 10^6$, the inverse has relative error of order $10^6\epsilon_{\mathrm{mach}}$. The double inversion (transported covariance → precision → back to covariance) compounds this.

In summary: the mixture formula's weakness is a single subtraction that can be cheaply guarded by spectral clamping. The precision formula's weakness is repeated matrix inversion of potentially ill-conditioned transported covariances, which is harder to stabilize without distorting the result.

## When to Use Each Formula

| Property | Mixture | Precision |
|---|---|---|
| Semantic role | Single-step Bayesian posterior | VFE equilibrium target |
| Width | Conservative (wider) | Aggressive (tighter) |
| Var[E] term | Included | Discarded |
| Gradient through means | Yes (via $\mu_j^{\to i}$) | No |
| Numerical risk | Cancellation (guardable) | Ill-conditioned inversion |
| Appropriate when | Message passing, initialization | Verifying convergence |

**Use `'mixture'` (default)** for the attention sublayer's covariance aggregation. This is the theoretically correct posterior for the mixture-of-sources generative model, provides a conservative initialization for the subsequent E-step, includes the between-component variance that encodes neighbor disagreement, and is cheaper to guard numerically.

**Use `'precision'`** to verify that the E-step is converging correctly. After sufficient E-step iterations, the covariance should approach the precision-aggregation fixed point (modulo the prior term). The precision formula can also serve as a diagnostic: if the E-step output differs from the precision target by more than expected, the VFE loop may need more iterations or a different learning rate schedule.

The config toggle `sigma_aggregation` in `BlockConfig` selects between the two:

```python
'sigma_aggregation': 'mixture',    # default: moment-matching posterior
'sigma_aggregation': 'precision',  # VFE equilibrium fixed point
```

## References

1. GL(K)\_attention.tex, Section 5.2.1: Mixture-of-sources generative model and value aggregation.
2. GL(K)\_supplementary.tex, Appendix B (eq. B.6): Covariance gradient and fixed-point equation $\Sigma_i^{-1} = \frac{1}{2}[\Sigma_{p,i}^{-1} + \sum_j \beta_{ij}(\Omega_{ij}\Sigma_j\Omega_{ij}^\top)^{-1}]$.
3. GL(K)\_supplementary.tex, Section B.2: Homogeneous limit $\Sigma_\infty = \Sigma_0$ and alignment-dominated regime.
