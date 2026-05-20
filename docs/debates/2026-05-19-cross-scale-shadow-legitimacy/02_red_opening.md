# Red Opening — cross-scale-shadow-legitimacy

## Steelman (opposing position)

The cross-scale shadow $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ is a gauge-equivariant amortized realization of the standard hierarchical-VI prior: the level-$(s{+}1)$ posterior plays the role that the next-level latent plays in ladder VAEs and the role that the data-driven hyperparameter plays in empirical Bayes, with the gauge transport $\Omega_{i,I}$ supplying the frame-change that makes the construction agent-relativistic; the manuscript discloses the structural commitment at line 546, retains the FEP citations only as a conceptual backbone (line 1023), and explicitly identifies the multi-ancestor extension with West–Harrison discount priors, log-linear opinion pools, and product-of-experts at line 2216, so the construction is a labeled refinement that inherits its legitimacy from a standard chain of hierarchical-Bayesian objects.

## Position

The cross-scale shadow at `Participatory_it_from_bit.tex:540` (Eq.~\ref{eq:cross_scale_shadow}) is an undeclared model substitution rather than a legitimate refinement of standard hierarchical variational inference, on four independent and individually sufficient grounds:

1. It replaces the standard generative-model conditional $p(s_\ell \mid s_{\ell+1})$ of Friston 2017 / Parr–Pezzulo–Friston 2022 with the variational posterior $q_I^{(s+1)}$, which is a different mathematical object class — a conditional density vs an optimized variational distribution — and the manuscript itself declines to display the reduction (line 546).
2. The proposed precedents (empirical Bayes, ladder VAEs, log-linear pools) each fail as legitimating analogues at the relevant point of contact: EB does not replace a conditional with the posterior of a *coupled subsystem* whose own posterior depends on $p$; ladder VAEs place a deterministic-function-of-the-next-level-latent into a *Gaussian conditional density*, not into an identity-with-transport between two posteriors; PoE / Genest–Zidek combine multiple given priors into one pooled prior, they do not legitimize the substitution of a posterior for a prior.
3. The resulting object is not a proper joint generative model: with $p_i^{(s)}$ a function of the variational parameters of $q_I^{(s+1)}$, and $q_I^{(s+1)}$ in turn a variational quantity optimized against data through the lower scales, the "prior" and "posterior" are no longer the two distinct components of a Bayesian model — they are mutually defined fixed-point variables of a coupled variational system. This collapses the prior/posterior distinction that hierarchical VI ([Wainwright–Jordan 2008 §3.4]) presupposes.
4. The manuscript's own line-546 admission ("structural commitment of the framework rather than a theorem of standard hierarchical variational inference … we do not display the reduction (or approximation)") is dispositive under the user's literature-as-source-of-truth standard: a construction whose author explicitly declines to derive it from the literature it is being called a refinement of is, by that author's own labeling, not in continuity with that literature.

## Evidence

**C1 — Friston 2017 / Parr–Pezzulo–Friston 2022 use a generative-model conditional, not a transported posterior.**

[Friston et al. 2017, "Active Inference: A Process Theory," *Neural Comput.* 29:1–49, §"Generative models and (deep) temporal models"] writes the hierarchical generative density as a product of *conditional* factors
$$p(\tilde o, \tilde s) = p(s_L)\prod_\ell p(s_\ell \mid s_{\ell+1})\,p(o \mid s_1),$$
where each $p(s_\ell \mid s_{\ell+1})$ is a fixed conditional probability density of the generative model and the posterior $q(s_\ell)$ is an *approximation to the marginal of the true posterior* under that fixed generative model. The paper is explicit that mean-field updates take the form $q(s_\ell) \propto \exp\!\big(\mathbb{E}_{q(s_{\ell+1})}[\log p(s_\ell \mid s_{\ell+1})] + \cdots\big)$: the prior contribution to the level-$\ell$ posterior is an expectation of the *log-conditional* under the next-level posterior, not the next-level posterior itself.

[Parr, Pezzulo, Friston 2022, *Active Inference*, MIT Press, Ch. 8 "Deep generative models"] uses the same factorization: hierarchical generative models are products of conditional likelihoods, with the level-$\ell$ prior implemented as $p(s_\ell^\tau \mid s_{\ell+1}^\tau)$ — a fixed conditional density parameter object, not the optimized $q$.

The manuscript at `Participatory_it_from_bit.tex:540–545` writes
$$p_i^{(s)}(c) = \Omega_{i,I}[q_I^{(s+1)}](c).$$
This sets the level-$s$ prior equal to the (transported) level-$(s{+}1)$ *variational posterior* $q_I^{(s+1)}$. The two objects are not the same class: a fixed conditional density of the generative model on one side, a data-optimized variational posterior on the other. Whatever the construction is, it is not the construction in Friston 2017 §"Generative models and deep temporal models" or Parr–Pezzulo–Friston 2022 Ch. 8.

**C2 — Empirical Bayes does not legitimate the substitution.**

[Carlin & Louis, *Bayesian Methods for Data Analysis*, 3rd ed., Ch. 5 §5.4–5.5] develops empirical Bayes as an *approximation* to a fully hierarchical model in which one integrates out the hyperprior or fixes hyperparameters at marginal-MLE / posterior-mean estimates obtained *from the same data and the same generative model*. The EB construction is
$$\hat p(\theta) := p(\theta \mid \hat\eta(y)), \qquad \hat\eta(y) = \arg\max_\eta p(y \mid \eta),$$
where $\eta$ is a hyperparameter of the *same* generative model and $\hat\eta(y)$ is a point estimate obtained by marginalizing out $\theta$. The same source (and [Robbins 1956, "An empirical Bayes approach to statistics," *Proc. Berkeley Symp.*]) emphasizes that the EB posterior is a *frequentist approximation* to the fully Bayesian posterior $p(\theta \mid y) = \int p(\theta \mid \eta) p(\eta \mid y)\,d\eta$; the EB step is the marginal-MLE estimate $\hat\eta$, not the posterior of a coupled subsystem.

The cross-scale shadow does not match this template. It substitutes for the level-$\ell$ prior the *full variational posterior* of a different agent at level $\ell{+}1$, and the level-$(\ell{+}1)$ posterior is itself driven by free-energy terms that depend (through the participatory loop, line 2186 ff.) on the level-$\ell$ beliefs. This is neither (i) marginal-MLE hyperparameter selection nor (ii) plug-in of a marginal posterior — it is the identification of two posteriors via gauge transport, with no marginalization performed and no approximation to a stated fully-hierarchical reference model displayed. [Carlin & Louis §5.5] explicitly distinguishes hierarchical Bayes from EB and warns that EB underestimates uncertainty in $\eta$ because it conditions on $\hat\eta$; the cross-scale shadow does not even perform that step — there is no $\hat\eta$, just an equality between two posteriors.

**C3 — Ladder VAEs are not the analogue.**

[Sønderby, Raiko, Maaløe, Sønderby, Winther 2016, "Ladder Variational Autoencoders," *NeurIPS* 2016, §3 "Generative model and inference"] defines the ladder VAE generative model as
$$p(z_1, \ldots, z_L \mid x)\text{-prior: } p(z_\ell \mid z_{\ell+1}) = \mathcal{N}\!\big(z_\ell;\, \mu_\theta(z_{\ell+1}),\, \sigma^2_\theta(z_{\ell+1})\,I\big),$$
i.e. the level-$\ell$ prior is a *Gaussian conditional density* whose parameters are deterministic functions of the *next-level latent variable* $z_{\ell+1}$, computed by a learned $\theta$. The inference network $q(z_\ell \mid \ldots)$ is a *separate* family, and the ladder VAE ELBO contains $\mathrm{KL}\!\big(q(z_\ell \mid \cdot) \,\|\, p(z_\ell \mid z_{\ell+1})\big)$ — a KL between the variational posterior and the *conditional prior*, with both objects from distinct families.

The cross-scale shadow does not match this construction. It sets $p_i^{(s)} \equiv \Omega_{i,I}[q_I^{(s+1)}]$: the prior is set equal to the (transported) variational posterior $q_I^{(s+1)}$, not to a Gaussian conditional density of $z_{\ell+1}$. The analogue would be ladder-VAE-with $p(z_\ell \mid z_{\ell+1}) := q(z_{\ell+1} \mid x)$ in the prior — which is not what Sønderby et al. 2016 §3 do. Their construction conditions a *new* Gaussian density on the value of the next-level latent; the manuscript identifies the prior with the next-level posterior itself.

**C4 — The line-546 admission is dispositive under the literature-as-source-of-truth standard.**

`Participatory_it_from_bit.tex:546` reads:

> "This is a structural commitment of the framework rather than a theorem of standard hierarchical variational inference: in the standard scheme [Friston2017, parr2022active] the level-$\ell$ prior is derived from a generative-model conditional $p(s_\ell \mid s_{\ell+1})$, not posited as a transported posterior, and we do not display the reduction (or approximation) of the standard hierarchical scheme to the present cross-scale shadow construction."

This sentence states three things that together are dispositive of the claim under debate. First, the manuscript labels the construction as a "structural commitment of the framework rather than a theorem of standard hierarchical variational inference" — i.e., explicitly *outside* the deductive closure of standard hierarchical VI. Second, the manuscript states the standard scheme uses a generative-model conditional $p(s_\ell \mid s_{\ell+1})$ rather than a transported posterior — i.e., explicitly *different from* the present construction. Third, the manuscript declines to display *either* a reduction *or* an approximation chain to the standard scheme — i.e., the relationship to the literature is not just unproven, it is not exhibited at all.

A construction that (a) is not a theorem of the standard scheme, (b) is acknowledged to differ from the standard scheme at the point under debate, and (c) has no displayed reduction or approximation to the standard scheme, is not a refinement of the standard scheme — it is a parallel construction. Under the user's literature-as-source-of-truth directive, the only way a claim of refinement survives this admission is if blue produces the missing reduction / approximation chain from a primary source. Without that, the construction is at best a *separate* framework that uses overlapping vocabulary.

**C5 — The cross-scale shadow does not yield a proper joint generative model.**

[Wainwright & Jordan 2008, *Graphical Models, Exponential Families, and Variational Inference*, §3.4 "Mean field methods"] formalizes variational inference as: fix a generative model $p(x, z)$ from an exponential family; choose a tractable family $\mathcal{Q}$; solve $q^* = \arg\min_{q \in \mathcal{Q}} \mathrm{KL}(q \| p(\cdot \mid x))$. The premise of the variational principle is that $p$ is held fixed while $q$ is optimized.

In Friston 2017 §"Generative models and deep temporal models" the hierarchical joint
$$p(\tilde o, \tilde s) = p(s_L)\prod_\ell p(s_\ell \mid s_{\ell+1})\,p(o \mid s_1)$$
is a proper normalized density: each conditional $p(s_\ell \mid s_{\ell+1})$ integrates to 1 over $s_\ell$ for each fixed $s_{\ell+1}$, and the product yields a single joint density over the full state hierarchy.

The cross-scale shadow construction violates both properties. (i) The level-$s$ "prior" $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ is *a function of variational parameters* — $q_I^{(s+1)}$ is the optimized variational distribution of agent $I$ at scale $s{+}1$, whose mean and covariance are computed by free-energy minimization. The "generative model" is therefore mutated by every E-step of every agent at every higher scale: there is no fixed $p$ against which $q$ is being optimized in the standard variational sense. (ii) The participatory loop closure at `Participatory_it_from_bit.tex:2186 ff.` (Eq.~\ref{eq:topdown_priors}) and the top-scale self-referential closure at lines 2218–2223 make $q_I^{(s+1)}$ itself depend on the level-$s$ beliefs $\{q_i^{(s)}\}$ (bottom-up aggregation) — so $p$ at scale $s$ is a function of the variational quantities that $p$ at scale $s$ is regularizing. This is a coupled variational fixed-point system, not a fixed generative model in the sense of [Friston 2017] or [Wainwright–Jordan 2008 §3.4]. The "ELBO" written against this construction is not an ELBO of any standard generative model — it is a free-energy functional of a coupled-agent consensus problem, which the manuscript itself acknowledges at `:1038` as a "consensus-energy ansatz rather than a generative-model derivation" and at `:1023` as "an engineered consensus energy, not a derivation from FEP alone."

**C6 — The Ouroboros identification at line 2216 covers the weighting structure, not the substitution.**

`Participatory_it_from_bit.tex:2216` identifies the geometric discount $\lambda_k = \lambda_0\rho^k$ with [West–Harrison 1997, *Bayesian Forecasting and Dynamic Models*, 2nd ed., Ch. 2 §"Discount factors"]; the additive form $\sum_k \lambda_k \mathrm{KL}(q_i \| p_k)$ with the log-linear opinion pool of [Genest & Zidek 1986, "Combining Probability Distributions: A Critique and an Annotated Bibliography," *Statistical Science* 1(1):114–148, §3 "Logarithmic opinion pool"] and with the product-of-experts of [Hinton 2002, "Training Products of Experts by Minimizing Contrastive Divergence," *Neural Comput.* 14:1771–1800, §2 "The basic idea"]; and the unrestricted-weight default with the generalized-Bayesian tempered posterior of [Bissiri, Holmes, Walker 2016, "A general framework for updating belief distributions," *JRSS-B* 78:1103, §2 "A general framework for updating belief distributions"].

All four sources operate on *given* component distributions and combine them into a pooled or tempered prior/posterior. None of the four substitutes the posterior of a coupled subsystem for the prior of a hierarchical generative model. Specifically:

- West–Harrison discount factors apply within a *single* dynamic linear model with a *fixed* observation density; the discount governs how much the posterior at time $t-1$ contributes to the prior at time $t$ within one agent's model. The "historical observation" is the same agent's own past data under the same generative model, not the posterior of a coupled neighbor.
- Genest–Zidek §3 takes as input a set of probability distributions $\{p_1, \ldots, p_n\}$ and pools them as $p^* \propto \prod_k p_k^{w_k}$. The inputs are pre-specified densities; the pool is an aggregation operator.
- Hinton 2002 PoE multiplies *expert distributions* together to obtain a sharper posterior; the expert densities are given.
- Bissiri–Holmes–Walker 2016 §2 derives a generalized posterior update $\pi(\theta \mid x) \propto \pi(\theta)\exp\{-w \ell(\theta, x)\}$ from coherent-update axioms; it temperatures the *likelihood* against a *fixed* prior.

Identifying the Ouroboros weighting with these objects covers the *additive-discount-on-KL fragments* part of the construction. It does not cover, and cannot legitimize, the upstream substitution at Eq.~\ref{eq:cross_scale_shadow} that defines each $p_k = \Omega_{i,I_k}[q_{I_k}^{(s+k)}]$ as the gauge transport of *another agent's variational posterior*. The line-2216 identification establishes that *if* one accepts the cross-scale shadow's identification of priors with transported posteriors, *then* the geometric-discounted sum of KL-against-those-priors is well-defined and matches a tempered-pool form. It does not establish that the upstream identification is itself a hierarchical-Bayesian object.

## Falsification conditions

This position is wrong if blue can produce any of the following:

1. A quotation from Friston 2017 / Parr–Pezzulo–Friston 2022 / a comparable canonical hierarchical-AIF reference in which the level-$\ell$ prior is defined as the level-$(\ell{+}1)$ *variational posterior* (not a generative-model conditional, and not a marginal of the joint under fixed generative parameters). A section number and equation are required.
2. A primary-source EB / hierarchical-Bayes treatment ([Robbins 1956], [Carlin–Louis Ch. 5], [Efron 2010]) in which the prior at one level is set equal to the optimized posterior of a *coupled subsystem* whose posterior depends on the prior in question, with the joint distribution shown to be well-defined and proper.
3. A construction in [Sønderby et al. 2016] or comparable ladder/hierarchical-VAE primary source in which the level-$\ell$ prior is set equal to the *posterior* $q(z_{\ell+1} \mid x)$ rather than to a Gaussian conditional density with parameters $\mu_\theta(z_{\ell+1}), \sigma^2_\theta(z_{\ell+1})$. Section and equation reference required.
4. An explicit reduction or approximation chain — derived in a primary source, not posited by the manuscript — that maps the standard hierarchical-VI generative-model conditional to the gauge-transported posterior of the cross-scale shadow under stated limits. The manuscript at line 546 declines to display this chain; blue must supply it from the literature.
5. A demonstration that the joint distribution $p(\{q_i^{(s)}\}, \{s_i^{(s)}\})_{s,i}$ implied by Eq.~\ref{eq:cross_scale_shadow} integrates to one under fixed generative parameters and is a valid generative model in the sense of [Wainwright–Jordan 2008 §3.4] (or any comparable canonical formulation of variational inference). Failing this, the "ELBO" written against this object is not an ELBO of a hierarchical-VI model.

Failure to satisfy any one of (1)–(5) leaves the line-546 admission unrebutted and the claim of "legitimate refinement of standard hierarchical VI" unsupported.
