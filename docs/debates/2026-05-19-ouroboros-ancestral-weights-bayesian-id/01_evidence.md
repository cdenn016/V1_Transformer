# Evidence Pack — ouroboros-ancestral-weights-bayesian-id

## Manuscript references (Attention/Participatory_it_from_bit.tex)

### The Ouroboros tower extension (the construction under debate)

- `Attention/Participatory_it_from_bit.tex:2196` — Section header: "**Ouroboros Tower Extension:** Optionally, constituents can receive shadows from multiple ancestral scales simultaneously, not just their immediate parent: a form of epigenetic information transfer. In this non-Markovian extension, agent $i$ at scale $s$ inherits state-fiber shadows from each ancestral generation,"
- `Attention/Participatory_it_from_bit.tex:2197-2201` — State-fiber shadow assignments:
  - $p_i^{(s)} \leftarrow \Omega_{i,I_1}[q_{I_1}^{(s+1)}]$ (parent)
  - $h_i^{(s,0)} \leftarrow \Omega_{i,I_2}[q_{I_2}^{(s+2)}]$ (grandparent)
  - $h_i^{(s,1)} \leftarrow \Omega_{i,I_3}[q_{I_3}^{(s+3)}]$ (great-grandparent)
- `Attention/Participatory_it_from_bit.tex:2202-2207` — Model-fiber shadow assignments with $(\tilde\Omega, s)$ replacing $(\Omega, q)$:
  - $r_i^{(s)} \leftarrow \tilde\Omega_{i,I_1}[s_{I_1}^{(s+1)}]$ (parent)
  - $\tilde h_i^{(s,0)} \leftarrow \tilde\Omega_{i,I_2}[s_{I_2}^{(s+2)}]$ (grandparent)
  - $\tilde h_i^{(s,1)} \leftarrow \tilde\Omega_{i,I_3}[s_{I_3}^{(s+3)}]$ (great-grandparent)
- `Attention/Participatory_it_from_bit.tex:2208` — **The load-bearing sentence under debate:** "This creates information flow from ALL ancestral scales on both fibers, with hyper-priors and their multi-generation analogues stored as additional fields that can be incorporated into the free energy functional with exponentially decaying weights $\lambda_k = \lambda_0 \cdot \rho^k$ where $k$ is the generational distance."

### Surrounding context — Self-Referential Closure (related but not under debate)

- `Attention/Participatory_it_from_bit.tex:2210-2215` — Self-referential closure at the top of the hierarchy. Different weighting: $w_j \propto \exp(-\overline{\mathrm{KL}}_j)$, *not* $\rho^k$. Out of scope.

### Manuscript's own characterization

- The manuscript labels the Ouroboros as a "non-Markovian extension" (line 2196), as "Optionally" (line 2196), and as "epigenetic information transfer" (line 2196). No generative model is specified; no derivation from a standard hierarchical-Bayesian object is given. The $\rho^k$ form is introduced as a *weighting choice* on free-energy contributions, not as a covariance structure or as the negative log-prior of a stated generative model.

### Prior debate result

- `docs/debates/2026-05-19-agent-meta-agent-hierarchy-theory/03_blue_rebuttal.md` — blue conceded the narrow derivability point: the Ouroboros tower exponential decay is not derived from any identified standard hierarchical-Bayesian object.
- `docs/debates/2026-05-19-agent-meta-agent-hierarchy-theory/04_verdict.md` — verdict accepted the integrated theory-section claim because sub-claim C only required correct *labeling*. The current debate revisits the derivability point with the question framed as a *disjunctive identification*.

## Canon excerpts

### From external_canon_inference.md

- **§3 — Hierarchical / nested formulations** `[Friston2017Graphical, ParrPezzuloFriston2022 Ch. 9]`: "For a hierarchical model $p(o, s_1, s_2, \ldots, s_L)$ with $s_\ell$ the latent at level $\ell$, the recognition distribution factorizes as $q(s_1, \ldots, s_L) = \prod_\ell q(s_\ell)$ (mean-field) or with more structure. F decomposes additively across levels under mean-field. **Cross-level couplings come from the generative model $p(s_\ell | s_{\ell+1})$.**" — the standard hierarchical FEP is Markovian in scale; cross-level coupling is *one-step*, not multi-step with exponential decay.
- **§3 — Pitfall (canon §10 item 5):** "Hierarchical mean-field vs point-passing. Passing a posterior *mean* (not the full posterior) between levels is a deterministic approximation, not variational inference proper." — the Ouroboros propagates posteriors from multiple ancestors, but the variational status of multi-ancestor coupling is not standard.

### Standard references the teams should cite

**Deep Gaussian processes (Identification I):**
- `[DamianouLawrence2013]` — *Deep Gaussian Processes*, AISTATS. Stack of GPs with input-output coupling at each layer. The "depth" is over the function composition, not the GP kernel itself.
- `[SalimbeniDeisenroth2017]` — *Doubly Stochastic Variational Inference for Deep Gaussian Processes*, NIPS. Standard modern DGP reference.
- `[RasmussenWilliams2006]` — *Gaussian Processes for Machine Learning*, MIT Press. The Ornstein-Uhlenbeck (Matérn 1/2) kernel $k(s,s') = \sigma^2 \exp(-|s-s'|/\ell)$ which under $\rho = \exp(-1/\ell)$ produces $\sigma^2 \rho^{|s-s'|}$ on integer indices.

**AR(1) and time-series priors (Identification II):**
- `[Hamilton1994 Ch. 3]` — *Time Series Analysis*. AR(1): $x_t = \rho x_{t-1} + \varepsilon_t$, stationary covariance $\mathrm{Cov}(x_t, x_{t+k}) = \sigma^2_\varepsilon \rho^k / (1 - \rho^2)$.
- `[BoxJenkinsReinsel2008]` — *Time Series Analysis: Forecasting and Control*. Standard reference.
- `[WestHarrison1997 Ch. 10]` — *Bayesian Forecasting and Dynamic Models*. The discount factor in dynamic linear models is *exactly* the geometric $\rho^k$ structure applied to historical observations.

**Power priors / discount-factor priors (Identification III):**
- `[IbrahimChen2000]` — "Power prior distributions for regression models," Statistical Science. The power prior raises the historical likelihood to a power $a_0 \in [0,1]$. Hierarchical extensions allow $a_0$ to vary across historical generations.
- `[ChenIbrahimShao2000]` — Hierarchical power priors.
- `[NeuenschwanderEtAl2010]` — meta-analytic-predictive prior — closely related family.

**Non-Markovian state-space models:**
- `[Beran2010]` — *Long-Memory Processes*. Fractional / long-memory time-series with power-law decay.
- `[HochreiterSchmidhuber1997]` — LSTM. Forget gate $f_t \in [0,1]$ produces effective discount factor for past inputs.

**The correspondence between free energy and generative models:**
- `[Friston2010]` — Standard variational free energy form. $F = E_q[\log q - \log p(o,s)] = \mathrm{KL}(q \| p(s)) + \text{accuracy}$. Adding a multi-source prior $p(s) = \prod_k p_k(s)^{\lambda_k} / Z$ produces additive $\sum_k \lambda_k \mathrm{KL}(q \| p_k)$ terms in F.
- `[BleiKuckelbirgJordan2017]` — VI primer. The correspondence between additive KL terms and product-of-experts priors / log-linear pools.
- `[HintonEtAl2002]` — Product-of-experts. The geometric mean of densities $p \propto \prod_k p_k^{\lambda_k}$ has negative log-prior $\sum_k \lambda_k(-\log p_k)$, which under Gaussian families produces additive KL terms after expansion around q.

## Mathematical preliminaries (neutral)

### The additive-KL ↔ product-prior correspondence

For Gaussian beliefs $q = \mathcal{N}(\mu_q, \Sigma_q)$ and Gaussian priors $p_k = \mathcal{N}(\mu_k, \Sigma_k)$, the weighted sum
$$ \sum_k \lambda_k \mathrm{KL}(q \| p_k) $$
is the negative log-density (up to entropy of $q$ which is shared) of a generative model with prior
$$ p(s) \propto \prod_k p_k(s)^{\lambda_k} $$
(a product-of-Gaussian-experts / log-linear pool). For Gaussian $p_k$ this product is also Gaussian, with precision $\Lambda = \sum_k \lambda_k \Lambda_k$ and mean $\mu = \Lambda^{-1} \sum_k \lambda_k \Lambda_k \mu_k$.

**This is the correspondence both teams must engage:** does this product-of-experts construction over ancestral generations correspond to any of the three standard objects (DGP, AR(1) scale-prior, power-prior SSM)?

### The exponential-kernel / AR(1) / geometric-discount equivalence

For the discrete-index Ornstein-Uhlenbeck process on integer scale indices, the kernel $k(s,s') = \sigma^2 \rho^{|s-s'|}$ is the stationary covariance of an AR(1): $x^{(s+1)} = \rho x^{(s)} + \varepsilon$, with $\varepsilon \sim \mathcal{N}(0, \sigma^2(1-\rho^2))$. The AR(1) is Markovian; the OU kernel is the *covariance* induced by this Markovian model, not a non-Markovian object.

This is the key structural question: the manuscript labels the Ouroboros "non-Markovian" (line 2196), but $\rho^k$ is the AR(1)/OU stationary covariance, which is induced by a *Markovian* model. The non-Markovian label is correct for the *agent-level prior* (which depends on multiple ancestors, not just the parent), but the *generative process* could still be Markovian-in-scale. This is the same distinction as in time series: an AR(1) process is Markovian, but the marginal observation at time $t$ has nonzero correlation with observations at all past times.

## What this evidence does NOT settle

- Whether the manuscript's $\lambda_k$ weights apply to *additive KL terms* in the free energy (which corresponds to a product-of-experts prior across ancestors) or to *covariance contributions* (which corresponds to a Gaussian process kernel). The line 2208 phrasing — "incorporated into the free energy functional with exponentially decaying weights $\lambda_k$" — is most naturally read as the additive-KL form, but the manuscript does not write the explicit $F$ functional with the $\lambda_k$ inserted.
- Whether the Ouroboros free-energy term is meant to be interpreted as a single agent's penalty for misalignment with all ancestors (the additive-KL reading) or as a covariance structure on the joint distribution of agent and ancestors (the GP/AR(1) reading). These two readings produce different generative models.
- Whether the manuscript's gauge transports $\Omega_{i,I_k}$ inside each KL term admit a clean embedding into any of the three candidate identifications. The standard hierarchical-Bayesian objects do not include gauge transports; this is an extra structure the framework introduces.
- Whether identifications I, II, III are *mutually exclusive*, or *equivalent under the standard correspondences*. (Spoiler: for Gaussian linear-Gaussian models, the AR(1) generative process produces the OU kernel marginally, so I and II coincide; III is a different object — a power prior is a re-weighting of historical likelihoods, not a covariance structure.)
- Whether the "non-Markovian" label at line 2196 is mathematically licensed: a model in which each agent's prior depends on multiple ancestors *can* be reduced to a Markovian model on an augmented state (the standard SSM trick). The manuscript does not address this reduction.

These are the cracks the red team should probe and the load-bearing structural questions the blue team must address.
