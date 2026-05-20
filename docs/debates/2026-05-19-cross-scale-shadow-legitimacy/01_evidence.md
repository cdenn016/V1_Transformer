# Evidence Pack — cross-scale-shadow-legitimacy

**Manuscript:** `Attention/Participatory_it_from_bit.tex`
**Section in scope:** §sec:cross_scale_shadows (line 536); §sec:agent_definition (line 612); §sec:participatory (line 2044), specifically Eq.~\ref{eq:topdown_priors} (line 2188); §sec:variational_free_energy (line 1005); §`Ouroboros Tower Extension` (line 2197)
**Canon location:** `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge\`
**Authoritative directive (user):** Teams consult the external literature as source of truth.

## Manuscript references — exact passages

- **Definition of the cross-scale shadow** — line 540–545:
  > $p_i^{(s)}(c) = \Omega_{i,I}[q_I^{(s+1)}](c), \quad r_i^{(s)}(c) = \tilde\Omega_{i,I}[s_I^{(s+1)}](c)$, $\quad$ (eq:cross_scale_shadow)

- **Manuscript's own labeling** — line 546:
  > "This is a structural commitment of the framework rather than a theorem of standard hierarchical variational inference: in the standard scheme [Friston2017, parr2022active] the level-$\ell$ prior is derived from a generative-model conditional $p(s_\ell | s_{\ell+1})$, not posited as a transported posterior, and we do not display the reduction (or approximation) of the standard hierarchical scheme to the present cross-scale shadow construction. Conditional on Eq.~\eqref{eq:cross_scale_shadow}, the hierarchy $r \to s \to p \to q \to \text{observations}$ posited at the agent definition below is realized cross-scale."

- **Re-stated as formal content of the feedback loop** — line 2188–2195, Eq.~\ref{eq:topdown_priors}:
  > "Equation~\eqref{eq:topdown_priors} is the formal content of the cross-scale shadow relation~\eqref{eq:cross_scale_shadow} introduced in Section~\ref{sec:statistical_manifolds}: the prior $p_i^{(s)}$ and hyper-prior $r_i^{(s)}$ at scale $s$ are not independent dynamical fields but are determined by the meta-agent at scale $s+1$, and the apparent four-field hierarchy $r \to s \to p \to q$ is in fact two primitive fields $(q, s)$ at each scale connected by cross-scale transport."

- **Foundation citation** — line 1015:
  > "The variational free energy principle~\cite{Friston2010, parr2022active} provides a tractable approximation to intractable Bayesian inference..."

- **Acknowledged-extension language** — line 1023:
  > "The functional developed in the remainder of this section is a *novel* multi-agent extension of Friston's single-agent free energy: the inter-agent belief-coupling terms ... are not present in the standard FEP literature~\cite{Friston2010, parr2022active}; ... We retain the FEP citations as the conceptual backbone (variational inference, the ELBO bound, the complexity/accuracy decomposition) but the multi-agent functional itself is an engineered consensus energy, not a derivation from FEP alone."

- **Ouroboros tower extension** — line 2216:
  > "The construction admits identification with a standard hierarchical-Bayesian object. The geometric weighting $\lambda_k = \lambda_0\rho^k$ on per-generation contributions is the same per-step discount used in the dynamic linear models of [WestHarrison1997]... the additive structure $\sum_k \lambda_k \mathrm{KL}(q_i \| p_k)$ ... corresponds under the standard correspondence between additive KL contributions and product priors [Friston2010, blei2017variational] to a Gaussian log-linear pool or product-of-experts prior [GenestZidek1986, HintonPoE2002]."

## External-literature waypoints

### Standard hierarchical variational inference / active inference (the canonical comparison)
- **Friston, "The free-energy principle: a unified brain theory?"**, *Nat. Rev. Neurosci.* 11:127–138 (2010). Sets up the single-agent variational free energy.
- **Friston, FitzGerald, Rigoli, Schwartenbeck, Pezzulo**, "Active inference: a process theory", *Neural Comput.* 29:1–49 (2017). Hierarchical generative model $p(o, s_1, ..., s_L) = \prod_\ell p(s_\ell | s_{\ell+1}) p(o|s_1)$ where the level-$\ell$ prior $p(s_\ell | s_{\ell+1})$ is a conditional of the generative model, not a transported posterior.
- **Parr, Pezzulo, Friston**, *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*, MIT Press 2022, Ch. 8 (Deep generative models). Same hierarchical structure as Friston 2017.

### Empirical Bayes (the strongest candidate for "transported posterior as prior")
- **Robbins**, "An empirical Bayes approach to statistics", *Proc. Berkeley Symp.* (1956). Foundational empirical-Bayes: replace the hyperprior with a point/posterior estimate from data.
- **Carlin & Louis**, *Bayesian Methods for Data Analysis*, 3rd ed., Ch. 5. Modern treatment of EB with the hierarchical generative model.
- **Efron**, *Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction*, Cambridge 2010.

### Amortized hierarchical VI / variational autoencoder lineage
- **Kingma & Welling**, "Auto-encoding variational Bayes", *ICLR* 2014. Original VAE; posterior $q(z|x)$ amortized as a function of $x$.
- **Rezende, Mohamed, Wierstra**, "Stochastic backpropagation and approximate inference in deep generative models", *ICML* 2014.
- **Sønderby, Raiko, Maaløe, Sønderby, Winther**, "Ladder variational autoencoders", *NeurIPS* 2016. Hierarchical VAE; the *prior* at each level is shaped by a deterministic function of the next-level latent — a possible analogue of the manuscript's $p_i^{(s)} \leftarrow \Omega_{i,I}[q_I^{(s+1)}]$ if one identifies the gauge-transport with a deterministic decoder.

### Message passing / belief propagation on graphical models
- **Pearl**, *Probabilistic Reasoning in Intelligent Systems*, 1988, Ch. 4. Belief propagation: messages flow between adjacent nodes in a graphical model.
- **Bishop**, *Pattern Recognition and Machine Learning*, 2006, Ch. 8 §8.4. Sum-product algorithm; messages = transported beliefs from neighbors.
- **Wainwright & Jordan**, *Graphical Models, Exponential Families and Variational Inference*, 2008. Variational message passing.

### Product of experts / log-linear opinion pools (cited by manuscript at line 2216)
- **Hinton**, "Training products of experts by minimizing contrastive divergence", *Neural Comput.* 14:1771 (2002).
- **Genest & Zidek**, "Combining probability distributions: a critique and an annotated bibliography", *Statistical Science* 1:114 (1986). External Bayesianity criterion.
- **West & Harrison**, *Bayesian Forecasting and Dynamic Models*, 2nd ed., 1997. Per-step discount factor for dynamic linear models.

### Tempered / generalized Bayes (cited by manuscript at line 2216)
- **Bissiri, Holmes, Walker**, "A general framework for updating belief distributions", *JRSS-B* 78:1103 (2016). Generalized-Bayesian posterior updates without a tightly defined likelihood; legitimizes a tempered free-energy with $\lambda_0, \rho > 0$ unconstrained.

## Sub-claims to weigh (compound but tightly bound)

C1 — Does Friston 2017 §4 / Parr-Pezzulo-Friston 2022 Ch. 8 use a generative-model conditional $p(s_\ell | s_{\ell+1})$, in contrast to the manuscript's transported posterior?

C2 — Is "level-$(\ell{+}1)$ posterior as level-$\ell$ prior" a recognized refinement in the empirical-Bayes literature (Robbins 1956; Carlin–Louis Ch. 5)?

C3 — Does the ladder-VAE construction (Sønderby et al. 2016) place a deterministic function of the next-level latent in the level-$\ell$ prior, and if so, does the manuscript's $\Omega_{i,I}[q_I^{(s+1)}]$ map onto a gauge-equivariant ladder-VAE prior?

C4 — Does the manuscript's own admission at line 546 (no reduction or approximation chain displayed) entail that the construction is not in continuity with Friston 2017 / Parr-Pezzulo-Friston 2022, or merely that the reduction is left as future work?

C5 — Is the resulting probabilistic model still a well-defined joint distribution $p(\{q_i\}_i, \{s_i\}_i)$ with proper normalization, or does identifying the prior at level $\ell$ with the posterior at level $\ell{+}1$ create a circularity (the posterior depends on the prior, the prior is the posterior of a different but coupled subsystem) that prevents the joint from being a valid generative model?

C6 — Does the Ouroboros-tower identification with West–Harrison 1997 / Genest–Zidek 1986 / Bissiri–Holmes–Walker 2016 at line 2216 retroactively legitimize the cross-scale shadow construction as a hierarchical-Bayesian object?

## Operational notes

- The manuscript line 546 contains an *unusual* admission: it explicitly labels the construction as a "structural commitment of the framework rather than a theorem of standard hierarchical variational inference." Both teams must engage with this admission rather than ignore it.

- The judge will weigh whether "structural commitment without a displayed reduction" qualifies as a legitimate refinement (blue) or as an undeclared model substitution (red). The decisive question is whether the cross-scale shadow can be derived from any standard hierarchical-Bayesian construction in the literature, or whether it must be posited as a separate framework.

- Both teams must cite primary sources with section/equation/page references. Strikes citing only the manuscript or only the project's vfe-knowledge are weak.
