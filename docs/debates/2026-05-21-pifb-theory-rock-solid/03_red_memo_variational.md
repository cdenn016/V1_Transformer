# Red Rebuttal Memo — variational

Lens: variational inference — ELBO, EM separation, mean-field, hierarchical VI, mixture-of-experts, FEP.

## One concession from blue's opening

Blue's defense item 4 correctly notes that the manuscript labels the mixture-of-sources construction as a *consensus-energy ansatz* at line 1038 and labels the source-independence factorization $Q(k|z)=q_i(k)$ as a structural assumption at line 1064. That labeling discipline is genuinely present and does meet the `external_canon_inference.md:29` requirement that "multi-agent coupling terms ... should be labeled as a novel construction requiring its own justification." Granted.

## Strongest attack on blue's core defense

Blue's defense item 4 puts decisive weight on the claim that the cross-scale shadow $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ is "closed by explicit reduction at line 552, not papered over" via the $\sigma^2 \to 0$ rigid-link limit of a Gibbs cross-scale factor $\chi_{i,\pi(i)} \propto \exp[-\|k_i - \Omega_{i,\pi(i)}k_{\pi(i)}\|^2/(2\sigma^2)]$, identified as the "deterministic-decoder limit of a ladder-VAE-style hierarchical prior~\cite{Sonderby2016}." Two distinct problems destroy this defense.

(1) **The $\sigma^2 \to 0$ limit produces a Dirac singularity that the manuscript itself flags as ill-defined.** As $\sigma^2 \to 0$, the Gibbs factor $\chi_{i,\pi(i)}(k_i, k_{\pi(i)}) \propto \exp[-\|k_i - \Omega_{i,\pi(i)}k_{\pi(i)}\|^2/(2\sigma^2)]$ converges (in the distributional sense) to $\delta(k_i - \Omega_{i,\pi(i)}k_{\pi(i)})$. The resulting "prior" $p_i^{(s)}$ is then a Dirac measure concentrated at the transported parent posterior. The manuscript itself states at line 1458, *inside §Theory*: "A Dirac choice $q_{e_k}(c) = \delta(c - c_k)$ ... gives an infinite KL whenever $q_i$ does not coincide pointwise with $\delta(c - c_k)$, since $\mathrm{KL}(q_i \| \delta(c - c_k)) = +\infty$ for any non-degenerate $q_i$." This is the standard absolute-continuity failure documented at `external_canon_math.md` (KL of Gaussians requires both measures on the same support). The cross-scale shadow at line 552 invokes the rigid-link limit *as the definition of $p_i^{(s)}$*. The $\mathrm{KL}(q_i^{(s)} \| p_i^{(s)})$ term in the canonical free energy at Eq.~\eqref{eq:free_energy_functional_final} therefore evaluates to $+\infty$ in the very limit blue's defense uses to "close" the standard-FEP gap. This is not a derivation closure; it is the manuscript's own line 1458 self-contradiction, with §Theory failing sub-claim 1 (mathematical correctness) inside the load-bearing reduction.

(2) **The Sønderby Ladder VAE precedent does not endorse the rigid-link limit.** Sønderby et al., *Ladder Variational Autoencoders*, NeurIPS 2016 ([arXiv:1602.02282](https://arxiv.org/abs/1602.02282)) constructs the top-down generative model as a learned-mean, learned-finite-variance Gaussian at each layer ($p_\theta(z_\ell | z_{\ell+1}) = \mathcal{N}(\mu_\theta(z_{\ell+1}), \mathrm{diag}(\sigma_\theta^2(z_{\ell+1})))$) — not a deterministic point-passing scheme. The standard hierarchical-VI corpus (`[Friston2017Graphical]`, `[ParrPezzuloFriston2022 Ch. 9]`, `[RanganathTranBlei2016]`) likewise uses finite-variance conditional densities for $p(s_\ell | s_{\ell+1})$. The manuscript's $\sigma^2 \to 0$ limit is not the deterministic-decoder limit of any of these standard hierarchical-VI constructions — it is a singular limit that none of the cited references take, and the standard hierarchical-VI corpus does not contain a published precedent for it (`external_canon_inference.md:60` is exactly this gap). Blue's defense item 4 cites `[Sonderby2016]` as endorsement; the citation is the right paper for the wrong claim. Per the canon-cop rubric on "wrong-domain citation" (2 strikes), the cite is malformed.

These two problems compound: even granting blue the lemma-A backstop (Lemma `lem:shadow_mf_optimum` referenced at manuscript line 552), the lemma cannot establish a well-defined joint marginal in the limit where the conditional density it relies on becomes Dirac.

## Strongest defense against blue's strongest attack

Blue's evidence item 4 does not attack red's opening directly; it engages on the variational front via the ladder-VAE move. The defense to strengthen is the conjugate-prior sub-strike on the per-coordinate precision at lines 1330–1346.

Blue's info-geometer memo (item 2) defends the per-coordinate $\alpha$ Bregman-Lagrangian as "an elementary scalar Lagrangian — sympy-verifiable in one line." Granted. But the *promotion* from the elementary scalar Lagrangian to the diagonal-Gaussian / Gamma-Normal conjugate-prior interpretation at lines 1330–1340 requires the full Gamma-Normal joint stationary equations of `[Bishop2006 §10.2]` and `[Beal2003]` *Variational Algorithms for Approximate Bayesian Inference* (Gatsby Unit, UCL). The manuscript cites Bishop at line 1336 but does not display the joint stationary equations; the conjugacy reading is asserted, not derived. This is exactly the variational-canon-fidelity gap the steelman raised. Blue's "the closed-form $\alpha^*$ is derived in two lines" defense rebuts only half of the steelman — the closed form is the *elementary scalar derivation*, not the *conjugate-prior promotion*. The promotion stands as an unsupported canonical-form claim within §Theory.

This sub-strike is secondary to the Dirac strike above and is offered as reinforcement, not as a separate vector.

## Newly-discovered canon

None beyond what the Phase-2 red harvest already recorded. `[Sonderby2016]` (full arXiv reference: arXiv:1602.02282), `[RanganathTranBlei2016]` (arXiv:1511.02386), and `[Beal2003]` (UCL Gatsby thesis) were all recorded in `01b_extended_evidence.md`.
