# Blue Memo — info-geometer

Lens: information geometry — Fisher metric, KL divergence, exponential families, dual connections, Cencov.

## Steelman (opposing position)

§Theory is not publication ready because (i) the multi-agent coupling $\sum_{ij}\beta_{ij}\mathrm{KL}(q_i\|\Omega_{ij}q_j)$ is not in the standard FEP form per `external_canon_inference.md:29` ("multi-agent coupling terms ... are user-introduced ... not the field standard"), and (ii) the per-coordinate precision construction at lines 1330–1352 invokes Gamma-Normal conjugacy from `[Bishop2006 §10.2]` but does not derive the Gamma prior — it cites Bishop and asserts the Gamma promotion as the natural generalization. A canonical-fidelity check should require the derivation rather than the citation.

## Defense from the information-geometric lens

(1) **The Gaussian KL form at line 1910 matches the textbook closed form.** The manuscript invokes the standard
$$D_{\mathrm{KL}}(\mathcal{N}(\mu_q,\Sigma_q)\|\mathcal{N}(\mu_p,\Sigma_p))=\tfrac12\big[\mathrm{tr}(\Sigma_p^{-1}\Sigma_q)+(\mu_p-\mu_q)^\top\Sigma_p^{-1}(\mu_p-\mu_q)-K+\log(|\Sigma_p|/|\Sigma_q|)\big]$$
verbatim against `[BleiKuckelbirgJordan2017]` and `[KingmaWelling2014 App. B]` (also stated at `external_canon_math.md:42`). The per-coordinate decomposition at line 1333 is the diagonal-Σ specialization that `external_canon_math.md:46` records as textbook. No flag.

(2) **The state-dependent precision optimum is a Bregman-Lagrangian closed form, not a citation-driven assertion.** Lines 1297–1320 derive $\alpha_i^*(c) = c_0/(b_0 + D_{\mathrm{KL}}(q_i\|p_i))$ from stationarity of $\alpha_i D_{\mathrm{KL}} + b_0\alpha_i - c_0\log\alpha_i$ with respect to $\alpha_i$, which is an elementary scalar Lagrangian — sympy-verifiable in one line. The log-barrier regularizer $R(\alpha)=b_0\alpha-c_0\log\alpha$ is the conjugate prior for the precision of a Gaussian under the Gamma family `[Bishop2006 §10.2]`, and the variational-EM treatment of the diagonal-Gaussian / Gamma-Normal mean-field at lines 1330–1340 is the canonical Bishop construction. The steelman's concern — that the manuscript invokes Bishop without derivation — is overstated: the closed-form $\alpha^*$ is derived in two lines on the same page, and the Gamma-Normal conjugacy is the standard naming for the construction (line 1336 cites $[\text{Bishop2006 §10.2}]$ correctly). What is *user-introduced* (the log-barrier instead of, say, a quadratic) is explicitly named as "a natural choice" at line 1306. No flag for canonical fidelity.

(3) **The envelope-theorem treatment correctly distinguishes the autograd surrogate from the reduced free energy.** Lines 1361–1398 (the equation block and the "Autograd versus reduced-free-energy gradients" paragraph) state
$$\nabla\langle E\rangle_\beta - \nabla\mathcal{F}_{\mathrm{red}} = -\tau^{-1}\mathrm{Cov}_{\beta^*}(E,\nabla_x E),$$
identifying the obstruction to gradient-equivalence as a covariance, and recovering vanishing under either (a) entropy-suppressed surrogate at the optimum, or (b) uncorrelated $(E,\nabla E)$ under $\beta^*$. This is the standard envelope theorem `[Milgrom Segal 2002]` applied to the soft-max attention Lagrangian, and the algebraic identity is correct: differentiating $-\tau\log Z = -\tau\log\sum_j\pi_j\exp(-E_j/\tau)$ yields $\nabla_x(-\tau\log Z) = \sum_j\beta_j^*\nabla_x E_j$ at the optimum, which is the receiver contribution at line 1392. No flag.

## Falsification condition for the claim

The info-geometric defense breaks if (a) any §Theory equation states the Gaussian KL in a form that differs from `external_canon_math.md:42` without explicit reparameterization, (b) the envelope-theorem reduction $-\tau\log Z$ at line 1378 of §Theory is used to claim gradient-equivalence with the autograd surrogate (which would contradict the covariance correction the manuscript itself states at line 1404), or (c) the multi-agent $\sum_{ij}\beta_{ij}\mathrm{KL}$ functional is presented as derivable from the FEP single-agent variational principle alone, rather than from the mixture-of-sources construction the manuscript explicitly identifies as an ansatz at line 1038. I have checked (b) and (c) and they hold — the manuscript names the surrogate distinction every time it uses the surrogate, and the mixture construction is flagged as ansatz at lines 1038, 1058, 1064. (a) is the empirical condition for §Theory mathematical correctness; it survives the spot checks I ran but a full equation-by-equation sympy sweep would be the test.

## Newly-discovered canon

`[Milgrom Segal 2002]` "Envelope theorems for arbitrary choice sets" is the canonical economics-side reference for the parametric envelope theorem the manuscript invokes at line 1386. The textbook statement is: for $V(x)=\max_\beta f(x,\beta)$, $\partial V/\partial x = \partial f/\partial x$ at $\beta=\beta^*(x)$ with no $\partial\beta^*/\partial x$ contribution. This is what licenses the receiver-side gradient at line 1391 without an explicit $\partial\beta^*/\partial x_i$ term. Should be added to `external_canon_inference.md` if not already; not in my current pass of `external_bibliography.md`.
