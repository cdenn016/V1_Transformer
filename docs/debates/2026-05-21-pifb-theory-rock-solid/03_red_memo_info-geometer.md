# Red Rebuttal Memo — info-geometer

Lens: information geometry — Fisher metric, KL divergence, exponential families, dual connections, Cencov uniqueness, Bregman.

## One concession from blue's opening

Blue's defense item 3 grants the manuscript's Gaussian KL at line 1910 matches the textbook closed form of `[BleiKuckelbirgJordan2017]` and `[KingmaWelling2014 App. B]` (`external_canon_math.md:42`) verbatim. Granted — the closed-form Gaussian KL at line 1910 is canonically faithful.

Blue's defense item 3 also correctly identifies the envelope-theorem reduction at lines 1361–1404 with the receiver-side gradient $\nabla_x(-\tau\log Z) = \sum_j \beta_j^* \nabla_x E_j$ at the optimum, and correctly states the autograd-vs-reduced-free-energy gap as $-\tau^{-1}\mathrm{Cov}_{\beta^*}(E, \nabla_x E)$ per `[Milgrom Segal 2002]` *"Envelope theorems for arbitrary choice sets"* (Econometrica 70(2)). Granted — the envelope-theorem treatment at lines 1361–1404 is canonically faithful.

## Strongest attack on blue's core defense

Blue's defense item 3 closes with: "the manuscript names the surrogate distinction every time it uses the surrogate." This is the blue falsification trigger (b): "If any subsequent §Theory derivation silently substitutes $\nabla\langle E\rangle_\beta$ for $\nabla\mathcal{F}_{\mathrm{red}}$ at the gradient level without re-flagging, sub-claim 1 fails."

Manuscript line 1967 contains exactly this substitution, silently:

> "The second-variation expressions below treat the attention weights $\beta_{ij}$ as fixed at their softmax-equilibrium values, equivalent to the envelope-theorem convention for analyzing the local geometry of equilibria. Off-equilibrium, full autograd through the softmax produces additional cross terms involving $\partial \beta_{ij}/\partial \mu$ and $\partial \beta_{ij}/\partial \Sigma$; these are subdominant in regimes of soft attention and consensus alignment, and dominant in regimes of sharp attention or rapid reorganization. The boxed forms below are stated in the isolated-agent limit $\beta_{ij} = 0$ where the distinction does not appear."

This *names* the distinction but immediately states the boxed forms apply *only* in the $\beta_{ij} = 0$ isolated-agent limit. The mass-analogy reading at lines 2046–2056 ("effective mass" formula at Eq.~\eqref{eq:effective_mass}) then writes:

$$M_i = \bar{\Lambda}_{p_i} + \sum_k \beta_{ik}\tilde{\Lambda}_{q_k} + \sum_j \beta_{ji}\Lambda_{q_i} + \Lambda_{o_i}$$

with $\beta_{ik}$ and $\beta_{ji}$ *not* set to zero (otherwise the "incoming relational" and "outgoing recoil" terms vanish and the analogy collapses). The boxed mass-matrix forms are stated for the $\beta_{ij} = 0$ isolated-agent limit (line 1967), but the load-bearing "effective mass" identification at Eq.~\eqref{eq:effective_mass} retains the $\beta_{ik}$ and $\beta_{ji}$ contributions. This is the gradient-level surrogate substitution blue's falsification trigger (b) names: the equation written down (Eq.~\eqref{eq:effective_mass}) uses the *non-zero-$\beta$ full autograd* form to populate "incoming/outgoing relational" while the *legitimacy of the box* is stated only for $\beta_{ij} = 0$. The covariance correction $-\tau^{-1}\mathrm{Cov}_{\beta^*}(E, \nabla_x E)$ named at line 1404 is not carried through; the surrogate-vs-reduced distinction is collapsed.

The asymmetric-attention reading at line 1994 reinforces the problem: "if attention weights $\beta_{ij}$ are treated as instantaneous and frozen-asymmetric ($\beta_{ik}\neq\beta_{ki}$) within the dynamics, the resulting flow is no longer the gradient flow of $\mathcal{F}$ alone, and the bilinear form $M_{\mu\mu}$ does not function as the inertia tensor of a conservative Hamiltonian." The manuscript names the failure, then applies the boxed mass form to the asymmetric-attention regime under the "Lyapunov / dissipative" relabel. But the Lyapunov reading does not restore the envelope-theorem equivalence — it abandons the conservative-Hamiltonian reading entirely. The manuscript wants both: the "effective mass" identification (which requires the Hessian-as-mass conservative reading) *and* the asymmetric-attention regime (which the manuscript itself says rules out conservative Hamiltonian readings). The two are inconsistent under standard Lagrangian mechanics `[Arnold1989 Ch. 5 §22–25]`.

## Strongest defense against blue's strongest attack

Blue's defense item 3 attacks only the question of whether the Gaussian KL and the envelope theorem are stated correctly. Both are stated correctly *at the point of statement*. The defense to strengthen is the canonical-fidelity strike on "attention is a geometric necessity rather than an architectural choice" (manuscript line 1411, by reference).

Per `[Cencov1972]` and `[BauerBruverisMichor2016]` *"Uniqueness of the Fisher–Rao metric on the space of smooth densities"* (Bull. London Math. Soc. 48:499–506), the Fisher metric is uniquely characterized as *the invariant Riemannian metric on a statistical manifold* up to scalar — but *no analogous uniqueness theorem* holds for the *functionals* built on it. The softmax-from-Lagrangian derivation at lines 1272–1287 recovers the softmax from an entropy-regularized soft-assignment Lagrangian; per `[Cuturi2013]` *"Sinkhorn distances"* (NeurIPS 2013), the same softmax form is recovered from many starting energies (squared distance, Wasserstein, entropy-regularized OT). The "geometric necessity" framing is therefore a category claim, not a uniqueness theorem. Blue's defense item 3 does not attack this strike.

## Newly-discovered canon

None beyond what the Phase-2 red harvest recorded. `[BauerBruverisMichor2016]`, `[Cuturi2013]`, `[PachecoSasai2024]` (arXiv:2402.05014), and `[Amari2016]` (Springer AMS 194) are in `01b_extended_evidence.md`.
