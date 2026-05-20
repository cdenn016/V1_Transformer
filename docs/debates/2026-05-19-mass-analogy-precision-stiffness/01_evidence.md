# Evidence Pack — mass-analogy-precision-stiffness

Neutral fact pack. No editorial framing. Agents should treat every item as something both sides can read identically.

## Manuscript anchor

Primary anchor: `Attention/Participatory_it_from_bit.tex` lines 1843–2040.

### Section structure

| Line | Element |
|---|---|
| 1843 | `\subsection{Statistical Precision as Configuration-Space Stiffness: A Mass Analogy}` |
| 1844 | `\label{sec:mass}` |
| 1846 | Opening paragraph: explicit framing as "precision-induced configuration-space metric and a Newtonian-shaped harmonic analogy under [kinetic-metric] postulate, not as a derivation of physical inertial mass" |
| 1848 | "What the Hessian gives, and what is added" paragraph: explicit `ω² = k/m` correspondence and disclosure that "Hessian of $\mathcal{F}$ … is therefore, in the first instance, a *stiffness* on belief configuration space, not a mass." Names §`sec:velocity_quadratic` as the postulate location. |
| 1850 | `Setup and Notation` subsubsection |
| 1858–1863 | Eq. `eq:precision_transport`: $\tilde{\Lambda}_{q_k} := (\Omega_{ik}\Sigma_k\Omega_{ik}^T)^{-1} = \Omega_{ik}^{-T}\Lambda_{q_k}\Omega_{ik}^{-1}$. The boxed sender-side identity $\Omega_{ji}^T \tilde{\Lambda}_{q_i}^{(j)} \Omega_{ji} = \Lambda_{q_i}$ is stated and the simpler O(d) simplifications $\tilde\Lambda_k \Omega_{ik} = \Omega_{ik}\Lambda_k$ are flagged as O(d)-only. |
| 1865 | `Extended Free Energy Functional` subsubsection. Eq. `eq:extended_free_energy`: $\mathcal{F} = \sum_i D_{\mathrm{KL}}(q_i\|p_i) + \sum_{i,k}\beta_{ik} D_{\mathrm{KL}}(q_i\|\Omega_{ik}[q_k]) - \sum_i \mathbb{E}_{q_i}[\log p(o_i|\theta)]$. |
| 1874 | `Component Free Energies for Gaussians`. Defers to Eq. `eq:gaussian_kl` (line 530). Sensory term expansion at 1879–1881. |
| 1883 | `First Variations (Gradient)`. Prior, consensus (receiver and sender, with explicit GL precision-transport), sensory partials. The sender-side $\partial/\partial\Sigma_k$ derivation explicitly carries the $-\tilde\Lambda_{q_k}\tilde d_{ik}\tilde d_{ik}^\top \tilde\Lambda_{q_k}$ quadratic-form contribution and explains it vanishes at consensus (line 1904). |
| 1919 | `Second Variations: The Mass Matrix` subsubsection |
| 1923–1927 | Eq. `eq:mass_block_structure`: $M = \begin{pmatrix} M_{\mu\mu} & C^{\mu\Sigma} \\ (C^{\mu\Sigma})^T & M_{\Sigma\Sigma}\end{pmatrix}$. |
| 1929 | `Notation hierarchy` paragraph: distinguishes (i) full $M_{\mu\mu}$ matrix, (ii) prior precision $\Sigma_{p,i}^{-1}$ as leading isolated-agent term, (iii) scalar $m_{\text{eff},i} := \mathrm{tr}([M_{\mu\mu}]_{ii})/K$. |
| 1932 | `Differentiation convention` paragraph: explicit envelope-theorem convention — softmax $\beta_{ij}$ treated as fixed at equilibrium. Validation regime stated as $\beta_{ij}=0$ isolated. |
| 1937 | Mean-sector diagonal block decomposition with the sender-as-receiver-of-$j$ contribution invoking the boxed identity. |
| 1938–1941 | Eq. `eq:mass_mu_diagonal`: $[M_{\mu\mu}]_{ii} = \bar{\Lambda}_{p_i} + \sum_k \beta_{ik}\tilde{\Lambda}_{q_k} + \sum_j \beta_{ji}\Lambda_{q_i} + \Lambda_{o_i}$. |
| 1943–1956 | Eq. `eq:mass_mu_offdiagonal`: $[M_{\mu\mu}]_{ik} = -\beta_{ik}\Omega_{ik}^{-T}\Lambda_{q_k} - \beta_{ki}\Lambda_{q_i}\Omega_{ki}^{-1}$. Includes algebraic step $\tilde\Lambda_{q_k}\Omega_{ik} = \Omega_{ik}^{-T}\Lambda_{q_k}$ on line 1947 and an O(d) reduction on 1956. |
| 1957–1960 | `Off-diagonal block caveats` (`sec:mass_block_caveats`): block-transpose symmetry by Schwarz, asymmetric-attention non-conservativity caveat, reciprocal-attention sufficient condition $\beta_{ik}=\beta_{ki}, \Omega_{ik}\Omega_{ki}=I$. |
| 1962–1973 | Covariance-sector second variation, $\partial^2 D_{\mathrm{KL}}(q_i\|p_i)/\partial\Sigma_i\partial\Sigma_i[V,W] = \frac12\mathrm{tr}[\Lambda_{q_i} V \Lambda_{q_i} W]$ with $\partial(\Sigma^{-1})/\partial\Sigma = -\Sigma^{-1}\otimes\Sigma^{-1}$. |
| 1975–1978 | Eq. `eq:mass_sigma_diagonal`: $[M_{\Sigma\Sigma}]_{ii} = \tfrac12(\Lambda_{q_i}\otimes\Lambda_{q_i})(1+\sum_k\beta_{ik}+\sum_j\beta_{ji})$ at consensus. |
| 1981 | Remark on consensus simplification: sender contribution from $D_{\mathrm{KL}}(q_j\|\tilde q_i)$ explicitly decomposed; sign-flip explanation and at-consensus collapse to $+\tfrac12 \Lambda_{q_i}\otimes\Lambda_{q_i}$. Off-consensus disclaimer. |
| 1983–1991 | Eq. `eq:mass_sigma_offdiagonal`: two-term GL form with outer $\Omega\otimes\Omega$ factors and $\tilde\Lambda$ defined by Eq. `eq:precision_transport`. |
| 1995–2010 | Cross block. Algebraic chain through $\delta\tilde\Lambda_{q_k} = -\Omega_{ik}^{-T}\Lambda_{q_k} V \Lambda_{q_k}\Omega_{ik}^{-1}$. Eq. `eq:cross_block`: $[C^{\mu\Sigma}]_{ik} = 0$ at consensus. |
| 2012–2023 | `Within-Framework Interpretation: Stiffness as Precision (Mass Analogy)` paragraph. Eq. `eq:effective_mass`: $M_i = \bar{\Lambda}_{p_i} + \sum_k\beta_{ik}\tilde\Lambda_{q_k} + \sum_j\beta_{ji}\Lambda_{q_i} + \Lambda_{o_i}$. Explicit "dimensionless on the framework's representational space rather than the kilogram dimension of physical mass." Rock illustration with $\bar\Sigma_{\text{rock}}\approx\epsilon I$. Explicit non-extension to quantum delocalization. |
| 2025–2033 | `Velocity-Quadratic Metric Form` (`sec:velocity_quadratic`). Eq. `eq:full_kinetic`: $\mathcal{M}_{\mathrm{geom}} = \tfrac12 \dot\mu^T M_{\mu\mu}\dot\mu + \tfrac12\mathrm{tr}[M_{\Sigma\Sigma}[\dot\Sigma,\dot\Sigma]] + \tfrac12\langle\dot\phi,\dot\phi\rangle_\mathfrak{g}$. Killing-form caveat for compact $\mathrm{SO}(K)$ vs. indefinite $\mathfrak{gl}(K)$; defers to position-dependent right-invariant form `eq:pullback_metric` for GL case. |
| 2036–2039 | Off-diagonal kinetic coupling $\mathcal{T}_{\text{couple}}$. |

### Cross-document anchors referenced

- Eq. `eq:gaussian_kl` at line 530: $\mathrm{KL}(\mathcal{N}(\mu_1,\Sigma_1)\|\mathcal{N}(\mu_2,\Sigma_2)) = \tfrac12[\log\frac{|\Sigma_2|}{|\Sigma_1|} + \mathrm{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^\top\Sigma_2^{-1}(\mu_2-\mu_1) - K]$. Standard form [BleiKuckelbirgJordan2017].
- §`sec:scope_limitations` (line 133): scope statement that bridging information-geometric quantities to SI units is unresolved.
- §`sec:phenomenological_interpretation` (line 2967): the "consensus-label hypothesis" referenced at line 2014.
- §`sec:framework` (line 181): the F functional that this section's second variation is computed on.
- §`sec:state_dependent_precision` (line 1286): the $\alpha_i$ per-agent precision parameter — distinct symbol from the cluster weights $w_i^I$.
- `eq:pullback_metric` (line 2551) and `eq:gauge_natural_gradient_def` (line 2544): the right-invariant gauge metric the section defers to in the GL case.

## Manuscript references (theory mode)

- `Attention/Participatory_it_from_bit.tex:1843` — section header
- `Attention/Participatory_it_from_bit.tex:1846` — Hessian-vs-mass disclaimer paragraph
- `Attention/Participatory_it_from_bit.tex:1848` — explicit "stiffness, not mass" framing and pointer to `sec:velocity_quadratic`
- `Attention/Participatory_it_from_bit.tex:1858–1863` — Eq. `eq:precision_transport` and the GL identity $\Omega_{ji}^T\tilde\Lambda_{q_i}^{(j)}\Omega_{ji} = \Lambda_{q_i}$
- `Attention/Participatory_it_from_bit.tex:1894–1904` — receiver- and sender-side first-variations including the quadratic-form sender-$\Sigma$ contribution
- `Attention/Participatory_it_from_bit.tex:1938–1941` — Eq. `eq:mass_mu_diagonal`
- `Attention/Participatory_it_from_bit.tex:1947` — algebraic step $\tilde\Lambda_{q_k}\Omega_{ik} = \Omega_{ik}^{-T}\Lambda_{q_k}$
- `Attention/Participatory_it_from_bit.tex:1953–1956` — Eq. `eq:mass_mu_offdiagonal`
- `Attention/Participatory_it_from_bit.tex:1957–1960` — `sec:mass_block_caveats` (asymmetric-attention)
- `Attention/Participatory_it_from_bit.tex:1975–1978` — Eq. `eq:mass_sigma_diagonal`
- `Attention/Participatory_it_from_bit.tex:1981` — at-consensus simplification remark
- `Attention/Participatory_it_from_bit.tex:1985–1991` — Eq. `eq:mass_sigma_offdiagonal`
- `Attention/Participatory_it_from_bit.tex:1995–2010` — cross block derivation and Eq. `eq:cross_block`
- `Attention/Participatory_it_from_bit.tex:2012–2023` — within-framework interpretation paragraph including Eq. `eq:effective_mass` and rock illustration
- `Attention/Participatory_it_from_bit.tex:2025–2039` — `sec:velocity_quadratic` postulate and kinetic coupling

## Canon excerpts (from `.claude/agents/vfe-knowledge/`)

From `external_canon_math.md`:
- KL is the second-order expansion of the Fisher metric at infinitesimal $q \approx p$: $\mathrm{KL} \approx \tfrac12(\Delta\theta)^\top g(\theta)(\Delta\theta) + O(\Delta\theta^3)$.
- Cencov's uniqueness theorem [Cencov1972]: the Fisher metric is the unique (up to scalar) Riemannian metric on a statistical manifold invariant under sufficient statistics.
- For Lie-algebra-valued parameters ($\phi \in \mathfrak{gl}(K)$), the metric depends on structure — do not assume Euclidean. Killing form valid as positive-definite bi-invariant inner product only for compact / semisimple Lie groups; on $\mathfrak{gl}(K)$ the Killing form is indefinite.
- Standard matrix-derivative identity: $\partial(\Sigma^{-1})/\partial\Sigma = -\Sigma^{-1}\otimes\Sigma^{-1}$ [MagnusNeudecker 1999, PetersenPedersen MatrixCookbook §2.2].
- Gaussian KL closed form: $\mathrm{KL}(\mathcal{N}_1\|\mathcal{N}_2) = \tfrac12[\log\frac{|\Sigma_2|}{|\Sigma_1|} + \mathrm{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^\top\Sigma_2^{-1}(\mu_2-\mu_1) - K]$ [BleiKuckelbirgJordan2017, KingmaWelling2014 App. B].

From `external_canon_inference.md`:
- The standard variational free energy [Friston2010, ParrPezzuloFriston2022 Ch. 2] is single-agent (or hierarchical with a single ancestral generative model). Multi-agent generalizations with gauge transport $\Omega_{ij}$ are **user constructions**, not field standard, and must be labeled as novel.
- The active-inference "precision-weighting" idea has $\Sigma_p^{-1}$ as a Bayesian-prior weight on observation evidence, not as a kinetic-energy coefficient. Identifying precision with mass requires an additional postulate.

External canonical forms not in repo but standard:
- Harmonic oscillator: $L = \tfrac12 m\dot x^2 - \tfrac12 k x^2$, $\omega^2 = k/m$ [Goldstein 2002 §1.2; Arnold 1989 §1].
- For a Lagrangian, mass is the coefficient of the kinetic form, stiffness is the Hessian of the potential at a minimum; conflating them requires a metric identification on configuration space [Arnold 1989 §1, §17; Marsden & Ratiu 1999 §1.4].
- Schwarz / Clairaut: mixed partials of a $C^2$ scalar commute, so any Hessian of a scalar functional satisfies $\partial^2 F/\partial x \partial y = \partial^2 F/\partial y \partial x$ — block-transpose symmetry is automatic [Rudin 1976 Theorem 9.41].
- For asymmetric / non-symmetric Hessian-like objects arising from non-gradient dynamical flows, the "mass matrix" reading requires a (possibly auxiliary) Lagrangian / Hamiltonian structure; in general dissipative settings the matrix is a Lyapunov / contraction metric rather than an inertia tensor [Khalil 2002 §4; Lohmiller & Slotine 1998].
- For an SPD-valued covariance $\Sigma$, the Fisher–Rao metric on the Gaussian manifold splits orthogonally between mean and covariance: $g_F = \Sigma^{-1} \oplus \tfrac12(\Sigma^{-1}\otimes\Sigma^{-1})$ [Amari & Nagaoka 2000 §3.5; Calvo & Oller 1990]. This is the second-variation of KL at consensus and matches the diagonal entries in `eq:mass_mu_diagonal` (the isolated-agent leading term $\bar\Lambda_{p_i}$) and `eq:mass_sigma_diagonal` (the unit-prior contribution $\tfrac12 \Lambda_{q_i}\otimes\Lambda_{q_i}$).
- Multi-particle "mass matrix" in classical mechanics: for $T = \tfrac12 \dot q^\top M(q) \dot q$, the matrix $M$ must be symmetric positive-definite to be a valid Riemannian metric and a valid inertia tensor (Routhian / Lagrangian formalism, [Arnold 1989 §22]).

## What this evidence does NOT settle

The evidence pack alone does not settle:

1. Whether the GL-form algebra on line 1947 ($\tilde\Lambda_{q_k}\Omega_{ik} = \Omega_{ik}^{-T}\Lambda_{q_k}$) is correct under all GL(d) — both sides should verify symbolically or by direct substitution from Eq. `eq:precision_transport`.
2. Whether the at-consensus collapse of the covariance-sector sender contribution to $+\tfrac12\Lambda_{q_i}\otimes\Lambda_{q_i}$ (line 1981) is correct — both sides should derive directly from $D_{\mathrm{KL}}(q_j\|\tilde q_i)$ varying $\Sigma_i$.
3. Whether the envelope-theorem convention (line 1933) is a legitimate move — both sides should evaluate against standard envelope-theorem conditions [Milgrom & Segal 2002, Theorem 3] and the regime in which the empirical mass-precision result is validated.
4. Whether the asymmetric-attention caveat (`sec:mass_block_caveats`) adequately disclaims the implications for the "mass matrix" framing, or whether the disclaimer is insufficient given the central role of the mass terminology throughout.
5. Whether the kinetic-metric postulate at `sec:velocity_quadratic` is itself well-motivated, or is an ad hoc reuse to license the Newtonian-shaped empirical scaling.
6. Whether the cross-block-vanishing-at-consensus result (Eq. `eq:cross_block`) is a manuscript statement that holds generally or only under the receiver-side $D_{\mathrm{KL}}(q_i\|\tilde q_k)$ contribution (the sender-side cross-block from $D_{\mathrm{KL}}(q_k\|\tilde q_i)$ is not separately computed in the displayed equations, and the manuscript leaves implicit whether both contributions vanish at consensus).
7. Whether the Killing-form-on-$\mathfrak{gl}(K)$ caveat (line 2033) is correctly disposed of by the deferral to `eq:pullback_metric`, or whether a more pointed reader-facing flag is warranted.

These are the open questions the debate is expected to resolve.
