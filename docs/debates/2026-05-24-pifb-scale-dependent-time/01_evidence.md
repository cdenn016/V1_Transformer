# Evidence Pack — pifb-scale-dependent-time

Neutral fact pack. No side-taking. All line numbers are post-merge (commit 5fb7e3dc).

## Manuscript references (Attention/Participatory_it_from_bit.tex)

- `:2584` — `\subsection{Time as Information Flow}`.
- `:2587` — `\subsubsection{Bit-Counting Time}`. Defines $\Delta\tau_i = \Delta I_i/(1\text{ bit})$, $\Delta I_i = \mathrm{KL}(q_i^{\text{new}}\|q_i^{\text{old}})$. Dimensional disclaimer: $\tau_i$ is a dimensionless count parameter, not SI seconds.
- `:2591` — inserted paragraph: Cramér–Rao distinguishability $g_{\mathcal{B}}(\delta\theta,\delta\theta)\gtrsim 1$, JND ellipsoid, Bateson attribution.
- `:2593–2595` (Eq. `kl_arclength`) — $\mathrm{KL}(q^{\text{new}}\|q^{\text{old}}) = \tfrac12\delta\mu^\top\Lambda\delta\mu = \tfrac12 ds^2$.
- `:2596` — "one JND unit $ds=1$ is $\tfrac12$ bit, not one bit"; bit is reparametrization-invariant.
- `:2608` — `\subsubsection{Fisher Arc Length Along Belief Trajectories}` (`sec:fisher_arc_length`); arc length $\Delta\tau=\int\sqrt{g_{\mathcal{B}}(\dot q,\dot q)}\,d\tau$, positive-definite Riemannian, explicitly NOT identified with relativistic proper time.
- `:2623–2624` — `\subsubsection{Scale-Dependent Resolution and the Meta-Agent Clock}` (`sec:scale_dependent_time`).
- `:2629–2632` (Eq. `bit_production_ratio`) — $Z_I := \mathbb{E}[\Delta\tau_I]/\mathbb{E}[\Delta\tau_i] = \mathrm{tr}(\Lambda\mathrm{Cov}(\Delta\mu_I))/\mathrm{tr}(\Lambda\mathrm{Cov}(\Delta\mu_i))$.
- `:2634–2636` (Eq. `Z_decomposition`) — $Z_I = (\text{coherent fraction}) + O(1/N)$; limits $1$ (coherent), $1/N$ (incoherent).
- `:2638–2647` (Eq. `coupling_laplacian`) — $\sum_{ij}\beta_{ij}\mathrm{KL}(q_i\|\Omega_{ij}[q_j]) \approx \mu^\top(L\otimes\Lambda)\mu$, $L_{ij}=-\beta_{ij}$, $L_{ii}=\sum_j\beta_{ij}$; consensus flow $\dot\mu = -2\eta_\mu(L\otimes I)\mu$; zero mode $\mathbf{1}/\sqrt N$ → barycenter → $Z_I=1/N$ from uniform power over $N$ modes.
- `:2649–2652` (Eq. `Ilow_eq_dispersion`) — $\mathcal{I}_{s\to s+1} = \sum_i \tfrac12(\mu_i-\mu_I)^\top\Lambda(\mu_i-\mu_I) = \tfrac12\sum_{a\ge2}\|\hat\mu_a\|_\Lambda^2$; spectral gap $\lambda_2$ (algebraic connectivity, cite Fiedler1973) sets relaxation rate $2\eta_\mu\lambda_2$ and cluster-formation timescale.
- `:2654` (last para) — renormalization verdict: bit reparametrization-invariant; $\hat\tau^{(s+1)}=\hat\tau^{(s)}/Z^{(s)}$ is comoving units; $Z^{(s)}$ is NOT a dynamical critical exponent ($\sim\xi^z$ at a critical point); promotion is the open RG fixed point flagged at `sec:meta_agent_rg`.

### Supporting (already-present) manuscript pieces the claim leans on

- `:2192` (Eq. `meta_agent_mu_impl`) — barycenter $\bar\mu_I = \sum_i w_i\Omega_{I,i}[\mu_i]/\sum_i w_i$.
- `:2195` (Eq. `meta_agent_sigma_impl`) — averaging covariance $\bar\Sigma_I = \sum_i w_i\Omega\Sigma_i\Omega^\top/\sum w_i$ (NOT product-of-experts; PIFB:2198 declines PoE).
- `:2244` (`sec:emergent_timescale`) — timescale hierarchy $\tau^{(s+1)}>\tau^{(s)}$, three structural mechanisms (averaging, consensus barriers, growing stiffness $[M_{\mu\mu}]_{II}$).
- `:2674` — Gaussian Fisher-Rao metric $g_{\mathcal{B}}(\delta q,\delta q)=\delta\mu^\top\Sigma^{-1}\delta\mu + \tfrac12\mathrm{tr}(\Sigma^{-1}\delta\Sigma\Sigma^{-1}\delta\Sigma)$.

## Canon excerpts (.claude/agents/vfe-knowledge/external_canon_math.md)

- `:35` — "KL is the second-order expansion of the Fisher metric at infinitesimal $q\approx p$: $\mathrm{KL}\approx \tfrac12(\Delta\theta)^\top g(\theta)(\Delta\theta)+O(\Delta\theta^3)$." [AmariNagaoka2000 Ch. 2]. Directly bears on component (B).
- `:42` — Gaussian KL closed form $\tfrac12[\mathrm{tr}(\Sigma_p^{-1}\Sigma_q)+(\mu_p-\mu_q)^\top\Sigma_p^{-1}(\mu_p-\mu_q)-K+\log(|\Sigma_p|/|\Sigma_q|)]$. The manuscript's $\tfrac12\delta\mu^\top\Lambda\delta\mu$ is the equal-covariance mean-only specialization.
- `:16–24` — Fisher information metric definition; Cencov uniqueness [Cencov1972].
- `:53–63` — natural gradient [Amari1998]; for Lie-algebra parameters the metric is not Euclidean.

## What this evidence does NOT settle

- Whether the manuscript's $\mathrm{Cov}(\Delta\mu_I)=\mathrm{Cov}(c)+v_\xi/N$ decomposition (Eq. Z_decomposition) is stated with sufficient assumptions (independence, stationarity, common $\Lambda$) to be a theorem rather than an illustrative limit. The canon is silent on this specific aggregation.
- Whether "incoherent injection projects onto the zero mode with weight $1/N$" requires the noise to be isotropic / the eigenbasis assumption, and whether the manuscript states it. (Graph-Laplacian spectral facts — algebraic connectivity, equipartition of variance over modes — are standard but not in the vfe-knowledge canon; agents should WebFetch Fiedler1973 / a spectral-graph-theory reference, e.g., Chung 1997, if needed.)
- Whether the claim "$Z^{(s)}$ is not a dynamical critical exponent" is correctly hedged. The dynamical critical exponent $z$ and the relation relaxation-time $\sim\xi^z$ are from dynamic-scaling RG [Hohenberg & Halperin, Rev. Mod. Phys. 49, 435 (1977)] — not in the vfe canon; agents should WebFetch if they need the precise definition.
- Whether the manuscript's Laplacian reduction (Eq. coupling_laplacian) is valid given that $\beta_{ij}=\mathrm{softmax}(\cdot)$ is generally asymmetric (the manuscript hedges with symmetrized weights $(\beta_{ij}+\beta_{ji})/2$); whether symmetrization is legitimate here.
- Whether identifying $\mathcal{I}_{s\to s+1}$ (a sum of forward KLs / dispersion) with "nonzero-mode energy" holds beyond the quadratic (Gaussian, equal-covariance) regime.
