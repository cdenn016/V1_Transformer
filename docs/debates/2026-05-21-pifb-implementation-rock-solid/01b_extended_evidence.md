# Extended evidence pack — pifb-implementation-rock-solid

Harvested canon from Phase 2 (opening). Both red and blue panels' newly-discovered canon, deduplicated. Judges read this in addition to `01_evidence.md`.

## Round 2 (opening) — red panel additions

### Differential geometry / Lie groups

- **Milnor, "Curvatures of Left Invariant Metrics on Lie Groups", *Advances in Mathematics* 21 (1976) 293–329.** A connected Lie group admits a bi-invariant Riemannian metric iff isomorphic to $G_c \times A$ with $G_c$ compact and $A$ abelian (standard result of the paper; section/lemma number to be confirmed against the printed source). Consequence for PIFB line 2160: $\mathrm{GL}^+(K)$ for $K \ge 2$ is non-compact and not of this form, so no bi-invariant Riemannian metric exists. The manuscript's Karcher barycenter at Eq. 2156 has no canonical construction on the gauge group used in any non-compact regime; the substitute is unspecified and "is a modeling decision the present implementation does not adjudicate" (line 2160).
- **Nakahara, *Geometry, Topology and Physics* (2nd ed., 2003), §10.3 "Parallel Transport and Holonomy"; §11.1.** Parallel transport on a principal $G$-bundle $\pi: P \to M$ is the horizontal lift of a curve on the base manifold $M$ with respect to a connection 1-form $A \in \Omega^1(P; \mathfrak{g})$. A frame-change $U_i U_I^{-1}$ between two frames is *not* parallel transport unless an underlying connection identifies it as such. Relevant to PIFB lines 2247 ($\Omega_{i,I}[q_I^{(s+1)}]$ cross-scale shadow) and 2254 ($\Omega_{i,I}$ "written as products of gauge-frame exponentials in the canonical form $U_i U_I^{-1}$").
- **Kobayashi, Nomizu, *Foundations of Differential Geometry* Vol. I (1963), §II.7 "Holonomy Group"; §III.2.** A connection with curvature $F = 0$ on a contractible base has trivial holonomy. An identity-copy substitute $\Omega_{i,I} = I$ corresponds to a trivial connection and carries no parallel-transport content. Relevant to PIFB line 2284's "frame-trivial substitute" disclosure.
- **Hall, *Lie Groups, Lie Algebras, and Representations* (2nd ed., 2015), §5.3 "The Baker–Campbell–Hausdorff Formula".** $\log(\exp X \exp Y) = X + Y + \frac{1}{2}[X, Y] + \frac{1}{12}\left([X, [X, Y]] - [Y, [X, Y]]\right) + \ldots$. The first-order truncation $\phi_I = \sum_i w_i \phi_i$ at PIFB line 2191 loses the $\frac{1}{2}\sum_{i<j} w_i w_j [\phi_i, \phi_j]$ correction; non-trivial for non-abelian $\mathfrak{gl}(K)$, with bound depending on the spectral radius of $\mathrm{ad}_{\phi_i}$ — unbounded on $\mathfrak{gl}(K)$.
- **Pennec, "Bi-invariant Means on Lie Groups with Cartan-Schouten Connections" (lecture notes, U. Pennsylvania, CIS-6100, 2012).** For non-compact Lie groups admitting no bi-invariant Riemannian metric, the bi-invariant mean is constructed via the Cartan-Schouten 0-connection (an affine connection without a Riemannian metric). Existence is *local*: within a normal convex neighborhood of a point, the iterated fixed-point converges at linear rate. Source: <https://www.cis.upenn.edu/~cis6100/Bi-Invar-Means-Pennec.pdf>.

### Information geometry / IB

- **Chechik, Globerson, Tishby, Weiss, "Information Bottleneck for Gaussian Variables", *JMLR* 6 (2005) 165–188.** The Gaussian-IB closed form: optimal $T = AX + \xi$ is a linear projection along the top canonical-correlation directions between $X$ and $Y$, requires *jointly Gaussian random vectors* $X, Y$ in a common Euclidean space (theorem number to be confirmed against the printed source; the §3 framework of the paper develops this). The closed form does not apply when $X$ is a tuple of distributions and group elements as at PIFB line 2131. Source: <https://www.jmlr.org/papers/volume6/chechik05a/chechik05a.pdf>.
- **Tishby, Pereira, Bialek, "The Information Bottleneck Method", *Allerton* (1999), §2.** The IB Lagrangian's variational closure requires a tractable encoder family $p(T \mid X)$. For distributions $X$ and group elements $U$, no canonical encoder family is supplied at PIFB Eq. 2133.
- **Amari, Nagaoka, *Methods of Information Geometry* (AMS Translations, 2000), Ch. 2; §3.4 "The Pythagorean theorem and projection".** KL divergence is the second-order expansion of the Fisher metric. The forward-KL barycenter on an e-flat exponential family is the m-projection onto the family; the unique minimizer is the moment-matched mean. Relevant to PIFB Eq. 2142 (forward-KL barycenter) and Eqs. 2181, 2184 (implementation drops the dispersion correction from the m-projection).

### Variational inference

- **Blei, Kuckelbirg, Jordan, "Variational Inference: A Review for Statisticians", *JASA* 112 (2017) 859–877, §3.** Closed-form weights for a forward-KL CAVI barycenter are normalized Lagrange-multiplier weights, not exp(-KL) softmin selectors. PIFB line 2129 uses the normalized $w_i^I$ form; line 2187 uses the unnormalized $\chi_i \exp(-\mathrm{KL})$ form; the relation between the two is asserted, not derived.
- **Bishop, *Pattern Recognition and Machine Learning* (2006), §10.7.** Moment-matched mixture barycenter for Gaussians: $\mu_I = \sum_i w_i \mu_i$, $\Sigma_I = \sum_i w_i [\Sigma_i + (\mu_i - \mu_I)(\mu_i - \mu_I)^\top]$. Dropping the dispersion correction (PIFB line 2179) is a deviation from the moment-matched barycenter; the "$\mathcal{O}(\varepsilon)$" approximation is regime-dependent.
- **Friston, Parr, de Vries, "The Graphical Brain: Belief Propagation and Active Inference", *Network Neuroscience* 1 (2017) 381–414, §3.** Hierarchical variational inference passes the full posterior $q(s_\ell)$ between levels; a deterministic point-passing scheme (PIFB Eq. 2247: $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ — prior = transported posterior) is *not* standard hierarchical variational inference, per the `external_canon_inference.md` §3 pitfall.
- **Bissiri, Holmes, Walker, "A general framework for updating belief distributions", *JRSSB* 78 (2016) 1103–1130, §2.** Tempered Bayes: $p(\theta | y) \propto p(\theta) \exp[-\lambda L(\theta, y)]$ with $\lambda$ a *learning rate*. The Ouroboros fragment at PIFB Eq. 2270 uses $\rho^k$ as a *geometric discount across generations*, not a learning rate. The citation at line 2275 needs a closer match (e.g., discounted infinite-horizon Bayes).

### Renormalization group

- **Wilson, "Renormalization Group and Critical Phenomena", *Phys. Rev. B* 4 (1971) 3174–3183.** Defining requirements of an RG analysis: rescaling transformation $\mathcal{R}_b$ acting on the action space, iteration to find fixed points $\mathcal{R}_b S^* = S^*$, linearization around fixed points to compute critical exponents. PIFB line 2197 disclaims all three; the "scale invariance of the functional form" claim at PIFB line 2210 is necessary but not sufficient for any of Wilson's three ingredients.
- **Wilson, "The Renormalization Group and Critical Phenomena", Nobel Lecture, 8 December 1982.** States that the substantive content of RG is fixed-point location and critical-exponent computation; parametric form-invariance of the action under local rescaling is necessary but not sufficient. Source: <https://www.nobelprize.org/uploads/2018/06/wilson-lecture-2.pdf>.
- **Cardy, *Scaling and Renormalization in Statistical Physics* (1996), Ch. 3.** RG procedure: coarse-grain, rescale, identify fixed points, compute critical exponents from the linearized flow. The threshold detector at PIFB line 2174 is not a coarse-graining map in the Wilson sense; it is a candidate-selection surrogate.

### Non-equilibrium thermodynamics

- **de Groot, Mazur, *Non-Equilibrium Thermodynamics* (1962), Ch. III.** Canonical non-equilibrium indicators: entropy production rate $\sigma = \sum_a J_a X_a$ (sum over thermodynamic fluxes $J_a$ and conjugate forces $X_a$), dissipation function, Onsager reciprocal relations. The aggregate indicator $E_{\mathrm{score}} = (\Phi_E + \Phi_I + V_\nabla)/3$ at PIFB line 2301 is an engineered linear combination, not a canonical entropy-production rate.
- **Glansdorff, Prigogine, *Thermodynamic Theory of Structure, Stability and Fluctuations* (1971).** Excess entropy production rate as a stability criterion for non-equilibrium macroscopic states. The "$E_\mathrm{score} > 1$" threshold at PIFB line 2301 has no canonical precedent in the Glansdorff-Prigogine framework.

### Philosophy of science

- **Popper, *The Logic of Scientific Discovery* (1959), §6.** Falsifiability requires specifying the conditions under which the theory would be false, *before* the experiment. A claim qualified by "we expect, though do not directly measure" (PIFB line 2228) while simultaneously asserting "the whole becomes qualitatively different from the sum of its parts" has emptied the falsification condition.
- **Lakatos, *The Methodology of Scientific Research Programmes* (1978), §1.4.** A research programme that protects its core by ad-hoc auxiliary hypotheses (e.g., "the simulator realization is deferred to a follow-up", PIFB line 2284) is degenerating, not progressive.

### Simulator code (primary source for sub-claim 6)

- **`C:\Users\chris and christine\Desktop\MAgent_Model-main\gauge_agent\meta_agents.py:55–66`** (`ConsensusDetector.belief_coherence`): returns `1.0 - E` where `E` is mean post-transport KL. The `1 - KL` form. PIFB line 2174 explicitly rejects this form.
- **`meta_agents.py:68–80`** (`ConsensusDetector.model_coherence`): returns `1.0 - E` analogously. Same `1 - KL` form.
- **`meta_agents.py:82–91`** (`ConsensusDetector.consensus_score`): returns `C_b * C_m`. Two factors, not the three-factor $P \cdot C_q \cdot C_s$ of PIFB line 2174.
- **`meta_agents.py:93–129`** (`ConsensusDetector.find_clusters`): thresholds on `gamma = C_b * C_m` at line 106 (`adj = (gamma > self.gamma_min).float()`). No spatial-overlap gating; the presence factor $P$ is not applied.
- **`meta_agents.py:49`** (`ConsensusDetector.__init__`): `gamma_min: float = 0.5`. Threshold value matches manuscript; threshold object does not.
- **`meta_agents.py:10–13`** (module docstring): claims `Γ = C_belief · C_model · Presence > Γ_min` (three factors). Drifts from the code at line 91 (two factors). Code-vs-comment drift; the code is canonical per CLAUDE.md "CODE FOCUS" policy.
- **`meta_agents.py:167–399`** (`MetaAgentFormation.form_meta_agent`): the bottom-up aggregation step. Uses `transport_mean(omega_ij, …)` and `transport_covariance(omega_ij, …)` — the sandwich product $\Omega \Sigma \Omega^\top$, structurally faithful to PIFB lines 2181–2186. Drops the dispersion term per PIFB line 2179. Rejects the precision-weighted product-of-experts alternative per PIFB line 2187. This bottom-up aggregation is structurally faithful; the consensus detector that *gates* the aggregation is not.

The cluster-formation step is mismatched; the post-formation aggregation step is matched. Sub-claim 6 falsified at the gating step.

## Round 2 (opening) — blue panel additions

### Differential geometry / Lie groups

- **Karcher, "Riemannian center of mass and mollifier smoothing", *Comm. Pure Appl. Math.* 30 (1977) 509-541.** Existence-uniqueness of the Karcher mean on a Riemannian manifold requires a convex normal ball of radius bounded by half the injectivity radius. For compact $\mathrm{SO}(N)$ with the bi-invariant metric, the injectivity radius is $\pi$, so balls of radius $< \pi/2$ suffice. Defends the PIFB line 2160 compact-group scoping.
- **Pennec, "Statistical Computing on Manifolds: From Riemannian Geometry to Computational Anatomy", in *Emerging Trends in Visual Computing* (LNCS 5416, 2009): 347-386.** Standard reference enumerating the polar-decomposition and SPD-restricted substitutes for non-compact $\mathrm{GL}^+(K)$ Karcher means — exactly the two candidate substitutes PIFB line 2160 acknowledges without adjudicating.
- **Bonnabel, Sepulchre, "Riemannian metric and geometric mean for positive semidefinite matrices of fixed rank", *SIAM J. Matrix Anal. Appl.* 31(3) (2009): 1055-1070.** SPD-restricted Karcher-mean construction; one of the two substitutes PIFB line 2160 enumerates.
- **Moakher, "Means and averaging in the group of rotations", *SIAM J. Matrix Anal. Appl.* 24(1) (2002): 1-16.** Closed-form Frechet/Karcher mean computations for $\mathrm{SO}(3)$ — the compact regime PIFB line 2160 actually uses in simulations.
- **Helgason, *Differential Geometry, Lie Groups, and Symmetric Spaces* (Academic Press, 1978), §III.6.** Killing form $B(X, Y) = 2K \mathrm{tr}(XY) - 2\mathrm{tr}(X)\mathrm{tr}(Y)$ on $\mathfrak{gl}(K, \mathbb{R})$ is indefinite for $K \geq 2$. Corroborates PIFB line 2160's "no bi-invariant Riemannian metric exists" claim for $\mathrm{GL}^+(K)$.

### Information geometry / pooling

- **Amari, "Integration of stochastic models by minimizing $\alpha$-divergence", *Neural Computation* 19(10) (2007): 2780-2796.** The $\alpha$-divergence-weighted barycenter unifies product-of-experts ($\alpha = 1$, m-flat) and mixture ($\alpha = -1$, e-flat). The forward-KL barycenter at PIFB Eq. 2141 is the $\alpha = -1$ case.
- **Neal, Hinton, "A view of the EM algorithm that justifies incremental, sparse, and other variants", in *Learning in Graphical Models* (1998): 355-368.** Incremental-EM framework justifying the fixed-point iteration at PIFB line 2187.
- **Genest, McConway, "Allocating the weights in the linear opinion pool", *J. Forecasting* 9(1) (1990): 53-73.** Treats data-driven weight allocation in pooling; the coherence-weighted form $w_i \propto \chi_i \exp(-\mathrm{KL})$ at PIFB line 2187 is a standard data-driven weight choice.
- **Heskes, "Selecting weighting factors in logarithmic opinion pools", *NIPS* 10 (1998): 266-272.** Unequal-weight log-linear pools; the geometric-decay $\lambda_k = \lambda_0 \rho^k$ at PIFB line 2275 is consistent with the Heskes treatment.

### Variational inference / information bottleneck

- **Slonim, Tishby, "Document clustering using word clusters via the information bottleneck method", *SIGIR* (2000): 208-215.** Agglomerative IB algorithm — the canonical clustering procedure with which the manuscript's threshold-detector hierarchy could be compared (PIFB line 2138 enumerates this comparison as one of three unsupplied IB ingredients).
- **Achille, Soatto, "Information dropout: Learning optimal representations through noisy computation", *IEEE TPAMI* 40(12) (2018): 2897-2905.** Independent corroboration of the Tishby sign convention $\mathcal{L} = I(T;X) - \beta I(T;Y)$ at PIFB line 2133.
- **Beal, *Variational Algorithms for Approximate Bayesian Inference* (PhD thesis, 2003), §3.3.** Variational EM as coordinate ascent on $\mathcal{F}$ — alternating maximization with respect to $q$ and model parameters. Standard framework consistent with the FE-improvement criterion at PIFB Eq. 2123.

### Philosophy of science

- **Bogen, Woodward, "Saving the Phenomena", *Philosophical Review* 97 (1988): 303-352.** Data-vs-phenomena distinction; relevant to PIFB line 2228's "we expect, though do not directly measure" disclosure — measured data vs theoretically-expected phenomena.
- **Hooker, "The Hardware Lottery", *CACM* 64(12) (Dec 2021): 58-65, <https://cacm.acm.org/research/the-hardware-lottery/>.** Tractability-driven research-direction selection should be disclosed; PIFB line 2174's "computationally expensive [continuous-time evaluation]... threshold-based detector as a practical surrogate" is exactly this kind of Hooker-style disclosure.
- **Sculley et al., "Hidden Technical Debt in Machine Learning Systems", *NeurIPS* (2015), §5.** Disclosure of the simulator-vs-transformer-codebase split at PIFB line 2284 is consistent with the Sculley framework's call for explicit technical-debt accounting.

### Simulator code (additional findings beyond evidence pack)

- **`MAgent_Model-main/gauge_agent/meta_agents.py:343-359`** (`MetaAgentFormation.form_meta_agent` frame averaging): computes `omega_avg = (w_q_b_om * omega_stack).sum(dim=0)` (line 358) — an *extrinsic* weighted Euclidean mean of group elements, not the Lie-algebra-additive form $\phi_I = \sum w_i \phi_i$ specified at PIFB line 2191. The docstring at lines 344-355 explicitly notes "the previous docstring claim of 'Lie-algebra-additive average' was wrong; corrected here." Second concrete code-vs-manuscript divergence beyond the consensus detector identified in `01_evidence.md`.
- **`meta_agents.py:217-238`** (`MetaAgentFormation.form_meta_agent` transport pass): correctly calls `transport_covariance(omega_ij, agent.sigma_q)` (line 230) — the two-sided sandwich $\Omega \Sigma \Omega^\top$. Confirms partial match with PIFB Eq. 2145, 2184.
- **`meta_agents.py:290-321`** (`MetaAgentFormation._fixed_point`): implements the saddle-point coherence-weighted iteration `w_raw = chi * torch.exp(-stable)` (line 300) — structurally faithful to the manuscript's $w_i^I(x) = \chi_i(x)\exp[-\mathrm{KL}(q_i^{(s)} \| \bar{q}_I^{(s)})]$ at line 2187. The fixed-point converges under high-coherence as the manuscript claims.


## Round 3 (rebuttal) — red panel additions

### Philosophy of science

- **Popper, *The Logic of Scientific Discovery* (Routledge Classics ed., 2002, original 1959), §15 "Strictly universal and strictly existential statements"** — "Every imaginable basic statement which is incompatible with the theory must be excluded by the theory" (page 70 in the Routledge edition). The falsification condition is *ex ante*; relaxing it after partial falsification is conventionalist twist (§19), not honest disclosure.
- **Lakatos, *Falsification and the Methodology of Scientific Research Programmes* (in *Criticism and the Growth of Knowledge*, ed. Lakatos & Musgrave, Cambridge UP, 1970), §1.** Distinguishes "progressive" from "degenerating" research programmes; ad-hoc auxiliary hypotheses introduced solely to absorb anomalies (without independent testable content) characterize the degenerating case. Source: <https://www.csun.edu/~vcsoc00i/classes/s497f09/s2/Lakatos.pdf>.

### Implementation engineering (primary-source code verification)

- **`MAgent_Model-main/gauge_agent/meta_agents.py:344-355` (in-repo code comment, primary source)** — the simulator code itself admits the manuscript-vs-implementation divergence on frame averaging: "a true intrinsic mean would be `matrix_exp(Σ_j w_j · matrix_log(omega_j))` per manuscript line 1911. The extrinsic form is kept because (a) the downstream consumers always invert via `safe_inv` / `robust_cholesky` so a near-singular average degrades gracefully, and (b) switching averaging semantics would shift downstream trajectories and is gated on a separate manuscript-alignment authorisation. The previous docstring claim of 'Lie-algebra-additive average' was wrong; corrected here." Confirms red's evidence-pack `meta_agents.py:343-359` claim with code-level admission.

### Information geometry

- **Smith, "A generalization of the Bayesian steady forecasting model", *J. Royal Statistical Society B* 41 (1979) 375-387.** Discounted-data Bayesian updating with geometric weights along the temporal axis; the standard predecessor to West-Harrison 1997 §10.7 for the geometric-discount construction PIFB Eq. 2270 actually uses (rather than the dynamic-linear-model discount of §6.3 that PIFB line 2275 cites).
- **West, Harrison, *Bayesian Forecasting and Dynamic Models* (2nd ed., Springer, 1997), §10.7 "Discounted Likelihoods".** The canonical home of the *infinite-horizon discounted updating* construction with geometric weights; closer to PIFB Eq. 2270 than the §6.3 dynamic-linear-model discount that line 2275 cites.
- **Amari, Nagaoka, *Methods of Information Geometry* (AMS Translations, 2000), Ch. 3 §3.5 "Projections and divergence-minimization".** The forward-KL barycenter is the m-projection onto the e-flat exponential family; for Gaussian mixtures the m-projection includes the law-of-total-variance dispersion term. Dropping the dispersion (PIFB line 2179) is a truncation, not an m-projection.

### Gauge theory / Lie groups

- **Moakher, "Means and averaging in the group of rotations", *SIAM J. Matrix Anal. Appl.* 24(1) (2002) 1-16.** §3 enumerates four distinct mean constructions on $\mathrm{SO}(N)$: extrinsic Euclidean (followed by polar projection), projected arithmetic, log-Euclidean (BCH-first-order Lie-algebra-additive), and Karcher (Riemannian). PIFB line 2191 claims log-Euclidean; `meta_agents.py:358` implements extrinsic Euclidean. Two distinct objects.
- **Pennec, Fillard, Ayache, "A Riemannian framework for tensor computing", *Int. J. Computer Vision* 66 (2006) 41-66, §4–§5.** Affine-invariant Riemannian framework on SPD matrices; explicit log-Euclidean mean vs Riemannian mean distinction. The framework PIFB §Theory line 593 invokes for the natural $\mathrm{GL}(K, \mathbb{R})$ setting is closer to the affine-invariant SPD construction than to the compact-Lie-group Karcher mean; the substitute the §Implementation caveat at line 2160 "does not adjudicate" has a canonical literature, contra blue's "not adjudicated" framing.

### Variational inference (primary-source IB citations)

- **Tishby, Pereira, Bialek, "The Information Bottleneck Method", *Allerton* (1999), §2.** Preprint <https://arxiv.org/abs/physics/0004057>. The IB Lagrangian variational closure requires a tractable encoder family $p(T \mid X)$ and a joint distribution $p(X, Y)$. PIFB $X = \{q_i, s_i, U_i\}_{i \in I}$ has no specified tractable encoder family; this is ingredient (a) at PIFB line 2138.
- **Chechik, Globerson, Tishby, Weiss, "Information Bottleneck for Gaussian Variables", *JMLR* 6 (2005) 165-188, §3 Theorem 3.1.** Closed-form Gaussian-IB requires *jointly Gaussian random vectors $X, Y$ in a common Euclidean space*. PIFB parent-state $T = (q_I, s_I, U_I)$ is not such an object; Theorem 3.1 does not apply at PIFB line 2138 as cited.

## Round 3 (rebuttal) — blue panel additions

### Philosophy of science

- **Hempel, "Deductive-Nomological vs. Statistical Explanation", in *Minnesota Studies in the Philosophy of Science* III, ed. Feigl & Maxwell (Univ. of Minnesota Press, 1962): 98-169.** Distinguishes explanation-by-derivation from explanation-by-illustration; toy models live in the latter category and have their own genre-specific publication standard. PIFB line 2106's "toy model demonstrating possibility, not a claim about physical reality" is canonical Hempel-style illustrative explanation.
- **Frigg, Hartmann, "Models in Science", *Stanford Encyclopedia of Philosophy* (Spring 2020 edition revision), §3 "Ontology of models".** Distinguishes "models of phenomena" from "models of theory" from "exploratory / toy models." PIFB §Implementation is in the third category by its own declaration at line 2106. Source: <https://plato.stanford.edu/entries/models-science/>.

### Variational inference

- **Wainwright, Jordan, "Graphical Models, Exponential Families, and Variational Inference", *Foundations and Trends in Machine Learning* 1(1-2) (2008): 1-305, §3.** Canonical reference for the principled-object-with-tractable-surrogate pattern. Section 3 develops the mean-field surrogate as a deliberate departure from the exact posterior with explicit error analysis — the foundational move of VI. PIFB line 2167 (FE-improvement is computationally expensive; threshold detector is the practical surrogate) is in this pattern.
- **Jordan, Ghahramani, Jaakkola, Saul, "An Introduction to Variational Methods for Graphical Models", *Machine Learning* 37(2) (1999): 183-233, §3.** Original Saul-Jaakkola-Jordan-Ghahramani VI framework. The pattern "write principled object; introduce surrogate when intractable; disclose the gap" is the founding move of mean-field VI; PIFB §Implementation enacts it correctly.

### Information geometry

- **Tishby, Zaslavsky, "Deep Learning and the Information Bottleneck Principle", *IEEE Information Theory Workshop* 2015: 1-5.** The IB Lagrangian's *abstract variational formulation* applies to arbitrary $X, Y$ with an appropriate encoder family; the Gaussian closed form of Chechik 2005 is a specialization. PIFB line 2133 cites the Tishby 1999 abstract Lagrangian (in scope); the Chechik 2005 citation at line 2138 is explicitly for the specialization whose non-applicability the manuscript itself enumerates.
- **Strouse, Schwab, "The Deterministic Information Bottleneck", *Neural Computation* 29(6) (2017): 1611-1630.** Generalization of IB to arbitrary deterministic encoders. Supports the reading of PIFB line 2138's "three ingredients" as a well-formed research direction (parametric-family choice + transition kernel + gauge-frame encoder) rather than a citation collapse.

### Gauge theory / Lie groups

- **Atiyah, *Geometry of Yang-Mills Fields* (Lezioni Fermiane, Scuola Normale Superiore, Pisa, 1979), Ch. 2.** Distinguishes "global gauge transformation" (a single group element acting on the whole bundle) from "parallel transport along a curve" (the path-ordered exponential of the connection 1-form). The simulator's $\Omega_{i,I} = U_I U_i^{-1}$ at `meta_agents.py:226-227` is non-trivial frame-change (the former), distinct from identity-copy; calling it "transport" in the Nakahara §10.3 sense (the latter) requires a connection 1-form that PIFB does not provide.
- **Bishop, Crittenden, *Geometry of Manifolds* (Academic Press, 1964), Ch. V §4.** Frame-change vs parallel transport: a frame field $\{e_i\}$ on a manifold admits frame-changes (sections of the frame bundle), which become parallel transport only when the connection 1-form annihilates the relevant horizontal subspace. PIFB does not provide the connection 1-form for the cross-scale identification; the frame-change is nonetheless mathematically substantive content (not identity-copy).

### Simulator code (blue verification beyond evidence pack)

- **`MAgent_Model-main/gauge_agent/meta_agents.py:226-227`** (`MetaAgentFormation.form_meta_agent` cross-scale transport construction): `omega_ij = torch.linalg.solve(agent.omega.data.transpose(-2, -1), ref_omega.transpose(-2, -1)).transpose(-2, -1)`. The transposed-solve gives $\omega_{ij}^\top = (\omega_i^\top)^{-1} \omega_{\mathrm{ref}}^\top$, so $\omega_{ij} = \omega_{\mathrm{ref}} \cdot \omega_i^{-1}$. This is the non-trivial product-of-exponentials form $U_I U_i^{-1}$ that PIFB line 2254 prescribes (with the reference frame as the meta-agent's frame). The simulator does NOT implement the identity-copy substitute that red's Vector 2 entertains; what remains unverified per PIFB line 2284 is whether this frame-change is parallel-transport-of-a-specified-connection in the stricter Nakahara §10.3 sense.
- **`MAgent_Model-main/gauge_agent/meta_agents.py:229-236`** (transport pass): `transport_mean(omega_ij, agent.mu_q.data)` and `transport_covariance(omega_ij, agent.sigma_q)`. The latter is the gauge-equivariant sandwich $\Omega \Sigma \Omega^\top$, matching CLAUDE.md hard constraint and Nakahara §10.3 [Nakahara 2003, §10.3] canonical covariance-transport form.
- **`MAgent_Model-main/gauge_agent/meta_agents.py:256-263, 300`** (χ-presence factor in `form_meta_agent` saddle-point): the per-agent χ values are extracted at lines 256-263 and applied in the coherence weights `w_raw = chi * torch.exp(-stable)` at line 300. The χ-presence factor is therefore in the meta-agent formation pipeline at the second stage, even though it is absent from the consensus-detector gating step at `meta_agents.py:93-129`. The pipeline is two-stage with χ at the second stage; the manuscript prescribes a one-stage detector with all three factors bundled. Structural overlap with mismatched stage placement.

