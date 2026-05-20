# Evidence Pack — pifb-rigorous-theorems

## Manuscript references

### The appendix under debate (lines 4592-4767)

**Opening / scope:**
- `Participatory_it_from_bit.tex:4592-4595` — subsection header with `\label{app:rigorous_rg}`. Scope declaration: "rigorous backbone for the body subsection sec:meta_agent_rg", three logically distinct claims (exact pushforward, closure question, emergence/retention rule), "finite-dimensional throughout; the continuum limit on $\mathcal{C}$ follows by mesh refinement", and "proved on a compact gauge group $G = \mathrm{SO}(K)$ with a bi-invariant Riemannian metric; the noncompact $\mathrm{GL}^+(K)$ case requires a gauge slice or Radon-Nikodym correction."

**State / coarse-graining setup:**
- `:4597-4598` — state and reference measure: $X_s = \{x_i\}_{i \in \mathcal{I}_s}$ with $x_i = (q_i, p_i, U_i, \chi_i)$; transport $\Omega_{ij} = U_i U_j^{-1}$; reference measure $\nu_s$ "assembled from Haar on frames, Lebesgue on means and masks, and a chosen measure on positive-definite covariances."
- `:4600-4613` — coarse-graining map $\mathcal{R}_s$: parent frame as Riemannian Karcher mean (citing Karcher 1977); forward-KL (M-projection) barycenter for belief; explicit moment-matching with between-child dispersion term retained.

**Theorems:**
- `:4618-4625` — **Theorem thm:rg_pushforward** (partition function and observable preservation). Proof: standard pushforward + change-of-variables.
- `:4627-4641` — **Theorem thm:rg_semigroup** (discrete semigroup composition). Proof: pre-image composition.
- `:4643` — adaptive partition note (state-dependent map, not homogeneous flow).
- `:4645-4655` — **Theorem thm:rg_covariance** (gauge covariance under base-local diagonal action). Proof uses bi-invariance, KL invariance under common pushforward, forward-KL barycenter uniqueness.
- `:4660-4677` — **Theorem thm:rg_exact_closure** (augmented-class closure). Schur complement form $A_{\mathrm{eff}} = A_{YY} - A_{Y\xi}A_{\xi\xi}^{-1}A_{\xi Y}$; log-determinant correction $V(Y) = \tfrac{\tau}{2}\log\det'A_{\xi\xi}(Y)$. Cites Wilson 1974, Cardy 1996 as the canonical Wilsonian effective-action correction.
- `:4684-4686` — **Remark** clarifying that strict Gaussian-KL closure is "empty in the operational regime of this manuscript" because $A_{\xi\xi}$ inherits $Y$-dependence from parent Fisher information.

**Closure error analysis:**
- `:4688-4689` — anharmonic remainder $\mathcal{E}_{\mathrm{anh}}$ via Laplace integration (Wong 2001). Magnitude $O(\tau^2 \|H^{-1}\|^3 \|T_3\|^2 + \tau^2 \|H^{-1}\|^2 \|T_4\|)$.
- `:4691-4700` — renormalized inter-cluster transport and holonomy spread $\mathcal{H}_{IJ}$.
- `:4702-4703` — edge-marginal compatibility and mismatch $\delta_I^{\mathrm{marg}}$.
- `:4708-4716` — **Proposition thm:rg_residual** (six-term schematic closure-residual bound):
$$\varepsilon_{s+1} \le C_1\sum_I (V_I^{(q)} + V_I^{(p)})^{3/2} + C_2\sum_{I,J}\mathcal{H}_{IJ} + C_3\sum_I \delta_I^{\mathrm{marg}} + C_4\sum_I m_I^{-1}\|A_{Y\xi}^{(I)}\|^2 + C_5\sum_I \mathcal{E}_{\mathrm{anh}}^{(I)} + C_6\sum_I \mathcal{E}_{\mathrm{nonG}}^{(I)}$$
- `:4718` — proposition-vs-theorem framing: "stated as a proposition rather than a theorem because the constants $C_k$, the closure norm $\|\cdot\|_\mathcal{B}$, and the regularity class of the parent ansatz are not pinned down here, and tightness is not established; a theorem-grade statement would require specifying each."
- `:4720` — three specifications for the theorem-grade upgrade (closure norm $\|\cdot\|_\mathcal{B}$, $C^4$ parent ansatz with bounded $\|F(q_I)\|, \|T_3\|, \|T_4\|$, Wick-contraction normalization).
- `:4722-4735` — **Theorem thm:rg_residual_explicit** (explicit dispersion and anharmonic constants):
$$C_1 = \frac{\sqrt 2}{12} \cdot \frac{\|F(q_I)\|_{\mathrm{op}}^{3/2}}{m_I^{1/2}} \qquad C_5 = \frac{1}{8} \cdot \frac{\|T_4\|_{\mathrm{op}}}{m_I^2} + \frac{5}{24} \cdot \frac{\|T_3\|_{\mathrm{op}}^2}{m_I^3}$$
with $5/24 = 1/12 + 1/8$ as the sum of sunset (1/12) and double-bubble (1/8) two-loop topologies. Cites CoverThomas2006 (Pinsker), BenderOrszag1999, wong2001asymptotic.
- `:4737` — density-vs-integrated reconciliation: relates 1/8 and 5/24 to the Hermite-evaluation $|H_4(0)|=3$, $|H_6(0)|=15$ via $1/8 = 3/24$ and $5/24 = 15/72$ but only at the centerpoint; for radius $R > 1.126$ the sup-norm of $H_n$ exceeds the central value, so the centerpoint identification fails as an $L^\infty$ bound.

**Detector retention:**
- `:4739-4740` — bounded-exponential detector $\Gamma_I = P_I C_q(I) C_p(I) \in [0, 1]$.
- `:4742-4744` — Remark: local derivability of Lipschitz bound under closure-ansatz conditions (i)-(v); Lipschitz constants bounded by parent Fisher information (cites Amari2016); saving term identified with constrained spectral gap.
- `:4746-4758` — **Theorem thm:rg_detector_retention** (positive retention gain). Proof: Lipschitz bound + detector-condition + arithmetic.

**Adiabatic + finite-size + summary:**
- `:4760-4761` — adiabatic elimination / slow manifold via Fenichel 1979 normal-hyperbolicity.
- `:4763-4764` — finite-size scaling: candidate observables ($S_N, \chi_N, M_N, P_\infty$); scaling ansatz; "we register this as a future-work program rather than as a delivered result."
- `:4766-4767` — summary: route from microscopic Gibbs measure through $\mathcal{R}_s$ to exact pushforward parent, then through $\Pi_\mathfrak{M}$ to closed augmented Gaussian theory.

### Prior debate-driven content (relevant cross-references)

- `docs/debates/2026-05-19-prop-1-tighter-quantitative/04_verdict.md` (REMAND) — the prior debate that produced the corrected Wick coefficients 1/8 and 5/24. The current Theorem thm:rg_residual_explicit at line 4722 is the manuscript's response to that verdict.

## Canon excerpts (teams should expand via WebFetch / textbook lookup)

### Measure theory / pushforward canon

- **Bogachev, V. I. (2007)**, *Measure Theory*, Springer. §3.6 covers pushforward measures and change-of-variables.
- **Folland, G. B. (1999)**, *Real Analysis*, 2nd ed., Wiley. §2.6 measure-theoretic change of variables.

### Karcher / Riemannian-mean canon

- **Karcher, H. (1977)**, "Riemannian center of mass and mollifier smoothing," *CPAM* 30(5), 509-541. Already cited at 4605.
- **Pennec, X. (2006)**, "Intrinsic statistics on Riemannian manifolds: basic tools for geometric measurements," *J. Math. Imaging Vis.* 25(1), 127-154. Local uniqueness conditions on convex normal balls.
- **Afsari, B. (2011)**, "Riemannian Lp center of mass: existence, uniqueness, and convexity," *Proc. AMS* 139(2), 655-673. Convexity radius for Riemannian centers of mass.

### Wilsonian RG / effective action canon

- **Wilson, K. G. (1971)**, "Renormalization Group and Critical Phenomena. I. Renormalization Group and the Kadanoff Scaling Picture," *Phys. Rev. B* 4(9), 3174-3183.
- **Wilson, K. G., Kogut, J. (1974)**, "The renormalization group and the $\epsilon$ expansion," *Physics Reports* 12(2), 75-199. Already cited as Wilson1974 at 4669.
- **Cardy, J. (1996)**, *Scaling and Renormalization in Statistical Physics*, Cambridge University Press. Already cited at 4669.
- **Polchinski, J. (1984)**, "Renormalization and effective Lagrangians," *Nucl. Phys. B* 231, 269-295. Wilsonian effective-action canonical form.
- **Zinn-Justin, J. (2002)**, *Quantum Field Theory and Critical Phenomena*, 4th ed., OUP. §10-12 cover the perturbative effective action including the log-det fluctuation correction.

### Laplace expansion / Wick contraction canon

- **Wong, R. (2001)**, *Asymptotic Approximations of Integrals*, SIAM. Already cited at 4689, 4706, 4734.
- **Bender, C. M., Orszag, S. A. (1999)**, *Advanced Mathematical Methods for Scientists and Engineers I*, Springer. §6.5 covers the saddle-point method with explicit Wick coefficients. Already cited at 4734.
- **Erdélyi, A. (1956)**, *Asymptotic Expansions*, Dover. Classical reference.
- **Hörmander, L. (1983)**, *The Analysis of Linear Partial Differential Operators I*, Springer. §7.7 oscillatory integrals, stationary phase.

### Pinsker inequality canon

- **Cover, T. M., Thomas, J. A. (2006)**, *Elements of Information Theory*, 2nd ed., Wiley-Interscience. Pinsker's inequality. Already cited as CoverThomas2006 at 4729.
- **Csiszár, I. (1967)**, "Information-type measures of difference of probability distributions and indirect observations," *Studia Sci. Math. Hungar.* 2, 299-318. The original Pinsker-Csiszár inequality $\|p-q\|_{TV}^2 \le \frac{1}{2}\mathrm{KL}(p\|q)$ — the $\tfrac{1}{\sqrt 2}$ factor at line 4729 follows from this. Verify the chain.

### Edgeworth expansion canon

- **Hall, P. (1992)**, *The Bootstrap and Edgeworth Expansion*, Springer. Standard reference for Edgeworth coefficients.
- **Petrov, V. V. (1975)**, *Sums of Independent Random Variables*, Springer.

### Slow manifold / Fenichel canon

- **Fenichel, N. (1979)**, "Geometric singular perturbation theory for ordinary differential equations," *J. Diff. Eq.* 31(1), 53-98. Already cited at 4761.
- **Jones, C. K. R. T. (1995)**, "Geometric singular perturbation theory," in *Dynamical Systems*, Springer LNM 1609.

### Information geometry canon

- **Amari, S. (2016)**, *Information Geometry and Its Applications*, Springer. Already cited as Amari2016 at 4743.
- **Amari, S., Nagaoka, H. (2000)**, *Methods of Information Geometry*, AMS.

## What this evidence does NOT settle

1. **Augmented-class closure iteration.** Theorem thm:rg_exact_closure presents $V(Y) = \tfrac{\tau}{2}\log\det' A_{\xi\xi}(Y)$ as "the canonical effective-action correction in the Wilsonian sense" but does not explicitly state that this augmentation iterates: at scale $s+2$, integrating out further internal modes around the new parent saddle produces another log-det term, and so on. Whether the "augmented Gaussian multi-agent class" is closed under iteration or grows at each step is a non-trivial Wilsonian question. The Remark at 4684-4686 says "the augmentation by the explicit log-determinant potential $V(Y)$ is the canonical Wilsonian effective-action correction at the Gaussian saddle and is itself iterable" — but "iterable" doesn't mean the class is fixed; it means the form is preserved. Verify against Cardy 1996 or Polchinski 1984.

2. **Gauge covariance under the compact-vs-noncompact distinction.** Theorem thm:rg_covariance proves the covariance result on $G = \mathrm{SO}(K)$ with bi-invariant Riemannian metric. The body's framework uses $\mathrm{GL}(K)$ or $\mathrm{GL}^+(K)$, which is noncompact. The opening at 4595 admits "the noncompact $\mathrm{GL}^+(K)$ case requires a gauge slice or Radon-Nikodym correction, as flagged in the body." But the rest of the appendix proves theorems on the compact group; the reader of body subsection sec:meta_agent_rg (which invokes the rigorous backbone) is told the compact case is proven. Whether this restriction is properly communicated to body-section readers is an editorial question.

3. **Karcher mean uniqueness conditions.** Theorem thm:rg_covariance assumes "locally unique Karcher means" but the standard convexity-radius result (Pennec 2006, Afsari 2011) gives uniqueness only on convex balls of radius below half the injectivity radius. The 4601 coarse-graining definition assumes "local frame coherence on a convex normal ball of $G$" — this needs to be of small-enough radius. Whether the local-frame-coherence condition implies the Pennec-Afsari convexity radius condition is not stated.

4. **C_1 Pinsker-Edgeworth chain.** The Theorem thm:rg_residual_explicit derivation of $C_1 = \tfrac{\sqrt 2}{12}\|F(q_I)\|^{3/2}/m_I^{1/2}$ via the Pinsker constant $\tfrac{1}{\sqrt 2}$ composed with the third-cumulant Edgeworth coefficient $\tfrac{1}{6}$ gives $\tfrac{1}{6\sqrt 2} = \tfrac{\sqrt 2}{12}$, which is arithmetically correct. The Taylor-route equivalent $C_1 = (\sqrt 2/3)\|T_3\|_{\mathrm{op}}/m_I^{3/2}$ presented at 4734 should be checked: from $\|T_3\|_{\mathrm{op}}/m_I^{3/2}$ to $\|F(q_I)\|^{3/2}/m_I^{1/2}$ requires the relation $\|T_3\| \sim \|F\| / m_I^{1/2}$ ... or similar dimensional matching. Verify the equivalence at the Gaussian saddle.

5. **C_5 Wick-coefficient breakdown.** The 5/24 = 1/12 + 1/8 decomposition is presented as "the sunset (1/12) and double-bubble (1/8) two-loop topologies of the leading saddle-point correction." Bender-Orszag §6.5 should confirm these are the canonical Wick-graph coefficients at two loops in the integrated free-energy correction. Verify against the standard reference.

6. **Detector retention Lipschitz hypothesis.** Theorem thm:rg_detector_retention requires a Lipschitz bound $\mathcal{F}_{\mathrm{parent}}^*(I) \le \mathcal{F}_{\mathrm{micro}}^*(I) - A_I + L_q V_I^{(q)} + L_p V_I^{(p)} + \varepsilon_I$ as hypothesis. The Remark at 4742-4744 says this is "locally derivable under the closure-ansatz conditions (i)-(v)." But conditions (i)-(v) are the same conditions used to establish the closure-ansatz residual itself. The argument structure is: assume closure-ansatz conditions hold → derive Lipschitz bound → invoke detector to conclude retention. This is internally consistent but the Remark could be read as circular if the closure conditions are themselves nontrivial. The "locality of the derivation is the regime in which the closure ansatz itself holds" at 4744 is the honest registration.

7. **Adiabatic / Fenichel hypothesis.** The Fenichel 1979 invocation at 4761 requires normal hyperbolicity, which the manuscript identifies with the constrained internal-mode gap $\lambda_{I,w} > 0$. Whether normal hyperbolicity in the Fenichel sense is equivalent to a spectral-gap condition on the constrained Hessian is the canonical mapping; verify against Jones 1995 §1.

8. **Finite-size scaling deferral.** The 4763-4764 paragraph correctly registers finite-size scaling as future work; the body subsection sec:meta_agent_rg accordingly uses "reorganization event" or "phase-transition-like event" rather than "phase transition." This is a self-consistency check: verify the body uses the weaker language.

Teams should verify points 1-7 against the canon. Point 8 is a manuscript-internal consistency check.
