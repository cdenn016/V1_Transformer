# Blue Rebuttal — prop-1-tighter-quantitative

## Concession

I concede substantial ground. After executing the sympy verification myself and reading `Attention/Participatory_it_from_bit.tex:4558` (`eq:rg_laplace`) and lines 4560-4561, I cannot defend the specific coefficient values $\tfrac{1}{24}$ and $\tfrac{1}{72}$ in the claim's $C_5$ as upper-bound coefficients on the post-Laplace closure residual under the manuscript's $L^\infty$ closure norm.

The manuscript writes the anharmonic remainder as the *integrated* correction $\mathcal{E}_{\mathrm{anh}}$ appearing in `eq:rg_laplace`:

$$\mathcal{F}_{s+1}(Y_I) = \mathcal{F}_{\mathrm{eff}}(Y_I) + \tfrac{\tau}{2}\log\det{}'H_I^\perp(Y_I) + \mathrm{const} + \mathcal{E}_{\mathrm{anh}},$$

with $\mathcal{E}_{\mathrm{anh}} = O(\tau^2 \|H^{-1}\|^3 \|T_3\|^2 + \tau^2 \|H^{-1}\|^2 \|T_4\|)$ at line 4561. This is the post-Laplace (integrated) anharmonic correction, not the pointwise bare action. Red's executed sympy session (lines 17-51 of `02_red_opening.md`) correctly computes the leading $-\log(Z/Z_0)$ expansion at second order in $(t_3, t_4)$ and finds coefficient $\tfrac{1}{8}$ on $t_4$ and $-\tfrac{5}{24}$ on $t_3^2$. My own verification reproduces this:

```python
# sympy: Gaussian moments <xi^4>_0 = 3, so <(1/24) T_4 xi^4>_0 = T_4/8.
V_anh = sp.Rational(1,24) * T4 * xi**4
shift = sp.integrate(V_anh * phi, (xi, -sp.oo, sp.oo))  # = T4/8
```

The leading post-Laplace correction inherits the Wick coefficients $\tfrac{1}{8}$ and $\tfrac{5}{24}$, which agree with `[BenderOrszag1999 §6.5 Eq. 6.5.4]` (1D) and `[Wong2001 §IX.5 Eq. 5.04]` (multidimensional, where the $\tfrac{5}{24}$ resolves into $\tfrac{1}{12}$ + $\tfrac{1}{8}$ on two separate cubic-cubic topologies). The claim's $\tfrac{1}{24}, \tfrac{1}{72}$ are the Edgeworth-Hermite *density-expansion* coefficients on $H_4$ and $H_6$, which appear in the density correction $p(x) = \phi(x)[1 + h_1(x)]$, not in the integrated $-\log(Z/Z_0)$ correction that the closure norm of `eq:rg_closure` (line 4534-4536) sees after `eq:rg_laplace` has integrated out the internal modes.

I also concede that the claim's $C_1 = \tfrac{1}{12} \|F(q_I)\|^{3/2}_{\mathrm{op}}/m_I^{1/2}$ is the wrong numerical coefficient. The honest Pinsker plus Edgeworth-Hermite-3 chain gives

$$\|P - Q\|_{TV} \le \sqrt{V/2} = V^{1/2}/\sqrt 2 \qquad ([\text{CoverThomas2006 §11.6}]),$$

composed with the cubic Edgeworth-density coefficient $\tfrac{1}{6}$ on $H_3$ ([AmariNagaoka2000 Ch. 8]), yielding

$$\tfrac{1}{6} \cdot \tfrac{1}{\sqrt 2} = \frac{\sqrt 2}{12} \approx 0.118,$$

not the claim's $\tfrac{1}{12} \approx 0.083$. Sympy confirms: `Rational(1,6) * 1/sqrt(2) == sqrt(2)/12`, off from $\tfrac{1}{12}$ by a factor of $\sqrt 2$. The claim's coefficient is too small by a factor of $\sqrt 2$.

These are real concessions. The proposition's specific numerical coefficients in the claim are not defensible as written.

## Core attack

Red's load-bearing structural attack on $C_1$ — the "missing cubic-cumulant tensor" argument at lines 83-87 of `02_red_opening.md` — is incompatible with the manuscript's specification 2 ("$C^4$ multi-agent Gaussian KL functional class on $\mathcal{N}_I$, with bounded Fisher operator norm $\|F(q_I)\|_{\mathrm{op}}$"). Red writes:

> "The $V_I^{3/2}$ correction term in a $L^\infty$ free-energy bound ... comes from the *cubic* term in the KL Taylor expansion, which is $\tfrac{1}{6} C_{ijk}(q_I)\, \delta\mu^i \delta\mu^j \delta\mu^k$ where $C_{ijk}$ is the third cumulant tensor (third derivative of $-\log q$). ... The Fisher operator norm $\|F\|_{\mathrm{op}}$ alone cannot generate a cubic term — the Fisher is the Hessian (quadratic curvature) of $-\log q$ by Cencov's theorem."

Sympy directly falsifies the premise that $C_{ijk}(q_I)$ is the relevant cubic tensor when $q_I$ is Gaussian:

```python
q  = exp(-(x-mu)**2/(2*sigma**2)) / sqrt(2*pi*sigma**2)
neg_log_q = -log(q)            # = log(sigma) + log(2*pi)/2 + (x-mu)^2/(2*sigma^2)
diff(neg_log_q, x, 3)          # = 0
diff(neg_log_q, x, 4)          # = 0
```

For Gaussian $q_I$, all derivatives of $-\log q_I$ at order three and higher vanish. The third cumulant tensor $C_{ijk}$ of $q_I$ is identically zero in spec 2's regularity class, so red's $\tfrac{1}{6}\|C\|_{\mathrm{op}} V_I^{3/2}/m_I^{3/2}$ bound is vacuous (right-hand side equals zero) and cannot be the relevant cubic contribution. Red's structural attack falsely identifies the cubic tensor with $C(q_I)$ rather than with the cubic Taylor coefficient $T_3$ of the *action* $\mathcal{F}_s$ at the parent saddle, which is what the manuscript at line 4561 actually denotes by $T_3$ and what spec 2 actually bounds. The cubic contribution to the $V^{3/2}$ scaling arises from the cubic Taylor of $\mathcal{F}_s$ evaluated at the displaced Fisher-normal coordinate, not from a third cumulant of the parent posterior.

This also rebuts red's dimensional argument at lines 88-89. In Fisher-normal coordinates (which the closure-norm spec 1 explicitly references: "$\mathcal{N}_I$ specified by ... high-coherence Gaussian-deviation region in Fisher-normal coordinates"), the Fisher metric is dimensionless and $\|\delta\mu\|^2_{F(q_I)}$ is the dimensionless squared geodesic distance. The KL is $V \sim \tfrac{1}{2}\|\delta\mu\|^2_{F(q_I)}$ in info units, $\|\delta\mu\|_{F(q_I)} = \sqrt{2V}$ is dimensionless, and the cubic Taylor remainder $|T_3[\delta\mu^{\otimes 3}]| \le \|T_3\|_{\mathrm{op}} (2V)^{3/2}$ is in info units. The proposition's denominator $m_I^{1/2}$ is the dimensional correction for the off-Fisher-normal mode mass. Red's "$\|F\|^{3/2}/m^{1/2}$ has units of $(\text{info})/(\text{length})^2$" calculation assumes a coordinate-free representation where Fisher carries length-squared dimension, which is not the convention `eq:rg_closure` adopts.

## Defense

What survives the concession is the *structural form* of the upgrade, not the specific numerical values. The proposition's quantitative-upgrade program is achievable; the claim's specific coefficients need correction. Three pieces of evidence support this position.

First, the scaling exponents and tensor-structure of $C_5$ are correct. Both my sympy and red's sympy independently confirm that the leading post-Laplace anharmonic correction at the Gaussian saddle decomposes as a sum of a quartic-vertex contribution (one factor of $T_4$, two of $H^{-1}$) and a squared-cubic-vertex contribution (two factors of $T_3$, three of $H^{-1}$). This is `[Wong2001 §IX.5 Eq. 5.04]` and matches the manuscript's existing $\mathcal{E}_{\mathrm{anh}} = O(\tau^2 \|H^{-1}\|^3 \|T_3\|^2 + \tau^2 \|H^{-1}\|^2 \|T_4\|)$ at line 4561. Under $\|H^{-1}\| \le m_I^{-1}$, the bound takes the proposition's form $C_5 = a \|T_4\|_{\mathrm{op}}/m_I^2 + b \|T_3\|_{\mathrm{op}}^2/m_I^3$ for some constants $a, b$. The claim's separation into quartic and squared-cubic terms with these exponents is correct; what is wrong is $(a, b) = (\tfrac{1}{24}, \tfrac{1}{72})$. The correct values under the operator-norm convention are $(a, b) = (\tfrac{1}{8}, \tfrac{5}{24})$.

Second, the scaling form of $C_1$ is achievable under the honest Pinsker+cubic-Taylor derivation, but with $\|T_3\|_{\mathrm{op}}$ replacing the claim's $\|F\|_{\mathrm{op}}^{3/2}$. From $V_I = \tfrac{1}{2}\|\delta\mu\|^2_{F(q_I)} + O(V_I^2)$ (evidence pack line 70), inverting gives $\|\delta\mu\|_{F(q_I)} \le \sqrt{2V_I}$, hence $\|\delta\mu\| \le \sqrt{2V_I/m_I}$ via the Fisher operator norm bound. The cubic Taylor of $\mathcal{F}_s$ at the parent then satisfies

$$|T_3[\delta\mu, \delta\mu, \delta\mu]| \le \|T_3\|_{\mathrm{op}} \|\delta\mu\|^3 \le \|T_3\|_{\mathrm{op}} (2V_I/m_I)^{3/2} = 2^{3/2} \|T_3\|_{\mathrm{op}} V_I^{3/2}/m_I^{3/2},$$

so the cubic contribution to the closure residual scales as $V_I^{3/2}$ with coefficient $\tfrac{2^{3/2}}{6} = \tfrac{\sqrt 2}{3}$ on $\|T_3\|_{\mathrm{op}}/m_I^{3/2}$ when the Taylor remainder is divided by the $3! = 6$ symmetrization factor. The honest $C_1$ is therefore

$$C_1 = \tfrac{\sqrt 2}{3} \cdot \frac{\|T_3\|_{\mathrm{op}}}{m_I^{3/2}} \quad\text{or, in the small-$\|T_3\|$ Edgeworth regime, } \tfrac{\sqrt 2}{12} \cdot \frac{\|F(q_I)\|^{3/2}_{\mathrm{op}}}{m_I^{1/2}}$$

depending on whether one tracks the Taylor remainder of $\mathcal{F}_s$ directly or one passes through the Edgeworth-density representation in standardized Fisher coordinates. The $V_I^{3/2}$ scaling is correct; the specific tensor entering depends on which intermediate inequality the proposition routes through.

Third, the manuscript's specification 1 (parent neighborhood "$\mathcal{N}_I$ specified by the convex normal ball of $G$ together with the high-coherence Gaussian-deviation region in Fisher-normal coordinates") and specification 2 ($C^4$ Gaussian KL functional class with bounded $\|F\|_{\mathrm{op}}, \|T_3\|_{\mathrm{op}}, \|T_4\|_{\mathrm{op}}$) are themselves coherent and well-defined. The sub-claim A in `00_claim.md` is the regularity-class specification, and nothing red has argued falsifies it. The convex normal ball of a compact $G = \mathrm{SO}(K)$ is standard ([Nakahara2003 §10.3], `[doCarmo1992 §3.3]`), the Fisher-normal coordinates around a Gaussian parent are standard ([AmariNagaoka2000 Ch. 4]), and the $C^4$ tensor norm conventions are standard ([HornJohnson2013 §4]). The upgrade program — give the schematic proposition a quantitatively tight statement with these specifications — is achievable.

The honest verdict that survives this debate is therefore: the upgrade *program* in the claim (specifications 1-2 plus explicit Laplace-derived $C_1, C_5$) is correct in form and is the right next step for the proposition. The specific numerical coefficients $\tfrac{1}{12}, \tfrac{1}{24}, \tfrac{1}{72}$ in the claim are wrong: the corrected values are $C_5 = \tfrac{1}{8}\|T_4\|_{\mathrm{op}}/m_I^2 + \tfrac{5}{24}\|T_3\|_{\mathrm{op}}^2/m_I^3$ and $C_1 = \tfrac{\sqrt 2}{12}\|F\|^{3/2}_{\mathrm{op}}/m_I^{1/2}$ (Edgeworth-route) or $C_1 = \tfrac{\sqrt 2}{3}\|T_3\|_{\mathrm{op}}/m_I^{3/2}$ (Taylor-route). The judge's appropriate action is REMAND with the corrected coefficient values, not falsification of the upgrade program. The claim's structural commitments — the $V^{3/2}$ scaling, the separation into dispersion and anharmonic contributions, the operator-norm convention on tensors, the Fisher-normal coordinate choice — all survive red's attack. Only the specific numerical prefactors need correction, and the corrected values are explicit from the standard Laplace asymptotics that both teams have now verified by sympy.
