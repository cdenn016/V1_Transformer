# Evidence Pack — prop-1-tighter-quantitative

Neutral fact pack. Both teams use this as the shared starting point.

## Manuscript references (Attention/Participatory_it_from_bit.tex)

- **Closure-residual definition (Eq.~\ref{eq:rg_closure}, line 4372):**
  $$\varepsilon_I = \frac{\inf_{c \in \mathbb{R}}\|\mathcal{F}^{\mathrm{exact}}_{s+1} - \mathcal{F}^{\mathrm{agent}}_{s+1} - c\|_{L^\infty(\mathcal{N}_I)}}{\tau + \|\mathcal{F}^{\mathrm{exact}}_{s+1}\|_{L^\infty(\mathcal{N}_I)}}.$$
  The closure norm is $L^\infty$ on the parent neighborhood $\mathcal{N}_I$ modulo additive constants. This is the norm the proposition's $\varepsilon_{s+1}$ refers to.

- **Internal Hessian and constrained gap (Eq.~\ref{eq:rg_internal_hessian}, Eq.~\ref{eq:rg_constrained_gap}, lines 4380, 4389):**
  $H_I^\perp(q_I) \approx L_I \otimes F(q_I)$ where $L_I$ is the weighted graph Laplacian of $(\beta_{ij}+\beta_{ji})/2$ and $F(q_I)$ is the Fisher at the parent. The constrained gap is $\lambda_{I,w} = \inf_{\xi: \sum w_i \xi_i = 0, \xi \neq 0} \xi^\top L_I \xi / \xi^\top \xi$ and the stiffness is $m_I = \lambda_{I,w} \lambda_{\min}(F(q_I))$. This $m_I$ is what enters the proposition's constants in the denominator.

- **Anharmonic remainder (existing manuscript statement, line 4399 and 4525-4526):** $\mathcal{E}_{\mathrm{anh}} = O(\tau^2 \|H^{-1}\|^3 \|T_3\|^2 + \tau^2 \|H^{-1}\|^2 \|T_4\|)$. With $\|H^{-1}\| \le m_I^{-1}$, this gives $\mathcal{E}_{\mathrm{anh}} = O(\tau^2 m_I^{-3} \|T_3\|^2 + \tau^2 m_I^{-2} \|T_4\|)$. The proposition's $C_5$ encodes the prefactors that the existing $O(\cdot)$ form leaves implicit.

- **Laplace approximation form (Eq.~\ref{eq:rg_laplace}, line 4396):** $\mathcal{F}_{s+1}(Y_I) = \mathcal{F}_{\mathrm{eff}}(Y_I) + \tfrac{\tau}{2}\log\det{}'H_I^\perp(Y_I) + \mathrm{const} + \mathcal{E}_{\mathrm{anh}}$.

- **Proposition 1 itself (line 4545, Eq. at lines 4548-4551):** The schematic bound under debate.

- **Self-disclosure at line 4554:** "the constants $C_k$, the closure norm $\|\cdot\|_\mathcal{B}$, and the regularity class of the parent ansatz are not pinned down here, and tightness is not established; a theorem-grade statement would require specifying each."

## Standard saddle-point / Laplace canon (the source of truth for this debate)

### Wong 2001, *Asymptotic Approximations of Integrals*, §II.4

For an integral of the form $I(\lambda) = \int_a^b g(t) e^{-\lambda h(t)} dt$ around a non-degenerate interior minimum $t_0$ of $h(t)$, the asymptotic expansion is
$$ I(\lambda) \sim \sqrt{\frac{2\pi}{\lambda h''(t_0)}} e^{-\lambda h(t_0)} \left[ g(t_0) + \frac{1}{\lambda} c_1 + O(\lambda^{-2}) \right], $$
with the leading correction
$$ c_1 = \frac{1}{8} \frac{g''(t_0)}{h''(t_0)} - \frac{1}{8} \frac{g'(t_0) h'''(t_0)}{h''(t_0)^2} + \frac{1}{24} \frac{g(t_0) h''''(t_0)}{h''(t_0)^2} - \frac{1}{24} \frac{g(t_0)(h'''(t_0))^2}{h''(t_0)^3} \cdot \frac{15}{8} - \frac{1}{8} \frac{g'(t_0)^2 \cdots}{\cdots}. $$

The relevant **multidimensional** Laplace expansion for $I = \int e^{-\lambda \Phi(x)} d^n x$ around minimum $x^* = 0$ with $\Phi(0) = 0$, $\Phi'(0) = 0$, $\Phi''(0) = H$ is `[Wong2001 §IX.5, Eq. 5.04]` and `[BenderOrszag1999 §6.5]`:
$$ I \sim (2\pi/\lambda)^{n/2} (\det H)^{-1/2} \cdot \left[ 1 - \frac{1}{\lambda} \mathcal{C}_1 + O(\lambda^{-2}) \right], $$
where the $\mathcal{C}_1$ correction is given by the standard "two-loop" expansion:
$$ \mathcal{C}_1 = \frac{1}{8} H^{ij,kl} \Phi_{ijkl} - \frac{1}{12} H^{ij,kl,mn} \Phi_{ijk} \Phi_{lmn} - \frac{1}{8} H^{ij,kl,mn} \Phi_{ikl} \Phi_{jmn}, $$
where $\Phi_{ij\ldots} = \partial_i \partial_j \cdots \Phi(0)$, and $H^{ij,kl} = H^{-1\,ij} H^{-1\,kl}$ (the various contraction patterns are specified by the tensor index structure).

The coefficients $\frac{1}{8}, \frac{1}{12}, \frac{1}{8}$ are the canonical Wick-contraction prefactors for the three distinct topologies of the leading correction:
- $\frac{1}{8} H^{ij} H^{kl} \Phi_{ijkl}$ — the "quartic theta" (one quartic vertex, two propagators forming a theta graph).
- $\frac{1}{12} H^{il} H^{jm} H^{kn} \Phi_{ijk} \Phi_{lmn}$ — the "sunset" or "non-crossing cubic-cubic" (two cubic vertices, three propagators).
- $\frac{1}{8} H^{ij} H^{kl} H^{mn} \Phi_{ikm} \Phi_{jln}$ — the "double-bubble" or "crossing cubic-cubic" (two cubic vertices, three propagators, different topology).

**Note on conventions:** The literature uses different sign and normalization conventions. Some texts write $\Phi_{ijk\ldots}$ for the *symmetric* tensor (already including the $1/n!$ factor from Taylor); others write $\partial_i \partial_j \ldots \Phi$ without the factorial. The coefficients $\frac{1}{8}, \frac{1}{12}, \frac{1}{8}$ are for the no-factorial convention. Other conventions give $\frac{1}{8} \cdot \frac{1}{4!} \cdot 4! = \frac{1}{8}$ etc. so the *value* is the same when reduced.

### Bender-Orszag 1999, *Advanced Mathematical Methods*, §6.5

Eq. 6.5.4: For $I(\lambda) = \int g(t) e^{-\lambda \phi(t)} dt$,
$$ I(\lambda) \sim \sqrt{\frac{2\pi}{\lambda \phi''(t_0)}} e^{-\lambda \phi(t_0)} g(t_0) \left[ 1 - \frac{1}{\lambda} \left( \frac{\phi''''(t_0)}{8(\phi''(t_0))^2} - \frac{5(\phi'''(t_0))^2}{24(\phi''(t_0))^3} \right) + O(\lambda^{-2}) \right]. $$

The 1D leading correction has coefficient $\frac{1}{8}$ on $\phi''''/(\phi'')^2$ and $-\frac{5}{24}$ on $(\phi''')^2/(\phi'')^3$. The "$\frac{5}{24}$" combines the $\frac{1}{12} + \frac{1}{8} = \frac{2+3}{24} = \frac{5}{24}$ from the two cubic-cubic topologies summing to a single 1D coefficient.

In multidimensional, the $\frac{1}{12}$ and $\frac{1}{8}$ are *separate* combinatorial coefficients on *separate* contractions. They cannot be summed without specifying the contraction pattern.

### Edgeworth expansion `[AmariNagaoka2000 Ch. 8, KolassaWiens1995]`

For a probability density $p(x) = \phi(x; \mu, \Sigma) [1 + h_1(x; \kappa_3, \kappa_4) + O(\kappa_4)]$, where $\phi$ is the Gaussian approximation and $h_1$ encodes the third and fourth cumulants:
$$ h_1(x) = \frac{1}{6} \kappa_3 H_3(x) + \frac{1}{24} \kappa_4 H_4(x) + \frac{1}{72} \kappa_3^2 H_6(x) + O(\kappa^3), $$
where $H_n$ are normalized Hermite polynomials and $\kappa_3, \kappa_4$ are third and fourth cumulants. The coefficients $\frac{1}{6}, \frac{1}{24}, \frac{1}{72}$ are *standardly tabulated* in textbook Edgeworth expansions.

This is the source of the proposed $\frac{1}{12}, \frac{1}{24}, \frac{1}{72}$ coefficients in the claim. Whether the proposed values *exactly* match an $L^\infty$-norm bound on $\varepsilon_I$ requires explicit derivation.

### Pinsker's inequality `[CoverThomas2006 §11.6]`

For two probability measures $P, Q$ on a common space:
$$ \|P - Q\|_{\mathrm{TV}} \le \sqrt{\frac{1}{2} D_{\mathrm{KL}}(P \| Q)}. $$

Applied to the closure residual: if $V_I = D_{\mathrm{KL}}(\tilde q_i \| q_I)$ measures dispersion, then total-variation distance from the parent is $O(V_I^{1/2})$. The next-order correction is $O(V_I^{3/2})$ from cubic terms, which is the source of the $V^{3/2}$ scaling in the proposition.

For Gaussian $\tilde q_i = \mathcal{N}(\mu_i, \Sigma_i)$ and $q_I = \mathcal{N}(\mu_I, \Sigma_I)$ with $\mu_i = \mu_I + \delta\mu_i$ and $\Sigma_i = \Sigma_I + \delta\Sigma_i$, the KL expands as
$$ D_{\mathrm{KL}}(\tilde q_i \| q_I) = \frac{1}{2} \delta\mu_i^\top F(q_I) \delta\mu_i + \frac{1}{4} \mathrm{tr}((F(q_I) \delta\Sigma_i)^2) + O(\|\delta\|^3), $$
so $V_I \sim \frac{1}{2} \|\delta\mu\|^2_{F(q_I)} + O(\|\delta\|^4)$ at leading order, and $V_I^{3/2}$ scales as $\|\delta\mu\|^3 \|F(q_I)\|^{3/2}$.

## Mathematical preliminaries for sympy verification

For agents using `sympy`: the verification of $C_5$'s coefficients reduces to:

1. Compute the Gaussian path integral $\int d^n\xi \exp\left[-\frac{1}{2}\xi^\top H \xi - \frac{1}{6}T_3[\xi,\xi,\xi] - \frac{1}{24}T_4[\xi,\xi,\xi,\xi]\right]$ to fourth order in the anharmonic vertices.
2. Expand $e^{-V} \approx 1 - V + V^2/2 - \cdots$ where $V = \frac{1}{6}T_3[\xi^3] + \frac{1}{24}T_4[\xi^4]$.
3. Evaluate the Gaussian moments using Wick's theorem.
4. Extract the coefficients of $T_4 \cdot H^{-2}$ and $T_3^2 \cdot H^{-3}$.

Standard answer (from `[Zinn-Justin2002 §1.4, Eq. 1.41]` or any QFT textbook on functional integrals):
- $T_4$ coefficient: $-\frac{1}{8} T_{4,ijkl} H^{-1\,ij} H^{-1\,kl}$ for the symmetric tensor contraction.
- $T_3^2$ coefficient: $-\frac{1}{12} T_{3,ijk} T_{3,lmn} H^{-1\,il} H^{-1\,jm} H^{-1\,kn}$ (non-crossing) and $-\frac{1}{8} T_{3,ijk} T_{3,lmn} H^{-1\,ij} H^{-1\,kl\cdot mn}$ (crossing topology); the dimensional-reduction sum recovers the 1D $-\frac{5}{24}$.

Bounding by operator norms:
- $|T_{4,ijkl} H^{-1\,ij} H^{-1\,kl}| \le \|T_4\|_{\mathrm{op}} \cdot \|H^{-1}\|_{\mathrm{op}}^2$ where $\|\cdot\|_{\mathrm{op}}$ is the operator norm of the tensor viewed as a multilinear map on the unit ball. For symmetric tensors, $\|T\|_{\mathrm{op}} = \max_{\|v\|=1} |T(v, \ldots, v)|$, and the Wick contraction is bounded by the trace inequality `[HornJohnson2013 §4]`.

The **proposed claim's $C_5$ coefficients** $\frac{1}{24}$ and $\frac{1}{72}$ differ from the standard Wick coefficients $\frac{1}{8}$ and $\frac{1}{12}+\frac{1}{8}$. The discrepancy may come from:
- The Edgeworth convention ($\frac{1}{24}$ on Hermite-$4$ matches the Edgeworth normalization, vs. $\frac{1}{8}$ on the contracted Wick term).
- An additional factor of $\frac{1}{3}$ somewhere (since $\frac{1}{72} = \frac{1}{24 \cdot 3}$).

Both teams must work out the exact correspondence via sympy.

## Standard references the teams should cite

- `[Wong2001]` Wong, R. (2001), *Asymptotic Approximations of Integrals*, SIAM Classics §IX.5 (multidimensional Laplace).
- `[BenderOrszag1999]` Bender, C. M., & Orszag, S. A. (1999), *Advanced Mathematical Methods for Scientists and Engineers*, Springer §6.4-6.5.
- `[CoverThomas2006]` Cover, T. M., & Thomas, J. A. (2006), *Elements of Information Theory*, Wiley §11.6 (Pinsker).
- `[AmariNagaoka2000]` Amari, S., & Nagaoka, H. (2000), *Methods of Information Geometry*, AMS / Oxford, Ch. 4 (Edgeworth expansion in dual coordinates).
- `[Zinn-Justin2002]` Zinn-Justin, J. (2002), *Quantum Field Theory and Critical Phenomena*, Oxford, §1.4 (functional integrals, Wick contractions, perturbation expansion). The QFT-canonical source for the coefficients $\frac{1}{8}, \frac{1}{12}, \frac{1}{8}$ on the three two-loop topologies.
- `[KolassaWiens1995]` Kolassa, J. E., & Wiens, D. P. (1995), "On the convergence of Edgeworth series." For convergence conditions of the Edgeworth expansion.
- `[HornJohnson2013]` Horn, R. A., & Johnson, C. R. (2013), *Matrix Analysis*, Cambridge, §4-5 (operator norms, trace inequalities).
- `[Amari2016]` Amari, S. (2016), *Information Geometry and Its Applications*, Springer, Ch. 4 (Fisher information; KL expansion at the Gaussian saddle).

## What this evidence does NOT settle

- Whether $\frac{1}{12}$ in the proposed $C_1$ is the correct operator-norm coefficient for the dispersion contribution under the manuscript's $L^\infty$ closure norm. The Edgeworth standard form has $\frac{1}{6}$ on Hermite-3; converting to an $L^\infty$ bound on $V^{3/2}$ involves a Pinsker-step and a normalization choice that the claim does not specify.
- Whether $\frac{1}{24}$ on $T_4$ in $C_5$ matches the standard $\frac{1}{8}$ Wick coefficient under the operator-norm bound. The $\frac{1}{24}$ matches the Edgeworth-Hermite-4 normalization but not the direct Wick coefficient.
- Whether $\frac{1}{72}$ on $T_3^2$ in $C_5$ is consistent with the sum-of-topologies $\frac{1}{12} + \frac{1}{8} = \frac{5}{24}$ that the standard literature gives. $\frac{1}{72}$ is *not* the same as $\frac{5}{24}$, so either the coefficient is wrong, or it corresponds to a *different* tensor norm convention that the claim does not specify.
- Whether the operator norm bounds used in deriving $C_1$ and $C_5$ are *tight* (theorem-grade) or *loose* (proposition-grade upper bounds). A tight bound requires the Wick contractions to be saturated by extremal tensor configurations.
- Whether the regularity class $C^4$ with bounded $\|T_3\|_{\mathrm{op}}, \|T_4\|_{\mathrm{op}}$ is the minimal regularity for the proposed expansion, or whether a stronger class (e.g., Sobolev $W^{4,\infty}$, Hölder $C^{4,\alpha}$) is required.

These are the structural cracks both teams must address. The decisive question is whether the proposed numerical coefficients $\frac{1}{12}, \frac{1}{24}, \frac{1}{72}$ can be derived from the standard Laplace asymptotic expansion, or whether they require independent justification — and if they require independent justification, what that justification is.
