# Red Opening — prop-1-tighter-quantitative

## Steelman (opposing position)

Proposition 1's schematic constants $C_1$ and $C_5$ can be promoted to explicit Laplace-derived values $\tfrac{1}{12}\|F\|^{3/2}/m_I^{1/2}$, $\tfrac{1}{24}\|T_4\|/m_I^2$, $\tfrac{1}{72}\|T_3\|^2/m_I^3$ by reading off the standard Edgeworth-Hermite normalizations of the cubic, quartic, and squared-cubic cumulant corrections of a Gaussian closure on $\mathcal{N}_I$ under the manuscript's existing $L^\infty$ closure norm.

## Position

The proposed coefficients $\tfrac{1}{24}$ on $\|T_4\|_{\mathrm{op}}/m_I^2$ and $\tfrac{1}{72}$ on $\|T_3\|_{\mathrm{op}}^2/m_I^3$ in $C_5$ are wrong: they are the Edgeworth *density-correction* coefficients (on Hermite polynomials $H_4$, $H_6$), not the *free-energy* / log-partition Wick coefficients required by the manuscript's $L^\infty$ closure norm on $\mathcal{F}^{\mathrm{exact}}_{s+1} - \mathcal{F}^{\mathrm{agent}}_{s+1}$ at line 4374. The correct standard values are $\tfrac{1}{8}$ and $\tfrac{5}{24}$. The proposed coefficients are smaller than the Wick result by factors of 3 and 15 respectively, so they cannot be obtained from any valid upper bound on the standard saddle-point asymptotic expansion. The proposed $C_1 = \tfrac{1}{12}\|F(q_I)\|^{3/2}/m_I^{1/2}$ is dimensionally inconsistent with the $L^\infty$ free-energy norm and is missing the cubic-cumulant tensor that $V^{3/2}$ scaling structurally requires.

## Evidence

### Executed sympy verification — 1D Laplace expansion at $m_I = 1$

Input action: $H(\xi) = \tfrac{1}{2}\xi^2 + \tfrac{1}{6} t_3 \xi^3 + \tfrac{1}{24} t_4 \xi^4$. Expand $Z = \int e^{-H}\,d\xi$ to second order in $(t_3, t_4)$ and compute $F = -\log(Z/Z_0)$:

```python
import sympy as sp
xi, t3, t4 = sp.symbols('xi t3 t4', real=True)
V = sp.Rational(1,6)*t3*xi**3 + sp.Rational(1,24)*t4*xi**4
# Gaussian moments <xi^(2n)>_0 = (2n-1)!!
def gauss_moment(n):
    if n % 2: return 0
    res = 1
    for k in range(1, n, 2): res *= k
    return res
def gaussian_expectation(expr):
    poly = sp.Poly(sp.expand(expr), xi)
    return sum(c * gauss_moment(m[0]) for m, c in poly.terms())
EV  = gaussian_expectation(V)        # = t4/8
EV2 = gaussian_expectation(V*V)      # = 5 t3^2/12 + 35 t4^2/192
ratio = 1 - EV + EV2/2               # Z/Z0 to 2nd order
x = sp.expand(ratio - 1)
logZ = sp.expand(x - x*x/2)
logZ = sum(t for t in sp.Add.make_args(logZ)
           if sp.Poly(t, t3, t4).total_degree() <= 2)
F_shift = sp.expand(-logZ)
```

Output:

```
<V>   = t4/8
<V^2> = 5*t3^2/12 + 35*t4^2/192
Z/Z0  = 5*t3^2/24 + 35*t4^2/384 - t4/8 + 1
log(Z/Z0)  = 5*t3^2/24 + t4^2/12 - t4/8     (truncated to 2nd order)
F_shift = -log(Z/Z0) = -5*t3^2/24 - t4^2/12 + t4/8

Coefficient of t4   in F_shift: 1/8
Coefficient of t3^2 in F_shift: -5/24
```

Magnitude-wise the free-energy shift carries $|t_4|/8$ and $5|t_3|^2/24$. With $m_I = 1$ (general $m_I$ inserts $1/m_I^2$ on the quartic and $1/m_I^3$ on the cubic-squared, scaling from $H \to m_I H$), the closure-residual upper bound contributed by anharmonics is

$$ \mathcal{E}_{\mathrm{anh}}^{(I)} \;\le\; \tfrac{1}{8}\, \frac{\|T_4\|_{\mathrm{op}}}{m_I^2} \;+\; \tfrac{5}{24}\, \frac{\|T_3\|_{\mathrm{op}}^2}{m_I^3}, $$

matching `[BenderOrszag1999 §6.5 Eq. 6.5.4]` (the textbook 1D Laplace correction has exactly $\phi''''/(8\,(\phi'')^2) - 5(\phi''')^2/(24\,(\phi'')^3)$). The claim's $\tfrac{1}{24}$ and $\tfrac{1}{72}$ are off by factors of $3$ and $15$ respectively, both *tighter* than what the Wick expansion permits.

### Origin of $\tfrac{1}{24}, \tfrac{1}{72}$ — Edgeworth density coefficients, not free-energy coefficients

These numbers are the Edgeworth-Hermite normalizations [evidence pack line 56-57; AmariNagaoka2000 Ch. 8]:

$$ p(x) = \phi(x)\left[1 + \tfrac{\kappa_3}{6} H_3(x) + \tfrac{\kappa_4}{24} H_4(x) + \tfrac{\kappa_3^2}{72} H_6(x) + O(\kappa^3)\right]. $$

These coefficients sit in front of *Hermite polynomials* in the *density* expansion. Hermite polynomials are $L^2(\phi)$-orthogonal to the constant function, so they contribute *zero* to the partition function at first order — they encode shape corrections, not normalization corrections. The manuscript's closure norm (line 4374, `eq:rg_closure`) is

$$ \varepsilon_I = \frac{\inf_c \|\mathcal{F}^{\mathrm{exact}}_{s+1} - \mathcal{F}^{\mathrm{agent}}_{s+1} - c\|_{L^\infty(\mathcal{N}_I)}}{\tau + \|\mathcal{F}^{\mathrm{exact}}_{s+1}\|_{L^\infty(\mathcal{N}_I)}}, $$

an $L^\infty$ norm on the *free energy* $\mathcal{F}$, not on the density $p = e^{-\mathcal{F}/\tau}$. The relevant coefficients are the integrated $\log Z$ corrections — the Wick coefficients $\tfrac{1}{8}$ and $\tfrac{5}{24}$ verified above — not the Hermite density coefficients $\tfrac{1}{24}$ and $\tfrac{1}{72}$. The claim conflates two different normalizations of two different objects.

### Cross-check against `[Wong2001 §IX.5 Eq. 5.04]` and evidence-pack line 34

The multidimensional two-loop expansion is

$$ \mathcal{C}_1 = \tfrac{1}{8} H^{ij} H^{kl} \Phi_{ijkl} \;-\; \tfrac{1}{12} H^{il}H^{jm}H^{kn} \Phi_{ijk}\Phi_{lmn} \;-\; \tfrac{1}{8} H^{ij}H^{kl}H^{mn} \Phi_{ikm}\Phi_{jln} $$

with coefficients $\tfrac{1}{8}$ (theta) and $\tfrac{1}{12} + \tfrac{1}{8} = \tfrac{5}{24}$ on the two cubic-cubic topologies. The 1D sympy check above reproduces this sum exactly. The claim's $\tfrac{1}{72}$ matches *neither* the sunset coefficient $\tfrac{1}{12}$ nor the double-bubble $\tfrac{1}{8}$ nor their sum $\tfrac{5}{24}$.

### Attack on $C_1$ — dimensional and structural

The claim asserts $C_1 = \tfrac{1}{12} \|F(q_I)\|_{\mathrm{op}}^{3/2} / m_I^{1/2}$ on $V_I^{3/2}$. Two problems:

1. **Missing cubic-cumulant tensor.** Pinsker's inequality `[CoverThomas2006 §11.6]` gives $\|P-Q\|_{TV} \le \sqrt{D_{KL}/2}$, so total-variation distance scales as $V_I^{1/2}$. The $V_I^{3/2}$ correction term in a $L^\infty$ free-energy bound on $\mathcal{F}^{\mathrm{exact}} - \mathcal{F}^{\mathrm{agent}}$ comes from the *cubic* term in the KL Taylor expansion, which is
$$\tfrac{1}{6} C_{ijk}(q_I)\, \delta\mu^i \delta\mu^j \delta\mu^k$$
where $C_{ijk}$ is the third cumulant tensor (third derivative of $-\log q$). A correct bound is
$$\varepsilon_I^{\text{cubic}} \le \tfrac{1}{6} \|C\|_{\mathrm{op}}\, |\delta\mu|^3 \le \tfrac{1}{6}\|C\|_{\mathrm{op}} \cdot (2 V_I / m_I)^{3/2} = \tfrac{1}{6}\cdot 2^{3/2}\, \|C\|_{\mathrm{op}}\, V_I^{3/2}/m_I^{3/2}.$$
The Fisher operator norm $\|F\|_{\mathrm{op}}$ alone cannot generate a cubic term — the Fisher is the Hessian (quadratic curvature) of $-\log q$ by Cencov's theorem `[Cencov1972, AmariNagaoka2000 Ch. 2]`. The claim has dropped the cubic-cumulant tensor entirely, replacing it with $\|F\|^{3/2}$.

2. **Dimensional check.** The closure norm $\varepsilon_I$ is dimensionless after the $\tau + \|\mathcal{F}\|$ denominator, but the prefactor times $V^{3/2}$ must be free-energy units (info) in the numerator. $V_I$ is in info units; $V_I^{3/2}$ is in $(\text{info})^{3/2}$. To produce free energy from $V^{3/2}$, the prefactor must carry units of $(\text{info})^{-1/2}$. The Fisher operator norm $\|F\|_{\mathrm{op}}$ has units of $(\text{info})/(\text{length})^2$, and $m_I = \lambda_{I,w}\lambda_{\min}(F)$ also has units of $(\text{info})/(\text{length})^2$. The proposed $\|F\|^{3/2}/m_I^{1/2}$ has units of $(\text{info}/(\text{length})^2)^{3/2 - 1/2} = (\text{info})/(\text{length})^2$, which times $V^{3/2}$ gives $(\text{info})^{5/2}/(\text{length})^2$ — not free energy. The expression cannot be dimensionally correct without an additional length-squared and info-power-of-$-3/2$ factor, supplied physically by $\|C\|_{\mathrm{op}}/m_I^{3/2}$ in a proper derivation.

3. **The "1/12" coefficient is unattributed.** It is neither the Edgeworth-$H_3$ coefficient ($\tfrac{1}{6}$), nor Pinsker's constant ($\tfrac{1}{2}$), nor the sunset Wick coefficient ($\tfrac{1}{12}$ on $T_3^2$, not on $V^{3/2}$). The product $\tfrac{1}{6} \cdot \tfrac{1}{2} = \tfrac{1}{12}$ from "Edgeworth then Pinsker" is not a standard composition: Pinsker bounds TV by $\sqrt{KL/2}$, not by $\sqrt{KL}/2$, so the composition gives $\tfrac{1}{6} \cdot 2^{-1/2} \neq \tfrac{1}{12}$.

### Tightness vs slackness — the Laplace-asymptotic argument cuts the wrong way

The claim asserts "explicit Laplace-method asymptotics" give the specific values. Laplace asymptotics give *exact* coefficients on *tensor contractions*, not on operator-norm bounds. Converting the contraction $\tfrac{1}{8} T_{4,ijkl} H^{-1\,ij} H^{-1\,kl}$ to an operator-norm bound $\tfrac{1}{8}\|T_4\|_{\mathrm{op}}\|H^{-1}\|_{\mathrm{op}}^2$ requires Cauchy-Schwarz / Hölder-type inequalities that *introduce* slackness factors depending on the dimension of the internal-mode space (e.g., factors of $n$ from $\mathrm{tr}(H^{-1})$ vs $\|H^{-1}\|_{\mathrm{op}}$) and the condition number of $H^{-1}$. A correct upper bound has $C_5 \ge \tfrac{1}{8}\|T_4\|_{\mathrm{op}}/m_I^2$, not smaller. The proposed $\tfrac{1}{24}$ is *tighter* than $\tfrac{1}{8}$ by a factor of 3, which would require either a sub-Wick contraction (impossible — Wick is exact at the Gaussian saddle) or a non-standard tensor norm convention.

## Falsification conditions

This red position is wrong if:

1. **Blue exhibits a sympy session in which the integrated free-energy correction $-\log(Z/Z_0)$ for $H = \tfrac{1}{2}\xi^\top H \xi + \tfrac{1}{6}T_3[\xi^3] + \tfrac{1}{24}T_4[\xi^4]$ has coefficient $\tfrac{1}{24}$ on $T_4$ and $\tfrac{1}{72}$ on $T_3^2$ under any reasonable tensor-norm convention** — including an explicit specification of how the operator norm $\|T_4\|_{\mathrm{op}}$ relates to the contraction $T_{4,ijkl} H^{-1\,ij} H^{-1\,kl}$ that yields a factor-of-3 reduction.

2. **Blue identifies a published derivation in which the closure-residual bound on the *free energy* (not the density) carries Edgeworth-Hermite normalizations $\tfrac{1}{24}, \tfrac{1}{72}$** — citing the specific theorem, page, equation in `[Wong2001]`, `[BenderOrszag1999]`, `[AmariNagaoka2000]`, `[Amari2016]`, or `[Zinn-Justin2002]`.

3. **Blue resolves the dimensional inconsistency of $C_1 = \tfrac{1}{12}\|F\|^{3/2}/m_I^{1/2}$** by exhibiting a derivation in which the $V^{3/2}$ scaling reduces to the Fisher operator norm alone (without the third-cumulant tensor) under the manuscript's $L^\infty$ closure norm.

4. **Blue exhibits a non-standard operator-norm convention on symmetric 3- and 4-tensors that recovers $\tfrac{1}{24}$ and $\tfrac{1}{72}$ as upper-bound coefficients** by showing that the chosen $\|T_4\|_{\mathrm{op}}$ in the claim equals $3 \cdot T_{4,ijkl}H^{-ij}H^{-kl}/\|H^{-1}\|_{\mathrm{op}}^2$ (or equivalent) — and that this convention is the one the manuscript implicitly adopts.

Absent these, the load-bearing assumption — that "standard Laplace asymptotics give these specific numerical values" — is falsified: the standard Laplace asymptotics give $\tfrac{1}{8}$ and $\tfrac{5}{24}$, not $\tfrac{1}{24}$ and $\tfrac{1}{72}$, and the proposition's quantitative upgrade fails on its load-bearing arithmetic.
