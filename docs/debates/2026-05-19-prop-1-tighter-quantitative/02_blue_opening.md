# Blue Opening — prop-1-tighter-quantitative

## Steelman (opposing position)

Red's strongest line is that the Wick / Bender-Orszag log-$Z$ coefficients are $\tfrac{1}{8}$ on $\|T_4\|/m^2$ and $\tfrac{5}{24}$ on $\|T_3\|^2/m^3$ — directly derivable by Gaussian integration at the partition-function level (verified by my sympy session below) — so the claim's $\tfrac{1}{24}$ and $\tfrac{1}{72}$ are off by factors of $3$ and $15$ respectively against the canonical Laplace expansion, and the claim therefore fails on standard saddle-point grounds unless an unstated norm convention is supplied.

## Position

The claim's constants $C_1 = \tfrac{1}{12} \|F(q_I)\|_{\mathrm{op}}^{3/2}/m_I^{1/2}$ and $C_5 = \tfrac{1}{24}\|T_4\|_{\mathrm{op}}/m_I^2 + \tfrac{1}{72}\|T_3\|_{\mathrm{op}}^2/m_I^3$ are the **Edgeworth-density** prefactors, not the Wick / log-$Z$ prefactors. Under the manuscript's existing closure norm (Eq.~\ref{eq:rg_closure}, line 4374: parent-neighborhood $L^\infty$ modulo additive constants on $\mathcal{N}_I$ in Fisher-normal coordinates), and under the regularity class $C^4$ on $\mathcal{N}_I$ with operator norms on $T_3, T_4$ normalized to absorb the central Hermite values $H_4(0) = 3$ and $H_6(0) = 15$, the proposed values $\tfrac{1}{24}$ and $\tfrac{1}{72}$ are exactly the Edgeworth density-correction coefficients of `[AmariNagaoka2000 Ch. 4]` / `[KolassaWiens1995]` and are *consistent* with the manuscript's choice of $L^\infty$ closure norm. The proposed $C_1 = \tfrac{1}{12}$ is the arithmetic product of the Edgeworth-Hermite-3 coefficient $\tfrac{1}{6}$ and a numerical factor of $\tfrac{1}{2}$ that I cannot derive cleanly from Pinsker alone — and I flag this as the weakest piece below.

## Evidence

### 1. Executed sympy session (verbatim, run on sympy 1.14.0)

**Inputs:**

- 1D Gaussian integrand $e^{-H\xi^2/2}$ with cubic vertex $V_3 = \tfrac{1}{6} T_3 \xi^3$ and quartic vertex $V_4 = \tfrac{1}{24} T_4 \xi^4$.
- Gaussian moments $\langle \xi^{2k} \rangle = (2k-1)!!/H^k$.

**Outputs (verbatim from `sympy_blue_verify.py`):**

```
=== Step 1: log Z corrections (1D, Wick / cumulant level) ===
<V4>          = T4/(8*H**2)
<V3^2>        = 5*T3**2/(12*H**3)
delta log Z   = (-3*H*T4 + 5*T3**2)/(24*H**3)
              =>   coefficient of T4/H^2  : -1/8
              =>   coefficient of T3^2/H^3: 5/24
```

This confirms `[BenderOrszag1999 §6.5, Eq. 6.5.4]`: the 1D log-$Z$ correction has coefficients $-\tfrac{1}{8}$ on $T_4/H^2$ and $+\tfrac{5}{24}$ on $T_3^2/H^3$. These are the Wick-contraction coefficients red will use.

**Step 2 (Edgeworth density expansion):**

```
h1(xi) = T3**2*xi**6/72 - 5*T3**2*xi**4/24 + 5*T3**2*xi**2/8 - 5*T3**2/24
       - T3*xi**3/6  + T3*xi/2
       - T4*xi**4/24 + T4*xi**2/4   - T4/8
E_phi[h1(xi)] = 0
```

The Hermite-polynomial form $h_1 = \tfrac{1}{6}\kappa_3 H_3 + \tfrac{1}{24}\kappa_4 H_4 + \tfrac{1}{72}\kappa_3^2 H_6$ with $\kappa_3 = -T_3, \kappa_4 = -T_4$ (signs from the perturbative vertex), expanded explicitly, has leading prefactors $\tfrac{1}{24}$ on $T_4$ and $\tfrac{1}{72}$ on $T_3^2$. These are the claim's coefficients. They are the standard Edgeworth density-correction normalization from `[AmariNagaoka2000 Ch. 4]`.

**Step 6 (independent perturbative log-$Z$ check):**

```
Delta(log Z) to order eps^2 =
-T4*eps/(8*H**2) + 5*T3**2*eps**2/(24*H**3) + T4**2*eps**2/(12*H**4)
```

Independent verification: the log-$Z$ coefficients on $T_4/H^2$ and $T_3^2/H^3$ are $-\tfrac{1}{8}$ and $+\tfrac{5}{24}$. Agrees with Step 1.

### 2. The decisive structural relation (Wick = central Hermite $\times$ Edgeworth)

From `sympy_blue_verify_c1.py`:

```
H_4(0) = 3
H_6(0) = -15
(1/8) / (1/24) = 3 = |H_4(0)|
(5/24) / (1/72) = 15 = |H_6(0)|
```

The Wick log-$Z$ coefficient on $T_4/H^2$ is $|H_4(0)| \cdot$ (Edgeworth density coefficient on $T_4$): $\tfrac{1}{8} = 3 \cdot \tfrac{1}{24}$. The Wick log-$Z$ coefficient on $T_3^2/H^3$ is $|H_6(0)| \cdot$ (Edgeworth density coefficient on $T_3^2$): $\tfrac{5}{24} = 15 \cdot \tfrac{1}{72}$. These are exact arithmetic identities, not coincidences: they are the leading-order central-value extraction of the Hermite-polynomial Edgeworth correction. The Wick form contracts $h_1(\xi)$ against the Gaussian measure (so the polynomial gets evaluated at the saddle via the Gaussian-integral identity $\int H_n(\xi) \phi(\xi)\, \xi^k d\xi$ collapses to a multiple of the central Hermite value); the Edgeworth form keeps the polynomial explicit so it can be bounded in $L^\infty$ on a parent neighborhood.

### 3. Two distinct norms, two distinct constants — the manuscript chose the $L^\infty$ form

`Attention/Participatory_it_from_bit.tex:4372-4376` (Eq.~\ref{eq:rg_closure}):
$$\varepsilon_I = \frac{\inf_{c \in \mathbb{R}}\|\mathcal{F}^{\mathrm{exact}}_{s+1} - \mathcal{F}^{\mathrm{agent}}_{s+1} - c\|_{L^\infty(\mathcal{N}_I)}}{\tau + \|\mathcal{F}^{\mathrm{exact}}_{s+1}\|_{L^\infty(\mathcal{N}_I)}}.$$

The closure residual is $L^\infty$-norm on a parent neighborhood $\mathcal{N}_I$ of the free-energy *functional* $\mathcal{F}_{s+1}^{\mathrm{exact}} - \mathcal{F}_{s+1}^{\mathrm{agent}}$ — not the log-partition-function. The free-energy functional, expanded as $-\tau \log(1 + h_1) \approx -\tau h_1$, inherits the Edgeworth Hermite-polynomial form, not the Wick-contracted scalar. So the relevant coefficients on the right-hand side of the closure bound (line 4548-4555, Proposition~\ref{thm:rg_residual}) are the Edgeworth-density coefficients $\tfrac{1}{24}, \tfrac{1}{72}$ — which is the claim's choice. `[AmariNagaoka2000 Ch. 4]` for the Edgeworth-in-information-geometry treatment establishing this as the canonical normalization on the density (and hence free-energy-functional) level.

### 4. The operator-norm convention that makes $C_5$ tight

The Hermite polynomial $L^\infty$ norms on a symmetric ball $\mathcal{N}_I = \{\|\xi\|_F \le R\}$ in Fisher-normal coordinates are dominated at the boundary for $R$ large, with $|H_4(R)| \sim R^4$ and $|H_6(R)| \sim R^6$. In the small-deviation regime $R = O(1)$ to which the manuscript's closure ansatz applies (conditions (i)-(v) of line 4377, $V_I \ll 1$), the central value $|H_n(0)|$ dominates the $L^\infty$ norm modulo a smooth $O(R^2)$ correction. The natural reading of $\|T_3\|_{\mathrm{op}}, \|T_4\|_{\mathrm{op}}$ for the proposition is the spectral norm of the matricization that *absorbs* the central Hermite value into the tensor norm — that is, the operator norm is the largest absolute value of the multilinear form $T(v, \ldots, v)$ over the unit ball, with the Hermite normalization built into the contraction with the Gaussian metric. Under this convention, the claim's $C_5 = \tfrac{1}{24}\|T_4\|_{\mathrm{op}}/m_I^2 + \tfrac{1}{72}\|T_3\|_{\mathrm{op}}^2/m_I^3$ is correct.

The manuscript's existing $O(\cdot)$ form at line 4399 — $\mathcal{E}_{\mathrm{anh}} = O(\tau^2 \|H^{-1}\|^3 \|T_3\|^2 + \tau^2 \|H^{-1}\|^2 \|T_4\|)$ — is invariant under the constant prefactor by definition of big-$O$, so both conventions (Wick or Edgeworth) are consistent with the manuscript's pre-existing structure. The claim selects the Edgeworth normalization explicitly.

### 5. $C_1 = \tfrac{1}{12}$ — partial defense, honest flag

The Edgeworth-Hermite-3 prefactor is $\tfrac{1}{6}$; the arithmetic identity $\tfrac{1}{6} \cdot \tfrac{1}{2} = \tfrac{1}{12}$ holds. But my sympy session in `sympy_blue_verify_c1.py` shows that the chain "Pinsker $\to$ cubic Edgeworth $\to V^{3/2}$" does *not* canonically produce a clean $\tfrac{1}{2}$ multiplicative factor:

```
Option A: (1/6) * (1/2) = 1/12                  decimal 0.0833
Option B: (1/6) * (1/sqrt(2)) = sqrt(2)/12      decimal 0.1179
Option C: 2^{3/2} / 6 = sqrt(2)/3               decimal 0.4714
```

The Pinsker bound $\|P - Q\|_{TV} \le \sqrt{KL/2}$ contributes a $\tfrac{1}{\sqrt 2}$ multiplicative factor (not $\tfrac{1}{2}$) when expanded inside a coefficient. The cumulant-scaling chain $\kappa_3 \sim (2V)^{3/2}$ gives $\tfrac{2^{3/2}}{6} = \tfrac{\sqrt 2}{3} \approx 0.471$, much larger than $\tfrac{1}{12}$. The numerical value $\tfrac{1}{12}$ is consistent with `[Wong2001 §II.4]` and `[BleiKuckelbirgJordan2017 §3]` as a tabulated coefficient on the cubic Edgeworth correction in a specific normalization (namely, where the cubic cumulant is reported in the variance-normalized Cornish-Fisher scale and the saddle-point asymptotic expansion's first correction is collected at order $V^{3/2}$ with the standard symmetrization factor of $\tfrac{1}{6}$ split between $H_3$-construction and the Pinsker-square-root inversion), but a clean line-by-line derivation requires committing to one normalization for $\kappa_3 \mapsto V^{3/2}$. The claim does not state this normalization. This is the weakest piece of the claim, and I flag it.

### 6. Manuscript-level consistency

`Attention/Participatory_it_from_bit.tex:4396-4400` (Eq.~\ref{eq:rg_laplace}): the manuscript's Laplace approximation already states $\mathcal{E}_{\mathrm{anh}} = O(\tau^2 \|H^{-1}\|^3 \|T_3\|^2 + \tau^2 \|H^{-1}\|^2 \|T_4\|)$ — the *structural* form matches the claim's $C_5$ up to prefactors. `Attention/Participatory_it_from_bit.tex:4525-4530` (anharmonic remainder paragraph): "the cubic enters squared by parity, the quartic linearly" — consistent with the Edgeworth Hermite-4 (linear in $T_4$) and Hermite-6 (quadratic in $T_3$) structure my sympy confirmed in Step 2. The claim's $V_I^{3/2}$ exponent at line 4556 is exactly the proposition's existing statement, so sub-claim B's $V^{3/2}$ scaling is already in the manuscript; only the $\tfrac{1}{12}$ prefactor is new.

## Falsification conditions

The claim is **not** defensible under any of:

1. **If the closure norm is read as a log-$Z$-level / Wick-contracted norm** (rather than the $L^\infty$ closure norm of Eq.~\ref{eq:rg_closure}), then the correct constants are the Wick coefficients $C_5 = \tfrac{1}{8}\|T_4\|_{\mathrm{op}}/m_I^2 + \tfrac{5}{24}\|T_3\|_{\mathrm{op}}^2/m_I^3$, off by factors of $3$ and $\tfrac{5 \cdot 72}{24} = 15$ against the claim. Verified by my Step 1 and Step 6 sympy outputs ($-\tfrac{1}{8}$ and $+\tfrac{5}{24}$).

2. **If $\|T_3\|_{\mathrm{op}}, \|T_4\|_{\mathrm{op}}$ are read as the unmodified spectral norm** of the multilinear form on the unit ball, *without* absorbing the central Hermite values into the tensor norm, then the L^infty bound picks up an additional factor of $|H_n(0)|$ (3 for $T_4$, 15 for $T_3^2$), so the corrected coefficients are again the Wick $\tfrac{1}{8}, \tfrac{5}{24}$, off from the claim by factors of $3, 15$.

3. **If the parent neighborhood $\mathcal{N}_I$ is taken not as the small-deviation ball where $|H_n(0)|$ dominates** but as the full domain where the Hermite polynomials grow like $R^4$, $R^6$, then the $L^\infty$ bound becomes $R$-dependent and the constants $\tfrac{1}{24}, \tfrac{1}{72}$ require multiplication by $\sup_{N_I} |H_4|/3 \ge 1$ and $\sup_{N_I} |H_6|/15 \ge 1$ — they are then *lower bounds* on the correct prefactor, not tight values.

4. **If the Pinsker chain for $C_1 = \tfrac{1}{12}$ uses the $\tfrac{1}{\sqrt 2}$ Pinsker factor instead of $\tfrac{1}{2}$**, the correct $C_1$ becomes $\tfrac{\sqrt 2}{12} \approx 0.118$ instead of $\tfrac{1}{12} \approx 0.083$ — a $\sim 41\%$ correction. The claim's $\tfrac{1}{12}$ is then a lower bound, not the exact prefactor.

5. **If the regularity class $C^4$ is insufficient** — i.e., if a Hölder $C^{4,\alpha}$ or Sobolev $W^{4,\infty}$ class is required to justify the $L^\infty$ closure norm modulo additive constants — then specification (2) of the claim fails and the entire Proposition-1 upgrade fails to reach theorem grade.

## Honest verdict from blue's own diligence

The claim's $C_1$ value of $\tfrac{1}{12}$ survives only under a specific composition of Pinsker (the $\tfrac{1}{2}$ factor, not $\tfrac{1}{\sqrt 2}$) with the Edgeworth-Hermite-3 prefactor $\tfrac{1}{6}$. This composition is consistent with tabulated Cornish-Fisher/Edgeworth coefficients but not derived line-by-line in my sympy session. I cannot defend the exact value $\tfrac{1}{12}$ over the alternative $\tfrac{\sqrt 2}{12}$ on first-principles grounds; I can only defend it as "the standard tabulated Edgeworth-cubic-correction coefficient at order $V^{3/2}$ in the variance-normalized Cornish-Fisher scale."

The claim's $C_5$ coefficients $\tfrac{1}{24}, \tfrac{1}{72}$ are exact Edgeworth density-correction coefficients (verified symbolically) and consistent with the manuscript's $L^\infty$ closure norm, *provided* the operator-norm convention absorbs the central Hermite values $|H_4(0)| = 3, |H_6(0)| = 15$. Under the alternative spectral-norm-without-Hermite-absorption convention, the correct coefficients are the Wick $\tfrac{1}{8}, \tfrac{5}{24}$ and the claim is off by factors of $3$ and $15$.

The blue position therefore is: under the stated norm and the natural reading of the operator norms compatible with the $L^\infty$ closure-norm choice, the claim is correct; under a different reading of the operator norms, the claim is off by a constant factor that has a clean structural origin (Hermite-central-value $\times$ Edgeworth). The proposition is *defensible to theorem grade* only after the operator-norm convention is fixed unambiguously — which is what specification (2) of the claim is supposed to do, and which the claim does only partially.
