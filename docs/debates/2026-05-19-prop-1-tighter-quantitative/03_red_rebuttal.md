# Red Rebuttal — prop-1-tighter-quantitative

## Concession

Blue's central-Hermite arithmetic is correct. Executed via sympy 1.14.0:

```
3 * (1/24) = 1/8     vs canonical Wick 1/8: match
15 * (1/72) = 5/24   vs canonical Wick 5/24: match
He_4(0) = 3, He_6(0) = -15  (probabilists' Hermite, |He_6(0)| = 15)
```

The arithmetic identity $\frac{1}{8} = |He_4(0)| \cdot \frac{1}{24}$ and $\frac{5}{24} = |He_6(0)| \cdot \frac{1}{72}$ is not a coincidence; it is the centerpoint-extraction relating Wick log-$Z$ coefficients to Edgeworth density-correction coefficients. I grant blue's Step 2 sympy output that $h_1(\xi)$ contains $\frac{1}{24}T_4 \xi^4$ and $\frac{1}{72}T_3^2 \xi^6$ as standard tabulated Edgeworth prefactors `[AmariNagaoka2000 Ch. 4]`. Sub-claim C's *Edgeworth-level* coefficients are correctly identified by blue as the Edgeworth normalization.

I also grant blue's own concession on $C_1$: the Pinsker chain does not derive a clean $\frac{1}{2}$ multiplicative factor; the canonical Pinsker constant is $\frac{1}{\sqrt 2}$, giving $\frac{\sqrt 2}{12}$, not $\frac{1}{12}$. Blue flagged this; I extend it below.

## Core attack

The central-Hermite reconciliation defends the claim **only at the centerpoint $\xi = 0$**, not as an $L^\infty$ norm bound on the parent neighborhood $\mathcal{N}_I$. The claim asserts a $L^\infty(\mathcal{N}_I)$ bound (specification 1, matching `Attention/Participatory_it_from_bit.tex:4429`, Eq.~\ref{eq:rg_closure}). An $L^\infty$ norm is the supremum over the neighborhood, not the centerpoint value. Blue's "central-Hermite-value" reconciliation conflates these.

Executed sympy verification (probabilists' Hermite polynomials $He_4(\xi) = \xi^4 - 6\xi^2 + 3$, $He_6(\xi) = \xi^6 - 15\xi^4 + 45\xi^2 - 15$):

```
R       sup|He4|_[-R,R]   He_4(0)   ratio       sup|He6|_[-R,R]   He_6(0)   ratio
0.500   3.000             3         1.000        15.000             15        1.000
1.000   3.000             3         1.000        16.000             15        1.067
1.500   5.438             3         1.812        23.246             15        1.550
2.000   6.000             3         2.000        23.246             15        1.550
3.000   30.000            3        10.000       103.246             15        6.883
4.000   163.000           3        54.333       961.000             15       64.067
```

The sup-norm of $|He_4|$ on $[-R, R]$ exceeds $|He_4(0)| = 3$ at the threshold $R = \sqrt{3 - \sqrt 3} \approx 1.126$ (exact sympy root of $He_4(\xi) = -3$), reaches $|He_4(\sqrt 3)| = 6$ at $R = \sqrt 3 \approx 1.732$ — twice the central value — and grows like $R^4$ for $R \gg 1$. For $He_6$ the sup crosses the central value at $R \approx 1$ and reaches $\sim 23$ by $R \approx 1.5$, with $R^6$ growth thereafter.

**Consequence.** The claim's coefficients $\frac{1}{24}$ on $\|T_4\|_{\mathrm{op}}/m_I^2$ and $\frac{1}{72}$ on $\|T_3\|_{\mathrm{op}}^2/m_I^3$ are *centerpoint* Edgeworth coefficients. To convert them into an $L^\infty(\mathcal{N}_I)$ bound on the free-energy *functional* $-\tau h_1(\xi)$ (the object actually bounded by Eq.~\ref{eq:rg_closure}), one must multiply by $\sup_{\xi \in \mathcal{N}_I} |He_n(\xi)|/|He_n(0)|$. That ratio is $1$ only on a tiny ball $R \lesssim 1$ (specifically $R < 1.126$ for $T_4$, $R \lesssim 1$ for $T_3^2$). For any larger neighborhood the coefficients $\frac{1}{24}, \frac{1}{72}$ are *not* upper bounds — they are *lower* bounds, and grow with the neighborhood radius as $R^4, R^6$ respectively.

This is blue's own falsification condition #3, restated with quantitative thresholds: the manuscript at line 4432 specifies the parent neighborhood only qualitatively ("the slow-manifold patch where the agent ansatz is fitted") with conditions (i)-(v) of which (ii) is $V_I \ll 1$. The condition $V_I \ll 1$ in KL is not equivalent to the Fisher-normal radius being below $\sqrt{3-\sqrt 3}$. The Fisher-normal radius corresponds to $\|\delta\mu\|_F$ in standardized coordinates; via the leading-order KL expansion $V_I \approx \frac{1}{2}\|\delta\mu\|_F^2$ (Eq. 01_evidence §Pinsker, line 70), $V_I \ll 1$ is consistent with $R \approx \sqrt{2V_I}$ which can be $\gg \sqrt{3-\sqrt 3}$ when $V_I$ is order-$0.3$ rather than infinitesimal. There is no specification in the manuscript pinning $\mathcal{N}_I$ to the radius regime where blue's centerpoint reconciliation holds.

## Defense

Blue's operator-norm convention defense (Evidence §4) "the operator norm is normalized to absorb the central Hermite values" rescues the arithmetic but at the price of changing the meaning of $\|T_3\|_{\mathrm{op}}, \|T_4\|_{\mathrm{op}}$ from the standard spectral norm (Horn & Johnson 2013 §4: largest absolute value of the multilinear form on the unit ball) to a non-standard "Hermite-absorbed" norm. Specification (2) of the claim states "bounded third-derivative tensor $\|T_3\|_{\mathrm{op}}$ (operator norm of the trilinear form on $\mathcal{N}_I$), and bounded fourth-derivative tensor $\|T_4\|_{\mathrm{op}}$" — this is the *standard* operator norm wording, not a Hermite-absorbed redefinition. Either the claim must be amended to specify the non-standard convention, or the coefficients must change to the Wick $\frac{1}{8}, \frac{5}{24}$.

Blue's own Step 5 of `sympy_blue_verify_c1.py` produces three options for $C_1$:

```
Option A: 1/12        decimal 0.0833 (the claim)
Option B: sqrt(2)/12  decimal 0.1179 (canonical Pinsker)
Option C: sqrt(2)/3   decimal 0.4714 (cumulant-scaling chain)
```

My sympy independently confirms $\frac{1}{6\sqrt 2} = \frac{\sqrt 2}{12} \approx 0.1179$, a factor of $\sqrt 2 \approx 1.414$ larger than $\frac{1}{12}$. The claim's value is consistent with no clean first-principles derivation that blue has produced or that I can produce; it is a tabulated number whose justification chain ("$\frac{1}{6} \cdot \frac{1}{2}$") replaces the canonical Pinsker $\frac{1}{\sqrt 2}$ with $\frac{1}{2}$ without citation.

The dimensional-reduction issue compounds this. The 1D Bender-Orszag result is $\frac{1}{8}, \frac{5}{24}$ on $T_4/H^2, T_3^2/H^3$ (verified by blue's Step 1 sympy: `<V4> = T4/(8*H^2)`, `<V3^2> = 5*T3^2/(12*H^3)`). The multidimensional Wong / Zinn-Justin result is $\frac{1}{8}, \frac{1}{12}, \frac{1}{8}$ on the three *separate* contraction topologies (01_evidence §Wong 2001 lines 34-40). The claim's $C_5$ uses a single coefficient $\frac{1}{72}$ on $\|T_3\|_{\mathrm{op}}^2$ without specifying which contraction topology — yet in the multidimensional setting (clusters $|I| \ge 2$ giving internal-mode spaces of dimension $|I| - 1 \ge 1$), the two cubic-cubic topologies cannot be combined under an operator-norm bound without further specification of how the contraction patterns relate to $\|T_3\|_{\mathrm{op}}$. The Edgeworth density-correction coefficient $\frac{1}{72}$ is a *scalar* (1D) coefficient on $\kappa_3^2 H_6(\xi)$; promoting it to a multidimensional operator-norm bound on $\|T_3\|_{\mathrm{op}}^2$ requires either (a) restricting to one-dimensional internal-mode space (i.e., clusters of size $|I| = 2$ exactly), or (b) an additional dimensional factor that the claim's $C_5$ does not include.

**Reconciliation.** Blue's central-Hermite identity defends the claim *under a specific non-standard convention*: (i) the closure norm $\|\cdot\|_\mathcal{B}$ is reinterpreted as the *centerpoint value* of the Edgeworth density correction rather than the $L^\infty$ supremum over $\mathcal{N}_I$; (ii) the operator norms $\|T_3\|_{\mathrm{op}}, \|T_4\|_{\mathrm{op}}$ absorb the central Hermite values $|He_4(0)| = 3, |He_6(0)| = 15$ into the tensor norm; (iii) the dimensional setting is restricted to scalar internal-mode space ($|I| = 2$); (iv) the Pinsker step is replaced by a tabulated Cornish-Fisher coefficient lookup rather than a derived bound. Under this fourfold non-standard convention, $\frac{1}{12}, \frac{1}{24}, \frac{1}{72}$ are consistent. Under the standard $L^\infty(\mathcal{N}_I)$ closure norm specified at `Attention/Participatory_it_from_bit.tex:4429`, the standard spectral operator norm on $T_3, T_4$, the multidimensional internal-mode setting that Proposition 1 applies to (arbitrary cluster size $|I|$), and the standard Pinsker constant $\frac{1}{\sqrt 2}$, the claim's coefficients are off by:

- $C_1$: factor of $\sqrt 2 \approx 1.414$ (low by $\sim 29\%$),
- $C_5$ quartic: factor of $3 \cdot (\sup |He_4|/3)$, exceeding $3$ as soon as $\mathcal{N}_I$ has radius $> 1.126$ in Fisher-normal coordinates,
- $C_5$ cubic-squared: factor of $15 \cdot (\sup |He_6|/15)$, exceeding $15$ as soon as the radius exceeds $\sim 1$.

The claim therefore cannot be upgraded to theorem grade as stated. It can be upgraded under blue's non-standard convention, but that convention departs from the wording of specifications (1) and (2) in the claim itself and from the canonical operator-norm definition. Specification (2) needs to be rewritten to say "Hermite-absorbed operator norm on the symmetric tensors $T_3, T_4$" and specification (1) needs to be rewritten to say "centerpoint value at the parent saddle" rather than "parent-neighborhood $L^\infty$ norm" — at which point the claim becomes a *different* claim than the one this debate is adjudicating.

Sub-claim B ($C_1 = \frac{1}{12}$): falsified at the claimed exact-value level; the canonical Pinsker constant gives $\frac{\sqrt 2}{12}$, blue conceded this explicitly. Sub-claim C ($C_5$ with $\frac{1}{24}, \frac{1}{72}$): falsified as an $L^\infty(\mathcal{N}_I)$ upper bound unless the parent neighborhood is restricted to radius $R < 1$ in Fisher-normal coordinates, a restriction the manuscript does not impose. Sub-claim A: the closure norm is well-defined as the manuscript states it, but the regularity class specification omits the dimensional-reduction issue (single-mode vs multi-mode internal space) that determines whether $\frac{1}{72}$ vs $\frac{1}{12} + \frac{1}{8} = \frac{5}{24}$ is the correct cubic-squared coefficient.
