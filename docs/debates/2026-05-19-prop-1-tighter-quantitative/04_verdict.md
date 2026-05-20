# Verdict — prop-1-tighter-quantitative

## Outcome

**REMAND with corrected coefficient values.**

Per-sub-claim outcomes:

- **Sub-claim A (closure norm and regularity class): BLUE_WINS.** The $L^\infty$ closure norm modulo additive constants on $\mathcal{N}_I$ is the manuscript's own definition at `Attention/Participatory_it_from_bit.tex:4372`, and the $C^4$ multi-agent Gaussian KL functional class with bounded $\|F\|_{\mathrm{op}}, \|T_3\|_{\mathrm{op}}, \|T_4\|_{\mathrm{op}}$ is a coherent regularity class for fourth-order saddle-point analysis. Red did not falsify either specification at the level of well-definedness; red's objections targeted the constants under that norm, not the norm itself. Sub-claim A survives.
- **Sub-claim B ($C_1 = \tfrac{1}{12}\|F\|^{3/2}/m_I^{1/2}$): RED_WINS.** Both sides agree the honest Pinsker chain gives $\tfrac{\sqrt 2}{12}$, not $\tfrac{1}{12}$. Blue conceded this explicitly in the opening (`02_blue_opening.md` Evidence §5) and again in the rebuttal (`03_blue_rebuttal.md` Concession). The claimed value $\tfrac{1}{12}$ is off by a factor of $\sqrt 2$ under the chain blue itself proposed and verified.
- **Sub-claim C ($C_5 = \tfrac{1}{24}\|T_4\|/m_I^2 + \tfrac{1}{72}\|T_3\|^2/m_I^3$): RED_WINS.** The integrated post-Laplace $-\log(Z/Z_0)$ coefficients are $\tfrac{1}{8}$ and $\tfrac{5}{24}$ as verified independently by both sympy sessions. The claim's $\tfrac{1}{24}, \tfrac{1}{72}$ are the Edgeworth density-correction coefficients on $H_4$ and $H_6$, not the integrated free-energy coefficients that the closure norm of `eq:rg_closure` (an $L^\infty$ norm on the integrated $\mathcal{F}_{s+1}$, which has already had internal modes integrated out by `eq:rg_laplace`) sees. Blue's central-Hermite reconciliation $\tfrac{1}{8} = 3 \cdot \tfrac{1}{24}$ and $\tfrac{5}{24} = 15 \cdot \tfrac{1}{72}$ recovers the claim's values only at the saddle centerpoint $\xi = 0$, but the closure norm is the supremum over $\mathcal{N}_I$, not the centerpoint value. Red's executed sympy at lines 22-33 of `03_red_rebuttal.md` demonstrates $\sup_{[-R,R]}|H_4|$ exceeds $|H_4(0)| = 3$ for $R > 1.126$ and grows like $R^4$; for any neighborhood larger than the threshold $R \approx 1$ in Fisher-normal coordinates, the claim's coefficients are not upper bounds. The manuscript's $V_I \ll 1$ qualifier does not pin $\mathcal{N}_I$ to this radius regime: $V_I = 0.3$ already corresponds to $R \approx \sqrt{0.6} \approx 0.77$ which is below threshold, but $V_I = 1$ gives $R \approx \sqrt 2 \approx 1.41$ which is above threshold. The claim does not impose $V_I \ll 0.3$ or equivalent.

## Decisive evidence

Three independent executed-sympy results, agreed by both teams, settle the math:

1. **Integrated free-energy coefficients are $\tfrac{1}{8}$ and $\tfrac{5}{24}$.** Both `02_red_opening.md` lines 17-51 and `02_blue_opening.md` Step 1 lines 22-29 of the sympy output independently compute $\langle V_4 \rangle = T_4/(8H^2)$ and $\langle V_3^2 \rangle = 5T_3^2/(12H^3)$, giving $-\log(Z/Z_0)$ coefficients $-\tfrac{1}{8}$ on $T_4/H^2$ and $+\tfrac{5}{24}$ on $T_3^2/H^3$. This matches `[BenderOrszag1999 §6.5 Eq. 6.5.4]` verbatim.

2. **Pinsker constant is $1/\sqrt 2$, not $1/2$.** Blue's Step 5 of `sympy_blue_verify_c1.py` (referenced at `02_blue_opening.md` Evidence §5) produces $\tfrac{1}{6} \cdot \tfrac{1}{\sqrt 2} = \tfrac{\sqrt 2}{12} \approx 0.118$, and blue's rebuttal explicitly concedes this is the honest value (`03_blue_rebuttal.md` Concession, lines 21-29). Red's rebuttal confirms via independent sympy that $\tfrac{1}{6\sqrt 2} = \tfrac{\sqrt 2}{12}$ (`03_red_rebuttal.md` lines 44-51).

3. **The sup-norm of $|H_4|$ on $[-R, R]$ exceeds the central value $|H_4(0)| = 3$ for $R > 1.126$.** Red's rebuttal sympy table (`03_red_rebuttal.md` lines 22-33) tabulates the sup-norm explicitly: $|H_4(\sqrt 3)| = 6$ at $R = \sqrt 3$, doubling the central value, and growth like $R^4$ thereafter. Blue's defense that "the small-deviation regime $R = O(1)$ to which the manuscript's closure ansatz applies" makes the central value dominant (`02_blue_opening.md` Evidence §4) is not supported by the manuscript's actual specification of $\mathcal{N}_I$, which is qualitative ("the slow-manifold patch where the agent ansatz is fitted" with $V_I \ll 1$) and does not impose $R < 1.126$ as a hard radius cutoff.

## Reasoning

Both teams executed sympy independently and converged on the same integrated Wick coefficients $\tfrac{1}{8}$ and $\tfrac{5}{24}$. This is the established mathematical fact, agreed by both sides. The arithmetic identity $\tfrac{1}{8} = 3 \cdot \tfrac{1}{24}$ via $H_4(0) = 3$ and $\tfrac{5}{24} = 15 \cdot \tfrac{1}{72}$ via $|H_6(0)| = 15$ is also agreed. The disagreement reduces to a single substantive question: does the manuscript's closure norm (an $L^\infty$ norm on the *integrated* free-energy functional $\mathcal{F}^{\mathrm{exact}}_{s+1} - \mathcal{F}^{\mathrm{agent}}_{s+1}$ modulo constants, on the parent neighborhood $\mathcal{N}_I$) see the integrated Wick coefficients or the Edgeworth density coefficients?

The integrated-vs-density distinction is settled by the manuscript's own Eq.~\ref{eq:rg_laplace} at line 4396: $\mathcal{F}_{s+1}(Y_I) = \mathcal{F}_{\mathrm{eff}}(Y_I) + \tfrac{\tau}{2}\log\det{}' H_I^\perp(Y_I) + \mathrm{const} + \mathcal{E}_{\mathrm{anh}}$. The closure residual sees $\mathcal{F}_{s+1}$, which is the *post-Laplace* free energy with internal modes already integrated out. The anharmonic correction $\mathcal{E}_{\mathrm{anh}}$ is the integrated $-\log(Z/Z_0)$ correction in the standard saddle-point asymptotic expansion, and its leading-order coefficients are the Wick coefficients $\tfrac{1}{8}$ and $\tfrac{5}{24}$, not the Edgeworth density coefficients $\tfrac{1}{24}$ and $\tfrac{1}{72}$. Blue's rebuttal concedes this explicitly at lines 6-19 of `03_blue_rebuttal.md`: "The leading post-Laplace correction inherits the Wick coefficients $\tfrac{1}{8}$ and $\tfrac{5}{24}$. ... The claim's $\tfrac{1}{24}, \tfrac{1}{72}$ are the Edgeworth-Hermite density-expansion coefficients on $H_4$ and $H_6$, which appear in the density correction $p(x) = \phi(x)[1 + h_1(x)]$, not in the integrated $-\log(Z/Z_0)$ correction."

Blue's only remaining defense of the original claim values (the "operator-norm absorbs central Hermite values" defense at `02_blue_opening.md` Evidence §4) requires reinterpreting $\|T_3\|_{\mathrm{op}}, \|T_4\|_{\mathrm{op}}$ from the standard spectral norm to a non-standard "Hermite-absorbed" norm. Specification 2 of the claim explicitly uses the standard wording "operator norm of the trilinear form on $\mathcal{N}_I$" without modification, so the non-standard convention is not what the claim states. Furthermore, the L^∞-versus-centerpoint problem (red's sympy table at `03_red_rebuttal.md` lines 22-33) means the Hermite-absorption defense fails for any neighborhood larger than $R \approx 1.126$, which the manuscript does not exclude.

The norm-convention and dimension-dependence issue (point 5 of the dispatch) is decided in favor of red on this evidence: the claim's silence on (a) whether the multidimensional internal-mode Wick contractions are summed over topologies (which gives $\tfrac{5}{24}$ in 1D, splitting into $\tfrac{1}{12} + \tfrac{1}{8}$ in $\ge 2$D), and (b) whether the operator norm absorbs the central Hermite values, leaves the claim's coefficients underspecified. Red's argument at `03_red_rebuttal.md` lines 52-53 that the 1D coefficient $\tfrac{5}{24}$ resolves in $\ge 2$D into two separate contraction patterns that cannot be combined under a single operator-norm bound without specifying topology, is sound and uncontested in blue's rebuttal.

The upgrade *program* (the structural form of the proposition's quantitative tightening — $V^{3/2}$ scaling, quartic-plus-squared-cubic decomposition, Laplace-derived constants, operator-norm conventions on $T_3, T_4$) survives both rebuttals. The specific numerical coefficients $\tfrac{1}{12}, \tfrac{1}{24}, \tfrac{1}{72}$ do not. This is a partial-REMAND case: sub-claim A holds, the upgrade program is the right next step for the proposition, but sub-claims B and C require corrected coefficient values before the proposition can be promoted to theorem grade.

## Action

The manuscript edit at `Attention/Participatory_it_from_bit.tex:4548-4551` (the statement of Proposition 1) must replace the schematic constants $C_1, C_5$ with the following corrected explicit values, derived under the manuscript's existing $L^\infty(\mathcal{N}_I)$ closure norm and the standard spectral operator norm on $T_3, T_4$:

$$ C_1 = \frac{\sqrt 2}{12} \cdot \frac{\|F(q_I)\|_{\mathrm{op}}^{3/2}}{m_I^{1/2}} \qquad \text{(Edgeworth-route via canonical Pinsker } \tfrac{1}{\sqrt 2}\text{)} $$

or equivalently, under the Taylor-route through the cubic Taylor of $\mathcal{F}_s$ at the parent saddle,

$$ C_1 = \frac{\sqrt 2}{3} \cdot \frac{\|T_3\|_{\mathrm{op}}}{m_I^{3/2}} \qquad \text{(Taylor-route, with explicit } T_3 \text{ tensor)}, $$

and

$$ C_5 = \frac{1}{8} \cdot \frac{\|T_4\|_{\mathrm{op}}}{m_I^2} + \frac{5}{24} \cdot \frac{\|T_3\|_{\mathrm{op}}^2}{m_I^3} \qquad \text{(integrated Wick coefficients)}. $$

These values are derived from `[BenderOrszag1999 §6.5 Eq. 6.5.4]` for the 1D case and `[Wong2001 §IX.5 Eq. 5.04]` for the multidimensional case (where the $\tfrac{5}{24}$ resolves into separate $\tfrac{1}{12}$ sunset and $\tfrac{1}{8}$ double-bubble topologies that must be summed under the spectral operator-norm bound). Both teams' sympy sessions converge on these values; the calculation is settled.

The claim's original values $\tfrac{1}{12}, \tfrac{1}{24}, \tfrac{1}{72}$ may be retained ONLY IF the proposition is rewritten to (i) specify that the closure norm is the *centerpoint value* at the parent saddle rather than the $L^\infty(\mathcal{N}_I)$ supremum, (ii) specify a non-standard "Hermite-absorbed" operator norm convention on $T_3, T_4$ that absorbs the central Hermite values $|H_4(0)| = 3, |H_6(0)| = 15$ into the tensor norm, (iii) restrict to scalar internal-mode space (cluster size $|I| = 2$), and (iv) use a tabulated Cornish-Fisher coefficient for $C_1$ in place of a derived Pinsker bound. This is a substantively different claim and is not what specifications (1)-(2) of the present claim assert.

Concrete manuscript edits required:

1. Replace the schematic $C_1, C_5$ at `Attention/Participatory_it_from_bit.tex:4548-4551` with the corrected values above.
2. Insert an explicit statement of the operator-norm convention in specification (2) — standard spectral norm, not Hermite-absorbed.
3. Insert an explicit specification of the parent neighborhood radius in specification (1) — either bound the Fisher-normal radius $R$ that defines $\mathcal{N}_I$, or accept that the sup-norm growth $R^4$ for $T_4$ and $R^6$ for $T_3^2$ is part of the bound.
4. Add a citation to `[BenderOrszag1999 §6.5 Eq. 6.5.4]` and `[Wong2001 §IX.5 Eq. 5.04]` as the source of the integrated Wick coefficients $\tfrac{1}{8}$ and $\tfrac{5}{24}$.
5. Add a citation to `[CoverThomas2006 §11.6]` for the Pinsker constant $\tfrac{1}{\sqrt 2}$ used in $C_1$.

Follow-up debates needed:

- **prop-1-multidimensional-cubic-topologies:** the claim's silence on whether the multidimensional cubic-cubic contractions are summed (yielding $\tfrac{5}{24}$) or kept separate (yielding $\tfrac{1}{12}$ sunset + $\tfrac{1}{8}$ double-bubble on different tensor norms) needs adjudication if internal-mode space dimension exceeds 1. The present verdict has assumed the 1D-aggregated $\tfrac{5}{24}$ for definiteness; the multidimensional case requires separate treatment.
- **prop-1-c2-c3-c4-c6:** the present claim scoped out $C_2, C_3, C_4, C_6$. A tighter proposition requires explicit values for these as well. Recommend separate debate per constant or a single debate covering all four.
- **prop-1-parent-neighborhood-radius:** specification (1) needs a quantitative pinning of the $\mathcal{N}_I$ radius in Fisher-normal coordinates. The condition $V_I \ll 1$ permits radii up to $R \approx \sqrt 2$, which exceeds the threshold $R \approx 1.126$ where $|H_4|$ starts growing beyond its central value. Either the radius is pinned below threshold or the sup-norm growth must be absorbed into the constants.

After these edits and follow-up debates, Proposition 1 can be upgraded to theorem grade with the corrected constants.
