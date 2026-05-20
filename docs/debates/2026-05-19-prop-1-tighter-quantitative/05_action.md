# Action — prop-1-tighter-quantitative

**From verdict:** REMAND with corrected coefficient values

Per-sub-claim: A BLUE (norm and regularity class hold); B RED ($C_1$ off by $\sqrt 2$); C RED ($C_5$ coefficients are Edgeworth-density not integrated-Wick).

## Recommended action

The upgrade *program* survives — the structural form of the tightening (closure norm, regularity class, $V^{3/2}$ scaling, quartic-plus-squared-cubic decomposition, operator-norm conventions on $T_3, T_4$, Laplace-derived constants) is correct. The specific numerical coefficients $\frac{1}{12}, \frac{1}{24}, \frac{1}{72}$ in the present claim are wrong under the standard conventions; both teams' sympy sessions converged on corrected values.

### Corrected coefficient values (verdict-mandated)

Under the manuscript's existing $L^\infty(\mathcal{N}_I)$ closure norm and the standard spectral operator norm on $T_3, T_4$:

- $C_1 = \dfrac{\sqrt 2}{12} \cdot \dfrac{\|F(q_I)\|_{\mathrm{op}}^{3/2}}{m_I^{1/2}}$ (Edgeworth-route via canonical Pinsker $\tfrac{1}{\sqrt 2}$).
- $C_1 = \dfrac{\sqrt 2}{3} \cdot \dfrac{\|T_3\|_{\mathrm{op}}}{m_I^{3/2}}$ (Taylor-route with explicit $T_3$ tensor at the parent saddle).
- $C_5 = \dfrac{1}{8} \cdot \dfrac{\|T_4\|_{\mathrm{op}}}{m_I^2} + \dfrac{5}{24} \cdot \dfrac{\|T_3\|_{\mathrm{op}}^2}{m_I^3}$ (integrated Wick coefficients).

Both teams' sympy sessions converged on these values; the calculation is settled.

### Manuscript edits required

To upgrade Proposition 1 to a theorem-grade statement with explicit constants, the following edits to `Attention/Participatory_it_from_bit.tex` are required:

1. Replace the schematic $C_1, C_5$ at lines 4548-4551 with the corrected values above. (Constants $C_2, C_3, C_4, C_6$ remain schematic in this round.)
2. Insert an explicit statement of the operator-norm convention in specification (2): standard spectral norm of the matricization, not "Hermite-absorbed."
3. Insert an explicit specification of the parent neighborhood radius in specification (1): either bound the Fisher-normal radius $R$ that defines $\mathcal{N}_I$, or accept the sup-norm growth $R^4$ for $T_4$ and $R^6$ for $T_3^2$ as part of the bound (in which case the constants get an additional $R^4, R^6$ factor that the present claim omits).
4. Add citations to `[BenderOrszag1999 §6.5 Eq. 6.5.4]` and `[Wong2001 §IX.5 Eq. 5.04]` as the source of the integrated Wick coefficients $\tfrac{1}{8}$ and $\tfrac{5}{24}$.
5. Add a citation to `[CoverThomas2006 §11.6]` for the Pinsker constant $\tfrac{1}{\sqrt 2}$ used in $C_1$.

## Alternative path (claim-as-written survives only here)

The claim's original values $\frac{1}{12}, \frac{1}{24}, \frac{1}{72}$ may be retained ONLY IF the proposition is rewritten to all four of:

- The closure norm is the *centerpoint value* at the parent saddle rather than the $L^\infty(\mathcal{N}_I)$ supremum.
- A non-standard "Hermite-absorbed" operator norm on $T_3, T_4$ absorbs the central Hermite values $|H_4(0)| = 3$ and $|H_6(0)| = 15$ into the tensor norm.
- The internal-mode space is restricted to scalar (cluster size $|I| = 2$).
- $C_1$ uses a tabulated Cornish-Fisher coefficient in place of a derived Pinsker bound.

This is a substantively different claim than the one debated. The verdict does not recommend this path.

## Follow-up debates

None required for $C_1$ or $C_5$ — the verdict settles them with corrected values. The following are *optional* extensions the user may schedule at their discretion:

1. **Math-mode debate on $C_2, C_3, C_4, C_6$.** The present debate scoped to $C_1, C_5$ as the structurally most-constrained; analogous Laplace-derived values for the holonomy, edge-marginal, off-diagonal-Hessian, and non-Gaussian constants would complete the theorem-grade upgrade. This is the natural next step.

2. **Math-mode debate on the multidimensional Wick resolution.** In $|I| - 1$-dimensional internal-mode spaces, the 1D coefficient $\frac{5}{24}$ on $T_3^2$ resolves into a sunset topology $\frac{1}{12}$ and a double-bubble topology $\frac{1}{8}$. A tight multidimensional bound would either sum these under the spectral operator-norm convention, or specify separate norms for each contraction pattern. The verdict leaves this as a refinement.

3. **Manuscript edit verification debate.** Once the corrected values are written into the manuscript, an explicit re-verification (against `[BenderOrszag1999, Wong2001]`) by sympy would confirm the upgrade is theorem-grade. This is optional and would be a "check-the-write-up" round.

## Open issues registered (no follow-up debate required)

- The conditional structure of the upgraded proposition (small dispersion $V_I \ll 1$, positive constrained gap $m_I > 0$, bounded $T_3, T_4$, slow/fast separation) is unchanged; only the constants are tightened.
- Sub-claim A is decisive on the norm and regularity-class specifications; no further work needed there.
- The verdict explicitly licenses Proposition 1's current schematic form to remain as-is in the manuscript pending the corrected-coefficient edit. The earlier RG-construction debate's paragraph-heading edit ("Structural decomposition of the closure residual") and proposition-grade labeling continue to apply.
