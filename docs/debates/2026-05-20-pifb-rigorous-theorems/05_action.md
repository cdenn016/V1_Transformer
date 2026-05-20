# Action — pifb-rigorous-theorems

**From verdict:** RED_WINS_NARROW

## Recommended action

Two REQUIRED edits applied. The OPTIONAL Karcher convexity-radius tightening is deferred (the verdict notes it is "editorial; the existing citation is correct in spirit").

### Edit 1 (required) — line 4734 Taylor-route equivalence deleted

Removed the sentence "An equivalent Taylor-route presentation of the dispersion constant in terms of the cubic Taylor tensor $T_3$ at the parent saddle is $C_1 = (\sqrt 2 / 3) \|T_3\|_{\mathrm{op}}/m_I^{3/2}$." Both teams' sympy verification confirmed the two expressions are dimensionally inconsistent (the Pinsker route gives $\|F\|^{3/2}/m_I^{1/2}$ with units $[\eta]^{-2}\cdot[\eta]^2/[\mathcal{F}]$ vs the Taylor route $\|T_3\|/m_I^{3/2}$ with different units). The Pinsker-route expression at line 4726 stands on its own as the load-bearing dispersion constant.

### Edit 2 (required) — line 4685 form-vs-fixed-point clarification

Added two sentences after "is itself iterable" clarifying that the iteration is form-preserving (Polchinski 1984 §2; Cardy 1996 §3.2) rather than fixed-point class-closing: at scales $s+1, s+2, \ldots$ each integration of internal modes produces a new log-determinant generator $V^{(s+1)}(Y)$ that adds to the running effective potential, with $A_{\xi\xi}^{(s+1)}$ incorporating second derivatives of $V^{(s)}$. Form-preservation is distinct from finite-parameter fixed-point class-closure, which would require additional renormalized-coupling flow analysis at the relevant fixed point.

### Edit 3 (optional, declined) — Karcher convexity radius at 4601/4605

Tightening "convex normal ball" to "convex normal ball of radius below $\tfrac{1}{2}\min(\mathrm{inj.rad}(G), \pi/\sqrt{\Delta})$" with citations to Pennec 2006 and Afsari 2011 was suggested but declined per the verdict: editorial only, existing citation correct in spirit.

## What was explicitly preserved

All seven theorem-grade results in the appendix:
- Theorem thm:rg_pushforward (partition function and observable preservation)
- Theorem thm:rg_semigroup (discrete semigroup composition)
- Theorem thm:rg_covariance (gauge covariance under compact $G = \mathrm{SO}(K)$ bi-invariant metric)
- Theorem thm:rg_exact_closure (augmented-class closure with Schur complement and log-det correction)
- Proposition thm:rg_residual (six-term schematic closure-residual bound)
- Theorem thm:rg_residual_explicit (explicit dispersion and anharmonic constants $C_1, C_5$ under three specifications — the Pinsker-route $C_1 = (\sqrt 2/12)\|F(q_I)\|^{3/2}/m_I^{1/2}$ and the Wick-coefficient $C_5 = (1/8)\|T_4\|/m_I^2 + (5/24)\|T_3\|^2/m_I^3$ stand)
- Theorem thm:rg_detector_retention (detector implies positive retention gain under Lipschitz hypothesis)

Honest scope qualifications (compact-vs-noncompact, proposition-grade $C_2,C_3,C_4,C_6$, finite-size scaling as future work, augmented-vs-strict closure) stand unchanged.

## Follow-up debates (if any)

None required. Two narrower questions remain admissible but optional:

1. **Whether the body subsection sec:meta_agent_rg accurately reflects the appendix's scope qualifications.** Manuscript-internal consistency check across the body-appendix interface.
2. **The Karcher convexity-radius citation tightening (Edit 3 above).** Editorial only.

Neither is required by the present verdict.
