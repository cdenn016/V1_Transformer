# Red Opening — pifb-rigorous-theorems

## Steelman (opposing position)

The Rigorous Theorem-Level Statements appendix at `Participatory_it_from_bit.tex:4592-4767` is an honestly-scoped, mathematically-correct rigorous backbone for the body subsection sec:meta_agent_rg: it proves three standard structural facts (pushforward preservation, semigroup composition, gauge covariance under bi-invariant SO(K) metric) using elementary measure theory and bi-invariance, registers the closure question as augmented-class rather than strict, derives the explicit dispersion and two-loop anharmonic coefficients via Pinsker-Edgeworth and integrated Wick contractions, and registers C_2, C_3, C_4, C_6, the noncompact GL+(K) case, and finite-size scaling as future-work with the correct epistemic register.

## Position

The appendix overclaims at two specific points that the standard literature does not support as stated. First, line 4734's "equivalent Taylor-route presentation" of the dispersion constant $C_1$ asserts that $C_1 = (\sqrt 2/12)\|F(q_I)\|_{\mathrm{op}}^{3/2}/m_I^{1/2}$ and $C_1 = (\sqrt 2/3)\|T_3\|_{\mathrm{op}}/m_I^{3/2}$ are interchangeable expressions for the same bound. They are not: dimensional analysis at the saddle gives the two expressions different units, so the equivalence is conditional on an unstated identification between the parent-belief Fisher tensor and the internal-mode third-derivative tensor. Second, the Remark at 4684-4686 collapses "form-iterability" of the Wilsonian log-det augmentation into "augmented-class closure" in a fixed-point sense; the two are distinct, and Polchinski 1984 / Cardy 1996 do not establish the latter for the construction the appendix presents.

## Evidence

### E1 — Dimensional incompatibility of the two C_1 presentations (R3, load-bearing)

The appendix's Theorem thm:rg_residual_explicit at lines 4724-4734 states two presentations of the dispersion constant $C_1$ and labels them "equivalent." Direct dimensional analysis at the parent Gaussian saddle shows they have different units.

The parent Fisher information $F(q_I)$ is the Hessian of $-\log q_I$ evaluated at the mean of $q_I$, with units of inverse variance: $[F(q_I)] = [\eta]^{-2}$, where $[\eta]$ is the length-scale of the internal mode.

The internal-mode constrained Hessian eigenvalue $m_I$ is by construction the Hessian of $\mathcal{F}_s$ along the constrained internal subspace ($A_{\xi\xi}$ at lines 4661-4665, with $m_I$ defined via the constrained gap at the equation reference \eqref{eq:rg_constrained_gap} cited at 4710): $[m_I] = [\mathcal{F}_s]/[\eta]^2$.

The cubic Taylor tensor $T_3$ at the saddle is the third derivative of $\mathcal{F}_s$ along internal modes: $[T_3] = [\mathcal{F}_s]/[\eta]^3$.

Substituting:

- Pinsker-route: $\|F(q_I)\|^{3/2}/m_I^{1/2}$ has units $([\eta]^{-2})^{3/2} / ([\mathcal{F}_s]/[\eta]^2)^{1/2} = [\mathcal{F}_s]^{-1/2}\,[\eta]^{-2}$.
- Taylor-route: $\|T_3\|/m_I^{3/2}$ has units $([\mathcal{F}_s]/[\eta]^3) / ([\mathcal{F}_s]/[\eta]^2)^{3/2} = [\mathcal{F}_s]^{-1/2}$.

The two expressions differ by a factor with units $[\eta]^{-2}$. They cannot both bound the same dimensionless ratio $\varepsilon_{s+1}/V_I^{3/2}$ unless an unstated identification ties Fisher information of the belief to a third-derivative tensor of the free energy. The natural candidate identification $\|F(q_I)\| \sim \|T_3\|^{2/3}\,m_I^{2/3}\,[\eta]^{-2/3}\,\ldots$ is not derived in the appendix, and no such identification appears in the standard Pinsker-Edgeworth chain in Cover-Thomas 2006 §11.6.

Executed verification (sympy / dimensional analysis transcript):
```
Pinsker x 1/6 = sqrt(2)/12  (arithmetic check: 1/(6 sqrt(2)) = sqrt(2)/12, correct)
||F(q_I)||^{3/2} / m_I^{1/2} = [F_s]^{-1/2} [eta]^{-2}
||T_3||         / m_I^{3/2} = [F_s]^{-1/2}
Difference: factor of [eta]^{-2}, the Fisher-natural inverse length-squared scale.
```

Canon citation: Cover & Thomas 2006, *Elements of Information Theory* §11.6 (Pinsker), and Hall 1992, *The Bootstrap and Edgeworth Expansion* §2.3 (Edgeworth coefficient $\tfrac{1}{6}$ on $T_3$). The Pinsker-Edgeworth chain in these references binds total variation distance through $\sqrt{KL}$ and proceeds via Edgeworth density correction at cubic order; the resulting $C_1$ is a *bound* in the parent-Fisher operator norm, not algebraically equal to the Taylor-tensor third-derivative coefficient.

### E2 — Form-iterability vs class-closure (R1, supporting)

The Remark at 4684-4686 closes: "the augmentation by the explicit log-determinant potential $V(Y)$ is the canonical Wilsonian effective-action correction at the Gaussian saddle and is itself iterable." This conflates two distinct properties.

*Form-iterability:* at scale $s+1$, integrating out internal modes produces (a) a Schur-complement quadratic in $Y$ and (b) a log-det term $V^{(s)}(Y) = \tfrac{\tau}{2}\log\det' A_{\xi\xi}^{(s)}(Y)$. The functional form "Gaussian quadratic + log-det" is preserved.

*Class-closure:* at scale $s+2$, integrating out the next layer of internal modes around the new parent saddle produces a new Schur complement *and* a new log-det $V^{(s+1)}(Y) = \tfrac{\tau}{2}\log\det' A_{\xi\xi}^{(s+1)}(Y)$. But $A_{\xi\xi}^{(s+1)}$ now incorporates second derivatives of $V^{(s)}(Y)$ — it is no longer a pure Gaussian quadratic Hessian. The "augmented multi-agent Gaussian KL class" defined at line 4669 as "the Gaussian quadratic KL functional class together with an explicit smooth scalar parent potential $V(Y)$" therefore *grows* under iteration: at scale $s+n$ the parent potential is a sum of distinct log-det generators built from the cumulative Schur stack.

Wilsonian fixed-point closure in Polchinski 1984 (Eq. 2.2 ff) and Cardy 1996 (§3.2) requires a *finite-parameter* family stable under the RG step, with relevant/irrelevant coupling flow analyzed via the linearization at the fixed point. The appendix's "augmented class" is form-preserved but parameter-growing; this is weaker than Wilsonian closure and the manuscript's framing at 4685 ("the substantive content of the theorem is the augmented-class form rather than strict Gaussian-KL closure") does not articulate the growth explicitly.

The body subsection sec:meta_agent_rg invokes the appendix for "rigorous closure," and the language of Theorem thm:rg_exact_closure at line 4667 ("Augmented-class closure with explicit log-determinant correction") suggests fixed-point closure to an unprimed reader. The honest statement is closer to "form preservation under one RG step, with the augmentation class growing under iteration."

### E3 — Lipschitz-bound derivation depends on body conditions (R5, flag)

The Remark at lines 4742-4744 states that the Lipschitz bound used in Theorem thm:rg_detector_retention is "locally derivable under the closure-ansatz conditions (i)-(v) by first-order Taylor expansion of the parent free energy around the high-coherence configuration." Conditions (i)-(v) are not stated in the appendix; they are referenced from the body subsection sec:meta_agent_rg. The appendix-as-standalone-rigorous-backbone is therefore conditional on body-section content for one of its load-bearing hypotheses.

The Remark closes with "the locality of the derivation is the regime in which the closure ansatz itself holds" — this is internally consistent but the appendix's standalone-rigor advertisement at 4592-4595 ("This appendix gives the rigorous backbone for the body subsection") is not fully delivered: the Lipschitz-bound regime is exactly the closure-ansatz regime.

### E4 — Concessions after primary-source verification

I checked two attack candidates that did not survive verification, and concede them:

- The decomposition $5/24 = 1/12 + 1/8$ as "sunset (1/12) and double-bubble (1/8) two-loop topologies" at line 4734 is mathematically defensible. Direct Wick counting on $\langle z^6\rangle = 15$ gives 6 sunset contractions (three cross-vertex propagators) and 9 tadpole-pair contractions (one cross-propagator with self-loops at each vertex), splitting as $6/72 + 9/72 = 1/12 + 1/8 = 5/24$ with the prefactor $\tfrac{1}{2!}\cdot(\tfrac{1}{3!})^2 = \tfrac{1}{72}$. Nomenclature ("double bubble" for the 9-pairing topology) is defensible since both self-loops form bubbles connected by one line; Zinn-Justin 2002 §10 uses similar nomenclature.
- The convexity-ball citation at line 4605 ("locally unique on convex balls of radius below the injectivity radius~\cite{karcher1977riemannian}") is loose: the Karcher 1977 / Afsari 2011 result requires radius below $\tfrac{1}{2}\min(\mathrm{inj.rad}, \pi/\sqrt{\Delta})$ with $\Delta$ the upper sectional-curvature bound, not below the full injectivity radius. This is an editorial flag, not a falsification — the citation is correct in spirit and the constraint qualitatively matches what the appendix uses.

## Falsification conditions

This position is wrong if any of the following holds:

1. **R3 is wrong** if the appendix discloses (or the manuscript elsewhere derives) the identification $\|F(q_I)\| \sim \|T_3\|^{2/3}\,m_I^{2/3}$ or equivalent dimensional bridge that makes $\|F\|^{3/2}/m^{1/2}$ and $\|T_3\|/m^{3/2}$ measure the same bound on $\varepsilon_{s+1}/V_I^{3/2}$. The natural candidate is that at the parent Gaussian saddle, the third derivative of the free energy $\mathcal{F}_s$ is the third derivative of the KL-Lagrangian and reduces to a Fisher-Riemannian curvature term of fixed dimension. If this derivation appears in the body subsection sec:meta_agent_rg or the prior debate at `docs/debates/2026-05-19-prop-1-tighter-quantitative/`, I concede.

2. **R1 is wrong** if Polchinski 1984 Eq. 2.2 (or Cardy 1996 §3.2) establishes that a single log-det augmentation of a Gaussian quadratic class is itself reabsorbed into a renormalized Gaussian quadratic + a single log-det at every subsequent step (i.e., the second log-det $V^{(s+1)}$ is equivalent to a re-parameterization of $V^{(s)}$ plus a re-shifted quadratic). If the Wilsonian effective-action canonical-form result is exactly "fixed point in form-modulo-renormalization" rather than "form-preserved with class growth," R1 collapses to a nomenclature dispute.

3. **R5 is wrong** if the conditions (i)-(v) of the body subsection are themselves first-principles derivable from the appendix's pushforward + closure setup (i.e., the appendix's setup is logically self-sufficient and the body conditions are derivative). If blue can show conditions (i)-(v) are merely renamed appendix hypotheses, the circularity dissolves and the Remark is just redundant cross-referencing.

The judge can rule for blue overall if any of R3 or R1 is broken under primary-source check; R5 alone is editorial.
