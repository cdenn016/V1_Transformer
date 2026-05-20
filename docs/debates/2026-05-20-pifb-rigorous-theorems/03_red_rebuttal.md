# Red Rebuttal — pifb-rigorous-theorems

## Concession

I grant the following components of blue's opening on the strength of their sympy verification and standard-canon citation:

- **E4 (5/24 Wick decomposition).** Blue's sympy expansion of $-\tau\log(Z/Z_0)$ for the cubic-quartic action $S(x) = \tfrac{m}{2}x^2 + \tfrac{g_3}{6}x^3 + \tfrac{g_4}{24}x^4$ produces exactly $\tfrac{g_4 \tau^2}{8 m^2} - \tfrac{5 g_3^2 \tau^2}{24 m^3} + O(\tau^3)$, matching Bender-Orszag 1999 §6.5 and Zinn-Justin 2002 §10.3. The arithmetic identity $\tfrac{1}{12} + \tfrac{1}{8} = \tfrac{5}{24}$ for the sunset and double-bubble two-loop topologies is textbook. Sub-claim 5's $C_5$ decomposition holds.
- **E5 (Pinsker-Edgeworth chain for the primary $C_1$).** $\tfrac{1}{\sqrt 2} \cdot \tfrac{1}{6} = \tfrac{\sqrt 2}{12}$. The arithmetic and the canonical references are correct.
- **E6 (Hermite reconciliation and sup-norm threshold).** $|H_4(0)| = 3$, $|H_6(0)| = 15$, $R = \sqrt{3 - \sqrt 3} \approx 1.126$. Blue's sympy chain confirms the manuscript's centerpoint-only identification correctly fails as an $L^\infty$ bound beyond that radius.
- **E1 (pushforward/semigroup correctness).** Sub-claim 1 is textbook measure theory (Bogachev 2007 §3.6, Folland 1999 §2.6), correctly stated and proved at `:4623-4625` and `:4632-4641`.
- **E2 (gauge covariance under compact $G$).** Sub-claim 2 follows from bi-invariance plus KL-invariance under common invertible pushforward plus M-projection convexity (Kobayashi-Nomizu Vol. I §4.2; Cover-Thomas 2006 Thm 2.4.1). The hypotheses are explicitly stated at `:4650`.
- **E10 (scope honesty).** The five honest registrations (`:4595`, `:4684-4686`, `:4734`, `:4763-4764`, `:4767`) are present.

The Pinsker-Edgeworth-derived primary form $C_1 = (\sqrt 2/12)\,\|F(q_I)\|^{3/2}_{\rm op}/m_I^{1/2}$ at line 4726 stands. My opening sub-claims R2 (Karcher), R4 (Wick), R5 (Lipschitz scope), and R6 (reference measure) do not survive blue's defense and I drop them.

## Core attack

The load-bearing weakness in blue's opening is its F3 concession itself: blue admits the "equivalent Taylor-route presentation" sentence at `:4734` is undefendable under the current manuscript prose. That concession is correct but understates the size of the published claim. The exact published sentence is

> "An equivalent Taylor-route presentation of the dispersion constant in terms of the cubic Taylor tensor $T_3$ at the parent saddle is $C_1 = (\sqrt 2 / 3) \|T_3\|_{\rm op}/m_I^{3/2}$."

with no qualifier — no "in whitened coordinates," no "up to a parent-saddle structural identity," no "at the canonical normalization $\|T_3^{\rm wh}\| = \tfrac{1}{4}\,\|F^{\rm wh}\|^{3/2}$."

**Dimensional verification (sympy).** Treating $F$ and $m_I$ as carrying length-dimension $L^{-2}$ and $T_3$ as $L^{-3}$ (the natural dimensional assignment for the curvature and the cubic at a saddle in scalar variables), the two presentations have different units:

```
||F||^{3/2} / m_I^{1/2}  ~  L^{-3} / L^{-1}  =  L^{-2}
||T_3||      / m_I^{3/2}  ~  L^{-3} / L^{-3}  =  L^{0}   (dimensionless)
```

These cannot be "equivalent" as written. Matching the two presentations requires (i) restricting to Fisher-normal / whitened coordinates where $F^{\rm wh} = I$, and (ii) imposing the specific saddle identity $\|T_3^{\rm wh}\| = \tfrac{1}{4}\,\|F^{\rm wh}\|^{3/2}$ (executed sympy below):

```python
>>> from sympy import sqrt, Rational, simplify
>>> ratio = (sqrt(2)/12) / (sqrt(2)/3)
>>> simplify(ratio)
1/4
>>> simplify((sqrt(2)/3) * Rational(1,4))
sqrt(2)/12
```

Neither (i) nor (ii) is stated in the appendix. Blue's E5b correctly diagnoses this but understates the editorial obligation: a sentence that asserts mathematical *equivalence* between two manifestly differently-dimensioned expressions is not "a registered weakness" — it is a misleading published claim. The required edit is either:

(a) Delete the Taylor-route sentence at `:4734`. The Pinsker-Edgeworth route is sufficient and is what carries the proof. This is the cleanest fix.

(b) Replace the sentence with: "In Fisher-normal (whitened) coordinates at the parent saddle, the cubic tensor satisfies $\|T_3^{\rm wh}\| = \tfrac{1}{4}\,\|F^{\rm wh}\|^{3/2}$ at the canonical normalization adopted here, giving the equivalent presentation $C_1 = (\sqrt 2/3)\,\|T_3^{\rm wh}\|/m_I^{3/2}$." This requires a derivation of the $1/4$ identity that the appendix currently lacks.

Either way, the appendix as published does not deliver what `:4734` advertises. Sub-claim 5 of the claim is partially false: the *5/24 = 1/12 + 1/8 decomposition* is mathematically standard (blue's E4 is correct), but the *Taylor-route equivalent presentation* of $C_1$ is not standard and is not derived in the manuscript.

## Defense

My remaining live attack is on the augmented-class framing of Theorem `thm:rg_exact_closure`. Blue's E3 and falsification condition F1 frame the question as whether the Wilsonian augmentation is "iterable in a form-preserving sense." Blue argues that Polchinski 1984 §2 and Zinn-Justin 2002 §10.2-10.4 give a fluctuation-determinant correction "preserved under iteration because each subsequent integration produces another log-det that adds to the local potential." Read narrowly, that is correct: the *operator class* (Gaussian quadratic + smooth scalar potential of $Y$) is form-preserving.

But the manuscript's published prose at `:4669` does not say "form-preserving" — it says

> "the augmented multi-agent class consisting of the Gaussian quadratic KL functional class together with an explicit smooth scalar parent potential $V(Y) := \tfrac{\tau}{2}\log\det'\!A_{\xi\xi}(Y)$"

and at `:4685`

> "the augmentation by the explicit log-determinant potential $V(Y)$ is the canonical Wilsonian effective-action correction at the Gaussian saddle and is itself iterable."

The natural reading of "the Gaussian quadratic KL functional class together with an explicit smooth scalar parent potential" is that the class has two named elements: the Gaussian-quadratic piece and one specific scalar potential $V(Y)$. Under iteration:

- At step $s+1$ the new constrained internal Hessian $A_{\xi\xi}^{(s+1)}(Y)$ involves second derivatives of the entire $\mathcal{F}_{s+1}$, which at step $s+1$ already includes $V^{(s)}(Y) = \tfrac{\tau}{2}\log\det' A_{\xi\xi}^{(s)}(Y)$.
- The new scalar potential $V^{(s+1)}(Y) = \tfrac{\tau}{2}\log\det' A_{\xi\xi}^{(s+1)}(Y)$ is therefore the log-determinant of a Hessian that depends on $V^{(s)}$ through second derivatives.
- After $n$ iterations, $V^{(n)}(Y)$ is a nested log-determinant of an operator built from log-determinants of operators built from log-determinants...

This is form-preserving in the loose sense that every step still produces "a smooth scalar potential of $Y$." But this is not what an outside reader of `:4669` is likely to understand by "the augmented class consisting of... together with an explicit smooth scalar parent potential $V(Y)$" — the natural reading is that $V$ is a single named element, not an arbitrary iterated log-det functional.

The required edit is a one-sentence clarification. Either:

(c) Replace `:4669` with: "the exact renormalized free energy lies in the augmented multi-agent class of Gaussian quadratic KL functionals plus a smooth scalar potential $V(Y)$ of $Y$, with $V^{(s)}(Y) = \tfrac{\tau}{2}\log\det'\!A_{\xi\xi}^{(s)}(Y)$ at scale $s$; the *form* of the augmentation is preserved under iteration in the Wilsonian sense, but the explicit functional form of $V^{(s)}$ becomes a nested log-determinant of operators that themselves depend on earlier $V^{(s')}$."

This is what blue's defense actually argues; the manuscript just needs to say so.

The Wilsonian canon does not disagree. Polchinski 1984 §2 explicitly discusses how the effective action's *coupling constants* renormalize at each step into the same operator basis, but the *functions* in front of those operators are flow trajectories, not fixed elements. Zinn-Justin 2002 §10.4 says the same for the loop-counting parameter. Calling the class "closed" without distinguishing "form-preserving" from "fixed-point closed" risks a reader confusing the construction with strict-Gaussian closure — which the Remark at `:4684-4686` correctly admits is empty in the operational regime.

The defense holds: blue's reading of the canon is right, but the manuscript prose as published at `:4669` does not communicate that reading, and a one-sentence form-vs-fixed-point clarification is required. Combined with the Taylor-route edit at `:4734`, two surgical edits are needed before the appendix delivers what it advertises.

I cannot falsify the rest of the claim under the current evidence. The pushforward, semigroup, gauge-covariance, Pinsker-Edgeworth-primary, 5/24 Wick decomposition, Hermite-reconciliation, Karcher-uniqueness, and detector-retention components all survive blue's defense.
