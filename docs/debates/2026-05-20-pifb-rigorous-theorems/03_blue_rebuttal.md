# Blue Rebuttal — pifb-rigorous-theorems

## Concession

I concede R3 (dimensional inconsistency of the Taylor-route presentation at line 4734). Red's dimensional analysis is correct and the manuscript sentence as written does not survive primary-source check.

Executed sympy dimensional verification at the parent Gaussian saddle, with $[\eta]$ the internal-mode length-scale and $[\mathcal{F}_s]$ the free-energy units:

```
F_dim   = eta^{-2}            (parent Fisher = Hessian of -log q_I)
m_dim   = F_s / eta^2          (constrained Hessian eigenvalue)
T3_dim  = F_s / eta^3          (cubic Taylor tensor at saddle)

Pinsker route ||F||^{3/2} / m_I^{1/2}  units: 1 / (sqrt(F_s) * eta^2)
Taylor  route ||T_3||   / m_I^{3/2}    units: 1 /  sqrt(F_s)
Ratio (Pinsker/Taylor): eta^{-2}
```

The two presentations differ by a factor with units $[\eta]^{-2}$. They cannot simultaneously bound the same dimensionless ratio $\varepsilon_{s+1}/V_I^{3/2}$. The natural identification that would make them equivalent — $\|F(q_I)\| \sim \|T_3\|^{2/3}\,m_I^{2/3}\,[\eta]^{-2/3}$ or similar — is not derived in the appendix, the body subsection sec:meta_agent_rg, or the cited Cover-Thomas 2006 §11.6 Pinsker-Edgeworth chain. The dimensionless prefactor ratio compounds the problem: the manuscript's $\sqrt 2/3$ for the Taylor route exceeds the $\sqrt 2/12$ Pinsker prefactor by a factor of 4, so even after a hypothetical dimensional bridge the algebra would not match.

The minimal remediation is to drop the sentence "An equivalent Taylor-route presentation of the dispersion constant in terms of the cubic Taylor tensor $T_3$ at the parent saddle is $C_1 = (\sqrt 2 / 3)\|T_3\|_{\mathrm{op}}/m_I^{3/2}$" from line 4734. A stronger remediation states that a Taylor-route bound exists with prefactor $\sqrt 2/3$ under the *Gaussian-saddle identification* $\|F(q_I)\| = c \|T_3\|^{2/3}/m_I^{1/3} \cdot [\eta]^{-2/3}$ derived at the parent barycenter — but the manuscript does not currently derive this, so the cleaner edit is removal.

I also concede that the Remark at lines 4684-4686 prose "augmented multi-agent class consisting of the Gaussian quadratic KL functional class together with an explicit smooth scalar parent potential $V(Y)$ ... is itself iterable" conflates form-iterability (which the appendix proves) with class-closure in the fixed-point Wilsonian sense (which the appendix does not prove). R1 is correct that these are distinct properties and the manuscript prose does not articulate the distinction. The minimal remediation is one clarifying sentence after line 4685.

## Core attack

Red's R3 lands but its scope is narrower than red's opening implies, and red's R1 does not unseat the appendix's primary load-bearing theorems. The decisive content of the appendix is not the Taylor-route equivalence sentence at 4734 — that sentence is a single editorial parenthetical inside Theorem thm:rg_residual_explicit. The load-bearing statement of the theorem is the Pinsker-route expression at line 4726, which red did not attack and which survives:

```
C_1 = (sqrt 2 / 12) * ||F(q_I)||_{op}^{3/2} / m_I^{1/2}
```

The arithmetic check — Pinsker constant $\tfrac{1}{\sqrt 2}$ composed with Edgeworth third-cumulant coefficient $\tfrac{1}{6}$ — is $\tfrac{1}{6\sqrt 2} = \tfrac{\sqrt 2}{12}$, which sympy confirms and which red explicitly granted in E1 line 32 ("Pinsker x 1/6 = sqrt(2)/12 (arithmetic check: 1/(6 sqrt(2)) = sqrt(2)/12, correct)"). The Pinsker-Edgeworth chain in Cover-Thomas 2006 §11.6 and Hall 1992 §2.3 is canonical, and the manuscript's citation of those references at 4729 is correct. The dispersion constant theorem stands; only the parenthetical "equivalent Taylor-route presentation" sentence fails.

Red's R1 articulates a real distinction (form-iterable vs class-closed in the Polchinski-Cardy fixed-point sense) but does not break the substantive content of Theorem thm:rg_exact_closure. The Schur-complement identity at 4671 and the log-determinant correction at 4676 are mathematically correct under the conditions (i)-(v) of the body subsection — red did not contest either. What red showed is that the *naming* "augmented-class closure" suggests fixed-point closure to an unprimed reader, while the actual content is form-preservation with potential class-growth across iterations. This is a clarifying-edit obligation at line 4685, not a falsification of the theorem.

Red's R5 (Lipschitz-bound circularity) is registered honestly by the manuscript itself at lines 4742-4744: the Remark closes with "the locality of the derivation is the regime in which the closure ansatz itself holds." Red conceded this is editorial in its own falsification condition 3 ("If blue can show conditions (i)-(v) are merely renamed appendix hypotheses, the circularity dissolves"). The appendix conditions (i)-(v) at 4669 (compact-gauge, flat-transport, Gaussian-quadratic, positive-gap) ARE the conditions invoked by the Lipschitz-bound derivation at 4743. The closure ansatz regime and the Lipschitz-bound regime coincide by construction, not by argumentative circularity.

## Defense

The appendix delivers four primary-source-verifiable computational results, three structural theorems, and one honest schematic proposition. The remediation owed after this debate is two clarifying edits, not a falsification of the rigorous backbone.

**Pinsker-route C_1 prefactor stands** [CoverThomas2006 §11.6, Hall1992 §2.3]: $C_1 = \tfrac{\sqrt 2}{12}\|F(q_I)\|^{3/2}/m_I^{1/2}$ via Pinsker $\tfrac{1}{\sqrt 2}$ × Edgeworth $\tfrac{1}{6}$. Sympy: $\tfrac{1}{6\sqrt 2} = \tfrac{\sqrt 2}{12}$.

**Wick coefficients on the integrated free energy stand** [BenderOrszag1999 §6.5, Wong2001]. Direct Wick counting on $\langle z^6\rangle = 15$ with prefactor $\tfrac{1}{2!(3!)^2} = \tfrac{1}{72}$ gives 6 sunset contractions $= \tfrac{6}{72} = \tfrac{1}{12}$ and 9 double-bubble contractions $= \tfrac{9}{72} = \tfrac{1}{8}$, summing to $\tfrac{15}{72} = \tfrac{5}{24}$. Red conceded this in E4 ("mathematically defensible ... Zinn-Justin 2002 §10 uses similar nomenclature"). Sympy confirms the rational arithmetic. The $\tfrac{1}{8}\|T_4\|/m_I^2$ coefficient stands by the same Wick count on $\langle z^4\rangle = 3$ with prefactor $\tfrac{1}{4!} = \tfrac{1}{24}$, giving $\tfrac{3}{24} = \tfrac{1}{8}$.

**Density-vs-integrated reconciliation stands** [BenderOrszag1999 §6.5]. The Hermite-evaluation arithmetic $|H_4(0)| = 3$ and $|H_6(0)| = 15$ gives $\tfrac{1}{8} = 3 \cdot \tfrac{1}{24}$ and $\tfrac{5}{24} = 15 \cdot \tfrac{1}{72}$ at the centerpoint exactly. The manuscript at 4737 honestly registers that for Fisher-normal radius $R > 1.126$ the sup-norm of $H_n$ exceeds the central value, so the centerpoint identification fails as an $L^\infty$ bound — this is the right scope qualifier and is not a defect.

**Pushforward and semigroup theorems stand** [Bogachev2007 §3.6, Folland1999 §2.6]. Theorem thm:rg_pushforward is the standard change-of-variables for pushforward measures; Theorem thm:rg_semigroup is the pre-image composition rule. Red did not contest either.

**Gauge covariance theorem stands on its compact-group hypothesis** [Karcher1977, Afsari2011, Amari2016]. The proof of Theorem thm:rg_covariance via bi-invariance of the Riemannian metric and KL-invariance under common pushforward is correct for $G = \mathrm{SO}(K)$. The opening at line 4595 explicitly flags that the noncompact $\mathrm{GL}^+(K)$ case requires a gauge slice or Radon-Nikodym correction — this is the right honest scope. Red conceded the Karcher convexity-radius citation is editorial at E4 line 63 ("This is an editorial flag, not a falsification").

**Schur-complement / log-determinant closure stands as form-preservation** [Wilson1974, Cardy1996, Polchinski1984]. The block-matrix Gaussian integration at lines 4674-4676 is textbook: integrating out internal Gaussian modes produces the Schur complement $A_{\mathrm{eff}} = A_{YY} - A_{Y\xi}A_{\xi\xi}^{-1}A_{\xi Y}$ plus $\tfrac{\tau}{2}\log\det' A_{\xi\xi}(Y)$. What red's R1 correctly observes is that "form-preservation" is the canonical Wilsonian statement, and that "class-closure" in the fixed-point sense requires additionally the renormalized-coupling flow analysis [Polchinski1984 Eq. 2.2; Cardy1996 §3.2]. The manuscript's "augmented Gaussian multi-agent class is itself iterable" is true in the form-preservation sense, but the prose at 4685 should add a clarifying sentence such as "the augmentation iterates as a sequence of log-determinant generators $V^{(s)}, V^{(s+1)}, \ldots$ whose flow is analyzed in the standard Wilsonian sense via renormalized couplings at the fixed point." This is a small edit at line 4685, not a falsification of the Schur identity.

**Lipschitz/detector retention stands conditionally and is honestly registered** [Amari2016, Fenichel1979]. Theorem thm:rg_detector_retention is a clean if-then: given the Lipschitz bound (hypothesis), the detector condition (hypothesis), and the gap condition eq:rg_app_detector_retention, then $\Delta_I > 0$. The Remark at 4742-4744 honestly notes that the Lipschitz bound is derivable locally under the same closure-ansatz conditions that the rest of the appendix uses. Red's R5 reduces to: the appendix's standalone-rigor advertisement at 4592-4595 should be read as "rigorous conditional on the closure-ansatz regime," not "rigorous unconditionally" — which is what the appendix actually says ("locality of the derivation is the regime in which the closure ansatz itself holds," 4744).

**Net remediation owed:**
1. Line 4734: drop the sentence "An equivalent Taylor-route presentation ... $C_1 = (\sqrt 2/3)\|T_3\|_{\mathrm{op}}/m_I^{3/2}$." The Pinsker-route expression at line 4726 stands on its own and is the load-bearing statement. (REQUIRED, R3.)
2. Line 4685: add a clarifying sentence distinguishing form-iterability from fixed-point class-closure, with an explicit pointer to the Polchinski/Cardy renormalized-coupling-flow framework as the canonical RG closure. (REQUIRED, R1.)
3. Line 4605: tighten the convexity-radius citation to $\tfrac{1}{2}\min(\mathrm{inj.rad}, \pi/\sqrt\Delta)$ per Afsari 2011. (OPTIONAL editorial, R2.)

The seven theorems of the appendix survive after these two surgical edits; the appendix continues to deliver the rigorous backbone the body subsection sec:meta_agent_rg invokes.

I cannot defend the appendix as currently written without the line-4734 edit. With it, the appendix's mathematical content is sound on the canon.
