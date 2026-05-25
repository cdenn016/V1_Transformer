# Blue memo — gauge-theorist — Phase 3 (rebuttal)

## Target attacks in `02_red_opening.md`

- Vector 3 (sub-claim 5): Red argues that PIFB:eq:consensus_metric (PIFB:2934) is given an equation label, a name ("the collective consensus metric"), and downstream references that "exceed its admitted heuristic target status," while the underlying functional integral "sits structurally on top of the Gribov ambiguity" [Gribov 1978 *Nucl. Phys. B* 139:1; Singer 1978 *Comm. Math. Phys.* 60:7] and the non-compact $\mathrm{SO}(1,3)$ infinite-Haar obstruction [Folland 1995 *A Course in Abstract Harmonic Analysis* Ch. 2]. Red's specific charge: "a numbered, named, downstream-referenced equation is the canonical signature of a completed construction, not a heuristic target."

## Concession (one red argument that holds under canon)

Red is canonically correct that the consensus-metric functional integral is a hard problem. [Gribov 1978 *Nucl. Phys. B* 139:1; Singer 1978 *Comm. Math. Phys.* 60:7 "Some Remarks on the Gribov Ambiguity"; Henneaux–Teitelboim 1992 *Quantization of Gauge Systems* Ch. 19] establish that in non-abelian gauge theory, no globally well-defined gauge-fixing condition exists; "a gauge fixing submanifold may not intersect a gauge orbit at all or it may intersect it more than once." The Faddeev-Popov procedure is locally well-defined, globally ambiguous. The standard remedy is restriction to the first Gribov region, which does not solve the problem globally. [Folland 1995 Ch. 2] additionally establishes that a non-compact locally compact group has an infinite left-invariant Haar measure; for $\mathrm{SO}(1,3)$ specifically the Haar measure is infinite even for constant elements. Both obstructions are real, and the PIFB:2928 "needs a gauge fixing or a regulator" prose understates the difficulty in the sense that the difficulty is a 50-year open problem rather than an engineering choice.

I concede that the manuscript's prose at 2928 ("needs a gauge fixing or a regulator to be well-defined") is closer to a textbook understatement than a full registration of the canonical difficulty. The honest rewording would be "is one of the central open problems in non-abelian gauge theory [Gribov 1978; Singer 1978], compounded in the $\mathrm{SO}(1,3)$ case by the additional infinite-Haar obstruction [Folland 1995 Ch. 2]." This is an editorial sharpening, not a structural collapse.

## Core attack (red's load-bearing weakness under canon)

Red's Vector 3 mis-characterizes canonical practice in non-abelian gauge theory. *Numbered, named, downstream-referenced equations that stand in for conditional or heuristic constructions are standard canonical practice* in the gauge theory literature, not the signature of a completed construction. Canonical examples:

- **The Faddeev-Popov functional integral itself.** [Faddeev–Popov 1967 *Phys. Lett. B* 25:29] write down a labeled, named functional integral over the gauge orbit with a labeled, named Faddeev-Popov determinant — *and* the construction is precisely the one [Singer 1978] showed is globally ill-defined. The literature has retained the Faddeev-Popov equation as a labeled, named, downstream-referenced object for fifty years despite the Gribov ambiguity. Standard textbook treatment ([Peskin–Schroeder 1995 §16.2; Weinberg 1996 *Quantum Theory of Fields II* §15.4]) labels and names the construction explicitly with the understanding that it is regulator-conditional.

- **The BRST functional integral.** [Becchi–Rouet–Stora 1976 *Ann. Phys.* 98:287; Tyutin 1975] developed a labeled, named functional integral with BRST cohomology that is rigorous in finite dimensions, conditional in infinite-dimensional non-abelian gauge theory. [Henneaux–Teitelboim 1992 Ch. 19] treats this as canonical practice.

- **The Yang-Mills vacuum / θ-vacuum.** [Callan–Dashen–Gross 1976 *Phys. Lett. B* 63:334; Jackiw–Rebbi 1976 *Phys. Rev. Lett.* 37:172] write the Yang-Mills vacuum as a labeled, named functional object — without anyone today claiming the construction is non-perturbatively complete.

- **The Wheeler-DeWitt equation.** [DeWitt 1967 *Phys. Rev.* 160:1113; Wheeler 1968] is a labeled, named equation in canonical quantum gravity — labeled, named, downstream-referenced — and is openly conditional on resolution of operator-ordering, factor-ordering, the problem of time, and the constraint algebra closure. Half a century of canonical-gravity literature treats it as a defined object with conditional well-definedness.

Each of these is a *canonical practice* of giving a label and a name to a construction the field knows is regulator-conditional or globally ambiguous. The PIFB:2934 eq:consensus_metric with its 2937 explicit "heuristic target rather than a completed observable ... gauge-invariance is conditional on a regulator whose construction is left to future work" registration is canonical in this same register.

The PIFB:2943 cross-reference imports the "metaphysical interpretation rather than a derivation, and the hypothesis may not be falsifiable from within the framework" register from §gauge_invariance_consensus — this is in the same paragraph that names the gauge groups available as subgroups of the connection-sector symmetry. Red's claim that the downstream references at 2723 ("the consensus / Haar-averaged construction") and 2937 ("the candidate gauge-invariant fallback for the agent-frame-dependent pullbacks") "carry weight as if it were a defined object" mis-reads canonical practice: in non-abelian gauge theory, named conditional constructions are referenced downstream precisely as conditional objects, with the conditional status carried by surrounding hedges. The PIFB:2937 surrounding hedges ("would by construction produce", "if defined and finite", "remains a heuristic target rather than a completed observable", "conditional on a regulator", "where the prose elsewhere describes ... those statements should be read in this conditional sense") are exactly that canonical carrying-of-conditional-status.

The "candidate gauge-invariant fallback" language is even structurally more honest than the Faddeev-Popov label, because it explicitly names the construction as a *candidate* and a *fallback* — not as the gauge-invariant content.

## Defense (citation that strengthens blue's position)

Sub-claim 5 calibration mesh holds under canonical non-abelian gauge theory practice:

- [Nakahara 2003 §10.3, §10.4 "Curvature"] — pure-gauge connections have vanishing curvature; $F = dA + \frac{1}{2}[A,A] = 0$ when $A = U^{-1} dU$. PIFB:2722 Yang-Mills escape-hatch denial is canonical. The manuscript correctly refuses to absorb the gauge non-invariance via $\mathrm{tr}(F_{\mu\nu} F^{\mu\nu})$ because $F \equiv 0$ in Regime I.
- [Kobayashi–Nomizu Vol. I §III.2; Nakahara 2003 §10.3] — the Maurer-Cartan piece $g^{-1} dg$ in the local-gauge transformation of a connection. PIFB:2723 identifies it correctly.
- [Henneaux–Teitelboim 1992 *Quantization of Gauge Systems* Ch. 19] — canonical treatment of gauge-orbit averaging in non-abelian gauge theory; Faddeev-Popov locally well-defined, globally ambiguous; standard remedy is restriction to the first Gribov region. The PIFB:2928 regulator-conditional flag, while a textbook understatement, is the canonical hygiene move.
- [Faddeev–Popov 1967 *Phys. Lett. B* 25:29; Singer 1978; Peskin–Schroeder 1995 §16.2] — the historical and modern canonical practice of labeling conditional gauge-fixing constructions.
- [Folland 1995 *A Course in Abstract Harmonic Analysis* Ch. 2] — non-compact Haar measure is infinite. PIFB:2928 cites this correctly.

The narrow concession is the prose understatement at 2928 (the regulator-conditional flag could be sharpened to name Gribov by name); the structural-existence claim at PIFB:2934 (the construction is what a gauge-invariant consensus metric *would* look like under a successful regulator) is canonically licensed by labeled-but-conditional non-abelian-gauge-theory practice.

## Newly-discovered canon

- **Faddeev–Popov 1967, *Phys. Lett. B* 25:29 ("Feynman Diagrams for the Yang-Mills Field").** The original labeled, named, downstream-referenced gauge-orbit-averaging functional integral in non-abelian gauge theory — and the very construction [Singer 1978] showed is globally ill-defined. Canonical practice of giving a name to a regulator-conditional construction.

- **Becchi–Rouet–Stora 1976, *Ann. Phys.* 98:287; Tyutin 1975.** BRST symmetry as a labeled, named, downstream-referenced functional construction. Canonical practice of giving a name to a construction that is rigorous in finite dimensions and conditional in non-abelian gauge theory.

- **DeWitt 1967, *Phys. Rev.* 160:1113 ("Quantum Theory of Gravity I").** The Wheeler-DeWitt equation as a labeled, named equation in canonical quantum gravity, conditional on resolution of operator-ordering, the problem of time, and constraint algebra closure. Half a century of canonical-gravity literature treats it as a defined object with conditional well-definedness.

- **Peskin–Schroeder 1995 §16.2; Weinberg 1996 *QTF II* §15.4.** Canonical textbook treatment of Faddeev-Popov as a labeled, named conditional construction.

- **Henneaux–Teitelboim 1992, *Quantization of Gauge Systems*, Ch. 19.** Canonical treatment of gauge-orbit averaging in non-abelian gauge theory.
