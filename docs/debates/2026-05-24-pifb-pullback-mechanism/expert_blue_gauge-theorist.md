# Expert Memo (Blue) — Gauge Theorist

## Lens

Connection one-forms, gauge transformations, equivariance vs invariance. The disputed object is the horizontal block $g^{\mathrm{tw}}_{\mu\nu} = \kappa(A^{(i)}_\mu, A^{(i)}_\nu)$ with $A^{(i)} = U_i^{-1}dU_i$ (:2739), and the live question is whether its gauge-noninvariance (:2768) disqualifies it from being part of a bona-fide bundle metric.

## Steelman of the claim

The horizontal block is a connection-dependent quadratic form on the base. Connection-dependent quadratic forms are *expected* to transform under change of trivialization — that is not a pathology, it is the generic behavior of any object built from a connection rather than from curvature. The manuscript does not hide this: it computes the transformation law explicitly (:2768), declines the illegitimate escape of pretending $\mathrm{tr}(FF)$ rescues it (correctly, since $F=0$ in Regime I makes the Yang–Mills invariant identically zero), and routes the genuinely gauge-invariant content to a downstream consensus/Haar-averaged construction. This is the disciplined way to handle a non-invariant ingredient: disclose its transformation, refuse a fake invariant, and locate the real invariant elsewhere.

## Derivation from external canon

The connection one-form is the prototypical gauge-covariant — not gauge-invariant — object. Under a change of section/trivialization $s \to s\cdot g$, the local connection form transforms inhomogeneously:
$$A \;\longrightarrow\; g^{-1}Ag + g^{-1}dg,$$
the Maurer–Cartan inhomogeneity being the second term [Nakahara2003 §10.4 "local form of the connection / gauge transformation"; KobayashiNomizu Vol. I §II.1, the transformation rule for the connection form under change of local section]. This is canon, and it is exactly the transformation the manuscript writes at :2768. Any quadratic form $\kappa(A_\mu, A_\nu)$ built pointwise from $A$ therefore picks up the cross terms $\kappa(A_\mu, \partial_\nu g\, g^{-1})$ etc. — precisely what the manuscript lists. A connection-built quadratic form being non-invariant under change of trivialization is the rule, not the exception; the only base-space scalars built from $A$ that *are* gauge-invariant are those built from the curvature $F = dA + \tfrac12[A,A]$, e.g. $\mathrm{tr}(F\wedge\star F)$ [external_canon_math.md §"Gauge invariance vs gauge equivariance"; Nakahara2003 §10.5 on curvature as the gauge-covariant field strength].

The manuscript correctly refuses to launder $\kappa(A,A)$ into a Yang–Mills invariant. In Regime I the connection is pure gauge, $A = U^{-1}dU$ with $U=\exp\phi$, so $F = dA + \tfrac12[A,A] \equiv 0$ identically [the flatness of a pure-gauge / Maurer–Cartan connection is standard — a connection of the form $g^{-1}dg$ has zero curvature, the Maurer–Cartan structure equation $d\theta + \tfrac12[\theta,\theta]=0$, Nakahara2003 §10.4]. With $F=0$ the Yang–Mills density vanishes and supplies no invariant. The manuscript states exactly this (:2768) rather than papering over it. An honest disclosure that the horizontal block is frame-dependent, plus an explicit refusal of the only fake invariant available, is the methodologically correct treatment.

The Kaluza–Klein lineage reinforces the point. In Kaluza–Klein the total-space metric built from a connection is invariant under the *vertical* gauge action (fiber-preserving), but its horizontal block, written in a local trivialization as a tensor on the base, depends on the chosen section — different sections give different base-tensor representatives related by the gauge transformation [Bleecker1981, gauge transformations of the Kaluza–Klein metric]. The base-space metric one reads off is section-dependent; the invariant object lives on the total space or after a quotient. The manuscript's routing of invariance to the consensus metric (a population average over gauge frames) is the analogue of taking the invariant quotient, restricted to the gauge orbit realized within a cognitive species.

## What is genuinely defensible vs what is interpretive

Defensible without reservation: (i) $A = U^{-1}dU$ is a legitimate (flat, pure-gauge) connection one-form; (ii) $\kappa(A,A)$ being gauge-noninvariant is expected and disclosed, not a defect; (iii) the "tw not YM" labeling is correct — it is a connection-dependent form, not a curvature invariant; (iv) the refusal of the $\mathrm{tr}(FF)$ escape hatch is honest given $F\equiv 0$.

Interpretive (flag, do not oversell): the claim that distinct gauge fixings of the *same* agent yielding distinct horizontal contributions is "consistent with the framework's treatment of distinct gauge frames as generating distinct pulled-back geometries" (:2768) reframes a gauge redundancy as physical content. Standard gauge theory treats a change of trivialization as a redundancy of description, not a change of physics. The manuscript's move — declaring the frame to carry agent-physical meaning (the "frame-as-state" reading, :2828) — is a substantive interpretive commitment, not a theorem. It is internally consistent but it is a postulate.

## Falsification condition

The horizontal-block claim fails if either: (a) $A=U^{-1}dU$ fails to be a connection one-form — it does not; a pure-gauge form satisfies the connection axioms and is flat [Nakahara2003 §10.4]; or (b) the manuscript claimed $\kappa(A,A)$ to be gauge-invariant or a Yang–Mills invariant — it explicitly does not, and explicitly disclaims the $\mathrm{tr}(FF)$ route at :2768. The only way to defeat the gauge-theoretic core is to show that a bundle metric must have a gauge-invariant horizontal block to count as a metric on $E$. That is false: a metric on the total space $E$ need only be a metric on $E$; its representation as a base-space tensor in a trivialization is allowed to be section-dependent, exactly as in Kaluza–Klein. The construction survives.

## Newly-discovered canon

- [Nakahara2003 §10.4] — local connection form, gauge transformation law $A \to g^{-1}Ag + g^{-1}dg$, Maurer–Cartan structure equation giving $F=0$ for pure-gauge connections.
- [KobayashiNomizu Vol. I §II.1] — connection form transformation under change of local section; curvature as the gauge-covariant object.
- [Bleecker1981] — gauge transformations of the Kaluza–Klein total-space metric; section-dependence of the base-tensor representative is standard.
