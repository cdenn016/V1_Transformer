# Memo — Red / gauge-theorist / opening

## Steelman
The hedged obstructions are stated correctly and I concede them outright. For constant g the
connection transforms by the adjoint action A → g⁻¹Ag, so tr(A_μ A_ν) is invariant by trace
cyclicity and a single-copy Haar average is trivial [Nakahara2003 §10.4]. For local g(c) the
connection picks up the inhomogeneous Maurer-Cartan term A → g⁻¹Ag + g⁻¹dg, so an honest
gauge-orbit average must integrate over Map(C, G) — an infinite-dimensional functional integral
needing a gauge-fixing/regulator (Faddeev-Popov) to be well-defined
[Nakahara2003 §10.1–10.4; Peskin&Schroeder §9.4]. The non-compact SO(1,3) case carries infinite
Haar even for constant g (geometer memo). Those three statements (:2977) are right.

## Falsify (the surviving overclaim by suggestion)
The enumeration at :2992 is the residual overclaim. The text states that the GL(K,ℝ)
belief-fiber invariance "taken together with the connection-sector complexification to
GL(K,ℂ)... makes all these gauge groups (U(1), SU(2), SU(3), and the Lorentz group SO(1,3))
available as subgroups of the connection-sector gauge symmetry." The hedge that follows
addresses only *dynamical selection*: "Whether specific subgroups are dynamically selected by
free energy minimization remains an open question." That hedge does not cover the more basic
problem with the sentence: **the enumeration itself is theory-laden retrofitting.** Every compact
matrix Lie group is a subgroup of GL(K,ℂ) for K large enough — U(1), SU(2), SU(3), but equally
SU(5), SO(10), G₂, E₈, Sp(n), and uncountably many others. Listing precisely the Standard-Model
factors U(1)×SU(2)×SU(3) plus the Lorentz group, and no others, is not a structural observation
that "GL(K,ℂ) has subgroups." It is a gesture at the Standard-Model gauge group, presented in a
paragraph titled "Connection to Physical Gauge Invariance" that explicitly invokes
"electromagnetism, Yang-Mills theory, and general relativity."

The Standard-Model gauge group SU(3)×SU(2)×U(1) is not derivable from first principles; its
factors and the 19 free parameters of the theory are fixed by experiment, and "a first-principles
explanation for the three gauge groups could not be furnished so far"
[Peskin&Schroeder, *Introduction to QFT*, on the empirical fixing of the SM gauge structure;
corroborated by the experimental-determination literature, see canon below]. Nothing in the
consensus framework picks U(1)×SU(2)×SU(3) over any other compact subgroup of GL(K,ℂ): there is
no dynamical principle, no anomaly-cancellation argument, no representation-content constraint,
no chirality structure — none of the machinery that even partially constrains the SM gauge group
in actual gauge theory. The "available as subgroups" sentence therefore borrows the SM's
explanatory prestige (these are *the* physical gauge groups) while supplying none of the content
that would connect the framework to them. The fix is one sentence: state that GL(K,ℂ) contains
*all* compact Lie subgroups and that singling out the SM factors is not motivated by anything in
the construction — or delete the enumeration.

## External primary-source citation (not the manuscript)
- Peskin & Schroeder, *An Introduction to Quantum Field Theory* (1995) — the SM gauge group
  SU(3)_c × SU(2)_L × U(1)_Y is an empirical input; gauge fixing of orbit integrals via
  Faddeev-Popov, §9.4 / §16. SM gauge structure not derived from first principles.
- Faddeev–Popov procedure: gauge-orbit integration overcounts physically equivalent
  configurations and requires gauge fixing to factor out the (formally infinite) gauge-group
  volume. https://en.wikipedia.org/wiki/Faddeev%E2%80%93Popov_ghost

## Falsification condition
Wrong if the manuscript supplies, at or near :2992, a principle that singles out U(1), SU(2),
SU(3), SO(1,3) from the other compact subgroups of GL(K,ℂ) — anomaly cancellation, irrep
content, a free-energy stationarity that selects exactly these — rather than asserting they are
"available." The dynamical-selection hedge concedes there is no such principle yet, which is
precisely why the enumeration overclaims by listing them as if motivated.

## Newly-discovered canon
- "Mathematical formulation of the Standard Model" — su(3)×su(2)×u(1) algebra and the
  particle quantum numbers are experimentally determined; 19 parameters fixed by experiment.
  https://en.wikipedia.org/wiki/Mathematical_formulation_of_the_Standard_Model
- Faddeev–Popov ghost / gauge fixing — orbit-volume divergence requires gauge fixing.
  https://en.wikipedia.org/wiki/Faddeev%E2%80%93Popov_ghost
