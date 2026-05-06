# Gauge Redundancy vs. Frame-as-State Audit

**Manuscript:** `Participatory_it_from_bit.tex`
**Date:** 2026-05-06
**Reviewer:** independent gauge-theory check

## Verdict

**Partially confirmed.** The manuscript does state both readings, and at the level of individual sentences several pairs are in plain tension. However, the manuscript also contains an explicit "Gauge-invariance disclosure" paragraph (line 1545) and a "Connection to Physical Gauge Invariance" subsection (line 1709) that together attempt a clean resolution: individual frames are pure gauge, the gauge-noninvariant pullback `tr(A_μ A_ν)` is admitted to be agent-frame-dependent, and the genuinely "objective" geometry is relocated to the consensus / gauge-averaged metric and to curvature invariants. The remaining defect is therefore not a hidden contradiction but a terminological one: the body text still calls frames "arbitrary choices with no physical content" (line 235) while the qualia section (line 2732) and the ontological-geometry subsection (lines 1546, 1553-1554) treat those same frames as the carriers of phenomenal structure. The framework can be made consistent by sharpening the language; the math itself is internally coherent because it ultimately disclaims gauge invariance for the per-agent metric. A referee will still object to the unqualified "no physical content" claim and to the use of the word "gauge" in two senses.

## Quoted Passages, Classified

### Interp 1 (gauge as redundancy / no physical content)

**Line 235** (`\subsection{Principal Bundles and Gauge Freedom}`):

> "More generally, agents maintain beliefs using internal coordinate systems, conceptual frameworks, or measurement conventions - their gauge frames. **These frames are arbitrary choices with no physical content,** yet they're necessary for representing probabilistic information."

This is the strongest Interp-1 claim. It denies the frame any ontological role.

**Line 426** (`\subsubsection{Gauge Covariance: The Fundamental Principle}`):

> "All physically or cognitively meaningful quantities must be invariant under simultaneous gauge transformations of all agents. ... no observable should change. This is the information-theoretic analog of general covariance ... **Individual frames $\phi_i$ are arbitrary (gauge freedom)**, but the relative pairing $U_i U_j^{-1}$ is physically meaningful."

Interp 1: only invariants are physical; individual frames are surplus.

**Line 1689** (`\subsection{Collective Geometry and Gauge Invariance}`):

> "This approach has a critical flaw: the average depends on each agent's arbitrary gauge frame choice $\phi_i$. ... **Gauge freedom represents redundancy in description, not physical degrees of freedom**, therefore the observable geometry must not depend on these arbitrary choices."

Interp 1, stated in textbook form: gauge = redundancy.

**Line 1707**:

> "This construction is gauge-invariant by design as no agent's arbitrary frame choice affects the collective geometry. ... This is the closest analog to 'objective reality' in the framework."

Interp 1, applied to the consensus metric.

### Interp 2 (frame as physical / cognitive state, contributing to ontology)

**Line 1525** (the bundle-pullback metric construction):

> "Equation~\eqref{eq:bundle_metric} is a fiber-respecting metric on $E_q$: ... the **horizontal block is agent-frame-dependent in the same sense as the section itself** ..."

**Lines 1545-1554** (the explicit disclosure paragraph):

> "Under a local gauge transformation $U_i \to U_i g(c)$, the connection $A^{(i)} = U_i^{-1} dU_i$ acquires a Maurer-Cartan piece $A^{(i)} \to g^{-1} A^{(i)} g + g^{-1} dg$, and **$\mathrm{tr}(A_\mu A_\nu)$ is therefore not invariant** ... **The horizontal block is consequently agent-frame-dependent** whereby different gauge fixings of the same agent yield different horizontal contributions **which is consistent with the framework's commitment that distinct gauge frames generate distinct pulled-back geometries.**"

This is Interp 2, openly. The framework "commits" to frames generating distinct geometries.

**Lines 2732, 2734** (`\subsubsection{Qualia as Gauge-Frame-Dependent Phenomenology}`):

> "Different gauge frames $\phi_i$ induce different metrics $G_i = \sigma_i^* g_{\mathcal{B}}$ through pullback from the same noumenal substrate $\mathcal{C}$. **The phenomenal character of experience then corresponds to the specific geometric structure of this induced metric.** ... Two agents ... operating with different gauge frames perceive different phenomenal geometries despite sampling the same underlying information."

Interp 2 in the strongest possible form: the frame is a phenomenal-state variable.

**Line 2747**:

> "**There is, in this reading, something it is like to occupy gauge frame $\phi_i$** because that frame induces a specific metric $G_i$ defining the agent's phenomenal space."

Interp 2: frames are loci of subjective experience.

### Direct contradictions

The pair (line 235, "arbitrary choices with no physical content") vs. (line 2747, "something it is like to occupy gauge frame $\phi_i$") is incompatible as written. Either frames have no physical content, or there is something it is like to occupy them, but not both. The qualia indeterminacy paragraph at line 2734 acknowledges this tension and gestures at a resolution via "multi-scale constituent structure," but it does not retract line 235.

## Computation: `tr(A_μ A_ν)` is not gauge-invariant

Under the manuscript's stated transformation $U_i \to U_i g(c)$ with $g$ depending on $c$:

The connection $A^{(i)}_\mu = U_i^{-1} \partial_\mu U_i$ transforms as
$$A_\mu \to (U_i g)^{-1} \partial_\mu (U_i g) = g^{-1} U_i^{-1} (\partial_\mu U_i) g + g^{-1} U_i^{-1} U_i \partial_\mu g = g^{-1} A_\mu g + g^{-1} \partial_\mu g.$$

This is the standard gauge transformation law (covariant adjoint piece + inhomogeneous Maurer-Cartan piece), and matches the formula the manuscript itself gives at line 437.

Therefore
$$\mathrm{tr}(A_\mu A_\nu) \to \mathrm{tr}\!\left[(g^{-1} A_\mu g + g^{-1} \partial_\mu g)(g^{-1} A_\nu g + g^{-1} \partial_\nu g)\right].$$

Expanding the product and using cyclic invariance of trace:

- $\mathrm{tr}(g^{-1} A_\mu g g^{-1} A_\nu g) = \mathrm{tr}(A_\mu A_\nu)$ ✓ (the only term that survives if $g$ is constant)
- $\mathrm{tr}(g^{-1} A_\mu g g^{-1} \partial_\nu g) = \mathrm{tr}(A_\mu \partial_\nu g \cdot g^{-1})$ — does not vanish unless $\partial_\nu g = 0$
- $\mathrm{tr}(g^{-1} \partial_\mu g g^{-1} A_\nu g) = \mathrm{tr}(\partial_\mu g \cdot g^{-1} A_\nu)$
- $\mathrm{tr}(g^{-1} \partial_\mu g g^{-1} \partial_\nu g)$ — pure Maurer-Cartan term

So
$$\mathrm{tr}(A_\mu A_\nu) \to \mathrm{tr}(A_\mu A_\nu) + \mathrm{tr}(A_\mu \partial_\nu g \cdot g^{-1}) + \mathrm{tr}(\partial_\mu g \cdot g^{-1} A_\nu) + \mathrm{tr}(g^{-1} \partial_\mu g \cdot g^{-1} \partial_\nu g),$$

invariant only when $\partial_\mu g = 0$ (constant $g$, i.e., a global / rigid gauge transformation, not a local one). The manuscript reproduces this formula correctly at line 1546. Confirmed: the horizontal pullback metric used in Eq.~(induced_metric_full) is gauge-covariant at best, not gauge-invariant, so under the textbook reading it is not an observable.

For contrast, the genuinely gauge-invariant quantities are: (i) integrated curvature densities such as $\int \mathrm{tr}(F_{\mu\nu} F^{\mu\nu}) \sqrt{|h|} d^d c$ for some base metric $h$ (Yang-Mills action); (ii) Wilson loops $\mathrm{tr}\,\mathcal{P}\exp(-\oint A)$; (iii) the consensus / gauge-averaged metric the manuscript constructs at line 1693.

## Census

| Pattern | Hits | Locations | Reading |
|---|---|---|---|
| `pure gauge` | 1 | 765 (flat-bundle, $F=0$) | technical, not the redundancy claim |
| `arbitrary` near `frame`/`gauge` | 6 | 235, 426, 503, 1689, 2663, 3190 | all Interp 1 |
| `no physical content` | 1 | 235 | Interp 1 |
| `physically meaningful` | 4 | 426, 428, 432 (paraphrase), 432 | Interp 1 (frames not meaningful, pairings are) |
| `phenomenal` near `frame` | 7 | 2732, 2734, 2736, 2747, 2752, 1811, 1817 | all Interp 2 |
| `gauge-invariant` / `gauge invariance` | ~30 | passim | Interp 1 framing of objectivity |
| disclosure / acknowledgment of frame-dependence | 2 main blocks | 1545-1554, 1689 | Interp 2 (admitted) |

## Where the manuscript already resolves the tension (and where it does not)

**Resolved.** Lines 1545-1554 and 1689-1715 jointly state the textbook position cleanly: the per-agent pullback `G_i = (σ_i)^* g_{E_q}` is gauge-covariant, not gauge-invariant; objective geometry on $\mathcal{C}$ is recovered only after gauge averaging or via curvature invariants; and "objective reality" is identified with the consensus metric, which is explicitly gauge-invariant by Haar averaging. This is internally consistent.

**Not resolved.** The qualia section (2732-2752) and the ontology paragraph at 1553-1554 ("agent-frame-dependence of perceived geometry") reuse the per-agent pullback `G_i` as a load-bearing ontological object. Under that use, $\phi_i$ is not pure gauge, because two gauge-equivalent $(\phi_i, \phi_i')$ would have to give the same phenomenal geometry, and the manuscript explicitly says they do not (line 2734: "qualitative inversion corresponds to gauge transformations that preserve informational structure while altering phenomenal character"). That sentence is logically incompatible with line 235 ("no physical content"). The qualia-indeterminacy paragraph attempts to dissolve this via "multi-scale constituent structure," but multi-scale composition does not change the single-scale gauge group action; the contradiction at the single-scale level remains.

**Two senses of "gauge" are being used.** Sense A (mathematical, sections 3-5): smooth $G$-bundle, transport $\Omega_{ij}$, Yang-Mills connection — fully covariant, $\phi_i$ is arbitrary. Sense B (philosophical, sections 6 and 9 on phenomenology and qualia): $\phi_i$ labels an agent's perspective and induces its phenomenal geometry — $\phi_i$ is a state variable, and gauge-equivalent frames give distinguishable phenomenologies (otherwise inverted-spectrum thought experiments would be vacuous in this framework). The manuscript switches between Sense A and Sense B without flagging the switch.

## Recommended Clarifying Text

Insert near the end of `\subsection{Principal Bundles and Gauge Freedom}` (after line 244), and adjust line 235 as indicated. The patch should also be cross-referenced from the qualia section.

```latex
\paragraph{Two roles for the gauge frame, and a terminological warning.}
The frame field $\{\phi_i\}$ plays two formally distinct roles in this work, and we
flag the distinction explicitly because the word ``gauge'' is used in both senses
in the literature.

\emph{Role A (mathematical transport).} The collection of transport operators
$\Omega_{ij} = U_i U_j^{-1}$ is gauge-covariant in the textbook Yang-Mills sense:
under a global right-translation $U_i \mapsto U_i g$ the operators $\Omega_{ij}$
are unchanged, and under a local transformation $U_i \mapsto U_i g(c)$ the
connection $A^{(i)} = U_i^{-1} dU_i$ transforms by the standard
$A^{(i)} \to g^{-1} A^{(i)} g + g^{-1} dg$. In this role, the individual frame
$\phi_i$ carries no observable content; only conjugation-invariant combinations
of $A^{(i)}$ (curvature scalars, Wilson loops) and gauge-orbit averages
(Section~\ref{sec:collective_geometry}) are gauge-invariant. This is the role
exercised by the dynamics, the cocycle and holonomy theorems, and the consensus-
metric construction.

\emph{Role B (frame as cognitive state).} The phenomenological reading developed
in Sections~\ref{sec:perception_consensus} and~\ref{sec:hard_problem} treats
the per-agent pullback $G_i = \sigma_i^* g_{\mathcal{B}}$ as the geometric
character of agent $i$'s experience. Because the horizontal block
$\mathrm{tr}(A^{(i)}_\mu A^{(i)}_\nu)$ is not gauge-invariant
(Section~\ref{sec:bundle_metric}), $G_i$ is not gauge-invariant either: distinct
gauge frames yield distinct pulled-back geometries. In this role $\phi_i$ is a
\emph{cognitive state variable}, not a redundant label, and the gauge group action
relates physically (or phenomenally) distinct agents rather than equivalent
descriptions of one agent.

These two roles are mathematically compatible but ontologically distinct, and the
older language calling individual frames ``arbitrary choices with no physical
content'' applies only to Role A. Where the framework asserts phenomenal
significance for $G_i$ (qualia, inverted-spectrum, the hard problem), $\phi_i$ is
not pure gauge in the redundancy sense; it is a state-like degree of freedom
whose orbit happens to coincide with the mathematical gauge orbit. We accordingly
use ``gauge-covariant'' for the transport structure, ``gauge-invariant'' for
quantities surviving Haar averaging or curvature integration, and reserve
``frame-dependent'' (rather than ``gauge-dependent'') for objects like $G_i$
whose dependence on $\phi_i$ is part of their intended ontological content.
```

In the line-235 paragraph, change

> "These frames are arbitrary choices with no physical content, yet they're necessary for representing probabilistic information."

to

> "Considered purely as labels for the transport structure $\{\Omega_{ij}\}$, these frames are gauge-redundant: only relative pairings $U_i U_j^{-1}$ enter the transport-level dynamics. Considered as the section data $\sigma_i^{(q)}$ from which an agent's induced geometry $G_i$ is pulled back, the frame is a state variable carrying frame-dependent (though not transport-observable) content. We disambiguate the two roles in the paragraph below and again in Sections~\ref{sec:bundle_metric} and~\ref{sec:qualia}."

In the qualia section (around line 2747), insert one sentence:

> "We emphasize that this assignment of phenomenal content to a particular frame uses $\phi_i$ in its Role-B sense (cognitive state variable), not in the Role-A redundancy sense; gauge-equivalent frames need not produce identical phenomenal geometries, and the inverted-spectrum reading at line~\ref{eq:inverted_spectrum} depends on this distinction."

These three edits remove the apparent contradiction without altering any equation, theorem, or numerical result. They also make the manuscript's actual position (which is `gauge-covariant transport` + `frame-as-state ontology`) defensible against a referee who reads only the mathematical-physics side.
