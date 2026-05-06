# Defect 6 — Lorentzian-signature framing audit

**Manuscript:** `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\Attention\Participatory_it_from_bit.tex`
**Date:** 2026-05-06
**Scope:** Census of every "signature" / "Lorentzian" passage and classification of each as honest (worked-example-conditioned) or overclaiming (implies $\mathrm{GL}(K,\mathbb{C})$ resolves/solves the signature problem). Recommended LaTeX rewrites for each overclaim.

The previous edit pass added two pieces of context that any rewrite must respect:

- A "Regime~I / Regime~II" distinction at `sec:connection_forms` (around line 438). Regime~I keeps $A = U^{-1}dU$, $F_{\mu\nu} \equiv 0$; Regime~II promotes $A_\mu$ to an independent connection variable. The worked example operates in Regime~I but exploits the gauge-noninvariant quantity $\mathrm{tr}(A_\mu A_\nu)$ with imaginary frame components.
- A "Regime status of the signature mechanism" paragraph at the head of `sec:signature_resolution` (line 1596) that classifies the signature mechanism as a Regime-I, frame-dependent quadratic form. Any honest reframing should refer back to that paragraph rather than reintroduce contradictory framing.

## 1. Census table

Honest (H) means the passage explicitly conditions the result on the postulates (imaginary $\phi_\tau$ by hand, real-part projection, 2D linearized, $T_\tau = T_x = T$ coincidence). Overclaiming (O) means it uses words like "concrete pathway", "resolves", "removes this restriction", or otherwise implies that complexification settles the problem. Mixed (M) means the same sentence does both and needs surgical edits.

| Line | Region | Quoted phrase (truncated) | Class |
|---:|---|---|:--:|
| 47 | Abstract | "...in a linearized 2D worked example with an imposed imaginary temporal generator (acknowledged as a postulate, not a derivation)... structurally compatible with a Lorentz subgroup $\mathrm{SO}(1,1)$, and a real-Lorentzian metric is recovered after a real-part projection; the four-dimensional, fully nonlinear extension and the dynamical selection of the imaginary generator are left open" | H |
| 112 | Level-3 epistemic note | "the natural $\mathrm{GL}(K)$ gauge symmetry and its complexification $\mathrm{GL}(K, \mathbb{C})$ provides a concrete pathway toward Lorentzian signature through the inclusion of non-compact subgroups such as $\mathrm{SO}(1,3)$." | **O** |
| 123 | Level-3 "Worked example only" disclaimer | "The Lorentzian-signature construction... is established in a 2D linearized toy model with a constant generator. The extension to four-dimensional, fully nonlinear $\mathrm{GL}(K,\mathbb{C})$ Yang-Mills dynamics, and the question whether imaginary gauge-frame components emerge dynamically rather than by hand, are open." | H |
| 238 | Two-roles paragraph | "...distinct signature mechanisms in the worked example of Section~\ref{sec:signature_resolution}." | H |
| 256 | `sec:agents_as_sections` | "...the pathway to $\mathrm{GL}(K, \mathbb{C})$ for resolving the Lorentzian signature problem (Section~\ref{sec:signature_resolution})..." | **O** |
| 505 | Gaussian-manifold paragraph | "Extending to mixture models, heavy-tailed distributions, or complex-valued amplitudes (Section~\ref{sec:signature_resolution}) is straightforward in principle but computationally expensive..." | M (the cross-reference is fine; "straightforward in principle" understates the open derivational gap) |
| 556 | `sec:gauge_group_choice` | "...the restriction to compact $\mathrm{SO}(3)$ is the source of the Lorentzian signature problem... The natural $\mathrm{GL}(K)$ symmetry and its complexification $\mathrm{GL}(K, \mathbb{C})$ removes this restriction." | **O** |
| 1466 | Fisher-arc-length paragraph | "The connection to physical time dilation, if there is one, requires the indefinite-metric extension of Section~\ref{sec:signature_resolution} and is not established here." | H |
| 1535 | Bundle-metric paragraph | "...already encountered in Sections~\ref{sec:cognitive_reference_frames} and~\ref{sec:signature_resolution}." | H (cross-ref only) |
| 1593 | Section title | "Temporal Structure and the Signature Problem" | H |
| 1596 | Regime-status paragraph | (see introductory remarks; this paragraph already conditions the construction on Regime I and on $\mathrm{tr}(A_\mu A_\nu)$ being gauge-noninvariant.) | H |
| 1604 | "Why Compact Gauge Groups Force Riemannian Geometry" | "The Lorentzian signature problem is therefore an artifact of the $\mathrm{SO}(3)$ restriction, not a fundamental limitation of the gauge-theoretic framework." | **O** (the restriction is *necessary* but not *sufficient* for the problem to be an "artifact"; calling it an artifact implies removing the restriction removes the problem, which is false) |
| 1608 | "GL(K) and GL(K,C) Resolution" subsubsection title | "The $\mathrm{GL}(K)$ and $\mathrm{GL}(K, \mathbb{C})$ **Resolution**" | **O** (title) |
| 1614 | §1 of resolution subsubsection | "More tellingly, the effective inner product structure on the base manifold... can acquire indefinite character when the transport operators span a non-compact group." | H ("can acquire" hedged) |
| 1618 | "The Concrete Pathway" subsubsection title | "The **Concrete** Pathway" | **O** (title; "concrete" is the explicit overstatement word) |
| 1623 | Pathway step (2) | "The induced metric **acquires** indefinite signature from the non-compact structure of $\mathrm{GL}(K, \mathbb{C})$." | **O** ("acquires" is a stronger claim than the worked example supports; it would only acquire indefinite signature with the additional imaginary-component postulate) |
| 1628 | Closing paragraph of §"The Concrete Pathway" | "Each step in this pathway is mathematically well-defined; the dynamical content... is unresolved... lifting [the compact-group restriction] does not by itself derive Lorentzian signature from information geometry." | H |
| 1630 | Worked example subsubsection title | "Worked Example: Lorentzian Signature from $\mathrm{GL}(2, \mathbb{C})$ Gauge Frames" | H |
| 1633–1671 | Worked example body | Postulates explicitly labeled, derivation gap flagged, real-part projection called out as separate postulate. | H |
| 1681 | "Temporal Direction from Belief Trajectories" | "Under the $\mathrm{GL}(K, \mathbb{C})$ extension, the temporal direction $\dot{q}$ **naturally acquires** the opposite signature from spatial directions $\dot{q}^\perp$, yielding..." | **O** ("naturally acquires" claims a derivation; the next sentence walks it back, but only partially) |
| 1687 | Same paragraph | "Under $\mathrm{GL}(K, \mathbb{C})$, it can in principle be derived from the non-compact group structure, though the full derivation remains future work." | H |
| 1712 | Consensus-metric paragraph | "...that the regulated gauge-averaged metric acquires indefinite signature is a plausibility argument for resolving the Lorentzian signature problem... rather than an established result..." | H |
| 1841 | Phenomenological-interpretation paragraph | "The Lorentzian signature problem has a candidate pathway in a 2D linearized worked example via $\mathrm{GL}(K, \mathbb{C})$..." | H |
| 1883 | Open-research-program list, item 2 | "investigate whether the signature problem **can be resolved** through deeper analysis of gauge structure, holonomy, or emergent phenomena." | H (uses "can be resolved" as an open question, not a claim) |
| 1914 | Pullback-summary paragraph | "We have not derived the Lorentzian signature. The assignment of opposite signs to temporal and spatial metric components remains a phenomenological postulate." | H |
| 1930 | "What we postulate" box | "Under the $\mathrm{GL}(K, \mathbb{C})$ extension... this signature **can in principle be derived** from the non-compact group structure." | M (hedged with "in principle"; still slightly overclaims because the worked example shows that complexification is *necessary but not sufficient* — the imaginary-component postulate and the real-part projection are both needed on top) |
| 2461 | Scaling-results "empirical signature" | unrelated word usage ("signature" in the sense of "fingerprint"). | n/a |
| 2539 | Inertial mass / pullback geometry section | "However, as detailed in Section~\ref{sec:signature_resolution}, this problem is an artifact of the compact gauge group restriction, not of information geometry itself. The natural gauge group $\mathrm{GL}(K)$... contain the Lorentz group $\mathrm{SO}(1,3)$ as a subgroup. Moreover... permits indefinite metric signatures in a linearized 2D worked example, with the imaginary temporal generator imposed by hand..." | M (the second half is honest; the "artifact of the compact gauge group restriction" claim is overclaim that contradicts the honest Status block at 2900) |
| 2814 | Lahav-Neemeh divergence III | "...a mechanism in which the Yang-Mills kinetic term on $\mathrm{GL}(K, \mathbb{C})$ gauge frames yields indefinite signature in a linearized two-dimensional worked example (Section~\ref{sec:signature_resolution}; the postulates of that example are stated honestly, and the four-dimensional, fully nonlinear extension is left open)." | H |
| 2886 | Limitations section title | "The Lorentzian Signature Problem" | H |
| 2890 | Limitations "Resolution Pathway" subitem | "**Resolution Pathway:** As detailed in Section~\ref{sec:signature_resolution}, the restriction to compact gauge groups was the source of this problem, not information geometry itself. The natural gauge group $\mathrm{GL}(K)$ and its complexification $\mathrm{GL}(K, \mathbb{C})$ provide a **concrete pathway**: $\mathrm{GL}(K, \mathbb{C})$ contains the Lorentz group $\mathrm{SO}(1,3)$ as a subgroup via $\mathrm{SL}(2, \mathbb{C}) \cong \mathrm{Spin}(1,3)$. Extension to complex-valued belief parameters and non-compact gauge subgroups yields indefinite pullback metrics **without ad hoc signature assignments**." | **O (worst)** |
| 2900 | Limitations "Status" subitem | "A candidate mechanism has been identified in a 2D linearised worked example. The signature problem is not yet resolved..." | H (this is the honest counterweight; the problem is that 2890 contradicts 2900 four lines apart) |
| 2917 | Quantum-extension subitem | "The $\mathrm{GL}(K, \mathbb{C})$ extension... provides a **natural bridge** to quantum structure... the $\mathrm{GL}(K, \mathbb{C})$ pathway provides **concrete mathematical tools** for pursuing one." | **O** (less acute than 2890 but uses the same overclaim vocabulary "concrete") |
| 2944 | Conclusion para 5 | "A candidate mechanism is exhibited in the worked example of Section~\ref{sec:worked_signature}: imaginary components of $\mathrm{GL}(K, \mathbb{C})$ gauge frames can produce indefinite (Lorentzian) signature on the base manifold... The construction is 2D and linearised, the imaginary component is imposed by hand, and the diagonal-metric outcome depends on a coincidence of generator choice. Whether the mechanism arises dynamically..., generalises to four nonlinear dimensions, and survives independent generator choices, all remain open." | H (this is a model honest paragraph; the rest of the manuscript should converge on this voice) |
| 2950 | Closing paragraph | "Can the $\mathrm{GL}(K, \mathbb{C})$ pathway to Lorentzian signature (Section~\ref{sec:signature_resolution}) be computationally implemented and validated?" | M (treats the pathway as a thing that could be "validated"; better phrasing would be "extended to four nonlinear dimensions and tested for dynamical selection of the imaginary component") |

## 2. Strongest overclaims, ranked

The four passages below are the ones that most need rewriting; each can stand alone as a positive claim about $\mathrm{GL}(K,\mathbb{C})$ "resolving" the signature problem, in contradiction to the honest text at lines 1628, 1671, and 2900.

**OC-1 (line 2890, worst)**

> **Resolution Pathway:** As detailed in Section~\ref{sec:signature_resolution}, the restriction to compact gauge groups was the source of this problem, not information geometry itself. The natural gauge group $\mathrm{GL}(K)$ and its complexification $\mathrm{GL}(K, \mathbb{C})$ provide a concrete pathway: $\mathrm{GL}(K, \mathbb{C})$ contains the Lorentz group $\mathrm{SO}(1,3)$ as a subgroup via $\mathrm{SL}(2, \mathbb{C}) \cong \mathrm{Spin}(1,3)$. Extension to complex-valued belief parameters and non-compact gauge subgroups yields indefinite pullback metrics without ad hoc signature assignments.

The last sentence is the inversion of fact. The worked example at 1633–1671 *requires* an ad hoc imaginary-component postulate plus an ad hoc real-part projection. "Without ad hoc signature assignments" is wrong as written.

**OC-2 (line 2890 paragraph header)**

> **Resolution Pathway**

A subsubsection literally titled "Resolution Pathway" inside a "Critical Open Problems" section is self-contradictory: the section header concedes the problem is open; the subitem header asserts a resolution.

**OC-3 (line 1618, subsubsection title and lead)**

> \subsubsection{The Concrete Pathway} ...
> 2. \textbf{$\mathrm{GL}(K, \mathbb{C})$ with complex exponential family distributions}: Extend gauge frames to $\phi \in \mathfrak{gl}(K, \mathbb{C})$ and distribution parameters to complex values. The KL divergence generalizes naturally to complex exponential families. The induced metric \emph{acquires} indefinite signature from the non-compact structure of $\mathrm{GL}(K, \mathbb{C})$.

"Concrete pathway" + "acquires indefinite signature from the non-compact structure" jointly imply the non-compactness is what does the work. The worked example shows it is not: real $\mathrm{GL}(K, \mathbb{R})$ is also non-compact and does not yield indefinite signature without the imaginary postulate.

**OC-4 (line 556, also lines 112, 256, 1604)**

> the restriction to compact $\mathrm{SO}(3)$ is the source of the Lorentzian signature problem... its complexification $\mathrm{GL}(K, \mathbb{C})$ removes this restriction.

This formulation, repeated four times across the manuscript, is the rhetorical seed of the overclaim. Lifting the compact-group restriction is *necessary* for the worked example (it permits imaginary frame components within the gauge structure) but not *sufficient* (the imaginary assignment and the real-part projection are independent postulates). Calling the compact-group restriction "the source" misnames the deficit.

## 3. Proposed LaTeX rewrites

Each rewrite preserves the existing label structure and is consistent with the Regime~I framing added to `sec:connection_forms` and the regime-status paragraph at line 1596. Where a section title is renamed, the existing `\label` should be retained so cross-references continue to work.

### Fix-1: Replace OC-1 (line 2890)

```latex
\textbf{Existence-toy status.} A linearized two-dimensional worked example
(Section~\ref{sec:worked_signature}) shows that $\mathrm{GL}(K, \mathbb{C})$
gauge frames can host an indefinite Yang-Mills kinetic form once two postulates
are admitted: an imaginary frame component along a chosen base direction, and a
real-part projection of the resulting complex-valued bilinear form. The
construction is therefore an existence demonstration that an indefinite
bilinear form can be manufactured inside the gauge structure, not a derivation
of Lorentzian signature from variational dynamics. Lifting the compact-group
restriction is necessary for this construction but is not by itself sufficient:
real $\mathrm{GL}(K, \mathbb{R})$ is non-compact yet yields a positive-definite
pullback. The signature is selected by the input choice of which frame
component is imaginary, not by the group structure alone. Whether free-energy
minimisation singles out an imaginary $\phi_\tau$ over a real one, whether the
$1{+}3$ split is preferred over $2{+}2$, and whether the construction extends
to four nonlinear dimensions, all remain open.
```

### Fix-2: Replace OC-2 (line 2890 paragraph header)

```latex
\textbf{Existence-toy status:} ...   % was \textbf{Resolution Pathway:}
```

Or, if a longer header is preferred, `\textbf{Status of the candidate $\mathrm{GL}(K,\mathbb{C})$ mechanism:}`.

### Fix-3: Rename and rewrite the §"Concrete Pathway" subsubsection (line 1618)

```latex
\subsubsection{Postulates Required for an Indefinite Pullback}
\label{sec:concrete_pathway}   % preserve the existing label

The compact-group restriction is necessary but not sufficient for indefinite
signature. We list the postulates needed in turn, separating the group-theoretic
prerequisite from the postulates that actually generate the negative
eigenvalue.

\begin{enumerate}
\item \textbf{Group prerequisite: $\mathrm{GL}(K, \mathbb{R})$.} Replacing the
compact $\mathrm{SO}(3)$ used in the simulations with a non-compact gauge group
removes positive-definiteness as a forced consequence of the group action. By
itself this is not enough: real $\mathrm{GL}(K, \mathbb{R})$ acts on
$\Sigma$ by congruence and preserves positive-definiteness by Sylvester's law,
so the Fisher-Rao pullback remains Riemannian.

\item \textbf{Complexification: $\mathrm{GL}(K, \mathbb{C})$ with complex
exponential families.} Allowing $\phi \in \mathfrak{gl}(K, \mathbb{C})$ opens
the door to imaginary frame components. Whether the variational dynamics
select a configuration with imaginary $\phi_\tau$ over the real one is the
open question; the formalism permits both.

\item \textbf{Imaginary-frame postulate.} An imaginary assignment along a
distinguished base direction ($\phi_\tau \to i\phi_\tau$ in
Eq.~\eqref{eq:complex_gauge_frame}) is what flips the sign of the
$\mathrm{tr}(A_\mu A_\nu)$ component. This is a Wick rotation performed inside
the Lie algebra rather than on the base coordinates, and it is imposed by hand
in Section~\ref{sec:worked_signature}.

\item \textbf{Real-part projection.} The Yang-Mills kinetic form
$G_{\mu\nu} = \mathrm{tr}(A_\mu A_\nu)$ is genuinely complex-valued under
postulate~3, and recovering a real Lorentzian metric requires the further step
$G^{\mathrm{Lor}}_{\mu\nu} := \mathrm{Re}(G_{\mu\nu})$ used in
Eq.~\eqref{eq:lorentzian_metric}. This projection is independent of postulate~3
and is also imposed by hand.

\item \textbf{Subgroup restriction: $\mathrm{SO}(1,3) \subset
\mathrm{GL}(4, \mathbb{C})$.} With one imaginary direction the preserved
subgroup is $\mathrm{SO}(1,1)$ in the worked example; the analogous
construction in four dimensions preserves $\mathrm{SO}(1,3)$, but the
$1{+}3$ versus $2{+}2$ split is selected by the input choice of how many
directions are taken imaginary, not by the dynamics.
\end{enumerate}

Each step is mathematically well-defined; the dynamical content (whether
free-energy minimisation actually selects step~3 over a real-valued
$\mathrm{GL}(K)$ configuration, whether step~5 selects $\mathrm{SO}(1,3)$ over
other $\mathrm{SO}(p,q)$ subgroups, and whether the real-part projection
in step~4 has any free-energy justification) is unresolved. The
compact-group restriction precludes Lorentzian signature, but lifting it does
not by itself derive Lorentzian signature from information geometry.
```

### Fix-4: Rename §1608 subsubsection and amend its body

```latex
\subsubsection{The $\mathrm{GL}(K)$ and $\mathrm{GL}(K, \mathbb{C})$ Pathway: What Becomes Possible}
\label{sec:gl_kc_pathway}

The $\mathrm{GL}(K)$ gauge invariance of KL divergence
(Section~\ref{sec:gauge_group_choice}) requires only invertibility, not
orthogonality. This has three consequences for the signature problem, none of
which by themselves derive Lorentzian signature; each merely opens additional
mathematical room that the compact restriction closes off.
```

(Then leave bullets 1–3 substantially as written, but change "The complexification $\mathrm{GL}(K, \mathbb{C})$ contains... can in principle distinguish timelike from spacelike directions through the group structure itself; whether the variational dynamics select such a distinction is not established here." — this last clause is already honest.)

In bullet (3), keep the existing "can exhibit" hedging; that paragraph (line 1614 area) is already honest.

### Fix-5: Replace OC-4 across all repetitions (lines 112, 256, 556, 1604, 2539)

The repeated phrase **"$\mathrm{GL}(K, \mathbb{C})$ removes this restriction"** / **"resolves the Lorentzian signature problem"** / **"this problem is an artifact of the compact gauge group restriction"** should be replaced uniformly with the following:

```latex
the natural gauge group $\mathrm{GL}(K)$ and its complexification
$\mathrm{GL}(K, \mathbb{C})$ remove the group-theoretic obstruction that
forces a compact-group construction to be Riemannian, but do not by themselves
derive an indefinite signature; the worked example of
Section~\ref{sec:worked_signature} requires an additional imaginary-component
postulate and a real-part projection.
```

Concrete per-line patches:

- **Line 112 (Level-3):** delete "provides a concrete pathway toward Lorentzian signature through the inclusion of non-compact subgroups such as $\mathrm{SO}(1,3)$." Replace with: "*permit* the worked example of Section~\ref{sec:worked_signature}, which exhibits indefinite signature once an imaginary frame component and a real-part projection are postulated; this is an existence demonstration rather than a derivation."
- **Line 256 (`sec:agents_as_sections`):** change "the pathway to $\mathrm{GL}(K, \mathbb{C})$ for resolving the Lorentzian signature problem" to "the $\mathrm{GL}(K, \mathbb{C})$ worked example for the signature problem".
- **Line 556 (`sec:gauge_group_choice`):** change the closing two sentences to: "The restriction to compact $\mathrm{SO}(3)$ forces positive-definiteness through the group action and therefore precludes any indefinite signature on the pullback metric; the $\mathrm{GL}(K)$ extension and its complexification $\mathrm{GL}(K, \mathbb{C})$ remove this group-theoretic obstruction without, on their own, deriving an indefinite signature (Section~\ref{sec:signature_resolution})."
- **Line 1604 ("Why Compact Gauge Groups Force Riemannian Geometry"):** change the final sentence to: "Compact-group transport therefore precludes Lorentzian signature by construction; lifting the compact-group restriction is a necessary but not sufficient condition for an indefinite pullback (Section~\ref{sec:worked_signature})."
- **Line 2539 (inertial-mass / pullback-geometry):** change "this problem is an artifact of the compact gauge group restriction, not of information geometry itself" to "the compact-group restriction is what forces the pullback to be Riemannian; lifting it opens but does not close the signature problem". Keep the rest of the paragraph (it is already honest about the imposed-by-hand imaginary generator).

### Fix-6: OC-3 in §"Temporal Direction from Belief Trajectories" (line 1681)

Replace:

> Under the $\mathrm{GL}(K, \mathbb{C})$ extension, the temporal direction $\dot{q}$ naturally acquires the opposite signature from spatial directions $\dot{q}^\perp$, yielding...

with:

> Under the worked example of Section~\ref{sec:worked_signature}, in which an imaginary $\mathrm{GL}(2, \mathbb{C})$ frame component and a real-part projection are postulated, the temporal direction $\dot{q}$ can be assigned the opposite signature from spatial directions $\dot{q}^\perp$, giving the schematic line element

(then keep the equation and the existing follow-up sentence "Under the $\mathrm{SO}(3)$ restriction, this signature assignment had to be postulated. Under $\mathrm{GL}(K, \mathbb{C})$, it can in principle be derived from the non-compact group structure, though the full derivation remains future work." — note that this honest sentence is *already* present at line 1687 and should be retained).

### Fix-7: Section title at 1593

Current title: `\subsection{Temporal Structure and the Signature Problem}` is acceptable. Recommend keeping it; renaming the *internal* "The Concrete Pathway" subsubsection (per Fix-3) is sufficient. The user-suggested rename "Signature: Worked Example and Open Problems" is a fine alternative for the *outer* subsection if a stronger signal is wanted; the trade-off is breaking the existing pattern of "X and the Y Problem" subsection titles elsewhere in the limitations region.

### Fix-8: Abstract (line 47) — confirmation

The abstract sentence is already adequately hedged ("linearized 2D worked example", "imposed imaginary temporal generator", "acknowledged as a postulate, not a derivation", "structurally compatible with a Lorentz subgroup $\mathrm{SO}(1,1)$", "real-Lorentzian metric is recovered after a real-part projection", "left open"). No change recommended.

The Level-3 "Worked example only" disclaimer at line 123 is also adequate as written; the conflict is between line 123 and line 112 four sentences earlier — Fix-5 above resolves the line-112 inconsistency.

### Fix-9: Conclusion paragraph (line 2944) — confirmation

The Conclusion paragraph at line 2944 is the model honest passage in the manuscript. No edits recommended; instead, the rest of the manuscript should converge on this voice.

### Fix-10: Line 2917 (quantum-extension subitem)

Replace "**provides a natural bridge to quantum structure**" with "is a natural place to look for quantum structure, although a rigorous derivation does not yet exist". Replace "the $\mathrm{GL}(K, \mathbb{C})$ pathway provides **concrete mathematical tools** for pursuing one" with "the $\mathrm{GL}(K, \mathbb{C})$ extension provides the algebraic vocabulary in which such a derivation could be formulated."

### Fix-11: Line 2950 (closing paragraph)

Replace "Can the $\mathrm{GL}(K, \mathbb{C})$ pathway to Lorentzian signature (Section~\ref{sec:signature_resolution}) be computationally implemented and validated?" with "Can the $\mathrm{GL}(K, \mathbb{C})$ worked example of Section~\ref{sec:worked_signature} be extended to four nonlinear dimensions, and does free-energy minimisation dynamically select an imaginary frame component over a real one?"

## 4. Summary

The manuscript already contains the correct, honest framing in five places: lines 47, 123, 1628, 1671 (worked-example subsubsection body), 2900, and the Conclusion paragraph at 2944. It also contains four to seven instances of contradictory overclaiming language at lines 112, 256, 556, 1604, 1618, 1681, and 2890. The fix is local (no derivations need to change) and consists of (a) renaming the two overclaiming subsection-level headers ("Resolution Pathway" subitem at 2890 and "The Concrete Pathway" subsubsection at 1618), and (b) replacing the recurring "removes this restriction" / "concrete pathway" / "resolution" wording with language that explicitly distinguishes the *necessary* condition (lifting compactness) from the *sufficient* postulates (imaginary frame component, real-part projection). The Regime~I status paragraph at line 1596 already does the work for the worked-example subsubsection; the rest of the manuscript should be brought into alignment with it.
