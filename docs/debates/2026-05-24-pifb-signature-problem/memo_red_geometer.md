# Memo — debate-expert-geometer — red — opening — pifb-signature-problem

## Lens
Differential geometry — what counts as a (pseudo-)metric, non-degeneracy, rank-changing operations.

## Steelman of the opposing position
The section is scrupulously honest that the unprojected $G_{\mu\nu}$ is rank-1 complex and that recovering a rank-2 real form is a "rank-changing operation, not merely a discard of an off-diagonal imaginary piece" (`:2887`); having flagged this explicitly, calling the projected object "structurally compatible with Lorentzian signature" is a fair label on an honestly-disclosed construction.

## My position (in service of red)
The genuine geometric object the framework produces is $G_{\mu\nu}=\mathrm{tr}(A_\mu A_\nu)$, and that object is **not a pseudo-metric**: it is complex-valued and degenerate (one zero eigenvalue), as the manuscript admits at `:2887` ($\det=0$). A pseudo-Riemannian metric is by definition a smooth, symmetric, *non-degenerate*, *real* $(0,2)$-tensor (do Carmo; Lee). The complex degenerate $G_{\mu\nu}$ fails non-degeneracy and reality simultaneously. The Lorentzian metric `:2889` is not the framework's object; it is a *different* object obtained by the map $G\mapsto\mathrm{Re}(G)$, which the section itself certifies has "no physical principle in the construction that mandates" it (`:2892`).

So the logical structure is: the framework produces $X$ (a degenerate complex form, not a metric); a metric $Y$ is produced by hand via an unmotivated projection of $X$; the section then claims the framework is "structurally compatible with Lorentzian signature" on the strength of $Y$. But $Y$ is not in the image of the framework's construction; it is in the image of (framework) $\circ$ (hand-applied projection). A metric obtained by projecting a non-metric is a metric one *writes down*, not one the framework *produces*. The rank-changing character makes this sharp: $\mathrm{Re}(\cdot)$ does not commute with the bundle structure (it is not a gauge-covariant or even a linear-over-$\mathbb{C}$ operation on the connection), so the projected object has no claim to be the pullback of anything geometric. It is the real part of a coordinate-expression, full stop.

The single-generator collapse is what makes the projection *necessary*: with $T_\tau=T_x=T$ the complex form is rank-1 (verified: eigenvalues $\{0,\ -2(\partial_\tau\psi-\partial_x\psi)(\partial_\tau\psi+\partial_x\psi)\}$), so a non-degenerate Lorentzian metric simply does not exist without throwing away the imaginary off-diagonal. The section's honesty about the rank change does not rescue the claim; it documents precisely why the framework, taken at its word, does not deliver a metric.

## Evidence
- do Carmo, *Riemannian Geometry* (1992), Ch. 0–1; Lee, *Introduction to Smooth Manifolds* (2nd ed., 2013), Ch. 13 (Riemannian/pseudo-Riemannian metrics): a (pseudo-)metric is a smooth symmetric *non-degenerate* real $(0,2)$-tensor field. A degenerate or complex-valued symmetric form is not a metric.
- sympy (executed): complex $G=\begin{psmallmatrix}-2(\partial_\tau\psi)^2 & 2i\,\partial_\tau\psi\,\partial_x\psi\\ 2i\,\partial_\tau\psi\,\partial_x\psi & 2(\partial_x\psi)^2\end{psmallmatrix}$ has $\det=0$ and eigenvalues $\{0,\,-2(\partial_\tau\psi-\partial_x\psi)(\partial_\tau\psi+\partial_x\psi)\}$ — degenerate. After $\mathrm{Re}(\cdot)$: $\det=-4(\partial_\tau\psi)^2(\partial_x\psi)^2<0$, eigenvalues $\{-2(\partial_\tau\psi)^2,\,+2(\partial_x\psi)^2\}$ — only then Lorentzian. The non-degeneracy is created by the projection.
- Manuscript `:2887`, `:2892`: the section concedes the rank-1$\to$rank-2 character and the absence of a mandating principle.

## Newly-discovered canon (for 01b_extended_evidence.md)
- Lee, J. M. (2013). *Introduction to Smooth Manifolds*, 2nd ed., Springer, Ch. 13 (Riemannian metrics) and the pseudo-Riemannian generalization: a metric tensor is by definition non-degenerate; degeneracy excludes the object from being a metric.
- O'Neill, B. (1983). *Semi-Riemannian Geometry with Applications to Relativity*, Academic Press, Ch. 2–3: definition of a Lorentzian metric as a non-degenerate real symmetric $(0,2)$-tensor of signature $(-,+,\dots,+)$; the canonical reference for the precise object the claim must produce.

## Falsification conditions
My position is wrong if (a) $\mathrm{Re}(\cdot)$ can be shown to be a gauge-covariant geometric operation (a genuine bundle map) rather than a coordinate-frame projection — the section makes no such argument and admits the opposite at `:2892`; or (b) the framework produces the rank-2 real Lorentzian form *without* the hand-applied projection under some non-collapsed generator assignment — but the section adopts the collapse and does not exhibit such a route.

## Confidence
HIGH — the degeneracy of the genuine object is the manuscript's own $\det=0$ at `:2887`, reproduced in sympy; the definition of a metric as non-degenerate is textbook (O'Neill, Lee).
