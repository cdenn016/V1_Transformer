# Memo — debate-expert-variational (red, opening)

## Steelman of the claim
The structural parallels (mass↔Fisher, energy↔free energy, action↔VFE) are consistently hedged
as "formal analogies without quantitative predictions" (:3113), the mass↔Fisher status is
disclosed as definitional with "no operationally independent measurement" (:3119), and the
action parallel correctly distinguishes first-order Riemannian natural-gradient flow from
second-order Hamilton dynamics (:3125). These are honest and, in isolation, correct.

## Falsification of the claim
The honesty of the parallels is genuine, and I concede it (see below). The attack is that the
honest disclaimers on the parallels quietly transfer the entire predictive burden onto the one
place where the manuscript drops the disclaimer: the :3129 α-prediction. The VFE functional in
this framework carries multiple free structural choices — the coupling weights (α, λ_h, β_ij,
γ_ij, τ in the canonical F), the choice of Gaussian sector vs gauge-frame sector, the irrep
decomposition, and the comparison structure between agents. Every one of these is a tunable knob
in the very machinery (:3117–3125) that is supposed to "feed" the α-derivation. The action-
principle parallel itself concedes the dynamics are "first-order Riemannian natural-gradient
flow" (:3125) with no conserved Hamiltonian and no derived Lagrangian ("we have not derived
Lagrangians … or predicted measurable physical quantities," :3127). A first-order descent flow
with freely chosen coupling weights has no canonical, parameter-free scalar from which a unique
dimensionless invariant like α could fall out; the natural-gradient preconditioner M^{-1} only
rescales the flow (Amari 1998) — it does not manufacture a scale-fixed pure number. So the
substrate the α-prediction rests on is, by the manuscript's own characterization at :3113–:3127,
a proof-of-concept analogy with open coupling parameters. Routing a sharp dimensionless
prediction through an admittedly-analogical, free-parameter substrate is what makes the
"symmetric test" non-symmetric: the same freedom that powers the analogies also absorbs any
α-miss.

## External primary-source citation
Amari, S. (1998), "Natural gradient works efficiently in learning," *Neural Computation*
10(2):251–276 [Amari1998]: the natural gradient is the Fisher-preconditioned steepest-descent
direction, M^{-1}∇F; it reparameterizes/rescales the descent on the statistical manifold and is
invariant to smooth reparameterization. It is a first-order flow with no intrinsic conserved
scalar and no mechanism that fixes an absolute dimensionless constant — consistent with the
manuscript's own :3125 statement that this is "analogy of form, not of derivation."

## Falsification condition for THIS memo
This attack fails if the manuscript identifies a parameter-free scalar functional of the VFE
machinery whose extremization yields α with no freedom in the coupling weights, manifold, irrep
choice, or comparison structure. The canonical F at the framework's core has at minimum α, λ_h,
β_ij, γ_ij, τ as free; the :3129–:3133 program names no fixing of these, so the substrate remains
open.

## Newly-discovered canon
- Amari (1998), "Natural gradient works efficiently in learning," Neural Computation 10(2):
  251–276 — natural gradient = M^{-1}∇F, first-order, reparameterization-invariant rescaling,
  no conserved scalar. [Amari1998] (already in external_bibliography.md; relevance to the α
  substrate is the new point.)
