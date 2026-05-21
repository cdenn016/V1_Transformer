# Claim — cross-layer-phi-handoff

**Mode:** implementation
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (transformer/vfe/stack.py, transformer/vfe/e_step.py, transformer/vfe/positional.py, PIFB:1502-1551, PIFB:1535-1545)
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge\

## Claim

PIFB §1502-1551 ("Dynamical Structure and Emergent Timescales") combined with PIFB §1535-1545 ("Gauge Frame Evolution") prescribes — not merely permits — that in the L-block transformer stack, the prior gauge frame `φ_p^{(ℓ+1)}` at layer ℓ+1 must be the *posterior* gauge frame `φ^{(ℓ)*}` of layer ℓ (analogous to the μ-cascade `μ_p^{(ℓ+1)} = μ^{(ℓ)*}`). `transformer/vfe/stack.py:145` hard-wires `new_prior_phi = initial_priors.phi` (embedding-value handoff) with no toggle to recover the posterior-φ cascade prescribed by the manuscript, and this is a falsifying deviation from the canonical hierarchical-VI structure.

## User context

Follow-up to `docs/debates/2026-05-20-vfe-module-purity-for-pifb/04_verdict.md`. The original verdict ruled PIFB:1537's "may evolve" modal phrasing makes φ-evolution permissive, not prescriptive, and dismissed the cross-layer φ-handoff attack. This debate re-examines that ruling with focus on §1502-1551 as a *whole* (timescale separation), not just the single "may evolve" sentence.

## Falsification conditions

- **Blue (defender) loses** if red shows (i) PIFB §1502-1551 read as a unified construction prescribes posterior-φ cascade as the canonical cross-layer structure (not just permits it), AND (ii) `stack.py:145` has no toggle to enable that cascade.
- **Red (attacker) loses** if blue shows either (a) PIFB §1502-1551 is genuinely permissive about cross-layer φ structure in the multi-layer transformer (no prescribed cascade), OR (b) `stack.py` (or its callers / `e_step.py`) does in fact route posterior φ into the next layer's prior φ under some toggle, even if the variable name `priors.phi` stays at the embedding value.
