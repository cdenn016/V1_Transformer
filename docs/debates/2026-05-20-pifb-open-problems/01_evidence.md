# Evidence Pack — pifb-open-problems

## Manuscript references

### The section under debate (lines 3460-3512)

- `Participatory_it_from_bit.tex:3460-3463` — section header with `\label{sec:open_problems}` and opening gating condition: "Several fundamental problems must be resolved before this framework can claim to explain physical reality rather than providing suggestive mathematical analogies."

- `:3465-3473` — **§ The Lorentzian Signature Problem** (3 paragraphs)
  - Problem: SO(3) → Fisher-Rao Riemannian positive-definite vs. spacetime requires (-,+,+,+).
  - Existence-toy status: non-compactness alone insufficient (Sylvester); 2D worked example imposes (i) designated temporal direction, (ii) gauge-frame component multiplied by i, (iii) real-part projection. GL(K,C) contains SO(1,3) via SL(2,C) ≅ Spin(1,3).
  - Remaining Work: dynamical emergence of imaginary frame components, (1,3) signature selection, non-linear extension.
  - Status: "candidate mechanism has been identified in a 2D linearized worked example. The signature problem is not yet resolved."

- `:3475-3479` — **§ Within-Species Pullback Agreement** (1 paragraph)
  - Problem: intersubjective objectivity depends on within-species pullback agreement
  - Status: open; framework provides candidate mechanism (slow-channel KL alignment via γ_ij KL(s_i || tildeOmega_ij s_j)) but does not quantify residual disagreement; consensus pullback metric of sec:consensus_metric remains regulator-dependent.

- `:3481-3490` — **§ Dimensional Structure and Physical Constants** with `\label{sec:dimensional_structure}` (4 paragraphs)
  - Philosophical Position: information is dimensionally fundamental; physical units emerge.
  - The Open Problem (Reframed): "at what stage of the consensus hierarchy do dimensionful quantities crystallize"
  - Testable Prediction: dimensionless ratios derivable from pure information geometry; Planck length might emerge as minimum resolvable information distance.
  - Status: "the framework has not yet derived any dimensionless constant from first principles. Successful derivation of even one such constant would constitute major evidence."

- `:3492-3496` — **§ Scaling and Phase Transitions** (2 paragraphs)
  - Problem: largest validated system is 8 agents, dim 13. Behavior at N>1000, K>768 unknown. "Limited computational resources (a single AMD 9900x CPU) have prevented deeper explorations."
  - Open Questions: hierarchical emergence persistence at large N, phase transitions, emergent phenomena, computational efficiency.

- `:3498-3502` — **§ Quantum Extension** (2 paragraphs)
  - Problem: classical probability vs needed quantum amplitudes.
  - Possible Approaches: GL(K,C) sector-split pathway; would require replacing probability fiber with density matrices / Hilbert space / quantum relative entropy; U(K) ⊂ GL(K,C) preserves Hermitian; U(1) introduces phases (electromagnetic). "No rigorous quantum extension currently exists, and the connection-sector GL(K,C) pathway provides only mathematical tools for pursuing one rather than a quantum theory in itself."

- `:3504-3506` — **§ Experimental Validation** (1 paragraph)
  - Problem: needs concrete experimental anchors beyond transformer validation.
  - Gauge Curvature Conjecture (sec:gauge_curvature_conjecture) provides "a falsifiable holonomy prediction for bidirectional language models."

- `:3508-3512` — **§ Computational Optimization** (2 paragraphs)
  - Problem: substantial overhead (matrix exponentials, Gaussian KL evaluation, Fisher-metric inversion).
  - Possible Approaches: learned approximations, structured covariances, approximate natural gradients, hardware acceleration. Natural-gradient convergence acceleration noted.

### Manuscript-internal cross-references that need to remain consistent

- `sec:worked_signature` — invoked at 3469 (Lorentzian Signature 2D worked example)
- `sec:signature_resolution` — invoked at 3469, 3502 (GL(K,C) signature mechanism)
- `sec:consensus_metric` — invoked at 3479 (consensus pullback metric, regulator-dependent)
- `sec:phenomenological_interpretation` — invoked at 3484 (dimensional emergence)
- `sec:gauge_curvature_conjecture` — invoked at 3506 (now-edited per Language and Cognition debate; the conjecture's "strongest testable prediction" framing was softened to "most readily testable conditional prediction... once the Regime II extension is implemented")

### Prior debate-driven content that touches this section

- `docs/debates/2026-05-20-pifb-discussion-language-cognition/04_verdict.md` (commit 2aeecc91) edited the Gauge Curvature Conjecture at line 3223 to add the Regime II implementation conditional. The Experimental Validation cross-reference at 3506 inherits this conditional.
- `docs/debates/2026-05-20-pifb-discussion-measurement/04_verdict.md` (commit 45a79e89) added Caticha 2015/2019 / Johnson-Caticha 2011 to the bib at the Measurement Analogy subsection. The Quantum Extension subsection at 3502 could cite these as the closest existing inference-route literature.

### CLAUDE.md / memory check on the AMD 9900x CPU claim

- `memory/project_aif_module.md` (memory entry "Canonical /aif Module") notes: "RTX 5090 host (32 GB)" — the host hardware referenced in the Scaling subsection at 3494 as "a single AMD 9900x CPU" may be outdated. Verify against current setup.

## Canon excerpts (teams should expand)

### Sylvester's law of inertia canon

- **Sylvester, J. J. (1852)**, "A demonstration of the theorem that every homogeneous quadratic polynomial is reducible by real orthogonal substitutions to the form of a sum of positive and negative squares," *Philosophical Magazine* 4(23). The signature-invariance theorem invoked at 3469.

### Lorentz / SL(2,C) canon

- **Penrose, R., Rindler, W. (1984)**, *Spinors and Space-Time*, Vol. 1, Cambridge. SL(2,C) ↔ Spin(1,3) double cover.
- **Weinberg, S. (1995)**, *The Quantum Theory of Fields*, Vol. 1, §2.5-2.7.

### Quantum measurement / measurement canon

- **Caticha, A. (2019)**, "The Entropic Dynamics Approach to Quantum Mechanics," *Entropy* 21(10), 943. Already added to bib at Measurement Analogy debate.
- **Johnson, D. T., Caticha, A. (2011)**, arXiv:1108.2550. Already added to bib.

### Renormalization group / phase transition canon (relevant to Scaling subsection)

- **Cardy, J. (1996)**, *Scaling and Renormalization in Statistical Physics*, Cambridge. Already cited at the Rigorous Theorems appendix.
- **Goldenfeld, N. (1992)**, *Lectures on Phase Transitions and the Renormalization Group*, Addison-Wesley.

### Planck-scale information / it-from-bit canon

- **Wheeler, J. A. (1990)**, "Information, Physics, Quantum" — already cited at Wheeler "law without law" subsection in the Participatory Approaches debate.
- **'t Hooft, G. (1993)**, "Dimensional reduction in quantum gravity," in *Salamfest*. The holographic principle, relevant to "Planck length as minimum resolvable information distance" claim at 3488.

## What this evidence does NOT settle

1. **Outdated hardware claim.** The "single AMD 9900x CPU" reference at line 3494 may not match the current setup. Per `memory/project_aif_module.md`, the host is RTX 5090 (32 GB). Verify against the current state. If the manuscript references CPU-only experiments, that's a historical fact; if it describes ongoing computational limitations, it's outdated.

2. **Caticha citation in Quantum Extension subsection.** The 3502 paragraph says "no rigorous quantum extension currently exists" — but Caticha 2015/2019/Johnson-Caticha 2011 (now in the bib via the Measurement Analogy debate) IS an existing inference-route derivation of Schrödinger evolution and the Born rule. The Quantum Extension subsection could acknowledge Caticha as the closest comparison.

3. **Experimental Validation cross-reference.** The 3506 paragraph says the Gauge Curvature Conjecture provides "a falsifiable holonomy prediction." Following the Language and Cognition debate edits at line 3223, this prediction is now explicitly conditional on the Regime II extension being implemented. The Experimental Validation subsection should inherit this conditional framing.

4. **Within-Species Pullback Agreement and the chain of conditionals.** The 3479 paragraph invokes sec:consensus_metric for the structural target, noting it "remains regulator-dependent and is presented there as a heuristic rather than a finite gauge-invariant observable." This is a chain of conditionals: within-species objectivity → consensus metric → regulator. Whether this chain is honest open-problem registration or just deferral is the calibration question.

5. **Dimensional Structure testable prediction.** The 3488 paragraph says "The Planck length $\ell_P = \sqrt{\hbar G/c^3}$ might emerge as the minimum resolvable information distance in the induced consensus metric, providing a natural scale." This is a vague prediction; comparison with 't Hooft 1993 holographic-principle dimensional analysis would tighten.

6. **Computational Optimization "1000× more compute per step" hypothetical.** The 3512 example "if gauge-attention required 1000× more compute per step but converged in 1000× fewer steps" is illustrative; whether natural-gradient acceleration at this magnitude is realistic should be verified against Amari 2016 or recent natural-gradient empirical literature.

Teams should verify points 1-3 (potential factual updates); points 4-6 are editorial.
