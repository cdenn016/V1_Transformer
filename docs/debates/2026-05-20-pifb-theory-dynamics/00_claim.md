# Claim — pifb-theory-dynamics

**Mode:** theory (math)
**Rounds:** condensed (Phase 2 + judge)
**Judge:** on
**Evidence scope:** Attention/Participatory_it_from_bit.tex lines 1005-2043

## Claim

The §Theory dynamics subsections covering the Variational Free Energy Functional (1005-1246), Complete Free Energy Functional (1247-1393), Recasting External Observations (1394-1468), Explicit Symmetry Breaking (1469-1501), Dynamical Structure and Emergent Timescales (1502-1570), Transformer Architectures as Zero-Dimensional Limit (1571-1843), and Statistical Precision as Configuration-Space Stiffness / Mass Analogy (1844-2043) collectively constitute the framework's dynamical core: derived KL-based attention from the variational principle (Section sec:framework, sec:complete_F); explicitly registered the envelope-theorem convention used in stationarity computations (sec:envelope); the Mass Analogy section explicitly registers the kinetic-metric postulate as separate from $\mathcal{F}$ (lines 1847, 1849, 2013-2024, 2026-2034) per the earlier Pullback Gravity / sec:mass debate; the Transformer Recovery section (sec:transformer_zero_dim) provides the boxed Q_i, K_j carving and the recovery of dot-product attention as gauge-fixed limit; the asymmetric-attention caveat at sec:mass_block_caveats (1958-1961) explicitly notes the Newtonian reading is recovered only in symmetric / isolated-agent limits.

## Sub-claims

1. The VFE functional construction at 1005-1246 correctly derives KL-based attention $\beta_{ij} = \mathrm{softmax}(-\mathrm{KL}/\tau)$ as the stationarity condition of the row-Lagrangian including the attention entropy term.
2. The Complete F functional at 1247-1393 with State-Dependent Prior Precision (1285) and Envelope Theorem (1334) is correctly framed.
3. The Mean-Gradient Equivalence and Cross-Entropy Resolution at 1394-1468 honestly registers the autograd-vs-reduced-F convention.
4. The Symmetry Breaking subsection at 1469-1501 correctly distinguishes the gauge-invariant interior from observation-induced breaking.
5. The Timescale Hierarchy at 1502-1570 (fast belief / slow model / static hyper-prior) is consistent with the project's CLAUDE.md.
6. The Transformer Recovery at 1571-1843 with explicit gauge-fixing conditions, untied query-key carving, value aggregation, and complete attention formula is mathematically rigorous; the asymmetric attention block at 1958-1961 is honest about the conservative-Hamiltonian limit.
7. The Mass Analogy at 1844-2043 explicitly registers stiffness/inertia postulate as separate (per the prior Pullback Gravity debate forward-reference fix at line 2026); the velocity-quadratic metric form at sec:velocity_quadratic is "This is a postulate, not a consequence of $\mathcal{F}$".

Red attacks: that the KL-based attention derivation at 1005-1246 may rely on the "entropy-suppressed surrogate" approximation flagged in CLAUDE.md; that the autograd-vs-reduced-F distinction at 1394+ may have subtle issues; that the asymmetric-attention caveat at 1958-1961 may not be cross-referenced enough.

Blue defends: that the derivation produces canonical softmax attention as stationarity condition; that the kinetic-metric postulate is explicitly registered as a postulate; that the asymmetric attention caveat IS cross-referenced from the Discussion subsection (Kinetic Term debate edit at commit a0d4a53f added explicit cross-reference).

Judge: likely BLUE_WINS or RED_WINS_NARROW.
