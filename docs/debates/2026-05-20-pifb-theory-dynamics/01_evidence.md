# Evidence Pack — pifb-theory-dynamics

## Manuscript references (1005-2043)

- 1005-1246: §The Variational Free Energy Functional. Includes:
  - Physical motivation (1007), Foundation (1013), Multi-agent extension (1021), Gauge-covariant pairwise couplings (1029), Attention from a Mixture-of-Sources Consensus Energy (1033), Non-Uniform Attention Priors (1114), Gauge Transport on Arbitrary Base Manifolds (1180), Base Priors (1225), The Variational Free Energy (1231)
- 1247-1393: §The Complete Free Energy Functional. State-Dependent Prior Precision (1285), Envelope Theorem and Reduced Free Energy (1334), Interpretation of Each Term (1384)
- 1394-1468: §Recasting External Observations — Mean-Gradient Equivalence and Cross-Entropy Resolution; Environmental Agents (1400); Formal Equivalence (1411); Structural Analogy with Variational Stationarity Principles (1450)
- 1469-1501: §Explicit Symmetry Breaking via Observations; Symmetry Breaking (1472)
- 1502-1570: §Dynamical Structure and Emergent Timescales; Natural Time-Scale Separation (1504), Fast Subsystem: Belief Dynamics (1510), Slow Subsystem: Model Learning (1524), Gauge Frame Evolution (1535), Timescale Hierarchy and Adiabatic Approximation (1547), Physical Analogy: Classical vs. Quantum Timescales (1553)
- 1571-1843: §Transformer Architectures as the Zero-Dimensional Limit. The Zero-Dimensional Limit and Gauge Fixing (1576), Setup: Agents as Gaussian Beliefs in Local Frames (1584), Connection to Standard Transformers (1602), Untied Query-Key Carving from Per-Token Frames (1675), Recovery of Dot-Product Attention as a Gauge-Fixed Limit (1717), Value Aggregation (1812), Complete Attention Formula (1829)
- 1844-2043: §Statistical Precision as Configuration-Space Stiffness: A Mass Analogy. Setup (1851), Extended F (1866), Component F (1875), First Variations (1884), Second Variations: The Mass Matrix (1920), Off-diagonal block caveats sec:mass_block_caveats (1958), Within-Framework Interpretation: Stiffness as Precision (2013), Velocity-Quadratic Metric Form sec:velocity_quadratic (2026)

## Prior debate-driven content

- **Kinetic Term Discussion debate (commit a0d4a53f)** edited lines 3087-3095 to clarify the kinetic-metric postulate and cross-reference sec:mass_block_caveats. The body section sec:mass should already be consistent.
- **Pullback Gravity debate (commit 61bb429f)** added a forward-reference paragraph at the end of sec:mass (after line 2024) to register the structural-tier decoupling.

## What this evidence does NOT settle

1. Whether the KL-based attention derivation at 1005-1246 actually shows softmax β as the strict stationarity condition (with the attention entropy term) or whether the "entropy-suppressed surrogate" flagged in CLAUDE.md introduces an approximation.
2. Whether the envelope-theorem convention at sec:envelope is used consistently throughout.
3. Whether the autograd-vs-reduced-F distinction at 1394+ has subtle issues with the cross-entropy term.
4. Whether the Mass Analogy already incorporates the verdicts from the Kinetic Term and Pullback Gravity debates (a consistency check).
