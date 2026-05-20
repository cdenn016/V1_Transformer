# Red Opening — pifb-open-problems

## Steelman (opposing position)

The Open Problems section is a model of honest open-problem registration: it opens with an explicit gating condition (3463), uses a uniform Problem / Existence-toy / Remaining Work / Status template for each subsection, registers each problem as open with no overclaim, and the Status lines have been audited against the manuscript's actual delivery through twelve prior Discussion-section debates, so cross-references and Status framings are already aligned with the rest of the document.

## Position

The section is mostly well-calibrated, but it is **not** internally consistent with the rest of the same manuscript on three specific points; the section was written before three later edits to adjacent sections landed, and three Status registrations now contradict the manuscript's own framing elsewhere. Specifically: (R1) line 3494 makes a hardware claim that contradicts the project's current state; (R2) line 3502 makes a "no rigorous quantum extension currently exists" claim that the same manuscript at line 3172 has already qualified by naming Caticha's Entropic Dynamics program as the closest existing inference-route derivation; (R3) line 3506 calls the Gauge Curvature Conjecture "a falsifiable holonomy prediction" without the Regime II conditional that the conjecture itself at line 3223 now explicitly requires.

The load-bearing assumption that fails is sub-claim 11's "the Status registrations are consistent with the manuscript's actual delivery as established in the prior twelve Discussion-section debates." That assumption holds for five of seven subsections; it fails for Scaling (3492), Quantum Extension (3502), and Experimental Validation (3506).

## Evidence

**R1 — Hardware claim at line 3494 contradicts current host.**

`Participatory_it_from_bit.tex:3494` reads: "Our largest validated system (8 agents, dimension 13) is tiny. Behavior at scale ($N>1000$, $K>768$) is unknown. Limited computational resources (a single AMD 9900x CPU) have prevented deeper explorations."

The memory entry `memory/project_aif_module.md` (Canonical /aif Module, 2026-05-19) records the project's current host: "The user has an **RTX 5090 (32 GB)** so the recommended demo preset (`horizon_D=2, beam_width=4`) is well within budget when Phase 2 lands." The grammatical form "have prevented" is present-perfect: it asserts a constraint continuing up to the present. Either the present-perfect is wrong (the constraint is historical, not ongoing, so the tense should be simple past "prevented") or the hardware identifier is wrong (the present-tense constraint is real but the device is now an RTX 5090 GPU, not the AMD 9900x CPU). Both readings require an edit. The status registration as written is also inconsistent with the manuscript framing this as a proof-of-concept study constrained by resources that no longer apply.

**R2 — "No rigorous quantum extension exists" at 3502 contradicts §3172 of the same manuscript.**

`Participatory_it_from_bit.tex:3502` (Quantum Extension Status) closes with: "While structural parallels with QBism are suggestive, no rigorous quantum extension currently exists, and the connection-sector $\mathrm{GL}(K, \mathbb{C})$ pathway provides only mathematical tools for pursuing one rather than a quantum theory in itself."

`Participatory_it_from_bit.tex:3172` (Measurement Analogy) reads: "Caticha's Entropic Dynamics program~\cite{Caticha2015, Caticha2019, JohnsonCaticha2011} provides the closest existing inference-route derivation of Schr\"odinger evolution and the Born rule from probability-on-configuration-space premises, though without the gauge-bundle structure of the present framework."

The bib confirms the three entries are present at `Attention/references.bib:143`, `:150`, `:160`. The Open Problems claim "no rigorous quantum extension currently exists" therefore needs the same qualifier already accepted at 3172: an inference-route derivation does exist (Caticha's program), and the gap the Open Problems subsection actually owns is the absence of a quantum extension **inside the connection-sector GL(K,C) pathway** the manuscript pursues. As written, 3502 reads as if no inference-route quantum derivation has ever been attempted, which contradicts the manuscript's own registration four hundred lines earlier. This is a within-document inconsistency, not an external-canon dispute.

**R3 — "Falsifiable holonomy prediction" at 3506 contradicts the conjecture's own hedged status at 3223.**

`Participatory_it_from_bit.tex:3506` reads: "The Gauge Curvature Conjecture (Section~\ref{sec:gauge_curvature_conjecture}) provides one such anchor in the form of a falsifiable holonomy prediction for bidirectional language models; extensions to neuroscience and physical systems require new experimental designs."

`Participatory_it_from_bit.tex:3223` (the conjecture itself, edited in the Language and Cognition debate at commit 2aeecc91) now reads: "This gauge curvature minimization principle represents the most readily testable conditional prediction within the framework's speculative-claims set, **once the Regime~II extension of Section~\ref{sec:discrete_regime_ii} is implemented and a connection field $\delta_{ij}$ is estimable from natural-language corpora**." The same paragraph at 3215 adds: "Under the Regime~I parameterization $A = U^{-1}dU$ used in the present implementation, $F_{\mu\nu}$ is identically zero and 'curvature minimization' is automatic; a non-trivial linguistic curvature signal therefore presupposes the Regime~II promotion."

The Open Problems cross-reference at 3506 calls the prediction "falsifiable" without inheriting the two-part conditional (Regime II implemented + connection field corpus-estimable) that the manuscript's own framing now requires. Falsifiability is exactly what the conditional structure suppresses until Regime II is built: with $F_{\mu\nu} \equiv 0$ under Regime I, there is no holonomy prediction to falsify. The cross-reference at 3506 therefore overstates the conjecture's current epistemic status relative to its source section.

## Falsification conditions

This Red position is wrong if any of the following hold:

1. The user's current host is in fact a single AMD 9900x CPU and the project is not running on an RTX 5090. (R1 collapses.)

2. Line 3502 already cites `Caticha2015`, `Caticha2019`, or `JohnsonCaticha2011`, **or** line 3172 does not in fact name Caticha's program as an existing inference-route derivation. (R2 collapses.)

3. Line 3506 already includes the Regime II conditional from line 3223, **or** the Regime II extension of `sec:discrete_regime_ii` is implemented and a connection field $\delta_{ij}$ is currently estimable from natural-language corpora. (R3 collapses.)

If all three falsification conditions are satisfied, the claim that the section is correctly scoped and calibrated holds across all seven subsections and Red concedes the debate.
