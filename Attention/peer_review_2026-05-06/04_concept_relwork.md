# Peer Review 04: Conceptual Coherence and Related-Work Positioning

**Manuscript:** `Participatory_it_from_bit.tex` (3,738 lines)
**Reviewer scope:** Section 1 (Introduction), Eliminating External Observations (~lines 920–991), Implications (~lines 2503–2854), Future Directions (~lines 2868–2907). Math derivations and empirical results are out of scope.
**Date:** 2026-05-06

---

## Summary verdict

The manuscript has been heavily hedged in a prior pass and now self-flags large portions of the philosophical chapters as "interpretive extension," "motivational reading," "may be unfalsifiable," and "open research question." That is honest, and a clear improvement over what an unhedged version would be. It also raises a sharper question that the manuscript does not face: if entire subsections are admittedly unfalsifiable or admittedly not derivations, do they belong in the same paper as the empirically validated transformer-emergence result, or do they belong in a philosophy companion? My recommendation is **major revisions**. Specifically: cut or move the most clearly admitted-unfalsifiable material (Sections 5.6 "Gauge Invariance as Cognitive Consensus" and 5.10.2 "Kuhnian Revolutions"), tighten the Wheeler/Kant load-bearing claim audit, fix three load-bearing related-work omissions (Cohen/Welling-style gauge-equivariant networks, Hohwy on predictive processing, Amari's foundational information-geometry papers in *related work*), and either engage IIT's relational/non-relational debate or drop the IIT comparison rather than dispatching it in one sentence. The manuscript's central validated claim — transformer attention as the zero-dimensional gauge limit — does not depend on any of the philosophy sections under my review, and the philosophy sections do not derive load-bearing structure for that claim. They earn their place to the extent they connect rigorously; where they connect by analogy, they should say so or be cut.

---

## Major coherence issues

### MC1. Wheeler "It from Bit" mapping is analogical, not derived, and the manuscript's own scope statement says so — but the surrounding prose does not consistently match.

Lines 942–944 ("Connection to Wheeler's 'It From Bit'") read:

> "Wheeler's slogan that physical 'it' derives from informational 'bit' motivates the present functional. The construction supplies a concrete mathematical realization in which geometric structure on $\mathcal{C}$ is induced by pullback of the agent belief dynamics rather than posited a priori."

This is correctly hedged. But line 95 (Cognitive-First Framework) states without hedge:

> "The framework realizes Wheeler's 'it from bit' vision mathematically."

And line 123 (Epistemic Status, Level 3): "We explore whether this structure might describe physical reality" but Level 1 already promised "validated" status to "transformer attention emerges" without distinguishing that the *Wheeler interpretation* of that derivation is not itself validated. The mapping from "transformer attention emerges from gauge VFE" (a derivation) to "this realizes 'it from bit'" (an interpretive claim about Wheeler's program) is not derived; it is a thematic association. The strongest version of the claim that survives the manuscript's own hedges is: *the formalism induces metric structure by pullback in a way that is consonant with one reading of Wheeler.* That should be the headline. "Realizes Wheeler's vision" overstates.

### MC2. Kant is decorative rather than load-bearing.

Section 1.2 (lines 64–72) and the recurring "noumenal" terminology (e.g. line 207: "$\mathcal{C}$ is a smooth $n$-dimensional manifold representing the noumenal substrate") borrow Kantian vocabulary heavily. Two diagnostics:

1. **The math does not change if Kant is removed.** $\mathcal{C}$ is just a smooth manifold over which agents maintain probability distributions; calling it "noumenal" is a label, not a constraint. The Fisher pullback construction would proceed identically if $\mathcal{C}$ were called "the latent base" with no Kant reference at all.
2. **Kant's noumenon is by stipulation structureless, but the manuscript later treats $\mathcal{C}$ as a smooth manifold with coordinates.** Line 215: "We use local coordinates $c = (c^1, \ldots, c^n)$ on $\mathcal{C}$." A coordinatized smooth manifold has structure (dimension, smooth atlas, neighborhood relations). Kant's noumenon does not. Either the Kant identification is a metaphor (in which case say so and stop borrowing the technical term) or the framework is more committal than Kant in a way that should be flagged.

The Scope and Limitations section (line 137) gestures at this — "the framework does not commit to the substrate being structurally inaccessible in Kant's sense" — but the body text continues to use "noumenal" as if the identification were tight. Pick one.

### MC3. The Hard Problem section claims relocation, not dissolution — but earlier text uses "dissolves."

Line 2730–2748 (Qualia as Gauge-Frame-Dependent Phenomenology) and 2754–2786 (Hard Problem Reconsidered) are now careful. Line 146 of the implications dump: "Our framework offers a *relocation* of the hard problem rather than a solution." Good. Line 153: "a reader who finds the hard problem compelling will find the relocation interesting but not satisfying." Good.

But the Lahav-Neemeh comparison (line 140 of dump) still says: "Read through Lahav and Neemeh's relational physicalism, this is exactly what dissolves the hard problem." This is doing rhetorical double-duty: claim "relocation" when challenged, claim "dissolves" when comparing to Lahav-Neemeh. The reader cannot tell whether the manuscript thinks the hard problem is dissolved or relocated. Pick one.

Standard objections from Chalmers and Levine (the explanatory-gap argument; conceivability of zombies; the knowledge argument) are not engaged. The manuscript engages Lahav-Neemeh's *agreement* extensively but does not engage *critics* of the relocation move. A relocation answer to the hard problem should at minimum acknowledge that someone who finds the original problem hard will find the relocated problem equally hard — which the manuscript now does in one sentence (line 153). One sentence is not engagement; it is gesture. Either expand or move the section to a companion.

### MC4. Lahav-Neemeh comparison oversells convergence in some places, undersells divergence in others.

The Lahav-Neemeh section (~line 2764, lines 165–204 of the dump) is the most careful philosophy section in the manuscript. The "Three points of mathematical contact" (line 186) and "Pan-agentic extension as substantive divergence" (line 170) are appropriately specific. **However:** the section is titled "Independent Convergence with Lahav and Neemeh's Relativistic Theory of Consciousness" (line 165), and the title overpromises. Divergences I, II, III are substantive — pan-agentic vs. relational physicalism, multi-scale vs. single-scale, second-postulate-and-Lorentz status. These are differences in kind, not in degree. The honest title would be "Comparison and Partial Convergence." The current title invites the reader to read the section as a vindication; the body text correctly does not vindicate.

Specifically, the claim at line 184 — "Lahav and Neemeh's single-scale Alice/Bob isomorphism is recovered as the special case in which both observers and the observed system are taken at the same scale and the cross-scale factor reduces to the identity" — is a derivation only if one already accepts the pan-agentic multi-scale apparatus. It is not a derivation of the Alice/Bob isomorphism *from* Lahav-Neemeh's premises; it is a derivation *given* the present framework's premises. This should be clearer. The manuscript hints at it ("a derived composition rather than a posited isomorphism") but doesn't say plainly: "we derive their isomorphism only after assuming our pan-agentic structure, which they do not assume."

### MC5. "Gauge invariance as cognitive consensus" is admittedly unfalsifiable. It should be cut or moved.

Line 41 of the implications dump (~line 2641 of the .tex): "this section advances a metaphysical interpretation, not a derivation. The thesis below ... may be unfalsifiable." Line 114: "this view may be unfalsifiable. Any observed physics can be interpreted as reflecting cognitive constraints rather than external reality. ... This interpretive flexibility suggests the framework functions as philosophical perspective rather than scientific hypothesis in the Popperian sense."

Apply the project's own rule (CLAUDE.md): "Remove content that doesn't earn its place through rigorous derivation." An admittedly unfalsifiable section in a paper claiming computational and empirical validation does not earn its place. The candidate disposition is: move to a separate philosophy companion, or compress to a single paragraph noting the possibility and its unfalsifiability and pointing to future work. The current 8-subsection treatment is disproportionate to what the section can earn.

The "Evolutionary thought experiment" subsection (~line 2689, lines 90–98 of the dump) is, by its own admission, a thought experiment, not a falsifiable prediction. The "alternative consensus realities" speculation is a just-so story: if humans evolved differently, physics would be different. This is true of nearly any cognitive framework and does not test anything specific to the gauge-VFE construction. Cut.

### MC6. "Kuhnian Revolutions as Collective Gauge Transformations" is decorative.

Lines 216–226 of the dump (~line 2815). The discriminating test: if I delete this subsection, does any later result depend on it? No. It does not feed into the framework, the transformer derivation, the Lahav-Neemeh comparison, or any open problem. It is an analogy applied to a famous philosophy-of-science topic.

The technical claim that "the transport operator $\Omega_{\text{old,new}}$ ... remains invertible by construction" but produces large KL divergences in transported beliefs (line 224) is a fair gauge-theoretic restatement of "translation gets hard," but it does not explain anything Kuhn did not already say informally. The paragraph that follows ("Concepts natural in one paradigm lack direct counterparts in the other") just restates Kuhn in gauge vocabulary.

By the project's own standard ("Apply this ruthlessly. Anywhere the manuscript could be challenged with 'what does this paragraph actually buy you?' and the honest answer is 'not much' — flag it as a candidate for cuts"), this is the cleanest cut candidate in scope.

### MC7. "Macroscopic Objects as Consensus Enforcers" and "Quantum Systems: Pre-Consensus Dynamics" are honest about being interpretive but should still be evaluated for whether they belong.

Line 2503 ("Macroscopic Objects"): "The following is an interpretive extension of the formalism beyond what the simulations of Section~\ref{sec:meta_agent_emergence} demonstrate. Macroscopic objects are not modelled directly as agents in those simulations; their description here as 'agents with sharp prior covariance' is a stipulation extending the multi-scale meta-agent construction (Appendix~\ref{app:examples}) by analogy."

Line 2540 ("Quantum Systems"): "the actual derivation of wavefunction collapse from classical Bayesian agents requires extending the framework to include quantum superposition states, which we have not done. Treat the following as a motivational reading awaiting a quantum extension rather than a derivation of measurement."

Both hedges are correct and should stay. But the question is: is "rocks-as-very-precise-agents" a derived or stipulated identification? It is stipulated. The Planck-scale identification ($\epsilon \sim \ell_P^2$) at line ~2530 is dimensionally suggestive but not dimensionally derived — the manuscript itself (line 2895, dimensional structure section) acknowledges that the bridge between information-geometric quantities and SI units is unresolved. So $\epsilon \sim \ell_P^2$ is a literary scale-hint, not a calculation. Either acknowledge that explicitly in the macroscopic-objects subsection or remove the Planck-scale claim.

The wavefunction-collapse-as-precision-dominance reading is suggestive and harmless if the hedge stays. But the quantum extension stub is not honest *enough* in one specific way: it claims (~line 2540) that classical Bayesian agents reproduce quantum measurement when high-precision apparatuses couple to low-precision quantum systems. This skips the actual hard part of measurement: superposition and Born-rule probabilities. A high-precision classical apparatus coupling to a low-precision classical system gives convergence; it does not give interference, entanglement, or non-classical correlations. The hedge says this. Make the hedge sharper.

### MC8. The "Why Was the Kinetic Term Missed?" framing (~line 2522) is a strong sociological claim with no citation of work that did consider it.

Line ~2524: "Standard treatments of active inference and variational free energy minimization employ first-order gradient descent, implicitly taking the overdamped limit in which inertial terms are negligible."

That is presented as if no one in active inference has worked on second-order or geodesic dynamics. Friston and collaborators have published on information geometry, geodesic active inference, and natural-gradient flow with curvature corrections. The manuscript does not cite this literature here. Either cite the work that *did* consider second-order structure (and explain how the present framework differs) or qualify the claim — "earlier emphasis was on the first-order regime, though several authors have considered higher-order corrections; we systematize this within gauge VFE." The current framing implies a generation of researchers missed something obvious, which is sociologically strong and should be either supported or softened.

---

## Interpretive-vs-derived claim audit

| Claim | Status | Location |
|---|---|---|
| "Transformer attention emerges as the zero-dimensional limit" | **Derived** (math) + **Empirically validated** (small + scaling study) | §1.6 Level 1, §4 |
| "The framework realizes Wheeler's 'it from bit' vision mathematically" | **Interpretive.** Pullback metric is constructed; "realizes Wheeler" is a thematic identification, not a theorem. | line 95; §3 (Eliminating External Observations) |
| "Spacetime emerges from pullbacks of agent beliefs" | **Math result limited to SO(3) Riemannian case; Lorentzian is a 2D linearized worked example with imposed imaginary generator.** | §1.6 Level 3, §6 (signature problem) |
| "Causality emerges from gauge-covariant transport" | **Interpretive.** Transport gives ordering, but "causality" in the physics sense (light cones, signal propagation) is not derived. | line 91 |
| "Physical laws emerge as information-theoretic necessities (e.g., second law from non-negative KL)" | **The ≥0 of KL is a theorem; identifying it with "the second law" is a thematic identification. Entropy in physics is more than non-negative divergence.** | line 92 |
| "The equivalence principle would, if it could be derived, be consistent with the framework" | **Conditional, honestly flagged.** The hedging is correct. | ~line 2515–2520 |
| "Macroscopic objects = agents with $\Sigma \approx \epsilon I$, $\epsilon \sim \ell_P^2$" | **Stipulated by analogy.** Planck-scale identification is dimensionally suggestive only. | ~line 2530 |
| "Quantum measurement = pre-consensus dynamics" | **Suggestive, not derived.** Classical agents don't reproduce superposition / Born rule. Hedge is present; could be sharper. | ~line 2540 |
| "Wavefunction collapse emerges as transition from pre-consensus to post-consensus" | **Stub claim.** Future work. Hedge is present. | ~line 2570 |
| "Gauge invariance is a cognitive consensus requirement, not a property of nature" | **Admittedly unfalsifiable.** | ~line 2641, line 114 of dump |
| "Qualia = pullback metric structure" | **Identification, not derivation.** Hedged in 2026-05-05 pass. | ~line 2730 |
| "Hard problem dissolved (Lahav-Neemeh reading)" / "relocated (structuralist reading)" | **Two readings; manuscript holds both, in tension.** Pick one. | ~lines 2754, 140 of dump |
| "Alice/Bob isomorphism recovered as derived special case" | **Derived only after assuming pan-agentic multi-scale structure that Lahav-Neemeh do not assume.** | ~line 184 of dump |
| "Kuhnian revolutions = collective gauge transformations" | **Analogy. Adds no new content beyond Kuhn's informal account.** | ~line 2815 |
| "Under-determination explained by multiple gauge-invariant models with equal empirical fit" | **Analogy / restatement.** Quine's point already covers this. | ~line 2828 |
| "Language is a gauge theory; linguistic evolution minimizes gauge curvature" | **Conjecture, flagged as falsifiable.** This is the clearest empirical handle in the philosophy chapters. Keep. | ~line 23 of dump |

---

## Related work gaps

### Critical (must fix)

1. **Cohen, Welling, Geiger and the gauge-equivariant neural network literature.** `Cohen2016` and SE(3)-Transformers (`Fuchs et al.`) are present in `references.bib` (lines 902–913) but **never cited in §1.7 Related Work**. This is the single closest comparison cluster: a paper claiming to derive a gauge-theoretic transformer must position itself relative to gauge-equivariant networks (group-equivariant CNNs, SE(3)-Transformers, Geometric Deep Learning by Bronstein et al.). The manuscript's distinguishing feature is that the gauge structure comes from variational inference rather than from imposed group equivariance, but that contrast cannot be made unless the comparison is in the related-work section. **Add a paragraph.**

2. **Predictive processing / Hohwy.** `hohwy2013predictive` is present in `references.bib` (line 100) but never cited. The Clark-Seth references are present in §1.4 but not in Related Work. Predictive processing is the closest cognitive-neuroscience neighbor of the framework's perception-as-inference reading and is mentioned in passing; it deserves a Related Work paragraph distinguishing the gauge-theoretic extension from standard predictive processing.

3. **Amari and information geometry as a *related-work program*, not just a citation for the Fisher metric.** `Amari2016`, `Amari1985`, `Amari1998`, `Ay2015` are present in `references.bib` (lines 586–610) but only `Amari2016` is cited in body text, and only as a math reference (Fisher metric, Gaussian curvature). Amari's work on information geometry as a foundation for statistical inference and natural-gradient learning is a major related-work cluster. The manuscript should have a Related Work paragraph distinguishing the gauge-theoretic free-energy construction from standard information-geometric variational inference (Amari, Ay-Jost-Lê-Schwachhöfer).

### Important (should fix)

4. **QBism beyond Fuchs2014.** Mermin's "QBism puts the scientist back into science" (2014) and Fuchs' more recent SIC-POVM work would strengthen the QBism comparison. The manuscript cites only one Fuchs paper, which is treated as proxy for the whole program.

5. **IIT (Tononi).** The dispatch — "Tononi's integrated information theory is observer-independent and therefore inhabits a different metaphysical position; we register the contrast rather than enter the IIT debate" (line 156, related work) — is too quick given how much the consciousness section overlaps IIT territory (information integration, hierarchical consciousness, $\Phi$ as integrated information). The Consciousness section (~line 2700) does engage IIT slightly more (`Tononi2004,Tononi2016`), but the related-work dispatch reads as avoidance. Either engage the technical comparison ($\Phi$ vs. meta-agent coherence; intrinsic vs. extrinsic accounts) or remove the IIT comparison entirely.

6. **Friston's higher-order / geodesic active inference.** Tied to MC8 above. The "kinetic term was missed" claim needs a citation map of who in active inference *has* considered second-order or geodesic structure (e.g., Friston-Da Costa-Parr work on path integrals; information length).

### Optional (nice to have)

7. **Wolfram's hypergraph models / ruliad.** Another participatory/computationalist program in the same conceptual neighborhood. Not in the bib.

8. **Carroll's Mad-Dog Everettianism / quantum-Bayesianism alternatives.** Comparison cluster for the observer-dependent reality reading.

9. **Deutsch's constructor theory** is cited (`Deutsch2015`) but only in passing. Constructor theory's possibility/impossibility framing maps onto the gauge-shareability claim and deserves more than one sentence.

---

## Citation accuracy spot-checks

I spot-checked the following load-bearing citations against the bib entries and well-known content of the cited works.

| Citation | Verdict | Note |
|---|---|---|
| Wheeler1990 — "Information, Physics, Quantum: The Search for Links" (1990 in *Complexity, Entropy, and the Physics of Information*, Zurek ed., Addison-Wesley) | **Correct** | Bib entry lines 437–446. This is the standard "it from bit" reference. |
| Wheeler1983 — "Law without law" in *Quantum Theory and Measurement* (Wheeler & Zurek eds., Princeton) | **Correct** | Bib entry lines 298–306. The "no phenomenon is a phenomenon until it is an observed phenomenon" quote (line 56) is associated with this volume. |
| Friston2010 — "The free-energy principle: a unified brain theory?" | **Correct** as the canonical FEP citation. |
| Lahav-Neemeh 2022 — "A Relativistic Theory of Consciousness" *Frontiers in Psychology* 12:704270 | **Correct** (bib line 2676). The paper does argue that consciousness is observer-relative by analogy to special relativity. The manuscript's characterization at lines 168–169 of the dump is fair. |
| LahavNeemeh2025 — arXiv:2502.07247 | **Correct** preprint citation. I have not verified the content of the 2025 preprint matches the manuscript's characterization (it is described as a "shortened version"). |
| Chalmers1995 — "Facing up to the problem of consciousness" | **Correct** standard citation for the hard problem. |
| Kuhn1962 — *The Structure of Scientific Revolutions* | **Correct** book citation. The manuscript's characterization of incommensurability (line 218 of dump) is reasonable but compressed; Kuhn's later work (postscript, 1969) qualified the strong incommensurability claim, and the manuscript does not engage that qualification. |
| Hoffman2019 — *The Case Against Reality* | **Correct** book; the manuscript's characterization (line 137 of intro) as Interface Theory of Perception is fair. |
| Amari2016 — "Information Geometry and Its Applications" | **Correct** book citation. The cited claim that Gaussian manifold has constant negative sectional curvature is a standard Amari result. |
| Cencov1982 (uniqueness of Fisher metric up to scaling) | **Correct** standard reference. |
| Carhart-Harris2014 — psychedelic states reference | **Reasonable** but the 2014 paper is about entropic brain hypothesis, not specifically about gauge-frame analogies; the citation is appropriate but the manuscript's framing of psychedelic states as "non-gauge-invariant perception" is the manuscript's interpretation, not Carhart-Harris's. Mark as interpretive use. |

**Bib hygiene issues** (not a coherence issue but worth flagging):

- `Verlinde2011` appears twice (lines 393 and 691).
- `Jacobson1995` appears twice (lines 404 and 673).
- `Rovelli1996` appears three times (lines 415, 791, 1523, 1569).
- `Hoffman2019` appears twice (lines 469 and 1672).
- `Gao2017` and `Gao2017published` are listed as alternatives with manual notes.

These duplicates may produce BibTeX warnings or silently use the last entry. Run `bibtex` and clean up.

---

## Candidate cuts

Listed in order of strongest case for cutting.

1. **Section 5.6 "Gauge Invariance as Cognitive Consensus" (~line 2641; lines 39–119 of dump).** Admittedly unfalsifiable by the manuscript itself. 8 subsections. Strongest cut. Compress to one paragraph in the philosophy of science section, or move to a separate companion piece. **The 5.6.4 "Evolutionary Thought Experiment" sub-subsection is the cleanest cut within this cut.**

2. **Section 5.10.2 "Kuhnian Revolutions as Collective Gauge Transformations" (~line 2815; lines 216–226 of dump).** Decorative analogy. Restates Kuhn in gauge vocabulary without adding new content. Cut entirely or compress to a single sentence in 5.10.

3. **Section 5.10.3 "Under-determination and Theory Choice" (~line 2828).** Same diagnosis: analogy that does not add to Quine. Cut or compress.

4. **The "evolutionary thought experiment: alternative consensus realities" subsection** (within 5.6, already flagged in cut #1). Just-so story; not a falsifiable prediction.

5. **"Why Was the Kinetic Term Missed?" (~line 2522).** Either back the sociological claim with citations of who *has* considered second-order structure (and how the present account differs), or fold the argument into a one-paragraph remark elsewhere. Currently it reads as an unsupported swipe at the field.

6. **"Implications for Language and Cognition" — gauge curvature conjecture is the strong part; the social/economic/scientific-community curvature speculations (lines 33 of dump) are weak.** "Market economies develop institutions and contracts that reduce transaction curvature" is the kind of "applies to everything" speculation that diminishes rather than supports the falsifiable linguistic prediction. Cut the social/economic extrapolation; keep the linguistic conjecture.

---

## Open questions for authors

1. **Is the framework's relationship to Kant load-bearing or decorative?** Specifically: would the math change if you replaced "noumenal" with "latent base manifold" throughout? If no, why keep Kant?

2. **Hard problem: dissolved or relocated?** The Lahav-Neemeh comparison says "dissolves"; the section's own header says "relocation." Pick one and be consistent.

3. **What does the paper claim about Wheeler's program — that it realizes Wheeler's vision (line 95), or that it provides "a concrete mathematical realization in which geometric structure on $\mathcal{C}$ is induced by pullback" (line 942)?** These are not the same claim. The first is about Wheeler; the second is about pullback construction. Reconcile.

4. **Why is Cohen-Welling-style gauge-equivariant network literature absent from Related Work?** This is the closest neighbor and the strongest contrast point.

5. **For the "kinetic term was missed" claim: who, specifically, missed it? Cite. If you mean Friston's first-order treatment, say so. If you mean a broader claim, support it.**

6. **Does the framework predict anything specifically about the physics of altered states beyond "they break consensus"?** The psychedelic-states discussion (line 82 of dump) gestures at empirical work but does not derive testable predictions. What would the framework predict that, e.g., the entropic-brain hypothesis would not?

7. **For the $\epsilon \sim \ell_P^2$ identification (~line 2530): is this dimensional analysis or rhetoric?** The dimensional-structure section (line 2895) admits the bridge between information-geometric quantities and SI units is unresolved. So how can the macroscopic-object subsection identify rock precision with the Planck length?

8. **For the IIT dispatch in Related Work: what specifically distinguishes the meta-agent / hierarchical consciousness picture here from Tononi's $\Phi$-maximization picture?** The contrast cannot be made unless engaged.

9. **For the Lahav-Neemeh "convergence" framing: would the authors agree that the convergence is structural (you both invoke frame-relativity) but not substantive (your pan-agentic multi-scale ontology is a strict strengthening they do not endorse)?**

10. **Is the gauge-curvature-minimization principle for language testable on existing corpora, or does it require new measurement infrastructure?** This is the single strongest empirical commitment in the philosophy chapters; it should not be left at the level of "concrete testable predictions" without specifying what data and what statistic.

---

## Bottom line

The empirical and mathematical core of the paper (transformer attention as the zero-dimensional gauge limit, the pullback construction, the multi-scale meta-agent simulations) does not require the philosophy chapters under review to be correct. Conversely, the philosophy chapters do not derive structure that the math requires. They are addenda, not load-bearing. The 2026-05-05 hedging pass has correctly converted many overclaims into honest interpretive statements. The next pass should ask, for each interpretive section: does it add enough to justify its length, given that it is admittedly interpretive? For roughly half the philosophy chapters in scope, the honest answer is no.
