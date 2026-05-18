# Peer Review — Participatory_it_from_bit.tex, Speculative Extensions (lines 2521-3128) — 2026-05-18

## Summary

The Speculative Extensions section is the manuscript's most philosophically ambitious and most ontologically vulnerable region. It runs from a definition of "bit-counting time" through a pullback construction of induced geometry, a worked Lorentzian-signature example via complexified frames, a consensus-metric construction with disclosed regulator gap, an eigenvalue-hierarchy reading of observable / subthreshold / internal sectors, an explicit Kantian phenomenal/noumenal mapping, and a "physical quantities as informational labels" stance. The section's strongest feature is its in-place self-disclosure: derivation gaps, postulate sets, regulator obstructions, and the unfalsifiability risk are flagged in the body and again in a "What is derived vs What is postulated" box at lines 3089-3107. The weakest features are (a) a misattributed aphorism that confuses Lloyd2002 with Wheeler at line 2534, (b) total absence of Page-Wootters 1983 and Connes-Rovelli 1994 — the two canonical "emergent time" constructions in the external literature, despite the section running on emergent-time arguments, (c) a citation drift at line 2534 attributing entanglement-entropy spacetime emergence to Jacobson 1995, which derives Einstein equations from horizon area entropy + Clausius and not from entanglement, (d) a structural finding I verified with sympy: the +tr convention with mixed compact and non-compact generators produces Lorentzian signature with a purely real frame, so the manuscript's "imaginary frame component is required" framing is too strong — what is required is some asymmetric input across base directions, of which i² = -1 is one realization, and (e) several (I) phenomenal/noumenal identifications that read declaratively in the body even though they are properly conditional or interpretive. Verdict: major revisions — disclosures are exceptional, but the literature-engagement gap on Page-Wootters / Connes-Rovelli is serious for the central thesis, and a small number of (I) sentences need their conditional qualifiers restored.

## Standards against which the section was reviewed

- [Wheeler1990] "Information, Physics, Quantum: The Search for Links" — primary source for "it from bit"; verified in `references.bib` line 469. The aphorism "time is what prevents everything from happening at once" is canonically attributed to Wheeler (popularised), with Ray Cummings 1922 cited by Quote Investigator as the actual origin; both predate Lloyd 2002 by decades.
- [Page-Wootters 1983] Page & Wootters, "Evolution without evolution: Dynamics described by stationary observables", Phys. Rev. D 27, 2885. The canonical emergent-time construction: time emerges from quantum entanglement between a clock subsystem and an observable subsystem of a globally stationary universe. **Not cited in the manuscript and not in `references.bib`.**
- [Connes-Rovelli 1994] arXiv:gr-qc/9406019, "Von Neumann Algebra Automorphisms and Time-Thermodynamics Relation in General Covariant Quantum Theories". Class. Quantum Grav. 11 (1994) 2899. The thermal-time hypothesis: physical time is the modular automorphism group of an equilibrium state of a Tomita-Takesaki algebra. **Not cited in the manuscript and not in `references.bib`.**
- [Rovelli2004] *Quantum Gravity* (Cambridge) — verified in `references.bib`. The relational-time chapter is the natural touchstone but the manuscript cites Rovelli at sentence-level only.
- [Lloyd2002] "Computational capacity of the universe", PRL 88, 237901, arXiv:quant-ph/0110141 — verified, but does NOT contain the aphorism the manuscript attributes to it at line 2534.
- [Kant1781] Kant, *Critique of Pure Reason* — verified in `references.bib` line 1584. Engagement is structural and the manuscript correctly disavows the strong constitutive-a-priori commitment at line 3048.
- [Nakahara2003], [Frankel2011], [KobayashiNomizu] — for associated-bundle pullbacks, Maurer-Cartan transformation of connection 1-forms, Yang-Mills invariants, and the (0,2)-tensor sandwich rule. The bundle pullback formulation in §2614-2691 is consistent with the standard treatment; the gauge-orbit average construction in §2871-2890 is honestly flagged as conditional on a non-constructed regulator.
- [LahavNeemeh2022, LahavNeemeh2025] — verified in `references.bib`; not load-bearing within the speculative range.
- [Wheeler1983] "Law without law" — verified; used for the "self-excited circuit" claim at line 3125. Wording is consistent with Wheeler's actual usage.
- [Hofstadter1979] *Gödel, Escher, Bach* — verified; cited at line 3125 for "strange loop". GEB introduces the term, but the more direct reference for the standalone phrase "strange loop" is Hofstadter 2007 *I Am a Strange Loop*. Minor.
- [Jacobson1995] "Thermodynamics of Spacetime", PRL 75, 1260, arXiv:gr-qc/9504004 — verified. Derives Einstein equations from horizon-area entropy + Clausius relation, NOT from entanglement entropy as the manuscript's gloss at line 2534 implies.
- Standard signature theory: a Lorentzian metric on $\mathbb{R}^n$ has signature $(-,+,\ldots,+)$ or $(+,-,\ldots,-)$ with one timelike and $n-1$ spacelike directions; SO(1,1) is the 2D Lorentz group of hyperbolic boosts; Sylvester's law of inertia (verified by sympy below in this review).

## Major Issues

### M-Spec1. The "time is what prevents everything from happening at once" aphorism is misattributed to Lloyd2002.

**Claim (manuscript), line 2534:** "Lloyd's computational universe~\cite{Lloyd2002} holds that 'time is what prevents everything from happening at once'."

**Claim kind:** (S) — a citation claim. Presented as if Lloyd2002 contains the quoted aphorism.

**Standard treatment:** The aphorism is canonically attributed to John Archibald Wheeler (who popularised it, with the disclaimer in [Wheeler1990] that he saw it as graffiti at the Pecan Street Cafe in Austin), with Ray Cummings credited by Quote Investigator as the actual origin in 1922. Lloyd2002 is "Computational capacity of the universe" (PRL 88, 237901, arXiv:quant-ph/0110141), which discusses the number of operations the universe can perform — it does not contain this aphorism. Confirmed by inspection of the arXiv abstract.

**Problem:** This is a citation-fabrication-level error: the manuscript ascribes a specific aphorism to a specific paper that does not contain it. The aphorism then reappears at line 3020 ("This realizes Wheeler's notion that time is 'what prevents everything from happening at once'") attributed to Wheeler. The two attributions are inconsistent within the manuscript itself, and only the line 3020 attribution is defensible (Wheeler popularised it; Cummings originated it).

**Required revision:** At line 2534, replace "Lloyd's computational universe~\cite{Lloyd2002} holds that 'time is what prevents everything from happening at once'." with either (a) a different summary of what Lloyd2002 actually claims (the universe is a quantum computer with computational capacity ~10^120 operations), or (b) move the aphorism to a Wheeler attribution and continue Lloyd2002 only as the "universe-as-computation" citation. The line 3020 attribution to Wheeler is the correct one and should be retained.

### M-Spec2. Page-Wootters 1983 and Connes-Rovelli 1994 — the two canonical "emergent time" constructions in the external literature — are entirely absent from "Time as Information Flow" and from `references.bib`.

**Claim (manuscript), §2526-2606:** The whole "Time as Information Flow" subsection advances time as emergent from belief-update dynamics ("Bit-Counting Time", Fisher arc length as internal clock, observer-dependent time, "this connects to several proposals in quantum gravity and foundations of physics"). The cited engagements are Lloyd2002, Rovelli2004, Jacobson1995.

**Claim kind:** (S→I) related-work gap.

**Standard treatment:** The two canonical references for emergent time in quantum gravity and foundations are:
- [Page-Wootters 1983] *Phys. Rev. D* 27, 2885 — derives apparent time evolution from stationary entangled states of a clock subsystem and a system subsystem. The natural standard for an "emergent time from informational correlations" thesis.
- [Connes-Rovelli 1994] arXiv:gr-qc/9406019, *Class. Quantum Grav.* 11, 2899 — derives a physical time flow as the modular automorphism group of an equilibrium thermal state in a Tomita-Takesaki algebra. The natural standard for an "emergent time from state-dependent flow" thesis.

A grep of `references.bib` and the manuscript turns up neither Page-Wootters nor Connes-Rovelli.

**Problem:** A manuscript that posits emergent time as a load-bearing claim and engages Wheeler's "it from bit" must engage Page-Wootters (Wheeler-tradition emergent time from entanglement) and Connes-Rovelli (the Rovelli-tradition emergent time from state). The current engagements (Lloyd2002, Rovelli2004 textbook citation, Jacobson1995) are weaker substitutes: Lloyd2002 is computational capacity, not emergent time; Rovelli2004 is cited at textbook level without the specific Page-Wootters or Connes-Rovelli machinery; Jacobson1995 derives Einstein equations from horizon thermodynamics, not from quantum-informational emergent time. The absence of Page-Wootters is the more glaring of the two because Page-Wootters is the closest precedent for the manuscript's own construction (subsystem of agents whose update parameter is internal to one or many "clocks").

**Required revision:** Add Page-Wootters 1983 and Connes-Rovelli 1994 to `references.bib`. In §2553 ("The formalism thus provides a natural notion of observer-dependent time arising from information geometry, consistent with relational and emergent approaches..."), insert one sentence locating the construction relative to Page-Wootters (entanglement-based emergent time) and Connes-Rovelli (modular-flow emergent time): "This is closest in spirit to [Page-Wootters 1983], where time emerges from correlation structure within a globally stationary universe, and to [Connes-Rovelli 1994], where physical time is the modular automorphism group of an equilibrium state. The present construction differs in that the clock parameter is informational rather than quantum (no Hilbert-space structure on the agent fiber) and in that the multi-agent setting permits a population of internal clocks rather than a single clock subsystem." Treat this as a related-work gap, not as a re-derivation requirement.

### M-Spec3. Jacobson 1995 derives Einstein equations from horizon area entropy + Clausius, not from entanglement entropy; the gloss at line 2534 misrepresents the mechanism.

**Claim (manuscript), line 2534:** "Information theoretic approaches to quantum gravity~\cite{Jacobson1995} derive spacetime structure from entanglement entropy, suggesting deeper connections between information and temporal ordering."

**Claim kind:** (S) — a citation claim about what Jacobson1995 actually does.

**Standard treatment:** [Jacobson1995] (arXiv:gr-qc/9504004) derives the Einstein field equation from $\delta Q = T \, dS$ applied to local Rindler horizons, with $S$ being entropy *proportional to horizon area* (the area-law assumption) and $T$ being the Unruh temperature seen by uniformly accelerated observers. The mechanism is thermodynamic, not entanglement-theoretic. Entanglement-entropy spacetime emergence is a later programme (Jacobson 2016, "Entanglement equilibrium and the Einstein equation"; the Van Raamsdonk programme; ER=EPR). The 1995 paper is sometimes called an "information-theoretic" derivation in the loose sense that area-law entropy is information-theoretic, but the cited mechanism is not entanglement entropy.

**Problem:** Manuscript characterises Jacobson1995 as deriving spacetime structure "from entanglement entropy". This is a citation drift — readers who chase the citation will find horizon thermodynamics with the Clausius relation, not entanglement entropy.

**Required revision:** At line 2534, replace "derive spacetime structure from entanglement entropy" with "derive Einstein's equation from local Rindler horizon thermodynamics ($\delta Q = T \, dS$ with $S$ proportional to horizon area)". If the intent is to invoke entanglement-entropy approaches, cite Van Raamsdonk 2010 (arXiv:1005.3035) or Jacobson 2016 (arXiv:1505.04753); these are distinct from Jacobson1995.

### M-Spec4. The "imaginary frame component is required for Lorentzian signature" framing is too strong: mixed-generator real-frame configurations within the +tr convention also yield Lorentzian signature.

**Claim (manuscript), §2755-2810, esp. line 2810:** "The central open question is whether a free-energy mechanism selects an imaginary $\phi_\tau$ over a real one, and whether the same mechanism singles out a $1{+}3$ rather than a $2{+}2$ split."

**Claim (manuscript), §2848-2853:** "Obtaining [the leading minus sign] requires the same postulates as the worked example of Section~\ref{sec:worked_signature}: an imaginary temporal component of the gauge frame ($\phi_\tau = i\psi_\tau \cdot T$ with non-compact $T$ and $\mathrm{tr}(T^2) > 0$), a real-part projection of the resulting complex frame-twist form, and the choice of $+\mathrm{tr}(AB)$ rather than $-\mathrm{tr}(AB)$ as the bilinear form on $\mathfrak{gl}(K, \mathbb{C})$. Sylvester's law of inertia rules out any sign flip from $\mathrm{GL}(K, \mathbb{R})$ alone, so $\mathrm{GL}(K, \mathbb{C})$ extension is necessary but not sufficient."

**Claim kind:** (N) — worked example; the framing is presented as the only structural route within the +tr convention.

**Sympy verification.** I evaluated three configurations symbolically (script at `/tmp/verify_worked2.py`):

```
Setup 1 (manuscript: non-compact T = diag(1,-1), imaginary temporal frame, +tr):
  tr(T^2) = 2,  G_tt = -2 psi_t^2,  G_xx = +2 psi_x^2     [Lorentzian]

Setup 2 (compact T = [[0,1],[-1,0]] in so(2), REAL frame on both, +tr):
  tr(T^2) = -2, G_tt = -2 psi_t^2,  G_xx = -2 psi_x^2     [definite negative]

Setup 3 (compact T_t = [[0,1],[-1,0]] for tau; non-compact T_x = diag(1,-1) for x; REAL frame, +tr):
  G_tt = -2 psi_t^2,  G_xx = +2 psi_x^2                   [Lorentzian]
```

Setup 3 produces a Lorentzian signature from the +tr convention with a real frame and zero imaginary input, by using a compact generator for the temporal direction and a non-compact generator for the spatial direction.

**Problem:** The manuscript's pathway argument (§2759-2767) lists "complexified gauge frames" as step 2 of the route to Lorentzian signature and writes "Sylvester's law of inertia rules out any sign flip from $\mathrm{GL}(K, \mathbb{R})$ alone, so $\mathrm{GL}(K, \mathbb{C})$ extension is necessary." This is true if one fixes a single generator $T$ on all base directions, as the worked example of §2774 does. It is *not* true under the more flexible setup in which the components $\phi_\mu$ along different base directions are allowed to use different generators of $\mathfrak{g}$ — which is the generic situation, not a contrived alternative. With a compact temporal generator and a non-compact spatial generator under the +tr convention, real-frame components yield $G_{\tau\tau} < 0$ and $G_{xx} > 0$ directly, without any complex extension or real-part projection.

The honest reading is: the imaginary-frame postulate is *one* asymmetric input that lifts the symmetry between base directions. Any sufficient input that distinguishes the temporal generator from the spatial generators — for instance, the choice of compact-vs-non-compact algebra element along the temporal axis — performs the same lift. The claim that GL(K, ℂ) extension is "necessary" is therefore stronger than the underlying construction supports.

**Required revision:** Rewrite the pathway argument at §2759-2767 to acknowledge that the asymmetry which produces an indefinite signature can be introduced either through (i) an imaginary frame component along a designated direction (the manuscript's pathway), or (ii) a choice of compact generator for one direction and non-compact generator for the others (a real alternative within the +tr convention). State explicitly that GL(K, ℂ) is *not* strictly necessary for indefinite signature — what is necessary is some asymmetric input distinguishing the temporal direction from the spatial ones, of which complexification is one route. The dynamical-selection open question (whether free-energy minimisation picks any specific asymmetric configuration) then applies symmetrically to both routes; this strengthens rather than weakens the manuscript's "central open question" framing at line 2810.

### M-Spec5. The real-part projection (line 2793) is a non-standard analytic-continuation step; the manuscript flags it as a "derivation gap" but does not contrast it with standard Wick rotation, which acts on coordinates rather than on the metric.

**Claim (manuscript), lines 2793-2798:** "With an imaginary temporal component, $G_{\mu\nu}$ is complex-valued: the diagonal entries are real (one negative, one positive), while $G_{\tau x}$ is purely imaginary. We therefore extract a real Lorentzian metric by taking the real part, $G^{\mathrm{Lor}}_{\mu\nu} := \mathrm{Re}(G_{\mu\nu})$, which sets $G^{\mathrm{Lor}}_{\tau x} = 0$ ... The real-part projection is an additional choice, separate from Eq.~\eqref{eq:complex_gauge_frame}: $G_{\mu\nu}$ as defined by the trace is genuinely complex-valued, and there is no physical principle in the construction that mandates discarding the imaginary off-diagonal piece. ... We flag this as a derivation gap."

**Claim kind:** (N) — worked example with explicit derivation gap acknowledged.

**Standard treatment:** Standard Wick rotation in physics analytically continues a coordinate (typically $t \to it$, or in the inverse direction $\tau = it$ giving Euclidean time), with the metric components and the action transforming as a consequence. The manuscript's prescription is the inverse: continue the Lie-algebra component ($\phi_\tau \to i\phi_\tau$), then *project the resulting complex metric back to real values by discarding imaginary parts*. There is no standard physics construction in which a complex metric is reduced to a real one by Re-projection.

**Problem:** The manuscript correctly flags the real-part projection as a "derivation gap" at line 2798, which is good practice. What is missing is the comparison with the standard Wick rotation: a reader who expects this construction to be a Wick analog will not realise that the standard route (continue coordinates, not algebra elements) gives a different mathematical operation. The manuscript at line 2772 says "Where the standard physicist's prescription continues $\tau \to i\tau$ on the base manifold, the construction below continues $\phi_\tau \to i\phi_\tau$ in the Lie algebra" — good — but does not add the second-order observation that the resulting metric is then complex-valued and the real-part projection has no Wick analog. The off-diagonal $G_{\tau x} = 2i p_t p_x$ (sympy-verified above) is not Wick-rotated away; it is *discarded by hand*.

**Required revision:** At §2793-2798, after acknowledging the derivation gap, add: "Standard Wick rotation does not encounter this step: continuing $\tau \to i\tau$ on the base manifold produces a real Euclidean metric (or, in inverse rotation, a real Lorentzian one) without complex-valued off-diagonal pieces to discard. The construction here is therefore not a direct Wick analog — it is a Wick-like continuation in the Lie algebra plus an additional real-projection step that has no Wick counterpart. The dynamical mechanism that would justify discarding the off-diagonal imaginary piece is the open problem flagged at the end of this subsection."

### M-Spec6. The Tr(T²) = 2 factor in Eq. (eq:lorentzian_metric) is an unnormalized-generator artifact, not a physical coefficient.

**Claim (manuscript), boxed Eq. (eq:lorentzian_metric) at line 2795:**
$$ds^2 = G^{\mathrm{Lor}}_{\mu\nu} dc^\mu dc^\nu = -2(\partial_\tau \psi_\tau)^2 d\tau^2 + 2(\partial_x \psi_x)^2 dx^2.$$

**Claim kind:** (N) — worked example.

**Verification (sympy):** With $T = \mathrm{diag}(1, -1)$, $\mathrm{tr}(T^2) = 2$; the coefficients ±2 inherit from this normalization. With $T \to T/\sqrt{2}$ (so $\mathrm{tr}(T^2) = 1$), the coefficients become ±1.

**Problem:** Not a derivation error, but the boxed equation suggests that the factor of 2 has structural meaning. It does not — it is a choice of generator normalization. A reader who tries to interpret 2(∂ψ)² physically (kinetic term coefficient, $c$-related scaling) will be misled. The construction is a *conformal class* of Lorentzian metrics; the overall scale is fixed by the generator-normalization convention, and a more standard normalization $\mathrm{tr}(T_a T_b) = \delta_{ab}$ would absorb the 2.

**Required revision:** At the box at §2795, add: "The coefficient ±2 reflects the unnormalized choice $T = \mathrm{diag}(1, -1)$ with $\mathrm{tr}(T^2) = 2$; under the standard normalization $\mathrm{tr}(T_a T_b) = \delta_{ab}$ the coefficients become ±1 and the metric reads $ds^2 = -(\partial_\tau \psi_\tau)^2 d\tau^2 + (\partial_x \psi_x)^2 dx^2$. The overall scale is a generator-normalization choice; what the construction fixes is a Lorentzian conformal class, not a metric with structurally significant coefficients."

### M-Spec7. Several (I) phenomenal/noumenal identifications read declaratively in the body even though they are properly conditional or interpretive.

**Claim (manuscript), line 2701:** "Different agents with different generative models therefore perceive different structural geometries on the same underlying noumenal base $\mathcal{C}$, and there is no agent-independent metric on $\mathcal{C}$, only the family of agent-dependent induced metrics arising from their slow-timescale model parameters."

**Claim (manuscript), line 3003:** "What we call 'physical reality' is always phenomenal reality for some agent or collection of agents."

**Claim (manuscript), line 3026:** "Objects we perceive as massive correspond to belief configurations with large characteristic Fisher information. The difficulty of accelerating massive objects reflects the information-theoretic difficulty of updating highly certain belief distributions."

**Claim (manuscript), line 3046:** "What we call 'mass' is the phenomenological label human measurement protocols assign to configurations of high Fisher information. What we call 'distance' is the label assigned to induced metric intervals on the base manifold. What we call 'time' is the label assigned to accumulated information updates."

**Claim kind:** (I) — interpretive identifications, presented as if (S) or by assertion.

**Standard treatment:** Within philosophy of physics, identifications between physical quantities and information-theoretic structures (e.g., "mass IS Fisher information") are interpretive metaphysical claims that go beyond the underlying mathematics. The mathematics gives a *structural analogy*: Fisher information appears in the natural-gradient update in the same syntactic position as mass appears in Newton's second law. It does not provide a derivation that physical mass equals (or is) Fisher information. The manuscript at lines 3050-3052 explicitly acknowledges this ("This is a question of philosophical interpretation that empirical evidence may inform but cannot definitively settle"), and the table at §3060-3076 honestly tags each correspondence ("formal analogy", "speculative", "dimensional gap", "structural similarity", "mathematical parallel", "undefined", "unmeasured"). The line 3026 sentence "$M_{\mathrm{eff}} \propto \Sigma_p^{-1}$ with $R^2 = 0.9998$" is a within-framework empirical regularity, not a derivation of physical mass.

**Problem:** The four sentences quoted above drop the conditional qualifier and read as factual identifications. Line 2701 ("there is no agent-independent metric") is in tension with the regulator caveat at line 2880 (a regulated consensus metric, if constructible, would be agent-independent); it should not be stated unconditionally. Line 3003 ("'physical reality' is always phenomenal reality") is a metaphysical thesis presented as a fact. Line 3026's second sentence ("Objects we perceive as massive correspond to belief configurations with large characteristic Fisher information") reads as a fact about physics; it is a fact within the framework's definition. Line 3046 ("What we call 'mass' is the phenomenological label...") is a metaphysical reframing, not a mathematical result. Each of these is honestly disclosed elsewhere (§3050-3052, §2880); the problem is local — these specific sentences invert the section's overall discipline.

The boxed disclaimer at §3089-3107 ("What is derived vs what is postulated") is genuinely strong and should be the model for the body prose.

**Required revision:**
- Line 2701: Insert "Within the manuscript's interpretive frame and conditional on the regulator caveat of §2880," at the start of "there is no agent-independent metric on $\mathcal{C}$..."
- Line 3003: Insert "Under this framework's interpretive commitment," at the start of "What we call 'physical reality' is always phenomenal reality..."
- Line 3026: Insert "Within the framework, where information is primal (§\ref{sec:phenomenological_interpretation})," at the start of "Objects we perceive as massive correspond to..."
- Line 3046: Already qualified at line 3048-3050; tighten the line 3046 sentences by adding "Under this framework's interpretive frame," to the first of the three sentences.

These are small edits that preserve the philosophical thesis while restoring the conditional voice that the surrounding paragraphs already use.

## Minor Issues

- §2535-2540 ("Minimum Time from Minimum Information"): The Planck-time argument is properly flagged as "highly speculative and requires substantial development to make rigorous". The unit-counting passage $t_P \sim \hbar/E_P \sim$ (action quantum)/(energy quantum) is a dimensional one-liner that does not depend on the framework's mechanism. Suggested tighter wording: "Whether $\Delta\tau_{\min}$ has any quantitative relation to $t_P$ is an open question of dimensional reconstruction (§ phenomenological_interpretation); the analogy here is structural only and does not constitute a derivation of Planck-scale physics from information geometry."

- §2697: Sentence "The slow-channel generative model $s_i$ does not directly shape $p_i$ at the immediate level; $s_i$ enters $p_i$ only through whatever cross-scale dynamics produced the parent's $q_I^{(s+1)}$." Verifies against the cross-scale shadow Eq. (cross_scale_shadow) — consistent.

- §2725, sector-split paragraph: The numerical-verification gloss "numerical verification at $K=2$ and $K=3$ with real SPD $\Sigma$ and a generic $\Omega \in \mathrm{GL}(K, \mathbb{C})$ produces a sandwich with imaginary part of the same order as its real part and a formal Gaussian KL with both a nonzero imaginary component and a negative real part" is a useful guardrail. Suggest citing the script or numerical setup that produced it (file path + commit) if the reviewer is to verify.

- §2751 ("Complexification and the Lorentz group"): "The passage from compact $\mathrm{SO}(4)$ to non-compact $\mathrm{SO}(1,3)$ is the Wick rotation between different real forms of $\mathrm{SO}(4, \mathbb{C})$." Standard Wick rotation acts on coordinates and *induces* the change in the orthogonal symmetry group; calling the group passage itself "the Wick rotation" inverts the standard direction. Suggested: "$\mathrm{SO}(4)$ and $\mathrm{SO}(1,3)$ are different real forms of $\mathrm{SO}(4, \mathbb{C})$; standard Wick rotation $\tau \to i\tau$ on the base manifold induces the passage from one to the other."

- §2829 ("Conformal-class ambiguity"): "The construction therefore establishes a Lorentzian conformal class $[g]$, not a unique metric." Honest and correct. Worth elevating: the worked example of §2774 also only establishes a conformal class (the ±2 coefficients are a generator-normalization choice, see M-Spec6). Suggest cross-referencing the two conformal-class statements.

- §2832 (Tension paragraph): "Naive continuum limits of such flows are parabolic and yield infinite signal speed, so the causal-cone route does not apply directly to the current implementation." Good. The three proposed routes (telegraph-type CTRW, hyperbolic second-order dynamics, architectural finite-speed constraint) are honestly flagged as not currently realised. Suggest adding a forward-reference to §sec:open_problems if those routes are to be picked up later in the manuscript.

- §2880-2890 (Consensus metric, regulator gap): Disclosure is good. Line 2880's "no finite gauge-orbit average over local $g$ exists without such a choice" is the load-bearing statement. The §3083 summary repeats the caveat ("conditional on a regulator that the present manuscript does not construct"). Existing reviewer's M5 already covers downstream invocation; my finding does not duplicate.

- §2920: "For $K=768$ (for example, a typical transformer embedding dimension), this gives $\dim(\mathcal{B}) = K(K+3)/2 = 296{,}064$." Verifiable: $768 \cdot 771 / 2 = 296{,}064$ ✓.

- §2932: "For human agents, we conjecture this comprises approximately 4 dimensions (1 temporal + 3 spatial)." Honestly tagged as conjecture; the eigenvalue-hierarchy decomposition gives a structural template, not a derivation of the count.

- §2956 ("The full induced metric invites the decomposition $G_i = \sum_{a \in \mathcal{D}_{\text{obs}}}...$"): The decomposition is a definitional partition by eigenvalue threshold; "invites" is the right verb. Good.

- §3014: "This interpretation may be unfalsifiable as any measurement can be reinterpreted as labeling information geometry, making the framework compatible with any result by construction." Strong and honest. Worth keeping; perhaps elevate to the start of §3050.

- §3026: The empirical claim "$M_{\mathrm{eff}} \propto \Sigma_p^{-1}$ with $R^2 = 0.9998$ and the harmonic-oscillator frequency scaling $\omega^2 \propto 1/M$" lives outside this review's scope (§\ref{sec:mass} is upstream). The cross-reference here is consistent with the present reading; no action.

- §3046: The structural-realist reading "in a neo-Kantian structural-realist form (Cassirer, Worrall, Ladyman & Ross)" is well-positioned. The Cassirer / Worrall / Ladyman & Ross citations are not in `references.bib` as visible from the grep; if they are cited only as namedrops without bibkeys, that is acceptable for a philosophical positioning sentence, but a citation would strengthen it.

- §3060-3076 (Correspondences table): Excellent and honestly tagged. The "Mathematical parallel" label for the action principle ($\mathcal{F}$ ↔ Action) is precise. The "$\hbar$: Minimal action per bit, Undefined" entry is the right disposition.

- §3125: "self-excited circuit~\cite{Wheeler1983}" — Wheeler1983 verified. "strange loop~\cite{Hofstadter1979}" — GEB introduces the term; the more directly load-bearing reference for *I Am a Strange Loop* (2007) is not cited but is acceptable to omit.

## Math Reviewer Items

### MR-Spec-1 Sympy verification of the worked example (lines 2787-2806).

The construction at §2774 with $T = \mathrm{diag}(1, -1)$, separable ansatz, and $\phi(\tau, x) = i\psi_\tau(\tau) T + \psi_x(x) T$ produces:
```
A_tau = i (d_tau psi_tau) T
A_x   = (d_x psi_x) T
G_tt = tr(A_tau A_tau) = -2 (d_tau psi_tau)^2
G_xx = tr(A_x A_x)     = +2 (d_x psi_x)^2
G_tx = tr(A_tau A_x)   = +2 i (d_tau psi_tau)(d_x psi_x)
Re(G_tx) = 0
```
And the Lorentz boost identity $\Lambda^T \eta \Lambda - \eta = 0$ for $\Lambda = \begin{pmatrix}\cosh\xi & \sinh\xi \\ \sinh\xi & \cosh\xi\end{pmatrix}$, $\eta = \mathrm{diag}(-1,+1)$, is symbolically zero. (Script: `/tmp/verify_worked.py`.)

The manuscript's calculation is symbolically correct given the postulates. The findings M-Spec4, M-Spec5, and M-Spec6 are about the *framing* of this otherwise-correct calculation.

### MR-Spec-2 Pullback metric formula (Eq. 2666-2670, boxed).

The bundle pullback $G^{(q)}_{i,\mu\nu}(c) = \kappa(A^{(i)}_\mu, A^{(i)}_\nu) + \mathbb{E}_{q_i(c)}[(\nabla^{(i)}_\mu \log q_i)(\nabla^{(i)}_\nu \log q_i)]$ is the natural pullback of the bundle metric $g_{E_q} = g^{\mathrm{tw}}_\mathcal{C} \oplus g_\mathcal{B}$ along a section, decomposed horizontal $\oplus$ vertical. The decomposition is the standard one for a fiber bundle with a connection [Nakahara2003, Frankel2011]. The Maurer-Cartan transformation rule for $A^{(i)} \to g^{-1} A^{(i)} g + g^{-1} dg$ under $U_i \to U_i g(c)$ is correctly stated at §2672. The non-invariance of $\kappa(A_\mu, A_\nu)$ under local $g$ is correctly disclosed. The vertical block as the score-function expectation matches the standard Fisher-Rao form. Consistent with the canon.

### MR-Spec-3 Sylvester / positive-definiteness preservation (§2741-2753).

For real $\Omega \in \mathrm{GL}(K, \mathbb{R})$ and SPD $\Sigma \in \mathbb{S}_+^K$, the congruence $\Omega \Sigma \Omega^\top$ is SPD by Sylvester's law of inertia [HornJohnson §4.5]. The manuscript invokes this correctly at §2743 and §2748 ("by Sylvester's law of inertia"). Consistent with the canon.

### MR-Spec-4 Indefinite-signature obstruction within +tr convention.

The manuscript's claim that GL(K, ℂ) extension is "necessary" for indefinite signature (line 2853) is conditional on using a single generator $T$ across base directions. With mixed generators (compact for one direction, non-compact for another), the +tr convention with a real frame produces indefinite signature directly. See M-Spec4 above.

## Editorial / Style

No banned phrases found within lines 2521-3128 (script-scanned: `key insight`, `crucially`, `critically`, `notably`, `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores`). No banned LaTeX spacing macros (`\;`, `\,`, `\!`) found in range. Equation punctuation is consistent — display equations end in commas or periods through the section.

Self-referential drafting language is absent from this range.

Two readability passes worth considering, neither blocking:

- §2530 ("Dimensional disclaimer ... With that disclaimer in place ..."): The disclaimer paragraph is dense. Splitting it into two short paragraphs (disclaimer first, definition second) would help.
- §3127 ("This is a within-framework observation about the threshold-detector single-seed dynamics of Section~\ref{sec:results}; we make no claim about thermodynamic perpetual motion or about cosmological complexity growth, both of which are outside the scope of the present construction"): Good disclaimer; line is long.

## Citation Verification

- [✓] [Wheeler1990] — `references.bib` line 469, verified.
- [✓] [Wheeler1983] — `references.bib` line 298, verified.
- [✓] [Kant1781] — `references.bib` line 1584, verified.
- [✓] [Rovelli2004] — `references.bib` line 1691, verified.
- [✓] [Lloyd2002] — `references.bib` line 1682, verified as a paper. **BUT** the aphorism attributed to it at line 2534 is NOT in Lloyd2002 (verified via arXiv abstract and external attribution literature: the aphorism is Wheeler-popularised / Cummings-originated, not Lloyd). See M-Spec1.
- [✓] [Jacobson1995] — `references.bib` line 404, verified. **BUT** the gloss at line 2534 ("derive spacetime structure from entanglement entropy") misrepresents the mechanism, which is horizon-area entropy + Clausius. See M-Spec3.
- [✓] [Hofstadter1979] — `references.bib` line 309, verified. Minor: the more direct reference for "strange loop" is Hofstadter 2007, but GEB introduces the term.
- [✓] [Friston2017] — `references.bib` line 571, verified.
- [✓] [Clark2016] — `references.bib` line 597, verified.
- [✓] [Seth2021] — `references.bib` line 480, verified.
- [✓] [Hoffman2019] — `references.bib` line 501, verified.
- [✓] [LahavNeemeh2022], [LahavNeemeh2025] — `references.bib` lines 2737, 2748, verified (not load-bearing within this range).
- [✗] **Page-Wootters 1983** — NOT in `references.bib`. Major related-work gap. See M-Spec2.
- [✗] **Connes-Rovelli 1994** — NOT in `references.bib`. Major related-work gap. See M-Spec2.
- [?] **Cassirer, Worrall, Ladyman & Ross** (line 3048, "neo-Kantian structural-realist form (Cassirer, Worrall, Ladyman & Ross)") — namedropped without bibkeys; could not locate `Cassirer*`, `Worrall*`, `Ladyman*`, or `LadymanRoss*` in `references.bib` via grep. Either add bibkeys or remove the parenthetical.
- [?] **Wheeler's "self-excited circuit"** at line 3125 — Wheeler1983 is the cited reference; the "self-excited circuit" phrase is consistent with Wheeler's diagrams in *Law without law*. Verified at level of citation but the specific Wheeler1983 page-number for the diagram is not in the bib (acceptable).

## Manuscript ↔ Code Consistency

The speculative section makes no specific implementation claims about the codebase beyond a forward reference at line 3026 to the empirical mass-precision study of §\ref{sec:mass}. That cross-reference lives outside this review's scope (the empirical regime is in the body manuscript, not in §sec:speculative_extensions). No further manuscript ↔ code consistency findings within the range 2521-3128.

## Novel-construction inventory (within range 2521-3128)

The following constructions in the speculative range are not standard in the external canon and need clear "novel" labelling. Most are already so labelled.

1. **Bit-counting time** $\Delta\tau_i = \Delta I_i / (1\text{ bit})$ (§2530) — novel definition. Honestly disclosed as "operationally defined" and "dimensionless count parameter". OK.
2. **Bundle metric $g_{E_q} = g^{\mathrm{tw}}_\mathcal{C} \oplus g_\mathcal{B}$** (Eq. bundle_metric, §2638) — novel construction (the frame-twist horizontal block is the user's). OK.
3. **Frame-twist quadratic form $\kappa(A_\mu, A_\nu)$** (Eq. horizontal_metric, §2643) — novel; correctly distinguished from Yang-Mills invariant $\mathrm{tr}(F_{\mu\nu} F^{\mu\nu})$ at §2651 and again at §2673. OK.
4. **Three-tier perceived-geometry reading** $G^{(s)}$ structural / $G^{(p)}$ expectational / $G^{(q)}$ epistemic (§2692-2705) — novel interpretive identification. The body labels it as "We propose..." (§2701), which is the right disposition.
5. **GL(K, ℂ) frame-twist with imaginary temporal component + real-part projection** (§2755-2810) — novel; honestly labelled as worked example with postulates. See M-Spec4, M-Spec5, M-Spec6 for residual framing issues.
6. **Causal-cone alternative route** (§2812-2835) — novel; correctly labelled as conditional on three additional postulates and as not directly applicable to the current first-order natural-gradient dynamics.
7. **Consensus metric via Haar-averaged gauge orbit** (§2871-2890) — novel; honestly labelled as heuristic conditional on a non-constructed regulator. Existing M5 covers downstream invocation.
8. **Three-sector eigenvalue decomposition** (observable / subthreshold / internal, §2912-2968) — novel; labelled as definitional partition by eigenvalue threshold.
9. **$(1+3)$ dimensional structure for human agents** (§2970-2974) — novel conjecture, properly tagged "we hypothesize" and "remains a toy model demonstration".
10. **Phenomenological-information-primal stance** (§3005-3014) — novel philosophical commitment, properly tagged "metaphysical postulate, not a derived result" at §3052.

All ten constructions are either correctly labelled as novel/interpretive/conditional or have minor framing issues addressed in the Major Issues above. No bare novel construction is presented as a standard result.

## Open questions

1. Whether the pathway argument at §2759-2767 is best left as a single complexification route or rewritten to acknowledge the mixed-generator alternative (M-Spec4). The mixed-generator alternative does not weaken the manuscript's central open question (dynamical selection of an asymmetric input distinguishing one base direction from others) — it generalises it. I recommend rewriting; the author may prefer to leave the single-generator presentation for pedagogical simplicity and footnote the alternative.
2. Whether the "real-part projection" gap (M-Spec5) deserves elevation to a Major in the existing review. I have left it as Major-Spec5 in this review since it interacts with the Wick-rotation literature, which a referee from physics will inspect.
3. Whether Cassirer / Worrall / Ladyman & Ross at line 3048 need bibkeys or should be reworded as a generic "structural realist tradition" reference.
4. Whether the manuscript should also engage Verlinde 2011 (entropic gravity, arXiv:1001.0785) and Van Raamsdonk 2010 (arXiv:1005.3035, entanglement-spacetime emergence) in the "Time as Information Flow" related-work paragraph, in addition to Page-Wootters and Connes-Rovelli. These would complete the modern "information-theoretic spacetime" literature.

## Overall Verdict

**Major revisions.** The section's mathematical content within its admitted postulate sets is correct (sympy-verified for the worked example; bundle-pullback formulation is consistent with [Nakahara2003, Frankel2011]; Sylvester invocation is correct). The in-place disclosures of derivation gaps and the "What is derived vs what is postulated" box at §3089-3107 are exceptionally strong relative to typical speculative-physics literature. The remaining issues are:

- **One citation-fabrication-level error** (Lloyd2002 misattribution of the Wheeler/Cummings aphorism at line 2534).
- **One major related-work gap** (Page-Wootters 1983 and Connes-Rovelli 1994 are absent from a manuscript whose emergent-time thesis is the closest descendant of those constructions).
- **One citation drift** (Jacobson1995 entanglement-entropy gloss at line 2534).
- **One overgeneralization** in the signature pathway (M-Spec4: GL(K, ℂ) extension is not strictly necessary; mixed-generator real-frame configurations within +tr also produce Lorentzian signature).
- **One framing issue** about the real-part projection (M-Spec5: the standard Wick rotation does not involve this step; the manuscript flags the gap but does not contrast with the standard).
- **One generator-normalization annotation** at the boxed Lorentzian metric (M-Spec6).
- **A handful of declarative (I) sentences** that need their conditional qualifiers restored (M-Spec7: §2701, §3003, §3026, §3046).

These are correctable without disturbing the section's structure. Once corrected the section's epistemic discipline will match the conditional voice it already commits to in its strongest passages (§2767, §2810, §2880, §3014, §3052, and the §3089 box).

The section should not be relegated to a follow-up paper. It is the manuscript's central interpretive contribution, and the in-place disclosures are valuable in their own right as a model for how to write speculative-physics theory honestly. With the Page-Wootters / Connes-Rovelli engagement added and the M-Spec1-7 issues addressed, this section will read as a careful interpretive proposal rather than as an overclaimed derivation — which is what the manuscript at its best already aims for.

## Files of relevance

- Manuscript range under review: `Attention/Participatory_it_from_bit.tex` lines 2521-3128.
- Bibliography file: `Attention/references.bib` (Page-Wootters and Connes-Rovelli need to be added; Cassirer / Worrall / Ladyman & Ross need bibkeys if retained).
- Sympy verification scripts (transient, in `/tmp/`): `verify_worked.py` (manuscript's worked example), `verify_worked2.py` (mixed-generator alternative). Re-runnable; results inline above.
- Existing review of broader manuscript: `Attention/REVIEW_PARTICIPATORY_IFB_2026-05-18/manuscript_review.md` (M5 and M6 cover adjacent issues; the present review's M-Spec1 through M-Spec7 are independent findings within the speculative range).
