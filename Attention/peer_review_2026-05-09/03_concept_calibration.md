# Peer Review 03: Claim Calibration, Scope, and Framing

**Reviewer:** Senior peer reviewer (claim-calibration axis)
**Manuscript:** `Attention/Participatory_it_from_bit.tex`
**Calibration commit under review:** 354d9613 ("Round 16: Calibration pass on five reviewer criticisms")
**Date:** 2026-05-09

## Scope

The Round-16 calibration pass touched four named sections in the body: `sec:transformers` (retitled "Recovery of ... as a Gauge-Fixed Limit"), `sec:formal_equivalence` (mean-gradient equivalence as primary, full equivalence as conditional), `sec:mass` (retitled to "... A Mass Analogy"), and `app:conditional_uniqueness` (retitled to "Conditional Representation Theorem"). This review checks whether (i) those calibrations were propagated into the abstract, introduction, discussion, conclusion, and the Wheeler/Lahav/consciousness/qualia conceptual scaffolding that depends on them, and (ii) whether other conceptual claims in the manuscript still overclaim relative to the formalism.

The headline finding is that the calibration was applied surgically to the four targeted sections but was not propagated to the abstract or to the conclusion. The same overclaims that the calibration removed from `sec:transformers` reappear, in stronger form, in the framing matter. There is also an internal contradiction inside `sec:lahav_convergence` whose subsection title was apparently not revisited.

---

## FINDINGS

### 1. Abstract still says "We prove ... reduce to" — directly contradicts the retitled `sec:transformers`. **BLOCKER**

**Location.** Line 54, abstract.
**Quoted claim.**
> "We prove that, under a zero-dimensional isotropic-Gaussian limit, the KL-consensus weights reduce to scaled dot-product attention up to standard normalization and bias assumptions (Section~\ref{sec:transformers})."

**Defect.** The Round-16 commit explicitly retitled `sec:transformers` to "Recovery of Dot-Product Attention as a Gauge-Fixed Limit" and replaced the lead paragraph with: "we therefore frame the result not as a derivation of $QK^\top$ attention from the gauge framework, but as a statement that standard attention is consistent with a broader gauge-covariant KL attention mechanism under these limits." The abstract still uses the strongest possible verbs — "We prove" + "reduce to" — and omits the two non-trivial inserted ingredients (gauge fixing to a shared frame, and a separately introduced learned bilinear $M$ that is essentially what makes the algebra factor through $QK^\top$). This is the single clearest case of the calibration introducing an inconsistency: the body and the abstract now contradict each other on the strength of the central technical claim.

**Suggested fix.** Replace with calibrated language matching the section, e.g.:
> "We show that, under gauge-fixing to a shared isotropic-Gaussian limit and the introduction of a learned bilinear compatibility $M$, scaled dot-product attention is recovered as a gauge-fixed limit of the KL-consensus construction (Section~\ref{sec:transformers}); we do not claim it is uniquely derived from the gauge framework, since once $M$ is introduced the factorisation $M = W_Q W_K^\top$ is essentially algebraic."

---

### 2. Conclusion says attention "emerges rigorously" and "from first principles" — same overclaim, post-calibration. **BLOCKER**

**Location.** Lines 3605 and 3621, Section `sec:conclusion`.
**Quoted claims.**
- (3605) "The framework's most concrete achievement is demonstrating that transformer attention mechanisms emerge rigorously as the zero-dimensional limit of gauge-theoretic variational inference."
- (3621) "demonstrating that attention mechanisms emerge from first principles ..."

**Defect.** Both sentences invoke the very framing the Round-16 calibration removed from the body. "Emerge rigorously as the zero-dimensional limit" and "emerge from first principles" assert non-conditional derivation; the body now disclaims exactly that. The conclusion is also where a reviewer or generalist reader will land first; leaving the abstract and conclusion uncalibrated nullifies most of the work the calibration commit performed.

**Suggested fix.** Replace the phrase "emerge rigorously as the zero-dimensional limit" with "are recovered as a gauge-fixed isotropic-Gaussian limit, with a separately introduced learned bilinear, of gauge-theoretic variational inference"; replace "emerge from first principles" with "are recovered under explicit limits from first principles".

---

### 3. "Independent Convergence with Lahav and Neemeh" — title contradicted by the section's own admission. **MAJOR**

**Location.** Section title at line 3443, contradicted by line 328 and by the introduction at line 162.
**Quoted claim (title):** `\subsection{Independent Convergence with Lahav and Neemeh's Relativistic Theory of Consciousness}`
**Quoted contradiction (line 328):**
> "The intellectual debt runs in the opposite direction: their reframing of phenomenal consciousness as a frame-relative quantity is the philosophical move that licenses the construction in this paper."

The introduction (~line 162) likewise cites Lahav-Neemeh as one of the three philosophical antecedents the framework's pan-agentic structuralism extends.

**Defect.** "Independent convergence" entails that two parties arrived at the same result without reading each other; the section itself states the opposite — that Lahav-Neemeh's reframing licensed the present construction. The substantive technical contribution (writing down a transformation law that they assert exists but do not write down, plus the multi-scale extension) does not require the "independent" framing and is undermined by it; an honest reader will catch the contradiction and discount the rest.

**Suggested fix.** Retitle to one of: "Mathematical Formalisation of Lahav and Neemeh's Cognitive Frames", "Structural Correspondence with the Relativistic Theory of Consciousness", or "Gauge-Theoretic Realisation of the Cognitive-Frame Transformation Law". Drop the word "independent" entirely. The honest substantive claim — that the framework supplies an explicit transformation law where their account asserts one without writing it down — is stronger than "independent convergence" and is what the section actually delivers.

---

### 4. "Self-Excited Perpetual Motion" subsubsection — Wheeler rhetoric not earned by the math. **MAJOR**

**Location.** Lines 2578–2588, subsubsection title and surrounding paragraphs.
**Quoted claim.**
> "\subsubsection{Self-Excited Perpetual Motion}
> The participatory loop sustains non-equilibrium dynamics through perpetual feedback. ... The system exhibits what Wheeler called a 'self-excited circuit'~\cite{Wheeler1983}; i.e. a self-sustaining loop where observation and reality co-constitute each other through perpetual informational exchange. ... The universe is not a passive arena but an active participant in its own evolution."

**Defect.** The actual mathematical/empirical content here is: in a single-seed (random seed 2) simulation with a threshold-based meta-agent detector, the system reorganises into a 13-scale structure and the four non-equilibrium diagnostics remain elevated. This is "ongoing dynamics in a single run with a threshold detector" — the manuscript flags the single-seed and detector-conditional caveats explicitly elsewhere (Section `sec:meta_agent_emergence` opening, Section `sec:scope_limitations`). The phrases "perpetual motion", "the universe is ... an active participant in its own evolution", and the unhedged Wheeler "self-excited circuit" identification are physics-loaded rhetoric ("perpetual motion" is, in physics, a banned concept; the manuscript is using it as a literary flourish that will read to a physics referee as a confusion of category). The subsubsection also asserts: "the complexity of the hierarchical wilderness increases globally through meta-agent formation. This may account for the observation of local areas of our universe exhibiting explosive complexity growth despite the second law's mandate toward increasing entropy" — a claim about cosmological-scale entropy that the framework has no right to make.

**Suggested fix.** Retitle to "Sustained Non-Equilibrium under the Threshold-Detector Dynamics". Replace the Wheeler-quote sentence with the calibrated language already present elsewhere ("a computational mechanism consistent with the participatory-structure picture, conditional on the threshold detector"). Delete the "second law's mandate toward increasing entropy" sentence; if retained, attach the same single-seed / threshold-detector / no-cosmology disclaimer.

---

### 5. "Comparison to Wheeler's Vision" table — no disclaimer that mass, spacetime, quantum theory are not derived. **MAJOR**

**Location.** Lines 2833–2849, subsubsection and Table `tab:wheeler_comparison`.
**Quoted claim.** A 6-row two-column table aligning Wheeler concepts ("It from bit", "Participatory universe", "Observer-dependent reality", "Delayed choice", "Self-excited circuit", "Law without law") with framework constructs ("Geometry via pullback", "Multi-scale feedback loops", "Gauge-frame-dependent metrics", "Transport operator dependence", "Non-equilibrium dynamics", "Info-theoretic necessities"). The table caption reads: "Correspondence between Wheeler's philosophical concepts and our mathematical realizations."

**Defect.** The word "realizations" overclaims. "Law without law $\to$ Info-theoretic necessities" is particularly strained: the manuscript does not derive any physical law, let alone show that physical law arises as an information-theoretic necessity. "Delayed choice $\to$ Transport operator dependence" implies the framework realises the delayed-choice quantum-eraser-style observer dependence; transport operators are gauge transformations between agent frames, not delayed-choice quantum measurements. The table will be read by Wheeler-aware referees as a claim of comprehensive realization of the participatory program, which is not what the body delivers.

**Suggested fix.** Either (a) demote the table to a "structural correspondence" table with caption: "Each row pairs a Wheeler concept with a framework construct that plays a structurally analogous role; we do not claim to have derived mass, spacetime geometry, or quantum mechanics from the framework, and the analogies in 'delayed choice' and 'law without law' are loose rather than mathematical"; or (b) drop the two weakest rows ("Delayed choice", "Law without law") entirely. Either way, follow the table with one explicit sentence: "Mass, spacetime, and quantum theory are not derived in this manuscript; this table is a structural map, not a derivation summary."

---

### 6. "Central Result: Spacetime from Information Geometry" boxed result — title overclaims body. **MAJOR**

**Location.** Boxed minipage at lines 2540–2563.
**Quoted claim (box header).** "Central Result: Spacetime from Information Geometry"
**Quoted body.** Under "What we derive rigorously" the box presents a Gram-Schmidt orthogonal decomposition of $T_q\mathcal{B}$ into $\mathbb{R}\dot q \oplus V_{\mathrm{obs}} \oplus V_{\mathrm{subthresh}} \oplus V_{\mathrm{internal}}$. Under "What we postulate" the box lists, as postulates, the Lorentzian signature, the imaginary frame component, the real-part projection, and the count of three large spatial eigenvalues.

**Defect.** Calling this the "Central Result: Spacetime from Information Geometry" sets the expectation that spacetime has been derived from information geometry. The box's own honest content is that spacetime requires every geometrically interesting feature — signature, imaginary axis, projection, dimension count — to be postulated, and what is "derived rigorously" is a textbook orthogonal decomposition with respect to the Fisher metric. The honest title is much weaker: "Decomposition of the Belief Tangent Space and the Postulates Required for a Spacetime Reading". A reader who reads only the box (boxes are read first) will leave with the wrong conclusion about what the manuscript establishes.

**Suggested fix.** Retitle the box to "Belief Tangent-Space Decomposition and the Spacetime Reading: What Is Derived and What Is Postulated". Replace "The radical claim:" subheading (which currently asserts "Phenomenal spacetime is a tiny 4D subspace selected by eigenvalue magnitude") with "An interpretive reading:" and add the qualifier that the eigenvalue-magnitude selection is conjectural and not derived from variational dynamics.

---

### 7. Conclusion: "We have shown rigorously that ... induce Riemannian metrics ... Gauge-invariant collective metrics can be constructed ... naturally produce multi-sector dimensional structure". **MAJOR**

**Location.** Line 2538 (end of `sec:pullback`) and propagated into the conclusion's framing.
**Quoted claim.**
> "We have shown rigorously that smooth sections of statistical bundles induce Riemannian metrics on base manifolds via Fisher-Rao pullback. These induced metrics are observer-dependent ... Gauge-invariant collective metrics can be constructed through gauge averaging, providing a framework for consensus geometry. The eigenvalue hierarchies of these induced metrics naturally produce multi-sector dimensional structure, separating observable, subthreshold informational, and internal components."

**Defect.** Three slips here:
1. "Gauge-invariant collective metrics can be constructed through gauge averaging" — `sec:consensus_metric` (which the calibration did not retouch but should have) explicitly admits that the gauge-orbit average over local $g(c)$ is "an infinite-dimensional functional integral that requires a gauge-fixing or regulator choice", that "no finite gauge-orbit average over local $g$ exists without such a choice", and that the non-compact $\mathrm{SO}(1,3)$ case has infinite Haar measure. So gauge-invariant collective metrics emphatically *cannot* be constructed without a regulator the manuscript does not provide. The "rigorously shown" phrasing here is in direct conflict with `sec:consensus_metric`.
2. "Naturally produce multi-sector dimensional structure" — the eigenvalue hierarchy producing exactly $1+3+\text{subthreshold}+\text{internal}$ requires further input (the manuscript even postulates "the observable sector has exactly 3 large spatial eigenvalues" in the boxed central result). "Naturally" is doing work that the math does not do.
3. The whole paragraph deserves an explicit calibration matching the calibrated `sec:consensus_metric` text.

**Suggested fix.** Replace "Gauge-invariant collective metrics can be constructed through gauge averaging" with "A candidate gauge-invariant collective metric can be defined through gauge averaging (Section~\ref{sec:consensus_metric}), conditional on a regulator that the present manuscript does not construct." Replace "naturally produce multi-sector dimensional structure" with "permit, under additional eigenvalue-magnitude postulates, a multi-sector dimensional reading".

---

### 8. "Bit $\to$ It $\to$ Bit Again" boxed equation — slogan dressed as math. **MINOR**

**Location.** Subsubsection at line 2828 and boxed display equation immediately following.
**Quoted claim.**
> "$\boxed{\text{Bit (beliefs)} \xrightarrow{\text{pullback}} \text{It (geometry)} \xrightarrow{\text{transport}} \text{Bit (updated beliefs)}.}$"

**Defect.** This is a verbal cycle in a `\boxed` mathematical-display environment. Boxing a slogan signals theorem-level content; nothing computational, dimensional, or quantitative follows. A reviewer will read this as rhetorical overreach.

**Suggested fix.** Either remove the `\boxed` and present as inline prose, or replace with an actual equation that captures the loop, e.g. the schematic update $q_i^{(n+1)} = \mathcal{F}_{\mathrm{transport}}(\mathcal{F}_{\mathrm{pullback}}(\{q_j^{(n)}\}))$, with the maps named precisely.

---

### 9. "Mass from Statistical Precision" lead paragraph still appears in TOC and prose despite section retitle. **MINOR**

**Location.** Section retitled at line 1519 to "Statistical Precision as Configuration-Space Stiffness: A Mass Analogy", but the conclusion still says (line 3171, in the discussion-level "We have shown"):
> "We have shown that within the framework the Fisher-Rao precision plays the role of effective mass: ..."

**Defect.** The Round-16 commit retitled the section and the within-framework subsection but did not propagate the calibrated framing into the discussion-level summary that follows in `sec:implications`. "Plays the role of effective mass" is the strongest within-framework statement that survives, and it should be qualified with "in the Newtonian-shaped harmonic analogy of Section~\ref{sec:mass} and conditional on the kinetic-metric postulate of Section~\ref{sec:velocity_quadratic}".

**Suggested fix.** Append the conditional clause; or replace "plays the role of effective mass" with "plays the role of a configuration-space stiffness that, conditional on the kinetic-metric postulate, scales as an effective mass in the harmonic limit".

---

### 10. Conclusion: "transformer architectures have deep theoretical foundations in gauge theory". **MINOR**

**Location.** Line 3617, conclusion.
**Quoted claim.** "It demonstrates that transformer architectures have deep theoretical foundations in gauge theory and variational inference, not merely empirical success."

**Defect.** Given the calibrated body now states the recovery is one gauge-fixed limit among several possible derivations of attention, "deep theoretical foundations" hides the non-uniqueness. Many other constructions also derive attention (kernel-based, energy-based, dictionary-learning, etc.); the gauge-theoretic route is one such.

**Suggested fix.** Replace "have deep theoretical foundations" with "admit a principled derivation, among several possible, from gauge theory and variational inference". This is honest and still defends the contribution.

---

### 11. Conclusion: "observer-dependent reality can be rigorously formalized". **MINOR**

**Location.** Line 3621, conclusion.
**Quoted claim.** "demonstrating that ... observer-dependent reality can be rigorously formalized ..."

**Defect.** What is rigorously formalized is "agent-frame-dependent pullback metrics on a statistical manifold." Calling this "observer-dependent reality" is the interpretive move the manuscript otherwise (in `sec:consciousness_section` opening paragraph and in the qualia subsection) carefully flags as interpretive. The conclusion drops the hedge.

**Suggested fix.** Replace with: "demonstrating that an information-geometric formalisation of agent-frame-dependent phenomenal geometry is mathematically well-defined; whether this constitutes a formalisation of observer-dependent reality in any stronger metaphysical sense is a separate interpretive question."

---

### 12. Abstract enumerates limitations but omits the consensus-metric regulator gap. **MINOR**

**Location.** Abstract, line 56.
**Quoted claim.** "The four-dimensional nonlinear extension, dynamical selection of the imaginary generator, dimensional analysis between informational and physical units, and a rigorous quantum extension are open."

**Defect.** The abstract's limitations enumeration is solid on the signature/dimension/quantum gaps but does not flag that the central "consensus metric" / "gauge-invariant collective geometry" construction (the object that the discussion repeatedly calls "the closest analog to objective reality") is conditional on a regulator the manuscript does not construct. This is a structural limitation comparable in importance to the imaginary-generator gap.

**Suggested fix.** Append to the limitations sentence: "; the gauge-orbit-averaged consensus metric used as the framework's candidate gauge-invariant geometry is conditional on a regulator that we do not construct (Section~\ref{sec:consensus_metric})."

---

### 13. Hard-Problem and qualia sections: well-calibrated; no remaining overclaim found. **NIT — POSITIVE**

**Location.** `sec:consciousness_section` and `sec:qualia` (lines 3393–3441).

**Assessment.** These sections honestly disclose: (a) interpretive status of the "qualia as gauge-frame-dependent phenomenology" claim ("We do not assert identity ... the framework supplies a structural correlate of phenomenal variation, not a derivation of phenomenal content"); (b) that the framework "relocates the question from 'why does neural computation feel like anything?' to 'why does the geometry of inference feel like anything?'" rather than answering it; (c) that "the framework provides mathematical vocabulary for discussing consciousness without explaining consciousness". The "Information-geometric structuralism" paragraph also explicitly disclaims the four classical positions it does not occupy. This is the calibration tone the rest of the conceptual scaffolding should match.

**No fix required.** Recommend using the language of these sections as the template for fixes in items 1, 2, 5, 6, 10, 11.

---

### 14. `sec:signature_resolution` and worked example: honestly calibrated. **NIT — POSITIVE**

**Location.** Section starting at line 2180, worked example at line 2225 (`sec:worked_signature`), causal-cone alternative at line 2317 (`sec:causal_cone_route`).

**Assessment.** The section states explicitly: (i) the construction is 2D and linearised; (ii) the imaginary temporal generator is *imposed*, not derived ("We make no claim that variational free-energy minimisation selects this configuration"); (iii) the real-part projection is a separate further postulate; (iv) the choice $+\mathrm{tr}(AB)$ vs $-\mathrm{tr}(AB)$ for the bilinear is a postulate; (v) the construction stays inside Regime I with $F_{\mu\nu} = 0$; (vi) the alternative causal-cone route conflicts with the framework's first-order parabolic dynamics. These disclosures are the honest calibration target — the abstract and conclusion (items 1, 2) should match this tone.

**No fix required.** This section is in fact the gold standard of calibration in the manuscript and should be cited internally as the template.

---

### 15. `sec:open_problems` Lorentzian-signature entry: tone-mismatch with conclusion. **MINOR**

**Location.** Lines 3548–3554, "The Lorentzian Signature Problem" entry.
**Quoted claim.** "**Status:** A candidate mechanism has been identified in a 2D linearised worked example. The signature problem is not yet resolved. ... This worked example demonstrates that an indefinite signature is at least compatible with the framework's gauge structure; it does not show that the framework derives it."

**Assessment.** This is calibrated correctly. However, the conclusion (line 3607) then says of the same worked example: "A candidate mechanism is exhibited ... in the worked example of Section~\ref{sec:worked_signature}: imaginary components ... can produce indefinite signature ..." with the disclaimers attached, but earlier in the same paragraph: "The framework's most concrete achievement is demonstrating that transformer attention mechanisms emerge rigorously ..." (item 2 above). The mismatch is that the open-problems list is honestly hedged while the conclusion's framing in item 2 is not.

**Suggested fix.** None for this item itself; this is flagged as evidence that the calibration tone is internally inconsistent. Fixing items 1, 2 brings the rest into alignment.

---

## SUMMARY

The Round-16 calibration commit was applied surgically and correctly to the four named target sections (`sec:transformers`, `sec:formal_equivalence`, `sec:mass`, `app:conditional_uniqueness`). It was not propagated to the abstract, the conclusion, or to the conceptual-scaffolding subsubsections that depend on those same sections. The result is a manuscript whose body now contradicts its own framing matter on the strength of the central technical claim ("recovered as a gauge-fixed limit" in the body vs "We prove ... reduce to" in the abstract and "emerge rigorously ... from first principles" in the conclusion).

Independently of the calibration pass, three structural overclaims survive in the discussion: (i) the "Independent Convergence" subsection title is contradicted by the section's own admission of intellectual debt to Lahav-Neemeh; (ii) the "Self-Excited Perpetual Motion" subsubsection deploys Wheeler rhetoric and physics-loaded vocabulary that the single-seed, detector-conditional simulation does not earn; (iii) the "Central Result" boxed display titled "Spacetime from Information Geometry" presents a textbook Gram-Schmidt decomposition under a header that asserts the framework derives spacetime.

The conceptually well-calibrated sections of the manuscript (the qualia / hard-problem treatment in `sec:consciousness_section` and the worked-example disclosures in `sec:worked_signature`) demonstrate that the author can write at the calibration level required. The fix is straightforward: copy the tone of those sections into the abstract, the conclusion, and the four discussion items flagged above.

### Verdict on the claim-calibration axis

**MAJOR REVISION REQUIRED — DO NOT ACCEPT IN CURRENT FORM.**

The two BLOCKER items (abstract and conclusion contradicting the calibrated body) make the manuscript currently self-inconsistent on its central technical claim. A reader who takes the abstract and conclusion at face value will leave with the impression that the framework derives transformer attention, derives spacetime from information geometry, and converges independently with Lahav-Neemeh — none of which the calibrated body supports. The defect is not theoretical; it is editorial follow-through. With the items 1–7 fixes applied, the manuscript reaches the calibration level set by `sec:worked_signature` and `sec:consciousness_section` and is in shape for substantive technical review.

### Priority order for fixes

1. Item 1 — abstract (single sentence rewrite). Required.
2. Item 2 — conclusion (two-phrase rewrite). Required.
3. Item 3 — Lahav-Neemeh subsection retitle. Required.
4. Items 4, 5, 6 — three retitles plus a one-sentence disclaimer each. Required.
5. Item 7 — `sec:pullback` summary calibration to match `sec:consensus_metric`. Required.
6. Items 8, 9, 10, 11, 12 — minor wording adjustments. Recommended.
7. Items 13, 14, 15 — no-action positive observations and consistency note.

### Things that are right and should be preserved

- The qualia / hard-problem honest non-novelty admission in `sec:consciousness_section`.
- The worked-example disclosures in `sec:worked_signature` and `sec:causal_cone_route`.
- The body-level retitles and proposition rewrites delivered by the Round-16 commit.
- The pan-agentic structuralism vs relational physicalism honest divergence catalogue in `sec:lahav_convergence` paragraphs (Divergences I, II, III, and the central-open-question paragraph), which is exactly the right tone for substantive philosophical positioning.
