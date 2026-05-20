# Blue Opening — pifb-discussion-lahav-neemeh

## Steelman (opposing position)

A strong red position would be that the subsection misrepresents Lahav-Neemeh 2022 by reading "the explicit transformation law is indicated rather than supplied" out of an essay-in-progress whose stated aim is precisely to describe such transformations; that the Alice/Bob "composition" at eq:alice_bob_composition is a trivial identity decomposition $A \cdot A^{-1} \cdot B = B$ dressed up as a derivation; and that the "explicit invariant scalar" claim at 3381 stands in tension with the 3378 admission that per-agent frame-independent invariance is an open question — the lay reader will hear "invariant scalar in the SR sense" and the diagonal-only qualifier three sentences earlier is too far away to do the disambiguating work.

## Position

The Lahav-Neemeh subsection at `Participatory_it_from_bit.tex:3356-3396` is correctly calibrated: (i) the 3359 characterization that the published Lahav-Neemeh transformation law is "indicated rather than supplied" is confirmed by primary-source review of *Frontiers in Psychology* 12, 704270; (ii) the Alice/Bob composition at eq:alice_bob_composition is mathematically valid as a structural decomposition of a direct transport, and the manuscript labels it as such rather than overclaiming a new derivation; (iii) the three divergences (substance/ontology, pan-agentism/multi-scale, second-postulate) are substantive rather than rhetorical; (iv) the diagonal-vs-per-agent invariance question is honestly registered at 3378 as an open question, and the 3381 "explicit invariant scalar" claim refers to the diagonal-action invariance that the 3378 paragraph establishes three sentences earlier; (v) the open-question registration at 3395-3396 names the phenomenal-aggregation problem as the principal piece of explanatory work the structuralist commitment still owes.

## Evidence

**E1. Primary-source confirmation that Lahav-Neemeh 2022 does not supply the cognitive-frame transformation law** — `WebFetch https://www.frontiersin.org/articles/10.3389/fpsyg.2021.704270/full`:

> "The paper does not provide an explicit transformation equation between first-person and third-person cognitive frames of reference... The authors acknowledge the need for transformations but remain vague: 'For any relativistic phenomenon there is a formal transformation between the observers of different frames of reference... Consciousness as a relativistic phenomenon also has such transformation rules. We will describe the transformations...' However, the provided excerpt ends before delivering these promised transformations in concrete mathematical form."

The paper provides the standard Lorentz transformation (Eq. 1) and cognitive-system equations (Eqs. 2-10), but the actual map between cognitive frames is asserted, not derived. This vindicates the manuscript's 3359 framing: "the explicit transformation law... is indicated rather than supplied." The framing is exact, not a strawman.

**E2. Primary-source confirmation that Lahav-Neemeh 2022 is silent on multi-scale extension** — same WebFetch:

> "Critical Gap: Missing Hierarchical Framework. No discussion appears regarding: Extension to nested or hierarchical cognitive systems; Multiple scales of consciousness; How transformations compose across different observer levels; Practical application of transformations between specific agents."

This supports the 3361 characterization that "Lahav and Neemeh's account is single-scale" and the 3386-3387 Divergence II framing. The pan-agentic multi-scale extension is a genuine extension, not a re-labeling.

**E3. Mathematical verification of the Alice/Bob composition** — executed sympy + scipy at the working directory:

```
LHS  (composed): exp(phi_B)*exp(-phi_i)
RHS  (direct)  : exp(phi_B)*exp(-phi_i)
Numerical composition error: 6.54e-16  (K=4 random GL(K) matrices, np.random.seed(0))
```

The composition $\Omega_{Bob,i} = [\exp(\phi_B)\exp(-\phi_A)] \cdot [\exp(\phi_A)\exp(-\phi_i)]$ collapses to $\exp(\phi_B)\exp(-\phi_i)$ exactly, because $\exp(-\phi_A)\exp(\phi_A) = I$ holds for any element of $\mathfrak{gl}(K)$ (same argument; commutativity not required). The manuscript at 3376 names this what it is: "The structural map their account indicates is, in the present framework, a derived composition rather than a posited correspondence, and the formalism specifies exactly which two transports it is built from." The composition rule is the standard transport-law composition for the two-exponential parameterization $\Omega_{ab} = \exp(\phi_a)\exp(-\phi_b)$, equivalent in form to the gauge-bundle parallel transport composition rule $U(c \leftarrow b) U(b \leftarrow a) = U(c \leftarrow a)$ [Nakahara2003 §10.1, parallel transport composition].

**E4. The "explicit invariant scalar" claim at 3381 is reconciled by the 3378 diagonal-action passage three sentences earlier** — `Participatory_it_from_bit.tex:3378`:

> "$\mathcal{F}$ is a scalar under the diagonal action of $\mathrm{GL}(K)$, even though its constituent KL terms transform between frames... The invariance just stated is under the diagonal subgroup of $\mathrm{GL}(K)^N$, i.e., one common gauge transformation applied identically to every agent... Whether a fully frame-independent scalar (analogous to $ds^2$) survives the full per-agent action of the gauge group is itself an open question."

Then `Participatory_it_from_bit.tex:3381` says "an explicit invariant scalar" without re-qualifying. A reader of 3381 in isolation could misread, but the manuscript's own 3378 paragraph immediately preceding establishes the diagonal-vs-per-agent distinction and explicitly registers the per-agent question as open. The intra-subsection reading carries the qualifier forward. The manuscript could tighten with "(under the diagonal action)" inserted at 3381, but the substance is consistent.

**E5. The three divergences are substantive, not rhetorical**:

- **Divergence I (substance/ontology, 3383-3384)** distinguishes Lahav-Neemeh's relational physicalism (physical reality ontologically prior, observer-relatively presented) from the framework's pan-agentic structuralism (no observer-independent substrate). The 3384 sentence is the load-bearing differentiation: "both deny that physical reality is independent of cognitive frame but it is not the same view, and it should not be quoted as if it were." This is precisely the intellectual hygiene that grounds a "Following Lahav and Neemeh" framing — close enough to invite the analogy, far enough that the analogy is named rather than collapsed.
- **Divergence II (pan-agentism/multi-scale, 3386-3387)** is the multi-scale extension whose absence in Lahav-Neemeh is confirmed by E2.
- **Divergence III (second-postulate/Lorentzian-signature, 3389-3390)** notes that neither the Lahav-Neemeh nor the present framework supplies an analogue of the second postulate of special relativity; the framework offers a candidate mechanism (frame-twist quadratic form yielding indefinite signature in the two-dimensional worked example at sec:signature_resolution) and explicitly registers the fully four-dimensional nonlinear extension as open.

**E6. The intellectual debt is correctly directed at 3381**: "The intellectual debt runs in the opposite direction: their reframing of phenomenal consciousness as a frame-relative quantity is the philosophical move that licenses the construction in this paper." The framework takes its philosophical license from Lahav-Neemeh, not the other way around. This is the right epistemic register.

**E7. The qualia-indeterminacy reframing is honest** — `Participatory_it_from_bit.tex:3392-3393`: "relocated rather than dissolved on the structuralist route adopted here." The 3395-3396 paragraph then states the phenomenal-aggregation problem as "the central open research question raised by the pan-agentic commitment, and it is honestly an open one: it is what would have to be done to convert the structuralist route on qualia indeterminacy from a relocation into a derivation." The subsection registers what it does not deliver. The judge should not penalize the manuscript for failing to solve a problem it explicitly names as unsolved.

## Falsification conditions

This position is wrong if any of the following hold:

**F1.** Direct primary-source review of Lahav-Neemeh 2022 §4-5 (beyond the §3 extract that the publicly available Frontiers HTML truncated for me) shows that the paper DOES supply an explicit transformation law between first-person and third-person cognitive frames — a numbered equation, matrix, group element, or named function — rather than asserting the correspondence at a conceptual level. The 3359 characterization then needs revision in place.

**F2.** Direct primary-source review of Lahav-Neemeh 2025 follow-up shows that the 2025 paper supplies the transformation law that the 2022 paper indicated. The manuscript cites both `\cite{LahavNeemeh2022, LahavNeemeh2025}`; if 2025 closes the gap, the 3359 characterization that "in the published formalization, the explicit transformation law... is indicated rather than supplied" must be revised to "in the 2022 paper, indicated rather than supplied; in the 2025 paper, [supplied form X]."

**F3.** Lahav-Neemeh 2022 or 2025 actually develops a multi-scale (nested-observer / hierarchical) extension of the relativistic theory, in which case the 3361 "single-scale" characterization and the Divergence II framing at 3386-3387 collapse.

**F4.** The 3381 "explicit invariant scalar" claim is taken (in context) to mean a per-agent gauge-invariant scalar analogous to $ds^2$, rather than the diagonal-action-invariant $\mathcal{F}$. If a careful reader of the subsection-as-a-whole still hears 3381 as overclaiming, R2 needs the in-place qualifier "(under the diagonal action of $\mathrm{GL}(K)$)" inserted at 3381 to keep the 3378 distinction live for the cross-reference reader.

**F5.** The cross-scale factor at eq:alice_bob_composition is not actually fixed by an independent consensus-aggregation construction in `sec:participatory` — i.e., if $\phi_{Alice}^{(N)}$ is not constrained to compose from its constituent $\phi_i^{(N-k)}$ in the way the 3376 prose claims. If the cross-scale factor is itself just a free parameter, the "composition" reduces to a renaming of $\Omega_{Bob,i}$ rather than a structural decomposition, and the manuscript's "structural map... is, in the present framework, a derived composition" overclaims. (This is independent of the trivial-identity worry: the composition is mathematically correct in either case; the question is whether the cross-scale piece carries independent content.)

**F6.** The 3395-3396 open-question paragraph turns out to understate the debt: if pan-agentic structuralism owes substantially more than just the phenomenal-aggregation problem (e.g., it owes an account of why scale-0 frames are gauge frames in the same algebraic sense as scale-N frames, and that account is not present in `sec:participatory`), the "registers the central open question" framing in the claim is incomplete.

I do not currently have evidence pointing at F1-F6. E1 and E2 actively support the position against F1 and F3. E3 confirms the math against the trivial-decomposition concern that motivates the F5 question (even if F5 turns out to hold, the math is still correct; what would change is the prose framing of "derived composition"). The position stands on the cited primary sources.
