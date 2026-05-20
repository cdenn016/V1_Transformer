# Verdict — pifb-spec-ext-signature-collective

## Outcome

RED_WINS (narrow)

## Decisive evidence

Three primary-source-anchored miscalibrations survive blue's defense and are conceded in substance by blue's own pre-empted attack and falsification list:

1. **Wick rotation framing at line 2745.** Standard Wick rotation, as defined in Wick 1954 (*Phys. Rev.* 96, 1124) and codified in Zinn-Justin 2002, *Quantum Field Theory and Critical Phenomena* §3, continues a real base coordinate to imaginary and yields a real Euclidean metric throughout; Hartle-Hawking 1983 (*Phys. Rev. D* 28, 2960) applies the same prescription to the gravitational path-integral measure. The manuscript at line 2745 labels the construction "a Wick rotation performed inside the gauge frame rather than on the base coordinates," but the very next paragraph at line 2771 admits the real-part projection "has no Wick counterpart" — an internal contradiction with the 2745 label. Blue's pre-emption explicitly concedes that "a one-clause editorial tightening at 2745 ... would make the lead-in match the 2771 honest admission."

2. **1+3 vs 2+2 selection at line 2779.** The manuscript at 2779 admits "the construction does not currently distinguish 1+3 from 2+2 on dynamical grounds." Yet the subsection title "Temporal Structure and the Signature Problem," together with the step-3 framing at line 2737 ("Subgroup restriction to SO(1,3)"), advertises Lorentzian (1+3) as the target. The construction reaches a generic (1, n−1) signature for one imaginary direction or (2, n−2) for two; 1+3 is fixed by input choice, not by free-energy dynamics.

3. **Cross-reference gap at lines 2864–2868.** The closing "Connection to Physical Gauge Invariance" paragraph advances the same "gauge invariance as multi-agent consensus requirement" hypothesis as the Discussion subsection at lines 3228–3303 (Gauge Invariance as Cognitive Consensus Requirement, edited at commit 68ebfec8), but without inheriting the parallel section's "metaphysical interpretation, not a derivation" (line 3233) and "may be unfalsifiable" (line 3302) disclaimers, and without a `\ref{sec:gauge_invariance_consensus}` cross-reference. The methodology document treats internal cross-section coherence as on-mode.

## Reasoning

Both openings converge on the same three editorial defects. Blue's defense of the underlying mathematics (sector split, Regime I admission, postulate enumeration at sec:worked_signature, causal-cone disjoint-postulate framing, consensus-metric heuristic-target register) is well-cited and survives — the math is calibrated. The narrow loss is rhetorical and structural: the 2745 Wick label overreaches what 2771 admits in the same paragraph; the subsection's title and step-3 enumeration imply a (1,3) selection that line 2779 explicitly disclaims; and the 2864–2868 closing paragraph escapes the unfalsifiability hedge that the parallel Discussion section carries. Red's cited canon (Wick 1954, Zinn-Justin 2002 §3, Hartle-Hawking 1983) and the within-manuscript contradiction at 2771 give red the stronger primary-source position on point 1; line 2779 supplies the internal contradiction for point 2; and the Discussion subsection at 3228–3303 supplies the calibration mismatch for point 3. Blue's seven sub-claims defending the mathematical substance stand; the rhetorical framing on these three loci does not.

Sylvester's-law concern at 2799 is granted to blue as expository convenience, not a falsifying issue, consistent with red's own concession.

## Action

Apply three editorial edits to `Attention/Participatory_it_from_bit.tex`:

1. **Line 2745 (Wick framing).** Replace "equivalent to a Wick rotation performed inside the gauge frame rather than on the base coordinates" with "structurally analogous to a Wick-like continuation performed inside the gauge frame rather than on the base coordinates, with an additional real-part projection step that has no Wick counterpart (registered in the paragraph below)." This aligns the lead-in with the 2771 admission and removes the overreach against Wick 1954 / Zinn-Justin 2002 §3 / Hartle-Hawking 1983.

2. **Signature subsection opening or worked-example lead-in (near line 2698 or near the boxed Eq. eq:lorentzian_metric at 2768).** Add a single sentence registering that the target of the construction is the Lorentzian conformal class with arbitrary (1, n−1) signature, and that the 1+3 selection is fixed by the choice of one imaginary frame direction rather than derived from free-energy dynamics. The line 2779 admission then becomes restatement rather than mid-construction reveal. The step-3 phrasing at line 2737 should also be softened from "Subgroup restriction to SO(1,3)" to "Subgroup restriction to an indefinite-signature SO(p,q), with SO(1,3) selected by the one-imaginary-direction input."

3. **Lines 2864–2868 (cross-reference and hedge).** Insert a `\Cref{sec:gauge_invariance_consensus}` cross-reference and a single clause inheriting the parallel section's epistemic register, e.g., "Following the discussion at \Cref{sec:gauge_invariance_consensus}, this is a metaphysical interpretation rather than a derivation, and the hypothesis may not be falsifiable from within the framework." This closes the internal calibration mismatch with lines 3228–3303.

No mathematical content changes. The seven sub-claims of mathematical substance (sector split, Regime I, postulate enumeration, conformal-class observation, causal-cone disjoint postulates, consensus-metric regulator dependence, open-question framing of subgroup selection) stand as defended by blue.
