# Verdict — supplementary-general-mathematical-framework

## Outcome

RED_WINS (narrow, editorial scope)

## Decisive evidence

Both sides converged on the textual record. The two citations that broke the tie are:

1. **Sub-claims δ and ε are textually absent from `Attention/GL(K)_supplementary.tex` lines 46–177.** Blue conceded at `03_blue_rebuttal.md` lines 7–9: "Lines 46–177 of `Attention/GL(K)_supplementary.tex` contain no Fisher-Rao metric, no Gaussian-KL closed form, no natural gradient, and no variational free energy functional." This concession is verified against the evidence pack at `01_evidence.md` lines 47–58 and Red's verbatim search at `02_red_opening.md` lines 13–15.

2. **The deferral architecture lacks the internal forward references it needs.** Blue conceded at `03_blue_rebuttal.md` line 9: "A grep over lines 46–177 returns one `\ref` (line 53, `Lemma~\ref{thm:vanishing_holonomy}` of the main text) and one plain-text reference ('Eq.~7 of the main paper' at line 122). There is no `\ref{app:covariance_dynamics}`, no forward reference to §C, no forward reference to §D." The compound claim at `00_claim.md` line 23 and line 25 made the charitable reading explicit: sub-claims δ and ε are admissible as "deferred elsewhere in the supplementary *with adequate forward references*." The section does not satisfy the condition the user attached to the charitable reading.

The compound claim asserts five sub-claims jointly. Sub-claims α and γ are verified (Red conceded at `03_red_rebuttal.md` lines 4–6 against [Nakahara2003 §9–10]). Sub-claim β is partially verified — the representations `ρ_q, ρ_p`, fibers `B_q, B_p`, and associated bundles `E_q, E_p` are defined at supplementary lines 55–62, with the operationalizing Gaussian KL deferred to §B lines 199–219. Sub-claims δ and ε fail under the strict reading of "complete and self-contained" and survive only under a charitable reading whose own precondition (adequate forward references) is not met by the section.

## Reasoning

The compound claim has two prongs. The "mathematically/theoretically pure" prong passes: no equation or derivation inside lines 46–177 is mathematically incorrect, the bundle scaffold matches [Nakahara2003 §9.1, §9.4, §10.1–§10.4] canonically, and the line-53 "Bundle triviality" disclosure satisfies the canon directive at `external_canon_math.md` §3 pitfall 7 (justifying trivially vanishing holonomy by both curvature = 0 and globally trivializable transition functions). On this prong neither side produced a mathematical defect.

The "complete" prong fails. The user's claim file lists five sub-claims as part of what the section is asserted to establish, and both sides agree on the record that two of the five (δ natural-gradient information geometry, ε variational EM machinery) are textually absent from the section and that the section contains no internal forward references to the supplementary chapters where the deferred material lives. Red's primary-source attack — the verbatim-absence finding plus the zero-forward-reference finding — is uncontested. Blue's defense reduces to scope reinterpretation ("the four-subsection structure defines the actual scope; the title should be narrowed") and to the editorial-fix framing of the gap. Both are reasonable but neither rescues the literal compound claim.

The notational issue at line 76 (`p_i := σ^i_p ∈ B_p`) versus main paper line 602 (`p_i(k_i) = N(k_i; μ_{0,i}^{(q)}, Σ_{0,i}^{(q)})`) is a type-level naming clash rather than a mathematical contradiction — the supplementary's `p_i` is a section (a map `U_i → E_p`) while the main paper's `p_i(k_i)` is a probability density (a point in the statistical manifold fiber). Blue's reconciliation at `03_blue_rebuttal.md` lines 17–19 is admissible, but it satisfies the conditions of Red's falsification 3 only by relabeling the supplementary's `σ^i_p` as the model section (what the main paper calls `r_i` or what Participatory calls `s_i`). This is editorial-fix territory; it does not by itself defeat the claim but it is a real load-bearing primitive-level inconsistency that the editorial pass must resolve.

Čencov 1982 uniqueness of the Fisher-Rao metric — invoked by the canon at `external_canon_math.md` §1 and cited by the Participatory paper at line 510 — is absent by name from the gauge-transformer manuscripts (Blue's concession at `02_blue_opening.md` falsification condition 7, item 47 of opening). Editorial scholarly gap, not a derivational defect.

Aggregating: the section is canonical on what it covers (α, β-representations, γ), silent on what it should either cover or forward-reference (δ, ε), and contains one primitive-level naming clash (`p_i` line 76 vs `p_i` main-paper line 602) plus one scholarly citation gap (Čencov). The aggregate is more editorial scope than any single §3–§5 debate's RED_WINS-narrow verdict, which is the standard the eleven-debate series applied. By that same standard the compound "complete and theoretically pure" claim does not survive intact.

The verdict is RED_WINS, but the scope of the remedy is editorial, not mathematical. No equation needs rewriting; no implementation needs changing; the bundle scaffold and gauge-connection apparatus are canonical and pure as written.

## Action

Apply four editorial corrections to `Attention/GL(K)_supplementary.tex` §General Mathematical Framework before the next manuscript revision:

1. **Add a closing forward-reference paragraph at line 175** (end of §General Mathematical Framework) pointing readers to the deferred material: "The Fisher-Rao metric, Gaussian KL closed form, and covariance dynamics are developed in §\ref{app:covariance_dynamics}; the gauge-frame gradients and differential of the matrix exponential are developed in §C; the natural-gradient descent on the Gaussian manifold and the SPD retraction are developed in §D; the variational free energy functional and its E-step/M-step decomposition are developed in the main paper §3.4–§3.5." This converts the existing on-the-record deferral architecture (which Blue verified at `03_blue_rebuttal.md` line 36) into a reader-visible scaffold and satisfies the "adequate forward references" precondition the user's claim file attached to the charitable reading.

2. **Disambiguate the `p_i` notation at supplementary line 76.** Either rename `σ^i_p` to `σ^i_s` so that line 76 reads `s_i(c) := σ^i_s(c) ∈ B_p(c)` (matching Participatory line 508's `s_i ∈ B_model` and main paper line 604's `r_i` on the model fiber with superscripts `(p)`), or add a footnote at line 76 stating that the supplementary's `p_i = σ^i_p` is the model section, distinct from the main paper's belief-channel base prior `p_i(k_i)` at line 602 which lives in the belief fiber `E_q`. Reconcile the agent definition at line 71 with the main paper's effective agent state `(q_i, p_i, s_i, r_i, φ_i)`.

3. **Qualify the §A.2.1 hierarchical meta-agent subsubsection at lines 96–130.** Either define `s_i` before its first use at line 103 (the model-consensus equation `s_i = Ω̃_{ij} s_j` introduces `s_i` without a prior definition), or delete the hierarchical-meta-agent subsubsection if it is not load-bearing for the main paper. Add a one-sentence qualifier stating that the cross-scale prior-propagation construction `p_i^(s) = Ω_{i,I}[q_I^(s+1)]` of the Participatory paper at lines 539–548 is a companion-paper commitment and that the gauge-transformer treats priors as primitive boundary data per main paper lines 602/608 and the `PriorBank` implementation.

4. **Add a Čencov 1982 citation** at the point where the Fisher-Rao metric is first invoked (supplementary line 615 in §D, or earlier if the metric is introduced in §General Mathematical Framework as part of remedy 1). The reference exists at `Attention/references.bib:1832,1854` but is not currently cited. The Participatory paper cites it at line 510.

Optionally — but not required — narrow the chapter title from "General Mathematical Framework" to "Bundle Scaffold and Gauge Connection" to match the actual four-subsection scope. Remedy 1 (forward references) is the lighter-weight repair; the title-narrowing is the heavier alternative.
