# Action — pifb-theory-rock-solid

**From verdict:** RED_WINS (binding, chief reconciliation; 3/3 first-pass judges concurred).

The claim "§Theory of `Attention/Participatory_it_from_bit.tex` is rock solid and publication ready" does **not** survive adversarial pressure-testing. Five of six conjunctive sub-claims carry verified wounds.

## Recommended action

Revision pass on `\section{Theory}` (lines 180–2070) of `Attention/Participatory_it_from_bit.tex`. The eight concrete edits from the chief verdict's Action section, ordered by consequence and ease:

### High consequence — fixes load-bearing wounds

1. **Resolve the body-vs-appendix naming inconsistency at the conditional-representation theorem** (lines 1044, 1118, 1252; appendix 4258, 4261, 4351, 4357). Rename body-text "conditional uniqueness theorem" → "conditional representation theorem"; rename LaTeX label `thm:uniqueness_app` → `thm:representation_app`; re-enumerate body assumptions to "(i)–(iv)" adding the real-analyticity hypothesis explicit in the appendix proof. (Wounds sub-claim 3.)

2. **Remove the literal `\textbf{TODO:}` token from §Theory at line 1880.** Either supply an operationally independent ω measurement (autocorrelation extraction under a stiffness M_μμ that does not coincide with the kinetic metric, per Arnold 1989 GTM 60 Ch. 5 §22–25), or relocate the deferred-empirical-test paragraph out of §Theory into a Limitations section after the implementation chapter. (Wounds sub-claim 6.)

3. **Rewrite the cross-scale shadow at lines 547–552 (§1.4).** The σ²→0 rigid-link identification with ladder-VAE is wrong-domain as cited (blue conceded). Either (a) develop the construction at finite σ² and label the rigid-link as a non-standard structural commitment with its own justification, or (b) re-derive as a deterministic point-passing approximation labeled as such, citing the canonical hierarchical-VI distinction. (Wounds sub-claim 1.)

4. **Resolve the load-bearing companion-paper delegations at lines 1209, 1607, 1615, 1818, 1875.** Either inline the cocycle/vanishing-holonomy proof (1209), the trivial-frame transformer-limit reduction (1607–1615), the multi-head gauge-group identification (1818), and the multi-head/RoPE/FFN derivations (1875); or restate §Theory scope to exclude these reductions from the section's load-bearing content. (Wounds sub-claim 5.)

5. **Address the §1.19 mass-analogy subsection (lines 1871–2069).** Either remove the ω² ∝ m_eff⁻¹ scaling claim and reframe as a pure stiffness-Hessian computation, or supply a separate kinetic-metric postulate that operationally distinguishes ω from m_eff. The line-2064 admission renders the scaling unfalsifiable as currently written. (Wounds sub-claims 1 and 4.)

### Medium consequence — internal-consistency cleanup

6. **Resolve the line-1967 β=0 vs β=β* convention tension.** Commit to the envelope-theorem fixed-β* convention (consistent with the displayed Eqs. eq:mass_mu_diagonal and eq:effective_mass) and remove the β=0 phrasing from the paragraph at line 1967. (Wounds sub-claim 3.)

### Low consequence — cosmetic/hygiene

7. **Tighten the Figure 1 caption at line 1411.** Replace "geometric necessities rather than arbitrary architectural choices" with body-faithful language matching the §1.13/§1.14 hedges ("engineered consensus energy," "consensus-energy ansatz," "substantive structural assumption"). Blue conceded the caption overstates the body. (Wounds sub-claim 4.)

8. **Add the residual-subgroup inline restatement at line 610.** Restate "under the residual constant-per-agent subgroup of Section \ref{sec:dual_role_rigor}" inline in the Convention paragraph at line 610. Blue conceded the hygiene gap. (Wounds sub-claim 4.)

## Granted to the manuscript (no edit needed)

The following constructions were verified canonically correct by both sides and the judges:
- **Sandwich product for covariance transport at line 614** — matches [Nakahara2003 §10.3] (2,0)-tensor action.
- **GL-correct precision transport at line 1894 with O(d) caveat at line 1897** — pre-empts the standard pitfall of substituting Ω^{-T} → Ω outside O(d).
- **Closed-form Gaussian KL at line 1910** — verbatim against [BleiKuckelbirgJordan2017, KingmaWelling2014 App. B].
- **Envelope-theorem treatment at lines 1361–1404** — correctly states the autograd-vs-reduced-free-energy gradient gap as −τ⁻¹Cov_{β*}(E, ∇_x E) per [Milgrom-Segal 2002].
- **Goldstone explicit-vs-spontaneous caveat at line 1518** — matches [Weinberg1995 Vol. II §19, Peskin-Schroeder Ch. 11] in distinguishing Zeeman-type explicit breaking from spontaneous breaking.
- **Notation overload pre-declaration at lines 183–200** — sub-claim 3 (internal consistency) is granted at the notation level; the wound at internal consistency is the conditional-representation theorem naming inconsistency, not the symbol conventions.

## Follow-up debates (if any)

None auto-spawned. The verdict is compound but the operationalization is single (publication-readiness predicate). The revision pass above is the next action, not further debate. If the user wants a focused debate on any single sub-claim post-revision, that can be a separate `/red-blue-debate` invocation.

Possible follow-up debate topics post-revision:
- **Cross-scale shadow under finite σ²**: whether the manuscript can establish the cross-scale shadow as a standard hierarchical-VI construction at finite σ², or whether it must be labeled as a non-standard structural commitment.
- **Mass-analogy revised framing**: whether a reframed §1.19 as pure stiffness-Hessian computation (without the ω² ∝ m_eff⁻¹ scaling claim) is publication-ready on its own merits.
- **Conditional Representation Theorem in Appendix A**: whether the appendix theorem, under its four-assumption (i)–(iv) form, actually closes the f-divergence uniqueness question the body invokes.

## Debate artifacts

All durable in `docs/debates/2026-05-21-pifb-theory-rock-solid/`:

- `00_claim.md` (claim + operationalization)
- `01_evidence.md` (shared evidence pack)
- `01b_extended_evidence.md` (canon harvested by experts across rounds)
- `02_red_opening.md`, `02_blue_opening.md` (Phase 2)
- `02_red_panel_choice.md`, `02_blue_panel_choice.md` (panel logs)
- `02_red_memo_*.md` × 5, `02_blue_memo_*.md` × 5 (per-lens memos)
- `02_canoncop_red.md`, `02_canoncop_blue.md` (Phase 2.5 — 0 strikes each)
- `03_red_rebuttal.md`, `03_blue_rebuttal.md` (Phase 3)
- `03_red_memo_*.md` × 5, `03_blue_memo_*.md` × 5
- `03_canoncop_red.md`, `03_canoncop_blue.md` (Phase 3.5 — 0 strikes each)
- `03b_red_surrebuttal.md`, `03b_blue_surrebuttal.md` (Phase 3b)
- `03b_canoncop_red.md`, `03b_canoncop_blue.md` (Phase 3b.5 — 0 strikes each)
- `04_verdict_canon.md`, `04_verdict_code.md`, `04_verdict_scope.md` (first-pass verdicts — 3× RED_WINS)
- `04_verdict.md` (binding chief verdict — RED_WINS, Rule 3 majority)
- `05_action.md` (this file)
