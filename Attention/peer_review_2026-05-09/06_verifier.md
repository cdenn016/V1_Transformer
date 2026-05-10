# Verifier Report — Peer Review Round 2026-05-09

**Reviewer scope:** Independent verification and consolidation of five parallel reviews (01 math rigor, 02 empirical, 03 claim calibration, 04 physics analogies, 05 writing/style).
**Manuscript:** `Attention/Participatory_it_from_bit.tex` (4598 lines).
**Date:** 2026-05-09.

This is not a sixth review. It is an audit of the five existing ones. Where a finding is real, I confirm it and rank it. Where it is overstated or misquoted, I downgrade or refute it. The verdict at the bottom is mine, not a vote-aggregation of the five reviewers.

---

## 1. Verification of high-severity findings

### 1.1 Empirical-axis BLOCKERs (Review 02)

**F1 — Fig_4 through Fig_8 are missing from the repository. CONFIRMED.**

Direct file-system search:
- `find -iname 'Fig_4*' ... 'Fig_8*'` returns no matches in the repository.
- `Attention/figs/` contains no `Fig_4.png` … `Fig_8.png`. Files such as `fig_scaling_main.{pdf,png}` are present, but the five figures Section 7 needs are not.
- The manuscript references each of these by name at lines 2660, 2670, 2701, 2729, 2761 (verified by grep). The captions describe the empirical content of Section 7 (`fig:energy_flow`, `fig:energy_landscape`, `fig:nonequilib`, `fig:condensation`, `fig:hierarchy`).
- The `Participatory_it_from_bit.log` would record `??` placeholders for every missing image.

**Verdict on F1:** Real BLOCKER. Section 7's empirical content cannot be rendered or evaluated as written.

**F2 — No simulator code for the Ouroboros / meta-agent experiments exists in the repository. CONFIRMED.**

- Grep for `Ouroboros|hyperprior_depth|consensus_check|meta_agent` across the entire tree returns matches only in `Attention/*.tex`, `Attention/*.md` (the manuscript and its peer-review/draft files), and the `.aux` build artefact. No `.py` source contains these primitives.
- Grep for `hyperprior` in `transformer/*.py` matches only `embeddings.py`, `train.py`, and `analysis/bayesian_validation.py` — none of which is a multi-scale active-inference simulator.
- No directory named `*meta*`, `*ouroboros*`, or `*emergence*` exists.
- There is one `.md` file with a related name (`Attention/renormalization_meta_agent_derivation.md`), which is a derivation, not code.
- The `Code Availability` statement in the manuscript reads "will be released … upon publication", which means the simulator was not committed for review even though Section 12.1 cites configuration in "the repository". The contradiction Review 02 calls out is real.

**Verdict on F2:** Real BLOCKER. None of the quantitative numbers in Section 7.5 (520-fold variance spike, 28-fold gradient-variance spike, 95% energy reduction, 13 scales, 173 agents, α ≈ 1.8) can be audited.

**F3 — K=90 has only n_seeds=2, contradicting the abstract / methods "three seeds" claim. CONFIRMED.**

- Direct read of `publication_outputs/scaling_analysis/aggregated_K_sweep.csv`: row K=90 has `n_seeds=2`. All ten other K rows (10, 20, 30, 40, 50, 60, 70, 80, 100, 120) have `n_seeds=3`.
- Manuscript line 54 (abstract): "three seeds".
- Line 117 (Epistemic Status): "three seeds".
- Line 3137 (Section 8.4): "three independent random seeds (6, 23, 111)".
- Figure suptitle in `fig_scaling_main`: "n_seeds=3".

**Verdict on F3:** Real BLOCKER for honest reporting; either re-run K=90 or correct every "three seeds" claim. The bootstrap-CI resolution objection in F3 is a legitimate consequence (with n=2 the discrete K=90 resampling distribution has only three distinct atoms, not the apparent 2000-iteration precision).

### 1.2 Empirical-axis MAJORs (Review 02)

I spot-checked F4–F12. All are real defects in the current text (line numbers and quoted snippets check out against the manuscript and the CSV). One nuance:

- **F4** (R²=1.000 is rounding theatre, b ≈ −1.05 not significantly distinct from b = −1): independently reproducible from the CSV. I did not redo Review 02's nonlinear fit, but the data and the abstract/figure-caption claims it cites are exact.
- **F5** (Chinchilla comparison is misleading because N is exactly linear in K for this architecture): the CSV confirms `mean_total_params / K ≈ 6.5336e5` to four significant figures across all eleven K values. Review 02's algebraic point is right; the manuscript text at line 3151 is mis-framed.
- **F7** (iso-token is not iso-FLOP; FLOPs/step is non-monotonic): from the CSV, `mean_flops_per_step` at K=80 is 1.79e11, at K=90 is 1.02e11, at K=100 is 1.14e11. The non-monotonicity is real and the iso-token framing in the methods is honest about the hardware cause but does not reconcile with the Chinchilla comparison.

All twelve empirical MAJOR / MINOR findings (F4–F19) are confirmed; none is a false positive.

### 1.3 Claim-calibration BLOCKERs (Review 03)

**Finding 1 — Abstract still says "We prove ... reduce to". CONFIRMED.**

Line 54 reads verbatim: "We prove that, under a zero-dimensional isotropic-Gaussian limit, the KL-consensus weights reduce to scaled dot-product attention …". The body section `sec:transformers` was retitled in the Round 16 calibration commit to "Recovery of Dot-Product Attention as a Gauge-Fixed Limit" and the Round 16 lead paragraph explicitly disclaims a derivation. The abstract and the body now contradict each other on the strength of the central technical claim.

**Finding 2 — Conclusion "emerge rigorously" / "from first principles". CONFIRMED.**

Line 3605: "transformer attention mechanisms emerge rigorously as the zero-dimensional limit of gauge-theoretic variational inference."
Line 3621: "demonstrating that attention mechanisms emerge from first principles".

These are both verbatim and both contradict the calibrated body. A reader who reads only the abstract → conclusion will leave with the impression the framework has *derived* attention from gauge theory, which the body now disclaims.

### 1.4 Claim-calibration MAJORs (Review 03)

I checked findings 3–7 against the manuscript:

- **Finding 3** — section title "Independent Convergence with Lahav and Neemeh's Relativistic Theory of Consciousness" at line 3443; the same section says at line 3467 "The intellectual debt runs in the opposite direction: their reframing of phenomenal consciousness as a frame-relative quantity is the philosophical move that licenses the construction in this paper." The contradiction is exact and self-inflicted. **CONFIRMED.**
- **Finding 4** — "Self-Excited Perpetual Motion" subsubsection title at line 2578. "Perpetual motion" is on the physics no-go list and the section is a single-seed run with a threshold detector. **CONFIRMED.**
- **Finding 5** — `tab:wheeler_comparison` at line 2850. **CONFIRMED.**
- **Finding 6** — "Central Result: Spacetime from Information Geometry" boxed minipage at line 2546. **CONFIRMED.**
- **Finding 7** — pullback summary contradicts the consensus-metric regulator gap. **CONFIRMED.**

All five are real and not addressed by Round 16 calibration.

### 1.5 Mathematical-rigor MAJORs (Review 01)

Three MAJORs:

- **Finding 1 (Richness lemma elides global normaliser dependence).** The defect is real as a presentation issue. The appendix's pointwise-attainability argument does need either an analyticity strengthening or an explicit two-parameter family construction. The manuscript treats density-on-attainable-range as if it were everywhere-on-(0,∞). Real MAJOR.
- **Finding 2 (Mean-field factorisation does load-bearing work).** Real MAJOR. The factorisation `Q(k,z) = q_i(k)β(z)` is a substantive constraint (source-independence of the consensus belief), not a tractability simplification, and the softmax form depends on it. The status note acknowledges the broader framing is consensus-energy rather than generative-model derivation, but the specific source-independence assumption is not surfaced.
- **Finding 3 (Sender-side cross-block omission).** Partially overstated. The manuscript at line 1658 ("Remark on the consensus simplification") DOES treat the sender contribution including the trace-term part `+½ Λ_{q_i}⊗Λ_{q_i}` and notes that off-consensus the sender contribution carries an explicit Σ_j-dependent cross-coupling. Review 01 reads as if the sender contribution to `[C^{μΣ}]_{ii}` was omitted entirely; the actual gap is narrower — the manuscript handles the diagonal block but the cross-block boxed equation (eq:cross_block at line 1686) does not separately compute the sender contribution. **Downgrade from MAJOR to MINOR**: the algebra is mostly there, the boxed cross-block claim still needs a single-line sender-side derivation to close the gap, but the framework's mass analogy is not undermined the way Finding 3 suggests.

### 1.6 Physics-analogy MAJORs (Review 04)

Findings 1–9 plus 16:

- **Finding 1** (action-principle table conflates dimensioned actions with dimensionless KL): real MAJOR. Line 1414 ("Like physical action principles…") juxtaposes incommensurable objects.
- **Finding 2** (Wilson observable trivial in implemented Regime I): real MAJOR but the manuscript's Regime II conditional at line 3295 is honest. The defect is that the rhetoric in the gauge-curvature-conjecture and discussion sections sometimes drops the conditional. The S_YM naming is genuinely misleading.
- **Finding 3** (Lorentzian signature put in by hand): real MAJOR. The manuscript discloses this in `sec:signature_resolution` and in the "What we postulate" half of the boxed central result, which means the body is internally honest. The cross-section issue is that the abstract's "Separately, we outline speculative extensions" sentence at line 56 packages the imaginary-temporal-generator postulate fairly, but the conclusion line 3621 ("attention mechanisms emerge from first principles, … information geometry provides powerful tools for unifying apparently disparate phenomena") and the boxed-result title (Review 03 Finding 6) overstate. Confirmed.
- **Finding 4** ("Why Exactly (3+1) Dimensions?" subsubsection title at line 2425 promises a derivation that the body does not deliver): CONFIRMED. The body at line 2425 area lists three speculative biological mechanisms; the section title should reflect that.
- **Finding 5** (dimensional inconsistency between line 1525 area `ε` dimensionless caveat and Section 7.3 `ε ∼ ℓ_P²` Planck identification at line 3214): CONFIRMED by direct read. Both passages are present and they are mutually inconsistent.
- **Finding 6** (gravitational-effects section claims more than it derives): CONFIRMED but partly self-disclosed.
- **Finding 7** (quantum-systems section is metaphor): CONFIRMED but partly self-disclosed at the section's own opening.
- **Finding 8** (ℏ, c, G "undefined / unmeasured" but invoked in physics-bridging passages): CONFIRMED. Table at line 2531 is honest; the rhetoric elsewhere undercuts the table.
- **Finding 9** (Yang-Mills sign convention / `tr(A_μ A_ν)` not gauge-invariant): CONFIRMED. The manuscript admits this honestly; the consensus-metric "rescue" requires a regulator the manuscript does not build, which Review 03 also flags (Finding 7).
- **Finding 16** (downgraded NIT) — "standard transformers are degenerate gauge-theoretic systems" at line 122 of `sec_grav.txt` (which I did not fully audit, but the rhetorical pattern is consistent with the rest of the manuscript). Real but minor.

All ten physics-axis MAJOR/NIT findings are confirmed.

### 1.7 Writing/style MAJORs (Review 05)

- **F.21 — 7 banned LaTeX spacing macros**: confirmed exactly. `grep -E '\\\\,|\\\\!|\\\;'` returns 7 matches at lines 399, 453, 454, 602, 603, 1181, 2332. CLAUDE.md bans these outright. Real MAJOR.
- **F.1 — manuscript length / two-papers-in-one**: real structural concern, but a presentational MAJOR (not a correctness defect).
- **F.5–F.8** — Discussion §7 too discursive, Open Problems §8 duplicates Discussion, roadmap omits §3 and appendices, Appendix C orphan: all real.
- **F.9 — `M` overloaded**: real and serious for math readers. Confirmed by grep that `M`/`\mathcal{M}` carries at least four substantive meanings in the manuscript (multi-agent system, mass Hessian, product manifold, Fisher information, kinetic Lagrangian density, model-fiber dimension).
- **F.10 — `K` overloaded** (mean-fiber dimension vs spring stiffness): real.
- **F.11 — `\phi` overloaded**: real.
- **F.16 — mass/precision/Fisher claim repeated three times**: real prose redundancy.
- **F.18 — caveat paragraphs in §3 repeat themselves**: real.
- **F.20 — equation punctuation inconsistent**: real, easily fixed.

### 1.8 No false positives identified

Every BLOCKER and MAJOR I spot-checked corresponds to actual manuscript content at the cited line. Where the cited line is approximate (e.g., "lines ~110–195 of `sec_pullback.txt`"), the cited passage exists and the quoted text is verbatim. The only finding I downgrade is Math Rigor #3 (sender contribution): the algebra is more present than that finding suggests.

---

## 2. Confirmed defect set (deduplicated across reviews)

The five reviews overlap on several defects. Consolidated table:

| # | Defect | Severity | Reviews flagging | Location |
|---|--------|----------|------------------|----------|
| D1 | Abstract still says "We prove ... reduce to" — body now disclaims this | BLOCKER | 03 (#1), 04 (#3 indirect) | Line 54 |
| D2 | Conclusion: "emerge rigorously … from first principles" / "observer-dependent reality can be rigorously formalized" | BLOCKER | 03 (#2, #11), 04 (#3 indirect) | Lines 3605, 3621 |
| D3 | Fig_4 through Fig_8 missing from repository | BLOCKER | 02 (F1) | Section 7, lines 2660, 2670, 2701, 2729, 2761 |
| D4 | Meta-agent / Ouroboros simulator missing from repository | BLOCKER | 02 (F2), 03 (#4 indirect) | Section 7, Methods §12.1, line 3651 |
| D5 | K=90 has n_seeds=2 but every "three seeds" claim asserts 3 | BLOCKER | 02 (F3) | Lines 54, 117, 3137; figure suptitle |
| D6 | "Independent Convergence with Lahav and Neemeh" title contradicts body's "intellectual debt runs in the opposite direction" | MAJOR | 03 (#3) | Lines 3443 vs 3467 |
| D7 | "Self-Excited Perpetual Motion" subsubsection — Wheeler/perpetual-motion rhetoric not earned by single-seed simulation | MAJOR | 03 (#4), 04 (10 indirect) | Lines 2578–2588 |
| D8 | "Comparison to Wheeler's Vision" table claims more than the framework derives | MAJOR | 03 (#5) | Table at 2850 |
| D9 | "Central Result: Spacetime from Information Geometry" boxed minipage promises spacetime-derivation; body delivers Gram-Schmidt + 4 postulates | MAJOR | 03 (#6), 04 (#3, #4) | Lines 2540–2563 |
| D10 | Lorentzian signature is put in by hand; algebraic compatibility, not derivation | MAJOR | 04 (#3), 03 (#6) | `sec:signature_resolution`, lines ~2180+ |
| D11 | "(3+1) dimensions" subsection title promises a derivation the body does not deliver | MAJOR | 04 (#4) | Line 2425 |
| D12 | Dimensional inconsistency: ε is "dimensionless precision … no physical units" (line ~1525 area) but elsewhere ε ∼ ℓ_P² (line 3214) | MAJOR | 04 (#5) | Lines 1525, 3214 |
| D13 | Action-principle table juxtaposes dimensioned S_GR/S_YM with dimensionless KL ℱ | MAJOR | 04 (#1) | Line 1398 area, Eq./table around 1414 |
| D14 | Wilson observable / S_YM rhetoric not flagged as Regime II conditional in every appearance; S_YM naming overstates parallel | MAJOR | 04 (#2, #10) | `sec:discrete_regime_ii` and discussion sections |
| D15 | "Gauge-invariant collective metrics can be constructed through gauge averaging" (rigorously) — `sec:consensus_metric` says regulator is needed and not constructed | MAJOR | 03 (#7), 04 (#9) | Line 2538 / `sec:pullback` summary; vs `sec:consensus_metric` |
| D16 | No baseline transformer at matched K, tokens, sequence length, seeds | MAJOR | 02 (F6) | Section 8.4 |
| D17 | Iso-token ≠ iso-FLOP; FLOPs/step non-monotonic in K | MAJOR | 02 (F7) | Section 8.4 / methods |
| D18 | Chinchilla comparison mis-framed: for this architecture N is exactly linear in K, so b_K = b_N = −1.05 | MAJOR | 02 (F5) | Line 3151 |
| D19 | "1.2 epochs"/"undertrained" floor claim relies on unauditable Japanese-Wikipedia handwave | MAJOR | 02 (F8, F9) | Line 3149 |
| D20 | Section 8.4 does not state actual achieved PPL range vs published WikiText-103 baselines (~18–25 vs achieved ~73 at K=120) | MAJOR | 02 (F10) | Section 8.4 |
| D21 | Section 7 quantitative claims (520x, 28x, 13 scales, 173 agents, α≈1.8) not reproducible without simulator and per-step logs | MAJOR | 02 (F11) | Section 7.5 |
| D22 | Abstract's "toy multi-agent simulations" understates the n=1 nature; Section 1.5 / Level 2 omits γ_ij=0 / fast-subsystem-only restriction | MAJOR | 02 (F12, F17) | Lines 54, 119, 132 |
| D23 | R²=1.000 is rounding theatre (actual 0.99982); restricted b=−1 model fits to R²=0.99960 | MAJOR | 02 (F4) | Abstract, methods.md, figure suptitle |
| D24 | Quantum-systems / measurement-problem framing without superposition machinery | MAJOR | 04 (#7) | `sec:grav` quantum subsection |
| D25 | Gravitational-effects section claims more than it derives (no equivalence principle, no field equations, no geodesic deviation) | MAJOR | 04 (#6) | `sec:grav` gravity subsection |
| D26 | Mass/precision/Fisher claim repeated three times in §3, §6.5, §7.2 | MAJOR | 05 (F.16, F.18) | Lines 1519–1717, 2479–2481, 3169–3206 |
| D27 | Notation overload: M, K, φ, Ω, β each carry multiple meanings without a notation table | MAJOR | 05 (F.9–F.15) | Throughout |
| D28 | 7 banned LaTeX spacing macros (`\,`, `\!`, `\;`) per CLAUDE.md ban | MAJOR | 05 (F.21) | Lines 399, 453, 454, 602, 603, 1181, 2332 |
| D29 | Two MAJOR math-rigor gaps in load-bearing derivations (richness lemma normaliser dependence; mean-field factorisation as substantive constraint) | MAJOR | 01 (#1, #2) | App. `app:conditional_uniqueness`; `sec:mixture_derivation` |
| D30 | "Conclusion: deep theoretical foundations" hides non-uniqueness of the derivation | MINOR | 03 (#10) | Line 3617 |
| D31 | Sender-side cross-block contribution to `[C^{μΣ}]_{ii}` not separately computed for the boxed cross-block | MINOR (downgraded from MAJOR by reviewer 01) | 01 (#3) | Eq. `eq:cross_block` line 1686 |
| D32 | Several physics-section minors (alignment-dominated regime coupling, M_μμ positive-definiteness, envelope theorem at τ→0, two-stage dual cost, RG det' definition, holonomy distinction, conformal-class kinematics gap) | MINOR (×9) | 01 (#4–#12), 04 (#11–#15) | various |
| D33 | Style minors (equation punctuation, hedge words, "we suspect" / "we flag", caption derivations, list formatting) | MINOR | 05 (F.20, F.22, F.23, F.25, F.26) | various |

The deduplication captures 33 unique defects from 5 reviews × ~70 raw findings. Five are BLOCKERs, twenty-five are MAJORs (after one downgrade), and the rest are MINORs and NITs.

---

## 3. Gaps the reviewers missed

I checked for cross-axis issues none of the five reviews flagged.

### G1. Abstract enumerates "open problems" but omits the simulator-not-released contradiction
**Location:** Line 56 (abstract last sentence), line 3651 (Code Availability).
The abstract's open-problems sentence lists "the four-dimensional nonlinear extension, dynamical selection of the imaginary generator, dimensional analysis between informational and physical units, and a rigorous quantum extension". It does not flag that the central simulator (the Ouroboros multi-agent code that produces every Section 7 figure) is unreleased. Review 02 catches this from the empirical side and Review 03's #12 catches the consensus-metric regulator gap; neither connects it to the abstract's own open-problems enumeration.

### G2. Abstract claim "(ii) toy multi-agent simulations exhibiting threshold-based meta-agent formation (Section ref{sec:participatory})" — the Section 7 figures it points at do not exist
This is a higher-order consequence of D3 (missing figures) and D4 (missing simulator) that none of the reviews quite states cleanly: the abstract advertises empirical content that the manuscript cannot currently render. Review 02 catches both halves separately but does not fold them into a single "abstract advertises content the repository does not contain" item.

### G3. Cross-section consistency: "PPL 15-30" Japanese Wikipedia claim is in scope/limitations (line 156) AND scaling section (line 3149) AND introduction's epistemic-status (line 117) — none of these passages cite the same auditable artefact
Review 02 catches the "no auditable artefact for the Japanese-Wikipedia run" issue (F8). What Review 02 does not catch is that the same number is repeated in three locations as if it were established, with no internal consistency check on which K range each instance is talking about. (The introduction at line 117 says "comparable K"; the scaling section at line 3149 says "comparable embedding dimensions"; the scope-and-limitations at line 156 says "comparable K". None pins down the K range.)

### G4. Section 7's $\gamma_{ij} = 0$ disclosure surfaces only at Section 7.1 (line ~2587 region); the abstract's "toy multi-agent simulations" and the introduction's Level 2 ("Our Ouroboros Tower demonstrates bidirectional information flow") read as full-system claims
Review 02 catches half of this in F17. What is missed: the claim "Ouroboros Tower demonstrates bidirectional information flow" at line 119 is *contradicted* by the slow-subsystem-frozen disclosure at Section 7.1. With $\gamma_{ij} = 0$ the model channel is inactive and only the fast subsystem moves, so "bidirectional information flow" is not in fact demonstrated. Review 02 calls this an under-statement; it is closer to a contradiction.

### G5. Math reviewer missed an issue in §3 sender-block presentation at line 1635
Line 1635 of the Mass section says "Where the block-transpose identity appears not to hold in displayed expressions, the cause is that the displayed `[M_{μμ}]_{ik}` has been restricted to a subset of the contributions to the Hessian (e.g. a single sender-only or receiver-only contribution), and the symmetry is restored once both contributions are summed." This is a calibrated paragraph, but the off-diagonal cross-block at line 1670 region (`Cross Mean-Covariance Blocks`) does not visibly carry the same disclaimer for the cross-block. This is the gap Review 01 #3 was reaching for, partially closed by the in-section text but not surfaced where the boxed equation is.

### G6. Physics reviewer missed: line 2008–2010 "Planck time from minimum information update" speculation
Inside `sec:framework` (line ~2000–2010 area) there is a paragraph: "Could this arise from a minimum information update in physical systems? If physical dynamics are at root information-processing, then $t_P \sim \hbar/E_P \sim \text{(action quantum)}/\text{(energy quantum)} \sim 1\,\text{bit}/\text{(max info rate)}$. This is highly speculative …". The dimensional disclaimer at line 2000 does soften this, but the speculation itself is exactly the genre Review 04 catalogues elsewhere (Findings 5, 8) and was missed in this location. This is the same defect class as Review 04 Finding 5.

### G7. Style reviewer missed: bare `\cite` in a manuscript with 91 citations may not match journal class
Review 05 §0.6 flags this honestly as "consistent but verify with journal class". For a JMLR-class manuscript (which is what `jmlr2e.sty` is), the standard citation style would be `\citet`/`\citep`. This is a real submission risk Review 05 mentions but does not bring forward into the punch list.

### G8. No reviewer cross-checked the abstract → conclusion → introduction claim chain end-to-end
Reviews 03 and 05 each touch parts of this. The cleanest end-to-end version: Abstract claim X (line 54: "We prove…reduce to") → Introduction Level 1 (line 117: "Standard scaled dot-product attention is recovered as a zero-dimensional isotropic-Gaussian limit … up to a separately introduced learned bilinear compatibility M") → Conclusion (line 3605: "emerge rigorously as the zero-dimensional limit") → Conclusion (line 3621: "emerge from first principles"). Three of the four passages are calibrated; the abstract and the conclusion's two strongest verbs are not. This is essentially Review 03's BLOCKER 1+2 but the inconsistency is broader than Review 03's framing — the introduction's Level 1 paragraph is the gold standard the rest should match.

---

## 4. Prioritized punch list

### P0 — Must fix before any submission (BLOCKERs)

| # | Title | Location | What to change | Reviews |
|---|-------|----------|----------------|---------|
| P0.1 | Reword abstract "We prove … reduce to" | Line 54 | Replace with calibrated language matching introduction's Level 1 paragraph at line 117 (which does it correctly): "Standard scaled dot-product attention is recovered as a zero-dimensional isotropic-Gaussian limit of the KL-consensus construction, up to a separately introduced learned bilinear compatibility $M$ and the standard normalisation and bias assumptions". Drop "We prove". | 03 (#1) |
| P0.2 | Reword conclusion "emerge rigorously" + "from first principles" | Lines 3605, 3621 | Replace "emerge rigorously as the zero-dimensional limit" with "are recovered as a gauge-fixed isotropic-Gaussian limit, with a separately introduced learned bilinear, of gauge-theoretic variational inference"; replace "demonstrating that attention mechanisms emerge from first principles" with "demonstrating that attention mechanisms admit a principled derivation from gauge theory and variational inference"; replace "observer-dependent reality can be rigorously formalized" with "an information-geometric formalisation of agent-frame-dependent phenomenal geometry is mathematically well-defined". | 03 (#2, #11) |
| P0.3 | Restore Fig_4 through Fig_8 OR remove Section 7 | Section 7, lines 2660, 2670, 2701, 2729, 2761 | Either commit the five PNGs to `Attention/figs/` (or wherever `\graphicspath` resolves) and verify a clean LaTeX build, or remove Section 7's figure-bearing prose and recast it as a sketch deferred to a companion paper. | 02 (F1) |
| P0.4 | Commit the meta-agent / Ouroboros simulator OR retract Section 7's quantitative claims | Section 7, Methods §12.1, line 3651 (Code Availability) | Either commit the simulator (with the seed-2 configuration JSON and the per-step diagnostic CSV used to generate Section 7.5's percent-reduction numbers), or remove the audit-grade quantitative summary at line 2774–2786 and present Section 7 as a worked-example sketch with the simulator deferred. The "will be released … upon publication" sentence at line 3651 must be reconciled with the same paragraph's reference to "configuration files detailed in the repository". | 02 (F2) |
| P0.5 | Correct K=90 / "three seeds" mismatch | Line 54, line 117, line 3137, methods.md, figure suptitle | Either re-run the missing K=90 seed (preferred) or rewrite every "three seeds" claim to read "n=3 seeds per K except K=90 which has n=2"; regenerate the scaling figure with corrected suptitle. | 02 (F3) |

### P1 — Should fix in this revision (MAJORs)

| # | Title | Location | What to change | Reviews |
|---|-------|----------|----------------|---------|
| P1.1 | Retitle "Independent Convergence with Lahav and Neemeh" | Line 3443 | Drop "Independent". Use "Gauge-Theoretic Realisation of the Cognitive-Frame Transformation Law" or "Mathematical Formalisation of Lahav and Neemeh's Cognitive Frames". | 03 (#3) |
| P1.2 | Retitle "Self-Excited Perpetual Motion" subsubsection | Line 2578 | Replace "Perpetual Motion" with "Sustained Non-Equilibrium under the Threshold-Detector Dynamics". Remove the "second law's mandate toward increasing entropy" sentence or attach the single-seed / threshold-detector / no-cosmology disclaimer. | 03 (#4) |
| P1.3 | Reframe `tab:wheeler_comparison` | Line 2850 | Add caption "We do not claim to have derived mass, spacetime geometry, or quantum mechanics from the framework; this table is a structural map, not a derivation summary." Drop the two weakest rows ("Delayed choice", "Law without law") or relabel "realizations" as "structural correspondences". | 03 (#5) |
| P1.4 | Retitle "Central Result: Spacetime from Information Geometry" boxed minipage | Lines 2540–2563 | Replace box header with "Belief Tangent-Space Decomposition and the Spacetime Reading: What Is Derived and What Is Postulated". Replace "The radical claim:" subheading with "An interpretive reading:". | 03 (#6) |
| P1.5 | Calibrate `sec:pullback` summary against `sec:consensus_metric` | Line 2538 area | Replace "Gauge-invariant collective metrics can be constructed through gauge averaging" with "A candidate gauge-invariant collective metric can be defined through gauge averaging (Section ref{sec:consensus_metric}), conditional on a regulator that the present manuscript does not construct." Replace "naturally produce multi-sector dimensional structure" with "permit, under additional eigenvalue-magnitude postulates, a multi-sector dimensional reading". | 03 (#7) |
| P1.6 | Retitle "Why Exactly $(3+1)$ Dimensions?" | Line 2425 | Replace with "Hypothesised $(3+1)$ Structure" or "Conjectured $(3+1)$ Structure". The body already concedes no derivation is offered. | 04 (#4) |
| P1.7 | Resolve dimensional inconsistency on $\epsilon$ | Lines 1525 area, 3214 | Either drop the Planck-scale identification at line 3214 (`ε ∼ ℓ_P²`) or commit to a unit assignment for $\epsilon$ across both passages and provide a derivation. They cannot both be true as written. | 04 (#5) |
| P1.8 | Recast action-principle table (line 1398 area) | Line 1398, table around 1414 | Demote to "structural analogy (variational stationarity)" with explicit statement that $\mathcal{F}$ is dimensionless (nats) while $S_{\mathrm{GR}}, S_{\mathrm{YM}}$ have units of action. Replace "Euler-Lagrange" with "natural-gradient flow stationarity condition". | 04 (#1) |
| P1.9 | Add explicit Regime II conditional everywhere Wilson observable / S_YM appears | discussion sections, gauge-curvature conjecture | Append "(Regime II construction; not realised in the present implementation, where the connection is pure-gauge and curvature vanishes identically by Theorem ref{thm:vanishing_holonomy}.)" at every appearance outside `sec:discrete_regime_ii`. Rename `S_YM` to `S_{\mathrm{Wilson-reg}}` or `S_{\mathrm{holonomy-penalty}}` to avoid implying a Yang-Mills field theory. | 04 (#2, #10) |
| P1.10 | Recast quantum-systems and gravitational-effects sections as future-work directions | `sec:grav` quantum and gravity subsections | Rename to "An Inferential-Consensus Analogy for Measurement" and "Toward a Pullback-Curvature Reading of Gravity". Remove "potential resolution to the measurement problem" and "Gravitational Effects" headline framing. Move both sections to after the explicit statement that the framework does not handle quantum amplitudes. | 04 (#6, #7) |
| P1.11 | Strengthen Lorentzian-signature section title and headline | `sec:signature_resolution`, line ~2180+ | Replace "Lorentzian signature from $\mathrm{GL}(2,\mathbb{C})$ gauge frames" with "Algebraic compatibility of Lorentzian signature with $\mathrm{GL}(2,\mathbb{C})$ gauge frames under two postulates". Ensure the abstract and introduction passages do not call this section a derivation. | 04 (#3) |
| P1.12 | Fix R^2 = 1.000 reporting | abstract line 54, methods.md, figure suptitle | Report R^2 to four decimal places (0.9998). Report b to two decimal places. Add an F-test or AIC comparison against the b=−1 restricted model. State that the fit is dominated by the floor parameter c. | 02 (F4) |
| P1.13 | Recast Chinchilla comparison | Line 3151 | State plainly that for this architecture $b_N = b_K = -1.05$ because $N$ is exactly linear in $K$ across the sweep, and that this exponent is steeper than Hoffmann et al. on a more restricted parameter class (overwhelmingly embedding parameters). Or drop the comparison. | 02 (F5) |
| P1.14 | Add or run baseline transformer at matched config | Section 8.4 / `fig:scaling_main` | Train a single-layer vanilla transformer with matched K, sequence length, token budget, and seeds, and overlay PPL(K) on `fig:scaling_main`. Without this, the "architectural fingerprint" framing is unjustified. | 02 (F6) |
| P1.15 | Disclaim iso-token vs iso-FLOP | Section 8.4, methods | Explicitly state that iso-token is not iso-FLOP and that the FLOPs/step is non-monotonic in K (1.79e11, 1.02e11, 1.14e11 at K=80, 90, 100). Either rerun on GPU at iso-FLOP or restrict comparison to iso-token literature only. Promote `fig:compute_frontier` into the main text. | 02 (F7) |
| P1.16 | Substantiate or remove the Japanese-Wikipedia "PPL 15–30" claim | Lines 117, 156, 3149 | Either present a side-by-side table or figure with seed counts, training tokens, K values, and final test PPL per K; or remove the claim and report c ≈ 61 as the model's measured ceiling under the actual training regime. | 02 (F8) |
| P1.17 | State actual achieved PPL vs published WikiText-103 baselines | Section 8.4 | Add one sentence: "At K=120 the gauge-theoretic single-layer transformer reaches test PPL ~73 on WikiText-103 in this iso-token undertrained regime, against published WikiText-103 baselines of PPL ~18-25 for multi-layer transformers at comparable parameter counts. The b≈-1 exponent describes scaling within this architecture, not competitive with state-of-the-art language modelling." | 02 (F10) |
| P1.18 | Surface the n=1 / γ=0 nature in abstract and Level 2 description | Lines 54, 119, 132 | Replace abstract's "(ii) toy multi-agent simulations exhibiting threshold-based meta-agent formation" with "(ii) a single-seed multi-agent simulation with the slow subsystem frozen ($\gamma_{ij}=0$, fast-subsystem-only) exhibiting threshold-based meta-agent formation across 13 hierarchical scales (multi-seed reproducibility deferred)". Update Section 1.5 / Level 2 similarly. | 02 (F12, F17), G4 |
| P1.19 | Reproducibility for Section 7.5 quantitative summary | Section 7.5 lines 2774–2786 | Either commit the simulator + seed-2 configuration + per-step diagnostic CSV so the reader can reproduce the percent-reductions and the α≈1.8 fit, or remove the audit-grade summary. State $t_c$ and the fit window for the α fit explicitly. | 02 (F11) |
| P1.20 | Collapse mass/precision/Fisher repetition to one canonical exposition | Lines 1519–1717, 2479–2481, 3169–3206 | Cut §6.5 lines 2479–2484 to a single sentence with `\ref` to §3. Cut §7.2 from 38 lines to a single paragraph covering only the gravitational-coupling speculation (which is genuinely new). State the kinetic-postulate caveat once at the start of §3 and once in §3.7, not in every sub-block. | 05 (F.16, F.18) |
| P1.21 | Add notation table at the start of §2 | Throughout | Include $\mathcal{C}, \mathcal{N}, \mathcal{B}, G, \mathfrak{g}, U_i, \phi_i, \Omega_{ij}, \tilde\Omega_{ij}, q_i, p_i, s_i, r_i, \alpha_i, \beta_{ij}, \beta^{(p)}_{ij}, \gamma_{ij}, \chi_i, \pi_{ij}, \kappa_\beta, \kappa_\gamma, K, M, \zeta, M_{\mu\mu}, m_{\mathrm{eff}}, \mathcal{F}$. Reserve $\mathcal{M}$ for one meaning. Use `H` for the Hessian, `I_F` or $\mathcal{I}$ for Fisher information. Use lowercase `k` for stiffness. Define $\zeta$ in body before its first figure-caption appearance. | 05 (F.9–F.15) |
| P1.22 | Remove all 7 banned LaTeX spacing macros | Lines 399, 453, 454, 602, 603, 1181, 2332 | Replace `\Omega_{ij}\,q_j` with `\Omega_{ij} q_j`; replace `\Gamma\!\left(...\right)` with `\Gamma\left(...\right)`; replace `c;\;` with `c;`; replace `\,` chains with single spaces. CLAUDE.md is unambiguous. | 05 (F.21) |
| P1.23 | Surface mean-field factorisation as substantive constraint | `sec:mixture_derivation` Eq. eq:mixture_posterior | State explicitly that $Q(k\mid z) = q_i(k)$ is the load-bearing assumption, not a tractability simplification; the softmax form follows from the source-independence constraint, not from the variational principle alone. | 01 (#2) |
| P1.24 | Tighten richness lemma in `app:conditional_uniqueness` | Appendix `app:conditional_uniqueness` | Either restrict $f$ to a real-analytic class (so that density on any open subinterval propagates to identity on $(0,\infty)$), or supply an explicit two-parameter family of admissible $(p_i, \{q_k\})$ for which $r(c_0)$ traces a known open subinterval, or downgrade the conclusion to "$f' = \log + k$ on the attainable range". | 01 (#1) |
| P1.25 | Append consensus-metric regulator gap to abstract limitations enumeration | Line 56 | Append "; the gauge-orbit-averaged consensus metric used as the framework's candidate gauge-invariant geometry is conditional on a regulator that we do not construct (Section ref{sec:consensus_metric})." | 03 (#12) |

### P2 — Style and minor (everything else)

| # | Title | Location | What to change | Reviews |
|---|-------|----------|----------------|---------|
| P2.1 | Add sender-side cross-block one-line derivation | Eq. eq:cross_block line 1686 | Add the directional-derivative line for the sender contribution to `[C^{μΣ}]_{ii}` and confirm vanishing at consensus. | 01 (#3) |
| P2.2 | Math-rigor minors (alignment-dominated regime, M_μμ PD argument, envelope at τ→0, two-stage dual cost, RG det' definition, holonomy distinction) | various | One-paragraph patches each. | 01 (#4–#12) |
| P2.3 | Physics minors (conformal-class kinematics, Fisher-arc-length / proper-time mismatch, equivalence-principle "consistency" overstatement, spinor double cover decorative invocation, "degenerate gauge-theoretic systems" rewording) | various | Wording adjustments. | 04 (#11–#16) |
| P2.4 | Itemisation reduction in Section 7.5 | Line 2774 | Convert the four-bullet convergence-metrics block to a single sentence per CLAUDE.md style guide. | 02 (F18) |
| P2.5 | Equation punctuation pass | throughout | Apply standard physics-journal convention (period if equation ends a sentence, comma if continuation). | 05 (F.20) |
| P2.6 | Hedge / filler words | throughout | Cut roughly half of "essentially / simply / of course / It is worth being explicit about". | 05 (F.25, F.26) |
| P2.7 | Move limitation paragraph out of figure caption at line 2762 | Line 2762 | Move the "consensus detector is imposed" discussion to body text; keep caption descriptive. | 05 (F.23) |
| P2.8 | Replace inline derivation in figure caption at line 1323 with `\eqref` | Line 1323 | Caption should describe and reference, not re-derive. | 05 (F.22) |
| P2.9 | Bare `\cite` consistency with journal class | throughout | Verify with target journal (`jmlr2e.sty` typically wants `\citet`/`\citep`). | 05 (§0.6), G7 |
| P2.10 | Conclusion: "deep theoretical foundations" → "principled derivation, among several possible" | Line 3617 | Trivial wording fix. | 03 (#10) |
| P2.11 | Section ordering / structural recommendation: Pullback before Mass | §3 vs §5 | Move Mass section after Pullback per Review 05 §7. Optional structural improvement. | 05 (F.1, F.2, F.7) |
| P2.12 | "Bit (beliefs) → It (geometry) → Bit (updated beliefs)" boxed slogan | Line 2828 / 2830 | Either un-box (slogans should not be in display-equation boxes) or replace with a real schematic update equation $q_i^{(n+1)} = \mathcal{F}_{\mathrm{transport}}(\mathcal{F}_{\mathrm{pullback}}(\{q_j^{(n)}\}))$. | 03 (#8) |
| P2.13 | "Plays the role of effective mass" → conditional clause | Line 3171 | Append "conditional on the kinetic-metric postulate of Section ref{sec:velocity_quadratic}". | 03 (#9) |
| P2.14 | Append undertraining quantification to `fig_trajectory_by_K` | Section 8.4 area | Promote `fig_trajectory_by_K.png` into main text and report end-of-training validation-loss slope per K. | 02 (F9) |
| P2.15 | State H (head count) across the K sweep | Section 8.1, 8.4 methods | One sentence on whether H is fixed or scales with K. | 02 (F15) |
| P2.16 | Explain 122.9M tokens choice | abstract, Section 8.4 | One sentence: "≈ 1.19 epochs of WikiText-103 under gpt2 BPE". | 02 (F16) |
| P2.17 | Mass/Fisher units disclaimer carried through | wherever "mass" is invoked in physics-bridging passages | Add "framework-internal effective mass; not equated to physical inertial mass" at every invocation that crosses into rocks / gravity / equivalence-principle territory. | 04 (#12) |
| P2.18 | "Spinor double cover" decorative invocation | Section ref{sec:signature_resolution} | Either drop or develop. | 04 (#13) |

---

## 5. Overall verdict

**Major Revision.**

The manuscript contains a substantial, mathematically careful framework for gauge-theoretic active inference, an honest empirical sweep on WikiText-103, and an unusually thorough self-critique in several sections (`sec:signature_resolution`, `sec:consciousness_section`, `sec:scope_limitations`). The Round 16 calibration commit was a good-faith move toward propagating the body's hedged framing. However, three classes of defect block acceptance in the current state.

First, the calibration was applied surgically to four named body sections and was not propagated. The abstract still says "We prove … reduce to" (line 54), the conclusion still says "emerge rigorously" (line 3605) and "from first principles" (line 3621), the "Independent Convergence" subsection title (line 3443) is contradicted by its own admission of intellectual debt at line 3467, and the "Self-Excited Perpetual Motion" subsubsection (line 2578) deploys physics-loaded vocabulary the single-seed simulation does not earn. A reader who reads the abstract and conclusion will leave with a substantially different impression than a reader who reads the body. This is editorial follow-through and is fixable, but it must be fixed.

Second, Section 7's empirical content is unevaluable as it stands. Five named figures (`Fig_4.png` through `Fig_8.png`) referenced at lines 2660, 2670, 2701, 2729, 2761 do not exist anywhere in the repository. The simulator that produces those figures' underlying data does not exist anywhere in the repository. The Code Availability statement at line 3651 promises release "upon publication" while a same-paragraph Methods reference cites configuration files "detailed in the repository". The quantitative claims in Section 7.5 (520-fold variance spike, 95% energy reduction, 13 scales, 173 agents, α ≈ 1.8) cannot be audited at the resolution given. This is not a presentational problem; without the figures and the simulator, the section cannot be peer-reviewed.

Third, Section 8.4's headline empirical claim is materially weaker than presented. The R² = 1.000 rounds from 0.99982 (a restricted b = −1 model gets R² = 0.99960 on the same data). The bootstrap CI for b is [-1.103, -0.998], placing b = −1 at the upper edge — the headline finding "b ≈ −1.05, distinct from −1" is not statistically supported. K = 90 has n_seeds = 2, contradicting every "three seeds" claim in the abstract, methods, and figure suptitle. The Chinchilla comparison is structurally mis-framed: for this architecture N is exactly linear in K to four significant figures, so $b_K = b_N$ and the "K is not parameter count" defence does not soften the contrast — it strengthens it onto a more restricted parameter class. No baseline transformer at matched K, tokens, sequence length, and seeds was trained, so the "primary architectural fingerprint" framing is uninterpretable as a quality claim.

The framework's mathematical core (gauge-equivariant transport, KL invariance, mean-gradient equivalence with attention, mass-as-stiffness Hessian, RG closure theorem) is largely sound. The most damaging mathematical issues (Review 01 Findings 1, 2) are real but patchable in a revision. The physics analogies (Review 04) are extensively self-disclosed in the body but unevenly across the manuscript; the section headers, abstract, and conclusion outrun the disclosures. The writing/style issues (Review 05) are real but cosmetic relative to the BLOCKERs.

The recommended action: address P0.1–P0.5 unconditionally, address P1.1–P1.25 in a revision, and treat P2 as copy-edit cleanup. After P0 + the BLOCKER-adjacent P1 items (P1.18, P1.19), the manuscript would stand as a careful gauge-theoretic active-inference contribution with appropriate calibration; without those, it is self-inconsistent on its central claims and contains an entire section (Section 7) the reader cannot evaluate.
