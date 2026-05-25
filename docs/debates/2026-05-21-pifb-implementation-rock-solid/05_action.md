# Action — pifb-implementation-rock-solid

**From verdict:** RED_WINS (binding, chief reconciliation; 3/3 first-pass judges concurred; blue's sur-rebuttal explicitly tipped the chief verdict).

The claim "§Implementation of `Attention/Participatory_it_from_bit.tex` (lines 2101–2304) is rock solid and publication ready" does **not** survive adversarial pressure-testing. Five of seven conjunctive sub-claims carry verified wounds; the defending side explicitly conceded RED_WINS under the agreed operationalization.

## Decisive findings

1. **Manuscript-vs-simulator consensus-detector form mismatch** (decisive evidence; sub-claim 6):
   - `meta_agents.py:55-66` implements `C = 1 − mean(KL)` — the form the manuscript at line 2174 explicitly rejects ("could give misleadingly positive products from two negative factors").
   - `meta_agents.py:82-91` returns `C_b · C_m` (two factors). Manuscript line 2174 specifies `Γ = P · C_q · C_s` (three factors).
   - `meta_agents.py:343-359` computes the extrinsic Euclidean weighted mean. Manuscript line 2191 claims the Lie-algebra-additive (log-Euclidean) form; the simulator's own in-code comment at lines 344–355 admits the divergence.
2. **IB Lagrangian out of canonical scope** (sub-claim 1): Chechik-Tishby-Globerson-Weiss 2005 *JMLR* 6 §3 Theorem 3.1 (closed-form Gaussian-IB) requires jointly Gaussian random vectors in a common Euclidean space. The parent state $T = (q_I, s_I, U_I)$ at PIFB line 2131 is not such an object.
3. **Cross-scale "transport" misnamed** (sub-claim 4): the simulator computes the frame-change $U_I U_i^{-1}$ at `meta_agents.py:226-227`. This is a frame change, not parallel transport of a connection (Nakahara 2003 §10.3 / Kobayashi-Nomizu 1963 §II.7); no connection 1-form is provided.
4. **Self-disclosed simulator-verification gap** (sub-claim 7): manuscript line 2284 ("Whether the released simulator code realizes the full transport $\Omega_{i,I}$ or a frame-trivial substitute is not independently verified in this manuscript") plus three other deferral markers (lines 2138 IB ingredients, 2197 RG fixed points, 2213 critical phenomena, 2228 single-seed measurement).
5. **Blue's tip to the chief** at `03b_blue_surrebuttal.md:25`: "Verdict tip to the chief: RED_WINS on operationalization-binding."

## Recommended action

Revision pass on `\section{Implementation}` (lines 2101–2304). The ten concrete edits from the chief verdict, ordered by consequence and ease:

### High consequence — fixes load-bearing manuscript-vs-code mismatches

1. **Consensus-detector form (line 2174 ↔ `meta_agents.py:55-91`)** — Either revise the manuscript to specify the simulator's two-factor `1 − mean(KL)` detector (withdraw the on-line argument against `1-KL`; drop the `[0,1]` bound claim; update threshold semantics), or revise the simulator to compute `exp(-V/τ_q)`, `exp(-V/τ_s)`, introduce the presence factor `P = mean(χ_i)`, and return `P · C_q · C_s`, then rerun §Results. Manuscript currently argues *against* the form the simulator implements.

2. **Frame-averaging form (line 2191 ↔ `meta_agents.py:343-359`)** — Either revise line 2191 to specify extrinsic Euclidean weighted mean with downstream regularization (dropping the BCH first-order log-Euclidean claim) or revise the simulator to compute `matrix_exp((w * matrix_log(omega_stack)).sum(dim=0))`, the log-Euclidean intrinsic mean per [Moakher 2002 *SIAM J. Matrix Anal. Appl.* 26(3) §3]. Simulator's in-code comment at lines 344–355 already admits this divergence.

3. **Implementation Note at line 2284 (sub-claim 7)** — Either supply an independent verification of the simulator transport (cite test files and assertion lines in `MAgent_Model-main/gauge_agent/`) or downgrade the §Implementation claims that depend on the verification not being deferred.

### Medium consequence — canonical-fidelity repairs

4. **IB Lagrangian framing (lines 2131–2138)** — Either remove the closed-form Chechik-Tishby attribution (since Theorem 3.1 requires jointly Gaussian random vectors) or reformulate with a non-Gaussian-IB scheme that does apply — e.g., deterministic IB [Strouse-Schwab 2017] or agglomerative IB [Slonim-Tishby 2000]. Retain the "research direction" label only if the three unsupplied ingredients at line 2138 are explicitly named as ingredients of that non-Gaussian variant.

5. **Cross-scale "transport" at line 2247** — Either specify the connection 1-form on a cross-scale principal bundle that licenses the "transport" label for $\Omega_{i,I}$, or relabel the operator throughout §Implementation as "frame-change" or "cross-scale frame re-expression" $U_I U_i^{-1}$.

6. **Dispersion-term truncation at line 2179** — Either retain the full dispersion correction in `eq:meta_agent_sigma_impl` (recovering the canonical m-projection per Amari-Nagaoka 2000 §3.5 / Bishop 2006 §10.7) or quantify the coherence regime within which the truncation is justified, citing [Beal 2003 §2.2.2].

### Lower consequence — scope and citation hygiene

7. **RG language (lines 2197–2213)** — Tighten the RG-inspired disclaimer to also constrain the line-2210 form-invariance claim and the line-2213 "critical phenomena, fixed points, or universal behavior" suggestion. Either restate line 2210 in non-RG language or supply the fixed-point identification Wilson 1971 / Cardy 1996 require.

8. **West-Harrison citation at line 2275** — Revise from West-Harrison §6.3 to §10.7 (the infinite-horizon discounted-likelihoods construction); add a [Smith 1979 *JRSSB* 41:375–387] anchor for the infinite-tower geometric-discount form, or restate the analogy more cautiously.

9. **Emergent-properties subsection at line 2228** — Either disclose the single-seed measurement limitation as an explicit falsifying boundary, or remove the substantive existential claim ("whole becomes qualitatively different from sum of its parts") and retain only the multi-seed measurement plan as future work.

10. **Karcher non-compact GL+(K) at line 2160** — Either engage [Pennec-Fillard-Ayache 2006 §4–§5] affine-invariant SPD literature or [Bonnabel-Sepulchre 2009] on intrinsic means on quotient manifolds and specify the substitute Riemannian structure, or scope the framework explicitly to SO(N) for §Implementation and acknowledge the §Theory line 593 GL+(K) framing is not exercised by §Implementation as written.

## Suggested execution order

The strongest single revision pass is **items 1, 2, and 5 together**: align simulator with the manuscript's prescribed Gibbs three-factor detector and log-Euclidean frame average; relabel "transport" → "frame-change" where it is mere $U_I U_i^{-1}$; rerun §Results and report any change. Item 3 (verification of simulator transport) closes the line-2284 wound by reading actual test files. Items 4, 6, 7, 8, 9, 10 are manuscript-only edits against named primary sources and can be applied in a single editorial pass.

## Granted to the manuscript (no edit needed)

The following constructions were verified by both sides and the judges:
- **Gauge-covariant variational barycenter** at `eq:meta_agent_barycenter` (line 2142) — closed form in eqs. `eq:meta_agent_mu_barycenter`, `eq:meta_agent_sigma_barycenter` is the canonical forward-KL barycenter [Bishop 2006 §10.7].
- **Variational FE-improvement criterion** at `eq:meta_agent_FE_criterion` (line 2123) — the principled rule for meta-agent formation; criticism is that the threshold detector is not a Jensen-bounded surrogate of it, not that the criterion is wrong.
- **Cross-scale information-flow metric** at `eq:cross_scale_information_flow` (line 2219) — the manuscript's caveat at line 2222 ("we avoid writing this quantity as a mutual information") is correct.
- **Lie-algebra additive frame** at line 2191 — formally correct as first-order BCH; the wound is that the simulator does not implement it (item 2 above), not that the formula is wrong.
- **Karcher caveat enumeration** at line 2160 — the two-substitute enumeration is openly disclosed; the wound is the unadjudicated choice plus the fact that the framework's claimed natural gauge group is GL+(K) at §Theory line 593 (item 10 above).

## Follow-up debates (if any)

None auto-spawned. The verdict's revision list is the next action, not further debate.

Possible follow-up debate topics post-revision:
- **Simulator-vs-manuscript audit (full)**: a code-mode debate verifying that all §Implementation displayed equations are realized by the simulator as written, after items 1, 2, 5 above are resolved.
- **RG framing (fixed-point identification)**: a theory-mode debate on whether the framework can supply even a sketch of the β-function and fixed-point computation that line 2213 suggests as future work.
- **IB refinement (non-Gaussian variant)**: a theory-mode debate on whether deterministic IB or agglomerative IB closes the canonical-fidelity gap at lines 2131–2138 once the Gaussian-IB attribution is removed.

## Debate artifacts

All durable in `docs/debates/2026-05-21-pifb-implementation-rock-solid/`:

- `00_claim.md` (claim + operationalization)
- `01_evidence.md` (shared evidence pack)
- `01b_extended_evidence.md` (canon harvested across rounds)
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
