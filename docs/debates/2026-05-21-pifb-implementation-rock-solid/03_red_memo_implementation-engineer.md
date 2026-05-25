# Red Memo — implementation-engineer (rebuttal)

## (i) Concession from blue's opening

Blue's enumeration of the three concrete code-vs-manuscript divergences at `meta_agents.py:56-66` (`1 - E` form), `meta_agents.py:89-91` (two-factor product), and `meta_agents.py:343-359` (extrinsic Euclidean frame mean) is *accurate and admitted in blue's opening*. The simulator docstring at `meta_agents.py:344-355` explicitly states "the previous docstring claim of 'Lie-algebra-additive average' was wrong; corrected here." This is blue's own concession; red carries sub-claim 6 by blue's admission.

## (ii) Strongest attack on blue's core defense

The red-team task brief instructs: "Verify red's evidence-pack discrepancy claim independently. Blue's opening accepted the consensus-detector form mismatch but framed it as a calibration choice." Re-greppping the simulator code confirms red's evidence-pack claim with primary-source backing:

**Verified path:line claims:**

- `MAgent_Model-main/gauge_agent/meta_agents.py:55-66` — `belief_coherence` returns `1.0 - E` where E is the mean post-transport KL (verified by direct read: line 62 `E = system.pairwise_alignment_energies('belief')`; line 66 `return 1.0 - E`). This is the `1 - KL` form.
- `MAgent_Model-main/gauge_agent/meta_agents.py:68-80` — `model_coherence` returns `1.0 - E` analogously (line 80: `return 1.0 - E`). Same `1 - KL` form.
- `MAgent_Model-main/gauge_agent/meta_agents.py:82-91` — `consensus_score` returns `C_b * C_m` (line 91: `return C_b * C_m`). Two factors, no presence factor $P$.
- `MAgent_Model-main/gauge_agent/meta_agents.py:93-129` — `find_clusters` thresholds on `gamma = self.consensus_score(system)` (line 103), then `adj = (gamma > self.gamma_min).float()` (line 106). No spatial-overlap gating; presence factor $P$ from manuscript line 2174 is not applied at the threshold step.
- `MAgent_Model-main/gauge_agent/meta_agents.py:343-359` — `omega_avg = (w_q_b_om * omega_stack).sum(dim=0)` (line 358) — extrinsic Euclidean weighted mean of group elements, *not* the Lie-algebra-additive form $\phi_I = \sum_i w_i \phi_i$ specified at PIFB line 2191.
- `MAgent_Model-main/gauge_agent/meta_agents.py:344-355` — comment block explicitly admits the divergence: "a true intrinsic mean would be `matrix_exp(Σ_j w_j · matrix_log(omega_j))` per manuscript line 1911. The extrinsic form is kept... The previous docstring claim of 'Lie-algebra-additive average' was wrong; corrected here."

Red's evidence-pack `path:line` claims are independently verified and stand. CLAUDE.md policy: "CODE FOCUS — when investigating and/or auditing the codebase do NOT rely on code comments...focus on the actual code and paths." The actual code at `meta_agents.py:91` returns `C_b * C_m` (two factors) and the actual code at `meta_agents.py:358` returns the extrinsic mean — both code paths are at variance with manuscript prose. Code-truth judge will weight `path:line` references with verified reachability at 3×.

Blue's framing — "concession is honest, not capitulation" (blue opening, line 48) — does not escape the operationalization. Sub-claim 6 reads "Where §Implementation states what the simulator implements [...] the simulator at `MAgent_Model-main/gauge_agent/` actually realizes those constructions — not a frame-trivial substitute." Three code paths fail this test on blue's own admission and red's independent verification.

## (iii) Strongest defense against blue's strongest attack

Blue's strongest attack on red's implementation finding would be: *the bottom-up aggregation step at `meta_agents.py:167-399` is structurally faithful to PIFB Eqs. 2181-2191 (gauge-covariant sandwich transport via `transport_covariance` at line 230, $\chi$-weighted fixed-point at lines 290-321), so a partial match is enough to discharge sub-claim 6.*

Red's counter-defense: sub-claim 6 is not a partial-credit predicate. The operationalization says "Where §Implementation states what the simulator implements [...] the simulator at `MAgent_Model-main/gauge_agent/` actually realizes those constructions" — for *every* such statement, not for some of them. The threshold detector (lines 2168-2174) is one of the three load-bearing operative implementations stated in §Implementation (the variational FE criterion, the gauge-covariant barycenter, and the threshold detector). The threshold detector is *the gating mechanism* that decides which clusters become meta-agents — without it, the bottom-up aggregation never fires. A faithful aggregation step gated by an unfaithful detector implements something the manuscript at line 2174 explicitly *argues against*. Sub-claim 6 falsified at the gating step.

## Newly-discovered canon

- **`MAgent_Model-main/gauge_agent/meta_agents.py:344-355` (in-repo)** — code comment admitting the divergence: "a true intrinsic mean would be `matrix_exp(Σ_j w_j · matrix_log(omega_j))` per manuscript line 1911. The extrinsic form is kept because (a) the downstream consumers always invert via `safe_inv` / `robust_cholesky` so a near-singular average degrades gracefully, and (b) switching averaging semantics would shift downstream trajectories and is gated on a separate manuscript-alignment authorisation. The previous docstring claim of 'Lie-algebra-additive average' was wrong; corrected here." This is the *code itself admitting* the manuscript-vs-implementation divergence — primary-source admission.
- **CLAUDE.md "CODE FOCUS" policy** — "when investigating and/or auditing the codebase do NOT rely on code comments...focus on the actual code and paths." Cited here as the user's own policy governing weighting in code-vs-comment disputes, not as canonical authority. The code at `meta_agents.py:91` and `meta_agents.py:358` is canonical for sub-claim 6.
