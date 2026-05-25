# Memo — debate-expert-implementation-engineer — blue — opening — pifb-implementation-rock-solid

## Lens

Runtime behavior of the actual code — `path:line` reading, reachability under simulator entry point, code-vs-comment divergence, code-vs-manuscript divergence.

## Active config used

The active artifact for §Implementation's sub-claim 6 (manuscript-vs-code consistency) is the simulator at `C:\Users\chris and christine\Desktop\MAgent_Model-main\gauge_agent\meta_agents.py`. The class `ConsensusDetector` (lines 35-149) is dispatched by `MetaAgentFormation.detect_and_form` (lines 382-398) which calls `consensus_detector.find_clusters(system)` (line 391). `find_clusters` (lines 93-129) consumes `consensus_score` (lines 82-91) which consumes `belief_coherence` (lines 55-66) and `model_coherence` (lines 68-80). The path is reachable; this is the operative consensus detector in the released simulator.

## Steelman of the opposing position

The simulator at `meta_agents.py:55-91` implements neither the Gibbs form $C_q = \exp[-V/\tau_q]$ at manuscript line 2169 nor the three-factor consensus score $\Gamma = P \cdot C_q \cdot C_s$ at manuscript line 2174 — it returns `1.0 - E` (line 66) for both coherence factors and the two-factor product $C_b \cdot C_m$ (line 91) for the consensus score; the manuscript at line 2174 argues *against* the `1 - KL` form the simulator implements, and the frame averaging at line 358 uses an extrinsic Euclidean mean rather than the Lie-algebra-additive form at manuscript line 2191. Three concrete code-vs-manuscript divergences in §Implementation's two adjacent subsubsections.

## My position (in service of blue)

Concede sub-claim 6 on three concrete code-path findings. The simulator's `ConsensusDetector` does not implement the manuscript's specified detector, and the frame averaging in `MetaAgentFormation.form_meta_agent` does not implement the manuscript's specified BCH-additive form. Independent of sub-claim 6, the simulator's *gauge-covariant moment averaging* at `meta_agents.py:217-238, 290-321` does implement the manuscript's specified two-sided sandwich transport (`transport_covariance` at line 230) and the saddle-point coherence-weighted fixed-point at line 292-321, both of which are structurally faithful realizations of manuscript Eq. 2181-2186 — so the simulator partially honors the manuscript and partially diverges from it.

Sub-claim 6 is operationally asymmetric: manuscript line 2284 itself self-discloses the wound ("Whether the released simulator code realizes the full transport $\Omega_{i,I}$ or a frame-trivial substitute is not independently verified in this manuscript; the simulator code release is deferred to a follow-up"). The honest blue defense is that the manuscript has already conceded this sub-claim in its own text — a publication-ready section can self-disclose a follow-up gap without thereby becoming unpublishable; what would block publication is a *hidden* gap.

## Evidence

- **`MAgent_Model-main/gauge_agent/meta_agents.py:56-66`** (verified reachable via `consensus_detector.find_clusters → consensus_score → belief_coherence`):
  ```python
  def belief_coherence(self, system: MultiAgentSystem) -> Tensor:
      E = system.pairwise_alignment_energies('belief')
      while E.dim() > 2:
          E = E.mean(-1)
      return 1.0 - E
  ```
  Returns `1.0 - E` where `E` is the mean post-transport KL. This is *not* the manuscript's Gibbs form $C_q = \exp[-V/\tau_q]$ at line 2169 and is the specific form the manuscript at line 2174 argues against ("a $1 - \mathrm{KL}$ surrogate would be signed and could give misleadingly positive products from two negative factors").

- **`meta_agents.py:82-91`** (verified reachable via `find_clusters`):
  ```python
  def consensus_score(self, system: MultiAgentSystem) -> Tensor:
      C_b = self.belief_coherence(system)
      C_m = self.model_coherence(system)
      return C_b * C_m
  ```
  Two factors only; no presence factor $P$. The manuscript at line 2174 specifies three factors $\Gamma = P \cdot C_q \cdot C_s$.

- **`meta_agents.py:343-359`** (verified reachable via `form_meta_agent → detect_and_form`):
  ```python
  # Round-2 finding #10: this is an extrinsic average and is not closed
  # in GL+(K) for arbitrary orientations — a true intrinsic mean would be
  # ``matrix_exp(Σ_j w_j · matrix_log(omega_j))`` per manuscript line 1911.
  ...
  omega_avg = (w_q_b_om * omega_stack).sum(dim=0)
  omega_model_avg = (w_s_b_om * omega_model_stack).sum(dim=0)
  ```
  The frame average is computed extrinsically on group elements via weighted sum of $\Omega$ matrices, *not* via the Lie-algebra-additive form $\phi_I = \sum w_i \phi_i$ specified at manuscript line 2191. The docstring at lines 344-355 explicitly admits "the previous docstring claim of 'Lie-algebra-additive average' was wrong; corrected here." This is a code-vs-manuscript divergence that does NOT appear in the evidence pack's "Mismatch" section and is therefore a finding for this debate.

- **`meta_agents.py:229-234`** (verified reachable in the transport pass):
  ```python
  mu_q_t_list.append(transport_mean(omega_ij, agent.mu_q.data))
  sig_q_t_list.append(transport_covariance(omega_ij, agent.sigma_q))
  ```
  The covariance transport calls `transport_covariance` which applies the two-sided sandwich $\Omega \Sigma \Omega^\top$ — structurally faithful to manuscript Eq. 2145 and the implementation formula Eq. 2184.

- **`meta_agents.py:290-321`** (verified reachable in the fixed-point inner loop):
  Implements the saddle-point coherence-weighted iteration `w_raw = chi * torch.exp(-stable)` (line 300) with `stable = kls - kls.min()` — this is faithfully the manuscript's saddle-point weights at line 2187 ($w_i^I(x) = \chi_i(x)\exp[-\mathrm{KL}(q_i^{(s)} \| \bar{q}_I^{(s)})]$). The fixed-point converges in the high-coherence regime as the manuscript claims.

## Newly-discovered context (for 01b_extended_evidence.md)

- **`meta_agents.py:343-359` frame-averaging mismatch**: not in the evidence pack. The simulator computes $\Omega_{\mathrm{avg}} = \sum_i w_i \Omega_i$ extrinsically rather than $\phi_{\mathrm{avg}} = \sum_i w_i \phi_i$ followed by $\Omega_{\mathrm{avg}} = \exp(\phi_{\mathrm{avg}})$ as manuscript line 2191 specifies. The extrinsic average is not closed in $\mathrm{GL}^+(K)$ for arbitrary orientations and the docstring at lines 343-355 admits this. This is a second concrete code-vs-manuscript divergence beyond the consensus detector.
- **`meta_agents.py:282` cold start change**: AUDIT4 V9 note says cold start changed from uniform to $\chi$-weighted pool. This is an internal simulator history note and does not affect manuscript correspondence.
- **Partial match on gauge-covariant moment averaging**: the simulator at lines 217-238 and 290-321 implements the manuscript's Eq. 2181-2187 faithfully (two-sided sandwich transport, $\chi$-weighted saddle-point fixed-point). The evidence pack's "Match" claim at lines 81-90 is correct for the moment-averaging portion; it is incorrect on its implicit claim that the frame averaging is "Lie-algebra-additive" (line 86 of evidence pack docstring claim).

## Falsification conditions

This implementation-engineer position is wrong if:

1. The simulator's `1.0 - E` and `C_b * C_m` paths are actually unreachable under the released simulator's default entry-point config (verify by tracing the simulator's `main` or top-level training script).
2. The simulator's `transport_covariance` at line 230 does not actually compute $\Omega \Sigma \Omega^\top$ but a one-sided form (would falsify the partial match on gauge-covariance).
3. The frame extrinsic-average at line 358 is later projected back to $\mathrm{GL}^+(K)$ via polar decomposition or similar, restoring approximate closure (worth checking downstream consumers — the docstring at lines 350-355 says downstream consumers invert via `safe_inv` / `robust_cholesky`, which degrades gracefully but does not constitute manuscript-faithful Lie-algebra averaging).
4. The manuscript line 2284 ("simulator code release is deferred to a follow-up") is interpreted by judges as a forward declaration that *retracts* sub-claim 6 from §Implementation's scope rather than as an acknowledgment of an unverified-in-manuscript gap.

## Confidence

HIGH on the three concrete code-vs-manuscript divergences (`1.0 - E`, two-factor product, extrinsic frame average); HIGH on the partial match for moment averaging via sandwich transport and $\chi$-weighted fixed-point; MEDIUM on whether sub-claim 6's literal failure is fatal to the whole claim given the self-disclosure at line 2284 — that adjudication is for the chief judge. Would shift on (4) if scope judge treats line 2284 as removing sub-claim 6 from the operationalization.
