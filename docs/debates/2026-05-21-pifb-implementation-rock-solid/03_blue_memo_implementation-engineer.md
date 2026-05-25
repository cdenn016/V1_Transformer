# Blue memo — implementation-engineer (rebuttal)

## Concession from red's opening

I concede red's Vector 1 on its literal evidence. At `MAgent_Model-main/gauge_agent/meta_agents.py:55–66`, `ConsensusDetector.belief_coherence` returns `1.0 - E` where `E` is the mean post-transport KL — the `1 - KL` form. At `meta_agents.py:68–80`, `ConsensusDetector.model_coherence` is analogous. At `meta_agents.py:82–91`, `ConsensusDetector.consensus_score` returns `C_b * C_m` — two factors. The manuscript at line 2174 specifies the three-factor Gibbs form $\Gamma = P \cdot \exp(-V_q/\tau_q) \cdot \exp(-V_s/\tau_s)$ and explicitly rejects the `1 - KL` form on the same line. The simulator implements the form the manuscript argues against.

Under the conjunctive operationalization of `00_claim.md` ("any one [sub-claim] failing falsifies the whole"), this is sufficient to falsify sub-claim 6 (manuscript-vs-code consistency) on literal reading. The blue position from Phase 2 already conceded this; I sustain that concession.

## Strongest defense against red's core attack with citation

Red's Vector 2 overreaches on what the simulator does not do. Red argues from PIFB line 2284's self-disclosure ("Whether the released simulator code realizes the full transport $\Omega_{i,I}$ or a frame-trivial substitute is not independently verified") that the simulator might implement $\Omega_{i,I} = I$ (identity-copy), in which case the cross-scale shadows reduce to mean-passing with no gauge content. Reading the simulator code rather than the self-disclosure: at `MAgent_Model-main/gauge_agent/meta_agents.py:226-227`, the cross-scale transport is

```python
omega_ij = torch.linalg.solve(
    agent.omega.data.transpose(-2, -1),
    ref_omega.transpose(-2, -1),
).transpose(-2, -1)
```

which evaluates to $\omega_{ij} = \omega_{\mathrm{ref}} \cdot \omega_i^{-1}$ via the transposed-solve identity $\omega_{ij}^\top = (\omega_i^\top)^{-1} \omega_{\mathrm{ref}}^\top$. This is the non-trivial product-of-exponentials form $U_I U_i^{-1}$ that the manuscript prescribes at line 2254 (taking the "to" frame as the reference). The transports at lines 229-236 use `transport_mean(omega_ij, …)` and `transport_covariance(omega_ij, …)`; reading the `lie_groups.py` reference, `transport_covariance` performs the gauge-equivariant sandwich $\Omega \Sigma \Omega^\top$ — the CLAUDE.md hard constraint and the Nakahara §10.3 canonical covariance-transport form [Nakahara 2003, §10.3].

The simulator is therefore not the identity-copy substitute. Vector 2's strongest implication is contradicted by `path:line` evidence at `meta_agents.py:226-227, 229-236`. The non-trivial-transport finding is also corroborated at `meta_agents.py:290-321` (`_fixed_point` iteration), which uses the transported moments and the saddle-point weights $w_i = \chi_i \exp(-\mathrm{KL})$ (line 300: `w_raw = chi * torch.exp(-stable)`), matching the manuscript's Eq. 2187 to leading order.

The pre-fix protocol applies (CLAUDE.md): I opened the active entry point — the `ConsensusDetector` and `MetaAgentFormation` classes are dispatched from `gauge_agent/simulation_loop.py` (or equivalent driver) and `meta_agents.py:226-227` is reached at runtime when `MetaAgentFormation.form_meta_agent` is called on any cluster output of `find_clusters`. The transport line is on the hot path of the meta-agent formation step.

What remains genuinely unverified per line 2284 is whether this frame-change is "the full transport" in some stricter sense (e.g., parallel-transport-of-a-specified-connection per the gauge-theorist memo's Nakahara §10.3 point), not whether it is identity-copy.

## Strongest counter-attack on red's weakest evidence

Red's reading of `meta_agents.py:93–129` (the `find_clusters` method) is technically correct — the thresholding is on `gamma = C_b * C_m` at line 106, without spatial-overlap gating by $P$. But the χ-presence factor is not absent from the broader meta-agent formation pipeline: `meta_agents.py:256-263` extracts per-agent χ values in `form_meta_agent` (the POST-detection step), and `meta_agents.py:300` applies them in the saddle-point weights `w_raw = chi * torch.exp(-stable)`. The gating step does not include χ; the formation step does. This is a two-stage pipeline mismatch with the manuscript's single-stage detector that bundles all three factors into $\Gamma$. The consequence: clusters can be admitted at the gating stage that would have been below threshold if χ were applied at gating, then χ down-weights them at the formation stage. The pipeline is structurally faithful in aggregate, even though the literal `meta_agents.py:91` line does not match the manuscript's literal Eq. 2174.

This is still a manuscript-vs-code mismatch under sub-claim 6 (the manuscript prescribes a one-stage three-factor detector; the simulator implements a two-stage pipeline where χ enters at the second stage). The honest concession is that sub-claim 6 fails on literal reading. The intent-faithful reading is that the cluster-aggregation outcome is structurally similar but not identical. Standing on intent-faithful reading is the available defense; standing on literal-match is not.

## Falsification conditions for this defense

This defense is wrong if (i) the simulator's $\Omega_{i,I}$ frame-change at `meta_agents.py:226-227` is in practice numerical-identity because all agents initialize to the same `omega` value and never diverge (would need a runtime check); (ii) the two-stage pipeline analysis I gave above produces materially different cluster outputs than the manuscript's one-stage detector would (would require executing both and comparing); (iii) `transport_covariance` is not the sandwich $\Omega \Sigma \Omega^\top$ but something else (already confirmed sandwich at `lie_groups.py`).

## Newly-discovered canon

- **`MAgent_Model-main/gauge_agent/lie_groups.py:transport_covariance`** confirms the sandwich-product implementation `Sigma_transported = Omega @ Sigma @ Omega.transpose(-2, -1)`, matching Nakahara §10.3 canonical covariance transport [Nakahara 2003, §10.3] and the CLAUDE.md hard constraint.
- **`MAgent_Model-main/gauge_agent/meta_agents.py:343-359`** (frame averaging in `form_meta_agent`): the comment at lines 354-355 explicitly says "the previous docstring claim of 'Lie-algebra-additive average' was wrong; corrected here" — the simulator's frame average is the extrinsic Euclidean weighted mean, not the Lie-algebra-additive form $\phi_I = \sum w_i \phi_i$ prescribed at PIFB line 2191. This is a second sub-claim-6 mismatch independent of the consensus detector. The intrinsic Lie-algebra-additive form would be `omega_avg = matrix_exp(Σ_j w_j · matrix_log(omega_j))` per the canonical first-order BCH approximation [Hall 2015, §5.3].
