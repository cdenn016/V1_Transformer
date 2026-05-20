# Verdict — vfe-active-inference-impl

## Outcome

RED_WINS

## Decisive evidence

The code's own self-disclosure at `transformer/core/active_inference.py:28-37` states verbatim:

> "NOVEL-CONSTRUCTION DISCLOSURE. The lambda_prag term is the entropy of the model's OWN predictive distribution at the current mu_i. This is NOT the canonical EFE pragmatic value, which per [ParrPezzuloFriston2022] is E_q[-log p^*(o)] — the expected negative log preference under a target distribution. The present construction is a self-evidencing / self-confidence surrogate ... lambda_prag is named for symmetry with the EFE literature; readers comparing this against [ParrPezzuloFriston2022] should note the substitution."

This matches the canon at `external_canon_inference.md` §2, which defines `G(π) = E_q[-log p(o|C)] - E_{q(o|π)}[KL(q(s|o,π) ‖ q(s|π))]` with pragmatic value `E_q[-log p^*(o|C)]` parameterized by goal context `C` (not the agent's own predictions), and the policy posterior `q(π) ∝ exp(-G(π))` over actions (not a belief retraction term). Blue concedes both the substitution gap and the absence of the augmentation in the published F functional at `Attention/GL(K)_attention.tex \label{eq:free_energy_functional_final}` (Blue Rebuttal, Concession 1 and 2).

## Reasoning

The compound claim is conjunctive: wiring AND theoretical justification "as an active-inference / expected-free-energy variant in the sense of the standard Friston / Parr-Pezzulo-Friston literature." Red conceded wiring in full. The debate turns on the theoretical-justification half.

Red has primary-source citations on three independent fronts. Canon §2 fixes the canonical pragmatic value as `E_q[-log p^*(o|C)]` with an exogenous preference `C`; the codebase computes `H[p_pred(v|μ)]` with no exogenous `C`. Canon §2 fixes EFE as a policy-posterior score; the codebase adds the augmentation to a belief retraction. The published five-term F functional in `Attention/GL(K)_attention.tex` does not derive the `+λ_prag H[p_pred] − λ_epi MI` augmentation, and Blue conceded this in writing.

Blue's defense rests on §10 pitfall 3 — "specific implementations follow from specific generative-model choices ... require the explicit generative model that connects them" — re-read as a license for novel constructions provided the generative model is stated. The canon text is a hygiene rule against "FEP implies X" claims, not an authorization that any specific extension qualifies as "in the sense of the standard literature." Blue produced no primary-source citation that the substitution `p^* = p_pred` is a recognized self-evidencing reduction in `[Friston2010]`, `[FristonEtAl2017]`, or Hohwy 2016 — only the assertion that such a reduction holds. Red's counter that the substitution is structurally the dark-room failure mode (any sharply-peaked readout minimizes `H[p_pred]` regardless of contact with reality, with the BALD term as the sole counterweight at half the weight under the active config's `λ_prag=1.0, λ_epi=0.5`) is unrebutted.

The Fisher-natural-gradient argument confuses metric with functional: `[Amari1998]` describes the geometry on the q-manifold, not a transformation that turns a non-canonical objective into a canonical one. Red's rebuttal articulates this distinction cleanly. The architectural separation Blue cites — canonical EFE in `vfe/efe.py`, surrogate in `vfe/active_inference.py` — is by the codebase's own design choice an acknowledgement that the E-step augmentation is not the canonical form; placing the canonical form elsewhere does not make the E-step augmentation canonical.

The decisive citation is the codebase's own disclosure: a construction labeled by its author as "NOT the canonical EFE pragmatic value" cannot simultaneously be defended as theoretically justifiable in the canonical sense. Honest labeling weighs for the engineering quality of the disclosure but against the conjunctive claim under debate. Combined with Blue's two material concessions (the substitution gap and the manuscript drift), the theoretical-justification half fails, and the compound claim with it.

## Action

The implementation is wired correctly and discloses its non-canonical substitution. The fix is to align the user-facing labeling and the manuscript with what the code already admits internally.

1. Rename the user-facing config and class to reflect the self-evidencing-surrogate character rather than canonical active inference. Concrete renames: `active_inference` → `self_evidencing_regularizer` in `VFEConfig`; `pragmatic_weight` → `self_confidence_weight`; `VFEActiveInference` → `VFESelfEvidencingRegularizer`. The epistemic / BALD-MI half is canonical and can keep its name. Retain a deprecation alias for one release if external configs depend on the old names.

2. Restrict the claim in `train_vfe.py:104-108` comments and any user-facing documentation to "self-evidencing surrogate at the E-step plus canonical EFE at generation time," matching the docstring disclosure at `transformer/core/active_inference.py:28-37`. Do not describe the E-step path as "active inference" in user-facing surfaces without the surrogate qualifier.

3. Add a manuscript appendix to `Attention/GL(K)_supplementary.tex` (or a new appendix file) deriving `F + F_AI` from the PriorBank generative model under the no-exogenous-preference setting, stating the substitution `p^* ← p_pred` explicitly, citing `[ParrPezzuloFriston2022 Ch. 2]` and `[Friston2010]`, and flagging the dark-room exposure with the BALD MI as the counterweight. The appendix must label this as a research-track extension that is not implied by the published five-term F functional at `\label{eq:free_energy_functional_final}`.

4. Add empirical falsifiers to the test suite. At minimum: (i) a finite-difference test that `∂F_AI/∂μ` agrees with the analytic derivative of `λ_prag · H[p_pred] − λ_epi · MI` on small-dim inputs; (ii) a sign test verifying that descent on `grad_mu + ai_grad_mu` increases MI (matching the EFE convention); (iii) a short controlled training comparison (`active_inference=True` vs `False`, with `epistemic_weight=0` as the dark-room control) that logs predictive entropy trajectories and validation loss across a fixed step budget. The current `tests/transformer/test_vfe_package.py:265-280` asserts only shape and is insufficient under the canon §10 sign-convention pitfall.

5. Mark the path research-track in `CLAUDE.md` alongside the existing RoPE × MahalanobisNorm documented limitation, so the "theoretically pure path under appropriate toggles" invariant is preserved: the pure path is `active_inference=False`; the research-track extension is `active_inference=True` (or its renamed equivalent).
