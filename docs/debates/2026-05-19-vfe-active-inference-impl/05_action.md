# Action — vfe-active-inference-impl

**From verdict:** RED_WINS

## Recommended action

The implementation is wired correctly and discloses its non-canonical substitution in code (`transformer/core/active_inference.py:28-37`). The verdict requires aligning the user-facing labeling and the manuscript with what the code already admits internally. Five concrete steps:

1. **Rename to reflect the self-evidencing-surrogate character.**
   - `VFEConfig.active_inference` → `self_evidencing_regularizer`
   - `VFEConfig.pragmatic_weight` → `self_confidence_weight`
   - `VFEActiveInference` → `VFESelfEvidencingRegularizer`
   - Epistemic / BALD-MI half is canonical and keeps its name.
   - Retain a one-release deprecation alias for external configs.

2. **Restrict the user-facing claim** in `transformer/vfe/train_vfe.py:104-108` comments and any documentation to "self-evidencing surrogate at the E-step plus canonical EFE at generation time," matching the existing docstring disclosure. Do not describe the E-step path as "active inference" in user-facing surfaces without the surrogate qualifier.

3. **Add a manuscript appendix** to `Attention/GL(K)_supplementary.tex` (or a new appendix file) that derives `F + F_AI` from the PriorBank generative model under the no-exogenous-preference setting, states the substitution `p^* ← p_pred` explicitly, cites `[ParrPezzuloFriston2022 Ch. 2]` and `[Friston2010]`, and flags the dark-room failure mode with the BALD MI as the counterweight. Label this an experimental extension not implied by the published five-term F functional at `\label{eq:free_energy_functional_final}`.

4. **Add empirical falsifiers** to `tests/transformer/test_vfe_package.py`. At minimum:
   - Finite-difference test that `∂F_AI/∂μ` agrees with the analytic derivative of `λ_prag · H[p_pred] − λ_epi · MI` on small-dim inputs.
   - Sign test verifying that descent on `grad_mu + ai_grad_mu` increases MI (matching the EFE convention).
   - Short controlled training comparison (`active_inference=True` vs `False`, with `epistemic_weight=0` as the dark-room control) logging predictive-entropy trajectories and validation loss across a fixed step budget.
   - The existing test at lines 265-280 asserts only shape and is insufficient under canon §10 sign-convention pitfalls.

5. **Mark research-track in `CLAUDE.md`** alongside the existing RoPE × MahalanobisNorm documented limitation, so the "theoretically pure path under appropriate toggles" invariant is preserved: the pure path is `active_inference=False`; the research-track extension is `active_inference=True` (or its renamed equivalent).

## Follow-up debates (if any)

The verdict treats the load-bearing proposition (theoretical justification consistent with the Friston / Parr-Pezzulo-Friston canon) as resolved against the claim. Two sub-questions are worth their own debate only if the user wishes to revisit:

- **"Self-evidencing in [Friston2010 §self-evidencing] / [Hohwy2016] licenses substituting `p^* ← p_pred` in EFE for a language model with no exogenous preference distribution `C`."** Blue did not produce a primary-source citation supporting this substitution; if such a citation exists, the verdict could be revisited.
- **"The E-step belief retraction is an admissible site for EFE augmentation in continuous-time active-inference formulations."** The canon's policy-posterior framing (`q(π) ∝ exp(-G(π))`) is process-theory; whether continuous-time perception-action formulations admit a belief-level EFE term is a literature question Blue gestured at but did not cite.
