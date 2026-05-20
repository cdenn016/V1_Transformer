# Red Opening — vfe-active-inference-impl

## Steelman (opposing position)

The `/vfe` `active_inference=True` path is wired without dead ends — the master toggle, weights, and `epistemic_samples` propagate from `train_vfe.py` through `VFEConfig`, `VFEModel.__init__`, the stack, the block, and into the inner E-step loop, where the returned `(grad_mu, grad_sigma)` are summed with the analytic VFE gradient before the shared Fisher projection (`vfe/e_step.py:1164-1175`); the kernel openly discloses its substitution of `H[p_pred]` for `E_q[-log p^*(o)]` and calls its functional a "self-evidencing surrogate" rather than canonical EFE; the BALD epistemic term enters with a sign that, when subtracted from the descent direction, points the E-step toward higher mutual information, matching the FEP "epistemic value" role; and the project explicitly preserves a *canonical* EFE in a separate module (`vfe/efe.py`) for generation-time policy selection, so the design pattern is "self-evidencing-augmented inference at the E-step, canonical-EFE at the policy step" — a defensible and self-consistent extension of the FEP / Parr-Pezzulo-Friston program.

## Position

The implementation is wired, but the *theoretical-justification* half of the claim fails: what the `/vfe` `active_inference=True` path computes is **not** active inference in the sense of [ParrPezzuloFriston2022] or [FristonEtAl2017]. Three things are wrong simultaneously, and any one is sufficient to falsify the claim:

1. The "pragmatic" term in `transformer/core/active_inference.py:213` is `+λ_prag · E_x[H[p_pred(v|μ_x)]]`, **not** `E_q[-log p^*(o|C)]`. There is no preference distribution `p^*(o|C)` anywhere in the call chain. Minimizing the predictive entropy of one's *own* readout is a self-confidence regularizer, not a goal-directed pragmatic value. The code's own docstring (`transformer/core/active_inference.py:28-37`) labels this a "NOVEL-CONSTRUCTION" substitution and acknowledges the name `lambda_prag` is borrowed "for symmetry with the EFE literature." A construction that the author labels as not-canonical-EFE cannot simultaneously be theoretically justified *as* canonical-EFE.

2. The functional `F_AI` is added to the **belief** gradients `(grad_μ, grad_σ)` at `vfe/e_step.py:1166-1170` and natural-gradient-projected with the variational F at `vfe/e_step.py:1172-1175`. In the Parr-Pezzulo-Friston framework, expected free energy `G(π)` is a **policy** score — it parameterizes a posterior over actions/policies `q(π) ∝ exp(-G(π))`, not a posterior over latents [ParrPezzuloFriston2022 Ch. 2; canon §2]. The `/vfe` E-step has no action set, no policy posterior, and no observation contingent on an action. Adding a G-shaped term to the E-step `(μ, σ)` retraction is a regularized variational inference borrowing the EFE term labels — not active inference. The codebase already knows this distinction, because `transformer/vfe/efe.py:1-10` documents the contrasting design ("This is the correct place for active inference in a language model: generation-time action selection, not target-conditioned E-step inference") — i.e., the project's own internal guidance contradicts the `/vfe` `active_inference=True` E-step augmentation.

3. The published free-energy functional in `Attention/GL(K)_attention.tex` (`\label{eq:free_energy_functional_final}`, restated in CLAUDE.md) has five terms — `α·KL`, `λ_h·KL`, β-coupled belief KL + attention entropy, γ-coupled model KL + meta entropy, and the observation likelihood. It contains **no** `+λ_prag · H[p_pred] − λ_epi · MI` augmentation. The `active_inference=True` path therefore optimizes a functional that the manuscript does not derive. Either the manuscript or the code is the source of truth; both cannot be, and the user has stated that the manuscript is the canonical statement of F.

The load-bearing assumption is "self-evidencing / self-confidence ⇒ canonical pragmatic value of EFE." If that substitution fails, the implementation is at best a regularizer with EFE-themed naming, and the claim's "theoretically justifiable as an active-inference / expected-free-energy variant" half collapses.

## Evidence

- **Canonical EFE form, primary source.** `external_canon_inference.md` §2 quotes the standard decomposition `G(π) = E_q[-log p(o|C)] - E_{q(o|π)}[KL(q(s|o,π) ‖ q(s|π))]` and the policy posterior `q(π) ∝ exp(-G(π))`, sourced to [ParrPezzuloFriston2022] and [FristonEtAl2017]. The pragmatic term is `E_q[-log p^*(o|C)]`, an *expectation under preferences*, not `H[p_pred]`. [ParrPezzuloFriston2022 Ch. 2]; [FristonEtAl2017].

- **Code admits the substitution is not canonical.** `transformer/core/active_inference.py:28-37`:
  > "NOVEL-CONSTRUCTION DISCLOSURE. The lambda_prag term is the entropy of the model's OWN predictive distribution at the current mu_i. This is NOT the canonical EFE pragmatic value, which per [ParrPezzuloFriston2022] is E_q[-log p^*(o)] — the expected negative log preference under a target distribution. The present construction is a self-evidencing / self-confidence surrogate ... lambda_prag is named for symmetry with the EFE literature; readers comparing this against [ParrPezzuloFriston2022] should note the substitution."
  This is a self-disclosure by the implementer that the term is not what its name advertises. The pragmatic term in the running implementation at `transformer/core/active_inference.py:212-218` is `pragmatic_term = pragmatic_weight * entropy.mean()`, with `entropy = -(probs * log_probs).sum(dim=-1)` — the Shannon entropy of the readout distribution at `μ`, with no preference distribution `p^*(o|C)` entering anywhere in the gradient.

- **F_AI is added to belief gradients, not used as a policy score.** `transformer/vfe/e_step.py:1164-1175`:
  ```python
  if active_inference_fn is not None:
      ai_grad_mu, ai_grad_sigma = active_inference_fn(mu, sigma)
      if ai_grad_mu is not None:
          grad_mu = grad_mu + ai_grad_mu
      if ai_grad_sigma is not None:
          grad_sigma = grad_sigma + ai_grad_sigma
  # 4. Natural gradient projection
  nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
      grad_mu, grad_sigma, sigma, eps=eps,
  )
  ```
  Compare canonical EFE [ParrPezzuloFriston2022 Ch. 2 / canon §2]: G enters as a policy score in `q(π) ∝ exp(-G(π))`. There is no action variable or policy posterior in `/vfe`'s E-step; `(μ, σ)` are belief parameters, not policy probabilities.

- **The project's own EFE module separates the two design patterns.** `transformer/vfe/efe.py:1-10` opens with: "This is the correct place for active inference in a language model: generation-time action selection, not target-conditioned E-step inference." The internal contrast acknowledges that the `active_inference=True` E-step augmentation is *not* what `/vfe/efe.py` calls "the correct place for active inference."

- **Manuscript free-energy functional does not contain the AI augmentation.** `Attention/GL(K)_attention.tex` `\label{eq:free_energy_functional_final}` (restated in CLAUDE.md) has five terms; no `+λ_prag · H − λ_epi · MI` term. Grep in `01_evidence.md` (lines 108-112) confirms `Attention/*.tex` has no "active inference" / "pragmatic" / "EFE" / "self-evidencing" passage decomposing F into a pragmatic + epistemic augmentation. The implementation extends the published functional with terms the published derivation does not justify.

- **Test coverage is insufficient to verify the theoretical claim.** `tests/transformer/test_vfe_package.py:269-280` asserts only output shape `(B, N, K)`. No sign check, no finite-difference agreement with `∂H/∂μ` or `∂(−MI)/∂μ`, no end-to-end ablation showing `active_inference=True` changes anything meaningful. Per `external_canon_inference.md` §10 pitfall list ("Sign convention" — `F` vs `−F` vs `ELBO` consistency), a sign convention claimed but never tested is a weak claim.

- **Full-covariance epistemic σ-gradient is zero by design.** `transformer/core/active_inference.py:235-237`: "Consequence for full-covariance: ... with sigma=0 in the readout, the epistemic term contributes zero gradient to Sigma in the full-cov case. This matches efe.py semantics (diagonal-only BALD) and is an accepted limitation documented here rather than a silent bias." An *accepted limitation* is not the same as *theoretical justification*; the claim under debate is "theoretically justifiable," and a path where the epistemic term silently vanishes on σ for the full-covariance case undermines the term's claimed role as an information-gain regularizer when full Σ is in use.

- **Sign / weight ratio is asserted, not justified.** Active config (`transformer/vfe/train_vfe.py:104-108`) sets `pragmatic_weight=1.0` and `epistemic_weight=0.5`. Net effect on belief μ: confident-readout pressure (`+1.0 · H`) is twice the information-gain pressure (`−0.5 · MI`). The default 2:1 prag-over-epi ratio is not derived in the manuscript or the canon (`external_canon_inference.md` §2 does not prescribe relative weights). It is a free hyperparameter with the strength of an opinion.

## Falsification conditions

The position above is wrong if any of the following are demonstrated:

1. The FEP / active-inference canonical literature contains a derivation in which `H[p_pred(v|μ_i)]` is mathematically equivalent to `E_q[-log p^*(o|C)]` under a stated condition that holds in the `/vfe` language-modeling setting (no external preference). A specific citation — paper, chapter, equation — is required. A "spirit-of-self-evidencing" gesture toward [Hohwy2016] or `[Friston2010]` §self-evidencing is not sufficient unless the equation `pragmatic = H[p_pred]` appears in a standard treatment with an argument tying it to `E_q[-log p^*]`.

2. The Parr-Pezzulo-Friston framework explicitly licenses adding the EFE functional G as an additive shaping term to a **belief** (state) posterior's gradient — i.e., not only to the policy posterior `q(π) ∝ exp(-G(π))` but also to the state posterior `q(s)` — and the `/vfe` E-step's role-shift is documented in their canon. A citation to a Parr/Pezzulo/Friston passage that authorizes this would close the gap.

3. The five-term `\label{eq:free_energy_functional_final}` in `Attention/GL(K)_attention.tex` is shown to be a special case (`λ_prag = λ_epi = 0`) of a larger seven-term functional that the manuscript actually derives, with the pragmatic + epistemic augmentation tied to a generative-model assumption that the manuscript states. If blue can point to the `.tex` line where the augmentation is derived, the manuscript-vs-code drift dissolves.

4. The single shape-only test is supplemented by a sign or finite-difference test in the repository that confirms `∂F_AI/∂μ` agrees with the analytic derivative of `λ_prag · H[p_pred] − λ_epi · MI`, *and* by an empirical demonstration (logged run) that `active_inference=True` improves a downstream metric in a controlled comparison. If blue produces the run logs, the "theoretically justifiable in practice" half strengthens.

If none of (1)–(4) is provided, the claim's theoretical-justification half is not supported by the canon or by the code base, and the construction reduces to a regularized variational inference with EFE-themed names.
