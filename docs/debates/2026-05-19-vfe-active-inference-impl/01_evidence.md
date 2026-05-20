# Evidence Pack — vfe-active-inference-impl

## Active config (entry point)

`transformer/vfe/train_vfe.py:104-108` sets:

```python
'active_inference':         True,
'pragmatic_weight':         1.0,
'epistemic_weight':         0.5,
'epistemic_samples':        4,
'decode_tau':               1.0,
```

`transformer/vfe/config.py:236-244` declares these as `VFEConfig` fields, defaults `active_inference=False`, `pragmatic_weight=1.0`, `epistemic_weight=0.5`, `epistemic_samples=4`, `decode_tau=1.0`. Comment in config (lines 241-244) flags that `learnable_decode_tau` was deliberately removed because the AI gradient runs in a fresh autograd graph with detached `μ,σ` leaves so `tau` would not actually be learnable.

## Code references (wiring)

- `transformer/vfe/config.py:236-244` — config fields. Default OFF; opt-in for this debate.
- `transformer/vfe/model.py:120-124` — `VFEModel.__init__` constructs `VFEActiveInference(cfg, self.prior_bank)` and stashes it on `self._active_inference_fn` when `cfg.active_inference` is True. Closely held: no extra `nn.Module` re-registration (callback is a plain attribute pointing to a child module already owned by the model).
- `transformer/vfe/model.py:193-196` — the callback is passed positionally into the stack: `self.stack(beliefs, initial_priors, mask, active_inference_fn=self._active_inference_fn)`.
- `transformer/vfe/stack.py:53-56` — `ActiveInferenceFn` type alias: `Callable[[torch.Tensor, torch.Tensor], Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]`.
- `transformer/vfe/stack.py:76-105` — `VFEStack.forward` threads the callback to each `VFEBlock`.
- `transformer/vfe/block.py:157-175` — `VFEBlock.forward` passes the callback to `self.e_step(...)`.
- `transformer/vfe/e_step.py:725-746` — `VFEEStep.forward` accepts `active_inference_fn: Optional[ActiveInferenceFn]`; docstring labels Law 1 enforced (no `targets` parameter exists).
- `transformer/vfe/e_step.py:1164-1170` — the φ-mode E-step inner loop applies the callback after the analytic VFE gradient is assembled and BEFORE the natural-gradient (Fisher) projection:
  ```python
  if active_inference_fn is not None:
      ai_grad_mu, ai_grad_sigma = active_inference_fn(mu, sigma)
      if ai_grad_mu is not None:
          grad_mu = grad_mu + ai_grad_mu
      if ai_grad_sigma is not None:
          grad_sigma = grad_sigma + ai_grad_sigma
  ```
- `transformer/vfe/e_step.py:1172-1175` — `compute_natural_gradient_gpu(grad_mu, grad_sigma, sigma, eps=eps)` applies the Fisher metric to the SUM of analytic VFE gradient + AI gradient. Confirms the docstring claim that all terms share one information-geometric descent.
- `transformer/vfe/e_step.py:2207-2213` — omega-direct E-step path repeats the additive AI gradient step in the same position.

## Callback definition

`transformer/vfe/active_inference.py:1-79`:
- `VFEActiveInference.__init__` stores `pragmatic_weight`, `epistemic_weight`, `epistemic_samples`, `decode_tau`, and stashes `prior_bank` in `__dict__` to avoid `nn.Module` child registration (lines 51-53). Comment: "Use `__dict__` to avoid `nn.Module` registering `prior_bank` as child."
- `forward(mu, sigma)` delegates to `_compute_active_inference_gradient` (imported from `transformer/core/active_inference.py`).

## Core EFE-gradient kernel

`transformer/core/active_inference.py:1-50` — module docstring labels itself "experimental" and explicitly discloses:

```
NOVEL-CONSTRUCTION DISCLOSURE. The lambda_prag term is the entropy
of the model's OWN predictive distribution at the current mu_i. This
is NOT the canonical EFE pragmatic value, which per
[ParrPezzuloFriston2022] is E_q[-log p^*(o)] — the expected negative
log preference under a target distribution. The present construction
is a self-evidencing / self-confidence surrogate that drives the
readout toward low-entropy (peaked) predictions without requiring an
external preference. lambda_prag is named for symmetry with the EFE
literature; readers comparing this against [ParrPezzuloFriston2022]
should note the substitution.
```

The kernel implements:
```
F_AI = lambda_prag * H[p_pred(v | mu_i)]            (pragmatic — self-confidence, not E_q[-log p^*])
     - lambda_epi  * MI(v; mu | q_i)                (epistemic — BALD MI via S MC samples)
```

`transformer/core/active_inference.py:94-349` — implementation details:
- Inference-mode guard returns `(None, None)` (lines 171-172).
- Two readout paths: `PriorBank.decode(mu, sigma, tau)` (KL-based) or `W_out` linear fallback (lines 182-187). The `decode_tau` is floored at `1e-8` (line 180).
- Pragmatic: detached `mu_f32` / `sigma_f32` leaves, `torch.autograd.grad(pragmatic_term, [mu_var, sigma_var], create_graph=False)` (lines 205-222). Returns `grad_mu_accum`, `grad_sigma_accum`.
- Epistemic: BALD MI estimated via the identity `MI = E_s[KL(p_s || p_bar)]`, two-pass Welford-style accumulation (Pass 1 in `no_grad` builds `log p_bar`; Pass 2 per-sample autograd accumulates per-sample gradients with reparameterized `mu_s = mu + sqrt(sigma) * eps`).
- Sigma gradient flows through both the readout (KL-based logits use Σ) and the reparameterization path (diagonal only); full-cov BALD has zero σ-gradient by design (lines 235-237).
- `lambda_epi` enters with a NEGATIVE sign (`epi_per_sample_weight = -epistemic_weight / epistemic_samples`, line 306). Net effect: descent on `F + F_AI` MAXIMIZES MI, consistent with EFE "epistemic value" being subtracted from G.

## Fail-fast guards

`transformer/core/active_inference.py:454-535` — `wire_readout_references` (called from the legacy `core` model, not from `vfe/`). Hard errors:
- `active_inference=True` + `closed_form_e_step=True` → ValueError (EFE would do nothing).
- `active_inference=True` + `use_deq=True` → ValueError (DEQ implicit-diff backward uses VFE-only Jacobian; gradient would be biased).

These guards are for the `transformer/core/` path. The `vfe/` package has neither `closed_form_e_step` nor `use_deq` flags in its `VFEConfig`, so the analogous footguns do not exist there.

## Tests

`tests/transformer/test_vfe_package.py:265-280`:
```python
class TestVFEActiveInference:
    def test_callback_produces_gradients(self, model, cfg):
        ai = VFEActiveInference(cfg, model.prior_bank)
        B, N, K = 2, 8, cfg.embed_dim
        mu = torch.randn(B, N, K)
        sigma = torch.ones(B, N, K)
        grad_mu, grad_sigma = ai(mu, sigma)
        assert grad_mu.shape == (B, N, K)
        assert grad_sigma.shape == (B, N, K)
```

The single test verifies shape only. No test verifies sign, scale, or any theoretical property (no test asserts that the gradient direction matches the analytic derivative of `H[p_pred]` or `-MI`; no test verifies that turning `active_inference` on actually changes belief trajectories; no test verifies the EFE-vs-novel-pragmatic substitution does what is claimed).

## Generation-time EFE (separate module)

`transformer/vfe/efe.py:1-150`:
- `VFEExpectedFreeEnergy` is generation-time action selection, scoring candidate next tokens by `G(a) = risk + ambiguity - epistemic_value`. Risk = `E_{q(o|a)}[-log p^*(o)]` via `compute_risk`. Ambiguity is the canonical EFE form (mean predictive entropy, NOT `H[p_bar]` — explicit comment at lines 90-95).
- This module is the canonical EFE form per `[ParrPezzuloFriston2022]`. It is NOT what is being added to the E-step. The E-step receives the `VFEActiveInference` callback (the self-evidencing surrogate), while generation uses `VFEExpectedFreeEnergy` (the canonical form). The two share `epistemic_samples` / `epistemic_weight` config fields but compute different functionals.

## Manuscript references

Grep across `Attention/*.tex` for "active inference", "pragmatic", "epistemic", "EFE", "self-evidencing":
- `Attention/GL(K)_attention.tex:887` — uses the word "epistemic" only in the multi-agent sense ("epistemic consensus driven by β_ij"). The standard EFE pragmatic-epistemic decomposition (Friston 2017 process theory) is NOT derived in the main paper.
- `Attention/GL(K)_attention.tex:2223` — long matching line (omitted from grep output).
- `Attention/Participatory_it_from_bit.tex` and `Attention/GL(K)_supplementary.tex` — discuss "epistemically dead" agents in the meta-agent / hierarchical-coupling context, NOT the EFE pragmatic+epistemic decomposition.
- The free-energy functional in `Attention/GL(K)_attention.tex` (`\\label{eq:free_energy_functional_final}` per CLAUDE.md) has FIVE terms (α·KL self-coupling, λ_h·KL hyper-prior, β-coupled belief KL+entropy, γ-coupled model KL+entropy, observation likelihood). It does NOT contain a `+ λ_prag · H[p_pred] − λ_epi · MI` augmentation. The EFE-augmented F is a `transformer/core/active_inference.py` software-side extension, not part of the published functional.

## Canon excerpts (external_canon_inference.md)

§2 (Active inference — standard form):
```
G(π)  =  E_q[-log p(o|C)]  -  E_{q(o|π)}[KL(q(s|o,π) ‖ q(s|π))]
       =  expected cost (under preferred outcomes C)  -  expected information gain
```
- Pragmatic = `E_{q(o|π)}[-log p^*(o|C)]` (expected cost under preferences C).
- Epistemic = `E_{q(o|π)}[KL(q(s|o,π) ‖ q(s|π))]` (information gain about latent s after observing o).
- Policy posterior: `q(π) ∝ exp(-G(π))`.
- Pitfall note (§10): "FEP is a variational principle; specific implementations follow from specific generative-model choices. Claims that FEP alone implies an architectural choice ... require the explicit generative model that connects them."

§3 (Hierarchical formulations): standard FEP is single-agent or hierarchical with a single ancestral generative model. Multi-agent gauge-transport couplings are NOVEL and require independent justification.

§10 (Pitfalls): "Form-1 vs Form-2 vs Form-3 conflation" — derivations that quote one form's interpretation while using another form's terms are sloppy. "Sign convention" — F vs −F vs ELBO must be consistent.

## What this evidence does NOT settle

1. **Whether the self-confidence-surrogate `H[p_pred]` is theoretically justifiable as a pragmatic value in the absence of an external preference distribution `p^*(o)`.** The code admits it is not the canonical form; it claims a "self-evidencing" interpretation. This needs to be evaluated against the FEP self-evidencing literature (Hohwy 2016, Friston 2010 §self-evidencing) — does that literature support replacing `E_q[-log p^*]` with `H[p_pred]` in a language-modeling setting where no preference distribution is naturally available?
2. **Whether mixing the EFE-augmented gradient INTO the E-step (belief-level inference) — rather than INTO a policy/action posterior — is a valid use of EFE.** In `[ParrPezzuloFriston2022]`, G is a *policy* score (`q(π) ∝ exp(-G(π))`), not a belief-update objective. The codebase's design — adding `∂F_AI/∂μ` to the E-step μ gradient — is a different operation: it biases the E-step toward beliefs whose readout is confident and informative. Is this still "active inference," or is it a regularized variational inference that borrows the EFE term labels?
3. **Whether the sign convention is internally consistent across the codebase.** Pragmatic enters as `+λ_prag · H[p_pred]` (minimize entropy ⇒ pursue confident beliefs); epistemic enters as `−λ_epi · MI` (maximize MI). With both attached at `λ_prag=1.0`, `λ_epi=0.5`, the net effect on belief μ is: confidence dominates information-seeking 2:1. Is this the intended balance, and does it match any analytic stability argument?
4. **Whether the BALD MI estimate at S=4 samples is statistically meaningful as a gradient signal.** S=4 is low for MC MI estimation; the gradient noise could be large enough that the `-λ_epi · MI` direction is dominated by noise. No empirical evidence in the repo (no ablation logs in the active config tree) demonstrates that the EFE-augmentation actually improves a downstream metric.
5. **Whether the test coverage adequately verifies behavior.** The sole unit test asserts only shape (B, N, K). No sign tests, no agreement with finite-difference, no end-to-end "active_inference=True changes training trajectory" smoke test.
6. **Whether the manuscript justifies this augmentation.** The published F functional does not include the pragmatic+epistemic augmentation; the `Attention/*.tex` grep returns no "active inference" / "pragmatic" / "EFE" / "self-evidencing" hits with the relevant decomposition. So the implementation extends F by terms not derived in the paper.
