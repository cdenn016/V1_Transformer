# Red Opening — skip-attention-em-phi-q-freezes-sigma

## Steelman (opposing position)

Under `BlockConfig.skip_attention=True` and `BlockConfig.em_mode='em_phi_q'`, `variational_ffn.py:2771-2775` detaches both `mu_current` and `sigma_current` at the EM boundary before returning, the attention sublayer that would otherwise carry the gradient is bypassed by `blocks.py:795` (`if not self.skip_attention:`), the FFN's internal `sigma_p` is detached at `variational_ffn.py:1564-1568` because `em_phi_q` resolves to `amortize_sigma=False`, so the only graph edges that could connect the final loss to `sigma_embed` are severed and `sigma_embed` cannot be updated by backprop.

## Position

The universal claim that "`sigma_embed` receives no autograd path during training **and therefore stays frozen at its initialization value across the entire training run**" is false. The conjunction `skip_attention=True ∧ em_mode='em_phi_q'` is not sufficient to guarantee either half of the claim:

1. The autograd path is **not** universally severed — `train.py:472-485` (M_alpha self-coupling) and `train.py:578-584` (lambda_hyper hyper-prior) construct loss terms over `sigma_s = attn_info['sigma_prior']`, which is a live differentiable clone of the raw embedding output (`model.py:549` — `sigma_prior = sigma_q.clone()`). These two loss components route gradient to `sigma_embed` independent of the FFN's EM-boundary detach and independent of the attention sublayer.
2. Even in the configuration where the autograd path is severed (M_alpha = lambda_hyper = M_beta = 0), `sigma_embed` is **not** frozen — the user's own `em_phi_q` config (`HEBBIAN_CONFIG` at `train_publication.py:363-449`) sets `use_p_flow=True` (line 408), which invokes `hebbian.update_embeddings_from_beliefs` (`hebbian.py:119-201`). At `hebbian.py:196-201`, `embed.log_sigma_diag.data[update_tokens] = torch.log(new_sigma.clamp(min=1e-6))` performs a non-autograd EMA write to `sigma_embed` derived from the E-step's evolved `sigma_beliefs`. P-flow is the explicit replacement learning rule for backprop in the Hebbian configuration; the warning text at `block_config.py:288-298` predates this design and is stale in its assertion that the embeddings "stay frozen at initialization."

## Evidence

- **`transformer/train.py:578-584`**:
  ```python
  kl_hyper = gaussian_kl_divergence(
      mu_q=mu_s,
      sigma_q=sigma_s,
      mu_p=mu_h_expanded,
      sigma_p=sigma_h_expanded,
  )
  ```
  `sigma_s = attn_info['sigma_prior']` (line 355), `attn_info['sigma_prior'] = sigma_prior` from `model.py:609`, and `sigma_prior = sigma_q.clone()` at `model.py:549`. `.clone()` preserves the autograd graph, so `sigma_s` is differentiably attached to `sigma_embed` via the embedding forward at `embeddings.py:626-628` (`sigma_diag = torch.exp(_ls).clamp(...)`). The hyper-prior block at `train.py:557-588` explicitly notes `# sigma_s NOT detached: gradient flows to sigma_embed (bidirectional)` (line 578).

- **`transformer/train.py:472-485`**:
  ```python
  sigma_q_for_kl = sigma_q.detach() if sigma_q is not None else None
  ...
  kl_per_agent = gaussian_kl_divergence(
      mu_q=mu_q,
      sigma_q=sigma_q_for_kl,
      mu_p=mu_p,
      sigma_p=sigma_p,
  )
  ```
  `sigma_p = sigma_s` at line 362 (live), and the comment block at lines 473-479 explicitly states "sigma_p kept LIVE: the M-step KL(q||p) is the correct gradient source for sigma_p when M_alpha > 0."

- **`transformer/core/hebbian.py:196-201`**:
  ```python
  current_sigma = torch.exp(embed.log_sigma_diag.data[update_tokens])
  new_sigma = (
      (1.0 - sigma_lr.unsqueeze(-1)) * current_sigma
      + sigma_lr.unsqueeze(-1) * update_sigma
  )
  embed.log_sigma_diag.data[update_tokens] = torch.log(new_sigma.clamp(min=1e-6))
  ```
  In-place write to `embed.log_sigma_diag.data` (a `.data` write that bypasses autograd entirely). The call chain to this site is fully concrete: `hebbian.py:631-646` reads `trainer.config.use_p_flow` and invokes `trainer.model.p_flow_update(..., sigma_beliefs=...)`, which delegates via `model.py:1664-1675` → `hebbian.p_flow_update_model:467-470` → `embeddings.py:734-749` → `hebbian.update_embeddings_from_beliefs` → the in-place `.data` write at line 201.

- **`transformer/train_publication.py:405,408`**: the user's `HEBBIAN_CONFIG` (the only config that pairs `em_mode='em_phi_q'` with the model anywhere in the repository) sets `em_mode='em_phi_q'` AND `use_p_flow=True`. The block at lines 354-362 explicitly documents the design: "μ_embed, σ_embed: P-flow EMA toward successful beliefs (prediction-error weighted)." The autograd severance is by design; P-flow is the substitute update rule.

- **Executed verification** (`/tmp/test_sigma_freeze.py`, ran against the legacy `GaugeTransformerLM` with `skip_attention=True`, `em_mode='em_phi_q'`, vocab=100, K=4, n_layers=1):

  Test 1 — `M_alpha=0, M_beta=0, lambda_hyper=0, mass_phi=0`:
  ```
  RESULT: log_sigma_diag.grad is None - autograd path is SEVERED
  SANITY mu_embed.weight.grad norm = 6.66e-03
  ```
  Confirms severance in the minimal case (matches the warning text's assertion for that narrow configuration).

  Test 3 — `lambda_hyper=1.0` (sigma_embed perturbed off init so dKL/dsigma ≠ 0):
  ```
  With lambda_hyper=1.0 perturbed: log_sigma_diag.grad norm = 8.11e-02
    nonzero entries: 64/400
  ```
  The autograd path is **not** severed when the hyper-prior is active.

  Test 4 — `M_alpha=1.0` (sigma_embed perturbed off init):
  ```
  With M_alpha=1.0 perturbed: log_sigma_diag.grad norm = 8.33e-02
    nonzero entries: 64/400
  ```
  The autograd path is **not** severed when the self-coupling KL(q||p) is active.

  At `M_alpha=lambda_hyper=0` with the embedding at its initialization point, lambda_hyper produced `grad norm = 0.0` (not `None`) because KL(s‖h) vanishes at `sigma_s = sigma_h`; the path exists, the gradient is just zero by symmetry at that one point. Once `sigma_embed` is perturbed, the gradient is nonzero.

- **`transformer/core/variational_ffn.py:1614`**:
  ```python
  sigma_current = sigma
  ```
  No detach. The E-step iteration's σ-retraction at `vfe_utils.py:600` is `sigma_new = sigma_safe * torch.exp(exp_arg)` — a multiplicative function of the input `sigma_current` — so the autograd path from output σ back to input `sigma_q` is preserved throughout the iteration loop. The detach happens only at the EM boundary (lines 2771-2775), after the iteration loop completes. This is the load-bearing line for the claim and it does NOT touch `sigma_p`, only the FFN-output beliefs.

## Falsification conditions

This position is wrong if any of the following holds:

1. **(Autograd-path falsification.)** Under `skip_attention=True` and `em_mode='em_phi_q'`, **all** loss components that take `sigma_s` or `sigma_p` as a live (non-detached) argument are gated off by `BlockConfig.__post_init__` or by an equivalent invariant — i.e., `M_alpha > 0` and `lambda_hyper > 0` are forbidden at config-validation time when the freeze combination is active. (I verified `block_config.py:256-348` — no such forbidding logic exists. `train.py` accepts `M_alpha > 0` and `lambda_hyper > 0` regardless of `em_mode`.)

2. **(P-flow falsification.)** The P-flow sigma branch at `hebbian.py:182-201` is gated by `embed.learnable_sigma and hasattr(embed, 'log_sigma_diag')` AND requires `sigma_beliefs is not None` in the upstream call. If under `em_phi_q` + `skip_attention=True` the `full_metrics['p_flow/sigma_q']` entry is `None` (e.g., because `sigma_q` is detached at the EM boundary and the trainer fails to populate `p_flow/sigma_q`), the σ-branch is skipped and only μ updates. I have verified the trainer call site at `hebbian.py:637-646` does pass `sigma_beliefs = full_metrics.get('p_flow/sigma_q')` (so a `None` here would defeat the σ update). `train.py:814-815` shows `metrics['p_flow/sigma_q'] = sigma_q.detach()` is set unconditionally when `sigma_q is not None`, which under the active config it is — so the σ branch fires. The remaining loophole is `embed.learnable_sigma`: if the embedding is constructed with `learnable_sigma=False`, P-flow does not update σ, but in that case `log_sigma_diag` is a buffer, not a parameter, and the freeze-vs-trainable framing of the claim is malformed.

3. **(Empirical falsification.)** A trained run under `HEBBIAN_CONFIG` produces `log_sigma_diag` values identical (bit-exact, or within ULP) to the initialization after some non-trivial number of training steps. (I have not run this; the textual + single-step empirical evidence above does not by itself prove that `sigma_embed` *moves* over many steps, only that the mechanism for it to move exists.)
