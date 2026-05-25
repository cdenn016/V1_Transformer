# Blue Opening — skip-attention-em-phi-q-freezes-sigma

## Steelman (opposing position)

The legacy M-step loss `compute_free_energy_loss` reads `sigma_s = attn_info['sigma_prior']` directly from the model's embedding-derived prior (kept attached through `_embed_and_prepare`) and uses it inside `gaussian_kl_divergence` for `KL(s_i || h)` (when `lambda_hyper > 0`) and `KL(q || p)` (when `M_alpha > 0`), so `log_sigma_diag` receives a non-zero gradient regardless of whether the FFN's E-step detaches `sigma_p` at the boundary; the claim that σ_embed "stays frozen at its initialization value across the entire training run" is therefore false in its universal form.

## Position

I cannot defend the universal form of this claim under the current evidence. What I can defend is the conditional form: **under `lambda_hyper=0` and `M_alpha=0` (the dataclass defaults at `transformer/training/config.py:124, 128`), with `skip_attention=True` and `em_mode='em_phi_q'` on the legacy `transformer/core/variational_ffn.py` path, `log_sigma_diag` has no autograd path to the cross-entropy loss and stays frozen.** This is a strictly weaker but accurate statement than the original claim.

Three structural facts establish the conditional form. First, the FFN's E-step detaches `sigma_p` at entry under `em_phi_q`: at `transformer/core/variational_ffn.py:1564-1568`,

```
_attach_sigma = self.amortized_inference and self.amortize_sigma
if sigma_prior is not None:
    sigma_p = sigma_prior if _attach_sigma else sigma_prior.detach()
```

with `em_phi_q` resolving to `amortize_sigma=False` at `transformer/core/em_modes.py:28`, so `_attach_sigma = True and False = False`, and `sigma_p = sigma_prior.detach()` is reached at runtime. Second, the EM exit boundary at `transformer/core/variational_ffn.py:2771-2775` detaches the entire FFN output under `em_phi_q`:

```
_em_active = self.em_phi_mode in ('E_phi_q', 'M_phi_p')
if _em_active:
    mu_current = mu_current.detach()
    if sigma_current is not None:
        sigma_current = sigma_current.detach()
```

so `sigma_ffn` returned by `self.ffn(...)` carries no autograd graph back to the embedding; the subsequent `sigma_q = sigma_ffn.clamp(...)` at `transformer/core/blocks.py:1024` propagates a detached tensor to the next layer. Third, the attention sublayer — which would otherwise expose an autograd path through `MahalanobisNorm(mu_q, sigma_q)` at `transformer/core/blocks.py:803` and through `W_O · μ_agg` — is gated off entirely at `transformer/core/blocks.py:795` by `if not self.skip_attention:`. With both the FFN-internal path and the attention path closed, the only remaining post-stack consumers of σ are `final_norm(mu_q, sigma_q)` at `transformer/core/model.py:1333` and `prior_bank.decode(mu_q, sigma_q, ...)` at `transformer/core/model.py:637`, both of which receive a σ_q that has been detached at every layer's EM exit.

## Evidence

- `transformer/core/em_modes.py:28` — `'em_phi_q': dict(amortized_inference=True, amortize_sigma=False, exact_phi_grad=False, em_phi_mode='E_phi_q')`. Confirms `amortize_sigma=False` under the claim's `em_mode`.
- `transformer/core/variational_ffn.py:1564-1568` — `sigma_p = sigma_prior if _attach_sigma else sigma_prior.detach()` with `_attach_sigma` resolving to `False` under `em_phi_q`. The FFN-internal prior is detached at runtime.
- `transformer/core/variational_ffn.py:2771-2777` — EM exit boundary applies `.detach()` to `mu_current`, `sigma_current`, and (for `E_phi_q`) `phi_current`. All FFN outputs leave detached.
- `transformer/core/blocks.py:795` — `if not self.skip_attention:` gates the entire attention sublayer, including the `MahalanobisNorm(mu_q, sigma_q)` call at line 803 and the `W_O · μ_agg` residual path. Under `skip_attention=True`, this block does not execute.
- `transformer/core/blocks.py:983-989` — Under `skip_attention=True`, `_active_norm = self.norm1` and `mu_normalized = self.norm1(mu_q)`. The MahalanobisNorm/CenteredMahalanobisNorm branches at lines 984-987 do consume `sigma_q`, but the `mu_normalized` they produce is fed into the FFN where any σ-gradient it carries is killed at the EM-exit `mu_current.detach()`.
- `transformer/core/model.py:548-549` — `mu_prior = mu_q.clone(); sigma_prior = sigma_q.clone()`. The embedding's σ flows attached into `attn_info['sigma_prior']`, but only consumers downstream of this dictionary that read `sigma_prior` *outside* the FFN can use that autograd path.
- `transformer/training/config.py:124, 128` — `M_alpha: float = 0.0` and `lambda_hyper: float = 0.0` are the dataclass defaults. Under these defaults, the two paths in `transformer/train.py` that consume `sigma_s = attn_info['sigma_prior']` attached are gated off.

## Pre-emption of the strongest red attack

The strongest attack — and the one I cannot rebut — is `transformer/train.py:557-584`. With `lambda_hyper > 0`, the loss includes

```
kl_hyper = gaussian_kl_divergence(
    mu_q=mu_s,
    sigma_q=sigma_s,
    mu_p=mu_h_expanded,
    sigma_p=sigma_h_expanded,
)
```

where `sigma_s = attn_info['sigma_prior']` (line 355), and `attn_info['sigma_prior']` is the embedding's σ cloned at `transformer/core/model.py:549` without any intervening detach. The comment block at `transformer/train.py:553-555` makes this explicit: "Previous bug: sigma_s was .detach()'d in the KL, giving zero gradient to sigma_embed. Now sigma_s flows through, providing bidirectional gradient: pulls sigma toward Σ_h if it inflates OR deflates." This is the user's own corrective annotation of a path that bypasses both the attention sublayer and the FFN E-step entirely. The same applies at `transformer/train.py:472, 480-485` when `M_alpha > 0`: `sigma_q_for_kl = sigma_q.detach()` but `sigma_p` (which is `attn_info['sigma_prior']`, attached) is kept live, so `KL(q||p)` provides a direct gradient to `log_sigma_diag`.

I cannot defend the universal claim against this. The most I can defend is the conditional form gated on `lambda_hyper=0` and `M_alpha=0`.

## Falsification conditions

This conditional position is wrong if any of the following holds in the user's *active* config:

1. **`lambda_hyper > 0` on the legacy entry point** (`transformer/training/experiment_runner.py:944, 2179` plumbs `lambda_hyper` from `TrainingConfig` to `compute_free_energy_loss`). The Level-3 hyperprior KL at `transformer/train.py:579-584` provides a direct gradient to `log_sigma_diag` that does not pass through either the attention sublayer or the FFN E-step.
2. **`M_alpha > 0` on the legacy entry point**. The self-coupling `KL(q || p)` at `transformer/train.py:480-485` keeps `sigma_p` live, where `sigma_p = sigma_s = attn_info['sigma_prior']` — the embedding-derived σ, attached.
3. **The active entry point is `transformer/vfe/train_vfe.py`**. Per `.claude/agents/vfe-knowledge/em_modes.md` line 22 ("The `vfe/` package does not currently honor `em_mode` switching — it is hardwired to this profile."), the `em_mode='em_phi_q'` selector has no runtime effect; the claim is then *out of scope*, not true. The `vfe/` package's own gradient profile (canon `em_modes.md` lines 16-21) and its loss-side consumers must be audited separately to determine σ_embed's gradient status under that path.

I have not verified the user's active values of `lambda_hyper` and `M_alpha`. CLAUDE.md is explicit that "User may not be running the config values which match the repo. always double check what values the user is using!" A judge ruling on this debate must inspect the active config; with the evidence in this pack alone, the universal claim cannot stand, and the conditional defense above is the strongest honest position available.
