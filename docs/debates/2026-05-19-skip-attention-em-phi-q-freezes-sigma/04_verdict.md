# Verdict — skip-attention-em-phi-q-freezes-sigma (user-adjudicated)

The judge phase was disabled (`judge=off`). Both sides' openings are below as factual relays. The user declares the verdict.

## Red opening summary

Red argued the claim is false in its universal form. The argument has two prongs.

First, the autograd path is not universally severed: `transformer/train.py:472-485` (M-step self-coupling, active when `M_alpha > 0`) and `transformer/train.py:578-584` (hyper-prior KL, active when `lambda_hyper > 0`) both consume `sigma_s = attn_info['sigma_prior']` attached. The chain `sigma_prior = sigma_q.clone()` at `transformer/core/model.py:549` preserves the autograd graph back to `sigma_embed` through `transformer/core/embeddings.py:626-628`. Red ran an executable verification (`/tmp/test_sigma_freeze.py`): under `lambda_hyper=1.0` with `sigma_embed` perturbed off init, `log_sigma_diag.grad` has norm `8.11e-02` with 64/400 nonzero entries; under `M_alpha=1.0` similarly, norm `8.33e-02`. Under the minimal case (all VFE loss weights zero), the gradient is `None` — matching the warning text under that narrow configuration.

Second, even when the autograd path is severed, `sigma_embed` is not frozen: the user's only `em_phi_q` config (`HEBBIAN_CONFIG` at `transformer/train_publication.py:363-449`) sets `use_p_flow=True`, which routes through `transformer/core/hebbian.py:201` where `embed.log_sigma_diag.data[update_tokens] = ...` performs a non-autograd EMA write to `sigma_embed` derived from the E-step's evolved σ beliefs. P-flow is the explicit substitute update rule.

Red conceded the FFN's EM-boundary detach at `transformer/core/variational_ffn.py:2771-2775` and the attention bypass at `transformer/core/blocks.py:795` are real. The claim's failure is in the universal quantifier and the "therefore stays frozen" causal chain.

## Blue opening summary

Blue produced a calibrated defense rather than a sycophantic one. Blue conceded that the universal form of the claim is not defensible on the code evidence and defended only the conditional form: under the dataclass defaults `M_alpha=0.0` and `lambda_hyper=0.0` (`transformer/training/config.py:124, 128`), on the legacy `transformer/core/variational_ffn.py` path, with `skip_attention=True` and `em_mode='em_phi_q'`, `log_sigma_diag` has no autograd path to the cross-entropy loss.

Three structural facts established the conditional form: (i) the FFN entry detaches `sigma_p` because `em_phi_q` resolves to `amortize_sigma=False`, so `_attach_sigma = False` at `transformer/core/variational_ffn.py:1564-1568`; (ii) the EM exit boundary at `transformer/core/variational_ffn.py:2771-2775` detaches `mu_current`, `sigma_current`, and (for `E_phi_q`) `phi_current`, so the FFN returns no autograd graph; (iii) the attention sublayer, which would otherwise expose autograd paths through `MahalanobisNorm` and `W_O · μ_agg`, is gated off entirely at `transformer/core/blocks.py:795`.

Blue pre-empted the strongest attack on the universal form, citing the same `transformer/train.py:557-584` path Red used. Blue's own annotation of the user's comment block at `transformer/train.py:553-555` ("Previous bug: sigma_s was .detach()'d in the KL, giving zero gradient to sigma_embed. Now sigma_s flows through...") explicitly identifies this as a previously-fixed bypass.

Blue's falsification conditions: the conditional position is wrong if (a) `lambda_hyper > 0`, (b) `M_alpha > 0`, or (c) the active entry point is `transformer/vfe/train_vfe.py` (where `em_mode` is hardwired and the selector has no runtime effect, making the claim out-of-scope rather than true). Blue flagged that it did not verify the user's active values of `lambda_hyper` and `M_alpha`, per the CLAUDE.md rule about always double-checking active config values.

## Your verdict

(User to fill in. Both sides converged on the same nuanced finding through different routes: the claim as stated is false in its universal form; it holds only under a narrow zero-loss-weight configuration on the legacy path, and is moot on the `vfe/` path. The actionable item is to verify the user's active `lambda_hyper` and `M_alpha` values, and to revise either the claim or the project policy assertion at `transformer/core/block_config.py:288-298` to reflect the conditional nature of the freeze.)
