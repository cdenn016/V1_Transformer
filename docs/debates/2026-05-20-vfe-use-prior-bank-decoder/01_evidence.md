# Evidence Pack — vfe-use-prior-bank-decoder

## Active config (entry point)

`transformer/vfe/train_vfe.py:36` — `'use_prior_bank': False` (user default; the dataclass default in `config.py:348` is also `False`). Toggle resolves to `cfg.use_prior_bank` consumed at `transformer/vfe/model.py:91`.

Relevant adjacent toggles in the user's active config:
- `diagonal_covariance: True` (`train_vfe.py:79`) — decode KL is exact under this setting.
- `gauge_fixed_priors: False` (`train_vfe.py:45`) — direct mode: per-token (μ_v, σ_v) lookup; φ_v retained only for transport.
- `decode_tau: 1.0` (`config.py:243`, `train_vfe.py:106`).

## Code references

### Decode path (the implementation claim)

- `transformer/vfe/prior_bank.py:427-508` — `VFEPriorBank.decode(mu_q, sigma_q, tau)`. Materializes all V priors (gauge-fixed or direct mode), extracts diagonals from sigma_q / sigma_p, and computes the fused diagonal-KL classifier.
- `transformer/vfe/prior_bank.py:485-499` — fused matmul construction:
  ```
  lhs = [sigma_q + mu_q^2, -2*mu_q]                                 # (B, N, 2K)
  rhs = [1/sigma_p,         mu_p / sigma_p]                          # (V, 2K)
  combined = lhs @ rhs.T                                             # (B, N, V)
  prior_bias = (mu_p**2 / sigma_p).sum(-1) + log(sigma_p).sum(-1)    # (V,)
  ```
- `transformer/vfe/prior_bank.py:505-506` — the actual logit construction:
  ```
  scale = exp(decode_log_scale.clamp(-3.0, 3.0))                      # learnable scalar
  logits = -0.5 * scale / tau * (combined + prior_bias.unsqueeze(0).unsqueeze(0))
  ```
- `transformer/vfe/prior_bank.py:208` — `self.decode_log_scale = nn.Parameter(torch.zeros(1))`. Initialized at 0 so scale=1 at construction; clamped to [exp(-3), exp(3)] ≈ [0.05, 20] during training.
- `transformer/vfe/prior_bank.py:23-34` (module docstring) — explicit acknowledgment that the implementation deviates from the canonical `logits = -KL/tau`: "This module instead computes `logits = -c * KL(q || pi_v) / tau` with a learnable scalar `c = exp(decode_log_scale)`. … Equivalent to a second softmax temperature stacked multiplicatively on `tau`."
- `transformer/training/optimizer.py:247-252, 311-316` — `decode_log_scale` is routed into the `m_sigma_params` / `'sigma'` optimizer group together with `base_log_sigma` and `sigma_log_embed` (per-group LR `m_sigma_lr`, default `5e-5`).

### Diagonal projection at decode (full-cov branch)

- `transformer/vfe/prior_bank.py:478-479` — `sigma_q_diag = torch.diagonal(sigma_q, dim1=-2, dim2=-1) if is_full_cov else sigma_q`. The decode unconditionally projects to a diagonal-only KL for O(V·K) efficiency.
- `transformer/vfe/prior_bank.py:436-439` (docstring) — "When beliefs are full-covariance (B, N, K, K), the decode uses diagonal projection for O(V·K) efficiency. This is a documented approximation at the decode boundary — encode and infer operate on the full Gaussian manifold, but decode projects to diagonal KL."

### Algebraic identity (combined + prior_bias = 2·KL + V-invariant constants)

For diagonal q ~ N(μ_q, diag(σ_q)) and p_v ~ N(μ_p_v, diag(σ_p_v)):
```
2·KL(q || p_v)  =  Σ_k [ σ_q_k/σ_p_v_k + (μ_q_k - μ_p_v_k)^2/σ_p_v_k  - 1 + log(σ_p_v_k) - log(σ_q_k) ]
combined+bias   =  Σ_k [ σ_q_k/σ_p_v_k + (μ_q_k - μ_p_v_k)^2/σ_p_v_k       + log(σ_p_v_k)               ]
```
Difference is `Σ_k [1 + log(σ_q_k)]`, which is constant across `v`. Hence at scale=1, tau=1:
```
softmax_v(-0.5·(combined+bias))  =  softmax_v(-KL(q || p_v))
```
i.e. the canonical claim holds **under softmax over v**, modulo a position-only additive constant.

### Encode path (gauge orbit, used in both toggle states)

- `transformer/vfe/prior_bank.py:400-425` — `encode(token_ids)` returns BeliefState. Gauge-fixed mode lifts the shared base prior `(μ_0, Σ_0)` to per-token `(μ_v, Σ_v) = (A_v μ_0, A_v Σ_0 A_v^T)` with `A_v = exp(Σ_a φ_v^a G_a)`. Direct mode looks up `(μ_v, σ_v)` per token, retaining `φ_v` only for downstream transport.
- `transformer/vfe/prior_bank.py:283-364` — `_apply_gauge_transform`. Uses the sandwich product `A_h diag(s_h) A_h^T` exactly when `diagonal_covariance=False`; otherwise applies `Σ_j A_{ij}^2 s_j` diagonal approximation.

### Model wiring

- `transformer/vfe/model.py:90-101` — `self.prior_bank = VFEPriorBank(cfg, generators)` is built unconditionally; `self.use_prior_bank = cfg.use_prior_bank`; when False, an `nn.Linear(K, V, bias=False)` is allocated and `output_proj` replaces the KL readout.
- `transformer/vfe/model.py:183-195` — decode branch:
  ```
  if self.use_prior_bank:
      logits = self.prior_bank.decode(mu_final, beliefs.sigma, tau=self.cfg.decode_tau)
  else:
      logits = self.output_proj(mu_final)        # nn.Linear, σ discarded
  ```
- `transformer/vfe/config.py:338-348` — toggle docstring. The True branch is described as "PriorBank.decode computes logits = -KL(q || π_v) / τ, reusing the same gauge-orbit prior used for encode (Law 3 — same manifold for encode/infer/decode). No additional decode parameters."

### CLAUDE.md framing of the constraint

`CLAUDE.md` "Hard Constraints" — "**NO NEURAL NETWORKS** … The only retained neural component is a linear output projection from K dimensions to vocabulary size (subsumed by the PriorBank decode, `logits = -KL(q || π_v)/τ`, when `use_prior_bank=True`)."

## Canon excerpts

### Gaussian KL closed form (canonical)

`[Bishop2006 Pattern Recognition and Machine Learning, App. B (multivariate Gaussian KL); also Cover & Thomas 2006 §8.6]`:
```
KL( N(μ_1, Σ_1) || N(μ_2, Σ_2) )
    = 0.5 [ tr(Σ_2^{-1} Σ_1) + (μ_2 - μ_1)^T Σ_2^{-1} (μ_2 - μ_1) - K + log|Σ_2|/|Σ_1| ]
```

### Generative classifiers (KL-as-logit form)

`[Bishop2006 §4.2 "Probabilistic Generative Models"]`: with class-conditional Gaussians `p(x|C_k) = N(μ_k, Σ_k)` and uniform class prior, the Bayes-optimal log-posterior is `log p(C_k|x) ∝ log p(x|C_k) = -0.5 (x - μ_k)^T Σ_k^{-1} (x - μ_k) - 0.5 log|Σ_k| + const`. Replacing the point observation `x` with a distribution `q(x)` and taking the expected log-likelihood gives `E_q[log p(x|C_k)]`, which equals `-H(q) - KL(q || p_k)`. Since `H(q)` does not depend on `k`, the discriminant becomes `-KL(q || p_k) + const` — the user's decode form.

### Free energy / ELBO standard form

`[Friston2010 Eq. 2.2; BleiKuckelbirgJordan2017 Eq. 3]`:
```
F[q]  =  E_q[-log p(o|s)]  +  KL(q(s) || p(s))           # = -ELBO
```
At the decode boundary, the user's framework is *not* directly minimizing F — see `transformer/vfe/model.py:11-32` ("the training objective assembled in forward() is `loss = ce_loss + 0.5 * mass_phi * ||phi||^2 + sum(block._aux_hyperparam_loss)`, NOT the manuscript free-energy functional F"). The outer M-step is structurally amortized inference; the decode KL is a discriminative readout at the V-classifier boundary, *not* a term in F.

### Manifold-of-Gaussians

`[Amari Information Geometry, Ch. 2-3]`: the space of Gaussian distributions is a statistical manifold with the Fisher information metric. The KL divergence is the canonical asymmetric divergence on this manifold. Per-token priors `{π_v}` constitute a finite point cloud on this manifold; the encoded belief `q_i` is also a point on the same manifold. `KL(q_i || π_v)` therefore is a well-defined divergence between points on a single statistical manifold.

### What the canon does *not* contain

- "softmax(-KL(q || π_v) / τ)" is **not** a standard formulation in the transformer literature [Vaswani2017, Bahdanau2015, etc.]. Standard transformers use `logits = W_O · h_i` (linear projection of the last-layer representation). The KL-to-prior form is the user's specific construction.
- Discriminative classifiers on the manifold-of-Gaussians via softmax(-KL) appear in retrieval / metric learning contexts (e.g., `[Vilnis-McCallum-2015 Word Representations via Gaussian Embedding]` uses KL-similarity between Gaussian word embeddings) but with a different application surface. The user's framing as a language-model decoder is not the dominant convention.

## What this evidence does NOT settle

1. Whether the learnable scalar `c = exp(decode_log_scale)` constitutes a "soundness violation" of the canonical claim or merely an equivalent re-parameterization of the temperature.
2. Whether the diagonal projection at decode (when `diagonal_covariance=False`) breaks Law 3 ("same manifold for encode/infer/decode") in a load-bearing way or is a documented O(V·K) approximation that does not corrupt the gauge-orbit structure of the prior bank itself.
3. Whether the manuscript's labeling of this decode as "VFE-native" survives the observation that the decode KL is *not* a term in the M-step loss F — it is a discriminative readout, not a variational update.
4. Whether the no-neural-networks constraint is fully discharged when an Embedding lookup of `(μ_v, σ_v, φ_v)` is treated as "not a neural network" — the Embedding tables in `_per_token_priors` (direct mode) hold `V·(2K + n_gen)` learnable scalars whose parameter budget exceeds an `nn.Linear(K, V)` by a factor of `(2K + n_gen)/K`. The pure constraint is whether MLPs / activations / learned QKV projections are present; the parameter-count argument is separate.
