# Blue Opening — vfe-use-prior-bank-decoder

## Steelman (opposing position)

The `use_prior_bank=True` branch is not the canonical decoder advertised in CLAUDE.md: it computes `logits = -c·KL(q||π_v)/τ` with a learnable scalar `c = exp(decode_log_scale)` rather than the documented `-KL/τ`, it silently diagonalises the Gaussian KL when `diagonal_covariance=False`, and the discriminative readout at the V-classifier boundary is not a term of the variational free-energy functional F, so labelling it "VFE-native" is at best informal and at worst category-confused.

## Position

The claim survives all four sub-propositions on the evidence. (i) The implemented logit reduces *exactly* to `softmax(-KL(q||π_v)/τ)` modulo a row-only (`q`-side, V-invariant) additive constant — verified symbolically. (ii) The decoder lives on the same statistical manifold of diagonal Gaussians used by encode and the E-step under the user's active config (`diagonal_covariance=True`, `train_vfe.py:79`), so Law 3 holds without approximation in the live path. (iii) No `nn.Linear`, no MLP, no activation, no learned QKV projection sits on the `use_prior_bank=True` decode path (`vfe/model.py:99-101` allocates `output_proj` only in the False branch); a single learnable scalar `decode_log_scale` is a softmax temperature, not a neural network. (iv) The Gaussian KL-classifier form follows by direct substitution into Bishop's generative-classifier discriminant ([Bishop2006 §4.2]) and equivalently appears as the "accuracy" term of Form-3 FEP under a per-token Gaussian likelihood ([Friston2010 Form 3] = `[external_canon_inference.md §1]`).

## Evidence

**1. The decode formula reduces to the canonical claim modulo softmax-invariant constants (sympy-verified).**

Construct the diagonal-K=1 Gaussian KL and compare to the implementation's `combined + prior_bias`:

```
KL_diag = (1/2)[σ_q/σ_p + (μ_q−μ_p)^2/σ_p − 1 + log σ_p − log σ_q]
combined+bias = (σ_q+μ_q^2)/σ_p − 2μ_qμ_p/σ_p + μ_p^2/σ_p + log σ_p
```

Sympy session (executed):

```
2*KL - (combined+bias) = -log(sigma_q) - 1
d/d mu_p     = 0
d/d sigma_p  = 0
```

The difference depends only on `(μ_q, σ_q)` — the row, not the column `v`. Hence per row `i`,
`softmax_v(−(combined+bias)/(2τ)) = softmax_v(−KL(q_i || π_v)/τ)` *exactly*, not approximately. Per-component K-fold sum extends identically. This discharges sub-proposition 1 algebraically.

**2. The learnable scalar `c = exp(decode_log_scale)` is a softmax temperature, not a neural-network exception.**

`prior_bank.py:208` declares `nn.Parameter(torch.zeros(1))`; `prior_bank.py:505-506` applies `scale = exp(decode_log_scale.clamp(−3, 3))` (one learnable scalar, range `[0.05, 20]`). At initialisation `scale = 1`, so the untrained model produces logits *bitwise identical* to the documented `−KL/τ`. The "Hard Constraints" block in CLAUDE.md bans (a) `nn.Linear`, (b) MLPs, (c) learned `W_Q/W_K/W_V`, and (d) activation functions; a scalar reparameterising the temperature is none of these. The precedent for inserting a scalar in front of the softmax argument is the `1/√d_k` factor of scaled dot-product attention ([Vaswani2017 §3.2.1], `[external_canon_transformers.md §1]`) and the learned-temperature variant adopted in `CLIP` ([Radford2021]). Calling a single learnable scalar a "neural network" empties the term.

**3. The Gaussian KL-classifier form is derivable from Bishop §4.2 by substitution.**

Bishop's generative-classifier discriminant under class-conditional Gaussians `p(x | C_v) = N(x; μ_v, Σ_v)` with uniform class prior is `log p(C_v | x) ∝ log p(x | C_v) = −(1/2)(x−μ_v)^T Σ_v^{−1}(x−μ_v) − (1/2) log|Σ_v| + const` ([Bishop2006 §4.2]). Replace the point observation `x` with a recognition distribution `q(s)` and take the expected log-likelihood:

```
E_q[log p(s | C_v)] = −(1/2) E_q[(s−μ_v)^T Σ_v^{−1}(s−μ_v)] − (1/2) log|Σ_v| + const
                   = −KL(q || π_v) − H(q) + const
```

Since `H(q)` does not depend on `v`, the Bayes-optimal log-posterior discriminant is `−KL(q || π_v) + const_v-invariant`. The user's decode is the exact softmax of this discriminant scaled by `1/τ`. Equivalently, this is the "accuracy" term of Form-3 FEP under a per-token Gaussian observation model — `E_q[−log p(o=v|s)] = KL(q||π_v) + H(q)` ([Friston2010], `[external_canon_inference.md §1, Form 3]`). Two independent canonical routes deliver the same readout; the claim is not a novel construction but the standard form for a Gaussian generative classifier evaluated under a Gaussian recognition distribution.

The closest existing application in the language-model literature is Gaussian word-embedding retrieval with KL-similarity ([Vilnis-McCallum 2015, *Word Representations via Gaussian Embedding*]); the user's contribution is to apply this discriminant at the language-model decode boundary while sharing the Gaussian-orbit parameters with the encoder. The mathematical form is canonical; the architectural placement is the user's choice.

**4. The "VFE-native / Law-3" claim survives under the user's active config.**

`train_vfe.py:79` sets `diagonal_covariance=True`. Under this setting:

- Encode (`prior_bank.py:400-425`) returns `(μ_v, σ_v) ∈ ℝ^K × ℝ^K_+` per token — diagonal Gaussian.
- E-step belief `q_i` lives in the same diagonal-Gaussian manifold.
- Decode (`prior_bank.py:478`) extracts the diagonal of `sigma_q` and `sigma_p` and computes the *exact* KL between the two diagonal Gaussians (no projection happens because both sides are already diagonal).

So in the live path the diagonal-projection caveat is vacuous: encode, infer, and decode all read and write the same Σ representation. The KL is the Fisher-information canonical asymmetric divergence on the statistical manifold of diagonal Gaussians ([AmariNagaoka2000 Ch. 2], `[external_canon_math.md §1]`); per-token priors are V points on this manifold; `q_i` is one more point; `softmax_v(−KL(q_i || π_v)/τ)` is the Gibbs distribution over the V cluster centres at temperature τ. This is the canonical manifold-of-Gaussians classifier.

**5. Parameter accounting on the gauge-fixed branch is strictly more frugal than the `nn.Linear` alternative.**

Under `gauge_fixed_priors=True` (the structurally pure variant, `prior_bank.py:180-186`), the prior-bank parameters are `base_μ ∈ ℝ^K`, `base_log_σ ∈ ℝ^K`, and `phi_embed ∈ ℝ^{V×n_gen}`. Per-token capacity is `V·n_gen`, typically smaller than `V·K` for the `nn.Linear(K, V)` alternative because `n_gen ≪ K^2` and the algebra of block-diagonal generators in this codebase has `n_gen` on the order of `K` rather than `K^2`. The decode therefore uses *fewer* learnable scalars than the documented neural exception, while routing them through a gauge-orbit construction tied to the encoder. This rules out the cynical reading that the prior-bank decode is just a renamed linear classifier with extra parameters.

**6. Pre-emption of the strongest attack — "the decode KL is not a term in F, so the 'VFE-native' label is unearned."**

Granted that the decode KL is not literally a summand of the manuscript free-energy functional F (the loss assembled in `vfe/model.py` is `ce_loss + 0.5·mass_φ·||φ||^2 + Σ block._aux_hyperparam_loss`, not F directly — `model.py:11-32`). The "VFE-native" claim in CLAUDE.md is not that the decode is a term of F; it is that the decode reuses the *same Gaussian manifold and same parameters* (Law 3 — `config.py:339-341`: "reusing the same gauge-orbit prior used for encode"). Sub-proposition 2 is about manifold sharing, not about being a term of F. The decode is the *observation-likelihood readout* on the same statistical manifold the E-step lives on, which is what "VFE-native at the decode boundary" denotes here. Form-3 FEP makes this explicit: the accuracy term `E_q[−log p(o|s)]` is part of F when an observation is realised; at training time the cross-entropy loss `−log softmax_v(logits)_{y}` *is* this accuracy term up to the V-invariant `H(q)` shift. The decode is not orthogonal to F; it is precisely the observation-likelihood face of F when the generative model declares per-token Gaussian likelihoods.

## Falsification conditions

This position is wrong if any of the following hold:

1. **Algebraic identity fails.** If `2·KL(q||π_v) − (combined + prior_bias)` depends on `v` (not solely on `q`-side quantities), then `softmax_v` does *not* equal `softmax(−KL/τ)` and sub-proposition 1 fails. Falsifier: a numeric example where the difference varies across `v`. (The sympy session above rules this out for diagonal Gaussians of arbitrary K; full-covariance would need an independent check, but the live config is diagonal.)

2. **`decode_log_scale` is structurally an MLP or activation in disguise.** If `decode_log_scale` were a function `f(input)` rather than a scalar parameter, or were applied per-token from a learned table, the "scalar temperature" defence collapses. Falsifier: any code path where the value applied at line `prior_bank.py:505` is not a single scalar shared across `(B, N, V)`. (Confirmed shared: `nn.Parameter(torch.zeros(1))`.)

3. **`nn.Linear` or activation present on the `use_prior_bank=True` path.** Falsifier: a `path:line` reference showing an `nn.Linear`, `nn.GELU`, `nn.ReLU`, etc. invoked on `mu_final` or `beliefs` between `model.py:184` and the return of `decode()`. (Verified absent.)

4. **The Bishop / Friston route is misapplied.** Falsifier: a textbook citation showing the expected log-likelihood under a Gaussian likelihood and Gaussian recognition is *not* `−KL(q||π_v) − H(q) + const`. (The identity is standard — see also `[external_canon_math.md §1]` for the closed-form KL between Gaussians.)

5. **Law 3 fails under the user's live config.** Falsifier: a code path where decode reads `(μ_p, σ_p)` from a representation different from the one encode wrote, under `diagonal_covariance=True`. (Verified: encode writes diagonal Gaussians, decode reads diagonal Gaussians from the same `phi_embed` / `mu_embed` / `sigma_log_embed` parameters.)

6. **Gauge-fixed parameter accounting wrong.** If `n_gen ≥ K` in any standard generator set produced by this codebase, the "fewer parameters than `nn.Linear(K, V)`" sub-claim collapses. Falsifier: a generator construction with `n_gen ≥ K`. (Standard block-diagonal `gl(K)` generators have `n_gen = Σ K_h^2`; for a single block `n_gen = K^2 > K` once `K > 1` — so this particular argument applies only in the gauge-fixed *block-diagonal* construction with small blocks, not for a single `K`-dimensional GL block. I retract the strong frugality claim for single-block configurations and concede that the strict parameter-frugality argument is config-dependent. The structural argument — no MLP, no activation, no learned QKV — survives independently.)

7. **`decode_log_scale` drifts during training so the deployed temperature departs from `c=1`.** If observed values of `decode_log_scale` deviate significantly from 0 in trained checkpoints, the deployed model is no longer the canonical `−KL/τ` but a temperature-rescaled variant. This does not falsify the *form* of the decoder (it is still `softmax(−KL · c/τ)`), but it falsifies any claim that the documented formula is preserved bitwise in trained models. The honest reading is the one already in the docstring (`prior_bank.py:23-34`): the deployed decoder is a softmax temperature stacked on top of `τ`. The user's CLAUDE.md text would be sharper if it said `logits = −c·KL/τ` with `c` learnable; I would support a minor manuscript edit to align the documentation. This is a documentation-precision issue, not a soundness violation.
