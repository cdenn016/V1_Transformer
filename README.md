# Gauge-Theoretic VFE Transformer

A research framework implementing gauge-covariant variational free energy (VFE) minimization for language modeling. The model carries no neural-network components: all representational capacity comes from iterative VFE minimization over Gaussian belief tuples `(mu, Sigma, phi)`. This codebase accompanies the manuscript

> **Attention as Gauge-Theoretic Variational Inference**
> Robert C. Dennis
> (`Attention/GL(K)_attention.tex`, with derivations in `Attention/GL(K)_supplementary.tex`)

The implementation lives in the `transformer/core/` package as `GaugeTransformerLM`: a configuration-driven realization of the theory in which attention, transport, and belief updates are all KL-divergence and natural-gradient operations. The single-E-step `transformer/vfe/` reformulation and the active-inference `aif/` package have been moved to a separate repository; this repository is the frozen `core/`-centered implementation.


## Thesis

Transformer attention is variational inference over information sources. Each token is modeled as a Gaussian agent `q_i = N(mu_i, Sigma_i)` carried in a local gauge frame `phi_i` on a statistical fiber bundle, and the attention weights are the stationary point of a single free-energy functional. The standard scaled dot-product rule

```
beta_ij = softmax(Q_i K_j^T / sqrt(d_k))
```

is recovered as a limit of the gauge-theoretic rule

```
beta_ij = softmax(-D_KL(q_i || Omega_ij q_j) / tau),    tau = kappa * sqrt(K)
```

under two successive specializations (isotropic covariances and a flat gauge connection), with the standard query–key product identified as `W_Q W_K^T = sigma^{-2} Omega^{-T}` at the level of the per-head bilinear form. Here `kappa` is a learnable scalar and the `sqrt(K)` factor sets the dimensional scale of the KL magnitude. The attention rule is `eq:mixture_softmax` in `GL(K)_attention.tex`; its per-head temperature form is `eq:per_head_temperature`.

The only retained linear map is the output projection from `K` dimensions to the vocabulary, and even that is subsumed by the prior-bank decode `logits = -D_KL(q || pi_v) / tau` when `use_prior_bank=True`. There are no MLPs, no activation functions, and no learned `W_Q`/`W_K`/`W_V` projections.


## What this codebase implements

`GL(K)_attention.tex` develops the single-belief-channel theory on a flat bundle: one Gaussian belief per token, GL(K) gauge transport between tokens, and the reduction to standard attention. The two-channel hierarchical form (a model channel `s_i`, the hyper-prior weight `lambda_h`, and the meta-attention coupling `gamma_ij`) and the non-flat holonomy regime are explicitly deferred to the companion paper *Participatory Realization: It from Bit* (`Attention/Participatory_it_from_bit.tex`, referenced at `GL(K)_attention.tex:619,665,677`).

The `transformer/core/` package realizes the single-channel theory through `GaugeTransformerLM`. Its variational FFN E-step minimizes the self-coupling `alpha * D_KL(q_i || p_i)`, the alignment coupling `sum_j beta_ij * D_KL(q_i || Omega_ij q_j)`, and the attention-entropy term `tau * beta_ij * log(beta_ij / pi_j)` by natural-gradient descent on `(mu, Sigma, phi)`. The model-coupling channel and the hyper-prior term are the companion-paper extension and are not implemented here. A large `BlockConfig` surface selects among gradient-flow modes, deep-equilibrium implicit differentiation, closed-form and Hebbian belief updates, reflection-augmented transport, and an opt-in non-flat connection.


## Free energy and attention

The reduced single-channel free energy minimized over beliefs is (`eq:free_energy_final`)

```
F_red[{q_i}] = sum_i D_KL(q_i || p_i)  -  tau * sum_i log Z_i  -  E_q[log p(o | {k_i})]
```

with pairwise energy `E_ij = D_KL(q_i || Omega_ij q_j)`, partition function `Z_i = sum_j pi_j exp(-E_ij / tau)`, and `tau = kappa * sqrt(K)`. Before the inner minimization over the attention distribution, the per-row alignment functional is (`eq:F_align_canonical_tau`)

```
F_align^(tau) = sum_j [ beta_ij * E_ij  +  tau * beta_ij * log(beta_ij / pi_j) ].
```

The entropy term `tau * beta_ij * log(beta_ij / pi_j)` is what makes the softmax a stationary point. Minimizing a purely linear functional of `beta` over the simplex would collapse to a delta function rather than a softmax; the entropy term restores the soft maximum, with `pi_j` the attention prior (uniform `1/N` over valid positions in the implementation).

The implementation must distinguish the canonical free energy above from the entropy-suppressed surrogate `sum_j beta_ij E_ij` obtained by holding `beta` fixed. Their gradients differ by an envelope-gap covariance term (`eq:autograd_envelope_gap`)

```
grad_x <E>_{beta*}  -  grad_x F_red  =  -tau^{-1} * Cov_{beta*}(E_ij, d E_ij / d x),    x in {mu, Sigma, phi}.
```

The supplementary material works with the surrogate for the covariance gradient and notes that the two forms agree for `Sigma` because the attention entropy does not depend on `Sigma_i` (`GL(K)_supplementary.tex`, App. B.1).


## Architecture and forward pass

`GaugeTransformerLM` (`transformer/core/model.py`) is a stack of `GaugeTransformerBlock`s. Each block is the pure-VFE form: a variational FFN that runs the belief E-step, which computes its own attention distribution `beta` internally and is the only message-passing path. The separate attention sublayer was removed on 2026-06-01 (see `edits_2026-06-01.md`); the `IrrepMultiHeadAttention` class and its KL-attention kernels remain in `attention.py` for the variational loss and ablation baselines.

```
token_ids
  |
  v
encode:  (mu, sigma, phi) <- token/prior embeddings   [PriorBank when use_prior_bank=True]
  |
  v
positional:  RoPE rotates mu (and sigma when rope_full_gauge != 'off'), or phi gauge composition
  |
  v
FOR EACH LAYER (GaugeTransformerBlock):
    variational FFN E-step (VariationalFFNDynamic):  [the entire block; computes its own beta]
        recompute beta_ij from current beliefs
        grad_F = alpha * grad D_KL(q || p)  +  sum_j beta_ij * grad E_ij  +  (attention entropy)
        natural gradient:  delta = Sigma * grad_F
        mu    <- mu    - self.lr  * delta_mu
        sigma <- retract(sigma, -sigma_lr * delta_sigma, trust region)
        phi   <- Lie-retract(phi, -lr * precond(grad_phi))        [geometric preconditioning]
    norm (MahalanobisNorm or RMSNorm)
  |
  v
decode:  logits = -D_KL(q || pi_v) / tau   [use_prior_bank=True]   or   linear projection mu -> vocab
```

The natural-gradient preconditioning by `Sigma` makes the update a Fisher-metric descent rather than a raw Euclidean step. The μ retraction step size is `self.lr` and the σ step size is `sigma_lr`, independently learnable via the `raw_sigma_lr` Parameter in `VariationalFFNDynamic`; both follow the same cosine decay. The prior covariance `sigma_p` is detached inside the E-step, enforcing the fast-belief / slow-model timescale separation. The `em_mode` selector controls gradient flow at the EM boundary (below).


## Theoretical framework

### GL(K) gauge invariance

The Kullback–Leibler divergence between two Gaussians is invariant under the full general linear group, not merely its orthogonal subgroup (`thm:glk_invariance`, `eq:glk_invariance`):

```
D_KL(Omega_* P || Omega_* Q) = D_KL(P || Q)    for any invertible Omega in GL(K).
```

The `(det Omega)^2` factors cancel in the log-determinant ratio (`GL(K)_attention.tex:552`), so transport operators need only be invertible, never orthogonal, and no re-orthogonalization is required. The result extends to the entire f-divergence family. A consequence emphasized in the manuscript is that the learned query and key maps of a standard transformer are themselves gauge transformations: the per-head bilinear form satisfies `W_Q W_K^T = sigma^{-2} Omega^{-T}` (`GL(K)_attention.tex:1244`). This is an identification at the level of the invertible head-space bilinear, not a parameter-level identity between the rectangular `W_Q` and `W_K` matrices (qualified at `GL(K)_attention.tex:1250-1259`).

### Recovering standard attention

Gauge transport is `Omega_ij = exp(phi_i) exp(-phi_j)`, which covers GL+(K) as a product of two matrix exponentials (`eq:gauge_frame_rotation`). Covariances transport by the sandwich product `Sigma -> Omega Sigma Omega^T` (`eq:gauge_action_gaussians`); transporting a covariance without the conjugation is the single most common correctness bug in this codebase. Under isotropic covariances alone, the KL between transported Gaussians is

```
D_KL = S(Omega_ij) + (1 / 2 sigma^2) || Omega_ij^{-1} mu_i - mu_j ||^2,
```

where the geometric bias (`eq:geometric_bias`) is

```
S(Omega) = (1/2) [ log det(Omega Omega^T) + Tr((Omega Omega^T)^{-1}) - d_k ].
```

The bias vanishes if and only if `Omega` is orthogonal, since `Omega Omega^T = I` gives `S = 0`. Isotropy alone is therefore insufficient to recover dot-product attention: the pair-dependent geometric bias survives. Two further specializations are needed. A constant gauge `Omega_ij = Omega` makes `S(Omega)` a shared constant that cancels in the softmax, and a constant-key-norm condition `|| mu_j ||^2 ≈ C` removes the remaining norm term. Layer normalization is one mechanism that imposes the constant-key-norm condition (high-dimensional concentration is another), which is the sense in which the manuscript frames it as the geometric condition for frame-independent inference. Composing the three specializations recovers `beta_ij = softmax(Q_i K_j^T / sqrt(d_k))` (`GL(K)_attention.tex`, §3.6).

The manuscript is careful about what it does and does not derive. It interprets the training loss and its gradients as variational quantities, but it does not claim to derive backpropagation itself: the chain rule is a general property of differentiable composition (`GL(K)_attention.tex:68`).

### The nonlinearity

A standard transformer applies an ad hoc pointwise nonlinearity (GELU). Here the nonlinearity emerges from differentiating the softmax attention (`eq:softmax_gradient_nonlinearity`):

```
d beta_ij / d mu_i = -(beta_ij / tau) [ d E_ij / d mu_i  -  sum_k beta_ik * d E_ik / d mu_i ],
```

and analogously for `Sigma` and `phi`. The belief update is thus a self-gated message-passing step whose gate is the attention distribution, with no separately parameterized activation.

### Multi-timescale dynamics and symmetry breaking

The free energy separates into a fast E-step (belief inference within a forward pass) and a slow M-step (parameter learning by backpropagation across passes), with the prior covariance frozen relative to the E-step. Standard transformers operate in the adiabatic limit where the slow variables are held fixed during inference. Without observations, the free energy has a gauge-symmetric vacuum in which all agents converge to a common belief modulo the gauge orbit; observations break this symmetry explicitly (not spontaneously), driving agents toward specialized representations (`GL(K)_attention.tex`, §3.5).


## Mechanisms in `transformer/core/`

The gauge group structure is declared in `BlockConfig` (`transformer/core/block_config.py`), the single source of truth for generators, irrep dimensions, and head count; training hyperparameters live in `TrainingConfig`.

Multi-head attention is the block-diagonal restriction `G = GL(d_head)^H` of the full gauge group, realized structurally through the irrep decomposition `irrep_spec` rather than a toggle (`GL(K)_attention.tex`, §3.9.1). Cross-head coupling adds a sparse off-diagonal `gl(K)` subspace through `cross_couplings`: selected head pairs are merged into super-blocks (`GL(K)_attention.tex`, §3.9.3).

Rotary position embeddings are treated as the restriction of the gauge group to `SO(2)^{K/2}` (`GL(K)_attention.tex`, §3.10). The flag `use_rope` enables them and `rope_full_gauge` is a tri-state selector: `'off'` rotates only the mean (the standard-transformer convention), while `'vfe_only'` and `'both'` also apply the sandwich product to the covariance. Under `diagonal_covariance` the non-`'off'` settings lift σ to full and the config validator warns about the cost; see the documented RoPE × MahalanobisNorm limitation in `CLAUDE.md`.

Numerical regularization of small SPD matrices defaults to a uniform Tikhonov ridge, which is not gauge-covariant because the identity does not transform as `Sigma -> h Sigma h^T`. Setting `gauge_covariant_ridge=True` replaces the ridge with `epsilon * (g g^T)`, where `g = exp(phi)` is the per-token frame, restoring covariance.

The gauge frame `phi` lives in `gl(K)` and requires geometric preconditioning because the backward pass through the matrix exponential amplifies non-compact directions; the preconditioners (clip / Cartan / Killing) are in `transformer/core/gauge_preconditioner.py`. Belief decoding through the prior bank computes `logits = -D_KL(q || pi_v) / tau` with a learnable decode temperature, replacing the linear output head when `use_prior_bank=True`.

The `em_mode` field (`transformer/core/em_modes.py`) selects gradient flow at the EM boundary: `'ift_phi'` (default) keeps `mu`, `sigma`, and `phi` attached through a single implicit-differentiation step; `'em_phi_q'` and `'em_phi_p'` detach the prior and treat `phi` as an E-step or M-step quantity respectively. The reflection-augmented O(K) transport and the opt-in non-flat connection (`transformer/core/connection.py`) are research variants.


## Results

The manuscript reports the headline language-modeling result on WikiText-103 (GPT-2 BPE, vocabulary 50,257), training with only KL divergences, gauge transport, and natural-gradient dynamics (`tab:glk_results`, `GL(K)_attention.tex`).

| Configuration | K | Layers | Heads | Params | Val PPL | Test PPL |
|---|---|---|---|---|---|---|
| Gauge VFE, GL(15) | 90 | 1 | 6 | 81.4M | 69.3 | **71.6** |
| Standard transformer, embedding-matched (d=90) | — | 1 | — | 4.6M | 97.2 | 118.6 |
| Modified KN-5 (matched BPE) | — | — | — | — | — | 134.8 |

The single-layer gauge model improves on the embedding-matched standard transformer by 1.66× and on the KN-5 baseline by 1.88×. Random-chance perplexity is roughly 50,000. Learned gauge frames develop interpretable categorical structure (punctuation, content words, and letters separate in both belief space `mu` and frame space `phi`) without category supervision.

The `transformer/core/` `GaugeTransformerLM` has also reported approximately test perplexity 61 on WikiText-103 (BPE-2, K=90, GL(15), 6 heads, RoPE, sequence length 128, one epoch) and approximately 24 on wiki-ja, with an earlier K=80, GL(10) single-layer configuration reporting train/test perplexity near 63/76 at roughly 50M parameters.


## Installation

Requirements: Python ≥ 3.10 with a CUDA-capable GPU recommended. Dependencies are declared in `pyproject.toml`.

```bash
git clone <repository-url>
cd V13_Gauge_Transformer
pip install -e .            # core dependencies
pip install -e .[viz]       # + matplotlib, seaborn, plotly
pip install -e .[all]       # + scikit-learn, umap-learn, shap, pymc, networkx
```

Core dependencies are `torch>=2.1.0`, `pytorch-lightning>=2.2.0`, `torchmetrics`, `numpy`, `tiktoken`, `datasets`, `tqdm`, and `scipy`. Optional extras add experiment tracking (`wandb`), visualization, and analysis packages.


## Usage

Entry points follow the click-to-run pattern: edit the config dictionary at the top of the file, then run it. The sole sanctioned command-line entry point is `train_publication.py`, which exposes a `--mode` selector for publication runs.

```bash
# Publication training (standard / em / hebbian / standard_attn_only / hybrid)
python -m transformer.train_publication --mode em

# Click-to-run training (edit the config dict in the file first)
python -m transformer.train

# Resume from a checkpoint
python -m transformer.resume_training

# Text generation and inference against a GaugeTransformerLM checkpoint
python generate.py
python inference.py
```

```bash
pytest tests/                                  # full suite
pytest tests/transformer/test_model.py         # GaugeTransformerLM
pytest tests/ -m "not slow"                    # skip slow tests
```


## Project structure

```
V13_Gauge_Transformer/
├── transformer/
│   ├── core/                      # GaugeTransformerLM stack + shared gauge/VFE math
│   │   ├── model.py               #   GaugeTransformerLM (encode -> blocks -> decode)
│   │   ├── blocks.py              #   GaugeTransformerBlock, MahalanobisNorm, RMSNorm
│   │   ├── block_config.py        #   BlockConfig — gauge group structure
│   │   ├── variational_ffn.py     #   VariationalFFNDynamic (natural-gradient E-step)
│   │   ├── attention.py           #   Transport + KL-attention kernels
│   │   ├── prior_bank.py          #   PriorBank (token priors + KL decode)
│   │   ├── em_modes.py            #   em_mode gradient-flow selector
│   │   ├── gauge_preconditioner.py#   phi preconditioning (clip / Cartan / Killing)
│   │   ├── connection.py          #   Opt-in non-flat connection (research)
│   │   ├── transport_ops.py       #   Sandwich transport, retraction utilities
│   │   ├── kl_computation.py      #   Gaussian KL kernels
│   │   ├── vfe_gradients.py       #   Analytic gradient kernels
│   │   └── types.py               #   BeliefState
│   ├── training/                  # Training infrastructure (experiment_runner, config)
│   ├── analysis/                  # Holonomy, scaling, semantics, publication metrics
│   ├── visualization/             # Plotting utilities
│   ├── data/                      # datasets.py (WikiText, wiki-ja), synthetic_gauge.py
│   ├── baselines/                 # Standard transformer + FLOPs counter
│   └── utils/                     # Checkpoint, evaluation
├── math_utils/                    # Generators, transport, push-pull, numerical helpers
├── scripts/                       # Analysis and verification scripts
├── tests/                         # Test suite
├── Attention/                     # Manuscripts (GL(K)_attention, supplementary, PIFB)
├── train_publication.py           # see transformer/train_publication.py (CLI entry)
├── generate.py                    # Text generation
├── inference.py                   # Inference
├── pyproject.toml                 # Dependency and packaging declaration
└── CLAUDE.md                      # Architecture reference and code standards
```


## Numerical stability

Large gauge groups are kept stable by scaling the attention logits as `-E_ij / (kappa * sqrt(K))` to prevent softmax saturation, clamping the KL at a dimension-dependent ceiling, applying per-parameter trust regions on the `(mu, sigma, phi)` updates, and transporting through matrix exponentials that are guaranteed to have positive determinant (so no re-orthogonalization is needed; only invertibility is required). The σ retraction applies a trust region to the whitened step `delta_sigma / sigma`, and the SPD eigenvalue path uses a gap-regularized backward to avoid degenerate-eigenvalue gradient blow-up.


## Research variants

`GaugeTransformerLM` is configuration-driven through a large `BlockConfig` surface. Beyond the default amortized `ift_phi` path it supports deep-equilibrium implicit differentiation, closed-form and Hebbian belief updates, the reflection-augmented O(K) transport, and an opt-in non-flat edge-local connection (`transformer/core/connection.py`), whose holonomy is the subject of the companion paper. The GL(K) manuscript proves that the default flat connection has vanishing holonomy (Lemma `thm:vanishing_holonomy`, `GL(K)_attention.tex:641`).


## Citation

```bibtex
@article{dennis2026attention,
  title   = {Attention as Gauge-Theoretic Variational Inference},
  author  = {Dennis, Robert C.},
  journal = {Preprint},
  year    = {2026}
}
```

The companion paper *Participatory Realization: It from Bit* (`Attention/Participatory_it_from_bit.tex`) develops the hierarchical two-channel free energy and the non-flat holonomy regime.


## License

This project is for research purposes. Contact: cdenn016@gmail.com
