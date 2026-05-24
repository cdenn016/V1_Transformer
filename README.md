# Gauge-Theoretic VFE Transformer

A research framework implementing gauge-covariant variational free energy (VFE) minimization for language modeling. The model carries no neural-network components: all representational capacity comes from iterative VFE minimization over Gaussian belief tuples `(mu, Sigma, phi)`. This codebase accompanies the manuscript

> **Attention as Gauge-Theoretic Variational Inference**
> Robert C. Dennis
> (`Attention/GL(K)_attention.tex`, with derivations in `Attention/GL(K)_supplementary.tex`)

The canonical implementation lives in the `transformer/vfe/` package, a clean single-E-step formulation of the theory. The older `transformer/core/` and `transformer/pure_vfe/` packages are retained as research variants and are summarized at the end of this document.


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

The `transformer/vfe/` package realizes the single-channel theory directly. Its E-step minimizes three terms of the free energy: the self-coupling `alpha * D_KL(q_i || p_i)`, the alignment coupling `sum_j beta_ij * D_KL(q_i || Omega_ij q_j)`, and the attention-entropy term `tau * beta_ij * log(beta_ij / pi_j)`. The model-coupling channel and the hyper-prior term are not part of this package; they belong to the companion-paper extension. The non-flat connection is available as an opt-in research feature (`use_non_flat_transport`, `transformer/vfe/non_flat.py`) but is off by default and has one documented gap (below).


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

The `transformer/vfe/` forward pass is a faithful realization of one variational E-step per layer, with no separate attention sublayer. Attention weights are recomputed inside the E-step at every inner iteration from the current beliefs, so there is no `skip_attention` toggle and no message-aggregation residual.

```
token_ids
  |
  v
encode:  (mu, sigma, phi) <- VFEPriorBank.encode(token_ids)
  |
  v
positional:  phi <- BCH-compose(phi_token, phi_pos)        [position is gauge composition, not an additive feature]
  |
  v
[if gauge_parameterization == 'omega_direct']: initialize per-block (Omega, Omega^{-1}) from phi
  |
  v
FOR EACH LAYER (VFEBlock):
    E-step (t = 1 .. n_e_steps):                            [VFEEStep]
        build transport Omega_ij = exp(phi_i) exp(-phi_j)
        recompute beta_ij = softmax(-E_ij / tau)            [per-head, tau = kappa * sqrt(d_head)]
        grad_F = alpha * grad D_KL(q || p)                  [self-coupling]
               + sum_j beta_ij * grad E_ij                  [alignment]
               + (attention-entropy contribution)
        natural gradient:  delta = Sigma * grad_F
        mu  <- mu  - e_mu_lr  * delta_mu        (optional trust region)
        sigma <- retract(sigma, -e_sigma_lr * delta_sigma, trust = e_sigma_q_trust)
        phi <- Lie-retract(phi, -e_phi_lr * precond(grad_phi))   [Killing-preconditioned; skipped if e_phi_lr == 0]
    optional equivariant head mixer (Schur commutant)       [use_equivariant_head_mixer]
    optional norm
  |
  v
cross-layer prior handoff:  next prior mu  <- prior_handoff_rho   blends posterior mu
                            next prior sigma <- prior_handoff_sigma blends posterior sigma
  |
  v
final norm
  |
  v
decode:  logits = -D_KL(q || pi_v) / tau   [use_prior_bank=True]   or   linear projection mu -> vocab
```

The natural-gradient preconditioning by `Sigma` is what makes the update a Fisher-metric descent rather than a raw Euclidean step. The three E-step learning rates are decoupled: `e_mu_lr` (config line 69), `e_sigma_lr` (line 70), and `e_phi_lr` (line 96), with the covariance trust region `e_sigma_q_trust` (line 71) clamping the whitened step `delta_sigma / sigma`. The prior covariance `sigma_p` is detached inside the E-step, enforcing the fast-belief / slow-model timescale separation.


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


## Mechanisms in `transformer/vfe/`

Each mechanism below is controlled by a field of `VFEConfig` (`transformer/vfe/config.py`); line numbers refer to that file.

Multi-head attention is the block-diagonal restriction `G = GL(d_head)^H` of the full gauge group, realized structurally through the irrep decomposition `irrep_spec` rather than a toggle (`GL(K)_attention.tex`, §3.9.1). Cross-head coupling adds a sparse off-diagonal `gl(K)` subspace through `cross_couplings` (line 303): selected head pairs are merged into super-blocks, and the basis is generically not closed under the matrix commutator. The startup check `validate_cross_head_closure` (default on) warns when the supplied basis is not a Lie subalgebra; `auto_close_cross_head_basis` (line 310) closes it under brackets at the cost of changing `phi_dim` and breaking checkpoint compatibility. The off-diagonal mixing construction is `GL(K)_attention.tex`, §3.9.3.

Rotary position embeddings are treated as the restriction of the gauge group to `SO(2)^{K/2}` (`GL(K)_attention.tex`, §3.10). The flag `use_rope` (line 220) enables them and `rope_full_gauge` (line 222) is a tri-state selector: `'off'` rotates only the mean (the standard-transformer convention), while `'vfe_only'` and `'both'` also apply the sandwich product to the covariance inside the E-step. The non-`'off'` settings require `diagonal_covariance=False`.

Numerical regularization of small SPD matrices defaults to a uniform Tikhonov ridge, which is not gauge-covariant because the identity does not transform as `Sigma -> h Sigma h^T`. Setting `gauge_covariant_ridge=True` (line 213) replaces the ridge with `epsilon * (g g^T)`, where `g = exp(phi)` is the per-token frame, restoring covariance.

The gauge frame `phi` lives in `gl(K)` and requires geometric preconditioning because the backward pass through the matrix exponential amplifies non-compact directions. The `phi_preconditioner` field (line 150) selects among `'clip'`, `'cartan'`, `'killing'`, and `'killing_per_block'`; the exact-pullback metric is described in the supplementary (App. C.4) but is rejected at runtime in this package (config validation, lines 609–621), so it is documentation rather than a reachable code path.

By default the gauge frame is parameterized through `phi` and transport is rebuilt each iteration. Setting `gauge_parameterization='omega_direct'` (line 277) switches to a canonical group-level retraction `Omega_new = Omega * exp(-eta * X)` on GL+(K), with `X` the projected pullback of `dF/dOmega` (`transformer/vfe/omega_direct.py`). This path requires `diagonal_covariance=True` (lines 579–585). The exact full-covariance decode `exact_full_cov_decode` requires the opposite, `diagonal_covariance=False` (lines 503–507), so the two canonical forms cannot coexist in a single configuration; each is independently reachable under its own toggle. Lifting the joint form would require implementing the open per-pair sandwich-KL kernel (`transformer/vfe/non_flat.py:483-487`).

Non-flat transport relaxes the flat-bundle assumption through an edge-local connection `Omega_ij = exp(phi_i) exp(delta_ij) exp(-phi_j)`, with `delta_ij` a zero-initialized antisymmetric bilinear form on `(mu_i, mu_j)`. It is enabled by `use_non_flat_transport` (line 335). The full-covariance per-pair path is not yet wired (the `NotImplementedError` above); the diagonal path is functional. The GL(K) manuscript proves that the default flat connection has vanishing holonomy (Lemma `thm:vanishing_holonomy`, `GL(K)_attention.tex:641`); the non-flat regime is the subject of the companion paper.

The self-coupling weight can be made adaptive through `E_learnable_alpha` (line 100), giving `alpha = c0 / (b0 + KL)` per latent dimension. The product-rule contribution from differentiating `alpha * KL` is included in the gradient kernel. This per-dimension Bayesian-precision generalization is a research-track feature and is not part of the manuscript derivation.


## Generation by active inference

Beyond likelihood decoding, the model can select continuations by minimizing expected free energy. The depth-1 policy `VFEExpectedFreeEnergy` (`transformer/vfe/efe.py`) scores each candidate next token by `G(a) = risk + ambiguity - epistemic`, then samples `q(a) ∝ exp(-gamma * G(a))`; it is reachable through `VFEModel.generate(use_efe=True)`. The standalone `transformer/aif/` package generalizes this to horizon-`D` policies through a beam tree search over candidate continuations, with the depth-1 case reproducing `vfe/efe.py`. Generation entry points are click-to-run: `transformer/aif/train_aif.py` runs generation (despite its name) and `transformer/aif/train_aif_augmented.py` adds a trajectory-as-policy EFE term to the training loss.


## Results

The manuscript reports the headline language-modeling result on WikiText-103 (GPT-2 BPE, vocabulary 50,257), training with only KL divergences, gauge transport, and natural-gradient dynamics (`tab:glk_results`, `GL(K)_attention.tex`).

| Configuration | K | Layers | Heads | Params | Val PPL | Test PPL |
|---|---|---|---|---|---|---|
| Gauge VFE, GL(15) | 90 | 1 | 6 | 81.4M | 69.3 | **71.6** |
| Standard transformer, embedding-matched (d=90) | — | 1 | — | 4.6M | 97.2 | 118.6 |
| Modified KN-5 (matched BPE) | — | — | — | — | — | 134.8 |

The single-layer gauge model improves on the embedding-matched standard transformer by 1.66× and on the KN-5 baseline by 1.88×. Random-chance perplexity is roughly 50,000. Learned gauge frames develop interpretable categorical structure (punctuation, content words, and letters separate in both belief space `mu` and frame space `phi`) without category supervision.

### Current `transformer/vfe/` runs

These are the numbers produced by the canonical `vfe/` package. Fill in from your own runs.

| Configuration | K | Layers | Heads | Seq len | Train PPL | Test PPL |
|---|---|---|---|---|---|---|
| `vfe/` (WikiText-103) | __ | __ | __ | __ | __ | __ |
| `vfe/` (wiki-ja) | __ | __ | __ | __ | __ | __ |

### Legacy `transformer/core/` runs

For reference, the earlier `transformer/core/` (`GaugeTransformerLM`) implementation reported approximately test perplexity 61 on WikiText-103 (BPE-2, K=90, GL(15), 6 heads, RoPE, sequence length 128, one epoch) and approximately 24 on wiki-ja, with an earlier K=80, GL(10) single-layer configuration reporting train/test perplexity near 63/76 at roughly 50M parameters. These values come from the prior implementation and are not directly comparable to the `vfe/` package; they are recorded here for continuity.


## Installation

Requirements: Python ≥ 3.10 with a CUDA-capable GPU recommended. Dependencies are declared in `pyproject.toml`.

```bash
git clone <repository-url>
cd V13_Gauge_Transformer
pip install -e .            # core dependencies
pip install -e .[viz]       # + matplotlib, seaborn, plotly
pip install -e .[all]       # + scikit-learn, umap-learn, shap, pymc, networkx
```

Core dependencies are `torch>=2.1.0`, `pytorch-lightning>=2.2.0`, `torchmetrics`, `numpy`, `tiktoken`, `datasets`, `tqdm`, and `scipy`. Optional extras add experiment tracking (`wandb`), a tokenizer for active-inference generation (`transformers`), visualization, and analysis packages.


## Usage

Entry points follow the click-to-run pattern: edit the config dictionary at the top of the file, then run it. There are no command-line arguments.

```bash
# Train the canonical VFE model (edit the config dict in the file first)
python -m transformer.vfe.train_vfe

# Hyperparameter sweeps over VFEConfig (one field at a time)
python -m transformer.vfe.vfe_ablation_suite

# Semantic-clustering visualization of mu / Sigma / phi / Omega
python -m transformer.vfe.run_semantic_clustering

# Active-inference generation, then training-time EFE augmentation
python -m transformer.aif.train_aif
python -m transformer.aif.train_aif_augmented
```

The root-level `generate.py` and `inference.py` scripts target the legacy `transformer/core/` `GaugeTransformerLM` and its checkpoints, not the `vfe/` package.

```bash
pytest tests/                                  # full suite
pytest tests/transformer/test_vfe_package.py   # vfe/ package
pytest tests/transformer/test_aif_package.py   # aif/ package
pytest tests/ -m "not slow"                    # skip slow tests
```


## Project structure

```
V13_Gauge_Transformer/
├── transformer/
│   ├── vfe/                       # Canonical VFE transformer (this README's focus)
│   │   ├── config.py              #   VFEConfig — single source of truth
│   │   ├── model.py               #   VFEModel (encode -> E-step stack -> decode)
│   │   ├── stack.py               #   VFEStack (layer loop + cross-layer prior handoff)
│   │   ├── block.py               #   VFEBlock (E-step + optional head mixer + norm)
│   │   ├── e_step.py              #   VFEEStep (natural-gradient belief inference)
│   │   ├── prior_bank.py          #   VFEPriorBank (token priors + KL decode)
│   │   ├── positional.py          #   Position as BCH gauge composition
│   │   ├── attention.py           #   Stateless transport + KL-attention kernels
│   │   ├── head_mixer.py          #   Schur-commutant equivariant head mixer
│   │   ├── non_flat.py            #   Opt-in edge-local connection / holonomy
│   │   ├── omega_direct.py        #   Group-level GL+(K) retraction
│   │   ├── efe.py                 #   Depth-1 expected-free-energy generation
│   │   ├── trainer.py             #   VFETrainer (AdamW + cosine schedule)
│   │   ├── train_vfe.py           #   Click-to-run training entry point
│   │   ├── vfe_ablation_suite.py  #   Click-to-run hyperparameter sweeps
│   │   ├── run_semantic_clustering.py
│   │   └── semantic_clustering/   #   Clustering of mu/Sigma/phi/Omega into figures
│   ├── aif/                       # Active-inference generation (wraps VFEModel)
│   │   ├── efe_score.py           #   Expected free energy (risk/ambiguity/epistemic)
│   │   ├── tree_search.py         #   Horizon-D beam search over policies
│   │   ├── generator.py           #   AIFGenerator
│   │   ├── train_aif.py           #   Click-to-run generation
│   │   └── train_aif_augmented.py #   Training with EFE-augmented loss
│   ├── core/                      # Legacy GaugeTransformerLM stack + shared math
│   ├── pure_vfe/                  # Legacy no-autograd PureVFETransformer
│   ├── analysis/                  # Holonomy, scaling, semantics, publication metrics
│   ├── visualization/             # Plotting utilities
│   ├── data/                      # datasets.py (WikiText, wiki-ja), synthetic_gauge.py
│   ├── training/                  # Legacy training infrastructure (core/-oriented)
│   ├── baselines/                 # Standard transformer + FLOPs counter
│   └── utils/                     # Checkpoint, evaluation (core/-oriented)
├── math_utils/                    # Generators, transport, push-pull, numerical helpers
├── scripts/                       # Analysis and verification scripts (core/-oriented)
├── tests/                         # Test suite (includes test_vfe_package, test_aif_package)
├── Attention/                     # Manuscripts (GL(K)_attention, supplementary, PIFB)
├── generate.py                    # Legacy core/ text generation
├── inference.py                   # Legacy core/ inference
├── pyproject.toml                 # Dependency and packaging declaration
└── CLAUDE.md                      # Architecture reference and code standards
```


## Numerical stability

Large gauge groups are kept stable by scaling the attention logits as `-E_ij / (kappa * sqrt(K))` to prevent softmax saturation, clamping the KL at a dimension-dependent ceiling, applying per-parameter trust regions on the `(mu, sigma, phi)` updates, and transporting through matrix exponentials that are guaranteed to have positive determinant (so no re-orthogonalization is needed; only invertibility is required). The covariance trust region `e_sigma_q_trust` clamps the whitened step `delta_sigma / sigma`, and the opt-in helpers in `transformer/vfe/_numerics.py` add NaN sentinels and a pre-exponential Frobenius clamp without changing the default numerical path.


## Legacy and research variants

The `transformer/core/` package is the original `GaugeTransformerLM` implementation, a configuration-dictionary-driven model with a large `BlockConfig` surface, the five-mode `em_mode` gradient-flow selector, deep-equilibrium implicit differentiation, closed-form and Hebbian variants, and the optional reflection-augmented O(K) transport. The `vfe/` package deliberately drops all of these in favor of a single E-step path, but it imports the stateless mathematical kernels from `core/` (transport, KL computation, the shared `BeliefState` type, preconditioners, and retraction utilities), so `core/` is a dependency of `vfe/` and is not removable.

The `transformer/pure_vfe/` package is an independent implementation with no `nn.Module`, no autograd, and no backpropagation: the model is a bank of Gaussian priors, the forward pass is the E-step, and learning is an analytic natural-gradient M-step on the priors and gauge frames. It serves as the most literal expression of the variational principle and is configured through its own `PureVFEConfig`.


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
