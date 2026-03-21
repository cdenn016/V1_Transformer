# Gauge-Theoretic VFE Transformer

Gauge-covariant variational free energy transformer for language modeling. No neural network components — all representational capacity from iterative VFE minimization over Gaussian belief tuples `(mu, Sigma, phi)`. See `README.md` for theoretical framework, experimental results, and installation.

## Hard Constraints

**NO NEURAL NETWORKS**: No `nn.Linear`, no MLPs, no learned W_Q/W_K/W_V projections, no activation functions (GELU, ReLU, etc.). The only retained neural component is a linear output projection from K dimensions to vocabulary size. If you are tempted to add an MLP or activation function, you are violating the core thesis.

**NO CLI ARGUMENTS**: Entry points use the click-to-run pattern. Edit config dicts directly in the file, then press Run. Do not add `argparse`, `click`, `typer`, or any CLI flag parsing.

**PRESERVE GAUGE EQUIVARIANCE**: Covariance transport must always use the sandwich product: `Sigma_transported = Omega @ Sigma @ Omega.T`. Never transport covariance without the conjugation. This is the single most common correctness bug.

## Codebase Map

### Entry Points


- `generate.py` — Interactive text generation with attention visualization
- `inference.py` — Model inference and token probability analysis
- `transformer_test.py` — BERT 144-head KL-alignment validation
- `roberta_diagnostics.py` — RoBERTa diagnostic analysis
- 'train_publication.py' - the primary training entry point

### `transformer/core/` — Architecture

- `model.py` — `GaugeTransformerLM` (main language model)
- `attention.py` — `IrrepMultiHeadAttention` (KL-divergence attention, no W_Q/W_K)
- `blocks.py` — `GaugeTransformerBlock` (attention + VFE FFN + residual)
- `variational_ffn.py` — `VariationalFFNDynamic` (VFE E-step belief evolution)
- `embeddings.py` — `GaugeTokenEmbedding`, `GaugePositionalEncoding`
- `gauge_utils.py` — Matrix exp pairs, Newton-Schulz orthogonalization, block-diagonal KL
- `gauge_preconditioner.py` — Phi gradient preconditioning (4 modes: clip, cartan, killing, pullback)
- `prior_bank.py` — `PriorBank` (unified embedding + output projection via KL)
- `connection.py` — `GaugeConnection` (non-flat parallel transport, holonomy experiments)
- `block_config.py` — `BlockConfig` dataclass (single source of truth for gauge group structure)
- `triton_kernels.py` — GPU-accelerated pairwise KL computation

### `transformer/analysis/` — Diagnostics

- `rg_metrics.py` — `RGDiagnostics` (modularity, effective rank, KL clustering)
- `rg_flow_analysis.py`, `rg_flow_enhanced.py` — RG flow dynamics across training
- `holonomy.py`, `holonomy_metrics.py` — Gauge connection curvature measurement
- `semantics.py` — Gauge frame clustering analysis
- `bayesian_validation.py` — BERT KL-alignment hypothesis testing
- `publication_metrics.py` — Publication-quality metrics and figures
- `trajectory.py` — E-step belief trajectory recording

### `transformer/training/` — Training Infrastructure

- 'train_publication.py' - primary training entrypoint.  
- `config.py` — `TrainingConfig` dataclass (single source of truth for hyperparameters)
- `optimizer.py` — Parameter-group-aware AdamW with per-type learning rates
- `lightning_module.py` — `GaugeTransformerLitModule` (PyTorch Lightning wrapper)
- `lightning_data.py` — `GaugeDataModule` (Lightning data pipeline)
- `lightning_pure_vfe.py` — Lightning module for pure VFE variant
- `metrics.py` — Loss, perplexity, BPC, VFE term breakdown
- `train_fast.py` — Optimized training loop with gradient accumulation

### `transformer/pure_vfe/` — Pure VFE (no nn.Module, no autograd)

- `model.py` — `PureVFETransformer` (raw tensor operations)
- `config.py`, `inference.py` (E-step), `learning.py` (M-step), `gauge.py`, `gaussians.py`
- `tests/test_gradients.py` — Finite-difference gradient validation

### Other `transformer/` Subdirectories

- `visualization/` — Attention heatmaps, belief space plots, training curves, trajectory plots
- `data/` — `datasets.py` (WikiText-2/103, OpenWebText loaders), `synthetic_gauge.py`
- `baselines/` — `standard_transformer.py` (dot-product attention baseline), `flops_counter.py`
- `utils/` — `checkpoint.py`, `evaluation.py`, `testing.py`
- `train.py` — Standard training loop
- `resume_training.py` — Checkpoint resume

### `math_utils/` — Mathematical Primitives

- `generators.py` — Lie algebra generators: SO(3), SO(N), GL(K), multi-irrep block-diagonal
- `transport.py` — Parallel transport: `Omega_ij = exp(phi_i) * exp(-phi_j)`
- `push_pull.py` — Gaussian pushforward under transport: `(mu, Sigma) -> (Omega @ mu, Omega @ Sigma @ Omega.T)`
- `numba_kernels.py` — Numba JIT: KL, Rodrigues, transported KL
- `cuda_kernels.py` — Optional CuPy CUDA kernels
- `numerical_utils.py` — Regularization, gradient clipping, condition estimates

### `tests/` — Test Suite

14+ test files with shared fixtures in `conftest.py`. Markers: `@pytest.mark.slow`, `@pytest.mark.gpu`, `@pytest.mark.integration`.

### `scripts/` — Analysis & Experiments

RG flow analysis, ablation suites, publication figure generation, spectral analysis, `train_lightning.py`.

## Development Workflow

### Testing

```bash
pytest                              # all tests
pytest -m "not slow"                # skip slow tests
pytest -m "not gpu"                 # CPU-only tests
pytest tests/transformer/           # transformer tests only
pytest transformer/pure_vfe/tests/  # pure VFE gradient tests
```

### Training

- **nn.Module VFE**: Edit config in `transformer/train.py` and run, or use `scripts/train_lightning.py`
- **Publication-quality**: `transformer/train_publication.py` (full RG diagnostics, holonomy monitoring)
- **Pure VFE (no autograd)**: Edit `PURE_VFE_CONFIG` in `run.py` and run
- **Resume**: `transformer/resume_training.py`

### Generation

- `generate.py` — Interactive generation from checkpoint with attention visualization

### Dependencies

Defined in `pyproject.toml`. Core: `torch>=2.1`, `pytorch-lightning>=2.2`, `numpy`, `numba`, `tiktoken`, `datasets`. Optional groups: `wandb`, `transformers`, `viz` (matplotlib, seaborn, plotly), `all`.

## Code Conventions

- Type hints on all function signatures
- Docstrings with LaTeX math where relevant (e.g., `r"""Computes KL(q || p) = ..."""`)
- Variable names match paper notation: `mu_q`, `Sigma`, `phi`, `Omega`, `kappa`, `alpha`, `beta_ij`, `gamma`
- Tensor shape comments at critical points in attention, transport, and KL computations
- `BlockConfig` is the single source of truth for gauge group structure (generators, irrep dims, head count)
- `TrainingConfig` is the single source of truth for training hyperparameters (learning rates, schedules, VFE weights)
- Validate gradient correctness with small-dim finite-difference smoke tests

## Mathematical Reference

Minimal equations for code review — see `README.md` for full derivations.

**VFE hierarchy**: `h → s → p → q → observations` (hyper-prior → models → priors → beliefs → data)

**Free energy**:
```
F = alpha * KL(q_i || p_i)                    # self-coupling: beliefs to priors
  + lambda_h * KL(s_i || h)                   # hyper-prior: models to centroid
  + beta_ij * KL(q_i || Omega_ij * q_j)       # belief coupling (attention)
  + gamma_ij * KL(s_i || Omega_ij * s_j)      # model coupling (meta-cognition)
  - E_q[log p(o | x)]                         # observation likelihood
```

**Attention**: `beta_ij = softmax(-KL(q_i || Omega_ij * q_j) / kappa)`

**Transport**: `Omega_ij = exp(phi_i) * exp(-phi_j)` — covers GL+(K) via product of two exponentials

**Covariance transport**: `Sigma_transported = Omega @ Sigma @ Omega.T` — the sandwich product

**Phi preconditioning**: clip (default, O(n_gen)), cartan (O(n_gen²)), killing (O(n_gen²)), pullback (O(n_gen³)) — see `transformer/core/gauge_preconditioner.py`

**Timescales**: Fast E-step (belief inference per forward pass) / Slow M-step (parameter learning via backprop)

## Communication Style

**Be direct.** State errors and concerns plainly. "This is wrong because X" not "This might potentially be slightly off." Always ultra-think and double check.

**Push back.** Challenge gaps in derivations, ask for justification. If a claim needs proof, ask for it. Maintain position under pushback — ask "What am I missing?" rather than capitulating.

**Skip praise preambles.** No "Great question!" openers. No "Excellent point!" — engage with the substance.

**Flag simpler alternatives.** Call out over-engineering. Ask what the complexity buys if something seems unnecessarily elaborate.

**Honest uncertainty.** "I'm not sure this is right" beats confident speculation. Acknowledge when something needs verification.

**No bullshit.** If a correspondence is interpretive rather than mathematically exact, say so explicitly. If something doesn't connect, admit the gap. Remove content that doesn't earn its place through rigorous derivation. Never dress up hand-waving as theorem. When asked "what does X have to do with anything?" — if the answer is "not much", say that.

## Manuscript Style

Write in academic prose, not bullet points. Use flowing paragraphs with clear logical progression.

**Banned patterns:** horizontal rules (`---`), "key insight", "crucially", "critically", "notably", "importantly", "it's worth noting", "interestingly", "fundamentally", "in particular", "leverages", "underscores". These are Claude-isms — never use them in manuscripts.

Minimize itemizations, lists, and enumerations. If content can be expressed as a paragraph, express it as a paragraph. Remove content that doesn't earn its place through rigorous derivation.

## Available Skills

19 domain-specific skills in `.claude/skills/`, invokable as slash commands:

**Research & Writing**: `/scientific-writing` (IMRAD manuscripts), `/literature-review` (systematic synthesis), `/peer-review` (structured evaluation), `/hypothesis-generation` (testable predictions), `/arxiv-database` (paper search)

**Statistics & Modeling**: `/statistical-analysis` (test selection, APA reporting), `/statsmodels` (OLS, GLM, time series), `/pymc` (Bayesian inference), `/scikit-learn` (ML pipelines)

**Visualization**: `/scientific-visualization` (publication figures), `/plotly` (interactive 3D), `/seaborn` (statistical plots)

**ML & Frameworks**: `/pytorch-lightning` (training infrastructure), `/shap` (feature attribution), `/umap-learn` (dimensionality reduction)

**Math & Computation**: `/sympy` (symbolic math), `/networkx` (graph analysis), `/pennylane` (quantum ML)

## Contributing

1. **Preserve gauge equivariance** — covariance transport is always `Omega @ Sigma @ Omega.T`
2. **Use natural gradients** for phi parameters — never raw Euclidean gradients on Lie algebra without preconditioning
3. **Test RG behavior** — new features should maintain or improve RG trends (modularity, effective rank)
4. **Document math** — include LaTeX notation in docstrings for any non-trivial formula
5. **Domain expertise areas**: differential geometry (SPD manifolds, Lie theory), variational inference (KL, ELBO, information geometry), gauge theory (equivariance, parallel transport, irreps), matrix algebra (eigendecomposition, matrix exponentials)
