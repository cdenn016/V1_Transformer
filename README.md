# Gauge-Theoretic Transformer

A research framework implementing gauge-covariant variational free energy (VFE) minimization for language modeling. This codebase accompanies the manuscript:

> **Attention, Transformers, and Backpropagation are Degenerate Limits of the Variational Free Energy Principle**
> Robert C. Dennis

**Key result:** A single-layer VFE transformer with no MLPs or neural network components
achieves **76 test perplexity** on WikiText-103 (BPE-2 tokenization) with K=80, GL(10),
8 heads, RoPE, and sequence length 128 after 1 epoch of training. All representational
capacity comes from iterative variational free energy minimization over belief tuples (μ, Σ, φ). The E-step *is* the computation---it is not a learned feed-forward network.


## Core Thesis

Language is a dynamic informational system: speakers encode and decode beliefs under uncertainty, and language models learn the statistical structure of this process. The mathematical framework natural to such systems---gauge-covariant variational free energy minimization over communicating agents on a statistical fiber bundle---explains why attention mechanisms work.

Standard transformer attention, the `1/sqrt(d_k)` scaling, layer normalization, and backpropagation all emerge as consequences of a single variational principle. The standard attention rule

```
beta_ij = softmax(Q_i K_j^T / sqrt(d_k))
```

is recovered as a **degenerate limit** of the gauge-theoretic attention

```
beta_ij = softmax(-D_KL(q_i || Omega_ij q_j) / tau)
```

through two simplifications: isotropic covariances and flat bundle.


## Architecture

```
        N(0, 1/(2·wd))  ← Level 3: hyper-prior (weight decay on embeddings)
                │
                │ wd·||θ_embed||²
                ▼
s_i = self.prior_mu[i]  ◄──── γ·KL(s_i||Ω_ij·s_j) ────► s_j
(position MODEL,                    (model coupling)
 slow timescale)
                │                                          ← Level 2: priors (M-step)
                │ p = w·π_token + (1-w)·s
                ▼
          p_i (PRIOR)
                │
                │ α·KL(q||p)
                ▼                                          ← Level 1: beliefs (E-step)
     q_i (beliefs, fast)  ◄──── β_ij·KL(q_i||Ω_ij·q_j) ────► q_j
                                      (attention)
```

Each agent (token) carries a Gaussian belief `q_i = N(mu_i, Sigma_i)` in a local gauge frame `phi_i`. The full VFE follows the standard FEP hierarchy `h -> s -> p -> q -> observations`:

```
F = sum_i  alpha D_KL(q_i || p_i)                         [self-coupling: beliefs to priors]
  + sum_i  lambda_h D_KL(s_i || h)                        [hyper-prior: models to centroid]
  + sum_ij beta_ij D_KL(q_i || Omega_ij q_j)              [belief coupling / attention]
  + sum_ij gamma_ij D_KL(s_i || Omega_ij s_j)             [model coupling / meta-cognition]
  - E_q[log p(o | x)]                                     [observation likelihood]
```

Attention weights emerge from information geometry:

```
beta_ij = softmax_j(-D_KL(q_i || Omega_ij q_j) / tau)
```

where `Omega_ij = exp(phi_i) exp(-phi_j)` is the gauge transport between agents. **No W_Q, W_K, W_V projections are used**---attention arises from the geometry of belief distributions.


## Theoretical Framework

### GL(K) Gauge Invariance (Theorem 1.1)

The KL divergence possesses maximal gauge symmetry: it is invariant under the full general linear group GL(K), not merely orthogonal subgroups.

```
D_KL(Omega_* P || Omega_* Q) = D_KL(P || Q)    for any invertible Omega in GL(K)
```

The (det Omega)^2 factors cancel in the log-determinant ratio. This means:
- Transport operators need only be **invertible**, not orthogonal
- No expensive re-orthogonalization is needed
- Learned projections W_Q, W_K in standard transformers **are themselves gauge transformations**, with `Omega = W_Q W_K^T` serving as a learned gauge transport

### Three Limits Recovering Standard Attention

| Limit | What it discards | Result |
|---|---|---|
| **1. Isotropic covariances** | Non-isotropic Σ_i → σ²I | KL reduces to geometric bias S(Ω) + squared Euclidean distance |
| **2. Constant gauge** | Position-dependent Ω_ij → constant Ω | Global gauge, no curvature; S(Ω) becomes a shared constant |
| **3. Learned projections** | σ⁻² Ω⁻ᵀ absorbed into W_Q W_K^T | Standard Q, K, V projections |

After all three limits: `beta_ij = softmax(Q_i K_j^T / sqrt(d_k))` --- standard transformer attention.

**Important:** Limit 1 alone does not suffice---the isotropic KL under general transport is:

```
D_KL = S(Ω_ij) + (1/2σ²) ||Ω_ij⁻¹ μ_i - μ_j||²
```

where S(Ω) = ½[log det(ΩΩᵀ) + Tr((ΩΩᵀ)⁻¹) - K] is the **geometric bias**. This vanishes if and only if Ω ∈ O(K) (orthogonal group), since ΩΩᵀ = I implies S = 0.

### The Nonlinearity

Standard transformer: GELU(x) — ad hoc nonlinearity.

Ours: ∂β_{ij}/∂θ — emerges from differentiating softmax attention:

```
β_{ij} = softmax(-KL_{ij} / κ)

∂β_{ij}/∂μ_i = -β_{ij} · [∂KL_{ij}/∂μ_i - Σ_k β_{ik} · ∂KL_{ik}/∂μ_i] / κ
∂β_{ij}/∂Σ_i = -β_{ij} · [∂KL_{ij}/∂Σ_i - Σ_k β_{ik} · ∂KL_{ik}/∂Σ_i] / κ
∂β_{ij}/∂φ_i = -β_{ij} · [∂KL_{ij}/∂φ_i - Σ_k β_{ik} · ∂KL_{ik}/∂φ_i] / κ
```

The **most general form** of the theory---no simplifying limits taken. Full non-isotropic covariances, non-trivial gauge transport, KL-divergence attention. **No MLPs, activation functions, learned W_Q/W_K/W_V, or positional encodings.** Only a linear output projection (from K dimensions to 50k vocabulary) is retained.

### Hierarchical Bayesian Structure

The full VFE defines a four-level hierarchy, each regularized by the level above:

```
Level 3:  N(0, 1/(2*wd))                          [hyper-prior — weight decay]
              |
              | wd * ||theta_embed||^2
              v
Level 2:  p_i = N(mu_p, Sigma_p), phi             [priors — learned embeddings (M-step)]
              |
              | alpha * KL(q || p)
              v
Level 1:  q_i = N(mu_q, Sigma_q)                  [beliefs — inferred per forward pass (E-step)]
              |
              | -E_q[log p(x | mu)]
              v
Level 0:  x_i                                      [observations — token targets]
```

**Level 3 (hyper-prior)** is implemented by AdamW weight decay on embedding parameters.
In a standard transformer, embeddings are lookup tables conventionally excluded from
weight decay. In the gauge transformer, embeddings are **statistical parameters**---means
`mu_p`, covariances `Sigma_p`, and gauge frames `phi`---that directly enter KL divergences,
matrix exponentials, and VFE gradients.

### Multi-Timescale Dynamics

The free energy naturally separates into:
- **Fast (E-step)**: Belief inference `dq_i/dt = -eta_q dF_fast/dq_i` --- what transformers do in a forward pass
- **Slow (M-step)**: Model learning `ds_i/dt = -eta_s dF_slow/ds_i` --- what backpropagation updates

Standard transformers operate in the adiabatic limit: slow variables frozen during inference, updated between passes.

### Multi-Head Attention as Block-Diagonal GL(K)

Multi-head attention restricts the full GL(d_k) to a block-diagonal subgroup:

```
G_multi-head = GL(d_head)^H  subset  GL(d_k)
```

Each head learns an independent GL(d_head) gauge transformation. For compact groups like SO(3), irreducible representations yield non-uniform head dimensions (1, 3, 5, 7, ...) with intrinsic geometric meaning.

### O(K) Gauge Transport

The gauge transport Ω_ij = exp(φ_i)·exp(-φ_j) via matrix exponentials always produces det > 0. Under Newton-Schulz orthogonalization (`enforce_orthogonal=True`), this projects to SO(K). To achieve full O(K) transport, the `learnable_reflection` option introduces per-token sign vectors s_i ∈ {±1}^K:

```
Ω_ij = diag(s_i) · exp(φ_i) · exp(-φ_j) · diag(s_j)
```

covering both connected components of O(K). Sign vectors are applied at embedding time via `μ_i ← s_i ⊙ μ_i` with straight-through estimator for gradient flow.

### Phi Gradient Preconditioning

The gauge frame φ ∈ gl(K) requires geometric preconditioning because the backward pass through `matrix_exp` amplifies non-compact (symmetric) directions exponentially. Four modes:

| Mode | Method | Cost |
|------|--------|------|
| `'clip'` (default) | Norm clipping | O(n_gen) |
| `'cartan'` | Cartan decomposition, fixed sym dampening | O(n_gen²) |
| `'killing'` | Killing form metric (no free params) | O(n_gen²) |
| `'pullback'` | Full pullback through exp (exact Riemannian) | O(n_gen³/token) |

For GL(K): `'pullback'` is the theoretically exact natural gradient; `'killing'` is a cheap principled approximation; `'cartan'` is the pragmatic engineering choice. For SO(N) all modes are roughly equivalent (compact group).

### DEQ Implicit Differentiation (E-Step Backward)

The E-step iterates natural gradient VFE descent to a fixed point μ*, Σ*. **Deep Equilibrium (DEQ) implicit differentiation** replaces unrolled backprop with a single backward pass through the fixed-point equation using the implicit function theorem:

```
(I - J)^{-1} v ≈ v + Jᵀv + (Jᵀ)²v + ...    (Neumann series, K terms)
```

Enabled via `use_deq=True`. Provides O(1) memory in iterations vs O(n_iterations) for unrolled backprop.

### Symmetry Breaking

Without observations, the free energy defines a gauge-symmetric vacuum---all agents converge to identical beliefs modulo gauge orbit (analogous to an untrained network). Observations break this symmetry, driving agents toward specialized representations determined by training data. Learning is thus interpreted as explicit symmetry breaking.


## Experimental Results

### BERT Validation (144 attention heads)

Quantitative comparison between gauge-aligned KL attention and standard dot-product attention on pretrained `bert-base-uncased`:

| Metric | Value |
|---|---|
| Optimal temperature | τ = 19.0 (theory: τ = 2√d = 16, 19% deviation) |
| Mean Pearson correlation | r = 0.821 |
| Median Pearson correlation | r = 0.889 |
| Heads with r > 0.8 | 68.1% |
| Heads with r > 0.9 | 49.3% |

**Key-norm bias prediction confirmed**: average ρ = -0.352 across all heads, significant in 92.4% of heads at p < 0.001. This gauge-theoretic prediction explains why layer normalization is a geometric necessity: it enforces constant key norms required for frame-independent inference.

### GL(K) Language Modeling (WikiText-103, vocab 50,257)

The most general form of the theory---no simplifying limits taken. Full non-isotropic covariances, non-trivial gauge transport, KL-divergence attention.

| Configuration | K | Layers | Gauge Mode | Train PPL | Test PPL | Parameters |
|---|---|---|---|---|---|---|
| **Best (1 epoch)** | **80** | **1** | **Ω ∈ GL(10)** | **63** | **76** | **~50M** |

For context: random-chance perplexity is ~50,000. The K=80, GL(10) configuration (8 heads, seq-length 128, RoPE) achieves **~600x improvement** over random chance with purely geometric attention.

**Emergent semantic structure**: Learned gauge frames develop interpretable categorical organization---punctuation, content words, and letters cluster separately in both belief space (μ) and gauge frame space (φ) without category supervision.


## Installation

**Requirements**: Python 3.9+ with CUDA-capable GPU (recommended)

```bash
git clone https://github.com/cdenn016/gauge-holonomy.git
cd gauge-holonomy
pip install torch numpy scipy numba matplotlib seaborn plotly networkx scikit-learn datasets tiktoken
```

### Dependencies

| Category | Packages |
|---|---|
| **Core** | PyTorch (>=2.0.0), NumPy, SciPy, Numba |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Data** | HuggingFace `datasets`, `tiktoken` |
| **Analysis** | scikit-learn, NetworkX |
| **Optional** | SymPy (symbolic derivations), Triton (GPU kernels), CuPy (CUDA kernels), `transformers` (BERT diagnostics) |


## Usage

### Training

```bash
# Standard VFE training with gauge-theoretic attention
python transformer/train.py

# Publication-quality training with full experimental features
python transformer/train_publication.py

# Resume from checkpoint
python transformer/resume_training.py --checkpoint path/to/model.pt
```

### Inference & Generation

```bash
# Interactive text generation with attention visualization
python generate.py

# Model inference and token probability analysis
python inference.py
```

### Analysis

```bash
# BERT validation (gauge-aligned KL vs dot-product attention)
python transformer_test.py

# RoBERTa diagnostics (key-norm CV, per-head temperatures)
python roberta_diagnostics.py

# RG flow analysis over training
python scripts/analyze_rg_flow.py

# Generate publication figures
python scripts/generate_publication_figures.py
```

### Testing

```bash
pytest tests/                           # Full suite
pytest tests/ -v -k "attention"         # Specific tests
pytest tests/ -m "not slow"             # Skip slow tests
pytest tests/ -m gpu                    # GPU-only tests
```


## Project Structure

```
gauge-holonomy/
├── transformer/                    # Gauge-theoretic transformer implementation
│   ├── core/                       #   Core architecture
│   │   ├── model.py                #     GaugeTransformerLM (full language model)
│   │   ├── attention.py            #     KL-divergence multi-head attention
│   │   ├── blocks.py               #     Transformer blocks with gauge transport
│   │   ├── variational_ffn.py      #     VFE feedforward (E-step belief inference)
│   │   ├── embeddings.py           #     Token/positional embeddings → (μ, Σ, φ)
│   │   ├── prior_bank.py           #     Token-dependent prior distributions
│   │   ├── gauge_utils.py          #     matrix_exp, Newton-Schulz, fused KL
│   │   ├── gauge_preconditioner.py #     Riemannian φ preconditioning
│   │   ├── connection.py           #     Gauge connection abstraction
│   │   ├── block_config.py         #     BlockConfig dataclass
│   │   └── triton_kernels.py       #     Triton-optimized KL and matrix exp
│   ├── analysis/                   #   Analysis & diagnostics
│   │   ├── rg_metrics.py           #     RG flow: meta-agents, modularity, effective rank
│   │   ├── rg_flow_analysis.py     #     RG flow tracking across layers/iterations
│   │   ├── rg_flow_enhanced.py     #     Full RG diagnostics
│   │   ├── publication_metrics.py  #     BPC, perplexity, statistical significance
│   │   ├── semantics.py            #     Clustering interpretation, emergent categories
│   │   ├── bayesian_validation.py  #     Bayesian validation of KL alignment
│   │   ├── holonomy.py             #     Holonomy measurement
│   │   └── trajectory.py           #     Belief trajectory tracking
│   ├── visualization/              #   Plotting utilities
│   │   ├── attention_viz.py        #     Attention heatmaps and KL plots
│   │   ├── belief_space_viz.py     #     Belief distribution visualization
│   │   ├── training_plots.py       #     Training curves
│   │   ├── trajectory_plots.py     #     Belief evolution
│   │   └── ablation_plots.py       #     Ablation comparisons
│   ├── data/                       #   Data loading
│   │   ├── datasets.py             #     WikiText-2/103, synthetic gauge language
│   │   └── synthetic_gauge.py      #     Synthetic language with controlled holonomy
│   ├── training/                   #   Training infrastructure
│   │   ├── config.py               #     TrainingConfig dataclass
│   │   ├── train_fast.py           #     Optimized training loop
│   │   ├── optimizer.py            #     Gauge-specific parameter grouping
│   │   └── metrics.py              #     Training metrics
│   ├── baselines/                  #   Reference implementations
│   │   └── standard_transformer.py #     StandardTransformerLM (dot-product attention + MLP)
│   ├── utils/                      #   Checkpoint, evaluation, testing utilities
│   ├── train.py                    #   Main training entry point
│   ├── train_publication.py        #   Publication-quality training
│   └── resume_training.py          #   Resume from checkpoint
├── math_utils/                     # Mathematical primitives
│   ├── generators.py               #   SO(3)/SO(N)/GL(K) Lie algebra generators
│   ├── transport.py                #   Parallel transport Ω_ij = exp(φ_i)·exp(-φ_j)
│   ├── push_pull.py                #   Gaussian pushforward under transport
│   ├── numba_kernels.py            #   Numba JIT kernels (KL, Rodrigues)
│   ├── cuda_kernels.py             #   CuPy CUDA kernels
│   └── numerical_utils.py          #   Stability helpers (regularization, clipping)
├── experiments/                    # Experiment configurations
│   └── configs/
│       └── flat_bundle_configs.py  #   Flat bundle hypothesis test configs
├── derivations/                    # Symbolic mathematical derivations
│   ├── constant_gauge_kl_derivation.py   # KL under constant GL(K) transport (SymPy)
│   ├── analytic_phi_grad_derivation.py   # ∂KL/∂Ω for diagonal covariances
│   └── constant_gauge_kl_derivation.md   # Derivation documentation
├── scripts/                        # Utility scripts
│   ├── analyze_rg_flow.py          #   RG flow analysis
│   ├── generate_publication_figures.py   # Publication figures
│   └── kn5_baseline.py             #   KN5 baseline comparison
├── tests/                          # Test suite
│   ├── transformer/                #   Core tests (attention, model, training, ...)
│   └── experiments/                #   Experiment tests (non-flat transport)
├── Data/                           # Experimental data and baselines
│   └── Standard_Baselines/         #   Standard transformer baseline results
├── generate.py                     # Interactive text generation
├── inference.py                    # Model inference and analysis
├── transformer_test.py             # BERT validation (144 heads)
├── roberta_diagnostics.py          # RoBERTa analysis
├── claude.md                       # Architecture reference and code standards
├── FLAT_BUNDLE_HYPOTHESES.md       # Testable predictions for flat bundle conjecture
├── FLAT_BUNDLE_IMPLEMENTATION_PLAN.md  # Implementation plan for hypothesis tests
├── REVIEW.md                       # Codebase review (theory–implementation alignment)
└── pytest.ini                      # Test configuration
```


## Numerical Stability

Key techniques enabling large gauge groups:

- **√K attention scaling**: `logits = -KL / (κ · √K)` prevents softmax saturation
- **Dimension-dependent KL clamping**: ceiling of `max(100, 5K)`
- **Per-parameter gradient clipping**: Independent budget per parameter group (μ, Σ, φ)
- **Loss √K normalization**: `loss = F / √K` for hyperparameter transferability
- **GL(K) transport**: No re-orthogonalization needed (only invertibility required). Optional O(K) enforcement via Newton-Schulz + learnable reflections


## Citation

```bibtex
@article{dennis2025attention,
  title={Attention, Transformers, and Backpropagation are Degenerate Limits
         of the Variational Free Energy Principle},
  author={Dennis, Robert C.},
  journal={Preprint},
  year={2025}
}
```

## License

This project is for research purposes. Contact: cdenn016@yahoo.com
