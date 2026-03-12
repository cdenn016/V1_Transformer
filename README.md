# Gauge-Theoretic Transformer

A research framework implementing gauge-covariant variational free energy (VFE) minimization for language modeling and multi-agent systems. This codebase accompanies the manuscript:

**Key result:** A single-layer VFE transformer with no MLPs or neural network components
achieves **76 test perplexity** on WikiText-103 (BPE-2 tokenization) with K=80, GL(10),
8 heads, RoPE, and sequence length 128 after just 1 epoch of training. All representational
capacity comes from iterative variational free energy minimization over belief tuples (μ, Σ, φ). The E-step *is* the
computation, it is not a learned feed-forward network.


## Core Thesis

Language is a dynamic informational system: speakers encode and decode beliefs under uncertainty, and language models learn the statistical structure of this process. The mathematical framework natural to such systems---gauge-covariant variational free energy minimization over communicating agents on a statistical fiber bundle---explains why attention mechanisms work.


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


Standard transformer attention, the `1/sqrt(d_k)` scaling, layer normalization, and backpropagation all emerge as consequences of a single variational principle. The standard attention rule

```
beta_ij = softmax(Q_i K_j^T / sqrt(d_k))
```

is recovered as a **degenerate limit** of the gauge-theoretic attention

```
beta_ij = softmax(-D_KL(q_i || Omega_ij q_j) / tau)
```

through two simplifications: isotropic covariances and flat bundle.

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

### Variational Free Energy

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


### Categorical Observation Precision (Transformer-Specific)

For transformers with softmax output p = softmax(W_out @ μ / τ):

```
Λ_o = (1/τ²) W^T (diag(p) - pp^T) W = (1/τ²) Cov_p(W)
```

This is the **Hessian of cross-entropy** with respect to μ:
- When p is peaked (confident): Λ_o has low rank, weak constraint
- When p is uniform (uncertain): Λ_o reflects full embedding structure
- Temperature τ scales precision (lower τ → higher precision)

### The Nonlinearity

Standard transformer: GELU(x) — ad hoc, nobody knows why it works

Ours: ∂β_{ij}/∂θ — emerges from differentiating softmax attention:

```
β_{ij} = softmax(-KL_{ij} / κ)

∂β_{ij}/∂μ_i = -β_{ij} · [∂KL_{ij}/∂μ_i - Σ_k β_{ik} · ∂KL_{ik}/∂μ_i] / κ
∂β_{ij}/∂Σ_i = -β_{ij} · [∂KL_{ij}/∂Σ_i - Σ_k β_{ik} · ∂KL_{ik}/∂Σ_i] / κ
∂β_{ij}/∂φ_i = -β_{ij} · [∂KL_{ij}/∂φ_i - Σ_k β_{ik} · ∂KL_{ik}/∂φ_i] / κ
```

The **most general form** of the theory---no simplifying limits taken. Full non-isotropic covariances, non-trivial gauge transport, KL-divergence attention. **No MLPs, activation functions, learned W_Q/W_K/W_V, or positional encodings.** Only a linear output projection (from K dimensions to 50k) is retained.

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
In a standard transformer, embeddings are lookup tables and conventionally excluded from
weight decay. In the gauge transformer, embeddings are **statistical parameters** --- means
`mu_p`, covariances `Sigma_p`, and gauge frames `phi` --- that directly enter KL divergences,
matrix exponentials, and VFE gradients. Without the hyper-prior, these parameters have no
regularization above Level 2 and can drift to magnitudes that destabilize training:

- Larger `mu_p` -> KL divergences saturate -> attention beta becomes one-hot
- Larger `Sigma_p` -> natural gradient `Sigma @ grad` overshoots -> VFE diverges
- Larger `phi` -> transport operators `Omega = exp(phi*G)` become extreme

The weight decay coefficient `wd` is the hyper-prior **precision**: larger `wd` = tighter
prior = stronger regularization. This is the same structure as empirical Bayes / Type-II
maximum likelihood, and in the FEP literature corresponds to the hierarchy of Markov blankets
where each level's sufficient statistics are regularized by the level above.

### Multi-Timescale Dynamics

The free energy naturally separates into:
- **Fast (E-step)**: Belief inference `dq_i/dt = -eta_q dF_fast/dq_i` --- what transformers do in a forward pass
- **Slow (M-step)**: Model learning `ds_i/dt = -eta_s dF_slow/ds_i` --- what backpropagation updates

Standard transformers operate in the adiabatic limit: slow variables frozen during inference, updated between passes.

### Three Limits Recovering Standard Attention

| Limit | What it discards | Result |
|---|---|---|
| **1. Isotropic covariances** | Non-isotropic Sigma_i -> sigma^2 I | KL reduces to squared Euclidean distance |
| **2. Flat bundle** | Position-dependent Omega_ij -> constant Omega | Global gauge, no curvature |
| **3. Learned projections** | sigma^{-2} Omega absorbed into W_Q W_K^T | Standard Q, K, V projections |

After all three limits: `beta_ij = softmax(Q_i K_j^T / sqrt(d_k))` --- standard transformer attention.

### Gradient Descent as Variational Inference

Gradient descent on the gauge-equivariant free energy recovers the standard transformer training update:

| FEP Framework | Neural Network |
|---|---|
| Free energy F[{q_i}] | Loss L(theta) |
| Belief q_i = N(mu_i, sigma^2 I) | Embedding h_i |
| Gauge transport Omega in GL(K) | W_Q W_K^T (learned) |
| -E_q[log p(o \| k)] | Cross-entropy loss |
| Vacuum (no observations) | Untrained network |
| Symmetry breaking | Training / learning |

### Multi-Head Attention as Block-Diagonal GL(K)

Multi-head attention restricts the full GL(d_k) to a block-diagonal subgroup:

```
G_multi-head = GL(d_head)^H  subset  GL(d_k)
```

Each head learns an independent GL(d_head) gauge transformation. For compact groups like SO(3), irreducible representations yield non-uniform head dimensions (1, 3, 5, 7, ...) with intrinsic geometric meaning.

### Multi-Timescale Dynamics

The free energy naturally separates into:
- **Fast (E-step)**: Belief inference `dq_i/dt = -eta_q dF_fast/dq_i` --- what transformers do in a forward pass
- **Slow (M-step)**: Model learning `ds_i/dt = -eta_s dF_slow/ds_i` --- what backpropagation updates

Standard transformers operate in the adiabatic limit: slow variables frozen during inference, updated between passes.

### Symmetry Breaking

Without observations, the free energy defines a gauge-symmetric vacuum---all agents converge to identical beliefs modulo gauge orbit (analogous to an untrained network). Observations break this symmetry, driving agents toward specialized representations determined by training data. Learning is thus interpreted as explicit symmetry breaking.

## Experimental Results

### BERT Validation (144 attention heads)

Quantitative comparison between gauge-aligned KL attention and standard dot-product attention on a pretrained `bert-base-uncased`:

| Metric | Value |
|---|---|
| Optimal temperature | tau = 19.0 (theory: tau = 2 sqrt(d) = 16, 19% deviation) |
| Mean Pearson correlation | r = 0.821 |
| Median Pearson correlation | r = 0.889 |
| Heads with r > 0.8 | 68.1% |
| Heads with r > 0.9 | 49.3% |

**Key-norm bias prediction confirmed**: average rho = -0.352 across all heads, significant in 92.4% of heads at p < 0.001. This gauge-theoretic prediction explains why layer normalization is a geometric necessity: it enforces constant key norms required for frame-independent inference.

### GL(K) Language Modeling (WikiText-103, vocab 50,257)

The **most general form** of the theory---no simplifying limits taken. Full non-isotropic covariances, non-trivial gauge transport, KL-divergence attention. **No MLPs, activation functions, learned W_Q/W_K/W_V, or positional encodings.** Only a linear output projection (from K dimensions to 50k) is retained.

| Configuration | K | Layers | Gauge Mode | Train PPL | Test PPL | Parameters |
|---|---|---|---|---|---|---|
| **Best (1 epoch)** | **80** | **1** | **Omega in GL(10)** | **63** | **76** | **~50M** |

For context: random-chance perplexity is ~50,000. The K=80, GL(10) (8 heads, seq-length = 128, RoPE) configuration achieves **~600x improvement** over random chance, substantially exceeding prior results and approaching standard transformer performance with purely geometric attention.

**Emergent semantic structure**: Learned gauge frames develop interpretable categorical organization---punctuation, content words, and letters cluster separately in both belief space (mu) and gauge frame space (phi) without any category supervision.


## Installation

**Requirements**: Python 3.9+ with CUDA-capable GPU (recommended)

```bash
git clone https://github.com/cdenn016/Gauge-Transformer.git
cd Gauge-Transformer
pip install torch numpy scipy numba matplotlib seaborn plotly networkx scikit-learn datasets tiktoken
```

### Core Dependencies

- **PyTorch** (>=2.0.0) with CUDA support
- **NumPy / SciPy** (numerical computation)
- **Numba** (JIT compilation for transport kernels)
- **Matplotlib / Seaborn / Plotly** (visualization)
- **NetworkX** (graph operations for meta-agent detection)
- **scikit-learn** (spectral clustering, metrics)
- **datasets** (HuggingFace, for WikiText)
- **tiktoken** or **transformers** (tokenization)

## Usage

### Transformer Training

```bash
# Standard VFE training with gauge-theoretic attention
python transformer/train.py

# Publication-quality training with all experimental features
python transformer/train_publication.py

# Text generation from trained model
python generate.py

# Model inference and analysis
python inference.py --checkpoint path/to/model.pt
```

### Multi-Agent Simulation

```bash
# Default simulation
python simulation_runner.py

# Presets
python simulation_runner.py --preset emergence    # Meta-agent emergence demo
python simulation_runner.py --preset ouroboros    # Ouroboros Tower (non-Markovian memory)
python simulation_runner.py --preset hamiltonian  # Underdamped symplectic dynamics
```

## Project Structure

```
Gauge-Transformer/
├── transformer/                # Gauge-theoretic transformer architecture
│   ├── core/                  # Core components
│   │   ├── model.py           #   GaugeTransformerLM (full language model)
│   │   ├── attention.py       #   KL-divergence multi-head attention
│   │   ├── blocks.py          #   Transformer blocks with gauge transport
│   │   ├── variational_ffn.py #   VFE feedforward (VFE_dynamic, hamiltonian modes)
│   │   ├── embeddings.py      #   Gauge token/positional embeddings
│   │   └── prior_bank.py      #   Token-dependent prior bank for pure FEP
│   ├── analysis/              # RG metrics, semantic analysis, publication metrics
│   ├── data/                  # Dataset loading (WikiText-2, WikiText-103)
│   ├── training/              # Training configuration and utilities
│   ├── train.py               # Main training loop
│   └── train_publication.py   # Publication-quality training
├── agent/                     # Multi-agent system
│   ├── agents.py              # Agent: section of statistical bundle
│   ├── system.py              # MultiAgentSystem orchestrator
│   ├── trainer.py             # Gradient-based trainer
│   └── hamiltonian_trainer.py # Symplectic integration trainer
├── geometry/                  # Differential geometry engine
│   ├── geometry_base.py       # Base manifold, support regions
│   ├── connection.py          # Gauge connections, curvature
│   ├── lie_algebra.py         # Lie bracket computations
│   └── geodesic_corrections.py# Riemannian geodesic corrections
├── gradients/                 # Free energy gradient engine
│   ├── free_energy_clean.py   # VFE component computation
│   ├── gradient_engine.py     # Full dF/dtheta for all parameters
│   ├── gradient_terms.py      # Individual gradient terms
│   └── softmax_grads.py       # Gradients through softmax weights
├── math_utils/                # Mathematical primitives
│   ├── generators.py          # SO(3)/GL(K) Lie algebra generators
│   ├── transport.py           # Parallel transport Omega_ij = exp(phi_i)exp(-phi_j)
│   ├── fisher_metric.py       # Fisher-Rao natural gradients
│   └── push_pull.py           # Gaussian pushforward under transport
├── meta/                      # Meta-agent emergence
│   ├── emergence.py           # RG flow analyzer, spectral clustering
│   ├── consensus.py           # Epistemic death detection
│   └── gradient_adapter.py    # Multi-scale gradient bridging
├── Docs/                      # Technical documentation and manuscripts
│   ├── attention manuscript/  # JMLR attention paper (main theory)
│   ├── gauge_ctm_derivation.tex
│   └── references.bib
├── Transformer Manuscript/    # Experimental results and figures
├── tests/                     # Test suite
├── config.py                  # System configuration
├── simulation_config.py       # Experiment presets
├── simulation_runner.py       # Multi-agent simulation orchestrator
├── generate.py                # Text generation
└── inference.py               # Model inference
```

## Architecture Details

### KL-Based Attention (No W_Q, W_K)

```
Attention weights:
  beta_ij = softmax_j(-D_KL(q_i || Omega_ij q_j) / kappa)

where D_KL between Gaussians:
  D_KL(q_i || Omega_ij q_j) = 0.5 * [
      tr((Omega_ij Sigma_j Omega_ij^T)^{-1} Sigma_i)
    + (mu_i - Omega_ij mu_j)^T (Omega_ij Sigma_j Omega_ij^T)^{-1} (mu_i - Omega_ij mu_j)
    - K
    + log(det(Omega_ij Sigma_j Omega_ij^T) / det(Sigma_i))
  ]

Message aggregation:
  m_i = sum_j beta_ij Omega_ij mu_j
```

### Two-Timescale Learning

**Fast (belief inference)**: Within each forward pass, beliefs evolve via natural gradient descent:
```
for step in range(belief_steps):
    beta_ij = softmax(-KL(q_i || Omega_ij q_j) / kappa)
    grad_q = dF/dq
    mu_q <- mu_q - mu_lr * Sigma_q @ grad_q    # natural gradient
```

**Slow (prior learning)**: Token priors update across training steps:
```
prior_bank[target_v] <- (1 - prior_lr) * prior_bank[target_v] + prior_lr * avg_belief
```

### Meta-Agent Emergence via Renormalization Group

The VFE has self-similar structure: meta-agents satisfy the same definition as individual agents.

```
Scale zeta=0:   Tokens q_i = N(mu_i, Sigma_i) interact via beta_ij
                        | clustering (KL -> 0 within groups)
Scale zeta=1:   Meta-agents q_A = N(mu_A, Sigma_A) interact via beta'_AB
                        | further clustering
Scale zeta=2:   Super-meta-agents ...
```

Detected via spectral clustering on the attention matrix with metrics: modularity Q(beta), effective rank, intra/inter-cluster KL divergence.

### Phi Gradient Preconditioning

The gauge frame φ ∈ gl(K) requires geometric gradient preconditioning because the backward pass through `matrix_exp` amplifies non-compact (symmetric) directions exponentially. Four modes are available via `phi_natural_gradient` config:

| Mode | Method | Position-dependent? |
|------|--------|-------------------|
| `'clip'` | Norm clipping | No |
| `'cartan'` | Cartan decomposition (fixed sym dampening) | No |
| `'killing'` | Killing form metric (no free params) | No |
| `'pullback'` | Full pullback through exp (exact) | Yes |

The `'pullback'` mode computes the Riemannian metric G_ab(φ) = ⟨Ψ(ad_X)(T_a), Ψ(ad_X)(T_b)⟩ where Ψ(z) = (e^z-1)/z is the dexp Jacobian. This is the theoretically exact natural gradient on the Lie group, automatically compensating for exponential amplification in non-compact directions.

### DEQ Implicit Differentiation (E-Step Backward)

The E-step iterates natural gradient VFE descent to a fixed point μ*, Σ*. Standard backpropagation unrolls through all iterations, which has O(n_iterations) memory cost and can produce noisy gradients for large iteration counts. **Deep Equilibrium (DEQ) implicit differentiation** replaces unrolled backprop with a single backward pass through the fixed-point equation.

**How it works:**
1. **Forward:** Run the normal E-step loop (unchanged)
2. **Backward:** Instead of backpropagating through all iterations, apply the implicit function theorem at the fixed point. The corrected gradient is `(I - J)^{-1} v` where `J` is the Jacobian of one E-step and `v` is the incoming gradient
3. **Neumann approximation:** `(I - J)^{-1} v ≈ v + Jᵀv + (Jᵀ)²v + ...` (K terms), computed via K vector-Jacobian products

**Usage:**
```python
# Legacy (dict config)
config['use_deq'] = True
config['deq_neumann_terms'] = 5  # default; 3 is often sufficient

# v2 (dataclass config)
config = GaugeTransformerConfig(use_deq=True, deq_neumann_terms=5, ...)
```

**When to use:**
- `n_vfe_iterations >= 5` (significant memory savings)
- Training is unstable with many E-step iterations (DEQ provides smoother gradients)
- You want gradients that respect the fixed-point structure rather than the trajectory

**Trade-offs:**
- Forward pass is identical (no speed change)
- Backward pass: K extra VJPs instead of unrolling through all iterations
- Memory: O(1) in iterations (vs O(n_iterations) for unrolled backprop)

## Numerical Stability

Key fixes enabling large gauge groups (see `NUMERICAL_STABILITY.md`):

- **sqrt(K) attention scaling**: `logits = -KL / (kappa * sqrt(K))` prevents softmax saturation
- **Dimension-dependent KL clamping**: ceiling of `max(100, 5K)`
- **Per-parameter gradient clipping**: Independent budget per parameter group (mu, Sigma, phi)
- **Loss sqrt(K) normalization**: `loss = F / sqrt(K)` for hyperparameter transferability
- **GL(K) transport**: No re-orthogonalization needed (only invertibility required)

## Testing

```bash
pytest tests/
pytest tests/ -v -k "transformer"
pytest tests/ --tb=short
```

## Documentation

- `Docs/attention manuscript/` --- Main manuscript with complete theory and proofs
- `NUMERICAL_STABILITY.md` --- Numerical stability guide for large K
- `transformer/PURE_FEP_TRANSFORMER_OVERVIEW.md` --- Pure FEP architecture overview
- `claude.md` --- Architecture overview and code standards

## Citation

If this work is useful, please cite:

```bibtex
@article{dennis2025attention,
  title={Attention, Transformers, and Backpropagation are Degenerate Limits of the Variational Free Energy Principle},
  author={Dennis, Robert C.},
  journal={Preprint},
  year={2025}
}
```

## License

This project is for research purposes.
