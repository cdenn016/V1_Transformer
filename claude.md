

# VFE Transformer with Renormalization Group Analysis

## Project Overview

This is a **Gauge-Theoretic Transformer** implementing **Variational Free Energy (VFE)** minimization for active inference. The key innovation is replacing learned attention projections (W_Q, W_K) with **information-geometric attention based on KL divergence**, enabling principled belief evolution and uncertainty quantification.


DO NOT USE NEURAL NETWORKS OR ARCHITECTURES


## Architecture


                h (hyper-prior)
                │
                │ KL(s||h) — regularization
                ▼
s_i = self.prior_mu[i]  ◄──── γ·KL(s_i||Ω_ij·s_j) ────► s_j
(position MODEL,                    (model coupling)
 slow timescale)
                │
                │ p = w·π_token + (1-w)·s
                ▼
          p_i (PRIOR)
                │
                │ α·KL(q||p)
                ▼
     q_i (beliefs, fast)  ◄──── β_ij·KL(q_i||Ω_ij·q_j) ────► q_j
                                      (attention)


### Key Components

| Component | File | Description |
|-----------|------|-------------|
| Transformer Block | `transformer/transformer_block.py` | Main block with gauge attention + FFN |
| Attention | `transformer/attention.py` | KL-divergence based attention (no W_Q, W_K!) |
| FFN | `transformer/ffn.py` | Unified FFN (learned, VFE, hamiltonian modes) |
| Variational FFN | `transformer/variational_ffn.py` | VFE implementations including VFE_dynamic |
| RG Metrics | `transformer/rg_metrics.py` | **NEW**: RG analysis tools |
| Model | `transformer/model.py` | Full language model |

### FFN Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `learned` | Standard MLP + GELU | Baseline |
| `variational_gradient_engine` / `VFE` | Fixed-β VFE descent | Standard active inference |
| `VFE_dynamic` | **Dynamic-β VFE** (β recomputed each step) | RG analysis, meta-agent emergence |


## Key Equations

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

### Multi-Timescale Dynamics

The free energy naturally separates into:
- **Fast (E-step)**: Belief inference `dq_i/dt = -eta_q dF_fast/dq_i` --- what transformers do in a forward pass
- **Slow (M-step)**: Model learning `ds_i/dt = -eta_s dF_slow/ds_i` --- what backpropagation updates

Standard transformers operate in the adiabatic limit: slow variables frozen during inference, updated between passes.



| Configuration | K | Layers | Gauge Mode | Train PPL | Test PPL | Parameters |
|---|---|---|---|---|---|---|
| **Best (1 epoch)** | **90** | **1** | **Omega in GL(10)** | **63** | **76** | **~59M** |

For context: random-chance perplexity is ~50,000. The K=90, GL(10) configuration achieves **558x improvement** over random chance, substantially exceeding prior results and approaching standard transformer performance with purely geometric attention.

## Contributing

When working on this codebase:

1. **Preserve gauge equivariance**: Covariance transport must be Σ_transported = Ω @ Σ @ Ω^T
2. **Use natural gradients**: Project Euclidean gradients via Fisher metric
3. **Test RG behavior**: New features should maintain or improve RG trends
4. **Document mathematical formulas**: Include LaTeX-style notation in docstrings





# Claude Code Guidelines for Gauge Transformer Project

## Domain Expertise

Apply these when working on this codebase:

- **Differential Geometry**: SPD manifolds, geodesics, affine-invariant metrics, Lie theory, fiber bundles
- **Variational Inference**: KL divergence, free energy, ELBO, information geometry
- **Gauge Theory**: Symmetries, equivariance, parallel transport, irreps
- **Matrix/Linear Algebra**: Eigendecomposition, Kronecker products, matrix exponentials

## Code Standards

- Write modular, testable functions with type hints
- Docstrings should include LaTeX math where relevant
- Variable names should match paper notation (e.g., `mu_q` for μ_q, `Sigma` for Σ)
- Check tensor shapes at each step when debugging
- Verify gradient flow with small-dim smoke tests


## Communication Style

**Be direct:**
- State errors and concerns plainly without excessive hedging
- "This is wrong because X" not "This might potentially be slightly off"
- Always ultra-think and double check


**minimize itemizations when working on manuscripts:**
- utilize academic prose 
- minimize the usage of bullet points, itemizations, lists, ---, "crucially", "critically", etc.

**Push back:**
- Challenge gaps in derivations, ask for justification
- If a claim needs proof, ask for it

**Skip praise preambles:**
- No "Great question!" openers—just answer
- No "Excellent point!"—just engage with the substance

**Flag simpler alternatives:**
- Call out over-engineering
- Ask what complexity buys if something seems unnecessarily elaborate

**Maintain position under pushback:**
- Don't fold immediately when disagreeing
- Ask "What am I missing?" rather than capitulating

**Honest uncertainty:**
- "I'm not sure this is right" beats confident speculation
- Acknowledge when something needs verification

**No bullshit:**
- If a correspondence is interpretive rather than mathematically exact, say so explicitly
- If something doesn't connect, don't force it—admit the gap
- Remove content that doesn't earn its place through rigorous derivation
- Never dress up hand-waving as theorem
- When asked "what does X have to do with anything?"—if the answer is "not much", say that


## VFE, Renormalization, and the Information Bottleneck

The gauge transformer framework unifies three deep theoretical perspectives: variational inference, renormalization group flow, and the information bottleneck principle.

### The Information Bottleneck (IB) Principle

Tishby's IB: Find representation Z of input X that predicts Y while compressing:

```
L_IB = I(Z; Y) - β · I(Z; X)
```

- **I(Z; Y)**: Preserve information relevant to target (prediction)
- **I(Z; X)**: Discard irrelevant input details (compression)
- **β**: Tradeoff parameter

### VFE IS the Information Bottleneck

The variational free energy:

```
F = sum_i  alpha D_KL(q_i || p_i)                         [self-coupling: beliefs to priors]
  + sum_i  lambda_h D_KL(s_i || h)                        [hyper-prior: models to centroid]
  + sum_ij beta_ij D_KL(q_i || Omega_ij q_j)              [belief coupling / attention]
  + sum_ij gamma_ij D_KL(s_i || Omega_ij s_j)             [model coupling / meta-cognition]
  - E_q[log p(o | x)]                                     [observation likelihood]
```


| IB Term | VFE Term | Meaning |
|---------|----------|---------|
| I(Z; X) | KL(q ‖ p) | Bits used beyond prior |
| I(Z; Y) | -CE | Prediction accuracy |
| β | α | Compression-accuracy tradeoff |

**The prior p is the reference channel** — beliefs at p carry zero information, deviations carry bits.

### Dynamic β: Adaptive Compression

The attention weights:

```
β_ij = softmax(-KL(q_i || Ω_ij·q_j) / κ)
```

This implements **input-dependent compression**:

- **Similar beliefs** (low KL) → high β → pool/average information
- **Distinct beliefs** (high KL) → low β → preserve separately

**The temperature κ IS the IB tradeoff**:
- High κ → soft attention → aggressive compression
- Low κ → sharp attention → selective preservation

### Renormalization = Hierarchical IB

Each VFE iteration (or layer) performs coarse-graining:

```
Raw tokens:    [t1] [t2] [t3] [t4] [t5] [t6]
                    ↓ VFE step (β clusters similar beliefs)
Meta-agents:   [   A   ] [   B   ] [   C   ]
                    ↓ VFE step
Coarser:       [     X     ] [     Y     ]
                    ↓
Output:        Minimal sufficient statistics for prediction
```

- **Tokens with KL≈0 merge** — within-group variation discarded
- **Between-group differences survive** — predictively relevant
- **RG fixed point** = optimal IB representation (can't compress further without losing prediction)

### Gauge Invariance: Geometric Compression

The transport Ω_ij enforces gauge invariance:

```
KL(q_i || Ω_ij·q_j) is gauge-invariant
```

- Multiple configurations differing by gauge → same representation
- Gauge-variant information automatically compressed out
- Only gauge-invariant (physical) information survives

This is a **symmetry-based prior** implementing compression geometrically.

### The Unified Picture

| Concept | IB View | VFE View | RG View |
|---------|---------|----------|---------|
| Compression | min I(Z;X) | KL(q‖p) → 0 | Coarse-graining |
| Prediction | max I(Z;Y) | min CE | Fixed point stability |
| Tradeoff | β parameter | κ temperature | Relevant vs irrelevant |
| Representation | Z | (μ, Σ) beliefs | Renormalized couplings |
| Hierarchy | Deep IB | VFE iterations | RG flow |

**Key insight**: Emergent block structure in β_ij reveals which tokens carry redundant information about the target and can be safely merged. The dynamics discovers the optimal compression automatically.




