# Empirical Comparisons: Gauge-Theoretic vs Standard Attention

Controlled experiments comparing KL-attention (gauge-theoretic) with dot-product attention (standard transformer) on WikiText-103 at K=40, 1 layer, seq_len=64, batch=128, 15k steps.

---

## Experimental Setup

All models share the same GELU FFN (hidden=160), RoPE positional encoding (base=5000), causal masking, AdamW optimizer, and dropout (0.1). Only the attention mechanism, embedding structure, and parameter budget differ.

### Architectures Tested

| Model | Attention | Embeddings | Decode | Params |
|---|---|---|---|---|
| Standard (K=40) | Dot-product (W_Q, W_K, W_V) | nn.Embedding | Linear (tied) | 4M |
| Standard (K=240) | Dot-product | nn.Embedding | Linear (tied) | 24M |
| Hybrid GL(10) | KL-divergence + GL(10) transport | nn.Embedding + sigma + phi | Linear (tied) | 24M |
| Hybrid SO(3) | KL-divergence + SO(3) transport | nn.Embedding + sigma + phi | Linear (tied) | 4.6M |
| Hybrid GL(10) + PriorBank | KL-divergence + GL(10) transport | PriorBank (mu, sigma, phi) | KL-decode | 24M |
| VFE dynamic | KL-divergence + GL(10) transport | PriorBank | KL-decode / Linear | 24M |

The parameter asymmetry between standard (4M) and GL(10) hybrid (24M) is structural: the standard transformer uses shared projection matrices (W_Q, W_K: 3200 params total) while the gauge model uses per-token gauge frames (phi_embed: 50257 x 400 = 20M params). This is the inherent cost of per-token gauge transport vs shared linear projections.

---

## Result 1: FFN Type Does Not Matter

At K=40, replacing the VFE Boltzmann gate with a standard GELU FFN produces identical PPL.

| Model | FFN | Decode | Test PPL |
|---|---|---|---|
| VFE dynamic | Boltzmann gate | PriorBank | ~129 |
| Hybrid | GELU | PriorBank | ~129 |
| VFE dynamic | Boltzmann gate | Linear | ~119 |
| Hybrid | GELU | Linear | ~119 |

The nonlinearity type contributes nothing at this scale. The bottleneck is elsewhere.

---

## Result 2: PriorBank Decode Is a Bottleneck

Switching from PriorBank KL-decode (rank-2K=80, geometrically constrained) to linear decode (rank-K=40, unconstrained directions) improves PPL by ~10 points.

| Decode | Test PPL |
|---|---|
| PriorBank KL-decode | ~129 |
| Linear projection | ~119 |

The KL-decode constrains logits to the manifold of KL divergences to prior distributions. The linear decode can learn arbitrary discrimination directions. More rank (2K vs K) does not compensate for the geometric constraint.

---

## Result 3: KL-Attention Matches Dot-Product at Matched LR

At LR=3e-4 (the standard transformer's optimal LR), KL-attention and dot-product attention converge to the same PPL at matched parameters.

| Model | Params | LR | Test PPL |
|---|---|---|---|
| Standard (K=240) | 24M | 3e-4 | 128 |
| Hybrid GL(10) | 24M | 3e-4 | ~129 |

At matched parameters AND matched learning rate, the two attention mechanisms are functionally equivalent. This confirms the manuscript's three-limit reduction empirically: when both use the same positional encoding, nonlinearity, output projection, and LR, KL-attention degenerates to dot-product attention.

---

## Result 4: Higher LR Tolerance Provides Modest Advantage

The gauge parameterization tolerates M-step learning rates of 0.05 for embedding parameters, while the standard transformer diverges above ~3e-4. At matched parameters with each model's optimal LR:

| Model | Params | LR | Test PPL |
|---|---|---|---|
| Standard (K=240) | 24M | 3e-4 | 128 |
| Hybrid GL(10) | 24M | 0.05 | 119 |

The higher LR gives a **7% PPL improvement** (128 vs 119). The mechanism: the backward pass through the KL divergence produces gradients scaled by precision $\Sigma_j^{-1}$, providing implicit per-dimension adaptive preconditioning. This is a natural gradient effect built into the forward computation that the dot product lacks.

This advantage is real but modest. It does not translate to a fundamentally different convergence regime.

---

## Result 5: Apparent Sample Efficiency Was Mostly Parameters

The initial comparison suggested a dramatic sample efficiency advantage:

| Model | Params | Steps to ~245 PPL | Tokens |
|---|---|---|---|
| Standard (K=40) | 4M | 10,000 | 82M |
| Hybrid GL(10) | 24M | 1,000 | 8.2M |

This 10x step advantage appeared to confirm the RG conjecture's prediction of $R(K) \propto \sqrt{K} \approx 6.3\text{x}$ sample efficiency. However, the parameter budgets are mismatched by 6x. Controlling for parameters:

| Model | Params | Steps to ~247 PPL | it/sec |
|---|---|---|---|
| Standard (K=240) | 24M | ~1,000 | 60 |
| Hybrid GL(10) | 24M | 1,000 | 8 |

At matched parameters, the step advantage vanishes. Both reach ~247 PPL at step 1000. The standard is 7.5x faster per step (no matrix_exp, no pairwise KL). In wall-clock time, the standard transformer reaches the same PPL **7.5x faster**.

The 10x step advantage in the unmatched comparison was almost entirely from having 6x more parameters, not from the KL-Mahalanobis structure.

---

## Result 6: At Matched Parameters, Standard Wins or Ties

The complete parameter-controlled comparison:

| Model | Params | LR | Test PPL |
|---|---|---|---|
| Standard (K=40) | 4M | 3e-4 | 212 |
| Hybrid SO(3) | 4.6M | 3e-4 | 270 |
| Standard (K=240) | 24M | 3e-4 | 128 |
| Hybrid GL(10) | 24M | 3e-4 | ~129 |
| Hybrid GL(10) | 24M | 0.05 | 119 |

At ~4M params: **the standard transformer wins by 58 PPL** (212 vs 270). The SO(3) gauge group covers only 12 of 40 embedding dimensions, leaving 28 dimensions without transport structure. The gauge overhead (sigma table, phi table, matrix_exp computation) actively hurts at this scale.

At ~24M params, matched LR: **tied** (128 vs 129). The 20M phi table gives the gauge model enough per-token features to match the standard model's larger K and wider MLP.

At ~24M params, optimal LR: **modest gauge advantage** (119 vs 128). The KL-Mahalanobis preconditioning enables higher LR, giving 7% lower PPL.

---

## Result 7: The Standard Transformer Wins at Convergence

Given sufficient training at larger scale (K=90, 60k steps):

| Model | Params | Test PPL |
|---|---|---|
| Gauge VFE (GL(15), K=90) | 81M | 71.6 |
| Standard (d=1280, param-matched) | 84M | 48.5 |
| Standard (d=90, embed-matched) | 4.6M | 118.6 |

The gauge model outperforms at matched embedding dimension (71 vs 119 at K=90) but underperforms at matched parameters (71 vs 48). The parameter overhead from per-token gauge frames prevents the gauge model from competing at matched total budget.

---

## Synthesis

### What the gauge framework does NOT provide

- **Better expressiveness.** At matched parameters and matched LR, KL-attention reproduces dot-product attention exactly (128 vs 129 PPL). The three-limit reduction is not just theory; it is empirically exact.
- **Sample efficiency at matched parameters.** The apparent 10x step advantage was 6x parameter advantage. At matched params, the step advantage vanishes.
- **Better wall-clock performance.** The per-step overhead of matrix exponentials and pairwise KL (7.5x slower) is not offset by the modest optimization advantage.

### What the gauge framework DOES provide

- **Theoretical explanation.** Standard transformer attention emerges as the degenerate limit of gauge-covariant VFE minimization. The $1/\sqrt{d_k}$ scaling, multi-head structure, and softmax normalization are necessary consequences of constrained KL minimization on a statistical manifold.
- **Modest optimization advantage at large parameter budgets.** The KL-Mahalanobis structure enables ~100x larger learning rates, translating to ~7% lower PPL at matched parameters and matched steps. This comes from implicit per-dimension preconditioning through the precision $\Sigma^{-1}$ in the backward pass.
- **Identification of architectural tradeoffs.** Per-token gauge frames (20M params) serve the same functional role as shared W_Q, W_K projections (3200 params). The gauge parameterization trades parameter efficiency for optimization conditioning. Whether this tradeoff is favorable depends on whether data or compute is the binding constraint.
- **Empirical confirmation of the three-limit reduction.** KL-attention = dot-product attention at matched architecture, a cleaner validation than the BERT correlation analysis.

### Implications for the manuscript

The RG conjecture's prediction of $R(K) \propto \sqrt{K}$ sample efficiency cannot be confirmed from these experiments because the parameter confound dominates. A clean test would require matching the total parameter budget, including the per-token gauge frame overhead. At matched parameters, the advantage is at most 7% from higher LR tolerance, far below the predicted $\sqrt{40} \approx 6.3\text{x}$.

The recommended framing: the gauge-theoretic VFE framework provides a principled derivation of transformer attention from variational inference, with the equivalence confirmed empirically. The unreduced gauge model offers modest optimization benefits (higher LR tolerance, ~7% PPL at matched params) at significant computational cost (7.5x slower per step, 6x more parameters for attention scoring). Its primary value is explanatory, not practical.

---

## Configurations

### Standard (K=40, 4M params)
```
embed_dim=40, n_heads=4, hidden_dim=160
LR=3e-4, RoPE, batch=128, 15k steps
```

### Standard (K=240, 24M params)
```
embed_dim=240, n_heads=4, hidden_dim=960
LR=3e-4, RoPE, batch=128, 15k steps
```

### Hybrid GL(10) (24M params)
```
embed_dim=40, irrep_spec=[('fund', 4, 10)], hidden_dim=160
KL-attention, GL(10) transport, nn.Embedding + sigma + phi
LR=3e-4 or 0.05, RoPE, batch=128, 15k steps
phi_embed: 50257 x 400 = 20M params (dominates)
```

### Hybrid SO(3) (4.6M params)
```
embed_dim=40, irrep_spec=[('fund', 4, 3)], hidden_dim=160
KL-attention, SO(3) transport, nn.Embedding + sigma + phi
LR=3e-4, RoPE, batch=128, 15k steps
phi_embed: 50257 x 12 = 0.6M params
```
