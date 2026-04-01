# Empirical Comparisons: Gauge-Theoretic vs Standard Attention

Controlled experiments comparing KL-attention (gauge-theoretic) with dot-product attention (standard transformer) on WikiText-103 at K=40, 1 layer, seq_len=64.

---

## Experimental Setup

Three architectures sharing the same GELU FFN (hidden=160), RoPE positional encoding, and causal masking. Only the attention mechanism and embedding structure differ.

| Component | Standard | Hybrid (PriorBank) | Hybrid (Linear) |
|---|---|---|---|
| Attention scoring | $Q_i K_j^\top / \sqrt{d}$ | $-D_{\text{KL}}(q_i \| \Omega_{ij}[q_j]) / \kappa\sqrt{K}$ | same |
| Embeddings | nn.Embedding (mu only) | PriorBank (mu, sigma, phi) | nn.Embedding + sigma + phi |
| Output projection | Linear (tied) | PriorBank KL-decode | Linear (tied) |
| Parameters | ~4M | ~24M | ~24M |
| Attention params | W_Q, W_K, W_V: 3200 shared | phi_embed: 20M per-token | same |

The parameter asymmetry (4M vs 24M) is structural: the standard transformer uses shared projection matrices (W_Q, W_K at 3200 params total) while the gauge model uses per-token gauge frames (phi at 20M params). This is the price of per-token gauge transport vs shared linear projections.

---

## Result 1: FFN Type Does Not Matter

At K=40, replacing the VFE Boltzmann gate with a standard GELU FFN produces identical PPL.

| Model | FFN | Decode | Test PPL |
|---|---|---|---|
| VFE dynamic | Boltzmann gate | PriorBank | ~129 |
| Hybrid | GELU | PriorBank | ~129 |
| VFE dynamic | Boltzmann gate | Linear | ~119 |
| Hybrid | GELU | Linear | ~119 |

The nonlinearity type (Boltzmann gate vs GELU) contributes nothing at this scale. Both architectures converge to the same PPL regardless of FFN. The bottleneck is elsewhere.

---

## Result 2: PriorBank Decode Is a Bottleneck

Switching from PriorBank KL-decode (rank-2K=80) to linear decode (rank-K=40, unconstrained) improves PPL by ~10 points across both architectures. The KL-decode constrains logits to lie on the manifold of KL divergences to prior distributions; the linear decode can learn arbitrary discrimination directions.

| Decode | PPL |
|---|---|
| PriorBank | ~129 |
| Linear | ~119 |

The PriorBank's geometric structure (logits = negative KL to priors) costs ~10 PPL compared to unconstrained linear projection, despite having twice the effective rank (2K vs K).

---

## Result 3: KL-Attention Matches Dot-Product at Identical Learning Rates

At LR=3e-4 (the standard transformer's optimal LR), KL-attention and dot-product attention produce the same training dynamics and converge to the same PPL.

This confirms the manuscript's three-limit reduction empirically: when both architectures use the same positional encoding (RoPE), same nonlinearity (GELU), same output projection (linear), and same learning rate, the gauge-theoretic KL-attention degenerates to dot-product attention. The theoretical equivalence is not just mathematical but operational.

---

## Result 4: The Gauge Parameterization Enables 100x Larger Learning Rates

The KL-attention architecture tolerates M-step learning rates of 0.05--0.1 for embedding parameters, while the standard transformer diverges above ~3e-4. This holds even without PriorBank (plain nn.Embedding + sigma + phi tables with KL-attention scoring).

The mechanism: the backward pass through the KL divergence

$$D_{\text{KL}}(q_i \| \Omega_{ij}[q_j]) \;\ni\; (\mu_i - \Omega_{ij}\mu_j)^\top \Sigma_j^{-1} (\mu_i - \Omega_{ij}\mu_j)$$

produces gradients to $\mu_i$ that are scaled by the precision $\Sigma_j^{-1}$. Dimensions with high variance (low precision) receive small gradients; dimensions with low variance (high precision) receive large gradients. This is a natural gradient effect built into the forward computation. The dot-product $Q_i K_j^\top$ has no such per-dimension adaptive scaling.

---

## Result 5: Sample Efficiency Advantage

At matched data budget (same tokens seen), the gauge model reaches substantially lower PPL.

| Model | Steps | Tokens Seen | it/sec | Val PPL |
|---|---|---|---|---|
| Standard (LR=3e-4) | 1,000 | 8.2M | 70 | 792 |
| Hybrid (LR=0.05) | 1,000 | 8.2M | 8 | 247 |
| Standard (LR=3e-4) | 10,000 | 82M | 70 | 244 |

The hybrid reaches PPL 247 in 1k steps. The standard requires 10k steps (10x more tokens) to reach the same PPL. The gauge model is approximately **3x more sample-efficient** — it extracts more information per token due to the implicit preconditioning from KL-Mahalanobis scoring.

However, the standard transformer is 8.75x faster per step (no matrix exponentials, no pairwise KL computation). In wall-clock time, both architectures reach ~245 PPL at roughly the same moment (~125--145 seconds). The gauge model's optimization advantage is offset by its computational overhead at this scale.

---

## Result 6: The Standard Transformer Wins at Convergence

Given sufficient training, the standard transformer continues improving past where the gauge model plateaus. At K=90 with 60k steps:

| Model | Params | Test PPL |
|---|---|---|
| Gauge VFE (GL(15), K=90) | 81M | 71.6 |
| Standard (d=1280, param-matched) | 84M | 48.5 |
| Standard (d=90, embed-matched) | 4.6M | 118.6 |

The gauge model outperforms at matched embedding dimension (71 vs 119 at K=90) but underperforms at matched parameters (71 vs 48). The gauge model spends most of its parameter budget on per-token gauge frames (phi: 20M+ params) that serve the same functional role as the standard transformer's shared W_Q, W_K matrices (a few thousand params).

---

## Synthesis: What the Gauge Framework Provides

The gauge-theoretic VFE framework does not produce a more expressive model than a standard transformer. The three-limit reduction (isotropic covariance, constant gauge, absorbed projections) recovers standard attention exactly, and this equivalence is confirmed empirically.

What the framework provides:

**1. Theoretical explanation.** Standard transformer attention emerges as the degenerate limit of gauge-covariant variational free energy minimization. The $1/\sqrt{d_k}$ scaling, multi-head structure, and softmax normalization are not architectural choices but necessary consequences of constrained KL minimization on a statistical manifold.

**2. Implicit preconditioning.** The KL-Mahalanobis scoring function provides per-token per-dimension gradient scaling through the precision $\Sigma^{-1}$ in the backward pass. This enables 100x larger learning rates and 3x better sample efficiency compared to dot-product attention, at the cost of higher per-step compute.

**3. Identification of architectural constraints.** The experiments isolate exactly what limits the gauge model:
- The FFN type (Boltzmann gate vs GELU) does not matter.
- The PriorBank KL-decode costs ~10 PPL vs linear decode.
- The KL-attention matches dot-product at identical LR.
- The per-token gauge frame parameterization is 6000x more parameter-expensive than shared projections for the same functional role.

**4. RG consistency.** The standard transformer appears to be the infrared fixed point: the gauge model flows toward it during training (richer structure helps early, becomes overhead late). Early training is "UV" where gauge structure helps; late training is "IR" where the degenerate limit suffices and the structural constraints become the ceiling.

---

## Implications for the Manuscript

The BERT validation (Section 4.1) confirms an algebraic identity, not the gauge framework. These controlled ablations provide stronger empirical support for the theory:

1. **Three-limit reduction confirmed:** KL-attention = dot-product attention at matched LR and architecture (Result 3). This is a cleaner validation than the BERT correlation.

2. **Sample efficiency from information geometry:** The 3x sample efficiency advantage (Result 5) is a concrete, measurable consequence of the gauge-theoretic parameterization. It is not predicted by the algebraic identity.

3. **Honest limitations identified:** The parameter overhead (Result 6) and the computational cost (Result 5) are structural tradeoffs, not engineering limitations. The per-token gauge frame is fundamentally more expensive than shared projections.

The recommended framing: "The gauge-theoretic VFE framework provides a principled derivation of transformer attention from variational inference, with measurable optimization advantages (sample efficiency) and identifiable structural costs (parameter and compute overhead) in the unreduced regime."

---

## Experimental Configurations

### Standard Transformer
```
embed_dim=40, n_heads=4, hidden_dim=160, max_seq_len=64
batch_size=128, max_steps=15000, LR=3e-4 (uniform AdamW)
use_rope=True, rope_base=5000, dropout=0.1
Parameters: ~4M
```

### Hybrid Gauge-Attention Transformer
```
embed_dim=40, irrep_spec=[('fund', 4, 10)], hidden_dim=160, max_seq_len=64
batch_size=128, max_steps=15000, LR=3e-4 or 0.05 (AdamW)
use_rope=True, rope_base=5000, dropout=0.1
KL-attention with GL(10) gauge transport, GELU FFN
use_prior_bank=False (linear decode)
Parameters: ~24M (dominated by phi_embed: 20M)
```
