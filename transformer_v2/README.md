# Gauge-Transformer v2

Refactored gauge-theoretic transformer for language modeling.
Config-driven architecture — every module receives a single `GaugeTransformerConfig` dataclass
instead of dozens of kwargs.

## Quick Start

```bash
# Edit MODEL_CONFIG and TRAIN_CONFIG in the file, then:
python transformer_v2/train_v2.py

# Or override via CLI:
python transformer_v2/train_v2.py --device cuda --max_steps 50000
python transformer_v2/train_v2.py --embed_dim 64 --n_layers 4 --dataset wikitext-2
python transformer_v2/train_v2.py --learning_rate 3e-4   # single LR mode
python transformer_v2/train_v2.py --use_p_flow --use_delta_rule
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `wikitext-103` | `wikitext-2` or `wikitext-103` |
| `--device` | `cpu` | `cpu`, `cuda`, or `auto` |
| `--max_steps` | (from config) | Total training steps |
| `--batch_size` | (from config) | Batch size |
| `--embed_dim` | (from config) | Embedding dimension K |
| `--n_layers` | (from config) | Transformer depth |
| `--max_seq_len` | (from config) | Context length N |
| `--gauge_group` | (from config) | `SO3`, `SON`, or `GLK` |
| `--gauge_dim` | (from config) | N for SO(N) |
| `--n_vfe_iterations` | (from config) | E-step iterations per forward |
| `--learning_rate` | — | Override all LRs (disables param groups) |
| `--checkpoint_dir` | `checkpoints_v2` | Output directory |
| `--use_wandb` | off | Enable W&B logging |
| `--use_p_flow` | off | EMA prior evolution |
| `--use_delta_rule` | off | Backprop-free W_out |
| `--seed` | 6 | Random seed |

### Output Files

```
checkpoints_v2/
  experiment_config.json   # Full config + git hash + system info
  metrics.csv              # Per-step training metrics
  best_model.pt            # Best validation checkpoint
  final_model.pt           # End-of-training checkpoint
  result_summary.json      # Final results summary
```

---

## Architecture

```
token_ids (B, N)
    │
    ▼
GaugeTokenEmbedding ──→ (μ, Σ, φ)     beliefs, covariances, gauge frames
    │
    ▼
GaugePositionalEncoding ──→ φ_composed  compose token + positional frames
    │
    ▼
┌─ GaugeTransformerBlock ×n_layers ──────────────────────────┐
│                                                             │
│   LayerNorm → IrrepMultiHeadAttention → Dropout → Residual │
│       β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / κ√K)        │
│                                                             │
│   LayerNorm → VariationalFFNDynamic → Residual              │
│       E-step: evolve (μ, Σ, φ) via natural gradient VFE    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
LayerNorm → Linear(K → V) → logits (B, N, V)
```

### Attention: KL-Divergence (No W_Q, W_K)

Attention weights emerge from information geometry — no learned query/key projections:

```
β_ij = softmax_j( -KL(q_i || Ω_ij[q_j]) / (κ · √K) )
```

Where `Ω_ij = exp(φ_i) exp(-φ_j)` is the parallel transport operator on the gauge bundle.

Multi-head attention uses irrep block structure:
- **SO(3):** heads = Wigner D-matrix irreps (ℓ=0,1,2,...)
- **SO(N):** heads = irrep blocks from generators
- **GL(K):** heads = block-diagonal GL(d_head) subgroups

### VFE FFN: Variational Free Energy Belief Evolution

Each E-step iteration performs dynamic-β VFE descent:

1. **Recompute β** from current beliefs: `β = softmax(-KL(q||Ω[q])/κ)`
2. **Compute gradients** `∂F/∂μ`, `∂F/∂σ` (self-coupling + alignment + softmax coupling)
3. **Natural gradient on μ:** `μ ← μ - η · Σ · ∂F/∂μ` (Fisher metric)
4. **SPD retraction on Σ:** preserves positive-definiteness
5. **Lie group retraction on φ:** with Cartan/Killing/pullback preconditioning

### Token Embeddings

Each token maps to a full agent belief:

```python
token_id → (μ_i, Σ_i, φ_i)
```

- `μ_i ∈ ℝ^K` — belief mean
- `Σ_i ∈ SPD(K)` — covariance (diagonal or full)
- `φ_i ∈ g` — gauge frame (Lie algebra element)

---

## Training Loss: Six-Term VFE

```
L = CE                                                  # observation likelihood
  + α   · Σ KL(q_i || p_i)                             # self-coupling
  + λ_β · Σ β_ij · KL(q_i || Ω_ij q_j)                # belief coupling
  + λ_γ · Σ γ_ij · KL(s_i || Ω_ij s_j)                # model coupling
  + λ_h · Σ KL(s_i || h)                               # hyper-prior
  + (α_φ/2) · Σ ||φ_i||²                               # gauge prior
```

Terms 1-3 operate on **beliefs** q (fast E-step). Terms 4-6 operate on **models** s (slow M-step).

Setting `λ_γ = λ_h = α_φ = 0` recovers the standard transformer training loss.

## Hierarchical Bayesian Structure

Weight decay on embedding parameters implements the top level of the VFE hierarchy:

```
Level 0:  x_i                    observations
Level 1:  q_i = N(μ_q, Σ_q)     beliefs        (VFE E-step)
Level 2:  p_i = N(μ_p, Σ_p)     priors         (M-step / backprop)
Level 3:  N(0, 1/(2·wd))        hyper-prior    (weight decay)
```

- **Level 1→2:** `α · KL(q||p)` pulls beliefs toward priors
- **Level 2→3:** Weight decay pulls priors toward zero (Gaussian hyper-prior)
- **Level 2→2:** `α_φ · ||φ||²` gauge prior on frames (also a hyper-prior)

All embedding groups (μ, Σ, φ) receive weight decay. Only biases and LayerNorm are excluded.

---

## Optimizer Parameter Groups

With `use_param_groups=True`, six groups with independent learning rates:

| Group | Parameters | Default LR | Weight Decay |
|-------|-----------|------------|--------------|
| `mu_embed` | Token means (μ_p) | 0.05 | `weight_decay` |
| `sigma_embed` | Covariances (log Σ_p) | 0.005 | `weight_decay` |
| `phi_embed` | Gauge frames (φ_p) | 0.005 | `weight_decay` |
| `attention` | Per-head κ, W_O | 0.005 | `weight_decay` |
| `ffn` | VFE step size, κ_heads, norms | 0.05 | `weight_decay` |
| `output` | W_out projection | 0.05 | `weight_decay` |
| `no_decay` | Biases, LayerNorm | 0.05 | 0.0 |

---

## Configuration Reference

### GaugeTransformerConfig (model architecture)

```python
from transformer_v2 import GaugeTransformerConfig

config = GaugeTransformerConfig(
    # Architecture
    vocab_size=50257,          # overridden by tokenizer
    embed_dim=10,              # belief dimension K
    n_layers=1,                # transformer depth
    max_seq_len=128,           # context length N

    # Gauge group
    gauge_group='GLK',         # 'SO3' | 'SON' | 'GLK'
    gauge_dim=10,              # N for SO(N), K for GL(K)
    gauge_mode='learned',      # 'learned' | 'trivial' (Ω=I)
    use_multi_irrep=True,
    irrep_spec=[('fund', 1, 10)],

    # Covariance
    diagonal_covariance=True,
    evolve_sigma=True,
    evolve_phi=True,
    evolve_phi_e_step=True,    # phi updates during E-step iterations

    # VFE E-step (inside FFN)
    alpha_ffn=1.0,             # precision α
    kappa_ffn=1.0,             # softmax temperature
    lambda_beta_ffn=1.0,       # belief coupling λ_β
    n_vfe_iterations=1,        # E-step iterations per forward

    # Training loss (M-step)
    alpha_loss=0.1,            # KL(q||p) weight
    lambda_beta_loss=0.0,      # belief alignment weight
    alpha_phi_loss=0.01,       # gauge prior weight

    # Phi evolution
    phi_natural_gradient='pullback',  # 'clip'|'cartan'|'killing'|'pullback'

    # Attention
    mask_self_attention=True,
    use_rope=True,
    per_head_kappa=True,
    use_output_projection=True,
    multihead_vfe=True,

    # Embeddings
    mu_init_std=1.0,
    tie_embeddings=False,
    use_positional_embedding=False,

    # Regularization
    dropout=0.0,
    use_layernorm=True,
    use_residual=True,
)
```

### TrainingConfig (optimizer, schedule, logging)

```python
from transformer_v2 import TrainingConfig

train_config = TrainingConfig(
    # Per-group learning rates
    use_param_groups=True,
    mu_lr=0.05,
    sigma_lr=0.005,
    phi_lr=0.005,
    attention_lr=0.005,
    ffn_lr=0.05,
    output_lr=0.05,

    # Optimizer
    weight_decay=0.01,         # hyper-prior precision
    grad_clip=1.0,

    # Schedule
    warmup_steps=100,
    max_steps=12500,
    lr_decay='cosine',         # 'cosine' | 'linear' | 'constant'

    # VFE loss overrides (None = use model config)
    alpha=0.1,
    lambda_beta=0,
    alpha_phi=0.01,

    # Logging
    log_every=100,
    eval_every=1000,
    save_every=25000,
    checkpoint_dir='checkpoints_v2',
    device='cpu',
)
```

---

## API

### Model

```python
from transformer_v2 import GaugeTransformerConfig, GaugeTransformerLM

config = GaugeTransformerConfig(vocab_size=50257, embed_dim=64)
model = GaugeTransformerLM(config)

# Basic forward
logits = model(token_ids)                           # (B, N, V)

# Forward with attention tracking (for VFE loss)
logits, attn_info = model.forward_with_attention(token_ids)
# attn_info: beta (n_layers, B, n_heads, N, N), kl, mu, sigma, phi, priors

# Forward with RG flow tracking
logits, rg_info = model.forward_with_rg_tracking(token_ids)
# rg_info: beta_history across VFE iterations

# Autoregressive generation
output_ids = model.generate(prompt_ids, max_new_tokens=100, temperature=0.8, top_k=50)
```

### Training

```python
from transformer_v2 import TrainingConfig, Trainer
from transformer.data.datasets import create_dataloaders

train_loader, val_loader, vocab_size = create_dataloaders(
    max_seq_len=128, batch_size=64, dataset='wikitext-103',
)

trainer = Trainer(model, train_loader, val_loader, train_config)

# Built-in training loop
trainer.train()

# Or custom loop
for batch in train_loader:
    metrics = trainer.train_step(batch)    # forward + backward + optimizer
    val_metrics = trainer.validate()       # val/loss, val/ce_loss, val/perplexity
    trainer.save_checkpoint('model.pt')
    trainer.step += 1
```

### Loss

```python
from transformer_v2 import compute_vfe_loss, compute_vfe_loss_from_config

# Explicit loss weights
loss, metrics = compute_vfe_loss(model, token_ids, targets, alpha=0.1, lambda_beta=0.5)

# Use model.config defaults
loss, metrics = compute_vfe_loss_from_config(model, token_ids, targets)
```

---

## File Structure

```
transformer_v2/
  config.py              GaugeTransformerConfig dataclass
  model.py               GaugeTransformerLM (forward, generate, p-flow, delta rule)
  blocks.py              GaugeTransformerBlock, GaugeTransformerStack
  attention.py           KL-divergence attention, IrrepMultiHeadAttention, RoPE
  variational_ffn.py     VariationalFFNDynamic (E-step VFE belief evolution)
  kl_ops.py              Pairwise KL computation, transport operators
  loss.py                Six-term VFE training loss
  train.py               TrainingConfig, Trainer
  train_v2.py            Click-to-run training script
  embeddings.py          GaugeTokenEmbedding, GaugePositionalEncoding
  prior_bank.py          PriorBank (pure FEP mode)
  generators.py          Lie algebra generators (SO3, SON, GLK)
  gauge_utils.py         Matrix exp, SPD retraction, phi retraction
  gauge_preconditioner.py  Cartan/Killing/pullback natural gradient
```

### Legacy Compatibility

```python
# Convert v1 config dict to v2 dataclass
config = GaugeTransformerConfig.from_legacy_dict(old_config_dict)
```
