# Transformer Modes Reference

This document describes the five primary training modes and the key configuration axes of the Gauge-Transformer. All modes are launched from `transformer/train_publication.py` by setting `DEFAULT_MODE`.

---

## Primary Training Modes

### EM (Gauge VFE + Implicit Differentiation)

**Config:** `EM_CONFIG` | **Mode string:** `'em'` | **Model class:** `GaugeTransformerLM`

The default and most principled mode. Gauge-covariant variational free energy transformer with proper expectation-maximization separation.

**E-step:** Natural gradient descent on the VFE with respect to belief parameters `(mu_q, Sigma_q, phi)` inside the forward pass. Fisher-preconditioned mean updates, SPD retraction for covariance, and Killing-form preconditioning for gauge frames. Adaptive prior coupling `alpha_i = c0 / (b0 + KL)` gates prior influence per dimension.

**M-step:** Backpropagation through an IFT-scaled gradient. The scale factor `s_k = (alpha / sigma_p^2) / A_k`, where `A_k` is the effective precision at the E-step fixed point, replaces the ad-hoc straight-through estimator (`s=1`) with the information-geometrically correct value. Cross-entropy gradients flow to `W_out` directly; gradients to embeddings flow via the IFT scale.

**Hierarchy:** `h` (fixed hyper-prior) -> `s` (embedding parameters) -> `p = s` (priors) -> `q` (E-step beliefs) -> observations.

Key flags:
- `implicit_em=True` enables IFT M-step scaling
- `amortized_inference=False` detaches E-step from prior gradient flow
- `optimizer_type='natural_gradient'` uses per-token block-diagonal empirical Fisher

### Hebbian (Gauge VFE + P-flow / Delta Rule)

**Config:** `HEBBIAN_CONFIG` | **Mode string:** `'hebbian'` | **Model class:** `GaugeTransformerLM`

Same gauge-VFE architecture as EM, but all parameter learning is local and backprop-free.

**E-step:** Identical to EM mode -- natural gradient VFE descent on `(mu_q, Sigma_q, phi)`.

**M-step (no backprop):**
- `mu_embed`, `sigma_embed`: P-flow exponential moving average toward successful beliefs, weighted by prediction error
- `phi_embed`: P-flow EMA toward E-step evolved phi (detached from computation graph)
- `W_out`: Delta rule `DeltaW = eta * (target - pred) outer mu^T` (Widrow-Hoff)

**Loss:** Cross-entropy only (`alpha=0`, `beta=0`). VFE regularizers are implicit in E-step dynamics, not in the training loss.

Key flags:
- `use_p_flow=True`, `use_delta_rule_w_out=True`
- `detach_phi=True` prevents backprop through gauge frames
- `p_flow_ema_decay=0.95`, `delta_rule_lr=0.1`

### Pure VFE (No Autograd, No Optimizer)

**Config:** `PURE_VFE_CONFIG` | **Mode string:** `'pure_vfe'` | **Model class:** `PureVFETransformer`

The purest realization: no `nn.Module`, no autograd, no optimizer. The entire system operates through analytic natural gradient descent on the gauge-covariant VFE.

**Architecture:** A prior bank of raw tensors -- one Gaussian `N(mu_v, Sigma_v)` per vocabulary token, with associated gauge frames. No linear projections, no output head. Logits are computed as `-KL(q || pi_v)`.

**Inference:** E-step VFE descent replaces the forward pass entirely.

**Learning:** M-step natural gradient on prior bank parameters with analytic closed-form gradients.

Key flags:
- `gauge_param='omega'` (direct GL(K)) or `'phi'` (Lie algebra)
- `n_esteps=12` controls inference depth (replaces network depth)
- `eta_E=0.1` (E-step step size), `eta_M=0.05` (M-step step size)
- `causal=True` for autoregressive masking

Files: `transformer/pure_vfe/model.py`, `inference.py`, `learning.py`, `config.py`

### Standard (Baseline)

**Config:** `STANDARD_CONFIG` | **Mode string:** `'standard'` | **Model class:** `StandardTransformerLM`

Conventional dot-product attention + learned MLP baseline for fair comparison. Parameter-matched to the gauge models at 1.52M parameters.

- **Attention:** `Q * K^T / sqrt(d)` with softmax
- **FFN:** `Linear -> GELU -> Linear`
- **Learning:** Full backpropagation via AdamW
- **Position:** Learned positional embeddings

### Standard Attention-Only (Ablation)

**Config:** `STANDARD_ATTN_ONLY_CONFIG` | **Mode string:** `'standard_attn_only'` | **Model class:** `StandardTransformerLM`

Standard transformer with the FFN disabled (`disable_ffn=True`) to isolate the attention mechanism contribution. Uses `d_model=90`, 9 heads with `head_dim=10` matching the gauge model's head dimension.

---

## Configuration Axes

### Gauge Geometry (`gauge_mode`)

Controls how transport operators Omega are computed.

| Mode | Behavior | Use case |
|------|----------|----------|
| `'learned'` | Per-token learnable frames `phi_i`; `Omega_ij = exp(phi_i * G) * exp(-phi_j * G)` | Default for EM, Hebbian, Pure VFE |
| `'trivial'` | `Omega = I` everywhere (identity transport) | Ablation: removes gauge effects |
| `'constant'` | Per-head learnable `Omega`, same for all token pairs | Intermediate: per-head but not per-token |

### Covariance Representation

| Setting | Shape | Cost | Description |
|---------|-------|------|-------------|
| `diagonal_covariance=False` | `(B, N, K, K)` | `O(N^2 K^2)` | Full SPD covariance matrices |
| `diagonal_covariance=True` | `(B, N, K)` | `O(N^2 K)` | Diagonal variances only |
| `isotropic_covariance=True` | `(B, N, 1)` | `O(N^2)` | Scalar `sigma^2 * I`; recovers standard attention when combined with `gauge_mode='trivial'` |

When using diagonal covariance, `exact_diagonal_transport` controls whether transport uses the exact sandwich product `diag(Omega * diag(sigma) * Omega^T)` (slower, exact) or an approximation (faster).

### Phi Gradient Preconditioning (`phi_natural_gradient`)

Four modes for preconditioning gradients on the Lie algebra, implemented in `transformer/core/gauge_preconditioner.py`.

| Mode | Cost | Description |
|------|------|-------------|
| `'clip'` | `O(n_gen)` | Simple norm clipping; ignores Lie algebra structure |
| `'cartan'` | `O(n_gen^2)` | Cartan decomposition: decomposes `gl(K) = so(K) + sym(K)`, dampens non-compact directions. Free parameter: `sym_dampening` |
| `'killing'` | `O(n_gen^2)` | Killing form natural gradient; metric determined entirely by algebra structure. No free parameters. Default for EM |
| `'pullback'` | `O(n_gen^3)` | Full pullback Riemannian metric through the exponential map. Position-dependent, theoretically exact, most expensive |

### Belief Evolution Flags

| `evolve_sigma` | `evolve_phi` | Behavior |
|-----------------|--------------|----------|
| `True` | `True` | Full evolution: means, covariances, and gauge frames all updated in E-step |
| `True` | `False` | Sigma-only: gauge frames static |
| `False` | `True` | Phi-only: covariances static |
| `False` | `False` | Static beliefs: minimal VFE dynamics |

Additionally, `evolve_phi_e_step=True` updates phi during each E-step VFE iteration (not just between iterations).

### Gauge Group and Parameterization

**Gauge group** (`gauge_group`):
- `'SO3'`: 3 generators, compact rotations
- `'SON'`: `N(N-1)/2` generators for dimension N
- `'GLK'`: `K^2` generators, non-compact general linear group (default for EM/Hebbian)

**Parameterization** (`gauge_param`):
- `'phi'`: Lie algebra parameterization. `Omega = exp(phi * G)`. Requires matrix exponentials. Reaches `GL+(K)` only (positive determinant)
- `'omega'`: Direct matrix parameterization. No matrix exp needed. Covers full `GL(K)` including reflections

**Irrep specification** (`irrep_spec`): List of `(type, multiplicity, dim)` tuples defining the block-diagonal structure. Types: `'scalar'` (dim 1), `'fund'` (dim N), `'wedge2'` (dim N(N-1)/2), `'sym2'` (dim N(N+1)/2 - 1).

### Positional Encoding

| Setting | Description |
|---------|-------------|
| `use_rope=True` | Rotary position embeddings: position-dependent `SO(2)^{K/2}` rotations on mu before KL |
| `use_rope=False` | No positional rotations |
| `pos_encoding_mode='learned'` | Learned positional embeddings (standard baseline) |
| `pos_encoding_mode='none'` | No additive positional encoding (used with RoPE) |

### Optimizer Types (`optimizer_type`)

| Type | Description |
|------|-------------|
| `'adamw'` | Standard AdamW with diagonal Fisher EMA |
| `'riemannian_adam'` | AdamW + Killing-form metric for phi + Fisher metric for location parameters |
| `'natural_gradient'` | Per-token block-diagonal empirical Fisher. Cost: `O(V * K^3)` per step. Controlled by `fisher_ema_decay` and `fisher_damping` |

### Non-flat Transport (Holonomy)

When `non_flat_transport=True`, transport acquires an edge-local connection `delta_ij`:

```
Omega_ij = exp(phi_i * G) * exp(alpha * delta_ij * G) * exp(-phi_j * G)
```

The connection is zero-initialized so the model starts flat and learns curvature only where data warrants it. Holonomy `H_ijk = Omega_ij * Omega_jk * Omega_ki != I` when `delta != 0`.

Key parameters:
- `cocycle_relaxation`: Scale for `delta_ij` (0 = flat, 1 = fully non-flat)
- `connection_type`: `'bilinear'` or `'mlp'`
- `holonomy_penalty`: Regularizer strength for `||H_ijk - I||^2_F`

Implementation: `transformer/core/connection.py`

### Deep Equilibrium (DEQ) Mode

When `use_deq=True`, implicit differentiation is used at the E-step fixed point instead of unrolling through all VFE iterations.

- `deq_neumann_terms`: Number of Neumann series terms for the backward pass
- `deq_include_phi`: Include phi in the joint fixed-point system `(mu, sigma, phi)`

### Implicit EM vs. Amortized Inference

- `implicit_em=True`: IFT-based M-step with principled gradient scaling `s_k = (alpha / sigma_p^2) / A_k`
- `amortized_inference=True`: Gradient flows through priors for learned E-step initialization (straight-through)
- `amortized_inference=False`: Detached E-step (no gradient from beliefs back to prior parameters)

---

## Mode Selection Guide

| Goal | Mode |
|------|------|
| Best principled performance | `em` with `implicit_em=True`, `optimizer_type='natural_gradient'` |
| Biologically plausible learning | `hebbian` with P-flow and delta rule |
| Purest VFE (no neural components) | `pure_vfe` |
| Fair baseline comparison | `standard` (parameter-matched) |
| Ablation: attention contribution | `standard_attn_only` |
| Ablation: gauge effects | Any gauge mode with `gauge_mode='trivial'` |
| Ablation: covariance role | Set `isotropic_covariance=True` |

---

## File Reference

| Component | File |
|-----------|------|
| Entry point (all modes) | `transformer/train_publication.py` |
| Gauge model | `transformer/core/model.py` |
| KL attention | `transformer/core/attention.py` |
| VFE FFN (E-step) | `transformer/core/variational_ffn.py` |
| Gauge blocks | `transformer/core/blocks.py` |
| Phi preconditioning | `transformer/core/gauge_preconditioner.py` |
| Block config | `transformer/core/block_config.py` |
| Pure VFE model | `transformer/pure_vfe/model.py` |
| Standard baseline | `transformer/baselines/standard_transformer.py` |
| Training config | `transformer/training/config.py` |
| Non-flat connection | `transformer/core/connection.py` |
