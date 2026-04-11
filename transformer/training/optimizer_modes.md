# M-Step Optimizer Modes for the Gauge-Theoretic VFE Transformer

## Introduction

The gauge-theoretic VFE transformer separates inference into a fast E-step (belief update via inner VFE iterations) and a slow M-step (parameter learning via backpropagation). The M-step updates embedding parameters — prior means $\mu_p$, prior log-variances $\log \sigma_p$, gauge frame coordinates $\phi$, and standard parameters (attention, FFN, output projection) — using one of three optimizer modes. Each mode differs in how it approximates the Fisher information matrix, which defines the natural gradient on the statistical manifold of model parameters.

The parameter update hierarchy is:

$$\text{observations} \;\to\; q_i \;(\text{beliefs, E-step}) \;\to\; p_i \;(\text{priors, M-step}) \;\to\; \mathcal{N}(0, \tfrac{1}{2\lambda}I) \;(\text{hyper-prior})$$

Weight decay $\lambda$ implements the top-level Gaussian hyper-prior: $-\log p(\theta) = \lambda \|\theta\|^2$, preventing prior drift in the absence of sufficient data coupling.

All three optimizers operate on the same parameter groups, described in Section 1. The three modes — AdamW (Section 2), Riemannian AdamW (Section 3), and Natural Gradient (Section 4) — offer increasing geometric fidelity at increasing computational cost.

---

## 1. Parameter Groups

The function `create_param_groups` partitions model parameters into groups with distinct learning rates and weight decay schedules, exploiting the natural gradient structure of the VFE hierarchy.

| Group | Parameters | Default LR | Weight Decay | Geometric Role |
|---|---|---|---|---|
| `mu_embed` | $\mu_p$ (prior means) | 0.1 | `embed_wd` | Location on statistical manifold |
| `sigma_embed` | $\log \sigma_p$ (prior log-variances) | 0.005 | `embed_wd` | Scale on statistical manifold |
| `phi_embed` | $\phi$ (Lie algebra coordinates) | 0.01 | `embed_wd` | Gauge frame on $GL^+(K)$ |
| `omega_embed` | $\Omega$ (direct group elements) | 0.01 | `embed_wd` | Gauge frame on $GL(K)$ |
| `attention` | $W_O$, $\kappa$, constant $\Omega$ | 0.01 | `wd` | Attention mechanism |
| `ffn` | VFE hyperparameters ($c_0$, $b_0$, learning rates) | 0.001 | `wd` | E-step dynamics |
| `no_decay` | LayerNorm, biases, gates | 0.001 | 0.0 | Scale-free parameters |
| `output` | Output projection (logits) | 0.001 | 0.0 | Vocabulary decoding |

Embedding weight decay (`embed_wd`, default 0.01) is distinct from general weight decay (`wd`, default 0.1). Setting `embed_wd = 0.0` yields an uninformative hyper-prior, relying entirely on the VFE loss terms ($\alpha \cdot \mathrm{KL}(q \| p)$ and $\alpha_\phi \cdot \|\phi\|^2/2$) to regularize embeddings.

---

## 2. AdamW (Default)

### Update Rule

Standard decoupled weight decay AdamW (Loshchilov and Hutter, 2019). For parameter $\theta$ with gradient $g_t = \nabla_\theta \mathcal{L}$:

$$m_t = \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t$$

$$v_t = \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = (1 - \eta \lambda) \, \theta_{t-1} - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where $\eta$ is the learning rate, $\lambda$ the weight decay coefficient, and $\epsilon$ a numerical stability constant.

### Fisher Interpretation

The second moment $v_t$ is an exponential moving average of $g_t^2$, which is a diagonal approximation to the empirical Fisher information matrix $\hat{F} = \mathbb{E}[g g^\top]$. The Adam update $\hat{m}_t / \sqrt{\hat{v}_t}$ therefore approximates a diagonal natural gradient with entry-wise variance normalization.

This diagonal approximation misses all off-diagonal correlations between dimensions. For gauge frame parameters $\phi \in \mathbb{R}^{n_\text{gen}}$, where the Lie algebra metric couples generators, this can be a poor approximation — but it is cheap and stable.

### Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `beta1` | 0.9 | First moment decay |
| `beta2` | 0.999 | Second moment decay |
| `eps` | $10^{-8}$ | Numerical stability |
| `weight_decay` | 0.1 | Decoupled L2 penalty |

### When to Use

AdamW is the safe default. Its diagonal Fisher approximation via the squared gradient EMA is crude but robust. The off-diagonal correlations captured by the full Fisher are a convergence speed luxury, not a correctness requirement. Use AdamW when stability matters more than convergence speed, or when debugging other components of the system.

---

## 3. Riemannian AdamW

### Overview

Riemannian AdamW extends AdamW by applying the geometrically correct metric tensor to gradients *before* the Adam moment update. Since the Lie algebra $\mathfrak{gl}(K)$ is a flat vector space, parallel transport is trivial and the exponential map (in the optimizer sense) is addition, so Riemannian Adam reduces to: (1) transform the gradient by the metric inverse, (2) run standard Adam on the transformed gradient.

Three parameter groups receive geometric preconditioning; all others use standard AdamW.

### Phi Parameters: Killing Form Metric

For gauge frame coordinates $\phi \in \mathfrak{gl}(K)$, the natural metric is the Cartan-involution-modified Killing form:

$$\tilde{g}_{ab} = 2K \, \mathrm{tr}(G_a^\top G_b) - 2 \, \mathrm{tr}(G_a) \, \mathrm{tr}(G_b)$$

where $\{G_a\}_{a=1}^{n_\text{gen}}$ are the Lie algebra generators. This is positive semidefinite, degenerate only on the center $\mathbb{R} \cdot I$ of $\mathfrak{gl}(K)$.

The preconditioned gradient is:

$$\tilde{\nabla}_\phi \mathcal{F} = \tilde{g}^{-1} \cdot \nabla_\phi \mathcal{F}$$

The eigenvalue structure of $\tilde{g}$ for the $E_{ij}$ basis of $\mathfrak{gl}(K)$ is:

- $\mathfrak{so}(K)$ directions (antisymmetric): eigenvalue $2K$
- $\mathrm{sym}_0(K)$ directions (symmetric traceless): eigenvalue $2K$
- Center (trace $\mathbb{R} \cdot I$): eigenvalue $0 \to$ regularized to `center_reg`

The center regularization defaults to $2K$, matching the non-center eigenvalues for isotropic conditioning (condition number $\approx 1$). Small values like $10^{-4}$ create condition numbers of $2K / \text{center\_reg}$ (e.g., $200{,}000\times$ for $K = 10$), amplifying the trace direction and causing phi runaway followed by attention collapse.

For $SO(N)$ with Frobenius-orthonormal generators ($\mathrm{tr}(G_a^\top G_b) = \delta_{ab}/2$, $\mathrm{tr}(G_a) = 0$):

$$\tilde{g} = K \cdot I \quad \Longrightarrow \quad \tilde{g}^{-1} = \frac{1}{K} \cdot I \qquad \text{(trivial scalar rescaling)}$$

### Mu Parameters: Fisher Metric for Gaussian Location

For prior mean embeddings $\mu_v$ of a Gaussian belief $\mathcal{N}(\mu_v, \Sigma_v)$, the Fisher information matrix for the location parameter is $F_\mu = \Sigma_v^{-1}$. The natural gradient is:

$$\tilde{\nabla}_\mu \mathcal{F} = F_\mu^{-1} \cdot \nabla_\mu \mathcal{F} = \Sigma_v \cdot \nabla_\mu \mathcal{F}$$

In the isotropic case $\Sigma_v = \sigma_v^2 I$, this reduces to element-wise scaling by the current variance $\sigma_v^2$ per token per dimension. High-uncertainty dimensions receive larger steps — the optimizer explores more where it knows less.

Implementation: the variance $\sigma_v^2 = \exp(\log \sigma^2_v)$ is read from the model's `log_sigma_diag` parameter and clamped to $[10^{-6}, 10]$.

### Sigma Parameters: Fisher Metric for Log-Variance

For the log-variance parameterization $\eta = \log \sigma^2$, the Fisher information is constant:

$$F_{\eta\eta} = \frac{1}{2}$$

The natural gradient is therefore:

$$\tilde{\nabla}_\eta \mathcal{F} = F_{\eta\eta}^{-1} \cdot \nabla_\eta \mathcal{F} = 2 \cdot \nabla_\eta \mathcal{F}$$

This is applied as a fixed factor-of-2 rescaling of the Euclidean gradient, equivalent to doubling the effective learning rate for sigma parameters.

### Full Update

For a phi parameter with Euclidean gradient $g_t$:

$$\tilde{g}_t = \tilde{g}^{-1} \cdot g_t \qquad \text{(Killing preconditioning)}$$

$$m_t = \beta_1 \, m_{t-1} + (1 - \beta_1) \, \tilde{g}_t$$

$$v_t = \beta_2 \, v_{t-1} + (1 - \beta_2) \, \tilde{g}_t^2$$

$$\theta_t = (1 - \eta \lambda) \, \theta_{t-1} - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

The preconditioning happens *before* the Adam EMA, so Adam's adaptive learning rate operates on the geometrically meaningful gradient, not the raw Euclidean one.

### Hyperparameters

All standard AdamW hyperparameters apply. The Killing metric inverse is precomputed once at optimizer creation from `model.generators` and has no tunable parameters (given `center_reg = 2K`).

### When to Use

Riemannian AdamW is the recommended choice for production training. It provides geometric preconditioning (Killing metric on $\phi$, Fisher on $\mu$, $2\times$ on $\sigma$) while Adam provides momentum and per-coordinate adaptation that prevents the mean-field collapse that can occur with raw natural gradients.

---

## 4. Natural Gradient Optimizer

### Overview

The Natural Gradient optimizer maintains a full $K \times K$ empirical Fisher information block per vocabulary token, updated via an exponential moving average of gradient outer products. This captures per-dimension correlations that diagonal approximations (Adam) miss.

### Fisher Estimation

For each vocabulary token $v$ with gradient $g_v \in \mathbb{R}^K$, the empirical Fisher block is updated as:

$$\hat{F}_v^{(t)} = (1 - \rho) \, \hat{F}_v^{(t-1)} + \rho \, g_v g_v^\top$$

where $\rho$ is the EMA decay rate. For early steps ($t < 20$), $\rho$ is clamped to $\min(\rho, \, 2/(t+1))$ for bias correction — preventing the Fisher from being dominated by the zero initialization.

Only tokens appearing in the current batch have their Fisher blocks updated (sparse update).

### Natural Gradient Step

The parameter update for embedding token $v$ is:

$$\theta_v \leftarrow \theta_v - \eta \, (\hat{F}_v + \lambda I)^{-1} \, g_v$$

where $\lambda$ is the Tikhonov damping parameter and $I$ is the $K \times K$ identity. The linear system $(\hat{F}_v + \lambda I) \, \tilde{g}_v = g_v$ is solved via `torch.linalg.solve` (batched Cholesky).

### Gradient Clipping

After the Fisher solve, the natural gradient is clipped to prevent explosion from ill-conditioned Fisher blocks:

$$\tilde{g}_v \leftarrow \tilde{g}_v \cdot \min\!\left(1, \; \frac{r_\text{max} \, \|g_v\|}{\|\tilde{g}_v\|}\right)$$

where $r_\text{max} = 10$. This ensures the natural gradient is at most $10\times$ the Euclidean gradient in norm.

### Non-Embedding Parameters

For non-embedding parameters (1D tensors or small 2D tensors with last dimension $< 4$), the optimizer falls back to plain gradient descent with decoupled weight decay:

$$\theta \leftarrow (1 - \eta \lambda) \, \theta - \eta \, g$$

No Fisher tracking is performed for these parameters.

### Damping: The Critical Hyperparameter

The damping parameter $\lambda$ controls the behavior for rarely-seen tokens. When token $v$ has been seen infrequently, $\hat{F}_v \approx 0$ and:

$$(\hat{F}_v + \lambda I)^{-1} \approx \frac{1}{\lambda} I$$

This amplifies the gradient by $1/\lambda$, clipped to $r_\text{max} = 10$. The dynamics are:

| $\lambda$ | Max amplification | Fisher threshold | Failure mode |
|---|---|---|---|
| $10^{-4}$ | $10{,}000\times$ (clipped to $10\times$) | $\hat{F}$ must reach $\sim 10^{-4}$ to dominate | Rare tokens systematically over-updated $\to$ embedding homogenization $\to$ attention collapse |
| $10^{-2}$ | $100\times$ (clipped to $10\times$) | $\hat{F}$ must reach $\sim 10^{-2}$ to dominate | Moderate; Fisher estimate stabilizes within a few token appearances |
| $10^{-1}$ | $10\times$ (clipped to $10\times$) | $\hat{F}$ must reach $\sim 10^{-1}$ to dominate | Conservative; natural gradient effect is mild until Fisher is well-estimated |

At $\lambda = 10^{-4}$, the $10\times$ clipping fires on every rare-token step, creating a systematic $10\times$ over-update relative to the Euclidean gradient. This happens for tokens seen fewer than $\sim 1/\rho \approx 20$ times. Over thousands of steps, this homogenizes rare embeddings (they all drift toward a similar direction defined by the clipped gradient), destroying the discriminative structure that attention depends on.

The recommended setting is $\lambda = 10^{-2}$.

### Computational Cost

| Operation | Cost per step | Notes |
|---|---|---|
| Gradient outer product | $O(|B| \cdot K^2)$ | $|B|$ = unique tokens in batch |
| Fisher solve | $O(|B| \cdot K^3)$ | Batched Cholesky via `linalg.solve` |
| Fisher storage | $O(V \cdot K^2)$ | For $V = 50{,}257$, $K = 64$: $\sim 819$ MB |
| Total per step | $O(|B| \cdot K^3)$ | Dominated by the solve |

For large vocabularies or large $K$, the memory cost of storing $V$ blocks of size $K \times K$ can be prohibitive. Consider reducing vocabulary size or embedding dimension if Fisher storage exceeds available GPU memory.

### Hyperparameters

| Parameter | Config key | Default | Recommended | Notes |
|---|---|---|---|---|
| EMA decay | `fisher_ema_decay` | 0.95 | 0.9 | Faster adaptation early; 0.95 is sluggish for the first $\sim 20$ steps |
| Damping | `fisher_damping` | $10^{-4}$ | $10^{-2}$ | **The default in `config.py` ($10^{-4}$) is too small.** Use $\geq 10^{-2}$ |
| Learning rate | `learning_rate` | $3 \times 10^{-4}$ | $3 \times 10^{-4}$ | Standard; the Fisher preconditioning handles per-parameter scaling |
| Weight decay | `weight_decay` | 0.1 | 0.1 | Decoupled; applied to all tokens every step |

### Diagnostic Output

The optimizer exposes `get_grad_norms() -> Dict[str, Dict[str, float]]`, returning per-group Euclidean and natural gradient norms from the last step. The ratio $\|\tilde{g}\| / \|g\|$ indicates how much the Fisher preconditioning modifies the gradient direction. A ratio near 1.0 means the Fisher is approximately identity (unhelpful); a large ratio means significant preconditioning is occurring.

### When to Use

The Natural Gradient optimizer is the most theoretically principled but the least stable. It is best suited for small-vocabulary experiments where the $O(V \cdot K^2)$ memory cost is manageable and the per-token Fisher has enough gradient signal to converge. For large-scale training, prefer Riemannian AdamW, which provides geometric preconditioning without the per-token Fisher overhead.

---

## 5. Comparison and Recommendations

| Property | AdamW | Riemannian AdamW | Natural Gradient |
|---|---|---|---|
| Fisher approximation | Diagonal (per-element $g^2$ EMA) | Block-diagonal by group (Killing + Fisher) | Per-token $K \times K$ blocks |
| Off-diagonal correlations | None | Within-group (Killing metric) | Full per-token |
| Momentum | Yes ($\beta_1$) | Yes ($\beta_1$) | No |
| Adaptive LR | Yes ($1/\sqrt{v_t}$) | Yes ($1/\sqrt{v_t}$) | No (fixed $\eta$) |
| Memory overhead | $2 \times |\theta|$ (moments) | $2 \times |\theta|$ + $n_\text{gen}^2$ | $V \times K^2$ (Fisher blocks) |
| Compute overhead | $O(|\theta|)$ | $O(|\theta| + n_\text{gen}^2)$ | $O(|B| \cdot K^3)$ |
| Free parameters | $\beta_1, \beta_2, \epsilon$ | Same as AdamW | $\rho, \lambda$ |
| Stability | High | High | Moderate (damping-sensitive) |

### Decision Guide

1. **Start with `riemannian_adam`**: geometric preconditioning plus Adam's stability. No additional hyperparameters to tune beyond standard AdamW.

2. **Fall back to `adamw`** if Riemannian Adam shows no benefit or you need to isolate optimizer effects during debugging. AdamW's diagonal Fisher is crude but never harmful.

3. **Use `natural_gradient` only** for small-vocabulary experiments ($V < 10{,}000$) where you want maximal Fisher fidelity and can afford the memory. Set `fisher_damping >= 1e-2` and `fisher_ema_decay = 0.9`.

### Configuration Example

```python
# === M-step: Optimizer (recommended) ===
'optimizer_type':        'riemannian_adam',
'fisher_ema_decay':      0.95,    # unused by riemannian_adam
'fisher_damping':        1e-2,    # unused by riemannian_adam, but sane if switching

# === M-step: Optimizer (natural gradient, small vocab only) ===
'optimizer_type':        'natural_gradient',
'fisher_ema_decay':      0.9,     # faster early adaptation
'fisher_damping':        1e-2,    # NOT 1e-4 — prevents rare-token amplification bug
```

---

## 6. Relationship to E-Step Preconditioning

The M-step optimizers described here are distinct from the E-step phi preconditioning in `gauge_preconditioner.py`, which offers four modes for the *inner* VFE loop:

| Mode | Cost | Position-dependent | Free parameters |
|---|---|---|---|
| `clip` | $O(n_\text{gen})$ | No | Clip threshold |
| `cartan` | $O(n_\text{gen}^2)$ | No | `sym_dampening` |
| `killing` | $O(n_\text{gen}^2)$ | No | None (metric from algebra) |
| `pullback` | $O(n_\text{gen}^3)$ | Yes | `series_order` |

The M-step Riemannian AdamW uses the same Killing form metric (`build_killing_form_preconditioner`) as the E-step `killing` mode, but applies it within the Adam framework rather than as a raw gradient transformation. The two systems are complementary: E-step preconditioning stabilizes belief inference within a single forward pass, while M-step preconditioning stabilizes parameter learning across training steps.
