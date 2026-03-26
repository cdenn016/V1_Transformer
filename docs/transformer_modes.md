# Transformer Modes Reference

This document describes the five primary training modes and the key configuration axes of the Gauge-Transformer. All modes are launched from `transformer/train_publication.py` by setting `DEFAULT_MODE`.

---

## Primary Training Modes

### EM (Gauge VFE + Implicit Differentiation)

**Config:** `EM_CONFIG` | **Mode string:** `'em'` | **Model class:** `GaugeTransformerLM`

The EM mode is the default and most principled training configuration. It implements a gauge-covariant variational free energy (VFE) transformer with proper expectation-maximization separation: a fast E-step that refines belief parameters within the forward pass, and a slow M-step that updates embedding parameters via backpropagation through an implicit-function-theorem (IFT) scaled gradient. No neural network components (MLPs, activation functions, learned projections) appear in the representational pathway — all capacity arises from iterative VFE minimization over Gaussian belief tuples $(\mu, \Sigma, \phi)$.

#### Generative Hierarchy and Timescale Separation

The model posits a four-level generative hierarchy with three distinct timescales:

$$h \;\to\; s \;\to\; p \;\to\; q \;\to\; o$$

The hyper-prior $h$ is fixed at initialization and never learned; it serves as a regularizing anchor. The model parameters $s$ coincide with the embedding parameters $(\mu_{\mathrm{embed}}, \sigma_{\mathrm{embed}}, \phi_{\mathrm{embed}})$ and evolve on the slow M-step timescale via gradient descent. The priors $p$ are identified with $s$ (i.e., $p = s$), so the prior means and covariances are the embedding values themselves. The beliefs $q_i = \mathcal{N}(\mu_{q,i}, \Sigma_{q,i})$ at each sequence position $i$ evolve on the fast E-step timescale within a single forward pass. Observations $o$ are discrete tokens, coupled to beliefs through a linear output projection $W_{\mathrm{out}}$.

This timescale separation ensures that the E-step reads the prior parameters but does not write gradients to them. The prior covariance $\sigma_p$ is detached during E-step iterations; the M-step residual gradient from cross-entropy to $\sigma_p$ is controlled by a separate scale factor (`sigma_ce_scale`, default 0.0 for full detachment).

#### The Variational Free Energy Functional

Each forward pass minimizes the VFE functional with respect to beliefs $q$. For a sequence of $N$ tokens with belief dimension $K$, the free energy decomposes into four terms (`variational_ffn.py:26–31`):

$$F = \underbrace{\alpha \sum_{i=1}^{N} \mathrm{KL}(q_i \| p_i)}_{\text{self-coupling}} + \underbrace{\lambda_\beta \sum_{i,j} \beta_{ij} \, \mathrm{KL}(q_i \| \Omega_{ij} q_j)}_{\text{belief alignment}} + \underbrace{\lambda_\gamma \sum_{i,j} \gamma_{ij} \, \mathrm{KL}(p_i \| \Omega_{ij} p_j)}_{\text{prior alignment}} + \underbrace{\mathrm{CE}(W_{\mathrm{out}} \mu_q, \; y)}_{\text{observation likelihood}}$$

The self-coupling term anchors each belief $q_i$ to its corresponding prior $p_i$, preventing unbounded drift during E-step iterations. The belief alignment term couples beliefs across sequence positions through gauge-transported KL divergences weighted by attention coefficients $\beta_{ij}$. The prior alignment term (controlled by $\lambda_\gamma$, typically 0.0 in EM mode) couples priors in a similar fashion for meta-cognitive regularization. The observation term is the standard cross-entropy loss between the linearly projected belief means and the target token distribution.

#### Gauge Transport

Each token $i$ carries a gauge frame parameterized by Lie algebra coordinates $\phi_i \in \mathbb{R}^{n_{\mathrm{gen}}}$, where $n_{\mathrm{gen}} = K^2$ for the default $\mathrm{GL}(K)$ gauge group. The transport operator between positions $i$ and $j$ is constructed as a product of two matrix exponentials (`attention.py:289–305`, `math_utils/transport.py:81–143`):

$$\Omega_{ij} = \exp\!\Big(\sum_a \phi_i^a \, T_a\Big) \cdot \exp\!\Big(-\sum_a \phi_j^a \, T_a\Big)$$

where $\{T_a\}_{a=1}^{n_{\mathrm{gen}}}$ are the Lie algebra generators of the gauge group. This factored form ensures $\Omega_{ij} \in \mathrm{GL}^+(K)$ (positive determinant) and satisfies the flat cocycle condition $\Omega_{ij} \cdot \Omega_{jk} = \Omega_{ik}$ and self-transport identity $\Omega_{ii} = I$.

Transport acts on Gaussian sufficient statistics as follows. For means:

$$\mu_j^{(i)} = \Omega_{ij} \, \mu_j$$

For covariances, the sandwich product is mandatory (this is a hard correctness constraint):

$$\Sigma_j^{(i)} = \Omega_{ij} \, \Sigma_j \, \Omega_{ij}^\top$$

In the diagonal covariance approximation ($\Sigma = \mathrm{diag}(\sigma)$), the transported diagonal entries are computed without materializing full covariance matrices:

$$\sigma_{j,k}^{(i)} = \sum_l (\Omega_{ij})_{kl}^2 \; \sigma_{j,l}$$

This reduces memory from $O(BN^2K^2)$ to $O(BN^2K)$ while preserving the essential gauge-covariant structure.

#### KL-Divergence Attention

Attention weights replace the standard dot-product mechanism with a softmax over pairwise KL divergences between gauge-transported beliefs (`attention.py:459, 634–702`):

$$\beta_{ij} = \mathrm{softmax}_j\!\left(\frac{-\mathrm{KL}(q_i \| \Omega_{ij} q_j)}{\kappa \sqrt{K}}\right)$$

The temperature $\kappa$ controls the sharpness of attention, and the $\sqrt{K}$ factor normalizes for the growth of KL divergence with dimension (analogous to the $\sqrt{d_k}$ scaling in dot-product attention). For diagonal Gaussians, the pairwise KL takes the closed form:

$$\mathrm{KL}(q_i \| \Omega_{ij} q_j) = \frac{1}{2}\left[\sum_k \frac{\sigma_{i,k}}{\sigma_{j,k}^{(i)}} + \sum_k \frac{(\mu_{i,k} - \mu_{j,k}^{(i)})^2}{\sigma_{j,k}^{(i)}} - K + \sum_k \log\frac{\sigma_{j,k}^{(i)}}{\sigma_{i,k}}\right]$$

where $\mu_j^{(i)}$ and $\sigma_j^{(i)}$ denote transported statistics. Rotary position embeddings (RoPE) are applied to means before computing KL, introducing position dependence without modifying the covariance geometry.

#### E-Step: Natural Gradient VFE Descent

The E-step minimizes $F$ with respect to $(\mu_q, \Sigma_q, \phi)$ via natural gradient descent over `ffn_n_iterations` steps. At each iteration, attention weights $\beta_{ij}$ are recomputed from the current beliefs (dynamic $\beta$), so that beliefs and attention co-evolve. The `VariationalFFNDynamic` module (`variational_ffn.py:2220–2448`) orchestrates this loop.

**Self-coupling gradients.** The gradient of the prior-anchoring term with respect to the belief mean, for diagonal covariances (`variational_ffn.py:1581–1608`):

$$\frac{\partial F_{\mathrm{self}}}{\partial \mu_{q,k}} = \alpha \, \frac{\mu_{q,k} - \mu_{p,k}}{\sigma_{p,k}}$$

and with respect to the belief variance:

$$\frac{\partial F_{\mathrm{self}}}{\partial \sigma_{q,k}} = \frac{\alpha}{2}\left(\frac{1}{\sigma_{p,k}} - \frac{1}{\sigma_{q,k}}\right)$$

For full covariance, these become $\alpha \, \Sigma_p^{-1}(\mu_q - \mu_p)$ and $\frac{\alpha}{2}(\Sigma_p^{-1} - \Sigma_q^{-1})$ respectively.

When adaptive precision is enabled (`ffn_learnable_alpha=True`), the coupling strength becomes dimension-dependent via Gamma-Normal conjugacy: $\alpha_k = c_0 / (b_0 + \mathrm{KL}_k)$, where $c_0$ and $b_0$ are learnable per-dimension parameters. The product rule then contributes an additional correction (`variational_ffn.py:1610–1619`):

$$\frac{\partial(\alpha_k \cdot \mathrm{KL}_k)}{\partial \theta} = \alpha_k \frac{\partial \mathrm{KL}_k}{\partial \theta} - \frac{\alpha_k^2}{c_0} \, \mathrm{KL}_k \, \frac{\partial \mathrm{KL}_k}{\partial \theta}$$

This gates prior influence per dimension: dimensions where beliefs already match priors ($\mathrm{KL}_k \approx 0$) receive strong coupling, while dimensions where beliefs have diverged receive weaker coupling, allowing them to fit the data.

**Belief alignment gradients — the nonlinearity.** The gradient of the alignment term with respect to $\mu_i$ decomposes via the product rule into a direct term and a softmax coupling term (`variational_ffn.py:1650–1747`):

$$\frac{\partial F_{\mathrm{align}}}{\partial \mu_i} = \lambda_\beta \underbrace{\sum_j \beta_{ij} \frac{\partial \mathrm{KL}_{ij}}{\partial \mu_i}}_{\text{direct}} + \lambda_\beta \underbrace{\sum_j \mathrm{KL}_{ij} \frac{\partial \beta_{ij}}{\partial \mu_i}}_{\text{softmax coupling}}$$

The direct term is a $\beta$-weighted average of per-pair KL gradients, where $\partial \mathrm{KL}_{ij} / \partial \mu_i = (\mu_i - \mu_j^{(i)}) / \sigma_j^{(i)}$ for diagonal covariances. The softmax coupling term arises because $\beta_{ij}$ itself depends on $\mu_i$ through the KL divergences:

$$\frac{\partial \beta_{ij}}{\partial \mu_i} = \frac{\beta_{ij}}{\kappa\sqrt{K}} \left[\bar{g}_i - \frac{\partial \mathrm{KL}_{ij}}{\partial \mu_i}\right]$$

where $\bar{g}_i = \sum_k \beta_{ik} \, \partial\mathrm{KL}_{ik}/\partial\mu_i$ is the attention-weighted average gradient. This softmax coupling is the principled nonlinearity of the architecture — it replaces the GELU or ReLU activation in standard transformers. The term vanishes when all pairwise gradients are equal (uniform attention), and is maximally active when attention is concentrated on a few positions whose gradients differ from the mean.

The sigma alignment gradient has an analogous structure, with $\partial \mathrm{KL}_{ij}/\partial \sigma_{i,k} = \frac{1}{2}(1/\sigma_{j,k}^{(i)} - 1/\sigma_{i,k})$ and a corresponding softmax coupling through $\partial \beta_{ij}/\partial \sigma_i$ (`variational_ffn.py:1750–1770`).

**Natural gradient projection.** Raw VFE gradients are projected onto the natural gradient using geometry-aware metrics. For the belief mean, the Fisher information metric of the Gaussian family yields:

$$\Delta \mu = -\eta_\mu \, \Sigma_q \, \nabla_\mu F$$

which whitens the gradient by the current posterior covariance. For the covariance, updates follow SPD (symmetric positive-definite) manifold retraction to ensure $\Sigma$ remains positive definite after each step. For the gauge frame $\phi$, the Killing-form natural gradient is applied (see below).

**Killing-form preconditioning.** The default preconditioner for $\phi$ gradients in EM mode uses the Cartan-involution-modified Killing form of $\mathfrak{gl}(K)$ as the Riemannian metric (`gauge_preconditioner.py:196–287`). The metric tensor in generator coordinates is:

$$\tilde{g}_{ab} = 2K \, \mathrm{tr}(T_a^\top T_b) - 2\,\mathrm{tr}(T_a)\,\mathrm{tr}(T_b)$$

The first term is $2K$ times the Frobenius inner product (the Gram matrix), and the second subtracts the outer product of generator traces, which removes the center direction $\mathbb{R} \cdot I$ of $\mathfrak{gl}(K) = \mathfrak{sl}(K) \oplus \mathbb{R} \cdot I$. This metric is positive semidefinite, degenerate only on the center. Its eigenvalue structure on the standard $E_{ij}$ basis of $\mathfrak{gl}(K)$ assigns eigenvalue $2K$ to both $\mathfrak{so}(K)$ (antisymmetric) and traceless $\mathrm{sym}_0(K)$ (symmetric) directions, with the center regularized by an additive $\lambda_{\mathrm{reg}} I$ (default $\lambda_{\mathrm{reg}} = 2K$ for isotropic conditioning). The natural gradient is then:

$$\tilde{\nabla}F^a = \sum_b [\tilde{g}^{-1}]^{ab} \, \frac{\partial F}{\partial \phi^b}$$

This metric has no free parameters — it is determined entirely by the Lie algebra structure. It treats compact and non-compact directions equally (both get eigenvalue $2K$), which is appropriate when $\|\phi\|$ is small. For large $\|\phi\|$ in non-compact directions, the position-dependent pullback metric (below) provides more accurate preconditioning.

**Pullback natural gradient.** The theoretically exact natural gradient on the Lie group pulls back the bi-invariant Frobenius metric on $\mathrm{GL}(K)$ through the exponential map (`gauge_preconditioner.py:334–400`). The differential of the exponential at $X = \sum_a \phi^a T_a$ is:

$$d\exp_X(T_a) = \exp(X) \cdot \Psi(\mathrm{ad}_X)(T_a)$$

where $\Psi(z) = (e^z - 1)/z = \sum_{k=0}^{\infty} z^k/(k+1)!$ and $\mathrm{ad}_X$ is the adjoint representation. Left-translating back to the identity gives the pullback metric:

$$G_{ab}(\phi) = \big\langle \Psi(\mathrm{ad}_X)(T_a), \; \Psi(\mathrm{ad}_X)(T_b) \big\rangle_{\mathrm{gram}}$$

where $\langle A, B \rangle_{\mathrm{gram}} = \mathrm{tr}(A^\top B)$ in generator coordinates. The adjoint map is computed via the structure constants $f^c_{ab}$ defined by $[T_a, T_b] = \sum_c f^c_{ab} T_c$, giving coordinate representation $[\mathrm{ad}_X]_{bc} = \sum_a \phi^a f^c_{ab}$, and $\Psi(\mathrm{ad}_X)$ is evaluated by truncating the Taylor series at order 6.

At $\phi = 0$, $\Psi = I$ and $G$ reduces to the Gram matrix (Frobenius inner product), recovering the flat metric. At large $\|\phi\|$ in non-compact (symmetric) directions, $G$ grows exponentially — the matrix exponential amplifies gradients by $\sim e^{\|\phi\|}$ in these directions, and the natural gradient $G(\phi)^{-1} \nabla F$ automatically compensates by shrinking steps proportionally. This position-dependent correction is what the Killing form metric misses. The cost is $O(n_{\mathrm{gen}}^3)$ per token per E-step iteration, compared to $O(n_{\mathrm{gen}}^2)$ for the Killing form.

The other two preconditioning modes — `'clip'` (simple norm clipping, no geometric awareness) and `'cartan'` (Cartan decomposition with a tunable symmetric dampening factor) — are documented in the Configuration Axes section below.

#### M-Step: Implicit Function Theorem Gradient

The M-step updates embedding parameters $\theta = (\mu_{\mathrm{embed}}, \sigma_{\mathrm{embed}}, \phi_{\mathrm{embed}}, W_{\mathrm{out}})$ via backpropagation, but with a correction that accounts for the E-step optimization. In standard EM, the M-step gradient is $\partial F / \partial \theta |_{q = q^*}$, computed at the E-step fixed point $q^*(\theta)$. With finite E-step iterations, $q$ has not converged, so the total derivative requires the implicit function theorem (`variational_ffn.py:89–101`):

$$\frac{dq^*}{d\theta} = -\left(\frac{\partial^2 F}{\partial q^2}\right)^{-1} \frac{\partial^2 F}{\partial q \, \partial \theta}$$

For diagonal Gaussians, the Hessian $\partial^2 F / \partial q^2$ is diagonal in the belief dimensions, yielding per-dimension scale factors. For the mean parameters (`variational_ffn.py:169–229`):

$$s_k^{(\mu)} = \frac{\alpha / \sigma_{p,k}^2}{\alpha / \sigma_{p,k}^2 + \sum_j \beta_{ij} / \sigma_{j,k}^2} \;\in\; [0, 1]$$

The numerator is the prior precision contribution and the denominator is the total effective precision $A_k$ at the E-step fixed point. For the covariance parameters:

$$s_k^{(\sigma)} = \frac{\alpha / \sigma_{p,k}^4}{\alpha / \sigma_{p,k}^4 + \sum_j \beta_{ij} / \sigma_{j,k}^4}$$

These scale factors interpolate between two extremes: $s = 0$ corresponds to pure EM (beliefs fully converged, no gradient flows back to embeddings), while $s = 1$ corresponds to the straight-through estimator (beliefs treated as if they were the embeddings). The IFT gives the information-geometrically correct intermediate value.

Implementation uses custom `torch.autograd.Function` classes (`ImplicitEMGradient` and `ImplicitEMGradientSigma`). In the forward pass, these are identity operations returning the E-step output unchanged. In the backward pass, the gradient flowing to embedding parameters is multiplied by the precomputed scale $s_k$, while the gradient flowing to $W_{\mathrm{out}}$ passes through unscaled. When `implicit_em=True`, beliefs are detached at E-step start (proper EM boundary), and the scale factors bridge the gap between the detached E-step and the gradient-requiring M-step.

Cross-entropy gradients reach $W_{\mathrm{out}}$ directly through the final belief means. Gradients reach the embedding means $\mu_{\mathrm{embed}}$ through the IFT scale. The KL loss $\mathrm{KL}(q^* \| p)$ provides an additional direct gradient to embedding parameters when $\alpha > 0$ in the M-step loss.

#### EM Mode Configuration

| Flag | Default | Role |
|------|---------|------|
| `implicit_em` | `True` | Enable IFT-based M-step gradient scaling |
| `amortized_inference` | `False` | Detach E-step from prior gradient flow |
| `optimizer_type` | `'natural_gradient'` | Per-token block-diagonal empirical Fisher for M-step |
| `ffn_alpha` | `1.0` | E-step prior coupling $\alpha$ |
| `ffn_lambda_belief` | `1.0` | E-step belief alignment weight $\lambda_\beta$ |
| `ffn_learnable_alpha` | `True` | Bayesian precision $\alpha_k = c_0/(b_0 + \mathrm{KL}_k)$ |
| `phi_natural_gradient` | `'killing'` | Lie algebra gradient preconditioning mode |
| `evolve_sigma` | `True` | Update covariances in E-step |
| `evolve_phi` | `True` | Update gauge frames in E-step |
| `evolve_phi_e_step` | `True` | Update $\phi$ at each VFE iteration (not just post-loop) |
| `diagonal_covariance` | `True` | Use $O(K)$ diagonal instead of $O(K^2)$ full covariance |

See the Configuration Axes section below for flags shared across modes (gauge geometry, covariance representation, positional encoding, non-flat transport, DEQ).

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
