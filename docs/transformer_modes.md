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

The Hebbian mode uses the same gauge-VFE architecture as EM but replaces all backpropagation-based parameter learning with local, biologically plausible update rules. The E-step is identical to EM mode — natural gradient VFE descent on $(\mu_q, \Sigma_q, \phi)$ as described above. The M-step replaces the global loss-gradient-optimizer loop with three local learning rules: P-flow for embeddings, P-flow for gauge frames, and the Widrow-Hoff delta rule for the output projection. No gradient flows backward through the computation graph; all parameter updates are computed from locally available quantities.

The M-step loss is pure cross-entropy with all VFE regularizers set to zero ($\alpha = 0$, $\beta = 0$, $\alpha_\phi = 0$, $\lambda_h = 0$). The VFE regularization is implicit in the E-step dynamics — priors anchor beliefs and gauge transport structures attention — but these terms do not appear in the training loss. This decoupling is possible because the P-flow and delta rule update embeddings directly from the E-step output, bypassing the loss function entirely.

#### P-Flow: Prediction-Error-Weighted EMA

P-flow updates each token embedding $\mu_v$ toward the beliefs $\mu_{q}$ that the E-step produced for that token, weighted by how well those beliefs predicted the next token. The update runs after the forward pass and is entirely within a `torch.no_grad()` context (`embeddings.py:558–645`, `model.py:1187–1226`).

For each token type $v$ appearing in the batch, the update collects all occurrences of $v$ across batch elements and sequence positions, computes prediction-error-weighted averages of the final beliefs, and applies an exponential moving average. The weighting uses a segment-wise softmax over negative cross-entropy errors, so occurrences where the model predicted well receive higher weight (`embeddings.py:520–556`):

$$w_{v,n} = \frac{\exp(-\ell_n)}{\sum_{m : \mathrm{id}_m = v} \exp(-\ell_m)}$$

where $\ell_n$ is the per-position cross-entropy loss and the sum runs over all positions in the batch where token $v$ appears. The weighted target belief for token $v$ is:

$$\bar{\mu}_v = \sum_{n : \mathrm{id}_n = v} w_{v,n} \, \mu_{q,n}$$

The embedding update is then an EMA step with a confidence-modulated learning rate:

$$\mu_v \;\leftarrow\; (1 - \eta_v) \, \mu_v \;+\; \eta_v \, \bar{\mu}_v$$

where the effective learning rate $\eta_v$ incorporates a confidence factor that scales inversely with the token's mean prediction error:

$$\eta_v = (1 - \rho) \cdot \frac{1}{1 + \bar{\ell}_v}$$

Here $\rho$ is the base EMA decay (default 0.95, so base $\eta = 0.05$) and $\bar{\ell}_v$ is the mean cross-entropy across all occurrences of token $v$ in the batch. Tokens that the model already predicts well receive larger updates; tokens with high error receive smaller, more conservative steps. This prevents the embeddings from being destabilized by poorly predicted occurrences.

The sigma embedding receives a parallel P-flow update with a 10$\times$ slower learning rate ($\eta_\sigma = 0.1 \cdot \eta_v$) for stability. The update targets $\bar{\sigma}_v$ (computed analogously from $\sigma_{q,n}$) and is applied in log-space to maintain positivity (`embeddings.py:626–645`):

$$\log \sigma_v \;\leftarrow\; (1 - \eta_\sigma) \, \log \sigma_v \;+\; \eta_\sigma \, \log \bar{\sigma}_v$$

#### Phi P-Flow: Gauge Frame Learning Without Backprop

In Hebbian mode, `detach_phi=True` disconnects $\phi$ from the computation graph, so no gradient from the loss reaches $\phi_{\mathrm{embed}}$. Instead, the gauge frame embeddings learn via a separate P-flow update that pushes $\phi_v$ toward the VFE-evolved values $\phi_{\mathrm{evolved}}$ from the E-step (`embeddings.py:647–695`, `model.py:1227–1280`).

The E-step evolves $\phi$ through natural gradient descent on the VFE (minimizing transported KL divergences), producing $\phi_{\mathrm{evolved}}$ that reflects the gauge geometry preferred by the current data. After the forward pass, this evolved value is persisted back to the embedding via EMA:

$$\phi_v \;\leftarrow\; (1 - \eta) \, \phi_v \;+\; \eta \, \bar{\phi}_v$$

where $\bar{\phi}_v$ is the prediction-error-weighted average of evolved phi values across all occurrences of token $v$ (using the same segment-wise softmax weights $w_{v,n}$ as the mu P-flow). The learning rate $\eta = 1 - \rho$ uses the base EMA decay without the confidence modulation applied to mu.

This creates a two-timescale system for gauge frames: within each forward pass, the E-step rapidly adapts $\phi$ to the local context (fast timescale), while across training steps, the embedding slowly accumulates the context-averaged preferred frame (slow timescale). The embedding converges toward the frame that, on average across contexts, minimizes the transported KL divergences for that token.

#### Delta Rule: Local Learning for $W_{\mathrm{out}}$

The output projection $W_{\mathrm{out}} \in \mathbb{R}^{V \times K}$ maps belief means to vocabulary logits. In Hebbian mode, instead of receiving gradients through backpropagation from the cross-entropy loss, $W_{\mathrm{out}}$ is updated by the Widrow-Hoff delta rule — a local learning rule that requires only the prediction error and the pre-synaptic activity (`model.py:1282–1343`):

$$\Delta W_{\mathrm{out}} = \eta_\delta \cdot (y - \hat{y}) \otimes \mu_q^\top$$

where $y \in \{0, 1\}^V$ is the one-hot encoded target, $\hat{y} = \mathrm{softmax}(W_{\mathrm{out}} \mu_q) \in [0,1]^V$ is the predicted distribution, $\mu_q \in \mathbb{R}^K$ is the belief mean from the E-step, and $\otimes$ denotes the outer product. The update is averaged over all non-padding positions in the batch:

$$W_{\mathrm{out}} \;\leftarrow\; W_{\mathrm{out}} + \frac{\eta_\delta}{|\mathcal{V}|} \sum_{n \in \mathcal{V}} (y_n - \hat{y}_n) \, \mu_{q,n}^\top$$

where $\mathcal{V}$ is the set of valid (non-padding) positions. This is equivalent to one step of gradient descent on the cross-entropy loss with respect to $W_{\mathrm{out}}$ alone (since $\partial \mathrm{CE} / \partial W_{\mathrm{out}} = (\hat{y} - y) \mu_q^\top$ for the softmax-cross-entropy pair), but computed without backpropagation through the rest of the graph.

#### Fully Backprop-Free Training

When all three local rules are active (`use_p_flow=True`, `use_delta_rule_w_out=True`, `detach_phi=True`), the Hebbian mode achieves fully backprop-free learning. The training loop still computes a forward pass (including the E-step VFE iterations) and a cross-entropy loss for logging, but no `loss.backward()` gradient propagation is needed for parameter updates. All three embedding types ($\mu$, $\sigma$, $\phi$) update via P-flow, and $W_{\mathrm{out}}$ updates via the delta rule.

The backprop learning rates (`mu_lr`, `sigma_lr`, `phi_lr`, `output_lr`) still appear in the config for the optimizer but are less important in practice — P-flow dominates the embedding updates, and the delta rule dominates $W_{\mathrm{out}}$. The VFE E-step internal parameters (learnable step sizes $\eta_\mu$, $\eta_\sigma$, learnable $c_0$/$b_0$ for adaptive alpha) still require gradients through the E-step computation, but these are E-step dynamics parameters, not representation parameters.

#### Hebbian Mode Configuration

| Flag | Default | Role |
|------|---------|------|
| `use_p_flow` | `True` | Enable P-flow EMA for $\mu_{\mathrm{embed}}$ and $\sigma_{\mathrm{embed}}$ |
| `use_delta_rule_w_out` | `True` | Enable Widrow-Hoff delta rule for $W_{\mathrm{out}}$ |
| `detach_phi` | `True` | Detach $\phi$ from backprop; learns via phi P-flow only |
| `p_flow_ema_decay` | `0.95` | EMA decay $\rho$ (higher = slower embedding drift) |
| `delta_rule_lr` | `0.1` | Step size $\eta_\delta$ for delta rule |
| `amortized_inference` | `False` | Priors detached (P-flow replaces gradient-based embedding learning) |
| `alpha`, `beta`, `alpha_phi`, `lambda_hyper` | All `0.0` | No VFE terms in training loss; regularization is implicit in E-step |

### Pure VFE (No Autograd, No Optimizer)

**Config:** `PURE_VFE_CONFIG` | **Mode string:** `'pure_vfe'` | **Model class:** `PureVFETransformer`

The Pure VFE mode is the most minimal realization of the gauge-theoretic free energy principle for sequence modeling. It uses no `nn.Module`, no autograd, no optimizer, and no neural network components whatsoever. The entire system — both inference and learning — operates through analytic natural gradient descent on the gauge-covariant variational free energy with hand-derived, closed-form gradients. All tensors are raw `torch.Tensor` objects manipulated within `torch.no_grad()` contexts.

#### Architecture: The Prior Bank

The "model" is a prior bank: a set of raw tensors indexed by vocabulary token ID (`pure_vfe/model.py:22–108`). Each token $v \in \{1, \ldots, V\}$ is associated with a Gaussian prior $\pi_v = \mathcal{N}(\mu_v, \Sigma_v)$ with $\mu_v \in \mathbb{R}^K$ and $\Sigma_v \in \mathrm{SPD}(K)$, plus gauge frames $\Omega_v \in \mathrm{GL}(K_h)^H$ (one per head). Positional information is encoded by a separate set of positional gauge frames $\Omega_{\mathrm{pos},n}$ that compose multiplicatively with the token frames: $\Omega_n = \Omega_{v_n} \cdot \Omega_{\mathrm{pos},n}$.

There are no linear projections, no learned query/key/value matrices, and no output head. The representational capacity comes entirely from the geometry of the prior bank and the VFE dynamics.

Two gauge parameterizations are supported (`gauge_param`). In the `'omega'` path (default), the gauge frames $\Omega_v \in \mathrm{GL}(K_h)$ are stored directly as matrices. In the `'phi'` path, Lie algebra coordinates $\phi_v \in \mathbb{R}^{K_h^2}$ are stored, with $\Omega_v = \exp(\sum_a \phi_v^a T_a)$ computed via the matrix exponential and $\mathrm{GL}(K_h)$ generators.

#### Inference: E-Step as Forward Pass

The forward pass is VFE descent. Given input token IDs $[v_1, \ldots, v_N]$, beliefs are initialized from the priors: $\mu^{(0)} = \mu_{v_n}$, $\Sigma^{(0)} = \Sigma_{v_n}$. The E-step then runs `n_esteps` iterations (default 12) of natural gradient descent on the belief VFE, which contains only the prior and alignment terms — no observation term (`pure_vfe/inference.py:160–400`):

$$F_E = \sum_i \alpha_i \, \mathrm{KL}(q_i \| p_i) + \sum_{i,j} \beta_{ij} \, \mathrm{KL}(q_i \| \Omega_{ij} q_j)$$

The state-dependent precision $\alpha_i = c_0 / (b_0 + \mathrm{KL}(q_i \| p_i))$ gates prior coupling per position, matching the adaptive alpha in the nn.Module modes.

At each iteration, the E-step computes three natural gradient updates in sequence.

**Mean update.** The gradient $\nabla_\mu F = \nabla_\mu^{\mathrm{prior}} + \nabla_\mu^{\mathrm{align}}$ is computed analytically (alignment term includes the softmax coupling, identical to the EM mode derivation). The natural gradient applies the Fisher-Rao metric for Gaussian means:

$$\Delta \mu = -\eta_E \, \Sigma \, \nabla_\mu F$$

A whitened trust region clips the update: $\|\Delta\mu / \sqrt{\sigma}\|$ is clamped to `trust_region_mu` (default 2.0) before the step is applied. The learning rate decays linearly across iterations from $\eta_E$ to $\eta_E \cdot (1 - \text{decay})$ (`inference.py:241–346`).

**Covariance update.** The covariance gradient per head is:

$$\nabla_{\Sigma_h} F = \frac{1}{2}\!\left[\alpha_i \, \Sigma_{p,h}^{-1} + \sum_j \beta_{ij} \, \Lambda_{ij,h} - (\alpha_i + 1)\,\Sigma_{q,h}^{-1}\right]$$

where $\Lambda_{ij,h} = \Omega_{ij,h}^{-\top} \Sigma_{j,h}^{-1} \Omega_{ij,h}^{-1}$ is the transported precision. The natural gradient on the SPD manifold is $\Delta\Sigma = -2\,\Sigma \, \mathrm{sym}(\nabla_\Sigma F) \, \Sigma$, and the retraction uses the matrix exponential to ensure positive definiteness:

$$\Sigma^{(t+1)} = \Sigma^{1/2} \, \exp\!\left(-\eta_\sigma \, \Sigma^{-1/2} \, \Delta\Sigma \, \Sigma^{-1/2}\right) \, \Sigma^{1/2}$$

with eigenvalues clamped to $[\epsilon_{\min}, \kappa_{\max}]$ for numerical stability (`inference.py:348–389`).

**Gauge frame update.** The gauge gradient $\partial F / \partial \Omega_i$ is derived from the chain rule through transported KL divergences. For each head $h$, the per-pair gradient decomposes as (`pure_vfe/gauge.py:161–240`):

$$\frac{\partial \mathrm{KL}_{ij}}{\partial \Omega_{ij}} = -\Lambda_{ij} \delta_{ij} \mu_j^\top - \Lambda_{ij}(\Sigma_i + \delta_{ij}\delta_{ij}^\top)\Omega_{ij}^{-\top} + \Omega_{ij}^{-\top}$$

where $\delta_{ij} = \mu_i - \Omega_{ij}\mu_j$. The chain rule to the token frame is $\partial \mathrm{KL}_{ij}/\partial \Omega_i = (\partial \mathrm{KL}_{ij}/\partial \Omega_{ij}) \cdot \Omega_j^{-\top}$, and the total gradient sums over attention-weighted pairs: $\nabla_{\Omega_i} F = \sum_j \beta_{ij} \, \partial \mathrm{KL}_{ij}/\partial \Omega_i$.

The natural gradient uses the left-invariant metric on $\mathrm{GL}(K)$. The Euclidean gradient is pulled back to the Lie algebra via $\xi = \Omega^\top \nabla_\Omega F$, clipped, and pushed forward:

$$\Delta\Omega = -\eta_\Omega \, \Omega \, \mathrm{clip}(\xi)$$

This ensures updates respect the group geometry (`pure_vfe/gauge.py:247–280`).

#### Decoding: KL-Based Logits

After the E-step converges, logits are computed without any linear projection. Each token's logit is the negative KL divergence from the converged belief to the corresponding prior (`pure_vfe/gaussians.py:619–670`):

$$\mathrm{logit}_v(i) = \frac{-\mathrm{KL}(q_i \| \pi_v)}{\tau_{\mathrm{decode}}}$$

where the KL is computed over the full Gaussian:

$$\mathrm{KL}(q_i \| \pi_v) = \frac{1}{2}\!\left[\mathrm{tr}(\Sigma_v^{-1} \Sigma_i) + (\mu_v - \mu_i)^\top \Sigma_v^{-1}(\mu_v - \mu_i) - K + \log\frac{|\Sigma_v|}{|\Sigma_i|}\right]$$

Tokens whose priors are close to the converged beliefs receive high logits; distant priors receive low logits. The decode temperature $\tau_{\mathrm{decode}}$ softens the distribution to prevent overconfidence. This replaces the standard $W_{\mathrm{out}} \mu$ linear projection with a fully geometric operation.

#### Learning: M-Step Natural Gradient

The M-step updates the prior bank parameters $\theta = \{\mu_v, \Sigma_v, \Omega_v\}$ using analytic gradients of the marginal VFE, which now includes the observation term. All gradients are derived in closed form — no `loss.backward()` is called anywhere (`pure_vfe/learning.py:221–400`).

**Mean gradient.** The VFE gradient for the prior mean $\mu_v$ aggregates sufficient statistics across all occurrences of token $v$ in the batch. The E-step converged beliefs $\mu^*_n$ provide the data term, and a hyper-prior $\mathcal{N}(0, \sigma_h^2 I)$ provides regularization:

$$\nabla_{\mu_v} F = -\Sigma_v^{-1}(\bar{\mu}^*_v - \mu_v) + \frac{\mu_v}{\sigma_h^2} + \nabla_{\mu_v}^{\mathrm{obs}}$$

where $\bar{\mu}^*_v = (1/n_v)\sum_{n: v_n = v} \mu^*_n$ is the count-weighted average of converged beliefs. The observation gradient $\nabla_{\mu_v}^{\mathrm{obs}}$ enters through the analytic cross-entropy gradient $\partial \mathrm{CE}/\partial \mathrm{logit} = \hat{y} - y$ (the softmax residual), chain-ruled through the KL decode:

$$\nabla_{\mu_v}^{\mathrm{obs}} = \Sigma_v^{-1} \frac{1}{n_v}\sum_n (\hat{y}_{n,v} - y_{n,v})(\mu^*_n - \mu_v)$$

The natural gradient is $\Delta\mu_v = -\eta_M \, \Sigma_v \, \nabla_{\mu_v} F$, with trust-region clipping and optional Adam momentum for variance reduction across batches.

**Covariance gradient.** The prior covariance $\Sigma_v$ gradient uses the converged second moments:

$$\nabla_{\Sigma_v} F = \frac{1}{2}\!\left[\Sigma_v^{-1} - \Sigma_v^{-1}\!\left(\bar{\Sigma}^*_v + \overline{(\mu^* - \mu_v)(\mu^* - \mu_v)^\top}_v\right)\!\Sigma_v^{-1}\right] + \frac{1}{2}\!\left(\frac{I}{\sigma_h^2} - \Sigma_v^{-1}\right)$$

The first bracket is the VFE data term (optimal $\Sigma_v$ would match the empirical second moment of converged beliefs); the second is the hyper-prior regularization. The natural gradient on SPD uses the same $\Delta\Sigma = -2\Sigma \, \mathrm{sym}(\nabla F) \, \Sigma$ formula as the E-step, with matrix exponential retraction.

**Gauge frame gradient.** For the `'omega'` parameterization, the M-step gradient pushes prior frames toward the E-step evolved frames: $\nabla_{\Omega_v} F \propto -(\bar{\Omega}^*_v - \Omega_v)$, with left-invariant natural gradient and trust-region clipping. For the `'phi'` parameterization, the chain rule through $\exp$ is applied: $\nabla_{\phi_v} = \sum_a [\Omega_v^\top \nabla_{\Omega_v} F]_{:,a} \cdot T_a$, with retraction on the Lie algebra (`learning.py:381–399`).

#### Gradient Accumulation

The `MStepAccumulator` class (`learning.py:44–218`) collects per-token sufficient statistics — occurrence counts, converged belief sums, outer product sums, and observation gradient quantities — into vocabulary-sized buffers across multiple micro-batches. After $K$ micro-batches, `apply_m_step_from_accumulated` consumes the accumulated (lower-variance) gradient in a single M-step update. All buffers are indexed by token ID, so different micro-batches with different token sets merge automatically via scatter-add.

#### Pure VFE Configuration

| Flag | Default | Role |
|------|---------|------|
| `n_esteps` | `12` | E-step iterations (replaces network depth) |
| `eta_E` | `0.1` | E-step natural gradient step size |
| `eta_M` | `0.05` | M-step natural gradient step size |
| `gauge_param` | `'omega'` | `'omega'` (direct $\mathrm{GL}(K)$) or `'phi'` (Lie algebra) |
| `causal` | `True` | Autoregressive masking in attention |
| `alpha_b0`, `alpha_c0` | `1.0`, `1.0` | State-dependent precision $\alpha = c_0/(b_0 + \mathrm{KL})$ |
| `hyper_var` | `100.0` | Hyper-prior variance $\sigma_h^2$ (larger = weaker regularization) |
| `use_adam_m_step` | `True` | Adam momentum buffers for M-step variance reduction |
| `use_rope` | `True` | Rotary position embeddings on $\mu$ before KL |
| `grad_accum_steps` | `1` | Micro-batches per M-step update |

Files: `transformer/pure_vfe/model.py`, `inference.py`, `learning.py`, `gaussians.py`, `gauge.py`, `config.py`

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

When `use_deq=True`, the backward pass through the E-step is replaced by implicit differentiation at the fixed point, avoiding the need to backpropagate through all VFE iterations. The forward pass is unchanged — the E-step loop runs for `ffn_n_iterations` steps as usual, producing converged beliefs $z^* = (\mu^*, \Sigma^*)$ (or $z^* = (\mu^*, \Sigma^*, \phi^*)$ when `deq_include_phi=True`). Only the backward pass differs.

#### Fixed-Point Condition and the IFT

At convergence, the E-step satisfies $z^* = g(z^*, \theta)$, where $g$ is one VFE natural gradient step and $\theta$ are model parameters. Differentiating both sides:

$$\frac{dz^*}{d\theta} = J \frac{dz^*}{d\theta} + \frac{\partial g}{\partial \theta}$$

where $J = \partial g / \partial z |_{z^*}$ is the Jacobian of one E-step evaluated at the fixed point. Solving:

$$\frac{dz^*}{d\theta} = (I - J)^{-1} \frac{\partial g}{\partial \theta}$$

The loss gradient with respect to $\theta$ is then $\nabla_\theta \mathcal{L} = \nabla_{z^*} \mathcal{L} \cdot (I - J)^{-1} \cdot \partial g / \partial \theta$, which requires the vector-Jacobian product $v^\top (I - J)^{-1}$ where $v = \nabla_{z^*} \mathcal{L}$.

#### Neumann Series Approximation

Computing $(I - J)^{-1}$ exactly is intractable for the high-dimensional belief state. The implementation approximates it via a truncated Neumann series (`variational_ffn.py:2101–2213`):

$$(I - J^\top)^{-1} v \;\approx\; v + J^\top v + (J^\top)^2 v + \cdots + (J^\top)^K v$$

where $K$ is controlled by `deq_neumann_terms` (default 5). Each term requires one vector-Jacobian product (VJP) through the E-step function $g$, computed via `torch.autograd.grad`. The series converges when the spectral radius $\rho(J) < 1$, which holds at a stable fixed point (the E-step is contractive near convergence).

The implementation uses custom `torch.autograd.Function` classes. `DEQFixedPoint` handles the $(\mu, \Sigma)$ fixed point; `DEQFixedPointFull` extends this to the joint $(\mu, \Sigma, \phi)$ system. In the forward pass, both return their inputs unchanged (identity). In the backward pass, they accumulate the Neumann series by iteratively applying VJPs through the E-step function.

#### Two Variants

**$(\mu, \Sigma)$-only** (`deq_include_phi=False`, default): Only the belief mean and covariance are treated as fixed-point variables. The gauge frame $\phi$ receives a straight-through gradient ($\partial \phi^* / \partial \phi_{\mathrm{init}} \approx I$). This is cheaper (two VJP targets per Neumann term) but leaves the $\phi$ M-step gradient uncorrected.

**Joint $(\mu, \Sigma, \phi)$** (`deq_include_phi=True`): All three E-step variables are included in the fixed-point system. The Neumann series corrects the gradient for all three simultaneously, eliminating the straight-through bias in the $\phi$ M-step gradient. At the joint fixed point, $\partial F/\partial \mu = 0$, $\partial F/\partial \Sigma = 0$, and $\partial F/\partial \phi = 0$, so the IFT applies to the full system. This adds one VJP target per Neumann term but provides the exact M-step gradient for $\phi$ as well.

#### Relationship to Implicit EM

DEQ and implicit EM address the same problem — correcting the M-step gradient for the E-step optimization — but with different approaches. Implicit EM computes a closed-form per-dimension scale factor $s_k$ from the diagonal structure of the Gaussian VFE Hessian, which is cheap but limited to the diagonal approximation. DEQ uses the full (non-diagonal) Jacobian via the Neumann series, which captures cross-dimensional interactions but requires $K$ additional VJP passes. In practice, implicit EM is preferred for the mean and covariance (where the diagonal approximation is accurate), while DEQ is most useful for correcting the $\phi$ gradient (where no diagonal closed form exists).

| Flag | Default | Role |
|------|---------|------|
| `use_deq` | `False` | Enable DEQ implicit differentiation for E-step backward |
| `deq_neumann_terms` | `5` | Neumann series truncation order $K$ |
| `deq_include_phi` | `False` | Include $\phi$ in the joint fixed-point system |

### Implicit EM vs. Amortized Inference

The `implicit_em` and `amortized_inference` flags jointly determine how M-step gradients reach the embedding parameters $\theta = (\mu_{\mathrm{embed}}, \sigma_{\mathrm{embed}}, \phi_{\mathrm{embed}})$. The E-step evolves beliefs $q = (\mu_q, \Sigma_q)$ by minimizing the VFE with priors held fixed, producing a final belief state $q^*$. The question is: when cross-entropy loss $\mathcal{L}(\mu_{q^*})$ is computed on these evolved beliefs and backpropagated, how does the gradient reach $\theta$?

Three regimes exist, each with different gradient paths and different theoretical justifications.

#### Amortized Inference (`amortized_inference=True`, `implicit_em=False`)

In the amortized path, the prior mean $\mu_p$ is **not** detached at E-step entry. Since $\mu_p$ is a view of $\mu_{\mathrm{embed}}$, the full computation graph from embeddings through VFE iterations to the final loss is preserved. The gradient $d\mathcal{L}/d\theta$ flows by standard backpropagation through the E-step dynamics — this is a "straight-through" estimator where $s = 1$ (all gradient passes unscaled).

Concretely, the E-step initializes beliefs at the prior: $\mu_q^{(0)} = \mu_p$. Each VFE iteration updates beliefs via the natural gradient:

$$\mu_q^{(t+1)} = \mu_q^{(t)} - \eta_\mu \, \Sigma_q^{(t)} \, \nabla_{\mu} F\big|_{q^{(t)}}$$

Because $\mu_q^{(0)} = \mu_p$ retains its gradient connection to $\mu_{\mathrm{embed}}$, backpropagation through this chain yields:

$$\frac{d\mathcal{L}}{d\mu_{\mathrm{embed}}} = \frac{\partial \mathcal{L}}{\partial \mu_{q^*}} \cdot \prod_{t=0}^{T-1} \left(I - \eta_\mu \, \Sigma_q^{(t)} \, \frac{\partial^2 F}{\partial \mu_q^2}\bigg|_{q^{(t)}}\right)$$

where the product of Jacobians $(I - \eta \, H_t)$ propagates through each VFE iteration. This is the standard unrolled differentiation approach. Its advantage is simplicity: no custom autograd functions are needed, and the gradient is exact for the finite number of iterations taken. The well-conditioned self-coupling gradient $\partial(\alpha \, \mathrm{KL})/\partial \mu_p = -\alpha(\mu_q - \mu_p)/\sigma_p$ provides a stable learning signal that pushes embeddings toward successful belief states (`variational_ffn.py:3435–3442`).

The disadvantage is that the gradient magnitude depends on the number of E-step iterations and can either vanish (if $\eta H \approx 1$, the Jacobian factors approach zero) or explode (if $\eta H > 1$). For a single iteration ($T = 1$), the Jacobian is simply $(I - \eta H_0)$, which is well-behaved. For deeper E-steps, gradient pathology becomes more likely.

The prior covariance $\sigma_p$ is always detached regardless of amortization mode. Leaving $\sigma_p$ live creates a positive feedback loop: the E-step gradient $\partial \mathrm{KL}/\partial \sigma_q \propto 1/\sigma_p$, so smaller $\sigma_p$ produces larger gradients that push $\sigma_p$ even smaller. The M-step loss $\lambda_h \, \mathrm{KL}(s \| h)$ provides the correct, bounded gradient path for $\sigma_{\mathrm{embed}}$ learning (`variational_ffn.py:3448–3456`).

#### Implicit EM (`implicit_em=True`)

In the implicit EM path, **both** $\mu_p$ and $\sigma_p$ are detached at E-step entry (`variational_ffn.py:3482–3484`). The E-step runs with no gradient connection to embeddings — a clean separation between inference and learning. After the E-step completes, the gradient path is re-established through the `ImplicitEMGradient` custom autograd function, which applies the IFT-derived scale factor (`model.py:1024–1029`):

$$\frac{d\mathcal{L}}{d\mu_{\mathrm{embed}}} = s_k^{(\mu)} \cdot \frac{\partial \mathcal{L}}{\partial \mu_{q^*}}$$

where the per-dimension scale factor is:

$$s_k^{(\mu)} = \frac{\alpha / \sigma_{p,k}^2}{\alpha / \sigma_{p,k}^2 + \sum_j \beta_{ij} / \sigma_{j,k}^2}$$

This replaces the product-of-Jacobians chain in the amortized path with a single multiplicative correction derived from the structure of the fixed-point equation. The IFT guarantees that this is the correct total derivative $dq^*/d\theta$ at convergence, and provides a principled interpolation for finite iterations.

When `implicit_em=True`, the `amortized_inference` flag is forced to have no effect on $\mu_p$ — even if set to `True`, the prior is detached because the IFT scale is the sole intended gradient path. Keeping $\mu_p$ live would double-count: embeddings would receive both the IFT-scaled gradient and the straight-through gradient through the self-coupling term.

An analogous scale factor applies to the covariance path via `ImplicitEMGradientSigma`:

$$s_k^{(\sigma)} = \frac{\alpha / \sigma_{p,k}^4}{\alpha / \sigma_{p,k}^4 + \sum_j \beta_{ij} / \sigma_{j,k}^4}$$

Both scale factors are computed from quantities available at the end of the E-step (final $\alpha_i$, $\sigma_p$, $\beta$, $\sigma_q$) and are detached from the computation graph — they modulate the gradient magnitude but do not themselves require gradients.

#### Detached E-Step (`amortized_inference=False`, `implicit_em=False`)

When both flags are `False`, $\mu_p$ is detached and no IFT re-attachment occurs. The E-step runs in isolation, and no gradient from $\mathcal{L}$ reaches the embedding means through the belief evolution path. Embeddings learn only through explicit regularization terms in the M-step loss (e.g., $\alpha \, \mathrm{KL}(q^* \| p)$ or $\lambda_h \, \mathrm{KL}(s \| h)$) and through direct gradient to $W_{\mathrm{out}}$.

This is the "pure EM" limit ($s = 0$) — beliefs are treated as latent variables inferred independently of the parameters that generated them. It is the most theoretically clean separation but provides the weakest learning signal to embeddings, since the cross-entropy gradient must pass through the explicit KL terms rather than flowing directly from prediction error.

#### Phi Gradient Paths

The gauge frame embeddings $\phi_{\mathrm{embed}}$ follow a parallel logic. When `detach_phi=True` (used in Hebbian mode), $\phi$ is detached from the computation graph entirely, enabling fully backprop-free training where $\phi_{\mathrm{embed}}$ learns via P-flow EMA instead of gradient descent (`variational_ffn.py:3505–3510`). When `detach_phi=False` (default in EM), $\phi$ retains its gradient connection through the E-step, and the DEQ option (`deq_include_phi=True`) can further correct the $\phi$ M-step gradient via the Neumann-series IFT.

#### Summary

| Setting | $\mu_p$ at E-step entry | Gradient path to $\mu_{\mathrm{embed}}$ | Scale factor |
|---------|------------------------|----------------------------------------|--------------|
| `amortized_inference=True` | Live (gradient retained) | Backprop through E-step Jacobian chain | $s = 1$ (straight-through) |
| `implicit_em=True` | Detached | IFT re-attachment after E-step | $s_k \in [0, 1]$ from fixed-point Hessian |
| Both `False` | Detached | Only via explicit M-step KL terms | $s = 0$ (pure EM) |

The implicit EM path is recommended for training because it provides the information-geometrically correct gradient without the vanishing/exploding gradient risks of unrolled differentiation, and without the weak learning signal of pure EM. The amortized path is useful for shallow E-steps ($T = 1$) where the Jacobian chain is short and well-conditioned.

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
