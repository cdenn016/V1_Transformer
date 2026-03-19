# Pure VFE Transformer: Implementation Plan

## The Idea

Strip away **all neural network machinery** — no `nn.Module`, no `autograd`, no `loss.backward()`, no Adam/SGD — and build a sequence model that operates **entirely** through variational free energy minimization via natural gradient descent on a statistical manifold.

The "model" is just a collection of Gaussian priors (one per vocabulary token). "Inference" IS the forward pass: running VFE descent IS how input is processed. "Learning" IS Bayesian model update: updating priors from posteriors via the model channel.

This implements the full hierarchical VFE from the manuscript (Eq. 7 of main paper + model-channel Eq. G.1 of supplementary) over a **single 0D base manifold** — the exact setting the paper describes for transformers.

---

## Mathematical Foundation

### The Full Gauge VFE (from GL(K) manuscript Eq. 7 + Supplementary G.1)

```
F_full[{q_i}, {s_v}] =

  Σ_i  α_i · KL(q_i || p_i)                           # (1) Belief-prior: regularize beliefs toward priors
+ Σ_i  KL(s_{v(i)} || r)                               # (2) Model-prior: regularize priors toward hyper-prior
+ Σ_{ij} β*_ij · KL(q_i || Ω_ij · q_j)                # (3) Belief alignment: inter-agent consensus
+ Σ_{ij} γ_ij · KL(s_{v(i)} || Ω̃_ij · s_{v(j)})       # (4) Model alignment: meta-cognitive consensus
- Σ_i  E_{q_i}[log p(o_i | k_i)]                       # (5) Observation likelihood
```

where:
- `q_i = N(μ_i, Σ_i)` — agent i's belief (inferred per-forward-pass)
- `p_i = N(μ_{v(i)}, Σ_{v(i)})` — prior from token v(i)'s entry in the prior bank
- `s_v = N(μ_v, Σ_v)` — the prior bank entry itself (learned via M-step)
- `r = N(0, σ²_h I)` — hyper-prior (weight decay equivalent)
- `Ω_ij = exp(φ_i) · exp(-φ_j)` — GL(K) parallel transport
- `β*_ij = softmax_j(-KL(q_i || Ω_ij · q_j) / τ)` — attention (already optimized)
- `α_i = c₀ / (b₀ + KL(q_i || p_i))` — state-dependent prior precision

### Timescale Separation (Two Nested Loops)

**E-step (fast, per-sequence):** Minimize F w.r.t. beliefs `{μ_i, Σ_i, φ_i}` holding priors fixed.
- This IS the "forward pass" — iterative VFE descent converges beliefs.
- Number of E-steps replaces "number of layers" in standard transformers.

**M-step (slow, per-batch):** Update priors `{μ_v, Σ_v}` via natural gradient on F.
- This IS "learning" — priors accumulate knowledge from observed posteriors.
- No backpropagation needed: gradients of KL between Gaussians are analytic.

### All Gradients Are Analytic (from manuscript Appendix B, C, D)

**Mean gradient (Eq. 21 of main paper):**
```
∂F/∂μ_i = α_i · Σ_{p,i}^{-1}(μ_i - μ_p)
         + Σ_j [ β_ij · (Ω_ij Σ_j Ω_ij^T)^{-1} (μ_i - Ω_ij μ_j)
                + (∂β_ij/∂μ_i) · KL(q_i || Ω_ij q_j) ]
         - ∂log p(o_i|μ_i)/∂μ_i

Natural gradient: Δμ_i = -η · Σ_i · ∂F/∂μ_i
```

**Covariance gradient (Eq. B.7 of supplementary):**
```
∂F/∂Σ_i = (1/2)[-2Σ_i^{-1} + Σ_{p,i}^{-1} + Σ_j β_ij (Ω_ij Σ_j Ω_ij^T)^{-1}]
         + Σ_j (∂β_ij/∂Σ_i) · KL(q_i || Ω_ij q_j)

Natural gradient: ΔΣ = -2 Σ · sym(∂F/∂Σ) · Σ
Retraction: affine-invariant exponential map on SPD manifold
```

**Gauge frame gradient (Eq. C.1 of supplementary):**
```
∂F/∂φ_i = Σ_j [β_ij · ∂KL_ij/∂φ_i + (∂β_ij/∂φ_i) · KL_ij]
         + Σ_k [β_ki · ∂KL_ki/∂φ_i + ...]

Natural gradient options:
  - Killing form: g̃^{-1} · ∂F/∂φ  (parameter-free)
  - Pullback:     G(φ)^{-1} · ∂F/∂φ  (position-dependent, exact)
```

**Prior (M-step) gradient:**
```
∂F/∂μ_v = -α · Σ_v^{-1} · Σ_{i: v(i)=v} (μ*_i - μ_v) + (μ_v - μ_h)/σ²_h

Natural gradient: Δμ_v = -η_M · Σ_v · ∂F/∂μ_v
```

**Softmax nonlinearity (∂β/∂μ):**
```
∂β_ij/∂μ_i = -(β_ij/τ) · [∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i]
```

### Observation Likelihood (Decoding)

For language modeling, the observation model uses KL-based decoding:
```
logit_v = -KL(q_N || π_v)     for each vocabulary token v

where π_v = N(μ_v, Σ_v) is the prior for token v
```

This replaces the standard linear output projection with a principled information-geometric decoder. The cross-entropy loss `CE(softmax(logits), target)` provides the observation gradient `∂log p(o|μ)/∂μ`.

### Positional Structure on 0D Base

With a 0D base manifold, position must be encoded somewhere:
- **Positional gauge offsets:** `φ_i = φ_{v(i)} + δ_pos(i)` where `δ_pos` adds a learnable (via M-step) positional component to gauge frames
- **Causal structure:** attention prior `π_j ∝ 1[j ≤ i]` restricts which agents can attend to which (causal masking from the VFE)

### Multi-Head Structure

Emerges from **block-diagonal covariances and gauge frames** (as in the existing GL(K) implementation):
- Σ_i = diag(Σ_i^(1), ..., Σ_i^(H)) with H blocks of size K/H
- Each block is an independent "head" with its own temperature τ_h
- Gauge frames are block-diagonal: φ_i = diag(φ_i^(1), ..., φ_i^(H))

---

## Architecture: `PureVFETransformer`

### Data Structures (No nn.Parameters)

```python
class PureVFETransformer:
    # Prior bank: one Gaussian per vocabulary token (THE model)
    prior_mu:    Tensor[V, K]       # Prior means
    prior_L:     Tensor[V, K]       # Prior Cholesky diag (diagonal approx)
    prior_phi:   Tensor[V, n_gen]   # Prior gauge frames

    # Positional gauge offsets
    pos_phi:     Tensor[max_len, n_gen]  # Position-dependent gauge offset

    # Hyper-prior
    hyper_mu:    Tensor[K]          # Hyper-prior mean (zero)
    hyper_var:   float              # Hyper-prior variance

    # Lie algebra generators
    generators:  Tensor[n_gen, K_h, K_h]  # GL(K_h) generators per head

    # Hyperparameters
    K: int          # Belief dimension
    H: int          # Number of heads
    n_esteps: int   # Number of E-step iterations (replaces depth)
    tau: float      # Attention temperature
    eta_E: float    # E-step learning rate
    eta_M: float    # M-step learning rate
    alpha_b0: float # Prior precision regularizer
    alpha_c0: float # Prior precision regularizer
```

### E-Step: Inference (Forward Pass)

```python
def inference(self, token_ids: Tensor[B, N]) -> Tensor[B, N, V]:
    """
    Process a sequence by running VFE descent. Returns logits.
    This IS the forward pass — no neural network involved.
    """
    B, N = token_ids.shape

    # 1. Initialize beliefs from priors
    mu    = self.prior_mu[token_ids]          # [B, N, K]
    L     = self.prior_L[token_ids]           # [B, N, K] (diagonal)
    phi   = self.prior_phi[token_ids]         # [B, N, n_gen]
    phi   = phi + self.pos_phi[:N]            # Add positional offset

    # 2. E-step: iterative VFE descent
    for step in range(self.n_esteps):
        # a. Compute transport operators per head
        Omega_ij = compute_transport(phi, self.generators)  # [B, H, N, N, K_h, K_h]

        # b. Compute KL divergences (analytic for Gaussians)
        kl_ij = compute_kl_transported(mu, L, Omega_ij)     # [B, H, N, N]

        # c. Compute attention weights (with causal mask)
        beta_ij = masked_softmax(-kl_ij / self.tau, causal=True)  # [B, H, N, N]

        # d. Compute state-dependent prior precision
        kl_prior = compute_kl_prior(mu, L, token_ids)       # [B, N]
        alpha = self.alpha_c0 / (self.alpha_b0 + kl_prior)  # [B, N]

        # e. Compute VFE gradients (ALL ANALYTIC)
        grad_mu    = compute_mu_gradient(mu, L, phi, beta_ij, alpha, token_ids)
        grad_L     = compute_L_gradient(mu, L, phi, beta_ij, alpha, token_ids)
        grad_phi   = compute_phi_gradient(mu, L, phi, beta_ij, self.generators)

        # f. Natural gradient preconditioning
        nat_grad_mu  = diag_sigma(L) * grad_mu            # Fisher for mean = Σ
        nat_grad_L   = spd_natural_gradient(L, grad_L)    # SPD manifold metric
        nat_grad_phi = killing_natural_gradient(grad_phi)  # Killing form

        # g. Update beliefs
        mu  = mu  - self.eta_E * nat_grad_mu
        L   = retract_spd(L, nat_grad_L, self.eta_E)      # SPD retraction
        phi = phi - self.eta_E * nat_grad_phi

    # 3. Decode: KL to each prior gives logits
    logits = compute_kl_to_priors(mu, L, self.prior_mu, self.prior_L)  # [B, N, V]
    return -logits  # Negative KL = log-likelihood proxy
```

### M-Step: Learning (Parameter Update)

```python
def learn(self, token_ids: Tensor[B, N], targets: Tensor[B, N]):
    """
    Update priors (the model) via natural gradient on VFE.
    This IS learning — no backpropagation involved.
    """
    # 1. Run inference to get converged beliefs
    mu_star, L_star, phi_star = self.inference_with_beliefs(token_ids)

    # 2. Compute observation gradient (from CE loss)
    logits = compute_kl_to_priors(mu_star, L_star, self.prior_mu, self.prior_L)
    obs_grad = compute_ce_gradient(logits, targets)  # Analytic softmax-CE gradient

    # 3. Compute prior gradients (M-step, analytic)
    for v in unique(token_ids):
        mask = (token_ids == v)
        n_v = mask.sum()

        # Gradient of KL(q*_i || p_v) w.r.t. prior mean
        mu_diff = mu_star[mask] - self.prior_mu[v]      # [n_v, K]
        Sigma_v_inv = 1.0 / (self.prior_L[v] ** 2)      # [K] (diagonal)
        grad_mu_v = -Sigma_v_inv * mu_diff.mean(0)       # Average over occurrences

        # Hyper-prior pull
        grad_mu_v += (self.prior_mu[v] - self.hyper_mu) / self.hyper_var

        # Gradient of KL w.r.t. prior variance (diagonal)
        grad_L_v = compute_prior_L_gradient(L_star[mask], self.prior_L[v])

        # Natural gradient for prior mean: Δμ = -η · Σ_v · ∂F/∂μ_v
        Sigma_v = self.prior_L[v] ** 2
        nat_grad_mu_v = Sigma_v * grad_mu_v

        # Update prior
        self.prior_mu[v] -= self.eta_M * nat_grad_mu_v
        self.prior_L[v]   = retract_spd_diag(self.prior_L[v], grad_L_v, self.eta_M)

    # 4. Update positional gauge frames
    grad_pos_phi = compute_pos_phi_gradient(phi_star, token_ids)
    self.pos_phi -= self.eta_M * grad_pos_phi

    # 5. Update prior gauge frames
    grad_prior_phi = compute_prior_phi_gradient(phi_star, token_ids)
    self.prior_phi -= self.eta_M * grad_prior_phi
```

---

## File Structure

```
transformer/pure_vfe/
├── __init__.py
├── model.py              # PureVFETransformer: the main model class
├── inference.py           # E-step: VFE descent (the forward pass)
├── learning.py            # M-step: prior updates (the backward pass)
├── gaussians.py           # Analytic KL, gradients, Fisher metrics for Gaussians
├── gauge.py               # GL(K) transport, Killing form, gauge frame operations
├── observation.py         # KL-based decoding, CE gradient computation
├── manifold.py            # SPD retraction, Lie algebra retraction, natural gradient
├── config.py              # PureVFEConfig dataclass
├── train.py               # Training loop (no optimizer — just M-step calls)
└── tests/
    ├── test_gradients.py  # Finite-difference validation of all analytic gradients
    ├── test_invariance.py # GL(K) gauge invariance checks
    └── test_convergence.py # E-step convergence diagnostics
```

---

## Implementation Steps

### Phase 1: Mathematical Core (`gaussians.py`, `gauge.py`, `manifold.py`)

1. **`gaussians.py`** — All Gaussian operations with analytic gradients:
   - `kl_divergence(mu1, L1, mu2, L2)` → KL + all partial derivatives
   - `kl_transported(mu_i, L_i, mu_j, L_j, Omega)` → KL through gauge transport
   - `kl_gradient_mu(...)` → ∂KL/∂μ_i (Eq. 21 of main paper)
   - `kl_gradient_sigma(...)` → ∂KL/∂Σ_i (Eq. B.7 of supplementary)
   - `softmax_gradient_mu(beta, kl, ...)` → ∂β/∂μ (Eq. 24 of main paper)
   - Diagonal covariance specialization for memory efficiency

2. **`gauge.py`** — Gauge frame operations:
   - `compute_transport(phi_i, phi_j, generators)` → Ω_ij = exp(φ_i)exp(-φ_j)
   - `kl_gradient_phi(...)` → ∂KL/∂φ (Eq. C.1 of supplementary)
   - `build_generators(K, H)` → block-diagonal GL(K/H) generators
   - `killing_form_preconditioner(generators)` → g̃^{-1}
   - Reuse existing `math_utils/generators.py` and `gauge_preconditioner.py`

3. **`manifold.py`** — Natural gradient + retraction:
   - `natural_gradient_mean(grad_mu, Sigma)` → Σ · ∇μ
   - `natural_gradient_sigma(grad_sigma, Sigma)` → SPD natural gradient
   - `retract_spd(L, delta, step)` → affine-invariant exponential map
   - `natural_gradient_phi(grad_phi, inv_metric)` → Killing form natural gradient

### Phase 2: Core Model (`model.py`, `inference.py`, `learning.py`, `observation.py`)

4. **`model.py`** — PureVFETransformer class:
   - Initialize prior bank from random Gaussians (or pretrained embeddings)
   - Initialize positional gauge offsets
   - No `nn.Module`, no `nn.Parameter` — just raw tensors
   - `forward()` → calls inference, returns logits
   - `update()` → calls learning M-step
   - Device management, serialization

5. **`inference.py`** — The E-step (forward pass):
   - Initialize beliefs from priors
   - Iterate VFE descent for `n_esteps` iterations
   - Compute transport, attention, gradients at each step
   - Natural gradient + retraction updates
   - Convergence monitoring (optional early stopping)
   - Return converged beliefs + logits

6. **`learning.py`** — The M-step (parameter update):
   - Accumulate posterior statistics from converged beliefs
   - Compute analytic gradients w.r.t. prior parameters
   - Natural gradient updates for prior means, covariances, gauge frames
   - Hyper-prior regularization (weight decay equivalent)
   - Optional: update positional gauge offsets

7. **`observation.py`** — Observation model:
   - `kl_to_priors(mu, L, prior_bank)` → logits via -KL(q || π_v) for each v
   - `ce_gradient(logits, targets)` → analytic softmax cross-entropy gradient
   - This is the only place where discrete tokens interface with continuous beliefs

### Phase 3: Training Loop + Validation (`train.py`, `config.py`, `tests/`)

8. **`config.py`** — Configuration:
   - `PureVFEConfig` dataclass with all hyperparameters
   - K, H, n_esteps, tau, eta_E, eta_M, alpha_b0, alpha_c0
   - Gauge group choice, covariance mode (diagonal/full)

9. **`train.py`** — Training loop:
   - Load data (WikiText-2 for initial experiments)
   - For each batch: `logits = model.forward(tokens)` → `model.update(tokens, targets)`
   - Log perplexity, VFE components, convergence metrics
   - No optimizer object — learning IS the M-step

10. **`tests/`** — Validation:
    - Finite-difference checks for ALL analytic gradients
    - GL(K) invariance: KL(G·P || G·Ω·Q) = KL(P || Ω·Q)
    - E-step convergence: VFE monotonically decreasing
    - Gradient sanity: natural gradient direction reduces VFE

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Covariance | Diagonal | Full is O(K³) per token; diagonal is O(K). Start diagonal, extend later. |
| Gauge group | GL(K/H) block-diagonal | Matches multi-head structure; proven in existing implementation |
| Phi preconditioning | Killing form | Parameter-free, proven effective in `gauge_preconditioner.py` |
| E-step count | 8-16 | Replaces depth; manuscript shows convergence in ~10 iterations |
| Temperature | τ = √(K/H) | Dimensional scaling from manuscript Sec. 4.1 |
| Prior precision | State-dependent α_i | Eq. 16 of main paper; allows adaptive regularization |
| Observation model | KL decoding | -KL(q_N || π_v) as logits; principled, already in prior_bank.py |
| Positional encoding | Gauge frame offsets | φ_i = φ_{token} + δ_pos(i); position in Lie algebra |
| Causal masking | Attention prior π_j | Set π_j = 0 for j > i (Eq. 11 of main paper) |

## What This Proves

If this works (achieves non-trivial perplexity on language modeling):

1. **Attention IS variational inference** — not just an analogy, but literally
2. **Backpropagation is unnecessary** — natural gradient on VFE suffices
3. **Neural networks are unnecessary** — the computational substrate is pure statistics
4. **The FEP can do language modeling** — Friston's principle generates a working transformer
5. **The gauge-theoretic framework is complete** — every component (attention, layers, learning, normalization) emerges from one equation

## Comparison to Existing Implementation

| Aspect | Current GL(K) Transformer | Pure VFE Transformer |
|--------|--------------------------|---------------------|
| Forward pass | E-step VFE + nn.Module overhead | Pure VFE descent only |
| Backward pass | autograd + Adam | Analytic M-step natural gradient |
| Embeddings | nn.Embedding | Raw prior bank tensors |
| Output projection | nn.Linear or KL decode | KL decode only |
| Optimizer | AdamW with 6 param groups | None — M-step IS optimization |
| Gradient computation | torch.autograd | Closed-form analytic |
| Training loop | loss.backward() + optimizer.step() | model.forward() + model.update() |
