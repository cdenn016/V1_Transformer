# Pure VFE Transformer: Architecture Review and Gap Analysis

Assessment of the pure variational free energy transformer for language modeling. No `nn.Module`, no autograd — all gradients analytical, all updates via natural gradient on the product manifold of Gaussian beliefs and GL(K) gauge frames.

---

## 1. Architecture

The model is a **prior bank**: raw tensors `(μ_v, Σ_v, Ω_v)` per vocabulary token.

```
prior_mu:    [V, K]              Per-token belief means
prior_Sigma: [V, K, K]           Per-token SPD covariances (full, not diagonal)
prior_Omega: [V, H, K_h, K_h]   Per-token GL(K_h) gauge frames per head
```

No embedding layers. No attention weights. No layer norm (optional toggle). No output projection. The model IS a probability distribution bank. Learning updates these directly via natural gradient.

### Forward Pass = E-Step (12 iterations of VFE descent)

```
Input: token_ids [B, N]

Initialize beliefs from priors:
  μ ← prior_μ[token_ids]     [B, N, K]
  Σ ← prior_Σ[token_ids]     [B, N, K, K]
  Ω ← prior_Ω[token_ids]     [B, N, H, K_h, K_h]

FOR step = 1 ... n_esteps (default 12):
  1. Precompute pairwise terms: ρ_i = Ω_i⁻¹ μ_i, P_i = Ω_iᵀ Σ_i⁻¹ Ω_i, etc.
  2. Optional RoPE: rotate μ via SO(2)^{K/2} before KL scoring
  3. Pairwise KL: KL(q_i || Ω_ij[q_j])  [B, H, N, N]  (causal masked)
  4. Attention: β_ij = softmax(-KL_ij / τ)
  5. Adaptive precision: α_i = c₀/(b₀ + KL(q_i || p_i))
  6. Mean gradient: ∇μ = alignment + prior + softmax correction (Eq. 21+24)
     Natural gradient: nat_μ = Σ · ∇μ  (Fisher-Rao)
     Update: μ ← μ - η_μ · nat_μ  (whitened trust region)
  7. Covariance gradient: ∇Σ = ½[α Σ_p⁻¹ + Σ_j β_ij(Ω⁻ᵀ P_j Ω⁻¹) - (α+1)Σ⁻¹]
     Natural gradient: nat_Σ = -2Σ sym(∇Σ) Σ  (SPD metric)
     Retract: Σ ← exp_map(Σ, η_Σ · nat_Σ)  (affine-invariant)
  8. Gauge gradient: ∂KL/∂Ω per pair, aggregated
     Lie algebra clip + trust region
     Update: Ω ← Ω - η_φ · Ω · ξ  (GL(K) retraction)
  9. NaN recovery if needed; early stop if VFE diverges

Decode: logits_v = -KL(q_i* || π_v) / τ_decode  [B, N, V]
```

### M-Step (prior bank update, analytical)

```
1. Accumulate sufficient statistics per vocab token:
   - n_v: count of token v in batch
   - μ*_avg: mean of converged beliefs for token v
   - Σ*_avg: mean of converged covariances
   - Ω*_avg: mean of converged gauge frames
   - obs quantities: CE gradient contributions

2. Prior μ_v update:
   grad = -Σ_v⁻¹(μ*_avg - μ_v) + μ_v/σ²_hyper + obs_grad
   nat_μ = Σ_v · grad  (Fisher-Rao)
   μ_v ← μ_v - η · nat_μ  (with Adam momentum, trust region, max-norm)

3. Prior Σ_v update:
   grad = ½[Σ_v⁻¹ - Σ_v⁻¹(Σ* + outer)Σ_v⁻¹] + ½[I/σ²_hyper - Σ_v⁻¹]
   nat_Σ = -2Σ sym(grad) Σ  (SPD natural gradient)
   Σ_v ← retract_spd(Σ_v, nat_Σ, η)  (eigenvalue floor enforced)

4. Prior Ω_v update:
   grad = -(Ω*_avg - Ω_v)
   Ω_v ← Ω_v - η · lie_algebra_clip(grad)  (conditioning regularized)
```

### KL-Decode (output projection)

No linear layer. Logits are negative KL divergences from converged beliefs to all V prior distributions:

```
logits[b, n, v] = -KL(q*_{b,n} || π_v) / τ_decode
```

Chunked over vocabulary (V ~ 50k) for memory. Fused matmul available for diagonal covariance:
```
lhs = [σ_q + μ_q², -2μ_q]   (B, N, 2K)
rhs = [1/σ_p, μ_p/σ_p]      (V, 2K)
logits ≈ -0.5 · lhs @ rhs.T / τ
```

---

## 2. What Works

- E-step gradients (μ, Σ, Ω) validated against finite differences (REL_TOL=1e-3, K=4)
- GL(K) invariance: KL(G·P || G·Ω·Q) = KL(P || Ω·Q) verified
- SPD retraction preserves positive-definiteness under test
- Causal masking: upper triangle masked with +inf before softmax
- RoPE: SO(2)^{K/2} rotations on μ before KL scoring
- Full covariance supported (not just diagonal)
- Adam momentum on M-step for variance reduction
- NaN recovery: beliefs reset to priors on numerical instability
- VFE divergence early stopping (halt if VFE increases >10%)
- Gradient accumulation across micro-batches via MStepAccumulator
- Cocycle condition Ω_ij Ω_jk = Ω_ik verified in tests
- Conditioning regularization: Ω pushed toward polar factor when κ > cond_max

---

## 3. Bugs and Concerns

### 3.1 Softmax Correction Weights Can Go Negative

```python
# gaussians.py, softmax-corrected alignment weights
correction = ((e_kl - kl_ij) / tau).clamp(-1.0, 2.0)
w = beta * (1.0 + correction)
```

The clamp prevents `w < 0`, but `w = 0` when `correction = -1` means high-KL pairs contribute zero gradient. Information from the most dissimilar (potentially most informative) neighbors is discarded. Consider a softer floor or a different parameterization of the correction.

### 3.2 M-Step Observation Gradients Are Untested

E-step gradients have finite-difference validation. M-step observation gradients (CE loss propagated to prior μ and prior Σ) are analytical but never numerically verified. The full Σ observation path (`sigma_obs_grad='full'`) involves:

```
Σ_v⁻¹ @ W @ Σ_v⁻¹
```

Four nested matrix multiplications with no intermediate clamping. If `W` contains NaN from bad observations, the product cascades. The diagonal mode (`sigma_obs_grad='diagonal'`) is safer but approximate.

**Recommendation:** Add finite-difference tests for `∂CE/∂μ_v` and `∂CE/∂Σ_v` in `test_gradients.py`.

### 3.3 RoPE Applied to Attention but Not to Gradients

β comes from rotated-μ KL, but the alignment gradient uses unrotated-μ KL. The code comments state this is intentional for "geometric consistency" (the chain rule ∂(β·KL)/∂μ should use the same coordinate space for both KL and ∂KL/∂μ). This is correct in principle, but the mismatch means the gradient direction does not exactly correspond to the steepest descent of the rotated-attention VFE. May cause slow convergence or mild oscillation.

### 3.4 Gauge Frames Update Every E-Step Iteration

The nn.Module version updates Ω once during an initial calibration loop, then freezes during belief iterations. The pure VFE updates Ω at every iteration, which is theoretically more principled (joint optimization over all variables) but:
- Computationally expensive: O(N² K²) gradient per iteration per head
- Potentially destabilizing: Ω and μ can chase each other
- The nn.Module version found freezing Ω during E-step to be sufficient

### 3.5 Prior Mean Max-Norm Is Very Loose

```python
prior_mu_max_norm: float = 10.0  # was 3.0, "too tight"
```

Initialization spreads μ_v at ~√(ln V / K) ≈ 0.58 for V=50k, K=32. The max-norm of 10.0 is 17x the initialization scale. It is not constraining anything in practice. Either tighten it or remove it.

### 3.6 Gauge Conditioning Blend Is Weak

```python
if cond(Ω) > cond_max:
    blend = 0.1 * (cond / cond_max - 1)
    Ω ← (1 - blend) * Ω + blend * polar_factor(Ω)
```

If κ = 100 and cond_max = 50, blend = 0.1. The new Ω is 90% the ill-conditioned original. Multiple steps may be needed to bring conditioning below threshold. Consider a stronger correction or direct singular value clamping.

---

## 4. Missing Features (vs nn.Module Version)

These features exist in `transformer/core/variational_ffn.py` but not in the pure VFE. They likely explain the performance gap.

### 4.1 Learnable Adaptive Precision α (HIGH IMPACT)

The nn.Module version implements state-dependent prior coupling:

```
α_k = softplus(c₀) / (softplus(b₀) + KL_k(q || p))
```

where `c₀, b₀` are learnable parameters updated via the M-step. When beliefs match priors (KL ≈ 0), α is large (trust prior). When beliefs drift (KL large), α decreases (release prior, let attention dominate).

The pure VFE has a fixed `α = c₀/(b₀ + KL)` with non-learnable `c₀, b₀`. Making these learnable and adding the product-rule correction `∂(α·KL)/∂θ -= (α²/c₀)·∂KL/∂θ` would be the single most impactful addition.

### 4.2 Lambda Weights for Gradient Components (HIGH IMPACT)

The nn.Module version has separate `E_lambda_belief` and `E_lambda_softmax` controlling the relative weight of the linear (direct alignment) and nonlinear (Boltzmann gate / softmax coupling) gradient components:

```
grad_mu = α·(self-coupling) + λ_belief·(direct alignment) + λ_softmax·(Boltzmann gate)
```

The pure VFE implicitly uses `λ_belief = λ_softmax = 1`. Making these configurable (and setting `λ_softmax > λ_belief`) increases the nonlinear fraction of the gradient, which is the model's only source of nonlinearity beyond the attention softmax.

### 4.3 Per-Head Multihead VFE (HIGH IMPACT)

The nn.Module version's `multihead_vfe=True` maintains separate β per head throughout all E-step iterations. Each head has its own attention pattern and its own Boltzmann gate. The pure VFE computes per-head KL and attention, but the gradient and update path should be verified to maintain full head independence throughout iterations.

### 4.4 Learnable E-Step Learning Rate (MEDIUM IMPACT)

The nn.Module version learns a scalar `η = softplus(raw_η)` updated via the M-step. This consistently helps — the model discovers the optimal step size rather than requiring manual tuning.

For the pure VFE, this could be implemented as a learnable parameter updated by the M-step: track the correlation between step size and VFE reduction, adjust η to maximize VFE decrease per step.

### 4.5 Cosine Decay Within E-Step Iterations (LOW IMPACT)

The nn.Module version decays the effective learning rate within E-step iterations:

```
decay = 0.1 + 0.9 * 0.5 * (1 + cos(π * step / n_steps))
effective_lr = base_lr * decay
```

This prevents overshooting in later iterations when beliefs are near convergence. Easy to add.

### 4.6 Learnable Per-Head Temperature (LOW-MEDIUM IMPACT)

The nn.Module version supports `learnable_head_kappa=True`: each head learns its own softmax temperature `κ_h = exp(log_κ_h)`. This enables head specialization (sharp vs diffuse attention). The pure VFE uses a single global τ (or τ = √K_h per head, fixed).

### 4.7 Observation Grounding in E-Step (EXCLUDED BY DESIGN)

The nn.Module version can compute `∂CE/∂μ` inside the E-step (`use_obs_in_vfe=True`). This was tested and causes catastrophic memorization (train PPL → 1.1, val PPL explodes). The pure VFE correctly places observations only in the M-step. This is not a missing feature — it is the right design.

---

## 5. Performance Characteristics

### Computational Cost

| Operation | Cost | Notes |
|---|---|---|
| Pairwise KL (per E-step) | O(B · H · N² · K_h²) | Bottleneck; CUDA kernel available |
| Gauge frame gradient | O(B · H · N² · K_h²) | Per-pair, per-head |
| SPD retraction | O(B · N · K³) | Eigendecomposition per token |
| KL-decode | O(B · N · V · K) | Chunked over V for memory |
| M-step scatter | O(B · N) per token type | Amortized over batch |
| **Total per step** | **O(n_esteps · B · N² · K²)** | **12x slower than single-pass** |

For PURE_VFE_CONFIG (K=32, N=64, B=32, n_esteps=12): ~12x slower than a comparable single-iteration model.

### Memory

```
Prior bank: V × (K + K² + H × K_h²) ≈ 50k × (32 + 1024 + 4×64) ≈ 65M floats ≈ 260 MB
Beliefs:    B × N × (K + K² + H × K_h²) ≈ 32 × 64 × 1312 ≈ 2.7M floats ≈ 11 MB
Pairwise:   B × H × N × N ≈ 32 × 4 × 64 × 64 ≈ 0.5M floats ≈ 2 MB
```

Prior bank dominates. Σ (V × K × K) is the largest component.

---

## 6. Priority Improvements for Language Modeling

Ordered by expected impact on PPL:

### Priority 1: Learnable α with Product-Rule Correction

```python
# In config:
learnable_alpha: bool = True
raw_c0: float = 1.0  # softplus → c₀
raw_b0: float = 1.0  # softplus → b₀

# In E-step:
c0 = softplus(raw_c0)
b0 = softplus(raw_b0)
kl_qp = KL(q_i || p_i)  # per-position, per-dim
alpha = c0 / (b0 + kl_qp)  # (B, N, K)

# Gradient includes product-rule correction:
grad_mu_self = alpha * Sigma_p_inv * (mu - mu_p)
grad_mu_self -= (alpha**2 / c0) * kl_qp * Sigma_p_inv * (mu - mu_p)

# M-step: update raw_c0, raw_b0 via natural gradient
```

### Priority 2: Lambda Weights

```python
# In config:
lambda_belief: float = 2.0    # direct alignment weight
lambda_softmax: float = 3.0   # Boltzmann gate weight

# In E-step gradient:
grad_mu = alpha * grad_self + lambda_belief * grad_align + lambda_softmax * grad_softmax
```

### Priority 3: Learnable E-Step LR

```python
# In model:
raw_lr: float = 0.0  # softplus → η; initialized to match mu_q_lr

# In E-step:
eta = softplus(raw_lr)
mu = mu - eta * nat_mu

# In M-step: update raw_lr by tracking VFE reduction per step
```

### Priority 4: Cosine Decay in E-Step

```python
# In E-step loop:
progress = step / n_esteps
decay = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
effective_lr = base_lr * decay
```

### Priority 5: Finite-Difference Tests for M-Step

```python
# In test_gradients.py:
def test_m_step_grad_mu_obs():
    """∂CE/∂μ_v via finite differences."""
    # Perturb prior_mu[v] by ±eps
    # Compute CE loss both ways
    # Compare with analytical gradient

def test_m_step_grad_Sigma_obs():
    """∂CE/∂Σ_v via finite differences (full mode)."""
    # Perturb prior_Sigma[v] eigenvalues by ±eps
    # Compute CE loss both ways
    # Compare with analytical gradient
```

---

## 7. The Fundamental Question

The nn.Module version achieves PPL 71.6 with **1 VFE iteration** and autograd-based M-step. The pure VFE has **12 iterations** but no autograd and fewer features.

The question: can pure natural gradient VFE — with no backpropagation, no learned projections, no neural components beyond a linear decode temperature — reach competitive perplexity?

The nn.Module version's advantage comes not from autograd per se but from features enabled by it: learnable α, learnable LR, per-head diversity, and the Boltzmann gate's interaction with gradient-based M-step. Adding the analytical equivalents of these features (Priority 1-4 above) to the pure VFE would be the cleanest test.

If the pure VFE with these features matches the nn.Module version, it demonstrates that language modeling is achievable as pure variational inference — no neural networks required, just iterative VFE minimization on a statistical manifold with gauge-covariant transport. That would be the strongest possible evidence for the gauge-theoretic framework.
