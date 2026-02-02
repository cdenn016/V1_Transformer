# Numerical Stability for Large Gauge Groups

## The Problem

Training with large K (e.g., K=100) can exhibit non-monotonic loss if numerical
issues cause transport operators to become singular or ill-conditioned. This
document explains the fixes applied, distinguishing between **bug fixes**
(mathematical correctness), **justified engineering** (sound approximations),
and **pragmatic choices** (hyperparameter convenience).

---

## 1. Transport Operator Invertibility (GL(K) Gauge Structure)

**Category: Theoretical clarification + simplified implementation**

### Key Insight: GL(K) Suffices for Gauge Invariance

**CRITICAL DISCOVERY:** The variational free energy is invariant under GL(K)
gauge transformations, not just SO(K)! This is because all f-divergences
(including KL) are invariant under pushforward by invertible linear maps:

```
D_KL(Ω·P || Ω·Q) = D_KL(P || Q)  for any Ω ∈ GL(K)
```

The Jacobian factors cancel in the density ratio. This means:
- Transport operators need only be **invertible** (det ≠ 0), not orthogonal
- No re-orthogonalization is required
- No Newton-Schulz iterations needed
- No SVD projection to SO(K) needed

### Old Approach (SO(K) - DEPRECATED)

Previously, the framework required Ω ∈ SO(K) (Ω^T Ω = I, det(Ω) = +1), which
required expensive re-orthogonalization steps to correct numerical drift.

### New Approach (GL(K))

```
Ω_ij = exp(φ_i · G) · exp(-φ_j · G) ∈ GL(K)
```

The only requirements are:
1. **Invertibility**: |det(Ω)| > ε (not singular)
2. **Numerical conditioning**: cond(Ω) < 10^10 (not ill-conditioned)

### Implementation

```python
# OLD (deprecated): Re-orthogonalization
Q = Q @ (3*I - Q.T @ Q) / 2  # Newton-Schulz - NO LONGER NEEDED

# NEW: Just check invertibility
if abs(det(Omega)) < eps:
    raise ValueError("Transport operator singular")
if cond(Omega) > 1e10:
    warnings.warn("Transport operator ill-conditioned")
```

### Why This Works

The proof is straightforward. For KL between Gaussians:
- Trace term: tr((ΩΣ₂Ωᵀ)⁻¹(ΩΣ₁Ωᵀ)) = tr(Σ₂⁻¹Σ₁) ✓
- Quadratic term: (Ω(μ₁-μ₂))ᵀ(ΩΣ₂Ωᵀ)⁻¹(Ω(μ₁-μ₂)) = (μ₁-μ₂)ᵀΣ₂⁻¹(μ₁-μ₂) ✓
- Log-det term: log(det(ΩΣ₂Ωᵀ)/det(ΩΣ₁Ωᵀ)) = log(detΣ₂/detΣ₁) ✓

The (det Ω)² terms cancel! Orthogonality was never required.

### Migration Notes

If you have code that enforces SO(K) constraints, you can safely remove:
- `_project_to_orthogonal()` calls
- Newton-Schulz re-orthogonalization
- `_validate_orthogonal()` checks (replace with `_validate_invertible()`)
- Cayley transform constraints
- Skew-symmetry enforcement (optional, keeps SO(K) subalgebra if desired)

---

## 2. Attention Temperature Scaling: `logits = -KL / (κ · √K)`

**Category: Justified engineering**

### Theory

Attention weights are a Boltzmann distribution over information distances:

```
β_ij = softmax_j( -KL(q_i || Ω_ij[q_j]) / κ )
```

The temperature κ is a free parameter in the theory—the active inference
framework does not specify its value.

### What went wrong

KL divergence between K-dimensional Gaussians sums K terms:

```
KL = 0.5 · [ Σ_k (σ_q[k]/σ_p[k])  +  Σ_k (Δμ[k])²/σ_p[k]  -  K  +  Σ_k log(σ_p[k]/σ_q[k]) ]
```

At initialization with σ_q ≈ σ_p, the trace term ≈ K and the −K cancels exactly,
giving KL ≈ 0. During training, per-dimension deviations accumulate. The
statistical question is how KL scales with K:

| Regime | KL scaling | Explanation |
|--------|-----------|-------------|
| Independent per-dim deviations | O(√K) variance | Central Limit Theorem |
| Correlated deviations (e.g., transport mixing) | O(K) worst case | All terms shift together |

With K=100 and κ=1.0, logits reach −50 to −100, making softmax essentially
one-hot. This produces:

- Zero gradients for all but one token pair (softmax saturation)
- Abrupt attention switching when the dominant target changes
- Non-monotonic loss as the model oscillates between attention patterns

### Fix

```python
dim_scale = math.sqrt(max(K, 1))
logits = -kl_matrix / (kappa * dim_scale)
```

This is equivalent to `κ_eff = κ · √K`.

### Justification

This is directly analogous to `1/√d_k` scaling in standard transformer attention
(Vaswani et al., 2017). There, the dot product Q·K sums d_k terms and is
normalized by √d_k to keep softmax non-degenerate. The KL divergence plays the
role of the dot product; K plays the role of d_k.

From a statistical mechanics perspective: the temperature of a Boltzmann
distribution should be set relative to the energy scale of the system. If typical
KL ∼ O(√K) (independent per-dimension deviations), then κ ∼ √K keeps the
distribution non-degenerate.

**The theory permits any κ.** The √K scaling chooses κ such that the same
numerical value of the hyperparameter produces similar attention entropy across
different K. This is a normalization convention, not a theoretical claim.

**Caveat:** Previously tuned κ values implicitly absorbed the K-dependence. For
K=3, the scaling factor is √3 ≈ 1.7 (mild). For K=100, it is 10 (significant).
If restoring a checkpoint trained without this scaling, adjust κ accordingly.

---

## 3. KL Clamp Ceiling: `max(100, 5K)`

**Category: Bug fix**

### Theory

The KL clamp is a numerical guard, not a theoretical quantity. Its purpose is to
prevent `inf` or overflow in downstream computation. The theory says nothing about
clamping—it's pure implementation.

### What went wrong

The ceiling was hardcoded at 100. With K=100, a per-dimension KL of just 1.0 nat
gives total KL = 100, hitting the ceiling for moderate divergence. When the clamp
activates:

- Gradient is exactly zero (flat region in the loss landscape)
- The model cannot improve beliefs that are moderately far from the prior
- These "dead zones" contribute to non-monotonic training

### Fix

```python
kl_ceil = max(100.0, 5.0 * K)
```

### Justification

Straightforward: a dimension-dependent quantity needs a dimension-dependent guard.
The factor 5 is generous (allows per-dimension KL up to 5 nats before clamping).

---

## 4. Per-Parameter-Group Gradient Clipping

**Category: Standard optimization practice**

### Theory

The gauge-theoretic framework is agnostic to optimization algorithm. Gradient
clipping is an implementation concern.

### What went wrong

With SO(100), parameter dimensions are:

| Parameter | Dimension per token | Fraction of total |
|-----------|--------------------:|------------------:|
| phi_embed | 4950 | ~96% |
| mu_embed  | 100  | ~2%  |
| sigma     | 100  | ~2%  |

Global `clip_grad_norm_` at 1.0 means phi consumes ~96% of the gradient budget.
The mu and sigma gradients are scaled down by ~50x, effectively frozen.

### Fix

When `use_param_groups=True`, clip each parameter group independently:

```python
for group in optimizer.param_groups:
    torch.nn.utils.clip_grad_norm_(group['params'], grad_clip)
```

### Justification

Standard practice in multi-scale optimization (used in DALL-E, Stable Diffusion,
etc.). Each parameter type gets its own gradient budget regardless of
dimensionality.

---

## 5. Free Energy Loss Normalization by √K

**Category: Pragmatic choice (changes the functional)**

### Theory

The variational free energy is:

```
F = α · Σ_i KL(q_i || p_i)                         [self-consistency]
  + λ_β · Σ_{i,j} β_ij · KL(q_i || Ω_ij q_j)      [belief alignment]
  + CE(logits, targets)                              [observation likelihood]
```

The theory says **minimize F**. It does not prescribe normalization of the KL
terms.

### What went wrong

The CE loss is O(1) in K—it operates over the vocabulary, not the latent
dimension. The KL terms scale as O(√K) to O(K) during training. With K=100 and
the default hyperparameters (α=0.1, λ_β=1.0), the KL terms dominate CE by 10–30x.
The model minimizes belief divergence at the expense of token prediction accuracy.

### Fix

```python
dim_scale = math.sqrt(max(K, 1))
belief_align_loss = lambda_beta * weighted_kl.sum(...).mean() / dim_scale
self_consistency_loss = alpha * kl_per_agent.mean() / dim_scale
```

### Justification

**Arguments for:**

1. **β-VAE analogy.** In β-VAE (Higgins et al., 2017), the KL term is explicitly
   scaled relative to reconstruction loss to balance complexity vs. accuracy. Our
   situation is structurally identical: KL (complexity) vs. CE (accuracy).

2. **Per-dimension information rate.** Dividing total KL by √K can be interpreted
   as measuring per-dimension divergence intensity rather than total divergence.
   Both are valid information-geometric quantities.

3. **Hyperparameter transferability.** The same α and λ_β values produce similar
   loss landscapes across different K, making experimental results comparable.

**Arguments against:**

1. **Changes the mathematical functional.** The theory defines F as the sum of
   specific information-theoretic quantities. Normalizing changes what is being
   minimized. Strictly, the fix is equivalent to redefining:
   ```
   α_eff = α / √K
   λ_β_eff = λ_β / √K
   ```

2. **User should tune hyperparameters.** A purist position: when changing K, the
   user should rescale α and λ_β by ~1/√K to maintain the same balance with CE.
   The normalization does this automatically but silently.

3. **Not derivable from the variational principle.** The free energy functional
   emerges from variational inference. The normalization is not part of that
   derivation.

### Recommendation

This is the least theoretically clean fix. If strict adherence to the theoretical
functional is preferred, **revert this change** and instead:

- Set `α = 0.1 / sqrt(K)` and `λ_β = 1.0 / sqrt(K)` in the training config
- Document that KL-based hyperparameters must be scaled with K

The current implementation chooses convenience (same hyperparameters work across
K) over theoretical purity (exact variational free energy). Both approaches
produce the same optimization landscape.

---

## Summary Table

| Fix | Category | Theoretical status | Reversible? |
|-----|----------|-------------------|-------------|
| GL(K) gauge (no re-orth) | Theoretical improvement | **GL(K) suffices** for gauge invariance | N/A—simpler than SO(K) |
| Attention √K scaling | Engineering | Justified (≡ standard 1/√d_k scaling) | Adjust κ if reverting |
| KL clamp ceiling | Bug fix | Dimension-dependent guard | No—always needed |
| Per-group grad clip | Optimization | Standard practice, theory-agnostic | Adjust grad_clip if reverting |
| Loss √K normalization | Pragmatic | Changes effective α, λ_β | Scale hyperparameters by 1/√K instead |

---

## When These Fixes Matter

The scaling fixes (√K attention, loss normalization) are most impactful when:

1. **Large latent dimension**: K ≥ 16
2. **Non-trivial transport**: `use_identity_transport=False` (gauge transport active)

For K ≤ 16, the effects are mild:
- √K scaling at K=3 is a factor of 1.7 (absorbed by κ tuning)
- KL ceiling of 100 is rarely hit
- Small phi_dim means no gradient budget imbalance

### GL(K) vs SO(K) Gauge Structure

With the GL(K) generalization, you have more flexibility:

| Gauge Group | Constraint | Use Case |
|-------------|------------|----------|
| GL(K) | det(Ω) ≠ 0 | Default—simplest, full flexibility |
| SL(K) | det(Ω) = 1 | Volume-preserving, non-compact |
| SO(K) | Ω^T Ω = I, det(Ω) = +1 | Orthogonal, compact, Haar measure exists |
| SO(3) | 3D rotations | Legacy, irrep structure |

For VFE-based objectives, **GL(K) is sufficient and recommended**. Use SO(K) only
if you need:
- Haar measure averaging (gauge consensus)
- Finite-dimensional irreps
- Volume preservation guarantees

---

## Files Modified

The fixes were applied consistently across all numerical paths:

| File | Fixes Applied |
|------|---------------|
| `math_utils/transport.py` | GL(K) support, invertibility checks (replaces orthogonality) |
| `geometry/gauge_consensus.py` | GL(K) reference measure sampling option |
| `transformer/core/embeddings.py` | GL(K) phi_dim documentation |
| `transformer/core/attention.py` | √K attention scaling, KL ceiling |
| `transformer/core/variational_ffn.py` | √K softmax coupling, KL ceiling |
| `transformer/train.py` | √K loss normalization, per-group grad clipping, KL ceiling |
| `transformer/train_publication.py` | Per-group grad clipping |

### GL(K) Migration (February 2026)

The following changes support GL(K) gauge structure:
1. `transport.py`: Removed mandatory orthogonal projection, added `_validate_invertible()`
2. `gauge_consensus.py`: Added `sample_glk_reference()` for non-Haar sampling
3. `embeddings.py`: Updated docs to explain phi_dim options for GL(K)
4. `NUMERICAL_STABILITY.md`: This document updated to reflect GL(K) theory

**Key insight:** Re-orthogonalization was never mathematically required—the VFE is
invariant under the full GL(K), not just SO(K). The orthogonal constraint was a
sufficient but not necessary condition for gauge invariance.
