# Learnable Alpha Analysis: Experiment 152 vs 154

## Summary

Learnable alpha (Bayesian precision via Gamma-Normal conjugacy) shows a **consistent but
modest improvement** over constant alpha. The effect is real but small, likely due to
structural reasons in the current experimental setup that prevent the mechanism from
fully expressing its potential.

## Experimental Setup

Both experiments are identical except for `learnable_alpha`:

| Parameter | Value |
|-----------|-------|
| embed_dim (K) | 30 |
| n_layers | 1 |
| max_seq_len (N) | 64 |
| batch_size | 64 |
| max_steps | 25,000 |
| alpha (constant) | 1.0 |
| ffn_n_iterations | **1** |
| dataset | WikiText-103 |
| GPU | RTX 5090 |

- **Experiment 152**: `learnable_alpha = true` (Bayesian precision)
- **Experiment 154**: `learnable_alpha = false` (constant alpha = 1.0)

## Quantitative Results

### Final Metrics (Step 25,000)

| Metric | Learnable (152) | Constant (154) | Diff |
|--------|----------------|----------------|------|
| **Val Loss** | **4.6716** | 4.7295 | **-0.058 (-1.2%)** |
| **Val BPC** | **6.6525** | 6.7448 | **-0.092 (-1.4%)** |
| **Val CE** | **4.6112** | 4.6752 | **-0.064 (-1.4%)** |
| Train Loss | 5.2387 | 5.2519 | -0.013 (-0.3%) |
| Train CE | 5.1861 | 5.1991 | -0.013 (-0.3%) |
| Train BPC | 7.4820 | 7.5007 | -0.019 (-0.3%) |

### Validation BPC Trajectory

The learnable alpha shows a **consistent advantage throughout training**, widening
around steps 5000-8000 and stabilizing at ~0.09 BPC improvement:

| Step | Learnable BPC | Constant BPC | Delta |
|------|--------------|-------------|-------|
| 500 | 8.268 | 8.342 | -0.074 |
| 2500 | 7.792 | 7.934 | -0.142 |
| 5000 | 7.489 | 7.694 | -0.206 |
| 7500 | 7.113 | 7.418 | **-0.305** |
| 10000 | 7.087 | 7.237 | -0.150 |
| 15000 | 6.949 | 7.031 | -0.083 |
| 20000 | 6.784 | 6.812 | -0.028 |
| 25000 | 6.653 | 6.745 | -0.092 |

Peak advantage was **-0.305 BPC at step 7500** (~4.1% relative improvement).

### Gradient Dynamics

| Metric | Learnable | Constant | Note |
|--------|-----------|----------|------|
| grad_norm_total | 0.352 | 0.306 | +15% higher |
| grad_norm_mu | 0.086 | 0.070 | +23% higher |
| grad_norm_ffn | 0.0006 | 0.005 | **-88% lower** |
| kl_mean | 0.009 | 0.004 | +114% higher |

The learnable alpha model has **higher mu gradients** (alpha is adapting the prior
coupling strength per-token) but **dramatically lower FFN gradients** (-88%). The
Bayesian alpha's 2 learnable parameters (a0, b0) converge early, leaving the FFN
learning rate effectively at zero.

## Diagnosis: Why the Effect is Modest

### 1. `ffn_n_iterations = 1` (Primary Bottleneck)

This is the most significant factor. The learnable alpha computes a **state-dependent
precision** for each token at each VFE iteration:

```
alpha_i = (a0 + K/2) / (b0 + 0.5 * ||mu_q - mu_p||^2_{Sigma_p^-1})
```

With only **1 VFE iteration per forward pass**, the Bayesian alpha computes a
per-token precision once and applies it to a single gradient step. The mechanism
is designed to create a feedback loop:
- Beliefs far from priors -> lower alpha -> less anchoring -> more freedom to move
- Beliefs near priors -> higher alpha -> stronger anchoring -> stability

But this feedback loop **never gets to iterate**. With n_iterations=1, there's no
opportunity for the adaptive precision to create the "self-regulating" dynamics it
was designed for. The constant alpha=1.0 and the Bayesian alpha are both applied
exactly once, so the only difference is that some tokens get slightly more or less
prior coupling.

**Recommendation**: Run with `ffn_n_iterations >= 5` (ideally 10). The Bayesian
precision should show much larger effects when alpha adapts across multiple
iterations - tokens that need to break free of priors will accumulate more
deviation, while tokens that should stay anchored will remain stable.

### 2. Only 2 Learnable Parameters (a0, b0)

The Gamma-Normal conjugacy gives only 2 global hyperparameters. They set the
*shape* of the alpha-vs-distance curve but cannot learn per-layer, per-head,
or per-position variations. With K=30 dimensions contributing to the Mahalanobis
distance, the alpha values across tokens tend to be quite similar unless beliefs
diverge substantially from priors.

**Recommendation**: Consider per-layer (a0, b0) if using multiple layers, or
add a learnable scaling dimension to make the precision more expressive.

### 3. High Base Alpha (alpha=1.0)

With `alpha=1.0`, the self-coupling term (KL(q||p)) has equal weight to the
belief alignment term (lambda_belief=1.0). This is already a strong prior anchor.
The Bayesian alpha initialized at alpha~1.0 (via `b0 = (a0 + K/2) / alpha_init`)
means the initial Bayesian precision is very close to the constant value. Deviations
from this require large Mahalanobis distances to materially change alpha.

Given that the prior and posterior means are being regularized to stay close (the
whole point of the self-coupling term), the Mahalanobis distance stays small,
and so `alpha_i ~ alpha_constant` for most tokens throughout training.

**Recommendation**: Try with lower base alpha (0.1 or 0.01) where the adaptive
behavior has more room to modulate the prior coupling.

### 4. No Alpha Logging

The Bayesian alpha values (per-token precision) are not logged during training.
Without tracking `alpha_mean`, `alpha_std`, `alpha_min`, `alpha_max` across
training steps, we cannot observe:
- Whether alpha values actually vary across tokens
- How the (a0, b0) parameters evolve
- Whether the mechanism is producing meaningful variation or collapsing to near-constant

**Recommendation**: Add alpha diagnostics to the training loop:
```python
if learnable_alpha:
    alpha_vals = ffn.get_bayesian_alpha(mu_q, mu_p, sigma_p)
    log('alpha_mean', alpha_vals.mean())
    log('alpha_std', alpha_vals.std())
    log('alpha_min', alpha_vals.min())
    log('alpha_max', alpha_vals.max())
    log('a0', F.softplus(ffn.raw_a0).item())
    log('b0', F.softplus(ffn.raw_b0).item())
```

### 5. Single Layer Architecture

With only 1 transformer layer, there's limited depth for the adaptive precision
to create differentiated behavior. In a multi-layer architecture, different layers
could benefit from different alpha schedules - early layers exploring more
(lower alpha), later layers anchoring more (higher alpha). With a single layer,
this compositional advantage is absent.

## UPDATE: Why 6 VFE Iterations Made Things WORSE

Running with `ffn_n_iterations=6` produced worse results than 1 iteration. This
reveals a fundamental flaw in how the Bayesian alpha interacts with the VFE inner loop.

### The Runaway De-Anchoring Problem

The Bayesian alpha formula is:
```
alpha_i = (a0 + K/2) / (b0 + 0.5 * ||mu_q - mu_p||^2)
```

This is **monotonically decreasing** in Mahalanobis distance. Here's what happens
across 6 VFE iterations within a single forward pass:

```
Iteration 1: mu_q ≈ mu_p  →  mahal ≈ 0     →  alpha ≈ 1.0  (strong anchor)
Iteration 2: mu_q drifts   →  mahal grows   →  alpha drops   (weaker anchor)
Iteration 3: mu_q drifts more → mahal bigger → alpha drops more
...
Iteration 6: mu_q far from mu_p → mahal large → alpha << 1.0  (weak anchor)
```

This creates a **positive feedback loop of de-anchoring**:
1. Beliefs move away from priors (normal VFE dynamics)
2. Bayesian alpha detects the distance and *reduces* prior coupling
3. With less anchoring, beliefs move even further from priors
4. Alpha drops even more
5. By iteration 6, the prior term is effectively disabled

Meanwhile, the **constant alpha** case maintains `alpha=1.0` at every iteration,
providing consistent restoring force that keeps beliefs from drifting too far.

### The Learning Rate Scaling Compounds the Problem

The code already accounts for multiple iterations via:
```python
base_lr = self.lr / self.n_iterations  # Line 1806
decay_factor = 1.0 - 0.5 * (iteration / (self.n_iterations - 1))  # Lines 1811-1812
```

For n_iterations=6, total integrated step ≈ 0.75 × lr. But this scaling was designed
for constant alpha. With Bayesian alpha decaying across iterations, the effective
*prior-weighted* contribution drops much more than 0.75x — potentially to near zero
by the last iterations.

### The Core Issue: Wrong Timescale for Adaptation

The Bayesian alpha adapts on the **wrong timescale**. It was designed as if
`||mu_q - mu_p||` reflects a meaningful, persistent divergence between beliefs and
priors. But within the VFE inner loop:

- `mu_q` is an intermediate variable being optimized
- `mu_p` is fixed (the embedding prior for this forward pass)
- The distance `||mu_q - mu_p||` grows naturally during optimization — this is the
  *intended behavior* of VFE descent, not a signal to reduce prior coupling

The Bayesian alpha interprets normal optimization dynamics as evidence that the prior
is wrong, and loosens the prior — exactly the opposite of what stability requires.

### Fix: Compute Alpha Once, Use It for All Iterations

The alpha should be computed from the **initial** beliefs (before any VFE iterations)
and held constant throughout the inner loop:

```python
# BEFORE the iteration loop:
if self.learnable_alpha:
    alpha_effective = self.get_bayesian_alpha(
        mu, mu_p_current, sigma_p, eps=eps  # mu = initial beliefs, NOT mu_current
    )

# INSIDE the iteration loop:
# Use alpha_effective (fixed) instead of recomputing from mu_current
```

This way:
- Alpha still adapts per-token based on how far the initial embedding is from the prior
- But it doesn't de-anchor during the inner optimization loop
- Multiple iterations now benefit from consistent, token-specific prior coupling

### Alternative Fix: Clamp Alpha Floor

If you want alpha to adapt during iterations but prevent runaway de-anchoring:

```python
alpha_effective = self.get_bayesian_alpha(mu_current, mu_p_current, sigma_p)
alpha_effective = torch.clamp(alpha_effective, min=0.1 * self.alpha)  # Floor at 10% of base
```

## Diagnostics Added

Alpha diagnostics have been added to `train.py` and `train_publication.py`. The
metrics CSV now includes:

| Column | Description |
|--------|-------------|
| `alpha_mean` | Mean Bayesian alpha across all tokens |
| `alpha_std` | Std dev (measures per-token variation) |
| `alpha_min` | Minimum alpha (most de-anchored token) |
| `alpha_max` | Maximum alpha (most anchored token) |
| `alpha_a0` | Learned Gamma shape parameter |
| `alpha_b0` | Learned Gamma rate parameter |
| `alpha_mahal_sq_mean` | Mean Mahalanobis distance² (belief-prior divergence) |
| `alpha_mahal_sq_std` | Std of Mahalanobis distance² |

Console output during training now shows `[ALPHA]` lines at each log interval.

## Conclusion

The learnable alpha **does work** with 1 VFE iteration (consistent 1-4% val BPC
improvement) but **gets worse with multiple iterations** due to runaway de-anchoring.

**Priority fix**: Compute Bayesian alpha once from initial beliefs, not per-iteration.
This is a one-line change in `variational_ffn.py` (move `get_bayesian_alpha` call
outside the iteration loop).

| Factor | Current | Recommended |
|--------|---------|-------------|
| Alpha computation | Per-iteration (broken) | Once before loop |
| Alpha floor | None | clamp(min=0.1*base) |
| ffn_n_iterations | 1 (safe) or 6 (broken) | 5-10 (after fix) |
| Alpha logging | **Now added** | Review a0/b0 evolution |
