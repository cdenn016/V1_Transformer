# Investigation: Two Uses of Alpha/Beta — FFN E-step vs Training Loss M-step

## Summary

The codebase has two distinct free energy computations that both use `alpha` and `beta`, but in different contexts with different defaults. This is **intentional EM algorithm design**, not redundancy.

| Aspect | FFN VFE Descent (E-step) | Training Loss (M-step) |
|--------|--------------------------|------------------------|
| **File** | `variational_ffn.py:1724–2250` | `train.py:344–600` |
| **Alpha default** | `ffn_alpha = 0.001` | `alpha = 0.1` |
| **Beta** | Dynamic (recomputed each iteration) | Static (final attention snapshot) |
| **Purpose** | Optimize beliefs given fixed model params | Optimize model params given fixed beliefs |
| **Targets** | `None` during training (see below) | Used for CE loss computation |

---

## 1. Inside the FFN: VFE Descent (E-step)

**Location**: `variational_ffn.py`, `forward()` method (line 1724)

At each of `n_iterations` VFE steps, the FFN:

1. **Recomputes β** dynamically: `β_ij = softmax(-KL(q_i||Ω_ij[q_j])/κ)` (line 1979)
2. **Computes VFE gradient** with three terms:
   - Self-coupling: `α · (μ_q - μ_p) / σ_p` — pulls beliefs toward priors
   - Belief alignment: `λ_β · Σ_j β_ij · ∂KL_ij/∂μ_i` — aligns beliefs across agents
   - Observation: `∂CE/∂μ` (only when targets provided) — see critical note below
3. **Updates beliefs** via natural gradient descent: `μ ← μ - η·F⁻¹·∇F(μ)` (line 2043–2066)

### Alpha in FFN

```python
# model.py:128
ffn_alpha = config.get('ffn_alpha', config.get('alpha', 0.001))
```

- Controls how strongly beliefs are anchored to embedding priors during VFE descent
- Defaults to `0.001` (weak anchoring — allows beliefs to move freely)
- Can be made learnable per-token via `learnable_alpha=True` (Bayesian precision)
- Passed through: `model.py:435` → `blocks.py:246` → `variational_ffn.py` constructor

### Beta in FFN

- **Recomputed at every VFE iteration** from current beliefs (line 1979–1994)
- Creates positive feedback: similar beliefs → higher β → stronger alignment → more similar beliefs
- Enables emergent "meta-agent" clustering

---

## 2. Training Loss: `compute_free_energy_loss` (M-step)

**Location**: `train.py:344–600`

After the complete forward pass, computes four loss terms:

```
Total_Loss = CE + λ_β · belief_align + α · self_consistency + λ_γ · model_align
```

### CE (Observation Likelihood)
Standard cross-entropy from final beliefs to targets. (line 425–430)

### Belief Alignment (λ_β term)
```
L_β = λ_β · Σ_{i,j} β_ij · KL(q_i || Ω_ij[q_j]) / (2√K)
```
Uses **static** β from the last attention layer output. (line 438–454)

### Self-Consistency (α term)
```
L_α = α · KL(q_i || p_i) / (2√K)
```
**This is the critical term that trains embedding priors.** Provides gradient flow to `mu_embed` and `sigma_embed` even when FFN outputs are detached. (line 466–488)

### Model Alignment (λ_γ term)
```
L_γ = λ_γ · Σ_{i,j} γ_ij · KL(p_i || Ω_ij[p_j]) / (2√K)
```
Regularizes embedding space gauge consistency. Disabled by default (`λ_γ = 0`). (line 511–548)

### Alpha in Loss

```python
# train.py:694 (TrainingConfig)
alpha: float = 0.1
```

- Controls embedding prior training strength
- Default `0.1` — **100× larger** than FFN's default `0.001`
- Comment at line 691: "alpha > 0 is CRITICAL for gradient flow to embeddings!"

### Beta in Loss

- **Static**: extracted from `attn_info['beta']` (final attention output, line 411)
- NOT recomputed — represents the "snapshot" after all VFE dynamics

---

## 3. Critical: Targets Not Passed During Training

```python
# train.py:409 — the critical line
logits, attn_info = model.forward_with_attention(token_ids, targets=None)
```

Despite `blocks.py:383` accepting targets and `variational_ffn.py:2024–2034` computing observation gradients when targets are present, **during training via `compute_free_energy_loss`, targets=None is passed**. This means:

- `has_observations = targets is not None and W_out is not None` → **False** (line 1843)
- The observation gradient `∂CE/∂μ` is **NOT computed** inside the FFN during training
- VFE descent uses only the prior anchoring (α) and belief alignment (β) terms
- The observation likelihood comes **solely** from the external CE loss

Comment at `train.py:405–408` explains: *"Passing targets allows VFE FFN to 'cheat' by using targets to adjust beliefs before CE is computed, causing CE to collapse to 0."*

This is correct EM design: the E-step should optimize beliefs using the current model, and the M-step should optimize model parameters using the resulting beliefs.

---

## 4. Config Hierarchy for Alpha

```
ffn_alpha = config.get('ffn_alpha',      # 1st: explicit FFN alpha
              config.get('alpha',          # 2nd: shared alpha
                          0.001))          # 3rd: hardcoded default

TrainingConfig.alpha = 0.1                 # Used in loss computation
```

Three scenarios:
1. **Default**: FFN gets `0.001`, loss gets `0.1` — they differ by 100×
2. **Set only `alpha` in config**: FFN falls back to config `alpha`, loss uses `TrainingConfig.alpha` — still independent
3. **Set `ffn_alpha` explicitly**: Fully decoupled

---

## 5. Why They're Different (By Design)

The 100× difference in defaults makes physical sense:

- **FFN α = 0.001 (weak)**: During belief optimization, you want beliefs to move freely to find the VFE minimum. Strong prior anchoring would prevent belief evolution.
- **Loss α = 0.1 (moderate)**: During parameter training, you need meaningful gradients flowing to embeddings. Too weak → embeddings don't learn from belief evolution. Too strong → beliefs can't diverge from priors (same as not having VFE at all).

This implements proper **Expectation-Maximization**:
- **E-step** (FFN): Minimize free energy over beliefs with weak prior constraint
- **M-step** (Loss): Update model parameters (embeddings, W_out) given optimized beliefs with moderate prior regularization

---

## 6. Key Code References

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| FFN alpha config | `model.py` | 123–128 | `ffn_alpha` from config with fallback chain |
| FFN VFE loop | `variational_ffn.py` | 1853–2184 | VFE descent with dynamic β, α, observation gradient |
| Self-coupling gradient | `variational_ffn.py` | 2000–2016 | `compute_vfe_gradients_gpu` with `alpha=alpha_effective` |
| Observation gradient | `variational_ffn.py` | 2024–2034 | Fresh `∂CE/∂μ` (only when targets ≠ None) |
| Training loss | `train.py` | 344–600 | Four-term free energy loss |
| Loss alpha | `train.py` | 466–488 | `self_consistency_loss = α · KL(q\|\|p) / 2√K` |
| Loss beta | `train.py` | 438–454 | `belief_align_loss = λ_β · Σβ·KL / 2√K` |
| Config defaults | `train.py` | 690–697 | `alpha=0.1, lambda_beta=1.0, lambda_gamma=0.0` |
| Targets=None | `train.py` | 405–409 | Prevents VFE "cheating" |
| Targets to FFN | `blocks.py` | 383–384 | Path exists but receives None during training |

---

## 7. Potential Improvements

1. **Documentation**: The 100× default difference between `ffn_alpha` (0.001) and loss `alpha` (0.1) is not documented in TrainingConfig. Adding a note would help users understand the relationship.

2. **Config unification**: Currently `ffn_alpha` falls back to `config['alpha']` then to `0.001`, while `TrainingConfig.alpha` defaults to `0.1`. This means setting `alpha` in the model config and `alpha` in training config can independently affect the two terms — which is correct but potentially confusing.

3. **Naming clarity**: Using `alpha` for both (with only a `ffn_` prefix to distinguish) can be confusing. Consider renaming to `prior_weight_estep` / `prior_weight_mstep` or similar in a future refactor.
