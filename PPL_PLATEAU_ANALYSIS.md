# PPL Plateau Analysis: Why ~71 Regardless of K=80-120

The model plateaus at test PPL ~71 on WikiText-103 with K=80-120, GL(10)/GL(15)/GL(20), 1 layer, 1 VFE iteration. Increasing K beyond ~80 does not improve PPL. This document identifies the K-independent bottlenecks.

---

## 1. The M-Step Loss Is Pure Cross-Entropy

Current EM_CONFIG loss weights:
```python
M_alpha  = 0.00   # KL(q||p) self-consistency → OFF
M_beta   = 0.0    # belief alignment → OFF
lambda_gamma = 0.0   # model alignment → OFF
lambda_hyper = 0.0   # hyper-prior → OFF
mass_phi = 0.01   # gauge prior (mild regularizer)
```

The only M-step loss driving learning is CE + weight_decay + 0.01 phi regularization. The model is trained as **pure cross-entropy** through a VFE-constrained forward pass. The VFE structure (attention via KL, belief evolution) is implicit in the forward pass but the M-step doesn't explicitly optimize VFE quantities.

This means: the E-step (VFE descent) determines the computational graph through which CE gradients flow, but the M-step loss has no term that rewards good VFE behavior (low self-coupling, good alignment, etc). The embeddings learn to minimize CE, not to be good priors for VFE inference.

**Implication:** The model may learn embeddings that are good for CE but poor for VFE dynamics. For example, embeddings could converge to a state where the single VFE iteration makes only a small correction, and all the useful information comes from the attention sublayer's linear aggregation, not from the Boltzmann gate nonlinearity.

**Possible fix:** Small nonzero M_alpha (e.g., 0.01-0.1) would encourage embeddings that are good initializations for the E-step, because KL(q*||p) measures how far the converged beliefs moved from the prior.

---

## 2. sigma_p Is Barely Learning

From TrainingConfig (not overridden in EM_CONFIG):
```python
sigma_ce_scale: float = 0.01  # 1% of CE gradient flows to sigma_p
```

The PriorBank decode computes:
```
logits_v = -0.5/τ * [(σ_q + μ_q²) @ (1/σ_p) - 2μ_q @ (μ_p/σ_p) + (μ_p²/σ_p + log σ_p)]
```

The gradient of this w.r.t. `log_sigma_p` flows through the `1/σ_p` terms. With sigma_ce_scale=0.01, this gradient is 100x dampened. sigma_p also receives:
- Weight decay (0.05) pulling toward σ=1
- M_alpha * ∂KL(q||p)/∂σ_p — but M_alpha=0.0
- lambda_hyper * ∂KL(s||h)/∂σ_p — but lambda_hyper=0.0

So sigma_p learns from **1% of CE gradient + weight decay only.** The per-token precision profiles σ_p (which control how sharply the decode discriminates between tokens) are essentially frozen near initialization.

**Why this matters regardless of K:** Increasing K adds more dimensions to μ_p, giving more mean-based discrimination. But the precision structure (which dimensions matter for which tokens) doesn't improve because σ_p barely learns. At K=80, the mean capacity saturates — further K gives diminishing returns because the missing discrimination is in precision, not in means.

**Possible fixes:**
- Increase sigma_ce_scale to 0.05-0.1 (with gradient clipping to prevent explosion)
- Add small M_alpha > 0 to provide VFE-based sigma gradient
- Use a separate, larger learning rate for log_sigma_p (currently 0.005, same as sigma in general)

---

## 3. The Boltzmann Gate May Be Underpowered

With 1 VFE iteration, the model's nonlinearity budget is:

| Component | Type | Count per position |
|---|---|---|
| Attention sublayer softmax | Nonlinear (but not trainable gate) | 1 per head |
| VFE FFN softmax (β recomputation) | Nonlinear (same) | 1 per head |
| Boltzmann gate (∂β/∂μ · KL) | Trainable nonlinearity | 1 per head |
| KL-decode softmax | Output nonlinearity | 1 |

The Boltzmann gate computes:
```
grad_softmax = λ_s · Σ_j KL_ij · β_ij · (∂KL_ij/∂μ_i - E_β[∂KL/∂μ]) / κ
```

This is a **competitive normalization**: each pair's contribution is weighted by how much its KL gradient deviates from the mean. The effective "width" of this nonlinearity is N×H (number of attention pairs × heads).

Compare to a standard 1-layer transformer's GELU MLP with hidden dim 4K:
- At K=90: MLP has 360 nonlinear units per position
- VFE has 128×6 = 768 competitive-normalization operations per position

The VFE has MORE nonlinear operations, but each one is weaker: the competitive normalization is a subtractive (zero-mean) correction, whereas GELU provides arbitrary positive/negative output. The Boltzmann gate can only redistribute gradient strength across neighbors, not inject new features.

**Why this matters regardless of K:** The Boltzmann gate's expressiveness is bounded by the attention pattern quality and the number of heads. Adding dimensions K doesn't add more nonlinear operations — it just makes each head operate in a higher-dimensional space. But the gate's competitive normalization structure doesn't benefit from more dimensions; it benefits from more heads (more independent normalizations).

**Possible fix:** More heads with smaller d_head. Move from GL(15)×6 to GL(10)×12 or GL(8)×15. Each head adds an independent Boltzmann gate, directly increasing nonlinear capacity without increasing K.

---

## 4. LayerNorm + VFE Interaction Mismatch

The block structure is:
```python
mu_normalized = LayerNorm(mu_q)          # Zero mean, unit variance
mu_attn = attention(mu_normalized, ...)  # Attention on LN'd beliefs
mu_q = mu_q + mu_attn                   # Residual in original space

mu_normalized = LayerNorm(mu_q)          # Again LN
mu_ffn = vfe_ffn(mu_normalized, ...)     # VFE E-step on LN'd beliefs
mu_q = mu_q + mu_ffn                    # Residual in original space
```

Inside the VFE FFN, the self-coupling gradient is:
```
grad_self = α · (μ_LN - μ_p) / σ_p
```

where μ_LN = LayerNorm(mu_q) has zero mean and unit variance across K dimensions, but μ_p is the raw embedding prior (not normalized). This compares normalized beliefs to unnormalized priors.

The alignment gradient uses the same LN'd beliefs for KL computation. The KL between LN'd beliefs is:
```
KL(q_LN_i || Ω_ij[q_LN_j]) = f(LN(μ_i), LN(μ_j), σ, Ω)
```

LayerNorm removes the mean and scales to unit variance. For GL(K) transport Ω_ij that includes scaling components, LN undoes the scaling. This effectively projects the gauge group from GL(K) to SL(K)×translations (the volume-preserving + mean-centering subgroup). The full GL(K) flexibility is not used.

**Why this matters regardless of K:** LayerNorm constrains the effective gauge group at all K values. The GL(K) transport can scale, rotate, and shear, but LN removes the scale. So the model operates in a constrained subspace of GL(K) regardless of how large K is.

**Possible fix:**
- Try RMSNorm instead of LayerNorm (preserves scale, only normalizes variance)
- Try removing LN from the VFE FFN input (the VFE's own dynamics should handle normalization via the self-coupling to prior)
- Use a "gauge-aware" normalization that preserves transport-relevant structure

---

## 5. Attention Sublayer May Be Redundant with VFE FFN

With `skip_attention=False` (default), the block computes:
1. Attention sublayer: β from initial embeddings → weighted message aggregation
2. VFE FFN: recomputes β from post-attention beliefs → one E-step update

Both compute attention. The VFE FFN's β is from post-attention beliefs, so it's a "refined" attention. But with 1 iteration, the refinement is modest. The two passes may be largely redundant, wasting capacity on similar computation.

The comment in blocks.py (line 274) says:
```
When skip_attention=True, the VFE E-step IS the entire block:
it computes its own β internally, so the separate attention sublayer
is redundant.
```

**This suggests the codebase itself recognizes the redundancy.**

**Possible fix:** Try `skip_attention=True`. This saves one attention computation and lets the VFE FFN handle all context integration. The freed compute could be used for 2 VFE iterations instead of 1 (attention sublayer + VFE FFN with 1 iter → VFE FFN with 2 iters).

---

## 6. Output Rank Is Not the Bottleneck (Eliminating a Hypothesis)

If output rank (2K) were the bottleneck, increasing K from 80→120 (rank 160→240) should improve PPL. It does not. So the model's discrimination ability is not limited by the output projection's rank. The 50k vocabulary can apparently be distinguished with ~160 effective dimensions.

This makes sense: standard transformers with tied embeddings at d=768 have output rank 768, but much of that is redundant. The information-theoretic minimum for 50k classes is log2(50k) ≈ 16 bits, which could be encoded in ~16 dimensions. The practical requirement is higher due to the softmax structure, but 160 dimensions is likely sufficient.

---

## 7. Training Duration and Data

The PPL=71 result used 1 epoch (~103M tokens, 60k steps). The Boltzmann gate takes ~1/4 epoch to activate (per earlier discussion). So the model has ~3/4 epoch of effective nonlinear training.

Standard 1-layer transformers on WikiText-103 at comparable params achieve PPL ~48. The gap (71 vs 48) is significant but not catastrophic. A second epoch would give the Boltzmann gate more time to contribute. However, overfitting is a concern with 1 layer.

**Possible fix:** Train for 2 epochs with appropriate regularization (weight decay, dropout on attention, or KL regularization via small M_alpha).

---

## 8. Concrete Experiments to Break the Plateau

Ordered by expected impact and ease of implementation:

### 8.1 Increase sigma_ce_scale (EASY, HIGH POTENTIAL)

Change from 0.01 to 0.05 or 0.1. This lets sigma_p actually learn from CE gradients, giving the model the ability to develop per-token precision profiles. Monitor sigma_p eigenvalue statistics during training to verify learning.

Risk: positive feedback (σ_p → 0 → gradient explosion). Mitigate with gradient clipping on sigma params and the existing sigma_max floor.

### 8.2 More Heads, Smaller d_head (EASY, HIGH POTENTIAL)

At K=90: switch from GL(15)×6 to GL(10)×9 or GL(8)×11. Each head adds an independent Boltzmann gate. This directly increases nonlinear capacity without changing K or parameter count.

At K=120: try GL(10)×12 or GL(8)×15.

### 8.3 skip_attention=True with 2 VFE iterations (MEDIUM, HIGH POTENTIAL)

Remove the attention sublayer. Use the freed compute for a second VFE iteration. This gives 2 Boltzmann gate applications instead of 1 redundant attention + 1 gate. Doubles the effective nonlinear depth.

This might technically violate the "1 VFE iteration" constraint depending on how strictly it is interpreted. But it preserves the "1 layer" constraint and has the same total compute.

### 8.4 Small M_alpha > 0 (EASY, MEDIUM POTENTIAL)

Try M_alpha = 0.01 to 0.1. This adds KL(q*||p) to the loss, which:
- Provides sigma_p with gradient (the KL depends on both σ_q and σ_p)
- Encourages embeddings that are good VFE initializations
- Regularizes beliefs to stay near priors (prevents the E-step from making useless corrections)

### 8.5 RMSNorm Instead of LayerNorm (MEDIUM, MEDIUM POTENTIAL)

RMSNorm: `μ_rms = μ / RMS(μ) * γ` (no mean subtraction). This preserves the mean of μ while normalizing scale. The GL(K) scaling component of gauge transport is partially preserved, unlike with full LayerNorm which removes it entirely.

### 8.6 Train for 2 Epochs (EASY, MEDIUM POTENTIAL)

Double the training tokens. The Boltzmann gate needs 1/4 epoch to activate; 2 epochs gives it 7/4 epoch of effective nonlinear training vs 3/4 currently. May need to reduce learning rate for stability.

### 8.7 Warm Up lambda_softmax (MEDIUM, MEDIUM POTENTIAL)

Start with high E_lambda_softmax (e.g., 5.0) to amplify the weak early Boltzmann gate, then decay to 2.0-3.0 once the gate activates naturally. This front-loads the nonlinearity.

---

## 9. The K-Independence Diagnostic

The plateau being K-independent (K=80 ≈ K=120) tells us the bottleneck is one of:

1. **Something that doesn't scale with K:** Number of nonlinear operations (heads × iterations), training duration, sequence length, attention pattern quality
2. **Something that scales with K but is capped:** sigma_p learning (capped by sigma_ce_scale), LayerNorm projection (caps effective gauge group)
3. **Something external:** Dataset size, 1-layer depth limitation

The most likely culprit is a combination of (1) insufficient Boltzmann gate capacity with 1 iteration and (2) sigma_p not learning due to the aggressive dampening. Both are K-independent.

The cleanest diagnostic: run a sweep of sigma_ce_scale ∈ {0.01, 0.05, 0.1, 0.2} at fixed K=90. If PPL improves with higher sigma_ce_scale, the sigma learning bottleneck is confirmed. If PPL is invariant, the bottleneck is the nonlinearity.
