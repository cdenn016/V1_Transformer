# Deep Mathematical Audit: Embedding, MahalanobisNorm, Observation Likelihood, Forward Pass, Training Loss, and Spectral Infrastructure

**Date:** 2026-04-11
**Scope:** `embeddings.py`, `blocks.py` (MahalanobisNorm), `model.py` (forward, output projection), `train.py` (compute_free_energy_loss), `analysis/` (spectral metrics)

---

## 1. Embedding Initialization and Geometry

**File:** `transformer/core/embeddings.py`

### 1a. mu_embed initialization

`mu_embed` is an `nn.Embedding(vocab_size, embed_dim)` initialized with `nn.init.normal_(weight, mean=0, std=init_std)` (line 231). Default `init_std = 2.0` (line 177).

**Assessment:** The comment at line 173 correctly explains the rationale: the old `1/sqrt(K)` initialization makes all pairwise distances `||mu_i - mu_j||^2` concentrate around `2K * (1/K) = 2`, giving near-uniform KL attention. The `init_std = 2.0` choice gives `||mu_i||^2 ~ K * 4`, so pairwise distances `||mu_i - mu_j||^2 ~ 8K`, creating genuine variance in KL scores. This is sound: it trades initial representational uniformity for sharper attention selectivity from step 0.

**No issues found.**

### 1b. log_sigma_diag initialization

Three branches:

- **gauge_fixed_priors:** `base_log_sigma_diag = Parameter(full(K, log(init_sigma_scale)))` (line 251-253). With default `init_sigma_scale = 1.0`, this gives `sigma = exp(0) = 1.0`.
- **learnable_sigma:** `log_sigma_diag = Parameter(full(V, K, log(init_sigma_scale)))` (line 264-266). Same `sigma = 1.0` per token per dimension.
- **frozen sigma:** `register_buffer('log_sigma_diag', full(K, log(init_sigma_scale)))` (line 269-271). Shared `sigma = 1.0` across all tokens.

The exp+clamp at lookup (lines 522-533) ensures `sigma_diag in [0.01, sigma_max]`.

**Assessment:** With `init_std = 2.0` and `sigma_init = 1.0`, the initial KL between two random tokens is:

```
KL(q_i || q_j) = 0.5 * [K + ||mu_i - mu_j||^2 / 1.0 - K + 0]
               = 0.5 * ||mu_i - mu_j||^2
               ~ 0.5 * 2K * (2.0)^2 = 4K
```

For K=90 this gives KL ~ 360, which when divided by `kappa * sqrt(d_h)` produces sharp softmax. This is consistent with the design intent.

**No issues found.** The nn.Parameter (not nn.Embedding) choice for `log_sigma_diag` is documented (lines 256-263) with a correct analysis of the AdamW interaction.

### 1c. phi_embed initialization

`phi_embed = nn.Embedding(vocab_size, phi_dim)` initialized as `normal_(0, phi_scale / sqrt(phi_dim))` (lines 311-315). Default `phi_scale = 0.3`.

**Assessment:** This gives `||phi||^2 ~ phi_dim * (phi_scale / sqrt(phi_dim))^2 = phi_scale^2 = 0.09`. So `||phi|| ~ 0.3` regardless of `phi_dim`. The transport operator `Omega = exp(phi * G)` is near identity: `Omega ~ I + 0.3 * G + O(0.045)`. This is the correct regime for early training -- large initial transport would scramble beliefs before the model learns meaningful representations.

**No issues found.**

### 1d. sigma_target (hyper-prior h)

Registered as a buffer at line 281-284:

```python
sigma_target_val = init_sigma_scale  # scalar
self.register_buffer('sigma_target', torch.full((embed_dim,), sigma_target_val))
```

**Assessment:** Correctly frozen (buffer, not parameter). The value is `init_sigma_scale = 1.0`, matching the initial sigma. The `_get_sigma_target` function in `train.py` (lines 201-253) retrieves this buffer and expands it to match `sigma_s` shape, with a fallback for old checkpoints. The buffer is never assigned `requires_grad`, so it is truly static.

**Verified: h is frozen as claimed in the VFE hierarchy.**

### 1e. Tied embeddings

Model.py line 310: `self.out_proj.weight = self.token_embed.mu_embed.weight`

This means `W_out = mu_embed` (shape `V x K`), so `logits[b,n,v] = mu_embed[v] . mu_q[b,n]`.

**Mathematical analysis:**

The output logit for token v is `z_v = mu_embed[v]^T * mu_q`. In the VFE framework, the observation likelihood is:

```
-E_q[log p(o=v | x)] = CE(z, target)
```

where `z_v = W_out[v] . mu_q`. With tied weights, `W_out[v] = mu_embed[v]`, so the logit is an inner product between the evolved belief and the prior mean of token v. This is the variational analog of a cosine similarity decoder.

**Gauge equivariance concern:** Under a global gauge transform `g`, mu_q transforms as `g * mu_q` but the embedding weights `mu_embed[v]` do NOT transform (they are fixed parameters). So `z_v = mu_embed[v]^T * (g * mu_q)` is NOT gauge-invariant unless `g = I` or unless the final LayerNorm/MahalanobisNorm absorbs the gauge dependence.

However, this is consistent with the architecture: the inverse cross-head permutation (line 1015/1284) restores the original dimension order before projection, and the final LayerNorm (line 944 in blocks.py, line 1281 in model.py) normalizes the scale. The gauge frame phi determines the INTERNAL representation; the output projection maps back to vocabulary space which is gauge-fixed by design. This is the standard gauge-fixing at the boundary -- analogous to how gauge theories fix gauge at asymptotic boundaries for observables.

**Assessment: Mathematically consistent.** The tied embedding creates a variational decoder `p(o=v|x) = softmax(mu_embed[v]^T * mu_q / T)` where `mu_q` is the gauge-fixed (post-LayerNorm) belief. No gauge equivariance violation.

### 1f. mu_normalize / mu_max_norm

Lines 584-591 in embeddings.py:

- `mu_normalize = True`: Projects to unit sphere `mu / ||mu||`.
- `mu_max_norm`: Clamps `||mu|| <= max_norm` via scaling.

**Assessment:** Applied AFTER positional embedding addition (line 579) and AFTER O(K) reflection (line 551). This means position information is preserved in direction but not norm. For KL-based attention where `KL ~ ||mu_i - mu_j||^2 / (2*sigma)`, unit-sphere normalization would make all KL values bounded by `2/sigma` (since `||mu_i - mu_j||^2 <= 4` on the unit sphere). This trades off representational capacity for stability. The `mu_max_norm` option is a softer version.

**No correctness issues.** Design trade-off is well-motivated.

### 1g. learnable_reflection (O(K) extension)

Lines 356-361: `sign_logit = nn.Embedding(V, K)` initialized to `+1`.

Forward (lines 547-551):
```python
z = self.sign_logit(token_ids)           # continuous latent
signs = z.sign()                          # hard {-1, +1}
signs = z + (signs - z).detach()          # STE: grad flows through z
mu = mu * signs                           # element-wise sign flip
```

**Assessment:** The straight-through estimator (STE) is correctly implemented: forward uses `sign(z)` (discrete), backward treats `sign` as identity. The initialization at `+1` means all tokens start at SO(K) (no reflection), and the model can learn to introduce reflections for specific tokens/dimensions.

The decode path (model.py lines 504-511) correctly applies the same sign vectors to `W_out`:
```python
W_signed = self.out_proj.weight * signs  # (V, K)
logits = mu_q @ W_signed.T
```

This ensures encode/decode consistency: if embedding flips dimension k for token v, the decode projection for v also flips dimension k.

**Verified correct.** The STE gradient `dL/dz = dL/d(sign(z)) * 1` is the standard choice for binary latent variables.

---

## 2. MahalanobisNorm

**File:** `transformer/core/blocks.py`, lines 75-148

### 2a. Norm computation

The class computes:

```
s^2 = mu^T * Sigma^{-1} * mu    (Mahalanobis quadratic form)
```

For diagonal sigma (line 129): `s^2 = sum_k(mu_k^2 / sigma_k)`
For full sigma (lines 134-143): `s^2 = mu^T * solve(Sigma, mu)` via `torch.linalg.solve`

Then normalizes: `mu_norm = mu * sqrt(K / (s^2 + eps))`

**Assessment:** This projects mu onto the surface `||mu||_M^2 = K` (constant Mahalanobis norm). The docstring's claim (line 94-96) is correct: after normalization, every token has the same Mahalanobis norm K, so the key-dependent bias in the KL attention formula cancels under softmax.

### 2b. Gauge equivariance

The docstring (lines 83-89) provides the proof sketch. Let me verify:

Under gauge transform `mu -> g*mu`, `Sigma -> g*Sigma*g^T`:

```
s^2 = (g*mu)^T * (g*Sigma*g^T)^{-1} * (g*mu)
     = mu^T * g^T * g^{-T} * Sigma^{-1} * g^{-1} * g * mu
     = mu^T * Sigma^{-1} * mu
```

So `s^2` is gauge-invariant (scalar). The scaling factor `sqrt(K/s^2)` is therefore gauge-invariant. The output `mu_norm = sqrt(K/s^2) * mu` transforms as `g * mu_norm = sqrt(K/s^2) * g * mu`, which is a vector. So MahalanobisNorm commutes with gauge transport.

**Verified: Gauge equivariance holds.**

### 2c. Application point

In `blocks.py` line 505, MahalanobisNorm is applied as a pre-norm:
```python
mu_normalized = self.norm1(mu_q, sigma_q) if isinstance(self.norm1, MahalanobisNorm) else self.norm1(mu_q)
```

This happens BEFORE attention and gauge transport. Since gauge equivariance holds, the ordering is mathematically correct.

### 2d. Isotropic limit

The docstring (lines 97-99) claims: when `Sigma = sigma^2 * I`, `s^2 = ||mu||^2 / sigma^2`, so `mu_norm = mu * sigma * sqrt(K / ||mu||^2) = sigma * mu / RMS(mu) * sqrt(K)`. This is standard RMSNorm scaled by sigma, which is correct.

**No issues found.** MahalanobisNorm is mathematically sound and gauge-equivariant.

---

## 3. Output Projection and Observation Likelihood

**File:** `transformer/core/model.py`

### 3a. Variational bound correctness

The observation term in the VFE is `-E_q[log p(o|x)]`. For categorical observations with softmax likelihood:

```
p(o=v | x) = softmax(W_out * x)_v
```

The exact variational bound would be:

```
-E_q[log p(o|x)] = -integral N(x; mu_q, Sigma_q) * log(softmax(W*x)_v) dx
```

This integral is intractable for non-trivial Sigma_q. The code uses the mean-field approximation:

```
-E_q[log p(o|x)] ~ -log p(o | x=mu_q) = CE(W_out * mu_q, target)
```

This is the zeroth-order (Laplace) approximation, NOT the full variational bound. The first-order correction would add a variance-dependent term:

```
~ CE(W*mu_q, target) + 0.5 * tr(Sigma_q * H)
```

where H is the Hessian of the log-likelihood. The code does not include this correction.

**Assessment:** The zeroth-order approximation is standard practice in variational autoencoders and variational inference. The missing variance term means the model has no direct gradient pressure from the observation likelihood to shrink Sigma_q (the KL terms provide indirect pressure). This is acceptable if the KL regularization is sufficient to prevent Sigma_q collapse/explosion, which the hyper-prior and self-coupling terms handle.

**Recommendation:** Document this as a Laplace approximation. If future work adds the Hessian correction, it would provide an additional learning signal for sigma evolution.

### 3b. CE loss sign

In `train.py` line 360-365:
```python
ce_loss_raw = F.cross_entropy(logits, targets, reduction='mean')
```

Total loss (line 590):
```python
total_loss = ce_loss + belief_align_loss + self_consistency_loss + ...
```

Since `F_VFE = KL + ... - E[log p(o|x)]`, and `CE = -E[log p(o|x)]`, the sign is:
```
F_VFE = CE + alpha * KL(q||p) + ...
```

All terms are positive (KL >= 0, CE >= 0), and we minimize F_VFE. This is correct.

**Verified: Signs are consistent.**

### 3c. Variance-dependent observation terms

No variance-dependent terms from the observation model are included (as discussed in 3a). The `sigma_ce_scale` parameter (in PriorBank) provides a residual CE-to-sigma gradient but this is a different mechanism (it controls how sigma affects the decode path, not an observation-model correction).

---

## 4. Model Forward Pass Data Flow Audit

**File:** `transformer/core/model.py`

### Complete data flow:

```
token_ids
  -> GaugeTokenEmbedding.forward(token_ids)
     -> mu_embed(token_ids)         -> mu       (B, N, K)
     -> exp(log_sigma_diag[...])    -> sigma    (B, N, K) or (B, N, K, K)
     -> phi_embed(token_ids)        -> phi      (B, N, phi_dim)
     -> [optional: sign_logit STE]  -> mu = mu * signs
     -> [optional: positional_embed] -> mu = mu + pos_embed
     -> [optional: mu_normalize]
  -> _apply_cross_head_perm(mu, sigma)  -> permuted dims
  -> mu_prior = mu.clone()
  -> sigma_prior = sigma.clone()
  -> phi = pos_encoding.compose(phi, ...)
  -> mask = causal_mask[:N, :N]
  -> GaugeTransformerStack.forward(mu, sigma, phi, generators, ...)
     -> for each block:
        -> norm1(mu)  [LayerNorm/RMSNorm/MahalanobisNorm]
        -> IrrepMultiHeadAttention(mu_norm, sigma, phi, generators)
           -> KL computation, softmax(beta), weighted aggregation
        -> residual: mu = mu + mu_attn  (or mu = mu + (mu_attn - mu_norm) for 'delta')
        -> sigma = sigma_attn.clamp(...)
        -> norm2(mu)
        -> VariationalFFNDynamic(mu_norm, sigma, phi, mu_prior, ...)
           -> E-step iterations: natural gradient descent on VFE
        -> residual: mu = mu + mu_ffn (or delta variant)
        -> sigma = sigma_ffn.clamp(...)
        -> [hierarchical: mu_prior = mu for next layer]
     -> final_norm(mu)
  -> _apply_cross_head_perm(mu, sigma, inverse=True)
  -> out_proj(mu) -> logits  (or PriorBank.decode, or sign-corrected W_signed)
```

### 4a. Prior setting

`mu_prior = mu_q.clone()` (line 422) and `sigma_prior = sigma_q.clone()` (line 423) are set from the embedding output BEFORE positional encoding. This means priors represent pure token semantics without position information.

**Assessment:** Correct. The E-step VFE term `alpha * KL(q||p)` pulls the evolved belief back toward the token-specific prior, providing a semantic anchor independent of position. Position information enters through phi (gauge transport) and optionally through positional mu-embeddings.

### 4b. Residual connection

Default `residual_type = 'additive'`: `mu = mu + mu_attn` (line 651) and `mu = mu + mu_ffn` (line 757).

The mu_attn from attention is the aggregated transported belief `sum_j beta_ij * Omega_ij * mu_j`. This is NOT zero-centered -- it's a full belief, not a correction. Adding it to mu_q gives `mu_new = mu_q + sum_j beta_ij * Omega_ij * mu_j`.

When self-attention dominates (beta_ii >> beta_ij for j != i) and Omega_ii = I, mu_attn ~ mu_normalized ~ mu_q / RMS(mu_q), so the residual adds a scaled copy of the input. This is analogous to the standard transformer where the attention output contains a copy of the query via the identity component of W_V.

The `delta` variant extracts `mu_attn - mu_normalized` to avoid this doubling. The comments (lines 632-651) correctly identify the trade-off: delta is mathematically cleaner for deep unnormalized stacks but empirically worse for single-layer LayerNorm'd configs.

**No correctness issues.** Design trade-off is well-analyzed.

### 4c. LayerNorm applies to mu only

Confirmed at lines 505, 672-675 in blocks.py: normalization is applied to `mu_q` only. Sigma and phi are NOT normalized. Sigma passes through the block and is updated by the attention aggregation and FFN E-step. Phi passes through and is updated by the FFN E-step (if `evolve_phi_e_step = True`).

**Verified: Correct separation.** LayerNorm on sigma would destroy the SPD structure. LayerNorm on phi would destroy the Lie algebra element interpretation.

### 4d. Sigma and phi propagation

Sigma: Updated in attention (line 664, replacement: `sigma_q = sigma_attn.clamp(...)`) and FFN (line 704-705, same replacement). Clamped to `[1e-4, sigma_max]`.

Phi: Updated by the FFN E-step via `phi_out` (line 680-695 -> `phi_out` from `self.ffn(...)`, then returned at line 762). The attention sublayer does NOT update phi -- only the E-step does.

**Verified correct.**

### 4e. Cross-head permutation

Applied BEFORE the transformer stack (line 418) and inverted AFTER (lines 1015, 1284). The permutation reorders K dimensions to make coupled head blocks contiguous. The output projection operates in the ORIGINAL (un-permuted) basis, so the inverse permutation is necessary.

The `forward_with_attention` path also inverts correctly (line 1284).

**Verified correct.**

---

## 5. Training Loss Consistency

**File:** `transformer/train.py`, `compute_free_energy_loss()`

### 5a. M-step loss composition

```
L_total = CE(W_out * mu_q, targets)                    # Observation (-E[log p(o|x)])
        + M_alpha * mean(KL(q_i || p_i)) / sqrt(K)     # Self-coupling
        + M_beta * mean(sum_ij beta_ij * KL_ij) / sqrt(K)  # Belief coupling
        + lambda_gamma * mean(sum_ij gamma_ij * KL_model_ij) / sqrt(K)  # Model coupling
        + lambda_hyper * mean(KL(s_i || h)) / sqrt(K)   # Hyper-prior
        + mass_phi/2 * mean(||phi_i||^2) / sqrt(K)      # Gauge prior
        + aux_loss                                       # Per-layer CE
        + holonomy_loss                                  # Non-flat transport
```

All KL terms are divided by `sqrt(K)` (dim_scale). CE is NOT divided by `sqrt(K)` by default (optional via `normalize_ce_by_dim`).

**Assessment:** The `sqrt(K)` scaling is a heuristic to make VFE term magnitudes independent of embedding dimension. Since each KL component contributes O(K) (sum over K dimensions), dividing by `sqrt(K)` gives O(sqrt(K)) per term. CE also scales roughly with log(V) which doesn't scale with K. The asymmetry between CE and KL scaling is acknowledged (line 367-373) and optionally correctable.

**Potential concern:** The dimension scaling should be consistent. If KL terms are divided by `sqrt(K)` but CE is not, then increasing K weakens the KL regularization relative to CE. This might explain why the model needs careful tuning of `M_alpha` across different K values. The `normalize_ce_by_dim` option addresses this.

### 5b. Beta detachment in M-step

Line 386: `beta_final = beta[-1].detach() if detach_beta_m_step else beta[-1]`

**Assessment:** Correct EM formulation. The M-step optimizes theta (model parameters) with the E-step posteriors held fixed. Beta is derived from the E-step (it's the attention weight = variational coupling posterior), so it should be detached. The comments at lines 379-385 correctly cite the envelope theorem: at E-step convergence, `dF/d(beta) = 0`, so the beta-gradient vanishes and detaching is the finite-iteration approximation.

Gamma is also detached (line 484): `gamma.detach() * kl_model`. Correct.

**Verified: Proper EM separation.**

### 5c. E-step vs M-step consistency

The E-step (inside VariationalFFNDynamic) minimizes:

```
F_E = alpha * KL(q||p) + sum_j beta_ij * KL(q_i || Omega_ij * q_j)
```

with respect to q (mu_q, sigma_q). The M-step loss includes:

```
L_M = CE + M_alpha * KL(q*||p) + M_beta * sum_ij beta* * KL(q*_i || Omega_ij * q*_j) + ...
```

where q* and beta* are the converged E-step values (detached). These are the same functional forms with the same detach pattern (q* is the E-step output with gradient flowing through the residual/implicit-EM path; beta* is detached).

**Verified: Consistent.**

### 5d. Dimension scaling

CE: `mean` reduction over `(B*N)` positions, producing a scalar in nats. Typical range: 3-6 nats for WikiText.

KL terms: `kl_per_agent` is `(B, N)`, then `.mean()` and divided by `sqrt(K)`. For K=90, sqrt(K) ~ 9.5. If mean KL ~ 10 per agent, the scaled contribution is ~ 1.05.

With default `M_alpha = 0.0` and `M_beta = 1.0`, the only non-CE loss is the belief alignment `M_beta * scaled_belief_kl`. This makes the training loss ~ CE + belief_alignment.

**Assessment:** The relative scaling appears reasonable. The `sqrt(K)` divisor ensures KL terms don't overwhelm CE for large K. However, users must tune `M_alpha` and `M_beta` relative to CE magnitude. The defaults (`M_alpha = 0.0`, `M_beta = 1.0`) effectively disable self-coupling in the M-step, relying entirely on the E-step for prior-belief coupling.

---

## 6. Spectral Analysis Infrastructure

### 6a. Gauge frame spectrum

**File:** `transformer/analysis/gauge_geometry.py`, lines 612-618

Computes eigenvalues of `exp(phi_i * G)` for each token:

```python
phi_matrix = build_phi_matrix(phi, generators)     # (B, N, K, K)
exp_phi = stable_matrix_exp_pair(phi_matrix)        # (B*N, K, K)
eigvals = torch.linalg.eigvals(exp_phi)             # (B*N, K) complex
spectrum = eigvals.abs()                             # modulus
```

**Assessment:** The eigenvalue moduli of `exp(phi*G)` are gauge-INVARIANT under conjugation: if `phi -> g*phi*g^{-1}`, then `exp(phi*G) -> g*exp(phi*G)*g^{-1}`, and conjugation preserves eigenvalues. This is the correct gauge-invariant spectral observable.

For SO(K): all eigenvalues have `|lambda| = 1` (unitary). Deviations measure numerical error.
For GL(K): eigenvalue moduli reflect the scaling/shearing components of the gauge frame. Spread in moduli indicates anisotropic gauge transport.

### 6b. Effective rank of belief embeddings

**File:** `transformer/analysis/semantics.py`, lines 1656-1658

```python
evals_norm = evals / evals.sum(dim=-1, keepdim=True)
entropy = -(evals_norm * torch.log(evals_norm)).sum(dim=-1)
eff_rank = torch.exp(entropy)
```

This is the standard effective rank via Shannon entropy of normalized eigenvalues. For a K-dimensional uniform spectrum, `eff_rank = K`. For a rank-1 spectrum, `eff_rank = 1`.

**Assessment:** This measures representational capacity of the covariance structure. If all sigma values collapse to identical values, `eff_rank = K` (uniform, not informative). If sigma develops strong anisotropy (some dimensions much more uncertain than others), `eff_rank < K` (the belief concentrates in a low-dimensional subspace).

This metric does verify theoretical predictions about symmetry breaking: as training progresses and phi learns to break the initial SO(K) symmetry, the sigma structure should develop preferred directions (lower effective rank in the gauge-transported frame).

### 6c. Attention distribution entropy

**File:** `transformer/train.py`, lines 623-624

```python
beta_safe = beta_avg.clamp(min=1e-10)
attn_entropy = -(beta_safe * beta_safe.log()).sum(dim=-1).mean()
```

Standard Shannon entropy of the attention distribution. High entropy = uniform attention (each token attends equally to all others). Low entropy = concentrated attention (each token attends to a few specific others).

**Assessment:** Correctly computed. Tracked as a training metric at line 642. This verifies the theoretical prediction that initial training should transition from near-uniform attention (high entropy) to structured attention patterns (lower entropy) as KL-based selectivity develops.

### 6d. Holonomy spectral gap

**File:** `transformer/analysis/holonomy_metrics.py`, lines 133-144

Computes SVD of holonomy matrices `C_ijk` and tracks the gap between the two largest singular values. For flat connections, `C_ijk = I`, so all singular values are 1 and the gap is 0.

**Assessment:** This is a correct diagnostic for curvature. Non-zero spectral gap indicates genuine non-flat transport (the connection has non-trivial holonomy). The metric is tracked per training step.

### 6e. Missing spectral analyses

The codebase does NOT compute:
- Singular values of the attention weight matrix `beta` (to detect low-rank attention patterns)
- Eigenvalue distribution of `mu_embed^T * mu_embed` (the Gram matrix of embeddings, detecting collapse)
- Spectral norm of the Jacobian `dmu_out/dmu_in` through each block (to detect gradient pathology)

These could be valuable additions for verifying theoretical predictions but their absence is not a correctness issue.

---

## BUG REPORT

### Bug 1: MahalanobisNorm not passed sigma in forward_with_attention

**File:** `transformer/core/model.py`

**Lines:** 1163, 1232, 1281

In `forward_with_attention()`, three calls to `self.transformer.final_norm(mu_q)` do NOT pass sigma_q, even when the norm is MahalanobisNorm:

```python
# Line 1163 (aux layer loss):
_aux_mu = self.transformer.final_norm(mu_q)

# Line 1232 (per-layer CE probe):
_probe_mu = self.transformer.final_norm(mu_q.detach())

# Line 1281 (main output normalization):
mu_q = self.transformer.final_norm(mu_q)
```

Compare with the correct handling in `GaugeTransformerStack.forward()` at line 944:
```python
mu_q = self.final_norm(mu_q, sigma_q) if isinstance(self.final_norm, MahalanobisNorm) else self.final_norm(mu_q)
```

**Impact:** When `norm_type = 'mahalnorm'`, the `forward_with_attention()` path falls back to RMSNorm (the MahalanobisNorm.forward() sigma=None branch, line 122-124), losing the covariance-aware normalization. This means:
1. The main training path (`forward_with_attention` is used during training) uses RMSNorm instead of MahalanobisNorm for the final normalization.
2. The inference path (`forward()` -> `GaugeTransformerStack.forward()`) correctly uses MahalanobisNorm.
3. Train/inference behavior mismatch.

**Severity:** Medium-High when `mahalnorm` is active. Low if `layernorm` or `rmsnorm` is used (they ignore sigma anyway).

**Fix:** Replace the three offending lines with:
```python
if isinstance(self.transformer.final_norm, MahalanobisNorm):
    result = self.transformer.final_norm(mu_q, sigma_q)
else:
    result = self.transformer.final_norm(mu_q)
```

### Bug 2 (Minor): Aux layer loss and probe loss use un-MahalanobisNorm'd mu

Lines 1163 and 1232 have the same issue but for diagnostic purposes. The aux layer CE loss (used in training when `aux_layer_loss = True`) computes logits from improperly normalized mu, which would give a biased training signal for non-final layers.

---

## Summary of Findings

| Area | Status | Notes |
|------|--------|-------|
| mu_embed init | OK | init_std=2.0 gives good KL variance |
| log_sigma_diag init | OK | sigma=1.0 with proper parameterization |
| phi_embed init | OK | ||phi||~phi_scale independent of phi_dim |
| sigma_target (h) | OK | Frozen buffer, never learned |
| Tied embeddings | OK | Gauge-fixed at boundary, mathematically consistent |
| mu_normalize | OK | Design trade-off, no correctness issue |
| learnable_reflection | OK | STE correct, encode/decode consistent |
| MahalanobisNorm formula | OK | Correct Mahalanobis quadratic form |
| MahalanobisNorm equivariance | OK | Proven gauge-invariant |
| Observation likelihood | NOTE | Laplace approximation (zeroth-order), no variance term |
| CE loss sign | OK | F = CE + KL + ..., all positive, minimized |
| Forward data flow | OK | Priors set before pos-encoding, residuals consistent |
| LayerNorm on mu only | OK | Correct separation |
| Cross-head permutation | OK | Correctly applied and inverted |
| Beta detachment | OK | Correct EM: E-step quantities fixed in M-step |
| Dimension scaling | NOTE | sqrt(K) heuristic, CE/KL asymmetry acknowledged |
| Gauge frame spectrum | OK | Eigenvalue moduli are gauge-invariant |
| Effective rank | OK | Standard Shannon entropy formulation |
| Attention entropy | OK | Correctly computed |
| MahalanobisNorm in forward_with_attention | **BUG** | Missing sigma passthrough |
