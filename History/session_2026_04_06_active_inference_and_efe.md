# Active Inference, Bootstrap Self-Distillation, and the E-Step Refactor

**Session dates:** 2026-04-06 to 2026-04-07
**Codebase:** Gauge-Theoretic VFE Transformer (V13)
**Scope:** Three numerical/correctness fixes, three experimental features, one principled active-inference extension to the variational E-step, one bootstrap self-distillation term, and a module-level refactor that lifts the entire active-inference path out of `variational_ffn.py` into a dedicated file. All changes are backwards compatible and gated behind explicit configuration flags.

## Abstract

This report documents a two-day working session that touched ten files in `transformer/core/` and produced one substantial theoretical extension to the variational free energy E-step, a second experimental E-step term inspired by non-contrastive self-distillation from vision, and a refactor that isolates both of those extensions into a dedicated `active_inference.py` module so that the principal VFE file no longer carries their implementation weight. The session opened with three correctness fixes discovered during diagnostic work on an underperforming 2-layer configuration: an additive sigma residual that had pegged the posterior covariance at its ceiling within a few forward passes, a missing chain-rule term in the fused RoPE attention gradient that produced wrong rather than merely suboptimal gradients on coupled-position belief updates, and a three-way default inconsistency on the `hierarchical_priors` flag across its dataclass declaration, its `from_config` deserializer, and the runtime `getattr` fallback. The conceptual centerpiece is the active-inference extension: the existing KL attention mechanism is shown to be structurally identical to Boltzmann policy selection over the complexity term of expected free energy, and the new code adds the missing pragmatic and epistemic halves so that the E-step minimizes the full EFE rather than only the complexity component. After an initial diagnostic showed that the EFE gradient produced no measurable effect at fresh initialization due to trust-region clipping, the apply-update logic was restructured to use an Euclidean gradient with a separate trust-region budget, and three audit findings were promoted to loud runtime warnings that fire at model construction when the active-inference path would silently do nothing. A separate thread introduced bootstrap self-distillation as a third E-step term, designed around the observation that the pragmatic entropy term is self-reinforcing on its own and requires the epistemic counterweight, and that a cross-entropy against a stop-gradient target has a data-dependent fixed point that neither of the first two terms possesses. The design went through one reviewer correction after the initial curvature argument conflated per-component with $\ell_2$ norms, and the corrected analysis (verified symbolically with SymPy) shows that both gradients have the same $O(\sigma_\ell / \sqrt{V})$ order at random initialization, with the distinction between them lying in fixed-point structure rather than magnitude. The active-inference path was then extracted into `transformer/core/active_inference.py` as a standalone module, shrinking `variational_ffn.py` by approximately 480 lines and producing bit-exact backward-compatible behaviour against the pre-refactor baseline. All 471 non-slow non-GPU tests continue to pass, both SymPy verification scripts run cleanly, and the refactored path has been verified in all four execution contexts: training, validation under `torch.no_grad`, decoding under `torch.inference_mode`, and training resumed after a generation interlude.

## 1. Background and Motivation

The session opened with diagnostics on training run 88, a 2-layer K=60 GL(10) configuration that was underperforming a matched 1-layer baseline. Inspection of the per-step metrics CSV revealed three pathologies: the diagonal posterior covariance $\sigma_q$ was pegged at the configured ceiling of 12.0 within the first hundred steps, the prior–belief KL was growing unboundedly from roughly 273 nats at initialization to 858 nats after a few thousand steps, and the learnable $\alpha$ coefficient (the Bayesian precision on the self-coupling term) was collapsing toward zero across both layers. None of these were obviously bugs; they could plausibly have reflected model misspecification or training-rate mismatch. The first half of the session was spent isolating which were genuine code defects and which were emergent behavior of a working system. Two of the three turned out to be code defects that had been masked at single-layer depth where the per-sublayer compounding was below the ceiling within a single forward pass.

The remainder of the session followed a sequence of theoretical questions raised by the user about whether various mechanical pieces of the model could be reformulated more principled. Each question was answered substantively (pushing back when the proposed change was not actually theoretically sound), and when the answer pointed at a principled implementation, the implementation was carried out and tested. The two most consequential threads — active inference as a closed-form prescription for what the E-step is missing, and bootstrap self-distillation as a cross-position coupling in prediction space — became the largest single pieces of work and are documented in detail in §4 and §5.

## 2. Numerical and Correctness Fixes

### 2.1 Sigma Residual Inflation

In the diagnostics from run 88, the posterior covariance $\sigma_q$ at every position rose to its configured ceiling `sigma_max = 12.0` within the first few forward passes and remained pinned there for the remainder of training. This had the immediate consequence of saturating the natural-gradient projection (which scales by $\sigma^2$) and the indirect consequence of making the prior coupling term degenerate, since the KL between a wide posterior and the comparatively tight prior is dominated by the trace term $\mathrm{tr}(\sigma_p^{-1} \sigma_q)$.

The block-level sublayer aggregation in `blocks.py` was using an additive update for the covariance even though the sub-layer functions return the *full* updated covariance, not a delta. Both the attention sublayer and the FFN sublayer were doing

$$\sigma_q \leftarrow \sigma_q + \sigma_{\text{sub}}$$

where `sigma_sub` was the *new* posterior covariance produced by the sublayer, not the *change* in the covariance. This double-counted absolute scale at every sublayer of every layer. With $L$ layers and 2 sublayers per layer, the multiplicative inflation factor was $2^{2L}$ at the worst case, which is why a 1-layer model trained correctly while the 2-layer model saturated the ceiling almost immediately. The misnamed `sigma_residual` flag in `BlockConfig` was supposed to be the gate for this behavior; in practice the additive path ran regardless of the flag.

The two sublayers in `blocks.py` were both changed from additive update to delta extraction by replacement:

```python
if self.evolve_sigma and sigma_attn is not None:
    sigma_q = sigma_attn.clamp(min=1e-4, max=self.sigma_max)
# ... after FFN sublayer
if self.evolve_sigma and sigma_ffn is not None:
    sigma_q = sigma_ffn.clamp(min=1e-4, max=self.sigma_max)
```

The mean $\mu$ has its own residual connection through `mu_q = mu_q + (mu_ffn - mu_normalized)` because the FFN receives a normalized input and returns a state in the same normalized frame; the covariance has no normalization step (RMSNorm acts on means, not covariances), so the cleanest semantics is direct replacement. The mirror change was made in `model.py:forward_with_attention` where the same code is duplicated for attention recording. The legacy `sigma_residual` flag was retained in `BlockConfig` as a deprecated no-op so that any saved configs continue to load. With the fix in place, $\sigma_q$ under the same training config drifts upward only modestly across layers and remains well below the ceiling, allowing the natural-gradient projection to operate in its design regime.

### 2.2 RoPE Chain Rule in the Fused Gradient Path

When the gauge transformer is configured with `use_rope=True`, the attention sublayer applies a rotary positional rotation to $\mu$ before computing the inter-position KL. The exact attention forward path was correct, but the *fused* attention-and-VFE-gradient path in `vfe_gradients.py` (the optimized routine that computes both $\beta$ and the belief gradient in a single pass over the block-diagonal $\Omega$ construction) was missing a chain-rule factor. The bug was silent — gradients were finite, training did not crash, and loss did decrease — but the gradient propagated to $\mu$ through the coupling term $\beta \cdot \mathrm{KL}$ was systematically wrong by a position-dependent rotation.

When RoPE is enabled, the attention-side KL is computed in the rotated frame:

$$\mathrm{KL}_{\text{rope}}(q_i \| \Omega_{ij} q_j) = \mathrm{KL}(N(R(\theta_i)\mu_i, \Sigma_i) \| N(\Omega_{ij} R(\theta_j) \mu_j, \Omega_{ij} \Sigma_j \Omega_{ij}^\top))$$

where $R(\theta_i)$ is the block-diagonal $\mathrm{SO}(2)^{d_k/2}$ rotation associated with position $i$. The attention weights are then

$$\beta_{ij} = \mathrm{softmax}_j\left(-\frac{\mathrm{KL}_{\text{rope}}(q_i \| \Omega_{ij} q_j)}{\kappa}\right)$$

The VFE gradient with respect to $\mu$ contains a softmax-coupling term

$$\frac{\partial F}{\partial \mu_i} \supset \sum_j \frac{\partial \beta_{ij}}{\partial \mu_i} \cdot \mathrm{KL}_{\text{rope}}(q_i \| \Omega_{ij} q_j) \cdot \nabla_{\mu_i}\mathrm{KL}_{\text{rope}}$$

which by chain rule requires the gradient of the rotated KL with respect to the *raw* (un-rotated) $\mu$:

$$\nabla_{\mu_i^{\text{raw}}} \mathrm{KL}_{\text{rope}} = R(\theta_i)^\top \cdot \nabla_{R(\theta_i)\mu_i} \mathrm{KL}_{\text{rope}}$$

The fused path was computing the inner gradient (the derivative of the KL with respect to the rotated $\mu$) but skipping the outer $R(\theta_i)^\top$ multiplication, so the gradient that went into the coupling term was the gradient *as if $\mu$ were already in the rotated frame*. For positions where $R(\theta_i)$ is far from identity, this is a substantial error in the coupling direction.

A new helper `_un_apply_rope_pair_outer(grad, base)` was added to `transport_ops.py`:

```python
def _un_apply_rope_pair_outer(grad: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    B, N_i, N_j, K = grad.shape
    half_K = K // 2
    cos_angles, sin_angles = _get_rope_cos_sin(K, N_i, base, grad.device, grad.dtype)
    cos_b = cos_angles[None, :, None, :]
    sin_b = sin_angles[None, :, None, :]
    g_even = grad[:, :, :, :2*half_K:2]
    g_odd  = grad[:, :, :, 1:2*half_K:2]
    out = grad.clone()
    out[:, :, :, :2*half_K:2] = g_even * cos_b + g_odd * sin_b
    out[:, :, :, 1:2*half_K:2] = -g_even * sin_b + g_odd * cos_b
    return out
```

This applies $R(\theta)^\top$ to a per-pair gradient tensor of shape `(B, N_i, N_j, K)`. Inside `_fused_attention_and_vfe_gradients_block_diag` and `_compute_vfe_gradients_block_diagonal_diag`, the per-block gradient delta `grad_kl_rope_block = delta_mu_kl / sigma_j_transported` is accumulated into a `grad_kl_rope_per_pair` tensor, and after the block loop completes the un-rotation is applied:

```python
if use_rope:
    grad_kl_for_coupling = _un_apply_rope_pair_outer(grad_kl_rope_per_pair, base=rope_base)
else:
    grad_kl_for_coupling = grad_kl_rope_per_pair
```

This corrected gradient is then used in the `d_beta_d_mu` accumulator that builds the softmax coupling term. The fix was validated against PyTorch autograd at machine precision. With manual computation of the chain rule, the relative error against autograd dropped to 2.22e-16 (the float32 epsilon). The fused path now matches autograd to 5e-8 across the full forward and backward, confirming both that the un-rotation is correct and that no other terms in the fused path are missing the same correction.

### 2.3 Hierarchical Priors Default Inconsistency

A separate audit pass later in the session uncovered a three-way inconsistency in the `hierarchical_priors` flag. The `BlockConfig` dataclass declared the default as `True`. The `BlockConfig.from_config` deserializer used `config.get('hierarchical_priors', False)`, defaulting to `False` when the key was absent from the config dict. The runtime fallback in `blocks.py` and `model.py` used `getattr(cfg, 'hierarchical_priors', False)`, also defaulting to `False`. The practical consequence was that a user editing the dataclass default saw their change take effect only on explicitly-constructed `BlockConfig` instances, while configs deserialized from dicts and runtime fallbacks on partially-populated configs both silently produced the opposite behavior. All three paths were unified to `True`, matching the dataclass declaration and making the hierarchical-priors path the default unless explicitly disabled.

## 3. Smaller Feature Additions

### 3.1 Hierarchical Priors

The `BlockConfig` flag `hierarchical_priors` (default now True, consistent across all three defaulting paths after the §2.3 fix) makes each layer's posterior $\mu$ become the next layer's prior $\mu$. The covariance prior `sigma_prior` continues to come from the embedding bank to prevent uncertainty cascade across layers — only the *mean* is hierarchically passed. The motivation is that without this, every layer pulls its posterior toward the same fixed embedding-bank prior, which makes deep layers indistinguishable from shallow layers in the prior coupling term and removes one of the main sources of depth-driven representation refinement. The change was wired through `block_config.py`, the forward loop in `blocks.py`, and the parallel `forward_with_attention` path in `model.py`. A subtle gradient-checkpointing fix was needed because the closure that captures `mu_prior` was binding by name rather than by value across the layer loop; the standard Python idiom of binding via default arguments was used:

```python
def create_block_fn(blk, _mu_prior=mu_prior, _omega=omega):
    def _fn(mu, sigma, phi):
        return blk(mu, sigma, phi, blk.generators, mask=mask, mu_prior=_mu_prior, omega=_omega)
    return _fn
```

without this default-argument trick the checkpointed closure would always see the *last* layer's `mu_prior` value rather than the per-layer value at the time the closure was created, producing silently wrong gradients.

### 3.2 Experimental: Full Gauge RoPE

The standard RoPE pattern in transformer literature applies the rotation only to the means used as queries and keys, leaving the value path (and in our case the covariance) unrotated. This is asymmetric in the gauge framework: if RoPE is interpreted as a position-dependent gauge frame in $\mathrm{SO}(2)^{d_k/2} \subset \mathrm{GL}(K)$, then the natural action on a Gaussian belief should act on both the mean and the covariance via the standard sandwich product:

$$\mu_i \mapsto R(\theta_i) \mu_i, \qquad \Sigma_i \mapsto R(\theta_i) \Sigma_i R(\theta_i)^\top$$

The asymmetric ($\mu$-only) form is what almost every published RoPE implementation does, including the standard one we inherited. The "full gauge" form has not, to our knowledge, been tried in published work, and is the framework-consistent choice given the gauge-theoretic interpretation in the GL(K) manuscript. We added an experimental `rope_full_gauge: bool = False` flag and a new gradient routine `_compute_rope_full_gauge_gradient_per_head` that lifts diagonal $\Sigma$ to a full covariance, applies the sandwich rotation in the KL computation, and computes the resulting belief gradient via `torch.autograd.grad` rather than analytically. This is slower than the analytic fused path and is not currently the default, but it provides an experimental ablation point for testing whether the framework-consistent rotation produces measurable differences in training dynamics.

Two execution-context bugs were found and fixed in this routine after launch. The first was a `RuntimeError` during validation: validation runs under `torch.no_grad()`, and `torch.autograd.grad` cannot be called on a tensor that does not require grad. The fix was to wrap the entire body of the routine in `with torch.enable_grad():`, which forces autograd back on locally for the duration of the call. The second was a related but stricter problem during text generation: `model.generate` uses `@torch.inference_mode()`, which is not the same as `torch.no_grad()`. Inference mode marks tensors as inference tensors that *cannot* be promoted to require_grad even inside a nested `enable_grad()` context. The fix was to detect inference mode at the dispatch site in `variational_ffn.py` and fall back to the standard analytic fused path:

```python
_use_rope_full = (
    getattr(self, '_rope_full_gauge_vfe', False)
    and self._use_rope_vfe
    and is_diagonal
    and _nonflat_omega is None
    and not torch.is_inference_mode_enabled()
)
```

The fallback path is correct (it has the chain-rule fix from §2.2) so the model still produces correct samples during decoding, just without the experimental full-gauge variant.

A third issue was discovered later in the session during the multi-agent audit pass: the cached `exp_phi_h` tensors in the full-gauge path carry the autograd graph of $\phi$, and `torch.autograd.grad(..., retain_graph=False)` inside the helper would free saved tensors that the outer M-step backward then expected to find, causing a cryptic "trying to backward through the graph a second time" error. The fix was to explicitly detach the cached block exponential pairs before using them in the full-gauge gradient computation, so that the autograd graph for the EFE gradient is strictly local and does not share state with the VFE gradient graph.

## 4. Active Inference Extension to the E-Step

This section describes the conceptual centerpiece of the session. Subsections §4.1 through §4.3 develop the theory, §4.4 describes the implementation, §4.5 addresses the execution-context issues, §4.6 reports empirical sanity checks, and §4.7 describes the restructuring of the trust-region logic that was required after an initial diagnostic showed that the EFE gradient was being crushed by the VFE update budget.

### 4.1 What was already there: KL attention as Boltzmann policy selection

The variational free energy minimized by the existing E-step contains the standard self-coupling and belief-coupling terms

$$F[q] = \alpha \sum_i \mathrm{KL}(q_i \| p_i) + \lambda \sum_{i,j} \beta_{ij} \mathrm{KL}(q_i \| \Omega_{ij} q_j) + (\text{minor terms})$$

where the attention weights themselves are computed from the same coupling KL via a softmax with temperature $\kappa$:

$$\beta_{ij} = \mathrm{softmax}_j\left(-\frac{\mathrm{KL}(q_i \| \Omega_{ij} q_j)}{\kappa}\right)$$

Compare this to the action-selection rule in active inference. An active-inference agent has beliefs $q(s)$ over hidden states, a generative model $p(o, s | \pi)$ parameterized by policies $\pi$, and selects actions by softmax of the negative expected free energy $G(\pi)$:

$$\pi^* \sim \mathrm{softmax}(-G(\pi)/\tau)$$

The expected free energy decomposes as

$$G(\pi) = \underbrace{\mathrm{KL}(q(s|\pi) \| p(s))}_{\text{complexity}} + \underbrace{(-\mathbb{E}_q[\log p(o|s,\pi)])}_{\text{pragmatic value}}$$

The KL attention rule is literally the action-selection rule with the policy space being "which other position to attend to" and the cost being the complexity term $\mathrm{KL}(q_i \| \Omega_{ij} q_j)$. The temperature $\kappa$ is the action precision. This is not an analogy: the equations match and the variable names map onto each other one-for-one. What is genuinely missing from the existing implementation is the pragmatic value term — the expected log-likelihood of the observation under the policy. In language modeling at inference time the true observation (the next token) is not available, which is precisely the situation that active inference is designed for: when the agent does not know the observation, it substitutes its own predictive distribution.

### 4.2 What was missing: pragmatic and epistemic value

Active inference, properly stated, requires both pragmatic and epistemic terms in the policy cost:

$$G(\pi) = \mathrm{complexity}(\pi) - \mathrm{pragmatic}(\pi) - \mathrm{epistemic}(\pi)$$

The pragmatic term encourages policies that lead to high-likelihood observations (or in our case, confident self-predictions). The epistemic term encourages policies that *reduce* uncertainty about the hidden state — it counter-balances the pragmatic term's tendency toward self-reinforcement. Without the epistemic term, the pragmatic term alone collapses into a fixed-point feedback loop: the agent strengthens its current prediction, becomes more confident in it, strengthens it more, and converges to a degenerate point distribution at whatever it initially happened to predict. With the epistemic term, the agent is rewarded for *informative* updates rather than just confident ones, and the loop is broken.

The implementation uses the BALD-style mutual-information form of the epistemic term:

$$\mathrm{MI}(v; \mu \mid q_i) = H[\mathbb{E}_{\mu \sim q_i} p(v|\mu)] - \mathbb{E}_{\mu \sim q_i}[H[p(v|\mu)]]$$

This is the Bayesian Active Learning by Disagreement score. It is non-negative, it is exactly zero when $q_i$ is a point mass (no disagreement between samples), and it is large when the predictive distribution differs strongly between samples drawn from $q_i$ — that is, when the parameter uncertainty in the belief carries decision-relevant information about the observation.

### 4.3 The augmented free energy

Putting both terms together, the augmented free energy minimized by the E-step is

$$F_{\text{AI}}[q] = F[q] + \lambda_{\text{prag}} \cdot H[p_{\text{pred}}(v|\mu_i)] - \lambda_{\text{epi}} \cdot \mathrm{MI}(v; \mu \mid q_i)$$

where $p_{\text{pred}}(v|\mu_i) = \mathrm{softmax}(-\mathrm{KL}(q_i \| \pi_v)/\tau)$ is the existing PriorBank readout, $\pi_v$ is the per-vocabulary prior in the prior bank, and the predictive distribution is computed without ever seeing the true target. The pragmatic term is positive and is *minimized* (entropy minimization toward confident predictions). The epistemic term enters with a negative sign so that minimizing $F_{\text{AI}}$ corresponds to *maximizing* mutual information.

Two design choices in this formulation deserve explicit mention. First, the pragmatic and epistemic terms operate at the *position* level (one term per token, summed over the batch) rather than the *pair* level (one term per attention edge). A per-pair formulation would compute $G_{ij}$ for each candidate attention source $j$ at each position $i$ and feed it into the softmax that produces $\beta_{ij}$. This would more closely match the standard active-inference structure where policies are explicit choices, but it would cost one PriorBank readout per attention pair per iteration — roughly $N^2 \cdot V \cdot K$ operations per layer per iteration, which is approximately 80x the cost of attention itself for the $K=60$, $V=50000$, $N=128$ configuration that is our reference setting. The position-level formulation is computationally tractable (one readout per position per iteration, roughly comparable to the cost of one attention pass) and theoretically captures the same content modulo the per-pair attribution. Second, the gradient is computed via `torch.autograd.grad` on a freshly-detached $\mu$ leaf rather than analytically. The PriorBank readout is differentiable in $\mu$ via a single closed-form fused matmul, so autograd works fine, and using it avoids having to derive and maintain a custom backward for what is structurally a one-off augmentation.

### 4.4 Implementation details

A module-level helper `_compute_active_inference_gradient` was added (originally inline in `variational_ffn.py`, later extracted to `transformer/core/active_inference.py` as described in §6) with the signature

```python
def _compute_active_inference_gradient(
    mu_current, sigma_current, prior_bank,
    pragmatic_weight, epistemic_weight, epistemic_samples, decode_tau,
) -> Optional[torch.Tensor]:
```

The body opens a `torch.enable_grad()` context, builds a fresh $\mu$ leaf via `mu_current.detach().requires_grad_(True)`, computes the pragmatic term as the mean over positions of the softmax entropy of the PriorBank decode, computes the epistemic term as the BALD MC estimate by sampling $S$ (default 4) reparameterized values from $N(\mu, \sigma)$, evaluates the readout at each, and computes the difference between the entropy of the average distribution and the average entropy of the per-sample distributions. The total free energy

$$F_{\text{AI}} = \lambda_{\text{prag}} \overline{H[p_{\text{pred}}]} - \lambda_{\text{epi}} \overline{\mathrm{MI}}$$

is then differentiated with respect to the $\mu$ leaf via `torch.autograd.grad(total_efe, mu_var)[0]`, and the result is detached and cast back to the original dtype before being combined with the analytic gradient that the rest of the E-step has already computed.

The injection point in the E-step is in `_vfe_iteration`, immediately after the existing observation-gradient block and before the natural-gradient projection. This location was chosen because it is *after* both the multi-head and single-$\beta$ paths converge (they merge at the observation-gradient computation), so the EFE gradient is added uniformly regardless of which attention path was taken. The injection block is gated on three conditions: the master toggle `_ai_enabled` must be True, at least one of the two weights must be positive, and the PriorBank reference must have been plumbed in by the model at construction time. If any of these is false, the EFE block is bypassed entirely.

The PriorBank reference is plumbed in by `model.py` via `__dict__` assignment to bypass `nn.Module`'s `__setattr__`:

```python
if self.prior_bank is not None:
    for _block in self.transformer.blocks:
        _block.ffn.__dict__['_prior_bank_ref'] = self.prior_bank
```

This is necessary because `nn.Module.__setattr__` would otherwise re-register the PriorBank as a sub-module of every FFN, which would double-count its parameters in `model.parameters()`, break checkpoint loading, and cause optimizer states to be associated with multiple "owners" of the same tensor. Bypassing the magic via `__dict__` stores the reference as a plain Python attribute that the module hierarchy does not see.

A master configuration toggle `active_inference: bool = False` was added to `BlockConfig` matching the project convention used by `non_flat_transport`, `rope_full_gauge`, and `hierarchical_priors`. The toggle gates the entire EFE path: when False, the weights have no effect regardless of their values; when True, the weights take effect.

### 4.5 Three execution contexts

The EFE helper had to be made robust to three different execution contexts that have subtly different rules about autograd. Under normal training (`model.train()` plus a normal forward pass) the helper just runs and produces a gradient. Under validation (`model.eval()` plus `torch.no_grad()`) the helper would crash without intervention because `requires_grad_(True)` is silently ignored under `no_grad`; the fix is the local `torch.enable_grad()` block, which restores grad mode for the duration of the helper. Under text generation (`model.generate()` which uses `@torch.inference_mode()`) the local `enable_grad` is *not* sufficient because inference mode marks tensors as inference tensors that cannot ever be promoted to `requires_grad`, even inside a nested `enable_grad` context. The fix is to detect inference mode via `torch.is_inference_mode_enabled()` at the top of the helper and return `None`, which causes the calling code to skip the EFE gradient injection entirely. This is the same defensive pattern used by the experimental `rope_full_gauge` path.

All four practical contexts have been verified to work end-to-end. Training forward plus backward with `active_inference=True` and both weights at 0.05 produces non-zero gradients on the PriorBank parameters (norm approximately 4.6 on the smoke-test config), confirming the autograd path through the readout is differentiable end-to-end. Validation under `torch.no_grad` succeeds with the EFE term still applied (the `enable_grad` block restores grad mode locally). Generation under `torch.inference_mode` succeeds with the EFE term skipped (the inference-mode check returns None and the analytic gradient runs alone). Resumed training after a generation interlude returns to the full EFE path correctly.

### 4.6 Empirical sanity check

A direct comparison between EFE-off and EFE-on at fixed seed and inputs confirms that the new path is doing what theory predicts. With the master toggle off and the weights at 1.0, the readout entropy averages 1.7090 nats and the maximum logit difference relative to a clean run is 0.000000 (the gate is closed and the weights are ignored). With the toggle on, the pragmatic weight at 1.0 alone, the average readout entropy drops to 1.6781 nats and the maximum logit difference is 0.501 — the pragmatic term is reducing prediction entropy as designed. With the epistemic weight at 1.0 alone, the average entropy rises slightly to 1.7119 and the maximum logit difference is 0.044 — the BALD term is pushing the belief into a region where the predictive distribution is more sensitive to belief samples, which is the expected qualitative behavior of mutual-information maximization. Both signs match theory. The fact that the epistemic effect is smaller in magnitude than the pragmatic effect at matched weights is expected: the BALD MI is a small quantity (a few nats at most) compared to the per-position entropy (which scales with $\log V$).

### 4.7 Trust-region restructuring

A follow-up diagnostic session revealed a subtle failure mode in the initial implementation. At fresh initialization on the user's local configuration ($V = 50257$, $K = 20$, $N = 16$, one layer, one E-step iteration), the EFE gradient produced no measurable difference relative to the EFE-off baseline — the user observed an identical training trajectory with and without the master toggle enabled. The diagnostic trace showed that the EFE gradient was being computed and returned non-zero values, but the *applied* update to $\mu$ was numerically indistinguishable from zero.

The cause was that the EFE gradient was being combined with the VFE gradient and then fed through the same natural-gradient projection and trust-region clip that applies to the main VFE update. Two things went wrong in this shared pipeline. First, the natural-gradient projection multiplies by $\sigma^2$ to account for the Fisher metric on Gaussian beliefs, and the VFE terms are indeed KL divergences whose natural gradient is correctly scaled by $\sigma^2$. The EFE terms are entropies and mutual informations of a softmax readout, which do not inherit the Fisher-metric justification for that scaling — applying $\sigma^2$ to the EFE gradient at initialization where $\sigma^2 \sim 1$ is a no-op in magnitude but a conceptually wrong step. Second, at large vocabulary the entropy gradient is small at near-uniform init (the analysis developed in §5.3 below gives the scaling as $O(\sigma_\ell / \sqrt{V})$), and the VFE coupling gradient is comparatively much larger. The shared trust-region clip $\|\Delta\mu / \sqrt{\sigma}\| \leq 2.0$ was being saturated almost entirely by the VFE contribution, and the EFE contribution was being clipped down to a fraction of its already-small value.

The fix was to restructure the EFE update into a strictly Euclidean step applied *after* the VFE step, with its own separate step size and its own whitened trust-region budget. Concretely, the new logic in `apply_ai_mu_updates` computes

$$\Delta\mu_{\text{EFE}} = -\eta_{\text{AI}} \cdot \nabla_{\mu} F_{\text{AI}}$$

with $\eta_{\text{AI}} = 1.0$ (configurable via `active_inference_lr`, an order of magnitude larger than the VFE step size), then clips the whitened norm $\|\Delta\mu_{\text{EFE}} / \sqrt{\sigma}\|$ to a separate budget of 0.5 (configurable via `active_inference_trust_region`), and adds the clipped update to the $\mu$ that the VFE step has already produced. The total whitened update bound is therefore $\|\Delta\mu_{\text{VFE}} + \Delta\mu_{\text{EFE}}\| \leq 2.0 + 0.5 = 2.5$, which is a modest and principled relaxation of the original bound. The Euclidean gradient is the correct direction for the entropy and mutual-information terms because they do not have a Fisher-metric justification on $\mu$, and the separate budget prevents the EFE contribution from being crushed by VFE saturation.

After this restructuring the EFE path produces a measurable effect on $\mu$ at fresh initialization (though still small in absolute magnitude, as predicted by the §5.3 curvature analysis), and the empirical comparison in §4.6 becomes reproducible from a cold start rather than requiring tens of training steps before the readout develops enough structure to produce a non-degenerate EFE gradient.

## 5. Bootstrap Self-Distillation

A separate thread in the session introduced a third E-step term — bootstrap self-distillation — motivated by two observations about the pragmatic and epistemic terms from §4. First, the pragmatic entropy term is self-reinforcing on its own and requires the epistemic counterweight; a single term with the right structural properties could be simpler. Second, at fresh initialization with a 50K-vocabulary softmax, both the pragmatic and epistemic gradients are of order $\sigma_\ell / \sqrt{V}$, which is small enough that the E-step augmentation does not begin to shape the belief update until the readout has developed some structure through training. A coupling term that operates in prediction space rather than belief space, structured as a cross-entropy against a stop-gradient target, could plausibly escape the near-uniform flat-extremum regime once data-dependent structure appears and would occupy a genuinely empty slot in the E-step coupling landscape. The full design document is `docs/bootstrap_self_distillation.md`; this section summarizes the main content and records the corrections that were made after reviewer feedback.

### 5.1 Design

Let $p^{(t)}_{\text{pred}}(v \mid \mu) = \mathrm{softmax}(-\mathrm{KL}(q \| \pi_v)/\tau)$ denote the PriorBank readout at iteration $t$, and let $\Omega_{ij}$ denote the parallel transport from position $j$ to position $i$ in the current gauge frame. The bootstrap self-distillation loss at position $i$ is

$$L_{\text{distill},i} = \sum_j \mathrm{sg}[\beta_{ij}] \cdot \mathrm{CE}(\mathrm{sg}[p^{(t)}_{\text{pred}}(v \mid \Omega_{ij}\mu_j)], p^{(t)}_{\text{pred}}(v \mid \mu_i))$$

where $\mathrm{CE}(p, q) = -\sum_v p_v \log q_v$ is the standard cross-entropy and $\mathrm{sg}[\cdot]$ is the stop-gradient operator. Two stop-gradients are present and both are essential. The first severs the gradient through the target distribution, which is the structural analogue of the target-network detachment in BYOL and DINO and is what prevents the trivial collapse in which both sides of the cross-entropy shrink toward the same degenerate point. The second severs the gradient through the attention weight $\beta_{ij}$, which addresses a distinct failure mode — the *attend-to-twins* collapse — analyzed symbolically with SymPy and described in detail in Section 6 of `bootstrap_self_distillation.md`. Without the second stop-gradient the loss admits a trivial descent direction in which position $i$ concentrates all of its attention on whichever neighbours already happen to agree with it, driving the loss to zero without conveying any useful information.

An equivalent aggregated form uses the attention-aggregated transported belief $\tilde\mu_i = \sum_j \beta_{ij} \Omega_{ij}\mu_j$ already computed by the attention sublayer as the target site:

$$L^{\text{agg}}_{\text{distill},i} = \mathrm{CE}(\mathrm{sg}[p^{(t)}_{\text{pred}}(v \mid \tilde\mu_i)], p^{(t)}_{\text{pred}}(v \mid \mu_i))$$

This costs a single readout per position per iteration, matching the cost of the existing pragmatic term, and its semantics ("match the consensus of your transported neighbours") is slightly different from the per-pair form ("match each of your transported neighbours, weighted") but arguably more faithful to the belief propagation interpretation developed in Section 8 of the design document.

### 5.2 Implementation

The aggregated form is implemented in `active_inference.py:_compute_distillation_gradient`. The function builds $\tilde\mu_i$ using per-head block-diagonal einsums on fully-detached attention weights and transport pairs, runs the PriorBank (or `W_out` fallback) decode at both $\mu_i$ and $\tilde\mu_i$, and computes the cross-entropy with a stop-gradient on the target. The decode happens inside a `torch.enable_grad()` context for the same reason as the EFE helper (so that validation under `torch.no_grad()` works) and is gated on `torch.is_inference_mode_enabled()` returning False (so that decoding under `torch.inference_mode()` falls back cleanly to the analytic path). The gradient is computed via `torch.autograd.grad(total_loss, mu_var)[0]`, detached, and cast back to the original dtype before being applied as a separate Euclidean step with its own step size and its own whitened trust-region budget, parallel to the EFE update structure developed in §4.7.

Four configuration flags were added to `BlockConfig`. The weight `active_inference_distill_weight` defaults to 0.0 and acts as the on/off gate for the term, deliberately independent of the `active_inference` master toggle so that the distillation term can be enabled standalone without the pragmatic and epistemic EFE terms. The step size `active_inference_distill_lr` defaults to 1.0, matching the EFE step size. The normalization flag `active_inference_distill_normalize` defaults to True, which divides the CE by $\log V$ so that a uniform distribution produces a unit-order contribution regardless of vocabulary size. The mode string `active_inference_distill_mode` defaults to `'aggregated'` and is the only mode currently implemented; a `'per_pair'` mode is reserved for future work.

### 5.3 The Section 11 curvature correction

The initial draft of the design document claimed that the bootstrap distillation gradient is larger than the entropy gradient at random initialization — specifically, that it scales as $O(1/\sqrt{V})$ while the entropy gradient decays to $o(1/V)$ — and concluded that distillation should "fire at initialization" while the pragmatic term does not. The user correctly identified this as a misconflation of per-component and $\ell_2$ norms, and the analysis was reworked from scratch and verified symbolically in `docs/_section11_curvature_verify.py`.

Writing the softmax logits at position $i$ as $\ell_v^{(i)}$ with mean zero and empirical standard deviation $\sigma_\ell$, the linearization around the zero-logit point gives the probability perturbation as $\varepsilon_v^{(i)} = (1/V)(\ell_v^{(i)} - \bar\ell^{(i)})$, so the per-component probability perturbation has magnitude $O(\sigma_\ell / V)$ and the perturbation vector $\boldsymbol\varepsilon$ lies in the simplex tangent subspace $\sum_v \varepsilon_v = 0$. The entropy gradient, Taylor-expanded around the uniform distribution, becomes

$$\frac{\partial H}{\partial z_k} = -\varepsilon_k + O(\varepsilon^2)$$

with per-component magnitude $|\varepsilon_k| \sim \sigma_\ell / V$ and $\ell_2$ norm across the vocabulary equal to $\sqrt{V \cdot (\sigma_\ell/V)^2} = \sigma_\ell / \sqrt{V}$. The distillation CE gradient is $p_{i,k} - \tilde p_{i,k}$ where $\tilde p_i$ is the attention-aggregated transported-neighbour readout, which is also near-uniform at init with the same scaling, so the CE gradient has per-component magnitude $O(\sigma_\ell / V)$ and $\ell_2$ norm $O(\sigma_\ell / \sqrt{V})$. Both gradients have the *same* leading-order magnitude. The corrected analysis is verified component-by-component with SymPy: the power series expansion of $\partial H / \partial z_k$ in $t$ (with $z_k = t \cdot e_k$) has leading coefficient exactly $-e_k / V$, matching the predicted $-\varepsilon_k$, and a numerical cross-check at $V = 6$ with random mean-zero logit perturbations produces $\|\nabla H\|_2 \approx 0.028$ and $\|\nabla L_{\text{distill}}\|_2 \approx 0.035$, a ratio near the $\sqrt{2}$ predicted for two independent perturbations of the same order and far from any $V$-scaled separation.

The curvature analysis gives the same conclusion. The entropy Hessian at the uniform distribution is

$$\left. \nabla^2_z H \right|_{z=0} = -\frac{1}{V} I + \frac{1}{V^2} \mathbf{1}\mathbf{1}^\top$$

with eigenvalues $-1/V$ on the simplex tangent space (orthogonal to $\mathbf{1}$). The CE Hessian at the current point $p$ is $\mathrm{diag}(p) - p p^\top$, which at $p = $ uniform gives

$$\left. \nabla^2_z L_{\text{distill}} \right|_{\mathrm{uniform}} = \frac{1}{V} I - \frac{1}{V^2} \mathbf{1}\mathbf{1}^\top$$

with eigenvalues $+1/V$ on the tangent space. Both objectives are equally ill-conditioned at random initialization with curvature magnitude $1/V$, and the common folklore that "CE has $O(1)$ curvature while entropy has $O(1/V)$" only holds once the target concentrates mass (which requires training).

The real distinction between the distillation term and the pragmatic term is not gradient magnitude at initialization but fixed-point structure and cross-position coherence. The entropy gradient at position $i$ is $-\varepsilon_i$, determined entirely by the position-local noise; the distillation gradient at position $i$ is $\varepsilon_i - \tilde\varepsilon_i$, a function of both local and aggregated noise from other positions. Once training develops structure in the readout, the distillation gradient's data-dependent fixed point (the attention-weighted consensus of transported neighbour predictions) grows sharper while the entropy gradient continues to push toward the nearest simplex vertex regardless of what the data wants. Session diagnostics on the user's configuration confirm the corrected prediction: at step 0 the max-logit difference between distillation-on and distillation-off is $1.24 \times 10^{-5}$, essentially identical to the pragmatic-only difference up to a small constant factor.

### 5.4 Four stability concerns

The revision pass after the Section 11 correction addressed four additional concerns raised by the user, each of which required either document revisions, code changes, or both.

**Uniform collapse is the dominant failure mode and was under-addressed.** The consensus condition at the distillation fixed point admits a degenerate attractor in which every position converges to the same near-unigram predictive distribution; this is the DINO collapse, the same failure mode that BYOL, SimSiam, and DINO all had to engineer additional mechanisms to prevent. The only mechanism in the design that counteracts uniform collapse is the M-step cross-entropy loss against actual token targets, which forces position-specific distinctions through supervised signal. The document was revised to downgrade the claim that M-step CE "prevents" collapse to the honest claim that it "opposes" collapse, with the outcome determined by the relative weights and by unproved attractor-basin analysis. A DINO-style centering mitigation (subtracting an EMA of the batch-mean target logits before computing the CE) was described as a future extension and is flagged but not implemented. The revision also adds an explicit guard against pure-VFE configurations: any code path that runs the E-step without a simultaneously-active supervised CE signal has no mechanism to prevent uniform collapse and should not enable this term. A runtime diagnostic was added to `wire_readout_references` that emits a reminder whenever the distillation term is enabled, with the note that the user must ensure a supervised signal is active during training.

**The initialization-magnitude prediction was stated too confidently.** The original draft gave an order-of-magnitude band for the max-logit difference at step 0 without providing a derivation. The revised text replaces the band with the measured value ($1.24 \times 10^{-5}$) from actual session diagnostics, so that any reader running the verification can match the number exactly rather than wondering whether their run falls in the quoted range.

**Pragmatic and distillation together is a correctness issue, not a tuning question.** This was the most consequential revision. The combination of the pragmatic term with the distillation term creates a new failure mode in which the two terms *mutually reinforce* uniform collapse. The pragmatic term sharpens $p_{\text{pred}}(\mu_i)$ toward the local argmax $v^*$; once one position has made progress toward concentrating at $v^*$, its transported prediction carries most of its probability mass at $v^*$; neighbouring positions then compute distillation targets that inherit the $v^*$ peak through attention aggregation; the distillation gradient at position $j$ pulls toward the $v^*$-peaked target; the pragmatic term at $j$ sharpens further from wherever the distillation pull has landed; and the cycle repeats on the next iteration with the peak reinforced at both positions. The epistemic term cannot break this loop because the distillation target is frozen by stop-gradient — the MI term rewards predictive distributions that depend on belief samples, but the target is an external frozen constant that carries no uncertainty signal to be increased. The document was revised to promote this from an open tuning question to a stability constraint, and a runtime warning was added to `wire_readout_references` that fires at model construction time whenever both the pragmatic weight and the distillation weight are positive:

```
[GaugeTransformerLM] active_inference_pragmatic_weight > 0 simultaneously with
active_inference_distill_weight > 0 is a STABILITY HAZARD, not a tuning question.
The two terms mutually reinforce uniform collapse: pragmatic sharpens toward the
local argmax, distillation propagates the argmax across positions via transported
neighbour readouts, and the cycle amplifies a collapsed prediction that the
epistemic term cannot break (its target is frozen by stop-gradient).
```

The recommended configuration is either distillation on with pragmatic and epistemic off, or pragmatic and epistemic on with distillation off. Running all three together should require explicit user acknowledgement.

**Gauge equivariance is a requirement on the implementation, not a theorem.** The original draft claimed that the distillation term's gauge equivariance follows from the invariance of Gaussian KL under simultaneous gauge transformation of its two fiducials. That claim is a theorem about the KL function — it holds for any invertible $g$ — but it does not automatically imply that the PriorBank readout is gauge-equivariant under position-dependent gauge transformations. The readout is a softmax over $\{\mathrm{KL}(q_i \| \pi_v)\}_v$, and the theorem applies only when the prior $\pi_v$ is transformed in sync with the query belief. The section was rewritten to frame gauge equivariance as a *requirement on the implementation* of the PriorBank: the prior and the query must be in the same gauge frame at the point of KL evaluation, and a position-dependent gauge transformation applied to the query must act simultaneously on the prior. Three implementation patterns that would silently violate the requirement are called out explicitly — a PriorBank that stores fixed global-frame priors while the query is per-position gauged, a future per-position-gauged prior extension that does not synchronize with the belief gauge, and experimental rotations that act on the query alone rather than as a whole-sandwich transformation. A checklist for implementers is included, and the first-pass implementation is verified to satisfy it by construction because it calls the existing `prior_bank.decode(mu, sigma, tau)` method and uses no gauge operations not already shared with the rest of the E-step.

## 6. Active Inference Module Extraction

By the end of the distillation work, `variational_ffn.py` had grown to approximately 4124 lines and contained the implementation of two distinct E-step augmentations (EFE and bootstrap distillation) intertwined with the principal VFE computation. The two active-inference features share no state with the core VFE loop beyond their injection points, they have their own helper functions, their own configuration attributes, their own runtime diagnostics, and their own execution-context guards. Keeping them inline in the VFE file was making the principal computation harder to read and harder to test in isolation.

The refactor lifted all of the active-inference code into a new standalone module `transformer/core/active_inference.py`. The module is organized into six logical units. The two computation helpers `_compute_active_inference_gradient` and `_compute_distillation_gradient` perform the autograd-based gradient computations for the EFE and distillation terms respectively. The dispatch wrapper `compute_ai_gradients` resolves the PriorBank and `W_out` references from the FFN instance's `__dict__` and calls both computation helpers, returning a tuple of optional gradients. The update applier `apply_ai_mu_updates` takes those gradients and applies each as a separate Euclidean step with its own whitened trust region, implementing the budget structure developed in §4.7. The configuration helper `configure_ffn_active_inference` is called from `blocks.py` at FFN construction time to read all thirteen active-inference attributes from the `BlockConfig` and install them as instance attributes on the FFN. The wiring helper `wire_readout_references` is called from `model.py` after the full module hierarchy is built to plumb PriorBank and `W_out` fallback references into each block's FFN, emit the diagnostic messages for the various configuration combinations, and install the stability-hazard and uniform-collapse warnings described in §5.4.

The effect on `variational_ffn.py` is a reduction from approximately 4124 lines to 3641 lines, with the removed content replaced by two single-line imports and a pair of function calls. The effect on `blocks.py` is a reduction from 22 inline attribute assignments to a single `configure_ffn_active_inference(self.ffn, cfg)` call. The effect on `model.py` is a reduction from 45 inline lines of diagnostic emission and reference wiring to a single `wire_readout_references(self.transformer, self.prior_bank, self.out_proj, logger=logger)` call.

The refactor was verified to be bit-exact backward-compatible against the pre-refactor baseline. Running the same smoke-test configuration with a fixed seed produces a baseline logits sum of $-33868.289062$ both before and after the refactor, with zero floating-point drift in the forward pass. All 471 non-slow non-GPU tests continue to pass. The runtime diagnostic warnings fire correctly in three tested configurations: distillation-only produces the two informational messages about the term being active and the uniform-collapse reminder, distillation plus pragmatic produces those two messages plus the stability-hazard warning, and pragmatic-only produces neither distillation message (confirming that the distillation-specific diagnostics are properly gated on the distillation weight alone).

A small follow-up change to the logging level was made after an initial training run did not display the distillation activation messages in the console. The messages were originally emitted at `INFO` level and were being filtered out by the default logging configuration, which only shows `WARNING` and above. Both the "distillation active" message and the uniform-collapse reminder were promoted to `WARNING` level to match the existing EFE activation diagnostic, which is also emitted as a `WARNING` for the same visibility reason.

## 7. Audit Findings and Loud Runtime Warnings

An audit pass using multiple sub-agents uncovered three configuration combinations in which the active-inference path would silently do nothing or interact incorrectly with other code paths, and a fourth configuration bug that was the three-way default inconsistency on `hierarchical_priors` already described in §2.3. Each of the three silent-skip findings was promoted to a loud runtime warning emitted at model construction time by `wire_readout_references`.

The first silent-skip finding is that `active_inference=True` combined with `closed_form_e_step=True` causes the entire EFE path to be bypassed, because the closed-form E-step uses an analytic fixed-point solve that sets `_n_iters = 0` and therefore skips the iterative VFE loop where the EFE gradient is injected. A user who enables both flags would see the training run progress without any error, but the EFE pragmatic and epistemic terms would have no effect on $\mu$. The warning text explicitly states this and recommends disabling one of the two flags.

The second silent-skip finding is that `active_inference=True` combined with `use_deq=True` produces a more subtle failure. The DEQ implicit-differentiation backward pass uses a Jacobian built from the VFE-only step operator, not the VFE+EFE composite operator. The forward pass therefore includes the EFE contribution correctly, but the M-step gradient is based on the wrong fixed-point operator and the learned parameters update in directions that do not correspond to minimizing the true augmented free energy. The warning recommends disabling one of the two flags.

The third silent-skip finding was the `hierarchical_priors` default inconsistency documented in §2.3, which is technically a different kind of bug (wrong defaults rather than a skipped code path) but was discovered in the same audit pass. All three audit findings were fixed in the same pass, and the loud-warning pattern at model construction time was adopted as a general strategy for catching configuration combinations that would otherwise silently produce wrong results.

## 8. Theoretical Discussions That Did Not Become Code

Several theoretical questions were raised during the session that were analyzed in detail but did not result in implementation, either because the proposed change was not in fact theoretically sound, because it violated a hard constraint of the codebase, or because the cleaner principled alternative was deferred to a separate ablation. These are recorded here because they shape the rationale for what was implemented.

### 8.1 The residual stream scale concern

The user raised the observation that the additive residual `mu_q = mu_q + (mu_ffn - mu_normalized)` might be problematic: the VFE delta is bounded by a trust region of size 2 in $\sigma$-normalized space, while `mu_q` itself can grow unboundedly across layers, so the signal-to-residual ratio degrades with depth. The proposed fix was a sigmoid-gated correction, $\mu_q \leftarrow \mu_q + \sigma(f(\Sigma)) \cdot (\mu_{\text{ffn}} - \mu_{\text{normalized}})$. The analysis in this session concluded that the concern is partially valid but overstated, and that the proposed fix violates a hard constraint. The trust region is not fixed; it is $\|\Delta\mu\| \leq 2\|\sigma\|$, which scales with uncertainty rather than being a constant. RMSNorm normalizes the input to the FFN before the VFE update, so the VFE update sees a unit-scale belief regardless of how large $\mu_q$ has grown in absolute terms, and the output projection sees the post-`final_norm` value, so unbounded residual growth does not leak directly into the logit scale. The proposed sigmoid gate is an activation function, and CLAUDE.md hard-bans activation functions from this codebase under the no-neural-networks constraint. A principled alternative — replacing the additive residual with a precision-weighted Bayesian combination, $\mu_* = \Sigma_*(\Lambda_q \mu_q + \Lambda_{\text{ffn}} \mu_{\text{ffn}})$ — was discussed but not implemented because it is a substantial architectural change that should be tested in a clean ablation rather than mixed into run-88 debugging at depth 2 where residual dispersion is not the dominant pathology.

### 8.2 Whether VFE prescribes additive belief combination

A direct theoretical question was raised: does VFE prescribe the operation `mu_q + mu_ffn`? The answer is no. VFE prescribes exactly three principled operations on Gaussian beliefs: natural gradient descent on a single belief, $\mu_{\text{new}} = \mu_{\text{old}} - \eta \Sigma \nabla_\mu F$; Bayesian product of two beliefs over the same latent, $\Lambda_* = \Lambda_a + \Lambda_b$ and $\mu_* = \Sigma_* (\Lambda_a \mu_a + \Lambda_b \mu_b)$; and replacement, $\mu_q \leftarrow \mu_{\text{ffn}}$. Additive combination is not in any of the three. The current `mu_q + (mu_ffn - mu_normalized)` operation has at best a loose interpretation as a "stale natural gradient step in the normalized frame" — it can be viewed as applying the gradient that the FFN computed at $\mathrm{RMSNorm}(\mu_q)$ to $\mu_q$ itself, which is valid only insofar as the free energy is approximately scale-invariant under RMSNorm. It is an inherited transformer-architecture pattern that has been retrofitted with a loose VFE interpretation, not a derivation from the free energy functional. The existing residual is *defensible* but not *derived*, and the honest position is to acknowledge the gap rather than dress it up as a theorem.

### 8.3 Self-observation as a feedback loop

The user proposed that a token could "observe its own prediction" — a middle ground between cheating (using the true target via `use_obs_in_vfe`) and ignoring (no observation at all). Three formulations were analyzed: reflexive conditioning on the argmax of the predictive distribution, soft entropy minimization on the predictive distribution, and the BALD-style mutual information. The first two were rejected because they create a self-reinforcement feedback loop in which the model is rewarded for being decisive rather than correct, and the M-step CE loss has to actively fight against the E-step. Only the BALD form is principled, because mutual information rewards updates that *reduce* uncertainty rather than ones that confirm the current guess. This analysis directly motivated the design choice in §4 to implement both pragmatic and epistemic terms in the EFE augmentation rather than the simpler pragmatic-only form, and it also shaped the §5 decision to make bootstrap self-distillation standalone-enable-able rather than requiring a counterweight — the stop-gradient on the target gives the distillation term a data-dependent fixed point that does not self-reinforce in the way the pragmatic entropy term does.

## 9. Files Modified

The session touched ten files across `transformer/core/` and `docs/`. The scope of each change is summarized below.

| File | Purpose of changes |
|---|---|
| `active_inference.py` | **New file** (742 lines). Module-level helpers `_compute_active_inference_gradient` and `_compute_distillation_gradient`, dispatch wrapper `compute_ai_gradients`, update applier `apply_ai_mu_updates`, configuration helper `configure_ffn_active_inference`, and wiring helper `wire_readout_references`. Contains all EFE and bootstrap-distillation logic extracted from `variational_ffn.py` in §6. |
| `block_config.py` | Added `hierarchical_priors`, `rope_full_gauge`, `active_inference` (master toggle), `active_inference_pragmatic_weight`, `active_inference_epistemic_weight`, `active_inference_epistemic_samples`, `active_inference_decode_tau`, `active_inference_trust_region`, `active_inference_lr`, `active_inference_distill_weight`, `active_inference_distill_lr`, `active_inference_distill_normalize`, `active_inference_distill_mode`. Marked `sigma_residual` as a deprecated no-op with a warning. Wired all new fields through `from_config()`. Fixed the three-way default inconsistency on `hierarchical_priors`. |
| `blocks.py` | Sigma residual fix at attention sublayer (~line 441) and FFN sublayer (~line 480). Hierarchical `mu_prior` update in `GaugeTransformerStack.forward` with default-arg closure capture for gradient checkpointing. Replaced 22 inline active-inference attribute assignments with a single `configure_ffn_active_inference(self.ffn, cfg)` call. |
| `model.py` | Mirror of sigma residual fix in `forward_with_attention`. Mirror of hierarchical `mu_prior` update. Replaced 45 inline lines of PriorBank wiring, silent-skip warnings, and active-inference diagnostics with a single `wire_readout_references(self.transformer, self.prior_bank, self.out_proj, logger=logger)` call. |
| `variational_ffn.py` | Reduced from ~4124 lines to 3641 lines by extracting the EFE and distillation code to `active_inference.py`. Retains the `_rope_full_gauge_vfe` initialization and dispatch, the full-gauge RoPE graph-detach fix, and the two-line import plus two-line call that plumbs into `active_inference.py`. |
| `vfe_gradients.py` | RoPE chain rule fix in `_fused_attention_and_vfe_gradients_block_diag`: accumulate `grad_kl_rope_per_pair` inside the block loop, apply `_un_apply_rope_pair_outer` after the loop when `use_rope=True`, and use the un-rotated gradient in the `d_beta_d_mu` term. Same fix in `_compute_vfe_gradients_block_diagonal_diag`. New experimental function `_compute_rope_full_gauge_gradient_per_head` wrapped in `with torch.enable_grad()`. Added `use_rope`, `rope_base` parameters to `compute_vfe_gradients_gpu` dispatcher with warnings for unsupported paths. |
| `transport_ops.py` | New helper `_un_apply_rope_pair_outer(grad, base)` applying $R(\theta)^\top$ to per-pair gradients. New helper `_apply_rope_to_covariance(sigma_full, base)` for `rope_full_gauge`. Documentation comments on `_apply_rope` clarifying the $\mu$-only convention and what the alternative full-gauge interpretation would entail. |
| `attention.py` | Documentation comment block at the `if use_rope:` line in `compute_attention_weights` explaining the attention-vs-value gauge factorization. No code changes; the file was correct already. |
| `docs/bootstrap_self_distillation.md` | **New file** (298 lines). Design document for bootstrap self-distillation. Sections on motivation, coupling landscape, formal proposal, gradient analysis, fixed-point analysis, attend-to-twins collapse, gauge equivariance requirements, generalized belief propagation connection, BYOL/DINO comparison, implementation strategies, corrected curvature analysis, stability constraints, and two SymPy verification appendices. |
| `docs/_bootstrap_distill_verify.py` | **New file** (147 lines). SymPy verification of the four central mathematical claims in the design document: CE gradient identity, entropy gradient two-critical-point structure, attend-to-twins gradient on attention scores, and Gaussian KL invariance under simultaneous gauge transformation. |
| `docs/_section11_curvature_verify.py` | **New file** (198 lines). SymPy verification of the four curvature claims in the Section 11 correction: entropy gradient leading-order coefficient at near-uniform, entropy Hessian at uniform, CE Hessian at uniform, and numerical cross-check of gradient norms at $V = 6$. |
| `docs/session_2026_04_06_active_inference_and_efe.md` | This file. Rewritten to cover the expanded scope through 2026-04-07. |

No training scripts or analysis tools were modified in this session.

## 10. Configuration Reference

To enable everything implemented in this session in a fresh training config, set the following keys in your `EM_CONFIG` (or equivalent flat config dict):

```python
# Sigma residual fix is automatic — no config required. The legacy
# 'sigma_residual' flag is now a no-op and can be removed from old configs.

# Hierarchical priors (now default True everywhere; explicit for clarity)
'hierarchical_priors': True,

# Experimental: full gauge interpretation of RoPE
'rope_full_gauge': False,         # set True to test the framework-consistent variant
'use_rope': True,
'rope_base': 1000.0,

# Active inference / EFE augmentation of the E-step
'active_inference': True,                       # MASTER TOGGLE (default False)
'active_inference_pragmatic_weight': 1.0,       # lambda_prag
'active_inference_epistemic_weight': 0.5,       # lambda_epi  (keep ON to avoid feedback loop)
'active_inference_epistemic_samples': 4,        # S, MC samples for BALD
'active_inference_decode_tau': 1.0,             # PriorBank decode temperature
'active_inference_lr': 1.0,                     # Step size for the Euclidean EFE mu-update
'active_inference_trust_region': 0.5,           # Whitened trust region for the EFE update

# Bootstrap self-distillation (standalone term, NOT gated by active_inference)
'active_inference_distill_weight': 0.05,        # lambda_distill; start small
'active_inference_distill_lr': 1.0,             # Euclidean step size
'active_inference_distill_normalize': True,     # Divide CE by log(V)
'active_inference_distill_mode': 'aggregated',  # only mode currently implemented

# Required for the active inference and distillation paths
'use_prior_bank': True,
```

Two usage notes apply to the combined configuration. If `active_inference_distill_weight > 0`, the `active_inference_pragmatic_weight` should be set to 0.0 to avoid the mutual-reinforcement uniform-collapse failure mode described in §5.4; a loud `STABILITY HAZARD` warning is emitted at model construction if both are positive. If `active_inference=True` is combined with either `closed_form_e_step=True` or `use_deq=True`, loud warnings are emitted at model construction to flag the silent-skip failure modes described in §7.

The defaults for all new flags are chosen so that an existing training config that does not mention them produces identical results to before the session: `active_inference=False` bypasses the entire EFE path, `active_inference_distill_weight=0.0` bypasses the distillation path, `rope_full_gauge=False` uses the standard analytic fused gradient (which now has the chain-rule fix), and `hierarchical_priors=True` was changed from False but the practical effect on existing runs is small because most existing configs were already passing `mu_prior` from the embedding bank to every layer.

## 11. Verification Summary

The complete non-slow non-GPU test suite (`pytest tests/transformer/ -m "not slow and not gpu"`) was run after each of the major changes and at the end of the session. All 471 tests pass. Twenty-one are skipped (expected, semantics tests requiring GPU or external data) and twelve are deselected by the marker filter. There are zero regressions.

The RoPE chain rule fix was additionally validated against PyTorch autograd to machine precision: manual computation of the corrected gradient matches autograd at 2.22e-16 relative error (the float32 epsilon), and the full fused path matches autograd at 5e-8 across forward and backward, sufficient to confirm that the un-rotation is the only correction needed and that no other terms are silently miscomputed.

The active-inference path was empirically verified to produce non-zero gradients on the PriorBank parameters when enabled, to reduce the average readout entropy when only the pragmatic term is on, to increase the readout entropy slightly when only the epistemic term is on (the expected BALD-MI direction), and to produce identical results to the EFE-off path when the master toggle is off regardless of weight values. After the §4.7 trust-region restructuring, the EFE path additionally produces a measurable (if small) effect on $\mu$ at fresh initialization, confirming that the separate Euclidean budget prevents VFE saturation from crushing the EFE contribution.

The bootstrap distillation path was verified end-to-end on the user's local configuration. At step 0 with `active_inference_distill_weight=1.0`, the measured max-logit difference between distillation-on and distillation-off is $1.24 \times 10^{-5}$, matching the corrected Section 11 prediction. The two SymPy verification scripts `docs/_bootstrap_distill_verify.py` and `docs/_section11_curvature_verify.py` both run cleanly and confirm the four gradient and Hessian claims in the design document.

The `active_inference.py` refactor was verified for bit-exact backward compatibility by comparing the logits sum on a fixed-seed smoke test before and after the extraction: the baseline value $-33868.289062$ matches to the last reported digit, confirming zero floating-point drift in the forward pass. The runtime diagnostic warnings fire correctly in three tested configurations as described in §6.

## 12. Future Directions

Three follow-up directions are worth flagging.

The first is the precision-weighted residual replacement discussed in §8.1. If the additive residual turns out to be the dominant bottleneck at depths 4 and above, replacing it with the principled Bayesian combination $\mu_* = \Sigma_*(\Lambda_q \mu_q + \Lambda_{\text{ffn}} \mu_{\text{ffn}})$ is the natural next experiment. This is a clean, isolated change with a clear theoretical motivation and a straightforward ablation against the baseline.

The second is per-pair active inference and per-pair distillation. Both the EFE and distillation terms currently operate at the position level, with the gradient computed from a single aggregated readout per position. The strict active-inference formulation would compute $G_{ij}$ separately for each candidate attention source $j$ and feed the result into the softmax that produces $\beta_{ij}$, and the strict per-pair distillation form would compute a separate CE per neighbour weighted by the stop-gradient attention. Both are roughly 80x more expensive at our standard configuration but have stronger theoretical justifications than the aggregated forms currently implemented.

The third is the DINO-style centering mitigation for bootstrap distillation. The current design relies on the M-step CE to oppose uniform collapse, with an honest acknowledgement that this is not proved sufficient. A centering EMA buffer subtracted from the target logits before computing the CE would move the uniform direction out of the equilibrium manifold and provide stronger formal collapse resistance. The stateful EMA buffer introduces non-trivial checkpoint and resumption concerns, so the mitigation is flagged as a follow-up rather than enabled by default, but it is the cleanest single-line addition that would improve the collapse-resistance story.

## Appendix A: Glossary of Symbols

| Symbol | Meaning |
|---|---|
| $\mu_i, \Sigma_i$ | Posterior mean and covariance of the belief at position $i$ |
| $\phi_i$ | Lie algebra coordinates parameterizing the gauge frame at position $i$ |
| $\Omega_{ij}$ | Parallel transport operator from position $j$ to position $i$, $\Omega_{ij} = \exp(\phi_i \cdot G) \exp(-\phi_j \cdot G)$ |
| $G$ | Lie algebra generators (depends on gauge group; SO(3), SO(N), GL(K)) |
| $R(\theta_i)$ | RoPE rotation at position $i$, block-diagonal $\mathrm{SO}(2)^{d_k/2} \subset \mathrm{GL}(K)$ |
| $\beta_{ij}$ | Attention weight from position $i$ (query) to position $j$ (key/source) |
| $\kappa$ | Attention temperature (action precision in active-inference language) |
| $\alpha$ | Self-coupling weight on $\mathrm{KL}(q_i \| p_i)$ |
| $\lambda_{\text{belief}}$ | Belief alignment weight on the coupling term |
| $\pi_v$ | Per-vocabulary prior in the PriorBank |
| $p_{\text{pred}}(v|\mu_i)$ | PriorBank readout, $\mathrm{softmax}(-\mathrm{KL}(q_i \| \pi_v)/\tau)$ |
| $H[\cdot]$ | Shannon entropy |
| $\mathrm{MI}(v; \mu \mid q)$ | Mutual information between vocabulary distribution and belief samples |
| $G(\pi)$ | Expected free energy of policy $\pi$ in active inference |
| $\lambda_{\text{prag}}, \lambda_{\text{epi}}$ | Active inference pragmatic and epistemic weights |
| $\lambda_{\text{distill}}$ | Bootstrap self-distillation weight |
| $\tilde\mu_i$ | Attention-aggregated transported belief at position $i$, $\sum_j \beta_{ij} \Omega_{ij} \mu_j$ |
| $\mathrm{sg}[\cdot]$ | Stop-gradient operator |
| $\varepsilon_v$ | Probability-space perturbation around uniform, $p_v = 1/V + \varepsilon_v$ |
| $\sigma_\ell$ | Empirical standard deviation of the softmax logits at initialization |

## Appendix B: Hard Constraints That Shaped the Work

Three constraints from `CLAUDE.md` shaped which solutions were implementable in this session.

The no-neural-networks constraint forbids `nn.Linear`, MLPs, and activation functions. This rules out the sigmoid-gated residual proposed in §8.1 and any other learned-gating mechanism. The retained exceptions are the linear $K \to V$ output projection and (when enabled) the `connection.py` MLP mode for non-flat transport experiments.

The no-CLI-arguments constraint requires all entry points to use the click-to-run pattern with config dicts edited directly in source files. No new CLI flags were added during the session; every new feature is reached by editing `EM_CONFIG` in `train_publication.py` (or the equivalent `BlockConfig` constructor for direct-instantiation code paths).

The preserve-gauge-equivariance constraint requires covariance transport to always use the sandwich product $\Sigma \to \Omega \Sigma \Omega^\top$. This shaped the RoPE full-gauge implementation in §3.2 (the experimental variant lifts diagonal $\Sigma$ to a full covariance specifically so that the rotation can be applied as a sandwich product rather than as an asymmetric $\mu$-only operation) and shaped the §5.4 revision of the distillation document (gauge equivariance is a requirement on the PriorBank implementation, not an automatic consequence of the Gaussian KL invariance theorem).
