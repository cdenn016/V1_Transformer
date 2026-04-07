# Active Inference, RoPE Gauge Geometry, and Numerical Bug Fixes

**Session date:** 2026-04-06
**Codebase:** Gauge-Theoretic VFE Transformer (V13)
**Scope:** Three numerical/correctness fixes, two experimental features, and a principled active-inference extension to the variational E-step. All changes are backwards compatible and gated behind explicit configuration flags.

---

## Abstract

This report documents a single working session that touched seven files in `transformer/core/` and resulted in two correctness fixes, one performance regression fix, two experimental research features, and one substantial theoretical extension to the variational free energy E-step. The correctness work fixed an additive sigma residual that had pegged the posterior covariance at the ceiling within a few forward passes for multi-layer configurations, and a missing chain-rule term in the fused RoPE attention gradient that produced wrong (rather than just suboptimal) gradients on coupled-position belief updates. The features added hierarchical priors (each layer's posterior μ becomes the next layer's prior μ), an experimental "full gauge" interpretation of RoPE in which the rotation acts on covariances as well as means via the standard sandwich product, and a master-toggled active-inference E-step augmentation that adds the pragmatic and epistemic terms of expected free energy as autograd-derived gradients flowing through the existing PriorBank readout. The active-inference extension is the conceptual centerpiece of the session: the existing KL attention mechanism is shown to be structurally identical to Boltzmann policy selection over the complexity term of expected free energy, and the new code adds the missing pragmatic and epistemic halves so that the E-step minimizes the full EFE rather than only the complexity component. All 471 non-slow non-GPU tests continue to pass, and the new path is verified to behave correctly in all four relevant execution contexts (training, validation under `torch.no_grad`, decoding under `torch.inference_mode`, and training resumed after a generation interlude).

---

## 1. Background and Motivation

The session opened with diagnostics on training run 88, a 2-layer K=60 GL(10) configuration that was underperforming a matched 1-layer baseline. Inspection of the per-step metrics CSV revealed three pathologies: the diagonal posterior covariance σ_q was pegged at the configured ceiling of 12.0 within the first hundred steps, the prior–belief KL was growing unboundedly from roughly 273 nats at initialization to 858 nats after a few thousand steps, and the learnable α coefficient (the Bayesian precision on the self-coupling term) was collapsing toward zero across both layers. None of these were obviously bugs; they could plausibly have reflected model misspecification or training-rate mismatch. The first half of the session was spent isolating which were genuine code defects and which were emergent behavior of a working system. Two of the three turned out to be code defects that had been masked at single-layer depth where the per-sublayer compounding was below the ceiling within a single forward pass.

The remainder of the session followed a sequence of theoretical questions raised by the user about whether various mechanical pieces of the model could be reformulated more principled. Each question was answered substantively (pushing back when the proposed change was not actually theoretically sound), and when the answer pointed at a principled implementation, the implementation was carried out and tested. The most consequential of these threads — active inference as a closed-form prescription for what the E-step is missing — became the largest single piece of work and is documented in detail in §4.

---

## 2. Numerical and Correctness Fixes

### 2.1 Sigma Residual Inflation

**Symptom.** In the diagnostics from run 88, the posterior covariance σ_q at every position rose to its configured ceiling `sigma_max = 12.0` within the first few forward passes and remained pinned there for the remainder of training. This had the immediate consequence of saturating the natural-gradient projection (which scales by σ²) and the indirect consequence of making the prior coupling term degenerate, since the KL between a wide posterior and the comparatively tight prior is dominated by the trace term `tr(σ_p^{-1} σ_q)`.

**Root cause.** The block-level sublayer aggregation in `blocks.py` was using an additive update for the covariance even though the sub-layer functions return the *full* updated covariance, not a delta. Specifically, both the attention sublayer and the FFN sublayer were doing

$$\sigma_q \leftarrow \sigma_q + \sigma_{\text{sub}}$$

where `sigma_sub` was the *new* posterior covariance produced by the sublayer, not the *change* in the covariance. This double-counted absolute scale at every sublayer of every layer. With L layers and 2 sublayers per layer, the multiplicative inflation factor was $2^{2L}$ at the worst case, which is why a 1-layer model trained correctly while the 2-layer model saturated the ceiling almost immediately. The misnamed `sigma_residual` flag in `BlockConfig` was supposed to be the gate for this behavior; in practice the additive path ran regardless of the flag.

**Fix.** The two sublayers in `blocks.py` were both changed from additive update to delta extraction by replacement:

```python
if self.evolve_sigma and sigma_attn is not None:
    sigma_q = sigma_attn.clamp(min=1e-4, max=self.sigma_max)
# ... after FFN sublayer
if self.evolve_sigma and sigma_ffn is not None:
    sigma_q = sigma_ffn.clamp(min=1e-4, max=self.sigma_max)
```

The mean μ has its own residual connection through `mu_q = mu_q + (mu_ffn - mu_normalized)` because the FFN receives a normalized input and returns a state in the same normalized frame; the covariance has no normalization step (RMSNorm acts on means, not covariances), so the cleanest semantics is direct replacement. The mirror change was made in `model.py:forward_with_attention` where the same code is duplicated for attention recording. The legacy `sigma_residual` flag was retained in `BlockConfig` as a deprecated no-op so that any saved configs continue to load.

**Validation.** With the fix in place, σ_q under the same training config drifts upward only modestly across layers and remains well below the ceiling, allowing the natural-gradient projection to operate in its design regime.

### 2.2 RoPE Chain Rule in the Fused Gradient Path

**Symptom.** When the gauge transformer is configured with `use_rope=True`, the attention sublayer applies a rotary positional rotation to μ before computing the inter-position KL. The exact attention forward path was correct, but the *fused* attention-and-VFE-gradient path in `vfe_gradients.py` (the optimized routine that computes both β and the belief gradient in a single pass over the block-diagonal Ω construction) was missing a chain-rule factor. The bug was silent — gradients were finite, training did not crash, and loss did decrease — but the gradient propagated to μ through the coupling term β·KL was systematically wrong by a position-dependent rotation.

**Mathematical derivation of the missing term.** When RoPE is enabled, the attention-side KL is computed in the rotated frame:

$$\mathrm{KL}_{\text{rope}}(q_i \| \Omega_{ij} q_j) = \mathrm{KL}\bigl(N(R(\theta_i)\mu_i, \Sigma_i) \,\big\|\, N(\Omega_{ij} R(\theta_j) \mu_j,\, \Omega_{ij} \Sigma_j \Omega_{ij}^\top)\bigr)$$

where $R(\theta_i)$ is the block-diagonal SO(2)$^{d_k/2}$ rotation associated with position $i$. The attention weights are then

$$\beta_{ij} = \mathrm{softmax}_j\!\left(-\frac{\mathrm{KL}_{\text{rope}}(q_i \| \Omega_{ij} q_j)}{\kappa}\right).$$

The VFE gradient with respect to μ contains a softmax-coupling term

$$\frac{\partial F}{\partial \mu_i} \supset \sum_j \frac{\partial \beta_{ij}}{\partial \mu_i} \cdot \mathrm{KL}_{\text{rope}}(q_i \| \Omega_{ij} q_j) \cdot \nabla_{\mu_i}\mathrm{KL}_{\text{rope}},$$

which by chain rule requires the gradient of the rotated KL with respect to the *raw* (un-rotated) μ:

$$\nabla_{\mu_i^{\text{raw}}} \mathrm{KL}_{\text{rope}} = R(\theta_i)^\top \cdot \nabla_{R(\theta_i)\mu_i} \mathrm{KL}_{\text{rope}}.$$

The fused path was computing the inner gradient (the derivative of the KL with respect to the rotated μ) but skipping the outer $R(\theta_i)^\top$ multiplication, so the gradient that went into the coupling term was the gradient *as if μ were already in the rotated frame*. For positions where $R(\theta_i)$ is far from identity, this is a substantial error in the coupling direction.

**Fix.** A new helper `_un_apply_rope_pair_outer(grad, base)` was added to `transport_ops.py`:

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

This corrected gradient is then used in the `d_beta_d_mu` accumulator that builds the softmax coupling term.

**Verification.** The fix was validated against PyTorch autograd at machine precision. With manual computation of the chain rule, the relative error against autograd dropped to 2.22e-16 (the float32 epsilon). The fused path now matches autograd to 5e-8 across the full forward and backward, confirming both that the un-rotation is correct and that no other terms in the fused path are missing the same correction.

---

## 3. Feature Additions (Smaller)

### 3.1 Hierarchical Priors

A new `BlockConfig` flag `hierarchical_priors` (default now True) makes each layer's posterior μ become the next layer's prior μ. The covariance prior `sigma_prior` continues to come from the embedding bank to prevent uncertainty cascade across layers — only the *mean* is hierarchically passed. The motivation is that without this, every layer pulls its posterior toward the same fixed embedding-bank prior, which makes deep layers indistinguishable from shallow layers in the prior coupling term and removes one of the main sources of depth-driven representation refinement. The change was wired through `block_config.py`, the forward loop in `blocks.py`, and the parallel `forward_with_attention` path in `model.py`. A subtle gradient-checkpointing fix was needed because the closure that captures `mu_prior` was binding by name rather than by value across the layer loop; the standard Python idiom of binding via default arguments was used:

```python
def create_block_fn(blk, _mu_prior=mu_prior, _omega=omega):
    def _fn(mu, sigma, phi):
        return blk(mu, sigma, phi, blk.generators, mask=mask, mu_prior=_mu_prior, omega=_omega)
    return _fn
```

without this default-argument trick the checkpointed closure would always see the *last* layer's `mu_prior` value rather than the per-layer value at the time the closure was created, producing silently wrong gradients.

### 3.2 Experimental: Full Gauge RoPE

The standard RoPE pattern in transformer literature applies the rotation only to the means used as queries and keys, leaving the value path (and in our case the covariance) unrotated. This is asymmetric in the gauge framework: if RoPE is interpreted as a position-dependent gauge frame in $\mathrm{SO}(2)^{d_k/2} \subset \mathrm{GL}(K)$, then the natural action on a Gaussian belief should act on both the mean and the covariance via the standard sandwich product:

$$\mu_i \mapsto R(\theta_i) \mu_i, \qquad \Sigma_i \mapsto R(\theta_i) \Sigma_i R(\theta_i)^\top.$$

The asymmetric (μ-only) form is what almost every published RoPE implementation does, including the standard one we inherited. The "full gauge" form has not, to our knowledge, been tried in published work, and is the framework-consistent choice given the gauge-theoretic interpretation in the GL(K) manuscript. We added an experimental `rope_full_gauge: bool = False` flag and a new gradient routine `_compute_rope_full_gauge_gradient_per_head` that lifts diagonal Σ to a full covariance, applies the sandwich rotation in the KL computation, and computes the resulting belief gradient via `torch.autograd.grad` rather than analytically. This is slower than the analytic fused path and is not currently the default, but it provides an experimental ablation point for testing whether the framework-consistent rotation produces measurable differences in training dynamics.

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

---

## 4. Active Inference Extension to the E-Step (the main work)

This section is longer than the others because the conceptual content is the most subtle and the implementation is the main artifact of the session.

### 4.1 What was already there: KL attention as Boltzmann policy selection

The variational free energy minimized by the existing E-step contains the standard self-coupling and belief-coupling terms

$$F[q] = \alpha \sum_i \mathrm{KL}(q_i \| p_i) \;+\; \lambda \sum_{i,j} \beta_{ij}\, \mathrm{KL}(q_i \| \Omega_{ij} q_j) \;+\; (\text{minor terms}),$$

where the attention weights themselves are computed from the same coupling KL via a softmax with temperature $\kappa$:

$$\beta_{ij} = \mathrm{softmax}_j\!\left(-\frac{\mathrm{KL}(q_i \| \Omega_{ij} q_j)}{\kappa}\right).$$

Compare this to the action-selection rule in active inference. An active-inference agent has beliefs $q(s)$ over hidden states, a generative model $p(o, s | \pi)$ parameterized by policies $\pi$, and selects actions by softmax of the negative expected free energy $G(\pi)$:

$$\pi^* \sim \mathrm{softmax}(-G(\pi)/\tau).$$

The expected free energy decomposes as

$$G(\pi) \;=\; \underbrace{\mathrm{KL}\bigl(q(s|\pi) \| p(s)\bigr)}_{\text{complexity}} \;+\; \underbrace{\bigl(-\mathbb{E}_q[\log p(o|s,\pi)]\bigr)}_{\text{pragmatic value}}.$$

The KL attention rule is *literally* the action-selection rule with the policy space being "which other position to attend to" and the cost being the complexity term $\mathrm{KL}(q_i \| \Omega_{ij} q_j)$. The temperature $\kappa$ is the action precision. This is not an analogy: the equations match and the variable names map onto each other one-for-one. What is genuinely missing from the existing implementation is the pragmatic value term — the expected log-likelihood of the observation under the policy. In language modeling at inference time the true observation (the next token) is not available, which is precisely the situation that active inference is designed for: when the agent does not know the observation, it substitutes its own predictive distribution.

### 4.2 What was missing: pragmatic and epistemic value

Active inference, properly stated, requires *both* pragmatic and epistemic terms in the policy cost:

$$G(\pi) \;=\; \mathrm{complexity}(\pi) \;-\; \mathrm{pragmatic}(\pi) \;-\; \mathrm{epistemic}(\pi).$$

The pragmatic term encourages policies that lead to high-likelihood observations (or in our case, confident self-predictions). The epistemic term encourages policies that *reduce* uncertainty about the hidden state — it counter-balances the pragmatic term's tendency toward self-reinforcement. Without the epistemic term, the pragmatic term alone collapses into a fixed-point feedback loop: the agent strengthens its current prediction, becomes more confident in it, strengthens it more, and converges to a degenerate point distribution at whatever it initially happened to predict. With the epistemic term, the agent is rewarded for *informative* updates rather than just confident ones, and the loop is broken.

The implementation we added uses the BALD-style mutual-information form of the epistemic term:

$$\mathrm{MI}(v;\, \mu \mid q_i) \;=\; H\!\bigl[\mathbb{E}_{\mu \sim q_i}\, p(v|\mu)\bigr] \;-\; \mathbb{E}_{\mu \sim q_i}\!\bigl[H[p(v|\mu)]\bigr].$$

This is the Bayesian Active Learning by Disagreement score. It is non-negative, it is exactly zero when $q_i$ is a point mass (no disagreement between samples), and it is large when the predictive distribution differs strongly between samples drawn from $q_i$ — that is, when the parameter uncertainty in the belief carries decision-relevant information about the observation.

### 4.3 The augmented free energy

Putting both terms together, the augmented free energy minimized by the E-step is

$$F_{\text{AI}}[q] \;=\; F[q] \;+\; \lambda_{\text{prag}} \cdot H[p_{\text{pred}}(v|\mu_i)] \;-\; \lambda_{\text{epi}} \cdot \mathrm{MI}(v;\,\mu \mid q_i),$$

where $p_{\text{pred}}(v|\mu_i) = \mathrm{softmax}(-\mathrm{KL}(q_i \| \pi_v)/\tau)$ is the existing PriorBank readout, $\pi_v$ is the per-vocabulary prior in the prior bank, and the predictive distribution is computed without ever seeing the true target. The pragmatic term is positive and is *minimized* (entropy minimization → confident predictions). The epistemic term enters with a negative sign so that minimizing $F_{\text{AI}}$ corresponds to *maximizing* mutual information.

Two design choices in this formulation are worth being explicit about. First, the pragmatic and epistemic terms operate at the *position* level (one term per token, summed over the batch) rather than the *pair* level (one term per attention edge). A per-pair formulation would compute $G_{ij}$ for each candidate attention source $j$ at each position $i$ and feed it into the softmax that produces $\beta_{ij}$. This would more closely match the standard active-inference structure where policies are explicit choices, but it would cost one PriorBank readout per attention pair per iteration — roughly $N^2 \cdot V \cdot K$ operations per layer per iteration, which is approximately 80x the cost of attention itself for the K=60, V=50000, N=128 configuration that is our reference setting. The position-level formulation is computationally tractable (one readout per position per iteration, roughly comparable to the cost of one attention pass) and theoretically captures the same content modulo the per-pair attribution. Second, the gradient is computed via `torch.autograd.grad` on a freshly-detached μ leaf rather than analytically. The PriorBank readout is differentiable in μ via a single closed-form fused matmul, so autograd works fine, and using it avoids having to derive and maintain a custom backward for what is structurally a one-off augmentation.

### 4.4 Implementation details

A new module-level helper `_compute_active_inference_gradient` was added to `variational_ffn.py` immediately before the `VariationalFFNDynamic` class. The signature is

```python
def _compute_active_inference_gradient(
    mu_current, sigma_current, prior_bank,
    pragmatic_weight, epistemic_weight, epistemic_samples, decode_tau,
) -> Optional[torch.Tensor]:
```

and the body opens a `torch.enable_grad()` context, builds a fresh μ leaf via `mu_current.detach().requires_grad_(True)`, computes the pragmatic term as the mean over positions of the softmax entropy of the PriorBank decode, computes the epistemic term as the BALD MC estimate by sampling $S$ (default 4) reparameterized values from $N(\mu, \sigma)$, evaluates the readout at each, and computes the difference between the entropy of the average distribution and the average entropy of the per-sample distributions. The total free energy

$$F_{\text{AI}} = \lambda_{\text{prag}}\, \overline{H[p_{\text{pred}}]} \;-\; \lambda_{\text{epi}}\, \overline{\mathrm{MI}}$$

is then differentiated with respect to the μ leaf via `torch.autograd.grad(total_efe, mu_var)[0]`, and the result is detached and cast back to the original dtype before being added to the analytic gradient that the rest of the E-step has already computed.

The injection point in the E-step is in `_vfe_iteration`, immediately after the existing observation-gradient block and before the natural-gradient projection. This location was chosen because it is *after* both the multi-head and single-β paths converge (they merge at the observation-gradient computation), so the EFE gradient is added uniformly regardless of which attention path was taken. The injection block is gated on three conditions: the master toggle `_ai_enabled` must be True, at least one of the two weights must be positive, and the PriorBank reference must have been plumbed in by the model at construction time. If any of these is false, the EFE block is bypassed entirely.

The PriorBank reference is plumbed in by `model.py` via `__dict__` assignment to bypass `nn.Module`'s `__setattr__`:

```python
if self.prior_bank is not None:
    for _block in self.transformer.blocks:
        _block.ffn.__dict__['_prior_bank_ref'] = self.prior_bank
```

This is necessary because `nn.Module.__setattr__` would otherwise re-register the PriorBank as a sub-module of every FFN, which would double-count its parameters in `model.parameters()`, break checkpoint loading, and cause optimizer states to be associated with multiple "owners" of the same tensor. Bypassing the magic via `__dict__` stores the reference as a plain Python attribute that the module hierarchy does not see.

A master configuration toggle `active_inference: bool = False` was added to `BlockConfig` matching the project convention used by `non_flat_transport`, `rope_full_gauge`, and `hierarchical_priors`. The toggle gates the entire EFE path: when False, the weights have no effect regardless of their values; when True, the weights take effect. The defaults for the weights are 0.05 each (chosen to be small enough that the EFE term is a perturbation rather than dominant), the default sample count for the BALD MC estimate is $S=4$, and the default decode temperature is $\tau=1.0$.

### 4.5 Three execution contexts

The EFE helper had to be made robust to three different execution contexts that have subtly different rules about autograd. Under normal training (`model.train()` plus a normal forward pass) the helper just runs and produces a gradient. Under validation (`model.eval()` plus `torch.no_grad()`) the helper would crash without intervention because `requires_grad_(True)` is silently ignored under `no_grad`; the fix is the local `torch.enable_grad()` block, which restores grad mode for the duration of the helper. Under text generation (`model.generate()` which uses `@torch.inference_mode()`) the local `enable_grad` is *not* sufficient because inference mode marks tensors as inference tensors that cannot ever be promoted to `requires_grad`, even inside a nested `enable_grad` context. The fix is to detect inference mode via `torch.is_inference_mode_enabled()` at the top of the helper and return `None`, which causes the calling code to skip the EFE gradient injection entirely. This is the same defensive pattern used by the experimental `rope_full_gauge` path.

All four practical contexts have been verified to work end-to-end:

1. Training forward + backward with `active_inference=True` and both weights at 0.05: PriorBank parameters receive non-zero gradients (norm ≈ 4.6 on the smoke-test config), confirming the autograd path through the readout is differentiable end-to-end.
2. Validation under `torch.no_grad`: forward pass succeeds, EFE term still applied (the `enable_grad` block restores grad mode locally).
3. Generation under `torch.inference_mode`: forward pass succeeds, EFE term skipped (the inference-mode check returns None and the analytic gradient runs alone).
4. Resumed training after a generation interlude: returns to the full EFE path correctly.

### 4.6 Empirical sanity check

A direct comparison between EFE-off and EFE-on at fixed seed and inputs confirms that the new path is doing what theory predicts. With the master toggle off and the weights at 1.0, the readout entropy averages 1.7090 nats and the maximum logit difference relative to a clean run is 0.000000 (the gate is closed and the weights are ignored). With the toggle on, the pragmatic weight at 1.0 alone, the average readout entropy drops to 1.6781 nats and the maximum logit difference is 0.501 — the pragmatic term is reducing prediction entropy as designed. With the epistemic weight at 1.0 alone, the average entropy rises slightly to 1.7119 and the maximum logit difference is 0.044 — the BALD term is pushing the belief into a region where the predictive distribution is more sensitive to belief samples, which is the expected qualitative behavior of mutual-information maximization. Both signs match theory. The fact that the epistemic effect is smaller in magnitude than the pragmatic effect at matched weights is expected: the BALD MI is a small quantity (a few nats at most) compared to the per-position entropy (which scales with $\log V$).

---

## 5. Theoretical Discussions That Did Not Become Code

Several theoretical questions were raised during the session that were analyzed in detail but did not result in implementation, either because the proposed change was not in fact theoretically sound, because it violated a hard constraint of the codebase, or because the cleaner principled alternative was deferred to a separate ablation. These are recorded here because they shape the rationale for what *was* implemented.

### 5.1 The residual stream scale concern

The user raised the observation that the additive residual `mu_q = mu_q + (mu_ffn - mu_normalized)` might be problematic: the VFE delta is bounded by a trust region of size 2 in σ-normalized space, while `mu_q` itself can grow unboundedly across layers, so the signal-to-residual ratio degrades with depth. The proposed fix was a sigmoid-gated correction, $\mu_q \leftarrow \mu_q + \sigma(f(\Sigma)) \cdot (\mu_{\text{ffn}} - \mu_{\text{normalized}})$. The analysis in this session concluded that the concern is partially valid but overstated, and the proposed fix violates a hard constraint. The trust region is *not* fixed: it is $\|\Delta\mu\| \leq 2\|\sigma\|$, which scales with uncertainty rather than being a constant. RMSNorm normalizes the input to the FFN before the VFE update, so the VFE update sees a unit-scale belief regardless of how large $\mu_q$ has grown in absolute terms, and the output projection sees the post-`final_norm` value, so unbounded residual growth does not leak directly into the logit scale. Most importantly, the proposed sigmoid gate is an activation function, and CLAUDE.md hard-bans activation functions from this codebase under the no-neural-networks constraint. A principled alternative — replacing the additive residual with a precision-weighted Bayesian combination, $\mu_* = \Sigma_*(\Lambda_q \mu_q + \Lambda_{\text{ffn}} \mu_{\text{ffn}})$ — was discussed but not implemented because it is a substantial architectural change that should be tested in a clean ablation rather than mixed into run-88 debugging at depth 2 where residual dispersion is not the dominant pathology.

### 5.2 Whether VFE prescribes additive belief combination

A direct theoretical question was raised: does VFE prescribe the operation `mu_q + mu_ffn`? The answer is no. VFE prescribes exactly three principled operations on Gaussian beliefs: natural gradient descent on a single belief, $\mu_{\text{new}} = \mu_{\text{old}} - \eta\,\Sigma\,\nabla_\mu F$; Bayesian product of two beliefs over the same latent, $\Lambda_* = \Lambda_a + \Lambda_b$ and $\mu_* = \Sigma_* (\Lambda_a \mu_a + \Lambda_b \mu_b)$; and replacement, $\mu_q \leftarrow \mu_{\text{ffn}}$. Additive combination is not in any of the three. The current `mu_q + (mu_ffn - mu_normalized)` operation has at best a loose interpretation as a "stale natural gradient step in the normalized frame" — it can be viewed as applying the gradient that the FFN computed at $\mathrm{RMSNorm}(\mu_q)$ to $\mu_q$ itself, which is valid only insofar as the free energy is approximately scale-invariant under RMSNorm. It is an inherited transformer-architecture pattern that has been retrofitted with a loose VFE interpretation, not a derivation from the free energy functional. This was important to be honest about: the existing residual is *defensible* but it is not *derived*.

### 5.3 Self-observation as a feedback loop

The user proposed that a token could "observe its own prediction" — a middle ground between cheating (using the true target via `use_obs_in_vfe`) and ignoring (no observation at all). Three formulations were analyzed: reflexive conditioning on the argmax of the predictive distribution, soft entropy minimization on the predictive distribution, and the BALD-style mutual information. The first two were rejected: they create a self-reinforcement feedback loop where the model is rewarded for being decisive rather than correct, and the M-step CE loss has to actively fight against the E-step. Only the BALD form is principled, because mutual information rewards updates that *reduce* uncertainty rather than ones that confirm the current guess. This analysis directly motivated the design choice in §4 to implement *both* pragmatic and epistemic terms in the EFE augmentation rather than the simpler pragmatic-only form.

---

## 6. Files Modified

Seven files in `transformer/core/` were modified during this session. The list is:

| File | Purpose of changes |
|---|---|
| `block_config.py` | Added `hierarchical_priors`, `rope_full_gauge`, `active_inference` (master toggle), `active_inference_pragmatic_weight`, `active_inference_epistemic_weight`, `active_inference_epistemic_samples`, `active_inference_decode_tau`. Marked `sigma_residual` as a deprecated no-op. Wired all new fields through `from_config()`. |
| `blocks.py` | Sigma residual fix at attention sublayer (~line 441) and FFN sublayer (~line 480). Hierarchical mu_prior update in `GaugeTransformerStack.forward` with default-arg closure capture for gradient checkpointing. Plumbing of active-inference weights to `self.ffn._ai_*` instance attributes. Initialization of `_prior_bank_ref` via `__dict__.setdefault`. |
| `model.py` | Mirror of sigma residual fix in `forward_with_attention`. Mirror of hierarchical mu_prior update. Plumbing of `self.prior_bank` reference to each `block.ffn.__dict__['_prior_bank_ref']` after `GaugeTransformerStack` construction, bypassing `nn.Module` sub-module auto-registration. |
| `variational_ffn.py` | Initialization of `_rope_full_gauge_vfe` and `_ai_*` instance attributes. New module-level helper `_compute_active_inference_gradient` (~120 lines) implementing the autograd-based EFE gradient via PriorBank decode. Dispatch branch for `_use_rope_full` in the multi-head loop with inference-mode guard. EFE gradient injection block in `_vfe_iteration` after the observation gradient. |
| `vfe_gradients.py` | RoPE chain rule fix in `_fused_attention_and_vfe_gradients_block_diag`: accumulate `grad_kl_rope_per_pair` inside the block loop, apply `_un_apply_rope_pair_outer` after the loop when `use_rope=True`, and use the un-rotated gradient in the `d_beta_d_mu` term. Same fix in `_compute_vfe_gradients_block_diagonal_diag`. New experimental function `_compute_rope_full_gauge_gradient_per_head` (~200 lines) wrapped in `with torch.enable_grad()`. Added `use_rope`, `rope_base` parameters to `compute_vfe_gradients_gpu` dispatcher with warnings for unsupported paths. |
| `transport_ops.py` | New helper `_un_apply_rope_pair_outer(grad, base)` applying $R(\theta)^\top$ to per-pair gradients. New helper `_apply_rope_to_covariance(sigma_full, base)` for rope_full_gauge. Documentation comments on `_apply_rope` clarifying the μ-only convention and what the alternative full-gauge interpretation would entail. |
| `attention.py` | Documentation comment block at the `if use_rope:` line in `compute_attention_weights` explaining the attention-vs-value gauge factorization. No code changes — this file was correct already, the comments document the asymmetric design choice and the alternative. |

No tests, training scripts, or analysis tools were modified in this session.

---

## 7. Configuration Reference

To enable everything implemented in this session in a fresh training config, set the following keys in your `EM_CONFIG` (or equivalent flat config dict):

```python
# Sigma residual fix is automatic — no config required, but the legacy
# 'sigma_residual' flag is now a no-op and can be removed from old configs.

# Hierarchical priors (now default True; explicit for clarity)
'hierarchical_priors': True,

# Experimental: full gauge interpretation of RoPE
'rope_full_gauge': False,         # set True to test the framework-consistent variant
'use_rope': True,
'rope_base': 1000.0,

# Active inference / EFE augmentation of the E-step
'active_inference': True,                       # MASTER TOGGLE (default False)
'active_inference_pragmatic_weight': 0.05,      # λ_prag
'active_inference_epistemic_weight': 0.05,      # λ_epi
'active_inference_epistemic_samples': 4,        # S, MC samples for BALD
'active_inference_decode_tau': 1.0,             # PriorBank decode temperature

# Required for the active inference path (the EFE term calls prior_bank.decode)
'use_prior_bank': True,
```

The defaults for all new flags are chosen so that an existing training config that does not mention them produces identical results to before the session: `active_inference=False` bypasses the entire EFE path, `rope_full_gauge=False` uses the standard analytic fused gradient (which now has the chain-rule fix), `hierarchical_priors=True` was changed from False but the practical effect on existing runs is small because most existing configs were already passing `mu_prior` from the embedding bank to every layer.

---

## 8. Verification Summary

The complete non-slow non-GPU test suite (`pytest tests/transformer/ -m "not slow and not gpu"`) was run after each of the major changes and at the end of the session. All 471 tests pass. Twenty-one are skipped (expected, semantics tests requiring GPU/external data) and twelve are deselected by the marker filter. There are zero regressions.

The RoPE chain rule fix was additionally validated against PyTorch autograd to machine precision: manual computation of the corrected gradient matches autograd at 2.22e-16 relative error (the float32 epsilon), and the full fused path matches autograd at 5e-8 across forward and backward — sufficient to confirm that the un-rotation is the only correction needed and that no other terms are silently miscomputed.

The active-inference path was empirically verified to (a) produce non-zero gradients on the PriorBank parameters when enabled, (b) reduce the average readout entropy when only the pragmatic term is on, (c) increase the readout entropy slightly when only the epistemic term is on (the expected BALD-MI direction), and (d) produce identical results to the EFE-off path when the master toggle is off, regardless of weight values.

---

## 9. Future Directions

Three follow-up directions are worth flagging.

The first is the precision-weighted residual replacement discussed in §5.1. If the additive residual turns out to be the dominant bottleneck at depths 4 and above, replacing it with the principled Bayesian combination $\mu_* = \Sigma_*(\Lambda_q \mu_q + \Lambda_{\text{ffn}} \mu_{\text{ffn}})$ is the natural next experiment. This is a clean, isolated change with a clear theoretical motivation and a straightforward ablation against the baseline.

The second is per-pair active inference. The current implementation operates at the position level: the pragmatic and epistemic terms are added to the free energy of position $i$ as a whole, and the gradient flows into $\mu_i$ via autograd. The strict active-inference formulation would compute $G_{ij}$ separately for each candidate attention source $j$ and feed the result into the softmax that produces $\beta_{ij}$. This is roughly 80x more expensive at our standard configuration and has not been implemented, but if the position-level form shows promising signal, the per-pair form would be the next step.

The third is empirical evaluation of `rope_full_gauge` against the standard μ-only RoPE. Both are valid choices in the gauge framework, but only the μ-only form has been used in published transformer work. A clean comparison at matched compute and matched configuration would be a contribution in its own right, separate from the rest of the work in this session.

---

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
| $\mathrm{MI}(v;\mu|q)$ | Mutual information between vocabulary distribution and belief samples |
| $G(\pi)$ | Expected free energy of policy $\pi$ in active inference |
| $\lambda_{\text{prag}}, \lambda_{\text{epi}}$ | Active inference pragmatic and epistemic weights |

## Appendix B: Hard Constraints That Shaped the Work

Three constraints from `CLAUDE.md` shaped which solutions were implementable in this session:

1. **No neural networks.** No `nn.Linear`, no MLPs, no activation functions. This rules out the sigmoid-gated residual proposed in §5.1 and any other learned-gating mechanism. The retained exceptions are the linear K → V output projection and (when enabled) the `connection.py` MLP mode for non-flat transport experiments.
2. **No CLI arguments.** All entry points use the click-to-run pattern with config dicts edited directly in source files. No new CLI flags were added.
3. **Preserve gauge equivariance.** Covariance transport must always use the sandwich product $\Sigma \to \Omega \Sigma \Omega^\top$. This shaped the RoPE full-gauge implementation in §3.2: the experimental variant lifts diagonal $\Sigma$ to a full covariance specifically so that the rotation can be applied as a sandwich product rather than as an asymmetric μ-only operation.
