# A Principled Gauge-VFE Transformer

## Executive Summary

A gauge-VFE language model replaces single hidden vectors with Gaussian beliefs at each token position:

$$q_i(z) = \mathcal{N}(z;\, \mu_i, \Sigma_i), \quad g_i = \exp(\phi_i \cdot G) \in \mathcal{G}.$$

Attention is geometric: $\beta_{ij} \propto \exp\bigl(-\tfrac{1}{\kappa}\, D(q_i \,\|\, \Omega_{ij}[q_j])\bigr)$, where $\Omega_{ij} = g_i g_j^{-1}$ is the gauge transport.

Each block performs an E-step on $(\mu, \Sigma, \phi)$ by minimizing a variational free energy built from prior consistency and gauge-transported alignment. The M-step is ordinary supervised language modeling: solve for $q^\star$ from the context alone, then minimize cross-entropy of the next-token prediction produced from $q^\star$. Do not feed the target token into the E-step during LM training.

Use the same statistical manifold for encode, inference, and decode. The cleanest implementation is a PriorBank that serves as both encoder and decoder: tokens map to priors $\pi_v$, and final logits are produced by KL-to-prior decoding.

---

## 1. Latent State and Gauge Action

At sequence position $i$, let the latent token state be

$$q_i(z) = \mathcal{N}(z;\, \mu_i, \Sigma_i), \quad \mu_i \in \mathbb{R}^K, \quad \Sigma_i \in \mathrm{SPD}(K),$$

together with a gauge frame coordinate

$$\phi_i \in \mathfrak{g}, \quad g_i = \exp(\phi_i \cdot G) \in \mathcal{G}.$$

A local gauge transformation $h_i \in \mathcal{G}$ acts by

$$\mu_i \mapsto h_i \mu_i, \quad \Sigma_i \mapsto h_i \Sigma_i h_i^\top, \quad g_i \mapsto h_i g_i.$$

The induced parallel transport from $j$ to $i$ is $\Omega_{ij} = g_i g_j^{-1}$, and the transported Gaussian is

$$\Omega_{ij}[q_j] = \mathcal{N}\!\bigl(\Omega_{ij} \mu_j,\; \Omega_{ij} \Sigma_j \Omega_{ij}^\top\bigr).$$

The covariance sandwich product $\Sigma \mapsto \Omega \Sigma \Omega^\top$ is the central correctness invariant of the framework.

**Gauge invariance of KL.** Under the gauge action $h_i$, the transport becomes $\Omega_{ij} \mapsto h_i \Omega_{ij}$, so both $q_i$ and $\Omega_{ij}[q_j]$ receive the same conjugation by $h_i$. Each of the three terms in the KL divergence is independently invariant: the trace term by cyclic invariance of trace, the Mahalanobis term by cancellation of $h_i^\top (h_i \cdot h_i^\top)^{-1} h_i = I$, and the log-determinant ratio by cancellation of $\det(h_i)^2$. The resulting attention weights $\beta_{ij}$ are therefore gauge-invariant.

---

## 2. Vocabulary as a Prior Bank

Let the vocabulary define a family of token priors

$$\Pi = \{\pi_v\}_{v=1}^{V}, \quad \pi_v(z) = \mathcal{N}(z;\, \mu_v^\pi, \Sigma_v^\pi).$$

The gauge-fixed orbit parameterization is

$$\pi_v = A_v \triangleright \pi_0, \quad A_v = \exp(\psi_v \cdot G),$$

so that $\mu_v^\pi = A_v \mu_0$ and $\Sigma_v^\pi = A_v \Sigma_0 A_v^\top$.

**Encoding.** For input token $x_i$, initialize beliefs from the token prior:

$$\mu_i^{(0)} = \mu_{x_i}^\pi, \quad \Sigma_i^{(0)} = \Sigma_{x_i}^\pi, \quad \phi_i^{(0)} = \psi_{x_i}.$$

**Decoding.** Given a final latent belief $q_i^\star$, define token logits by KL-to-prior:

$$\ell_{i,v} = -\frac{1}{\tau}\, \mathrm{KL}(q_i^\star \,\|\, \pi_v), \quad p_\theta(y_i = v \mid x_{<i}) = \mathrm{softmax}_v(\ell_{i,v}).$$

This is the correct readout when hidden states, priors, and the observation model all live on the same Gaussian statistical manifold.

---

## 3. Positional Structure as Gauge Composition

Position enters as a contribution to the gauge frame, not as an additive Euclidean feature. Let $p_i \in \mathfrak{g}$ be a learned or fixed positional Lie-algebra element. Then

$$\phi_i^{(0)} = \mathrm{BCH}(\phi_i^{\mathrm{token}},\, p_i), \quad \text{equivalently} \quad g_i^{(0)} = \exp(\phi_i^{\mathrm{token}} \cdot G)\, \exp(p_i \cdot G).$$

This makes transport $\Omega_{ij} = g_i g_j^{-1}$ depend on relative position in a gauge-covariant way.

**BCH truncation.** The Baker-Campbell-Hausdorff formula gives $\log(\exp X \exp Y) = X + Y + \tfrac{1}{2}[X,Y] + \cdots$. First-order truncation $\mathrm{BCH}_1(X,Y) = X + Y$ is exact for abelian groups and a reasonable approximation for non-abelian groups when $\|X\|, \|Y\| \ll 1$ (the commutator $[X,Y]$ is $O(\|X\|\,\|Y\|)$). For large gauge fields, second-order BCH or exact group multiplication should be used; both are available in the codebase.

---

## 4. Gauge-KL Attention

Define the pairwise geometric divergence at layer $\ell$:

$$D_{ij}^{(\ell)} = \mathrm{KL}\!\bigl(q_i^{(\ell)} \,\big\|\, \Omega_{ij}^{(\ell)}[q_j^{(\ell)}]\bigr).$$

Attention weights are

$$\beta_{ij}^{(\ell)} = \frac{\exp\!\bigl(-D_{ij}^{(\ell)} / \kappa_\ell\bigr)\, m_{ij}}{\sum_k \exp\!\bigl(-D_{ik}^{(\ell)} / \kappa_\ell\bigr)\, m_{ik}},$$

where $m_{ij}$ is the causal mask. This is the core architectural thesis: attention is a Gibbs kernel on the statistical manifold with gauge transport, not produced by learned $W_Q, W_K$ projections.

The transported message mean is

$$\bar{\mu}_i^{(\ell)} = \sum_j \beta_{ij}^{(\ell)}\, \Omega_{ij}^{(\ell)} \mu_j^{(\ell)}.$$

For covariance aggregation, there are two principled choices.

**4.1 Mixture (moment-matching).** Treat the attention-weighted transported beliefs as a Gaussian mixture and match the first two moments:

$$\bar{\Sigma}_i = \sum_j \beta_{ij} \bigl(\Omega_{ij} \Sigma_j \Omega_{ij}^\top + (\Omega_{ij} \mu_j - \bar{\mu}_i)(\Omega_{ij} \mu_j - \bar{\mu}_i)^\top\bigr).$$

This has the interpretation: total uncertainty = average within-component uncertainty + between-component spread.

**4.2 Precision (product-of-experts).** Aggregate in natural (precision) coordinates:

$$\bar{\Lambda}_i = \sum_j \beta_{ij}\, \Lambda_{ij}, \quad \bar{\eta}_i = \sum_j \beta_{ij}\, \eta_{ij},$$

where $\Lambda_{ij} = (\Omega_{ij} \Sigma_j \Omega_{ij}^\top)^{-1}$ and $\eta_{ij} = \Lambda_{ij}\, \Omega_{ij} \mu_j$. Then $\bar{\Sigma}_i = \bar{\Lambda}_i^{-1}$ and $\bar{\mu}_i = \bar{\Sigma}_i\, \bar{\eta}_i$.

Precision aggregation produces tighter posteriors (intersection of evidence), while mixture aggregation produces broader posteriors (union of evidence). For attention-based message passing, mixture is the safer default — precision aggregation can cause pathological contraction when many sources agree.

---

## 5. The Per-Layer E-Step Free Energy

At layer $\ell$, let $p_i^{(\ell)}$ denote the layer prior and $q_i^{(\ell)}$ the current belief. The E-step free energy is

$$F(q, \phi) = \alpha \sum_i \mathrm{KL}(q_i^{(\ell)} \,\|\, p_i^{(\ell)}) + \lambda_{\mathrm{align}} \sum_{i,j} \beta_{ij}^{(\ell)}\, \mathrm{KL}(q_i^{(\ell)} \,\|\, \Omega_{ij}^{(\ell)}[q_j^{(\ell)}]) + \lambda_{\mathrm{soft}}\, C(q, \phi) + \lambda_{\mathrm{reg}}\, R(q, \phi).$$

The first term is prior consistency, the second is belief alignment through gauge transport, $C$ is the softmax-coupling correction (the $\partial\beta/\partial\theta \cdot \mathrm{KL}$ term arising from differentiating through the softmax), and $R$ collects target-free regularizers.

The alignment gradient decomposes as

$$\frac{\partial}{\partial\theta} \sum_j \beta_{ij}\, \mathrm{KL}_{ij} = \sum_j \beta_{ij}\, \frac{\partial \mathrm{KL}_{ij}}{\partial\theta} + \sum_j \mathrm{KL}_{ij}\, \frac{\partial \beta_{ij}}{\partial\theta},$$

where the first term is the direct (Boltzmann-gated) coupling and the second is the softmax-coupling correction. These are controlled independently by $\lambda_{\mathrm{align}}$ and $\lambda_{\mathrm{soft}}$.

**Exclusion of targets from the E-step.** For autoregressive language modeling, this free energy must not contain the supervised next-token target as an observation. The E-step infers beliefs from context and priors only. When targets are injected into the E-step (via the `use_obs_in_vfe` flag), the model can shortcut the representation by reading the answer directly, creating a train/test mismatch: training PPL is artificially low while validation PPL degrades. Evaluation already forces `use_obs_in_vfe=False`, confirming the asymmetry.

---

## 6. E-Step Updates

Let $t$ index the inner-loop iterations.

**Mean update (natural gradient).**

$$\mu_i^{(t+1)} = \mu_i^{(t)} - \eta_\mu\, \Sigma_i^{(t)}\, \nabla_{\mu_i} F.$$

This is the Gaussian-location natural gradient: the Fisher information for the mean parameter of $\mathcal{N}(\mu, \Sigma)$ with fixed $\Sigma$ is $F_\mu = \Sigma^{-1}$, so the natural gradient is $\tilde{\nabla}_\mu F = F_\mu^{-1} \nabla_\mu F = \Sigma \nabla_\mu F$ (Amari, 1998).

**Covariance update (SPD retraction).**

$$\Sigma_i^{(t+1)} = \Sigma_i^{(t)} \exp\!\bigl(-\eta_\Sigma\, \Sigma_i^{(t)}\, \nabla_{\Sigma_i} F\, \Sigma_i^{(t)}\bigr).$$

This is an affine-invariant retraction on $\mathrm{SPD}(K)$. The natural gradient for the covariance parameter of a Gaussian is $\tilde{\nabla}_\Sigma F = 2\,\Sigma\, \nabla_\Sigma F\, \Sigma$ (from the Fisher metric on the covariance manifold), and the matrix exponential ensures the result remains SPD.

**Gauge-frame update (Lie-algebra descent).**

$$\phi_i^{(t+1)} = \phi_i^{(t)} - \eta_\phi\, M_\phi^{-1}(\phi_i^{(t)})\, \nabla_{\phi_i} F,$$

where $M_\phi$ is a natural metric on the Lie algebra: the Killing form (for semisimple groups) or a pullback metric.

---

## 7. Cross-Layer Hierarchical Prior Handoff

A deep model requires a rule that turns layer $\ell$'s posterior into layer $(\ell+1)$'s prior. The exact version is

$$\mu_p^{(\ell+1)} = \mu^{(\ell)\star}, \quad \Sigma_p^{(\ell+1)} = \Sigma^{(\ell)\star}.$$

The current implementation uses a stabilized variant: $\mu_p$ is propagated from the previous layer's posterior, but $\Sigma_p$ is frozen at the embedding value to prevent progressive tightening (sigma cascade). A damped interpolation

$$\mu_p^{(\ell+1)} = (1 - \rho_\mu)\, \mu_p^{(\ell)} + \rho_\mu\, \mu^{(\ell)\star}$$

is not currently implemented but would be a safer practical choice. The sigma-freezing strategy is a pragmatic stabilization, not a theoretically principled choice — it prevents cascade at the cost of limiting the depth of the hierarchical Bayesian structure.

---

## 8. Gauge-Equivariant Normalization

After the final layer, normalize with the Mahalanobis norm:

$$\hat{\mu}_i = \mu_i^\star \cdot \sqrt{\frac{K}{\mu_i^{\star\top}\, \Sigma_i^{\star -1}\, \mu_i^\star + \varepsilon}}.$$

The Mahalanobis form $\mu^\top \Sigma^{-1} \mu$ is a gauge scalar: under $\mu \to g\mu$, $\Sigma \to g\Sigma g^\top$, one has

$$(g\mu)^\top (g\Sigma g^\top)^{-1} (g\mu) = \mu^\top g^\top g^{-\top} \Sigma^{-1} g^{-1} g\, \mu = \mu^\top \Sigma^{-1} \mu.$$

The scaling factor is therefore gauge-invariant, making this the correct gauge-equivariant analogue of RMS-style normalization.

---

## 9. Decode and Language-Model Objective

Given the final normalized belief $q_i^\star = \mathcal{N}(\hat{\mu}_i, \Sigma_i^\star)$, define logits by KL decode:

$$\ell_{i,v} = -\frac{1}{\tau}\, \mathrm{KL}(q_i^\star \,\|\, \pi_v), \quad p_\theta(y_i = v \mid x_{<i}) = \mathrm{softmax}_v(\ell_{i,v}).$$

The training objective is

$$\mathcal{L}(\theta) = \sum_i \bigl(-\log p_\theta(y_i \mid q_i^\star(x_{<i}))\bigr) + \lambda_{\mathrm{hyper}} H(\theta),$$

where $q_i^\star(x_{<i})$ is obtained from context-only inference and $H(\theta)$ collects hyper-prior regularizers.

This is the clean variational split: the E-step infers $q^\star$ from context only; the M-step updates slow parameters to improve next-token prediction from $q^\star$.

---

## 10. Target-Free Active Inference in the E-Step

If active-inference shaping is desired during training-time inference, only target-free terms are permissible. Let $\hat{p}_i(v) = p_\theta(v \mid q_i)$ be the model's own predictive distribution from its current belief. Then:

**Pragmatic (entropy minimization):**

$$F_{\mathrm{prag}} = \lambda_{\mathrm{prag}} \sum_i H[\hat{p}_i].$$

**Epistemic (BALD mutual information):**

$$F_{\mathrm{epi}} = -\lambda_{\mathrm{epi}} \sum_i I(z_i;\, y_i \mid q_i).$$

These gradients are folded into $\nabla_\mu F$ and $\nabla_\Sigma F$ before the Fisher natural-gradient projection, so all terms share the same information-geometric descent. The pragmatic term alone is a self-reinforcing attractor toward entropy collapse; the epistemic term provides the necessary counterpressure. When distillation is active, the pragmatic term should be disabled to avoid a mutual-reinforcement instability.

---

## 11. Generation-Time Expected Free Energy

For genuine active inference in an LLM, the action is the next token. For each candidate next token $a$:

$$G_t(a) = \underbrace{\mathbb{E}_{q(o \mid a)}[-\log p^\star(o)]}_{\text{risk}} + \underbrace{\mathbb{E}_{q(z \mid a)}[H[p(o \mid z)]]}_{\text{ambiguity}} - \underbrace{I_q(z;\, o \mid a)}_{\text{epistemic value}}.$$

The policy posterior is $q_t(a) \propto \exp(-\gamma\, G_t(a))$.

This is the correct place for active inference in a language model: generation-time action selection, not target-conditioned E-step inference during supervised training. The epistemic term is structurally weaker for self-generated text (no exogenous observations) and should be downweighted or disabled in pure autoregressive generation.

---

## 12. The Cleaned-Up Architecture

$$\{x_i\}_{i=1}^N \xrightarrow{\text{PriorBank encode}} \{(\mu_i, \Sigma_i, \phi_i)\}_{i=1}^N \xrightarrow{\text{positional BCH}} \{(\mu_i, \Sigma_i, \phi_i)\}_{i=1}^N$$

$$\xrightarrow{L \text{ gauge-VFE blocks}} \{(\mu_i^{(L)\star}, \Sigma_i^{(L)\star}, \phi_i^{(L)\star})\}_{i=1}^N \xrightarrow{\text{MahalanobisNorm}} \{(\hat{\mu}_i, \Sigma_i^{(L)\star})\}_{i=1}^N$$

$$\xrightarrow{\text{PriorBank decode}} p_\theta(y_i \mid x_{<i}).$$

Each block performs: compute $\beta_{ij}$ from gauge-transported KL; compute the E-step gradients of $F$; update $(\mu, \Sigma, \phi)$ via natural gradient on the belief manifold; pass the posterior onward as the next layer's prior.

---

## 13. Three Design Laws

**Law 1: Inference must not see the answer key.** For ordinary language-model training, the E-step may depend on the context and the model's own predictions, but not on the supervised next token.

**Law 2: Transport must act on covariance by conjugation.** Any gauge transport of Gaussians must satisfy $\Sigma \mapsto \Omega\, \Sigma\, \Omega^\top$. This is the non-negotiable correctness invariant.

**Law 3: Encode, infer, and decode on the same manifold.** If tokens are represented as Gaussian priors, then inference should evolve Gaussian beliefs, and decoding should compare the final belief against Gaussian token priors by KL.

---

## 14. Practical Implementation Choices

**Recommended first version.** Flat transport; Lie-algebra $\phi$ parameterization; block-diagonal full covariance per head; PriorBank encode/decode; Mahalanobis normalization; context-only E-step; CE-only M-step; no observation term in E-step.

**Recommended second version.** Add: damped posterior-to-prior transfer across layers; target-free pragmatic/epistemic active-inference terms inside the E-step; learned per-layer temperatures $\kappa_\ell$ and $\tau$.

**Recommended third version.** Add: non-flat edge connections; generation-time EFE policy over candidate next tokens.

---

## 15. Relationship to the Current Codebase

The specification above is a principled consolidation of mechanisms the current code already contains:

- KL-based gauge-transport attention is already the attention mechanism (`transformer/core/attention.py`).
- The VFE block already performs iterative E-step belief evolution (`transformer/core/variational_ffn.py`).
- Positional gauge composition via $\phi$-space is implemented with multiple BCH modes (`transformer/core/embeddings.py`).
- PriorBank provides the unified encode/decode abstraction (`transformer/core/prior_bank.py`).
- MahalanobisNorm provides gauge-equivariant normalization (`transformer/core/blocks.py`).
- Active-inference modules distinguish between target-free E-step shaping and generation-time EFE policy (`transformer/core/active_inference.py`, `transformer/core/expected_free_energy.py`).

The previously-present `use_obs_in_vfe` flag was removed 2026-04-16: the E-step never sees target tokens. Target tokens enter the objective only through the outer CE loss term in `compute_free_energy_loss` (the observation-likelihood term of $F$).

---

## 16. Final Statement

A principled gauge-VFE transformer is a deep amortized variational model in which: token states are local Gaussian beliefs; gauge frames define transport between token beliefs; attention is the Gibbs kernel of gauge-transported divergence; each layer performs an inner variational E-step on $(\mu, \Sigma, \phi)$; the vocabulary is a bank of Gaussian priors; decoding is KL-to-prior softmax; the M-step is ordinary next-token learning from context-only inferred beliefs; and active inference, when used canonically, operates at generation time over candidate next-token actions.

---

## 17. Clean Implementation Plan: `transformer/vfe/`

### Motivation

The existing `transformer/core/` implementation (~19k LOC, 22 files) contains all the mechanisms described above but is heavily entangled. `BlockConfig` has 60+ fields. `variational_ffn.py` alone is 2,893 lines with five EM modes, DEQ, closed-form, hebbian, and implicit EM paths interleaved. Every module depends on `BlockConfig`. A separate `transformer/pure_vfe/` package (~3.3k LOC) takes the opposite approach — no autograd, no backprop, no `nn.Module` — but lacks active inference, cross-layer handoff, and BCH composition.

The `transformer/vfe/` package bridges these two worlds. It uses PyTorch autograd for the M-step (backprop through the E-step to update priors and embeddings) but has a single clean E-step path: iterative natural gradient with `straight_through` gradient flow. It imports stateless math utilities from `core/` without inheriting the entanglement. Target: ~2,400 LOC with all features from Sections 1--16.

Excluded by design: DEQ, closed-form E-step, hebbian paths. Only the iterative natural-gradient E-step with `straight_through` gradient flow is implemented. No EM mode branching.

### File Structure

```
transformer/vfe/
├── __init__.py           (~30 LOC)   Public API
├── config.py             (~120 LOC)  VFEConfig dataclass (~25 fields)
├── types.py              (~40 LOC)   BeliefState NamedTuple
├── prior_bank.py         (~250 LOC)  Encode tokens→Gaussians, decode beliefs→logits
├── positional.py         (~80 LOC)   BCH positional composition
├── attention.py          (~200 LOC)  KL-based gauge-covariant attention (stateless functions)
├── e_step.py             (~350 LOC)  Iterative VFE minimization inner loop
├── block.py              (~180 LOC)  Single VFE layer: E-step + normalization
├── stack.py              (~100 LOC)  L layers + cross-layer prior handoff
├── model.py              (~250 LOC)  Full model: embed → stack → decode → loss
├── active_inference.py   (~150 LOC)  Target-free E-step terms (callback)
├── efe.py                (~150 LOC)  Generation-time EFE policy
├── trainer.py            (~300 LOC)  Training loop with E/M-step separation
└── train_vfe.py          (~150 LOC)  Click-to-run training script
```

### VFEConfig

A clean dataclass with ~25 fields organized in five semantic groups (structure, E-step dynamics, covariance mode, gauge geometry, training), compared to the 60+ fields in `BlockConfig`. No `em_mode` selector, no DEQ/hebbian/closed-form/implicit-EM fields. The flag `use_obs_in_vfe` does not exist because the E-step has no observation term — Law 1 is architecturally enforced.

### BeliefState

A `NamedTuple` carrying the Gaussian belief triple:

```python
class BeliefState(NamedTuple):
    mu: torch.Tensor       # (B, N, K)
    sigma: torch.Tensor    # (B, N, K) diagonal or (B, N, K, K) full
    phi: torch.Tensor      # (B, N, n_gen)
```

This flows through the entire pipeline as the single data type.

### VFEPriorBank

Gauge-fixed orbit parameterization (Section 2). Learnable parameters: `base_mu` $(K)$, `base_log_sigma` $(K)$, `phi_embed` $(V, n_{\mathrm{gen}})$. Encode maps token IDs to initial beliefs via $A_v = \exp(\psi_v \cdot G)$. Decode produces logits $-\mathrm{KL}(q^\star \| \pi_v)/\tau$ for all $V$ tokens using fused block-diagonal KL kernels. No separate `nn.Linear` output projection — Law 3 is enforced by using the same PriorBank for both ends.

### VFEPositionalEncoding

Learnable `pos_phi` $(N_{\max}, n_{\mathrm{gen}})$. At `bch_order=1`: simple addition $\phi + p$. At `bch_order \geq 2$: full BCH via `lie_compose_bch_general_torch`.

### Attention Module

Stateless functions, no class. `compute_kl_attention` produces $\beta_{ij}$ from current beliefs via fused block-diagonal KL. `aggregate_beliefs` computes transported message means and covariances in either mixture or precision mode. RoPE is applied to $\mu$ when enabled.

### VFEEStep — The Critical Module

Replaces the 2,893-line `variational_ffn.py` with a single clear inner loop (~350 LOC):

```
for t in range(n_e_steps):
    1. Compute transport and KL attention (beta recomputed each iteration)
    2. Compute VFE gradients: F = α·KL(q||p) + λ_align·Σβ·KL + λ_soft·C(q,φ)
    3. Optional: add active inference gradients (injected callback)
    4. Natural gradient projection: nat_grad_mu = Σ @ grad_mu
    5. Mean update: mu -= η_μ · nat_grad_mu
    6. SPD retraction for sigma
    7. Phi update with Killing form preconditioning
return BeliefState(mu, sigma, phi)
```

Law 1 enforcement: the `forward` signature takes `(beliefs, priors, mask, active_inference_fn)` — there is no `targets` parameter, no `W_out`, no observation gradient. Target leakage is structurally impossible.

### VFEBlock, VFEStack, VFEModel

Each `VFEBlock` runs the E-step and applies normalization (MahalanobisNorm or RMSNorm). The `VFEStack` chains $L$ blocks with cross-layer prior handoff: $\mu_p^{(\ell+1)} = \mu^{(\ell)\star}$ (attached for straight-through gradient flow), $\Sigma_p^{(\ell+1)} = \Sigma_{\mathrm{embedding}}$ (frozen). The `VFEModel` wires everything together:

$$\texttt{token\_ids} \xrightarrow{\texttt{prior\_bank.encode}} \texttt{beliefs} \xrightarrow{\texttt{pos\_enc}} \texttt{beliefs} \xrightarrow{\texttt{stack}} \texttt{beliefs} \xrightarrow{\texttt{norm}} \texttt{beliefs} \xrightarrow{\texttt{prior\_bank.decode}} \texttt{logits}$$

### Active Inference Integration

Active inference does not contaminate the E-step code. It is injected as a callback function:

```python
ai_fn = VFEActiveInference(cfg, prior_bank) if cfg.active_inference else None
beliefs = self.stack(beliefs, priors, mask, active_inference_fn=ai_fn)
```

The callback computes pragmatic (entropy minimization) and epistemic (BALD MI) gradients, which are added to `grad_mu` and `grad_sigma` before Fisher projection — all terms share one information-geometric descent per Amari 1998.

Generation-time EFE (Section 11) lives in a separate `efe.py` module, accessed only during autoregressive decoding.

### Training

The trainer implements the clean E-step/M-step split:

```python
logits = model(token_ids)                              # E-step: context-only inference
loss = F.cross_entropy(logits.view(-1, V), targets)    # M-step objective
loss.backward()                                         # Gradients flow through E-step
optimizer.step()
```

Parameter groups with per-type learning rates: prior means ($3\times$), prior covariances ($0.15\times$), gauge frames ($0.3\times$), E-step learnable params ($0.03\times$).

---

## 18. Import Architecture

The `transformer/vfe/` package has a strict one-way dependency: it imports stateless math functions from `transformer/core/` and `math_utils/`, but nothing in `core/` imports from `vfe/`.

**Imported from `core/`** (all stateless, no `BlockConfig` dependency):

| Source | Functions | Used by |
|--------|-----------|---------|
| `core.gauge_utils` | `stable_matrix_exp_pair`, `fused_block_matrix_exp_pairs`, `fused_block_diagonal_kl_diag`, `fused_block_diagonal_kl_full` | prior_bank, attention |
| `core.vfe_gradients` | `compute_vfe_gradients_gpu`, `compute_natural_gradient_gpu` | e_step |
| `core.vfe_utils` | `retract_sigma_e_step`, `_retract_phi` | e_step |
| `core.gauge_preconditioner` | `build_killing_form_preconditioner`, `apply_killing_form_natural_gradient` | e_step |
| `core.transport_ops` | RoPE helpers | attention |
| `core.blocks` | `MahalanobisNorm`, `RMSNorm` | block |
| `core.expected_free_energy` | `compute_risk`, `compute_ambiguity` | efe |
| `math_utils.generators` | `generate_so3_generators`, `generate_soN_generators`, `generate_glK_generators`, `lie_compose_bch_general_torch` | model, positional |

**Re-implemented cleanly** (not imported from `core/`): `BlockConfig`, `GaugeTransformerLM`, `GaugeTransformerBlock/Stack`, `VariationalFFNDynamic`, `PriorBank`, `IrrepMultiHeadAttention`, and all EM mode machinery.

---

## 19. Implementation Sequence and Verification

### Phases

| Phase | Files | Depends on |
|-------|-------|-----------|
| 1. Foundation | `config.py`, `types.py`, `__init__.py` | nothing |
| 2. Embedding | `prior_bank.py`, `positional.py` | Phase 1 + `core.gauge_utils` |
| 3. Core | `attention.py`, `e_step.py` | Phase 1 + `core.{vfe_gradients, vfe_utils, gauge_preconditioner}` |
| 4. Assembly | `block.py`, `stack.py`, `model.py` | Phases 1--3 |
| 5. Extensions | `active_inference.py`, `efe.py` | Phase 4 + `core.expected_free_energy` |
| 6. Training | `trainer.py`, `train_vfe.py` | Phases 4--5 |

### Verification Checklist

1. Model constructs and forward pass runs without error on random token IDs.
2. `loss.backward()` produces non-zero gradients on all `prior_bank` parameters.
3. Gauge invariance: apply random $h$ to all beliefs at one position, verify $\beta_{ij}$ unchanged.
4. Sandwich product: assert `sigma_transported == Omega @ sigma @ Omega.T` in attention transport.
5. Law 1 enforcement: verify `VFEEStep.forward` signature has no `targets` parameter (static check).
6. Training smoke test: 100 steps on synthetic gauge data, verify loss decreases.
7. Comparison: same config on both `VFEModel` and `GaugeTransformerLM` produces similar loss curves.
