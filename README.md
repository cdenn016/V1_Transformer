# Gauge-Theoretic Transformer

A research framework implementing gauge-covariant variational free energy (VFE) minimization for language modeling. This codebase accompanies the manuscript:

> **Attention, Transformers, and Backpropagation are Degenerate Limits of the Variational Free Energy Principle**
> Robert C. Dennis

**Key result:** A single-layer VFE transformer with no MLPs or neural network components
achieves **61 test perplexity** on WikiText-103 (BPE-2 tokenization) with K=90, GL(15),
6 heads, RoPE, and sequence length 128 after 1 epoch of training and test PPL = 24 on wiki-ja. All representational
capacity comes from iterative variational free energy minimization over belief tuples (μ, Σ, φ). The E-step *is* the computation: it is not a learned feed-forward network.

A separate **Pure VFE** mode eliminates autograd and backpropagation entirely---the model is a prior bank updated by natural-gradient M-steps.


## Core Thesis

Language is a dynamic informational system: speakers encode and decode beliefs under uncertainty, and language models learn the statistical structure of this process. The mathematical framework natural to such systems---gauge-covariant variational free energy minimization over communicating agents on a statistical fiber bundle---explains why attention mechanisms work.

Standard transformer attention, the `1/sqrt(d_k)` scaling, layer normalization, and backpropagation all emerge as consequences of a single variational principle. The standard attention rule

```
beta_ij = softmax(Q_i K_j^T / sqrt(d_k))
```

is recovered as a **degenerate limit** of the gauge-theoretic attention

```
beta_ij = softmax(-D_KL(q_i || Omega_ij q_j) / tau)
```

through two simplifications: isotropic covariances and flat bundle.


## Architecture

```
        N(0, 1/(2·wd))  ← Level 3: hyper-prior (weight decay on embeddings)
                │
                │ wd·||θ_embed||²
                ▼
s_i = self.prior_mu[i]  ◄──── γ·KL(s_i||Ω_ij·s_j) ────► s_j
(position MODEL,                    (model coupling)
 slow timescale)
                │                                          ← Level 2: priors (M-step)
                │ p = w·π_token + (1-w)·s
                ▼
          p_i (PRIOR)
                │
                │ α·KL(q||p)
                ▼                                          ← Level 1: beliefs (E-step)
     q_i (beliefs, fast)  ◄──── β_ij·KL(q_i||Ω_ij·q_j) ────► q_j
                                      (attention)
```

Each agent (token) carries a Gaussian belief `q_i = N(mu_i, Sigma_i)` in a local gauge frame `phi_i`. The full VFE follows the standard FEP hierarchy `h -> s -> p -> q -> observations`:

```
F = sum_i  alpha D_KL(q_i || p_i)                         [self-coupling: beliefs to priors]
  + sum_i  lambda_h D_KL(s_i || h)                        [hyper-prior: models to centroid]
  + sum_ij [ beta_ij  D_KL(q_i || Omega_ij q_j)
            + tau beta_ij  log(beta_ij / pi_ij) ]         [belief coupling + attention entropy]
  + sum_ij [ gamma_ij D_KL(s_i || Omega_ij s_j)
            + tau gamma_ij log(gamma_ij / pi^(s)_ij) ]    [model coupling + meta entropy]
  - E_q[log p(o | x)]                                     [observation likelihood]
```

The `tau beta_ij log(beta_ij / pi_ij)` term is the attention-distribution entropy with prior `pi_ij` (uniform `1/N` over valid positions in code). It is constitutive of the variational stationarity that produces the softmax β below — minimizing a pure linear functional of β on the simplex would give a delta function, not a softmax. Manuscript reference: `\label{eq:free_energy_functional_final}` in `Attention/Participatory_it_from_bit.tex`.

Attention weights emerge as the row-Lagrangian stationary point of F:

```
beta_ij = softmax_j(-D_KL(q_i || Omega_ij q_j) / tau)
```

where `Omega_ij = exp(phi_i) exp(-phi_j)` is the gauge transport between agents and `tau = kappa * sqrt(K)` is the effective softmax temperature (κ is a learnable scalar; the √K factor normalizes the KL magnitude across head dimensions). **No W_Q, W_K, W_V projections are used**---attention arises from the geometry of belief distributions.


### Forward Pass (Actual Code Flow)

```
Input: token_ids
    ↓
Embeddings: (μ_q, σ_q, φ) ← GaugeTokenEmbedding(token_ids)
    ↓
Position Encoding: φ ← φ_token + φ_pos  [μ unchanged]
    ↓
FOR EACH LAYER:
    ├─ ATTENTION SUBLAYER (optional: if skip_attention=False):
    │   ├─ PreNorm: μ̃ ← LayerNorm(μ)
    │   ├─ Per-head KL-attention:
    │   │   β_ij = softmax(−KL(q_i || Ω_ij[q_j]) / κ√K)
    │   ├─ Per-head message aggregation:
    │   │   μ_agg = Σ_j β_ij · Ω_ij @ μ_j
    │   ├─ Output projection: μ_out = W_O · concat(μ_agg_h)
    │   └─ Residual: μ ← μ + μ_out
    │
    └─ VFE E-STEP SUBLAYER:
        ├─ PreNorm: μ̃ ← LayerNorm(μ)
        ├─ E-step iterations (t = 1...T):
        │   ├─ Recompute β_ij from current beliefs q^(t)
        │   │   (attention sublayer's β is discarded)
        │   ├─ Compute VFE gradient decomposition:
        │   │   ∇F = α·∇KL(q||p)           [self-coupling]
        │   │      + λ_β·Σ_j β·∇KL_ij        [alignment]
        │   │      + λ_sm·Σ_j ∂β/∂μ·KL_ij     [softmax coupling] λ_β = λ_sm = 1 
        │   ├─ Natural gradient: Δμ = −η · Σ · ∇F
        │   └─ Update: q^(t+1) ← q^(t) + Δμ
        │
        └─ Residual: μ ← μ + μ_final

Final LayerNorm → Output Projection → Logits
```

**Architectural note: the attention sublayer is optional, not mathematically required.** The theory (Algorithm 1) identifies each layer with a single E-step iteration where attention (β computation) and belief updates are unified. The E-step recomputes β internally and discards any attention-sublayer β. The attention sublayer's only forward-pass contribution is the message-aggregation residual (μ ← μ + W_O · μ_agg), which approximates one step of the alignment gradient but uses a learned projection W_O instead of the Fisher metric and lacks step-size control. Setting `skip_attention=True` recovers the pure VFE architecture (clean E-step + M-step, no separate attention layer).

**When `skip_attention=True` requires care: `em_mode` selection.** The detaching EM modes `em_phi_p`, `em_phi_q`, and `implicit_ift` (rows with σ_p detached or φ detached or both — see the EM modes table below) detach σ_p and/or φ inside the FFN's E-step. In those modes the attention sublayer is the *sole* autograd path back to `sigma_embed` and `phi_embed`. Combining any of them with `skip_attention=True` will silently freeze σ_embed and φ_embed at initialization — Σ gets zero variance across tokens, φ/Ω cluster overlap completely in token visualizations. Use `em_mode='straight_through'` (default) or `em_mode='ift_phi'` for a clean `skip_attention=True` configuration where the FFN's E-step itself provides the M-step gradients for σ and φ. `BlockConfig.__post_init__` warns when an incompatible combination is detected.


## Training Modes

The framework supports three training modes that span the spectrum from standard deep learning to fully analytic variational inference:

| Mode | Architecture | Learning | Entry Point |
|------|-------------|----------|-------------|
| `VFE_dynamic` (default) | GaugeTransformerLM | Autograd + EM dynamics | 
| `standard` | StandardTransformerLM | Standard backprop baseline | 
| `pure_fep` | PureVFETransformer | No autograd; analytic natural gradient |

**VFE_dynamic** is the default gauge-covariant mode: attention weights β recompute at each VFE iteration, and belief updates follow natural gradient descent on the full free energy. **Standard** provides a dot-product attention + MLP baseline for controlled comparison. **Pure FEP** is the most radical: no `nn.Module`, no `loss.backward()`, no optimizer---the model is a bank of Gaussian priors updated by analytic natural gradient M-steps.


## Theoretical Framework

### GL(K) Gauge Invariance (Theorem 1.1)

The KL divergence possesses maximal gauge symmetry: it is invariant under the full general linear group GL(K), not merely orthogonal subgroups.

```
D_KL(Omega_* P || Omega_* Q) = D_KL(P || Q)    for any invertible Omega in GL(K)
```

The (det Omega)^2 factors cancel in the log-determinant ratio. This means:
- Transport operators need only be **invertible**, not orthogonal
- No expensive re-orthogonalization is needed
- Learned projections W_Q, W_K in standard transformers **are themselves gauge transformations**, with `Omega = W_Q W_K^T` serving as a learned gauge transport

### Three Limits Recovering Standard Attention

| Limit | What it discards | Result |
|---|---|---|
| **1. Isotropic covariances** | Non-isotropic Σ_i → σ²I | KL reduces to geometric bias S(Ω) + squared Euclidean distance |
| **2. Constant gauge** | Position-dependent Ω_ij → constant Ω | Global gauge, no curvature; S(Ω) becomes a shared constant |
| **3. Learned projections** | σ⁻² Ω⁻ᵀ absorbed into W_Q W_K^T | Standard Q, K, V projections |

After all three limits: `beta_ij = softmax(Q_i K_j^T / sqrt(d_k))` --- standard transformer attention.

**Important:** Limit 1 alone does not suffice---the isotropic KL under general transport is:

```
D_KL = S(Ω_ij) + (1/2σ²) ||Ω_ij⁻¹ μ_i - μ_j||²
```

where S(Ω) = ½[log det(ΩΩᵀ) + Tr((ΩΩᵀ)⁻¹) - K] is the **geometric bias**. This vanishes if and only if Ω ∈ O(K) (orthogonal group), since ΩΩᵀ = I implies S = 0.

### The Nonlinearity

Standard transformer: GELU(x) — ad hoc nonlinearity.

Ours: ∂β_{ij}/∂θ — emerges from differentiating softmax attention:

```
β_{ij} = softmax(-KL_{ij} / κ)

∂β_{ij}/∂μ_i = -β_{ij} · [∂KL_{ij}/∂μ_i - Σ_k β_{ik} · ∂KL_{ik}/∂μ_i] / κ
∂β_{ij}/∂Σ_i = -β_{ij} · [∂KL_{ij}/∂Σ_i - Σ_k β_{ik} · ∂KL_{ik}/∂Σ_i] / κ
∂β_{ij}/∂φ_i = -β_{ij} · [∂KL_{ij}/∂φ_i - Σ_k β_{ik} · ∂KL_{ik}/∂φ_i] / κ
```

The **most general form** of the theory---no simplifying limits taken. Full non-isotropic covariances, non-trivial gauge transport, KL-divergence attention. **No MLPs, activation functions, learned W_Q/W_K/W_V, or positional encodings.** Only a linear output projection (from K dimensions to 50k vocabulary) is retained.

### Hierarchical Bayesian Structure

The full VFE defines a four-level hierarchy, each regularized by the level above:

```
Level 3:  N(0, 1/(2*wd))                          [hyper-prior — weight decay]
              |
              | wd * ||theta_embed||^2
              v
Level 2:  p_i = N(mu_p, Sigma_p), phi             [priors — learned embeddings (M-step)]
              |
              | alpha * KL(q || p)
              v
Level 1:  q_i = N(mu_q, Sigma_q)                  [beliefs — inferred per forward pass (E-step)]
              |
              | -E_q[log p(x | mu)]
              v
Level 0:  x_i                                      [observations — token targets]
```

**Level 3 (hyper-prior)** is implemented by AdamW weight decay on embedding parameters.
In a standard transformer, embeddings are lookup tables conventionally excluded from
weight decay. In the gauge transformer, embeddings are **statistical parameters**---means
`mu_p`, covariances `Sigma_p`, and gauge frames `phi`---that directly enter KL divergences,
matrix exponentials, and VFE gradients.

### Multi-Timescale Dynamics

The free energy naturally separates into:
- **Fast (E-step)**: Belief inference `dq_i/dt = -eta_q dF_fast/dq_i` --- what transformers do in a forward pass
- **Slow (M-step)**: Model learning `ds_i/dt = -eta_s dF_slow/ds_i` --- what backpropagation updates

Standard transformers operate in the adiabatic limit: slow variables frozen during inference, updated between passes.

### Block-Diagonal Multi-Head (Default)

Standard multi-head attention restricts the full GL(d_k) to a block-diagonal subgroup:

```
G_multi-head = GL(d_head)^H  subset  GL(d_k)
```

Each head learns an independent GL(d_head) gauge transformation, and the H factors commute. For compact groups like SO(3), irreducible representations yield non-uniform head dimensions (1, 3, 5, 7, ...) with intrinsic geometric meaning. The block-diagonal KL decomposition (`use_block_diagonal_kl=True`) exploits this structure for an O(K · d_head) reduction in transport memory.

### Cross-Head Coupling (Sparse gl(K) Subspace)

Setting `cross_couplings = [(a₁, b₁), (a₂, b₂), ...]` enables sparse off-diagonal gauge mixing between selected head pairs. The basis is no longer GL(d_head)^H but a sparse matrix subspace of gl(K):

```
g_coupled = ⨁_h gl(d_head)_h  ⊕  span{ E_{ij}^{(a,b)} : (a,b) in cross_couplings }
```

where E_{ij}^{(a,b)} is the elementary matrix that maps head a's row subspace into head b's column subspace. Pairs are directional; include both (a,b) and (b,a) for symmetric coupling.

Mathematical caveats users should know:

* **The basis is generically not closed under the matrix commutator.** With couplings `[(0,1),(1,2)]`, the commutator [E^{(0,1)}, E^{(1,2)}] lives in an E^{(0,2)} block that is absent from the basis. The bracket and Baker-Campbell-Hausdorff routines in `math_utils/generators.py` (`glK_bracket_torch`, `glK_compose_bch_torch`) silently project commutators back onto the basis via Frobenius inner product, so composition on a non-closed basis is a projected approximation rather than an exact Lie composition. Use `validate_generator_closure(generators)` to detect this; the model emits a startup warning by default (suppress via `validate_cross_head_closure=False`).

* **Opt-in pure path.** Setting `auto_close_cross_head_basis=True` calls `close_under_brackets`, which iteratively appends the bracket residuals (via thin-SVD deflation) to obtain a true Lie subalgebra. This changes `phi_dim`, breaks checkpoint compatibility, and may add generators that span across the user-supplied super-block partition; see the in-line warning emitted at model init.

* **Super-block factorization for the KL.** Heads that are transitively connected by `cross_couplings` are merged into super-blocks (`merge_coupled_heads`) and contiguousized (`reorder_cross_head_generators`). The block-diagonal KL machinery then operates on these super-blocks. As a consequence, per-block parameters in attention — including the learnable κ_h (`log_kappa_per_head`) — are per-super-block in the coupled path, not per original head.

* **Validation cost.** `validate_generator_closure` runs in O(n_gen² · K³) once at startup. For K ≲ 128 this is sub-second; for large K it can take tens of seconds. Set `validate_cross_head_closure=False` to skip the check on production runs whose coupling pattern you have already verified.

* **Hard incompatibility.** `cross_couplings` non-empty with `use_block_diagonal_kl=False` raises at construction: the cross-head builder reorders generators into super-block coordinates that the per-head fallback path cannot honor.

* **Duplicates are dropped, orientation preserved.** Exact duplicate pairs are removed with a warning; (a,b) and (b,a) are treated as distinct (different generator blocks).

### O(K) Gauge Transport

The gauge transport Ω_ij = exp(φ_i)·exp(-φ_j) via matrix exponentials always produces det > 0. Under Newton-Schulz orthogonalization (`enforce_orthogonal=True`), this projects to SO(K). To achieve full O(K) transport, the `learnable_reflection` option introduces per-token sign vectors s_i ∈ {±1}^K:

```
Ω_ij = diag(s_i) · exp(φ_i) · exp(-φ_j) · diag(s_j)
```

covering both connected components of O(K). Sign vectors are applied at embedding time via `μ_i ← s_i ⊙ μ_i` with straight-through estimator for gradient flow.

### Non-Flat Transport and Holonomy

The default transport assumes a **flat bundle**: `Ω_ij = exp(φ_i)·exp(-φ_j)` satisfies the cocycle condition `Ω_ij · Ω_jk = Ω_ik` exactly, meaning parallel transport around any closed loop is trivial. The `GaugeConnection` module (`transformer/core/connection.py`) generalizes this by introducing edge-local Lie algebra elements `δ_ij`:

```
Ω_ij = exp(φ_i · G) · exp(α · δ_ij · G) · exp(-φ_j · G)
```

Two parameterizations are available:
- **Bilinear** (default): `δ_ij^a = μ_i^T W^a μ_j` --- one bilinear form per generator, parameter-efficient
- **MLP**: `δ_ij = MLP([μ_i; μ_j])` --- more expressive, higher memory

Both are **zero-initialized** so the model starts in the flat regime and learns curvature only where the data warrants it.

**Holonomy** measures the curvature of the learned connection. For a triangle `(i, j, k)`:

```
C_ijk = exp(δ_ij · G) · exp(δ_jk · G) · exp(δ_ki · G)
```

When the connection is flat, `C_ijk = I` for all triples. The Frobenius norm `‖C_ijk - I‖_F` quantifies deviation from flatness. A `holonomy_penalty_loss()` regularizer pushes the model toward flatness, controlled by `holonomy_penalty` in the config.

Holonomy analysis tools include per-snapshot statistics (`transformer/analysis/holonomy_metrics.py`), curvature-by-distance profiles, flatness trajectories over training, and visualization (`transformer/visualization/holonomy_plots.py`). A synthetic gauge language generator (`transformer/data/synthetic_gauge.py`) produces data with controlled holonomy for testing.

When the gauge group is SO(N) and the irrep specification contains a non-scalar irrep with multiplicity ≥ 2 (for example, `irrep_spec=[('fund', 2, 3)]`), the per-head connection cannot localize the shared generators to a single head block. The block-init path detects this via `partition_generators_by_block` and falls back to a single global `GaugeConnection` over the full embedding dimension, emitting a `RuntimeWarning` to record the choice. The output shape is identical to the per-head path so downstream transport is unaffected; the diagnostic attribute `block._connection_mode` is set to `'global'` rather than `'per_head'`.

### Gauge-Covariant Ridge Conditioning

Numerical regularization of small SPD matrices in attention defaults to a uniform Tikhonov ridge ε·I. This breaks gauge covariance because the identity does not transform as Σ → h Σ hᵀ under a local gauge change h ∈ GL(K). The flag `gauge_covariant_ridge=True` replaces the offending sites with ε·(g·gᵀ), where g = exp(φ) is the per-token gauge frame. The covariant ridge transforms correctly under all admissible local gauge changes and restores Σ_t → h Σ_t hᵀ exactness in the aggregate-messages and Cholesky-fallback paths. The flag is off by default to preserve checkpoint comparability for ablations; non-flat transport configurations correctly honor the flag because the producer caches now carry both `'Omega'` and `'exp_phi'` keys end-to-end.

### RoPE as Gauge Transport (Tri-State σ Rotation)

Rotary position embeddings restrict the GL(K) gauge group to SO(2)^{K/2}: a position-dependent rotation R(θ_i) acts on each pair of consecutive coordinates. The gauge interpretation prescribes that R must act on Gaussian beliefs by both μ → R μ AND Σ → R Σ Rᵀ, which is the standard sandwich product for covariance transport. The standard-transformer convention rotates only μ and leaves Σ raw, which preserves the diagonal-σ optimization at the cost of breaking strict GL(K) covariance.

The `rope_full_gauge` flag is a tri-state selector that lets the user trade off cost against geometric consistency:

| Mode | Attention σ | FFN VFE σ | Notes |
|------|-------------|-----------|-------|
| `'off'` (default) | raw | raw | Standard-transformer pattern; fastest |
| `'vfe_only'` | raw | R Σ Rᵀ | Full-gauge inside the FFN VFE iteration only; attention β values still come from the μ-only KL. Equivalent to the legacy `True` setting. |
| `'both'` | R Σ Rᵀ | R Σ Rᵀ | Strict GL(K) covariance through both sublayers. Requires `diagonal_covariance=False`; rejected at attention forward otherwise. |

Backwards compatibility is preserved: legacy boolean values are coerced (`True → 'vfe_only'`, `False → 'off'`) by `_coerce_rope_full_gauge` in `block_config.py`. Invalid strings or types raise at `BlockConfig` construction time. The `'both'` mode lifts diagonal σ to full covariance and applies `_apply_rope_to_covariance` before the per-head KL dispatch; the lifted full-cov σ flows through the existing per-head machinery without further changes.

Independently, the legacy code path that handled `use_rope=True` without a per-head `irrep_dims` decomposition has been replaced with a hard `ValueError`. That path skipped the rope chain-rule fix (using raw-μ gradients instead of R^T·∂KL_RoPE/∂(R μ)) and silently produced biased descent directions; users now receive an actionable error message pointing to either `irrep_dims` or `use_rope=False`.

### E-Step Concentration Correction

When `E_learnable_alpha=True`, the effective self-coupling weight α = c₀/(b₀ + KL) is itself a function of the belief parameters via the per-dimension KL. Differentiating α·KL with respect to (μ, σ) therefore carries a product-rule contribution −(α²/c₀)·KL_k·(dKL/dθ) in addition to the obvious α·dKL/dθ term. The VFE gradient kernel `compute_vfe_gradients_gpu` applies this correction whenever its `alpha_c0` argument is provided, and the `VFEEStep` module now passes `softplus(self.raw_c0)` through both the standard and rope-full-gauge per-head call sites. The same correction is injected into the autograd-based rope helper via a phantom term −0.5·(α²/c₀)·KL² on the self-energy whose autograd derivative reproduces the missing analytic contribution. Without these wirings the descent direction was biased by the omitted product-rule term whenever the Bayesian adaptive α was active.

### Phi Gradient Preconditioning

The gauge frame φ ∈ gl(K) requires geometric preconditioning because the backward pass through `matrix_exp` amplifies non-compact (symmetric) directions exponentially. Four modes:

| Mode | Method | Cost |
|------|--------|------|
| `'clip'` (default) | Norm clipping | O(n_gen) |
| `'cartan'` | Cartan decomposition, fixed sym dampening | O(n_gen²) |
| `'killing'` | Killing form metric (no free params) | O(n_gen²) |
| `'pullback'` | Full pullback through exp (exact Riemannian) | O(n_gen³/token) |

For GL(K): `'pullback'` is the theoretically exact natural gradient; `'killing'` is a cheap principled approximation; `'cartan'` is the pragmatic engineering choice. For SO(N) all modes are roughly equivalent (compact group).

### DEQ Implicit Differentiation (E-Step Backward)

The E-step iterates natural gradient VFE descent to a fixed point μ*, Σ*. **Deep Equilibrium (DEQ) implicit differentiation** replaces unrolled backprop with a single backward pass through the fixed-point equation using the implicit function theorem:

```
(I - J)^{-1} v ≈ v + Jᵀv + (Jᵀ)²v + ...    (Neumann series, K terms)
```

Enabled via `use_deq=True`. Provides O(1) memory in iterations vs O(n_iterations) for unrolled backprop.

### Symmetry Breaking

Without observations, the free energy defines a gauge-symmetric vacuum---all agents converge to identical beliefs modulo gauge orbit (analogous to an untrained network). Observations break this symmetry, driving agents toward specialized representations determined by training data. Learning is thus interpreted as explicit symmetry breaking.

### Pure VFE Transformer

The `PureVFETransformer` (`transformer/pure_vfe/model.py`) takes the variational principle to its logical conclusion: **no `nn.Module`, no autograd, no backpropagation**. The entire model is a prior bank---one Gaussian `N(μ_v, Σ_v)` per vocabulary token plus per-head gauge frames `Ω_v ∈ GL(K_h)`:

- **Forward pass = E-step**: Given token IDs, look up priors, then run VFE natural gradient descent to infer posterior beliefs `q_i`. Attention weights emerge from KL geometry exactly as in the autograd version.
- **Learning = M-step**: Update priors via analytic natural gradient on the free energy. No `loss.backward()`, no optimizer. Gauge frames update via the left-invariant natural gradient on GL(K):

```
ξ = Ωᵀ · ∂F/∂Ω          (pullback to Lie algebra)
ΔΩ = -η · Ω · clip(ξ)   (push forward with trust region)
```

Covariances update on the SPD manifold via retraction. The `PureVFEConfig` (`transformer/pure_vfe/config.py`) controls E-step iterations, trust regions, SPD safeguards, and gauge frame stability. Quick-start: edit the config in `run.py` and execute directly.


## Experimental Results

### BERT Validation (144 attention heads)

Quantitative comparison between gauge-aligned KL attention and standard dot-product attention on pretrained `bert-base-uncased`:

| Metric | Value |
|---|---|
| Optimal temperature | τ = 19.0 (theory: τ = 2√d = 16, 19% deviation) |
| Mean Pearson correlation | r = 0.821 |
| Median Pearson correlation | r = 0.889 |
| Heads with r > 0.8 | 68.1% |
| Heads with r > 0.9 | 49.3% |

**Key-norm bias prediction confirmed**: average ρ = -0.352 across all heads, significant in 92.4% of heads at p < 0.001. This gauge-theoretic prediction explains why layer normalization is a geometric necessity: it enforces constant key norms required for frame-independent inference.

### GL(K) Language Modeling (WikiText-103, vocab 50,257)

The most general form of the theory---no simplifying limits taken. Full non-isotropic covariances, non-trivial gauge transport, KL-divergence attention.

| Configuration | K | Layers | Gauge Mode | Train PPL | Test PPL | Parameters |
|---|---|---|---|---|---|---|
| **Best (1 epoch)** | **90** | **1** | **Ω ∈ GL(15)** | **59** | **61** | **~80M** |

For context: random-chance perplexity is ~50,000. The K=90, GL(15) configuration (6 heads, seq-length 128, RoPE) achieves **~600x improvement** over random chance with purely geometric attention.

**Emergent semantic structure**: Learned gauge frames develop interpretable categorical organization---punctuation, content words, and letters cluster separately in both belief space (μ) and gauge frame space (φ) without category supervision.


## Installation

**Requirements**: Python 3.9+ with CUDA-capable GPU (recommended)

```bash
git clone https://github.com/cdenn016/gauge-holonomy.git
cd gauge-holonomy
pip install torch numpy scipy numba matplotlib seaborn plotly networkx scikit-learn datasets tiktoken
```

### Dependencies

| Category | Packages |
|---|---|
| **Core** | PyTorch (>=2.0.0), NumPy, SciPy, Numba |
| **Training** | PyTorch Lightning, Weights & Biases (`wandb`) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Data** | HuggingFace `datasets`, `tiktoken` |
| **Analysis** | scikit-learn, NetworkX |
| **Optional** | SymPy (symbolic derivations), Triton (GPU kernels), CuPy (CUDA kernels), `transformers` (BERT diagnostics), PyMC (Bayesian analysis), SHAP, UMAP-learn |


## Usage

### Training

```bash
# PyTorch Lightning training (recommended — supports VFE_dynamic, standard, pure_fep)
python scripts/train_lightning.py

# Quick-start Pure VFE (edit config in file, then run)
python run.py

# Legacy training scripts
python transformer/train.py                                # Standard VFE training
python transformer/train_publication.py                    # Publication-quality with ablations
python transformer/resume_training.py --checkpoint path/to/model.pt
```

### Inference & Generation

```bash
# Interactive text generation with attention visualization
python generate.py

# Model inference and token probability analysis
python inference.py
```

### Analysis

```bash
# BERT validation (gauge-aligned KL vs dot-product attention)
python transformer_test.py

# RoBERTa diagnostics (key-norm CV, per-head temperatures)
python roberta_diagnostics.py

# RG flow analysis over training
python scripts/analyze_rg_flow.py

# Ablation suite (systematic hyperparameter sweeps)
python scripts/run_ablation_suite.py

# Interactive visualization (UMAP + Plotly + SHAP)
python scripts/run_interactive_viz.py

# Gauge frame spectral analysis (validates W_Q W_K^T = σ⁻² Ω⁻ᵀ on BERT/GPT-2)
python scripts/gauge_frame_spectral_analysis.py

# Bayesian RG exponent analysis
python scripts/rg_exponent_bayesian.py

# Generate publication figures
python scripts/generate_publication_figures.py
```

### Testing

```bash
pytest tests/                           # Full suite
pytest tests/ -v -k "attention"         # Specific tests
pytest tests/ -m "not slow"             # Skip slow tests
pytest tests/ -m gpu                    # GPU-only tests
```


## Project Structure

```
gauge-holonomy/
├── transformer/                    # Gauge-theoretic transformer implementation
│   ├── core/                       #   Core architecture
│   │   ├── model.py                #     GaugeTransformerLM (full language model)
│   │   ├── attention.py            #     KL-divergence multi-head attention
│   │   ├── blocks.py               #     Transformer blocks with gauge transport
│   │   ├── variational_ffn.py      #     VFE feedforward (E-step belief inference)
│   │   ├── embeddings.py           #     Token/positional embeddings → (μ, Σ, φ)
│   │   ├── prior_bank.py           #     Token-dependent prior distributions
│   │   ├── gauge_utils.py          #     matrix_exp, Newton-Schulz, fused KL
│   │   ├── gauge_preconditioner.py #     Riemannian φ preconditioning
│   │   ├── connection.py           #     GaugeConnection (non-flat transport, holonomy)
│   │   └── block_config.py         #     BlockConfig dataclass (60+ unified params)
│   ├── pure_vfe/                   #   Pure VFE transformer (no autograd)
│   │   ├── model.py                #     PureVFETransformer (prior bank architecture)
│   │   ├── train.py                #     Pure VFE training loop
│   │   ├── gauge.py                #     GL(K) transport and natural gradients
│   │   ├── config.py               #     PureVFEConfig dataclass
│   │   ├── inference.py            #     E-step belief inference
│   │   ├── learning.py             #     M-step natural gradient updates
│   │   ├── gaussians.py            #     Gaussian distribution utilities
│   │   └── cuda_ext.py             #     Optional CUDA kernel compilation
│   ├── analysis/                   #   Analysis & diagnostics
│   │   ├── publication_metrics.py  #     BPC, perplexity, statistical significance
│   │   ├── semantics.py            #     Clustering interpretation, emergent categories
│   │   ├── bayesian_validation.py  #     Bayesian validation of KL alignment
│   │   ├── holonomy.py             #     Holonomy computation and penalty loss
│   │   ├── holonomy_metrics.py     #     HolonomySnapshot/Profile, curvature analysis
│   │   └── trajectory.py           #     Belief trajectory tracking
│   ├── visualization/              #   Plotting utilities
│   │   ├── attention_viz.py        #     Attention heatmaps and KL plots
│   │   ├── belief_space_viz.py     #     Belief distribution visualization
│   │   ├── interactive_belief_viz.py #   UMAP + Plotly + SHAP visualizations
│   │   ├── holonomy_plots.py       #     Curvature diagnostics visualization
│   │   ├── training_plots.py       #     Training curves
│   │   ├── trajectory_plots.py     #     Belief evolution
│   │   ├── ablation_plots.py       #     Ablation comparisons
│   │   ├── belief_space_frequent.py #    Frequent token analysis
│   │   └── attention_context.py    #     Context-aware attention analysis
│   ├── data/                       #   Data loading
│   │   ├── datasets.py             #     WikiText-2/103, wiki-ja, OpenWebText
│   │   └── synthetic_gauge.py      #     Synthetic language with controlled holonomy
│   ├── training/                   #   Training infrastructure
│   │   ├── config.py               #     TrainingConfig (standard/VFE_dynamic/pure_fep)
│   │   ├── train_fast.py           #     Optimized training loop
│   │   ├── optimizer.py            #     Gauge-specific parameter grouping
│   │   ├── metrics.py              #     Training metrics
│   │   ├── lightning_module.py     #     GaugeTransformerLitModule (Lightning)
│   │   ├── lightning_pure_vfe.py   #     PureVFELitModule (Lightning)
│   │   ├── lightning_data.py       #     GaugeDataModule (Lightning)
│   │   └── holonomy_callback.py    #     Holonomy tracking callback
│   ├── baselines/                  #   Reference implementations
│   │   ├── standard_transformer.py #     StandardTransformerLM (dot-product + MLP)
│   │   └── flops_counter.py        #     FLOPs comparison (gauge vs standard)
│   ├── utils/                      #   Checkpoint, evaluation, testing utilities
│   ├── train.py                    #   Main training entry point
│   ├── train_publication.py        #   Publication-quality training with ablations
│   └── resume_training.py          #   Resume from checkpoint
├── math_utils/                     # Mathematical primitives
│   ├── generators.py               #   SO(3)/SO(N)/GL(K) Lie algebra generators
│   ├── transport.py                #   Parallel transport Ω_ij = exp(φ_i)·exp(-φ_j)
│   ├── push_pull.py                #   Gaussian pushforward under transport
│   └── numerical_utils.py          #   Stability helpers (regularization, clipping)
├── scripts/                        # Analysis and utility scripts
│   ├── train_lightning.py          #   Lightning training (3 modes: standard/VFE/pure_fep)
│   ├── run_ablation_suite.py       #   Systematic hyperparameter sweeps
│   ├── run_interactive_viz.py      #   Interactive UMAP + Plotly + SHAP visualization
│   ├── gauge_frame_spectral_analysis.py  # Spectral analysis (BERT/GPT-2 validation)
│   ├── generate_publication_figures.py   # Publication figures
│   └── kn5_baseline.py             #   KN5 baseline comparison
├── tests/                          # Test suite (18 test files)
├── Data/                           # Experimental data and baselines
│   └── Standard_Baselines/         #   Standard transformer baseline results
├── run.py                          # Click-to-run Pure VFE training
├── generate.py                     # Interactive text generation
├── inference.py                    # Model inference and analysis
├── transformer_test.py             # BERT validation (144 heads)
├── roberta_diagnostics.py          # RoBERTa analysis
├── claude.md                       # Architecture reference and code standards
├── dynamic_language_plan.md        # Diachronic language evolution plan
└── pytest.ini                      # Test configuration
```


## Numerical Stability

Key techniques enabling large gauge groups:

- **√K attention scaling**: `logits = -KL / (κ · √K)` prevents softmax saturation
- **Dimension-dependent KL clamping**: ceiling of `max(100, 5K)`
- **Per-parameter gradient clipping**: Independent budget per parameter group (μ, Σ, φ)
- **Loss √K normalization**: `loss = F / √K` for hyperparameter transferability
- **GL(K) transport**: No re-orthogonalization needed (only invertibility required). Optional O(K) enforcement via Newton-Schulz + learnable reflections


## Citation

```bibtex
@article{dennis2025attention,
  title={Attention, Transformers, and Backpropagation are Degenerate Limits
         of the Variational Free Energy Principle},
  author={Dennis, Robert C.},
  journal={Preprint},
  year={2025}
}
```

## License

This project is for research purposes. Contact: cdenn016@yahoo.com
