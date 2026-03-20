# Pure VFE Transformer — Implementation Plan

## Context

The GL(K) manuscript ("Attention as Gauge-Theoretic Variational Inference") proves that standard transformer attention is a degenerate limit of variational free energy minimization over a statistical fiber bundle. The existing codebase implements this with a hybrid approach: E-step VFE descent for inference, but still uses `nn.Module`, autograd, and Adam for learning.

**Goal:** Build a "pure VFE transformer" that removes ALL neural network machinery. No `nn.Module`, no `autograd`, no `loss.backward()`, no optimizer. The entire system — both inference AND learning — operates through natural gradient descent on the gauge-covariant VFE with analytic closed-form gradients. This is the purest possible realization of the free energy principle applied to sequence modeling.

**Why this matters:** If it achieves non-trivial perplexity, it proves that attention, depth, learning, and normalization all emerge from a single variational equation — not as engineering choices, but as mathematical consequences.

---

## Key Design Decisions

### 1. Full covariance Σ stored directly (not Cholesky)

Work with Σ ∈ SPD(K) throughout. The VFE and all its gradients are expressed in terms of Σ and Σ⁻¹ — this keeps gauge invariance manifest and the math transparent. SPD is maintained by the affine-invariant exponential retraction (eigendecomposition-based, manuscript Appendix D). No Cholesky factorization needed in the core loop.

### 2. Direct Ω_i ∈ GL(K) — no exp(φ) parameterization

Instead of storing Lie algebra coordinates φ_i and computing Ω_i = exp(φ_i), we store the group elements Ω_i ∈ GL⁺(K) directly. This eliminates:
- Matrix exponential computation
- dexp differential computation
- Lie algebra generators, structure constants
- Killing form / Cartan / pullback preconditioning

Transport: `Ω_ij = Ω_i · Ω_j⁻¹` — just matrix multiply + inverse.
Gradient: `∂F/∂Ω_i` via standard matrix calculus through KL formula.
Natural gradient: left-translate to identity via `ΔΩ_i = -η · Ω_i · (Ω_i⁻ᵀ · ∂F/∂Ω_i)`.
Invertibility: maintained by small steps from identity initialization.

The cocycle condition `Ω_ij · Ω_jk = Ω_ik` is still automatically satisfied since `Ω_i · Ω_j⁻¹ · Ω_j · Ω_k⁻¹ = Ω_i · Ω_k⁻¹`.

### 3. M-step only observation gradient

E-step minimizes belief VFE (prior + alignment terms only). Observation gradient (∂CE/∂params) enters exclusively in M-step when updating priors. Clean EM separation.

### 4. Tiny scale for fast iteration

K=32, N=64, H=4 heads of dimension 8. Full Σ costs 32²=1024 floats per token. At batch_size=8, N=64: ~0.5M floats for all covariances — fits easily.

---

## Architecture Overview

**The "model" is a prior bank** — one Gaussian N(μ_v, Σ_v) per vocabulary token, plus gauge frames Ω_v ∈ GL(K_h) per head and positional gauge offsets.

**"Forward pass" = E-step:** Initialize beliefs from priors, run VFE descent via natural gradient for n_steps iterations. Returns logits via KL-based decoding.

**"Learning" = M-step:** Update prior bank parameters via natural gradient on the marginal VFE. No backprop — all gradients are analytic.

---

## Files to Create

All new code under `transformer/pure_vfe/`:

### 1. `transformer/pure_vfe/config.py` — Configuration

```python
@dataclass
class PureVFEConfig:
    vocab_size: int = 50257
    belief_dim: int = 32         # K: full belief dimension
    n_heads: int = 4             # H: number of heads (block-diagonal)
    head_dim: int = 8            # K/H per head
    n_esteps: int = 12           # E-step iterations (replaces "depth")
    tau: float = 2.83            # Attention temperature (√head_dim)
    eta_E: float = 0.1           # E-step learning rate
    eta_M: float = 0.001         # M-step learning rate
    alpha_b0: float = 1.0        # Prior precision regularizer
    alpha_c0: float = 1.0        # Prior precision regularizer
    hyper_var: float = 1.0       # Hyper-prior variance
    max_seq_len: int = 64        # Sequence length
    sigma_init: float = 1.0      # Initial covariance scale
    omega_init_scale: float = 0.01  # Initial gauge frame perturbation
    trust_region_mu: float = 1.0
    trust_region_sigma: float = 0.3
    trust_region_omega: float = 0.3
    causal: bool = True
    device: str = "cuda"
```

### 2. `transformer/pure_vfe/gaussians.py` — Analytic Gaussian Operations

All KL divergences and their exact gradients for **full covariance Σ** (stored directly as SPD matrices). No autograd.

**Covariance representation:** Store Σ ∈ SPD(K) directly. SPD is maintained by the affine-invariant exponential retraction on the SPD manifold. All VFE formulas use Σ and Σ⁻¹ natively — no Cholesky conversion.

Key functions:
- `kl_gaussian(mu_p, Sigma_p, mu_q, Sigma_q)` → KL(P||Q) for full Gaussians
  - `= ½[tr(Σ_q⁻¹ Σ_p) + (μ_q - μ_p)ᵀ Σ_q⁻¹ (μ_q - μ_p) - K + ln(det Σ_q / det Σ_p)]`
- `kl_transported(mu_i, Sigma_i, mu_j, Sigma_j, Omega_ij)` → KL(q_i || Ω_ij · q_j)
  - Transported q_j: N(Ω·μ_j, Ω·Σ_j·Ωᵀ)
  - Uses identity: (ΩΣΩᵀ)⁻¹ = Ω⁻ᵀΣ⁻¹Ω⁻¹
- `grad_kl_mu_i(mu_i, Sigma_i, mu_j, Sigma_j, Omega_ij)` → ∂KL/∂μ_i
  - `= (ΩΣ_jΩᵀ)⁻¹(μ_i - Ω·μ_j)` (from Eq. 21 of main paper)
- `grad_kl_Sigma_i(Sigma_i, Sigma_j, Omega_ij)` → ∂KL/∂Σ_i
  - `= ½[-Σ_i⁻¹ + (ΩΣ_jΩᵀ)⁻¹]` (from Eq. B.7 of supplementary)
- `natural_grad_sigma(grad_Sigma, Sigma)` → -2·Σ·sym(∂F/∂Σ)·Σ
  - The Fisher-Rao natural gradient on SPD(K) (manuscript Appendix D)
- `retract_spd(Sigma, delta_Sigma, step_size)` → affine-invariant exp map
  - Eigendecompose Σ = V Λ Vᵀ
  - Whiten: B = Λ⁻¹/² Vᵀ δΣ V Λ⁻¹/²
  - Clip Frobenius norm of B (trust region)
  - Diagonalize B = U Λ_B Uᵀ
  - Retract: Σ_new = V Λ¹/² U exp(τ·Λ_B) Uᵀ Λ¹/² Vᵀ
  - Enforce spectral floor ε_min and condition cap κ_max
- `grad_beta_mu_i(beta, kl_vals, grad_kl_mu, tau)` → softmax correction ∂β/∂μ_i (Eq. 24)
  - `= -(β_ij/τ)[∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i]`
- `kl_to_prior_bank(mu, Sigma, prior_mu, prior_Sigma)` → [B,N,V] logits for decoding

**Memory at K=32, N=64:** 64 tokens × 32² = 65K floats per batch element for Σ.

### 3. `transformer/pure_vfe/gauge.py` — Direct GL(K) Gauge Transport

**No Lie algebra parameterization.** Store Ω_i ∈ GL⁺(K_h) per head directly.

Key functions:
- `compute_transport(Omega_i, Omega_j)` → Ω_ij = Ω_i · Ω_j⁻¹
  - Just: `Omega_i @ torch.linalg.inv(Omega_j)` or `Omega_i @ torch.linalg.solve(Omega_j, I)`
- `transport_gaussian(mu, Sigma, Omega_ij)` → (Ω·μ, Ω·Σ·Ωᵀ)
- `grad_kl_Omega_i(mu_i, Sigma_i, mu_j, Sigma_j, Omega_i, Omega_j)` → ∂F/∂Ω_i
  - Chain rule through Ω_ij = Ω_i · Ω_j⁻¹ and KL formula
  - ∂KL/∂Ω_ij from manuscript: involves transported precision × mean diff + trace term
  - ∂Ω_ij/∂Ω_i = I ⊗ Ω_j⁻¹ (Kronecker product, or equivalently: ∂(Ω_i M)/∂Ω_i[ab] = e_a M_b)
- `natural_grad_omega(grad_Omega, Omega)` → left-invariant natural gradient on GL(K)
  - `= Ω · (Ω⁻ᵀ · ∂F/∂Ω)` — left-translate Euclidean gradient to identity
  - This is the bi-invariant metric natural gradient on GL(K)
- `init_omega(V, K_h, scale)` → V gauge frames initialized near identity
  - `Ω_v = I + scale · randn(K_h, K_h)` ensuring det > 0

**No imports from existing gauge code** — this is much simpler than the exp(φ) path.

### 4. `transformer/pure_vfe/inference.py` — E-Step (The Forward Pass)

```python
@torch.no_grad()
def e_step(token_ids, model, config):
    """
    Run VFE descent to convergence. This IS the forward pass.
    E-step minimizes belief VFE only (prior + alignment terms).
    Observation gradient enters ONLY in M-step.

    Returns: (mu, Sigma, Omega, logits, vfe_history)
    """
    B, N = token_ids.shape
    K_h = config.head_dim
    H = config.n_heads

    # Initialize beliefs from priors
    mu    = model.prior_mu[token_ids].clone()              # [B, N, K]
    Sigma = model.prior_Sigma[token_ids].clone()           # [B, N, K, K]

    # Initialize per-head gauge frames (from prior + positional offset)
    # Omega_i per head: [B, N, H, K_h, K_h]
    Omega = model.prior_Omega[token_ids].clone()           # [B, N, H, K_h, K_h]
    Omega = Omega @ model.pos_Omega[:N]                    # Compose with positional

    vfe_history = []
    for step in range(config.n_esteps):
        # --- Per-head block-diagonal processing ---
        # Reshape mu into heads: [B, N, H, K_h]
        mu_h = mu.view(B, N, H, K_h)
        Sigma_h = extract_block_diag(Sigma, H, K_h)       # [B, N, H, K_h, K_h]

        # 1. Compute pairwise transport: Ω_ij = Ω_i · Ω_j⁻¹  [B, H, N, N, K_h, K_h]
        Omega_ij = compute_pairwise_transport(Omega)

        # 2. Compute KL divergences for all pairs (full covariance per head)
        kl_ij = compute_pairwise_kl(mu_h, Sigma_h, Omega_ij)  # [B, H, N, N]

        # 3. Attention weights (with causal mask)
        beta = masked_softmax(-kl_ij / config.tau, causal=config.causal)  # [B, H, N, N]

        # 4. State-dependent prior precision
        kl_prior = kl_to_own_prior(mu, Sigma, token_ids, model)  # [B, N]
        alpha = config.alpha_c0 / (config.alpha_b0 + kl_prior)   # [B, N]

        # 5. Analytic VFE gradients (no observation term!)
        grad_mu    = vfe_grad_mu(mu_h, Sigma_h, Omega_ij, beta, alpha, model, token_ids)
        grad_Sigma = vfe_grad_Sigma(Sigma_h, Omega_ij, beta, alpha, model, token_ids)
        grad_Omega = vfe_grad_Omega(mu_h, Sigma_h, Omega, Omega_ij, beta)

        # 6. Natural gradient preconditioning
        #    Mean:  Δμ = -η · Σ · ∂F/∂μ (Fisher-Rao for Gaussian mean)
        nat_mu = torch.einsum('...ij,...j->...i', Sigma, grad_mu)  # [B, N, K]

        #    Covariance: ΔΣ = -2Σ · sym(∂F/∂Σ) · Σ (SPD natural gradient)
        nat_Sigma = natural_grad_sigma(grad_Sigma, Sigma_h)  # [B, N, H, K_h, K_h]

        #    Gauge: ΔΩ = Ω · (Ω⁻ᵀ · ∂F/∂Ω) (left-invariant on GL(K))
        nat_Omega = natural_grad_omega(grad_Omega, Omega)     # [B, N, H, K_h, K_h]

        # 7. Trust-region clipping + update
        mu    = mu - config.eta_E * clip_norm(nat_mu, config.trust_region_mu)
        Sigma = retract_spd_blocks(Sigma, nat_Sigma, config.eta_E, config)
        Omega = Omega - config.eta_E * clip_norm(nat_Omega, config.trust_region_omega)

        vfe_history.append(compute_vfe(mu_h, Sigma_h, Omega_ij, beta, alpha, model, token_ids))

    # Decode: logit_v = -KL(q_i || π_v)
    logits = -kl_to_prior_bank(mu, Sigma, model.prior_mu, model.prior_Sigma)

    return mu, Sigma, Omega, logits, vfe_history
```

### 5. `transformer/pure_vfe/learning.py` — M-Step (Parameter Updates)

```python
@torch.no_grad()
def m_step(token_ids, targets, mu_star, Sigma_star, Omega_star, model, config):
    """
    Update prior bank via natural gradient on marginal VFE.
    Observation gradient enters HERE (not in E-step).
    """
    B, N = token_ids.shape

    # 1. Observation gradient (analytic softmax-CE)
    logits = -kl_to_prior_bank(mu_star, Sigma_star, model.prior_mu, model.prior_Sigma)
    # ∂CE/∂logits = softmax(logits) - one_hot(targets) — exact, no autograd
    ce_grad = softmax_ce_gradient(logits, targets)  # [B, N, V]

    # 2. For each vocabulary token seen in batch, compute & apply gradient
    unique_v = token_ids.unique()
    for v in unique_v:
        mask = (token_ids == v)                          # [B, N] boolean
        n_v = mask.sum()
        if n_v == 0:
            continue

        Sigma_v = model.prior_Sigma[v]                   # [K, K]
        Sigma_v_inv = safe_inv(Sigma_v)                  # [K, K]

        # --- Prior mean gradient ---
        # ∂KL(q*||p_v)/∂μ_v = -Σ_v⁻¹(μ* - μ_v) (summed/averaged over occurrences)
        mu_diff = mu_star[mask] - model.prior_mu[v]      # [n_v, K]
        grad_mu_v = -(Sigma_v_inv @ mu_diff.mean(0))     # [K]
        grad_mu_v += model.prior_mu[v] / config.hyper_var # Hyper-prior pull

        # Observation gradient: ∂CE/∂μ_v through KL decoding
        # logit_v = -KL(q||π_v), ∂(-KL)/∂μ_v = Σ_v⁻¹(μ_q - μ_v)
        # So ∂CE/∂μ_v = Σ_{i} ce_grad[i,v] · Σ_v⁻¹(μ*_i - μ_v)
        ce_weights = ce_grad[:, :, v][mask]               # [n_v]
        obs_grad_mu = Sigma_v_inv @ (ce_weights.unsqueeze(-1) * mu_diff).mean(0)
        grad_mu_v += obs_grad_mu

        # Natural gradient: Δμ_v = -η · Σ_v · ∂F/∂μ_v
        model.prior_mu[v] -= config.eta_M * (Sigma_v @ grad_mu_v)

        # --- Prior covariance gradient ---
        # ∂KL(q*||p_v)/∂Σ_v = ½[Σ_v⁻¹ - Σ_v⁻¹·E[Σ*]·Σ_v⁻¹ - Σ_v⁻¹·E[Δμ·Δμᵀ]·Σ_v⁻¹]
        Sigma_star_avg = Sigma_star[mask].mean(0)         # [K, K]
        outer_avg = (mu_diff.unsqueeze(-1) * mu_diff.unsqueeze(-2)).mean(0)
        grad_Sigma_v = 0.5 * (Sigma_v_inv - Sigma_v_inv @ (Sigma_star_avg + outer_avg) @ Sigma_v_inv)
        grad_Sigma_v += 0.5 * Sigma_v_inv / config.hyper_var  # Hyper-prior

        nat_grad_Sv = natural_grad_sigma(grad_Sigma_v, Sigma_v)
        model.prior_Sigma[v] = retract_spd(Sigma_v, nat_grad_Sv, config.eta_M)

        # --- Prior gauge frame gradient ---
        grad_Omega_v = compute_prior_Omega_gradient(Omega_star[mask], model.prior_Omega[v])
        nat_grad_Ov = natural_grad_omega(grad_Omega_v, model.prior_Omega[v])
        model.prior_Omega[v] -= config.eta_M * nat_grad_Ov

    # 3. Update positional gauge offsets
    update_pos_Omega(Omega_star, token_ids, model.pos_Omega, config)
```

### 6. `transformer/pure_vfe/model.py` — Main Model Class

```python
class PureVFETransformer:
    """
    Pure variational free energy transformer.
    No nn.Module. No autograd. No backprop.
    """
    def __init__(self, config: PureVFEConfig):
        self.config = config
        K = config.belief_dim
        H = config.n_heads
        K_h = config.head_dim
        V = config.vocab_size
        dev = config.device

        # Prior bank (THE model — raw tensors, not nn.Parameters)
        self.prior_mu = torch.randn(V, K, device=dev) * 0.02
        self.prior_Sigma = torch.eye(K, device=dev).unsqueeze(0).expand(V, -1, -1).clone()
        self.prior_Sigma *= config.sigma_init

        # Gauge frames: Ω_v per head, stored as K_h × K_h matrices near identity
        self.prior_Omega = torch.eye(K_h, device=dev).unsqueeze(0).unsqueeze(0).expand(
            V, H, -1, -1).clone()
        self.prior_Omega += torch.randn(V, H, K_h, K_h, device=dev) * config.omega_init_scale

        # Positional gauge: Ω_pos per head (composed with token gauge)
        self.pos_Omega = torch.eye(K_h, device=dev).unsqueeze(0).unsqueeze(0).expand(
            config.max_seq_len, H, -1, -1).clone()
        self.pos_Omega += torch.randn(config.max_seq_len, H, K_h, K_h, device=dev) * config.omega_init_scale

    def forward(self, token_ids):
        """Returns logits. Inference IS VFE descent."""
        mu, Sigma, Omega, logits, vfe = e_step(token_ids, self, self.config)
        return logits

    def update(self, token_ids, targets):
        """Update priors. Learning IS M-step natural gradient."""
        mu, Sigma, Omega, logits, vfe = e_step(token_ids, self, self.config)
        m_step(token_ids, targets, mu, Sigma, Omega, self, self.config)
        return logits

    def save(self, path):
        torch.save({
            'prior_mu': self.prior_mu, 'prior_Sigma': self.prior_Sigma,
            'prior_Omega': self.prior_Omega, 'pos_Omega': self.pos_Omega,
            'config': self.config,
        }, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path)
        model = cls(data['config'])
        model.prior_mu = data['prior_mu']
        model.prior_Sigma = data['prior_Sigma']
        model.prior_Omega = data['prior_Omega']
        model.pos_Omega = data['pos_Omega']
        return model
```

### 7. `transformer/pure_vfe/train.py` — Training Loop

```python
def train(config):
    model = PureVFETransformer(config)
    dataset = load_wikitext2(seq_len=config.max_seq_len)

    for epoch in range(n_epochs):
        for step, batch in enumerate(dataset):
            tokens = batch['input_ids'].to(config.device)
            targets = batch['targets'].to(config.device)

            logits = model.update(tokens, targets)

            # Log (CE is computed for monitoring only — not used for gradients)
            with torch.no_grad():
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
                ppl = torch.exp(loss)
            if step % 10 == 0:
                print(f"epoch {epoch} step {step}: loss={loss:.3f} ppl={ppl:.1f}")
```

### 8. `transformer/pure_vfe/tests/test_gradients.py` — Gradient Validation

Finite-difference checks for every analytic gradient:
- `test_kl_grad_mu`: ∂KL/∂μ vs finite diff
- `test_kl_grad_Sigma`: ∂KL/∂Σ vs finite diff (symmetric perturbations)
- `test_kl_grad_Omega`: ∂KL/∂Ω vs finite diff
- `test_vfe_grad_mu`: full VFE gradient including softmax correction
- `test_gl_invariance`: KL(G·P || G·Ω·Q) = KL(P || Ω·Q) for random G ∈ GL(K)
- `test_e_step_monotonic`: VFE decreases at each E-step
- `test_spd_retraction`: Σ stays SPD after retraction

---

## Existing Code to Reuse

| File | What to Import | Purpose |
|------|---------------|---------|
| `transformer/data/datasets.py` | `WikiTextDataset` | Data loading |
| `transformer/core/variational_ffn.py` | `_safe_spd_inv` | Robust SPD inversion |

**Note:** No gauge_utils, no generators, no gauge_preconditioner — the direct-Ω approach doesn't need any of this.

## Implementation Order

1. **`config.py`** — All hyperparameters
2. **`gaussians.py`** — KL with full Σ + all analytic gradients + SPD retraction
3. **`gauge.py`** — Direct GL(K) transport (Ω_i · Ω_j⁻¹), ∂F/∂Ω, natural gradient on GL(K)
4. **`tests/test_gradients.py`** — Validate ALL gradients against finite differences first!
5. **`inference.py`** — E-step loop
6. **`learning.py`** — M-step updates with observation gradient
7. **`model.py`** — PureVFETransformer class
8. **`train.py`** — Training loop
9. **Run on WikiText-2** — First perplexity numbers

## Verification

1. **Gradient correctness:** Every analytic gradient matches finite-difference to relative error < 1e-4
2. **GL(K) invariance:** KL invariant under simultaneous gauge transform (to machine precision)
3. **E-step monotonicity:** VFE strictly decreases at each iteration
4. **SPD preservation:** Σ stays positive-definite throughout all E-steps
5. **Non-trivial perplexity:** Significantly below random chance (~V ≈ 50k) on WikiText-2

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Slow per-step | K=32, N=64 is tiny; profile hot paths for vectorization |
| M-step instability | Conservative η_M, trust regions, hyper-prior regularization |
| Full-cov KL decoding O(V·K²) | Batch over vocab chunks; K=32 → 1024 per pair — feasible |
| SPD retraction cost O(K³) | K=32 → 32³≈33K flops — trivial |
| Ω_i becoming singular | Init near identity, small steps, monitor det(Ω_i) |
| E-step non-convergence | Monitor VFE; adaptive step size; early stopping on plateau |
| M-step obs gradient chain | CE grad is softmax-one_hot; chain through KL is tractable matrix calc |
