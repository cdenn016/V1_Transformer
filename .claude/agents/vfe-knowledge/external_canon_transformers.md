# External Canon — Attention, Transformers, Geometric Deep Learning, Manifold Optimization

**Status:** source of truth for both agents. The user's manuscripts and codebase are evaluated *against* these standard treatments.

Citations resolve to `external_bibliography.md`.

**Citation hygiene.** Section numbers and equation labels (e.g., `[Vaswani2017 §3.2.1]`) are best-effort pointers; verify before citing the specific number in a finding. When in doubt, cite only the source tag (`[Vaswani2017]`).

---

## 1. Scaled dot-product attention — standard form

[Vaswani2017]

```
Attention(Q, K, V)  =  softmax( Q Kᵀ / √d_k ) V
```

where:
- `Q ∈ ℝ^{N × d_k}` queries from input X via learned `W_Q ∈ ℝ^{d_model × d_k}`.
- `K ∈ ℝ^{N × d_k}` keys from input X via learned `W_K ∈ ℝ^{d_model × d_k}`.
- `V ∈ ℝ^{N × d_v}` values from input X via learned `W_V ∈ ℝ^{d_model × d_v}`.
- `√d_k` is the dimension-scaling factor in the denominator; **the standard justification** [Vaswani2017 §3.2.1] is that for unit-variance Q, K, the dot product `Q_i · K_j` has variance d_k, so dividing by √d_k restores unit variance and prevents the softmax from saturating.

### Multi-head attention
```
MHA(X)  =  Concat( head_1, ..., head_h ) W_O
head_i  =  Attention(X W_Q^i, X W_K^i, X W_V^i)
```
where each head uses its own `W_Q^i, W_K^i, W_V^i ∈ ℝ^{d_model × d_k}` with `d_k = d_model / h`.

### The KQV interpretation
- Q ↔ "what am I looking for"
- K ↔ "what do I offer"
- V ↔ "what I contribute if matched"
- Softmax(QKᵀ/√d_k) ↔ assignment matrix (rows sum to 1).

### What is *not* in the standard transformer
- A variational-inference interpretation. The standard transformer is heuristic; the variational interpretation is the user's contribution (and others' — see §3 below for alternative interpretations).
- A gauge structure. Standard transformers have no notion of frame, fiber bundle, or parallel transport.
- A free-energy functional. Standard transformers are trained by minimizing cross-entropy (or other downstream losses), not by minimizing a free-energy F.

The user's framework attempts to **derive** these as a structure on top of attention. Whether the derivation is sound, and whether standard attention is recovered as a limit, is what the agents are checking.

## 2. The √d_k scaling — what's standard, what's user-specific

**Standard:** `1/√d_k` from [Vaswani2017]. Justified by variance argument.

**User's framework:** `τ = κ √K` where κ is a *learnable* hyperparameter and K is the per-head dimension. The √K factor is the standard scaling (with K playing the role of d_k). The κ is user-introduced — it's a learnable scalar on top of the standard scaling.

**What to verify:** when a manuscript writes `τ = κ √K`, it should label κ as the user's addition (a learned temperature), not as part of the standard. The agent should flag any claim that the standard √d_k scaling "is" the user's κ √K — they are related but the κ is extra.

## 3. Alternative interpretations of attention

The user's "attention as gauge-theoretic variational inference" is *one* of several mathematical interpretations of attention. Others include:

- **Kernel interpretation** [Tsai2019]: softmax attention approximates a kernel `K(x, y) = exp(x · y / √d)`. Different kernels give different attention. Linear attention [Katharopoulos2020] uses an explicit feature map to linearize.
- **Hopfield interpretation** [Ramsauer2021]: attention implements the update rule of modern (continuous-state) Hopfield networks; the energy is a log-sum-exp of dot products.
- **Geometric deep learning** [Bronstein2021]: attention is permutation-equivariant message passing on the complete graph; group-equivariant variants exist for graphs with symmetries.
- **Predictive coding** [Millidge2021]: under specific conditions, attention can implement predictive coding's prediction-error minimization.

**None of these are uniquely "the" interpretation.** When the user's manuscript claims attention "is" their formulation, the claim should be qualified: "we present a gauge-theoretic interpretation of attention; we are not claiming this is the unique mathematical structure underlying attention." The user's recent commits already soften some such claims; the agent should help maintain this discipline.

## 4. Layer normalization

[BaKirosHinton2016]

```
LayerNorm(x)_i  =  γ_i · (x_i - μ) / √(σ² + ε)  +  β_i
```
where μ and σ² are the mean and variance computed across the feature dimension of x.

The user's framework claims LN is "the geometric condition for frame-independent inference." This is a specific gauge-theoretic interpretation. The standard interpretation is heuristic (stabilizes training, reduces covariate shift). Whether the user's derivation actually shows LN follows from gauge invariance is what the agent must check — and whether the *direction* of derivation (gauge invariance → LN) is mathematically tight or just suggestive.

## 5. Positional encoding

[Vaswani2017]: sinusoidal positional encoding `PE(pos, 2i) = sin(pos / 10000^{2i/d})`, `PE(pos, 2i+1) = cos(...)`.

[Su2024RoPE]: rotary position embedding (RoPE) — multiplies Q and K by position-dependent rotation matrices. The dot product `(R_m q)ᵀ(R_n k) = qᵀ R_{n-m} k` depends only on relative position.

**Standard claim:** RoPE preserves the inner product structure under translation (`R_n` is orthogonal, so `(R_m q)ᵀ(R_m k) = qᵀ k`). RoPE rotates Q and K but does *not* rotate V.

**Relevance to the user's framework:** the user's `MahalanobisNorm(μ, σ)` divides μ by σ. If RoPE rotates μ (because μ is the "query/key analog") but not σ, then the Mahalanobis ratio mixes rotated and unrotated quantities. This is the documented RoPE×Mahalanobis gap. **Standard RoPE does not have this issue because standard RoPE does not normalize by a separate σ — the user's framework introduces this combination.** Verify the user discloses this limitation; do not flag it as a violation of standard RoPE (it isn't — it's a user-framework specific issue).

## 6. Causal masking

Standard causal mask: set logits to −∞ for positions j > i before softmax. Implements left-to-right attention. The user's framework interprets this as a "non-uniform attention prior π_j": π_j = 0 for j > i is equivalent to the −∞ logit shift (the contribution to softmax is `log π_j + (other terms)`).

**Verify:** the user's "non-uniform prior" interpretation matches the standard "additive bias" interpretation. They should be algebraically equivalent (since `softmax(x + log π) ∝ π · softmax_input(x)`).

## 7. Geometric / gauge-equivariant deep learning

[Bronstein2021, CohenGeigerKohlerWelling2018]

Gauge-equivariant networks: a class of architectures where the layers commute with the action of a gauge group on the data. Standard examples:
- **Spherical CNNs** [CohenEtAl2018]: convolutions on the sphere equivariant under SO(3).
- **Gauge-equivariant message passing** [HaanWeilerCohen2020]: networks on manifolds with chosen gauge.

The user's framework is in this tradition but specifically claims `GL(K)` (general linear) as the gauge group of attention, not a compact group like SO(K). This is a less common choice — `GL(K)` is non-compact and the metric on it (Killing form etc.) is not positive-definite. The agent should flag whether the user has addressed this: how is the natural-gradient metric chosen on GL(K)?

## 8. Optimization on manifolds — retractions and trust regions

[AbsilMahonySepulchre2008]

A **retraction** on a manifold M is a smooth map `R: T M → M` satisfying:
1. `R_p(0) = p`.
2. `dR_p(0) = id_{T_p M}` (preserves tangent directions to first order).

The exponential map is a canonical retraction on a Lie group: `R_p(v) = p · exp(v)`. Other retractions (cheaper to compute) also exist, e.g., Cayley retractions for orthogonal groups.

For **SPD matrices** (positive-definite covariance matrices):
- Affine-invariant metric: `g_P(X, Y) = tr(P⁻¹ X P⁻¹ Y)`. Geodesics are matrix exponentials; the exponential retraction is `R_P(X) = P^{½} exp(P^{-½} X P^{-½}) P^{½}`.
- Log-Cholesky parameterization: parameterize `P = L Lᵀ` with L lower-triangular positive-diagonal; optimize on the log of the diagonal entries.
- Bures–Wasserstein metric [BhatiaJainLim2019]: a different Riemannian metric on SPD related to optimal transport.

For **diagonal SPD** (σ > 0), the simplest retraction is multiplicative: `σ_new = σ · exp(δ)` for unconstrained δ. This is what the user's framework uses for σ. This is standard.

**Trust regions** on manifolds: standard practice is to clamp the tangent step before the retraction. The user's `E_sigma_q_trust` clamps `δσ/σ` (the whitened tangent). This is the affine-invariant scaling — standard.

## 9. Natural gradient on Lie groups

[Amari1998, AbsilMahonySepulchre2008]

For a parameter θ ∈ G (a Lie group), the natural gradient with respect to the Fisher metric on the induced statistical manifold is:
```
∇̃_θ L  =  g⁻¹(θ) ∇L
```
where g is the Fisher matrix in the chosen parameterization. For a left-invariant metric on G, this can be computed in the Lie algebra: `∇̃ L|_θ = θ · (g_e⁻¹ Lᵀ_θ ∇L)` where g_e is the metric at the identity.

For φ ∈ gl(K) (Lie algebra elements), the user's framework should specify which metric on gl(K) it uses. Common choices:
- **Frobenius inner product:** `⟨A, B⟩ = tr(AᵀB)`. Positive-definite. Standard for the matrix space.
- **Killing form:** `⟨A, B⟩_K = tr(ad_A ad_B)`. Sign-indefinite on non-compact groups; degenerate on solvable Lie algebras. **Not positive-definite on gl(K).**

The user's `gauge_preconditioner.py` should make this explicit. **The agent should check which metric is used and whether it's positive-definite. Using the Killing form on a non-compact Lie algebra without justification is a standard pitfall [AbsilMahonySepulchre2008].**

## 10. Pitfalls the agents must check for

1. **"Standard transformer is a special case."** The user claims this; the derivation must show that under the stated limits (isotropic Gaussian, flat connection, etc.), the user's β reduces to softmax(QKᵀ/√d_k). The agent must check whether the derivation is tight or has hidden assumptions.
2. **`W_Q W_Kᵀ = σ⁻² Ω⁻ᵀ`** (a specific claim in `GL(K)_attention.tex`). This is an *identification* between learned projections and gauge transformations. The agent should check whether the identification is unique (it isn't — `W_Q W_Kᵀ` is rank-deficient in general; many `Ω` satisfy this).
3. **κ as "the same as" 1/√d_k.** They are related (κ multiplies the standard scaling); they are not equivalent. κ is learnable; √d_k is dimensional.
4. **LN derived from gauge invariance.** Verify the derivation is tight or qualify the claim.
5. **RoPE×Mahalanobis as a "gauge violation."** Standard RoPE doesn't have a separate σ; the issue is in the user's combined framework. Frame the finding accordingly.
6. **GL(K) as gauge group without addressing non-compactness.** Killing form is indefinite; what metric is used on the Lie algebra?
7. **Identifying attention with a unique mathematical structure.** Multiple interpretations exist; "is" claims should be qualified.
8. **Exponential map surjectivity on GL⁺(K).** Not surjective; some matrices have no real log. The user's framework parameterizes a subset of GL⁺(K) via `exp(φ_i) exp(−φ_j)`. The reachable subset depends on the φ's.
9. **Single-layer scaling vs SOTA scaling.** A user's manuscript (`Participatory_it_from_bit.tex`) honestly notes that their PPL ≈ 73 at K=120 is above WikiText-103 multi-layer baselines (~18–25). Preserve this honesty; the scaling exponent claim is within the single-layer architecture, not vs SOTA.
10. **Conflating learned attention with derived attention.** The user's GL(K) gauge transformer has *no* learned QKV projections — attention emerges from the KL on gauge-transported beliefs. Standard transformers have learned QKV. When comparing, ensure the comparison is apples-to-apples.

---

## Acknowledgments to other references mentioned in the manuscripts

The user's manuscripts cite many papers (Bahdanau, Hinton, Tsai, Ramsauer, Bronstein, Cohen, etc.). For the agent's purposes, the *short list* in `external_bibliography.md` is the working set. For deeper checks, the agent can WebFetch arxiv preprints by ID.
