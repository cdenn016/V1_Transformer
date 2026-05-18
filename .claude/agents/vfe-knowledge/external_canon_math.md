# External Canon — Information Geometry, Differential Geometry, Gauge Theory

**Status:** source of truth for both agents. The user's manuscripts and codebase are evaluated *against* these standard treatments, not the other way around.

Citations resolve to `external_bibliography.md`.

**Citation hygiene.** Section numbers and equation labels appearing below (e.g., `[Nakahara2003 §10.3]`, `[AmariNagaoka2000 Ch. 2]`) are best-effort pointers. Before citing a *specific* section or equation number in a finding, verify it against the actual source via WebFetch (for papers) or by asking the user (for books not available online). When in doubt, cite only the source tag (`[Nakahara2003]`) and let the reader locate the relevant section.

---

## 1. Information geometry

### Statistical manifold
A statistical manifold is a parameterized family of probability distributions `{p(x; θ) : θ ∈ Θ}` treated as a smooth manifold with `θ` as coordinates [AmariNagaoka2000].

### Fisher information metric
The Fisher information matrix at point `θ` is
```
g_ij(θ) = E_{p(x;θ)} [ ∂_i log p · ∂_j log p ]
        = -E_{p(x;θ)} [ ∂_i ∂_j log p ]    (under regularity)
```
This is a Riemannian metric on the statistical manifold.

**Cencov's uniqueness theorem [Cencov1972]:** the Fisher metric is the *unique* (up to scalar) Riemannian metric on a statistical manifold that is invariant under sufficient statistics. Any metric that fails this invariance is not a valid information metric.

### KL divergence
```
KL(q ‖ p) = ∫ q(x) log(q(x)/p(x)) dx     [continuous]
          = Σ_x q(x) log(q(x)/p(x))      [discrete]
```
Properties [KullbackLeibler1951, AmariNagaoka2000 Ch. 2]:
- `KL ≥ 0`, with equality iff `q = p` a.e.
- **Asymmetric:** `KL(q ‖ p) ≠ KL(p ‖ q)` in general.
- Not a metric (no triangle inequality, asymmetric). It is a *divergence*.
- KL is the second-order expansion of the Fisher metric at infinitesimal `q ≈ p`: `KL ≈ ½ (Δθ)ᵀ g(θ) (Δθ) + O(Δθ³)`.
- Undefined when `q(x) > 0` and `p(x) = 0` (or returns `+∞`).

### Closed-form KL between Gaussians

For `q = N(μ_q, Σ_q)`, `p = N(μ_p, Σ_p)`, both K-dimensional:
```
KL(q ‖ p) = ½ [ tr(Σ_p⁻¹ Σ_q) + (μ_p − μ_q)ᵀ Σ_p⁻¹ (μ_p − μ_q) − K + log(|Σ_p|/|Σ_q|) ]
```
For diagonal Σ (write `σ²` for diagonal entries):
```
KL(q ‖ p) = ½ Σ_k [ log(σ_p,k²/σ_q,k²) + (σ_q,k² + (μ_q,k − μ_p,k)²)/σ_p,k² − 1 ]
```
This is textbook [BleiKuckelbirgJordan2017, KingmaWelling2014 Appendix B]. Any derivation that ends in a different closed form for Gaussian KL is wrong unless it explicitly reparameterizes (e.g., uses precision instead of covariance).

### Dual connections
On a statistical manifold there are two natural torsion-free connections, the **exponential (e-) connection** and the **mixture (m-) connection**, dual with respect to the Fisher metric [AmariNagaoka2000 Ch. 3]. Exponential families are *e-flat*; mixture families are *m-flat*. Flatness under one connection does not imply flatness under the other.

### Natural gradient [Amari1998]
The natural gradient of a loss `L(θ)` on a statistical manifold is
```
∇̃ L = g(θ)⁻¹ ∇L
```
where `g` is the Fisher matrix. Properties:
- Invariant under reparameterization (this is the *defining* property).
- Steepest descent in the Fisher–Riemannian sense.
- Adam, RMSProp, etc. are not natural gradient — they precondition with empirical second-moment estimates, not the Fisher [Amari1998 §4].

For Lie-algebra-valued parameters (e.g., φ ∈ gl(K)), the appropriate metric depends on the structure — *do not assume Euclidean*. Common choices: Killing form (for semisimple Lie algebras), invariant inner product, or task-specific Fisher computed on the induced statistical family.

---

## 2. Differential geometry

### Smooth manifolds and tangent spaces
A smooth K-manifold M has at each point p a tangent space `T_p M ≅ ℝ^K`. Standard reference [Lee2013].

### Bundles
- **Fiber bundle:** `π : E → M` with typical fiber F, locally `E ≅ U × F`.
- **Vector bundle:** F is a vector space and transition maps are linear.
- **Principal G-bundle:** F = G (Lie group), right action of G on E preserving fibers, `π(eg) = π(e)` for all g ∈ G.

Standard references: [Nakahara2003 Ch. 9–10], [KobayashiNomizu Vol. I Ch. I–II], [Frankel2011].

### Connection on a principal bundle
A connection assigns at each point e ∈ P a horizontal subspace `H_e ⊂ T_e P` complementary to the vertical subspace. Equivalently, a g-valued 1-form `A ∈ Ω^1(P; g)` satisfying:
1. `A(ξ*) = ξ` for fundamental vector fields ξ* of ξ ∈ g.
2. `R_g* A = Ad(g⁻¹) A` (equivariance under the right action).

The **curvature** `F = dA + ½[A, A]` measures the failure of flatness. Curvature = 0 iff connection is flat (locally trivializable as a constant bundle).

### Parallel transport
Given a curve γ: [0,1] → M and a connection on P, the parallel transport `P_γ : π⁻¹(γ(0)) → π⁻¹(γ(1))` is the unique horizontal lift. For an associated vector bundle E = P ×_ρ V (with representation ρ: G → GL(V)), parallel transport acts as `ρ(P_γ) ∈ GL(V)`.

### Transport of tensors — the sandwich identity (THIS IS THE STANDARD)

For a **type-(k, ℓ)** tensor `T ∈ V^⊗k ⊗ V*^⊗ℓ`, parallel transport by `g ∈ G` (acting via ρ) transforms T as
```
T_new = ρ(g)^⊗k ⊗ ρ(g⁻¹)*^⊗ℓ T
```
**Special cases the agent must recognize:**

- **Vector (k=1, ℓ=0):** `v → ρ(g) v`. One-sided.
- **Covector (k=0, ℓ=1):** `α → α ρ(g⁻¹) = α ρ(g)⁻¹`. One-sided, inverse.
- **(0,2)-tensor / bilinear form / covariance matrix (k=0, ℓ=2):** `T → ρ(g⁻¹)*ᵀ T ρ(g⁻¹) = ρ(g⁻ᵀ) T ρ(g⁻¹)`. **Two-sided sandwich.** For the orthogonal group, ρ(g⁻¹) = ρ(g)ᵀ and this reduces to `T → ρ(g)ᵀ T ρ(g)`. For general GL(K) with ρ the defining representation, the sandwich keeps the (0,2) character but uses inverses.
- **(2,0)-tensor (like a covariance pushed forward, e.g., inverse covariance under a covariant push):** `T → ρ(g) T ρ(g)ᵀ`.

The user's framework names "covariance transport" as `Σ_new = Ω Σ Ωᵀ`. This corresponds to treating Σ as a (2,0)-tensor under the GL(K) action with Ω = ρ(g). If Σ is instead identified as a (0,2)-tensor in the formalism, the correct rule is `Σ_new = Ω⁻ᵀ Σ Ω⁻¹`. **The agent must check which identification the user is making and verify consistency.** Whichever side is chosen, the sandwich (two-sided conjugation) is the standard for bilinear forms.

References: [Nakahara2003 §10.3, §11.1], [KobayashiNomizu Vol. I §III.2], [Frankel2011 Ch. 17].

### Lie algebra exponential
For a matrix Lie group G with Lie algebra g, the exponential map is `exp(X) = I + X + X²/2! + ...`. Properties:
- `exp(A + B) = exp(A) exp(B)` **only when [A, B] = 0**.
- In general, the Baker–Campbell–Hausdorff (BCH) formula applies: `log(exp(A) exp(B)) = A + B + ½[A,B] + (1/12)([A,[A,B]] − [B,[A,B]]) + ...`.
- So `exp(φ_i) exp(−φ_j) = exp(φ_i − φ_j + ½[φ_i, −φ_j] + ...) ≠ exp(φ_i − φ_j)` in general.
- The product `exp(A) exp(−B)` parameterizes a subset of G (the "two-exponential" parameterization). It is *not* surjective onto all of G in general; surjectivity depends on G being exponential (e.g., compact, or simply connected nilpotent).
- For **GL⁺(K, ℝ)** (orientation-preserving real GL): the exponential map is *not* surjective. The standard counterexample is a Jordan block with a negative eigenvalue: `J = [[−1, 1], [0, −1]] ∈ GL⁺(2, ℝ)` (since `det(J) = 1 > 0`) has no real logarithm. (Note: `−I_2 ∈ GL⁺(2, ℝ)` *does* have a real logarithm — `−I_2 = exp(π · S)` where `S = [[0, −1], [1, 0]]` — so a diagonal-with-negative-eigenvalues matrix is not in itself the obstruction; the obstruction is the non-diagonalizable case [Hall *Lie Groups, Lie Algebras, and Representations*, Ch. on the matrix exponential]). The two-exponential product `exp(φ_i) exp(−φ_j)` parameterizes a subset of GL⁺(K) that depends on the φ's; surjectivity onto GL⁺(K) is not guaranteed and the user's framework should not assume it.

References: [Frankel2011 Ch. 14, 17], [KobayashiNomizu Vol. I §1.5].

### Gauge invariance vs gauge equivariance
- **Gauge invariant:** a quantity `Q` is invariant under the gauge action: `Q(g · s) = Q(s)`.
- **Gauge equivariant:** a quantity transforms covariantly: `Q(g · s) = ρ(g) Q(s)` for some representation ρ.

Predictions/observables are typically gauge invariant. Internal representations are typically gauge equivariant. Conflating these is a common error.

### Holonomy
Around a closed loop γ at point p, parallel transport gives `Hol(γ) ∈ G`. The holonomy group `Hol_p(A) ⊂ G` measures how far the connection is from flat. For a flat connection, `Hol_p(A) = {e}` for contractible loops (but may be non-trivial for non-contractible loops, giving the monodromy).

---

## 3. Standard pitfalls the agents must check for

These are mistakes the standard literature warns against. Findings can cite "this is the [Nakahara2003 §10.3 pitfall]" or similar.

1. **One-sided conjugation on a bilinear form.** `Σ → Ω Σ` or `Σ → Σ Ω` is wrong for a (0,2) or (2,0) tensor. Must be two-sided.
2. **Missing transpose in the sandwich.** `Σ → Ω Σ Ω` (without `T`) is wrong unless Ω is orthogonal.
3. **`exp(A + B) = exp(A) exp(B)` assumed without commutation.** Violates BCH.
4. **Confusing covariant and contravariant indices.** A natural test: under change of frame R, does the index transform as R or R⁻ᵀ?
5. **Treating Adam/RMSProp as natural gradient.** They are *not* — they precondition with empirical second moments, not Fisher [Amari1998].
6. **Symmetrizing KL.** `½(KL(q‖p) + KL(p‖q))` loses Cencov-invariance. Jensen-Shannon does too. These are not Fisher-canonical.
7. **Assuming flatness without checking curvature.** "Trivially vanishing holonomy" must be justified by curvature = 0 or by the connection class being globally trivializable.
8. **Conflating Lie algebra dimension with manifold dimension.** dim(g) = dim(G) but typically `dim(G) ≠ dim(M)` for the base manifold of a G-bundle.
9. **Using natural gradient with the wrong Fisher.** The Fisher on the *parameter manifold* is not the same as Fisher on the *distribution manifold* unless the parameterization is invertible and smooth.
10. **Equating "scaled dot-product attention" with "kernel attention".** They are equivalent in form but the kernel interpretation only holds when softmax is replaced or linearized [Tsai2019, Katharopoulos2020].

---

## 4. What a "novel" claim vs a "standard" claim looks like

When the agent finds a user-defined object or operation:

- If it matches a standard treatment (e.g., the user's KL formula matches Amari Ch. 2): **standard-consistent**. No flag.
- If it differs in form but reduces to standard under appropriate limits (e.g., multi-agent F that reduces to single-agent F when N=1): **principled extension**. The reduction should be derived in the manuscript or in code. Flag if the reduction is asserted but not shown.
- If it differs and does *not* reduce to standard, or claims standard-equivalence without proof: **novel claim**. The manuscript must label it as novel and provide independent justification. Asserting standard-equivalence without proof is the error to flag.

The agent's job is to make this distinction visible — *not* to reject novel claims, but to require that they be labeled and justified appropriately.
