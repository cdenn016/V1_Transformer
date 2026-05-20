# Evidence Pack — subclaim-B-degenerate-sigma

Neutral fact pack. Both teams work from this file.

## Manuscript references

### Section header (`Attention/GL(K)_attention.tex:1024`)

> "\paragraph{Deterministic Beliefs via Scaled Limit.}"

The paragraph header names the operation as a "scaled limit" and the resulting beliefs as "deterministic." Whether either descriptor is accurate is the subject of the debate.

### Naive Σ → 0 is ill-defined (`:1028`–`:1036`)

> "The deterministic limit requires nuance. Naively setting `Σ_i → 0` yields Dirac delta beliefs `q_i(k_i) → δ(k_i - μ_i)`, but the KL divergence between distinct Diracs is infinite:
>
> `D_KL(δ(k - μ_i) ‖ δ(k - μ_j)) = +∞ for μ_i ≠ μ_j.`
>
> This divergence reflects the absolute continuity requirement in the definition of KL: `D_KL(P ‖ Q)` is finite only when P is absolutely continuous with respect to Q (i.e., P(A) > 0 implies Q(A) > 0 for all measurable A). Distinct Dirac measures have disjoint support and therefore violate this condition.
>
> We may remedy this by taking a joint scaling limit where the belief variance σ² (discussed below) remains finite but is absorbed into learned parameters."

Manuscript explicitly acknowledges the literal `Σ → 0` limit is mathematically ill-defined and proposes a reparameterization instead.

### Isotropic-covariance assumption (`:1038`)

> "Next, we assume isotropic covariances `Σ_i = Σ_j = σ² I` with `σ² > 0`."

The reduction proceeds with `σ² > 0` finite — not with `Σ = 0`.

### Mahalanobis identity verification (`:1133`)

> "Equation full_kl_general is the unique inhomogeneous form taken by the gauge-covariant Gaussian KL when the covariance field is unconstrained and the frames are general GL(d_k) elements; it has been verified symbolically against the direct Gaussian KL to machine precision."

### Joint absorption of σ⁻² and Ω⁻ᵀ (`:1252`)

> "Next, rather than completely taking σ → 0 we recognize that σ⁻² and Ω⁻ᵀ always appear together in the combination σ⁻²Ω⁻ᵀ. The learned matrices `W_Q, W_K` can then be considered to parametrize this combined quantity directly, rendering σ an implicit (finite) scale factor absorbed into the learned weights. **Therefore, the full limit need not be taken.**"

Manuscript's own statement that the limit is not taken.

### Rescaled KL stays finite (`:1112`)

> "The rescaled KL divergence `D̃_KL(q_i ‖ Ω_{ij}q_j) := σ² · D_KL(q_i ‖ Ω_{ij}q_j) → ½‖Ω_{ij}^{-1}μ_i - μ_j‖² as σ → 0` remains finite and equals half the squared distance between means as measured in the key's reference frame."

The rescaled (σ²-multiplied) KL has a finite limit; the unrescaled KL diverges. The downstream use is of the rescaled form via the joint absorption.

### Section §5.7 summary (`:1958`)

> "Under the successive limits: (i) isotropic beliefs with σ⁻² absorbed into learned projections, (ii) constant gauge (Ω_{ij} = Ω for all pairs), and (iii) learned gauge via projections (W_Q W_K^T = σ⁻²Ω^{-T} ∈ GL(d_k))..."

Limit (i) is named as "σ⁻² absorbed into learned projections," not as "σ → 0." The summary is consistent with the line-1252 reparameterization.

## Canon excerpts — `external_canon_math.md` §1

### KL between Gaussians

> "For `q = N(μ_q, Σ_q)`, `p = N(μ_p, Σ_p)`, both K-dimensional:
>
> `KL(q ‖ p) = ½ [tr(Σ_p^{-1} Σ_q) + (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q) - K + log(|Σ_p|/|Σ_q|)]`
>
> ... For diagonal Σ (write σ² for diagonal entries): ..."

### KL properties

> "- KL ≥ 0, with equality iff q = p a.e.
> - **Undefined when q(x) > 0 and p(x) = 0** (or returns +∞)."

The absolute-continuity caveat (Dirac vs Dirac case) is canonical [KullbackLeibler1951, AmariNagaoka2000 Ch. 2].

## Canon excerpt — limit vs reparameterization

A *limit* in the analytic sense is `lim_{ε → 0} f(ε)`. A *reparameterization* is a substitution `θ_new = g(θ_old)` that re-expresses the same object in different variables. The two are distinct:

- The reparameterization `(σ, Ω) → (M)` where `M := σ⁻²Ω⁻ᵀ` is bijective for finite `σ > 0`. Under this reparameterization, the relevant quantity in the KL (the Mahalanobis-quadratic-times-scalar) is exact for any finite `σ`.
- The limit `σ → 0` of the original KL is `+∞` (ill-defined).
- The limit `σ → 0` of the *rescaled* KL `σ² · D_KL` is `½‖Ω^{-1}μ_i - μ_j‖²` (well-defined).

The manuscript's operation is the reparameterization, not the limit. The "Scaled Limit" naming at line 1024 is therefore a misnomer if read strictly, but is consistent with the *rescaled-KL limit* described at line 1112.

## What this evidence does NOT settle

1. **Naming.** "Deterministic Beliefs via Scaled Limit" (the paragraph header at line 1024) and the language "the full limit need not be taken" (line 1252) sit in tension. Does the section title misrepresent the operation? Blue may argue the title refers to the *rescaled-KL* limit (which is well-defined) and that "deterministic beliefs" is a heuristic for "beliefs whose variance is implicit in the learned weight scale." Red may argue the title implies an analytic σ → 0 limit that the manuscript itself admits is ill-defined.

2. **Does "deterministic beliefs" mean what the section title says?** A reader encountering "deterministic beliefs via scaled limit" without reading line 1252 will likely infer that beliefs are *literally* deterministic in the reduction (Dirac at μ). The manuscript instead has beliefs `q_i = N(μ_i, σ²I)` with σ finite throughout, and the W matrices' implicit scale absorbs σ. This is a semantic mismatch between the section header and the operation.

3. **Does the reparameterization preserve information?** The map `(σ, Ω) → σ⁻²Ω⁻ᵀ` is many-to-one in the forward direction (multiple `(σ, Ω)` pairs produce the same combined matrix M). Going "back" from M to (σ, Ω) is therefore not unique. This is fine for the reduction direction (gauge → transformer is well-defined) but means a standard transformer's `W_Q W_K^T` does not uniquely determine the gauge-theoretic `(σ, Ω)`. The first debate's verdict flagged this as a separate concern.

4. **Downstream consequences.** Sub-claim B has limited downstream relevance: whether the operation is called a "limit" or a "reparameterization" does not change the resulting `softmax(QK^T/√d_k)V` formula. The dispute is purely about epistemic framing. The judge should rule on whether the manuscript's naming is precise enough to be "mathematically valid" in the first debate's headline sense.
