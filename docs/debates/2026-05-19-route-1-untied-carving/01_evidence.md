# Evidence Pack — route-1-untied-carving

Neutral fact pack. Both teams work from this file.

## Manuscript references — `Attention/GL(K)_attention.tex` §5.2.1 (line 1122–1161)

### Section opening (`:1125`)

> "The trivial-frame reduction below is one realisation of standard attention. A second, complementary reduction retains `Ω_{ij} = U_i U_j^{-1}` non-trivially throughout and exhibits standard attention with untied query-key projections directly from the gauge-covariant KL, without inserting a separate bilinear M by hand."

### Exact KL decomposition (`:1128–1136`)

> "Under arbitrary SPD covariances `Σ_i` (independent of the frames `U_i`), the transported Gaussian KL admits the exact decomposition
>
> `D_KL(q_i ‖ Ω_{ij#} q_j) = ½[log det Σ_j - log det Σ_i + 2 log|det U_i| - 2 log|det U_j| - d_k + tr(H_j P_i) + x_i^T H_j x_i - 2 x_i^T k_j + r_j]`  (Eq. eq:full_kl_general)
>
> with the common-frame quantities
> `x_i = U_i^{-1} μ_i, P_i = U_i^{-1} Σ_i U_i^{-T}, H_j = U_j^T Σ_j^{-1} U_j, k_j = U_j^T Σ_j^{-1} μ_j, r_j = μ_j^T Σ_j^{-1} μ_j.`
>
> Equation `eq:full_kl_general` is the unique inhomogeneous form taken by the gauge-covariant Gaussian KL when the covariance field is unconstrained and the frames are general GL(d_k) elements; it has been verified symbolically against the direct Gaussian KL to machine precision."

### Cross-term carving (`:1138–1148`)

> "The cross term `-2 x_i^T k_j` is the only coupling between query and key indices through the belief means. It carves cleanly as a dot product between per-token vectors,
>
> `x_i^T k_j = (U_i^{-1} μ_i)^T (U_j^T Σ_j^{-1} μ_j) = Q_i^T K_j`  (Eq. eq:untied_cross_term)
>
> where the gauge-derived query and key projections are
>
> `Q_i = U_i^{-1} μ_i, K_j = U_j^T Σ_j^{-1} μ_j.`  (Eq. eq:gauge_qk, BOXED)
>
> For the same token, the two projections satisfy `W_Q^(i) = U_i^{-1}` and `W_K^(i) = U_i^T Σ_i^{-1}`, with the inverse-transpose tying broken by the precision factor `Σ_i^{-1}`. The construction is therefore genuinely untied: W_Q and W_K are different functions of the per-token belief, not the same projection up to symmetry."

### Implicit pair-dependent bilinear (`:1150–1155`)

> "The implicit pair-dependent bilinear is
> `M_{ij} := Ω_{ij}^{-T} Σ_j^{-1} = U_i^{-T} U_j^T Σ_j^{-1}`  (Eq. eq:gauge_bilinear)
>
> As `(U_i, U_j)` ranges over `GL(d_k)²` and `Σ_j` over `SPD(d_k)`, polar decomposition shows that `M_{ij}` ranges over all of `GL(d_k)`. The expressive power of the gauge-derived bilinear is therefore identical to that of an independently learned `W_Q W_K^T ∈ GL(d_k)` in standard attention. Generically `M_{ij}` is non-symmetric, recovering the asymmetry of `W_Q^T W_K` that the symmetric `C^{-1}` closure of the trivial-frame route does not."

### Reduction to standard form via closure (`:1157–1158`)

> "The remaining terms in Eq. `eq:full_kl_general` fall into two groups under the row-softmax over j. Terms depending on j alone, namely `r_j, log det Σ_j`, and `log|det U_j|`, contribute a **key-side prior bias** that is absorbed into `log π_{ij}`. Terms coupling i and j through the belief covariances rather than the means, namely `x_i^T H_j x_i` and `tr(H_j P_i)`, are gauge-theoretic uncertainty corrections beyond vanilla attention; they vanish under the closure `Σ_j = U_j C U_j^T` for shared SPD C, which collapses `H_j` to the constant `C^{-1}`. Under this closure the carving in Eq. `eq:gauge_qk` becomes the symmetric form `Q_i = C^{-1/2} U_i^{-1} μ_i, K_j = C^{-1/2} U_j^{-1} μ_j`, and standard attention with a single shared bilinear `C^{-1}` is recovered."

### Identification with rotary positional structure (`:1160`)

> "The natural identification of the per-token frame `U_i` with a real transformer architecture is the per-position rotational frame of rotary positional embeddings, in which `U_i ∈ O(d_k)` is a block-diagonal rotation depending on token position. In that case `U_i^{-1} = U_i^T` and the closure `Σ_j = U_j C U_j^T` holds automatically for any C commuting with the rotation block structure. The carving in Eq. `eq:gauge_qk` reduces to `Q_i = U_i^T μ_i` and `K_j = U_j^T μ_j`, which is the rotation-modulated query-key projection used in rotary attention, with `μ_i` playing the role of the position-independent content vector. The trivial-frame specialisation `U_i = U` for all i removes the per-token positional variation; in that limit `Ω_{ij} = I` and the analysis of Section dot_product_derivation below applies, with the learned bilinear M replacing the structural role played by per-token frame variation."

## Direct algebraic facts

### What is `r_j` under the closure `Σ_j = U_j C U_j^T`?

`r_j = μ_j^T Σ_j^{-1} μ_j = μ_j^T (U_j C U_j^T)^{-1} μ_j = μ_j^T U_j^{-T} C^{-1} U_j^{-1} μ_j = (U_j^{-1} μ_j)^T C^{-1} (U_j^{-1} μ_j)`.

Defining the symmetric-carving key `K_j^{sym} = C^{-1/2} U_j^{-1} μ_j` (the line-1158 form):

`r_j = (U_j^{-1} μ_j)^T C^{-1} (U_j^{-1} μ_j) = (C^{1/2} K_j^{sym})^T C^{-1} (C^{1/2} K_j^{sym}) = (K_j^{sym})^T K_j^{sym} = ‖K_j^{sym}‖²`.

**Under the closure, `r_j` IS the squared key norm.** This is the same content-dependent quantity that drove the first debate's RED_WINS verdict at Route 2.

### What is `log det Σ_j` under the closure?

`log det Σ_j = log det(U_j C U_j^T) = log|det U_j|² + log det C = 2 log|det U_j| + log det C`.

So `log det Σ_j - 2 log|det U_j| = log det C` — a constant in j.

The remaining j-only term after this cancellation is `r_j = ‖K_j^{sym}‖²` (plus `log det C`, a constant).

### What "absorption into log π_{ij}" means

The standard softmax shift-invariance: `softmax_j(s_{ij} - c_j) = (exp(-c_j) · softmax_j(s_{ij}))_{normalized}`. If we set `log π_{ij} = c_j`, then the softmax becomes `π_j · exp(-s_{ij}) / Σ_k π_k · exp(-s_{ik})`. With `c_j = r_j / (2τ) + log det C / (2τ) + log det Σ_i / (2τ) + ...`, the j-only bias is absorbed into a non-uniform prior `π_{ij} = π_j' = exp(c_j)`.

This is mathematically valid. The question is what kind of "prior" this is:
- Constant `c_j` → uniform `π_j` → standard Vaswani §3.2.1.
- Position-dependent `c_j` (only depends on `j`'s position, not content) → ALiBi / causal mask / relative-position bias (Vaswani §3.2.1 + standard extensions).
- Content-dependent `c_j` (depends on `μ_j` through `r_j = ‖K_j^{sym}‖²`) → NOT a fixed prior; it's a content-dependent bias on the logits.

Standard Vaswani §3.2.1 uses uniform prior. The absorption of `r_j` into `log π_{ij}` produces a content-dependent additive bias that is NOT a fixed prior.

## Canon excerpts

### `external_canon_transformers.md` §1 — Standard form

> "`Attention(Q, K, V) = softmax(Q K^T / √d_k) V` ... Q, K from input X via learned `W_Q, W_K ∈ ℝ^{d_model × d_k}`."

Standard form: scaled dot-product `Q^T K` logits, divided by `√d_k`, fed through softmax. **No additive bias on the logits.** Mask (`-∞` for masked positions) is an inequality-constraint extension, not a constituent of the base form.

### `external_canon_transformers.md` §6 — Causal masking

> "Standard causal mask: set logits to `-∞` for positions `j > i` before softmax. ... The user's framework interprets this as a 'non-uniform attention prior `π_j`': `π_j = 0` for `j > i` is equivalent to the `-∞` logit shift."

Causal masking, ALiBi, and relative-position biases are *extensions* of the Vaswani §3.2.1 base form, implemented via additive logit biases. They are not constituents of the base form.

### Vaswani 2017 §3.2.1 — primary source

The standard scaled dot-product attention rule:
```
Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```
No additive bias. Multi-head extensions (§3.2.2), masking (§3.2.3), and positional encodings (§3.5) are described as separate constructs on top of the base form.

The framework's claim that Route 1 "recovers Vaswani §3.2.1" requires the post-reduction form to match `softmax(Q^T K / √d_k) V` with no additional bias. If `log π_{ij}` is non-uniform (e.g., contains `r_j = ‖K_j‖²`), the recovered form is `softmax((Q^T K - r_j/2 + ...) / √d_k) V` — a different attention rule.

### Equivalence relation to first debate's verdict (Route 2)

The first debate (`docs/debates/2026-05-19-reduction-to-standard-transformer/04_verdict.md`) returned RED_WINS on the headline that Route 2 reduces *exactly* to Vaswani §3.2.1. The decisive evidence was the manuscript's own "approximate" language at line 1198 for the `‖μ_j‖²` key-norm cancellation, with the LN escape failing under standard `[BaKirosHinton2016]` LN with per-feature affine γ, β.

Under the closure `Σ_j = U_j C U_j^T`, Route 1's `r_j` equals `‖K_j^{sym}‖²` — the same key-norm quantity Route 2 needed to cancel. Route 1 handles it by absorption into `log π_{ij}` rather than by approximate cancellation. The relocation does not eliminate the quantity; it relabels its handling.

## What this evidence does NOT settle

1. **Is "absorption into `log π_{ij}`" with content-dependent `c_j(μ_j)` legitimately within Vaswani §3.2.1?** Blue may argue:
   - The standard softmax shift-invariance makes any j-only bias absorbable; the resulting attention rule is structurally identical to softmax-based attention; the "prior" framing captures all such biases including content-dependent ones.
   - Standard transformers in practice use prior-extension mechanisms (causal masking, ALiBi, T5 relative bias); content-dependent prior is just a generalization of these.

   Red may argue:
   - Vaswani §3.2.1 explicitly uses uniform prior; any non-uniform prior is an extension *on top of* §3.2.1, not §3.2.1 itself.
   - Content-dependent additive bias `r_j(μ_j)/τ` is a non-standard attention rule not present in Vaswani's construction; calling it "Vaswani §3.2.1 with absorbed prior" misrepresents the recovery.

2. **Does the closure `Σ_j = U_j C U_j^T` hold automatically for standard transformers?** Blue cites the rotary-frames identification at line 1160: under RoPE, `U_i ∈ O(d_k)` is orthogonal so `U_i^{-1} = U_i^T`, and the closure holds automatically for any C commuting with the rotation block structure. But standard transformers without RoPE (e.g., the original Vaswani 2017 with sinusoidal positional encoding) do not have per-token `U_i` at all; the closure is vacuously trivial or undefined depending on how `U_i = I` is interpreted.

3. **Is the rotary identification at line 1160 a recovery of *RoPE* (a specific transformer variant) or of *Vaswani §3.2.1* (the base form)?** The two are different. RoPE was introduced by Su et al. 2024 as an extension of standard transformer attention. Route 1's recovery to "rotation-modulated query-key projection" is recovery of RoPE-style attention, not Vaswani §3.2.1.

4. **Trivial-frame specialization.** Manuscript at line 1161: "The trivial-frame specialisation `U_i = U` for all i removes the per-token positional variation; in that limit `Ω_{ij} = I` and the analysis of Section dot_product_derivation below applies." Under `U_i = U`, the symmetric-carving keys `K_j^{sym} = C^{-1/2} U^{-1} μ_j` are all in the same frame, and `r_j = ‖K_j^{sym}‖²` still equals the key-norm squared. The trivial-frame specialization does NOT escape the key-norm bias; it just makes it explicit. This is the same content as Route 2.

5. **Untied vs symmetric carving.** The non-symmetric form at line 1145 (`W_Q^(i) = U_i^{-1}, W_K^(i) = U_i^T Σ_i^{-1}`) is "genuinely untied" — a structural feature the framework claims standard attention has. Under the closure, the symmetric form `Q_i = C^{-1/2} U_i^{-1} μ_i, K_j = C^{-1/2} U_j^{-1} μ_j` is recovered, which is *symmetric* in `i ↔ j` (same projection up to the C^{-1/2} prefix). The "untied" feature of the gauge-derived projections is therefore *broken* by the closure that recovers standard attention. Whether this is a problem for the recovery direction is interpretive.

6. **What "exactly" means in the headline.** Same headline-level ambiguity as the first debate: "exactly recovers Vaswani §3.2.1" can mean (i) exactly equals the boxed `softmax(QK^T/√d_k) V` formula with uniform prior, or (ii) exactly equals the boxed formula modulo additive prior bias absorption. Reading (i) is the strict reading; reading (ii) is the broad reading that allows extensions. The headline claim says "Vaswani §3.2.1" — primary-source reading is the strict one.
