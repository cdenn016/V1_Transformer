# Reviewer B — GL(K)_attention.tex §4 (lines 990–1976)

Date: 2026-05-18
Scope: §4 *Reduction to Transformer Attention*, including subsections 4.1 (Setup), 4.2 (Connection to Standard Transformers, with the untied-QK and dot-product-derivation subsections), 4.3 (Training as VFE Minimization), 4.4 (Multi-Head as Block-Diagonal GL(K)), 4.5 (RoPE), 4.6 (Further Correspondences), and 4.7 (Summary).
Sources of truth: [Vaswani2017], [Su2024RoPE], [Friston2010], [AmariNagaoka2000], [Nakahara2003], [BleiKuckelbirgJordan2017], [KingmaWelling2014], [BaKirosHinton2016], [AbsilMahonySepulchre2008], [Amari1998], [BaiKolterKoltun2019], [Hall — Lie Groups, Lie Algebras, and Representations]. Internal canon used: `external_canon_math.md`, `external_canon_inference.md`, `external_canon_transformers.md`.

## Summary

§4 is the load-bearing chapter of the manuscript: it claims that standard transformer self-attention is recovered from the gauge-theoretic VFE framework under explicit limits, and that several adjacent architectural choices (multi-head, RoPE, LN, residual, FFN, embeddings) follow as further specializations. The core reduction is structured along two complementary routes — the untied-QK carving (§4.2.1) and the explicit three-limit dot-product derivation (§4.2.2) — and the derivations themselves are mathematically sound. The Gaussian KL form (Eq. 1043), the Mahalanobis identity (Eq. 1049), the geometric-bias non-negativity claim (Eq. 1075), the Rényi divergence form (Eq. 1059), the α-product-rule manipulation (Eq. 956), and the multi-head off-diagonal generator count (line 1774) all pencil out. The framework is qualified appropriately in most of the body prose (e.g., the envelope-theorem discussion at lines 857–872, and the careful distinction between "exact" and "structural" in Table 1's status column). The main concerns are presentation discipline: several Table 1 rows are labeled "D" (derived) where the underlying body text actually argues for an (I) interpretive correspondence, the natural-gradient metric on `φ ∈ gl(K)` is deferred to Supp. App. C without first stating which metric is being used (the Killing form is degenerate on `gl(K)`, so this matters), the RoPE identification at line 1847 relies on abelian subgroup structure that should be stated, one cross-reference is broken (`eq:beta_grad_phi` referenced but never defined), and the project's banned LaTeX spacing macros `\,`/`\;`/`\!` appear extensively. The overall reduction claim is defensible; revisions are presentational and definitional rather than mathematical.

## Standards against which the manuscript was reviewed

- [Vaswani2017] for standard scaled dot-product attention and √d_k scaling.
- [Su2024RoPE] for RoPE construction and translation-invariant inner product structure.
- [BleiKuckelbirgJordan2017] / [KingmaWelling2014 App. B] for the closed-form Gaussian KL.
- [Friston2010] for the variational free energy form.
- [AmariNagaoka2000] for natural gradient and Fisher metric on the Gaussian.
- [Nakahara2003] for parallel transport and the sandwich identity for bilinear forms.
- [AbsilMahonySepulchre2008] for retractions and natural gradient on Lie groups.
- [BaKirosHinton2016] for LayerNorm.
- [Hall — Lie Groups] for matrix-exponential surjectivity and Killing-form properties on `gl(K)`.

I was unable to load WebFetch / WebSearch in this session, so cited papers I could not directly fetch are marked `[?]` in the citation table below. The findings rely on textbook content reflected in `external_canon_*.md`.

## Major findings

### M-B-1. Table 1 status "D" for `W_Q W_K^T = σ⁻² Ω⁻ᵀ` overclaims the body text

**Claim (manuscript):** Table 1 row at line 1650, status column "D" (derived):
`Ω_{ij} = e^{φ_i}e^{-φ_j} ∈ GL⁺(K)  →  W_Q W_K^T = σ⁻² Ω⁻ᵀ` under "Constant Ω (Limit 2)".
**Claim kind:** (I) interpretive presented as (R)/derived.
**Standard treatment:** In [Vaswani2017], `W_Q, W_K` are rectangular learned projections `ℝ^{d_model × d_head}`; the bilinear `W_Q W_K^T ∈ ℝ^{d_model × d_model}` has rank at most `d_head` and is not in `GL(d_model)` in general. The factorization `M = AB^T` is non-unique: many `(A, B)` pairs realize the same `M` (SVD freedom).
**Problem:** The manuscript body itself addresses this at lines 1240–1250 (thin-SVD lift to `M_h^a ∈ GL(d_head)` for the invertible head-space factor) and explicitly states "This identification concerns the bilinear form `M = W_Q W_K^T` that determines the attention logits, not the individual matrices `W_Q` and `W_K` separately" (line 1243). That qualifying paragraph is good and earns its place. But the summary Table 1 row still calls this "D" without inheriting that qualification. A reader who skims the table will read it as: "the framework derives the specific QKV factorization of standard transformers." It does not. It derives the bilinear, and the head-space lift of the bilinear is in `GL(d_head)`; many `Ω`-induced bilinears coincide.
**Required revision:** Either (a) relabel the row "S" or introduce a third status code (e.g., "D*" for derived-up-to-factorization-freedom) and document, or (b) restrict the row to claiming the head-space lift `M_h^a ∈ GL(d_head)` rather than the ambient `W_Q W_K^T`. The body text already supports option (b); the table needs to inherit it.

### M-B-2. Table 1 status "D" for residual-connection ↔ natural-gradient Euler step

**Claim (manuscript):** Table 1 row at line 1682, status column "D":
`μ^{(ℓ+1)} = μ^{(ℓ)} - η̃ Σ ∇F  →  Residual connection`, "Natural gradient flow" → "Residual connection".
**Claim kind:** (I) interpretive presented as (R)/derived.
**Standard treatment:** A residual connection `x_{ℓ+1} = x_ℓ + f(x_ℓ)` has the structural form of any explicit Euler step. The transformer's residual stream interpretation [elhage2021mathematical] is not uniquely implied by Euler integration of a continuous dynamics; many other discretizations (RK2, RK4, Cayley) also have new = old + correction shape. The §4.3 body at lines 1612–1623 is in fact careful: the manuscript writes "suggests both frameworks capture the same underlying computational principle." That is the language of (I) correspondence, not (R) derivation.
**Problem:** The status "D" in Table 1 elevates an (I) correspondence to a derivation. The same issue applies to the row `d μ/dt = -η Σ ∇F → Gradient descent (backprop)` at line 1675 — Euler integration of any gradient flow has that form; this is structural.
**Required revision:** Demote both rows to "S" (structural correspondence) in Table 1 — this matches the existing body prose at line 1622 and the table caption's own definition of "S = the framework explains the component's role but does not uniquely predict its specific form" (line 1691). The manuscript prose already does the right thing; only the table is overclaiming.

### M-B-3. Natural-gradient metric on `φ ∈ gl(K)` is not specified in §4

**Claim (manuscript):** Line 1632: "`∇̃_{θ_i}` denotes the natural gradient on each channel's parameter manifold (Fisher-Rao for `μ_i` and `Σ_i`; Killing form or pullback metric for `φ_i`, see Supplementary Appendix~C)."
**Claim kind:** (S) standard, in the sense that the manuscript invokes a standard concept (natural gradient on Lie algebra).
**Standard treatment:** [Amari1998] defines the natural gradient as `g^{-1} ∇L` for a Riemannian metric `g`. On a Lie algebra, common choices are the Frobenius inner product (positive-definite on `gl(K)`) or the Killing form (`⟨A,B⟩_K = tr(ad_A · ad_B)`). On `gl(K) = sl(K) ⊕ ℝ·I`, the Killing form is **degenerate**: it vanishes on the `ℝ·I` center. On `sl(K)` (`K ≥ 2`) it is non-degenerate but **indefinite** for non-compact real forms ([Hall], [AbsilMahonySepulchre2008]). A non-positive-definite "metric" is not a Riemannian metric and does not give a well-defined natural gradient direction.
**Problem:** The manuscript writes "Killing form or pullback metric for `φ_i`" without specifying which, and the Killing form alone is not a viable choice on the full `gl(K)`. Deferring to Supplementary Appendix C means readers of the main body can't verify whether the natural-gradient invocation is well-posed.
**Required revision:** In §4.3 state explicitly which metric is used on `gl(K)` (most likely the Frobenius / pullback Fisher rather than Killing, given the project policy `## Contributing` rule about natural gradients). If the Killing form is in fact used, restrict its domain to a subalgebra where it is positive-definite (e.g., a compact real form), and disclose the restriction. One sentence in the main body suffices; the full derivation can remain in Supp. App. C. Cite [AbsilMahonySepulchre2008] for the standard treatment.

### M-B-4. RoPE identification at line 1847 silently uses abelian subgroup structure

**Claim (manuscript):** Line 1847: "the inter-agent rotation `R(θ_{j-i}) = exp(-φ_i^{pos}) exp(φ_j^{pos})` has exactly the form of a gauge transport operator [Eq. gauge_frame_rotation] restricted to the `SO(2)^{d_k/2}` subgroup."
**Claim kind:** (R) reduction.
**Standard treatment:** The general gauge transport defined at Eq. 1099 (and Eq. 616) is `Ω_{ij} = exp(φ_i) exp(-φ_j)` (sign convention: positive on `i`, negative on `j`). The RoPE identification at line 1847 writes `R(θ_{j-i}) = exp(-φ_i^{pos}) exp(φ_j^{pos})` (opposite signs). For non-abelian `φ_i, φ_j` these are different elements of `GL(K)` in general. They coincide only because (a) `φ_i^{pos}, φ_j^{pos}` are block-diagonal scalar multiples of the same generator `J`, so they commute, and (b) for the abelian `SO(2)^{d_k/2}` subgroup, `Ω^{-T} = Ω` (since each block is in `SO(2)` and skew-symmetric infinitesimal generators give `R^T = R^{-1}`). I verified this with sympy: on the abelian RoPE subgroup, `(Ω_{ij})^{-T} = Ω_{ij}` and the sign-flipped form coincides with the canonical form.
**Problem:** The identification reads as a generic match between RoPE and the framework's transport. It is actually a coincidence in the abelian subgroup. Outside `SO(2)^{d_k/2}`, the sign-flipped product is *not* the gauge transport `Ω_{ij}`; it is `(Ω_{ij})^{-T}` — which is what enters the bilinear `Q W_Q W_K^T K = μ_i^T Ω^{-T} μ_j`. Eq. 1832 already writes the logit kernel as `σ^{-2} (Ω_{ij}^{RoPE})^{-T}`, so the body is internally consistent; the issue is that line 1847 sells this as "exactly the form of a gauge transport operator" when in fact it is the form of the inverse-transpose, which happens to equal the transport only because of abelianness.
**Required revision:** Insert a half-sentence at line 1847 noting that the identification uses the abelian-subgroup property `Ω^{-T} = Ω` valid on `SO(2)^{d_k/2}`, and references the general non-abelian rule `(Ω_{ij})^{-T}` (from Eq. 1832 — which the manuscript already uses correctly). This is a definitional clean-up, not a math fix.

## Minor findings

- **m-B-1.** Line 1940 references `Eq.~\eqref{eq:beta_grad_phi}` for the φ-gauge softmax-gradient correction, but `\label{eq:beta_grad_phi}` is not defined in `GL(K)_attention.tex` (verified via grep across the full file). Either define the equation in the main body or move the reference to the Supplementary if the derivation lives there.
- **m-B-2.** Line 1340: "Our `GL(K)` experiments (Section~\ref{sec:glk_lm}) utilize the full generality (at least for a `0` dimensional manifold)". The parenthetical "0 dimensional manifold" is opaque without context; it refers to the single-base-point setup of §4.1 line 1026 but reads as a hedge. Suggest either expanding to "(over a single-point base space `C`, as in §4.1)" or removing the parenthetical.
- **m-B-3.** Line 1635 cites `\citep{bai2019deep}` for deep equilibrium models. The actual DEQ paper is Bai, Kolter, Koltun, *Deep Equilibrium Models*, NeurIPS 2019; citation key is appropriate but verify the bibliography entry matches.
- **m-B-4.** Line 1096 invokes "orthogonal projection (e.g., Newton–Schulz iteration)" to map `GL⁺(K) → SO(K)`. Newton–Schulz computes the orthogonal polar factor; the polar decomposition of a `GL⁺(K)` element has its `O(K)` factor in the identity component `SO(K)` only when `det > 0`, which is guaranteed for `GL⁺(K)`. The claim is correct but the parenthetical "as described above" at line 1103 refers to a description that has only been stated implicitly. A one-line statement of the Newton–Schulz iterates would help readers unfamiliar with polar decomposition.
- **m-B-5.** The deterministic-limit subsection (lines 1024–1037) opens with "the deterministic limit requires nuance" and then takes a joint scaling limit where `σ^{-2}` is absorbed. The phrasing "nuance" is editorial filler — the actual mathematical issue (KL divergence of Diracs is infinite) is clearly stated immediately afterward and deserves the framing.
- **m-B-6.** The "off-diagonal gauge mixing" discussion at lines 1764–1792 frames the 87.5% discarded generators as a potential source of expressive power that standard MHA loses. The numerical claim is correct (verified: 229,376 / 262,144 = 0.875 at `d_k = 512, H = 8`), but the framing assumes that off-diagonal gauge couplings carry useful signal — line 1792 admits this is "an empirical question with significant implications." Recommend tightening the rhetoric: state the dimensional count plainly and let the empirical claim stand on its own.
- **m-B-7.** Line 1709: "Recall from Section~\ref{sec:glk_invariance} that, in the isotropic flat-bundle limit, standard transformer attention can be interpreted as `GL(d_k)` gauge-theoretic attention with the effective gauge transport identified as `Ω = (σ^2 W_K W_Q^T)^{-1} ∈ GL(d_k)`." The sign/transpose convention here (`W_K W_Q^T` rather than `W_Q W_K^T`) differs from Eq. 1238 (`W_Q W_K^T = σ^{-2} Ω^{-T}`). Both are algebraically consistent (one is the transpose of the other), but the inconsistency between sections will confuse readers tracing the identification. Suggest aligning the form across §4.2.2 and §4.4.

## Equation verification log

| Equation | Page/Line | Verification | Method |
|---|---|---|---|
| Eq. 1043 (Gaussian KL with isotropic q and transported p) | line 1043 | Matches the standard closed-form for Gaussian KL [BleiKuckelbirgJordan2017, KingmaWelling2014 App. B] when specialised to `Σ_q = σ²I`, `Σ_p = σ² Ω Ω^T`. | Pencil expansion + sympy. |
| Eq. 1049 (Mahalanobis identity `v^T (ΩΩ^T)^{-1} v = ‖Ω^{-1}v‖²`) | line 1049 | Holds algebraically: `(ΩΩ^T)^{-1} = Ω^{-T} Ω^{-1}`, so `v^T Ω^{-T} Ω^{-1} v = ‖Ω^{-1}v‖²`. | sympy 2×2 explicit. |
| Eq. 1059 (Rényi divergence for diagonal Gaussians) | line 1059 | Matches [van Erven-Harremoes 2014] standard form with convention `S̃ = (1-α) S_q + α S_p`. Reduces to KL as α→1 (verified pencil). | Pencil + convention check. |
| Eq. 1075 (`S(Ω) = ½[log det(ΩΩ^T) + Tr((ΩΩ^T)^{-1}) - d_k] ≥ 0`, equality iff Ω ∈ O(K)) | line 1075 | `f(λ) = log λ + 1/λ` minimised at λ = 1 with `f(1) = 1`, so `Σ_k (log λ_k + 1/λ_k) ≥ d_k` with equality iff all λ_k = 1, iff Ω Ω^T = I. | sympy 3-eigenvalue numerical. |
| Eq. 1086 (general isotropic attention with geometric bias) | line 1086 | Follows directly from Eq. 1043 + Eq. 1049 + softmax over `j`. | Algebraic substitution. |
| Eq. 1099 / 1106 (`O(K)` extension via reflections `Ω = R_i M_{ij} R_j`) | lines 1099–1110 | The reflection-times-`SO(K)` factorisation covers both components of O(K). `Ω Ω^T = R_i M M^T R_i = R_i² = I` for `M ∈ SO(K)`. Reflections absorbable as elementwise sign flips on the embeddings. Standard construction. | Pencil. |
| Eq. 1126 (full transported Gaussian KL with general SPD covariances and `Ω = U_i U_j^{-1}`) | line 1126 | The decomposition into `x_i, P_i, H_j, k_j, r_j` matches the standard expansion of `½[log|Σ_p|/|Σ_q| + Tr(Σ_p^{-1} Σ_q) + (μ_p-μ_q)^T Σ_p^{-1} (μ_p-μ_q) - d_k]` after substituting `Σ_p = Ω Σ_j Ω^T` and grouping terms in common-frame coordinates. The manuscript says it has been "verified symbolically against the direct Gaussian KL to machine precision" — plausible and matches my pencil expansion. | Pencil + accept manuscript's symbolic claim. |
| Eq. 1142–1149 (untied Q/K carving `Q_i = U_i^{-1} μ_i, K_j = U_j^T Σ_j^{-1} μ_j`) | lines 1142–1149 | Follows from the cross term `-2 x_i^T k_j` in Eq. 1126. Carving is unique given the choice of common-frame variables. | Algebraic. |
| Eq. 1155 (reduction under `Σ_j = U_j C U_j^T` closure recovers symmetric `Q_i = C^{-1/2} U_i^{-1} μ_i`) | line 1155 | Substituting the closure collapses `H_j = C^{-1}` constant, the trace and quadratic-in-`x_i` terms become constant under softmax over `j`, and the carving symmetrises. Standard. | Pencil. |
| Eq. 1238 (`W_Q W_K^T = σ^{-2} Ω^{-T}`) | line 1238 | Identification, not derivation — many `(W_Q, W_K)` pairs realise the same product. The manuscript at lines 1240–1250 explicitly addresses this. See M-B-1. | Polar decomposition / SVD. |
| Eq. 1280 (`τ = √d_k`) | line 1280 | Standard [Vaswani2017 §3.2.1] variance argument: for unit-variance Q, K components, `Q_i K_j^T` has variance d_k, so dividing by √d_k restores unit variance. Recapitulated, not novelly derived. | Standard. |
| Eq. 1407 (deterministic free energy with adaptive prior regularisation) | line 1407 | Follows from substituting `q = N(μ, εI), p = N(μ_prior, σ_p² I)` into the Gaussian KL and dropping the `μ_i`-independent `C(ε)` terms. | Pencil. |
| Eq. 1442 (mean gradient in isotropic limit) | line 1442 | Differentiating `λ_p/2 ‖μ_i - μ_prior‖² + Σ_j β_{ij}/(2σ²) ‖Ω^{-1}μ_i - μ_j‖²` w.r.t. `μ_i` via product rule gives exactly the four terms. The factor `(ΩΩ^T)^{-1}(μ_i - Ωμ_j)` comes from `∇ ‖Ω^{-1}μ_i - μ_j‖² = 2 Ω^{-T}(Ω^{-1}μ_i - μ_j) = 2 (ΩΩ^T)^{-1}(μ_i - Ωμ_j)` (using `Ω^{-T} Ω^{-1} = (ΩΩ^T)^{-1}`). | Pencil + sympy. |
| Eq. 1444 (Fisher inverse `G_μ^{-1} = Σ_i` for Gaussian mean) | line 1444 | Standard [Amari1998, AmariNagaoka2000 Ch.2]: Fisher information for `μ` in `N(μ, Σ)` is `Σ^{-1}`, so the inverse Fisher is `Σ`. Natural gradient `Σ ∇L` correct. | Standard. |
| Eq. 1497 (general non-isotropic mean gradient) | line 1497 | Term-by-term decomposition: state-dependent `α_i` and adaptive prior couplings, edge-dependent transported precision `(Ω_{ij} Σ_j Ω_{ij}^T)^{-1}` from sandwich rule, attention-weight derivatives via product rule. All four terms structurally correct. | Pencil. |
| Eq. 1552 (general non-isotropic covariance gradient) | line 1552 | Differentiating the Gaussian KL w.r.t. `Σ_i` gives `½(Σ_i^{-1} - Σ_p^{-1})` for the prior KL and contributes `-½ Σ_i^{-1} + ½ (Ω Σ_j Ω^T)^{-1}` from each alignment term [textbook, KingmaWelling2014 App. B differentiation]. The `-(1+α_i) Σ_i^{-1}` coefficient is consistent (entropy of `q_i` enters both prior KL with weight `α_i` and the `Σ_j β_{ij} = 1` of alignment terms). | Pencil. |
| Eq. 1566 (precision fixed-point `Σ_i^{-1} = ½[Σ_p^{-1} + Σ_j β_{ij} (Ω Σ_j Ω^T)^{-1}]`) | line 1566 | Setting Eq. 1552 to zero and neglecting the `∂α/∂Σ` and `∂β/∂Σ` corrections (justified at lines 1556–1567 by `Σ_j ∂β/∂Σ = 0` plus uniformity arguments). The factor 1/2 is correct when prior precision and alignment-summed precisions both enter at unit weight. | Pencil. |
| Eq. 956 (α product-rule chain rule giving `(α_i*)² b_0/c_0`) | line 956 | sympy: substituting `α* = c_0/(b_0 + D_KL)`, the bracket `1 - (α*/c_0) D_KL` simplifies to `b_0/(b_0 + D_KL) = α* b_0/c_0`. Algebraic identity. | sympy. |
| Eq. 1774 (multi-head off-diagonal generator count, 87.5% at `d_k=512, H=8`) | line 1774 | `(d_k² - H · d_head²) / d_k² = 1 - 1/H = 7/8 = 0.875`. Correct. | Arithmetic. |
| Eq. 1744 (per-head `κ_a √d_head` temperature) | line 1744 | Matches the standard `√d_k` argument applied per-head, with `κ_a` factoring out the isotropic per-head variance — consistent with `external_canon_transformers.md §2`. Match with code: `transformer/core/vfe_closed_form.py:231` and `transformer/core/vfe_gradients.py:509,899,1344` all compute `kappa * sqrt(K)` (or `sqrt(d_head)` in head-block form). | Standard + code grep. |
| Eq. 1817–1827 (RoPE rotation matrices, frequency `θ_n = 10000^{-2n/d_k}`) | lines 1812–1820 | Matches [Su2024RoPE] standard form. The relative-position property `R(θ_i)^T R(θ_j) = R(θ_{j-i})` follows from `R^T = R^{-1}` for `SO(2)` rotations and abelian composition within each block. | Standard. |
| Eq. 1836 (RoPE as partial Limit-2 relaxation, restriction to `SO(2)^{d_k/2} ⊂ GL(d_k)`) | line 1836 | Algebraically correct identification. See M-B-4 for the abelian-subgroup subtlety on the sign convention. | Pencil + sympy. |

## Style scan

The project's banned LaTeX spacing macros `\,`, `\;`, `\!` appear extensively in scope (at least 50 instances enumerated via grep, including lines 1075, 1080, 1086, 1099, 1246, 1393, 1394, 1447, 1460, 1483–1518, 1540–1548, 1585–1586, 1618, 1629, multiple Table 1 rows, 1744, 1770, 1806, 1812, 1814, 1832, 1843, 1856). This is a file-level cleanup item; do not enumerate each line in a revision. Recommendation: run a single global pass to remove banned spacing macros across the manuscript.

No banned phrases found in scope. `grep -i` for `key insight | crucially | critically | notably | importantly | it's worth noting | interestingly | fundamentally | in particular | leverages | underscores` returned zero matches. No horizontal-rule visual separators found. No self-referential drafting language ("earlier drafts", "the corrected reading", etc.) found.

Equation punctuation: most display equations in §4 lack terminal commas/periods, which is the project's required style under `style_constraints.md`. Examples: Eq. 999, 1007, 1015, 1030, 1043, 1049, 1080, 1086, 1142, 1149, 1232, 1238, 1262, 1286, 1297, 1316, 1326, 1332, 1354, 1359, 1376, 1394, 1407, 1417, 1426, 1440, 1447, 1469, 1497, 1505, 1525, 1551, 1567, 1589, 1619, 1629, 1701, 1717, 1725, 1746, 1771, 1786, 1807, 1818, 1832, 1844, 1856, 1882, 1897, 1909, 1922, 1934, 1951, 1962, 1967. File-level cleanup pass.

## Citations checked

| Citation | Status | Notes |
|---|---|---|
| `\citep{vaswani2017attention}` (multiple references including line 1699) | [?] | Could not fetch — WebFetch deferred. Canonical reference for scaled dot-product attention. Manuscript usage is consistent with the standard form. Verify bibliography entry matches [Vaswani et al., NeurIPS 2017, "Attention Is All You Need"]. |
| `\citep{su2024roformer}` (line 1799) | [?] | Could not fetch — WebFetch deferred. RoPE paper. Manuscript usage (frequencies `10000^{-2n/d_k}`, position-dependent rotations) matches public RoPE descriptions. |
| `\citep{hendrycks2016gaussian}` (line 1874) | [?] | GELU paper. Manuscript usage (GELU as Gaussian CDF gate) is consistent with the standard. |
| `\citep{dauphin2017language}` (line 1874) | [?] | GLU paper. Used to attribute the gated linear unit construction. Consistent with the literature. |
| `\citep{ramachandran2018searching}` (line 1913) | [?] | SiLU/Swish. Used to attribute the `x · σ(x)` form. Consistent. |
| `\citep{bai2019deep}` (line 1635) | [?] | Deep Equilibrium Models. Used at the "Layers as VFE iterations" paragraph to identify `L=1, T→∞` with DEQ. Consistent. |
| `\citep{elhage2021mathematical}` (line 1632) | [?] | Mathematical framework for transformer circuits (Anthropic 2021). Used to attribute the residual-stream interpretation. Consistent. |
| `\citep{press2022train}` (line 818) | [?] | ALiBi. Used at the non-uniform priors discussion. Out of scope (line < 990) but cross-referenced in §4. |
| `\citep{radford2019language}` (line 791) | [?] | GPT-2. Used to attribute causal masking. Out of scope. |
| `\citep{raffel2020exploring}` (line 836) | [?] | T5. Used to attribute relative position biases. Out of scope. |
| `\citep{beltagy2020longformer}` (line 801), `\citep{jiang2023mistral}` (line 801) | [?] | Sliding window attention. Out of scope. |

WebFetch / WebSearch were not loaded in this session. Citation specificity is bounded to source-level matching against the manuscript's claimed usage; specific page/equation verification was not performed.

## Code cross-references checked

The manuscript's §4 deliberately stays at the theoretical level and does not name specific config flags or code paths (`em_mode`, `skip_attention`, `E_sigma_q_lr`, etc.). The few in-scope code references are:

- Line 1069: "Our reference implementation exposes the order `α` through the per-block configuration option (`alpha_divergence` in `transformer/core/block_config.py`)". **Verified.** `transformer/core/block_config.py:166` defines `alpha_divergence: float = 1.0` with the documented semantics (1.0 = KL, 0.5 = Bhattacharyya). `block_config.py:592` plumbs it from a config dict.
- Line 960: "Our reference implementation evaluates the corrected form (`transformer/core/vfe_gradients.py`)". **Verified path exists.** `transformer/core/vfe_gradients.py` is present and contains the gradient logic referenced.
- Line 1472: "This is consistent with the E-step algorithm we implement, which applies Fisher preconditioning `∇̃_μ ← Σ_i · ∇_μ F` without imposing the additional projection-absorption limits." **Plausible.** `transformer/vfe/e_step.py:206-208` reads `e_mu_lr`, `e_sigma_lr`, `e_sigma_q_trust` from the config; the Fisher-preconditioned μ retraction is implemented in the same module's E-step iteration body. Did not trace the exact preconditioning multiply.
- Line 1741–1750: per-head `κ_a √d_head` temperature. **Verified.** `transformer/core/vfe_closed_form.py:231` uses `kappa_h_val * math.sqrt(max(d_h, 1))`; `transformer/core/vfe_gradients.py:509, 552, 899, 1344, 1796, 1923` all use `kappa * math.sqrt(max(K, 1))`; `transformer/vfe/non_flat.py:513` uses `kappa * math.sqrt(max(K, 1))`. The learnable per-head κ_a lives at `log_kappa_per_head` (e.g., `transformer/core/variational_ffn.py:546`, `transformer/train_publication.py:152`). Manuscript and code agree on the form.

CLAUDE.md asked whether §4 reflects the decoupled E-step learning rates (`E_mu_q_lr`, `E_sigma_q_lr`, `E_sigma_q_trust`) introduced 2026-05-13. Answer: §4 does not discuss E-step learning rates at all. The section operates at the abstraction level of `η Σ_i ∇F` (Eq. 1447, 1509, 1618) with `η` left as a single scalar. This is acceptable for a theory paper; the manuscript-vs-code level of abstraction is different from the audit-level. No finding.

CLAUDE.md asked whether §4 describes the `em_mode` table. Answer: §4 does not name `em_mode`, `skip_attention`, or the `'ift_phi'`/`'em_phi_q'`/`'em_phi_p'`/`'vfe_default'` taxonomy. Again, this is a theory paper, not a code-spec. No finding.

CLAUDE.md asked whether §4 acknowledges the RoPE × MahalanobisNorm known gap. Answer: §4 frames RoPE as a restriction to `SO(2)^{d_k/2}` and does not claim strict SE(K) covariance for the diagonal-σ path. The body at line 1863 actually goes further and points out the asymmetry between attention-side and value-side gauge in RoPE ("This decomposition is supported (but not required) by the full framework"). No over-claim. No finding.

CLAUDE.md's pointer to "line 1261 explicitly distinguishes canonical F from entropy-suppressed surrogate" is stale: line 1261 is a `Q_i K_j^T` identity, not the F-vs-surrogate distinction. The actual distinction is at line 766 (canonical row-Lagrangian with `τβ log(β/π)` entropy term), line 855 (statement that the entropy term is "essential: it is this term that makes the β-optimization yield the softmax rather than an arg-min"), and line 1354 (vacuum free energy `α_i KL(q_i ‖ p_i) - τ Σ_i log Z_i`). The manuscript correctly handles the canonical-vs-surrogate distinction. No finding for the manuscript; correct the CLAUDE.md pointer if it is used as a navigation aid.

## Novel-construction inventory

§4 introduces or specialises the following constructions that are not direct copies of standard literature; each should be (and mostly already is) labeled novel where appropriate:

- **Geometric bias `S(Ω)`** (Eq. 1075). The non-negativity claim is a property of `log + reciprocal` and is standard once you write it that way; the *role* `S(Ω)` plays as a pair-dependent bias inside the attention logits is novel as far as I know. Manuscript labels it correctly as a derived structure.
- **`O(K)` extension via per-agent diagonal reflections `Ω = R_i M_{ij} R_j`** (Eq. 1099, lines 1091–1112). This is the user's construction to reach the orientation-reversing component of `O(K)` that `exp(φ_i) exp(-φ_j)` cannot reach. The reduction to elementwise sign flips on embeddings (Eq. 1106) is clean.
- **Untied-QK carving from per-token frames** (§4.2.1, Eq. 1142). The decomposition `Q_i = U_i^{-1} μ_i, K_j = U_j^T Σ_j^{-1} μ_j` with the bilinear `M_{ij} = Ω_{ij}^{-T} Σ_j^{-1}` is the user's own derivation. Worth labelling explicitly as a novel route compared to the more standard "absorb-Ω-into-W_Q-W_K" reduction in §4.2.2.
- **State-dependent precision `α_i = c_0/(b_0 + D_KL(q_i ‖ p_i))`** (Eq. 937) as a generalisation of L2 weight decay. This is the user's construction. Manuscript labels appropriately.
- **Off-diagonal gauge mixing** (§4.4.3, Eq. 1786) as a generalisation of standard MHA. Manuscript labels appropriately.
- **Per-head κ_a temperature** (Eq. 1744) tied to per-head covariance via `κ_a ∝ σ_a²`. The connection between Σ-anisotropy and head-specific temperature is the user's construction.
- **Boltzmann-gate GLU interpretation of the FFN** (Eq. 1897). Status "S" in Table 1 — manuscript correctly labels this as structural correspondence rather than derivation.

## Open questions

- **Q1.** Does the `vfe/` package's hardwired gradient profile (`vfe_default` in CLAUDE.md table, `mu_p` attached, `sigma_p` frozen at embedding, `phi` updated via `_update_phi`) match the implicit assumptions of the §4.3 gradient expressions, in particular the natural-gradient flow at Eq. 1447 and Eq. 1525? §4 does not engage with this question, which is reasonable for a theory paper, but a reader bridging to the code may need a sentence at the start of §4.3 stating the gradient profile assumed.
- **Q2.** The §4.2.1 untied-QK carving (Eq. 1142) and the §4.2.2 trivial-frame reduction (Eq. 1238) "coincide on standard scaled dot-product attention" per line 1022. The two routes give different identifications: §4.2.1 gives `W_Q^{(i)} = U_i^{-1}, W_K^{(i)} = U_i^T Σ_i^{-1}` (token-dependent), §4.2.2 gives `W_Q W_K^T = σ^{-2} Ω^{-T}` (constant). Strictly, the §4.2.1 route assumes a per-token frame, while §4.2.2 assumes a shared frame. The "coincide" claim is true under the closure `Σ_j = U_j C U_j^T` plus shared `U`, so the statement is correct but the closure assumption is doing nontrivial work and should be made explicit at line 1022.
- **Q3.** The state-dependent precision discussion (Eq. 937, §3 sec:state_dependent_precision) and the constant-α reduction (line 1413) are alternatives, and §4.3 uses the constant-α form in Eq. 1416 onward while §4.3 also presents the general non-isotropic form (Eq. 1497) with `α_i` and `∂α_i/∂μ_i` terms. The two cases are differentiated correctly. Question: in the empirical experiments referenced (Section sec:glk_lm), which regime is used by default? Cross-reference to a config flag would clarify.

## Overall Verdict

**Major revisions.** The mathematical content of §4 is largely sound: the Gaussian KL, the Mahalanobis identity, the geometric bias non-negativity, the Rényi extension, the α-product-rule, and the multi-head decomposition all pencil out. The two routes to standard attention (untied-QK carving and the explicit three-limit dot-product derivation) are complementary and the manuscript appropriately notes that they "coincide on standard scaled dot-product attention." The envelope-theorem discussion at lines 857–872 correctly distinguishes the gradient of `F_red` from the autograd gradient of `Σ_j β_{ij}^* E_{ij}`, and the manuscript respects this distinction in subsequent sections.

Required revisions are presentational and definitional rather than mathematical:

1. **Table 1 status-column discipline (M-B-1, M-B-2):** several rows labeled "D" (derived) are actually (I) interpretive — the `W_Q W_K^T ↔ Ω^{-T}` identification is non-unique, and the Euler-step ↔ residual-connection correspondence is structural. Relabel as "S" or introduce a third status code. The body prose already does the right thing; only the table needs to inherit it.

2. **Natural-gradient metric on `gl(K)` (M-B-3):** §4.3 line 1632 invokes "Killing form or pullback metric" without specifying which. The Killing form is degenerate on `gl(K)` (vanishes on the `ℝ·I` center). One sentence in the body to commit to a metric choice (likely Frobenius / pullback Fisher) would resolve.

3. **RoPE abelian-subgroup disclosure (M-B-4):** line 1847's claim that the sign-flipped product "has exactly the form of a gauge transport operator" works only on the abelian `SO(2)^{d_k/2}` subgroup, where `Ω^{-T} = Ω`. State the restriction.

4. **Broken cross-reference (m-B-1):** `eq:beta_grad_phi` referenced at line 1940 but not defined. Define or move to Supp.

5. **File-level style cleanup:** banned `\,`/`\;`/`\!` spacing macros throughout; missing equation punctuation throughout. Single global pass.

After these revisions, §4 stands as a careful and largely defensible derivation of standard transformer attention as a limit of the gauge-theoretic VFE framework. The core thesis — that attention emerges from variational inference on Gaussian beliefs in local frames, with the standard transformer recovered under explicit isotropy + flat-bundle + constant-gauge limits — is supported by the derivations as currently written, modulo the presentational issues above.
