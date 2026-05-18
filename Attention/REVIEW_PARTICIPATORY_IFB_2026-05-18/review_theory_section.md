# Peer Review — Participatory_it_from_bit.tex, Theory Section (lines 175–2039) — 2026-05-18

## Summary

The theory section sets up a principal `G`-bundle picture of multi-agent inference, defines beliefs/priors/models/hyper-priors as smooth sections of associated bundles, introduces gauge transport `Ω_ij = exp(φ_i)exp(−φ_j)`, derives softmax attention as the stationary point of an entropy-regularized KL-consensus problem, writes a multi-agent variational free energy functional (Eq. eq:pointwise_free_energy and the boxed eq:free_energy_functional_final), reduces it to standard scaled dot-product attention in a zero-dimensional, isotropic-Gaussian, shared-frame limit, and develops a Hessian-based "mass analogy" for the second variation of `F` at the consensus point. The mathematical content is largely correct under the disclosures the manuscript already makes; the dominant remaining issues are framing: (i) the multi-agent functional is presented at §1011–1023 as the standard `[Friston2010, Parr2022]` free energy "extended" to multi-agent settings, but the inter-agent KL coupling and the attention-entropy term are absent from standard FEP and the (S)→(N) demotion is buried in §1034 rather than stated where the citation appears; (ii) the cross-scale shadow relation Eq. eq:cross_scale_shadow is repeatedly called a "theorem" when it is a definitional commitment; (iii) the two transformer-reduction routes are clearly disclosed, but Section sec:dot_product_derivation contains a slip at line 1761 that writes `W_Q W_K^T = (1/σ²) M ∈ GL(d_k)` even though the immediately following text (lines 1764–1779) shows this identification only holds on the per-head image subspace and the ambient `W_Q W_K^T` is rank-deficient; (iv) the mass analogy is now well-flagged as a stiffness + kinetic-postulate, but line 2032 uses the Killing form `−tr(φ²)` on `g` without restating that this is positive definite only for compact `G` (the body uses `gl(K)`); and (v) several minor sign/convention slips and one banned-phrase occurrence. The Lemma at thm:vanishing_holonomy is correct as stated (an algebraic cocycle identity, not a Maurer-Cartan derivation) and the Regime I / Regime II disclosures are conscientious. Verdict: minor-to-moderate revisions for framing in §1019, §531, §916, §1597, §1842, plus the targeted math fixes below.

## Standards against which the manuscript was reviewed

- [Friston2010] — single-agent variational free energy `F = E_q[log q − log p(o,s)]`.
- [ParrPezzuloFriston2022], [FristonEtAl2017] — active inference, hierarchical formulations.
- [AmariNagaoka2000], [Amari1998], [Amari2016] — Fisher-Rao metric, natural gradient, dual connections, SPD geometry.
- [Cencov1972] — uniqueness of Fisher metric.
- [BleiKuckelbirgJordan2017], [KingmaWelling2014] — variational inference, ELBO, Gaussian KL closed form.
- [Nakahara2003], [Frankel2011], [KobayashiNomizu] — principal bundles, associated bundles, connections, Maurer-Cartan form, parallel transport, transport laws for tensors.
- [Vaswani2017] — standard scaled dot-product attention `softmax(QKᵀ/√d_k) V`.
- [Tsai2019], [Ramsauer2021], [Millidge2021], [Bronstein2021] — alternative interpretations of attention.
- [AbsilMahonySepulchre2008] — manifold optimization, retractions, non-compact-group natural gradients.
- [DempsterLairdRubin1977] — EM separation.
- [BaKirosHinton2016] — layer normalization (reference for downstream sections, used in the transformer reduction §1745–1751).

Books cited without specific section numbers as I do not have direct access; citation pointers are descriptive.

## Major Issues

### M1. The pointwise free energy is cited as `[Friston2010, Parr2022]` at §1011–1023, but the multi-agent functional that immediately follows is novel and the (S)→(N) demotion is buried in §1034.

**Claim (manuscript, §1011–1015):** "The variational free energy principle [Friston2010, Parr2022] provides a tractable approximation to intractable Bayesian inference. ... The variational free energy provides an upper bound `F[q] = E_{q(x)}[log q(x) − log p(o,x)] ≥ −log p(o)`." §1019 then says "A single agent minimizing F[q] performs standard variational inference. Our framework extends this to multiple interacting agents..." and §1227–1239 writes the boxed pointwise functional Eq. eq:pointwise_free_energy containing `Σ_ij β_ij KL(q_i ‖ Ω_ij q_j) + τ β_ij log(β_ij/π_ij)` and the model-channel analogue.

**Claim kind:** (S) as written; the operative functional is (N).

**Standard treatment:** [Friston2010] standard `F = E_q[log q − log p(o,s)]`; equivalently `F = KL(q‖p(s|o)) − log p(o) = accuracy + complexity`. No inter-agent KL coupling; no `τ β log(β/π)` term; no gauge-transported "prior" `Ω_ij q_j`. Multi-agent FEP extensions in the published literature (e.g., variational ecology) use different couplings and do not contain the gauge-transport-coupled `Ω_ij` term.

**Problem:** The manuscript does disclose the status of the inter-agent coupling at §1034 ("best read as a consensus-energy ansatz rather than as a generative-model derivation"). That disclosure is essential and correct. The issue is its placement: a reader who finishes §1019 ("Our framework extends...") has been told they are reading an extension of `[Friston2010]`, but the actual functional that follows in Eq. eq:pointwise_free_energy is (N), not (S). The §1034 disclosure recovers the framing only for the inter-agent term, not for the attention-entropy term and not for the global claim "this is FEP". The boxed Eq. eq:free_energy_functional_final at §1248–1259 reads, on a casual scan, as Friston's F with extra terms — but it is its own functional with its own derivation.

**Required revision:** Insert a one-sentence disclosure at the head of §1019: "The multi-agent functional Eq. eq:pointwise_free_energy is a novel extension of Friston's single-agent F; the inter-agent KL coupling `Σ_ij β_ij KL(q_i ‖ Ω_ij q_j)`, the attention-entropy term `τ β_ij log(β_ij/π_ij)`, and the model-channel analogue with `γ_ij` are not present in the standard FEP literature [Friston2010, ParrPezzuloFriston2022]. The construction is internally consistent and we display the derivation of `β*` from the entropy-regularized row Lagrangian in Section sec:mixture_derivation; the framing as an extension of FEP refers to the single-agent self-term `KL(q_i‖p_i) − E_{q_i}[log p(o|k_i)]`, not to the alignment terms." Then keep §1034 as the detailed sub-disclosure on the consensus-energy ansatz status.

### M2. Cross-scale shadow Eq. eq:cross_scale_shadow is repeatedly called a "theorem" when it is a definition.

**Claim (manuscript):** §535–541 defines `p_i^{(s)}(c) = Ω_{i,I}[q_I^{(s+1)}](c)` and `r_i^{(s)}(c) = Ω̃_{i,I}[s_I^{(s+1)}](c)`. §541: "As a consequence, p and q live on the same statistical manifold B_state (so the matched-bundle property of Section sec:working_framework is a theorem rather than a simplifying assumption)". §918: "Beliefs and priors share the latent-state manifold B_state ... as theorems following from the cross-scale shadow relation eq:cross_scale_shadow rather than as separate assumptions."

**Claim kind:** (S)→(N). The "theorem" language hides the (N) status of the relation itself.

**Standard treatment:** [ParrPezzuloFriston2022 Ch. 9], [Friston2017Graphical]: hierarchical variational inference uses a generative model `p(o, s_1, ..., s_L)` in which `p(s_ℓ | s_{ℓ+1})` is part of the model and the level-ℓ posterior is `q(s_ℓ)`, computed by minimizing F. The level-ℓ prior is not (in general) the transported level-(ℓ+1) posterior. Substituting `p(s_ℓ) ← Ω_{ℓ, ℓ+1} q(s_{ℓ+1})` is a strong structural commitment.

**Problem:** The matched-bundle property `p, q ∈ B_state` is trivially true given the definition Eq. eq:cross_scale_shadow that puts them on the same manifold by construction. Calling that consequence a "theorem" is correct in a vacuous sense (a definition has consequences) but misleading: the operative content is the definition Eq. eq:cross_scale_shadow itself, which is a (N) postulate that does not follow from FEP and is not standard in hierarchical variational inference. The same wording recurs at §918 and is part of the working-framework simplification list (§910).

**Required revision:** Re-word §541 to: "Because the prior `p_i^{(s)}` is defined as the gauge transport of the meta-agent's posterior `q_I^{(s+1)}`, beliefs and priors share the latent-state manifold `B_state` by construction. This is a structural commitment of the framework, not a theorem of standard hierarchical variational inference, which instead derives the level-ℓ prior from a generative-model conditional `p(s_ℓ | s_{ℓ+1})`." Apply the same edit at §918. Add a remark at §531 contrasting the cross-scale shadow with the standard hierarchical-generative-model scheme and noting that the reduction (or approximation) of one to the other is not displayed.

### M3. Rank-deficient `W_Q W_K^T` slip at line 1761.

**Claim (manuscript, line 1761):** "`W_Q W_K^⊤ = (1/σ²) M ∈ GL(d_k)`, which holds directly when `W_Q, W_K ∈ R^{d_k × d_k}` are square per-head projections."

**Claim kind:** (R) reduction to standard attention.

**Standard treatment:** [Vaswani2017]: `W_Q, W_K ∈ R^{d_model × d_head}` are *rectangular*, with `d_head = d_model / H < d_model` for `H > 1`. The product `W_Q W_K^T ∈ R^{d_model × d_model}` has rank at most `d_head < d_model`, so it is *not* an element of `GL(d_model)`; it is a low-rank matrix in `R^{d_model × d_model}`.

**Problem:** Line 1761 boxes the identification `W_Q W_K^T = (1/σ²) M ∈ GL(d_k)` and then says it "holds directly when `W_Q, W_K ∈ R^{d_k × d_k}` are square per-head projections". This is a hypothetical square case; the actual transformer architecture uses rectangular projections. The manuscript handles the rectangular case in the very next sentence via thin-SVD (lines 1764–1779) and correctly notes "rank-deficient and therefore not an element of `GL(d_{model})`". But Eq. eq:wqwk_square as boxed is the wrong form for the standard transformer reduction — it holds only for an architecture (square per-head projections) that standard transformers do not use. The corresponding text in `GL(K)_attention.tex` is recorded as having an analogous fix; the present manuscript inherits the slip.

**Required revision:** Replace the boxed Eq. eq:wqwk_square at line 1761 with "`A_Q A_K^T = (1/σ²) M_h ∈ GL(d_head)` (the invertible head-space factor of the thin-SVD decomposition introduced below)", and demote the square-projections case to a remark: "if (and only if) the per-head projections were square, `M_h` would equal `W_Q W_K^T` directly." This makes the canonical reduction the head-space-kernel form `M_h` rather than the rank-deficient ambient product. Alternatively keep the boxed form but explicitly tag it "(only the artificial square-projection case; standard transformer is rectangular and uses thin-SVD as below)".

### M4. The `τ = κ √K` factorization at line 1242 is partially disclosed but not at the canonical reduction point §1797–1808.

**Claim (manuscript, §1242):** "In the working implementation the temperature is factorised as `τ = κ√K`, with `κ` a learnable scalar and the `√K` factor the dimension scaling familiar from scaled dot-product attention." §1802 (in sec:dot_product_derivation): "`τ = √d_k`."

**Claim kind:** (R) reduction to [Vaswani2017].

**Standard treatment:** [Vaswani2017 §3.2.1]: the temperature is `√d_k`, justified by the unit-variance / dot-product argument. There is no learnable `κ` in standard transformers.

**Problem:** §1797–1808 ("Temperature scaling") derives `τ = √d_k` for the reduction to standard transformers, but the rest of the manuscript uses `τ = κ √K` (§1242 and elsewhere). The two are reconciled only if `κ = 1`. The manuscript does not state, at §1797 or in the boxed reduction Eq. (line 1807), that the gauge-theoretic temperature carries a learnable `κ` beyond the standard scaling, and therefore that the reduction to `softmax(QK^T / √d_k)` is up to a learnable scalar. The abstract similarly says "up to a separately introduced learned bilinear compatibility M and the standard normalisation and bias assumptions"; it should add "and up to a learnable temperature κ on top of the dimensional `√K` scaling."

**Required revision:** At §1802, replace "`τ = √d_k`" with "`τ = κ√d_k` with `κ` a learnable scalar of order 1; standard transformer attention [Vaswani2017] corresponds to fixing `κ = 1`." Mirror this in the boxed reduction Eq. (line 1807) caption.

### M5. The mass analogy section at §1842+ is well-flagged but the kinetic-postulate disclosure should be at the top of the section heading.

**Claim (manuscript, §1842–1847):** The opening paragraph and the "What the Hessian gives, and what is added" paragraph correctly identify the Hessian as a stiffness, the kinetic-metric reuse of the same matrix as a postulate, and the empirical match of `ω² ∝ 1/m_eff` as contingent on the postulate. §2027 ("This is a postulate, not a consequence of F") repeats this.

**Claim kind:** (I) interpretive / (N) novel postulate.

**Standard treatment:** Classical mechanics: `L = T − V`, dispersion `ω² = k/m`. The Hessian of `V` gives `k` (stiffness), the coefficient of `½ ẋ² in T` gives `m` (inertia). These are independent objects.

**Problem:** Once the reader reaches §2011–2022 ("Within-Framework Interpretation"), the language is "the effective mass of agent i in the mean sector is M_i = Λ̄_{p,i} + Σ_k β_{ik} Λ̃_{q,k} + ..." (Eq. eq:effective_mass) and uses the word "mass" throughout. The fact that this is a stiffness, not a mass, is recoverable only by following the §1847 disclosure forward; the local reading of §2011–2022 sounds like a derivation of inertial mass. The empirical scaling `ω² ∝ 1/m_eff` is reported in Section sec:mass which the present range does not cover, but the precision-as-mass identification is introduced here.

**Required revision:** Add a one-line reminder at the start of §2011 ("Within-Framework Interpretation: Stiffness as Precision"): "We reuse `M_i` as both the stiffness (Hessian of `F`) and the inertia (kinetic-metric coefficient, postulated in §sec:velocity_quadratic). Without the kinetic-metric postulate, `M_i` is only a stiffness." This is the substance of §1847 but placed where the reader first encounters the word "effective mass". Also rename the boxed Eq. eq:effective_mass label to "effective stiffness" or "stiffness/mass" to keep the dual reading explicit at the equation level.

### M6. Killing form on `gl(K)` is indefinite, but Eq. line 2032 writes the kinetic-metric `−tr(φ²)` without restating this.

**Claim (manuscript, line 2032):** "where the gauge frame `φ ∈ g` carries the Killing form metric `⟨φ̇, φ̇⟩_g = −tr(φ̇²)`."

**Claim kind:** (S) standard for compact `g`; (N) / problematic for non-compact `g`.

**Standard treatment:** [AbsilMahonySepulchre2008]: for a Lie algebra `g`, the Killing form `B(X, Y) = tr(ad_X ad_Y)`. For `so(K)` (compact), `B` restricted to the algebra is negative-definite, so `−B` is a valid positive-definite bi-invariant inner product. For `gl(K, ℝ)` (non-compact), `B` is indefinite: it has a non-trivial kernel (the center `R · I`) and is sign-indefinite on `sl(K)`. There is no bi-invariant Riemannian metric on `GL(K)`. The manuscript acknowledges this honestly at §2099 ("for non-compact `G` (such as `GL^+(K)`, where the Killing form on `gl(K)` is indefinite) no bi-invariant Riemannian metric exists") and at §2582–2589.

**Problem:** Eq. line 2032 in the theory body writes `⟨φ̇, φ̇⟩_g = −tr(φ̇²)` without flagging that this is positive definite only on the compact part (i.e., on the antisymmetric `so(K)` subalgebra). For `gl(K) = so(K) ⊕ sym(K)`, `−tr(φ²) = −tr((φ_a + φ_s)²) = −tr(φ_a²) − tr(φ_s²) − 2 tr(φ_a φ_s) = −tr(φ_a²) − tr(φ_s²)` (the cross term vanishes), and since `−tr(φ_a²) ≥ 0` (good) but `−tr(φ_s²) ≤ 0` (sign-indefinite, since `φ_s²` is symmetric and positive-semidefinite when `φ_s` is symmetric, so `tr(φ_s²) ≥ 0` and `−tr(φ_s²) ≤ 0`), the proposed kinetic metric is *not* positive definite on `gl(K)`. It is positive *semi*-definite when restricted to `so(K)` (and indeed it is just `½ ‖φ_a‖²_F`).

**Required revision:** At line 2032 insert a parenthetical "(valid for compact `G = SO(K)`; for the general `GL(K)` case the kinetic-metric must be the position-dependent right-invariant form of Eq. line 2582–2589 because the Killing form is indefinite on `gl(K)`)". This is consistent with what §2099 and §2582 already say; the body §1842–2039 mass section just needs the cross-reference.

## Minor Issues

### m1. §1086–1099 Lagrangian derivation introduces τ post hoc.

The mixture-of-sources Lagrangian Eq. line 1085 uses `F_align = Σ_j β_ij(E_ij + log β_ij − log π_j)` with implicit `τ = 1`. The temperature is reintroduced at §1107 by rescaling `E_ik → E_ik/τ`. This is mathematically equivalent to using `F_align = Σ_j β_ij(E_ij + τ log β_ij − τ log π_j)` from the outset, but the latter form is the one that matches the canonical functional Eq. eq:pointwise_free_energy and the boxed Eq. eq:free_energy_functional_final. Recommend deriving the Lagrangian with τ explicit; the algebra is identical and the reader does not have to track the rescaling later.

### m2. §1066 KL identification step is correct but informal.

Eq. line 1066: "`∫ dk q_i(k) log(q_i(k)/P(k|z=j)) = D_KL[q_i ‖ Ω_ij q_j]`". The reader should be reminded that `P(k|z=j) = (Ω_ij q_j)(k)` was *defined* as the transported density (line 1044), so this is by definition, not by computation. Recommend "by the construction `P(k|z=j) := (Ω_ij q_j)(k)` of the mixture model".

### m3. §1116–1122 causal-mask prior `π_j^(causal)`.

Eq. line 1119 writes `π_j^(causal) = 1/i` for `j ≤ i`. This is correctly normalized over the active set `{j : j ≤ i}` (which has size `i`). Standard transformer masking uses an additive `−∞` bias on the logits; the manuscript correctly notes "softmax(x + log π) ∝ π · softmax_input(x)". The manuscript reduces to the standard masking exactly only when the active-set normalization is treated as a multiplicative factor that cancels in the softmax row; the current form `π = 1/i` introduces a token-position-dependent normalization that is unnecessary (any constant `π > 0` for `j ≤ i` and `π = 0` otherwise yields the same softmax). Recommend writing `π_j^(causal) ∝ 1[j ≤ i]` to avoid the `1/i` constant that has no effect.

### m4. §1234 KL closed form is correct but uses the symbol `K` for both "dimension" and the latent variable in places.

Eq. eq:gaussian_kl at line 525 writes the closed-form Gaussian KL with `K` as the dimension. The mixture-of-sources construction §1037 uses `k` as the latent variable. The notation block at §178–189 acknowledges symbol overloading. No mathematical issue, just remind the reader at the first use of `K` for dimension that it is unrelated to the latent `k`.

### m5. §1217 cocycle proof.

Lemma thm:vanishing_holonomy at §1201–1215 is correctly stated as an algebraic cocycle identity from the vertex-local parameterisation. The accompanying remark §1217 says "In the gauge-theoretic dictionary, it expresses flatness of the Regime I connection: with `A^(i)_μ = U_i^{-1} ∂_μ U_i` the Maurer-Cartan identity forces `F^(i)_μν ≡ 0`". This is correct: `F = dA + A ∧ A` for `A = U^{-1} dU` gives `F = d(U^{-1} dU) + (U^{-1} dU) ∧ (U^{-1} dU) = −U^{-1} dU ∧ U^{-1} dU + U^{-1} dU ∧ U^{-1} dU = 0`. No issue with the statement; the proof at line 1213 is the cleanest form ("direct computation") and is correct.

### m6. §1235–1236 `tau` is reused for the temperature; §1497 reuses `τ` for time.

Notation block §187 acknowledges this: "`τ` without subscript is the attention/entropy temperature in the canonical free energy; `τ_i` is the dimensionless information-update clock of Section sec:fisher_arc_length". §1497 ("Dynamical Structure and Emergent Timescales") uses the temperature τ implicitly while also discussing timescales. The notation is internally consistent given §187 but the reader will benefit from a parenthetical reminder at §1497 ("note: τ in this and subsequent equations denotes the attention temperature of Eq. eq:pointwise_free_energy, not the timescale τ_i of Section sec:fisher_arc_length").

### m7. §1241 mention of "the entropy-suppressed surrogate" is mathematically correct and the gradient-mismatch identity `−τ^{-1} Cov_β(KL, ∇KL)` is the right form.

The covariance form is `Cov_β(KL, ∇KL) = E_β[KL · ∇KL] − E_β[KL] · E_β[∇KL]`. The displayed sign and the conclusion that the surrogate and the reduced functional do not share stationary points off equilibrium is standard. No issue here; this is one of the cleaner parts of the section.

### m8. §1416–1419 mean-gradient-vs-full-equivalence proposition is well stated.

The disclosure that the environmental-agent KL replacement gives mean-gradient equivalence but not full variational equivalence (with the entropy `−½ Σ_i^{-1}` mismatch) is correct and matches what §1429–1441 derives. Standard active-inference observation likelihoods are cross-entropies, not KLs; the manuscript's choice to substitute cross-entropy at §1437–1439 to recover full equivalence is the right move.

### m9. §1454–1460 structural-analogy table.

The table footnote at line 1459 correctly notes that `F` is dimensionless while physical actions have units of action. The "natural-gradient flow rather than Euler-Lagrange equations" is correct. No issue.

### m10. §1655 `S(Ω_{ij}) ≥ 0` with equality iff `Ω ∈ O(K)`.

The claim is that `S(Ω) = ½[log det(ΩΩ^T) + tr((ΩΩ^T)^{-1}) − d_k] ≥ 0` with equality iff `Ω` is orthogonal. The standard fact is: for any positive-definite `A`, `½[log det A + tr(A^{-1}) − d]` evaluated at `A = ΩΩ^T` is a Bregman-divergence-like quantity. Let `λ_1, ..., λ_d` be eigenvalues of `ΩΩ^T` (all positive). Then `½[log det A + tr(A^{-1}) − d] = ½ Σ_i [log λ_i + 1/λ_i − 1]`. Each term `log λ + 1/λ − 1 ≥ 0` with equality iff `λ = 1` (Bregman duality / convexity of `log λ + 1/λ`). So `S(Ω) ≥ 0` with equality iff all eigenvalues of `ΩΩ^T` are 1, i.e. `ΩΩ^T = I`, i.e. `Ω ∈ O(K)`. The manuscript's claim is correct. (Minor: the formula `S = ½[log det + tr(inverse) − d]` is the Burg/Itakura-Saito divergence applied to `(ΩΩ^T, I)`; a citation to the information-geometry literature would help, but is optional.)

### m11. §1604–1610 Dirac-KL divergence at `δ(k − μ_i)` vs `δ(k − μ_j)`.

The manuscript says `KL(δ_a ‖ δ_b) = +∞` for `a ≠ b`. Strictly, when `q` and `p` are mutually singular, the KL is `+∞`; when they coincide, it is `0`. The manuscript's "absolute continuity" argument at line 1610 is correct. No issue.

### m12. §1659 frame role.

At line 1659 the manuscript writes "Rather than transporting the key belief into the query's frame via `Ω_ij` and measuring with the induced metric, the KL divergence equivalently transports the query into the key's frame via `Ω_ij^{-1}` and measures the Euclidean distance there." This is the identity `‖Ω^{-1}μ_i − μ_j‖² = (Ω^{-1}μ_i − μ_j)^T (Ω^{-1}μ_i − μ_j) = (μ_i − Ωμ_j)^T (ΩΩ^T)^{-1} (μ_i − Ωμ_j)` (line 1621), which is correct for `Ω` invertible. No issue; the equivalence is exact.

### m13. Eq. eq:full_kl_general (line 1677) carving.

The decomposition `D_KL(q_i ‖ Ω_{ij#} q_j) = ½[log det Σ_j − log det Σ_i + 2 log|det U_i| − 2 log|det U_j| − d_k + tr(H_j P_i) + x_i^T H_j x_i − 2 x_i^T k_j + r_j]` with `x_i = U_i^{-1} μ_i`, `P_i = U_i^{-1} Σ_i U_i^{-T}`, `H_j = U_j^T Σ_j^{-1} U_j`, `k_j = U_j^T Σ_j^{-1} μ_j`, `r_j = μ_j^T Σ_j^{-1} μ_j` is asserted as "verified symbolically against the direct Gaussian KL to machine precision". I have not re-verified this; the structure of the terms is consistent with the Gaussian KL identity, and the "verified symbolically" claim is reasonable but is the kind of claim a reviewer would normally re-derive. If the author has a sympy or numerical verification artifact, citing it would strengthen this. Otherwise mark as "(verified by symbolic computation, derivation omitted)".

### m14. §1700 `M_{ij} := Ω_{ij}^{-T} Σ_j^{-1} = U_i^{-T} U_j^T Σ_j^{-1}`.

The claim is that as `(U_i, U_j) ∈ GL(d_k)^2` and `Σ_j ∈ SPD(d_k)` vary, `M_{ij}` ranges over all of `GL(d_k)`. This follows from polar decomposition / SVD: any `M ∈ GL(d_k)` can be written as a product of three factors `U_i^{-T} U_j^T Σ_j^{-1}` for some choice of `U_i, U_j, Σ_j`. The manuscript is correct that the gauge-derived bilinear has the same expressive power as a freely learned `W_Q W_K^T ∈ GL(d_k)`. No issue. The "generically non-symmetric `M_{ij}`" claim is also correct.

### m15. §1862 dual transport law for precision.

Eq. eq:precision_transport `Λ̃_{q_k} := (Ω_{ik} Σ_k Ω_{ik}^T)^{-1} = Ω_{ik}^{-T} Λ_{q_k} Ω_{ik}^{-1}` is the correct dual transport. The standard sandwich for covariance (a (2,0)-tensor with respect to the defining GL(K) representation) is `Σ → Ω Σ Ω^T`; for the precision (the inverse, a (0,2)-tensor) it is `Λ → Ω^{-T} Λ Ω^{-1}`. The manuscript correctly distinguishes these, flags that for orthogonal `Ω` they coincide, and uses the GL-valid form throughout the mass section. This is one of the cleaner discipline points of the manuscript.

## Math Reviewer Items

### MR-1. Eq. eq:gaussian_kl at line 525.

`KL(N(μ_1, Σ_1) ‖ N(μ_2, Σ_2)) = ½[log(|Σ_2|/|Σ_1|) + tr(Σ_2^{-1} Σ_1) + (μ_2 − μ_1)^T Σ_2^{-1} (μ_2 − μ_1) − K]`. Matches [BleiKuckelbirgJordan2017 / KingmaWelling2014 Appendix B] with `(q, p) ↔ (1, 2)`. Correct.

### MR-2. Eq. eq:transport_def at line 778.

`Ω_{ij}(c) = exp[φ_i(c)] exp[−φ_j(c)] ∈ G`. The manuscript correctly does *not* write `exp(φ_i − φ_j)`; the product-of-two-exponentials form is BCH-correct. §790 explicitly notes that the Lie-algebra additive shorthand `φ_i ↦ φ_i + ξ` coincides with the group-level form only for commuting `ξ`. Correct.

### MR-3. Eq. eq:wqwk_square at line 1761.

`W_Q W_K^T = (1/σ²) M ∈ GL(d_k)` is misstated as applying to standard transformer architectures, which use rectangular `W_Q, W_K ∈ R^{d_model × d_head}`. The boxed equation is mathematically correct in the artificial square-projections case but is not the canonical reduction. Fix per M3 above.

### MR-4. §1655 geometric bias `S(Ω) ≥ 0`.

`S(Ω) = ½ Σ_i [log λ_i + 1/λ_i − 1] ≥ 0` with equality iff `λ_i = 1` ∀ i, i.e. iff `ΩΩ^T = I`. Correct.

### MR-5. §1903 sender-side covariance gradient.

`∂ KL(q_i ‖ Ω_{ik#} q_k) / ∂ Σ_k = ½ Ω_{ik}^T [Λ̃_{q_k} − Λ̃_{q_k}(Σ_i + d̃_{ik} d̃_{ik}^T) Λ̃_{q_k}] Ω_{ik}` with `d̃_{ik} = Ω_{ik} μ_k − μ_i`. The chain rule via `∂ Λ̃ / ∂ Σ_k = −Λ̃ ⊗ Λ̃` and `∂ Σ̃_k / ∂ Σ_k = Ω ⊗ Ω` is correct. The vanishing of `d̃_{ik} d̃_{ik}^T` at consensus is correct and licenses the clean boxed form Eq. eq:mass_mu_diagonal. Correct.

### MR-6. Eq. eq:mass_mu_offdiagonal at line 1952.

`[M_{μμ}]_{ik} = −β_{ik} Ω_{ik}^{-T} Λ_{q_k} − β_{ki} Λ_{q_i} Ω_{ki}^{-1}` for `i ≠ k`, using the GL-correct precision transport. Reduces to `−β_{ik} Ω_{ik} Λ_{q_k} − β_{ki} Λ_{q_i} Ω_{ki}^T` under `O(d)`. The substitution `Ω^{-T} → Ω` valid only under orthogonal transport is correctly flagged. Correct.

### MR-7. §2032 kinetic-metric on `g`.

`⟨φ̇, φ̇⟩_g = −tr(φ̇²)`. Positive definite on `so(K)` (compact, antisymmetric); sign-indefinite on `gl(K) = so(K) ⊕ sym(K)`. Fix per M6.

### MR-8. Eq. eq:cross_scale_shadow at line 535.

The transport-of-meta-agent-posterior definition is internally consistent under `Ω_{i, I}` cross-scale transport. The unification at §885 ("the meta-agent frame construction `φ_I = Σ w_i φ_i` keeps `φ_I` in the same Lie algebra `g`") is the Karcher-mean / barycenter ansatz; the appendix discussion at §2099 correctly flags that for non-compact `G` no bi-invariant Riemannian metric supports a clean Karcher mean and the additive form is the first-order BCH approximation. The cross-scale transport definition itself is novel; flag per M2.

### MR-9. Conditional uniqueness theorem of Appendix sec:conditional_uniqueness.

Referenced at §1108 and §1241 as the justification for choosing the forward KL among the f-divergence family. Out of scope (this is the appendix); flagged only that the assumptions (i)-(iii) "locality, linear coupling, and exponential-family closure of the minimizer" do load-bearing work here and should be verifiable from the appendix. I have not reviewed the appendix in this pass.

## Editorial / Style

- §552, line 552: "the framework employs them in two distinct roles that the reader must keep separate" — this is fine, but the surrounding paragraph at §552 contains "the same symbol `φ_i` does double duty in this manuscript" and the paragraph runs to ~400 words in a single block. Recommend splitting into the (a) redundancy reading and (b) state reading paragraphs with a paragraph break between them.
- §572, line 574: "the framework as constructed below operates exclusively in local trivializing patches" — the paragraph is dense and would benefit from breaking after "satisfy the standard Čech cocycle condition `Ω_{ij} Ω_{jk} = Ω_{ik}` on triple overlaps".
- §677, line 677: "Perfect consensus is an idealization that is not the criterion for meta-agent formation: imposed pointwise it is the epistemic-death limit of Section sec:epistemic_collapse, not a healthy coarse-graining target." This sentence opens a paragraph by negating the previous paragraph's headline claim — recommend rewriting to lead with the realistic criterion: "The criterion that licenses a meta-agent is graph-weighted average internal disagreement modulo gauge transport, with closure relative to the outside; imposing perfect consensus pointwise (the previous paragraph's idealization) is the epistemic-death limit of Section sec:epistemic_collapse rather than a healthy coarse-graining target."
- §745 "this distinguishes epistemic death (informational equilibrium) from naive coordinate singularities (gauge-dependent artifacts)" — banned phrase scan: "naive" is fine. No banned-phrase occurrences in 175–2039.
- §1175 "The framework provides a uniform encoding of empirically successful position-prior families" — slightly self-congratulatory. Recommend "Each of these mechanisms is recovered as a special case of Eq. eq:softmax_attention_general by a particular choice of `π_j`; the derivation does not select among them on first-principles grounds, so the prior remains a modelling decision."
- §1382 "The agent's own generative model `s_i` is regulated separately by its hyper-prior `r_i`" — clear and good.
- §1842 "The framework presented in Section sec:framework defines a variational free energy F that serves as the potential for agent dynamics" — "serves as the potential" is the soft form of "is the potential"; given the §1842–1847 disclosure about stiffness vs mass, this opening sentence anticipates the disclosure correctly. No change.
- Equation punctuation: lines 525, 778, 816, 1234, 1259, 1276, 1346, 1616, 1648, 1655, 1677, 1689, 1693, 1700, 1707, 1717, 1724, 1789, 1807, 1836, 1869, 1880, 1888, 1896, 1902, 1915, 1939, 1953, 1976, 1988, 2008, 2019, 2030, 2037 — most display equations terminate without comma/period. Project style requires equation punctuation. This is a global cleanup item; not done.
- Banned LaTeX spacing macros `\;`, `\,`, `\!`: I did a scan for `\;`, `\,`, `\!` in lines 175–2039 and found none. Good.
- No `key insight`, `crucially`, `critically`, `notably`, `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores` occurrences in 175–2039. Good — the discipline is being maintained.

## Citation Verification

- [✓] [Friston2010] — references.bib line 562. Cited at §1011, §1013, §1015, §1224, §1382, §1392. Standard form `F = E_q[log q − log p(o,s)]` is correctly represented at §1014–1017.
- [✓] [Parr2022] — references.bib line 580. Cited at §1011, §1224, §1392.
- [✓] [Cencov1982] — references.bib line 1785. Cited at §505 for uniqueness of Fisher metric. The user uses the date 1982 (the English translation); the original Russian work is 1972. Either is acceptable.
- [✓] [Nakahara2003] — references.bib line 661. Cited at §586.
- [✓] [Frankel2011] — references.bib line 669. Cited at §586.
- [✓] [Amari2016] — references.bib line 609. Cited at §924, §1517.
- [✓] [amari2016information] — references.bib line 2188. Cited at §931. Duplicate of Amari2016; cleanup suggestion: pick one.
- [✓] [Dennis2025trans] — references.bib line 1506. Cited extensively as the companion paper.
- [✓] [LahavNeemeh2022], [LahavNeemeh2025] — references.bib lines 2737, 2748. Cited at §765, §767, §771.
- [✓] [DonnellyFreidel2016] — references.bib line 426. Cited at §558.
- [✓] [BartlettRudolphSpekkens2007], [Vanrietvelde2020] — references.bib lines 437, 448. Cited at §558.
- [✓] [Rovelli1996] — references.bib line 415. Cited at §558.
- [✓] [WilsonConfinement1974], [KogutSusskind1975], [Creutz1983] — references.bib lines 869, 879, 889. Cited at §833, §873.
- [✓] [press2022train] (ALiBi), [beltagy2020longformer], [jiang2023mistral] (Mistral), [raffel2020exploring] (T5), [radford2019language] (GPT-2) — all present and correctly cited at §1130, §1140, §1156, §1172.
- [✓] [Wilson1971], [Cardy1996] — references.bib lines 1136, 1145. Cited at §887.
- [✓] [Born1927], [Carr1981] — references.bib lines 1705, 1715. Cited at §1546.
- [✓] [Nickel2017] — references.bib line 1728. Cited at §985.
- [?] [Vaswani2017] — not directly cited by name in lines 175–2039 of the theory section, but the "scaled dot-product attention", "softmax attention", and "standard transformer" claims at §1242, §1567, §1597, §1807, §1836 are about [Vaswani2017]. The lack of an explicit citation in the theory section is a soft omission — the manuscript should cite [Vaswani2017] at §1597 when first referring to "Standard Transformers".

## Manuscript ↔ Code Consistency

- Eq. eq:transport_def `Ω_{ij} = exp(φ_i) exp(−φ_j)` matches `transformer/core/transport_ops.py:408` `Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)`, computed as a product of two separately-formed matrix exponentials via `stable_matrix_exp_pair` (line 373). The code does not collapse to `exp(φ_i − φ_j)`. Consistent.
- Eq. line 603 `ρ(g) · N(μ, Σ) = N(g μ, g Σ g^T)` (sandwich on Σ) matches the codebase convention enforced at `transformer/core/attention.py:563–564`, `transformer/core/block_equivariant_mixer.py:18, 177`, and `transformer/core/block_config.py:512`. The "single most common correctness bug" (per CLAUDE.md) is correctly avoided in code. Consistent.
- Eq. eq:precision_transport `Λ̃ = Ω^{-T} Λ Ω^{-1}` (line 1859) is the dual-transport law for precision. I have not located a specific code line that exercises this; the codebase mostly uses covariance directly (sandwich `Ω Σ Ω^T`) and inverts when needed. No inconsistency observed; verify if the mass-matrix off-diagonal blocks Eq. eq:mass_mu_offdiagonal at line 1952 are ever computed in code (they appear to be a theory-only construct for the mass analogy section, with empirical validation in Section sec:mass operating in the `β_{ij} = 0` isolated-agent limit per §1932).
- Eq. eq:free_energy_functional_final at §1248–1259 is the canonical functional. The codebase's `transformer/vfe/free_energy.py` (not read in this pass but indexed by codebase_map) implements the operative simplification `γ_ij = 0`, `λ_h = 0`, slow subsystem frozen, per §625 and §1262 — consistent with the manuscript's frozen-slow-subsystem regime.
- Eq. eq:beta_optimal at line 1264 `β*_{ij}(c) = χ_{ij}(c) π_{ij}(c) exp[−KL/τ] / Σ_k χ_{ik}(c) π_{ik}(c) exp[−KL/τ]` matches the attention computation in `transformer/core/attention.py` (e.g., the `compute_attention_weights` routines), modulo project-policy multi-head decomposition. The `τ = κ √K` factorization in code matches §1242.

## Novel-construction inventory

- Multi-agent free energy with inter-agent KL coupling `Σ_ij β_ij KL(q_i ‖ Ω_ij q_j)` (§1023, Eq. eq:pointwise_free_energy). Novel; not in [Friston2010, ParrPezzuloFriston2022]. See M1.
- Attention-entropy term `τ β_ij log(β_ij/π_ij)` as part of the free energy functional (Eq. eq:pointwise_free_energy). Novel as part of FEP; standard as an entropy-regularized Lagrangian for the constrained-softmax problem (Sinkhorn / OT literature).
- Cross-scale shadow definition Eq. eq:cross_scale_shadow (§535). Novel; not in standard hierarchical variational inference. See M2.
- Two-exponential transport `Ω_{ij} = exp(φ_i) exp(−φ_j)` (Eq. eq:transport_def). Novel parameterization; standard observation that this is *not* surjective onto `GL^+(K)` is correctly flagged at §973.
- Mixture-of-sources construction with `P(k|z=j) = Ω_{ij} q_j` (§1037–1044). Novel; the "status of the construction" note at §1034 correctly tags it as a consensus-energy ansatz, not a generative-model derivation.
- Edge-relaxed cocycle and Wilson observable (§823–875). Novel as a research extension; well-flagged as Regime II content with the lattice-gauge-theory citations.
- "Mass analogy" / precision-as-stiffness construction (§1842–2039). Novel; well-flagged as a kinetic-postulate + Hessian.
- Untied-QK carving from per-token frames (Eq. eq:gauge_qk at line 1693). Novel as a derivation; the identification with RoPE at §1708 is interpretive.
- Renyi-α attention extension (Eq. eq:renyi_attention_itfb at line 1635). Novel; the closed-form for diagonal Gaussians is standard. The framework's "carries over by replacing `∂ D_KL` with `∂ D_α`" claim is plausible but should be qualified as "all results in this section that depend only on the differentiability of `D_KL`".
- GL(K) gauge group with non-orthogonal Σ-transport (§975, §989). Standard in that GL(K) is the linear sector of the affine Gaussian symmetry; novel as the gauge group of an attention construction.

## Open questions

- §531 cross-scale shadow: the manuscript provides Eq. eq:cross_scale_shadow as a *definition*, but does not provide a *derivation* of how (or whether) this approximates standard hierarchical variational inference with `p(s_ℓ | s_{ℓ+1})` as part of a generative model. A reduction in some limit (e.g., delta-prior `p(s_ℓ | s_{ℓ+1}) = δ(s_ℓ − Ω q_{ℓ+1})`) would tighten the (S)→(N) story.
- §1700 `M_{ij} := Ω_{ij}^{-T} Σ_j^{-1}` ranges over `GL(d_k)` "via polar decomposition". A short proof at §1700 (or in the appendix) would help; it follows from `M_{ij} = U_i^{-T} U_j^T Σ_j^{-1}`, with `U_i, U_j` independent `GL(d_k)` elements and `Σ_j^{-1}` an independent SPD element — together these parameterize all of `GL(d_k)^2 × SPD(d_k)`, and the product map is surjective onto `GL(d_k)`. The argument is standard but not displayed.
- §1684 "verified symbolically against the direct Gaussian KL to machine precision" — if a sympy / pencil-and-paper artifact exists, citing it (even informally) makes the claim re-checkable.
- §2032 Killing-form kinetic metric on `g`: should be restricted to compact `G` or replaced by the right-invariant inner product form of §2582–2589 for `gl(K)`. See M6.

## Overall Verdict

Minor-to-moderate revisions. The mathematical content of the theory section (lines 175–2039) is mostly correct under the disclosures the manuscript already makes. The remaining issues are concentrated in framing — (M1) the standard-FEP citation umbrella applied to a novel multi-agent functional, (M2) the "theorem" language applied to the cross-scale-shadow definition, (M3) the misleading boxed Eq. eq:wqwk_square that holds only for an architecture standard transformers do not use, (M4) the missing learnable-`κ` disclosure at the canonical reduction point, (M5) the mass-analogy local reading that drifts back into mass language without re-stating the postulate, and (M6) the Killing-form kinetic-metric needing a compact-group qualifier where it appears in the body. None of these requires a rewrite; each is a localized clarification or one-sentence addition. The Lemma at thm:vanishing_holonomy is correct, the Lagrangian derivation of softmax is correct, the dual-transport law for precision is GL-correct, and the two-exponential transport is BCH-honest. The honest disclosures already present in §1034, §1847, §2027, §2099, and §2582 demonstrate that the author knows where the framework departs from standards; the requested edits propagate those disclosures to the locations where a reader first encounters the relevant claim.
