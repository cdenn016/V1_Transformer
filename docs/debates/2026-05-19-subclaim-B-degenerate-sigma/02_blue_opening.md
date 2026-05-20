# Blue Opening — subclaim-B-degenerate-sigma

## Steelman (opposing position)

A reader meeting the paragraph header at `Attention/GL(K)_attention.tex:1026` reads "Deterministic Beliefs via Scaled Limit" and a section labelled `\subsection{Connection to Standard Transformers}` whose preamble at `:1024` literally says "takes three successive limits"; combined with the section §5.7 summary at `:1962` ("Under the successive limits...") and Eq.~`\eqref{eq:gauge_qk}`-style phrasing throughout, the natural reading is that an analytic `σ → 0` is being taken. Since the manuscript itself admits at `:1030`-`:1036` that this analytic limit produces `KL = +∞` for distinct Diracs (violating absolute continuity, as canonised in `external_canon_math.md` §1 KL properties: "Undefined when q(x) > 0 and p(x) = 0 (or returns +∞)"), the section header advertises an operation that the manuscript does not — and mathematically cannot — perform. Naming a reparameterization a "limit" misrepresents the operation in a way that could mislead a reader into believing the construction passes through an ill-defined step.

That is the strongest form of red's case. It is a real semantic infelicity. It is not, however, a mathematical defect.

## Position

The operational content of §5.2.2's reduction is the **reparameterization** `M := σ⁻²Ω⁻ᵀ ∈ GL(d_k)`, exact and well-defined for every finite `σ > 0`. The phrase "the full limit need not be taken" at `:1254` is a correct characterization of what is done. The label "Deterministic Beliefs via Scaled Limit" is stylistic packaging — "scaled limit" refers consistently to the **rescaled-KL limit** `σ²·D_KL → ½‖Ω⁻¹μ_i − μ_j‖²` at `:1114`, which is finite and well-defined by elementary scaling, and "deterministic" is a colloquial shorthand for "variance absorbed, no longer a free belief parameter." The reduction never executes an analytic `σ → 0` on the unrescaled KL; the only object reaching a limit is the σ²-rescaled KL, whose limit is finite by construction.

## Evidence

- **Manuscript explicitly disclaims the analytic limit.** At `Attention/GL(K)_attention.tex:1254`: *"Next, rather than completely taking σ → 0 we recognize that σ⁻² and Ω⁻ᵀ always appear together in the combination σ⁻²Ω⁻ᵀ. The learned matrices W_Q, W_K can then be considered to parametrize this combined quantity directly, rendering σ an implicit (finite) scale factor absorbed into the learned weights. **Therefore, the full limit need not be taken.**"* The operative phrase is "rather than completely taking σ → 0" — the manuscript names and rejects the analytic limit, and replaces it with a reparameterization.

- **The σ²·D_KL limit IS well-defined.** At `:1114`: *"The rescaled KL divergence `D̃_KL(q_i ‖ Ω_{ij}q_j) := σ²·D_KL(q_i ‖ Ω_{ij}q_j) → ½‖Ω_{ij}⁻¹μ_i − μ_j‖² as σ → 0` remains finite."* Per the canonical Gaussian KL form (`external_canon_math.md` §1, isotropic case `Σ = σ²I`): `D_KL = ½[S(Ω) + σ⁻²‖Ω⁻¹μ_i − μ_j‖²]`. Multiplying by `σ²` gives `σ²·S(Ω) + ½‖Ω⁻¹μ_i − μ_j‖²`, whose `σ → 0` limit is `½‖Ω⁻¹μ_i − μ_j‖²`. This is the limit the section header references. It is a textbook well-defined limit of a rescaled object — no absolute-continuity violation, no `+∞`, no analytic pathology.

- **The manuscript acknowledges the unrescaled-limit pathology at `:1030`–`:1036`** and uses that acknowledgement to motivate the reparameterization. The chain is: "naively `Σ → 0` gives `+∞`" (the manuscript states the obstruction explicitly, citing absolute continuity as in `external_canon_math.md` §1) → "we remedy by joint scaling where σ² remains finite but is absorbed into learned parameters" at `:1038` → "Σ_i = σ²I with σ² > 0" at `:1040` (finite throughout the derivation) → joint absorption at `:1254`. At no point in the chain is the unrescaled KL evaluated at `σ = 0`.

- **σ stays positive in every algebraic step.** Eq.~`\eqref{eq:isotropic_general_omega}` at `:1082` has `1/(2σ²)` with `σ² > 0`. The constant-gauge reduction at `:1169` retains `1/(2σ²)`. The softmax cancellations at `:1180`–`:1196` retain `1/σ²`. The W_QW_K^T identification at `:1240` reads `W_QW_K^T = σ⁻²Ω⁻ᵀ ∈ GL(d_k)` — explicitly invertible, requiring `σ < ∞`. The reduction is a bijective reparameterization on `(σ, Ω) ↦ σ⁻²Ω⁻ᵀ ∈ GL(d_k)` for any finite `σ > 0`; nothing is sent to zero.

- **The §5.7 summary is consistent with the reparameterization reading.** At `:1962`: *"(i) isotropic beliefs with σ⁻² absorbed into learned projections."* Limit (i) is named as **absorption**, not as `σ → 0`. The paragraph header's word "scaled limit" and the summary's word "absorbed" point at the same operation; the summary's phrasing is unambiguous about which one.

- **First debate's verdict precedent.** The orchestrator's first-pass verdict on this sub-claim found it "defensible — line 1252 makes this explicit." That verdict is not load-bearing here — the judge should weigh evidence afresh — but the line cited (now `:1254` after the textual shift) is the same line that is the strongest piece of textual evidence for the blue position. The verdict was tracking the right citation.

## Falsification conditions

This position is wrong if **any** of the following hold:

1. **The manuscript algebraically evaluates D_KL at σ = 0 anywhere in §5.2.2.** If a step takes the unrescaled KL and substitutes `σ = 0`, the operation is not a reparameterization but an attempt at the ill-defined limit. I have read `:1024`–`:1290` and find no such step: every `σ` appearing in an algebraic expression is either positive-and-finite (`:1040`, `:1082`, `:1169`, `:1175`, `:1240`) or the manuscript names the rescaled object explicitly (`:1114`). If red produces a manuscript line where the unrescaled KL is evaluated at `σ = 0` and the result is used downstream, blue's position falls.

2. **The "scaled limit" naming is read by the cited algebra as the unrescaled limit, not the rescaled one.** If the section header at `:1026` together with the derivation can be shown to require `σ → 0` of the *unrescaled* KL (rather than of `σ²·D_KL`) to license any downstream identification, then the title misrepresents the operation in a way that materially changes the mathematics. Blue's claim is that nothing downstream of `:1254` requires anything beyond the joint reparameterization at finite σ.

3. **"Deterministic beliefs" is required to be Dirac in the downstream construction.** If any equation after `:1254` requires beliefs to be Dirac measures (rather than `N(μ_i, σ²I)` with σ implicit-in-W), then the section header's "deterministic" carries mathematical content that is unsupported. I find no such requirement: the final attention formula at `:1288` reads `softmax(QK^T/√d_k)`, which is recovered with finite-σ Gaussian beliefs whose σ is absorbed into the learned `W_Q, W_K`. If red shows otherwise, blue concedes.

4. **The reparameterization is non-injective in a way that breaks the reduction.** The map `(σ, Ω) ↦ σ⁻²Ω⁻ᵀ` is many-to-one in the forward direction (any `(λσ, λ²Ω⁻ᵀ)`-type rescaling produces the same `M` after redefinition). Many-to-one in the forward direction is *not* a defect for the reduction (gauge → transformer); it would be a defect for the converse identification (transformer → gauge), but that is sub-claim D, not B. If red can show that the non-injectivity affects sub-claim B's algebra, blue concedes.

5. **A standard reference treats this construction as a `σ → 0` analytic limit and demonstrates the manuscript's framing diverges.** The closest standard analog is the temperature-zero / variance-zero limit in EM and in variational lower bounds. The standard treatment in [BleiKuckelbirgJordan2017] and the canonical KL form in `external_canon_math.md` §1 evaluates `D_KL` for `σ > 0` and observes the absolute-continuity pathology at `σ = 0`. The manuscript's framing — work at finite σ, absorb σ into learned weights — matches standard practice. If red identifies a canonical reference that calls this same operation an analytic limit (rather than a reparameterization or absorption), blue's stylistic-defence falls back to a weaker form.

## What blue concedes upfront (pre-emptive)

The section header at `:1026` is stylistically imperfect. "Deterministic Beliefs via Scaled Limit" puts the word "Limit" in the title while the operative line at `:1254` says "the full limit need not be taken." A reader who stops at the header gets a different mental picture than a reader who reaches `:1254`. The cleanest names would be "Implicit-Variance Reparameterization" or "Scale-Absorbed Form." Blue grants this is a naming defect.

The mathematical content, however, is exact: the reduction is the bijective reparameterization `(σ, Ω, isotropic q) ↦ (σ⁻²Ω⁻ᵀ, isotropic q with σ implicit)`, executed at finite `σ > 0`, with the rescaled-KL limit at `:1114` as the only object that is actually sent to a limit and is finite by construction. The manuscript is right; the title is loose.
