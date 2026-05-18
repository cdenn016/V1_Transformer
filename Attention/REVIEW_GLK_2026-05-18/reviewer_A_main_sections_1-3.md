# Reviewer A — GL(K)_attention.tex §1–§3 (lines 1–989)

## Summary

The slice under review (Abstract + §1 Introduction + §2 Methods + §3 Gauge-Covariant Variational Free Energy through "Summary") sets up the gauge-theoretic VFE framework and derives multi-agent attention as the variational solution of a mixture-of-sources generative model. The core mathematical content is largely correct: the GL(K) invariance proof (Thm 1) is a clean restatement of a textbook fact, the mixture-of-sources Lagrangian derivation of softmax (lines 715–762) is rigorous and matches standard maximum-entropy variational arguments, the envelope/autograd-gap formula `−τ⁻¹ Cov_β(E, ∂E/∂x)` (Eq:autograd_envelope_gap) is correct, and the state-dependent-precision derivation (Eq:state_dependent_alpha) is correct. The three most load-bearing concerns are: (i) Theorem 2 ("Vanishing Holonomy") is a one-line algebraic identity that follows trivially from the cocycle product `g_i g_j⁻¹ · g_j g_k⁻¹ · g_k g_i⁻¹ = I`; calling this a "theorem" overstates it and the surrounding language ("theorem of the architecture, not an approximation," line 656) is misleading — what is actually proved is that any vertex-only parameterization is automatically pure gauge, which is closer to a definition than a result; (ii) the manuscript imports the bold identification `W_Q W_Kᵀ = σ⁻²Ω⁻ᵀ` into the abstract (line 49), Table~\ref{tab:gauge_vs_transformer} (line 480), and into the comparison table summary (line 566) before it is derived in §4 — within the scope of this review, that identification is asserted not derived, and as the canon notes, the identification is not unique because `W_Q W_Kᵀ` is generically rank-deficient (many Ω satisfy any given low-rank product); (iii) the very large number of banned `\;`, `\,`, `\!` spacing macros throughout the manuscript (project style policy) — these are pervasive and require a global cleanup pass, not piecemeal fixes. There is also a minor but pervasive overstatement issue: §1 promises recovery of "the full suite of transformer architectural choices" from "a single variational principle," while §3 actually delivers only the attention rule and positional priors. Layer norm, multi-head, and the W_Q/W_K identification are deferred to §4 and depend on additional limits. The abstract and §1 should temper their reach to match what §3 actually proves.

## Standards against which the manuscript was reviewed

- [Friston2010] (and [ParrPezzuloFriston2022]) for the canonical single-agent variational free energy form.
- [AmariNagaoka2000] for KL invariance under sufficient-statistic transformations / GL(K) push-forward of Gaussians.
- [BleiKuckelbirgJordan2017], [KingmaWelling2014 Appendix B] for the closed-form Gaussian KL.
- [Nakahara2003], [KobayashiNomizu] for principal-bundle connection, holonomy, cocycle condition.
- [Vaswani2017] for standard scaled dot-product attention `softmax(QKᵀ/√d_k)V`.
- [Press2022Train], [Su2024RoPE], [Beltagy2020Longformer] for positional-bias mechanisms (ALiBi, RoPE, sliding window).
- Project style: `CLAUDE.md ## Style` and `## Scientific Writing Rules`; `style_constraints.md`.

## Major findings

### M-A-1. Theorem 2 (Vanishing Holonomy) is a definitional identity, not a theorem

**Claim (lines 640–650):** "Theorem (Vanishing Holonomy). For gauge transport of the form Ω_ij = g_i g_j⁻¹ with vertex-local group elements g_i ∈ G, the holonomy around any closed loop vanishes: H_ijk = Ω_ij Ω_jk Ω_ki = g_i g_j⁻¹ · g_j g_k⁻¹ · g_k g_i⁻¹ = I."

**Claim kind:** (S)-presented but in fact (I)/trivial-identity.

**Standard treatment:** A connection on a principal bundle is *flat* iff its curvature 2-form vanishes; on a discrete graph, flatness is equivalent to the cocycle condition `Ω_ij Ω_jk = Ω_ik` [Nakahara2003 §10.3, §11.1; KobayashiNomizu Vol. I §II.9]. A parameterization in which every edge is determined by vertex-local data — exactly `Ω_ij = g_i g_j⁻¹` — is by definition a *pure-gauge* connection (the gauge transform of the trivial connection), and pure-gauge connections always have trivial holonomy on contractible loops. This is the definition of "pure gauge," not a derivable theorem.

**Problem:** Calling this a "theorem of the architecture, not an approximation" (line 656) misleads the reader into thinking something has been *proved* about an a-priori-non-trivial geometric object. What has actually been done is: the manuscript *chooses* a vertex-only parameterization for Ω, and that choice automatically forces pure-gauge structure. The geometric content is "we have chosen a flat connection." Listing this as "Theorem 2" alongside Thm 1 (which is a genuine, if standard, invariance statement) inflates the math. The next paragraph (lines 657–665) correctly acknowledges that non-trivial holonomy requires an *edge-local* δ_ij — i.e. the user knows this — but the labeling on the preceding theorem environment doesn't match.

**Required revision:** Downgrade `\begin{theorem}` to `\begin{lemma}` or `\begin{proposition}`, retitle as "Cocycle identity for vertex-parameterized transport," and replace "theorem of the architecture, not an approximation" with the more honest framing: "the present parameterization commits to a flat connection by construction; non-trivial holonomy requires additional edge-local degrees of freedom (Eq:edge_relaxed_omega_glk)."

### M-A-2. Identification `W_Q W_Kᵀ = σ⁻²Ω⁻ᵀ` is asserted in scope without derivation, and is not unique

**Claim:** Abstract (line 49): "...the GL(K) invariance of KL divergence implies that W_Q, W_K are themselves gauge transformations with W_Q W_Kᵀ = σ⁻²Ω⁻ᵀ"; Table~\ref{tab:gauge_vs_transformer} (line 480): "Gauge transformation: ... W_Q W_Kᵀ = σ⁻²Ω⁻ᵀ (learned, constant)"; line 566: "...the constant gauge identification satisfies W_Q W_Kᵀ = σ⁻²Ω⁻ᵀ."

**Claim kind:** Presented as (S)/(R), actually (I)/identification.

**Standard treatment:** In standard transformers, `W_Q ∈ ℝ^{d_model × d_k}` and `W_K ∈ ℝ^{d_model × d_k}` are rectangular learned projection matrices [Vaswani2017 §3.2]. The product `W_Q W_Kᵀ ∈ ℝ^{d_model × d_model}` has rank at most `d_k`, generically rank-deficient (rank `d_k` ≤ `d_model`). For a constant gauge `Ω ∈ GL(K)`, `Ω⁻ᵀ` is full-rank `K × K` (when K = d_model). Equating a generically rank-deficient quantity to a full-rank quantity is well-defined only as an *underdetermined identification*: the identification picks out one of infinitely many Ω consistent with any given `W_Q W_Kᵀ` (or restricts the gauge subgroup).

**Problem:** The identification appears three times in scope (lines 49, 480, 566) and once in the summary table (line 1650 — out of scope but referenced from within scope). All three in-scope appearances assert the identification as a *consequence* of GL(K) invariance, but the only thing GL(K) invariance shows is that the *form* of the logit is gauge-covariant. The specific equality `W_Q W_Kᵀ = σ⁻²Ω⁻ᵀ` is a *parameterization choice* identifying the learned bilinear form `W_Q W_Kᵀ` with the (inverse-transpose of a) constant gauge — it does not follow from anything proved before §4. In §1–§3 the reader is asked to accept this as a result; the derivation (such as it is) sits in §4 (the line 1238 expression `W_Q W_Kᵀ = (1/σ²) Ω⁻ᵀ ∈ GL(d_k)`) which is outside this scope.

**Required revision:** In the abstract and in Table~\ref{tab:gauge_vs_transformer} (line 480), label the identification as either "we will show in §4 that under the isotropic, constant-gauge limit one may parameterize W_Q W_Kᵀ as σ⁻²Ω⁻ᵀ" (forward reference, parameterization framing), or remove the abstract mention until after the derivation. Additionally, when the derivation is reached in §4, explicitly acknowledge that the identification is not unique: many Ω in GL(K) (and in particular in the lower-rank `GL(d_k)` if the standard transformer uses `d_k < d_model`) are consistent with any given low-rank `W_Q W_Kᵀ`. (This last issue is for the §4 reviewer to confirm.)

### M-A-3. Abstract and §1 promise more than §3 delivers

**Claim:** Abstract (line 48): "standard transformer architectural choices are recovered as special cases of the variational geometry: the temperature scaling 1/√d_k from dimensional concentration of the KL divergence, layer normalization as the geometric condition for frame-independent inference, multi-head attention as a block-diagonal restriction of the gauge algebra, and causal masking and positional biases from non-uniform attention priors π_j." §1 line 60: "a unified geometric framework from which the full suite of transformer architectural choices emerges as consequences of a single variational principle."

**Claim kind:** (S)-style global claim, actually a forward-reference to §4 and Supplementary.

**Standard treatment:** N/A — this is a manuscript-internal scope claim.

**Problem:** Within §1–§3 (lines 1–989), only the *attention rule itself* (softmax over −KL/τ) and the *positional-prior mechanisms* (causal mask, sliding window, ALiBi, T5-style relative bias) are actually derived. Temperature scaling `1/√d_k`, layer norm, multi-head, and the W_Q/W_K identification are deferred to §4 (out of scope here). A reader of §3 reaches "Summary" (line 984) believing the framework has delivered all four, but only two have been shown. The summary (line 986) is more honest: it only claims (i) FEP-style single-agent F, (ii) mixture-of-sources alignment + softmax — those are in scope. The abstract overpromises relative to §3 alone.

**Required revision:** Either soften the abstract (e.g. "we derive attention and positional-prior mechanisms from a variational principle; under further limits established in §4, the dimensional scaling, layer normalization, and multi-head structure are recovered as well") or restructure so the four limits are surveyed at the §3 summary and not promised at abstract level until they have been delivered.

### M-A-4. The "Theorem 1 / GL(K) Gauge Invariance" framing is correct but standard; the surrounding prose flirts with overclaiming

**Claim:** Line 518: "...this is a standard result in information geometry (see, e.g., \citet{amari2016information}), but its implications for transformer attention have not been systematically developed."

**Claim kind:** (S)-acknowledged, with a contribution claim attached.

**Standard treatment:** Invariance of KL (and all f-divergences) under any common invertible change of variables is standard [AmariNagaoka2000 Ch. 2; equivalently the data-processing inequality with equality for invertible maps]. The proof in lines 531–556 is correct (trace cyclicity for the trace term, cancellation of `(det Ω)²` in the log-det term — symbolically verified).

**Problem:** The manuscript correctly cites this as standard. No major issue with the theorem itself. However, the proof is unnecessarily long for a fact the manuscript itself labels as textbook. The minor concern is the framing in the contribution claim ("but its implications for transformer attention have not been systematically developed") — this is a softer version of an "is" claim that the canon flags. The user's framework is *one* gauge-theoretic view of attention; the kernel view [Tsai2019], Hopfield view [Ramsauer2021], geometric DL view [Bronstein2021] are alternatives the manuscript already acknowledges in §1 line 64.

**Required revision:** Either compress the proof (5 lines suffice for a textbook fact) or include it but explicitly mark it as "for completeness." Soften "have not been systematically developed" to "have not, to our knowledge, been developed specifically for transformer attention." This is minor — flagged here because it's the same overclaiming pattern as M-A-3.

### M-A-5. Banned spacing macros pervasive throughout the scope

**Claim:** Project policy bans `\;`, `\,`, `\!` in LaTeX (CLAUDE.md "Scientific Writing Rules"; `style_constraints.md`).

**Standard treatment:** N/A — project style policy.

**Problem:** The Grep over lines 1–989 finds ~40+ uses of `\,`, `\;`, `\!` inside display math and figure captions. Specific lines within scope: 297, 351, 352, 447, 448, 450, 451, 494, 495, 525, 561, 591, 593, 602, 604, 624, 627, 642, 644, 645, 646, 649, 689, 693, 715, 721, 760, 766, 815, 832, 847, 849, 861, 866, 869, 905, 907, 915, 935. (Non-exhaustive — every equation block contains at least one.)

**Required revision:** Global cleanup pass: strip `\;`, `\,`, `\!` from all in-body equations. Replace with normal LaTeX spacing (which is appropriate for `amsmath` displays). This is one find-replace operation but should not be done as part of editing other content (per project policy, surgical changes only).

## Minor findings

### m-A-1. (line 365)

The "geometry" of the 0D base manifold is described as the "Methods" lead. The first paragraph (line 366) is dense and could open with the single-sentence summary "In what follows, agents are tokens occupying a single 0-dimensional base manifold; gauge connections, curvature, and parallel transport reduce to algebraic identities on vertex-local frame elements."

### m-A-2. (line 367)

"This separation between the geometric base (which determines gauge connections and curvature) and the combinatorial index structure (which determines masking and positional biases) is important." The word "important" is acceptable, not banned. No edit needed; flagged only because the parenthetical is doing a lot of work — consider breaking into two sentences for readability.

### m-A-3. (line 408 — notation table)

"`$\alpha$ ($\alpha_i$) | $\mathbb{R}^+$ | Prior precision; state-dependent variant (Sec.~\ref{sec:state_dependent_precision})`" — the notation reads as if `α` is the scalar and `α_i` the per-agent version; in §3.7 (line 944) the manuscript uses `α_i = α = 1` as the constant-precision specialisation. Recommend wording the table as: "`α | ℝ⁺ | Constant prior precision (default α=1); α_i is the per-agent state-dependent variant (Sec.~\ref{sec:state_dependent_precision})`."

### m-A-4. (line 409)

"`$\kappa$ ($\tau$) | $\mathbb{R}^+$ | Attention temperature; $\kappa_a \propto \sigma_a^2$ in isotropic limit (Sec.~\ref{sec:multihead})`" — the notation overloads κ and τ as if they are interchangeable. From CLAUDE.md and the canonical free energy, τ = κ√K is *not* the same as κ alone. Recommend: "`κ | ℝ⁺ | Learnable per-head attention scaling; τ = κ√K is the effective temperature in the canonical free energy. Per-head limit κ_a ∝ σ_a² discussed in Sec.~\ref{sec:multihead}.`"

### m-A-5. (line 424 "for non-geometers")

"For interdisciplinary readers, it may be helpful to visualize..." — fine prose. The paragraph reads well. No change needed.

### m-A-6. (line 427) — apostrophe orientation

"agent's 'internal orientation' toward the world" uses curly quotes in the rendered text — confirm in the .tex they are ASCII or LaTeX-compatible (` ` and `' '`), since some Windows editors silently insert U+2018/U+2019. Verify after typesetting.

### m-A-7. (line 510) — "in this way is pure gauge and has identically vanishing curvature"

This sentence asserts a fact ("a connection derived from a single-valued gauge function `φ_i(c)` in this way is pure gauge and has identically vanishing curvature `F_μν = 0`"). The assertion is correct (`A = U⁻¹ dU` is the gauge transform of the zero connection; its field strength vanishes by direct computation). Recommend adding a one-line justification or pointing to Supplementary Appendix A so the reader doesn't have to take it on faith.

### m-A-8. (line 564) — "Disconnected transformations are reserved for future study."

Fine, but flag as a real limitation in §1 / Limitations rather than as an aside. The user's parameterization is `exp(φ)`, which only reaches the identity component `GL⁺(K)`. The full GL(K) is not reached. The canon notes the exponential map is not surjective on GL⁺(K, ℝ) either, so even the identity component is not fully reachable by `exp(φ)` for any single `φ ∈ gl(K)`. The product `exp(φ_i) exp(−φ_j)` reaches a strictly larger subset but still not all of GL⁺(K) — verify the user is aware. (If aware and not material to claims, fine as a footnote.) [Hall, *Lie Groups, Lie Algebras, and Representations* — citation pending verification of specific chapter.]

### m-A-9. (line 569)

"In what follows we take the gauge group dimension to be equal to the embedding dimension, a simplifying choice that does not restrict the generality of the derivations below." Calling this "no restriction" is a strong claim — recommend "we take K_gauge = K_embed for notational simplicity; nothing in the derivations depends on this choice except as noted."

### m-A-10. (lines 859–862)

The envelope-theorem statement uses display style `$$ ... $$` (lines 859–862) inside a paragraph rather than `\begin{equation}` / `\begin{equation*}`. This is inconsistent with the rest of the manuscript (which uses numbered equations or `\begin{equation*}`). Recommend converting to `\begin{equation*}` for consistency.

### m-A-11. (line 974)

"much, much more" — Stylistically loose for a JMLR submission. Recommend "and additional geometric structure beyond the scope of this paper."

### m-A-12. (line 982)

"Whether the vacuum manifold supports approximate zero modes of the Hessian ... is a question we leave to future work." — Fine, but the phrasing is a bit casual. Consider "Whether the vacuum manifold supports approximate zero modes of the Hessian along the gauge orbit is left to future work."

## Equation verification log

| Eq label | Line | Verified | Notes |
|---|---|---|---|
| `eq:representations` (4) | 434 | ✓ | Pair of representations of GL(K), trivially correct. |
| `eq:gauge_action_gaussians` (5) | 443 | ✓ | The action `Σ → ρ(Ω) Σ ρ(Ω)ᵀ` is the standard sandwich for covariances; this is the (2,0) interpretation in canon §2. |
| Thm 1 proof (eq:glk_invariance / 534, 540, 545, 551) | 533–555 | ✓ | Trace cyclicity for the trace term; `(det Ω)²` cancels in the log-det. Symbolically verified. |
| `eq:dual_latents` (Eq for latent vars) | 580 | ✓ | Definitions only. |
| `eq:gaussian_states` | 589 | ✓ | Standard MVG ansatz. |
| `eq:base_priors` | 601 | ✓ | Standard. |
| `eq:gauge_frame_rotation` (Ω_ij = e^φ_i e^{-φ_j}) | 615 | ✓ | Two-exponential parameterization; does *not* collapse to `exp(φ_i − φ_j)` (BCH); user does not claim it does. Numerically verified non-equality. |
| `eq:gauge_action_on_vectors` | 622 | ✓ | Definition. |
| Thm 2 (eq:holonomy) | 644 | Trivial (downgrade) | One-line cocycle algebra; see M-A-1. |
| `eq:edge_relaxed_omega_glk` | 659 | ✓ structurally | Edge-local connection insertion preserves gauge covariance law; full proof deferred to companion paper [Dennis2025it]. |
| `eq:single_agent_fep` (Eq 5 of single-agent F) | 673 | ✓ | Matches [Friston2010] Form-3 (accuracy + complexity) up to sign. F = D_KL(q_i ‖ p_i) − E_q[log p(o_i|k_i)]. |
| `eq:mixture_joint` | 689 | ✓ | Standard mixture joint. |
| `eq:mixture_posterior` | 705 | ✓ | Mean-field factorization. |
| `eq:mixture_free_energy` | 722 | ✓ | KL[Q‖P] correctly expanded to Σ_j β_j [KL(q_i‖P(·|z=j)) + log β_j − log π_j]. Symbolically verified. |
| `eq:mixture_energy_entropy` | 731 | ✓ | Energy − entropy form, equivalent rewrite. |
| Lagrangian (line 741) | 741 | ✓ | Sympy-verified: ∂L/∂β_k = 0 ⇒ β_k = π_k exp(−E_k)/Z. |
| Stationarity equation (line 747) | 747 | ✓ | Direct differentiation; symbolically verified. |
| `eq:mixture_softmax_general` | 754 | ✓ | β_ik = π_k exp(−E_ik)/Σ_m π_m exp(−E_im). |
| `eq:mixture_softmax` | 761 | ✓ | Uniform-prior special case. |
| `eq:F_align_canonical_tau` | 767 | ✓ | Canonical row-Lagrangian with τ coupled to entropy/prior term. F* = −τ log Z verified symbolically. Matches the canonical F in CLAUDE.md. |
| Causal/window/ALiBi/T5 (eq:causal_attention, eq:alibi_attention, eq:relative_bias_attention) | 788, 816, 833 | ✓ | Direct substitutions of the corresponding π_k into the general softmax. Algebraically clean. |
| `eq:free_energy_final` | 843 | ✓ | The reduced F = Σ KL(q_i‖p_i) − τ Σ log Z_i − E[log p(o)] follows by envelope substitution; the entropy/log-Z bookkeeping is correctly handled. |
| Envelope statement (line 859) | 859 | ✓ | Standard envelope theorem; the cross-term vanishes because ∂F/∂β = 0 at β*. |
| `eq:autograd_envelope_gap` | 870 | ✓ | The covariance form −τ⁻¹ Cov_β(E, ∂E/∂x) follows from the softmax sensitivity. Symbolically verified. |
| `eq:free_energy_adaptive` | 905 | ✓ | Augmented F with α_i and R(α_i). |
| `eq:precision_regularizer` | 916 | ✓ | b_0 α − c_0 log α is the standard log-barrier (Gamma-like). |
| Optimality condition (line 925) | 925 | ✓ | ∂F/∂α = KL + b_0 − c_0/α = 0. |
| `eq:state_dependent_alpha` | 935 | ✓ | α* = c_0/(b_0 + KL). Symbolically verified. |
| `eq:alpha_chain_rule` | 951 | ✓ | Direct differentiation of α* = c_0/(b_0 + KL). |
| `eq:alpha_product_rule` | 956 | ✓ | The collapse uses 1 − (α*/c_0) KL = b_0/(b_0+KL) = α* b_0/c_0; gives effective prefactor (α*)² b_0/c_0. Verified algebraically. |
| `eq:belief_dynamics` | 968 | ✓ | Gradient flow ansatz; trivially correct. |

**No equations in §1–§3 are wrong as far as I can verify.** The major findings concern *framing* (what is and isn't a theorem; what is asserted vs derived; what is in scope vs forward-referenced), not arithmetic errors.

## Style scan

**Banned spacing macros (`\;`, `\,`, `\!`)** within lines 1–989 — non-exhaustive list of lines with at least one hit:

297, 351, 352, 447, 448, 450, 451, 494, 495, 525, 561, 591, 593, 602, 604, 624, 627, 642, 644, 645, 646, 649, 689, 693, 715, 721, 760, 766, 815, 832, 847, 849, 861, 866, 869, 905, 907, 915, 935.

This is essentially every numbered equation. Global cleanup needed.

**Banned Claude-ism phrases** ("key insight", "crucially", "notably", "importantly", "it's worth noting", "interestingly", "fundamentally", "in particular", "leverages", "underscores"): grep finds **none** within lines 1–989. Style is clean on the phrasal axis.

**Horizontal rules** (`---` in body): grep finds none in body text within scope. The em-dashes that appear in TikZ comments (lines 78, 93, etc.) are inside `%` comments — not rendered. OK.

**Self-referential drafting** ("earlier drafts", "the corrected reading", etc.): grep finds none. Good.

## Citations checked

| Citation | Where in scope | Claim it backs | Verification |
|---|---|---|---|
| `friston2010free` | line 60, 670 | "Friston's Free Energy Principle"; single-agent F form | [✓] [Friston2010] — the cited form `F = KL(q ‖ p) − E_q[log p(o|s)]` matches the standard Form-3 decomposition. |
| `parr2022active` | 60, 670 | Active inference textbook | [✓] [ParrPezzuloFriston2022] is the textbook; cited in canon. |
| `friston2017graphical` | 60 | Graphical / hierarchical formulation | [✓] [Friston2017Graphical] — present in bib, supports hierarchical-FEP cite. |
| `ramstead2020variational` | 60 | Variational ecology / multi-agent | [✓] [Ramstead2020] — present. |
| `bahdanau2014neural` | 60 | Attention background | [?] Standard cite; not in canon `external_canon_*` but well-known. |
| `amari1998natural` | 60 | Natural gradient | [✓] [Amari1998] — canonical natural-gradient paper. |
| `bronstein2021geometric` | 60, 62 | Geometric DL | [✓] [Bronstein2021] — canonical. |
| `vaswani2017attention` | 60 | Standard attention | [✓] [Vaswani2017] — canonical. |
| `tsai2019transformer` | 64 | Kernel interpretation of attention | [✓] [Tsai2019] — canonical. |
| `katharopoulos2020transformers` | 64 | Linear attention | [✓] [Katharopoulos2020] — canonical. |
| `ramsauer2021hopfield` | 64 | Hopfield interpretation | [✓] [Ramsauer2021] — canonical. |
| `bogacz2017tutorial`, `millidge2021predictive` | 64 | Predictive coding | [✓] [Bogacz2017, Millidge2021] — canonical for the PC/FEP connection. |
| `amari2016information` | 518 | GL(K) invariance / info geometry textbook | [✓] [Amari2016] — present; supports the textbook claim. |
| `nakahara2003geometry` | 364 | General differential geometry | [✓] [Nakahara2003] — canonical. |
| `frankel2011geometry` | 364 | Geometry of physics | [✓] [Frankel2011] — canonical. |
| `radford2019language` | 791 | GPT causal masking | [✓] standard cite. |
| `press2022train` | 819 | ALiBi | [✓] standard cite (canonical for ALiBi). |
| `raffel2020exploring` | 836 | T5 relative bias | [✓] standard cite (canonical for T5). |
| `beltagy2020longformer` | 801 | Longformer sliding window | [✓] standard cite. |
| `jiang2023mistral` | 801 | Mistral sliding window | [✓] standard cite. |
| `weinberg1995quantum` | 980 | Elitzur's theorem / gauge symmetry | [✓] standard physics textbook. |
| `WilsonConfinement1974` | 665 | Wilson loop / lattice gauge | [✓] foundational reference. |
| `KogutSusskind1975` | 665 | Hamiltonian lattice gauge | [✓] foundational reference. |
| `Creutz1983` | 665 | Lattice gauge textbook | [✓] standard reference. |
| `Dennis2025it` | 665 | Companion paper | [?] cannot verify — author's own forthcoming work. Verify the cited paper exists and is publicly available before submission. |
| `clark2019does`, `foerster2016learning`, `wooldridge2009introduction`, `fuchs2020se3`, `thomas2018tensor`, `kondor2018generalization`, `finzi2020generalizing`, `bonnabel2013stochastic`, `absil2008optimization`, `tishby2015deep`, `shwartz2017opening`, `hinton2022forward`, `rao1999predictive`, `kullback1951information`, `sternberg1994group`, `fulton1991representation`, `baez1994gauge` | various | Background | [?] All present in references.bib (verified by grep). Not load-bearing for any specific equation in scope; treated as background. |

**Summary:** all citation keys used within scope (1–989) are present in `references.bib`. The four canonical references (Friston, Amari, Nakahara, Vaswani) are correctly attached to the claims they back. No load-bearing citation is missing or misattributed within scope.

## Code cross-references checked

| Manuscript line | Claim about code | Verified |
|---|---|---|
| line 960 | "Our reference implementation evaluates the corrected form (`transformer/core/vfe_gradients.py`)" | [✓] file exists at `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\core\vfe_gradients.py`. Did not verify the file actually implements Eq (eq:alpha_product_rule) — out of review scope; flag for code auditor. |

Other code references within scope: none. The §1–§3 slice is largely derivation-only; code-implementation cross-references concentrate in §4–§5.

## Novel-construction inventory

These constructions appear in §1–§3 and are *not* part of the standard FEP / VI / gauge-theory literature. They should be (and largely are) acknowledged as the manuscript's contribution rather than standard:

1. **Multi-agent variational free energy with gauge-transported pairwise KL terms** `Σ_ij β_ij D_KL(q_i ‖ Ω_ij q_j)` (lines 715–732 derivation, line 843 final form). Standard FEP is single-agent (or hierarchically nested with one ancestral generative model) [Friston2010]; variational-ecology extensions [Ramstead2020] use different couplings. The user's mixture-of-sources construction is a clean variational derivation of this specific form, and the manuscript should preserve this contribution claim (it does, line 986). No flag needed for novelty itself.
2. **Attention prior π_j as the explicit slot for masking / positional bias** (lines 776–836). The unified treatment of causal mask, sliding window, ALiBi, and T5 relative bias as four choices of π_j is genuinely useful as exposition. Standard transformers introduce these as architectural ad-hoc choices; the manuscript reframes them as Bayesian priors. The manuscript correctly notes this (line 836). Good.
3. **State-dependent prior precision α_i = c_0/(b_0 + KL)** (§3.7, eq:state_dependent_alpha). Adaptive precision is a known idea in active inference [ParrPezzuloFriston2022 — adaptive precision via β-Gaussian or Gamma priors over precision]. The specific log-barrier `R(α) = b_0 α − c_0 log α` and its closed-form optimum is the user's specific instantiation. Treat as novel-specific-instantiation, not novel-concept.
4. **The autograd-vs-envelope covariance gap** `−τ⁻¹ Cov_β(E, ∂E/∂x)` (eq:autograd_envelope_gap). This is a clean application of the envelope theorem + softmax sensitivity. Standard tools; the explicit expression as a covariance is the user's framing. Useful, correct.
5. **Vanishing-holonomy "theorem"** (Thm 2): see M-A-1 — not novel, just trivial.

## Open questions

1. **Is the cocycle "Theorem 2" a deliberate framing choice or an oversight?** The user clearly knows that vertex-only Ω_ij = g_i g_j⁻¹ is by definition pure gauge (line 510, line 657 spells out the edge-local relaxation). So why is the trivial identity labeled as a theorem? Two readings: (a) it is meant as a *result about the chosen architecture* (i.e. "given the parameterization we chose, here is what follows") — in which case label it as a Proposition with an explicit "given our parameterization" hypothesis; (b) it is residual scaffolding from an earlier draft. Recommend the author choose explicitly.
2. **What does the abstract mean by "GL(K) invariance of KL implies W_Q W_Kᵀ = σ⁻²Ω⁻ᵀ"?** GL(K) invariance is necessary but not sufficient for this identification. Several Ω satisfy it. I read this as a forward reference to §4 where a specific limit pins one choice down. If correct, the abstract should say "we identify" rather than "implies."
3. **Is the κ in `τ = κ√K` the same κ as `κ_a` in §3.7 multi-head reference (line 409)?** The notation table says "κ_a ∝ σ_a² in isotropic limit (Sec. multihead)." This suggests κ is a per-head scalar, not a single global learnable scalar. Recommend a one-line statement at first use clarifying whether κ is global, per-head, or both (the canonical F in CLAUDE.md has a single κ).

## Overall verdict

**Major revisions required** — but the substantive math in scope is correct. The three required revisions are (1) downgrade Thm 2 to Proposition/Lemma and rewrite the surrounding "theorem of the architecture, not an approximation" rhetoric, (2) either soften the abstract / Table 2 / line 566 about `W_Q W_Kᵀ = σ⁻²Ω⁻ᵀ` to "we will identify ... in §4" or relocate the identification claim entirely out of §1–§3, and (3) global cleanup of banned `\;`, `\,`, `\!` spacing macros. Once these are addressed, the derivations as such are sound, the citations check out, and the mixture-of-sources attention argument is a genuine contribution. Recommend a single revision pass plus a peer re-read of §4 (out of my scope), which is where the load-bearing claims about the standard transformer reduction actually live.
