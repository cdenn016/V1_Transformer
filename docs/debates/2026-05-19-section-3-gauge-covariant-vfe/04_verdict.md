# Verdict — section-3-gauge-covariant-vfe

## Outcome

RED_WINS (narrow, editorial scope)

## Decisive evidence

Three lines, taken together, decide the compound claim:

1. `Attention/GL(K)_attention.tex:914` — the manuscript introduces the precision regularizer with the phrase "A natural choice is a log-barrier form such as" and proceeds at line 917 to write `R(α_i) = b_0 α_i - c_0 log α_i` without any identification of this expression with a hyperprior.

2. `01_evidence.md:91-93` recording the canonical form from `[Bishop 2006 §2.3.6; Murphy 2012 §4.6.1]`: `Gamma(α; c, b) = b^c/Γ(c) · α^{c-1} exp(-b α)` with negative log density `-log p(α) = b α - (c-1) log α + const`. Matching to the manuscript form yields `R(α_i) = -log Gamma(α_i; c_0 + 1, b_0) + const` — an exact identification with the conjugate prior on Gaussian precision.

3. Blue's own rebuttal at `03_blue_rebuttal.md:5` concedes the verification: "I verified this by grepping `Attention/GL(K)_supplementary.tex` for `Gamma`, `conjugate`, `hyperprior`, `Bishop`, `Murphy`, `b_0`, `c_0`, `log-barrier`, and `natural choice` — no Gamma identification appears anywhere in the supplementary. Red's falsification conditions (1) and (2) both fail in red's favor: the manuscript does not state the identification, and the appendix forward-reference does not deliver it. The editorial gap red identifies is real."

The forward reference at `Attention/GL(K)_attention.tex:902` to the appendix's "hierarchical structure of the generative model" does not resolve to a Gamma-prior derivation, as blue confirmed by direct grep.

## Reasoning

The compound claim asserts that §3 is "robust, theoretically pure, and mathematically correct." The §4–§5 debate series established the editorial standard for this manuscript: a derivation that depends on a canonical Bayesian or convex-optimization construction must label that construction explicitly, with primary-source citation (the canonical-F-vs-surrogate verdict added the gap formula and convention statement at line 874; the softmax-β verdict added the attention-entropy term and the Cuturi/Boyd citation chain). Sub-claim D fails that same standard at line 914: the regularizer is exactly the negative log-density of a `Gamma(c_0 + 1, b_0)` conjugate prior on Gaussian precision per `[Bishop 2006 §2.3.6; Murphy 2012 §4.6.1]`, the closed-form `α_i* = c_0/(b_0 + KL)` at line 937 is exactly the Gamma-MAP shrinkage estimator, yet the manuscript labels the choice "natural" without identifying the standard Bayesian machinery that makes it canonical. Both teams reached this conclusion on the same primary sources after Phase 3.

Blue's defense that this is "editorial improvement, not a structural correctness issue" (`02_blue_opening.md:49`) and that the gap reduces to a single-citation fix (`03_blue_rebuttal.md:29-34`) is correct on the scope of the remedy and on the mathematical content (no equation needs to change, no implementation needs to change, no gradient changes). Blue's Move-1 rebuttal that empirical Bayes does not formally require a conjugate prior `[Murphy 2012 §5.6; Casella 1985]` is also correct. But the compound claim is stated as "theoretically pure" — a standard the §4–§5 series enforced by demanding that load-bearing canonical correspondences be named. Under that standard, an under-cited canonical identification on a load-bearing piece of §3.7 falls short of "theoretically pure," even if it does not fall short of "mathematically correct."

Sub-claims C, E, F survive clean (both sides agreed; red conceded C and F in rebuttal, withdrew E). Sub-claim A (Ω parameterization) is honestly disclosed at line 656; red's residual attack on framing-level prominence at line 612 is a minor editorial qualifier, not a structural defect. Sub-claim B (dual-fiber state space) is declared non-load-bearing at line 677 and no §3 equation references `Φ_i, Φ̃_i`; red's residual attack on `K_q` vs `K_p` independence language is a one-line clarification opportunity, not a structural defect.

The compound-claim adjudication rule at `00_claim.md:36` reads: "If one or two sub-claims fail editorially but the others survive structurally, the judge should issue RED_WINS with the falling sub-claims identified, and the action items should be scoped accordingly." Sub-claim D falls editorially on a load-bearing piece; sub-claims A and B carry minor framing qualifications; the compound claim as written ("robust, theoretically pure, and mathematically correct") does not survive intact. The remedy is scoped to citation additions and framing flags, not to equation rewrites.

## Action

Apply scoped editorial fixes to §3 of `Attention/GL(K)_attention.tex`. No equation changes, no implementation changes, no structural revision.

1. Sub-claim D (primary). At `Attention/GL(K)_attention.tex:914-917`, replace "A natural choice is a log-barrier form such as" with an identification sentence and citation. Suggested text following blue's rebuttal proposal: "Choose `R(α_i) = b_0 α_i - c_0 log α_i`, the negative log-density of a `Gamma(α_i; c_0 + 1, b_0)` distribution — the conjugate prior for the precision parameter of a Gaussian likelihood `[Bishop 2006 §2.3.6; Murphy 2012 §4.6.1]`." At line 937 add a one-clause note that `α_i* = c_0/(b_0 + KL(q_i ‖ p_i))` is the MAP estimate of `α_i` under that Gamma prior with the linear-in-α self-coupling penalty `α_i · KL(q_i ‖ p_i)` playing the role of the sufficient statistic. The empirical-Bayes claim at line 944 then inherits the standard Gamma-EB procedure rather than reading as a free assertion.

2. Sub-claim A (secondary). At `Attention/GL(K)_attention.tex:612` where Ω is first introduced as a "gauge transport operator," add a forward-pointing flag — one clause — that the §3 parameterization `Ω_ij = exp(φ_i) exp(-φ_j)` restricts the framework to a globally trivial principal G-bundle (flat connection), with the edge-relaxed non-flat form deferred to `[Dennis 2025 it]` per the disclosure at line 656. This avoids the framing gap where a reader of the §3.2 introduction expects non-trivial gauge transport before reaching the Lemma 1 / line-656 disclosure.

3. Sub-claim B (secondary). At `Attention/GL(K)_attention.tex:580-583` where `k_i ∈ R^{K_q}` and `m_i ∈ R^{K_p}` are introduced, add one sentence noting that `K_q` and `K_p` are independent dimensional parameters of the scaffold, and that §3 operates entirely on the belief channel `(q_i, p_i, β_ij, Ω_ij)` so that `K_p` plays no role in any §3 equation; the model-channel scaffold is deferred to the companion treatment per line 677.

Sub-claims C, E, F require no action.

After these edits, the compound claim of §3 — that the derivation is robust, theoretically pure, and mathematically correct — is defensible at the same editorial standard the §4–§5 series enforced. The mathematics is already correct; the citations and framing then match.
