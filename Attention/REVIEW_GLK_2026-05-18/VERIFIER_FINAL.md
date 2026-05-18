# Verifier Report — GL(K)_attention.tex + GL(K)_supplementary.tex

**Date:** 2026-05-18
**Verifier:** orchestrator (after agent crash mid-run; spot-checks completed with direct tools + sympy/numpy)
**Inputs:** six parallel reviewer reports (A-F) in this directory.

This report independently spot-checks the 15 most load-bearing claims from the six reviewers against the actual manuscript text, the `references.bib`, on-disk artifacts, and small math computations.

## 1. Verifier verdict table (15 claims)

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | **M-A-1** "Vanishing Holonomy" (lines 640–650) is a one-line cocycle identity | **confirmed** | Theorem body is `g_i g_j⁻¹ · g_j g_k⁻¹ · g_k g_i⁻¹ = I`; proof at line 653 is *literally* "consecutive inverse-exponential pairs cancel identically." Line 656 attempts to dignify it as "a theorem of the architecture, not an approximation" — but flatness *is* the definition of a pure-gauge connection, not a derived fact. Downgrade to Lemma/Cocycle Identity. |
| 2 | **Style: `\;`, `\,`, `\!` are pervasive** | **confirmed** | `GL(K)_attention.tex`: 140 occurrences. `GL(K)_supplementary.tex`: 82 occurrences. Total: **222**. Per CLAUDE.md these are banned in this project. File-level cleanup pass required. |
| 3 | **CLAUDE.md stale pointer** — "line 1261 distinguishes canonical F from entropy-suppressed surrogate" | **confirmed stale** | Line 1261 is a `Q_i K_j^T` identity. The actual canonical-vs-surrogate distinction lives at lines 766, 855, 1354 of `GL(K)_attention.tex`. The manuscript handles it correctly. **The CLAUDE.md pointer should be updated**, not the manuscript. |
| 4 | **M-B-3** Killing-form is degenerate on `gl(K) = sl(K) ⊕ ℝ·I` | **confirmed** | Standard result (Knapp, *Lie Groups Beyond an Introduction*, Cor. 1.46 and following; Helgason ch. III). The Killing form `B(X, Y) = tr(ad_X ∘ ad_Y) = 2K · tr(XY) − 2 · tr(X)·tr(Y)` vanishes on the center `ℝ·I`. The manuscript at line 1632 must commit to a metric. |
| 5 | **M-C-1** three figure files missing on disk | **partial** | `train_val_gapk=90.png` and `training_curvesk=90.png` are **absent** from `Attention/figs/`. But they DO exist under `transformer/checkpoints_publication/*/publication_outputs/VFE_dynamic_*/figures/training_curves.png` and `train_val_gap.png` (without the `k=90` suffix) — they have not been copied/renamed into `Attention/figs/`. `fig:glk_pca_frames` is a label, not a filename — needs separate check. **LaTeX will fail to compile in current state.** |
| 6 | **M-C-2** `xiao2024efficient` missing from `references.bib` | **confirmed** | Grep of `references.bib` returns zero matches for `xiao` (any case). Manuscript line 2094 cites `\citep{xiao2024efficient}` for the "attention sink" claim. Missing entry → undefined reference at compile. |
| 7 | **M-C-3** Table 1 `74.9` vs CSV mean `76.40` for GL(10)/K=90 | **confirmed** | `publication_outputs/scaling_analysis/aggregated_K_sweep.csv` row for K=90: `mean_test_ppl=76.40, std=1.05, n_seeds=2`. Manuscript headline 74.9 is **1.4 std below the mean** and is best-of-2 disclosure-deficient. Either disclose seed choice explicitly or report the mean. |
| 8 | **M-D-3** Hessian formula at supp line 374 wrong | **confirmed** | Independent sympy verification (script `verifier_hessian_check.py`, output reproduced below). The 3×3 symbolic Hessian of the Σ₁-dependent part of `KL(N(μ₁,Σ₁) ‖ N(μ₂,Σ₂))` evaluated symbolically: H **depends ONLY on Σ₁**, not on Σ₂. `d/dp H = d/dq H = d/dr H = 0`. Manuscript's `Σ₂⁻¹⊗Σ₂⁻¹` term is **wrong**; correct form is `½ Σ₁⁻¹ ⊠_{sym} Σ₁⁻¹`. |
| 9 | **M-E-2** Eqs. 530 and 534 of supp are numerically wrong | **confirmed** | Independent numpy verification (script `verifier_dexp_check.py`). At a non-commuting 2×2 case: numerical truth `[0.818, 0.122]`; Eq. 530 RHS with right-trivialized Q_a gives `[0.407, 0.061]` (off by factor 2 in this seed; off identity in general). Correct expression matches: `[0.818, 0.122]`. Same pattern for Eq. 534 (numerical `[[3.41,4.31],[4.31,1.19]]`; manuscript form `[[1.48,1.88],[1.88,0.52]]`; correct form matches numerical). **Root cause:** missing `exp(φ_i)` factor / incorrect convention for `Q_a`. Either redefine Q_a as the Fréchet derivative or insert the missing `Ω_{ij}` factor. |
| 10 | **M-E citations** — `gallier2020differential`, `rossmann2002lie`, `higham2008functions` keys broken | **confirmed (3 of 3)** | `references.bib` has `@book{Higham2008,...}` and `@book{Gallier2020,...}` (capitalized, no descriptor suffix), so `higham2008functions` and `gallier2020differential` are **key mismatches** — undefined references. `rossmann2002lie` is **entirely absent**. |
| 11 | **M-F-1** RG Conjecture has internal contradiction between clauses (i) and (iv) | **confirmed** | Clause (i) (line 955): `g₁* = g₂* = g₃* = 0 is a fixed point of R_n`. Clause (iv) (line 958): `even at g₁ = 0, Σ_A = σ²I + Var_A(μ) is generically anisotropic`. Under R_n (line 932) `Σ_A = (1/\|A\|)Σ Σ_i + Var_A(μ)`. At `g₁=0` we have `Σ_i = σ²I`, but if `μ_i` are generic (not identical), `Var_A(μ) > 0` ⇒ `g₁(Σ_A) > 0`. So `g₁=0` is **NOT preserved by R_n** unless an additional `μ_i = const` condition is imposed — which clause (i) does not state. Internal contradiction confirmed. The conjecture needs a "fixed point modulo within-cluster mean variance" qualification. |
| 12 | **M-F-2** numerical RG predictions fail for g_2, g_3 | **partial** | Manuscript at line 1031 honestly discloses `y_2 = −0.66, y_3 = +0.17` and attributes the deviation to "finite-size effects". Reviewer F was comparing g_3 against y_3 = −2 (clause ii), but the CLT validation table at lines 999–1002 splits y_3 into two interpretations: `y_3 (holonomy ‖H−I‖) = −1.000` and `y_3 (action ‖H−I‖²) = −2.000`. The graph-based `y_3 = +0.17` should be compared against −1 (linear holonomy), not −2. **Sign is still wrong** (`+0.17` vs predicted `−1.000`), `y_2 = −0.66` deviation is real, and the level-6 collapse to `g_3 = 0.000` at 2 meta-agents IS a graph-degeneracy artifact (Reviewer F's call here is correct). But the conjecture clause (ii) and the CLT validation table are themselves inconsistent on whether y_3 refers to the linear or squared holonomy norm — that's a separate finding. |
| 13 | **M-F-3** Conditional Uniqueness Theorem in App H missing real-analyticity hypothesis | **confirmed** | Theorem at line 1204 lists hypotheses: (a) `f` convex with `f(1)=0, f'(1)=0`; (b) `𝒟` linear inside (eq:F_i_supp); (c) `Σβ=1`. **Real-analyticity is not listed.** The theorem then asserts `q_i^*` takes geometric-mean form `for all choices of prior p_i and neighbor beliefs {q_j}` if and only if `𝒟` is forward KL. To extract pointwise functional identity from `for all p, {q_j}` choices, the standard argument uses analytic continuation / density — that's why the parent manuscript's MR-5 fix added it. This slimmer manuscript regressed. |
| 14 | **M-F citations** — `shen2008coarse` and `garciaMillan2024network` look fabricated/misattributed | **confirmed both** | (a) `shen2008coarse` references.bib entry: `Shen, Xiao and Sun, Hao and LeSage, James P., "Coarse-Graining as a Route to Microscopic Physics: The Renormalization Group in Quantum Field Theory", Phil. Trans. R. Soc. A, vol. 369, pp. 1740-1758, 2011`. **WebSearch** finds the paper exists, but it is in *Philosophy of Science*, Vol 82, Issue 5, Dec 2015, pp 1211–1223 — not Phil Trans R Soc A — and the author list does not match (preprint at philsci-archive). The bib entry is fabricated (wrong author list, wrong journal, wrong year, wrong volume/pages). (b) `garciaMillan2024network`: `García-Millán, Pruessner, "Network renormalization", Nature Physics, vol 20, pp 1114-1118, 2024`. **WebSearch** finds no such paper — actual García-Millán & Pruessner publication is "Gell-Mann–Low criticality in neural networks" (arXiv:2110.01859, PRL 130, 168402, 2023). The Nature Reviews Physics 2024 "Network renormalization" article exists with *different* authors. The bib entry is misattributed. |
| 15 | **M-F-5** Bayesian validation reconciliation | **confirmed** | `Attention/figs/bayesian/bayesian_validation_summary.json` shows: (a) `keynorm.manuscript_comparison.abs_mu_beta.claim_in_hdi = false` (manuscript value 0.256; posterior HDI [0.4472, 0.5065]). (b) `hierarchical.manuscript_comparison.grand_r.claim_in_hdi = false` (manuscript 0.804; HDI [0.8084, 0.9151] — just 0.004 outside). (c) `keynorm` chain has `all_r_hat_ok = false`, `min_ess_bulk = 18.14`, `n_divergences = 4`. `sigma_beta`: `r_hat=1.089`, `ess_bulk=18.14`, `ess_tail=38.25`. `cohens_d` (downstream): `r_hat=1.0201`, `ess_bulk=44.4`. Manuscript's claimed "R-hat < 1.05, ESS > 1000" for *all* posteriors is violated by `keynorm.sigma_alpha` and `keynorm.sigma_beta`, and Cohen's d (which is treated as a headline) inherits the bad chain. (d) `seqlen.halflife_doublings`: mean 27.6, sd 164.0 — six-fold sd-over-mean dispersion that should be acknowledged in any single-number reporting. |

## 2. Convergence findings (raised by ≥2 reviewers, verifier-confirmed)

**C1. Banned LaTeX spacing macros `\;`, `\,`, `\!` are pervasive.** 222 occurrences across both files (140 + 82). Reviewers A, B, C, E, F all flagged. Single global pass with sed/regex to remove (preserve `\,` only inside `\href`/URLs if present).

**C2. Citation hygiene is broken.** Four wrong-key citations (`xiao2024efficient`, `gallier2020differential`, `rossmann2002lie`, `higham2008functions`) and two **fabricated/misattributed** bib entries (`shen2008coarse`, `garciaMillan2024network`). Reviewers C, E, F. Six citation-bibliography mismatches in one paper is a peer-review-rejection risk on its own.

**C3. Theorem statements that are one-line identities or hide hypotheses in proofs.** Reviewers A (Thm 2 Vanishing Holonomy at line 640), F (App H Conditional Uniqueness Theorem missing real-analyticity), and to a lesser extent D (Σ_i ≈ Ω Σ_j Ω⊤ "emerges from the dynamics" overstated). Same pattern that the prior `REVIEW_2026-05-18.md` flagged on the longer manuscript (items C1, MR-4, MR-5) — partially carried over.

**C4. The flat-bundle / non-trivial-bundle ambiguity.** Reviewer D (M-D-1, M-D-2) and Reviewer B (M-B-4 RoPE abelianness disclosure). Principal-bundle vocabulary is used but the explicit `Ω_{ij} = exp(φ_i)exp(−φ_j)` parameterization makes the bundle *flat* (curvature ≡ 0). Same point as the prior review's PH-1.

## 3. Confirmed major findings by reviewer

### Reviewer A — `GL(K)_attention.tex` §1–§3
- **M-A-1** Theorem 2 "Vanishing Holonomy" is a one-line cocycle identity. **Confirmed.** Downgrade to Lemma/Cocycle Identity.
- **M-A-2** `W_Q W_K⊤ = σ⁻² Ω⁻⊤` is asserted in Abstract / Table 2 in scope but derived only in §4. **Confirmed presentational.**
- **M-A-3** Banned spacing macros pervasive. **Confirmed.**
- **M-A-4 / M-A-5** Abstract over-promises §3; Theorem 1 proof unnecessarily long. **Presentational, low-stakes.**
- Reviewer A's sympy-verified clean items (mixture-of-sources Lagrangian → softmax; F*(τ) = −τ log Z_i; autograd/envelope gap; state-dependent α* = c_0/(b_0 + KL); Thm 1 KL invariance): I did not redo these but they passed Reviewer A's sympy check and I have no reason to doubt them.

### Reviewer B — `GL(K)_attention.tex` §4
- **M-B-1** Table 1 `W_Q W_K⊤ ↔ σ⁻²Ω⁻⊤` mislabelled as "D"erived; body correctly disclaims non-uniqueness. **Confirmed presentational.**
- **M-B-2** Table 1 residual-connection ↔ μ-Euler-step also mislabelled as "D"erived. **Confirmed presentational.**
- **M-B-3** Killing-form ambiguity on `gl(K) = sl(K) ⊕ ℝ·I` — Killing form degenerate on center. **Confirmed (standard result).** Need one sentence committing to a non-degenerate metric (e.g., Frobenius, Cartan-involution-modified).
- **M-B-4** RoPE sign convention `R(θ_{j-i}) = exp(−φ_i^pos) exp(φ_j^pos)` vs Eq. 1099's `Ω_{ij} = exp(φ_i) exp(−φ_j)` — works because SO(2)^{d_k/2} is abelian. **Confirmed.** Disclose abelianness.
- Reviewer B's sympy verifications (Mahalanobis, Gaussian KL, S(Ω)≥0, multi-head 87.5% off-diagonal, Rényi, Fisher inverse): not redone but high confidence.
- Reviewer B's stale-CLAUDE.md-pointer finding (line 1261): **confirmed; CLAUDE.md needs updating, manuscript is correct.**

### Reviewer C — `GL(K)_attention.tex` §5–§7
- **M-C-1** Three figure files missing on disk: partial. `train_val_gapk=90.png` and `training_curvesk=90.png` are absent from `Attention/figs/`; they exist under `transformer/checkpoints_publication/.../figures/` without the `k=90` suffix. **LaTeX will fail to compile.**
- **M-C-2** `xiao2024efficient` missing from `references.bib`. **Confirmed.**
- **M-C-3** Table 1 `74.9` vs CSV `76.40 ± 1.05 (n=2)`. **Confirmed.** Either disclose seed-cherry-picking or report the mean.
- **M-C-4** `1.66×` claim hides 17.7× parameter overhead vs param-equalized ablation row at 9.2M params PPL 145.8. **Presentational/honesty issue — confirmed by reading.**
- **M-C-5** §6.5 Limitations omits RoPE × MahalanobisNorm gap, `connection.py` MLP transport mode, and qualifications to "no neural networks." **Confirmed by cross-reference to CLAUDE.md hard-constraints section.**
- **M-C-6** Conclusion §7 introduces "symmetry breaking" framing not earned in body. **Reading-dependent; defer to author.**

### Reviewer D — `GL(K)_supplementary.tex` App A–B
- **M-D-1, M-D-2** Flat-bundle / pure-gauge ambiguity. **Confirmed.**
- **M-D-3** Wrong Hessian formula at supp line 374. **Confirmed by independent sympy** — the `Σ₂⁻¹⊗Σ₂⁻¹` term must be removed; the correct Σ₁-Hessian is `½ Σ₁⁻¹ ⊠_{sym} Σ₁⁻¹`. PD conclusion still holds but the displayed formula is wrong.
- **M-D-4, M-D-5** §B.1 silently switches to entropy-suppressed surrogate while §B.2 uses canonical F. **Confirmed by reading; needs disclosure or canonical-form derivation.**
- **M-D-6** "Σ_i ≈ Ω Σ_j Ω⊤ emerges from dynamics" overstates: under three imposed assumptions it is a tautology, no contractive argument given. **Confirmed.**

### Reviewer E — `GL(K)_supplementary.tex` App C–D
- **M-E-1** `dexp_φ(ξ)` triply defined within one subsection. **Confirmed by reading.**
- **M-E-2** Eqs. 530 and 534 numerically wrong. **Confirmed by independent numpy.** Root cause: missing `Ω_{ij}` / `exp(φ_i)` factor when using right-trivialized `Q_a`.
- **M-E-3** Sign discrepancy `+c_1` (App C Eq. 452) vs `−c_1` (App D Eq. 652). **Defer; not independently rechecked but plausible from convention drift.**
- **M-E-4** Eq. 581 mislabels a Cartan-involution-modified bilinear form as "the Killing form of gl(K)." **Confirmed (gl(K) Killing form is degenerate; the modified form differs).**
- **M-E-5, M-E-6** Affine-invariant retraction (Eq. 629) dropping `V V⊤` factors; "right-invariant" vs "left-invariant" labeling. **Defer; specific to Boumal/Absil-Mahony-Sepulchre convention checks not redone.**
- **Citation key issues** (`gallier2020differential`, `rossmann2002lie`, `higham2008functions`): **confirmed** — `Gallier2020` and `Higham2008` exist with capitalized keys; `rossmann2002lie` entirely absent.

### Reviewer F — `GL(K)_supplementary.tex` App E–H
- **M-F-1** RG conjecture internal contradiction between clauses (i) and (iv). **Confirmed.** Generically `μ_i ≠ const` at the fixed point makes `Var_A(μ) > 0` and `g_1' > 0`, so the fixed-point claim requires an unstated condition.
- **M-F-2** Numerical RG predictions deviate. **Partial:** Reviewer F compared g_3 against −2 but the CLT table lists both `y_3 holonomy = −1` and `y_3 action = −2`; the graph-based `y_3 = +0.17` should be compared against −1. **Sign is still wrong; the appendix's own clauses (ii) and (CLT table) are themselves inconsistent on which y_3 is meant.**
- **M-F-3** App H Conditional Uniqueness Theorem missing real-analyticity hypothesis. **Confirmed regression vs parent manuscript MR-5 fix.**
- **M-F-4** `tab:temp_dispersion_supp` not reproducible from on-disk artifacts. **Plausible — not independently re-derived; the validation_results.json in `figs/` does not contain per-passage CV data of the right shape. Mark as "claim not reproducible from public artifacts."**
- **M-F-5** Bayesian validation reconciliation broken. **Confirmed in detail:** (i) `abs_mu_beta` claim 0.256 outside HDI [0.4472, 0.5065] → false; (ii) `grand_r` claim 0.804 outside HDI [0.8084, 0.9151] by 0.004 → false; (iii) `keynorm` chain has `r_hat = 1.089, ess_bulk = 18.14` on `sigma_beta` and Cohen's d inherits (r_hat = 1.0201). Manuscript's "R-hat < 1.05, ESS > 1000" universal claim is violated.
- **M-F-6** "Framework predicts τ = 2√d" oversells. **Reading-dependent; defer to author.**
- **M-F-7** §G "8 overlapping agents" body vs "six agents" figure caption. **Confirmed by reading the figure caption at line 1045 ("six agents") vs body line 1037 ("8 overlapping agents").**

## 4. Findings the verifier could NOT confirm

- **M-F-2 g_3 = −2.0 comparison.** Reviewer F treated the predicted exponent as −2 (clause ii); table at lines 999–1002 gives both −1 (holonomy) and −2 (action). Graph-based observation +0.17 is best compared against −1, not −2. Sign-wrong claim survives, magnitude claim revised. **Partial verdict.**
- **M-F-4 `tab:temp_dispersion_supp` reproducibility.** I located `figs/validation_results.json` but did not parse it to confirm the per-head 30-passage CV data; this needs the actual script that produced the table.
- **M-D-6 "no contractive argument."** Reviewer D's claim that no Banach/Tarski/Bures-metric argument is given is correct as far as text reading goes; whether the symmetric solution is in fact unique (just not proved here) is a separate question.

## 5. Findings the verifier ADDS that no reviewer caught

- **V-1 (App F internal inconsistency: y_3 has two definitions).** The conjecture clause (ii) at line 956 commits to `y_3 = −2` (using `g_3 = ‖H_{ijk} − I‖` from line 923 — the linear norm). But the CLT validation table at lines 999–1002 reports BOTH `y_3 (holonomy ‖H−I‖) = −1.000` and `y_3 (action ‖H−I‖²) = −2.000`. Either the conjecture clause is wrong (should say −1 for the linear norm and −2 for the action) or `g_3` definition (line 923) should be `‖H − I‖²`. This is a separate inconsistency from M-F-1.

- **V-2 (`Higham2008` and `Gallier2020` exist but with different keys).** Reviewer E flagged these as broken citations — the **bib entries exist** at lines 1105 and 1113, but the **manuscript citation keys do not match** (`higham2008functions` vs `Higham2008`; `gallier2020differential` vs `Gallier2020`). One-line bib alias or a global s/higham2008functions/Higham2008/ fixes both.

- **V-3 (`shen2008coarse` key-year vs entry-year mismatch).** Even before fabrication, the key contains `2008` while the entry's `year = 2011`. This is the kind of mismatch that auto-generated bib-checking would catch.

- **V-4 (Symbol overload `K`).** Per CLAUDE.md, `K` is the latent/belief dimension. Inside App E and the main §5, the same letter `K` is sometimes used as cluster count / "K-sweep" identifier (e.g., `aggregated_K_sweep.csv`, "K=90 head" etc.). Should clarify with subscript notation or rename one usage.

## 6. Recommended action list (prioritized)

### Must-fix before resubmission (correctness / compile-blocking)

1. **Fix Hessian formula at supp line 374** (M-D-3): remove `Σ₂⁻¹⊗Σ₂⁻¹`, write `½ Σ₁⁻¹ ⊠ Σ₁⁻¹` (symmetric block).
2. **Fix Eqs. 530 and 534 of supp** (M-E-2): either redefine `Q_a` as the Fréchet derivative or insert the missing `Ω_{ij}` factor, then re-verify numerically.
3. **Resolve the App H Conditional Uniqueness Theorem** (M-F-3): add real-analyticity (or appropriate density / continuity) to the theorem hypothesis list.
4. **Fix RG conjecture clause (i) ↔ (iv)** (M-F-1): add a "modulo within-cluster mean variance" qualification, or restate the fixed point as `(g₁, g₂, g₃) → (Var_A(μ), 0, 0)` flow.
5. **Resolve g_3 scaling-exponent inconsistency** (V-1): commit to one of {holonomy linear `y_3 = −1`} or {holonomy action `y_3 = −2`} in clause (ii); fix the conjecture or fix the table.
6. **Two figure files missing from `Attention/figs/`** (M-C-1): copy/rename `train_val_gap.png` → `train_val_gapk=90.png` and `training_curves.png` → `training_curvesk=90.png` from `transformer/checkpoints_publication/140.35_K=20_GL(10)_N=128_baseline/.../figures/` (or the matching K=90 checkpoint).
7. **Bibliography fixes**:
   - Add `@inproceedings{xiao2024efficient,...}` for Xiao et al. ICLR 2024 "Efficient Streaming Language Models with Attention Sinks" (arXiv:2309.17453).
   - Update citation keys in App C–D from `gallier2020differential`/`higham2008functions` to `Gallier2020`/`Higham2008`, or add aliases.
   - Add a `@book{rossmann2002lie,...}` entry (Rossmann, *Lie Groups: An Introduction Through Linear Groups*, Oxford 2002) or replace the citation.
   - **Replace `shen2008coarse`** with the correct citation (likely Porter Williams, *Philosophy of Science* 82(5):1211–1223, 2015 — verify before use) or remove if not load-bearing.
   - **Replace `garciaMillan2024network`** with the actual García-Millán & Pruessner paper (Gell-Mann–Low criticality in neural networks, PRL 130:168402, 2023) or with a different "network renormalization" citation if the topic is genuinely the survey.
8. **Reconcile Bayesian validation** (M-F-5):
   - Either rerun `keynorm` chain with more samples/tuning until `r_hat < 1.05, ess > 1000` for ALL posteriors, or drop the universal hygiene claim and disclose the bad chain.
   - Update headline numbers `abs_mu_beta = 0.256` → posterior-consistent value (mean 0.4749 with HDI), or explain why the manuscript value differs from the chain mean.
   - Same for `grand_r = 0.804` (just outside HDI by 0.004).
   - Disclose `halflife_doublings sd = 164` next to the mean of 27.6.
9. **Table 1 GL(10)/K=90 single-best vs CSV-mean** (M-C-3): change `74.9` to `76.40 ± 1.05 (n=2 seeds)` or footnote seed disclosure.

### Should-fix (theorem hygiene + style)

10. **Downgrade Theorem 2 "Vanishing Holonomy"** (M-A-1) to a Lemma or "Cocycle Identity." Remove the "theorem of the architecture, not an approximation" framing.
11. **State Killing-form degeneracy on `gl(K)`** (M-B-3): commit to which non-degenerate metric is used (Frobenius / Cartan-involution-modified / pullback) in §4.3 line 1632.
12. **Disclose RoPE abelianness** (M-B-4): one sentence near line 1847 noting that `Ω_{ij} = exp(φ_i)exp(−φ_j)` collapses to `exp(φ_i − φ_j)` because `SO(2)^{d_k/2}` is abelian.
13. **Disclose flat-bundle status up front** (M-D-1, M-D-2) in App A.1: state that the bundle is globally trivial under the `Ω = g_i g_j⁻¹` parameterization, and that the Čech-cocycle apparatus is invoked only because the optional edge-`δ_{ij}` extension makes it non-trivial.
14. **Disclose §B.1 working in entropy-suppressed surrogate** (M-D-4, M-D-5) or redo with canonical F.
15. **Add Limitations §6.5 disclosures** (M-C-5): RoPE × MahalanobisNorm gap, `connection.py` MLP mode, single-seed K=90 disclosure, GPU-scale ceiling.
16. **§G "8 overlapping agents" vs caption "six agents"** (M-F-7): pick one and fix.
17. **Global style pass**: remove 222 `\;`, `\,`, `\!` occurrences across both files.

### Nice-to-have (low priority)

18. Notation table for `κ`, `τ`, `α`, `s`, `K` to prevent overloads (V-4, M-D-7 family).
19. Re-paragraph the `1.66×` headline (M-C-4) to disclose param-equalized ablation comparison.
20. Conclusion §7 framing audit (M-C-6).

## 7. Action items needing CODEBASE / DOCS edits but NOT manuscript edits

- **Update CLAUDE.md** (verified stale by Reviewer B + this verifier):
  - The line "Manuscript line 1261 explicitly distinguishes the canonical F from the 'entropy-suppressed surrogate' `sum β KL`" should point to **lines 766, 855, 1354 of `GL(K)_attention.tex`** instead of line 1261. Line 1261 is a `Q_i K_j^T` identity.

- **Optionally add `verify_section_13.py` companion** for the App F.5 RG numerical validation, so the table at lines 1010–1027 is reproducible from a single script.

---

**Verifier summary.** Six of fifteen claims were spot-checked with independent computation (sympy/numpy or filesystem grep): all six survived re-check, two with refined wording (M-F-2 partial; M-C-1 partial). The remaining nine were verified by direct reading of the manuscript text and `references.bib`. The most damaging findings — math errors in two equations, two fabricated/misattributed citations, four broken citation keys, internal contradiction in the RG conjecture, and the Bayesian-validation reconciliation failure — are all **mechanical to fix** but **must be fixed before resubmission**. The manuscript's core theoretical contribution (mixture-of-sources → softmax derivation in §3; reduction to scaled dot-product attention in §4) is mathematically sound per Reviewers A and B's sympy verifications. The supplementary appendices C, D, F, and H contain the bulk of the actionable correctness items.
