# Fresh-eyes review — `GL(K)_attention.tex` + `GL(K)_supplementary.tex` — 2026-05-18 (night)

**Reviewer:** `vfe-manuscript-reviewer` (fresh-eyes pass over current file state, post-fix)
**Inputs:** `VERIFIER_FINAL.md` (15 prior findings), `2026-05-18_edits.md` (today's applied fixes), current `.tex` files.
**Scope:** (a) triage prior findings against current state, (b) re-evaluate four partial verdicts, (c) fresh-eyes math/citation/notation sweep.
**Authority:** standard references (Knapp, Hall, Boumal, Nakahara, Higham, Vaswani, Friston). User CLAUDE.md is *project policy* and not theoretical canon.

---

## (a) Still-Present Triage of 15 Verifier Findings

| # | Verifier ID | Verdict | Evidence (file, line) |
|---|---|---|---|
| 1 | M-A-1 (Vanishing Holonomy → Lemma) | **FIXED** | `GL(K)_attention.tex:640` now reads `\begin{lemma}[Cocycle Identity for Vertex-Frame Transport]`. **But** see new finding N-3 below: cross-references at lines 512, 1730, 2215, 2271 still hard-code "Theorem~\ref{...}". |
| 2 | C1 (banned `\;` `\,` `\!`) | **FIXED** | Direct substring count: 0 in main, 0 in supp (verified via `chr(92)+';'`, `chr(92)+','`, `chr(92)+'!'`). Earlier regex-based counts that returned hundreds were matching `\\` (row breaks) inside tables, not the spacing macros. |
| 3 | CLAUDE.md stale pointer (line 1261) | **FIXED** | `2026-05-18_edits.md:1003-1006` documents the pointer update. Not a manuscript edit. |
| 4 | M-B-3 (Killing form degeneracy disclosure) | **PARTIALLY-FIXED** | Main `GL(K)_attention.tex:1632` correctly commits to "the Cartan-involution-modified bilinear form `⟨X,Y⟩ = −½ tr(X θ(Y))`" and notes Killing degeneracy on the center. **But** supplementary `GL(K)_supplementary.tex:580–588` still calls the **modified** form "Killing form" and makes a **false positive-definiteness claim** about the actual Killing form. See new finding N-1. |
| 5 | M-C-1 (missing figure files) | **FIXED** | `Attention/figs/training_curvesk=90.png` and `train_val_gapk=90.png` present on disk (verified by `ls`). |
| 6 | M-C-2 (`xiao2024efficient` missing) | **FIXED** | `references.bib` contains `@inproceedings{xiao2024efficient,...}` (entry confirmed by grep). |
| 7 | M-C-3 (`74.9` vs CSV mean `76.40 ± 1.05`) | **FIXED** | `GL(K)_attention.tex:2136,2158` now reports `76.4 ± 1.05` (n=2). |
| 8 | M-D-3 (Hessian formula at supp line 374) | **FIXED** | `GL(K)_supplementary.tex:376–381` now reads `½ Σ₁⁻¹ ⊠_sym Σ₁⁻¹` with prose stating "the Hessian depends only on Σ₁". |
| 9 | M-E-2 (supp Eqs. 530, 534 numerically wrong) | **FIXED** | `GL(K)_supplementary.tex:529–537` now uses `Q_a^{(i)} Ω_{ij}` form with the identity `D_φ(exp)[T_a] = Q_a^{(i)} exp(φ_i)` made explicit at line 537. |
| 10 | M-E citations (`gallier2020differential`, `rossmann2002lie`, `higham2008functions`) | **FIXED** | All three bib entries now match cited keys (`Gallier2020`, `Higham2008`, `rossmann2002lie`). |
| 11 | M-F-1 (RG conjecture clause (i) vs (iv)) | **FIXED** | `GL(K)_supplementary.tex:920–925, 953–960` now splits anisotropy into `g_1^{(orig)}` and `g_1^{(emer)}`; clause (i) restricted to intrinsic channel; clause (iv) describes emergent floor. |
| 12 | M-F-2 (numerical RG predictions fail) | **PARTIALLY-FIXED — needs author decision.** See partial re-evaluation below. The conjecture clause (ii) now explicitly lists `y_3 = −1` (linear) and `y_3^{(action)} = −2` (squared), removing the internal ambiguity. The empirical graph-based `y_3 = +0.17` still has the wrong sign relative to the linear prediction `−1`, but the manuscript at `GL(K)_supplementary.tex:1031` acknowledges this as a finite-size effect rather than presenting it as a confirmation. |
| 13 | M-F-3 (App H real-analyticity hypothesis) | **FIXED** | `GL(K)_supplementary.tex:1209` adds "real-analytic on $(0,\infty)$" to the hypothesis list. `:1281` invokes the identity theorem for real-analytic functions to extend from a subinterval to all of $(0,\infty)$. |
| 14 | M-F citations (`shen2008coarse`, `garciaMillan2024network`) | **FIXED** | Replaced with `kadanoff1966scaling` and `garciaperez2018multiscale` (confirmed by grep at `GL(K)_supplementary.tex:99,128`; bib entries confirmed at `references.bib:2402,2353`). |
| 15 | M-F-5 (Bayesian validation reconciliation) | **FIXED** | `GL(K)_supplementary.tex:899–909` now reports the `keynorm` chain mixing caveat (`r-hat = 1.09, ESS_bulk = 18` on `σ_β`; Cohen's d inheriting `r-hat = 1.02, ESS_bulk = 44`), grand `r̄ = 0.804` placed alongside hierarchical posterior `[0.808, 0.915]`, and halflife `27.6 ± 164` dispersion disclosed. |

**Summary of (a):** 13 of 15 confirmed prior findings **fixed**. 1 **partially fixed** (M-B-3 — main text fixed but supplementary regressed; see N-1). 1 **partially-fixed/needs-author-decision** (M-F-2 — clause-ii ambiguity resolved, but empirical-sign disagreement still present; see partial re-evaluation).

---

## (b) Partial Re-evaluation

### M-F-2 — graph-based RG `y_3 = +0.17` vs prediction

**Verdict:** **partial → confirmed but acknowledged honestly.**

Under the current clause (ii) (`GL(K)_supplementary.tex:956`), the prediction for the linear-holonomy exponent is `y_3 = −1` (the squared-norm `y_3^{(action)} = −2` is now an explicit separate quantity, resolving V-1). The graph-based measurement `y_3 = +0.17` (line 1031) is **wrong-signed** relative to the linear prediction.

The manuscript does **not** present this as a confirmation. Line 1031 attributes the deviation to "finite-size effects inherent in spectral clustering" and notes that "at N = 128 with binary coarse-graining, the deepest levels contain only 4–8 meta-agents." This is an honest disclosure rather than overselling, and the CLT validation (`:993–1005`) confirms the mathematical scaling exactly. The level-6 collapse `g_3 = 0.000` at 2 meta-agents (line 1024) is correctly identified as a graph-degeneracy artifact (a triangle requires three vertices).

**However**, the prose still uses "deviate from the CLT predictions" without flagging that the sign itself is wrong — a `+0.17` measurement is qualitatively different from a "near-zero" deviation that might be explained by small N. Author decision recommended: either (i) acknowledge that the graph-based exponents are not in the same regime as the CLT predictions (different physical mechanism: spectral-clustering correlations vs i.i.d. averaging) and weaken "deviate from" to "do not match," or (ii) defer the graph-based numerics entirely to the companion paper and replace the table with the CLT-only validation.

### M-F-4 — `tab:temp_dispersion_supp` reproducibility

**Verdict:** **confirmed still unreproducible.**

`Attention/figs/validation_results.json` (`phase4_multi_model`) contains per-model `grand_mean_r`, `n_passages`, `n_heads`, `head_dim` — but not the `Key-norm CV`, `τ_opt`, `r@τ*`, or `Temp disp (CV)` columns reported in `tab:temp_dispersion_supp`. The JSON's `n_passages = 20` for all five models, but the table caption states "30 passages." No script under `transformer/analysis/` produces output of the shape required to populate the four extra columns.

**Required revision:** either commit the producing script to the repo (e.g. as `transformer/analysis/temp_dispersion_validation.py`) with reproducible seeds and the 30-passage corpus list, or qualify the table as "computed offline; the underlying per-head statistics are available on request" and disclose the passage count discrepancy with `tab:multi_model_supp` (which uses 20 passages from the same JSON).

### M-D-6 — "Σ_i ≈ Ω Σ_j Ω^⊤ emerges from the dynamics"

**Verdict:** **fixed.**

`GL(K)_supplementary.tex:385` now reads: "Under these assumptions the alignment `Σ_i ≈ Ω_{ij} Σ_j Ω_{ij}^⊤` is consistent with the variational dynamics; we do not claim a contractive proof of global uniqueness here." This is the appropriate disclosure given the absence of a Banach/Bures-metric argument. The change converts the overclaim "emerges from dynamics" to "consistent with dynamics," which is what the math supports.

### V-1 — `y_3` definition split between conjecture and CLT table

**Verdict:** **fixed.**

Conjecture clause (ii) at `GL(K)_supplementary.tex:956` now explicitly lists both: "`y_3 = -1` (with `y_3^{(action)} = -2` for the squared norm)." The CLT validation table at lines 999–1002 reports both values consistently. Internal inconsistency closed.

---

## (c) Fresh-Eyes Findings (Not in `VERIFIER_FINAL.md`)

### N-1. **Supp App C §3 falsely claims the actual Killing form on `gl(K)` is positive definite on `sl(K)`** [Major]

**Claim (manuscript):** `GL(K)_supplementary.tex:580–588`:

> "The Killing form of `gl(K)` is `κ(X, Y) = 2K tr(XY) − 2 tr(X) tr(Y)`. In the generator basis, this yields the metric
> [Eq. 584] `g̃_{ab} = 2K tr(T_a^⊤ T_b) − 2 tr(T_a) tr(T_b)`,
> which is positive definite on `sl(K) = ker(tr)` and degenerate on the center `R·I`..."

**Claim kind:** (S) standard — invokes Killing form by name as the natural-gradient metric.

**Standard treatment:** The Killing form of `gl(K)` is `B(X,Y) = 2K tr(XY) − 2 tr(X) tr(Y)`. Its restriction to `sl(K)` is `B|_{sl(K)}(X,Y) = 2K tr(XY)`. This is the standard Cartan-Killing bilinear form on a real semisimple Lie algebra and is **sign-indefinite** on real `sl(K)` for `K ≥ 2` (Knapp, *Lie Groups Beyond an Introduction*, Cor. 1.46; Helgason, *Differential Geometry, Lie Groups, and Symmetric Spaces*, Ch. III). Under the Cartan decomposition `sl(K) = so(K) ⊕ Sym₀(K)` (skew + symmetric traceless), `B` is **negative** on `so(K)` and **positive** on `Sym₀(K)`. The signature is `(dim Sym₀(K), dim so(K)) = ((K²+K−2)/2, K(K−1)/2)`.

**Numerical verification (own check):** On `sl(3)`, `B(skew, skew) = −12` for each standard skew generator and `B(sym, sym) = +12` for each symmetric traceless generator. The Killing form is **not** positive definite.

**Problem:** Three things are conflated:
1. Eq. 581 (`κ(X,Y) = 2K tr(XY) − 2 tr(X)tr(Y)`) is the actual Killing form — **no transpose**. Indefinite on `sl(K)`.
2. Eq. 584 (`g̃_{ab} = 2K tr(T_a^⊤ T_b) − 2 tr(T_a) tr(T_b)`) is a **different** bilinear form — it has a transpose. For `T_a ∈ so(K)`, this gives `tr(T_a^⊤ T_a) = −tr(T_a²) = ‖T_a‖_F²` (positive), so this form IS positive definite (it is the Cartan-involution-modified form `−B(X, θ(Y))` with `θ(X) = −X^⊤`).
3. The prose at line 588 says "this yields the metric [Eq. 584]" and "which is positive definite on `sl(K)`" — but the "this" syntactically refers to the Killing form `κ` of line 581, and the displayed Eq. 584 has a silent transpose insertion that converts the Killing form into a different bilinear form.

The main text `GL(K)_attention.tex:1632` already states the correct treatment: "for `φ_i ∈ gl(K) = sl(K) ⊕ R·I` the Killing form is degenerate on the central direction, so we use the Cartan-involution-modified bilinear form `⟨X,Y⟩ = −½ tr(X θ(Y))` with `θ(X) = −X^⊤`, which is positive-definite on the full algebra; see Supplementary Appendix C."

**Required revision:** rewrite App C §3 paragraph "3. Killing form natural gradient" to:
- State the Killing form `B(X,Y) = 2K tr(XY) − 2 tr(X)tr(Y)`, note it is indefinite on `sl(K)` and degenerate on `R·I`;
- State separately that the **metric used in this paper** is the Cartan-involution-modified form `g̃(X,Y) = −½ B(X, θ(Y))` (equivalently `2K tr(X^⊤ Y) − 2 tr(X) tr(Y)`), positive definite on `gl(K)`;
- Cite Knapp or Hall for the Cartan involution / compact-form construction.

This is stronger than the prior M-E-4 finding ("mislabels a Cartan-involution-modified form as the Killing form") because it adds the false PD claim about the actual Killing form.

### N-2. **`Q_a` symbol overloaded between right- and left-trivialization in App C vs App D** [Major]

**Claim (manuscript):**
- App C `GL(K)_supplementary.tex:537`: "`Q_a^{(i)} ≡ dexp_{φ_i}(T_a)` is the right-trivialised differential of the exponential map at `φ_i` in the direction of generator `T_a`, defined by the identity `D_{φ_i}(exp)[T_a] = Q_a^{(i)} exp(φ_i)`."
- App C Eq. 453 (line 453, the SO(3) closed form): `dexp_φ(T_a) = T_a + c_1(θ)[φ, T_a] + c_2(θ)[φ, [φ, T_a]]`.
- App D Eq. 648 (line 648): `∂/∂φ^a exp(X) = exp(X) · Q_a` (note the order: `exp(X)` is on the LEFT, `Q_a` on the RIGHT — left-trivialization).
- App D Eq. 652 (line 652): `Q_a = T_a − c_1(θ) ad_X(T_a) + c_2(θ) ad_X²(T_a)` (note the **minus** sign on `c_1`).

**Claim kind:** (S) standard.

**Standard treatment:** The right-trivialized `dexp` and the left-trivialized `dexp` differ by an `Ad` factor. Concretely:
- Right-trivialized: `∂_φ exp(X)|_{X=φ}[T_a] = R_a · exp(φ)` with `R_a = T_a + c_1 ad_φ(T_a) + c_2 ad_φ²(T_a)` (+, +).
- Left-trivialized: `∂_φ exp(X)|_{X=φ}[T_a] = exp(φ) · L_a` with `L_a = T_a − c_1 ad_φ(T_a) + c_2 ad_φ²(T_a)` (−, +).

Both formulas are standard (Hall, *Lie Groups, Lie Algebras and Representations*, Ch. 2; Helgason, Ch. II §1).

**Numerical verification (own check):** For random skew-symmetric `φ ∈ so(3)`, the numerical right-trivialization matches `(+c_1, +c_2)` and the left-trivialization matches `(−c_1, +c_2)` exactly. Both formulas are correct under their stated convention.

**Problem:** the manuscript uses the **same symbol** `Q_a` for **both** conventions:
- In App C the right-trivialized `Q_a^{(i)} = dexp_{φ_i}(T_a)` with formula `+c_1`.
- In App D the left-trivialized `Q_a = D_φ(exp)[T_a] / exp(X)` with formula `−c_1`.

A reader following both appendices encounters the same symbol with apparently contradictory closed forms. The verifier's M-E-3 deferred "sign discrepancy" was correctly suspecting a convention drift, but the bug is **symbol overload**, not a sign error in either formula individually.

**Required revision:** harmonize to one convention throughout, OR rename to disambiguate (e.g. `Q_a^R` for right-trivialized in App C, `Q_a^L` for left-trivialized in App D), AND state the identity `R_a · exp(φ) = exp(φ) · L_a = exp(φ) · Ad_{exp(−φ)}(R_a)` relating the two. The App C `GL(K)_supplementary.tex:537` parenthetical already gestures at this ("An equivalent expression in terms of the Fréchet derivative is...; both conventions appear in the implementation, but the numerical retraction uses the right-trivialised form"), but does not actually carry the disambiguation into App D.

### N-3. **`\ref{thm:vanishing_holonomy}` cross-references still print "Theorem" despite the Lemma downgrade** [Minor — mechanical]

**Claim (manuscript):** `GL(K)_attention.tex:512`, `:1730`, `:2215`, `:2271` all read `Theorem~\ref{thm:vanishing_holonomy}`.

**Standard treatment:** LaTeX `\ref` prints only the counter value, not the environment type. The label `\label{thm:vanishing_holonomy}` was placed inside `\begin{lemma}...\end{lemma}` at line 640–650, so `\ref{...}` correctly resolves to the lemma's number, but the hard-coded prefix word "Theorem" before `~\ref{...}` is still wrong.

**Problem:** the PDF will read "Theorem N" at all four sites where the labeled object is a Lemma. This is the same class of issue that `cleveref` was designed to prevent (`\cref` auto-detects type), but the manuscript uses raw `\ref`.

**Required revision (mechanical):** in `GL(K)_attention.tex` replace all four occurrences:

```
Theorem~\ref{thm:vanishing_holonomy}  →  Lemma~\ref{thm:vanishing_holonomy}
```

at lines 512, 1730, 2215, 2271. Optionally also rename the label key from `thm:vanishing_holonomy` to `lem:cocycle_identity` for hygiene (but this is a bigger refactor — the four `\ref` keys would also need updating).

### N-4. **RoPE "abelianness collapses both forms to `exp(φ_j − φ_i)`" — mathematically wrong** [Major — regression]

**Claim (manuscript):** `GL(K)_attention.tex:1847`:

> "The opposite-sign placement of `φ^{(pos)}` relative to the general definition `Ω_{ij} = exp(φ_i)exp(−φ_j)` is harmless on this subgroup, where the factors commute and abelianness collapses both forms to `exp(φ_j − φ_i)`."

**Claim kind:** (R) reduction — claims the two RoPE/gauge forms collapse to the same expression under abelianness.

**Standard treatment:** On an abelian group, `exp(A) exp(B) = exp(A + B)`. Applied to the two forms:
- Our gauge transport: `exp(φ_i) exp(−φ_j) = exp(φ_i − φ_j)`.
- RoPE form: `exp(−φ_i) exp(φ_j) = exp(φ_j − φ_i)`.

These differ by the sign of the exponent argument. For an orthogonal rotation `R(θ) = exp(θJ)` with `J^⊤ = −J`, `R(−θ) = R(θ)^⊤ = R(θ)^{−1}`. So `exp(φ_i − φ_j) = R(θ_i − θ_j) = R(θ_j − θ_i)^⊤`.

**Numerical verification (own check):** with `φ_i = 0.3 J`, `φ_j = 0.7 J`:
- `exp(φ_i)exp(−φ_j) = [[ 0.92106, +0.38942],[−0.38942, 0.92106]]` = `R(−0.4)`.
- `exp(−φ_i)exp(φ_j) = [[ 0.92106, −0.38942],[+0.38942, 0.92106]]` = `R(+0.4)`.
- These are **transposes** (inverses) of each other; they are **not equal**.

**Problem:** the prose statement "abelianness collapses both forms to `exp(φ_j − φ_i)`" is **mathematically false**. Abelianness gives **two distinct** forms `exp(φ_i − φ_j)` and `exp(φ_j − φ_i)` which are inverses, not equal. The defensible weaker statement is that the two forms differ by a transpose, and that this transpose can be absorbed into the asymmetric `W_Q ≠ W_K` convention (i.e. swapping `M ↔ M^⊤` in the logit kernel `Q_i^⊤ M K_j` corresponds to swapping the roles of `Q` and `K`).

This appears to be a regression introduced by the 2026-05-18 patch addressing M-B-4 ("disclose abelianness of `SO(2)^{d_k/2}`"). The fix overshot: instead of disclosing abelianness as the **structural reason** the two forms are compatible (transpose-equivalent up to convention), it asserted incorrectly that abelianness makes them identical.

**Required revision:** rewrite line 1847 to:

> "The opposite-sign placement of `φ^{(pos)}` relative to the general definition `Ω_{ij} = exp(φ_i)exp(−φ_j)` differs by a transpose: in the abelian setting `exp(φ_i)exp(−φ_j) = exp(φ_i − φ_j) = R(θ_i − θ_j)` while `exp(−φ_i)exp(φ_j) = R(θ_j − θ_i) = R(θ_i − θ_j)^⊤`. The transpose can be absorbed into the convention for `W_Q` vs `W_K` in the logit kernel `Q_i^⊤ M K_j`, since `Q_i^⊤ M K_j = K_j^⊤ M^⊤ Q_i` swaps roles symmetrically."

### N-5. **"Ψ is the inverse Bernoulli function" — name is non-standard / misleading** [Minor]

**Claim (manuscript):** `GL(K)_supplementary.tex:594–597`:

> `d\exp_\phi(T_a) = \exp(\phi) \cdot \Psi(\mathrm{ad}_\phi)(T_a), \qquad \Psi(z) = \frac{e^z - 1}{z} = \sum_{k=0}^{\infty} \frac{z^k}{(k+1)!},`
> "where `Ψ` is the inverse Bernoulli function."

**Claim kind:** (S) — names a standard special function.

**Standard treatment:** The "Bernoulli generating function" in the Lie-theory and dexp context is `B(z) = z/(e^z − 1) = 1 − z/2 + Σ_{k≥1} (B_{2k}/(2k)!) z^{2k}`, whose Taylor coefficients are (up to factorials) the Bernoulli numbers (Hall, Ch. 2; Helgason; standard reference: NIST DLMF §24.2). The function `Ψ(z) = (e^z − 1)/z = Σ_{k≥0} z^k/(k+1)!` is the **multiplicative inverse** `1/B(z)` of the Bernoulli function. Its Taylor coefficients are the simple reciprocal factorials `1/(k+1)!`, **not Bernoulli numbers**.

**Problem:** calling `Ψ(z) = (e^z − 1)/z` "the inverse Bernoulli function" is at best ambiguous (functional inverse? multiplicative inverse?) and at worst suggests Ψ generates Bernoulli numbers, which it does not. The function `Ψ` appears in `dexp = (e^{ad} − 1)/ad`, and `B = ad/(e^{ad} − 1)` appears in `dexp^{−1}` — the inverse-of-`dexp` relationship is what gives `B` its standard role (e.g. in the Magnus expansion).

**Required revision:** rewrite line 597 to either:
- "where `Ψ(z) = (e^z − 1)/z` is the generating series for `dexp` (its multiplicative inverse `z/(e^z − 1)` is the Bernoulli generating function appearing in `dexp^{−1}` and the Magnus expansion; see Hall, *Lie Groups, Lie Algebras and Representations*, Ch. 2 or NIST DLMF §24.2)";
- or drop the name "inverse Bernoulli function" entirely and just call it "the dexp generating series."

### N-6. **`tab:temp_dispersion_supp` reports 30 passages but the on-disk JSON contains 20** [Minor — empirical / reproducibility]

**Claim (manuscript):** `GL(K)_supplementary.tex:855`:

> "Per-head temperature dispersion diagnostic across five architectures (`d_head = 64`, **30 passages**). Temperature dispersion is the strongest predictor of the cross-model correlation deficit."

**Problem:** `Attention/figs/validation_results.json` reports `n_passages = 20` for all five `phase4_multi_model` entries. The companion `tab:multi_model_supp` in `GL(K)_supplementary.tex:830` is consistent with this 20-passage corpus ("Multi-model validation at `τ = 19.0` across **20 passages**"). The `tab:temp_dispersion_supp` 30-passage figure is inconsistent with both the JSON artifact and the upstream table.

**Required revision:** if the 30-passage analysis was a separate run, either commit its output (CSV with `Key-norm CV`, `τ_opt`, `r@τ*`, `r@19`, `Temp disp (CV)` per model) to `Attention/figs/` and reference it; or rerun the per-head dispersion analysis on the 20-passage corpus and update the table caption. This finding is upstream of M-F-4 (table reproducibility) — even fixing the script won't reconcile the table to the existing artifact unless the passage count is harmonized.

### N-7. **Polar-decomposition argument for `Ω_{ij}` surjectivity onto `GL⁺(K)` could use a one-line citation** [Minor]

**Claim (manuscript):** `GL(K)_attention.tex:2058` and `GL(K)_supplementary.tex:552`:

> "the pairwise transport `Ω_{ij} = exp(φ_i)exp(−φ_j)` is a free product of two exponentials, which does cover all of `GL⁺(K)`: any `A ∈ GL⁺(K)` admits a polar decomposition `A = PO` with `P` symmetric positive-definite and `O ∈ SO(K)`, both of which are single exponentials (`P = exp(log P)`, `O = exp(S)` for some `S ∈ so(K)`)."

**Claim kind:** (R) reduction — surjectivity of `(φ_i, φ_j) ↦ exp(φ_i)exp(−φ_j)` onto `GL⁺(K)`.

**Standard treatment:** Polar decomposition of `GL⁺(K)` matrices is standard (Horn & Johnson, *Topics in Matrix Analysis*; Bhatia, *Positive Definite Matrices*). The non-trivial steps the argument hinges on are:
1. `exp: Sym(K) → Sym⁺(K)` is bijective (standard; matrix log on Sym⁺(K)).
2. `exp: so(K) → SO(K)` is surjective (standard for connected compact groups).
3. The composition `exp(P-direction) · exp(so-direction)` then covers `Sym⁺(K) · SO(K) = GL⁺(K)`.

**Problem:** the argument as stated is a **construction** showing surjectivity (given `A`, find `(φ_i, φ_j)`), but the prose lets the reader infer that the **map** `(φ_i, φ_j) ↦ exp(φ_i)exp(−φ_j)` has the polar-decomposition structure. The construction works by choosing `φ_i ∈ Sym(K)` and `φ_j ∈ so(K)` (or vice versa), which is a restriction on the input pair, not a property of the map. As `(φ_i, φ_j)` ranges over `gl(K)²` freely, most pairs do **not** sit in `(Sym(K), so(K))`.

**Required revision:** the surjectivity statement is correct as an existential claim; clarify the prose to:

> "Surjectivity holds in the existential sense: for every `A ∈ GL⁺(K)` there exists at least one pair `(φ_i, φ_j) ∈ gl(K)²` with `Ω_{ij} = A`. By polar decomposition, choose `φ_i ∈ Sym(K)` with `exp(φ_i) = P` (where `A = PO`) and `φ_j ∈ so(K)` with `exp(−φ_j) = O`. The map is not injective — different pairs `(φ_i, φ_j)` can produce the same `Ω`, as the orbit `(φ_i + a I, φ_j + a I)` for `a ∈ R` is a trivial example."

Optional one-line citation: Bhatia, *Positive Definite Matrices*, §1.5; or Higham, *Functions of Matrices*, §6.4 on matrix logarithm of SPD.

---

## Citation Verification (since the prior pass)

Citation hygiene is dramatically improved from the prior round. Spot-checks against the bib file confirm `xiao2024efficient`, `Gallier2020`, `Higham2008`, `rossmann2002lie`, `kadanoff1966scaling`, `garciaperez2018multiscale` all resolve. No new citation-key mismatches detected.

One **content** observation: `garciaperez2018multiscale` is cited at supplementary line 128 for "the gauge-theoretic analogue of renormalization group flow in statistical field theory" alongside `anderson1984basic` and `wilson1974renormalization`. The García-Pérez et al. paper "Multiscale unfolding of real networks by geometric renormalization" (Nature Physics 14:583, 2018) is about network geometric RG and is reasonable in this slot — verified as a real paper, contents match the context. ✓

`kadanoff1966scaling` ("Scaling laws for Ising models near Tc," Physics 2, 1966) is the canonical block-spin / coarse-graining citation and is appropriate. ✓

---

## Manuscript ↔ Code Consistency

Not re-checked in this pass (prior verifier covered codebase touch-points; the fresh findings here are mathematical/notational and do not implicate code paths).

---

## Open Questions for Author

1. **N-4 (RoPE abelian claim).** Does the author want to preserve the "harmless" framing (in which case a sign/convention disclosure is needed), or rewrite to admit the transpose distinction and absorb it into the asymmetric `W_Q ≠ W_K` machinery?

2. **N-1 (Killing form in App C).** Does the author want App C §3 to align with the main text §4.3 Cartan-modified treatment? If so, the App C `Killing form natural gradient` paragraph should be rewritten to call the metric what it is ("Cartan-involution-modified Killing form" or equivalently the "compact-form Killing"), and the indefinite genuine Killing form should be relegated to a remark.

3. **N-6 (passage count mismatch in temp dispersion table).** Was the 30-passage analysis a separate offline run, or is the "30" in `tab:temp_dispersion_supp:855` a typo for `20`?

---

## Overall Verdict

Compared to the state captured in `VERIFIER_FINAL.md`, the manuscripts are in dramatically better shape: 13/15 prior findings are mechanically fixed, all citation-key mismatches resolved, banned spacing macros eliminated, the App H real-analyticity hypothesis restored, the RG conjecture internal contradiction resolved by the `g_1^{(orig)}`/`g_1^{(emer)}` split, the Hessian formula corrected, and the `dexp` numerical correctness restored. The two prior partial-verdicts (M-D-6 and V-1) are now properly disclosed/fixed.

The fresh-eyes pass surfaces **two major** new issues (N-1 false positive-definiteness of the actual Killing form, N-4 mathematically-wrong abelian-collapse claim introduced by a 2026-05-18 patch), **one major** notation issue (N-2 `Q_a` symbol overload between Apps C and D), and **four minor** items (N-3 stale "Theorem~\ref" cross-references after the Lemma downgrade, N-5 "inverse Bernoulli" naming, N-6 30-vs-20 passage mismatch, N-7 polar-decomposition surjectivity wording). N-1, N-2, N-3, N-4 are all mechanically fixable in one focused pass.

**Recommendation:** another single mechanical patch pass over the seven items above brings the manuscript to a state where the only remaining open questions are author-decision items (M-F-2 graph-RG sign disclosure framing, M-F-4 / N-6 temp-dispersion script commit).
