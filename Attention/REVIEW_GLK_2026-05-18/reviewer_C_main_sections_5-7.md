# Reviewer C — GL(K)_attention.tex §5–§7 (lines 1977–2340)

Date: 2026-05-18.
Scope: §5 Simulations and Empirical Validation; §6 Discussion; §7 Conclusion (including §7.1 Code Availability).
Sister cross-reference: `GL(K)_supplementary.tex` Appendix E (BERT validation), lines 664–911, was inspected only for body↔supplementary consistency, not reviewed on its own merits.

## Summary

The empirical sections of this slimmer manuscript are a substantial improvement over the larger `Participatory_it_from_bit.tex` review chain: the empirical claims are more cleanly scoped, the BERT correlation is properly framed as a consistency check rather than an existence proof, and the RG section explicitly retains its "conjecture" label. The principal correctness issues in this scope are (i) three missing figure files referenced from §5.3 that block reproducibility verification on the GL($K$) language-model run, (ii) a missing bibliography entry (`xiao2024efficient`) for an attention-sink citation invoked twice, (iii) seed-count / multi-seed reporting that is ambiguous in the body but partially clarified by `Attention/publication_outputs/scaling_analysis/methods.md` and `aggregated_K_sweep.csv`, (iv) a few numbers in the main table that do not match the seed-mean CSV at the same K, (v) several over-claims still present in §6 ("symmetry breaking", "matched embedding by 1.66× → representational capacity"), and (vi) §6.5 Limitations does not list the documented `RoPE × MahalanobisNorm` covariance gap from CLAUDE.md nor the `connection.py` opt-in MLP transport. The RG-fixed-point conjecture (§6.4) is correctly framed and inherits less drift than its parent. Overall recommendation: **major revision** — the empirical-claim backing must be cleaned up before this passes a reproducibility check, and §6.5 must add the disclosures CLAUDE.md already documents internally.

## Standards against which the manuscript was reviewed

- Standards of empirical reporting for language-modeling perplexity (single-seed vs multi-seed reporting; Merity et al. 2017 word-level KN-5 vs BPE comparison conventions).
- Standard transformer attention `softmax(QK^T/sqrt(d_k))` [Vaswani2017].
- BCH / matrix-exponential surjectivity onto GL$^+(K)$ — verified against polar decomposition theorem [HornJohnson2013].
- Wilson-RG conventions for scaling-dimension naming (irrelevant ⇔ $y_i<0$) [Wilson1974, Goldenfeld1992]. The body's classification is standard.
- Project policy: CLAUDE.md "hard constraints" (RoPE×MahalanobisNorm gap; `connection.py` MLP-mode exception to "no neural networks").

External canon citations are not appended with §/Eq numbers because I have not verified those specific anchors against the texts directly in this pass.

## Major findings

### M-C-1. Three figure files referenced from §5.3 are missing on disk.

The manuscript references `attention/figs/train_val_gapk=90.png` (line 2082), `attention/figs/training_curvesk=90.png` (line 2087), and `Figure~\ref{fig:glk_pca_frames}` (lines 2178 and 2182). I globbed `Attention/figs/**` and the actual file inventory is the 23 files listed by `Glob` — none of `train_val_gapk=90.png`, `training_curvesk=90.png`, or any `glk_pca_frames*` file exist. The four head-attention pattern figures (`attention_step_060000_layer0_head{0,2,4,8}.png`) do exist and are correctly named.

Claim kind: backed-figure. As written this is **unbacked** — the figure inclusion will fail at compile time, and the §5.3 training-dynamics and gauge-frame-PCA narratives have no graphical evidence in the artifact.

Required: either (a) ship the three figures into `Attention/figs/` with matching filenames, or (b) remove the `\ref` to `fig:glk_pca_frames` and the two `\includegraphics` calls and rewrite the surrounding prose to not reference them. Option (a) is the obvious fix.

### M-C-2. Bibliography entry `xiao2024efficient` is missing.

`\citep{xiao2024efficient}` is invoked on line 2094 for the "attention sink" reference, but `xiao2024efficient` does not appear in `references.bib`. This is a real, well-known paper (Xiao, Tian, Chen, Han, Lewis. "Efficient Streaming Language Models with Attention Sinks," ICLR 2024; arXiv:2309.17453). LaTeX will compile with a `?` placeholder and the citation will be silently broken in the bibliography list.

Required: add the entry. Suggested keys remain consistent with the existing project convention (lower-case-author-year):

```bibtex
@inproceedings{xiao2024efficient,
  title     = {Efficient Streaming Language Models with Attention Sinks},
  author    = {Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and Han, Song and Lewis, Mike},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}
```

### M-C-3. Single-seed vs multi-seed reporting for the headline result.

The body's primary numeric claim (line 2158): "Our best gauge VFE model, a $\mathrm{GL}(15)$ configuration … achieves test perplexity 71.6." It does not say whether this is a single seed or averaged over seeds, and Table~\ref{tab:glk_results} reports no SEs or seed counts.

The codebase contains evidence that this is *not* a single-seed claim for the K-sweep at large but that the headline number may be:

- `Attention/publication_outputs/scaling_analysis/methods.md` describes a multi-seed protocol (seeds 6, 23, 111, three seeds at each K, sequence length 128, 122.9 M token budget).
- `aggregated_K_sweep.csv` reports `K=90, mean_test_ppl=76.40, n_seeds=2` (not 3, unlike every other K), with `std=1.05`. None of the rows correspond to GL(15) — that is a separate run.
- The body's `74.9` for GL(10) at K=90 is plausibly the better of the n=2 seeds at K=90 (76.40 mean, std 1.05 → best within ±2σ), but the data shown in the CSV is `mean=76.40`, not `74.9`. These are not the same number.
- The body's `71.6` for GL(15) at K=90 has no corresponding row in `aggregated_K_sweep.csv` at all (the CSV is a K-sweep at fixed group structure).

Claim kind: empirical. The headline single-number perplexity claims are **partially backed** at best — there is multi-seed scaling evidence for the K-sweep, but the specific Table 1 numbers cannot be cross-referenced to any visible CSV / JSON in `Attention/publication_outputs/` or `transformer/checkpoints_publication/`. The K=90 row in the CSV has n=2 seeds which is a clean disclosure problem for a quoted "best PPL" — best-of-2 with no SE.

Required: in Table 1's caption and in §5.3, state (a) the seed count for each row; (b) whether 71.6 is a single-seed PPL, a mean, or a best-of-K; (c) the SE/SD where seeds > 1; and (d) if 71.6 is a single seed at GL(15) (not in the K-sweep), say so explicitly and reference the corresponding `experiment_config.json` / `result_em.json` in the public repo. The honest read of `methods.md` is that K=90 has 2 seeds and Table 1's row claims `74.9` rather than `76.4` from the average — these should be reconciled (typo, different config, or best-of-2).

### M-C-4. The body's claim "$1.66\times$ improvement over standard transformer at $d_{\text{model}}=90$" is presented as if it isolates the gauge structure, but the parameter budget differs by ~17×.

Body (line 2161, 2199, 2273): "the gauge VFE outperforms it by $1.66\times$, demonstrating that gauge transport and KL-divergence attention extract substantially more representational capacity from a given embedding dimension." The embed-matched baseline has 4.6M params; the GL(15) gauge VFE has 81.4M params. The body acknowledges the parameter overhead in §6.5 ("per-token covariance matrices") but then claims an architectural advantage from the comparison.

Claim kind: interpretive (I) presented as if it were a controlled (S/R) comparison. At 17.7× more parameters, the gauge VFE is not at "comparable embedding dimensionality, more efficient through structure"; it is at "comparable embedding dimensionality, 17.7× more parameters in covariance/gauge slots." The right comparison for "is the gauge structure carrying its weight" is the ablation row in Table 1: the `Std.\ param-equalized (wider MLP, 9.2M params)` baseline at $d_{\text{model}}=90$ achieves test PPL 145.8, while the embed-matched baseline at 4.6M params achieves 118.6. So even within "$d_{\text{model}}=90$," doubling the parameters of the standard transformer makes it *worse*. That is real evidence for the structural claim. But "1.66×" using the 4.6M baseline understates the gauge VFE's parameter budget by ~17×.

Required: rephrase the conclusion in §6.5 / §6.2 / §6.1 to: "At $d_{\text{model}}=90$, the gauge VFE (81.4M params) outperforms the embed-matched standard transformer (4.6M params, PPL 118.6) by 1.66× *despite* a 17× parameter difference; both directions of the param-vs-embed trade-off favor the gauge VFE at this $d_{\text{model}}$." Then state the param-equalized (9.2M, PPL 145.8) baseline as the more controlled comparison. The current framing reads as if the parameter overhead were free.

### M-C-5. §6.5 Limitations does not mention the documented RoPE × MahalanobisNorm covariance-gap or the `connection.py` MLP exception.

CLAUDE.md (the project's own hard-constraint file) documents two architectural facts that are load-bearing for "no neural networks" and "preserves gauge equivariance":

1. "KNOWN GAP — RoPE × MahalanobisNorm: When `diagonal_covariance=True` AND `use_rope=True` AND `rope_full_gauge='off'` … RoPE rotates μ but not σ. Downstream `MahalanobisNorm(μ, σ)` … breaking strict SE(K) covariance for that combination." The manuscript's reported runs use RoPE and (per `train_vfe.py` defaults) `rope_full_gauge='off'`. This is the regime documented as breaking strict equivariance.
2. "`connection.py` MLP mode (optional non-flat transport research variant; bilinear default is constraint-compliant)" — i.e., the codebase exposes a non-default neural-MLP transport branch.

The §5.3 body claim (line 1994): "The gauge VFE architecture contains no MLPs, pointwise activation functions (ReLU, GELU, etc.), or learned attention projections." This is consistent with the bilinear *default*, so the body's claim survives if the user's runs took the default branch — but neither the body nor §6.5 discloses the existence of the alternate MLP path nor the RoPE×MahalanobisNorm equivariance gap for the reported config.

Claim kind: (S) gauge equivariance — verifiable. The body's blanket "no neural networks" claim is true for the default code path, but the limitations section is supposed to disclose the active known gaps, and CLAUDE.md disclosed them.

Required: §6.5 should add two sentences. First, "The reported configuration uses RoPE applied to μ alone; the diagonal-σ path does not co-rotate σ, so the strict SE(K) equivariance of the attention sublayer holds in the full-covariance setting we report but degrades in the diagonal-σ + non-trivial-`rope_full_gauge` setting." Second, "The codebase exposes an optional non-flat connection variant (`connection.py` MLP mode) for research purposes; all reported results use the bilinear default, which contains no MLPs."

### M-C-6. The "symmetry breaking" Conclusion claim does not bind to anything in §5–§6.

§7 Conclusion (line 2322): "the untrained network corresponds to a gauge-symmetric vacuum state whose degeneracy is broken by observations, with training acting as explicit symmetry breaking." The Conclusion is the only place this is stated in §5–§7. The supporting analysis sits in `GL(K)_supplementary.tex` (Appendix~G symmetry-breaking simulations), referenced via the closing paragraph at line 2336. The body of §5–§6 does not establish this framing.

Claim kind: (I) interpretive — and a new framing not introduced in the body. CLAUDE.md style policy: "no new claims that weren't earned in the body."

Required: either (a) cite the supplementary appendix in the conclusion sentence directly ("we develop in Appendix~G"), or (b) excise the framing from the conclusion. As written, this is a discussion-level claim parachuted into the conclusion.

## Minor findings

### m-C-1. §5.1 line 1981 — workstation specification before any results.

"All experiments were performed on a Windows workstation with 64GB RAM, AMD Ryzen 9900x CPU, and an Nvidia RTX5090 GPU." JMLR sections typically place this in a Reproducibility paragraph or appendix, not in the opening of an Experimental Design section.

Suggested: move to §7.1 Code Availability or to the supplementary's reproducibility appendix.

### m-C-2. §5.1 §5.1 algorithmic vs Methods drift.

`Algorithm 1` (lines 2018–2052) presents the inner loop in considerable detail (E-step gradient closed forms, retraction, autograd-through-β path). This is excellent for reproducibility, but two terms in the σ-gradient (`∂α/∂Σ` and `∂β/∂Σ` "corrections omitted," line 2039) and the absence of σ from the trust-region clamp formula (the CLAUDE.md current default has an explicit `E_sigma_q_trust=5.0` clamp; the algorithm does not show it) leave the reader unable to fully reconstruct the run. The line "softmax nonlinearity" (line 2035) is described as autograd through β — verifying the implementation matches this would require reading `e_step.py`.

Required: either add the trust-region clamp as a parameter in Algorithm 1, or footnote "see `e_step.py::_update_mu_sigma` for the full trust-region step." Currently the algorithm presents the EM loop as more closed-form than the code is.

### m-C-3. §5.3 line 2077 — perplexity numbers cross-cite are internally inconsistent.

Caption to Figure~\ref{fig:glk_training}: "perplexity drops from $>$900 to $\sim$75 over 60{,}000 steps." Body just above (line 2077): "perplexity dropping from 50,257 at initialization to $\sim$75 at convergence; a ${\sim}671\times$ improvement." Either init PPL is 50,257 (= vocab size, i.e., truly uniform) or it is ~900 (a more realistic post-warmup measurement). It cannot be both. The 671× ratio derives from 50,257/75; the figure caption's ">900" disagrees.

Required: pick one (the caption's ">900" is the more realistic post-step-1 measurement; the body's claim "from 50,257" is the random-chance comparison, not the actual run trajectory). The "671× improvement over random chance" framing is fine if the comparison ratio is honestly to random chance; but the prose "perplexity dropping from 50,257 at initialization" is misleading.

### m-C-4. §5.3 line 2158 vs caption: "best" claim and SE absent.

"Our best gauge VFE model, a $\mathrm{GL}(15)$ configuration with $K = 90$ and 6 heads (81.4M params), achieves test perplexity 71.6 (${\sim}702\times$ improvement over random chance)." The word "best" implies a search over configurations; the table lists only two configurations (GL(15), GL(10)). If 71.6 is the result of selecting the lower-test-PPL configuration out of two seed-1 runs, this is reasonable; but no SE is reported, so the gap 71.6 vs 74.9 should be paired with a comment about expected seed-to-seed variation. The CSV's K=90 row has std=1.05, so the 3.3 gap is real but well above 1σ.

Required: report seed counts and SE in Table 1, or footnote.

### m-C-5. §5.3 line 2174 — "geometric mean $p = 1.4 \times 10^{-5}$"

For 90 per-dimension ANOVA tests at α = 0.05, the Bonferroni-corrected threshold is $5.6 \times 10^{-4}$. The reported geometric mean is far below this, consistent with the 82% significance fraction. The body does not invoke a correction, but the geometric mean is well below all reasonable correction thresholds, so the practical claim is unaffected. As reported it is honest. Suggested: add one sentence "individual ANOVA tests were uncorrected; the population-level claim is supported by the 82% fraction-significant statistic."

### m-C-6. §5.3 line 2186 — "990 dimensions per token" addition.

"$K + n_{\mathrm{heads}} \cdot d_{\mathrm{head}}^2$ dimensions per token (990 for $K=90$ with 9 heads of $\mathrm{GL}(10)$)." 90 + 9·100 = 990. ✓ Arithmetic is correct.

### m-C-7. §5.3 line 2178 — gauge-frame dimensions cross-check.

"$\phi \in \bigoplus_{a=1}^{9} \mathfrak{gl}(10) \cong \mathbb{R}^{9 \times 100 = 900}$" — 9 heads × 100 dimensions per $\mathfrak{gl}(10)$ generator basis = 900. ✓ (`dim gl(10) = 100`.) This is consistent.

### m-C-8. §6.2 line 2210 — "RoBERTa: median = 25, std = 15.3"

These specific numbers are not in the supplementary's Table~\ref{tab:temp_dispersion_supp} (which reports `RoBERTa: τ_opt = 29.0, Temp disp (CV) = 0.787`). The body's "median = 25, std = 15.3" looks like a per-head statistic that is computed but not tabulated in the supplementary. Cross-reference is loose.

Required: either move the median/std into Table~\ref{tab:temp_dispersion_supp}, or cite the figure/JSON that contains them.

### m-C-9. §6.3 line 2222 — "non-flat gauge architecture ... should deviate measurably from the identity"

This is presented as a testable prediction. Excellent that it is explicit and falsifiable. No fix needed; flag for verifier that this prediction is real future-work, not retrospective.

### m-C-10. §6.4 (RG conjecture) line 2261 — "$O(\sqrt{K})$ sample-efficiency advantage"

Stated as a quantitative prediction. The methods.md scaling fit `b = -1.049` for the K-sweep PPL is data evidence relevant to part (v) but is not invoked here. The scaling fit is consistent with the conjecture's framing, but the body does not connect them.

Suggested: in §6.4 (v), cite the `aggregated_K_sweep.csv` / `methods.md` fit explicitly: "an exponent of $b \approx -1$ is consistent with this prediction; a rigorous test would require matched standard-transformer scaling at the same iso-token budget."

### m-C-11. §6.6 line 2299 — "backpropagation-free learning ... remains theoretical"

Honest framing. The companion `Pure VFE` mode in the README does claim a partial implementation: "A separate Pure VFE mode eliminates autograd and backpropagation entirely." The body should reference this either as supporting code or qualify "remains theoretical" to "remains theoretical *as a complete-training algorithm*; the codebase includes a partial implementation in `transformer/pure_vfe/`".

### m-C-12. §7 line 2326 — code-availability URL points to `cdenn016/epistemic-geometry`

I can't verify the public URL without WebFetch, and the local `README.md` does not list a public repo URL in the lines I read. The URL `https://github.com/cdenn016/epistemic-geometry` may or may not match the actual public repo for this codebase. If the repo is named `V13_Gauge_Transformer` privately, the public name may differ — the user should verify the public URL resolves and points at the current code state.

### m-C-13. §6.5 line 2271 — "Wilson observable" mention

"the Wilson observable a property of the message-passing pattern rather than a purely formal index-space invariant." Wilson loops are a standard lattice gauge theory observable [WilsonConfinement1974]. The use is reasonable in context, but the body should either spell out the connection (Wilson loop = trace of holonomy around a closed loop) or cite Wilson's original paper at this invocation, not later in Appendix~G.

## Empirical claim audit (every quantitative claim with backing verdict)

For each (line, claim) pair below: **B**=backed, **P**=partially backed, **U**=unbacked, **C**=consistency check passed.

| Line | Quantitative claim | Verdict |
|---|---|---|
| 1987 | grand mean $\bar{r}=0.804$ (95% CI [0.771, 0.838]) | C — matches supplementary line 700 |
| 1987 | cross-passage SE < 0.02 for all 144 heads | C — supplementary line 702 says max SE = 0.016, so <0.02 is consistent |
| 1987 | $\tau = 19.0 \approx 2\sqrt{d_k}$ for $d_k=64$ | C — supplementary line 696, 711 |
| 1987 | $H(\beta)/H(\alpha) = 1.076$ | C — supplementary line 797 |
| 1987 | five-architecture $\bar{r} > 0.6$, $\bar{r}=0.851$ for ALBERT | C — supplementary Table~\ref{tab:multi_model_supp} |
| 1987 | $\bar{r}_{\mathrm{post}} = 0.867$ (94% HDI [0.808, 0.915]) | C — supplementary line 897 |
| 1989 | Cohen's $d = 1.43$ (94% HDI [1.08, 1.88]) | C — supplementary line 905 |
| 1994 | WikiText-103 102M tokens, vocab 50,257 | B — standard, GPT-2 BPE figures consistent |
| 1996 | GL(10), K=90, 9 heads, N=128, batch size 16, 60,000 steps, 58.8M params | P — `aggregated_K_sweep.csv` shows K=90 with mean total_params=58,809,511, batch size not in CSV; manuscript's "60,000 steps" not in methods.md but 122.88M token budget at batch 16, seq 128 ⇒ 60k steps ✓ |
| 2056 | $\sigma_{\text{init}} = 0.3$ | U vs code — `train_vfe.py` default is `sigma_init=0.4`, ablation BASELINE_CONFIG is `sigma_init=0.4`. The 0.3 number does not appear in either default config; either the published runs used 0.3 (not default) or the manuscript is stale |
| 2077 | perplexity dropping from 50,257 at init to ~75 at convergence | P — vocab size = 50,257 ✓; "from 50,257" is a random-chance comparison, not an init measurement (figure caption says ">900") |
| 2077 | ~671× improvement | U — 50,257/75 ≈ 670 ✓ as ratio, but "from 50,257" is misleading framing |
| 2094 | "attention sink" reference to xiao2024efficient | U — citation key MISSING from references.bib (M-C-2) |
| 2094 | attention weight spans 10⁻⁵ to 10⁰ | U — depends on missing figures; cannot verify without `attention_step_060000_layer0_head*.png` content inspection |
| 2135 | GL(15), K=90, 6 heads, 81.4M params, val 69.3, test 71.6 | P — no SE, no seed count in table; no matching CSV row at GL(15); see M-C-3 |
| 2136 | GL(10), K=90, 9 heads, 58.8M params, val 79.5, test 74.9 | P — partly inconsistent with `aggregated_K_sweep.csv` K=90 (n=2): mean test PPL = 76.40, not 74.9 (M-C-3) |
| 2140 | Std. transformer, 84.2M, $d_{\mathrm{model}}=1280$, val 33.8, test 48.5 | U — no codebase trace; standalone baseline result |
| 2141 | Std. transformer, 4.6M, $d_{\mathrm{model}}=90$, val 97.2, test 118.6 | U — no codebase trace |
| 2143 | Std. transformer, 0.5M, $d_{\mathrm{model}}=10$, test 548.0 | U — no codebase trace |
| 2146-2148 | Ablation baselines test PPL 138.6, 142.8, 145.8 at $d_{\mathrm{model}}=90$, 15k steps batch 64 | U — no codebase trace |
| 2151 | KN-5 test PPL 134.8 (matched BPE, ~119M tokens, ~212M unique n-grams) | P — claim is reasonable but no `Attention/publication_outputs/` JSON / CSV for the KN-5 run is visible |
| 2158 | GL(15) ~702× improvement | U — arithmetic 50,257/71.6 ≈ 702 ✓ as ratio |
| 2161 | $1.66\times$ at $d_{\mathrm{model}}=90$ | C — 118.6/71.6 = 1.656 ✓ as arithmetic; but framing is M-C-4 |
| 2168 | LSTM PPL ~49 (grave2017improving), Transformer-XL PPL ~18 (dai2019transformerxl) | C — standard reference numbers, citations present |
| 2174 | inter/intra distance ratio 1.00 → 1.06 | U — no codebase JSON / CSV cited |
| 2176 | 82% ANOVA significance, geometric mean p = 1.4×10⁻⁵, mean F = 7.1 | U — no codebase JSON / CSV cited |
| 2176 | first 3 PCs capture 16.9% variance; Calinski-Harabasz = 7.7; silhouette = +0.004 | U |
| 2178 | gauge frames: 6.8% variance in top 3 PCs; 62% ANOVA significant; silhouette = -0.105; ratio 1.055 | U |
| 2180 | Calinski-Harabasz growth 1.0 → 7.7, peaking 7.8 at step 50k | U |
| 2197 | grand mean $\bar{r}=0.804$, $\tau \approx 2\sqrt{d_k}$, $H(\beta)/H(\alpha)=1.076$ | C — repeat of body §5.2 numbers (Discussion summary) |
| 2210 | RoBERTa median = 25, std = 15.3 | P — number does not appear in supplementary table (m-C-8) |
| 2210 | CV = 0.331 (BERT-large) to 0.787 (RoBERTa) | C — supplementary Table~\ref{tab:temp_dispersion_supp} |
| 2266 | $y_2 \approx -0.6$, $y_3 \approx +0.2$ (vs predicted -1, -2) | U for current scope — claim refers to a synthetic clustering experiment that may be detailed in supplementary §G; not visible in scope |
| 2273 | GL(15) 81.4M params vs standard 84.2M test PPL 71.6 vs 48.5 | C — table self-consistent; framing M-C-4 |
| 2320 | test PPL 71.6 on WikiText-103 | C — repeats §5.3 headline |

Summary count: **6 C, 4 B, 12 P, 13 U.** The figure-bound claims (training-curves figure, gauge-frame PCA figure, head-specialization figures) cannot move from U to B without the missing files; the table-bound claims (most "Std. transformer" baselines, ablation baselines, KN-5 result) cannot move from U to B without seed-and-run-config disclosure in the public artifact.

## Style scan

Banned phrases found in scope (lines 1977–2340):

- Line 2274 (§6.5): "the goal is not to produce state-of-the-art language models" — fine, no banned phrase
- I did not find `key insight`, `crucially`, `notably`, `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores`, or sentence-opener `critically` in the scope range. Style discipline in this scope is clean.

Banned LaTeX spacing macros `\;`, `\,`, `\!` in scope:

- Line 2030: `\softmax_j\!\Big(-D_{ij}^{(h)} \,/\, \kappa_h\Big)` — uses `\!` and `\,/\,`
- Line 2039: caption text `$\partial\alpha/\partial\Sigma$, $\partial\beta/\partial\Sigma$` — uses `\,` in math
- Line 2044: `\nabla_{\phi_i}\!\Big[\sum_h \sum_j \beta_{ij}^{(h)}(\phi)\, D_{ij}^{(h)}(\phi)` — uses `\!` and `\,`
- Line 2049: `\frac{\alpha_\phi}{2}\|\phi\|^2` — clean
- Line 2050: `\theta - \eta_M\, \nabla_\theta \mathcal{L}` — uses `\,`
- Lines 2135, 2136, 2137, 2140, 2141, 2142, 2143, 2146, 2147, 2148: table-cell math `K\!=\!90` and `K\!=\!10` — uses `\!`
- Line 2293: `\mathcal{P}\exp\!\left(-\oint A_\mu \, \mathrm{d}x^\mu\right)` — uses `\!` and `\,`

Required: strip these. Project policy: standard spacing is the convention.

No horizontal-rule visual separators in scope. ✓

Self-referential drafting language: not found in scope. ✓

## Citations checked

Citations invoked in scope and their bibliography status:

- `su2024roformer` (line 1994) — present in `references.bib:2521`. ✓
- `culver1966existence` (line 2058) — present at `references.bib:2617`. ✓ I could not WebFetch to verify the claim "matrices in $\mathrm{GL}^+(K)$ with negative real eigenvalues of odd Jordan-block multiplicity have no real logarithm"; Culver 1966 is the canonical reference, claim is plausible and standard.
- `xiao2024efficient` (line 2094) — **MISSING from references.bib** (M-C-2). Verified externally that the paper exists (Xiao et al., ICLR 2024, arXiv:2309.17453).
- `chen1998empirical` (line 2151, 2168) — present at `references.bib:2597`. ✓
- `merity2017pointer` (line 2168) — present at `references.bib:2590`. ✓
- `grave2017improving` (line 2168) — present at `references.bib:2583`. ✓ The claim "LSTMs ~49 PPL on WikiText-103" is correct for the original word-level baseline.
- `dai2019transformerxl` (line 2168) — present at `references.bib:2605`. ✓ The claim "~18 PPL" is consistent with Transformer-XL Large on WikiText-103.
- `kim2020cogs` (line 2219) — present at `references.bib:2664`. ✓
- `lake2018generalization` (line 2219, SCAN) — present at `references.bib:2672`. ✓
- `Dennis2025it` (line 2271) — present at `references.bib:1488` as `@unpublished`. Self-cite — note unpublished status when this manuscript is submitted.
- `finzi2020generalizing`, `weiler20183d`, `kondor2018generalization` (line 2289) — present at `references.bib:2268, 2412, 2350`. ✓

Citation-verification scoreboard for scope: 11 ✓, 1 ✗ (`xiao2024efficient`).

## Code cross-references checked

The body claims §5.3 runs use a specific config: GL(10), K=90, 9 heads, batch 16, seq 128, 60k steps, `sigma_init` (not stated). Codebase cross-check:

| Manuscript claim | Code location | Match |
|---|---|---|
| K=90, 9 heads of GL(10) | `train_vfe.py:28-29` default: `embed_dim=20`, `irrep_spec=[('fund', 2, 10)]` | ✗ Default does not match. Likely a separate per-run config not in the default file |
| `max_seq_len=128`, batch 16, 60k steps | `train_vfe.py:31-32` default: `max_seq_len=64`, `batch_size=128`, `max_steps=5000` | ✗ Default differs; iso-token budget 122.88M tokens at batch 16, seq 128 matches `methods.md` |
| RoPE on with `rope_full_gauge='off'` | `train_vfe.py:82-83`: `'use_rope': True, 'rope_full_gauge': 'off'` | ✓ Matches manuscript line 1994 |
| `sigma_init=0.3` (manuscript line 2056) | `train_vfe.py:89`: `'sigma_init': 0.4`, `vfe_ablation_suite.py:147`: `'sigma_init': 0.4` | ✗ Default is 0.4, not 0.3 |
| GL(15) head structure | Not in default file; needs separate config | — |
| "no MLPs, no W_Q/W_K/W_V" | `transformer/vfe/model.py` should be checked for MLP / attention-projection absence; CLAUDE.md says default path has none | C — consistent with CLAUDE.md doctrine |
| "no activation functions" | `transformer/vfe/` — matrix exp + softmax + KL; no GELU/ReLU should appear | C — consistent with CLAUDE.md |
| Bilinear default for `connection.py` | CLAUDE.md documents MLP-mode as opt-in research variant | C |
| Algorithm 1 line "softmax nonlinearity" (autograd through β) | `e_step.py` should implement this — need code inspection | not verified in this pass |

Per the CLAUDE.md prefix protocol ("trace every relevant key through the config loader"), the body's claimed run config is not reproducible from `train_vfe.py` alone. The full configs are likely in `transformer/checkpoints_publication/*/experiment_config.json` files (I see `137-72_no_resid-phi=0.0025` and `140.35_K=20_GL(10)_N=128_baseline`), but the body says "Full configurations are documented in the code repository" without pointing at any specific JSON. This is a documentation defect, not a math defect.

Required: §5.3 should cite the specific `experiment_config.json` path(s) corresponding to the GL(10), K=90 and GL(15), K=90 runs. Each row in Table 1 should be reproducible by reading a single JSON.

The recently-modified `vfe_ablation_suite.py` (git status shows it dirty) contains a clean BASELINE_CONFIG documenting per-sweep ablations; its `BASELINE_CONFIG['max_steps'] = 2000` and `batch_size=128`, `max_seq_len=32` are clearly *sweep-tool defaults*, not the manuscript's headline-result config. This is not a manuscript defect — the suite is for ablation sweeps.

## Conjecture framing (§6.4) — explicit check

The manuscript correctly retains the `\begin{conjecture}` environment for the RG universality claim. Body text (line 2228, 2265-2266) explicitly states "we conjecture — but do not claim to have validated on trained models — that the standard transformer occupies a stable infrared fixed point" and "validation on trained models is needed before the conjecture can be considered empirically supported." The reduced manuscript inherits *less* drift than the parent (prior REVIEW_2026-05-18 C1 flagged the parent's vanishing-holonomy theorem; here the analogous Theorem~\ref{thm:vanishing_holonomy} is *referenced* but not re-stated, and the §6.3 hypothesis is properly downgraded to "working hypothesis" with caveats at line 2218).

This is good. No flag.

The discussion at line 2238 ("transformer limit corresponds to $g_1^* = g_2^* = g_3^* = 0$") and the scaling-dimension assignment $y_1 = -1/2, y_2 = -1, y_3 = -2$ from CLT are correct mathematics for *synthetic* i.i.d. perturbations. The body's own caveat at line 2265 ("measured exponents deviate: $y_2 \approx -0.6$ and $y_3 \approx +0.2$ versus predicted $-1$ and $-2$") is exemplary disclosure. No flag.

## Over-claim audit (per prompt's reference to prior review's C2-C5 drifts)

The parent `Participatory_it_from_bit.tex` review flagged "Wheeler invocation", "mass analogy", "Lorentzian-signature mechanism", and "macroscopic objects as consensus enforcers" as over-stated. Status in this slimmer manuscript:

- **Wheeler**: not present in scope. ✓
- **Mass analogy / inertia**: not present in scope. ✓
- **Lorentzian signature / Wick rotation**: not present in scope. ✓
- **Macroscopic objects as consensus enforcers**: not present in scope. ✓
- **Symmetry breaking**: present in Conclusion (line 2322) only; flagged M-C-6.
- **"Attention is X"**: §7 line 2317-2318 says "attention emerges as the communication mechanism of distributed variational inference … derived the generalized attention weight … from variational free energy minimization." This is the user's framework's interpretation; the phrasing "emerges" / "derived from variational free energy minimization" is consistent with the (R)-style reduction-to-softmax framing rather than (I)-style "attention is X." The §3.2.1-style softmax interpretation from [Vaswani2017] is acknowledged via the limit-recovery framing. No flag.

This is a clean inheritance — the slimmer manuscript has shed the rhetorical drifts of the parent.

## Other observations (out of scope but worth flagging)

- Several config-vs-runtime gaps would benefit from a single Reproducibility appendix listing: (i) `experiment_config.json` paths in the public repo for each row of Table 1; (ii) GPU class, software stack, expected wall-clock; (iii) random seeds used. The body's statement at line 2329 "All experiments reported in the results section can be reproduced using the provided configuration files and random number generator seeds documented in the repository" is good-faith but should be backed by specific paths.

- The `2026-05-18_edits.md` document (the day-of-edits artifact required by CLAUDE.md's Post Edit Policy) is in the Attention directory and documents only the tqdm progress-bar edits — unrelated to this review's scope. No manuscript-level edits are recorded for today, which is consistent with this being a review-only pass.

## Overall verdict

**Major revision.** The manuscript is conceptually well-scoped and the conjecture framing in §6.4 is correctly cautious. But the empirical sections leave too many quantitative claims unbacked in the artifact: three missing figure files (M-C-1), one missing citation key (M-C-2), table-vs-CSV inconsistencies for the K=90 result (M-C-3), a representational-capacity claim that elides the parameter overhead (M-C-4), and a Limitations section that does not disclose the two architectural caveats already documented in the project's own CLAUDE.md (M-C-5). Each of these is straightforward to fix without rewriting the theory. The over-claim concerns from the parent review do not propagate into this slimmer manuscript; the principal remaining work is reproducibility hygiene.

Sources consulted for citation/empirical verification:

- [Efficient Streaming Language Models with Attention Sinks (Xiao et al., ICLR 2024 / arXiv:2309.17453)](https://arxiv.org/abs/2309.17453)
- [StreamingLLM project page](https://hanlab.mit.edu/projects/streamingllm)
