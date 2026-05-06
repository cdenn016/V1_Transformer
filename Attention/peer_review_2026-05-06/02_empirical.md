# Empirical Methodology and Reporting Audit
**Manuscript:** `Attention/Participatory_it_from_bit.tex` (3,738 lines)
**Reviewer scope:** empirical sections, scaling validation, training dynamics, mass-precision validation, computational details, reproducibility.
**Date:** 2026-05-06

## Summary verdict

**Recommendation: major revisions.**

The empirical core of the manuscript rests on three claims of unequal quality. The cleanest is the multi-seed power-law scaling fit on WikiText-103 (lines 2437--2456): the protocol is documented, parameters and seeds are given, bootstrap CIs are reported, and the fit is reproducible from stored configs in principle. Even there, the manuscript reports `R^2 = 1.000` on N=11 per-K means without showing residuals, omits a non-gauge-equivariant baseline at matched K (so the "architectural fingerprint" reading is unsupported), and over-reads a 1.2-epoch undertrained run as a "scaling law." The WikiText-2 numbers (lines 2427--2431) are single-seed without CIs, baselines, or test-split details, and the manuscript itself concedes this. The mass-precision "validation" cited at line 1180 (`R^2 = 0.9998`) is referenced but no methods, seeds, fit form, or sample size appear in the main text. Reproducibility is gated on a "will be made available upon publication" repository (lines 2957--2961), which is below the standard for a manuscript whose central empirical claims invoke specific perplexity numbers, parameter counts, and a slope. Multiple cross-references to a "companion paper" (Dennis2025trans) for seeds, CIs, and test splits leave the headline numbers in this manuscript essentially uncheckable from the text itself. Accept conditional on the issues below; not currently accept-worthy as written.

## Major issues

### 1. WikiText-2 character-level numbers have no statistical content
**Location:** lines 2427--2431, abstract line 47, Epistemic Status line 108.

> "the full gauge VFE architecture achieved 20\% lower perplexity (PPL 18.06 vs 22.6) than the standard baseline while using 25\% fewer parameters (6,534 vs 8,688)" (2429)

Single seed. No CI, no SE, no replication, no test/val split disclosure, no tokenizer disclosure ("character-level" is stated but vocabulary size is not), no statement of how many runs were considered before reporting the headline. PPL differences of 20% on a 6,534-parameter character-level model trained at context length 32 and embedding dimension 11 are within typical seed-to-seed noise for non-gauge baselines on WikiText; without seed variance the comparison has no inferential weight. The manuscript does flag the result as "proof of principle" at 2440, but the *abstract* (line 47) and *Epistemic Status* (line 108) still propagate the 20% / 25% / r=0.821 numbers as if they were validated. The abstract should not contain a single-seed number that cannot be checked from the manuscript itself.

**Required fix:** either (a) re-run with multiple seeds and report mean ± 95% CI for both architectures, or (b) remove the percent-improvement claim from the abstract and Epistemic Status and replace with "single-seed proof of principle, see companion paper for protocol details." The current text has the rhetorical force of a validated comparison without the statistical content.

### 2. The `r = 0.821` BERT-correlation claim is opaque
**Location:** line 108 (Epistemic Status), abstract context.

> "achieves $r = 0.821$ correlation with BERT attention patterns"

The manuscript gives no method for this number. Which BERT? Which layer? Which heads? Pearson, Spearman, or some matrix similarity? Aggregated over how many sequences? Tested for significance? "r = 0.821" is meaningless without specifying the random-baseline distribution; on small attention matrices, r = 0.821 between two random softmaxed matrices is not extraordinary. "Seed/CI/test-split details are deferred to the companion paper" (line 108) means the headline number in the *Epistemic Status* of this manuscript cannot be evaluated. Either give the protocol here or drop the number.

### 3. Mass-precision "validation" lacks any methods description in this manuscript
**Location:** line 1180.

> "This identification is computationally validated in the empirical mass-precision study of Section~\ref{sec:mass}, which confirms $M_{\mathrm{eff}} \propto \Sigma_p^{-1}$ with $R^2 = 0.9998$ and the harmonic-oscillator frequency scaling $\omega^2 \propto 1/M$."

Section `sec:mass` (the entire "Mass from Statistical Precision" section, lines 1094--1286) contains *no experimental description* — no number of agents, no number of $\Sigma_p$ values swept, no fit form, no residual diagnostics, no seed, no integrator step size, no how-was-$\omega$-extracted procedure. The Methods section at line 2943 mentions only "Hamiltonian belief dynamics for the mass-precision experiments use a velocity-Verlet symplectic integrator with energy drift $<0.001\%$ over 25 time units" — that is an integrator-stability claim, not an experimental protocol. So the headline `R^2 = 0.9998` and the "harmonic-oscillator frequency scaling" refer to a study whose design is not described in the paper.

**Required fix:** add a Methods subsection that states (a) the predicted relationship (slope and units in the regression of $\omega^2$ on $\mathrm{tr}(\Sigma_p^{-1})$), (b) sample size in $\Sigma_p$ values and replicate count, (c) the fit form and whether the slope itself was tested against the predicted unit slope rather than only an `R^2`, (d) what was held fixed (isolated-agent limit is mentioned, but $K$, $\beta_{ij}$, observation-noise settings should be explicit), (e) seeds. As written this is a *qualitative consistency check* dressed as a quantitative validation; the project's CLAUDE.md ("be direct," "flag interpretive correspondences vs. rigorous claims") demands that the manuscript itself mark the difference.

### 4. No matched-parameter, non-gauge-equivariant baseline in the WikiText-103 sweep
**Location:** lines 2437--2456 and Figure ref `fig:scaling_main`.

The fitted exponent $b = -1.049$ is presented as "the architectural fingerprint of the gauge-theoretic model" (line 108) and contrasted with Chinchilla's $-0.34$ (line 2456). The contrast is methodologically weak as the manuscript itself acknowledges ("$K$ is embedding dimension rather than total parameter count," 2456) — but the fix is not a hand-wave about the $K$-to-$N$ map being approximately linear. The fix is to run the *same sweep* on a standard transformer with attention projections at matched $K$ values and matched iso-token budget, and fit the same `aK^b + c`. Without that, a reader cannot tell whether `b ≈ -1` is gauge-specific or is what any small projection-free model exhibits at 1.2 epochs of WikiText-103. The "primary architectural fingerprint that survives across regimes" claim (line 2456) is unsupported.

### 5. "Scaling law" at 1.2 epochs is undertrained, and the manuscript admits it
**Location:** lines 2455--2456.

> "the WikiText-103 floor reflects an undertrained-convergence regime at $1.2$ epochs rather than a structural limit of the architecture"

If the floor `c ≈ 61.17` is an artifact of undertraining, then the *exponent* $b$ is also fit on undertrained perplexities. There is no guarantee that the $K$-scaling exponent in the converged regime equals the exponent in the undertrained regime; in standard transformers it does not (Chinchilla itself depends on the converged-loss regime). The manuscript invokes Japanese Wikipedia at ~1B tokens (line 2455) as evidence that the architecture can converge to PPL 15--30 — but those experiments are not in this paper, are on a different language and tokenizer, and are described qualitatively. So the central scaling claim of the paper is fitted on data the authors themselves describe as not converged. Either (a) train longer (multiple epochs) on a subset of $K$ values to show the exponent is stable as training proceeds, or (b) reframe `b ≈ -1` as a *partial-training* exponent rather than an architectural law.

### 6. R² = 1.000 on 11 points without residual or leverage diagnostics
**Location:** line 2448, 2453.

`scipy.optimize.curve_fit` of a 3-parameter form `aK^b + c` to 11 means is severely under-determined for an honest goodness-of-fit reading; an `R^2 = 1.000` printout is what the routine returns when residuals are below the float-printing precision relative to the response variance, but it tells the reader nothing about model adequacy. The bootstrap CI on `b` is the right object — and it is reported, which is good — but the `R^2 = 1.000` should be removed (or replaced with residual standard error, max signed residual, and a residual plot in the SI). It currently invites the reader to treat the fit as exact, when 11 points and 3 parameters cannot support such a reading.

### 7. Bootstrap design: resampling seeds within $K$ does not propagate $K$-design uncertainty
**Location:** line 2444.

> "drawing seeds with replacement within each $K$ value to keep the sweep design intact"

This is a defensible choice for reporting per-$K$ mean uncertainty, but it does *not* propagate uncertainty in the *exponent* coming from the choice of $K$ grid. With only three seeds per $K$, the resampling-with-replacement bootstrap on three items is on the very edge of being well-defined; many resamples will have one seed sampled 2 or 3 times, and the resampling distribution is therefore discrete with very small support. A more defensible report would be (a) seed-only bootstrap as currently done, plus (b) a leave-one-$K$-out re-fit to assess sensitivity to the design points, plus (c) a parametric residual bootstrap. The CI [-1.103, -0.998] is reported as the headline uncertainty on a number that drives the abstract; it deserves a sensitivity check.

### 8. Single-seed simulation results are described in language that suggests reproducibility
**Location:** lines 2009 ("All runs used random seed 2"), 2011, 2127--2153, 2212--2226.

The Ouroboros Tower simulations (Section 6.1) are run on a *single seed* (seed 2). The manuscript correctly flags "we describe the event as a reorganisation in a single run and reserve the phase-transition vocabulary for future multi-seed work" (2143). However, downstream rhetoric escalates: "Participatory Structure Validated" as a section title (2212), "We have demonstrated computationally that..." (2215), and the abstract claims "multi-scale meta-agent dynamics with bidirectional consensus run in working simulations" (line 47). A single-seed run is *one demonstration*, not a validation. The exponent `α ≈ 1.8` in the single-trajectory critical-region fit (2138) appropriately disclaims interpretation, but the section header at 2212 should not say "Validated."

### 9. Training-dynamics figures (Figs. 9--11) are 100-step runs at small scale and are over-interpreted
**Location:** lines 2461--2484.

> "the full gauge VFE architecture (Figure~\ref{fig:train_vfe}) achieves comparable convergence despite using only geometric updates through natural gradient descent" (2461)
> "validates that variational free energy minimization on gauge-theoretic statistical manifolds can match or exceed standard neural network performance" (caption fig:train_vfe, 2483)

100 training steps on a 6,534-parameter character-level model on WikiText-2 cannot validate "match or exceed standard neural network performance." The figures are useful as proof-of-life. The captions should read "preliminary indication that the architecture is trainable," not "validates...match or exceed." Loss values quoted ($\approx 3.12, 2.99, 2.83$) carry no error bar.

### 10. Code, data, and config availability is conditional on publication
**Location:** lines 2957, 2960--2961.

> "The configuration files, simulation logs, and analysis scripts that reproduce the figures in this work will be made publicly available upon publication."
> "...will be released at https://github.com/cdenn016/Participatory-It-From-Bit-Universe upon publication."

This is below current ML/physics reporting standards. For a manuscript whose principal empirical claims are (a) specific PPL numbers, (b) a fitted exponent with bootstrap CI, (c) seed values, (d) a separate companion-paper deferral for `r = 0.821` and seeds, the code and configs should be available now, at submission, in a public read-only form (anonymous repo, Zenodo DOI with an anonymous token, etc.). The introductory line 155 already links a *different* repository (`https://github.com/cdenn016/Gauge-theory-of-machine-learning`) for "open-source implementations" — which appears inconsistent with the Code Availability promise of a *new* repo "upon publication." The reader cannot tell which repo is the source of truth for the WikiText-103 sweep. **Critical for project's own CLAUDE.md rule** ("User may not be running the config values which match the repo. always double check what values the user is using!") — the same risk applies to readers, doubly so when the repo is not public yet.

### 11. The "Why was the kinetic term missed?" subsection is post-hoc, not falsifiable
**Location:** lines 2491--2501.

> "Standard treatments of active inference and variational free energy minimization employ first-order gradient descent, implicitly taking the overdamped limit..."

The argument is plausible but it is rationalization of a discovery, not a falsifiable diagnostic. The manuscript does not distinguish what was *predicted* from kinetic-term considerations *before* observing the mass-precision empirical regression versus what was *explained* after. As written, this reads "we found a second-order structure that prior work neglected; here is why prior work neglected it." That is fine as commentary, but it should not be presented in a section that bears on the empirical validation. A useful tightening: state explicitly whether the harmonic-oscillator $\omega^2 \propto 1/M$ scaling was a *prediction* of the kinetic-term framework or an *observation* that the kinetic-term framework was constructed to explain. The two have very different evidential weight.

### 12. "Iso-token budget on a single 9900X CPU" is a flag worth raising
**Location:** line 2944.

> "large-scale gauge-transformer training (Section~\ref{sec:scaling_validation}) was run on the same machine, hence the iso-token rather than iso-FLOP budget choice."

Training a single-layer transformer on 122.9M tokens on an AMD Ryzen 9 9900X CPU is unusual at this scale. With 11 K-values × 3 seeds = 33 runs at 122.9M tokens each, this is a large CPU compute commitment. The choice of iso-token rather than iso-FLOP is *not* primarily a hardware choice — for fairness in an ML scaling study it is the standard choice — but the rationalization given ("hence the iso-token rather than iso-FLOP budget choice") suggests the iso-token choice is the *consequence* of the CPU-only setup, which is a curious framing. More important: at this throughput, has the per-run wall-clock been recorded? Are all 33 runs consistent in batch size handling? A table giving (K, batch size, steps, tokens, wall-clock, seed) would resolve this in one paragraph.

## Minor issues

### M1. WikiText-103 tokenizer is mentioned but eval protocol is not.
Line 2456 mentions "gpt2 BPE 50,257-vocabulary" for WikiText-103 in the cross-dataset comparison aside, but the main scaling-validation paragraph (2442) does not state the tokenizer used for the fit itself. State explicitly which tokenizer is used for WT-103, what context length (sequence length 128 is mentioned, good), and how perplexity is computed (per-token vs per-character vs per-BPE-token? sliding window? non-overlapping?). The PPL numbers in `aK^b + c = 1805.55K^{-1.049} + 61.17` cannot be compared against any other paper without these details.

### M2. Abstract states `b = -1.05`, body states `b = -1.049`.
Line 47 vs line 2447. Round consistently or, better, give the body's full precision in the abstract.

### M3. Abstract says "K \in [10, 120]" but the grid is `{10, 20, ..., 100, 120}` (11 values).
The notation `[10, 120]` suggests an interval. Either say "11 values in [10, 120]" or list the grid in the abstract footnote.

### M4. "Three independent seeds (6, 23, 111)" — no power analysis.
Three seeds is the bare minimum to estimate variance; standard ML practice for a scaling-law claim is 5+. A short justification ("3 seeds chosen as compute-budget compromise; per-K SD across 3 seeds was [value]") would strengthen the report.

### M5. Companion-paper deferrals are excessive.
Lines 47, 108: PPL 18.06, 6,534 params, r=0.821, and "seed/CI/test-split details" are all deferred to `Dennis2025trans`. The current paper's empirical claims should be self-contained. Either inline the protocol details or drop the precise numbers from this manuscript.

### M6. Inconsistent repository pointers.
Line 155 footer: `https://github.com/cdenn016/Gauge-theory-of-machine-learning`. Line 2961: `https://github.com/cdenn016/Participatory-It-From-Bit-Universe`. The reader should be told explicitly which repo backs which result.

### M7. "All runs used random seed 2 for reproducibility" (line 2009).
A single seed is not "reproducibility" in the multi-seed sense; it is determinism. Replace with "this run used seed 2; reproducibility in the multi-seed sense is not established (see Section~\ref{sec:scope_limitations})."

### M8. Table 4 ("Wheeler vs our implementation," 2189) is not empirical content but is presented adjacent to empirical claims.
Move it earlier in the Discussion or mark explicitly as illustrative correspondence to avoid the impression that it is a results table.

### M9. The 825× slowdown number (line 2433) appears with no measurement protocol.
Compared to which baseline implementation? On what hardware? Wall-clock or FLOPs? At what K? "825×" is precise enough to invite checking; either give the measurement or write "approximately three orders of magnitude."

### M10. "Energy drift <0.001% over 25 time units" (line 2944).
For which experiment? At which step size? Add the integrator step size and the reference energy used for the relative drift.

### M11. `R^2 = 1.000` reported to four significant figures.
On 11 points with a 3-parameter fit, four-figure `R^2` is over-precise reporting; report the residual SD on the response scale, or `R^2` to 3 figures with a max residual.

### M12. No statement that the test split was held out from any K-tuning or seed-selection.
Standard hygiene: state "test perplexity was computed once at the final checkpoint; no test-set early stopping was used" or equivalent.

### M13. Multi-comparison correction is not relevant here, but the manuscript uses no p-values anywhere.
That is appropriate. Worth a one-line note in Methods that no NHST is performed; the reader will not assume p-hacking.

## Reproducibility audit

| Item | Reported in manuscript | Effective reproducibility |
|---|---|---|
| WT-103 grid (K values) | Yes (line 2442) | High |
| WT-103 seeds (6, 23, 111) | Yes | High *if config files are released* |
| Tokenizer | Mentioned indirectly (line 2456) for cross-dataset comparison; not stated for the main fit | Medium |
| Sequence length | Yes (128, line 2442) | High |
| Iso-token budget | Yes (122.9M, line 2442) | High |
| Batch size schedule | "adjusted across K to satisfy memory limits while preserving the budget exactly" (line 2442) — not tabulated | Low |
| Optimizer / learning rate / schedule | Not in main text — referred to "experiment_config.json ... in the repository" (line 2944) | Low until repo is public |
| Hardware / wall-clock | CPU type given (line 2944), wall-clock not given | Low |
| Code / configs URL | Promise pending publication (line 2961) | Currently zero |
| WikiText-2 protocol | Companion-paper deferral (line 108) | Zero from this manuscript |
| BERT-correlation `r = 0.821` protocol | Companion-paper deferral (line 108) | Zero from this manuscript |
| Mass-precision experiment protocol | Not in Methods (line 2943--2945 covers integrator only) | Zero |
| Ouroboros simulation seed | Single seed (2), config table at line 1984 | High for the single run, zero for the population |
| Bootstrap procedure | Yes (2444), with `2000 of 2000 successful resamples` | High |
| Fit constraints | Yes (`b in (-2,0), c >= 1`, line 2444) | High |

**Net:** the WikiText-103 scaling fit is the most reproducible result in the empirical core. The mass-precision and BERT-correlation results, both cited in the abstract / Epistemic Status, are not reproducible from the manuscript alone.

## Open questions for authors

1. What is the exponent if you fit a standard transformer (with $W_Q, W_K, W_V$ projections) on the *same* WT-103 sweep at matched $K$ and matched 122.9M tokens? Without this point, the "architectural fingerprint" claim (line 108) is unsupported.

2. What is the slope of $\omega^2$ vs $\mathrm{tr}(\Sigma_p^{-1})$ in the mass-precision experiment, with uncertainty? The manuscript reports `R^2 = 0.9998` (line 1180) but not the slope. A "validation" of $M_{\mathrm{eff}} \propto \Sigma_p^{-1}$ requires testing the slope against the predicted value, not only a high `R^2`.

3. Does the WT-103 exponent stabilize under longer training? A subset run (3--5 K values) at 5--10 epochs would address Major Issue 5.

4. What is the per-K seed standard deviation? The bootstrap CI on `b` is reported (line 2447), but the per-K spread that drives that CI is not. Please tabulate `K, mean PPL, SD across 3 seeds, n_seeds`.

5. Was any hyperparameter tuning performed on the test split? On the validation split? Across seeds? The current text is silent.

6. The abstract and Epistemic Status promote `r = 0.821`, PPL 18.06, and 6,534 / 8,688 parameter counts. Are these the headline numbers from a multi-run distribution, or are they single-run point values selected after the fact? Cherry-picking risk requires explicit address.

7. Will the GitHub repository (line 2961) be live at the time of journal review, or only post-acceptance? Reviewers cannot verify the empirical core under the latter policy.

8. Is the predicted *exponent* of the kinetic-term scaling (Section "Why was the kinetic term missed?") a prediction made *before* observing the empirical mass-precision regression, or a post-hoc explanation of it? The text does not say.

---
*End of empirical audit.*
