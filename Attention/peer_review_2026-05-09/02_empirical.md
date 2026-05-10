# Empirical Methodology Peer Review — Participatory_it_from_bit.tex

Reviewer scope: empirical evidence axis only. The mathematical / interpretive
content is reviewed elsewhere. This review evaluates the strength of the
empirical claims in Section 7 ("Results"), Section 8.4 ("Multi-Seed Scaling
Validation on WikiText-103"), Section 12 ("Methods"), the abstract, and the
introduction.

Manuscript file:
`C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\Attention\Participatory_it_from_bit.tex`
(4598 lines).

Supporting artefacts inspected:
- `publication_outputs/scaling_analysis/aggregated_K_sweep.csv` (per-K seed-mean test/val PPL, parameter and FLOP counts)
- `publication_outputs/scaling_analysis/methods.md`
- `publication_outputs/scaling_analysis/fig_scaling_main.{pdf,png}`
- `publication_outputs/scaling_analysis/fig_trajectory_by_K.png`
- `transformer/analysis/scaling_stats.py` (`fit_power_law`, `_resample_seeds_per_axis`)
- `transformer/analysis/scaling_plots.py` (`fig_scaling_main`)
- `scripts/scaling_analysis_K.py`

Recomputed independently from the CSV:
- 3-parameter fit `PPL = a K^b + c` on per-K means: a=1805.6, b=-1.0489, c=61.17, R^2 = 0.99982.
- Restricted 2-parameter fit `PPL = a/K + c` (b fixed at -1): a=1629.1, c=58.74, R^2 = 0.99960.
- Total parameter count: N = 6.5336e6 K (linear in K to four significant figures across all eleven points).
- FLOPs per step at K=80, 90, 100 are 179.1, 101.9, 114.4 GFLOPs respectively (non-monotonic in K because batch size is rebalanced to hit the iso-token budget on a single CPU).

---

## FINDINGS

### F1. [BLOCKER] Figures Fig_4 through Fig_8 (the entire empirical content of Section 7 "Results") are absent from the repository

Location: Section 7.1–7.5, lines 2659–2779; figure refs `fig:energy_flow`,
`fig:energy_landscape`, `fig:nonequilib`, `fig:condensation`, `fig:hierarchy`.

Quoted claim:
> `\includegraphics[width=0.95\linewidth]{Fig_4.png}` ... `\includegraphics[width=0.95\linewidth]{Fig_5.png}` ... `\includegraphics[width=0.95\linewidth]{Fig_6.png}` ... `\includegraphics[width=0.85\linewidth]{Fig_7.png}` ... `\includegraphics[width=0.5\linewidth]{Fig_8.png}`

Defect: a recursive search of the repository
(`find ... -iname 'Fig_4.png' -o ...`) returns no matches. None of these
five figures, on which the entire qualitative narrative of Section 7
(Phase I / II / III, energy decomposition, condensation timeline, final
hierarchy at t=200) depends, exist as files. The compiled PDF therefore
either fails to render these figures or renders only the LaTeX `??`
placeholder. As a peer reviewer I cannot evaluate the empirical content of
Section 7 because there is no figure to evaluate.

Suggested fix: locate or regenerate Fig_4–Fig_8 from the underlying
simulation logs and place them under `Attention/figs/` (or wherever
`\graphicspath` resolves), then verify the manuscript compiles with all
five figures present. Alternatively, if these figures correspond to the
items in `Attention/figs/figures/` referenced by the run name, rename them
to match the `\includegraphics` calls. Until this is done Section 7 is
unevaluable.

### F2. [BLOCKER] No simulation code for the meta-agent / Ouroboros Tower experiments is present in the repository

Location: Section 7.1 (lines 2587–2610); Methods Section 12.1 (line 3629);
Code Availability statement (line 3651).

Quoted claim:
> "Multi-agent simulations of the Ouroboros Tower were run on a single AMD Ryzen 9 9900X CPU"
>
> "We conducted meta-agent emergence experiments using the configuration detailed in the repository."
>
> "The implementation, including the gauge-theoretic active-inference simulator and the multi-seed scaling-validation pipeline, will be released at https://github.com/cdenn016/Participatory-It-From-Bit-Universe upon publication."

Defect: a content search for the simulation primitives mentioned in
Section 7 (`Ouroboros`, `hyperprior_depth`, `tau_KL`,
`consensus_check`, `meta_agent`, `emergence`) returns no Python source
files anywhere in the working tree. The configuration table
(`tab:deep_emergence_compact`) lists 14 hyperparameters
($\tau_{\mathrm{KL}}$, $\zeta_{\max}$, $\eta_{\mu_q}$, $\eta_{\Sigma_q}$,
$\eta_{\mu_p}$, $\eta_{\Sigma_p}$, hyperprior depth and decay,
consensus check interval, snapshot interval, $N_0$, K, $D_x$, max
steps) but the simulator that consumes these parameters is not in
the repository. The reader cannot reproduce any number in
Section 7 from current artefacts. Compounding the issue, the Code
Availability statement says the code "will be released ... upon
publication" while the same paragraph cites configuration files
"detailed in the repository" — the two claims are inconsistent.

Suggested fix: commit the meta-agent simulator and the seed-2
configuration JSON before submission, or remove the empirical Section 7
narrative and recast it as a worked-example sketch. The current state
(quantitative claims of "approximately 520-fold variance spike",
"approximately 28-fold gradient-variance spike", "$|\mathrm{d}E/\mathrm{d}t|
\approx 2.8$", "$\mathrm{NE} \approx 0.63$", "$\mathcal{F}_{\text{final}} =
3.2$ (95% reduction)", "$\langle \mathrm{KL}\rangle = 0.034$ (97%
reduction)", "13 scales", "173 agents") cannot be audited.

### F3. [BLOCKER] Direct factual mismatch: K=90 has only n=2 seeds, but the abstract, methods, and main scaling text all assert "three seeds"

Location: abstract line 54; epistemic-status line 117; Section 8.4 line 3137;
methods.md (`publication_outputs/scaling_analysis/methods.md`).

Quoted claim:
> Abstract: "trained on WikiText-103 at an iso-token budget of $122.9$M tokens across $K \in [10,120]$ and three seeds"
>
> Section 8.4: "We swept the embedding dimension K over eleven values $\{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120\}$ with three independent random seeds (6, 23, 111)"
>
> methods.md: "with three random seeds (6, 23, 111)"

Defect: `aggregated_K_sweep.csv` row K=90 reports `n_seeds = 2` while every
other K reports `n_seeds = 3`. One run failed or was discarded and was not
re-run. Stating "three seeds" without qualification overstates the design.
The bootstrap routine (`_resample_seeds_per_axis` in
`transformer/analysis/scaling_stats.py`) draws n=2 with replacement at
K=90, which gives only three distinct realisations of the K=90 mean (HH, HT,
TT in the obvious notation), so the reported 2000 successful bootstrap
iterations dramatically over-state the effective sample resolution of the
K=90 contribution to the CI; the resampling distribution at K=90 is
discrete with three atoms.

Suggested fix: either re-run the missing K=90 seed and update the CSV,
fit, and CI, or amend every "three seeds" claim to read "two or three
seeds per K (n=3 for K∈{10,20,30,40,50,60,70,80,100,120}, n=2 for K=90)"
and re-evaluate whether the K=90 row should be excluded from the
power-law fit altogether. The current text is incorrect as written.

### F4. [MAJOR] R^2 = 1.000 is a misleading goodness-of-fit advertisement; the b ≈ -1.05 result is barely identified above the trivial b = -1 floor model

Location: abstract line 54; epistemic-status line 117; Section 8.4 line 3137;
fig_scaling_main.png caption (suptitle of figure); methods.md.

Quoted claim:
> "$b \approx -1.05$ (95\% bootstrap CI $[-1.10, -1.00]$, $R^2 = 1.000$)"
>
> Figure suptitle: "WikiText-103 scaling: PPL vs K (R^2=1.000, n_seeds=3)"

Defect: there are two distinct empirical issues with how the fit quality
is presented.

(a) **R^2 = 1.000 is rounding theatre.** The actual R^2 on the per-K
seed means is 0.99982. A two-parameter restricted model `PPL = a/K + c`
(the b=-1 special case) gets R^2 = 0.99960 on the same data. The third
parameter b absorbs about 0.022% of additional variance over the b=-1
null. Reporting "R^2 = 1.000" suggests the cubic-parameter fit captures
something the two-parameter model misses; the data say the opposite. The
floor parameter c is doing essentially all the explanatory work, and the
remaining a/K^b term cannot distinguish b=-1 from b=-1.05 at any
empirically meaningful resolution.

(b) **The bootstrap CI [-1.103, -0.998] places b = -1 at the upper edge
of the interval.** Combined with point (a), this means the headline
finding "b ≈ -1.05, distinct from -1" is not supported by the data —
b = -1 is statistically and practically indistinguishable from b = -1.05
on this fit. The text cannot simultaneously claim that "the b ≈ -1
exponent is the primary architectural fingerprint" (line 3151) and report
the exponent to three significant figures as -1.049; either b is well
determined and the CI should exclude -1 (it does not), or it is not, in
which case the claim should be "b is consistent with -1".

Suggested fix: report R^2 to four decimal places (0.9998) so the reader
cannot misread a fourth-significant-figure improvement as perfect fit;
report b = -1.05 ± 0.05 (or "consistent with -1") rather than -1.049 to
three decimals; explicitly compare against the b=-1 restricted model
(an F-test or AIC comparison would be appropriate); and state that the
fit is dominated by the floor parameter c.

### F5. [MAJOR] The Chinchilla comparison (b ≈ -1 vs b ≈ -0.34) is misleading because for this architecture N is exactly linear in K

Location: Section 8.4 line 3151; methods.md final paragraph.

Quoted claim:
> "the gauge-theoretic model loses about 2.4 nats per decade of K. ... K is embedding dimension rather than total parameter count, and the gauge model has no learned attention projections (W_Q, W_K, W_V are absent), so the K-to-N map is approximately linear here while in a standard transformer it is approximately quadratic."

Defect: the manuscript frames the K vs N distinction as if it explains
away the ten-fold gap between the gauge-model exponent (~-1) and the
Chinchilla exponent (~-0.34). It does not. From `aggregated_K_sweep.csv`:

| K | total_params | N/K |
|---|--------------|-----|
| 10  | 6.534M  | 653,359 |
| 60  | 39.20M  | 653,409 |
| 120 | 78.42M  | 653,469 |

N is essentially exactly 6.5336·10^5 · K across the entire sweep
(log-log slope = 1.000). In other words, for this architecture
b_K = b_N. Refitting `PPL = a N^b + c` against N directly gives
b_N = -1.0489, identical to b_K to four significant figures. The
"K-to-N is linear" statement is therefore not a footnote that justifies
non-comparison with Chinchilla; it is the statement that, for a
parameter-count comparison, this architecture's exponent really is ~ -1,
and that exponent is dramatically steeper than the Chinchilla N
exponent. Whether this means the gauge model is more parameter-efficient
in the undertrained regime, or merely that the embedding-dominated
architecture has a different scaling regime, is the substantive question;
the current framing obscures it.

A second issue: the gauge model's parameter count is overwhelmingly
embedding parameters (50,257 vocab × K ≈ 50K · K ≈ 78% of N at K=120,
and the ratio N/K ≈ 6.53·10^5 includes additional per-token PriorBank
parameters). The "scaling exponent" is therefore primarily an
embedding-table scaling exponent, and the parallel to a transformer
scaling law (which scales attention and MLP parameters) is structurally
weak.

Suggested fix: either (i) drop the Chinchilla comparison, or (ii) state
plainly that for this architecture b_N = b_K = -1.05, and that this
exponent is steeper than Hoffmann et al.'s b_N ≈ -0.34 in the regime
tested, while qualifying that the gauge model's N is essentially the
embedding-table size, not a comparable composition of attention, MLP,
and embedding parameters. The current text reads as if the K-to-N
distinction defuses the comparison, when in fact it strengthens it
(into a steeper exponent on a more restricted parameter class).

### F6. [MAJOR] No baseline transformer at matched K, tokens, sequence length, and seeds is reported

Location: Section 8.4 entire; figure `fig:scaling_main`.

Quoted claim:
> "We swept the embedding dimension K over eleven values ... fitting $\mathrm{PPL} = aK^b + c$ with $b \approx -1.05$"

Defect: there is no baseline. A scaling-law claim has empirical content
only relative to comparison curves. A vanilla transformer (with
$W_Q/W_K/W_V$, an MLP, and pointwise activations) trained at the same
122.9M token budget, the same sequence length 128, the same eleven K
values, and the same three seeds would establish whether the gauge
model's b = -1.05 exponent is favourable, neutral, or unfavourable
relative to the standard architecture. As written, the headline number
is uninterpretable as a quality claim — it is just a number. The
manuscript explicitly markets the result as architecturally
distinguishing ("the b ≈ -1 exponent is the primary architectural
fingerprint that survives across regimes", line 3151), but produces no
comparison architecture against which to measure it. The single
side-comparison made — Japanese Wikipedia at ~1B tokens with the same
architecture (Section 8.4 line 3149) — is informative about the
floor c, not about the architectural exponent, because no other
architecture is being trained on Japanese Wikipedia either.

Suggested fix: train at minimum a single-layer vanilla transformer with
matched K, sequence length, token budget, and seeds, and overlay its
PPL(K) curve and fit on `fig:scaling_main`. Report the two exponents
side by side. Without this, the empirical claim of Section 8.4 reduces
to "we trained our model and the loss decreased with K", which is a
sanity check, not a scaling law. The Discussion's "primary
architectural fingerprint" framing requires a fingerprint comparison
to mean anything.

### F7. [MAJOR] "iso-token budget of 122.9M tokens" is not iso-compute and is not comparable to the cited scaling-law literature

Location: abstract line 54; Section 8.4 line 3134; Methods line 3629
("hence the iso-token rather than iso-FLOP budget choice").

Quoted claim:
> "training every configuration with sequence length 128 for an iso-token budget matched at 122.9 M tokens, with batch size adjusted across K to satisfy memory limits while preserving the budget exactly"

Defect: from `aggregated_K_sweep.csv`, FLOPs per step at K=80, 90, 100
are 179.1, 101.9, 114.4 GFLOPs respectively, and total training FLOPs
spread from 0.62 PFLOPs (K=10) to 8.4 PFLOPs (K=120) — a 13.6x range.
Iso-token does not control compute. The Hoffmann et al. (2022)
Chinchilla scaling laws against which Section 8.4 compares its exponent
are formulated in iso-FLOP planes (and on multi-layer architectures with
fully learned attention). Comparing an iso-token, single-layer,
embedding-dominated, undertrained sweep to those exponents conflates at
least four axes (training compute, depth, attention parameterisation,
training-vs-converged regime). The Methods candidly notes the budget
choice is forced by hardware ("a single AMD Ryzen 9 9900X CPU"), which
is honest but means the sweep is not comparable to published scaling
laws as a methodology. The non-monotonic FLOPs profile (K=80 uses 1.76x
the FLOPs of K=90 and 1.57x the FLOPs of K=100) further means the
"iso-budget" is not even iso-FLOP within the sweep itself.

Suggested fix: either rerun on a GPU at iso-FLOP and report
FLOPs-matched curves, or restrict the comparison to architectures
trained at iso-token in the literature (Kaplan et al. 2020 has some such
runs) and explicitly disclaim Chinchilla comparability. The
`fig:compute_frontier` figure (which the figure-output directory
indicates was generated but is not referenced in the main text) appears
to plot FLOPs and parameter-count Pareto frontiers; pull it into the
manuscript so the reader can see the compute spread.

### F8. [MAJOR] The fitted floor c = 61.17 is not given an interpretive comparison; the "Japanese Wikipedia at PPL 15-30" handwave is unauditable

Location: Section 8.4 line 3149; methods.md.

Quoted claim:
> "the same single-layer gauge-theoretic transformer trained on the larger Japanese Wikipedia corpus (about one billion tokens, roughly an order of magnitude more data than the iso-token budget used here) reaches test perplexities in the 15 to 30 range across comparable embedding dimensions, indicating that the WikiText-103 floor reflects an undertrained-convergence regime at 1.2 epochs rather than a structural limit of the architecture"

Defect: no table, no figure, no seed count, no fit, no per-K
breakdown, no per-K comparison, and no auditable artefact for the
Japanese Wikipedia run is offered anywhere in the manuscript or the
inspected `publication_outputs/`. The reader is asked to accept on
authority that PPL drops from c ≈ 61 to "15 to 30" when 10x more data
is used, with the further qualification that tokeniser, vocabulary,
language, and corpus all change simultaneously. This is the load-bearing
empirical claim that c is undertrained and not a structural ceiling, and
it is supported by no presented data.

Suggested fix: present the Japanese Wikipedia results as a side-by-side
table or figure with seed counts, training tokens, K values, and final
test PPL per K. Without it, the "undertrained" framing is unverifiable
and the c = 61 floor must be reported as the model's measured ceiling
under the actual training regime.

### F9. [MAJOR] "1.2 epochs" undertraining is corroborated by the trajectory figure but not quantified beyond a single number

Location: Section 8.4 line 3149; `fig_trajectory_by_K.png` in
`publication_outputs/scaling_analysis/`.

Quoted claim:
> "the WikiText-103 floor reflects an undertrained-convergence regime at 1.2 epochs"

Defect: inspecting `fig_trajectory_by_K.png` shows that all eleven K
curves are still descending at the end of training, with no visible
plateau on either training or validation PPL. This is consistent with
the "undertrained" claim but means the fitted exponent b ≈ -1.05 is the
exponent of an undertrained sweep, not a converged-loss scaling law.
Kaplan et al. and Hoffmann et al. report converged or near-converged
exponents; this manuscript reports an exponent in a regime where the
fit could shift substantially with continued training. A small and
straightforward addition would be to report each run's final-step
gradient norm and validation-loss slope (loss change per 10k tokens) at
the end of training, so the reader can see the undertraining quantitatively.
"1.2 epochs" by itself is uninformative — a small, easy-to-converge
model could be over-trained at 0.5 epochs and a hard-to-converge model
could be undertrained at 100. The trajectory figure should be promoted
into the main text rather than living only in the publication output
directory.

Suggested fix: include `fig_trajectory_by_K.png` (or an equivalent) in
the main manuscript next to `fig:scaling_main`, with a note quantifying
the end-of-training validation-loss slope per K.

### F10. [MAJOR] The empirical claim is "scaling within this architecture", not "scaling competitive with state of the art" — the manuscript does not draw the distinction

Location: abstract; epistemic-status (Level 1, line 117); Section 8.4;
Discussion.

Quoted claim:
> Abstract: "fitting $\mathrm{PPL} = aK^b + c$ with $b \approx -1.05$ ... in an undertrained regime"
>
> Section 8.4: "the b ≈ -1 exponent is the primary architectural fingerprint"

Defect: there is no claim, anywhere, that the gauge transformer
matches or approaches state-of-the-art language modelling on
WikiText-103. The model achieves PPL ≈ 73 at K=120 with 78M parameters;
published WikiText-103 results sit in the 18–25 PPL range with
multi-layer transformers at comparable parameter counts (e.g. baseline
Transformer-XL is around PPL 24 with ~150M parameters). The reader
should be told this directly. The manuscript instead reports an
internal scaling exponent and frames it as an "architectural
fingerprint", inviting the reader to mistake an internal scaling law
for a competitive scaling claim. The Section 1.7 "scope" disclaimer is
honest about the undertrained regime and the K range, but does not say
"this model is not competitive with multi-layer baselines on
WikiText-103 perplexity at any K we tested".

Suggested fix: add one sentence to Section 8.4 stating the actual PPL
range achieved and the published WikiText-103 baseline range so the
reader can locate the result. Without that anchor the b ≈ -1 exponent
floats free of any comparison and is easy to misread as a scaling-law
contribution to the literature.

### F11. [MAJOR] Section 7 "Quantitative Characterization" is not reproducible at the level required to verify

Location: Section 7.5 lines 2774–2786.

Quoted claim:
> "Total free energy: $\mathcal{F}_{\text{final}} = 3.2$ (95\% reduction from initialization)
>
> Mean inter-agent KL divergence: $\langle\mathrm{KL}(q_i\|\Omega_{ij}[q_j])\rangle = 0.034$ (97\% reduction)
>
> Hierarchical depth: 13 scales
>
> Total agent count: $N = 173$"
>
> "a power-law fit $\Delta E^2 \propto |t - t_c|^{-\alpha}$ to the rising portion gives $\alpha \approx 1.8$ with no error bar from a single trajectory"

Defect: these numbers depend on
(a) the missing simulator (F2), (b) the missing figures (F1), (c) the
choice of $t_c$ in the power-law fit (no value given), (d) the fitting
window (line 2776 says "rising portion" without bounds), and (e) the
particular convention by which "95% reduction" and "97% reduction" are
computed (which initialization value? mean over agents or sum?). The
text correctly disclaims that these are single-seed and that the
"critical exponent" vocabulary is not earned, and that disclaiming is
appropriate. But the convergence-metrics block is presented as if it
were an audit-grade summary, and it is not at the resolution given.

Suggested fix: include the simulator, the seed-2 configuration JSON,
and the per-step diagnostic CSV (energy decomposition, KL matrix,
agent counts per scale) so the reader can reproduce the percent
reductions and the $\alpha \approx 1.8$ fit window. The fitted $t_c$
and the fit window must be stated.

### F12. [MAJOR] The Section 7 single-seed disclaimer is present but the abstract softening ("toy multi-agent simulations") understates the gap between Section 8 and Section 7

Location: abstract line 54; Section 1.5 (Epistemic Status, lines 119,
132); Section 1.7 (Scope, line 132).

Quoted claim (abstract):
> "(ii) toy multi-agent simulations exhibiting threshold-based meta-agent formation"

Defect: the Results section repeatedly and prominently labels Section 7
as a "single illustrative run (random seed 2)" and is appropriately
hedged (figure captions, paragraph headings, and the closing
qualitative-characterisation block all carry the n=1 disclaimer). This
is good. However, the abstract's compression to "toy multi-agent
simulations" hides the n=1 nature: "toy" suggests "simplified scale",
not "n=1 seed". A reader sampling abstract → table of contents →
Section 7 figures should encounter the single-seed nature on first
contact; in the abstract it is not explicit. Compounding this, Section
1.5 line 119 ("This runs in working code for systems of 200 agents
across 25 hierarchical scales") presents the simulation as a working
demonstration without the seed count. The honest framing in Sections
7.0–7.5 deserves to be promoted to the abstract.

Suggested fix: amend the abstract to read "(ii) a single-seed
multi-agent simulation exhibiting threshold-based meta-agent formation
across 13 hierarchical scales (multi-seed reproducibility deferred)".
Update Section 1.5's Level 2 paragraph similarly.

### F13. [MINOR] `fig_scaling_main.{png,pdf}` is publication quality on the basics but has minor issues

Location: figure `fig:scaling_main` (line 3145).

Defect (positive observation first): the figure has axis labels
($K$, Perplexity), error bars (per-K seed std), a fitted-curve overlay
with a 95% bootstrap CI ribbon, dual log-log and linear-y panels, an
appropriate legend, and a 300-DPI PNG/PDF dual export. This is
basic-publication-quality. Three nits:

(a) the figure suptitle "WikiText-103 scaling: PPL vs K (R^2=1.000,
n_seeds=3)" perpetuates the misleading R^2 = 1.000 (see F4) and the
incorrect "n_seeds=3" (see F3, K=90 has n=2);

(b) the error bars are labelled "test PPL (seed mean)" with std as
the bar, which is unconventional — for n=3 the SEM (std/sqrt(3)) is
the standard choice for a "mean ± uncertainty" display, and the
caption should say which is plotted;

(c) the legend is on the right panel only and is at default font size
inside the plotting area; on a publication PDF this is acceptable but
slightly cramped given the bootstrap-CI ribbon entry.

Suggested fix: regenerate with R^2 to four decimals, n_seeds string
honest about the K=90 n=2 row, and SEM (or 95% CI) error bars labelled
explicitly in the caption. The script
`transformer/analysis/scaling_plots.py` is straightforward to amend.

### F14. [MINOR] The bootstrap implementation correctly resamples seeds within each K, but the n=3 design fundamentally limits CI resolution

Location: `transformer/analysis/scaling_stats.py:107-121` and 165-188.

Defect: the implementation is correct in design — `_resample_seeds_per_axis`
draws seeds with replacement within each K, recomputes per-K means,
and feeds the resampled mean curve back into the nonlinear fit.
This is the appropriate hierarchical-bootstrap procedure for the
design. However, with n=3 seeds per K (and n=2 at K=90), the
within-K resampling distribution has only $\binom{3+3-1}{3} = 10$
distinct multinomial outcomes per K (3 distinct outcomes at K=90).
2000 bootstrap iterations therefore vastly oversample a coarse
discrete distribution; the apparent precision of the CI to three
decimal places ([-1.103, -0.998]) is misleading at the design level.
With three seeds per K, the honest reporting precision for b is two
decimal places, not three; and a bias-corrected accelerated (BCa)
interval would be more defensible than a percentile interval given the
small n. None of this is a coding bug; it is a design limitation that
the manuscript does not surface.

Suggested fix: report b and CI bounds to two decimal places, note
n=3 seeds per K (n=2 at K=90), acknowledge that the bootstrap CI is
percentile-based rather than BCa, and consider running additional
seeds — three seeds per K is the lowest resolution at which a
seed-resampling bootstrap is even nominally meaningful.

### F15. [MINOR] Multi-head architecture footnote in Section 8.1 is not validated by the K-sweep

Location: Section 8.1 line 2884 (in `awk` snippet from
`/tmp/results_section2.txt`).

Quoted claim:
> "in modern multi-head transformer architectures the embedding dimension $d_{\text{model}}$ is partitioned into $H$ heads of dimension $d_{\text{head}} = d_{\text{model}}/H$ and each head carries its own gauge frame, so $K=d_k$ here corresponds to the per-head dimension $d_{\text{head}}$ rather than the full model width"

Defect: the empirical sweep is over K (per-head dimension). The
manuscript does not state H in the trained models, so the relation
between K and the full model width $d_{\text{model}}$ is implicit. If
H is fixed across the sweep, then $d_{\text{model}}$ also varies
linearly with K and the parameter-scaling story is unchanged; if H
varies, the comparison across K rows is structurally inconsistent.
This is straightforwardly fixable but should be stated.

Suggested fix: state H (and head count behaviour across the K sweep)
in the methods.

### F16. [MINOR] The 122.9M token figure is not explained

Location: abstract; Section 8.4.

Quoted claim:
> "iso-token budget matched at 122.9 M tokens"

Defect: WikiText-103's training split is approximately 103M tokens
under standard tokenisation. 122.9M is approximately 1.19x the
training set, so the "1.2 epochs" claim follows arithmetically, but
this is not stated explicitly in the methods. The reader must
reverse-engineer. A single sentence ("the budget corresponds to
approximately 1.19 epochs of the WikiText-103 training split under
gpt2 BPE tokenisation") would close the gap.

Suggested fix: add the sentence.

### F17. [MINOR] Methods Section 12.1 says the slow subsystem (s, r) is frozen in the simulations, but the abstract and Section 7 narrative talk about meta-agent / Ouroboros dynamics that conceptually require the slow subsystem

Location: Methods Section 12.1; Section 7.1 "Operational simplifications"
paragraph.

Quoted claim:
> Section 7.1: "Throughout the simulations of this section, the slow subsystem $(s_i, r_i)$ is frozen with $\gamma_{ij} = 0$, and the dynamic field is the operational pair $(q_i, p_i)$."

Defect: this honest disclosure means that the "Ouroboros" / meta-agent
dynamics shown in Section 7 are running on the fast subsystem only.
Several diagnostics ("model coherence" in clustering, "prior alignment"
in the figures) are described as operational proxies for the canonical
quantities. This is fine as a worked demonstration, but the abstract
and the introduction's Level 2 description ("Our Ouroboros Tower
demonstrates bidirectional information flow") do not surface that the
demonstrated dynamics are the fast-subsystem-only proxy. A reader who
reads the abstract → Section 1.5 will conclude that the full hierarchical
model has been demonstrated; only at Section 7.1 do they learn that
$\gamma_{ij} = 0$ is hardwired and the slow channel is inactive.

Suggested fix: amend the Level 2 description and the abstract to say
"a fast-subsystem-only operational variant of the Ouroboros Tower";
note that $\gamma_{ij} = 0$ disables the slow-channel coupling.

### F18. [NIT] The "convergence metrics" itemisation in Section 7.5 violates the project style guide

Location: Section 7.5 line 2774.

Defect: the manuscript style guide in `CLAUDE.md` says "Minimize
itemizations, lists, and enumerations. If content can be expressed
as a paragraph, express it as a paragraph." The convergence metrics
block is a four-bullet list of numbers that could be a single sentence.
The Phase II / Phase III / Final architecture sections are itemisation-heavy
throughout. This is a stylistic minor that can be addressed in copy-edit.

Suggested fix: convert to flowing prose where possible.

### F19. [NIT] The single-seed power-law fit text uses "α ≈ 1.8" without disclaiming the choice of $t_c$ and fitting window

Location: Section 7.5 line 2776.

Defect: as noted in F11(c)/(d), the $\alpha \approx 1.8$ fit requires
specifying $t_c$ and the rising-portion window, which are not given.
The disclaimer that this is not a critical exponent is appropriate; the
specific number 1.8 should still be given enough provenance to be
checkable.

Suggested fix: state $t_c$, the window, and the fitting routine.

---

## SUMMARY

The empirical content of the manuscript splits into two parts that
need to be evaluated separately.

**Section 8.4 (WikiText-103 multi-seed scaling).** The supporting code
and per-K aggregated CSV are present and the bootstrap implementation
in `transformer/analysis/scaling_stats.py` is correctly designed
(seed-resampling within K, per-K mean refit). The main scaling figure is
publication-quality on the basics. However, the headline empirical
claim "$b \approx -1.05$, 95% CI $[-1.10, -1.00]$, $R^2 = 1.000$" is
substantially weaker than presented. The $R^2 = 1.000$ rounds from
0.99982 and a restricted 2-parameter $b = -1$ model gets $R^2 = 0.99960$
on the same data, so the third parameter $b$ is barely identified
(F4); the abstract claim of "three seeds" is factually wrong at K=90,
which has n=2 (F3); the manuscript's own Chinchilla comparison is
mis-framed because $N$ is exactly linear in $K$ for this architecture
($N \approx 6.534 \times 10^5 \cdot K$), so $b_K = b_N$ and the
distinction the text invokes does not soften the comparison (F5);
no baseline transformer at matched $K$, tokens, sequence length, and
seeds is reported, so the "scaling exponent" is uninterpretable as a
quality claim relative to standard architectures (F6); the iso-token
budget is not iso-FLOP and the FLOPs profile is non-monotonic in $K$
(F7); the load-bearing "undertrained, not a structural ceiling" claim
is supported only by an unauditable Japanese-Wikipedia handwave (F8);
and the manuscript at no point distinguishes "internal scaling within
this architecture" from "scaling competitive with state of the art"
(F10), even though at K=120 the model achieves PPL ≈ 73 against
published baselines in the PPL 18–25 range. Several of these issues
are structural (F6, F8, F10) and would require additional experiments
to address; others (F3, F4, F5) are presentational and can be fixed in
the manuscript.

**Section 7 (meta-agent emergence).** Two blocking issues.
First, the figures Fig_4 through Fig_8 referenced throughout the
section do not exist anywhere in the repository (F1) — a recursive
search returns no matches. The compiled PDF cannot be rendering these
figures correctly, and the qualitative narrative of Phase I / II / III
(energy descent, reorganisation event, hierarchical condensation,
final architecture at $t = 200$) is therefore unevaluable. Second,
the simulator that produces these results is not in the repository
(F2); a search for the simulation primitives mentioned in Section 7.1
($\tau_{\mathrm{KL}}$, hyperprior depth, consensus-check interval,
Ouroboros, meta-agent, emergence) returns no Python source files. The
quantitative claims in Section 7.5 ("520-fold variance spike",
"95% energy reduction", "$\alpha \approx 1.8$") cannot be audited
against either figures or code. The single-seed nature of Section 7
is honestly disclosed throughout (figure captions, section headers,
and qualitative-characterisation block all label the run as
"single-seed run (random seed 2)" and reserve phase-transition
vocabulary for future multi-seed work), which is to the manuscript's
credit. The abstract's "toy multi-agent simulations" phrasing
slightly understates the n=1 nature, and the Level 2 epistemic-status
description does not surface the $\gamma_{ij} = 0$ / fast-subsystem-only
restriction, both of which deserve a minor amendment (F12, F17).

**Verdict on the empirical-evidence axis.**

For Section 8.4, the empirical evidence is **insufficient as a scaling-law
contribution** in the form currently presented, primarily because no
baseline architecture is trained against (F6), the iso-token budget is
not iso-FLOP (F7), the headline $b \approx -1.05$ is barely identified
above the trivial $b = -1$ null and is statistically consistent with $b
= -1$ at the 95% level (F4), and the K=90 / "three seeds" mismatch
needs to be corrected (F3). The result is recoverable as a more
modest claim — "an internal scaling regularity, $b$ consistent with $-1$,
in an undertrained iso-token sweep on a single-layer architecture
without learned attention projections" — but it would no longer support
the "scaling-law" or "architectural fingerprint" framing currently used.
The Chinchilla comparison should be either dropped or recast (F5).

For Section 7, the empirical evidence is **unevaluable** in the
manuscript's current state. Without Fig_4–Fig_8 (F1) and without the
simulator (F2), no peer reviewer can assess whether the reported
phenomenology (energy spikes, condensation cascade, final 13-scale
hierarchy) actually occurs as described. These are blocking issues for
acceptance of Section 7 as written. Section 7's status as a
single-seed illustrative demonstration is honestly disclosed in the
text body, which would make it acceptable as a worked example if the
two blocking issues were addressed; the section should not, however,
be advertised as a "demonstration" until the figures and code are
present.

The manuscript's overall epistemic-status framework (Levels 1, 2, 3) is
admirable and the scope/limitations section is more thorough than is
typical. Several of the findings above are presentational — the
manuscript is more honest in its qualifying paragraphs than in its
abstract and figure captions, and bringing the abstract / captions into
line with the qualifying paragraphs would close several MAJOR findings
without requiring additional experiments. The two BLOCKER issues for
Section 7 (missing figures, missing simulator) and the BLOCKER issue
for Section 8.4 (factual mismatch on seed count) are not fixable in copy
editing and must be resolved before resubmission.
