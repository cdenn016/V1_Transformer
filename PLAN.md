# Manuscript Update Plan: JMLR Attention 11-3-25.tex

## Overview

This plan describes the changes needed to update the manuscript at
`Docs/attention manuscript/JMLR Attention 11-3-25.tex` with the new experimental
data uploaded to `Transformer Manuscript/`. The manuscript has several placeholder
values and outdated results that need to be replaced with data from 9 recent
experiments (Feb 3-9, 2026).

---

## 1. Fix Placeholder: "XXXX configurations" (Line 1350)

**Current text (line 1350):**
```
We trained XXXX configurations: (1) a 3-layer model with trivial gauge transport
($\Omega = I$), $d_{\text{embed}} = 30$, $d_{\text{gauge}} = 15$, batch size 24;
and (2) a 1-layer model with learned gauge transport ($\Omega \in \mathrm{GL}(30)$),
$d_{\text{embed}} = 30$, $d_{\text{gauge}} = 30$, batch size 32.
```

**Action:** Replace "XXXX" with the actual number and update the configuration
descriptions to reflect the experiments actually run. The new data includes 9
configurations spanning K=24-80, N=64-128, 1-3 layers, GL(4)-GL(30), and 25k-150k
training steps. The manuscript should describe the key representative configurations
rather than listing all 9.

**Proposed replacement:** Describe at minimum 4 representative configurations:
1. **Exp 135:** 3-layer, trivial gauge (GL(15)), K=30, N=64, 50k steps (depth comparison)
2. **Exp 150:** 1-layer, learned gauge GL(30), K=30, N=64, 50k steps (gauge comparison)
3. **Exp 121:** 1-layer, GL(20), K=80, N=64, 150k steps (best test PPL = 121.07)
4. **Exp 124:** 2-layer, GL(16), K=48, N=64, 70k steps (multi-head comparison)
5. **Exp 152 vs 154:** Learnable vs constant alpha comparison (K=30, GL(10), N=64)

---

## 2. Update Results Table (Lines 1557-1569)

**Current Table 5 (GL(30) language modeling results):**

| Configuration | Layers | Gauge Mode | Train PPL | Test PPL |
|---|---|---|---|---|
| Trivial gauge | 3 | Fixed Omega=I | 125.1 | 135.3 |
| Learned gauge | 1 | GL(30) | 113.9 | 151.8 |

**Action:** Replace with comprehensive results table from new experiments. The new
data provides much richer results:

| Config | Layers | K | Gauge | Steps | Params | Train PPL | Test PPL |
|---|---|---|---|---|---|---|---|
| Exp 135 | 3 | 30 | Trivial GL(15) | 50k | 49.8M | 125.09 | 135.31 |
| Exp 150 | 1 | 30 | GL(30) | 50k | 49.8M | 113.92 | 151.81 |
| Exp 125 | 1 | 30 | GL(30), N=128 | 75k | 49.8M | 139.14 | 125.08 |
| Exp 124 | 2 | 48 | GL(16) | 70k | 45.8M | 124.30 | 124.84 |
| Exp 121 | 1 | 80 | GL(20) | 150k | 92.5M | 108.86 | 121.07 |
| Exp 152 | 1 | 30 | GL(10), α learnable | 25k | 19.6M | 107.25* | 154.07 |
| Exp 154 | 1 | 30 | GL(10), α=1 | 25k | 19.6M | 107.25* | 154.07 |

*Note: Exp 152 val PPL 100.60 vs Exp 154 val PPL 107.25 -- learnable alpha advantage.

The best test PPL is **121.07** (Exp 121), a significant improvement over the
previously reported ~135. The abstract currently says "~125"; this should be
updated to reflect the best result.

---

## 3. Fill in Clustering Metrics Table (Lines 1598-1616)

**Current Table 7** has 6 rows with placeholder "(to be computed on trained model)".

**Action:** Fill with actual computed values. Best data comes from Experiments 121,
125, and 152 which have complete clustering analysis.

**Recommended values (using Exp 125 GL(30) as the most representative K=30 model):**

| Metric | μ ∈ R^30 | φ ∈ R^900 |
|---|---|---|
| Silhouette score ∈ [-1,1] | -0.076 | 0.020 |
| Calinski-Harabasz index | 16.07 | 6.97 |
| Inter/intra distance ratio | 1.151 | 1.055 |
| ANOVA frac. significant dims | 0.867 (26/30) | 0.77 (77/100*) |
| PCA: 3 comp. variance | ~39% | ~11% |
| PCA: 50% variance @ n comp. | 6 | 50 |
| PCA: 90% variance @ n comp. | 21 | >50** |

*First 100 dimensions sampled from 900.
**Phi space is 900-dimensional; very distributed representation.

Alternative: Use Exp 152 (K=30, most recent):
- μ: silhouette=-0.061, CH=19.52, ratio=1.201, ANOVA=96.7%, PCA3=40.9%, 50%@5, 90%@21
- φ: silhouette=-0.002, CH=9.23, ratio=1.108, ANOVA=87%, PCA3=15.7%, 50%@22, 90%@>50

**Discussion:** The negative silhouette scores for μ indicate overlapping categories
(expected since linguistic categories are soft, not hard clusters). The positive
silhouette for φ in some experiments suggests gauge frames develop slightly better
categorical separation than belief means. The high ANOVA fraction (87-97% of
dimensions significant) confirms the structure is pervasive, not concentrated in a
few dimensions.

---

## 4. Update Abstract (Lines 43-49)

**Current:** "test perplexity ~125"

**Action:** Update to reflect the best achieved result: "test perplexity ~121"
(from Exp 121, K=80, GL(20), 150k steps). Or keep "~125" as a conservative
representative value since the K=30 configurations which are more directly
comparable to the manuscript's stated architecture achieve 125-135.

**Recommendation:** Update to "test perplexity $\sim$121--135" to capture the range
across configurations, or use "test perplexity $\sim$125" as representative.

---

## 5. Update Training Dynamics Section (Lines 1534-1551)

**Current description** references:
- "Both models achieve... ~125-165 at convergence"
- "The 3-layer model converges faster... while the 1-layer learned-gauge model
   achieves better final perplexity (165 vs. 184)"

These numbers (165, 184) don't match any current experiment. The manuscript appears
to reference older runs.

**Action:** Update the training dynamics description to match the actual results:
- Exp 135 (3-layer trivial): Train PPL 125.09, Test PPL 135.31
- Exp 150 (1-layer GL(30)): Train PPL 113.92, Test PPL 151.81
- Figure caption (line 1549) references "165 vs. 184" -- update to actual values

**Updated caption proposal:**
"The 3-layer model generalizes better (test PPL 135) while the 1-layer
learned-gauge model achieves lower training perplexity (114) at the cost of a
larger train-test gap (114 vs. 152), consistent with gauge expressiveness providing
additional capacity that can overfit."

---

## 6. Add Learnable Alpha Results Section

**New section to add** after the existing GL(K) results (Section 5.3.3):

The experiments 152 (learnable alpha) vs 154 (constant alpha) provide a direct
empirical test of the state-dependent precision theory from Section 3.5.

**Key findings to report:**
- Validation BPC improvement: 1.4% (6.65 vs 6.74)
- Peak advantage at step 7500: 4.1% improvement
- Learnable alpha shows faster early convergence
- Effect is modest due to single VFE iteration (ffn_n_iterations=1)
- The Bayesian precision parameters (a0, b0) are learned end-to-end

This validates the theoretical prediction from Eq. 18 (state-dependent alpha)
while acknowledging that the full benefit requires multiple VFE iterations per
forward pass.

---

## 7. Add Scaling and Architecture Comparison Results

**New subsection** describing the scaling behavior across configurations:

| Dimension (K) | Gauge Dim | Params | Test PPL | Test BPC |
|---|---|---|---|---|
| 24 (Exp 188) | 4 | 8.4M | 188.75 | 7.56 |
| 30 (Exp 125) | 30 | 49.8M | 125.08 | 6.97 |
| 48 (Exp 124) | 16 | 45.8M | 124.84 | 6.96 |
| 80 (Exp 121) | 20 | 92.5M | 121.07 | 6.92 |

Key finding: Test perplexity improves with embedding dimension K, consistent with
the theoretical prediction (Section 5.3.4) that increasing K is the principled path
to higher output capacity.

---

## 8. Update Figure References

**Current figures referenced in manuscript:**
- `gl30_training_curves_3layer.png` -- exists in Docs/attention manuscript/
- `gl30_training_curves_1layer.png` -- exists in Docs/attention manuscript/
- `gl30_belief_clustering.png` -- exists
- `gl30_gauge_frame_clustering.png` -- exists

**Action:** Verify these figures correspond to the updated experiment results. If
they are from older runs, generate new versions from the current experiments. The
`Transformer Manuscript/*/publication_outputs/*/figures/` directories contain
updated figures for each experiment that could replace these.

**New figures available for inclusion:**
- Training curves from each experiment
- Belief clustering evolution (multiple timesteps)
- Gauge frame clustering evolution (multiple timesteps)
- Attention heatmaps and entropy plots
- Train-validation gap plots

---

## 9. Update Computational Requirements (Line 1314)

**Current:** "AMD Ryzen 9900x CPU, and an Nvidia RTX5090 GPU"

**Action:** Verify this matches the actual hardware. The experiment configs confirm:
GPU: RTX 5090, Python 3.12.7, PyTorch 2.10.0. The CPU listed should also be
verified against the experiment metadata (the explore agent found the same hardware
specs in configs).

---

## 10. Update Configuration Details (Lines 1346-1351)

**Current text describes only 2 configurations** and mentions "50,000 training
steps (~50M parameters per model)".

**Action:** Update to reflect the full range:
- Training steps range from 25k to 150k
- Parameters range from 8.4M to 92.5M
- Sequence lengths: 64 and 128 tokens
- Gauge dimensions: GL(4) to GL(30)

---

## 11. Update Code Availability Section (Lines 1704-1708)

**Current:** References `https://github.com/cdenn016/epistemic-geometry`

**Action:** Verify this is still the correct/desired repository URL. The actual
codebase in this repo is `Gauge-Transformer` which may have a different public URL.

---

## Priority Order for Implementation

1. **Critical (blocks publication):**
   - Fix "XXXX" placeholder (Item 1)
   - Fill clustering metrics table (Item 3)
   - Update results table with actual numbers (Item 2)
   - Update training dynamics text to match data (Item 5)

2. **Important (strengthens paper):**
   - Update abstract perplexity number (Item 4)
   - Add learnable alpha results (Item 6)
   - Add scaling comparison (Item 7)

3. **Minor (polish):**
   - Update figure references (Item 8)
   - Verify computational requirements (Item 9)
   - Update configuration details (Item 10)
   - Verify code URL (Item 11)

---

## Data Sources for Each Change

| Change | Primary Data Source |
|---|---|
| XXXX placeholder | All experiment_config.json files |
| Results table | result_VFE_dynamic.json from each experiment |
| Clustering metrics | semantic_analysis_history.json from Exp 121, 125, 152 |
| Training dynamics | training_history.csv from Exp 135, 150 |
| Learnable alpha | Exp 152 vs 154 comparison, learnable_alpha_analysis.md |
| Scaling results | result_VFE_dynamic.json from Exp 188, 125, 124, 121 |
| Figures | publication_outputs/*/figures/ from relevant experiments |
