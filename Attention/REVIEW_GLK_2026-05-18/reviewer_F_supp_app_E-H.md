# Reviewer F — GL(K)_supplementary.tex Appendix E–H (lines 664–1323)

Date: 2026-05-18
Scope: Sections E.1–E.9 (BERT validation), F.1–F.5 (RG universality), G.1–G.3 (symmetry breaking), H (KL uniqueness via variational duality). Approximately 660 manuscript lines.

## Summary

The supplementary's Appendix E is its strongest piece: the BERT alpha–beta comparison is honestly framed (E.2's "Scope of the Validation" admits the algebraic-identity argument up front), the multi-passage protocol is reproducible from `Attention/figs/validation_results.json`, the multi-model and sequence-length tables match the JSON to four significant figures, and the Bayesian analysis in E.9 is appropriately conservative. Three issues recur: per-head data for Table E.7 is not present in the released JSON; the manuscript reports `|rho_beta| = 0.256` at the head-aggregated level but the Bayesian analysis posterior `mu_beta = -0.475` is outside its HDI (the on-disk `bayesian_validation_summary.json` records `claim_in_hdi: false` for `abs_mu_beta`); and the "theory-predicted optimal tau = 2*sqrt(d)" framing is presented as a "framework prediction" but the derivation is a one-line concentration-of-measure argument that is generic to any squared-distance-form softmax. Appendix F overstates its result: the conjecture as written contains an internal contradiction between clauses (i) (fixed point at all g_i = 0) and (iv) (within-cluster variance generates anisotropy from g_1 = 0), and the linearized-RG matrix omits the source-term coupling that (iv) explicitly introduces; the numerical fits for g_2 and g_3 disagree with the conjectured exponents by factors of 1.5 and infinity-sign respectively, and the manuscript attributes both to "finite-size effects" with no quantitative finite-size analysis. Appendix G presents observation-driven specialization of agent norms as "symmetry breaking," but never identifies the residual subgroup, and contains an arithmetic inconsistency between body text (8 agents) and figure caption (6 agents). Appendix H's Conditional Uniqueness Theorem is still missing the real-analyticity hypothesis that the prior MR-5 fix added to the main paper — this is unresolved drift between the two manuscripts; the proof's "ratio ranges over all of R+" justification in Step 3 is a different argument than the analyticity-based one and is plausible but requires a richness lemma the theorem statement does not advertise.

## Standards against which the manuscript was reviewed

- [Vaswani2017] for scaled dot-product attention and the sqrt(d_k) scaling.
- [BleiKuckelbirgJordan2017] for variational ELBO framing and Gaussian KL closed forms.
- [Friston2010, ParrPezzuloFriston2022] for the canonical free-energy form (single agent).
- [Nakahara2003], [Frankel2011], [KobayashiNomizu] for parallel transport and gauge structure.
- [AmariNagaoka2000], [Csiszar1967], [van Erven & Harremoës 2014 IEEE Trans IT 60(7)] for f-divergence theory.
- Wilson/Cardy/Polchinski RG references (cited in `external_bibliography.md` "Coverage gaps") — these books were not retrieved at review time; cited at source level only.
- [Devlin2018BERT, sanh2019distilbert, liu2019roberta, lan2019albert] for the multi-model lineup.
- The Bayesian-modeling part is checked against [Hoffman&Gelman2014, salvatier2016probabilistic] but principally against the user's own `bayesian_validation.py` and its on-disk outputs.

Books not retrieved at review time: Wilson, Cardy, Polchinski (RG); Kobayashi-Nomizu Vol I (holonomy); Amari & Nagaoka. These are cited at source level only.

## Major findings

### M-F-1. Appendix F Conjecture: internal contradiction between fixed-point and emergent-anisotropy clauses

**Claim (lines 951–962, conj:rg_universality_supp):**
(i) `g_1* = g_2* = g_3* = 0` is a fixed point.
(ii) All scaling dimensions y_1, y_2, y_3 < 0; fixed point is IR-stable.
(iii) All finite-coupling models flow to the transformer limit.
(iv) The coarse-graining map generates anisotropy from within-cluster variance: at `g_1 = 0`, `Sigma_A = sigma^2 I + Var_A(mu)` is generically anisotropic.

**Claim kind:** (N) novel, presented as theorem-like Conjecture; classified by the manuscript as conjecture.

**Standard treatment:** In Wilson-style RG, a *fixed point* of the coarse-graining map `R_n` is a coupling `g*` such that `R_n(g*) = g*`. Clauses (i) and (iv) cannot both hold for the same map: if (iv) generates non-zero anisotropy from `g_1 = 0`, then `R_n(g_1=0) != 0`, so `g_1=0` is *not* a fixed point of the full nonlinear map. It can at best be a fixed point of the projection of the map onto the "original channel" component, which is a different (lower-dimensional) statement. The linearised RG matrix in Eq. (rg_matrix_supp) is a 3x3 diagonal of `n^{-1/2}, n^{-1}, n^{-2}` with no source term, which mathematically encodes "all couplings decay under R_n with no driving term" — exactly the opposite of what clause (iv) states.

**Problem:** As written, the conjecture is mathematically incoherent. The user's numerical Table 3 actually demonstrates this incoherence: `g_1^(tot)` jumps from 0.300 to 42.47 between levels 0 and 1 — a 142x increase, the opposite of a decay — and then fluctuates around 30–50 across all later levels. The text on line 1029 acknowledges this ("does not decay, [...] the emergent within-cluster variance dominates at every coarse-grained level"), but the linearized-RG-matrix construction is left in place as if it described the system.

**Required revision:**
1. Recast: g_1=0 is a fixed point of the *projected* map on the original channel, not of the full map.
2. State explicitly that under the full map the emergent channel sources non-zero anisotropy at every step, so the trajectory of total anisotropy is *not* described by the 3x3 linearised matrix.
3. Drop "Stability" and "Universality" in clauses (ii)–(iii) as written, or restrict them to the projected/original-channel sub-claim and rename accordingly.
4. The figure-of-merit "universality class" claim only survives if recast as: "the projected anisotropy decays with exponent -1/2 under R_n regardless of starting g_1, demonstrating the original-channel scaling." This is a real, defensible result; the universality-class language is bigger than the result supports.

### M-F-2. Appendix F numerical "validation" for g_2 and g_3 disagrees with predictions; "finite-size" attribution is asserted but not quantitative

**Claim (lines 1029–1031):** "graph-based exponents for g_2 = -0.66, g_3 = +0.17 deviate from CLT predictions (-1, -2), which we attribute to finite-size effects."

**Claim kind:** (R) reduction (the CLT validation in Table on line 997–1003) plus (I) interpretive (the graph-based exponents are reframed as "finite-size corrections").

**Standard treatment:** A real RG validation needs at minimum (a) error bars on each level's coupling, (b) a finite-size scaling analysis with at least two N values (typically N, 2N, 4N) so the user can extrapolate to thermodynamic limit, (c) verification that the deepest levels (`N_l = 2, 4`) are not in the regime where finite-size corrections dominate.

**Problem:** Verified by direct fit of the table:
- y_1^(orig) = -0.4972 (predicted -0.500) — matches.
- y_2 = -0.6604 (predicted -1.0) — off by 50%.
- y_3 = +0.173 from non-zero levels (predicted -2.0) — sign-wrong by infinity.

These deviations are not "finite-size corrections" in any conventional sense — `y_3 = +0.17` (positive, monotonically growing g_3 with cluster size from 0.035 to 0.518) is a *relevant operator* signal in RG language, the opposite of what (iii)/(ii) claim. The drop of g_3 to 0.000 at level 6 (`N_l = 2`) is a degeneracy artifact — a 2-node graph has no triangles, so g_3 is undefined — not a "deepest-level finite-size effect." The text on line 1031 quietly excludes this point but presents the exclusion as if the trend supported the conjecture.

**Required revision:**
1. Provide error bars on each level (e.g., bootstrap over multiple seeds for the synthetic VFE system).
2. Repeat at N = 256, 512 to demonstrate finite-size scaling: if the deviations from (-1, -2) shrink as N grows, the "finite-size" explanation gains support; if not, the conjecture is empirically falsified for g_2 and g_3.
3. Acknowledge the g_3 trend is growing, not "decaying with finite-size correction." This is sign-wrong and cannot be hand-waved.
4. Drop "the CLT validation confirms that the mathematical content of the scaling predictions is exact" — the CLT validation confirms the *math of independent averaging*, not the predictions for the gauge-transformer attention graph. These are different claims.

### M-F-3. Appendix H Conditional Uniqueness Theorem: real-analyticity hypothesis still hidden in proof

**Claim (lines 1201–1219, thm:uniqueness_supp):** "Let D(q,p) be a convex f-divergence with f convex, f(1)=0, f'(1)=0. Suppose D is linear inside the free energy and the attention weights satisfy sum beta_ij = 1. Then the geometric-mean Boltzmann form holds for all priors and beliefs iff D is the forward KL."

**Claim kind:** (N) the theorem is presented as a uniqueness result.

**Standard treatment:** The prior peer review of `GL(K)_attention.tex` flagged MR-5: the analogous theorem hid real-analyticity in the Richness Lemma of the proof, and the user's documented fix (REVIEW_2026-05-18.md, section 10.1) was to add "real-analytic on (0,∞)" to the theorem statement of `app:conditional_uniqueness` in the main paper. The supplementary's version (this Appendix H) was not updated in lockstep.

**Problem:** The supplementary's Step 3 reads: "the ratio q_i/[Omega_ij q_j] ranges over all of R+, therefore f'(t) = log t + k must hold globally" (line 1281). This is a different argument from the analyticity argument:
- *If* the ratio actually attains every value in (0, ∞) as p_i, {q_j} vary, the global pointwise identity is immediate and analyticity is not needed.
- *If* the ratio only attains an interval (a, b) ⊂ R+ (for instance, if there are integrability constraints on q_i^* coming from finiteness of mass), the pointwise identity only gives `f'(t) = log t + k` on (a, b); to extend to (0, ∞) you need analyticity (or some other regularity that propagates the equation).

The proof does not justify the "ranges over all of R+" assertion — it is stated without a lemma. Either:
1. The user gives a Richness Lemma (free choice of p_i and q_j fills out R+), in which case analyticity is unnecessary and Step 3 stands; or
2. The user adds the analyticity hypothesis to the theorem statement (as the main-paper version did).

The two are not equivalent. The main paper took the analyticity route; the supplementary went a different way without saying so.

**Required revision:** Pick one:
- Add "f real-analytic on (0, ∞)" to the theorem hypotheses and align the supplementary with the main paper (the easier fix; matches MR-5 resolution).
- Provide an explicit Richness Lemma: "for any t > 0 and any base point c, there exist normalizable p_i and {q_j} such that q_i*(c) / [Omega_ij(c) q_j(c)] = t at c." This needs care: q_i* is determined by p_i and {q_j} via the geometric-mean formula, so the assertion is that the family of induced ratios is dense in (0, ∞). Plausible but not free — it has to be proved.

Either way, do not present the proof as if Step 3's "ranges over all of R+" is self-evident.

### M-F-4. Appendix E.7 Per-Head Temperature Dispersion table: numbers not reproducible from the released JSON

**Claim (Table tab:temp_dispersion_supp, lines 842–857):** five-row table with "Key-norm CV", "tau_opt", "r@tau*", "r@19", "Temp disp (CV)" for ALBERT/BERT-base/BERT-large/DistilBERT/RoBERTa, on "30 passages."

**Claim kind:** (S/R) empirical reproducibility claim.

**Standard treatment:** Tables in an empirical section of a JMLR-style paper need either a referenced figure/script that produces them or a JSON/CSV with the underlying numbers.

**Problem:** Searching `Attention/figs/` and the BERT validation outputs:
- `validation_results.json` contains phases 1–7 (single-passage, multi-passage, tau sweep, multi-model, identity decomposition, entropy, seqlen) and reports `n_passages = 20` for the multi-model phase. Tables E.6 and E.8 are reproducible from this JSON.
- *No file in the repository* contains "Key-norm CV", "Temp disp (CV)", or 30-passage per-model statistics. `grep -r temp_dispersion` and `grep -r per_head_tau` return zero hits in code, JSONs, and CSVs.
- The script that generated `validation_results.json` is not visible in `Attention/` or `transformer/` (no `bert_validation.py`, `transformer_validation.py`, or similar exists; only the post-hoc `bayesian_validation.py`).

**Required revision:** Provide one of:
1. A reproducibility JSON / CSV with the five-model x five-statistic per-head data for Table E.7, and an explicit script/notebook that produced it. Even a single-cell `phase8_temp_dispersion` block in `validation_results.json` would suffice.
2. If this data was produced by a private script, commit it to the repo (under `scripts/` or `transformer/analysis/`) before publication.

The discrepancy between "30 passages" in E.7 and "20 passages" in E.5/E.6/E.9 is unexplained.

### M-F-5. Appendix E.9 keynorm bias: posterior |mu_beta| = 0.475 disagrees with manuscript point estimate 0.256

**Claim (line 905 / Sec. E.9.3):** "For KL-distance attention, the posterior population mean is mu_beta-hat = -0.475 (94% HDI [-0.507, -0.447]). The standardized effect size is Cohen's d = 1.43."

**Claim kind:** (S) standard Bayesian inference output.

**Standard treatment:** Posterior HDI should *contain* the point estimate from the same data; if not, that's a sign of model misspecification or data subsetting mismatch.

**Problem:** `Attention/figs/bayesian/bayesian_validation_summary.json` records:
```
keynorm.manuscript_comparison.abs_mu_beta:
  manuscript_value: 0.256
  posterior_mean:   0.4749
  hdi_94:           [0.4472, 0.5065]
  claim_in_hdi:     false
```
The JSON's own diagnostic flags this: the manuscript's frequentist point estimate (0.256, reported in §E.4 line 766 as "for KL-based attention beta at tau=19.0, |rho|-bar = 0.256") is *outside* the 94% HDI of the posterior model that was built on the same per-head correlations. The manuscript reports the posterior in §E.9.3 with the correct value (-0.475) but never mentions the discrepancy with §E.4's 0.256. A reader comparing the two sections will find them apparently contradictory.

Cohen's d also has a diagnostic issue: `keynorm` model `r_hat = 1.089` for `sigma_beta` and `ess_bulk = 18.1421` for `sigma_beta` (the JSON's `keynorm.diagnostics.all_r_hat_ok = false`, `n_divergences: 4`). The manuscript's E.9 opening line says "All models achieved R-hat < 1.05 and effective sample sizes exceeding 1000 for primary parameters, with no pathological divergences on the quantities of interest." If the user considers `sigma_beta` not a "primary parameter," that is defensible — but Cohen's d depends on sigma_beta directly, and Cohen's d *is* a primary headline quantity (Sec. E.9.3).

**Required revision:**
1. Reconcile §E.4 (0.256) vs §E.9.3 (-0.475). Most likely §E.4 reports a head-aggregated mean of `|rho|` while §E.9.3's hierarchical posterior shrinks to the population mean of `rho` (note the sign change: |rho| vs -rho). State this explicitly. If §E.4's 0.256 is `mean(|rho|)` over 144 heads but §E.9.3's posterior is on the signed `rho` with shrinkage toward population mean, that explains the gap quantitatively — but the manuscript must say so.
2. Either (a) restate the E.9 opening qualification ("R-hat < 1.05 except sigma_beta which had R-hat = 1.089 with 4 divergences, an artifact noted in the JSON") or (b) report the diagnostics in line with the JSON record. Burying a non-converged chain that drives the headline Cohen's d through the diagnostics is not acceptable for a JMLR-grade validation.
3. The "claim_in_hdi: false" record on disk should be reflected in the manuscript or the JSON's record fixed.

### M-F-6. Appendix E temperature framing: "theory-predicted tau = 2*sqrt(d)" oversells a one-line concentration argument

**Claim (lines 696, 711, 742–748):** "The empirical optimum at tau = 19.0 is within 19% of the theoretical prediction tau = 2*sqrt(d) = 16 for d = 64. The factor of sqrt(d) from dimensional concentration of dot-product variance, and the factor of 2 from the 1/2 prefactor in Gaussian KL... finite-dimensional corrections of order unity to the optimal temperature."

**Claim kind:** (R) reduction (claim that the gauge framework predicts a specific value) but classified by the manuscript as a framework-specific prediction.

**Standard treatment:** [Vaswani2017 §3.2.1] gives the variance argument exactly: dot products `Q · K` of unit-variance components in dimension d_k have variance d_k, so dividing by sqrt(d_k) restores unit variance and prevents softmax saturation. This argument is generic to *any* softmax-of-dot-products attention with i.i.d. components. It does not invoke the gauge framework.

**Problem:** The manuscript presents the sqrt(d_k) scaling as a gauge-framework prediction. The factor of 2 from the Gaussian KL is the only piece of derivation that is gauge-framework-specific: the squared-distance form `exp(-||Q-K||^2/tau)` has a `1/(2 sigma^2)` prefactor that gives 2*sqrt(d_k) once you map sigma^2 to the Vaswani 1/sqrt(d_k) calibration. So the "factor of 2" is the only framework-specific piece, not the whole prediction.

Moreover: the squared-distance and dot-product forms are algebraically related under softmax (line 738 acknowledges this). For *constant key norms*, `exp(-||Q-K||^2 / tau) = exp((2 Q.K - ||Q||^2 - ||K||^2)/tau)`, which under softmax with constant ||K||^2 across j becomes `softmax(2 Q.K / tau) = softmax(Q.K / (tau/2))`. So the effective dot-product temperature is tau/2 — confirming algebraically that the squared-distance form's optimal tau is 2x the dot-product form's optimal 1/sqrt(d_k) calibration. This is a one-line algebraic correspondence, not a framework-specific prediction.

The "framework predicts tau = 2 sqrt(d_k)" framing also conflates the *learnable* kappa in `tau = kappa*sqrt(K)` (CLAUDE.md, project conventions) with the *fitted* tau = 19 here. The Vaswani standard absorbs kappa = 1; the framework allows learnable kappa. The factor-of-19% deviation could be the empirical kappa, not "finite-dimensional corrections."

**Required revision:**
1. Re-attribute sqrt(d_k) to [Vaswani2017 §3.2.1] (it is the standard variance argument, not a framework prediction).
2. Re-attribute the factor of 2 to the Gaussian KL prefactor (this *is* framework-specific to the squared-distance form, and earns the framework one bit of empirical credit).
3. Re-frame the 19% deviation either as a finite-d correction (with a quantitative leading-order estimate, not a generic "subdominant fluctuations" appeal) or as the fitted kappa (about 1.19), not both.
4. The Bayesian posterior HDI [11.6, 31.5] from §E.9.2 contains both 16 and 19; the right way to report this is "the theoretical prediction tau = 2 sqrt(d_k) = 16 falls within the 94% HDI of the data," not as "19% deviation."

### M-F-7. Appendix G: "symmetry breaking" presented without identifying the residual subgroup

**Claim (lines 1037–1060, Appendix G.1–G.3):** Without observations, 8 (sic) agents in SO(3)/l=4 irrep converge to a symmetric vacuum. With observations, "agents flow toward unique norms, with distinct feature directions emerging as specialized modes of the previously symmetric space."

**Claim kind:** (I) interpretive — "symmetry breaking" carries strong physics connotations.

**Standard treatment:** [Peskin-Schroeder QFT, Ch. 11 on SSB; Weinberg Vol II Ch. 19] In SSB language, a symmetry G is spontaneously broken to a subgroup H ⊂ G when:
- The Lagrangian / free energy is G-invariant.
- The vacuum state (ground state) is not G-invariant but is invariant under H.
- The number of Goldstone modes equals dim G/H.

**Problem:** Three issues:
1. **Agent count inconsistency.** Body text (line 1037) says "8 overlapping agents"; figure caption of `fig:mu_q_center_vacuum_supp` (line 1045) says "Norm trajectories for all six agents." The same caption text is on line 1055 (figure mu_q_center_observed_supp). One of these is wrong; either 6 or 8 must be authoritative.

2. **G-invariance not demonstrated.** The body says agents are initialized with `gamma_ij = 0` (model alignment off), but the manuscript does not verify that the F functional is SO(3)-invariant on this configuration. Without the invariance check on F, "symmetry breaking" is ungrounded — the perturbed state can specialize for many non-symmetry reasons.

3. **Residual subgroup never identified.** SSB requires identifying H ⊂ G. The manuscript says agents "flow toward unique norms" but never asks: in what subgroup of SO(3) is the broken-symmetry vacuum invariant? If the answer is "trivial subgroup" (full breaking), that should be stated, and the count of Goldstone modes (dim SO(3) = 3) should be reported. If the answer is "U(1)" (axial symmetry around a specialized direction), state it. Without H, this is not symmetry breaking but symmetry-distinguished trajectories — which is a weaker, descriptive claim.

**Required revision:**
1. Fix the 6-vs-8 inconsistency.
2. State (and ideally verify numerically) that the F functional with the given observation model and gamma_ij = 0 is SO(3)-invariant.
3. Identify the residual subgroup H, count the Goldstone modes, or rename "symmetry breaking" to "observation-driven specialization" if the SSB language can't be earned.
4. Move "model-channel formalism" §G.3 to the model-channel discussion in the main appendix structure — it is the only piece of Appendix G that is not about the simulation, and its placement after the SSB figures is jarring.

## Minor findings

### m-F-1. E.2: "all 144 heads achieve p < 0.001 significance on 100% of passages" (line 704)
This is consistent with the JSON's `frac_sig_005 = 1.0` per head. Note that with 144 heads x 105 passages = 15,120 simultaneous tests, the Bonferroni-corrected significance threshold (3.3e-6 as stated) corresponds to the conservative bound, not exact rejection at any specific p-value level. The manuscript states this correctly but the phrasing "all 15,120 head-passage comparisons survive Bonferroni correction" is slightly imprecise — the Bonferroni-adjusted *test* is whether each individual p-value is below alpha/(15120). The claim should be "every individual test's p-value is below 3.3e-6", and the JSON shows the per-head `frac_sig_005 = 1.0` already implies p < 0.05 for every test; the stronger Bonferroni claim needs the raw p-values, not just the alpha=0.05 fraction. Verify the underlying script reports the correct adjustment.

### m-F-2. E.4: ambiguity in "|rho| = 0.256 vs |rho| = 0.164"
Line 768 states "the stronger key-norm effect for beta at tau = 19 compared to the |rho|-bar = 0.164 observed at tau = 1" but does not give a citation/derivation for the *predicted* sign of this effect; the framework only predicts `|rho_beta| > |rho_alpha|` *at fixed temperature*, not a temperature-dependent intensification curve. The manuscript treats the tau-dependent intensification as a prediction; it is at best a side-effect observation.

### m-F-3. E.5 entropy comparison: ratio H(beta)/H(alpha) = 1.076 reported, ratio "approx 1.08"
Line 800 says "neither collapsed nor diffuse." A factor of 1.076 means beta is slightly more diffuse than alpha. The manuscript should report this directionally (beta is more entropic) rather than just "matched." This is a 7.6% gap.

### m-F-4. E.6: "ALBERT's parameter-sharing... regularizing attention patterns toward the distance-based structure predicted by the gauge theory" (line 835)
This is an interpretive (I) claim presented in (R) language. ALBERT has higher r=0.851 than BERT-base's r=0.804; the parameter-sharing-as-regularization story is a plausible post-hoc rationalization. An alternative null story is equally compatible: ALBERT has fewer effective per-layer parameters, so each head's W_Q W_K^T is closer to its random initialization; random projections at this dimension already give high r between alpha and beta by the algebraic-identity argument of §E.2 line 738. Without a controlled comparison (e.g., reset-W comparison or scaled-d comparison) the gauge-theory interpretation is one of several. Soften to "ALBERT's higher r is consistent with parameter-sharing regularization, though alternative explanations exist."

### m-F-5. E.8: sequence-length sensitivity standard errors not reported
Table tab:seqlen_supp gives mean and std per N, but std/sqrt(n_passages) standard errors are not given. With 105 passages and std ~ 0.20, SE ~ 0.020. The deltas between consecutive N (e.g., 0.798 → 0.747, delta = 0.051 from N=32 to N=256) are 2.5 SE, which is significant but not overwhelming. Report SE alongside std, or compute paired-passage CIs (the same passages were truncated to different N, so paired comparison is appropriate).

### m-F-6. F.1: g_2 definition uses `Omega` (without index) on line 922
"`g_2 = ||Omega_ij - Omega|| / ||Omega||` (deviation from constant transport)" — `Omega` without index appears to be the population mean of Omega_ij over (i,j) pairs, but this is not stated. Make explicit.

### m-F-7. F.4 "testable predictions" claims (lines 969–985)
Most of the listed predictions are framework-internal sanity checks (item 3: extract attention matrices and measure how g scales with cluster size — this is *re-running the analysis already in F.5*, not an independent test). Item 1 (R(K) ∝ sqrt(K) for training-data-efficiency ratio) is potentially independent but lacks specification of (a) what "matched parameter counts" means when the gauge VFE has no parameter QKV projections, and (b) what the null hypothesis is — a constant R(K) would falsify, but the manuscript doesn't pre-register the falsifier.

### m-F-8. H Step 1 derivative formula
The text on line 1230 gives `delta D / delta q_i(c) = f'(r(c))` for the variational derivative of `int (Omega q_j) f(q_i/(Omega q_j)) dc`. Direct calculation: `delta[(Omega q_j)(c) f(q_i(c)/(Omega q_j)(c))] / delta q_i(c') = delta(c - c') f'(q_i(c)/(Omega q_j)(c))`. This is correct as stated; no issue. The displayed step is fine.

### m-F-9. H "GL(K) invariance theorem" cross-reference (line 1318)
The summary cites "Theorem 1 in the main paper" — verify this label resolves correctly in the main `GL(K)_attention.tex`. If it has been renumbered or relabeled, the supplementary will have a dangling reference.

### m-F-10. E.9.4 "halflife_doublings: 27.6"
The on-disk JSON for `halflife_doublings` reports `sd = 163.99` and `hdi_3% = 10.49, hdi_97% = 34.96`. The HDI is much wider than the mean (10.5 to 35.0 doublings is more than 10x range), but the manuscript reports "posterior mean 27.6 doublings, corresponding to sequence lengths far beyond any practical transformer context window." With sd = 164 on a mean of 27.6, the posterior is very poorly identified, and reporting only the mean is misleading. Report the HDI alongside the mean.

## Empirical claim audit

For each quantitative claim in Appendix E, the verdict on backing data:

| Claim | Location | Backing | Verdict |
|---|---|---|---|
| 105 passages, 144 heads, 12 layers, BERT-base | §E.1, line 670 | `validation_results.json` keys | Backed |
| Mean r-bar = 0.795, median 0.880 (single-passage) | line 698 | `phase1_single_passage.forward_kl.mean_r = 0.7949` | Backed (rounding) |
| 94 heads (65.3%) at r > 0.8, 67 (46.5%) at r > 0.9 (single-passage) | line 698 | `n_heads_r_gt_08 = 94, n_heads_r_gt_09 = 67` | Backed |
| Grand mean r = 0.804, median 0.876 (multi-passage) | line 700 | `phase2_multi_passage.corpus_summary.grand_mean_r = 0.8040` | Backed |
| 93 heads (64.6%) at mean r > 0.8, 59 (41.0%) at r > 0.9 | line 700 | `heads_mean_r_gt_08 = 93, _gt_09 = 59` | Backed |
| Mean SE = 0.004, max SE = 0.016 (cross-passage) | line 702 | per-head `se_r` array spans ~0.002–0.016 | Backed |
| empirical optimum tau = 19.0 | line 696 | `phase3_tau_sweep.best_tau = 19.0` | Backed |
| Grand mean at tau=19 r-bar = 0.804, 95% CI [0.771, 0.838] | line 700 | `best_ci = [0.7750, 0.8378]` | Backed (rounding) |
| |rho_alpha|-bar = 0.139, |rho_beta|-bar = 0.256 (Sec E.4) | line 766 | not directly in JSON but computable from per-head `mean_keynorm_alpha/beta` | Indirect; needs spot-check |
| |rho|-bar = 0.164 at tau=1 | line 768 | not reproducible from `validation_results.json` (tau=1 single point not exported in keynorm form) | Not backed |
| H(alpha) = 1.774, H(beta) = 1.784 nats, ratio = 1.076 (20 passages) | line 797 | `phase6_entropy.mean_entropy_alpha = 1.7741, _beta = 1.7843, _ratio = 1.0757` | Backed |
| Five-model table at tau=19 (20 passages): ALBERT 0.851, BERT-base 0.804, BERT-large 0.778, DistilBERT 0.746, RoBERTa 0.612 | Table 1, line 824–829 | `phase4_multi_model.*.grand_mean_r` | Backed to 4 sig figs |
| Five-model per-head temperature dispersion table (Table 4) | E.7 | NOT IN `validation_results.json`; no script located | **Not backed** |
| Sequence-length table N=16..512 | Table 2, line 870–878 | `phase7_seqlen` list | Backed |
| Bayesian hierarchical grand_r = 0.867 (94% HDI [0.808, 0.915]) | line 897 | `hierarchical.parameters.grand_r.mean = 0.8665, hdi = [0.8084, 0.9151]` | Backed |
| Bayesian tau_opt = 21.2 (HDI [11.6, 31.5]) | line 901 | `temperature.parameters.tau_opt.mean = 21.1953, hdi = [11.63, 31.50]` | Backed |
| Bayesian mu_alpha = -0.144 (HDI [-0.195, -0.096]) | line 904 | `keynorm.parameters.mu_alpha.mean = -0.1438, hdi = [-0.195, -0.0957]` | Backed |
| Bayesian mu_beta = -0.475 (HDI [-0.507, -0.447]) | line 905 | `keynorm.parameters.mu_beta.mean = -0.4749, hdi = [-0.5065, -0.4472]` | Backed BUT `claim_in_hdi: false` for the manuscript's |rho_beta| = 0.256 reference — see M-F-5 |
| Cohen's d = 1.43 (HDI [1.08, 1.88]) | line 905 | `cohens_d.mean = 1.432, hdi = [1.08, 1.88]` | Backed BUT `r_hat = 1.0201, ess_bulk = 44.4` on `cohens_d`, marginal — disclose |
| P(|mu_beta| > |mu_alpha|) = 1.000 | line 905 | `prob_beta_stronger.mean = 1.0` | Backed |
| Multi-model population mu = 0.758 (HDI [0.653, 0.853]) | line 909 | `multimodel.parameters.pop_mu.mean = 0.7577, hdi = [0.6531, 0.8525]` | Backed |
| Beta_decay = 0.016 per doubling (HDI [0.006, 0.024]) | line 909 | `seqlen.parameters.beta_decay.mean = 0.0156, hdi = [0.0059, 0.024]` | Backed |
| All R-hat < 1.05 and ESS > 1000 for primary parameters, no divergences | line 889 | hierarchical: yes (4 divs not in `keynorm.diagnostics` for cohens_d and sigma_beta); see M-F-5 | Overstated |

## Theorem hygiene log (F & H)

**Appendix F: Conjecture rg_universality_supp (line 951).** Labeled as "Conjecture" — appropriate. Contents have an internal contradiction between clauses (i) and (iv) — see M-F-1. The "scaling dimensions" derivation in line 944's RG matrix omits the source-term coupling that clause (iv) introduces; this is a structural inconsistency in the statement.

**Appendix F: "Numerical Validation" claim at line 1031** ("the CLT validation confirms that the mathematical content of the scaling predictions is exact") — overstated. The CLT validation confirms the math of averaging independent quantities; it does not confirm the scaling predictions for the gauge-transformer attention graph, which is the object of (i)–(iv). See M-F-2.

**Appendix H: Theorem thm:uniqueness_supp (line 1202).** Hypotheses listed: convex f, f(1)=0, f'(1)=0, linearity in F, sum_j beta_ij = 1. The proof's Step 3 requires "the ratio q_i*/(Omega q_j) ranges over all of R+ as p_i and {q_j} vary." This richness claim is not a listed hypothesis. The main paper resolved this same issue (MR-5 in REVIEW_2026-05-18.md, pass 10) by adding "real-analytic on (0, ∞)" to the theorem statement. The supplementary did not follow suit. Either add the analyticity hypothesis (matching the main paper) or add a Richness Lemma (proving the ratio is dense in R+). See M-F-3.

**Appendix H: "Bregman divergence... Hessian induces the Fisher-Rao metric" (line 1308).** True for KL on simplex / exponential families [AmariNagaoka2000 Ch. 3], but only under the negative-entropy potential and on the appropriate manifold. The phrasing is loose ("yields the mixed/exponential family-projection Pythagorean theorem"). This is interpretive context, not a load-bearing claim, but state the assumptions.

## Style scan

- Banned spacing macros `\;`, `\,`, `\!`: present throughout (`\,\|\,` in KL notation, `\!` in `f\!\bigl`, etc.). The CLAUDE.md project rule and `style_constraints.md` ban these. Examples: lines 187, 197, 200, 244, 263, 322, 327, 469, 530, 644, 1156, 1167, 1207, 1226, 1242, and many more. **All `\;`, `\,`, `\!` macros should be removed in a cleanup pass.** (Note: the cited line numbers are in the scope above and in the broader manuscript.)
- Banned Claude-isms: none of "key insight", "crucially", "critically", "notably", "importantly", "interestingly", "fundamentally", "in particular", "leverages", "underscores" appear in the scope. Clean.
- Horizontal rules `---`: none in the scope. Clean.
- Self-referential drafting language: none in the scope. Clean.

## Citations checked

- [✓] `devlin2018bert` (Devlin et al. 2019 BERT) — entry exists in `references.bib` line 2261; verified by external knowledge as standard BERT paper.
- [✓] `salvatier2016probabilistic` (PyMC3, Salvatier et al. 2016) — entry exists line 2684; verified externally.
- [✓] `hoffman2014no` (Hoffman & Gelman, NUTS) — entry exists line 2694; verified externally.
- [?] `shen2008coarse` (Shen, Sun, LeSage, "Coarse-graining as a route to microscopic physics: The renormalization group in quantum field theory," *Phil. Trans. R. Soc. A* 369) — entry exists in `references.bib` line 2379. **Cannot verify**: web search returns no such paper by these authors with this title in *Phil. Trans. R. Soc. A* 369. The journal/title/authors combination looks fabricated or seriously misattributed. The user should verify against their own copy or replace with the correct citation. (Cited in `GL(K)_supplementary.tex` line 97 in the meta-agent emergence subsection, outside the strict review scope, but flagged because it bleeds into Appendix F's coarse-graining discussion.)
- [?] `garciaMillan2024network` (García-Millán & Pruessner, "Network renormalization," *Nature Physics* 2024) — entry exists line 2331. **Cannot verify in this form**: web search returns the actual paper by these authors as "Gell-Mann-Low criticality in neural networks" arXiv:2110.01859 (2021), in a different venue. The 2024 Nature Physics title appears fabricated. Either fix the citation or replace with the correct title and venue.
- [?] `anderson1984basic` (Anderson & Rosenfeld, *Neurocomputing: Foundations of Research*, MIT Press 1988) — entry exists line 2195 with a `note` admitting "Often cited as Anderson 1984 for earlier technical report editions." Citation key year-of-publication mismatch (`1984` in key vs `1988` in entry). Choose one and be consistent.
- [✓] `wilson1974renormalization` — entry exists, standard Wilson 1974 RG paper.
- [✓] `cardy1996` — entry exists at line 1145, *Scaling and Renormalization in Statistical Physics* (1996, Cardy).
- [✓] `nakahara2003geometry`, `frankel2011geometry`, `blei2017variational`, `amari2016information` — all exist and are appropriate at their cite-sites.
- [?] BERT-family multi-model citations: Devlin (BERT), Sanh (DistilBERT), Liu (RoBERTa), Lan (ALBERT) — should all be in `references.bib`. Spot-checked only `devlin2018bert`. The user should grep for `sanh2019`, `liu2019roberta`, `lan2019albert` and confirm.

## Code cross-references checked

| Manuscript reference | Code/data backing | Verdict |
|---|---|---|
| 105-passage corpus, all 144 BERT heads | `Attention/figs/validation_results.json` | Present |
| Phase 1–7 of the validation protocol | `validation_results.json` keys phase1..phase7 | Present, with names matching the protocol description |
| `Attention/figs/tau_sweep_across_passages.png` | File exists at glob path | Present |
| `Attention/figs/correlation_distribution_all_heads.png` | File exists | Present |
| `Attention/figs/cross_passage_stability.png` | File exists | Present |
| `Attention/figs/key_norm_bias_analysis.png` | File exists | Present |
| `Attention/figs/keynorm_cross_passage.png` | File exists | Present |
| `Attention/figs/attention_entropy_by_layer.png` | File exists | Present |
| `Attention/figs/Symmetric Vacuum.png`, `Symmetry Breaking.png` (App G) | Files exist | Present |
| Bayesian results | `Attention/figs/bayesian/bayesian_validation_summary.json`, `transformer/analysis/bayesian_validation.py` | Present |
| Phase 8 / per-head temperature dispersion (Table E.7) | **Not located** in `validation_results.json`; no separate JSON/CSV; no script visible | Missing — see M-F-4 |
| BERT extraction / corpus generation script | Not located in `Attention/`, `transformer/`, or `scripts/` (no `bert_validation.py`, no `transformer_validation.py`, no `compare_attention.py`) | Missing |
| `verify_section_13.py` | Verifies M-projection Gaussian barycenter, not Appendix E/F/G/H content | Out of scope for E–H |
| Appendix F `K = 90, N = 128, 7 RG levels`, 200 trials per coupling | No script located that produces Table tab:rg_flow_supp | Missing — recommended to commit the RG validation script |
| Appendix G `8 (or 6) overlapping agents on 2D base, SO(3), l=4` | No script located | Missing — recommended to commit the symmetry-breaking simulation script |
| Appendix H proof | Pure derivation; verified symbolically via sympy (see methodology) | Math correct mod M-F-3 |

## Style scan summary

The `\,`, `\;`, `\!` spacing macros are pervasive throughout the supplementary and explicitly banned by `style_constraints.md` and CLAUDE.md. A LaTeX-side cleanup pass is required. No banned phrases or horizontal rules in the scope.

## Open questions

1. **Per-head temperature dispersion (E.7) data provenance.** Where was Table 4 produced? "30 passages" disagrees with the 20-passage protocol of E.5/E.6 and the 105-passage protocol of E.2. Need the underlying script and JSON.
2. **|rho_beta| sign and aggregation conventions.** §E.4 reports `|rho|-bar = 0.256` (mean of absolute value across 144 heads); §E.9.3 reports posterior mean `mu_beta = -0.475` (signed, after hierarchical shrinkage). These are different quantities and should not be presented as inconsistent — but the manuscript does present them in a way that *looks* inconsistent without explaining the aggregation. Authors: state the aggregation in each section explicitly.
3. **Appendix F: is g_1=0 a fixed point of R_n or only of its original-channel projection?** The conjecture is incoherent as written; recast.
4. **Appendix H richness vs analyticity.** Pick one path and align with the main paper's MR-5 resolution.
5. **Appendix G agent count.** 6 or 8?
6. **Appendix G residual symmetry subgroup.** Without H, "symmetry breaking" is not earned.

## Overall verdict

**Major revisions** required. The empirical content of Appendix E is solid and largely backed by released JSON files; the framing is honest about the algebraic-identity caveat (E.2's "Scope of the Validation" paragraph is well done). However, six issues stand in the way of acceptance as written: (1) Appendix F's conjecture is internally inconsistent and its numerical validation contradicts the predictions for g_2 and g_3 in ways the manuscript hand-waves; (2) Appendix H still lacks the analyticity hypothesis that the prior MR-5 fix added to the main paper, despite both manuscripts being part of the same submission; (3) Table E.7's data is not reproducible from any released artifact; (4) E.9.3's posterior summary contradicts E.4's point estimate without acknowledgment; (5) Appendix G's "symmetry breaking" lacks the residual-subgroup identification that the SSB language requires, and has an agent-count inconsistency between text and figure caption; (6) several BibTeX entries (`shen2008coarse`, `garciaMillan2024network`) appear fabricated or seriously misattributed.

None of these are unrecoverable. M-F-3 (Appendix H analyticity) and M-F-4 (Table E.7 data) are 1-paragraph and 1-script fixes respectively. M-F-1 and M-F-2 (Appendix F) require either a restructure of the conjecture or honest reporting of the negative numerical result; both are tractable. The empirical work in Appendix E is genuinely substantial and the Bayesian analysis is appropriate scientific practice — the framing edits in M-F-5 and M-F-6 are matters of tightening, not redoing.
