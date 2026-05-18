# Diagnostics Glossary — What the Codebase Actually Tracks

**Descriptive file.** This catalogs the diagnostic quantities the codebase computes, with file:function pointers. It is *not* the source of truth for what those diagnostics mean — the agent evaluates the diagnostics against the standard literature (`external_canon_*.md`).

**Verify before relying.** This glossary is a snapshot as of 2026-05-18; the code can move. The agent must confirm a diagnostic still exists (via Grep / Read at the cited path) before citing it in an analysis report.

## Training-step metrics (per-step CSVs)

Source: `transformer/training/metrics_tracking.py::MetricsTracker` — the `self.headers` list starting at line 42. **Verify against the actual list before citing specific columns; the headers change as new diagnostics are added.** Snapshot as of 2026-05-18:

### Core, losses, basic metrics
- `step`, `timestamp`
- `train_loss_total`, `train_loss_ce`, `train_loss_ce_raw`
- `train_loss_belief_align`, `train_loss_self_consistency`, `train_loss_model_coupling` — the user's F-decomposition terms (verify mapping to manuscript F terms in `user_theory_summary.md`; [Friston2010] for the underlying decomposition)
- `val_loss`, `val_ce`
- `train_ppl`, `train_bpc`, `val_ppl`, `val_bpc` — perplexity and bits-per-character

### Attention statistics
- `beta_mean`, `beta_std` — first/second moments of β distribution
- `kl_mean`, `kl_std` — mean and std of the cross-token KL terms (the input to the softmax)
- `attention_entropy`, `attention_concentration` — distributional sharpness of β; [Vaswani2017] for saturated-softmax dynamics

### Learning rates (per parameter group)
- `mu_lr`, `sigma_lr`, `phi_lr`, `ffn_lr` — separate LRs for μ, σ, φ, FFN groups

### Gradient norms
- `grad_norm_total`, `grad_norm_mu`, `grad_norm_ffn`

### Bayesian α diagnostics
- `alpha_mean`, `alpha_std`, `alpha_min`, `alpha_max`
- `alpha_c0`, `alpha_b0`, `alpha_c0_std`, `alpha_b0_std` — α posterior parameters
- `alpha_mahal_sq_mean`, `alpha_mahal_sq_std` — Mahalanobis-squared statistics for α

### Per-head learnable κ
- `kappa_mean`, `kappa_std`, `kappa_min`, `kappa_max`

### Performance
- `step_time`, `tokens_per_sec`

### Numerical health
- `num_chol_recover`, `num_chol_fail`, `num_nan_replace`, `num_inv_pinv` — fallback counters

### φ embedding spectral diagnostics (**this is where "effective rank" lives**)
- `phi_effective_rank` — effective rank of the φ embedding matrix
- `phi_rank_ratio` — effective rank / nominal rank
- `phi_top1_variance_fraction`, `phi_top5_variance_fraction` — energy in top singular modes
- `phi_spectral_gap`, `phi_frobenius_norm`
- `phi_mean_token_norm`, `phi_std_token_norm`

### VFE gradient decomposition (E-step component analysis)
- `vfe_grad_mu_self`, `vfe_grad_mu_direct`, `vfe_grad_mu_softmax`, `vfe_grad_mu_total`
- `vfe_grad_sigma_self`, `vfe_grad_sigma_align_direct`, `vfe_grad_sigma_softmax`, `vfe_grad_sigma_total`
- `vfe_kl_pairwise_mean`, `vfe_kl_pairwise_max`, `vfe_kappa_scaled`
- `vfe_kl_frac_above_90pct`, `vfe_kl_p95` — tail behavior of pairwise KL

### Covariance health
- `sigma_q_mean`, `sigma_q_min`, `sigma_q_max`, `sigma_q_std`
- `sigma_q_cond_mean`, `sigma_q_cond_max` — condition numbers; [AmariNagaoka2000] for Fisher singularity → over-confidence
- `sigma_p_mean`, `sigma_p_min`, `sigma_p_max`
- `prior_belief_kl_mean`, `prior_belief_kl_max`, `prior_belief_kl_std`

### Transport & attention structure
- `phi_norm_mean`, `phi_norm_std`, `phi_norm_max`
- `phi_pairwise_dist_mean`, `phi_pairwise_dist_max`
- `attn_entropy_per_head_mean`, `attn_entropy_per_head_std`, `attn_entropy_per_head_min`, `attn_entropy_per_head_max`
- `head_correlation_mean`

### Holonomy
**Note (from the headers list comment):** holonomy columns moved to a dedicated CSV — `PublicationMetrics.holonomy_csv_path` — and are *not* in the main per-step CSV. See "Holonomy diagnostics" section below for the dedicated CSV structure.

### Sibling per-layer and per-iteration CSVs
- `LayerDiagnosticsCSV.HEADERS` (line 302): per-layer-per-step. Columns include `mu_input_norm`, `mu_output_norm`, `delta_mu_norm`, `delta_mu_relative`, ... (read the source for the full list).
- `IterationDiagnosticsCSV.HEADERS` (line 329): per-VFE-iteration-per-layer-per-step. Columns include `grad_mu_norm`, `grad_sigma_norm`, `nat_grad_mu_norm`, `nat_grad_mu_raw_norm`, ... (read the source).

The `MetricsTracker` also writes a sibling JSON with `git_commit`, GPU info, and PyTorch version at run start — **use this for reproducibility tracing.**

## Validation / evaluation metrics

Source: `MetricsTracker.record_validation`, `PublicationMetrics::record_validation` (`transformer/analysis/publication_metrics.py:1154`). Typically: `val_loss`, `val_ce`, `val_ppl`, `val_bpc`, plus held-out test versions.

## Holonomy diagnostics — `transformer/analysis/holonomy_metrics.py`

`HolonomySnapshot` (per layer, per step, per head; both `'raw'` and `'scaled'` variants by cocycle_relaxation factor):

| Field | Meaning |
|---|---|
| `mean_norm`, `std_norm`, `median_norm`, `max_norm` | `‖C_ijk − I‖_F` summary stats over sampled (i,j,k) triples. `C_ijk = Ω_ik · (Ω_ij · Ω_jk)⁻¹` measures the holonomy around the triangle. |
| `frac_gt_001`, `frac_gt_01` | Fractions of triples with holonomy norm above 0.01 / 0.1. **Flatness check** — `[Nakahara2003]` for the connection-flat-iff-zero-curvature equivalence. |
| `mean_spectral_gap` | Mean σ₁ − σ₂ of `C_ijk`. |
| `mean_wilson_trace` | Mean `|tr(C_ijk)| / K`. Standard Wilson loop diagnostic from lattice gauge theory. |
| `nan_fraction` | Fraction of triples with non-finite entries — typically from matrix_exp instability when delta spectral radius exceeds float precision threshold. **If high, run is numerically unstable.** |
| `delta_max_spec`, `delta_p95_spec`, `delta_mean_spec` | Spectral norm of the raw `delta_matrix` across all N² (i,j) edges. **Governs matrix_exp stability.** |

`HolonomyProfile` (`holonomy_metrics.py:101`) aggregates snapshots over training.

`compute_curvature_by_distance` and `compute_flatness_trajectory` track how curvature evolves with sequence distance and over training steps.

## Gauge-geometry diagnostics — `transformer/analysis/gauge_geometry.py`

| Function | Meaning | Standard reference |
|---|---|---|
| `compute_yang_mills_energy(F)` | YM action `∫ tr(F ∧ *F)` on the curvature tensor | [Bleecker1981] for gauge theory and variational principles; standard YM action in any gauge theory text |
| `compute_yang_mills_energy_from_holonomy` | YM-equivalent from holonomy measurements | Same; small-loop limit of holonomy = curvature |
| `compute_yang_mills_energy_from_omega` | YM from Ω directly | Same |
| `compute_gauge_field_energy` | Total gauge-field energy | [Bleecker1981] |
| `compute_gauge_invariants` | Gauge-invariant scalars (traces of powers of F, etc.) | [Nakahara2003] for invariant theory |
| `compute_gauge_orbit_dimension` | Effective dimension of the gauge orbit | — |

## Fiber-trajectory diagnostics — `transformer/analysis/fiber_trajectory.py`

| Function | Meaning |
|---|---|
| `compute_arc_length` | Length of belief trajectory through the statistical manifold |
| `compute_velocity_profile` | Trajectory velocity over E-step iterations |
| `compute_convergence_curve` | E-step convergence (proxy for fixed-point quality, relevant for IFT [BaiKolterKoltun2019]) |
| `compute_geodesic_deviation` | Departure of the trajectory from the Fisher-Riemannian geodesic. **Standard:** large deviation = trajectory not on the natural-gradient path (`[Amari1998]`) |
| `FiberTrajectoryStats` | Summary statistics dataclass |

## Trajectory / per-layer recording — `transformer/analysis/trajectory.py`

`TrajectoryRecorder` records `embeddings`, `layer_input`, `attention`, `layer_output` per forward pass.

`LayerTrajectory` and `ForwardTrajectory` are the container dataclasses.

## Semantic clustering — `transformer/analysis/semantics.py`

| Function | Meaning |
|---|---|
| `compute_clustering_metrics` | k-means or similar on gauge frames; returns silhouette, ARI, NMI, etc. |
| `compute_semantic_field_coherence` | How aligned gauge frames are within semantic clusters |
| `compute_omega_clustering_metrics` | Similar but on transport operators Ω |
| `analyze_gauge_semantics`, `analyze_omega_semantics`, `analyze_sigma_semantics` | Per-quantity semantic analyses |
| `analyze_holonomy_semantic_correlation` | Cross-correlation between holonomy and semantic structure |
| `SemanticTrajectoryTracker` | Records semantic-structure evolution over training |

## Scaling-law fits — `transformer/analysis/scaling_stats.py`

`PowerLawFit` dataclass:
- Fits `PPL(x) = a · x^b + c` on per-axis seed means.
- Returns `a, b, c`, bootstrap CIs (`a_ci, b_ci, c_ci`), `r_squared`, `axis_grid`, `pred_*` dense arrays for ribbon plotting.
- Bootstrap: resamples seeds *within* each axis value (preserves design; doesn't conflate seed noise with axis effects).
- Log-log linear fallback when nonlinear fit fails.

**Standard reference:** [Kaplan2020] and [Hoffmann2022] for the scaling-law framework; the specific form `a·x^b + c` is standard for "irreducible loss" parameterizations.

## Bayesian validation — `transformer/analysis/bayesian_validation.py`

`ValidationData` dataclass and machinery for Bayesian uncertainty quantification on validation results. Standard reference: [Gelman et al. *Bayesian Data Analysis*] for the general framework; the `pymc` skill in this project for the implementation.

## Publication aggregator — `transformer/analysis/publication_metrics.py`

`PublicationMetrics` (line 1051) — the top-level class. Methods:
- `record_step` / `record_training_step` — per-step recording.
- `record_validation` — eval recording.
- `compute_holonomy_diagnostics` (line 1323) — holonomy snapshot wrapper.
- `compute_gauge_geometry_diagnostics` (line 1677) — gauge-geometry wrapper.
- `compute_fiber_trajectory_diagnostics` (line 1958) — fiber-trajectory wrapper.

`PublicationFigures` (line 371) renders the standard figure set: training curves, attention heatmaps, model comparison bars, scaling-study curves, gauge-frame clustering, attention entropy.

## What's "normal" vs "anomalous" — calibrated against external canon

These are the standard interpretive priors the agent should use. Numerical thresholds are heuristic and should be confirmed against the user's prior runs (`scaling_*` outputs, ablation logs) before flagging.

| Observation | Standard interpretation | Cite |
|---|---|---|
| `loss` not decreasing over many steps | E/M dynamics not converging; check LR schedule, gradient norm, init | — |
| `vfe_likelihood` decreasing but `vfe_coupling` increasing | Tradeoff between fitting data and matching cross-token priors; expected during warm-up | [Friston2010] for the F decomposition tradeoff |
| `attention_entropy` (or `attn_entropy_per_head_mean`) collapses to zero | β-distribution becomes peaked / deterministic; loses the regularizing entropy term | [Vaswani2017] (saturated softmax); also user's own entropy-regularization requirement |
| `attention_entropy` stays near max (≈ log N) | β is uniform; attention is doing nothing | Same |
| `grad_norm_total` explodes | Standard gradient instability; check LR, clipping, init | — |
| `nan_fraction` in HolonomySnapshot > 0.01 | matrix_exp instability — delta spectral radius too large; reduce E-step LR or add trust region | — |
| `phi_effective_rank` collapses (or `phi_rank_ratio` → 0, or `phi_top1_variance_fraction` → 1) | φ embedding collapses to a low-rank subspace; gauge frames lose representational diversity | [Amari1998] for natural-gradient pathology when Fisher singular; standard rank-collapse diagnostic in deep learning |
| Holonomy `frac_gt_01` not decreasing | Connection not approaching flat; if a flat-bundle claim is made, this is evidence against | [Nakahara2003] flatness ⇔ zero curvature ⇔ zero holonomy on contractible loops |
| Scaling exponent `b` near −1 with high R² | Standard 1/K scaling; consistent with `Participatory_it_from_bit.tex` finding | [Kaplan2020], [Hoffmann2022] for the general scaling-law framework |
| Convergence curve plateaus far above zero | E-step not reaching fixed point; IFT claims invalid | [BaiKolterKoltun2019] requires convergence to a fixed point |
| Geodesic deviation large | Trajectory off the natural-gradient path; consider preconditioner sanity check | [Amari1998] |
| Yang-Mills energy non-zero in a flat-bundle claim | Curvature is present; flat-bundle claim wrong | [Bleecker1981] |
