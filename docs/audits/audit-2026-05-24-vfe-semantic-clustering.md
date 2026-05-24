# Post-Edit Audit — 2026-05-24 — VFE Semantic-Clustering Module

Standalone semantic-clustering visualization package for the `transformer/vfe/`
implementation. Clusters per-token belief geometry and emits separate
publication-quality figures for `mu`, `Sigma`, the `phi` coefficient vector, and
the transport `Omega = exp(phi.G)`, for both contextual and vocab-level views,
plus a metrics sidecar. Built brainstorm → spec → plan → parallel TDD build.

Spec: `docs/superpowers/specs/2026-05-24-vfe-semantic-clustering-design.md`
Plan: `docs/superpowers/plans/2026-05-24-vfe-semantic-clustering.md`

## Files created

Package `transformer/vfe/semantic_clustering/`:
- `__init__.py` — public API (BeliefBundle, extract_contextual, extract_vocab, run_clustering, four plot fns).
- `bundle.py` — `BeliefBundle` dataclass (the shared data contract).
- `extract.py` — `extract_contextual` / `extract_vocab`.
- `geometry.py` — `mu_distances`, `sigma_distances`, `phi_vector_distances`, `omega_geodesic_distances`.
- `projection.py` — `project` (UMAP-primary, t-SNE/MDS fallback, precomputed).
- `clustering.py` — `cluster` (agglomerative + silhouette-swept auto-k; optional HDBSCAN).
- `metrics.py` — `common_metrics`, `sigma_metrics`, `phi_metrics`, `mu_metrics`.
- `plotting.py` — four separate publication figures (PDF + PNG@300).
- `pipeline.py` — `run_clustering` (one-view orchestration; writes metrics.json/csv).

Entry point:
- `transformer/vfe/run_semantic_clustering.py` — click-to-run CONFIG dict, no CLI args.

Tests `tests/transformer/vfe/`:
- `test_semantic_clustering_bundle.py` (1), `_geometry.py` (8), `_extract.py` (2),
  `_projection.py` (2), `_clustering.py` (2), `_metrics.py` (3), `_plotting.py` (1),
  `_smoke.py` (2). Total: 21 tests, all passing. (Geometry gained two full-cov
  regression tests — Bhattacharyya-full and Mahalanobis-full diag-vs-full
  agreement — after the verifier flagged the full-cov paths lacked committed
  coverage.)

## Active-config trace (CLAUDE.md runtime-reachability requirement)

`transformer/vfe/train_vfe.py` active values, verified at runtime:
`embed_dim=200`, `irrep_spec=[('fund',20,10)]` → K=200, n_gen=2000, 20 heads of
dim 10 (`effective_block_dims=[10]*20`), `diagonal_covariance=True`, `n_layers=1`,
`use_prior_bank=False`, `gauge_parameterization='phi'` (default; the per-layer
E-step cache is populated — no `omega_direct` warning fires).

Ran `run_clustering(model, source='vocab', max_points=24)` against a model built
with this exact config: all 10 artifacts produced, no NaNs, `sigma`
effective_rank_mean = 196.18 ∈ [1, 200], `phi` energy fractions sum to 1.000000,
`omega` mean ||Ω−I||_F = 4.545. The library is reachable and correct under the
user's active config, not just the tiny smoke model.

## Key technical decisions / deviations

- **Generator-bank provenance (gauge-correctness):** `extract.py` stores
  `model.generators` (registered at `prior_bank.py:132` / `model.py:82` — the
  exact tensor the bank exponentiates in `_compute_block_exp_pairs`), verified
  bit-equal to the bank buffer. `geometry.omega_geodesic_distances` rebuilds
  per-head `Ω_h = expm(A_full[block_h])` from this bank, so the Ω geodesic
  reproduces the model's own transport. No basis invented.
- **`extract_vocab` uses `pb.encode(ids)`** rather than reading raw `mu_embed`/
  `sigma_log_embed`/`phi_embed`. The `VFEConfig` default `gauge_fixed_priors=True`
  has no such attributes (the gauge-orbit prior is built differently); `encode`
  dispatches both parameterizations and returns the exact per-token prior,
  including φ determinant-control. (The active train_vfe.py config uses
  `gauge_fixed_priors=False`, the direct-lookup mode — `encode` handles both.)
- **`metrics.phi_metrics` does NOT reuse `cross_coupling_metrics.phi_energy_partition`.**
  That function splits ||φ||² between per-head GL blocks and *inter-head*
  cross-coupling blocks (keyed off `cfg.cross_couplings`); the new partition is
  *within* each gl(d) block between diagonal `E_aa` and off-diagonal `E_ab`
  generators. Different question, incompatible signatures. The generator index
  layout (`c = i·d + j`, diagonal indices `{0,3,4,7}` for `[2,2]`) was verified
  against `math_utils/generators.py::generate_glK_multihead_generators`.
- **Cross-module metric-key reconciliation (integration fix):** `plotting.py`
  originally read `mean_norm` (μ headline) and `mean_omega_dist_from_identity`
  (Ω headline), but `metrics.py` produces `norm_mean` and
  `omega_dist_from_identity_mean`. Aligned the plotting consumer to the metrics
  producer. Caught at integration; neither module's own tests covered the
  cross-module key contract.
- **`plotting.py` backend neutrality:** importing `pub_style` transitively runs
  `transformer/visualization/__init__.py`, which calls `matplotlib.use('Agg')`
  at import. `plotting.py` snapshots and restores the active backend around the
  import so merely importing the module is backend-neutral.
- **Non-metric dissimilarity:** Bhattacharyya is not a metric. UMAP precomputed /
  MDS / average-linkage agglomerative accept it; the t-SNE fallback branch is
  wrapped in try/except and degrades to MDS.
- **Cost control:** `run_clustering(max_points=200)` subsamples tokens because the
  Ω geodesic is O(n²·heads·logm) and Bhattacharyya is O(n²).

## Environment

`pip install umap-learn>=0.5.0` → umap-learn 0.5.12 (numba 0.65.1, llvmlite
0.47.0) installed cleanly on Python 3.14 / numpy 2.4.4. Primary projection path
is UMAP (precomputed); sklearn t-SNE/MDS are the guarded fallbacks.

## Constraints honored

No neural-network components; no CLI args (click-to-run CONFIG); gauge-correct Ω
reconstruction from the model's own generator bank; E-step never sees targets
(no `targets` passed in either extraction path); standalone (no
`transformer.analysis.semantics` import); diagonal **and** full-cov Σ paths
implemented and tested; designed for general `n_layers` (defaults to final).

## Test result

`python -m pytest tests/transformer/vfe/test_semantic_clustering_*.py -q` →
**21 passed**.

## Wired into train_vfe.py (post-training hook)

`transformer/vfe/train_vfe.py` now runs the module automatically at the end of a
training run. Added a module-level toggle `RUN_SEMANTIC_CLUSTERING = True`
(+ `SEMANTIC_CLUSTERING_MAX_POINTS = 200`) and a testable helper
`run_post_training_semantic_clustering(model, output_dir, loader, dataset, ...)`.
After `trainer.train()` and the final val eval, the `__main__` block calls the
helper inside a `try/except` (non-fatal — a viz failure never breaks a completed
run). It writes the vocab view (learned per-type priors) and the contextual view
(one real `val_loader` batch, inputs only — Law 1) into
`<output_dir>/semantic_clustering/{vocab,contextual}/`. Token decode labels come
from `val_loader.dataset` when available. Regression test:
`tests/transformer/vfe/test_train_vfe_clustering_hook.py` (tiny model + fake tuple
loader → both views produce all four figures + metrics). Suite now **22 passed**.

## Independent verification

A separate verifier agent independently confirmed (read-only + numerical
checks): no legacy-semantics coupling; no NN / no CLI; Law 1 (no targets in
either extraction path); the Ω geodesic reconstruction matches the model's own
transport kernel `fused_block_matrix_exp_pairs` (`core/gauge_utils.py:312-318`)
to **max diff 1.98e-7**; Bhattacharyya diagonal-vs-full agreement to 4.4e-16;
effective rank bounded in [1,K]; and all four plot headline keys are present in
the dicts `pipeline.py` passes (headlines render, not silently omitted).
