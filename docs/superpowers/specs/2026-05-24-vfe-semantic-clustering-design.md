# Standalone /vfe Semantic-Clustering Visualization Module — Design

Date: 2026-05-24
Status: approved (design); implementation pending
Author: pair (user + Claude)

## Goal

A standalone visualization + metrics module for the `transformer/vfe/` package that
clusters per-token belief geometry and emits **separate** publication-quality images for
the belief mean `mu`, the belief covariance `Sigma`, and the gauge frame `phi` (plus the
group element `Omega = exp(phi·G)`), together with a machine-readable metrics sidecar.

It must NOT be ported into or coupled to the legacy `transformer/analysis/semantics.py`
(3180-line, GPT-2-coupled, `transformer/core/`-coupled, not wired into the vfe path). The
new module reads `transformer/vfe/` attributes directly and re-derives the handful of
geometric primitives it needs.

## Approved decisions (from brainstorming)

1. **Cluster source: BOTH** — contextual (post-E-step beliefs from a text sample) AND
   vocab-level (per-token-type encode-bank embeddings), produced as separate runs/outputs.
2. **phi representation: BOTH** — the flat coefficient vector (`phi ∈ R^n_gen`) AND the
   intrinsic group element `Omega = exp(phi·G) ∈ GL(K)`.
3. **Distance geometry: geometry-faithful** — manifold-aware distances for each quantity,
   feeding precomputed-distance projection/clustering.
4. **DR backend: install `umap-learn` as primary**, with sklearn t-SNE/MDS fallback
   (try-guarded import so absence degrades gracefully).

Defaults (not separately asked, flag if wrong):
- Cluster coloring = unsupervised discovered clusters; tokenizer-aware category overlay is
  optional and OFF by default (the legacy GPT-2 0–255 categorizer is not reused).
- Orchestrator loads a trained checkpoint (`CONFIG['checkpoint_path']`); a fresh-model
  fallback exists only for pipeline smoke-testing.

## Verified factual basis (investigators + verifier, 2026-05-24)

- `BeliefState` (`transformer/core/types.py:11-33`): NamedTuple `(mu, sigma, phi, omega)`.
  Shapes: `mu (B,N,K)`; `sigma (B,N,K)` diagonal or `(B,N,K,K)` full per
  `cfg.diagonal_covariance` (`config.py:142`); `phi (B,N,n_gen)`.
- Final-layer beliefs: `model.forward_with_beliefs(token_ids) -> (logits, BeliefState)`
  (`vfe/model.py:203-225`). Returned `mu` is post-final-norm; `sigma`/`phi` are converged
  E-step values. No `targets` parameter (Law 1 safe).
- Per-layer beliefs: `block.e_step._last_attention_state` dict with keys `mu_q, sigma_q,
  phi, ...` written on the last E-step iteration when `_capture_attention_state` (default
  `True`, `e_step.py:392`) is set. `sigma_q` is always reduced to `(B,N,K)` diagonal
  (conditional `.diagonal()` only on the full-cov branch; outcome is diagonal regardless).
  The `omega_direct` branch does NOT populate this cache.
- Legacy `transformer/analysis/semantics.py` is NOT imported anywhere in `transformer/vfe/`
  — a standalone module has no collision risk.
- Reusable primitives to re-derive (NOT import): Bhattacharyya distance
  (`analysis/semantics.py:2396`), Omega geodesic `‖logm(Ω_i⁻¹Ω_j)‖_F`
  (`analysis/semantics.py:1632`), diagonal Fisher-Rao (`analysis/fiber_trajectory.py:115`),
  effective rank `exp(H(λ/Σλ))`.
- Installed: scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8, seaborn 0.13.2,
  numpy 2.4.4, torch 2.11.0, networkx 3.6.1. NOT installed: umap-learn, plotly, shap,
  kaleido, hdbscan. → must `pip install umap-learn`; everything else available; HDBSCAN
  available via `sklearn.cluster.HDBSCAN` (no standalone pkg needed).
- House style: `transformer/visualization/pub_style.py` exports `set_pub_style`,
  `PUB_COLORS`, `PUB_CYCLE` (Okabe-Ito, savefig dpi=300, bbox tight). Convention: save both
  PDF and PNG@300; output to `<checkpoint_dir>/<subdir>` with `./outputs/<subdir>` fallback;
  click-to-run CONFIG (no CLI).
- ACTIVE user config (`vfe/train_vfe.py`): `embed_dim=200`, `irrep_spec=[('fund',20,10)]`
  (K=200, 20 heads × dim 10, n_gen=2000), `diagonal_covariance=True`, `n_layers=1`,
  `use_prior_bank=False`, `gauge_parameterization` absent → `'phi'` (φ-mode), `em_mode` is
  not a vfe field. So today: one layer, diagonal Sigma `(B,N,200)`, φ-mode (cache populated).

## Architecture — package `transformer/vfe/semantic_clustering/`

One file per concern (small, independently testable units). Orchestrator at
`transformer/vfe/run_semantic_clustering.py` (click-to-run).

```
transformer/vfe/semantic_clustering/
  __init__.py        # public API: BeliefBundle, run_clustering, the per-quantity plots
  bundle.py          # BeliefBundle dataclass — the shared data contract
  extract.py         # contextual + vocab-level extraction → BeliefBundle
  geometry.py        # distance matrices: mu, sigma (Bhattacharyya/log-Euclidean), phi-vec, Omega-geodesic
  projection.py      # project(D|X, method) → 2D/3D coords; UMAP primary, t-SNE/MDS fallback
  clustering.py      # unsupervised labels + auto-k (silhouette/CH sweep)
  metrics.py         # common + per-quantity metrics → dict
  plotting.py        # plot_mu / plot_sigma / plot_phi_vector / plot_omega → PDF+PNG@300
transformer/vfe/run_semantic_clustering.py   # CONFIG dict orchestrator (no CLI)
tests/transformer/vfe/
  test_semantic_clustering_geometry.py
  test_semantic_clustering_extract.py
  test_semantic_clustering_metrics.py
  test_semantic_clustering_smoke.py
```

### Data contract — `BeliefBundle` (bundle.py)
```
@dataclass
class BeliefBundle:
    mu: Tensor          # (n, K)
    sigma: Tensor       # (n, K) diagonal OR (n, K, K) full
    phi: Tensor         # (n, n_gen)
    token_ids: Tensor   # (n,)
    token_strings: list[str] | None
    generators: Tensor | None   # (n_gen, K, K) or per-head block list, for Omega reconstruction
    irrep_dims: list[int]       # per-head dims, e.g. [10]*20 — for block-diagonal Omega
    source: str         # 'contextual' | 'vocab'
    layer: int | str    # 'final' | int
    diagonal: bool
```

### Unit responsibilities

**extract.py**
- `extract_contextual(model, token_ids, layer='final') -> BeliefBundle`. No targets passed
  (Law 1). `layer='final'` → `forward_with_beliefs`; `layer=int` → read
  `model.stack.blocks[i].e_step._last_attention_state` after setting
  `_capture_attention_state=True`. Warn if `cfg.gauge_parameterization=='omega_direct'`.
- `extract_vocab(model, token_ids=None) -> BeliefBundle`. Read encode bank: `mu_embed`,
  `sigma_log_embed`/`base_log_sigma` → sigma, `phi_embed`. One row per token type (or a
  supplied subset). Works under `use_prior_bank=False`.
- Token strings via the dataset/tokenizer when available; else None (do not hard-couple to
  GPT-2).

**geometry.py** (all return a symmetric `(n,n)` distance matrix, zero diagonal)
- `mu_distances(mu, sigma=None, metric)` — `'euclidean'` | `'mahalanobis'` (Σ̄-whitened).
- `sigma_distances(sigma, mu=None, metric)` — `'bhattacharyya'` (joint Gaussian) |
  `'logeuclidean'` (SPD). Fast diagonal path AND full-cov `logm` path; both tested.
- `phi_vector_distances(phi, whiten=True)` — Euclidean, PCA-whitened first (R^2000 regime).
- `omega_geodesic_distances(phi, generators, irrep_dims)` — per-head
  `Ω_h = expm(φ_h·G_h)`, `d² = Σ_h ‖logm(Ω_{h,i}⁻¹ Ω_{h,j})‖_F²`. Block-diagonal, per-head
  10×10 in active config.

**projection.py**
- `project(matrix, method='umap', n_components=2, precomputed=True)`. UMAP
  (`metric='precomputed'`) primary; fall back to `TSNE(metric='precomputed')` then
  `MDS(dissimilarity='precomputed')` if umap import fails. Euclidean feature path may use PCA.

**clustering.py**
- `cluster(matrix, method='agglomerative', precomputed=True, k='auto')`. Auto-k by
  silhouette/Calinski sweep over a small k range. HDBSCAN optional via sklearn.

**metrics.py**
- `common_metrics(matrix_or_X, labels)` → silhouette, Calinski-Harabasz, Davies-Bouldin,
  inter/intra ratio, PCA variance profile (feature path). Plain keys (no `embed_name` prefix).
- `sigma_metrics(sigma)` → per-token effective rank `exp(H(λ/Σλ))`, logdet, trace, anisotropy.
- `phi_metrics(phi, omega?, irrep_dims)` → energy partition (diag vs cross-coupling
  generators), `‖Ω−I‖_F`, `det Ω`, eigenvalue spectrum summary.
- `mu_metrics(mu)` → norm distribution summary.

**plotting.py** — four separate figures, each PDF+PNG@300 via `set_pub_style()`:
- `plot_mu_clustering`, `plot_sigma_clustering`, `plot_phi_vector_clustering`,
  `plot_omega_clustering`. 2D scatter (3D optional) colored by discovered cluster, envelope
  overlays, metrics annotation box. Local minimal-style fallback if pub_style import fails.

**run_semantic_clustering.py** — CONFIG dict: checkpoint_path, corpus sample
(file/n_tokens/seq_len), layer, methods, output_dir, do_contextual, do_vocab. Loads model,
builds the (optional) text sample loader, runs requested views, writes images + metrics.

### Outputs
```
<checkpoint_dir>/semantic_clustering/
  contextual/  mu_clustering.{pdf,png}  sigma_clustering.{pdf,png}
               phi_vector_clustering.{pdf,png}  omega_clustering.{pdf,png}  metrics.{json,csv}
  vocab/       (same set)
```
Fallback root: `./outputs/semantic_clustering/`.

## Constraints honored
- NO neural-network components added (pure analysis/visualization).
- NO CLI arguments — click-to-run CONFIG dict.
- Gauge-correct `Omega = exp(phi·G)` reconstruction; geodesic via `logm`. (No covariance
  transport occurs here, so the sandwich product is not invoked.)
- E-step must not see targets — contextual extraction passes no targets.
- Runtime config verification: load `cfg` from checkpoint; warn on `omega_direct`
  (empty per-layer cache) and report `diagonal_covariance`.
- Both diagonal and full-cov Sigma distance paths implemented and tested.
- Designed for general `n_layers`; defaults to final layer.
- Standalone: no `transformer.analysis.semantics` import.
- Post-edit `.md` log under `docs/audits/` (one per day, updated as edits land).

## Testing (TDD — tests before implementation)
- geometry: distance matrices symmetric, zero diagonal, non-negative; diagonal Σ path
  agrees with full-cov path when the full matrix is diagonal; Omega geodesic equals
  quadrature of per-head geodesics; identical inputs → zero distance.
- extract: returned shapes match `cfg` (mu (n,K), sigma (n,K) under active diagonal config,
  phi (n,n_gen)); contextual path passes no targets.
- metrics: silhouette ∈ [-1,1]; effective rank ∈ [1,K]; energy partition fractions sum to 1.
- smoke: fresh small model → full pipeline runs end-to-end, four PDFs+PNGs and metrics
  files produced for both views, no exceptions, no NaNs in metrics.

## Out of scope (YAGNI)
- The broken `analyze_holonomy_semantic_correlation` (legacy) — not ported.
- Plotly/HTML interactive views, SHAP attribution (deps not installed; not requested).
- Training-time live trajectory tracking (separate concern from this static analysis).
