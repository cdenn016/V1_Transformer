# Evidence Pack — semantic-clustering-correctness

Neutral fact pack assembled by main Claude from direct code reads, the run's
metrics sidecars, and two investigation agents + one independent verifier agent
(all three loaded the real checkpoint `phi_embed` and re-ran the numerical checks).
No fact below depends on a docstring; each is a code path or an executed number.

## Active config (verified from run metrics + entry point)

Entry point: `transformer/vfe/run_semantic_clustering.py`. The repo CONFIG default
is `embed_dim=200, irrep_spec=[("fund",20,10)]`, but the ACTIVE RUN used a locally
modified config. From `vfe_runs/142.02=test-PPL_K=20_GL(10)/.../semantic_clustering/{vocab,contextual}/metrics.json`
`meta`: `K=20`, `irrep_dims=[10,10]`, `diagonal_covariance=true`, `n_tokens=200`,
`projection=umap`, `clustering=agglomerative`. From `system_info.json`:
`embed_dim=20`, `irrep_spec=[['fund',2,10]]`, `gauge_group='GLK'`, `cross_couplings=[]`.
So: 2 heads of GL(10), n_gen = 2·10² = 200. Checkpoint:
`vfe_runs/142.02=test-PPL_K=20_GL(10)/best_model.pt`. Tokenizer: `cl100k_base`
(tiktoken). The contextual figure labels (Atlantic, lobster, claws, eggs, American,
eastern) indicate a wikitext-103 passage about Atlantic lobsters; the exact text is
NOT persisted to the run dir.

## Code references

- `geometry.py:52-107` — `mu_distances`: euclidean / mahalanobis (global avg-cov whitener `sbar=sig.mean(0)`).
- `geometry.py:110-212` — `sigma_distances`: Bhattacharyya (default) / log-Euclidean.
- `geometry.py:215-270` — `phi_vector_distances`: **PCA-whitened** Euclidean on raw φ (`whiten=True` default; divides each retained PC by its singular value; `max_comps=50`).
- `geometry.py:285-357` — `omega_geodesic_distances`: per-head `A_h = A_full[a:b,a:b]`, `Ω_h=expm(A_h)`, geodesic `d_h=||log(Ω_i⁻¹Ω_j)||_F`, total `sqrt(Σ_h d_h²)`; chordal `||Ω_i-Ω_j||_F` fallback on non-finite logm.
- `geometry.py:337-340` — the block-restriction `A_full[a:b,a:b]` then `expm`. **Unguarded** (no assertion off-block is zero).
- `pipeline.py:111-122` — `_per_token_omega`: exponentiates the FULL K×K `A` (used only for the `||Ω-I||_F` / `det(Ω)` metrics, not the distance matrix).
- `pipeline.py:62-80` — `_subsample`: reindexes mu/sigma/phi/token_ids (and token_strings if present) with one shared `idx`, seed-fixed.
- `pipeline.py:88-93` — `_sanitize_label`: strips only C0 (U+0000–001F) + DEL (U+007F); does NOT strip whitespace or U+FFFD.
- `pipeline.py:96-108` — `_decode_strings`: per-id `dataset.decode([i])`, run once after subsample (line 182).
- `pipeline.py:136-240` — `run_clustering`: same `strings` list passed to all four plot calls (193, 202, 210, 222).
- `extract.py:79-121` — `extract_contextual`: `mu=...reshape(-1,K)` flatten `(B,N)->(B·N)` (line 98); ids flattened in same order (line 106). One row per occurrence.
- `extract.py:124-187` — `extract_vocab`: `token_ids=arange(min(vocab_size,max_tokens))`; reads `prior_bank.encode(ids)` (dispatches both gauge_fixed_priors paths). One row per id.
- `extract.py:117,183` — bundle `irrep_dims = list(cfg.effective_block_dims)`.
- `plotting.py:103` — `_MAX_ANNOTATIONS = 30`.
- `plotting.py:224-235` — annotates `range(min(len(token_strings), coords.shape[0], 30))` = first 30 ARRAY-ORDER points; `xytext=(2,2)` offset.
- `clustering.py:21-159` — agglomerative average-linkage on precomputed D; auto-k sweeps `range(2,9)` maximizing silhouette (`metric="precomputed"`); small-n guard n<4 → single cluster.
- `projection.py:34-173` — UMAP(`metric="precomputed"`) primary → t-SNE → MDS fallback; tiny-n guard.
- `metrics.py:214-291` — `phi_metrics`: diag-vs-offdiag energy partition; optional Ω summaries (`||Ω-I||_F`, `det Ω`).
- `transformer/vfe/model.py:419-491` — generator-bank selection; single multihead spec + empty cross_couplings → `generate_glK_multihead_generators(20,2)`; `auto_close_cross_head_basis=True` warned at 447-454 (can add generators spanning super-blocks).
- `transformer/vfe/config.py:883-902` — `effective_block_dims` returns `super_block_dims` under cross_couplings, else `irrep_dims`.
- `math_utils/generators.py:870-953` — `generate_glK_multihead_generators`: block-diagonal multihead glK bank.

## Executed numerical evidence (investigator A + independent verifier, real checkpoint φ)

- **Generator block-diagonality:** max off-block |entry| of the 200-generator bank = **0.0 (exact)**; 0 generators with cross-block support.
- **Ω path consistency:** `max | expm(A_full) − blockdiag(expm(A_h)) | ≈ 4.4e-16` over random φ at scale 0.05–0.3. The two Ω code paths are numerically identical under this config.
- **D_φ (whitened) vs D_Ω Spearman (upper triangle, n=200):** **0.952** (first-200 ids) and **0.972** (random-200 vocab types). Both p≈0.
  - D_φ_unwhitened vs D_Ω = 0.9696 (isolates exp/BCH nonlinearity → ~0.03 cost at 93% off-diag energy).
  - D_φ_whitened vs D_φ_unwhitened = 0.9849 (isolates whitening → ~0.005 cost).
- **Reproduced** `omega_dist_from_identity_mean` ≈ 2.39 vs metrics.json 2.43 (gap = raw `phi_embed` vs `encode`-applied trace control).
- **tiktoken cl100k_base, exact subsample** `np.sort(np.random.default_rng(0).choice(256,200,replace=False))`, decode each id individually + `_sanitize_label`: **107 → U+FFFD ('�'), 23 → '' (empty), only 72/200 unique strings, 130/200 are duplicates.** First-30 annotated (sorted, lowest ids) decode to single ASCII punctuation/digit/letter fragments (`! " # $ % ( * + , - 0 1 …`), all distinct within that set.
- **Index alignment** `strings[i] ↔ coords[i] ↔ token_ids[i]`: verified correct, no off-by-one (subsample reindexes all arrays with one idx; decode once after subsample; one strings list to all four plots; `project()` preserves row order).

## Metrics sidecar headline numbers

- vocab: φ silhouette 0.282 / Ω 0.260; both k=2. mu k=2 (0.245); sigma k=2 (0.533). Ω det_mean 1.40, ||Ω-I|| 2.43.
- contextual: φ silhouette 0.351 / Ω 0.299; both k=2. mu k=5 (0.253); sigma k=5 (0.385). Ω det_mean 245, ||Ω-I|| 6.15.

## Canon excerpts to anchor the geometric sub-axes

- Affine-invariant SPD geodesic: `[Pennec, Fillard, Ayache 2006, IJCV "A Riemannian Framework for Tensor Computing"]`; log-Euclidean metric: `[Arsigny, Fillard, Pennec, Ayache 2006/2007 SIAM J. Matrix Anal.]`.
- Bi-invariant / canonical metric on matrix Lie groups and `||log(g⁻¹h)||`: `[Nakahara 2003 §5–6]`; matrix-group geodesics.
- BCH expansion `log(exp(-A)exp(B)) = B−A + ½[−A,B]+…`: `[Hall, "Lie Groups, Lie Algebras, and Representations" 2015 §5]`.
- PCA whitening / Mahalanobis whitening: `[Bishop PRML 2006 §12.1]`; whitening is a metric (re)choice, not distance-preserving.
- UMAP is not distance/orientation preserving (embeddings are rotation/reflection-free; global structure not metrically faithful): `[McInnes, Healy, Melville 2018 arXiv:1802.03426]`.
- Silhouette on a precomputed non-metric dissimilarity is heuristic: `[Rousseeuw 1987 J. Comput. Appl. Math.]`; Bhattacharyya is non-metric.
- Effective rank as exp-entropy of normalized spectrum: `[Roy & Vetterli 2007, "The effective rank"]`.

## What this evidence does NOT settle

- Whether PCA-whitening on φ is a *sound default* or makes the φ panel not the algebra-image of the Ω panel (a soundness/purity judgment, not just a number).
- Whether shipping a byte-fallback-dominated vocab view is a *completeness/scientific-integrity defect* vs a benign cosmetic limitation.
- Whether "correct-by-design" is an adequate defense when the design (per-occurrence + no disclosure) predictably misleads a viewer.
- Whether the unguarded block-restriction warrants a runtime guard given it only breaks under a non-default opt-in toggle.
- Whether the geodesic `||log(Ω_i⁻¹Ω_j)||_F` is the *correct* distance for GL⁺(K) elements (vs an affine-invariant or other canonical metric), and whether quadrature-summing per-head geodesics is justified.
