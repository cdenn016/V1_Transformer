# Claim — semantic-clustering-correctness

**Mode:** code (with math/theory sub-axes on geometric soundness)
**Rounds:** 2
**Judge:** on (panel=full — 3 first-pass judges + chief reconciler + canon-cop)
**Panel:** full
**Evidence scope:** auto (transformer/vfe/semantic_clustering/ + the active run dir)
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge\

## Claim

The two phenomena the user flagged in the VFE semantic-clustering module
(`transformer/vfe/semantic_clustering/`) are correct-by-design, not implementation
bugs — (1) the visual divergence between `phi_vector_clustering` and `omega_clustering`
despite `Omega = exp(sum_c phi_c G_c)` is a projection-layout artifact, and (2)
duplicate token labels in the *contextual* figures are the defined per-occurrence
semantics — while the module's genuine defects are confined to presentation /
completeness (undisclosed per-occurrence semantics; a byte-fallback-dominated vocab
view; array-order rather than salience-based label selection; an unguarded block
restriction that only breaks under a non-default toggle), none of which is a
correctness bug in the geometry math or the index alignment.

## Load-bearing sub-propositions

1. **(Geometry/correctness)** φ and Ω encode the *same* cluster structure: the
   φ-distance matrix (PCA-whitened Euclidean) and the Ω-geodesic distance matrix are
   ~0.96 Spearman-correlated on the real run; both independently pick k=2 with
   comparable silhouette; the generator bank is exactly block-diagonal so the two Ω
   code paths (`geometry.omega_geodesic_distances` block-restrict-then-exp vs
   `pipeline._per_token_omega` full-matrix exp) agree to ~1e-16. The residual visual
   divergence is UMAP orientation/reflection freedom + the metric choice (whitened
   Euclidean vs un-whitened affine-invariant geodesic), which together cost <0.05
   correlation even at 93% off-diagonal (non-commutative) generator energy.

2. **(Semantics/correctness)** Duplicate labels (`the`, `of`, `Atlantic`) in the
   contextual figures are correct-by-design: `extract_contextual` flattens
   `(B,N)->(B*N)`, one row per occurrence, so a recurring token type yields multiple
   legitimate rows with identical labels. Index alignment
   `strings[i] <-> coords[i] <-> token_ids[i]` is correct (no off-by-one).

3. **(Completeness — the conceded weaknesses)** The per-occurrence semantics are
   never disclosed in the figure; the vocab view is near-meaningless because the
   first 256 cl100k_base ids are byte-fallbacks (107→U+FFFD, 23→empty, only 72/200
   unique); `_MAX_ANNOTATIONS=30` label selection annotates the first 30 array-order
   points rather than salience/cluster-representative points; and the
   `omega_geodesic_distances` block restriction (geometry.py:337-340) is unguarded
   and would silently drop off-(super)block algebra under `auto_close_cross_head_basis=True`.

## What a RED win looks like

Red prevails if any of: (a) the φ/Ω divergence reflects a real metric/math error
(e.g. the whitening makes the comparison invalid, or the geodesic is mis-derived
against the canonical affine-invariant / log-Euclidean metric), (b) the duplicate
labels indicate a genuine data bug rather than per-occurrence semantics, (c) a
"completeness" item is actually a correctness/scientific-integrity defect severe
enough that calling the module "correct" is misleading, or (d) the index alignment /
block-restriction has a present (not merely latent) bug under the active config.

## User context

User runs a LOCALLY MODIFIED config (per CLAUDE.md "LOCALLY DEFINED CONFIGS"): the
active run is K=20 (embed_dim=20), irrep_dims=[10,10] (2 heads GL(10)),
diagonal_covariance=True, n_gen=200, tokenizer cl100k_base — NOT the repo CONFIG
default (embed_dim=200). User wants adversarial resolution of correctness, accuracy,
AND completeness. The user explicitly flagged the two phenomena and asked whether the
module is right.
