# Blue Panel Choice — semantic-clustering-correctness (Phase 2 opening)

Mode: code (with math/theory sub-axes on geometric soundness). Panel of 5, philosophy-of-science mandatory.

| Expert | Why this lens defends the claim |
|--------|---------------------------------|
| `philosophy-of-science` (mandatory) | Frame-checks whether "correct-by-design" is a legitimate defense vs an equivocation, and polices the claim's bundling of correctness + completeness so the defense does not smuggle the manuscript in as authority. |
| `gauge-theorist` | Strongest defender: establishes that a block-diagonal (direct-product) generator bank makes block-restrict-then-exp identical to full-exp-then-restrict, and that `‖log(Ω_i⁻¹Ω_j)‖_F` is the standard left-invariant geodesic with per-head quadrature justified by the direct-product structure [Nakahara2003, Hall2015]. |
| `info-geometer` | Owns the weakest link: whether PCA-whitening on φ is a sound metric choice (Cencov/Fisher), plus the Mahalanobis-μ and non-metric-Bhattacharyya questions. Instructed to concede if whitening fails Cencov-invariance. |
| `ml-engineer` | Owns the projection-artifact argument: UMAP is not distance/orientation preserving [McInnes2018]; silhouette on a non-metric dissimilarity is heuristic [Rousseeuw1987]; the visual divergence is an embedding artifact, not a math bug. |
| `implementation-engineer` | Owns the active-config trace (K=20, irrep_dims=[10,10], cross_couplings=[]) and the latent-vs-active distinction: `extract_contextual` flatten at `extract.py:98` structurally guarantees per-occurrence duplicates; the unguarded block-restriction at `geometry.py:337-340` is latent under the active config. |

Discounted: `numerical-analyst` (the 4e-16 path-agreement is already executed in the evidence pack; the open question is interpretation, owned by gauge-theorist). `transformer-ml` (the module is post-hoc visualization, not attention/RoPE/normalization). `variational` (no ELBO/EM-boundary question in this claim). `code-quality` (design-smell concerns are exactly the conceded UX/completeness items, not the correctness load-bearing axis; folded into implementation-engineer).
