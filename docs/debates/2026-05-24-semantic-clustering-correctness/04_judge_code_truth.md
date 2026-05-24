# Verdict (code-truth) — semantic-clustering-correctness

## My re-traced active config

Traced from `transformer/vfe/vfe_runs/142.02=test-PPL_K=20_GL(10)/system_info.json` (`config` block, the run that produced the figures) — NOT the repo `run_semantic_clustering.py:39-46` CONFIG default (`embed_dim=200`, `checkpoint_path=None`). The repo default is inert for this debate; the active run used a locally-modified config per the CLAUDE.md "LOCALLY DEFINED CONFIGS" caveat.

| Key | Active value (system_info.json) | Line |
|-----|--------------------------------|------|
| `embed_dim` (K) | 20 | :24 |
| `irrep_spec` | `[["fund", 2, 10]]` (2 heads × GL(10)) | :25-31 |
| `diagonal_covariance` | `true` | :53 |
| `cross_couplings` | `[]` (empty) | :82 |
| `auto_close_cross_head_basis` | `false` | :83 |
| `gauge_parameterization` | `"phi"` | :79 |
| `gauge_fixed_priors` | `false` | :74 |
| `use_prior_bank` | `false` | :89 |

Derived: `n_gen = n_heads · d_head² = 2 · 10² = 200`. Sidecar `vocab/metrics.json:54-61` and `contextual/metrics.json:54-61` confirm `K=20`, `irrep_dims=[10,10]`, `diagonal_covariance=true`, `n_tokens=200`. No divergence from the openings' trace. The active config matches what the dispatch stated.

## Reachability verification

| path:line | Cited by | Reachable under active config? | Notes |
|-----------|----------|--------------------------------|-------|
| `extract.py:98` flatten `(B,N,K)->(B·N,K)` (mu) | both | YES | `cfg.embed_dim=20`; `reshape(-1, K)`. ids flattened in same order at `extract.py:106` (`token_ids.reshape(-1)`). Same-order flatten verified. |
| `extract.py:106` ids reshape | blue | YES | One row per occurrence by construction; duplicates are structural. |
| `geometry.py:215-270` `phi_vector_distances`, `whiten=True` default | both | YES | Called from `pipeline.py:206`. SVD on subsampled centered φ (`:258`), divide each PC by its singular value (`:261-266`). PCA-whitening confirmed; data/seed-dependent. |
| `geometry.py:337-340` block-restrict-then-exp | both | YES (executed exactly) | Slices `A_full[a:b,a:b]` then `expm`, unguarded. Under active config the bank is exactly block-diagonal so the slice is exact (see executed numbers). |
| `geometry.py:349-352` `M=solve(Oi,Oj)`; `‖log M‖_F` | both | YES | GL⁺(K) left-invariant distance on group elements (invariant under Ω→AΩ), NOT the SPD-cone congruence distance. Code, not docstring, is canonical. |
| `geometry.py:290` "affine-invariant" docstring | red (opening, withdrawn) | n/a | Docstring drift. 0-weight per rubric. Red withdrew the Pennec attachment in rebuttal. |
| `pipeline.py:62-80` `_subsample` one shared `idx` | blue | YES | All arrays reindexed with one `idx`; alignment preserved. |
| `pipeline.py:88-93` `_sanitize_label` strips C0/DEL only | red, blue | YES | Does NOT strip U+FFFD or empty. Verified by code read. |
| `pipeline.py:111-122` `_per_token_omega` full-exp | both | YES | Exponentiates full K×K A; used for `‖Ω−I‖`/`det` metrics. Agrees with geometry path (executed). |
| `pipeline.py:182, 193/202/211/223` decode-once + one strings list | blue | YES | Decode runs after subsample on reindexed ids; one `strings` list to all four plots. |
| `plotting.py:226-227` `range(n_annot)`, `token_strings[i]`/`coords[i]` | both | YES | Array-order first-30 annotation, NOT salience. Confirmed by code read. |
| `model.py:485` `generate_glK_multihead_generators(20,2)` | blue | YES | `cross_couplings=[]` ⇒ `model.py:423` False ⇒ skip 424-484 ⇒ line 485 fires. |
| `model.py:438-455` `auto_close_cross_head_basis` branch | red (latent) | NO | Nested inside `if cfg.cross_couplings:` (`:423`). Doubly unreachable: `cross_couplings=[]` AND `auto_close=false`. The off-block-spanning generator path cannot fire under the active config. Defect is latent, not present. |

## Evidence audit

| Side | path:line (verified) | path:line (unverified) | Test outputs | External citations | Comment/docstring cites |
|------|----------------------|------------------------|--------------|--------------------|--------------------------|
| Red  | `geometry.py:215-270,261-266` (φ whitening); `geometry.py:337-340` (unguarded restrict); `geometry.py:349-352` (GL⁺(K) geodesic, post-concession); `pipeline.py:88-93` (no U+FFFD strip); `plotting.py:224-235` (array-order annot); `run_semantic_clustering.py:152-153` (vocab id range) | `model.py:447-454` cited as reachable — refuted (gated, unreachable under active config) | reproduced 107 U+FFFD / 23 empty / 72 unique (I re-ran: matches) | Pennec 2006 (withdrawn, −2 strikes); Amari 1998; McInnes 2018; Bishop 2006; Tufte 1983; Wilson 2017; Popper 1959; Hall 2015 | geometry.py:290 docstring (red's own attack on it; 0-weight) |
| Blue | `extract.py:98,106` (same-order flatten); `pipeline.py:62-80,182,193-223` (alignment); `geometry.py:339,355-356` (block-restrict + quadrature); `pipeline.py:119-121` (full-exp); `model.py:419-491` (generator selection); `config.py:893-902` (effective_block_dims=irrep_dims) | none material | 0.0 exact off-block; 4.44e-16 Ω-path agreement; 0.952–0.972 φ/Ω Spearman; both k=2 (I re-ran block-diag + path: matches) | Nakahara 2003; Hall 2015; Lee 2013; McInnes 2018; Rousseeuw 1987; Cencov 1972 | none as authority |

## Concessions made
- Red conceded: per-head quadrature is the exact product-manifold metric (not approximation); block-diagonal exp factors exactly so the two Ω paths agree structurally (4.4e-16); index alignment verified, no off-by-one, so contextual duplicate labels are structural per-occurrence semantics (falsification (b) dead); Pennec/Fillard/Ayache 2006 citation withdrawn (canon-cop's 2-strike wrong-domain ruling accepted).
- Blue conceded: the vocab figure's "completeness-not-correctness" framing is too strong — it is an accuracy defect on the vocab figure's presentation (byte-fallback labels + array-order annotation); the unguarded block-restriction (geometry.py:337-340) warrants a guard (latent/prophylactic).

## Decisive evidence

Executed under the re-traced active config, using the exact builder `generate_glK_multihead_generators(20, 2)` reached at `model.py:485`:
- bank off-block max |entry| = **0.0 (exact)**, 0 generators with cross-block support;
- `max | expm(A_full) − blockdiag(expm(A_h)) | = 4.440892098500626e-16` over random φ at scale 0.05–0.3.

Combined with the config trace at `system_info.json:82-83` (`cross_couplings=[]`, `auto_close_cross_head_basis=false`) and `model.py:423` (the cross-head/auto-close branch is gated on non-empty `cross_couplings`, so the off-block-spanning path at `model.py:438-455` is unreachable), this makes `geometry.py:339-340`'s block-restriction numerically exact under the active config. The geometry math (`omega_geodesic_distances` block-restrict-then-exp ≡ full-exp) and index alignment (`extract.py:98,106` same-order flatten + `pipeline.py:62-80,182,193-223` one shared idx / one strings list) are therefore correct at the code level under the active config.

Independently reproduced (tiktoken cl100k_base, `np.random.default_rng(0).choice(256,200)` sorted, decode each id + `_sanitize_label`): **107 U+FFFD, 23 empty, 72 unique**, first-30 array-order labels = ASCII punctuation/digit/letter fragments. This is a real defect — but the code computes it correctly: the decode of byte-fallback ids correctly returns U+FFFD, `_sanitize_label` (`pipeline.py:88-93`) correctly strips only C0/DEL, `plotting.py:227` correctly takes array-order. No line computes a wrong number. The defect is in the input id range (`run_semantic_clustering.py:66` `vocab_sample=256` under cl100k_base) and the label-selection strategy, not in any arithmetic or alignment.

## My weighted scores

Red:
- verified path:line (whitening, restrict, geodesic, sanitize, annotation, vocab range): 6 × 3 = 18
- reproduced test output (vocab decode counts): +3
- unverified/refuted reachability (model.py:447-454 cited as reachable — gated, false under active config): +1 then effectively neutralized; counted as 1
- external citations (Amari, McInnes, Bishop, Tufte, Wilson, Popper, Hall): 7 × 1 = 7
- Pennec wrong-domain (canon-cop 2 strikes): −2
- docstring cite (geometry.py:290): 0
- **Red weighted total: 27**

Blue:
- verified path:line (flatten same-order, alignment chain, block-restrict, quadrature, full-exp, generator selection, effective_block_dims): 7 × 3 = 21
- reproduced test output (0.0 off-block, 4.44e-16 path agreement): +3 (re-executed by me, confirmed)
- external citations (Nakahara, Hall, Lee, McInnes, Rousseeuw, Cencov): 6 × 1 = 6
- **Blue weighted total: 30**

The two totals are close because red landed genuine, code-verified hits on the vocab figure. But those hits are computationally-correct-yet-presentationally-weak (input-selection and label-salience), which blue conceded as defects without conceding any computational/alignment/geometry bug. On the code-truth axis — does the code do what the correctness sub-claims (geometry math + contextual semantics + index alignment) assert — every disputed correctness sub-claim resolves in blue's favor under verified path:line and executed numbers, and red conceded the contestable geometry/alignment points in its own rebuttal.

## Outcome (this judge)

BLUE_WINS

## Reasoning

The three correctness sub-claims this judge is asked to adjudicate — geometry math, contextual per-occurrence semantics, and index alignment — are all true at the code level under the independently re-traced active config. I re-ran the two load-bearing numerical checks from scratch with the exact in-repo builder reached at `model.py:485`: the 200-generator bank is exactly block-diagonal (off-block max 0.0) and `expm(A_full)` equals `blockdiag(expm(A_h))` to 4.44e-16, so the unguarded block-restriction at `geometry.py:339-340` produces the same Ω as the model's own full-matrix transport at `pipeline.py:119-121`. The `auto_close_cross_head_basis` path that would break this is doubly unreachable under the active config (gated on non-empty `cross_couplings` at `model.py:423`, and the toggle is `false`), so red's "latent" geometry defect is genuinely latent, not present — and red conceded as much. The flatten at `extract.py:98,106` reshapes μ and ids in identical order and the alignment chain at `pipeline.py:62-80,182,193-223` uses one shared subsample index and one strings list across all four plots, so contextual duplicate labels are structural per-occurrence semantics, not a data bug — red conceded this too (falsification (b) dead). Red's strongest surviving hits are the vocab-view defects, which I reproduced exactly (107 U+FFFD, 23 empty, 72 unique; array-order first-30 annotation), but every one of those lines computes its result correctly: the defect is the input id range and the label-selection strategy, not any arithmetic, geometry, or alignment error. Under my rubric the standard is what the code does, and the code does what the correctness sub-claims say; the vocab figure being misleading-but-correctly-computed is a frame/scope question outside this judge's weighting. Red's only attempt to name a present-config geometry error (the Pennec affine-invariant attachment) was a canon-cop 2-strike wrong-domain citation that red withdrew. Blue's verified path:line and re-executed numbers carry every disputed correctness sub-claim; red's verified hits, though real, do not establish a computational or alignment bug.

## Action

Accept the correctness sub-claims (geometry math, contextual per-occurrence semantics, index alignment) as code-verified true under the active config. Treat the vocab-view defect as a genuine accuracy/presentation defect that blue already conceded — the chief and the scope judge should decide whether it is severe enough to deny the module the global label "correct," since that is a frame question this judge does not weight. Independent of outcome, two concrete code fixes follow and do not change the verdict: (1) add a runtime guard or assertion at `geometry.py:339-340` that the discarded off-block entries of `A_full` vanish (or fall back to full-matrix exp) so the block-restriction does not silently truncate non-zero off-block algebra if a future config sets `cross_couplings != []` with `auto_close_cross_head_basis=True`; (2) for the vocab view, either skip byte-fallback ids (`run_semantic_clustering.py:66` id range under cl100k_base) or have `_sanitize_label` flag/drop U+FFFD and empty rows, and select the ≤30 annotated points by cluster-representativeness rather than array order at `plotting.py:226-227`.
