# Memo — implementation-engineer (blue, Phase 2)

Authored in-role by the coordinator (dispatch tool unavailable); synthesized into `02_blue_opening.md`.

## Active-config trace
`system_info.json`: embed_dim=20, irrep_spec=[['fund',2,10]], gauge_group='GLK', cross_couplings=[]. Therefore `config.py:893-902 effective_block_dims` returns `irrep_dims = [10,10]` (the `cross_couplings` branch is not taken). `model.py:419-491` selects `generate_glK_multihead_generators(20,2)` → block-diagonal bank. diagonal_covariance=True, n_tokens=200, tokenizer cl100k_base. The block-restriction `A_full[a:b,a:b]` at `geometry.py:339` slices exactly the two GL(10) head blocks under this config — exact, no off-block dropped.

## Position
(1) Contextual duplicate labels are structural: `extract.py:98,102,104,106` reshape `(B,N,*)->(B·N,*)` with token_ids flattened in the same order → one row per occurrence by definition. (2) Index alignment verified: `_subsample` reindexes all arrays with one shared `idx` (`pipeline.py:62-80`); `_decode_strings` runs once post-subsample (`pipeline.py:182`); one `strings` list to all four plots (`pipeline.py:193,202,210,222`); `project()` preserves row order. No off-by-one.

## Latent vs active
The unguarded block-restriction (`geometry.py:337-340`, no off-block assertion) is a *latent* defect: it only breaks under the opt-in `auto_close_cross_head_basis=True` toggle that can add super-block-spanning generators (`model.py:447-454`). Under the active config it is exact. Robustness gap, not a present correctness bug.

## Primary-source citation
`path:line` references above are the canonical record of code behavior per `debate_methodology.md` code-mode rules; reachability confirmed against `system_info.json`.
