# memo_red_implementation-engineer (rebuttal) — semantic-clustering-correctness

## Concede the alignment; it is verified

`extract_contextual` flattens (B,N,·)→(B·N,·) for mu/sigma/phi and flattens token_ids in the same order (`extract.py:98-106`). `_subsample` reindexes every array with one shared idx (`pipeline.py:62-80`); `_decode_strings` runs once after subsample (`pipeline.py:182`); the same strings list goes to all four plots (`pipeline.py:193,202,210,222`); `project()` preserves row order. No off-by-one, no decode/coord skew. Falsification condition (b) (a data-corruption bug) is dead, and I concede it. The duplicate labels are the structural per-occurrence output, not corruption.

## What survives: the figure is not interpretable as shipped, and that is a present defect

The vocab decode is executed and reproduced (investigator A + verifier, exact subsample `np.sort(rng(0).choice(256,200,replace=False))`, per-id `dataset.decode([i])` + `_sanitize_label`): 107/200 → U+FFFD, 23/200 → '', 72/200 unique, 130/200 duplicates. `_sanitize_label` (`pipeline.py:88-93`) strips only C0+DEL — it does *not* strip or flag U+FFFD or empty, so the unrenderable rows reach the figure as-is. This is the active config (cl100k_base, n_tokens=200, evidence pack §"Active config"). It is reached at runtime; it is not latent.

The annotation path makes it worse, deterministically. `plotting.py:224-235` annotates `range(min(len(token_strings), coords.shape[0], 30))` — the first 30 *array-order* points, and the active subsample is sorted ascending by id, so the annotated set is the 30 lowest ids: ASCII punctuation/digit fragments (`! " # $ % ( * + , - 0 1 …`). The labels a reader actually sees are the least semantically meaningful 30 of an already-65%-junk set. So the figure does not merely *omit* salience; it actively annotates anti-salient points. That is a present, reachable behavior under the active config, not a robustness gap behind a toggle.

## The unguarded block-restriction: latent, but the guard's absence is a real correctness exposure

`geometry.py:337-340` slices `A_full[a:b,a:b]` then `expm`, with no assertion that off-block entries are zero. Under the active config (`cross_couplings=[]`, bank max off-block = 0.0 exact) the restriction is exact and the result is correct — concede that. But the code is *unconditionally* a block-restriction regardless of the bank: under `auto_close_cross_head_basis=True` (`model.py:447-454`, can add super-block-spanning generators) the slice silently discards off-block algebra and the geodesic is computed on the wrong group element with no error. The contrast with `pipeline._per_token_omega`, which exponentiates the full K×K A (`pipeline.py:111-122`), means the two paths would *diverge* the moment the bank stops being block-diagonal — the 4.4e-16 agreement is a property of the current bank, not of the code. A correctness-sensitive distance routine that is silently wrong under a shipped toggle, with no guard, is a latent correctness defect, not a UX item.

## Newly-discovered canon

- None beyond the pack; this memo is `path:line` evidence for what the code does under the active config. (Reachability traced: active config → cl100k_base/n=200 → `_decode_strings` at `pipeline.py:182` → `_sanitize_label` at `pipeline.py:88-93` does not strip U+FFFD → `plotting.py:224-235` annotates first-30 array-order.)
