# Memo — implementation-engineer (red, opening)

## Config trace (active run, K=20 / irrep_dims=[10,10])

Entry point `transformer/vfe/run_semantic_clustering.py`. The vocab view is built at
main():151-164: `n = min(vocab_sample, vocab_size)` then `vocab_ids = torch.arange(n)`
(lines 152-153) with a cl100k_base decode adapter (`_vocab_dataset`, lines 187-197). With the
repo `vocab_sample=256` this means the vocab view extracts **token ids 0..255**, then
`_subsample` (pipeline.py:62) takes 200 of them. In cl100k_base the first 256 ids are the
single-byte fallbacks; decoding each id individually (`dataset.decode([i])`, pipeline.py:103)
yields the U+FFFD / empty / punctuation-fragment labels the verifier reproduced (107→'�',
23→'', 72/200 unique). This path is reached and executed under the active config — it is not
hypothetical. The vocab figure the user is looking at is a scatter of byte-fallback priors.

## The two genuine code defects

**(1) The block-restriction at geometry.py:337-340 is a latent correctness defect, masked by the
active toggle.** The code computes `A_full = einsum(phi, G)` over the full K×K, then takes the
diagonal block `A_h = A_full[a:b, a:b]` and exponentiates it. This is correct **only if** the
generator bank has no cross-block support. Under the active config (`cross_couplings=[]`,
`auto_close_cross_head_basis` not set), the bank is exactly block-diagonal — the verifier measured
max off-block |entry| = 0.0 and `||expm(A_full) − blockdiag(expm(A_h))|| ≈ 4.4e-16`. So under THIS
config the path is correct. But the restriction is unguarded: there is no assertion that
`A_full[a:b, c:d]` (c≠a) is zero before discarding it. Under `auto_close_cross_head_basis=True`
(model.py:447-454 warns it can add generators spanning super-blocks) the bank acquires off-block
support, `A_full` becomes non-block-diagonal, and geometry.py:339 silently truncates the
off-block algebra. `expm` of a truncated block ≠ the block of `expm` of the full matrix when the
matrix is not block-diagonal — the Ω geodesic would then be computed on a different group element
than the model's actual transport (which exponentiates the full A in `_per_token_omega`,
pipeline.py:111-122, and in the model's own `_compute_block_exp_pairs`). The two Ω code paths would
diverge. The defense calls this "completeness"; it is a correctness bug that is currently
non-triggering, which is a different and weaker safety property than "correct."

**(2) The φ-distance is subsample-dependent through the whitener.** `_subsample` (pipeline.py:62)
fixes a seed, so the run is reproducible, but `phi_vector_distances` whitens by the SVD of the
subsampled centered matrix (geometry.py:249, 258). The distance any token pair receives depends on
which other 199 tokens were drawn. This is a real coupling between the sampling step and the
geometry; the Ω geodesic has no such coupling (it is pairwise-intrinsic). The figure presents both
as fixed per-token geometry.

## What would falsify my position

For defect (1): if there were a runtime assertion or a config guard forbidding the block-restriction
path when off-block generators exist, it would be safe. There is none at geometry.py:337-340. The
"only breaks under a non-default toggle" defense is exactly the latent-bug signature — correct under
the tested config, silently wrong under a reachable one.

## Newly-discovered canon

- No external canon; this memo is a config-trace and reachability analysis. All claims are
  `path:line` plus the verifier's executed numbers (off-block max = 0.0; `||expm(A_full) −
  blockdiag(expm(A_h))|| ≈ 4.4e-16`). The non-block-diagonal failure of "exp of a block = block of
  exp" is elementary linear algebra (`expm` is a power series; cross-block products appear at second
  order when off-block entries are nonzero).
