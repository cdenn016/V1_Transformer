# Action — pifb-spec-3-consensus-dim

**From verdict:** RED_WINS

## Recommended action

Three sentence-level edits. The math (spectral theorem, pullback rank theorem, non-compact Haar measure) and the canonical apparatus survive — but three specific sentences inside the subsection contradict the careful framing established elsewhere in the same subsection.

### Edit 1 — Rewrite the base-vs-fiber statement at PIFB:3024

Current: "only a tiny sliver (e.g. 4 dimensions out of K) becomes phenomenal spacetime. The remainder is internal structure that never manifests spatially or perceptually."

`K` here is the Gaussian fiber parameter (set to 768 at line 2968, giving `dim(B) ≈ 296,064`). The "4 out of K" comparison treats `K` as the observable-dimension bound, which contradicts the rank-bound established 50 lines earlier at 2970: the pullback metric `G_i(c)` is an `n x n` tensor on `T_c C` with at most `n = dim(C)` eigenvalues, and "vast majority of dimensions" refers to fiber directions not sampled by `d sigma_i`.

Repair: rewrite as "Only `|D_obs|` directions (e.g., 4 for a human cognitive agent) appear as phenomenal spacetime; the remaining `dim(C) - |D_obs|` base directions carry sub-threshold or internal eigenvalues. The fiber-side directions in `T_q B` that are not sampled by `d sigma_i` constitute an independent layer of `dim(B) - dim(C)` unsampled directions, which is the sense in which 'most of B is invisible' as discussed at 2996."

### Edit 2 — Cross-reference signature postulates at PIFB:2980

Current: "For human agents, we conjecture this comprises approximately 4 dimensions (1 temporal + 3 spatial)."

The "1 temporal" qualifier requires an indefinite metric (negative eigenvalue), which the spectrum at line 2956 forbids (positive semi-definite by construction). The temporal eigenvalue exists only if the GL(K,C) postulates from §sec:signature_resolution are in force. Grep confirms zero references to `sec:signature_resolution` or `sec:worked_signature` inside lines 2945-3024.

Repair: insert a forward-reference clause: "For human agents, we conjecture this comprises approximately 4 dimensions; under the indefinite-signature postulates of §sec:signature_resolution (imaginary frame component along one base direction with real-part projection), the 4-dimensional sector admits a (1 temporal + 3 spatial) Lorentzian decomposition."

### Edit 3 — Qualify "all perspectives are valid" at PIFB:3045

Current: "all perspectives are valid."

This is unqualified strict relativism and is in tension with the within-species consensus framing at line 2937 ("the load-bearing one for an 'objective perceived geometry' reading"). The honest reading is gauge-orbit equivalence (Rovelli's relational QM), not unconditional relativism.

Repair: replace "all perspectives are valid" with "all perspectives are valid within their own gauge frame, and the structural-tier consensus construction of §sec:consensus_metric (conditional on a regulator) is the framework's candidate for a gauge-invariant shared geometry — to which individual agent geometries are reconciled within a cognitive species rather than absolutely."

## Follow-up debates (if any)

None. After applying Edits 1-3 the operational reading of "rock-solid" is satisfied: math correct, canon correctly invoked, no implicit conflations, no silent inheritance of out-of-scope postulates, no unqualified relativism in tension with the framework's own consensus apparatus.
