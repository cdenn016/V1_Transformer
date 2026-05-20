# Verdict — multi-head-block-diagonal

## Outcome

BLUE_WINS

## Decisive evidence

Red's own concession in `03_red_rebuttal.md` paragraph 1: "I grant sub-claims α and β. The block-diagonal structural identification at `Attention/GL(K)_attention.tex:1716` is the unique `GL(d_k)`-restriction compatible with the per-head operator independence in [Vaswani2017 §3.2.2], and the thin-SVD lift at `:1727` inherits sub-claim C's BLUE_WINS verdict ... I further concede that the dimensional arithmetic at `:1781` is exact ... and that the per-head holonomy decomposition at `:1737–1743` is mathematically correct."

The headline claim's "exactly the gauge-theoretic block-diagonal restriction" rests on the architectural identification α and β, which Red concedes on the evidence. Red's remaining attacks on γ, δ, ε produce no primary-source falsification of substantive content; they are framing complaints. Blue conceded γ's "discarded" verb is framework-internal labeling at `Attention/GL(K)_attention.tex:1781` and accepted a one-clause labeling fix — which is the editorial-fix threshold the adjudication instructions define for BLUE_WINS.

## Reasoning

The compound claim partitions into structural identification (α, β, per-head holonomy) and three contested framings (γ, δ, ε). Red conceded the structural identification on Vaswani §3.2.2. The contested framings each fail to admit primary-source falsification.

On γ, both sides agree the arithmetic `512² − 8·64² = 229,376` and `229,376 / 262,144 = 0.875` is exact. Blue conceded in the rebuttal that the "discarded!" exclamation at `:1781` is framework-internal language measuring codimension within `gl(d_k)`, not a deficit of the standard transformer's parameter manifold. Canon pitfall #10 in `external_canon_transformers.md` §10 applies as a labeling rule. The substance survives under the label "framework-internal codimension on `gl(d_k)`."

On δ, the manuscript at `:1797` compares "the full-`GL(d_k)` transport" against "the block-diagonal-plus-output-projection factorization" — the latter phrase denotes a single-layer architectural unit. Vaswani §3.2.2 defines multi-head attention as a single-layer construct; layer stacking is `§3.1`. Within single-layer scope, off-diagonal `Ω_{ij}^{ab}` enters the KL argument to the softmax that produces `β`, while `W_O` operates only after every per-head softmax has run. That is a precise structural distinction. Red's rebuttal explicitly grants the per-layer statement and attacks only the unqualified "more expressive" wording. Red produced no primary source establishing that "more expressive than the block-diagonal-plus-`W_O` factorization" must be read as a multi-layer function-class supremacy claim.

On ε, both sides concede the empirical content: per-head attention temperature variation exists in trained standard transformers and arises from `‖W_Q^a‖, ‖W_K^a‖` because Q, K are linear outputs of learned projections. The evidence pack item 4 itself labels the retention-versus-novelty dispute "interpretive." Red provided no primary source falsifying the empirical-content reading of "retain" at `:1757`. Vaswani §3.2.1 specifies the uniform `1/√d_k` dimensional factor, which Blue does not contest — the framework's `κ_a` is the explicit per-head scalar that makes the implicit weight-norm variation a named parameter.

Per `debate_methodology.md`, "a cited claim outweighs an uncited assertion" and the judge must not split when one side is correct on the evidence. Red cites primary sources for framing critiques; Blue cites the same primary sources for the substantive identification. The substantive identification carries the headline claim. The framing critiques resolve to editorial labeling fixes, which is the BLUE_WINS threshold the adjudication instructions specify.

## Action

Apply the following manuscript edits to `Attention/GL(K)_attention.tex` and close the multi-head correspondence file.

At `:1781`, replace the framework-implicit deficit language with a framework-internal codimension label. The sentence "87.5\% of the gauge algebra is discarded" should read as a statement about the codimension of `⊕_a gl(d_head)` inside `gl(d_k)`, the framework's own ambient algebra, not as a deficit of the standard transformer's parameter manifold (which is `3·H·d_model·d_head + d_model² = 1{,}048{,}576` parameters for `d_k = 512, H = 8`, incommensurable in structure with `gl(d_k) = ℝ^{d_k × d_k}`). The exclamation mark should go with the rephrasing.

At `:1797`, insert a single-layer qualifier on "more expressive." The natural scope of multi-head attention in Vaswani §3.2.2 is one attention sublayer; multi-layer stacking lives in `§3.1`. The sentence should read approximately "the full-`GL(d_k)` transport is therefore more expressive *within a single attention sublayer* than the block-diagonal-plus-output-projection factorization, as it couples the attention-score-defining KL across heads at the same layer." This preempts the function-class reading Red attacked.

At `:1745–1757`, the "retain" wording stands under the empirical-content reading. No edit required if the surrounding paragraph already establishes that the empirical per-head temperature variation in trained standard transformers (via `‖W_Q^a‖, ‖W_K^a‖`) is what `κ_a` exposes as a named slot. If clarity is desired, add one clause: "Vaswani §3.2.1 specifies a uniform `1/√d_k` dimensional factor across heads; the per-head variation is implicit in the trained projection magnitudes, and the framework exposes that empirical content as the explicit per-head scalar `κ_a`." That is a clarification, not a correction.

The headline claim stands with these two labeling fixes (and the optional ε clarification). No further debate is required on multi-head as block-diagonal restriction. The follow-up question for the next debate, if pursued, is the empirical separation question Red raised under δ falsification F2: whether full-`GL(d_k)` single-layer transport with off-diagonal blocks is empirically distinguishable from multi-layer block-diagonal-plus-`W_O` stacks at fixed parameter budget. That is a function-class / depth-efficiency question outside the present claim's scope.
