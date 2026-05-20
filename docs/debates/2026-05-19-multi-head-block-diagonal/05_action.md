# Action — multi-head-block-diagonal

**From verdict:** BLUE_WINS

## Recommended action

**Accept the headline claim** that standard multi-head attention is the block-diagonal `GL(d_head)^H ⊂ GL(d_k)` gauge structure with the thin-SVD lift, inheriting sub-claim C's BLUE_WINS verdict. The structural identification at `Attention/GL(K)_attention.tex:1716, :1727, :1732` holds.

**Two manuscript labeling fixes** per the verdict's action:

### Edit 1 — `:1781` (the "87.5% discarded" sentence)

Currently:

> "the complement `m` has dimension `d_k² - H · d_head²`. For typical architectures (e.g., `d_k = 512, H = 8, d_head = 64`), this is `512² - 8 × 64² = 229,376` generators out of `262,144` total implying that *87.5% of the gauge algebra is discarded* by the multi-head factorization!"

**Fix:** rephrase to make the framework-internal framing explicit. The "discard" is from the framework's full `GL(d_k)` reference, not from a comparison to the standard transformer's parameter manifold (which has a different dimension `3·H·d_model·d_head + d_model²` and is not a subset of `gl(d_k)`).

Recommended replacement:

> "the complement `m` has codimension `d_k² - H · d_head²` within `gl(d_k)`. For typical architectures (e.g., `d_k = 512, H = 8, d_head = 64`), this is `512² - 8 × 64² = 229,376` generators out of `262,144` total: the multi-head architectural choice projects out `87.5%` of the framework's `gl(d_k)` generators, retaining only the block-diagonal subalgebra `⊕_a gl(d_head)`."

This preserves the dimensional fact, drops the exclamation, and labels the framing as framework-internal (codimension within `gl(d_k)`, not comparison to the standard's parameter manifold).

### Edit 2 — `:1797` (the "more expressive than block-diagonal-plus-output-projection" claim)

Currently:

> "The full-`GL(d_k)` transport is therefore more expressive than the block-diagonal-plus-output-projection factorization, as it couples the attention computation across heads rather than only the output computation."

**Fix:** add a single-layer qualifier so the comparison object is unambiguous.

Recommended replacement:

> "The full-`GL(d_k)` transport is therefore more expressive than the single-layer block-diagonal-plus-output-projection factorization, as it couples the attention computation across heads within one layer rather than only the output computation; standard transformers recover cross-head awareness across multiple layers via `W_O` followed by next-layer projections, so the framework's advantage is at the per-layer level."

This addresses red's function-class objection (multi-layer transformers compose cross-head mixing) without weakening the per-layer claim.

### Optional clarification at `:1745–1757` (per-head temperature)

The verdict notes the "retain" wording at line 1757 stands under the empirical-content reading. An optional one-clause clarification would distinguish the framework's *explicit* `σ_a²` parameter slot from the standard's *implicit* per-head variation via learned weight magnitudes. Recommended only if the user wants to address the parameter-slot-vs-empirical-content ambiguity directly.

Recommended optional addition at `:1757`:

> "The framework retains the per-head covariance variation that standard transformers implement implicitly through `‖W_Q^a‖, ‖W_K^a‖`; making this variation explicit as a learnable `σ_a²` parameter is a re-parameterization that surfaces empirical structure already present in trained standard transformers."

## Follow-up debates (if any)

One open queue item remains:

1. **Route 1 (untied carving) alone reduces to Vaswani §3.2.1** (§5.2.1). Blue's strongest unrefuted move in the first debate; still open.

The cumulative picture across the four §4–§5 debates (softmax-β, sub-claims A-D, canonical-F-vs-surrogate, multi-head):
- F_align^(τ) is an engineered soft-assignment Lagrangian (softmax-β verdict).
- The §5 reduction's individual steps are correct under stated preconditions (sub-claims A-D + multi-head).
- The implementation descends on the surrogate, not on F_red (canonical-F verdict; now reconciled at `:967`, `:2008`).
- Multi-head IS the block-diagonal restriction, with labeling fixes for `:1781`, `:1797`.

The remaining Route 1 debate would close the §5 reduction questions; the original first-debate headline ("every intermediate step exactly mathematically valid") still fails on sub-claim E (the approximate key-norm cancellation) as the only headline-level failure.
