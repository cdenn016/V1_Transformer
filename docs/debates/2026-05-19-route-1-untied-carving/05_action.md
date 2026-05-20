# Action — route-1-untied-carving

**From verdict:** RED_WINS

## Recommended action

The verdict closes the last open queue item from the original first-debate series. The Route 1 "exact reduction to Vaswani §3.2.1" claim fails on the same structural problem as Route 2: under the closure `Σ_j = U_j C U_j^T`, the absorbed j-only term `r_j = ‖K_j^{sym}‖²` is content-dependent on `μ_j`. The recovery is to "Vaswani §3.2.1 + content-dependent additive key-side bias," which is a generalization of §3.2.1, not §3.2.1 itself.

The cumulative picture across the seven §4–§5 debates is now complete:
- F_align^(τ) is an engineered soft-assignment Lagrangian, not a joint-FEP-derived ELBO (softmax-β verdict).
- The §5 reduction's individual structural identifications (sub-claims A-D, multi-head) hold under their stated preconditions with editorial labeling fixes applied.
- The implementation descends on the surrogate `⟨E⟩_{β*}`, not on F_red — now reconciled at `:967, :2008` (canonical-F verdict).
- **Neither Route 1 nor Route 2 recovers Vaswani §3.2.1 in the strict sense.** Route 2 fails on approximate key-norm cancellation; Route 1 fails on exact absorption of content-dependent prior bias. Both routes hit the same content-dependent quantity `‖K_j‖²`; the framework's reductions land at "Vaswani §3.2.1 + key-norm additive bias," not at Vaswani §3.2.1 itself.

### Three manuscript edits

#### Edit 1 — §5.2.1 headline framing (around line 1125)

Currently:

> "The trivial-frame reduction below is one realisation of standard attention. A second, complementary reduction retains `Ω_{ij} = U_i U_j^{-1}` non-trivially throughout and exhibits standard attention with untied query-key projections directly from the gauge-covariant KL, without inserting a separate bilinear M by hand."

**Recommended replacement:** soften "standard attention" to specify the destination as a structural identification of the `Q^T K` form with the key-side bias slot occupied, not as recovery of Vaswani §3.2.1's uniform-prior boxed form.

> "The trivial-frame reduction below is one realisation of the standard attention structure. A second, complementary reduction retains `Ω_{ij} = U_i U_j^{-1}` non-trivially throughout and exhibits a structural identification with standard scaled dot-product attention's `Q^T K` form, with untied query-key projections derived directly from the gauge-covariant KL and a content-dependent key-side bias absorbed into the additive prior slot. This is a generalization of \citet{vaswani2017attention} §3.2.1, which uses uniform prior; the gauge framework's key-side bias is the absorbed `j`-only terms of the transported KL decomposition (Eq. \ref{eq:full_kl_general}) and is structurally analogous to additive prior mechanisms such as causal masking and ALiBi."

#### Edit 2 — line 1158 "standard attention recovered" wording

Currently:

> "Under this closure the carving in Eq. `eq:gauge_qk` becomes the symmetric form `Q_i = C^{-1/2} U_i^{-1} μ_i, K_j = C^{-1/2} U_j^{-1} μ_j`, and standard attention with a single shared bilinear `C^{-1}` is recovered."

**Recommended replacement:**

> "Under this closure the carving in Eq. `eq:gauge_qk` becomes the symmetric form `Q_i = C^{-1/2} U_i^{-1} μ_i, K_j = C^{-1/2} U_j^{-1} μ_j` with a single shared bilinear `C^{-1}`. The resulting attention rule is structurally `softmax_j(Q_i^T K_j / √d_k + b_{ij}) V_j` where `b_{ij} = -r_j/(2\tau) + \text{const}` is the absorbed key-side bias, with `r_j = ‖C^{-1/2} U_j^{-1} μ_j‖² = ‖K_j‖²` the squared key-norm. The framework recovers the structural form of standard attention modulo this content-dependent additive bias; the strict Vaswani §3.2.1 form with uniform prior is obtained only when `b_{ij}` is approximately constant in `j` (e.g., under layer-normalized key-norms, the same condition discussed in Section~\ref{sec:dot_product_derivation}'s key-bias-cancellation paragraph)."

#### Edit 3 — line 1160 rotary-positional-structure framing

Currently:

> "The natural identification of the per-token frame `U_i` with a real transformer architecture is the per-position rotational frame of rotary positional embeddings, in which `U_i ∈ O(d_k)` is a block-diagonal rotation depending on token position."

**Recommended replacement:**

> "The natural identification of the per-token frame `U_i` with a real transformer architecture is the per-position rotational frame of rotary positional embeddings \citep{su2024roformer}, in which `U_i ∈ O(d_k)` is a block-diagonal rotation depending on token position. The recovery in this case is to RoPE-style attention (which is itself an extension of \citet{vaswani2017attention} §3.2.1 with position-dependent gauge), not to the original Vaswani §3.2.1 uniform-prior form."

These three edits make explicit what the strict primary-source reading of "Vaswani §3.2.1" requires, while preserving the manuscript's structural-identification claim (which the debate found defensible).

### Cumulative open queue

**No open follow-up debates remain from the seven-debate series.** The seven debates that have now closed:

1. §5 transformer reduction (RED_WINS on compound headline).
2. Softmax-β stationarity (RED_WINS, manuscript adopted option-(b) framing).
3. Sub-claim A flat bundle (BLUE_WINS).
4. Sub-claim B degenerate Σ (BLUE_WINS).
5. Sub-claim C Q K^T identification (BLUE_WINS).
6. Sub-claim D V identification (BLUE_WINS).
7. Canonical F vs surrogate (RED_WINS).
8. Multi-head block-diagonal (BLUE_WINS).
9. Route 1 untied carving (RED_WINS — this debate).

(Nine total when counting the first debate as one and the four sub-claims A-D as four separate debates.)

### Cumulative editorial state of §5

The §5 transformer reduction section has been substantially revised across the debate series:

- **Section title and preconditions** (sub-claim B): Limit-language softened to reduction-language; explicit preconditions stated.
- **σ⁻²Ω⁻ᵀ identification** (sub-claim C): `d_k = d_head` clarified; value-level scope statement added.
- **W_V definition** (sub-claim D): `d_k = d_head` clarified; parallel thin-SVD lift documented.
- **§5.7 summary** (sub-claims A and B): "successive limits" → "successive reductions"; gauge-equivariant qualifier rewritten.
- **Multi-head §5.4** (multi-head verdict): "87.5% discarded" reframed as codimension; "more expressive" qualified with "single-layer."
- **§4.7 implementation language** (canonical-F verdict): `:967` Belief Dynamics and `:2008` E-step rewritten to descend on `⟨E⟩_{β*}` rather than on F_red.
- **§4.6 framing** (softmax-β verdict): adopted soft-assignment-Lagrangian framing via Cuturi/Boyd citations; "derived from first principles" softened.
- **§5.2.1 Route 1** (this verdict, pending): three edits proposed above.

After Edit 1–3 are applied, the §5 reduction is internally consistent with the framework's actual mathematical content as adjudicated across nine debates.
