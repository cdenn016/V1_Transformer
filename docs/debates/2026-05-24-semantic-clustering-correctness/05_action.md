# Action — semantic-clustering-correctness

**From verdict:** REMAND (chief reconciliation; panel=full — canon-strict REMAND, code-truth BLUE_WINS, scope REMAND)

## The user's two flagged phenomena — resolved

Both are **correct-by-design, not bugs**:

- **φ/Ω visual divergence** (`phi_vector_clustering` vs `omega_clustering` looking different despite `Ω = exp(Σ_c φ_c G_c)`): the two distance matrices agree at Spearman 0.952–0.972 and both independently pick k=2; the visual gap is a UMAP non-isometric-projection layout artifact (free global rotation/reflection, not distance-preserving). Per-head quadrature is the exact product-manifold metric on GL(10)×GL(10); the two Ω code paths agree to 4.4e-16. No geometry or index bug.
- **Duplicate `Atlantic`/`the`/`of` labels in the contextual figures**: per-occurrence semantics of the `(B,N)->(B·N)` flatten (`extract.py:98`), one row per occurrence; recurring token types legitimately repeat. Index alignment verified off-by-one-free.

## Recommended action

### (i) Confirmed correct — no action
Sub-claims A (geometry) and B (contextual per-occurrence semantics). The geometry math, the per-head geodesic quadrature, the two-Ω-path agreement (4.4e-16), the contextual flatten, and the full index-alignment chain are correct under the active config (K=20, irrep_dims=[10,10], cross_couplings=[]).

### (ii) Real defect to FIX — sub-claim C, the VOCAB figure
Severity: **presentation/accuracy defect (shipped-figure integrity)**, not a math/alignment bug. Each row's arithmetic is correct, but the vocab figure licenses a false reader inference: under cl100k_base, the first-256-id extraction gives 107/200 U+FFFD, 23/200 empty, only 72/200 unique, and the 30 annotated points are the lowest (anti-salient) array-order ids. Three concrete fixes:
1. `_sanitize_label` (`pipeline.py:88-93`) — strip or flag U+FFFD and empty strings so byte-fallback rows do not reach the figure as semantic labels.
2. Annotation path (`plotting.py:224-235`) — replace array-order `range(min(...,30))` with salience / cluster-representative point selection.
3. Entry point (`run_semantic_clustering.py:66`, the cl100k_base id range) — choose an informative id range that skips the byte-fallback band, or document the byte-fallback limitation in the figure title/caption.

### (iii) Latent guard — optional, sub-claim D
The block-restriction at `geometry.py:337-340` is numerically exact under the active config (off-block = 0.0). It is unguarded: before anyone sets `auto_close_cross_head_basis=True` (which can add super-block-spanning generators), add an assertion that the discarded off-block entries of `A_full` vanish, or fall back to full-matrix exp. Cheap to close; not a present bug.

### Minor cleanup (outside the binding sub-claim set)
`geometry.py:290` docstring carries a wrong-domain "affine-invariant" label — the code computes a left-invariant GL⁺(K) group geodesic, not the SPD-cone congruence distance. Optional doc fix.

## Follow-up debates (if any)

None required. The REMAND decomposition is fully resolved by the panel: A/B confirmed correct, C is a known defect with a concrete fix list, D is a latent guard. No sub-claim needs its own debate. If the user wants to escalate sub-claim C, the open question would be narrowly: "Is the vocab view salvageable as a semantic artifact for byte-level BPE tokenizers, or should it be restricted to word-level / non-fallback id ranges?" — a design question, not an adversarial one.
