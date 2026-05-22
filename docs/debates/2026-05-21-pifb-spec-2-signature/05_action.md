# Action — pifb-spec-2-signature

**From verdict:** RED_WINS

## Recommended action

Four edits required. The underlying signature mechanism (frame-twist quadratic form, +tr/-tr conventions, GL(K,C) sector split, Sylvester's-law arguments, the causal-cone alternative route) survives — but a load-bearing postulate is currently unstated, and a cross-section coherence claim is unsupported.

### Edit 1 — Name the single-generator postulate at PIFB:2826

The decomposition at line 2824 introduces two generators `T_tau`, `T_x`; equation 2826 silently collapses to a single shared `T`. Sympy verification: with `T_tau = diag(1,-1)` and `T_x = [[0,1],[-1,0]]` (admissible compact element of `gl(2,C)`), the signature comes out `(-,-)` not `(-,+)`. The Lorentzian conclusion depends on this collapse.

Repair: insert a postulate-naming sentence before equation 2826: "We further postulate that the temporal and spatial generators are taken to be a single shared `T = diag(1,-1)` for concreteness; the case of distinct `T_tau, T_x` requires separate treatment and is not adopted here. The signature conclusion depends on this single-generator choice."

### Edit 2 — Reframe the real-part projection at PIFB:2841 as rank-changing

Under the single-generator collapse, the unprojected complex `G_{mu nu}` is a rank-1 outer product (`det = 0`). The real-part projection is therefore not "discarding an imaginary off-diagonal piece" but a rank-1-complex → rank-2-real operation.

Repair: replace "the real-part projection sets `G^Lor_{tau x} = 0` and leaves..." with "Under the single-generator postulate, the complex bilinear form `G_{mu nu}` has rank 1; the real-part projection is a rank-changing operation that yields a rank-2 real form, not merely a discard of an off-diagonal imaginary part. The dynamical mechanism that would justify this rank change is the open problem flagged at the end of this subsection."

### Edit 3 — Extend the postulate list at PIFB:2856 from three to four

The "Three features" paragraph lists (1) signature from connection, (2) postulates: imaginary frame + real-part projection, (3) total metric indefinite when frame-twist dominates. Add a fourth: "(4) The worked example uses a single generator `T` along both base directions; the case of distinct generators requires separate treatment and is not adopted here."

### Edit 4 — Add cross-section regulator note around PIFB:2858

Acknowledge that §sec:consensus_metric at 2928 documents the obstruction to extracting gauge-invariant content from the frame-twist quadratic form under non-compact SO(1,3) (infinite Haar). The signature construction places its result on exactly this object, so the Lorentzian signature is observable-from-within-an-agent's-gauge-fixing but is not currently a gauge-invariant assertion at the multi-agent / consensus level.

Repair: insert a paragraph near 2858: "The Lorentzian signature obtained here lives on the frame-twist quadratic form `tr(A_mu A_nu)`, which §sec:consensus_metric establishes is gauge-noninvariant and admits no regulator-free gauge-orbit average under non-compact SO(1,3). The signature conclusion is therefore an observable feature of the within-an-agent's-gauge-fixing pullback, not yet a gauge-invariant assertion at the multi-agent or consensus level. The relation between this within-agent signature and a hypothetical consensus-level signature is a follow-up open problem."

## Follow-up debates (if any)

None. After applying Edits 1-4 the section reframes as "structural existence demonstration of Lorentzian-compatible signature under the four named postulates", which is a defensible and publishable claim.
