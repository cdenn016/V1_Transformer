# Blue Rebuttal — subclaim-A-flat-bundle

## Concession

I grant the §5.7 conflation. The summary paragraph at `Attention/GL(K)_attention.tex:1974` ("the GL(K) gauge invariance theorem ... guarantees that the learned projections W_Q, W_K are valid gauge transformations") overreaches the theorem it cites. Theorem `thm:glk_invariance` at `Attention/GL(K)_attention.tex:520–555` proves a single-Ω simultaneous push-forward invariance of the KL — a textbook f-divergence property (`external_canon_math.md` §1, KL invariance under bijective change of variables). It is not a per-agent equivariance theorem, and the summary line invokes it as if it were. Red's stabilizer computation under the connection transformation law `Ω_{ij} → g_i Ω_{ij} g_j^{-1}` is correct: fixing `Ω_{ij} = Ω` collapses the stabilizer to the diagonal subgroup `g_i = g`, not the full per-agent group `G^N`. The gauge-invariance vs. gauge-equivariance distinction from `external_canon_math.md` §2 applies to that summary paragraph as written.

That concession is bounded. It targets a rhetorical line in §5.7. It does not target sub-claim A.

## Core attack

Red attacks the wrong reading of sub-claim A.

Sub-claim A asserts the flat-bundle limit is a **well-defined limit** that **does not destroy** the upstream gauge-equivariance arguments. The user-context block in `00_claim.md:17-21` makes the load-bearing distinction explicit:

> (i) The flat-bundle limit is a *specialization* ... Specialization does not destroy the upstream argument; it instantiates it at a particular point.
> (ii) The flat-bundle limit is a *gauge fixing*: ... The fixed-gauge object is just an ordinary Euclidean construction.

The evidence pack at `01_evidence.md:66-68` is more explicit, naming the two readings of "preserves":

> **Strong reading.** "Preserves" = the gauge-fixed equations remain gauge-equivariant. This is **false** in standard gauge theory.
> **Weak reading.** "Preserves" = the upstream framework (before specialization) remains gauge-equivariant; the specialization simply selects one orbit representative without altering the upstream theorem. This is **true**, trivially.

Sub-claim A asserts the weak reading. Red's entire opening attacks the strong reading. Red proves the gauge-fixed object is not stabilized by the full per-agent group — a property the weak reading never claims. Red proves the §5.7 summary conflates two invariances — a separate problem, not the claim under debate. The load-bearing weakness in red's opening: it does not engage the distinction `01_evidence.md` itself draws, treating sub-claim A as if it asserted the strong reading.

The upstream theorem `thm:glk_invariance` at `Attention/GL(K)_attention.tex:520–555` holds for every `Ω ∈ GL(K)`, including `Ω = I` and `Ω = const`. Selecting one element of the group does not retroactively void the theorem at the other elements. The upstream framework remains gauge-equivariant in the precise sense it was equivariant before; the limit picks one orbit representative. That is the weak reading, and that is what sub-claim A asserts.

## Defense

Three lines.

First, on what sub-claim A actually asserts. The claim states the limit is "well-defined" and "does not destroy the upstream gauge-equivariance arguments." Well-defined: setting `Ω_{ij} = Ω` and absorbing `Ω^{-T}` into `W_Q W_K^T` produces standard scaled dot-product attention (`Attention/GL(K)_attention.tex:1115`, `:1158`, `:1958`), and this reduction is internally consistent — the determinant cancellation in the KL (theorem proof at `:539–553`) holds for any `Ω`, including constant. Does not destroy: the unfixed framework retains its full GL(K) equivariance as a property of the framework before the limit. The claim says nothing about per-agent equivariance of the post-limit standard transformer.

Second, on `W_Q, W_K` as gauge transformations. The §5.7 summary at `:1974` reads "are valid gauge transformations." Red reads this as "implement gauge equivariance." These differ. The claim that a learned matrix is *valued in* the gauge group (an element of GL(d_k)) is a statement about the codomain of the parameterization; the claim that the architecture is *equivariant under* the per-agent gauge action is a statement about the symmetries of the map. The first does not entail the second. Sub-claim A defends the first interpretation: in the flat-bundle limit, the learned projections inherit the structural role that `Ω^{-T}` played in the unfixed framework, and they live in GL(d_k). They do not, in the gauge-fixed object, implement a per-agent equivariance — and the claim does not assert they do.

Third, on the supplementary admission at `Attention/GL(K)_supplementary.tex:53`. Red treats this as a confession that the framework is empirically empty in the flat-bundle subclass. It is not. The admission is a scope statement: the working framework lives in the trivial-curvature subclass (`F = 0`, trivial holonomy by construction via the triangle identity `Ω_{ij}Ω_{jk}Ω_{ki} = I` at `01_evidence.md:54`), with the non-trivial-curvature extension reserved for the Wilson-loop / Regime II generalization. The trivial-curvature subclass still has non-trivial per-edge frame misalignment: `Ω_{ij} = exp(φ_i) exp(-φ_j) ≠ I` whenever `φ_i ≠ φ_j`. The vertex-frame parameterization makes the bundle trivial globally but the per-edge transport non-trivial pointwise. This is exactly the regime the GL(K) attention rule operates in, and exactly the regime in which the upstream theorem applies non-trivially: each edge KL is computed with a non-identity `Ω_{ij}` even though the global holonomy is trivial. The further reduction `Ω_{ij} = Ω` is the additional limit that flattens edge misalignment to a constant — and that further limit is what sub-claim A names as the "flat-bundle / constant-gauge" limit.

The position I defend: sub-claim A is true under the weak reading explicitly named in `01_evidence.md:66-68`, and the weak reading is what the claim text asserts. Red's strongest hits — the stabilizer collapse under the per-agent action, and the §5.7 conflation — establish that the strong reading fails. Both teams agree the strong reading fails. The dispute is whether sub-claim A asserts the strong reading. The claim text does not.

Falsification conditions for this defense, restated for the judge: I am wrong if (1) the manuscript at `Attention/GL(K)_attention.tex` actually requires the strong reading of "preserves" for some downstream argument outside §5.7 — i.e., if some other section uses per-agent equivariance of the gauge-fixed standard transformer as a load-bearing premise, not just the §5.7 summary rhetoric; or (2) the constant-Ω limit is shown to be ill-defined as a limit of the upstream theorem (e.g., the determinant cancellation in the KL proof fails at `Ω = const` or the absorption into `W_Q W_K^T` is internally inconsistent). Red has not established either. Red has established that the §5.7 summary, read maximally, asserts the strong reading and overreaches its cited theorem — which is a real defect of that paragraph, but a separable one from the well-definedness of the flat-bundle limit itself.
