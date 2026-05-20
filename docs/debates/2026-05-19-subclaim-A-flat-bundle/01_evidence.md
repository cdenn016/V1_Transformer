# Evidence Pack — subclaim-A-flat-bundle

Neutral fact pack. Both teams work from this file.

## Manuscript references

### Constant-gauge specialization (`Attention/GL(K)_attention.tex:1115`)

> "Standard transformer attention emerges when we further assume a single global frame shared by all agents, `Ω_{ij} = Ω` for all i,j, corresponding to a flat connection with no position-dependent frame misalignment. The geometric bias `S(Ω)` becomes constant across all agent pairs and cancels under softmax, while `Ω^{-1}` is absorbed into learned projections."

### Trivial-frame collapse (`Attention/GL(K)_attention.tex:1158`, in Route 1 discussion)

> "The trivial-frame specialisation `U_i = U` for all i removes the per-token positional variation; in that limit `Ω_{ij} = I` and the analysis of Section [dot-product-derivation] applies, with the learned bilinear M replacing the structural role played by per-token frame variation."

### Bundle triviality (`Attention/GL(K)_supplementary.tex:53`)

> "Throughout the main paper we work with the vertex-frame parameterization `Ω_{ij} = exp(φ_i)exp(-φ_j)`, which encodes a *globally trivial* principal G-bundle: a single chart `N ≅ C × G` suffices up to gauge equivalence, and the curvature 2-form of the induced connection vanishes identically (Lemma vanishing_holonomy of the main text). The principal-bundle vocabulary developed in this appendix is therefore strictly more general than the parameterization used by the language model and the BERT validation, both of which live in the flat-bundle subclass."

This is a candid admission: the entire main-paper framework lives in the *flat-bundle subclass*. The "non-flat" generalization is the edge-relaxed Wilson-loop extension (Regime II) and is not the working framework.

### GL(K) gauge-invariance claim (Theorem reference)

The manuscript invokes `\ref{thm:glk_invariance}` in §5.7 summary at line 1974: "The GL(K) gauge invariance theorem (Theorem glk_invariance) is essential as it guarantees that the learned projections W_Q, W_K are valid gauge transformations, explaining why transformers can learn arbitrary (invertible) attention patterns."

The theorem statement (locate via grep) establishes that the attention rule `β_{ij} = softmax(-D_KL(q_i ‖ Ω_{ij} q_j)/τ)` is invariant under simultaneous gauge transformation `μ_i → g_i μ_i, φ_i → g_i φ_i g_i^{-1}` (or analogous). This is the headline equivariance argument.

### Section §5 framing (`Attention/GL(K)_attention.tex:992`)

> "In this section we demonstrate that standard transformer self-attention emerges as a set of limiting cases of our gauge-theoretic framework."

### Section §5.7 summary (`Attention/GL(K)_attention.tex:1958`)

> "Under the successive limits: (i) isotropic beliefs with σ⁻² absorbed into learned projections, (ii) constant gauge (Ω_{ij} = Ω for all pairs), and (iii) learned gauge via projections (W_Q W_K^T = σ⁻²Ω^{-T} ∈ GL(d_k)), variational natural-gradient descent on the gauge-equivariant free energy is: [training update] and recovers the standard training update [...]"

The manuscript treats constant-gauge as Limit 2 of a chain of three.

## Canon excerpts — `external_canon_math.md` §2

### Gauge invariance vs gauge equivariance

> "- **Gauge invariant:** a quantity Q is invariant under the gauge action: `Q(g · s) = Q(s)`.
> - **Gauge equivariant:** a quantity transforms covariantly: `Q(g · s) = ρ(g) Q(s)` for some representation ρ.
>
> Predictions/observables are typically gauge invariant. Internal representations are typically gauge equivariant. Conflating these is a common error."

### Flat-bundle / trivial-bundle [Nakahara2003 §10.1]

A trivial principal G-bundle is one that admits a global section, equivalently `P ≅ M × G`. The connection on a trivial bundle has curvature `F = dA + A∧A`; flat bundles are those with `F = 0`. A flat principal bundle on a simply connected base has trivial holonomy. The `Ω_{ij} = exp(φ_i)exp(-φ_j)` parameterization is *globally trivial* in this sense (single chart, vanishing curvature on the chart) per the supplementary's own admission.

### Holonomy and trivial holonomy

> "Around a closed loop γ at point p, parallel transport gives `Hol(γ) ∈ G`. The holonomy group `Hol_p(A) ⊂ G` measures how far the connection is from flat."

For the vertex-parameterized `Ω_{ij} = exp(φ_i)exp(-φ_j)`, the triangle holonomy `Ω_{ij} Ω_{jk} Ω_{ki} = exp(φ_i)exp(-φ_j) · exp(φ_j)exp(-φ_k) · exp(φ_k)exp(-φ_i) = I` is identically trivial. The framework is *not* in the topologically non-trivial regime; it lives entirely in the trivial-holonomy class.

## Canon excerpt — gauge equivariance under specialization

A framework with action `G ↷ F` (group `G` acting on field space `F`) is *gauge equivariant* under the diagonal action `g · (f, A) = (ρ(g) f, Ad(g) A)` if the equations of motion are invariant under this action. **Specializing to a particular gauge `A = A_0`** (e.g., the trivial connection `A_0 = 0`) is a *gauge fixing*: it picks one representative from each orbit. The equations of motion, evaluated at the gauge-fixed configuration, are no longer manifestly equivariant — they have been reduced to ordinary equations in the chosen gauge. The original equivariance is *not destroyed* (it remains a property of the unrestricted equations), but it does not give the gauge-fixed equations any equivariance of their own.

This is the canonical reading from gauge theory ([Nakahara2003 §10.4], [Frankel2011 Ch. 18]): a gauge-fixed Lagrangian is not gauge-invariant; the gauge invariance is a property of the unfixed Lagrangian or of the path integral, which still admits the full gauge action up to gauge fixing.

## What this evidence does NOT settle

1. **Specialization vs gauge fixing.** Setting `Ω_{ij} = Ω` (constant) or `Ω_{ij} = I` (trivial) is mathematically a *specialization* of the framework (pick a particular `Ω ∈ GL(d_k)`). It is *also* a *gauge fixing* in the standard sense: by choosing the constant trivial element you have fixed the gauge to the trivial gauge. The two terminologies describe the same operation under different ontologies. The judge should decide which terminology the manuscript implicitly uses and whether the equivariance argument it makes downstream is consistent with that terminology.

2. **What "preserves gauge equivariance" means after gauge fixing.** Two readings:
   - **Strong reading.** "Preserves" = the gauge-fixed equations remain gauge-equivariant. This is **false** in standard gauge theory.
   - **Weak reading.** "Preserves" = the upstream framework (before specialization) remains gauge-equivariant; the specialization simply selects one orbit representative without altering the upstream theorem. This is **true**, trivially.

3. **The role of `W_Q, W_K` "as valid gauge transformations."** The §5.7 summary at line 1974 says `W_Q, W_K` ARE valid gauge transformations. Under the flat-bundle limit, the framework reduces to: gauge transformations are the W matrices. Two interpretations:
   - The framework imports gauge structure into the standard transformer (post-hoc identification).
   - The standard transformer was *always* implementing a gauge structure, made explicit by the framework.

   Both readings depend on whether the gauge structure has empirical content beyond the standard transformer's learned weight matrices.

4. **Comparison to standard gauge-equivariant networks [Bronstein2021, CohenGeigerKohlerWelling2018].** Standard gauge-equivariant networks are equivariant under a *compact* group like `SO(K)`. The manuscript uses `GL(K)`, a non-compact group with no canonical bi-invariant metric. Whether `GL(K)`-equivariance is operationally meaningful in the way `SO(K)`-equivariance is, especially after gauge-fixing to a single element of `GL(K)`, is open. Canon `external_canon_transformers.md` §7 flags this: "the user's framework ... specifically claims `GL(K)` (general linear) as the gauge group of attention, not a compact group like `SO(K)`. This is a less common choice — `GL(K)` is non-compact."

5. **Backward direction.** Does the flat-bundle limit *recover* gauge equivariance? Once you take Ω = I or Ω = const, the downstream structure is just standard transformer. Can you re-introduce gauge equivariance after the fact? Probably yes (the W matrices act as gauge transformations on the embedded representations), but this is a different equivariance (acting on learned weights) than the original (acting on the gauge frame field `φ_i`).
