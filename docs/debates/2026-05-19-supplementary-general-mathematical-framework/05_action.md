# Action — supplementary-general-mathematical-framework

**From verdict:** RED_WINS (narrow, editorial scope)

## Summary of verdict

The compound claim that `\section{General Mathematical Framework}` of `Attention/GL(K)_supplementary.tex` (lines 46–177) is "complete and mathematically/theoretically pure" fails on the "complete" prong. The "mathematically pure" prong survives — no equation in the section is incorrect, and the bundle scaffold matches [Nakahara 2003 §9–10] canonically. The "complete" prong fails because sub-claims δ (natural-gradient information geometry) and ε (variational EM machinery) are textually absent from lines 46–177, and the section contains zero internal forward references to the supplementary chapters (§B at line 178+, §C at line 388+, §D at line 611+) where the deferred material lives.

Sub-claims α (bundle scaffold) and γ (gauge connection and transport) verified per [Nakahara 2003 §9–10]; both sides agreed in rebuttals. Sub-claim β (Gaussian-belief state space representations) partially verified — representations defined at supplementary lines 55–62, closed-form Gaussian KL deferred to §B lines 199–219. The notational issue at line 76 (`p_i := σ^i_p ∈ B_p`) vs main paper line 602 (`p_i(k_i) ∈ E_q`) is a primitive-level type-name clash between a section and a probability density.

## Recommended action

Four editorial corrections to `Attention/GL(K)_supplementary.tex` §General Mathematical Framework, plus one optional title narrowing. No equation changes, no implementation changes, no structural revision.

### Edit 1 — closing forward-reference paragraph at end of §General Mathematical Framework (after line 174)

Insert a one-paragraph forward-reference block immediately before `\section{Covariance Dynamics and Equilibrium Analysis}` at line 178:

> The Fisher-Rao metric, Gaussian KL closed form, and covariance dynamics are developed in §\ref{app:covariance_dynamics}; the gauge-frame gradients and differential of the matrix exponential are developed in §\ref{app:gauge_frame_gradients}; the natural-gradient descent on the Gaussian manifold and the SPD retraction are developed in §\ref{app:variational_descent}; the variational free energy functional and its E-step/M-step decomposition are developed in the main paper §3.4–§3.5.

This converts the existing on-record deferral architecture into a reader-visible scaffold and satisfies the "adequate forward references" precondition for the charitable reading of "complete."

### Edit 2 — disambiguate `p_i` notation at line 76

Replace at line 76:

> `We write $q_i(c) := \sigma^i_q(c) \in \mathcal{B}_q(c)$ and $p_i(c) := \sigma^i_p(c) \in \mathcal{B}_p(c)$ for the belief and model at base point $c$.`

with:

> `We write $q_i(c) := \sigma^i_q(c) \in \mathcal{B}_q(c)$ for the belief and $s_i(c) := \sigma^i_p(c) \in \mathcal{B}_p(c)$ for the model at base point $c$. (The symbol $p_i$ is reserved in the main paper for the belief-channel base prior $p_i(k_i) \in \mathcal{B}_q$, an object distinct from the model section $s_i \in \mathcal{B}_p$ defined here.)`

This eliminates the primitive-level naming clash with main paper line 602 and makes the supplementary's agent state `(q_i, s_i, φ_i)` reconcilable with the main paper's `(q_i, p_i, s_i, r_i, φ_i)` (with `p_i, r_i` derived as cross-scale shadows in the Participatory framework, or treated as PriorBank primitive data in the gauge-transformer framework).

### Edit 3 — qualify §A.2.1 Hierarchical Meta-Agent Emergence (lines 96–130)

The model-consensus equation at line 103 introduces `s_i` without a prior definition. After Edit 2 defines `s_i` at line 76, this is resolved. Add one additional sentence to §A.2.1 (after line 104):

> The cross-scale prior-propagation construction $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ developed in the companion paper~\citep{Dennis2025it} treats priors as transported shadows of the meta-agent's posterior at the next-higher scale. The gauge-theoretic transformer of the main paper treats priors as primitive boundary data (the per-token PriorBank, see main paper Section~\ref{sec:embeddings}); the hierarchical formulation sketched above is the framework-level scaffold that the companion paper exercises in detail.

### Edit 4 — Čencov 1982 citation when Fisher-Rao metric is first invoked

After Edit 1 introduces the forward reference at the end of §General Mathematical Framework, the first natural invocation of the Fisher-Rao metric is at §D Variational Gradient Descent line 615 (per blue's verification at `03_blue_rebuttal.md`). At line 615 (and at the forward-reference paragraph from Edit 1 if Fisher-Rao is named there), add citation `\citep{Chentsov1982}` (or `\citep{Cencov1982}` matching the bib entry — verify which citekey exists). The Participatory paper cites Čencov at line 510 for the sufficient-statistics-invariance uniqueness theorem; the same citation should appear in the supplementary at the corresponding point.

### Optional — narrow chapter title

Optionally narrow the §General Mathematical Framework title to "Bundle Scaffold and Gauge Connection" to match the four-subsection scope (Principal Bundle, Agents and Multi-Agent Systems, Bundle Morphisms, Gauge Frames and Connections). Edit 1 (forward references) is the lighter-weight remedy and is sufficient.

## Bib additions

Verify the existence of the Čencov 1982 bibkey in `Attention/references.bib`. Per the verdict's note at line 41, the reference exists at lines 1832/1854. If absent, add:

```
@book{Chentsov1982,
  author = {Chentsov, N. N.},
  title = {Statistical Decision Rules and Optimal Inference},
  series = {Translations of Mathematical Monographs},
  volume = {53},
  publisher = {American Mathematical Society},
  year = {1982},
  address = {Providence, RI},
  note = {Original Russian edition 1972.}
}
```

## Cumulative debate-series state

Twelfth debate in the gauge-transformer manuscript audit series. Closed queue:

1. §5 transformer reduction (RED_WINS).
2. Softmax-β stationarity (RED_WINS).
3. Sub-claim A flat bundle (BLUE_WINS).
4. Sub-claim B degenerate Σ (BLUE_WINS).
5. Sub-claim C QK^T identification (BLUE_WINS).
6. Sub-claim D V identification (BLUE_WINS).
7. Canonical F vs surrogate (RED_WINS).
8. Multi-head block-diagonal (BLUE_WINS).
9. Route 1 untied carving (RED_WINS).
10. FFN softmax-gradient correction (RED_WINS).
11. §3 Gauge-Covariant VFE (RED_WINS narrow).
12. **Supplementary §General Mathematical Framework (RED_WINS narrow, this debate).**

The §3–§5 main-paper derivation chain is now fully audited; the supplementary §A is now audited. Logical follow-ups (deferred unless the user prioritizes):
- Supplementary §B Covariance Dynamics — derivation of σ-fixed-point equations and gradient flow.
- Supplementary §C Gauge Frame Gradients — KL gradient through transport and matrix-exp differential.
- Supplementary §D Variational Gradient Descent — natural gradient implementation specifics.
- Participatory_it_from_bit.tex §Theory — separately audited as the companion paper's foundational chapter.

## Follow-up debates

None required from this verdict. Optional follow-ups would target other supplementary chapters (§B, §C, §D) or the Participatory paper's §Theory, but they are not prerequisites for the §General Mathematical Framework edits.
