# Evidence Pack — disc-rg-token-fixed-point

## Manuscript references

### Conjecture statement

- `Attention/GL(K)_attention.tex:2234` — section header `\subsection{Conjecture: The Transformer as a Renormalization Group Fixed Point}`.
- `Attention/GL(K)_attention.tex:2237` — explicit hedge: "we conjecture---but do not claim to have validated on trained models---that the standard transformer occupies a stable infrared fixed point of this flow."
- `Attention/GL(K)_attention.tex:2243-2247` — coupling definitions:
  - $g_1^{(\mathrm{orig})} = \|\Sigma_i - \sigma^2 I\| / \sigma^2$ (intrinsic anisotropy)
  - $g_2 = \|\Omega_{ij} - \Omega\| / \|\Omega\|$ (gauge variation)
  - $g_3 = \|H_{ijk} - I\|$ (linear holonomy), where $H_{ijk} = \Omega_{ij}\Omega_{jk}\Omega_{ki}$
- `Attention/GL(K)_attention.tex:2247` — total anisotropy split: $g_1^{(\mathrm{tot})} = g_1^{(\mathrm{orig})} + g_1^{(\mathrm{emer})}$ with $g_1^{(\mathrm{emer})} = \|\mathrm{Var}_A(\mu)\|/\sigma^2$.
- `Attention/GL(K)_attention.tex:2249-2254` — coarse-graining $R_n$ definition: meta-agent beliefs $\mu_A = (1/|A|)\sum_{i\in A}\mu_i$, $\Sigma_A = (1/|A|)\sum_i \Sigma_i + \mathrm{Var}_A(\mu)$.
- `Attention/GL(K)_attention.tex:2255-2260` — CLT-derived scaling under independence assumption: $(g_1^{(\mathrm{orig})})' = n^{-1/2} g_1^{(\mathrm{orig})}$, $g_2' = n^{-1} g_2$, $g_3' = n^{-1} g_3$, with $y_1 = -1/2$, $y_2 = -1$, $y_3 = -1$ (linear), $y_3^{(\mathrm{action})} = -2$ (squared norm).
- `Attention/GL(K)_attention.tex:2262-2272` — Conjecture `conj:rg_universality` parts (i)–(v):
  - (i) Intrinsic fixed point $g_1^{(\mathrm{orig}),*} = g_2^* = g_3^* = 0$ is a fixed point of $R_n$.
  - (ii) **Stability.** "All scaling dimensions are negative. The fixed point is infrared-stable."
  - (iii) **Universality.** All finite-coupling models flow (in the intrinsic channel) to the transformer limit under repeated coarse-graining.
  - (iv) Emergent anisotropy in the total channel from $\mathrm{Var}_A(\mu)$.
  - (v) Efficiency gap: $O(K^2)$ absorbed DoF $\Rightarrow$ $O(\sqrt{K})$ sample efficiency.

### Empirical concession (decisive for sub-claim 2)

- `Attention/GL(K)_attention.tex:2274-2275` — verbatim **manuscript admission**:
  > "We have verified that the CLT exponents are exact (to three significant figures) under direct averaging of synthetic i.i.d.\ perturbations, confirming the mathematical basis. However, when the same predictions are tested on attention-graph coarse-graining with spectral clustering (which introduces finite-size correlations), the measured exponents deviate: $y_2 \approx -0.6$ and $y_3 \approx +0.2$ versus the linear-norm predictions $-1$ and $-1$ (the action-norm prediction $y_3^{(\mathrm{action})} = -2$ applies to the squared holonomy $\|H-I\|^2$, not the linear form measured on the graph). These deviations arise from clustering artifacts (unequal cluster sizes, correlated assignments) rather than a failure of the underlying scaling argument, but they underscore that *validation on trained models is needed before the conjecture can be considered empirically supported*."

The sign reversal $y_3 = -1$ (predicted) versus $y_3 \approx +0.2$ (measured) is the load-bearing tension. A *positive* scaling dimension would make $g_3$ a **relevant** operator, growing under repeated coarse-graining, which would falsify part (ii) of the conjecture as applied to trained transformers.

- `Attention/GL(K)_attention.tex:2277` — closing paragraph claims the ordering of the three limits in §5 is dictated by the RG hierarchy ($y_2 = y_3 = -1$ contract faster than $y_1 = -1/2$), and reads LayerNorm as a projection toward the isotropic submanifold via the emergent-anisotropy term.

### Background sections invoked

- `Attention/GL(K)_attention.tex:995-1352` — §5 "Reduction to Transformer Attention" containing the three named limits (flat bundle, degenerate Σ, σ⁻²Ω⁻ᵀ absorption). The conjecture's claim that the RG flow dictates the ordering of these limits is referenced at `:2277`.
- `Attention/GL(K)_attention.tex:1716-1773` — multi-head $\bigoplus_a \mathfrak{gl}(d_{\mathrm{head}})$ block-diagonal structure (Section 5.4), relevant to part (v) of the conjecture.

## Prior debate verdicts (scope-defining)

- `docs/debates/2026-05-19-rg-construction-meta-agent/04_verdict.md` — **BLUE_WINS with calibration** on the **companion paper's** Wilsonian RG over the belief hierarchy in `Attention/Participatory_it_from_bit.tex` §4. **This is a different RG construction.** The judge for the current debate **may not** import that verdict as decisive evidence here. The token-graph $R_n$ map of GL(K)_attention.tex §8.4 is the only RG flow under adjudication.

## Canon excerpts — Wilsonian RG, CLT, spectral clustering

### Wilsonian RG: scaling dimensions and irrelevance

[Wilson1971] / [Wilson1974] / [Cardy1996 §3]: Under a coarse-graining map $R_b$ (Kadanoff block-spin or Wilsonian momentum-shell), a coupling $g$ transforms as $g' = b^{y_g} g + O(g^2)$ near a fixed point, where $y_g$ is the **scaling dimension** of the corresponding operator.
- $y_g > 0$: **relevant** operator, grows under coarse-graining, drives the system away from the fixed point.
- $y_g < 0$: **irrelevant** operator, shrinks under coarse-graining, system flows toward the fixed point in this direction.
- $y_g = 0$: **marginal**, requires higher-order analysis.

**IR-stable fixed point definition** [Cardy1996 §3.2; Goldenfeld1992 §9.4]: a fixed point such that all couplings in a neighborhood have negative scaling dimensions. The basin of attraction defines the universality class.

If any one scaling dimension at the candidate fixed point is positive, the fixed point is **not IR-stable** along that direction; the flow generically does not converge to it.

### Central limit theorem and N⁻¹/² scaling

For i.i.d.\ zero-mean random variables $X_i$ with finite variance $\sigma^2$, the sample mean $\bar{X}_n = (1/n)\sum_i X_i$ has standard deviation $\sigma / \sqrt{n}$ [Billingsley1995 §27]. For correlated variables with covariance structure $C_{ij}$, the variance of the mean is $(1/n^2) \sum_{ij} C_{ij}$; under strong off-diagonal correlations $C_{ij} \sim C_0$ the variance scales as $n^{-1}\cdot 1 = O(1)$, breaking the CLT $n^{-1/2}$ rate.

**Equivalently:** the CLT scaling dimensions hold *only* under the independence (or weak-correlation) assumption. Failure of the prediction on real data is direct evidence that the independence assumption is violated.

### Trained-transformer attention statistics

[Voita et al. 2019, "Analyzing Multi-Head Self-Attention"] and [Clark et al. 2019, "What Does BERT Look At?"] show that attention patterns in trained transformers are *highly structured* — heads specialize in syntactic relations (subject-verb agreement, coreference, next-token prediction), positional patterns (previous/next token, sentence boundaries), and cross-attention modes. These patterns produce *strong correlations between attention weights at neighboring tokens*. Independence of perturbations across tokens — the assumption underwriting the CLT exponents — is empirically unsupported for trained transformers.

[Park et al. 2019, "Spectrally Adaptive Common Spatial Patterns"] and the broader spectral-clustering literature [Ng-Jordan-Weiss 2001; von Luxburg 2007] document that finite-cluster effects produce systematic biases in eigenvalue-derived statistics. **Whether the observed deviation $y_3 \approx +0.2$ is artifact or signal is the empirical question.** Cardy1996 §3.5 and Cardy1996 Ch. 4 discuss finite-size scaling and explicit corrections; the standard practice in finite-size-scaling RG is to compute the correction analytically, not to ascribe deviations to "artifact" without quantification.

### Statistical-mechanics analogues and what "RG-on-token-graph" requires

[Kadanoff1966 / Wilson1971]: real-space block-spin RG requires (i) a coarse-graining map that combines local degrees of freedom into a renormalized degree of freedom at a coarser lattice, (ii) a rescaling of fields, (iii) demonstration that the renormalized Hamiltonian lies in the same parametric class as the original (closure under iteration). The user's $R_n$ at `:2249-2254` supplies the coarse-graining map but does not iterate beyond a single step in the manuscript; whether the meta-agent beliefs $(\mu_A, \Sigma_A)$ live in the same parametric family as the original beliefs $(\mu_i, \Sigma_i)$ such that $R_n$ can be iterated is not demonstrated.

(The closure question is a known concern in the companion-paper RG debate `2026-05-19-rg-construction-meta-agent`, where the augmented-class form `[Cardy1996 §3.3]` was the resolution; whether that resolution transfers to the token-graph $R_n$ is open.)

### CLT exponents alone are not RG

Trivial CLT averaging gives $\bar{X}_n - \mu = O(n^{-1/2})$ — this is a statement about i.i.d.\ random variables, not about a renormalization group flow on a *Hamiltonian* or *action*. Calling this "the RG flow" requires identifying:
- The action / free-energy functional whose couplings transform.
- The closure property of the coarse-graining (`Cardy1996 §3.3`).
- Demonstration that the scaling dimensions are computed at a critical point (Wilson's $\epsilon$-expansion or fixed-point linearization), not just from naïve dimensional analysis of the coarse-graining map.

The manuscript at `:2255-2260` derives the scaling dimensions from CLT (`n^{-1/2}` for $g_1$, `n^{-1}` for $g_2$, $g_3$), which is dimensional analysis of the averaging map, not from a fixed-point linearization of an RG flow. This is a real gap.

## What this evidence does NOT settle

- Whether the i.i.d. assumption at `:2255` ("Under the assumption that perturbations $\Delta_i$ and transport variations $\delta\Omega_{ij}$ are approximately independent across tokens") holds for *trained* transformer attention graphs. Empirically (Voita 2019, Clark 2019) it does not.
- Whether the spectral-clustering deviation $y_3 \approx +0.2$ is fully accounted for by clustering artifacts. The manuscript asserts this but offers no quantitative finite-size correction.
- Whether the $R_n$ map closes on the same parametric family ($\{(\mu_i, \Sigma_i, \phi_i)\}$) under iteration, which is required to call the construction an RG in the Wilsonian sense.
- Whether part (v) — the $O(\sqrt{K})$ efficiency gap — follows from anything other than naïve operator-counting of the absorbed DoF.
- Whether the "ordering of the three limits" claim at `:2277` (that the RG hierarchy dictates Limit~3 → Limit~2 → Limit~1) is anything more than a post-hoc rationalization, given that the limits were introduced in §5 before the RG analysis in §8.4.
