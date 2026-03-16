# Peer Review: "Attention as Gauge-Theoretic Variational Inference"

**Manuscript:** GL(K)_attention.tex
**Author:** Robert C. Dennis (Independent Researcher)
**Target Venue:** Journal of Machine Learning Research (JMLR)
**Reviewer Date:** 2026-03-16
**Recommendation:** Major Revisions

---

## Summary Statement

This manuscript proposes a gauge-theoretic framework that derives transformer attention from first principles, modeling each token as a Gaussian agent on a statistical fiber bundle whose inter-agent communication is mediated by GL(K) gauge transport. The generalized attention weight $\beta_{ij} = \text{softmax}(-D_{\text{KL}}[q_i \| \Omega_{ij} q_j]/\tau)$ emerges from variational free energy minimization of a mixture-of-sources generative model. Under successive limits (isotropic covariances, constant gauge, learned projections), the standard transformer attention formula $\text{softmax}(QK^\top/\sqrt{d_k})$ is recovered. The framework is validated through (a) correlation analysis of KL-based vs. dot-product attention on frozen BERT across 105 passages ($\bar{r} = 0.804$), and (b) training a GL(10) gauge transformer on WikiText-103 that achieves perplexity 71.6 without MLPs, learned projections, or activation functions.

**Overall Assessment:** This is an ambitious, intellectually stimulating manuscript that offers a genuinely novel geometric perspective on transformer attention. The mathematical framework is carefully constructed, and the derivation connecting gauge-theoretic VFE to standard attention is technically sound. The identification of transformer architectural choices (masking, ALiBi, RoPE, layer normalization, residual connections, multi-head structure) as consequences of the variational principle is the paper's greatest conceptual contribution. However, the manuscript suffers from several significant issues that require revision: the empirical validation, while interesting, does not adequately distinguish the theory from simpler mathematical explanations; the experimental comparison is confounded by major parameter-count asymmetries; and the manuscript's length and scope require substantial tightening for a journal submission.

### Key Strengths

- A mathematically rigorous and internally consistent framework that connects attention, gauge theory, and variational inference in a principled way
- The derivation of standard attention as a degenerate limit is clean, well-structured, and convincing
- The conditional uniqueness theorem for the forward KL divergence (Appendix A) is an elegant result
- The comprehensive correspondence table (Table 3) unifying multiple architectural choices under one principle is valuable
- Experimental validation spans multiple architectures and includes a novel neural-network-free language model
- Honest treatment of limitations and clear articulation of testable predictions

### Key Weaknesses

- The BERT validation conflates an algebraic identity with empirical evidence for gauge theory (acknowledged but insufficiently addressed)
- The GL(K) language model comparison is confounded by a 14x embedding dimension asymmetry with the parameter-matched baseline
- The manuscript is excessively long (~2,880 lines) for JMLR and needs significant streamlining
- Several "derivations" are better described as structural analogies (the D/S distinction in Table 3 is important but underemphasized)
- The RG universality conjecture and associated sections feel premature and insufficiently motivated by the current evidence

---

## Major Comments

### M1. The BERT Validation Primarily Tests an Algebraic Identity, Not Gauge Theory

The manuscript acknowledges (Section 5.2.1, "Scope of the Validation") that the high $\alpha$-$\beta$ correlation follows largely from the algebraic identity $Q_i K_j^\top / \sqrt{d} = (-\|Q_i - K_j\|^2 + \|Q_i\|^2 + \|K_j\|^2) / 2\sqrt{d}$, which under approximately constant norms (layer normalization) reduces the KL-distance and dot-product forms to monotonic transformations of each other. The four claimed non-trivial predictions (temperature scaling, key-norm bias, entropy matching, per-layer variation) are reasonable but modest in discriminative power:

- **Temperature scaling** ($\tau \approx 2\sqrt{d_k}$): The factor $\sqrt{d_k}$ is already well-known from the original Vaswani et al. scaling argument. The additional factor of 2 from the KL $\frac{1}{2}$ prefactor is a consequence of switching between the dot-product and squared-distance parameterizations; any squared-distance formulation would predict a similar factor. This is not unique to gauge theory.
- **Key-norm bias**: The prediction that key-norm heterogeneity introduces bias is a generic consequence of the expansion $\|Q-K\|^2 = \|Q\|^2 + \|K\|^2 - 2QK^\top$, not specific to gauge geometry. The modest effect size ($\bar{\rho} = 0.256$) further limits its discriminative value.
- **Entropy matching**: At the optimal temperature (by definition, the one maximizing correlation), entropy matching is expected from any well-calibrated temperature parameter.

**Suggestion:** Reframe the BERT validation more modestly as a *consistency check* rather than a *validation of gauge theory*. The real validation of the framework's value lies in the GL(K) language model and the architectural taxonomy. The BERT section could be significantly shortened (perhaps 2-3 pages rather than ~8), with the algebraic identity acknowledged upfront and the residual predictions presented as secondary corroborating evidence.

### M2. GL(K) Language Model Comparison is Confounded

The experimental comparison suffers from significant confounds that weaken the central empirical claim:

- **The "outperforms by 1.66x" comparison** (PPL 71.6 vs. 118.6) is at matched *embedding dimension* ($d_{\text{model}} = 90$), where the standard transformer has only 4.6M parameters vs. the GL(K) model's 58.8M. This is a 12.8x parameter advantage for the gauge model. The per-token covariance matrices and gauge frame parameters ($\Sigma_i \in \mathbb{R}^{K \times K}$, $\phi_i \in \mathbb{R}^{K^2}$) constitute an enormous parameter overhead that is not present in the standard architecture.
- **The parameter-matched comparison** (PPL 71.6 vs. 48.5 at ~84.2M params) shows the standard transformer *outperforms* the gauge model by 32%. This is the more relevant comparison for practical claims about model quality.
- **Different positional encoding schemes** are used: the standard transformer uses learned positional embeddings while the GL(K) model uses RoPE (which the manuscript itself derives as a position-dependent gauge frame restriction to $\text{SO}(2)^{d_k/2}$). While both models have positional encoding, the use of different schemes introduces a confound---any performance difference could partly reflect the choice of positional encoding rather than the gauge-theoretic architecture per se.
- **Training for only 1 epoch** on WikiText-103 means both models are likely undertrained, making perplexity comparisons less informative.

It is noteworthy that the GL(K) model's use of RoPE is itself a prediction of the gauge framework (Section 3.5 derives RoPE as position-dependent gauge frames in $\text{SO}(2)^{d_k/2}$), which lends internal consistency. However, since the standard baseline uses a different positional encoding scheme (learned embeddings), attributing performance differences solely to gauge-theoretic attention vs. dot-product attention is complicated by this confound.

**Suggestion:** (a) Present the parameter-matched comparison as the primary result, not the embedding-matched one. The "1.66x improvement" headline is misleading given the parameter asymmetry. (b) Add intermediate baselines: a standard transformer at $d_{\text{model}} = 90$ with MLP disabled (to isolate the attention mechanism contribution) and with parameter count equalized via wider layers. (c) Include a baseline where both models use the same positional encoding (e.g., both with RoPE) to isolate the attention mechanism contribution. (d) Consider training both models to convergence (multiple epochs) for a fairer comparison. (e) Report FLOPs per training step and total training compute, since the gauge model requires matrix exponentials and per-pair KL computations that are substantially more expensive.

### M3. Manuscript Length and Structure

At ~2,880 lines of LaTeX (approximately 45-50 pages with figures), the manuscript is roughly twice the typical length for a JMLR submission. The content spans: gauge-theoretic framework development, reduction to standard attention, architectural correspondences (masking, ALiBi, RoPE, multi-head, FFN, residual connections, layer normalization), BERT validation across 5 models and 105 passages, GL(K) language modeling, symmetry breaking simulations, conditional uniqueness of forward KL, RG universality conjecture with numerical validation, and discussion of future directions. This is at least two (possibly three) papers' worth of material.

**Suggestion:** Consider one of the following reorganizations:
- **Option A (Recommended):** Focus the main paper on the core framework (Sections 2-4) and the GL(K) language model (the strongest empirical evidence). Move the BERT validation, RG universality, and symmetry breaking simulations to a companion paper or supplementary material.
- **Option B:** Retain the current scope but drastically compress each section. The BERT validation can be 2-3 pages. The RG section can be 1-2 pages stating the conjecture with numerical evidence, deferring the detailed analysis to a follow-up.

### M4. The "Derived" vs. "Structural" Distinction Needs Stronger Treatment

Table 3 introduces a valuable distinction between "D" (derived from the variational principle) and "S" (structural correspondence). This is intellectually honest and important, but several entries merit scrutiny:

- **Layer normalization (S):** The manuscript argues that LN eliminates key-norm bias, which is a necessary condition for the isotropic limit. But LN was not *predicted* by the gauge framework; it was imposed empirically in transformers. The framework explains *why* LN is beneficial (a post-hoc rationalization) but does not uniquely predict *which* normalization scheme would be used.
- **GELU/SiLU activation (S):** The Boltzmann-gated linear unit derivation (Section 3.6.1) is suggestive but involves several approximations and the $n_s = 2$ binary limit to recover SiLU. The connection to GELU requires further approximations via the CDF of a Gaussian energy. This derivation is interesting but stretches the claim of "structural correspondence."
- **Backpropagation (S):** Characterizing the chain rule as "variational message-passing" is a structural analogy, not a derivation. Any differentiable system has a chain rule.
- **Residual connections (D):** Labeled as derived, but residual connections emerge from *any* iterative update scheme $x_{t+1} = x_t + \Delta x_t$, not specifically from gauge-theoretic VFE.

**Suggestion:** (a) Include a clear paragraph in the discussion explicitly stating which predictions are *unique* to gauge theory (i.e., could not be obtained from simpler frameworks like kernel attention, Hopfield networks, or generic variational inference) versus which are general properties of variational/iterative optimization. (b) Consider downgrading "residual connections" from D to S. (c) The GELU/SiLU derivation would be more appropriately presented as a suggestive analogy rather than a structural correspondence.

### M5. The RG Universality Section is Premature

The RG universality conjecture (Conjecture 6.1) and its numerical validation occupy approximately 4 pages and make predictions (scaling exponents, compute crossover, emergent anisotropy) that are validated only on synthetic data. The connection between the CLT-based scaling argument and actual trained transformers is speculative. Specific concerns:

- The CLT validation confirms that averaging $n$ independent random matrices reduces norms as $1/\sqrt{n}$---a mathematical fact that does not require the gauge framework.
- The graph-based RG flow shows significant deviations from predictions ($y_2 = -0.66$ vs. predicted $-1.0$; $y_3 = +0.17$ vs. predicted $-2.0$), attributed to finite-size effects, but the deviations are large enough to question the quantitative predictions.
- The "Gaussian universality class" identification ($\nu = 2$, $\eta = 0$) follows from the CLT assumptions, making it tautological.

**Suggestion:** Either (a) significantly condense this section to a brief conjecture with testable predictions (1 page), explicitly acknowledging that validation on *trained* models is needed, or (b) defer the RG analysis entirely to a companion paper where it can be properly developed with trained-model evidence.

---

## Minor Comments

### m1. Missing Error Analysis on GL(K) Perplexity
The reported test perplexity of 71.6 has no confidence interval, standard deviation across runs, or sensitivity analysis to random seed. For a single-run result at 1 epoch, variability could be significant. Please report at least 3 independent runs with different seeds.

### m2. Holonomy Claims and Flat Bundle
The vanishing holonomy result (Theorem 3.2) is presented as a theorem of the architecture, which is correct for the parameterization $\Omega_{ij} = \exp(\phi_i)\exp(-\phi_j)$. However, this is a *consequence of the parameterization choice*, not a discovery about transformers. Any cocycle-form transport has trivially vanishing holonomy. The significance is in the *hypothesis* that this flat bundle corresponds to compositional language structure, which is interesting but untested.

### m3. Notation Overload
The manuscript uses $K$ for both belief fiber dimension and as part of $D_{\text{KL}}$, and $q_i$ denotes both a distribution and its density. While conventions are stated, this creates potential confusion. Consider using $d$ for the fiber dimension and reserving $K$ for KL-related notation.

### m4. Figure Quality
The TikZ fiber bundle figure (Figure 1) is well-crafted. However, several experimental figures (e.g., correlation distributions, training curves) appear to be screenshots or low-resolution exports. For JMLR submission, all figures should be vector graphics (PDF/SVG) at publication quality.

### m5. Symmetry Breaking Simulations (Section 6.1)
The symmetry breaking simulations use SO(3) on a 2D base manifold, which differs from the language modeling setup (GL(10) on a 0D base). The manuscript acknowledges this but still presents the simulations as evidence for the symmetry breaking prediction. The simulations would be more convincing if performed with the same gauge group and dimensionality as the language model.

### m6. Temperature Dispersion Analysis
Section 5.2.4 introduces per-head temperature dispersion as an explanation for RoBERTa's lower correlation. This is a reasonable hypothesis but is presented as if validated, when it is actually a post-hoc explanation. The analysis shows correlation between dispersion and mean $r$, but does not demonstrate causation. Consider tempering the language.

### m7. Prior Bank and Embedding Matrix Correspondence
The manuscript states that the prior bank $\{(\mu_p^c, \Sigma_p^c, \phi_c)\}$ reduces to the standard embedding matrix in the deterministic isotropic limit. This is correct but worth noting that the prior bank has $O(V \cdot K^2)$ parameters (from the covariance and gauge frame entries) while the embedding matrix has $O(V \cdot K)$. This contributes to the parameter-count asymmetry discussed in M2.

### m8. Code Availability
The manuscript references a GitHub repository (epistemic-geometry), but the code should include specific version tags, seeds, and configuration files for reproducibility. Consider depositing a tagged release on Zenodo for permanent archival.

### m9. Missing Baselines
The language modeling comparison lacks several relevant baselines:
- A standard transformer using RoPE (matching the GL(K) model's positional encoding) to control for the positional encoding confound
- A GL(K) model using learned positional embeddings (matching the standard baseline) for the same reason
- Linear attention models (to compare with the kernel interpretation)
- A pure distance-based attention model without the gauge machinery (to isolate the contribution of gauge transport)

### m10. Scope of "Without Neural Components"
The claim that the GL(K) model operates "without neural components" requires qualification. The model still uses: (a) a linear output projection $W_{\text{out}}$ (which *is* a neural network layer), (b) softmax normalization, (c) gradient descent with Adam optimizer, and (d) backpropagation through the computation graph. The claim should be more precisely stated as "without learned attention projections, MLPs, or pointwise activation functions."

### m11. Citation Balance
The related work in the introduction covers kernel perspectives, Hopfield networks, and information bottleneck, but does not discuss recent work on linear attention variants (Katharopoulos et al. is mentioned but cursorily), state-space models, or the extensive literature on attention as optimization (e.g., Ramsauer et al.'s modern Hopfield connection deserves deeper engagement). The manuscript would benefit from a more thorough comparison with existing mathematical frameworks for attention.

### m12. Elitzur's Theorem Caveat
The paragraph on Elitzur's theorem (Section 2.5) is appropriate but somewhat buried. Given the potential for confusion between gauge redundancy and physical symmetry breaking, this distinction deserves more prominent treatment, perhaps as a clearly labeled remark or a boxed caveat.

### m13. Bayesian Validation
The Bayesian hierarchical model (Section 5.2.5, referenced but not fully shown in the main text) uses a shrinkage estimator that raises the grand mean from 0.804 to 0.867. While methodologically sound, this creates the impression of "boosting" the headline number. Present both the raw and Bayesian-corrected means with equal prominence, and emphasize that the raw mean is the more conservative estimate.

---

## Questions for Authors

1. **Uniqueness of the gauge-theoretic explanation:** Can you identify a prediction of the gauge framework that is (a) quantitative, (b) testable with existing pretrained models, and (c) not derivable from simpler mathematical frameworks (kernel attention, Hopfield networks, or generic variational EM)? The temperature prediction is close but is arguably also derivable from dimensional analysis of the squared-distance form.

2. **Scalability:** What are the computational costs (FLOPs, wall-clock time) of the GL(K) model relative to a standard transformer at matched parameter count? The manuscript acknowledges 10-30x overhead from matrix exponentials but does not provide concrete measurements.

3. **Non-flat bundles:** The manuscript hypothesizes that compositional language operates in the flat bundle regime. Are there specific linguistic phenomena (pragmatics, irony, garden-path sentences) where non-trivial holonomy would be predicted? This would provide a falsifiable prediction distinguishing the framework from flat-connection alternatives.

4. **Value aggregation asymmetry:** The manuscript notes that in standard transformers (and especially RoPE), the gauge transport is applied to attention scoring but not to value aggregation. In the gauge framework, the same $\Omega_{ij}$ mediates both. Does this asymmetry in standard transformers represent a loss of gauge consistency, and is there empirical evidence that applying transport to both scoring and aggregation improves performance?

5. **Convergence of VFE iterations:** How many inner E-step iterations ($T_{\text{inner}}$) are used in practice, and how sensitive is the model's performance to this hyperparameter? Is there evidence that the beliefs converge to a fixed point within the allocated iterations?

---

## Reporting Standards Assessment

| Criterion | Status | Notes |
|---|---|---|
| Reproducibility | Partial | Code available but no version tags; single-run perplexity without confidence intervals |
| Statistical reporting | Good | CIs, SEs reported for BERT validation; missing for GL(K) model |
| Baselines | Incomplete | Missing matched-PE baselines, linear attention, ablations |
| Figures | Adequate | Some figures may need vector format for publication |
| Data availability | Good | WikiText-103 is public; BERT models are public |
| Hyperparameter sensitivity | Missing | No ablation over $K$, gauge group dimension, $\alpha$, $T_{\text{inner}}$ |

---

## Final Assessment

This manuscript makes a genuinely original theoretical contribution by connecting transformer attention to gauge theory and variational inference. The mathematical framework is carefully constructed, internally consistent, and produces a compelling taxonomic organization of transformer architectural choices. The conditional uniqueness theorem for forward KL and the GL(K) language model are particularly noteworthy contributions.

However, the empirical evidence does not yet meet the standard required to support the manuscript's strongest claims. The BERT validation largely confirms an algebraic identity rather than testing gauge theory specifically. The GL(K) language model comparison is confounded by parameter asymmetries and limited training. The RG universality section is premature. The manuscript's excessive length obscures its genuine contributions.

With focused revision --- particularly clarifying the empirical claims, tightening the manuscript scope, adding proper baselines and ablations, and more carefully distinguishing derived results from structural analogies --- this could become a significant contribution to the theoretical understanding of attention mechanisms. The mathematical framework itself is sound and valuable; the revision should ensure the empirical evidence matches the ambition of the theory.

**Recommendation:** Major revisions, with the expectation that a revised manuscript addressing the above concerns would be suitable for publication at JMLR.
