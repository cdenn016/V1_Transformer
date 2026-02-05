# Peer Review: "Attention, Transformers, and Backpropagation are Degenerate Limits of the Variational Free Energy Principle"

**Manuscript:** JMLR Attention 11-3-25.tex
**Author:** Robert C. Dennis
**Venue:** JMLR (preprint)
**Reviewer Recommendation:** Major Revision

---

## 1. Summary

This paper proposes that standard transformer attention, the 1/sqrt(d_k) temperature scaling, layer normalization, and backpropagation all emerge as degenerate (limiting) cases of a gauge-covariant variational free energy principle defined over a statistical fiber bundle. Each token is modeled as an "agent" carrying Gaussian beliefs (mu, Sigma) in a local gauge frame (phi), and inter-agent communication is mediated by GL(K) transport operators. The attention weights arise as softmax of negative KL divergences between gauge-transported beliefs. Three successive limits --- (i) isotropic covariances, (ii) flat bundle (constant gauge), (iii) learned projections absorbing sigma^{-2} and Omega --- recover standard scaled dot-product attention exactly.

The paper presents three lines of empirical evidence: (1) a BERT validation showing correlation between KL-based attention and standard attention across 144 heads, (2) agent-based simulations demonstrating vacuum symmetry and observation-driven symmetry breaking, and (3) GL(30) language models trained on WikiText-103 that achieve test perplexity ~135 without MLPs, activation functions, or learned W_Q/W_K/W_V projections. The paper additionally provides appendices on covariance dynamics, conditional uniqueness of the forward KL, and numerical methods.

---

## 2. Strengths

### 2.1 Ambitious and Original Conceptual Framework
The paper attempts a genuinely ambitious unification: deriving attention mechanisms from first principles via gauge theory and the free energy principle. The conceptual vision --- that tokens are agents performing variational inference over a fiber bundle --- is creative and provides a rich mathematical language for understanding transformer architectures. Even if the specific claims require qualification (see below), the overarching perspective is thought-provoking and has potential to stimulate new research directions.

### 2.2 GL(K) Invariance Observation
The identification that KL divergence is invariant under the full GL(K) group (Theorem 1) --- and that this implies W_Q, W_K can be interpreted as gauge transformations with Omega = W_Q W_K^T --- is a clean and useful observation. While the GL(K) invariance of f-divergences is a standard result in information geometry (as the author acknowledges), its application to reinterpret transformer projections is novel and yields the testable prediction about key-norm bias. This is arguably the strongest technical contribution.

### 2.3 Key-Norm Bias Prediction and Validation
The framework generates a concrete, non-obvious prediction: that key-norm heterogeneity introduces a systematic negative bias in attention allocation, with correlation rho ~ -0.35 across BERT heads (92.4% statistically significant). This prediction is derived from the gauge-theoretic formulation and validated empirically. The interpretation of layer normalization as implementing a gauge-theoretic cancellation condition is compelling and provides genuine explanatory value.

### 2.4 GL(30) Language Modeling as Proof of Concept
Training a language model that achieves test perplexity ~135 on WikiText-103 using *only* KL divergences, gauge transport, and natural gradient descent --- with no MLPs, activation functions, or standard attention projections --- is a non-trivial proof of concept. This demonstrates that the mathematical structure alone carries meaningful computational content, which supports the theoretical framework.

### 2.5 Symmetry Breaking Demonstration
The agent-based simulations comparing vacuum (no observations) vs. observation-driven dynamics clearly demonstrate the claimed symmetry breaking phenomenon. The convergence of all agents to identical norms in the vacuum case, versus differentiation under observations, is a clean and interpretable result.

### 2.6 Mathematical Presentation
The mathematical derivations are generally careful and well-presented. The step-by-step reduction from gauge-theoretic free energy to standard attention (Section 4) is clear and easy to follow. The proofs (GL(K) invariance, covariance dynamics, conditional uniqueness) are correctly executed.

---

## 3. Major Concerns

### 3.1 The "Derivation" of Backpropagation is Largely Tautological

The paper claims that "backpropagation emerges naturally from our framework" (Section 4.7). However, what is shown is simply that taking the gradient of a differentiable loss function yields the chain rule. This is calculus, not a derivation of backpropagation from first principles.

Specifically, the paper:
1. Defines a free energy functional F[{mu_i}]
2. Takes its gradient dF/dmu_i
3. Discretizes: mu_i^{t+1} = mu_i^t - eta * dF/dmu_i
4. Notes that for layered architectures, dF/dmu^{(l)} involves dmu^{(l+1)}/dmu^{(l)} (the chain rule)
5. Declares this to be backpropagation

But *any* differentiable loss function minimized by gradient descent yields the chain rule through layered compositions. The statement "backpropagation is not an optimization trick but a consequence of variational inference" is misleading --- it is a consequence of differentiable function composition, regardless of whether the loss is called "free energy" or anything else. The paper should either strengthen this claim with genuinely non-trivial content (e.g., showing that the *specific structure* of backpropagation through attention layers has a gauge-theoretic interpretation beyond the chain rule) or significantly moderate the claim.

### 3.2 The Three Limits Collapse the Gauge Structure

The paper's central mathematical result is that three limits reduce gauge-theoretic attention to standard attention. However, these limits are so aggressive that they eliminate essentially all of the gauge-theoretic structure:

- **Isotropic covariances** (Sigma_i = sigma^2 I): Eliminates all covariance structure, reducing the statistical manifold to Euclidean space. KL divergence becomes squared Euclidean distance.
- **Flat bundle** (Omega_ij = Omega): Eliminates all position-dependent gauge structure, all curvature, and all holonomy. The connection becomes trivial.
- **Learned projections absorb everything**: sigma^{-2} and Omega are absorbed into W_Q, W_K, making the gauge parameter unobservable.

After these limits, the derivation reduces to the well-known identity:

    -||mu_i - Omega*mu_j||^2 = -||mu_i||^2 - ||Omega*mu_j||^2 + 2*mu_i^T*Omega*mu_j

followed by softmax cancellation of query-dependent terms. The relationship between negative squared Euclidean distance and dot product is elementary. The question the paper must address more directly is: **what does the gauge-theoretic superstructure buy beyond this textbook identity?**

The answer may lie in the key-norm bias prediction and the design insights (layer normalization, multi-head structure), which are genuinely interesting. But the paper oversells the reduction as showing that transformers "are" a degenerate gauge theory, when a more accurate statement is that one *can* construct a gauge theory whose certain limit coincides with transformers --- which is a weaker claim.

### 3.3 GL(30) Experimental Results Require Better Contextualization

The paper reports test perplexity of 135.3 (3-layer) and 151.8 (1-layer) on WikiText-103, framing this as a "370x improvement over random chance." While technically accurate (random perplexity for vocab 50,257 is indeed ~50,000), this framing is misleading for two reasons:

1. **Any reasonable language model achieves similar improvement ratios.** A simple bigram model would also show 100x+ improvement. The relevant comparison is against standard transformer baselines of comparable parameter count (~50M parameters), which typically achieve perplexity 30-60 on WikiText-103. The paper's models underperform standard transformers by roughly 3-5x.

2. **Missing baseline comparisons.** The paper provides no comparison against:
   - Standard transformer of equal parameter count
   - Simple n-gram or count-based baselines
   - Other non-backpropagation methods (e.g., Hebbian learning, forward-forward algorithm)

   Without these comparisons, it is impossible to assess whether the gauge-theoretic inductive bias provides value *relative to existing approaches*, or whether the architecture is simply memorizing token co-occurrence statistics through a computationally expensive mechanism.

The paper should either include proper baselines or explicitly reframe the contribution as a proof of concept (which it is) rather than implying competitive performance.

### 3.4 BERT Validation Tests a Mathematical Identity, Not the Gauge Theory

The BERT validation computes KL-based attention weights using Q and K vectors *already learned by BERT* and compares them to BERT's dot-product attention. A mean correlation of r = 0.821 is presented as evidence for the gauge-theoretic framework.

However, this correlation is largely a mathematical consequence. For high-dimensional vectors where ||Q_i|| and ||K_j|| are approximately constant (enforced by layer normalization in BERT):

    -||Q_i - K_j||^2 = -||Q_i||^2 - ||K_j||^2 + 2*Q_i*K_j^T ~ 2*Q_i*K_j^T + const

So softmax(-||Q_i - K_j||^2 / tau) ~ softmax(Q_i*K_j^T / tau') for some effective temperature tau'. The correlation between these quantities is expected *independently of any gauge theory*. What the experiment actually validates is that BERT's layer normalization makes key norms approximately constant --- which is by design, not a prediction of the theory.

The BERT experiment would be more convincing if it tested a *prediction of the gauge theory that goes beyond this mathematical identity*, such as:
- Whether the optimal temperature tau exhibits the predicted scaling with d_k across models of different sizes
- Whether models without layer normalization show the predicted key-norm bias magnitude
- Whether the correlation pattern across layers matches gauge-theoretic predictions about representation structure

### 3.5 The Precision Choice (Eq. 20) is a Design Decision, Not a Derivation

A critical step in the paper is choosing the coupling precisions:

    Lambda_ij := tau * (Omega_ij * Sigma_j * Omega_ij^T)^{-1}

This choice makes the quadratic alignment term proportional to the KL divergence, but *only in the alignment regime* where Sigma_i ~ Omega_ij * Sigma_j * Omega_ij^T. The paper acknowledges this approximation but doesn't adequately discuss:

1. **When does the alignment regime hold?** For randomly initialized agents or early in training, covariances are far from aligned. The approximation is worst precisely when it matters most (early dynamics that determine convergence behavior).

2. **What happens outside this regime?** The quadratic form and KL divergence can differ substantially when covariances are mismatched. The paper should quantify the approximation error.

3. **Why this choice and not another?** Many other choices of Lambda_ij are possible. Each would lead to a different "derived" attention mechanism. The paper should acknowledge that the KL-based attention emerges from a *specific* design choice within the framework, not inevitably from the framework itself.

### 3.6 Figure/Text Inconsistencies in GL(K) Experiments

The gauge frame clustering figure (Fig. 8b) is titled "SO(42) Gauge Frames (PCA from 900D)" while the manuscript consistently describes GL(30) experiments with phi in gl(30) ~ R^{900}. For GL(30), the Lie algebra dimension is 30x30 = 900, which matches the "900D" label. But "SO(42)" is inconsistent with GL(30) --- SO(42) would have dimension 42*41/2 = 861, not 900.

This was traced to a bug in `transformer/analysis/semantics.py:identify_gauge_group()`, which inferred the gauge group from phi dimension using only the SO(N) formula N(N-1)/2, without checking for GL(K) (where phi_dim = K^2). For phi_dim=900, this computed n_approx=42, producing "SO(42)" instead of "GL(30)". The code has been fixed, but the existing figures and JSON data retain the old labels. For a journal submission, the figures must be regenerated with the corrected labels.

Additionally, the training curves (Fig. 7) show the 1-layer model with notably worse generalization (larger train-val gap) than described --- the text mentions that "training and validation curves track closely until late training," but the 1-layer figure shows a growing gap starting from ~step 10k.

---

## 4. Moderate Concerns

### 4.1 The 0D Base Manifold Eliminates Most Gauge Theory Content

The paper restricts to a 0-dimensional base manifold for the transformer connection, which eliminates gauge connections (A_mu), field strengths (F_mu_nu), curvature, and holonomy --- arguably the most interesting and distinctive features of gauge theory. What remains is essentially group-valued frame transformations between agents, which is a much simpler algebraic structure than a gauge theory proper.

The paper uses full gauge-theoretic language (connections, curvature, parallel transport, holonomy) in the introduction, general framework, and discussion, but then discards all of it when making contact with transformers. This creates a disconnect between the theoretical ambition and the actual content. The paper should be more transparent about this: what is presented is an *algebraic* structure (group actions on statistical manifolds), not a gauge *field* theory (which requires a non-trivial base manifold with spatial derivatives).

### 4.2 Symmetry Breaking Analogy is Underdeveloped

The paper draws an analogy between observation-driven agent specialization and spontaneous symmetry breaking, referencing Goldstone's theorem. However:

1. Goldstone's theorem predicts *massless excitations* (Goldstone modes) for each broken continuous symmetry generator. The paper does not identify or analyze these modes. Are there low-frequency excitations of the agent system corresponding to rotations along the gauge orbit? What is the spectrum of the Hessian at the symmetry-broken state?

2. In the strict sense, what is described is *explicit* symmetry breaking (observations act as an external field that selects a ground state), not *spontaneous* symmetry breaking (where the Hamiltonian is symmetric but the ground state is not). The vacuum state where agents converge to gauge-equivalent beliefs is closer to SSB, but the transition to the observed state is driven by an explicit symmetry-breaking term. The paper should be more precise about this distinction.

### 4.3 Conditional Uniqueness of Forward KL (Appendix C)

The conditional uniqueness theorem states that among f-divergences with linear coupling and exponential-family closure, forward KL is unique. This is a useful result, but the conditions are somewhat restrictive:

- The exponential-family closure assumption essentially *defines* the forward KL (it is the unique Bregman divergence generating the negative entropy potential).
- The theorem's value would be enhanced by showing that relaxing any single condition leads to a different class of divergences, and discussing whether those alternatives are viable.

The proof sketch (particularly 1 => 2) should be made more rigorous for a JMLR submission.

### 4.4 Natural Gradient Convergence Claims Need Qualification

The paper states: "natural gradient descent on the Fisher metric offers exponential convergence k = O(log(1/epsilon)) compared to the k = O(1/epsilon) rate of standard gradient descent." While natural gradient descent can achieve faster convergence under certain conditions (e.g., when the loss landscape has favorable curvature in the Fisher metric), the O(log(1/epsilon)) rate requires strong convexity in the natural gradient sense, which is not guaranteed for neural network training in general.

The paper partially qualifies this ("whether this translates to practical speedup... requires controlled empirical comparison"), but the initial claim should be stated with more care.

### 4.5 Missing Related Work

The paper omits several relevant lines of work:

- **Transformers as kernel methods**: The connection between softmax attention and kernel functions (Tsai et al., 2019; Katharopoulos et al., 2020) provides an alternative mathematical foundation.
- **Attention and Hopfield networks**: Ramsauer et al. (2021) show that attention implements modern Hopfield network retrieval, providing a complementary energy-based perspective.
- **Forward-forward algorithm**: Hinton (2022) proposes backpropagation-free training, directly relevant to the claim that backpropagation is a "degenerate limit."
- **Predictive coding networks**: These implement variational free energy minimization similar to the author's framework but have a longer and more established history in the neuroscience-ML interface.
- **The information bottleneck interpretation of deep learning**: Shwartz-Ziv & Tishby (2017) provide a related information-theoretic view of layer-wise representation learning.
- **Bayesian attention/uncertainty-aware attention**: Transformers with explicit uncertainty (Fan et al., 2020; Xiao et al., 2020) are directly relevant to the covariance-enriched representation.

### 4.6 Manuscript Length and Organization

At approximately 2,565 lines of LaTeX (~60 pages with figures), the manuscript is quite long. Significant portions of the appendix (Section A.1: general bundle theory, curvature types) reproduce standard differential geometry textbook material that JMLR readers with the requisite background would already know. This material should either be significantly condensed or moved to supplementary material, with appropriate references.

The main text itself could be tightened: the derivation in Section 4 is presented twice (once conceptually, once formally), and the discussion section repeats claims from the introduction almost verbatim.

---

## 5. Minor Issues

### 5.1 Notation and Formatting
- The paper uses both beta_ij (gauge theory) and alpha_ij (standard transformers) for attention weights. The switch between these is sometimes confusing, particularly in the BERT validation section.
- Several displayed equations use `$$ $$` rather than `\begin{equation}`, preventing cross-referencing. See, e.g., the equations between Eqs. 21 and 22, and in the Conditional Uniqueness appendix.
- Table 4 (GL(30) results) reports train PPL and test PPL but not validation PPL, despite the training curves showing separate validation curves.

### 5.2 Date and Version Inconsistencies
- The JMLR heading says 2025, but directory timestamps in the codebase suggest experiments were run in February 2026. The manuscript date should be current.
- The Code Availability section points to `github.com/cdenn016/epistemic-geometry`, but the GL(K) transformer code appears to live in a different repository (the Gauge-Transformer codebase). The paper should clarify which repository contains which experiments.

### 5.3 Reproducibility Gaps
- The reproducibility section mentions `generalized_simulation.py` and `configs/vacuum_exp.py`, but the actual codebase has different file organization (e.g., `simulation_runner.py`, `simulation_config.py`). Commands should be verified against the actual repository structure.
- The GL(30) experiments are trained for "50,000 training steps (~50M parameters per model)" --- these are confusingly different quantities. Steps and parameters should be described separately.
- Hyperparameters for the GL(30) experiments (Table 7) show tau = 1.0, but the BERT validation finds optimal tau = 19.0. The discrepancy should be explained (presumably these are different experimental contexts).

### 5.4 Claims That Require Softening
- "Attention may be a universal feature of multi-agent systems performing distributed inference under geometric constraints" (Section 6.4) --- this is speculative and should be clearly labeled as such.
- "We have offered a novel view of syntax and semantics itself; words, ideas, and knowledge as abstract communicating agents" (Section 6.4) --- this philosophical claim is not supported by the experimental results, which demonstrate token-level language modeling, not syntactic or semantic analysis.
- "Standard attention transformers behave remarkably similar to a 0-dimensional gauge theory" (Section 6.4) --- this should state "can be described as" rather than "behave similar to," since any mathematical object can be embedded in many different frameworks.

### 5.5 Low Variance Explained in PCA Projections
The gauge frame PCA (Figure 8b) shows the first three principal components explaining only 3.6% + 3.0% + 2.2% = 8.8% of total variance. The "clear categorical separation" described in the text is visible in these dimensions, but it accounts for less than 9% of the structure. The paper should report quantitative clustering metrics (e.g., silhouette scores, adjusted Rand index) rather than relying on visual inspection of low-variance projections.

Similarly, the belief embedding PCA explains only ~49% of variance in the first three components. The "tight central cluster" of content words vs. "spread" of punctuation could be an artifact of projecting high-dimensional data into 3D.

### 5.6 Training Stability Questions
The gradient norm plots (Fig. 7d, both panels) show characteristic spikes and oscillations, particularly in the phi (gauge frame) gradients. The 1-layer model shows phi gradient spikes of 4 orders of magnitude around step 30k-40k. While the model recovers, this suggests potential instability that should be discussed. Are these spikes related to gauge frame singularities (det(Omega) near 0)? Do they correspond to phase transitions in the learned representation?

### 5.7 Typos and Small Errors
- Line 29: `\jmlrheading{}{2025}{}{}{}{}` --- year should be updated
- Line 38: `\editor{TBD}` --- should be filled or removed
- Line 209: "Python 3.9+" with "NumPy, SciPy, and Joblib" --- the GL(K) experiments clearly require PyTorch, which is not listed
- Section 5.3.2 caption: "The 3-layer model converges faster (plateaus by step 25k)" --- from the figure, the model is still decreasing at step 25k
- The "epistemic death" terminology (Appendix A.2) is colorful but undefined for readers outside the active inference community

---

## 6. Questions for the Authors

1. **What specific testable predictions does the gauge theory make that cannot be derived from the simpler observation that KL(isotropic Gaussians) = scaled squared Euclidean distance?** The key-norm bias prediction is one; are there others?

2. **How does the GL(30) model compare against a standard transformer of equal parameter count on WikiText-103?** This comparison is essential for evaluating whether the gauge-theoretic inductive bias provides any practical advantage.

3. **Can you provide quantitative metrics (not just PCA visualizations) for the semantic clustering in belief and gauge frame spaces?** Silhouette scores, V-measure, or similar metrics would substantiate the "emergent semantic structure" claim.

4. **The paper claims attention weights can converge to uniform distributions while performance improves (Section 6.2). Is this observed in the GL(30) experiments? If so, what quantitative evidence supports this?** This is a strong claim that implies information is encoded primarily in transport operators rather than attention weights.

5. **For the symmetry breaking analysis, what is the spectrum of the Hessian at the vacuum fixed point? Are there zero modes corresponding to the gauge orbit (Goldstone modes)?** This would strengthen the SSB analogy from a metaphor to a formal result.

6. **Why does the gauge frame PCA figure show "SO(42)" while the text describes GL(30)?** Which experiment generated the figure?

---

## 7. Evaluation Summary

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Originality | High | Novel conceptual framework unifying gauge theory, FEP, and transformers |
| Technical Soundness | Moderate | Core math is correct; some claims overstated; key approximation under-discussed |
| Significance | Moderate | Interesting theoretical perspective; limited practical impact demonstrated |
| Clarity | Moderate | Well-written but too long; some sections redundant |
| Experimental Rigor | Low-Moderate | Missing baselines; BERT test validates math identity; figure inconsistency |
| Reproducibility | Low-Moderate | Code available but file paths inconsistent; missing dependency info |

---

## 8. Recommendation

**Major Revision.** The paper presents an original and ambitious theoretical framework with genuine insights (GL(K) invariance interpretation, key-norm bias prediction, proof-of-concept language modeling without standard NN components). However, several central claims are overstated relative to what is actually demonstrated:

1. The "derivation of backpropagation" should be reframed as showing that gradient descent on the free energy has a variational interpretation, not as deriving the chain rule from first principles.
2. The GL(30) experiments need proper baseline comparisons.
3. The BERT validation needs to distinguish between testing the gauge theory vs. testing a mathematical identity.
4. The figure inconsistency (SO(42) vs. GL(30)) must be resolved.
5. The manuscript should be shortened by ~30%, with standard textbook material moved to supplementary material.

The core contribution --- showing that transformer attention can be understood as a limiting case of variational inference on a statistical fiber bundle, with the GL(K) invariance providing a clean interpretation of learned projections --- is genuinely interesting and worth publishing. But the paper needs to be more honest about what is a derivation, what is an interpretation, and what is a rewriting of known results in new notation.
