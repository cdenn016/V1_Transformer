# Research Directions for the VFE Gauge-Theoretic Framework

## Abstract

The Gauge-Transformer implements a variational free energy (VFE) objective over Gaussian belief tuples $(\mu, \Sigma, \phi)$ with GL$^+(K)$ gauge transport, replacing all neural attention components with information-geometric computation. This document identifies ten research directions that extend the framework's mathematical foundations into fiber bundle geometry, information geometry, topological field theory, and renormalization. Each direction is grounded in the existing codebase infrastructure, positioned against the current literature (2023--2026), and assessed for tractability and novelty. Direction 7c ($\alpha$-divergence generalization) receives an expanded treatment with closed-form derivations and gradient formulas verified by symbolic computation.

---

## 1. Intra-Fiber VFE Trajectories

The fiber at each token position $i$ in the Gauge-Transformer is the space of Gaussian beliefs $\mathcal{G}_K = \{\mathcal{N}(\mu, \Sigma) : \mu \in \mathbb{R}^K, \Sigma \in \text{SPD}(K)\}$. This space carries the Fisher-Rao Riemannian metric

$$ds^2 = d\mu^\top \Sigma^{-1} d\mu + \tfrac{1}{2}\operatorname{tr}(\Sigma^{-1} d\Sigma \, \Sigma^{-1} d\Sigma),$$

making it a $(K + K(K+1)/2)$-dimensional Riemannian manifold of constant negative curvature (for fixed $K$, the sectional curvatures of the Fisher-Rao metric on the full Gaussian family are all $\leq 0$).

During VFE E-step iterations, each token's belief traces a path $(\mu^{(t)}, \Sigma^{(t)})_{t=0}^T$ through this fiber. The existing `TrajectoryRecorder` (`transformer/analysis/trajectory.py`, lines 119+) records beliefs across layers but does not capture the iteration-level dynamics within each layer's E-step. The following sub-questions are open.

**1a. Geodesic deviation.** The natural gradient update $\theta^{(t+1)} = \theta^{(t)} - \eta \, G^{-1}(\theta^{(t)}) \nabla F(\theta^{(t)})$, where $G$ is the Fisher information matrix, follows geodesics of the information manifold in the continuous-time limit. Measuring the geodesic deviation $\|(\mu^{(t)}, \Sigma^{(t)}) - \gamma(t)\|_{\text{FR}}$, where $\gamma$ is the Fisher-Rao geodesic with the same endpoints, would quantify how well the discrete VFE iterations approximate information-geometric gradient flow. Deviations would characterize the curvature of the VFE loss landscape relative to the Fisher-Rao metric.

**1b. Lyapunov exponents.** Perturbing initial beliefs $(\mu^{(0)} + \epsilon, \Sigma^{(0)})$ and tracking the divergence of resulting trajectories gives Lyapunov exponents for the E-step dynamical system. High exponents indicate sensitive dependence on initial conditions (chaotic belief dynamics); low exponents indicate robust convergence. These could be computed per-token and per-layer, revealing which positions have stable versus fragile inference.

**1c. Trajectory arc length as semantic complexity.** The arc length $L = \int_0^T \sqrt{ds^2/dt^2} \, dt$ in the Fisher-Rao metric measures the total "inferential work" the model performs at each position. A testable prediction: content words (nouns, verbs carrying semantic payload) require longer trajectories than function words (determiners, prepositions); ambiguous tokens explore larger volumes of the fiber before settling.

**1d. Phase portraits.** Project the fiber into 2D via PCA on the Fisher metric and visualize the VFE flow field. Fixed points are attractors (stable beliefs); saddle points indicate bifurcations in the inference landscape.

**Infrastructure.** `TrajectoryRecorder` (layer-level tracking exists), `vfe_gradients.py` (iteration loop), `gauge_preconditioner.py` (Fisher/Killing metric computation exists). Missing: per-iteration recording within E-step, Fisher-Rao distance computation, geodesic solver, Lyapunov estimator.

---

## 2. Explicit Curvature Tensor and Yang-Mills Energy

The codebase computes holonomy $C_{ijk} = \exp(\delta_{ij} G) \cdot \exp(\delta_{jk} G) \cdot \exp(\delta_{ki} G)$ around triangular loops (`transformer/analysis/holonomy.py`, lines 32--86) as a proxy for curvature, but never extracts the curvature 2-form $F$ itself. For small triangles with area $A$, the Ambrose-Singer theorem relates holonomy to curvature:

$$C_{ijk} \approx I + F_{ij} \cdot A_{ijk} + \mathcal{O}(A^2),$$

so the field strength tensor can be recovered via $F_{ij} \approx \log(C_{ijk}) / A_{ijk}$, where $\log$ is the matrix logarithm and $A_{ijk}$ is a suitable area measure on the token graph (e.g., based on pairwise KL distances).

**2a. Yang-Mills energy.** Define $S_{\text{YM}} = \sum_{i < j} \|F_{ij}\|_F^2$. This is the natural energy functional for gauge fields in physics. Tracking $S_{\text{YM}}$ during training would reveal whether VFE minimization implicitly minimizes the gauge field energy, and if so, at what rate relative to the cross-entropy objective.

**2b. Curvature decomposition.** For $\mathfrak{gl}(K) = \mathfrak{sl}(K) \oplus \mathbb{R}$, decompose $F_{ij}$ into its traceless component (non-abelian curvature, analogous to the strong/weak force) and trace component (abelian curvature, analogous to electromagnetism). Studying which component dominates and how this changes with training could reveal the model's internal gauge structure.

**2c. Bianchi identity verification.** The discrete Bianchi identity $D_{[i} F_{jk]} = 0$ (antisymmetrized covariant derivative of curvature vanishes) should hold for any well-defined connection. Measuring violations quantifies discretization artifacts and could guide mesh refinement of the token graph.

**2d. Curvature-semantics correlations.** Compute $\|F_{ij}\|_F$ for all token pairs and test whether large curvature correlates with non-trivial semantic transport (e.g., between subject and verb, between anaphora and antecedent).

**Infrastructure.** `holonomy.py` (computes $C_{ijk}$, lines 32--86), `holonomy_metrics.py` (spectral diagnostics), `connection.py` (defines $\delta_{ij}$). Missing: matrix logarithm extraction, Yang-Mills energy computation, Bianchi identity checker, irrep decomposition of $F$.

---

## 3. Topological Invariants of Learned Bundles

No work in the existing ML literature computes topological invariants of learned gauge bundles. The Gauge-Transformer's discrete GL$^+(K)$ connection provides a natural setting for discrete Chern-Weil theory.

**3a. Chern classes.** For a GL$(K)$ connection with curvature $F$, the first Chern class is $c_1 = (1/2\pi) \sum_{\text{plaquettes}} \operatorname{tr}(F_{ij})$, and the second Chern class involves $\operatorname{tr}(F \wedge F)$. In the discrete setting on the token graph, these reduce to sums over triangles. The Chern number is an integer topological invariant that cannot change under smooth deformation of the connection. If the model develops non-trivial Chern numbers during training, it has discovered topologically non-trivial structure in the data.

**3b. Instanton number.** For SL$(K)$ connections (traceless, achievable via the existing `apply_slk_projection()` in `gauge_preconditioner.py`), the instanton number $\nu = (1/8\pi^2) \int \operatorname{tr}(F \wedge F)$ counts topological winding. In lattice gauge theory, instantons mediate tunneling between degenerate vacua. Whether trained language models develop instantonic gauge configurations is an entirely open question.

**3c. Persistent homology of the KL filtration.** Use the pairwise $\text{KL}(q_i \| \Omega_{ij} q_j)$ matrix as a distance function and compute the Vietoris-Rips persistent homology. The resulting barcode diagram reveals connected components ($H_0$), loops ($H_1$), and voids ($H_2$) in the token interaction structure at different divergence thresholds. This connects the TDA literature to gauge-theoretic attention through the KL metric.

**Infrastructure.** Holonomy matrices $C_{ijk}$ (exist), KL matrices (exist). Missing: Chern-Weil computation pipeline, instanton detector, persistent homology (could use `giotto-tda` or `ripser`).

---

## 4. RoPE as Gauge Connection

The literature search (covering 2023--2026) reveals a clear gap. Liu et al. (2025, arXiv:2504.06308) formalize RoPE in the language of $\mathfrak{so}(n)$ Lie algebras and maximal abelian subalgebras, but explicitly do not make the gauge-theoretic connection. No published work identifies RoPE as implementing a position-dependent gauge connection within a fiber bundle structure.

The Gauge-Transformer manuscript already derives $R(\theta_{j-i}) = \exp(-\phi_i^{\text{pos}}) \cdot \exp(\phi_j^{\text{pos}})$, which is the flat gauge transport in $\text{SO}(2)^{K/2} \subset \text{GL}(K)$. The `transport_ops.py` docstring (lines 102--123) sketches this identification. A standalone publication could formalize the equivalence, derive the rope-base scaling result from wavelength-sequence matching (see `analysis/rope_base_scaling_analysis.md`), and compute the curvature of the combined connection (RoPE position gauge $\oplus$ learned content gauge).

The `rope_full_gauge` flag (rotating $\Sigma$ via $R\Sigma R^\top$ in addition to $\mu \to R\mu$) implements the gauge-consistent interpretation where the full Gaussian belief is transported, versus the standard practice of rotating only $\mu$. The pragmatic equivalence of these two modes (when $\Sigma$ is diagonal and approximately SO(2)-invariant) has a clean explanation in terms of the block-diagonal structure of the rotation relative to the covariance.

**Infrastructure.** Full RoPE implementation (`transport_ops.py`), `rope_full_gauge` experimental flag, the rope-base analysis document. Missing: formal writeup, curvature computation of the combined connection.

---

## 5. Renormalization Group Flow

The VFE hierarchy $(h \to s \to p \to q)$ has a natural self-similar structure: meta-agents (clusters of tokens with aggregated beliefs) satisfy the same VFE objective as individual agents, but at a coarser scale. The old codebase (`TransformerOld/transformer/analysis/rg_metrics.py`) implemented modularity, effective rank, and cluster statistics for attention matrices, but this analysis was never ported to the current architecture.

**5a. Coarse-graining.** Cluster tokens by attention weight similarity (e.g., spectral clustering on $\beta_{ij}$). Each cluster $A$ defines a meta-agent $Q_A$ with aggregated belief $\bar{\mu}_A = \sum_{i \in A} w_i \mu_i$, $\bar{\Sigma}_A = \sum_{i \in A} w_i (\Sigma_i + (\mu_i - \bar{\mu}_A)(\mu_i - \bar{\mu}_A)^\top)$. The meta-agents interact via a coarse-grained attention matrix $\beta'_{AB} = \sum_{i \in A, j \in B} \beta_{ij}$. This defines a single RG step.

**5b. Running couplings.** Define effective coupling constants at each layer $\ell$ by measuring the ratio of self-coupling to cross-coupling KL. The $\beta$-functions $\beta_\alpha(\ell) = d\alpha_{\text{eff}}/d\ell$ characterize how the gauge theory flows with depth. Fixed points of these functions would identify universality classes of the learned representations.

**5c. Scaling dimensions.** Under coarse-graining, operators (attention matrices, transport operators, belief moments) transform with definite scaling dimensions $\Delta$: $\mathcal{O}_\ell \sim \ell^{-\Delta}$. Measuring these dimensions and checking whether they satisfy conformal bootstrap constraints would test whether the Gauge-Transformer develops conformal structure at criticality.

**Infrastructure.** `TransformerOld/transformer/analysis/rg_metrics.py` (modularity, effective rank), `TrajectoryRecorder` (layer-by-layer tracking). Missing: port to V13, coarse-graining algorithm, $\beta$-function computation, scaling dimension measurement.

---

## 6. Spectral Flow and Gauge Anomalies

**6a. Spectral flow.** Track the eigenvalue spectrum of transport operators $\Omega_{ij}$ through training and across layers. Eigenvalue crossings (where two eigenvalues exchange) as a function of a continuous parameter (layer depth or training step) constitute spectral flow, a topological invariant related to the Atiyah-Singer index theorem. The number of crossings counts the net chirality of the gauge field.

**6b. Determinant evolution.** Since $\det(\Omega_{ij}) = \exp(\operatorname{tr}(\phi_i - \phi_j))$, the log-determinant $\log\det(\Omega_{ij}) = \operatorname{tr}(\phi_i - \phi_j)$ measures volume distortion under transport. Tracking this across layers reveals whether the model systematically compresses ($\det < 1$) or expands ($\det > 1$) belief volumes, and whether this pattern correlates with layer function (early layers as feature extraction, late layers as decision-making).

**6c. Numerical gauge anomalies.** The gauge equivariance of VFE is exact in infinite precision: $F[\Lambda \cdot q] = \Lambda \cdot F[q]$ for any $\Lambda \in \text{GL}^+(K)$. Under finite-precision arithmetic (float16, float32), this symmetry breaks. Measuring the equivariance violation $\|F[\Lambda \cdot q] - \Lambda \cdot F[q]\|$ for random gauge transformations would characterize numerical gauge anomalies and could inform precision requirements for training.

**Infrastructure.** `holonomy_metrics.py` (spectral gap, Wilson traces), `gauge_utils.py` (matrix exponentials). Missing: eigenvalue tracking across training, spectral flow detector, equivariance violation measurement.

---

## 7. Information-Geometric Dual Structure and $\alpha$-Divergence Generalization

The Gaussian family $\mathcal{G}_K$ is one of the most studied objects in information geometry (Amari, 2016). It is a dually flat manifold under the $\pm 1$-connections: the exponential connection $\nabla^{(e)}$ and the mixture connection $\nabla^{(m)}$ are both flat, and the KL divergence is the canonical divergence of this dual structure. The VFE framework currently uses only the KL divergence ($\alpha = 1$ in Amari's convention), leaving the full family of $\alpha$-divergences unexplored.

### 7a. Dual Flatness Verification

The $e$-affine coordinates (natural parameters) of $\mathcal{N}(\mu, \Sigma)$ are $\theta = (\Sigma^{-1}\mu, -\frac{1}{2}\Sigma^{-1})$, and the $m$-affine coordinates (expectation parameters) are $\eta = (\mu, \mu\mu^\top + \Sigma)$. The KL divergence decomposes as the Bregman divergence of the log-partition function:

$$\text{KL}(p \| q) = \psi(\theta_q) + \varphi(\eta_p) - \theta_q \cdot \eta_p,$$

where $\psi$ and $\varphi$ are Legendre duals. The VFE E-step updates beliefs in $\theta$-coordinates (precision-weighted mean updates in `vfe_closed_form.py`, lines 174--177). Verifying that these updates respect the $e$-affine structure (linear in $\theta$) would confirm that the closed-form E-step performs exact $e$-geodesic projection.

### 7b. $\alpha$-Connection Interpretation

The gauge transport $\Omega_{ij}$ acts on the natural parameters of the key belief. Under the $\alpha$-connection framework, transport along a curve should preserve $\alpha$-affine structure. Testing whether $\Omega$-transport preserves the Bregman structure of the KL would determine whether the gauge connection is compatible with the information-geometric dual structure.

### 7c. $\alpha$-Divergence Generalization of Attention (Expanded Treatment)

#### Motivation

The current attention mechanism uses the KL divergence $\text{KL}(q_i \| \Omega_{ij}[q_j])$ as the scoring function. Replacing this with the Renyi $\alpha$-divergence $D_\alpha(q_i \| \Omega_{ij}[q_j])$ yields a one-parameter family of attention mechanisms that interpolates between different statistical comparison geometries. The $\alpha = 1$ case recovers the current model exactly, providing a backward-compatible generalization.

The $\alpha$-divergence family corresponds to the family of $\alpha$-connections in Amari's information geometry. The KL ($\alpha = 1$) uses the exponential connection $\nabla^{(1)}$; the reverse KL ($\alpha \to 0$) uses the mixture connection $\nabla^{(-1)}$; the Bhattacharyya distance ($\alpha = 1/2$) uses the geometric mean connection. Implementing $\alpha$-divergence attention is therefore equivalent to navigating the family of statistical connections on the Gaussian belief manifold.

#### Closed-Form for Diagonal Gaussians

For diagonal Gaussians $P = \mathcal{N}(\mu_P, \operatorname{diag}(\sigma_P^2))$ and $Q = \mathcal{N}(\mu_Q, \operatorname{diag}(\sigma_Q^2))$, the Renyi $\alpha$-divergence admits a per-dimension decomposition $D_\alpha(P \| Q) = \sum_{k=1}^K d_\alpha(s_k, t_k, \Delta_k)$, where $s_k = \sigma_{P,k}^2$, $t_k = \sigma_{Q,k}^2$, $\Delta_k = \mu_{Q,k} - \mu_{P,k}$, and

$$d_\alpha(s, t, \Delta) = \frac{1}{2(\alpha - 1)} \log\!\left(\frac{s^{1-\alpha} \, t^\alpha}{(1-\alpha)s + \alpha t}\right) + \frac{\alpha}{2} \cdot \frac{\Delta^2}{(1-\alpha)s + \alpha t}.$$

This formula was verified symbolically (SymPy): the limit $\alpha \to 1$ recovers the standard KL divergence per dimension,

$$\lim_{\alpha \to 1} d_\alpha(s, t, \Delta) = \frac{1}{2}\left(\frac{s}{t} - 1 + \log\frac{t}{s} + \frac{\Delta^2}{t}\right),$$

confirming backward compatibility.

The blended covariance $\sigma_{\alpha}^2 = (1-\alpha)s + \alpha t$ that appears in both terms interpolates between the query covariance ($\alpha = 0$), the arithmetic mean ($\alpha = 1/2$), and the key covariance ($\alpha = 1$, the standard KL denominator). For $\alpha > 1$, the blend $(1-\alpha)s + \alpha t = \alpha t - (\alpha - 1)s$ requires $t/s > (\alpha - 1)/\alpha$ for positivity, restricting the safe operating range. For $\alpha \leq 1$, the blend is always positive definite.

#### Position-Content Decomposition

A structurally significant property of the $\alpha$-divergence is its differential treatment of mean-difference (Mahalanobis) and covariance-ratio (log-det) terms. When the query and key have equal covariances ($s = t$), the log-det term vanishes and

$$d_\alpha(s, s, \Delta) = \frac{\alpha}{2} \cdot \frac{\Delta^2}{s},$$

which scales linearly with $\alpha$. Since the Mahalanobis term carries all position-dependent information (via RoPE-rotated means), $\alpha$ directly controls the strength of the positional signal in attention.

When the means are equal ($\Delta = 0$), the divergence reduces to

$$d_\alpha(s, t, 0) = \frac{1}{2(\alpha - 1)} \log\!\left(\frac{s^{1-\alpha} \, t^\alpha}{(1-\alpha)s + \alpha t}\right),$$

a pure covariance divergence with complex $\alpha$-dependence (via the weighted geometric-to-arithmetic mean ratio). This term is position-independent and reflects content-based belief similarity.

The consequence is that $\alpha$ provides a control over the position-to-content signal ratio in attention that is independent of the temperature $\kappa$. Changing $\kappa$ scales all divergence contributions equally; changing $\alpha$ reweights the Mahalanobis (position-sensitive) term relative to the log-det (covariance-sensitive) term. This makes $\alpha$ a complementary tuning knob to $\kappa$ and rope-base: where rope-base controls how many dimensions carry positional information, and $\kappa$ controls the sharpness of the softmax, $\alpha$ controls how strongly the existing positional signal influences attention relative to content.

#### Gradients for the E-Step

The VFE E-step requires gradients of $D_\alpha$ with respect to the query belief parameters $(\mu_i, \sigma_i^2)$. For the diagonal case, these are:

$$\frac{\partial d_\alpha}{\partial \mu_P} = \frac{\alpha \, (\mu_P - \mu_Q^t)}{(1-\alpha)\sigma_P^2 + \alpha\,\sigma_Q^{t\,2}},$$

$$\frac{\partial d_\alpha}{\partial \sigma_P^2} = -\frac{\alpha(\sigma_Q^{t\,2} - \sigma_P^2)}{2\,\sigma_P^2\,\bigl((1-\alpha)\sigma_P^2 + \alpha\,\sigma_Q^{t\,2}\bigr)} - \frac{\alpha(1-\alpha)\,\Delta^2}{2\,\bigl((1-\alpha)\sigma_P^2 + \alpha\,\sigma_Q^{t\,2}\bigr)^2}.$$

At $\alpha = 1$, the mean gradient reduces to $(\mu_P - \mu_Q^t)/\sigma_Q^{t\,2}$ and the covariance gradient to $(1/\sigma_Q^{t\,2} - 1/\sigma_P^2)/2$, matching the standard KL gradients in `vfe_gradients.py` (lines 1342--1345). The computational cost is identical to the KL case: $\mathcal{O}(K)$ per dimension pair, with one additional intermediate (the blended covariance $\sigma_\alpha^2$).

#### Closed-Form E-Step Compatibility

The existing closed-form E-step (`vfe_closed_form.py`, lines 128--243) solves for the precision-weighted mean $\mu_i^* = (\text{total information}) / (\text{total precision})$. The $\alpha$-divergence mean gradient is linear in $\mu_P$ when $\sigma$ is held fixed (because $\Delta = \mu_Q^t - \mu_P$ is linear in $\mu_P$), so the mean update retains a closed form:

$$\mu_i^* = \frac{\alpha_{\text{self}} \, \mu_p / \tilde{\sigma}_p + \lambda \sum_j \beta_{ij} \, \alpha_{\text{div}} \, \mu_{t,j} / \tilde{\sigma}_j}{\alpha_{\text{self}} / \tilde{\sigma}_p + \lambda \sum_j \beta_{ij} \, \alpha_{\text{div}} / \tilde{\sigma}_j},$$

where $\tilde{\sigma}_j = (1-\alpha_{\text{div}})\sigma_i^2 + \alpha_{\text{div}} \sigma_{t,j}^2$ is the blended covariance. Since $\tilde{\sigma}_j$ depends on $\sigma_i^2$ (the query covariance being solved for), this creates a mild coupling that can be resolved by Picard iteration, which the codebase already implements for the softmax coupling term (lines 339--413).

#### Behavioral Predictions by $\alpha$ Range

For $\alpha < 1$, the $\alpha$-divergence is *mode-covering*: it penalizes $P$ for placing mass where $Q$ has low density, but is tolerant of $P$ spreading mass beyond $Q$'s support. In attention terms, this produces softer, more diffuse attention patterns because the divergence between dissimilar beliefs is smaller (the blended covariance is larger, reducing the Mahalanobis contribution). The extreme case $\alpha \to 0$ weights only the overlap region.

For $\alpha > 1$, the divergence is *tail-sensitive*: it amplifies differences in the tails of the distributions, producing sharper, sparser attention. However, the positivity constraint on the blended covariance restricts the operating range.

The $\alpha = 1/2$ case is of particular interest: the blended covariance becomes the arithmetic mean $(\sigma_P^2 + \sigma_Q^{t\,2})/2$, and the divergence is related to the Bhattacharyya coefficient $\text{BC}(P, Q) = \int \sqrt{p\,q}\,dx$ via $D_{1/2} = -2\log\text{BC}$. The Bhattacharyya distance is symmetric in $(P, Q)$ and bounded, which could improve numerical stability in cases where the KL divergence exhibits large asymmetries between query and transported key beliefs.

#### Numerical Properties

| $\alpha$ | Equal-variance $d_\alpha(1,1,1)$ | Different-variance $d_\alpha(1,2,1)$ | Behavior |
|:--------:|:--:|:--:|:--|
| 0.25 | 0.125 | 0.133 | Very soft attention |
| 0.50 | 0.250 | 0.226 | Symmetric, bounded |
| 0.75 | 0.375 | 0.294 | Slightly softer than KL |
| 1.00 | 0.500 | 0.347 | Standard KL |
| 1.50 | 0.750 | 0.423 | Sharper attention |
| 2.00 | 1.000 | 0.477 | Very sharp, restricted domain |

The per-dimension divergence scales linearly with $\alpha$ in the equal-variance case (pure Mahalanobis) and sublinearly when variances differ (log-det term partially compensates).

#### Integration Points in Codebase

Implementation requires modifications to six files, all backward-compatible when $\alpha_{\text{div}} = 1.0$:

| File | Lines | Modification |
|:-----|:-----:|:-------------|
| `block_config.py` | 119--122 | Add `alpha_divergence: float = 1.0` field |
| `kl_computation.py` | 221--268 | Parameterize `_kl_kernel_diagonal` by $\alpha$ |
| `kl_computation.py` | 87--218 | Parameterize `_kl_kernel_dense` by $\alpha$ |
| `vfe_gradients.py` | 1318--1345 | Replace $\partial\text{KL}/\partial\mu$, $\partial\text{KL}/\partial\sigma$ with $\alpha$-forms |
| `attention.py` | 361--364 | Replace KL with $D_\alpha$ in logit computation |
| `vfe_closed_form.py` | 174--243 | Use blended covariance in precision-weighted mean |

The diagonal kernel modification is representative. The current implementation (lines 260--265):

```python
trace_term = (sigma_q / sigma_t).sum(dim=-1)
delta = mu_t - mu_q
mahal_term = ((delta ** 2) / sigma_t).sum(dim=-1)
logdet_term = (torch.log(sigma_t) - torch.log(sigma_q)).sum(dim=-1)
kl = 0.5 * (trace_term + mahal_term - K + logdet_term)
```

would become (for general $\alpha$):

```python
sigma_blend = (1 - alpha) * sigma_q + alpha * sigma_t  # blended covariance
delta = mu_t - mu_q
mahal_term = (alpha * delta**2 / sigma_blend).sum(dim=-1)
logdet_term = ((1-alpha)*torch.log(sigma_q) + alpha*torch.log(sigma_t)
               - torch.log(sigma_blend)).sum(dim=-1) / (alpha - 1)
d_alpha = 0.5 * (mahal_term + logdet_term)
```

with a Taylor expansion fallback near $\alpha = 1$ to avoid the $1/(\alpha - 1)$ singularity.

#### Proposed Experimental Protocol

An ablation sweep over $\alpha \in \{0.25, 0.5, 0.75, 1.0, 1.5\}$ at fixed $K = 20$, $B = 10$, $N = 128$ would characterize the perplexity-vs-$\alpha$ landscape. The prediction from the position-content decomposition is that $\alpha < 1$ will produce more diffuse attention and potentially worse perplexity on next-token prediction (which requires sharp positional discrimination), while $\alpha > 1$ will sharpen attention at the cost of numerical stability. The optimal $\alpha$ likely lies in $[0.75, 1.25]$, close to the current KL but potentially shifted by the specific balance of positional and content information in the data.

A second experiment would fix $\alpha$ and sweep rope-base, testing whether $\alpha$ and rope-base interact (since both control positional signal strength through different mechanisms). If they are approximately independent, the optimal $(B, \alpha)$ pair can be found by separate 1D sweeps; if they interact strongly, a 2D grid search would be necessary.

---

## 8. Gauge-Fixing and the Moduli Space

The VFE is invariant under simultaneous gauge transformations of all beliefs and gauge frames: $\mu_i \to g\mu_i$, $\Sigma_i \to g\Sigma_i g^\top$, $\phi_i \to \phi_i + \log g$ for any $g \in \text{GL}^+(K)$. This gauge orbit is a $K^2$-dimensional submanifold of the parameter space on which the loss function is exactly constant.

**8a. Characterize the gauge orbit.** Given a trained model, compute the dimensionality of the gauge orbit and the number of physical (gauge-invariant) degrees of freedom. For $N$ tokens with $K$-dimensional beliefs, the full parameter count is $N(K + K + K^2)$ (means, variances, phi), and the gauge group has $K^2$ generators, so the moduli space has dimension $N(2K + K^2) - K^2$.

**8b. Gauge-fixed visualization.** Choose a canonical gauge (e.g., axial gauge $\phi_1 = 0$, or Coulomb gauge $\sum_i \phi_i = 0$) and visualize the gauge-fixed beliefs. Different gauge choices emphasize different structural features, just as Coulomb and Lorenz gauges reveal different aspects of electrodynamics.

**8c. Gribov copies.** Does the gauge-fixing have a unique solution, or do multiple gauge-inequivalent configurations satisfy the same gauge condition? The existence of Gribov copies would mean the moduli space has non-trivial topology, with implications for optimization landscape geometry (multiple basins separated by gauge-orbit boundaries).

**Infrastructure.** `gauge_preconditioner.py` (SL$(K)$ projection, Cartan decomposition). Missing: gauge-fixing algorithms, moduli space dimension counting, Gribov copy detection.

---

## 9. Wilson Loop Spectrum as Linguistic Diagnostic

The holonomy computation (`holonomy.py`, lines 32--86) computes $C_{ijk}$ around triangular loops in the token graph. Generalizing to larger, linguistically motivated loops yields a diagnostic that connects gauge geometry to syntax and semantics.

**9a. Wilson loops along syntactic paths.** Given a dependency parse, compute holonomy along the path from subject to verb to object and back. The Wilson loop trace $W = |\operatorname{tr}(C_{\text{loop}})|/K$ is a gauge-invariant scalar. If $W \approx 1$ (near-identity holonomy), the gauge connection is approximately flat along the syntactic path, meaning beliefs can be transported without distortion. If $W \ll 1$, the connection is curved, and transport along the syntactic path produces significant belief rotation.

**9b. Area law versus perimeter law.** In lattice gauge theory, the scaling of $\langle W(C) \rangle$ with loop area versus perimeter distinguishes confining from deconfining phases. An area law $\langle W \rangle \sim \exp(-\sigma \cdot \text{Area})$ means the gauge field creates a confining potential (information is trapped locally). A perimeter law $\langle W \rangle \sim \exp(-\mu \cdot \text{Perimeter})$ means free propagation. Measuring this scaling for the Gauge-Transformer's learned gauge fields would reveal the effective "confinement" properties of learned linguistic representations.

**9c. Wilson loop operator product expansion.** Study how the Wilson trace correlates with linguistic properties of the enclosed tokens: part-of-speech tags, dependency relations, coreference links. Tokens enclosed by high-curvature loops may correspond to syntactic or semantic boundaries.

**Infrastructure.** `holonomy.py` (triangle loops exist), `holonomy_metrics.py` (Wilson traces). Missing: arbitrary-loop holonomy computation (path-ordered product along arbitrary token subsequences), syntactic path extraction (needs a parser), area/perimeter scaling analysis.

---

## 10. Cross-Architecture Gauge Diagnostics

Apply the holonomy and curvature analysis to standard pretrained transformers (BERT, GPT-2, LLaMA) by treating their $W_Q W_K^\top$ matrices as implicit transport operators. The script `scripts/gauge_frame_spectral_analysis.py` already performs some spectral analysis of pretrained models.

**10a. Extract $\Omega_{\text{eff}}$ from pretrained models.** For each attention head, the effective transport is $\Omega_{\text{eff}} = W_Q^\top W_K / \sqrt{d_k}$. Compute holonomy of this implicit connection and compare to the Gauge-Transformer's explicit gauge connection.

**10b. Flatness comparison.** Test the prediction that standard transformers develop approximately flat connections (low holonomy, because the Q/K/V projection structure constrains the gauge freedom), while the Gauge-Transformer can learn genuinely curved connections where the data warrants it.

**10c. Gauge symmetry breaking patterns.** Standard transformers start with $\text{GL}(d_k)$ parameter-space gauge symmetry (van Nierop, 2024; "Maximal Gauge Symmetry", 2025 under review) and break it during training. The Gauge-Transformer preserves $\text{GL}(K)$ gauge equivariance by construction. Comparing the residual symmetries at convergence could reveal whether the two architectures converge to similar effective gauge structures despite their different starting points.

**Infrastructure.** `gauge_frame_spectral_analysis.py` (pretrained model analysis). Missing: holonomy computation on pretrained models, flatness comparison framework.

---

## Priority Ranking

| Rank | Direction | Effort | Novelty | Paper Potential | Synergies |
|:----:|:----------|:------:|:-------:|:---------------:|:---------:|
| 1 | 7c. $\alpha$-divergence attention | Low | High | Standalone + ablation paper | Complements rope-base |
| 2 | 1. Intra-fiber VFE trajectories | Medium | Very high | Standalone | Foundation for 2, 6 |
| 3 | 4. RoPE as gauge connection | Low | High | Clear literature gap | Extends rope-base analysis |
| 4 | 2. Curvature tensor + Yang-Mills | Medium | Very high | Standalone | Requires 1 for context |
| 5 | 9. Wilson loop linguistics | Medium | Very high | High if results clean | Builds on 2 |
| 6 | 5. RG flow | Medium | High | Extends existing work | Independent |
| 7 | 3. Topological invariants | High | Extreme | Landmark if successful | Requires 2 |
| 8 | 6. Spectral flow | Medium | High | Novel diagnostic | Independent |
| 9 | 8. Moduli space | High | High | Deep theory | Requires 2, 3 |
| 10 | 10. Cross-architecture | Medium | Medium | Comparative study | Independent |

Directions 7c, 1, and 2 form a natural first phase: implementing $\alpha$-divergence attention is low-effort and immediately testable, studying fiber trajectories provides foundational understanding for all subsequent geometric analysis, and extracting the curvature tensor from existing holonomy data opens the door to the remaining physics-inspired directions.
