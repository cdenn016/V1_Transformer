# Bootstrap Self-Distillation for the Gauge-VFE E-Step

**Status:** design document, not yet implemented
**Date:** 2026-04-06
**Related:** `session_2026_04_06_active_inference_and_efe.md`, `_bootstrap_distill_verify.py` (SymPy proofs)

---

## Abstract

The active-inference extensions recently added to the gauge-VFE transformer introduce two terms to the E-step free energy: a pragmatic term that minimizes the entropy of the readout distribution, and an epistemic term that maximizes the BALD-style mutual information between belief samples and the predictive distribution. The pragmatic term is self-reinforcing on its own and requires the epistemic term as a counterweight; both operate purely on the position-local distribution and do not use the sequence structure. This document proposes a third term, **bootstrap self-distillation**, in which each position's belief update is regularized to match the gauge-transported predictive distribution of its attention-weighted neighbours, with the target detached by a stop-gradient. The term is structurally a cross-entropy against a stop-gradient target, formally identical to the objectives used in BYOL and DINO, but with the view-augmentation role played by the parallel transport operator $\Omega_{ij}$. The term occupies a genuinely empty slot in the E-step coupling landscape: it couples positions in predictive-distribution space rather than belief space, it is data-dependent at initialization (so unlike the pragmatic term it fires immediately without needing prior training), and it has a principled interpretation as generalized belief propagation projected into observation space. A stop-gradient on the attention weights is required to prevent an "attend-to-twins" collapse mode distinct from the entropy-collapse mode of the pragmatic term. Gradient and fixed-point claims are verified symbolically via SymPy and collected in Appendix A.

---

## 1. Motivation

The session that introduced active inference into the E-step established two facts that together motivate this proposal. First, the pragmatic self-observation term $\lambda_{\text{prag}} \cdot H[p_{\text{pred}}(v \mid \mu_i)]$ is self-reinforcing: on its own it drives the belief toward whatever point-distribution is currently favored by the readout, a feedback loop that the epistemic counterterm was introduced specifically to break. Second, at fresh initialization with a 50K-vocabulary softmax, the entropy gradient is effectively zero because the readout sits near its maximum-entropy extremum, where both the gradient and the Hessian of $H$ vanish. Diagnostics measured a max-logit difference of approximately $10^{-5}$ between EFE on and EFE off at random initialization, growing to order unity only after 50–200 training steps had broken the near-uniformity of the readout.

A satisfying resolution would be a term with three properties. It should be data-dependent even at initialization, so that the E-step feels the effect of active inference from the first forward pass. It should not require a counterweight to avoid collapse, so that a single term can be introduced without introducing a second. It should couple positions in a way that the existing KL attention coupling does not, so that it contributes genuinely new information rather than a reformulation of a term already present. All three requirements are satisfied by the bootstrap self-distillation variant analyzed below, provided a subtle second stop-gradient is applied to the attention weights themselves and not only to the target distribution.

## 2. The E-step coupling landscape

The variational free energy currently minimized in the E-step is, schematically,

$$F[q] \;=\; \underbrace{\alpha \sum_i \mathrm{KL}(q_i \| p_i)}_{\text{self-coupling}} \;+\; \underbrace{\lambda_{\text{belief}} \sum_{ij} \beta_{ij}\, \mathrm{KL}(q_i \| \Omega_{ij} q_j)}_{\text{neighbour coupling in Q-space}} \;+\; \underbrace{\lambda_{\text{prag}}\, H[p_{\text{pred}}(v|\mu_i)] - \lambda_{\text{epi}}\, \mathrm{MI}(v; \mu \mid q_i)}_{\text{active-inference extensions}}$$

with the attention weights themselves given by the Boltzmann policy

$$\beta_{ij} \;=\; \mathrm{softmax}_j\!\left(-\frac{\mathrm{KL}(q_i \| \Omega_{ij} q_j)}{\kappa}\right).$$

The existing couplings fall into two classes. The self-coupling and the neighbour KL coupling both operate in the *belief fiber* — they measure discrepancies between $(\mu, \Sigma)$ triples via Gaussian KL divergences, and the transport operator $\Omega_{ij}$ moves beliefs between fibers so that the comparison is well-defined. The pragmatic and epistemic terms operate entirely on the *local* readout distribution at a single position, using the PriorBank to map the local belief into a probability distribution over the vocabulary and then penalizing either its entropy or its mutual information with belief samples.

Neither class couples positions *in prediction space*. The belief-fiber couplings act on a $K$-dimensional representation that is related to the vocabulary readout only through a many-to-one map: two beliefs that are close in $Q$-space can produce predictive distributions that differ substantially near a decision boundary, and conversely two beliefs that are far in $Q$-space can produce predictive distributions that are nearly identical if their difference lies in the null space of the PriorBank projection. There is therefore an identifiable gap in the coupling landscape, corresponding to constraints of the form "position $i$'s *prediction* should agree with its neighbours' (gauge-transported) *predictions*" that the current terms neither impose nor imply.

## 3. Formal proposal

Let $p^{(t)}_{\text{pred}}(v \mid \mu) = \mathrm{softmax}(-\mathrm{KL}(q \| \pi_v)/\tau)$ denote the PriorBank readout at iteration $t$, and let $\Omega_{ij}$ denote the parallel transport from position $j$ to position $i$ in the current gauge frame. The bootstrap self-distillation loss at position $i$ is

$$\boxed{\;L_{\text{distill},i} \;=\; \sum_j \mathrm{sg}[\beta_{ij}] \cdot \mathrm{CE}\!\left(\mathrm{sg}\!\left[p^{(t)}_{\text{pred}}(v \mid \Omega_{ij}\mu_j)\right], \; p^{(t)}_{\text{pred}}(v \mid \mu_i)\right)\;}$$

where $\mathrm{CE}(p, q) = -\sum_v p_v \log q_v$ is the standard cross-entropy, and $\mathrm{sg}[\cdot]$ is the stop-gradient operator that prevents gradients from flowing through its argument. Two stop-gradients are present in the definition, and both are essential. The first severs the gradient through the target distribution, which is the structural analogue of the target-network detachment in BYOL and DINO and is what prevents the trivial collapse in which both sides of the cross-entropy shrink toward the same degenerate point. The second severs the gradient through the attention weight $\beta_{ij}$, which addresses a distinct collapse mode — the *attend-to-twins* collapse — that is derived explicitly in Section 6 below. Without the second stop-gradient the loss admits a trivial descent direction in which position $i$ concentrates all of its attention on whichever neighbours already happen to agree with it, driving the loss to zero without conveying any useful information.

An equivalent aggregated form, which is substantially cheaper to compute at the cost of replacing the per-pair distillation with a consensus-to-consensus distillation, uses the attention-aggregated transported belief $\tilde\mu_i = \sum_j \beta_{ij}\, \Omega_{ij}\mu_j$ already computed by the attention sublayer as the target site:

$$L^{\text{agg}}_{\text{distill},i} \;=\; \mathrm{CE}\!\left(\mathrm{sg}\!\left[p^{(t)}_{\text{pred}}(v \mid \tilde\mu_i)\right],\; p^{(t)}_{\text{pred}}(v \mid \mu_i)\right).$$

This form costs a single readout per position per iteration, matching the cost of the existing pragmatic term, and its semantics ("match the consensus of your transported neighbours") is slightly different from the per-pair form ("match each of your transported neighbours, weighted") but is arguably more faithful to the belief propagation interpretation developed in Section 8.

## 4. Gradient analysis

The gradient of the distillation loss with respect to position $i$'s logits follows from a standard calculation which SymPy verifies component by component. Let $\mathbf{z}_i$ denote the pre-softmax logits of the readout at position $i$, so that $p^{(t)}_{\text{pred}}(v \mid \mu_i) = \mathrm{softmax}(\mathbf{z}_i)_v$. For a single neighbour $j$ with target $\mathbf{t}_j \equiv \mathrm{sg}[p^{(t)}_{\text{pred}}(v \mid \Omega_{ij}\mu_j)]$, the cross-entropy gradient is

$$\frac{\partial}{\partial z_{i,k}}\!\left[-\sum_v t_{j,v}\, \log p^{(t)}_{\text{pred},v}(v \mid \mu_i)\right] \;=\; p^{(t)}_{\text{pred},k}(v \mid \mu_i) \;-\; t_{j,k}.$$

The derivation uses only the chain rule and the softmax identity $\partial \log \mathrm{softmax}(\mathbf{z})_{v'}/\partial z_{v} = \delta_{vv'} - \mathrm{softmax}(\mathbf{z})_v$, combined with the constraint $\sum_v t_{j,v} = 1$. A direct SymPy computation (Appendix A, Claim 1) confirms this identity for all three components of a three-class softmax, with a specific numerical check at the agreement fixed point $p = t = (1/4, 1/4, 1/2)$ returning exactly zero in every component.

Summing over neighbours with attention weights held fixed (stop-gradient on $\beta$), the full gradient is

$$\frac{\partial L_{\text{distill},i}}{\partial z_{i,k}} \;=\; \sum_j \mathrm{sg}[\beta_{ij}] \left( p^{(t)}_{\text{pred},k}(v \mid \mu_i) \;-\; p^{(t)}_{\text{pred},k}(v \mid \Omega_{ij}\mu_j) \right)$$

$$= p^{(t)}_{\text{pred},k}(v \mid \mu_i) \;-\; \sum_j \mathrm{sg}[\beta_{ij}] \cdot p^{(t)}_{\text{pred},k}(v \mid \Omega_{ij}\mu_j).$$

The right-hand side is structurally identical to the gradient of a cross-entropy loss against a *weighted average of stop-gradient targets*, which is the standard form for a multi-teacher distillation objective. The fixed point is reached exactly when the position-$i$ predictive distribution equals the attention-weighted consensus of the (transported) neighbour predictive distributions, and the residual at any point is a convex combination of the pointwise mismatches.

The contrast with the pragmatic term's gradient is mathematically sharp. For the entropy $H[p] = -\sum_v p_v \log p_v$, SymPy confirms that the gradient with respect to the logits is

$$\frac{\partial H[p]}{\partial z_k} \;=\; -p_k \cdot (\log p_k + H[p]),$$

which has the unusual property of vanishing at *both* the uniform distribution ($p_k = 1/V$, so $\log p_k + H = -\log V + \log V = 0$) and in the limit of any point distribution (where either $p_k \to 0$ killing the prefactor, or $p_k \to 1$ with $\log p_k \to 0$ and $H \to 0$). The entropy has two classes of critical points separated by a manifold of non-degenerate gradients, and minimizing it is a non-convex descent toward the nearest point-distribution vertex of the simplex. The distillation loss has one class of critical points, determined by the data-dependent targets $\{p^{(t)}_{\text{pred}}(v \mid \Omega_{ij}\mu_j)\}_j$, and convergence to that single consensus point is not tied to sharpness but to agreement.

This distinction is precisely what makes the bootstrap distillation term fire immediately at random initialization. The pragmatic gradient is small at initialization because the readout sits at a flat extremum of the entropy, but the distillation gradient is small at initialization only if the target agrees with the current prediction, which at random initialization it does not — the transported neighbour predictions are themselves random and differ from the local random prediction by a quantity of order unity in total-variation distance. The SymPy verification script reports a gradient of exactly zero at agreement and a numerical example with `CE = (0, 1, 1)` producing nontrivial gradient components $(-2/9, 1/9, 1/9)$ at the uniform state.

## 5. Fixed-point analysis

The stationary condition for the distillation loss at position $i$, with the attention weights fixed by stop-gradient, is

$$p^{(t)}_{\text{pred}}(v \mid \mu_i) \;=\; \sum_j \beta_{ij}\cdot p^{(t)}_{\text{pred}}(v \mid \Omega_{ij}\mu_j) \qquad \forall v.$$

This is a consensus condition in predictive-distribution space: the prediction at each site must equal the attention-weighted average of the transported neighbour predictions. If the attention weights themselves satisfy a consistency condition with the belief state (which they do, through the Boltzmann softmax of the KL attention), then this fixed point is jointly defined by the data and the gauge structure of the sequence.

The consensus condition does not degenerate to a single point-distribution for two independent reasons. First, the M-step cross-entropy loss against actual token targets forces position-specific distinctions: positions carrying different target tokens cannot satisfy a common consensus without incurring CE loss, and the balance between the distillation regularizer and the M-step CE determines whether the consensus point matches the targets or the targets win out over the consensus. Second, in the aggregated form, the target itself is computed from a weighted average of transported beliefs whose weights are position-dependent, so the "consensus" at different positions is generically different. The aggregated form's fixed point satisfies

$$p^{(t)}_{\text{pred}}(v \mid \mu_i) \;=\; p^{(t)}_{\text{pred}}\!\left(v \;\big|\; \sum_j \beta_{ij}\Omega_{ij}\mu_j\right) \qquad \forall v,$$

which is pointwise in the vocabulary and solved at the point where the belief $\mu_i$ maps to the same readout distribution as the attention-aggregated transported belief $\tilde\mu_i$. Since the PriorBank readout is a many-to-one map, this does not require $\mu_i = \tilde\mu_i$ in the fiber — only that the two beliefs produce the same distribution over the vocabulary. The aggregated fixed-point set is therefore a higher-dimensional submanifold of the belief space than the exact-agreement condition, which is consistent with the aggregated form being a weaker and cheaper regularizer than the per-pair form.

## 6. The attend-to-twins collapse

The stop-gradient on the attention weights in the loss definition is not cosmetic; without it the loss admits a trivial descent direction that defeats the entire purpose of the term. Consider the loss written with $\beta$ in the autograd graph:

$$L'_i \;=\; \sum_j \beta_{ij}(\mathbf{q}, \Omega) \cdot \mathrm{CE}_{ij},$$

where $\mathrm{CE}_{ij} \equiv \mathrm{CE}(\mathrm{sg}[p^{(t)}_{\text{pred}}(\Omega_{ij}\mu_j)],\, p^{(t)}_{\text{pred}}(\mu_i))$ is the per-pair distillation cross-entropy with a stop-gradient target, and $\beta_{ij}$ depends differentiably on the beliefs and transports through the Boltzmann softmax of the KL attention. Let $\ell_{ij}$ denote the pre-softmax attention score $-\mathrm{KL}(q_i \| \Omega_{ij} q_j)/\kappa$, so that $\beta_{ij} = \mathrm{softmax}_j(\boldsymbol{\ell}_i)_j$.

Differentiation gives

$$\frac{\partial L'_i}{\partial \ell_{ik}} \;=\; \beta_{ik}\cdot(\mathrm{CE}_{ik} - \langle \mathrm{CE}_{i} \rangle_\beta),$$

where $\langle \mathrm{CE}_{i} \rangle_\beta = \sum_j \beta_{ij}\,\mathrm{CE}_{ij}$ is the current attention-weighted mean CE. SymPy verifies this identity symbolically (Appendix A, Claim 3). The sign of the gradient is positive for neighbours whose per-pair CE is *above average* (disagreeing neighbours) and negative for neighbours whose per-pair CE is *below average* (agreeing neighbours). Gradient descent therefore *decreases* the scores for disagreeing neighbours and *increases* the scores for agreeing ones, and the attention mass flows to the neighbours that already agree with the local prediction.

A concrete numerical example makes the failure mode stark. With three neighbours at initially uniform attention $\beta = (1/3, 1/3, 1/3)$ and per-pair CE values $\mathrm{CE} = (0, 1, 1)$ — meaning neighbour 1 already agrees perfectly and neighbours 2 and 3 disagree by equal amounts — SymPy computes the gradient on the attention scores as

$$\left.\frac{\partial L'_i}{\partial \boldsymbol{\ell}_i}\right|_{\text{uniform}} = \left(-\tfrac{2}{9},\; \tfrac{1}{9},\; \tfrac{1}{9}\right).$$

A single gradient descent step increases $\ell_{i1}$ and decreases $\ell_{i2}, \ell_{i3}$, which through the softmax increases $\beta_{i1}$ toward 1 and drives $\beta_{i2}, \beta_{i3}$ toward zero. After convergence, position $i$ attends only to neighbour 1, the target is trivially whatever position $i$ was already predicting, and the loss is zero — not because the model has learned anything, but because it has found a trivial agreement by ignoring dissenting voices. This is the attend-to-twins collapse, and it is distinct from the entropy-collapse failure mode of the pragmatic term in that the failure involves the *coupling structure* rather than the *local prediction*.

The remedy is to treat $\beta_{ij}$ as a stop-gradient constant inside the distillation loss. With that modification, the gradient on the attention scores contributed by the distillation term is identically zero, the only gradient flow is through the local prediction $p_{\text{pred}}(v \mid \mu_i)$, and the loss can only be decreased by changing the prediction to better match the targets — which is the intended behaviour. The attention weights themselves are still learned by the attention sublayer through the standard KL-attention mechanism; the stop-gradient merely prevents the distillation loss from reshaping the attention distribution in a self-serving way.

## 7. Gauge equivariance

For the distillation term to be consistent with the gauge framework of the rest of the model, it must transform covariantly under position-dependent gauge transformations. Let $\{g_i\}_{i=1}^N$ be a collection of invertible linear maps acting on the belief fibers, transforming the belief states as $\mu_i \mapsto g_i \mu_i$ and $\Sigma_i \mapsto g_i \Sigma_i g_i^\top$. The transport operator transforms as $\Omega_{ij} \mapsto g_i \Omega_{ij} g_j^{-1}$, so that the transported neighbour belief $\Omega_{ij}\mu_j \mapsto g_i (\Omega_{ij} \mu_j)$ correctly lands in position $i$'s transformed frame. Both the local belief $\mu_i$ and the transported neighbour belief $\Omega_{ij}\mu_j$ are therefore acted on by the same $g_i$ after the gauge transformation, and the question of gauge invariance reduces to whether the readout $p^{(t)}_{\text{pred}}(v \mid \mu)$ is invariant under a *simultaneous* gauge transformation of its argument $\mu$ and the PriorBank prior $\pi_v$.

The readout is computed as a softmax over KL distances $\mathrm{KL}(q \| \pi_v)$, and the gauge invariance reduces further to whether the Gaussian KL is invariant under simultaneous gauge transformation of both its arguments. The SymPy verification script (Appendix A, Claim 4) computes $\mathrm{KL}(\mathcal{N}(g\mu_1, g\Sigma_1 g^\top) \| \mathcal{N}(g\mu_2, g\Sigma_2 g^\top))$ and simplifies the difference with $\mathrm{KL}(\mathcal{N}(\mu_1, \Sigma_1) \| \mathcal{N}(\mu_2, \Sigma_2))$ to zero symbolically, with a numerical cross-check using a specific non-orthogonal $2 \times 2$ gauge transformation $g = \begin{psmallmatrix}2 & 1 \\ 1 & 3\end{psmallmatrix}$ returning exactly the same numerical KL value under the transformed and untransformed computations. The invariance holds for general invertible $g$, not only orthogonal ones, because the Mahalanobis form $\delta^\top \Sigma^{-1} \delta$ is invariant under any invertible change of variables that acts on both $\delta$ and $\Sigma$.

Because the priors in PriorBank are shared across positions rather than being position-specific objects carrying their own gauge indices, the invariance proof requires that the priors be interpreted as "templates in a canonical frame" that participate in the gauge transformation at whichever position they are currently being compared against. This is the same convention used by the existing self-coupling $\mathrm{KL}(q_i \| p_i)$ in the free energy functional and does not introduce any new assumptions beyond what the rest of the codebase already relies on.

## 8. Connection to generalized belief propagation

Generalized belief propagation (GBP) in a factor graph computes marginal beliefs at each node by passing messages from neighbouring factors, with each message being the marginal prediction that the sender node would make about the receiver's state. In a pairwise factor graph over variables $\{x_i\}$ with edge factors $\psi_{ij}(x_i, x_j)$, the BP update rule takes the form

$$m_{j \to i}(x_i) \propto \int \psi_{ij}(x_i, x_j) \cdot q_j(x_j) \cdot \prod_{k \in N(j) \setminus i} m_{k \to j}(x_j)\, dx_j,$$

and the belief at node $i$ is updated toward the product of the prior and the incoming messages. The fixed point of BP is exactly the condition that each node's belief is consistent with the messages received from its neighbours, which for pairwise factor graphs reduces to "the belief at $i$ agrees with the marginalization of the joint belief inferred from any single neighbour."

The bootstrap distillation loss is the projection of this condition into predictive-distribution space, with the edge factors replaced by gauge transport operators and the belief integrals replaced by PriorBank readouts. Specifically, the message from neighbour $j$ to position $i$ in the distillation interpretation is the transported predictive distribution $p^{(t)}_{\text{pred}}(v \mid \Omega_{ij}\mu_j)$, and the fixed point condition at position $i$ is that the local predictive distribution equals the attention-weighted mixture of incoming messages. The attention weights $\beta_{ij}$ play the role of trust coefficients assigned to each neighbour, determined by the quality of the transported match between the local and neighbouring beliefs. The stop-gradient on the messages corresponds to the standard BP assumption that each node computes its own belief locally, treating incoming messages as external inputs that are updated on a slower timescale — the same role that the target network plays in BYOL and the EMA teacher plays in DINO.

What makes the interpretation non-trivial in the gauge-transformer setting, and what distinguishes it from a generic message-passing reformulation, is that the transport operator $\Omega_{ij}$ is a nontrivial geometric object rather than a learned projection. In a standard neural message-passing layer the message from $j$ to $i$ would be a learned MLP applied to $\mu_j$, and the resulting framework would be equivalent to a graph neural network with whatever inductive bias the MLP architecture imposes. In the gauge transformer the transport operator is constrained to belong to a specific Lie group ($\mathrm{GL}(K)$, or one of its subgroups) and satisfies a cocycle condition (or its relaxation under the non-flat transport extension), and the resulting messages are therefore gauge-covariant in a precise sense: the message from $j$ to $i$ equals the inverse message from $i$ to $j$ after left-right transposition under the gauge group action, and the consistency of messages across triangles $ijk$ is measured by the holonomy $\Omega_{ij}\Omega_{jk}\Omega_{ki}$. These are the same invariants that govern the gauge transport in the belief coupling term, and the distillation term therefore inherits the geometric structure of the rest of the model rather than introducing a parallel mechanism.

## 9. Connection to BYOL, DINO, and non-contrastive self-distillation

The non-contrastive self-distillation objectives in BYOL and DINO can be written abstractly as

$$L_{\text{distill}}(\theta) \;=\; \mathrm{CE}\!\left(\mathrm{sg}\!\left[f_{\theta^{\text{teacher}}}(x^{\text{view}_2})\right],\; f_\theta(x^{\text{view}_1})\right),$$

where $x^{\text{view}_1}$ and $x^{\text{view}_2}$ are two augmented views of the same input, the teacher network $f_{\theta^{\text{teacher}}}$ is an exponential moving average of the student weights, and the stop-gradient prevents the trivial collapse in which both networks output a constant. BYOL adds a predictor head to the student for further asymmetry; DINO replaces the EMA with a centering-plus-sharpening operation.

The bootstrap distillation loss has the same structural form with two substitutions. The "view augmentation" is replaced by the gauge transport $\Omega_{ij}$, which maps the belief at position $j$ into position $i$'s frame and therefore serves as a principled "view" of the sequence at position $i$ from the perspective of position $j$. The teacher-student split is replaced by the stop-gradient on the target distribution, which severs the dependence of the target on the current parameters without requiring a separate network. The absence of an EMA is possible because the gauge transport is a deterministic geometric operation rather than a learned transformation — the transport operator is computed from the current beliefs and does not need its own momentum buffer. The centering-and-sharpening operations of DINO, which are needed to prevent the collapse toward uniform or toward a single class, are replaced here by the M-step CE loss at each position, which forces position-specific distinctions and prevents both types of collapse.

A substantive difference from the vision self-distillation setting is that in DINO and BYOL the two views are sampled stochastically from an augmentation distribution, whereas in the bootstrap distillation the "views" are deterministic and structured by the sequence position and the gauge connection. This means there is no stochasticity in the target for a fixed forward pass, which has the advantage that the gradient signal is low-variance and the disadvantage that there is no implicit regularization from view sampling. The stochasticity can be recovered, if desired, by sampling a subset of neighbours per iteration rather than using the full sum — a variant that trades off gradient variance against computational cost and is discussed in Section 10.

## 10. Implementation strategies

The per-pair form of the loss requires $N^2$ PriorBank readouts per iteration per layer, which at the standard configuration of $N = 128$, $K = 60$, and $V = 50257$ amounts to approximately 49 GFLOPs per iteration per layer. This is roughly $130\times$ the cost of the position-level pragmatic term and would dominate the forward pass in a typical training configuration. Three implementation strategies are available.

The **aggregated form** uses the attention-aggregated transported belief $\tilde\mu_i = \sum_j \beta_{ij}\Omega_{ij}\mu_j$ that the attention sublayer already computes, and evaluates a single PriorBank readout at $\tilde\mu_i$ as the target:

$$L^{\text{agg}}_{\text{distill},i} \;=\; \mathrm{CE}\!\left(\mathrm{sg}\!\left[p^{(t)}_{\text{pred}}(v \mid \tilde\mu_i)\right],\; p^{(t)}_{\text{pred}}(v \mid \mu_i)\right).$$

This is $O(N V K)$ per iteration, identical to the cost of the pragmatic term. The semantic shift from the per-pair form is that the target becomes the prediction at a single aggregated belief rather than a mixture of predictions at individual transported beliefs, and because the PriorBank readout is nonlinear in $\mu$ these are not equivalent:

$$p_{\text{pred}}\!\left(v \mid \sum_j \beta_{ij}\Omega_{ij}\mu_j\right) \;\neq\; \sum_j \beta_{ij}\,p_{\text{pred}}(v \mid \Omega_{ij}\mu_j)$$

in general. The aggregated form is a stronger condition because it asks the two beliefs $\mu_i$ and $\tilde\mu_i$ to map to identical readouts, whereas the per-pair form asks the readout at $\mu_i$ to match a mixture that can be satisfied by a larger set of $\mu_i$ values.

The **subsampled per-pair form** reduces the cost by randomly selecting a fixed number $M \ll N$ of neighbours per position per iteration and computing the per-pair loss only over that subset:

$$L^{\text{sub}}_{\text{distill},i} \;=\; \frac{1}{M}\sum_{j \in S_i(t)} \mathrm{sg}[\beta_{ij}] \cdot \mathrm{CE}\!\left(\mathrm{sg}[p^{(t)}_{\text{pred}}(\Omega_{ij}\mu_j)],\, p^{(t)}_{\text{pred}}(\mu_i)\right),$$

with $S_i(t) \subset \{1,\ldots,N\}$ resampled per iteration. The stochastic estimator is unbiased for the per-pair form if the subsample is drawn uniformly, and approximately unbiased if the subsample is drawn proportional to the (stop-gradient) attention weights. The cost is $O(N M V K)$, which for $M = 8$ and $N = 128$ is about $16\times$ smaller than the full per-pair form and comparable to the per-position pragmatic cost within a small constant factor. The subsampled form retains the per-pair semantics (the target is a specific mixture of transported predictions, not the prediction at a specific mixture) but introduces gradient variance.

The **iteration-cached form** exploits the fact that the transported belief $\Omega_{ij}\mu_j$ is already computed by the attention sublayer for the belief coupling term. Caching the readout $p^{(t)}_{\text{pred}}(\Omega_{ij}\mu_j)$ at the point of transport and reusing it inside the distillation loss avoids recomputing the PriorBank decode, at the cost of additional memory proportional to $N^2 V$ per layer per iteration. For $V = 50257$ and $N = 128$ this is about 3.3 GB of memory per layer, which is impractical at the current configuration but may become reasonable for smaller vocabularies or lower-precision caches.

The recommended default is the aggregated form. It matches the cost of the existing pragmatic term, its semantic interpretation as "distill toward the consensus" is the cleanest of the three and corresponds most faithfully to the belief propagation analogy, and its fixed-point condition defines a higher-dimensional manifold that is easier for the model to satisfy without incurring M-step CE penalties. The per-pair and subsampled forms can be added as configuration options for empirical comparison.

## 11. Expected empirical behaviour

Unlike the pragmatic self-observation term, the bootstrap distillation term is expected to produce a nonzero gradient at random initialization. At fresh initialization, the PriorBank readout at each position is approximately uniform (as a consequence of the softmax being near its maximum-entropy extremum on 50K classes with random embeddings), and the transported neighbour readout is also approximately uniform but with independent random deviations from uniformity at each pair $(i, j)$. The cross-entropy between two near-uniform but not-identical distributions is slightly larger than the entropy of either distribution alone, and the gradient of the CE with respect to the logits is proportional to the difference between the target and the current prediction, which at random initialization is an $O(1/\sqrt{V})$ perturbation rather than the $o(1/V)$ signal that the entropy gradient decays to at the same point. Concretely, the expected gradient magnitude at initialization is governed by the variance of the random embeddings rather than by the distance from the entropy extremum, and this variance is nonzero by construction.

The prediction is therefore that after implementing the distillation term, the max-logit difference between EFE on and EFE off should be visible at step zero of a fresh training run, rising from the order $10^{-5}$ reported in the pragmatic-only diagnostics to something in the range $10^{-2}$ to $10^{-1}$ for the same model and the same random seed. The signal should not depend strongly on vocabulary size beyond a weak logarithmic scaling through the CE normalization, in contrast to the pragmatic term's strong adverse scaling with $V$ that was identified in the session diagnostics. A minimal verification protocol for an eventual implementation would run the existing `scripts/verify_active_inference.py` diagnostic with the distillation flag on and the pragmatic and epistemic weights set to zero, compare the max-logit difference to a baseline with all active-inference terms off, and confirm that the difference is visible at fresh initialization rather than only after pre-training.

Beyond initialization, the behavior over training is harder to predict without empirical measurement. The distillation term will compete with the M-step CE loss in determining the fixed point of the E-step: regions of the sequence where the attention-weighted consensus agrees with the true target will see a synergistic signal, and regions where the consensus disagrees with the target will see a conflict that the weighted sum of the two losses resolves according to the relative magnitudes. The interesting empirical question is whether the distillation term helps or hurts the early-training convergence rate, and whether it produces qualitatively different belief dynamics in the E-step trajectory (as measured by the per-layer $\mu$ norm drift and the KL consistency across adjacent iterations).

## 12. Open questions

Several questions are not resolved by the analysis above and deserve attention before or during implementation.

The first is whether the distillation term should be computed on the readout entropy scale or on the KL distance scale. The current pragmatic and epistemic terms operate on the softmax readout distribution, which lives in the probability simplex. The distillation term as defined above also operates on the softmax readout. An alternative would be to define the target as the unnormalized logits $-\mathrm{KL}(q_j \| \pi_v)/\tau$ rather than the softmax of those logits, and to use a mean-squared-error loss instead of a cross-entropy. This would make the term a direct regularizer on the KL distance structure rather than on the categorical distribution, and might have different dynamics. The analysis in this document does not cover the MSE-on-logits variant and it is flagged as a follow-up.

The second is the appropriate balance between the per-pair and aggregated forms. The aggregated form is cheaper and semantically cleaner, but the per-pair form has a direct interpretation as distilling from a mixture of teachers which may produce more robust representations in the presence of noisy neighbours. An ablation comparing the two forms on a controlled synthetic task would clarify whether the cheaper aggregated form gives up any empirical advantage.

The third is the interaction with the existing pragmatic and epistemic terms. If the distillation term is added alongside the existing active-inference terms, the three terms are expected to be partially redundant (the pragmatic term pushes toward confidence, the epistemic term pushes toward informativeness, the distillation term pushes toward agreement, and at the consensus fixed point all three are simultaneously satisfied by a confident, informative, consensual prediction). If the three terms are combined with independent weights, the weight on the pragmatic term may need to be reduced or zeroed out — or the distillation term may make the pragmatic and epistemic terms redundant entirely and allow their removal. This is an empirical question that depends on the scale of the respective gradients in practice.

The fourth is whether the stop-gradient on the attention weights should be implemented with `tensor.detach()` or with a one-iteration lag on the attention weights (analogous to the iteration-to-iteration bootstrap in Section 4's brief mention of variants 1 and 2). The two are mathematically equivalent in expectation but differ in gradient variance, and the choice may matter in practice.

The fifth is the question of normalization. The CE loss at vocabulary size $V$ has a natural scale of $\log V \approx 10.8$ nats for a 50K vocabulary, which is much larger than the KL attention values of order unity. Without normalization, the distillation term will dominate the free energy at initialization and the rest of the loss terms will be effectively ignored. A sensible default is to normalize the loss by $\log V$ so that a uniform distribution produces a unit-order contribution, but the exact normalization constant is a design choice.

---

## Appendix A: SymPy verification script

The four central mathematical claims of this document — the gradient of the cross-entropy against a stop-gradient target, the gradient of the entropy with its zero at both uniform and point distributions, the attend-to-twins gradient on the attention scores, and the gauge equivariance of the Gaussian KL — are verified symbolically by the script `docs/_bootstrap_distill_verify.py`. The script can be re-run at any time by invoking `python docs/_bootstrap_distill_verify.py` from the repository root and produces the following verification output for each claim.

**Claim 1.** For a three-class softmax with logits $(z_1, z_2, z_3)$ and target $(t_1, t_2, t_3)$ with $\sum t_k = 1$, the cross-entropy $-\sum_k t_k \log \mathrm{softmax}(\mathbf{z})_k$ has gradient $\mathrm{softmax}(\mathbf{z})_k - t_k$ component by component. SymPy confirms the identity for all three components. At the agreement fixed point with $(t_1, t_2, t_3) = (1/4, 1/4, 1/2)$ and $z_k = \log t_k$, every gradient component evaluates to exactly zero.

**Claim 2.** For a three-class softmax, the gradient of the entropy $H[\mathrm{softmax}(\mathbf{z})]$ with respect to the logits is $-\mathrm{softmax}(\mathbf{z})_k \cdot (\log \mathrm{softmax}(\mathbf{z})_k + H)$, matching the closed-form derivative component by component. At the uniform distribution $\mathbf{z} = (0, 0, 0)$, all three gradient components evaluate to exactly zero. In the limit of a point distribution $\mathbf{z} \to (\infty, 0, 0)$, all three components evaluate to zero as well, confirming the two-critical-point structure of the entropy that distinguishes it from the monotonic bootstrap CE.

**Claim 3.** With three neighbours and cross-entropy values $(a, b, c)$, attention weights computed by softmax on pre-softmax scores $(\ell_1, \ell_2, \ell_3)$, and the non-stop-gradient loss $L' = \sum_j \beta_j \cdot \mathrm{CE}_j$, SymPy confirms that $\partial L'/\partial \ell_k = \beta_k(\mathrm{CE}_k - \langle \mathrm{CE} \rangle_\beta)$ component by component. A specific numerical example with uniform initial attention and $\mathrm{CE} = (0, 1, 1)$ yields gradient components $(-2/9, 1/9, 1/9)$, confirming that gradient descent concentrates attention on the agreeing neighbour and confirming the attend-to-twins collapse diagnosis.

**Claim 4.** For two-dimensional Gaussians with scalar-times-identity covariance and a general two-by-two invertible gauge matrix $g$, the Gaussian KL divergence satisfies $\mathrm{KL}(\mathcal{N}(g\mu_1, g\Sigma_1 g^\top) \| \mathcal{N}(g\mu_2, g\Sigma_2 g^\top)) = \mathrm{KL}(\mathcal{N}(\mu_1, \Sigma_1) \| \mathcal{N}(\mu_2, \Sigma_2))$. SymPy computes the difference of the two expressions and simplifies it to zero. A numerical sanity check with $\mu_1 = (1, 2)$, $\mu_2 = (3, 5)$, $\Sigma_1 = 2I$, $\Sigma_2 = 3I$, and $g = \begin{psmallmatrix}2 & 1 \\ 1 & 3\end{psmallmatrix}$ returns $\mathrm{KL}_{\text{orig}} = \mathrm{KL}_{\text{transformed}} = 2.2387984414414976$ to float64 precision.

The SymPy verifications are independent of the PyTorch implementation and can be used to regression-test any future autograd-based computation of the distillation gradient.
