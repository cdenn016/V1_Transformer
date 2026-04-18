# How the Gauge VFE Transformer Learns

A complete walkthrough of the amortized inference training pipeline, from raw text to learned representations, written for readers who may not have a background in gauge theory or variational inference.

## Table of Contents

1. [What Is Each Token?](#1-what-is-each-token)
2. [The Training Data](#2-the-training-data)
3. [One Forward Pass: The E-Step](#3-one-forward-pass-the-e-step)
4. [How Tokens Learn to Align: The M-Step](#4-how-tokens-learn-to-align-the-m-step)
5. [A Training Experiment End to End](#5-a-training-experiment-end-to-end)
6. [How This Differs from a Standard Transformer](#6-how-this-differs-from-a-standard-transformer)
7. [Glossary](#7-glossary)

---

## 1. What Is Each Token?

In a standard transformer, every token is a **point** in space --- a single vector of numbers. In this system, every token is a **probability distribution** (a Gaussian "cloud" of uncertainty) plus a **gauge frame** (a local coordinate system). Each token carries three things:

| Component | Symbol | Shape (current config) | What it represents |
|-----------|--------|------------------------|--------------------|
| **Mean** | $\mu$ | 20 numbers | "Where the token believes it is" in a 20-dimensional latent space |
| **Variance** | $\sigma^2$ | 20 numbers | "How uncertain the token is" along each dimension |
| **Gauge frame** | $\phi$ | 100 numbers ($K^2 = 20^2$) | "What coordinate system the token uses" --- a rotation/stretch encoded as a Lie algebra element |

These are **learned per token**. There is a lookup table with 50,257 rows (one per BPE subword in the GPT-2 vocabulary), and each row stores that token's $(\mu, \sigma^2, \phi)$. At initialization:

- $\mu$ values are random normal (std = 1.0).
- All $\sigma^2$ start at 1.0 (maximum ignorance).
- $\phi$ values are small random (std $\approx$ 0.1, near the identity transform).

With `gauge_fixed_priors = True` (the current default), there is actually a **single base distribution** $\pi_0 = \mathcal{N}(\mu_0, \Sigma_0)$, and each token's distribution is obtained by applying that token's gauge transform to the base:

$$\pi_{\text{cat}} = \exp(\phi_{\text{cat}} \cdot G) \;\triangleright\; \pi_0$$

This enforces that all token distributions are related by gauge transformations by construction. The entire vocabulary lives in a single geometric orbit of the gauge group.

---

## 2. The Training Data

The training data is **WikiText-103** --- 103 million tokens of curated Wikipedia articles, tokenized with GPT-2's BPE tokenizer (subword units, not raw characters). A training sample is a contiguous window of 64 tokens (`max_seq_len = 64`) from a Wikipedia article:

```
Example window (decoded for readability):

input:  "The domestic cat is a small , typically furry , carnivorous mammal .
         They are often called house cats when"

target: "domestic cat is a small , typically furry , carnivorous mammal .
         They are often called house cats when kept"
```

The target is the input shifted by one position: at every position, the model must predict the next token.

Each batch contains 128 such windows (`batch_size = 128`). Windows are drawn from random positions in the concatenated corpus, so each batch sees a diverse cross-section of Wikipedia. In total, one batch covers $128 \times 64 = 8{,}192$ tokens.

There is nothing random or synthetic about the training data. These are real English sentences from real articles, fed through a standard BPE tokenizer that maps text to integer IDs in the range $[0, 50256]$.

---

## 3. One Forward Pass: The E-Step

When a batch arrives, here is what happens step by step. We trace a concrete example with the fragment `"the cat sat on the mat"`.

### Step 1 --- Embedding Lookup

Each token ID hits the embedding table and retrieves its belief triple:

```
"the"  -->  (mu_the,  sigma2_the,  phi_the)      20-dim mean, 20-dim variance, 100-dim frame
"cat"  -->  (mu_cat,  sigma2_cat,  phi_cat)
"sat"  -->  (mu_sat,  sigma2_sat,  phi_sat)
"on"   -->  (mu_on,   sigma2_on,   phi_on)
"the"  -->  (mu_the,  sigma2_the,  phi_the)      same "the" as position 0
"mat"  -->  (mu_mat,  sigma2_mat,  phi_mat)
```

At this point, "the" at position 0 and "the" at position 4 have **identical** $(\mu, \sigma^2, \phi)$ because they are the same word.

### Step 2 --- Position Encoding via Rotary Embeddings

RoPE (Rotary Position Embedding) applies position-dependent rotations to $\mu$. This makes "the" at position 0 differ from "the" at position 4 in its mean vector, while the gauge frame $\phi$ remains unchanged. Position is encoded geometrically, not by adding a learned vector.

### Step 3 --- Building Transport Operators

This is where gauge theory enters. For every pair of tokens $(i, j)$, the system computes a **transport operator**:

$$\Omega_{ij} = \exp(\phi_i \cdot G) \;\cdot\; \exp(-\phi_j \cdot G)$$

where $G$ are Lie algebra generators (matrices that encode the symmetry group). With the irrep spec `[('fund', 2, 10)]`, there are **2 attention heads**, each operating on a 10-dimensional slice of the 20-dim belief vector.

The transport operator $\Omega_{ij}$ "translates" token $j$'s beliefs into token $i$'s coordinate system --- analogous to converting Celsius to Fahrenheit before comparing temperatures. If two tokens already use the same coordinate system ($\phi_i \approx \phi_j$), then $\Omega_{ij} \approx I$ and no translation is needed.

### Step 4 --- KL-Divergence Attention

Instead of the standard dot-product attention ($q \cdot k$), this system measures how **statistically surprising** one token's belief is to another, after coordinate alignment:

$$\beta_{ij} = \text{softmax}_j\!\left(\frac{-\text{KL}(q_i \;\|\; \Omega_{ij}[q_j])}{\kappa}\right)$$

In words: "How surprised is token $i$ when it sees token $j$'s belief, after translating $j$ into $i$'s coordinate system?"

- Tokens that are **similar** (low KL after transport) receive high attention weights.
- Tokens that are **dissimilar** receive low attention weights.

After training, "cat" and "dog" have similar $\mu$ vectors (both are common animals with similar syntactic roles). Their KL divergence is low, so they attend to each other strongly. "Cat" and "the" have higher KL --- different grammatical roles, different distributions --- so their mutual attention is weaker.

The temperature $\kappa$ (initialized at 3.16, learnable per head) controls how sharp or diffuse the attention distribution is.

### Step 5 --- VFE E-Step (Belief Evolution)

This is the core innovation. With `ffn_n_iterations = 1` (one VFE gradient step per forward pass), the system updates each token's beliefs by minimizing a **variational free energy** that balances three forces:

**Force 1 --- Prior pull** ($\alpha = 1$): Don't drift too far from your embedding prior.

$$\frac{\partial F}{\partial \mu_{\text{cat}}}\bigg|_{\text{prior}} = \alpha \;\cdot\; \frac{\mu_{\text{cat}} - \mu_{\text{cat}}^{\text{embed}}}{\sigma^2_{\text{prior}}}$$

*"Stay close to what 'cat' usually means."*

**Force 2 --- Alignment pull** ($\lambda_{\text{belief}} = 1$): Move toward what your attended neighbors believe.

$$\frac{\partial F}{\partial \mu_{\text{cat}}}\bigg|_{\text{align}} = \sum_j \beta_{\text{cat},j} \;\cdot\; \nabla \text{KL}(q_{\text{cat}} \;\|\; \Omega_{\text{cat},j}[q_j])$$

*"Adjust your belief toward a weighted average of your neighbors' beliefs (translated into your frame)."*

**Force 3 --- Softmax coupling** ($\lambda_{\text{softmax}} = 1$): The nonlinear coupling through the softmax derivative. This arises from the fact that attention weights $\beta_{ij}$ themselves depend on beliefs, creating cross-talk between all pairs. This is the principled replacement for GELU/ReLU activation functions.

These three gradients are combined with **Fisher-Rao preconditioning** (natural gradient descent that accounts for the geometry of probability distributions) and applied as one update step:

$$\mu_{\text{cat}}^{\text{new}} = \mu_{\text{cat}} - \eta \;\cdot\; \mathcal{F}^{-1} \;\cdot\; (\nabla_{\text{prior}} + \nabla_{\text{align}} + \nabla_{\text{softmax}})$$

The covariance and gauge frame also update:

- $\sigma^2$ evolves via SPD (symmetric positive-definite) manifold retraction.
- $\phi$ evolves via Killing-form-preconditioned Lie algebra descent.

After this step, "cat" in the sentence "the cat sat on the mat" has a **contextually modified** belief. It has been pulled toward "the", "sat", "on", and "mat" in proportion to how much attention it paid to each, while being anchored to its prior meaning. The result is a context-dependent representation, built entirely from information geometry rather than neural network layers.

### Step 6 --- Output Projection

The evolved mean $\mu_{\text{cat}}^{\text{new}}$ (20-dimensional) is projected to vocabulary logits via the model's only neural component, a single linear layer:

$$\text{logits}_{\text{cat}} = W_{\text{out}} \cdot \mu_{\text{cat}}^{\text{new}} \qquad (20) \to (50{,}257)$$

The model predicts: given "the cat", what comes next? The target is "sat". The cross-entropy loss measures how much probability mass the model placed on the correct next token.

---

## 4. How Tokens Learn to Align: The M-Step

The E-step (Section 3) happens **inside** the forward pass and does not directly update the embedding table. The actual learning of permanent parameters happens via **backpropagation through the entire E-step**. This is what "amortized inference" means: instead of running the E-step to convergence and then differentiating implicitly, we differentiate directly through the single VFE gradient step.

### 4.1 What Gets Updated

The loss function has two terms:

1. **Cross-entropy** between predicted next tokens and actual next tokens (the language modeling signal).
2. **Gauge prior** $\tfrac{m_\phi}{2}\|\phi\|^2$ with $m_\phi = 0.01$ (a small penalty that keeps gauge frames from growing unboundedly large).

Gradients flow backward through the output projection, through the E-step belief evolution, through the KL-divergence attention, through the transport operators, and all the way back to the embedding table entries $(\mu_{\text{embed}}, \sigma^2_{\text{embed}}, \phi_{\text{embed}})$.

Six different learning rates update six parameter groups:

| Parameter Group | Learning Rate | What It Controls |
|-----------------|---------------|------------------|
| $\mu$ embeddings | 0.05 | Where each token's belief center lives |
| $\sigma^2$ embeddings | 0.005 | How uncertain each token is (conservative updates) |
| $\phi$ embeddings | 0.0075 | Each token's gauge frame orientation |
| Attention params | 0.005 | Output mixing weights |
| VFE hyperparams | 0.05 | E-step adaptive parameters ($\alpha_i$, learned LR) |
| Output projection | 0.05 | The linear map from beliefs to vocabulary logits |

The **Riemannian Adam** optimizer applies geometry-aware updates: $\phi$ gets Killing-form preconditioning (respecting the Lie algebra structure), $\mu$ gets Fisher-metric scaling (respecting the Gaussian manifold geometry).

### 4.2 How 'Cat' and 'Dog' Become Aligned

Over thousands of training steps, the following process occurs:

**Distributional similarity emerges from context.** Both "cat" and "dog" appear frequently after "the" and before verbs like "sat", "ran", "slept". Through backpropagation:

- Their $\mu$ vectors converge to nearby regions of the 20-dimensional space, because having similar means produces similar output distributions over next tokens, which is what the cross-entropy loss rewards.

- Their gauge frames $\phi$ evolve so that the transport operator $\Omega_{\text{cat,dog}}$ approaches the identity matrix. When two tokens' frames are nearly aligned, the KL divergence between them (after transport) is small, and they attend to each other strongly. The training signal reinforces this: contexts where "cat" appears are similar to contexts where "dog" appears, so aligning their frames improves prediction.

- Their covariances $\sigma^2$ shrink on dimensions where the model is confident about their shared meaning (e.g., "is an animal", "is a noun") and remain large on dimensions where there is genuine ambiguity (e.g., size, domestication status).

**Dissimilar tokens diverge.** Tokens with different syntactic roles (e.g., "cat" vs. "quickly") develop different $\mu$ vectors and different $\phi$ orientations. Their transported KL is large, producing low mutual attention, which the training loss reinforces because attending to irrelevant tokens produces poor next-token predictions.

**Gauge structure captures abstract relations.** The gauge frame $\phi$ encodes something beyond simple similarity. Two tokens can have very different $\mu$ vectors but be related by a structured transformation $\Omega$. For example, singular and plural forms of a noun might differ in $\mu$ but be connected by a consistent gauge rotation. The gauge group provides a richer vocabulary of relationships than Euclidean distance alone.

---

## 5. A Training Experiment End to End

Starting from `python transformer/train_publication.py` with the default EM config:

### Configuration Summary

```
Model:    K=20, 1 layer, 2 heads (10-dim each), GL(20) gauge group
Data:     WikiText-103, 64-token windows, batch size 128
E-step:   1 VFE iteration, alpha=1, lambda_belief=1, lambda_softmax=1
M-step:   Riemannian Adam, linear LR decay over 15,000 steps
Params:   ~2M total (no Q, K, V, or FFN weight matrices)
```

### Timeline

**Step 0** --- All embeddings are random. $\mu \sim \mathcal{N}(0, 1)$, $\sigma^2 = 1$, $\phi \sim \mathcal{N}(0, 0.1)$. The model assigns roughly uniform probability to all 50,257 tokens. Perplexity $\approx 50{,}257$ (random guessing).

**Step 100** --- Warmup complete (100 steps). Learning rates reach their full values. The output projection $W_{\text{out}}$ begins to separate high-frequency tokens ("the", "of", "and", ",", ".") from the rest. Perplexity drops to $\sim 2{,}000\text{--}5{,}000$.

**Step 1,000** --- First evaluation on the validation set. The model has processed roughly 8 million tokens ($128 \times 64 \times 1{,}000$). Perplexity $\sim 300\text{--}500$. Common function words are predicted reliably. Gauge frames are starting to differentiate: nouns form one cluster in $\phi$-space, verbs another. Covariances $\sigma^2$ have begun shrinking on informative dimensions.

**Step 5,000** --- Perplexity $\sim 150\text{--}250$. Clear attention patterns emerge:
- Articles attend strongly to the nouns they modify (high $\beta$ between "the" and noun positions).
- Verbs attend to their subjects (syntactic alignment via gauge transport).
- The KL matrix $\text{KL}(q_i \| \Omega_{ij}[q_j])$ shows visible block structure --- groups of tokens that are mutually aligned in gauge space.

**Step 15,000** --- Training ends. Perplexity $\sim 100\text{--}200$ (depending on exact hyperparameters and K). All 50,257 token embeddings have organized into gauge-aligned clusters. The $\phi$ embeddings encode semantic and syntactic similarity via transport distances: tokens that play similar roles in language are connected by near-identity transport operators, while tokens that play different roles are separated by large gauge rotations.

### What the Checkpoint Contains

The saved checkpoint stores:
- The embedding table: 50,257 rows of $(\mu, \sigma^2, \phi)$
- The output projection $W_{\text{out}}$: a $(20 \times 50{,}257)$ matrix
- Attention parameters (output mixing, per-head temperatures)
- VFE hyperparameters (learned $\alpha$, learned E-step learning rates)
- Optimizer state (Adam moments for all parameter groups)
- Training metadata (step count, best validation loss, config)

---

## 6. How This Differs from a Standard Transformer

| Aspect | Standard Transformer | Gauge VFE Transformer |
|--------|---------------------|-----------------------|
| **Token representation** | A point vector $x \in \mathbb{R}^d$ | A Gaussian distribution $\mathcal{N}(\mu, \Sigma)$ plus a gauge frame $\phi$ |
| **Attention mechanism** | Dot product of learned $Q, K$ projections: $\text{softmax}(QK^T/\sqrt{d})$ | KL divergence after gauge transport: $\text{softmax}(-\text{KL}/\kappa)$. No learned $Q, K$ matrices |
| **Feed-forward network** | Two linear layers with GELU activation | One step of VFE natural gradient descent. No neural network, no activation function |
| **Source of nonlinearity** | GELU, softmax | Softmax coupling term $\partial\beta/\partial\mu$ (derivative of attention weights w.r.t. beliefs) |
| **Position encoding** | Sinusoidal or learned vectors added to embeddings | RoPE rotations on $\mu$ (gauge-compatible) |
| **Learning** | Backpropagation only | E-step (belief inference inside forward pass) + M-step (backpropagation through E-step) |
| **Parameters** | $\sim 6$M for comparable $d$ (Q, K, V, FFN weights) | $\sim 2$M (embeddings + output projection only) |
| **Only neural component** | Everything | The final linear output projection $W_{\text{out}}: \mathbb{R}^K \to \mathbb{R}^V$ |

The fundamental difference is architectural philosophy. A standard transformer learns **what to compute** (via weight matrices in attention and FFN layers). This system specifies **what to optimize** (a variational free energy functional) and lets the mathematics of KL divergence, gauge transport, and natural gradients determine the computation. The representations emerge from the geometry of the problem rather than from learned linear projections.

---

## 7. Glossary

**Agent.** Each token position in the sequence. In the gauge-theoretic view, each token is an "agent" that holds beliefs, communicates with neighbors, and updates its beliefs to minimize surprise. The sequence length `max_seq_len` is the number of agents.

**Amortized inference.** Instead of running the E-step to full convergence and then differentiating implicitly, gradients flow directly through the finite number of E-step iterations via standard backpropagation. Faster but introduces a small bias (the E-step may not have fully converged).

**BPE (Byte-Pair Encoding).** The tokenization scheme used by GPT-2. Splits text into subword units (e.g., "unhappiness" $\to$ ["un", "happiness"]). The vocabulary has 50,257 entries.

**Covariance / Variance ($\Sigma$, $\sigma^2$).** The uncertainty in a token's belief. With `diagonal_covariance = True`, this is a vector of per-dimension variances rather than a full matrix. Dimensions with small variance are ones the model is confident about; dimensions with large variance represent genuine ambiguity.

**E-step.** The "inference" phase. Given fixed model parameters (embeddings, output projection), find the beliefs $(\mu, \Sigma)$ that minimize the variational free energy. Happens inside the forward pass via natural gradient descent.

**Fisher-Rao preconditioning (natural gradient).** A technique from information geometry. Instead of following the raw gradient $\nabla F$, follow $\mathcal{F}^{-1} \nabla F$ where $\mathcal{F}$ is the Fisher information matrix. This accounts for the curvature of the probability distribution space, producing updates that are invariant to reparameterization.

**Gauge frame ($\phi$).** A Lie algebra element that determines each token's local coordinate system. The matrix exponential $\exp(\phi \cdot G)$ maps $\phi$ to a group element (an invertible matrix). Two tokens with similar $\phi$ values "see the world" in similar coordinates.

**Gauge transport ($\Omega_{ij}$).** The operator $\Omega_{ij} = \exp(\phi_i \cdot G) \cdot \exp(-\phi_j \cdot G)$ that converts token $j$'s beliefs into token $i$'s coordinate system. This is the mechanism that allows the model to compare beliefs that live in different local frames.

**GL(K).** The general linear group of $K \times K$ invertible matrices. The gauge group for this model. Transport operators $\Omega_{ij}$ live in $\text{GL}(K)$, meaning they can express any invertible linear transformation, not just rotations.

**Irrep (irreducible representation).** A building block of the gauge group's action. The config `[('fund', 2, 10)]` means 2 copies of the fundamental (vector) representation, each 10-dimensional, for a total $K = 20$. Each copy becomes one attention head.

**Killing form.** The natural inner product on a Lie algebra, derived from the structure constants. Used to precondition $\phi$ gradients so that updates respect the algebraic structure of the gauge group.

**KL divergence.** $\text{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} dx$. Measures how different two probability distributions are. Always non-negative; zero only when $p = q$. For Gaussians, has a closed-form expression involving means, variances, and dimensionality.

**M-step.** The "learning" phase. Given the beliefs found by the E-step, update model parameters (embeddings, output projection) to reduce the training loss. Happens via backpropagation through the entire E-step computation graph.

**Mean ($\mu$).** The center of a token's Gaussian belief distribution. This is the primary information carrier --- the output projection acts on $\mu$ to produce next-token predictions.

**Perplexity (PPL).** $\text{PPL} = \exp(\text{cross-entropy})$. Measures how "surprised" the model is by the test data. A perplexity of 100 means the model is, on average, as uncertain as if it were choosing uniformly among 100 equally likely tokens. Lower is better. Random baseline for a 50,257-token vocabulary is 50,257.

**SPD manifold.** The space of symmetric positive-definite matrices. Covariance matrices live on this manifold. Updates to $\Sigma$ must preserve positive-definiteness, which is achieved via Riemannian retraction rather than naive additive updates.

**Variational free energy (VFE).** The objective function that the E-step minimizes. Combines prior consistency (stay close to embeddings), belief alignment (agree with neighbors after transport), and the softmax coupling (nonlinear interaction through attention weights). The M-step then reduces the cross-entropy loss by adjusting parameters so that the E-step's equilibrium beliefs produce better next-token predictions.
