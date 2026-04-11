# RoPE Base Frequency Scaling in Low-Dimensional KL-Divergence Attention

## Abstract

Standard transformer architectures employ Rotary Position Embeddings (RoPE) with a base frequency parameter $B = 10{,}000$, a convention calibrated for high-dimensional key spaces ($d_{\text{head}} \geq 64$) and long sequence lengths ($N \geq 2048$). The Gauge-Transformer operates in a qualitatively different regime: belief dimensionality $K = 20$, sequence lengths $N = 64$--$128$, and attention computed via Kullback--Leibler divergence rather than dot-product similarity. Empirical observation shows that perplexity improves monotonically as the RoPE base decreases from $10{,}000$ to $10$, with $B = 10$ yielding the best train, validation, and test performance under GL(10) gauge structure. This document derives the mathematical basis for this scaling behavior. We show that the number of positionally active rotation pairs scales as $\lfloor (K/2) \cdot \log(N/2\pi) / \log B \rfloor + 1$, and that the optimal base satisfying a full-utilization criterion is $B^{*} = (N/2\pi)^{K/(K-2)}$, which evaluates to $B^{*} \approx 13.2$ for $N = 64$ and $B^{*} \approx 28.5$ for $N = 128$ at $K = 20$. We further decompose the KL-based attention logit into position-independent covariance terms and position-dependent Mahalanobis terms, demonstrating that the latter's magnitude is controlled by the count of active rotation pairs. The standard base of $10{,}000$ leaves 9 of 10 rotation pairs positionally inert at these sequence lengths, reducing the total adjacent-position discriminability by a factor of $2.3\times$ relative to $B = 10$.

## 1. Introduction

Rotary Position Embeddings encode sequence position through dimension-pair-wise SO(2) rotations applied to query and key vectors before attention scoring (Su et al., 2024). The rotation frequency for each dimension pair $k$ is determined by a geometric progression governed by a single hyperparameter, the base frequency $B$. In the original RoFormer formulation and subsequent adoption by LLaMA, GPT-NeoX, and related architectures, $B = 10{,}000$ was selected to span wavelengths from $2\pi$ (approximately 6 tokens) up to roughly $2\pi \cdot 10{,}000 \approx 63{,}000$ tokens, accommodating sequence lengths of several thousand while providing fine-grained short-range discrimination through the highest-frequency pairs.

This choice is well-matched to the standard operating regime. A typical transformer head uses $d_{\text{head}} = 64$ or $128$, yielding 32 or 64 rotation pairs. Even if half these pairs have wavelengths exceeding the training sequence length and therefore contribute negligible positional signal, the remaining 16--32 active pairs provide sufficient positional resolution. The information-theoretic requirement is modest: distinguishing $N$ positions requires $\lceil \log_2 N \rceil$ bits, which is 11 bits for $N = 2048$ and can be distributed across tens of active pairs with substantial redundancy.

The Gauge-Transformer operates outside this regime in three respects. First, the belief dimensionality is $K = 20$, yielding only $K/2 = 10$ rotation pairs --- an order of magnitude fewer than standard architectures. Second, training and evaluation sequences are short, with $N = 64$ to $128$ tokens. Third, attention is computed not through dot-product similarity but through the KL divergence between Gaussian belief distributions $q_i = \mathcal{N}(\mu_i, \Sigma_i)$ after gauge transport $\Omega_{ij}$, with the attention weight given by $\beta_{ij} = \text{softmax}(-\text{KL}(q_i \| \Omega_{ij}[q_j]) / \kappa\sqrt{K})$. RoPE enters this computation by rotating the query belief means $\mu_i \mapsto R(\theta_i)\mu_i$ prior to KL evaluation, where $R(\theta_i) \in \text{SO}(2)^{K/2}$ is a block-diagonal rotation matrix whose angles depend on position $i$ and the frequency spectrum.

Under these conditions, the standard base $B = 10{,}000$ produces a severe mismatch: the vast majority of rotation pairs have wavelengths exceeding the sequence length by orders of magnitude and therefore inject no usable positional information into the KL-based attention computation. The following sections make this argument precise.

## 2. Mathematical Framework

### 2.1 RoPE Frequency Spectrum

The implementation in `transport_ops.py` (lines 54--64) defines the inverse frequency for rotation pair $k \in \{0, 1, \ldots, K/2 - 1\}$ as

$$\theta_k = B^{-2k/K} = B^{-k/(K/2)},$$

yielding a geometric progression from $\theta_0 = 1$ (fastest rotation) to $\theta_{K/2-1} = B^{-(K-2)/K}$ (slowest). The rotation angle applied at sequence position $n$ for pair $k$ is $n \cdot \theta_k$ radians, so the wavelength --- the number of positions required for a full $2\pi$ rotation --- is

$$\lambda_k = \frac{2\pi}{\theta_k} = 2\pi \cdot B^{2k/K}.$$

This expression makes the scaling transparent. The shortest wavelength is always $\lambda_0 = 2\pi \approx 6.3$ positions, independent of $B$. The longest wavelength is

$$\lambda_{\max} = \lambda_{K/2 - 1} = 2\pi \cdot B^{(K-2)/K}.$$

For $K = 20$, this evaluates to $\lambda_{\max} = 2\pi \cdot B^{9/10}$, a near-linear function of $B$. The following table records $\lambda_{\max}$ across the range of base values tested empirically.

| $B$ | $\lambda_{\max}$ | Ratio to $N = 128$ |
|----:|------------------:|--------------------:|
| 10 | 49.9 | 0.39$\times$ |
| 50 | 212.5 | 1.66$\times$ |
| 100 | 396.4 | 3.10$\times$ |
| 1,000 | 3,149.1 | 24.6$\times$ |
| 10,000 | 25,013.8 | 195.4$\times$ |

At $B = 10{,}000$, the slowest rotation pair completes less than $0.5\%$ of a full cycle over $N = 128$ positions. At $B = 10$, the slowest pair completes approximately 2.6 full cycles, ensuring that every pair contributes meaningfully to positional encoding.

### 2.2 Active Pair Count

A rotation pair $k$ is defined as *positionally active* with respect to sequence length $N$ when its wavelength satisfies $\lambda_k \leq N$, meaning that the pair completes at least one full rotation over the sequence and can therefore discriminate positions across the full range. Solving $2\pi \cdot B^{k/(K/2)} \leq N$ for $k$ yields the condition

$$k \leq \frac{K}{2} \cdot \frac{\log(N / 2\pi)}{\log B},$$

so the number of active pairs is

$$N_{\text{active}} = \min\!\left(\left\lfloor \frac{K}{2} \cdot \frac{\log(N / 2\pi)}{\log B} \right\rfloor + 1,\;\; \frac{K}{2}\right).$$

The table below evaluates this expression for $K = 20$ ($K/2 = 10$ total pairs) at two sequence lengths.

| $B$ | $N_{\text{active}}$ ($N\!=\!64$) | $N_{\text{active}}$ ($N\!=\!128$) |
|----:|:--------------------------------:|:---------------------------------:|
| 10 | **10** / 10 | **10** / 10 |
| 50 | 6 / 10 | 8 / 10 |
| 100 | 6 / 10 | 7 / 10 |
| 1,000 | 4 / 10 | 5 / 10 |
| 10,000 | 3 / 10 | 4 / 10 |

At $B = 10$, all rotation pairs are active for both sequence lengths. At the standard $B = 10{,}000$, only 3 or 4 pairs carry positional information --- the remaining 6 or 7 pairs have wavelengths exceeding the sequence length by factors of $10\times$ to $4{,}000\times$ and contribute negligibly to position discrimination.

### 2.3 Optimal Base Derivation

A natural criterion for the base parameter is *full utilization*: requiring that the slowest rotation pair complete exactly one full cycle over the sequence, i.e., $\lambda_{\max} = N$. Setting $2\pi \cdot B^{(K-2)/K} = N$ and solving for $B$ gives

$$B^{*} = \left(\frac{N}{2\pi}\right)^{K/(K-2)}.$$

For $K = 20$, this reduces to $B^{*} = (N/2\pi)^{10/9}$. Numerical evaluation yields $B^{*} = 13.2$ for $N = 64$ and $B^{*} = 28.5$ for $N = 128$. These values confirm that the empirically optimal $B = 10$ lies close to the theoretical full-utilization base for the shorter sequence lengths at which the Gauge-Transformer is trained, and that the standard $B = 10{,}000$ overshoots by nearly three orders of magnitude.

The exponent $K/(K-2)$ is close to unity for large $K$ (e.g., $K/(K-2) = 1.03$ at $K = 64$), so for standard transformer dimensions the optimal base is approximately $N/2\pi$ regardless of $K$. The distinction emerges at small $K$: at $K = 20$, the exponent is $10/9 \approx 1.11$, amplifying the base relative to the naive $N/2\pi$ estimate. This amplification reflects the fact that fewer pairs must span the same frequency range, requiring slightly compressed wavelengths per pair.

## 3. KL Divergence Decomposition

### 3.1 Position-Dependent and Position-Independent Terms

The attention weight between positions $i$ and $j$ in the Gauge-Transformer is determined by

$$\beta_{ij} = \text{softmax}_j\!\left(\frac{-\text{KL}(q_i \| \Omega_{ij}[q_j])}{\kappa \sqrt{K}}\right),$$

where $q_i = \mathcal{N}(\mu_i^{\text{rope}}, \Sigma_i)$ is the query belief with RoPE-rotated mean, $\Omega_{ij}[q_j] = \mathcal{N}(\Omega_{ij}\mu_j, \Omega_{ij}\Sigma_j\Omega_{ij}^\top)$ is the gauge-transported key belief (without RoPE, since only query means are rotated in the default configuration), and $\kappa$ is the temperature. The KL divergence between these $K$-dimensional Gaussians decomposes as

$$\text{KL}(q_i \| \Omega_{ij}[q_j]) = \underbrace{\frac{1}{2}\left[\text{tr}(\Sigma_t^{-1}\Sigma_i) - K + \log\frac{|\Sigma_t|}{|\Sigma_i|}\right]}_{\text{KL}_{\text{cov}}:\; \text{position-independent}} + \underbrace{\frac{1}{2}(\mu_t - \mu_i^{\text{rope}})^\top \Sigma_t^{-1} (\mu_t - \mu_i^{\text{rope}})}_{\text{KL}_{\text{pos}}:\; \text{position-dependent}},$$

where $\mu_t = \Omega_{ij}\mu_j$ and $\Sigma_t = \Omega_{ij}\Sigma_j\Omega_{ij}^\top$. The covariance term $\text{KL}_{\text{cov}}$ depends on $\Sigma_i$ and $\Sigma_t$ but not on the RoPE rotation, and is therefore position-independent. All positional information enters through the Mahalanobis term $\text{KL}_{\text{pos}}$, whose magnitude is controlled by the difference between the transported key mean $\mu_t$ and the RoPE-rotated query mean $\mu_i^{\text{rope}} = R(\theta_i)\mu_i$.

Since $\text{KL}_{\text{cov}}$ is approximately constant across keys $j$ (assuming beliefs have similar covariance structure), the softmax attention pattern is governed almost entirely by the variation in $\text{KL}_{\text{pos}}$ across $j$. A weak positional signal --- small variation in $\text{KL}_{\text{pos}}$ --- yields diffuse, position-insensitive attention.

### 3.2 Mahalanobis Discriminability

To isolate the effect of RoPE on positional discrimination, consider two positions $i$ and $j$ carrying the same content ($\mu_i = \mu_j = \mu$) with isotropic covariance $\Sigma = \sigma^2 I$ and identity transport ($\Omega_{ij} = I$). Under these simplifications, $\text{KL}_{\text{cov}} = 0$ and

$$\text{KL}_{\text{pos}}(i, j) = \frac{1}{2\sigma^2} \|R(\theta_i)\mu - R(\theta_j)\mu\|^2 = \frac{1}{2\sigma^2} \sum_{k=0}^{K/2-1} 2\|\mu_k\|^2 \bigl(1 - \cos(\Delta \cdot \theta_k)\bigr),$$

where $\Delta = |i - j|$ is the positional separation and $\mu_k \in \mathbb{R}^2$ is the content vector for the $k$-th rotation pair.

Taylor expansion of the cosine factor for pairs with $\theta_k \cdot \Delta \ll 1$ (i.e., wavelength $\lambda_k \gg \Delta$) gives

$$1 - \cos(\Delta \cdot \theta_k) = \frac{(\Delta \cdot \theta_k)^2}{2} - \frac{(\Delta \cdot \theta_k)^4}{24} + \mathcal{O}(\theta_k^6),$$

which tends to zero quadratically in $\theta_k$. For $B = 10{,}000$ and $K = 20$, pair $k = 9$ has $\theta_9 = 2.51 \times 10^{-4}$, so the cosine factor at $\Delta = 1$ evaluates to $3.15 \times 10^{-8}$ --- a contribution indistinguishable from zero in single precision. By contrast, at $B = 10$ the same pair yields $\theta_9 = 0.126$ and a cosine factor of $7.91 \times 10^{-3}$, a contribution five orders of magnitude larger.

The total adjacent-position discriminability $D^2(\Delta = 1)$, assuming unit content vectors $\|\mu_k\| = 1$ and $\sigma^2 = 1$, is the sum $\sum_k 2(1 - \cos\theta_k)$ over all $K/2$ pairs. For $K = 20$, this evaluates to $D^2 = 2.548$ at $B = 10$ and $D^2 = 1.106$ at $B = 10{,}000$, a ratio of $2.30\times$.

The full per-pair breakdown for $K = 20$ at the two extreme base values is recorded below.

**Table 1.** Per-pair angular resolution at $\Delta = 1$ (adjacent positions), $K = 20$.

| Pair $k$ | $\theta_k$ ($B\!=\!10$) | $2(1 - \cos\theta_k)$ | $\theta_k$ ($B\!=\!10{,}000$) | $2(1 - \cos\theta_k)$ |
|:---------:|:-----------------------:|:----------------------:|:-----------------------------:|:----------------------:|
| 0 | 1.000 | $9.19 \times 10^{-1}$ | 1.000 | $9.19 \times 10^{-1}$ |
| 1 | 0.794 | $5.98 \times 10^{-1}$ | 0.398 | $1.56 \times 10^{-1}$ |
| 2 | 0.631 | $3.85 \times 10^{-1}$ | 0.158 | $2.51 \times 10^{-2}$ |
| 3 | 0.501 | $2.46 \times 10^{-1}$ | 0.063 | $3.98 \times 10^{-3}$ |
| 4 | 0.398 | $1.56 \times 10^{-1}$ | 0.025 | $6.31 \times 10^{-4}$ |
| 5 | 0.316 | $9.92 \times 10^{-2}$ | 0.010 | $1.00 \times 10^{-4}$ |
| 6 | 0.251 | $6.28 \times 10^{-2}$ | 0.004 | $1.58 \times 10^{-5}$ |
| 7 | 0.200 | $3.97 \times 10^{-2}$ | 0.002 | $2.51 \times 10^{-6}$ |
| 8 | 0.158 | $2.51 \times 10^{-2}$ | $6.3 \times 10^{-4}$ | $3.98 \times 10^{-7}$ |
| 9 | 0.126 | $1.58 \times 10^{-2}$ | $2.5 \times 10^{-4}$ | $6.31 \times 10^{-8}$ |
| **Total** | | **2.548** | | **1.106** |

At $B = 10$, the signal is distributed across all 10 pairs, with even the slowest pair contributing measurably. At $B = 10{,}000$, pairs 4 through 9 contribute less than $10^{-3}$ collectively --- the positional signal is concentrated almost entirely in pair 0, with diminishing contributions from pairs 1--3. The model is left with approximately one effective degree of positional freedom.

### 3.3 Mean Positional KL and Attention Pattern Sharpness

Averaging $\text{KL}_{\text{pos}}$ over all position pairs $(i, j)$ with $i \neq j$ for $N = 128$, assuming unit content vectors and unit variance, yields the following characterization of positional signal strength across base values.

| $B$ | Mean $\text{KL}_{\text{pos}}$ | Std. Dev. | $\text{KL}_{\text{pos}}(\Delta\!=\!1)$ | $\text{KL}_{\text{pos}}(\Delta\!=\!64)$ |
|----:|------------------------------:|----------:|----------------------------------------:|-----------------------------------------:|
| 10 | 10.055 | 2.163 | 1.274 | 8.660 |
| 50 | 9.761 | 2.485 | 0.870 | 9.864 |
| 100 | 8.891 | 2.493 | 0.783 | 11.218 |
| 1,000 | 6.207 | 1.932 | 0.625 | 7.654 |
| 10,000 | 4.764 | 1.581 | 0.553 | 5.318 |

Two patterns emerge. First, the mean positional KL decreases monotonically with increasing $B$, from 10.055 at $B = 10$ to 4.764 at $B = 10{,}000$. This reflects the loss of positionally active dimensions. Second, the standard deviation --- which controls the sharpness of softmax attention patterns --- is likewise maximized at small $B$. Since the attention logit is $-\text{KL}_{\text{pos}} / (\kappa\sqrt{K})$, the absolute spread of logits across key positions scales with the standard deviation of $\text{KL}_{\text{pos}}$. A standard deviation of 2.163 (at $B = 10$) versus 1.581 (at $B = 10{,}000$) translates directly into sharper, more position-selective attention weights after softmax normalization.

The adjacent-position signal $\text{KL}_{\text{pos}}(\Delta = 1)$ deserves particular attention. In autoregressive language modeling, the model must discriminate tokens that are one position apart --- the immediately preceding token from those two or more steps back. This local discrimination is 2.3$\times$ stronger at $B = 10$ than at $B = 10{,}000$ (1.274 vs. 0.553). For the Gauge-Transformer, where attention is the sole mechanism for positional awareness (there are no residual position embeddings or position-dependent biases), this reduction in local discriminability directly degrades the model's ability to learn position-sensitive patterns.

## 4. Wavelength--Sequence Length Matching

### 4.1 Full Wavelength Spectrum

The complete wavelength spectrum for all 10 rotation pairs at $K = 20$ is presented in Table 2, with cells shaded to indicate whether the wavelength exceeds the sequence length $N = 128$.

**Table 2.** Wavelength $\lambda_k$ (in tokens) for each rotation pair $k$ at $K = 20$, across base values. Values exceeding $N = 128$ represent positionally inactive pairs.

| Pair $k$ | $B = 10$ | $B = 50$ | $B = 100$ | $B = 1{,}000$ | $B = 10{,}000$ |
|:---------:|---------:|---------:|----------:|---------------:|----------------:|
| 0 | 6.3 | 6.3 | 6.3 | 6.3 | 6.3 |
| 1 | 7.9 | 9.3 | 10.0 | 12.5 | 15.8 |
| 2 | 10.0 | 13.7 | 15.8 | 25.0 | 39.6 |
| 3 | 12.5 | 20.3 | 25.0 | 49.9 | 99.6 |
| 4 | 15.8 | 30.0 | 39.6 | 99.6 | 250.1 |
| 5 | 19.9 | 44.4 | 62.8 | 198.7 | 628.3 |
| 6 | 25.0 | 65.7 | 99.6 | 396.4 | 1,578.3 |
| 7 | 31.5 | 97.2 | 157.8 | 791.0 | 3,964.4 |
| 8 | 39.6 | 143.7 | 250.1 | 1,578.3 | 9,958.2 |
| 9 | 49.9 | 212.4 | 396.4 | 3,149.1 | 25,013.8 |

At $B = 10$, all wavelengths fall below $N = 128$, and in fact below $N = 64$, confirming full positional utilization at both sequence lengths. At $B = 100$, pairs 7--9 exceed $N = 128$. At $B = 10{,}000$, only pairs 0--3 have wavelengths shorter than 128, and pairs 5--9 have wavelengths exceeding $N$ by factors ranging from $5\times$ to $195\times$.

### 4.2 Entropy-Based Capacity Analysis

Each active rotation pair provides a periodic function of position whose phase can distinguish positional subsets. An information-theoretic lower bound on the number of active pairs required to uniquely encode $N$ positions is $\lceil \log_2 N \rceil$, since each pair partitions the sequence into at most a constant number of discriminable regions. For $N = 64$, 6 bits are required; for $N = 128$, 7 bits.

At $B = 10$, all 10 pairs are active, exceeding the 7-bit requirement with 3 bits of redundancy. This redundancy is beneficial: it allows the model to represent relative position through multiple frequency channels, analogous to how the cochlea represents pitch through overlapping frequency-tuned hair cells rather than a single resonator. At $B = 1{,}000$, only 4--5 pairs are active, falling below the 6--7-bit threshold and forcing the model to operate with an under-determined positional code. At $B = 10{,}000$, only 3--4 pairs are active, providing less than half the required capacity.

The mapping between active pair count and empirical perplexity ordering ($B = 10 > 50 > 100 > 1{,}000$) is monotonic: every reduction in active pair count corresponds to an increase in perplexity. This is consistent with the hypothesis that perplexity degradation at large $B$ is caused by positional under-resolution rather than other confounds.

## 5. Discussion

### 5.1 Why the Standard Base Fails

The standard RoPE base of $B = 10{,}000$ was designed for an operating regime characterized by $d_{\text{head}} \geq 64$ (yielding $\geq 32$ rotation pairs), sequence lengths $N \geq 2{,}048$, and dot-product attention. In that regime, even if the bottom half of pairs have wavelengths exceeding $N$, the remaining 16--32 pairs provide ample positional resolution. The Gauge-Transformer violates all three assumptions: $K/2 = 10$ pairs is an order of magnitude fewer, $N = 64$--$128$ is an order of magnitude shorter, and KL-based attention amplifies the effect of dead dimensions because the Mahalanobis term sums over all $K$ dimensions explicitly rather than through a single dot product.

The cost of dead rotation pairs is not merely wasted capacity. In the KL formulation, the Mahalanobis distance $\|\mu_t - \mu_i^{\text{rope}}\|_{\Sigma_t^{-1}}^2$ sums contributions from all dimension pairs. Dead pairs contribute near-zero positional signal but still carry content signal (since $\mu_t \neq \mu_i$ in general). The ratio of positional to content signal in the attention logit therefore decreases as pairs become inactive, diluting the model's ability to modulate attention based on position. With 10 active pairs at $B = 10$, position and content are roughly balanced across the $K$-dimensional belief space. With 3 active pairs at $B = 10{,}000$, the positional signal is confined to 6 of 20 dimensions while content occupies all 20, creating an asymmetry that biases attention toward position-invariant patterns.

### 5.2 Interaction with Gauge Transport

The gauge transport operator $\Omega_{ij} \in \text{GL}^+(K)$ acts on all $K$ dimensions of the key beliefs. RoPE rotations also act on all $K$ dimensions, but only inject positional information into the $2 \cdot N_{\text{active}}$ dimensions corresponding to active pairs. When $N_{\text{active}} = K/2$, the positional encoding is distributed uniformly across the belief space, and the gauge transport has maximal freedom to align content representations without interfering with positional signals. When $N_{\text{active}} \ll K/2$, the positional signal is concentrated in a low-dimensional subspace, and the gauge transport must simultaneously perform content alignment in the full space while avoiding disruption of the few position-encoding dimensions. This tension may explain why the perplexity degradation at large $B$ is more severe than the raw discriminability ratio ($2.3\times$) would suggest.

### 5.3 Scaling Recommendations

The optimal base formula $B^{*} = (N/2\pi)^{K/(K-2)}$ provides a principled starting point for RoPE base selection in architectures with non-standard dimensionality or sequence length. For the Gauge-Transformer at $K = 20$, this yields $B^{*} \in [13, 29]$ for $N \in [64, 128]$, consistent with the empirical finding that $B = 10$ performs best. The formula generalizes: for a hypothetical $K = 40$ Gauge-Transformer trained on $N = 256$ sequences, it predicts $B^{*} = (256/2\pi)^{40/38} = (40.7)^{1.053} \approx 48$. For the standard regime of $K = 128, N = 4096$, it gives $B^{*} = (652)^{1.016} \approx 680$ --- substantially below $10{,}000$ but less dramatically so, since the large number of rotation pairs provides tolerance for inactive pairs.

The formula can be adjusted for less aggressive utilization. Requiring $\lambda_{\max} = c \cdot N$ for a coverage factor $c > 1$ (allowing the slowest pair to span $c$ sequence lengths) gives $B^{*}(c) = (cN/2\pi)^{K/(K-2)}$. Setting $c = 2$ for the Gauge-Transformer at $K = 20, N = 128$ yields $B^{*} \approx 65$, which is close to the empirically second-best value of $B = 50$.

## 6. Conclusion

The monotonic improvement in Gauge-Transformer perplexity with decreasing RoPE base is a direct consequence of wavelength--sequence length mismatch at low belief dimensionality. With $K = 20$ and $N = 64$--$128$, the standard base of $B = 10{,}000$ renders 6--7 of 10 rotation pairs positionally inert, reducing adjacent-position discriminability by $2.3\times$ and leaving the model with fewer positional bits than the $\lceil\log_2 N\rceil$ minimum required for unique position encoding. The optimal base $B^{*} = (N/2\pi)^{K/(K-2)}$ evaluates to approximately 13--29 for this operating regime, confirming that $B = 10$ is near-optimal rather than anomalous. This is not a hyperparameter quirk but the mathematically correct scaling for the architecture's dimensionality and sequence length.

## Appendix A: Derivation Details

### A.1 Frequency Formula Verification

The frequency formula used in the implementation (`transport_ops.py:63`) is

```python
freqs = 1.0 / (base ** (torch.arange(0, half_K) / half_K))
```

which produces $\theta_k = B^{-k/\text{half\_K}} = B^{-2k/K}$ for $k = 0, \ldots, \text{half\_K} - 1$. The corresponding rotation angles at position $n$ are $\alpha_{n,k} = n \cdot \theta_k$, and the 2D rotation applied to dimension pair $(2k, 2k+1)$ is

$$R_k(\alpha_{n,k}) = \begin{pmatrix} \cos(n\theta_k) & -\sin(n\theta_k) \\ \sin(n\theta_k) & \cos(n\theta_k) \end{pmatrix}.$$

The full rotation matrix is block-diagonal: $R(\theta_n) = \text{diag}(R_0, R_1, \ldots, R_{K/2-1}) \in \text{SO}(2)^{K/2} \subset \text{GL}(K)$.

### A.2 Optimal Base Derivation

Starting from the full-utilization criterion $\lambda_{\max} = N$:

$$2\pi \cdot B^{(K-2)/K} = N$$
$$B^{(K-2)/K} = \frac{N}{2\pi}$$
$$B = \left(\frac{N}{2\pi}\right)^{K/(K-2)}.$$

For $K = 20$: $B^{*} = (N/2\pi)^{10/9}$.

Evaluating at $N = 64$: $B^{*} = (64/6.283)^{10/9} = (10.186)^{1.111} = 13.18$.

Evaluating at $N = 128$: $B^{*} = (128/6.283)^{10/9} = (20.372)^{1.111} = 28.48$.

### A.3 SymPy Verification

All expressions in this document were verified symbolically using SymPy 1.13. The frequency spectrum, wavelength formula, active pair count, optimal base expression, and Taylor expansion of the Mahalanobis cosine factor were computed symbolically and evaluated numerically at the parameter values reported in the tables. The LaTeX representations of the key expressions are:

$$\theta_k = B^{-\frac{2k}{K}}, \qquad \lambda_k = 2\pi B^{\frac{2k}{K}}, \qquad \lambda_{\max} = 2\pi B^{\frac{K-2}{K}}$$

$$k_{\max} = \frac{K \log(N/2\pi)}{2\log B}, \qquad B^{*} = \left(\frac{N}{2\pi}\right)^{\frac{K}{K-2}}$$

$$D^2_k = \frac{2\mu_k^2}{\sigma^2}\left(1 - \cos\!\left(B^{-\frac{2k}{K}}\Delta\right)\right), \qquad N_{\text{active}} = \left\lfloor \frac{K\log(N/2\pi)}{2\log B} \right\rfloor + 1$$
