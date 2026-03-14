# KL Divergence Under Constant GL(K) Gauge Transport

**Manuscript Limit 2** — full (non-isotropic, non-diagonal) covariances.

> Symbolic derivation verified with SymPy. See
> [`constant_gauge_kl_derivation.py`](constant_gauge_kl_derivation.py) for
> executable proofs.

---

## 1. Setup

Each token $i$ carries a Gaussian belief $q_i = \mathcal{N}(\mu_i, \Sigma_i)$
with $\mu_i \in \mathbb{R}^K$ and $\Sigma_i \in \mathrm{SPD}(K)$.

**Constant gauge** means every head uses a single learnable
$\Omega \in \mathrm{GL}(K)$ for all token pairs $(i,j)$, rather than
the pair-dependent $\Omega_{ij} = \exp(\varphi_i)\exp(-\varphi_j)$ of the
learned gauge.

The **pushforward** (parallel transport) of $q_j$ through $\Omega$ is:

$$
\Omega \cdot q_j \;=\; \mathcal{N}\!\bigl(\Omega\,\mu_j,\; \Omega\,\Sigma_j\,\Omega^\top\bigr)
$$

Attention weights are then:

$$
\beta_{ij} \;=\; \mathrm{softmax}_j\!\Bigl(-\frac{1}{\tau}\,D_\mathrm{KL}(q_i \;\|\; \Omega \cdot q_j)\Bigr),
\qquad \tau = \kappa\sqrt{K}
$$

---

## 2. General KL Formula

For $P = \mathcal{N}(\mu_P, \Sigma_P)$ and $Q = \mathcal{N}(\mu_Q, \Sigma_Q)$:

$$
D_\mathrm{KL}(P \| Q) \;=\; \tfrac{1}{2}\Bigl[
  \mathrm{tr}\!\bigl(\Sigma_Q^{-1}\Sigma_P\bigr)
  + (\mu_Q - \mu_P)^\top \Sigma_Q^{-1}(\mu_Q - \mu_P)
  - K
  + \ln\frac{\det\Sigma_Q}{\det\Sigma_P}
\Bigr]
$$

---

## 3. Substituting the Constant Transport

Substituting $\mu_Q = \Omega\mu_j$, $\Sigma_Q = \Omega\Sigma_j\Omega^\top$ into the KL:

$$
D_\mathrm{KL}(q_i \;\|\; \Omega \cdot q_j)
\;=\; \tfrac{1}{2}\Bigl[
  \underbrace{\mathrm{tr}\!\bigl((\Omega\Sigma_j\Omega^\top)^{-1}\Sigma_i\bigr)}_{\text{trace term}}
  + \underbrace{(\Omega\mu_j - \mu_i)^\top(\Omega\Sigma_j\Omega^\top)^{-1}(\Omega\mu_j - \mu_i)}_{\text{Mahalanobis term}}
  - K
  + \underbrace{\ln\det(\Omega\Sigma_j\Omega^\top) - \ln\det\Sigma_i}_{\text{log-det term}}
\Bigr]
$$

We now simplify each term using the key identity:

$$
(\Omega\Sigma_j\Omega^\top)^{-1} = \Omega^{-\top}\Sigma_j^{-1}\Omega^{-1}
$$

> **Verified symbolically** for $K{=}2$ with full SPD matrices. &#x2713;

---

## 4. Term-by-Term Decomposition

### 4a. Trace Term

$$
\mathrm{tr}\!\bigl(\Omega^{-\top}\Sigma_j^{-1}\Omega^{-1}\Sigma_i\bigr)
\;=\; \mathrm{tr}\!\bigl(\Sigma_j^{-1}\,\underbrace{\Omega^{-1}\Sigma_i\,\Omega^{-\top}}_{\widetilde{Q}_i}\bigr)
$$

where $\widetilde{Q}_i = \Omega^{-1}\Sigma_i\,\Omega^{-\top}$ is the
**pulled-back query covariance**.

> **Critical:** This term couples $\Omega$ with *both* $\Sigma_i$ and
> $\Sigma_j$, making it pair-dependent even though $\Omega$ is constant.
> There is **no** clean separation $S(\Omega) + \text{distance}$ in the
> general non-isotropic case.

### 4b. Mahalanobis Term

$$
(\Omega\mu_j - \mu_i)^\top \Omega^{-\top}\Sigma_j^{-1}\Omega^{-1}(\Omega\mu_j - \mu_i)
\;=\; (\mu_j - \Omega^{-1}\mu_i)^\top \Sigma_j^{-1}(\mu_j - \Omega^{-1}\mu_i)
$$

Defining $\tilde{q}_i = \Omega^{-1}\mu_i$ (the **pulled-back query mean**):

$$
\|\mu_j - \tilde{q}_i\|^2_{\Sigma_j^{-1}}
$$

> **Verified:** The forward-push form (used in code) equals the pullback form
> (used in manuscript). &#x2713;

### 4c. Log-Determinant Term

$$
\ln\det(\Omega\Sigma_j\Omega^\top) - \ln\det\Sigma_i
\;=\; \underbrace{2\ln|\!\det\Omega|}_{\text{constant}} + \ln\det\Sigma_j - \ln\det\Sigma_i
$$

---

## 5. The Full Formula

Collecting all terms:

$$
\boxed{
D_\mathrm{KL}(q_i \;\|\; \Omega \cdot q_j)
\;=\; \tfrac{1}{2}\Bigl[
  \mathrm{tr}\!\bigl(\Sigma_j^{-1}\widetilde{Q}_i\bigr)
  + \|\mu_j - \Omega^{-1}\mu_i\|^2_{\Sigma_j^{-1}}
  - K
  + 2\ln|\!\det\Omega|
  + \ln\frac{\det\Sigma_j}{\det\Sigma_i}
\Bigr]
}
$$

where $\widetilde{Q}_i = \Omega^{-1}\Sigma_i\,\Omega^{-\top}$.

---

## 6. What Cancels Under Softmax

Since $\beta_{ij} = \mathrm{softmax}_j(-D_\mathrm{KL}/\tau)$, terms that are
constant w.r.t. $j$ vanish:

| Term | Depends on $j$? | Cancels? |
|------|:---:|:---:|
| $\mathrm{tr}(\Sigma_j^{-1}\widetilde{Q}_i)$ | yes (via $\Sigma_j$) | **no** |
| $\|\mu_j - \Omega^{-1}\mu_i\|^2_{\Sigma_j^{-1}}$ | yes (via $\mu_j, \Sigma_j$) | **no** |
| $-K$ | no | yes |
| $2\ln|\!\det\Omega|$ | no | yes |
| $\ln\det\Sigma_j$ | yes | **no** |
| $-\ln\det\Sigma_i$ | no (query only) | yes |

**Effective attention logit:**

$$
\boxed{
\ell_{ij} \;=\; -\frac{1}{2\tau}\Bigl[
  \mathrm{tr}\!\bigl(\Sigma_j^{-1}\widetilde{Q}_i\bigr)
  + \|\mu_j - \Omega^{-1}\mu_i\|^2_{\Sigma_j^{-1}}
  + \ln\det\Sigma_j
\Bigr]
}
$$

### Interpretation

| Term | Role |
|------|------|
| $\mathrm{tr}(\Sigma_j^{-1}\widetilde{Q}_i)$ | **Covariance mismatch.** Measures how well the pulled-back query uncertainty $\widetilde{Q}_i$ aligns with key $j$'s geometry. |
| $\|\mu_j - \Omega^{-1}\mu_i\|^2_{\Sigma_j^{-1}}$ | **Content matching.** Mahalanobis distance between the pulled-back query mean and the key mean, in the key's own metric. |
| $\ln\det\Sigma_j$ | **Precision weighting.** Penalizes uncertain keys; confident keys receive higher attention. |

---

## 7. Geometric Bias $S(\Omega)$

Define the **geometric bias**:

$$
S(\Omega) \;=\; \tfrac{1}{2}\bigl[\ln\det(\Omega\Omega^\top) + \mathrm{tr}\!\bigl((\Omega\Omega^\top)^{-1}\bigr) - K\bigr]
$$

### Properties

- $S(\Omega) \geq 0$ for all $\Omega \in \mathrm{GL}(K)$
- $S(\Omega) = 0 \iff \Omega \in \mathrm{O}(K)$

> **Verified:** $S(R_\theta) = 0$ for 2D rotation, $S(\mathrm{diag}(s_1,s_2)) = \ln(s_1 s_2) + \tfrac{1}{2}(s_1^{-2} + s_2^{-2}) - 1 \geq 0$. &#x2713;

### When does $S(\Omega)$ cleanly separate?

**Only in the isotropic case** $\Sigma_i = \sigma^2 I$ for all $i$:

$$
\mathrm{tr}(\Sigma_j^{-1}\widetilde{Q}_i)
\;=\; \mathrm{tr}\!\bigl(\sigma^{-2}I \cdot \sigma^2(\Omega\Omega^\top)^{-1}\bigr)
\;=\; \mathrm{tr}\!\bigl((\Omega\Omega^\top)^{-1}\bigr)
$$

which depends **only** on $\Omega$. The full KL then becomes:

$$
D_\mathrm{KL}\big|_{\text{iso}} \;=\; S(\Omega) + \frac{1}{2\sigma^2}\|\mu_j - \Omega^{-1}\mu_i\|^2
$$

In the general non-isotropic case, no such decomposition exists — the trace
term $\mathrm{tr}(\Sigma_j^{-1}\widetilde{Q}_i)$ mixes $\Omega$, $\Sigma_i$,
and $\Sigma_j$ inseparably.

---

## 8. Connection to Standard Transformer (Limit 3)

Starting from the isotropic constant-gauge formula and absorbing
$\Omega^{-1}$ into learned projections:

$$
\beta_{ij} \;\propto\; \exp\!\Bigl(-\frac{1}{2\sigma^2\tau}\|\mu_j - \Omega^{-1}\mu_i\|^2\Bigr)
$$

Expanding the squared norm:

$$
\|\mu_j - \Omega^{-1}\mu_i\|^2 = \|\mu_j\|^2 - 2\mu_j^\top\Omega^{-1}\mu_i + \|\Omega^{-1}\mu_i\|^2
$$

Under softmax over $j$, the $\|\Omega^{-1}\mu_i\|^2$ term cancels (query-only), leaving:

$$
\ell_{ij} \;\propto\; \mu_j^\top\Omega^{-1}\mu_i - \tfrac{1}{2}\|\mu_j\|^2
$$

The first term is **dot-product attention** $K_j^\top Q_i$ with
$Q_i = \Omega^{-\top}\mu_i$. The second is a **key-norm penalty** that
standard transformers absorb via LayerNorm ($\|\mu_j\| \approx \sqrt{K}$)
and the $1/\sqrt{d_k}$ scaling (matching $\tau = \sqrt{K}$).

---

## 9. GL(K) Gauge Invariance

The KL divergence is invariant under simultaneous gauge transformation
$G \in \mathrm{GL}(K)$:

$$
D_\mathrm{KL}(G \cdot q_i \;\|\; (G\Omega G^{-1}) \cdot (G \cdot q_j))
\;=\; D_\mathrm{KL}(q_i \;\|\; \Omega \cdot q_j)
$$

This follows because $G\Omega G^{-1} \cdot G q_j = G(\Omega \cdot q_j)$, and
KL is invariant under invertible affine pushforward.

> **Verified numerically** with $\Omega = \bigl(\begin{smallmatrix}2&1\\0&1\end{smallmatrix}\bigr)$ and $G = \bigl(\begin{smallmatrix}3&1\\1&2\end{smallmatrix}\bigr)$: difference = 0. &#x2713;

---

## 10. Numerical Example ($K{=}2$)

$$
\Omega = \begin{pmatrix}2&1\\0&1\end{pmatrix}, \quad
\Sigma_i = \begin{pmatrix}2&\tfrac{1}{2}\\\tfrac{1}{2}&1\end{pmatrix}, \quad
\Sigma_j = \begin{pmatrix}1&\tfrac{1}{4}\\\tfrac{1}{4}&3\end{pmatrix}, \quad
\mu_i = \begin{pmatrix}1\\0\end{pmatrix}, \quad
\mu_j = \begin{pmatrix}0\\1\end{pmatrix}
$$

| Quantity | Value |
|----------|------:|
| $\det\Omega$ | $2$ |
| $S(\Omega)$ | $\ln 2 - \tfrac{1}{4} \approx 0.443$ |
| $D_\mathrm{KL}(q_i \;\|\; \Omega \cdot q_j)$ (full) | $\approx 0.739$ |
| $D_\mathrm{KL}$ (isotropic, $\sigma^2{=}1$) | $\tfrac{3}{8} + \ln 2 \approx 1.068$ |
| $S(\Omega) + \tfrac{1}{2}\|\mu_j - \Omega^{-1}\mu_i\|^2$ | $\tfrac{3}{8} + \ln 2$ &#x2713; |

The isotropic case decomposes cleanly. The non-isotropic case does not:
$D_\mathrm{KL} - S(\Omega) = 0.296 \neq \tfrac{1}{2}\|\cdot\|^2$.

---

## Summary

| # | Result |
|---|--------|
| 1 | **No** clean $S(\Omega) + \text{distance}$ decomposition in the non-isotropic case |
| 2 | Trace term $\mathrm{tr}(\Sigma_j^{-1}\widetilde{Q}_i)$ couples $\Omega$ with both $\Sigma_i$ and $\Sigma_j$ |
| 3 | $2\ln|\!\det\Omega|$, $K$, and $\ln\det\Sigma_i$ cancel under softmax |
| 4 | $\Omega^{-1}$ acts as learned projection: $\tilde{q}_i = \Omega^{-1}\mu_i$ (analogous to $W_Q$) |
| 5 | GL(K) gauge invariance: $D_\mathrm{KL}(G{\cdot}P \;\|\; G{\cdot}\Omega{\cdot}Q) = D_\mathrm{KL}(P \;\|\; \Omega{\cdot}Q)$ |
| 6 | Isotropic limit recovers $S(\Omega) + \frac{1}{2\sigma^2}\|\mu_j - \Omega^{-1}\mu_i\|^2$ |
| 7 | $S(\Omega) = 0 \iff \Omega \in \mathrm{O}(K)$ |
