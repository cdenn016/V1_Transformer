# Closed-Form E-Step with Picard Corrections: Full Derivation

## 1. Problem Setup

The VFE E-step minimizes the variational free energy with respect to the belief mean $\mu_i$ at each position $i$. The objective decomposes into three gradient terms:

$$
F(\mu_i) = F_{\text{self}}(\mu_i) + F_{\text{direct}}(\mu_i) + F_{\text{softmax}}(\mu_i)
$$

where the beliefs are Gaussian $q_i = \mathcal{N}(\mu_i, \Sigma_i)$, priors are $p_i = \mathcal{N}(\mu_p, \Sigma_p)$, and transported beliefs are $\Omega_{ij} q_j = \mathcal{N}(\Omega_{ij} \mu_j,\; \Omega_{ij} \Sigma_j \Omega_{ij}^\top)$.

The three terms are:

$$
F_{\text{self}} = \alpha \cdot \text{KL}(q_i \| p_i)
$$

$$
F_{\text{direct}} = \lambda \sum_j \beta_{ij} \cdot \text{KL}(q_i \| \Omega_{ij} q_j)
$$

$$
F_{\text{softmax}} = \lambda_s \sum_j \text{KL}_{ij} \cdot \frac{\partial \beta_{ij}}{\partial \mu_i}
$$

The first two terms are quadratic in $\mu_i$ (the "linear" part). The third involves the softmax Jacobian and is nonlinear.

## 2. KL Divergence and Its Gradient

For two Gaussians $\mathcal{N}(\mu_a, \Sigma_a)$ and $\mathcal{N}(\mu_b, \Sigma_b)$:

$$
\text{KL}(\mathcal{N}_a \| \mathcal{N}_b) = \frac{1}{2}\Big[\text{tr}(\Sigma_b^{-1}\Sigma_a) + (\mu_a - \mu_b)^\top \Sigma_b^{-1}(\mu_a - \mu_b) - K + \ln\frac{|\Sigma_b|}{|\Sigma_a|}\Big]
$$

Only the Mahalanobis term depends on $\mu_a$. The gradient with respect to $\mu_a$ is:

$$
\nabla_{\mu_a}\text{KL} = \Sigma_b^{-1}(\mu_a - \mu_b)
$$

For the transported KL term $\text{KL}(q_i \| \Omega_{ij} q_j)$, the transported covariance is $\Sigma_{j,t} = \Omega_{ij}\Sigma_j\Omega_{ij}^\top$ and:

$$
\nabla_{\mu_i}\text{KL}(q_i \| \Omega_{ij} q_j) = \Sigma_{j,t}^{-1}(\mu_i - \Omega_{ij}\mu_j)
$$

## 3. Transported Precision Identity

The inverse of the transported covariance factors as:

$$
\Sigma_{j,t}^{-1} = (\Omega_{ij}\Sigma_j\Omega_{ij}^\top)^{-1} = \Omega_{ij}^{-\top}\Sigma_j^{-1}\Omega_{ij}^{-1}
$$

With the factored transport $\Omega_{ij} = e^{\phi_i \cdot G}\, e^{-\phi_j \cdot G}$, the inverse is:

$$
\Omega_{ij}^{-1} = e^{\phi_j \cdot G}\, e^{-\phi_i \cdot G}
$$

Substituting and defining $E_i \equiv e^{-\phi_i \cdot G}$:

$$
\Sigma_{j,t}^{-1} = E_i^\top \underbrace{\Big(e^{\phi_j \cdot G}\Big)^\top \Sigma_j^{-1}\, e^{\phi_j \cdot G}}_{Q_j}\; E_i
$$

The intermediate quantity $Q_j$ depends only on position $j$ and is independent of $i$. This separation is the key to efficient batched computation.

## 4. Closed-Form Fixed Point

Setting the gradient of the linear terms to zero:

$$
\nabla_{\mu_i}(F_{\text{self}} + F_{\text{direct}}) = \alpha\,\Sigma_p^{-1}(\mu_i - \mu_p) + \lambda\sum_j \beta_{ij}\,\Sigma_{j,t}^{-1}(\mu_i - \Omega_{ij}\mu_j) = 0
$$

Collecting terms in $\mu_i$:

$$
\underbrace{\Big[\alpha\,\Sigma_p^{-1} + \lambda\sum_j \beta_{ij}\,\Sigma_{j,t}^{-1}\Big]}_{A_i}\;\mu_i = \underbrace{\alpha\,\Sigma_p^{-1}\mu_p + \lambda\sum_j \beta_{ij}\,\Sigma_{j,t}^{-1}\Omega_{ij}\mu_j}_{b_i}
$$

The mean fixed point is:

$$
\mu_i^* = A_i^{-1}\,b_i
$$

For the covariance, the $\Sigma_i$-dependent terms in the VFE are the trace terms $\frac{1}{2}\text{tr}(\Sigma_k^{-1}\Sigma_i)$ and the entropy terms $-\frac{1}{2}\ln|\Sigma_i|$. Setting $\partial F / \partial \Sigma_i = 0$:

$$
\frac{1}{2}A_i - \frac{1}{2}\underbrace{(\alpha + \lambda)}_{c}\,\Sigma_i^{-1} = 0
$$

where $c = \alpha + \lambda\sum_j\beta_{ij} = \alpha + \lambda$ (since $\sum_j\beta_{ij} = 1$ under softmax). Each KL contributes a $-\ln|\Sigma_i|$ entropy term; the total coefficient is $c$. Solving:

$$
\Sigma_i^* = (\alpha + \lambda)\,A_i^{-1}
$$

The factor $(\alpha + \lambda)$ arises because the entropy terms favor larger covariance (more uncertainty), inflating the precision-only solution $A^{-1}$ by $(\alpha + \lambda)$. With per-dimension $\alpha_k$ (learnable precision), the factor becomes $c_k = \alpha_k + \lambda$ per dimension.

$A_i$ is the posterior precision matrix (SPD by construction as a sum of SPD matrices) and $b_i$ is the information vector.

## 5. Diagonal Specialization

When all covariances are diagonal ($\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_K)$) and transport is diagonal:

$$
\Sigma_{j,t}^{-1}[k,k] = \frac{1}{\omega_{ij,k}^2\,\sigma_{j}[k]}
$$

The precision and information reduce to element-wise operations:

$$
A_i[k] = \frac{\alpha}{\sigma_p[k]} + \lambda\sum_j \frac{\beta_{ij}}{\omega_{ij,k}^2\,\sigma_j[k]}
$$

$$
b_i[k] = \frac{\alpha\,\mu_p[k]}{\sigma_p[k]} + \lambda\sum_j \frac{\beta_{ij}\,\mu_j[k]}{\omega_{ij,k}\,\sigma_j[k]}
$$

$$
\mu_i^*[k] = b_i[k] / A_i[k] \qquad \sigma_i^*[k] = (\alpha + \lambda) / A_i[k]
$$

This is the existing implementation in `variational_ffn.py` (lines 3551-3616). The diagonal approximation discards off-diagonal entries of $\Omega\,\text{diag}(\sigma_j)\,\Omega^\top$ when $\Omega$ is non-diagonal, which is exact only when $\Omega$ is itself diagonal or when $\Omega$ is orthogonal with diagonal source covariance.

## 6. Full-Covariance Batched Computation

Using the $Q_j$ factorization, the alignment terms can be computed without an $N \times N$ pair loop.

**Precompute** (once per head, $O(B \cdot N \cdot d_h^3)$):

$$
Q_j = (e^{\phi_j \cdot G})^\top\,\Sigma_j^{-1}\,e^{\phi_j \cdot G} \qquad r_j = e^{-\phi_j \cdot G}\,\mu_j
$$

**Aggregate** ($O(B \cdot N^2 \cdot d_h^2)$):

$$
\bar{Q}_i = \sum_j \beta_{ij}\,Q_j \qquad \overline{Qr}_i = \sum_j \beta_{ij}\,Q_j\,r_j
$$

**Transform to position $i$'s frame** ($O(B \cdot N \cdot d_h^2)$):

$$
A_{\text{align},i} = \lambda\,E_i^\top\,\bar{Q}_i\,E_i \qquad b_{\text{align},i} = \lambda\,E_i^\top\,\overline{Qr}_i
$$

**Solve** ($O(B \cdot N \cdot d_h^3)$ via Cholesky):

$$
A_i = \alpha\,\Sigma_p^{-1} + A_{\text{align},i} \qquad b_i = \alpha\,\Sigma_p^{-1}\mu_p + b_{\text{align},i}
$$

$$
L\,L^\top = A_i \qquad \mu_i^* = L^{-\top}L^{-1}b_i \qquad \Sigma_i^* = L^{-\top}L^{-1}
$$

### Cost comparison (per head, $d_h = 15$, $N = 256$)

| Operation | FLOPs |
|-----------|-------|
| Attention ($N^2 \cdot d_h$) | 983K |
| Full-cov aggregation ($N^2 \cdot d_h^2$) | 14.7M |
| Cholesky solve ($N \cdot d_h^3$) | 864K |

The aggregation step dominates at $O(N^2 d_h^2)$ vs attention's $O(N^2 d_h)$, giving a factor-of-$d_h$ overhead. For $d_h = 15$ this is a 15x multiplier on the aggregation, though the Cholesky solve itself is cheaper than attention.

### Verified einsum expressions

The following einsum index orderings were verified numerically to machine precision ($2.32 \times 10^{-15}$):

```python
# Q_j: exp_phi_j^T @ Sigma_j^{-1} @ exp_phi_j
Q_j = einsum('bjlk, bjlm, bjmn -> bjkn', exp_phi, Sigma_j_inv, exp_phi)

# r_j: exp_neg_phi_j @ mu_j
r_j = einsum('bjkl, bjl -> bjk', exp_neg_phi, mu_h)

# Aggregation
Q_agg = einsum('bij, bjkl -> bikl', beta, Q_j)
Qr_j  = einsum('bjkl, bjl -> bjk', Q_j, r_j)
Qr_agg = einsum('bij, bjk -> bik', beta, Qr_j)

# Frame transform (CRITICAL: 'bikl' for E_i^T, NOT 'bilk')
A_align = lambda_belief * einsum('bikl, bikm, bimn -> biln', E_i, Q_agg, E_i)
b_align = lambda_belief * einsum('bikl, bik -> bil', E_i, Qr_agg)
```

The transpose convention is: for a tensor `M[b,i,k,l]` representing a matrix with rows $k$ and columns $l$, the einsum index `'bikl'` contracts over column index $l$, producing $M^\top \cdot (\ldots)$.

## 7. Softmax Coupling Gradient

The nonlinear term dropped by the closed-form is:

$$
\nabla_{\mu_i} F_{\text{softmax}} = \lambda_s \sum_j \text{KL}_{ij}\,\frac{\partial \beta_{ij}}{\partial \mu_i}
$$

The softmax Jacobian is:

$$
\frac{\partial \beta_{ij}}{\partial \mu_i} = -\frac{\beta_{ij}}{\kappa}\Big(\frac{\partial \text{KL}_{ij}}{\partial \mu_i} - \sum_k \beta_{ik}\frac{\partial \text{KL}_{ik}}{\partial \mu_i}\Big)
$$

where $\partial \text{KL}_{ij}/\partial \mu_i = \Sigma_{j,t}^{-1}(\mu_i - \Omega_{ij}\mu_j)$ is the per-pair KL gradient (a $K$-vector).

For full covariance, this involves the full precision matrix $\Sigma_{j,t}^{-1}$ in matrix-vector products rather than the element-wise operations used in the diagonal case.

## 8. Picard Iteration

At the closed-form fixed point $\mu_0$, the linear gradient vanishes by construction:

$$
\nabla(F_{\text{self}} + F_{\text{direct}})\big|_{\mu_0} = 0
$$

The full gradient equals the softmax term alone:

$$
\nabla F(\mu_0) = \nabla F_{\text{softmax}}(\mu_0)
$$

The Picard iteration uses the closed-form covariance $\Sigma_0 = A^{-1}$ as the exact preconditioner (inverse Hessian of the linear part):

$$
\mu^{(n+1)} = \mu_0 - \Sigma_0\,\nabla F_{\text{softmax}}(\mu^{(n)})
$$

Each iteration evaluates the softmax gradient at the current point and applies a single preconditioned correction, always relative to the closed-form base point $\mu_0$.

### Diagonal version

$$
\mu^{(n+1)}[k] = \mu_0[k] - \sigma_0[k] \cdot \nabla_k F_{\text{softmax}}(\mu^{(n)})
$$

Element-wise multiplication. Already implemented.

### Full-covariance version

$$
\mu^{(n+1)} = \mu_0 - \Sigma_0 \,\nabla F_{\text{softmax}}(\mu^{(n)})
$$

Per-position matrix-vector product:

```python
correction_h = einsum('bijk, bik -> bij', Sigma_star_h, grad_softmax_h)
```

### Residual after one Picard step

After the first correction $\mu_1 = \mu_0 - \Sigma_0\,\nabla F_{\text{softmax}}(\mu_0)$, the linear gradient at $\mu_1$ is:

$$
\nabla F_{\text{linear}}(\mu_1) = A(\mu_1 - \mu_0) = A\big(-\Sigma_0\,\nabla F_{\text{softmax}}(\mu_0)\big) = -\nabla F_{\text{softmax}}(\mu_0)
$$

So the full gradient at $\mu_1$ is:

$$
\nabla F(\mu_1) = -\nabla F_{\text{softmax}}(\mu_0) + \nabla F_{\text{softmax}}(\mu_1)
$$

The residual depends only on how much the softmax gradient changed between $\mu_0$ and $\mu_1$.

### Convergence condition

The map $T(\mu) = \mu_0 - \Sigma_0\,\nabla F_{\text{softmax}}(\mu)$ is contractive when:

$$
\rho\big(\Sigma_0 \cdot H_{\text{softmax}}\big) < 1
$$

where $H_{\text{softmax}}$ is the Hessian of $F_{\text{softmax}}$ and $\rho(\cdot)$ denotes the spectral radius. Since $\Sigma_0 = A^{-1}$ has eigenvalues bounded by $1/\lambda_{\min}(A)$, and softmax derivatives are bounded by $1/(4\kappa)$, convergence requires roughly:

$$
\frac{\lambda_s}{\kappa \cdot \lambda_{\min}(A)} < 1
$$

Empirically, corrections decay as $0.026 \to 0.010 \to 0.005$ per step (contractive even when softmax terms are $O(1)$ relative to linear terms). Two to three steps capture most of the nonlinear correction.

## 9. Sigma Under Picard

The Picard iteration updates only $\mu$, keeping $\Sigma_0$ fixed. This is correct because:

1. At the linear fixed point, $\Sigma_0 = A^{-1}$ is the exact posterior covariance under the linearized VFE.

2. The softmax coupling term does not contribute to the $\Sigma$ update at first order. The trace and log-determinant terms in KL are $\mu$-independent; only the Mahalanobis term generates $\mu$ gradients.

3. Recomputing $A_i$ at each Picard step would defeat the purpose of the closed-form factorization.

A second-order sigma correction is available via $\Sigma_1 = (A_0 + H_{\text{softmax}}(\mu_1))^{-1}$, but this requires the full softmax Hessian and is rarely needed.

## 10. Implementation Reference

### Configuration (`block_config.py`)

```python
n_picard_steps: int = 0        # 0 = pure closed-form, 2-3 recommended
picard_trust_region: float = 5.0   # whitened Mahalanobis trust region
closed_form_e_step: bool = False   # enable closed-form path
```

### Code locations

- Diagonal closed-form: `variational_ffn.py:3520-3631`
- Diagonal Picard loop: `variational_ffn.py:3637-3695` (approx, after closed-form block)
- Full-covariance closed-form: to be implemented as `else` branch of `is_diagonal` guard
- Q_j factorization: new code in the full-covariance branch
- Softmax gradient: pattern at `variational_ffn.py:1025-1029` (block-diagonal diagonal path)
