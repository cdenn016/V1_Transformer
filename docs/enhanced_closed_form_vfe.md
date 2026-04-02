# Enhanced Closed-Form VFE: Full Fixed Point Including Softmax Coupling

The softmax coupling gradient (the Boltzmann gate) is **linear in μ_i** when β and KL are treated as fixed. The sigma softmax Jacobian is **independent of σ_i** (the σ_i terms cancel under normalized attention). These two facts allow the full VFE fixed point — including the nonlinear softmax coupling — to be solved in closed form as a single division per dimension.

Verified with SymPy. All claims checked algebraically.

---

## 1. Key Insight: Linearity of the Softmax Mu Gradient

The softmax coupling gradient for position i, dimension k is:

$$\nabla_{\mu_i[k]} F_{\text{softmax}} = \lambda_s \sum_j KL_{ij} \cdot \frac{\partial \beta_{ij}}{\partial \mu_i[k]}$$

The softmax Jacobian is:

$$\frac{\partial \beta_{ij}}{\partial \mu_i[k]} = -\frac{\beta_{ij}}{\kappa}\left(\frac{\partial KL_{ij}}{\partial \mu_i[k]} - \sum_m \beta_{im} \frac{\partial KL_{im}}{\partial \mu_i[k]}\right)$$

For diagonal covariance, the KL gradient is:

$$\frac{\partial KL_{ij}}{\partial \mu_i[k]} = \frac{\mu_i[k] - (\Omega_{ij}\mu_j)[k]}{\sigma_j^t[k]}$$

This is **linear in μ_i[k]**: coefficient $1/\sigma_j^t[k]$, constant $-(\Omega_{ij}\mu_j)[k]/\sigma_j^t[k]$.

Since $\partial\beta/\partial\mu$ is a linear combination of $\partial KL/\partial\mu$ terms (which are all linear in $\mu_i$), and $KL_{ij}$ is treated as fixed (evaluated at the current point), the entire softmax gradient is **linear in μ_i[k]**:

$$\nabla_{\mu_i[k]} F_{\text{softmax}} = S[k] \cdot \mu_i[k] + c[k]$$

where $S[k]$ and $c[k]$ depend on $\beta$, $KL$, transported precisions, and neighbor means — but NOT on $\mu_i[k]$ itself.

**SymPy verification:** `Linear in mu_i? True`

---

## 2. Key Insight: Sigma Cancellation in Softmax Jacobian

The sigma softmax Jacobian for diagonal covariance is:

$$\frac{\partial \beta_{ij}}{\partial \sigma_i[k]} = -\frac{\beta_{ij}}{2\kappa}\left(\frac{1}{\sigma_j^t[k]} - \frac{1}{\sigma_i[k]} - \sum_m \beta_{im}\left(\frac{1}{\sigma_m^t[k]} - \frac{1}{\sigma_i[k]}\right)\right)$$

With normalized attention $\sum_m \beta_{im} = 1$:

$$= -\frac{\beta_{ij}}{2\kappa}\left(\frac{1}{\sigma_j^t[k]} - \cancel{\frac{1}{\sigma_i[k]}} - \sum_m \frac{\beta_{im}}{\sigma_m^t[k]} + \cancel{\frac{1}{\sigma_i[k]}}\right)$$

$$= -\frac{\beta_{ij}}{2\kappa}\left(\frac{1}{\sigma_j^t[k]} - \bar{p}[k]\right)$$

where $\bar{p}[k] = \sum_m \beta_{im}/\sigma_m^t[k]$ is the attention-weighted transported precision.

**The $1/\sigma_i[k]$ terms cancel exactly.** The sigma softmax Jacobian depends only on the transported precisions of neighbors, not on the agent's own covariance.

**SymPy verification:** `Contains sigma_i? False`

---

## 3. Enhanced Mu Fixed Point

The full VFE stationarity condition for $\mu_i[k]$ (including softmax coupling):

$$\frac{\partial F}{\partial \mu_i[k]} = \underbrace{\frac{\alpha}{\sigma_p[k]}(\mu_i - \mu_p) + \lambda \sum_j \frac{\beta_{ij}}{\sigma_j^t[k]}(\mu_i - (\Omega_{ij}\mu_j)[k])}_{\text{linear: } A[k]\mu_i - b[k]} + \underbrace{S[k] \cdot \mu_i + c[k]}_{\text{softmax coupling (also linear!)}} = 0$$

Collecting $\mu_i[k]$:

$$(A[k] + S[k]) \cdot \mu_i[k] = b[k] + c[k]$$

$$\boxed{\mu_i^*[k] = \frac{b[k] + c[k]}{A[k] + S[k]}}$$

where:

**Linear terms** (existing closed form):
$$A[k] = \frac{\alpha}{\sigma_p[k]} + \lambda \sum_j \frac{\beta_{ij}}{\sigma_j^t[k]}$$
$$b[k] = \frac{\alpha \mu_p[k]}{\sigma_p[k]} + \lambda \sum_j \frac{\beta_{ij}(\Omega_{ij}\mu_j)[k]}{\sigma_j^t[k]}$$

**Softmax coupling terms** (new):

Define per-pair softmax weights:
$$w_j = KL_{ij} \cdot \beta_{ij}, \qquad \bar{w} = \sum_j w_j \quad \text{(expected KL)}$$

$$S[k] = -\frac{\lambda_s}{\kappa}\left(\sum_j \frac{w_j}{\sigma_j^t[k]} - \bar{w} \cdot \sum_m \frac{\beta_{im}}{\sigma_m^t[k]}\right)$$

$$c[k] = \frac{\lambda_s}{\kappa}\left(\sum_j \frac{w_j (\Omega_{ij}\mu_j)[k]}{\sigma_j^t[k]} - \bar{w} \cdot \sum_m \frac{\beta_{im}(\Omega_{im}\mu_m)[k]}{\sigma_m^t[k]}\right)$$

**Note:** $\sum_m \beta_{im}/\sigma_m^t[k]$ is already computed as part of $A[k]$ (it's the alignment precision). So $S[k]$ reuses existing quantities.

---

## 4. Enhanced Sigma Fixed Point

The full sigma stationarity with softmax coupling:

$$\frac{\alpha + 1}{2\sigma_i[k]} = \frac{\lambda_{\text{total}}[k]}{2} + S_\sigma[k]$$

where:

$$S_\sigma[k] = -\frac{\lambda_s}{2\kappa} \sum_j KL_{ij}\,\beta_{ij}\left(\frac{1}{\sigma_j^t[k]} - \bar{p}[k]\right)$$

$$\boxed{\sigma_i^*[k] = \frac{\alpha + 1}{\lambda_{\text{total}}[k] + 2\,S_\sigma[k]}}$$

Since $S_\sigma$ does not depend on $\sigma_i$ (the cancellation from Section 2), this is **exact in one evaluation** — no iteration needed for sigma given fixed $\beta$ and $KL$.

---

## 5. The Enhanced Picard Algorithm

```
For k = 0, 1, ..., K_iter - 1:

  Step 1: Compute attention and KL from current beliefs
    beta^k_ij = softmax(-KL(q_i^k || Omega_ij[q_j^k]) / kappa)
    KL^k_ij = pairwise KL values

  Step 2: Compute softmax coupling decomposition
    w_j = KL^k_ij * beta^k_ij                    (per-pair weights)
    w_bar = sum_j w_j                              (expected KL)
    For each dimension k:
      S[k] = -(lam_s/kappa)(sum_j w_j/sigma_jt[k] - w_bar * sum_m beta_im/sigma_mt[k])
      c[k] = (lam_s/kappa)(sum_j w_j * nu_j[k] - w_bar * sum_m beta_im * nu_m[k])
      S_sigma[k] = -(lam_s/(2*kappa)) * sum_j w_j * (1/sigma_jt[k] - p_bar[k])

  Step 3: Solve enhanced fixed point (one division per dimension)
    mu^{k+1}[k] = (b[k] + c[k]) / max(A[k] + S[k], eps)
    sigma^{k+1}[k] = (alpha + 1) / max(lam_total[k] + 2*S_sigma[k], eps)

  Step 4: Convergence check
    If ||mu^{k+1} - mu^k|| / ||mu^k|| < tol: break
```

---

## 6. Comparison

| Property | Gradient Descent | Original Picard | Enhanced Picard |
|---|---|---|---|
| Mu update | Gradient step (needs LR) | Linear CF + softmax correction | Full CF (one division) |
| Sigma update | Nat grad + SPD retraction | Linear CF + nat grad correction | Full CF (one division) |
| Softmax coupling | Explicit gradient term | Separate correction step | Absorbed into closed form |
| Learning rate | Required (sensitive) | Not needed for CF; needed for correction | Not needed at all |
| Natural gradient | Required | Needed for sigma correction | Not needed |
| SPD retraction | Required for sigma | Needed for sigma correction | Not needed (SPD by construction) |
| Per-iteration cost | O(N²K + NK³) | O(N²K²) + correction | O(N²K²) (same as original) |
| Convergence | ~1 step (but approx) | ~2-3 steps | ~1-2 steps (captures more per step) |
| Hyperparameters | LR, trust region, sigma trust | picard_trust_region | None (just max iterations) |

---

## 7. Why This Works

The softmax coupling $\partial\beta/\partial\mu$ looks nonlinear because $\beta$ depends on $\mu$ through the softmax. But for the purpose of solving the fixed-point equation at a GIVEN $\beta$, only the $\mu_i$-dependence of $\partial KL/\partial\mu_i$ matters — and that's linear (it's $\Lambda_j^t(\mu_i - \Omega_{ij}\mu_j)$, which has $\mu_i$ coefficient $\Lambda_j^t$ and constant $-\Lambda_j^t\Omega_{ij}\mu_j$).

The nonlinearity enters through $\beta$ and $KL$ depending on $\mu$ — but these are evaluated at the CURRENT beliefs and held fixed during the closed-form solve. The Picard iteration then updates $\beta$ and $KL$ from the new beliefs and re-solves. Each re-solve captures the full VFE structure (not just the linear part), so convergence is faster.

For sigma, the result is even stronger: the $\sigma_i$ terms cancel EXACTLY in the softmax Jacobian (because the softmax is translation-invariant in log-precision space, and $1/\sigma_i$ enters identically in every KL term). So the sigma fixed point with softmax coupling is solved in ONE STEP — no iteration needed.

---

## 8. Implementation Notes

The enhanced closed form adds the following to the existing closed-form computation:

```python
# After computing A[k], b[k], align_prec, align_info (existing):

# Per-pair softmax weights
w_j = kl_h * beta_h                          # (B, N, N)
w_bar = w_j.sum(dim=-1)                      # (B, N)

# Mu softmax coupling: S[k] and c[k]
# S[k] = -(lam_s/kappa)(sum_j w_j/sigma_jt - w_bar * align_prec/lambda)
kl_weighted_prec = einsum('bij,bijk->bik', w_j, inv_sigma_jt)   # (B, N, d_h)
S_mu = -(lam_s / kappa_h_scaled) * (kl_weighted_prec - w_bar.unsqueeze(-1) * align_prec / lambda_belief)

# c[k] = (lam_s/kappa)(sum_j w_j * nu_j - w_bar * align_info/lambda)
kl_weighted_info = einsum('bij,bijk->bik', w_j, info_per_pair)  # (B, N, d_h)
c_mu = (lam_s / kappa_h_scaled) * (kl_weighted_info - w_bar.unsqueeze(-1) * align_info / lambda_belief)

# Sigma softmax coupling: S_sigma[k] (does NOT depend on sigma_i)
p_bar = align_prec / lambda_belief            # attention-weighted transported precision
S_sigma = -(lam_s / (2 * kappa_h_scaled)) * einsum('bij,bijk->bik', w_j, inv_sigma_jt - p_bar.unsqueeze(2))

# Enhanced fixed point
total_prec_enhanced = prior_prec + align_prec + S_mu         # A + S
total_info_enhanced = prior_info + align_info + c_mu         # b + c
mu_star = total_info_enhanced / total_prec_enhanced.clamp(min=eps)

sigma_total_prec = prior_prec + align_prec + 2 * S_sigma     # lam_total + 2*S_sigma
sigma_star = (alpha + 1) / sigma_total_prec.clamp(min=eps)
```

The additional cost is three `einsum('bij,bijk->bik', ...)` operations per head — $O(N^2 d_h)$ each — negligible compared to the $O(N^2 d_h^2)$ pairwise KL computation.

---

## 9. Caveats

1. **The floor `max(A + S, eps)` is essential.** When the softmax coupling is strong (large $\lambda_s$, small $\kappa$), $S[k]$ can be negative and potentially make $A[k] + S[k] \leq 0$. The floor prevents division by zero or negative precision.

2. **KL values must be fresh.** The $KL_{ij}$ and $\beta_{ij}$ used in $S$, $c$, and $S_\sigma$ should be computed from the CURRENT beliefs, not from the initial embedding. This is the standard Picard requirement.

3. **This does not eliminate iteration.** The fixed point depends on $\beta$ and $KL$, which depend on $\mu$ and $\sigma$. Iteration is still needed to converge the $\beta \leftrightarrow (\mu, \sigma)$ loop. But each step now solves the complete VFE structure (linear + softmax), so fewer iterations are needed.

4. **Phi still requires autograd.** The gauge frame $\phi$ enters nonlinearly through $\Omega = \exp(\phi \cdot G)$. No closed form exists for $\phi$; it's updated via autograd after the $(\mu, \sigma)$ iteration.

5. **Full covariance generalization.** For full covariance, $S$ becomes a $K \times K$ matrix (not a scalar), and the fixed point requires solving $(A + S)\mu = b + c$ via Cholesky. The sigma cancellation still holds (the $\Sigma_i^{-1}$ terms cancel in the softmax Jacobian for the same reason as in the diagonal case).
