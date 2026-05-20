# Action Items (bundled, four sections)

All edits target `Attention/Participatory_it_from_bit.tex`.

## A1. §Methods, line 3556 — Computational Details

Reframe the iso-token rationale and name the GPU. Either:

**Option A (minimal):** delete the "hence the iso-token rather than iso-FLOP budget choice" clause; the per-K FLOP numbers later in 3551 already make the methodological choice transparent.

**Option B (preferred):** name the GPU. Replace
> "large-scale gauge-transformer training (Section~\ref{sec:scaling_validation}) was run on the same machine, hence the iso-token rather than iso-FLOP budget choice."

with
> "large-scale gauge-transformer training (Section~\ref{sec:scaling_validation}) was run on the same workstation with an NVIDIA RTX 5090 GPU; the iso-token budget was chosen over iso-FLOP because per-K batch-size discretization made FLOP-matching impractical without sub-batch padding (per-K FLOP counts are reported in Section~\ref{sec:methods_scaling})."

## A2. §Covariance Dynamics, line 4154 — "Collapse to attention-weighted message passing"

Add a one-line conditioning note. Replace
> "Substituting the covariance-alignment fixed point above into this expression cancels the inner $(\Omega_{ij}\Sigma_j\Omega_{ij}^\top)^{-1}$ factor against the outer $\Sigma_i$..."

with
> "In the homogeneous near-identity regime of \eqref{eq:beta_weighted_precision}–(4123), where $\Sigma_i \approx \Omega_{ij}\Sigma_j\Omega_{ij}^\top$ holds per edge rather than only $\beta$-averaged, substituting the per-edge covariance-alignment identity into this expression cancels the inner $(\Omega_{ij}\Sigma_j\Omega_{ij}^\top)^{-1}$ factor against the outer $\Sigma_i$..."

## A3. §Quadratic Forms, lines 4214–4221 — β symbol clash

Rename the §Quadratic-Forms quantity to $\tilde\beta_{ij}$ to distinguish from the main-text softmax. Edit (4216–4221) to read:

> "$$\tilde\beta_{ij}:=\frac{\tau^{(q)}_{ij}}{2},\qquad\tilde\gamma_{ij}:=\frac{\tau^{(s)}_{ij}}{2}.$$
> These raw couplings measure the bare belief- and model-alignment strengths between agents. The normalized softmax weights $\beta_{ij}, \gamma_{ij}$ of the main text are obtained from these raw couplings by passing them through the entropy-regularized variational selection of \eqref{eq:softmax_attention_general} (or whichever label points to the softmax definition); the assignment $\beta_{ij} = \tilde\beta_{ij}$ holds only under the uniform-coupling, no-entropy-regularization specialization in which $\tilde\beta$ and the row-simplex constraint coincide. In all subsequent equations we set $\tau^{(q)}_{ij} = \tau^{(s)}_{ij} = 1$ as a notational anchor for the quadratic-form derivation, recovering the softmax-normalized form via the variational selection above."

Also update the `\to \infty` limit text — the unbounded limit applies to $\tilde\beta$, not to the normalized $\beta$.

## A4. §Augmented Hierarchical, Lemma 3 (lines 4481–4502) — MF-VI vs joint marginalization

This is the load-bearing edit. The lemma as written conflates two operations. Recommended restructure:

**Step 1.** Rename the lemma:
> **Lemma (Cross-Scale Shadow as Rigid-Link Joint Marginal).**

**Step 2.** Restate the claim. Replace the current statement with:
> "Assume the variational posterior of the parent $q_{\pi(i)}^{(s+1)}$ is Gaussian, $q_{\pi(i)}^{(s+1)} = \mathcal{N}(\mu_{\pi(i)}^{(s+1)}, \Sigma_{\pi(i)}^{(s+1)})$, and that the within-scale couplings $\beta_{ij}, \gamma_{ij}$ are set to zero so that the only term in~\eqref{eq:mf_fixedpoint} that constrains $q_i^{(s)}$ is the upward parent message. Then the joint marginal of the Gibbs factor against the parent posterior is the Gaussian
> $$
> \int \chi_{i,\pi(i)}^{(s)}(k_i, k_{\pi(i)})\, q_{\pi(i)}^{(s+1)}(k_{\pi(i)})\, dk_{\pi(i)} = \mathcal{N}\big(\Omega_{i,\pi(i)} \mu_{\pi(i)}^{(s+1)},\; \Omega_{i,\pi(i)} \Sigma_{\pi(i)}^{(s+1)} \Omega_{i,\pi(i)}^\top + \sigma^2 I\big),
> $$
> and in the rigid-link limit $\sigma^2 \to 0$ this marginal converges to the gauge pushforward $\Omega_{i,\pi(i)}\big[q_{\pi(i)}^{(s+1)}\big]$ exactly. The marginal of the augmented joint, restricted to the parent message, therefore recovers the cross-scale shadow $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ of Eq.~\eqref{eq:cross_scale_shadow}."

**Step 3.** Rewrite the proof to be a single-track joint-marginalization argument. The proof should:

1. Note that under zeroed within-scale couplings, the only edge in the factor graph incident on $k_i^{(s)}$ is the parent edge, so the marginal of the joint over $k_i$ is $\int \chi(k_i, k_\pi) q_\pi(k_\pi) dk_\pi$.
2. Carry out this Gaussian linear-transformation marginal (cite Bishop §2.3) to obtain $\mathcal{N}(\Omega\mu_\pi, \Omega\Sigma_\pi\Omega^T + \sigma^2 I)$.
3. Take $\sigma^2 \to 0$; the Gibbs factor collapses to $\delta(k_i - \Omega k_\pi)$ and the marginal becomes $\Omega_* q_\pi$.

**Step 4.** Add a short remark contrasting this with the mean-field VI fixed point. Suggested text:

> "**Remark (MF-VI vs joint marginal).** A separate computation, the mean-field VI fixed point at parent-fixed, gives $q_i^{(s),\mathrm{MF}} = \mathcal{N}(\Omega \mu_\pi, \sigma^2 I)$ with rigid-limit point mass $\delta(k_i - \Omega \mu_\pi)$ — the parent covariance is lost under mean-field decoupling. The cross-scale shadow construction uses the joint marginal of Lemma~\ref{lem:shadow_mf_optimum}, not the MF-VI optimum; the joint marginal retains the parent covariance through the rigid limit."

**Step 5.** Update §A.X "Implications for the Status of the Framework" (4504–4512) where it says "the cross-scale shadow ... is the rigid-link limit of the mean-field optimum": replace with "is the rigid-link limit of the joint marginal of the augmented joint over the parent."

**Step 6.** Update line 546 (the main-text cross-reference to Lemma 3) to read "joint marginal" rather than "mean-field optimum." Quoted target text:
> "the cross-scale shadow $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ is the rigid-link limit $\sigma^2 \to 0$ of the **joint marginal** of this joint over the parent (Lemma~\ref{lem:shadow_mf_optimum})"

## A5. §RG Construction body, line 4572 — hypothesis label gap

Rename hypothesis (e) to (d), or restore the missing (d). Either is acceptable; the rename is the minimum-change option.

Replace
> "...requires four hypotheses: (a) $G_{\xi\xi}$ is invertible on the internal-mode complement; (b) a spectral gap separates internal from retained modes, quantified by $\lambda_{I,w} > 0$ from~\eqref{eq:rg_constrained_gap}; (c) the internal equilibrium is normally hyperbolic, with the constrained internal Hessian $H_I^\perp$ having eigenvalues bounded away from zero on the constraint subspace; and (e) no near-critical soft mode within $\mathcal{N}_I$..."

with
> "...requires four hypotheses: (a) $G_{\xi\xi}$ is invertible on the internal-mode complement; (b) a spectral gap separates internal from retained modes, quantified by $\lambda_{I,w} > 0$ from~\eqref{eq:rg_constrained_gap}; (c) the internal equilibrium is normally hyperbolic, with the constrained internal Hessian $H_I^\perp$ having eigenvalues bounded away from zero on the constraint subspace; and (d) no near-critical soft mode within $\mathcal{N}_I$..."

Then update the downstream reference "Under (a)--(c), (e), and the timescale-separation..." → "Under (a)--(d) and the timescale-separation..." and "Failure of any of (a)--(c), (e)..." → "Failure of any of (a)--(d)...".

---

## Priority ranking

1. **A4 (Lemma 3 MF-VI vs joint marginal)** — load-bearing math correction.
2. **A2 (collapse-to-attention conditioning)** — load-bearing correctness clarification.
3. **A3 (β symbol clash)** — clarity and correctness.
4. **A5 (label gap)** — cosmetic but unambiguous.
5. **A1 (GPU naming / iso-token rationale)** — presentation only.
