# Verdict (bundled, four sections)

**Judge weighting:** Red identified five concrete issues; Blue conceded all five and recommended targeted fixes. The principal finding (R3.3, MF-VI vs joint-marginal conflation) is mathematically clear and survives Blue's defense. The other findings are smaller but real. No "false positive" red claims to discount.

---

## Section 1 — §Methods (3530–3559) and §Rock Example (3833–3906)

**Verdict: MINOR EDITS REQUIRED.**

- **3556 (Computational Details):** The "hence the iso-token rather than iso-FLOP budget choice" causal link is weak. Recommend either deleting the "hence" clause or naming the GPU (RTX 5090) so the reader understands that "the same machine" denotes the CPU+GPU workstation hosting both the Ouroboros simulations and the WikiText sweep. Not a correctness issue; presentation only.

The Rock Example section (3833–3906) is clean. No edit required.

---

## Section 2 — §Mathematical Details (3909–3964)

**Verdict: BLUE_WINS — no edits required.**

All four subsections (Gaussian KL, Fisher information, SO(3) generators with [G_i,G_j] = ε_{ijk}G_k, Cholesky parametrization) are standard and correctly stated.

---

## Section 3 — §Covariance Dynamics (3966–4158), §Quadratic Forms (4160–4221), §Augmented Hierarchical (4408–4512)

**Verdict: SUBSTANTIVE EDITS REQUIRED.**

Three real issues:

1. **Line 4154, "Collapse to attention-weighted message passing":** The cancellation Σ_i (Ω_{ij}Σ_jΩ_{ij}^T)^{-1} = I requires per-edge alignment, not the β-weighted average fixed point. The collapse paragraph cites "the covariance-alignment fixed point above" without restating that the per-edge form requires the homogeneity-plus-near-identity sub-regime established at (4118–4124). Edit: add a one-line conditioning note that the cancellation holds in the homogeneous, near-identity regime, citing (4123) directly.

2. **Lines 4214–4221, §Quadratic Forms β symbol clash:** β_{ij} := τ^(q)_{ij}/2 is an unbounded coupling strength (limit "β_{ij} → ∞" invoked at 4221), incompatible with the main-text β_{ij} as a row-normalized softmax probability with $\sum_j β_{ij} = 1$. Edit: rename the §Quadratic-Forms quantity to $\tilde\beta_{ij}$ (or equivalent) and add one line clarifying that the main-text softmax β_{ij} arises from the entropy-regularized variational selection applied to these raw couplings, not from the assignment $\beta_{ij} = \tau^{(q)}_{ij}/2$ itself.

3. **Lemma 3 (lines 4481–4502), MF-VI vs joint marginalization:** Principal finding. The lemma title and statement claim a mean-field VI optimum with covariance $\Omega\Sigma_\pi\Omega^T + \sigma^2 I$ and rigid-limit pushforward $\Omega_*q_\pi$. The actual MF-VI fixed point at parent-fixed gives $q_i = \mathcal{N}(\Omega\mu_\pi, \sigma^2 I)$ with rigid limit $\delta(k_i - \Omega\mu_\pi)$ — a point mass, not the pushforward. The covariance $\Omega\Sigma_\pi\Omega^T + \sigma^2 I$ is the **joint marginal** $\int p(k_i, k_\pi)\,dk_\pi$; the proof at line 4497 confirms this by writing "Integrating the joint density of $(k_i, k_{\pi(i)})$ explicitly," which is joint marginalization, a different operation from MF-VI. Edit: re-title and restate the lemma as a joint-marginal identity in the rigid limit (which holds and recovers the cross-scale shadow), and either drop the MF-VI framing or carry out the MF-VI step separately and identify its rigid-limit answer as the point mass at the transported mean. The structural cross-scale shadow claim survives under the corrected framing; only the proof needs the operation-swap repaired.

---

## Section 4 — §RG Construction body (4514–4592)

**Verdict: MINOR EDITS REQUIRED.**

- **Line 4572:** Hypothesis labels run (a), (b), (c), (e). Missing (d). Either rename (e) → (d), or restore the missing (d). The body content is otherwise structurally aligned with the rigorous appendix at 4595 (already debated and resolved).

---

## Aggregate

- §Methods: minor presentation edit at 3556.
- §Math Details: clean.
- §Covariance Dynamics + §Quadratic Forms + §Augmented Hierarchical: three substantive edits, with Lemma 3 being the load-bearing one.
- §RG body: one cosmetic label fix at 4572.

The Lemma 3 finding is the headline. Under the corrected framing (joint-marginal in the rigid limit, not MF-VI optimum), the cross-scale shadow construction remains rigorous; the proof needs the operation-swap repaired and the lemma retitled.
