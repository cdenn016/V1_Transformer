# Blue Opening (bundled, four sections)

## Section 1 — §Methods and §Rock Example

**B1.1 (response to R1.1).** Conceded in part. The "hence" at line 3556 is rhetorical: the iso-token vs iso-FLOP choice is reported transparently with per-K FLOP counts already disclosed in the same paragraph (3551), so there is no concealment. The GPU is not named because it is implicit in "PyTorch + CUDA + cuDNN," but explicitly naming it costs nothing and removes any ambiguity. Acceptable cleanup; not a correctness issue.

**B1.2.** Methods and Rock Example are otherwise clean — agreed.

## Section 2 — §Mathematical Details

**B2.1.** No defense needed. Section is correct as stated.

## Section 3 — §Covariance Dynamics, §Quadratic Forms, §Augmented Hierarchical

**B3.1 (response to R3.1).** Conceded. The cancellation at (4156) is valid only under per-edge alignment Σ_i = Ω_{ij}Σ_j Ω_{ij}^T for every j the sum runs over, which the βwt-fixed-point alone does not provide. A one-line clarification — "in the homogeneous, near-identity regime of §A.X established at (4123)" — is sufficient and should be added.

**B3.2 (response to R3.2).** Conceded. The Quadratic Forms section uses β_ij as a coupling strength in (4214–4221) and the main-text β_ij is a softmax probability. The cleanest fix is to relabel the Quadratic-Forms quantity as $\tilde\beta_{ij}$ or "raw coupling strength" and add a single line at line 4221 noting that the normalized softmax β_ij of the main text is recovered when the raw couplings are passed through the entropy-regularized variational selection that defines the softmax (cross-reference to the body's softmax definition). The reduction $\tau \to 1$ to a uniform $\tilde\beta_{ij} = 1/2$ is a notational anchor for the §Quadratic Forms derivation only; the main-text β is determined by the softmax of −KL/τ.

**B3.3 (response to R3.3, the principal finding).** The Red reading is correct on the mathematics. Mean-field VI applied to the augmented joint with parent fixed gives $q_i = \mathcal{N}(\Omega\mu_\pi, \sigma^2 I)$ — the receiver-side conditional Gaussian — whose rigid-link limit is $\delta(k_i - \Omega\mu_\pi)$, a point mass. The claim in (4485) that $q_i = \mathcal{N}(\Omega\mu_\pi, \Omega\Sigma_\pi\Omega^T + \sigma^2 I)$ is the joint marginal $\int p(k_i, k_\pi)\,dk_\pi$, not the MF-VI fixed point. The proof text at (4497) explicitly says "Integrating the joint density" — confirming that the proof switched operations mid-stride.

The defensible reformulation is: state the lemma as a **joint-marginal identity** in the rigid limit, not a mean-field-VI optimum. The augmented-joint construction supports the cross-scale shadow $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ as the rigid-link limit of the joint marginal $\int p(k_i, k_I)\,dk_I$, which equals the gauge pushforward when $\sigma^2 \to 0$. This is rigorous and corresponds to the manuscript's stated structural commitment. The mean-field-VI framing is not required for the cross-scale shadow claim and should be removed from the lemma title and statement; alternatively, both objects should be named separately, with the mean-field optimum identified as the degenerate point mass at $\Omega\mu_\pi$ in the rigid limit and the joint-marginal identification carried out as a separate step that recovers the pushforward.

The substantive content of the cross-scale shadow construction is unaffected — the joint-marginal identification holds. Only the lemma framing needs to be corrected.

## Section 4 — §RG Construction body

**B4.1 (response to R4.1).** Conceded. Label (d) is missing. Either rename (e) to (d), or restore the missing (d).

**B4.2.** No further defense needed.
