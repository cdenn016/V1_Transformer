# Red Opening (bundled, four sections)

## Section 1 — §Methods (3530–3559) and §Rock Example (3833–3906)

**R1.1 — Hardware/budget rationale is non-sequitur.** Line 3556 reads: "large-scale gauge-transformer training was run on the same machine, hence the iso-token rather than iso-FLOP budget choice." Iso-token vs iso-FLOP is dictated by methodological commitments (controlling for data exposure vs compute), not by single-machine deployment. The "hence" is a weak causal connective that does not carry. The CPU is named (Ryzen 9 9900X) but the GPU is not, even though "CUDA cuDNN seeds are synchronized" implies GPU was used and the working-config note records an RTX 5090. Either the GPU should be named or the explanatory link to iso-token should be reframed.

**R1.2 — All other Methods claims (seed values, K sweep range, bootstrap protocol, LLM-assistance disclosure) are concretely stated and verifiable against the repo. Rock Example derivations (3855–3905) are internally consistent: the ε → 0 scaling on both mean and covariance back-action is correctly computed, and the natural-gradient pre-factor argument is sound. No issue.**

## Section 2 — §Mathematical Details (3909–3964)

**R2.1 — Section is essentially clean.** Gaussian KL formula (3917) is the standard form. Fisher block (3927–3930) and natural-gradient factor of two (3933–3936) are correct. SO(3) commutation relation [G_i,G_j] = ε_{ijk}G_k (3945) is the standard form for so(3) basis generators. Cholesky parametrization (3961) is correct. No issue.

## Section 3 — §Covariance Dynamics (3966–4158), §Quadratic Forms (4160–4221), §Augmented Hierarchical (4408–4512)

**R3.1 — "Collapse to attention-weighted message passing" claim under-conditions its hypothesis.** Line 4154 ("Substituting the covariance-alignment fixed point above into this expression cancels the inner factor against the outer Σ_i") requires the per-edge identity Σ_i = Ω_{ij}Σ_j Ω_{ij}^T for every j the sum runs over. The β-weighted fixed point at (4101) — Σ_i^{-1} ≈ ⟨(Ω_{ij}Σ_j Ω_{ij}^T)^{-1}⟩_β — does not deliver per-edge cancellation; only the homogeneity-plus-near-identity sub-regime at (4118–4124) does. The collapse paragraph references "the covariance-alignment fixed point above" without restating that the per-edge form requires the strong sub-regime, leaving the cancellation claim under-conditioned.

**R3.2 — β_ij has two incompatible definitions across adjacent appendices.** In §Covariance Dynamics (4087–4099) and throughout the main text, β_ij is the row-normalized softmax of −KL/τ with $\sum_j β_{ij} = 1$, bounded in [0,1]. In §Quadratic Forms at line 4217, β_{ij} := τ^(q)_{ij}/2 is defined as an unbounded coupling strength, and the text at 4221 explicitly invokes the limit "β_{ij} → ∞." Then "we set τ to be... a constant global value which we set to 1," which gives β_{ij} = 1/2 (uniform), contradicting the softmax-distributional usage. The same symbol carries two non-overlapping meanings without any reconciliation note. At minimum a footnote is required distinguishing the §Quadratic Forms "raw coupling" β from the §Free Energy "normalized softmax" β.

**R3.3 — Lemma 3 (Cross-Scale Shadow as Mean-Field Optimum) conflates mean-field VI with joint marginalization.** This is the principal red finding. The lemma at line 4481 claims the mean-field marginal $q_i^{(s)} = \mathcal{N}(\Omega \mu_\pi, \Omega \Sigma_\pi \Omega^T + \sigma^2 I)$ with rigid-limit $\Omega_* q_\pi$. The proof at line 4494 correctly writes the mean-field stationarity:
$$
\log q_i(k_i) = \text{const} - \frac{1}{2\sigma^2}\,\mathbb{E}_{q_\pi}\big[\|k_i - \Omega k_\pi\|^2\big]
= \text{const} - \frac{1}{2\sigma^2}\big[\|k_i - \Omega \mu_\pi\|^2 + \operatorname{tr}(\Omega \Sigma_\pi \Omega^T)\big].
$$
The trace term is k_i-independent and absorbs into normalization. The remaining quadratic in k_i identifies $q_i$ as $\mathcal{N}(\Omega \mu_\pi, \sigma^2 I)$ — covariance $\sigma^2 I$, not $\Omega\Sigma_\pi\Omega^T + \sigma^2 I$. The rigid-link limit $\sigma^2 \to 0$ then collapses to $\delta(k_i - \Omega\mu_\pi)$, a point mass at the transported parent mean, not to the gauge pushforward $\Omega_* q_\pi$.

The closed form claimed in (4485) is the joint marginal $\int p(k_i, k_\pi)\,dk_\pi$, not the mean-field VI optimum. The proof text mid-stride says "Integrating the joint density of $(k_i, k_{\pi(i)})$ explicitly" — that is joint marginalization, a different operation. The lemma either needs to be retitled "Joint Marginal in the Rigid Limit" (and the variational-inference framing dropped), or the conclusion restated as the point-mass $\delta(k_i - \Omega\mu_\pi)$ in the rigid limit (with the covariance loss flagged as the cost of mean-field decoupling).

This bears directly on the cross-scale shadow status claim: the manuscript wants the shadow to be a rigorous pushforward $\Omega_* q_\pi$ recovering full covariance. If mean-field VI does not give this, the augmented-joint reduction does not produce the cross-scale shadow as stated; it produces a degenerate point-mass at the transported mean, with covariance information lost in the rigid limit.

**R3.4 — Other Augmented Hierarchical content (Lemma 1 well-definedness, Lemma 2 mean-field fixed-point equations) is standard and correctly stated.** No additional red.

## Section 4 — §RG Construction body (4514–4592)

**R4.1 — Hypothesis labeling error.** Line 4572: "Schur-complement elimination of the internal modes requires four hypotheses: (a)..., (b)..., (c)..., and (e) no near-critical soft mode." Four hypotheses are labeled (a), (b), (c), (e) — (d) is missing. Either a hypothesis was deleted and the labels were not renumbered, or (d) should be reinstated. Cosmetic but unambiguous.

**R4.2 — Other RG-body content (exact pushforward identity, closure-residual definition, internal Hessian and Schur-complement structure, raw-conductance coarse-graining, retention rules) is structurally aligned with the rigorous appendix at 4595 and self-consistent. No additional red beyond R4.1.**
