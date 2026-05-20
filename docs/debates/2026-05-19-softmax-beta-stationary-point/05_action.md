# Action — softmax-beta-stationary-point

**From verdict:** RED_WINS

## Recommended action

The verdict offers two mutually exclusive manuscript edits to `Attention/GL(K)_attention.tex` §4.6. Choose one:

### Option (a) — Rigorize framing (a): write the joint generative model

Write the joint generative model `p(o_{1:N}, k_{1:N}, z_{1:N})` over all agents' latent states with parameters not depending on `{q_i}`. Perform the mean-field factorization `Q = Π_i q_i(k_i) β_i(z_i)`. Derive `Σ_j β_{ij} D_KL(q_i ‖ Ω_{ij} q_j)` as the variational coupling produced by the joint ELBO. Cite the multi-agent variational ecology source the construction inherits from (Friston2017Graphical, Ramstead2020).

This preserves the stronger reading of the headline ("derived via the row-Lagrangian from FEP first principles") and would require a non-trivial paragraph or two of new text plus a citation chain. Recommended only if the user is prepared to commit to the FEP-derivation framing and can locate a primary source that justifies the variational-quantity-dependent generative-model construction.

### Option (b) — Adopt framing (b) explicitly and weaken the headline

Drop the "augmented joint model" invocation at line 697. State that `F_align^(τ)` is an engineered soft-assignment Lagrangian with the `τ β log(β/π)` entropy term added to make `β = softmax(-E/τ)` a stationary point. Cite Cuturi 2013 §4 (entropy-regularized assignment / Sinkhorn) and Boyd & Vandenberghe §5.5 (KKT on the simplex). Replace "derived" with "constructed" or "posited" in the relevant headline wording around §4.6.

This keeps the algebra (sub-claims B–E) intact and is honest about the upstream framing the supplementary at line 183 already uses ("the canonical free energy of the main text *adds* the τβ_{ij}log(β_{ij}/π_{ij}) entropy term to make the softmax form of β a stationary point" — additive language). Recommended as the lower-risk edit; it matches the supplementary's own wording and avoids the FEP-derivation overreach.

### Companion edits (apply in either option)

Two editorial cleanups flagged by both teams as orthogonal to the framing question:

1. **Add a one-line strict-convexity statement.** After the Lagrangian at line 741, note that the Hessian of `Σ_j β_j(E_j + log β_j - log π_j)` is `diag(1/β_k)` on the open simplex, strictly positive-definite, so the interior stationary point is the unique global minimum. This converts "exact stationary point" to "unique global minimum" without adding more than one sentence.

2. **Add a one-line qualifier at the "+ const" framing at line 734.** The energy-entropy decomposition `F_align = ⟨E⟩_β - H(β) + const` holds with "const" = `log N` under uniform π = 1/N; under non-uniform π the `-Σ β log π = ⟨-log π⟩_β` term is β-dependent (it enters the softmax as the prior bias `log π_k` at line 753, not as an additive constant). A parenthetical "(under uniform π; for general π, the `-log π` term enters the softmax as a prior bias)" suffices.

## Follow-up debates (if any)

The compound claim's enumeration surfaced five sub-claims. The verdict ruled:
- **Sub-claim A** (variational construction from a joint generative model) — **fails**, the joint model is not written.
- **Sub-claims B, C, D, E** (energy-entropy decomposition, KKT stationarity, uniform-π specialization, τ-rescaling) — **pass**, conceded clean by red and verified by both teams via sympy.

No remaining sub-claim in this debate needs its own follow-up. The headline-level dispute is settled.

The debate queue from the prior session's `05_action.md` still has two open items:

1. **Canonical F vs entropy-suppressed surrogate** — gradients differ by `-τ⁻¹ Cov_β(KL, ∇KL)` (envelope-gap at line 866–871; supplementary §B.1 line 183). The current verdict's reasoning at line 183 of the supplementary already touches this: the manuscript admits the entropy term is *added* to make β stationary, which is the canonical-F side; the supplementary's working form drops the entropy term (surrogate). A follow-up debate on whether the gradient gap is correctly characterized and whether the autograd-vs-envelope gap propagates to a measurable training difference would be the natural next debate in this series.

2. **Multi-head = block-diagonal GL(K)** — structural correspondence (§5.4, line 1696). The rectangular-projection caveat at line 1720 intersects the first debate's verdict (RED_WINS on §5 reduction). A focused sub-debate would test whether the thin-SVD lift is a structural correspondence or a chosen factorization.

3. **Route 1 (untied carving) alone reduces to Vaswani §3.2.1** — added during the first debate; remains open. Blue's strongest unrefuted move in that debate; warrants its own adjudication.
