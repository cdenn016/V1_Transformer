# Verdict — softmax-beta-stationary-point

## Outcome

RED_WINS

## Decisive evidence

Blue's own rebuttal concession at `03_blue_rebuttal.md:4-6`:

> "no joint generative model `p(k_1, …, k_N, z_1, …, z_N)` is written down, no mean-field factorization of such a joint is performed, and no derivation of `Σ_j β_{ij} D_KL(q_i ‖ Ω_{ij} q_j)` from such an ELBO is shown anywhere in the two-file manuscript."

Cross-cut by `Attention/GL(K)_attention.tex:697` (the manuscript admits the per-row generative model's components `P(k|z=j) = Ω_{ij}q_j` are themselves variational quantities of other agents) and by `Attention/GL(K)_supplementary.tex:183` (the manuscript's own downstream working form is the "entropy-suppressed surrogate" with β fixed, and the canonical-F-with-entropy is described as the form that "adds the τβ log(β/π) entropy term to make the softmax form of β a stationary point" — additive language, not derivational).

## Reasoning

The claim is compound and explicitly enumerates "mixture-of-sources generative model" as the first of six intermediate steps that must be "mathematically valid," with sub-claim A naming the construction as "a well-defined Bayesian generative model." Blue's defense rebases the headline onto Question 2 ("given `F_align^(τ)` as written, β = softmax(-E/τ) is its exact stationary point") and concedes Question 1 ("is `F_align^(τ)` the correct upstream Bayesian functional derived from a joint generative model whose components do not depend on the variational quantities"). This rebasing contradicts the claim's own text: the enumeration cannot be read to exclude its first listed step, and sub-claim A is the load-bearing assertion of upstream Bayesian validity. Blue's CAVI rescue ([BleiKuckelbirgJordan2017 §3.2]) requires a fixed joint generative model `p(x_1, ..., x_M)`; blue's rebuttal grants the manuscript does not write one. The per-row model at line 688 has `P(k|z=j) = Ω_{ij} q_j` whose components are another agent's variational posterior, which the canon at `external_canon_inference.md` §1 flags as "novel if claimed to follow from FEP alone." The manuscript at line 697 invokes both "augmented joint" and "engineered consensus energy" framings and asserts equivalence without proof; the supplementary at line 183 quietly retreats to the surrogate form and uses additive language for the entropy term, corroborating that the entropy term is posited as a Lagrangian for the soft-assignment problem rather than derived from a joint ELBO. Sub-claims B, C, D, E (the row-Lagrangian algebra, the +1 absorption, the simplex normalization, the τ rescaling, the uniform-π specialization) are conceded clean by red and independently verified by both teams via sympy; the headline's "every intermediate step mathematically valid" clause fails on sub-claim A only. The headline is true under the weaker reading "exact stationary point of the posited soft-assignment Lagrangian" and false under the stronger reading the claim text demands ("derived via the row-Lagrangian with every intermediate step mathematically valid," where the enumeration lists the generative model as step one).

## Action

Apply one of two manuscript edits to §4.6 of `Attention/GL(K)_attention.tex`:

(a) **Rigorize framing (a).** Write the joint generative model `p(o_{1:N}, k_{1:N}, z_{1:N})` over all agents' latent states with parameters not depending on `{q_i}`, perform the mean-field factorization `Q = Π_i q_i(k_i) β_i(z_i)`, and derive `Σ_j β_{ij} D_KL(q_i ‖ Ω_{ij} q_j)` as the variational coupling produced by the joint ELBO. Cite the multi-agent variational ecology source (e.g., Friston2017Graphical, Ramstead2020) the construction is inherited from. This preserves the stronger reading of the headline.

(b) **Adopt framing (b) explicitly and weaken the headline.** Drop the "derived from a joint ELBO" invocation at line 697, state that `F_align^(τ)` is an engineered soft-assignment Lagrangian with the τβ log(β/π) entropy term added to make β = softmax(-E/τ) a stationary point (cite Cuturi 2013 §4 and the Boyd & Vandenberghe §5.5 KKT-on-the-simplex treatment), and replace "derived" with "constructed" or "posited" in the relevant headline. This keeps the algebra (sub-claims B-E) intact and is honest about the upstream framing the supplementary at line 183 already uses.

Both edits also need a one-line statement of strict convexity of `Σ_j β_j(E_j + log β_j - log π_j)` on the open simplex (Hessian `diag(1/β_k) > 0`) so that "exact stationary point" reads as "unique global minimum," and a one-line qualifier at line 734 that the "+ const" framing is uniform-π shorthand (under non-uniform π the `-log π` term is a prior bias inside the softmax, not a constant outside it). These were flagged by both teams as editorial cleanups.

The user's first-debate outcome (RED_WINS on §5 reduction) has already established that §5 takes the §4 attention rule as input; this verdict closes the upstream question by declaring the §4 derivation's outer Bayesian framing (sub-claim A) unfinished while preserving the inner soft-assignment algebra (sub-claims B-E) as standard and correct.
