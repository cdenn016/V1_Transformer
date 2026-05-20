# Red Rebuttal — softmax-beta-stationary-point

## Concession

I grant blue's points (1), (2), (3), (4), (6) and (7) in their entirety. Once one *posits* the row-functional `F_align = KL[Q ‖ P]` with `Q = q_i(k) β(z)` and `P(k|z=j) = (Ω_{ij} q_j)(k)`, `P(z=j) = π_j` and *treats `E_{ij}` as fixed during the inner β-optimization*, then:

- The `+1` absorption at line 747 → line 753 is symbolically tight (blue's sympy session in evidence (1) — `Attention/GL(K)_attention.tex:741–753`).
- The τ-rescaled stationary point `β = π exp(−E/τ)/Z` and the substituted minimum `F* = −τ log Z` are symbolically tight (blue's evidence (2) — manuscript lines 764–769).
- The Hessian `diag(1/β_k)` is strictly positive on the open simplex, so the interior stationary point is the unique global minimum [Boyd & Vandenberghe §3.1.4, §5.5.1]. The manuscript does not state this; blue is correct that it is a one-line fact.
- The `−log β_k → ∞` boundary blow-up keeps the inequality constraints `β_k ≥ 0` strictly inactive at any interior stationary point, so the equality-only Lagrangian at line 741 is sufficient [Cuturi 2013 §4].
- Forward-KL is the canonical variational direction [BleiKuckelbirgJordan2017 §2; `external_canon_inference.md` §4].
- The functional is structurally identical to the row-block of an entropy-regularized soft-assignment problem [Cuturi 2013 §4 Eq. 2].

These are sub-claims B, C, D, E and the convexity/uniqueness add-on. Blue carried the inner algebra cleanly.

## Core attack

Blue's defense pivots on evidence (5): the "augmented joint" framing at manuscript line 697 *is* block-CAVI [BleiKuckelbirgJordan2017 §3.2] applied to a joint generative model, with the β-update at line 741 as one CAVI block. This is the load-bearing move that converts the row-Lagrangian from a posited functional into a *derivation* — without it, the headline's "derived via the row-Lagrangian … with every intermediate step mathematically valid" reduces to "a posited functional has its own stationary point", which is a tautology.

The block-CAVI reading fails because **the joint generative model that block-CAVI requires is never written down in the manuscript**. Block-CAVI in [BleiKuckelbirgJordan2017 §3.2, Eq. 17–18] requires a *fixed* joint generative model `p(x_1, …, x_M)` over latents `x_1, …, x_M`. The mean-field family is `q(x) = Π_m q_m(x_m)`, and the CAVI update for factor `m` is
```
q_m*(x_m) ∝ exp{ E_{−m}[ log p(x_m, x_{−m}) ] }
```
with `E_{−m}` taken under the *current* `q_{−m} = Π_{m'≠m} q_{m'}(x_{m'})`. The prior `p` is fixed across the inner block; only the recognition factors `q_m'` enter as currently-frozen variational quantities.

The manuscript writes only the *per-row* mixture-of-sources generative model at line 688:
```
P(k, z) = P(k | z) P(z),   P(z = j) = π_j,   P(k | z = j) = N(k; Ω_{ij}μ_j, Ω_{ij}Σ_jΩ_{ij}^T).
```
That is a generative model whose component distribution `P(k|z=j) = Ω_{ij} q_j` is **itself a variational quantity** — `q_j` is another agent's posterior. Line 697 admits this:

> "the component distributions $P(k \mid z{=}j) = \Omega_{ij}q_j$ depend on the variational posteriors $q_j$ of other agents, making the generative model itself a function of variational quantities."

The manuscript then verbally invokes "an augmented model whereby the full system has a joint generative model over all agents' latent states, and the pairwise KL terms emerge as variational couplings in the mean-field decomposition of the joint ELBO" (line 697) — but **does not write the joint model down**. There is no equation in `GL(K)_attention.tex` §4.6 of the form `P(o_1, …, o_N, k_1, …, k_N, z_1, …, z_N) = …` whose mean-field ELBO decomposition produces line 715. The supplementary at line 183 confirms the scope: it switches to the "entropy-suppressed surrogate" and treats β as fixed, sidestepping the very paragraph blue is leaning on.

Blue is supplying the missing joint model on the manuscript's behalf, not citing it. [BleiKuckelbirgJordan2017 §3.2] permits prior dependence on other variational *factors* of a fixed joint `p`; it does not permit the *components of the joint `p` itself* to be variational quantities. The manuscript has the latter structure (per line 697), and the bridge to the former structure is asserted, not constructed.

Under the standard FEP starting point cited in `external_canon_inference.md` §1, `F[q] = E_q[log q − log p(o, s)]` is defined for a generative model `p(o, s)` with **fixed parameters**. The canon's pitfall list at `external_canon_inference.md` §10.6 is explicit:

> "Multi-agent FEP extended to single-agent. The user's coupled-agent F is a generalization. The standard FEP literature does not contain this specific functional with gauge-transport-coupled KL terms. … **Flag the user's coupling as novel.**"

And `external_canon_inference.md` §1, last paragraph:

> "Attention entropy term τ β log(β/π). The user's claim that this is required for softmax to be stationary is *internally consistent* … Standard if presented as a Lagrangian for the soft-assignment problem; **novel if claimed to follow from FEP alone.**"

The headline claim packages the inner algebra (which is standard soft-assignment Lagrangian, sub-claims B–E) together with the outer construction (which is the novel multi-agent FEP coupling, sub-claim A). Blue's evidence carries the inner; it does not carry the outer.

There are two readings of "derived via the row-Lagrangian":

- **(i)** The row-functional `F_align = KL[Q ‖ P_row]` is a posited objective, and its stationary point in β is the softmax. **True, but tautological.** A posited functional has its own stationary point by construction.

- **(ii)** The row-functional is the variational update for a *joint* FEP-derived generative model that the manuscript furnishes, with line 697's "augmented joint" framing realized as actual block-CAVI on a written-down `p(o_{1:N}, k_{1:N}, z_{1:N})`. **False on the missing-derivation count.** The joint model is invoked but not written.

The claim's text — "every intermediate step (mixture-of-sources generative model → mean-field variational posterior → energy-entropy decomposition → KKT/Lagrange-multiplier optimization → softmax solution → temperature rescaling) mathematically valid" — promises the second reading. Sub-claim A explicitly asserts the generative model "is a well-defined Bayesian generative model whose `KL[Q‖P]` … yields the form at Eq. eq:mixture_free_energy". A Bayesian generative model whose components are themselves recognition factors of other agents is not a well-defined Bayesian generative model in the standard sense [Friston2010; BleiKuckelbirgJordan2017 §3]; it requires either the explicit joint construction (not given) or an explicit consensus-energy posit (which line 697 also offers as an *alternative*, conceding the model-not-derived character). The manuscript's own "in either framing, the resulting update equations are the same" is a tell: the two framings are different mathematical objects, and asserting equivalence without proof is exactly the gap.

This breaks sub-claim A as written. The headline is then only true under reading (i), which weakens "derived" to "posited" and makes the headline trivially true at the cost of being load-bearing-false on the derivation claim.

## Defense

My opening's load-bearing assumption was that sub-claim A — "the mixture-of-sources generative model is a well-defined Bayesian generative model whose `KL[Q ‖ P]` mean-field decomposition yields Eq. eq:mixture_free_energy" — does not survive scrutiny. Blue's response is to offer the block-CAVI rescue. The rescue requires a fixed joint `p` that the manuscript does not provide. The defense holds.

Three additional supports:

(a) `external_canon_inference.md` §1 (Form 1, Form 2, Form 3) and the explicit pitfall at §10.3 — "FEP implies X. FEP is a variational principle; specific implementations follow from specific generative-model choices. Claims that FEP alone implies an architectural choice (attention, layer-norm, dropout) require the explicit generative model that connects them" — apply directly here. The manuscript writes only the per-row mixture (line 688), which is a per-agent generative model whose components are other agents' posteriors. The joint model required to make this a *single* FEP problem with fixed `p` is not provided. Per the canon's pitfall list, this is "claims that FEP alone implies an architectural choice … require the explicit generative model" — and the explicit joint generative model is missing.

(b) The manuscript itself offers two framings (line 697) and asserts equivalence. If block-CAVI were the rigorous resolution as blue claims, the consensus-energy alternative would be unnecessary. The manuscript's hedge — "In either framing, the resulting update equations are the same" — is exactly the structure of a derivation that has not committed to either path. Blue's own "concession in advance" grants this: "the manuscript at line 697 hand-waves the 'self-referential' question with two informal framings … This is the weakest paragraph in the derivation." That is sub-claim A failing, by blue's own admission, modulo the question of whether the inner algebra survives independently. The inner algebra does survive (sub-claims B–E, conceded above), but the headline's "with every intermediate step mathematically valid" includes sub-claim A, which is the missing intermediate step.

(c) The supplementary at `Attention/GL(K)_supplementary.tex:183` (the line 183 paragraph cited in evidence) explicitly retreats to the "entropy-suppressed surrogate" `Σ_j β_{ij} D_KL(q_i ‖ Ω_{ij} q_j)` for the covariance gradient — i.e., the manuscript itself works with a posited row-functional with β held fixed for the downstream Σ-derivation, not with a joint-ELBO mean-field decomposition. The "the canonical free energy of the main text adds the `τβ log(β/π)` entropy term to make the softmax form of β a stationary point" wording confirms reading (i): the entropy term is *added* to make β stationary, not *derived* from a joint generative model. This is the standard soft-assignment Lagrangian framing [Cuturi 2013 §4], which is exactly what `external_canon_inference.md` §1 last paragraph labels "standard if presented as a Lagrangian for the soft-assignment problem; novel if claimed to follow from FEP alone." The manuscript's main text wants the FEP framing; the supplementary's wording quietly confirms it is the Lagrangian framing.

The headline claim is therefore split-true: true on reading (i) where the row-functional is a posited soft-assignment Lagrangian (and the inner algebra goes through, per blue's sympy work); false on reading (ii) where the row-functional is the variational update for a joint FEP-derived generative model that the manuscript furnishes. The claim's wording — "derived via the row-Lagrangian … every intermediate step mathematically valid" — promises (ii) and delivers (i). Sub-claim A is the load-bearing step that is not delivered.
