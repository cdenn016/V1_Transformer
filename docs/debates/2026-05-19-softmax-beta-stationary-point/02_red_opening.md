# Red Opening — softmax-beta-stationary-point

## Steelman (opposing position)

The row-Lagrangian at `GL(K)_attention.tex` lines 738–753 is a textbook entropy-regularized soft-assignment minimization on the simplex; sympy confirms the algebra; primary sources (Cuturi 2013 §4 *Sinkhorn Distances*; Boyd & Vandenberghe *Convex Optimization* §5.5) give the same Gibbs/softmax stationary point under uniform prior `π_k = 1/N`, so the headline "exact stationary point" result for `F_align^(τ) = Σ_j[β_ij E_ij + τ β_ij log(β_ij/π_j)]` is mathematically correct.

## Position

The claim is compound: it asserts (i) softmax is the exact stationary point of `F_align^(τ)`, **and** (ii) this is **derived** via the row-Lagrangian construction at §4.6 lines 679–769 with **every intermediate step mathematically valid**. Part (i) holds; part (ii) does **not** hold as written, because the very first intermediate step — writing `F_align = KL[Q ‖ P]` at line 715 — uses a "generative model" `P(k | z = j) = Ω_{ij} q_j` whose component distributions are themselves variational quantities (line 697). The manuscript admits this and offers two verbal framings as the resolution; **neither framing is actually executed in the manuscript**, so step (b) is a *definition by fiat*, not a derivation. Step (d) at line 725 ("with no approximation") is in direct tension with the line-697 admission. The headline therefore fails on the "every intermediate step mathematically valid" clause; what survives is the weaker statement that *once `F_align` is posited as a functional of `(β, q_i, {q_j})` with `q_j` held fixed for the β-update*, the softmax form is its exact stationary point under entropy regularization.

## Evidence

**1. The manuscript itself flags the self-reference at line 697** (`Attention/GL(K)_attention.tex:697`):

> "the component distributions $P(k | z=j) = \Omega_{ij}q_j$ depend on the variational posteriors $q_j$ of other agents, making the generative model itself a function of variational quantities."

The manuscript then offers two resolutions: (a) "augmented model whereby the full system has a joint generative model over all agents' latent states, and the pairwise KL terms emerge as variational couplings in the mean-field decomposition of the joint ELBO"; (b) "consensus energy whereby the alignment free energy ... can be viewed as an **engineered energy functional** that penalizes belief disagreement after gauge transport, motivated by the KL term of a fixed generative model" (emphasis added; manuscript's own word).

**2. Neither resolution is exhibited.** A grep of both `Attention/GL(K)_attention.tex` and `Attention/GL(K)_supplementary.tex` for `augmented joint`, `consensus energy`, `joint generative model`, `joint ELBO` returns line 697 of the main text as the **only hit in the entire two-file manuscript**. No joint generative model `p(k_1, \ldots, k_N, z_1, \ldots, z_N)` is written down anywhere. No mean-field factorization of such a joint is performed. No derivation of `Σ_j β_{ij} D_KL(q_i ‖ Ω_{ij} q_j)` as the variational coupling produced by decomposing such an ELBO is given. Framing (a) is invoked as a name, never as a derivation.

**3. The standard FEP / variational Bayes generative model is parameter-fixed.** [Friston2010 Eq. 2.2; `external_canon_inference.md` §1] gives variational free energy as `F[q] = E_q[log q(s) − log p(o, s)]` where `p(o, s)` is a *fixed* (parameter-fixed) generative model and `q(s)` is the recognition density. The KL-bound interpretation `F = KL(q ‖ p(s | o)) − log p(o)` is *only* a bound on `−log p(o)` when `p` does not depend on `q`. The same canon excerpt explicitly flags multi-agent gauge-coupled KL forms as "Standard if presented as a Lagrangian for the soft-assignment problem; **novel if claimed to follow from FEP alone**" (`external_canon_inference.md` §1, last paragraph). The manuscript's framing (a) ("variational couplings in the mean-field decomposition of the joint ELBO") is the FEP-derivation framing, which the canon flags as the one requiring independent justification when used with gauge-coupled KL terms.

**4. Internal tension between lines 725 and 697.** Line 725 states the identification `P(k | z=j) = (\Omega_{ij} q_j)(k)` proceeds "with no approximation." Line 697 has already admitted that this identification makes the generative model depend on variational quantities, requiring framing (a) or (b) as a remedy. If line 725's "no approximation" is meant as a calculus statement (the Gaussian integral evaluates exactly), it is correct. If it is meant to certify the entire `F_align = KL[Q ‖ P]` construction as a Bayesian KL bound, it conflicts with line 697's admission that this is either an unexecuted ELBO derivation or an engineered functional. The manuscript's "every intermediate step mathematically valid" reading needs to disambiguate which meaning is in force.

**5. The "engineered energy functional" wording is a self-disclosure** (line 697). The manuscript's own framing (b) describes the construction as **engineered**, i.e., posited rather than derived. Under framing (b), step (b) of the derivation (line 715, writing `F_align = KL[Q ‖ P]`) is a definition, not a derivation. The row-Lagrangian then minimizes this posited functional exactly — which is sub-claim C (algebra), not sub-claim A (variational construction). The claim's "every intermediate step mathematically valid" verbiage glosses the definition-versus-derivation distinction.

**6. Algebra of the row-Lagrangian (lines 738–753) verified by sympy.** I ran:
```
E_k, b_k, pi_k, lam = sp.symbols('E_k b_k pi_k lam', positive=True, real=True)
expr = E_k + sp.log(b_k) + 1 - sp.log(pi_k) - lam
sol = sp.solve(expr, b_k)
# Output: Stationary b_k = [pi_k*exp(-E_k + lam - 1)]
```
Normalizing over k absorbs `exp(λ − 1) = 1/Σ_m π_m exp(−E_m)`, producing line 753's `β_{ik} = π_k exp(−E_{ik}) / Σ_m π_m exp(−E_{im})`. Under uniform `π_k = 1/N`, the prior factors cancel and line 760's softmax form is exact. Sub-claims C, D, E are clean; the falsification target is sub-claim A.

**7. Minor strike — "+const" framing at line 734.** The framing `F_align = ⟨E⟩_β − H(β) + const` is exact for uniform π (where `−Σ_j β_{ij} log π_j = log N`, constant). For non-uniform π, the "const" is `⟨−log π⟩_β`, which is β-dependent. The claim restricts to uniform π so this is consistent with the headline, but it indicates that line 734 is uniform-π-specific phrasing dropped into a derivation that aims at general π up through line 753. A clean derivation would either restrict to uniform π earlier or move "const" inside the energy as a prior bias.

## Falsification conditions

This position is wrong if any of the following is exhibited:

- **(F1)** Blue exhibits the joint generative model `p(k_1, \ldots, k_N, z_1, \ldots, z_N)` over all agents' latent states *with parameters not depending on `{q_i}`*, and derives `Σ_j β_{ij} D_{KL}(q_i ‖ \Omega_{ij} q_j)` as the variational coupling produced by mean-field decomposition of its ELBO — either by pointing to specific manuscript lines that contain such a derivation, or by citing a primary source (Friston-style multi-agent FEP paper, Ramstead variational ecology, etc.) whose construction the manuscript exactly inherits.

- **(F2)** Blue shows that the standard FEP / variational Bayes derivation tolerates generative models whose parameters depend on the variational posterior, i.e., that `F[q] = E_q[log q − log p_q]` with `p_q` itself a function of `q` retains the standard ELBO bound interpretation. Primary-source citation required.

- **(F3)** Blue shows the manuscript explicitly disclaims framing (a) and adopts only framing (b) (engineered functional), in which case sub-claim A reduces to a definition, the headline must be read as "exact stationary point of the *posited* functional," and the present attack is conceded but the headline is also significantly weakened.

If none of (F1)–(F3) is produced, the headline as written fails the "every intermediate step mathematically valid" clause: step (b) at line 715 is not a derivation, and the manuscript's own line 697 acknowledges this without supplying the remedy.

## Citations

- `Attention/GL(K)_attention.tex:697` — manuscript self-disclosure of variational-quantity-dependent generative model.
- `Attention/GL(K)_attention.tex:715` — `F_align = KL[Q ‖ P]` step under attack.
- `Attention/GL(K)_attention.tex:725` — "with no approximation" claim in tension with line 697.
- `Attention/GL(K)_attention.tex:741–753` — algebra confirmed clean by sympy.
- `Attention/GL(K)_attention.tex:760, 766` — boxed softmax + τ-canonical form.
- `Attention/GL(K)_supplementary.tex:183` — supplementary §B.1 confirms canonical-F-vs-surrogate distinction; does not exhibit framing (a).
- [Friston2010 Eq. 2.2] standard variational free energy with parameter-fixed `p(o, s)`.
- `external_canon_inference.md` §1 — "Multi-agent coupling terms ... not in standard FEP ... user-introduced ... novel if claimed to follow from FEP alone."
- [Cuturi 2013 *Sinkhorn Distances* §4] — entropy-regularized soft-assignment standard Gibbs form (cited by evidence pack); confirms post-line-715 algebra.
- [Boyd & Vandenberghe *Convex Optimization* §5.5] — KKT on the simplex for the entropy-regularized objective.
- [Wainwright & Jordan 2008 §3.6] — convexity of `Σ β log β` on the open simplex (cited by evidence pack; supports uniqueness of the stationary point, which the headline does not assert but the manuscript wording implies).
- Grep of `Attention/*.tex` for `augmented joint|consensus energy|joint generative model|joint ELBO` returns line 697 only.
- sympy session above verifying the row-Lagrangian stationarity.
