# Reviewer D — GL(K)_supplementary.tex Appendix A–B (lines 1–386)

Date: 2026-05-18
Scope: lines 1–386 of `Attention/GL(K)_supplementary.tex`.
Standards used: Nakahara *Geometry, Topology and Physics* (2nd ed., 2003); Kobayashi–Nomizu *Foundations of Differential Geometry*, Vol. I; Frankel *The Geometry of Physics* (3rd ed., 2011); Amari & Nagaoka *Methods of Information Geometry* (2000); Friston 2010; Blei–Kucukelbir–McAuliffe 2017; Kingma–Welling 2014 (Appendix B); Magnus & Neudecker *Matrix Differential Calculus* (1999).

## Summary

Appendix A reviews principal-bundle / associated-bundle machinery and the gauge frame field, then derives the inter-agent transport operator $\Omega_{ij} = e^{\phi_i}e^{-\phi_j}$ and quotes the standard gauge-transformation formula for the connection. Appendix B specialises to a single-agent reduced free energy and derives the gradient with respect to $\Sigma_i$, the fixed-point equation $\Sigma_i^{-1} = \tfrac12[\Sigma_{p,i}^{-1} + \sum_j \beta_{ij}(\Omega_{ij}\Sigma_j\Omega_{ij}^\top)^{-1}]$, several limiting regimes, and a brief gradient-flow stability sketch. The geometry of A is described in general terms; the manuscript does not state whether the bundle is trivial, flat, or carries curvature, and the connection one-form chosen for the working derivation is the pure-gauge form $A^{(i)} = U_i^{-1}\partial_\mu U_i$ (curvature identically zero) — a fact the manuscript should make explicit. The covariance derivation in B is essentially correct in form but starts from a *reduced* free energy that drops the attention-entropy, hyper-prior, and model-coupling terms that the main paper and the project's canonical $\mathcal{F}$ contain; this needs to be flagged in the supplementary itself. One Hessian identity (Eq. between lines 374 and 378) is wrong as written. Several smaller correctness, sign-convention, and citation issues are flagged below.

## Major findings

### M-D-1. Pure-gauge connection presented as a general gauge-theoretic apparatus
**Claim (line 146–172):** $A^{(i)}_\mu(c) = U_i^{-1}(c) \partial_\mu U_i(c)$ is defined as the local connection one-form; field strength $F^{(i)}_{\mu\nu}$ is written down (Eq. line 153); the transformation rule $A^{(i)} = \Omega_{ij} A^{(j)} \Omega_{ij}^{-1} + \Omega_{ij}\partial_\mu \Omega_{ij}^{-1}$ is quoted.
**Claim kind:** (S) standard, but with a hidden specialization.
**Standard treatment:** A connection of the form $A = U^{-1}\,dU$ for a single $U: M \to G$ is a *pure-gauge* connection. Its curvature is identically zero: this is the Maurer–Cartan identity applied to a gauge-related $A$ (see Nakahara Ch. 10; Frankel Ch. 17). Equivalently, such a connection is locally trivializable, i.e., obtained from the trivial connection by a single gauge transformation. The manuscript states this is a "local" connection (line 144) but does not acknowledge that *each* $A^{(i)}$ is flat by construction. Whether the *collection* $\{A^{(i)}\}$ on a non-trivial principal bundle assembles into a non-flat connection depends on whether the $\phi_i$ on overlaps satisfy a cocycle condition and on the topology of $\mathcal{C}$ — none of which is discussed.
**Problem:** The manuscript writes $F^{(i)}_{\mu\nu}$ as if it were a non-trivial object, then never uses it. The connection form chosen is the one that always gives $F \equiv 0$. A reader from differential geometry will read this as either an oversight or a tacit admission that the bundle is globally trivial. The prior review of `Participatory_it_from_bit.tex` (item PH-1) raised exactly this concern; the GL(K) supplementary inherits it without addressing it.
**Required revision:** State explicitly whether the working regime is flat (the apparatus collapses to a globally trivial bundle, in which case $\Omega_{ij}$ is a transition function on $\mathcal{C}$ rather than the parallel transport of a non-flat connection) or genuinely curved (in which case the working connection cannot be the pure-gauge $U_i^{-1}dU_i$ globally — there must be a connection one-form that is *not* of this form on at least one chart, and the $\phi_i$ are sections of an Ehresmann connection rather than maps to $\mathfrak{g}$). The current presentation hovers between the two without committing.

### M-D-2. Bundle morphism vs transition function vs transport operator — undefined and used interchangeably
**Claim (lines 130–135, 156–166):** "Bundle morphisms and transport operators" are introduced as separate objects. $\Omega_{ij}(c) = e^{\phi_i(c)} e^{-\phi_j(c)}$ is defined as "the inter-agent gauge transformation" (line 158). The same $\Omega_{ij}$ is then used in Appendix B as the operator that transports beliefs (Eq. line 230).
**Claim kind:** (S) standard if presented as transition function on a globally trivial bundle; (N) novel if any other reading is intended.
**Standard treatment:** On a principal $G$-bundle, the transition functions $g_{ij}: U_i \cap U_j \to G$ relate two local trivializations and satisfy the cocycle condition $g_{ij}g_{jk} = g_{ik}$ on triple overlaps. They are not, in general, the parallel transport — parallel transport along a curve depends on the connection and on the path. On a globally trivial bundle with the trivial connection, transition functions and parallel-transport operators coincide trivially because both are determined by the global trivialization. The manuscript implicitly assumes this coincidence but does not state it.
**Problem:** Three distinct objects are conflated: (i) the bundle transition function $g_{ij}$ for a non-trivial bundle, (ii) the parallel transport map $P_\gamma$ along a curve $\gamma$, and (iii) the user's $\Omega_{ij} = e^{\phi_i}e^{-\phi_j}$ which is constructed purely from local gauge-frame fields. They are only the same thing under the globally trivial / flat / single-chart assumption. The §A.3 paragraph (lines 130–135) introduces "transport operators" without writing them down — only listing several types — and the §A.4 definition of $\Omega_{ij}$ is dropped in without saying which of the three it is.
**Required revision:** Add one paragraph stating that the framework operates in the globally trivial / flat regime (which the main paper appears to assume), and that $\Omega_{ij}$ is the *gauge-frame difference* on the trivial bundle, which in this regime coincides with both the transition function and the parallel transport. Then $\Omega_{ij}$'s satisfaction of the cocycle condition $\Omega_{ij}\Omega_{jk} = \Omega_{ik}$ should be verified — and it is *not* automatically satisfied for $e^{\phi_i}e^{-\phi_j}$ in non-abelian $G$, since $e^{\phi_i}e^{-\phi_j}\cdot e^{\phi_j}e^{-\phi_k} = e^{\phi_i}e^{-\phi_k}$ is automatic, but a triple consistency check across three frames in a non-trivial bundle would require more (see Nakahara §9.2). This needs explicit treatment.

### M-D-3. Hessian identity (between lines 374 and 378) is wrong
**Claim (Eq. just below line 374):**
$$\frac{\partial^2 D_{\mathrm{KL}}}{\partial \Sigma_1 \partial \Sigma_1} \sim \Sigma_1^{-1} \otimes \Sigma_1^{-1} + \Sigma_2^{-1} \otimes \Sigma_2^{-1}$$
followed by "manifestly positive definite".
**Claim kind:** (S) standard claim — should follow from the standard Gaussian KL.
**Standard treatment:** Starting from the manuscript's own first-derivative identity $\partial D_{\mathrm{KL}}/\partial \Sigma_1 = \tfrac12(-\Sigma_1^{-1} + \Sigma_2^{-1})$, the second derivative is taken term-by-term. The trace $\mathrm{tr}(\Sigma_2^{-1}\Sigma_1)$ is *linear* in $\Sigma_1$, so it contributes nothing to $\partial^2/\partial \Sigma_1^2$. Only the $-\log|\Sigma_1|$ piece survives. Using the matrix-calculus identity $\partial \Sigma_1^{-1}/\partial \Sigma_1 = -\Sigma_1^{-1} \boxtimes \Sigma_1^{-1}$ (in suitable Kronecker / "boxtimes" form; see Magnus & Neudecker Ch. 8 for the symmetric variant), one obtains
$$\frac{\partial^2 D_{\mathrm{KL}}(q_i\|p_i)}{\partial \Sigma_i \partial \Sigma_i} = \tfrac12 \Sigma_i^{-1} \boxtimes \Sigma_i^{-1}.$$
The $\Sigma_2^{-1} \otimes \Sigma_2^{-1}$ term in the manuscript does not arise.
**Problem:** As written, the identity is incorrect. The PD conclusion still holds (because $\Sigma_i^{-1} \boxtimes \Sigma_i^{-1}$ alone is PD on the space of symmetric matrices), but the formula must be corrected, or a careful explanation provided of what summation across the prior + alignment terms is being collapsed into a single symbolic identity.
**Required revision:** Either (a) drop the second term and rewrite the Hessian as $\tfrac12 \Sigma_i^{-1} \boxtimes \Sigma_i^{-1}$ per KL (so the full $\mathcal{F}_i$ Hessian is $(1+\sum_j\beta_{ij})\cdot\tfrac12 \Sigma_i^{-1}\boxtimes \Sigma_i^{-1} = \Sigma_i^{-1}\boxtimes\Sigma_i^{-1}$ at $\sum\beta=1$), or (b) clarify that the symbolic "$\sim$" stands for "PD bound" and write the actual bound. The PD claim is then proved at the level of the symmetric Kronecker product, citing Magnus & Neudecker. Also state the missing assumption that $\Sigma_1, \Sigma_2 \succ 0$ (line 380 mentions this only after the inequality).

### M-D-4. Appendix B derivation starts from a *reduced* free energy without saying so
**Claim (Eq. lines 186–192):** The free energy is
$$\mathcal{F}_i = D_{\mathrm{KL}}(q_i\|p_i) + \sum_{j\neq i}\beta_{ij} D_{\mathrm{KL}}(q_i\|\Omega_{ij}q_j) - \mathbb{E}_{q_i}[\log p(o_i|k_i)].$$
**Claim kind:** (S) presented as the free energy; in fact (R) reduction to single-agent / no-entropy form.
**Standard treatment:** The project's canonical free energy (per `CLAUDE.md` and the main paper's `\label{eq:free_energy_functional_final}`) contains additionally (i) a hyper-prior term $\lambda_h \mathrm{KL}(s_i\|h)$, (ii) an attention-entropy term $\tau \beta_{ij}\log(\beta_{ij}/\pi_{ij})$, (iii) a model-coupling term $\gamma_{ij} \mathrm{KL}(s_i\|\Omega_{ij}s_j)$, and (iv) its own meta-entropy. The main paper (around line 1261) reportedly distinguishes the canonical $\mathcal{F}$ from "the entropy-suppressed surrogate $\sum \beta\,\mathrm{KL}$", noting that gradients differ by $-\tau^{-1}\mathrm{Cov}_\beta(\mathrm{KL},\nabla\mathrm{KL})$. The supplementary's $\mathcal{F}_i$ is the surrogate.
**Problem:** Appendix B presents the surrogate as "the free energy" of agent $i$ without flagging that the entropy and model-channel terms have been dropped. A reader reaches the boxed gradient (line 252–267) and the fixed-point equation (line 277–286) believing these are gradients of the canonical functional, when in fact they are gradients of the surrogate. The downstream interpretive statements ("each agent's precision is the $\beta$-weighted combination of its own prior precision and the transported neighbor precisions", line 288) are correct for the surrogate but *not* for the canonical $\mathcal{F}$ — for which the entropy-derivative correction is non-vanishing at intermediate $\tau$.
**Required revision:** Open §B.1 with one sentence: "Throughout this appendix we work with the *reduced* free energy that omits the attention-entropy, hyper-prior, and model-coupling terms; the corrections from those terms are discussed in §3.5–3.6 of the main paper." Then keep the derivation. Also add at the end of §B.2: "The fixed-point equation \eqref{eq:sigma_fixed_point_beta} holds for the reduced functional; the canonical $\mathcal{F}$ adds a $\tau$-dependent attention-entropy correction that does not vanish at finite $\tau$ unless the $\partial\beta_{ij}/\partial\Sigma_i$ term in Eq. \eqref{eq:Sigma_gradient_final} is zero."

### M-D-5. The "envelope theorem" claim is asserted but unverified for the *canonical* $\mathcal{F}$
**Claim (line 269):** "This $\partial\beta/\partial\Sigma$ term arises from differentiating the attention-weighted energy $\sum_j \beta_{ij} E_{ij}$ via the product rule; it is absent from the gradient of the reduced free energy $\mathcal{F}_{\mathrm{red}} = -\tau\log Z_i + \cdots$ by the envelope theorem (see main text, Section 3.5)."
**Claim kind:** (R) reduction to the envelope form.
**Standard treatment:** The envelope theorem holds when $\beta_{ij}$ is the *unconstrained optimizer* of the inner problem $\min_\beta \sum_j \beta_{ij} E_{ij} + \tau\sum_j\beta_{ij}\log(\beta_{ij}/\pi_{ij})$ subject to $\sum_j\beta_{ij}=1$. In that case, the Lagrangian optimum $\beta^*_{ij} \propto \pi_{ij}\exp(-E_{ij}/\tau)$ satisfies the FOC, and by the envelope theorem $\partial/\partial\Sigma_i$ of $\min_\beta(\cdot)$ at $\beta = \beta^*$ equals $\partial(\cdot)/\partial\Sigma_i|_{\beta^*}$ — i.e., the $\partial\beta/\partial\Sigma$ correction vanishes. This requires that the attention-entropy term $\tau\beta\log(\beta/\pi)$ be present in the functional being optimized.
**Problem:** §B.1 already (M-D-4) drops the entropy term. So within Appendix B's stated functional, $\beta$ is *not* the optimizer of any inner Lagrangian — it is plugged in by hand as $\beta_{ij} = \mathrm{softmax}_j(-D_{\mathrm{KL}}/\tau)$, a specific assignment. The envelope theorem does not apply to the functional being differentiated in §B.1. The remark "by the envelope theorem" is therefore citing a fact about a *different* functional (the canonical $\mathcal{F}$ that includes the entropy term).
**Required revision:** Either (a) include the entropy term so the functional being differentiated is the canonical one and the envelope theorem applies, or (b) state explicitly that the $\partial\beta/\partial\Sigma$ correction is being kept in the surrogate gradient and only *vanishes* when one passes to the canonical $\mathcal{F}$ via inclusion of the entropy term. The current wording confuses these.

### M-D-6. Existence and uniqueness of the symmetric solution is not proved
**Claim (§B.2, lines 273–311):** A "symmetric solution" $\Sigma_i = \Sigma_\infty$ is exhibited under three assumptions (lines 294–298): identical agents, $\Omega_{ij}\approx I$, shared prior $\Sigma_{p,i}=\Sigma_0$. Eq. line 311 reads $\Sigma_\infty = \Sigma_0$. The fixed-point equation \eqref{eq:sigma_fixed_point_beta} is treated as having a "fixed point" without convergence analysis. The terminal sentence (line 384) says the alignment configuration "emerges from the dynamics".
**Claim kind:** (R) for the symmetric reduction; (I) for "emerges from the dynamics" without proof.
**Standard treatment:** Existence-uniqueness of a fixed point for a contraction map on the open cone of positive-definite matrices follows from either (a) a Banach fixed-point argument on a complete metric on $\mathcal{P}_K$ (e.g., the Thompson metric), (b) a monotone-operator argument on the cone (Tarski), or (c) a strict-convexity / variational argument on the free energy. None of the three is invoked. The Hessian sketch in §B.2.3 is local (positive definiteness at the equilibrium) and does not establish uniqueness — only local stability of *any* equilibrium it lands on.
**Problem:** The supplementary presents the symmetric fixed point as if it were the equilibrium, when in fact (i) symmetric ansatz $\Sigma_i=\Sigma_\infty$ is *imposed*, not derived; (ii) under the stated assumptions the fixed point trivially reduces to $\Sigma_\infty = \Sigma_0$ (an identity), which is not informative about non-symmetric configurations; (iii) the closing sentence "Σ_i ≈ Ω_ij Σ_j Ω_ij^⊤ emerges from the dynamics itself rather than as an imposed constraint" (line 384) overstates what has been shown — the manuscript has only shown that a *limit* in which Σ_i = Σ_j and Ω_ij = I makes the alignment ansatz trivially true. That is not "emergence".
**Required revision:** Either (a) prove convergence — e.g., show the map $T(\Sigma) = \tfrac12[\Sigma_p^{-1} + \sum_j\beta_{ij}(\Omega_{ij}\Sigma_j\Omega_{ij}^\top)^{-1}]^{-1}$ is a contraction in some metric on $\mathcal{P}_K^N$ (the Thompson metric on the cone or the Bures/Wasserstein metric on Gaussians are candidates) and cite the relevant theorem; or (b) downgrade the language: replace "emerges from the dynamics" with "is consistent with the linearized dynamics around the symmetric solution".

## Minor findings

### m-D-1. Bundle vocabulary defined but not used
Lines 130–135 (§A.3) list six classes of bundle morphisms/transport operators ($\Omega^{(q)}, \Omega^{(p)}, \Lambda^s_{s'}, \tilde\Lambda^s_{s'}, \Theta, \tilde\Theta, \Phi, \tilde\Phi$) but only $\Omega_{ij}$ is used in the rest of the supplement. Recommend either removing the unused ones or deferring their introduction to where they are first used. The cross-scale operators in particular are tied to the meta-agent discussion (§A.2.1) and the RG conjecture in Appendix F (out of scope here) — locate them there.

### m-D-2. Symbol overload: $\sigma$ used for both "section of a bundle" and "standard deviation"
Line 66 defines an agent as $\mathcal{A}^i = (\sigma^i_q, \sigma^i_p)$ — bundle sections. Elsewhere in the project (and presumably later in the supplement / main paper), $\sigma$ denotes Gaussian standard deviation. The supplement's $\sigma$-as-section never resurfaces in §B (which uses $\Sigma$ as full covariance), so there is no direct conflict on the visible pages, but the global manuscript uses both. Recommend an alternative for the sections (e.g., $s^i_q, s^i_p$ — though this collides with the $s$ used for "model" in the canonical $\mathcal{F}$). The lazy fix is to write "section" out as $\sigma^{\text{sec}}$. The user should pick a discipline-wide convention.

### m-D-3. The free energy of §B.1 has unit prior coefficient; the main paper's promoted form has $\alpha_i$
Line 271 says "$\alpha_i = 1$" without earlier introducing $\alpha_i$. The main paper (Section 3.6 per the cross-reference at line 333) promotes a per-agent state-dependent $\alpha_i$ via a log-barrier regularizer. The supplement should either (a) state up front that the derivation specializes to $\alpha = 1$, or (b) carry $\alpha_i$ through. The "$-2$" coefficient on $\Sigma_i^{-1}$ (line 271) is correct only at $\alpha_i = 1$; at general $\alpha_i$ it becomes $-(\alpha_i + \sum_j\beta_{ij}) = -(\alpha_i + 1)$. The text at line 333 already notes this — but the boxed Eq. \eqref{eq:Sigma_gradient_final} is written at $\alpha_i=1$ without disclaimer.

### m-D-4. Eq. line 165: $\rho$ undefined at point of use
The action $q_j \mapsto \Omega_{ij} q_j := \rho(\Omega_{ij}) q_j$ uses an unspecified representation $\rho$. §A.1 (line 53) defined $\rho_q$ and $\rho_p$ separately. The supplement should be clear which $\rho$ is meant in §A.4 — likely $\rho_q$ for belief transport. Recommend $\rho_q(\Omega_{ij})$ explicitly.

### m-D-5. The cocycle condition on $\Omega_{ij}$ is needed but not stated
For $\Omega_{ij}$ to be the transition function of a principal bundle, the cocycle condition $\Omega_{ij}\Omega_{jk} = \Omega_{ik}$ must hold on triple overlaps (Nakahara §9.2; KN Vol. I §I.5). With $\Omega_{ij} = e^{\phi_i}e^{-\phi_j}$ this is automatic by direct cancellation:
$$e^{\phi_i}e^{-\phi_j} \cdot e^{\phi_j}e^{-\phi_k} = e^{\phi_i}e^{-\phi_k}.$$
This should be stated as a virtue of the parameterization (and is a non-trivial special-case reason the construction is consistent on a globally trivial bundle).

### m-D-6. Notation $\Omega_{ij}q_j$ for transporting a distribution should be explained
At line 230 the supp writes $\Omega_{ij} q_j = \mathcal{N}(\Omega_{ij}\mu_j, \Omega_{ij}\Sigma_j\Omega_{ij}^\top)$. This is correct under the pushforward of a Gaussian by a linear map (Nakahara §5; Amari & Nagaoka), but the supplement assumes this without statement. Recommend one sentence: "for Gaussian $q_j = \mathcal{N}(\mu_j,\Sigma_j)$, the pushforward by the linear map $\Omega_{ij}$ is $\mathcal{N}(\Omega_{ij}\mu_j, \Omega_{ij}\Sigma_j\Omega_{ij}^\top)$." This is also the place to flag the sandwich rule for the covariance: cite Nakahara on the transformation of (2,0) or (0,2) tensors under $\Omega \in \mathrm{GL}(K)$.

### m-D-7. The "symmetrization step" expected in a Σ-derivative is not discussed
Standard matrix calculus on symmetric matrices distinguishes the *unconstrained* derivative (which is symmetric in this case anyway because both $\Sigma_i^{-1}$ and $\Sigma_2^{-1}$ are symmetric) from the *symmetric-tangent* derivative (which equals $\tfrac12(\partial+\partial^\top)$ of the unconstrained derivative). For the KL formulas here, the two coincide because the gradient is already symmetric. Recommend a footnote stating this — it removes ambiguity for readers expecting a Magnus–Neudecker $\mathrm{sym}(\cdot)$ projector.

### m-D-8. Sign convention: forward vs reverse KL is not stated up front
Appendix B uses $D_{\mathrm{KL}}(q_i\|p_i)$ — *forward* / "inclusive" KL with $q$ as the recognition distribution. This matches the standard VI / ELBO convention (Blei et al. 2017; Friston 2010 Form 3). Recommend stating this convention explicitly at the start of §B.1 along with the citation, because some FEP-flavored work uses the reverse direction.

### m-D-9. Eq. line 153 names $F^{(i)}_{\mu\nu}$ without using it again
The curvature 2-form is defined and then never referenced. If the manuscript does not use it, drop the equation — keep §A.4 lean. If it is used elsewhere (Appendix F, RG?), defer the definition to where it bites.

### m-D-10. "Free attention" vs "soft-attention" temperature parameter $\tau$ also overloaded
Symbol $\tau$ appears in §B.2 as the softmax temperature in $\beta_{ij}$ (line 319). The project elsewhere uses $\tau = \kappa\sqrt{K}$. Whether $\tau$ in §B.2 is the bare $\tau$ or the effective $\kappa\sqrt{K}$ is unstated. Clarify.

### m-D-11. Eq. 100–102: "Belief consensus" $q_i = \Omega_{ij}q_j$ for *meta-agent* membership is stronger than the alignment condition discussed in §B
§A.2's meta-agent definition demands *exact* gauge-related equality, while §B's "alignment" only requires approximate (modulo the dynamics). These are different conditions. The supplement should distinguish "consensus" (exact, defining meta-agents) from "alignment" (approximate, dynamically reached).

## Equation verification log

| Eq. (line) | Manuscript form | Verified against | Verdict |
|---|---|---|---|
| Line 200–217 (Gaussian KL) | Standard closed form, $-d$ at end | Blei et al. 2017; KingmaWelling2014 App. B | Correct |
| Line 222–228 ($\partial D_{\mathrm{KL}}/\partial \Sigma_1$) | $\tfrac12[-\Sigma_1^{-1}+\Sigma_2^{-1}]$ | Magnus–Neudecker; direct computation | Correct |
| Line 230 (Gaussian under linear pushforward) | $\Omega q = \mathcal{N}(\Omega\mu,\Omega\Sigma\Omega^\top)$ | Standard Gaussian transformation | Correct; should be stated as such |
| Eq. \eqref{eq:Sigma_gradient_final} (lines 252–267) | Boxed gradient at $\alpha=1$ | Term-by-term verification | Correct at $\alpha=1$; missing $\alpha$ disclaimer (m-D-3) |
| Eq. \eqref{eq:sigma_fixed_point_beta} (lines 277–286) | Fixed-point equation in precisions | Follows from setting boxed gradient to zero and assuming $\sum_j\partial\beta_{ij}/\partial\Sigma_i = 0$ | Correct *within* the surrogate; the canonical $\mathcal{F}$ adds a non-vanishing correction (M-D-4) |
| Eq. line 311 ($\Sigma_\infty = \Sigma_0$) | Symmetric reduction | Follows from $\Sigma^{-1}=\tfrac12(\Sigma_0^{-1}+\Sigma^{-1})$ | Trivially correct; informativeness is M-D-6 |
| Eq. line 320–330 (softmax form of $\beta$) | Standard | Standard softmax | Correct |
| Eq. \eqref{eq:beta_weighted_precision} (lines 336–344) | $\Sigma_i^{-1} \approx \langle (\Omega\Sigma_j\Omega^\top)^{-1}\rangle_\beta$ | Follows from setting $\alpha_i\ll1$ and $\sum\partial\beta=0$ | Correct under stated regime |
| Eq. line 354–356 ($\Sigma_i\approx\Omega_{ij}\Sigma_j\Omega_{ij}^\top$) | Derived from previous | Direct | Correct as an *implication of three assumptions* in the limit; not "emergence" (M-D-6) |
| Eq. line 365–369 (gradient flow $d\Sigma/dt = -\eta\partial\mathcal{F}/\partial\Sigma$) | Definition | Standard | Correct; Euclidean gradient flow on $\mathrm{P}_K$ — not natural — should be flagged that on $\mathrm{P}_K$ one would normally use the affine-invariant / Bures metric for stability analysis [AmariNagaoka2000] |
| Eq. between lines 374 and 378 (Hessian) | $\Sigma_1^{-1}\otimes\Sigma_1^{-1} + \Sigma_2^{-1}\otimes\Sigma_2^{-1}$ | Direct second derivative | **Wrong** (M-D-3) — the $\Sigma_2$ term does not arise |

## Style scan

- Banned spacing macros `\;` and `\,`: searched lines 1–386, found **none**. The supplement obeys the style rule on this slice.
- Banned phrases: searched for `key insight`, `crucially`, `critically` (as sentence opener), `notably`, `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores`. Of these, "in particular" — flagged as banned — is not present in lines 1–386. None of the other banned phrases appear in the scoped lines.
- Horizontal rules `---`: none in scope.
- Equation punctuation: most display equations end with comma/period as required. Some are missing terminal punctuation (e.g., Eq. line 311 ends with `.` good; Eq. line 165 ends `\rho(\Omega_{ij}(c)) q_j(c).` good; Eq. line 142 ends `\text{Lie}(G),` with comma — OK because the sentence continues). One minor cleanup: Eq. line 158–160 ($\Omega_{ij}(c) = \exp[\phi_i(c)] \exp[-\phi_j(c)] \in G$) ends without punctuation, then the next sentence begins "transports agent $j$..." — add a comma.
- Itemizations: §B.2 list (i),(ii),(iii) at lines 294–298 is a list inside what could be a paragraph. Per project style "Minimize itemizations" — fold into prose.
- Self-referential drafting language: none found in scope.

## Citations checked

| Citation | In `references.bib`? | Relevance to claim | Verdict |
|---|---|---|---|
| `nakahara2003geometry` | yes (line 244) | Differential geometry / bundles. Appropriate. | [✓] presence; full content not verified |
| `frankel2011geometry` | yes (line 2285) | Same. Appropriate. | [✓] presence |
| `blei2017variational` | yes (line 2217) | VI. Appropriate for §A.1's mention of statistical manifolds. | [✓] presence |
| `amari2016information` | yes (line 2188) | Info geometry. Appropriate. | [✓] presence |
| `shen2008coarse` | yes (line 2379) | Coarse-graining. Cited at line 97 for "coarse-graining" — fine. | [✓] presence; coarse-graining is general enough |
| `anderson1984basic`, `wilson1974renormalization`, `garciaMillan2024network` | yes | RG analogy. Cited at line 126. Appropriate as RG references. | [✓] presence; the RG *analogy* itself is presented as analogy, no formal claim |

Citations missing that should be present:

- **Kobayashi–Nomizu** — for the principal bundle / connection / curvature definitions in §A.1, §A.3, §A.4. The supplement cites only Nakahara and Frankel; for the foundational definitions and the gauge-transformation identity for $A$, KN Vol. I is the canonical reference.
- **Magnus–Neudecker** *Matrix Differential Calculus* (1999) — for the matrix derivatives in §B.1. Currently uncited.
- **Bleecker** *Gauge Theory and Variational Principles* (1981) — would be appropriate for connecting the gauge-theoretic apparatus to variational dynamics.
- **Friston 2010** and **Blei et al. 2017** are not cited *in §B* even though the free energy and KL are central. The supp cites them in §A.1 (line 49) but should re-cite at the start of §B.1 where the free-energy form is asserted.
- **A Banach / contraction / Bures-metric reference** (e.g., Bhatia *Positive Definite Matrices*, 2007; Nielsen 2020 on Bures geometry) — should be cited if §B.2.3 makes any stability claim beyond a one-line Hessian sketch (M-D-6).

Citations not retrievable / not verified (full text):

- Nakahara 2003 (textbook, not online); Frankel 2011 (textbook, not online); KN Vol. I (textbook, not online). [?] for content; reviewer relies on standard knowledge of those works.

## Code cross-references checked

Appendix A–B is mathematical / theoretical; little of it has a direct code counterpart. The closest binds:

- $\Omega_{ij} = e^{\phi_i}e^{-\phi_j}$ (line 159) is the user's transport operator. Per `codebase_map.md` the user implements this in the gauge-transport modules. The supplement matches the implementation as stated in `CLAUDE.md`. ✓
- $\Sigma$-transport sandwich $\Omega \Sigma \Omega^\top$ (line 230, line 245, line 283, line 340–342) matches the project's `Sigma_transported = Omega @ Sigma @ Omega.T` rule. ✓
- The fixed-point equation \eqref{eq:sigma_fixed_point_beta} is a theoretical statement; the project's E-step iteratively updates $\Sigma$ via a retraction and does *not* solve the fixed-point in closed form. The supplement's equation is therefore a theoretical equilibrium claim, not a description of the algorithm. This is fine — but the supp could note it.
- Eq. line 365 ($d\Sigma/dt = -\eta\partial\mathcal{F}/\partial\Sigma$) is Euclidean gradient flow. The project's E-step uses an *affine-invariant* / SPD-manifold retraction in $\sigma$-space (per `CLAUDE.md`'s description of `retract_spd_diagonal_torch`). The supp's choice of Euclidean flow is therefore *not* the project's chosen geometry. Recommend either changing the supp to natural / affine-invariant gradient flow, or adding a footnote that the equilibrium analysis here uses Euclidean flow purely for the local stability sketch and the implementation uses an SPD-manifold retraction.

Nothing in Appendix A–B requires editing the codebase. The flag is purely about supplement–code consistency in stated dynamics (m-D-7's neighborhood).

## Verdict

Major revisions required, of a *clarification* rather than *correction* character with one exception (M-D-3, the Hessian, must be corrected). The mathematics in Appendix B is mostly right *for the reduced functional it works with*, but Appendix B does not say that is the functional it is working with. Appendix A defines a connection that is identically flat and writes down a curvature that is identically zero, without saying so — a long-standing ambiguity inherited from the larger manuscript. The single-agent symmetric-solution claim in §B.2 is presented as "emergence from dynamics" but is in fact a tautology under three assumptions. With the revisions above the appendix becomes a clean exposition of the reduced-functional covariance fixed-point — which is what it should be.
