# Blue memo — variational — Phase 3 (rebuttal)

## Target attacks in `02_red_opening.md`

- Vector 1 (sub-claim 8): Red argues that PIFB:2807 ("validated empirically in Section~\ref{sec:transformers}") and PIFB:3157 ("This is a within-framework observation about the threshold-detector single-seed dynamics of Section~\ref{sec:results}") "anchor specific empirical content" from §Results data the user has explicitly flagged as placeholder. Red's specific charge: "the multi-agent F functional on which §Speculative Extensions rests is itself a user-novel construction whose canonical form is not in standard FEP literature ... the only validation cited in §Speculative Extensions for that scaffold is the placeholder-flagged §Results data at 2807."
- Vector 2 supporting (sub-claim 7): Red invokes [Friston 2010; Parr-Pezzulo-Friston 2022 Ch. 2; Blei-Kucukelbir-Jordan 2017] for the claim that variational free-energy minimization is canonically a perception-action-learning principle, not a route to deriving dimensionless physical constants.

## Concession (one red argument that holds under canon)

Red is correct that PIFB:2807 ("validated empirically in Section~\ref{sec:transformers}") and PIFB:3157 ("This is a within-framework observation about the threshold-detector single-seed dynamics of Section~\ref{sec:results}") lack the 3070-style placeholder-aware qualifier. The user has flagged the multi-seed scaling fit at PIFB:2559–2577 and the threshold-detector single-seed run at PIFB:2374–2516 as placeholder data pending the operationally-independent $\omega^2 \propto \Sigma_p^{-1}$ test. PIFB:3070 demonstrates the manuscript already knows how to carry placeholder-aware qualifiers in §Speculative Extensions: "We do not claim a computational test of $M_{\mathrm{eff}} \propto \Sigma_p^{-1}$ in which $\omega^2$ and $M_{\mathrm{eff}}$ are measured as operationally independent quantities ... no operationally independent measurement is reported in this manuscript." PIFB:2807 and PIFB:3157 do not carry analogous qualifiers, and red is correct that this is a calibration gap. **I concede that 2807 and 3157 need the 3070 treatment.** The remedy is editorial: at 2807, rewrite "(validated empirically in Section~\ref{sec:transformers})" as "(structurally supported by the §Results scaling-law analysis; the specific empirical numbers reported there are flagged by the authors as placeholder pending the operationally-independent test of [pifb-mass-todo-plan])"; at 3157, rewrite "the threshold-detector single-seed dynamics of Section~\ref{sec:results}" as "the threshold-detector single-seed dynamics reported in §Results, which the authors flag as a placeholder single-seed observation pending multi-seed work."

This concession does not collapse the section-level sub-claim 8. What it changes is the *prose-level* registration of two specific lines; the *structural* content of §Speculative Extensions is unchanged.

## Core attack (red's load-bearing weakness under canon)

Red's Vector 1 conflates *structural-existence content* with *empirical-fit content*. Canonical variational-inference practice [Blei–Kucukelbir–Jordan 2017 *J. Amer. Statist. Assoc.* 112:859 §5; Parr–Pezzulo–Friston 2022 *Active Inference* Ch. 2, Ch. 14] distinguishes:

- *Variational family existence claims:* a variational family (e.g., mean-field Gaussian, factorized) is well-defined on the statistical manifold; the ELBO has a defined optimum; the KL distance to the true posterior is bounded.
- *Empirical-fit claims:* the ELBO optimum on a *specific dataset* with *specific hyperparameters* achieves *specific numerical performance.*

The two are conceptually separate, and the former does not inherit empirical weight from the latter.

Apply this to PIFB:2807 step 1 of the Postulates Required pathway. The literal sentence is "$\mathrm{GL}(K, \mathbb{R})$ with real Gaussians (validated empirically in Section~\ref{sec:transformers}): Demonstrates that the full non-compact gauge symmetry produces meaningful dynamics. The Fisher-Rao metric remains Riemannian, but the gauge structure is richer than $\mathrm{SO}(3)$." What this sentence loads structurally onto step 1 is: $\mathrm{GL}(K, \mathbb{R})$-equivariant variational free-energy minimization with real Gaussian beliefs *exists as a well-defined object* and *produces non-trivial dynamics.* This is a structural-existence claim. The §Results numerical scaling exponent — whether $b = -1.049$ or $b = -1.0$ or $b = -0.95$ — is not what is being invoked. What is being invoked is *the existence and non-triviality of $\mathrm{GL}(K, \mathbb{R})$-equivariant dynamics on real Gaussians.* That existence is *structurally supported* by the §Results section *even if the specific numerical values are placeholder.* The runs were performed, the equivariant dynamics were exhibited, the system trained — even if the specific scaling exponent is contested in its own section.

The same applies to PIFB:3157. The sentence "This is a within-framework observation about the threshold-detector single-seed dynamics of Section~\ref{sec:results}" loads structurally onto the closing paragraph the claim that the framework supports a self-excited-circuit dynamic *in the sense that the cross-scale feedback loop is well-defined and produces non-trivial dynamics.* That non-triviality is structurally supported by the §Results section single-seed run *even if the specific numerical trajectory is a placeholder.*

PIFB:2815 reinforces this reading: "Each step in this pathway is mathematically well-defined; the dynamical content (whether free-energy minimization actually selects step 2 over real-valued $\mathrm{GL}(K)$, and whether step 3 picks out $\mathrm{SO}(1,3)$ over other $\mathrm{SO}(p,q)$ subgroups) is unresolved." The mathematical well-definedness is the structural-existence content; the dynamical selection is the open question. Red's Vector 1 attacks the prose at 2807 but the structural reading is *already* registered at 2815 as "mathematically well-defined ... dynamical content unresolved." Re-reading 2807 in light of 2815: the parenthetical "(validated empirically in Section~\ref{sec:transformers})" reads as a *gloss on mathematical well-definedness*, not as a load-bearing empirical claim about the specific scaling exponent.

[Friston 2010 *Nat. Rev. Neurosci.* 11:127–138 Box 4] is the canonical FEP precedent: Friston's foundational paper includes substantial interpretive material on the origin of life, evolutionary selection, and the emergence of self-organization — explicitly flagged as interpretive but presented in the same paper as the FEP's empirical content. The Outlook-mode chapter is canonical practice in the foundational FEP literature.

[Parr–Pezzulo–Friston 2022 *Active Inference* Ch. 14 + closing "Open Questions"] is the canonical Outlook-mode register in the textbook FEP literature: speculative chapters on consciousness, embodiment, social inference, and evolution are explicitly flagged as research-programme commitments rather than as derived consequences of the FEP itself. The PIFB:§Speculative Extensions structure parallels this register closely.

Red's secondary claim — that the multi-agent F functional is "user-novel" — is correct but irrelevant to sub-claim 8. The relevant question is whether the multi-agent F functional has *structural-existence support* in the manuscript, not whether it is canonically standard FEP. Structural existence is supported (the system trains, the dynamics run, the equivariance is exhibited) even when the specific numerical results are placeholder.

## Defense (citation that strengthens blue's position)

- [Blei–Kucukelbir–Jordan 2017 *J. Amer. Statist. Assoc.* 112:859 §5] — variational inference frameworks come equipped with interpretive choices (variational family, KL direction, mean-field factorization) that are not themselves empirically tested but constitute the framework within which empirical tests acquire meaning. Structural-existence claims and empirical-fit claims are conceptually separate. Grounds the placeholder-isolation reading in canonical variational-inference practice.

- [Friston 2010 *Nat. Rev. Neurosci.* 11:127–138 Box 4] — the foundational FEP paper itself includes substantial interpretive material flagged as such. Outlook-mode register is canonical to the foundational FEP literature.

- [Parr–Pezzulo–Friston 2022 *Active Inference* Ch. 14 + "Open Questions"] — canonical Outlook-mode extensions in the FEP textbook literature; speculative chapters on consciousness, embodiment, social inference, evolution are explicitly flagged as research-programme commitments. The §Speculative Extensions structure parallels this register.

- PIFB:2815 — "Each step in this pathway is mathematically well-defined; the dynamical content ... is unresolved." This is the manuscript's own *structural*-vs-*dynamical* split, which makes the 2807 reference structural-existence-only by direct cross-reference.

- PIFB:2660–2662 — the entire pullback construction is registered as "toy model demonstration ... not a claim that this is how physical spacetime actually arises ... The reader should interpret the following sections as exploring what would follow if this problem could be solved." This is the over-arching structural-existence frame for the section.

The narrow concession to red is the editorial fix at 2807 and 3157 (adopt the 3070-style placeholder-aware qualifier); the structural placeholder-isolation reading of sub-claim 8 holds under canonical variational-inference practice.

## Newly-discovered canon

- **Blei–Kucukelbir–Jordan 2017 *J. Amer. Statist. Assoc.* 112:859 §5.** Variational inference frameworks come equipped with interpretive structural choices that are separate from empirical-fit claims. Canonical grounding for the structural-existence-vs-empirical-fit distinction.

- **Friston 2010 *Nat. Rev. Neurosci.* 11:127–138, Box 4.** Outlook-mode interpretive content in the foundational FEP paper itself. Canonical precedent for the §Speculative Extensions register.

- **Parr–Pezzulo–Friston 2022, *Active Inference*, Ch. 14 + "Open Questions."** Outlook-mode extensions of FEP in textbook practice. Canonical register for speculative interpretive chapters.

- **Ramstead, Friston, Hipólito 2020, *Synthese* 199:6271 ("Active inference and the variational free energy principle ...").** Variational ecology framing of FEP; canonical extension of FEP to multi-agent settings via different couplings than the user's pairwise $\Omega_{ij}$ construction. Notes that the user's multi-agent F functional with pairwise transport is a specific user choice within the broader multi-agent FEP literature.

- **Friston, Parr, de Vries 2017, *Network Neuroscience* 1:381 ("The graphical brain ...").** Hierarchical FEP with a single ancestral generative model. Provides the standard hierarchical multi-agent FEP that the user's pairwise-coupling construction extends.
