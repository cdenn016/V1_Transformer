# Claim — cross-scale-shadow-legitimacy

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (focus: `Attention/Participatory_it_from_bit.tex` §sec:cross_scale_shadows lines 536–548 and §sec:agent_definition lines 612–631; Friston 2017; Parr–Pezzulo–Friston 2022)
**Canon location:** `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge\`

## Claim

The cross-scale shadow construction posited at Eq.~\ref{eq:cross_scale_shadow} of `Attention/Participatory_it_from_bit.tex` (line 540 onward),

> $p_i^{(s)}(c) = \Omega_{i,I}\big[q_I^{(s+1)}\big](c)$,
> $r_i^{(s)}(c) = \tilde\Omega_{i,I}\big[s_I^{(s+1)}\big](c)$,

which defines the scale-$s$ prior $p_i$ and hyper-prior $r_i$ as the gauge-transport into agent $i$'s frame of the level-$(s{+}1)$ meta-agent's posterior $q_I^{(s+1)}$ and generative model $s_I^{(s+1)}$, is a *legitimate refinement* of the standard hierarchical variational-inference / hierarchical active-inference scheme as developed in Friston 2010, Friston 2017, and Parr–Pezzulo–Friston 2022 — rather than an undeclared substitution of the level-$\ell$ prior $p(s_\ell | s_{\ell+1})$ in the standard generative model with the level-$(\ell{+}1)$ posterior $q(s_{\ell+1})$.

## User context

- The manuscript itself flags this distinction at line 546: *"This is a structural commitment of the framework rather than a theorem of standard hierarchical variational inference: in the standard scheme [Friston2017, parr2022active] the level-$\ell$ prior is derived from a generative-model conditional $p(s_\ell \mid s_{\ell+1})$, not posited as a transported posterior, and we do not display the reduction (or approximation) of the standard hierarchical scheme to the present cross-scale shadow construction."*
- The user previously directed: **"teams should consult the literature as source of truth."** This directive carries over.
- The previous debate on the omnibus Theory section purity (`2026-05-19-pifb-theory-section-purity`) noted this as the third recommended follow-up sub-debate (`05_action.md` §Follow-up debates item 3).

## Load-bearing tension

The cross-scale shadow turns a generative-model conditional (the standard hierarchical-VI / hierarchical-AIF object) into a transported posterior. Two readings are available:

- **Blue reading (legitimate refinement):** The transported-posterior construction is an Empirical-Bayes-style instantiation of the hierarchical prior in which the level-$(\ell{+}1)$ posterior plays the role of the conjugate-family hyperprior; the gauge transport $\Omega_{i,I}$ is the frame-change machinery required when constituents and the meta-agent maintain distinct frames. This is mathematically related to mean-field-with-message-passing and to amortized hierarchical VI (Kingma–Welling 2014; Rezende–Mohamed 2015; Sønderby et al. 2016) and the framework is honest that it is a structural commitment, not a derivation.

- **Red reading (undeclared substitution):** Replacing $p(s_\ell | s_{\ell+1})$ with $\Omega_{i,I}[q_I^{(s+1)}]$ is not a refinement of hierarchical VI — it changes the generative model. The "ELBO" the manuscript later writes against this construction is therefore an ELBO for a different probabilistic model than the one Friston 2017 / Parr–Pezzulo–Friston 2022 work with. Without an explicit reduction or approximation chain (which the manuscript admits it does not display), this is not a refinement; it is a separate framework that uses overlapping vocabulary.

The judge must decide whether the construction is in continuity with the canonical hierarchical-AIF literature or whether it is a model substitution wearing the same notation.

## Adjudication note

Per the user's directive, the judge weighs by external literature (Friston 2010, Friston 2017, Parr–Pezzulo–Friston 2022, Kingma–Welling 2014, Rezende–Mohamed 2015, Sønderby et al. 2016, Hinton 2012 / Kingma 2014 Empirical-Bayes precedents, Robbins 1956 on empirical Bayes generally) and not by the project's own canon files.
