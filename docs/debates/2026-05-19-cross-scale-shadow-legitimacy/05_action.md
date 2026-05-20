# Action — cross-scale-shadow-legitimacy

**From verdict:** RED_WINS.

The cross-scale shadow construction $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ at Eq.~\ref{eq:cross_scale_shadow} is a **declared structural commitment** (per manuscript line 546) but not a *refinement* of standard hierarchical VI in the sense of Friston 2017 / Parr–Pezzulo–Friston 2022 Ch. 8. The verdict treats it as a *separate framework* in partial continuity with several literature families (empirical-Bayes, ladder VAE, message-passing, log-linear pooling, tempered Bayes) at downstream points, but distinct from each at the upstream substitution step.

## Recommended manuscript edits

Three concrete edits to `Attention/Participatory_it_from_bit.tex` capture the verdict without retracting any substantive theoretical content. None of the edits alters the construction itself — only its framing.

### Edit 1 — line 546 (cross-scale shadow declaration)

Current text (line 546 sentence two):
> "in the standard scheme [Friston2017, parr2022active] the level-$\ell$ prior is derived from a generative-model conditional $p(s_\ell \mid s_{\ell+1})$, not posited as a transported posterior, and we do not display the reduction (or approximation) of the standard hierarchical scheme to the present cross-scale shadow construction."

Recommended addition (immediately after the existing sentence):
> The standard scheme's $p(s_\ell \mid s_{\ell+1})$ is a conditional density of a fixed generative joint, while $\Omega_{i,I}[q_I^{(s+1)}]$ is a transported variational posterior of a coupled inferential subsystem. The two are mathematically distinct object types even after gauge transport; we therefore present the present construction as a separate declared framework, not as a refinement of the standard hierarchical-VI / hierarchical-AIF scheme. The continuity with the FEP literature is at the variational-principle backbone (the ELBO, complexity/accuracy decomposition, natural-gradient flow), not at the hierarchical generative-model structure.

### Edit 2 — line 1023 (FEP citation scope)

Current text (line 1023):
> "We retain the FEP citations as the conceptual backbone (variational inference, the ELBO bound, the complexity/accuracy decomposition) but the multi-agent functional itself is an engineered consensus energy, not a derivation from FEP alone."

Recommended addition (appended to the existing sentence):
> The same scope applies to the cross-scale shadow construction of Section~\ref{sec:cross_scale_shadows}: the ELBO written against $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ is the ELBO of a coupled variational fixed-point system rather than of any hierarchical generative model in the [Friston2017] / [parr2022active Ch.~8] sense. The FEP citations remain valid as the variational-principle backbone but do not legitimize the upstream prior substitution.

### Edit 3 — line 2216 (Ouroboros four-anchor scope)

Current text (line 2216, end of paragraph):
> "the strictly-normalized special case $\lambda_0 = 1 - \rho$ enforces $\sum_{k \ge 0}\lambda_k = 1$ and recovers the externally-Bayesian log-linear pool of~[GenestZidek1986]; the unrestricted default $\lambda_0, \rho > 0$ used in this framework is a tempered free energy in the generalized-Bayesian sense of~[BissiriHolmesWalker2016], which permits the overall strength of ancestral coupling to be scaled independently of the decay rate."

Recommended addition (appended to the paragraph):
> The four anchor identifications — [WestHarrison1997] dynamic-discount, [Hinton 2002] product-of-experts, [GenestZidek1986] log-linear pool, [BissiriHolmesWalker2016] tempered Bayes — legitimize the additive-discounted-KL *pooling* form once the per-generation shadow distributions $p_k = \Omega_{i,I_k}[q_{I_k}^{(s+k)}]$ are accepted as inputs. The shadow substitution itself remains the separate structural commitment of Section~\ref{sec:cross_scale_shadows}; the pooling identification covers the downstream aggregation, not the upstream substitution.

## Open obligation if user wishes to flip to BLUE

If the user wants this debate's outcome reversed in a future revision, the obligation that follows from the verdict is to provide *one* of:

a) **Reduction or approximation chain.** A derivation from the standard hierarchical conditional $p(s_\ell \mid s_{\ell+1})$ to $\Omega_{i,I}[q_I^{(s+1)}]$ under stated limits, citing a primary source. The most plausible candidates are (i) a degenerate-mixing-distribution limit of a parameterized Sønderby-style prior, (ii) a tight-Gaussian-approximation limit of a variational EM M-step that places the M-step estimator into the prior slot, or (iii) a non-parametric Robbins-empirical-Bayes argument made explicit for coupled subsystems.

b) **Primary-source treatment of coupled-subsystem variational message passing.** A literature passage that establishes the implicit joint of a coupled variational fixed-point system (priors equated to posteriors of adjacent subsystems) as a well-defined probability distribution. Wainwright & Jordan 2008 §3.4 was the closest blue could approach in this debate but explicitly handles fixed factor graphs only. A primary source that *does* handle the coupled-subsystem case would flip C5 and is the single most load-bearing literature gap.

Either (a) or (b) would convert the per-sub-claim map and the omnibus verdict.

## Per-sub-claim verdict map (for the user's reference)

| Sub-claim | Topic | Outcome | Decisive citation |
|---|---|---|---|
| C1 | Friston 2017 hierarchical conditional vs transported posterior | RED | Friston 2017 §4 hierarchical generative density form |
| C2 | Empirical Bayes (parametric or non-parametric) as precedent | RED | Robbins 1956 / Carlin–Louis §5.4–5.5; EB estimators are data-marginal-driven, not coupled-subsystem-driven |
| C3 | Ladder VAE (Sønderby et al. 2016 §3) analogue | RED | Type-signature distinction: parameterized conditional density vs transported variational posterior |
| C4 | Manuscript line 546 admission as honest scholarship | BLUE on the "declared vs undeclared" axis; does not carry "refinement" |
| C5 | Well-defined joint distribution via message passing | RED (blue concession at `03_blue_rebuttal.md` lines 28–30; Wainwright–Jordan §3.4 unrebutted) |
| C6 | Ouroboros four-anchor identification at line 2216 | RED on substitution; BLUE on pooling form (blue concession at `03_blue_rebuttal.md` lines 26–27) |
| Omnibus | Cross-scale shadow is a refinement, not an undeclared substitution | **RED** — declared substitution, not refinement |

## Status

This debate is closed. The recommended three-edit fix path converts the manuscript framing to the declared-separate-framework reading that is fully defensible under the literature-as-source-of-truth standard.
