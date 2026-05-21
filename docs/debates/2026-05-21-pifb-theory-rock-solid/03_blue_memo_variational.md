# Blue Memo — Variational (Phase 3 Rebuttal)

## (i) Concession from red's opening

Red's invocation of Ranganath-Tran-Blei 2016 [arXiv:1511.02386, "Hierarchical Variational Models"] and Sønderby 2016 [arXiv:1602.02282, "Ladder Variational Autoencoders"] as canonical hierarchical-VI precedents is correctly placed. Standard hierarchical VI builds a hierarchical $q$ either by placing a prior on variational parameters (Ranganath-Tran-Blei) or by passing a finite-variance Gaussian through the top-down generative branch with non-degenerate covariances at every level (Sønderby ladder VAE). The manuscript's cross-scale shadow at line 552 — $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ as a deterministic transported posterior in the $\sigma^2 \to 0$ rigid-link limit — does not match either precedent in its exact form.

I concede: the cross-scale shadow construction is not literally Sønderby/Ranganath-Tran-Blei. The manuscript at line 552 acknowledges this: "This relation is a structural commitment of the framework rather than a theorem of standard hierarchical variational inference."

## (ii) Strongest defense against red's core attack

The defense is on **per-citation reclassification of red's 14-count claim against the falsification-condition (ii) threshold of ≥10 "load-bearing reduction steps."** Red's enumeration at lines 636, 943, 1042, 1209, 1294, 1352, 1365, 1607, 1615, 1623, 1676, 1702, 1818, 1875 is verified at the line numbers, but the *load-bearing* characterization varies dramatically across the 14:

**Strictly load-bearing delegations** (the derivation is *not* in §Theory; the companion supplies it):
- Line 1209: cocycle proof "follows the companion paper" — the proof is delegated.
- Line 1607 + 1615: transformer-limit reduction "remainder follows the development in" — the trivial-frame route is delegated.
- Line 1818: multi-head block-diagonal gauge group identification ("therefore the block-diagonal subgroup… [Dennis2025trans]") — the lift is delegated.
- Line 1875: multi-head, RoPE, FFN derivations "derived in detail in" — three extensions delegated.

Strict count: **5 load-bearing delegations, not 14.** Red's bar in falsification (ii) is ≥10. The bar is not crossed under a strict reading.

**Provenance / adoption description / "see also" — derivation is in §Theory:**
- Line 636: convention follow ("we follow the convention"). The convention is *stated in §Theory*; the companion is credited as the locus the convention was previously published.
- Line 943: adaptation of standard Amari linear-pushforward theorem to local notation — the theorem itself is stated and proved (or quoted from Amari) in §Theory.
- Line 1042: mixture-of-sources "adapted from"; the full derivation lines 1046–1113 is in §Theory.
- Line 1294: "discusses precision optimization as a learning problem" in companion — the closed-form derivation $\alpha^* = c_0/(b_0 + D_{\text{KL}})$ is in §Theory at lines 1295–1346.
- Line 1352: companion "adopts the per-coordinate form" — the per-coordinate derivation is in §Theory at lines 1330–1352.
- Line 1365: companion "carries the corresponding chain-rule slot" — the chain rule is derived in §Theory at lines 1354–1365 (Eq. eq:alpha_product_rule_itfb).
- Line 1623: rectangular multi-head "treated in" — single-head GL($d_k$) construction is in §Theory; multi-head shape is mentioned but not derived in §1.16.
- Line 1676: Rényi attention "carries the reference implementation in" — Rényi construction Eq. (eq:renyi_attention_itfb) is *in §Theory*.
- Line 1702: literal phrase "(see also the companion paper)" — see-also pointer.

The honest reclassification is **5 load-bearing + 9 provenance/adopts/see-also**, against red's threshold of ≥10 load-bearing reduction steps. Red's falsification condition (ii) sets a quantitative bar; under the operative criterion, that bar is not crossed by the actual citation pattern.

This is also a defense against sub-claim 5 (self-containedness). Of the 5 strict delegations, 3 of them (multi-head, RoPE, FFN at line 1875) are **extensions outside §Theory's core scope** — the section's stated scope is the single-block GL(K) construction; multi-head and RoPE are reductions to standard architectures whose full development belongs in the companion paper *as the manuscript's organizational choice*. The cocycle proof at 1209 is a 2-line consequence the reader can reconstruct; the transformer-limit at 1607/1615 is the bridge to standard scaled dot-product attention, which the manuscript shows as a three-limit collapse without re-deriving QK^T mechanics. Two of the five delegations are *organizational choices*, not derivational gaps.

## (iii) Counter-attack on red's weakest evidence

Red's strike on the cross-scale shadow uses the Dirac-delta-singularity observation at lines 1458–1459 to argue the manuscript itself flags the $\sigma^2 \to 0$ limit as ill-defined. This is a *misreading of the manuscript at the lines red cites*. Lines 1458–1459 concern the *observation likelihood as Dirac sensory agent* — a Dirac $q_{e_k}(c) = \delta(c - c_k)$ giving infinite KL when $q_i \neq \delta$. This is a different limit from the cross-scale rigid-link $\sigma^2 \to 0$ of Appendix A. The observation-side Dirac and the cross-scale rigid-link are different objects: the former is a sensory likelihood collapse, the latter is a regularization sequence on a *prior* that the manuscript explicitly defines as the "rigid-link" limit of a finite-precision conditional $p(s_\ell | s_{\ell+1}, \sigma)$.

Sønderby 2016 §3.1 builds the ladder VAE generative branch with **explicit finite-variance Gaussian conditionals** $p(s_\ell | s_{\ell+1}) = \mathcal{N}(\mu_\ell(s_{\ell+1}), \sigma_\ell^2)$. The rigid-link limit $\sigma_\ell \to 0$ is exactly the limit of zero conditional variance — a Dirac on the prior conditional, not a Dirac on the observation likelihood. This is *legitimate* as a regularization sequence of Sønderby-style finite-variance models; it is what one would call a "deterministic decoder" limit of the ladder VAE. The manuscript's framing at line 552 — "structural commitment of the framework rather than a theorem of standard hierarchical variational inference" — is honest about being a limit of standard hierarchical VI, not an instance of it.

The pivotal question for sub-claim 5 is whether the cross-scale shadow is *self-contained* in the sense the operationalization requires. The manuscript supplies the augmented-joint construction in Appendix A as the bridge to standard hierarchical VI; whether that appendix actually closes the gap is the operationalization's open question 2 (`01_evidence.md:86`), which neither side has independently verified by reading the appendix. The defensible blue position is that this is a finite-resolvable question — read the appendix and decide — not a §Theory-section-level rejection.

## Newly-discovered canon

- `[Bishop2006 §10.1.3]` — *Pattern Recognition and Machine Learning*, Springer. Conjugate-prior mean-field treatment of Gaussian families with Gamma-Normal prior on precision; ground for the rigid-link limit as a standard regularization of conjugate-prior mean-field. Already in `external_canon_inference.md`. No new canon contribution.
