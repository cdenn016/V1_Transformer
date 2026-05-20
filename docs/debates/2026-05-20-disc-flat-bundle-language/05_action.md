# Action — disc-flat-bundle-language

**From verdict:** RED_WINS

## Recommended action

Revise `Attention/GL(K)_attention.tex:2221-2232` (§8.3 "The Flat Bundle Limit and the Geometry of Language") to one of two forms.

**Form A (preserve intent at evidence-supported scope).** Replace the hypothesis at `:2226` with a structural observation: the GL(K) gauge transformer and standard transformers both implement transport in the flat-bundle class by construction of their respective parameterizations (per-token-frame coboundary $\Omega_{ij} = \exp(\phi_i)\exp(-\phi_j)$ for the former; position-independent $W_Q W_K^\top$ for the latter), and this is compatible with — but does not derive from — the algebraic-homomorphism reading of the Principle of Compositionality in Frege–Montague–Partee. Drop the "approximately path-independent in the sense of gauge transport" framing of natural language as a linguistic claim. Drop sub-claim (b) or restate it as a teleological motivation with no information-theoretic derivation. Replace the falsification protocol at `:2232` with the paraphrase-invariance dispersion measurement Blue proposed in rebuttal, but only after supplying an independent operational definition of "compositional" versus "pragmatic" triples that does not presuppose the hypothesis.

**Form B (retract as substantive hypothesis).** Retract §8.3 as a substantive linguistic hypothesis and present it as a speculative connection in an explicitly-labeled "open conjectures" subsection, acknowledging that (i) the gauge-transport reading of compositionality is a novel construction not derivable from the formal-semantics canon, (ii) the falsification protocol at `:2232` is ill-posed within the user's own framework (the per-token-frame parameterization telescopes to identity around any closed triangle by Nakahara 2003 §10.4), and (iii) the architectural-matching claim has empirical content only conditional on an operational criterion not yet supplied. The COGS / SCAN results [Kim & Linzen 2020; Lake & Baroni 2018] should be cited with their actual empirical sign (transformer failure on structural generalization), and the carve-out for the "compositional-semantic core" should be motivated independently of the hypothesis it is invoked to save.

## Decisive citation that drove the verdict

WebFetch on `https://arxiv.org/abs/1003.4394` (Coecke, Sadrzadeh & Clark 2010) confirmed that the abstract establishes codomain uniformity of the categorical meaning map, not derivation-equivalence in the gauge-transport sense. This was Blue's load-bearing canonical-semantics citation and did not support the inference Blue drew. Combined with the manuscript's self-admission at `:2228` that "compositional-semantic component" lacks an operational definition, and with Popper 1959 §6's demarcation criterion, the hypothesis at `:2226` failed on its own primary sources.

## Follow-up debates

- `disc-paraphrase-invariance-protocol` — if the user wishes to formalize Blue's re-anchored paraphrase-invariance falsifier into a manuscript-grade operational criterion. The current §8.3 text does not contain that criterion; constructing it requires an independent definition of "compositional" versus "pragmatic" triples that does not presuppose the hypothesis under test.

No other sub-claims of §8.3 require separate adjudication. The mathematical existence of the flat-bundle limit was already settled in `2026-05-19-subclaim-A-flat-bundle/04_verdict.md` (BLUE_WINS) and was explicitly out of scope for this debate.
