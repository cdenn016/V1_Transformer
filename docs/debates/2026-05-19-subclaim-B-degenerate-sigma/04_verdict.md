# Verdict — subclaim-B-degenerate-sigma

## Outcome

BLUE_WINS

## Decisive evidence

`Attention/GL(K)_attention.tex:1252` read together with `:1036`–`:1038` and the sympy-verified identity at `:1133`. The manuscript at `:1036`–`:1038` explicitly names the analytic `Σ → 0` limit, identifies it as ill-defined via the absolute-continuity requirement of KL (canon `external_canon_math.md` §1; [Kullback & Leibler 1951; Amari & Nagaoka 2000 Ch. 2]), and replaces it with a joint scaling where `σ² > 0` remains finite. Line 1252 then states the operation as `M := σ⁻²Ω⁻ᵀ` absorbed into `W_Q W_K^T` at finite `σ`. The full Gaussian KL identity feeding this absorption is certified at `:1133` as "verified symbolically against the direct Gaussian KL to machine precision" — canonical sympy-grade evidence per `debate_methodology.md` theory/math protocol. The line-1252 statement therefore correctly characterizes the operation it labels, which is exactly what the claim asserts.

## Reasoning

The claim has two components: (1) the operation is a literal change of parameterization where `σ⁻²` and `Ω⁻ᵀ` always appear jointly as `σ⁻²Ω⁻ᵀ` and are absorbed into `W_Q W_K^T`, and (2) the line-1252 statement "the full limit need not be taken" correctly characterizes that operation. Both sides agree on the substance: σ stays finite, no `σ = 0` substitution occurs anywhere in the algebraic chain (`:1040`, `:1082`, `:1169`, `:1240`), and the downstream formula is the same. Red conceded falsification conditions 1 and 3 of blue's opening.

Red's load-bearing attack — that "literal change of parameterization" demands bijectivity, and that `(σ, Ω) → σ⁻²Ω⁻ᵀ` is many-to-one — does not survive the canon red itself invokes. The scaling redundancy `(σ, Ω) ↔ (cσ, c²Ω)` for `c > 0` is precisely a one-parameter gauge action, and quotient maps onto orbit spaces are the standard form of reparameterization in gauge theory ([Nakahara 2003 §10.1, §10.3] on gauge-equivalence classes; [Peskin & Schroeder 1995 §9.4] on gauge-fixing as reparameterization onto the gauge-fixed slice) and on statistical manifolds with redundant coordinates ([Amari & Nagaoka 2000 Ch. 2]). On the orbit space the map is bijective. Red's own falsification condition (2) named this exact escape: "the map is bijective onto its image under a suitable quotient... `(σ, Ω) ↔ (cσ, c²Ω)`, and this quotient is the canonical parameterization." That condition is satisfied by the construction itself, which red did not contest with a counter-citation in rebuttal.

Red's textual attack — that the section header at `:1024` and the §5.7 summary at `:1958` collectively frame the operation as a "limit" rather than a reparameterization — attacks a target the claim does not raise. The claim under debate explicitly scopes the textual half to "the manuscript's claim at line 1252 correctly characterizes the operation," not to "the section title at line 1024 correctly characterizes the operation." Blue conceded the section header is stylistically imperfect and named the cleaner phrasing ("Implicit-Variance Reparameterization"). That concession does not propagate to the line-1252 statement, which is the textual referent the claim actually defends.

Citation discipline: blue cites manuscript lines, the canonical isotropic Gaussian KL form from `external_canon_math.md` §1, sympy verification at `:1133`, and three external canon sources (Amari & Nagaoka, Nakahara, Peskin & Schroeder) for the quotient-reparameterization definition. Red cites manuscript lines and the canon's KL absolute-continuity requirement but fails to produce a canon citation that disqualifies many-to-one quotient maps from the term "reparameterization." On the contested definition, blue has cited canon and red has asserted.

## Action

Accept the operational content of sub-claim B as correctly stated: the §5.2.2 reduction is a finite-σ joint absorption of `σ⁻²Ω⁻ᵀ` into the learned projections, equivalent to a quotient reparameterization onto the orbit space of the scaling action, and the line-1252 statement accurately describes that operation. Separately, fix the labeling defect blue conceded: rename the paragraph header at `Attention/GL(K)_attention.tex:1024` from "Deterministic Beliefs via Scaled Limit" to "Deterministic Beliefs via Implicit-Variance Reparameterization" (or similar), and rewrite the §5.7 summary phrase at `:1958` from "successive limits" to "successive reductions" with limit (i) phrased as absorption rather than as a limit. This labeling fix is outside the scope of sub-claim B but is a known imperfection both sides agreed on.
