# Verdict — pifb-signature-problem (binding, chief reconciliation)

## First-pass verdicts

| Judge | Outcome | Decisive evidence |
|-------|---------|-------------------|
| canon-strict | RED_WINS | [Lee 2013, *Introduction to Smooth Manifolds* (2nd ed.), Ch. 13] — a pseudo-Riemannian metric is by definition a smooth, symmetric, non-degenerate, real (0,2)-tensor; the framework natively produces a complex rank-1 (det=0) form, and the Lorentzian form of Eq. `lorentzian_metric` (:2889) exists only after the real-part projection the manuscript itself certifies at :2892 has "no physical principle ... that mandates" it. Plus the blue-conceded separability sign-flip: dropping :2877 separability gives Re(G_ττ)=2((∂_τψ_x)²−(∂_τψ_τ)²), sign-indefinite. |
| code-truth   | N/A | No code implements the §"Temporal Structure and the Signature Problem" span (Attention/Participatory_it_from_bit.tex:2819–2953). No jurisdiction; abstained per claim header "code-truth N/A". |
| scope        | REMAND | The claim is a conjunction of two independently-evaluable propositions joined by "and": (A) the worked example and causal-cone route are *correct existence demonstrations* of structural compatibility, NOT derivations; (B) *every* required postulate and gap is *accurately and completely disclosed*. The two conjuncts have opposite truth values — (A) TRUE on both panels' sympy and [Horn&Johnson §4.5]/[Hall Ch.5], (B) FALSE on the :2877 separability sign-flip both sides agree is undisclosed. The frame-breaking observation is the manuscript's own asymmetry: :2872 discloses the single-generator-collapse sign consequence; :2877 discloses the separability ansatz but stops short of its sign consequence. |

## Reconciliation rule applied

**Rule 2 — scope override for REMAND on equivocation.** The scope judge declared REMAND with a concrete well-formedness failure: the claim packs two independently-evaluable propositions (the correctness/existence-demonstration core and the "accurately and completely disclosed" completeness clause) whose truth values diverge, and both sides converged on that divergence rather than contesting a single proposition. Rule 2 fires before Rule 3 (majority) is consulted, so canon-strict's RED_WINS does not control. This is not an override of canon-strict on its domain — canon-strict itself flagged scope's REMAND-on-equivocation as "a defensible alternative reconciliation," and the methodology grants scope special standing on REMAND.

## Decisive evidence (binding)

The binding decisive evidence is the **packing of two opposite-truth-value conjuncts inside one claim**, established by external canon on both sides:

- **Conjunct (A) is TRUE** on verified external canon: signature is a congruence invariant [Horn & Johnson, *Matrix Analysis* (2nd ed.), §4.5, Sylvester's law of inertia], grounding the causal-cone count diag(−c²,h) → signature (−,+,…,+) (:2923) and the GL(K,ℝ) positive-definiteness no-go (:2837, :2950); the group facts SL(2,ℂ)≅Spin⁺(1,3) and SO⁺(1,3) as the identity component of O(1,3) [Hall, *Lie Groups, Lie Algebras, and Representations*, Ch. 5]; the +tr vs −tr convention dependence by Cartan's criterion [Knapp, *Lie Groups Beyond an Introduction*, Ch. I/VI]; reproduced line-for-line by both panels' sympy (`01b_extended_evidence.md` lines 8–12, 34–38).

- **Conjunct (B) is FALSE** on one item, established by [Lee 2013, Ch. 13] (a metric is by definition real, symmetric, non-degenerate — the native object is complex rank-1, det=0) together with the verified sympy that dropping the :2877 separability ansatz yields **Re(G_ττ)=2((∂_τψ_x)²−(∂_τψ_τ)²), sign-indefinite** (`01b_extended_evidence.md` line 37; red rebuttal; blue rebuttal Concession 2 — "a genuine granularity gap in the completeness sub-claim ... and blue grants it"). The manuscript discloses the sign consequence for the single-generator collapse at :2872 but not the parallel sign consequence for separability at :2877 — the asymmetry that falsifies the word "completely."

A conjunction with one true and one false conjunct cannot be voted a clean win for either side; the frame-correct disposition is to decompose.

## Outcome (binding)

**REMAND**

## Reasoning

Rule 2 fired: the scope judge declared REMAND on the concrete ground that the claim packs two independently-evaluable propositions whose truth values diverge, and per the fixed rule ordering the chief stops at the first rule that fires and adopts the scope judge's sub-claim list. Canon-strict's RED_WINS is not overridden on its own domain — its [Lee 2013] metric-definition finding and the blue-conceded separability sign-flip both survive intact and are carried forward as the decisive evidence for sub-claim 2. What changes is the disposition: rather than voting the compound claim RED because two conjuncts fail, the binding verdict separates the conjunct that is decisively sound (the math/group-fact/existence-demonstration core, on which both sides and verified external canon agree) from the conjunct that fails (the "completely disclosed" clause, false on the :2877 item by both sides' agreement). This avoids attributing the failure of the completeness clause to the existence-demonstration core, which would misreport a sound result as falsified, and it avoids upholding "completely" when both sides agree it is literally false on the :2877 item. The binding decisive evidence is the packing itself: conjunct (A) true on [Horn&Johnson §4.5], [Hall Ch.5], [Knapp Ch.I/VI] and both panels' sympy; conjunct (B) false on [Lee 2013 Ch.13] plus the verified sign-indefinite Re(G_ττ) without separability. The existence-demonstration core needs no change; the two failing items are surgical manuscript fixes, not a structural rework.

## Action

Spawn two sub-claims as their own debates (decompose the conjunction), and apply the two manuscript fixes both sides already agreed on. The existence-demonstration core needs no change.

**Sub-claim 1 (correctness + existence-scope core — expected BLUE on respawn):** "Within Attention/Participatory_it_from_bit.tex:2819–2953, the GL(2,ℂ) worked-example trace algebra (Eq. `lorentzian_metric`, :2889), the rank-one-complex→rank-two-real characterization (:2887), the causal-cone Sylvester count (Eq. `causal_cone_metric`, :2923), and the group-theory facts (SL(2,ℂ)≅Spin⁺(1,3), SO⁺(1,1)/SO⁺(1,3) local frame group, ±tr convention dependence) are correct, and the section's self-description as an *existence demonstration of structural compatibility, not a derivation of signature from variational dynamics* is honest." Both sides converge BLUE; respawn only to make the disposition binding.

**Sub-claim 2 (completeness of the disclosure ledger — RED until the manuscript edit lands):** "The disclosure at Attention/Participatory_it_from_bit.tex:2877 is complete in that it discloses not only that the separable ansatz ψ_τ=ψ_τ(τ), ψ_x=ψ_x(x) suppresses cross-derivative terms, but also that dropping it can flip the sign of Re(G_ττ) to 2((∂_τψ_x)²−(∂_τψ_τ)²) and render the temporal direction spacelike, breaking the (−,+) guarantee." Currently FALSE by both sides' agreement; resolves to BLUE only after the one-sentence manuscript edit lands at :2877.

**Manuscript fixes (editor task — both sides already agreed the replacement language; this needs an editor, not another debate):**

1. **Conditionalize the framing.** Wherever the section asserts "structurally compatible with Lorentzian signature" — :2907, :2932, :2952, and the parallel assertions at :3029 and :3109 — replace the bare phrase with the conditional form both sides accepted: *"compatible with Lorentzian signature conditional on two postulates — an imaginary φ_τ assignment and a rank-changing real-part projection that the construction does not motivate."* Optionally note that the signature lives on the gauge-noninvariant tr(A_μ A_ν) sector that the Fisher metric and free-energy functional do not touch (:2902, :2952), so "compatible" means an indefinite non-statistical bilinear form can be adjoined to the positive-definite statistical one, not that the statistical content selects the signature.

2. **Add the separability sign-consequence sentence at :2877.** State that dropping the separability ansatz can render the temporal direction spacelike — Re(G_ττ)=2((∂_τψ_x)²−(∂_τψ_τ)²) is sign-indefinite — mirroring the sign-consequence disclosure already present for the single-generator collapse at :2872. This closes the one completeness gap both sides conceded.

The math/group-fact/existence-demonstration core (Eqs. `lorentzian_metric`, `causal_cone_metric`; SL(2,ℂ)≅Spin⁺(1,3); Sylvester count) is undisputed and canon-confirmed; no change.
</invoke>
