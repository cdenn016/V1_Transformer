# Verdict (scope) — pifb-signature-problem

(Template note: the methodology document names this artifact `04_verdict_scope.md`; the dispatch prompt names it `04_judge_scope.md`. Written under the dispatch name so the chief-judge in this run finds it. Chief: reconcile the filename if the reconciliation step greps for `04_verdict_scope.md`.)

## Claim well-formedness

| Check | Result |
|-------|--------|
| Single declarative sentence? | No — it is a **conjunction of two independently-evaluable propositions** joined by "and": (A) the worked example and causal-cone route are *correct existence demonstrations* of structural compatibility, NOT derivations; (B) *every* required postulate and gap *is accurately and completely disclosed*. The two conjuncts have different truth values and must be split. |
| Falsifiable? What observation would refute? | Yes, and both sides stated the conditions. (A) refuted by a wrong trace/Sylvester/group fact, or by the manuscript asserting dynamical selection. (B) refuted by exhibiting one load-bearing assumption outside the disclosure ledger. Blue named the (B) falsifier in its own opening (falsification condition 3); red produced the instance. |
| Domain (theory / code / both)? | Theory + formal math. No code implements this span (claim header: "code-truth N/A"). The one code-adjacent fact (first-order natural-gradient flow is parabolic, :2929) is conceded by both sides and is not load-bearing for the scope verdict. |
| Key terms anchored? | Anchored: "existence demonstration / not derivation" (manuscript self-definition at :2866, :2902, :2907; Popper §15 existential/universal asymmetry); "signature," "rank," "congruence-invariant" (Sylvester, Horn & Johnson §4.5); the group facts (Hall ch. 5, Knapp ch. I/VI). **Under-anchored: "structurally compatible *with Lorentzian signature*."** This phrase is the live equivocation — see below. Blue's own rebuttal concedes it "is stronger than the construction earns." |

## Claim drift across rounds

| Side | Round | What was actually argued | Drift from 00_claim.md? |
|------|-------|--------------------------|--------------------------|
| Red | Opening | "Engineered by free input choices, not a property of the framework"; framed the separability ansatz as an "undisclosed dependency"; ran a Popper-demarcation attack ("framework forbids no signature"). | **Drift, against the claim's own perimeter.** The claim explicitly says "NOT derivations of signature from variational dynamics." Red's "engineered, not derived" / "framework forbids no signature" attacks a derivation/selection claim the manuscript never makes (it self-describes identically at :2907: "the signature is a chosen input, not a derived output"). The "undisclosed dependency" framing overstated: :2877 *names* the ansatz and writes out the dropped terms. |
| Red | Rebuttal | Retracted the over-broad framing explicitly ("I retract one line from my own panel"); conceded 13 of 14 ledger items, the algebra, Sylvester, group facts, and the existence-demonstration framing; narrowed to (i) the **:2877 sign-flip non-disclosure** and (ii) the **wording overclaim** "structurally compatible with Lorentzian signature." | **Drift corrected.** The surviving two prongs are frame-legitimate and both land on the (B) completeness conjunct and the under-anchored phrase, not on a derivation strawman. |
| Blue | Opening | Defended both halves; pre-registered the narrow ground it does NOT defend (dynamical selection, projection motivation, statistical coupling); produced the 14-item disclosure ledger. | No drift. Blue defended exactly the claim and pre-fenced the strawman perimeter. |
| Blue | Rebuttal | Conceded three items: the bare phrase overstates ("Blue does not defend the unqualified gloss"); the :2877 sign-flip is a "genuine granularity gap" in the completeness clause; the causal-cone route does not apply to the implemented dynamics. Defended the existence-demonstration core and the correctness half. | No drift. Blue conceded the (B) hit precisely and held the (A) core. |

The two sides **converge**: both agree (A) is correct, both agree the (B) "completely" clause fails on exactly the :2877 sign-flip item, and both agree the bare phrase "structurally compatible with Lorentzian signature" overstates relative to the conditional form.

## False dichotomies / equivocations detected

1. **Equivocation on "structurally compatible *with Lorentzian signature*"** (the live one). Two readings: (weak/existential) "a configuration in the admissible space yields a (−,+) form" — TRUE, verified by sympy; (strong/content-bearing) "the framework specifically privileges Lorentzian signature over (+,+), (−,−), 2+2" — FALSE, since the same construction yields every signature by input tuning (manuscript's own :2872, :2900, :2950). The claim's phrase reads as the strong sense; the manuscript's surrounding sentences (:2907) and Blue's defense read it as the weak sense. Blue conceded the strong reading is not earned. This equivocation is the reason the phrase must be rewritten, not merely upheld or struck.

2. **"Engineered, not derived" is not a refutation — it restates the manuscript's self-description.** Red's opening treated "the signature is a chosen input" as an attack; the manuscript asserts verbatim at :2907 "the signature is a chosen input, not a derived output." Attacking a claim by restating the target's own disclaimer is a frame error. Red's rebuttal correctly abandoned this as an attack and re-cast it as support for the genericity/overclaim point. Net: this prong does not move the (A) conjunct; it sharpens why the phrase in (B)/header overclaims.

3. **No theory/code equivocation** (no code implements this; both sides treat the parabolic-flow fact as conceded background).

## Scope leakage detected

1. **The dynamical-selection open problem is correctly EXCLUDED by the claim** ("NOT derivations of signature from variational dynamics"). It is out of scope *for the claim* — which is different from being out of scope for the debate. Any argument turning on "free energy does not select the imaginary φ_τ" (red's Popper-demarcation thread, :2861/:2906/:2907/:2929/:2952) attacks beyond the claim's perimeter and cannot defeat conjunct (A). This is the single most common way this debate could have leaked; both sides ultimately respected the fence (Blue pre-registered it; Red's rebuttal stopped using it as a defeater).

2. **The base/fiber statistical-decoupling question** (signature lives on gauge-noninvariant tr(A_μ A_ν); Fisher metric stays positive-definite; free energy blind to the choice at :2952) is real and both sides agree on the *facts*, but whether decoupling is a *defect* is flagged as an unsettled open problem by the evidence pack (line 30) and by the info-geometer concession. It is not settleable inside this claim's existence-demonstration scope. It informs the phrasing fix but is not itself a sub-claim this debate can resolve.

3. No small-N-empirical-settles-theory leakage; the sympy verifications are symbolic (general ∂ψ), not numerical point checks, so the existence claim is settled at the right grain.

## My verdict reasoning

The claim is a conjunction, and on the rebuttals' own convergence its two conjuncts have opposite truth values. Conjunct (A) — the algebra (:2883–2889), the Sylvester count (:2923), the group facts (SL(2,ℂ)≅Spin⁺(1,3), SO⁺(1,1)/(1,3), Wick between real forms), and the "existence demonstration, not derivation" framing — is correct and decisively Blue's; both panels' sympy agree line-for-line and Red concedes it "plainly and without reservation." Conjunct (B) — disclosure "accurately *and completely*" — is FALSE on exactly one item: :2877 names the separability ansatz and writes out the suppressed cross-derivative terms but does *not* disclose that their presence flips Re(G_ττ) to 2((∂_τψ_x)² − (∂_τψ_τ)²), which can make the temporal direction spacelike and break the (−,+) guarantee. Blue conceded this as "a genuine granularity gap... blue grants it." Separately, the header phrase "structurally compatible with Lorentzian signature" reads in the strong content-bearing sense while the construction earns only the weak existential sense — Blue conceded "Blue does not defend the unqualified gloss." A conjunction with one true and one false conjunct cannot be voted BLUE_WINS (the "completely" clause is literally false on a both-sides-agreed item) nor RED_WINS (the math/existence core is decisively sound and the convergence is genuine, not a split). The frame-correct disposition is to decompose: this is the scope judge's highest-value move and the chief defers to scope on REMAND-for-equivocation. The two surviving live items are concrete and fixable — one manuscript sentence at :2877 and one phrasing change — and each resolves cleanly in opposite directions on respawn.

## Decisive evidence

The frame-breaking observation is the **asymmetry the manuscript itself creates between two parallel disclosures**. At :2872 the manuscript discloses the single-generator-collapse dependence *and its sign consequence* ("the trace identities below would change sign, and the indefinite signature would not emerge"). At :2877 it discloses the separability ansatz and writes out the dropped terms but stops short of the sign consequence — it frames suppression as keeping "the displayed result" clean, not as load-bearing for the signature. Red's sympy (extended evidence line 37; reproduced in the red rebuttal): without separability Re(G_ττ) = 2((∂_τψ_x)² − (∂_τψ_τ)²), sign indefinite. The manuscript does the right thing for the collapse and the wrong thing for separability; that single asymmetry falsifies the word "completely" in conjunct (B) while leaving conjunct (A) untouched. That is precisely the signature of a packed conjunction that must be remanded, not a clean win for either side.

## Outcome (this judge)

REMAND

## If REMAND or OUT_OF_SCOPE

Sub-claims to spawn (decompose the conjunction; each resolves cleanly):

- **Sub-claim 1 (correctness + existence-scope core — likely BLUE on respawn):** "Within Attention/Participatory_it_from_bit.tex:2819–2953, the GL(2,ℂ) worked-example trace algebra (Eq. lorentzian_metric, :2889), the rank-one-complex→rank-two-real characterization (:2887), the causal-cone Sylvester count (Eq. causal_cone_metric, :2923), and the group-theory facts (SL(2,ℂ)≅Spin⁺(1,3), SO⁺(1,1)/SO⁺(1,3) local frame group, ±tr convention dependence) are correct, and the section's self-description as an *existence demonstration of structural compatibility, not a derivation of signature from variational dynamics* is honest." Both sides converge BLUE; respawn only to make the disposition binding.

- **Sub-claim 2 (completeness of the disclosure ledger — likely RED on respawn until manuscript edit):** "The disclosure at Attention/Participatory_it_from_bit.tex:2877 is complete in that it discloses not only that the separable ansatz ψ_τ=ψ_τ(τ), ψ_x=ψ_x(x) suppresses cross-derivative terms, but also that dropping it can flip the sign of Re(G_ττ) to 2((∂_τψ_x)²−(∂_τψ_τ)²) and render the temporal direction spacelike, breaking the (−,+) guarantee." Currently FALSE on both sides' agreement; resolves to BLUE only after the one-sentence manuscript edit lands at :2877.

Action item (NOT a sub-claim — both sides already agreed the replacement language; this needs an editor, not another debate):

- Replace the header/`:2907`/`:2932` phrasing "structurally compatible with Lorentzian signature" with the conditional form both sides accepted: **"compatible with Lorentzian signature conditional on two postulates — an imaginary φ_τ assignment and a rank-changing real-part projection that the construction does not motivate."** This closes the strong/weak equivocation on the under-anchored phrase. Optionally note that the signature lives on the gauge-noninvariant tr(A_μ A_ν) sector that the Fisher metric and free-energy functional do not touch (:2902, :2952), so "compatible" means "an indefinite non-statistical bilinear form can be adjoined to the positive-definite statistical one," not "the statistical content selects the signature."
