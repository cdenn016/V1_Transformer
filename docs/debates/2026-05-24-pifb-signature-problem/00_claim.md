# Claim — pifb-signature-problem (Debate 2 of 6)

**Mode:** theory + formal math (verify the trace algebra, Sylvester, group facts)
**Rounds:** 2 (+ optional sur-rebuttal); Judge: panel (canon-strict + scope) + chief; code-truth N/A (no code implements this)
**Panel:** full — coordinators pick 5-of-10 (philosophy-of-science mandatory; gauge-theorist, geometer, info-geometer, numerical-analyst indicated)
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge
**Manuscript span:** Attention/Participatory_it_from_bit.tex:2819–2953

## Claim

The §"Temporal Structure and the Signature Problem" treatment is sound: the $\mathrm{GL}(K,\mathbb{C})$ worked example (Eq. `lorentzian_metric`, :2889) and the finite-speed causal-cone route (Eq. `causal_cone_metric`, :2920) are correct *existence demonstrations* that the framework is structurally compatible with Lorentzian signature — NOT derivations of signature from variational dynamics — and every required postulate and gap (imaginary $\phi_\tau$ assignment, real-part projection, rank-1→rank-2 change, $\pm$ bilinear-form sign choice, single-generator collapse, conformal-class ambiguity, dimension non-selection, first-order-dynamics tension, gauge-noninvariance of $\mathrm{tr}(A_\mu A_\nu)$) is accurately and completely disclosed.

## Sub-points the debate must test

1. **Worked-example algebra.** $G_{\tau\tau}=i^2(\partial_\tau\psi_\tau)^2\mathrm{tr}(T^2)=-2(\partial\psi)^2$, $G_{xx}=+2(\partial\psi)^2$, $G_{\tau x}\in i\mathbb{R}$ with $T=\mathrm{diag}(1,-1)$; the rank-1 complex form ($\det=0$) → rank-2 real Lorentzian after $\mathrm{Re}(\cdot)$. Is the trace algebra correct? Is the real-part projection a legitimate operation or a fatal gap (the manuscript calls it a "derivation gap")? Verify symbolically.
2. **Group-theory facts.** Local frame group $\mathrm{SO}^+(1,1)$ (2D) / $\mathrm{SO}^+(1,3)$ (4D); $\mathrm{SL}(2,\mathbb{C})\cong\mathrm{Spin}^+(1,3)\subset\mathrm{GL}(2,\mathbb{C})$; Wick rotation between real forms of $\mathrm{SO}(4,\mathbb{C})$. Are these stated correctly (canon: standard Lie group / Lorentz-group facts)?
3. **Sector split.** Claim that complex $\Omega$ renders $\Omega\Sigma\Omega^\top$ non-Hermitian and Gaussian KL complex/negative, so belief transport stays in $\mathrm{GL}(K,\mathbb{R})$ while only the connection complexifies. Is this coherent — does confining complexity to the connection sector while keeping a real Fisher fiber actually deliver a signature on the BASE metric, or is the indefinite $\mathrm{tr}(A_\mu A_\nu)$ doing all the work in a way that is disconnected from the (positive-definite) statistical content?
4. **Causal-cone route.** Sylvester's law: finite max speed $c_\mathcal{I}$ + positive-definite spatial $h$ → $g=-c_\mathcal{I}^2 d\tau^2 + h$ has signature $(-,+,\dots,+)$. Correct? Are the disclosed gaps (conformal-class, dimension non-selection, first-order parabolic-flow tension giving infinite signal speed) complete and correct?
5. **Honesty of the "existence demonstration, not derivation" framing.** Given the real-part projection has "no physical principle mandating it" (manuscript's own words), is calling the result "structurally compatible with Lorentzian signature" defensible, or does it still overclaim?

## User context

Debate 2 of 6 (see `../2026-05-24-pifb-emergent-spacetime-SERIES.md`). The manuscript is exceptionally hedged here; the real test is whether the hedges are CORRECT and COMPLETE or whether residual errors/overclaims survive even within the existence-demonstration framing. This section inherits the $\kappa(A,A)=\mathrm{tr}(A_\mu A_\nu)$ horizontal block from Debate 1 (whose canonical-layout status was REMANDed).
