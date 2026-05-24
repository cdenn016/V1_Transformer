# Action — pifb-dimensional-sectors (Debate 4 of 6)

**From verdict:** REMAND (chief judge, Rule 2 scope-override; canon-strict first-pass RED_WINS on the strong reading, scope REMAND-on-equivocation).

## What survives unchanged (granted on external canon)

- Base-vs-fiber bookkeeping (:3019, :3045, :3073): $G_i(c)=\sigma_i^*g_{\mathcal{B}}$ is an $n\times n$ PSD tensor on $T_c\mathcal{C}$, rank $\le n=\dim\mathcal{C}$; the "invisible majority" is correctly a statement about $d\sigma_i$ sampling $\le n$ of $\dim(\mathcal{B})$ fiber directions [Lee 2013 Ch. 11/13]. Correct and careful.
- Dimension arithmetic $\dim(\mathcal{B})=K(K+3)/2$, $K=768\Rightarrow296{,}064$ [Amari–Nagaoka].
- The $(3+1)$ structure is honestly labeled "(Speculative)" with "makes no quantitative predictions ... that could be tested experimentally" (:3071), and the supposition does NOT leak downstream (red withdrew this vector after a grep found four downstream re-hedges at :3188–3192).

## Recommended manuscript fixes (three)

1. **Recast the sector definitions (:3026, :3034, :3042) in reparametrization-invariant quantities, or explicitly mark them chart-relative.** Decisive canon [Cencov 1972; Amari–Nagaoka Ch. 2]: Fisher invariance is invariance of the metric *tensor* under sufficient statistics, NOT of pullback eigenvalue *magnitudes*. Under a base chart change $\tilde G=J^{-\top}GJ^{-1}$ the eigenvalues rescale, so absolute thresholds $\lambda_a>\Lambda_{\text{obs}}$ name a chart, not an intrinsic direction. Blue conceded this ("Red is right on this"). Fix by thresholding on relative gaps / eigenvalue ratios (chart-invariant) or by stating explicitly that the sectors are defined relative to a chosen base chart.
2. **Change ":3055–3062 'satisfy' → 'are assumed to satisfy / are postulated to satisfy'."** The in-span text asserts the hierarchy $|\mathcal{D}_{\text{obs}}|\ll|\mathcal{D}_{\text{subthresh}}|\ll|\mathcal{D}_{\text{internal}}|$ and $\lambda_{\text{obs}}\gg\lambda_{\text{subthresh}}\gg\lambda_{\text{internal}}$ *unconditionally* ("satisfy"), but the manuscript's own postulate box at :3190 lists "the eigenvalue hierarchy itself" under "What is postulated." Reconcile the in-span verb with :3190. A stable three-cluster separation also requires demonstrated spectral gaps [Davis–Kahan 1970] the section does not exhibit; stating the hierarchy as a posit fixes this honestly.
3. **Conditionalize the :3029 "(1+3) Lorentzian decomposition" per Debate 2's signature action.** A PSD spectrum admits only a Euclidean subspace [Sylvester; Nakahara §7.1–7.2]; the temporal sign is imported from the signature postulates and inherits that section's conditionalization (Debate 2 action items). The sentence already self-flags this; ensure the conditional language matches the signature-section fix.

## Spawned sub-claims (from scope)

- **A:** are the sectors reparametrization-invariant features or chart-relative? (Both sides concede chart-relative → fix #1.)
- **B:** is the :3055–3062 in-span hierarchy consistent with the :3190 "postulated" box? (No → fix #2.)

## Carryover

Inherits Debate 2's signature REMAND for the temporal sign (fix #3). The $\dim\mathcal{C}=0$ vacuity in the transformer limit is a scope caveat the :3021 "when $\dim\mathcal{C}$ is large enough" gate already handles — not a defect.
