# Claim — disc-rg-token-fixed-point

**Mode:** theory
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto, anchored on `Attention/GL(K)_attention.tex` §8.4 (lines 2234–2277)
**Canon location:** C:/Users/chris and christine/Desktop/V13_Gauge_Transformer/.claude/agents/vfe-knowledge/

## Claim

All three coupling constants $g_1^{(\mathrm{orig})}, g_2, g_3$ in the gauge VFE family are irrelevant operators (negative scaling dimensions $y_1 = -1/2$, $y_2 = -1$, $y_3 = -1$) under the token-graph coarse-graining map $R_n$, the intrinsic-channel fixed point $g_1^{(\mathrm{orig}),*} = g_2^* = g_3^* = 0$ is therefore infrared-stable, and the spectral-clustering measurement of $y_3 \approx +0.2$ reported at `Attention/GL(K)_attention.tex:2275` is attributable to finite-size clustering artifacts (unequal cluster sizes, correlated assignments) rather than to a genuine failure of the underlying CLT scaling argument.

## User context

This is Conjecture~\ref{conj:rg_universality} from `Attention/GL(K)_attention.tex:2262-2272` together with the scope-and-status paragraph at `:2274-2275`. The manuscript itself concedes that the empirical $y_3$ measured by spectral coarse-graining is approximately $+0.2$ — the **opposite sign** from the predicted $y_3 = -1$ — but blames this on clustering artifacts. The debate is over whether this attribution is defensible or whether the sign reversal is evidence against the conjecture's IR-stability claim.

## Scope (binding for both teams and the judge)

- **In scope.** The token-graph coarse-graining map $R_n$ as defined at `:2249-2254`, the scaling-dimension predictions at `:2255-2260`, parts (i)–(iii) of the conjecture at `:2262-2272`, and the empirical concession at `:2274-2275`.
- **Out of scope.** The companion paper's belief-hierarchy Wilsonian RG (`Attention/Participatory_it_from_bit.tex` §4 / `2026-05-19-rg-construction-meta-agent`). The judge **may not** import the prior verdict from that debate as decisive evidence here — these are different RG constructions. The token-graph $R_n$ is the only RG flow under adjudication.
- Part (iv) (emergent anisotropy) and part (v) ($O(\sqrt{K})$ efficiency gap) of the conjecture are corollaries downstream of (i)–(iii); they are inside the conjecture's domain but the debate's load-bearing terrain is (ii) stability and the $y_3$ sign question.

## Sub-claims (for joint adjudication)

1. **Mathematical CLT result.** Under the assumption of independence of perturbations $\Delta_i$ and transport variations $\delta\Omega_{ij}$ across tokens, the CLT-derived scaling dimensions $y_1 = -1/2$, $y_2 = -1$, $y_3 = -1$ are correct as a statement about i.i.d.\ averaging.
2. **Empirical attribution.** The observed $y_3 \approx +0.2$ and $y_2 \approx -0.6$ from spectral coarse-graining of attention graphs are predominantly clustering artifacts (unequal cluster sizes, correlated assignments) rather than evidence of broken independence across tokens in trained transformers.
3. **IR-stability of the transformer limit.** Part (ii) of the conjecture — "all scaling dimensions are negative; the fixed point is infrared-stable" — holds for *trained transformers*, not merely for synthetic i.i.d.\ perturbations.

Sub-claim 1 is the pure-math statement (CLT exponents). Sub-claim 2 is the empirical attribution that the manuscript makes at `:2275`. Sub-claim 3 is the load-bearing physics claim about transformers as IR-stable fixed points.
