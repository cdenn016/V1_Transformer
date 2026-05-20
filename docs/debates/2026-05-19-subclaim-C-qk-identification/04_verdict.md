# Verdict — subclaim-C-qk-identification

## Outcome

BLUE_WINS

## Decisive evidence

`Attention/GL(K)_attention.tex:1252`, read together with red's own falsification condition #4 at `02_red_opening.md:35`. The manuscript states verbatim: "The gauge-theoretic identification $\sigma^{-2}\Omega^{-\top} \leftrightarrow M_h^a$ operates at the level of the invertible head-space factor, not the ambient low-rank kernel." The sub-claim under debate (`00_claim.md:11`) inherits exactly this scope: "at the level of the invertible head-space factor $M_h^a := A_Q^a (A_K^a)^\top \in \mathrm{GL}(d_{\text{head}})$." Red's falsification condition #4 explicitly grants: "If the claim is read narrowly as 'the gauge framework's transport $\sigma^{-2}\Omega^{-\top}$ plays the role of the invertible factor $M_h^a$ of the standard's bilinear form'... then the claim is defensible and my attack is reduced to a labeling objection." The claim before the judge is that narrow reading; red's own steelman at `02_red_opening.md:5` further concedes the value-level identity that constitutes the head-space identification.

## Reasoning

Both teams agree on the algebraic content. The thin-SVD identity $W_Q^a (W_K^a)^\top = U_Q^a M_h^a (U_K^a)^\top$ is exact (red's rebuttal concession at `03_red_rebuttal.md:5`, blue's defense at `02_blue_opening.md:13`). The dispute reduces to what "recovers the standard inner-product score" requires.

Vaswani 2017 §3.2.1 (canon `external_canon_transformers.md:16`) defines attention by the scalar score $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$; the operational content of attention is the scalar logit, not the algebraic location of the invertible factor in the parameterization. The gauge framework's $\mu_i^\top M_h^a \mu_j$ produces the same scalar logit as the standard's $\mu_i^\top W_Q^a (W_K^a)^\top \mu_j$ when restricted to the head subspace via the shared isometric factors $U_Q^a, U_K^a$. Maps that compute the same scalar on the same input agree as maps; whether the invertible factor is bundled into $Q_i = A_Q^{a\top} U_Q^{a\top} \mu_i$ (standard) or carried in the kernel as $M_h^a$ (gauge) is a representation choice.

Red's strongest move in the rebuttal — that the standard's $Q_i^\top K_j$ is a $d_{\text{head}}$ dot product whereas the gauge form's $\mu_i^\top \Omega^{-\top} \mu_j$ is a $d_{\text{model}}$ bilinear form, so they are scalar-equal but not algebraically identical objects — is a real distinction of representation, but it does not falsify a value-level identification on the head subspace. Red produces no canonical citation defining "recover" in a stricter parameter-level sense. Canon pitfall #2 (`external_canon_transformers.md:139`) flags the non-uniqueness of the inverse map (transformer → gauge); the manuscript discloses the descent at `:1245–1252` and the sub-claim is scoped accordingly. Canon pitfall #10 (`:147`) warns against conflating learned vs derived in training-dynamics comparisons; the claim is explicitly forward-attention only and inherits scope from the prior debate verdict (`01_evidence.md:68`).

The dimension-switch objection at `:1240` is real (the headline equation places the product in $\mathrm{GL}(d_k)$ which is sloppy when $d_k$ reads as $d_{\text{model}}$), but red's falsification condition #3 grants that the sub-claim as adjudicated has already descended to $d_{\text{head}}$ — and `:1247` (label `eq:head_space_kernel`) makes the head-space scope explicit. The non-uniqueness rescaling $(\sigma^2, \Omega) \to (c\sigma^2, c\Omega)$ is a gauge-fixing observation; the bilinear form $\mu_i^\top M_h^a \mu_j$ is invariant under it, so the value-level claim survives.

Red's attack lands against a stronger reading than the claim asserts. The sub-claim is value-level by construction (head-space bilinear form), the manuscript discloses the scope honestly, and red's own falsification conditions concede that this scoping is defensible. Blue wins on the evidence.

## Action

Accept sub-claim C at the head-space scope. The first debate's compound-claim verdict already restricted "exact reduction" to fail at the headline level; sub-claim C captures the part that is defensible. Recommended follow-up: tighten the manuscript's notation at `Attention/GL(K)_attention.tex:1240` so that $\mathrm{GL}(d_k)$ in the headline equation is consistent with $\mathrm{GL}(d_{\text{head}})$ used in the descent paragraph at `:1245-1252` — either rewrite the headline at `:1240` directly in terms of $M_h^a \in \mathrm{GL}(d_{\text{head}})$, or define $d_k$ at first use as the per-head dimension. Either change removes the lingering ambiguity red identifies at `02_red_opening.md:21` without altering the mathematical content. The Route 1 alternative identification at `:1142` (queued follow-up debate) remains the cleaner path that avoids the rectangular-projection issue entirely.
