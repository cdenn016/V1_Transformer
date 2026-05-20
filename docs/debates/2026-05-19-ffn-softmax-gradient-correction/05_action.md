# Action — ffn-softmax-gradient-correction

**From verdict:** RED_WINS

## Summary of verdict

The user's compound claim — that the envelope-theorem justification in §5.3 is exact only at the joint stationary point of $(\mu, \beta)$ and that the softmax-gradient correction is non-negligible during inner E-step iterations — is falsified by primary-source evidence on the manuscript's adopted canonical entropy-augmented F.

Sub-claim A (envelope geometry) was conceded by Blue. The envelope theorem is pointwise in the outer parameter [Milgrom-Segal 2002 Theorem 1; Boyd-Vandenberghe 2004 §5.5.4]; block-stationarity in $\beta$ at the current $\mu_t$ — which is built into the iteration at `transformer/vfe/e_step.py:828` by the assignment $\beta_t := \mathrm{softmax}(-E(\mu_t)/\tau)$ — is the hypothesis. Joint stationarity is not required.

Sub-claim B (FFN derivation completeness) was falsified by Red's symbolic-plus-numerical verification. Under the canonical entropy-augmented F at `Attention/GL(K)_attention.tex:766, :855, :1354`, the row-Lagrangian KKT identity at $\beta_{ij}^* = \pi_j \exp(-E_{ij}/\tau)/Z_i$ gives $\tau\log\beta_{ij}^* - \tau\log\pi_j = -E_{ij} - \tau\log Z_i$. The bracket multiplying $\partial\beta_{ij}/\partial\mu$ collapses to a $j$-independent constant $-\tau\log Z_i$, and $\sum_j \partial\beta_{ij}/\partial\mu = \partial(1)/\partial\mu = 0$ eliminates the correction term identically. Red's PyTorch verification produced $|\nabla_\mu F_{\text{autograd, with entropy}} - \nabla_\mu F_{\text{envelope}}| = 5.96\times 10^{-8}$ at a non-stationary iterate (machine precision), confirming the correction vanishes at every inner-loop iterate under canonical F, not only at joint stationarity.

The narrowest defensible reading of the user's intuition — that under the entropy-suppressed surrogate $\sum_j \beta_{ij} E_{ij}$ the correction is non-zero off the joint stationary point — is correct but is a statement about a functional the manuscript explicitly labels as a surrogate at line 766 and `Attention/GL(K)_supplementary.tex` §B.1, not about the canonical F.

## Recommended action

One editorial sentence to be added to §5.3 before line 1941 (Eq. eq:softmax_gradient_nonlinearity), and a phrase replacement at line 1945. No mathematical content needs revision.

### Edit 1 — insertion before line 1941

Insert a sentence immediately before the displayed $\partial\beta_{ij}/\partial\mu$ equation:

> The bare softmax sensitivity displayed below is the gradient channel that contributes to $\nabla_{\mu_i} \sum_j \beta_{ij} E_{ij}$ (the entropy-suppressed surrogate of Section sec:final_free_energy); under the canonical entropy-augmented F at Eq. eq:free_energy_final, this contribution cancels identically against $\tau \sum_j (\partial\beta_{ij}/\partial\mu_i) \log(\beta_{ij}/\pi_j)$ via the row-Lagrangian KKT identity at $\beta = \beta^*$, by the same cancellation that produces the $-\tau\log Z_i$ reduction.

### Edit 2 — phrase replacement at line 1945

Replace:

> "This centered gradient (deviation of the per-source KL gradient from its $\beta$-weighted mean) provides a higher-order non-linearity beyond the first-order GLU gate."

with:

> "This centered gradient is an additional gradient channel that contributes only to the gradient of the entropy-suppressed surrogate $\sum_j \beta_{ij} E_{ij}$ and vanishes identically in the gradient of the canonical entropy-augmented F (see Eq. eq:autograd_envelope_gap at line 870 for the covariance form of the gap)."

### Edit 3 — qualifier at lines 1949–1950

The "FFN depth from VFE iterations" paragraph says "Each iteration applies the GLU map [eq:vfe_glu] composed with the softmax-gradient correction [eq:softmax_gradient_nonlinearity]." Qualify this with the conditional:

> The composed map describes the per-iteration update under descent on the entropy-suppressed surrogate; under the canonical entropy-augmented F adopted at §4.7, the composition collapses to the bare GLU map by the cancellation identified above.

## Cumulative debate-series state

This is the tenth debate in the §4–§5 series. The closed queue:

1. §5 transformer reduction (RED_WINS).
2. Softmax-β stationarity (RED_WINS — option-(b) framing).
3. Sub-claim A flat bundle (BLUE_WINS).
4. Sub-claim B degenerate $\Sigma$ (BLUE_WINS).
5. Sub-claim C $QK^T$ identification (BLUE_WINS).
6. Sub-claim D $V$ identification (BLUE_WINS).
7. Canonical F vs surrogate (RED_WINS).
8. Multi-head block-diagonal (BLUE_WINS).
9. Route 1 untied carving (RED_WINS).
10. **FFN softmax-gradient correction (RED_WINS — this debate).**

No follow-up debates remain from the §5.3 FFN derivation.

## Follow-up debates

None. The editorial action items above complete the §5.3 corrections.
