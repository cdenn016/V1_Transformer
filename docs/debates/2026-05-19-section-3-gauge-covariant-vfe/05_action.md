# Action — section-3-gauge-covariant-vfe

**From verdict:** RED_WINS (narrow, editorial scope)

## Summary of verdict

The compound claim — that §3 of `Attention/GL(K)_attention.tex` is "robust, theoretically pure, and mathematically correct" — falls on Sub-claim D under the editorial standard established by the §4–§5 debate series. Sub-claims C (single-agent FEP citation), E (curved-base parenthetical as future-work signaling), and F (Elitzur disclaimer) survive clean; both sides reached convergence. Sub-claims A (Ω parameterization) and B (dual-fiber state space) carry minor framing qualifications. The remedy is scoped to citation additions and framing flags. No equation changes, no implementation changes, no structural revision.

The mathematics of §3 is correct. The "theoretically pure" prong of the compound claim falls on the under-cited Gamma conjugate-prior identification at §3.7 lines 914–937.

## Recommended action

Three editorial edits to `Attention/GL(K)_attention.tex`, all scoped to §3.

### Edit 1 — Sub-claim D primary fix (§3.7 lines 914–937)

At line 914–917, replace:

> "A natural choice is a log-barrier form such as:
> $$R(α_i) = b_0 α_i - c_0 \log α_i, \qquad b_0 > 0, c_0 > 0.$$"

with:

> "Choose $R(α_i) = b_0 α_i - c_0 \log α_i$ (with $b_0, c_0 > 0$), the negative log-density of a $\mathrm{Gamma}(α_i; c_0 + 1, b_0)$ distribution — the conjugate prior for the precision parameter of a Gaussian likelihood \citep{bishop2006pattern, murphy2012machine}."

At line 937–940, after the boxed $α_i^* = c_0/(b_0 + D_{\mathrm{KL}}(q_i \| p_i))$, add one clarifying clause:

> "This $α_i^*$ is the MAP estimate of $α_i$ under the Gamma prior $p(α_i) \propto α_i^{c_0} e^{-b_0 α_i}$, with the linear-in-$α$ self-coupling penalty $α_i \cdot D_{\mathrm{KL}}(q_i \| p_i)$ playing the role of the sufficient statistic in the precision posterior."

The empirical-Bayes claim at line 944 then inherits the standard Gamma-EB procedure rather than reading as a free assertion.

### Edit 2 — Sub-claim A secondary flag (§3.2 line 612)

Where Ω is first introduced as a "gauge transport operator," add a forward-pointing flag (one clause) to make the flat-bundle restriction visible at the definition rather than deferred until line 656. Suggested addition immediately after line 615:

> "This vertex-frame parameterization restricts the framework to a globally trivial principal $G$-bundle (flat connection); the edge-relaxed non-flat extension \eqref{eq:edge_relaxed_omega_glk} is deferred to the companion paper \citep{Dennis2025it} and developed in §3.2.1 below."

This avoids the framing gap where a reader of the §3.2 introduction expects non-trivial gauge transport before reaching the Lemma 1 / line-656 disclosure.

### Edit 3 — Sub-claim B secondary clarification (§3.1 lines 580–583)

Where $k_i \in \mathbb{R}^{K_q}$ and $m_i \in \mathbb{R}^{K_p}$ are introduced as separate fiber bundles, add one sentence at the end of the equation block (after line 597) noting:

> "$K_q$ and $K_p$ are independent dimensional parameters of the scaffold. The present work operates entirely on the belief channel $(q_i, p_i, β_{ij}, Ω_{ij})$, so $K_p$ does not enter any §3 equation; the model-channel scaffold is exercised in the companion treatment per the deferral at \eqref{eq:single_agent_fep}."

## Bib additions

Two new entries in the `Attention/references.bib` MACHINE LEARNING / TEXTBOOKS section:
- `bishop2006pattern` — Bishop, "Pattern Recognition and Machine Learning," Springer 2006.
- `murphy2012machine` — Murphy, "Machine Learning: A Probabilistic Perspective," MIT Press 2012.

(Verify these are not already present before adding.)

## Sub-claims C, E, F

No action. Both teams agreed:
- Sub-claim C (single-agent FEP citation) passed clean per [friston2010free, parr2022active]; the form at Eq. eq:single_agent_fep matches the canonical FEP per `external_canon_inference.md`.
- Sub-claim E (curved-base parenthetical) passed clean; line 574's restriction to $c = c^*$ and the §3 equations' independence from curved-base machinery satisfy red's withdrawal in rebuttal.
- Sub-claim F (Elitzur disclaimer) passed clean per [weinberg1995quantum]; the line-983 reparameterization-symmetry framing is textbook-correct.

## Cumulative debate-series state

Eleventh debate in the §3–§5 series. The closed queue:

1. §5 transformer reduction (RED_WINS).
2. Softmax-β stationarity (RED_WINS).
3. Sub-claim A flat bundle (BLUE_WINS, §5 reduction sub-claim).
4. Sub-claim B degenerate $\Sigma$ (BLUE_WINS).
5. Sub-claim C $QK^T$ identification (BLUE_WINS).
6. Sub-claim D $V$ identification (BLUE_WINS).
7. Canonical F vs surrogate (RED_WINS).
8. Multi-head block-diagonal (BLUE_WINS).
9. Route 1 untied carving (RED_WINS).
10. FFN softmax-gradient correction (RED_WINS).
11. **§3 Gauge-Covariant VFE (RED_WINS narrow, this debate).**

After these editorial fixes are applied, the §3 derivation chain is fully audited at the editorial standard established by the prior debates.

## Follow-up debates

None. The §3 audit is complete after the three editorial edits above.
