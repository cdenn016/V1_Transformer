# Action — disc-rg-token-fixed-point

**From verdict:** RED_WINS

## Recommended action

Three coordinated edits to `Attention/GL(K)_attention.tex` §8.4 (lines 2234–2277), all of which Blue conceded are warranted by the debate record.

**Edit 1 — Artifact attribution at `:2275`.** Replace the verbatim sentence "These deviations arise from clustering artifacts (unequal cluster sizes, correlated assignments) rather than a failure of the underlying scaling argument" with a two-hypothesis open statement: (H1) finite-size clustering bias in the spectral coarse-graining map; (H2) genuine cross-token correlation in trained-transformer attention generating an $O(1)$ floor on the variance of the mean per [Billingsley1995 §27, correlated-mean identity]; with the explicit note that the manuscript does not currently discriminate between H1 and H2 and that a quantitative finite-size correction of the form $y_3^{\mathrm{meas}}(L) = -1 + c L^{-\omega} + \ldots$ would be required to support H1.

**Edit 2 — Conjecture split at `:2262-2272`.** Split `conj:rg_universality` into:
- A **conditional theorem** (parts i and ii under the independence assumption at `:2255`): under independence of perturbations $\Delta_i$ and transport variations $\delta\Omega_{ij}$ across tokens, the CLT exponents $y_1 = -1/2$, $y_2 = -1$, $y_3 = -1$ hold and the intrinsic-channel origin is IR-stable in the textbook [Cardy1996 §3.2] sense. This is canonical i.i.d. mathematics [Billingsley1995 §27].
- An **empirical conjecture**: that this scaling structure applies to *trained* transformer attention graphs. This is the manuscript-declared "non-trivial content" at `:2275` and is currently empirically unsupported — the spectral-clustering measurement of $y_3 \approx +0.2$ classifies $g_3$ as a relevant operator under [Cardy1996 §3.2], not as irrelevant.

The hedges currently at `:2237` ("we conjecture --- but do not claim to have validated on trained models") and `:2255` ("an assumption that holds by construction but requires verification in trained models") should be promoted into the conjecture statement itself rather than left as section-level commentary.

**Edit 3 — Post-hoc ordering claim at `:2277`.** Temper the closing sentence claiming "the ordering of the three limits in Section~\ref{sec:transformer_limit} is dictated by the RG hierarchy" to acknowledge that §5's ordering was algebraically determined upstream of §8.4 and is at best *consistent with* — not derived from — the RG hierarchy.

## Decisive citation that drove the verdict

[Cardy1996 §3.2; Goldenfeld1992 §9.4] standard Wilsonian definition: an IR-stable fixed point requires all neighborhood couplings to have negative scaling dimensions, and a single $y_g > 0$ classifies that coupling as relevant. Applied to the manuscript's own measurement at `:2275` ($y_3 \approx +0.2$), this defeats IR-stability along $g_3$ on the empirical probe the manuscript actually ran. Blue's finite-size-scaling defense via [Cardy1996 §3.5] was foreclosed by Blue's own admission that closing the $-1 \to +0.2$ gap requires the correction to exceed the asymptotic term, which by [Cardy1996 §3.5 eq. 3.40–3.43] places the data in the crossover regime, not the finite-size-correction regime of an IR-stable fixed point.

## Follow-up debates

None on the in-scope claim. The closure question on the augmented $(\mu_A, \Sigma_A)$ family and the empirical question of whether $y_3 \approx +0.2$ is signal or artifact are both legitimate open questions but require **empirical work** rather than further adversarial adjudication:

- Quantitative finite-size correction analysis of the spectral-clustering map (compute $c, \omega$ in $y_3^{\mathrm{meas}}(L) = -1 + cL^{-\omega}$ from larger-$L$ runs or analytical Laplacian-eigenvalue bias arguments [von Luxburg 2007 §5]).
- Trained-transformer coarse-graining experiments with independence-enforced graph partitions to discriminate H1 from H2.

These are experimental, not debate questions. The manuscript should be revised per Edits 1–3 above; the empirical follow-up may be addressed in the companion paper.

## Out-of-scope note

The companion paper's belief-hierarchy Wilsonian RG (`Attention/Participatory_it_from_bit.tex` §4) was explicitly out of scope. The `2026-05-19-rg-construction-meta-agent` verdict (BLUE_WINS with calibration) was not imported. The [Cardy1996 §3.3] effective-action augmented-class *citation* was in-scope and Blue invoked it correctly for part (iv) of the conjecture — that aspect was not where the debate was decided.
