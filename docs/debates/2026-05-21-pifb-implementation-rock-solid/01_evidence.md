# Evidence Pack — pifb-implementation-rock-solid

Mode: theory. Canon location: `.claude/agents/vfe-knowledge/`. Working directory: `docs/debates/2026-05-21-pifb-implementation-rock-solid/`.

## Scope

§Implementation of `Attention/Participatory_it_from_bit.tex` runs **lines 2101–2304**, with seven subsections:

| Line | Subsection |
|------|------------|
| 2101 | `\section{Implementation}` |
| 2103 | The Participatory Universe: Multi-Scale Emergence and Dynamics |
| 2108 | Hierarchical Scale Structure |
| 2116 | Meta-Agent Formation: Variational Principle and Threshold-Based Implementation |
| 2195 | Bottom-Up Emergence and Renormalization Group Structure |
| 2232 | Emergent Timescale Hierarchy Across Scales |
| 2243 | Top-Down Participation: Closing the Loop |
| 2295 | Non-Equilibrium Dynamics and Perpetual Evolution |

§Implementation ends just before `\section{Results}` at line 2305. Per the user's "PIFB Codebase Split" auto-memory entry, the simulator code is in `C:\Users\chris and christine\Desktop\MAgent_Model-main\gauge_agent\`, not in the V13 transformer codebase.

## Key manuscript references

### Meta-Agent Formation (§Variational Principle, lines 2119–2160)

- **Free-energy improvement criterion (line 2123, `eq:meta_agent_FE_criterion`)**: $\mathcal{F}^*[q_I, p_I, s_I, r_I, \{q_i, s_i\}_{i \in I}] + C(I) < \mathcal{F}^*[\{q_i, p_i, s_i, r_i\}_{i \in I}]$.
- **Quadratic-vs-sublinear scaling argument (line 2129)**: savings scale as $\sim |I|(|I|-1)\varepsilon$, while $C(I)$ grows sub-linearly or logarithmically.
- **IB Lagrangian (line 2133, `eq:meta_agent_IB`)**: $\mathcal{L}_{\mathrm{IB}}[T \mid X] = I(T;X) - \beta_{\mathrm{IB}}I(T;Y)$, with `[tishby1999information, bialek2001predictability, chechik2005information]` cited.
- **Variational barycenter (line 2142, `eq:meta_agent_barycenter`)**: $q_I^* = \arg\min_{q_I} \sum_{i \in I} w_i^I \mathrm{KL}(\Omega_{Ii} q_i \| q_I) + \lambda \mathrm{KL}(q_I \| p_I)$, with closed-form $\mu_I^*$ at `eq:meta_agent_mu_barycenter` and $\Sigma_I^*$ at `eq:meta_agent_sigma_barycenter`.
- **Karcher frame barycenter (line 2156, `eq:meta_agent_frame_barycenter`)**: $U_I^* = \arg\min_{U \in G} \sum_{i \in I} w_i d_G(U, U_i)^2$, with explicit non-compact $\mathrm{GL}^+(K)$ caveat at line 2160.

### Threshold-Based Implementation (lines 2162–2174)

- **Belief coherence (line 2168)**: $C_q(\{i\}, x) = \exp\left[-\frac{1}{\tau_q |\{i\}|^2}\sum_{i,j \in \{i\}} \mathrm{KL}(q_i^{(s)}(x) \| \Omega_{ij}[q_j^{(s)}](x))\right] \in [0, 1]$.
- **Model coherence**: $C_s(\{i\}, x) = \exp[-V^{(s)}_{\{i\}}(x)/\tau_s]$.
- **Presence**: $P(\{i\}, x) = \frac{1}{|\{i\}|}\sum_i \chi_i(x)$.
- **Consensus score (line 2174)**: $\Gamma(\{i\}, x) = P \cdot C_q \cdot C_s \in [0, 1]$. **Three multiplicative factors**.
- **Explicit rejection of `1 - KL` form (line 2174)**: "A bounded $\Gamma \in [0, 1]$ matters because $\mathrm{KL}$ is unbounded above, so a $1 - \mathrm{KL}$ surrogate would be signed and could give misleadingly positive products from two negative factors; the Gibbs form $\exp[-V/\tau]$ avoids this and is the same exponential detector that reappears in Appendix~\ref{app:rigorous_rg}." The manuscript argues against `1-KL` and in favor of `exp(-KL/τ)`.
- **Threshold values**: $\Gamma_{\min} = 0.5$, $N_{\min} = 2$ (line 2174).
- **Self-disclosed status (line 2174)**: "The relation between the detector and the variational criterion is heuristic and partial rather than a derivation… we do not establish that the detector's product form exactly tracks the variational-criterion savings even in the high-coherence limit. Whether a continuous-time evaluation of~\eqref{eq:meta_agent_FE_criterion} reproduces the same hierarchical organization that the threshold-based detector produces is open."

### Gauge-Covariant Belief Construction (lines 2176–2194)

- **Working implementation formulae (`eq:meta_agent_mu_impl`, `eq:meta_agent_sigma_impl`, lines 2181–2186)**: weighted average over transported moments. Manuscript drops dispersion term as "leading-order approximation in the high-coherence regime" (line 2179).
- **Saddle-point weight (line 2187)**: $w_i^I(x) = \chi_i(x) \exp[-\mathrm{KL}(q_i^{(s)} \| \bar{q}_I^{(s)})]$, "implicit and solved by fixed-point iteration."
- **Aggregation alternative rejected (line 2187)**: "the precision-weighted product-of-experts form $\bar\Lambda_I = \sum_i \tilde{w}_i \tilde{\Lambda}_i$ is a different aggregation interpretation (consensus sharpening rather than pooled uncertainty) that the present construction does not adopt."
- **Lie-algebra-additive frame (line 2191)**: $\phi_I^{(s+1)}(x) = \sum_i w_i \phi_i^{(s)}(x) / \sum_i w_i$, "first-order Baker–Campbell–Hausdorff approximation."

### Bottom-Up Emergence and RG (lines 2195–2213)

- **RG-inspired status (line 2197)**: "RG-inspired rather than a literal RG analysis: we do not exhibit a β-function, locate fixed points, or demonstrate scale invariance beyond the parametric form, and the analogy is structural rather than computational."
- **Scale-invariance of functional form (lines 2203–2211)**: $S^{(s+1)}$ has same form as $S^{(s)}$, justifying RG-style aggregation.
- **Future-work flag (line 2213)**: "The investigation of RG fixed points and critical exponents in this information-theoretic setting constitutes an important direction for future work we are currently engaged in."

### Cross-Scale Information Flow (lines 2215–2224)

- **Eq. eq:cross_scale_information_flow (line 2219)**: $\mathcal{I}_{s \to s+1} = \sum_I \sum_{i \in I} \mathrm{KL}(q_i^{(s)} \| \Omega_{i,I}[q_I^{(s+1)}])$.
- **Caveat (line 2222)**: "we avoid writing this quantity as a mutual information, since mutual information requires a joint distribution on $(k_i, k_I)$ rather than a pair of marginals."

### Emergent Properties at Higher Scales (lines 2226–2230)

- **Empirical disclosure (line 2228)**: "We expect, though do not directly measure in the single-seed run of Section~\ref{sec:results}, that meta-agents exhibit properties not present at lower scales."

### Top-Down Participation (lines 2243–2293)

- **Cross-scale shadow priors (line 2247, `eq:topdown_priors`)**: $p_i^{(s)}(x) = \Omega_{i,I}[q_I^{(s+1)}](x)$, $r_i^{(s)}(x) = \tilde\Omega_{i,I}[s_I^{(s+1)}](x)$.
- **Ouroboros free-energy fragment (line 2270, `eq:ouroboros_F`)**: $\mathcal{F}_{\mathrm{ouro}} = \sum_{k \ge 0} \lambda_0 \rho^k [\mathrm{KL}(q_i \| \Omega_{i,I_k}[q_{I_k}^{(s+k)}]) + \mathrm{KL}(s_i \| \tilde\Omega_{i,I_k}[s_{I_k}^{(s+k)}])]$.
- **Pooling identifications (line 2275)**: West-Harrison 1997 (dynamic discount), Hinton 2002 (product-of-experts), Genest-Zidek 1986 (log-linear pool), Bissiri-Holmes-Walker 2016 (tempered Bayes). Four anchor citations.
- **Self-Referential Closure (line 2277)**: top-scale agent prior is coherence-weighted average over all active agents and scales.
- **Implementation Note (line 2284)**: "Whether the released simulator code realizes the full transport $\Omega_{i,I}$ or a frame-trivial substitute is not independently verified in this manuscript; the simulator code release is deferred to a follow-up (Section~\ref{sec:methods_metagent}). The transformer codebase referenced in the abstract is a separate code path with its own cross-layer prior handoff (an identity-copy with damping, not a multi-scale transport) and should not be read as the simulator implementation of the present subsection."

### Non-Equilibrium Dynamics (lines 2295–2303)

- **Equilibrium-score indicators**: energy flux $\Phi_E(t) = |d\mathcal{F}/dt|$, information flux $\Phi_I(t)$, gradient variance $V_\nabla(t)$, combined as $E_{\mathrm{score}} = (\Phi_E + \Phi_I + V_\nabla)/3$, with threshold $E_{\min} = 1.0$.
- **Mechanisms maintaining non-equilibrium (line 2303)**: new observations, top-down feedback, gauge freedom, spatial separation.

## Manuscript-vs-simulator findings (sub-claim 6)

The manuscript declares specific implementation choices for the threshold detector (§Implementation lines 2162–2174) and the gauge-covariant belief construction (lines 2176–2194). The actual simulator code at `C:\Users\chris and christine\Desktop\MAgent_Model-main\gauge_agent\meta_agents.py` is the operative implementation.

### Match: gauge-covariant barycenter (lines 2181–2191 ↔ `meta_agents.py:167–399`)

The `MetaAgentFormation.form_meta_agent` method (`meta_agents.py:167–399`) docstring explicitly references manuscript lines 1902, 1905, 1907, 1909, 1911. Implementation:
- Transports constituent moments to the reference frame (`meta_agents.py:217–238`) via `transport_mean(omega_ij, …)` and `transport_covariance(omega_ij, …)` — gauge-covariant.
- Fixed-point iteration on $(\mu, \Sigma, w)$ with weight $w_i \propto \chi_i \exp[-\mathrm{KL}]$ (line 182 docstring).
- Lie-algebra-additive frame averaging (line 193–194 docstring), matching manuscript line 2191.
- Drops the dispersion term explicitly per manuscript line 2179.
- Rejects the precision-weighted product-of-experts alternative per manuscript line 2187.

This is a structurally faithful realization of the manuscript's working implementation formulae.

### Mismatch: consensus-detector form (lines 2168–2174 ↔ `meta_agents.py:55–91`)

The `ConsensusDetector` class (`meta_agents.py:35–149`) implements a coherence form that diverges from the manuscript in three specific ways:

1. **Coherence formula**: simulator at `meta_agents.py:56–66` returns `1.0 - E` (where `E` is the mean post-transport KL). The manuscript at line 2169 specifies $C_q = \exp\left[-V/\tau_q\right]$ — the Gibbs exponential form. The manuscript explicitly rejects the simulator's `1 - KL` form at line 2174: "a $1 - \mathrm{KL}$ surrogate would be signed and could give misleadingly positive products from two negative factors; the Gibbs form $\exp[-V/\tau]$ avoids this."
2. **Consensus-score factors**: simulator at `meta_agents.py:89–91` returns $C_{\mathrm{belief}} \cdot C_{\mathrm{model}}$ — two factors. The manuscript at line 2174 specifies $\Gamma = P \cdot C_q \cdot C_s$ — three factors, including the presence factor $P$.
3. **Bounds**: simulator's `1 - E` is not bounded in $[0, 1]$ when KL exceeds 1; the manuscript's exponential form is.

This is a substantive manuscript-vs-code discrepancy: the manuscript argues against the form the simulator implements.

### Self-disclosed gap (line 2284)

The Implementation Note at line 2284 itself states: "Whether the released simulator code realizes the full transport $\Omega_{i,I}$ or a frame-trivial substitute is not independently verified in this manuscript; the simulator code release is deferred to a follow-up (Section~\ref{sec:methods_metagent})." This is the manuscript flagging its own sub-claim 7 (no unresolved gaps) wound.

## Canon excerpts (external, from `vfe-knowledge/external_canon_*.md` plus relevant primary sources)

- **Renormalization group canon**: Wilson 1971 "Renormalization Group and Critical Phenomena" Phys. Rev. B 4 (3174–3183); Cardy 1996 *Scaling and Renormalization in Statistical Physics*. RG requires: (i) a rescaling transformation $\mathcal{R}_b$ acting on the action space, (ii) iteration to find fixed points $\mathcal{R}_b S^* = S^*$, (iii) linearization around fixed points to compute critical exponents. The manuscript at line 2197 explicitly disclaims (ii) and (iii); only the parametric form-invariance is established.

- **Information bottleneck canon**: Tishby-Pereira-Bialek 1999 *Allerton* "The information bottleneck method"; Chechik-Globerson-Tishby-Weiss 2005 *JMLR* "Information bottleneck for Gaussian variables" §3 (Gaussian-IB closed form via canonical correlation analysis); Bialek-Nemenman-Tishby 2001 *Neural Computation* "Predictability, Complexity and Learning" §3. The IB Lagrangian sign convention varies in the literature; canonical Tishby form is $\mathcal{L} = I(T;X) - \beta I(T;Y)$, matching the manuscript at line 2133.

- **Gaussian-IB closed form** [Chechik-Tishby 2005 Theorem 3.1]: optimal $T = AX + \xi$ with $A$ a precision-weighted projection along canonical-correlation eigenvectors.

- **Hierarchical Bayesian pooling**:
  - Genest-Zidek 1986 *Statistical Science* "Combining Probability Distributions: A Critique and an Annotated Bibliography" §3 (log-linear opinion pool, external Bayesianity).
  - Hinton 2002 *Neural Computation* "Training Products of Experts by Minimizing Contrastive Divergence" §2 (PoE form $p \propto \prod_k p_k^{\lambda_k}$).
  - West-Harrison 1997 *Bayesian Forecasting and Dynamic Models* Ch. 6 (multiplicative discount factor $\delta \in (0, 1]$).
  - Bissiri-Holmes-Walker 2016 *JRSSB* "A general framework for updating belief distributions" §2 (generalized-Bayesian update with loss-based likelihood).

- **Karcher / Riemannian mean** [Karcher 1977 *CPAM*]: for a smooth Riemannian manifold $M$ with metric $d$, the Karcher mean $\bar{x} = \arg\min_x \sum_i d(x, x_i)^2$ exists and is unique on convex normal balls of radius bounded by the injectivity radius. For compact connected Lie groups with bi-invariant metric, the existence-uniqueness condition is satisfied on balls of radius $< \pi/2$. For non-compact $\mathrm{GL}^+(K)$, no bi-invariant Riemannian metric exists (the Killing form on $\mathfrak{gl}(K)$ is indefinite); substitutes are required.

- **Baker–Campbell–Hausdorff** [Hall 2015 *Lie Groups, Lie Algebras, and Representations* §5.3]: $\log(e^X e^Y) = X + Y + \tfrac{1}{2}[X, Y] + \tfrac{1}{12}[X-Y, [X,Y]] + \ldots$. The Lie-algebra-additive average $\phi_I = \sum_i w_i \phi_i$ is exact for abelian $G$ and accurate to $\mathcal{O}(\|\phi\|^2)$ otherwise.

- **Wheeler participatory universe** [Wheeler 1983 in *Some Strangeness in the Proportion*]; [Wheeler 1990 "Information, Physics, Quantum: The Search for Links"]: a philosophical picture of observer-observed feedback, not a formal mathematical structure. The manuscript at line 2106 says explicitly: "This section presents the mathematical framework for the feedback loop and demonstrates that participatory-like dynamics are consistent with variational free energy minimization in the post-detection regime. We emphasize this is a toy model demonstrating possibility, not a claim about physical reality."

- **Non-equilibrium statistical mechanics** [de Groot-Mazur 1962 *Non-Equilibrium Thermodynamics*; Prigogine 1977 *Nobel lecture*]: standard non-equilibrium indicators are entropy production, dissipation function, Onsager reciprocal relations. The manuscript's $E_{\mathrm{score}} = (\Phi_E + \Phi_I + V_\nabla)/3$ is an aggregate indicator, not a canonical entropy-production rate.

## What this evidence does NOT settle

1. Whether the threshold-detector form mismatch (manuscript Gibbs `exp(-KL/τ)` vs simulator `1 - KL`) is the wound it appears to be, or whether the manuscript has been revised to disclose this. The manuscript line 2174 explicitly argues against the `1-KL` form the simulator uses; this needs a verdict on which artifact governs ("the manuscript states what was implemented" vs "the simulator is the artifact and the manuscript needs to be amended to match").
2. Whether the RG-inspired status disclosure at line 2197 is sufficient to discharge sub-claim 4 (falsifiability/scope) for the entire §Bottom-Up Emergence subsection, or whether the subsection still trades on RG language load-bearingly elsewhere (e.g., line 2210's $S^{(s+1)}$ scale-invariance claim, line 2213's "critical phenomena, fixed points, or universal behavior" suggestion).
3. Whether the IB Lagrangian framing at line 2133 (cited as "research direction" with three unsupplied ingredients at line 2138) is genuinely flagged as a research direction or whether downstream prose treats it as load-bearing for the variational closure of the FE-improvement criterion.
4. Whether the "Emergent Properties at Higher Scales" subsection (lines 2226–2230) constitutes a substantive claim ("Whole becomes qualitatively different from sum of its parts") backed only by the undisclosed expectation ("We expect, though do not directly measure"), and whether reviewer-realism for this section is acceptable for publication.
5. Whether the Karcher non-compact $\mathrm{GL}^+(K)$ caveat at line 2160 is honored later in §Implementation when the simulator's `gauge_agent/` is documented as operating in $\mathrm{SO}(3)$ / $\mathrm{GL}(K)$ regimes — i.e., whether the caveat actually constrains anything the section claims.
6. Whether the four pooling-anchor citations at line 2275 (West-Harrison, Hinton PoE, Genest-Zidek, Bissiri-Holmes-Walker) are each correctly cited (right paper, right claim) for the ouroboros tower's discount weighting.
7. Whether the non-equilibrium-dynamics aggregate indicator $E_{\mathrm{score}}$ at line 2301 ($(\Phi_E + \Phi_I + V_\nabla)/3$ with $E_{\min} = 1.0$) is a canonical non-equilibrium measure or an engineered indicator without prior literature precedent.

The opening teams should pick the strongest 2–3 vectors per side from this list (and beyond) and synthesize.
