# Evidence Pack — eliminating-external-observations

Neutral fact pack. No editorial framing.

## Manuscript anchor

Primary anchor: `Attention/Participatory_it_from_bit.tex` lines 1394–1467.

### Section structure and line-by-line content

| Line | Element |
|---|---|
| 1394 | `\subsection{Eliminating External Observations: A Self-Contained Framework}` |
| 1396 | Setup: standard formulation has $-\mathbb{E}_{q_i}[\log p_i(o_i|c)]$ with $o_i$ "received from external reality." Cites Friston2010, parr2022active for active-inference framing. |
| 1398 | The motivating claim: "observation terms can be represented by environmental-agent couplings at the mean-gradient level; full variational equivalence requires either a cross-entropy substitution or a fixed-covariance restriction." Self-contained framework conditioned on those substitutions. |
| 1400 | `\subsubsection{Environmental Agents}` |
| 1402 | Standard formulation: observation enters as likelihood $p_i(o|c)$ in F. |
| 1404 | Replacement: environmental agents $\mathcal{E} = \{e_k\}$ that "encode the observational information in their beliefs." Each $e_k$ has $q_{e_k}(c)$ "sharp (low entropy) around the 'true' state corresponding to observation $o_k$." Coupling via Eq. at 1406. |
| 1406 | $\sum_{k \in \mathcal{E}} \beta_{i,e_k}\,\mathrm{KL}(q_i(c) \| \Omega_{i,e_k}[q_{e_k}(c)])$. |
| 1409 | Mechanism: "When environmental agents have sharp beliefs concentrated on specific values (corresponding to definite observations), the coupling drives agent $i$ to align its beliefs with those values — exactly as observation terms would." Env agents declared "internal to the framework, subject to the same information-geometric dynamics as all other agents." |
| 1411 | `\subsubsection{Formal Equivalence}` |
| 1415–1418 | Proposition statement. $\mathcal{F}_{\text{obs}} = \mathcal{F}_{\text{internal}}[\{q_i\}] - \sum_i \int \chi_i(c)\,\mathbb{E}_{q_i(c)}[\log p_i(o_i|c)]\,dc$. |
| 1419–1423 | $\mathcal{F}_{\text{agent}} = \mathcal{F}_{\text{internal}}[\{q_i\}] + \sum_{i,k}\int \chi_{ik}(c)\,\beta_{i,e_k}(c)\,\mathrm{KL}(q_i(c) \| \Omega_{i,e_k}[q_{e_k}(c)])\,dc$. The proposition claims mean-gradient agreement $\partial_{\mu_i}\mathcal{F}_{\text{obs}} = \partial_{\mu_i}\mathcal{F}_{\text{agent}}$ for any $\Sigma_o > 0$. Full variational equivalence at the joint $(\mu_i, \Sigma_i)$ level "requires an additional restriction: either replacing the environmental KL by a cross-entropy coupling $-\mathbb{E}_{q_i}[\log q_{e_k}]$, or restricting to fixed-covariance dynamics in which the entropy discrepancy is constant." |
| 1425 | Dirac caveat: $q_{e_k}(c) = \delta(c - c_k)$ gives $\mathrm{KL}(q_i \| \delta(c - c_k)) = +\infty$ for non-degenerate $q_i$. Refers to Eq. `eq:dirac_kl` (line 1612). Requires finite sensory precision $\Sigma_o > 0$. |
| 1426–1430 | Concrete construction: $q_{e_k}(c) = \mathcal{N}(c\mid c_k, \Sigma_o)$; $p_{e_k}(c) = q_{e_k}(c)$; $\beta_{i,e_k}(c) = 1$. |
| 1431 | Gauge-fixing: "$\Omega_{i,e_k} = I$ implicitly, identifying the sensor's gauge frame with the receiving agent's; this gauge-fixing is the implicit content of the explicit symmetry breaking discussed in Section sec:symmetry_breaking, where environmental agents enter the free energy with frames fixed to specific values rather than being averaged over the gauge orbit." |
| 1431–1434 | Eq.: $\mathrm{KL}(q_i \| q_{e_k}) = \tfrac12[(\mu_i - c_k)^\top \Lambda_o (\mu_i - c_k) + \mathrm{tr}(\Lambda_o \Sigma_i) + \log\tfrac{|\Sigma_o|}{|\Sigma_i|} - d]$. |
| 1435 | Mean gradient: $\Lambda_o(\mu_i - c_k)$. Stated to match $\partial[-\mathbb{E}_{q_i}\log p(o_k\mid c)]/\partial\mu_i$ exactly. |
| 1436–1438 | Covariance gradient: $\partial_{\Sigma_i}\mathrm{KL}(q_i\|q_{e_k}) = \tfrac12(\Lambda_o - \Sigma_i^{-1})$. |
| 1439 | Comparison: $\partial_{\Sigma_i}[-\mathbb{E}_{q_i}\log p(o_k\mid c)] = \tfrac12\Lambda_o$. Discrepancy $-\tfrac12\Sigma_i^{-1}$ from $-\tfrac12\log|\Sigma_i|$ entropy term inside KL. Explicit statement: "equivalent at the mean-gradient level but not as full variational equivalents." |
| 1441–1445 | Cross-entropy resolution: $\mathcal{F}_{\text{agent}}^{\mathrm{xent}}(q_i, q_{e_k}) := -\mathbb{E}_{q_i}[\log q_{e_k}] = \mathrm{KL}(q_i\|q_{e_k}) + H(q_i)$, with $H(q_i) = \tfrac12\log|2\pi e\Sigma_i|$. Recovers negative log-likelihood "exactly when $q_{e_k}$ is the Gaussian sensor density." |
| 1447 | Closing motivation: "The observation-free formulation is more parsimonious ontologically — it requires only agents and their couplings, with no external reality providing special inputs. In this view our so-called Markov blankets composed of sensory agents (cells, organs, etc) are themselves composed of sensory Markov blankets (receptors, proteins, molecules, etc), and onward down to single bits." |
| 1449–1466 | `\subsubsection{Structural Analogy with Variational Stationarity Principles}` — table comparing classical mechanics, GR, gauge theory action principles to $\mathcal{F}$. Explicit disclaimer at 1463–1464: "Structural analogy, not dimensional equivalence … The analogy is at the level of variational stationarity, not at the level of comparable physical content." |

### Cross-document anchors

- §`sec:agent_definition` at line 613: the manuscript's own definition of an agent. Per the definition (lines 617–626):
  - Two primitive smooth sections: belief $q_i$ on state fiber, generative model $s_i$ on model fiber.
  - Two derived sections: prior $p_i = \Omega_{i,I}[q_I^{(s+1)}]$ (cross-scale shadow), hyper-prior $r_i = \tilde\Omega_{i,I}[s_I^{(s+1)}]$.
  - Gauge frame field $\phi_i: \mathcal{U}_i \to \mathfrak{g}$.
  - Variational hierarchy $r_i \to s_i \to p_i \to q_i \to \text{observations}$.

- §`sec:symmetry_breaking` at line 1469: discusses environmental agents as the source of explicit (not spontaneous) gauge-symmetry breaking. "Environmental agents enter the free energy with fixed gauge frames, and these fixed frames play the role of an external source field analogous to a Zeeman term in a ferromagnet" (line 1484).

- §`sec:cross_scale_shadows` at line 536: cross-scale shadow relation $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$. The shadow is "a structural commitment of the framework rather than a theorem of standard hierarchical variational inference" (line 546).

- Eq. `eq:dirac_kl` at line 1612: $D_{\mathrm{KL}}(\delta(k-\mu_i) \| \delta(k-\mu_j)) = +\infty$ for $\mu_i \neq \mu_j$, with absolute-continuity reasoning.

- Eq. `eq:gaussian_kl` at line 530: standard closed-form Gaussian KL.

### Specific sympy-checkable claims

For Gaussian $q_i = \mathcal{N}(\mu_i, \Sigma_i)$ and $q_{e_k} = \mathcal{N}(c_k, \Sigma_o)$ with $\Omega_{i,e_k} = I$, in $\mathbb{R}^d$:

- **K1 (mean-gradient match).** Claim: $\partial_{\mu_i}\mathrm{KL}(q_i \| q_{e_k}) = \Lambda_o (\mu_i - c_k)$ and $\partial_{\mu_i}[-\mathbb{E}_{q_i}\log\mathcal{N}(o_k; \mu_i, \Sigma_o)] = \Lambda_o(\mu_i - c_k)$ when $o_k$ is identified with $c_k$.

  Wait: the standard active-inference observation likelihood treats $o_k$ as a fixed data point and $\theta$ (or here $c$) as the latent state with prior $q$ on it: $p(o_k|c) = \mathcal{N}(o_k; c, \Sigma_o)$, so $-\mathbb{E}_{q_i}[\log p(o_k|c)] = \tfrac12(o_k - \mu_i)^\top \Lambda_o (o_k - \mu_i) + \tfrac12\mathrm{tr}(\Lambda_o \Sigma_i) + \text{const}$, and $\partial_{\mu_i}[-\mathbb{E}_{q_i}\log p(o_k|c)] = -\Lambda_o(o_k - \mu_i) = \Lambda_o(\mu_i - o_k)$. Sign convention check.

- **K2 (covariance-gradient mismatch).** Claim: $\partial_{\Sigma_i}\mathrm{KL}(q_i\|q_{e_k}) - \partial_{\Sigma_i}[-\mathbb{E}_{q_i}\log p(o_k|c)] = -\tfrac12\Sigma_i^{-1}$. Sympy-verifiable.

- **K3 (cross-entropy identity).** Claim: $-\mathbb{E}_{q_i}[\log q_{e_k}] = \mathrm{KL}(q_i \| q_{e_k}) + H(q_i)$ with $H(q_i) = \tfrac12\log|2\pi e\Sigma_i|$. Textbook identity [CoverThomas2006 §2.5]; sympy-verifiable.

- **K4 (Gaussian KL form).** Already canon: $\mathrm{KL}(\mathcal{N}_1\|\mathcal{N}_2) = \tfrac12[\log\frac{|\Sigma_2|}{|\Sigma_1|} + \mathrm{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^\top\Sigma_2^{-1}(\mu_2-\mu_1) - d]$ [BleiKuckelbirgJordan2017].

### Manuscript references

- `Attention/Participatory_it_from_bit.tex:1394` — subsection header
- `Attention/Participatory_it_from_bit.tex:1396` — setup paragraph
- `Attention/Participatory_it_from_bit.tex:1398` — conditional equivalence statement
- `Attention/Participatory_it_from_bit.tex:1406` — env-agent KL coupling
- `Attention/Participatory_it_from_bit.tex:1415–1423` — proposition statement
- `Attention/Participatory_it_from_bit.tex:1425` — Dirac caveat
- `Attention/Participatory_it_from_bit.tex:1426–1430` — concrete construction ($q_{e_k}$, $p_{e_k} = q_{e_k}$, $\beta_{i,e_k}=1$)
- `Attention/Participatory_it_from_bit.tex:1431` — gauge-fixing $\Omega_{i,e_k} = I$
- `Attention/Participatory_it_from_bit.tex:1431–1434` — explicit KL closed form
- `Attention/Participatory_it_from_bit.tex:1435` — mean gradient
- `Attention/Participatory_it_from_bit.tex:1436–1439` — covariance gradient + discrepancy
- `Attention/Participatory_it_from_bit.tex:1441–1445` — cross-entropy resolution
- `Attention/Participatory_it_from_bit.tex:1447` — parsimony / Markov-blanket-of-sensors framing
- `Attention/Participatory_it_from_bit.tex:613–631` — manuscript's own agent definition (sec:agent_definition)
- `Attention/Participatory_it_from_bit.tex:1469–1499` — sec:symmetry_breaking
- `Attention/Participatory_it_from_bit.tex:1610–1614` — Dirac-KL eq:dirac_kl

## Canon excerpts

From `external_canon_inference.md` and `external_canon_math.md`:
- Standard variational free energy [Friston2010, ParrPezzuloFriston2022 Ch. 2]: $\mathcal{F}[q(s)] = -\mathbb{E}_q[\log p(o, s)] - H(q)$. Observations enter as data via the joint $p(o, s)$.
- The expected log-likelihood (cross-entropy) $-\mathbb{E}_{q(s)}[\log p(o|s)]$ is the canonical observation term in FEP — not the KL between belief and likelihood density [Friston2010 §3].
- Standard KL divergence: $\mathrm{KL}(q \| p) = \mathbb{E}_q[\log q] - \mathbb{E}_q[\log p] = -H(q) - \mathbb{E}_q[\log p]$. So $-\mathbb{E}_q[\log p] = \mathrm{KL}(q\|p) + H(q)$ identically [CoverThomas2006 §2.5].
- Multi-agent generalizations of FEP exist [Friston2017Graphical, Ramstead2020] but the agent-environment Markov blanket is the canonical separator; observations are not agents in that literature.
- The Markov-blanket formalism in FEP [Friston2013, Pearl1988 §3.2] partitions states into internal, sensory, active, external. Sensory states are passive carriers of external information; they are not autonomous agents in the FEP sense.
- For a Gaussian density $\mathcal{N}(\mu, \Sigma)$, the differential entropy is $H = \tfrac12\log|2\pi e\Sigma| = \tfrac12(d \log(2\pi e) + \log|\Sigma|)$ [CoverThomas2006 §8.4].

External canonical references not in repo:
- The cross-entropy and KL relationship $-\mathbb{E}_q[\log p] = H(q, p) = H(q) + \mathrm{KL}(q\|p)$ is a textbook identity [CoverThomas2006 §2.5; MacKay2003 §2.5].
- In active inference under the Laplace approximation [Friston2007 §3.1; ParrPezzuloFriston2022 §4.3], the observation term in the free energy is the expected negative log-likelihood (cross-entropy), $-\mathbb{E}_{q(s)}[\log p(o|s)]$, with $p(o|s) = \mathcal{N}(o; g(s), \Sigma_o)$ a Gaussian likelihood. The Hessian of this term in $\mu$ at the Laplace point is the observation precision $\Lambda_o = \Sigma_o^{-1}$.
- Multi-agent / hierarchical active inference [Friston2017Graphical, Hesp2021] generally treats subordinate agents as full FEP agents with their own internal model, not as fixed posterior carriers with frozen $q = p$.

## What this evidence does NOT settle

1. Is the user's intended reading — "observations as cross-scale agent communication" — actually delivered by the construction at lines 1426–1430, or is this a same-scale agent substitution with frozen dynamics? The text at line 1447 evokes a cross-scale Markov-blanket-of-sensors picture but the construction does not specify a scale for environmental agents.
2. Do environmental agents as constructed (with $q_{e_k} = p_{e_k}$, fixed Σ_o, gauge $\Omega = I$, $\beta = 1$) satisfy the manuscript's own agent definition at §`sec:agent_definition` (lines 617–626) — which requires primitive sections $q_i$ and $s_i$, derived sections $p_i$ and $r_i$ via cross-scale shadow, gauge frame $\phi_i$, and embedding in a hierarchy?
3. Under what regime is the cross-entropy substitution at line 1442 itself consistent with the rest of the framework's $\mathrm{KL}(q_i \| \Omega q_j)$ inter-agent coupling form? The cross-entropy is not a symmetric agent-coupling form; it introduces a special-case operator just for observation terms.
4. Is the fixed-covariance alternative at line 1423 / 1445 a meaningful regime in the framework, or a degenerate restriction that contradicts the framework's covariance-sector dynamics (E-step σ updates documented in CLAUDE.md)?
5. Does the gauge-fixing $\Omega_{i,e_k} = I$ on line 1431 violate the gauge-equivariance hard constraint, or is it a legitimate gauge-fixing operation that the §`sec:symmetry_breaking` explicit-symmetry-breaking reading licenses?
6. Sympy verification of K1–K3 above with attention to sign conventions: $o_k$ vs. $c_k$, whether the observation likelihood is $p(o_k|c) = \mathcal{N}(o_k; c, \Sigma_o)$ (the $o$-given-$c$ direction) or $p(c|o_k)$ (the $c$-given-$o$ direction), and whether the manuscript's mean-gradient statement at line 1435 is sign-correct under its own convention.

The teams should resolve these.
