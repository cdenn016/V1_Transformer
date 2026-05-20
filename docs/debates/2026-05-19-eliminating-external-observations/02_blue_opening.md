# Blue Opening — eliminating-external-observations

## Steelman (opposing position)

The strongest attack runs as follows. The section's title and the parsimony claim at line 1447 advertise a "self-contained framework" with "only agents and their couplings," but the construction that actually backs this claim is mathematically partial in two respects and ontologically partial in a third. Mathematically, the KL form agrees with the standard active-inference observation term only at the mean-gradient level; the covariance sector differs by $-\tfrac12 \Sigma_i^{-1}$, and "full variational equivalence" is bought only by a substitution to cross-entropy that breaks the framework's otherwise uniform $\mathrm{KL}(q_i \| \Omega_{ij} q_j)$ inter-agent coupling form. Ontologically, the constructed environmental agents have $q_{e_k} = p_{e_k}$, $\beta_{i,e_k}=1$, $\Omega_{i,e_k}=I$ — they are frozen sensor carriers, not full agents in the §`sec:agent_definition` sense (lines 617–626 specify two primitive sections $q_i, s_i$, two derived sections $p_i, r_i$ via a cross-scale shadow, and a gauge frame $\phi_i$). And the user's intended reading — "observations as cross-scale agent communication" — is gestured at by the closing sentence at line 1447 but is not realized by the construction at lines 1426–1430, which is scale-local: $q_{e_k}(c)$ lives on the same fiber as $q_i(c)$, with no $(s+1)$ or $(s-1)$ index.

## Position

Both C1 and C2 hold under a calibrated reading. C1 (mathematical correctness of the explicit gradient computations at lines 1431–1445) is verified exactly by sympy: the closed-form Gaussian KL matches [BleiKuckelbirgJordan2017] and [AmariNagaoka2000 Ch. 2], the mean gradients of $\mathrm{KL}(q_i \| q_{e_k})$ and of $-\mathbb{E}_{q_i}\log p(o_k|c)$ agree at $\Lambda_o(\mu_i - c_k)$ once $o_k$ is identified with the noumenal location $c_k$, the covariance-gradient discrepancy is exactly $-\tfrac12 \Sigma_i^{-1}$, and the cross-entropy identity $-\mathbb{E}_q[\log q_{e_k}] = \mathrm{KL}(q_i \| q_{e_k}) + H(q_i)$ is the textbook [CoverThomas2006 §2.5] decomposition. C2 (motivational adequacy) holds because the proposition at lines 1415–1418 is appropriately conditional — it explicitly states that mean-gradient equivalence holds for any $\Sigma_o > 0$ and that full variational equivalence requires either the cross-entropy substitution or the fixed-covariance restriction. That conditional scoping is responsible rather than overclaiming. The cross-scale Markov-blanket-of-sensors picture at line 1447 is supported by the manuscript's pre-existing agent-hierarchy machinery at §`sec:agent_definition` lines 617–630, with the section under debate furnishing the scale-local specialization that the cross-scale generalization composes from.

## Evidence

### Sympy verification of the three load-bearing identities

Executed sympy session (2D diagonal-covariance specialization, fully symbolic):

```
KL(q_i || q_{e_k}) closed form yielded by [BleiKuckelbirgJordan2017]:
  KL = (1/2)[ (mu_i - c_k)^T Lambda_o (mu_i - c_k)
              + tr(Lambda_o Sigma_i) + log(|Sigma_o|/|Sigma_i|) - d ]

dKL/dmu1 = (mu1 - c1)/so1
dKL/dmu2 = (mu2 - c2)/so2
                                                # this is Lambda_o (mu_i - c_k)

-E_q[log p(o_k|c)] under p(o|c) = N(o; c, Sigma_o):
d(-ELL)/dmu1 = (mu1 - ok1)/so1
d(-ELL)/dmu2 = (mu2 - ok2)/so2

After identifying o_k = c_k:
  dKL/dmu - d(-ELL)/dmu = 0   componentwise.    [K1 match]

dKL/ds1 = (s1 - so1)/(2 s1 so1)  =  1/(2 so1) - 1/(2 s1)
d(-ELL)/ds1 = 1/(2 so1)
Discrepancy dKL/ds1 - d(-ELL)/ds1 = -1/(2 s1)   [exactly -1/2 Sigma_i^{-1}, K2 match]

CE - (KL + H(q_i)) = 0                          [K3 match, Cover-Thomas §2.5]
```

This verifies the three quantitative claims of the manuscript section verbatim: the mean-gradient identity at line 1435, the covariance-gradient discrepancy $-\tfrac12 \Sigma_i^{-1}$ at lines 1436–1439, and the cross-entropy decomposition at lines 1442–1443.

### Canon citations grounding the form

The Gaussian KL closed form at line 1433 of the manuscript matches the textbook expression in [BleiKuckelbirgJordan2017] and [AmariNagaoka2000 Ch. 2 Eq. on closed-form Gaussian KL], as recorded in `external_canon_math.md` §1 ("Closed-form KL between Gaussians"). The cross-entropy / KL identity at line 1443 is [CoverThomas2006 §2.5] (also [MacKay2003 §2.5]). The Gaussian differential entropy $H(q) = \tfrac12 \log |2\pi e \Sigma|$ at line 1445 matches [CoverThomas2006 §8.4].

### Standard active-inference form recovered under the cross-entropy substitution

`external_canon_inference.md` §2 records the canonical active-inference form: the observation term in the free energy under a Gaussian likelihood $p(o|s) = \mathcal{N}(o; g(s), \Sigma_o)$ is the expected negative log-likelihood, $-\mathbb{E}_{q(s)}[\log p(o|s)]$ [Friston2007 §3.1; ParrPezzuloFriston2022 §4.3]. The cross-entropy substitution at line 1442 is therefore not a contrivance to patch up the construction; it returns the framework to the literal canonical observation term [Friston2010, ParrPezzuloFriston2022 Ch. 2]. The substitution is a coordinate change from KL to cross-entropy, related by the q-entropy $H(q_i)$ as the sympy session confirmed. The KL coupling form, with its additional $-\tfrac12 \log |\Sigma_i|$ entropy term inside KL, generates the documented additional $-\tfrac12 \Sigma_i^{-1}$ "pull toward larger covariance" — and the section flags exactly this pull and identifies it as the source of the discrepancy at line 1439.

### Dirac caveat is mathematically correct and recovers the FEP limit honestly

The line-1425 caveat states $\mathrm{KL}(q_i \| \delta(c - c_k)) = +\infty$ for non-degenerate $q_i$, with reference to Eq. `eq:dirac_kl` at line 1612. This is the standard absolute-continuity statement: KL is finite only when the first argument is absolutely continuous with respect to the second, and a non-degenerate $q_i$ is not absolutely continuous with respect to a point mass [external_canon_math.md §1, KullbackLeibler1951]. The construction therefore uses a finite-precision Gaussian sensor with $\Sigma_o > 0$ — which then recovers the standard active-inference small-precision dynamics in the $\Sigma_o \to 0$ limit, with the divergent likelihood term subtracted via standard regularization. The handling is correct.

### Gauge-fixing $\Omega_{i,e_k} = I$ is licensed, not hidden, by `sec:symmetry_breaking`

Line 1431 of the section states explicitly that "$\Omega_{i,e_k} = I$ implicitly, identifying the sensor's gauge frame with the receiving agent's; this gauge-fixing is the implicit content of the explicit symmetry breaking discussed in Section sec:symmetry_breaking." Section `sec:symmetry_breaking` at line 1484 then says explicitly: "environmental agents enter the free energy with fixed gauge frames, and these fixed frames play the role of an external source field analogous to a Zeeman term in a ferromagnet." The asymmetry between env agents and full agents is not eliminated and not concealed — it is named, given a physical analogue (Zeeman field), and identified as the explicit-symmetry-breaking mechanism of the framework. The gauge-equivariance hard constraint is not violated because env agents are declared to be the source of an explicit symmetry-breaking perturbation, not symmetric agents whose frames must be averaged over.

### Conditional scoping at the proposition statement is precise

The proposition at lines 1415–1418 reads, verbatim: "mean-gradient flow agrees with that of $\mathcal{F}_{\text{obs}}$, $\partial_{\mu_i}\mathcal{F}_{\text{obs}} = \partial_{\mu_i}\mathcal{F}_{\text{agent}}$, for any environmental sensory precision $\Sigma_o > 0$. Full variational equivalence at the level of the joint $(\mu_i, \Sigma_i)$ gradient flow, $\delta \mathcal{F}_{\text{obs}}/\delta q_i = \delta \mathcal{F}_{\text{agent}}/\delta q_i$, requires an additional restriction: either replacing the environmental KL by a cross-entropy coupling $-\mathbb{E}_{q_i}[\log q_{e_k}]$, or restricting to fixed-covariance dynamics." This is conditional, calibrated language. The mean-gradient claim is unconditional; the full-variational claim is conditional and the conditions are stated. This is not the overclaim a red-team attack would target — it is its opposite, an explicit scope statement.

### Cross-scale reading: licensed by the manuscript's own hierarchy machinery

The user's intended reading — observations as agent communication "potentially across various scales" — is supported by composing two pieces of existing manuscript machinery with the section under debate as the scale-local specialization. Section `sec:agent_definition` at lines 617–630 defines an agent at scale $s$ with cross-scale shadow $p_i^{(s)} = \Omega_{i,I}[q_I^{(s+1)}]$ from a level-$(s{+}1)$ meta-agent. Section `sec:cross_scale_shadows` at line 536 furnishes the explicit cross-scale relation. The Markov-blanket-of-sensors sentence at line 1447 ("cells, organs, etc, are themselves composed of sensory Markov blankets — receptors, proteins, molecules, etc — onward down to single bits") is the heuristic content of this composition: at scale $s$, an env agent $e_k$ for a receiving agent at scale $s+1$ is realized as a (possibly aggregated) constituent at scale $s$, whose belief $q_{e_k}$ is sharp around the noumenal value the receiving agent is to read. The mean-gradient analysis is scale-local and is therefore intact at any scale at which one applies the construction. The cross-scale reading is delivered by composition, not by a separate cross-scale construction; the section under debate furnishes the scale-local kernel.

## Falsification conditions

This defense fails — i.e., the claim does not hold under the evidence — if any of the following can be established by red.

The defense of C1 fails if a sympy or finite-difference derivation produces a closed-form Gaussian KL or its derivatives that differ from the manuscript's expressions at lines 1433–1438, or if the cross-entropy decomposition at line 1443 fails to reduce to $\mathrm{KL}(q_i\|q_{e_k}) + H(q_i)$ when computed directly. The sympy session above closes both possibilities for the diagonal specialization; a red counter-derivation must produce a contradiction with that session or generalize to a non-diagonal failure mode that the section claims to cover but does not.

The defense of C2 fails if red can demonstrate that the section's conditional scoping at lines 1415–1418 is somewhere in the text overridden or contradicted by an unconditional claim of equivalence that the math does not support. The only candidate sentence is the line-1447 parsimony statement ("requires only agents and their couplings, with no external reality providing special inputs"); if this sentence is read as unconditional, it is in tension with the math, and the manuscript would need a parenthetical that ties the parsimony statement back to the cross-entropy or fixed-covariance regime. This is the strongest available attack on C2; the steelman version of blue concedes that the parsimony sentence at line 1447 is partially unconditional in its current form even though the immediately preceding sentence at the start of the paragraph states the cross-entropy condition explicitly.

The cross-scale reading fails as a defense of C2 if red can show that the framework's cross-scale shadow at §`sec:cross_scale_shadows` (line 536) is not invoked anywhere in the construction at lines 1426–1430 — i.e., that the construction does not specify a scale $s$ for $e_k$ and does not connect $q_{e_k}$ to a cross-scale shadow. Inspection of the text confirms that the construction is scale-local; the cross-scale reading is delivered only by composition with the §`sec:agent_definition` machinery, not by the construction itself. Red can argue that the line-1447 sentence asks the reader to make that composition silently, which is a valid editorial criticism. Blue concedes this as an editorial finding; it does not falsify the mathematical content but it is a fair criticism of the framing.

The env-agent definition fails as fully consistent with §`sec:agent_definition` if red establishes that $q_{e_k} = p_{e_k}$ violates the cross-scale derived-prior relation $p_i = \Omega_{i,I}[q_I^{(s+1)}]$ at line 622. The honest reading is that env agents are a special sensor-class case in which the cross-scale shadow degenerates: $\Omega_{i,e_k} = I$ and $p_{e_k} = q_{e_k}$ make $e_k$ a degenerate sub-agent. This is a real asymmetry between env agents and constituent agents and red can press it. Blue concedes that env agents are not full agents in the dynamical sense — they do not minimize their own F-contribution because $q = p$ and $\beta = 1$ are fixed by construction — and treats them as a sensor sub-class licensed by `sec:symmetry_breaking` as the explicit-symmetry-breaking source rather than as full constituent agents.

## Honest concessions stated upfront

Three concessions on which blue does not contest red:

First, the env-agent construction at lines 1426–1430 places environmental agents in a degenerate sub-class of the §`sec:agent_definition` agent definition: $q_{e_k} = p_{e_k}$, $\beta_{i,e_k} = 1$, $\Omega_{i,e_k} = I$, frozen. They are not "agents in the same dynamical sense" as constituent agents in the framework; line 1409 of the manuscript asserts they are "subject to the same information-geometric dynamics," but the construction itself fixes them — they do not minimize free energy of their own initiative. The honest reading is that env agents are a sensor sub-class licensed by the explicit-symmetry-breaking story of `sec:symmetry_breaking`, not full constituent agents. The text on line 1409 is mildly stronger than the construction warrants.

Second, the cross-entropy substitution at line 1442 introduces a coupling form ($-\mathbb{E}_{q_i}[\log q_{e_k}]$) that differs from the framework's otherwise uniform inter-agent $\mathrm{KL}(q_i \| \Omega_{ij}q_j)$ coupling form. The substitution-by-name is honest about its scope (a recovery of standard FEP observation dynamics in the covariance sector), but it does mean that the env-agent / observation channel uses a different operator from the agent-agent channel.

Third, the cross-scale reading at line 1447 ("cells, organs, ... onward down to single bits") is the user's intended interpretation, and it is supported by composition with the framework's hierarchy machinery at §`sec:agent_definition` and §`sec:cross_scale_shadows`, but it is not delivered by a separate explicit construction within the section. A red-team finding that the section under debate does not itself instantiate a cross-scale env-agent (e.g., $q_{e_k}^{(s)}$ with $s$ distinct from the receiving agent's scale) is correct and fair.

These three concessions do not, jointly, falsify the section's claim. The proposition at lines 1415–1418 is conditional and the conditions are stated; the math at lines 1431–1445 is exact (sympy-verified); the gauge-fixing is named and licensed by `sec:symmetry_breaking`; the cross-scale reading is delivered by composition with established framework machinery. The claim stands.
