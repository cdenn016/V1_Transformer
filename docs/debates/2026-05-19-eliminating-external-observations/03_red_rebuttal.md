# Red Rebuttal — eliminating-external-observations

## Concession

Red grants C1. Blue's sympy session at `02_blue_opening.md` lines 17–38 reproduces the three load-bearing closed-form identities verbatim against the manuscript at `Participatory_it_from_bit.tex:1431–1445`: the Gaussian KL form [BleiKuckelbirgJordan2017; AmariNagaoka2000 Ch. 2], the mean gradient $\Lambda_o(\mu_i - c_k)$, the covariance discrepancy $-\tfrac12 \Sigma_i^{-1}$, and the cross-entropy decomposition $-\mathbb{E}_{q_i}[\log q_{e_k}] = \mathrm{KL}(q_i\|q_{e_k}) + H(q_i)$ [CoverThomas2006 §2.5]. Red's opening did not challenge C1 and does not challenge it now. The math is right; what the math licenses is the entire dispute.

Red also concedes that the proposition statement at `Participatory_it_from_bit.tex:1415–1423` is, in isolation, conditional and calibrated: it scopes mean-gradient equivalence to "any $\Sigma_o > 0$" and full variational equivalence to the cross-entropy substitution or fixed-covariance restriction. Blue is correct that that paragraph is responsible.

The concession ends there. C2 fails on three independent grounds that blue's opening does not close.

## Core attack

### 1. Blue's "delivered by composition" defense fails its own falsification condition

Blue's `02_blue_opening.md` lines 62–64 defends the cross-scale reading by composition with §`sec:agent_definition` lines 617–630. The manuscript's own agent definition at `Participatory_it_from_bit.tex:622` requires the prior to be a cross-scale shadow: $p_i = \Omega_{i,I}[q_I^{(s+1)}]$, where $I$ "indexes the level-$(s{+}1)$ meta-agent that envelopes agent $i$ as a constituent." The construction at `Participatory_it_from_bit.tex:1428` sets $p_{e_k}(c) = q_{e_k}(c)$ by hand. There is no $I$, no scale index $s$, no level-$(s{+}1)$ meta-agent whose belief is being transported. Setting $p_{e_k} = q_{e_k}$ short-circuits the shadow relation that the framework's own agent definition makes constitutive of being an agent.

Blue's own falsification condition at `02_blue_opening.md` lines 73–74 reads: "The cross-scale reading fails as a defense of C2 if red can show that the framework's cross-scale shadow at §`sec:cross_scale_shadows` (line 536) is not invoked anywhere in the construction at lines 1426–1430." Inspection of `Participatory_it_from_bit.tex:1426–1430` confirms the shadow is not invoked. Blue concedes this as "a fair editorial criticism." Red presses further: the shadow is not editorial decoration, it is the manuscript's definition of what a prior is (line 622). Without the shadow, $p_{e_k}$ is not a prior in the framework's sense — it is an identification by stipulation. "Delivered by composition" is therefore not delivered at all; the composition that blue invokes requires a piece that the construction explicitly omits.

### 2. The section's bookends override the middle's conditional language

Blue's defense at `02_blue_opening.md` lines 58–60 rests on the conditional scoping of the proposition. The subsection title at `Participatory_it_from_bit.tex:1394` is unconditional: "Eliminating External Observations: A Self-Contained Framework." The closing parsimony sentence at `Participatory_it_from_bit.tex:1447` is unconditional: "more parsimonious ontologically — it requires only agents and their couplings, with no external reality providing special inputs." The bookend pair is what the user and any non-author reader takes as the section's thesis. The conditional middle is buried between two unconditional declarations.

Blue concedes exactly this point at `02_blue_opening.md` line 72: "the steelman version of blue concedes that the parsimony sentence at line 1447 is partially unconditional in its current form even though the immediately preceding sentence at the start of the paragraph states the cross-entropy condition explicitly." Red elevates this from a partial concession to a falsification of C2: a section whose title and closing sentence assert self-containment without an external-reality channel, while its body achieves only conditional mean-gradient equivalence (and full equivalence only under an operator substitution that breaks the framework's homogeneous KL coupling), is overclaiming at exactly the points where readers form their impression of what was established.

### 3. The Zeeman analogy is honest about externalism — which falsifies self-containment

Blue's defense of the gauge-fixing at `02_blue_opening.md` lines 54–56 invokes the explicit-symmetry-breaking story at `Participatory_it_from_bit.tex:1484`: "environmental agents enter the free energy with fixed gauge frames, and these fixed frames play the role of an external source field analogous to a Zeeman term in a ferromagnet." A Zeeman term in a ferromagnet is, by construction in statistical physics, an external magnetic field $H$ coupled to the magnetization via $-\mathbf{H}\cdot\mathbf{M}$ [Kittel2005 §12; Goldenfeld1992 §5.6]. The Zeeman field is the canonical example of an external source; explicit-symmetry-breaking by an external source is precisely the non-spontaneous case.

The manuscript itself spells the externalism out at `Participatory_it_from_bit.tex:1495`: "The observation term … explicitly breaks gauge invariance by coupling agent beliefs to external data represented in specific gauge frames." And at line 1491 (figure caption): "environmental agents carry fixed gauge frames that act as an external source selecting a preferred orbit representative." Blue's licensing mechanism therefore licenses the gauge-fixing by routing it through a section that calls the env agents an external source. The honesty of the Zeeman analogy at §`sec:symmetry_breaking` is the same honesty that makes the line-1394 title and the line-1447 parsimony claim untenable: if env agents act as an external source field, the framework is not self-contained, it is self-contained-plus-an-external-source.

### 4. The cross-entropy substitution breaks the framework's homogeneous coupling form

Blue concedes at `02_blue_opening.md` line 84 that the cross-entropy substitution at `Participatory_it_from_bit.tex:1442` "introduces a coupling form … that differs from the framework's otherwise uniform inter-agent $\mathrm{KL}(q_i \| \Omega_{ij}q_j)$ coupling form." The canonical free energy in `CLAUDE.md` and at `eq:free_energy_functional_final` couples every agent pair via $\beta_{ij}\mathrm{KL}(q_i \| \Omega_{ij}q_j) + \tau\beta_{ij}\log(\beta_{ij}/\pi_{ij})$ — a homogeneous KL operator. Replacing this with cross-entropy for env-agent terms only means the framework now contains two coupling operators: KL between full agents, cross-entropy between full agent and env agent. The line-1447 claim of "only agents and their couplings" then must read "agents and *two distinct kinds of* couplings," one of which (cross-entropy) is exactly the standard FEP observation term [Friston2010 §3; ParrPezzuloFriston2022 §4.3].

Blue's defense at `02_blue_opening.md` line 48 frames this as "a coordinate change from KL to cross-entropy, related by the q-entropy $H(q_i)$." A coordinate change does not change the operator; it changes the parameterization. KL and cross-entropy differ by $H(q_i)$, which depends on $\Sigma_i$ — that is the entire content of the covariance-gradient discrepancy $-\tfrac12 \Sigma_i^{-1}$ at `Participatory_it_from_bit.tex:1438`. The two operators have different functional derivatives in $\Sigma_i$ and therefore produce different covariance dynamics. They are not coordinate-related; they are functionally distinct. Calling them a coordinate change conceals the structural break that the section requires.

### 5. Frozen dynamics is not "communication between agents"

Blue concedes at `02_blue_opening.md` line 82 that "env agents are not full agents in the dynamical sense — they do not minimize their own F-contribution because $q = p$ and $\beta = 1$ are fixed by construction" and treats them as "a sensor sub-class." The user's framing in `00_claim.md` lines 15–17 is that "observations are then communication between agents potentially across various scales." Communication is bidirectional information exchange. The construction at `Participatory_it_from_bit.tex:1426–1430` is unidirectional: $q_{e_k}$ is a fixed Gaussian centered at the noumenal location $c_k$ with precision $\Sigma_o$, and $p_{e_k} = q_{e_k}$ means $e_k$ has zero E-step dynamics (it is at its prior already). The receiving agent $i$ updates in response to $e_k$; $e_k$ never updates in response to $i$. That is the formal structure of observation, not communication. Re-labeling the carrier as an "agent" does not turn observation into communication; it relabels the source.

## Defense

Red's opening attacked the load-bearing assumption that the section's title and closing parsimony claim are supported by the body. The five points above are not separate; they assemble into one argument. The section's title and closing sentence advertise self-containment without external inputs. The body achieves this only by (a) introducing env agents as a degenerate sub-class with $q=p$, $\beta=1$, $\Omega=I$ — none of which are "subject to the same information-geometric dynamics as all other agents" (`Participatory_it_from_bit.tex:1409`, which blue concedes overstates), (b) gauge-fixing those agents under an explicit-symmetry-breaking story that the manuscript's own §`sec:symmetry_breaking` names as an external source field (`Participatory_it_from_bit.tex:1484, 1491, 1495`), (c) recovering full variational equivalence in the covariance sector only under a cross-entropy substitution that is the standard FEP observation operator and breaks the framework's homogeneous KL coupling form (`Participatory_it_from_bit.tex:1442`), and (d) omitting the cross-scale shadow relation that the framework's own agent definition (`Participatory_it_from_bit.tex:622`) requires for any agent's prior.

Each of (a)–(d) is conceded by blue separately. Red's claim is that they cannot all be conceded without the section's bookend claims being falsified. Blue's defense strategy is to grant each piece, then assemble the pieces into a "self-contained framework conditioned on these substitutions." That is a different and weaker claim than the section actually makes at lines 1394 and 1447.

Red's falsification condition from the opening was: "if env agents enter under explicit symmetry breaking with fixed external frames, the framework is not self-contained." This is exactly what `Participatory_it_from_bit.tex:1484` says env agents do. Blue cites that sentence as the licensing mechanism for the gauge-fixing. The licensing mechanism falsifies the self-containment claim by its own language: "external source field analogous to a Zeeman term."

The section under debate is mathematically correct (C1 holds) and motivationally overclaiming (C2 fails). A faithful rewrite would replace the unconditional title and closing sentence with the body's conditional content — that under the cross-entropy substitution, the env-agent formalism recovers the standard FEP observation term, and that this is a re-expression of the FEP observation channel using the framework's notation, rather than its elimination. That rewrite would be honest. The current text is not.

## Citations

- `Participatory_it_from_bit.tex:1394` — subsection title (unconditional "self-contained framework").
- `Participatory_it_from_bit.tex:1409` — env agents declared "subject to the same information-geometric dynamics" (overstatement conceded by blue).
- `Participatory_it_from_bit.tex:1415–1423` — proposition statement (conditional language).
- `Participatory_it_from_bit.tex:1426–1430` — concrete construction; $p_{e_k} = q_{e_k}$; no $(s+1)$ scale index.
- `Participatory_it_from_bit.tex:1431` — gauge-fixing $\Omega_{i,e_k} = I$ routed through §`sec:symmetry_breaking`.
- `Participatory_it_from_bit.tex:1438` — covariance gradient with $-\tfrac12 \Sigma_i^{-1}$ discrepancy.
- `Participatory_it_from_bit.tex:1442–1443` — cross-entropy substitution as separate operator.
- `Participatory_it_from_bit.tex:1447` — closing parsimony sentence ("no external reality providing special inputs"; "onward down to single bits").
- `Participatory_it_from_bit.tex:1484` — Zeeman analogy: env agents as "external source field."
- `Participatory_it_from_bit.tex:1491` — figure caption: "external source selecting a preferred orbit representative."
- `Participatory_it_from_bit.tex:1495` — observation term "explicitly breaks gauge invariance by coupling agent beliefs to external data."
- `Participatory_it_from_bit.tex:622` — agent definition requires $p_i = \Omega_{i,I}[q_I^{(s+1)}]$; cross-scale shadow not invoked at 1428.
- `external_canon_inference.md` §2; [Friston2010 §3]; [ParrPezzuloFriston2022 §4.3] — cross-entropy as the canonical FEP observation operator.
- [Kittel2005 §12; Goldenfeld1992 §5.6] — Zeeman term as external source field.
- [CoverThomas2006 §2.5] — $-\mathbb{E}_q[\log p] = \mathrm{KL}(q\|p) + H(q)$ identity (operator distinctness in $\Sigma$-sector).
