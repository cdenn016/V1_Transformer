# Evidence Pack — pifb-discussion-language-cognition

## Manuscript references

### The Language and Cognition subsection under debate (Discussion §3200-3225)

- `Participatory_it_from_bit.tex:3200` — subsection title: "Implications for Language and Cognition"
- `Participatory_it_from_bit.tex:3202` — subsubsection: "Language as Multi-Agent Coordination"
- `Participatory_it_from_bit.tex:3204` — opening: "language may be at root a multi-agent coordination problem...each token functions as an autonomous agent maintaining probabilistic beliefs $q_i$..." (now with the Ramstead2019 enactive-inference citation added per the constructivism debate)
- `Participatory_it_from_bit.tex:3206` — "This resembles what linguists call pragmatic communication: agents coordinate to jointly minimize uncertainty and establish shared meaning. The framework thus provides a mathematical formalization of informal ideas about language as cooperative inference."
- `Participatory_it_from_bit.tex:3208` — subsubsection: "The Gauge Curvature Conjecture" with label sec:gauge_curvature_conjecture
- `Participatory_it_from_bit.tex:3211` — load-bearing Regime II preamble: "*Regime II conditional.* The conjecture below is meaningful only in the Regime II extension of the framework...In Regime I the curvature $F_{\mu\nu}$ vanishes identically by the Maurer–Cartan identity (Lemma thm:vanishing_holonomy), so a curvature-minimization principle has no content under the present implementation. The conjecture is therefore a conditional prediction: *if* the Regime II edge-relaxed cocycle of Section sec:discrete_regime_ii is realized for natural language, *then* linguistic evolution should drive the Wilson observable $W_{ijk} = \operatorname{Re}\operatorname{Tr}(H_{ijk})$ of (eq:wilson_observable) toward its maximal value $K$..."
- `Participatory_it_from_bit.tex:3213` — conjecture: "We propose a potentially falsifiable conjecture: in the Regime II extension, language is a gauge theory and linguistic evolution is driven by minimization of gauge field curvature."
- `Participatory_it_from_bit.tex:3217` — natural-language manifestations: "gauge curvature manifests as semantic ambiguity and context-dependence. High curvature regions correspond to words or phrases whose meaning depends strongly on discourse path...Garden path sentences represent local curvature spikes...Creolization may provide an example: when pidgins develop into creoles with systematic grammar, we might observe rapid curvature reduction as the language acquires consistent structure."
- `Participatory_it_from_bit.tex:3219` — syntax-from-attention claim: "Syntactic structure itself might emerge from the geometry of information flow. Strong attention weights $\beta_{ij}$ indicate high information coupling between tokens; thresholding this coupling matrix $A_{ij} = \mathbb{I}[\beta_{ij} > \theta]$ may yield dependency graphs that closely match syntactic parse trees. This would then suggest that syntax is not imposed by explicit rules but emerges as the skeleton of efficient information transport. Grammar would then be an information-theoretic necessity rather than cultural convention - it is what 'glues' agents into higher order meta-agents (cultures, villages, etc)."
- `Participatory_it_from_bit.tex:3221` — generalization: "Beyond language, we conjecture that any system of interacting information-processing agents evolves to minimize gauge curvature as a pre-requisite for meta-agent emergence. Social systems develop shared norms and conventions...Scientific communities establish terminology, results, and notation that flatten communication gauge structure...Neural systems meanwhile, may wire themselves to minimize curvature in neural information flow...Additionally, market economies develop institutions and contracts that reduce transaction curvature...This principle might apply to any emergent informational system or it may only be a pretty way to describe something more ephemeral."
- `Participatory_it_from_bit.tex:3223` — falsifiability statement: "This gauge curvature minimization principle represents the strongest testable prediction of our framework. Unlike claims about emergent spacetime or consciousness, curvature in linguistic or social systems is measurable. One can compute field strength tensors from observed communication patterns, track curvature changes during language evolution using historical corpora, measure curvature differences between creoles and pidgins, or quantify curvature reduction in developing scientific terminology. These are concrete, falsifiable predictions that do not require solving the Lorentzian signature problem or establishing dimensional analysis. If gauge curvature fails to correlate with measures of linguistic efficiency, communicative success, or evolutionary fitness, the framework's core claim about information geometry governing real systems would be falsified."
- `Participatory_it_from_bit.tex:3225` — structural-parallel disclaimer: "The structural parallel is suggestive without being a unification...the framework is not a unification of physical forces with cognition or language; it is a single mathematical scaffolding that admits applications in several domains."

### Cross-referenced manuscript machinery

- `Participatory_it_from_bit.tex:sec:connection_forms` (around 797) — connection forms and parallel transport
- `Participatory_it_from_bit.tex:sec:discrete_regime_ii` (around 828) — discrete Regime II edge-relaxed cocycle
- `Participatory_it_from_bit.tex:thm:vanishing_holonomy` — the Maurer-Cartan identity for Regime I
- `Participatory_it_from_bit.tex:eq:wilson_observable` — Wilson observable definition

## Canon excerpts (teams should expand)

### BERTology / attention-as-syntax canon (relevant to 3219)

- **Clark, K., Khandelwal, U., Levy, O., Manning, C. D. (2019)**, "What does BERT look at? An analysis of BERT's attention," *ACL BlackboxNLP*. The canonical empirical attention-vs-syntax analysis. Shows certain attention heads track specific syntactic relations (e.g., direct objects, prepositional phrases). Already in references.bib (entries `clark2019does` and a duplicate `Clark2019`).
- **Hewitt, J., Manning, C. D. (2019)**, "A Structural Probe for Finding Syntax in Word Representations," *NAACL*. The structural-probe paper: BERT embeddings linearly encode parse-tree distance.
- **Vig, J. (2019)**, "A Multiscale Visualization of Attention in the Transformer Model," *ACL Demo*. Attention visualization tool.
- **Voita, E., Talbot, D., Moiseev, F., Sennrich, R., Titov, I. (2019)**, "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned," *ACL*. Shows attention head specialization for syntactic functions.
- **Rogers, A., Kovaleva, O., Rumshisky, A. (2020)**, "A Primer in BERTology: What we know about how BERT works," *TACL*. Comprehensive BERTology survey.

### Linguistic evolution / grammar canon (relevant to 3217 "creolization")

- **Bickerton, D. (1984)**, "The language bioprogram hypothesis," *Behavioral and Brain Sciences* 7, 173-188. Canonical creolization paper.
- **McWhorter, J. H. (2001)**, "The world's simplest grammars are creole grammars," *Linguistic Typology* 5(2-3), 125-166.
- **Christiansen, M. H., Chater, N. (2008)**, "Language as shaped by the brain," *Behavioral and Brain Sciences* 31, 489-509. Language-as-evolutionary-adaptation thesis.
- **Kirby, S. (2017)**, "Culture and biology in the origins of linguistic structure," *Psychonomic Bulletin & Review* 24(1), 118-137.
- **Hawkins, J. A. (2004)**, *Efficiency and Complexity in Grammars*, Oxford. Information-theoretic / efficiency-based account of grammar.
- **Piantadosi, S. T., Tily, H., Gibson, E. (2011)**, "Word lengths are optimized for efficient communication," *PNAS* 108(9), 3526-3529. Information-theoretic constraints on word length.
- **Piantadosi, S. T. (2014)**, "Zipf's word frequency law in natural language: A critical review and future directions," *Psychonomic Bulletin & Review* 21(5), 1112-1130.
- **Gibson, E., Futrell, R., Piantadosi, S. T., Dautriche, I., Mahowald, K., Bergen, L., Levy, R. (2019)**, "How efficiency shapes human language," *Trends in Cognitive Sciences* 23(5), 389-407. Information-theoretic shaping of language structure.

### Gauge-theoretic / category-theoretic linguistics canon

- **Coecke, B., Sadrzadeh, M., Clark, S. (2010)**, "Mathematical Foundations for a Compositional Distributional Model of Meaning," *Lambek Festschrift*. Categorical-compositional approach to semantics — not gauge-theoretic but mathematically structural.
- **Smolensky, P. (1990)**, "Tensor product variable binding and the representation of symbolic structures in connectionist systems," *Artificial Intelligence* 46, 159-216. Foundational connectionist-symbolic approach.

### Free-energy / active-inference linguistics canon (already cited)

- **Friston, K., Frith, C. (2015)**, "Active inference, communication and hermeneutics," *Cortex* 68, 129-143. Active-inference treatment of communication.
- **Ramstead, M. J. D., Kirchhoff, M. D., Friston, K. J. (2019)** — already cited at 3204 (recent constructivism patch).

## What this evidence does NOT settle

1. Whether the syntax-from-attention claim at 3219 should engage with the BERTology empirical literature (Clark et al. 2019, Hewitt-Manning 2019, Voita et al. 2019, Rogers et al. 2020) that has already studied this correspondence empirically. The manuscript writes "thresholding this coupling matrix...may yield dependency graphs that closely match syntactic parse trees" as a prospective hypothesis, when this has been empirically investigated for years. Either cite the BERTology work or qualify the claim relative to existing results.
2. Whether the "grammar as information-theoretic necessity rather than cultural convention" claim at 3219 is in tension with the sociolinguistic literature (Bickerton 1984, McWhorter 2001) showing grammar varies contingently across creole formation, language contact, and language change. The manuscript's framing as "necessity" overstates against this canon.
3. Whether the "creolization" claim at 3217 ("we might observe rapid curvature reduction") is testable in practice. It depends on having a quantitative operationalization of curvature for natural-language corpora, which the framework has not implemented. The Wilson observable at 3211 is "well-defined for any learned connection field $\delta_{ij}$" but requires the Regime II extension and is "an index-space invariant of the connection tensor rather than a property of the data flow" for autoregressive attention — limiting practical testability.
4. Whether the "strongest testable prediction" framing at 3223 is correct. If Regime II is not currently implemented and the conjecture is meaningful only in Regime II, then the prediction is prospective — testable in principle, not testable now. The "strongest" superlative may overstate.
5. Whether the generalization at 3221 to "social systems, scientific communities, neural systems, market economies" is metaphor extension or substantive prediction. The closing self-tag "this principle might apply to any emergent informational system or it may only be a pretty way to describe something more ephemeral" is honest about this — but the prose preceding the tag asserts specific predictions about each domain without primary-source backing.
6. Whether the Friston-Frith 2015 "active inference communication" paper should be cited as relevant active-inference linguistics literature.

Teams should verify points 1-5 against the BERTology and language-evolution canon. Point 6 is a one-citation editorial improvement.
