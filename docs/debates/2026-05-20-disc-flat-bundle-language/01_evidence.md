# Evidence Pack — disc-flat-bundle-language

## Manuscript references

- `Attention/GL(K)_attention.tex:2221` — section header `\subsection{The Flat Bundle Limit and the Geometry of Language}`.
- `Attention/GL(K)_attention.tex:2224` — opening sentence: "both the GL(K) gauge transformer and standard transformers operate in the flat bundle regime with trivially vanishing holonomy"; cocycle $\Omega_{ij}\Omega_{jk}\Omega_{ki} = I$ by construction.
- `Attention/GL(K)_attention.tex:2226` — **hypothesis statement (working hypothesis, not formal conjecture):** "The compositional-semantic component of natural language---the semantic content that supports reliable, context-independent communication---is approximately path-independent in the sense of gauge transport. Transformers are architecturally matched to this regime."
- `Attention/GL(K)_attention.tex:2228` — three sub-claims and manuscript's own self-labeling: claims (a) and (c) "substantive and, in principle, testable given an operational criterion for compositional semantics (e.g., performance on compositional generalization benchmarks such as COGS \citep{kim2020cogs} or SCAN \citep{lake2018generalization})"; claim (b) is "a plausibility argument, not a derivation."
- `Attention/GL(K)_attention.tex:2230` — content-dependent path-independence: "the learned gauge frames $\phi_i$ provide rich, context-sensitive coordinate systems while the cocycle condition $\Omega_{ij} \Omega_{jk} = \Omega_{ik}$ ensures that the resulting transport remains flat"; bag-of-words counterexample disclaimer.
- `Attention/GL(K)_attention.tex:2232` — concession that "irony, pragmatic inference, context-dependent ambiguity, and culturally situated meaning are cases where the 'same' utterance acquires different meaning depending on the interpretive path through context," and proposed test: train a non-flat gauge architecture and measure learned holonomy on pragmatic-inference tasks.

### Background sections invoked by §8.3

- `Attention/GL(K)_attention.tex:635-668` — `\subsubsection{Vanishing Holonomy and the Flat Bundle}` (Lemma `thm:vanishing_holonomy`).
- `Attention/GL(K)_attention.tex:515-555` — `\subsubsection{GL(K) Gauge Invariance of KL Divergence}` (Theorem `thm:glk_invariance`).

## Prior debate verdicts (binding scope: do not re-litigate)

- `docs/debates/2026-05-19-subclaim-A-flat-bundle/04_verdict.md` — **BLUE_WINS** on the mathematical claim that the limit $\Omega_{ij} = \Omega$ is a well-defined specialization and the upstream theorem holds at that point. **In this debate the mathematical existence of the flat-bundle limit is not under attack.** What is under attack is whether (a) natural language is path-independent and (c) transformers' flat-bundle structure is a causal contributor to their empirical success.
- `docs/debates/2026-05-19-subclaim-C-qk-identification/04_verdict.md` and `docs/debates/2026-05-19-reduction-to-standard-transformer/04_verdict.md` — established that the reduction to standard scaled dot-product attention is approximate (key-norm cancellation is approximate; LayerNorm does not exactly cancel `‖μ_j‖²`). The flat-bundle component of the reduction was not the failing piece.

## Canon excerpts — formal semantics, compositionality, gauge theory

### Compositionality in linguistics

The Principle of Compositionality (Frege; Montague): the meaning of a complex expression is determined by the meanings of its constituents and the rules used to combine them [Montague1970 "Universal Grammar"; Partee1984].

A standard operationalization of compositional semantics:
- COGS benchmark [Kim & Linzen 2020, "COGS: A Compositional Generalization Challenge Based on Semantic Interpretation"]: tests systematic generalization to novel combinations of known primitives.
- SCAN benchmark [Lake & Baroni 2018, "Generalization without Systematicity"]: similar with simpler command-action mapping.

**No canonical formal-semantics result establishes "path-independence in the sense of gauge transport" as a property of natural language compositional semantics.** The gauge-transport framing is the user's contribution.

### What "path-independent" means in differential geometry

In a principal bundle with connection, parallel transport is path-independent **iff the connection is flat** (curvature 2-form vanishes), equivalently iff the holonomy group is trivial / contained in the identity component [Nakahara2003 §10.4; KobayashiNomizu Vol. I §II.9]. The discrete-graph analogue is that for any closed loop of edges $i \to j \to k \to i$, the composed transport $\Omega_{ij}\Omega_{jk}\Omega_{ki} = I$.

**The bundle-of-meanings analogy is metaphorical.** Formal semantics treats meaning as a function from contexts to denotations; whether this function factors through a flat-bundle structure on a token graph is not a standard linguistic question. Mapping "compositional semantics" to "flat bundle on token graph" requires either:
- An operational criterion that connects benchmark performance (COGS, SCAN) to measured holonomy of a learned transport, or
- A theoretical bridge that derives compositionality-as-path-independence from independent linguistic principles.

The manuscript supplies neither; it asserts the analogy and proposes the operational criterion as future work (`:2232`).

### Counterexamples and degenerate cases

- **Bag-of-words models** are trivially path-independent yet poor language models — explicitly conceded at `:2230`.
- **Non-compositional phenomena** (irony, idioms, anaphora, pragmatic inference, indexicality) are abundant in language and conceded at `:2232`.
- **Distinguishing compositional from non-compositional content** is not given an operational definition. Without one, the hypothesis at `:2226` is unfalsifiable as stated.

### Cocycle as a structural property of $\Omega_{ij} = \exp(\phi_i)\exp(-\phi_j)$

The cocycle $\Omega_{ij}\Omega_{jk}\Omega_{ki} = I$ holds for the user's parameterization **by construction** (the manuscript states this at `:2224`). Standard transformers have $W_Q W_K^T$ as a position-independent constant; their "transport" is trivially flat for the same reason a constant gauge field is flat. The architectural matching argument therefore needs to do real work explaining why this specific structural property of both architectures is the right one to credit for transformer success, as opposed to e.g.:
- The softmax non-linearity (which standard transformers have).
- The MLPs and pointwise activations (which the gauge VFE deliberately removes; see `:2143` no-MLP description in Table~\ref{tab:glk_results}).
- The high parameter count and embedding dimension.
- Inductive biases that have nothing to do with bundle flatness (e.g., scale, depth, positional encoding).

### What canon does and does not say about transformers and language

- [Vaswani2017] introduces transformers and attention; no claim about "flat bundles" or "compositionality of natural language."
- The cognitive-science / linguistics literature on compositional generalization in neural language models is mixed: COGS and SCAN both show LSTM/transformer models *fail* on systematic compositional generalization without specific architectural choices [Kim & Linzen 2020; Lake & Baroni 2018]. Recent work shows large transformers can be made to perform better with chain-of-thought-style prompting or auxiliary losses, but the question of whether transformers' inductive bias is "matched" to compositional semantics is empirically open.

## What this evidence does NOT settle

- Whether "compositional-semantic component" has a non-circular operational definition that would make sub-claim (a) testable.
- Whether the cocycle property of $\Omega_{ij} = \exp(\phi_i)\exp(-\phi_j)$ is doing real explanatory work, or whether the hypothesis is unfalsifiable because both transformers and the gauge VFE share this property by construction (any positive empirical correlate cannot distinguish "transformers match language" from "transformers and gauge VFE both happen to have flat transports by parameterization choice").
- Whether sub-claim (b) (functional pressure toward path-independence) is anything more than a teleological just-so story — communicative systems also use heavily context-dependent meaning (deixis, irony, idiom) extensively and functionally.
- Whether the proposed falsifying experiment at `:2232` (training a non-flat gauge architecture and measuring holonomy on pragmatic tasks) is well-posed: the architecture would need a non-cocycle parameterization, but the user's framework derives $\Omega$ from per-token frames, which automatically satisfies the cocycle.
