# Red Opening — section-3-gauge-covariant-vfe

## Steelman (opposing position)

After the editorial corrections from the §4–§5 debate series, §3 of `Attention/GL(K)_attention.tex` (lines 571–992) presents a self-consistent multi-agent gauge-covariant variational free energy in which every load-bearing piece — dual-fiber state space, vertex-frame Ω parameterization, flat-bundle restriction, single-agent FEP recall, mixture-of-sources reduction, log-barrier α-regularizer, belief dynamics, Elitzur-disclaimed symmetry breaking — is either standard-form (and correctly cited) or honestly labeled as a novel construction with the deferral to companion work explicit.

## Position

Sub-claim D is the load-bearing falsifier of the compound claim's "theoretically pure" prong: the manuscript at lines 902–921 introduces the precision regularizer `R(α) = b₀α - c₀ log α` purely phenomenologically (line 914: "A natural choice is a log-barrier form such as"), and does not identify it as the negative log-density of a Gamma hyperprior on α — the conjugate prior for the precision parameter of a Gaussian likelihood — even though that identification is exact, standard, and decisive for grounding the construction in standard Bayesian machinery. The "natural choice" framing is an under-justified hand-wave in a section that is explicitly claimed to be theoretically pure. Under the compound-claim adjudication rule in `00_claim.md` ("If one or two sub-claims fail editorially but the others survive structurally, the judge should issue RED_WINS with the falling sub-claims identified"), this single editorial gap on a load-bearing piece is sufficient to topple the compound claim as written.

Sub-claims A, B, C, E, and F survive primary-source pressure under the available evidence.

## Evidence

- **Manuscript text at `Attention/GL(K)_attention.tex:914`.** Verbatim: "A natural choice is a log-barrier form such as:". The justification provided in lines 916–921 is phenomenological — "penalizes arbitrarily large precision (preventing `α_i → ∞`)" and "log-barrier that enforces `α_i > 0` and penalizes vanishing precision (preventing `α_i → 0`)" — not theoretical. No reference is given for the form; no derivation from a hyperprior is shown; no citation to the conjugate-prior canon appears.

- **Canonical conjugate prior for Gaussian precision [Bishop 2006 §2.3.6 "The Gaussian Distribution"; Murphy 2012 §4.6.1].** The Gamma distribution `Gamma(α; c, b) = b^c/Γ(c) · α^{c-1} exp(-b α)` for `α > 0`, `c, b > 0` is the unique conjugate prior for the precision (inverse variance) parameter of a univariate Gaussian likelihood. Its negative log-density is `-log p(α) = bα - (c - 1) log α + const`. Matching the manuscript's `R(α_i) = b₀α_i - c₀ log α_i` to this form gives `b = b₀`, `c - 1 = c₀`, i.e., `R(α_i) = -log Gamma(α_i; c₀ + 1, b₀) + const`. The identification is exact, not approximate. This is confirmed in the evidence pack at `01_evidence.md:91-93`.

- **The closed-form solution at `Attention/GL(K)_attention.tex:937` is the Gamma-prior MAP estimator.** Setting `∂F/∂α = KL + b₀ - c₀/α = 0` yields `α* = c₀/(b₀ + KL)`. This is precisely the posterior-mode estimate of α under a `Gamma(c₀ + 1, b₀)` prior with linear-in-α "log-likelihood" `-α · KL` (the self-coupling penalty playing the role of a sufficient statistic). The manuscript writes the formula but does not connect it to MAP-under-Gamma — losing both the standard-Bayesian grounding and the hyper-parameter-interpretation handles (`b₀` is the rate, `c₀ + 1` is the shape, the mode under data is the shrinkage-toward-prior estimator).

- **Project communication-style rule [`CLAUDE.md §Communication Style`].** "No bullshit. If a correspondence is interpretive rather than mathematically exact, say so explicitly. ... Never dress up hand-waving as theorem." The "natural choice" framing at line 914 is the canonical hand-wave the rule prohibits — exactly because the Gamma-conjugate-prior correspondence IS mathematically exact and SHOULD be stated, not papered over with "natural."

- **The evidence pack pre-acknowledges this gap.** `01_evidence.md:110`: "Whether the log-barrier regularizer is theoretically motivated or merely 'convenient.' The Gamma-conjugate-prior derivation above suggests it IS theoretically motivated, but the manuscript at line 914 only says 'a natural choice,' not 'the negative log-density of a Gamma hyperprior.' A red-team strike: the manuscript should make this explicit (editorial clarification, not structural correction)." The compound claim's blue defense cannot dispute the gap exists; it can only dispute load-bearingness.

- **Load-bearingness argument.** Sub-claim D is structurally load-bearing for the framework: the state-dependent α-mechanism at §3.7 is what gives the framework its adaptive prior precision, and the entire empirical-Bayes story at line 944 ("Both hyper-parameters may be learned via empirical Bayes") presupposes that `b₀, c₀` have a hyper-parameter interpretation. Without the Gamma identification, `b₀` and `c₀` are unmotivated constants; with the Gamma identification, they are the rate and (shape − 1) of a conjugate hyperprior, and empirical-Bayes is the standard EB-for-Gamma procedure [Murphy 2012 §3.4 "Empirical Bayes"]. The manuscript currently has the formula but not the standard machinery that makes the formula make sense. That gap is purity-violating.

## Falsification conditions

This position is wrong if any of the following:

1. **The manuscript at lines 902–921 already identifies R(α) as the negative log-density of a Gamma hyperprior on α and cites Bishop2006 §2.3.6 or Murphy2012 §4.6.1 (or an equivalent canonical source) for Gamma as the conjugate prior on Gaussian precision.** Verifiable by quoting the relevant manuscript line.

2. **The manuscript elsewhere in the body or supplementary (with explicit forward-reference from §3.7) derives R(α) from a Gamma hyperprior and the §3.7 phrasing is a deliberate cross-reference rather than a stand-alone "natural choice" assertion.** Verifiable by quoting the cross-reference at lines 902–921 and the destination derivation.

3. **The "natural choice" framing is, on the project's own communication-style rules, not a violation — i.e., `CLAUDE.md §Communication Style` permits "natural choice" as adequate justification for a regularizer when the conjugate-prior correspondence is exact.** This would require blue to cite the project rule that permits the under-justification, which would contradict the existing "No bullshit. ... Never dress up hand-waving as theorem" rule.

4. **The empirical-Bayes claim at line 944 is independently grounded without needing the Gamma identification.** This would require blue to exhibit the empirical-Bayes update equations for `b₀, c₀` derived from a different generative story (e.g., a method-of-moments derivation that does not invoke Gamma).

If none of (1)–(4) holds, the editorial gap on Sub-claim D stands, and the compound claim's "theoretically pure" prong fails. By the compound-claim adjudication rule in `00_claim.md` (worst load-bearing sub-claim decides), the verdict is RED_WINS with Sub-claim D as the falling piece and the remedy scoped to: add the Gamma-hyperprior identification at §3.7 with `[Bishop2006 §2.3.6]` or equivalent citation, and re-state the closed-form `α* = c₀/(b₀ + KL)` as the MAP estimator under that hyperprior so the empirical-Bayes claim at line 944 inherits standard machinery.
