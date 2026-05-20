# Action — ouroboros-ancestral-weights-bayesian-id

**From verdict:** BLUE_WINS (top-level, disjunctive claim)

Per-identification: strong proposition BLUE; Identification I (deep GP) RED; Identification II (AR(1)) RED on manuscript-as-written, recoverable under latent-vs-marginal reading; Identification III (broadened to discount-factor DLM `[WestHarrison1997 Ch. 10]`) BLUE.

## Recommended action

The verdict licenses the Ouroboros tower as a standard hierarchical-Bayesian construction (Gaussian log-linear pool / product-of-experts with geometric forgetting, equivalently the discount-factor dynamic-linear-model family) — but requires four concrete manuscript revisions to convert the licensed identification into a published, defensible claim. All four target `Attention/Participatory_it_from_bit.tex` in the Ouroboros tower extension section.

1. **Write the explicit free-energy functional at line 2208.** Replace the prose sentence about "exponentially decaying weights $\lambda_k = \lambda_0 \cdot \rho^k$" with the explicit fragment $\sum_k \lambda_0 \rho^k \, \mathrm{KL}(q_i \,\|\, \Omega_{i,I_k}[q_{I_k}^{(s+k)}])$ inside the $F$ functional, with a parallel fragment for model-fiber shadows using $(\tilde\Omega, s)$ in place of $(\Omega, q)$. Without the explicit functional, the role of $\rho^k$ remains underspecified.

2. **Add the textbook citation at the same location.** Identify the construction as a Gaussian log-linear pool / product-of-experts with geometric forgetting `[GenestZidek1986, HintonEtAl2002, BleiKuckelbirgJordan2017]` or as a discount-factor dynamic linear model `[WestHarrison1997 Ch. 10]`. The two are related but distinct; commit to one as primary and cite the other as adjacent.

3. **Commit on normalization at the same location.** State explicitly whether $\sum_{k \geq 0} \lambda_k = \lambda_0/(1-\rho) = 1$ is imposed. If imposed, the construction is a strictly-normalized variational free energy under a standard log-linear-pool prior. If not imposed, the construction is a tempered / generalized-Bayesian free energy in the sense of `[BissiriHolmesWalker2016, GrunwaldVanOmmen2017]`, and the citation should say so. The current ambiguity is the basis of red's tempering objection.

4. **Disambiguate "non-Markovian" at line 2196.** Specify whether the qualifier refers to the generative scale-process (forcing a long-memory / fractional model per `[Beran2010 §2]`) or to the agent-level marginal prior under a Markovian-in-scale AR(1) latent generator (per `[CappeMoulinesRyden2005 §1.3, Hamilton1994 §3.2]`). If the revision commits to the agent-level-marginal reading under an AR(1) latent process, Identification II is also recovered; if it commits to the genuinely-multi-source-prior reading, only Identification III broadened carries the disjunction.

## Follow-up debates

None required. The verdict resolves the question.

The earlier debate's blue concession at `docs/debates/2026-05-19-agent-meta-agent-hierarchy-theory/03_blue_rebuttal.md` (Ouroboros as "free-form posit ... out-of-scope for theory mode") is superseded by the present verdict: the construction is no longer free-form once the four revisions above are applied, and the textbook identification is the log-linear pool with geometric forgetting / discount-factor DLM.

## Open issue surfaced (no follow-up debate, but worth a note)

Identification III in the strict `[IbrahimChen2000]` power-prior reading remains dead — power priors discount likelihoods, not KL terms between beliefs. The successful identification is the broader discount-factor-DLM family `[WestHarrison1997 Ch. 10]` named on the same line of the claim's reference list. If a future paper wants to cite the Ouroboros as "a power prior," that citation would be technically inaccurate; the accurate name is "discount-factor dynamic linear model" or "log-linear pool with geometric forgetting."
