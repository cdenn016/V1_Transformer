# Memo — debate-expert-variational (red, sur-rebuttal)

## Position

Blue's rebuttal §Defense brands the manuscript's threshold detector (line 2174) as the canonical principled-object-with-tractable-surrogate pattern of variational inference, citing Wainwright-Jordan 2008 *Foundations and Trends in Machine Learning* 1(1-2) §3, Blei-Kucukelbir-Jordan 2017 *JASA* 112 §2, and the Jordan-Ghahramani-Jaakkola-Saul 1999 *Machine Learning* 37(2) §3 founding-move framing. This defense conflates two distinct VI patterns: (a) the *bounded-surrogate* pattern, where the surrogate provably bounds the principled object (mean-field VI gives an ELBO that lower-bounds the log-evidence, by Jensen's inequality); and (b) the *unbounded-heuristic* pattern, where the surrogate is computationally cheap but its relation to the principled object is unestablished. Canonical VI is (a), not (b).

## Why this matters

Wainwright-Jordan 2008 §3 develops the mean-field surrogate with explicit error analysis: the optimization of the variational free energy provides a lower bound on the marginal log-likelihood, and the bound is tight at the true posterior. Blei-Kucukelbir-Jordan 2017 §2.2 (equations 1–4) writes this out: $\log p(x) \geq \mathbb{E}_q[\log p(x, z)] - \mathbb{E}_q[\log q(z)] = \mathrm{ELBO}(q)$, derived by Jensen. The error analysis is built in. Jordan-Ghahramani-Jaakkola-Saul 1999 §3 makes the same move: the variational free energy provides a bound; bounds can be improved by enriching the family.

Now PIFB line 2174 in full: "we do not establish that the detector's product form exactly tracks the variational-criterion savings even in the high-coherence limit. Whether a continuous-time evaluation of [the FE-improvement criterion at line 2123] reproduces the same hierarchical organization that the threshold-based detector produces is open." This is not a Jensen-style bound. It is not even a one-sided bound. It is an open question whether the surrogate tracks the principled object *at all*, even in the high-coherence limit. There is no error analysis; there is no monotonicity argument; there is no certificate that the threshold detector's $C_b \cdot C_m \cdot P$ form (or the simulator's two-factor $C_b \cdot C_m$ form) lies on the same side of the FE-improvement criterion as the principled object.

This is the unbounded-heuristic pattern, not the canonical VI pattern. Calling it "the founding move of mean-field VI performed correctly" (blue's rebuttal §Defense final sentence on Vector 3) misreads the founding move.

## On the Amari forward-KL barycenter

Blue is correct that PIFB Eq. 2142 is in the canonical Amari forward-KL barycenter family [Amari, Nagaoka 2000 §3.4; Amari 2007 *Neural Computation* 19(10)]. Red does not contest this. The contest is at line 2179 (dispersion-term drop) and line 2187 (saddle-point weight). The dispersion drop is, as Bishop 2006 §10.7 makes explicit, a deviation from the moment-matched barycenter; the manuscript labels it $\mathcal{O}(\varepsilon)$ in the high-coherence regime, and that label is honest *for the high-coherence regime*. Whether the simulator actually operates in the high-coherence regime at runtime is not verified in §Implementation. The label is a regime-conditional pass, not an absolute pass.

## On the Bissiri-Holmes-Walker citation

The variational memo from Phase 2 (round 2 opening) flagged the BHW citation at line 2275 as a tempered-Bayes / learning-rate construction, whereas the Ouroboros fragment at line 2270 uses $\rho^k$ as a geometric discount across hierarchical generations. Blue's rebuttal does not address this; the manuscript's own line 2275 hedge ("the role of historical time is played here by the hierarchical scale-distance $k$") is an *analogy*, not a derivation. West-Harrison 1997 §6.3 is the dynamic-linear-model discount; PIFB Eq. 2270 is closer to West-Harrison 1997 §10.7 discounted-likelihoods or Smith 1979 *JRSSB* 41. Citation precision wound on sub-claim 2 (canonical fidelity), not load-bearing on its own but contributing to the cumulative count.

## Newly-discovered canon

None — Wainwright-Jordan 2008, Blei-Kucukelbir-Jordan 2017, Jordan-Ghahramani-Jaakkola-Saul 1999, Amari-Nagaoka 2000, Bishop 2006, BHW 2016, West-Harrison 1997, Smith 1979 already on the record.
