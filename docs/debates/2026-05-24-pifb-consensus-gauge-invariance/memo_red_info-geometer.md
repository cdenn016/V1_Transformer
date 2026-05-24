# Memo — Red / info-geometer / opening

## Steelman
The manuscript correctly notes (:2992) that "the GL(K,ℝ) gauge invariance of KL divergence on
the real Gaussian belief fiber" is a property of the construction, and it does not claim to
derive that invariance from anything — it takes it as given and asks what follows.

## Falsify
That is exactly the problem for the consensus-gauge thesis built on top of it. The invariance
of KL on the belief fiber is not an incidental feature; it is forced by the choice of objective.
The Fisher information metric is, up to scale, the **unique** Riemannian metric on a statistical
manifold invariant under sufficient statistics (Cencov / Chentsov uniqueness theorem
[Cencov1972; AmariNagaoka2000 Ch. 2]). KL is the second-order expansion of that metric and
inherits the invariance: for q = N(μ_q, Σ_q), p = N(μ_p, Σ_p), a frame change U ↦ Ug acts by
the (0,2)/(2,0) sandwich on Σ and a linear action on μ, and the closed-form Gaussian KL is
unchanged because the trace and quadratic terms are conjugation-invariant
[AmariNagaoka2000 Ch. 2; KingmaWelling2014 App. B]. The framework selected a divergence whose
defining virtue is frame-invariance.

Now read the consensus-gauge thesis (:2990): "For agents with different internal reference
frames to agree on shared geometric structure, that structure must be gauge-invariant."
On the belief fiber, the structure agents compare *is already* gauge-invariant, because the
comparison functional (KL) is. The "consensus requirement" therefore imposes no new constraint
on the belief fiber — it is satisfied identically. A requirement that is auto-satisfied cannot
*select* gauge invariance; the gauge invariance was an input. This is the information-geometric
content of the circularity the philosophy-of-science memo names.

The manuscript's hedge ("may not be falsifiable") does not reach this. A reader could grant the
hedge and still walk away believing the framework *explains why* shared structure is
gauge-invariant. It does not. It assumes a Cencov-invariant objective and then observes that
the objective is invariant.

## External primary-source citation (not the manuscript)
- Cencov, N. N. (1972), *Statistical Decision Rules and Optimal Inference* — Fisher metric is
  the unique (up to scale) metric invariant under sufficient statistics. Restated and extended
  by Ay–Jost–Lê–Schwachhöfer, "Information geometry and sufficient statistics," *Probab.
  Theory Relat. Fields* (2015), arXiv:1207.6736; and Dowty (2018) for exponential families.
  https://arxiv.org/pdf/1207.6736

## Falsification condition
Wrong if (i) the Gaussian-KL closed form is NOT invariant under the frame action U ↦ Ug —
i.e., a sympy/finite-difference check shows tr(Σ_p⁻¹Σ_q) + Mahalanobis + log-det ratio changes
under conjugation by an admissible g — or (ii) the framework's belief-fiber objective is
something other than a Cencov-invariant divergence (then the consensus constraint would be
non-trivial). Neither holds: the invariance is the manuscript's own stated premise (:2992).

## Newly-discovered canon
- Chentsov's theorem (overview): Fisher metric unique up to scale, invariant under sufficient
  statistics. https://en.wikipedia.org/wiki/Chentsov's_theorem
- Ay, Jost, Lê, Schwachhöfer (2015/2017), "Information geometry and sufficient statistics" —
  full generalization of Chentsov to infinite sample sizes; Fisher metric + Amari–Chentsov
  tensor uniquely characterized by sufficient-statistic invariance. arXiv:1207.6736.
