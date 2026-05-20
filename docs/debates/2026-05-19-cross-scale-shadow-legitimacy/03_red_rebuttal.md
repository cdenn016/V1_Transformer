# Red Rebuttal — cross-scale-shadow-legitimacy

## Concession

Blue's C4 is granted. The manuscript's labeling at line 546 of `Attention/Participatory_it_from_bit.tex`
— "This is a structural commitment of the framework rather than a theorem of standard
hierarchical variational inference… we do not display the reduction (or approximation) of the
standard hierarchical scheme to the present cross-scale shadow construction" — is honest
scholarship. The manuscript names the move rather than concealing it. Red does not accuse the
authors of dressing the substitution up as a derivation; the substitution is acknowledged.

What is contested is whether honest acknowledgment of a substitution converts that substitution
into a *refinement* of the standard hierarchical VI / hierarchical AIF scheme. A label is not a
derivation, and honest labeling is not equivalent to literature continuity.

## Core attack

Blue's load-bearing argument is that the cross-scale shadow sits inside the empirical-Bayes
family (C2) and is structurally analogous to the ladder-VAE deterministic prior (C3). Both
analogies fail on the *type signature* of the object placed in the prior slot.

**EB analogy fails (C2).** Robbins 1956 "An Empirical Bayes Approach to Statistics" and its
modern parametric-EB extensions [Carlin–Louis, *Bayesian Methods for Data Analysis*, 3rd ed.,
Ch. 5 §5.1–5.4; Efron, *Large-Scale Inference*, 2010, §1.1] replace an unknown *hyperparameter*
$\eta$ of a *fixed* generative model $p(y|\theta)p(\theta|\eta)$ with an estimate $\hat\eta(y)$
derived from the *observed data*. The structural slot is occupied by a parameter (or hyperprior
with parameters); the substitution is from a parameter estimator computed against data. The
empirical-Bayes literature, as summarized in the Robbins-tradition review [Ignatiadis,
*Empirical Bayes: From Herbert Robbins to Modern Theory and Applications*, lecture notes
2023, §1–§2], characterizes the move as "postulating that the parameters are independent but
with an unknown prior" and using "a fully data-driven estimator to compete with the Bayesian
oracle that knows the true prior."

The cross-scale shadow does something different in kind. It replaces the conditional density
$p(s_\ell | s_{\ell+1})$ — a *generative-model component* — not with a data-driven estimator of
a hyperparameter, but with $\Omega_{i,I}[q_I^{(s+1)}]$ — *another variational posterior* that
is itself the optimization output of a coupled inference, whose own prior is
$\Omega_{I,J}[q_J^{(s+2)}]$, another variational posterior, ad infinitum until the boundary
$r_i^{(s_{\max})}$ at line 548. The substitution is from a posterior of a *coupled subsystem
with its own free parameters and its own inference dynamics*, not from a function of observed
data. This is not the EB move; it is a different move that re-uses the verb "replace."

**Ladder VAE analogy fails (C3).** Sønderby et al. 2016 §3 specifies the generative prior as a
parameterized Gaussian density: "the stochastic layer below each stochastic layer is specified
as a fully factorized Gaussian distribution, and the functions $\mu(\cdot)$ and $\sigma^2(\cdot)$
in the generative and VAE inference use MLP parameters" [Sønderby, Raiko, Maaløe, Sønderby,
Winther, "Ladder Variational Autoencoders," NeurIPS 2016, §3]. The level-$\ell$ prior
$p_\theta(z_\ell | z_{\ell+1})$ is a *member of the generative model* — a density whose
parameters $\theta$ are learned jointly with the rest of the model. It evaluates to a number
given $(z_\ell, z_{\ell+1})$; it is sampled from to generate $z_\ell$ conditional on
$z_{\ell+1}$.

The cross-scale shadow replaces $p_i^{(s)}$ not with a parameterized density indexed by the
higher-level *sample* $z_{\ell+1}$, but with the variational *posterior distribution*
$q_I^{(s+1)}$ itself, transported by $\Omega_{i,I}$. The mathematical type signature is
different: ladder-VAE's prior is `(z_ell, z_{ell+1}) -> density value`; the shadow's prior is
`(c) -> Omega-transported posterior distribution evaluated at c`. The Sønderby construction
preserves the standard hierarchical generative-model factorization
$p(z_1, \dots, z_L) = \prod_\ell p_\theta(z_\ell | z_{\ell+1}) \cdot p_\theta(z_L)$ — every
factor is a density, none is a posterior. The shadow does not preserve this factorization
because each $p_i^{(s)}$ is bound to the inference output of a separately-optimized variational
problem at level $s+1$. The structural slot is the same (a function from the level above
defines the prior at the level below); the contents differ in kind (parameterized density vs
inference output).

**Message-passing / implicit-joint defense fails (C5).** Blue argues the joint
$p(\{q_i\}, \{s_i\})$ is "implicit, given by the stationary distribution of the message-passing
dynamics" with reference to Wainwright–Jordan 2008 §3.4 and Bishop 2006 §8.4. In standard
variational message passing — Wainwright–Jordan 2008, *Graphical Models, Exponential Families,
and Variational Inference*, Foundations and Trends in Machine Learning §3 — the messages are
updates of variational parameters in an optimization problem whose target is the *marginals*
of a *fixed* underlying joint $p(x_1, \dots, x_N)$ defined by a *fixed* factor graph. The
fixed-point dynamics compute (or approximate) the marginals of that fixed model; they do not
*define* the model. Pearl 1988, *Probabilistic Reasoning in Intelligent Systems*, Ch. 4, and
Bishop 2006 §8.4 follow the same scheme: the factor graph is given, the messages are derived
from it.

The cross-scale shadow construction has no fixed factor graph behind it. The factors at level
$\ell$ are equated to the variational posteriors at level $\ell+1$, which are themselves the
outputs of an optimization whose factors at level $\ell+1$ are equated to the variational
posteriors at level $\ell+2$. There is no underlying joint $p(\{q_i\}, \{s_i\})$ whose
marginals the message passing approximates; there is only a coupled variational fixed-point
system whose stationary point is *defined* by the algorithm. Calling this an "implicit joint"
does not exhibit a joint distribution; it relabels the algorithmic fixed point as a
distribution. Wainwright–Jordan §3.4 does not endorse this move — their implicit-joint
characterization of belief propagation always presupposes a fixed underlying model whose
marginals are the inference target.

**The four-anchor Ouroboros identification (C6) covers the weights, not the substitution.**
The line 2216 identification is rigorous as far as it goes: it shows that the *geometric
weighting* $\lambda_k = \lambda_0 \rho^k$ matches West–Harrison 1997 discount factors, the
*additive-KL-to-product-prior* equivalence matches Genest–Zidek 1986 and Hinton 2002, and the
*tempered-Bayes* form $\lambda_0, \rho > 0$ matches Bissiri–Holmes–Walker 2016. None of these
four citations substitute a level-$(\ell{+}1)$ posterior for a level-$\ell$ prior. West–Harrison
applies discounts to a fixed dynamic linear model; Genest–Zidek aggregates expert distributions
on a fixed sample space; Hinton trains products of experts via contrastive divergence on a
fixed observation; Bissiri–Holmes–Walker construct a generalized posterior against a fixed loss
functional. The four anchors legitimize the *weights and pooling form* of the Ouroboros tower,
not the prior-as-posterior substitution. The shadow construction does not inherit literature
legitimacy from anchors that address a different structural feature.

## Defense

Strengthening the load-bearing assumption from the red opening: the manuscript's hierarchical
generative model is structurally different from the Friston 2017 / Parr–Pezzulo–Friston 2022
hierarchical generative model, and the difference is at the type signature of the prior, not
at the parameterization.

Friston, FitzGerald, Rigoli, Schwartenbeck, Pezzulo, "Active Inference: A Process Theory,"
*Neural Computation* 29:1–49, 2017, §"Hierarchical generative models," and Parr, Pezzulo,
Friston, *Active Inference*, MIT Press 2022, Ch. 8 "Deep generative models," specify the joint
$p(o, s_1, \dots, s_L) = p(o|s_1) \prod_\ell p(s_\ell | s_{\ell+1}) \cdot p(s_L)$ as a product
of *fixed conditional densities*, each with its own parameters. The variational posterior $q$
is then constructed against this fixed joint; the inference machinery (the ELBO bound, the
mean-field factorization, the message passing) is *derived* from the fixed joint.

The cross-scale shadow makes the conditional $p(s_\ell | s_{\ell+1})$ a function of the
variational posterior at level $\ell+1$. This is structurally inconsistent with the Friston
2017 / Parr–Pezzulo–Friston 2022 joint: the priors are no longer fixed densities; they are
functionals of the inference outputs. The ELBO the manuscript later writes at
§sec:variational_free_energy is therefore an ELBO for a *different probabilistic object* than
the Friston 2017 ELBO — not a refinement, but a different functional defined against a
different (implicitly-specified, never exhibited) joint.

Blue's C1 — "the standard scheme is itself a structural commitment, not a theorem" — is
correct but does not legitimize the substitution. Two constructions can both be structural
commitments and still be in different structural-commitment families. Friston 2017 commits to
a fixed-density hierarchical generative model in the standard graphical-models tradition
[Pearl 1988; Bishop 2006 Ch. 8]; the cross-scale shadow commits to a coupled variational
fixed-point system with prior-equals-transported-posterior. Both are commitments; they are
commitments to different objects. The line 546 admission acknowledges this — it locates the
shadow construction *outside* the standard scheme — and the absence of a displayed reduction
chain means the manuscript has not exhibited continuity between the two commitments.

The variational-EM literature reinforces the distinction. In Neal–Hinton 1998 ("A View of the
EM Algorithm that Justifies Incremental, Sparse, and Other Variants," in Jordan, ed., *Learning
in Graphical Models*) and Beal 2003 PhD thesis ("Variational Algorithms for Approximate
Bayesian Inference," Gatsby Unit, §1.3), the M-step updates the *parameters* of a fixed
generative model using the variational posterior from the E-step. The model structure — which
density appears in the prior slot — does not change between iterations. The cross-scale shadow
changes the structural identity of the prior density itself: the prior $p_i^{(s)}$ is not the
parametric density of a fixed model with parameters updated via M-step; it is *the variational
posterior* of a coupled subsystem at scale $s+1$. This is structurally distinct from
variational EM and from every standard hierarchical VI construction.

The load-bearing claim of the red opening therefore stands: the cross-scale shadow is a
separate framework that uses the same vocabulary (prior, posterior, ELBO, hierarchy) as
Friston 2017 / Parr–Pezzulo–Friston 2022. The honest labeling at line 546 is correct
scholarship; what it labels is a substitution at the level of the generative model, not a
refinement within the hierarchical-VI family. The Ouroboros tower's pooling/weighting
identification at line 2216 is rigorous for what it covers — the weights and the pooling form
— but does not extend literature legitimacy to the underlying prior-equals-posterior
substitution.

Sources consulted (primary literature, beyond the manuscript and shared evidence pack):

- Sønderby, Raiko, Maaløe, Sønderby, Winther, "Ladder Variational Autoencoders," NeurIPS 2016,
  §3 [arxiv.org/abs/1602.02282].
- Friston, FitzGerald, Rigoli, Schwartenbeck, Pezzulo, "Active Inference: A Process Theory,"
  *Neural Computation* 29(1):1–49, 2017, §"Hierarchical generative models"
  [openaccess.city.ac.uk/16683].
- Wainwright, Jordan, *Graphical Models, Exponential Families, and Variational Inference*,
  Foundations and Trends in Machine Learning, 2008, §3 (variational characterization of belief
  propagation) and §4 (mean field).
- Robbins, "An Empirical Bayes Approach to Statistics," *Proc. Third Berkeley Symp.* 1956;
  Carlin–Louis, *Bayesian Methods for Data Analysis* 3rd ed., Ch. 5; Efron, *Large-Scale
  Inference*, 2010 §1.1; Ignatiadis, *Empirical Bayes: From Robbins to Modern Theory and
  Applications*, lecture notes 2023 §1–2.
- Neal, Hinton, "A View of the EM Algorithm that Justifies Incremental, Sparse, and Other
  Variants," in Jordan ed., *Learning in Graphical Models*, 1998; Beal, *Variational
  Algorithms for Approximate Bayesian Inference*, PhD thesis, Gatsby Unit, 2003, §1.3.
