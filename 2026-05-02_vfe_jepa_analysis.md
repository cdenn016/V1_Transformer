# 2026-05-02 Analysis — Relationship of the Gauge-Theoretic VFE Transformer to JEPA

## Scope

This document traces the formal relationship between the gauge-theoretic
variational free energy framework implemented in this repository and Yann
LeCun's Joint Embedding Predictive Architecture (JEPA), including its
hierarchical and action-conditioned extensions (H-JEPA, I-JEPA, V-JEPA,
V-JEPA 2, V-JEPA 2-AC). It also positions the active-inference extension in
`transformer/vfe/efe.py` and `transformer/core/expected_free_energy.py`
against current variational JEPA and EFE-planning literature. The
conclusion is that the overlap is structural rather than analogical, but
the variational JEPA framing was independently published in January 2026
(Huang, arXiv:2601.14354), so the novelty of the present framework lies
elsewhere — in gauge-theoretic structure, hierarchical generative model,
KL-as-distance with analytic covariance transport, and a complete Friston
EFE policy rather than risk-only goal-conditioned planning.

## The formal mapping

LeCun's JEPA energy (position paper §4, OpenReview 2022; equivalently the
I-JEPA loss with L1 substituted for L2) is

$$
E_\text{JEPA}(x, y, z) = D\bigl(s_y, \text{Pred}(s_x, z)\bigr),
\qquad s_x = \text{Enc}_x(x), \quad s_y = \text{Enc}_y(y).
$$

The belief-coupling term in this framework, written in `vfe/block.py` and
`vfe/stack.py` (and discussed in `transformer/VFE_Transformer_Idea.md` §3),
is

$$
F_\text{couple} = \beta_{ij} \cdot \mathrm{KL}\left(q_i \Vert \Omega_{ij} \cdot q_j\right),
$$

with $\Omega_{ij} = \exp(\phi_i)\exp(-\phi_j) \in GL^+(K)$ and the
covariance transported by the sandwich product $\Omega \Sigma \Omega^\top$
(`math_utils/transport.py`). Identifying $s_x \leftrightarrow q_j$,
$s_y \leftrightarrow q_i$, $\text{Pred}(\cdot, z) \leftrightarrow
\Omega_{ij}(\cdot)$, and $D \leftrightarrow \mathrm{KL}$, the two scalar
functionals occupy the same place in the architecture and play the same
role: a representation-space compatibility score between a source and a
target, with an auxiliary uncertainty channel. The substantive design
choices that distinguish them are the predictor class, the distance, the
encoding of uncertainty, the encoder-tying mechanism, and the
collapse-prevention strategy.

The predictor class differs categorically. JEPA's $\text{Pred}$ is an
unconstrained ViT or MLP — a universal function approximator. The transport
$\Omega_{ij}$ here is an element of $GL^+(K)$ parameterized by the gauge
fields $\phi_i, \phi_j$, with at most $K^2$ real degrees of freedom per
position pair before any sharing. JEPA's predictor expresses arbitrary
nonlinear maps; the gauge transport expresses only positively-oriented
invertible linear maps, plus whatever nonlinearity is encoded in the
position-dependent gauge fields. This is a deliberate inductive bias — the
"no neural networks" thesis is the bet that the equivariance constraint and
the structural collapse-avoidance it brings with it outweigh the lost
expressivity. JEPA papers make the opposite bet.

The distance differs. JEPA's $D$ is L1 (I-JEPA), L2 (LeCun position paper),
or in some variants a contrastive InfoNCE-style score (C-JEPA, NeurIPS
2024). This framework uses KL between Gaussians, which is the proper
Bregman divergence on the natural-parameter space of the exponential family
and gives Fisher-Rao geometry on the manifold of beliefs. KL recovers a
quadratic distance only in the small-perturbation limit; in general it is
asymmetric and curvature-aware. For Gaussian beliefs this is almost free
computationally and gives a principled information-geometric descent
through the Fisher projection step in `vfe/e_step.py`.

The encoding of uncertainty differs. JEPA's latent $z$ is an auxiliary
variable that can either be sampled (VAE-style) or inferred (energy-based
inference); it absorbs information about $y$ that is not predictable from
$x$. This framework carries uncertainty as the covariance $\Sigma$ of the
belief itself and propagates it analytically via $\Omega \Sigma
\Omega^\top$. The information content is similar — both encode "what cannot
be predicted from the source" — but the calculus is different. JEPA's
predictive moment is sample-based; here it is moment-matched. The closest
JEPA-tradition analog is an explicit Gaussian assumption on $z$ with
analytic propagation, which would essentially recover the variational JEPA
of Huang 2026 (arXiv:2601.14354).

Encoder tying differs in form but plays the same role. I-JEPA uses an EMA
target encoder that receives no gradient updates, preventing the trivial
collapse where $\text{Enc}_x = \text{Enc}_y$ both produce constants. This
framework uses a single PriorBank for both ends of the coupling. The
analogous stop-gradient in this codebase is on the cross-layer cascade,
where the previous layer's posterior $\mu_q$ becomes the next layer's prior
$\mu_p$ with the gradient detached in several `em_mode` settings (see the
table in `CLAUDE.md` §"EM modes" and the mode definitions in
`transformer/vfe/config.py`). Both mechanisms prevent the encoder from
collapsing onto its own target; the JEPA solution stops gradient on the
target side, the VFE solution (in cascaded layers) stops gradient on the
prior side. Whether these are formally equivalent under any choice of
generative model is an open question worth a careful derivation.

The collapse-prevention story is the most consequential difference. I-JEPA
has been argued in the literature to be insufficiently protected against
collapse by EMA alone, which led to C-JEPA (Chen et al., NeurIPS 2024,
arXiv:2410.19560) bolting VICReg's variance and covariance regularizers
onto the I-JEPA objective. Even with that addition, the mechanism is
extrinsic — added to the loss because the architecture does not provide
collapse-avoidance on its own. The KL term $\alpha \cdot \mathrm{KL}(q_i \Vert p_i)$
in this framework's free energy penalizes collapse directly: a belief
that contracts to a constant pays unbounded information cost to a
non-degenerate prior. Huang's VJEPA paper makes essentially this argument
formally and provides "formal guarantees for collapse avoidance" as one of
its main theoretical contributions. Variational JEPA gets collapse
prevention from the variational objective itself, while deterministic JEPA
must add it back through architectural tricks or auxiliary losses.

## Active inference and planning: VFE EFE versus V-JEPA 2-AC

V-JEPA 2-AC (Bardes et al., arXiv:2506.09985) plans by minimizing a
goal-conditioned energy

$$
E(\hat{a}_{1:T}; z_k, s_k, z_g) = \bigl\| P_\phi(\hat{a}_{1:T}; s_k, z_k) - z_g \bigr\|_1,
$$

over an action sequence $\hat{a}_{1:T}$ via the cross-entropy method. This
is a risk-only objective in Friston's decomposition: the L1 distance from
the rolled-out latent to the goal latent is a divergence between the
expected outcome and the preferred outcome, with no ambiguity term and no
epistemic-value term. The empirical claim that the energy landscape is
"smooth and locally convex" is what lets CEM converge in $O(\text{16
seconds per action})$ rather than minutes.

The active-inference extension here, in `VFEExpectedFreeEnergy`
(`transformer/vfe/efe.py`) and `compute_efe`
(`transformer/core/expected_free_energy.py`), implements the canonical
Parr–Pezzulo–Friston (2022) decomposition

$$
G_t(a) = \underbrace{\mathbb{E}_{q(o|a)}[-\log p^{\star}(o)]}_{\text{risk}}
     + \underbrace{\mathbb{E}_{q(z|a)}[H[p(o|z)]]}_{\text{ambiguity}}
     - \underbrace{I_q(z; o \mid a)}_{\text{epistemic value}},
$$

with the policy posterior $q_t(a) \propto \exp(-\gamma G_t(a))$. The risk
term takes three modes (`'current_belief'`, `'target'`, `'uniform'`) with
the divergent-log issue handled correctly in the target mode by switching
from cross-entropy to NLL. The epistemic term is a BALD mutual information
estimate computed by Monte Carlo sampling from the belief and reading out
predictive entropies through the PriorBank decode.

V-JEPA 2-AC's CEM-on-energy is a strict special case of this scheme: take
$p^\star(o) = \delta(o = o_g)$ (a delta preference at the goal), drop the
ambiguity and epistemic terms, and replace the Gibbs sample-from-policy
with an argmin via CEM. The user's framework therefore subsumes V-JEPA
2-AC's planner as the risk-only, deterministic-readout, delta-preference
limit. Operationally this means the framework has explicit machinery for
exploration (epistemic value drives the agent toward latents whose
predictive distribution is most affected by uncertainty) and for
ambiguity-aversion (avoid actions whose outcome distribution is high
entropy under the current belief), neither of which V-JEPA 2-AC currently
implements.

The de Vries and Nuijten 2025 paper (arXiv:2504.14898) derives precisely
the inverse of this relationship: starting from a variational free energy
on a generative model with preference and epistemic priors, EFE-based
planning emerges as variational inference on that model. This validates
the move from variational free energy (this framework's $F$) to expected
free energy (this framework's $G$) as a derivation rather than a
postulate.

## Three meanings of "energy"

The word "free energy" gets used across this literature in three distinct
senses, and conflating them invites confusion.

The first is JEPA's $E(x, y, z)$, an EBM-style scalar that is an
unnormalized negative log-density with no entropy term. This is "energy"
in the Hopfield/Boltzmann sense, recoverable from the LeCun 2006 EBM
tutorial. Training minimizes this energy on positive pairs while pushing
it up on negative pairs (or relying on architectural anti-collapse for
non-contrastive variants).

The second is Helmholtz/variational free energy $F = \mathbb{E}_q[-\log
p(o,x)] - H[q]$, equivalently $\mathrm{KL}(q \Vert p) - \log p(o)$. This is the
$F$ minimized by this framework's E-step. It contains both an energy term
(the negative log joint) and an entropy term (the differential entropy of
$q$), which is what makes it "free." Minimizing $F$ over $q$ tightens the
ELBO; minimizing it over the prior parameters performs maximum-likelihood
learning under the variational approximation.

The third is expected free energy $G(a) = \text{Risk} + \text{Ambiguity} -
\text{Epistemic}$, a forward-looking objective averaged over predicted
future states under each candidate policy. This is the active-inference
quantity formalized by Parr, Pezzulo and Friston (2022) and derived from
$F$ in de Vries and Nuijten (2025).

The chain is JEPA energy (unnormalized negative log-density) ⊂ variational
free energy (adds entropy and prior KL) ⊂ expected free energy (extends to
actions and policies). LeCun's published JEPA primarily lives at level
one, with risk-only level-three planning attached at inference time in
V-JEPA 2-AC. Huang's VJEPA lives at level two for the world model, with
no active-inference extension as of the v1 manuscript. This framework
spans levels two and three with a complete EFE decomposition.

## Position relative to current literature

Three recent papers establish what is and is not novel in the present
framework's positioning, and being honest about this matters for any
downstream writeup.

Huang's "VJEPA: Variational Joint Embedding Predictive Architectures as
Probabilistic World Models" (arXiv:2601.14354, January 2026) introduces
the variational JEPA framing, learns a predictive distribution over future
latents via a variational objective, gives "formal guarantees for collapse
avoidance," and establishes that "sequential modeling does not require
autoregressive observation likelihoods." The Bayesian JEPA extension
(BJEPA) factorizes the predictive belief into a "learned dynamics expert
and a modular prior expert" with a Product of Experts intersection. The
parallels to this codebase's $q$ (belief) and $p$ (prior) factorization
are real and the variational-objective framing is shared. The present
framework was not the first to make this move and the writeup should not
claim that it was.

Chen et al.'s C-JEPA (NeurIPS 2024, arXiv:2410.19560) demonstrates
empirically that I-JEPA's EMA mechanism is inadequate against collapse and
that bolting VICReg onto the JEPA objective fixes it. This supports the
framing that JEPA literature is actively wrestling with collapse and that
a variational objective with a $\mathrm{KL}(q \Vert p)$ term to a non-degenerate
prior gives this for free. Cite this when arguing that the variational
formulation is not just a theoretical preference but a fix for a
documented failure mode.

De Vries and Nuijten's "Expected Free Energy-based Planning as Variational
Inference" (arXiv:2504.14898, April 2025) derives EFE from variational
free energy with preference and epistemic priors. The active-inference
extension in this framework instantiates that derivation in a JEPA-style
world model with a complete EFE decomposition (risk + ambiguity -
epistemic), which is a step beyond the risk-only V-JEPA 2-AC planner. The
derivation is theirs; the instantiation in a gauge-theoretic JEPA is this
framework's.

What this framework retains as distinguishing contributions, after
accounting for the above, is the following set of design choices, all of
which are absent from VJEPA, BJEPA, and the V-JEPA family. The transport
operator is a Lie group element rather than a free predictor, which gives
strict gauge equivariance under SE(K) and GL+(K) at the level of the
sandwich-transported covariance. The generative model is hierarchical with
explicit levels $h \to s \to p \to q$ and metacognition couplings
$\gamma_{ij} \mathrm{KL}(s_i \Vert \Omega_{ij} s_j)$ between models, which is
genuinely beyond H-JEPA's stack-of-encoders construction. Inference is
iterative E-step optimization over beliefs at every forward pass rather
than amortized through an encoder, which trades compute for adaptivity
and provides a different gradient profile (six distinct EM modes
documented in `CLAUDE.md`). The application is autoregressive language
modeling rather than vision, and the action space in EFE is the
next-token vocabulary rather than continuous robot actions.

## Honest assessment of the novelty story

The accurate one-line positioning is that this framework instantiates the
variational JEPA paradigm with gauge-theoretic structure (Lie group
transport, sandwich covariance) and a hierarchical generative model
($h \to s \to p \to q$), and extends it at inference time to the full
Friston expected free energy policy that V-JEPA 2-AC's CEM-on-energy is a
risk-only special case of. The variational JEPA framing itself, including
collapse-avoidance guarantees and the Bayesian-filter interpretation, was
established by Huang in January 2026; the EFE-as-variational-inference
framing was established by de Vries and Nuijten in April 2025. The novel
contributions are the gauge-theoretic and hierarchical structure on the
generative model side, the iterative E-step inference scheme, and the
language-modeling application of the active-inference extension.

The three honest caveats. The gauge transport's expressive limit is
$GL^+(K)$ acting linearly on belief means; whether this restriction wins
empirically against a free predictor at scale is unresolved, and the
"no-NN" thesis is the bet rather than the finding. The encoder-tying
analogy between I-JEPA's EMA stop-gradient and this framework's detached
prior in cascaded layers is suggestive but not derived; a careful
comparison of the two gradient flows is owed. And the variational JEPA
literature is moving fast — between Huang (Jan 2026) and BJEPA, the
"factorize belief into dynamics expert and prior expert" idea is now in
the air, and any writeup positioning the present framework against current
literature should track new arXiv submissions monthly.

## Files referenced

- `transformer/vfe/efe.py` — `VFEExpectedFreeEnergy` policy (78 lines)
- `transformer/vfe/active_inference.py` — E-step pragmatic+epistemic shaping (78 lines)
- `transformer/core/expected_free_energy.py` — risk/ambiguity/epistemic primitives, full EFE compute, EFE generation loop (516 lines)
- `transformer/core/active_inference.py` — `_compute_active_inference_gradient` for the E-step callback (522 lines)
- `transformer/vfe/block.py` and `vfe/stack.py` — belief-coupling KL term and cross-layer cascade
- `math_utils/transport.py` — sandwich-product covariance transport
- `transformer/VFE_Transformer_Idea.md` §3, §11 — VFE coupling derivation, active-inference framing

## Sources

- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence v0.9.2*. OpenReview, BZ5a1r-kVsf.
- Dawid, A. & LeCun, Y. (2023). *Introduction to Latent Variable Energy-Based Models: A Path Towards Autonomous Machine Intelligence*. arXiv:2306.02572.
- Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., Ballas, N. (2023). *Self-Supervised Learning From Images With a Joint-Embedding Predictive Architecture*. CVPR 2023, arXiv:2301.08243.
- Chen, S. et al. (2024). *Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning* (C-JEPA). NeurIPS 2024, arXiv:2410.19560.
- Bardes, A. et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. arXiv:2506.09985. Includes V-JEPA 2-AC action-conditioned planner.
- Huang, Y. (2026). *VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models*. arXiv:2601.14354. Includes BJEPA extension.
- de Vries, B. & Nuijten, M. (2025). *Expected Free Energy-based Planning as Variational Inference*. arXiv:2504.14898.
- Parr, T., Pezzulo, G. & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.
- Friston, K. (2010). *The free-energy principle: a unified brain theory?*. Nature Reviews Neuroscience 11, 127–138.
- Da Costa, L. et al. (2020). *Active inference on discrete state-spaces: A synthesis*. Journal of Mathematical Psychology 99, 102447.
