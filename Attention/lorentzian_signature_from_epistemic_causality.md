# Lorentzian Signature from Finite-Speed Epistemic Causality

This note gives a step-by-step derivation of Lorentzian signature from finite-speed information propagation in a gauge-theoretic / variational epistemic framework.

The main conclusion is:

$$
\boxed{\text{Direct Fisher pullback of a real Gaussian belief fiber under }\mathrm{GL}(K,\mathbb{R})\text{ transport cannot produce Lorentzian signature.}}
$$

However,

$$
\boxed{\text{finite-speed epistemic influence} \Rightarrow \text{causal cones} \Rightarrow \text{a Lorentzian conformal class }[g].}
$$

The note is intended as a complementary route to the manuscript's existing $\mathrm{GL}(K,\mathbb{C})$ frame-twist construction in `sec:signature_resolution`, not a replacement. The two routes use disjoint postulate sets and answer different questions: the $\mathrm{GL}(K,\mathbb{C})$ route exhibits structural compatibility of an indefinite quadratic form on the connection sector with Lorentzian signature; the causal-cone route below derives a Lorentzian conformal class from finite-speed dynamics. Neither produces a unique metric without further postulates, and neither resolves the open problem of dynamical selection.

## 1. Why the direct Fisher pullback cannot by itself give Lorentzian signature

The argument below establishes only the narrow no-go statement that the *direct fiber-wise* pullback of a positive Fisher-Rao metric, transported by real $\mathrm{GL}(K,\mathbb{R})$ frames acting on a real Gaussian belief fiber, is positive semidefinite. It does not rule out indirect real-frame routes such as Killing-form bilinear forms on non-compact Lie algebras, indefinite Bregman divergences arising from generalised exponential families, or hyperbolic-type information geometries built from non-Gaussian fibers. The manuscript's existing $\mathrm{GL}(K,\mathbb{C})$ construction in `sec:signature_resolution` is one such indirect route: it leaves the fiber statistics positive, complexifies only the connection sector, and locates the indefinite signature in the Lie-algebra Killing form $\mathrm{tr}(A_\mu A_\nu)$ rather than in a Fisher pullback. The narrow no-go below is therefore consistent with that route, and is not a refutation of it.

Let $\mathcal{B}$ be a statistical manifold of belief states, and let

$$
g^{F}_{\mathcal{B}}
$$

be its Fisher-Rao metric. For a smooth belief section

$$
q : \mathcal{C} \to \mathcal{B},
$$

the pullback information metric on the base space $\mathcal{C}$ is

$$
G_{\mu\nu}
=
q^* g^{F}_{\mathcal{B}}
=
g^{F}_{\mathcal{B}}
\left(\partial_\mu q,\partial_\nu q\right).
$$

For any tangent vector

$$
v=v^\mu \partial_\mu \in T_c\mathcal{C},
$$

we have

$$
G_{\mu\nu}v^\mu v^\nu
=
g^{F}_{\mathcal{B}}
\left(v^\mu\partial_\mu q,v^\nu\partial_\nu q\right)
\geq 0.
$$

Therefore

$$
\boxed{G \text{ is positive semidefinite.}}
$$

It may be degenerate if the map $q$ fails to be an immersion, but it cannot have one negative eigenvalue. Hence a Fisher pullback metric cannot directly produce signature

$$
(-,+,+,+).
$$

The same obstruction holds for real Gaussian KL geometry. If

$$
\Sigma \succ 0
$$

and

$$
\Omega \in \mathrm{GL}(K,\mathbb{R}),
$$

then the pushed-forward covariance is

$$
\Sigma \mapsto \Omega\Sigma\Omega^\top.
$$

For every nonzero vector $x$,

$$
x^\top \Omega\Sigma\Omega^\top x
=
(\Omega^\top x)^\top \Sigma (\Omega^\top x)
>
0,
$$

because $\Omega^\top x\neq 0$ when $\Omega$ is invertible. Thus

$$
\Omega\Sigma\Omega^\top \succ 0.
$$

So real gauge transport preserves positive definiteness. Consequently,

$$
\boxed{
\text{real Gaussian KL}
+
\text{real Fisher geometry}
+
\mathrm{GL}(K,\mathbb{R})\text{ transport}
\not\Rightarrow
\text{Lorentzian signature}.
}
$$

This is the basic no-go result.

## 2. Why the complex-frame construction is only a toy model

A common toy construction introduces an imaginary temporal gauge component. For example, suppose a connection-like field has components

$$
A_\tau=i(\partial_\tau \psi_\tau)T,
$$

and

$$
A_x=(\partial_x \psi_x)T,
$$

where $T$ is a generator with

$$
\mathrm{tr}(T^2)>0.
$$

If one defines

$$
G_{\mu\nu}^{\mathrm{tw}}
=
\mathrm{tr}(A_\mu A_\nu),
$$

then

$$
G_{\tau\tau}^{\mathrm{tw}}
=
\mathrm{tr}(A_\tau A_\tau)
=
i^2(\partial_\tau \psi_\tau)^2\mathrm{tr}(T^2)
=
-(\partial_\tau \psi_\tau)^2\mathrm{tr}(T^2),
$$

whereas

$$
G_{xx}^{\mathrm{tw}}
=
\mathrm{tr}(A_x A_x)
=
(\partial_x \psi_x)^2\mathrm{tr}(T^2).
$$

Thus one obtains a line element of the form

$$
ds^2
=
-a(\tau,x)d\tau^2+b(\tau,x)dx^2,
$$

with

$$
a(\tau,x)>0,
\qquad
b(\tau,x)>0.
$$

This produces a Lorentzian sign pattern, but only because the temporal component was manually multiplied by $i$. Therefore this construction shows that a Lorentzian signature is available after complexification, but it does not explain why the temporal direction should be imaginary or why exactly one direction should carry the opposite sign.

Thus the complex construction should be interpreted as a Wick-rotation toy model, not as a derivation.

## 3. The correct route: derive Lorentzian structure from causal cones

The better route is operational and real:

$$
\boxed{
\text{local epistemic updating}
\Rightarrow
\text{finite information speed}
\Rightarrow
\text{causal cone}
\Rightarrow
\text{Lorentzian metric}.
}
$$

The key idea is that Lorentzian signature is the quadratic encoding of a causal distinction:

- timelike: influence possible;
- null: influence possible at maximal speed;
- spacelike: influence impossible.

The negative sign in the metric is not inserted by hand. It is forced by the algebraic representation of an influence cone.

## 4. Discrete epistemic dynamics and finite graph speed

Let agents be indexed by vertices $i$ of a graph $\Gamma$. Let $n\in\mathbb{Z}$ denote discrete update time. Agent $i$ at update step $n$ has belief state

$$
q_i^n.
$$

Suppose the update rule is local:

$$
q_i^{n+1}
=
\Phi_i\left(q_i^n,\{q_j^n:j\in \mathcal{N}(i)\}\right),
$$

where $\mathcal{N}(i)$ is the neighbor set of $i$.

If a perturbation is introduced at vertex $i$ at time $n$, then after one update it can affect only vertices in $\mathcal{N}(i)$. After two updates, it can affect only vertices within graph distance two, and so on.

Let

$$
d_\Gamma(i,j)
$$

be the graph distance between $i$ and $j$. Then a perturbation at $(i,n)$ can influence $(j,n+m)$ only if

$$
d_\Gamma(i,j)\leq m.
$$

If each graph edge corresponds to a spatial information length $\Delta x$ and each update corresponds to a time increment $\Delta \tau$, then the maximal information speed is

$$
c_{\mathcal{I}}
=
\frac{\Delta x}{\Delta \tau}.
$$

More generally, if an update can propagate across at most $r$ graph edges per step, then

$$
c_{\mathcal{I}}
=
\frac{r\Delta x}{\Delta \tau}.
$$

Thus locality of the update rule gives a finite causal propagation speed.

## 5. Continuum limit and spatial information metric

Assume the large-scale limit of the agent graph gives a smooth spatial manifold

$$
\Sigma.
$$

Let

$$
h_{ab}(x)
$$

be a positive-definite spatial information metric on $\Sigma$. This metric may come from a Fisher pullback restricted to the observable spatial sector:

$$
h_{ab}
=
\sum_i w_i(x)\,g^F_{\mathcal{B}}
\left(\nabla_a q_i,\nabla_b q_i\right),
$$

or from another positive information-geometric construction. The essential assumption is

$$
h_{ab}v^a v^b>0
$$

for all nonzero spatial tangent vectors $v\in T_x\Sigma$.

Let $\tau$ be the epistemic update parameter. Then the effective continuum base has local product form

$$
\mathcal{M}
\cong
\mathbb{R}_\tau\times \Sigma.
$$

A displacement in $\mathcal{M}$ has the form

$$
dc^\mu=(d\tau,dx^a).
$$

The spatial information distance squared is

$$
ds_{\Sigma}^2
=
h_{ab}dx^a dx^b.
$$

Finite-speed epistemic propagation means that influence over update interval $d\tau$ is possible only when

$$
h_{ab}dx^a dx^b
\leq
c_{\mathcal{I}}^2 d\tau^2.
$$

The boundary of maximal influence is therefore

$$
h_{ab}dx^a dx^b
=
 c_{\mathcal{I}}^2 d\tau^2.
$$

This is the null-cone condition.

## 6. Constructing the Lorentzian metric from the cone

Move all terms in the null-cone condition to one side:

$$
-c_{\mathcal{I}}^2d\tau^2+h_{ab}dx^a dx^b=0.
$$

This is the null condition of the quadratic form

$$
g
=
-c_{\mathcal{I}}^2\,d\tau\otimes d\tau
+
h_{ab}\,dx^a\otimes dx^b.
$$

Equivalently, the line element is

$$
\boxed{
ds^2
=
g_{\mu\nu}dc^\mu dc^\nu
=
-c_{\mathcal{I}}^2d\tau^2+h_{ab}dx^a dx^b.
}
$$

Since $h_{ab}$ is positive definite, the eigenvalues of $g$ consist of one negative eigenvalue and $\dim\Sigma$ positive eigenvalues. Therefore

$$
\boxed{
\mathrm{signature}(g)=(-,+,\ldots,+).
}
$$

If

$$
\dim\Sigma=3,
$$

then

$$
\boxed{
\mathrm{signature}(g)=(-,+,+,+).
}
$$

Thus Lorentzian signature is obtained from the finite-speed causal structure of epistemic influence.

## 7. Timelike, null, and spacelike epistemic separation

Given

$$
ds^2=-c_{\mathcal{I}}^2d\tau^2+h_{ab}dx^a dx^b,
$$

there are three cases.

### Timelike epistemic separation

Influence is possible with submaximal speed when

$$
h_{ab}dx^a dx^b
<
c_{\mathcal{I}}^2d\tau^2.
$$

Equivalently,

$$
ds^2<0.
$$

### Null epistemic separation

Influence propagates at the maximal information speed when

$$
h_{ab}dx^a dx^b
=
c_{\mathcal{I}}^2d\tau^2.
$$

Equivalently,

$$
ds^2=0.
$$

### Spacelike epistemic separation

Influence is impossible within the available update interval when

$$
h_{ab}dx^a dx^b
>
c_{\mathcal{I}}^2d\tau^2.
$$

Equivalently,

$$
ds^2>0.
$$

So the Lorentzian sign convention encodes the operational distinction between possible and impossible epistemic influence.

## 8. Conditional theorem

The derivation can be stated as a theorem.

### Theorem: Lorentzian conformal class from finite-speed epistemic influence

Let

$$
\mathcal{M}\cong \mathbb{R}_\tau\times\Sigma
$$

be the continuum limit of a locally updated multi-agent epistemic system. Assume:

1. There is a globally distinguished update parameter $\tau$, defining a foliation $\mathcal{M} \cong \mathbb{R}_\tau \times \Sigma$. This is a substantive postulate; elsewhere in the manuscript Fisher arc length is treated as per-agent and need not coincide across agents, so this assumption is not automatically satisfied and must be interpreted as a coarse-grained shared update parameter on which the causal-cone construction conditions.
2. The transverse observable sector $\Sigma$ carries a positive-definite information metric $h_{ab}$.
3. Epistemic perturbations propagate with finite maximal speed $c_{\mathcal{I}}$.
4. The boundary of possible influence is the level set of a quadratic form in $(d\tau, dx^a)$. This is a Riemannian, not Finslerian, assumption: it requires that the influence norm be of inner-product type. A generic finite-speed bound $|dx|_\star \le c_\mathcal{I}\,d\tau$ for an arbitrary norm $|\cdot|_\star$ would yield a Finsler cone whose null set is not the level set of a quadratic form; the present theorem applies only when $|\cdot|_\star$ comes from a positive-definite inner product, which is supplied by $h_{ab}$.

Then the boundary of possible influence is the null cone of the quadratic form

$$
g
=
-c_{\mathcal{I}}^2\,d\tau\otimes d\tau
+
h_{ab}\,dx^a\otimes dx^b.
$$

Moreover, $g$ has Lorentzian signature

$$
(-,+,\ldots,+),
$$

and the cone determines $g$ only up to a positive conformal factor: $g$ and $\lambda g$ for any smooth positive function $\lambda$ have identical null sets. The theorem therefore establishes a Lorentzian *conformal class* $[g]$, not a unique metric. Selecting a representative requires fixing the overall scale, equivalent to choosing a unit for $c_\mathcal{I}$.

If $\dim\Sigma=3$, then

$$
\mathrm{signature}(g)=(-,+,+,+).
$$

### Proof

Finite-speed propagation says that influence is possible only inside the cone

$$
h_{ab}dx^a dx^b
\leq
c_{\mathcal{I}}^2d\tau^2.
$$

The boundary of this cone is

$$
h_{ab}dx^a dx^b
=
c_{\mathcal{I}}^2d\tau^2.
$$

Rearranging gives

$$
-c_{\mathcal{I}}^2d\tau^2+h_{ab}dx^a dx^b=0.
$$

Define

$$
g_{\mu\nu}dc^\mu dc^\nu
=
-c_{\mathcal{I}}^2d\tau^2+h_{ab}dx^a dx^b.
$$

Then the influence boundary is exactly the null set of $g$. Since $c_{\mathcal{I}}^2>0$ and $h_{ab}$ is positive definite, $g$ has one negative eigenvalue and $\dim\Sigma$ positive eigenvalues. Therefore its signature is

$$
(-,+,\ldots,+).
$$

This proves the claim.

## 9. Relation to gauge and Fisher structures

The positive spatial metric $h_{ab}$ may be built from gauge-covariant Fisher pullbacks. For example, with gauge-covariant derivative

$$
\nabla_a^{(i)}q_i
$$

one can define

$$
h_{ab}
=
\sum_i w_i\,
 g^F_{q_i}\left(\nabla_a^{(i)}q_i,\nabla_b^{(i)}q_i\right),
$$

with weights

$$
w_i\geq 0,
\qquad
\sum_i w_i=1.
$$

Because Fisher geometry is positive semidefinite, this gives

$$
h_{ab}v^a v^b\geq 0.
$$

If the observable spatial sector is nondegenerate, then $h$ is positive definite.

The Lorentzian sign does not come from the Fisher metric itself. It comes from adjoining the update direction $\tau$ and encoding finite-speed influence as a causal cone.

Thus the decomposition is:

$$
\boxed{
\text{Fisher/KL geometry supplies the positive spatial information metric }h.
}
$$

$$
\boxed{
\text{Finite-speed epistemic dynamics supplies the causal cone.}
}
$$

$$
\boxed{
\text{The causal cone supplies the Lorentzian sign structure.}
}
$$

## 10. How finite speed may arise inside the epistemic dynamics

The conditional theorem requires finite propagation speed. There are two natural ways this can arise.

### 10.1 Discrete local updating

If the agent dynamics are intrinsically discrete and local,

$$
q_i^{n+1}
=
\Phi_i\left(q_i^n,\{q_j^n:j\in\mathcal{N}(i)\}\right),
$$

then information propagates at finite graph speed. After $m$ update steps, influence can travel at most $m$ graph neighborhoods. This gives a discrete causal cone before any continuum approximation.

The discrete cone has only graph-distance content; whether a *continuum* cone with finite information speed $c_\mathcal{I}$ survives the scaling limit is a non-trivial separate problem. Naive scaling limits of bounded-degree local rules with diffusive timescales typically yield parabolic continuum equations whose Green's function has support everywhere, i.e., infinite signal speed in the continuum. Recovering a finite continuum cone requires either a continuous-time random-walk construction with bounded jump size and exponentially distributed waiting times (yielding a telegraph-equation continuum limit) or another mechanism that preserves the discrete-locality bound under rescaling. The discrete cone is therefore necessary but not sufficient for a continuum Lorentzian conformal class.

### 10.2 Hyperbolic epistemic field dynamics (consistency check, not derivation)

If the continuum dynamics are *postulated* to be second order or hyperbolic, then the principal symbol consistently reproduces the cone of section 6. This subsection is a consistency check rather than a derivation of finite speed: the hyperbolic principal symbol that delivers a finite cone is exactly the structure presupposed by writing a wave equation. A derivation of finite speed would have to start from microscopic premises that do not already encode Lorentzian structure.

For a field $\theta(\tau,x)$, suppose

$$
M\partial_\tau^2\theta
+
\Gamma\partial_\tau\theta
=
\nabla_a\left(D^{ab}\nabla_b\theta\right)-\nabla V(\theta).
$$

The principal part is

$$
M\partial_\tau^2\theta-D^{ab}\nabla_a\nabla_b\theta.
$$

Using the plane-wave ansatz $\theta\sim e^{i(k_a x^a-\omega\tau)}$, the principal symbol gives

$$
-M\omega^2+D^{ab}k_a k_b=0,
$$

so $\omega^2=M^{-1}D^{ab}k_a k_b$, and the characteristic cone is

$$
-Md\tau^2+D^{-1}_{ab}dx^a dx^b=0,
$$

or equivalently $-c_{\mathcal{I}}^2d\tau^2+h_{ab}dx^a dx^b=0$ for appropriate identifications of $c_{\mathcal{I}}$ and $h_{ab}$. The mixed sign in this expression is fixed by the $-\partial_\tau^2$ versus $+\nabla_a\nabla_b$ relative sign in the assumed equation; deriving the relative sign from first principles would require an independent argument that the underlying epistemic dynamics admit a Hamiltonian or symplectic structure rather than a gradient-flow structure.

The route therefore presupposes hyperbolic or telegraph-type epistemic dynamics. Pure first-order diffusion or gradient flow is parabolic in continuum limits and yields infinite signal speed in the heat-kernel sense; the framework's own E-step natural-gradient flow $\mu_i \leftarrow \mu_i - \eta\,\nabla_{\mu}\mathcal{F}$ in `transformer/vfe/e_step.py` falls under this parabolic class on its face. Either the discretisation must be shown to preserve a finite cone in some scaling limit, or the framework dynamics must be replaced by a hyperbolic counterpart in any regime where the causal-cone route is to apply.

## 11. Why first-order gradient flow is insufficient

A standard first-order gradient flow has the schematic form

$$
\partial_\tau q
=
-\nabla \mathcal{F}(q).
$$

In spatial continuum limits, such equations often become parabolic or diffusive. A simple diffusion equation is

$$
\partial_\tau q
=
D\Delta q.
$$

The heat kernel for this equation has support everywhere for every $\tau>0$. Thus perturbations propagate with infinite speed in the continuum model.

Therefore a purely diffusive gradient-flow limit does not naturally define Lorentzian causal cones. This is not a hypothetical concern for the present framework. The E-step natural-gradient update $\mu_i \leftarrow \mu_i - \eta\,\Sigma_i\,\nabla_\mu\mathcal{F}$ implemented in `transformer/vfe/e_step.py` is the discrete-time Euler scheme for a first-order natural-gradient flow on the belief manifold. Its naive continuum limit at fixed step ratio is parabolic, and standard scaling-limit arguments do not preserve a finite causal speed. The causal-cone route therefore does not apply directly to the framework's E-step dynamics as currently implemented.

To obtain a Lorentzian causal structure from epistemic dynamics, the manuscript would need one of the following routes, none of which is currently realised:

1. a non-trivial discrete-to-continuum scaling limit (for example a continuous-time random walk with bounded jump size and exponentially distributed waiting times) of the discrete update graph that preserves a finite information speed;
2. a replacement of first-order natural-gradient flow with a second-order hyperbolic epistemic dynamics;
3. a telegraph-type finite-latency relaxation between the natural-gradient drift and an inertial response;
4. an explicit finite-speed communication constraint imposed at the architectural level.

The conditional implication "finite-speed influence implies a Lorentzian conformal class" is independent of which route is taken; what is open is whether any of these routes is consistent with the framework's existing E-step semantics.

## 12. What this derivation establishes and what it does not establish

This derivation establishes the conditional implication

$$
\boxed{
\text{finite-speed epistemic influence}
\Rightarrow
\text{Lorentzian signature}.
}
$$

It does not, by itself, establish:

$$
\boxed{\dim\Sigma=3.}
$$

If $\dim\Sigma=d$, then the emergent signature is

$$
(-,+,\ldots,+)
$$

with $d$ positive directions.

To obtain physical $1+3$ spacetime, the framework still needs an additional argument selecting

$$
d=3.
$$

Possible routes include:

1. a rank-three dominant observable sector of the Fisher pullback metric;
2. an $\mathrm{SO}(3)$ representation-theoretic selection mechanism;
3. stability or universality of three large information-geometric dimensions;
4. empirical identification of three macroscopic spatial directions;
5. a separate topological or dynamical dimension-selection theorem.

The Lorentzian sign pattern is derivable from causal structure. The number of spatial dimensions is a separate problem.

## 13. Complementary route to the manuscript's Lorentzian-signature section

The manuscript already contains a $\mathrm{GL}(K,\mathbb{C})$ frame-twist construction in `sec:signature_resolution` that exhibits structural compatibility between an indefinite Lie-algebra Killing form and Lorentzian signature, with the imaginary temporal generator and the real-part projection both flagged as postulates. The route below is complementary to that construction: it uses a disjoint postulate set (real frames throughout, plus a finite information speed) and produces a Lorentzian *conformal class* on a base product manifold rather than a unique metric. Neither route is intended as a derivation of physical spacetime; both are existence statements about how Lorentzian signature can be made structurally compatible with the framework, and the open problem of dynamical selection survives in both routes.

### Claim 1: narrow no-go for direct Fisher pullback

The direct fiber-wise pullback of a positive Fisher metric, transported by real $\mathrm{GL}(K,\mathbb{R})$ frames acting on a real Gaussian belief fiber, is positive semidefinite. Lorentzian signature is therefore not obtainable along this specific route. Indirect real-frame routes (for example Killing-form bilinears on non-compact Lie algebras) are not ruled out by this statement.

### Claim 2: causal-cone construction

If local epistemic dynamics define a finite maximal speed of information propagation $c_\mathcal{I}$ relative to a positive spatial information metric $h_{ab}$, then the boundary of possible epistemic influence is the null set of the quadratic form

$$
g = -c_{\mathcal{I}}^2\,d\tau\otimes d\tau + h_{ab}\,dx^a\otimes dx^b.
$$

### Claim 3: Lorentzian conformal class from the cone

The form $g$ has Lorentzian signature $(-,+,\ldots,+)$ by Sylvester's law. Because $g$ and $\lambda g$ for any positive $\lambda$ share the same null cone, the cone fixes $g$ only up to a positive conformal factor. Producing a unique Lorentzian metric requires an additional postulate fixing the overall scale, equivalent to a choice of unit for the information speed $c_\mathcal{I}$.

### Claim 4: remaining open problems

The dimension count is open: if $\dim\Sigma = d$, the construction yields signature $(-,+,\ldots,+)$ with $d$ positive directions, but it does not select $d=3$. The dynamical assumption of finite information speed is also open: the framework's existing first-order natural-gradient E-step has parabolic continuum limits and does not by itself produce a finite cone. Reconciliation requires either a non-trivial scaling limit of the discrete update graph or a replacement of the dynamics by a hyperbolic counterpart. The relation of the resulting epistemic-spacetime geometry to physical spacetime, including the per-agent plural-time structure of the rest of the manuscript, is also open: section 8 below assumes a globally distinguished update parameter $\tau$, while elsewhere the manuscript treats Fisher arc length as per-agent and explicitly refrains from identifying it with relativistic proper time.

## 14. Manuscript-ready paragraph

A concise manuscript version, complementary to the manuscript's existing $\mathrm{GL}(K,\mathbb{C})$ frame-twist construction, is:

The direct pullback of a positive Fisher-Rao metric to a base manifold by real $\mathrm{GL}(K,\mathbb{R})$-transported Gaussian beliefs is positive semidefinite, and a Lorentzian metric cannot be obtained along this specific route. Indirect real-frame routes are not excluded: the manuscript's $\mathrm{GL}(K,\mathbb{C})$ construction in `sec:signature_resolution` locates indefinite signature in the Lie-algebra Killing form $\mathrm{tr}(A_\mu A_\nu)$ rather than in a Fisher pullback. A second indirect route, complementary to the $\mathrm{GL}(K,\mathbb{C})$ construction, comes from finite-speed causality. If the agent dynamics define a coarse-grained shared update parameter $\tau$ and local epistemic interactions propagate perturbations at finite maximal speed $c_{\mathcal{I}}$ relative to a positive spatial information metric $h_{ab}$, with influence-boundary of inner-product type, then possible influence satisfies $h_{ab}dx^a dx^b \le c_{\mathcal{I}}^2 d\tau^2$, and the boundary is the null set of

$$
ds^2 = -c_{\mathcal{I}}^2 d\tau^2 + h_{ab}dx^a dx^b.
$$

This quadratic form has Lorentzian signature $(-,+,\ldots,+)$, with $(-,+,+,+)$ in the three-dimensional observable case. The cone determines this metric only up to a positive conformal factor, so the construction establishes a Lorentzian *conformal class* rather than a unique metric. The dynamical premise is non-trivial for the present framework: the E-step natural-gradient flow implemented in `transformer/vfe/e_step.py` is first-order and has parabolic continuum limits with infinite signal speed, so the causal-cone route does not apply directly to the current dynamics. Either a non-trivial scaling limit must preserve a finite cone, or a hyperbolic counterpart of the dynamics must be introduced. The route is therefore an existence statement complementary to the $\mathrm{GL}(K,\mathbb{C})$ frame-twist statement, not a derivation of physical spacetime, and the dimension count and the dynamical mechanism remain open.

## 15. Summary

The corrected derivation is:

$$
\boxed{
\text{Fisher/KL geometry} \Rightarrow \text{positive spatial information metric }h.
}
$$

$$
\boxed{
\text{local finite-speed epistemic update} \Rightarrow \text{causal influence cone}.
}
$$

$$
\boxed{
\text{causal cone boundary}:
 h_{ab}dx^a dx^b=c_{\mathcal{I}}^2d\tau^2.
}
$$

$$
\boxed{
\text{Lorentzian line element}:
 ds^2=-c_{\mathcal{I}}^2d\tau^2+h_{ab}dx^a dx^b.
}
$$

$$
\boxed{
\mathrm{signature}(g)=(-,+,\ldots,+),\qquad g\text{ determined by the cone only up to conformal factor.}
}
$$

This is a complementary route to the manuscript's existing $\mathrm{GL}(K,\mathbb{C})$ frame-twist construction in `sec:signature_resolution`. The two routes use disjoint postulate sets: the $\mathrm{GL}(K,\mathbb{C})$ route postulates an imaginary temporal gauge generator and a real-part projection; the causal-cone route postulates a globally distinguished update parameter, a positive spatial information metric, finite information speed, and an inner-product-type influence norm. Neither produces a unique metric without further postulates, neither selects $\dim\Sigma = 3$, and the causal-cone route is in tension with the framework's first-order natural-gradient E-step dynamics, which has parabolic continuum limits.
