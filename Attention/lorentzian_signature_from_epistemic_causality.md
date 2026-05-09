# Lorentzian Signature from Finite-Speed Epistemic Causality

This note gives a step-by-step derivation of Lorentzian signature from finite-speed information propagation in a gauge-theoretic / variational epistemic framework.

The main conclusion is:

$$
\boxed{\text{Real Fisher/KL geometry alone cannot produce Lorentzian signature.}}
$$

However,

$$
\boxed{\text{finite-speed epistemic influence} \Rightarrow \text{causal cones} \Rightarrow \text{Lorentzian conformal metric}.}
$$

The derivation below is written in a form suitable for insertion into a manuscript.

---

## 1. Why the classical Fisher/KL construction cannot by itself give Lorentzian signature

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

---

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

---

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

---

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

---

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

---

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

---

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

---

## 8. Conditional theorem

The derivation can be stated as a theorem.

### Theorem: Lorentzian signature from finite-speed epistemic influence

Let

$$
\mathcal{M}\cong \mathbb{R}_\tau\times\Sigma
$$

be the continuum limit of a locally updated multi-agent epistemic system. Assume:

1. There is a distinguished update parameter $\tau$.
2. The transverse observable sector $\Sigma$ carries a positive-definite information metric $h_{ab}$.
3. Epistemic perturbations propagate with finite maximal speed $c_{\mathcal{I}}$.
4. The boundary of possible influence is smooth and nondegenerate.

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
(-,+,\ldots,+).
$$

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

---

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

---

## 10. How finite speed may arise inside the epistemic dynamics

The conditional theorem requires finite propagation speed. There are two natural ways this can arise.

### 10.1 Fundamental discrete local updating

If the agent dynamics are fundamentally discrete and local,

$$
q_i^{n+1}
=
\Phi_i\left(q_i^n,\{q_j^n:j\in\mathcal{N}(i)\}\right),
$$

then information propagates at finite graph speed. After $m$ update steps, influence can travel at most $m$ graph neighborhoods. This gives a discrete causal cone before any continuum approximation.

This is the cleanest route for an agent-based model.

### 10.2 Hyperbolic epistemic field dynamics

If the continuum dynamics are second order or inertial, one may obtain finite-speed propagation from a hyperbolic equation. For a field $\theta(\tau,x)$, consider

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

Using the plane-wave ansatz

$$
\theta\sim e^{i(k_a x^a-\omega\tau)},
$$

the principal symbol gives

$$
-M\omega^2+D^{ab}k_a k_b=0.
$$

Hence

$$
\omega^2=M^{-1}D^{ab}k_a k_b.
$$

The characteristic cone is then determined by

$$
-Md\tau^2+D^{-1}_{ab}dx^a dx^b=0,
$$

or, equivalently,

$$
-c_{\mathcal{I}}^2d\tau^2+h_{ab}dx^a dx^b=0
$$

for appropriate identifications of $c_{\mathcal{I}}$ and $h_{ab}$.

This route requires genuine hyperbolic or telegraph-type epistemic dynamics. Pure first-order diffusion or gradient flow is generally not enough.

---

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

Therefore a purely diffusive gradient-flow limit does not naturally define Lorentzian causal cones.

To obtain Lorentzian signature from dynamics, the manuscript should assume or derive one of the following:

1. a fundamental discrete local update graph;
2. a hyperbolic second-order epistemic dynamics;
3. a telegraph-type finite-latency dynamics;
4. an explicit finite-speed communication constraint.

Without one of these, Lorentzian signature is not derived.

---

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

---

## 13. Recommended replacement for the manuscript's Lorentzian-signature section

The manuscript should replace the complex-frame derivation with the following claim structure.

### Claim 1: no-go result

Real Fisher/KL geometry gives positive semidefinite pullback metrics. Therefore Lorentzian signature cannot be obtained from real classical statistical geometry alone.

### Claim 2: causal route

If local epistemic dynamics define finite-speed information propagation, then they define a causal cone.

### Claim 3: Lorentzian metric from cone

Given a positive spatial information metric $h_{ab}$ and finite information speed $c_{\mathcal{I}}$, the quadratic form

$$
g=-c_{\mathcal{I}}^2d\tau^2+h_{ab}dx^a dx^b
$$

has Lorentzian signature and has null cone equal to the boundary of epistemic influence.

### Claim 4: remaining open problem

The emergence of exactly three spatial dimensions, and the identification of the resulting Lorentzian metric with physical spacetime rather than observer-relative epistemic geometry, remain open problems.

---

## 14. Manuscript-ready paragraph

A concise manuscript version is:

Real Fisher-Rao geometry cannot by itself yield Lorentzian signature: the pullback of a positive Fisher metric is positive semidefinite, and real Gaussian gauge transport preserves covariance positivity. Thus a Lorentzian metric cannot be derived merely by applying real $\mathrm{GL}(K)$ transport to classical Gaussian beliefs. The appropriate source of Lorentzian structure is instead causal. If the agent dynamics define a distinguished update parameter $\tau$ and local epistemic interactions propagate perturbations at finite maximal speed $c_{\mathcal{I}}$ relative to a positive spatial information metric $h_{ab}$, then possible influence satisfies

$$
h_{ab}dx^a dx^b\leq c_{\mathcal{I}}^2d\tau^2.
$$

The boundary of possible influence is therefore

$$
h_{ab}dx^a dx^b=c_{\mathcal{I}}^2d\tau^2,
$$

which is precisely the null condition of

$$
ds^2=-c_{\mathcal{I}}^2d\tau^2+h_{ab}dx^a dx^b.
$$

Since $h_{ab}$ is positive definite, this quadratic form has signature $(-,+,\ldots,+)$, and for a three-dimensional observable sector it has signature $(-,+,+,+)$. Hence Lorentzian signature is derivable conditionally from finite-speed epistemic causality, not from Fisher geometry alone. The remaining burden is to derive finite-speed propagation from the microscopic update rule, or to impose it as a locality/communication constraint.

---

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
\mathrm{signature}(g)=(-,+,\ldots,+).
}
$$

This replaces the previous complex $\mathrm{GL}(K,\mathbb{C})$ construction with a real causal derivation.
