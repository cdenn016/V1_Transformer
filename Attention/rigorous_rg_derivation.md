<!-- GitHub-compatible math version. -->

# Rigorous Renormalization-Group Construction for Gauge-Covariant Meta-Agent Emergence

This note rewrites the renormalization section as a step-by-step mathematical construction. The purpose is to separate three logically distinct claims:

1. **Exact RG:** any measurable gauge-covariant coarse-graining map induces an exact renormalized free energy by pushforward of the variational Gibbs measure.
2. **Closure:** the exact renormalized free energy remains in the original multi-agent Gaussian KL functional class only under additional hypotheses. Closure is exact in a linear-Gaussian, compact-gauge, flat-transport regime and approximate otherwise.
3. **Emergence/retention:** the threshold detector used in simulations should not be identified with the RG itself. It can instead be justified as a sufficient-condition surrogate for positive variational retention gain in a high-coherence regime.

The resulting claim is stronger and more defensible than saying the simulation is merely “RG-like,” but it avoids overstating what is actually proved.

---

## 0. Executive statement suitable for the manuscript

A precise version of the renormalization claim is:

Let $X_s$ denote the finite-dimensional state of all scale-$s$ agents on a finite grid approximation of the base $\mathcal C$, and let $\mathcal F_s(X_s)$ be the scale-$s$ variational free energy. Given a measurable gauge-covariant coarse-graining map $R_s:X_s\mapsto X_{s+1}$, the pushforward

$$
\mathbb P_{s+1}=(R_s)_*\mathbb P_s, \quad \mathbb P_s=Z_s^{-1}e^{-\mathcal F_s/\tau}\,d\nu_s,
$$

defines an exact renormalized free energy $\mathcal F_{s+1}^{\mathrm{exact}}$. This exact RG step preserves the partition function and all retained observables, and it satisfies the usual composition law under successive coarse-grainings. The nontrivial question is whether $\mathcal F_{s+1}^{\mathrm{exact}}$ lies in, or near, the original multi-agent Gaussian KL functional class. In the compact-gauge, linear-Gaussian, flat-transport, spectrally gapped high-coherence regime, closure holds up to additive constants and Schur-complement renormalization of the quadratic form. Away from this regime, the closure residual is controlled by barycentric dispersion, transport/holonomy variation, edge-marginal incompatibility, anharmonic Laplace corrections, and timescale-separation error. The simulation threshold detector is therefore best interpreted as a candidate-selection surrogate for the conditions under which a parent effective theory has positive variational retention gain.

This statement gives a rigorous RG backbone while correctly classifying the threshold detector as an implementation surrogate rather than the fundamental mathematical operation.

---

## 1. Finite-dimensional scale state

Work first on a finite grid approximation of the base manifold $\mathcal C$. Let $\Lambda_s\subset \mathcal C$ denote the finite grid at scale $s$. The continuum limit can be discussed later by mesh refinement under regularity assumptions.

At scale $s$, let the index set of agents be $\mathcal I_s$. A scale-$s$ state is

$$
X_s=\{x_i^{(s)}\}_{i\in\mathcal I_s},\quad x_i^{(s)}=(q_i,p_i,U_i,\chi_i),
$$

where, at each grid point $c\in\Lambda_s$,

$$
q_i(c)=\mathcal N(\mu_i(c),\Sigma_i(c)),\quad p_i(c)=\mathcal N(m_i(c),\Pi_i(c)),
$$

$$
U_i(c)\in G,\qquad\chi_i(c)\in\{0,1\}.
$$

Here:

- $q_i$ is the belief field.
- $p_i$ is the model/prior field or the relevant model-sector Gaussian.
- $U_i$ is the local gauge frame.
- $\chi_i$ is the support mask.
- $G$ is the gauge group.

For the rigorous theorem-level construction, take

$$
G=\mathrm{SO}(K)
$$

or any compact matrix Lie group equipped with a bi-invariant Riemannian metric. The noncompact $\mathrm{GL}(K)$ case can be treated later with gauge fixing, a regulator, or restriction to a compact subgroup/polar factor. Compactness matters because it gives a well-behaved Haar measure and local uniqueness of Karcher means.

The scale-$s$ gauge transport from $j$ to $i$ is

$$
\Omega_{ij}=U_iU_j^{-1}.
$$

For $G=\mathrm{SO}(K)$, the action on Gaussian moments is

$$
\Omega_{ij}q_j=\mathcal N(\Omega_{ij}\mu_j,\Omega_{ij}\Sigma_j\Omega_{ij}^\top).
$$

The same notation applies to the model sector $p_j$, possibly with a distinct representation of $G$ on the model fiber.

---

## 2. Microscopic free-energy functional

The exact RG construction does not require a specific free-energy functional. It only requires a measurable function $\mathcal F_s:X_s\to\mathbb R\cup\{+\infty\}$. For closure, however, we need a functional class.

A representative scale-$s$ multi-agent Gaussian KL functional is

$$
\mathcal F_s(X_s)=\sum_{i\in\mathcal I_s}\mathcal F_i(q_i,p_i)+\sum_{i,j\in\mathcal I_s}\beta_{ij}^{(q,s)}D_{ij}^{(q)}+\sum_{i,j\in\mathcal I_s}\beta_{ij}^{(p,s)}D_{ij}^{(p)}+\mathcal{F_s^{\mathrm{higher}}},
$$

where

$$
D_{ij}^{(q)}=\int_{\Lambda_s}\chi_i(c)\chi_j(c)\mathrm{KL}\!\left(q_i(c)\middle\|\Omega_{ij}(c)q_j(c)\right)\,dc,
$$

$$
D_{ij}^{(p)}=\int_{\Lambda_s}\chi_i(c)\chi_j(c)\mathrm{KL}\!\left(p_i(c)\middle\|\Omega_{ij}^{(p)}(c)p_j(c)\right)\,dc.
$$

The term $\mathcal F_s^{\mathrm{higher}}$ may include observation terms, self-terms, cross-scale priors, curvature penalties, entropy penalties, or implementation-specific regularizers.

For the strongest theorem below, specialize further to a local quadratic expansion of $\mathcal F_s$ near coherent clusters.

---

## 3. Step 1: define admissible clusters

A coarse-graining step requires a partition or cover of microscopic agents by candidate clusters.

Define the overlap graph $\mathcal G_s=(\mathcal I_s,E_s)$ by

$$
(i,j)\in E_s
\quad\Longleftrightarrow\quad
\sum_{c\in\Lambda_s}\chi_i(c)\chi_j(c)>0.
$$

A candidate parent cluster is a finite set $I\subset\mathcal I_s$ satisfying:

1. **Multi-child condition:** $|I|\ge 2$.
2. **Connected overlap:** the induced subgraph $\mathcal G_s[I]$ is connected.
3. **Nonempty multi-child support:**
   $$
\chi_I(c):=\mathbf 1\!\left\{\sum_{i\in I}\chi_i(c)\ge 2\right\}
$$
   is nonzero somewhere.
4. **Local frame coherence:** the frames $\{U_i(c):i\in I\}$ lie in a convex normal ball of $G$ for each $c$ in the parent support.
5. **Belief/model coherence:** the post-transport barycentric dispersions defined below are finite and sufficiently small.

The parent support is not allowed to grow into regions covered by only one child. Thus

$$
\chi_I(c)=\mathbf 1\!\{\sum_{i\in I}\chi_i(c)\ge 2\}.
$$

This encodes the rule that a parent lives strictly over genuine overlap regions.

Let $\mathcal P_s=\{I_1,\ldots,I_{N_{s+1}}\}$ be the selected collection of clusters. For theorem-level simplicity, take $\mathcal P_s$ to be a partition of $\mathcal I_s$. For the implementation, overlapping parents can be allowed; then $R_s$ maps to a larger parent state space and the pushforward construction still works if the selection rule is measurable.

---

## 4. Step 2: define the parent frame by a Karcher mean

For each admissible cluster $I$, define the parent frame as the Riemannian center of mass

$$
\boxed{U_I(c)=\arg\min_{U\in G}\sum_{i\in I}w_i^I(c)\,d_G(U,U_i(c))^2.}
$$

Here $w_i^I(c)\ge 0$, $\sum_{i\in I}w_i^I(c)=1$, and $d_G$ is the geodesic distance induced by a bi-invariant metric on the compact group $G$.

For $G=\mathrm{SO}(K)$, a standard choice is

$$
d_G(U,V)^2=\|\log(U^{-1}V)\|_F^2.
$$

When all $U_i(c)$ lie in a geodesic ball of radius less than the injectivity radius, the minimizer is locally unique.

Define child-to-parent transport by

$$
\boxed{\Omega_{Ii}(c)=U_I(c)U_i(c)^{-1}.}
$$

The transported child belief is

$$
\widetilde q_i^I(c)=\Omega_{Ii}(c)q_i(c),
$$

with moments

$$
\widetilde\mu_i^I(c)=\Omega_{Ii}(c)\mu_i(c),
$$

$$
\widetilde\Sigma_i^I(c)=\Omega_{Ii}(c)\Sigma_i(c)\Omega_{Ii}(c)^\top.
$$

---

## 5. Step 3: define the parent belief by a forward-KL barycenter

The parent belief is the minimizer of the transported child-parent alignment energy:

$$
\boxed{q_I(c)=\arg\min_{q\in\mathcal G_K}\sum_{i\in I}w_i^I(c)\mathrm{KL}\!\left(\widetilde q_i^I(c)\middle\|q\right),}
$$

where $\mathcal G_K$ is the family of $K$-dimensional Gaussian distributions.

Because the parent appears in the second argument of the forward KL, the solution is moment matching. If

$$
\widetilde q_i^I(c)=\mathcal N(\widetilde\mu_i^I(c),\widetilde\Sigma_i^I(c)),
$$

then

$$
\boxed{
\mu_I(c)=\sum_{i\in I}w_i^I(c)\widetilde\mu_i^I(c),
}
$$

and

$$
\boxed{
\Sigma_I(c)=\sum_{i\in I}w_i^I(c)
\left[
\widetilde\Sigma_i^I(c)+(\widetilde\mu_i^I(c)-\mu_I(c))(\widetilde\mu_i^I(c)-\mu_I(c))^\top
\right].
}
$$

The second term in $\Sigma_I$ is the between-child dispersion. It must be retained in the exact Gaussian barycenter. Dropping it is a high-coherence approximation, not an identity.

The same construction defines the parent model/prior state $p_I$:

$$
p_I(c)=\arg\min_{p\in\mathcal G_K}\sum_{i\in I}w_i^I(c)\mathrm{KL}\!\left(\widetilde p_i^I(c)\middle\|p\right).
$$

---

## 6. Step 4: define the coarse-graining map $R_s$

For each cluster $I\in\mathcal P_s$, define

$$
Y_I=(q_I,p_I,U_I,\chi_I).
$$

The global parent state is

$$
X_{s+1}=Y_s=\{Y_I:I\in\mathcal P_s\}.
$$

The scale-$s$ coarse-graining map is

$$
\boxed{
R_s:X_s\mapsto X_{s+1}=Y_s.
}
$$

Explicitly,

$$
R_s(X_s)=\{
(q_I,p_I,U_I,\chi_I):I\in\mathcal P_s
\},
$$

with $q_I,p_I,U_I,\chi_I$ defined by the formulas above.

This map is the mathematical object that was missing from the informal RG analogy.

---

## 7. Step 5: exact RG by pushforward

Define the scale-$s$ variational Gibbs measure

$$
\boxed{
 d\mathbb P_s(X_s) = Z_s^{-1}e^{-\mathcal F_s(X_s)/\tau}\,d\nu_s(X_s),
}
$$

where $\tau>0$ is the variational temperature and $d\nu_s$ is a reference measure on scale-$s$ states. For compact $G$, the frame component of $\nu_s$ can use Haar measure. The Gaussian parameter part can use Lebesgue measure on means and a chosen measure on positive-definite covariances.

The exact renormalized measure is the pushforward

$$
\boxed{
\mathbb P_{s+1}=(R_s)_*\mathbb P_s.
}
$$

That is,

$$
\mathbb P_{s+1}(A)=\mathbb P_s(R_s^{-1}(A))
$$

for measurable sets $A$ in the parent state space.

If $\mathbb P_{s+1}$ is absolutely continuous with respect to a parent reference measure $\nu_{s+1}$, define the exact renormalized free energy by

$$
\boxed{
 d\mathbb P_{s+1}(Y) = Z_{s+1}^{-1}e^{-\mathcal F_{s+1}^{\mathrm{exact}}(Y)/\tau}\,d\nu_{s+1}(Y).
}
$$

Equivalently,

$$
\boxed{
 e^{-\mathcal F_{s+1}^{\mathrm{exact}}(Y)/\tau} = \int_{R_s(X)=Y} e^{-\mathcal F_s(X)/\tau}\,d\nu_s(X\mid Y),
}
$$

up to an additive constant in $\mathcal F_{s+1}^{\mathrm{exact}}$.

In delta-function notation,

$$
\boxed{
 e^{-\mathcal F_{s+1}^{\mathrm{exact}}(Y)/\tau} = \int \delta(Y-R_s(X)) e^{-\mathcal F_s(X)/\tau} d\nu_s(X).
}
$$

This is the exact Wilsonian step.

---

## 8. Theorem 1: partition-function and observable preservation

### Theorem

Let $R_s:X_s\to X_{s+1}$ be measurable, and define $\mathbb P_{s+1}=(R_s)_*\mathbb P_s$. Then:

1. The partition function is preserved up to the chosen normalization convention:
   $$
Z_{s+1}=Z_s.
$$
2. For every bounded measurable parent observable $A:X_{s+1}\to\mathbb R$,
   $$
\boxed{\mathbb E_{s+1}[A(Y)]   =   \mathbb E_s[A(R_s(X))].   }
$$

### Proof

By definition of pushforward,

$$
\int_{X_{s+1}} A(Y)\,d\mathbb P_{s+1}(Y)=\int_{X_s} A(R_s(X))\,d\mathbb P_s(X).
$$

Taking $A=1$ gives normalization preservation. If the unnormalized measures are used, the same calculation gives $Z_{s+1}=Z_s$ after integrating over all parent states.

This proves the claim. $\square$

---

## 9. Theorem 2: discrete semigroup/composition law

### Theorem

Let

$$
X_s\xrightarrow{R_s}X_{s+1}\xrightarrow{R_{s+1}}X_{s+2}.
$$

Then the exact pushforward RG satisfies

$$
\boxed{
(R_{s+1})_*(R_s)_*\mathbb P_s=(R_{s+1}\circ R_s)_*\mathbb P_s.
}
$$

Thus the RG transformations compose as a discrete semigroup.

### Proof

For any measurable set $A\subset X_{s+2}$,

$$
(R_{s+1})_*(R_s)_*\mathbb P_s(A)=(R_s)_*\mathbb P_s(R_{s+1}^{-1}(A))
$$

$$
=
\mathbb P_s(R_s^{-1}(R_{s+1}^{-1}(A)))=\mathbb P_s((R_{s+1}\circ R_s)^{-1}(A)).
$$

This is exactly $(R_{s+1}\circ R_s)_*\mathbb P_s(A)$. $\square$

If the blocking scale is homogeneous and logarithmic scale is denoted by $\ell$, this can be written as

$$
\boxed{
\mathcal R_{\ell_2}\mathcal R_{\ell_1}=\mathcal R_{\ell_1+\ell_2}.
}
$$

If the partition $\mathcal P_s$ is selected adaptively by a threshold detector, the exact pushforward still exists whenever the selection rule is measurable. However, the map is then state-dependent and not a homogeneous one-parameter RG flow. Fixed-point analysis should therefore be performed either on a frozen deterministic blocking rule or on the induced stochastic/adaptive RG map.

---

## 10. Gauge covariance of the coarse-graining map

There are two different actions one might call “gauge.” They should not be conflated.

### 10.1 Global internal frame relabeling

For a single $g\in G$, define

$$
U_i\mapsto U_i g.
$$

Then

$$
\Omega_{ij}=U_iU_j^{-1}
\mapsto
U_i g (U_j g)^{-1}
=U_iU_j^{-1}.
$$

This is a redundancy of the frame coordinates.

### 10.2 Base-local diagonal active gauge action

For a smooth $h:\mathcal C\to G$, define

$$
U_i(c)\mapsto h(c)U_i(c),
\qquad
q_i(c)\mapsto h(c)_\# q_i(c),
\qquad
p_i(c)\mapsto h(c)_\# p_i(c).
$$

Then

$$
\Omega_{ij}(c)
\mapsto
h(c)U_i(c)U_j(c)^{-1}h(c)^{-1}.
$$

The transport is covariant, not invariant.

### Theorem 3: covariance of $R_s$

Assume:

1. $G$ is compact and the Karcher mean is locally unique.
2. The weights $w_i^I$ are gauge-invariant functions of gauge-invariant quantities, such as transported KL divergences and support masks.
3. The belief and model barycenters are defined by forward-KL minimization after transporting all children to the parent frame.

Then the coarse-graining map is gauge-covariant:

$$
\boxed{
R_s(h\cdot X_s)=h\cdot R_s(X_s).
}
$$

### Proof sketch

The Karcher frame is defined by minimizing

$$
\sum_i w_i^I d_G(U,U_i)^2.
$$

For a bi-invariant metric,

$$
d_G(hU,hU_i)=d_G(U,U_i).
$$

Therefore if $U_I$ minimizes the original objective, $hU_I$ minimizes the transformed objective. Hence

$$
U_I\mapsto hU_I.
$$

The child-to-parent transport transforms as

$$
\Omega_{Ii}=U_IU_i^{-1}
\mapsto
hU_IU_i^{-1}h^{-1}.
$$

Transported child distributions transform by common pushforward. Since KL is invariant under common invertible pushforward,

$$
\mathrm{KL}(a\|b)=\mathrm{KL}(h_\#a\|h_\#b),
$$

and since the forward-KL barycenter is unique in the Gaussian family, the parent barycenter transforms by the same pushforward. Thus $q_I\mapsto h_\#q_I$ and similarly $p_I\mapsto h_\#p_I$. The support mask is unchanged. Therefore $R_s$ commutes with the diagonal gauge action. $\square$

This proves that the coarse-graining operation is compatible with the gauge structure in the compact/local-uniqueness regime.

---

## 11. Closure problem and projected RG

The exact pushforward free energy $\mathcal F_{s+1}^{\mathrm{exact}}$ generally contains all interactions allowed by symmetry: higher-order parent interactions, non-Gaussian terms, long-range couplings, nonlinear frame terms, and memory terms. Thus exact RG does **not** automatically preserve the original multi-agent Gaussian KL form.

Define the agent-form functional class

$$
\mathfrak M=\{\mathcal F_\theta:\theta\in\Theta\},
$$

where $\theta$ contains the parent means, covariances, frames, masks, raw couplings, transport parameters, curvature couplings, and any retained hyperparameters.

Define the projection of the exact free energy back into the ansatz class by

$$
\boxed{
\Pi_{\mathfrak M}(\mathcal F)=\arg\min_{\theta\in\Theta}\inf_{a\in\mathbb R}
\|\mathcal F-\mathcal F_\theta-a\|_{\mathcal B}.
}
$$

Here $\|\cdot\|_{\mathcal B}$ is a chosen norm over parent configurations, for example:

- an $L^2$ norm under $\mathbb P_{s+1}$,
- a local Taylor norm near a saddle,
- a supremum norm over a coherence neighborhood,
- or an empirical simulation norm over sampled trajectories.

The projected RG map is

$$
\boxed{
\theta_{s+1}=R_{\mathfrak M}(\theta_s):=\Pi_{\mathfrak M}(\mathcal F_{s+1}^{\mathrm{exact}}).
}
$$

Define the closure residual

$$
\boxed{
\varepsilon_{s+1}=\inf_{\theta\in\Theta}\inf_{a\in\mathbb R}
\frac{
\|\mathcal F_{s+1}^{\mathrm{exact}}-\mathcal F_\theta-a\|_{\mathcal B}
}{
\tau+\|\mathcal F_{s+1}^{\mathrm{exact}}\|_{\mathcal B}
}.
}
$$

Then:

- exact closure means $\varepsilon_{s+1}=0$;
- approximate closure means $\varepsilon_{s+1}\ll1$;
- failure of closure means the parent theory needs additional operators/interactions.

This makes the RG claim falsifiable and numerically testable.

---

## 12. Linear-Gaussian exact closure theorem

The cleanest rigorous closure theorem is obtained in a restricted but important regime.

### Assumptions

Fix a cluster $I$. Assume:

1. **Compact gauge:** $G=\mathrm{SO}(K)$ or compact equivalent.
2. **Flat intra-cluster transport:** after transporting all children to the parent frame, the intra-cluster transports factor consistently:
   $$
\Omega_{ij}=\Omega_{iI}\Omega_{Ij}
$$
   within the cluster.
3. **Gaussian/quadratic regime:** near the parent barycenter, the belief-sector free energy is quadratic in the transported means.
4. **Fixed or slowly varying covariance:** covariance variations are either fixed or included in a quadratic Gaussian tangent-space expansion.
5. **Positive internal spectral gap:** internal disagreement modes have a Hessian bounded below by $m_I>0$ on the constrained subspace.
6. **Small anharmonicity:** cubic and higher derivatives are negligible or controlled.

Transport child means into the parent frame:

$$
\widehat\mu_i^I=\Omega_{Ii}\mu_i.
$$

Decompose them as

$$
\boxed{
\widehat\mu_i^I=\mu_I+\xi_i,
}
$$

with barycentric constraint

$$
\boxed{
\sum_{i\in I}w_i^I\xi_i=0.
}
$$

The parent coordinate $\mu_I$ is the retained collective coordinate. The variables $\xi_i$ are internal disagreement modes to be integrated out.

---

## 13. Internal Laplacian and constrained gap

Assume equal transported covariance $\Sigma_I$ for clarity and define the precision

$$
\Lambda_I=\Sigma_I^{-1}.
$$

The pairwise quadratic alignment energy inside $I$ has the form

$$
\mathcal F_I^{\mathrm{int}}=\frac12
\sum_{i,j\in I}a_{ij}^I
(\xi_i-\xi_j)^\top\Lambda_I(\xi_i-\xi_j),
$$

where $a_{ij}^I\ge0$ are symmetric intra-cluster edge weights.

Let $L_I$ be the weighted graph Laplacian:

$$
(L_I)_{ii}=\sum_{j\ne i}a_{ij}^I,
\qquad
(L_I)_{ij}=-a_{ij}^I\quad (i\ne j).
$$

Then

$$
\boxed{
\mathcal F_I^{\mathrm{int}}=\frac12\xi_I^\top(L_I\otimes\Lambda_I)\xi_I.
}
$$

The unconstrained Laplacian has a zero mode corresponding to uniform shifts of all children. That zero mode is exactly the retained parent coordinate. On the constrained subspace

$$
\mathcal N_I=\left\{\xi:\sum_iw_i^I\xi_i=0\right\},
$$

define the weighted internal gap

$$
\boxed{
\lambda_{I,w}=\inf_{\xi\in\mathcal N_I,\xi\ne0}
\frac{\xi^\top L_I\xi}{\xi^\top\xi}.
}
$$

Then the internal Hessian is

$$
\boxed{
H_I^\perp=L_I|_{\mathcal N_I}\otimes\Lambda_I,
}
$$

and the internal mass/stiffness gap is

$$
\boxed{
 m_I=\lambda_{I,w}\lambda_{\min}(\Lambda_I).
}
$$

If $m_I>0$, internal disagreement modes are massive and can be integrated out perturbatively.

---

## 14. Exact quadratic integration and Schur complement

Let the parent/slow variables be $Y$ and the internal modes be $\xi$. Near a coherent cluster, write the microscopic free energy as a block quadratic form:

$$
\mathcal F_s(Y,\xi)=\mathcal F_0+
\frac12
\begin{bmatrix}
Y-Y_0\\
\xi
\end{bmatrix}^\top
\begin{bmatrix}
A_{YY} & A_{Y\xi}\\
A_{\xi Y} & A_{\xi\xi}
\end{bmatrix}
\begin{bmatrix}
Y-Y_0\\
\xi
\end{bmatrix}.
$$

Assume $A_{\xi\xi}$ is positive definite on the constrained internal subspace. Then the minimizing internal mode is

$$
\boxed{
\xi^*(Y)=-A_{\xi\xi}^{-1}A_{\xi Y}(Y-Y_0).
}
$$

Substituting back gives the effective free energy

$$
\boxed{
\mathcal F_{\mathrm{eff}}(Y)=\mathcal F_0+\frac12(Y-Y_0)^\top
A_{\mathrm{eff}}
(Y-Y_0),
}
$$

where

$$
\boxed{
A_{\mathrm{eff}}=A_{YY}-A_{Y\xi}A_{\xi\xi}^{-1}A_{\xi Y}.
}
$$

This is the Schur complement.

At finite temperature, integrating rather than minimizing yields

$$
\int e^{-\mathcal F_s(Y,\xi)/\tau}\,d\xi=C(\det A_{\xi\xi})^{-1/2}\exp[-\mathcal F_{\mathrm{eff}}(Y)/\tau].
$$

Thus

$$
\boxed{
\mathcal F_{s+1}^{\mathrm{exact}}(Y)=\mathcal F_{\mathrm{eff}}(Y)+\frac{\tau}{2}\log\det{}' A_{\xi\xi}(Y)+\mathrm{const}.
}
$$

The prime means determinant restricted to the internal constrained subspace.

If $A_{\xi\xi}$ is independent of $Y$, the log-determinant is an additive constant. If $A_{\xi\xi}$ depends smoothly on $Y$, the log-determinant contributes a local parent potential.

### Theorem 4: exact closure in the quadratic regime

Under the compact-gauge, flat-transport, Gaussian-quadratic, positive-gap assumptions above, the exact renormalized free energy remains in the Gaussian quadratic KL functional class up to additive constants and log-determinant local parent terms. The renormalized quadratic form is the Schur complement

$$
A_{\mathrm{eff}}=A_{YY}-A_{Y\xi}A_{\xi\xi}^{-1}A_{\xi Y}.
$$

### Proof

The microscopic free energy is quadratic in $(Y,\xi)$, and the constraint subspace removes only the zero mode corresponding to the retained parent coordinate. Integration over $\xi$ is a finite-dimensional Gaussian integral. Gaussian integration produces another quadratic form in $Y$, with Schur-complement matrix $A_{\mathrm{eff}}$, plus the log-determinant of the internal Hessian. A quadratic form in Gaussian natural parameters is equivalent to a Gaussian KL/free-energy term with renormalized precision/couplings. Therefore the class closes. $\square$

This theorem is the rigorous core of the RG section.

---

## 15. Laplace approximation outside the exactly quadratic regime

If $\mathcal F_s(Y,\xi)$ is not exactly quadratic, expand around the internal saddle $\xi^*(Y)$:

$$
\mathcal F_s(Y,\xi)=\mathcal F_s(Y,\xi^*)+\frac12\eta^\top H_I^\perp(Y)\eta+\frac{1}{3!}T_3(Y)[\eta^3]
+
\frac{1}{4!}T_4(Y)[\eta^4]
+\cdots,
$$

where $\eta=\xi-\xi^*(Y)$.

Laplace integration gives

$$
\boxed{
\mathcal F_{s+1}^{\mathrm{exact}}(Y)=\mathcal F_s(Y,\xi^*(Y))+\frac{\tau}{2}\log\det{}'H_I^\perp(Y)
+
\mathcal E_{\mathrm{anh}}(Y)
+
\mathrm{const}.
}
$$

A standard local bound has the form

$$
\boxed{
\mathcal E_{\mathrm{anh}}=O\!\left(
\tau^2\|H^{-1}\|^3\|T_3\|^2
+
\tau^2\|H^{-1}\|^2\|T_4\|
\right),
}
$$

with all tensors evaluated in a neighborhood of the saddle. The cubic term enters quadratically because odd Gaussian moments vanish at leading order.

Since $\|H^{-1}\|\lesssim m_I^{-1}$, the anharmonic error is small when

$$
\boxed{
\tau\|T_3\|m_I^{-3/2}\ll1,
\qquad
\tau\|T_4\|m_I^{-2}\ll1.
}
$$

This formalizes the high-coherence, spectrally gapped, weakly anharmonic regime.

---

## 16. Renormalized raw couplings

Attention weights $\beta_{ij}$ are row-normalized, so they should not be coarse-grained directly. Instead use raw conductances.

Let an edge energy be

$$
E_{ij}=\mathrm{KL}(q_i\|\Omega_{ij}q_j)
$$

or the combined belief/model edge energy. Define

$$
\boxed{
\kappa_{ij}=\pi_{ij}\exp(-E_{ij}/\tau),
}
$$

where $\pi_{ij}$ is an edge prior or base measure. Then

$$
\boxed{
\beta_{ij}=\frac{\kappa_{ij}}{\sum_k\kappa_{ik}}.
}
$$

For parent clusters $I,J$, define the renormalized conductance

$$
\boxed{
\kappa_{IJ}^{R}=\sum_{i\in I}\sum_{j\in J}w_i^I w_j^J\kappa_{ij}.
}
$$

Then the parent attention/coupling is

$$
\boxed{
\beta_{IJ}^{R}=\frac{\kappa_{IJ}^{R}}{\sum_L\kappa_{IL}^{R}}.
}
$$

This has three advantages:

1. It preserves row normalization after coarse-graining.
2. It avoids the incorrect identity $\sum_J\sum_{i\in I,j\in J}\beta_{ij}=1$, which generally fails.
3. It reduces to the common child conductance when all children in a cluster are identical.

---

## 17. Renormalized inter-cluster transport

For $i\in I$, $j\in J$, transport the microscopic edge into parent coordinates:

$$
\boxed{
\Theta_{ij}^{IJ}=\Omega_{Ii}\Omega_{ij}\Omega_{jJ}.
}
$$

Here:

- $\Omega_{Ii}$ transports from child $i$'s frame to parent $I$'s frame.
- $\Omega_{ij}$ transports from child $j$'s frame to child $i$'s frame.
- $\Omega_{jJ}$ transports from parent $J$'s frame to child $j$'s frame, depending on convention. If the convention is reversed, use the inverse consistently.

Define the parent transport by a weighted Karcher mean of these transported microscopic edges:

$$
\boxed{
\Omega_{IJ}^{R}=\arg\min_{\Omega\in G}
\sum_{i\in I}\sum_{j\in J}
 w_i^I w_j^J\kappa_{ij}
 d_G(\Omega,\Theta_{ij}^{IJ})^2.
}
$$

If the microscopic transport is flat across the cluster pair, then all $\Theta_{ij}^{IJ}$ coincide and

$$
\Omega_{IJ}^{R}=\Theta_{ij}^{IJ}.
$$

If they do not coincide, define the transport-spread or holonomy residual

$$
\boxed{
\mathcal H_{IJ}=\sum_{i\in I}\sum_{j\in J}
 w_i^I w_j^J\kappa_{ij}
 d_G(\Omega_{IJ}^{R},\Theta_{ij}^{IJ})^2.
}
$$

This residual measures how much pairwise microscopic transport fails to descend to a single parent-parent transport. It is one of the main closure-error terms.

---

## 18. Edge-marginal compatibility of barycentric weights

Let the row and column conductance marginals for a cluster pair $(I,J)$ be

$$
r_i^{I\to J}=\sum_{j\in J}\kappa_{ij},
\qquad
s_j^{J\leftarrow I}=\sum_{i\in I}\kappa_{ij}.
$$

For first-order internal residuals to vanish, the cluster weights should match these edge marginals:

$$
\boxed{
 w_i^I = \frac{r_i^{I\to J}}{\sum_{i'\in I}r_{i'}^{I\to J}},
 \qquad
 w_j^J = \frac{s_j^{J\leftarrow I}}{\sum_{j'\in J}s_{j'}^{J\leftarrow I}}.
}
$$

A single cluster $I$ usually interacts with several partners $J_1,\ldots,J_m$. One set of weights $w_i^I$ cannot satisfy every partner-specific marginal condition unless the marginals are colinear. Define the mismatch

$$
\boxed{
\delta_{I}^{\mathrm{marg}}=\sum_J \rho_{IJ}
\sum_{i\in I}
\left|
 w_i^I - \frac{r_i^{I\to J}}{\sum_{i'}r_{i'}^{I\to J}}
\right|,
}
$$

where $\rho_{IJ}$ are normalized partner weights. This mismatch also contributes to the closure residual.

---

## 19. Closure residual bound

Define the belief barycentric dispersion

$$
\boxed{
V_I^{(q)}=\sum_{i\in I}w_i^I
\mathrm{KL}\!\left(\widetilde q_i^I\middle\|q_I\right).
}
$$

Define the model-sector dispersion

$$
\boxed{
V_I^{(p)}=\sum_{i\in I}w_i^I
\mathrm{KL}\!\left(\widetilde p_i^I\middle\|p_I\right).
}
$$

Let

$$
V_I=V_I^{(q)}+V_I^{(p)}.
$$

A useful closure statement is:

### Theorem 5: controlled approximate closure

Assume:

1. compact gauge group and locally unique barycenters;
2. cluster dispersions $V_I$ are small;
3. inter-cluster transport spread $\mathcal H_{IJ}$ is small;
4. edge-marginal mismatch $\delta_I^{\mathrm{marg}}$ is small;
5. internal Hessian gap $m_I$ is uniformly positive;
6. anharmonic derivatives are bounded;
7. non-Gaussian cumulants beyond second order are small in the chosen closure norm;
8. slow/parent modes are separated from internal modes.

Then the exact renormalized free energy satisfies

$$
\boxed{
\varepsilon_{s+1}
\le
C_1\sum_I V_I^{3/2}
+
C_2\sum_{I,J}\mathcal H_{IJ}
+
C_3\sum_I\delta_I^{\mathrm{marg}}
+
C_4\sum_I m_I^{-1}\|A_{Y\xi}^{(I)}\|^2
+
C_5\sum_I \mathcal E_{\mathrm{anh}}^{(I)}
+
C_6\sum_I\mathcal E_{\mathrm{nonG}}^{(I)}.
}
$$

Here:

- $V_I^{3/2}$ appears because the Gaussian/KL expansion error after matching first two moments starts at cubic order in small internal deviations.
- $\mathcal H_{IJ}$ measures failure of microscopic transports to collapse to a single parent transport.
- $\delta_I^{\mathrm{marg}}$ measures incompatibility between barycentric weights and edge marginals.
- $m_I^{-1}\|A_{Y\xi}^{(I)}\|^2$ is the Schur/slaving correction from slow-internal coupling.
- $\mathcal E_{\mathrm{anh}}^{(I)}$ is the Laplace anharmonic correction.
- $\mathcal E_{\mathrm{nonG}}^{(I)}$ measures failure of the Gaussian family to be closed under the exact pushforward.

### Interpretation

This theorem says the parent theory is reliable precisely when:

$$
\boxed{
\text{small dispersion}
+
\text{small holonomy spread}
+
\text{compatible weights}
+
\text{large internal gap}
+
\text{weak anharmonicity}
\quad\Longrightarrow\quad
\text{small closure residual}.
}
$$

This is the rigorous version of the intuitive statement that coherent clusters admit effective parent descriptions.

---

## 20. Fixed points and relevant directions

Once the projected RG map is defined,

$$
\theta_{s+1}=R_{\mathfrak M}(\theta_s),
$$

an RG fixed point is

$$
\boxed{
\theta^*=R_{\mathfrak M}(\theta^*).
}
$$

Linearize around $\theta^*$:

$$
\delta\theta_{s+1}=DR_{\mathfrak M}(\theta^*)\delta\theta_s.
$$

Let

$$
DR_{\mathfrak M}(\theta^*)v_a=\lambda_av_a.
$$

Then:

$$
\boxed{
|\lambda_a|>1\quad\Rightarrow\quad v_a\text{ relevant},
}
$$

$$
\boxed{
|\lambda_a|<1\quad\Rightarrow\quad v_a\text{ irrelevant},
}
$$

$$
\boxed{
|\lambda_a|=1\quad\Rightarrow\quad v_a\text{ marginal}.
}
$$

This gives a literal RG analysis once $R_{\mathfrak M}$ is estimated analytically or numerically.

Important qualification:

- Fixed points of $R_{\mathfrak M}$ are fixed points of the projected/closed parent theory.
- They are fixed points of the exact pushforward only when the closure residual vanishes or is negligible.

---

## 21. Finite-size scaling and universality tests

To support a stronger phase-transition or universality claim, measure observables across system sizes $N$, grid sizes, seeds, and thresholds.

Candidate observables:

1. Mean cluster size:
   $$
S_N(t)=\frac{1}{N}\sum_I |I|^2.
$$
2. Susceptibility-like variance:
   $$
\chi_N(t)=N\left(\mathbb E[S_N(t)^2]-\mathbb E[S_N(t)]^2\right).
$$
3. Correlation length over the overlap graph:
   $$
\xi_N(t)^2
   =
   \frac{\sum_{i,j}d(i,j)^2 C_{ij}(t)}{\sum_{i,j}C_{ij}(t)}.
$$
4. Coherence order parameter:
   $$
M_N(t)=\frac{1}{N^2}\sum_{i,j}\exp[-\mathrm{KL}(q_i\|\Omega_{ij}q_j)/\tau_q].
$$
5. Largest meta-agent fraction:
   $$
P_\infty(N,t)=\frac{1}{N}\max_I |I|.
$$

A finite-size scaling ansatz would be

$$
\boxed{
M_N(g)=N^{-\beta/\nu}\Phi_M\left((g-g_c)N^{1/\nu}\right),
}
$$

$$
\boxed{
\chi_N(g)=N^{\gamma/\nu}\Phi_\chi\left((g-g_c)N^{1/\nu}\right),
}
$$

where $g$ is a control parameter, such as coupling strength, temperature, or threshold. Evidence for universality would require collapse of curves across $N$ and robustness across microscopic details.

Until this is done, the manuscript should say “reorganization event” or “phase-transition-like event,” not “phase transition” in the statistical-mechanical sense.

---

## 22. Detector as a sufficient-condition surrogate

The existing detector uses coherence scores of the schematic form

$$
1-\overline{\mathrm{KL}}.
$$

This is not intrinsically normalized because KL is unbounded. Replace it with exponential coherence.

Define

$$
\boxed{
C_q(I)=\exp[-V_I^{(q)}/\tau_q],
}
$$

$$
\boxed{
C_p(I)=\exp[-V_I^{(p)}/\tau_p].
}
$$

Define normalized presence/overlap

$$
\boxed{
P_I=\frac{1}{|\Lambda_s|}
\sum_{c\in\Lambda_s}\chi_I(c)
}
$$

or, if a local version is desired,

$$
P_I(c)=\mathbf 1\!\left\{\sum_{i\in I}\chi_i(c)\ge2\right\}.
$$

Then define the detector score

$$
\boxed{
\Gamma_I=P_I\exp[-V_I^{(q)}/\tau_q-V_I^{(p)}/\tau_p].
}
$$

This satisfies

$$
0\le\Gamma_I\le1.
$$

The detector selects $I$ if

$$
\boxed{
\Gamma_I>\Gamma_{\min},
\qquad
|I|\ge N_{\min},
\qquad
m_I>m_{\min}.
}
$$

The gap condition $m_I>m_{\min}$ is important if the detector is supposed to approximate an RG closure condition rather than only a similarity threshold.

---

## 23. Variational retention gain

Define the optimized microscopic description length of cluster $I$ by

$$
\mathcal L_{\mathrm{micro}}(I)=\mathcal F_{\mathrm{micro}}^*(I)+C_{\mathrm{micro}}(I).
$$

Define the optimized parent description length by

$$
\mathcal L_{\mathrm{parent}}(I)=\mathcal F_{\mathrm{parent}}^*(I)+C_{\mathrm{parent}}(I)+\varepsilon_I.
$$

The variational retention gain is

$$
\boxed{
\Delta_I=\mathcal L_{\mathrm{micro}}(I)-\mathcal L_{\mathrm{parent}}(I).
}
$$

A parent is retained when

$$
\boxed{
\Delta_I>0.
}
$$

This is the principled version of meta-agent formation.

The threshold detector should be justified as a computationally cheap sufficient condition for $\Delta_I>0$, not as the definition of emergence.

---

## 24. Theorem 6: detector implies positive retention gain under explicit assumptions

### Assumptions

Suppose that, for candidate cluster $I$, the parent approximation satisfies

$$
\mathcal F_{\mathrm{parent}}^*(I)
\le
\mathcal F_{\mathrm{micro}}^*(I)-A_I+L_qV_I^{(q)}+L_pV_I^{(p)}+\varepsilon_I,
$$

where:

- $A_I>0$ is the free-energy saving from replacing pairwise child-child alignment with child-parent alignment;
- $L_q,L_p$ are local Lipschitz constants measuring sensitivity to belief/model dispersion;
- $\varepsilon_I$ is the closure residual.

Let the net complexity cost be

$$
C_I=C_{\mathrm{parent}}(I)-C_{\mathrm{micro}}(I).
$$

Then

$$
\Delta_I>0
$$

whenever

$$
\boxed{
L_qV_I^{(q)}+L_pV_I^{(p)}+\varepsilon_I+C_I<A_I.
}
$$

### Detector implication

The exponential detector condition

$$
\Gamma_I=P_I\exp[-V_I^{(q)}/\tau_q-V_I^{(p)}/\tau_p]>
\Gamma_{\min}
$$

implies

$$
\boxed{
\frac{V_I^{(q)}}{\tau_q}+\frac{V_I^{(p)}}{\tau_p}
<
\log\left(\frac{P_I}{\Gamma_{\min}}\right).
}
$$

For a cluster-specific sufficient condition, it is enough that

$$
\boxed{
\max(L_q\tau_q,L_p\tau_p)
\log\left(\frac{P_I}{\Gamma_{\min}}\right)
+
\varepsilon_I+C_I
<
A_I.
}
$$

For a uniform sufficient condition over all selected clusters, use $P_I\le 1$, giving the conservative bound

$$
\boxed{
\max(L_q\tau_q,L_p\tau_p)
\log\left(\frac{1}{\Gamma_{\min}}\right)
+
\varepsilon_I+C_I
<
A_I.
}
$$

Under this condition,

$$
\boxed{
\Gamma_I>\Gamma_{\min}
\quad\Longrightarrow\quad
\Delta_I>0.
}
$$

### Proof

The first inequality gives

$$
\mathcal L_{\mathrm{parent}}(I)
\le
\mathcal F_{\mathrm{micro}}^*(I)
-
A_I
+
L_qV_I^{(q)}
+
L_pV_I^{(p)}
+
\varepsilon_I
+C_{\mathrm{parent}}(I).
$$

Subtracting from $\mathcal L_{\mathrm{micro}}=\mathcal F_{\mathrm{micro}}^*+C_{\mathrm{micro}}$ gives

$$
\Delta_I
\ge
A_I
-
L_qV_I^{(q)}
-
L_pV_I^{(p)}
-
\varepsilon_I
-C_I.
$$

Thus $\Delta_I>0$ whenever

$$
L_qV_I^{(q)}+L_pV_I^{(p)}+\varepsilon_I+C_I<A_I.
$$

The detector condition bounds the weighted dispersion. If the conservative threshold inequality above holds, then the sufficient condition for positive retention follows. $\square$

This theorem connects the practical detector to the variational rule without pretending the detector is fundamental.

---

## 25. Estimating the saving term $A_I$

A simple estimate is obtained by comparing the number of alignment terms.

Suppose a coherent cluster has pairwise average transported KL

$$
\overline D_{\mathrm{pair}}=\frac{1}{|I|(|I|-1)}
\sum_{i\ne j}\mathrm{KL}(q_i\|\Omega_{ij}q_j),
$$

and child-parent average transported KL

$$
\overline D_{\mathrm{parent}}=\frac{1}{|I|}
\sum_i\mathrm{KL}(\Omega_{Ii}q_i\|q_I).
$$

If microscopic alignment uses all ordered pairs with weight $\lambda_{\mathrm{pair}}$, while the parent description uses child-parent edges with weight $\lambda_{\mathrm{parent}}$, a crude saving estimate is

$$
\boxed{
A_I\approx\lambda_{\mathrm{pair}}|I|(|I|-1)\overline D_{\mathrm{pair}}
-\lambda_{\mathrm{parent}}|I|\overline D_{\mathrm{parent}}.
}
$$

In high coherence, both divergences are small, but the first term scales quadratically in $|I|$, whereas the second scales linearly. A parent becomes favorable when the quadratic saving dominates the linear parent cost and closure residual.

A more precise estimate uses the exact optimized free energies before and after insertion:

$$
\boxed{
A_I=\mathcal F_{\mathrm{micro}}^*(I)-\mathcal F_{\mathrm{parent},no\ overhead}^*(I).
}
$$

This is the quantity to measure in simulations if one wants to validate the detector.

---

## 26. Adiabatic elimination and autonomous parent flow

The RG construction produces an effective parent free energy. To justify autonomous parent dynamics, one also needs internal modes to relax faster than parent modes.

Let the natural-gradient dynamics near a parent cluster be

$$
\begin{bmatrix}
\dot Y\\
\dot\xi
\end{bmatrix}=-\begin{bmatrix}
G_{YY} & G_{Y\xi}\\
G_{\xi Y} & G_{\xi\xi}
\end{bmatrix}^{-1}
\begin{bmatrix}
\nabla_Y\mathcal F\\
\nabla_\xi\mathcal F
\end{bmatrix}.
$$

If the constrained internal Hessian has gap $m_I$, then internal relaxation time scales like

$$
t_{\mathrm{int}}\sim m_I^{-1}.
$$

Let $t_{\mathrm{slow}}$ be the timescale for parent variables. Adiabatic elimination requires

$$
\boxed{
t_{\mathrm{int}}\ll t_{\mathrm{slow}}.
}
$$

Under normal hyperbolicity, there is a slow manifold

$$
\xi=\xi^*(Y)+O(t_{\mathrm{int}}/t_{\mathrm{slow}}).
$$

The effective parent flow is

$$
\boxed{
\dot Y=-\mathrm{grad}_{G_{\mathrm{eff}}}\mathcal F_{s+1}^{\mathrm{exact}}(Y)
+O\!\left(m_I^{-1}\|A_{Y\xi}\|\|\dot Y\|\right)
+O(\varepsilon_I).
}
$$

with effective metric

$$
\boxed{
G_{\mathrm{eff}}=G_{YY}-G_{Y\xi}G_{\xi\xi}^{-1}G_{\xi Y}.
}
$$

This provides a rigorous dynamical condition for when a parent behaves as an autonomous agent rather than merely a summary statistic.

---

## 27. Simulation-facing algorithm

A rigorous implementation should separate candidate selection, exact/projection diagnostics, and parent insertion.

### Algorithm: RG-consistent meta-agent formation

Given scale-$s$ state $X_s$:

1. Build the overlap graph $\mathcal G_s$.
2. Enumerate connected candidate clusters $I$ with $|I|\ge N_{\min}$.
3. For each cluster:
   1. Compute support $\chi_I$.
   2. Compute Karcher frame $U_I$.
   3. Transport children into parent frame.
   4. Compute Gaussian KL barycenters $q_I,p_I$, including covariance dispersion terms.
   5. Compute dispersions $V_I^{(q)},V_I^{(p)}$.
   6. Compute internal graph Laplacian and gap $m_I$.
   7. Estimate closure residual terms: $\mathcal H_{IJ}$, $\delta_I^{\mathrm{marg}}$, anharmonicity, non-Gaussianity.
   8. Estimate retention gain $\Delta_I$, or apply the sufficient-condition detector.
4. Select non-conflicting clusters maximizing total positive retention gain.
5. Insert parent agents.
6. Define parent-parent conductances $\kappa_{IJ}^R$, normalized couplings $\beta_{IJ}^R$, and parent transports $\Omega_{IJ}^R$.
7. Continue dynamics on the enlarged or coarse-grained state.

The key change from a simple threshold detector is that the detector should be interpreted as an approximation to steps 3.6--3.8.

---

## 28. What to change in the manuscript wording

Replace strong wording such as:

The simulation demonstrates renormalization-group emergence of meta-agents.

with:

The theory admits an exact pushforward RG under a gauge-covariant barycentric coarse-graining map. The simulations currently implement a threshold-based candidate selector that approximates the high-coherence, positive-retention regime of this RG construction. Thus the reported hierarchy is detector-mediated but mathematically interpretable as an approximate RG coarse-graining when closure residuals are small.

Replace:

Meta-agents emerge from the free energy functional.

with:

Meta-agents are licensed by a variational retention criterion. The current implementation uses a threshold detector as a surrogate for this criterion; verifying equivalence between detector-selected and retention-selected hierarchies is an empirical follow-up.

Replace:

The hierarchy is a renormalization group flow.

with:

The exact pushforward of the variational Gibbs measure defines a genuine RG transformation. The implemented hierarchy is an approximation to the projected RG flow in the Gaussian multi-agent ansatz class.

---

## 29. Minimal theorem block for direct insertion

The following concise block can be inserted into the manuscript.

### Proposition: exact RG step

Let $R_s:X_s\to X_{s+1}$ be the gauge-covariant barycentric coarse-graining map defined by parent masks, Karcher frame means, and forward-KL Gaussian barycenters. Let

$$
d\mathbb P_s=Z_s^{-1}e^{-\mathcal F_s/\tau}d\nu_s.
$$

Then

$$
\mathbb P_{s+1}=(R_s)_*\mathbb P_s
$$

has exact renormalized free energy

$$
e^{-\mathcal F_{s+1}^{\mathrm{exact}}(Y)/\tau}=\int\delta(Y-R_s(X))e^{-\mathcal F_s(X)/\tau}d\nu_s(X),
$$

up to an additive constant. This step preserves retained observables and satisfies the composition law

$$
(R_{s+1})_*(R_s)_*=(R_{s+1}\circ R_s)_*.
$$

### Proposition: Gaussian closure

If $G$ is compact, cluster transports are flat within each block, the microscopic free energy is quadratic in Gaussian natural coordinates near the barycenter, and the constrained internal Hessian has positive gap, then integrating out internal disagreement modes yields another Gaussian quadratic parent free energy. The renormalized quadratic form is

$$
A_{\mathrm{eff}}=A_{YY}-A_{Y\xi}A_{\xi\xi}^{-1}A_{\xi Y},
$$

with a finite-temperature correction

$$
\frac{\tau}{2}\log\det{}'A_{\xi\xi}.
$$

Thus the multi-agent Gaussian KL class is exactly closed in this regime up to additive constants and local log-determinant terms.

### Proposition: approximate closure

Away from the exact quadratic regime, the closure residual obeys a bound of the schematic form

$$
\varepsilon_{s+1}
\le
C_1\sum_I V_I^{3/2}
+C_2\sum_{I,J}\mathcal H_{IJ}
+C_3\sum_I\delta_I^{\mathrm{marg}}
+C_4\sum_I m_I^{-1}\|A_{Y\xi}^{(I)}\|^2
+C_5\sum_I\mathcal E_{\mathrm{anh}}^{(I)}
+C_6\sum_I\mathcal E_{\mathrm{nonG}}^{(I)}.
$$

Thus closure is controlled by small barycentric dispersion, small holonomy spread, compatible weights, a positive internal gap, weak anharmonicity, and approximate Gaussianity.

### Proposition: detector-retention link

With bounded detector score

$$
\Gamma_I=P_I\exp[-V_I^{(q)}/\tau_q-V_I^{(p)}/\tau_p],
$$

and retention gain

$$
\Delta_I=\mathcal L_{\mathrm{micro}}(I)-\mathcal L_{\mathrm{parent}}(I),
$$

if, cluster by cluster,

$$
\max(L_q\tau_q,L_p\tau_p)
\log(P_I/\Gamma_{\min})+
\varepsilon_I+C_I<A_I,
$$

or uniformly,

$$
\max(L_q\tau_q,L_p\tau_p)
\log(1/\Gamma_{\min})+
\varepsilon_I+C_I<A_I,
$$

then

$$
\Gamma_I>\Gamma_{\min}\quad\Longrightarrow\quad\Delta_I>0.
$$

Thus the threshold detector is a sufficient-condition surrogate for positive variational retention gain under explicit high-coherence and small-residual assumptions.

---

## 30. Bottom line

The rigorous RG route is:

$$
\boxed{
\text{microscopic agent Gibbs measure}
\xrightarrow{\text{gauge-covariant }R_s}
\text{exact pushforward parent measure}
\xrightarrow{\Pi_{\mathfrak M}}
\text{closed Gaussian multi-agent parent theory}
}
$$

with the logical chain

$$
\boxed{
\text{coherent cluster}
\Rightarrow
\text{small dispersion and large internal gap}
\Rightarrow
\text{small closure residual}
\Rightarrow
\text{valid parent effective theory}
\Rightarrow
\text{positive retention gain}
\Rightarrow
\text{meta-agent retained}.
}
$$

This makes the section rigorous without pretending that the current threshold simulation alone proves a universal RG flow or a statistical-mechanical phase transition.
