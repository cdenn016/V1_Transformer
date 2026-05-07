# Rigorous Renormalization Derivation for Meta-Agent Emergence

## 0. Purpose and status

This note gives a corrected, step-by-step renormalization construction for meta-agent emergence in the gauge-covariant variational free-energy framework. The purpose is to replace heuristic parent creation by a controlled coarse-graining statement:

> A meta-agent emerges when a gauge-covariant coarse variable is produced by the renormalized free-energy functional, remains stable under internal fluctuations, approximately closes its own dynamics, and is retained by a specified selection principle.

The construction is written in finite-dimensional form, for example on a finite grid approximation of the base space. This avoids unnecessary infinite-dimensional measure-theoretic issues. A continuum version requires the usual regularity and mesh-refinement assumptions.

This corrected version makes the following qualifications explicit:

1. The rigorous gauge-covariance theorem is first stated for compact/unimodular gauge groups, such as $SO(K)$, or for noncompact groups after gauge fixing or measure regularization.
2. Passive local gauge transformations are common base-local transformations $h(c)$ acting on all frames over the same base point, not independent child-by-child active frame changes.
3. Parent beliefs are defined by transported M-projection barycenters, i.e. moment-matching minimizers of $\sum_i w_iD_{\mathrm{KL}}(\widetilde q_i\Vert q)$.
4. Inter-block renormalized couplings are unweighted sums of microscopic couplings unless the microscopic functional itself includes block weights.
5. The internal stability constant is a weighted constrained spectral gap, not always the ordinary $\lambda_2$ of the graph Laplacian.
6. RG relevance and MDL/Bayesian retention are distinct notions and should not be conflated.
7. Adiabatic elimination requires either metric block-diagonality or the Schur-complement effective metric.

---

## 1. Microscopic agent state at scale $s$

Let the scale-$s$ agent index set be

$$
I_s=\{1,\dots,N_s\}.
$$

Each agent $i\in I_s$ is defined over a support region

$$
U_i\subset\mathcal C,
$$

where $\mathcal C$ is the base space. On a finite grid, $U_i$ is represented by a binary or soft mask

$$
\chi_i(c)\in\{0,1\}
\quad\text{or}\quad
\chi_i(c)\in[0,1].
$$

Each agent carries a belief field

$$
q_i(c)=\mathcal N(\mu_i(c),\Sigma_i(c)),
$$

a model/prior field

$$
p_i(c)=\mathcal N(m_i(c),S_i(c)),
$$

and a gauge frame

$$
G_i(c)\in\mathcal G.
$$

The notation $G_i(c)$ is used for the group-valued frame to avoid conflict with the support $U_i$.

The pure-gauge transport from agent $j$ to agent $i$ is

$$
\Omega_{ij}(c)=G_i(c)G_j(c)^{-1}.
$$

For an independent curved connection, $\Omega_{ij}$ should instead be a path-ordered parallel transport. The derivation below applies to either case as long as the transport law is gauge-covariant and the relevant measures are well-defined.

---

## 2. Microscopic free-energy functional

Let the full scale-$s$ microscopic state be

$$
X_s=\{q_i,p_i,G_i,\chi_i\}_{i\in I_s}.
$$

A representative belief-sector free energy is

$$
\mathcal F_s^q(X_s)
=
\sum_{i\in I_s}\int_{U_i}
D_{\mathrm{KL}}(q_i(c)\Vert p_i(c))\,dc
+
\sum_{i,j\in I_s}\int_{U_i\cap U_j}
\beta_{ij}(c)
D_{\mathrm{KL}}\!\left(
q_i(c)\middle\Vert\Omega_{ij}(c)q_j(c)
\right)dc.
$$

The full functional may include model-sector terms, gauge-frame regularization, curvature penalties, entropy penalties, observation/environment terms, and timescale-dependent couplings:

$$
\mathcal F_s(X_s)
=
\mathcal F_s^q(X_s)
+\mathcal F_s^p(X_s)
+\mathcal F_s^{\mathrm{gauge}}(X_s)
+\mathcal F_s^{\mathrm{obs}}(X_s)
+\cdots.
$$

Unless explicitly stated otherwise, the cross-block interaction term contains no additional factors of block barycenter weights $w_i^Bw_j^C$. This convention matters for the renormalized inter-block coupling derived below.

The structural assumptions are:

1. $\mathcal F_s$ is well-defined on the microscopic state space.
2. $\mathcal F_s$ is invariant under passive gauge transformations.
3. The reference measure over microscopic states is invariant, or else a gauge fixing/regularization has been chosen.
4. For noncompact groups such as $GL(K)$, the rigorous theorem requires an explicit gauge slice, regulator, or Radon--Nikodym correction. The compact $SO(K)$ case is the clean baseline.

---

## 3. Microscopic Gibbs/variational measure

Define the scale-$s$ Gibbs or variational measure

$$
d\mathbb P_s(X_s)
=
\frac{1}{Z_s}\exp\left[-\frac{1}{\tau}\mathcal F_s(X_s)\right]d\nu_s(X_s),
$$

where $d\nu_s$ is the reference measure over Gaussian parameters, masks, and gauge frames, and

$$
Z_s
=
\int\exp\left[-\frac{1}{\tau}\mathcal F_s(X_s)\right]d\nu_s(X_s).
$$

Here $\tau>0$ is the RG/statistical temperature. In the variational or zero-temperature limit,

$$
\tau\to0,
$$

log-sum-exp coarse-graining becomes minimization.

---

## 4. Overlap hypergraph and admissible blocks

Define the pairwise overlap graph at scale $s$ by

$$
(i,j)\in E_s
\quad\Longleftrightarrow\quad
U_i\cap U_j\neq\varnothing.
$$

More generally, define an overlap hypergraph whose hyperedges are nonempty multi-agent overlaps.

A subset $B\subset I_s$ is an admissible block if its induced pairwise overlap graph is connected. That is, for any $i,j\in B$, there exists a chain

$$
i=i_0,i_1,\dots,i_m=j
$$

such that

$$
U_{i_a}\cap U_{i_{a+1}}\neq\varnothing
$$

for each $a$.

The parent support is not the union of child supports. It is the multi-child overlap region

$$
U_B
=
\left\{c\in\mathcal C:\sum_{i\in B}\chi_i(c)\ge2\right\}.
$$

Equivalently,

$$
U_B
=
\bigcup_{i<j,\;i,j\in B}(U_i\cap U_j),
$$

possibly restricted to a connected component.

This enforces the rule:

> A parent lives strictly over regions where two or more children overlap. The parent support does not grow into single-child regions.

For each point $c\in U_B$, define the active child set

$$
B(c)=\{i\in B:c\in U_i\}.
$$

Choose normalized barycenter weights

$$
w_i^B(c)\ge0,
\qquad
\sum_{i\in B(c)}w_i^B(c)=1.
$$

These may be uniform, support-weighted, precision-weighted, coherence-weighted, or edge-marginal weights. The choice affects first-order closure residuals.

---

## 5. Parent gauge frame as a Karcher barycenter

The parent gauge frame is defined as the group barycenter

$$
G_B(c)
=
\arg\min_{G\in\mathcal G}
\sum_{i\in B(c)}w_i^B(c)d_{\mathcal G}(G,G_i(c))^2,
$$

where $d_{\mathcal G}$ is a chosen geodesic distance on the gauge group.

For compact groups such as $SO(K)$, this barycenter is locally unique when the child frames lie in a geodesically convex neighborhood. For noncompact groups such as $GL(K)$, one must choose a gauge slice, impose a regularized metric, or restrict to a well-behaved subgroup.

The child-to-parent transport is

$$
\Omega_{Bi}(c)=G_B(c)G_i(c)^{-1}.
$$

Each child belief is transported into the parent frame:

$$
\widetilde q_i^B(c)=\Omega_{Bi}(c)q_i(c).
$$

For Gaussian beliefs,

$$
\widetilde\mu_i^B(c)=\Omega_{Bi}(c)\mu_i(c),
$$

$$
\widetilde\Sigma_i^B(c)=\Omega_{Bi}(c)\Sigma_i(c)\Omega_{Bi}(c)^\top.
$$

---

## 6. Parent belief as a transported M-projection barycenter

Define the parent belief by the transported M-projection barycenter

$$
q_B(c)
=
\arg\min_{q\in\mathcal Q}
\sum_{i\in B(c)}w_i^B(c)
D_{\mathrm{KL}}\!\left(\widetilde q_i^B(c)\middle\Vert q\right).
$$

This is sometimes informally called a reverse-KL barycenter, but the statistical-geometry name is the M-projection or moment-matching projection. It is natural for coarse-graining because it summarizes the transported child mixture by matching moments.

For Gaussian children

$$
\widetilde q_i^B(c)=\mathcal N(\widetilde\mu_i^B(c),\widetilde\Sigma_i^B(c)),
$$

the minimizer is Gaussian:

$$
q_B(c)=\mathcal N(\mu_B(c),\Sigma_B(c)),
$$

with

$$
\mu_B(c)=\sum_{i\in B(c)}w_i^B(c)\widetilde\mu_i^B(c),
$$

and

$$
\Sigma_B(c)
=
\sum_{i\in B(c)}w_i^B(c)
\left[
\widetilde\Sigma_i^B(c)
+
\left(\widetilde\mu_i^B(c)-\mu_B(c)\right)
\left(\widetilde\mu_i^B(c)-\mu_B(c)\right)^\top
\right].
$$

The second term is the between-child dispersion. It is essential for exact coarse-graining. Dropping it is a high-coherence approximation.

Analogously, define parent model fields $p_B$, latent fields, or precision fields by the same transported barycenter principle.

Thus each admissible block $B$ determines a parent state

$$
Y_B=(q_B,p_B,G_B,\chi_B),
$$

where $\chi_B$ is the indicator or soft mask of $U_B$.

---

## 7. The coarse-graining map

For one block $B$, define

$$
\mathcal R_B(X_s)=Y_B.
$$

For a selected family of admissible blocks $\mathfrak B_s$, define the scale-$s$ RG map

$$
\mathcal R_s(X_s)=Y_{s+1}=\{Y_B\}_{B\in\mathfrak B_s}.
$$

The blocks need not form a partition. Parent agents may overlap, and different child subsets may induce different parents at the same base point. Exact RG only requires $\mathcal R_s$ to be a measurable coarse map.

---

## 8. Exact renormalized free energy by pushforward measure

The cleanest exact RG definition is measure-theoretic. Define the coarse reference measure by pushforward:

$$
\nu_{s+1}=(\mathcal R_s)_*\nu_s.
$$

The unnormalized coarse measure is

$$
d\widetilde\rho_{s+1}(Y)
=(\mathcal R_s)_*\left(
\exp\left[-\frac{1}{\tau}\mathcal F_s(X)\right]d\nu_s(X)
\right).
$$

When this measure is absolutely continuous with respect to $\nu_{s+1}$, define $\mathcal F_{s+1}$ by the Radon--Nikodym derivative

$$
\boxed{
\exp\left[-\frac{1}{\tau}\mathcal F_{s+1}(Y)\right]
=
\frac{d\widetilde\rho_{s+1}}{d\nu_{s+1}}(Y).
}
$$

In coordinates, this is often written as the formal delta-function integral

$$
\mathcal F_{s+1}(Y)
=
-\tau\log\int
\delta(Y-\mathcal R_s(X))
\exp\left[-\frac{1}{\tau}\mathcal F_s(X)\right]d\nu_s(X),
$$

but the pushforward definition is the rigorous statement and avoids hidden Jacobian assumptions.

In the zero-temperature limit,

$$
\boxed{
\mathcal F_{s+1}(Y)
=
\inf_{X:\mathcal R_s(X)=Y}\mathcal F_s(X)
}
$$

up to subleading entropic corrections.

---

## 9. Partition-function preservation

By the pushforward definition,

$$
Z_{s+1}
=
\int\exp\left[-\frac{1}{\tau}\mathcal F_{s+1}(Y)\right]d\nu_{s+1}(Y)
=
\int d\widetilde\rho_{s+1}(Y).
$$

Since $\widetilde\rho_{s+1}$ is the pushforward of the unnormalized microscopic measure,

$$
\int d\widetilde\rho_{s+1}(Y)
=
\int\exp\left[-\frac{1}{\tau}\mathcal F_s(X)\right]d\nu_s(X)
=Z_s.
$$

Thus

$$
\boxed{Z_{s+1}=Z_s.}
$$

This preservation is exact under the pushforward-measure convention.

---

## 10. Coarse-observable preservation

Let $A(Y)$ be any observable depending only on the coarse variables. Then

$$
\mathbb E_{s+1}[A(Y)]
=
\frac{1}{Z_{s+1}}
\int A(Y)d\widetilde\rho_{s+1}(Y).
$$

Because $\widetilde\rho_{s+1}$ is the pushforward of the microscopic unnormalized measure,

$$
\boxed{
\mathbb E_{s+1}[A(Y)]
=
\mathbb E_s[A(\mathcal R_s(X))].
}
$$

Therefore the parent-level theory exactly reproduces every observable that depends only on the retained parent variables.

This is the first rigorous sense in which the parent is not imposed: it is the retained variable of the pushed-forward microscopic measure.

---

## 11. Local passive gauge covariance of the coarse map

The rigorous gauge-covariance statement is about passive gauge transformations, not arbitrary independent active changes of each child frame.

Let

$$
h:\mathcal C\to\mathcal G
$$

be a common base-local passive gauge transformation acting at the same base point on all agents:

$$
G_i(c)\mapsto h(c)G_i(c),
$$

$$
q_i(c)\mapsto h(c)_\#q_i(c),
\qquad
p_i(c)\mapsto h(c)_\#p_i(c),
$$

where $h_\#q$ denotes pushforward of the distribution. For Gaussian beliefs,

$$
\mu_i\mapsto h\mu_i,
\qquad
\Sigma_i\mapsto h\Sigma_i h^\top.
$$

Then the pure-gauge transport transforms as

$$
\Omega_{ij}(c)
\mapsto
h(c)\Omega_{ij}(c)h(c)^{-1}.
$$

The parent Karcher barycenter transforms equivariantly:

$$
G_B(c)\mapsto h(c)G_B(c),
$$

provided the distance $d_{\mathcal G}$ is left-invariant, as it is for compact groups with a bi-invariant metric.

Therefore

$$
\Omega_{Bi}(c)
=G_B(c)G_i(c)^{-1}
\mapsto
h(c)\Omega_{Bi}(c)h(c)^{-1}.
$$

Transported children transform by a common parent-frame pushforward:

$$
\widetilde q_i^B(c)
=\Omega_{Bi}(c)q_i(c)
\mapsto
h(c)_\#\widetilde q_i^B(c).
$$

Since the M-projection barycenter is equivariant under common pushforward,

$$
q_B(c)\mapsto h(c)_\#q_B(c).
$$

Thus

$$
\boxed{\mathcal R_B(h\cdot X)=h\cdot\mathcal R_B(X).}
$$

Independent child-specific transformations $g_i(c)$ are different: they change relative frames and therefore represent active changes of the epistemic configuration unless an additional rule is supplied to define the induced parent transformation. They are not passive gauge redundancies.

---

## 12. Gauge invariance of the renormalized free energy

### Proposition

Assume:

1. $\mathcal F_s(h\cdot X)=\mathcal F_s(X)$ for passive gauge fields $h(c)$;
2. the reference measure $\nu_s$ is invariant under the passive gauge action, or the theory has been gauge-fixed/regularized;
3. the coarse map is equivariant: $\mathcal R_s(h\cdot X)=h\cdot\mathcal R_s(X)$.

Then

$$
\boxed{\mathcal F_{s+1}(h\cdot Y)=\mathcal F_{s+1}(Y)}
$$

up to the same gauge-fixing or regularization convention.

### Proof sketch

Using the pushforward definition, the microscopic unnormalized measure

$$
\exp[-\mathcal F_s(X)/\tau]d\nu_s(X)
$$

is invariant under $h$. Since $\mathcal R_s$ is equivariant, its pushforward is also invariant under the induced action on $Y$. Therefore the Radon--Nikodym density defining $\mathcal F_{s+1}$ is invariant.

This avoids the coordinate delta-function step

$$
\delta(hY-h\mathcal R_s(X))=\delta(Y-\mathcal R_s(X)),
$$

which is only harmless for unimodular/volume-preserving actions or when the Jacobian has been included in the measure convention.

For $SO(K)$ this is clean. For $GL(K)$ the theorem requires a gauge slice, regulator, or explicit Jacobian/Radon--Nikodym correction.

---

## 13. Gaussian closure under the parent map

### Proposition

If all child beliefs are Gaussian and the parent belief is defined by

$$
q_B
=
\arg\min_q\sum_iw_iD_{\mathrm{KL}}(\widetilde q_i^B\Vert q),
$$

then $q_B$ is Gaussian with mean and covariance

$$
\mu_B=\sum_iw_i\widetilde\mu_i^B,
$$

$$
\Sigma_B
=
\sum_iw_i
\left[
\widetilde\Sigma_i^B
+(\widetilde\mu_i^B-\mu_B)(\widetilde\mu_i^B-\mu_B)^\top
\right].
$$

### Derivation

Let

$$
q=\mathcal N(\mu,\Sigma),
\qquad
\widetilde q_i^B=\mathcal N(\widetilde\mu_i,\widetilde\Sigma_i).
$$

The objective is

$$
J(q)=\sum_iw_iD_{\mathrm{KL}}(\widetilde q_i\Vert q),
\qquad
\sum_iw_i=1.
$$

Since

$$
D_{\mathrm{KL}}(\widetilde q_i\Vert q)
=-H(\widetilde q_i)-\mathbb E_{\widetilde q_i}[\log q(x)],
$$

and $H(\widetilde q_i)$ is independent of $q$, minimizing $J$ is equivalent to minimizing

$$
-\sum_iw_i\mathbb E_{\widetilde q_i}[\log q(x)].
$$

For a Gaussian $q$,

$$
-\log q(x)
=
\frac12\log\det\Sigma
+
\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)
+\text{const}.
$$

Therefore

$$
J(\mu,\Sigma)
=
\frac12\log\det\Sigma
+
\frac12\sum_iw_i
\mathbb E_{\widetilde q_i}
[(x-\mu)^\top\Sigma^{-1}(x-\mu)]
+\text{const}.
$$

The expectation is

$$
\mathbb E_{\widetilde q_i}[(x-\mu)(x-\mu)^\top]
=
\widetilde\Sigma_i
+(\widetilde\mu_i-\mu)(\widetilde\mu_i-\mu)^\top.
$$

The minimizing mean is the weighted mean

$$
\mu_B=\sum_iw_i\widetilde\mu_i.
$$

Substituting $\mu_B$, the minimizing covariance is the weighted second central moment

$$
\Sigma_B
=
\sum_iw_i
\left[
\widetilde\Sigma_i
+(\widetilde\mu_i-\mu_B)(\widetilde\mu_i-\mu_B)^\top
\right].
$$

Thus the Gaussian family is closed under the RG parent map.

---

## 14. Local expansion near a coherent block

Work in the parent frame and in Fisher-normal coordinates around $q_B$. Write the transported child beliefs as

$$
\widetilde q_i^B=q_B+\xi_i,
$$

with barycenter constraint

$$
\sum_iw_i\xi_i=0.
$$

For small disagreement,

$$
D_{\mathrm{KL}}(q_B+\xi_i\Vert q_B)
=
\frac12\|\xi_i\|_{F(q_B)}^2+O(\|\xi_i\|^3),
$$

where $F(q_B)$ is the Fisher information metric at $q_B$.

Similarly,

$$
D_{\mathrm{KL}}(q_B+\xi_i\Vert q_B+\xi_j)
=
\frac12\|\xi_i-\xi_j\|_{F(q_B)}^2+O(\|\xi\|^3).
$$

Assume an ordered-pair convention for $\sum_{ij}$, or adjust the Laplacian by the corresponding factor if unordered pairs are used. Then the intra-block alignment energy has the leading form

$$
\mathcal F_B^{\mathrm{int}}
=
\frac12\sum_{i,j\in B}\beta_{ij}\|\xi_i-\xi_j\|_{F(q_B)}^2+O(\|\xi\|^3).
$$

This can be written as

$$
\mathcal F_B^{\mathrm{int}}
=
\frac12\xi^\top H_B^\perp(q_B)\xi+O(\|\xi\|^3),
$$

where, to leading order,

$$
H_B^\perp(q_B)\approx L_B\otimes F(q_B).
$$

Here $L_B$ is the weighted graph Laplacian of the internal overlap/coupling graph.

If $B$ is connected and the internal couplings are positive, then the ordinary graph gap satisfies

$$
\lambda_2(L_B)>0.
$$

However, the barycenter constraint is weighted:

$$
\sum_iw_i\xi_i=0.
$$

Therefore the correct stiffness constant is not generally $\lambda_2(L_B)$. It is the constrained weighted gap

$$
\boxed{
\lambda_{B,w}
=
\inf_{\xi\neq0,\;\sum_iw_i\xi_i=0}
\frac{\xi^\top L_B\xi}{\xi^\top\xi}.
}
$$

The internal stiffness is

$$
\boxed{
m_B
=\lambda_{B,w}\lambda_{\min}(F(q_B)).
}
$$

Only for uniform weights, or for compatible weighted inner products, does this reduce directly to $\lambda_2(L_B)\lambda_{\min}(F)$.

Thus internal disagreement modes are massive when

$$
m_B>0.
$$

The barycentric parent variable is the surviving collective coordinate.

---

## 15. Laplace approximation to the exact RG integral

Near a coherent block, decompose the microscopic variables into the retained parent variable $Y_B$ and internal disagreement variables $\xi$:

$$
X_B\leftrightarrow(Y_B,\xi).
$$

Assume the free energy expands as

$$
\mathcal F_s(X_B)
=
\mathcal F_{\mathrm{eff}}(Y_B)
+\frac12\xi^\top H_B^\perp(Y_B)\xi
+\frac{1}{3!}T_3(Y_B)[\xi^3]
+\frac{1}{4!}T_4(Y_B)[\xi^4]
+\cdots.
$$

The exact RG integral over internal modes is

$$
e^{-\mathcal F_{s+1}(Y_B)/\tau}
=
\int\exp\left[-\frac{1}{\tau}\mathcal F_s(Y_B,\xi)\right]d\xi.
$$

Keeping the Gaussian term gives

$$
e^{-\mathcal F_{s+1}(Y_B)/\tau}
\approx
\exp\left[-\frac{1}{\tau}\mathcal F_{\mathrm{eff}}(Y_B)\right]
(2\pi\tau)^{d_\perp/2}
(\det{}'H_B^\perp(Y_B))^{-1/2}.
$$

Taking $-\tau\log$ gives

$$
\boxed{
\mathcal F_{s+1}(Y_B)
=
\mathcal F_{\mathrm{eff}}(Y_B)
+\frac{\tau}{2}\log\det{}'H_B^\perp(Y_B)
+\text{const}
+\mathcal E_{\mathrm{anh}}.
}
$$

The prime on $\det{}'$ means that barycentric zero modes are excluded; only internal disagreement modes are integrated out.

The leading anharmonic corrections after taking $-\tau\log$ have the standard Watson/Laplace scaling

$$
\boxed{
\mathcal E_{\mathrm{anh}}
=
O\!\left(
\tau^2\|H^{-1}\|^3\|T_3\|^2
+
\tau^2\|H^{-1}\|^2\|T_4\|
\right),
}
$$

with higher-order terms controlled by higher derivatives and powers of $\tau H^{-1}$. In a linear-Gaussian/quadratic regime, the Laplace formula is exact.

---

## 16. Renormalized inter-block couplings

Let $B$ and $C$ be two emergent blocks. Suppose their internal dispersions are small:

$$
V_B=\sum_{i\in B}w_i^BD_{\mathrm{KL}}(\widetilde q_i^B\Vert q_B)\ll1,
$$

$$
V_C=\sum_{j\in C}w_j^CD_{\mathrm{KL}}(\widetilde q_j^C\Vert q_C)\ll1.
$$

The microscopic cross-block energy is

$$
\mathcal F_{BC}^{\mathrm{micro}}
=
\sum_{i\in B}\sum_{j\in C}
\beta_{ij}
D_{\mathrm{KL}}\!\left(q_i\middle\Vert\Omega_{ij}q_j\right).
$$

Transporting to block frames and expanding around $q_B,q_C$ gives

$$
D_{\mathrm{KL}}\!\left(q_i\middle\Vert\Omega_{ij}q_j\right)
=
D_{\mathrm{KL}}\!\left(q_B\middle\Vert\Omega_{BC}q_C\right)
+\ell_i^B(\xi_i)+\ell_j^C(\eta_j)
+O(\|\xi\|^2+\|\eta\|^2+\mathcal H_{BC}),
$$

where $\eta_j$ denotes the internal fluctuation of child $j\in C$, and $\mathcal H_{BC}$ measures transport inconsistency, frame variation, or holonomy variation across the block.

The zeroth-order term is independent of $i,j$. Therefore, under the microscopic convention in Section 2,

$$
\boxed{
\beta_{BC}^R
=
\sum_{i\in B}\sum_{j\in C}\beta_{ij}.
}
$$

Thus

$$
\boxed{
\mathcal F_{BC}^{\mathrm{micro}}
=
\beta_{BC}^R
D_{\mathrm{KL}}\!\left(q_B\middle\Vert\Omega_{BC}q_C\right)
+\mathcal R_{BC}^{(1)}
+O(V_B+V_C+\mathcal H_{BC}).
}
$$

The first-order residual $\mathcal R_{BC}^{(1)}$ vanishes only under a compatibility condition between barycenter weights and edge marginals. Define

$$
r_i=\sum_{j\in C}\beta_{ij},
\qquad
s_j=\sum_{i\in B}\beta_{ij},
\qquad
\beta_{BC}^R=\sum_{ij}\beta_{ij}.
$$

The first-order terms vanish if, for the $B$ and $C$ barycenters relevant to this interaction,

$$
\boxed{
w_i^B=\frac{r_i}{\beta_{BC}^R},
\qquad
w_j^C=\frac{s_j}{\beta_{BC}^R}.
}
$$

If different barycenter weights are used, the first-order mismatch must be included in the closure residual. A useful diagnostic is

$$
\mathcal R_{BC}^{(1)}
\sim
\left\|\sum_{i\in B}r_i\xi_i\right\|_{F}
+
\left\|\sum_{j\in C}s_j\eta_j\right\|_{F}.
$$

The weighted formula

$$
\sum_{ij}w_i^Bw_j^C\beta_{ij}
$$

is correct only for a different microscopic functional in which the cross-block energy itself contains the factor $w_i^Bw_j^C$.

Therefore the renormalized parent free energy has the same structural form as the child free energy, up to determinant terms and controlled residuals:

$$
\boxed{
\mathcal F_{s+1}^{\mathrm{agent}}
=
\sum_B\int_{U_B}D_{\mathrm{KL}}(q_B\Vert p_B^R)dc
+
\sum_{B,C}\int_{U_B\cap U_C}
\beta_{BC}^R
D_{\mathrm{KL}}\!\left(q_B\middle\Vert\Omega_{BC}q_C\right)dc
+
\text{model/gauge terms}
+
\text{determinant corrections}
+
\text{closure residuals}.
}
$$

---

## 17. Closure residual

Define the exact RG free energy from the pushforward construction:

$$
\mathcal F_{s+1}^{\mathrm{exact}}.
$$

Define the projected agent-form free energy:

$$
\mathcal F_{s+1}^{\mathrm{agent}}.
$$

The closure residual is

$$
\boxed{
\varepsilon_s
=
\left\|\mathcal F_{s+1}^{\mathrm{exact}}
-\mathcal F_{s+1}^{\mathrm{agent}}\right\|.
}
$$

This residual includes, at minimum:

1. internal dispersion errors $V_B,V_C$;
2. transport/holonomy variation errors $\mathcal H_{BC}$;
3. first-order edge-marginal/barycenter-weight mismatch terms;
4. Laplace anharmonic corrections;
5. non-Gaussian closure errors if the Gaussian family is only an approximation;
6. gauge-fixing or measure-regularization corrections for noncompact groups.

A block family is RG-closed when

$$
\varepsilon_s\ll1.
$$

In practice, $\varepsilon_s$ can be estimated by comparing microscopic and parent-level predictions for held-out coarse observables, free-energy changes, or reconstructed child statistics.

---

## 18. Two distinct selection principles: RG relevance and MDL/Bayesian retention

Exact RG always defines a coarse free energy. However, two different questions must be separated.

### 18.1 Wilsonian relevance

Let $\theta_s$ denote the couplings/parameters of the scale-$s$ free-energy functional and let

$$
\theta_{s+1}=R(\theta_s)
$$

be the induced RG flow. Near a fixed point $\theta^*$,

$$
\theta^*=R(\theta^*),
$$

linearize:

$$
\delta\theta_{s+1}=DR_{\theta^*}\delta\theta_s.
$$

If

$$
DR_{\theta^*}v_a=\lambda_av_a,
$$

then:

- $|\lambda_a|>1$: relevant direction;
- $|\lambda_a|<1$: irrelevant direction;
- $|\lambda_a|=1$: marginal direction.

A parent variable is Wilsonian-relevant when its associated coarse coordinate or coupling lies in a relevant or marginal eigendirection of the RG flow.

### 18.2 MDL/Bayesian retention

Finite simulations also need a representation-selection rule: should this parent be explicitly stored as an agent?

Define the microscopic description cost for block $B$:

$$
\mathcal L_{\mathrm{micro}}(B)
=\mathcal F_s^\star(B)+C_{\mathrm{micro}}(B),
$$

where $\mathcal F_s^\star(B)$ is the optimized microscopic free energy over the block and $C_{\mathrm{micro}}(B)$ is the cost of representing children explicitly.

Define the parent description cost:

$$
\mathcal L_{\mathrm{parent}}(B)
=\mathcal F_{s+1}^\star(B)+C(B)+\varepsilon_B.
$$

The emergence/retention gain is

$$
\boxed{
\Delta_B
=\mathcal L_{\mathrm{micro}}(B)-\mathcal L_{\mathrm{parent}}(B).
}
$$

A finite representation retains the parent when

$$
\boxed{\Delta_B>0.}
$$

This is a model-selection or description-length criterion, not automatically the same as Wilsonian relevance.

To make this criterion fully specified, one must define $C(B)$. Possible choices include:

1. **Bayesian evidence prior**:

   $$
   C(B)=-\tau\log P(B).
   $$

2. **BIC/MDL penalty**:

   $$
   C(B)=\frac{\tau}{2}k_B\log n_B,
   $$

   where $k_B$ is the number of parent parameters and $n_B$ is the effective support/sample size.

3. **Variational complexity**:

   $$
   C(B)=D_{\mathrm{KL}}(Q_B\Vert P_B),
   $$

   where $Q_B$ is the posterior over parent parameters and $P_B$ is a structural prior.

Until such a penalty is fixed, $\Delta_B>0$ is a schema for retention rather than a complete theorem.

---

## 19. Relation to practical thresholds

The existing threshold rule can be reinterpreted as a fast candidate-selection approximation to the RG and retention criteria.

A block $B$ should be considered a candidate when:

### 1. Multi-child overlap exists

$$
U_B\neq\varnothing.
$$

### 2. The overlap graph is connected

$$
\lambda_2(L_B)>0.
$$

### 3. Weighted internal stiffness is positive

$$
\lambda_{B,w}>0,
\qquad
m_B=\lambda_{B,w}\lambda_{\min}(F(q_B))>0.
$$

### 4. Barycentric dispersion is small

$$
V_B=\sum_iw_i^BD_{\mathrm{KL}}(\widetilde q_i^B\Vert q_B)<\theta_V.
$$

### 5. Internal alignment is strong

For example,

$$
A_B=\exp(-V_B/\tau_A)>\theta_A.
$$

### 6. Closure residual is small

$$
\varepsilon_B<\theta_\varepsilon.
$$

### 7. Representation-retention gain is positive

$$
\Delta_B>0,
$$

provided the complexity penalty $C(B)$ has been specified.

Thus thresholds become computational surrogates for candidate detection, not the fundamental definition of emergence.

---

## 20. Autonomy from adiabatic elimination

Let $Y_B$ denote parent variables and $\xi_B$ internal child-disagreement variables. Suppose the block free energy locally decomposes as

$$
\mathcal F_s(Y_B,\xi_B)
=
\mathcal F_{s+1}(Y_B)
+\frac12\xi_B^\top H_B^\perp(Y_B)\xi_B
+O(\|\xi_B\|^3).
$$

Assume the internal Hessian has spectral gap

$$
H_B^\perp\succeq m_BI.
$$

The microscopic natural-gradient flow has the form

$$
\dot z=-G(z)^{-1}\nabla\mathcal F_s(z),
\qquad
z=(Y_B,\xi_B).
$$

In block coordinates the metric generally has cross terms:

$$
G=
\begin{pmatrix}
G_{YY} & G_{Y\xi}\\
G_{\xi Y} & G_{\xi\xi}
\end{pmatrix}.
$$

Therefore the naive split into independent $Y$ and $\xi$ flows is valid only if $G_{Y\xi}=0$ or is negligible. In general, the effective slow metric is the Schur-complement metric

$$
\boxed{
G_{\mathrm{eff}}
=
G_{YY}-G_{Y\xi}G_{\xi\xi}^{-1}G_{\xi Y}.
}
$$

Under standard slow-fast assumptions, internal disagreement modes relax on timescale

$$
t_{\mathrm{int}}\sim m_B^{-1}.
$$

When parent variables evolve more slowly, adiabatic elimination gives

$$
\xi_B(t)=O(m_B^{-1})+O(\varepsilon_B).
$$

Substituting into the slow equation yields

$$
\boxed{
\dot Y_B
=-\operatorname{grad}_{G_{\mathrm{eff}}}\mathcal F_{s+1}(Y_B)
+O(m_B^{-1})
+O(\varepsilon_B).
}
$$

This is the autonomy result:

> Once internal disagreement modes are gapped and the metric reduction is controlled, the parent evolves approximately according to a closed renormalized free-energy flow.

---

## 21. Corrected RG emergence theorem

### Theorem: RG emergence of gauge-covariant meta-agents

Let $B\subset I_s$ be an admissible connected overlap block. Assume:

1. child beliefs and model fields lie in a Gaussian statistical manifold;
2. the gauge group is compact/unimodular, such as $SO(K)$, or noncompact gauge degrees have been gauge-fixed or regularized;
3. passive gauge transformations are common base-local maps $h(c)$;
4. the child frames lie in a geodesically convex neighborhood, so the parent Karcher frame is unique;
5. the transported parent belief is the M-projection barycenter;
6. transported child beliefs have finite barycentric dispersion;
7. the internal coupling graph of $B$ has positive constrained weighted gap

   $$
   \lambda_{B,w}>0;
   $$

8. the exact RG free energy is approximated by the agent-form free energy with residual $\varepsilon_B$;
9. natural-gradient dynamics either block-diagonalize in $(Y_B,\xi_B)$ or use the Schur-complement effective metric;
10. if the parent is to be explicitly retained in a finite representation, a Bayesian/MDL/variational complexity penalty $C(B)$ is specified.

Then the coarse variable

$$
Y_B=\mathcal R_B(X_B)
$$

is an emergent meta-agent in the following controlled senses.

### 1. Gauge covariance

$$
\mathcal R_B(h\cdot X_B)=h\cdot\mathcal R_B(X_B),
$$

and the renormalized free energy satisfies

$$
\mathcal F_{s+1}(h\cdot Y_B)=\mathcal F_{s+1}(Y_B)
$$

under the same compactness/gauge-fixing/measure assumptions.

### 2. Gaussian closure

If the children are Gaussian, then the parent is Gaussian, with exact moment-matched mean and covariance, including the between-child dispersion term.

### 3. Free-energy closure

The exact renormalized free energy closes on the same agent-form functional up to residual error:

$$
\mathcal F_{s+1}^{\mathrm{exact}}(Y_B)
=
\mathcal F_{s+1}^{\mathrm{agent}}(Y_B)+O(\varepsilon_B).
$$

### 4. Stability

Internal disagreement modes satisfying the weighted barycenter constraint are gapped:

$$
\xi^\top H_B^\perp\xi\ge m_B\|\xi\|^2,
$$

with

$$
m_B=\lambda_{B,w}\lambda_{\min}(F(q_B)).
$$

Therefore microscopic perturbations that do not alter the barycenter decay at rate controlled by $m_B$.

### 5. Autonomy

The parent obeys a closed renormalized free-energy flow up to controlled error:

$$
\dot Y_B
=-\operatorname{grad}_{G_{\mathrm{eff}}}\mathcal F_{s+1}(Y_B)
+O(m_B^{-1})+O(\varepsilon_B).
$$

### 6. Retention in a finite representation

If a complexity penalty $C(B)$ has been specified, the parent is explicitly retained when

$$
\Delta_B>0.
$$

This is a finite-representation model-selection rule. It is distinct from Wilsonian relevance, though the two may agree in favorable regimes.

Thus the parent is not a heuristic cluster. It is a stable, gauge-covariant, approximately autonomous coarse degree of freedom whose explicit representation is justified when a specified retention criterion is met.

---

## 22. Interpretation

The RG construction converts meta-agent emergence from an algorithmic rule into a controlled variational coarse-graining statement.

The old heuristic rule was:

$$
\text{if overlap/coherence exceeds threshold, create parent.}
$$

The corrected RG interpretation is:

$$
\text{construct parent variables by a gauge-covariant coarse map, integrate out internal modes, and retain explicit parents only when closure, stability, and the chosen selection rule justify retention.}
$$

More explicitly:

1. Children interact over overlaps.
2. Their transported KL couplings define local collective modes.
3. The RG map integrates out internal child-disagreement modes.
4. Connected, coherent blocks have a constrained weighted spectral gap.
5. Their transported M-projection barycenter survives as a stable collective coordinate.
6. The exact parent free energy is the pushforward of the microscopic variational measure.
7. The parent-level agent-form free energy is valid when closure residuals are controlled.
8. The parent evolves autonomously when internal modes are gapped and metric reduction is controlled.
9. The parent is explicitly retained in a simulation when a specified MDL/Bayesian/variational criterion is positive, or studied as Wilsonian-relevant when it corresponds to a relevant/marginal RG direction.

In one sentence:

> A meta-agent is a gauge-covariant renormalized sufficient statistic of a coherent overlap block; it is dynamically meaningful when internal disagreement modes are irrelevant and representationally retained when a specified selection criterion favors it.

---

## 23. Computable diagnostics for simulation

The theory suggests the following diagnostics for the existing codebase.

### 1. Parent support validity

Check

$$
U_B\subseteq\left\{c:\sum_i\chi_i(c)\ge2\right\}.
$$

No parent mask should extend into single-child regions.

### 2. Overlap graph connectivity

Compute the internal graph Laplacian $L_B$ and verify

$$
\lambda_2(L_B)>0.
$$

This is a connectivity diagnostic, not always the exact stability constant.

### 3. Weighted constrained spectral gap

Compute

$$
\lambda_{B,w}
=
\inf_{\xi\neq0,\;\sum_iw_i\xi_i=0}
\frac{\xi^\top L_B\xi}{\xi^\top\xi}.
$$

Then compute

$$
m_B=\lambda_{B,w}\lambda_{\min}(F(q_B)).
$$

This is the relevant local stiffness diagnostic.

### 4. Transported barycentric dispersion

Compute

$$
V_B=\sum_iw_i^BD_{\mathrm{KL}}(\widetilde q_i^B\Vert q_B).
$$

Small $V_B$ means high internal coherence.

### 5. Exact Gaussian parent covariance

Verify the parent covariance includes the dispersion term:

$$
\Sigma_B
=
\sum_iw_i
\left[
\widetilde\Sigma_i
+(\widetilde\mu_i-\mu_B)(\widetilde\mu_i-\mu_B)^\top
\right].
$$

If this term is omitted, label the implementation as a high-coherence approximation.

### 6. Inter-block renormalized coupling

Under the Section 2 microscopic functional, compute

$$
\beta_{BC}^R=\sum_{i\in B}\sum_{j\in C}\beta_{ij}.
$$

Do not multiply by $w_i^Bw_j^C$ unless the microscopic cross-block energy explicitly includes those weights.

### 7. Edge-marginal weight compatibility

For a cross-block interaction, compute

$$
r_i=\sum_{j\in C}\beta_{ij},
\qquad
s_j=\sum_{i\in B}\beta_{ij}.
$$

Check whether

$$
w_i^B\approx\frac{r_i}{\beta_{BC}^R},
\qquad
w_j^C\approx\frac{s_j}{\beta_{BC}^R}.
$$

If not, include the first-order mismatch in the closure residual.

### 8. Closure residual

Estimate

$$
\varepsilon_B
=\|\mathcal F_{s+1}^{\mathrm{exact}}-\mathcal F_{s+1}^{\mathrm{agent}}\|.
$$

Practically, compare microscopic and parent-level predictions for held-out coarse observables.

### 9. Retention gain

After specifying a complexity penalty $C(B)$, compute

$$
\Delta_B
=\mathcal L_{\mathrm{micro}}(B)-\mathcal L_{\mathrm{parent}}(B).
$$

A parent should be explicitly retained only if

$$
\Delta_B>0.
$$

### 10. Autonomy error

Compare the actual parent trajectory from the microscopic simulation to the renormalized parent flow:

$$
E_{\mathrm{auto}}
=
\left\|\dot Y_B+\operatorname{grad}_{G_{\mathrm{eff}}}\mathcal F_{s+1}(Y_B)\right\|.
$$

Autonomy means

$$
E_{\mathrm{auto}}=O(m_B^{-1})+O(\varepsilon_B).
$$

---

## 24. Limitations

This RG result supports a precise emergence claim:

$$
\boxed{
\text{Gauge-covariant interacting agents admit stable, autonomous, approximately closed coarse variables that are themselves agents.}
}
$$

It does not by itself prove:

1. Lorentzian spacetime;
2. physical constants;
3. quantum measurement;
4. consciousness;
5. transformer attention;
6. nontrivial curvature from pure-gauge frame mismatch;
7. rigorous noncompact $GL(K)$ gauge averaging without gauge fixing or regularization.

Those require additional structures.

---

## 25. Manuscript-ready summary paragraph

The hierarchy construction can be reformulated as an exact finite-dimensional renormalization procedure. Given a microscopic ensemble of local gauge-covariant agents, admissible blocks are defined by connected components of the overlap hypergraph, and each block is mapped to a parent variable by a gauge-covariant transported M-projection barycenter. The renormalized free energy is the pushforward of the microscopic variational measure under this coarse map, which preserves the partition function and all observables depending only on the retained variables. In the Gaussian case, the parent remains Gaussian with moment-matched mean and covariance, including the between-child dispersion term. Near a coherent block, internal disagreement modes have a Hessian proportional to the overlap-graph Laplacian tensored with the Fisher metric; with nonuniform barycenter weights, the relevant stiffness is the constrained weighted spectral gap rather than the ordinary graph gap. Integrating these modes out yields a parent-level free energy of agent form, corrected by a determinant term, anharmonic Laplace corrections, edge-marginal closure residuals, and transport/holonomy residuals. The inter-block coupling renormalizes as the unweighted sum of microscopic couplings under the stated microscopic functional. A meta-agent is therefore not created by thresholding: thresholds are computational detectors for blocks with valid multi-child support, small barycentric dispersion, positive weighted spectral gap, controlled closure residual, and, once a complexity penalty has been specified, positive retention gain.
