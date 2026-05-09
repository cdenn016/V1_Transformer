# Nontrivial Gauge Transport Derivation of Transformer-Style Attention

This note derives the attention weights directly from nontrivial relative gauge transports

$$
\Omega_{ij} \neq I,
$$

rather than setting $\Omega_{ij}=I$ and inserting a separate bilinear matrix $M$ by hand.

The main conclusion is:

$$
\boxed{
\Omega_{ij}=U_iU_j^{-1}
\quad \Longrightarrow \quad
\text{query-key compatibility arises from relative gauge transport.}
}
$$

With arbitrary SPD covariances $\Sigma_i$ (no frame-covariant closure imposed), the cross term in the KL-consensus logit factors as $Q_i^\top K_j$ with the untied projections

$$
\boxed{
Q_i=U_i^{-1}\mu_i,
\qquad
K_j=U_j^\top\Sigma_j^{-1}\mu_j,
\qquad
V_j=U_j^{-1}\nu_j.
}
$$

The implicit pair-dependent bilinear $\Omega_{ij}^{-\top}\Sigma_j^{-1}=U_i^{-\top}U_j^\top\Sigma_j^{-1}$ ranges over all of $\mathrm{GL}(K)$ as the per-token data $(U_i,U_j,\Sigma_j)$ varies, matching the expressive power of an independently learned $W_QW_K^\top$ in standard scaled dot-product attention. The carving uses only the existing belief and frame fields; no external learned bilinear $M$ and no extra query/key morphisms are required.

Under the additional closure $\Sigma_j=U_jCU_j^\top$ for shared SPD $C$, the carving collapses to the symmetric tied form

$$
Q_i=C^{-1/2}U_i^{-1}\mu_i,
\qquad
K_j=C^{-1/2}U_j^{-1}\mu_j,
\qquad
V_j=U_j^{-1}\nu_j,
$$

recovering scaled dot-product attention with a single shared SPD bilinear $C^{-1}$ as a corollary.

---

## 1. Conventions

Let token/agent $i$ carry a Gaussian belief in its own local fiber:

$$
q_i=\mathcal N(\mu_i,\Sigma_i),
$$

and let token/agent $j$ carry

$$
q_j=\mathcal N(\mu_j,\Sigma_j).
$$

The transport from agent $j$'s frame to agent $i$'s frame is

$$
\Omega_{ij}:E_j\to E_i.
$$

We use the frame convention

$$
\boxed{
\Omega_{ij}=U_iU_j^{-1}.
}
$$

Here $U_i$ maps a common latent coordinate system into agent $i$'s local frame. Thus if $x_i$ denotes the common-frame coordinate associated with $\mu_i$, then

$$
\boxed{
\mu_i=U_ix_i,
\qquad
x_i=U_i^{-1}\mu_i.
}
$$

Similarly,

$$
\mu_j=U_jx_j,
\qquad
x_j=U_j^{-1}\mu_j.
$$

Transporting $\mu_j$ into agent $i$'s frame gives

$$
\Omega_{ij}\mu_j
=
U_iU_j^{-1}\mu_j
=
U_ix_j.
$$

So the relative transport compares $\mu_i=U_ix_i$ with $\Omega_{ij}\mu_j=U_ix_j$ inside agent $i$'s frame.

---

## 2. Gauge-transported Gaussian belief

If

$$
q_j=\mathcal N(\mu_j,\Sigma_j),
$$

then the push-forward of $q_j$ by $\Omega_{ij}$ is

$$
\boxed{
\Omega_{ij\#}q_j
=
\mathcal N\left(\Omega_{ij}\mu_j,\Omega_{ij}\Sigma_j\Omega_{ij}^\top\right).
}
$$

Therefore the gauge-covariant discrepancy from $j$ to $i$ is

$$
\boxed{
D_{ij}
=
D_{\mathrm{KL}}\left(q_i\,\Vert\,\Omega_{ij\#}q_j\right).
}
$$

Substituting the Gaussian forms gives

$$
D_{ij}
=
D_{\mathrm{KL}}\left(
\mathcal N(\mu_i,\Sigma_i)
\,\Vert\,
\mathcal N(\Omega_{ij}\mu_j,\Omega_{ij}\Sigma_j\Omega_{ij}^\top)
\right).
$$

---

## 3. General Gaussian KL formula

For two $K$-dimensional Gaussians

$$
q_0=\mathcal N(m_0,S_0),
\qquad
q_1=\mathcal N(m_1,S_1),
$$

the forward KL divergence is

$$
D_{\mathrm{KL}}(q_0\Vert q_1)
=
\frac12
\left[
\log\frac{\det S_1}{\det S_0}
-K
+\operatorname{tr}(S_1^{-1}S_0)
+(m_0-m_1)^\top S_1^{-1}(m_0-m_1)
\right].
$$

For the transported Gaussian, set

$$
m_0=\mu_i,
\qquad
S_0=\Sigma_i,
$$

and

$$
m_1=\Omega_{ij}\mu_j,
\qquad
S_1=\Omega_{ij}\Sigma_j\Omega_{ij}^\top.
$$

Then

$$
\boxed{
\begin{aligned}
D_{ij}
=
\frac12
\Big[
&\log\frac{\det(\Omega_{ij}\Sigma_j\Omega_{ij}^\top)}{\det\Sigma_i}
-K \\
&+\operatorname{tr}\left((\Omega_{ij}\Sigma_j\Omega_{ij}^\top)^{-1}\Sigma_i\right) \\
&+(\mu_i-\Omega_{ij}\mu_j)^\top
(\Omega_{ij}\Sigma_j\Omega_{ij}^\top)^{-1}
(\mu_i-\Omega_{ij}\mu_j)
\Big].
\end{aligned}
}
$$

This is the exact nontrivial-$\Omega_{ij}$ Gaussian KL.

---

## 4. Rewrite using precision matrices

Let

$$
\Lambda_j=\Sigma_j^{-1}.
$$

Then

$$
(\Omega_{ij}\Sigma_j\Omega_{ij}^\top)^{-1}
=
\Omega_{ij}^{-\top}\Lambda_j\Omega_{ij}^{-1}.
$$

Also,

$$
\det(\Omega_{ij}\Sigma_j\Omega_{ij}^\top)
=
(\det\Omega_{ij})^2\det\Sigma_j.
$$

Since the covariance determinant is positive, the logarithm can be written as

$$
\log\det(\Omega_{ij}\Sigma_j\Omega_{ij}^\top)
=
\log\det\Sigma_j+2\log|\det\Omega_{ij}|.
$$

The quadratic term can be simplified as

$$
\begin{aligned}
&(\mu_i-\Omega_{ij}\mu_j)^\top
\Omega_{ij}^{-\top}\Lambda_j\Omega_{ij}^{-1}
(\mu_i-\Omega_{ij}\mu_j)
\\
&\qquad =
(\Omega_{ij}^{-1}\mu_i-\mu_j)^\top
\Lambda_j
(\Omega_{ij}^{-1}\mu_i-\mu_j).
\end{aligned}
$$

Therefore

$$
\boxed{
\begin{aligned}
D_{ij}
=
\frac12
\Big[
&\log\det\Sigma_j-
\log\det\Sigma_i
+2\log|\det\Omega_{ij}|
-K \\
&+\operatorname{tr}\left(\Omega_{ij}^{-\top}\Lambda_j\Omega_{ij}^{-1}\Sigma_i\right) \\
&+(\Omega_{ij}^{-1}\mu_i-\mu_j)^\top
\Lambda_j
(\Omega_{ij}^{-1}\mu_i-\mu_j)
\Big].
\end{aligned}
}
$$

---

## 5. Insert the relative gauge transport

Now substitute

$$
\Omega_{ij}=U_iU_j^{-1}.
$$

Then

$$
\Omega_{ij}^{-1}=U_jU_i^{-1},
$$

and

$$
\Omega_{ij}^{-\top}=U_i^{-\top}U_j^\top.
$$

Define common-frame variables for agent $i$:

$$
\boxed{
x_i=U_i^{-1}\mu_i,
\qquad
P_i=U_i^{-1}\Sigma_iU_i^{-\top}.
}
$$

Define key-side natural parameters for agent $j$:

$$
\boxed{
H_j=U_j^\top\Lambda_jU_j,
}
$$

$$
\boxed{
k_j=U_j^\top\Lambda_j\mu_j,
}
$$

and

$$
\boxed{
r_j=\mu_j^\top\Lambda_j\mu_j.
}
$$

Then the transported inverse-mean term is

$$
\Omega_{ij}^{-1}\mu_i
=
U_jU_i^{-1}\mu_i
=
U_jx_i.
$$

So the quadratic term becomes

$$
\begin{aligned}
&(\Omega_{ij}^{-1}\mu_i-\mu_j)^\top
\Lambda_j
(\Omega_{ij}^{-1}\mu_i-\mu_j)
\\
&\qquad =
(U_jx_i-\mu_j)^\top\Lambda_j(U_jx_i-\mu_j)
\\
&\qquad =
x_i^\top U_j^\top\Lambda_jU_jx_i
-2x_i^\top U_j^\top\Lambda_j\mu_j
+\mu_j^\top\Lambda_j\mu_j
\\
&\qquad =
x_i^\top H_jx_i-2x_i^\top k_j+r_j.
\end{aligned}
$$

The trace term becomes

$$
\begin{aligned}
\operatorname{tr}\left(\Omega_{ij}^{-\top}\Lambda_j\Omega_{ij}^{-1}\Sigma_i\right)
&=
\operatorname{tr}\left(U_i^{-\top}U_j^\top\Lambda_jU_jU_i^{-1}\Sigma_i\right)
\\
&=
\operatorname{tr}\left(U_j^\top\Lambda_jU_jU_i^{-1}\Sigma_iU_i^{-\top}\right)
\\
&=
\operatorname{tr}(H_jP_i).
\end{aligned}
$$

The determinant term is

$$
2\log|\det\Omega_{ij}|
=
2\log|\det U_i|-2\log|\det U_j|.
$$

Therefore the full transported KL can be written as

$$
\boxed{
\begin{aligned}
D_{ij}
=
\frac12
\Big[
&\log\det\Sigma_j-
\log\det\Sigma_i
+2\log|\det U_i|-
2\log|\det U_j|
-K \\
&+\operatorname{tr}(H_jP_i)
+x_i^\top H_jx_i
-2x_i^\top k_j
+r_j
\Big].
\end{aligned}
}
$$

This formula is exact.

---

## 6. Attention weights from entropy-regularized KL consensus

The entropy-regularized source-selection problem gives weights of the form

$$
\boxed{
\beta_{ij}
=
\frac{\pi_{ij}\exp(-D_{ij}/\tau)}
{\sum_k\pi_{ik}\exp(-D_{ik}/\tau)}.
}
$$

Equivalently,

$$
\boxed{
\beta_{ij}=\operatorname{softmax}_j\left(\log\pi_{ij}-D_{ij}/\tau\right).
}
$$

If $\pi_{ij}$ is uniform, this reduces to

$$
\boxed{
\beta_{ij}=\operatorname{softmax}_j\left(-D_{ij}/\tau\right).
}
$$

For fixed query $i$, any term in $D_{ij}$ that does not depend on $j$ cancels inside the row-wise softmax.

From the exact expression above, the row-constant terms include

$$
-\log\det\Sigma_i+2\log|\det U_i|-K.
$$

After dropping row-constant terms, the effective nontrivial-gauge logit is

$$
\boxed{
\ell_{ij}
=
\log\pi_{ij}
+
\frac{1}{\tau}x_i^\top k_j
-
\frac{1}{2\tau}x_i^\top H_jx_i
-
\frac{1}{2\tau}\operatorname{tr}(H_jP_i)
-
\frac{1}{2\tau}r_j
-
\frac{1}{2\tau}\log\det\Sigma_j
+
\frac{1}{\tau}\log|\det U_j|.
}
$$

Thus

$$
\boxed{
\beta_{ij}=\operatorname{softmax}_j(\ell_{ij}).
}
$$

The leading compatibility term is

$$
\boxed{
x_i^\top k_j
=
(U_i^{-1}\mu_i)^\top(U_j^\top\Sigma_j^{-1}\mu_j).
}
$$

This is the nontrivial-gauge source of query-key compatibility.

---

## 7. Interpretation of the exact logit

The exact logit contains more than a dot product:

$$
\ell_{ij}
=
\text{compatibility}
-
\text{query distortion}
-
\text{covariance distortion}
-
\text{key precision norm}
-
\text{volume correction}
+
\text{prior bias}.
$$

Term by term:

### 7.1 Compatibility term

$$
\frac{1}{\tau}x_i^\top k_j.
$$

This is the analogue of $Q_i^\top K_j/\sqrt{d_k}$.

### 7.2 Query distortion term

$$
-\frac{1}{2\tau}x_i^\top H_jx_i.
$$

This penalizes query $i$ when viewed through key $j$'s precision geometry.

### 7.3 Covariance distortion term

$$
-\frac{1}{2\tau}\operatorname{tr}(H_jP_i).
$$

This penalizes mismatch between query uncertainty and key-side precision.

### 7.4 Key precision norm

$$
-\frac{1}{2\tau}r_j
=
-\frac{1}{2\tau}\mu_j^\top\Sigma_j^{-1}\mu_j.
$$

This is the Mahalanobis key-norm correction.

### 7.5 Volume correction

$$
-\frac{1}{2\tau}\log\det\Sigma_j+
\frac{1}{\tau}\log|\det U_j|.
$$

This accounts for covariance volume and frame-volume effects.

Hence the fully general nontrivial-$\Omega_{ij}$ theory gives a richer object than vanilla dot-product attention. Vanilla attention is a special case.

---

## 8. Clean frame-covariant specialization

The cleanest way to recover standard dot-product attention is not to impose $\Sigma_i=\sigma^2I$ in every local frame for arbitrary $U_i\in\mathrm{GL}(K)$. That choice is not generally gauge-covariant when the frames are non-orthogonal.

Instead, assume there is a shared common-frame covariance

$$
C\succ 0
$$

and each local covariance is obtained by pushing $C$ forward by $U_i$:

$$
\boxed{
\Sigma_i=U_iCU_i^\top.
}
$$

Similarly,

$$
\boxed{
\Sigma_j=U_jCU_j^\top.
}
$$

Then

$$
P_i=U_i^{-1}\Sigma_iU_i^{-\top}=C.
$$

Also,

$$
\Lambda_j=\Sigma_j^{-1}=U_j^{-\top}C^{-1}U_j^{-1}.
$$

Therefore

$$
H_j=U_j^\top\Lambda_jU_j=C^{-1},
$$

and

$$
k_j=U_j^\top\Lambda_j\mu_j=C^{-1}U_j^{-1}\mu_j=C^{-1}x_j.
$$

Finally,

$$
r_j=\mu_j^\top\Lambda_j\mu_j=x_j^\top C^{-1}x_j.
$$

Under this assumption, transporting $q_j$ into frame $i$ gives

$$
\Omega_{ij\#}q_j
=
\mathcal N(U_ix_j,U_iCU_i^\top).
$$

But

$$
q_i=\mathcal N(U_ix_i,U_iCU_i^\top).
$$

Thus the two covariances are identical after transport. The KL reduces exactly to

$$
\boxed{
D_{ij}
=
\frac12(x_i-x_j)^\top C^{-1}(x_i-x_j).
}
$$

Expanding the quadratic gives

$$
D_{ij}
=
\frac12x_i^\top C^{-1}x_i
-x_i^\top C^{-1}x_j
+\frac12x_j^\top C^{-1}x_j.
$$

For fixed $i$, the first term is row-constant and cancels in the row-wise softmax. Therefore

$$
\beta_{ij}
\propto
\pi_{ij}\exp\left(
\frac{1}{\tau}x_i^\top C^{-1}x_j
-
\frac{1}{2\tau}x_j^\top C^{-1}x_j
\right).
$$

If the source prior is uniform, this becomes

$$
\boxed{
\beta_{ij}
=
\operatorname{softmax}_j\left(
\frac{1}{\tau}x_i^\top C^{-1}x_j
-
\frac{1}{2\tau}x_j^\top C^{-1}x_j
\right).
}
$$

This is Gaussian/RBF attention in a shared common-frame metric.

---

## 9. Dot-product attention from key-norm cancellation

Standard dot-product attention does not include the key-norm correction

$$
-\frac{1}{2\tau}x_j^\top C^{-1}x_j.
$$

Therefore, to recover vanilla dot-product attention, one needs one of the following assumptions:

1. constant key Mahalanobis norm,
2. explicit key normalization,
3. approximate concentration of key norms,
4. absorption of the key norm into a learned or structural key bias,
5. choice to retain the richer Gaussian/RBF attention instead of vanilla attention.

If

$$
x_j^\top C^{-1}x_j=\rho^2
$$

is constant across $j$, then the key-norm term is also row-constant and cancels in the softmax. We obtain

$$
\boxed{
\beta_{ij}
=
\operatorname{softmax}_j\left(
\frac{1}{\tau}x_i^\top C^{-1}x_j
\right).
}
$$

Now define

$$
\boxed{
Q_i=C^{-1/2}x_i=C^{-1/2}U_i^{-1}\mu_i,
}
$$

and

$$
\boxed{
K_j=C^{-1/2}x_j=C^{-1/2}U_j^{-1}\mu_j.
}
$$

Then

$$
x_i^\top C^{-1}x_j=Q_i^\top K_j.
$$

Therefore

$$
\boxed{
\beta_{ij}
=
\operatorname{softmax}_j\left(\frac{Q_i^\top K_j}{\tau}\right).
}
$$

Choosing

$$
\tau=\sqrt{d_k}
$$

or absorbing the scale into $C$ gives

$$
\boxed{
\beta_{ij}
=
\operatorname{softmax}_j\left(\frac{Q_i^\top K_j}{\sqrt{d_k}}\right).
}
$$

This is standard scaled dot-product attention, derived from nontrivial relative gauge transport.

---

## 10. Orthogonal or unitary gauge transports

The derivation becomes especially simple when each $U_i$ is orthogonal or unitary. For real orthogonal frames,

$$
U_i^{-1}=U_i^\top.
$$

Assume

$$
U_i\in O(K),
\qquad
\Sigma_i=\sigma^2I.
$$

Then

$$
\Omega_{ij}=U_iU_j^\top
$$

is also orthogonal. Therefore

$$
\Omega_{ij}\Sigma_j\Omega_{ij}^\top
=
\sigma^2\Omega_{ij}\Omega_{ij}^\top
=
\sigma^2I.
$$

The covariance terms in the KL cancel, leaving

$$
\boxed{
D_{ij}
=
\frac{1}{2\sigma^2}\|\mu_i-\Omega_{ij}\mu_j\|^2.
}
$$

Expand the norm:

$$
\begin{aligned}
\|\mu_i-\Omega_{ij}\mu_j\|^2
&=\|\mu_i\|^2+\|\Omega_{ij}\mu_j\|^2-2\mu_i^\top\Omega_{ij}\mu_j
\\
&=\|\mu_i\|^2+\|\mu_j\|^2-2\mu_i^\top\Omega_{ij}\mu_j.
\end{aligned}
$$

Hence

$$
D_{ij}
=
\frac{1}{2\sigma^2}\|\mu_i\|^2
-
\frac{1}{\sigma^2}\mu_i^\top\Omega_{ij}\mu_j
+
\frac{1}{2\sigma^2}\|\mu_j\|^2.
$$

For fixed $i$, the query norm term cancels in the row-wise softmax. Thus

$$
\beta_{ij}
\propto
\pi_{ij}\exp\left(
\frac{1}{\tau\sigma^2}\mu_i^\top\Omega_{ij}\mu_j
-
\frac{1}{2\tau\sigma^2}\|\mu_j\|^2
\right).
$$

With uniform $\pi_{ij}$ and constant or normalized key norms,

$$
\boxed{
\beta_{ij}
=
\operatorname{softmax}_j\left(
\frac{1}{\tau\sigma^2}\mu_i^\top\Omega_{ij}\mu_j
\right).
}
$$

Now use

$$
\Omega_{ij}=U_iU_j^\top.
$$

Then

$$
\mu_i^\top\Omega_{ij}\mu_j
=
\mu_i^\top U_iU_j^\top\mu_j.
$$

Define

$$
\boxed{
Q_i=U_i^\top\mu_i,
\qquad
K_j=U_j^\top\mu_j.
}
$$

Then

$$
\boxed{
\mu_i^\top\Omega_{ij}\mu_j=Q_i^\top K_j.
}
$$

Therefore

$$
\boxed{
\beta_{ij}
=
\operatorname{softmax}_j\left(\frac{Q_i^\top K_j}{\tau\sigma^2}\right).
}
$$

Choosing

$$
\tau\sigma^2=\sqrt{d_k}
$$

gives

$$
\boxed{
\beta_{ij}
=
\operatorname{softmax}_j\left(\frac{Q_i^\top K_j}{\sqrt{d_k}}\right).
}
$$

This is the simplest exact route from nontrivial $\Omega_{ij}$ to standard scaled dot-product attention.

---

## 11. Value aggregation

Attention is not only a rule for weights. It also aggregates values.

Let $\nu_j$ be the value-like field carried by token/agent $j$. To aggregate values into agent $i$'s frame, the geometrically correct update is

$$
\boxed{
m_i=\sum_j\beta_{ij}\Omega_{ij}\nu_j.
}
$$

Substitute

$$
\Omega_{ij}=U_iU_j^{-1}.
$$

Then

$$
m_i
=
\sum_j\beta_{ij}U_iU_j^{-1}\nu_j.
$$

Factor out $U_i$:

$$
\boxed{
m_i
=
U_i\sum_j\beta_{ij}U_j^{-1}\nu_j.
}
$$

Move the output into the common frame:

$$
\boxed{
U_i^{-1}m_i
=
\sum_j\beta_{ij}U_j^{-1}\nu_j.
}
$$

Define

$$
\boxed{
V_j=U_j^{-1}\nu_j.
}
$$

Then

$$
\boxed{
U_i^{-1}m_i
=
\sum_j\beta_{ij}V_j.
}
$$

This is the standard transformer value aggregation rule, but written in gauge-covariant form.

In the orthogonal case,

$$
V_j=U_j^\top\nu_j.
$$

Thus the nontrivial-gauge transformer has

$$
\boxed{
Q_i=U_i^{-1}\mu_i,
\qquad
K_j=U_j^{-1}\mu_j,
\qquad
V_j=U_j^{-1}\nu_j,
}
$$

with the optional shared precision whitening factor $C^{-1/2}$ applied to $Q_i$ and $K_j$.

---

## 12. General GL(K) with locally isotropic covariances

It is tempting to assume

$$
\Sigma_i=\Sigma_j=\sigma^2I
$$

while also allowing arbitrary

$$
U_i,U_j\in\mathrm{GL}(K).
$$

This is not the cleanest gauge-covariant assumption. Non-orthogonal transformations do not preserve $\sigma^2I$.

Still, this case is useful because it shows what extra terms appear when $\Omega_{ij}$ is non-orthogonal.

With

$$
\Sigma_i=\Sigma_j=\sigma^2I,
$$

the transported covariance is

$$
\Omega_{ij}\Sigma_j\Omega_{ij}^\top
=
\sigma^2\Omega_{ij}\Omega_{ij}^\top.
$$

The KL becomes

$$
D_{ij}
=
\frac12
\left[
\log\det(\Omega_{ij}\Omega_{ij}^\top)
-K
+\operatorname{tr}\left((\Omega_{ij}\Omega_{ij}^\top)^{-1}\right)
+\frac{1}{\sigma^2}\|\Omega_{ij}^{-1}\mu_i-\mu_j\|^2
\right].
$$

Define the geometric stretch penalty

$$
\boxed{
S(\Omega_{ij})
=
\frac12
\left[
\log\det(\Omega_{ij}\Omega_{ij}^\top)
-K
+\operatorname{tr}\left((\Omega_{ij}\Omega_{ij}^\top)^{-1}\right)
\right].
}
$$

Then

$$
D_{ij}
=
S(\Omega_{ij})
+
\frac{1}{2\sigma^2}\|\Omega_{ij}^{-1}\mu_i-\mu_j\|^2.
$$

Expand the squared term:

$$
\begin{aligned}
\|\Omega_{ij}^{-1}\mu_i-\mu_j\|^2
&=
\mu_i^\top\Omega_{ij}^{-\top}\Omega_{ij}^{-1}\mu_i
-2\mu_i^\top\Omega_{ij}^{-\top}\mu_j
+\|\mu_j\|^2.
\end{aligned}
$$

Therefore the effective logit is

$$
\boxed{
\ell_{ij}
=
\log\pi_{ij}
+
\frac{1}{\tau\sigma^2}\mu_i^\top\Omega_{ij}^{-\top}\mu_j
-
\frac{1}{2\tau\sigma^2}\mu_i^\top\Omega_{ij}^{-\top}\Omega_{ij}^{-1}\mu_i
-
\frac{1}{2\tau\sigma^2}\|\mu_j\|^2
-
\frac{1}{\tau}S(\Omega_{ij}).
}
$$

This is not vanilla dot-product attention. It is a richer quadratic gauge attention rule.

The extra terms are:

$$
-\frac{1}{2\tau\sigma^2}\mu_i^\top\Omega_{ij}^{-\top}\Omega_{ij}^{-1}\mu_i,
$$

which penalizes query distortion under the inverse transport;

$$
-\frac{1}{2\tau\sigma^2}\|\mu_j\|^2,
$$

which is the key-norm correction; and

$$
-\frac{1}{\tau}S(\Omega_{ij}),
$$

which penalizes non-volume-preserving and non-isometric stretch.

Thus, for general $\mathrm{GL}(K)$ with locally isotropic covariance, the model naturally produces quadratic gauge attention rather than pure dot-product attention.

---

## 12.5 Untied query-key projections without extraneous structure

A sharper reading of section 5 yields untied $W_Q, W_K$ directly from the existing belief and frame fields, without imposing the frame-covariant closure of section 8 and without introducing extra bundles or query/key morphisms.

Return to the exact KL of section 5 with arbitrary SPD covariances $\Sigma_i, \Sigma_j$ (no closure):

$$
D_{ij}
=
\frac12
\Big[
\log\det\Sigma_j-\log\det\Sigma_i
+2\log|\det U_i|-2\log|\det U_j|
-K
+\operatorname{tr}(H_jP_i)
+x_i^\top H_jx_i
-2x_i^\top k_j
+r_j
\Big].
$$

The cross term $-2x_i^\top k_j$ is the only term coupling query and key indices through the belief means. Substituting the definitions of $x_i$ and $k_j$,

$$
\boxed{
x_i^\top k_j
=
(U_i^{-1}\mu_i)^\top(U_j^\top\Sigma_j^{-1}\mu_j)
=
\mu_i^\top\Omega_{ij}^{-\top}\Sigma_j^{-1}\mu_j
=
Q_i^\top K_j,
}
$$

where the gauge-derived query and key projections are

$$
\boxed{
Q_i=U_i^{-1}\mu_i,
\qquad
K_j=U_j^\top\Sigma_j^{-1}\mu_j.
}
$$

These two projections are not inverse-transposes of each other at the same token: $W_Q^{(i)}=U_i^{-1}$ and $W_K^{(i)}=U_i^\top\Sigma_i^{-1}$, with the inverse-transpose relation broken by the precision factor $\Sigma_i^{-1}$. The carving is therefore genuinely untied; $W_Q$ and $W_K$ are different functions of the per-token belief state, not the same projection up to symmetry. Numerical verification at $K=4,5$ over twenty random trials gives median Frobenius distance $\|W_Q^{(i)}-(W_K^{(i)})^{-\top}\|_F$ on the order of 25 to 40, with no value below 5; tying would require $\Sigma_i=I$ in every frame.

### 12.5.1 Surjectivity onto $\mathrm{GL}(K)$

The implicit pair-dependent bilinear is

$$
M_{ij}
:=
\Omega_{ij}^{-\top}\Sigma_j^{-1}
=
U_i^{-\top}U_j^\top\Sigma_j^{-1}.
$$

As $(U_i,U_j)$ ranges over $\mathrm{GL}(K)^2$ and $\Sigma_j$ over $\mathrm{SPD}(K)$, $M_{ij}$ ranges over all of $\mathrm{GL}(K)$. To realise any target $T\in\mathrm{GL}(K)$, set $U_j=I$, $\Sigma_j=I$, $U_i=T^{-\top}$; then $M_{ij}=T$. The expressive power of the gauge-derived bilinear is therefore identical to that of an independently learned $W_QW_K^\top\in\mathrm{GL}(K)$ in standard scaled dot-product attention. In particular $M_{ij}$ is generically non-symmetric, recovering the asymmetry of $W_Q^\top W_K$ that the symmetric tied $C^{-1}$ closure of section 8 cannot.

Although $M_{ij}$ depends on the pair $(i,j)$, its evaluation $\mu_i^\top M_{ij}\mu_j$ factorises through per-token vectors $Q_i, K_j$, exactly as in standard attention. Pair-independence of $M$ is not required for factorisation; what is required is that the dependence on $(i,j)$ enters only through the per-token data $(U_i,\mu_i)$ and $(U_j,\mu_j,\Sigma_j)$, which is the structure of the gauge-covariant carving.

### 12.5.2 Status of the remaining KL terms

The cross term carries the $W_QW_K^\top$ content of attention. The remaining terms in section 5 fall into two groups under the row-softmax over $j$.

Terms depending on $j$ alone -- $r_j$, $\log\det\Sigma_j$, $\log|\det U_j|$ -- contribute a key-side prior bias. They are absorbed into $\log\pi_{ij}$ and shift the distribution over which keys to attend to without altering the bilinear coupling between query and key content.

Terms coupling $i$ and $j$ through the belief covariances rather than the means -- $x_i^\top H_jx_i$ and $\operatorname{tr}(H_jP_i)$ -- are the gauge-theoretic uncertainty corrections beyond vanilla attention. They vanish under the closure $\Sigma_j=U_jCU_j^\top$ for shared SPD $C$, which collapses $H_j$ to the constant $C^{-1}$ and ties the carving to the symmetric form of section 8. They survive as legitimate uncertainty-aware terms when the closure is dropped; numerical experiments at $K=4,5$ show that without closure these terms are not row-constant in $i$ for fixed $j$, with cross-row variance $10^2$ to $10^4$ times that of the cross term, so they materially modify attention outside the closure.

### 12.5.3 Identification with rotary positional structure

The natural identification of $U_i$ with a real transformer architecture is the per-position rotational frame of rotary positional embeddings, in which $U_i\in\mathrm{O}(K)$ is a block-diagonal rotation depending on token position. Then $U_i^{-1}=U_i^\top$, the closure $\Sigma_j=U_jCU_j^\top$ holds for any $C$ commuting with the rotation block structure, and the carving reduces to $Q_i=U_i^\top\mu_i$, $K_j=U_j^\top\mu_j$. This is the rotation-modulated query-key projection used in rotary attention, with $\mu_i$ playing the role of the position-independent content vector. The trivial-frame specialisation $U_i=U$ removes per-token positional variation and reduces the framework to the constant-frame route of section 14 in which a learned bilinear $M$ replaces the structural role of frame variation.

### 12.5.4 What the untied carving does and does not prove

The carving establishes that nontrivial relative gauge transport produces untied scaled dot-product attention with a pair-dependent bilinear of full $\mathrm{GL}(K)$ expressive power, using only the existing belief field $(\mu,\Sigma)$ and per-token frames $U_i$, with no extra bundles or query/key morphisms. It does not by itself reduce the full Gaussian KL to bare $Q^\top K/\sqrt{d_k}$; the uncertainty corrections remain, and standard scaled dot-product attention is the limit in which either the closure $\Sigma_j=U_jCU_j^\top$ holds or those corrections are dominated by the cross term in the high-dimensional regime. What is recovered without further assumptions is untied $Q^\top K$ as the leading bilinear coupling.

---

## 13. Corrected theorem statement

A clean theorem for the manuscript is the following.

### Theorem: Nontrivial gauge transports recover scaled dot-product attention under frame-covariant Gaussian closure

Let each token/agent $i$ carry a Gaussian belief

$$
q_i=\mathcal N(\mu_i,\Sigma_i).
$$

Assume there exist invertible local frames $U_i\in\mathrm{GL}(K)$ and a shared covariance $C\succ0$ such that

$$
\mu_i=U_ix_i,
\qquad
\Sigma_i=U_iCU_i^\top.
$$

Let the relative transport from $j$ to $i$ be

$$
\Omega_{ij}=U_iU_j^{-1}.
$$

Then

$$
\Omega_{ij\#}q_j
=
\mathcal N(U_ix_j,U_iCU_i^\top),
$$

and hence

$$
D_{\mathrm{KL}}(q_i\Vert\Omega_{ij\#}q_j)
=
\frac12(x_i-x_j)^\top C^{-1}(x_i-x_j).
$$

Consequently, entropy-regularized KL source selection gives

$$
\beta_{ij}
=
\operatorname{softmax}_j\left(
\log\pi_{ij}
-
\frac{1}{2\tau}(x_i-x_j)^\top C^{-1}(x_i-x_j)
\right).
$$

If $\pi_{ij}$ is uniform and the key Mahalanobis norms $x_j^\top C^{-1}x_j$ are constant, normalized, or absorbed into a key bias, then

$$
\beta_{ij}
=
\operatorname{softmax}_j\left(
\frac{1}{\tau}x_i^\top C^{-1}x_j
\right).
$$

With

$$
Q_i=C^{-1/2}U_i^{-1}\mu_i,
\qquad
K_j=C^{-1/2}U_j^{-1}\mu_j,
$$

this becomes

$$
\beta_{ij}
=
\operatorname{softmax}_j\left(\frac{Q_i^\top K_j}{\tau}\right).
$$

Choosing $\tau=\sqrt{d_k}$ gives standard scaled dot-product attention:

$$
\boxed{
\beta_{ij}
=
\operatorname{softmax}_j\left(\frac{Q_i^\top K_j}{\sqrt{d_k}}\right).
}
$$

The corresponding gauge-covariant value update is

$$
\boxed{
U_i^{-1}m_i=\sum_j\beta_{ij}V_j,
\qquad
V_j=U_j^{-1}\nu_j.
}
$$

---

## 14. What this fixes relative to the Omega = I derivation

The weaker derivation sets

$$
\Omega_{ij}=I
$$

and then inserts an external bilinear form

$$
M=W_QW_K^\top.
$$

That route is mathematically possible but conceptually unsatisfying because the central geometric object $\Omega_{ij}$ disappears before attention is recovered.

The corrected derivation keeps

$$
\Omega_{ij}=U_iU_j^{-1}\neq I.
$$

Then the query-key dot product arises from comparing local beliefs after transport:

$$
\mu_i^\top\Omega_{ij}\mu_j
=
(U_i^{-1}\mu_i)^\top(U_j^{-1}\mu_j)
$$

in the orthogonal case, or more generally

$$
x_i^\top C^{-1}x_j
=
(C^{-1/2}U_i^{-1}\mu_i)^\top(C^{-1/2}U_j^{-1}\mu_j).
$$

Thus $Q$ and $K$ are gauge-normalized belief coordinates, not independently inserted projections.

The corrected conceptual chain is:

$$
\boxed{
\text{local frames }U_i,U_j
\quad\Longrightarrow\quad
\Omega_{ij}=U_iU_j^{-1}
\quad\Longrightarrow\quad
D_{\mathrm{KL}}(q_i\Vert\Omega_{ij\#}q_j)
\quad\Longrightarrow\quad
\operatorname{softmax}(-D_{ij}/\tau)
\quad\Longrightarrow\quad
\text{attention}.
}
$$

---

## 15. Manuscript-ready replacement paragraph

The transformer reduction should be stated approximately as follows.

> In the zero-dimensional specialization, spatial dependence over the base $\mathcal C$ is suppressed, but token-agents need not share a common internal frame. Let each token carry a local frame $U_i$ and let the relative gauge transport be $\Omega_{ij}=U_iU_j^{-1}$. For Gaussian beliefs $q_i=\mathcal N(\mu_i,\Sigma_i)$, the gauge-covariant discrepancy is $D_{ij}=D_{\mathrm{KL}}(q_i\Vert\Omega_{ij\#}q_j)$. Under the frame-covariant covariance closure $\Sigma_i=U_iCU_i^\top$, this KL reduces exactly to $\frac12(x_i-x_j)^\top C^{-1}(x_i-x_j)$, where $x_i=U_i^{-1}\mu_i$. Entropy-regularized source selection therefore yields Gaussian/RBF attention in the common-frame metric $C^{-1}$. If key Mahalanobis norms are constant, normalized, or absorbed into a key bias, this reduces to scaled dot-product attention with $Q_i=C^{-1/2}U_i^{-1}\mu_i$ and $K_j=C^{-1/2}U_j^{-1}\mu_j$. Value aggregation is likewise gauge-covariant: $m_i=\sum_j\beta_{ij}\Omega_{ij}\nu_j$, or equivalently $U_i^{-1}m_i=\sum_j\beta_{ij}V_j$ with $V_j=U_j^{-1}\nu_j$. Thus standard attention appears as a special normalized limit of nontrivial gauge transport, rather than requiring $\Omega_{ij}=I$ plus an externally inserted bilinear form.

---

## 16. Recommended claim calibration

The strongest accurate claim, sharper than the original frame-covariant version, is:

> Nontrivial relative gauge transports $\Omega_{ij}=U_iU_j^{-1}$ generate attention through transported Gaussian KL divergence, and the cross term in the resulting logit factors as $Q_i^\top K_j$ with $Q_i=U_i^{-1}\mu_i$ and $K_j=U_j^\top\Sigma_j^{-1}\mu_j$. The two projections are genuinely untied; the implicit pair-dependent bilinear ranges over all of $\mathrm{GL}(K)$ as the per-token frames and precisions vary, matching the expressive power of an independently learned $W_QW_K^\top$ in standard scaled dot-product attention. The construction uses only the existing belief and frame fields; no extra bundles or query/key morphisms are required.

The manuscript should avoid the stronger claim:

> Gauge transport always equals standard transformer attention as a complete identity.

The fully general nontrivial-$\Omega_{ij}$ model produces, in addition to the cross term that becomes $Q^\top K$, uncertainty corrections coupling query and key covariances that are absent from vanilla attention. Standard scaled dot-product attention is recovered when either those corrections are dominated by the cross term in the high-dimensional regime or the closure $\Sigma_j=U_jCU_j^\top$ is imposed; the closure additionally collapses the carving to the symmetric tied form.

A precise hierarchy is:

1. General $\mathrm{GL}(K)$ Gaussian case with arbitrary SPD covariance: untied gauge attention with cross-coupling uncertainty corrections.
2. General $\mathrm{GL}(K)$ Gaussian case, cross term only: untied scaled dot-product attention with implicit bilinear in $\mathrm{GL}(K)$.
3. Frame-covariant Gaussian covariance $\Sigma_j=U_jCU_j^\top$: tied scaled dot-product attention with shared SPD bilinear $C^{-1}$.
4. Orthogonal/unitary frames with isotropic covariance: rotary positional structure realised exactly.

---

## 17. Final compact derivation

Start with

$$
q_i=\mathcal N(\mu_i,\Sigma_i),
\qquad
q_j=\mathcal N(\mu_j,\Sigma_j),
$$

and

$$
\Omega_{ij}=U_iU_j^{-1}.
$$

Assume

$$
\mu_i=U_ix_i,
\qquad
\Sigma_i=U_iCU_i^\top.
$$

Then

$$
\Omega_{ij\#}q_j
=
\mathcal N(U_ix_j,U_iCU_i^\top).
$$

Therefore

$$
D_{ij}
=
D_{\mathrm{KL}}(q_i\Vert\Omega_{ij\#}q_j)
=
\frac12(x_i-x_j)^\top C^{-1}(x_i-x_j).
$$

The KL softmax gives

$$
\beta_{ij}
=
\operatorname{softmax}_j\left(-\frac{1}{2\tau}(x_i-x_j)^\top C^{-1}(x_i-x_j)\right).
$$

Expanding the quadratic and dropping row-constant terms gives

$$
\beta_{ij}
=
\operatorname{softmax}_j\left(
\frac{1}{\tau}x_i^\top C^{-1}x_j
-
\frac{1}{2\tau}x_j^\top C^{-1}x_j
\right).
$$

With key-norm normalization,

$$
\beta_{ij}
=
\operatorname{softmax}_j\left(\frac{1}{\tau}x_i^\top C^{-1}x_j\right).
$$

Set

$$
Q_i=C^{-1/2}U_i^{-1}\mu_i,
\qquad
K_j=C^{-1/2}U_j^{-1}\mu_j.
$$

Then

$$
\boxed{
\beta_{ij}
=
\operatorname{softmax}_j\left(\frac{Q_i^\top K_j}{\tau}\right).
}
$$

With $\tau=\sqrt{d_k}$,

$$
\boxed{
\beta_{ij}
=
\operatorname{softmax}_j\left(\frac{Q_i^\top K_j}{\sqrt{d_k}}\right).
}
$$

Finally, values transform as

$$
m_i=\sum_j\beta_{ij}\Omega_{ij}\nu_j.
$$

Equivalently,

$$
\boxed{
U_i^{-1}m_i=\sum_j\beta_{ij}V_j,
\qquad
V_j=U_j^{-1}\nu_j.
}
$$

This is the nontrivial-gauge derivation of transformer-style attention.
