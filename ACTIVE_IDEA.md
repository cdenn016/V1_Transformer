# Active Inference for the Gauge Transformer: Canonical Formulation

The two anchors from the existing code are:

**PriorBank already gives a principled token likelihood:**

$$p(y_t = v \mid q_t) \propto \exp\!\bigl(-\mathrm{KL}(q_t \| \pi_v) / \tau\bigr),$$

where each token $v$ has a prior belief $\pi_v = \mathcal{N}(\mu_v, \Sigma_v)$.

**The current active-inference branch** is an experimental add-on that uses entropy and BALD-style MI, differentiates them on a freshly detached $\mu$, and applies the result as separate Euclidean $\mu$-updates outside the main VFE geometry.

What follows is the mathematically cleaner replacement.

---

## 1) State, action, outcome, policy

Let the latent belief state at token position $t$ be

$$q_t(z_t) = \mathcal{N}(z_t;\, \mu_t, \Sigma_t), \quad \phi_t \in \mathfrak{g},$$

so the full internal state is

$$x_t := (\mu_t, \Sigma_t, \phi_t).$$

Let the action be the emitted next token:

$$a_t \in V.$$

Let the outcome after taking that token action be the next observed symbol or feedback:

$$o_{t+1} \in V \quad \text{or more generally} \quad o_{t+1} \in \mathcal{O}.$$

A one-step policy is just a distribution over next tokens:

$$q_t(a_t).$$

For a multi-step horizon $H$, the policy becomes a token sequence:

$$\pi_t = (a_t, a_{t+1}, \ldots, a_{t+H-1}) \in V^H.$$

---

## 2) Generative model specialized to the architecture

Three pieces are needed.

### (a) Token likelihood from PriorBank

For any latent state $z$, define token logits by

$$\ell_v(z) = -\frac{1}{\tau}\,\mathrm{KL}\!\bigl(q(z) \| \pi_v\bigr),$$

and thus

$$p(o = v \mid z) = \frac{\exp(\ell_v(z))}{\sum_{u \in V} \exp(\ell_u(z))}.$$

This is exactly the `PriorBank.decode` semantics the code describes.

### (b) Action-conditioned latent transition

A predictive state transition is needed:

$$p(z_{t+1} \mid z_t, a_t, C_t),$$

where $C_t$ is the current context.

In this architecture, the natural deterministic approximation is:

$$x_{t+1}(a) = T_\theta\!\bigl(x_t,\, C_t \oplus a\bigr),$$

meaning: append candidate token $a$ to the context, then run one forward latent update of the transformer/VFE stack.

For a stochastic version:

$$q(z_{t+1} \mid a_t) \approx \delta\!\bigl(z_{t+1} - \mu_{t+1}(a)\bigr)$$

or

$$q(z_{t+1} \mid a_t) = \mathcal{N}\!\bigl(\mu_{t+1}(a),\, \Sigma_{t+1}(a)\bigr).$$

### (c) Preferences over future outcomes

Define a preferred outcome distribution:

$$p^*(o_{t+1} \mid G_t),$$

where $G_t$ is the goal/task/prompt state. Examples:

- **Supervised training:** $p^*(o_{t+1}) = \delta(o_{t+1} = y_{t+1}^{\text{target}})$
- **Generation:** a reward-conditioned or task-conditioned target distribution over acceptable next tokens
- **Safety / style:** a preference distribution that downweights undesirable continuations

This is the piece the current entropy-minimization pragmatic term is missing.

---

## 3) One-step expected free energy for token actions

For each candidate next token $a$, define:

$$G_t(a) = \underbrace{E_{q(o_{t+1} \mid a)}\bigl[-\log p^*(o_{t+1})\bigr]}_{\text{risk / preference mismatch}} + \underbrace{E_{q(z_{t+1} \mid a)}\bigl[H\bigl(p(o_{t+1} \mid z_{t+1})\bigr)\bigr]}_{\text{ambiguity}} - \underbrace{I_q(z_{t+1};\, o_{t+1} \mid a)}_{\text{epistemic value}}.$$

That is the clean active-inference object. Now specialize each term.

**Risk.** If the next-state prediction under token $a$ is $q(z_{t+1} \mid a)$, then

$$q(o_{t+1} = v \mid a) = \int p(v \mid z)\, q(z \mid a)\, dz.$$

Then risk is

$$\mathrm{Risk}_t(a) = \sum_{v \in V} q(o_{t+1} = v \mid a)\,\bigl(-\log p^*(v)\bigr).$$

In the supervised delta-preference case $p^*(v) = \delta(v = y_{t+1})$, this becomes the ordinary negative log-probability of the target token under the predicted post-action outcome distribution.

**Ambiguity.**

$$\mathrm{Amb}_t(a) = E_{q(z_{t+1} \mid a)}\!\left[-\sum_{v \in V} p(v \mid z_{t+1})\,\log p(v \mid z_{t+1})\right].$$

This penalizes action $a$ if it leads to latent states whose likelihood model is intrinsically ambiguous.

**Epistemic value.**

$$\mathrm{Epi}_t(a) = I_q(z_{t+1};\, o_{t+1} \mid a) = H\bigl[q(o_{t+1} \mid a)\bigr] - E_{q(z_{t+1} \mid a)}\bigl[H\bigl[p(o_{t+1} \mid z_{t+1})\bigr]\bigr].$$

This is the proper information-gain term. It is close in spirit to the current BALD-style term, but here it is conditioned on an explicit candidate action $a$, not just on the current belief. The current code uses the BALD identity but only as a detached $\mu$-side update.

**Overall:**

$$G_t(a) = \mathrm{Risk}_t(a) + \mathrm{Amb}_t(a) - \mathrm{Epi}_t(a).$$

---

## 4) Policy posterior over next tokens

Define the next-token policy posterior by Boltzmann selection on expected free energy:

$$q_t(a_t = a) = \frac{\exp\!\bigl(-\gamma\, G_t(a)\bigr)}{\sum_{u \in V} \exp\!\bigl(-\gamma\, G_t(u)\bigr)},$$

where $\gamma > 0$ is a policy precision.

This is the correct analog of "sample next token from the policy."

There are now two distributions:

- **Likelihood/readout** $p(o \mid z)$ from `PriorBank.decode`
- **Policy posterior** $q(a)$ from expected free energy

They are not the same object.

---

## 5) How this plugs into the VFE objective

The current E-step objective is documented as:

$$F = \alpha \sum_i \mathrm{KL}(q_i \| p_i) + \lambda_{\text{belief}} \sum_{i,j} \beta_{ij}\,\mathrm{KL}(q_i \| \Omega_{ij}\, q_j) + \lambda_{\text{softmax}} \sum_{i,j} \mathrm{KL}_{ij}\,\partial\beta_{ij}/\partial\theta + \mathrm{CE}(W_{\text{out}}\,\mu,\, \text{targets}),$$

with natural-gradient descent in $(\mu, \Sigma, \phi)$.

The clean active-inference replacement is:

$$J_t\bigl(q_t,\, q_t(a_t)\bigr) = F_t(q_t) + \eta\, E_{q_t(a_t)}\!\bigl[G_t(a_t)\bigr] + \frac{1}{\gamma}\, H\bigl[q_t(a_t)\bigr].$$

Equivalent expanded form:

$$J_t = F_t(q_t) + \eta \sum_{a \in V} q_t(a)\, G_t(a) + \frac{1}{\gamma} \sum_{a \in V} q_t(a)\, \log q_t(a).$$

Then:

- **E-step over latent beliefs:** $(\mu_t, \Sigma_t, \phi_t) \leftarrow \arg\min\, J_t$
- **Policy inference:** $q_t(a_t) \propto \exp(-\gamma\, G_t(a_t))$

No detached side graph. No separate Euclidean AI update. One objective.

---

## 6) Gradients needed

For theoretical correctness, differentiate with respect to the full state:

$$\nabla_{\mu_t, \Sigma_t, \phi_t}\, J_t.$$

That means the EFE piece contributes:

$$\nabla_x E_{q(a)}\bigl[G_t(a)\bigr] = \sum_{a \in V} q(a)\, \nabla_x G_t(a) + \sum_{a \in V} G_t(a)\, \nabla_x q(a), \quad x \in \{\mu, \Sigma, \phi\}.$$

In practice, if $q(a) \propto \exp(-\gamma\, G(a))$ is solved explicitly, then the second term is already encoded through the softmax dependence of $q(a)$ on $G(a)$.

So the clean implementation path is:

1. Compute $G_t(a)$ for a candidate set of actions
2. Build $q_t(a)$
3. Form the scalar objective $L_{\text{AI},t} = \eta \sum_a q_t(a)\, G_t(a) + \frac{1}{\gamma} \sum_a q_t(a)\, \log q_t(a)$
4. Backpropagate this scalar through the same graph that produced $x_t = (\mu_t, \Sigma_t, \phi_t)$
5. Merge with the existing VFE gradients before natural-gradient projection / SPD retraction / $\phi$-update

That is what the current implementation does not do. It instead keeps EFE as a detached $\mu$-only auxiliary path.

---

## 7) Practical approximation specialized to this model

Exact evaluation over all $a \in V$ is too expensive. So use a top-$K$ candidate set:

$$A_t = \mathrm{TopK}\!\bigl(p(o_{t+1} \mid z_t)\bigr)$$

from the current `PriorBank.decode` distribution.

For each $a \in A_t$:

1. Append $a$ hypothetically
2. Run one latent transition: $x_{t+1}(a) = T_\theta(x_t,\, C_t \oplus a)$
3. Compute the predictive outcome distribution: $p_{t+1}^{(a)}(v) = p(o_{t+1} = v \mid x_{t+1}(a))$
4. Estimate $G_t(a) = \mathrm{Risk}_t(a) + \mathrm{Amb}_t(a) - \mathrm{Epi}_t(a)$

Then

$$q_t(a) = \frac{\exp(-\gamma\, G_t(a))}{\sum_{u \in A_t} \exp(-\gamma\, G_t(u))}.$$

This makes the computation tractable and still mathematically aligned.

---

## 8) Sequence-of-next-tokens version

Extend to horizon $H$. Let

$$\pi = (a_t, a_{t+1}, \ldots, a_{t+H-1}).$$

Roll forward recursively:

$$x_{t+k+1}(\pi) = T_\theta\!\bigl(x_{t+k}(\pi),\, C_{t+k}(\pi) \oplus a_{t+k}\bigr).$$

Then define cumulative EFE:

$$G_t(\pi) = \sum_{k=0}^{H-1} \bigl(\mathrm{Risk}_{t+k}(\pi) + \mathrm{Amb}_{t+k}(\pi) - \mathrm{Epi}_{t+k}(\pi)\bigr).$$

And policy posterior:

$$q_t(\pi) \propto \exp\!\bigl(-\gamma\, G_t(\pi)\bigr).$$

Exact search over $V^H$ is impossible, so use:

- Beam search on $-G_t(\pi)$
- Particle search
- Top-$K$ branching at each step

This is the mathematically clean version of "policy is a sequence of next tokens."

---

## 9) Training vs generation

This split is important.

**During generation:** The token really is an action: $a_t \sim q_t(a_t)$. So the above active-inference formulation is literal.

**During teacher-forced training:** The next token in the dataset is not chosen by the model, so it is better treated as an observed outcome. In that case, the generative and policy machinery can still be trained by using a preference distribution concentrated on the target token:

$$p^*(o_{t+1}) = \delta(o_{t+1} = y_{t+1}^{\text{target}}),$$

but conceptually this is supervised preference fitting, not autonomous policy selection.

---

## 10) Exact implementation changes

Replace the current active-inference path with this design:

**A. New scalar objective inside VariationalFFNDynamic.** Add

$$L_{\text{AI},t} = \eta \sum_a q_t(a)\, G_t(a) + \frac{1}{\gamma} \sum_a q_t(a)\, \log q_t(a)$$

to the same forward graph as the VFE objective.

**B. Remove detached AI gradients.** Delete the conceptual pattern: detached `mu_current` $\to$ local autograd in `active_inference.py` $\to$ separate Euclidean `apply_ai_mu_updates`. That breaks the unified geometry the VFE path is built around. The current module explicitly does this detached, local update.

**C. Differentiate through full state.** Include gradients for $\mu$, $\Sigma$, $\phi$ -- not only $\mu$.

**D. Require PriorBank for theory-facing runs.** Since `PriorBank.decode` is the principled KL-based token likelihood and the `W_out` fallback is explicitly less principled, do not use the fallback for claims about active inference.

**E. Keep the current incompatibility checks.** The code already correctly hard-errors when active inference is combined with `closed_form_e_step=True`, because then the iterative loop where EFE acts is bypassed. Keep that logic.

---

## 11) Minimal workable first version

If a clean first implementation is desired without exploding compute:

1. Use current latent state $x_t = (\mu_t, \Sigma_t, \phi_t)$
2. Use `PriorBank.decode` for $p(o \mid z)$
3. Build candidate token set $A_t$ from top-$K$ decode logits
4. For each $a \in A_t$, compute one-step rollout $x_{t+1}(a)$
5. Compute $G_t(a) = \mathrm{Risk}_t(a) + \mathrm{Amb}_t(a) - \mathrm{Epi}_t(a)$
6. Form $q_t(a) \propto \exp(-\gamma\, G_t(a))$
7. Add $\eta \sum_a q_t(a)\, G_t(a) + \frac{1}{\gamma} \sum_a q_t(a)\, \log q_t(a)$ to the same scalar loss used for the E-step
8. Backpropagate jointly through the full state

That gives a real next-token active-inference policy while staying as close as possible to the existing architecture.

The next step after that is the horizon-$H$ sequence-policy version.
