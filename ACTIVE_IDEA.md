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

The next step after that is the horizon-$H$ sequence-policy version (Section 12).

---

## 12) Horizon-$H$ sequence-policy extension

The one-step formulation (Sections 3--7) selects a single next token by minimizing $G_t(a)$. The horizon extension plans over token *sequences* and executes only the first token — the standard model-predictive-control (MPC) version of active inference.

### 12.1) Receding-horizon control

A candidate policy is a future token sequence:

$$\pi_t = (a_t, a_{t+1}, \ldots, a_{t+H-1}), \quad a_{t+k} \in V.$$

The policy posterior is:

$$q_t(\pi_t) \propto \exp\!\bigl(-\gamma\, G_t(\pi_t)\bigr).$$

At run time, execute only the first token:

$$a_t \sim q_t(a_t) \quad \text{where} \quad q_t(a_t) = \sum_{\pi : \pi_0 = a_t} q_t(\pi),$$

or greedily:

$$a_t = \arg\min_{a_t}\, E_{q_t(\pi \mid a_t)}\bigl[G_t(\pi)\bigr].$$

After emitting $a_t$, update the context and belief state, then replan from $t+1$. This is the correct horizon-$H$ active-inference loop.

### 12.2) Rollout dynamics

The internal state includes the context:

$$x_t := (\mu_t, \Sigma_t, \phi_t, C_t).$$

For each policy $\pi$, define a recursive rollout:

$$x_{t+k+1}(\pi) = T_\theta\!\bigl(x_{t+k}(\pi),\, a_{t+k}\bigr), \quad k = 0, \ldots, H-1,$$

with $x_t(\pi) = x_t$. Here $T_\theta$ means: append token $a_{t+k}$ to the running context, run the model forward, extract the new belief state $(\mu, \Sigma, \phi)$. The policy is defined through its effect on future latent belief trajectories.

### 12.3) Per-step outcome model

At each rolled-out state $x_{t+k+1}(\pi)$, the predictive outcome distribution is:

$$p_\theta(o_{t+k+1} = v \mid x_{t+k+1}(\pi)) = \frac{\exp\!\bigl(-\mathrm{KL}(q_{t+k+1}(\pi) \| \pi_v) / \tau\bigr)}{\sum_{u \in V} \exp\!\bigl(-\mathrm{KL}(q_{t+k+1}(\pi) \| \pi_u) / \tau\bigr)}.$$

This is exactly `PriorBank.decode` evaluated at the rolled-out beliefs. Do not use the linear $W_{\text{out}}$ fallback for the horizon planner.

### 12.4) Expected free energy for a token sequence

For each future step $k = 0, \ldots, H-1$:

$$\mathrm{Risk}_{t+k}(\pi) = E_{q(o_{t+k+1} \mid \pi)}\bigl[-\log p^*(o_{t+k+1} \mid G_t)\bigr],$$

$$\mathrm{Amb}_{t+k}(\pi) = E_{q(z_{t+k+1} \mid \pi)}\bigl[H\bigl(p(o_{t+k+1} \mid z_{t+k+1})\bigr)\bigr],$$

$$\mathrm{Epi}_{t+k}(\pi) = I_q(z_{t+k+1};\, o_{t+k+1} \mid \pi).$$

The horizon-$H$ expected free energy is:

$$G_t(\pi) = \sum_{k=0}^{H-1} \bigl(\mathrm{Risk}_{t+k}(\pi) + \mathrm{Amb}_{t+k}(\pi) - \mathrm{Epi}_{t+k}(\pi)\bigr) + V_{\text{term}}(x_{t+H}(\pi)),$$

where $V_{\text{term}}$ is an optional terminal value that prevents short horizons from overvaluing immediate confidence:

$$V_{\text{term}}(x_{t+H}) = \lambda_{\text{term}}\, E_{q(o_{t+H+1} \mid x_{t+H})}\bigl[-\log p^*(o_{t+H+1})\bigr].$$

This is the risk evaluated at the terminal state — how well the final rolled-out beliefs predict preferred outcomes.

### 12.5) The epistemic term is weaker for self-generated rollouts

This is the most important conceptual point for the horizon extension. If future "observations" inside the horizon are the model's own emitted tokens, then those outcomes are under the model's control, not exogenous. The epistemic term $I(z_{t+k+1}; o_{t+k+1} \mid \pi)$ becomes much less meaningful because the model is not gathering external evidence — it is observing the consequences of its own choices.

For pure autoregressive self-rollout:

- **Risk** still makes sense — does the sequence lead to preference-consistent predictions?
- **Ambiguity** still makes sense — does the sequence lead to confident predictions?
- **Epistemic** is weaker and can become almost vacuous.

The epistemic term becomes genuinely meaningful when the horizon includes at least one exogenous observation: a user reply, a tool result, a retrieval result, an environment state update. In dialogue, this is natural: $\pi_t$ = the model's next few tokens, $o_{t+H}$ = the user's response.

This motivates two cases:

**Case A: Language-only planning.** The policy is a self-generated token sequence. Use:

$$G_t(\pi) = \sum_{k=0}^{H-1} \bigl(\mathrm{Risk}_{t+k}(\pi) + \lambda_{\text{amb}}\, \mathrm{Amb}_{t+k}(\pi)\bigr) + V_{\text{term}}.$$

Drop or heavily downweight the epistemic term.

**Case B: Agentic planning.** The policy includes actions whose consequences include an external observation. Use the full:

$$G_t(\pi) = \sum_{k=0}^{H-1} \bigl(\mathrm{Risk}_{t+k}(\pi) + \lambda_{\text{amb}}\, \mathrm{Amb}_{t+k}(\pi) - \lambda_{\text{epi}}\, \mathrm{Epi}_{t+k}(\pi)\bigr) + V_{\text{term}}.$$

This is the version that most faithfully earns the label "active inference."

### 12.6) Tractable search over token-sequence policies

Exact enumeration of $|V|^H$ sequences is impossible. The planner uses approximate search.

**Step 1: Candidate pruning.** At each state $x$, build a small candidate action set from the current readout:

$$A(x) = \mathrm{TopK}\!\bigl(p(o \mid x)\bigr).$$

**Step 2: Beam expansion.** Construct a beam of partial policies. Starting from $K$ first-token candidates, extend each by its own top-$K$ continuations. Prune to beam width $B$ by cumulative $G$. For each partial sequence, maintain:

- The current rollout state $x_{t+k}(\pi)$
- The accumulated EFE $\sum_{j < k} G_{t+j}$
- Optional uncertainty statistics

**Step 3: Receding-horizon execution.** After scoring all complete sequences of length $H$, marginalize or minimize over the first token, emit it, and replan from the updated state.

**Pseudocode:**

```
function beam_search_efe(model, context, H, K, B, gamma, preferences):
    # Initialize beam with top-K first tokens
    current_logits = model.forward(context)
    candidates_0 = TopK(softmax(current_logits[-1]), K)
    beam = [(token, 0.0, context) for token in candidates_0]
    
    for step in range(H):
        next_beam = []
        for (partial_seq, cum_G, ctx) in beam:
            # Rollout: append last token, forward pass
            x_rolled = model.forward(ctx + [partial_seq[-1]], return_agents=True)
            logits_rolled = model._compute_logits(x_rolled.mu, x_rolled.sigma)
            probs_rolled = softmax(logits_rolled[-1])
            
            # Score this step
            risk_k = compute_risk(probs_rolled, preferences)
            amb_k = compute_ambiguity(probs_rolled)
            G_k = risk_k + lambda_amb * amb_k
            
            if step < H - 1:
                # Expand: top-K continuations
                next_tokens = TopK(probs_rolled, K)
                for a_next in next_tokens:
                    next_beam.append((
                        partial_seq + [a_next],
                        cum_G + G_k,
                        ctx + [partial_seq[-1]]
                    ))
            else:
                # Terminal: add V_term
                V_term = lambda_term * risk_at_terminal(probs_rolled, preferences)
                next_beam.append((partial_seq, cum_G + G_k + V_term, ctx))
        
        # Prune to beam width B
        next_beam.sort(key=lambda x: x[1])  # ascending G (lower is better)
        beam = next_beam[:B]
    
    # Marginalize over first token
    first_token_scores = {}
    for (seq, G_total, _) in beam:
        a0 = seq[0]
        if a0 not in first_token_scores:
            first_token_scores[a0] = []
        first_token_scores[a0].append(G_total)
    
    # Policy over first tokens
    q_a0 = softmax(-gamma * min_G_per_first_token)
    return sample(q_a0)
```

### 12.7) Integration with the VFE machinery

**Option 1: Planner at generation time only (recommended first).** Leave the VFE E-step as the latent-state updater. Place the sequence-policy planner on top during decoding only:

1. Run VFE inference to obtain $x_t = (\mu_t, \Sigma_t, \phi_t)$
2. Build candidate token sequences via beam search
3. Roll out $x_{t+1}, \ldots, x_{t+H}$ for each beam entry
4. Compute $G_t(\pi)$ for each sequence
5. Choose first token, replan

**Option 2: Planner as part of the training objective (deferred).** Add $\eta\, E_{q(\pi)}[G_t(\pi)] + \frac{1}{\gamma}\, H[q(\pi)]$ to the scalar objective and differentiate through the planner. More unified theoretically, but much more expensive and brittle. Inherits the same incompatibilities that the current code already guards against (closed-form E-step, DEQ).

### 12.8) Recommended implementation path

**Phase 1:** Implement `beam_search_efe(model, context, H, K, B, gamma, preferences)` in `expected_free_energy.py`, building on the existing `rollout_candidates` and `compute_efe` functions.

**Phase 2:** At decode time: infer $x_t$ via the VFE stack, run beam search over token sequences, choose first token by minimizing $G$, append token, replan.

**Phase 3:** Only after validating Phase 1--2, consider making the planner differentiable for training.

### 12.9) First implementation formula

For language-only planning (Case A), the first horizon planner uses:

$$G_t(\pi) = \sum_{k=0}^{H-1} \left[\sum_{v \in V} p(v \mid x_{t+k+1}(\pi))\,\bigl(-\log p^*(v)\bigr) + \lambda_{\text{amb}}\, H\bigl(p(\cdot \mid x_{t+k+1}(\pi))\bigr)\right] + \lambda_{\text{term}} \sum_{v \in V} p(v \mid x_{t+H}(\pi))\,\bigl(-\log p^*(v)\bigr).$$

Once external observations enter the horizon (agentic setting), upgrade to the full Risk + Ambiguity $-$ Epistemic formulation.

I’m extending the one-step token-action setup into a proper finite-horizon planner now: rollout dynamics, sequence policy posterior, tractable search, and how the gradients should flow back into the same VFE geometry.

Thought for 1m 23s

Good. The horizon-
𝐻
H version should be done as receding-horizon policy inference over token sequences, not as “pick 
𝐻
H tokens once and commit forever.”

That gives you the clean active-inference structure:

𝜋
𝑡
=
(
𝑎
𝑡
,
𝑎
𝑡
+
1
,
…
,
𝑎
𝑡
+
𝐻
−
1
)
,
𝑎
𝑡
+
𝑘
∈
𝑉
,
π
t
	​

=(a
t
	​

,a
t+1
	​

,…,a
t+H−1
	​

),a
t+k
	​

∈V,

with a policy posterior

𝑞
𝑡
(
𝜋
𝑡
)
∝
exp
⁡
 ⁣
(
−
𝛾
 
𝐺
𝑡
(
𝜋
𝑡
)
)
,
q
t
	​

(π
t
	​

)∝exp(−γG
t
	​

(π
t
	​

)),

and at run time you execute only the first token 
𝑎
𝑡
a
t
	​

, then re-infer at 
𝑡
+
1
t+1 from the updated belief state and context.

That is the standard model-predictive-control version of active inference, and it is the right way to use token sequences as policies.

Your current code is not doing this yet. The active-inference module is still an experimental EFE add-on based on entropy and BALD-style MI, computed on a freshly detached 
𝜇
μ leaf and then applied as a separate Euclidean 
𝜇
μ-update outside the main VFE graph. PriorBank.decode is already the principled KL-based token likelihood, while the W_out fallback is explicitly described as less principled.

1) Horizon-
𝐻
H sequence policy

Let the internal state be

𝑥
𝑡
:
=
(
𝜇
𝑡
,
Σ
𝑡
,
𝜙
𝑡
,
𝐶
𝑡
)
,
x
t
	​

:=(μ
t
	​

,Σ
t
	​

,ϕ
t
	​

,C
t
	​

),

where 
𝐶
𝑡
C
t
	​

 is the current token context.

A candidate policy is a future token sequence

𝜋
𝑡
=
(
𝑎
𝑡
,
𝑎
𝑡
+
1
,
…
,
𝑎
𝑡
+
𝐻
−
1
)
.
π
t
	​

=(a
t
	​

,a
t+1
	​

,…,a
t+H−1
	​

).

For each policy, define a rollout:

𝑥
𝑡
+
𝑘
+
1
(
𝜋
)
=
𝑇
𝜃
 ⁣
(
𝑥
𝑡
+
𝑘
(
𝜋
)
,
𝑎
𝑡
+
𝑘
)
,
𝑘
=
0
,
…
,
𝐻
−
1
,
x
t+k+1
(π)
	​

=T
θ
	​

(x
t+k
(π)
	​

,a
t+k
	​

),k=0,…,H−1,

with

𝑥
𝑡
(
𝜋
)
=
𝑥
𝑡
.
x
t
(π)
	​

=x
t
	​

.

Here 
𝑇
𝜃
T
θ
	​

 is your action-conditioned latent transition. In the exact version, 
𝑇
𝜃
T
θ
	​

 means:

append token 
𝑎
𝑡
+
𝑘
a
t+k
	​

 to the running context,
run the model forward,
extract the new latent belief state 
(
𝜇
,
Σ
,
𝜙
)
(μ,Σ,ϕ).

So the policy is not defined directly in observation space; it is defined through its effect on future latent belief trajectories.

2) Outcome model at each future step

At each future rollout state 
𝑥
𝑡
+
𝑘
+
1
(
𝜋
)
x
t+k+1
(π)
	​

, define the predictive outcome distribution using PriorBank.decode:

𝑝
𝜃
(
𝑜
𝑡
+
𝑘
+
1
=
𝑣
∣
𝑥
𝑡
+
𝑘
+
1
(
𝜋
)
)
=
exp
⁡
 ⁣
(
−
K
L
(
𝑞
𝑡
+
𝑘
+
1
(
𝜋
)
∥
𝜋
𝑣
)
/
𝜏
)
∑
𝑢
∈
𝑉
exp
⁡
 ⁣
(
−
K
L
(
𝑞
𝑡
+
𝑘
+
1
(
𝜋
)
∥
𝜋
𝑢
)
/
𝜏
)
.
p
θ
	​

(o
t+k+1
	​

=v∣x
t+k+1
(π)
	​

)=
∑
u∈V
	​

exp(−KL(q
t+k+1
(π)
	​

∥π
u
	​

)/τ)
exp(−KL(q
t+k+1
(π)
	​

∥π
v
	​

)/τ)
	​

.

This is exactly the right place to use PriorBank, because the module already defines token decoding as KL to token priors.

If you want theoretical cleanliness, do not use the linear W_out fallback for the horizon planner.

3) Expected free energy for a token sequence

For each future step, define:

Risk
R
i
s
k
𝑡
+
𝑘
(
𝜋
)
=
𝐸
𝑞
(
𝑜
𝑡
+
𝑘
+
1
∣
𝜋
)
[
−
log
⁡
𝑝
\*
(
𝑜
𝑡
+
𝑘
+
1
∣
𝐺
𝑡
)
]
,
Risk
t+k
	​

(π)=E
q(o
t+k+1
	​

∣π)
	​

[−logp
\*
(o
t+k+1
	​

∣G
t
	​

)],

where 
𝑝
\*
p
\*
 is a preference distribution conditioned on the task/goal 
𝐺
𝑡
G
t
	​

.

Ambiguity
A
m
b
𝑡
+
𝑘
(
𝜋
)
=
𝐸
𝑞
(
𝑧
𝑡
+
𝑘
+
1
∣
𝜋
)
[
𝐻
(
𝑝
(
𝑜
𝑡
+
𝑘
+
1
∣
𝑧
𝑡
+
𝑘
+
1
)
)
]
.
Amb
t+k
	​

(π)=E
q(z
t+k+1
	​

∣π)
	​

[H(p(o
t+k+1
	​

∣z
t+k+1
	​

))].
Epistemic value
E
p
i
𝑡
+
𝑘
(
𝜋
)
=
𝐼
𝑞
(
𝑧
𝑡
+
𝑘
+
1
;
𝑜
𝑡
+
𝑘
+
1
∣
𝜋
)
.
Epi
t+k
	​

(π)=I
q
	​

(z
t+k+1
	​

;o
t+k+1
	​

∣π).

Then the horizon-
𝐻
H expected free energy is

𝐺
𝑡
(
𝜋
)
=
∑
𝑘
=
0
𝐻
−
1
(
R
i
s
k
𝑡
+
𝑘
(
𝜋
)
+
A
m
b
𝑡
+
𝑘
(
𝜋
)
−
E
p
i
𝑡
+
𝑘
(
𝜋
)
)
+
𝑉
term
(
𝑥
𝑡
+
𝐻
(
𝜋
)
)
,
G
t
	​

(π)=
k=0
∑
H−1
	​

(Risk
t+k
	​

(π)+Amb
t+k
	​

(π)−Epi
t+k
	​

(π))+V
term
	​

(x
t+H
(π)
	​

),

where 
𝑉
term
V
term
	​

 is an optional terminal value.

The terminal value is useful because short horizons otherwise overvalue immediate confidence. A reasonable first terminal term is:

𝑉
term
(
𝑥
𝑡
+
𝐻
)
=
𝜆
term
 
𝐸
𝑞
(
𝑜
𝑡
+
𝐻
+
1
∣
𝑥
𝑡
+
𝐻
)
[
−
log
⁡
𝑝
\*
(
𝑜
𝑡
+
𝐻
+
1
)
]
.
V
term
	​

(x
t+H
	​

)=λ
term
	​

E
q(o
t+H+1
	​

∣x
t+H
	​

)
	​

[−logp
\*
(o
t+H+1
	​

)].
4) The crucial nuance: pure self-generated token rollouts weaken the epistemic term

This is the biggest conceptual issue for sequence-token active inference.

If the future “observations” inside the horizon are just the model’s own emitted tokens, then those outcomes are mostly under the model’s control. In that case, the epistemic term

𝐼
(
𝑧
𝑡
+
𝑘
+
1
;
𝑜
𝑡
+
𝑘
+
1
∣
𝜋
)
I(z
t+k+1
	​

;o
t+k+1
	​

∣π)

becomes much less meaningful, because the model is not gathering external evidence; it is largely observing the consequences of its own choices.

So for pure autoregressive self-rollout:

the risk term still makes sense,
the ambiguity term still makes sense,
the epistemic term is weaker and can become almost vacuous.

The epistemic term becomes genuinely meaningful when the horizon includes at least one exogenous observation, such as:

a user reply,
a tool result,
a retrieval result,
an environment state update.

That means the cleanest use of sequence-policy active inference is:

internal token sequence as action prefix
followed by an expected external observation

In dialogue, that is very natural:

𝜋
𝑡
=
my next few tokens/actions
,
𝑜
𝑡
+
𝐻
=
user/tool/environment response
.
π
t
	​

=my next few tokens/actions,o
t+H
	​

=user/tool/environment response.

That is where epistemic value really lives.

5) Receding-horizon control rule

Once you compute 
𝐺
𝑡
(
𝜋
)
G
t
	​

(π), define

𝑞
𝑡
(
𝜋
)
=
exp
⁡
(
−
𝛾
𝐺
𝑡
(
𝜋
)
)
∑
𝜋
′
exp
⁡
(
−
𝛾
𝐺
𝑡
(
𝜋
′
)
)
.
q
t
	​

(π)=
∑
π
′
	​

exp(−γG
t
	​

(π
′
))
exp(−γG
t
	​

(π))
	​

.

Then choose only the first token:

𝑎
𝑡
∼
𝑞
𝑡
(
𝑎
𝑡
)
or
𝑎
𝑡
=
arg
⁡
min
⁡
𝑎
𝑡
𝐸
𝑞
𝑡
(
𝜋
∣
𝑎
𝑡
)
[
𝐺
𝑡
(
𝜋
)
]
.
a
t
	​

∼q
t
	​

(a
t
	​

)ora
t
	​

=arg
a
t
	​

min
	​

E
q
t
	​

(π∣a
t
	​

)
	​

[G
t
	​

(π)].

After emitting 
𝑎
𝑡
a
t
	​

, update the context and belief state, then replan.

This is the correct horizon-
𝐻
H active-inference loop.

6) Tractable search over token-sequence policies

You cannot enumerate 
∣
𝑉
∣
𝐻
∣V∣
H
. So the planner should be approximate.

The right first version is:

Step 1: candidate pruning

At each state 
𝑥
x, build a small candidate action set

𝐴
(
𝑥
)
=
TopK
⁡
(
𝑝
(
𝑜
∣
𝑥
)
)
.
A(x)=TopK(p(o∣x)).

That uses the current PriorBank.decode distribution as a proposal.

Step 2: beam or particle expansion

Construct a beam of partial policies:

(
𝑎
𝑡
)
,
(
𝑎
𝑡
,
𝑎
𝑡
+
1
)
,
…
,
(
𝑎
𝑡
,
…
,
𝑎
𝑡
+
𝐻
−
1
)
.
(a
t
	​

),(a
t
	​

,a
t+1
	​

),…,(a
t
	​

,…,a
t+H−1
	​

).

For each partial sequence, keep:

current rollout state,
accumulated 
𝐺
G,
optional uncertainty statistics.
Step 3: receding-horizon execution

After choosing the best or sampled first token, discard the rest of the beam and replan from the new state.

That is much more stable than trying to backprop through a huge discrete tree.

7) A clean decomposition for your use case

For your model, I would separate two cases.

Case A: “language-only planning”

The policy is a future token sequence generated by the model itself.

Then use:

𝐺
𝑡
(
𝜋
)
=
∑
𝑘
=
0
𝐻
−
1
(
R
i
s
k
𝑡
+
𝑘
(
𝜋
)
+
𝜆
amb
 
A
m
b
𝑡
+
𝑘
(
𝜋
)
)
+
𝑉
term
.
G
t
	​

(π)=
k=0
∑
H−1
	​

(Risk
t+k
	​

(π)+λ
amb
	​

Amb
t+k
	​

(π))+V
term
	​

.

I would either drop the epistemic term here or weight it very lightly, because there is little genuine external information gain.

Case B: “agentic planning”

The policy is a token/tool/action sequence whose consequences include an external observation.

Then use the full

𝐺
𝑡
(
𝜋
)
=
R
i
s
k
+
A
m
b
i
g
u
i
t
y
−
E
p
i
s
t
e
m
i
c
.
G
t
	​

(π)=Risk+Ambiguity−Epistemic.

This is the version that most faithfully deserves the label “active inference.”

8) How this should integrate with the VFE machinery

There are really two valid ways to integrate the horizon planner.

Option 1: planner at generation time only

Leave the current VFE E-step as the latent-state updater, and place the sequence-policy planner on top of it only during decoding.

Then the architecture is:

run VFE inference to obtain 
𝑥
𝑡
=
(
𝜇
𝑡
,
Σ
𝑡
,
𝜙
𝑡
)
x
t
	​

=(μ
t
	​

,Σ
t
	​

,ϕ
t
	​

),
build candidate token sequences,
roll out 
𝑥
𝑡
+
1
,
…
,
𝑥
𝑡
+
𝐻
x
t+1
	​

,…,x
t+H
	​

,
compute 
𝐺
𝑡
(
𝜋
)
G
t
	​

(π),
choose first token and replan.

This is the most practical first implementation.

Option 2: planner as part of the training objective

Add

𝜂
 
𝐸
𝑞
(
𝜋
)
[
𝐺
𝑡
(
𝜋
)
]
+
1
𝛾
𝐻
[
𝑞
(
𝜋
)
]
ηE
q(π)
	​

[G
t
	​

(π)]+
γ
1
	​

H[q(π)]

to the scalar objective and differentiate through the planner.

That is more theoretically unified, but much more expensive and much more brittle. It also inherits the same warning your current code already discovered: if you use closed-form or DEQ-style implicit differentiation, the planner must be built into the same forward operator, otherwise the gradients are wrong. The current code already warns or hard-fails around active-inference incompatibilities for similar reasons.

So I would do Option 1 first.

9) Recommended implementation path

The clean implementation sequence is:

Phase 1

Implement a new module, conceptually token_policy_planner.py, with:

rollout_one_step(x, a) -> x_next
score_action_sequence(x, pi, goal) -> G(pi)
beam_search_policies(x, goal, H, K, B)

where:

𝐻
H = horizon,
𝐾
K = top-
𝐾
K branching,
𝐵
B = beam width.
Phase 2

At decode time:

infer current latent state with the transformer/VFE stack,
run beam search on token sequences,
choose first token by minimizing 
𝐺
G,
append token, observe new state, replan.
Phase 3

Only after that, consider making the planner differentiable for training.

10) The exact mathematical object I would use first

For your first horizon planner, I would use:

𝐺
𝑡
(
𝜋
)
=
∑
𝑘
=
0
𝐻
−
1
[
∑
𝑣
∈
𝑉
𝑝
(
𝑣
∣
𝑥
𝑡
+
𝑘
+
1
(
𝜋
)
)
(
−
log
⁡
𝑝
\*
(
𝑣
)
)
+
𝜆
amb
𝐻
(
𝑝
(
⋅
∣
𝑥
𝑡
+
𝑘
+
1
(
𝜋
)
)
)
]
+
𝜆
term
∑
𝑣
∈
𝑉
𝑝
(
𝑣
∣
𝑥
𝑡
+
𝐻
(
𝜋
)
)
(
−
log
⁡
𝑝
\*
(
𝑣
)
)
.
G
t
	​

(π)=
k=0
∑
H−1
	​

[
v∈V
∑
	​

p(v∣x
t+k+1
(π)
	​

)(−logp
\*
(v))+λ
amb
	​

H(p(⋅∣x
t+k+1
(π)
	​

))]+λ
term
	​

v∈V
∑
	​

p(v∣x
t+H
(π)
	​

)(−logp
\*
(v)).

This gives you a clean finite-horizon risk + ambiguity planner using the exact token likelihood that your architecture already defines through PriorBank.

Then, once you have external observations in the horizon, upgrade it to:

𝐺
𝑡
(
𝜋
)
=
∑
𝑘
=
0
𝐻
−
1
(
R
i
s
k
𝑡
+
𝑘
(
𝜋
)
+
𝜆
amb
A
m
b
𝑡
+
𝑘
(
𝜋
)
−
𝜆
epi
E
p
i
𝑡
+
𝑘
(
𝜋
)
)
+
𝑉
term
.
G
t
	​

(π)=
k=0
∑
H−1
	​

(Risk
t+k
	​

(π)+λ
amb
	​

Amb
t+k
	​

(π)−λ
epi
	​

Epi
t+k
	​

(π))+V
term
	​

.

That is the mathematically proper sequence-policy version.

Bottom line

The horizon-
𝐻
H extension should be:

sequence of future tokens as policy
roll forward latent belief states under those candidate tokens
score each sequence by cumulative expected free energy
execute only the first token
replan after each step

And for your setting, the practical first move is:

use PriorBank.decode as the outcome model,
use beam-search over top-
𝐾
K token continuations,
use risk + ambiguity first,
reserve the full epistemic term for horizons that include genuine external observations.

The next thing to write is the exact beam-search pseudocode and the state-rollout interface against your current GaugeTransformerLM forward path.


