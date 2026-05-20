# Canonical Expected Free Energy for a Transformer Language Model

**Agent:** Investigation Agent 1 (lit/math).
**Scope:** Mathematical foundation for the active-inference build-out. The user's intuition that Friston's policies map to future strings of tokens is canonical [ParrPezzuloFriston2022 Ch. 7]; this document writes out what that identification commits one to, term by term.

The aim of this document is to fix the generative model, the recognition density, the canonical expected free energy `G(π)`, its multi-step recursion, the preference distribution recommendation, and the pitfalls the implementation must avoid. The companion documents in this directory specify the code layout (`03_architecture_design.md`), the compute budget (`04_compute_feasibility.md`), and the verifier's cross-checks (`05_verifier_report.md`).

## §1. Mapping between the active-inference generative model and the transformer LM

The discrete-state active inference of [ParrPezzuloFriston2022 Ch. 7] posits a generative model with hidden states `s_t`, observations `o_t`, actions `a_t`, and policies `π = (a_{T+1}, a_{T+2}, ..., a_{T+D})` that index sequences of actions over a planning horizon `D`. A recognition density `q(s | π)` approximates the posterior given a policy, and an outcome predictive `q(o | π) = E_{q(s|π)}[ p(o | s) ]` averages over latents. Specializing to a generative model whose latent factors live on the gauge-equivariant manifold used in `/vfe` yields the following identifications.

**Hidden state** at position `t` is the Gaussian belief tuple
```
s_t = (mu_t, sigma_t, phi_t),
```
where `mu_t in R^K` is the mean of `q(z_t)`, `sigma_t in R^K_+` is the per-coordinate variance (diagonal Gaussian), and `phi_t in gl(K)` is the Lie-algebra element parameterizing the gauge frame. The recognition density factorizes over positions in the standard mean-field form,
```
q(s_{1:N} | π) = prod_{t=1}^N q_t(mu_t, sigma_t, phi_t),                                (1)
```
which is the mean-field across tokens already used by `/vfe` [BleiKuckelbirgJordan2017 §3.2; canon §6]. Cross-position information is carried by the variational functional `F`, not by joint correlations in `q`.

**Observation** at position `t` is the realized token `o_t = v in V` from vocabulary `V` of size `|V|`. The likelihood is the PriorBank decode,
```
p(o_t = v | s_t) = softmax_v ( -KL( q_t || pi_v ) / tau ),                              (2)
```
which is the existing readout of `transformer/vfe/prior_bank.py:VFEPriorBank.decode` with the `use_prior_bank=True` path. Here `pi_v` is the PriorBank entry for token `v` and `tau` the readout temperature. Outside the PriorBank decode, a linear output projection from `R^K` to `R^|V|` is the only retained neural component allowed by the CLAUDE.md hard constraints.

**Action** at position `t+1` is the choice to emit a specific token,
```
a_t in A = V,                                                                            (3)
```
so the action space coincides with the vocabulary. The "select an action" operation is the commitment to write a specific symbol; "execute the action" is the appending of that symbol to the context and the re-running of the E-step over the extended sequence.

**Policy** of horizon `D` is the future token string
```
π = (a_{T+1}, a_{T+2}, ..., a_{T+D}) in V^D,                                             (4)
```
which is precisely the user's intuition. The space of policies has cardinality `|V|^D`; tractability for `|V| = 50257` and `D > 1` requires a tree expansion strategy (deferred to `03_architecture_design.md`).

**Transition kernel.** Per the standard process-theory formulation [FristonEtAl2017 §3], the transition is `p(s_{t+1} | s_t, a_t)`. In the LM specialization the transition is deterministic given the realized action,
```
s_{t+1} = E-step ( context || a_t ),                                                     (5)
```
i.e. once token `a_t` is appended to the context, the next state is whatever the variational E-step of `/vfe` returns when re-run on the extended sequence. The deterministic-given-action transition is a known special case of [ParrPezzuloFriston2022 Ch. 7]; it removes one of the two posteriors in the canonical EFE decomposition because `q(s_{t+1} | a_t, s_t)` becomes a delta.

**Preferences** `p^*(o | C)` are the agent's prior beliefs about what observations it should encounter. For an LM with no exogenous goal context `C`, three principled choices are available and §4 below recommends one as the default.

**Likelihood-versus-readout note.** The PriorBank decode in equation (2) is the agent's belief that the latent `s_t` will generate token `v`. The next-step prediction over the action `a_{t+1}` is logically a different object: it asks which token the agent's policy posterior will select. The two coincide when one identifies "next observation" with "next emitted token," because the LM is its own environment under self-play. The implementation must keep them notationally distinct.

## §2. Canonical single-step expected free energy

The standard variational free energy for a generative model `p(o, s)` admits three algebraically equivalent forms [Friston2010; canon §1],
```
F[q] = E_q[ log q(s) - log p(o, s) ]                                                     (6)
     = KL( q(s) || p(s | o) ) - log p(o)                                                 (7)
     = E_q[ -log p(o | s) ] + KL( q(s) || p(s) ),                                        (8)
```
which are Form 1 (variational definition), Form 2 (gap-plus-evidence), and Form 3 (accuracy-plus-complexity). The expected free energy of a policy `π` for a single action `a` (so `D = 1`) is the policy-conditioned analogue evaluated under the predictive `q(o, s | π)` [ParrPezzuloFriston2022 Ch. 2; FristonEtAl2017 §3],
```
G(π) = E_{q(o, s | π)} [ log q(s | π) - log p(o, s | π) ].                               (9)
```

This Form-1 expression rearranges into a Form-3 decomposition by replacing `p(o, s | π) = p(o | s) p(s | π)` and inserting a preference `p^*(o | C)` as the agent's prior over outcomes [FristonEtAl2017 Eq. 2; ParrPezzuloFriston2022 §2.4],
```
G(π) = E_{q(o | π)} [ -log p^*(o | C) ]                                                (10)
       - E_{q(o | π)} [ KL( q(s | o, π) || q(s | π) ) ].
```
The first term is the **pragmatic value** (expected cost under preferences) and the second is the **epistemic value** (expected information gain about latents given the action). The minus sign in front of the KL is the canonical sign convention: `G` is energy-to-minimize, so larger expected information gain reduces `G`. A third equivalent rearrangement, used in the codebase's `compute_efe`, expands the epistemic term using the chain rule on `q(s, o | π) = q(s | π) p(o | s)` and writes
```
G(π) = E_{q(o | π)} [ -log p^*(o | C) ]                                                (11)
       + E_{q(s | π)} [ H[ p(o | s) ] ]
       - I_{q(s, o | π)} ( s ; o ).
```
The first term is **risk**, the second is **ambiguity** (the expected entropy of the likelihood under the belief), and the third is the **BALD-style mutual information** [HoulsbyEtAl2011 Eq. 1; canonical for active learning] which equals the epistemic value above. The equality (10) = (11) holds because
```
E_{q(o|π)}[ KL( q(s | o, π) || q(s | π) ) ]  =  I_q(s ; o)
                                              =  H[ q(o | π) ] - E_{q(s|π)}[ H[p(o | s)] ].
```

**Specialization to the LM** with the mapping of §1. Equation (11) becomes
```
G(a) = E_{p(o | s_{t+1}(a))} [ -log p^*(o | C) ]                                       (12)
       + E_{q(s_{t+1} | a)} [ H[ p(o | s_{t+1}) ] ]
       - I_{q(s_{t+1}, o | a)} ( s_{t+1} ; o ),
```
where `s_{t+1}(a)` is the deterministic state that results from appending token `a` to the context and running the E-step. Under the diagonal-Gaussian recognition of `/vfe`, the expectations over `s_{t+1}` are Monte-Carlo estimates with samples `z_s = mu_{t+1} + sqrt(sigma_{t+1}) * eps_s`. The BALD MI in equation (12) is
```
I ( s ; o | a )  =  H[ p_bar(o | a) ] - (1/S) sum_s H[ p( o | z_s ) ],                 (13)
```
with `p_bar(o | a) = (1/S) sum_s p(o | z_s)`. This is exactly the quantity that `transformer/core/expected_free_energy.py:compute_epistemic_value` returns when called with `return_mean_H=True`, and `transformer/vfe/efe.py:VFEExpectedFreeEnergy._compute_epistemic_value` computes for the `/vfe` package. The implementation is canonical to the BALD form.

**Policy posterior.** The standard process-theory choice is the Gibbs softmin [FristonEtAl2017 §3; ParrPezzuloFriston2022 §2.4],
```
q(π) ∝ exp ( -gamma * G(π) ),                                                          (14)
```
with `gamma > 0` an inverse-temperature ("policy precision"). This is the sign convention used in `transformer/vfe/efe.py` and is consistent throughout the document: `G` decreases when the policy is good, the action posterior peaks on the minimum.

The decomposition (10) and the BALD form (11) are equivalent only under the Form-1 expression (9). A common pitfall is to write `G(π) = risk + ambiguity + entropy_of_marginal` instead of `G(π) = risk + ambiguity - MI`; the difference is exactly `BALD MI` and produces the `log |V|` cancellation under uniform preferences flagged in `transformer/core/expected_free_energy.py:118-126`. The build-out must use (11) as the operative form.

## §3. Sophisticated inference for multi-step policies

For horizons `D > 1` the policy posterior in equation (14) should weight each candidate sequence by the EFE summed over its steps, but a naive sum ignores that the agent will replan after observing each outcome. **Sophisticated inference** [Friston2021SophisticatedInference; ParrPezzuloFriston2022 Ch. 9] computes the EFE recursively through a planning tree, treating each step as a decision conditional on the (predicted) observations realized so far.

Let `G_d ( π_{1:d} | s_d )` denote the expected free energy of having committed to the partial policy `π_{1:d} = (a_1, ..., a_d)` reaching predicted state `s_d`. At a leaf depth `d = D`, the recursion bottoms out at the single-step EFE,
```
G_D ( π_{1:D} | s_{D-1} ) = E_{p(o_D | s_D(a_D))} [ -log p^*(o_D | C) ]                (15)
                            + E_{q(s_D | a_D)} [ H[ p(o_D | s_D) ] ]
                            - I ( s_D ; o_D | a_D ).
```
At an internal node `d < D`, the recursion bootstraps from the expected EFE of the next decision, anticipating that the agent will form a policy posterior over `a_{d+1}` after observing `o_d` [Friston2021SophisticatedInference Eq. 3; ParrPezzuloFriston2022 §9.3]:
```
G_d ( π_{1:d} | s_{d-1} ) = E_{p(o_d | s_d(a_d))} [ -log p^*(o_d | C) ]                (16)
                            + E_{q(s_d | a_d)} [ H[ p(o_d | s_d) ] ]
                            - I ( s_d ; o_d | a_d )
                            + E_{q(o_d | a_d)} [ E_{q(a_{d+1} | s_d, o_d)} [
                                  G_{d+1} ( π_{1:d}, a_{d+1} | s_d )
                            ] ],
```
with the child-action posterior at depth `d+1` itself derived from `G_{d+1}` via
```
q ( a_{d+1} | s_d, o_d ) ∝ exp ( -gamma * G_{d+1} ( π_{1:d}, a_{d+1} | s_d ) ).         (17)
```
Equation (16) is the canonical sophisticated-inference recursion of [Friston2021SophisticatedInference]: each step's contribution is the local risk-plus-ambiguity-minus-MI, and the future is summarized by the softmin-averaged EFE of the next decision, not by a sum over a fixed future sequence. The naive multi-step EFE `sum_{d=1}^D G_d` is recovered when one freezes the policy across the horizon (no replanning), which is the **deep-tree expansion** of [ParrPezzuloFriston2022 §7.4]. The two converge in the limit of deterministic outcomes and zero policy entropy, and diverge when the agent's predicted observations are uncertain.

**Recommendation for the build-out.** A naive policy-sum approximation,
```
G_naive ( π_{1:D} ) = sum_{d=1}^D G_d ( π_{1:d} | s_{d-1} ),                            (18)
```
combined with a beam search over the tree of candidate token strings, is the cheapest and most interpretable starting point. Full sophisticated-inference recursion with (16) and (17) is an extension that should be guarded behind a toggle and benchmarked against the beam-search baseline. The reason for this ordering is that the recursion in (16) requires expanding the action posterior at every internal node, which multiplies the per-token cost of the existing `compute_efe` by the branching factor at every depth; the compute analysis in `04_compute_feasibility.md` should quantify the tradeoff. Beam search with a width `B << |V|` and the depth-`D` policy-sum (18) preserves the single-action commit-and-replan semantics of [ParrPezzuloFriston2022 Ch. 7] at the price of a coarser tree.

A third option, retained for completeness, is a **single-step EFE with replanning**: commit at each step to the minimum-`G_1` action and re-evaluate at the next position. This degenerates the planning tree to depth 1 and is the existing behavior of `transformer/vfe/efe.py:VFEExpectedFreeEnergy.select_action`. It is the cheapest path and a fine baseline; the deeper expansions only earn their place if they measurably reduce policy regret on a held-out benchmark.

## §4. Preference distribution for a language model — recommendation

The pragmatic term `E_q[ -log p^*(o | C) ]` in equation (10) requires a preference distribution. For an LM with no exogenous goal `C`, three options have literature support and the build-out must pick one default.

The first option, **empirical training-data marginal**, sets
```
p^*(o = v) = freq_{train}(v),                                                           (19)
```
the unigram frequency of token `v` in the training corpus. Under this preference the pragmatic term becomes
```
E_{q(o | a)} [ -log p^*(o) ] = sum_v q(v | a) [ -log freq_{train}(v) ],                (20)
```
which is the cross-entropy of the model's next-token predictive against the corpus marginal. Two properties follow. First, when `q(o | a)` is treated as a Dirac at the observed training token `y`, the pragmatic term reduces exactly to the NLL `-log freq_{train}(y)`; combined with the complexity term in equation (8), one recovers the standard variational `F = NLL + KL(q || p)` of [Friston2010 Eq. A.4], so active inference and standard LM training share an objective at the bottom of the hierarchy. Second, the preference is independent of the agent's current belief, which closes the dark-room loophole [Friston2012Darkroom; canon §10 pitfall 4]: the empirical marginal has mass everywhere the training data does, so confidently emitting the same token to drive `H[p_pred]` to zero does not minimize the pragmatic term. This is the **recommended default**.

The second option, **low-entropy self-preference**,
```
p^*(o | s) ∝ exp ( -beta * H[ p(o | s) ] ),                                            (21)
```
implements the self-evidencing reading of [Hohwy2016 §2] under which the agent prefers outcomes that confirm its own predictions. This is the preference that the existing `transformer/vfe/active_inference.py` implicitly assumed, and the verdict at `docs/debates/2026-05-19-vfe-active-inference-impl/04_verdict.md` flagged the substitution as non-canonical because it removes the exogenous `C` argument from `p^*(o | C)`. The choice is internally consistent under Hohwy's reading but exposes the dark-room failure mode: the BALD MI in equation (11) is the sole counterweight, and at the existing config's `lambda_prag = 1.0, lambda_epi = 0.5` it is structurally underweighted (verdict §2). Retain as a configurable option labeled `self_evidencing`, not as the default.

The third option, **task-conditioned preference**,
```
p^*(o | C) = p_task(o | C),                                                            (22)
```
treats `C` as an externally supplied goal — a reward model, a constraint set, a target document, or an RLHF preference distribution [Christiano2017RLHF; Bai2022ConstitutionalAI; citation pending verification for the explicit AIF–RLHF connection]. This is the natural choice when the LM is deployed for goal-directed behavior. The implementation should expose it as a configurable preference (an injected `p^*` tensor) but not as the default for unconditional generation.

**Default for the build-out:** option one, the empirical training-data marginal. The justification is that it (a) makes the pragmatic term well-defined without an exogenous goal, (b) reduces to standard NLL when evaluated at observed training tokens, (c) closes the dark-room failure mode by anchoring `p^*` to data rather than to the model's own predictions, and (d) sits at the canonical Form-3 decomposition of `F` in equation (8) — minimizing F under this preference is minimizing accuracy-plus-complexity in the [Friston2010] sense. Options two and three are exposed as configurable alternatives with explicit labels (`self_evidencing` and `task_conditioned`).

## §5. Pitfalls the build-out must avoid

The canon pitfall list at `external_canon_inference.md` §10 enumerates ten failure modes; six apply directly to this build-out and one further failure mode is specific to LMs.

**Form-1 / Form-2 / Form-3 conflation** [canon §10 pitfall 1]. The decomposition (10) writes `G` in the canonical Form 3 of [ParrPezzuloFriston2022 §2.4] (pragmatic + epistemic). The BALD form (11) is also Form 3 with the epistemic value re-expressed as ambiguity minus MI; both are admissible but they are not the same as the variational Form-1 expression (9), which uses the joint posterior `q(s, o | π)` and does not split the latent and observation entropies. The implementation must label each term by which form it lives in and must not mix them in a single equation.

**Sign convention drift between training and generation** [canon §10 pitfall 2]. Standard VFE training minimizes `F` (gradient descent on `F`, ascent on `-F = ELBO`). Standard active inference minimizes `G` and forms `q(π) ∝ exp(-gamma G)`. Both signs agree: smaller is better. The existing `vfe/efe.py` uses `exp(-gamma G)`; this convention must persist into any training-time use of the EFE-augmented objective. The dangerous failure mode is multiplying by `+gamma` somewhere in a multi-step recursion and silently inverting the policy posterior; the verifier in `05_verifier_report.md` should include a finite-difference sign test on the leaf EFE.

**Overclaims that the FEP implies architectural choices** [canon §10 pitfall 3]. The FEP is a variational principle; specific implementations follow from specific generative-model choices, not from FEP alone. Claims of the form "active inference implies sophisticated tree search" or "active inference implies beam search over policies" must be unpacked: the tree search follows from (i) the choice to plan with horizon `D > 1` and (ii) the cardinality `|V| = 50257` of the action space; the FEP supplies the cost `G(π)` over policies, not the search algorithm. The plan documents must be careful with the rhetoric here.

**Dark-room failure mode** [Friston2012Darkroom; canon §10 pitfall 4]. The pragmatic term alone is minimized by an agent that confidently emits the most-preferred token regardless of context. The BALD MI is the canonical counterweight, but at low `gamma * epistemic_weight` the agent collapses to mode-greedy generation. The recommended default of §4 (empirical marginal) reduces the dark-room exposure because the preference has support everywhere; the `self_evidencing` option does not have this property and must be paired with a non-trivial `epistemic_weight`.

**Hierarchical mean-field versus point-passing** [canon §10 pitfall 5]. The `/vfe` cascade passes the previous layer's posterior mean `mu_q` as the next layer's prior mean `mu_p`, with `sigma_prior` reset to the embedding scale. This is a deterministic point estimate at each level, not a full variational hierarchical scheme [Friston2017Graphical; canon §3]. The build-out's planning tree inherits the same structure: the predicted `s_d` at internal nodes is the deterministic E-step output, not a posterior over `s_d`. This is a defensible approximation but must be labeled as such; the document must not claim the planning tree is "exact Bayesian rollout."

**Variational ≠ Bayesian** [canon §10 pitfall 7]. The recognition density `q` approximates the true posterior; the approximation gap is the KL in Form 2 (equation 7). Any claim that the EFE module implements "exact Bayesian planning" is wrong; it implements variational planning with the diagonal-Gaussian mean-field of equation (1).

**LM-specific pitfall — action–observation degeneracy.** In the LM the action `a_t` and the next observation `o_{t+1}` are typographically the same object: both are tokens. This is a special structural feature of self-played language modeling and does not generalize to active inference in environments where actions and observations are different objects. The implementation must distinguish (a) the action `a_t` that the agent commits to and that drives the deterministic transition, from (b) the predicted observation `o_{t+1}` that the agent expects to see after that commitment. In a pure-LM setting the two collapse, so `q(o_{t+1} | a_t)` becomes a delta on `a_t` and the ambiguity term `E_{q(s|a)}[H[p(o|s)]]` becomes the entropy of the agent's own next-token readout. This collapse is the technical content of the LM specialization and should be derived explicitly in the manuscript appendix rather than assumed.

**Surrogate-versus-canonical labeling.** Per the verdict, the path with `p^*(o | s) ∝ exp(-beta H[p_pred])` is a self-evidencing surrogate and not canonical EFE. The build-out must keep this path under its `self_evidencing` label and not advertise it as active inference at the user-facing surface; the verifier should grep for "active inference" in docstrings under that branch and flag.

## §6. Citations

The load-bearing references for this document are the active-inference textbook and the original EFE papers; supporting references cover variational inference, mean-field structure, the IFT for fixed points, and BALD MI.

The canonical statement of variational free energy in equations (6)–(8) is [Friston2010]. The active-inference process theory with the policy posterior of equation (14) is [FristonEtAl2017]. The textbook treatment of discrete-state active inference with policies as action sequences (equation (4)) and the canonical EFE decomposition (10)–(11) is [ParrPezzuloFriston2022], specifically Chapter 2 for the EFE and Chapters 7 and 9 for finite-horizon planning with sophisticated inference. The sophisticated-inference recursion in equations (15)–(17) is the central construction of [Friston2021SophisticatedInference: Friston, K., Da Costa, L., Hafner, D., Hesp, C., Parr, T. "Sophisticated Inference." Neural Computation 33(3): 713–763, 2021 — full citation needed; verifier please confirm volume/page]. The mean-field decomposition in equation (1) and the canonical KL convention `KL(q || p)` follow [BleiKuckelbirgJordan2017]. The BALD mutual information of equation (13) is [HoulsbyEtAl2011: Houlsby, N., Huszár, F., Ghahramani, Z., Lengyel, M. "Bayesian Active Learning for Classification and Preference Learning." arXiv:1112.5745, 2011 — full citation needed; verifier please confirm]. The dark-room failure mode is [Friston2012Darkroom: Friston, K., Thornton, C., Clark, A. "Free-energy minimization and the dark-room problem." Frontiers in Psychology 3:130, 2012 — full citation needed]. The Hohwy self-evidencing reading is [Hohwy2016: Hohwy, J. "The Self-Evidencing Brain." Noûs 50(2): 259–285, 2016 — full citation needed]. The RLHF connection mentioned for the task-conditioned preference of equation (22) is [Christiano2017RLHF: Christiano, P. et al. "Deep Reinforcement Learning from Human Preferences." NeurIPS 2017 — full citation needed] and [Bai2022ConstitutionalAI: Bai, Y. et al. "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073, 2022 — full citation needed]; the explicit identification of RLHF preferences with `p^*(o | C)` in the EFE has not been verified to a primary source and is offered as the natural reading rather than a settled correspondence.

The Form-3 identity `F = accuracy + complexity` and its relation to NLL in §4 is [Friston2010 Eq. A.4]; the verifier should confirm the appendix-A numbering against a retrievable copy of the paper. The IFT background that the `em_mode='ift_phi'` path of `/vfe` claims is [BaiKolterKoltun2019]; while not directly cited in the equations above, it is the reference for any claim that gradients through the E-step fixed point are computed via the implicit-function theorem rather than amortized backprop.

All section pointers (e.g., `[ParrPezzuloFriston2022 §2.4]`) are best-effort and should be verified by the verifier agent against the actual text before they appear in the manuscript appendix that the verdict's action item 3 requires. Tags marked `[full citation needed]` carry an unverified bibliographic record that the verifier must fill in or downgrade to source-level citation.
