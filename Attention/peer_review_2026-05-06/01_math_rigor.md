# Peer Review: Mathematical and Theoretical Rigor

**Manuscript:** `Attention/Participatory_it_from_bit.tex` (3,738 lines)
**Reviewer scope:** Geometric foundations (§2), free energy functional, mass matrix, appendices.
**Date:** 2026-05-06

## Summary verdict

The mathematical scaffolding hangs together: the bundle/section/connection
definitions are formally consistent, the GL(K) gauge invariance theorem
(Thm.~1) is correct as stated, the vanishing-holonomy theorem
(Thm.~`thm:vanishing_holonomy`) is a clean algebraic observation, and the
sandwich law for covariance transport `Sigma -> Omega Sigma Omega^T` is used
consistently throughout the first-variation calculus. The manuscript is also
unusually self-aware about the difference between derivation and analogy: the
mass-as-precision identification is openly labelled "interpretive within the
framework", the explicit-vs-spontaneous symmetry-breaking distinction is
named correctly, the horizontal-block gauge-invariance disclosure for the
pullback metric is honest, and the mixture-of-sources construction is
explicitly downgraded from "derivation" to "engineered consensus-energy
ansatz" with a status paragraph at line 604.

That said, two real bugs remain in the first-variation/mass-matrix block, the
"exact for Gaussians" appendix oversells what is actually proved, and the
conditional-uniqueness theorem is borderline tautological in a way that the
main-text framing does not surface. None are fatal; all are local. **Recommendation: minor revision.**

---

## Major issues

### M1. Sender covariance gradient is missing the quadratic-term contribution; valid only at consensus.
**Location:** `Participatory_it_from_bit.tex:1153`

Quoted:
> `\partial D_{\mathrm{KL}}(q_i \| \tilde{q}_k)/\partial \Sigma_k = (1/2)\Omega_{ik}^T[\tilde{\Lambda}_{q_k} - \tilde{\Lambda}_{q_k}\Sigma_i\tilde{\Lambda}_{q_k}]\Omega_{ik}`

This expression captures the log-det and the trace contributions but drops
the derivative of the Mahalanobis term `(\mu_i - \tilde\mu_k)^T \tilde\Sigma_k^{-1}(\mu_i - \tilde\mu_k)`
through `\tilde\Sigma_k^{-1}`. With `S := \Omega_{ik}\Sigma_k\Omega_{ik}^T` and
`\Delta := \mu_i - \tilde\mu_k`, the chain rule via `dS^{-1} = -S^{-1}dS\,S^{-1}`
gives the additional term

```
- (1/2) \Omega_{ik}^T \tilde\Lambda_{q_k} \Delta \Delta^T \tilde\Lambda_{q_k} \Omega_{ik}.
```

This vanishes only at consensus where `\mu_i = \tilde\mu_k`. The "First
Variations" section presents this gradient unconditionally, before the
Hessian section announces a consensus-evaluation convention at line 1182. As
written, the expression at line 1153 is incorrect off-consensus. Either (a)
add the quadratic-term derivative explicitly, or (b) move the
"at consensus" qualifier from line 1182 up to the first-variation section so
that both the unconditional gradient and the conditional one are clearly
flagged.

This same gap propagates to the off-diagonal covariance Hessian block at
line 1235 (Eq.~`eq:mass_sigma_offdiagonal`), which is also derived assuming
consensus but is boxed without that qualifier on the box itself; the caveat
appears only in the surrounding paragraph. Boxed equations are read as canonical;
recommend annotating the box explicitly with "(at consensus)".

### M2. Section title "Exact expansion for Gaussian beliefs" oversells what is proved.
**Location:** `Participatory_it_from_bit.tex:3473` (subsection header)
and `Participatory_it_from_bit.tex:3517-3522` (regime caveat).

Quoted at line 3473:
> `\subsection{Exact expansion for Gaussian beliefs}`

Quoted at line 3520:
> "When beliefs are approximately aligned (the regime enforced by the coupling itself),
> the covariances satisfy $\Sigma_i \approx \Omega_{ij}\Sigma_j\Omega_{ij}^\top$, and the
> trace and log-determinant terms approximately cancel. In this alignment regime, ...
> the quadratic expectation becomes $\frac{1}{4}\mathbb{E}_{q_iq_j}[...] \approx \frac{\tau^{(q)}_{ij}}{2}D_{\mathrm{KL}}(q_i\|\Omega_{ij}q_j) + \text{const}$."

Two distinct claims are conflated. The quadratic-form expectation
identity (Eq.~`eq:belief_quadratic_expansion`) is exact - it is just the
standard `E[x^T A x] = tr(A Cov(x)) + (E x)^T A (E x)` applied to a Gaussian
difference. The identification of this expectation with the KL divergence,
however, is **approximate** and requires the alignment assumption
`\Sigma_i \approx \Omega_{ij}\Sigma_j\Omega_{ij}^T` so that the trace and
log-determinant terms cancel. The section title and the framing at line 3471
("we now show that the pairwise quadratic expectations can be expressed in
terms of KL divergences") suggest exactness; the body of the section
delivers an approximate identification.

This matters because the boxed Eq.~`eq:beta_weighted_precision` and the
consensus simplification of Eq.~`eq:mass_sigma_diagonal` rely on the
alignment regime, but the title-level "exact" framing leads to derivation
language that obscures the regime dependency. Recommend retitling to "Quadratic-form
expansion and KL identification in the alignment regime", and adding a single
line at the end of the section explicitly stating which regime each line is
exact in.

### M3. Conditional Uniqueness Theorem is borderline tautological; the framing oversells.
**Location:** `Participatory_it_from_bit.tex:3573-3661` (App. C.6, `app:conditional_uniqueness`).

Quoted at line 3573:
> "the forward KL divergence used throughout the main text is the unique divergence
> consistent with three jointly required properties: closed-form Gibbs-type belief
> updates, dual interpretation of the attention weights via the envelope theorem,
> and linear coupling preserving the softmax form."

Quoted at line 3596:
> "The uniqueness of the term is conditional and follows from three assumptions:
> (i) $\mathcal{D}$ is local in $c$, of f-divergence form ...; (ii) the coupling is
> linear; and (iii) the minimizing belief $q_i^*$ remains in the exponential-family (log-linear)."

Assumption (iii) ("the minimizing belief remains in the exponential family /
log-linear") is essentially the same content as the conclusion target of the
proof, namely that the stationarity condition produces the geometric-mean
Boltzmann form `q_i^*(c) \propto p_i(c)^{1/2}\prod_j[\Omega_{ij}q_j(c)]^{\beta_{ij}/2}`.
The reverse implication (Step 3, line 3637) demands that the stationary form
be log-linear in `\log q_i, \log p_i, \log(\Omega_{ij}q_j)` for *every* prior
and neighbour configuration; assumption (iii) imposes exactly that.

As a result, the theorem is best read as: "if we *demand* exponential-family
closure of the minimiser for all priors and neighbours, then within the
f-divergence class, KL is unique up to affine equivalence." That is a real
result with a clean proof, but it is not the "unique f-divergence consistent
with the envelope-theorem dual interpretation" framing of the
introduction. The envelope theorem in this manuscript holds for any
f-divergence (it is a property of the variational problem, not of the
divergence). What is unique to KL is exponential-family closure, which the
authors essentially assume. Recommend rewriting the introduction to App. C.6
to state this honestly: KL is the unique f-divergence preserving
exponential-family closure under linear coupling, and the dual interpretation
follows from envelope but is not the discriminator.

This is also relevant to the main-text framing at line 815:
> "the forward KL divergence $D_{\mathrm{KL}}(q_i \| \Omega_{ij}q_j)$ is the unique
> f-divergence that preserves exponential-family closure under linear coupling and
> yields a consistent dual interpretation for the attention weights."

The "yields a consistent dual interpretation" clause is what is overselling -
it suggests dual consistency is the discriminator when in fact it is closure.

### M4. Stability claim in App. C.3 is supported by only one Hessian sector; full coupled-system stability is not established.
**Location:** `Participatory_it_from_bit.tex:3464-3470`.

Quoted at line 3463:
> "Local stability of the equilibrium follows from the positive-definiteness of the
> Hessian. For the Gaussian KL terms, $\partial^2 D_{\mathrm{KL}}/\partial\Sigma_1\partial\Sigma_1 \sim \Sigma_1^{-1}\otimes\Sigma_1^{-1}$,
> which is manifestly positive definite for any $\Sigma_1\succ 0$. Hence the covariance
> alignment fixed-point is an attractor of the variational dynamics."

This argument shows positive-definiteness of *one* diagonal-block contribution
to the per-agent Hessian. It is silent about (a) the off-diagonal blocks
between agents (which carry sign in
Eq.~`eq:mass_sigma_offdiagonal`), (b) the contribution of the
$\beta_{ij}$-softmax dependence on $q$ (which the manuscript itself flags as
present off-equilibrium at line 1182), and (c) the simultaneous mean-sector
Hessian, which is necessary for joint $(\mu, \Sigma)$ stability. The boxed
fixed-point Eq.~`eq:sigma_fixed_point_beta` is a multi-agent coupled
system; per-agent positive-definiteness of one term is not sufficient to
conclude that the coupled system has a stable attractor.

Recommend either downgrading "is an attractor of the variational dynamics" to
"is a critical point of the per-agent free energy and the per-agent
covariance Hessian is positive definite at this point", or supplying a
separate analysis of the full coupled Hessian under the symmetric-attention,
small-mismatch assumptions of the homogeneous limit (which would be
straightforward in that special case).

---

## Minor issues

### m1. Cross-scale frame averaging is not canonical for non-abelian G.
**Location:** `Participatory_it_from_bit.tex:456`.

Quoted:
> "$\phi_I^{(s+1)}(x) = \sum_{i \in I} w_i(x) \phi_i^{(s)}(x) / \sum_{i \in I} w_i(x)$"

This is a Lie-algebra weighted average. For non-abelian G,
`exp(\sum w_i \phi_i)` is *not* a geometric/Karcher mean of `\{exp(\phi_i)\}`,
nor is it gauge-equivariant under right-translation in any frame-canonical way.
The choice is well-defined as a function of the constituent frames, but
arbitrary among many candidates (Karcher mean, log-Euclidean mean,
left/right-invariant means). The manuscript does not justify the choice or
note its non-canonical status. A sentence acknowledging that this is a
modelling choice (and that an alternative such as the Karcher mean would
yield a different but also reasonable construction) would be appropriate.

### m2. "Differentiation convention" boilerplate is correct but should appear earlier.
**Location:** `Participatory_it_from_bit.tex:1182`.

The convention switch from autograd-style first variations (line 1133-1162)
to envelope-theorem-style second variations (line 1185 onward) is announced
in a paragraph that comes *after* the first-variation section that uses the
opposite convention. Reordering would prevent confusion: state the convention
hierarchy once at the top of §3 (Mass from Statistical Precision), then apply
it.

### m3. The boxed identity `\Omega_{ji}^T \tilde{\Lambda}_{q_i}^{(j)} \Omega_{ji} = \Lambda_{q_i}` is correctly characterised as GL-valid, but the sender second derivative at line 1187 invokes it implicitly without a forward pointer.
**Location:** `Participatory_it_from_bit.tex:1187` ("From consensus as sender to agent $j$").

The chain "$\partial^2 D_{\mathrm{KL}}(q_j \| \tilde{q}_i)/\partial \mu_i \partial \mu_i^T = \Omega_{ji}^T \tilde{\Lambda}_{q_i}^{(j)} \Omega_{ji} = \Lambda_{q_i}$"
relies on the identity stated three pages earlier at line 1112. A
back-reference (`\eqref{eq:precision_transport}`) would help.

### m4. Line 1207 caveat applies to an asymmetric attention regime that is then routinely assumed elsewhere.
**Location:** `Participatory_it_from_bit.tex:1207-1209`.

Quoted:
> "The mass matrix $M_{\mu\mu}$ is symmetric only when $\beta_{ik} = \beta_{ki}$
> (reciprocal attention) and $\Omega_{ik} = \Omega_{ki}^\top$ (reciprocal gauge transport)."

The manuscript notes this caveat clearly, but at line 1259-1267 the "mass as
precision" reading then applies the symmetric form as if the caveat does not
matter. The Newtonian-analogy reading is restricted to the symmetric regime;
the paragraph at line 1207 should be cross-referenced from the
mass-as-precision section so the reader does not import the latter into the
asymmetric regime where the kinetic-energy interpretation fails.

### m5. The "Goldstone parallel" concession at line 1003 is welcome, but the figure caption at line 1011 still reads as if a derivation has occurred.
**Location:** `Participatory_it_from_bit.tex:1011`.

Quoted:
> "The breaking is explicit (analogous to a Zeeman term) rather than spontaneous: ..."

The body text at line 1003 correctly characterises the symmetry breaking as
explicit and disclaims the Goldstone-mode derivation. The figure caption
already echoes this. Good. No change needed - flagged here only as a positive
example of the discipline elsewhere recommended.

### m6. "Pure gauge" claim at line 778 holds in the manuscript's parameterisation but should note the global topology assumption.
**Location:** `Participatory_it_from_bit.tex:778`.

Quoted:
> "For connections derived from a single-valued gauge function $\phi_i(c)$, the connection
> is pure gauge and the curvature vanishes identically: $F_{\mu\nu}^{(i)} = 0$."

This is correct but assumes `\phi_i: \mathcal{U}_i \to \mathfrak{g}` is
*globally* single-valued on `\mathcal{U}_i`. On a non-simply-connected base
(e.g., the periodic-boundary simulations mentioned at line 552), this is a
non-trivial assumption: a single-valued $\phi$ on $\mathbb{R}^2$ that survives
periodic identification need not exist for arbitrary boundary data. The
manuscript already commits to flat $\mathcal{C}$ which sidesteps this for the
simulations. A one-sentence acknowledgement that "single-valued" requires the
domain to admit a global section of the principal bundle would be appropriate
since the manuscript frames the cocycle as a theorem of the architecture.

### m7. Reference Eq.~`eq:cross_scale_shadow` is invoked many times before it is defined.
**Location:** Used at lines 224, 271, 295, 308, 478 (and many more) before any
labelled `eq:cross_scale_shadow` definition I could locate.

A search for `\label{eq:cross_scale_shadow}` should confirm the definition
sits in the meta-agent / participatory section. The manuscript should
forward-reference its first usage explicitly (e.g., "see
Eq.~`\eqref{eq:cross_scale_shadow}` in Section X for the definition") rather
than allowing the reader to encounter the symbol without source.

### m8. Sign convention in Killing-form metric.
**Location:** `Participatory_it_from_bit.tex:1276`.

Quoted:
> "$\langle \dot{\phi}, \dot{\phi} \rangle_{\mathfrak{g}} = -\mathrm{tr}(\dot{\phi}^2)$"

For `\dot{\phi} \in \mathfrak{so}(3)` (real antisymmetric matrix),
`\mathrm{tr}(\dot{\phi}^2) \le 0` so the minus sign correctly produces a
non-negative metric. For general `\mathfrak{gl}(K)` (which the body of the
paper extends to), `\mathrm{tr}(\dot{\phi}^2)` is sign-indefinite and the
expression is not a metric. The Killing form on `\mathfrak{gl}(K)` is
`B(X,Y) = 2K \mathrm{tr}(XY) - 2\mathrm{tr}(X)\mathrm{tr}(Y)` (i.e., it is not
just `-\mathrm{tr}(X^2)`), and is degenerate on the centre. The expression at
line 1276 is therefore correct only for `\mathfrak{so}(N)`. Either restrict
the formula to that case explicitly or replace with an unambiguous
positive-definite trace form like `\mathrm{tr}(\dot\phi^T \dot\phi)`.

### m9. Mahalanobis-on-rotated-mu issue with RoPE-style operators is not flagged.
**Location:** General comment, no specific line.

The project's own CLAUDE.md flags a known gap when RoPE rotates `\mu` but not
`\sigma`. The manuscript inherits this mismatch class structurally: the
mean-sector gradient at line 1146 uses `\tilde\Lambda_{q_k}(\mu_i - \tilde\mu_k)`
under a transport that is the same for both `\mu` and `\Sigma`. This is fine
for the present paper. But if the framework is extended to operators that
transport `\mu` differently from `\Sigma` (which the companion paper does),
the formal symmetry of `Omega Sigma Omega^T` and `Omega mu` is lost and the
Mahalanobis term picks up a frame mismatch. A note in the working-framework
simplifications section (around line 480-565) saying "we maintain a single
transport `\Omega_{ij}` for both `\mu` and `\Sigma` throughout this paper;
operator pairs that transport mean and covariance independently are out of
scope" would forestall confusion.

### m10. The "vacuum / gauge orbit" claim at line 996 is stronger than the proof supports.
**Location:** `Participatory_it_from_bit.tex:996`.

Quoted:
> "all agent beliefs $\mu_i(c)$ converge to states with identical norms despite
> occupying distinct coordinates in the $2\ell_q + 1 = 19$ dimensional fiber. ...
> This behavior reveals that agents occupy a gauge orbit - a sub-manifold of states
> related by SO(3) transformations."

Identical-norm vacuum *is consistent with* a gauge-orbit interpretation but
does not establish it: any rotation-invariant attractor (e.g., concentration
on an SO(3)-symmetric subset that is not a single orbit) would also produce
this. The claim is empirically supported by the figure but should be hedged
("consistent with agents occupying a single gauge orbit") rather than
asserted ("reveals that agents occupy a gauge orbit"). Minor wording.

---

## Derivation cross-checks

### CC1. GL(K) gauge invariance (Thm.~1, line 510-549).
Re-derived all three KL components under `\Omega_*` pushforward. Trace term
uses `(\Omega \Sigma_Q \Omega^T)^{-1} = \Omega^{-T}\Sigma_Q^{-1}\Omega^{-1}`,
which is correct. Quadratic and log-det terms are correctly handled.
**Verdict: clean.**

### CC2. Vanishing-holonomy theorem (line 778-790).
The cocycle `\Omega_{ij}\Omega_{jk} = U_i U_j^{-1} U_j U_k^{-1} = U_i U_k^{-1} = \Omega_{ik}`
relies only on `U_j^{-1} U_j = I`, which is algebraically valid in any group.
The note at line 466-475 that this works for non-commutative groups *because*
of the vertex-local parameterisation is correct. **Verdict: clean. The
flatness is genuine, not an approximation.**

### CC3. Mass diagonal block at line 1196 (`eq:mass_mu_diagonal`).
Verified term by term:
- prior: `\bar\Lambda_{p_i}` (correct, second derivative of quadratic term in `\mu_i`)
- consensus as receiver: `\sum_k \beta_{ik}\tilde\Lambda_{q_k}` (correct)
- consensus as sender: `\sum_j \beta_{ji} \Omega_{ji}^T \tilde\Lambda_{q_i}^{(j)} \Omega_{ji} = \sum_j \beta_{ji}\Lambda_{q_i}` using Eq.~`eq:precision_transport`. The simplification holds for any `\Omega \in \mathrm{GL}(d)` (algebraic, not requiring orthogonality), as the manuscript correctly notes.
- sensory: `\Lambda_{o_i}` (correct).
**Verdict: clean.**

### CC4. Sender Σ_k gradient (line 1153).
Re-derived from `D_{\mathrm{KL}}(q_i \| \tilde q_k)` with `\tilde S = \Omega_{ik}\Sigma_k\Omega_{ik}^T`
and `\tilde S^{-1} = \Omega_{ik}^{-T}\Lambda_k\Omega_{ik}^{-1}`:
- log-det: `+(1/2) \Sigma_k^{-1} = +(1/2) \Omega_{ik}^T \tilde\Lambda_k \Omega_{ik}` (using
the boxed identity).
- trace: `-(1/2) \Omega_{ik}^T \tilde\Lambda_k \Sigma_i \tilde\Lambda_k \Omega_{ik}`.
- Mahalanobis (depends on `\tilde S^{-1}` only): `-(1/2)\Omega_{ik}^T \tilde\Lambda_k (\mu_i-\tilde\mu_k)(\mu_i-\tilde\mu_k)^T \tilde\Lambda_k \Omega_{ik}`.

The manuscript drops the third term. **Verdict: incomplete off-consensus
(see M1).** Boxed expression is correct only at consensus.

### CC5. Geometric-mean Boltzmann fixed point (App. C.6).
Stationarity condition with `f(t) = t\log t - t + 1` gives `f'(t) = \log t`.
Substituting into the general stationarity Eq.~`eq:general_stationarity_app`:
`\log q_i - \log p_i + 1 + \sum_j \beta_{ij}(\log q_i - \log\Omega_{ij}q_j) = \lambda`.
With `\sum_j \beta_{ij} = 1` this gives `2\log q_i = \log p_i + \sum_j \beta_{ij}\log(\Omega_{ij}q_j) + (\lambda - 1)`,
matching Eq.~`eq:geometric_mean_target_app`. **Verdict: clean. Reverse implication
(line 3637) is also correctly carried, but its content is closer to "log-linear
closure forces KL" than to "envelope theorem forces KL" — see M3.**

### CC6. Off-diagonal mass mean block (line 1199, `eq:mass_mu_offdiagonal`).
Re-derived:
- From `\partial^2 D_{\mathrm{KL}}(q_i \| \tilde q_k)/\partial \mu_i \partial \mu_k`:
the gradient w.r.t. `\mu_i` is `\tilde\Lambda_k(\mu_i - \Omega_{ik}\mu_k)`. Differentiating
w.r.t. `\mu_k` gives `-\tilde\Lambda_k \Omega_{ik}`. Using `\tilde\Lambda_k\Omega_{ik} = \Omega_{ik}^{-T}\Lambda_k\Omega_{ik}^{-1}\Omega_{ik} = \Omega_{ik}^{-T}\Lambda_k`. Correct.
- From `\partial^2 D_{\mathrm{KL}}(q_k \| \tilde q_i)/\partial \mu_i \partial \mu_k`:
gradient w.r.t. `\mu_k` is `\tilde\Lambda_i^{(k)}(\mu_k - \Omega_{ki}\mu_i)`. Differentiating w.r.t. `\mu_i` gives `-\tilde\Lambda_i^{(k)}\Omega_{ki} = -\Omega_{ki}^{-T}\Lambda_{q_i}\Omega_{ki}^{-1}\Omega_{ki} = -\Omega_{ki}^{-T}\Lambda_{q_i}`. The manuscript writes `-\Lambda_{q_i}\Omega_{ki}^{-1}` instead. These are equal only when `\Lambda_{q_i}` and `\Omega_{ki}` satisfy `\Omega_{ki}^{-T}\Lambda_{q_i} = \Lambda_{q_i}\Omega_{ki}^{-1}`, i.e., `\Lambda_{q_i}\Omega_{ki}^{-1} = \Omega_{ki}^{-T}\Lambda_{q_i}` ⟹ `\Omega_{ki}^T \Lambda_{q_i} \Omega_{ki}^{-1} = \Lambda_{q_i}` ... which **is not** an identity unless `\Omega_{ki}` commutes with `\Lambda_{q_i}` or is orthogonal.

Re-checking: the gradient of `D_{\mathrm{KL}}(q_k \| \tilde q_i)` with respect to
`\mu_k` is `\tilde\Lambda_i^{(k)}(\mu_k - \Omega_{ki}\mu_i) = \Omega_{ki}^{-T}\Lambda_{q_i}\Omega_{ki}^{-1}(\mu_k - \Omega_{ki}\mu_i)`.
Differentiating w.r.t. `\mu_i` (which appears via `-\Omega_{ki}\mu_i`):
`-\Omega_{ki}^{-T}\Lambda_{q_i}\Omega_{ki}^{-1}\Omega_{ki} = -\Omega_{ki}^{-T}\Lambda_{q_i}`.

So the correct expression is `-\beta_{ki}\Omega_{ki}^{-T}\Lambda_{q_i}`, not
`-\beta_{ki}\Lambda_{q_i}\Omega_{ki}^{-1}`. **This appears to be a minor sign/transposition
error that I want to flag tentatively** - it is possible I have the convention for
"derivative with respect to a matrix in a left-acting vs right-acting product"
confused. The form `\Lambda_{q_i}\Omega_{ki}^{-1}` would arise if we treat the
derivative as a row vector acting from the left. The manuscript's first-variation
sender mean gradient at line 1152 uses `\Lambda_{q_k}\Omega_{ik}^{-1}(\tilde\mu_k - \mu_i)`,
which has the same structural form. If the convention is consistent, the mass-block
expression is consistent with it; if the convention is "gradient w.r.t. column vector"
then the second expression should be `\Omega_{ki}^{-T}\Lambda_{q_i}`. **I am not
fully confident this is a bug** — flag for the authors to clarify the gradient
convention used (column vs row gradient).

### CC7. Alignment-dominated regime / homogeneous limit boundary (App. C.3).
In the homogeneous limit `\Sigma_i = \Sigma_\infty`, `\Omega_{ij}\approx I`,
`\Sigma_{p,i} = \Sigma_0`, the fixed point gives `\Sigma_\infty = \Sigma_0`
(line 3398). In the alignment-dominated limit `\tau\to 0`, the boxed
Eq.~`eq:beta_weighted_precision` becomes
`\Sigma_i^{-1} \approx \langle (\Omega_{ij}\Sigma_j\Omega_{ij}^T)^{-1}\rangle_\beta`.
At the boundary where both regimes are imposed simultaneously
(`\Omega_{ij}\approx I`, `\Sigma_i \approx \Sigma_j`), the expression collapses
to `\Sigma_i^{-1} \approx \Sigma_j^{-1}` which is an alignment statement, but
it does not match the homogeneous limit answer `\Sigma_\infty = \Sigma_0`
because the alignment-dominated regime drops the prior contribution. The
mismatch is consistent with the regime caveats stated, but the manuscript
does not point out the boundary mismatch explicitly. A sentence noting that
the two limits are *not* compatible — the homogeneous limit retains the
prior, the alignment-dominated limit discards it — would clarify.

---

## Open questions for the authors

1. **(Re M1)** At what point in the manuscript does the "consensus evaluation"
   convention begin? The first-variation section reads as unconditional; the
   second-variation section announces the consensus convention. Is the
   first-variation expression at line 1153 intended to be evaluated only at
   consensus? If so, please mark it; if not, please add the missing
   quadratic-term gradient.

2. **(Re M3)** The conditional-uniqueness theorem proof (line 3633-3661) is
   clean, but the *premise* assumption (iii) "minimizing belief stays in
   exponential family" is in fact equivalent to the conclusion. Is the intended
   theorem statement "KL is the unique f-divergence preserving exponential-family
   closure under linear coupling and uniform-`\sum_j\beta_{ij} = 1`"? If so,
   please rephrase the introduction to drop the "consistent dual interpretation
   for the attention weights" framing, which I believe is non-discriminating.

3. **(Re CC6)** What gradient convention is used in the off-diagonal mass-block
   expression at line 1199 — `\partial f/\partial\mu_k` as a row vector or as
   a column vector? The choice changes whether `\Omega_{ki}^{-T}` or
   `\Omega_{ki}^{-1}` is the canonical pre-multiplier. Please add a one-line
   convention statement at the start of the second-variation section (or
   confirm it is implied by the chain `\Omega_{ji}^T\tilde\Lambda \Omega_{ji}`).

4. **(Re m4)** The "mass as precision" reading is used both in the
   symmetric-attention regime (where the kinetic-energy interpretation is
   consistent) and as a general intuition pump. The asymmetric-attention
   caveat at line 1207 disclaims a Hamiltonian reading. How should a reader
   interpret the "outgoing recoil" term in Eq.~`eq:effective_mass` when
   attention is asymmetric? Is it a Lyapunov contribution rather than an
   inertial one?

5. **(Re m1)** The cross-scale frame averaging
   `\phi_I^{(s+1)} = \sum w_i \phi_i^{(s)}` is well-defined but non-canonical
   for non-abelian G. Did the authors consider the Karcher mean
   `\arg\min_g \sum w_i d_G(g, \exp(\phi_i))^2` and reject it for
   tractability, or was the linear average the conceptual default? The choice
   affects whether the meta-agent frame is gauge-invariant under
   left-translation in any natural sense.

---

## Items I did NOT find issues with (so authors do not need to defend them)

- GL(K) gauge invariance theorem and proof (line 510-549).
- Vanishing-holonomy theorem (line 778-790) — algebraically clean, correctly
  characterised as "theorem of the architecture, not approximation".
- Sandwich product `\Omega \Sigma \Omega^T` for covariance transport, applied
  consistently throughout.
- Sender mean gradient at line 1152 (under the manuscript's gradient
  convention; see CC6 caveat).
- The honest disclaimers at lines 604-615 (mixture-of-sources status), 1003
  (Goldstone), 1259-1267 (mass-as-precision interpretation), 1545-1553
  (horizontal-block gauge-invariance disclosure), and 1583-1602 (signature
  problem). These are well-handled and meet the project's own pristine-codebase
  rigor standards.
