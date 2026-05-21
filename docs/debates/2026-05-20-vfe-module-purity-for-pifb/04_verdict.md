# Verdict — vfe-module-purity-for-pifb

## Outcome

RED_WINS

## Decisive evidence

The per-dim Bayesian alpha gap. Three load-bearing facts settle it:

- `Attention/Participatory_it_from_bit.tex:1285-1333` (Eq:state_dependent_alpha) prescribes a per-agent scalar `alpha_i* = c0 / (b0 + KL(q_i || p_i))` with a single scalar log-barrier `R(alpha_i) = b0 * alpha_i - c0 * log(alpha_i)`.
- `transformer/vfe/e_step.py:50-66` (module docstring, verified in-tree) admits per-K-dimension `raw_c0, raw_b0 in nn.Parameter` of shape `(K,)`, returns `c0 / (b0 + kl_k)` with per-dim `kl_k`, and explicitly states "This is a stronger generalisation than the published scalar form and is not currently derived in the manuscript."
- For K > 1 the two forms are algebraically inequivalent: per-dim `c0 / (b0 + kl_k)` summed over k does not reduce to scalar `c0 / (b0 + sum_k kl_k)`. No flag in `VFEConfig` constrains `raw_c0`, `raw_b0` to scalars or routes the denominator to the total KL.

Blue conceded this gap in `02_blue_opening.md:51` ("If red attacks here, blue concedes"); Red adopted it as a falsifying construction in `03_red_rebuttal.md:33`. The falsification text in `00_claim.md:23-25` triggers on "any single ... theoretical construction in participatory_it_from_bit.tex that has *no corresponding code path* in `transformer/vfe/` under any toggle setting." PIFB:1311's scalar construction satisfies that trigger.

## Reasoning

The falsification clause's "under any toggle setting" phrasing licenses a per-construction existential reading — separate toggle allowed per PIFB construction. Under that reading, Red's mutual-exclusion attack (omega_direct vs exact_full_cov_decode at `config.py:566-573` and `config.py:491-495`) is irrelevant: the claim does not require one config to realize both simultaneously. Similarly, three of Red's surviving attacks fail on closer reading. The phi-preconditioner attack collapses because PIFB:2558 explicitly licenses "for any choice of inner product on g" and PIFB:2576 designates the group-level retraction as canonical with the chart-coordinate form labeled "implementation specialization"; `omega_direct.py:284-289` builds a Frobenius gram on the generator basis, which is positive-definite by basic linear algebra on a linearly independent basis [Hall 2015 §A.7] and right-invariant by [Absil/Mahony/Sepulchre 2008 §3.6.2]. The Wilson observable attack is out of scope: PIFB:824 places the Regime II edge-relaxed cocycle and its Wilson regularizer in the framework's "gravitational and signature-related extrapolations," and the user's evidence pack restricts the debate to language modeling. The Vaswani-limit attack is self-defeating: PIFB:1574 itself states standard attention is "recovered as a gauge-fixed and isotropic-Gaussian limit ... not derived uniquely" and PIFB:1582 says the empirical experiments implement the general gauge model without taking those limits. The cross-layer phi-handoff attack is overstated: PIFB:1537 uses "may evolve" (modal), and `stack.py:142-144` documents that posterior phi does cascade via `beliefs.phi` even when the prior phi stays at the embedding.

What remains is the per-dim alpha gap that both teams agree on. PIFB:1311 derives a scalar construction; `transformer/vfe/e_step.py:50-66` implements an inequivalent per-dim form and labels it not-derived-in-the-manuscript. No toggle in `VFEConfig` recovers the scalar form. This is one PIFB construction with no realizing code path under any toggle setting — the literal trigger condition of the falsification clause.

The compound claim therefore fails on at least one construction. Per the user's stipulated falsification text, that suffices for Red.

## Action

Either add a scalar-alpha toggle to `VFEConfig` that constrains `raw_c0`, `raw_b0` to scalars (`(1,)` rather than `(K,)`) and computes the denominator as `b0 + sum_k kl_k`, restoring the canonical PIFB:1311 form as a reachable configuration; or derive the per-dim Gamma-Normal conjugacy and per-dim log-barrier in `Attention/Participatory_it_from_bit.tex` so the implemented form has manuscript support. Either action closes the gap that decided this debate. Separately, the omega_direct vs exact_full_cov_decode mutual exclusion at `config.py:566-573` and `config.py:491-495` should be revisited if the user wants a single configuration that realizes both canonical group-level retraction and the (0,2)-tensor sandwich decode simultaneously — that gap is real under a single-config reading of the existential but did not bind under the per-construction reading the falsification clause licensed.
