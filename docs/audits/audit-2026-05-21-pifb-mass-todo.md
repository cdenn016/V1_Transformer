# Audit 2026-05-21 — PIFB $\omega^2 \propto m_{\text{eff}}^{-1}$ TODO (lines 1880, 3028)

**CORRECTION 2026-05-21 (post-audit):** The codebase claims in this document (Section 1 rows referencing `transformer/vfe/...` paths, and Section 3 verdict about "no symplectic integrator") were verified against the WRONG repository. The correct codebase backing `Participatory_it_from_bit.tex` is at `C:\Users\chris and christine\Desktop\MAgent_Model-main`, which DOES have a symplectic leapfrog (`gauge_agent/dynamics.py:1067, 1606-1637`), dual mass conventions, and a `runs/hamiltonian_oscillator/` directory. See `MAgent_Model-main/docs/audit-2026-05-21-mass-todo-feasibility.md` (investigator) and `MAgent_Model-main/docs/audit-2026-05-21-mass-todo-VERIFIER.md` (verifier) for the corrected code-feasibility verdict. The manuscript-context and info-geometry sections below remain valid.

Final verifier consolidating three investigator memos (manuscript reviewer, codebase auditor — WRONG REPO, info-geometer).

## 1. Verification table

| Claim | Status | Evidence |
|-------|--------|----------|
| $M_{\mu\mu}$ defined at PIFB:1958 as Hessian sector | VERIFIED | Eq. eq:mass_block_structure at lines 1957-1960. |
| Eq. eq:mass_mu_diagonal at PIFB:1973 gives $\bar\Lambda_p + \sum\beta\tilde\Lambda_q + \sum\beta\Lambda_q + \Lambda_o$ | VERIFIED | Lines 1972-1975, boxed. |
| Isolated-agent collapse to $\bar\Lambda_p = \Sigma_p^{-1}$ | VERIFIED | Lines 1963-1964 ("Notation hierarchy") state this explicitly: $\Sigma_{p,i}^{-1}$ is the leading isolated-agent term when $\beta_{ij}=0$, $\Lambda_{o_i}=0$. |
| Kinetic postulate at PIFB:2066 reuses the same $M_{\mu\mu}$ | VERIFIED | Eq. eq:full_kinetic at 2065-2068 writes $\tfrac12 \dot\mu^\top M_{\mu\mu} \dot\mu$ — literally the same matrix. |
| Line 2064 admits "when $k$ and $m$ are both equal to $M_{\mu\mu}$ by construction, $\omega^2$ reduces to a per-direction unit relation" | VERIFIED | Verbatim at line 2064. |
| `transformer/vfe/config.py:101` `lambda_align: float = 1.0` zeros β·KL coupling at 0 | VERIFIED | Line 101 exact match. (Semantic claim "zeros coupling at 0" matches the Boltzmann GLU weight comment; reasonable.) |
| `e_step.py:1349-1357` is forward-Euler natural-gradient descent on μ | VERIFIED | Lines 1346-1357: `_delta_mu = self.e_mu_lr * nat_grad_mu; mu = mu - _delta_mu`. Pure explicit Euler. |
| `vfe_gradients.py:2188-2191` diagonal natural gradient is `nat_grad_mu = sigma * grad_mu` | VERIFIED | Line 2190: `nat_grad_mu = sigma_safe * grad_mu`. First-order ∇F preconditioned by Σ. |
| No symplectic/leapfrog/2nd-order integrator in `e_step.py` | VERIFIED | `grep` for leapfrog/symplectic/verlet in `transformer/vfe/` returns zero hits. The σ retraction at 1360-1370 is geodesic SPD retraction, not symplectic. |
| `prior_bank.py:194` has `base_log_sigma` parameter; clamped exp at 227-234 | VERIFIED | Line 194: `self.base_log_sigma = nn.Parameter(torch.full((K,), sigma_init_log))`. Lines 227-234: `torch.exp(...).clamp(0.01, self.sigma_max)`. |
| `scripts/vfe_convergence.py` exists | VERIFIED | Glob returns `scripts/vfe_convergence.py` and `scripts/vfe_convergence_obs.py`. |
| Info-geometer rescue: diagonal $\bar\Lambda_p = \Sigma_p^{-1}I$, $\Lambda_q = \Sigma_q^{-1}I$ gives $\omega^2 = \Sigma_q/\Sigma_p$ | VERIFIED | $\det(\Sigma_p^{-1}I - \omega^2 \Sigma_q^{-1}I) = 0 \Rightarrow \omega^2 = \Sigma_q/\Sigma_p \propto \Sigma_p^{-1}$ at fixed $\Sigma_q$. Algebra trivially correct. |

## 2. Convergence assessment

All three investigators converge on the central diagnosis: **the test as literally written is tautological under the current postulate** because the manuscript itself reuses $M_{\mu\mu}$ as both stiffness $k$ and inertia $m$ (PIFB:2064 admits this in print). No contradictions among the three memos. The codebase auditor and info-geometer additionally agree that a *reframed* test is feasible — the auditor identifies the existing forward-Euler natural-gradient loop as the operational platform for measuring a relaxation rate $\gamma$, and the info-geometer identifies posterior-Fisher $\Lambda_q$ vs prior-Fisher $\bar\Lambda_p$ as the only logically clean way to separate the two roles. The codebase verification confirms zero symplectic/Hamiltonian infrastructure exists, so a true oscillation-frequency $\omega$ test is structurally infeasible without writing new integrator code.

## 3. Final reconciled verdict on the TODO

**Literally as written (operationally independent $\omega$ and $m_{\text{eff}}$): NOT feasible** in the manuscript's current framework. The framework collapses $k = m = M_{\mu\mu}$ by postulate; the codebase implements relaxational dynamics ($\dot\mu = -\Sigma \nabla_\mu \mathcal{F}$) that produce exponential decay, not oscillations. There is no $\omega$ to measure — only a relaxation rate $\gamma$.

**Honest scope: rewrite the TODO.** The manuscript at line 2064 already provides the correct framing — fitting $\omega$ from autocorrelation of a natural-gradient trajectory under a stiffness $M_{\mu\mu}$ that does not coincide with the kinetic metric. The TODO at line 3028 should be aligned with this: either (a) replaced with a relaxation-rate test ($\gamma \propto \bar\Lambda_{p,i}$ under fixed $\Sigma_q$, swept over $\bar\Sigma_p$), or (b) reframed to use the info-geometer's $\det(\bar\Lambda_p - \omega^2 \Lambda_q) = 0$ secular equation, which requires introducing an independent kinetic-metric posit (posterior Fisher) distinct from the potential Hessian.

**Recommended concrete action:**
1. Rewrite the TODO at PIFB:3028 to (a) acknowledge the framework's $k = m$ identification (cite the line 2064 admission), (b) state that a true dispersion test requires either an independent kinetic-metric posit OR a relaxation-rate surrogate, and (c) cite `scripts/vfe_convergence.py` as the existing harness if pursuing the relaxation-rate route.
2. Optionally write `scripts/mass_relaxation_test.py` to fit $\gamma_i$ from $\log\|\mu_i(t) - \mu_i^*\|$ vs $t$ under the existing `e_step.py` Euler loop with `lambda_align=0` (isolated-agent limit) while sweeping `base_log_sigma`; predicted scaling is $\gamma \propto \bar\Sigma_{p,i}^{-1}$.

## 4. Confidence and gaps

**High confidence:** the tautology diagnosis (verbatim in the manuscript), the absence of symplectic integrators (grep is exhaustive), and the trivial algebra of the info-geometer's diagonal example.

**Medium confidence:** that the auditor's `lambda_align=0` route fully decouples the βs in the active runtime — I verified the config line but did not trace `lambda_align` through every consumer in the fused E-step kernels; an off-equilibrium softmax-grad term might leak unless `lambda_soft=0` is also set. Anyone implementing the test must verify both flags.

**What would change my mind:** if `e_step.py` had a hidden second-order/symplectic branch I missed (it doesn't — grep was clean), or if the manuscript at some other section actually constructs an independent kinetic metric beyond the postulate at 2061-2069 (the surrounding prose at 2046-2059 and 3026-3028 confirms it does not — both sections explicitly identify mass with the Hessian/Fisher).
