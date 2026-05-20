# Verdict — pifb-theory-dynamics

## Outcome

BLUE_WINS

## Decisive evidence

The seven sub-claims of `00_claim.md` are anchored by manuscript lines that both teams verified directly in `Attention/Participatory_it_from_bit.tex`:

- Softmax attention as the row-Lagrangian stationarity condition of the full F including the `tau beta log(beta/pi)` entropy term: lines 1238, 1245, 1266 (Lagrangian written out), 1268 (boxed `beta_optimal`), 1282 (repeated in §Complete F), 1371 (autograd-vs-reduced-F identity `nabla<E>_beta - nabla F_red = -tau^{-1} Cov_{beta*}(E, nabla_x E)`).
- Kinetic-metric postulate registered as separate from F: 1847 ("once the kinetic-metric postulate of Section sec:velocity_quadratic is in place"), 1849 ("in the first instance, a stiffness on belief configuration space, not a mass" with `[arnold1989mathematical]` citation), 2013-2024 ("Stiffness as Precision" within-framework interpretation), 2026 (Pullback Gravity forward-reference to `sec:gravity_pullback`), 2031 (paragraph title "This is a postulate, not a consequence of F").
- Transformer recovery framed as gauge-fixed limit, not unique derivation: 1574 ("not derived uniquely from the gauge-theoretic framework"), 1666 (correction of constant-gauge presentation as structurally inconsistent), 1699 (boxed `Q_i = U_i^{-1} mu_i`, `K_j = U_j^T Sigma_j^{-1} mu_j` derived from `eq:gauge_qk`), 1712 (closure `Sigma_j = U_j C U_j^T`), 1742-1758 (three sufficient regimes for key-bias cancellation with explicit "outside these regimes" caveat), 1837 (boxed complete attention with `kappa = 1` recovering `[vaswani2017attention]`), 1842 ("the limits are deliberately aggressive").
- Asymmetric-attention caveat at 1958-1961 with cross-references at 1849 and 1992.

Red's opening is a concession on the evidence: "I cannot falsify sub-claims 1-7 under the evidence at lines 1005-2043" (02_red_opening.md, line 9). Red attempted three lines of attack (R1 entropy-suppressed surrogate, R2 mass-analogy cross-references, R3 transformer recovery and gauge-fixing) and verified each was pre-empted by manuscript text. No Phase 3 rebuttals were run.

## Reasoning

The methodology rule is "weigh by evidence, not rhetoric." Red performed the falsification attempt the protocol demands and reported the result honestly: every load-bearing disclosure the project canon and external standards (`[Wainwright-Jordan 2008 §3.4]` for entropic-regularization stationarity, `[Arnold 1989 Ch. 4]` for the operationally-independent inertia tensor requirement, `[Vaswani 2017 §3.2.1]` for the scaled dot-product baseline) require is present at the cited line numbers. Blue produced seven evidence vectors (B1-B7) with falsification conditions (F1-F7), each anchored to specific manuscript lines and matching external canonical forms. The two openings agree on the substantive verdict; the only delta is presentation discipline.

Red's falsification conditions (1)-(3) are either out of strict scope (condition 3, autograd-vs-reduced-F at training-loop level, is correctly flagged as a `code`-mode question) or contingent on facts outside the 1005-2043 range that the manuscript correctly forward-references rather than re-derives (conditions 1 and 2, both of which are stated by the manuscript itself as conditional rather than asserted as unconditional). Blue's F1-F7 are tied to specific contradictions that would have to appear in the cited line ranges; none do. Red's own search confirmed this.

Under the methodology, "a claim verified by executed code/sympy outweighs a claim asserted by reference alone" and "a claim that survives the opposing team's rebuttal outweighs a claim that does not." Blue's claims survive because red verified them rather than challenging them. The verdict is not a split — both teams reached the same conclusion on the evidence, and blue carries the burden of proof under the original claim with cited primary-source backing.

## Action

Accept the §Theory dynamics block (lines 1005-2043 of `Attention/Participatory_it_from_bit.tex`) as currently written. No manuscript edits required. The seven sub-claims of `00_claim.md` stand.

Out-of-scope follow-up flagged by red (condition 3): under `code` mode, verify that the running training loop uses the canonical reduced-F gradient rather than the entropy-suppressed surrogate, since the two share the same value at `beta = beta*` but differ in gradient by `-tau^{-1} Cov_{beta*}(E, nabla_x E)`. This is a separate audit, not a manuscript fix.
