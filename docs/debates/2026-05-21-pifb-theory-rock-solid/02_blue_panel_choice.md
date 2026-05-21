# Blue Panel Choice — pifb-theory-rock-solid (Phase 2, Opening)

Mode: theory. Panel: full (5 experts). Coordinator: blue.

The dispatched 5-expert panel matches the theory-mode default. Each lens has direct purchase on at least one of the six conjunctive sub-claims operationalized in `00_claim.md` (mathematical correctness, canonical fidelity, internal consistency, falsifiability/scope, self-containedness, no unresolved gaps).

| Tag | One-sentence justification |
|---|---|
| `geometer` | Differential-geometry lens checks the principal-bundle setup, sandwich-product covariance transport at line 608, and the precision-transport law at line 1894 against Nakahara/KobayashiNomizu — these are §Theory's load-bearing geometric primitives and blue's strongest sub-claim-1/2 defense vectors. |
| `info-geometer` | Information-geometry lens checks the canonical Gaussian KL (Eq. for $D_{\mathrm{KL}}$ at line 1910), the per-agent precision $\alpha_i^* = c_0/(b_0+D_{\mathrm{KL}})$ derivation at lines 1285–1359, the envelope-theorem reduction at 1361–1398, and Cencov-uniqueness of the Fisher metric — directly load-bearing for sub-claims 1 and 2. |
| `variational` | Variational-inference lens checks the canonical free energy at line 1252, the softmax-from-Lagrangian derivation at 1266–1281, the cross-entropy/environmental-agent equivalence at 1438–1475, and the mixture-of-sources construction at 1033–1113 — these are blue's strongest sub-claim-1 defenses for the F functional itself. |
| `gauge-theorist` | Gauge-theory lens checks the dual-role gauge frame (lines 555–565), $\Omega = \exp(\phi_i)\exp(-\phi_j)$ parameterization of $\mathrm{GL}^+(K)$, edge-modes / quantum-reference-frame justification (DonnellyFreidel, BartlettRudolphSpekkens, Vanrietvelde), and SSB analogy caveat at 1499–1527 — load-bearing for sub-claim 4 (falsifiability/scope: analogy labeled as analogy). |
| `philosophy-of-science` | Mandatory. Frame-checks the conjunctive operationalization itself, polices manuscript-as-authority circularity, and adjudicates whether the publication-bar reading of "rock solid and publication ready" treats sub-claims 5/6 (companion-paper cites, TODO marker) as actually blocking — the only lens that can credibly argue the strict literal reading is over-strict relative to peer-review practice. |

Notes on what was rejected:

- `transformer-ml` was excluded because the §Theory transformer-limit reduction (line 1860, line 1870) is cleanly cited to Vaswani2017 and the boxed dot-product form is uncontested by the evidence pack; the harder claims (multi-head, RoPE, FFN) are explicitly punted to the companion paper, which is precisely the sub-claim-5 vulnerability that `philosophy-of-science` will adjudicate.
- `numerical-analyst` was excluded because §Theory makes no specific numerical-stability claim load-bearing for "rock solid"; the symbolic "to machine precision" claim at line 1717 is for one specific equation and either holds or doesn't independently of the lens.
- `ml-engineer`, `implementation-engineer`, `code-quality` were excluded because the claim is restricted to §Theory and not to implementation behaviour.

Red and blue may pick different panels — this is by design. Adversarial panel selection is permitted and judged on the synthesized output.
