# Red Panel Choice — pifb-pullback-mechanism (Phase 2, opening)

Mode: theory (differential geometry / information geometry). Full panel, 5-of-10.

| Expert | Justification (one sentence) |
|--------|------------------------------|
| `geometer` | The load-bearing question is whether $g_{E_q}$ matches the canonical Sasaki / Kaluza–Klein bundle-metric form, whose horizontal slot is the pulled-back base metric, not the connection-norm $\kappa(A,A)$ the manuscript substitutes. |
| `gauge-theorist` | The horizontal block is built from the connection one-form $A^{(i)}=U_i^{-1}dU_i$ and is admitted gauge-noninvariant (:2768); the canonical KK metric is connection-dependent yet gauge-invariant intrinsically — the distinction decides whether $\kappa(A,A)$ is a metric on $E$ or only on a chosen trivialization. |
| `info-geometer` | Cencov uniqueness forces the Fisher piece, but the $\kappa(A,A)$ horizontal piece depends on a choice external to the statistical manifold — whether the sum is still an information metric is exactly this lens. |
| `variational` | The tier-identification "perceived space $=G^{(s)}$" rests on Friston2017 locating structural perception in parameter-learning vs state-inference; whether the source supports that reading is hierarchical-active-inference territory. |
| `philosophy-of-science` | Mandatory. Frame-checks whether "we propose ... most naturally identified with $G^{(s)}$" is a falsifiable claim or an interpretive preference, and catches any manuscript-as-authority circularity in the "bona-fide bundle metric" naming. |

Dropped from the theory default: none beyond the default set — `numerical-analyst` excluded (no fp32/conditioning/retraction surface in this claim); `transformer-ml`, `implementation-engineer`, `code-quality`, `ml-engineer` excluded (pure theory span, no code/config under test).

## Primary attack vector (synthesis target)

Single load-bearing attack: the canonical bundle metric (Kaluza–Klein / Sasaki) has horizontal block = **pulled-back base metric** $g_{ab}(x)$, written invariantly so the total-space metric is gauge-invariant. The manuscript (:2734, :2739) puts $\kappa(A_\mu,A_\nu)$ — the connection-norm-squared — in the horizontal slot because it has no base metric on $\mathcal{C}$ to use (the construction's purpose is to manufacture one). That substitution is non-canonical, and the resulting $g_{E_q}$ is gauge-noninvariant (admitted :2768), unlike the canonical KK metric. If this holds against Nakahara / Kobayashi–Nomizu, "bona-fide bundle-metric pullback" fails, and the "$L^2$-only vs bundle-metric" distinction at :2726 collapses to "the same $G^{(q)}$ formula plus a connection-dependent additive term."

Secondary vector: the tier-identification (:2795) lacks a falsification condition and the Friston2017 citation is read more strongly than the source supports.
