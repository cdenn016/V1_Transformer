# Notation Dictionary — Canonical Symbols Across `Attention/`

**Descriptive file.** This is a snapshot of notation as of 2026-05-18, drawn from the actual `.tex` files. The cross-manuscript-consistency agent uses this to check that the same symbol means the same thing across manuscripts.

**The dictionary itself is not canon.** The standard-literature meanings (see `external_canon_*.md`) are canon. A user-defined symbol with a consistent project-wide meaning is fine even if non-standard, as long as the meaning is consistent across manuscripts.

## Already-detected drifts (verify still present before reporting)

These were found at 2026-05-18 via grep on `\\newcommand` and `\\providecommand`:

| Symbol | `GL(K)_attention.tex` | `GL(K)_supplementary.tex` | `belief_inertia_unified.tex` | `Participatory_it_from_bit.tex` | Drift? |
|---|---|---|---|---|---|
| `\KL` | `\operatorname{KL}` | `\operatorname{KL}` | `\mathrm{KL}` | `\mathrm{KL}` | **Yes** (split across two pairs of manuscripts). Visual rendering is similar but not identical (`\operatorname` adds thin spaces around the operator). |
| `\softmax` | `\operatorname{softmax}` | `\operatorname{softmax}` | (not defined) | `\mathrm{softmax}` | **Yes** (operator vs mathrm). |
| Trace | `\Tr := \operatorname{Tr}` | `\Tr := \operatorname{Tr}` | `\tr := \mathrm{tr}` | (not defined) | **Yes** — capitalization differs. Standard mathematical convention is `\operatorname{tr}` (lowercase); `Tr` is sometimes used for finite-dim matrix trace to distinguish from operator trace. Either is acceptable but they should not be mixed. |

When reviewing, recommend a unified preamble shared via `\input{}` to eliminate this class of drift entirely.

## Variable conventions — verify each manuscript uses these consistently

These are the load-bearing symbols. The dictionary below states *what the project uses* (drawn from CLAUDE.md, the main paper abstract, and the supplementary appendices); the consistency agent verifies each manuscript uses them with these meanings.

| Symbol | Meaning | Standard reference |
|---|---|---|
| `K` | Belief-space dimension (per-head, or per-agent fiber dim) | — |
| `N` | Sequence length | — |
| `q, q_i` | Belief / recognition distribution (Gaussian over the K-fiber) | [Friston2010] — `q` is standard FEP notation for recognition density |
| `p, p_i` | Prior on beliefs | [Friston2010] |
| `s, s_i` | Latent state / model | [ParrPezzuloFriston2022] |
| `h` | Hyper-prior | — (FEP literature uses various; verify the manuscript defines it) |
| `o` | Observations | [Friston2010] |
| `μ, μ_q, μ_p` | Gaussian mean for q, p | Standard |
| `Σ, Σ_q, Σ_p` | Gaussian covariance | Standard |
| `σ, σ_q, σ_p` | Diagonal-Σ entries (or standard deviation) | Standard |
| `φ_i` | Gauge frame at agent i — element of `gl(K)` | — (user-introduced); cite [Nakahara2003] for Lie-algebra-valued gauge frames |
| `Ω_ij` | Transport from j to i: `Ω_ij = exp(φ_i) exp(−φ_j)` | — (user-introduced specific form); cite [Nakahara2003] for the general associated-bundle transport |
| `κ` | Learnable inverse-temperature scalar | — (user-introduced) |
| `τ` | Effective softmax temperature: `τ = κ √K` | [Vaswani2017] for the √K piece; κ is user |
| `β_ij` | Attention weight from j to i | [Vaswani2017] for attention; user formula `softmax(−KL/τ)` |
| `γ_ij` | Meta-attention weight (model-to-model coupling) | — (user-introduced; verify defined consistently) |
| `α, α_i` | Self-coupling weight (belief-to-prior). **Disambiguation in progress per commit `89e7982d`** — verify `α_i` has a single project-wide meaning and is distinguished from any other α in the paper (e.g., a different α used for learning rate or significance level) | — |
| `λ_h` | Hyper-prior weight | — |
| `π_ij` | Attention prior (uniform `1/N` by default) | — |
| `F` | Variational free energy | [Friston2010] |
| `G` | Expected free energy (active inference) | [ParrPezzuloFriston2022] |
| `\mathcal{C}` | Base manifold | [Nakahara2003] |
| `\mathcal{N}` | Total space of principal bundle | [Nakahara2003] |
| `\mathcal{E}_q, \mathcal{E}_p` | Associated bundles | [Nakahara2003] |

## Equation-form invariants

These equation forms should appear identically (or trivially equivalent) wherever they appear in the manuscripts. The consistency agent flags substantive variants.

| Equation | Canonical form | Where it appears (verify) |
|---|---|---|
| Covariance transport | `Σ_t = Ω Σ Ω^T` (sandwich; `[Nakahara2003]`) | `GL(K)_attention`, supplementary, `belief_inertia_unified` |
| Transport factorization | `Ω_ij = exp(φ_i) exp(−φ_j)` | `GL(K)_attention`, supplementary, `Participatory_it_from_bit` |
| Attention | `β_ij = softmax_j(−KL(q_i ‖ Ω_ij q_j) / τ)` | `GL(K)_attention` (main derivation), supplementary, `belief_inertia_unified` (uses the same coupling form), `Participatory_it_from_bit` |
| Free energy | Per CLAUDE.md / `user_theory_summary.md`, with the `τ β log(β/π)` entropy term | `GL(K)_attention` §4, supplementary, others as referenced |
| Temperature | `τ = κ √K` | All manuscripts that introduce τ |
| Closed-form KL between Gaussians | Per `external_canon_math.md` §1 | Anywhere KL is expanded |

**If a manuscript writes a variant** (e.g., `Σ → Ω Σ` without the transpose, or `Ω_ij = exp(φ_i − φ_j)` collapsing the two exponentials, or attention with `+KL` instead of `−KL`), that is a high-priority finding.

## Citation key conventions

The user's `.bib` (or inline `\bibitem`) keys should match across manuscripts when the same paper is cited.

Common keys to verify (from spot-checks of recent commits and the abstracts):

| Citation | Likely key (verify) |
|---|---|
| Friston 2010 free energy principle | `friston2010free` or similar |
| Vaswani 2017 attention | `vaswani2017attention` |
| Amari & Nagaoka 2000 | `amari2000methods` or `amariNagaoka2000` |
| Nakahara 2003 | `nakahara2003geometry` |
| Bronstein 2021 GDL | `bronstein2021geometric` |
| Kingma & Welling 2014 VAE | `kingma2014auto` or `kingmaWelling2014` |
| Su 2024 RoFormer / RoPE | `su2024roformer` |
| Bai-Kolter-Koltun 2019 DEQ | `bai2019deep` |
| Ramsauer 2021 Hopfield | `ramsauer2021hopfield` |

**Inconsistent keys** (same paper, different bibkey in different manuscripts) cause silent duplication when manuscripts are bundled or when shared bibliographies are extracted. The consistency agent flags these.

## Section / equation cross-references

When manuscript A references "as discussed in Manuscript B §X.Y", that reference should resolve. The consistency agent walks all `\ref{}` and `\cite{}` and `\Cref{}` that cross manuscript boundaries (in the inline prose, even though `\ref` typically only works within one document) and flags broken or stale references.

## `MR-N` markers — not in manuscript source

At 2026-05-18 there are no `MR-N` markers in `Attention/*.tex` (grep returned no matches). `MR-N` exists only in commit messages (e.g., `89e7982d docs(manuscript): pass 8 - alpha_i disambiguation (math reviewer MR-1 partial)`). The consistency agent tracks `MR-N` status via `git log --grep="MR-[0-9]"`, not by scanning source. If the user later introduces `MR-N` as comments or `\todo{}` macros in source, the agent should scan source too.
