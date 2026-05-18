# Cross-Manuscript Consistency Report — `Participatory_it_from_bit.tex` vs the rest of `Attention/`

**Date:** 2026-05-18
**Scope:** Drift between `Participatory_it_from_bit.tex` and the other manuscripts in `Attention/`, restricted to notation, equations, citations, MR-N items, style, narrative, and EM-mode/config terminology.
**Method:** ordered checklist per `.claude/agents/vfe-knowledge/consistency_methodology.md`. Source-of-truth principle per `.claude/agents/vfe-knowledge/README.md`.
**Output discipline:** This agent does not edit manuscripts; it surfaces drift for the user to resolve.

## Manuscripts in scope

- `Attention/Participatory_it_from_bit.tex` (4686 lines, current target)
- `Attention/GL(K)_attention.tex` (2340 lines, main JMLR paper)
- `Attention/GL(K)_supplementary.tex` (1323 lines, supplementary to main)
- `Attention/belief_inertia_unified.tex` (2288 lines, sociological-dynamics companion)
- `Attention/jmlr_coverletter.tex` (96 lines, cover letter for the main paper)
- `Attention/tikz.tex` (322 lines, standalone figure source)
- `Attention/references.bib` (shared bibliography)

## Notation drift table

| # | Symbol / object | This paper (Participatory) | GL(K)_attention | GL(K)_supplementary | belief_inertia_unified | Severity |
|---|---|---|---|---|---|---|
| N1 | `\KL` macro | `\mathrm{KL}` at `Participatory:33` (via `\providecommand`) | `\operatorname{KL}` at `GL(K)_attention.tex:28` | `\operatorname{KL}` at `GL(K)_supplementary.tex:22` | `\mathrm{KL}` at `belief_inertia_unified.tex:13` | Minor |
| N2 | `\softmax` macro | `\mathrm{softmax}` at `Participatory:34` | `\operatorname{softmax}` at `GL(K)_attention.tex:27` | `\operatorname{softmax}` at `GL(K)_supplementary.tex:21` | (not defined as macro) | Minor |
| N3 | Trace operator | (no macro defined; uses `\mathrm{tr}` and `\mathrm{Tr}` inline) | `\Tr := \operatorname{Tr}` at `GL(K)_attention.tex:29` | `\Tr := \operatorname{Tr}` at `GL(K)_supplementary.tex:23` | `\tr := \mathrm{tr}` at `belief_inertia_unified.tex:14` | Minor |
| N4 | Bare `κ` (no subscript) | Killing form on the gauge Lie algebra (documented at `Participatory:185`) | Learnable scalar inverse temperature; column header at `GL(K)_attention.tex:409` reads `κ (τ) … attention temperature` | Same as main | Bounded-confidence temperature (lines `belief_inertia_unified.tex:842–971`) | **Major** — same letter, three meanings across the set |
| N5 | Attention temperature scalar in the κ-family | `κ_β` (belief), `κ_γ` (model) at e.g. `Participatory:3668, 3852` | `κ_a` per head at `GL(K)_attention.tex:1744`; bare `κ` in the column table at `409` | (uses `\kappa_a` analogous to main) | (n/a — uses bare `κ` for bounded-confidence) | Major |
| N6 | `τ` (bare) | Attention/entropy temperature in canonical F (Participatory documents `τ = κ√K` at `Participatory:1241`) | Same role; `τ = √d_k` derived as the standard-attention recovery limit at `GL(K)_attention.tex:1280`; CLAUDE.md canon: `τ = κ√K` | Same as main | **Overloaded within the paper**: attention temperature at `belief_inertia_unified.tex:157`, decay/proper time at `173, 192, 433, 485, 602, 646, 663, 1144`, flow-time at `734, 746` | **Major** within belief_inertia; consistent between Participatory and main |
| N7 | Belief mean / covariance subscripting | `μ_i`, `Σ_i` for q; `μ_i^p, Σ_i^p` (superscript) for prior; `μ_i^s, Σ_i^s` for model; `μ_i^r, Σ_i^r` for hyper-prior — at `Participatory:632–639` | `μ_{q,i}, Σ_{q,i}, μ_{p,i}, Σ_{p,i}` (comma-subscript) at `GL(K)_attention.tex:591–594` | Matches main: `Σ_{p,i}` etc. | `\muQ := \mu^q`, `\muP := \mu^p`, `\SigQ := \Sigma^q`, `\SigP := \Sigma^p` (superscript) per preamble `belief_inertia_unified.tex:16–19` | Minor — three conventions for the same objects |
| N8 | Model-fiber dimension | `M` at `Participatory:496, 511, 639` (m ∈ ℝ^M) | `K_p` at `GL(K)_attention.tex:383, 391, 397, 399, 400` | Inherits from main | (not used; single fiber) | Minor — disjoint per-paper meaning, but cross-paper drift |
| N9 | Gauge frame codomain | `φ_i: U_i → 𝔤` (Lie algebra of arbitrary gauge group `G`) at `Participatory:619, 628` | `φ_i: U_i → 𝔤𝔩(K)` at `GL(K)_attention.tex:619` | Same as main | (φ used; sociological-fibers context) | Minor — Participatory is the more general framework; not a drift if explicitly declared (see NC1) |
| N10 | `λ_h` (hyper-prior weight on model channel) | Defined explicitly at `Participatory:1253, 1262` | **Not used** in `GL(K)_attention.tex` (no occurrences; model channel relegated to Appendix per `GL(K)_attention.tex:677`) | Inherits from main; no `λ_h` | (n/a — single fiber) | Major drift in F-functional scope (see E2 / NC1) |

## Equation drift list

| # | Equation | Participatory location | Other-manuscript location | Issue / canonical form | Severity |
|---|---|---|---|---|---|
| E1 | Covariance sandwich `Σ → Ω Σ Ω^T` | Correct, used consistently at e.g. `Participatory:940, 948, 952, 957, 963, 975, 3697, 3703, 3736, 3955` | Correct in `GL(K)_attention.tex:613–631` and `GL(K)_supplementary.tex:230, 245, 260, 283, 340–341, 355, 383, 484–498`; correct in `belief_inertia_unified.tex:130, 256` | No drift — all four manuscripts use the sandwich form `Ω Σ Ω^T` [Nakahara2003]. | None |
| E2 | Transport factorization `Ω_ij = exp(φ_i) exp(−φ_j)` | Correct at `Participatory:465, 778, 3429, 3431, 3794, 3955` | Correct at `GL(K)_attention.tex:615`, `GL(K)_supplementary.tex:159, 430`, `belief_inertia_unified.tex` (via `Ω_{ij}` macro, no collapse to `exp(φ_i − φ_j)` found) | No drift — all four manuscripts use the two-exponential form. The Frechet derivative path is also consistent. | None |
| E3 | Attention `β_ij = softmax_j(−KL(q_i ‖ Ω_ij q_j) / τ)` | Canonical form at `Participatory:1126, 1153, 1170, 1234, 1264, 1832` — uses bare `τ` | Same canonical form at `GL(K)_attention.tex:787, 815, 832, 1086, 1181, 1744, 1886` — uses bare `τ` in the body, `κ_a √d_head` in the implementation eq. | Same form at `belief_inertia_unified.tex:157` | Consistent across the body of all three Gaussian/attention papers. The Participatory **appendix** at `Participatory:3668, 3687, 3697, 3703, 3725, 3736, 3829, 3839, 3848, 3852` uses `κ_β` and `κ_γ` instead of `τ` for the same role; the SO(3) appendix never restates `τ = κ_β` (or names the relation) and never restates the `√K` factor. This is the N5 drift surfaced as an equation-level issue. | Major |
| E4 | Temperature factorization `τ = κ √K` | Stated at `Participatory:1241` ("In the working implementation the temperature is factorised as $\tau = \kappa\sqrt{K}$"). | Main paper does not write `τ = κ √K` in the body. The closest forms are `τ = √d_k` at `GL(K)_attention.tex:1280, 1665` (recovery limit), and the multi-head `κ_a √d_head` factorization at `GL(K)_attention.tex:1741, 1744`. CLAUDE.md states `τ = κ √K` as canonical and the project's MR-2 commit history confirms the `Participatory:1241` insertion was made specifically to satisfy MR-2 (see `Attention/REVIEW_2026-05-18.md:63, 324, 803, 874`). | Main paper should adopt the same explicit `τ = κ√K` statement to match the canon and the companion. | Major (cross-manuscript canon drift) |
| E5 | Full F functional | Eq. `eq:free_energy_functional_final` at `Participatory:1248–1259` includes **both** the belief-channel and model-channel KL coupling terms with weights `β_ij` and `γ_ij`, the entropy regularizers `τ β log(β/π)` and `τ γ log(γ/π^{(s)})`, and the explicit `λ_h` weight on the model-prior block. | Main paper's `eq:free_energy_final` at `GL(K)_attention.tex:844–852` writes only the reduced form `Σ_i KL(q_i ‖ p_i) − τ Σ_i log Z_i − E_q[log p(o|·)]`. The model channel `KL(s_i ‖ r_i)` and meta-attention `γ_ij` are **not** in the main paper's F; they are mentioned as "an analogous expression holds for a model channel (Appendix~\ref{app:model_channel})" at `GL(K)_attention.tex:677`. | Participatory's F is a strict superset of the main paper's F. The relationship is consistent if labeled as a superset and acknowledged from both directions; see NC1. | Narrative (see NC1), not equation |
| E6 | `α_i` self-coupling, log-barrier `R(α_i) = b_0 α_i − c_0 log α_i`, closed-form `α_i^* = c_0 / (b_0 + KL(q_i ‖ p_i))` | At `Participatory:1287–1306, 1318–1325` | At `GL(K)_attention.tex:900–958` — identical structure | (n/a in belief_inertia) | No drift. MR-1 alpha_i resolution applied consistently in both papers per commit `89e7982d` and `Attention/REVIEW_2026-05-18.md:718–722`. | None |
| E7 | Entropy-suppressed surrogate vs reduced F gap = `−τ^{-1} Cov_β(KL, ∇KL)` | At `Participatory:1241, 1278` and again at `Participatory:3671` | At `GL(K)_attention.tex:866–871` (`eq:autograd_envelope_gap`) | (n/a) | No drift. Same identity, stated in both. | None |

## Citation drift list

| # | Paper | Bibkey in Participatory | Bibkey in GL(K)_attention | Bibkey in supplementary | Bibkey in belief_inertia | Bib entry status | Severity |
|---|---|---|---|---|---|---|---|
| C1 | Friston 2010 "Free-energy principle" | `Friston2010` (used 6× in Participatory) | `friston2010free` (used at `GL(K)_attention.tex:60, 670`) | (not directly cited) | `Friston2010` (`belief_inertia_unified.tex:72, 1207`) | **Two duplicate entries** in `references.bib`: `Friston2010` at line `562` and `friston2010free` at line `2293`. Same paper. | **Major** |
| C2 | Vaswani et al. 2017 "Attention is all you need" | (not cited at all — verified by `grep -i vaswani Participatory_it_from_bit.tex` → no matches; yet the paper recovers scaled dot-product attention with `τ = √d_k` at `Participatory:1802, 1807, 1836`) | `vaswani2017attention` at `GL(K)_attention.tex:60, 1699` | (not cited) | (not cited) | Duplicate entries `Vaswani2017` (line `818`) and `vaswani2017attention` (line `2403`) exist in `.bib`. | **Major** — missing citation in Participatory body where attention recovery is derived |
| C3 | Friston et al. 2017 "Active inference: a process theory" | `Friston2017` (used at `Participatory:69, 165, 2703`) | `friston2017graphical` (used at `GL(K)_attention.tex:60`) — **different paper** | (not directly cited) | (not cited) | The two bibkeys actually point to **different** Friston 2017 papers: `Friston2017` (line `571`) is the *Neural Computation* "process theory" paper; `friston2017graphical` (line `2303`) is the *Network Neuroscience* "graphical brain" paper. Both are legitimate but separate citations. **Not a drift.** | None |
| C4 | Parr, Pezzulo & Friston 2022 *Active Inference* | `Parr2022` (used at `Participatory:77, 165, 1013, 1225, 1392, 2703` etc.) | `parr2022active` (used at `GL(K)_attention.tex:60, 670`) | (not directly cited) | `parr2022active` (used at `belief_inertia_unified.tex:84`) | Duplicate entries `Parr2022` (line `580`) and `parr2022active` (line `188`). Same book. | **Major** |
| C5 | Amari 2016 *Information Geometry and Its Applications* | `Amari2016` (used at `Participatory:924, 1517`) | `amari2016information` (used at `GL(K)_attention.tex:518`) | `amari2016information` (used at `GL(K)_supplementary.tex:49`) | `Amari2016` (`belief_inertia_unified.tex:74, 80, 993, 1111, 2266`) | Duplicate entries `Amari2016` (line `609`) and `amari2016information` (line `2188`). Same book. **Particularly invidious:** `Participatory:931` writes `\cite{amari2016information}` (lowercase) and `Participatory:924` writes `\cite{Amari2016}` (CamelCase) — the **same manuscript** cites the same book under both keys. | **Major** (cross-paper and within-Participatory) |
| C6 | Nakahara 2003 *Geometry, Topology and Physics* | `Nakahara2003` (used at `Participatory:586, 599`) | `nakahara2003geometry` (used at `GL(K)_attention.tex:364`) | `nakahara2003geometry` (`GL(K)_supplementary.tex:49`) | (not cited) | Duplicate entries `Nakahara2003` (line `661`) and `nakahara2003geometry` (line `244`). Same book. | **Major** |
| C7 | Frankel 2011 *The Geometry of Physics* | `Frankel2011` (`Participatory:586`) | (not cited) | `frankel2011geometry` (`GL(K)_supplementary.tex:49`) | (not cited) | Duplicate entries `Frankel2011` (line `669`) and `frankel2011geometry` (line `2285`). Same book. | **Major** |
| C8 | Companion paper — the gauge-theoretic transformer paper | `Dennis2025trans` (used ~20× across Participatory at `1199, 1284, 1328, 1569, 1577, 1585, 1638, 1664, 1779, 1840, 2589, 4268`, etc.) | (does not self-cite the transformer paper by tag; cites `Dennis2025it` instead at `665, 2271` to point to **this** Participatory paper) | (not cited) | `Dennis2025` (bare, at `belief_inertia_unified.tex:139 (×2), 2098`) — **broken bibkey** | `Dennis2025trans` exists at `references.bib:1506`; `Dennis2025it` exists at `1488`; `Dennis2025atten` exists at `1514`; bare **`Dennis2025` does NOT exist** in the `.bib`. The three `belief_inertia_unified.tex` cites resolve to `[?]` at compile time. | **Critical** — three broken `\citep{}` calls in `belief_inertia_unified.tex` |
| C9 | Wheeler 1990 "Information, Physics, Quantum" | `Wheeler1990` at `Participatory:65, 2610, 3228, 3241` | (not cited) | (not cited) | (not cited) | Single entry `Wheeler1990` at `references.bib:469`. No cross-paper drift. | None |
| C10 | Cencov 1982 (Russian original 1972) — KL/Fisher uniqueness | `Cencov1982` at `Participatory:505` | (not cited; uses `kullback1951information` at `GL(K)_attention.tex:364` for related material) | (not cited) | (not cited) | Single entry `Cencov1982` at `1785`. No duplicate; flagged because main paper's KL/Fisher uniqueness discussion may benefit from the same source-of-truth citation. | Minor (citation-pool gap, not duplication) |

## Recommended unifications (concrete diffs)

The clean fix for the bibkey schism is a **single canonical key per paper**, applied uniformly across all manuscripts. Either case-convention is defensible; the existing project skew (main paper lowercase, Participatory + belief_inertia CamelCase) suggests one or the other should be chosen and the other deleted. Recommend lowercase to match the main JMLR submission (which is the most polished file).

1. **U1 — Friston 2010 → `friston2010free`.** In `Participatory_it_from_bit.tex` change all 6 `\cite{Friston2010…}` to `\cite{friston2010free…}`. In `belief_inertia_unified.tex` change both `\citep{Friston2010,…}` and any `\cite{Friston2010}` to `friston2010free`. Delete the `@article{Friston2010, …}` block at `references.bib:562–569`.
2. **U2 — Vaswani 2017 → `vaswani2017attention`.** Add `\cite{vaswani2017attention}` next to every Participatory equation that recovers scaled dot-product attention (`Participatory:1800, 1802, 1807, 1836` — the recovery derivation is currently citation-less). Delete the duplicate `@inproceedings{Vaswani2017, …}` block at `references.bib:818–825`.
3. **U3 — Parr, Pezzulo & Friston 2022 → `parr2022active`.** Replace `Parr2022` in `Participatory_it_from_bit.tex` (6 sites). Delete `@book{Parr2022, …}` at `references.bib:580`.
4. **U4 — Amari 2016 → `amari2016information`.** Replace `Amari2016` in `Participatory_it_from_bit.tex` (2 sites — note that line `Participatory:931` already uses `amari2016information` so the manuscript is inconsistent with itself). Replace in `belief_inertia_unified.tex` (5 sites). Delete `@book{Amari2016, …}` at `references.bib:609–614`.
5. **U5 — Nakahara 2003 → `nakahara2003geometry`.** Replace `Nakahara2003` in `Participatory_it_from_bit.tex` (2 sites). Delete `@book{Nakahara2003, …}` at `references.bib:661`.
6. **U6 — Frankel 2011 → `frankel2011geometry`.** Replace `Frankel2011` in `Participatory_it_from_bit.tex` (1 site at line `586`). Delete `@book{Frankel2011, …}` at `references.bib:669`.
7. **U7 — Fix the three broken `\citep{Dennis2025}` in `belief_inertia_unified.tex`.** Determine intent: if the citation should point to the gauge-transformer paper, rename to `Dennis2025trans` (matches Participatory's usage). If it should point to the Participatory paper itself, rename to `Dennis2025it`. The two sites are `belief_inertia_unified.tex:139` (twice on same line) and `belief_inertia_unified.tex:2098` (where the context says "adapted from the GL(K) attention framework" — this is `Dennis2025trans`).
8. **U8 — Shared LaTeX preamble for `\KL`, `\softmax`, `\Tr`/`\tr`.** Extract a `preamble_common.tex` containing the canonical macros (suggested: `\providecommand{\KL}{\operatorname{KL}}`, `\providecommand{\softmax}{\operatorname{softmax}}`, `\providecommand{\Tr}{\operatorname{Tr}}`). `\input{}` it at the top of each manuscript. Remove the per-file `\providecommand` and `\newcommand` for these macros.
9. **U9 — Notation harmonization for Gaussian fields.** Adopt one of `μ_{q,i}` (main), `μ_i^q` (belief_inertia preamble), or `μ_i` + `μ_i^p` (Participatory) across all manuscripts. The main paper's `μ_{q,i}, Σ_{q,i}, μ_{p,i}, Σ_{p,i}` is the most explicit and disambiguation-friendly. Recommend that convention.
10. **U10 — `τ = κ√K` in the main paper.** Insert one sentence in `GL(K)_attention.tex` near the temperature discussion at line `1741` (or after `eq:free_energy_final` at line `855`) stating: "In the working implementation the temperature is factorised as $\tau = \kappa\sqrt{K}$ with $\kappa$ a learnable scalar; the dimension-scaling factor $\sqrt{K}$ recovers $\sqrt{d_k}$ in the per-head limit." This matches the canon in CLAUDE.md, matches Participatory line `1241`, and makes the cross-paper convention explicit.
11. **U11 — Participatory appendix `κ_β` / `κ_γ` should be tied to `τ`.** In the SO(3) Gaussian appendix (`Participatory:3653–3870`), add one sentence at the start identifying `κ_β` and `κ_γ` as the belief-channel and model-channel instantiations of `τ` from the body's canonical free energy (Eq. `eq:pointwise_free_energy`, line `1234`), so the appendix subscript convention does not appear unannounced. This is a documentation fix; the math is correct.
12. **U12 — Remove the two "earlier draft" self-references in Participatory.** Lines `Participatory:189` and `Participatory:2068` both write "to avoid the notational collision the earlier draft had". This violates the user's `feedback_no_self_referential_history.md` rule. Rewrite as "to avoid notational collision, …" or strike the clause entirely.
13. **U13 — Cover-letter title vs main paper title.** `jmlr_coverletter.tex:13–14` reads *"Attention, Transformers, and Backpropagation are Degenerate Limits of the Variational Free Energy Principle"*. The main paper title in `GL(K)_attention.tex` is *"Attention as Gauge-Theoretic Variational Inference"*. The `.bib` even contains a third title under `Dennis2025atten` ("Attention, Transformers, and Backpropagation are Degenerate Limits …"). Resolve to one. If the cover letter intentionally pitches the broader framing, add a one-line bridge ("our submission is titled X; we frame it as Y for the broader audience"). Otherwise update the cover letter.
14. **U14 — `tikz.tex:1` missing backslash on `documentclass`.** Already flagged in `manuscript_index.md`. If `tikz.tex` is `\input{}`-ed elsewhere, leave it. Otherwise prepend the missing `\`.

## Cross-paper MR-N references

`MR-N` markers do not appear in `.tex` source (grep returned no matches in any manuscript file). They live exclusively in commit messages and in `Attention/REVIEW_2026-05-18.md`. The MR table relevant to Participatory:

| MR-id | Status per `REVIEW_2026-05-18.md` | Cross-paper relevance |
|---|---|---|
| MR-1 (symbol overloads `α_i, κ, s, τ`) | Resolved for Participatory: `α_i` renamed (pass 8, commit `89e7982d`); `κ`, `s`, `τ` documented at `Participatory:178–189` (pass 9, commit `f9ada1dd`) | Main paper has the same `α_i` form (E6) so the rename is consistent. **`κ` is not similarly disambiguated in `GL(K)_attention.tex`** — main paper uses bare `κ` as the attention temperature scalar where Participatory uses it as the Killing form. See N4. Consider extending the Notation paragraph into the main paper. |
| MR-2 (`τ = κ√K` missing) | Resolved in Participatory at line `1241` (pass 1) | **Not yet resolved in the main paper.** See U10. |
| MR-3 (`λ_h` undefined) | Closed in pass 11 (commit `738bbd84`) for Participatory | Main paper does not use `λ_h` at all because its F omits the model channel; no parallel issue there. |
| MR-4, MR-5, MR-6 | Resolved in passes 1, 10 for Participatory | Internal to Participatory; not cross-paper. |
| MR-7 (self-coupling `i=j` in attention entropy) | Status not in `REVIEW_2026-05-18.md` Section 16 closure table | Same potential issue exists in the main paper's F (Eq `eq:free_energy_final`). Worth a parallel review. |
| MR-8 (`Ω_ik^{-1} = Ω_ki` cocycle in mass-block transpose check) | Status not in closure table | Specific to Participatory's Eq. 1665 area; not a cross-paper drift. |

## Style drifts (per-manuscript counts)

Banned phrases per `style_constraints.md` (case-insensitive; sentence-position not filtered, so review individually before mass-editing):

| Banned phrase / pattern | Participatory | GL(K)_attention | GL(K)_supplementary | belief_inertia |
|---|---|---|---|---|
| `key insight` / `crucially` / `critically` / `notably` / `importantly` / `it's worth noting` / `interestingly` / `fundamentally` / `in particular` / `leverages` / `underscores` (combined) | 5 | 0 | 0 | 8 |
| Banned LaTeX `\,` | 7 | 169 | 119 | 34 |
| Banned LaTeX `\;` | 0 | 40 | 17 | 0 |
| Banned LaTeX `\!` | 0 | 67 | 24 | 1 |
| Horizontal rules (`---`, `\hrule`, `\rule{}{}`) | 0 | 0 | 0 | 0 |
| Self-referential drafting language ("earlier draft" etc.) | **2** (`Participatory:189, 2068`) | 0 | 0 | 0 |

Per-manuscript banned-phrase instances in Participatory:

- `Participatory:114` "in particular" (×1, in a paragraph defining pan-agentic scope; consider deleting)
- `Participatory:1638` "in particular" (in α_i product-rule discussion)
- `Participatory:3164` "in particular" (in equivalence-principle discussion)
- `Participatory:3190` "(in particular we are not claiming …)" — parenthetical; lower-priority
- `Participatory:3321` "in particular" (Norton analysis discussion)

belief_inertia banned phrases:

- `belief_inertia_unified.tex:195` "Crucially"
- `belief_inertia_unified.tex:305` "fundamentally"
- `belief_inertia_unified.tex:439` "critically" (regime descriptor, technical use — likely keep)
- `belief_inertia_unified.tex:622` "critically" (regime descriptor)
- `belief_inertia_unified.tex:624` "characteristic of a damped harmonic oscillator. The measured decay times match …" (no banned phrase actually triggered after re-check)
- `belief_inertia_unified.tex:1219` "Critically"
- `belief_inertia_unified.tex:1967` "critically damped" (regime — keep)
- `belief_inertia_unified.tex:2085` "crucially"

The main paper and the supplementary are clean on banned phrases. The supplementary and main paper have the heaviest banned-spacing-macro load by a large margin and should be cleaned in a dedicated pass; Participatory is already nearly clean on banned macros (7 occurrences of `\,`, zero `\;` or `\!`).

## Narrative-consistency findings

### NC1. The F functional in the main paper is a strict subset of Participatory's F; the relationship is asymmetrically declared.

**Main paper's F** (Eq. `eq:free_energy_final`, `GL(K)_attention.tex:844–852`) is the reduced single-channel form `Σ_i KL(q_i ‖ p_i) − τ Σ_i log Z_i − E_q[log p(o|·)]`. The model channel `KL(s_i ‖ r_i)` and meta-attention `γ_ij` are deferred to an appendix and stated as "an analogous expression holds" at `GL(K)_attention.tex:677`.

**Participatory's F** (Eq. `eq:free_energy_functional_final`, `Participatory:1248–1259`) treats belief channel and model channel as co-equal, weights the model channel by `λ_h`, and threads both through the variational hierarchy `r → s → p → q → o`.

**Cross-declaration:**
- Participatory acknowledges the relationship at `Participatory:625` ("We follow the convention of the gauge-theoretic transformer companion paper~\cite{Dennis2025trans}") and again at `Participatory:1284, 1328` and many other sites.
- The main paper does **not** explicitly point at Participatory as the broader framework. It cites `Dennis2025it` at `GL(K)_attention.tex:665, 2271` only for the Regime II / vertex-edge connection construction, not as the broader-F superset.

**Recommended resolution:** add one sentence to the main paper's F discussion (e.g., near `GL(K)_attention.tex:677`) saying "the symmetric (belief + model) two-channel form, with hyper-prior weight $\lambda_h$, is developed in the companion participatory framework~\cite{Dennis2025it} (their Eq. [`eq:free_energy_functional_final`]); the form used here is its specialization to the unit-weight single-channel limit relevant for language modelling."

### NC2. `τ` is overloaded in `belief_inertia_unified.tex` but not in the other manuscripts.

In `belief_inertia_unified.tex`, `τ` is used as:
- attention temperature (line `157`, matching the main and Participatory canon),
- proper time / Fisher arc-length (`173, 192`),
- mechanical decay time `τ = M/γ` (`433, 485, 602, 646, 663, 1144`),
- a flow-time parameter in a forward-Euler iteration (`734, 746`).

This is a within-paper overload only — it doesn't propagate to other manuscripts. But it does mean that any reader bundling all four manuscripts and grepping for `τ` will get inconsistent meaning. Participatory's `Notation and symbol conventions` paragraph at `Participatory:178–189` recognizes this pattern and resolves it with subscripts (`τ_i` for the information clock; `τ^{(q)}_{ij}` for coupling strengths in the appendix). The same disambiguation discipline should be applied to `belief_inertia_unified.tex`: use `τ_M = M/γ` for the mechanical decay time and reserve bare `τ` for the attention temperature, or document the overload explicitly at the top of the manuscript.

### NC3. `κ` is the same letter for three structurally distinct objects across the manuscript set.

- Bare `κ` in CLAUDE.md and in the main paper's symbol table (`GL(K)_attention.tex:409`) is the learnable attention temperature scalar.
- Bare `κ` in `Participatory:185` (and used in the body) is the Killing form on the gauge Lie algebra (negative-definite for compact forms; indefinite for `gl(K)`).
- Bare `κ` in `belief_inertia_unified.tex:842–971` is the bounded-confidence temperature.

Each manuscript is internally consistent and each documents its own convention (Participatory explicitly at the Notation paragraph; belief_inertia by usage; main paper by the symbol table). The cross-paper user who reads them together will need to know which paper they are in. Recommend adding a single line in each paper at first use of `κ` saying "`κ` denotes [the local object]; see also `[companion paper]` where the same letter denotes [the other object]."

### NC4. Vaswani 2017 is not cited in Participatory's attention-recovery section.

`Participatory:1800–1836` derives scaled dot-product attention as a degenerate limit. The recovery derivation does not cite `Vaswani2017` or `vaswani2017attention` anywhere. This is a citation gap independent of the bibkey duplication issue (C2). The Vaswani paper is the standard reference for the recovered form and should be cited at the point of recovery. The main paper does cite `vaswani2017attention` at the analogous derivation point (`GL(K)_attention.tex:1699`).

## EM mode / config terminology

`em_mode` values (`'ift_phi' | 'em_phi_q' | 'em_phi_p' | 'vfe_default'`), `skip_attention`, `use_prior_bank`, and other config switches from CLAUDE.md do **not** appear in any of the manuscript files (grep returned no matches for `em_mode`, `ift_phi`, `em_phi_q`, `em_phi_p`, `vfe_default`, `skip_attention`, `use_prior_bank` in `Attention/*.tex`). No cross-manuscript drift on EM-mode terminology because none of the manuscripts surface that vocabulary. If any manuscript intends to describe the implementation's EM-boundary semantics, the absence of these terms is itself a finding — but no manuscript currently does, so this is out of scope here.

## Open questions

- **OQ1.** Is the asymmetry in NC1 intentional? If the main JMLR paper is meant to read as standalone for an ML audience while Participatory is the broader framework, the asymmetric cross-reference may be deliberate. The recommendation in U10 / NC1 should be confirmed with the user before applying.
- **OQ2.** Is the `Dennis2025` bare key in `belief_inertia_unified.tex` intended to be `Dennis2025trans` (the JMLR submission) or `Dennis2025it` (Participatory)? Context at line `2098` says "adapted from the GL(K) attention framework", suggesting `Dennis2025trans`. Confirm before fixing U7.
- **OQ3.** Is the cover-letter title (U13) intentionally different from the main paper title to pitch the broader framing, or is one stale?
- **OQ4.** Should `κ_β, κ_γ` in Participatory's appendix be renamed to `τ_β, τ_γ` (to match the bare-`τ` of the body), or should the body's `τ` be renamed to `κ` (matching CLAUDE.md and the main paper) — and the Killing form moved to `K_g` or `B`? The two are both legitimate fixes for the N5 drift; the user's preference determines the surgical direction.

## Summary

- Total notation drifts: 10 (N1–N10). Two Minor on macro choices, one Major on `κ` overload (N4), one Major on `κ_β/κ_γ` vs `τ` in the Participatory appendix (N5), one Major on `τ` overload in belief_inertia (N6), one Major on `λ_h` and model-channel scope (N10).
- Total equation drifts: 2 Major (E3: appendix vs body κ_β/τ inconsistency in Participatory; E4: main paper missing `τ = κ√K` statement). 0 sandwich-product or transport-factorization issues — the load-bearing equations are consistent across all four manuscripts.
- Total citation drifts: 7 (C1–C7 + C10). **6 Major bibkey duplications** (Friston 2010, Vaswani 2017, Parr 2022, Amari 2016, Nakahara 2003, Frankel 2011), **1 Critical broken citation** (`\citep{Dennis2025}` in `belief_inertia_unified.tex`), 1 missing citation (Vaswani 2017 in Participatory's attention-recovery derivation).
- MR-N status: MR-1, MR-2, MR-3, MR-4, MR-5, MR-6 closed for Participatory per `Attention/REVIEW_2026-05-18.md`. MR-2 still open for the main paper (no `τ = κ√K` statement).
- Style: Participatory has the lowest banned-LaTeX-macro count of the four; cleanest by a wide margin. Banned-phrase count 5 in Participatory, 8 in belief_inertia, 0 in main and supplementary. Two "earlier draft" self-references in Participatory (`189, 2068`) violate the `no self-referential history` rule.
- **Highest-severity item:** **C8 — three broken `\citep{Dennis2025}` calls in `belief_inertia_unified.tex` (lines 139, 2098)**. These will compile as `[?]` and ship in a submitted manuscript. Fix before any further build.
- **Recommended next action:** apply U1–U7 (canonical bibkeys + broken-citation fix) as a single bibliography-cleanup commit, then U8 (shared preamble), then U10 (`τ = κ√K` in main paper). The notation harmonization in U9 and the appendix-vs-body κ/τ cleanup in U11 can wait for a separate pass. U12 ("earlier draft" removal) is a 30-second edit; do it now.
