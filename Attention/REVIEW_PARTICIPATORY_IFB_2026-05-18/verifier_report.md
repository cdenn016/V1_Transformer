# Verifier Report — Participatory_it_from_bit.tex Review

Date: 2026-05-18. Each verdict is re-derived from the primary source.

## Cross-Manuscript Consistency Agent

[1] CONFIRMED — `belief_inertia_unified.tex:139` reads (mid-line): "The natural gauge group for KL-based belief coupling is $G = \mathrm{GL}(d)$, the general linear group of invertible $d \times d$ matrices \citep{Dennis2025}." And `:2098` reads: "This derivation, adapted from the $\mathrm{GL}(K)$ attention framework \citep{Dennis2025}, establishes the coupling as posterior inference over which neighboring agent generated the focal agent's latent state." Both contain bare `\citep{Dennis2025}` keys.

[2] PARTIAL — Friston duplicate CONFIRMED at lines 562 (`@article{Friston2010,`) and 2293 (`@article{friston2010free,`). Amari duplicate CONFIRMED at 609 (`@book{Amari2016,`) and 2188 (`@book{amari2016information,`). Vaswani duplicate CONFIRMED at 818 (`@inproceedings{Vaswani2017,`) and **2420** (`@inproceedings{vaswani2017attention,`) — agent reported line 2403, off by ~17 lines. Nakahara, Parr, Frankel duplicates also CONFIRMED at the cited line pairs (244/661, 188/580, 669/2285) per grep output. Duplicates exist as claimed; only the Vaswani line number was off.

[3] CONFIRMED — `Participatory_it_from_bit.tex:924` reads: "Under the Fisher--Rao metric, the Gaussian manifold has nonpositive sectional curvature~\cite{Amari2016}" (uses `Amari2016`). Line 931 reads: "the standard linear-pushforward statement~\cite{amari2016information} adapted from the gauge-theoretic transformer companion paper~\cite{Dennis2025trans}" (uses `amari2016information`). Both citation keys appear within 7 lines of each other.

[4] CONFIRMED — `grep -n "cite\*?\{[^}]*[Vv]aswani[^}]*\}"` against `Participatory_it_from_bit.tex` returns "No matches found." Zero `\cite{Vaswani...}` or `\cite{vaswani...}` instances in the manuscript despite the bib having two Vaswani entries.

[5] PARTIAL — `Participatory_it_from_bit.tex:189` contains: "we use disjoint symbols to avoid the notational collision **the earlier draft had**." CONFIRMED. Line 2068 contains the same phrase: "we use disjoint symbols for the two roles to avoid the notational collision **the earlier draft had**." CONFIRMED. Both passages contain self-referential drafting language per the project's "no self-referential drafting history" rule.

## Codebase Auditor Agent

[6] PARTIAL — Grep for `(ouroboros|meta_agent|cross_scale|hierarchical|participat)` (case-insensitive) returns 47 file matches — including `transformer/core/blocks.py`, `transformer/train.py`, `transformer/analysis/holonomy.py`, `transformer/core/variational_ffn.py`, `transformer/core/attention.py`, `transformer/baselines/standard_transformer.py`, `scripts/gauge_frame_spectral_analysis.py`, `README.md`. Most are textual mentions/comments rather than dedicated modules, but the claim "NO file matching" is REFUTED in the strict sense — these terms appear in many code/doc files. There is, however, no dedicated `ouroboros.py` or `meta_agent.py` module file (those keywords don't appear in filenames, only contents).

[7] CONFIRMED — `glob **/Fig_[4-8].png` returns "No files found". Broader glob `**/Fig*.png` also returns "No files found" in repo. Only lowercase `fig_*.png` files exist (e.g., `Attention/figs/fig_scaling_main.png`). The manuscript's `Fig_4.png` through `Fig_8.png` references have no corresponding files.

[8] CONFIRMED — `transformer/vfe/stack.py:85-94` reads:
```
rho_mu = self.prior_handoff_rho
if rho_mu == 1.0:
    new_prior_mu = beliefs.mu
else:
    new_prior_mu = (1 - rho_mu) * priors.mu + rho_mu * beliefs.mu
# Σ handoff: blend posterior with embedding to prevent cascade
rho_sigma = self.prior_handoff_sigma
if rho_sigma == 0.0:
    new_prior_sigma = initial_priors.sigma  # frozen (legacy default)
```
No Ω transport is applied — `beliefs.mu` is copied directly into `new_prior_mu`. CONFIRMED gauge-blind handoff.

[9] CONFIRMED — `_apply_rope` at `transformer/core/transport_ops.py:92` (and `transformer/pure_vfe/inference.py:62`) only rotates μ. The docstring states explicitly (line 101): "This implementation rotates only μ, not the covariance Σ." `rope_full_gauge` default in `transformer/vfe/config.py:100` reads: `rope_full_gauge: RopeFullGaugeMode = 'off'`. User's `train_vfe.py:83` also explicitly sets `'rope_full_gauge': 'off'`. CONFIRMED.

[10] PARTIAL — `transformer/vfe/train_vfe.py` config: `n_layers=1` CONFIRMED (line 45), `use_prior_bank=False` CONFIRMED (line 34), `kappa=1.0` CONFIRMED (line 58), `learnable_kappa=False` CONFIRMED (line 37). K=20 claim PARTIAL — `embed_dim=20` (line 28) with `irrep_spec=[('fund', 2, 10)]` (line 29). Per `vfe/config.py:246`, total dim is `sum(mult * dim) = 2 * 10 = 20`, so K_total = 20 is correct; but the underlying irrep block dim is 10, not 20. Depending on what "K" denotes, both readings are defensible — embed_dim/K=20 is confirmed.

## Manuscript Reviewer Agent

[11] PARTIAL — Total `\;|\,|\!` count in `Participatory_it_from_bit.tex` is **3** (confirmed via `grep -c`). All three are `\,` (zero `\;` or `\!`). Locations: lines **699, 707, 1459** — NOT line 1638 as claimed. Line 1459 is the table caption: "`($\mathrm{kg}\,\mathrm{m}^2\,\mathrm{s}^{-1}$, equivalently $J\,s$)`" — contains two `\,` instances. Agent had the right count and two of three line numbers but missed line 1459 and wrongly cited line 1638.

[12] CONFIRMED — `Participatory_it_from_bit.tex:116` starts subsection "Epistemic Status" with explicit three-level disclaimer (Level 1 implemented; Level 2 mathematical implementation; Level 3 speculative physical interpretation). Line 128 starts subsection `\label{sec:scope_limitations}` reading: "The reader should hold the claims that follow to a clearly delimited scope... We do not claim to have derived quantum mechanics or general relativity from first principles, and several of the speculative results are demonstrated only at the level of worked examples." Both passages explicitly disclaim the IFB framing.

[13] CONFIRMED — `Participatory_it_from_bit.tex:2607` reads exactly: `\subsection{It From Bit: The Pullback Construction}` followed by `\label{sec:pullback}` on line 2608. Title matches the claim.

[14] CONFIRMED — Grep for `Omega.{1,40}Sigma.{1,40}Omega` across the manuscript returns ~40 matches, every one of which follows the sandwich form `Ω Σ Ω^T` (or `Ω Σ Ω^\top`). Sampled cases: line 940 (`\Omega\Sigma_P\Omega^\top`), line 1857 (`\Omega_{ik}\Sigma_k\Omega_{ik}^T`), line 4180 (`(\Omega_{ij}\Sigma_j\Omega_{ij}^\top)^{-1}`), line 4221 (`(\tilde{\Omega}_{ij}\Sigma_{s,j}\tilde{\Omega}_{ij}^\top)^{-1}`). No missing-transpose or reversed-order variants observed.

## Overall Assessment

The **manuscript-reviewer** had the highest accuracy on structural/textual claims (epistemic status, sandwich-product consistency, section title at 2607 all confirmed verbatim; only one banned-macro line number was wrong — line 1459 not 1638). The **cross-manuscript-consistency** agent's substantive findings all hold (Dennis2025 bare keys, six bibliography duplicates, mixed Amari2016/amari2016information citation keys, zero Vaswani citations despite duplicate bib entries, self-referential drafting language) with one off-by-17 line number on the Vaswani bib entry. The **codebase-auditor** agent's most actionable claims (gauge-blind handoff in `stack.py:85-94`, RoPE rotates only μ, missing `Fig_4-8.png`, user's config values) all hold, but the "NO file matching ouroboros/meta_agent/etc." claim is too strong — 47 files mention those keywords (no dedicated module files, but pervasive textual mentions). All three reports are **actionable**: the user should treat the bibliography duplicates, missing figures, gauge-blind handoff, banned-macro locations, and self-referential drafting language as confirmed defects to fix, while spot-checking specific line numbers (Vaswani bib line, banned-macro line 1638→1459) before edits.
