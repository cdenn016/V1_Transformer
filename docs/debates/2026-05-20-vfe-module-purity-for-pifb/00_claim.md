# Claim — vfe-module-purity-for-pifb

**Mode:** implementation
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (transformer/vfe/**, Attention/Participatory_it_from_bit.tex, external canon)
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge\

## Claim

There exists a configuration path (set of toggle/flag values) through the `transformer/vfe/` module that exactly realizes the variational free energy minimization framework for language modeling as constructed in `Attention/Participatory_it_from_bit.tex`, without approximation, hand-waving, missing terms, broken gauge equivariance, or theoretical gaps.

## User context

Scope restrictions:
- Language modeling aspects only — ignore other modalities (vision, etc.).
- Defaults (layernorm, residual-norm placement, etc.) are *not* under attack; the debate is about whether a theoretically pure path *exists* under appropriate toggles.
- Teams must consult `Attention/Participatory_it_from_bit.tex` for the theoretical/mathematical construction the code must implement.
- Teams must perform literature searches and cite canonical sources (Friston, Amari, Nakahara, Vaswani et al.) — the manuscript is the claim under evaluation, not the canon.
- The `transformer/vfe/` package is the canonical "pure" path per CLAUDE.md and recent project memory; the legacy `transformer/core/variational_ffn.py` path is out of scope.

## Falsification conditions (re-stated from user)

- **Blue (defender) loses** if red identifies *any single* mathematical gap, missing term, incorrect gradient flow, broken gauge equivariance, or theoretical construction in `participatory_it_from_bit.tex` that has *no corresponding code path* in `transformer/vfe/` under any toggle setting.
- **Red (attacker) loses** if blue can demonstrate, for *every* alleged gap, an explicit code path + config setting in `transformer/vfe/` that implements the construction exactly as written.

## Sub-claim structure

This is a compound claim of the form "for every construction X in PIFB, ∃ a toggle setting that realizes X in transformer/vfe/". The load-bearing proposition is the existential — a single counterexample (one construction X with no realizing toggle) falsifies blue.
