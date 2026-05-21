# Claim — single-config-mutual-exclusion

**Mode:** implementation
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto (transformer/vfe/config.py, transformer/vfe/omega_direct.py, transformer/vfe/prior_bank.py, PIFB:1619-1626, PIFB:2538-2580)
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge\

## Claim

There exists a single `VFEConfig` instance that simultaneously realizes both (a) the canonical group-level Lie-group natural-gradient retraction `Ω^{t+1} = Ω^t · exp(-η · ∇̃F)` of PIFB:2566-2570 (`eq:gauge_group_retraction`), and (b) the exact (0,2)-tensor sandwich covariance transport `Σ → Ω Σ Ω^T` of PIFB:1619-1626 / Nakahara 2003 §10.3, without diagonal approximation, anywhere in the forward path (E-step, encode, decode).

## User context

This is a follow-up to the verdict in `docs/debates/2026-05-20-vfe-module-purity-for-pifb/04_verdict.md`. That verdict adopted the per-construction reading of the existential and ruled the mutual-exclusion attack irrelevant. This debate forces the single-configuration reading explicitly.

## Falsification conditions

- **Blue (defender) loses** if red exhibits *config.py* validation code that rejects any single configuration combining `gauge_parameterization='omega_direct'` with the exact (0,2)-tensor sandwich transport on Σ at all sites where Σ is transported (E-step pairwise KL kernel, encode prior bank, decode prior bank).
- **Red (attacker) loses** if blue produces a concrete `VFEConfig(...)` keyword set that constructs without raising, runs `_encode_step_decode(token_ids)` without raising, and demonstrably realizes both canonical forms (group-level Ω retraction inside the E-step inner loop AND exact `Ω Σ Ω^T` transport at every Σ-transport site — no diagonal approximation, no fall-back).
