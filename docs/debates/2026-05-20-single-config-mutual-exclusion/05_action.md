# Action — single-config-mutual-exclusion

**From verdict:** RED_WINS

## Recommended action

`gauge_parameterization='omega_direct'` requires `diagonal_covariance=True` (`transformer/vfe/config.py:566-573`); `exact_full_cov_decode=True` requires `diagonal_covariance=False` (`transformer/vfe/config.py:484-507`). The canonical group-level retraction (PIFB:2566-2570) and the canonical sandwich `Σ → Ω Σ Ω^T` (PIFB:1619-1626, Nakahara 2003 §10.3) therefore cannot coexist in a single `VFEConfig` instance.

Two paths forward, listed by surgery cost:

1. **Documentation fix (applied in this round).** Document the mutual exclusion in `CLAUDE.md` as a known structural limitation, naming both validation guards and the open `NotImplementedError` at `non_flat.py:500-505` that would close the gap. The per-construction reading (each canonical form is independently reachable under its own toggle) remains the operative reading; the single-config reading is acknowledged as currently unmet.

2. **Code fix (deferred).** Implement the open `NotImplementedError` at `transformer/vfe/non_flat.py:500-505` — a per-pair logdet-aware sandwich KL kernel — and remove the `vfe/config.py:566-573` guard. The encode-side sandwich at `prior_bank.py:326-329` already exists; only the E-step pairwise-Ω full-cov KL kernel is missing. This is a research-grade kernel addition (~200-400 LoC by analogy with the existing diagonal kernel), best taken on as its own task with derivation in a dedicated subsection.

This round applies option 1. Option 2 is recorded as a follow-up.

## Follow-up debates (if any)

None directly. The gap is settled; the choice is documentation now vs. kernel implementation later.
