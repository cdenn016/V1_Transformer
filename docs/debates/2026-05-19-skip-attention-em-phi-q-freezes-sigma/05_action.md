# Action — skip-attention-em-phi-q-freezes-sigma

**From verdict:** user-adjudicated

## Recommended action

Both teams independently established that the project policy assertion at `transformer/core/block_config.py:288-298` (the warning text issued when `skip_attention=True` and `em_mode='em_phi_q'`) is **too strong**. The actual behavior is:

1. **On the legacy `transformer/core/variational_ffn.py` path** — `sigma_embed` is frozen *only* when `lambda_hyper=0` AND `M_alpha=0`. Non-zero values for either route a direct autograd path to `log_sigma_diag` that bypasses the FFN E-step and the attention sublayer. Red's executable verification at `lambda_hyper=1.0` and `M_alpha=1.0` confirms each path independently.

2. **On the `transformer/vfe/` path** — `em_mode` is not honored (per `.claude/agents/vfe-knowledge/em_modes.md`), so the claim is out-of-scope for that entry point.

3. **With P-flow enabled** — `sigma_embed` receives non-autograd EMA updates from `transformer/core/hebbian.py:201` regardless of the autograd graph state. The user's only `em_phi_q` config (`HEBBIAN_CONFIG`) sets `use_p_flow=True`, so P-flow is the design-intended update rule in that configuration.

Concrete next steps:

- Verify the user's active `lambda_hyper` and `M_alpha` in the configuration actually being run (per the pre-fix protocol in CLAUDE.md).
- Decide whether to revise the warning text at `transformer/core/block_config.py:288-298` to reflect the conditional form (e.g., "freezes `sigma_embed` autograd path only when `lambda_hyper=0` AND `M_alpha=0` AND `use_p_flow=False`").
- If the documentation in CLAUDE.md is intended as the single source of truth, update the `skip_attention` / `em_mode` interaction paragraph to match the conditional form.

## Follow-up debates (if any)

- "Under `HEBBIAN_CONFIG` defaults (the only config that activates `em_mode='em_phi_q'` in the repository), does `sigma_embed` actually drift across a training run? Run for N steps, log `log_sigma_diag.std()` per step." This is an empirical sub-question that the cheap-mode debate did not fully resolve (Red flagged "bit-exact stability across multi-step run" as the one outstanding loose end).
- "Is the comment block at `transformer/train.py:553-555` accurate that the previous behavior was a bug, and the current attached `sigma_s` in `KL(s||h)` is the corrected design?" — a separate theory-mode debate would clarify whether the corrected design is consistent with the standard variational EM literature (FEP, Friston 2010).
