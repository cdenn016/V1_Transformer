# Claim — skip-attention-em-phi-q-freezes-sigma

**Mode:** implementation
**Rounds:** 1
**Judge:** off
**Evidence scope:** auto
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge\

## Claim

When `BlockConfig.skip_attention=True` is combined with `BlockConfig.em_mode='em_phi_q'`, the `sigma_embed` parameter receives no autograd path during training and therefore stays frozen at its initialization value across the entire training run.

## User context

This claim was raised as part of a smoke test for the `red-blue-debate` skill. The combination is documented in CLAUDE.md as a known interaction that `BlockConfig.__post_init__` warns about. The debate should focus on whether the actual code path supports the claim under the active configuration, not whether the documentation says so.
