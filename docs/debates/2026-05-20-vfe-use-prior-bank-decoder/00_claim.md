# Claim — vfe-use-prior-bank-decoder

**Mode:** implementation
**Rounds:** 2
**Judge:** on
**Evidence scope:** auto
**Canon location:** C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\.claude\agents\vfe-knowledge

## Claim

The theory and implementation of `use_prior_bank=True` in `transformer/vfe/` is theoretically sound and correctly implemented as a VFE-native decoder via `logits = -KL(q || pi_v)/tau`, consistent with the gauge-theoretic VFE framework's no-neural-networks constraint.

## User context

The toggle gates whether the final decode step is the gauge-orbit KL-to-prior readout (`use_prior_bank=True`) or the documented neural exception `nn.Linear(K, V)` on `mu_final` (`use_prior_bank=False`). CLAUDE.md states: "The only retained neural component is a linear output projection from K dimensions to vocabulary size (subsumed by the PriorBank decode, `logits = -KL(q || π_v)/τ`, when `use_prior_bank=True`)." The PriorBank also drives encode in both modes — the toggle controls decode only.

## Load-bearing sub-propositions

1. The implemented decode matches the stated formula `logits = -KL(q || pi_v)/tau` modulo softmax-invariant constants.
2. The decode is "VFE-native" in the sense of arising on the same Gaussian manifold used by encode and the E-step (Law 3).
3. The decode satisfies the no-neural-networks constraint — no nn.Linear / MLP / activation present on the `use_prior_bank=True` path.
4. The Gaussian KL classifier form is a recognized canonical decoder (or is derivable from canonical primitives such as Bishop's generative-classifier framework and Friston's free-energy functional).
