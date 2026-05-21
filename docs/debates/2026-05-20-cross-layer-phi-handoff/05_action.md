# Action — cross-layer-phi-handoff

**From verdict:** BLUE_WINS

## Recommended action

`priors.phi` has zero downstream readers in `transformer/vfe/` (grep-verified). The φ cascade across the L-block depth dimension happens via `beliefs.phi`: `block.py:172` returns updated `beliefs.phi` from the E-step; `stack.py:94-95` re-passes that BeliefState into the next block; `e_step.py:779` reads `phi = beliefs.phi` to initialize the inner loop. The named `priors.phi` slot is operationally inert and the embedding-value write at `stack.py:145` is therefore not a falsifying deviation.

The verdict's recommended action — rewrite the `stack.py:18-34` docstring as a positive statement explaining the principled σ/φ asymmetry — has been applied in this round. No `prior_handoff_phi` toggle is added (it would gate a parameter no consumer reads).

## Follow-up debates (if any)

None.
