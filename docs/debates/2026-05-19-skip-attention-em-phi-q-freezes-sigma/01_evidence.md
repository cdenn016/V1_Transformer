# Evidence Pack — skip-attention-em-phi-q-freezes-sigma

## Active config (implementation mode)

The active VFE entry point is `transformer/vfe/train_vfe.py` (the click-to-run pattern; configuration is in the file body, not CLI). For this debate, the relevant config keys are:

- `BlockConfig.skip_attention`: declared at `transformer/core/block_config.py:482` with default `False`. The user's claim assumes this is set to `True`.
- `BlockConfig.em_mode`: declared at `transformer/core/block_config.py:153` with default `'ift_phi'`. Allowed values are `Literal['ift_phi', 'em_phi_q', 'em_phi_p']`. The user's claim assumes this is set to `'em_phi_q'`.

Both teams must verify the runtime path under these resolved values.

## Code references

### Project-policy assertion of the incompatibility

- `transformer/core/block_config.py:276-293` — `BlockConfig.__post_init__` explicitly checks the combination and issues a warning:

  ```
  _detaching_modes = {'em_phi_p', 'em_phi_q'}
  if self.skip_attention and self.em_mode in _detaching_modes:
      ... f"em_mode={self.em_mode!r} combined with skip_attention=True will "
          ... "Compatible mode for skip_attention=True: 'ift_phi'."
  ```

  This warning is the project's own assertion that the combination silently freezes σ_embed and φ_embed. The debate must verify this assertion against the actual code paths, not just take the warning at face value.

### EM-mode internal flags

- `transformer/core/em_modes.py:28` — `em_phi_q` resolves to `dict(amortized_inference=True, amortize_sigma=False, exact_phi_grad=False, em_phi_mode='E_phi_q')`.

  Note `amortize_sigma=False`. Combined with `amortized_inference=True`, what does this imply for σ's autograd path inside the FFN E-step? Both teams should check what these flags do in `transformer/core/variational_ffn.py`.

- `transformer/core/em_modes.py:29` — `em_phi_p` resolves to similar flags but with `em_phi_mode='M_phi_p'`. Out of scope for this debate but useful for reference.

### Attention sublayer gating

- `transformer/core/blocks.py:476` — `self.skip_attention = getattr(cfg, 'skip_attention', False)`.
- `transformer/core/blocks.py:795` — `if not self.skip_attention:` guards the entire attention sublayer forward. Under `skip_attention=True`, the attention sublayer's contribution to the forward pass is bypassed.
- `transformer/core/blocks.py:787-792` — comment block stating "When skip_attention=True, the VFE E-step IS the entire block".
- `transformer/core/blocks.py:980-989` — under `skip_attention=True`, `norm1` is applied instead of `norm2`, and the residual structure is different.

### W_O projection (potential autograd path)

The attention sublayer's `W_O · μ_agg` residual is the engineering heuristic that, per the project policy assertion, is the sole autograd path back to `sigma_embed` and `phi_embed` under the detaching EM modes. Both teams should:

1. Locate `W_O` in `transformer/core/blocks.py` and confirm whether `sigma_embed` flows through it.
2. Confirm whether, under `skip_attention=True`, `W_O` is reached at all (it is part of the attention sublayer).

### sigma_embed origin and use

`sigma_embed` is referenced in 15 files. The most relevant for this debate:

- `transformer/core/embeddings.py` — declaration / initialization of `sigma_embed`.
- `transformer/vfe/stack.py` — passes `sigma_embed` into the VFE stack as the prior σ.
- `transformer/vfe/trainer.py` — optimizer registration.

Both teams should confirm: (a) is `sigma_embed` an `nn.Parameter` registered with the optimizer; (b) under `skip_attention=True` + `em_mode=em_phi_q`, does any operation in the forward pass produce a gradient back to it through the autograd graph?

## Canon excerpts

This debate is implementation-mode, not theory-mode, so the primary citations are code paths. The relevant canon background from `vfe-knowledge/em_modes.md`:

- The user's table for `em_phi_q` mode states σ and μ are detached at the EM boundary. This is a structural property of the mode, not just a side effect.

`vfe-knowledge/e_step_constraints.md` may also be relevant for what counts as the "E-step boundary."

## What this evidence does NOT settle

1. The exact tensor expression that connects (or fails to connect) `sigma_embed` to the loss under the active config. This requires reading the forward pass in `variational_ffn.py` and the cross-layer cascade in `vfe/stack.py`.
2. Whether there is a non-attention path from `sigma_embed` to the loss that the project policy assertion might have missed (e.g., through the prior bank decode, or through the embedding norm in the input layer).
3. Whether `sigma_embed.requires_grad` is set to `False` somewhere under this configuration (a different freeze mechanism than autograd-path absence).
4. Whether the "freeze" claim holds for the very first forward pass too, or only for subsequent backward passes. (The claim says "silently freezes at initialization" — the debate must verify what "at initialization" means in this context.)

Both teams should treat these as the contested points the debate must resolve.
