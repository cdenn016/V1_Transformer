# EM Modes — User's Gradient-Flow Machinery

**This file describes user-specific machinery in this codebase.** It is descriptive — the agents use it to know what the code is doing and to evaluate that against the standard EM / IFT literature ([Friston2010], [BaiKolterKoltun2019], [BleiKuckelbirgJordan2017]; see `external_canon_inference.md` §5). It is not theoretical canon.

The single-string selector `em_mode` on `BlockConfig` controls gradient flow at the EM boundary. Replaces the user's earlier 5-flag system (`amortized_inference`, `amortize_sigma`, `exact_phi_grad`, `implicit_em`, `em_phi_mode`).

## The table (from CLAUDE.md)

| `em_mode`             | μ_p        | σ_p              | φ                       | At EM exit         |
|-----------------------|------------|------------------|-------------------------|--------------------|
| `'ift_phi'` (default) | attached   | attached         | full IFT                | attached           |
| `'em_phi_q'`          | detached   | detached         | evolves in E-step       | all detached       |
| `'em_phi_p'`          | detached   | detached         | frozen in E-step        | μ,σ detached       |
| `'vfe_default'` (transformer/vfe/) | attached | frozen at embedding¹ | full autograd | attached |

¹ The `transformer/vfe/` package operates in its **own gradient profile** that does not correspond to any of the three `em_mode` values above. Specifically:

- `mu_p` is attached (the previous layer's posterior `μ_q` becomes the next layer's `μ_p` via `vfe/stack.py`).
- `sigma_p` is structurally frozen at the embedding value when `prior_handoff_sigma=0` (the default).
- `phi` is updated each E-step iteration via `_update_phi` in `vfe/e_step.py`, which constructs a fresh detached leaf `phi_for_grad = phi.detach().requires_grad_(True)` for the local alignment-loss backward, then retracts `phi` and reassigns it.
- The iteration sequence as a whole is autograd-tracked, but `phi` is **not globally cloned per outer iteration**.
- The `vfe/` package does **not currently honor `em_mode` switching** — it is hardwired to this profile. Selecting any other `em_mode` requires routing through the legacy `transformer/core/variational_ffn.py` path.

## Attention sublayer interaction (CRITICAL — silent freeze trap)

The pure VFE architecture (`skip_attention=True`) is the theoretically clean form: the FFN's E-step computes its own β internally and updates beliefs. The separate attention sublayer at the top of `GaugeTransformerBlock.forward` is an **engineering heuristic** (β-weighted message aggregation through `W_O · μ_agg` residual).

- `skip_attention=True` works cleanly with `em_mode='ift_phi'` (default).
- `skip_attention=True` is **INCOMPATIBLE** with `em_phi_p` and `em_phi_q` — those modes detach σ_p and/or φ inside the FFN, so the attention sublayer is their **sole autograd path back to `sigma_embed` and `phi_embed`**. With `skip_attention=True` AND a detaching mode, σ_embed and φ_embed **silently stay frozen at initialization**.
- `BlockConfig.__post_init__` warns when this combination is detected. **Audit check:** verify the warning is still in place and that it triggers under the bad combination.

## What the auditor must verify

For a given user config:

1. Identify `em_mode` and `skip_attention`. Find them in the active config file (not a default).
2. Trace `mu_p` through the block: is it `.detach()`ed where the table says it should be?
3. Trace `sigma_p` similarly.
4. Trace `phi` through `_update_phi` (in `vfe/e_step.py`) or the IFT path (in `core/variational_ffn.py`): does the gradient flow match the table?
5. If `skip_attention=True` AND `em_mode in {'em_phi_p', 'em_phi_q'}`: this is the silent-freeze trap. The warning in `BlockConfig.__post_init__` must fire.

## Common drift patterns to flag

- Code attaches `sigma_p` to gradient under `'em_phi_q'` — should be detached.
- Code detaches `mu_p` under `'ift_phi'` — should be attached.
- Code claims "full IFT" but uses a single-step gradient (no fixed-point Jacobian solve) — that is amortized inference, not IFT. Critical.
- Code uses `em_mode` as a runtime branch without honoring the `vfe/` package's hardwired profile — drift.

## Manuscript check

Manuscripts that describe EM modes must:

- State explicitly which mode is the default.
- Note that `vfe/` doesn't honor mode switching.
- Document the silent-freeze trap if discussing `skip_attention`.

If a manuscript claims `em_phi_p` and `skip_attention=True` "work together cleanly", that is wrong — flag as a Major issue.
