# Audit Memo 02 — EM-boundary autograd under `ift_phi` + `skip_attention`

LENS: Do `mu_p`, `log sigma_p`, `phi` embeddings receive gradient under
`ift_phi` + `skip_attention`? Is there a silent-freeze gap analogous to the
documented `em_phi_p` / `em_phi_q` failure?

Date: 2026-06-01. Environment: torch CPU. Config audited = LIVE production
values (skip_attention=True production path), NOT dataclass defaults.

## Verdict

**Central claim holds: `ift_phi` has NO silent-freeze gap.** Under the live
config, all three embedding parameters receive nonzero gradient from a real
forward+CE backward:

| param | grad norm | alive |
|-------|-----------|-------|
| `token_embed.mu_embed.weight`   | 1.4678 | yes |
| `token_embed.log_sigma_diag`    | 0.2363 | yes |
| `token_embed.phi_embed.weight`  | 1.3064 | yes |

The detaching-mode pathology (`em_phi_p`/`em_phi_q` freeze σ_embed/φ_embed when
`skip_attention=True` because the attention sublayer was their only autograd
path) does NOT recur for `ift_phi`. In `ift_phi` the FFN E-step is itself the
autograd path: `em_phi_mode='amortized'` ⇒ no detach at the EM boundary
(`variational_ffn.py:2790-2796`, `_em_active=False`), and the priors stay
attached (`amortized_inference=True`, `amortize_sigma=True`).

Two parameters DO receive no gradient, but neither is an EM-boundary defect —
both are decode-side dead weight (see Findings EM-1, EM-2).

## How each embedding is reached (autograd trace, CE → embedding)

Resolved FFN flags under the live config (empirically dumped):
`amortized_inference=True, amortize_sigma=True, exact_phi_grad=True,
em_phi_mode='amortized', update_phi=True, update_phi_per_iteration=True,
update_sigma=True, gauge_param='phi', gauge_mode='learned', n_iterations=1,
closed_form_e_step=False, diagonal_covariance=True`. `block.skip_attention=True`.

Decode path: `use_prior_bank=False` and `use_output_projection=False` ⇒
`model._compute_logits` returns `self.out_proj(mu_q)` (`model.py:651`) — a
linear projection on **μ only**. The final belief covariance `sigma_q` never
enters the CE loss.

- **mu_embed**: `mu_q` flows directly into `out_proj` ⇒ logits ⇒ CE. Alive.

- **phi_embed**: `phi_current = phi` (attached embedding;
  `variational_ffn.py:1637`). φ enters transport `Ω_ij = exp(φ_i)exp(−φ_j)`
  which weights/transports messages into the μ update ⇒ μ_q ⇒ logits. The
  gradient flows through the **attached base point** φ_current, NOT through the
  per-iteration ∂F/∂φ. The per-iteration φ gradient is computed on a FRESH leaf
  `phi_for_grad = phi_current.clone().requires_grad_(True)` via
  `torch.autograd.grad(..., create_graph=False)` (`variational_ffn.py:1211,
  1342-1349`); with `create_graph=False` that `grad_phi` is a **detached
  constant**. `_retract_phi(phi_current, delta_phi=-grad_phi)`
  (`variational_ffn.py:2397, 2756`) therefore moves φ_current by a constant
  offset, and the M-step gradient reaches `phi_embed` only through
  φ_current-as-base-point. This is the documented amortized / straight-through
  estimator (docstring `variational_ffn.py:1195-1207`, EM-boundary comment
  `2783-2785`), explicitly NOT a true IFT gradient — consistent with the code's
  own admission. The point for THIS lens: `phi_embed` is trained. No gap.

- **log_sigma_diag** (per-token belief σ): alive via the **μ natural
  gradient**, not via the σ output. `sigma_current = sigma` (the embedding's
  per-token σ; `variational_ffn.py:1614`) preconditions the μ update:
  `nat_grad_mu = sigma_safe * grad_mu` (diagonal path,
  `vfe_gradients.py:2208`). So log_sigma_diag ⇒ Σ_q-preconditioner ⇒ μ_q ⇒
  logits. Confirmed by control: forcing `amortize_sigma=False` (detaching
  σ_p) leaves the grad essentially unchanged (0.2363 → 0.2327), proving the
  dominant path is the initial-belief-σ → nat_grad_μ preconditioner, with the
  attached prior σ_p a secondary contributor. The **output** σ_q is dead under
  mu-only decode.

## Findings

### EM-1 (dead-weight, low) — `raw_sigma_lr` receives no gradient under live config
`transformer.blocks.0.ffn.raw_sigma_lr` has `grad is None` after backward.
`E_learnable_lr=True` makes it an `nn.Parameter` (`variational_ffn.py:651`)
backing the σ-retraction step size `self.sigma_lr` (`684-691`) →
`_sigma_step = self.sigma_lr * _decay_factor` (`1785`) → SPD retraction whose
sole output is `sigma_current`. Because the σ_q output never reaches the
mu-only linear decode, `raw_sigma_lr` is structurally unable to receive
gradient. CONTROL that isolates the cause as decode-driven (not a stray
detach): with `use_prior_bank=True` (KL decode consumes σ_q), `raw_sigma_lr`
flips `None` → nonzero (7.9e-10). Scope: dead under (mu-only decode) AND
(last/only layer). With `n_layers>1` an earlier layer's σ_q feeds the next
block's μ natural gradient and the param could come alive; live config is
`n_layers=1`, so all `raw_sigma_lr` are dead. This is the same
"allocated-but-silently-ignored" anti-pattern that `BlockConfig.__post_init__`
already raises on for `n_picard_steps` and the mixer flags. Not a correctness
bug — σ genuinely cannot affect mu-only predictions, so there is nothing to
learn.

### EM-2 (dead-code, info) — `norm2.weight/bias` receive no gradient under skip_attention
`transformer.blocks.0.norm2.{weight,bias}` have `grad is None`. Expected:
`skip_attention=True` uses `norm1` for the (single) VFE sublayer
(`blocks.py:983-989`) and `norm2` is never invoked. Benign dead weight in the
state_dict for skip_attention configs; same class as EM-1.

### EM-3 (observation, info) — `ift_phi` φ gradient is amortized/straight-through, not IFT
Not a gap for this lens (φ_embed is trained), but flagged for accuracy: the
`ift_phi` mode name is historical. The per-iteration ∂F/∂φ is detached
(`create_graph=False`) and only the attached φ base point carries M-step
gradient. True IFT lives behind `use_deq=True ∧ deq_include_phi=True`
(`vfe_deq.py`). The docstrings already state this; no code change needed.

## Method / reproducibility
Tiny CPU model via `GaugeTransformerLM(cfg)` with cfg = LIVE EM_CONFIG values
(embed_dim=20, irrep_spec=[('fund',2,10)], GL(10) 2 heads d=10, n_layers=1,
ffn_n_iterations=1, vocab_size=128, max_seq_len=16, skip_attention=True,
em_mode='ift_phi', diagonal_covariance=True, gauge_param='phi',
gauge_mode='learned', gauge_group='GLK', all other live toggles as specified).
Forward on random ids (B=2,N=16), CE against random targets, `loss.backward()`,
then enumerate `named_parameters()` distinguishing nonzero-grad / zero-norm /
`grad is None`. Controls: `use_prior_bank` on/off; forced `amortize_sigma=False`.
