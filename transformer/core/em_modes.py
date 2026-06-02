"""
EM mode definitions.

Maps the single-string em_mode selector to the internal flag set that controls
gradient flow at the EM boundary. Extracted from block_config.py to decouple
variational_ffn.py from the config module.

Three modes supported:
    'ift_phi'  : amortized μ,σ + amortized φ gradient via a single
                 `torch.autograd.grad` on a fresh `phi_for_grad` leaf
                 (default; mathematically pure path under
                 ``skip_attention=True``). NOTE: this is amortized
                 inference in the [BaiKolterKoltun2019] sense, NOT a true
                 implicit-function-theorem gradient. The `ift_phi` label
                 is the historical mode name kept for back-compat with
                 saved configs and is not renamed here.
    'em_phi_q' : amortized μ; σ and φ in q; all detached at EM boundary.
    'em_phi_p' : amortized μ; σ amortized; φ frozen in E-step (M-step only).

See CLAUDE.md for the em_mode table and semantics.
"""


EM_MODE_TABLE = {
    'ift_phi':  dict(amortized_inference=True, amortize_sigma=True,  exact_phi_grad=True,  em_phi_mode='amortized'),
    'em_phi_q': dict(amortized_inference=True, amortize_sigma=False, exact_phi_grad=False, em_phi_mode='E_phi_q'),
    'em_phi_p': dict(amortized_inference=True, amortize_sigma=False, exact_phi_grad=False, em_phi_mode='M_phi_p'),
}
