"""
EM mode definitions and inference.

Maps the single-string em_mode selector to the internal flag set that controls
gradient flow at the EM boundary. Extracted from block_config.py to decouple
variational_ffn.py from the config module.

See CLAUDE.md for the em_mode table and semantics.
"""


EM_MODE_TABLE = {
    'straight_through': dict(amortized_inference=True,  amortize_sigma=True,  exact_phi_grad=False, implicit_em=False, em_phi_mode='amortized'),
    'ift_phi':          dict(amortized_inference=True,  amortize_sigma=True,  exact_phi_grad=True,  implicit_em=False, em_phi_mode='amortized'),
    'em_phi_q':         dict(amortized_inference=True,  amortize_sigma=False, exact_phi_grad=False, implicit_em=False, em_phi_mode='E_phi_q'),
    'em_phi_p':         dict(amortized_inference=True,  amortize_sigma=False, exact_phi_grad=False, implicit_em=False, em_phi_mode='M_phi_p'),
    'implicit_ift':     dict(amortized_inference=False, amortize_sigma=False, exact_phi_grad=False, implicit_em=True,  em_phi_mode='amortized'),
}


def infer_em_mode(config: dict) -> str:
    """Resolve em_mode from config dict, with backward compatibility for old flags."""
    if 'em_mode' in config:
        return config['em_mode']
    # Legacy flag inference
    implicit_em = config.get('implicit_em', False)
    amortized = config.get('amortized_inference', True)
    if implicit_em:
        return 'implicit_ift'
    # Legacy non-amortized path (Hebbian) — detaches beliefs like implicit_ift
    # but without the IFT scaling. Kept for backward compatibility only.
    if not amortized:
        return 'implicit_ift'
    em_phi_mode = config.get('em_phi_mode', 'amortized')
    if em_phi_mode == 'E_phi_q':
        return 'em_phi_q'
    if em_phi_mode == 'M_phi_p':
        return 'em_phi_p'
    exact = config.get('exact_phi_grad', False)
    if exact:
        return 'ift_phi'
    return 'straight_through'
