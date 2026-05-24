"""Full-covariance exact-decode conditioning (audit 2026-05-24 full-cov).

At large ‖φ‖ the gauge transform `exp(φ·G)` has a huge dynamic range, and the
fp32 sandwich `A diag(s) Aᵀ` loses positive-definiteness to catastrophic
cancellation (genuinely indefinite, min eigenvalue ≈ −0.1 at φ-scale 40). The
old `_decode_exact_full_cov` did a single `cholesky(Σ + 1e-6·I)`, which cannot
lift a large negative eigenvalue, so decode raised `_LinAlgError` mid-forward.
The fix projects each prior block back onto the SPD cone (symmetrize → eigh →
clamp → reconstruct) before the Cholesky, so decode stays finite.

NOTE: this hardens decode against operator misuse / runaway φ; at this scale the
prior is numerically corrupt, so the logits are robust-but-not-useful — it is no
substitute for `phi_trace_clamp` in a real training run.
"""
import torch

from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel


def _full_cov_bank(scale_phi: float):
    cfg = VFEConfig(
        vocab_size=32, embed_dim=8, irrep_spec=[("fund", 2, 4)],
        diagonal_covariance=False, exact_full_cov_decode=True,
        gauge_fixed_priors=True, use_prior_bank=True, use_rope=False,
    )
    model = VFEModel(cfg)
    bank = model.prior_bank
    with torch.no_grad():
        bank.phi_embed.weight.mul_(scale_phi)  # drive the sandwich indefinite
    bank._decode_cache = None
    return cfg, bank


def test_full_cov_exact_decode_finite_under_runaway_phi():
    torch.manual_seed(0)
    cfg, bank = _full_cov_bank(scale_phi=40.0)  # prior covariance goes indefinite here
    B, N, K = 2, 4, cfg.embed_dim
    mu_q = torch.randn(B, N, K)
    A = torch.randn(B, N, K, K)
    sigma_q = A @ A.transpose(-1, -2) + 0.1 * torch.eye(K)  # SPD full-cov posterior
    logits = bank.decode(mu_q, sigma_q)  # routes to _decode_exact_full_cov
    assert logits.shape == (B, N, cfg.vocab_size)
    assert torch.isfinite(logits).all(), (
        "exact full-cov decode produced non-finite logits / raised under runaway "
        "‖φ‖ — the prior SPD projection before Cholesky is missing."
    )
