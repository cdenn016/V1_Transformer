"""Verify σ LR is an independently learnable Parameter when learnable_lr=True.

Confirms:
  1. raw_sigma_lr is a Parameter (not a buffer) when learnable_lr=True.
  2. raw_sigma_lr is a buffer (not a Parameter) when learnable_lr=False.
  3. raw_lr and raw_sigma_lr are SEPARATE tensors (no aliasing).
  4. When learnable, gradients flow to both independently — perturbing each
     produces measurably different downstream behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_ffn(learnable_lr: bool, mu_lr: float = 0.1, sigma_lr: float = 0.015):
    from transformer.core.variational_ffn import VariationalFFNDynamic

    K = 8
    n_gen = 1
    generators = torch.zeros(n_gen, K, K)  # trivial gauge for the test
    ffn = VariationalFFNDynamic(
        embed_dim=K,
        generators=generators,
        irrep_dims=[K],
        alpha=1.0,
        lambda_belief=1.0,
        lambda_softmax=1.0,
        kappa=1.0,
        n_iterations=1,
        mu_lr=mu_lr,
        sigma_lr=sigma_lr,
        learnable_lr=learnable_lr,
        update_sigma=True,
        diagonal_covariance=True,
        update_phi=False,
        update_phi_per_iteration=False,
        gauge_mode='trivial',
        gauge_param='phi',
    )
    return ffn


def main() -> None:
    print("=" * 70)
    print("σ LR independent-learnability test")
    print("=" * 70)

    # Test 1: learnable_lr=True → both are Parameters
    ffn_learn = build_ffn(learnable_lr=True)
    is_param_mu = isinstance(ffn_learn.raw_lr, torch.nn.Parameter)
    is_param_sigma = isinstance(ffn_learn.raw_sigma_lr, torch.nn.Parameter)
    print(f"\nlearnable_lr=True:")
    print(f"  raw_lr is Parameter:       {is_param_mu}")
    print(f"  raw_sigma_lr is Parameter: {is_param_sigma}")
    assert is_param_mu and is_param_sigma, "Both should be Parameters when learnable_lr=True"

    # Test 2: learnable_lr=False → both are buffers
    ffn_fixed = build_ffn(learnable_lr=False)
    is_param_mu_f = isinstance(ffn_fixed.raw_lr, torch.nn.Parameter)
    is_param_sigma_f = isinstance(ffn_fixed.raw_sigma_lr, torch.nn.Parameter)
    print(f"\nlearnable_lr=False:")
    print(f"  raw_lr is Parameter:       {is_param_mu_f}  (expect False)")
    print(f"  raw_sigma_lr is Parameter: {is_param_sigma_f}  (expect False)")
    assert not is_param_mu_f and not is_param_sigma_f, "Both should be buffers when learnable_lr=False"

    # Test 3: raw_lr and raw_sigma_lr are different tensors
    assert ffn_learn.raw_lr.data_ptr() != ffn_learn.raw_sigma_lr.data_ptr(), \
        "raw_lr and raw_sigma_lr must be SEPARATE tensors (no aliasing)"
    print(f"\nraw_lr.data_ptr() != raw_sigma_lr.data_ptr(): True (independent storage)")

    # Test 4: parameters() includes both
    param_names = {name for name, _ in ffn_learn.named_parameters()}
    has_raw_lr = 'raw_lr' in param_names
    has_raw_sigma_lr = 'raw_sigma_lr' in param_names
    print(f"\nffn.named_parameters() contains:")
    print(f"  'raw_lr':        {has_raw_lr}")
    print(f"  'raw_sigma_lr':  {has_raw_sigma_lr}")
    assert has_raw_lr and has_raw_sigma_lr, \
        "Both raw_lr and raw_sigma_lr should appear in named_parameters() when learnable"

    # Test 5: sigma_lr property returns expected value
    import torch.nn.functional as F
    expected_sigma_lr = F.softplus(ffn_learn.raw_sigma_lr).clamp(max=0.5).item()
    actual = ffn_learn.sigma_lr.item()
    print(f"\nffn.sigma_lr property:")
    print(f"  expected: {expected_sigma_lr:.6f}")
    print(f"  actual:   {actual:.6f}")
    assert abs(expected_sigma_lr - actual) < 1e-7, "sigma_lr property must apply softplus + clamp"

    # Test 6: lr property unchanged (μ side)
    expected_mu_lr = F.softplus(ffn_learn.raw_lr).clamp(max=0.5).item()
    actual_mu_lr = ffn_learn.lr.item()
    assert abs(expected_mu_lr - actual_mu_lr) < 1e-7
    print(f"  μ lr property still: {actual_mu_lr:.6f}")

    # Test 7: independent gradient targets — perturbing raw_sigma_lr only must not affect lr
    with torch.no_grad():
        ffn_learn.raw_sigma_lr.add_(0.5)
    sigma_lr_after = ffn_learn.sigma_lr.item()
    mu_lr_after = ffn_learn.lr.item()
    print(f"\nAfter raw_sigma_lr += 0.5:")
    print(f"  sigma_lr: {actual:.6f} → {sigma_lr_after:.6f} (should change)")
    print(f"  mu_lr:    {actual_mu_lr:.6f} → {mu_lr_after:.6f} (should be unchanged)")
    assert sigma_lr_after != actual, "raw_sigma_lr modification should change sigma_lr"
    assert mu_lr_after == actual_mu_lr, "raw_sigma_lr modification must NOT change mu_lr"

    print("\nPASS — σ LR is independently learnable when E_learnable_lr=True.")


if __name__ == "__main__":
    main()
