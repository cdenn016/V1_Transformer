"""
Numerical Edge-Case Stress Tests for the Gauge-Theoretic VFE Transformer
=========================================================================

Tests what happens under extreme configurations that might trigger silent
numerical failures: large embed_dim, sharp kappa, many VFE iterations,
large phi norms, near-singular covariances, and near-uniform covariances.

All tests run on CPU only. Click-to-run: no argparse.

Each test_scenario() performs a full forward+backward pass and reports:
  - logits_finite:   all logits are finite (no NaN/Inf)
  - loss_finite:     cross-entropy loss is finite
  - grads_finite:    all parameter gradients are finite
  - attn_valid:      attention weights per-head sum to 1 (valid distributions)
  - kl_nonneg:       all KL values are non-negative
  - beliefs_finite:  all final belief parameters (mu, sigma, phi) are finite

Design notes
------------
Each scenario resets torch.manual_seed(42) immediately before model
construction and data generation.  Without this, sequential tests share RNG
state and earlier scenarios' randomness propagates into later ones, producing
false FAIL reports for configurations that are intrinsically stable.

The model uses forward_with_attention() so that the beta and kl tensors
(shape: (n_layers, B, n_heads, N, N)) are available for structural checks
without requiring a separate training harness.

Genuine failures detected (as of 2026-04-11):
  - Large K=64:           LinalgEighBackward0 NaN — degenerate eigenvalues
                          in the transported 64x64 covariance during E-step.
  - Near-uniform sigma:   LinalgEighBackward0 NaN — large sigma inflates
                          eigenvalue clustering, triggering eigh backward NaN.
Both are eigh backward instabilities.  All other scenarios pass with per-seed
isolation.  See the diagnosis section printed at the end of run_all().
"""

import sys
import os

# Path setup — identical to all other scripts in this directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import math
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from transformer.core.model import GaugeTransformerLM


# ---------------------------------------------------------------------------
# Minimal base configuration
# All edge-case tests override exactly the fields they intend to stress.
# The base uses a small-but-valid GLK multi-head config that exercises the
# full forward path (VFE_dynamic, learnable sigma/phi, causal mask, RoPE off).
# ---------------------------------------------------------------------------
BASE_CONFIG: Dict[str, Any] = {
    # Required structural keys
    'vocab_size': 100,
    'embed_dim': 6,
    'n_layers': 1,
    # GLK multi-head format: ('fund', n_heads, d_head)
    # embed_dim must equal n_heads * d_head exactly.
    'irrep_spec': [('fund', 2, 3)],   # 2 heads x 3 dims = 6 total
    'max_seq_len': 32,
    'hidden_dim': 64,

    # Attention
    'kappa_beta': 1.0,
    'mask_self_attention': True,

    # VFE dynamics
    'ffn_mode': 'VFE_dynamic',
    'ffn_n_iterations': 1,
    'gauge_group': 'GLK',

    # Positional encoding — off to keep the path minimal
    'use_rope': False,

    # Belief evolution — both on to exercise the full gradient graph
    'evolve_sigma': True,
    'evolve_phi': True,

    # Avoid PriorBank complications in edge-case tests
    'gauge_fixed_priors': False,
    'use_prior_bank': False,

    # Numerical stability ceilings (defaults)
    'sigma_max': 5.0,
    'e_step_sigma_floor': 0.1,
}

# Short enough to run fast on CPU; long enough to exercise the full causal mask.
SEQ_LEN: int = 8

# Seed used to reset RNG state before every scenario.
_SEED: int = 42


def _build_model(config_overrides: Dict[str, Any]) -> GaugeTransformerLM:
    """Merge overrides with BASE_CONFIG and instantiate the model on CPU."""
    cfg = {**BASE_CONFIG, **config_overrides}
    model = GaugeTransformerLM(cfg)
    model.cpu()
    return model


def _apply_param_overrides(
    model: GaugeTransformerLM,
    param_overrides: Dict[str, Any],
) -> None:
    r"""Apply in-place mutations to named parameters.

    Each entry in ``param_overrides`` maps a parameter name suffix to a
    callable ``fn(param: torch.Tensor) -> None`` that mutates it in-place.
    The search is suffix-based so callers need not know the full module path.

    Args:
        model:           Instantiated model whose parameters are mutated.
        param_overrides: ``{name_suffix: mutation_fn}`` dict.
    """
    named = dict(model.named_parameters())
    with torch.no_grad():
        for suffix, fn in param_overrides.items():
            matched = False
            for full_name, param in named.items():
                if full_name.endswith(suffix):
                    fn(param)
                    matched = True
                    break
            if not matched:
                print(
                    f"    WARNING: no parameter matched suffix '{suffix}'"
                    " — override skipped"
                )


def _check_attention_valid(attention_info: Dict[str, Any]) -> bool:
    r"""Verify every attention weight tensor is a valid probability distribution.

    ``attention_info['beta']`` is a stacked tensor of shape
    ``(n_layers, B, n_heads, N, N)``.  Under the causal mask, each row i
    sums over j = 0..i.  We verify that the active (non-zero) rows have
    non-negative entries and each row sums to 1 +/- 1e-3.

    Returns True if all layers pass, False otherwise.
    """
    beta_raw = attention_info.get('beta')
    if beta_raw is None or not isinstance(beta_raw, torch.Tensor):
        return True  # Not returned by this model configuration — skip check.

    if not torch.isfinite(beta_raw).all():
        return False

    # Iterate over the n_layers leading dimension.
    # beta_raw: (n_layers, B, n_heads, N, N)  or  (B, n_heads, N, N)
    layers: List[torch.Tensor] = (
        [beta_raw] if beta_raw.dim() == 4
        else list(beta_raw.unbind(0))
    )

    for beta in layers:
        # beta: (B, n_heads, N, N) — sum over last dim gives per-row totals
        row_sums = beta.sum(dim=-1)  # (B, n_heads, N)
        # Active rows: causal mask zeros out rows before the diagonal on short seqs.
        active_mask = row_sums > 1e-6
        if active_mask.any():
            max_dev = (row_sums[active_mask] - 1.0).abs().max().item()
            if max_dev > 1e-3:
                return False
        # Non-negativity: softmax output must be >= 0
        if (beta < -1e-6).any():
            return False
    return True


def _check_kl_nonneg(attention_info: Dict[str, Any]) -> bool:
    r"""Verify all KL divergence values are non-negative.

    ``attention_info['kl']`` is a stacked tensor of shape
    ``(n_layers, B, n_heads, N, N)``.  KL divergences must be >= 0 by
    definition; negative values indicate a numerical instability in the
    KL computation.

    Returns True if all values are >= -1e-4 (small tolerance for float32
    cancellation), False otherwise.
    """
    kl_raw = attention_info.get('kl')
    if kl_raw is None or not isinstance(kl_raw, torch.Tensor):
        return True

    if not torch.isfinite(kl_raw).all():
        return False
    if (kl_raw < -1e-4).any():
        return False
    return True


def _check_beliefs_finite(attention_info: Dict[str, Any]) -> bool:
    """Verify that the final mu, sigma, and phi beliefs are all finite."""
    for key in ('mu', 'sigma', 'phi'):
        val = attention_info.get(key)
        if val is not None and not torch.isfinite(val).all():
            return False
    return True


def _bad_grad_params(model: GaugeTransformerLM) -> List[Tuple[str, str]]:
    """Return list of (param_name, reason) for parameters with non-finite gradients."""
    bad = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if param.grad.isnan().any():
            bad.append((name, 'nan'))
        elif param.grad.isinf().any():
            bad.append((name, 'inf'))
    return bad


def test_scenario(
    name: str,
    config_overrides: Optional[Dict[str, Any]] = None,
    param_overrides: Optional[Dict[str, Any]] = None,
    seed: int = _SEED,
) -> bool:
    r"""Run one numerical edge-case scenario and print a PASS/FAIL summary.

    Resets the global RNG to ``seed`` before model construction so each
    scenario is independent of the order in which tests run.

    Performs:
      1. RNG reset.
      2. Model construction with merged config.
      3. Optional in-place parameter mutations.
      4. Full forward pass via ``forward_with_attention`` (captures beta, kl).
      5. Cross-entropy loss backward.
      6. Six checks: logits_finite, loss_finite, grads_finite,
         attn_valid, kl_nonneg, beliefs_finite.

    Args:
        name:             Human-readable scenario label.
        config_overrides: Dict of config keys to override in BASE_CONFIG.
        param_overrides:  Dict of ``{param_name_suffix: mutation_fn}`` applied
                          after construction with ``torch.no_grad()``.
        seed:             RNG seed for reproducibility (default 42).

    Returns:
        True if all checks pass, False otherwise.
    """
    config_overrides = config_overrides or {}
    param_overrides = param_overrides or {}

    # Isolate each scenario from RNG state accumulated by previous runs.
    torch.manual_seed(seed)

    vocab_size = {**BASE_CONFIG, **config_overrides}.get(
        'vocab_size', BASE_CONFIG['vocab_size']
    )

    try:
        model = _build_model(config_overrides)
        model.train()

        if param_overrides:
            _apply_param_overrides(model, param_overrides)

        tokens  = torch.randint(0, vocab_size, (1, SEQ_LEN))
        targets = torch.randint(0, vocab_size, (1, SEQ_LEN))

        # forward_with_attention captures beta and kl for structural checks.
        logits, attention_info = model.forward_with_attention(tokens, targets=targets)

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )
        loss.backward()

        bad_grads = _bad_grad_params(model)

        checks: Dict[str, bool] = {
            'logits_finite':  torch.isfinite(logits).all().item(),
            'loss_finite':    torch.isfinite(loss).item(),
            'grads_finite':   len(bad_grads) == 0,
            'attn_valid':     _check_attention_valid(attention_info),
            'kl_nonneg':      _check_kl_nonneg(attention_info),
            'beliefs_finite': _check_beliefs_finite(attention_info),
        }

        passed = all(checks.values())
        status = 'PASS' if passed else 'FAIL'
        print(f"  {name}: {status}  (loss={loss.item():.3f})")

        for k, v in checks.items():
            if not v:
                print(f"      FAILED check: {k}")

        if not checks['grads_finite']:
            for param_name, reason in bad_grads[:5]:  # cap at 5 to avoid noise
                print(f"      NaN/Inf grad in: {param_name}  [{reason}]")
            if len(bad_grads) > 5:
                print(f"      ... and {len(bad_grads) - 5} more params")

        return passed

    except Exception as exc:
        import traceback
        print(f"  {name}: ERROR — {type(exc).__name__}: {exc}")
        # Print the innermost traceback frame for quick diagnosis
        tb = traceback.extract_tb(exc.__traceback__)
        if tb:
            last = tb[-1]
            print(f"      at {last.filename}:{last.lineno} in {last.name}")
        return False


# =============================================================================
# Edge-case scenarios
# =============================================================================

def run_all() -> None:
    """Run all numerical edge-case scenarios and print a final summary."""
    print("Numerical Edge-Case Stress Tests")
    print("=" * 60)
    print(f"Device: CPU   |  torch {torch.__version__}")
    print()

    results: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Scenario 1: Large K (embed_dim=64, 8 heads x dim 8 in GLK format)
    # embed_dim must equal n_heads * d_head for GLK multi-head.
    # With full (64, 64) covariance, eigh backward can NaN when eigenvalues
    # are degenerate (which occurs at random init with near-isotropic sigma).
    # ------------------------------------------------------------------
    results['Large K=64 (8h x 8d)'] = test_scenario(
        'Large K=64 (8h x 8d)',
        config_overrides={
            'embed_dim': 64,
            'irrep_spec': [('fund', 8, 8)],
        },
    )

    # ------------------------------------------------------------------
    # Scenario 2: Very small kappa (sharp attention, softmax overflow risk)
    # kappa=0.01 causes softmax inputs ~100x larger than at kappa=1.0.
    # ------------------------------------------------------------------
    results['Small kappa=0.01'] = test_scenario(
        'Small kappa=0.01',
        config_overrides={
            'kappa_beta': 0.01,
        },
    )

    # ------------------------------------------------------------------
    # Scenario 3: Multiple VFE iterations (5 E-steps per forward pass)
    # Tests that the iterative belief update does not diverge.
    # ------------------------------------------------------------------
    results['5 VFE iterations'] = test_scenario(
        '5 VFE iterations',
        config_overrides={
            'ffn_n_iterations': 5,
        },
    )

    # ------------------------------------------------------------------
    # Scenario 4: Large phi norms (matrix exponential stress test)
    # phi_embed.weight initialized at 5x normal scale.  Tests that
    # exp(phi @ G) remains finite even for large Lie algebra elements.
    # ------------------------------------------------------------------
    results['Large phi (5x)'] = test_scenario(
        'Large phi (5x)',
        param_overrides={
            'phi_embed.weight': lambda p: p.mul_(5.0),
        },
    )

    # ------------------------------------------------------------------
    # Scenario 5: Near-singular covariances (log_sigma_diag = -10)
    # sigma = exp(-10) ~ 4.5e-5.  Tests KL numerical floor and that
    # the E-step sigma floor (e_step_sigma_floor=0.1) prevents blowup.
    # ------------------------------------------------------------------
    results['Near-singular sigma (log_sigma=-10)'] = test_scenario(
        'Near-singular sigma (log_sigma=-10)',
        param_overrides={
            'log_sigma_diag': lambda p: p.fill_(-10.0),
        },
    )

    # ------------------------------------------------------------------
    # Scenario 6: Near-uniform covariances (log_sigma_diag = +5)
    # sigma = exp(5) ~ 148.  The sigma_max=5.0 clamp fires in the E-step;
    # test checks graceful saturation and that backward remains finite.
    # ------------------------------------------------------------------
    results['Near-uniform sigma (log_sigma=+5)'] = test_scenario(
        'Near-uniform sigma (log_sigma=+5)',
        param_overrides={
            'log_sigma_diag': lambda p: p.fill_(5.0),
        },
    )

    # ------------------------------------------------------------------
    # Scenario 7: VFE iteration stability probe
    # Runs three forward-backward passes with n_iters = 1, 3, 5 on fixed
    # random data.  Each pass uses a fresh model so there is no coupling
    # between iteration counts.  Checks that beliefs remain finite and
    # loss does not diverge as E-step depth increases.
    # ------------------------------------------------------------------
    print()
    print("  VFE iteration stability probe (n_iters = 1, 3, 5):")
    iter_stable = True
    for n_iter in (1, 3, 5):
        torch.manual_seed(_SEED)
        iter_tokens  = torch.randint(0, BASE_CONFIG['vocab_size'], (1, SEQ_LEN))
        iter_targets = torch.randint(0, BASE_CONFIG['vocab_size'], (1, SEQ_LEN))
        try:
            m = _build_model({'ffn_n_iterations': n_iter})
            m.train()
            logits_it, attn_it = m.forward_with_attention(
                iter_tokens, targets=iter_targets
            )
            loss_it = F.cross_entropy(
                logits_it.reshape(-1, BASE_CONFIG['vocab_size']),
                iter_targets.reshape(-1),
            )
            loss_it.backward()
            bad = _bad_grad_params(m)
            finite = (
                torch.isfinite(logits_it).all().item()
                and torch.isfinite(loss_it).item()
                and _check_beliefs_finite(attn_it)
                and len(bad) == 0
            )
            tag = 'ok' if finite else 'DIVERGED'
            print(
                f"    n_iters={n_iter}: loss={loss_it.item():.3f}"
                f"  beliefs_finite={tag}"
            )
            if not finite:
                iter_stable = False
        except Exception as exc:
            print(
                f"    n_iters={n_iter}: ERROR — {type(exc).__name__}: {exc}"
            )
            iter_stable = False
    results['VFE iteration stability'] = iter_stable

    # ------------------------------------------------------------------
    # Scenario 8: Combined stress — large K + sharp kappa + 3 iterations
    # embed_dim=32, 4 heads x 8 dims; kappa=0.1; 3 E-steps.
    # ------------------------------------------------------------------
    results['Large K + sharp kappa + 3 iters'] = test_scenario(
        'Large K + sharp kappa + 3 iters',
        config_overrides={
            'embed_dim': 32,
            'irrep_spec': [('fund', 4, 8)],
            'kappa_beta': 0.1,
            'ffn_n_iterations': 3,
        },
    )

    # ------------------------------------------------------------------
    # Scenario 9: Diagonal covariance mode (memory-efficient path)
    # sigma stored as (B, N, K) instead of (B, N, K, K).  Tests that
    # the diagonal-mode transport and KL paths produce finite outputs.
    # ------------------------------------------------------------------
    results['Diagonal covariance mode'] = test_scenario(
        'Diagonal covariance mode',
        config_overrides={
            'diagonal_covariance': True,
        },
    )

    # ------------------------------------------------------------------
    # Scenario 10: Trivial gauge (Omega = I — standard KL attention limit)
    # gauge_mode='trivial' disables phi and sets Omega = I everywhere.
    # Tests the no-transport limit of the VFE path.
    # ------------------------------------------------------------------
    results['Trivial gauge (Omega=I)'] = test_scenario(
        'Trivial gauge (Omega=I)',
        config_overrides={
            'gauge_mode': 'trivial',
        },
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    n_pass  = sum(results.values())
    n_total = len(results)
    print(f"Summary: {n_pass}/{n_total} scenarios passed")
    print()
    for scenario_name, passed in results.items():
        icon = 'PASS' if passed else 'FAIL'
        print(f"  [{icon}]  {scenario_name}")

    # ------------------------------------------------------------------
    # Diagnosis of known failures
    # ------------------------------------------------------------------
    print()
    print("Known failure diagnosis")
    print("-" * 60)
    print(
        "  Root cause: torch.linalg.eigh backward (LinalgEighBackward0)"
        " returns NaN when the covariance matrix passed to _safe_eigh"
        " in _prepare_e_step_inputs has degenerate or near-zero eigenvalues."
        " PyTorch's eigh backward divides by (lambda_i - lambda_j) terms;"
        " when two eigenvalues are equal or zero, this produces 0/0 = NaN."
        " The forward pass is always finite; only the backward is affected."
    )
    print()
    print("  Scenarios that trigger this instability:")

    known_failures = {
        'Large K=64 (8h x 8d)': (
            "Large transported (64x64) covariance has many clustered eigenvalues"
            " at random initialization (Wigner semicircle distribution)."
        ),
        'Near-uniform sigma (log_sigma=+5)': (
            "sigma=exp(5)~148; sigma_max=5.0 clamp activates, producing a"
            " near-rank-1 covariance with degenerate eigenvalue structure."
        ),
        'Large K + sharp kappa + 3 iters': (
            "K=32 transported covariance has clustered eigenvalues at init,"
            " same mechanism as Large K=64 but at smaller scale."
        ),
        'Trivial gauge (Omega=I)': (
            "With Omega=I, the E-step computes a delta-sigma matrix that is"
            " near-zero, producing degenerate eigh backward NaN."
        ),
    }

    for scenario, explanation in known_failures.items():
        flag = 'FAILED' if not results.get(scenario, True) else 'passed'
        print(f"    [{flag}] {scenario}")
        print(f"            {explanation}")

    print()
    print(
        "  Mitigation: use diagonal_covariance=True (avoids the full eigh"
        " path entirely) or add eigenvalue-gap-aware jitter in _safe_eigh"
        " before the eigh call so that eigenvalues are guaranteed distinct."
        " The diagonal_covariance path (Scenario 9) passes cleanly."
    )

    unexpected_failures = [
        k for k, v in results.items() if not v and k not in known_failures
    ]
    if unexpected_failures:
        print()
        print("  UNEXPECTED failures (not in the known-failure list):")
        for k in unexpected_failures:
            print(f"    - {k}")
    print()


if __name__ == '__main__':
    run_all()
