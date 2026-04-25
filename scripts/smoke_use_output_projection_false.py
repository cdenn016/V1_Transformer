"""
Smoke test: `use_output_projection=False` full-covariance stability fix.

Runs the Exp-74 BASELINE_CONFIG with `use_output_projection=False` for
enough steps to cross the observed failure point (step 552 on the original
run). Reports the `gauge_kl_full_chol_fallback_diag` counter at completion.

Success criteria:
    - Training completes without NaN loss.
    - `gauge_kl_full_chol_fallback_diag` counter stays at 0 (or near 0),
      indicating the spd_eigfloor guard prevented Cholesky failure in
      `transformer/core/gauge_utils.py:fused_block_diagonal_kl_full`.

Click-to-run: edit SMOKE_MAX_STEPS if needed, then run the file.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 stdout so tqdm.write can emit Greek letters (β, φ, κ, σ) that
# appear in training logs. Without this, Windows cp1252 terminals raise
# UnicodeEncodeError mid-training.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import torch

from math_utils import numerical_monitor
from scripts.run_ablation_suite import BASELINE_CONFIG
from transformer.training.experiment_runner import run_single_experiment
from transformer.training.utils import set_all_seeds


SMOKE_MAX_STEPS = 700   # past the observed step-552 failure point
SMOKE_SEED = 6           # matches BASELINE stride_base_seed for reproducibility
SMOKE_OUTPUT = Path(__file__).parent / "smoke_results" / "wo_false_smoke"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device = {device}")

    cfg = copy.deepcopy(BASELINE_CONFIG)
    cfg["use_output_projection"] = False
    cfg["max_steps"] = SMOKE_MAX_STEPS
    cfg["assert_finite_loss"] = False  # Let us observe if NaN occurs instead of raising.
    cfg["log_interval"] = 50

    SMOKE_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Snapshot counter state before run so we can delta across the training.
    numerical_monitor.flush()  # reset

    set_all_seeds(SMOKE_SEED)

    result = run_single_experiment(
        config=cfg,
        ffn_mode=cfg.get("ffn_mode", "VFE_dynamic"),
        device=device,
        checkpoint_dir=SMOKE_OUTPUT,
        enable_publication_metrics=False,
        quiet=True,
        skip_test_eval=True,
        skip_post_training_viz=True,
    )

    counters = numerical_monitor.flush()
    chol_fallback = counters.get("gauge_kl_full_chol_fallback_diag", 0)
    nan_replace = counters.get("fused_kl_nan_replace", 0)

    print("\n" + "=" * 60)
    print("SMOKE TEST RESULT")
    print("=" * 60)
    print(f"  use_output_projection = False")
    print(f"  max_steps             = {SMOKE_MAX_STEPS}")
    print(f"  final val PPL         = {result.get('final_ppl', 'n/a')}")
    print(f"  chol_fallback_count   = {chol_fallback}")
    print(f"  kl_nan_replace_count  = {nan_replace}")
    print("-" * 60)
    if chol_fallback == 0:
        print("  [PASS] No Cholesky fallback in attention-KL path.")
    else:
        print(f"  [WARN] {chol_fallback} fallback events — floor may need tuning.")
    print("=" * 60)


if __name__ == "__main__":
    main()
