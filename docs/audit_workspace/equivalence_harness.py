# -*- coding: utf-8 -*-
"""
Equivalence GATE harness for the attention-removal refactor.
============================================================

Captures a behavior-preserving baseline on the tiny live-patterned
``skip_attention=True`` + ``em_mode='ift_phi'`` GaugeTransformerLM:

  * scalar CE loss
  * full {param_name -> grad_norm} table after a real forward + CE backward

The refactor (memo 07: remove the IrrepMultiHeadAttention sublayer) must
reproduce this baseline to tolerance.

DESIGN NOTES (why this is a trustworthy gate, not an init-order trap):

  * We PIN WEIGHTS.  The canonical capture saves the model state_dict next to
    the JSON.  When ``--gate`` is passed (or ``run_gate()`` is called) the
    harness rebuilds the model, LOADS the pinned weights, then runs
    forward/backward and compares.  This isolates "same math given same
    weights" from "same construction-RNG init order".  Removing the attention
    module shifts the global RNG init order, so a seed-0-init-only gate would
    spuriously FAIL a behavior-preserving refactor.  Loading pinned weights
    removes that confound.

  * Inputs are CONSTRUCTION-RNG-INDEPENDENT.  token_ids / targets are built
    with ``torch.arange(...) % vocab`` (deterministic), NOT drawn from the
    global stream after the model is built.

  * Device is forced to CPU (torch 2.11 CPU-only in this audit env).

  * grad is None is recorded as JSON ``null`` (dead weight), distinct from a
    real zero grad_norm of 0.0.  The gate must distinguish dead from zero-grad.

Run::

    python equivalence_harness.py            # capture baseline + state_dict + JSON
    python equivalence_harness.py --gate      # reload pinned weights, recompute, compare
"""

import json
import sys
from pathlib import Path

import torch

# --- repo root on path ------------------------------------------------------
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]  # docs/audit_workspace/equivalence_harness.py -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARTIFACT_DIR = HERE.parent
JSON_PATH = ARTIFACT_DIR / "baseline_skip_attention.json"
WEIGHTS_PATH = ARTIFACT_DIR / "baseline_skip_attention.weights.pt"

SEED = 0
DEVICE = torch.device("cpu")
ATOL = 1e-6   # absolute tolerance on loss / grad_norm for the gate
RTOL = 1e-5   # relative tolerance


# ---------------------------------------------------------------------------
# Tiny config patterned EXACTLY on the live EM_CONFIG
# (transformer/train_publication.py).  Only sizes are shrunk; every toggle
# that affects the gradient path is held at its live value.
# ---------------------------------------------------------------------------
def tiny_live_config():
    return {
        # --- shrunk sizes ---
        "vocab_size": 128,
        "embed_dim": 20,
        "max_seq_len": 16,
        "n_layers": 1,
        "ffn_n_iterations": 1,
        "gauge_dim": 10,
        "irrep_spec": [("fund", 2, 10)],   # GL(10), 2 heads d=10
        "hidden_dim": 64,
        # --- live gradient-path toggles ---
        "gauge_group": "GLK",
        "gauge_mode": "learned",
        "gauge_param": "phi",
        "skip_attention": True,
        "em_mode": "ift_phi",
        "diagonal_covariance": True,
        "isotropic_covariance": False,
        "exact_diagonal_transport": False,
        "evolve_sigma": True,
        "evolve_phi": True,
        "evolve_phi_e_step": True,
        "learnable_head_kappa": False,
        "use_residual": False,
        "use_layernorm": True,
        "norm_type": "layernorm",
        "use_rope": True,
        "rope_base": 100,
        "rope_full_gauge": "off",
        "non_flat_transport": False,
        "closed_form_e_step": False,
        "n_picard_steps": 0,
        "alpha_divergence": 0.3,
        "E_alpha": 1,
        "E_lambda_belief": 10,
        "E_lambda_softmax": 0,
        "E_learnable_alpha": True,
        "E_learnable_lr": True,
        "phi_trace_clamp": 0.75,
        "phi_project_slk": False,
        "e_step_sigma_floor": 0.01,
        "sigma_max": 12.0,
        "phi_natural_gradient": "killing",
        "enforce_orthogonal": False,
        "active_inference": False,
        "use_prior_bank": False,
        "use_output_projection": False,
        "use_equivariant_head_mixer": False,
        "kappa_beta": 1,
        "tie_embeddings": False,
        "ffn_mode": "VFE_dynamic",
        "pos_encoding_mode": "none",
        "dropout": 0.0,
    }


def build_model(cfg):
    """Deterministic construction under the fixed seed, forced to CPU."""
    from transformer.core.model import GaugeTransformerLM

    torch.manual_seed(SEED)
    model = GaugeTransformerLM(cfg)
    model = model.to(DEVICE)
    model.train()
    return model


def deterministic_inputs(cfg, batch_size=2):
    """Inputs that do NOT consume global construction RNG."""
    B = batch_size
    N = cfg["max_seq_len"]
    V = cfg["vocab_size"]
    # deterministic, reproducible, independent of init order
    base = torch.arange(B * N, dtype=torch.long).reshape(B, N)
    token_ids = (base * 7 + 3) % V
    targets = (base * 13 + 5) % V
    return token_ids.to(DEVICE), targets.to(DEVICE)


def forward_backward(model, cfg, batch_size=2):
    """Real forward + CE backward.  Returns (loss_float, grad_norms dict)."""
    token_ids, targets = deterministic_inputs(cfg, batch_size)

    model.zero_grad(set_to_none=True)
    logits = model(token_ids)                       # (B, N, V)
    V = logits.shape[-1]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1)
    )
    loss.backward()

    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            grad_norms[name] = None          # dead weight (JSON null)
        else:
            grad_norms[name] = float(p.grad.detach().norm().item())
    return float(loss.item()), grad_norms


def resolved_flags(model):
    """Dump the RESOLVED path actually wired (NOT the config dict).

    CLAUDE.md: a config dict can silently fall back to dataclass defaults, so
    the baseline must assert the live path was actually built.
    """
    block = model.transformer.blocks[0]
    ffn = block.ffn
    return {
        "block.skip_attention": bool(block.skip_attention),
        "ffn.em_phi_mode": getattr(ffn, "em_phi_mode", None),
        "ffn.amortized_inference": bool(getattr(ffn, "amortized_inference", None)),
        "ffn.amortize_sigma": bool(getattr(ffn, "amortize_sigma", None)),
        "ffn.exact_phi_grad": bool(getattr(ffn, "exact_phi_grad", None)),
        "ffn.gauge_mode": getattr(ffn, "gauge_mode", None),
        "ffn.gauge_param": getattr(ffn, "gauge_param", None),
        "ffn.diagonal_covariance": bool(getattr(ffn, "diagonal_covariance", None)),
        "ffn.n_iterations": getattr(ffn, "n_iterations", None),
        "model.use_prior_bank": bool(getattr(model, "use_prior_bank", False)),
    }


def assert_live_path(flags):
    """Hard-assert the resolved path matches the live skip_attention+ift_phi path."""
    expect = {
        "block.skip_attention": True,
        "ffn.em_phi_mode": "amortized",
        "ffn.amortized_inference": True,
        "ffn.amortize_sigma": True,
        "ffn.gauge_mode": "learned",
        "ffn.gauge_param": "phi",
        "ffn.diagonal_covariance": True,
        "model.use_prior_bank": False,
    }
    for k, v in expect.items():
        assert flags.get(k) == v, f"RESOLVED PATH MISMATCH: {k}={flags.get(k)!r}, expected {v!r}"


def capture():
    cfg = tiny_live_config()
    model = build_model(cfg)
    flags = resolved_flags(model)
    assert_live_path(flags)

    loss, grad_norms = forward_backward(model, cfg)

    # Pin weights so the gate compares same-math-given-same-weights.
    torch.save(model.state_dict(), WEIGHTS_PATH)

    out = {
        "loss": loss,
        "grad_norms": grad_norms,
        "param_count": len(grad_norms),
        "config_used": {
            "seed": SEED,
            "device": "cpu",
            "batch_size": 2,
            "tiny_config": cfg,
            "resolved_flags": flags,
            "weights_file": WEIGHTS_PATH.name,
            "atol": ATOL,
            "rtol": RTOL,
        },
    }
    with open(JSON_PATH, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"[capture] loss={loss:.8f}  params={len(grad_norms)}")
    print(f"[capture] wrote {JSON_PATH.name} and {WEIGHTS_PATH.name}")
    return out


def run_gate():
    """Reload pinned weights, recompute, and compare to the baseline JSON.

    This is what the refactor re-runs.  Build the (possibly refactored) model,
    load the pinned weights with strict=False (post-refactor key migrations are
    allowed; mismatched keys are reported), recompute loss + grad norms, and
    assert they match the captured baseline to tolerance.
    """
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"No baseline at {JSON_PATH}; run capture first.")
    with open(JSON_PATH) as f:
        base = json.load(f)

    cfg = tiny_live_config()
    model = build_model(cfg)
    incompat = model.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location=DEVICE), strict=False
    )
    if incompat.missing_keys or incompat.unexpected_keys:
        print(f"[gate] WARNING load_state_dict missing={incompat.missing_keys} "
              f"unexpected={incompat.unexpected_keys}")

    loss, grad_norms = forward_backward(model, cfg)

    ok = True
    if abs(loss - base["loss"]) > ATOL + RTOL * abs(base["loss"]):
        ok = False
        print(f"[gate] LOSS MISMATCH: got {loss:.8f}, baseline {base['loss']:.8f}")

    # Behavior-preserving invariant for the attention-removal refactor:
    #   * loss identical
    #   * every ALIVE baseline param still exists with the same grad norm
    #   * a DEAD baseline param (grad is None) MAY be removed — that is an
    #     intended dead-weight cleanup (e.g. norm2 under skip_attention) and
    #     cannot change the math, since a null-grad param affects neither the
    #     forward nor any gradient. It must NOT come back alive if it survives.
    #   * NO NEW ALIVE param may appear (that would be a behavior change).
    base_g = base["grad_norms"]
    removed_dead = []
    for name, bval in base_g.items():
        gval = grad_norms.get(name, "MISSING")
        if gval == "MISSING":
            if bval is None:
                removed_dead.append(name)          # OK: intended dead-weight removal
            else:
                ok = False
                print(f"[gate] ALIVE PARAM MISSING after refactor: {name} (baseline {bval:.8e})")
            continue
        if (bval is None) != (gval is None):
            ok = False
            print(f"[gate] DEAD/ALIVE MISMATCH {name}: baseline={bval}, got={gval}")
        elif bval is not None:
            if abs(gval - bval) > ATOL + RTOL * abs(bval):
                ok = False
                print(f"[gate] GRAD MISMATCH {name}: got {gval:.8e}, baseline {bval:.8e}")

    new_alive = [n for n, v in grad_norms.items() if n not in base_g and v is not None]
    if new_alive:
        ok = False
        print(f"[gate] NEW ALIVE PARAM(S) after refactor (behavior change): {new_alive}")
    if removed_dead:
        print(f"[gate] (info) intended dead-weight params removed: {removed_dead}")

    print("[gate] PASS" if ok else "[gate] FAIL")
    return ok


if __name__ == "__main__":
    if "--gate" in sys.argv:
        sys.exit(0 if run_gate() else 1)
    capture()
