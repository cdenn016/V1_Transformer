# -*- coding: utf-8 -*-
"""
Companion equivalence GATE for gauge_mode='constant' (the BR-1 hazard path).
===========================================================================

Verifier A flagged that the primary learned-gauge gate
(``equivalence_harness.py``) passes TRIVIALLY for the attention-removal refactor:
under ``gauge_mode='learned'`` the attention sublayer is never called and
``constant_omega`` is None, so deleting the attention module cannot change that
gate's loss or grads.  The real BR-1 hazard lives under ``gauge_mode='constant'``,
where ``constant_omega`` is the learned transport and is genuinely trained through
the FFN E-step even with ``skip_attention=True``.

This gate captures a baseline under ``gauge_mode='constant'`` on the PRE-refactor
code (where ``constant_omega`` is registered on ``...attention.constant_omega.*``)
and, after the refactor (where it must re-home to ``...ffn.constant_omega.*``),
asserts:

  * the CE loss is unchanged given the same pinned weights (key-remapped),
  * ``constant_omega`` is still ALIVE (nonzero grad) — i.e. it did NOT vanish
    from the graph / freeze at identity,
  * its grad norm matches the baseline to tolerance,
  * the optimizer routes the re-homed ``ffn.constant_omega.*`` to the intended
    learning-rate group (BR-1a: must NOT silently move from M_attention_lr).

Run::

    python equivalence_gate_constant.py            # capture (run on PRE-refactor code)
    python equivalence_gate_constant.py --gate      # reload + remap + compare (POST-refactor)
"""

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARTIFACT_DIR = HERE.parent
JSON_PATH = ARTIFACT_DIR / "baseline_skip_attention_constant.json"
WEIGHTS_PATH = ARTIFACT_DIR / "baseline_skip_attention_constant.weights.pt"

SEED = 0
DEVICE = torch.device("cpu")
ATOL = 1e-6
RTOL = 1e-5

# The intended LR-group + lr the re-homed constant_omega must land in.  Under the
# pre-refactor code attention.constant_omega.* routes to the 'attention' group at
# config.M_attention_lr.  The refactor adds an explicit 'constant_omega' routing
# rule so ffn.constant_omega.* lands in the SAME effective lr (behavior-neutral).
EXPECT_LR_KEY = "M_attention_lr"


def _remap_constant_omega_key(name: str) -> str:
    """attention.constant_omega.* -> ffn.constant_omega.* (the BR-1 re-home)."""
    return name.replace(".attention.constant_omega.", ".ffn.constant_omega.")


def tiny_constant_config():
    return {
        "vocab_size": 128, "embed_dim": 20, "max_seq_len": 16, "n_layers": 1,
        "ffn_n_iterations": 1, "gauge_dim": 10, "irrep_spec": [("fund", 2, 10)],
        "hidden_dim": 64,
        "gauge_group": "GLK",
        "gauge_mode": "constant",            # <-- the BR-1 path
        "gauge_param": "phi",
        "skip_attention": True,
        "em_mode": "ift_phi",
        "diagonal_covariance": True,
        "isotropic_covariance": False,
        "exact_diagonal_transport": False,
        "evolve_sigma": True,
        # evolve_phi is forced False by BlockConfig.__post_init__ for constant gauge;
        # leave the request here, the model resolves it.
        "evolve_phi": True,
        "evolve_phi_e_step": True,
        "learnable_head_kappa": False,
        "use_residual": False, "use_layernorm": True, "norm_type": "layernorm",
        "use_rope": True, "rope_base": 100, "rope_full_gauge": "off",
        "non_flat_transport": False, "closed_form_e_step": False, "n_picard_steps": 0,
        "alpha_divergence": 0.3, "E_alpha": 1, "E_lambda_belief": 10, "E_lambda_softmax": 0,
        "E_learnable_alpha": True, "E_learnable_lr": True,
        "phi_trace_clamp": 0.75, "phi_project_slk": False, "e_step_sigma_floor": 0.01,
        "sigma_max": 12.0, "phi_natural_gradient": "killing", "enforce_orthogonal": False,
        "active_inference": False, "use_prior_bank": False,
        "use_output_projection": False, "use_equivariant_head_mixer": False,
        "kappa_beta": 1, "tie_embeddings": False, "ffn_mode": "VFE_dynamic",
        "pos_encoding_mode": "none", "dropout": 0.0,
    }


def build_model(cfg):
    from transformer.core.model import GaugeTransformerLM
    torch.manual_seed(SEED)
    model = GaugeTransformerLM(cfg).to(DEVICE)
    model.train()
    return model


def deterministic_inputs(cfg, batch_size=2):
    B, N, V = batch_size, cfg["max_seq_len"], cfg["vocab_size"]
    base = torch.arange(B * N, dtype=torch.long).reshape(B, N)
    return ((base * 7 + 3) % V).to(DEVICE), ((base * 13 + 5) % V).to(DEVICE)


def forward_backward(model, cfg):
    token_ids, targets = deterministic_inputs(cfg)
    model.zero_grad(set_to_none=True)
    logits = model(token_ids)
    V = logits.shape[-1]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
    loss.backward()
    grad = {}
    for name, p in model.named_parameters():
        grad[name] = None if p.grad is None else float(p.grad.detach().norm().item())
    return float(loss.item()), grad


def _find_constant_omega(grad):
    return {k: v for k, v in grad.items() if "constant_omega" in k}


def capture():
    cfg = tiny_constant_config()
    model = build_model(cfg)
    co = [n for n, _ in model.named_parameters() if "constant_omega" in n]
    assert co, ("No constant_omega parameter found under gauge_mode='constant'. "
                "Expected ...attention.constant_omega.* on the pre-refactor code.")
    loss, grad = forward_backward(model, cfg)
    co_grad = _find_constant_omega(grad)
    assert all(v is not None and v > 0 for v in co_grad.values()), \
        f"constant_omega is not trained on the pre-refactor code: {co_grad}"
    torch.save(model.state_dict(), WEIGHTS_PATH)
    out = {
        "loss": loss,
        "grad_norms": grad,
        "constant_omega_keys": co,
        "param_count": len(grad),
        "config_used": {"seed": SEED, "device": "cpu", "tiny_config": cfg,
                        "weights_file": WEIGHTS_PATH.name, "atol": ATOL, "rtol": RTOL},
    }
    with open(JSON_PATH, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"[capture-constant] loss={loss:.8f} params={len(grad)} "
          f"constant_omega={co} grads={co_grad}")
    return out


def _check_lr_routing(model):
    """Confirm the re-homed ffn.constant_omega.* routes to the intended LR group."""
    try:
        from transformer.training.optimizer import create_param_groups
        from transformer.training.config import TrainingConfig
    except Exception as exc:  # pragma: no cover
        print(f"[gate-constant] (skip LR check: {exc})")
        return True
    cfg = TrainingConfig()
    groups = create_param_groups(model, cfg, verbose=False)
    expected_lr = getattr(cfg, EXPECT_LR_KEY)
    id_to_group = {}
    for g in groups:
        for p in g["params"]:
            id_to_group[id(p)] = (g.get("name"), g["lr"])
    ok = True
    for name, p in model.named_parameters():
        if "constant_omega" in name:
            grp = id_to_group.get(id(p))
            if grp is None:
                ok = False
                print(f"[gate-constant] LR ROUTING: {name} not in any param group")
            elif abs(grp[1] - expected_lr) > 1e-12:
                ok = False
                print(f"[gate-constant] LR ROUTING DRIFT: {name} -> group {grp[0]!r} "
                      f"@ lr={grp[1]} (expected {EXPECT_LR_KEY}={expected_lr})")
            else:
                print(f"[gate-constant] LR routing OK: {name} -> {grp[0]!r} @ lr={grp[1]}")
    return ok


def run_gate():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"No constant baseline at {JSON_PATH}; run capture on pre-refactor code first.")
    with open(JSON_PATH) as f:
        base = json.load(f)

    cfg = tiny_constant_config()
    model = build_model(cfg)

    # Load pinned weights with the BR-1 key remap (attention.* -> ffn.*).
    pinned = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    remapped = {_remap_constant_omega_key(k): v for k, v in pinned.items()}
    incompat = model.load_state_dict(remapped, strict=False)
    # The only acceptable missing/unexpected keys are attention-module keys that
    # no longer exist; constant_omega MUST have loaded under its new ffn.* name.
    model_keys = set(dict(model.named_parameters()).keys()) | set(model.state_dict().keys())
    co_loaded = [k for k in remapped if "constant_omega" in k and k in model_keys]
    if not co_loaded:
        print(f"[gate-constant] FAIL: remapped constant_omega did not match any model key. "
              f"missing={incompat.missing_keys} unexpected={incompat.unexpected_keys}")
        return False

    loss, grad = forward_backward(model, cfg)
    ok = True

    if abs(loss - base["loss"]) > ATOL + RTOL * abs(base["loss"]):
        ok = False
        print(f"[gate-constant] LOSS MISMATCH: got {loss:.8f}, baseline {base['loss']:.8f}")

    # Compare grad norms, remapping the baseline constant_omega keys to ffn.*.
    base_g = base["grad_norms"]
    co_alive = False
    for bname, bval in base_g.items():
        cname = _remap_constant_omega_key(bname)
        gval = grad.get(cname, "MISSING")
        if "constant_omega" in cname:
            if gval in (None, "MISSING") or gval == 0.0:
                ok = False
                print(f"[gate-constant] CONSTANT_OMEGA DEAD after refactor: {cname}={gval} "
                      f"(baseline {bval}) -> Omega frozen at identity / vanished from optimizer")
            else:
                co_alive = True
                if bval is not None and abs(gval - bval) > ATOL + RTOL * abs(bval):
                    ok = False
                    print(f"[gate-constant] CONSTANT_OMEGA GRAD MISMATCH {cname}: "
                          f"got {gval:.8e}, baseline {bval:.8e}")
                else:
                    print(f"[gate-constant] constant_omega alive + matched: {cname} grad={gval:.8e}")
            continue
        if gval == "MISSING":
            if bval is not None:
                ok = False
                print(f"[gate-constant] ALIVE PARAM MISSING: {bname} (baseline {bval:.8e})")
        elif bval is not None and gval is not None:
            if abs(gval - bval) > ATOL + RTOL * abs(bval):
                ok = False
                print(f"[gate-constant] GRAD MISMATCH {bname}: got {gval:.8e}, baseline {bval:.8e}")

    if not co_alive:
        ok = False
        print("[gate-constant] FAIL: no constant_omega grad confirmed alive")

    ok = _check_lr_routing(model) and ok
    print("[gate-constant] PASS" if ok else "[gate-constant] FAIL")
    return ok


if __name__ == "__main__":
    if "--gate" in sys.argv:
        sys.exit(0 if run_gate() else 1)
    capture()
