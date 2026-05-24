"""
VFETrainer: training loop with full metrics, diagnostics, and publication output.

    E-step: model.forward(token_ids) — infer q* from context only (no target leak)
    M-step: loss.backward() — gradients flow through E-step via semi-gradient backprop

Reuses PublicationMetricsTracker (CSV) and TrainingTracker/PublicationFigures
from the legacy training infrastructure for full diagnostic parity.
"""

import dataclasses
import json
import math
import subprocess
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel

if TYPE_CHECKING:
    import numpy as _np_typing
    from transformer.vfe.e_step import VFEEStep
    from transformer.training.metrics_tracking import PublicationMetricsTracker
    from transformer.analysis.publication_metrics import TrainingTracker

logger = logging.getLogger(__name__)

try:
    # Use plain `tqdm` (not `tqdm.auto`). tqdm.auto dispatches to
    # `tqdm.notebook` when an IPython kernel is detected (VS Code
    # Jupyter, Interactive Python, PyCharm cells), which renders an
    # HTML widget that some Run-button consumers swallow entirely.
    # The plain entry point always uses the stderr console renderer,
    # matching `experiment_runner.py:1253` exactly.
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:  # tqdm optional — falls back to a plain range + print
    _tqdm = None
    _HAS_TQDM = False


def _format_duration(secs: float) -> str:
    """Compact human-readable duration: '42s', '5m23s', '1h05m', '2d03h'."""
    secs = max(0.0, float(secs))
    if secs < 60.0:
        return f"{secs:.0f}s"
    if secs < 3600.0:
        m = int(secs // 60)
        s = int(secs - 60 * m)
        return f"{m}m{s:02d}s"
    if secs < 86400.0:
        h = int(secs // 3600)
        m = int((secs - 3600 * h) // 60)
        return f"{h}h{m:02d}m"
    d = int(secs // 86400)
    h = int((secs - 86400 * d) // 3600)
    return f"{d}d{h:02d}h"


def _gauge_group_label(cfg: VFEConfig) -> str:
    r"""Folder/log tag for the gauge group.

    For ``gauge_group='GLK'`` the reduced gauge group is the block-diagonal
    product :math:`GL(d_1) \oplus \cdots \oplus GL(d_B)` over the effective
    block partition, so the tag is ``GL(d_h)`` — the per-head block dimension —
    NOT ``GL(embed_dim)``. Example: 2 heads of dimension 10 (``embed_dim=20``)
    is ``GL(10)``, not ``GL(20)``. Mixed block dimensions are joined, e.g.
    ``GL(1,3)``. ``effective_block_dims`` is used so cross-coupled super-blocks
    report their merged ``GL(d_super)`` dimension.
    """
    gg = cfg.gauge_group
    if gg == 'GLK':
        dims = sorted(set(cfg.effective_block_dims))
        inner = ",".join(str(d) for d in dims)
        return f"GL({inner})"
    if gg == 'SO3':
        return "SO(3)"
    if gg == 'SON':
        return f"SO({cfg.embed_dim})"
    return str(gg)


class VFETrainer:
    # Class-level annotations for attributes set in __init__ to None
    # (Optional types so downstream methods type-check with mypy).
    _metrics_tracker: "Optional[PublicationMetricsTracker]"
    _pub_tracker: "Optional[TrainingTracker]"
    _last_val_input_ids: Optional[torch.Tensor]

    r"""Training loop for VFEModel with full metrics and diagnostics.

    Implements the clean E-step/M-step split with comprehensive logging:
        E-step: forward pass infers beliefs from context only
        M-step: backprop through E-step to update priors/embeddings

    Args:
        model: VFEModel instance.
        cfg: VFEConfig with training hyperparameters.
        train_loader: Training data loader yielding (input_ids, target_ids).
        val_loader: Optional validation data loader.
        device: Device to train on.
        output_dir: Directory for metrics CSV, checkpoints, and figures.
    """

    def __init__(
        self,
        model: VFEModel,
        cfg: VFEConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        device: str = 'cpu',
        output_dir: Optional[str] = None,
        generate_figures: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.generate_figures = generate_figures

        # Optimizer with per-type learning rates
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.global_step = 0

        # Output directory for metrics, checkpoints, figures
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Per-run snapshot of config, environment, and git SHA. Without
            # this, post-hoc "which flag did this run use" analysis is
            # impossible — see post_edit_2026-05-19.md Wave 6.
            self._dump_system_info()

        # Metrics infrastructure (lazy init in train())
        self._metrics_tracker = None
        self._pub_tracker = None

        # Attention-plot scratch space — captured in evaluate(), consumed
        # in _plot_attention_patterns(). None means no eval has run yet.
        self._last_val_input_ids = None

        # Per-step diagnostic gates. Default True (preserve current
        # semantics for callers that don't manage these). The training
        # loop flips them to False on non-log / non-eval steps so the
        # per-forward CPU syncs and post-iteration `compute_gauge_transport`
        # rebuild are skipped on the ~99% of steps that don't consume them.
        self._aggregate_diagnostics_this_step: bool = True

        # Initialise the per-block attention-state capture flag from the
        # current default (True). The flag actually lives on each block's
        # VFEEStep; the trainer toggles it on/off via helper.
        self._set_capture_attention_state(True)

    def _dump_system_info(self) -> None:
        """Write ``output_dir/system_info.json`` with config + environment.

        Snapshot taken once at trainer init. Captures:

        - ``config``: ``dataclasses.asdict(cfg)`` with non-JSON-serializable
          fields (tensors, modules) stringified via ``default=str``.
        - ``git_sha``: ``git rev-parse HEAD`` (None on failure).
        - ``git_branch``: current branch (None on failure).
        - ``torch_version``, ``cuda_version``, ``cudnn_version``.
        - ``gpu``: GPU name and capability if CUDA available.
        - ``python_version``, ``platform``.
        - ``initial_seed``: ``torch.initial_seed()`` (the seed the caller
          already set via ``torch.manual_seed``).
        - ``start_time``: ISO-8601 UTC timestamp.

        Errors are logged but never raised — system-info is diagnostic
        only and must not block training.
        """
        if self.output_dir is None:
            return
        try:
            cfg_dict = dataclasses.asdict(self.cfg)
        except TypeError:
            cfg_dict = {k: str(v) for k, v in vars(self.cfg).items()}

        def _git(args: List[str]) -> Optional[str]:
            try:
                out = subprocess.run(
                    ['git'] + args,
                    capture_output=True, text=True, timeout=5,
                    check=False,
                )
                if out.returncode == 0:
                    return out.stdout.strip()
            except (FileNotFoundError, subprocess.SubprocessError, OSError):
                pass
            return None

        gpu_info: Dict[str, Any] = {}
        if torch.cuda.is_available():
            try:
                idx = torch.cuda.current_device()
                gpu_info = {
                    'name': torch.cuda.get_device_name(idx),
                    'capability': list(torch.cuda.get_device_capability(idx)),
                    'count': torch.cuda.device_count(),
                }
            except Exception as exc:
                gpu_info = {'error': str(exc)}

        info: Dict[str, Any] = {
            'start_time_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'git_sha': _git(['rev-parse', 'HEAD']),
            'git_branch': _git(['rev-parse', '--abbrev-ref', 'HEAD']),
            'git_dirty': _git(['status', '--porcelain']) not in (None, ''),
            'torch_version': torch.__version__,
            'cuda_version': getattr(torch.version, 'cuda', None),
            'cudnn_version': (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            ),
            'gpu': gpu_info or None,
            'python_version': sys.version.split()[0],
            'platform': sys.platform,
            'device': self.device,
            'initial_seed': torch.initial_seed(),
            'output_dir': str(self.output_dir),
            'config': cfg_dict,
        }

        try:
            out_path = self.output_dir / 'system_info.json'
            with out_path.open('w', encoding='utf-8') as fh:
                json.dump(info, fh, indent=2, default=str, sort_keys=False)
        except OSError as exc:
            logger.warning("Failed to write system_info.json: %s", exc)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW optimizer with per-group LRs (cfg.m_*_lr)."""
        cfg = self.cfg
        model = self.model

        # Group parameters by type. Match order is significant: a param name
        # containing 'phi_embed' must hit the m_phi bucket before the
        # 'e_step' bucket (no overlap in practice today, but the order
        # enforces it).
        m_mu_params = []
        m_sigma_params = []
        m_phi_params = []
        m_hyper_params = []
        m_other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # base_mu (gauge-fixed mode) and mu_embed (direct mode) both go in
            # the m_mu group — they are alternate parameterizations of the
            # same conceptual quantity (prior mean).
            if 'base_mu' in name or 'mu_embed' in name:
                m_mu_params.append(param)
            elif 'base_log_sigma' in name or 'sigma_log_embed' in name:
                m_sigma_params.append(param)
            elif 'phi_embed' in name or 'pos_phi' in name:
                m_phi_params.append(param)
            elif 'decode_log_scale' in name or 'e_step' in name:
                # decode_log_scale is the learnable decode softmax temperature
                # (τ_decode), parametrically analogous to the attention κ which
                # also lives in this hyper group. A prior implementation placed
                # it in m_sigma_lr, conflating decode-temperature tuning with
                # belief-σ tuning.
                # NOTE: _phi_preconditioner is a register_buffer (not a
                # Parameter), so named_parameters() never yields it; it stays
                # frozen at init by design. The pattern is intentionally
                # omitted here.
                m_hyper_params.append(param)
            else:
                m_other_params.append(param)

        param_groups = [
            {'params': m_mu_params,    'lr': cfg.m_mu_lr,    'name': 'm_mu'},
            {'params': m_sigma_params, 'lr': cfg.m_sigma_lr, 'name': 'm_sigma'},
            {'params': m_phi_params,   'lr': cfg.m_phi_lr,   'name': 'm_phi'},
            {'params': m_hyper_params, 'lr': cfg.m_hyper_lr, 'name': 'm_hyper'},
            {'params': m_other_params, 'lr': cfg.m_other_lr, 'name': 'm_other'},
        ]
        # Filter empty groups (e.g. m_hyper is empty when raw_c0/raw_b0 are
        # not registered because E_learnable_alpha=False and learnable_kappa=False).
        param_groups = [g for g in param_groups if g['params']]

        return torch.optim.AdamW(
            param_groups,
            lr=cfg.m_other_lr,    # AdamW requires a default; every group sets its own.
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Build cosine schedule with linear warmup."""
        cfg = self.cfg

        def lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return step / max(cfg.warmup_steps, 1)
            progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _compute_gradient_norms(self) -> Dict[str, float]:
        """Compute gradient norms per parameter group (after backward, before step).

        Groups parameter gradients by name, then computes one ``norm()`` per
        group on the concatenated flat vector — previously ran one
        ``.norm().item()`` per parameter (one CUDA sync each), now one
        ``.tolist()`` call at the end (one sync total).
        """
        groups: Dict[str, List[torch.Tensor]] = {
            'mu': [], 'sigma': [], 'phi': [], 'ffn': [], 'other': [],
        }
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad.data.flatten()
            if 'base_mu' in name or 'mu_embed' in name:
                groups['mu'].append(g)
            elif (
                'base_log_sigma' in name
                or 'sigma_log_embed' in name
                or 'decode_log_scale' in name
            ):
                groups['sigma'].append(g)
            elif 'phi_embed' in name or 'pos_phi' in name:
                groups['phi'].append(g)
            elif 'e_step' in name:
                groups['ffn'].append(g)
            else:
                groups['other'].append(g)

        group_norms: List[torch.Tensor] = []
        keys: List[str] = []
        for k, tensors in groups.items():
            keys.append(k)
            if tensors:
                group_norms.append(torch.cat(tensors).norm())
            else:
                group_norms.append(torch.zeros((), device=next(self.model.parameters()).device))
        # Total is the L2 of the per-group norms.
        all_norms = torch.stack(group_norms + [torch.stack(group_norms).norm()])
        values = all_norms.tolist()  # single CUDA sync
        out = {k: values[i] for i, k in enumerate(keys)}
        out['total'] = values[-1]
        return out

    @torch.no_grad()
    def _attention_summary(self) -> Dict[str, float]:
        r"""Summary statistics over the most recent attention/KL matrices.

        Reads ``_last_attention`` (``beta``) and ``_last_kl_matrix`` directly
        from every block's E-step. Both tensors are written unconditionally
        on the final E-step iteration (``vfe/e_step.py:531-533``) — independent
        of ``track_layer_diagnostics`` — so this works on every forward
        without enabling the expensive ``.item()`` diagnostics dict.

        Returns:
            Dict with keys ``beta_kl`` (mean of beta * KL, the per-pair
            alignment contribution before the lambda_align scaling),
            ``beta_mean``, ``attn_entropy`` (-sum beta log beta), and
            ``attn_concentration`` (mean max-beta-per-row). Empty if no
            attention has been computed yet.
        """
        blocks = list(self.model.stack.blocks)
        if not blocks:
            return {}
        # Stack the four per-block statistics into one tensor and pull all
        # numbers back to CPU in a single .tolist() call — previously fired
        # four .item() syncs per block (so 16 for n_layers=4).
        per_block: List[torch.Tensor] = []
        for b in blocks:
            beta = getattr(b.e_step, '_last_attention', None)
            if beta is None:
                continue
            kl = getattr(b.e_step, '_last_kl_matrix', None)
            # Match e_step.py's _BETA_LOG_FLOOR = 1e-30 — same op (β·log β
            # entropy), same float64-underflow boundary.
            beta_safe = beta.clamp(min=1e-30)
            # `beta_kl` (mean β·KL) needs the per-pair KL matrix. Some E-step
            # paths do not surface one — notably the rope_full_gauge per-head
            # branch (vfe/e_step.py:898-966) returns only β and sets
            # `_last_kl_matrix=None`. Emit NaN for that single field but still
            # report the β-only statistics so the summary is non-empty;
            # otherwise the trainer prints a bare `β: nan` that reads like a
            # numerical blow-up rather than an unavailable diagnostic.
            beta_kl = (
                (beta * kl).mean() if kl is not None
                else torch.full((), float('nan'), device=beta.device, dtype=beta.dtype)
            )
            per_block.append(torch.stack([
                beta_kl,
                beta.mean(),
                -(beta_safe * beta_safe.log()).sum(-1).mean(),
                beta.max(dim=-1)[0].mean(),
            ]))
        if not per_block:
            return {}
        means = torch.stack(per_block).mean(dim=0).tolist()  # single sync
        return {
            'beta_kl':            means[0],
            'beta_mean':          means[1],
            'attn_entropy':       means[2],
            'attn_concentration': means[3],
        }

    @torch.no_grad()
    def _generate_sample(
        self,
        prompt: str = "The",
        max_new_tokens: int = 30,
        temperature: float = 0.9,
        top_k: int = 30,
        display_chars: int = 100,
    ) -> str:
        """Generate a short text sample to eyeball that the model is learning.

        Uses ``dataset.encode`` / ``dataset.decode`` from the training dataset
        and ``VFEModel.generate`` (top-k temperature sampling). Returns the
        decoded text truncated to ``display_chars`` characters. Returns the
        empty string on any failure so a broken sampler never aborts the
        validation block.
        """
        ds = self.train_loader.dataset
        if not (hasattr(ds, 'encode') and hasattr(ds, 'decode')):
            return ""
        try:
            prompt_ids = ds.encode(prompt)
            if not prompt_ids:
                return ""
            prompt_tensor = torch.tensor(
                [prompt_ids], device=self.device, dtype=torch.long,
            )
            was_training = self.model.training
            self.model.eval()
            generated = self.model.generate(
                prompt_ids=prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            if was_training:
                self.model.train()
            text = ds.decode(generated[0])
            text = " ".join(text.split())   # collapse whitespace for one-line display
            return text[:display_chars]
        except (RuntimeError, AttributeError, IndexError, KeyError, ValueError) as e:
            # Surface the failure mode but don't abort training on a sampling glitch.
            logger.warning("sample generation failed: %s: %s", type(e).__name__, e)
            return f"<sample failed: {type(e).__name__}: {e}>"

    def _set_capture_attention_state(self, value: bool) -> None:
        """Flip every block's VFEEStep._capture_attention_state.

        The post-iteration attention-state rebuild (`compute_gauge_transport`
        on the converged φ inside `e_step.py`) is only consumed by
        `_plot_attention_patterns`. Setting False on non-eval steps skips
        that rebuild — a matrix-exp per token per block per forward.
        """
        for block in self.model.stack.blocks:
            block.e_step._capture_attention_state = value

    def _collect_e_step_diagnostics(self) -> Dict[str, float]:
        """Collect E-step diagnostics aggregated across all layers.

        Numeric scalar metrics are averaged across layers; the final layer's
        values are also exposed under the suffix ``_final`` for comparison.
        """
        blocks = list(self.model.stack.blocks)
        per_layer = [
            getattr(b.e_step, '_last_diagnostics', {}) or {} for b in blocks
        ]
        diag: Dict[str, float] = {}
        if not per_layer:
            return diag
        # Mean over layers for every numeric key present in any layer.
        all_keys = set().union(*per_layer)
        for k in all_keys:
            vals = [
                float(ld[k]) for ld in per_layer
                if k in ld and isinstance(ld[k], (int, float))
            ]
            if vals:
                diag[k] = sum(vals) / len(vals)
        # Also surface the final layer's values under `_final` suffix.
        for k, v in per_layer[-1].items():
            if isinstance(v, (int, float)):
                diag[f'{k}_final'] = float(v)
        return diag

    def _collect_bayesian_alpha_stats(self) -> Dict[str, float]:
        """Collect Bayesian alpha diagnostics aggregated across all layers.

        Per-layer scalars are stacked into a single tensor per layer and
        pulled to host with one ``.tolist()`` per layer, then averaged
        across layers. Previously a stale early-return inside the loop
        body reported only ``blocks[0]`` when ``n_layers>1``.
        """
        if not self.cfg.E_learnable_alpha:
            return {}
        per_layer: List[List[float]] = []
        with torch.no_grad():
            for block in self.model.stack.blocks:
                es = block.e_step
                c0 = F.softplus(es.raw_c0)
                b0 = F.softplus(es.raw_b0)
                alpha_at_zero = c0 / b0  # alpha when KL=0
                packed = torch.stack([
                    c0.mean(),
                    b0.mean(),
                    c0.std(unbiased=False) if c0.numel() > 1 else torch.zeros((), device=c0.device, dtype=c0.dtype),
                    b0.std(unbiased=False) if b0.numel() > 1 else torch.zeros((), device=b0.device, dtype=b0.dtype),
                    alpha_at_zero.mean(),
                    alpha_at_zero.std(unbiased=False) if alpha_at_zero.numel() > 1 else torch.zeros((), device=alpha_at_zero.device, dtype=alpha_at_zero.dtype),
                    alpha_at_zero.min(),
                    alpha_at_zero.max(),
                ])
                per_layer.append(packed.detach().cpu().tolist())
        if not per_layer:
            return {}
        n = len(per_layer)
        means = [sum(row[i] for row in per_layer) / n for i in range(8)]
        return {
            'alpha_c0':     means[0],
            'alpha_b0':     means[1],
            'alpha_c0_std': means[2],
            'alpha_b0_std': means[3],
            'alpha_mean':   means[4],
            'alpha_std':    means[5],
            'alpha_min':    means[6],
            'alpha_max':    means[7],
        }

    def _collect_kappa_stats(self) -> Dict[str, float]:
        """Collect learnable kappa diagnostics.

        Stacks per-block kappa tensors and pulls them in one
        ``.cpu().tolist()`` — previously fired one ``.item()`` sync per
        block per call.
        """
        if not self.cfg.learnable_kappa:
            return {}
        with torch.no_grad():
            # Accumulate scalar κ values across blocks AND heads (when
            # kappa_per_head=True, effective_kappa is (H,)). Flattening so
            # the mean/min/max are taken over the joined (n_blocks × n_heads)
            # population gives a single summary number per metric that
            # interprets the same whether per-head is on or off.
            scalars: List[float] = []
            for block in self.model.stack.blocks:
                k = block.e_step.effective_kappa
                if isinstance(k, torch.Tensor):
                    if k.dim() == 0:
                        scalars.append(float(k.detach().item()))
                    else:
                        scalars.extend(k.detach().flatten().cpu().tolist())
                else:
                    scalars.append(float(k))
        if not scalars:
            return {}
        return {
            'kappa_mean': sum(scalars) / len(scalars),
            'kappa_min': min(scalars),
            'kappa_max': max(scalars),
        }

    def _decode_first_sample_tokens(self) -> Optional[List[str]]:
        r"""Decode the most recent val-batch sample 0 to per-position
        token strings, for attention-heatmap axis labels.

        Returns ``None`` if no eval has populated `self._last_val_input_ids`
        yet, or if the dataset lacks a `.decode` method (e.g. synthetic
        TensorDataset). Each label is truncated to 6 characters to fit on
        tick marks.
        """
        ids = getattr(self, '_last_val_input_ids', None)
        if ids is None:
            return None
        ds = getattr(self.val_loader, 'dataset', None) if self.val_loader else None
        decode = getattr(ds, 'decode', None)
        if decode is None:
            return None
        try:
            # cl100k_base byte-pair tokens that span multiple chars may render
            # replacement boxes when the [:6] slice cuts mid-codepoint —
            # acceptable visual artifact, not a correctness bug.
            return [str(decode([int(t)]))[:6] for t in ids.tolist()]
        except (UnicodeDecodeError, ValueError, KeyError, RuntimeError) as e:
            logger.debug("axis-label decode failed: %s: %s", type(e).__name__, e)
            return None

    def _compute_per_head_beta(self, e_step: "VFEEStep") -> "Optional[_np_typing.ndarray]":
        r"""Compute per-head softmax β for sample 0 from the post-iteration state.

        Returns ``None`` when the cached ``_last_attention_state`` is absent
        (track_layer_diagnostics not active, fused path not used, or n_e_steps=0).
        Otherwise returns a numpy array of shape ``(H, N, N)`` with row-stochastic
        β_h matrices per gauge head.

        Math: for each block h with cached forward/inverse exp pairs
        ``(g_i^h, g_j^{-h})`` and post-iteration (μ_q^h, σ_q^h):

            μ_q_rope         = _apply_rope(μ_q, base=rope_base)                       if use_rope else μ_q
            μ_q^h            = μ_q_rope[..., block_h]                                 slice per head
            Ω_ij^h           = g_i^h @ g_j^{-h}                                       (N_i, N_j, d_h, d_h)
            μ_t[i,j,k]       = Σ_l Ω_ij^h[k,l] · μ_q^h[j,l]                           transport μ_q_j to i's frame
            σ_t[i,j,k]       = Σ_l Ω_ij^h[k,l]² · σ_q^h[j,l]    (diagonal approx)     transport σ_q_j to i's frame
            KL_h(i,j)        = ½ Σ_k [σ_q^h[i,k]/σ_t + (μ_q^h[i,k]-μ_t)²/σ_t
                                     - 1 - log(σ_q^h[i,k]/σ_t)]
            β_h(i,j)         = softmax_j(-KL_h(i,j) / (κ · √d_h))

        Σ is left un-rotated to match the training kernel's behaviour under
        ``rope_full_gauge='off'`` (CLAUDE.md "KNOWN GAP" clause); the diag-σ
        path forbids σ rotation at the config level.

        The training-time β at this position remains the collapsed (B,N,N)
        tensor stored in ``_last_attention`` -- this helper is diagnostic only.
        Computes only on sample 0 to keep memory bounded.
        """
        state = getattr(e_step, '_last_attention_state', None)
        if state is None:
            return None
        import torch as _t
        import numpy as _np
        from transformer.core.transport_ops import _apply_rope
        mu_q = state['mu_q'][0:1]            # (1, N, K)
        sigma_q = state['sigma_q'][0:1]       # (1, N, K)
        # Apply RoPE before slicing per head — the training kernel
        # (vfe_gradients.py:1074) rotates the full mu_q in one shot, and
        # because each head's d_h is a contiguous slice on K (the user's
        # block-diagonal generators guarantee this), slicing-then-rotating
        # the per-head pairs (2k, 2k+1) is bit-equivalent to
        # rotating-then-slicing. _apply_rope clones internally so the
        # cached state tensor is not mutated. Falls back to identity when
        # use_rope/rope_base are missing (snapshots written before the
        # 2026-05-19 RoPE-state extension).
        if state.get('use_rope', False):
            mu_q = _apply_rope(mu_q, base=float(state.get('rope_base', 10000.0)))
        bep = state['block_exp_pairs']
        _k = state['kappa']
        # Convert kappa to a Python float ONLY here (plot path), not in the
        # E-step iteration. One host sync per plot vs one per forward.
        if isinstance(_k, torch.Tensor):
            kappa = float(_k.detach().cpu().item())
        else:
            kappa = float(_k)
        irrep_dims = state['irrep_dims']
        if not irrep_dims:
            return None

        beta_per_head = []
        eps = 1e-6
        block_start = 0
        with _t.no_grad():
            # Hoist the causal mask outside the per-head loop — N is constant.
            N_seq = mu_q.shape[1]
            causal = _t.triu(
                _t.ones(N_seq, N_seq, device=mu_q.device, dtype=_t.bool),
                diagonal=1,
            )
            for h, d_h in enumerate(irrep_dims):
                block_end = block_start + d_h
                mu_h = mu_q[..., block_start:block_end]                    # (1, N, d_h)
                sig_h = sigma_q[..., block_start:block_end].clamp(min=eps)  # (1, N, d_h)
                exp_h, exp_h_inv = bep[h]
                if exp_h is None or exp_h_inv is None:
                    block_start = block_end
                    continue
                exp_h = exp_h[0:1]            # (1, N, d_h, d_h)
                exp_h_inv = exp_h_inv[0:1]    # (1, N, d_h, d_h)
                # Ω_ij^h: (1, N, N, d_h, d_h) -- materialize at sample-0 scale
                Omega = _t.einsum('bikm,bjml->bijkl', exp_h, exp_h_inv)
                mu_t = _t.einsum('bijkl,bjl->bijk', Omega, mu_h)            # (1, N, N, d_h)
                sig_t = _t.einsum('bijkl,bjl->bijk', Omega ** 2, sig_h).clamp(min=eps)
                mu_i = mu_h.unsqueeze(2)                                    # (1, N, 1, d_h)
                sig_i = sig_h.unsqueeze(2).clamp(min=eps)                   # (1, N, 1, d_h)
                kl = 0.5 * (
                    (sig_i / sig_t).sum(dim=-1)
                    + ((mu_i - mu_t) ** 2 / sig_t).sum(dim=-1)
                    - d_h
                    + _t.log(sig_t / sig_i).sum(dim=-1)
                )                                                            # (1, N, N)
                # Per-head softmax with per-head dim scaling τ_h = κ · √d_h.
                # Causal mask matches the kernel: queries can attend to keys at
                # equal or earlier positions only.
                tau_h = float(kappa) * (float(d_h) ** 0.5)
                logits = -kl / tau_h                                         # (1, N, N)
                logits = logits.masked_fill(causal, float('-inf'))
                beta_h = _t.softmax(logits, dim=-1)                          # (1, N, N)
                beta_per_head.append(beta_h[0].cpu().numpy())
                block_start = block_end

        if not beta_per_head:
            return None
        return _np.stack(beta_per_head, axis=0)  # (H, N, N)

    def _plot_attention_patterns(self, step: int) -> None:
        r"""Save publication-quality β heatmap PNGs, one file per (layer, head)
        panel, into ``output_dir/attention/attention_step_{step:08d}_L{layer}
        _H{head}.png`` (the collapsed-β fallback omits the ``_H{head}`` part).

        Per-head panels when ``block.e_step._last_attention_state`` is
        populated (the default). Falls back to a single collapsed-β panel
        per layer when the per-head state cache is missing.

        Sample 0 of the most recent val batch. Causal upper triangle is
        NaN-masked and rendered light grey. Color scale (log10 β over
        [-5, 0]) is shared across all panels by construction so PNGs are
        directly comparable across layers, heads, and steps. Failure to
        plot is logged and swallowed so a matplotlib glitch never aborts
        training.
        """
        if self.output_dir is None or not self.generate_figures:
            return
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from transformer.visualization.pub_style import set_pub_style
        except ImportError as exc:
            logger.warning(f"  matplotlib unavailable, skipping attention plot: {exc}")
            return

        try:
            blocks = list(self.model.stack.blocks)
            # Each panel: (layer_idx, head_idx_or_None, (N, N) array).
            panels: List[Tuple[int, Optional[int], Any]] = []
            for layer_idx, block in enumerate(blocks):
                # Prefer the actual training-time per-head βs when the
                # per-head softmax dispatch populated them (cfg.per_head_softmax
                # True AND fused path AND H > 1). This shows the same β tensor
                # the E-step gradients descended on, including RoPE. Falls
                # back to the state-snapshot reconstruction
                # (_compute_per_head_beta) when per-head βs are absent — e.g.,
                # single-softmax dispatch or single-head config.
                ph_tensor = getattr(block.e_step, '_last_attention_per_head', None)
                if ph_tensor is not None:
                    ph_np = ph_tensor[0].detach().cpu().numpy()  # (H, N, N)
                    for h in range(ph_np.shape[0]):
                        panels.append((layer_idx, h, ph_np[h]))
                    continue
                beta_per_head = self._compute_per_head_beta(block.e_step)
                if beta_per_head is not None:
                    for h in range(beta_per_head.shape[0]):
                        panels.append((layer_idx, h, beta_per_head[h]))
                else:
                    # Fallback: collapsed β (no per-head reconstruction available)
                    beta_t = getattr(block.e_step, '_last_attention', None)
                    if beta_t is None:
                        continue
                    panels.append((layer_idx, None, beta_t[0].detach().cpu().numpy()))
            if not panels:
                return

            labels = self._decode_first_sample_tokens()

            set_pub_style()
            log_floor = -5.0
            cmap = plt.get_cmap('viridis').copy()
            cmap.set_bad('#dddddd')

            out_dir = self.output_dir / 'attention'
            out_dir.mkdir(parents=True, exist_ok=True)

            saved: List[Path] = []
            for layer_idx, head_idx, b in panels:
                fig, ax = plt.subplots(figsize=(5.5, 4.8))
                log_b = np.log10(np.clip(b, 10.0 ** log_floor, 1.0))
                iu = np.triu_indices_from(log_b, k=1)
                log_b[iu] = np.nan
                im = ax.imshow(log_b, cmap=cmap, vmin=log_floor, vmax=0.0, aspect='equal')
                if head_idx is None:
                    panel_label = f'layer {layer_idx}'
                    suffix = f'L{layer_idx}'
                else:
                    panel_label = f'layer {layer_idx} | head {head_idx}'
                    suffix = f'L{layer_idx}_H{head_idx}'
                ax.set_title(rf'Attention $\beta$  |  step {step}  |  {panel_label}')
                ax.set_xlabel('key pos')
                ax.set_ylabel('query pos')
                if labels is not None:
                    N = b.shape[0]
                    stride = max(1, N // 16)
                    ticks = list(range(0, N, stride))
                    tlabels = [labels[t] for t in ticks if t < len(labels)]
                    if len(tlabels) == len(ticks):
                        ax.set_xticks(ticks)
                        ax.set_yticks(ticks)
                        ax.set_xticklabels(tlabels, rotation=45, ha='right', fontsize=7)
                        ax.set_yticklabels(tlabels, fontsize=7)
                ax.grid(False)
                cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
                cbar.set_label(r'$\log_{10} \beta_{ij}$')
                # TODO(future): for very long runs (>1M steps) consider a
                # rotating "keep last K" policy or a stride-by-K scheme.
                # At eval_interval=1000 / max_steps=30k with H heads this
                # produces ~30·H files / ~5·H MB total — no rotation needed.
                out_path = out_dir / f'attention_step_{step:08d}_{suffix}.png'
                fig.savefig(out_path, bbox_inches='tight')
                plt.close(fig)
                saved.append(out_path)
            logger.info(
                f"  attention plots saved: {len(saved)} file(s) under {out_dir}"
            )
        except (OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
            logger.exception("  attention plot failed: %s", exc)

    def _collect_cross_coupling_artifacts(
        self,
    ) -> Tuple[List[Dict[str, Any]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""Pull per-layer artifacts needed by ``cross_coupling_metrics`` /
        ``cross_coupling_viz`` from each block's ``_last_attention_state``.

        Returns:
            per_layer: list of dicts (one per block), each containing keys
                ``phi``, ``sigma``, ``omega``, ``beta_per_block``,
                ``omega_pairwise`` (any of which may be ``None`` if the
                cache is missing that artifact on this layer).
            generators: the model's generator buffer.
            cfg: the active VFEConfig.

        No-op safe: returns empty list when no block has populated state.
        """
        import torch as _t
        per_layer: List[Dict[str, Any]] = []
        for block in self.model.stack.blocks:
            e_step = block.e_step
            state = getattr(e_step, '_last_attention_state', None)
            if state is None:
                continue
            # Sample-0 slice to keep RAM bounded; the diagnostics aggregate
            # over (B, N) anyway, so a single sample is a faithful estimator.
            phi = state.get('phi')
            sigma = state.get('sigma_q')
            mu_q = state.get('mu_q')
            bep = state.get('block_exp_pairs')
            irrep_dims = state.get('irrep_dims') or []

            # Assemble (B, N, K, K) omega = block_diag(exp(phi · G^h)) from
            # the per-block exp pairs. Only the (sample-0) slice is kept.
            omega_full: Optional[_t.Tensor] = None
            omega_pairwise: Optional[List[Tuple[_t.Tensor, _t.Tensor]]] = None
            if bep is not None and irrep_dims:
                # Use sample 0 only for the global-Ω heatmap.
                exp_h0_list = []
                exp_neg_h0_list = []
                ok = True
                for p in bep:
                    if p[0] is None:
                        ok = False
                        break
                    exp_h0_list.append(p[0][0:1])         # (1, N, d_h, d_h)
                    exp_neg_h0_list.append(
                        p[1][0:1] if p[1] is not None else p[0][0:1].transpose(-1, -2)
                    )
                if ok:
                    K = sum(irrep_dims)
                    N = exp_h0_list[0].shape[1]
                    omega_full = _t.zeros(
                        1, N, K, K,
                        device=exp_h0_list[0].device,
                        dtype=exp_h0_list[0].dtype,
                    )
                    cursor = 0
                    for d_h, eh in zip(irrep_dims, exp_h0_list):
                        omega_full[..., cursor:cursor + d_h, cursor:cursor + d_h] = eh
                        cursor += d_h
                    # Pairwise per-block (sample 0). Materialize Ω_ij^h =
                    # exp_h(i) · exp_h_inv(j).
                    omega_pairwise = []
                    for d_h, eh, eh_inv in zip(
                        irrep_dims, exp_h0_list, exp_neg_h0_list,
                    ):
                        Om = _t.einsum('bikm,bjml->bijkl', eh, eh_inv)
                        Om_inv = _t.einsum('bjkm,biml->bijkl', eh_inv, eh)
                        omega_pairwise.append((Om, Om_inv))

            # Beta per super-block: stack the per-head (super-block)
            # attention tensors when the e_step has produced them. Convert
            # the trainer's ``_last_attention_per_head`` (B, H, N, N) into
            # (B, N, N, H_super) as the metric expects.
            beta_per_block: Optional[_t.Tensor] = None
            ph = getattr(e_step, '_last_attention_per_head', None)
            if ph is not None:
                # ph: (B, H, N, N) — permute to (B, N, N, H)
                beta_per_block = ph.detach().permute(0, 2, 3, 1).contiguous()

            per_layer.append(dict(
                phi=phi[0:1] if phi is not None else None,
                sigma=sigma[0:1] if sigma is not None else None,
                mu_q=mu_q[0:1] if mu_q is not None else None,
                omega=omega_full,
                omega_pairwise=omega_pairwise,
                beta_per_block=beta_per_block,
            ))
        return per_layer, self.model.generators, self.model.cfg

    def _save_cross_coupling_diagnostics(self, step: int) -> Dict[str, Any]:
        r"""Compute cross-coupling metrics and save figures at ``step``.

        No-op when ``cfg.cross_couplings`` is empty or the output dir is
        unset; safe to call unconditionally from the eval branch. Emits two
        artifacts:
        - ``cross_coupling/cross_coupling_step_{step}.json`` — scalar
          metrics per layer plus per-super-block arrays serialized to lists.
        - ``cross_coupling/cross_coupling_step_{step}_*.png`` — figures
          generated by ``cross_coupling_viz`` (one per plot type).

        Returns the computed metrics dict so callers can also forward
        scalar entries to a CSV / pub tracker.
        """
        if self.output_dir is None:
            return {}
        cfg = self.model.cfg
        if not cfg.is_cross_coupled:
            return {}
        try:
            import numpy as _np
            import matplotlib.pyplot as _plt
            from transformer.vfe import cross_coupling_metrics as _ccm
            from transformer.vfe import cross_coupling_viz as _ccv
        except ImportError as exc:
            logger.warning(
                f"  cross-coupling diagnostics: import failed ({exc}); skipping."
            )
            return {}

        try:
            per_layer, generators, _ = self._collect_cross_coupling_artifacts()
            if not per_layer:
                return {}

            out_dir = self.output_dir / 'cross_coupling'
            out_dir.mkdir(parents=True, exist_ok=True)

            per_layer_metrics: List[Dict[str, Any]] = []
            for layer_idx, layer in enumerate(per_layer):
                if layer.get('phi') is None or layer.get('sigma') is None:
                    per_layer_metrics.append({})
                    continue
                m = _ccm.collect_cross_coupling_metrics(
                    phi=layer['phi'],
                    sigma=layer['sigma'],
                    cfg=cfg,
                    omega=layer.get('omega'),
                    omega_pairwise=layer.get('omega_pairwise'),
                    beta_per_block=layer.get('beta_per_block'),
                    kl_per_block=None,  # not currently cached per super-block
                )
                # Convert numpy arrays to lists for JSON serialization.
                m_json = {
                    k: (v.tolist() if hasattr(v, 'tolist') else v)
                    for k, v in m.items()
                }
                per_layer_metrics.append(m_json)

            # Persist scalar/array metrics as JSON.
            json_path = out_dir / f'cross_coupling_step_{step:08d}.json'
            with open(json_path, 'w') as fh:
                json.dump({
                    'step': step,
                    'cross_couplings': list(cfg.cross_couplings),
                    'super_block_dims': cfg.super_block_dims,
                    'super_block_head_groups': cfg.super_block_head_groups,
                    'per_layer': per_layer_metrics,
                }, fh, indent=2, default=str)

            # Generate figures from layer-0 state (representative; the
            # generator basis and super-block partition are layer-invariant).
            layer0 = per_layer[0]
            saved_figs: List[str] = []

            def _save(name: str, fig) -> None:
                p = out_dir / f'cross_coupling_step_{step:08d}_{name}.png'
                fig.savefig(p, bbox_inches='tight', dpi=200)
                _plt.close(fig)
                saved_figs.append(p.name)

            _save('generator_sparsity',
                  _ccv.plot_generator_sparsity(generators, cfg))
            _save('super_block_graph',
                  _ccv.plot_super_block_graph(cfg))
            if layer0.get('omega') is not None:
                strength = _ccm.omega_block_strength(layer0['omega'], cfg)
                _save('omega_block_strength',
                      _ccv.plot_omega_block_strength(strength, cfg))

            # Layer-aggregated plots: phi-energy partition + diagnostics bar.
            phi_energy_per_layer = []
            attn_entropy_per_layer = []
            for layer in per_layer:
                if layer.get('phi') is None:
                    continue
                phi_energy_per_layer.append(
                    _ccm.phi_energy_partition(layer['phi'], cfg)
                )
                if layer.get('beta_per_block') is not None:
                    attn_entropy_per_layer.append(
                        _ccm.per_super_block_attention_entropy(
                            layer['beta_per_block'], cfg,
                        )
                    )
            if phi_energy_per_layer:
                _save('phi_energy',
                      _ccv.plot_phi_energy_partition(phi_energy_per_layer, cfg))

            eff_rank = _ccm.per_super_block_effective_rank(layer0['sigma'], cfg)
            attn_ent0 = (
                _ccm.per_super_block_attention_entropy(layer0['beta_per_block'], cfg)
                if layer0.get('beta_per_block') is not None else None
            )
            _save('super_block_diagnostics',
                  _ccv.plot_super_block_diagnostics(
                      effective_rank=eff_rank,
                      attention_entropy=attn_ent0,
                      cfg=cfg,
                  ))

            logger.info(
                f"  cross-coupling diagnostics saved: "
                f"{len(saved_figs)} figure(s), 1 JSON under {out_dir}"
            )
            # Aggregate scalar summary for CSV/log return.
            agg: Dict[str, Any] = {}
            if phi_energy_per_layer:
                agg['cc_phi_energy_cross_share'] = float(_np.mean(
                    [d['phi_energy_cross_share'] for d in phi_energy_per_layer]
                ))
                agg['cc_phi_energy_cross'] = float(_np.mean(
                    [d['phi_energy_cross'] for d in phi_energy_per_layer]
                ))
                agg['cc_phi_energy_total'] = float(_np.mean(
                    [d['phi_energy_total'] for d in phi_energy_per_layer]
                ))
            return agg
        except (OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
            logger.exception("  cross-coupling diagnostics failed: %s", exc)
            return {}

    def train_step(
        self,
        batch: "Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]",
    ) -> Dict[str, float]:
        r"""Single training step with full metrics collection.

        Args:
            batch: Either a dict with ``'input_ids'`` and ``'target_ids'``,
                or a 2-tuple ``(input_ids, target_ids)`` (the legacy
                TensorDataset form). Both are accepted by the dataloaders
                this trainer consumes.

        Returns:
            Comprehensive metrics dict for logging and CSV output.
        """
        self.model.train()
        t0 = time.time()
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(self.device)
            target_ids = batch[1].to(self.device)
        else:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

        # E-step: infer q* from context only (no target in E-step)
        # M-step: CE loss from q* → target
        # ce_for_log is the unscaled, regularizer-free CE (used for PPL/BPC reporting);
        # `loss` is the optimizer's combined target (may include 1/sqrt(K) scaling
        # if normalize_ce_by_dim=True and/or 0.5·mass_phi·||phi||² regularizer).
        logits, loss, ce_for_log = self.model(input_ids, targets=target_ids)

        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Collect gradient norms BEFORE clipping — only on log steps. The
        # per-group cat+norm+tolist costs a CUDA sync and ~80MB of concat on
        # the phi group at default V; the result is only consumed at log
        # boundaries (lines 727-732 and `_log_to_csv`), so on non-log steps
        # the work is wasted. The zero-fallback dict preserves downstream
        # dict reads.
        if self._aggregate_diagnostics_this_step:
            grad_norms = self._compute_gradient_norms()
        else:
            grad_norms = {
                'mu': 0.0, 'sigma': 0.0, 'phi': 0.0,
                'ffn': 0.0, 'other': 0.0, 'total': 0.0,
            }

        # Gradient clipping
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )

        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        step_time = time.time() - t0
        # NOTE: a previous draft deferred `loss.item()` / `ce_for_log.item()`
        # to log boundaries to save 2 CUDA syncs per non-log step. Reverted
        # because deferral masks NaN/Inf losses on non-log steps — early-
        # training divergence would go undetected for up to log_interval
        # steps. The smoke test in `tests/smoke_test_entropy.py` relies on
        # per-step `metrics['loss']` for its non-finite assertion. The 1-5 ms
        # per-step sync cost is not worth losing that diagnostic.
        loss_val = loss.item()        # combined optimizer loss (CE_scaled + mass_phi reg)
        ce_val = ce_for_log.item()    # unscaled, regularizer-free CE
        ppl = math.exp(min(ce_val, 20.0))
        bpc = ce_val / math.log(2)
        lr = self.scheduler.get_last_lr()[0]

        # Build comprehensive metrics dict
        B, N = input_ids.shape
        metrics = {
            # Core loss
            'loss': loss_val,    # combined optimizer loss
            'ce': ce_val,        # unscaled cross-entropy (PPL/BPC source of truth)
            'ppl': ppl,
            'bpc': bpc,
            'lr': lr,
            # Timing
            'step_time': step_time,
            'tokens_per_sec': (B * N) / step_time if step_time > 0 else 0,
            # Gradient norms (M-step)
            'grad_norm_total': grad_norms['total'],
            'grad_norm_mu': grad_norms['mu'],
            'grad_norm_sigma': grad_norms['sigma'],
            'grad_norm_phi': grad_norms['phi'],
            'grad_norm_ffn': grad_norms['ffn'],
            'grad_norm_other': grad_norms['other'],
        }

        # Diagnostic aggregation (per-layer iterate + .item() syncs) only
        # fires when the training loop has flagged this step for logging.
        # On non-log steps the result is built but discarded by the caller,
        # so skipping spares N_layers · (8 + 13 + 1) host syncs per step.
        if self._aggregate_diagnostics_this_step:
            e_diag = self._collect_e_step_diagnostics()
            metrics.update({f'{k}': v for k, v in e_diag.items()})
            metrics.update(self._collect_bayesian_alpha_stats())
            metrics.update(self._collect_kappa_stats())

        return metrics

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None, max_samples: int = 12800) -> Dict[str, float]:
        """Evaluate on validation set.

        Args:
            loader: Data loader to evaluate on. Defaults to self.val_loader.
            max_samples: Maximum samples to evaluate (prevents hang on large datasets).

        Returns:
            Dict with 'val_loss', 'val_ppl', 'val_bpc'.
        """
        loader = loader or self.val_loader
        if loader is None:
            return {}

        was_training = self.model.training
        self.model.eval()
        # Accumulate on-device. One .item() sync per loop instead of 2/batch.
        total_ce_gpu: Optional[torch.Tensor] = None
        total_tokens_gpu: Optional[torch.Tensor] = None
        total_samples = 0

        first_batch_seen = False
        for batch in loader:
            if total_samples >= max_samples:
                break

            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
                target_ids = batch[1].to(self.device)
            else:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)

            # Capture sample 0 of the first val batch for attention-heatmap
            # axis labels.
            if not first_batch_seen:
                self._last_val_input_ids = input_ids[0].detach().cpu()
                first_batch_seen = True

            _, _, ce_for_log = self.model(input_ids, targets=target_ids)
            n_tokens_t = (target_ids != -100).sum()
            ce_contrib = ce_for_log * n_tokens_t.to(ce_for_log.dtype)
            if total_ce_gpu is None:
                total_ce_gpu = ce_contrib
                total_tokens_gpu = n_tokens_t
            else:
                total_ce_gpu = total_ce_gpu + ce_contrib
                total_tokens_gpu = total_tokens_gpu + n_tokens_t
            total_samples += input_ids.shape[0]

        total_tokens = int(total_tokens_gpu.item()) if total_tokens_gpu is not None else 0
        total_ce = float(total_ce_gpu.item()) if total_ce_gpu is not None else 0.0
        avg_ce = total_ce / max(total_tokens, 1)
        if was_training:
            self.model.train()
        # `val_loss` semantically means avg unscaled CE (matches val_ppl/val_bpc).
        # Default configs (mass_phi=0, normalize_ce_by_dim=False) are unaffected.
        return {
            'val_loss': avg_ce,
            'val_ppl': math.exp(min(avg_ce, 20.0)),
            'val_bpc': avg_ce / math.log(2),
        }

    @torch.no_grad()
    def run_test_evaluation(self, max_samples: int = 128000) -> Dict[str, float]:
        r"""Final end-of-training evaluation on the held-out test split.

        Mirrors the publication path's ``run_test_evaluation`` in
        ``experiment_runner.py:119-239``: token-weighted CE across up to
        ``max_samples`` test samples (default 128000 = ~2000 batches at
        ``batch_size=64``, matching the publication default for consistency
        across configs with different batch sizes), then reports test PPL,
        BPC (corrected by ``tokens_per_char`` when available), random
        baseline ``= vocab_size``, and improvement factor. Uses the same
        ``VFEModel.forward`` path as ``evaluate()`` — pure CE, no VFE
        regularizer rescaling — so the reported test loss is directly
        comparable to ``val_loss`` from the periodic evaluations.

        Returns an empty dict if ``self.test_loader`` is ``None``.
        """
        loader = self.test_loader
        if loader is None:
            return {}

        logger.info("=" * 70)
        logger.info("FINAL TEST SET EVALUATION")
        logger.info("=" * 70)
        logger.info(f"  Evaluating up to {max_samples} samples...")

        was_training = self.model.training
        self.model.eval()
        total_ce_gpu: Optional[torch.Tensor] = None
        total_tokens_gpu: Optional[torch.Tensor] = None
        total_samples = 0
        num_batches = 0

        for batch in loader:
            if total_samples >= max_samples:
                break
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
                target_ids = batch[1].to(self.device)
            else:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)

            _, _, ce_for_log = self.model(input_ids, targets=target_ids)
            n_tokens_t = (target_ids != -100).sum()
            ce_contrib = ce_for_log * n_tokens_t.to(ce_for_log.dtype)
            if total_ce_gpu is None:
                total_ce_gpu = ce_contrib
                total_tokens_gpu = n_tokens_t
            else:
                total_ce_gpu = total_ce_gpu + ce_contrib
                total_tokens_gpu = total_tokens_gpu + n_tokens_t
            total_samples += input_ids.shape[0]
            num_batches += 1

            if num_batches % 100 == 0:
                logger.info(
                    f"  Evaluated {total_samples}/{max_samples} samples "
                    f"({num_batches} batches)..."
                )

        total_tokens = int(total_tokens_gpu.item()) if total_tokens_gpu is not None else 0
        total_ce_tokens = float(total_ce_gpu.item()) if total_ce_gpu is not None else 0.0
        if was_training:
            self.model.train()

        test_ce = total_ce_tokens / max(total_tokens, 1)
        test_ppl = math.exp(min(test_ce, 20.0))
        # BPC = (CE_nats / ln 2) * tokens_per_char. The tokens-per-char ratio
        # comes from the test loader's dataset; when unavailable we fall back
        # to bits-per-token with a one-time warning, matching the publication
        # path's bpc_from_dataset behavior.
        from transformer.training.bpc import bpc_from_dataset
        test_bpc = bpc_from_dataset(
            test_ce, loader, fallback_key='vfe_run_test_evaluation',
        )
        random_ppl = float(self.cfg.vocab_size)
        improvement = (random_ppl / test_ppl) if test_ppl > 0 else 0.0

        logger.info(
            f"Test Set Results ({total_samples} samples across "
            f"{num_batches} batches):"
        )
        logger.info(f"  Cross-entropy loss: {test_ce:.4f}")
        logger.info(f"  Perplexity:         {test_ppl:.2f}")
        logger.info(f"  Bits per character: {test_bpc:.3f}")
        logger.info(f"  Random baseline:    {random_ppl:.0f}")
        logger.info(f"  Improvement:        {improvement:.1f}x better than random")
        logger.info("=" * 70)

        results = {
            'test_loss': test_ce,
            'test_ppl': test_ppl,
            'test_bpc': test_bpc,
            'random_ppl': random_ppl,
            'improvement': improvement,
            'num_samples': total_samples,
            'num_batches': num_batches,
        }

        if self.output_dir is not None:
            import json
            out_path = self.output_dir / 'test_results.json'
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"  Test results saved: {out_path}")

        return results

    # Columns the vfe/ training path cannot populate. They depend on
    # PublicationTrainer-only instrumentation (_compute_phi_diagnostics in
    # experiment_runner.py:439, _VFE_GRAD_DEBUG in core/vfe_utils.py, and
    # numerical_monitor flush) that the vfe/ package does not import. Listed
    # here so the CSV header doesn't carry permanently empty columns.
    _CSV_DISABLED_COLUMNS = (
        # No second-moment stat for kl/kappa in vfe diagnostics
        'kl_std',
        'kappa_std',
        # Bayesian Mahalanobis surface not produced by vfe e_step
        'alpha_mahal_sq_mean', 'alpha_mahal_sq_std',
        # numerical_monitor not wired into vfe/
        'num_chol_recover', 'num_chol_fail', 'num_nan_replace', 'num_inv_pinv',
        # _compute_phi_diagnostics lives only on PublicationTrainer
        'phi_effective_rank', 'phi_rank_ratio',
        'phi_top1_variance_fraction', 'phi_top5_variance_fraction',
        'phi_spectral_gap', 'phi_frobenius_norm',
        'phi_mean_token_norm', 'phi_std_token_norm',
        # _VFE_GRAD_DEBUG instrumentation lives in core/vfe_gradients.py,
        # not in the vfe/ E-step
        'vfe_grad_mu_self', 'vfe_grad_mu_direct',
        'vfe_grad_mu_softmax', 'vfe_grad_mu_total',
        'vfe_grad_sigma_self', 'vfe_grad_sigma_align_direct',
        'vfe_grad_sigma_softmax', 'vfe_grad_sigma_total',
        'vfe_kl_pairwise_mean', 'vfe_kl_pairwise_max', 'vfe_kappa_scaled',
        'vfe_kl_frac_above_90pct', 'vfe_kl_p95',
        # Condition-number and min/max for sigma_p not computed in vfe e_step
        'sigma_q_cond_mean', 'sigma_q_cond_max',
        'sigma_p_min', 'sigma_p_max',
        # Transport pairwise stats not collected in vfe diagnostics
        'phi_pairwise_dist_mean', 'phi_pairwise_dist_max',
        'attn_entropy_per_head_mean', 'attn_entropy_per_head_std',
        'attn_entropy_per_head_min', 'attn_entropy_per_head_max',
        'head_correlation_mean',
    )

    def _init_metrics(self) -> None:
        """Initialize metrics infrastructure (CSV + publication tracker)."""
        if self.output_dir is None:
            return
        try:
            from transformer.training.metrics_tracking import PublicationMetricsTracker
            from transformer.training.bpc import tokens_per_char_from_dataset
            csv_path = self.output_dir / 'metrics.csv'
            self._metrics_tracker = PublicationMetricsTracker(
                csv_path,
                tokens_per_char=tokens_per_char_from_dataset(
                    getattr(self, 'train_loader', None),
                ),
                disabled_columns=self._CSV_DISABLED_COLUMNS,
            )
            logger.info(f"Metrics CSV: {csv_path}")
        except ImportError:
            logger.warning("PublicationMetricsTracker not available — CSV logging disabled")

        try:
            from transformer.analysis.publication_metrics import TrainingTracker
            from transformer.training.bpc import tokens_per_char_from_dataset
            self._pub_tracker = TrainingTracker(
                save_dir=(self.output_dir / 'figures') if self.output_dir else None,
                tokens_per_char=tokens_per_char_from_dataset(
                    getattr(self, 'train_loader', None),
                ),
            )
            logger.info("Publication metrics tracker initialized")
        except ImportError:
            logger.warning("TrainingTracker not available — publication metrics disabled")

    def _log_to_csv(self, step: int, metrics: Dict[str, float], batch_size: int, seq_len: int) -> None:
        """Log metrics to CSV via PublicationMetricsTracker."""
        if self._metrics_tracker is None:
            return

        # VFE decomposed loss components (belief_align, self_consistency,
        # model_coupling) are not recovered in this training
        # path — the VFE objective is minimized implicitly via the E-step
        # inner loop, so those columns are omitted rather than logged as
        # misleading zeros.
        #
        # Key-namespace remap: PublicationMetricsTracker.log_step reads
        # most diagnostic columns via prefixed lookups (``bayesian/*``,
        # ``cov/*``, ``transport/*``, ``kappa/*``) — see
        # metrics_tracking.py:171-250. The vfe E-step writes bare names
        # into ``_last_diagnostics`` (e_step.py:829), so we re-emit each
        # bare value under both names: the bare name is harmless filler,
        # the prefixed name is what actually populates the column. ``None``
        # is preferred over ``0`` for un-collected metrics so the CSV cell
        # reads as blank (matching publication-side semantics) rather than
        # as a misleading literal zero.
        # NOTE: ``train_loss_total`` is the optimizer's target loss. Under
        # ``normalize_ce_by_dim=True`` (default for VFE-style training) this is
        # ``(CE_raw + mass_phi_reg + sum_aux_hyperparam) / sqrt(K)``. The raw
        # cross-entropy is also exposed under ``train_loss_ce`` / ``_raw`` so
        # readers do not have to infer the rescaling — both columns are
        # included to make the relationship obvious in dashboards.
        csv_metrics = {
            # Bare-name lookups (metrics_tracking.py:136-157)
            'train_loss_total': metrics['loss'],
            'train_loss_ce': metrics['ce'],
            'train_loss_ce_raw': metrics['ce'],
            'train_ppl': metrics['ppl'],
            'train_bpc': metrics['bpc'],
            'beta_mean': metrics.get('beta_mean'),
            'beta_std': metrics.get('beta_std'),
            'kl_mean': metrics.get('kl_mean'),
            'attention_entropy': metrics.get('attention_entropy'),
            'attention_concentration': metrics.get('attention_concentration'),
            # Bayesian alpha (bare → bayesian/* prefix)
            'bayesian/alpha_mean': metrics.get('alpha_mean'),
            'bayesian/alpha_std': metrics.get('alpha_std'),
            'bayesian/alpha_min': metrics.get('alpha_min'),
            'bayesian/alpha_max': metrics.get('alpha_max'),
            'bayesian/c0': metrics.get('alpha_c0'),
            'bayesian/b0': metrics.get('alpha_b0'),
            'bayesian/c0_std': metrics.get('alpha_c0_std'),
            'bayesian/b0_std': metrics.get('alpha_b0_std'),
            # Learnable kappa (bare → kappa/per_head_* prefix)
            'kappa/per_head_mean': metrics.get('kappa_mean'),
            'kappa/per_head_min': metrics.get('kappa_min'),
            'kappa/per_head_max': metrics.get('kappa_max'),
            # Covariance health (bare → cov/* prefix)
            'cov/sigma_q_mean': metrics.get('sigma_q_mean'),
            'cov/sigma_q_min': metrics.get('sigma_q_min'),
            'cov/sigma_q_max': metrics.get('sigma_q_max'),
            'cov/sigma_q_std': metrics.get('sigma_q_std'),
            'cov/sigma_p_mean': metrics.get('sigma_p_mean'),
            'cov/prior_belief_kl_mean': metrics.get('prior_belief_kl_mean'),
            'cov/prior_belief_kl_max': metrics.get('prior_belief_kl_max'),
            'cov/prior_belief_kl_std': metrics.get('prior_belief_kl_std'),
            # Transport (bare → transport/* prefix)
            'transport/phi_norm_mean': metrics.get('phi_norm_mean'),
            'transport/phi_norm_std': metrics.get('phi_norm_std'),
            'transport/phi_norm_max': metrics.get('phi_norm_max'),
        }

        grad_norms = {
            'total': metrics.get('grad_norm_total', 0),
            'mu': metrics.get('grad_norm_mu', 0),
            'sigma': metrics.get('grad_norm_sigma', 0),
            'phi': metrics.get('grad_norm_phi', 0),
            'ffn': metrics.get('grad_norm_ffn', 0),
        }

        lrs = {}
        # Map current param-group names → legacy CSV column names so downstream
        # analysis tooling that expects {mu_embed, sigma_embed, phi_embed, ffn,
        # other} keeps working after the m_*_lr rename. Groups not in the map
        # (none today) pass through under their own name. Map is shared with
        # PublicationMetricsTracker.log_step so both ends stay in lock-step.
        from transformer.training.metrics_tracking import VFE_LR_GROUP_MAP
        for group in self.optimizer.param_groups:
            name = group.get('name', 'default')
            lrs[VFE_LR_GROUP_MAP.get(name, name)] = group['lr']

        self._metrics_tracker.log_step(
            step, csv_metrics, lrs, grad_norms,
            metrics.get('step_time', 0), batch_size, seq_len,
        )

    def _log_to_pub_tracker(self, step: int, metrics: Dict[str, float]) -> None:
        """Log metrics to publication tracker for figure generation."""
        if self._pub_tracker is None:
            return

        train_metrics = {
            'loss': metrics['loss'],
            'ce_loss': metrics['ce'],
            'ce_loss_raw': metrics['ce'],
            'attention_entropy': metrics.get('attention_entropy', 0),
            'attention_concentration': metrics.get('attention_concentration', 0),
        }

        grad_norms = {
            'total': metrics.get('grad_norm_total', 0),
            'mu': metrics.get('grad_norm_mu', 0),
            'sigma': metrics.get('grad_norm_sigma', 0),
            'phi': metrics.get('grad_norm_phi', 0),
            'ffn': metrics.get('grad_norm_ffn', 0),
            'other': metrics.get('grad_norm_other', 0),
        }

        e_step_norms = {
            'nat_grad_mu': metrics.get('nat_grad_mu_norm', 0),
            'nat_grad_sigma': metrics.get('nat_grad_sigma_norm', 0),
        }

        B = self.cfg.batch_size
        N = self.cfg.max_seq_len
        self._pub_tracker.record(
            step=step,
            epoch=step / max(len(self.train_loader), 1),
            train_metrics=train_metrics,
            grad_norms=grad_norms,
            lr=metrics.get('lr', 0),
            step_time=metrics.get('step_time', 0),
            batch_size=B,
            seq_len=N,
            e_step_norms=e_step_norms,
        )

    def _save_checkpoint(self, step: int) -> None:
        """Save model checkpoint."""
        if self.output_dir is None:
            return
        ckpt_dir = self.output_dir / 'checkpoints'
        ckpt_dir.mkdir(exist_ok=True)
        path = ckpt_dir / f'step_{step}.pt'
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        logger.info(f"  Checkpoint saved: {path}")

    def _generate_figures(self) -> None:
        """Generate publication figures at end of training."""
        if self._pub_tracker is None or self.output_dir is None:
            return
        if not self.generate_figures:
            return
        try:
            from transformer.analysis.publication_metrics import PublicationFigures
            fig_dir = self.output_dir / 'figures'
            fig_dir.mkdir(exist_ok=True)
            figures = PublicationFigures(fig_dir)
            figures.plot_training_curves(self._pub_tracker)
            figures.plot_gradient_norms_split(self._pub_tracker)
            logger.info(f"Publication figures saved to {fig_dir}")
        except (OSError, ImportError, ValueError, RuntimeError) as e:
            # logger.exception attaches the traceback so a real bug surfaces;
            # the bare-Exception form previously downgraded AttributeError /
            # TypeError programming bugs to a single WARN line.
            logger.exception(f"Figure generation failed: {e}")

        # VFE dynamics dashboard from CSV
        if self._metrics_tracker is not None:
            try:
                from transformer.visualization.vfe_dynamics_plots import (
                    generate_all_vfe_figures,
                )
                vfe_fig_dir = self.output_dir / 'vfe_dynamics_figures'
                vfe_fig_dir.mkdir(exist_ok=True)
                csv_path = self.output_dir / 'metrics.csv'
                generate_all_vfe_figures(csv_path, vfe_fig_dir)
                logger.info(f"VFE dynamics figures saved to {vfe_fig_dir}")
            except (OSError, ImportError, ValueError, RuntimeError, KeyError) as e:
                logger.exception(f"VFE dynamics figure generation failed: {e}")

        # plot_training_curves / plot_gradient_norms_split return Figure
        # objects that they savefig'd but did not close — close everything
        # left behind so figures don't accumulate across runs (the
        # "More than 20 figures have been opened" RuntimeWarning).
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except ImportError:
            pass

    def train(self, num_steps: Optional[int] = None, log_interval: Optional[int] = None) -> None:
        """Main training loop with full metrics, checkpoints, and figure generation.

        Args:
            num_steps: Number of steps to train. Defaults to cfg.max_steps.
            log_interval: Log every N steps. Defaults to cfg.log_interval.
        """
        num_steps = num_steps or self.cfg.max_steps
        log_interval = log_interval or self.cfg.log_interval
        eval_interval = self.cfg.eval_interval
        checkpoint_interval = self.cfg.checkpoint_interval
        _epoch_idx = 0
        _fn = getattr(self.train_loader.dataset, "set_epoch", None)
        if callable(_fn):
            _fn(_epoch_idx)
        data_iter = iter(self.train_loader)
        t0 = time.time()

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Starting training: {num_steps} steps, {n_params:,} params, device={self.device}")

        # Initialize metrics infrastructure
        self._init_metrics()

        best_val_loss = float('inf')

        # On Windows the default console codec (cp1252) cannot encode the
        # Greek letter β used in the log/validation format. Reconfigure
        # stdout/stderr to UTF-8 with replacement so the bar + tqdm.write
        # never raise UnicodeEncodeError mid-training. No-op on platforms
        # where the codec already handles it.
        import sys as _sys
        for _stream in (_sys.stdout, _sys.stderr):
            _reconf = getattr(_stream, 'reconfigure', None)
            if callable(_reconf):
                # Narrow exception list — UnicodeEncodeError is the realistic
                # failure on Windows consoles; AttributeError covers reduced
                # stream wrappers. Anything else (KeyboardInterrupt, OSError)
                # should propagate.
                try:
                    _reconf(encoding='utf-8', errors='replace')
                except (AttributeError, ValueError, UnicodeError):
                    pass

        # tqdm bar. Matches `experiment_runner.py:1252-1263` exactly: plain
        # `tqdm(range(...), desc="Training", initial=0, total=num_steps)` with
        # no `dynamic_ncols` (which queries terminal size every refresh and
        # misbehaves in some non-TTY consumers) and no `mininterval`/`leave`
        # overrides (tqdm's defaults render correctly in PowerShell and bash).
        # The description stays STATIC; tqdm's built-in postfix supplies live
        # `it/s`. The formatted Step|Loss|... line emits via `tqdm.write` only
        # at log_interval boundaries (above the bar in scrollback).
        if _HAS_TQDM:
            pbar = _tqdm(
                range(num_steps),
                desc="Training",
                initial=0,
                total=num_steps,
            )
            _write = _tqdm.write
            _iter = pbar
        else:
            pbar = None
            _write = print
            _iter = range(num_steps)

        for step in _iter:
            # Get next batch (with tuple or dict support)
            try:
                batch = next(data_iter)
            except StopIteration:
                _epoch_idx += 1
                _fn = getattr(self.train_loader.dataset, "set_epoch", None)
                if callable(_fn):
                    _fn(_epoch_idx)
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Handle both tuple (input_ids, target_ids) and dict formats
            if isinstance(batch, (list, tuple)):
                batch = {'input_ids': batch[0], 'target_ids': batch[1]}

            # Tell train_step whether this step's diagnostics will be
            # consumed by the log/CSV writer. On non-log steps train_step
            # still computes loss + grads, but skips the per-layer
            # diagnostic aggregation (which dominates non-GPU CPU sync time
            # for n_layers > 1).
            self._aggregate_diagnostics_this_step = (
                (step + 1) % log_interval == 0
            )
            # Attention-state capture is only needed on the forward that
            # immediately precedes the periodic attention plot, which fires
            # inside evaluate() — gate the train-step forward off.
            self._set_capture_attention_state(False)
            metrics = self.train_step(batch)

            # CSV + publication-tracker row is written every log_interval
            # steps, matching the publication-side cadence at
            # experiment_runner.py:1321,1327. Without this gate the CSV
            # accumulates one (mostly empty) row per step and val rows land
            # on steps that have no train row to update.
            is_log_step = (step + 1) % log_interval == 0
            if is_log_step:
                # batch was normalised to a dict at line 1260; index by key.
                _ids = batch['input_ids']
                B, N = _ids.shape[0], _ids.shape[1]
                self._log_to_csv(step + 1, metrics, B, N)
                self._log_to_pub_tracker(step + 1, metrics)

            # Formatted per-log-interval emit. Live per-step it/s is shown by
            # tqdm's own postfix; we never overwrite the bar's description.
            # `_attention_summary()` requires a `.tolist()` sync, so it only
            # runs at log boundaries (not every step). The `set_description`
            # call before `_write` mirrors `experiment_runner.py:1383-1384` --
            # tqdm's `set_description` defaults to `refresh=True`, which forces
            # a bar redraw and flushes any pending bar state to the console
            # (important in IPython / Spyder consoles that buffer stderr).
            if (step + 1) % log_interval == 0:
                if pbar is not None:
                    rate = pbar.format_dict.get('rate') or 0.0
                else:
                    step_time = metrics.get('step_time', 0.0)
                    rate = (1.0 / step_time) if step_time > 0 else 0.0
                attn = self._attention_summary()
                beta_kl = attn.get('beta_kl', float('nan'))
                # `beta_kl` (mean β·KL alignment energy) is undefined for E-step
                # paths that surface no KL matrix (e.g. rope_full_gauge). Fall
                # back to the always-available attention entropy H(β) so the
                # line reports a real β diagnostic instead of a bare `nan`.
                if math.isfinite(beta_kl):
                    beta_field = f"β·KL: {beta_kl:.4f}"
                else:
                    beta_field = f"H(β): {attn.get('attn_entropy', float('nan')):.3f}"
                msg = (
                    f"Step {step+1}/{num_steps} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"CE: {metrics['ce']:.4f} | "
                    f"{beta_field} | "
                    f"PPL: {metrics['ppl']:.1f} | "
                    f"it/s: {rate:.2f}"
                )
                if pbar is not None:
                    pbar.set_description("Training")  # refresh=True (default) flushes bar
                _write(msg)

            # Periodic evaluation
            if self.val_loader is not None and (step + 1) % eval_interval == 0:
                # Re-enable per-block attention-state capture so the eval
                # forward populates `_last_attention_state` for the plot
                # below. Reset to False after the plot is written.
                self._set_capture_attention_state(True)
                val_metrics = self.evaluate()
                # Attention summary AND heatmap save MUST happen before
                # _generate_sample(): model.generate() runs the model with
                # (B=1, N=1) for each new token, overwriting every block's
                # _last_attention / _last_diagnostics with single-token state.
                # Reading them after generate yields stale tensors, not the
                # val-batch β the heatmap is meant to show.
                attn = self._attention_summary()
                self._plot_attention_patterns(step + 1)
                # Cross-head coupling diagnostics + figures. No-op when
                # cfg.cross_couplings is empty; otherwise saves a JSON of
                # per-layer metrics + a small fan-out of figures under
                # output_dir/cross_coupling/.
                cc_metrics = self._save_cross_coupling_diagnostics(step + 1)
                sample_text = self._generate_sample()

                _write("")  # blank line above the block
                _write(f"  Validation @ step {step+1}:")
                _write(f"    Loss: {val_metrics['val_loss']:.4f}")
                _write(f"    CE: {val_metrics['val_loss']:.4f}")
                _write(f"    PPL: {val_metrics['val_ppl']:.2f}")
                _write(f"    BPC: {val_metrics['val_bpc']:.3f}")
                if attn:
                    _write(
                        f"    Attn entropy: {attn['attn_entropy']:.3f} | "
                        f"concentration: {attn['attn_concentration']:.3f}"
                    )
                if sample_text:
                    _write(f"    Sample: {sample_text}...")
                _write("")  # blank line below the block

                # Record validation in publication tracker
                if self._pub_tracker is not None:
                    self._pub_tracker.record_validation(step + 1, {
                        'loss': val_metrics['val_loss'],
                        'ce_loss': val_metrics['val_loss'],
                        'perplexity': val_metrics['val_ppl'],
                    })

                # Log validation to CSV (map to expected keys)
                if self._metrics_tracker is not None:
                    _val_log: Dict[str, float] = {
                        'loss': val_metrics['val_loss'],
                        'ce_loss': val_metrics['val_loss'],
                        'perplexity': val_metrics['val_ppl'],
                    }
                    if cc_metrics:
                        _val_log.update({k: float(v) for k, v in cc_metrics.items()
                                          if isinstance(v, (int, float))})
                    self._metrics_tracker.log_val(step + 1, _val_log)

                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    if self.output_dir:
                        best_path = self.output_dir / 'best_model.pt'
                        torch.save(self.model.state_dict(), best_path)

            # Periodic checkpoints
            if self.output_dir and (step + 1) % checkpoint_interval == 0:
                self._save_checkpoint(step + 1)

        if pbar is not None:
            pbar.close()

        total_time = time.time() - t0
        logger.info(f"Training complete: {num_steps} steps in {total_time:.1f}s")

        # Flush metrics CSV
        if self._metrics_tracker is not None:
            self._metrics_tracker.save()
            logger.info(f"Metrics CSV saved: {self.output_dir / 'metrics.csv'}")

        # Generate publication figures
        self._generate_figures()

        # Save final checkpoint
        if self.output_dir:
            self._save_checkpoint(num_steps)

        # Final test-set evaluation. Run once after training completes and
        # only if a test_loader was supplied (the click-to-run entry point
        # at train_vfe.py opts in via include_test=True). Mirrors the
        # publication path at experiment_runner.py:2234-2241.
        test_results: Dict[str, float] = {}
        if self.test_loader is not None:
            test_results = self.run_test_evaluation()

        # Rename run directory to encode the measured test PPL + structural
        # config (per-user 2026-05-18). Old: vfe_runs/<dataset>_<timestamp>/;
        # new: vfe_runs/<test_ppl>=test-PPL_K=<K>_<gauge_label>/. Cosmetic —
        # swallow any I/O error so it can't kill an otherwise good run.
        if self.output_dir is not None and 'test_ppl' in test_results:
            try:
                test_ppl = float(test_results['test_ppl'])
                K = int(self.cfg.embed_dim)
                group_label = _gauge_group_label(self.cfg)
                new_name = f"{test_ppl:.2f}=test-PPL_K={K}_{group_label}"
                new_path = self.output_dir.parent / new_name
                if new_path.exists() and new_path.resolve() != self.output_dir.resolve():
                    from datetime import datetime as _dt
                    new_path = self.output_dir.parent / f"{new_name}_{_dt.now().strftime('%H%M%S')}"
                if new_path.resolve() != self.output_dir.resolve():
                    self.output_dir.rename(new_path)
                    self.output_dir = new_path
                    logger.info(f"Run directory renamed to: {self.output_dir}")
            except (OSError, FileExistsError, PermissionError) as exc:
                logger.warning(f"Run directory rename skipped: {exc}")

        # Save publication tracker history
        if self._pub_tracker and self.output_dir:
            summary = self._pub_tracker.get_summary()
            if summary:
                logger.info(f"Final train PPL: {summary.get('final_train_ppl', 'N/A'):.1f}")
                logger.info(f"Best val PPL: {summary.get('best_val_ppl', 'N/A')}")
