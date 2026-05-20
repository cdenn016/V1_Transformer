"""
AIFAugmentedTrainer: VFETrainer + EFE-augmented training objective.

Drop-in replacement for :class:`transformer.vfe.trainer.VFETrainer` that
adds the canonical training-time EFE augmentation
:math:`L_{\\text{total}} = L_{\\text{CE}} + \\lambda_{\\text{AIF}}\\,
L_{\\text{AIF}}` per Phase 4 of the build-out (see
``docs/plans/2026-05-19-aif-transformer-buildout/06_plan.md`` §6).

Construction takes an :class:`AIFConfig` alongside the standard
:class:`VFEConfig`; the rest of the training infrastructure (optimizer,
schedulers, gradient clipping, checkpointing, CSV metrics, attention
plots, eval loop) is inherited unchanged from ``VFETrainer``. The only
override is :meth:`train_step`, which replaces the model's combined
loss with a manually-composed CE plus the AIF augmentation while
preserving every regularizer ``VFEModel.forward`` adds (mass-phi,
auxiliary hyperparameter loss, ``normalize_ce_by_dim`` scaling).

Law 1 preserved by construction: the E-step inside
``model.forward_with_beliefs`` has no ``targets`` parameter; the
training step computes CE from the resulting logits after the E-step
returns. No target tokens reach the inference path.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer.aif.preferences import Preference, build_preference
from transformer.aif.training_loss import compute_training_efe_loss
from transformer.vfe.trainer import VFETrainer

if TYPE_CHECKING:
    from transformer.aif.config import AIFConfig
    from transformer.vfe.config import VFEConfig
    from transformer.vfe.model import VFEModel


class AIFAugmentedTrainer(VFETrainer):
    r"""VFETrainer with the Phase-4 trajectory-as-policy AIF augmentation.

    The training loss is

    .. math::
        L_{\text{total}} = L_{\text{CE}} +
            \lambda_{\text{AIF}}\, L_{\text{AIF}}

    where :math:`L_{\text{CE}}` matches the standard
    :class:`VFETrainer` loss exactly (including ``normalize_ce_by_dim``
    scaling, ``mass_phi`` regularizer, and per-block auxiliary
    hyperparameter loss) and :math:`L_{\text{AIF}}` is the per-batch
    EFE mean from
    :func:`transformer.aif.training_loss.compute_training_efe_loss`.

    Args:
        model: trained-from-scratch ``VFEModel``.
        vfe_cfg: ``VFEConfig`` (same role as in ``VFETrainer``).
        aif_cfg: ``AIFConfig`` with
            ``training_objective='efe_augmented'``.
        train_loader, val_loader, test_loader, device, output_dir:
            forwarded to ``VFETrainer``.
        preference: optional pre-built ``Preference`` instance. If
            ``None``, one is built from
            ``aif_cfg.preference_type`` / ``aif_cfg.preference_path``.

    Raises:
        ValueError: if ``aif_cfg.training_objective != 'efe_augmented'``.
    """

    def __init__(
        self,
        model: 'VFEModel',
        vfe_cfg: 'VFEConfig',
        aif_cfg: 'AIFConfig',
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        device: Union[str, torch.device] = 'cuda',
        output_dir: str = 'aif_runs',
        preference: Optional[Preference] = None,
    ) -> None:
        if aif_cfg.training_objective != 'efe_augmented':
            raise ValueError(
                "AIFAugmentedTrainer requires aif_cfg.training_objective="
                "'efe_augmented'. Use VFETrainer directly for standard_vfe "
                "training."
            )
        super().__init__(
            model=model,
            cfg=vfe_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
        )
        self.aif_cfg = aif_cfg
        if preference is None:
            preference = build_preference(
                preference_type=aif_cfg.preference_type,
                preference_path=aif_cfg.preference_path,
                low_entropy_beta=aif_cfg.low_entropy_beta,
            )
        self.preference = preference
        # Hoist the loss-scale factor used by the underlying VFEModel so the
        # AIF augmentation is scaled consistently with CE under
        # ``normalize_ce_by_dim=True``.
        self._loss_scale = (
            1.0 / (self.cfg.embed_dim ** 0.5)
            if self.cfg.normalize_ce_by_dim else 1.0
        )

    def train_step(
        self,
        batch: "Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]",
    ) -> Dict[str, float]:
        r"""Override of :meth:`VFETrainer.train_step` that folds in
        :math:`\lambda_{\text{AIF}}\, L_{\text{AIF}}`.

        Reproduces the exact loss-composition logic from
        ``VFEModel.forward`` (so the standard VFE training behaviour is
        preserved bitwise when ``aif_loss_weight=0``), then adds the
        AIF augmentation. The returned metrics dict includes an
        ``aif_loss`` field for CSV / dashboard logging.
        """
        self.model.train()
        t0 = time.time()
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(self.device)
            target_ids = batch[1].to(self.device)
        else:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

        # E-step: model.forward_with_beliefs runs the encode + positional +
        # stack + final-norm + decode pipeline and returns both logits AND
        # the converged BeliefState. No targets passed (Law 1 preserved).
        logits, beliefs = self.model.forward_with_beliefs(input_ids)

        # === CE branch — mirrors VFEModel.forward at lines ~209-258 ===
        ce_loss = F.cross_entropy(
            logits.view(-1, self.cfg.vocab_size),
            target_ids.view(-1),
            ignore_index=-100,
        )
        ce_for_log = ce_loss.detach()
        ce_loss = ce_loss * self._loss_scale
        loss = ce_loss

        # Mass-phi gauge prior — mirrors VFEModel.forward.
        if self.cfg.mass_phi > 0:
            phi_norm_sq = (beliefs.phi ** 2).sum() / (
                beliefs.phi.shape[0] * beliefs.phi.shape[1]
            )
            loss = loss + self._loss_scale * (0.5 * self.cfg.mass_phi * phi_norm_sq)

        # Auxiliary hyperparameter loss — mirrors VFEModel.forward.
        for block in self.model.stack.blocks:
            aux = getattr(block.e_step, '_aux_hyperparam_loss', None)
            if aux is not None:
                loss = loss + self._loss_scale * aux

        # === AIF augmentation ===
        aif_loss = compute_training_efe_loss(
            logits=logits,
            beliefs=beliefs,
            preference=self.preference,
            prior_bank=self.model.prior_bank,
            cfg=self.aif_cfg,
        )
        loss = loss + self._loss_scale * self.aif_cfg.aif_loss_weight * aif_loss
        aif_for_log = aif_loss.detach()
        # =========================

        # Backward.
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Grad-norm collection — identical to VFETrainer.train_step. Gated
        # on log-step flag so the per-group cat+norm sync cost is paid only
        # at log boundaries.
        if self._aggregate_diagnostics_this_step:
            grad_norms = self._compute_gradient_norms()
        else:
            grad_norms = {
                'mu': 0.0, 'sigma': 0.0, 'phi': 0.0,
                'ffn': 0.0, 'other': 0.0, 'total': 0.0,
            }

        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip,
            )

        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        step_time = time.time() - t0
        loss_val = loss.item()
        ce_val = ce_for_log.item()
        aif_val = aif_for_log.item()
        ppl = math.exp(min(ce_val, 20.0))
        bpc = ce_val / math.log(2)
        lr = self.scheduler.get_last_lr()[0]

        B, N = input_ids.shape
        metrics = {
            'loss': loss_val,
            'ce': ce_val,
            'aif_loss': aif_val,
            'ppl': ppl,
            'bpc': bpc,
            'lr': lr,
            'step_time': step_time,
            'tokens_per_sec': (B * N) / step_time if step_time > 0 else 0,
            'grad_norm_total': grad_norms['total'],
            'grad_norm_mu': grad_norms['mu'],
            'grad_norm_sigma': grad_norms['sigma'],
            'grad_norm_phi': grad_norms['phi'],
            'grad_norm_ffn': grad_norms['ffn'],
            'grad_norm_other': grad_norms['other'],
        }

        if self._aggregate_diagnostics_this_step:
            e_diag = self._collect_e_step_diagnostics()
            metrics.update({f'{k}': v for k, v in e_diag.items()})
            metrics.update(self._collect_bayesian_alpha_stats())
            metrics.update(self._collect_kappa_stats())

        return metrics
