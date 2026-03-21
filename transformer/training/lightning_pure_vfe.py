"""
PyTorch Lightning Module for Pure VFE Transformer
===================================================

Wraps PureVFETransformer as a LightningModule with manual optimization,
since PureFEP uses analytic natural gradient descent (no autograd, no optimizer).

Usage:
    from transformer.training.lightning_pure_vfe import PureVFELitModule
    from transformer.pure_vfe.config import PureVFEConfig

    config = PureVFEConfig(vocab_size=50257, belief_dim=32, ...)
    lit_model = PureVFELitModule(config)
"""

import math

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from transformer.pure_vfe.config import PureVFEConfig
from transformer.pure_vfe.model import PureVFETransformer


class PureVFELitModule(pl.LightningModule):
    """
    Lightning wrapper for PureVFETransformer.

    Uses manual optimization (automatic_optimization=False) because PureFEP
    manages its own parameter updates via analytic natural gradient descent
    in the M-step. No PyTorch optimizer is needed.

    Single-GPU only: PureFEP's direct tensor mutation is incompatible with DDP.
    """

    def __init__(self, pure_vfe_config: PureVFEConfig):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.pure_vfe_config = pure_vfe_config
        self.model = None  # Created in setup() to respect device placement

    def setup(self, stage=None):
        if self.model is not None:
            return
        self.pure_vfe_config.device = str(self.device)
        self.model = PureVFETransformer(self.pure_vfe_config)
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, token_ids: torch.Tensor):
        return self.model.forward(token_ids)

    # ------------------------------------------------------------------
    # Training step (manual optimization)
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        logits, ce_loss, vfe = self.model.update(input_ids, target_ids)

        ppl = math.exp(min(ce_loss, 20.0))

        # vfe may be a list (per E-step iteration) or scalar; log final value
        vfe_scalar = vfe[-1] if isinstance(vfe, (list, tuple)) else vfe
        vfe_first = vfe[0] if isinstance(vfe, (list, tuple)) and len(vfe) > 0 else vfe_scalar

        self.log('train/loss', ce_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train/ce_loss', ce_loss, on_step=True, on_epoch=False)
        self.log('train/vfe', vfe_scalar, on_step=True, on_epoch=False)
        self.log('train/perplexity', ppl, prog_bar=True, on_step=True, on_epoch=False)

        # VFE convergence diagnostics
        if isinstance(vfe, (list, tuple)) and len(vfe) > 1:
            vfe_ratio = vfe[-1] / max(abs(vfe[0]), 1e-8) if vfe[0] != 0 else 0.0
            self.log('train/vfe_first', vfe_first, on_step=True, on_epoch=False)
            self.log('train/vfe_ratio', vfe_ratio, on_step=True, on_epoch=False)
            self.log('train/vfe_steps', float(len(vfe)), on_step=True, on_epoch=False)

        # Prior health diagnostics (every 50 steps to avoid overhead)
        if batch_idx % 50 == 0:
            with torch.no_grad():
                from transformer.pure_vfe.gauge import monitor_omega_health
                sig_eigs = torch.linalg.eigvalsh(self.model.prior_Sigma[:100])
                self.log('diag/sigma_min', sig_eigs[..., 0].min().item(), on_step=True, on_epoch=False)
                self.log('diag/sigma_max', sig_eigs[..., -1].max().item(), on_step=True, on_epoch=False)
                mu_norms = self.model.prior_mu.norm(dim=-1)
                self.log('diag/mu_norm_mean', mu_norms.mean().item(), on_step=True, on_epoch=False)
                self.log('diag/mu_norm_max', mu_norms.max().item(), on_step=True, on_epoch=False)
                health = monitor_omega_health(self.model.prior_Omega[:100], "Omega")
                self.log('diag/omega_cond_max', health['Omega/cond_max'], on_step=True, on_epoch=False)
                self.log('diag/omega_cond_mean', health['Omega/cond_mean'], on_step=True, on_epoch=False)

        # Manual optimization: step the dummy optimizer to increment global_step,
        # which Lightning uses for max_steps tracking and checkpoint scheduling.
        opt = self.optimizers()
        opt.step()

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        logits = self.model.forward(input_ids)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=-100,
        )
        ce_loss = loss.item()
        ppl = math.exp(min(ce_loss, 20.0))

        self.log('val/ce_loss', ce_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/perplexity', ppl, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimizer (none — PureFEP uses analytic natural gradients)
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        # Dummy optimizer required by Lightning even with manual optimization.
        # PureFEP uses analytic natural gradients — this optimizer is never stepped.
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.SGD([dummy_param], lr=0.0)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint):
        state = {
            'prior_mu': self.model.prior_mu.cpu(),
            'prior_Sigma': self.model.prior_Sigma.cpu(),
            'prior_Omega': self.model.prior_Omega.cpu(),
            'pos_Omega': self.model.pos_Omega.cpu(),
        }
        if self.model.prior_phi is not None:
            state['prior_phi'] = self.model.prior_phi.cpu()
            state['pos_phi'] = self.model.pos_phi.cpu()
        checkpoint['pure_vfe_state'] = state
        checkpoint['pure_vfe_config'] = self.pure_vfe_config

    def on_load_checkpoint(self, checkpoint):
        if 'pure_vfe_state' not in checkpoint:
            return
        # Ensure model exists before loading state
        if self.model is None:
            config = checkpoint.get('pure_vfe_config', self.pure_vfe_config)
            config.device = str(self.device)
            self.model = PureVFETransformer(config)
        state = checkpoint['pure_vfe_state']
        dev = self.device
        self.model.prior_mu = state['prior_mu'].to(dev)
        self.model.prior_Sigma = state['prior_Sigma'].to(dev)
        self.model.prior_Omega = state['prior_Omega'].to(dev)
        self.model.pos_Omega = state['pos_Omega'].to(dev)
        if 'prior_phi' in state and self.model.prior_phi is not None:
            self.model.prior_phi = state['prior_phi'].to(dev)
            self.model.pos_phi = state['pos_phi'].to(dev)
