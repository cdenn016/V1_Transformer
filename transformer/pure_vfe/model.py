"""
PureVFETransformer: the model class.

No nn.Module. No autograd. No backprop.
The "model" is a prior bank: one Gaussian N(μ_v, Σ_v) per vocabulary token,
plus gauge frames Ω_v ∈ GL(K_h) per head and positional gauge offsets.

"Forward pass" = E-step VFE descent.
"Learning" = M-step natural gradient on priors.
"""

import math

import torch

from .config import PureVFEConfig
from .inference import e_step
from .learning import m_step, MStepAccumulator, apply_m_step_from_accumulated
from .gauge import init_omega, init_phi, make_gl_generators, phi_to_omega


class PureVFETransformer:
    """
    Pure variational free energy transformer.
    No nn.Module. No autograd. No backprop.

    Inference and learning via natural gradient descent
    on the gauge-covariant variational free energy.
    """

    def __init__(self, config: PureVFEConfig):
        self.config = config
        K = config.belief_dim
        H = config.n_heads
        K_h = config.head_dim
        V = config.vocab_size
        N_max = config.max_seq_len
        dev = config.device

        # -----------------------------------------------------------
        # Prior bank (THE model — raw tensors, not nn.Parameters)
        # -----------------------------------------------------------

        # Prior means: spread must be O(√(ln V / K)) so KL differences
        # between priors are comparable to ln(V), breaking the uniform
        # softmax fixed point.  0.02 is far too small (see issue analysis).
        self.prior_mu = torch.randn(V, K, device=dev) * 0.5

        # Prior covariances: σ²I (SPD, stored directly)
        self.prior_Sigma = (
            config.sigma_init * torch.eye(K, device=dev)
            .unsqueeze(0).expand(V, -1, -1).clone()
        )

        # Gauge frames: two parameterizations togglable via config.gauge_param
        if config.gauge_param == 'phi':
            # Lie algebra path: φ ∈ gl(K_h), Ω = exp(φ)
            n_gen_h = K_h * K_h
            self.prior_phi = init_phi(
                (V, H, n_gen_h), scale=config.omega_init_scale, device=dev
            )
            self.pos_phi = init_phi(
                (N_max, H, n_gen_h), scale=config.omega_init_scale, device=dev
            )
            self.gl_generators = make_gl_generators(K_h, device=dev)  # [K_h², K_h, K_h]

            # Compute Omega from phi for use in E-step
            self.prior_Omega = phi_to_omega(self.prior_phi, self.gl_generators)
            self.pos_Omega = phi_to_omega(self.pos_phi, self.gl_generators)
        else:
            # Direct GL(K_h) storage (default 'omega' path)
            neg_frac = getattr(config, 'omega_negative_det_fraction', 0.0)
            self.prior_Omega = init_omega(
                (V, H, K_h, K_h), scale=config.omega_init_scale, device=dev,
                negative_det_fraction=neg_frac,
            )
            self.pos_Omega = init_omega(
                (N_max, H, K_h, K_h), scale=config.omega_init_scale, device=dev,
                negative_det_fraction=neg_frac,
            )
            self.prior_phi = None
            self.pos_phi = None
            self.gl_generators = None

        # -----------------------------------------------------------
        # Adam momentum buffers for M-step (Feature 2)
        # -----------------------------------------------------------
        self.adam_step = 0
        if getattr(config, 'use_adam_m_step', False):
            # First and second moments for prior_mu natural gradient
            self.m1_mu = torch.zeros(V, K, device=dev)
            self.m2_mu = torch.zeros(V, K, device=dev)
            # First moment only for Sigma and Omega (adaptive scaling
            # interacts poorly with manifold geometry)
            self.m1_Sigma = torch.zeros(V, K, K, device=dev)
            self.m1_Omega = torch.zeros(V, H, K_h, K_h, device=dev)
            self.m1_pos_Omega = torch.zeros(N_max, H, K_h, K_h, device=dev)
        else:
            self.m1_mu = None
            self.m2_mu = None
            self.m1_Sigma = None
            self.m1_Omega = None
            self.m1_pos_Omega = None

        # -----------------------------------------------------------
        # Training step counter (for LR scheduling)
        # -----------------------------------------------------------
        self.global_step = 0

    def _lr_scale(self) -> float:
        r"""Compute LR multiplier from warmup + cosine decay schedule.

        Returns a scalar in (0, 1] that is applied uniformly to all 5
        per-variable learning rates.
        """
        cfg = self.config
        warmup = getattr(cfg, 'warmup_steps', 0)
        schedule = getattr(cfg, 'lr_schedule', 'constant')
        step = self.global_step

        if schedule == 'constant' and warmup <= 0:
            return 1.0

        # Warmup phase
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup

        # Cosine decay phase
        if schedule == 'cosine':
            max_steps = getattr(cfg, 'max_steps', 30000)
            decay_start = max(warmup, 0)
            decay_steps = max(max_steps - decay_start, 1)
            progress = (step - decay_start) / decay_steps
            progress = min(progress, 1.0)
            min_ratio = getattr(cfg, 'min_lr_ratio', 0.1)
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        return 1.0

    def get_effective_lrs(self) -> dict:
        r"""Compute scheduled learning rates for all 5 variable groups.

        Applies warmup + cosine decay uniformly to ``mu_q_lr``,
        ``sigma_q_lr``, ``phi_lr``, ``mu_p_lr``, ``sigma_p_lr``.
        """
        cfg = self.config
        scale = self._lr_scale()
        return {
            'mu_q_lr': cfg.mu_q_lr * scale,
            'sigma_q_lr': cfg.sigma_q_lr * scale,
            'phi_lr': cfg.phi_lr * scale,
            'mu_p_lr': cfg.mu_p_lr * scale,
            'sigma_p_lr': cfg.sigma_p_lr * scale,
        }

    def forward(self, token_ids):
        """
        Returns logits. Inference IS VFE descent.

        Args:
            token_ids: [B, N] long tensor

        Returns:
            logits: [B, N, V]
        """
        mu, Sigma, Omega, logits, vfe, _diag = e_step(token_ids, self, self.config)
        return logits

    def forward_with_attention(self, token_ids):
        """
        Forward pass returning both logits and attention weights for visualization.

        Args:
            token_ids: [B, N] long tensor

        Returns:
            logits: [B, N, V]
            beta: [B, H, N, N] attention weights
            diagnostics: dict
        """
        mu, Sigma, Omega, logits, vfe, diagnostics = e_step(
            token_ids, self, self.config
        )
        beta = diagnostics.get('final_beta', None)
        return logits, beta, diagnostics

    def update(self, token_ids, targets):
        """
        Full forward + backward. Returns logits and loss.

        Args:
            token_ids: [B, N] long tensor
            targets: [B, N] long tensor

        Returns:
            logits: [B, N, V]
            ce_loss: scalar float
            vfe: list of VFE values per E-step
            diagnostics: dict with E-step gradient norms and NaN events
        """
        effective_lrs = self.get_effective_lrs()
        mu, Sigma, Omega, logits, vfe, diagnostics = e_step(
            token_ids, self, self.config, effective_lrs=effective_lrs
        )
        ce_loss = m_step(token_ids, targets, mu, Sigma, Omega, self, self.config,
                         logits=logits, effective_lrs=effective_lrs)
        self.global_step += 1
        return logits, ce_loss, vfe, diagnostics

    def create_accumulator(self):
        """Create an M-step accumulator for gradient accumulation.

        Usage::

            accum = model.create_accumulator()
            for micro_batch in micro_batches:
                mu, Sigma, Omega, logits, vfe, diag = e_step(ids, model, config)
                accum.accumulate(ids, targets, mu, Sigma, Omega, model, config, logits)
            ce_loss = apply_m_step_from_accumulated(accum, model, config)
            model.global_step += 1
            accum.reset()
        """
        return MStepAccumulator(self.config, self.config.device)

    def sync_omega_from_phi(self):
        """Recompute Omega from phi (call after phi updates in M-step)."""
        if self.prior_phi is not None:
            self.prior_Omega = phi_to_omega(self.prior_phi, self.gl_generators)
            self.pos_Omega = phi_to_omega(self.pos_phi, self.gl_generators)

    def save(self, path):
        """Save model state to disk."""
        state = {
            'prior_mu': self.prior_mu.cpu(),
            'prior_Sigma': self.prior_Sigma.cpu(),
            'prior_Omega': self.prior_Omega.cpu(),
            'pos_Omega': self.pos_Omega.cpu(),
            'config': self.config,
            'global_step': self.global_step,
            'adam_step': self.adam_step,
        }
        if self.prior_phi is not None:
            state['prior_phi'] = self.prior_phi.cpu()
            state['pos_phi'] = self.pos_phi.cpu()
        # Save momentum buffers if they exist
        if self.m1_mu is not None:
            state['m1_mu'] = self.m1_mu.cpu()
            state['m2_mu'] = self.m2_mu.cpu()
            state['m1_Sigma'] = self.m1_Sigma.cpu()
            state['m1_Omega'] = self.m1_Omega.cpu()
            state['m1_pos_Omega'] = self.m1_pos_Omega.cpu()
        torch.save(state, path)

    @classmethod
    def load(cls, path, device=None):
        """Load model from disk."""
        data = torch.load(path, weights_only=False)
        config = data['config']
        if device is not None:
            config.device = device
        model = cls(config)
        dev = config.device
        model.prior_mu = data['prior_mu'].to(dev)
        model.prior_Sigma = data['prior_Sigma'].to(dev)
        model.prior_Omega = data['prior_Omega'].to(dev)
        model.pos_Omega = data['pos_Omega'].to(dev)
        if 'prior_phi' in data and model.prior_phi is not None:
            model.prior_phi = data['prior_phi'].to(dev)
            model.pos_phi = data['pos_phi'].to(dev)
        model.global_step = data.get('global_step', 0)
        model.adam_step = data.get('adam_step', 0)
        # Restore momentum buffers
        if 'm1_mu' in data and model.m1_mu is not None:
            model.m1_mu = data['m1_mu'].to(dev)
            model.m2_mu = data['m2_mu'].to(dev)
            model.m1_Sigma = data['m1_Sigma'].to(dev)
            model.m1_Omega = data['m1_Omega'].to(dev)
            model.m1_pos_Omega = data['m1_pos_Omega'].to(dev)
        return model

    def to(self, device):
        """Move all tensors to device."""
        self.config.device = str(device)
        self.prior_mu = self.prior_mu.to(device)
        self.prior_Sigma = self.prior_Sigma.to(device)
        self.prior_Omega = self.prior_Omega.to(device)
        self.pos_Omega = self.pos_Omega.to(device)
        if self.prior_phi is not None:
            self.prior_phi = self.prior_phi.to(device)
            self.pos_phi = self.pos_phi.to(device)
            self.gl_generators = self.gl_generators.to(device)
        if self.m1_mu is not None:
            self.m1_mu = self.m1_mu.to(device)
            self.m2_mu = self.m2_mu.to(device)
            self.m1_Sigma = self.m1_Sigma.to(device)
            self.m1_Omega = self.m1_Omega.to(device)
            self.m1_pos_Omega = self.m1_pos_Omega.to(device)
        return self

    def param_count(self):
        """Total number of learnable scalar parameters."""
        K = self.config.belief_dim
        V = self.config.vocab_size
        H = self.config.n_heads
        K_h = self.config.head_dim
        N = self.config.max_seq_len

        mu_params = V * K
        sigma_params = V * K * K

        if self.config.gauge_param == 'phi':
            n_gen_h = K_h * K_h
            gauge_params = V * H * n_gen_h
            pos_params = N * H * n_gen_h
            gauge_key = 'prior_phi'
            pos_key = 'pos_phi'
        else:
            gauge_params = V * H * K_h * K_h
            pos_params = N * H * K_h * K_h
            gauge_key = 'prior_Omega'
            pos_key = 'pos_Omega'

        total = mu_params + sigma_params + gauge_params + pos_params
        return {
            'prior_mu': mu_params,
            'prior_Sigma': sigma_params,
            gauge_key: gauge_params,
            pos_key: pos_params,
            'total': total,
        }
