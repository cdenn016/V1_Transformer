"""
PureVFETransformer: the model class.

No nn.Module. No autograd. No backprop.
The "model" is a prior bank: one Gaussian N(μ_v, Σ_v) per vocabulary token,
plus gauge frames Ω_v ∈ GL(K_h) per head and positional gauge offsets.

"Forward pass" = E-step VFE descent.
"Learning" = M-step natural gradient on priors.
"""

import torch

from .config import PureVFEConfig
from .inference import e_step
from .learning import m_step
from .gauge import init_omega


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

        # Gauge frames: Ω_v ∈ GL⁺(K_h) per head, near identity
        self.prior_Omega = init_omega(
            (V, H, K_h, K_h), scale=config.omega_init_scale, device=dev
        )

        # Positional gauge: Ω_pos per position per head
        self.pos_Omega = init_omega(
            (N_max, H, K_h, K_h), scale=config.omega_init_scale, device=dev
        )

    def forward(self, token_ids):
        """
        Returns logits. Inference IS VFE descent.

        Args:
            token_ids: [B, N] long tensor

        Returns:
            logits: [B, N, V]
        """
        mu, Sigma, Omega, logits, vfe = e_step(token_ids, self, self.config)
        return logits

    def update(self, token_ids, targets):
        """
        Full forward + backward. Returns logits and loss.

        Args:
            token_ids: [B, N] long tensor
            targets: [B, N] long tensor

        Returns:
            logits: [B, N, V]
            ce_loss: scalar float
        """
        mu, Sigma, Omega, logits, vfe = e_step(token_ids, self, self.config)
        ce_loss = m_step(token_ids, targets, mu, Sigma, Omega, self, self.config,
                         logits=logits)
        return logits, ce_loss, vfe

    def save(self, path):
        """Save model state to disk."""
        torch.save({
            'prior_mu': self.prior_mu.cpu(),
            'prior_Sigma': self.prior_Sigma.cpu(),
            'prior_Omega': self.prior_Omega.cpu(),
            'pos_Omega': self.pos_Omega.cpu(),
            'config': self.config,
        }, path)

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
        return model

    def to(self, device):
        """Move all tensors to device."""
        self.config.device = str(device)
        self.prior_mu = self.prior_mu.to(device)
        self.prior_Sigma = self.prior_Sigma.to(device)
        self.prior_Omega = self.prior_Omega.to(device)
        self.pos_Omega = self.pos_Omega.to(device)
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
        omega_params = V * H * K_h * K_h
        pos_params = N * H * K_h * K_h

        total = mu_params + sigma_params + omega_params + pos_params
        return {
            'prior_mu': mu_params,
            'prior_Sigma': sigma_params,
            'prior_Omega': omega_params,
            'pos_Omega': pos_params,
            'total': total,
        }
