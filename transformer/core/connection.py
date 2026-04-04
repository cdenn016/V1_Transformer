"""
Gauge Connection: Edge-local Lie algebra elements for non-flat transport.
=========================================================================

Parameterizes the "connection" δ_ij on the token graph so that the transport
operator becomes:

    Ω_ij = exp(φ_i · G) · exp(δ_ij · G) · exp(-φ_j · G)

When δ_ij = 0, this is identically the flat transport Ω_ij = exp(φ_i)·exp(-φ_j).

The connection controls holonomy:
    H_ijk = exp(φ_i) · C_ijk · exp(-φ_i)
where C_ijk = exp(δ_ij·G) · exp(δ_jk·G) · exp(δ_ki·G) is the connection holonomy.

Two parameterizations:
    - Bilinear:  δ_ij^a = μ_i^T W^a μ_j  (parameter-efficient, default)
    - MLP:       δ_ij = MLP([μ_i; μ_j])   (more expressive, higher memory)

Both are zero-initialized so the model starts in the flat regime.
"""

import math
import torch
import torch.nn as nn


class GaugeConnection(nn.Module):
    """Edge-local connection producing Lie algebra elements for non-flat transport.

    For each edge (i, j), produces δ_ij ∈ ℝ^{n_gen} such that:
        Ω_ij = exp(φ_i · G) · exp(δ_ij · G) · exp(-φ_j · G)

    The connection is zero-initialized: δ_ij = 0 at init -> flat transport.
    The model learns to deviate from flatness only where the data warrants it.

    Args:
        d_head: Per-head dimension (belief mean dimension for this head).
        n_gen: Number of Lie algebra generators for this head's gauge group.
        connection_type: 'bilinear' or 'mlp'.
        hidden_dim: Hidden dimension for MLP connection (ignored for bilinear).
        antisymmetrize: If True, enforce δ_ij = -δ_ji (natural for connections
                        on undirected graphs). Only applies to bilinear.
    """

    def __init__(
        self,
        d_head: int,
        n_gen: int,
        connection_type: str = 'bilinear',
        hidden_dim: int = 64,
        antisymmetrize: bool = False,
        init_scale: float = 0.0,
    ):
        super().__init__()
        self.d_head = d_head
        self.n_gen = n_gen
        self.connection_type = connection_type
        self.antisymmetrize = antisymmetrize

        if connection_type == 'bilinear':
            # δ_ij^a = μ_i^T W^a μ_j -- one bilinear form per generator
            # Parameters: n_gen x d_head x d_head
            #
            # init_scale=0 -> zero init (flat saddle point -- no gradient signal!)
            # init_scale>0 -> small random init breaks the flat saddle point so
            # the optimizer can discover useful curvature.  Recommended: 0.01.
            W = torch.zeros(n_gen, d_head, d_head)
            if init_scale > 0:
                W = W + init_scale * torch.randn_like(W) / math.sqrt(d_head)
            self.W = nn.Parameter(W)
        elif connection_type == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(2 * d_head, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_gen),
            )
            # Zero-init output layer -> flat at initialization
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            raise ValueError(f"Unknown connection_type: {connection_type}. Use 'bilinear' or 'mlp'.")

    def forward(self, mu_i: torch.Tensor, mu_j: torch.Tensor) -> torch.Tensor:
        """Compute edge-local connection δ_ij.

        Args:
            mu_i: (B, N, d_head) query belief means.
            mu_j: (B, N, d_head) key belief means.

        Returns:
            delta: (B, N, N, n_gen) Lie algebra coefficients per edge.
        """
        if self.connection_type == 'bilinear':
            W = self.W
            if self.antisymmetrize:
                # W -> (W - W^T) / 2 ensures δ_ij = -δ_ji
                W = (W - W.transpose(-1, -2)) / 2

            # δ_ij^a = μ_i^T W^a μ_j
            # mu_i: (B, N, d) -- broadcast over j
            # W:    (n_gen, d, d)
            # mu_j: (B, N, d) -- broadcast over i
            # Result: (B, N, N, n_gen)
            delta = torch.einsum('bid,adg,bjg->bija', mu_i, W, mu_j)
        elif self.connection_type == 'mlp':
            B, N, D = mu_i.shape
            # Expand to all pairs: (B, N, N, 2D)
            mu_i_exp = mu_i.unsqueeze(2).expand(-1, -1, N, -1)
            mu_j_exp = mu_j.unsqueeze(1).expand(-1, N, -1, -1)
            pair = torch.cat([mu_i_exp, mu_j_exp], dim=-1)
            delta = self.net(pair)  # (B, N, N, n_gen)

        return delta

    def extra_repr(self) -> str:
        return (
            f"d_head={self.d_head}, n_gen={self.n_gen}, "
            f"type={self.connection_type}, antisym={self.antisymmetrize}"
        )
