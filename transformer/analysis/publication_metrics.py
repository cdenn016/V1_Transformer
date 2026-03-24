"""
Publication Metrics for Gauge-Theoretic VFE Transformer
========================================================

Generates publication-quality figures and tracks training dynamics for the
gauge-theoretic VFE transformer. Supports all gauge groups (SO(3), SO(N), GL(K))
and both diagonal and full covariance modes.

Key Figures:
1. Training Curves - Loss, PPL, BPC over training steps
2. Attention Heatmaps - KL-divergence based attention patterns
3. Model Comparison - Bar charts comparing architectures
4. Scaling Study - Performance vs embedding dimension K
5. Gauge Frame Clustering - Semantic structure of phi embeddings
6. Attention Entropy - Sharpness of KL-attention over training

Author: Robert C. Dennis
Date: December 2025
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import json
import csv
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    GridSpec = None
    ticker = None
    MATPLOTLIB_AVAILABLE = False

# Import gauge frame semantic analysis
from .semantics import (
    analyze_gauge_semantics,
    plot_gauge_frame_clustering,
    analyze_omega_semantics,
    plot_omega_clustering,
    analyze_sigma_semantics,
    analyze_holonomy_semantic_correlation,
    compute_semantic_field_coherence,
    SemanticTrajectoryTracker,
)

# Import holonomy analysis
from .holonomy import compute_holonomy, holonomy_statistics
from .holonomy_metrics import (
    HolonomySnapshot,
    HolonomyProfile,
    compute_holonomy_snapshot,
    compute_curvature_by_distance,
    compute_flatness_trajectory,
)


# =============================================================================
# Publication Style Settings
# =============================================================================

PUBLICATION_STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}


def apply_publication_style():
    """Apply publication-quality matplotlib styling."""
    plt.rcParams.update(PUBLICATION_STYLE)


def format_step_axis(ax):
    """Format x-axis to show steps as 'k' notation (e.g., 150000 -> 150k)."""
    def step_formatter(x, pos):
        if x >= 1000:
            return f'{x/1000:.0f}k'
        return f'{x:.0f}'
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(step_formatter))


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrainingSnapshot:
    """Single snapshot of training metrics."""
    step: int
    epoch: float
    train_loss: float
    train_ce: float
    train_ppl: float
    train_bpc: float
    val_loss: Optional[float] = None
    val_ce: Optional[float] = None
    val_ppl: Optional[float] = None
    val_bpc: Optional[float] = None
    grad_norm_total: float = 0.0
    grad_norm_mu: float = 0.0
    grad_norm_sigma: float = 0.0
    grad_norm_phi: float = 0.0
    grad_norm_ffn: float = 0.0
    grad_norm_other: float = 0.0
    # E-step natural gradient norms (from VFE iterations inside forward pass)
    e_step_nat_grad_mu: float = 0.0
    e_step_nat_grad_sigma: float = 0.0
    e_step_grad_phi: float = 0.0
    e_step_nat_grad_mu_clipped: float = 0.0
    e_step_nat_grad_sigma_clipped: float = 0.0
    # M-step Fisher-preconditioned gradient norms (from NaturalGradientOptimizer)
    mstep_nat_grad_mu: float = 0.0
    mstep_nat_grad_sigma: float = 0.0
    mstep_nat_grad_phi: float = 0.0
    mstep_nat_grad_other: float = 0.0
    lr_current: float = 0.0
    tokens_per_sec: float = 0.0
    step_time: float = 0.0
    beta_loss: float = 0.0  # Belief alignment term
    attention_entropy: float = 0.0  # Entropy of attention weights (higher = more uniform)
    attention_concentration: float = 0.0  # Concentration of attention (higher = more peaked)
    # Holonomy diagnostics (non-flat transport curvature)
    holonomy_mean_norm: float = 0.0       # Mean ‖C_ijk - I‖_F across sampled triples
    holonomy_max_norm: float = 0.0        # Max ‖C_ijk - I‖_F
    holonomy_frac_gt_01: float = 0.0      # Fraction of triples with ‖C-I‖ > 0.1
    holonomy_spectral_gap: float = 0.0    # Mean eigenvalue spread of C_ijk
    holonomy_wilson_trace: float = 0.0    # Mean |tr(C)/K - 1| (Wilson loop deviation)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results from a single experiment configuration."""
    name: str
    config: Dict[str, Any]
    final_val_ppl: float
    final_val_bpc: float
    best_val_ppl: float
    total_params: int
    training_time: float
    tokens_per_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Training Tracker
# =============================================================================

class TrainingTracker:
    """
    Track training dynamics for publication figures.

    Records loss, PPL, BPC, gradients over training steps.
    """

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("./outputs/figures")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[TrainingSnapshot] = []

    def record(
        self,
        step: int,
        epoch: float,
        train_metrics: Dict[str, float],
        grad_norms: Optional[Dict[str, float]] = None,
        lr: float = 0.0,
        step_time: float = 0.0,
        batch_size: int = 1,
        seq_len: int = 1,
        e_step_norms: Optional[Dict[str, float]] = None,
        mstep_natural_norms: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Record a training step.

        Args:
            mstep_natural_norms: Per-group dict from NaturalGradientOptimizer.get_grad_norms().
                Maps group name → {'euclidean': float, 'natural': float}.
        """
        tokens_per_sec = (batch_size * seq_len) / step_time if step_time > 0 else 0

        train_ce = train_metrics.get('ce_loss', train_metrics.get('ce', 0))
        train_bpc = train_ce / math.log(2) if train_ce > 0 else 0
        train_ppl = math.exp(min(train_ce, 20)) if train_ce > 0 else float('inf')

        # Extract M-step natural gradient norms per group
        def _mstep_nat(group_name: str) -> float:
            if mstep_natural_norms is None:
                return 0.0
            entry = mstep_natural_norms.get(group_name, None)
            if entry is None:
                return 0.0
            return entry.get('natural', 0.0)

        snapshot = TrainingSnapshot(
            step=step,
            epoch=epoch,
            train_loss=train_metrics.get('loss', 0),
            train_ce=train_ce,
            train_ppl=train_ppl,
            train_bpc=train_bpc,
            beta_loss=train_metrics.get('beta_loss', train_metrics.get('beta', 0)),
            grad_norm_total=grad_norms.get('total', 0) if grad_norms else 0,
            grad_norm_mu=grad_norms.get('mu', 0) if grad_norms else 0,
            grad_norm_sigma=grad_norms.get('sigma', 0) if grad_norms else 0,
            grad_norm_phi=grad_norms.get('phi', 0) if grad_norms else 0,
            grad_norm_ffn=grad_norms.get('ffn', 0) if grad_norms else 0,
            grad_norm_other=grad_norms.get('other', 0) if grad_norms else 0,
            e_step_nat_grad_mu=e_step_norms.get('nat_grad_mu', 0) if e_step_norms else 0,
            e_step_nat_grad_sigma=e_step_norms.get('nat_grad_sigma', 0) if e_step_norms else 0,
            e_step_grad_phi=e_step_norms.get('grad_phi', 0) if e_step_norms else 0,
            e_step_nat_grad_mu_clipped=e_step_norms.get('nat_grad_mu_clipped', 0) if e_step_norms else 0,
            e_step_nat_grad_sigma_clipped=e_step_norms.get('nat_grad_sigma_clipped', 0) if e_step_norms else 0,
            mstep_nat_grad_mu=_mstep_nat('mu_embed'),
            mstep_nat_grad_sigma=_mstep_nat('sigma_embed'),
            mstep_nat_grad_phi=_mstep_nat('phi_embed'),
            mstep_nat_grad_other=_mstep_nat('output'),
            lr_current=lr,
            tokens_per_sec=tokens_per_sec,
            step_time=step_time,
            attention_entropy=train_metrics.get('attention_entropy', 0),
            attention_concentration=train_metrics.get('attention_concentration', 0),
            holonomy_mean_norm=train_metrics.get('holonomy/mean_norm', 0),
            holonomy_max_norm=train_metrics.get('holonomy/max_norm', 0),
            holonomy_frac_gt_01=train_metrics.get('holonomy/frac_gt_0.1', 0),
            holonomy_spectral_gap=train_metrics.get('holonomy/spectral_gap', 0),
            holonomy_wilson_trace=train_metrics.get('holonomy/wilson_trace', 0),
        )

        self.history.append(snapshot)

    def record_validation(self, step: int, val_metrics: Dict[str, float]):
        """Record validation metrics for a given step."""
        for snapshot in reversed(self.history):
            if snapshot.step == step:
                val_ce = val_metrics.get('ce_loss', val_metrics.get('loss', 0))
                snapshot.val_loss = val_metrics.get('loss', val_ce)
                snapshot.val_ce = val_ce
                snapshot.val_ppl = val_metrics.get('perplexity',
                    math.exp(min(val_ce, 20)) if val_ce > 0 else float('inf'))
                snapshot.val_bpc = val_ce / math.log(2) if val_ce > 0 else None
                break

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.history:
            return {}

        val_ppls = [s.val_ppl for s in self.history if s.val_ppl is not None]
        val_bpcs = [s.val_bpc for s in self.history if s.val_bpc is not None]

        return {
            'total_steps': self.history[-1].step,  # Use actual step number, not entry count
            'final_train_loss': self.history[-1].train_loss,
            'final_train_ppl': self.history[-1].train_ppl,
            'final_train_bpc': self.history[-1].train_bpc,
            'final_val_ppl': val_ppls[-1] if val_ppls else None,
            'final_val_bpc': val_bpcs[-1] if val_bpcs else None,
            'best_val_ppl': min(val_ppls) if val_ppls else None,
            'best_val_bpc': min(val_bpcs) if val_bpcs else None,
            'avg_tokens_per_sec': np.mean([s.tokens_per_sec for s in self.history]),
            'total_time_sec': sum([s.step_time for s in self.history]),
        }

    def save_json(self, filename: str = "training_history.json"):
        """Save history to JSON."""
        data = {
            'summary': self.get_summary(),
            'history': [s.to_dict() for s in self.history],
        }
        with open(self.save_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)

    def save_csv(self, filename: str = "training_history.csv"):
        """Save history to CSV."""
        if not self.history:
            return
        with open(self.save_dir / filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].to_dict().keys()))
            writer.writeheader()
            for snapshot in self.history:
                writer.writerow(snapshot.to_dict())


# =============================================================================
# Publication Figure Generator
# =============================================================================

class PublicationFigures:
    """
    Generate publication-quality figures for gauge VFE transformer paper.

    Plots training curves, KL-attention heatmaps, model comparisons,
    scaling studies, and attention entropy diagnostics.
    """

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("./outputs/figures")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        apply_publication_style()

    def plot_training_curves(
        self,
        tracker: TrainingTracker,
        save_name: str = "training_curves",
        show_components: bool = True,
        start_step: int = 100,
    ) -> "Any":
        """
        Plot training curves: Loss, PPL, BPC.

        Args:
            tracker: TrainingTracker with recorded history
            save_name: Filename for saved figure
            show_components: If True, show loss decomposition
            start_step: Skip initial steps to avoid transient spikes (default: 5)

        Returns:
            matplotlib Figure
        """
        if not tracker.history:
            raise ValueError("No training history to plot")

        # Filter history to skip initial transient
        history = [s for s in tracker.history if s.step >= start_step]
        if not history:
            history = tracker.history  # Fallback if all filtered out
        steps = [s.step for s in history]

        # Validation data
        val_steps = [s.step for s in history if s.val_ppl is not None]
        val_ces = [s.val_ce for s in history if s.val_ce is not None]
        val_ppls = [s.val_ppl for s in history if s.val_ppl is not None]
        val_bpcs = [s.val_bpc for s in history if s.val_bpc is not None]

        if show_components:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes = axes.reshape(1, -1)

        # (a) CE Loss curves
        ax = axes[0, 0] if show_components else axes[0, 0]
        ax.plot(steps, [s.train_ce for s in history], 'b-',
                label='Train', alpha=0.7, linewidth=1)
        if val_ces:
            ax.plot(val_steps, val_ces,
                    'r-', label='Val', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('(a) Cross-Entropy Loss')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # (b) Perplexity
        ax = axes[0, 1] if show_components else axes[0, 1]
        ax.semilogy(steps, [s.train_ppl for s in history], 'b-',
                    label='Train', alpha=0.7, linewidth=1)
        if val_ppls:
            ax.semilogy(val_steps, val_ppls, 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Perplexity')
        ax.set_title('(b) Perplexity')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # (c) BPC
        ax = axes[1, 0] if show_components else axes[0, 2]
        ax.plot(steps, [s.train_bpc for s in history], 'b-',
                label='Train', alpha=0.7, linewidth=1)
        if val_bpcs:
            ax.plot(val_steps, val_bpcs, 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Bits per Token')
        ax.set_title('(c) Bits per Token')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if show_components:
            # (d) Gradient norms
            ax = axes[1, 1]
            ax.semilogy(steps, [s.grad_norm_total for s in history],
                        'k-', label='Total', alpha=0.8, linewidth=1.5)
            ax.semilogy(steps, [s.grad_norm_mu for s in history],
                        'b--', label='μ', alpha=0.6, linewidth=1)
            ax.semilogy(steps, [s.grad_norm_sigma for s in history],
                        'g--', label='Σ', alpha=0.6, linewidth=1)
            ax.semilogy(steps, [s.grad_norm_phi for s in history],
                        'r--', label='φ', alpha=0.6, linewidth=1)
            ax.semilogy(steps, [s.grad_norm_other for s in history],
                        'c:', label='Other', alpha=0.5, linewidth=1)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('(d) Gradient Norms')
            ax.legend(loc='upper right', ncol=3)
            ax.grid(True, alpha=0.3)

        # Format x-axis to show steps as k notation (150000 -> 150k)
        for ax in axes.flat:
            format_step_axis(ax)

        plt.tight_layout()

        
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=300)

        return fig

    def plot_gradient_norms_split(
        self,
        tracker: TrainingTracker,
        save_name: str = "gradient_norms_split",
        start_step: int = 100,
    ) -> "Any":
        """
        Plot E-step (q) and M-step (p) gradient norms as two separate figures.

        The E-step figure shows natural gradient norms from VFE iterations
        (belief inference within a single forward pass). The M-step figure
        shows backprop gradient norms on embedding parameters.

        Args:
            tracker: TrainingTracker with recorded history
            save_name: Base filename prefix for saved figures
            start_step: Skip initial steps to avoid transient spikes

        Returns:
            Tuple of (fig_mstep, fig_estep) matplotlib Figures
        """
        import matplotlib.pyplot as plt

        history = [s for s in tracker.history if s.step >= start_step]
        if not history:
            return None, None

        steps = [s.step for s in history]

        # --- M-step (p) gradient norms: embedding parameters only ---
        fig_m, ax_m = plt.subplots(1, 1, figsize=(7, 4.5))
        m_groups = [
            ('grad_norm_total', 'Total', 'k-', 0.8, 1.5),
            ('grad_norm_mu', r'$\mu$ embed', 'b--', 0.6, 1),
            ('grad_norm_sigma', r'$\Sigma$ embed', 'g--', 0.6, 1),
            ('grad_norm_phi', r'$\varphi$ embed', 'r--', 0.6, 1),
        ]
        for attr, label, style, alpha, lw in m_groups:
            vals = [getattr(s, attr) for s in history]
            if any(v > 0 for v in vals):
                ax_m.semilogy(steps, vals, style, label=label, alpha=alpha, linewidth=lw)
        ax_m.set_xlabel('Training Step')
        ax_m.set_ylabel('Gradient Norm')
        ax_m.set_title(r'M-step ($p$) Gradient Norms — Embedding Parameters')
        ax_m.legend(loc='upper right', ncol=2)
        ax_m.grid(True, alpha=0.3)
        format_step_axis(ax_m)
        fig_m.tight_layout()
        fig_m.savefig(self.save_dir / f"{save_name}_mstep.png", dpi=300)

        # --- M-step (p) gradient norms: FFN / decoder / other ---
        fig_d, ax_d = plt.subplots(1, 1, figsize=(7, 4.5))
        d_groups = [
            ('grad_norm_ffn', 'FFN / decoder', 'm-', 0.7, 1.5),
            ('grad_norm_other', 'Other', 'c--', 0.6, 1),
        ]
        has_decoder_data = any(
            getattr(s, 'grad_norm_ffn', 0) > 0 or getattr(s, 'grad_norm_other', 0) > 0
            for s in history
        )
        if has_decoder_data:
            for attr, label, style, alpha, lw in d_groups:
                vals = [getattr(s, attr) for s in history]
                if any(v > 0 for v in vals):
                    ax_d.semilogy(steps, vals, style, label=label, alpha=alpha, linewidth=lw)
        ax_d.set_xlabel('Training Step')
        ax_d.set_ylabel('Gradient Norm')
        ax_d.set_title(r'M-step ($p$) Gradient Norms — Decoder / Other')
        ax_d.legend(loc='upper right')
        ax_d.grid(True, alpha=0.3)
        format_step_axis(ax_d)
        fig_d.tight_layout()
        fig_d.savefig(self.save_dir / f"{save_name}_decoder.png", dpi=300)

        # --- E-step (q) gradient norms: natural gradients from VFE iterations ---
        fig_e, ax_e = plt.subplots(1, 1, figsize=(7, 4.5))
        e_groups = [
            ('e_step_nat_grad_mu', r'$\nabla_\mu$ (raw)', 'b-', 0.7, 1.5),
            ('e_step_nat_grad_mu_clipped', r'$\nabla_\mu$ (clipped)', 'b--', 0.5, 1),
            ('e_step_nat_grad_sigma', r'$\nabla_\Sigma$ (raw)', 'g-', 0.7, 1.5),
            ('e_step_nat_grad_sigma_clipped', r'$\nabla_\Sigma$ (clipped)', 'g--', 0.5, 1),
            ('e_step_grad_phi', r'$\nabla_\varphi$', 'r-', 0.7, 1.5),
        ]
        has_estep_data = any(
            getattr(s, 'e_step_nat_grad_mu', 0) > 0 for s in history
        )
        if has_estep_data:
            for attr, label, style, alpha, lw in e_groups:
                vals = [getattr(s, attr) for s in history]
                if any(v > 0 for v in vals):
                    ax_e.semilogy(steps, vals, style, label=label, alpha=alpha, linewidth=lw)
            ax_e.set_xlabel('Training Step')
            ax_e.set_ylabel('Gradient Norm')
            ax_e.set_title('E-step (q) Gradient Norms — Natural Gradients in VFE Iterations')
            ax_e.legend(loc='upper right', ncol=2)
            ax_e.grid(True, alpha=0.3)
        else:
            ax_e.text(0.5, 0.5, 'No E-step gradient data recorded',
                     transform=ax_e.transAxes, ha='center', va='center',
                     fontsize=14, color='gray')
            ax_e.set_title('E-step (q) Gradient Norms — No Data')
        format_step_axis(ax_e)
        fig_e.tight_layout()
        fig_e.savefig(self.save_dir / f"{save_name}_estep.png", dpi=300)

        return fig_m, fig_e, fig_d

    def plot_mstep_fisher_preconditioning(
        self,
        tracker: TrainingTracker,
        save_name: str = "mstep_fisher_preconditioning",
        start_step: int = 100,
    ) -> "Any":
        r"""Plot M-step raw vs Fisher-preconditioned gradient norms.

        Shows per-parameter-group comparison of $\|\partial L / \partial \theta\|$
        (Euclidean) vs $\|\hat{F}^{-1} \partial L / \partial \theta\|$
        (natural gradient after Fisher inversion). The ratio between these
        curves reveals how much the empirical Fisher reshapes the update
        direction for each parameter type.

        Only produces output when the NaturalGradientOptimizer is active
        and has logged per-group norms.

        Args:
            tracker: TrainingTracker with recorded history
            save_name: Filename for saved figure
            start_step: Skip initial steps to avoid transient spikes

        Returns:
            matplotlib Figure, or None if no M-step natural gradient data
        """
        import matplotlib.pyplot as plt

        history = [s for s in tracker.history if s.step >= start_step]
        if not history:
            return None

        # Check if any M-step natural gradient data exists
        has_data = any(
            getattr(s, 'mstep_nat_grad_mu', 0) > 0
            or getattr(s, 'mstep_nat_grad_sigma', 0) > 0
            or getattr(s, 'mstep_nat_grad_phi', 0) > 0
            for s in history
        )
        if not has_data:
            return None

        steps = [s.step for s in history]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Per-group: raw (Euclidean) vs natural (Fisher-preconditioned)
        groups = [
            (r'$\mu$ embed', 'grad_norm_mu', 'mstep_nat_grad_mu', 'tab:blue'),
            (r'$\Sigma$ embed', 'grad_norm_sigma', 'mstep_nat_grad_sigma', 'tab:green'),
            (r'$\varphi$ embed', 'grad_norm_phi', 'mstep_nat_grad_phi', 'tab:red'),
        ]

        for ax, (label, eucl_attr, nat_attr, color) in zip(axes, groups):
            eucl_vals = [getattr(s, eucl_attr) for s in history]
            nat_vals = [getattr(s, nat_attr) for s in history]

            has_eucl = any(v > 0 for v in eucl_vals)
            has_nat = any(v > 0 for v in nat_vals)

            if has_eucl:
                ax.semilogy(steps, eucl_vals, '-', color=color, alpha=0.7,
                           linewidth=1.5, label=r'Euclidean $\|g\|$')
            if has_nat:
                ax.semilogy(steps, nat_vals, '--', color=color, alpha=0.7,
                           linewidth=1.5, label=r'Natural $\|\hat{F}^{-1}g\|$')

            ax.set_xlabel('Training Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title(label)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            format_step_axis(ax)

        fig.suptitle(
            r'M-step Fisher Preconditioning: $\|g\|$ vs $\|\hat{F}^{-1}g\|$',
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=300,
                   bbox_inches='tight')

        return fig

    def plot_attention_heatmap(
        self,
        beta: torch.Tensor,
        tokens: Optional[List[str]] = None,
        save_name: str = "attention_heatmap",
        title: str = "KL-Divergence Attention Weights",
        head_idx: Optional[int] = None,
    ) -> "Any":
        """
        Plot attention heatmap from KL-divergence based attention.

        Args:
            beta: Attention weights (B, N, N) or (B, H, N, N)
            tokens: Optional token labels
            save_name: Filename for saved figure
            title: Figure title
            head_idx: If multi-head, which head to plot (None = average)

        Returns:
            matplotlib Figure
        """
        # Handle multi-head attention
        if beta.dim() == 4:  # (B, H, N, N)
            if head_idx is not None:
                attn = beta[0, head_idx].detach().cpu().numpy()
                title = f"{title} (Head {head_idx})"
            else:
                attn = beta[0].mean(dim=0).detach().cpu().numpy()
                title = f"{title} (Averaged)"
        else:  # (B, N, N)
            attn = beta[0].detach().cpu().numpy()

        N = attn.shape[0]

        fig, ax = plt.subplots(figsize=(8, 7))

        # Use log scale for better visualization of attention distribution
        im = ax.imshow(attn, cmap='Blues', aspect='auto')

        # Labels
        if tokens is not None and len(tokens) <= 32:
            tokens_display = [t[:10] for t in tokens]  # Truncate long tokens
            ax.set_xticks(range(len(tokens_display)))
            ax.set_yticks(range(len(tokens_display)))
            ax.set_xticklabels(tokens_display, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(tokens_display, fontsize=8)
        else:
            # Show every 10th tick for long sequences
            tick_spacing = max(1, N // 10)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        ax.set_xlabel('Key Position (j)')
        ax.set_ylabel('Query Position (i)')
        ax.set_title(title)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('β_ij')

        plt.tight_layout()

        
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=300)

        return fig

    def plot_model_comparison(
        self,
        results: List[ExperimentResult],
        save_name: str = "model_comparison",
        metric: str = "ppl",
    ) -> "Any":
        """
        Plot bar chart comparing different model architectures.

        Args:
            results: List of ExperimentResult from different configs
            save_name: Filename for saved figure
            metric: 'ppl' or 'bpc'

        Returns:
            matplotlib Figure
        """
        if not results:
            raise ValueError("No results to plot")

        names = [r.name for r in results]

        if metric == "ppl":
            values = [r.final_val_ppl for r in results]
            ylabel = "Validation Perplexity"
            title = "Model Comparison: Perplexity"
        else:
            values = [r.final_val_bpc for r in results]
            ylabel = "Validation BPC"
            title = "Model Comparison: Bits per Token"

        # Color scheme
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(results)]

        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(range(len(names)), values, color=colors, edgecolor='black', linewidth=1)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        # Add parameter count annotation
        param_text = '\n'.join([f"{r.name}: {r.total_params/1000:.1f}K params" for r in results])
        ax.text(0.98, 0.98, param_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=300)

        return fig

    def plot_scaling_study(
        self,
        k_values: List[int],
        ppl_values: List[float],
        param_counts: Optional[List[int]] = None,
        save_name: str = "scaling_study",
    ) -> "Any":
        """
        Plot scaling study: PPL vs embedding dimension K.

        Args:
            k_values: List of embedding dimensions tested
            ppl_values: Corresponding perplexity values
            param_counts: Optional parameter counts
            save_name: Filename for saved figure

        Returns:
            matplotlib Figure
        """
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # PPL vs K
        color = 'tab:blue'
        ax1.set_xlabel('Embedding Dimension (K)')
        ax1.set_ylabel('Validation Perplexity', color=color)
        line1 = ax1.plot(k_values, ppl_values, 'o-', color=color,
                         linewidth=2, markersize=8, label='PPL')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        # Mark optimal K
        min_idx = np.argmin(ppl_values)
        ax1.axvline(x=k_values[min_idx], color='red', linestyle='--',
                   alpha=0.5, label=f'Optimal K={k_values[min_idx]}')
        ax1.scatter([k_values[min_idx]], [ppl_values[min_idx]],
                   color='red', s=150, zorder=5, marker='*')

        # Add parameter counts on secondary axis
        if param_counts:
            ax2 = ax1.twinx()
            color = 'tab:gray'
            ax2.set_ylabel('Parameters (K)', color=color)
            line2 = ax2.plot(k_values, [p/1000 for p in param_counts],
                            's--', color=color, alpha=0.6, linewidth=1,
                            markersize=5, label='Params')
            ax2.tick_params(axis='y', labelcolor=color)

        ax1.set_title('Scaling Study: Performance vs Embedding Dimension')
        ax1.legend(loc='upper right')

        plt.tight_layout()

        
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=300)

        return fig

    def plot_train_val_gap(
        self,
        tracker: TrainingTracker,
        save_name: str = "train_val_gap",
    ) -> "Any":
        """
        Plot train-validation gap over training.

        Shows generalization behavior - gap should stay small.

        Args:
            tracker: TrainingTracker with recorded history
            save_name: Filename for saved figure

        Returns:
            matplotlib Figure
        """
        history = tracker.history

        # Get steps where we have both train and val
        val_steps = []
        gaps_bpc = []

        for s in history:
            if s.val_bpc is not None:
                val_steps.append(s.step)
                gaps_bpc.append(s.val_bpc - s.train_bpc)

        if not val_steps:
            raise ValueError("No validation data to plot")

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(val_steps, gaps_bpc, 'b-o', linewidth=1.5, markersize=4)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.fill_between(val_steps, gaps_bpc, 0, alpha=0.3,
                        color='green' if np.mean(gaps_bpc) < 0 else 'red')

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Val BPC - Train BPC')

        # Title with final gap as subtitle
        final_gap = gaps_bpc[-1]
        ax.set_title(f'Generalization Gap\nFinal: {final_gap:+.3f}')
        ax.grid(True, alpha=0.3)

        # Format x-axis to show steps as k notation (150000 -> 150k)
        format_step_axis(ax)

        plt.tight_layout()

        
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=300)

        return fig

    def plot_attention_entropy(
        self,
        tracker: TrainingTracker,
        save_name: str = "attention_entropy",
        start_step: int = 100,
    ) -> "Any":
        """
        Plot attention entropy over training.

        This diagnostic shows whether attention sharpens (entropy decreases)
        or stays uniform (entropy stays high) during training. For meaningful
        attention patterns, entropy should decrease as the model learns to
        focus on relevant tokens.

        Uniform attention (high entropy) suggests the KL-divergence attention
        mechanism may not be learning discriminative patterns.

        Args:
            tracker: TrainingTracker with recorded history
            save_name: Filename for saved figure
            start_step: Skip initial steps to avoid transient spikes (default: 5)

        Returns:
            matplotlib Figure
        """
        # Filter history to skip initial transient
        history = [s for s in tracker.history if s.step >= start_step]
        if not history:
            history = tracker.history  # Fallback if all filtered out
        steps = [s.step for s in history]
        entropies = [s.attention_entropy for s in history]
        concentrations = [s.attention_concentration for s in history]

        if not steps or all(e == 0 for e in entropies):
            raise ValueError("No attention entropy data to plot")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # (a) Attention Entropy over training
        ax = axes[0]
        ax.plot(steps, entropies, 'b-', linewidth=1.5, alpha=0.8)

        # Add smoothed trend line
        if len(steps) > 10:
            window = min(len(steps) // 10, 50)
            smoothed = np.convolve(entropies, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
            ax.plot(smooth_steps, smoothed, 'r-', linewidth=2, label='Smoothed', alpha=0.9)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Attention Entropy (nats)')
        ax.set_title('(a) Attention Entropy Over Training')
        ax.grid(True, alpha=0.3)

        # Add reference lines
        # For context length N, max entropy is log(N)
        if entropies:
            max_entropy = max(entropies)
            ax.axhline(y=max_entropy, color='gray', linestyle='--', linewidth=1,
                      label=f'Max observed: {max_entropy:.2f}')

        # Annotate final value
        if entropies:
            final_entropy = entropies[-1]
            ax.annotate(f'Final: {final_entropy:.3f}',
                       xy=(steps[-1], final_entropy),
                       xytext=(-50, 10), textcoords='offset points',
                       fontsize=10, ha='right',
                       arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

        ax.legend(loc='upper right')

        # (b) Attention Concentration over training
        ax = axes[1]
        ax.plot(steps, concentrations, 'g-', linewidth=1.5, alpha=0.8)

        # Add smoothed trend line
        if len(steps) > 10:
            window = min(len(steps) // 10, 50)
            smoothed = np.convolve(concentrations, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
            ax.plot(smooth_steps, smoothed, 'r-', linewidth=2, label='Smoothed', alpha=0.9)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Attention Concentration')
        ax.set_title('(b) Attention Concentration Over Training')
        ax.grid(True, alpha=0.3)

        # Annotate final value
        if concentrations:
            final_conc = concentrations[-1]
            ax.annotate(f'Final: {final_conc:.3f}',
                       xy=(steps[-1], final_conc),
                       xytext=(-50, 10), textcoords='offset points',
                       fontsize=10, ha='right',
                       arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

        ax.legend(loc='upper left')

        # Format x-axis to show steps as k notation (150000 -> 150k)
        for ax in axes:
            format_step_axis(ax)

        plt.tight_layout()

        # Add figure caption
        fig.text(0.5, -0.02,
                'Lower entropy = sharper attention (focused). Higher entropy = uniform attention (diffuse).',
                ha='center', fontsize=9, style='italic')

        
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')

        return fig


# =============================================================================
# Main Publication Metrics Coordinator
# =============================================================================

class PublicationMetrics:
    """
    Main coordinator for publication metrics and figures.

    Usage:
        metrics = PublicationMetrics("wikitext103_experiment")

        # During training
        for step in range(max_steps):
            metrics.record_step(step, epoch, train_metrics, grad_norms)
            if step % eval_interval == 0:
                metrics.record_validation(step, val_metrics)

        # After training
        metrics.save_all()
        metrics.generate_figures()
    """

    def __init__(self, experiment_name: str, base_dir: Optional[Path] = None):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir) if base_dir else Path("./outputs")
        self.experiment_dir = self.base_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = TrainingTracker(self.experiment_dir)
        self.figures = PublicationFigures(self.experiment_dir / "figures")

        self.comparison_results: List[ExperimentResult] = []
        self.scaling_data: Dict[str, List] = {'k': [], 'ppl': [], 'params': []}

        # Gauge frame semantic analysis history
        self.semantic_analysis_history: List[Dict[str, Any]] = []
        self.semantic_analysis_interval: int = 10000  # Default: analyze every 10k steps

        # Semantic trajectory tracker (lightweight per-step snapshots)
        self.semantic_tracker = SemanticTrajectoryTracker(interval=5000)

        # Holonomy tracking for non-flat transport experiments
        self.holonomy_history: List[HolonomyProfile] = []
        self.holonomy_interval: int = 500  # Default: compute every 500 steps
        self.holonomy_sample_size: int = 500  # Triples per computation

        print(f"[PublicationMetrics] Initialized: {self.experiment_dir}")

    def record_step(
        self,
        step: int,
        epoch: float,
        train_metrics: Dict[str, float],
        grad_norms: Optional[Dict[str, float]] = None,
        lr: float = 0.0,
        step_time: float = 0.0,
        batch_size: int = 1,
        seq_len: int = 1,
        e_step_norms: Optional[Dict[str, float]] = None,
        mstep_natural_norms: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Record training step metrics."""
        self.tracker.record(step, epoch, train_metrics, grad_norms,
                           lr, step_time, batch_size, seq_len,
                           e_step_norms=e_step_norms,
                           mstep_natural_norms=mstep_natural_norms)

    def record_training_step(
        self,
        step: int,
        epoch: float,
        train_metrics: Dict[str, float],
        diagnostics: Optional[Dict] = None,
        grad_norms: Optional[Dict[str, float]] = None,
        lrs: Optional[Dict[str, float]] = None,
        step_time: float = 0.0,
        batch_size: int = 1,
        seq_len: int = 1,
        e_step_norms: Optional[Dict[str, float]] = None,
        mstep_natural_norms: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Record training step metrics (compatibility wrapper)."""
        # Extract lr from lrs dict if provided
        lr = lrs.get('mu_embed', 0.0) if lrs else 0.0
        self.tracker.record(step, epoch, train_metrics, grad_norms,
                           lr, step_time, batch_size, seq_len,
                           e_step_norms=e_step_norms,
                           mstep_natural_norms=mstep_natural_norms)

    def record_validation(self, step: int, val_metrics: Dict[str, float]):
        """Record validation metrics."""
        self.tracker.record_validation(step, val_metrics)

    def add_comparison_result(self, result: ExperimentResult):
        """Add a result for model comparison."""
        self.comparison_results.append(result)

    def add_scaling_point(self, k: int, ppl: float, params: int):
        """Add a point for scaling study."""
        self.scaling_data['k'].append(k)
        self.scaling_data['ppl'].append(ppl)
        self.scaling_data['params'].append(params)

    def set_semantic_analysis_interval(self, interval: int):
        """Set how often to run gauge frame semantic analysis (in steps)."""
        self.semantic_analysis_interval = interval

    def should_run_semantic_analysis(self, step: int) -> bool:
        """Check if semantic analysis should run at this step."""
        if self.semantic_analysis_interval <= 0:
            return False
        return step % self.semantic_analysis_interval == 0

    def run_semantic_analysis(
        self,
        model: Any,
        step: int,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run gauge frame semantic analysis on the model.

        This analyzes whether gauge frames φ encode semantic relationships
        by computing distance metrics between token classes and word pairs.

        Args:
            model: The model with mu_embed and phi_embed attributes
            step: Current training step
            verbose: Whether to print analysis results

        Returns:
            Dictionary with analysis results
        """
        figures_dir = self.experiment_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        results = analyze_gauge_semantics(
            model=model,
            step=step,
            save_dir=figures_dir,
            save_plots=True,
            verbose=verbose,
        )

        # Store in history
        self.semantic_analysis_history.append(results)

        # Record lightweight trajectory snapshot
        if self.semantic_tracker.should_record(step):
            self.semantic_tracker.record(model, step)

        
        return results

    def maybe_record_semantic_trajectory(self, model: Any, step: int) -> None:
        """Record a lightweight semantic trajectory snapshot if due.

        Call this every step (or every log step). The tracker's internal interval
        determines whether a snapshot is actually recorded. This is separate from
        run_semantic_analysis() so trajectory snapshots can be taken at a higher
        frequency than full semantic analysis.
        """
        if self.semantic_tracker.should_record(step):
            try:
                self.semantic_tracker.record(model, step)
            except Exception:
                pass  # Silent — trajectory is diagnostic, not critical

    def run_final_semantic_analysis(self, model: Any, verbose: bool = True) -> Dict[str, Any]:
        """
        Run final comprehensive semantic analysis at end of training.

        Args:
            model: The trained model
            verbose: Whether to print detailed results

        Returns:
            Dictionary with analysis results
        """
        print("\n[PublicationMetrics] Running final gauge frame semantic analysis...")

        figures_dir = self.experiment_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        results = analyze_gauge_semantics(
            model=model,
            step=None,  # Final analysis, no step number
            save_dir=figures_dir,
            save_plots=True,
            verbose=verbose,
        )

        # Save analysis history to JSON
        if self.semantic_analysis_history:
            history_path = self.experiment_dir / "semantic_analysis_history.json"
            with open(history_path, 'w') as f:
                # Convert non-serializable items (including numpy types)
                def make_serializable(obj):
                    if isinstance(obj, dict):
                        return {k: make_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [make_serializable(v) for v in obj]
                    elif isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (int, float, str, bool, type(None))):
                        return obj
                    else:
                        return str(obj)  # Fallback to string representation

                serializable_history = [make_serializable(entry) for entry in self.semantic_analysis_history]
                json.dump(serializable_history, f, indent=2)
            print(f"  Saved semantic analysis history to: {history_path}")

        # Save semantic trajectory
        if self.semantic_tracker.snapshots:
            traj_path = self.experiment_dir / "semantic_trajectory.json"
            self.semantic_tracker.save(traj_path)
            trajectory_summary = self.semantic_tracker.summarize()
            results['trajectory_summary'] = trajectory_summary
            print(f"  Saved semantic trajectory ({len(self.semantic_tracker.snapshots)} snapshots) to: {traj_path}")

        # Holonomy-semantic correlation
        try:
            holonomy_sem = analyze_holonomy_semantic_correlation(
                model=model, n_tokens=200, n_triangles=100, verbose=verbose,
            )
            if 'error' not in holonomy_sem:
                results['holonomy_semantic'] = holonomy_sem
        except Exception as e:
            if verbose:
                print(f"  [WARN] Holonomy-semantic analysis failed: {e}")

        return results

    # ------------------------------------------------------------------
    # Holonomy (Non-Flat Transport Curvature)
    # ------------------------------------------------------------------

    def set_holonomy_interval(self, interval: int, sample_size: int = 500):
        """Configure holonomy computation frequency.

        Args:
            interval: Compute holonomy every N steps (0 = disabled).
            sample_size: Number of random triples per computation.
        """
        self.holonomy_interval = interval
        self.holonomy_sample_size = sample_size

    def should_compute_holonomy(self, step: int) -> bool:
        """Check if holonomy diagnostics should run at this step."""
        if self.holonomy_interval <= 0:
            return False
        return step % self.holonomy_interval == 0

    def compute_holonomy_diagnostics(
        self,
        model: Any,
        step: int,
        verbose: bool = False,
    ) -> Optional[Dict[str, float]]:
        """Compute holonomy metrics from the model's current state.

        Finds all blocks with non-flat transport (GaugeConnection),
        recomputes exp(δ_ij · G) from embeddings, and computes holonomy.

        Args:
            model: The GaugeTransformerLM model.
            step: Current training step.
            verbose: Print diagnostics.

        Returns:
            Flat dict of holonomy metrics suitable for logging, or None
            if the model has no non-flat transport.
        """
        blocks = self._find_gauge_blocks(model)
        if not blocks:
            return None

        snapshots = []
        all_norms = []

        with torch.no_grad():
            for layer_idx, block in blocks:
                exp_delta = self._extract_exp_delta(model, block)
                if exp_delta is None:
                    if verbose:
                        print(f"  [HOLONOMY] Skipping layer {layer_idx}: could not extract exp_delta")
                    continue

                snap = compute_holonomy_snapshot(
                    exp_delta,
                    step=step,
                    layer=layer_idx,
                    head=0,
                    sample_size=self.holonomy_sample_size,
                    seed=42,
                )
                snapshots.append(snap)
                all_norms.append(snap.mean_norm)

        if not snapshots:
            return None

        profile = HolonomyProfile(
            step=step,
            snapshots=snapshots,
            global_mean_norm=float(np.mean(all_norms)),
            global_max_norm=float(np.max(all_norms)),
        )
        self.holonomy_history.append(profile)

        # Build flat logging dict
        log_dict = {
            'holonomy/mean_norm': profile.global_mean_norm,
            'holonomy/max_norm': profile.global_max_norm,
        }
        # Aggregate detailed metrics across all layers
        if snapshots:
            log_dict.update({
                'holonomy/frac_gt_0.1': float(np.mean([s.frac_gt_01 for s in snapshots])),
                'holonomy/spectral_gap': float(np.mean([s.mean_spectral_gap for s in snapshots])),
                'holonomy/wilson_trace': float(np.mean([s.mean_wilson_trace for s in snapshots])),
            })

        # Track connection weight norms to verify W is actually evolving
        w_norms = []
        for _, block in blocks:
            gc = getattr(block, 'gauge_connection', None)
            if gc is not None:
                W = getattr(gc, 'W', None)  # bilinear
                if W is not None:
                    w_norms.append(W.data.norm().item())
                else:
                    # MLP: track output layer weight norm
                    net = getattr(gc, 'net', None)
                    if net is not None:
                        w_norms.append(net[-1].weight.data.norm().item())
        if w_norms:
            log_dict['holonomy/connection_w_norm'] = float(np.mean(w_norms))

        if verbose:
            w_info = f" | ‖W‖={np.mean(w_norms):.6f}" if w_norms else ""
            # Per-snapshot detail: show the actual within-layer distribution
            for s in snapshots:
                print(f"\n  [HOLONOMY] L{s.layer}H{s.head} "
                      f"mean ‖C-I‖={s.mean_norm:.6f} | "
                      f"max={s.max_norm:.6f} | "
                      f"std={s.std_norm:.6f} | "
                      f"median={s.median_norm:.6f} | "
                      f"frac>0.1={s.frac_gt_01:.3f} | "
                      f"spectral_gap={s.mean_spectral_gap:.6f}"
                      f"{w_info}\n\n")
            # Summary line when multiple layers/heads
            if len(snapshots) > 1:
                print(f"\n  [HOLONOMY] global: "
                      f"mean={profile.global_mean_norm:.6f} | "
                      f"max={profile.global_max_norm:.6f} | "
                      f"cross-layer std={np.std([s.mean_norm for s in snapshots]):.6f}\n\n")

        return log_dict

    def generate_holonomy_figures(
        self,
        model: Any = None,
        save_prefix: str = 'holonomy',
    ):
        """Generate all holonomy publication figures.

        Args:
            model: Optional model for extracting current exp_delta (for
                   distribution and spectrum plots at current state).
            save_prefix: Filename prefix for saved figures.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("[WARN] matplotlib not available, skipping holonomy figures")
            return

        try:
            from transformer.visualization.holonomy_plots import (
                plot_holonomy_evolution,
                plot_holonomy_distribution,
                plot_holonomy_summary,
                plot_wilson_spectrum,
                plot_curvature_vs_distance,
                plot_layer_holonomy_profile,
            )
        except ImportError:
            print("[WARN] holonomy_plots not available, skipping figures")
            return

        figures_dir = self.experiment_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        if not self.holonomy_history:
            print("[WARN] No holonomy history to plot")
            return

        trajectory = compute_flatness_trajectory(self.holonomy_history)

        # Figure: Evolution over training
        if len(trajectory['steps']) > 1:
            try:
                # Convert per_layer_mean dict → 2D ndarray for plot function
                plm_dict = trajectory['per_layer_mean']
                layer_indices = trajectory.get('layer_indices', [])
                if plm_dict and layer_indices:
                    per_layer_mean = np.column_stack([plm_dict[li] for li in layer_indices])
                else:
                    per_layer_mean = None

                plot_holonomy_evolution(
                    steps=trajectory['steps'],
                    global_mean=trajectory['global_mean'],
                    global_max=trajectory['global_max'],
                    per_layer_mean=per_layer_mean,
                    layer_indices=layer_indices if layer_indices else None,
                    title='Holonomy Evolution During Training',
                    output_path=figures_dir / f'{save_prefix}_evolution.png',
                )
                plt.close('all')
                print(f"  Saved {save_prefix}_evolution.png")
            except Exception as e:
                print(f"[WARN] Holonomy evolution plot failed: {e}")

        # Figure: Per-layer profile (using latest snapshot)
        latest = self.holonomy_history[-1]
        if latest.snapshots:
            try:
                layer_means = np.array([s.mean_norm for s in latest.snapshots])
                layer_stds = np.array([s.std_norm for s in latest.snapshots])
                plot_layer_holonomy_profile(
                    layer_means=layer_means,
                    layer_stds=layer_stds,
                    title='Holonomy by Layer (Final)',
                    output_path=figures_dir / f'{save_prefix}_layer_profile.png',
                )
                plt.close('all')
                print(f"  Saved {save_prefix}_layer_profile.png")
            except Exception as e:
                print(f"[WARN] Holonomy layer profile plot failed: {e}")

        # Figure: Distribution + Spectrum (requires model for current exp_delta)
        if model is not None:
            blocks = self._find_gauge_blocks(model)
            if blocks:
                try:
                    _, block = blocks[0]
                    exp_delta = self._extract_exp_delta(model, block)
                    if exp_delta is not None:
                        C, norms, _ = compute_holonomy(
                            exp_delta, sample_size=min(2000, self.holonomy_sample_size * 4)
                        )
                        norms_np = norms.detach().cpu().float().mean(dim=0).numpy()
                        C_np = C.detach().cpu().float()[0].numpy()  # first batch

                        plot_holonomy_distribution(
                            norms_np,
                            title='Holonomy Norm Distribution (Final)',
                            output_path=figures_dir / f'{save_prefix}_distribution.png',
                        )
                        plt.close('all')
                        print(f"  Saved {save_prefix}_distribution.png")

                        plot_wilson_spectrum(
                            C_np,
                            title='Wilson Loop Eigenvalue Spectrum (Final)',
                            output_path=figures_dir / f'{save_prefix}_wilson_spectrum.png',
                        )
                        plt.close('all')
                        print(f"  Saved {save_prefix}_wilson_spectrum.png")

                        # Curvature vs distance
                        dist_data = compute_curvature_by_distance(exp_delta)
                        plot_curvature_vs_distance(
                            bin_centers=dist_data['distances'],
                            mean_norms=dist_data['mean_norms'],
                            std_norms=dist_data['std_norms'],
                            counts=np.ones_like(dist_data['distances']),
                            title='Curvature vs Token Distance (Final)',
                            output_path=figures_dir / f'{save_prefix}_curvature_vs_distance.png',
                        )
                        plt.close('all')
                        print(f"  Saved {save_prefix}_curvature_vs_distance.png")

                except Exception as e:
                    print(f"[WARN] Holonomy distribution/spectrum plots failed: {e}")

        # Save holonomy history to JSON
        try:
            history_data = []
            for h in self.holonomy_history:
                entry = {'step': h.step, 'global_mean': h.global_mean_norm,
                         'global_max': h.global_max_norm}
                for s in h.snapshots:
                    entry[f'L{s.layer}_mean'] = s.mean_norm
                    entry[f'L{s.layer}_spectral_gap'] = s.mean_spectral_gap
                    entry[f'L{s.layer}_wilson'] = s.mean_wilson_trace
                history_data.append(entry)
            with open(self.experiment_dir / 'holonomy_history.json', 'w') as f:
                json.dump(history_data, f, indent=2)
            print(f"  Saved holonomy_history.json")
        except Exception as e:
            print(f"[WARN] Could not save holonomy history JSON: {e}")

    def _find_gauge_blocks(self, model: Any) -> list:
        """Find all GaugeTransformerBlocks with non-flat transport."""
        blocks = []
        # Try both attribute names: 'transformer' (GaugeTransformerLM) and 'stack' (legacy)
        stack = getattr(model, 'transformer', None) or getattr(model, 'stack', None)
        if stack is not None:
            block_list = getattr(stack, 'blocks', None)
            if block_list is not None:
                for i, block in enumerate(block_list):
                    if getattr(block, 'non_flat_transport', False):
                        blocks.append((i, block))
        return blocks

    def _extract_exp_delta(self, model: Any, block: Any) -> Optional[torch.Tensor]:
        """Recompute exp(δ_ij · G) from current model state.

        Uses the raw connection delta (without cocycle_relaxation scaling)
        so that holonomy diagnostics measure the connection's intrinsic
        curvature even when cocycle_relaxation is scheduled from 0→1.

        When use_prior_bank=True, uses PriorBank embeddings (which are the
        actual inputs to gauge_connection during training) instead of the
        unused token_embed.mu_embed weights.
        """
        try:
            # Prefer PriorBank embeddings when available — these are the actual
            # inputs to gauge_connection during training. Using token_embed when
            # prior_bank is active would show frozen diagnostics because
            # token_embed.mu_embed never receives gradients in that mode.
            prior_bank = getattr(model, 'prior_bank', None)
            if prior_bank is not None and getattr(model, 'use_prior_bank', False):
                if getattr(prior_bank, 'gauge_fixed_priors', False):
                    # Gauge-fixed: mu = R_v @ base_mu. Sample first N tokens.
                    N = min(32, getattr(prior_bank, 'vocab_size', 32))
                    token_ids = torch.arange(N, device=next(model.parameters()).device)
                    pb_out = prior_bank.encode(token_ids.unsqueeze(0))
                    mu = pb_out[0]  # (1, N, K)
                else:
                    # Standard PriorBank: per-token mu lookup
                    prior_mu = getattr(prior_bank, 'prior_mu', None)
                    if prior_mu is None:
                        return None
                    N = min(32, prior_mu.shape[0])
                    mu = prior_mu[:N].unsqueeze(0)  # (1, N, K)
            else:
                embed = getattr(model, 'token_embed', None) or getattr(model, 'token_embedding', None)
                if embed is None:
                    return None
                mu_embed = getattr(embed, 'mu_embed', None)
                if mu_embed is None:
                    return None

                N = min(32, mu_embed.weight.shape[0])
                mu = mu_embed.weight[:N].unsqueeze(0)  # (1, N, K)

            generators = getattr(model, 'generators', None)
            if generators is None:
                return None

            delta = block.gauge_connection(mu, mu)  # (1, N, N, n_gen)
            # Use raw delta for diagnostics — cocycle_relaxation=0 would
            # zero out the connection and produce trivially flat holonomy,
            # hiding the learned curvature structure.
            delta_matrix = torch.einsum('bija,akl->bijkl', delta, generators)
            exp_delta = torch.linalg.matrix_exp(delta_matrix.float())
            return exp_delta
        except Exception:
            return None

    def save_all(self):
        """Save all metrics to files."""
        self.tracker.save_json()
        self.tracker.save_csv()

        # Save comparison results
        if self.comparison_results:
            with open(self.experiment_dir / "comparison_results.json", 'w') as f:
                json.dump([r.to_dict() for r in self.comparison_results], f, indent=2)

        # Save scaling data
        if self.scaling_data['k']:
            with open(self.experiment_dir / "scaling_data.json", 'w') as f:
                json.dump(self.scaling_data, f, indent=2)

        print(f"[PublicationMetrics] Saved to {self.experiment_dir}")

    def generate_figures(self, attention_weights: Optional[torch.Tensor] = None, model: Any = None):
        """Generate all publication figures."""
        figures_generated = []

        # Training curves
        try:
            self.figures.plot_training_curves(self.tracker)
            figures_generated.append("training_curves")
            plt.close()
        except Exception as e:
            print(f"[WARN] Could not generate training curves: {e}")

        # Split gradient norms (E-step / M-step)
        try:
            self.figures.plot_gradient_norms_split(self.tracker)
            figures_generated.append("gradient_norms_split")
            plt.close('all')
        except Exception as e:
            print(f"[WARN] Could not generate split gradient norms: {e}")

        # M-step Fisher preconditioning (raw vs natural gradient)
        try:
            fig = self.figures.plot_mstep_fisher_preconditioning(self.tracker)
            if fig is not None:
                figures_generated.append("mstep_fisher_preconditioning")
                plt.close('all')
        except Exception as e:
            print(f"[WARN] Could not generate Fisher preconditioning plot: {e}")

        # Train-val gap
        try:
            self.figures.plot_train_val_gap(self.tracker)
            figures_generated.append("train_val_gap")
            plt.close()
        except Exception as e:
            print(f"[WARN] Could not generate train-val gap: {e}")

        # Attention entropy over training
        try:
            self.figures.plot_attention_entropy(self.tracker)
            figures_generated.append("attention_entropy")
            plt.close()
        except Exception as e:
            print(f"[WARN] Could not generate attention entropy plot: {e}")

        # Model comparison
        if self.comparison_results:
            try:
                self.figures.plot_model_comparison(self.comparison_results)
                figures_generated.append("model_comparison")
                plt.close()
            except Exception as e:
                print(f"[WARN] Could not generate model comparison: {e}")

        # Scaling study
        if self.scaling_data['k']:
            try:
                self.figures.plot_scaling_study(
                    self.scaling_data['k'],
                    self.scaling_data['ppl'],
                    self.scaling_data['params']
                )
                figures_generated.append("scaling_study")
                plt.close()
            except Exception as e:
                print(f"[WARN] Could not generate scaling study: {e}")

        # Attention heatmap
        if attention_weights is not None:
            try:
                self.figures.plot_attention_heatmap(attention_weights)
                figures_generated.append("attention_heatmap")
                plt.close()
            except Exception as e:
                print(f"[WARN] Could not generate attention heatmap: {e}")

        # Holonomy figures (non-flat transport)
        if self.holonomy_history:
            try:
                self.generate_holonomy_figures(model=model)
                figures_generated.append("holonomy")
            except Exception as e:
                print(f"[WARN] Could not generate holonomy figures: {e}")

        print(f"[PublicationMetrics] Generated figures: {', '.join(figures_generated)}")

    def generate_all_figures(self, attention_weights: Optional[torch.Tensor] = None, model: Any = None):
        """Alias for generate_figures (compatibility)."""
        self.generate_figures(attention_weights, model=model)

    def generate_interpretability_outputs(
        self,
        model: Any,
        sample_batch: Tuple,
        tokenizer: Any = None,
        device: str = 'cpu',
    ):
        """
        Generate interpretability outputs (attention patterns, etc).

        Args:
            model: The trained model
            sample_batch: A sample batch (input_ids, target_ids)
            tokenizer: Optional tokenizer for decoding
            device: Device to run on
        """
        try:
            input_ids, _ = sample_batch
            input_ids = input_ids.to(device)

            # IMPORTANT: Show what sequence we're visualizing!
            print(f"\n[PublicationMetrics] Generating interpretability outputs")
            print(f"  Sequence shape: {input_ids.shape}")
            print(f"  Token IDs (first 20): {input_ids[0, :20].tolist()}")

            # Decode tokens if tokenizer available
            tokens = None
            if tokenizer is not None:
                try:
                    decoded_text = tokenizer.decode(input_ids[0].tolist())
                    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
                    print(f"  Decoded text: {decoded_text[:200]}{'...' if len(decoded_text) > 200 else ''}")
                except Exception as e:
                    print(f"  [WARN] Could not decode tokens: {e}")
            else:
                print(f"  [WARN] No tokenizer provided - cannot decode sequence")

            model.eval()
            with torch.no_grad():
                if hasattr(model, 'forward_with_attention'):
                    _, attn_info = model.forward_with_attention(input_ids, targets=None)
                    beta = attn_info.get('beta')
                    if beta is not None:
                        # Use final layer for publication figures
                        if beta.dim() == 5:
                            beta = beta[-1]  # (B, n_heads, N, N)
                        # Plot with tokens if available
                        if hasattr(self.figures, 'plot_attention_heatmap'):
                            self.figures.plot_attention_heatmap(
                                beta,
                                tokens=tokens,
                                save_name="attention_heatmap",
                                title="KL-Divergence Attention"
                            )
                        self.generate_figures(attention_weights=beta, model=model)
            model.train()
            print(f"[PublicationMetrics] ✓ Generated interpretability outputs")
        except Exception as e:
            print(f"[WARN] Could not generate interpretability outputs: {e}")

    def print_summary(self):
        """Print experiment summary."""
        summary = self.tracker.get_summary()

        print("\n" + "=" * 60)
        print(f"EXPERIMENT: {self.experiment_name}")
        print("=" * 60)

        if summary:
            print(f"Total Steps:     {summary.get('total_steps', 0)}")
            print(f"Final Train PPL: {summary.get('final_train_ppl', 0):.2f}")
            print(f"Final Val PPL:   {summary.get('final_val_ppl', 'N/A')}")
            print(f"Best Val PPL:    {summary.get('best_val_ppl', 'N/A')}")
            print(f"Final Train BPC: {summary.get('final_train_bpc', 0):.4f}")
            print(f"Final Val BPC:   {summary.get('final_val_bpc', 'N/A')}")
            print(f"Throughput:      {summary.get('avg_tokens_per_sec', 0):.0f} tok/s")
            print(f"Total Time:      {summary.get('total_time_sec', 0)/60:.1f} min")

        # Holonomy summary
        if self.holonomy_history:
            latest = self.holonomy_history[-1]
            print(f"\nHolonomy (Non-Flat Transport):")
            print(f"  Mean ‖C-I‖_F:  {latest.global_mean_norm:.4f}")
            print(f"  Max ‖C-I‖_F:   {latest.global_max_norm:.4f}")
            if latest.snapshots:
                print(f"  Layers tracked: {len(latest.snapshots)}")
                for s in latest.snapshots:
                    print(f"    L{s.layer}: mean={s.mean_norm:.4f}, "
                          f"spectral_gap={s.mean_spectral_gap:.4f}, "
                          f"wilson_dev={s.mean_wilson_trace:.4f}")
            print(f"  History points: {len(self.holonomy_history)}")

        print("=" * 60)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_results_table(results: List[ExperimentResult]) -> str:
    """Create LaTeX table from experiment results."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Experimental Results on WikiText-103}",
        r"\label{tab:results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{PPL} $\downarrow$ & \textbf{BPC} $\downarrow$ & \textbf{Params} \\",
        r"\midrule",
    ]

    for r in results:
        lines.append(
            f"{r.name} & {r.final_val_ppl:.1f} & {r.final_val_bpc:.3f} & {r.total_params/1000:.1f}K \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Publication Metrics - Demo")
    print("=" * 60)

    # Create metrics tracker
    metrics = PublicationMetrics("demo_experiment")

    # Simulate training
    np.random.seed(42)
    for step in range(500):
        loss = 8.0 * np.exp(-step / 200) + 5.5 + np.random.randn() * 0.1
        train_metrics = {
            'loss': loss,
            'ce_loss': loss - 0.1,
            'beta_loss': 0.1,
        }
        grad_norms = {
            'total': 2.0 / (step + 1) + 0.1,
            'mu': 1.0 / (step + 1) + 0.05,
            'sigma': 0.5 / (step + 1) + 0.02,
            'phi': 0.3 / (step + 1) + 0.01,
        }

        metrics.record_step(
            step=step,
            epoch=step / 100,
            train_metrics=train_metrics,
            grad_norms=grad_norms,
            lr=0.001,
            step_time=0.5,
            batch_size=32,
            seq_len=128,
        )

        if step % 50 == 0:
            val_loss = loss + 0.2 + np.random.randn() * 0.05
            metrics.record_validation(step, {'loss': val_loss, 'ce_loss': val_loss})

    # Add comparison results
    metrics.add_comparison_result(ExperimentResult(
        name="Standard Transformer",
        config={},
        final_val_ppl=350.0,
        final_val_bpc=8.45,
        best_val_ppl=340.0,
        total_params=500000,
        training_time=3600,
        tokens_per_sec=8000,
    ))
    metrics.add_comparison_result(ExperimentResult(
        name="Gauge VFE",
        config={},
        final_val_ppl=300.0,
        final_val_bpc=8.23,
        best_val_ppl=290.0,
        total_params=450000,
        training_time=5400,
        tokens_per_sec=5000,
    ))

    # Add scaling data
    for k in [7, 11, 25, 37, 63]:
        ppl = 250 + 50 * abs(k - 11) / 10 + np.random.randn() * 10
        metrics.add_scaling_point(k, ppl, k * 50257 + k * k * 100)

    # Save and generate
    metrics.save_all()
    metrics.generate_figures()
    metrics.print_summary()

    print("\n Demo complete! Check ./outputs/demo_experiment/")
