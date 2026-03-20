"""
Analysis and Metrics Module
===========================

Tools for analyzing transformer behavior:
- RG metrics: Renormalization group flow analysis
- RG flow enhanced: Gauge-frame-aware RG coarse-graining
- Publication metrics: Comprehensive metrics for papers
- Trajectory: Belief trajectory tracking
- Semantics: Gauge semantics analysis
- Bayesian validation: PyMC models for manuscript claim validation
- Holonomy metrics: Curvature diagnostics for non-flat transport
"""

from transformer.analysis.rg_metrics import (
    compute_rg_diagnostics,
    RGDiagnostics,
    RGFlowSummary,
)
from transformer.analysis.rg_flow_enhanced import (
    FullRGDiagnostics,
    CoarseGrainedState,
    HierarchicalRGState,
    compute_full_rg_diagnostics,
    build_hierarchical_rg_state,
    summarize_hierarchical_rg,
)
from transformer.analysis.trajectory import (
    TrajectoryRecorder,
    LayerTrajectory,
    ForwardTrajectory,
)
from transformer.analysis.holonomy import (
    compute_holonomy,
    holonomy_penalty_loss,
    holonomy_statistics,
    holonomy_by_token_pairs,
)
from transformer.analysis.holonomy_metrics import (
    HolonomySnapshot,
    HolonomyProfile,
    compute_holonomy_snapshot,
    compute_curvature_by_distance,
    compute_flatness_trajectory,
)

__all__ = [
    # RG metrics
    'compute_rg_diagnostics',
    'RGDiagnostics',
    'RGFlowSummary',

    # Enhanced RG (gauge-aware)
    'FullRGDiagnostics',
    'CoarseGrainedState',
    'HierarchicalRGState',
    'compute_full_rg_diagnostics',
    'build_hierarchical_rg_state',
    'summarize_hierarchical_rg',

    # Trajectory tracking
    'TrajectoryRecorder',
    'LayerTrajectory',
    'ForwardTrajectory',

    # Holonomy (curvature diagnostics)
    'compute_holonomy',
    'holonomy_penalty_loss',
    'holonomy_statistics',
    'holonomy_by_token_pairs',
    'HolonomySnapshot',
    'HolonomyProfile',
    'compute_holonomy_snapshot',
    'compute_curvature_by_distance',
    'compute_flatness_trajectory',
]
