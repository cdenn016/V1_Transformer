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
]
