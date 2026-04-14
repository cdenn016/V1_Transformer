"""
Analysis and Metrics Module
===========================

Tools for analyzing transformer behavior:
- Publication metrics: Comprehensive metrics for papers
- Trajectory: Belief trajectory tracking
- Semantics: Gauge semantics analysis
- Bayesian validation: PyMC models for manuscript claim validation
- Holonomy metrics: Curvature diagnostics for non-flat transport
- Gauge geometry: Yang-Mills energy, curvature tensor, and gauge orbits
"""

from transformer.analysis.trajectory import (
    TrajectoryRecorder,
    LayerTrajectory,
    ForwardTrajectory,
)
from transformer.analysis.semantics import (
    analyze_gauge_semantics,
    analyze_omega_semantics,
    analyze_sigma_semantics,
    analyze_holonomy_semantic_correlation,
    compute_semantic_field_coherence,
    SemanticTrajectoryTracker,
    SEMANTIC_FIELDS,
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
from transformer.analysis.gauge_geometry import (
    extract_curvature_tensor,
    compute_yang_mills_energy,
    compute_yang_mills_energy_from_holonomy,
    compute_gauge_field_energy,
    decompose_curvature,
    gauge_fix_axial,
    gauge_fix_coulomb,
    compute_gauge_invariants,
    gauge_orbit_sample,
    compute_gauge_orbit_dimension,
)

__all__ = [
    # Trajectory tracking
    'TrajectoryRecorder',
    'LayerTrajectory',
    'ForwardTrajectory',

    # Semantics
    'analyze_gauge_semantics',
    'analyze_omega_semantics',
    'analyze_sigma_semantics',
    'analyze_holonomy_semantic_correlation',
    'compute_semantic_field_coherence',
    'SemanticTrajectoryTracker',
    'SEMANTIC_FIELDS',

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

    # Gauge geometry (curvature, Yang-Mills, gauge orbits)
    'extract_curvature_tensor',
    'compute_yang_mills_energy',
    'compute_yang_mills_energy_from_holonomy',
    'compute_gauge_field_energy',
    'decompose_curvature',
    'gauge_fix_axial',
    'gauge_fix_coulomb',
    'compute_gauge_invariants',
    'gauge_orbit_sample',
    'compute_gauge_orbit_dimension',
]
