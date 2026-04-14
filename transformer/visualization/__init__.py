"""
Visualization Module
====================

Plotting and visualization tools for the gauge-theoretic transformer.

- Training plots: VFE loss curves, free energy components, learning rate schedules
- Ablation plots: Gauge VFE vs standard transformer comparisons
- Trajectory plots: Belief (mu, sigma, phi) evolution through layers
- Attention visualization: KL-divergence-based attention pattern analysis
- Belief space visualization: Token embedding structure in belief space (mu, sigma, phi)
- Interactive visualization: UMAP + Plotly 3D + SHAP attribution (interactive_belief_viz)
"""

from .pub_style import set_pub_style, PUB_COLORS, PUB_CYCLE
from .training_plots import load_metrics_csv, plot_head_kappas
from .vfe_dynamics_plots import load_vfe_metrics, generate_all_vfe_figures
from .fiber_plots import generate_all_fiber_figures, plot_arc_length_heatmap
from .holonomy_plots import (
    plot_holonomy_distribution,
    plot_holonomy_evolution,
    plot_holonomy_summary,
    plot_wilson_spectrum,
    plot_curvature_vs_distance,
    plot_layer_holonomy_profile,
    plot_flat_vs_nonflat_comparison,
)
from .gauge_geometry_plots import (
    generate_all_gauge_geometry_figures,
    plot_yang_mills_evolution,
    plot_gauge_invariant_scatter,
    plot_gauge_orbit_pca,
)
from .belief_space_viz import visualize_belief_space

__all__ = [
    # pub_style
    "set_pub_style",
    "PUB_COLORS",
    "PUB_CYCLE",
    # training_plots
    "load_metrics_csv",
    "plot_head_kappas",
    # vfe_dynamics_plots
    "load_vfe_metrics",
    "generate_all_vfe_figures",
    # fiber_plots
    "generate_all_fiber_figures",
    "plot_arc_length_heatmap",
    # holonomy_plots
    "plot_holonomy_distribution",
    "plot_holonomy_evolution",
    "plot_holonomy_summary",
    "plot_wilson_spectrum",
    "plot_curvature_vs_distance",
    "plot_layer_holonomy_profile",
    "plot_flat_vs_nonflat_comparison",
    # gauge_geometry_plots
    "generate_all_gauge_geometry_figures",
    "plot_yang_mills_evolution",
    "plot_gauge_invariant_scatter",
    "plot_gauge_orbit_pca",
    # belief_space_viz
    "visualize_belief_space",
]
