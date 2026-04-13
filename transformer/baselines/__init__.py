"""
Baseline Models
===============

Standard transformer implementations for comparison,
including intermediate baselines for peer review ablation studies.
"""

from transformer.baselines.standard_transformer import StandardTransformerLM
from transformer.baselines.flops_counter import (
    count_standard_transformer_flops,
    count_gauge_transformer_flops,
    format_flops,
    compare_flops,
    print_flops_comparison,
)

__all__ = [
    'StandardTransformerLM',
    'count_standard_transformer_flops',
    'count_gauge_transformer_flops',
    'format_flops',
    'compare_flops',
    'print_flops_comparison',
]
