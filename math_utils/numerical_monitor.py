"""
Lightweight counter for numerical fallback events during training.

Usage:
    from math_utils.numerical_monitor import record, flush

    record("chol_recover")   # increment counter
    events = flush()         # get counts and reset
"""
from typing import Dict

_counts: Dict[str, int] = {}


def record(event: str, count: int = 1) -> None:
    """Increment counter for a numerical fallback event.

    Args:
        event: Event name (e.g., 'chol_recover', 'nan_replace')
        count: Number of occurrences to add (default 1). Use for
               element-level counting: ``record('neg_clamp', n=neg_mask.sum().item())``
    """
    _counts[event] = _counts.get(event, 0) + count


def flush() -> Dict[str, int]:
    """Return current counts and reset all counters."""
    result = dict(_counts)
    _counts.clear()
    return result
