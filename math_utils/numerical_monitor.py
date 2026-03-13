"""
Lightweight counter for numerical fallback events during training.

Usage:
    from math_utils.numerical_monitor import record, flush

    record("chol_recover")   # increment counter
    events = flush()         # get counts and reset
"""
from typing import Dict

_counts: Dict[str, int] = {}


def record(event: str) -> None:
    """Increment counter for a numerical fallback event."""
    _counts[event] = _counts.get(event, 0) + 1


def flush() -> Dict[str, int]:
    """Return current counts and reset all counters."""
    result = dict(_counts)
    _counts.clear()
    return result
