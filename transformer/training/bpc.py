r"""
Bits-Per-Character Computation
==============================

Converts a per-token cross-entropy (in nats) to true bits-per-character of the
source text:

.. math::

    \mathrm{BPC} = \frac{\mathrm{CE}_\mathrm{nats}}{\ln 2}
                   \cdot \frac{n_\mathrm{tokens}}{n_\mathrm{chars}}

The second factor (``tokens_per_char``) is the BPE compression ratio of the
tokenizer on the dataset. Without it, the reported number is bits-per-token,
not bits-per-character — wrong by ``1 / tokens_per_char`` for any non-character
tokenizer. For GPT-2 BPE on English wikitext that's ~4×; for cl100k_base on
Japanese wikipedia that's ~10%.

Datasets that compute and store ``tokens_per_char`` (see
``transformer/data/datasets.py::WikiText2TiktokenDataset``) expose the ratio
as a property. When it is unavailable (legacy token caches without sidecar
metadata), this module falls back to bits-per-token and emits a one-time
warning per process.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

_FALLBACK_WARNED: set = set()


def tokens_per_char_from_dataset(dataset) -> Optional[float]:
    """Return ``dataset.tokens_per_char`` or unwrap from a Subset; None if absent.

    Accepts a torch ``DataLoader``, a ``Dataset``, a ``Subset``, or None. Walks
    one level of indirection (Subset.dataset, DataLoader.dataset) when needed.
    """
    if dataset is None:
        return None
    inner = getattr(dataset, 'dataset', dataset)  # DataLoader -> Dataset, Subset -> base
    inner = getattr(inner, 'dataset', inner)      # Subset of Subset, etc.
    tpc = getattr(inner, 'tokens_per_char', None)
    if tpc is None:
        return None
    try:
        return float(tpc)
    except (TypeError, ValueError):
        return None


def compute_bpc(
    ce_nats: float,
    tokens_per_char: Optional[float],
    *,
    fallback_key: Optional[str] = None,
) -> float:
    r"""Convert per-token CE in nats to bits-per-character.

    When ``tokens_per_char`` is None or non-positive, falls back to
    bits-per-token and emits a one-time warning keyed by ``fallback_key`` so
    a single training run does not spam the log.
    """
    if ce_nats is None:
        return float('nan')
    bits_per_token = float(ce_nats) / math.log(2.0)
    if tokens_per_char is None or not (tokens_per_char > 0):
        key = fallback_key or '__default__'
        if key not in _FALLBACK_WARNED:
            _FALLBACK_WARNED.add(key)
            logger.warning(
                "BPC computed without tokens_per_char correction (key=%s). "
                "Reported value is bits-per-token, not bits-per-character. "
                "Rebuild the dataset's token cache to populate the sidecar "
                "metadata or pass tokens_per_char explicitly.",
                key,
            )
        return bits_per_token
    return bits_per_token * float(tokens_per_char)


def bpc_from_dataset(ce_nats: float, dataset, *, fallback_key: Optional[str] = None) -> float:
    """Convenience: read ``tokens_per_char`` off the dataset and compute BPC."""
    return compute_bpc(
        ce_nats,
        tokens_per_char_from_dataset(dataset),
        fallback_key=fallback_key,
    )
