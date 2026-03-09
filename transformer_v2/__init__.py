# -*- coding: utf-8 -*-
"""
Gauge-Transformer v2 — Refactored Core
========================================

Public API for the gauge-theoretic transformer.
"""

from .config import GaugeTransformerConfig
from .model import GaugeTransformerLM
from .blocks import GaugeTransformerBlock, GaugeTransformerStack
from .attention import (
    compute_attention_weights,
    aggregate_messages,
    IrrepMultiHeadAttention,
    create_attention_mask,
)
from .kl_ops import compute_kl_matrix, compute_transport_operators
from .embeddings import GaugeTokenEmbedding, GaugePositionalEncoding
from .prior_bank import PriorBank
from .variational_ffn import VariationalFFNDynamic

__all__ = [
    # Config
    'GaugeTransformerConfig',

    # Model
    'GaugeTransformerLM',

    # Blocks
    'GaugeTransformerBlock',
    'GaugeTransformerStack',

    # Attention
    'compute_attention_weights',
    'aggregate_messages',
    'IrrepMultiHeadAttention',
    'create_attention_mask',

    # KL ops
    'compute_kl_matrix',
    'compute_transport_operators',

    # Embeddings
    'GaugeTokenEmbedding',
    'GaugePositionalEncoding',

    # Prior bank
    'PriorBank',

    # VFE FFN
    'VariationalFFNDynamic',
]
