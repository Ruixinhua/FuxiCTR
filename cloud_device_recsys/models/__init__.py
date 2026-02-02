# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
Unified Models Module

This module provides a centralized model registry and build_model function
for all pipeline stages (retrieval, preranking, reranking).
"""

from .registry import MODEL_REGISTRY, build_model, get_available_models

# Re-export individual model classes for convenience
from .dual_tower_retrieval import DualTowerRetrieval
from .din_ranker import DINRanker
from .lightweight_ranker import LightweightRanker
from .device_reranker import DeviceReranker

__all__ = [
    # Registry functions
    'MODEL_REGISTRY',
    'build_model',
    'get_available_models',
    # Model classes
    'DualTowerRetrieval',
    'DINRanker',
    'LightweightRanker',
    'DeviceReranker',
]
