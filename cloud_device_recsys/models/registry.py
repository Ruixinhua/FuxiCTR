# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
Model Registry

Provides a centralized registry for all pipeline models and a unified
build_model function that instantiates models based on configuration.
"""

import datetime
import logging
from typing import Dict, Any, Type, Optional

from fuxictr.features import FeatureMap

logger = logging.getLogger(__name__)


def _lazy_import_models():
    """Lazy import to avoid circular dependencies."""
    from .dual_tower_retrieval import DualTowerRetrieval
    from .din_ranker import DINRanker
    from .lightweight_ranker import LightweightRanker
    from .device_reranker import DeviceReranker
    
    return {
        # Retrieval models
        "DualTowerRetrieval": DualTowerRetrieval,
        
        # Preranking models
        "DINRanker": DINRanker,
        "LightweightRanker": LightweightRanker,
        
        # Reranking models
        "DeviceReranker": DeviceReranker,
    }


# Global registry - populated lazily
MODEL_REGISTRY: Dict[str, Type] = {}


def get_available_models() -> list:
    """Get list of available model names."""
    if not MODEL_REGISTRY:
        MODEL_REGISTRY.update(_lazy_import_models())
    return list(MODEL_REGISTRY.keys())


def build_model(
    model_name: str,
    feature_map: FeatureMap,
    model_params: Dict[str, Any],
    output_dir: Optional[str] = None,
    add_timestamp: bool = True,
    **kwargs
):
    """
    Build a model instance from the registry.
    
    Args:
        model_name: Name of the model (must be in MODEL_REGISTRY)
        feature_map: FuxiCTR FeatureMap instance
        model_params: Model-specific parameters
        output_dir: Output directory for model checkpoints
        add_timestamp: Whether to add timestamp to model_id
        **kwargs: Additional parameters passed to model constructor
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_name is not in registry
    """
    # Ensure registry is populated
    if not MODEL_REGISTRY:
        MODEL_REGISTRY.update(_lazy_import_models())
    
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. Available models: {available}"
        )
    
    model_cls = MODEL_REGISTRY[model_name]
    
    # Prepare default parameters
    default_params = {
        'verbose': 1,
        'metrics': ['AUC', 'logloss'],
        'gpu': -1,
        'optimizer': 'adam',
        'loss': 'binary_crossentropy',
    }
    
    # Add model_root if output_dir provided
    if output_dir:
        default_params['model_root'] = output_dir
    
    # Merge: defaults < model_params < kwargs
    params = {**default_params, **model_params, **kwargs}
    
    # Remove 'model' key if present (it's not a model parameter)
    params.pop('model', None)
    
    # Add unique timestamp to model_id to prevent overwrites
    if add_timestamp:
        base_model_id = params.get("model_id", model_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        params["model_id"] = f"{base_model_id}_{timestamp}"
    
    logger.info(f"Building model: {model_name}")
    model = model_cls(feature_map, **params)
    
    # Log parameter count if available
    if hasattr(model, 'count_parameters'):
        model.count_parameters()
    
    return model
