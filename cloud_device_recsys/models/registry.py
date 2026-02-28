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
from .losses import wrap_model_with_diversity

logger = logging.getLogger(__name__)


def _lazy_import_models():
    """Lazy import to avoid circular dependencies."""
    from .dual_tower_retrieval import DualTowerRetrieval
    from .din_ranker import DINRanker
    from .device_reranker import DeviceReranker
    
    return {
        # Retrieval models
        "DualTowerRetrieval": DualTowerRetrieval,
        
        # Preranking models
        "DINRanker": DINRanker,

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
    model_cls = None
    
    # 1. Try to load from model_zoo first
    try:
        import model_zoo
        if hasattr(model_zoo, model_name):
            # model_zoo.<ModelName>.src.<ModelName>
            # But based on __init__.py import, it seems model_zoo exports the class directly in some cases
            # or the module. Let's inspect model_zoo imports.
            # From model_zoo/__init__.py: from .DIN.src import DIN
            # So model_zoo.DIN is the class.
            model_cls = getattr(model_zoo, model_name)
            logger.info(f"Found model '{model_name}' in model_zoo.")
    except ImportError:
        pass
        
    # 2. If not in model_zoo, try local registry
    if model_cls is None:
        if not MODEL_REGISTRY:
            MODEL_REGISTRY.update(_lazy_import_models())
            
        if model_name in MODEL_REGISTRY:
            model_cls = MODEL_REGISTRY[model_name]
            logger.info(f"Found model '{model_name}' in local registry.")

    if model_cls is None:
        available = list(MODEL_REGISTRY.keys())
        if 'model_zoo' in locals():
            available += [m for m in dir(model_zoo) if not m.startswith('__')]
        raise ValueError(
            f"Unknown model: '{model_name}'."
        )
    
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
    model_params = params
    
    # Remove 'model' key if present (it's not a model parameter)
    params.pop('model', None)
    
    # Extract diversity loss params before model construction
    # (model_zoo models don't accept these in __init__)
    use_diversity_loss = params.pop('use_diversity_loss', False)
    diversity_lambda = params.pop('diversity_lambda', 0.7)
    diversity_theta = params.pop('diversity_theta', 0.7)
    diversity_item_features = params.pop('diversity_item_features', None)
    
    # Add unique timestamp to model_id to prevent overwrites
    if add_timestamp:
        base_model_id = params.get("model_id", model_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        params["model_id"] = f"{base_model_id}_{timestamp}"
    
    logger.info(f"Building model: {model_name}")
    model = model_cls(feature_map, **params)
    
    # Apply vocabulary pruning if prune_info is available on the feature_map
    vocab_prune_info = getattr(feature_map, '_vocab_prune_info', None)
    if vocab_prune_info and vocab_prune_info.features:
        from .compact_embedding import apply_vocab_pruning_to_model
        logger.info(f"Applying vocabulary pruning to {model_name}...")
        apply_vocab_pruning_to_model(model, vocab_prune_info, feature_map)
    
    # Apply diversity loss wrapper if requested
    if use_diversity_loss:
        model = wrap_model_with_diversity(
            model,
            use_diversity_loss=True,
            diversity_lambda=diversity_lambda,
            diversity_theta=diversity_theta,
            diversity_item_features=diversity_item_features,
        )
    
    # Log parameter count if available
    if hasattr(model, 'count_parameters'):
        model.count_parameters()
    
    return model
