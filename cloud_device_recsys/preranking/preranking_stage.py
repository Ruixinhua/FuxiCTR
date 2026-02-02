# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Pre-ranking Stage Implementation

This module wraps the LightweightRanker model as a pipeline stage.
"""

import os
import sys
import csv
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
import torch
from tqdm import tqdm # Added tqdm import

from ..pipeline.base_stage import BaseStage, StageType
from ..pipeline.stage_output import StageOutput, CandidateSet, CandidateItem
from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from .models.din_ranker import DINRanker

from fuxictr.features import FeatureMap


class PrerankingStage(BaseStage):
    """
    Pre-ranking stage for efficient candidate scoring.
    
    Takes candidates from retrieval and produces a refined set.
    Only uses FG1 (non-personalized) and FG2 (cloud-personalized) features.
    """
    
    def __init__(self,
                 feature_map: FeatureMap,
                 feature_group_manager: FeatureGroupManager,
                 model_params: Dict[str, Any],
                 output_dir: str = "./outputs/preranking",
                 top_k: int = 100,
                 use_diversity: bool = False,
                 diversity_weight: float = 0.1,
                 **kwargs):
        """
        Initialize pre-ranking stage.
        
        Args:
            feature_map: FuxiCTR FeatureMap
            feature_group_manager: Feature group manager
            model_params: Parameters for LightweightRanker model
            output_dir: Output directory
            top_k: Number of candidates to pass to next stage
            use_diversity: Whether to apply diversity in selection
            diversity_weight: Weight for diversity consideration
        """
        super().__init__(
            stage_name="preranking",
            stage_type=StageType.PRERANKING,
            feature_group_manager=feature_group_manager,
            allowed_feature_groups=[FeatureGroup.FG1, FeatureGroup.FG2],
            output_dir=output_dir,
            **kwargs
        )
        
        self.feature_map = feature_map
        self.top_k = top_k
        self.use_diversity = use_diversity
        self.diversity_weight = diversity_weight
        self.model_params = model_params
        self.metrics_k = model_params['metrics_k']
        self.model: Optional[DINRanker] = None
        self.best_weights_path = None
        # Item features storage for lookups during evaluation/processing
        self.item_features_df = None
    
    def load_item_features(self, item_pool_path: str):
        """Load item features from parquet file for inference lookup"""
        import pandas as pd
        self.logger.info(f"Loading item features from {item_pool_path}...")
        try:
            self.item_features_df = pd.read_parquet(item_pool_path)
            # Create index on item_id for faster lookup
            item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
            if item_id_col in self.item_features_df.columns:
                self.item_features_df = self.item_features_df.set_index(item_id_col)
            self.logger.info(f"Loaded {len(self.item_features_df)} items into feature memory.")
        except Exception as e:
            self.logger.error(f"Failed to load item features: {e}")
            raise
    
    def build_model(self):
        """Build and initialize the pre-ranking model"""
        # Ensure output directories exist
        model_dir = os.path.join(self.output_dir, self.feature_map.dataset_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Add default FuxiCTR required parameters
        default_params = {
            'verbose': 1,
            'model_root': self.output_dir,
            'metrics': ['AUC', 'logloss'],
            'embedding_dim': 32,
            'gpu': -1,
            'optimizer': 'adam',
            'loss': 'binary_crossentropy',
        }
        params = {**default_params, **self.model_params}
        
        self.model = DINRanker(
            self.feature_map,
            **params
        )
        self.logger.info(f"Built DINRanker model, saving to {model_dir}")
        return self.model
    
    def train(self,
              train_data: Any,
              valid_data: Optional[Any] = None,
              **kwargs) -> Dict[str, float]:
        """
        Train the pre-ranking model with custom loop.
        
        Args:
            train_data: Training data generator
            valid_data: Validation data generator (used for internal training)
            **kwargs: Training parameters
            
        Returns:
            Training metrics
        """
        if self.model is None:
            self.build_model()
        
        self.logger.info("Starting pre-ranking model training (Custom Loop)")
        self.best_weights_path = os.path.join(self.model.model_dir, self.model.model_id + ".model")

        epochs = kwargs.get('epochs', 1)
        metrics = {}
        
        # Ensure optimizer is initialized
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
             self.logger.info("Initializing optimizer...")
             # FuxiCTR's compile handles optimizer/loss setup
             self.model.compile(
                 optimizer=kwargs.get("optimizer", self.model_params.get("optimizer", "adam")),
                 loss=self.model_params.get("loss", "binary_crossentropy"),
                 lr=kwargs.get("learning_rate", 1e-3)
             )
             
        # Setup model for manual training (required by train_step)
        self.model._total_steps = 0
        self.model._stop_training = False
        self.model._max_gradient_norm = kwargs.get("max_gradient_norm", 10.0)
        self.model._verbose = kwargs.get("verbose", 1)
        self.model._epoch_index = 0

        for epoch in range(epochs):
            self.model._epoch_index = epoch
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Manual Training Loop
            self.model.train()
            total_loss = 0.0
            steps = 0
            
            # Iterate over train_data directly
            for batch_data in train_data:
                loss = self.model.train_step(batch_data)
                
                total_loss += loss.item()
                steps += 1
            
            avg_loss = total_loss / steps if steps > 0 else 0.0
            self.logger.info(f"Train Loss: {avg_loss:.6f}")
            
            # Validation (Ranking Metrics)
            self.logger.info(f"Evaluating epoch {epoch + 1}...")
            valid_metrics = self.evaluate(valid_data)
            metrics.update(valid_metrics)
            self.logger.info(f"Validation (Ranking): {valid_metrics}")

        self.model.save_weights(self.best_weights_path)
        self.logger.info(f"Saved model checkpoint to {self.best_weights_path}")

        # Save metrics to CSV
        metrics_path = os.path.join(self.output_dir, "training_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}" if isinstance(value, float) else value])
        
        return metrics

    def process(self,
                input_data: StageOutput,
                **kwargs) -> StageOutput:
        """
        Process candidates from retrieval stage - scores and selects top-K.
        
        Args:
            input_data: StageOutput from previous stage containing candidate sets
            **kwargs: Additional parameters (e.g., top_k override)

        Returns:
            StageOutput with Top-K candidates.
        """
        from ..utils import process_and_rank_candidates

        if os.path.exists(self.best_weights_path):
            self.model.load_weights(self.best_weights_path)
            self.logger.info(f"Loaded best weights from {self.best_weights_path}")
        else:
            self.logger.warning(f"No best weights found at {self.best_weights_path}. Using current model state.")
        
        # Ensure item features are loaded
        if self.item_features_df is None:
            raise ValueError("Item features not loaded. Call load_item_features() first.")

        return process_and_rank_candidates(
            model=self.model,
            feature_map=self.feature_map,
            input_data=input_data,
            item_features_df=self.item_features_df,
            stage_name=self.stage_name,
            mode='process',
            top_k=kwargs.get('top_k', self.top_k),
            logger=self.logger,
            **kwargs
        )

    def evaluate(self,
                 input_data: StageOutput,
                 metrics_k: List[int] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate pre-ranking model with list-wise metrics.
        
        Args:
            input_data: StageOutput containing candidate sets with labels
            metrics_k: List of K values for Recall@K and nDCG@K
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        from ..utils import process_and_rank_candidates
                 
        if self.item_features_df is None:
            raise ValueError("Item features not loaded. Call load_item_features() first.")

        if metrics_k is None:
            metrics_k = self.metrics_k

        return process_and_rank_candidates(
            model=self.model,
            feature_map=self.feature_map,
            input_data=input_data,
            item_features_df=self.item_features_df,
            stage_name=self.stage_name,
            mode='evaluate',
            metrics_k=metrics_k,
            logger=self.logger,
            **kwargs
        )
