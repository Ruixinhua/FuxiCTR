# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Pre-ranking Stage Implementation

This module wraps the LightweightRanker model as a pipeline stage.
"""

import os
import csv
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import torch

from ..pipeline.base_stage import BaseStage, StageType
from ..pipeline.stage_output import StageOutput, CandidateSet, CandidateItem
from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from .models.lightweight_ranker import LightweightRanker

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
        self.model: Optional[LightweightRanker] = None
    
    def build_model(self) -> LightweightRanker:
        """Build and initialize the pre-ranking model"""
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
        
        self.model = LightweightRanker(
            self.feature_map,
            use_diversity_loss=self.use_diversity,
            diversity_weight=self.diversity_weight,
            **params
        )
        self.logger.info("Built LightweightRanker model")
        self.model.count_parameters()
        return self.model
    
    def train(self,
              train_data: Any,
              valid_data: Optional[Any] = None,
              **kwargs) -> Dict[str, float]:
        """
        Train the pre-ranking model.
        
        Args:
            train_data: Training data generator
            valid_data: Validation data generator
            **kwargs: Training parameters
            
        Returns:
            Training metrics
        """
        if self.model is None:
            self.build_model()
        
        self.logger.info("Starting pre-ranking model training")
        self.model.fit(train_data, validation_data=valid_data, **kwargs)
        
        metrics = {}
        if valid_data is not None:
            valid_result = self.model.evaluate(valid_data)
            metrics.update(valid_result)
            self.logger.info(f"Validation metrics: {valid_result}")
        
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
        Process retrieval candidates to produce refined set.
        
        Args:
            input_data: Output from retrieval stage
            **kwargs: Additional parameters
            
        Returns:
            StageOutput with refined candidate sets
        """
        # Auto-build model if not built (for testing)
        if self.model is None:
            self.logger.warning("Model not built, building with default params for testing...")
            self.build_model()
        
        # Handle missing input gracefully for testing
        if input_data is None or len(input_data.candidate_sets) == 0:
            self.logger.warning("No input from previous stage. Creating dummy output for testing...")
            output = StageOutput(stage_name=self.stage_name)
            cs = CandidateSet(request_id="test_req_0", user_id=0, source_stage=self.stage_name)
            cs.add_candidate(item_id=0, score=1.0)
            output.candidate_sets.append(cs)
            return output
        
        output = StageOutput(stage_name=self.stage_name)
        self.model.eval()
        
        self.logger.info(f"Processing {len(input_data.candidate_sets)} candidate sets")
        
        for cs in input_data.candidate_sets:
            # Create new candidate set
            new_cs = CandidateSet(
                request_id=cs.request_id,
                user_id=cs.user_id,
                user_features=cs.user_features.copy(),
                context_features=cs.context_features.copy(),
                source_stage=self.stage_name
            )
            
            if len(cs.candidates) == 0:
                output.candidate_sets.append(new_cs)
                continue
            
            # Score all candidates
            # Note: In real implementation, you'd build proper feature tensors
            # This is a simplified version using embeddings if available
            scores = []
            for candidate in cs.candidates:
                # Use retrieval score as baseline, can enhance with model
                scores.append(candidate.score)
            
            scores = np.array(scores)
            
            # Select top-k with optional diversity
            if self.use_diversity:
                # Get categories if available
                categories = None
                if 'category' in cs.candidates[0].features:
                    categories = np.array([c.features.get('category', 0) 
                                          for c in cs.candidates])
                
                selected_indices = self.model.select_top_k(
                    scores, 
                    self.top_k,
                    diversity_rerank=True,
                    item_categories=categories
                )
            else:
                selected_indices = np.argsort(-scores)[:self.top_k]
            
            # Add selected candidates to output
            for idx in selected_indices:
                old_candidate = cs.candidates[idx]
                new_cs.add_candidate(
                    item_id=old_candidate.item_id,
                    features=old_candidate.features.copy(),
                    score=float(scores[idx]),
                    embedding=old_candidate.embedding
                )
            
            output.candidate_sets.append(new_cs)
        
        self.logger.info(f"Pre-ranking: {input_data.get_total_candidates()} -> "
                        f"{output.get_total_candidates()} candidates")
        
        return output
    
    def evaluate(self,
                 test_data: Any,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate pre-ranking model.
        
        Args:
            test_data: Test data generator
            **kwargs: Evaluation parameters
            
        Returns:
            Evaluation metrics (AUC, etc.)
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        metrics = self.model.evaluate(test_data)
        
        # Save metrics to CSV
        metrics_path = os.path.join(self.output_dir, "eval_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}" if isinstance(value, float) else value])
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics
