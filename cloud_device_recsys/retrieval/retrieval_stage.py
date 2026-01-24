# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Retrieval Stage Implementation

This module wraps the DualTowerRetrieval model as a pipeline stage.
"""

import os
import csv
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Set
import logging
import torch

from ..pipeline.base_stage import BaseStage, StageType
from ..pipeline.stage_output import StageOutput, CandidateSet, CandidateItem
from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from .models.dual_tower_retrieval import DualTowerRetrieval, RetrievalMetrics

from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader


class RetrievalStage(BaseStage):
    """
    Retrieval stage for candidate generation.
    
    Uses dual-tower model to retrieve top-K candidates from item pool.
    Only uses FG1 (non-personalized) and FG2 (cloud-personalized) features.
    """
    
    def __init__(self,
                 feature_map: FeatureMap,
                 feature_group_manager: FeatureGroupManager,
                 model_params: Dict[str, Any],
                 output_dir: str = "./outputs/retrieval",
                 top_k: int = 1000,
                 **kwargs):
        """
        Initialize retrieval stage.
        
        Args:
            feature_map: FuxiCTR FeatureMap
            feature_group_manager: Feature group manager
            model_params: Parameters for DualTowerRetrieval model
            output_dir: Output directory
            top_k: Number of candidates to retrieve
        """
        super().__init__(
            stage_name="retrieval",
            stage_type=StageType.RETRIEVAL,
            feature_group_manager=feature_group_manager,
            allowed_feature_groups=[FeatureGroup.FG1, FeatureGroup.FG2],
            output_dir=output_dir,
            **kwargs
        )
        
        self.feature_map = feature_map
        self.top_k = top_k
        self.model_params = model_params
        self.model: Optional[DualTowerRetrieval] = None
        
        # Item index for retrieval
        self.item_embeddings: Optional[np.ndarray] = None
        self.item_ids: Optional[List[Any]] = None
    
    def build_model(self) -> DualTowerRetrieval:
        """Build and initialize the retrieval model"""
        # Add default FuxiCTR required parameters
        default_params = {
            'verbose': 1,
            'model_root': self.output_dir,
            'metrics': ['AUC', 'logloss'],
            'embedding_dim': 64,
            'gpu': -1,
            'optimizer': 'adam',
            'loss': 'binary_crossentropy',
        }
        # Merge with provided params (provided params take precedence)
        params = {**default_params, **self.model_params}
        
        self.model = DualTowerRetrieval(
            self.feature_map,
            **params
        )
        self.logger.info("Built DualTowerRetrieval model")
        self.model.count_parameters()
        return self.model
    
    def train(self,
              train_data: Any,
              valid_data: Optional[Any] = None,
              **kwargs) -> Dict[str, float]:
        """
        Train the retrieval model.
        
        Args:
            train_data: Training data generator
            valid_data: Validation data generator
            **kwargs: Training parameters (epochs, etc.)
            
        Returns:
            Training metrics
        """
        if self.model is None:
            self.build_model()
        
        self.logger.info("Starting retrieval model training")
        
        # Use FuxiCTR's fit method
        self.model.fit(train_data, validation_data=valid_data, **kwargs)
        
        # Evaluate on validation
        metrics = {}
        if valid_data is not None:
            valid_result = self.model.evaluate(valid_data)
            metrics.update(valid_result)
            self.logger.info(f"Validation metrics: {valid_result}")
        
        # Save training metrics to CSV
        metrics_path = os.path.join(self.output_dir, "training_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}" if isinstance(value, float) else value])
        
        return metrics
    
    def build_item_index(self, item_data: Any) -> None:
        """
        Build item embedding index for retrieval.
        
        Args:
            item_data: Data generator for items
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() or train() first.")
        
        self.logger.info("Building item embedding index")
        
        self.model.eval()
        item_embeddings = []
        item_ids = []
        
        with torch.no_grad():
            for batch_data in item_data:
                # Get item embeddings
                embeddings = self.model.get_item_embedding(batch_data)
                item_embeddings.append(embeddings.cpu().numpy())
                
                # Extract item IDs (assuming first column or specific field)
                # This may need customization based on data format
                batch_size = embeddings.shape[0]
                # Placeholder: use batch index if item_id not available
                item_ids.extend(range(len(item_ids), len(item_ids) + batch_size))
        
        self.item_embeddings = np.vstack(item_embeddings)
        self.item_ids = item_ids
        
        self.logger.info(f"Built index with {len(self.item_ids)} items, "
                        f"embedding dim: {self.item_embeddings.shape[1]}")
        
        # Save item embeddings to CSV
        embeddings_path = os.path.join(self.output_dir, "item_embeddings.csv")
        np.savetxt(embeddings_path, self.item_embeddings, delimiter=',')
        
        item_ids_path = os.path.join(self.output_dir, "item_ids.csv")
        with open(item_ids_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['item_id'])
            for item_id in self.item_ids:
                writer.writerow([item_id])
    
    def process(self,
                input_data: Optional[StageOutput] = None,
                user_data: Any = None,
                **kwargs) -> StageOutput:
        """
        Retrieve candidates for users.
        
        Args:
            input_data: Not used for retrieval (first stage)
            user_data: User data generator or features
            **kwargs: Additional parameters
            
        Returns:
            StageOutput with candidate sets for each user
        """
        # Auto-build model if not built (for testing/validation)
        if self.model is None:
            self.logger.warning("Model not built, building with default params for testing...")
            self.build_model()
        
        # If no item index, create a dummy output for testing
        if self.item_embeddings is None:
            self.logger.warning("Item index not built. Creating dummy output for pipeline testing...")
            output = StageOutput(stage_name=self.stage_name)
            # Create a dummy candidate set to verify pipeline flow
            cs = CandidateSet(
                request_id="test_req_0",
                user_id=0,
                source_stage=self.stage_name
            )
            cs.add_candidate(item_id=0, score=1.0)
            output.candidate_sets.append(cs)
            self.logger.info("Created dummy output for pipeline verification")
            return output
        
        output = StageOutput(stage_name=self.stage_name)
        
        self.model.eval()
        request_id = 0
        
        with torch.no_grad():
            for batch_data in user_data:
                # Get user embeddings
                user_embeddings = self.model.get_user_embedding(batch_data)
                user_embeddings = user_embeddings.cpu().numpy()
                
                # Retrieve top-k for each user
                indices, scores = self.model.retrieve_top_k(
                    user_embeddings, 
                    self.item_embeddings, 
                    top_k=self.top_k
                )
                
                # Create candidate sets
                batch_size = user_embeddings.shape[0]
                for i in range(batch_size):
                    cs = CandidateSet(
                        request_id=f"req_{request_id}",
                        user_id=request_id,
                        source_stage=self.stage_name
                    )
                    
                    for j in range(self.top_k):
                        item_idx = indices[i, j]
                        cs.add_candidate(
                            item_id=self.item_ids[item_idx],
                            score=float(scores[i, j]),
                            embedding=self.item_embeddings[item_idx]
                        )
                    
                    output.candidate_sets.append(cs)
                    request_id += 1
        
        self.logger.info(f"Retrieved candidates for {request_id} users")
        return output
    
    def evaluate(self,
                 test_data: Any,
                 ground_truth: Optional[np.ndarray] = None,
                 k_values: List[int] = [10, 50, 100, 500, 1000],
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            test_data: Test data generator
            ground_truth: Ground truth item indices for each query
            k_values: K values for Recall@K and HitRate@K
            
        Returns:
            Evaluation metrics
        """
        if self.model is None or self.item_embeddings is None:
            raise ValueError("Model and item index must be built before evaluation")
        
        self.model.eval()
        all_user_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in test_data:
                user_embeddings = self.model.get_user_embedding(batch_data)
                all_user_embeddings.append(user_embeddings.cpu().numpy())
                
                labels = self.model.get_labels(batch_data)
                if labels is not None:
                    all_labels.append(labels.cpu().numpy())
        
        user_embeddings = np.vstack(all_user_embeddings)
        
        # Retrieve for all users
        indices, scores = self.model.retrieve_top_k(
            user_embeddings, 
            self.item_embeddings, 
            top_k=max(k_values)
        )
        
        metrics = {}
        
        # If ground truth provided, compute retrieval metrics
        if ground_truth is not None:
            for k in k_values:
                recall_k = RetrievalMetrics.recall_at_k(indices, ground_truth, k)
                hit_rate_k = RetrievalMetrics.hit_rate_at_k(indices, ground_truth, k)
                metrics[f'Recall@{k}'] = recall_k
                metrics[f'HitRate@{k}'] = hit_rate_k
        
        # Save evaluation metrics to CSV
        metrics_path = os.path.join(self.output_dir, "eval_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}" if isinstance(value, float) else value])
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics
