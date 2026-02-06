# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Pre-ranking Stage Implementation

This module wraps the preranking model as a pipeline stage.
"""

import os
import csv
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from ..pipeline.base_stage import BaseStage, StageType
from ..pipeline.stage_output import StageOutput
from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from ..models import build_model as registry_build_model
from ..models import DINRanker  # For type hints
from ..models.losses import bpr_loss, margin_ranking_loss, softmax_cross_entropy_loss
from ..data.negative_sampler import NegativeSampler
from ..utils import filter_feature_map

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
            model_params: Parameters for preranking model
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
        
        # Filter feature_map to only include allowed features (FG1, FG2)
        self.feature_map = filter_feature_map(feature_map, feature_group_manager, self.allowed_feature_groups,
                                              use_feature_encoder=model_params.get("use_feature_encoder", False))
        self.feature_map.default_emb_dim = model_params['embedding_dim']
        self.top_k = top_k
        self.use_diversity = use_diversity
        self.diversity_weight = diversity_weight
        self.model_params = model_params
        self.metrics_k = model_params['metrics_k']
        self.monitor = model_params.get('monitor', 'Recall@100')
        self.model: Optional[DINRanker] = None
        self.best_weights_path = None
        # Negative sampling parameters
        self.num_negatives = model_params.get('num_negatives', 4)
        self.loss_type = model_params.get('loss_type', 'bpr')  # 'bpr', 'margin', 'softmax'
        self.margin = model_params.get('margin', 1.0)
        self.negative_sampler: Optional[NegativeSampler] = None
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
        """Build and initialize the pre-ranking model using unified registry"""
        # Ensure output directories exist
        model_dir = os.path.join(self.output_dir, self.feature_map.dataset_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Get model name from config, default to DINRanker
        model_name = self.model_params.get('model', 'DINRanker')
        
        self.model = registry_build_model(
            model_name=model_name,
            feature_map=self.feature_map,
            model_params=self.model_params,
            output_dir=self.output_dir,
        )
        self.logger.info(f"Built {model_name} model, saving to {model_dir}")
        return self.model
    
    def train(self,
              train_data: Any,
              valid_data: Optional[Any] = None,
              **kwargs) -> Dict[str, float]:
        """
        Train the pre-ranking model with custom loop and best model monitoring.
        
        Args:
            train_data: Training data generator (positive examples only for negative sampling)
            valid_data: Validation data generator (used for internal training)
            **kwargs: Training parameters including:
                - epochs: Number of training epochs
                - patience: Early stopping patience
                - mode: 'max' or 'min' for monitor metric
                - reduce_lr_on_plateau: Whether to decay LR on no improvement
                - lr_decay_factor: LR decay factor (default 0.1)
            
        Returns:
            Training metrics
        """
        import torch
        
        if self.model is None:
            self.build_model()
        
        self.logger.info("Starting pre-ranking model training (Custom Loop)")
        self.best_weights_path = os.path.join(self.model.model_dir, self.model.model_id + ".model")

        epochs = kwargs.get('epochs', 1)
        patience = kwargs.get('patience', 2)
        mode = kwargs.get('mode', 'max')
        metrics = {}
        
        # Initialize negative sampler if using negative sampling
        use_negative_sampling = self.num_negatives > 0 and self.item_features_df is not None
        if use_negative_sampling:
            item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
            self.negative_sampler = NegativeSampler(self.item_features_df, item_id_col=item_id_col)
            self.logger.info(f"Negative Sampling: {self.num_negatives} negatives per positive, loss_type={self.loss_type}")
        else:
            if self.num_negatives > 0 and self.item_features_df is None:
                self.logger.warning("num_negatives > 0 but item_features_df not loaded. Using standard training.")
        
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

        self.logger.info(f"Start Training: epochs={epochs}, monitor={self.monitor}, patience={patience}")
        
        best_metric = -np.inf if mode == "max" else np.inf
        stopping_steps = 0
        
        item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')

        for epoch in range(epochs):
            self.model._epoch_index = epoch
            self.logger.info(f"*** Epoch {epoch + 1}/{epochs} ***")
            
            # Training Loop
            self.model.train()
            total_loss = 0.0
            steps = 0
            
            for batch_data in train_data:
                if use_negative_sampling:
                    # Negative sampling training
                    loss = self._train_step_with_negatives(batch_data, item_id_col, torch)
                else:
                    # Standard training
                    loss = self.model.train_step(batch_data)
                
                total_loss += loss.item()
                steps += 1
            
            avg_loss = total_loss / steps if steps > 0 else 0.0
            if use_negative_sampling:
                self.logger.info(f"Train Loss ({self.loss_type}): {avg_loss:.6f}")
            else:
                self.logger.info(f"Train Loss: {avg_loss:.6f}")
            
            # Validation (Ranking Metrics)
            self.logger.info(f"Evaluating epoch {epoch + 1}...")
            valid_metrics = self.evaluate(valid_data)
            metrics.update(valid_metrics)
            self.logger.info(f"Validation (Ranking): {valid_metrics}")
            
            # Monitor-based best model saving
            curr_val = valid_metrics.get(self.monitor, 0.0)
            is_best = (curr_val > best_metric) if mode == "max" else (curr_val < best_metric)
            
            if is_best:
                best_metric = curr_val
                stopping_steps = 0
                self.model.save_weights(self.best_weights_path)
                self.logger.info(f"New Best {self.monitor}={curr_val:.6f}! Model Saved.")
            else:
                stopping_steps += 1
                self.logger.info(f"No improvement. Patience {stopping_steps}/{patience}")
                
                # Decay LR on plateau
                if kwargs.get("reduce_lr_on_plateau", True):
                    old_lr = self.model.optimizer.param_groups[0]['lr']
                    new_lr = self.model.lr_decay(factor=kwargs.get("lr_decay_factor", 0.1))
                    self.logger.info(f"Decay LR: {old_lr:.6f} -> {new_lr:.6f}")
                
                if stopping_steps >= patience:
                    self.logger.info("Early Stopping.")
                    break

        # Restore best weights
        if os.path.exists(self.best_weights_path):
            self.model.load_weights(self.best_weights_path)
            self.logger.info(f"Restored best weights from {self.best_weights_path}")

        # Save metrics to CSV
        metrics_path = os.path.join(self.output_dir, "valid_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}" if isinstance(value, float) else value])
        
        return metrics
    
    def _train_step_with_negatives(self, batch_data, item_id_col: str, torch):
        """
        Single training step with negative sampling for pairwise ranking.
        
        Args:
            batch_data: Positive example batch
            item_id_col: Name of item ID column
            torch: Torch module reference
            
        Returns:
            Loss tensor
        """
        batch_dict = dict(batch_data)
        
        # Get positive item IDs
        pos_item_ids = batch_dict[item_id_col].cpu().numpy()
        
        # Sample negative items for each positive
        neg_item_ids = self.negative_sampler.sample_negatives_batch(
            pos_item_ids, self.num_negatives
        )  # [batch_size, num_negatives]
        
        # Get positive predictions
        pos_output = self.model.forward(batch_data)
        pos_scores = pos_output['y_pred']  # [B, 1]
        
        # Get negative predictions
        neg_scores_list = []
        for neg_idx in range(self.num_negatives):
            neg_ids = neg_item_ids[:, neg_idx]  # [B]
            
            # Get negative item features
            neg_features = self.negative_sampler.get_item_features(neg_ids.tolist())
            
            # Build negative batch dict (copy user features, replace item features)
            neg_batch_dict = {}
            for key, val in batch_dict.items():
                if key == item_id_col:
                    neg_batch_dict[key] = torch.tensor(neg_ids, device=self.model.device)
                elif key in neg_features.columns:
                    # Replace with negative item feature
                    neg_val = neg_features[key].values
                    neg_batch_dict[key] = torch.tensor(neg_val, device=self.model.device)
                else:
                    # Keep user features
                    neg_batch_dict[key] = val.to(self.model.device) if hasattr(val, 'to') else val
            
            neg_output = self.model.forward(neg_batch_dict)
            neg_scores_list.append(neg_output['y_pred'])
        
        # Stack negative scores: [B, num_negatives]
        neg_scores = torch.cat(neg_scores_list, dim=1)
        
        # Compute pairwise ranking loss
        if self.loss_type == 'bpr':
            loss = bpr_loss(pos_scores, neg_scores)
        elif self.loss_type == 'margin':
            loss = margin_ranking_loss(pos_scores, neg_scores, margin=self.margin)
        elif self.loss_type == 'softmax':
            loss = softmax_cross_entropy_loss(pos_scores, neg_scores)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Add regularization
        if hasattr(self.model, 'regularization_loss'):
            loss = loss + self.model.regularization_loss()
        
        # Backprop
        self.model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model._max_gradient_norm)
        self.model.optimizer.step()
        
        return loss

    def process(self,
                input_data: StageOutput,
                **kwargs) -> Tuple[StageOutput, Dict[str, float]]:
        """
        Process candidates from retrieval stage - scores and selects top-K.
        
        Args:
            input_data: StageOutput from previous stage containing candidate sets
            **kwargs: Additional parameters (e.g., top_k override, compute_metrics)

        Returns:
            Tuple of (StageOutput with Top-K candidates, metrics dict)
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

        compute_metrics = kwargs.pop('compute_metrics', True)
        
        return process_and_rank_candidates(
            model=self.model,
            feature_map=self.feature_map,
            input_data=input_data,
            item_features_df=self.item_features_df,
            stage_name=self.stage_name,
            return_output=True,
            compute_metrics=compute_metrics,
            top_k=kwargs.get('top_k', self.top_k),
            logger=self.logger,
            metrics_k=self.metrics_k,
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

        _, metrics = process_and_rank_candidates(
            model=self.model,
            feature_map=self.feature_map,
            input_data=input_data,
            item_features_df=self.item_features_df,
            stage_name=self.stage_name,
            return_output=False,
            compute_metrics=True,
            metrics_k=metrics_k or self.metrics_k,
            logger=self.logger,
            **kwargs
        )
        return metrics
