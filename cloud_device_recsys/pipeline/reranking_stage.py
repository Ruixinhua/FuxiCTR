# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Re-ranking Stage Implementation

This module wraps the DeviceReranker model as a pipeline stage.
"""

import os
import csv
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from ..pipeline.base_stage import BaseStage, StageType
from ..pipeline.stage_output import StageOutput
from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from ..models import build_model as registry_build_model
from ..models import DeviceReranker  # For type hints
from ..utils import filter_feature_map

from fuxictr.features import FeatureMap


class RerankingStage(BaseStage):
    """
    Re-ranking stage for final recommendation.
    
    Runs on device with access to all features including FG3 (private).
    Produces final top-K recommendations.
    """
    
    def __init__(self,
                 feature_map: FeatureMap,
                 feature_group_manager: FeatureGroupManager,
                 model_params: Dict[str, Any],
                 output_dir: str = "./outputs/reranking",
                 top_k: int = 10,
                 support_distillation: bool = False,
                 **kwargs):
        """
        Initialize re-ranking stage.
        
        Args:
            feature_map: FuxiCTR FeatureMap (with FG3 features)
            feature_group_manager: Feature group manager
            model_params: Parameters for DeviceReranker model
            output_dir: Output directory
            top_k: Number of final recommendations
            support_distillation: Whether to enable distillation
        """
        # Re-ranking uses ALL feature groups
        super().__init__(
            stage_name="reranking",
            stage_type=StageType.RERANKING,
            feature_group_manager=feature_group_manager,
            allowed_feature_groups=[FeatureGroup.FG1, FeatureGroup.FG2, FeatureGroup.FG3],
            output_dir=output_dir,
            **kwargs
        )
        
        # Filter feature_map to only include allowed features (FG1, FG2, FG3)
        self.feature_map = filter_feature_map(feature_map, feature_group_manager, self.allowed_feature_groups,
                                              use_feature_encoder=model_params.get("use_feature_encoder", False))
        self.feature_map.default_emb_dim = model_params['embedding_dim']
        self.top_k = top_k
        self.support_distillation = support_distillation
        self.model_params = model_params
        self.metrics_k = model_params['metrics_k']
        self.model: Optional[DeviceReranker] = None
        self.monitor = model_params.get('monitor', 'nDCG@10')
        self.best_weights_path = None

        # Item features storage for lookups
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
    
    def build_model(self) -> DeviceReranker:
        """Build and initialize the re-ranking model using unified registry"""
        # Ensure output directories exist
        model_dir = os.path.join(self.output_dir, self.feature_map.dataset_id)
        os.makedirs(model_dir, exist_ok=True)

        # Get model name from config, default to DeviceReranker
        model_name = self.model_params.get('model', 'DeviceReranker')
        
        self.model = registry_build_model(
            model_name=model_name,
            feature_map=self.feature_map,
            model_params=self.model_params,
            output_dir=self.output_dir,
            support_distillation=self.support_distillation,
        )
        self.logger.info(f"Built {model_name} model, saving to {model_dir}")
        return self.model
    
    def train(self,
              train_data: Any,
              valid_data: Optional[Any] = None,
              teacher_model: Optional[Any] = None,
              **kwargs) -> Dict[str, float]:
        """
        Train the re-ranking model with best model monitoring.
        
        Args:
            train_data: Training data generator
            valid_data: Validation data generator
            teacher_model: Optional teacher model for distillation
            **kwargs: Training parameters including:
                - epochs: Number of training epochs
                - patience: Early stopping patience
                - mode: 'max' or 'min' for monitor metric
                - reduce_lr_on_plateau: Whether to decay LR on no improvement
                - lr_decay_factor: LR decay factor (default 0.1)
            
        Returns:
            Training metrics
        """
        if self.model is None:
            self.build_model()
            
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
             self.logger.info("Initializing optimizer...")
             self.model.compile(
                 optimizer=kwargs.get("optimizer", "adam"),
                 loss="binary_crossentropy",
                 lr=kwargs.get("learning_rate", 1e-3)
             )

        self.best_weights_path = os.path.join(self.model.model_dir, self.model.model_id + ".model")
        self.logger.info("Starting re-ranking model training (Custom Loop)")
        
        epochs = kwargs.get('epochs', 1)
        patience = kwargs.get('patience', 2)
        mode = kwargs.get('mode', 'max')
        metrics = {}
        
        # Setup model for manual training
        self.model._total_steps = 0
        self.model._stop_training = False
        self.model._max_gradient_norm = kwargs.get("max_gradient_norm", 10.0)
        self.model._verbose = kwargs.get("verbose", 1)
        self.model._epoch_index = 0
        
        self.logger.info(f"Start Training: epochs={epochs}, monitor={self.monitor}, patience={patience}")
        
        best_metric = -np.inf if mode == "max" else np.inf
        stopping_steps = 0
        
        # Distillation training if teacher provided
        if teacher_model is not None and self.support_distillation:
            self.logger.info("Using knowledge distillation training")
            metrics = self.model.distill_from_teacher(
                train_data, teacher_model, **kwargs
            )
            # Save after distillation
            self.model.save_weights(self.best_weights_path)
            self.logger.info(f"Saved distillation model checkpoint to {self.best_weights_path}")
        else:
            # Standard training (Manual Loop) with monitor
            for epoch in range(epochs):
                self.model._epoch_index = epoch
                self.logger.info(f"*** Epoch {epoch + 1}/{epochs} ***")
                
                self.model.train()
                total_loss = 0.0
                steps = 0
                
                for batch_data in train_data:
                    loss = self.model.train_step(batch_data)
                    total_loss += loss.item()
                    steps += 1
                
                avg_loss = total_loss / steps if steps > 0 else 0.0
                self.logger.info(f"Train Loss: {avg_loss:.6f}")
                
                if valid_data is not None:
                    self.logger.info(f"Evaluating epoch {epoch + 1}...")
                    
                    # List-wise metrics (nDCG/Recall)
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
        metrics_path = os.path.join(self.output_dir, "training_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}" if isinstance(value, float) else value])
        
        return metrics

    def process(self,
                input_data: StageOutput,
                **kwargs) -> Tuple[StageOutput, Dict[str, float]]:
        """
        Re-rank candidates to produce final recommendations - scores and selects top-K.
        
        Args:
            input_data: StageOutput from previous stage containing candidate sets
            **kwargs: Additional parameters (e.g., top_k override, compute_metrics)
            
        Returns:
            Tuple of (StageOutput with re-ranked Top-K candidates, metrics dict)
        """
        from ..utils import process_and_rank_candidates
        
        if os.path.exists(self.best_weights_path):
            self.model.load_weights(self.best_weights_path)
            self.logger.info(f"Loaded best weights from {self.best_weights_path}")
        else:
            self.logger.warning(f"No best weights found at {self.best_weights_path}. Using current model state.")
                
        # Ensure item features are loaded
        if self.item_features_df is None:
            self.logger.error("Item features not loaded. Call load_item_features() first.")
            return StageOutput(stage_name=self.stage_name), {}

        return process_and_rank_candidates(
            model=self.model,
            feature_map=self.feature_map,
            input_data=input_data,
            item_features_df=self.item_features_df,
            stage_name=self.stage_name,
            return_output=True,
            compute_metrics=kwargs.pop('compute_metrics', True),
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
        Evaluate re-ranking model with list-wise metrics (nDCG, Recall).
        
        Args:
            input_data: StageOutput containing candidate sets with labels
            metrics_k: List of K values for Recall@K and nDCG@K
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        from ..utils import process_and_rank_candidates

        if self.item_features_df is None:
            self.logger.error("Item features not loaded. Call load_item_features() first.")
            return {}
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
