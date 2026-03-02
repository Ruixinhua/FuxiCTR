# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Re-ranking Stage Implementation

This module wraps the DeviceReranker model as a pipeline stage.
"""
import torch
import os
import csv
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from ..pipeline.base_stage import BaseStage, StageType
from ..pipeline.stage_output import StageOutput
from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from ..models import build_model as registry_build_model
from ..models import DeviceReranker  # For type hints
from ..models.losses import bpr_loss, margin_ranking_loss, softmax_cross_entropy_loss
from ..data.negative_sampler import NegativeSampler
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
        self.allowed_feature_groups = kwargs.pop('allowed_feature_groups', [FeatureGroup.FG1, FeatureGroup.FG2, FeatureGroup.FG3])
        super().__init__(
            stage_name="reranking",
            stage_type=StageType.RERANKING,
            feature_group_manager=feature_group_manager,
            allowed_feature_groups = self.allowed_feature_groups,
            output_dir=output_dir,
            **kwargs
        )
        # Filter feature_map to only include allowed features (FG1, FG2, FG3)
        self.feature_map = filter_feature_map(feature_map, feature_group_manager, self.allowed_feature_groups,
                                              use_feature_encoder=model_params.get("use_feature_encoder", False))
        self.feature_map.default_emb_dim = model_params['embedding_dim']
        self.use_logit = model_params.get('use_logit', True)
        self.top_k = top_k
        self.support_distillation = support_distillation
        self.model_params = model_params
        self.metrics_k = model_params['metrics_k']
        self.model: Optional[DeviceReranker] = None
        self.monitor = model_params.get('monitor', 'Recall@1')
        self.best_weights_path = None
        # Negative sampling parameters
        self.num_negatives = model_params.get('num_negatives', 0)
        self.loss_type = model_params.get('loss_type', 'bpr')  # 'bpr', 'margin', 'softmax'
        self.margin = model_params.get('margin', 1.0)
        self.negative_sampler: Optional[NegativeSampler] = None

        # Cloud score feature injection
        self.use_cloud_score = model_params.get('use_cloud_score', False)
        self.cloud_score_teacher = None  # Pre-ranking model reference

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
    
    def set_cloud_score_teacher(self, teacher_model):
        """Set pre-ranking model to generate cloud scores during training.
        
        The teacher model (pre-ranking, FG1+FG2 only) provides cloud_score
        as a numeric feature for the reranking model. This does NOT violate
        privacy: the teacher only uses cloud-available features.
        
        Note: The teacher keeps its sigmoid activation. We apply torch.logit()
        during injection to recover discriminative raw logits from the
        compressed sigmoid output (~0.9999).
        """
        self.cloud_score_teacher = teacher_model
        self.cloud_score_teacher.eval()
        for param in self.cloud_score_teacher.parameters():
            param.requires_grad = False
        self.logger.info("Cloud score teacher model set (frozen, eval mode)")

    def build_model(self) -> DeviceReranker:
        """Build and initialize the re-ranking model using unified registry"""
        # Register cloud_score as numeric feature BEFORE model construction
        if self.use_cloud_score and 'cloud_score' not in self.feature_map.features:
            self.feature_map.features['cloud_score'] = {
                'type': 'numeric',
                'source': '',
            }
            self.feature_map.num_fields = self.feature_map.get_num_fields()
            self.feature_map.set_column_index()
            self.logger.info("Registered 'cloud_score' as numeric feature in feature_map")

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
            train_data: Training data generator (positive examples only for negative sampling)
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
        
        # Initialize negative sampler if using negative sampling
        use_negative_sampling = self.num_negatives > 0 and self.item_features_df is not None
        if use_negative_sampling:
            item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
            self.negative_sampler = NegativeSampler(self.item_features_df, item_id_col=item_id_col)
            self.logger.info(f"Negative Sampling: {self.num_negatives} negatives per positive, loss_type={self.loss_type}")
        else:
            if self.num_negatives > 0 and self.item_features_df is None:
                self.logger.warning("num_negatives > 0 but item_features_df not loaded. Using standard training.")
        
        # Setup model for manual training
        self.model._total_steps = 0
        self.model._stop_training = False
        self.model._max_gradient_norm = kwargs.get("max_gradient_norm", 10.0)
        self.model._verbose = kwargs.get("verbose", 1)
        self.model._epoch_index = 0
        
        self.logger.info(f"Start Training: epochs={epochs}, monitor={self.monitor}, patience={patience}")
        
        best_metric = -np.inf if mode == "max" else np.inf
        stopping_steps = 0
        
        item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
        
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
                    if use_negative_sampling:
                        loss = self._train_step_with_negatives(batch_data, item_id_col)
                    else:
                        # Inject cloud_score for standard training (non-negative-sampling)
                        if self.use_cloud_score and self.cloud_score_teacher is not None:
                            batch_dict = dict(batch_data)
                            with torch.no_grad():
                                teacher_out = self.cloud_score_teacher.forward(batch_dict)
                                # Logit + z-score: recover discrimination, control magnitude
                                logit_score = torch.logit(
                                    teacher_out['y_pred'].detach().squeeze(-1), eps=1e-7
                                )
                                batch_dict['cloud_score'] = (
                                    (logit_score - logit_score.mean()) / (logit_score.std() + 1e-8)
                                )
                            loss = self.model.train_step(batch_dict)
                        else:
                            loss = self.model.train_step(batch_data)
                    total_loss += loss.item()
                    steps += 1
                
                avg_loss = total_loss / steps if steps > 0 else 0.0
                if use_negative_sampling:
                    self.logger.info(f"Train Loss ({self.loss_type}): {avg_loss:.6f}")
                else:
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
    
    def _train_step_with_negatives(self, batch_data, item_id_col: str):
        """
        Single training step with negative sampling for pairwise ranking.
        
        Optimized: batches all negatives into a single forward pass instead of
        processing each negative separately.
        
        Args:
            batch_data: Positive example batch
            item_id_col: Name of item ID column

        Returns:
            Loss tensor
        """
        batch_dict = dict(batch_data)
        batch_size = len(batch_dict[item_id_col])
        
        # Get positive item IDs
        pos_item_ids = batch_dict[item_id_col].cpu().numpy()
        
        # Sample negative items for each positive: [B, num_negatives]
        neg_item_ids = self.negative_sampler.sample_negatives_batch(
            pos_item_ids, self.num_negatives
        )
        
        # === Build batched negative dict: repeat user features, use negative item features ===
        # Flatten: [B, num_neg] -> [B * num_neg]
        neg_ids_flat = neg_item_ids.reshape(-1)
        neg_features = self.negative_sampler.get_features_by_ids(neg_ids_flat)
        
        neg_batch_dict = {}
        for key, val in batch_dict.items():
            if key == item_id_col:
                neg_batch_dict[key] = torch.tensor(neg_ids_flat, device=self.model.device)
            elif key == 'cloud_score':
                continue  # Will compute via teacher below
            elif key in neg_features:
                # Use negative item feature
                val = neg_features[key].to_numpy(copy=False)
                if not np.isscalar(val[0]):
                    val = np.vstack(val)  # Ensure 2D for multi-valued features
                neg_batch_dict[key] = torch.tensor(val, device=self.model.device)
            else:
                # Repeat user features along batch dimension: [B, ...] -> [B * num_neg, ...]
                if hasattr(val, 'to'):
                    val = val.to(self.model.device)
                    neg_batch_dict[key] = val.repeat_interleave(self.num_negatives, dim=0)
                else:
                    neg_batch_dict[key] = val
        
        # === Cloud Score Injection (positive + negative, jointly normalized) ===
        if self.use_cloud_score and self.cloud_score_teacher is not None:
            with torch.no_grad():
                teacher_pos_out = self.cloud_score_teacher.forward(batch_dict)
                pos_logit = torch.logit(
                    teacher_pos_out['y_pred'].detach().squeeze(-1), eps=1e-7
                )
                teacher_neg_out = self.cloud_score_teacher.forward(neg_batch_dict)
                neg_logit = torch.logit(
                    teacher_neg_out['y_pred'].detach().squeeze(-1), eps=1e-7
                )
                # Joint z-score normalization (pos + neg for consistent scale)
                all_logits = torch.cat([pos_logit, neg_logit])
                mean, std = all_logits.mean(), all_logits.std() + 1e-8
                batch_dict['cloud_score'] = (pos_logit - mean) / std
                neg_batch_dict['cloud_score'] = (neg_logit - mean) / std

        # Forward passes
        pos_output = self.model.forward(batch_dict)

        neg_output = self.model.forward(neg_batch_dict)
        if self.use_logit:
            pos_scores = pos_output.get('logit', pos_output['y_pred'])  # [B, 1]
            neg_scores_flat = neg_output.get('logit', neg_output['y_pred'])  # [B * num_neg, 1]
        else:
            pos_scores = pos_output['y_pred']
            neg_scores_flat = neg_output['y_pred']  # [B * num_neg, 1]
        # Reshape back: [B * num_neg, 1] -> [B, num_neg]
        neg_scores = neg_scores_flat.view(batch_size, self.num_negatives)
        
        # Compute pairwise ranking loss
        if self.loss_type == 'bpr':
            loss = bpr_loss(pos_scores, neg_scores)
        elif self.loss_type == 'margin':
            loss = margin_ranking_loss(pos_scores, neg_scores, margin=self.margin)
        elif self.loss_type == 'softmax':
            loss = softmax_cross_entropy_loss(pos_scores, neg_scores)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
            
        # Add diversity loss if enabled (only on positive samples for recommendation diversity)
        if getattr(self.model, 'use_diversity_loss', False):
            # Get embeddings from positive output (must use pos_output, not _last_feat_emb_dict
            # which gets overwritten by negative forward pass)
            pos_feat_emb_dict = pos_output.get('feat_emb_dict')
            if pos_feat_emb_dict is not None:
                div_loss = self.model.compute_diversity_regularization(
                    pos_feat_emb_dict, pos_scores
                )
                if div_loss is not None:
                    loss = self.model.add_diversity_to_loss(loss, div_loss)
        
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
        Re-rank candidates to produce final recommendations - scores and selects top-K.
        
        Args:
            input_data: StageOutput from previous stage containing candidate sets
            **kwargs: Additional parameters (e.g., top_k override, compute_metrics)
            
        Returns:
            Tuple of (StageOutput with re-ranked Top-K candidates, metrics dict)
        """
        from ..metric_utils import process_and_rank_candidates
        
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
            inject_cloud_score=self.use_cloud_score,
            ranking_candidates_df=kwargs.pop('preranking_candidates_df', None),
            **kwargs
        )

    def evaluate(self,
                 input_data: StageOutput,
                 metrics_k: List[int] = None,
                 preranking_output: Optional[StageOutput] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate re-ranking model with list-wise metrics (nDCG, Recall).
        
        Args:
            input_data: StageOutput containing candidate sets with labels (full pool, e.g. 1000 candidates)
            metrics_k: List of K values for Recall@K and nDCG@K
            preranking_output: Optional StageOutput with preranking-filtered candidates (e.g. top-100).
                               When provided, Recall@K/nDCG@K are computed within this filtered subset,
                               while AUC/gAUC use the full input_data pool for fair cross-stage comparison.
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        from ..metric_utils import process_and_rank_candidates

        if self.item_features_df is None:
            self.logger.error("Item features not loaded. Call load_item_features() first.")
            return {}
        
        ranking_candidates_df = None
        if preranking_output is not None:
            ranking_candidates_df = preranking_output.candidates_df
            self.logger.info(f"Fair eval: scoring on {input_data.get_total_candidates()} candidates, "
                             f"ranking restricted to {len(ranking_candidates_df)} preranking candidates")

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
            inject_cloud_score=self.use_cloud_score,
            ranking_candidates_df=ranking_candidates_df,
            **kwargs
        )
        return metrics
