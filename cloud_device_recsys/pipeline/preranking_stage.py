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
import torch
from typing import Dict, List, Optional, Any, Tuple

from ..pipeline.base_stage import BaseStage, StageType
from ..pipeline.stage_output import StageOutput
from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from ..models import build_model as registry_build_model
from ..models import DINRanker  # For type hints
from ..models.losses import bpr_loss, margin_ranking_loss, softmax_cross_entropy_loss, compute_diversity_loss_per_user
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
                 allowed_feature_groups: List[FeatureGroup] = None,
                 output_dir: str = "./outputs/preranking",
                 top_k: int = 100,
                 **kwargs):
        """
        Initialize pre-ranking stage.
        
        Args:
            feature_map: FuxiCTR FeatureMap
            feature_group_manager: Feature group manager
            model_params: Parameters for preranking model
            allowed_feature_groups: Allowed feature groups
            output_dir: Output directory
            top_k: Number of candidates to pass to next stage
            use_diversity: Whether to apply diversity in selection
        """
        if allowed_feature_groups is None:
            allowed_feature_groups = [FeatureGroup.FG1, FeatureGroup.FG2]
            
        super().__init__(
            stage_name="preranking",
            stage_type=StageType.PRERANKING,
            feature_group_manager=feature_group_manager,
            allowed_feature_groups=allowed_feature_groups,
            output_dir=output_dir,
            **kwargs
        )
        
        # Filter feature_map to only include allowed features (FG1, FG2)
        self.feature_map = filter_feature_map(feature_map, feature_group_manager, self.allowed_feature_groups,
                                              use_feature_encoder=model_params.get("use_feature_encoder", False))
        self.feature_map.default_emb_dim = model_params['embedding_dim']
        self.use_logit = model_params.get('use_logit', True)
        self.top_k = top_k
        self.use_diversity_loss = model_params.get('use_diversity_loss', False)
        if self.use_diversity_loss:
            self.logger.info(
                f"Computing diversity loss theta: {model_params.get('diversity_theta', 0.7)} "
                f"lambda: {model_params.get('diversity_lambda', 0.7)} "
                f"kernel: {model_params.get('diversity_kernel', 'rbf')} "
                f"gamma: {model_params.get('diversity_gamma', 1.0)}")
        # Delayed diversity loss parameters
        self.diversity_start_epoch = model_params.get('diversity_start_epoch', -1)
        self.diversity_epochs = model_params.get('diversity_epochs', 5)
        self.diversity_warmup_epochs = model_params.get('diversity_warmup_epochs', 0)
        self.model_params = model_params
        self.metrics_k = model_params['metrics_k']
        self.monitor = model_params.get('monitor', 'Recall@100')
        self.patience = model_params.get('patience', 2)
        self.model: Optional[DINRanker] = None
        self.best_weights_path = None
        # Negative sampling parameters
        self.num_negatives = model_params.get('num_negatives', 0)
        self.diversity_num_negatives = model_params.get('diversity_num_negatives', self.num_negatives)
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
    
    def _set_diversity_enabled(self, enabled: bool):
        """
        Toggle diversity loss on the model at runtime.
        
        Supports both wrapper-based models (model._diversity_enabled) and
        mixin-based models (model._use_diversity_loss).
        """
        if self.model is None:
            return
        # Wrapper-based model (from wrap_model_with_diversity)
        if hasattr(self.model, '_diversity_enabled'):
            self.model._diversity_enabled = enabled
        # Mixin-based model (DiversityLossMixin)
        if hasattr(self.model, '_use_diversity_loss'):
            self.model._use_diversity_loss = enabled
        self.logger.info(f"Diversity loss {'enabled' if enabled else 'disabled'} on model")

    def _set_diversity_lambda(self, lambda_value: float):
        """
        Set diversity lambda on the model at runtime.
        """
        if self.model is None:
            return
        if hasattr(self.model, '_diversity_lambda'):
            self.model._diversity_lambda = lambda_value
        if hasattr(self.model, '_diversity_logger'):
             # Optional: log adjust if needed, but per-epoch log is better
             pass

    def _run_training_phase(self, phase_name, train_data, valid_data,
                            epochs, patience, mode, use_negative_sampling,
                            item_id_col, best_metric, metrics, 
                            diversity_warmup_epochs=0, target_diversity_lambda=0.01,
                            **kwargs):
        """
        Run a single training phase (used by both Phase 1 and Phase 2).
        
        Args:
            phase_name: Name for logging (e.g., "Phase 1", "Phase 2")
            train_data: Training data generator
            valid_data: Validation data generator
            epochs: Max epochs for this phase
            patience: Early stopping patience
            mode: 'max' or 'min' for monitor metric
            use_negative_sampling: Whether to use negative sampling
            item_id_col: Name of item ID column
            best_metric: Starting best metric value
            metrics: Metrics dict to update (mutated in-place)
            diversity_warmup_epochs: Number of epochs to warmup diversity lambda
            target_diversity_lambda: Final target value for diversity lambda
            **kwargs: Additional training parameters
            
        Returns:
            Updated best_metric value
        """
        stopping_steps = 0
        
        for epoch in range(epochs):
            self.model._epoch_index = epoch
            
            # --- Diversity Warmup Logic ---
            if diversity_warmup_epochs > 0 and self.use_diversity_loss:
                if epoch < diversity_warmup_epochs:
                    warmup_lambda = (epoch + 1) / diversity_warmup_epochs * target_diversity_lambda
                    # Clamp to target
                    current_lambda = min(warmup_lambda, target_diversity_lambda)
                    self._set_diversity_lambda(current_lambda)
                    self.logger.info(f"[{phase_name}] Diversity Warmup: lambda={current_lambda:.6f} (Epoch {epoch+1}/{diversity_warmup_epochs})")
                else:
                    # Ensure it is at target
                    if getattr(self.model, '_diversity_lambda', 0) != target_diversity_lambda:
                        self._set_diversity_lambda(target_diversity_lambda)
                        self.logger.info(f"[{phase_name}] Diversity Warmup Complete: lambda={target_diversity_lambda}")

            self.logger.info(f"*** [{phase_name}] Epoch {epoch + 1}/{epochs} ***")
            
            # Training Loop
            self.model.train()
            total_loss = 0.0
            steps = 0
            
            for batch_data in train_data:
                if use_negative_sampling and self.num_negatives > 0:
                    loss = self._train_step_with_negatives(batch_data, item_id_col, torch)
                else:
                    loss = self.model.train_step(batch_data)
                
                total_loss += loss.item()
                steps += 1
            
            avg_loss = total_loss / steps if steps > 0 else 0.0
            if use_negative_sampling:
                self.logger.info(f"[{phase_name}] Train Loss ({self.loss_type}): {avg_loss:.6f}")
            else:
                self.logger.info(f"[{phase_name}] Train Loss: {avg_loss:.6f}")
            
            # Validation (Ranking Metrics)
            self.logger.info(f"[{phase_name}] Evaluating epoch {epoch + 1}...")
            valid_metrics = self.evaluate(valid_data)
            metrics.update(valid_metrics)
            self.logger.info(f"[{phase_name}] Validation (Ranking): {valid_metrics}")
            
            # Monitor-based best model saving
            curr_val = valid_metrics.get(self.monitor, 0.0)
            is_best = (curr_val > best_metric) if mode == "max" else (curr_val < best_metric)
            
            if is_best:
                best_metric = curr_val
                stopping_steps = 0
                self.model.save_weights(self.best_weights_path)
                self.logger.info(f"[{phase_name}] New Best {self.monitor}={curr_val:.6f}! Model Saved.")
            else:
                stopping_steps += 1
                self.logger.info(f"[{phase_name}] No improvement. Patience {stopping_steps}/{patience}")
                
                # Decay LR on plateau
                if kwargs.get("reduce_lr_on_plateau", True):
                    old_lr = self.model.optimizer.param_groups[0]['lr']
                    new_lr = self.model.lr_decay(factor=kwargs.get("lr_decay_factor", 0.1))
                    self.logger.info(f"[{phase_name}] Decay LR: {old_lr:.6f} -> {new_lr:.6f}")
                
                if stopping_steps >= patience:
                    self.logger.info(f"[{phase_name}] Early Stopping.")
                    break
        
        return best_metric

    def train(self,
              train_data: Any,
              valid_data: Optional[Any] = None,
              **kwargs) -> Dict[str, float]:
        """
        Train the pre-ranking model with custom loop and best model monitoring.
        
        Supports two-phase training when use_diversity_loss is enabled:
        - Phase 1: Train without diversity loss until early stopping
        - Phase 2: Load best weights, enable diversity loss, fine-tune
        
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

        if self.model is None:
            self.build_model()
        
        self.logger.info("Starting pre-ranking model training (Custom Loop)")
        self.best_weights_path = os.path.join(self.model.model_dir, self.model.model_id + ".model")

        epochs = kwargs.pop('epochs', 1)
        patience = kwargs.pop('patience', self.patience)
        mode = kwargs.pop('mode', 'max')
        initial_lr = kwargs.pop('learning_rate', self.model_params.get('learning_rate', 1e-3))
        metrics = {}
        
        # Initialize negative sampler if using negative sampling
        use_negative_sampling = self.num_negatives > 0 and self.item_features_df is not None
        if use_negative_sampling:
            item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
            self.negative_sampler = NegativeSampler(
                self.item_features_df,
                item_id_col=item_id_col,
            )
            self.logger.info(f"Negative Sampling: {self.num_negatives} negatives per positive, loss_type={self.loss_type}")
        else:
            if self.num_negatives > 0 and self.item_features_df is None:
                self.logger.warning("num_negatives > 0 but item_features_df not loaded. Using standard training.")
        
        # Ensure optimizer is initialized
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
             self.logger.info("Initializing optimizer...")
             self.model.compile(
                 optimizer=kwargs.get("optimizer", self.model_params.get("optimizer", "adam")),
                 loss=self.model_params.get("loss", "binary_crossentropy"),
                 lr=initial_lr
             )
             
        # Setup model for manual training (required by train_step)
        self.model._total_steps = 0
        self.model._stop_training = False
        self.model._max_gradient_norm = kwargs.get("max_gradient_norm", 10.0)
        self.model._verbose = kwargs.get("verbose", 1)
        self.model._epoch_index = 0

        item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
        best_metric = -np.inf if mode == "max" else np.inf

        # ======================================================================
        # Determine training strategy
        # ======================================================================
        use_two_phase = self.use_diversity_loss  # Two-phase only when diversity is requested
        
        if use_two_phase:
            # --- Phase 1: Train WITHOUT diversity loss ---
            self._set_diversity_enabled(False)
            # Also disable diversity in the stage-level flag for _train_step_with_negatives
            phase1_use_diversity = self.use_diversity_loss
            self.use_diversity_loss = False

            if self.diversity_start_epoch != 0:
                phase1_epochs = self.diversity_start_epoch if self.diversity_start_epoch > 0 else epochs
                self.logger.info(
                    f"=== Phase 1: Base Training (no diversity) ==="
                    f" epochs={phase1_epochs}, monitor={self.monitor}, patience={patience}"
                )
                best_metric = self._run_training_phase(
                    phase_name="Phase 1",
                    train_data=train_data,
                    valid_data=valid_data,
                    epochs=phase1_epochs,
                    patience=patience,
                    mode=mode,
                    use_negative_sampling=use_negative_sampling,
                    item_id_col=item_id_col,
                    best_metric=best_metric,
                    metrics=metrics,
                    **kwargs
                )
                # Restore best weights from Phase 1 as starting point for Phase 2
                if os.path.exists(self.best_weights_path):
                    self.model.load_weights(self.best_weights_path)
                    self.logger.info(f"Phase 1 complete. Best {self.monitor}={best_metric:.6f}. Loaded best weights.")

            # --- Phase 2: Fine-tune WITH diversity loss ---
            self.use_diversity_loss = phase1_use_diversity  # Restore the flag
            self._set_diversity_enabled(True)
            
            # Reset learning rate to initial value for Phase 2
            for param_group in self.model.optimizer.param_groups:
                param_group['lr'] = initial_lr
            self.logger.info(f"Reset learning rate to {initial_lr} for Phase 2")
            
            self.logger.info(
                f"=== Phase 2: Diversity Fine-tuning ==="
                f" epochs={self.diversity_epochs}, monitor={self.monitor}, patience={patience}, warmup={self.diversity_warmup_epochs}"
            )
            
            # Verify target lambda
            target_lambda = self.model_params.get('diversity_lambda', 0.7)
            lr_decay_factor = kwargs.pop("lr_decay_factor", 0.5)
            self.num_negatives = self.diversity_num_negatives  # Update num_negatives for Phase 2 if specified
            best_metric = self._run_training_phase(
                phase_name="Phase 2 (Diversity)",
                train_data=train_data,
                valid_data=valid_data,
                epochs=self.diversity_epochs,
                patience=patience,
                mode=mode,
                use_negative_sampling=use_negative_sampling,
                item_id_col=item_id_col,
                best_metric=best_metric,
                metrics=metrics,
                diversity_warmup_epochs=self.diversity_warmup_epochs,
                target_diversity_lambda=target_lambda,
                lr_decay_factor=lr_decay_factor,
                **kwargs
            )
        else:
            # --- Single-phase training (no diversity) ---
            self.logger.info(f"Start Training: epochs={epochs}, monitor={self.monitor}, patience={patience}")
            
            best_metric = self._run_training_phase(
                phase_name="Training",
                train_data=train_data,
                valid_data=valid_data,
                epochs=epochs,
                patience=patience,
                mode=mode,
                use_negative_sampling=use_negative_sampling,
                item_id_col=item_id_col,
                best_metric=best_metric,
                metrics=metrics,
                **kwargs
            )
        self.logger.info(f"Training complete. Best {self.monitor}={best_metric:.6f}.")

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
        
        Optimized: batches all negatives into a single forward pass instead of
        processing each negative separately.
        
        Args:
            batch_data: Positive example batch
            item_id_col: Name of item ID column
            torch: Torch module reference
            
        Returns:
            Loss tensor
        """
        batch_dict = dict(batch_data)
        batch_size = len(batch_dict[item_id_col])
        
        # Get positive item IDs
        pos_item_ids = batch_dict[item_id_col].cpu().numpy()
        neg_item_ids = self.negative_sampler.sample_negatives_batch(
            pos_item_ids, self.num_negatives
        )
        
        # Get positive predictions
        pos_output = self.model.forward(batch_data)
        # BPR and Softmax losses require logits, not probabilities
        if self.use_logit:
            pos_scores = pos_output.get('logit', pos_output['y_pred'])  # [B, 1]
        else:
            pos_scores = pos_output['y_pred']
        
        # === Optimized: Batch all negatives into single forward pass ===
        # Flatten: [B, num_neg] -> [B * num_neg]
        neg_ids_flat = neg_item_ids.reshape(-1)
        
        # Get features for all negatives at once using the optimized method
        neg_features = self.negative_sampler.get_features_by_ids(neg_ids_flat)
        
        # Build batched negative dict: repeat user features, use negative item features
        neg_batch_dict = {}
        for key, val in batch_dict.items():
            if key == item_id_col:
                neg_batch_dict[key] = torch.tensor(neg_ids_flat, device=self.model.device)
            elif key in neg_features.columns:
                # Use negative item feature
                val = neg_features[key].to_numpy(copy=False)
                if not np.isscalar(val[0]):
                    val = np.vstack(val)  # Ensure 2D for multi-valued features
                neg_batch_dict[key] = torch.tensor(val, device=self.model.device)
            else:
                # Repeat user features along batch dimension: [B, ...] -> [B * num_neg, ...]
                if hasattr(val, 'to'):
                    val = val.to(self.model.device)
                    # Repeat each element num_negatives times along batch dim (dim=0)
                    neg_batch_dict[key] = val.repeat_interleave(self.num_negatives, dim=0)
                else:
                    neg_batch_dict[key] = val

        neg_output = self.model.forward(neg_batch_dict)
        # BPR and Softmax losses require logits, not probabilities
        if self.use_logit:
            neg_scores_flat = neg_output.get('logit', neg_output['y_pred'])  # [B * num_neg, 1]
        else:
            neg_scores_flat = neg_output['y_pred']
        
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
            
        # Add per-user diversity loss if enabled
        if self.use_diversity_loss:
            diversity_theta = self.model_params.get('diversity_theta', 0.7)
            diversity_lambda = getattr(self.model, '_diversity_lambda', self.model_params.get('diversity_lambda', 0.01))
            diversity_kernel = self.model_params.get('diversity_kernel', 'cosine')
            diversity_gamma = self.model_params.get('diversity_gamma', 1.0)
            diversity_delta = compute_diversity_loss_per_user(
                model=self.model,
                pos_inputs=batch_data,
                neg_inputs=neg_batch_dict,
                pos_scores=pos_scores,
                neg_scores_flat=neg_scores_flat,
                num_negatives=self.num_negatives,
                theta=diversity_theta,
                lambda_=diversity_lambda,
                kernel=diversity_kernel,
                gamma=diversity_gamma,
            )
            loss = loss + diversity_delta

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
        from ..metric_utils import process_and_rank_candidates
        if os.path.exists(self.best_weights_path) and kwargs.pop('load_best_model', True):
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
        from ..metric_utils import process_and_rank_candidates
                 
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
