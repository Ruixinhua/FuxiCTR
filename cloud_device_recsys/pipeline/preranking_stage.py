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
        self._log_active_feature_summary()
        self.feature_map.default_emb_dim = model_params['embedding_dim']
        self.use_logit = model_params.get('use_logit', True)
        self.top_k = top_k
        self.use_diversity_loss = model_params.get('use_diversity_loss', False)
        if self.use_diversity_loss:
            self.logger.info(
                f"Computing diversity loss theta: {model_params.get('diversity_theta', 0.7)} "
                f"lambda: {model_params.get('diversity_lambda', 0.7)} "
                f"kernel: {model_params.get('diversity_kernel', 'gram')} "
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
        self.evaluate_pool_diversity = model_params.get('evaluate_pool_diversity', False)
        self.loss_type = model_params.get('loss_type', 'bpr')  # 'bpr', 'margin', 'softmax'
        self.margin = model_params.get('margin', 1.0)
        self.use_in_batch_negatives = model_params.get('use_in_batch_negatives', False)
        self.in_batch_negative_chunk_size = max(1, int(model_params.get('in_batch_negative_chunk_size', 128)))
        self.in_batch_negative_sample_size = int(model_params.get('in_batch_negative_sample_size', 255))
        self.negative_sampler: Optional[NegativeSampler] = None
        # Inference batch size for process_and_rank_candidates
        self.inference_batch_size = model_params.get('inference_batch_size', 50000)
        # Item features storage for lookups during evaluation/processing
        self.item_features_df = None

    def _log_active_feature_summary(self):
        """Log the feature groups that are actually retained in this stage's filtered feature_map."""
        impression_col = getattr(self.feature_map, 'dataset_config', {}).get('impression_id_col', 'impression_id')
        special_cols = {impression_col, 'group_id', 'click', 'clk', 'label', *self.feature_map.labels}

        active_model_features = [
            name for name in self.feature_map.features.keys()
            if name not in special_cols
        ]

        group_buckets = {
            FeatureGroup.FG1: [],
            FeatureGroup.FG2: [],
            FeatureGroup.FG3: [],
        }
        unassigned = []

        for feat_name in active_model_features:
            group = self.feature_group_manager.feature_assignments.get(feat_name)
            if group in group_buckets:
                group_buckets[group].append(feat_name)
            else:
                unassigned.append(feat_name)

        self.logger.info(
            "Active model features after filtering: total=%d, FG1=%d, FG2=%d, FG3=%d, unassigned=%d",
            len(active_model_features),
            len(group_buckets[FeatureGroup.FG1]),
            len(group_buckets[FeatureGroup.FG2]),
            len(group_buckets[FeatureGroup.FG3]),
            len(unassigned),
        )
        self.logger.info("Active FG1 Features: %s", ", ".join(group_buckets[FeatureGroup.FG1]) or "(none)")
        self.logger.info("Active FG2 Features: %s", ", ".join(group_buckets[FeatureGroup.FG2]) or "(none)")
        self.logger.info("Active FG3 Features: %s", ", ".join(group_buckets[FeatureGroup.FG3]) or "(none)")
        if unassigned:
            self.logger.info("Active Unassigned Features: %s", ", ".join(unassigned))

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
                            epochs, patience, mode, use_pairwise_training,
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
            use_pairwise_training: Whether to use pairwise/in-batch training
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
                if use_pairwise_training:
                    loss = self._train_step_with_negatives(batch_data, item_id_col, torch)
                else:
                    loss = self.model.train_step(batch_data)
                
                total_loss += loss.item()
                steps += 1
            
            avg_loss = total_loss / steps if steps > 0 else 0.0
            if use_pairwise_training:
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

    @staticmethod
    def _uses_dp_gradient_perturbation(model: Any) -> bool:
        """Return True when a model exposes DP-aware gradient update hooks."""
        return (
            hasattr(model, "_clip_gradients")
            and callable(getattr(model, "_clip_gradients"))
            and hasattr(model, "_add_dp_noise")
            and callable(getattr(model, "_add_dp_noise"))
            and hasattr(model, "max_grad_norm_per_sample")
            and hasattr(model, "noise_multiplier")
        )

    @classmethod
    def _apply_gradient_update(cls, model: Any, loss: torch.Tensor, batch_size: int):
        """
        Apply one optimizer update, dispatching to model-specific DP logic when available.

        The preranking pairwise loop bypasses ``model.train_step()``, so DP models such
        as DPSGD need their clipping/noise path re-applied here.
        """
        model.optimizer.zero_grad()
        loss.backward()

        cls._finalize_gradient_update(model, batch_size)

    @classmethod
    def _finalize_gradient_update(cls, model: Any, batch_size: int):
        """Finalize one optimizer update after gradients have already been accumulated."""

        if cls._uses_dp_gradient_perturbation(model):
            model._clip_gradients()
            model._add_dp_noise(batch_size)
            if hasattr(model, "_dp_steps"):
                model._dp_steps += 1
            if hasattr(model, "_total_samples"):
                model._total_samples += batch_size
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), model._max_gradient_norm)

        model.optimizer.step()

    def _get_pairwise_scores(self, model_output: Dict[str, Any]) -> torch.Tensor:
        """Extract the score tensor used by pairwise and in-batch losses."""
        if self.use_logit:
            return model_output.get('logit', model_output['y_pred'])
        return model_output['y_pred']

    @staticmethod
    def _supports_pairwise_auxiliary_loss(model: Any) -> bool:
        """Return True when pairwise training should preserve model-specific auxiliary losses."""
        return type(model).__name__ in {"DualRec", "FedCAR", "FedCIA"}

    def _compute_pairwise_auxiliary_loss(self, model_output: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Re-apply model-specific auxiliary losses when pairwise training bypasses model.train_step().

        The pairwise preranking loop optimizes ranking losses directly and therefore skips the
        BaseModel -> add_loss() path where several privacy-preserving models define their
        collaborative regularizers. This helper mirrors only those auxiliary terms, leaving the
        pairwise ranking loss as the primary supervised signal for positives vs. negatives.
        """
        model_name = type(self.model).__name__
        aux_loss = None

        if model_name == "DualRec":
            personalized_mask = model_output["personalized_mask"]
            if personalized_mask.any():
                kd_loss = self.model._compute_kd_loss(
                    model_output["device_logit"][personalized_mask],
                    model_output["cloud_logit"][personalized_mask].detach(),
                )
                aux_loss = self.model.kd_loss_weight * kd_loss

                if self.model.mutual_reg_weight > 0:
                    reverse_kd = self.model._compute_kd_loss(
                        model_output["cloud_logit"][personalized_mask],
                        model_output["device_logit"][personalized_mask].detach(),
                    )
                    aux_loss = aux_loss + self.model.mutual_reg_weight * reverse_kd

            if self.model.odr_loss_weight > 0:
                odr_loss = self.model._compute_odr_loss(
                    model_output["device_logit"],
                    model_output["cloud_logit"],
                )
                aux_loss = odr_loss * self.model.odr_loss_weight if aux_loss is None else aux_loss + self.model.odr_loss_weight * odr_loss

            return aux_loss

        if model_name == "FedCAR":
            if self.model.contrastive_weight > 0:
                contrastive_loss = self.model._info_nce_loss(
                    model_output["cloud_proj"].detach(),
                    model_output["device_proj"],
                )
                aux_loss = self.model.contrastive_weight * contrastive_loss

            if self.model.use_prototype and self.model.training:
                self.model._update_prototype(model_output["cloud_proj"])
                device_proj_norm = torch.nn.functional.normalize(model_output["device_proj"], dim=-1)
                proto_norm = torch.nn.functional.normalize(self.model.global_prototype.unsqueeze(0), dim=-1)
                prototype_loss = 1 - (device_proj_norm * proto_norm).sum(dim=-1).mean()
                weighted_proto_loss = self.model.prototype_weight * prototype_loss
                aux_loss = weighted_proto_loss if aux_loss is None else aux_loss + weighted_proto_loss

            return aux_loss

        if model_name == "FedCIA":
            cloud_sim = self.model._compute_similarity_matrix(
                model_output["cloud_latent"].detach(),
                add_noise=True,
            )
            device_sim = self.model._compute_similarity_matrix(
                model_output["device_latent"],
                add_noise=False,
            )
            align_loss = self.model.similarity_align_weight * torch.nn.functional.mse_loss(device_sim, cloud_sim)
            aux_loss = align_loss

            if self.model.reverse_align_weight > 0:
                cloud_sim_live = self.model._compute_similarity_matrix(
                    model_output["cloud_latent"],
                    add_noise=False,
                )
                device_sim_detached = self.model._compute_similarity_matrix(
                    model_output["device_latent"].detach(),
                    add_noise=False,
                )
                reverse_loss = self.model.reverse_align_weight * torch.nn.functional.mse_loss(
                    cloud_sim_live,
                    device_sim_detached,
                )
                aux_loss = aux_loss + reverse_loss

            return aux_loss

        return None

    def _get_item_feature_keys(self, item_id_col: str) -> set:
        """Infer which batch columns should be swapped when replacing candidate items."""
        item_feature_keys = {item_id_col}
        if self.item_features_df is not None:
            item_feature_keys.update(self.item_features_df.columns)
        if self.feature_group_manager is not None:
            for feat_name, group in self.feature_group_manager.feature_assignments.items():
                if group == FeatureGroup.FG1:
                    item_feature_keys.add(feat_name)
        return item_feature_keys

    def _build_in_batch_chunk(
        self,
        batch_dict: Dict[str, Any],
        item_feature_keys: set,
        row_start: int,
        row_end: int,
        item_indices: torch.Tensor,
    ) -> Dict[str, Any]:
        """Build one in-batch chunk with explicit candidate indices for each user row."""
        candidates_per_row = item_indices.size(1)
        flat_item_indices = item_indices.reshape(-1)
        pair_batch_dict = {}

        for key, val in batch_dict.items():
            if hasattr(val, 'to'):
                val = val.to(self.model.device)
                if key in item_feature_keys:
                    pair_batch_dict[key] = val.index_select(0, flat_item_indices)
                else:
                    user_chunk = val[row_start:row_end]
                    pair_batch_dict[key] = user_chunk.repeat_interleave(candidates_per_row, dim=0)
            else:
                pair_batch_dict[key] = val
        return pair_batch_dict

    @staticmethod
    def _sample_in_batch_negative_indices(
        batch_size: int,
        row_start: int,
        row_end: int,
        negatives_per_row: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample batch-local negatives with replacement while excluding each row's own positive index.
        """
        row_count = row_end - row_start
        sampled = torch.randint(0, batch_size - 1, (row_count, negatives_per_row), device=device)
        row_indices = torch.arange(row_start, row_end, device=device).unsqueeze(1)
        return sampled + (sampled >= row_indices).long()

    def _backward_in_batch_loss(
        self,
        batch_dict: Dict[str, Any],
        item_id_col: str,
        loss_weight: float = 1.0,
    ) -> float:
        """
        Backprop the in-batch cross-entropy loss chunk-by-chunk to cap peak activation memory.
        """
        batch_size = len(batch_dict[item_id_col])
        chunk_size = min(self.in_batch_negative_chunk_size, batch_size)
        max_negatives = max(batch_size - 1, 0)
        use_sampled_negatives = (
            self.in_batch_negative_sample_size > 0
            and self.in_batch_negative_sample_size < max_negatives
        )
        item_feature_keys = self._get_item_feature_keys(item_id_col)
        total_loss_value = 0.0

        for row_start in range(0, batch_size, chunk_size):
            row_end = min(row_start + chunk_size, batch_size)
            row_count = row_end - row_start
            row_indices = torch.arange(row_start, row_end, device=self.model.device)

            if use_sampled_negatives:
                neg_indices = self._sample_in_batch_negative_indices(
                    batch_size=batch_size,
                    row_start=row_start,
                    row_end=row_end,
                    negatives_per_row=self.in_batch_negative_sample_size,
                    device=self.model.device,
                )
                item_indices = torch.cat([row_indices.unsqueeze(1), neg_indices], dim=1)
                in_batch_targets = torch.zeros(row_count, dtype=torch.long, device=self.model.device)
            else:
                item_indices = torch.arange(batch_size, device=self.model.device).unsqueeze(0).expand(row_count, -1)
                in_batch_targets = row_indices

            pair_batch_dict = self._build_in_batch_chunk(
                batch_dict=batch_dict,
                item_feature_keys=item_feature_keys,
                row_start=row_start,
                row_end=row_end,
                item_indices=item_indices,
            )
            pair_output = self.model.forward(pair_batch_dict)
            pair_scores = self._get_pairwise_scores(pair_output).reshape(row_count, item_indices.size(1))
            chunk_loss = torch.nn.functional.cross_entropy(
                pair_scores,
                in_batch_targets,
                reduction='sum',
            )
            scaled_chunk_loss = loss_weight * chunk_loss / batch_size
            total_loss_value += scaled_chunk_loss.detach().item()
            scaled_chunk_loss.backward()

        return total_loss_value

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
        
        # Initialize negative sampler if using explicit negative sampling
        use_negative_sampling = self.num_negatives > 0 and self.item_features_df is not None
        use_pairwise_training = use_negative_sampling or self.use_in_batch_negatives
        if use_negative_sampling:
            item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
            self.negative_sampler = NegativeSampler(
                self.item_features_df,
                item_id_col=item_id_col,
            )
            self.logger.info(f"Negative Sampling: {self.num_negatives} negatives per positive, loss_type={self.loss_type}")
        if self.use_in_batch_negatives:
            if self.in_batch_negative_sample_size > 0:
                self.logger.info(
                    "In-batch negatives enabled (sampled cross-entropy, chunk_size=%d, negatives_per_example=%d)",
                    self.in_batch_negative_chunk_size,
                    self.in_batch_negative_sample_size,
                )
            else:
                self.logger.info(
                    "In-batch negatives enabled (full-batch cross-entropy, chunk_size=%d)",
                    self.in_batch_negative_chunk_size,
                )
        if use_pairwise_training and self._supports_pairwise_auxiliary_loss(self.model):
            self.logger.info(
                "Pairwise training will preserve model-specific auxiliary losses for %s.",
                type(self.model).__name__,
            )
        if self.num_negatives > 0 and self.item_features_df is None:
            self.logger.warning("num_negatives > 0 but item_features_df not loaded. Explicit negative sampling disabled.")
        
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
                    use_pairwise_training=use_pairwise_training,
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
                use_pairwise_training=use_pairwise_training,
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
                use_pairwise_training=use_pairwise_training,
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
        Single training step with explicit negatives and/or in-batch negatives.
        
        Args:
            batch_data: Positive example batch
            item_id_col: Name of item ID column
            torch: Torch module reference
            
        Returns:
            Loss tensor
        """
        batch_dict = dict(batch_data)
        batch_size = len(batch_dict[item_id_col])

        neg_batch_dict = None
        neg_scores_flat = None
        neg_scores = None
        pos_scores = None
        pos_output = None
        aux_loss = None

        if self.num_negatives > 0 or self._supports_pairwise_auxiliary_loss(self.model):
            pos_output = self.model.forward(batch_data)
            if self._supports_pairwise_auxiliary_loss(self.model):
                aux_loss = self._compute_pairwise_auxiliary_loss(pos_output)

        if self.num_negatives > 0:
            if self.negative_sampler is None:
                raise ValueError("Negative sampler not initialized. Call train() after loading item features.")

            pos_scores = self._get_pairwise_scores(pos_output)
            pos_item_ids = batch_dict[item_id_col].cpu().numpy()
            neg_item_ids = self.negative_sampler.sample_negatives_batch(
                pos_item_ids, self.num_negatives
            )

            # Flatten: [B, num_neg] -> [B * num_neg]
            neg_ids_flat = neg_item_ids.reshape(-1)
            neg_features = self.negative_sampler.get_features_by_ids(neg_ids_flat)

            # Build batched negative dict: repeat user features, use negative item features
            neg_batch_dict = {}
            for key, val in batch_dict.items():
                if key == item_id_col:
                    neg_batch_dict[key] = torch.tensor(neg_ids_flat, device=self.model.device)
                elif key in neg_features.columns:
                    val = neg_features[key].to_numpy(copy=False)
                    if not np.isscalar(val[0]):
                        val = np.vstack(val)  # Ensure 2D for multi-valued features
                    neg_batch_dict[key] = torch.tensor(val, device=self.model.device)
                else:
                    if hasattr(val, 'to'):
                        val = val.to(self.model.device)
                        neg_batch_dict[key] = val.repeat_interleave(self.num_negatives, dim=0)
                    else:
                        neg_batch_dict[key] = val

            neg_output = self.model.forward(neg_batch_dict)
            neg_scores_flat = self._get_pairwise_scores(neg_output)
            neg_scores = neg_scores_flat.view(batch_size, self.num_negatives)

        explicit_loss = None
        if neg_scores is not None:
            if self.loss_type == 'bpr':
                explicit_loss = bpr_loss(pos_scores, neg_scores)
            elif self.loss_type == 'margin':
                explicit_loss = margin_ranking_loss(pos_scores, neg_scores, margin=self.margin)
            elif self.loss_type in ('softmax', 'sampled_softmax'):
                explicit_loss = softmax_cross_entropy_loss(pos_scores, neg_scores)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

        if explicit_loss is None and not self.use_in_batch_negatives:
            raise ValueError("Pairwise training requires explicit negatives or use_in_batch_negatives=True.")

        in_batch_loss_weight = 0.0
        if self.use_in_batch_negatives:
            if explicit_loss is not None:
                explicit_loss = 0.5 * explicit_loss
                in_batch_loss_weight = 0.5
            else:
                in_batch_loss_weight = 1.0
            
        loss = None
        if explicit_loss is not None:
            loss = explicit_loss

        # Add per-user diversity loss if enabled
        if self.use_diversity_loss and neg_batch_dict is not None and neg_scores_flat is not None:
            diversity_theta = self.model_params.get('diversity_theta', 0.7)
            diversity_lambda = getattr(self.model, '_diversity_lambda', self.model_params.get('diversity_lambda', 0.01))
            diversity_kernel = self.model_params.get('diversity_kernel', 'gram')
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
            loss = diversity_delta if loss is None else loss + diversity_delta

        if aux_loss is not None:
            loss = aux_loss if loss is None else loss + aux_loss

        # Add regularization
        if hasattr(self.model, 'regularization_loss'):
            reg_loss = self.model.regularization_loss()
            loss = reg_loss if loss is None else loss + reg_loss

        self.model.optimizer.zero_grad()
        total_loss_value = 0.0

        if loss is not None:
            total_loss_value += loss.detach().item()
            loss.backward()

        if in_batch_loss_weight > 0:
            total_loss_value += self._backward_in_batch_loss(
                batch_dict=batch_dict,
                item_id_col=item_id_col,
                loss_weight=in_batch_loss_weight,
            )

        self._finalize_gradient_update(self.model, batch_size)

        return torch.tensor(total_loss_value, device=self.model.device)

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
            if not os.path.exists(self.best_weights_path):
                self.logger.warning(f"No best weights found at {self.best_weights_path}. Using current model state.")
            else:
                self.logger.info(f"Skipping loading best weights as per argument. Using current model state.")
        
        # Ensure item features are loaded
        if self.item_features_df is None:
            raise ValueError("Item features not loaded. Call load_item_features() first.")

        compute_metrics = kwargs.pop('compute_metrics', True)
        evaluate_pool_diversity = kwargs.pop('evaluate_pool_diversity', self.evaluate_pool_diversity)
        
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
            evaluate_pool_diversity=evaluate_pool_diversity,
            inference_batch_size=self.inference_batch_size,
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
        evaluate_pool_diversity = kwargs.pop('evaluate_pool_diversity', self.evaluate_pool_diversity)
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
            evaluate_pool_diversity=evaluate_pool_diversity,
            inference_batch_size=self.inference_batch_size,
            **kwargs
        )
        return metrics
