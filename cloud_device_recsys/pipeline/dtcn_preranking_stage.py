# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
DTCN Pre-ranking Stage Implementation

Integrates the DTCN (Dual Tower Contrastive Network) architecture into the
cloud-side preranking stage. Uses two sub-models:
  - Cloud model (student): FG1 + FG2 features — used for inference
  - Full model (teacher):  FG1 + FG2 + FG3 features — provides CL signal

Supports two training modes:
  1. Pre-trained teacher: Load a frozen full model and train cloud model
     with CL (distance loss).
  2. Joint training: Train both models simultaneously with task loss + CL.

At inference time, only the cloud model is used.
"""

import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any

from .preranking_stage import PrerankingStage
from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from ..models import build_model as registry_build_model
from ..models.losses import bpr_loss, margin_ranking_loss, softmax_cross_entropy_loss

from fuxictr.features import FeatureMap


class DTCNPrerankingStage(PrerankingStage):
    """
    DTCN Pre-ranking stage: cloud-side dual-model training with contrastive learning.

    Extends PrerankingStage with:
    - A second "full model" (teacher) that sees all features (FG1+FG2+FG3)
    - Contrastive learning (distance loss) between teacher and student predictions
    - Two training modes: pre-trained frozen teacher vs joint training
    - Inference uses only the cloud model (student)
    """

    def __init__(self,
                 feature_map: FeatureMap,
                 full_feature_map: FeatureMap,
                 feature_group_manager: FeatureGroupManager,
                 model_params: Dict[str, Any],
                 dtcn_params: Dict[str, Any],
                 allowed_feature_groups: List[FeatureGroup] = None,
                 output_dir: str = "./outputs/preranking",
                 top_k: int = 100,
                 **kwargs):
        """
        Initialize DTCN pre-ranking stage.

        Args:
            feature_map: Cloud feature map (FG1+FG2)
            full_feature_map: Full feature map (FG1+FG2+FG3) for teacher model
            feature_group_manager: Feature group manager
            model_params: Cloud (student) model parameters
            dtcn_params: DTCN-specific parameters including:
                - full_model: model architecture name for full model
                - full_model_params: parameters for full model
                - full_model_path: path to pre-trained full model (optional)
                - freeze_full_model: whether to freeze full model (default True if path provided)
                - cl_loss_weight: weight for CL loss (default 0.1)
                - full_model_loss_weight: weight for full model task loss (default 1.0)
                - cloud_model_loss_weight: weight for cloud model task loss (default 1.0)
            allowed_feature_groups: Allowed feature groups for cloud model
            output_dir: Output directory
            top_k: Number of top candidates to keep
        """
        # Initialize parent PrerankingStage (cloud model)
        super().__init__(
            feature_map=feature_map,
            feature_group_manager=feature_group_manager,
            model_params=model_params,
            allowed_feature_groups=allowed_feature_groups,
            output_dir=output_dir,
            top_k=top_k,
            **kwargs
        )

        # Store full feature map for teacher model
        self.full_feature_map = full_feature_map
        self.dtcn_params = dtcn_params

        # DTCN configuration
        self.full_model_name = dtcn_params.get('full_model', model_params.get('model', 'DINRanker'))
        self.full_model_params = dtcn_params.get('full_model_params', {})
        self.full_model_path = dtcn_params.get('full_model_path', None)
        self.freeze_full_model = dtcn_params.get('freeze_full_model',
                                                  self.full_model_path is not None)

        # CL loss configuration
        self.cl_loss_weight = dtcn_params.get('cl_loss_weight', 0.1)
        self.full_model_loss_weight = dtcn_params.get('full_model_loss_weight', 1.0)
        self.cloud_model_loss_weight = dtcn_params.get('cloud_model_loss_weight', 1.0)

        # Full model instance (built later)
        self.full_model = None
        self.full_best_weights_path = None

        self.logger.info("DTCNPrerankingStage initialized:")
        self.logger.info(f"  Cloud model: {model_params.get('model', 'DINRanker')} (FG1+FG2)")
        self.logger.info(f"  Full model: {self.full_model_name} (FG1+FG2+FG3)")
        self.logger.info(f"  CL loss weight: {self.cl_loss_weight}")
        self.logger.info(f"  Full model path: {self.full_model_path or 'None (train from scratch)'}")
        self.logger.info(f"  Freeze full model: {self.freeze_full_model}")

    def build_model(self):
        """Build both cloud model and full model."""
        # Build cloud model (parent)
        super().build_model()

        # Build full model
        self._build_full_model()

        return self.model

    def _build_full_model(self):
        """Build and optionally load the full (teacher) model."""
        # Merge default params with full model specific params
        full_params = copy.deepcopy(self.model_params)
        full_params.update(self.full_model_params)
        full_params['model'] = self.full_model_name

        self.full_model = registry_build_model(
            model_name=self.full_model_name,
            feature_map=self.full_feature_map,
            model_params=full_params,
            output_dir=os.path.join(self.output_dir, 'full_model'),
        )

        # Save path for full model best weights
        full_model_dir = os.path.join(self.output_dir, 'full_model', self.full_feature_map.dataset_id)
        os.makedirs(full_model_dir, exist_ok=True)
        self.full_best_weights_path = os.path.join(
            self.full_model.model_dir, self.full_model.model_id + ".model"
        )

        # Load pre-trained weights if provided
        if self.full_model_path and os.path.exists(self.full_model_path):
            self.full_model.load_weights(self.full_model_path)
            self.logger.info(f"Loaded pre-trained full model from: {self.full_model_path}")

        # Freeze full model if configured
        if self.freeze_full_model:
            for param in self.full_model.parameters():
                param.requires_grad = False
            self.full_model.eval()
            self.logger.info("Full model parameters frozen (teacher mode)")

        self.logger.info(f"Full model built: {self.full_model_name} "
                         f"({sum(p.numel() for p in self.full_model.parameters())} params)")

    def _compute_cl_loss(self, cloud_logit, full_logit):
        """
        Compute contrastive learning loss (distance loss) between two models.

        Uses MSE loss on pre-sigmoid logits for stronger gradient signal.

        Args:
            cloud_logit: Cloud model logits (pre-sigmoid) [B, 1]
            full_logit: Full model logits (pre-sigmoid) [B, 1]

        Returns:
            CL loss (scalar tensor)
        """
        if cloud_logit is None or full_logit is None:
            return torch.tensor(0.0)

        # Detach full model logits if frozen (no gradient through teacher)
        if self.freeze_full_model:
            full_logit = full_logit.detach()

        return F.mse_loss(cloud_logit, full_logit, reduction='mean')

    def _create_joint_optimizer(self):
        """Create a joint optimizer for both models (when training from scratch)."""
        if self.freeze_full_model:
            # Only optimize cloud model
            params = list(self.model.parameters())
        else:
            # Optimize both models jointly
            params = list(self.model.parameters()) + list(self.full_model.parameters())

        optimizer_name = self.model_params.get('optimizer', 'adam')
        lr = self.model_params.get('learning_rate', 1e-3)

        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=lr)
        elif optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(params, lr=lr)
        elif optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=lr)
        else:
            optimizer = torch.optim.Adam(params, lr=lr)

        return optimizer

    def train(self,
              train_data: Any,
              full_train_data: Any,
              valid_data: Optional[Any] = None,
              **kwargs) -> Dict[str, float]:
        """
        Train the DTCN pre-ranking models with custom joint training loop.

        Args:
            train_data: Training data generator for cloud model (FG1+FG2)
            full_train_data: Training data generator for full model (FG1+FG2+FG3)
            valid_data: Validation data (StageOutput) for evaluation
            **kwargs: Training parameters (epochs, patience, etc.)

        Returns:
            Training metrics
        """
        if self.model is None:
            self.build_model()

        self.logger.info("Starting DTCN pre-ranking training")
        self.best_weights_path = os.path.join(
            self.model.model_dir, self.model.model_id + ".model"
        )

        epochs = kwargs.pop('epochs', 1)
        patience = kwargs.pop('patience', self.patience)
        mode = kwargs.pop('mode', 'max')
        kwargs.pop('learning_rate', None)  # consumed but not used here (LR comes from optimizer)
        metrics = {}

        # Create joint optimizer
        joint_optimizer = self._create_joint_optimizer()

        # Initialize negative sampler for cloud model if needed
        use_negative_sampling = self.num_negatives > 0 and self.item_features_df is not None
        item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')

        if use_negative_sampling:
            from ..data.negative_sampler import NegativeSampler
            self.negative_sampler = NegativeSampler(
                self.item_features_df,
                item_id_col=item_id_col,
            )
            self.logger.info(f"Negative Sampling: {self.num_negatives} negatives, loss_type={self.loss_type}")

        # Setup models for manual training
        max_gradient_norm = kwargs.get("max_gradient_norm",
                                       self.model_params.get("max_gradient_norm", 10.0))
        self.model._total_steps = 0
        self.model._stop_training = False
        self.model._max_gradient_norm = max_gradient_norm
        self.model._verbose = kwargs.get("verbose", 1)
        self.model._epoch_index = 0

        best_metric = -np.inf if mode == "max" else np.inf
        stopping_steps = 0

        for epoch in range(epochs):
            self.model._epoch_index = epoch
            self.logger.info(f"*** [DTCN] Epoch {epoch + 1}/{epochs} ***")

            # Training loop
            self.model.train()
            if not self.freeze_full_model:
                self.full_model.train()

            total_loss = 0.0
            total_cloud_loss = 0.0
            total_full_loss = 0.0
            total_cl_loss = 0.0
            steps = 0

            # Iterate both data loaders in lockstep
            for cloud_batch, full_batch in zip(train_data, full_train_data):
                loss, cloud_loss, full_loss, cl_loss = self._dtcn_train_step(
                    cloud_batch, full_batch,
                    joint_optimizer, max_gradient_norm,
                    use_negative_sampling, item_id_col
                )

                total_loss += loss.item()
                total_cloud_loss += cloud_loss
                total_full_loss += full_loss
                total_cl_loss += cl_loss
                steps += 1

            avg_loss = total_loss / steps if steps > 0 else 0.0
            avg_cloud = total_cloud_loss / steps if steps > 0 else 0.0
            avg_full = total_full_loss / steps if steps > 0 else 0.0
            avg_cl = total_cl_loss / steps if steps > 0 else 0.0
            self.logger.info(
                f"[DTCN] Total Loss: {avg_loss:.6f} | "
                f"Cloud Loss: {avg_cloud:.6f} | "
                f"Full Loss: {avg_full:.6f} | "
                f"CL Loss: {avg_cl:.6f}"
            )

            # Validation — use cloud model only
            self.logger.info(f"[DTCN] Evaluating epoch {epoch + 1}...")
            valid_metrics = self.evaluate(valid_data)
            metrics.update(valid_metrics)
            self.logger.info(f"[DTCN] Validation: {valid_metrics}")

            # Monitor-based best model saving
            curr_val = valid_metrics.get(self.monitor, 0.0)
            is_best = (curr_val > best_metric) if mode == "max" else (curr_val < best_metric)

            if is_best:
                best_metric = curr_val
                stopping_steps = 0
                self.model.save_weights(self.best_weights_path)
                if not self.freeze_full_model:
                    self.full_model.save_weights(self.full_best_weights_path)
                self.logger.info(f"[DTCN] New Best {self.monitor}={curr_val:.6f}! Model Saved.")
            else:
                stopping_steps += 1
                self.logger.info(f"[DTCN] No improvement. Patience {stopping_steps}/{patience}")

                # Decay LR on plateau
                if kwargs.get("reduce_lr_on_plateau", True):
                    factor = kwargs.get("lr_decay_factor", 0.1)
                    for pg in joint_optimizer.param_groups:
                        old_lr = pg['lr']
                        pg['lr'] = old_lr * factor
                    self.logger.info(
                        f"[DTCN] LR decay: {old_lr:.6f} -> {old_lr * factor:.6f}"
                    )

                if stopping_steps >= patience:
                    self.logger.info(f"[DTCN] Early Stopping at epoch {epoch + 1}.")
                    break

        self.logger.info(f"[DTCN] Training complete. Best {self.monitor}={best_metric:.6f}")

        # Restore best cloud model weights
        if os.path.exists(self.best_weights_path):
            self.model.load_weights(self.best_weights_path)
            self.logger.info(f"Restored best cloud model weights from {self.best_weights_path}")

        # Save metrics
        import csv
        metrics_path = os.path.join(self.output_dir, "valid_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}" if isinstance(value, float) else value])

        return metrics

    def _dtcn_train_step(self, cloud_batch, full_batch,
                         optimizer, max_gradient_norm,
                         use_negative_sampling, item_id_col):
        """
        Single DTCN training step: compute cloud loss + full loss + CL loss.

        Args:
            cloud_batch: Batch data for cloud model (FG1+FG2)
            full_batch: Batch data for full model (FG1+FG2+FG3)
            optimizer: Joint optimizer
            max_gradient_norm: Max gradient norm for clipping
            use_negative_sampling: Whether to use negative sampling
            item_id_col: Item ID column name

        Returns:
            Tuple of (total_loss, cloud_loss_val, full_loss_val, cl_loss_val)
        """
        # --- Cloud model forward + loss ---
        if use_negative_sampling and self.num_negatives > 0:
            cloud_loss, cloud_output = self._compute_pairwise_loss(
                self.model, cloud_batch, item_id_col, is_cloud=True
            )
        else:
            cloud_output = self.model.forward(cloud_batch)
            cloud_pred = cloud_output['y_pred']
            cloud_label = self.model.get_labels(cloud_batch)
            cloud_loss = self.model.loss_fn(cloud_pred, cloud_label, reduction='mean')
            if hasattr(self.model, 'regularization_loss'):
                cloud_loss = cloud_loss + self.model.regularization_loss()

        # Extract cloud logits for CL loss
        cloud_logit = cloud_output.get('logit', cloud_output['y_pred'])

        # --- Full model forward + loss ---
        if self.freeze_full_model:
            with torch.no_grad():
                full_output = self.full_model.forward(full_batch)
            full_loss_val = 0.0
            full_loss = torch.tensor(0.0, device=cloud_logit.device)
        else:
            if use_negative_sampling and self.num_negatives > 0:
                full_loss, full_output = self._compute_pairwise_loss(
                    self.full_model, full_batch, item_id_col, is_cloud=False
                )
            else:
                full_output = self.full_model.forward(full_batch)
                full_pred = full_output['y_pred']
                full_label = self.full_model.get_labels(full_batch)
                full_loss = self.full_model.loss_fn(full_pred, full_label, reduction='mean')
                if hasattr(self.full_model, 'regularization_loss'):
                    full_loss = full_loss + self.full_model.regularization_loss()
            full_loss_val = full_loss.item()

        # Extract full logits for CL loss
        full_logit = full_output.get('logit', full_output['y_pred'])

        # --- CL loss (distance loss on logits) ---
        cl_loss = self._compute_cl_loss(cloud_logit, full_logit)

        # --- Combined loss ---
        total_loss = (self.cloud_model_loss_weight * cloud_loss
                      + self.full_model_loss_weight * full_loss
                      + self.cl_loss_weight * cl_loss)

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        # Clip gradients for cloud model
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_gradient_norm)
        if not self.freeze_full_model:
            torch.nn.utils.clip_grad_norm_(self.full_model.parameters(), max_gradient_norm)
        optimizer.step()

        return total_loss, cloud_loss.item(), full_loss_val, cl_loss.item()

    def _compute_pairwise_loss(self, model, batch_data, item_id_col, is_cloud=True):
        """
        Compute pairwise ranking loss (BPR/margin/softmax) with negative sampling.

        Args:
            model: The model to compute loss for
            batch_data: Positive batch data
            item_id_col: Item ID column name
            is_cloud: Whether this is the cloud model (uses self.feature_map)

        Returns:
            Tuple of (loss, positive_predictions)
        """
        batch_dict = dict(batch_data)
        batch_size = len(batch_dict[item_id_col])

        # Get positive item IDs
        pos_item_ids = batch_dict[item_id_col].cpu().numpy()
        neg_item_ids = self.negative_sampler.sample_negatives_batch(
            pos_item_ids, self.num_negatives
        )

        # Positive forward
        pos_output = model.forward(batch_data)
        if self.use_logit:
            pos_scores = pos_output.get('logit', pos_output['y_pred'])
        else:
            pos_scores = pos_output['y_pred']

        # Build negative batch
        neg_ids_flat = neg_item_ids.reshape(-1)
        neg_features = self.negative_sampler.get_features_by_ids(neg_ids_flat)

        neg_batch_dict = {}
        for key, val in batch_dict.items():
            if key == item_id_col:
                neg_batch_dict[key] = torch.tensor(neg_ids_flat, device=model.device)
            elif key in neg_features.columns:
                val_np = neg_features[key].to_numpy(copy=False)
                if not np.isscalar(val_np[0]):
                    val_np = np.vstack(val_np)
                neg_batch_dict[key] = torch.tensor(val_np, device=model.device)
            else:
                if hasattr(val, 'to'):
                    val = val.to(model.device)
                    neg_batch_dict[key] = val.repeat_interleave(self.num_negatives, dim=0)
                else:
                    neg_batch_dict[key] = val

        neg_output = model.forward(neg_batch_dict)
        if self.use_logit:
            neg_scores_flat = neg_output.get('logit', neg_output['y_pred'])
        else:
            neg_scores_flat = neg_output['y_pred']

        neg_scores = neg_scores_flat.view(batch_size, self.num_negatives)

        # Compute pairwise loss
        if self.loss_type == 'bpr':
            loss = bpr_loss(pos_scores, neg_scores)
        elif self.loss_type == 'margin':
            loss = margin_ranking_loss(pos_scores, neg_scores, margin=self.margin)
        elif self.loss_type == 'softmax':
            loss = softmax_cross_entropy_loss(pos_scores, neg_scores)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Add regularization
        if hasattr(model, 'regularization_loss'):
            loss = loss + model.regularization_loss()

        return loss, pos_output
