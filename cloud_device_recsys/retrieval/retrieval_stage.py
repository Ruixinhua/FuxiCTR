# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
Retrieval Stage Implementation

This module wraps the DualTowerRetrieval model as a pipeline stage.
"""

import os
import sys
import csv
import numpy as np
import datetime
from typing import Dict, List, Optional, Any
import torch
from tqdm import tqdm

from ..pipeline.base_stage import BaseStage, StageType
from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from .models.dual_tower_retrieval import DualTowerRetrieval

from fuxictr.features import FeatureMap


class RetrievalStage(BaseStage):
    """
    Retrieval stage for candidate generation.
    
    Uses dual-tower model to retrieve top-K candidates from item pool.
    Only uses FG1 (non-personalized) and FG2 (cloud-personalized) features.
    """
    
    def __init__(self,
                 feature_map: FeatureMap,
                 feature_group_manager: FeatureGroupManager,
                 allowed_feature_groups,
                 model_params: Dict[str, Any],
                 output_dir: str = "./outputs/retrieval",
                 top_k: int = 1000,
                 **kwargs):
        """
        Initialize retrieval stage.
        
        Args:
            feature_map: FuxiCTR FeatureMap
            feature_group_manager: Feature group manager
            allowed_feature_groups: allowed feature groups
            model_params: Parameters for DualTowerRetrieval model
            output_dir: Output directory
            top_k: Number of candidates to retrieve
        """
        super().__init__(
            stage_name="retrieval",
            stage_type=StageType.RETRIEVAL,
            feature_group_manager=feature_group_manager,
            allowed_feature_groups=allowed_feature_groups,
            output_dir=output_dir,
            **kwargs
        )
        # Ensure output directory exists (BaseModel expects it)
        os.makedirs(os.path.join(output_dir, feature_map.dataset_id), exist_ok=True)
        
        # Filter feature_map to only include allowed features
        # This fixes the issue where device features (FG3) were included in User Tower
        self.feature_map = self._filter_feature_map(feature_map, feature_group_manager)
        self.top_k = top_k
        self.model_params = model_params
        self.model: Optional[DualTowerRetrieval] = None
        
        # Item index for retrieval
        self.item_embeddings: Optional[np.ndarray] = None
        self.item_ids: Optional[List[Any]] = None

    def _filter_feature_map(self, feature_map: FeatureMap, fg_manager: FeatureGroupManager) -> FeatureMap:
        """Create a new FeatureMap with only allowed features"""
        import copy
        from collections import OrderedDict
        
        new_fm = copy.deepcopy(feature_map)
        new_features = OrderedDict()

        for name, spec in feature_map.features.items():
            # Check if feature belongs to allowed groups
            group = fg_manager.feature_assignments.get(name)
            is_allowed = False
            for allowed_grp in self.allowed_feature_groups:
                # Compare enum members directly if possible, or string representation
                if group == allowed_grp or str(group) == str(allowed_grp):
                    is_allowed = True
                    break
            
            # Use 'label' and 'score' always if present (needed for training/indexing)
            if name in ['label', 'score', 'impression_id', 'group_id']:
                is_allowed = True
                
            if is_allowed:
                new_features[name] = spec
                
        new_fm.features = new_features
        # Re-set indices
        new_fm.set_column_index()
        return new_fm
    
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
        
        # Add unique timestamp to model_id to prevent overwrites
        base_model_id = params.get("model_id", "DualTowerRetrieval")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        params["model_id"] = f"{base_model_id}_{timestamp}"
        
        self.model = DualTowerRetrieval(
            self.feature_map,
            **params
        )
        self.logger.info("Built DualTowerRetrieval model")
        self.model.count_parameters()
        return self.model
    
    def train(self, train_data, valid_data=None, item_data=None, **kwargs):
        """Train loop with validation"""
        if self.model is None:
            self.build_model()
        
        epochs = kwargs.get("epochs", 1)
        patience = kwargs.get("patience", 2)
        monitor = kwargs.get("monitor", "Recall@1000")
        mode = kwargs.get("mode", "max")
        
        self.logger.info(f"Start Training: epochs={epochs}, monitor={monitor}")
        
        best_metric = -np.inf if mode == "max" else np.inf
        stopping_steps = 0
        best_weights_path = os.path.join(self.model.model_dir, self.model.model_id + ".model")
        
        # Setup model for manual training
        self.model._total_steps = 0
        self.model._stop_training = False
        self.model._max_gradient_norm = kwargs.get("max_gradient_norm", 10.0)
        self.model._verbose = kwargs.get("verbose", 1)
        self.model._eval_steps = 1e9
        
        
        for epoch in range(epochs):
            self.model._epoch_index = epoch
            self.logger.info(f"*** Epoch {epoch+1} ***")
            
            self.train_epoch(train_data)
            
            # Validation
            self.logger.info("Building Item Index...")
            self.build_item_index(item_data)

            metrics = self.evaluate(valid_data)
            curr_val = metrics.get(monitor, 0.0)
            is_best = (curr_val > best_metric) if mode == "max" else (curr_val < best_metric)

            if is_best:
                best_metric = curr_val
                stopping_steps = 0
                self.model.save_weights(best_weights_path)
                self.logger.info(f"New Best {monitor}! Model Saved.")
            else:
                stopping_steps += 1
                self.logger.info(f"No improve. Patience {stopping_steps}/{patience}")
                
                # Decay LR on plateau
                if kwargs.get("reduce_lr_on_plateau", True):
                    old_lr = self.model.optimizer.param_groups[0]['lr']
                    new_lr = self.model.lr_decay(factor=kwargs.get("lr_decay_factor", 0.1))
                    self.logger.info(f"Decay LR: {old_lr:.6f} -> {new_lr:.6f}")
                
                if stopping_steps >= patience:
                    self.logger.info("Early Stopping.")
                    break

        # Restore best
        if os.path.exists(best_weights_path):
             self.model.load_weights(best_weights_path)

    def train_epoch(self, data_generator):
        """
        Train the model for one epoch.
        Reference: fuxictr/pytorch/models/rank_model.py
        """
        self.model.train()
        train_loss = 0
        total_batches = 0
        
        if self.model._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
            
        for batch_index, batch_data in enumerate(batch_iterator):
            self.model._batch_index = batch_index
            self.model._total_steps += 1
            
            loss = self.model.train_step(batch_data)
            train_loss += loss.item()
            total_batches += 1
            if self.model._stop_training:
                break
        
        if total_batches > 0:
            avg_loss = train_loss / total_batches
             # Check if regularization is actually working/enabled
            reg_loss = self.model.regularization_loss() if hasattr(self.model, 'regularization_loss') else 0
            self.logger.info(f"Train loss: {avg_loss:.6f} (Reg Loss: {reg_loss:.6f})")

    def build_item_index(self, item_data):
        """Build item embeddings index from iterator"""
        self.model.eval()
        
        emb_list, id_list = [], []
        # Get IDs if available, else sequential
        config = getattr(self.feature_map, 'dataset_config', {})
        item_id_col = config.get('item_id_col', 'cand_item_id')
        with torch.no_grad():
            for batch in tqdm(item_data, file=sys.stdout):
                # Get embeddings using model helper
                embs = self.model.get_item_embedding(batch)
                emb_list.append(embs.cpu().numpy())
                id_list.append(dict(batch)[item_id_col].cpu().numpy().reshape(-1))
        self.item_embeddings = np.vstack(emb_list)
        self.item_ids = np.concatenate(id_list) if id_list else np.array([])
        
        self.logger.info(f"Index Built: {self.item_embeddings.shape} items")

    def evaluate(self, test_data, k_values=[1000, 5000, 10000], **kwargs) -> Dict[str, float]:
        """Compute Recall@K against built item index"""
        self.logger.info("Evaluating...")
        self.model.eval()
        
        user_embs, gt_ids = [], []
        config = getattr(self.feature_map, 'dataset_config', {})
        item_id_col = config.get('item_id_col', 'cand_item_id')
        self.logger.info(f"Evaluation Start. ID Col: {item_id_col}")
        with torch.no_grad():
            for batch in tqdm(test_data, file=sys.stdout):
                u_emb = self.model.get_user_embedding(batch)
                user_embs.append(u_emb.cpu().numpy())
                gt_ids.append(dict(batch)[item_id_col].cpu().numpy().reshape(-1))
        user_embs = np.vstack(user_embs)
        ground_truth = np.concatenate(gt_ids) if gt_ids else np.zeros((len(user_embs),))
        
        # Map item IDs to index positions in self.item_embeddings
        item_id_to_idx = {uid: i for i, uid in enumerate(self.item_ids)}
        
        # Convert ground truth to indices
        # Optimization: use numpy vectorized map? Dict lookup faster via list comp
        gt_indices = np.array([item_id_to_idx.get(gid, -1) for gid in ground_truth])
        
        # Retrieve and Metric
        # chunk_size should be small enough to avoid OOM
        # 5000 * 200k * 4 bytes approx 4GB. Safe.
        chunk_size = kwargs.get("chunk_size", 50000)
        total_recall = {k: 0.0 for k in k_values}
        n_samples = len(user_embs)
        # user_embs = torch.from_numpy(user_embs).to(self.model.device)
        # item_embeddings = torch.from_numpy(self.item_embeddings).to(self.model.device)
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            u_chunk = user_embs[i:end]
            gt_chunk = gt_indices[i:end]
            
            # Score: (B, Dim) @ (pool, Dim).T -> (B, pool)
            # Memory warning: chunk_size=5000, pool=200k => 1B floats = 4GB. Safe.
            scores = np.dot(u_chunk, self.item_embeddings.T)
            # scores = self.model.cal_similarity(u_chunk, item_embeddings)
            
            max_k = max(k_values)
            
            # Fast Top-K using argpartition
            # We need top max_k. 
            topk_idx = np.argpartition(-scores, max_k, axis=1)[:, :max_k]
            
            # Argpartition is unsorted. Sort it for correct K-slicing
            top_scores = np.take_along_axis(scores, topk_idx, axis=1)
            sorted_args = np.argsort(-top_scores, axis=1)
            topk_sorted = np.take_along_axis(topk_idx, sorted_args, axis=1)
            
            for k in k_values:
                preds_k = topk_sorted[:, :k]
                # Broadcasting check: (B, K) == (B, 1)
                hits = (preds_k == gt_chunk[:, None]).sum(axis=1)
                total_recall[k] += (hits > 0).sum()
                
        metrics = {f"Recall@{k}": v / n_samples for k, v in total_recall.items()}
        self.logger.info(f"Validation: {metrics}")
        # Save evaluation metrics to CSV
        metrics_path = os.path.join(self.output_dir, "eval_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}"])
        return metrics
