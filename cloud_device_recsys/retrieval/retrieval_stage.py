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
from ..pipeline.stage_output import StageOutput, CandidateItem, CandidateSet
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
        self.best_weights_path = None
        self.recall_k_values = model_params['recall_k_values']
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
        self.best_weights_path = os.path.join(self.model.model_dir, self.model.model_id + ".model")
        epochs = kwargs.get("epochs", 1)
        patience = kwargs.get("patience", 2)
        monitor = kwargs.get("monitor", "Recall@1000")
        mode = kwargs.get("mode", "max")
        
        self.logger.info(f"Start Training: epochs={epochs}, monitor={monitor}")
        
        best_metric = -np.inf if mode == "max" else np.inf
        stopping_steps = 0

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
                self.model.save_weights(self.best_weights_path)
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
        if os.path.exists(self.best_weights_path):
             self.model.load_weights(self.best_weights_path)

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
            batch_iterator = tqdm(data_generator, disable=True, file=sys.stdout)
            
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
            for batch in tqdm(item_data, disable=True, file=sys.stdout):
                # Get embeddings using model helper
                embs = self.model.get_item_embedding(batch)
                emb_list.append(embs.cpu().numpy())
                id_list.append(dict(batch)[item_id_col].cpu().numpy().reshape(-1))
        self.item_embeddings = np.vstack(emb_list)
        self.item_ids = np.concatenate(id_list) if id_list else np.array([])
        self.item_id_to_idx = {item_id: i for i, item_id in enumerate(self.item_ids)}
        
        self.logger.info(f"Index Built: {self.item_embeddings.shape} items")

    def process(self,
                input_data: Any, # This will be typically a FuxiCTR DataGenerator or a DataFrame with user queries and ground truth
                item_data=None, # FuxiCTR DataGenerator for item features to build index
                **kwargs) -> StageOutput:
        """
        Processes user queries to retrieve top-K candidates from the item pool.
        This method is responsible for generating the 1000 candidate item list
        including positive examples and score-based negative examples.

        Args:
            input_data: A FuxiCTR DataGenerator or similar object containing
                        user queries and their corresponding ground truth labels.
            item_data: A FuxiCTR DataGenerator for item features to build the index.
            **kwargs: Additional parameters.

        Returns:
            StageOutput with refined candidate sets, each containing positive
            examples and score-based negative examples up to self.top_k.
        """
        self.logger.info(f"Starting retrieval process for candidate generation (top_k={self.top_k}).")

        # 1. Ensure model is built
        if self.model is None:
            self.logger.warning("Model not built, building now.")
            self.build_model()

        if os.path.exists(self.best_weights_path):
             self.model.load_weights(self.best_weights_path)
             self.logger.info(f"Loaded best weights from {self.best_weights_path}")
        else:
             self.logger.warning(f"No best weights found at {self.best_weights_path}. Using current model state.")

        if self.item_embeddings is None or self.item_ids is None or self.item_id_to_idx is None:
            self.logger.info("Building item index...")
            if item_data is None:
                raise ValueError("item_data must be provided to build item index for retrieval processing.")
            self.build_item_index(item_data)
        
        self.model.eval() # Set model to evaluation mode
        output = StageOutput(stage_name=self.stage_name)
        
        item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
        impression_id_col = getattr(self.feature_map, 'dataset_config', {}).get('impression_id_col', 'impression_id')

        user_infos = {} # Map request_id to {user_emb, ground_truth_items, user_features, context_features}
        
        # Iterate through input_data to get user embeddings, ground truth, and other user/context features
        self.logger.info("Extracting user embeddings and ground truth from input data...")
        with torch.no_grad():
            for batch_index, batch_data in enumerate(tqdm(input_data, desc="Processing input data", disable=True, file=sys.stdout)):
                batch_dict = dict(batch_data)
                
                # Get user embeddings
                u_emb = self.model.get_user_embedding(batch_data)
                if impression_id_col not in batch_dict:
                    raise ValueError(f"Impression ID column '{impression_id_col}' not found in input data batch.")
                current_request_ids = batch_dict.get(impression_id_col)
                current_gt_item_ids = batch_dict.get(item_id_col)
                for i in range(len(u_emb)):
                    req_id = current_request_ids[i].item() if isinstance(current_request_ids[i], torch.Tensor) else current_request_ids[i]
                    
                    if req_id not in user_infos:
                        user_infos[req_id] = {
                            'user_emb': u_emb[i].cpu().numpy(),
                            'ground_truth_items': set(),
                            'user_features': {},
                        }

                        for feat_name, feat_val in batch_dict.items():
                            if feat_name in [item_id_col, impression_id_col, 'label']:
                                continue
                            
                            val = feat_val[i]
                            if isinstance(val, torch.Tensor):
                                val = val.cpu().numpy()
                            # Convert 0-d array to scalar
                            if np.ndim(val) == 0:
                                val = val.item()
                            user_infos[req_id]['user_features'][feat_name] = val

                    # Add ground truth item if present
                    if len(current_gt_item_ids) > 0:
                        gt_item = current_gt_item_ids[i].item() if isinstance(current_gt_item_ids[i], torch.Tensor) else current_gt_item_ids[i]
                        user_infos[req_id]['ground_truth_items'].add(gt_item)
        
        if not user_infos:
            self.logger.warning("No user information extracted from input data. Returning empty StageOutput.")
            return output

        self.logger.info(f"Scoring {len(user_infos)} unique requests against {len(self.item_ids)} items...")
        
        for req_id, info in tqdm(user_infos.items(), desc="Generating candidates", disable=True, file=sys.stdout):
            u_emb = info['user_emb']
            true_positives = info['ground_truth_items']
            
            # Score this user against all items
            scores = np.dot(u_emb, self.item_embeddings.T).flatten() # (num_items,)
            
            # Create a temporary list of (score, item_id, item_idx_in_pool) tuples for sorting
            scored_items_temp = []
            for item_idx_in_pool, item_score in enumerate(scores):
                item_id = self.item_ids[item_idx_in_pool]
                scored_items_temp.append((item_score, item_id, item_idx_in_pool))
            
            # Sort by score in descending order
            scored_items_temp.sort(key=lambda x: x[0], reverse=True)
            
            # Build the candidate list (positives + score-based negatives)
            candidate_list = []
            added_item_ids = set() # To prevent duplicates
            
            # Add true positives first
            for tp_item_id in true_positives:
                if tp_item_id not in added_item_ids:
                    # Retrieve score for the true positive
                    tp_score = 0.0
                    if tp_item_id in self.item_id_to_idx:
                        tp_score = scores[self.item_id_to_idx[tp_item_id]]
                    candidate_list.append(CandidateItem(item_id=tp_item_id, score=float(tp_score), label=1))
                    added_item_ids.add(tp_item_id)
            
            # Fill the rest with top-scoring items, avoiding duplicates and already-added positives
            for item_score, item_id, _ in scored_items_temp:
                if len(candidate_list) >= self.top_k:
                    break
                if item_id not in added_item_ids:
                    candidate_list.append(CandidateItem(item_id=item_id, score=float(item_score), label=0))
                    added_item_ids.add(item_id)
            
            # Create CandidateSet for this request
            cs = CandidateSet(
                request_id=req_id,
                candidates=candidate_list,
                user_features=info['user_features'],
                source_stage=self.stage_name
            )
            output.candidate_sets.append(cs)
            
        self.logger.info(f"Generated {output.get_total_candidates()} candidates across {len(output.candidate_sets)} requests.")
        output.end_time = datetime.datetime.now().isoformat()
        return output

    def evaluate(self, test_data, k_values=None, **kwargs) -> Dict[str, float]:
        """Compute Recall@K against built item index"""
        self.logger.info("Evaluating...")
        self.model.eval()
        if k_values is None:
            k_values = self.recall_k_values
        
        user_embs, gt_ids = [], []
        config = getattr(self.feature_map, 'dataset_config', {})
        item_id_col = config.get('item_id_col', 'cand_item_id')
        self.logger.info(f"Evaluation Start. ID Col: {item_id_col}")
        with torch.no_grad():
            for batch in tqdm(test_data, disable=True, file=sys.stdout):
                u_emb = self.model.get_user_embedding(batch)
                user_embs.append(u_emb.cpu().numpy())
                gt_ids.append(dict(batch)[item_id_col].cpu().numpy().reshape(-1))
        user_embs = np.vstack(user_embs)
        ground_truth = np.concatenate(gt_ids) if gt_ids else np.zeros((len(user_embs),))
        
        if self.item_id_to_idx is None: # Ensure item_id_to_idx is built
            if self.item_ids is None:
                raise RuntimeError("item_ids and item_id_to_idx not built. Call build_item_index first.")
            self.item_id_to_idx = {uid: i for i, uid in enumerate(self.item_ids)}
            self.logger.warning("item_id_to_idx was not pre-built, built it dynamically for evaluation.")

        # Convert ground truth to indices
        gt_indices = np.array([self.item_id_to_idx.get(gid, -1) for gid in ground_truth])

        # Retrieve and Metric
        chunk_size = kwargs.get("chunk_size", 50000)
        total_recall = {k: 0.0 for k in k_values}
        n_samples = len(user_embs)
        
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            u_chunk = user_embs[i:end]
            gt_chunk = gt_indices[i:end]
            
            scores = np.dot(u_chunk, self.item_embeddings.T)
            
            # Safe Top-K
            current_pool_size = scores.shape[1]
            current_max_k = min(max(k_values), current_pool_size)
            
            if current_max_k == current_pool_size:
                # Full sort if K is entire pool
                topk_idx = np.argsort(-scores, axis=1)
            else:
                partition_k = min(current_max_k, current_pool_size - 1)
                topk_idx = np.argpartition(-scores, partition_k, axis=1)[:, :current_max_k]
            
            # Sort top K for ranking metrics if needed (though recall is set-based, we sort for safety/future)
            top_scores = np.take_along_axis(scores, topk_idx, axis=1)
            sorted_args = np.argsort(-top_scores, axis=1)
            topk_sorted = np.take_along_axis(topk_idx, sorted_args, axis=1)
            
            for k in k_values:
                if k > current_pool_size:
                    k_eff = current_pool_size
                else:
                    k_eff = k
                
                preds_k = topk_sorted[:, :k_eff]
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
