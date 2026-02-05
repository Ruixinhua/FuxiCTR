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
from ..config.feature_groups import FeatureGroupManager
from ..models import build_model as registry_build_model
from ..models import DualTowerRetrieval  # For type hints
from ..utils import filter_feature_map

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
        use_feature_encoder = model_params.get("use_feature_encoder", False)
        self.feature_map = filter_feature_map(feature_map, feature_group_manager, self.allowed_feature_groups,
                                              use_feature_encoder=use_feature_encoder)
        self.logger.info(f"Use feature encoder: {use_feature_encoder}")
        self.top_k = top_k
        self.model_params = model_params
        self.model: Optional[DualTowerRetrieval] = None
        self.best_weights_path = None
        self.metrics_k = model_params['metrics_k']
        self.monitor = model_params.get('monitor', 'Recall@1000')
        # Item index for retrieval
        self.item_embeddings: Optional[np.ndarray] = None
        self.item_ids: Optional[List[Any]] = None

    def build_model(self) -> DualTowerRetrieval:
        """Build and initialize the retrieval model using unified registry"""
        # Get model name from config, default to DualTowerRetrieval
        model_name = self.model_params.get('model', 'DualTowerRetrieval')
        
        self.model = registry_build_model(
            model_name=model_name,
            feature_map=self.feature_map,
            model_params=self.model_params,
            output_dir=self.output_dir,
        )
        self.logger.info(f"Built {model_name} model")
        return self.model
    
    def train(self, train_data, valid_data=None, item_data=None, **kwargs):
        """Train loop with validation"""
        if self.model is None:
            self.build_model()
        self.best_weights_path = os.path.join(self.model.model_dir, self.model.model_id + ".model")
        epochs = kwargs.get("epochs", 1)
        patience = kwargs.get("patience", 2)
        mode = kwargs.get("mode", "max")
        
        self.logger.info(f"Start Training: epochs={epochs}, monitor={self.monitor}")
        
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
            curr_val = metrics.get(self.monitor, 0.0)
            is_best = (curr_val > best_metric) if mode == "max" else (curr_val < best_metric)

            if is_best:
                best_metric = curr_val
                stopping_steps = 0
                self.model.save_weights(self.best_weights_path)
                self.logger.info(f"New Best {self.monitor}! Model Saved.")
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

    def _retrieve_and_score(
        self,
        input_data: Any,
        item_data: Any = None,
        return_output: bool = True,
        compute_metrics: bool = False,
        metrics_k: List[int] = None,
        **kwargs
    ) -> tuple:
        """
        Unified retrieval and scoring method.
        
        This is the core method that handles both candidate generation (process)
        and metrics computation (evaluate) in a single pass.
        
        Args:
            input_data: FuxiCTR DataGenerator with user queries
            item_data: Item data for building index (if needed)
            return_output: Whether to return StageOutput with candidates
            compute_metrics: Whether to compute Recall@K metrics
            metrics_k: K values for Recall@K computation
            **kwargs: Additional parameters (chunk_size, etc.)
            
        Returns:
            Tuple of (StageOutput or None, metrics dict or None)
        """
        # 1. Ensure model is built and weights loaded
        if self.model is None:
            raise RuntimeError("No model found. Train model first!")

        # 2. Build item index if needed
        if self.item_embeddings is None or self.item_ids is None:
            if item_data is None:
                raise ValueError("item_data must be provided to build item index.")
            self.logger.info("Building item index...")
            self.build_item_index(item_data)
        
        if self.item_id_to_idx is None:
            self.item_id_to_idx = {item_id: i for i, item_id in enumerate(self.item_ids)}
        
        self.model.eval()
        metrics_k = self.metrics_k if metrics_k is None else metrics_k
        total_recall = {k: 0.0 for k in metrics_k} if compute_metrics else None
        
        # Setup output
        output = StageOutput(stage_name=self.stage_name) if return_output else None
        
        item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
        impression_id_col = getattr(self.feature_map, 'dataset_config', {}).get('impression_id_col', 'impression_id')

        # =====================================================================
        # Phase 1: Batch extract user embeddings and metadata
        # =====================================================================
        request_ids_list = []
        user_embs_list = []
        ground_truths_list = []  # List of sets for process, list of single items for evaluate
        user_features_list = [] if return_output else None
        
        # Use a dict to deduplicate and aggregate ground truth per request
        request_info_temp = {}
        
        self.logger.info("Extracting user embeddings and ground truth from input data...")
        with torch.no_grad():
            for batch_data in tqdm(input_data, desc="Extracting embeddings", disable=True, file=sys.stdout):
                batch_dict = dict(batch_data)
                
                u_emb = self.model.get_user_embedding(batch_data)
                current_request_ids = batch_dict.get(impression_id_col)
                current_gt_item_ids = batch_dict.get(item_id_col)
                
                for i in range(len(u_emb)):
                    req_id = current_request_ids[i].item() if isinstance(current_request_ids[i], torch.Tensor) else current_request_ids[i]
                    gt_item = current_gt_item_ids[i].item() if isinstance(current_gt_item_ids[i], torch.Tensor) else current_gt_item_ids[i]
                    
                    if req_id not in request_info_temp:
                        emb_idx = len(request_ids_list)
                        request_ids_list.append(req_id)
                        user_embs_list.append(u_emb[i].cpu().numpy())
                        
                        # Extract user features only if returning output
                        if return_output:
                            user_features = {}
                            for feat_name, feat_val in batch_dict.items():
                                if feat_name in [item_id_col, impression_id_col, 'label']:
                                    continue
                                if feat_name not in self.feature_group_manager.get_user_features():
                                    continue
                                val = feat_val[i]
                                if isinstance(val, torch.Tensor):
                                    val = val.cpu().numpy()
                                if np.ndim(val) == 0:
                                    val = val.item()
                                user_features[feat_name] = val
                            user_features_list.append(user_features)
                        
                        request_info_temp[req_id] = {
                            'emb_idx': emb_idx,
                            'ground_truth': set()
                        }
                    
                    # Add ground truth item
                    request_info_temp[req_id]['ground_truth'].add(gt_item)
        
        if not request_ids_list:
            self.logger.warning("No user information extracted from input data.")
            return output, {} if compute_metrics else None
        
        # Stack all user embeddings
        user_embs = np.vstack(user_embs_list)
        num_requests = len(request_ids_list)
        num_items = len(self.item_ids)
        
        # Build ground truth list aligned with request order
        for req_id in request_ids_list:
            ground_truths_list.append(request_info_temp[req_id]['ground_truth'])
        
        self.logger.info(f"Scoring {num_requests} unique requests against {num_items} items...")
        
        # =====================================================================
        # Phase 2: Batch scoring with chunked processing
        # =====================================================================
        chunk_size = kwargs.get('chunk_size', 5000)
        
        # Determine fetch_k based on what we need
        max_positives = max(len(gt) for gt in ground_truths_list) if ground_truths_list else 0
        if compute_metrics:
            max_k_for_metrics = max(metrics_k)
            fetch_k = min(max(self.top_k + max_positives, max_k_for_metrics), num_items)
        else:
            fetch_k = min(self.top_k + max_positives, num_items)
        
        for chunk_start in range(0, num_requests, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_requests)
            user_chunk = user_embs[chunk_start:chunk_end]
            chunk_len = chunk_end - chunk_start
            
            # Batch matrix multiplication
            scores = np.dot(user_chunk, self.item_embeddings.T)
            
            # Efficient top-K selection using np.argpartition
            if fetch_k < num_items:
                partition_k = min(fetch_k, num_items - 1)
                topk_indices = np.argpartition(-scores, partition_k, axis=1)[:, :fetch_k]
            else:
                topk_indices = np.tile(np.arange(num_items), (chunk_len, 1))
            
            # Get scores for top-K indices and sort them
            topk_scores = np.take_along_axis(scores, topk_indices, axis=1)
            sorted_order = np.argsort(-topk_scores, axis=1)
            topk_indices_sorted = np.take_along_axis(topk_indices, sorted_order, axis=1)
            topk_scores_sorted = np.take_along_axis(topk_scores, sorted_order, axis=1)
            
            # Process each user in chunk
            for i in range(chunk_len):
                global_idx = chunk_start + i
                req_id = request_ids_list[global_idx]
                true_positives = ground_truths_list[global_idx]
                
                user_topk_indices = topk_indices_sorted[i]
                user_topk_scores = topk_scores_sorted[i]
                
                # Compute metrics if requested
                if compute_metrics:
                    # Convert ground truth to indices for this user
                    gt_indices_set = {self.item_id_to_idx.get(gt_id, -1) for gt_id in true_positives}
                    gt_indices_set.discard(-1)  # Remove invalid indices
                    
                    for k in metrics_k:
                        k_eff = min(k, len(user_topk_indices))
                        preds_k = set(user_topk_indices[:k_eff].tolist())
                        if preds_k & gt_indices_set:  # Intersection check
                            total_recall[k] += 1
                
                # Build candidate set if returning output
                if return_output:
                    user_features = user_features_list[global_idx]
                    
                    candidate_list = []
                    added_item_ids = set()
                    
                    # Add true positives first (with their actual scores)
                    for tp_item_id in true_positives:
                        if tp_item_id not in added_item_ids:
                            tp_score = 0.0
                            if tp_item_id in self.item_id_to_idx:
                                tp_idx = self.item_id_to_idx[tp_item_id]
                                tp_score = scores[i, tp_idx]
                            candidate_list.append(CandidateItem(item_id=tp_item_id, score=float(tp_score), label=1))
                            added_item_ids.add(tp_item_id)
                    
                    # Fill with top-K items
                    for j in range(len(user_topk_indices)):
                        if len(candidate_list) >= self.top_k:
                            break
                        item_idx = user_topk_indices[j]
                        item_id = self.item_ids[item_idx]
                        if item_id not in added_item_ids:
                            candidate_list.append(CandidateItem(
                                item_id=item_id,
                                score=float(user_topk_scores[j]),
                                label=0
                            ))
                            added_item_ids.add(item_id)
                    
                    cs = CandidateSet(
                        request_id=req_id,
                        candidates=candidate_list,
                        user_features=user_features,
                        source_stage=self.stage_name
                    )
                    output.candidate_sets.append(cs)
        
        # Finalize metrics
        metrics = None
        if compute_metrics:
            metrics = {f"Recall@{k}": v / num_requests for k, v in total_recall.items()}
            self.logger.info(f"Metrics: {metrics}")
            
            # Save metrics to CSV
            metrics_path = os.path.join(self.output_dir, "eval_metrics.csv")
            with open(metrics_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['metric_name', 'value'])
                for name, value in sorted(metrics.items()):
                    writer.writerow([name, f"{value:.6f}"])
        
        # Finalize output
        if return_output:
            self.logger.info(f"Generated {output.get_total_candidates()} candidates across {len(output.candidate_sets)} requests.")
            output.end_time = datetime.datetime.now().isoformat()
            if metrics:
                output.metadata['metrics'] = metrics
        
        return output, metrics

    def process(self,
                input_data: Any,
                item_data=None,
                **kwargs):
        """
        Process user queries to retrieve top-K candidates from the item pool.
        
        Optionally computes evaluation metrics during processing.
        
        Args:
            input_data: FuxiCTR DataGenerator with user queries
            item_data: FuxiCTR DataGenerator for item features (to build index)
            **kwargs: Additional parameters (chunk_size, etc.)
            
        Returns:
            StageOutput with candidate sets
        """
        self.logger.info(f"Starting retrieval process for candidate generation (top_k={self.top_k}).")
        
        compute_metrics = kwargs.pop('compute_metrics', True)
        metrics_k = kwargs.pop('metrics_k', self.metrics_k)
        
        output, metrics = self._retrieve_and_score(
            input_data=input_data,
            item_data=item_data,
            return_output=True,
            compute_metrics=compute_metrics,
            metrics_k=metrics_k,
            **kwargs
        )
        
        return output, metrics

    def evaluate(self, test_data, metrics_k=None, **kwargs) -> Dict[str, float]:
        """
        Compute Recall@K metrics against built item index.
        
        This is a lightweight version that only computes metrics without
        generating the full StageOutput.
        
        Args:
            test_data: FuxiCTR DataGenerator with test data
            metrics_k: List of K values for Recall@K
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of metric names to values
        """
        self.logger.info("Evaluating...")
        
        _, metrics = self._retrieve_and_score(
            input_data=test_data,
            item_data=None,  # Assume index already built
            return_output=False,
            compute_metrics=True,
            metrics_k=metrics_k,
            **kwargs
        )
        
        return metrics or {}

