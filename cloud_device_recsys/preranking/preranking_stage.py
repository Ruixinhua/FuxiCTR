# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Pre-ranking Stage Implementation

This module wraps the LightweightRanker model as a pipeline stage.
"""

import os
import csv
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import torch
from tqdm import tqdm # Added tqdm import

from ..pipeline.base_stage import BaseStage, StageType
from ..pipeline.stage_output import StageOutput, CandidateSet, CandidateItem
from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from .models.din_ranker import DINRanker

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
            model_params: Parameters for LightweightRanker model
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
        
        self.feature_map = feature_map
        self.top_k = top_k
        self.use_diversity = use_diversity
        self.diversity_weight = diversity_weight
        self.model_params = model_params
        self.model: Optional[DINRanker] = None
    
    def build_model(self) -> DINRanker:
        """Build and initialize the pre-ranking model"""
        # Add default FuxiCTR required parameters
        default_params = {
            'verbose': 1,
            'model_root': self.output_dir,
            'metrics': ['AUC', 'logloss'],
            'embedding_dim': 32,
            'gpu': -1,
            'optimizer': 'adam',
            'loss': 'binary_crossentropy',
        }
        params = {**default_params, **self.model_params}
        
        self.model = DINRanker(
            self.feature_map,
            use_diversity_loss=self.use_diversity,
            diversity_weight=self.diversity_weight,
            **params
        )
        self.logger.info("Built LightweightRanker model")
        self.model.count_parameters()
        return self.model
    
    def train(self,
              train_data: Any,
              valid_data: Optional[Any] = None,
              **kwargs) -> Dict[str, float]:
        """
        Train the pre-ranking model.
        
        Args:
            train_data: Training data generator
            valid_data: Validation data generator
            **kwargs: Training parameters
            
        Returns:
            Training metrics
        """
        if self.model is None:
            self.build_model()
        
        self.logger.info("Starting pre-ranking model training")
        self.model.fit(train_data, validation_data=valid_data, **kwargs)
        
        metrics = {}
        if valid_data is not None:
            valid_result = self.model.evaluate(valid_data)
            metrics.update(valid_result)
            self.logger.info(f"Validation metrics: {valid_result}")
        
        # Save metrics to CSV
        metrics_path = os.path.join(self.output_dir, "training_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}" if isinstance(value, float) else value])
        
        return metrics
    
    def _convert_features_to_tensors(self, feature_dict: Dict[str, Any], is_user_context: bool = False, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """
        Converts a dictionary of feature values (from CandidateSet/CandidateItem)
        into a dictionary of torch.Tensors, compatible with FuxiCTR models.
        Handles sequence features and uses feature_map for types and vocab.
        
        Args:
            feature_dict: Dictionary where keys are feature names and values are raw feature data.
            is_user_context: True if processing user/context features (may contain sequences).
            batch_size: Expected batch size (1 for single user/candidate item).

        Returns:
            A dictionary of torch.Tensors.
        """
        tensor_dict = {}
        for feature_name, raw_value in feature_dict.items():
            if feature_name not in self.feature_map.features:
                self.logger.debug(f"Feature '{feature_name}' not found in feature_map, skipping.")
                continue

            feature_spec = self.feature_map.features[feature_name]
            feature_type = feature_spec.get('type')

            if feature_type == 'categorical':
                # For categorical features, we expect a single value (int) or list of ints for multi-hot
                if isinstance(raw_value, list):
                    # For multi-hot, pad to max_len if needed. FuxiCTR handles this through DataGenerator.
                    # For now, treat as single value if list, or take first element.
                    # A more robust solution involves creating a specific DataGenerator/Dataset for inference.
                    # For a simple scalar categorical, just convert to tensor.
                    value = torch.tensor([raw_value[0]] if raw_value else [0], dtype=torch.long)
                else:
                    value = torch.tensor([raw_value], dtype=torch.long)
            elif feature_type == 'sequence':
                if not is_user_context:
                    self.logger.warning(f"Sequence feature '{feature_name}' found in candidate item. This is unexpected for DIN.")
                    continue
                # For sequence features, raw_value is expected to be a list of item IDs (int)
                # Pad sequence to max_len as specified in dataset_config.yaml
                max_len = feature_spec.get('max_len', 50) # Default max_len
                if not isinstance(raw_value, list):
                    raw_value = [raw_value] # Wrap single value in list for consistency
                
                padded_sequence = raw_value[:max_len] + [0] * (max_len - len(raw_value)) # Pad with 0
                value = torch.tensor([padded_sequence], dtype=torch.long) # Add batch dimension
            elif feature_type == 'numeric':
                # Numeric features are typically floats
                value = torch.tensor([raw_value], dtype=torch.float)
            else:
                self.logger.warning(f"Unsupported feature type '{feature_type}' for feature '{feature_name}'. Skipping.")
                continue
            
            # Ensure tensor has a batch dimension (e.g., [1, ...])
            if value.ndim == 1:
                value = value.unsqueeze(0) # Add batch dimension if missing
            
            tensor_dict[feature_name] = value.to(self.model.device)

        return tensor_dict

    def process(self,
                input_data: StageOutput,
                **kwargs) -> StageOutput:
        """
        Process retrieval candidates to produce refined set using DIN.
        Generates 100 candidates for the next stage (reranking).
        
        Args:
            input_data: Output from retrieval stage
            **kwargs: Additional parameters
            
        Returns:
            StageOutput with refined candidate sets
        """
        if self.model is None:
            self.logger.warning("Model not built, building with default params for testing...")
            self.build_model()
        
        # Load best weights if available
        best_weights_path = os.path.join(self.model.model.model_dir, self.model.model.model_id + ".model")
        if os.path.exists(best_weights_path):
             self.model.model.load_weights(best_weights_path)
             self.logger.info(f"Loaded best weights from {best_weights_path}")
        else:
             self.logger.warning(f"No best weights found at {best_weights_path}. Using current model state.")

        if input_data is None or len(input_data.candidate_sets) == 0:
            self.logger.warning("No input from previous stage. Creating dummy output for testing...")
            output = StageOutput(stage_name=self.stage_name)
            cs = CandidateSet(request_id="test_req_0", user_id=0, source_stage=self.stage_name)
            cs.add_candidate(item_id=0, score=1.0)
            output.candidate_sets.append(cs)
            return output
        
        output = StageOutput(stage_name=self.stage_name)
        self.model.eval() # Set model to evaluation mode
        
        self.logger.info(f"Processing {len(input_data.candidate_sets)} candidate sets from retrieval.")
        
        item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
        
        for cs in tqdm(input_data.candidate_sets, desc="Pre-ranking candidates", file=sys.stdout):
            if len(cs.candidates) == 0:
                output.candidate_sets.append(CandidateSet(
                    request_id=cs.request_id,
                    user_id=cs.user_id,
                    user_features=cs.user_features.copy(),
                    context_features=cs.context_features.copy(),
                    source_stage=self.stage_name
                ))
                continue
            
            # 1. Prepare User Features
            user_features_for_model = self._convert_features_to_tensors({**cs.user_features, **cs.context_features}, is_user_context=True)
            
            # 2. Prepare Candidate Item Features
            candidate_item_features_list = []
            original_candidate_items = [] # Store original CandidateItem objects to get labels
            
            for candidate_item in cs.candidates:
                item_feat_dict = candidate_item.features.copy()
                item_feat_dict[item_id_col] = candidate_item.item_id # Ensure item_id is present for DIN target
                candidate_item_features_list.append(item_feat_dict)
                original_candidate_items.append(candidate_item) # Keep track of original objects

            # Convert list of dicts to dict of lists, then to dict of tensors
            # Example: [{'a':1, 'b':10}, {'a':2, 'b':20}] -> {'a':[1,2], 'b':[10,20]} -> {'a':tensor, 'b':tensor}
            candidate_features_batched = {}
            if candidate_item_features_list:
                for key in candidate_item_features_list[0].keys():
                    raw_values = [d[key] for d in candidate_item_features_list if key in d]
                    # For sequence features passed as candidate features, handle as regular categorical/numeric
                    # For now, assuming candidate features are not sequences in a way DIN expects for user behavior
                    if self.feature_map.features.get(key, {}).get('type') == 'sequence':
                         self.logger.warning(f"Sequence feature '{key}' found in candidate item features. Treating as categorical (taking first element if list).")
                         raw_values = [v[0] if isinstance(v, list) else v for v in raw_values]

                    # Convert raw_values list to tensor
                    if self.feature_map.features.get(key, {}).get('type') == 'numeric':
                        candidate_features_batched[key] = torch.tensor(raw_values, dtype=torch.float).to(self.model.device)
                    else: # Assume categorical
                        candidate_features_batched[key] = torch.tensor(raw_values, dtype=torch.long).to(self.model.device)
            
            # Ensure tensors have correct shape for score_candidates (num_candidates, feature_dim)
            # score_candidates expects dictionary of (num_candidates, ...) for candidate_features
            # and dictionary of (1, ...) for user_features.
            
            # 3. Score Candidates using DINRanker
            model_scores_np = self.model.score_candidates(user_features_for_model, candidate_features_batched)
            # model_scores_np should be (1, num_candidates) or (num_candidates,)
            # Flatten to (num_candidates,)
            if model_scores_np.ndim > 1:
                model_scores_np = model_scores_np.flatten()

            # Create a list of (new_score, original_candidate_item, original_index) tuples
            scored_candidates_with_original = []
            for i, score in enumerate(model_scores_np):
                original_item = original_candidate_items[i]
                scored_candidates_with_original.append((score, original_item, i))
            
            # Sort by new score (descending)
            scored_candidates_with_original.sort(key=lambda x: x[0], reverse=True)
            
            # 4. Select Top 100 Candidates (positives + score-based negatives)
            new_candidate_list = []
            added_item_ids = set()
            
            # Add true positives (label=1) from original candidates first
            # These are guaranteed to be relevant and should be preserved if available
            for _, original_item, _ in scored_candidates_with_original:
                if original_item.label == 1 and original_item.item_id not in added_item_ids:
                    # Update score to DIN's prediction
                    new_candidate_list.append(CandidateItem(
                        item_id=original_item.item_id,
                        features=original_item.features,
                        score=model_scores_np[original_item.metadata.get('original_index', -1)], # Use original_index to get score if stored, else lookup
                        label=1 # Preserve label
                    ))
                    added_item_ids.add(original_item.item_id)
            
            # Fill the rest with top-scoring candidates (mostly negatives, or positives not yet added)
            for new_score, original_item, original_index in scored_candidates_with_original:
                if len(new_candidate_list) >= self.top_k:
                    break
                if original_item.item_id not in added_item_ids:
                    new_candidate_list.append(CandidateItem(
                        item_id=original_item.item_id,
                        features=original_item.features,
                        score=new_score,
                        label=original_item.label # Preserve original label
                    ))
                    added_item_ids.add(original_item.item_id)
            
            # 5. Create new CandidateSet for reranking
            new_cs = CandidateSet(
                request_id=cs.request_id,
                user_id=cs.user_id,
                user_features=cs.user_features.copy(),
                context_features=cs.context_features.copy(),
                candidates=new_candidate_list,
                source_stage=self.stage_name
            )
            output.candidate_sets.append(new_cs)
        
        self.logger.info(f"Pre-ranking processed {input_data.get_total_candidates()} candidates into "
                        f"{output.get_total_candidates()} candidates for next stage.")
        
        return output
    
    def evaluate(self,
                 test_data: Any, # FuxiCTR DataGenerator (e.g., from Dataset.to_torch_dataset)
                 metrics_k: List[int] = [100], # For Recall@k and nDCG@k
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate pre-ranking model for ranking metrics (Recall@K, nDCG@K).
        
        Args:
            test_data: FuxiCTR DataGenerator for evaluation (contains batches of user/item features and true labels).
                       Expected to yield dictionaries with 'user_inputs', 'item_inputs', 'labels', 'request_ids'.
            metrics_k: List of K values for Recall@K and nDCG@K.
            **kwargs: Evaluation parameters
            
        Returns:
            Evaluation metrics (Recall@K, nDCG@K)
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        self.model.eval()
        
        item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
        impression_id_col = getattr(self.feature_map, 'dataset_config', {}).get('impression_id_col', 'impression_id')
        
        # Create a temporary storage for collected data
        # Key: request_id, Value: list of (item_id, pred_score, true_label)
        collected_data_for_ranking_eval = {} 
                
        self.logger.info("Starting pre-ranking evaluation...")
        with torch.no_grad():
            for batch_data in tqdm(test_data, desc="Collecting Predictions for Ranking Evaluation", file=sys.stdout):
                # FuxiCTR DataGenerator typically yields a dictionary where keys are feature names
                # and values are tensors. It also includes 'label' and 'impression_id_col' (request_id)
                # We need to separate user_features and candidate_features from this batch_data.
                
                # Assuming batch_data is a single dictionary for one request_id, with multiple candidate items
                # The _convert_features_to_tensors method expects single values for user_features,
                # and then a list of dicts for candidate features. This is not directly compatible with a batch.
                
                # For `evaluate` to work with `score_candidates`, the `test_data` must be structured such that
                # each batch corresponds to one user/request and contains all its candidates.
                # If test_data yields single (user, item, label) samples, we need to first aggregate.
                
                # Let's use the model.forward() method, which is designed for standard FuxiCTR batch_data
                # and then map it back to (request_id, item_id, pred, label) for ranking metrics.
                
                preds_batch_tensor = self.model.forward(batch_data)['y_pred'] # (batch_size, 1)
                preds_batch = preds_batch_tensor.flatten().cpu().numpy() # (batch_size,)
                
                labels_batch = batch_data['label'].flatten().cpu().numpy()
                request_ids_batch = batch_data[impression_id_col].flatten().cpu().numpy()
                item_ids_batch = batch_data[item_id_col].flatten().cpu().numpy()

                for i in range(len(preds_batch)):
                    req_id = request_ids_batch[i]
                    item_id = item_ids_batch[i]
                    pred_score = preds_batch[i]
                    true_label = labels_batch[i]

                    if req_id not in collected_data_for_ranking_eval:
                        collected_data_for_ranking_eval[req_id] = []
                    collected_data_for_ranking_eval[req_id].append((item_id, pred_score, true_label))
        
        # --- After collecting all data, calculate ranking metrics ---
        total_recall_at_k = {k: 0 for k in metrics_k}
        total_ndcg_at_k = {k: 0.0 for k in metrics_k}
        num_relevant_queries = 0
        
        for req_id, items_data in tqdm(collected_data_for_ranking_eval.items(), desc="Calculating Ranking Metrics", file=sys.stdout):
            # items_data is a list of (item_id, pred_score, true_label) for one request_id
            
            # Sort candidates by predicted score
            items_data.sort(key=lambda x: x[1], reverse=True) # Sort by pred_score
            
            # Extract sorted labels and item_ids
            sorted_item_ids = [x[0] for x in items_data]
            sorted_labels = [x[2] for x in items_data]
            
            true_relevant_items_count = sum(sorted_labels)
            if true_relevant_items_count == 0:
                continue # Skip queries with no relevant items
            
            num_relevant_queries += 1
            
            for k in metrics_k:
                # Calculate Recall@K
                top_k_labels = sorted_labels[:k]
                hits = sum(1 for label in top_k_labels if label == 1)
                # Recall@K for a single query = (Number of relevant items in top K) / (Total number of relevant items for that query)
                total_recall_at_k[k] += (hits / true_relevant_items_count)
                
                # Calculate nDCG@K
                dcg = 0.0
                # ideal_labels = sorted([label for label in sorted_labels if label == 1], reverse=True)[:k]
                # For nDCG, we need ideal labels, which are just the true_relevant_items_count 1s
                # followed by 0s if k > true_relevant_items_count
                
                # Calculate DCG for predicted list
                for rank, label in enumerate(top_k_labels, 1):
                    if label == 1:
                        dcg += 1.0 / np.log2(rank + 1)
                
                # Calculate IDCG (ideal DCG)
                idcg = 0.0
                for rank in range(1, min(true_relevant_items_count, k) + 1):
                    idcg += 1.0 / np.log2(rank + 1)
                
                if idcg > 0:
                    total_ndcg_at_k[k] += (dcg / idcg)
                else:
                    total_ndcg_at_k[k] += 0.0 # If no relevant items or k=0, nDCG is 0
        
        # Final metrics
        metrics = {}
        if num_relevant_queries > 0:
            for k in metrics_k:
                metrics[f"Recall@{k}"] = total_recall_at_k[k] / num_relevant_queries
                metrics[f"nDCG@{k}"] = total_ndcg_at_k[k] / num_relevant_queries
        else:
            self.logger.warning("No relevant queries found for ranking evaluation.")
            for k in metrics_k:
                metrics[f"Recall@{k}"] = 0.0
                metrics[f"nDCG@{k}"] = 0.0
                        
                # Also include default FuxiCTR evaluation metrics if available
                # (model.evaluate is usually called to get AUC/logloss)
                # For now, let's just stick to ranking metrics as requested.
                
                # Save metrics to CSV
                metrics_path = os.path.join(self.output_dir, "eval_metrics.csv")
                with open(metrics_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['metric_name', 'value'])
                    for name, value in sorted(metrics.items()):
                        writer.writerow([name, f"{value:.6f}"])
                
                self.logger.info(f"Evaluation metrics: {metrics}")
                return metrics
