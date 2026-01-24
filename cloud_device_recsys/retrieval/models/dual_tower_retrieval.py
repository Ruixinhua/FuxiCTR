# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Dual Tower Retrieval Model

This module implements a dual-tower (two-tower) model for candidate retrieval.
Based on the DSSM architecture from FuxiCTR.
"""

import torch
from torch import nn
import numpy as np
from typing import Dict, List, Optional, Any, Set
import logging

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block


class DualTowerRetrieval(BaseModel):
    """
    Dual-tower model for candidate retrieval.
    
    Architecture:
    - User Tower: Processes FG1 + FG2 features related to user
    - Item Tower: Processes FG1 + FG2 features related to item
    - Similarity: Dot product of user and item embeddings
    
    Only uses FG1 (non-personalized) and FG2 (cloud-personalized) features.
    FG3 (device-only) features are NOT allowed.
    """
    
    def __init__(self,
                 feature_map,
                 model_id="DualTowerRetrieval",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 user_tower_units=[256, 128, 64],
                 item_tower_units=[256, 128, 64],
                 user_tower_activations="ReLU",
                 item_tower_activations="ReLU",
                 user_tower_dropout=0.1,
                 item_tower_dropout=0.1,
                 batch_norm=True,
                 use_l2_norm=True,
                 temperature=0.1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        """
        Initialize Dual Tower Retrieval model.
        
        Args:
            feature_map: FuxiCTR FeatureMap object
            model_id: Model identifier
            gpu: GPU device ID (-1 for CPU)
            learning_rate: Learning rate
            embedding_dim: Embedding dimension for features
            user_tower_units: Hidden units for user tower MLP
            item_tower_units: Hidden units for item tower MLP
            user_tower_activations: Activation function for user tower
            item_tower_activations: Activation function for item tower
            user_tower_dropout: Dropout rate for user tower
            item_tower_dropout: Dropout rate for item tower
            batch_norm: Whether to use batch normalization
            use_l2_norm: Whether to L2 normalize embeddings
            temperature: Temperature for similarity computation
        """
        super(DualTowerRetrieval, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs
        )
        
        self.use_l2_norm = use_l2_norm
        self.temperature = temperature
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create embedding layer
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        
        # Count user and item fields
        user_fields = sum(1 if feature_spec.get("source") == "user" else 0
                         for _, feature_spec in feature_map.features.items())
        item_fields = sum(1 if feature_spec.get("source") == "item" else 0
                         for _, feature_spec in feature_map.features.items())
        
        # Fallback: if source not specified, use feature name patterns
        if user_fields == 0 or item_fields == 0:
            self.logger.warning("Feature source not configured, using name-based detection")
            user_fields = 0
            item_fields = 0
            for name, spec in feature_map.features.items():
                if any(p in name.lower() for p in ['user', 'his', 'seq', 'click', 'exposure']):
                    user_fields += 1
                else:
                    item_fields += 1
        
        self.logger.info(f"User fields: {user_fields}, Item fields: {item_fields}")
        
        # User tower
        self.user_tower = MLP_Block(
            input_dim=embedding_dim * max(1, user_fields),
            output_dim=user_tower_units[-1],
            hidden_units=user_tower_units[:-1],
            hidden_activations=user_tower_activations,
            output_activation=None,
            dropout_rates=user_tower_dropout,
            batch_norm=batch_norm
        )
        
        # Item tower
        self.item_tower = MLP_Block(
            input_dim=embedding_dim * max(1, item_fields),
            output_dim=item_tower_units[-1],
            hidden_units=item_tower_units[:-1],
            hidden_activations=item_tower_activations,
            output_activation=None,
            dropout_rates=item_tower_dropout,
            batch_norm=batch_norm
        )
        
        # Store field counts for embedding separation
        self.user_fields = user_fields
        self.item_fields = item_fields
        
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
    
    def forward(self, inputs):
        """
        Forward pass.
        
        Args:
            inputs: Input batch
            
        Returns:
            Dictionary with y_pred and optionally user/item embeddings
        """
        X = self.get_inputs(inputs)
        feat_emb_dict = self.embedding_layer(X)
        
        # Separate user and item embeddings
        user_emb = self.embedding_layer.dict2tensor(feat_emb_dict, feature_source="user")
        item_emb = self.embedding_layer.dict2tensor(feat_emb_dict, feature_source="item")
        
        # If source not configured, use all embeddings
        if user_emb.shape[1] == 0:
            all_emb = self.embedding_layer.dict2tensor(feat_emb_dict)
            # Split by field count (approximate)
            mid = all_emb.shape[1] // 2
            user_emb = all_emb[:, :mid, :]
            item_emb = all_emb[:, mid:, :]
        
        # Flatten and pass through towers
        user_out = self.user_tower(user_emb.flatten(start_dim=1))
        item_out = self.item_tower(item_emb.flatten(start_dim=1))
        
        # L2 normalize if requested
        if self.use_l2_norm:
            user_out = torch.nn.functional.normalize(user_out, p=2, dim=-1)
            item_out = torch.nn.functional.normalize(item_out, p=2, dim=-1)
        
        # Compute similarity (dot product)
        similarity = (user_out * item_out).sum(dim=-1, keepdim=True)
        
        # Scale by temperature
        similarity = similarity / self.temperature
        
        # Apply output activation (sigmoid for binary classification)
        y_pred = self.output_activation(similarity)
        
        return_dict = {
            "y_pred": y_pred,
            "user_embedding": user_out,
            "item_embedding": item_out
        }
        return return_dict
    
    def get_user_embedding(self, inputs) -> torch.Tensor:
        """
        Get user embedding for a batch of inputs.
        
        Args:
            inputs: Input batch
            
        Returns:
            User embedding tensor [batch_size, embedding_dim]
        """
        X = self.get_inputs(inputs)
        feat_emb_dict = self.embedding_layer(X)
        user_emb = self.embedding_layer.dict2tensor(feat_emb_dict, feature_source="user")
        
        if user_emb.shape[1] == 0:
            all_emb = self.embedding_layer.dict2tensor(feat_emb_dict)
            mid = all_emb.shape[1] // 2
            user_emb = all_emb[:, :mid, :]
        
        user_out = self.user_tower(user_emb.flatten(start_dim=1))
        
        if self.use_l2_norm:
            user_out = torch.nn.functional.normalize(user_out, p=2, dim=-1)
        
        return user_out
    
    def get_item_embedding(self, inputs) -> torch.Tensor:
        """
        Get item embedding for a batch of inputs.
        
        Args:
            inputs: Input batch
            
        Returns:
            Item embedding tensor [batch_size, embedding_dim]
        """
        X = self.get_inputs(inputs)
        feat_emb_dict = self.embedding_layer(X)
        item_emb = self.embedding_layer.dict2tensor(feat_emb_dict, feature_source="item")
        
        if item_emb.shape[1] == 0:
            all_emb = self.embedding_layer.dict2tensor(feat_emb_dict)
            mid = all_emb.shape[1] // 2
            item_emb = all_emb[:, mid:, :]
        
        item_out = self.item_tower(item_emb.flatten(start_dim=1))
        
        if self.use_l2_norm:
            item_out = torch.nn.functional.normalize(item_out, p=2, dim=-1)
        
        return item_out
    
    def retrieve_top_k(self, 
                       user_embeddings: np.ndarray,
                       item_embeddings: np.ndarray,
                       top_k: int = 1000) -> tuple:
        """
        Retrieve top-K items for each user using brute-force search.
        
        Args:
            user_embeddings: User embeddings [num_users, dim]
            item_embeddings: Item embeddings [num_items, dim]
            top_k: Number of items to retrieve
            
        Returns:
            Tuple of (indices [num_users, top_k], scores [num_users, top_k])
        """
        # Compute all similarities
        similarities = np.dot(user_embeddings, item_embeddings.T)  # [num_users, num_items]
        
        # Get top-k indices and scores
        top_k = min(top_k, similarities.shape[1])
        top_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)
        
        return top_indices, top_scores


class RetrievalMetrics:
    """
    Metrics for retrieval evaluation.
    
    Supports: Recall@K, HitRate@K
    """
    
    @staticmethod
    def recall_at_k(retrieved_items: np.ndarray, 
                    ground_truth: np.ndarray,
                    k: int) -> float:
        """
        Compute Recall@K.
        
        Args:
            retrieved_items: Retrieved item indices [num_queries, num_retrieved]
            ground_truth: Ground truth item indices [num_queries] or list of lists
            k: K value for Recall@K
            
        Returns:
            Recall@K score
        """
        total_recall = 0.0
        num_queries = len(retrieved_items)
        
        for i in range(num_queries):
            retrieved_k = set(retrieved_items[i][:k])
            if isinstance(ground_truth[i], (list, np.ndarray)):
                relevant = set(ground_truth[i])
            else:
                relevant = {ground_truth[i]}
            
            if len(relevant) > 0:
                recall = len(retrieved_k & relevant) / len(relevant)
                total_recall += recall
        
        return total_recall / num_queries if num_queries > 0 else 0.0
    
    @staticmethod
    def hit_rate_at_k(retrieved_items: np.ndarray,
                      ground_truth: np.ndarray,
                      k: int) -> float:
        """
        Compute HitRate@K (whether any relevant item is in top-K).
        
        Args:
            retrieved_items: Retrieved item indices [num_queries, num_retrieved]
            ground_truth: Ground truth item indices [num_queries] or list of lists
            k: K value for HitRate@K
            
        Returns:
            HitRate@K score
        """
        hits = 0
        num_queries = len(retrieved_items)
        
        for i in range(num_queries):
            retrieved_k = set(retrieved_items[i][:k])
            if isinstance(ground_truth[i], (list, np.ndarray)):
                relevant = set(ground_truth[i])
            else:
                relevant = {ground_truth[i]}
            
            if len(retrieved_k & relevant) > 0:
                hits += 1
        
        return hits / num_queries if num_queries > 0 else 0.0
