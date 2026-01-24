# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Lightweight Ranker for Pre-ranking

This module implements an efficient pre-ranking model for scoring
thousands of candidates from the retrieval stage.
"""

import torch
from torch import nn
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block


class LightweightRanker(BaseModel):
    """
    Lightweight ranking model for pre-ranking stage.
    
    Features:
    - Efficient MLP-based architecture
    - Uses only FG1 + FG2 features (no FG3)
    - Optional diversity loss for varied recommendations
    - Fast inference for thousands of candidates
    """
    
    def __init__(self,
                 feature_map,
                 model_id="LightweightRanker",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=32,
                 hidden_units=[128, 64, 32],
                 hidden_activations="ReLU",
                 dropout_rates=0.1,
                 batch_norm=True,
                 use_diversity_loss=False,
                 diversity_weight=0.1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        """
        Initialize Lightweight Ranker.
        
        Args:
            feature_map: FuxiCTR FeatureMap object
            model_id: Model identifier
            gpu: GPU device ID (-1 for CPU)
            learning_rate: Learning rate
            embedding_dim: Embedding dimension (smaller for efficiency)
            hidden_units: Hidden layer sizes
            hidden_activations: Activation function
            dropout_rates: Dropout rate
            batch_norm: Whether to use batch normalization
            use_diversity_loss: Whether to add diversity regularization
            diversity_weight: Weight for diversity loss
        """
        super(LightweightRanker, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs
        )
        
        self.use_diversity_loss = use_diversity_loss
        self.diversity_weight = diversity_weight
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Feature embedding layer
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        
        # Calculate input dimension
        num_fields = len(feature_map.features)
        input_dim = embedding_dim * num_fields
        
        self.logger.info(f"LightweightRanker: {num_fields} fields, "
                        f"embedding_dim={embedding_dim}, input_dim={input_dim}")
        
        # MLP for scoring
        self.mlp = MLP_Block(
            input_dim=input_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activations=hidden_activations,
            output_activation=None,
            dropout_rates=dropout_rates,
            batch_norm=batch_norm
        )
        
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
    
    def forward(self, inputs):
        """
        Forward pass.
        
        Args:
            inputs: Input batch
            
        Returns:
            Dictionary with y_pred
        """
        X = self.get_inputs(inputs)
        feat_emb_dict = self.embedding_layer(X)
        feat_emb = self.embedding_layer.dict2tensor(feat_emb_dict)
        
        # Flatten embeddings
        flat_emb = feat_emb.flatten(start_dim=1)
        
        # MLP prediction
        logits = self.mlp(flat_emb)
        y_pred = self.output_activation(logits)
        
        return_dict = {"y_pred": y_pred}
        return return_dict
    
    def compute_loss(self, return_dict, y_true):
        """
        Compute loss with optional diversity regularization.
        
        Args:
            return_dict: Output from forward pass
            y_true: Ground truth labels
            
        Returns:
            Total loss
        """
        # Base loss (BCE or other)
        base_loss = super().compute_loss(return_dict, y_true)
        
        if not self.use_diversity_loss:
            return base_loss
        
        # Diversity loss: encourage varied predictions within batch
        y_pred = return_dict["y_pred"]
        
        # Simple diversity: penalize predictions too similar to batch mean
        batch_mean = y_pred.mean()
        diversity_loss = -torch.var(y_pred)  # Negative variance = penalize uniformity
        
        total_loss = base_loss + self.diversity_weight * diversity_loss
        
        return total_loss
    
    def score_candidates(self, 
                        candidate_features: torch.Tensor,
                        batch_size: int = 1024) -> np.ndarray:
        """
        Score a batch of candidates efficiently.
        
        Args:
            candidate_features: Tensor of candidate features
            batch_size: Batch size for scoring
            
        Returns:
            Array of scores
        """
        self.eval()
        all_scores = []
        
        with torch.no_grad():
            num_candidates = candidate_features.shape[0]
            for i in range(0, num_candidates, batch_size):
                batch = candidate_features[i:i+batch_size]
                if next(self.parameters()).is_cuda:
                    batch = batch.cuda()
                
                feat_emb_dict = self.embedding_layer(batch)
                feat_emb = self.embedding_layer.dict2tensor(feat_emb_dict)
                flat_emb = feat_emb.flatten(start_dim=1)
                
                logits = self.mlp(flat_emb)
                scores = self.output_activation(logits)
                
                all_scores.append(scores.cpu().numpy())
        
        return np.vstack(all_scores).flatten()
    
    def select_top_k(self,
                     scores: np.ndarray,
                     k: int,
                     diversity_rerank: bool = False,
                     item_categories: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Select top-K candidates with optional diversity consideration.
        
        Args:
            scores: Candidate scores
            k: Number of candidates to select
            diversity_rerank: Whether to apply diversity reranking
            item_categories: Category IDs for diversity (optional)
            
        Returns:
            Indices of selected candidates
        """
        k = min(k, len(scores))
        
        if not diversity_rerank or item_categories is None:
            # Simple top-k by score
            return np.argsort(-scores)[:k]
        
        # MMR-style diversity reranking
        selected = []
        remaining = set(range(len(scores)))
        category_counts = {}
        
        for _ in range(k):
            best_idx = None
            best_score = float('-inf')
            
            for idx in remaining:
                score = scores[idx]
                
                # Penalize if category already selected
                cat = item_categories[idx]
                if cat in category_counts:
                    penalty = 0.1 * category_counts[cat]
                    score = score * (1 - penalty)
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
                cat = item_categories[best_idx]
                category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return np.array(selected)
    
    def get_embedding(self, inputs) -> torch.Tensor:
        """
        Get intermediate embedding for a batch.
        
        Args:
            inputs: Input batch
            
        Returns:
            Flattened embedding tensor
        """
        X = self.get_inputs(inputs)
        feat_emb_dict = self.embedding_layer(X)
        feat_emb = self.embedding_layer.dict2tensor(feat_emb_dict)
        return feat_emb.flatten(start_dim=1)
