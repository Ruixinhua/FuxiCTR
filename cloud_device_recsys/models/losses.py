# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Loss Functions and Mixins for Recommendation Models.

This module provides:
1. Standalone loss functions (e.g., diversity_loss) for easy reuse
2. Mixin classes for adding loss capabilities to existing models
"""

import torch
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


def compute_diversity_loss(
    item_embeddings: torch.Tensor,
    y_pred: torch.Tensor,
    theta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute diversity loss based on item embedding similarity and prediction scores.
    
    This loss encourages diversity in recommendations by penalizing similar items
    being predicted together. The formula is:
        diversity_loss = theta * sum(y_pred) + (1 - theta) * log_det(similarity_matrix)
    
    Args:
        item_embeddings: Item embedding tensor of shape (batch_size, embedding_dim)
                        or (batch_size, num_items, embedding_dim)
        y_pred: Prediction scores of shape (batch_size, 1) or (batch_size,)
        theta: Weight between prediction sum and diversity term (default: 0.7)
        eps: Small epsilon for numerical stability in logdet (default: 1e-6)
        
    Returns:
        Diversity loss scalar (higher value = more diverse)
        
    Example:
        >>> item_emb = model.get_item_embeddings(batch)  # (B, D)
        >>> y_pred = model(batch)["y_pred"]               # (B, 1)
        >>> div_loss = compute_diversity_loss(item_emb, y_pred, theta=0.7)
        >>> total_loss = base_loss - lambda_ * div_loss
    """
    # Normalize embeddings for cosine similarity
    item_vectors_normalized = torch.nn.functional.normalize(item_embeddings, p=2, dim=-1)
    
    # Compute cosine similarity matrix
    if item_embeddings.dim() == 3:
        # (batch_size, num_items, embedding_dim) -> per-sample similarity
        cosine_similarity = torch.bmm(
            item_vectors_normalized, 
            item_vectors_normalized.transpose(-1, -2)
        )
    else:
        # (batch_size, embedding_dim) -> global batch similarity
        cosine_similarity = torch.matmul(
            item_vectors_normalized, 
            item_vectors_normalized.t()
        )
    
    # Apply the formula (1 + cosine_sim) / 2 to get values in [0, 1]
    similarity_matrix = (1 + cosine_similarity) / 2
    
    # Add small identity matrix for numerical stability before logdet
    identity = torch.eye(
        similarity_matrix.size(-1), 
        device=similarity_matrix.device
    ) * eps
    
    if similarity_matrix.dim() == 3:
        identity = identity.unsqueeze(0).expand(similarity_matrix.size(0), -1, -1)
    
    log_det_similarity = torch.logdet(similarity_matrix + identity)
    
    # Handle potential NaN from logdet (e.g., non-positive definite matrix)
    if similarity_matrix.dim() == 3:
        log_det_similarity = torch.where(
            torch.isnan(log_det_similarity) | torch.isinf(log_det_similarity),
            torch.zeros_like(log_det_similarity),
            log_det_similarity
        )
        log_det_similarity = log_det_similarity.mean()
    else:
        if torch.isnan(log_det_similarity) or torch.isinf(log_det_similarity):
            log_det_similarity = torch.tensor(0.0, device=item_embeddings.device)
    
    # Sum of predicted scores
    r_ui_sum = torch.sum(y_pred)
    
    # Diversity Loss Calculation
    diversity_loss = theta * r_ui_sum + (1 - theta) * log_det_similarity
    
    return diversity_loss


def compute_item_similarity_matrix(
    item_embeddings: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute item-item similarity matrix from embeddings.
    
    Args:
        item_embeddings: Item embedding tensor of shape (batch_size, embedding_dim)
        normalize: Whether to L2-normalize embeddings before computing similarity
        
    Returns:
        Similarity matrix of shape (batch_size, batch_size) with values in [0, 1]
    """
    if normalize:
        item_embeddings = torch.nn.functional.normalize(item_embeddings, p=2, dim=-1)
    
    cosine_similarity = torch.matmul(item_embeddings, item_embeddings.t())
    
    # Transform to [0, 1] range
    similarity_matrix = (1 + cosine_similarity) / 2
    
    return similarity_matrix


class DiversityLossMixin:
    """
    Mixin class to add diversity loss capability to any model.
    
    This mixin can be added to any model inheriting from BaseModel to enable
    diversity-aware training. It provides:
    - Configuration parameters for diversity loss
    - Methods to compute item embeddings and similarity matrix
    - Override of compute_loss to include diversity regularization
    
    Usage:
        class MyModel(DiversityLossMixin, BaseModel):
            def __init__(self, ..., use_diversity_loss=False, **kwargs):
                # Initialize DiversityLossMixin first
                DiversityLossMixin.__init__(
                    self,
                    use_diversity_loss=use_diversity_loss,
                    diversity_lambda=kwargs.pop('diversity_lambda', 0.7),
                    diversity_theta=kwargs.pop('diversity_theta', 0.7),
                )
                # Then initialize BaseModel
                BaseModel.__init__(self, ...)
                
            def get_item_embeddings(self, feat_emb_dict):
                # Return item embeddings for diversity calculation
                return feat_emb_dict['item_id']
    """
    
    def __init__(
        self,
        use_diversity_loss: bool = False,
        diversity_lambda: float = 0.7,
        diversity_theta: float = 0.7,
        diversity_item_features: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize diversity loss parameters.
        
        Args:
            use_diversity_loss: Whether to use diversity loss
            diversity_lambda: Weight of diversity loss in total loss (default: 0.7)
            diversity_theta: Weight between prediction sum and diversity term (default: 0.7)
            diversity_item_features: List of feature names to use for item embeddings.
                                    If None, subclass must implement get_diversity_item_embeddings()
        """
        self._use_diversity_loss = use_diversity_loss
        self._diversity_lambda = diversity_lambda
        self._diversity_theta = diversity_theta
        self._diversity_item_features = diversity_item_features or []
        self._diversity_logger = logging.getLogger(self.__class__.__name__)
        
    @property
    def use_diversity_loss(self) -> bool:
        return self._use_diversity_loss
    
    @use_diversity_loss.setter
    def use_diversity_loss(self, value: bool):
        self._use_diversity_loss = value
        
    @property
    def diversity_lambda(self) -> float:
        return self._diversity_lambda
    
    @property
    def diversity_theta(self) -> float:
        return self._diversity_theta
    
    def get_diversity_item_embeddings(
        self, 
        feat_emb_dict: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Extract item embeddings for diversity calculation.
        
        Override this method if you need custom logic for extracting item embeddings.
        
        Args:
            feat_emb_dict: Dictionary of feature embeddings from embedding layer
            
        Returns:
            Concatenated item embeddings tensor or None if not available
        """
        if not self._diversity_item_features:
            return None
            
        item_feature_embs = []
        for feature_name in self._diversity_item_features:
            if feature_name in feat_emb_dict:
                item_feature_embs.append(feat_emb_dict[feature_name])
        
        if not item_feature_embs:
            return None
            
        return torch.cat(item_feature_embs, dim=-1)
    
    def compute_diversity_regularization(
        self,
        feat_emb_dict: Dict[str, torch.Tensor],
        y_pred: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compute the diversity regularization term.
        
        Args:
            feat_emb_dict: Dictionary of feature embeddings
            y_pred: Prediction scores
            
        Returns:
            Diversity loss value or None if diversity loss is disabled/unavailable
        """
        if not self._use_diversity_loss:
            return None
            
        item_embeddings = self.get_diversity_item_embeddings(feat_emb_dict)
        if item_embeddings is None:
            return None
            
        return compute_diversity_loss(
            item_embeddings=item_embeddings,
            y_pred=y_pred,
            theta=self._diversity_theta,
        )
    
    def add_diversity_to_loss(
        self,
        base_loss: torch.Tensor,
        diversity_loss: Optional[torch.Tensor],
        log_losses: bool = False,
    ) -> torch.Tensor:
        """
        Combine base loss with diversity regularization.
        
        Args:
            base_loss: The base training loss (e.g., BCE)
            diversity_loss: The diversity loss term (can be None)
            log_losses: Whether to log loss components
            
        Returns:
            Total loss with diversity regularization
        """
        if diversity_loss is None:
            return base_loss
            
        total_loss = base_loss - self._diversity_lambda * diversity_loss
        
        if log_losses:
            self._diversity_logger.info(
                f"Base Loss: {base_loss.item():.6f}, "
                f"Diversity Loss: {diversity_loss.item():.6f}, "
                f"Total Loss: {total_loss.item():.6f}"
            )
            
        return total_loss
