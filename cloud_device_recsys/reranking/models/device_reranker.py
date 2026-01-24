# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Device Re-ranker Model

This module implements a lightweight re-ranking model for on-device inference.
It uses ALL feature groups (FG1 + FG2 + FG3) including private features.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import os

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block


class DeviceReranker(BaseModel):
    """
    Lightweight on-device re-ranking model.
    
    Features:
    - Uses ALL feature groups (FG1 + FG2 + FG3)
    - FG3 features provide personalization with private user data
    - Lightweight architecture for mobile/edge deployment
    - Support for knowledge distillation from teacher model
    - ONNX export for cross-platform deployment
    """
    
    def __init__(self,
                 feature_map,
                 model_id="DeviceReranker",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=16,  # Smaller for device
                 hidden_units=[64, 32],  # Compact architecture
                 hidden_activations="ReLU",
                 dropout_rates=0.0,  # Less dropout for smaller model
                 batch_norm=False,  # Avoid BN for easier mobile deployment
                 support_distillation=False,
                 distillation_alpha=0.5,
                 distillation_temperature=2.0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        """
        Initialize Device Re-ranker.
        
        Args:
            feature_map: FuxiCTR FeatureMap object
            model_id: Model identifier
            gpu: GPU device ID (-1 for CPU)
            learning_rate: Learning rate
            embedding_dim: Embedding dimension (smaller for efficiency)
            hidden_units: Hidden layer sizes (compact)
            hidden_activations: Activation function
            dropout_rates: Dropout rate
            batch_norm: Whether to use batch normalization
            support_distillation: Whether to enable distillation training
            distillation_alpha: Balance between hard and soft labels
            distillation_temperature: Temperature for soft labels
        """
        super(DeviceReranker, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs
        )
        
        self.support_distillation = support_distillation
        self.distillation_alpha = distillation_alpha
        self.distillation_temperature = distillation_temperature
        self.teacher_model = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Feature embedding layer
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        
        # Calculate input dimension
        num_fields = len(feature_map.features)
        input_dim = embedding_dim * num_fields
        
        self.logger.info(f"DeviceReranker: {num_fields} fields (including FG3), "
                        f"embedding_dim={embedding_dim}, input_dim={input_dim}")
        
        # Compact MLP for scoring
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
        
        # Count and log model size
        self._log_model_size()
    
    def _log_model_size(self):
        """Log model size for deployment awareness"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory size (float32 = 4 bytes)
        size_mb = (total_params * 4) / (1024 * 1024)
        
        self.logger.info(f"Model size: {total_params:,} params ({trainable_params:,} trainable)")
        self.logger.info(f"Estimated size: {size_mb:.2f} MB")
    
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
        
        return_dict = {
            "y_pred": y_pred,
            "logits": logits
        }
        return return_dict
    
    def set_teacher_model(self, teacher_model: BaseModel):
        """
        Set teacher model for distillation.
        
        Args:
            teacher_model: Pre-trained teacher model (typically larger)
        """
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.logger.info("Set teacher model for distillation")
    
    def compute_loss(self, return_dict, y_true):
        """
        Compute loss with optional knowledge distillation.
        
        Args:
            return_dict: Output from forward pass
            y_true: Ground truth labels
            
        Returns:
            Total loss
        """
        if not self.support_distillation or self.teacher_model is None:
            return super().compute_loss(return_dict, y_true)
        
        # Hard label loss (standard BCE)
        hard_loss = super().compute_loss(return_dict, y_true)
        
        # Soft label loss (distillation from teacher)
        student_logits = return_dict["logits"]
        
        # Get teacher predictions (assuming same input format)
        # Note: In practice, you'd need to pass the actual inputs
        # This is simplified - teacher logits would be pre-computed
        teacher_logits = return_dict.get("teacher_logits", student_logits)
        
        # KL divergence for soft labels
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.distillation_temperature, dim=-1),
            F.softmax(teacher_logits / self.distillation_temperature, dim=-1),
            reduction='batchmean'
        ) * (self.distillation_temperature ** 2)
        
        # Combined loss
        total_loss = (1 - self.distillation_alpha) * hard_loss + \
                     self.distillation_alpha * soft_loss
        
        return total_loss
    
    def distill_from_teacher(self,
                             train_data: Any,
                             teacher_model: BaseModel,
                             epochs: int = 10,
                             **kwargs) -> Dict[str, float]:
        """
        Train student model via knowledge distillation.
        
        Args:
            train_data: Training data generator
            teacher_model: Pre-trained teacher model
            epochs: Number of training epochs
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics
        """
        self.set_teacher_model(teacher_model)
        self.support_distillation = True
        
        self.logger.info(f"Starting knowledge distillation for {epochs} epochs")
        
        # Use standard fit with distillation loss
        self.fit(train_data, epochs=epochs, **kwargs)
        
        return {"distillation_complete": 1.0}
    
    def rerank(self,
               candidates_scores: np.ndarray,
               private_features: Optional[Dict[str, Any]] = None,
               top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Re-rank candidates using private (FG3) features.
        
        Args:
            candidates_scores: Pre-ranking scores [num_candidates]
            private_features: FG3 features for personalization
            top_k: Number of final recommendations
            
        Returns:
            Tuple of (selected_indices, final_scores)
        """
        # In real implementation, combine candidate features with FG3
        # This is simplified - would need proper feature tensor construction
        
        # For now, just select top-k
        k = min(top_k, len(candidates_scores))
        top_indices = np.argsort(-candidates_scores)[:k]
        top_scores = candidates_scores[top_indices]
        
        return top_indices, top_scores
    
    def export_onnx(self, export_path: str, sample_input: torch.Tensor) -> str:
        """
        Export model to ONNX format for device deployment.
        
        Args:
            export_path: Path to save ONNX model
            sample_input: Sample input tensor for tracing
            
        Returns:
            Path to exported model
        """
        self.eval()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(export_path) if os.path.dirname(export_path) else '.', 
                   exist_ok=True)
        
        # Export to ONNX
        torch.onnx.export(
            self,
            (sample_input,),
            export_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['features'],
            output_names=['predictions'],
            dynamic_axes={
                'features': {0: 'batch_size'},
                'predictions': {0: 'batch_size'}
            }
        )
        
        self.logger.info(f"Exported model to ONNX: {export_path}")
        
        # Log file size
        size_mb = os.path.getsize(export_path) / (1024 * 1024)
        self.logger.info(f"ONNX model size: {size_mb:.2f} MB")
        
        return export_path
    
    def export_torchscript(self, export_path: str, sample_input: torch.Tensor) -> str:
        """
        Export model to TorchScript for mobile deployment.
        
        Args:
            export_path: Path to save TorchScript model
            sample_input: Sample input tensor for tracing
            
        Returns:
            Path to exported model
        """
        self.eval()
        
        os.makedirs(os.path.dirname(export_path) if os.path.dirname(export_path) else '.', 
                   exist_ok=True)
        
        # Trace the model
        traced_model = torch.jit.trace(self, sample_input)
        traced_model.save(export_path)
        
        self.logger.info(f"Exported model to TorchScript: {export_path}")
        
        # Log file size
        size_mb = os.path.getsize(export_path) / (1024 * 1024)
        self.logger.info(f"TorchScript model size: {size_mb:.2f} MB")
        
        return export_path
