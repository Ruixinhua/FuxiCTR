# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Re-ranking Stage Implementation

This module wraps the DeviceReranker model as a pipeline stage.
"""

import os
import csv
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import torch

from ..pipeline.base_stage import BaseStage, StageType
from ..pipeline.stage_output import StageOutput, CandidateSet
from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from .models.device_reranker import DeviceReranker

from fuxictr.features import FeatureMap


class RerankingStage(BaseStage):
    """
    Re-ranking stage for final recommendation.
    
    Runs on device with access to all features including FG3 (private).
    Produces final top-K recommendations.
    """
    
    def __init__(self,
                 feature_map: FeatureMap,
                 feature_group_manager: FeatureGroupManager,
                 model_params: Dict[str, Any],
                 output_dir: str = "./outputs/reranking",
                 top_k: int = 10,
                 support_distillation: bool = False,
                 **kwargs):
        """
        Initialize re-ranking stage.
        
        Args:
            feature_map: FuxiCTR FeatureMap (with FG3 features)
            feature_group_manager: Feature group manager
            model_params: Parameters for DeviceReranker model
            output_dir: Output directory
            top_k: Number of final recommendations
            support_distillation: Whether to enable distillation
        """
        # Re-ranking uses ALL feature groups
        super().__init__(
            stage_name="reranking",
            stage_type=StageType.RERANKING,
            feature_group_manager=feature_group_manager,
            allowed_feature_groups=[FeatureGroup.FG1, FeatureGroup.FG2, FeatureGroup.FG3],
            output_dir=output_dir,
            **kwargs
        )
        
        self.feature_map = feature_map
        self.top_k = top_k
        self.support_distillation = support_distillation
        self.model_params = model_params
        self.model: Optional[DeviceReranker] = None
    
    def build_model(self) -> DeviceReranker:
        """Build and initialize the re-ranking model"""
        # Add default FuxiCTR required parameters
        default_params = {
            'verbose': 1,
            'model_root': self.output_dir,
            'metrics': ['AUC', 'logloss'],
            'embedding_dim': 16,
            'gpu': -1,
            'optimizer': 'adam',
            'loss': 'binary_crossentropy',
        }
        params = {**default_params, **self.model_params}
        
        self.model = DeviceReranker(
            self.feature_map,
            support_distillation=self.support_distillation,
            **params
        )
        self.logger.info("Built DeviceReranker model")
        self.model.count_parameters()
        return self.model
    
    def train(self,
              train_data: Any,
              valid_data: Optional[Any] = None,
              teacher_model: Optional[Any] = None,
              **kwargs) -> Dict[str, float]:
        """
        Train the re-ranking model.
        
        Args:
            train_data: Training data generator
            valid_data: Validation data generator
            teacher_model: Optional teacher model for distillation
            **kwargs: Training parameters
            
        Returns:
            Training metrics
        """
        if self.model is None:
            self.build_model()
        
        self.logger.info("Starting re-ranking model training")
        
        # Distillation training if teacher provided
        if teacher_model is not None and self.support_distillation:
            self.logger.info("Using knowledge distillation training")
            metrics = self.model.distill_from_teacher(
                train_data, teacher_model, **kwargs
            )
        else:
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
    
    def process(self,
                input_data: StageOutput,
                private_features: Optional[Dict[str, Any]] = None,
                **kwargs) -> StageOutput:
        """
        Re-rank candidates to produce final recommendations.
        
        Args:
            input_data: Output from pre-ranking stage
            private_features: FG3 features per request_id (device-only)
            **kwargs: Additional parameters
            
        Returns:
            StageOutput with final recommendations
        """
        # Auto-build model if not built (for testing)
        if self.model is None:
            self.logger.warning("Model not built, building with default params for testing...")
            self.build_model()
        
        # Handle missing input gracefully for testing
        if input_data is None or len(input_data.candidate_sets) == 0:
            self.logger.warning("No input from previous stage. Creating dummy output for testing...")
            output = StageOutput(stage_name=self.stage_name)
            cs = CandidateSet(request_id="test_req_0", user_id=0, source_stage=self.stage_name)
            cs.add_candidate(item_id=0, score=1.0)
            output.candidate_sets.append(cs)
            return output
        
        output = StageOutput(stage_name=self.stage_name)
        self.model.eval()
        
        self.logger.info(f"Re-ranking {len(input_data.candidate_sets)} candidate sets")
        
        for cs in input_data.candidate_sets:
            new_cs = CandidateSet(
                request_id=cs.request_id,
                user_id=cs.user_id,
                user_features=cs.user_features.copy(),
                context_features=cs.context_features.copy(),
                source_stage=self.stage_name
            )
            
            if len(cs.candidates) == 0:
                output.candidate_sets.append(new_cs)
                continue
            
            # Get private features for this request if available
            req_private_features = None
            if private_features is not None:
                req_private_features = private_features.get(cs.request_id)
            
            # Get candidate scores
            scores = np.array([c.score for c in cs.candidates])
            
            # Re-rank with private features
            selected_indices, final_scores = self.model.rerank(
                scores,
                private_features=req_private_features,
                top_k=self.top_k
            )
            
            # Add selected candidates
            for i, idx in enumerate(selected_indices):
                old_candidate = cs.candidates[idx]
                new_cs.add_candidate(
                    item_id=old_candidate.item_id,
                    features=old_candidate.features.copy(),
                    score=float(final_scores[i]),
                    embedding=old_candidate.embedding
                )
            
            output.candidate_sets.append(new_cs)
        
        self.logger.info(f"Re-ranking: {input_data.get_total_candidates()} -> "
                        f"{output.get_total_candidates()} candidates")
        
        return output
    
    def evaluate(self,
                 test_data: Any,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate re-ranking model.
        
        Args:
            test_data: Test data generator
            **kwargs: Evaluation parameters
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        metrics = self.model.evaluate(test_data)
        
        # Save metrics to CSV
        metrics_path = os.path.join(self.output_dir, "eval_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}" if isinstance(value, float) else value])
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def export_for_device(self,
                          export_dir: str,
                          sample_input: torch.Tensor,
                          formats: List[str] = ['onnx', 'torchscript']) -> Dict[str, str]:
        """
        Export model for device deployment.
        
        Args:
            export_dir: Directory for exported models
            sample_input: Sample input tensor
            formats: Export formats ('onnx', 'torchscript')
            
        Returns:
            Dictionary mapping format to export path
        """
        os.makedirs(export_dir, exist_ok=True)
        exports = {}
        
        if 'onnx' in formats:
            onnx_path = os.path.join(export_dir, "device_reranker.onnx")
            self.model.export_onnx(onnx_path, sample_input)
            exports['onnx'] = onnx_path
        
        if 'torchscript' in formats:
            ts_path = os.path.join(export_dir, "device_reranker.pt")
            self.model.export_torchscript(ts_path, sample_input)
            exports['torchscript'] = ts_path
        
        # Save export info to CSV
        info_path = os.path.join(export_dir, "export_info.csv")
        with open(info_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['format', 'path', 'size_mb'])
            for fmt, path in exports.items():
                size_mb = os.path.getsize(path) / (1024 * 1024)
                writer.writerow([fmt, path, f"{size_mb:.2f}"])
        
        return exports
