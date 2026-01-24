# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Pipeline Coordinator

This module orchestrates the multi-stage recommendation pipeline.
"""

import os
import json
import csv
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from .base_stage import BaseStage, StageType
from .stage_output import StageOutput


class PipelineCoordinator:
    """
    Coordinates the multi-stage recommendation pipeline.
    
    Supports:
    - Running full pipeline (Retrieval → Pre-ranking → Re-ranking)
    - Running individual stages
    - Resuming pipeline from intermediate outputs
    - Saving results to CSV files
    """
    
    STAGE_ORDER = [StageType.RETRIEVAL, StageType.PRERANKING, StageType.RERANKING]
    
    def __init__(self, 
                 output_dir: str = "./outputs",
                 save_intermediate: bool = True,
                 save_csv: bool = True):
        """
        Initialize pipeline coordinator.
        
        Args:
            output_dir: Base directory for pipeline outputs
            save_intermediate: Whether to save intermediate stage outputs
            save_csv: Whether to save outputs as CSV files
        """
        self.output_dir = output_dir
        self.save_intermediate = save_intermediate
        self.save_csv = save_csv
        self.stages: Dict[StageType, BaseStage] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Pipeline run tracking
        self.run_id: Optional[str] = None
        self.run_metadata: Dict[str, Any] = {}
    
    def register_stage(self, stage: BaseStage) -> None:
        """Register a stage with the coordinator"""
        self.stages[stage.stage_type] = stage
        self.logger.info(f"Registered stage: {stage.stage_name} ({stage.stage_type.value})")
    
    def get_stage(self, stage_type: StageType) -> Optional[BaseStage]:
        """Get a registered stage by type"""
        return self.stages.get(stage_type)
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _get_stage_output_path(self, stage_type: StageType) -> str:
        """Get path for stage output file"""
        return os.path.join(self.output_dir, f"{stage_type.value}_output.pkl")
    
    def run_full_pipeline(self, 
                          initial_data: Any = None,
                          **kwargs) -> Dict[str, StageOutput]:
        """
        Run the complete pipeline from Retrieval to Re-ranking.
        
        Args:
            initial_data: Initial data for the retrieval stage
            **kwargs: Additional parameters for stages
            
        Returns:
            Dictionary mapping stage type to StageOutput
        """
        self.run_id = self._generate_run_id()
        self.run_metadata = {
            'run_id': self.run_id,
            'start_time': datetime.now().isoformat(),
            'stages_run': [],
            'kwargs': str(kwargs)
        }
        
        self.logger.info(f"Starting full pipeline run: {self.run_id}")
        
        results: Dict[str, StageOutput] = {}
        prev_output: Optional[StageOutput] = None
        
        for stage_type in self.STAGE_ORDER:
            if stage_type not in self.stages:
                self.logger.warning(f"Stage {stage_type.value} not registered, skipping")
                continue
            
            stage = self.stages[stage_type]
            self.logger.info(f"Running stage: {stage.stage_name}")
            
            # First stage uses initial_data, others use previous output
            if stage_type == StageType.RETRIEVAL:
                output = stage.process(input_data=initial_data, **kwargs)
            else:
                output = stage.process(input_data=prev_output, **kwargs)
            
            output.mark_complete()
            results[stage_type.value] = output
            self.run_metadata['stages_run'].append(stage_type.value)
            
            # Save intermediate output
            if self.save_intermediate:
                stage.save_output(output, save_csv=self.save_csv)
            
            self.logger.info(f"Stage {stage_type.value} complete: "
                           f"{output.get_total_candidates()} candidates, "
                           f"{output.duration_seconds:.2f}s")
            
            prev_output = output
        
        # Save pipeline summary
        self.run_metadata['end_time'] = datetime.now().isoformat()
        self._save_pipeline_summary(results)
        
        self.logger.info(f"Pipeline run {self.run_id} complete")
        return results
    
    def run_stage(self,
                  stage_type: StageType,
                  prev_output_path: Optional[str] = None,
                  **kwargs) -> StageOutput:
        """
        Run a single stage.
        
        Args:
            stage_type: Type of stage to run
            prev_output_path: Path to previous stage output (if not first stage)
            **kwargs: Stage-specific parameters
            
        Returns:
            StageOutput from the stage
        """
        if stage_type not in self.stages:
            raise ValueError(f"Stage {stage_type.value} not registered")
        
        stage = self.stages[stage_type]
        
        # Load previous output if needed
        prev_output = None
        if prev_output_path:
            prev_output = stage.load_previous_output(prev_output_path)
        elif stage_type != StageType.RETRIEVAL:
            # Try to load from default path
            prev_stage_idx = self.STAGE_ORDER.index(stage_type) - 1
            if prev_stage_idx >= 0:
                prev_stage_type = self.STAGE_ORDER[prev_stage_idx]
                default_path = self._get_stage_output_path(prev_stage_type)
                if os.path.exists(default_path):
                    prev_output = stage.load_previous_output(default_path)
                    self.logger.info(f"Loaded previous output from {default_path}")
                else:
                    raise ValueError(f"Previous stage output required but not found: {default_path}")
        
        self.logger.info(f"Running stage: {stage.stage_name}")
        output = stage.process(input_data=prev_output, **kwargs)
        output.mark_complete()
        
        # Save output
        stage.save_output(output, save_csv=self.save_csv)
        
        self.logger.info(f"Stage {stage_type.value} complete: "
                        f"{output.get_total_candidates()} candidates")
        
        return output
    
    def train_stage(self,
                    stage_type: StageType,
                    train_data: Any,
                    valid_data: Optional[Any] = None,
                    **kwargs) -> Dict[str, float]:
        """
        Train a specific stage's model.
        
        Args:
            stage_type: Type of stage to train
            train_data: Training data
            valid_data: Validation data
            **kwargs: Training parameters
            
        Returns:
            Training metrics
        """
        if stage_type not in self.stages:
            raise ValueError(f"Stage {stage_type.value} not registered")
        
        stage = self.stages[stage_type]
        self.logger.info(f"Training stage: {stage.stage_name}")
        
        metrics = stage.train(train_data, valid_data, **kwargs)
        
        # Save training metrics to CSV
        metrics_path = os.path.join(self.output_dir, f"{stage_type.value}_training_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}" if isinstance(value, float) else value])
        
        self.logger.info(f"Saved training metrics to {metrics_path}")
        return metrics
    
    def evaluate_stage(self,
                       stage_type: StageType,
                       test_data: Any,
                       **kwargs) -> Dict[str, float]:
        """
        Evaluate a specific stage on test data.
        
        Args:
            stage_type: Type of stage to evaluate
            test_data: Test data
            **kwargs: Evaluation parameters
            
        Returns:
            Evaluation metrics
        """
        if stage_type not in self.stages:
            raise ValueError(f"Stage {stage_type.value} not registered")
        
        stage = self.stages[stage_type]
        self.logger.info(f"Evaluating stage: {stage.stage_name}")
        
        metrics = stage.evaluate(test_data, **kwargs)
        
        # Save evaluation metrics to CSV
        metrics_path = os.path.join(self.output_dir, f"{stage_type.value}_eval_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}" if isinstance(value, float) else value])
        
        self.logger.info(f"Saved evaluation metrics to {metrics_path}")
        return metrics
    
    def _save_pipeline_summary(self, results: Dict[str, StageOutput]) -> str:
        """Save pipeline run summary to CSV"""
        summary_path = os.path.join(self.output_dir, f"pipeline_summary_{self.run_id}.csv")
        
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Pipeline metadata
            writer.writerow(['# Pipeline Run Summary'])
            writer.writerow(['run_id', self.run_id])
            writer.writerow(['start_time', self.run_metadata.get('start_time', '')])
            writer.writerow(['end_time', self.run_metadata.get('end_time', '')])
            writer.writerow([])
            
            # Stage results
            writer.writerow(['stage', 'num_requests', 'total_candidates', 'duration_s'] + 
                          list(next(iter(results.values())).metrics.keys()) if results else [])
            
            for stage_name, output in results.items():
                row = [
                    stage_name,
                    len(output.candidate_sets),
                    output.get_total_candidates(),
                    f"{output.duration_seconds:.2f}"
                ]
                row.extend([f"{v:.6f}" if isinstance(v, float) else v 
                           for v in output.metrics.values()])
                writer.writerow(row)
        
        self.logger.info(f"Saved pipeline summary to {summary_path}")
        return summary_path
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        status = {
            'registered_stages': {st.value: stage.stage_name 
                                 for st, stage in self.stages.items()},
            'output_dir': self.output_dir,
            'available_outputs': []
        }
        
        # Check for available stage outputs
        for stage_type in self.STAGE_ORDER:
            output_path = self._get_stage_output_path(stage_type)
            if os.path.exists(output_path):
                status['available_outputs'].append({
                    'stage': stage_type.value,
                    'path': output_path,
                    'modified': datetime.fromtimestamp(
                        os.path.getmtime(output_path)).isoformat()
                })
        
        return status
