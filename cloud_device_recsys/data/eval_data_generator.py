# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Evaluation Data Generator

Generates evaluation data and computes metrics for each pipeline stage.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import csv
from collections import defaultdict

from ..pipeline.stage_output import StageOutput, CandidateSet
from .item_pool import ItemPool


class EvalDataGenerator:
    """
    Generates evaluation data and computes metrics for each stage.
    
    Supports:
    - Retrieval: Recall@K, HitRate@K against full item pool
    - Pre-ranking: nDCG@K, AUC on retrieval candidates
    - Re-ranking: nDCG@K, AUC on pre-ranking candidates
    """
    
    def __init__(self, output_dir: str = "./eval_results"):
        """
        Initialize evaluation data generator.
        
        Args:
            output_dir: Directory for evaluation results
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        os.makedirs(output_dir, exist_ok=True)
    
    # ==================== Retrieval Metrics ====================
    
    def compute_recall_at_k(self,
                            retrieved_items: Dict[Any, List[Any]],
                            ground_truth: Dict[Any, List[Any]],
                            k: int) -> float:
        """
        Compute Recall@K for retrieval.
        
        Args:
            retrieved_items: {user_id: [retrieved_item_ids]}
            ground_truth: {user_id: [relevant_item_ids]}
            k: K value
            
        Returns:
            Average Recall@K
        """
        recalls = []
        
        for user_id in ground_truth:
            if user_id not in retrieved_items:
                recalls.append(0.0)
                continue
            
            retrieved = set(retrieved_items[user_id][:k])
            relevant = set(ground_truth[user_id])
            
            if len(relevant) == 0:
                continue
            
            recall = len(retrieved & relevant) / len(relevant)
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def compute_hit_rate_at_k(self,
                              retrieved_items: Dict[Any, List[Any]],
                              ground_truth: Dict[Any, List[Any]],
                              k: int) -> float:
        """
        Compute HitRate@K (whether any relevant item is in top-K).
        
        Args:
            retrieved_items: {user_id: [retrieved_item_ids]}
            ground_truth: {user_id: [relevant_item_ids]}
            k: K value
            
        Returns:
            HitRate@K
        """
        hits = 0
        total = 0
        
        for user_id in ground_truth:
            total += 1
            
            if user_id not in retrieved_items:
                continue
            
            retrieved = set(retrieved_items[user_id][:k])
            relevant = set(ground_truth[user_id])
            
            if len(retrieved & relevant) > 0:
                hits += 1
        
        return hits / total if total > 0 else 0.0
    
    # ==================== Ranking Metrics ====================
    
    def compute_ndcg_at_k(self,
                          predictions: np.ndarray,
                          labels: np.ndarray,
                          k: int,
                          group_ids: Optional[np.ndarray] = None) -> float:
        """
        Compute nDCG@K.
        
        Args:
            predictions: Predicted scores
            labels: Ground truth labels (relevance)
            k: K value
            group_ids: Group IDs for grouped evaluation
            
        Returns:
            Average nDCG@K
        """
        if group_ids is None:
            # Single group
            return self._ndcg_single_group(predictions, labels, k)
        
        # Multiple groups
        unique_groups = np.unique(group_ids)
        ndcgs = []
        
        for gid in unique_groups:
            mask = group_ids == gid
            if mask.sum() < 2:
                continue
            
            group_pred = predictions[mask]
            group_label = labels[mask]
            
            ndcg = self._ndcg_single_group(group_pred, group_label, k)
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def _ndcg_single_group(self, predictions: np.ndarray, labels: np.ndarray, k: int) -> float:
        """Compute nDCG for a single group"""
        k = min(k, len(predictions))
        
        # Sort by prediction score
        sorted_indices = np.argsort(-predictions)
        sorted_labels = labels[sorted_indices][:k]
        
        # DCG
        gains = 2 ** sorted_labels - 1
        discounts = np.log2(np.arange(2, k + 2))
        dcg = np.sum(gains / discounts)
        
        # Ideal DCG
        ideal_labels = np.sort(labels)[::-1][:k]
        ideal_gains = 2 ** ideal_labels - 1
        idcg = np.sum(ideal_gains / discounts)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def compute_auc(self,
                    predictions: np.ndarray,
                    labels: np.ndarray,
                    group_ids: Optional[np.ndarray] = None) -> float:
        """
        Compute AUC (Area Under ROC Curve).
        
        Args:
            predictions: Predicted scores
            labels: Ground truth labels
            group_ids: Group IDs for grouped AUC
            
        Returns:
            AUC value
        """
        from sklearn.metrics import roc_auc_score
        
        if group_ids is None:
            # Global AUC
            try:
                return roc_auc_score(labels, predictions)
            except ValueError:
                return 0.5  # When only one class present
        
        # Grouped AUC (gAUC)
        unique_groups = np.unique(group_ids)
        aucs = []
        weights = []
        
        for gid in unique_groups:
            mask = group_ids == gid
            group_labels = labels[mask]
            group_preds = predictions[mask]
            
            # Skip if only one class
            if len(np.unique(group_labels)) < 2:
                continue
            
            try:
                auc = roc_auc_score(group_labels, group_preds)
                aucs.append(auc)
                weights.append(mask.sum())
            except ValueError:
                continue
        
        if not aucs:
            return 0.5
        
        # Weighted average
        return np.average(aucs, weights=weights)
    
    # ==================== Stage Evaluation ====================
    
    def evaluate_retrieval(self,
                           stage_output: StageOutput,
                           ground_truth: Dict[Any, List[Any]],
                           k_values: List[int] = [10, 50, 100, 500, 1000]) -> Dict[str, float]:
        """
        Evaluate retrieval stage.
        
        Args:
            stage_output: Output from retrieval stage
            ground_truth: {user_id: [relevant_item_ids]}
            k_values: K values for metrics
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info("Evaluating retrieval stage")
        
        # Build retrieved items from stage output
        retrieved = {}
        for cs in stage_output.candidate_sets:
            user_id = cs.user_id
            item_ids = [c.item_id for c in sorted(cs.candidates, 
                                                   key=lambda x: x.score, 
                                                   reverse=True)]
            retrieved[user_id] = item_ids
        
        # Compute metrics
        metrics = {}
        
        for k in k_values:
            recall = self.compute_recall_at_k(retrieved, ground_truth, k)
            hit_rate = self.compute_hit_rate_at_k(retrieved, ground_truth, k)
            
            metrics[f'Recall@{k}'] = recall
            metrics[f'HitRate@{k}'] = hit_rate
        
        self.logger.info(f"Retrieval metrics: {metrics}")
        
        # Save metrics
        self._save_metrics('retrieval', metrics)
        
        return metrics
    
    def evaluate_ranking(self,
                         stage_name: str,
                         predictions: np.ndarray,
                         labels: np.ndarray,
                         group_ids: Optional[np.ndarray] = None,
                         k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate ranking stage (pre-ranking or re-ranking).
        
        Args:
            stage_name: Stage name for logging
            predictions: Predicted scores
            labels: Ground truth labels
            group_ids: Group IDs for grouped metrics
            k_values: K values for nDCG
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info(f"Evaluating {stage_name} stage")
        
        metrics = {}
        
        # nDCG@K
        for k in k_values:
            ndcg = self.compute_ndcg_at_k(predictions, labels, k, group_ids)
            metrics[f'nDCG@{k}'] = ndcg
        
        # AUC
        auc = self.compute_auc(predictions, labels)
        metrics['AUC'] = auc
        
        # Grouped AUC if groups provided
        if group_ids is not None:
            gauc = self.compute_auc(predictions, labels, group_ids)
            metrics['gAUC'] = gauc
        
        self.logger.info(f"{stage_name} metrics: {metrics}")
        
        # Save metrics
        self._save_metrics(stage_name, metrics)
        
        return metrics
    
    def compare_stages(self,
                       preranking_metrics: Dict[str, float],
                       reranking_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare metrics between pre-ranking and re-ranking stages.
        
        Args:
            preranking_metrics: Metrics from pre-ranking
            reranking_metrics: Metrics from re-ranking
            
        Returns:
            Comparison results
        """
        comparison = {
            'preranking': preranking_metrics,
            'reranking': reranking_metrics,
            'improvements': {}
        }
        
        # Compute improvements
        common_metrics = set(preranking_metrics.keys()) & set(reranking_metrics.keys())
        
        for metric in common_metrics:
            pre_val = preranking_metrics[metric]
            re_val = reranking_metrics[metric]
            
            if pre_val > 0:
                improvement = (re_val - pre_val) / pre_val * 100
                comparison['improvements'][metric] = {
                    'absolute': re_val - pre_val,
                    'relative_percent': improvement
                }
        
        # Log comparison
        self.logger.info("Stage comparison:")
        for metric, imp in comparison['improvements'].items():
            self.logger.info(f"  {metric}: {imp['relative_percent']:.2f}% improvement")
        
        # Save comparison
        self._save_comparison(comparison)
        
        return comparison
    
    def _save_metrics(self, stage_name: str, metrics: Dict[str, float]) -> str:
        """Save metrics to CSV"""
        path = os.path.join(self.output_dir, f"{stage_name}_metrics.csv")
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}"])
        
        self.logger.info(f"Saved {stage_name} metrics to {path}")
        return path
    
    def _save_comparison(self, comparison: Dict) -> str:
        """Save stage comparison to CSV"""
        path = os.path.join(self.output_dir, "stage_comparison.csv")
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'preranking', 'reranking', 'improvement_pct'])
            
            for metric in comparison['improvements']:
                pre_val = comparison['preranking'].get(metric, 0)
                re_val = comparison['reranking'].get(metric, 0)
                imp = comparison['improvements'][metric]['relative_percent']
                writer.writerow([metric, f"{pre_val:.6f}", f"{re_val:.6f}", f"{imp:.2f}"])
        
        self.logger.info(f"Saved stage comparison to {path}")
        return path
    
    def generate_eval_report(self,
                             all_metrics: Dict[str, Dict[str, float]],
                             output_path: str = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            all_metrics: {stage_name: {metric: value}}
            output_path: Path for report
            
        Returns:
            Path to report file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "evaluation_report.csv")
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['# Evaluation Report'])
            writer.writerow([])
            
            for stage_name, metrics in all_metrics.items():
                writer.writerow([f'## {stage_name.upper()}'])
                writer.writerow(['metric', 'value'])
                for name, value in sorted(metrics.items()):
                    writer.writerow([name, f"{value:.6f}"])
                writer.writerow([])
        
        self.logger.info(f"Generated evaluation report: {output_path}")
        return output_path
