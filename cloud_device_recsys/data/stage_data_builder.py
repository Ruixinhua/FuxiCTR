# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Stage Data Builder

Builds stage-specific training, validation, and test datasets.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import csv
from dataclasses import dataclass

from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from ..pipeline.stage_output import StageOutput, CandidateSet
from .item_pool import ItemPool


@dataclass
class StageDataConfig:
    """Configuration for stage-specific data"""
    stage_name: str
    allowed_groups: List[FeatureGroup]
    top_k: int  # Number of candidates to use/output
    metrics: List[str]  # Metrics to compute


class StageDataBuilder:
    """
    Builds training and evaluation data for each pipeline stage.
    
    Handles:
    - Retrieval: Full item pool evaluation for Recall@K
    - Pre-ranking: Subset evaluation on retrieval candidates
    - Re-ranking: Final evaluation with FG3 features
    """
    
    # Stage configurations
    STAGE_CONFIGS = {
        'retrieval': StageDataConfig(
            stage_name='retrieval',
            allowed_groups=[FeatureGroup.FG1, FeatureGroup.FG2],
            top_k=1000,
            metrics=['Recall@10', 'Recall@50', 'Recall@100', 'HitRate@10', 'HitRate@100']
        ),
        'preranking': StageDataConfig(
            stage_name='preranking',
            allowed_groups=[FeatureGroup.FG1, FeatureGroup.FG2],
            top_k=100,
            metrics=['nDCG@10', 'nDCG@50', 'AUC']
        ),
        'reranking': StageDataConfig(
            stage_name='reranking',
            allowed_groups=[FeatureGroup.FG1, FeatureGroup.FG2, FeatureGroup.FG3],
            top_k=10,
            metrics=['nDCG@5', 'nDCG@10', 'AUC']
        )
    }
    
    def __init__(self,
                 fg_manager: FeatureGroupManager,
                 item_pool: ItemPool,
                 output_dir: str = "./data/stage_data"):
        """
        Initialize stage data builder.
        
        Args:
            fg_manager: Feature group manager
            item_pool: Item pool for retrieval
            output_dir: Output directory for stage data
        """
        self.fg_manager = fg_manager
        self.item_pool = item_pool
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def get_stage_config(self, stage_name: str) -> StageDataConfig:
        """Get configuration for a stage"""
        if stage_name not in self.STAGE_CONFIGS:
            raise ValueError(f"Unknown stage: {stage_name}")
        return self.STAGE_CONFIGS[stage_name]
    
    def filter_features(self,
                        df: pd.DataFrame,
                        stage_name: str,
                        keep_label: bool = True,
                        label_col: str = 'label') -> pd.DataFrame:
        """
        Filter DataFrame to only include allowed features for stage.
        
        Args:
            df: Input DataFrame
            stage_name: Stage name
            keep_label: Whether to keep label column
            label_col: Label column name
            
        Returns:
            Filtered DataFrame
        """
        config = self.get_stage_config(stage_name)
        
        # Get allowed features
        allowed = set()
        for group in config.allowed_groups:
            allowed.update(self.fg_manager.get_features_by_group(group))
        
        # Filter columns
        keep_cols = [c for c in df.columns if c in allowed]
        
        # Add label if requested
        if keep_label and label_col in df.columns and label_col not in keep_cols:
            keep_cols.append(label_col)
        
        # Add meta columns (but NOT user_id for cloud stages - it's FG3)
        # Only include group_id, request_id, sample_id as meta columns
        meta_patterns = ['group_id', 'request_id', 'sample_id']
        meta_cols = [c for c in df.columns 
                    if any(p in c.lower() for p in meta_patterns)]
        for c in meta_cols:
            if c not in keep_cols:
                keep_cols.append(c)
        
        result = df[keep_cols]
        
        self.logger.info(f"Filtered for {stage_name}: {len(df.columns)} -> {len(keep_cols)} columns")
        
        return result
    
    def build_retrieval_eval_data(self,
                                  positive_samples: pd.DataFrame,
                                  user_features: List[str],
                                  item_features: List[str],
                                  label_col: str = 'label') -> Dict[str, Any]:
        """
        Build evaluation data for retrieval stage.
        
        For retrieval, each positive sample needs to be scored against
        all items in the pool to compute Recall@K.
        
        Args:
            positive_samples: DataFrame with positive samples only
            user_features: List of user-side features
            item_features: List of item-side features
            label_col: Label column name
            
        Returns:
            Dictionary with eval data components
        """
        self.logger.info(f"Building retrieval eval data from {len(positive_samples)} positive samples")
        
        # Extract user queries (unique users from positive samples)
        user_cols = [c for c in positive_samples.columns if c in user_features]
        
        if not user_cols:
            self.logger.warning("No user features found in data")
            user_cols = [c for c in positive_samples.columns 
                        if any(p in c.lower() for p in ['user', 'his', 'seq'])]
        
        # Get unique users
        users_df = positive_samples[user_cols].drop_duplicates()
        
        # Get ground truth: which items each user interacted with
        item_cols = [c for c in positive_samples.columns if c in item_features]
        
        if not item_cols:
            item_cols = [c for c in positive_samples.columns
                        if any(p in c.lower() for p in ['item', 'cate', 'brand', 'adgroup'])]
        
        # Build ground truth mapping
        ground_truth = {}
        primary_user = user_cols[0] if user_cols else None
        primary_item = item_cols[0] if item_cols else None
        
        if primary_user and primary_item:
            for user_id, group in positive_samples.groupby(primary_user):
                ground_truth[user_id] = group[primary_item].tolist()
        
        eval_data = {
            'users': users_df,
            'user_features': user_cols,
            'items': self.item_pool.get_all_items(),
            'item_features': item_cols,
            'ground_truth': ground_truth,
            'num_users': len(users_df),
            'num_items': self.item_pool.get_item_count()
        }
        
        self.logger.info(f"Retrieval eval: {len(users_df)} users, "
                        f"{self.item_pool.get_item_count()} items, "
                        f"{len(ground_truth)} ground truth entries")
        
        return eval_data
    
    def build_stage_eval_data(self,
                              stage_name: str,
                              data: pd.DataFrame,
                              prev_stage_output: Optional[StageOutput] = None,
                              label_col: str = 'label') -> Tuple[pd.DataFrame, Dict]:
        """
        Build evaluation data for a specific stage.
        
        Args:
            stage_name: Stage name
            data: Full validation/test data
            prev_stage_output: Output from previous stage (candidates)
            label_col: Label column name
            
        Returns:
            Tuple of (filtered_data, metadata)
        """
        config = self.get_stage_config(stage_name)
        
        if stage_name == 'retrieval':
            # For retrieval, use all positive samples
            filtered = self.filter_features(data, stage_name)
            positive = filtered[filtered[label_col] == 1]
            
            return positive, {
                'eval_type': 'full_pool',
                'num_samples': len(positive),
                'num_items': self.item_pool.get_item_count()
            }
        
        elif stage_name == 'preranking':
            # For pre-ranking, filter to only candidates from retrieval
            if prev_stage_output is None:
                self.logger.warning("No retrieval output, using full data for preranking")
                filtered = self.filter_features(data, stage_name)
                return filtered, {'eval_type': 'full_data', 'num_samples': len(filtered)}
            
            # Get candidate item IDs from retrieval
            candidate_items = set()
            for cs in prev_stage_output.candidate_sets:
                candidate_items.update(cs.get_item_ids())
            
            # Filter data to only include candidate items
            item_col = self._get_primary_item_col(data)
            if item_col:
                filtered = data[data[item_col].isin(candidate_items)]
            else:
                filtered = data
            
            filtered = self.filter_features(filtered, stage_name)
            
            return filtered, {
                'eval_type': 'retrieval_candidates',
                'num_samples': len(filtered),
                'num_candidates': len(candidate_items)
            }
        
        elif stage_name == 'reranking':
            # For re-ranking, filter to candidates from pre-ranking
            if prev_stage_output is None:
                self.logger.warning("No preranking output, using full data for reranking")
                filtered = self.filter_features(data, stage_name)
                return filtered, {'eval_type': 'full_data', 'num_samples': len(filtered)}
            
            candidate_items = set()
            for cs in prev_stage_output.candidate_sets:
                candidate_items.update(cs.get_item_ids())
            
            item_col = self._get_primary_item_col(data)
            if item_col:
                filtered = data[data[item_col].isin(candidate_items)]
            else:
                filtered = data
            
            # Re-ranking uses all feature groups including FG3
            filtered = self.filter_features(filtered, stage_name)
            
            return filtered, {
                'eval_type': 'preranking_candidates',
                'num_samples': len(filtered),
                'num_candidates': len(candidate_items)
            }
        
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
    
    def _get_primary_item_col(self, df: pd.DataFrame) -> Optional[str]:
        """Get primary item column from DataFrame"""
        item_patterns = ['item_id', 'adgroup_id', 'cate_id', 'product_id']
        for pattern in item_patterns:
            for col in df.columns:
                if pattern in col.lower():
                    return col
        return None
    
    def save_stage_data(self,
                        stage_name: str,
                        train_data: pd.DataFrame,
                        valid_data: pd.DataFrame,
                        test_data: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """
        Save stage-specific data to files.
        
        Args:
            stage_name: Stage name
            train_data: Training data
            valid_data: Validation data
            test_data: Test data
            
        Returns:
            Dictionary of saved file paths
        """
        stage_dir = os.path.join(self.output_dir, stage_name)
        os.makedirs(stage_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save training data
        train_path = os.path.join(stage_dir, 'train.parquet')
        train_data.to_parquet(train_path, index=False)
        saved_files['train'] = train_path
        
        # Save validation data
        valid_path = os.path.join(stage_dir, 'valid.parquet')
        valid_data.to_parquet(valid_path, index=False)
        saved_files['valid'] = valid_path
        
        # Save test data
        if test_data is not None:
            test_path = os.path.join(stage_dir, 'test.parquet')
            test_data.to_parquet(test_path, index=False)
            saved_files['test'] = test_path
        
        # Save summary
        summary_path = os.path.join(stage_dir, 'data_summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['split', 'num_samples', 'num_features'])
            writer.writerow(['train', len(train_data), len(train_data.columns)])
            writer.writerow(['valid', len(valid_data), len(valid_data.columns)])
            if test_data is not None:
                writer.writerow(['test', len(test_data), len(test_data.columns)])
        saved_files['summary'] = summary_path
        
        self.logger.info(f"Saved {stage_name} data to {stage_dir}")
        
        return saved_files
