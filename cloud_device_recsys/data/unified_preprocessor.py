# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Unified Preprocessor

Preprocesses raw data with all feature groups (FG1+FG2+FG3).
Each stage then filters to use only its allowed features.
"""

import os
import logging
import json
import csv
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np

from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from fuxictr.preprocess.feature_processor import FeatureProcessor
from fuxictr.preprocess.build_dataset import build_dataset, transform


class UnifiedPreprocessor:
    """
    Unified data preprocessor for multi-stage recommendation.
    
    Processes raw data once with all features, then provides
    filtered views for each pipeline stage.
    """
    
    def __init__(self,
                 data_config: Dict[str, Any],
                 feature_group_manager: FeatureGroupManager,
                 output_dir: str = "./data/processed"):
        """
        Initialize unified preprocessor.
        
        Args:
            data_config: Dataset configuration (feature_cols, label_col, etc.)
            feature_group_manager: Manager for feature group assignments
            output_dir: Directory for processed data
        """
        self.data_config = data_config
        self.fg_manager = feature_group_manager
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Paths for processed data
        self.unified_train_path = os.path.join(output_dir, "train.parquet")
        self.unified_valid_path = os.path.join(output_dir, "valid.parquet")
        self.unified_test_path = os.path.join(output_dir, "test.parquet")
        self.feature_map_path = os.path.join(output_dir, "feature_map.json")
        self.item_pool_path = os.path.join(output_dir, "item_pool.parquet")
        self.feature_processor_path = os.path.join(output_dir, "feature_processor.pkl")
    
    def is_preprocessed(self) -> bool:
        """Check if data has already been preprocessed"""
        return (os.path.exists(self.unified_train_path) and 
                os.path.exists(self.feature_map_path))
    
    def preprocess(self,
                   train_data: str,
                   valid_data: Optional[str] = None,
                   test_data: Optional[str] = None,
                   force_rebuild: bool = False,
                   **kwargs) -> Dict[str, str]:
        """
        Preprocess raw data with all feature groups.
        
        Args:
            train_data: Path to training data
            valid_data: Path to validation data
            test_data: Path to test data
            force_rebuild: Force rebuild even if data exists
            **kwargs: Additional parameters for FeatureProcessor
            
        Returns:
            Dictionary with paths to processed data files
        """
        if self.is_preprocessed() and not force_rebuild:
            self.logger.info("Preprocessed data already exists, skipping...")
            return self._get_processed_paths()
        
        self.logger.info("Starting unified data preprocessing...")
        
        # Create feature processor with all features
        feature_processor = FeatureProcessor(
            feature_cols=self.data_config.get('feature_cols', []),
            label_col=self.data_config.get('label_col', {}),
            dataset_id=self.data_config.get('dataset_id', 'unified'),
            data_root=self.output_dir,
            **kwargs
        )
        
        # Build dataset using FuxiCTR's pipeline
        processed_train, processed_valid, processed_test = build_dataset(
            feature_encoder=feature_processor,
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            rebuild_dataset=True,
            **self.data_config, 
            **kwargs
        )
        
        # Save feature processor for later use
        feature_processor.save_pickle(self.feature_processor_path)
        
        # Auto-assign feature groups based on processed features
        if hasattr(feature_processor, 'feature_map'):
            self.fg_manager.auto_assign_groups(feature_processor.feature_map)
        
        # Save feature group assignments
        fg_path = os.path.join(self.output_dir, "feature_groups.csv")
        self.fg_manager.save_to_csv(fg_path)
        
        # Extract and save item pool
        self._extract_item_pool(processed_train)
        
        self.logger.info("Unified preprocessing complete")
        
        return self._get_processed_paths()
    
    def _get_processed_paths(self) -> Dict[str, str]:
        """Get paths to processed data files"""
        paths = {
            'train': self.unified_train_path if os.path.exists(self.unified_train_path) else None,
            'valid': self.unified_valid_path if os.path.exists(self.unified_valid_path) else None,
            'test': self.unified_test_path if os.path.exists(self.unified_test_path) else None,
            'feature_map': self.feature_map_path,
            'item_pool': self.item_pool_path if os.path.exists(self.item_pool_path) else None,
            'feature_groups': os.path.join(self.output_dir, "feature_groups.csv")
        }
        return paths
    
    def _extract_item_pool(self, train_data_path: str) -> None:
        """
        Extract unique items from training data.
        
        Args:
            train_data_path: Path to processed training data
        """
        self.logger.info("Extracting item pool from training data...")
        
        try:
            # Identify item features (FG1 - non-personalized item features)
            fg1_features = self.fg_manager.get_features_by_group(FeatureGroup.FG1)
            
            # Common item feature patterns
            item_patterns = ['item', 'cate', 'brand', 'price', 'campaign', 'adgroup', 'customer']
            item_features = [f for f in fg1_features 
                           if any(p in f.lower() for p in item_patterns)]
            
            if not item_features:
                self.logger.warning("No item features identified for item pool")
                return
            
            self.logger.info(f"Item features for pool: {item_features}")
            
            # Load training data and extract unique items
            train_path = train_data_path
            if os.path.isdir(train_data_path):
                train_path = os.path.join(train_data_path, "*.parquet")
            
            df = pl.scan_parquet(train_path)
            
            # Select only item features that exist in the data
            available_cols = df.collect_schema().names()
            item_features = [f for f in item_features if f in available_cols]
            
            if not item_features:
                self.logger.warning("No matching item features found in data")
                return
            
            # Get unique items
            item_pool = df.select(item_features).unique().collect()
            
            # Save item pool
            item_pool.write_parquet(self.item_pool_path)
            
            self.logger.info(f"Saved {len(item_pool)} unique items to {self.item_pool_path}")
            
            # Also save as CSV for easier inspection
            csv_path = self.item_pool_path.replace('.parquet', '.csv')
            item_pool.write_csv(csv_path)
            
        except Exception as e:
            self.logger.error(f"Error extracting item pool: {e}")
    
    def get_features_for_stage(self, stage_name: str) -> Set[str]:
        """
        Get allowed features for a specific stage.
        
        Args:
            stage_name: Stage name ('retrieval', 'preranking', 'reranking')
            
        Returns:
            Set of allowed feature names
        """
        if stage_name in ['retrieval', 'preranking']:
            # Cloud stages: FG1 + FG2
            return self.fg_manager.get_cloud_features()
        elif stage_name == 'reranking':
            # Device stage: FG1 + FG2 + FG3
            return self.fg_manager.get_device_features()
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
    
    def load_data_for_stage(self,
                            stage_name: str,
                            split: str = 'train') -> pd.DataFrame:
        """
        Load data filtered for a specific stage.
        
        Args:
            stage_name: Stage name
            split: Data split ('train', 'valid', 'test')
            
        Returns:
            Filtered pandas DataFrame
        """
        # Get data path
        paths = self._get_processed_paths()
        data_path = paths.get(split)
        
        if data_path is None:
            raise FileNotFoundError(f"No {split} data found")
        
        # Load data
        if os.path.isdir(data_path):
            data_path = os.path.join(data_path, "*.parquet")
        
        df = pl.scan_parquet(data_path)
        
        # Get allowed features
        allowed_features = self.get_features_for_stage(stage_name)
        
        # Get label column
        label_col = self.data_config.get('label_col', {}).get('name', 'label')
        
        # Select only allowed features + label
        available_cols = df.collect_schema().names()
        select_cols = [c for c in available_cols 
                      if c in allowed_features or c == label_col]
        
        # Also keep meta columns (group_id, etc.)
        meta_cols = [c for c in available_cols 
                    if 'group_id' in c.lower() or c in ['request_id', 'user_id']]
        select_cols = list(set(select_cols + meta_cols))
        
        result = df.select(select_cols).collect().to_pandas()
        
        self.logger.info(f"Loaded {split} data for {stage_name}: "
                        f"{len(result)} samples, {len(select_cols)} features")
        
        return result
    
    def get_positive_samples(self, split: str = 'test') -> pd.DataFrame:
        """
        Get only positive samples (label=1) for retrieval evaluation.
        
        Args:
            split: Data split
            
        Returns:
            DataFrame with positive samples only
        """
        paths = self._get_processed_paths()
        data_path = paths.get(split)
        
        if data_path is None:
            raise FileNotFoundError(f"No {split} data found")
        
        if os.path.isdir(data_path):
            data_path = os.path.join(data_path, "*.parquet")
        
        label_col = self.data_config.get('label_col', {}).get('name', 'label')
        
        df = pl.scan_parquet(data_path)
        positive_df = df.filter(pl.col(label_col) == 1).collect().to_pandas()
        
        self.logger.info(f"Got {len(positive_df)} positive samples from {split}")
        
        return positive_df
    
    def save_preprocessing_summary(self) -> str:
        """Save summary of preprocessing to CSV"""
        summary_path = os.path.join(self.output_dir, "preprocessing_summary.csv")
        
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['item', 'value'])
            writer.writerow(['output_dir', self.output_dir])
            writer.writerow(['train_path', self.unified_train_path])
            writer.writerow(['valid_path', self.unified_valid_path])
            writer.writerow(['test_path', self.unified_test_path])
            writer.writerow(['item_pool_path', self.item_pool_path])
            
            # Feature group counts
            fg1_count = len(self.fg_manager.get_features_by_group(FeatureGroup.FG1))
            fg2_count = len(self.fg_manager.get_features_by_group(FeatureGroup.FG2))
            fg3_count = len(self.fg_manager.get_features_by_group(FeatureGroup.FG3))
            writer.writerow(['fg1_features', fg1_count])
            writer.writerow(['fg2_features', fg2_count])
            writer.writerow(['fg3_features', fg3_count])
        
        self.logger.info(f"Saved preprocessing summary to {summary_path}")
        return summary_path
