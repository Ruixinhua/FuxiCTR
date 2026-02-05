#!/usr/bin/env python
# =============================================================================
# Standalone Data Preprocessing Script
# =============================================================================
"""
Preprocesses raw data using FuxiCTR's FeatureProcessor.
Reads configuration from dataset_config.yaml and processes train/valid/test splits.

Usage:
    python cloud_device_recsys/run_preprocess.py \
        --raw_data_root /path/to/raw_data \
        --dataset_id TaobaoOpenMCC \
        --config_dir ./cloud_device_recsys/config

    # With limited rows for testing:
    python cloud_device_recsys/run_preprocess.py \
        --raw_data_root /path/to/raw_data \
        --dataset_id TaobaoOpenMCC \
        --n_rows 1000 \
        --force_rebuild
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import polars as pl
import gc

from fuxictr.preprocess.feature_processor import FeatureProcessor
from fuxictr.preprocess.build_dataset import transform


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(
        output_dir, 
        f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('FuxiCTR-Preprocess')


# =============================================================================
# Custom Preprocessing Functions
# =============================================================================

def generate_impression_id(ddf: pl.LazyFrame, impression_col: str = 'impression_id') -> pl.LazyFrame:
    """
    Generate impression_id if not present in the data.
    
    Args:
        ddf: Polars LazyFrame
        impression_col: Column name for impression ID
        
    Returns:
        LazyFrame with impression_id column
    """
    schema = ddf.collect_schema()
    
    if impression_col not in schema.names():
        logging.info(f"Generating {impression_col} column...")
        ddf = ddf.with_row_index(impression_col)
    
    return ddf


def filter_positive_samples(ddf: pl.LazyFrame, label_col: str) -> pl.LazyFrame:
    """
    Filter to positive samples only (label == 1).
    
    Args:
        ddf: Polars LazyFrame
        label_col: Name of label column
        
    Returns:
        Filtered LazyFrame
    """
    return ddf.filter(pl.col(label_col) == 1)


# =============================================================================
# Configuration Helpers
# =============================================================================

def load_dataset_config(config_dir: str, dataset_id: str) -> Dict[str, Any]:
    """
    Load dataset configuration from dataset_config.yaml.
    
    Args:
        config_dir: Directory containing config files
        dataset_id: Dataset ID to load
        
    Returns:
        Dataset configuration dict
    """
    config_path = os.path.join(config_dir, 'dataset_config.yaml')
    
    with open(config_path, 'r') as f:
        all_configs = yaml.safe_load(f)
    
    if dataset_id not in all_configs:
        raise ValueError(f"Dataset '{dataset_id}' not found in {config_path}. "
                        f"Available: {list(all_configs.keys())}")
    
    config = all_configs[dataset_id]
    config['dataset_id'] = dataset_id
    return config


def expand_feature_cols(feature_cols: List[Dict]) -> List[Dict]:
    """
    Expand compact feature definitions (with name lists) into individual features.
    
    Example:
        Input:  {name: [a, b, c], type: categorical, dtype: int, feature_group: FG1}
        Output: [{name: a, type: categorical, dtype: int, feature_group: FG1},
                 {name: b, type: categorical, dtype: int, feature_group: FG1},
                 {name: c, type: categorical, dtype: int, feature_group: FG1}]
                 
    Note: Sequence columns are always read as strings first (for splitting),
    regardless of the configured dtype.
    """
    expanded = []
    
    for col in feature_cols:
        name_or_list = col.get('name')
        
        if isinstance(name_or_list, list):
            # Handle share_embedding_map for sequences
            share_embedding_map = col.get('share_embedding_map', {})
            
            for name in name_or_list:
                new_col = col.copy()
                new_col['name'] = name
                
                # Remove list-based fields
                if 'share_embedding_map' in new_col:
                    del new_col['share_embedding_map']
                
                # Apply share_embedding from map if exists
                if name in share_embedding_map:
                    new_col['share_embedding'] = share_embedding_map[name]
                
                # Set active based on feature_group
                feature_group = new_col.get('feature_group', '')
                new_col['active'] = feature_group != 'drop'
                
                # Sequence columns must be read as strings for splitting
                if new_col.get('type') == 'sequence':
                    new_col['dtype'] = 'str'
                
                expanded.append(new_col)
        else:
            new_col = col.copy()
            new_col['active'] = col.get('feature_group', '') != 'drop'
            
            # Sequence columns must be read as strings for splitting
            if new_col.get('type') == 'sequence':
                new_col['dtype'] = 'str'
            
            expanded.append(new_col)
    
    return expanded


def build_label_col(config: Dict) -> List[Dict]:
    """
    Build label column specification from config.
    
    Args:
        config: Dataset configuration
        
    Returns:
        List with label column specification
    """
    label_config = config.get('label_col', {})
    return [{
        'name': label_config.get('name', 'label'),
        'dtype': label_config.get('dtype', 'float')
    }]


# =============================================================================
# Preprocessing Pipeline
# =============================================================================

class DataPreprocessor:
    """
    Orchestrates data preprocessing using FuxiCTR's FeatureProcessor.
    """
    
    def __init__(
        self,
        raw_data_root: str,
        dataset_id: str,
        config_dir: str = './cloud_device_recsys/config',
        output_dir: Optional[str] = None,
        n_rows: Optional[int] = None
    ):
        self.raw_data_root = raw_data_root
        self.dataset_id = dataset_id
        self.config_dir = config_dir
        self.n_rows = n_rows
        
        # Load configuration
        self.config = load_dataset_config(config_dir, dataset_id)
        
        # Override paths if provided
        if output_dir:
            self.config['processed_data_root'] = output_dir
        
        # Use output dir from config
        self.output_dir = self.config.get('processed_data_root', 
                                          f'./data/{dataset_id}')
        
        # Setup logging
        self.logger = setup_logging(self.output_dir)
        
        # Expand feature columns
        self.feature_cols = expand_feature_cols(self.config.get('feature_cols', []))
        self.label_cols = build_label_col(self.config)
        
        # Preprocessing options
        self.preprocess_opts = self.config.get('preprocessing', {})
        
        self.logger.info("=" * 60)
        self.logger.info("FuxiCTR Data Preprocessing")
        self.logger.info("=" * 60)
        self.logger.info(f"Dataset ID: {dataset_id}")
        self.logger.info(f"Raw data root: {raw_data_root}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Feature columns: {len(self.feature_cols)}")
        self.logger.info(f"N rows limit: {n_rows if n_rows else 'None'}")
    
    def get_raw_path(self, split: str) -> str:
        """Get path to raw data file for a split."""
        data_format = self.config.get('raw_data_format', 'csv')
        return os.path.join(self.raw_data_root, f"{split}.{data_format}")
    
    def preprocess_split(
        self,
        ddf: pl.LazyFrame,
        split: str,
        is_eval: bool = False
    ) -> pl.LazyFrame:
        """
        Apply custom preprocessing to a data split.
        
        Args:
            ddf: Polars LazyFrame
            split: Split name (train, valid, test)
            is_eval: Whether this is an evaluation split
            
        Returns:
            Preprocessed LazyFrame
        """
        # Generate impression_id if configured
        if self.preprocess_opts.get('generate_impression_id', True) and split != 'train':
            impression_col = self.config.get('impression_id_col', 'impression_id')
            ddf = generate_impression_id(ddf, impression_col)
        
        # Filter positive samples for eval splits
        label_col = self.label_cols[0]['name']
        if is_eval and self.preprocess_opts.get('positive_only_eval', True):
            before_count = ddf.select(pl.count()).collect().item()
            ddf = filter_positive_samples(ddf, label_col)
            after_count = ddf.select(pl.count()).collect().item()
            self.logger.info(f"[{split}] Filtered positive samples: {before_count} -> {after_count}")
        
        return ddf
    
    def run(self, force_rebuild: bool = False) -> None:
        """
        Run the preprocessing pipeline.
        
        Args:
            force_rebuild: Force rebuild even if output exists
        """
        # Check if already processed
        feature_map_path = os.path.join(self.output_dir, 'feature_map.json')
        
        if os.path.exists(feature_map_path) and not force_rebuild:
            self.logger.info(f"Feature map already exists: {feature_map_path}")
            self.logger.info("Use --force_rebuild to reprocess.")
            return
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize FeatureProcessor
        self.logger.info("Initializing FeatureProcessor...")
        feature_processor = FeatureProcessor(
            feature_cols=self.feature_cols,
            label_col=self.label_cols,
            dataset_id=self.dataset_id,
            data_root=os.path.dirname(self.output_dir),
            min_categr_count=self.preprocess_opts.get('min_categr_count', 1)
        )
        
        # Get raw data format
        data_format = self.config.get('raw_data_format', 'csv')
        
        # =====================================================================
        # Read all data splits
        # =====================================================================
        all_ddfs = []
        split_ddfs = {}  # Store preprocessed LazyFrames for each split
        
        # Read train data
        train_path = self.get_raw_path('train')
        train_ddf = None
        if os.path.exists(train_path):
            self.logger.info(f"Reading train data: {train_path}")
            train_ddf = feature_processor.read_data(
                train_path, 
                data_format=data_format,
                n_rows=self.n_rows
            )
            train_ddf = self.preprocess_split(train_ddf, 'train', is_eval=False)
            all_ddfs.append(train_ddf)
            split_ddfs['train'] = train_ddf
        
        # Read valid data
        valid_path = self.get_raw_path('valid')
        if os.path.exists(valid_path):
            self.logger.info(f"Reading valid data: {valid_path}")
            valid_ddf = feature_processor.read_data(
                valid_path,
                data_format=data_format,
                n_rows=self.n_rows
            )
            valid_ddf = self.preprocess_split(valid_ddf, 'valid', is_eval=True)
            all_ddfs.append(valid_ddf)
            split_ddfs['valid'] = valid_ddf
        
        # Read test data
        test_path = self.get_raw_path('test')
        if os.path.exists(test_path):
            self.logger.info(f"Reading test data: {test_path}")
            test_ddf = feature_processor.read_data(
                test_path,
                data_format=data_format,
                n_rows=self.n_rows
            )
            test_ddf = self.preprocess_split(test_ddf, 'test', is_eval=True)
            all_ddfs.append(test_ddf)
            split_ddfs['test'] = test_ddf
        
        if not all_ddfs:
            self.logger.error("No data files found!")
            return
        
        # =====================================================================
        # Build vocabulary from ALL data (train + valid + test)
        # =====================================================================
        self.logger.info("Building vocabulary from ALL data splits...")
        
        # Concatenate all splits for vocabulary building
        if len(all_ddfs) > 1:
            combined_ddf = pl.concat(all_ddfs)
            self.logger.info(f"Combined {len(all_ddfs)} splits for vocabulary building")
        else:
            combined_ddf = all_ddfs[0]
        
        # Apply FuxiCTR preprocessing to combined data
        combined_ddf = feature_processor.preprocess(combined_ddf)
        
        # Fit on combined data to build complete vocabulary
        self.logger.info("Fitting feature processor on ALL data...")
        feature_processor.fit(
            combined_ddf,
            min_categr_count=self.preprocess_opts.get('min_categr_count', 1),
            rebuild_dataset=True
        )
        
        # Clear combined data from memory
        del combined_ddf
        gc.collect()
        
        # =====================================================================
        # Transform and save each split separately
        # =====================================================================
        for split_name, split_ddf in split_ddfs.items():
            self.logger.info(f"Transforming and saving {split_name} data...")
            
            # Re-read the data (since we consumed it during concatenation)
            split_path = self.get_raw_path(split_name)
            split_ddf = feature_processor.read_data(
                split_path,
                data_format=data_format,
                n_rows=self.n_rows
            )
            
            # Apply custom preprocessing
            is_eval = split_name in ['valid', 'test']
            split_ddf = self.preprocess_split(split_ddf, split_name, is_eval=is_eval)
            
            # Apply FuxiCTR preprocessing
            split_ddf = feature_processor.preprocess(split_ddf)
            
            # Transform and save
            transform(
                feature_processor,
                split_ddf,
                split_name,
                block_size=0,  # Auto-detect based on memory
                saved_format='parquet'
            )
            
            del split_ddf
            gc.collect()
        
        self.logger.info("=" * 60)
        self.logger.info("Preprocessing complete!")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Vocabulary built from: {list(split_ddfs.keys())}")
        self.logger.info("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess raw data using FuxiCTR FeatureProcessor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process TaobaoOpenMCC dataset
    python run_preprocess.py \\
        --raw_data_root /path/to/TaobaoOpenMCC/with_item_id \\
        --dataset_id TaobaoOpenMCC

    # Process with limited rows for testing
    python run_preprocess.py \\
        --raw_data_root /path/to/raw_data \\
        --dataset_id TaobaoOpenMCC \\
        --n_rows 1000 \\
        --force_rebuild
        """
    )
    
    parser.add_argument(
        '--raw_data_root',
        type=str,
        required=True,
        help='Path to raw data directory containing train/valid/test files'
    )
    
    parser.add_argument(
        '--dataset_id',
        type=str,
        required=True,
        help='Dataset ID from dataset_config.yaml (e.g., TaobaoOpenMCC, TaobaoAd)'
    )
    
    parser.add_argument(
        '--config_dir',
        type=str,
        default='./cloud_device_recsys/config',
        help='Path to config directory (default: ./cloud_device_recsys/config)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory (default: from config)'
    )
    
    parser.add_argument(
        '--force_rebuild',
        action='store_true',
        help='Force rebuild even if output already exists'
    )
    
    parser.add_argument(
        '--n_rows',
        type=int,
        default=None,
        help='Limit number of rows to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Run preprocessing
    preprocessor = DataPreprocessor(
        raw_data_root=args.raw_data_root,
        dataset_id=args.dataset_id,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        n_rows=args.n_rows
    )
    
    preprocessor.run(force_rebuild=args.force_rebuild)


if __name__ == '__main__':
    main()
