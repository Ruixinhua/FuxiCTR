#!/usr/bin/env python
# =============================================================================
# Data Preprocessing Script (v2)
# =============================================================================

"""
Preprocesses raw data for cloud-device recommendation pipeline.
Uses new config format with compact feature definitions.

Usage:
    python preprocess_data.py --config ./config
    python preprocess_data.py --config ./config --force_rebuild
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
import pandas as pd

from cloud_device_recsys.config.config_parser import ConfigParser
from cloud_device_recsys.config.feature_groups import FeatureGroupManager, FeatureGroup
from cloud_device_recsys.data.item_pool import ItemPool


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('DataPreprocessor')


def generate_impression_id(df: pl.LazyFrame, config: dict) -> pl.LazyFrame:
    """Generate impression_id if not present"""
    impression_col = config.get('impression_id_col', 'impression_id')
    schema = df.collect_schema()
    
    if impression_col not in schema.names():
        # Generate unique ID from row number
        df = df.with_row_index(impression_col)
    
    return df


def filter_positive_samples(df: pl.LazyFrame, label_col: str) -> pl.LazyFrame:
    """Filter to positive samples only"""
    return df.filter(pl.col(label_col) == 1)


def preprocess_split(
    raw_path: str,
    output_path: str,
    dataset_config: dict,
    is_eval: bool = False,
    logger: logging.Logger = None
) -> str:
    """
    Preprocess a single data split.
    
    Args:
        raw_path: Path to raw CSV
        output_path: Path for output parquet
        dataset_config: Dataset configuration
        is_eval: If True, filter to positive samples only
        logger: Logger instance
        
    Returns:
        Output path
    """
    if not os.path.exists(raw_path):
        if logger:
            logger.warning(f"Raw file not found: {raw_path}")
        return None
    
    if logger:
        logger.info(f"Processing: {raw_path}")
    
    # Load data
    df = pl.scan_csv(raw_path)
    
    # Generate impression_id if needed
    if dataset_config.get('preprocessing', {}).get('generate_impression_id', True):
        df = generate_impression_id(df, dataset_config)
    
    # Filter for eval data
    label_col = dataset_config.get('label_col', {}).get('name', 'label')
    if is_eval and dataset_config.get('preprocessing', {}).get('positive_only_eval', True):
        df = filter_positive_samples(df, label_col)
        if logger:
            logger.info("Filtered to positive samples only")
    
    # Collect and save
    result = df.collect()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.write_parquet(output_path)
    
    if logger:
        logger.info(f"Saved {len(result)} rows to {output_path}")
    
    return output_path


def build_feature_vocab(
    df: pl.DataFrame,
    feature_cols: list,
    output_path: str,
    logger: logging.Logger = None
) -> dict:
    """Build feature vocabulary from data"""
    vocab = {}
    
    for feat in feature_cols:
        name = feat['name']
        feat_type = feat.get('type', 'categorical')
        
        if feat_type == 'categorical':
            if name in df.columns:
                unique_vals = df[name].unique().to_list()
                vocab[name] = {str(v): i for i, v in enumerate(unique_vals, start=1)}
        elif feat_type == 'sequence':
            # For sequences, vocab is built from splitted values
            if name in df.columns:
                splitter = feat.get('splitter', ',')
                all_vals = set()
                for val in df[name].drop_nulls().to_list():
                    if isinstance(val, str):
                        all_vals.update(val.split(splitter))
                vocab[name] = {str(v): i for i, v in enumerate(sorted(all_vals), start=1)}
    
    # Save vocab
    import json
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    if logger:
        logger.info(f"Saved vocabulary to {output_path}")
    
    return vocab


def main():
    parser = argparse.ArgumentParser(description='Preprocess data v2')
    parser.add_argument('--config', type=str, default='./cloud_device_recsys/config',
                       help='Config directory')
    parser.add_argument('--dataset_id', type=str, default=None,
                       help='Dataset ID to process (from dataset_config.yaml)')
    parser.add_argument('--force_rebuild', action='store_true',
                       help='Force rebuild')
    parser.add_argument('--n_rows', type=int, default=None,
                       help='Limit rows for testing')
    
    args = parser.parse_args()
    
    # Load config
    config_parser = ConfigParser(args.config)
    config = config_parser.get_full_config(dataset_id=args.dataset_id)
    
    dataset_config = config['dataset']
    pipeline_config = config['pipeline']
    
    # Setup
    output_dir = dataset_config['processed_paths']['train'].rsplit('/', 1)[0]
    logger = setup_logging(output_dir)
    
    logger.info("=" * 60)
    logger.info("Cloud-Device Recommendation - Data Preprocessing v2")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_config.get('dataset_id')}")
    
    raw_paths = dataset_config['raw_paths']
    processed_paths = dataset_config['processed_paths']
    
    # Check if already preprocessed
    if os.path.exists(processed_paths['train']) and not args.force_rebuild:
        logger.info("Preprocessed data exists. Use --force_rebuild to reprocess.")
        return
    
    # Process train
    preprocess_split(
        raw_path=raw_paths.get('train'),
        output_path=processed_paths['train'],
        dataset_config=dataset_config,
        is_eval=False,
        logger=logger
    )
    
    # Process valid (positive only)
    preprocess_split(
        raw_path=raw_paths.get('valid'),
        output_path=processed_paths['valid'],
        dataset_config=dataset_config,
        is_eval=True,
        logger=logger
    )
    
    # Process test (positive only)
    preprocess_split(
        raw_path=raw_paths.get('test'),
        output_path=processed_paths['test'],
        dataset_config=dataset_config,
        is_eval=True,
        logger=logger
    )
    
    # Build feature vocabulary from train
    logger.info("Building feature vocabulary...")
    train_df = pl.read_parquet(processed_paths['train'])
    
    feature_cols = dataset_config.get('feature_cols_expanded', [])
    vocab = build_feature_vocab(
        df=train_df,
        feature_cols=feature_cols,
        output_path=processed_paths['feature_vocab'],
        logger=logger
    )
    
    # Create FuxiCTR-compatible feature_map with vocab_size
    logger.info("Creating FuxiCTR-compatible feature_map...")
    import json
    
    # Build features list in FuxiCTR format: [{feature_name: {type, vocab_size, ...}}, ...]
    features_list = []
    total_features = 0
    input_length = 0
    num_fields = 0
    
    for feat in feature_cols:
        feat_name = feat['name']
        feat_type = feat.get('type', 'categorical')
        feat_group = feat.get('feature_group', 'FG1')
        
        # Determine source based on feature_group:
        # FG1 (non_personalized) = item features
        # FG2 (cloud_personalized) = user behavioral features
        # FG3 (device_only) = user privacy features
        if feat_group in ['FG1', 'non_personalized']:
            source = 'item'
        else:  # FG2 or FG3
            source = 'user'
        
        feat_spec = {
            'source': source,
            'type': feat_type,
        }
        
        if feat_type in ['categorical', 'sequence']:
            # Get vocab_size from vocabulary
            if feat_name in vocab:
                feat_spec['vocab_size'] = len(vocab[feat_name]) + 1  # +1 for padding
            else:
                feat_spec['vocab_size'] = 2  # Default minimum
            
            feat_spec['padding_idx'] = 0
            total_features += feat_spec['vocab_size']
            
            if feat_type == 'sequence':
                max_len = feat.get('max_len', 50)
                feat_spec['max_len'] = max_len
                feat_spec['feature_encoder'] = 'layers.MaskedAveragePooling()'
                input_length += max_len
                
                # Add share_embedding if specified
                if 'share_embedding' in feat:
                    feat_spec['share_embedding'] = feat['share_embedding']
            else:
                input_length += 1
        elif feat_type == 'numeric':
            input_length += 1
        
        num_fields += 1
        features_list.append({feat_name: feat_spec})
    
    # Build complete feature_map
    feature_map = {
        'dataset_id': dataset_config.get('dataset_id'),
        'num_fields': num_fields,
        'total_features': total_features,
        'input_length': input_length,
        'labels': [dataset_config.get('label_col', {}).get('name', 'label')],
        'features': features_list
    }
    
    with open(processed_paths['feature_map'], 'w') as f:
        json.dump(feature_map, f, indent=4)
    logger.info(f"Saved FuxiCTR feature_map to {processed_paths['feature_map']}")
    
    # Extract item pool
    logger.info("Extracting item pool...")
    item_pool_config = dataset_config.get('item_pool', {})
    item_pool_raw = raw_paths.get('cand_item_list')
    
    if item_pool_raw and os.path.exists(item_pool_raw):
        item_pool_df = pl.read_csv(item_pool_raw)
        item_pool_path = os.path.join(output_dir, "item_pool.parquet")
        item_pool_df.write_parquet(item_pool_path)
        logger.info(f"Saved {len(item_pool_df)} items to {item_pool_path}")
    
    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
