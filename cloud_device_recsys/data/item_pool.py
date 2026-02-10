# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
Item Pool Generation Utilities

This module provides utilities for generating and managing item pools
for the recommendation pipeline. The item pool is essential for:
- Building item embeddings index (Retrieval stage)
- Negative sampling during training
- Item feature lookup during inference (Preranking/Reranking)
"""

import os
import logging
from typing import List
import pandas as pd


def extract_item_corpus(
    input_paths: List[str],
    output_path: str,
    item_id_col: str,
    item_feature_cols: List[str],
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Extract a unique item corpus from one or more dataset files.
    
    This function reads items from multiple data files (train/valid/test),
    extracts unique items based on item_id, and saves the result.
    
    Args:
        input_paths: List of paths to input datasets (parquet or csv)
        output_path: Path to save the output item pool (parquet)
        item_id_col: Column name for the unique item ID
        item_feature_cols: List of item feature column names to include
        logger: Optional logger instance
        
    Returns:
        DataFrame containing the unique item pool
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    cols_to_keep = [item_id_col] + item_feature_cols
    all_items = []
    
    for input_path in input_paths:
        if not os.path.exists(input_path):
            logger.warning(f"Input file not found: {input_path}, skipping...")
            continue
            
        logger.info(f"Loading data from {input_path}...")
        
        if input_path.endswith('.parquet'):
            # Only load required columns for efficiency
            try:
                df = pd.read_parquet(input_path, columns=cols_to_keep)
            except Exception:
                # Fall back to loading all columns if specified columns don't exist
                df = pd.read_parquet(input_path)
                df = df[[c for c in cols_to_keep if c in df.columns]]
        elif input_path.endswith('.csv'):
            df = pd.read_csv(input_path, usecols=lambda c: c in cols_to_keep)
        else:
            logger.warning(f"Unsupported file format: {input_path}, skipping...")
            continue
        
        # Check for missing columns
        missing_cols = [col for col in cols_to_keep if col not in df.columns]
        if missing_cols:
            logger.warning(f"Columns not found in {input_path}: {missing_cols}")
            # Use available columns only
            available_cols = [c for c in cols_to_keep if c in df.columns]
            df = df[available_cols]
        else:
            df = df[cols_to_keep]
        
        all_items.append(df)
        logger.info(f"  Loaded {len(df)} rows from {os.path.basename(input_path)}")
    
    if not all_items:
        raise ValueError("No valid input files found")
    
    # Concatenate all items and drop duplicates
    item_df = pd.concat(all_items, ignore_index=True)
    original_count = len(item_df)
    item_df = item_df.drop_duplicates(subset=[item_id_col])
    
    logger.info(f"Total rows: {original_count} -> Unique items: {len(item_df)}")
    
    # Save to output path
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    item_df.to_parquet(output_path, index=False)
    logger.info(f"Saved item pool to {output_path}")
    
    return item_df


def ensure_item_pool(
    data_paths: dict,
    dataset_config: dict,
    feature_group_manager,
    logger: logging.Logger = None,
    force_regenerate: bool = False
) -> str:
    """
    Ensure item pool exists, generating it if necessary.
    
    This function checks if the item pool file exists. If not, or if
    force_regenerate is True, it generates the item pool from the
    train/valid/test data files.
    
    Args:
        data_paths: Dict with keys 'train_path', 'valid_path', 'test_path', 'item_pool_path'
        dataset_config: Dataset configuration dict with 'item_id_col' and optionally 'item_features'
        feature_group_manager: FeatureGroupManager to get FG1 (item) features
        logger: Optional logger instance
        force_regenerate: If True, regenerate even if file exists
        
    Returns:
        Path to the item pool file
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    item_pool_path = data_paths.get('item_pool_path')
    
    if item_pool_path is None:
        raise ValueError("item_pool_path not found in data_paths")
    
    # Check if item pool exists and skip regeneration if not forced
    if os.path.exists(item_pool_path) and not force_regenerate:
        logger.info(f"Item pool already exists at {item_pool_path}")
        return item_pool_path
    
    # Get item ID column and feature columns
    item_id_col = dataset_config.get('item_id_col', 'cand_item_id')
    
    # Get FG1 (item) features from feature group manager
    from ..config.feature_groups import FeatureGroup
    item_feature_cols = []
    for feat_name, group in feature_group_manager.feature_assignments.items():
        if group == FeatureGroup.FG1 and feat_name != item_id_col:
            item_feature_cols.append(feat_name)
    
    # Also add any explicitly configured item features
    explicit_item_features = dataset_config.get('item_features', [])
    for feat in explicit_item_features:
        if feat not in item_feature_cols and feat != item_id_col:
            item_feature_cols.append(feat)
    
    logger.info(f"Generating item pool with features: {item_feature_cols}")
    
    # Collect input paths (valid, test only - not train, as item pool is for evaluation)
    input_paths = []
    for key in ['valid_path', 'test_path']:
        path = data_paths.get(key)
        if path and os.path.exists(path):
            input_paths.append(path)
    
    if not input_paths:
        raise ValueError("No valid data paths found for item pool generation")
    
    # Generate item pool
    extract_item_corpus(
        input_paths=input_paths,
        output_path=item_pool_path,
        item_id_col=item_id_col,
        item_feature_cols=item_feature_cols,
        logger=logger
    )
    
    return item_pool_path


def ensure_full_item_pool(
    data_paths: dict,
    dataset_config: dict,
    feature_group_manager,
    logger: logging.Logger = None,
    force_regenerate: bool = False
) -> str:
    """
    Ensure a full item pool (train + valid + test) exists for negative sampling.
    
    Unlike ensure_item_pool() which only uses valid/test items (for evaluation),
    this function includes training data items to provide a much larger and more
    representative negative sampling pool.
    
    Args:
        data_paths: Dict with keys 'train_path', 'valid_path', 'test_path', 'full_item_pool_path'
        dataset_config: Dataset configuration dict
        feature_group_manager: FeatureGroupManager to get FG1 (item) features
        logger: Optional logger instance
        force_regenerate: If True, regenerate even if file exists
        
    Returns:
        Path to the full item pool file
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    full_item_pool_path = data_paths.get('full_item_pool_path')
    
    if full_item_pool_path is None:
        raise ValueError("full_item_pool_path not found in data_paths")
    
    # Check if full item pool exists and skip regeneration if not forced
    if os.path.exists(full_item_pool_path) and not force_regenerate:
        logger.info(f"Full item pool already exists at {full_item_pool_path}")
        return full_item_pool_path
    
    # Get item ID column and feature columns
    item_id_col = dataset_config.get('item_id_col', 'cand_item_id')
    
    # Get FG1 (item) features from feature group manager
    from ..config.feature_groups import FeatureGroup
    item_feature_cols = []
    for feat_name, group in feature_group_manager.feature_assignments.items():
        if group == FeatureGroup.FG1 and feat_name != item_id_col:
            item_feature_cols.append(feat_name)
    
    # Also add any explicitly configured item features
    explicit_item_features = dataset_config.get('item_features', [])
    for feat in explicit_item_features:
        if feat not in item_feature_cols and feat != item_id_col:
            item_feature_cols.append(feat)
    
    logger.info(f"Generating full item pool (train+valid+test) with features: {item_feature_cols}")
    
    # Collect input paths: train (or train_positive) + valid + test
    input_paths = []
    
    # Prefer train_positive if it exists (for pairwise mode), else use full train
    # from .positive_data import get_positive_train_path
    # train_path = data_paths.get('train_path')
    # if train_path:
    #     train_positive_path = get_positive_train_path(train_path)
    #     if os.path.exists(train_positive_path):
    #         input_paths.append(train_positive_path)
    #         logger.info(f"Using train_positive for full item pool: {train_positive_path}")
    #     elif os.path.exists(train_path):
    #         input_paths.append(train_path)
    #         logger.info(f"Using full train for full item pool: {train_path}")
    
    for key in ['valid_path', 'test_path', 'train_path']:
        path = data_paths.get(key)
        if path and os.path.exists(path):
            input_paths.append(path)
    
    if not input_paths:
        raise ValueError("No valid data paths found for full item pool generation")
    
    # Generate full item pool
    extract_item_corpus(
        input_paths=input_paths,
        output_path=full_item_pool_path,
        item_id_col=item_id_col,
        item_feature_cols=item_feature_cols,
        logger=logger
    )
    
    return full_item_pool_path


def validate_item_pool_coverage(
    item_pool_path: str,
    data_path: str,
    item_id_col: str,
    logger: logging.Logger = None
) -> dict:
    """
    Validate that the item pool covers all items in a dataset.
    
    Args:
        item_pool_path: Path to item pool parquet file
        data_path: Path to data file to validate against
        item_id_col: Column name for item ID
        logger: Optional logger instance
        
    Returns:
        Dict with validation results:
            - total_data_items: Number of unique items in data
            - total_pool_items: Number of items in pool
            - covered_items: Number of data items covered by pool
            - missing_items: Set of item IDs not in pool
            - coverage_rate: Percentage of data items covered
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Load item pool
    item_pool = pd.read_parquet(item_pool_path)
    pool_item_ids = set(item_pool[item_id_col].unique())
    
    # Load data file
    if data_path.endswith('.parquet'):
        data_df = pd.read_parquet(data_path, columns=[item_id_col])
    else:
        data_df = pd.read_csv(data_path, usecols=[item_id_col])
    
    data_item_ids = set(data_df[item_id_col].unique())
    
    # Calculate coverage
    covered_items = data_item_ids & pool_item_ids
    missing_items = data_item_ids - pool_item_ids
    
    result = {
        'total_data_items': len(data_item_ids),
        'total_pool_items': len(pool_item_ids),
        'covered_items': len(covered_items),
        'missing_items': missing_items,
        'coverage_rate': len(covered_items) / len(data_item_ids) * 100 if data_item_ids else 100.0
    }
    
    if missing_items:
        logger.warning(f"Item pool coverage: {result['coverage_rate']:.2f}% "
                      f"({len(missing_items)} items missing from pool)")
    else:
        logger.info(f"Item pool coverage: 100% ({len(covered_items)} items)")
    
    return result
