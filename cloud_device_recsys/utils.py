from __future__ import annotations
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import argparse
from typing import Tuple, List, Dict, Any, Optional
from .pipeline.stage_output import StageOutput


def filter_feature_map(feature_map, fg_manager, allowed_feature_groups, use_feature_encoder=False):
    """
    Create a new FeatureMap with only features belonging to allowed feature groups.
    
    This function filters features based on their assigned feature group, keeping only
    those that belong to the specified allowed groups. Labels and special columns 
    are always preserved.
    
    Args:
        feature_map: FuxiCTR FeatureMap object to filter
        fg_manager: FeatureGroupManager with feature assignments
        allowed_feature_groups: List of allowed FeatureGroup enums (e.g., [FeatureGroup.FG1, FeatureGroup.FG2])
        use_feature_encoder: If true, use feature encoder
    Returns:
        A new FeatureMap containing only the allowed features (deep copy of original)
    """
    import copy
    from collections import OrderedDict
    
    new_fm = copy.deepcopy(feature_map)
    new_features = OrderedDict()
    use_features = []
    user_features = fg_manager.get_user_features()
    for name, spec in feature_map.features.items():
        # Check if feature belongs to allowed groups
        group = fg_manager.feature_assignments.get(name)
        is_allowed = False
        for allowed_grp in allowed_feature_groups:
            # Compare enum members directly if possible, or string representation
            if group == allowed_grp or str(group) == str(allowed_grp):
                is_allowed = True
                if name in user_features:
                    spec['source'] = 'user'
                else:
                    spec['source'] = 'item'
                break
        impression_col = feature_map.dataset_config.get('impression_id_col', 'impression_id')
        # Always keep label, score, and special columns (needed for training/indexing)
        if name in [impression_col, 'group_id', 'click', 'clk', 'label'] + feature_map.labels:
            is_allowed = True
            
        if is_allowed:
            if not use_feature_encoder:
                spec['feature_encoder'] = None
            new_features[name] = spec
            use_features.append(name)
            
    new_fm.features = new_features
    new_fm.use_features = use_features
    # Re-set column indices after filtering
    new_fm.set_column_index()
    new_fm.num_fields = len(new_fm.features)
    return new_fm


def setup_logging(output_dir) -> None:
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to ensure we control output
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s: %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logging.info(f"Logging initialized. Saving to {log_file}")


def get_data_dir(dataset_config, dataset_id=None):
    """Get the data directory from dataset configuration."""
    if 'processed_data_root' in dataset_config:
        data_dir = dataset_config['processed_data_root']
    else:
        data_dir = os.path.join(dataset_config.get('data_root', './data'), dataset_id)
    return data_dir


def create_debug_dataset(input_path: str, n_rows: int, output_path: str, logger) -> str:
    """
    Create a small debug dataset from the original parquet file (memory efficient).
    
    Args:
        input_path: Path to the original parquet file
        n_rows: Number of rows to sample
        output_path: Path to save the debug dataset
        logger: Logger instance
        
    Returns:
        Path to the debug dataset (or original if creation failed)
    """
    import pyarrow.parquet as pq
    
    if not os.path.exists(output_path):
        logger.info(f"Creating debug dataset (n={n_rows}) from {input_path}...")
        try:
            # Read efficiently using pyarrow
            pf = pq.ParquetFile(input_path)
            # Read first row group (usually enough for debug)
            table = pf.read_row_group(0) 
            df = table.to_pandas()
            
            # Ensure n_rows doesn't exceed dataframe length
            sample_n = min(n_rows, len(df))
            df_sample = df.head(sample_n)
            df_sample.to_parquet(output_path)
            logger.info(f"Saved debug dataset to {output_path}")
        except Exception as e:
            logger.warning(f"Failed to create debug dataset: {e}. Using original.")
            return input_path
    else:
        logger.info(f"Using existing debug dataset: {output_path}")
    return output_path


def get_data_paths(dataset_config: dict, pipeline_config: dict, logger):
    """
    Get common data paths for all stages.
    
    Args:
        dataset_config: Dataset configuration dict
        pipeline_config: Pipeline configuration dict
        logger: Logger instance
    
    Returns:
        dict with keys: data_dir, data_format, debug_n_rows, debug_dir, 
                        processed_data_root, train_path, valid_path, test_path, item_pool_path
    """
    data_dir = get_data_dir(dataset_config, pipeline_config['dataset_id'])
    data_format = dataset_config.get('processed_data_format', 'parquet')
    
    # Check for debug mode
    debug_n_rows = pipeline_config.get('debug', {}).get('n_rows')
    if debug_n_rows is not None:
        logger.info(f"Debug mode enabled: using {debug_n_rows} rows.")
        debug_dir = os.path.join(data_dir, f"debug_{debug_n_rows}")
        os.makedirs(debug_dir, exist_ok=True)
        processed_data_root = debug_dir
    else:
        debug_dir = None
        processed_data_root = dataset_config.get('processed_data_root', data_dir)
    
    # Build standard paths
    train_path = os.path.join(processed_data_root, f'train.{data_format}')
    valid_path = os.path.join(processed_data_root, f'valid.{data_format}')
    test_path = os.path.join(processed_data_root, f'test.{data_format}')
    
    # Item pool path (test/valid items only - for evaluation/index building)
    item_pool_config = dataset_config.get('item_pool', {})
    item_pool_file = item_pool_config.get('file', 'cand_item_list')
    item_pool_path = os.path.join(dataset_config.get('processed_data_root', data_dir), f'{item_pool_file}.parquet')
    
    # Full item pool path (train+valid+test items - for negative sampling)
    full_item_pool_path = os.path.join(dataset_config.get('processed_data_root', data_dir), 'cand_items_all.parquet')
    
    return {
        'data_dir': data_dir,
        'data_format': data_format,
        'debug_n_rows': debug_n_rows,
        'debug_dir': debug_dir,
        'processed_data_root': processed_data_root,
        'train_path': train_path,
        'valid_path': valid_path,
        'test_path': test_path,
        'item_pool_path': item_pool_path,
        'full_item_pool_path': full_item_pool_path,
        'item_pool_file': item_pool_file,
    }


def prepare_debug_paths(paths: dict, dataset_config: dict, logger) -> dict:
    """
    Create debug datasets if in debug mode and update paths accordingly.
    
    Args:
        paths: Dict from get_data_paths()
        dataset_config: Dataset configuration dict  
        logger: Logger instance
        
    Returns:
        Updated paths dict with debug dataset paths
    """
    debug_n_rows = paths['debug_n_rows']
    debug_dir = paths['debug_dir']
    data_format = paths['data_format']
    
    if debug_n_rows is not None and data_format == 'parquet':
        original_data_root = dataset_config.get('processed_data_root', paths['data_dir'])
        original_train = os.path.join(original_data_root, f'train.{data_format}')
        original_valid = os.path.join(original_data_root, f'valid.{data_format}')
        original_test = os.path.join(original_data_root, f'test.{data_format}')
        # original_item_pool = os.path.join(original_data_root, f"{paths['item_pool_file']}.parquet")
        
        # Create debug datasets and update paths
        paths['train_path'] = create_debug_dataset(original_train, debug_n_rows, os.path.join(debug_dir, 'train.parquet'), logger)
        paths['valid_path'] = create_debug_dataset(original_valid, debug_n_rows, os.path.join(debug_dir, 'valid.parquet'), logger)
        paths['test_path'] = create_debug_dataset(original_test, debug_n_rows, os.path.join(debug_dir, 'test.parquet'), logger)
        # paths['item_pool_path'] = create_debug_dataset(original_item_pool, debug_n_rows, os.path.join(debug_dir, f"{paths['item_pool_file']}.parquet"), logger)
    
    return paths


# =========================================================================
# Shared Inference Utilities for Preranking/Reranking Stages
# =========================================================================

def build_inference_batch_for_candidates(
    user_features: Dict[str, Any],
    candidate_item_ids: List[Any],
    item_features_df: pd.DataFrame,
    feature_map,
    item_id_col: str = 'cand_item_id',
    fallback_item_features: Dict[str, Any] = None,
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """
    Build inference batch for a single query's candidate items.
    
    For each candidate item, replicates user features and looks up item features
    from the item pool DataFrame.
    
    Args:
        user_features: User feature dict {feature_name: value} for this query
        candidate_item_ids: List of candidate item IDs to score
        item_features_df: DataFrame with item features, indexed by item_id
        feature_map: FuxiCTR FeatureMap for dtype info
        item_id_col: Column name for item ID
        fallback_item_features: Fallback features for first item (GT) if missing from pool
        
    Returns:
        Tuple of:
            - batch_dict: {feature_name: np.ndarray} ready for model inference
            - valid_indices: Indices of items that were successfully built
    """
    num_items = len(candidate_item_ids)
    
    # 1. Replicate user features for all items
    user_feature_batch = {}
    for k, val in user_features.items():
        if isinstance(val, np.ndarray) and val.ndim > 0:
            user_feature_batch[k] = np.tile(val, (num_items, 1))
        else:
            user_feature_batch[k] = np.full((num_items,), val)
    
    # 2. Look up item features
    item_feature_cols = list(item_features_df.columns)
    item_feature_batch = {k: [] for k in item_feature_cols}
    if item_id_col not in item_feature_batch and item_id_col in feature_map.features:
        item_feature_batch[item_id_col] = []
    
    valid_indices = []
    
    for idx, item_id in enumerate(candidate_item_ids):
        if item_id in item_features_df.index:
            valid_indices.append(idx)
            row = item_features_df.loc[item_id]
            for col in item_feature_cols:
                item_feature_batch[col].append(row[col])
            if item_id_col in item_feature_batch:
                item_feature_batch[item_id_col].append(item_id)
        elif idx == 0 and fallback_item_features is not None:
            # Use fallback for GT item (first item)
            valid_indices.append(idx)
            for col in item_feature_cols:
                if col in fallback_item_features:
                    item_feature_batch[col].append(fallback_item_features[col])
                else:
                    item_feature_batch[col].append(0)
            if item_id_col in item_feature_batch:
                item_feature_batch[item_id_col].append(item_id)
        # else: skip this item
    
    # 3. Convert to numpy arrays
    for k in item_feature_batch:
        item_feature_batch[k] = np.array(item_feature_batch[k])
    
    # 4. Slice user features to match valid items
    num_valid = len(valid_indices)
    for k in user_feature_batch:
        user_feature_batch[k] = user_feature_batch[k][:num_valid]
    
    # 5. Merge user and item features
    batch_dict = {**user_feature_batch, **item_feature_batch}
    
    return batch_dict, valid_indices


def batch_to_tensors(
    batch_dict: Dict[str, np.ndarray],
    feature_map,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Convert numpy batch to PyTorch tensors based on feature types.
    
    Args:
        batch_dict: {feature_name: np.ndarray}
        feature_map: FuxiCTR FeatureMap for dtype info
        device: Target device for tensors
        
    Returns:
        {feature_name: torch.Tensor}
    """
    tensor_batch = {}
    for k, v in batch_dict.items():
        if k not in feature_map.features:
            continue
        ftype = feature_map.features[k]['type']
        if ftype == 'sequence':
            tensor_batch[k] = torch.tensor(v, dtype=torch.long).to(device)
        elif ftype == 'categorical':
            tensor_batch[k] = torch.tensor(v, dtype=torch.long).to(device)
        else:
            tensor_batch[k] = torch.tensor(v, dtype=torch.float).to(device)
    return tensor_batch

def save_stage_output(stage_output: StageOutput, output_dir: str, prefix: str,
                      logger: logging.Logger = None) -> str:
    """
    Save a StageOutput object to disk using Parquet format for efficiency.

    Args:
        stage_output: StageOutput object to save
        output_dir: Directory to save the output file
        prefix: Prefix for the output filename (e.g., 'retrieval_valid', 'preranking_test')
        logger: Optional logger instance

    Returns:
        Path to the saved directory (Parquet format)
    """
    import time
    if stage_output is None:
        raise ValueError('StageOutput object must not be None')

    os.makedirs(output_dir, exist_ok=True)
    
    # Use Parquet format (directory-based)
    dirname = f"{prefix}_stage_output"
    dirpath = os.path.join(output_dir, dirname)
    
    start_time = time.time()
    stage_output.save_parquet(dirpath)
    elapsed = time.time() - start_time

    if logger:
        logger.info(f"Saved {stage_output.stage_name} output ({stage_output.get_num_requests()} requests, "
                    f"{stage_output.get_total_candidates()} total candidates) to {dirpath} in {elapsed:.2f}s")

    return dirpath

def load_stage_output(filepath: str, logger: logging.Logger = None) -> Optional[StageOutput]:
    """
    Load a StageOutput object from disk. Auto-detects format (Parquet directory or pickle file).

    Args:
        filepath: Path to the Parquet directory or pickle file
        logger: Optional logger instance

    Returns:
        StageOutput object, or None if loading fails
    """
    import time
    from .pipeline.stage_output import StageOutput  # Runtime import to avoid circular dependency
    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError(f"Path {filepath} not found")

    try:
        start_time = time.time()
        
        # Auto-detect format: directory with candidates.parquet = Parquet, else pickle
        if os.path.isdir(filepath) and os.path.exists(os.path.join(filepath, 'candidates.parquet')):
            stage_output = StageOutput.load_parquet(filepath)
            fmt = "Parquet"
        else:
            # Fallback to pickle for backward compatibility
            stage_output = StageOutput.load(filepath)
            fmt = "pickle"
        
        elapsed = time.time() - start_time
        
        if logger:
            logger.info(f"Loaded {stage_output.stage_name} output ({fmt}) from {filepath}: "
                        f"{stage_output.get_num_requests()} requests, "
                        f"{stage_output.get_total_candidates()} total candidates in {elapsed:.2f}s")
        return stage_output
    except Exception as e:
        if logger:
            logger.error(f"Failed to load stage output from {filepath}: {e}")
        return None

def load_stage_outputs_from_dir(
        output_dir: str,
        prev_stage_name: str,
        logger: logging.Logger = None,
        load_test: bool = True
) -> Tuple[Optional[StageOutput], Optional[StageOutput]]:
    """
    Load valid and test stage outputs from a directory. Auto-detects Parquet or pickle format.

    Args:
        output_dir: Directory containing stage output files (e.g., './outputs/exp_xxx/stage_outputs')
        prev_stage_name: Name of the previous stage (e.g., 'retrieval' or 'preranking')
        logger: Optional logger instance
        load_test: If True, load test stage outputs

    Returns:
        Tuple of (valid_output, test_output), both can be None if loading fails
    """
    # Try Parquet format first (directory), then fall back to pickle (.pkl file)
    test_output = None
    if load_test:
        test_parquet = os.path.join(output_dir, f"{prev_stage_name}_test_stage_output")
        test_pickle = os.path.join(output_dir, f"{prev_stage_name}_test_stage_output.pkl")
        test_path = test_parquet if os.path.isdir(test_parquet) else test_pickle
        test_output = load_stage_output(test_path, logger)
    valid_parquet = os.path.join(output_dir, f"{prev_stage_name}_valid_stage_output")
    valid_pickle = os.path.join(output_dir, f"{prev_stage_name}_valid_stage_output.pkl")
    valid_path = valid_parquet if os.path.isdir(valid_parquet) else valid_pickle
    valid_output = load_stage_output(valid_path, logger)
    return valid_output, test_output


def enrich_stage_output_user_features(
        stage_output: StageOutput,
        data_path: str,
        fg_manager,
        impression_id_col: str = 'impression_id',
        logger: logging.Logger = None
) -> StageOutput:
    """
    Enrich StageOutput with missing FG3 user features from original data files.
    
    This provides backward compatibility for stage_output files that were saved
    without FG3 features (due to filtered feature_map in DataLoader).
    
    Args:
        stage_output: StageOutput to enrich
        data_path: Path to original data file (parquet/csv) with complete features
        fg_manager: FeatureGroupManager with feature assignments
        impression_id_col: Column name for request IDs (default: 'impression_id')
        logger: Optional logger instance
        
    Returns:
        StageOutput with enriched user features (modified in-place)
    """
    if stage_output is None:
        return None
        
    if logger is None:
        logger = logging.getLogger(__name__)
    
    user_features_df = stage_output.user_features_df
    if user_features_df is None or len(user_features_df) == 0:
        logger.warning("No user features to enrich")
        return stage_output
    
    # Get all user features (FG2 + FG3) that should be present
    all_user_features = fg_manager.get_user_features()
    existing_columns = set(user_features_df.columns)
    missing_features = all_user_features - existing_columns
    
    if not missing_features:
        logger.info("All expected user features already present, no enrichment needed")
        return stage_output
    
    logger.info(f"Found {len(missing_features)} missing user features: {sorted(missing_features)}")
    
    # Load source data
    if not os.path.exists(data_path):
        logger.warning(f"Data file not found: {data_path}, cannot enrich features")
        return stage_output
    
    try:
        if data_path.endswith('.parquet'):
            source_df = pd.read_parquet(data_path)
            if impression_id_col not in source_df.columns:
                source_df[impression_id_col] = source_df.index  # Assume index is impression_id if column missing
        else:
            source_df = pd.read_csv(data_path)
        
        # Check which columns actually exist in source
        available_missing = [f for f in missing_features if f in source_df.columns]
        if not available_missing:
            logger.warning(f"None of the missing features found in source data: {data_path}")
            return stage_output
        
        logger.info(f"Loading {len(available_missing)} features from source: {sorted(available_missing)}")
        
        # Deduplicate source data by impression_id (keep first occurrence)
        cols_to_keep = [impression_id_col] + available_missing
        source_df = source_df[cols_to_keep].drop_duplicates(subset=[impression_id_col], keep='first')
        
        # Merge with existing user features
        merge_key = 'request_id' if 'request_id' in user_features_df.columns else impression_id_col
        source_merge_key = impression_id_col
        
        # Rename source key if needed
        if merge_key != source_merge_key and merge_key in user_features_df.columns:
            source_df = source_df.rename(columns={source_merge_key: merge_key})
        
        # Perform left merge to add missing features
        enriched_df = user_features_df.merge(
            source_df, 
            on=merge_key, 
            how='left'
        )
        
        # Replace the user features DataFrame
        stage_output._user_features_df = enriched_df
        
        logger.info(f"Successfully enriched user features: {len(user_features_df)} -> {len(enriched_df)} rows, "
                   f"added {len(available_missing)} columns")
        
    except Exception as e:
        logger.error(f"Failed to enrich user features: {e}")
    
    return stage_output

def parse_pipeline_args():
    """Parse command line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description='Cloud-Device Recommendation Pipeline')
    parser.add_argument('--config', type=str, default='./config',
                       help='Configuration directory')
    parser.add_argument('--pipeline_id', type=str, default='pipeline_config',
                       help='Pipeline configuration ID')
    parser.add_argument('--dataset_id', type=str, default=None,
                       help='Dataset ID from dataset_config.yaml')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'retrieval', 'preranking', 'reranking', 'train', 'evaluate', 'joint_train', 'dtcn_preranking'],
                       help='Execution mode')
    parser.add_argument('--stage', type=str, default=None,
                       choices=['retrieval', 'preranking', 'reranking'],
                       help='Stage name for train/evaluate mode')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--prev_output_path', type=str, default=None,
                       help='Path to directory containing previous stage outputs (stage_outputs/). '
                            'Will auto-load {stage}_valid_stage_output.pkl and {stage}_test_stage_output.pkl')
    parser.add_argument('--experiment_id', type=str, default=None,
                       help='Unique identifier for the current experiment run')
    parser.add_argument('--seed', type=int, default=2024,
                       help='Random seed')
    
    # Validation control arguments
    parser.add_argument('--run_retrieval_test', type=int, default=0,
                       help='Whether to run retrieval test evaluation (1=yes, 0=no)')
    parser.add_argument('--run_preranking_test', type=int, default=0,
                       help='Whether to run preranking test evaluation (1=yes, 0=no)')
    parser.add_argument('--run_reranking_test', type=int, default=0,
                       help='Whether to run reranking test evaluation (1=yes, 0=no)')

    parser.add_argument('--n_rows', type=int, default=None,
                       help='Override debug n_rows (set to small number for quick testing)')
    parser.add_argument('--save_stage_outputs', type=bool, default=False,
                       help='Save intermediate stage outputs (StageOutput) to disk for later reuse')
    return parser.parse_args()


