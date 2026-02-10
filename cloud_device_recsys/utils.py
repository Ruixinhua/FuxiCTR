from __future__ import annotations
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import argparse
from typing import Tuple, List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
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
    
    # Item pool path
    item_pool_config = dataset_config.get('item_pool', {})
    item_pool_file = item_pool_config.get('file', 'cand_item_list')
    item_pool_path = os.path.join(dataset_config.get('processed_data_root', data_dir), f'{item_pool_file}.parquet')
    
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


def compute_ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    metrics_k: List[int] = None
) -> Dict[str, float]:
    """
    Compute ranking metrics for a single query/user.
    
    Metrics computed:
        - Recall@K: Proportion of relevant items in top-K
        - nDCG@K: Normalized Discounted Cumulative Gain at K
        - MRR: Mean Reciprocal Rank (1 / rank of first relevant item)
        - gAUC: Group AUC (AUC computed for this single user's predictions)
    
    Args:
        scores: Predicted scores for each item (1D array/list)
        labels: Ground truth labels (1 for relevant, 0 otherwise, 1D array/list)
        metrics_k: List of K values for Recall@K and nDCG@K metrics
        
    Returns:
        Dict of metric names to values
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    
    # Sort by score descending with PESSIMISTIC tie-breaking:
    # When scores are tied, positive items (label=1) are ranked LAST among ties.
    # This is achieved by using a secondary sort key: +labels (so label=0 comes before label=1)
    # We use lexsort which sorts by the LAST key first (in ascending order).
    # lexsort((labels, -scores)) means: primary sort by -scores DESC, secondary by labels ASC
    # Since label=0 < label=1 in ascending order, negatives come first among ties.
    sorted_indices = np.lexsort((labels, -scores))
    sorted_labels = labels[sorted_indices]
    
    true_relevant_count = np.sum(sorted_labels)
    if true_relevant_count == 0:
        return {}
    
    metrics = {}
    
    # ========== MRR: Mean Reciprocal Rank ==========
    # Find the rank of the first relevant item (1-indexed)
    first_relevant_positions = np.where(sorted_labels == 1)[0]
    if len(first_relevant_positions) > 0:
        first_relevant_rank = first_relevant_positions[0] + 1  # Convert to 1-indexed
        metrics['MRR'] = 1.0 / first_relevant_rank
    else:
        metrics['MRR'] = 0.0
    
    # ========== gAUC: Group AUC ==========
    # AUC for this single user/group
    # Count pairs: (positive, negative) where positive has higher score
    num_positive = int(true_relevant_count)
    num_negative = len(labels) - num_positive
    
    if num_positive > 0 and num_negative > 0:
        # Efficient AUC calculation using ranking
        # For each positive sample, count how many negatives have lower scores
        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]
        
        # Count concordant pairs
        concordant = 0
        ties = 0
        for pos_score in positive_scores:
            concordant += np.sum(negative_scores < pos_score)
            ties += np.sum(negative_scores == pos_score)
        
        # AUC = (concordant + 0.5 * ties) / (num_positive * num_negative)
        metrics['gAUC'] = (concordant + 0.5 * ties) / (num_positive * num_negative)
    else:
        # Cannot compute AUC without both positive and negative samples
        metrics['gAUC'] = 0.0
    
    # ========== Recall@K and nDCG@K ==========
    if metrics_k is None:
        metrics_k = []
        
    for k in metrics_k:
        top_k = sorted_labels[:k]
        hits = np.sum(top_k)
        
        # Recall@K
        metrics[f'Recall@{k}'] = float(hits / true_relevant_count)
        
        # nDCG@K
        dcg = 0.0
        for rank, label in enumerate(top_k, 1):
            if label == 1:
                dcg += 1.0 / np.log2(rank + 1)
        
        # IDCG: best possible DCG with true_relevant_count items
        idcg = 0.0
        for rank in range(1, min(int(true_relevant_count), k) + 1):
            idcg += 1.0 / np.log2(rank + 1)
        
        if idcg > 0:
            metrics[f'nDCG@{k}'] = dcg / idcg
        else:
            metrics[f'nDCG@{k}'] = 0.0
    
    return metrics


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
        logger: logging.Logger = None
) -> Tuple[Optional[StageOutput], Optional[StageOutput]]:
    """
    Load valid and test stage outputs from a directory. Auto-detects Parquet or pickle format.

    Args:
        output_dir: Directory containing stage output files (e.g., './outputs/exp_xxx/stage_outputs')
        prev_stage_name: Name of the previous stage (e.g., 'retrieval' or 'preranking')
        logger: Optional logger instance

    Returns:
        Tuple of (valid_output, test_output), both can be None if loading fails
    """
    # Try Parquet format first (directory), then fall back to pickle (.pkl file)
    valid_parquet = os.path.join(output_dir, f"{prev_stage_name}_valid_stage_output")
    valid_pickle = os.path.join(output_dir, f"{prev_stage_name}_valid_stage_output.pkl")
    test_parquet = os.path.join(output_dir, f"{prev_stage_name}_test_stage_output")
    test_pickle = os.path.join(output_dir, f"{prev_stage_name}_test_stage_output.pkl")
    
    valid_path = valid_parquet if os.path.isdir(valid_parquet) else valid_pickle
    test_path = test_parquet if os.path.isdir(test_parquet) else test_pickle

    valid_output = load_stage_output(valid_path, logger)
    test_output = load_stage_output(test_path, logger)

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
            # Only load required columns for efficiency
            required_cols = [impression_id_col] + list(missing_features)
            source_df = pd.read_parquet(data_path, columns=required_cols)
        else:
            source_df = pd.read_csv(data_path, usecols=lambda c: c in [impression_id_col] + list(missing_features))
        
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
                       choices=['full', 'retrieval', 'preranking', 'reranking', 'train', 'evaluate'],
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


def process_and_rank_candidates(
    model: Any,
    feature_map: Any,
    input_data: StageOutput,
    item_features_df: pd.DataFrame,
    stage_name: str,
    return_output: bool = True,
    compute_metrics: bool = False,
    metrics_k: List[int] = None,
    top_k: int = 100,
    logger: logging.Logger = None,
    inference_batch_size: int = 100000,
    **kwargs
) -> Tuple[Optional[StageOutput], Dict[str, float]]:
    """
    Batch-optimized core logic for ranking candidates in Preranking and Reranking stages.
    
    Optimizations:
    1. Batches candidates across all requests for efficient model inference
    2. Vectorized item feature lookup using DataFrame indexing
    3. Chunked processing to manage memory
    4. Uses np.argpartition for efficient top-K selection
    
    Args:
        model: The trained model (must implement forward/predict logic)
        feature_map: FeatureMap object
        input_data: StageOutput from previous stage
        item_features_df: DataFrame containing item features
        stage_name: Name of the current stage (e.g., 'preranking', 'reranking')
        return_output: If True, generate and return StageOutput with top-K candidates
        compute_metrics: If True, compute and return ranking metrics
        metrics_k: K values for evaluation metrics (required if compute_metrics=True)
        top_k: Top-K candidates to select (used when return_output=True)
        logger: Logger instance
        inference_batch_size: Number of candidates to process per batch (default 50K)
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (StageOutput or None, metrics_dict)
    """
    if logger is None:
        logger = logging.getLogger(stage_name)
    
    from .pipeline.stage_output import StageOutput  # Runtime import to avoid circular dependency
    
    # Validate arguments
    if compute_metrics and not metrics_k:
        metrics_k = [10, 50, 100]
        
    # Initialize outputs
    output = StageOutput(stage_name=stage_name) if return_output else None
    metrics = {}
    
    action = []
    if return_output:
        action.append("Processing")
    if compute_metrics:
        action.append("Evaluating")
    logger.info(f"{'/'.join(action)} {input_data.get_num_requests()} requests...")

    if item_features_df is None:
        logger.error("Item features DataFrame is None. Cannot rank candidates.")
        return output, metrics

    model.eval()
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
    
    item_id_col = getattr(feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
    item_feature_cols = list(item_features_df.columns)
    
    # ========== Phase 1: Get data from DataFrame (fully DataFrame-based) ==========
    candidates_df = input_data.candidates_df
    user_features_df = input_data.user_features_df
    
    if len(candidates_df) == 0:
        logger.warning("No candidates in input data.")
        return output, metrics
    
    # 1. Build user features lookup from DataFrame (Optimized)
    user_features_dict = {}
    if user_features_df is not None and not user_features_df.empty:
        # Use to_dict('records') for faster iteration than iterrows
        records = user_features_df.to_dict('records')
        for row in records:
            req_id = row.pop('request_id')
            user_id = row.pop('user_id', None)
            # Filter None/NaN values
            features = {k: v for k, v in row.items() if v is not None and (isinstance(v, np.ndarray) or not pd.isna(v))}
            user_features_dict[req_id] = (user_id, features)

    # 2. Group candidates by request_id (Optimized)
    # Sort by request_id to ensure contiguous blocks. Use mergesort for stability.
    candidates_df = candidates_df.sort_values(by='request_id', kind='mergesort')
    
    request_ids_arr = candidates_df['request_id'].values
    all_item_ids = candidates_df['item_id'].values
    all_labels = candidates_df['label'].fillna(0).astype(int).values
    
    # Use np.unique to count occurrences of each request_id
    # Since we sorted, unique_request_ids will be sorted
    unique_request_ids, request_counts = np.unique(request_ids_arr, return_counts=True)
    
    # Compute request offsets using cumsum (Vectorized)
    request_offsets = np.zeros(len(unique_request_ids) + 1, dtype=int)
    request_offsets[1:] = np.cumsum(request_counts)
    
    # Build request metadata
    # We iterate over unique_request_ids (M items), dict lookup is O(1)
    request_metadata = []
    for req_id in unique_request_ids:
        user_id, user_feats = user_features_dict.get(req_id, (None, {}))
        request_metadata.append((req_id, user_id, user_feats))
    
    total_candidates = len(all_item_ids)
    num_requests = len(request_metadata)
    
    if total_candidates == 0:
        return output, metrics
    
    logger.info(f"Finish collecting {total_candidates} candidates from {num_requests} requests.")
    # ========== Phase 2: Vectorized item feature lookup ==========
    all_item_ids_arr = np.array(all_item_ids)
    all_labels_arr = np.array(all_labels)
    
    # Ensure dtype consistency between candidate item IDs and item_features_df index
    # This fixes issues where item IDs may be stored as float (e.g., 182966.0) but index is int64
    index_dtype = item_features_df.index.dtype
    if all_item_ids_arr.dtype != index_dtype:
        logger.info(f"Converting candidate item_ids from {all_item_ids_arr.dtype} to {index_dtype}")
        try:
            all_item_ids_arr = all_item_ids_arr.astype(index_dtype)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert item_ids dtype: {e}")
    
    # Find which items exist in the item pool
    valid_mask = np.isin(all_item_ids_arr, item_features_df.index)
    valid_item_ids = all_item_ids_arr[valid_mask]
    
    # Debug: Check how many candidates are filtered out
    num_filtered = total_candidates - len(valid_item_ids)
    if num_filtered > 0:
        # Get filtered items for debugging
        filtered_mask = ~valid_mask
        filtered_item_ids = all_item_ids_arr[filtered_mask]
        filtered_labels = all_labels_arr[filtered_mask]
        num_positive_filtered = np.sum(filtered_labels == 1)
        
        logger.warning(f"Phase 2: {num_filtered}/{total_candidates} ({100*num_filtered/total_candidates:.1f}%) "
                      f"candidates filtered out (item not in item_features_df)")
        logger.warning(f"  - Positive items filtered: {num_positive_filtered}/{num_filtered}")
        logger.warning(f"  - Sample filtered item IDs: {filtered_item_ids[:5].tolist()}")
        logger.warning(f"  - item_features_df index dtype: {item_features_df.index.dtype}, "
                      f"candidate item_id dtype: {all_item_ids_arr.dtype}")
    
    # Batch lookup item features
    if len(valid_item_ids) > 0:
        item_features_lookup = item_features_df.loc[valid_item_ids]
    else:
        logger.warning("No valid items found in item pool.")
        return output, metrics
    
    # ========== Phase 3: Prepare user feature metadata ==========
    valid_global_indices = np.where(valid_mask)[0]
    num_valid = len(valid_global_indices)
    request_idx_for_valid = np.searchsorted(request_offsets[1:], valid_global_indices, side='right')
    user_feature_names = list(request_metadata[0][2].keys()) if request_metadata and request_metadata[0][2] else []
    logger.info("Finish preparing user feature metadata for valid candidates.")
    # ========== Phase 4: Chunked model inference (OPTIMIZED) ==========
    all_scores = np.zeros(num_valid, dtype=np.float32)
    
    # Optimization 1: Pre-compute column type info to avoid repeated dtype detection
    col_type_info = {}  # {col: ('sequence'|'scalar'|'direct', target_dtype)}
    for col in item_feature_cols:
        sample_values = item_features_lookup[col].head(1).values
        if len(sample_values) == 0:
            col_type_info[col] = ('direct', np.float32)
            continue
        if sample_values.dtype == np.object_:
            sample = sample_values[0]
            if isinstance(sample, (np.ndarray, list)) and hasattr(sample, '__len__') and len(sample) > 1:
                col_type_info[col] = ('sequence', None)
            else:
                col_type_info[col] = ('scalar', np.float32)
        else:
            col_type_info[col] = ('direct', sample_values.dtype)
    
    # Optimization 2: Pre-compute user feature type info
    user_feat_is_sequence = {}
    if user_feature_names and len(request_idx_for_valid) > 0:
        sample_features = request_metadata[0][2]
        for feat_name in user_feature_names:
            sample_val = sample_features.get(feat_name, 0)
            user_feat_is_sequence[feat_name] = isinstance(sample_val, np.ndarray) and sample_val.ndim > 0
    
    # Check for optional FP16 inference
    use_fp16 = kwargs.get('use_fp16', True) and device.type == 'cuda'
    logger.info(f"Using {use_fp16} fp16 precision.")
    import time
    
    # ========== OPTIMIZATION: Vectorized user feature broadcasting ==========
    # Key insight: 29M candidates share only 29K unique requests' user features.
    # Instead of building 29M-row arrays, we build 29K-row arrays (one per request)
    # and use numpy advanced indexing to broadcast user features to candidates.
    t_precompute_start = time.time()
    
    num_requests = len(request_metadata)
    
    # Build compact user feature arrays indexed by request_idx (only 29K rows)
    request_user_features = {}  # {feat_name: np.ndarray of shape (num_requests, ...)}
    request_user_ids = None
    
    if user_feature_names and num_requests > 0:
        # Build user_id array indexed by request (29K rows, not 29M)
        request_user_ids = np.array([request_metadata[req_idx][1] for req_idx in range(num_requests)])
        
        # Build each user feature array indexed by request
        for feat_name in user_feature_names:
            is_seq = user_feat_is_sequence.get(feat_name, False)
            if is_seq:
                # Sequence features: need np.stack
                user_vals = [request_metadata[req_idx][2].get(feat_name, np.zeros(1)) 
                             for req_idx in range(num_requests)]
                request_user_features[feat_name] = np.stack(user_vals)
            else:
                # Scalar features: simple array
                user_vals = [request_metadata[req_idx][2].get(feat_name, 0) 
                             for req_idx in range(num_requests)]
                arr = np.array(user_vals)
                if arr.dtype == np.object_:
                    try:
                        arr = arr.astype(np.float32)
                    except (ValueError, TypeError):
                        arr = arr.astype(np.int64)
                request_user_features[feat_name] = arr
    
    precompute_time = time.time() - t_precompute_start
    logger.info(f"Pre-computed {len(request_user_features)} user feature arrays for {num_requests} requests in {precompute_time:.2f}s")
    
    # Fine-grained timing for bottleneck analysis
    timing_stats = {
        'user_feature_prep': 0.0,
        'item_feature_extract': 0.0,
        'tensor_conversion': 0.0,
        'model_forward': 0.0,
        'result_transfer': 0.0,
    }
    inference_all_start = time.time()
    num_batches = 0
    
    for chunk_start in range(0, num_valid, inference_batch_size):
        chunk_end = min(chunk_start + inference_batch_size, num_valid)
        num_batches += 1
        
        # ===== TIMING: User feature preparation (NOW VECTORIZED BROADCASTING) =====
        t_user_start = time.time()
        batch_dict = {}
        
        # Get request indices for this chunk of candidates
        chunk_req_indices = request_idx_for_valid[chunk_start:chunk_end]
        
        # Fast numpy advanced indexing: broadcast user features from requests to candidates
        for feat_name in user_feature_names:
            batch_dict[feat_name] = request_user_features[feat_name][chunk_req_indices]
        
        # Add user_id to batch (needed by reranking models)
        if request_user_ids is not None:
            batch_dict['user_id'] = request_user_ids[chunk_req_indices]
        else:
            user_ids = [request_metadata[req_idx][1] for req_idx in chunk_req_indices]
            batch_dict['user_id'] = np.array(user_ids)
        timing_stats['user_feature_prep'] += time.time() - t_user_start
        
        # ===== TIMING: Item feature extraction =====
        t_item_start = time.time()
        chunk_item_features = item_features_lookup.iloc[chunk_start:chunk_end]
        for col in item_feature_cols:
            col_type, target_dtype = col_type_info[col]
            col_values = chunk_item_features[col].values
            
            if col_type == 'sequence':
                try:
                    batch_dict[col] = np.stack(col_values)
                except ValueError:
                    batch_dict[col] = np.array([np.array(x) for x in col_values])
            elif col_type == 'scalar':
                try:
                    batch_dict[col] = col_values.astype(np.float32)
                except (ValueError, TypeError):
                    try:
                        batch_dict[col] = col_values.astype(np.int64)
                    except (ValueError, TypeError):
                        batch_dict[col] = np.array([int(x) if x is not None else 0 for x in col_values], dtype=np.int64)
            else:  # 'direct'
                batch_dict[col] = col_values
        
        # Add item_id column if needed
        if item_id_col not in batch_dict and item_id_col in feature_map.features:
            batch_dict[item_id_col] = valid_item_ids[chunk_start:chunk_end]
        timing_stats['item_feature_extract'] += time.time() - t_item_start
        
        # ===== TIMING: Tensor conversion =====
        t_tensor_start = time.time()
        tensor_batch = {}
        for k, v in batch_dict.items():
            if k not in feature_map.features:
                continue
            ftype = feature_map.features[k]['type']
            
            # Ensure array is contiguous and use from_numpy to avoid copy
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            if not v.flags['C_CONTIGUOUS']:
                v = np.ascontiguousarray(v)
            
            if ftype == 'sequence' or ftype == 'categorical':
                # Handle None values in object arrays before conversion
                if v.dtype == np.object_:
                    v = np.array([x if x is not None else 0 for x in v.flat]).reshape(v.shape)
                v_typed = v.astype(np.int64) if v.dtype != np.int64 else v
                tensor_batch[k] = torch.from_numpy(v_typed).to(device, non_blocking=True)
            else:
                # Handle None values in object arrays before conversion
                if v.dtype == np.object_:
                    v = np.array([x if x is not None else 0.0 for x in v.flat]).reshape(v.shape)
                v_typed = v.astype(np.float32) if v.dtype != np.float32 else v
                tensor_batch[k] = torch.from_numpy(v_typed).to(device, non_blocking=True)
        timing_stats['tensor_conversion'] += time.time() - t_tensor_start
        
        # ===== TIMING: Model forward pass =====
        t_forward_start = time.time()
        with torch.no_grad():
            if use_fp16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_dict = model(tensor_batch)
            else:
                pred_dict = model(tensor_batch)
        timing_stats['model_forward'] += time.time() - t_forward_start
        
        # ===== TIMING: Result transfer to CPU =====
        t_transfer_start = time.time()
        chunk_scores = pred_dict['y_pred'].detach().cpu().numpy().flatten()
        timing_stats['result_transfer'] += time.time() - t_transfer_start
        
        all_scores[chunk_start:chunk_end] = chunk_scores
        
        # Clean up to free memory
        del batch_dict, tensor_batch
    
    total_inference_time = time.time() - inference_all_start
    
    # Log detailed timing breakdown
    logger.info(f"===== Inference Timing Breakdown ({num_batches} batches, batch_size={inference_batch_size}) =====")
    logger.info(f"  User feature prep:    {timing_stats['user_feature_prep']:8.2f}s ({100*timing_stats['user_feature_prep']/total_inference_time:5.1f}%)")
    logger.info(f"  Item feature extract: {timing_stats['item_feature_extract']:8.2f}s ({100*timing_stats['item_feature_extract']/total_inference_time:5.1f}%)")
    logger.info(f"  Tensor conversion:    {timing_stats['tensor_conversion']:8.2f}s ({100*timing_stats['tensor_conversion']/total_inference_time:5.1f}%)")
    logger.info(f"  Model forward:        {timing_stats['model_forward']:8.2f}s ({100*timing_stats['model_forward']/total_inference_time:5.1f}%)")
    logger.info(f"  Result transfer:      {timing_stats['result_transfer']:8.2f}s ({100*timing_stats['result_transfer']/total_inference_time:5.1f}%)")
    logger.info(f"  Total inference time: {total_inference_time:8.2f}s")
    logger.info(f"Finish model inference for all valid candidates.")
    # ========== Phase 5: Scatter results back to requests (OPTIMIZED - DataFrame output) ==========
    # Create mapping from valid indices back to original global indices
    global_to_valid_idx = np.full(total_candidates, -1, dtype=np.int64)
    global_to_valid_idx[valid_global_indices] = np.arange(num_valid)
    num_queries = 0
    valid_queries = 0
    if compute_metrics:
        total_metrics = {f'{m}@{k}': 0.0 for k in metrics_k for m in ['Recall', 'nDCG']}
        total_metrics['MRR'] = 0.0
        total_metrics['gAUC'] = 0.0

    effective_top_k = kwargs.get('top_k', top_k)
    
    # Build output DataFrame directly (fast path)
    output_candidates_data = []
    
    for req_idx, (req_id, user_id, user_feats) in enumerate(request_metadata):
        start_idx = int(request_offsets[req_idx])
        end_idx = int(request_offsets[req_idx + 1])
        
        if start_idx == end_idx:
            continue
        
        # Get valid indices for this request
        req_global_indices = np.arange(start_idx, end_idx)
        req_valid_mask = valid_mask[start_idx:end_idx]
        req_valid_positions = np.where(req_valid_mask)[0]
        
        if len(req_valid_positions) == 0:
            continue
        
        # Get scores for this request's valid items
        req_valid_global = req_global_indices[req_valid_mask]
        req_valid_idx = global_to_valid_idx[req_valid_global]
        req_scores = all_scores[req_valid_idx]
        req_labels = all_labels_arr[start_idx:end_idx][req_valid_mask]
        req_item_ids = all_item_ids_arr[start_idx:end_idx][req_valid_mask]
        
        # Generate output if requested
        if return_output:
            num_to_select = min(effective_top_k, len(req_valid_positions))
            
            # Use argpartition for efficient top-K
            if num_to_select < len(req_scores):
                topk_local_idx = np.argpartition(-req_scores, num_to_select - 1)[:num_to_select]
                topk_scores = req_scores[topk_local_idx]
                sorted_order = np.argsort(-topk_scores)
                topk_local_idx = topk_local_idx[sorted_order]
            else:
                topk_local_idx = np.argsort(-req_scores)
            
            # Build output rows (DataFrame-first)
            for i in topk_local_idx:
                output_candidates_data.append({
                    'request_id': req_id,
                    'item_id': req_item_ids[i],
                    'score': float(req_scores[i]),
                    'label': int(req_labels[i]) if not pd.isna(req_labels[i]) else None
                })
        num_queries += 1
        # Compute metrics if requested
        if compute_metrics and np.sum(req_labels) > 0:
            num_candidates = len(req_labels)
            num_positive = int(np.sum(req_labels))
            num_negative = num_candidates - num_positive
            valid_queries += 1
            # Detailed debugging for first 3 queries
            if num_queries < 3:
                pos_scores = req_scores[req_labels == 1]
                neg_scores = req_scores[req_labels == 0]
                # Pessimistic rank: all negatives with score >= positive come first
                pos_rank_pessimistic = int(np.sum(neg_scores >= pos_scores[0])) + 1
                logger.info(f"[DEBUG] Query {req_id}: {num_candidates} cands ({num_positive} pos, {num_negative} neg)")
                logger.info(f"[DEBUG]   Positive score: {pos_scores[0]:.6f}")
                logger.info(f"[DEBUG]   Negative scores: min={neg_scores.min():.6f}, max={neg_scores.max():.6f}, mean={neg_scores.mean():.6f}")
                logger.info(f"[DEBUG]   Positive rank (pessimistic): {pos_rank_pessimistic} (1=best)")
                logger.info(f"[DEBUG]   # negatives >= pos: {np.sum(neg_scores >= pos_scores[0])}, # negatives < pos: {np.sum(neg_scores < pos_scores[0])}")
            
            query_metrics = compute_ranking_metrics(req_scores, req_labels, metrics_k)
            
            # Log individual query metrics for first 3 queries
            if num_queries < 3 and query_metrics:
                logger.info(f"[DEBUG]   Query metrics: MRR={query_metrics.get('MRR', 0):.4f}, gAUC={query_metrics.get('gAUC', 0):.4f}")
            
            if query_metrics:
                for metric_name, value in query_metrics.items():
                    total_metrics[metric_name] += value
    
    # Finalize outputs using DataFrame-first API
    if return_output:
        output_candidates_df = pd.DataFrame(output_candidates_data)
        output = StageOutput.from_dataframes(
            stage_name=stage_name,
            candidates_df=output_candidates_df,
            user_features_df=input_data.user_features_df,
            metrics={},
            metadata={}
        )
        logger.info(f"{stage_name.capitalize()}: Total {input_data.get_total_candidates()} candidates -> "
                    f"Filtered {output.get_total_candidates()} candidates")
    
    if compute_metrics:
        if num_queries > 0:
            for metric_name in total_metrics:
                if metric_name == 'gAUC':  # gAUC should averaged across valid queries
                    metrics[metric_name] = total_metrics['gAUC'] / valid_queries if valid_queries > 0 else 0.0
                else:
                    metrics[metric_name] = total_metrics[metric_name] / num_queries
        else:
            logger.warning("No valid queries with positive labels for ranking evaluation.")
    
    return output, metrics
