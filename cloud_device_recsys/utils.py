import logging
import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import argparse
from typing import Tuple, List, Dict, Any, Optional

from .pipeline.stage_output import CandidateSet, CandidateItem, StageOutput


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


def load_pipeline_config(config_dir: str, pipeline_id: str) -> dict:
    """Load pipeline configuration"""
    config_path = os.path.join(config_dir, f"{pipeline_id}.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "pipeline_config.yaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


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
        original_item_pool = os.path.join(original_data_root, f"{paths['item_pool_file']}.parquet")
        
        # Create debug datasets and update paths
        paths['train_path'] = create_debug_dataset(original_train, debug_n_rows, os.path.join(debug_dir, 'train.parquet'), logger)
        paths['valid_path'] = create_debug_dataset(original_valid, debug_n_rows, os.path.join(debug_dir, 'valid.parquet'), logger)
        paths['test_path'] = create_debug_dataset(original_test, debug_n_rows, os.path.join(debug_dir, 'test.parquet'), logger)
        # paths['item_pool_path'] = create_debug_dataset(original_item_pool, debug_n_rows, os.path.join(debug_dir, f"{paths['item_pool_file']}.parquet"), logger)
    
    return paths


def evaluate_stage_output(stage_output, logger=None, metrics_k=None):
    """
    Helper to evaluate StageOutput (CandidateSets).
    
    Computes Recall@K and nDCG@K for each K in metrics_k.
    
    Args:
        stage_output: StageOutput object containing candidate sets
        logger: Optional logger instance
        metrics_k: List of K values for metrics computation
        
    Returns:
        dict of metric names to values
    """
    if not stage_output or not stage_output.candidate_sets:
        return {}
    
    total_recall = {k: 0.0 for k in metrics_k}
    total_ndcg = {k: 0.0 for k in metrics_k}
    num_queries = 0
    
    for cs in stage_output.candidate_sets:
        # Sort candidates by score descending
        sorted_candidates = sorted(cs.candidates, key=lambda x: x.score, reverse=True)
        sorted_labels = [c.label for c in sorted_candidates]
        
        true_relevant_count = sum(sorted_labels)
        if true_relevant_count == 0:
            continue
            
        num_queries += 1
        
        for k in metrics_k:
            top_k = sorted_labels[:k]
            hits = sum(top_k)
            total_recall[k] += (hits / true_relevant_count)
            
            dcg = 0.0
            for rank, label in enumerate(top_k, 1):
                if label == 1:
                    dcg += 1.0 / np.log2(rank + 1)
            
            idcg = 0.0
            for rank in range(1, min(int(true_relevant_count), k) + 1):
                idcg += 1.0 / np.log2(rank + 1)
                
            if idcg > 0:
                total_ndcg[k] += (dcg / idcg)
                
    metrics = {}
    if num_queries > 0:
        for k in metrics_k:
            metrics[f"{stage_output.stage_name}_pipeline_Recall@{k}"] = total_recall[k] / num_queries
            metrics[f"{stage_output.stage_name}_pipeline_nDCG@{k}"] = total_ndcg[k] / num_queries
            
    if logger:
        logger.info(f"Pipeline Metrics for {stage_output.stage_name}: {metrics}")
        
    return metrics


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
    Compute Recall@K and nDCG@K for a single query.
    
    Args:
        scores: Predicted scores for each item
        labels: Ground truth labels (1 for relevant, 0 otherwise)
        metrics_k: List of K values for metrics
        
    Returns:
        Dict of metric names to values
    """
    # Sort by score descending
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    
    true_relevant_count = np.sum(sorted_labels)
    if true_relevant_count == 0:
        return {}
    
    metrics = {}
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
    Save a StageOutput object to disk using pickle format.
    
    Args:
        stage_output: StageOutput object to save
        output_dir: Directory to save the output file
        prefix: Prefix for the output filename (e.g., 'retrieval_valid', 'preranking_test')
        logger: Optional logger instance
        
    Returns:
        Path to the saved file
    """
    if stage_output is None:
        raise ValueError('StageOutput object must not be None')

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{prefix}_stage_output.pkl"
    filepath = os.path.join(output_dir, filename)
    
    stage_output.save(filepath)
    
    if logger:
        logger.info(f"Saved {stage_output.stage_name} output ({len(stage_output.candidate_sets)} requests, "
                   f"{stage_output.get_total_candidates()} total candidates) to {filepath}")
    
    return filepath


def load_stage_output(filepath: str, logger: logging.Logger = None) -> Optional[StageOutput]:
    """
    Load a StageOutput object from disk.
    
    Args:
        filepath: Path to the pickle file
        logger: Optional logger instance
        
    Returns:
        StageOutput object, or None if loading fails
    """
    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")

    try:
        stage_output = StageOutput.load(filepath)
        if logger:
            logger.info(f"Loaded {stage_output.stage_name} output from {filepath}: "
                       f"{len(stage_output.candidate_sets)} requests, "
                       f"{stage_output.get_total_candidates()} total candidates")
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
    Load valid and test stage outputs from a directory.
    
    Args:
        output_dir: Directory containing stage output files (e.g., './outputs/exp_xxx/stage_outputs')
        prev_stage_name: Name of the previous stage (e.g., 'retrieval' or 'preranking')
        logger: Optional logger instance
        
    Returns:
        Tuple of (valid_output, test_output), both can be None if loading fails
    """
    valid_path = os.path.join(output_dir, f"{prev_stage_name}_valid_stage_output.pkl")
    test_path = os.path.join(output_dir, f"{prev_stage_name}_test_stage_output.pkl")
    
    valid_output = load_stage_output(valid_path, logger)
    test_output = load_stage_output(test_path, logger)
    
    return valid_output, test_output


def select_top_k_candidates(
    request_id: str,
    user_id: Any,
    user_features: Dict[str, Any],
    candidates: List[CandidateItem],
    scores: np.ndarray,
    valid_indices: List[int],
    top_k: int,
    source_stage: str
) -> CandidateSet:
    """
    Select top-K candidates based on scores and create a CandidateSet.
    
    Args:
        request_id: Request/impression ID
        user_id: User ID
        user_features: User feature dict
        candidates: Original list of CandidateItem objects
        scores: Predicted scores (aligned with valid_indices)
        valid_indices: Indices of items that were scored
        top_k: Number of top candidates to select
        source_stage: Name of the current stage
        
    Returns:
        CandidateSet with top-K candidates
    """
    # Update scores for valid candidates
    scored_candidates = []
    for idx, score in zip(valid_indices, scores):
        cand = candidates[idx]
        cand.score = float(score)
        scored_candidates.append(cand)
    
    # Sort by score descending and take top-K
    scored_candidates.sort(key=lambda x: x.score, reverse=True)
    top_k_candidates = scored_candidates[:top_k]
    
    return CandidateSet(
        request_id=request_id,
        user_id=user_id,
        user_features=user_features,
        candidates=top_k_candidates,
        source_stage=source_stage
    )


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
    
    parser.add_argument('--n_rows', type=int, default=None,
                       help='Override debug n_rows (set to small number for quick testing)')
    parser.add_argument('--save_stage_outputs', type=bool, default=True,
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
    inference_batch_size: int = 50000,
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
    logger.info(f"{'/'.join(action)} {len(input_data.candidate_sets)} requests...")

    if item_features_df is None:
        logger.error("Item features DataFrame is None. Cannot rank candidates.")
        return output, metrics

    model.eval()
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
    
    item_id_col = getattr(feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
    item_feature_cols = list(item_features_df.columns)
    
    # ========== Phase 1: Collect all candidates and build metadata ==========
    # Flatten all candidates from all requests
    all_item_ids = []
    all_labels = []
    request_offsets = [0]  # Start index for each request
    request_metadata = []  # Store (request_id, user_id, user_features, candidates) per request
    
    for cs in input_data.candidate_sets:
        if not cs.candidates:
            raise ValueError(f"Please ensure that input StageOutput contains candidates for all requests. ")

        item_ids = [c.item_id for c in cs.candidates]
        labels = [c.label if c.label is not None else 0 for c in cs.candidates]
        
        all_item_ids.extend(item_ids)
        all_labels.extend(labels)
        request_offsets.append(request_offsets[-1] + len(item_ids))
        request_metadata.append((cs.request_id, cs.user_id, cs.user_features, cs.candidates))
    
    total_candidates = len(all_item_ids)
    num_requests = len(request_metadata)
    
    if total_candidates == 0:
        if return_output:
            for req_id, user_id, user_feats, _ in request_metadata:
                output.candidate_sets.append(CandidateSet(
                    request_id=req_id, user_id=user_id,
                    user_features=user_feats, candidates=[], source_stage=stage_name
                ))
        return output, metrics
    logger.info(f"Finish collecting {total_candidates} candidates from {num_requests} requests.")
    # ========== Phase 2: Vectorized item feature lookup ==========
    all_item_ids_arr = np.array(all_item_ids)
    all_labels_arr = np.array(all_labels)
    
    # Find which items exist in the item pool
    valid_mask = np.isin(all_item_ids_arr, item_features_df.index)
    valid_item_ids = all_item_ids_arr[valid_mask]
    
    # Batch lookup item features
    if len(valid_item_ids) > 0:
        item_features_lookup = item_features_df.loc[valid_item_ids]
    else:
        logger.warning("No valid items found in item pool.")
        if return_output:
            for req_id, user_id, user_feats, _ in request_metadata:
                output.candidate_sets.append(CandidateSet(
                    request_id=req_id, user_id=user_id,
                    user_features=user_feats, candidates=[], source_stage=stage_name
                ))
        return output, metrics
    
    # ========== Phase 3: Prepare user feature metadata (Deferred to Phase 4) ==========
    # User features are now built lazily per chunk in Phase 4 to reduce peak memory
    valid_global_indices = np.where(valid_mask)[0]
    num_valid = len(valid_global_indices)
    request_idx_for_valid = np.searchsorted(request_offsets[1:], valid_global_indices, side='right')
    user_feature_names = [k for k in request_metadata[0][2].keys()]
    
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
        sample_req_idx = request_idx_for_valid[0]
        for feat_name in user_feature_names:
            sample_val = request_metadata[sample_req_idx][2].get(feat_name, 0)
            user_feat_is_sequence[feat_name] = isinstance(sample_val, np.ndarray) and sample_val.ndim > 0
    
    # Check for optional FP16 inference
    use_fp16 = kwargs.get('use_fp16', True) and device.type == 'cuda'
    
    for chunk_start in range(0, num_valid, inference_batch_size):
        chunk_end = min(chunk_start + inference_batch_size, num_valid)
        # Optimization 3: Build user features lazily per chunk (reduces peak memory)
        chunk_req_indices = request_idx_for_valid[chunk_start:chunk_end]
        batch_dict = {}
        
        for feat_name in user_feature_names:
            is_seq = user_feat_is_sequence.get(feat_name, False)
            if is_seq:
                user_vals = [request_metadata[req_idx][2].get(feat_name, np.zeros(1)) for req_idx in chunk_req_indices]
                batch_dict[feat_name] = np.stack(user_vals)
            else:
                user_vals = [request_metadata[req_idx][2].get(feat_name, 0) for req_idx in chunk_req_indices]
                arr = np.array(user_vals)
                if arr.dtype == np.object_:
                    try:
                        arr = arr.astype(np.float32)
                    except (ValueError, TypeError):
                        arr = arr.astype(np.int64)
                batch_dict[feat_name] = arr
        
        # Item features for this chunk (use pre-computed type info)
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
        
        # Optimization 4: Optimized tensor conversion (use from_numpy for contiguous arrays)
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
                v_typed = v.astype(np.int64) if v.dtype != np.int64 else v
                tensor_batch[k] = torch.from_numpy(v_typed).to(device, non_blocking=True)
            else:
                v_typed = v.astype(np.float32) if v.dtype != np.float32 else v
                tensor_batch[k] = torch.from_numpy(v_typed).to(device, non_blocking=True)
        
        # Model inference with optional FP16
        with torch.no_grad():
            if use_fp16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_dict = model(tensor_batch)
            else:
                pred_dict = model(tensor_batch)
            # Optimization 5: Use non_blocking for async transfer
            chunk_scores = pred_dict['y_pred'].detach().cpu().numpy().flatten()
        
        all_scores[chunk_start:chunk_end] = chunk_scores
        
        # Clean up to free memory
        del batch_dict, tensor_batch
        
    logger.info("Finish model inference for all valid candidates.")
    # ========== Phase 5: Scatter results back to requests ==========
    # Create mapping from valid indices back to original global indices
    global_to_valid_idx = np.full(total_candidates, -1, dtype=np.int64)
    global_to_valid_idx[valid_global_indices] = np.arange(num_valid)
    
    if compute_metrics:
        total_metrics = {f'{m}@{k}': 0.0 for k in metrics_k for m in ['Recall', 'nDCG']}
        num_queries = 0
    
    effective_top_k = kwargs.get('top_k', top_k)
    
    for req_idx, (req_id, user_id, user_feats, candidates) in enumerate(request_metadata):
        start_idx = request_offsets[req_idx]
        end_idx = request_offsets[req_idx + 1]
        
        if start_idx == end_idx:
            if return_output:
                output.candidate_sets.append(CandidateSet(
                    request_id=req_id, user_id=user_id,
                    user_features=user_feats, candidates=[], source_stage=stage_name
                ))
            continue
        
        # Get valid indices for this request
        req_global_indices = np.arange(start_idx, end_idx)
        req_valid_mask = valid_mask[start_idx:end_idx]
        req_valid_positions = np.where(req_valid_mask)[0]  # Positions within request
        
        if len(req_valid_positions) == 0:
            if return_output:
                output.candidate_sets.append(CandidateSet(
                    request_id=req_id, user_id=user_id,
                    user_features=user_feats, candidates=[], source_stage=stage_name
                ))
            continue
        
        # Get scores for this request's valid items
        req_valid_global = req_global_indices[req_valid_mask]
        req_valid_idx = global_to_valid_idx[req_valid_global]
        req_scores = all_scores[req_valid_idx]
        req_labels = all_labels_arr[start_idx:end_idx][req_valid_mask]
        
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
            
            # Map back to original candidate positions
            topk_positions = req_valid_positions[topk_local_idx]
            
            new_candidates = []
            for i, pos in enumerate(topk_positions):
                orig_cand = candidates[pos]
                score = req_scores[topk_local_idx[i]]
                new_candidates.append(CandidateItem(
                    item_id=orig_cand.item_id,
                    score=float(score),
                    label=orig_cand.label
                ))
            
            output.candidate_sets.append(CandidateSet(
                request_id=req_id, user_id=user_id,
                user_features=user_feats, candidates=new_candidates, source_stage=stage_name
            ))
        
        # Compute metrics if requested
        if compute_metrics and np.sum(req_labels) > 0:
            query_metrics = compute_ranking_metrics(req_scores, req_labels, metrics_k)
            if query_metrics:
                num_queries += 1
                for metric_name, value in query_metrics.items():
                    total_metrics[metric_name] += value
    
    # Finalize outputs
    if return_output:
        logger.info(f"{stage_name.capitalize()}: Total {input_data.get_total_candidates()} candidates -> "
                    f"Filtered {output.get_total_candidates()} candidates")
    
    if compute_metrics:
        if num_queries > 0:
            for metric_name in total_metrics:
                metrics[metric_name] = total_metrics[metric_name] / num_queries
        else:
            logger.warning("No valid queries with positive labels for ranking evaluation.")
    
    return output, metrics
