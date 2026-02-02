import logging
import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Tuple, List, Dict, Any

from .pipeline.stage_output import CandidateSet, CandidateItem

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
    # if debug_n_rows:
    #     item_pool_path = os.path.join(debug_dir, f'{item_pool_file}.parquet')
    # else:
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


def evaluate_stage_output(stage_output, logger=None, metrics_k=[10, 50, 100]):
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
    metrics_k: List[int] = [10, 50, 100]
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
