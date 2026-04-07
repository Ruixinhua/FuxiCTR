import logging
import numpy as np
import pandas as pd
import torch
from torchmetrics.functional.classification import binary_auroc

from typing import Tuple, List, Dict, Any, Optional
from .pipeline.stage_output import StageOutput

def compute_ranking_metrics(
        scores: np.ndarray,
        labels: np.ndarray,
        metrics_k: List[int] = None,
        pre_sorted: bool = False,
        top_k_for_metrics: Optional[int] = None,
        item_features_arr: Optional[np.ndarray] = None,
        sorted_order: Optional[np.ndarray] = None,
        full_pool_scores: Optional[np.ndarray] = None,
        full_pool_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute ranking metrics for a single query/user.

    Metrics computed:
        - Recall@K: Proportion of relevant items in top-K
        - nDCG@K: Normalized Discounted Cumulative Gain at K
        - Diversity@K: 1 - average categorical pairwise similarity of top-K items
        - MRR: Mean Reciprocal Rank over all items
        - gAUC: Group AUC over all items
        - MRR@<top_k_for_metrics>: MRR restricted to top-K items
        - gAUC@<top_k_for_metrics>: Group AUC restricted to top-K items

    Args:
        scores: Predicted scores for each item (1D array/list).
                Used for Recall@K/nDCG@K. Also used for gAUC/MRR unless
                full_pool_scores/full_pool_labels are provided.
        labels: Ground truth labels (1 for relevant, 0 otherwise, 1D array/list)
        metrics_k: List of K values for Recall@K, nDCG@K and Diversity@K metrics
        pre_sorted: If True, scores and labels are already sorted by descending score
                    (e.g., with random tie-breaking from process_and_rank_candidates).
                    The function will skip re-sorting and use the input order as-is.
                    If False (default), uses pessimistic tie-breaking (positives ranked last among ties).
        top_k_for_metrics: If set, additionally compute gAUC and MRR restricted to the top-K
                           items (by score). Useful for aligning preranking evaluation (1000 items)
                           with reranking evaluation (100 items). The suffix @K is appended to
                           distinguish these from the full-pool metrics.
        item_features_arr: Optional 2D array (num_candidates, num_features) of item categorical
                           features for this query (unsorted, indexed same as scores/labels).
                           Required to compute Diversity@K.
        sorted_order: Permutation array mapping sorted positions to original indices in
                      item_features_arr. Required when item_features_arr is provided.
        full_pool_scores: Optional pre-sorted scores from the full scoring pool (e.g. retrieval
                          ~1000 candidates). When provided, gAUC/MRR/gAUC@K/MRR@K are computed
                          on these instead of on ``scores``.
        full_pool_labels: Corresponding labels for full_pool_scores. Required when
                          full_pool_scores is provided.

    Returns:
        Dict of metric names to values
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)

    if pre_sorted:
        # Trust the input ordering — scores/labels are already sorted by descending score.
        # This is consistent with the candidate ranking in process_and_rank_candidates,
        # where ties are randomly ordered (stable sort preserves random order within ties).
        sorted_labels = labels
        sorted_scores = scores
    else:
        # Sort by score descending with PESSIMISTIC tie-breaking:
        # When scores are tied, positive items (label=1) are ranked LAST among ties.
        sorted_indices = np.lexsort((labels, -scores))
        sorted_labels = labels[sorted_indices]
        sorted_scores = scores[sorted_indices]

    if full_pool_labels is not None:
        true_relevant_count = np.sum(full_pool_labels)
    else:
        true_relevant_count = np.sum(sorted_labels)
    
    if true_relevant_count == 0:
        return {}

    metrics = {}

    def _compute_gauc(pos_scores, neg_scores):
        """Concordant-pair AUC for one user."""
        concordant = 0
        ties = 0
        for ps in pos_scores:
            concordant += np.sum(neg_scores < ps)
            ties += np.sum(neg_scores == ps)
        n_pos, n_neg = len(pos_scores), len(neg_scores)
        if n_pos == 0 or n_neg == 0:
            return 0.0
        return (concordant + 0.5 * ties) / (n_pos * n_neg)

    def _compute_mrr(lbl_array):
        """1 / rank of first positive in label array (already sorted)."""
        pos_positions = np.where(lbl_array == 1)[0]
        if len(pos_positions) == 0:
            return 0.0
        return 1.0 / (pos_positions[0] + 1)

    # ========== MRR & gAUC over ALL items ==========
    # When full_pool_scores/labels are provided, use them for gAUC/MRR
    # (full scoring pool e.g. retrieval ~1000) instead of scores/labels
    # (which may be a filtered subset e.g. preranking ~100).
    if full_pool_scores is not None and full_pool_labels is not None:
        fp_scores = np.asarray(full_pool_scores, dtype=np.float64)
        fp_labels = np.asarray(full_pool_labels)
        if pre_sorted:
            fp_sorted_labels = fp_labels
            fp_sorted_scores = fp_scores
        else:
            fp_order = np.lexsort((fp_labels, -fp_scores))
            fp_sorted_labels = fp_labels[fp_order]
            fp_sorted_scores = fp_scores[fp_order]
        gauc_scores = fp_scores
        gauc_labels = fp_labels
        gauc_sorted_labels = fp_sorted_labels
        gauc_sorted_scores = fp_sorted_scores
    else:
        gauc_scores = scores
        gauc_labels = labels
        gauc_sorted_labels = sorted_labels
        gauc_sorted_scores = sorted_scores

    metrics['MRR'] = _compute_mrr(gauc_sorted_labels)

    gauc_num_positive = int(np.sum(gauc_labels))
    gauc_num_negative = len(gauc_labels) - gauc_num_positive
    if gauc_num_positive > 0 and gauc_num_negative > 0:
        metrics['gAUC'] = _compute_gauc(
            gauc_scores[gauc_labels == 1], gauc_scores[gauc_labels == 0]
        )
    else:
        metrics['gAUC'] = 0.0

    # ========== MRR@K & gAUC@K restricted to top-K items ==========
    if top_k_for_metrics is not None and top_k_for_metrics <= len(gauc_sorted_labels):
        k = top_k_for_metrics
        topk_labels = gauc_sorted_labels[:k]
        topk_scores = gauc_sorted_scores[:k]
        # Fair comparison: only output metrics if the top-K subset has at least one positive
        if np.sum(topk_labels) > 0:
            metrics[f'MRR@{k}'] = _compute_mrr(topk_labels)
            topk_pos = topk_scores[topk_labels == 1]
            topk_neg = topk_scores[topk_labels == 0]
            metrics[f'gAUC@{k}'] = _compute_gauc(topk_pos, topk_neg)

    # ========== Recall@K, nDCG@K, and Diversity@K ==========
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

        # Diversity@K — 1 minus average pairwise categorical similarity
        if item_features_arr is not None and sorted_order is not None and k > 1:
            real_k = min(k, item_features_arr.shape[0])
            if real_k > 1:
                # Index into item_features_arr using sorted_order to get top-real_k features
                feats = item_features_arr[sorted_order[:real_k]]  # (real_k, num_features)
                sim_matrix = np.zeros((real_k, real_k), dtype=np.float64)
                for f in range(feats.shape[1]):
                    col = feats[:, f].reshape(-1, 1)
                    sim_matrix += (col == col.T).astype(np.float64)
                sim_matrix /= feats.shape[1]  # normalise to [0, 1]
                # Average pairwise similarity (exclude diagonal self-pairs)
                avg_sim = (sim_matrix.sum() - np.trace(sim_matrix)) / (real_k * (real_k - 1))
                metrics[f'Diversity@{k}'] = float(1.0 - avg_sim)
            else:
                metrics[f'Diversity@{k}'] = 0.0

    return metrics


def compute_pool_diversity(item_features_arr: np.ndarray, sample_size: int = 30000) -> float:
    """
    Compute diversity for the entire item pool or a large set of candidates.
    Diversity is calculated as 1 minus the average pairwise categorical similarity.
    For large arrays, this computes similarity on a random sample to maintain performance.

    Args:
        item_features_arr: 2D array (num_items, num_features) of item categorical features.
        sample_size: Maximum number of items to sample for calculation. O(N^2) complexity.

    Returns:
        Pool diversity score [0.0, 1.0]. Returns 0.0 if array is empty or 1 item.
    """
    n_items = item_features_arr.shape[0]
    if n_items <= 1:
        return 0.0

    # Sample if too large
    if n_items > sample_size:
        idx = np.random.choice(n_items, sample_size, replace=False)
        feats = item_features_arr[idx]
        real_n = sample_size
    else:
        feats = item_features_arr
        real_n = n_items

    sim_matrix = np.zeros((real_n, real_n), dtype=np.float64)
    for f in range(feats.shape[1]):
        col = feats[:, f].reshape(-1, 1)
        sim_matrix += (col == col.T).astype(np.float64)
    sim_matrix /= feats.shape[1]

    avg_sim = (sim_matrix.sum() - np.trace(sim_matrix)) / (real_n * (real_n - 1))
    return float(1.0 - avg_sim)


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
        inject_cloud_score: bool = False,
        ranking_candidates_df: Optional[pd.DataFrame] = None,
        evaluate_pool_diversity: bool = False,
        cloud_teacher_model: Any = None,
        cloud_teacher_unmap_tensors: Optional[Dict[str, torch.Tensor]] = None,
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
        top_k: Top-K candidates to select. Also used to compute gAUC@K, MRR@K, and AUC@K
                            restricted to the top-K items per user for fair comparison across stages.
        logger: Logger instance
        inference_batch_size: Number of candidates to process per batch (default 50K)
        **kwargs: Additional arguments

    Returns:
        Tuple of (StageOutput or None, metrics_dict)
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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

    # Cloud score injection: compute teacher logits on-the-fly per batch
    # (replaces old approach that used preranking scores from candidates_df['score'],
    # which caused a train/eval mismatch since training used actual teacher logits)
    use_teacher_for_cloud_score = inject_cloud_score and cloud_teacher_model is not None
    residual_inject = kwargs.get('residual_inject', False)
    residual_weight = kwargs.get('residual_weight', 1.0)
    use_teacher_for_residual = residual_inject and cloud_teacher_model is not None
    if use_teacher_for_cloud_score:
        logger.info("Cloud score injection enabled: will compute teacher logits per batch")
    elif use_teacher_for_residual:
        cloud_score_scale = kwargs.get('cloud_score_scale', 1.0)
        logger.info(f"Residual inject enabled: residual_weight={residual_weight}, "
                    f"cloud_score_scale={cloud_score_scale}")
    elif inject_cloud_score:
        logger.warning("Cloud score injection requested but no cloud_teacher_model provided. "
                       "cloud_score feature will NOT be injected.")

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

        logger.warning(f"Phase 2: {num_filtered}/{total_candidates} ({100 * num_filtered / total_candidates:.1f}%) "
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

    # Precompute diversity feature matrix: shape (num_valid, num_item_features)
    # Uses item_features_df columns (cate_id, brand, …) for categorical similarity.
    # Kept as float64 so integer IDs remain exact.
    item_feat_diversity_arr: np.ndarray = item_features_lookup.values.astype(np.float64)

    # ========== Phase 3: Prepare user feature metadata ==========
    valid_global_indices = np.where(valid_mask)[0]
    num_valid = len(valid_global_indices)
    request_idx_for_valid = np.searchsorted(request_offsets[1:], valid_global_indices, side='right')
    user_feature_names = list(request_metadata[0][2].keys()) if request_metadata and request_metadata[0][2] else []
    logger.info("Finish preparing user feature metadata for valid candidates.")

    # Cloud score: teacher model will compute logits per chunk in Phase 4.
    # No pre-computation needed here.
    # ========== Phase 4: Chunked model inference (OPTIMIZED) ==========
    # Pre-allocate numpy array for all predictions
    all_scores = np.zeros(num_valid, dtype=np.float32)
    all_student_scores = np.zeros(num_valid, dtype=np.float32)
    all_residual_scores = np.zeros(num_valid, dtype=np.float32)

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
    use_fp16 = kwargs.get('use_fp16', False) and device.type == 'cuda'
    logger.info(f"Using {use_fp16} fp16 precision.")
    import time

    # ========== OPTIMIZATION: Pre-convert ALL features to tensor-ready dtypes ==========
    # This eliminates per-batch dtype detection, None handling, and dtype casting.
    # In the loop, we only do numpy slicing + torch.from_numpy + .to(device).
    t_precompute_start = time.time()

    num_requests = len(request_metadata)

    # --- Step 1: Determine target numpy dtype for each feature ---
    # feature_map tells us sequence/categorical -> int64, numeric -> float32
    feature_target_dtype = {}  # {feat_name: np.int64 or np.float32}
    for feat_name, feat_spec in feature_map.features.items():
        ftype = feat_spec['type']
        if ftype in ('sequence', 'categorical'):
            feature_target_dtype[feat_name] = np.int64
        else:
            feature_target_dtype[feat_name] = np.float32

    # --- Step 2: Pre-convert user features (29K rows, indexed by request) ---
    request_user_features = {}  # {feat_name: np.ndarray with correct dtype}
    request_user_ids = None

    if user_feature_names and num_requests > 0:
        request_user_ids = np.array([request_metadata[req_idx][1] for req_idx in range(num_requests)])

        for feat_name in user_feature_names:
            is_seq = user_feat_is_sequence.get(feat_name, False)
            target_dtype = feature_target_dtype.get(feat_name, np.float32)

            if is_seq:
                user_vals = [request_metadata[req_idx][2].get(feat_name, np.zeros(1))
                             for req_idx in range(num_requests)]
                arr = np.stack(user_vals)
            else:
                user_vals = [request_metadata[req_idx][2].get(feat_name, 0)
                             for req_idx in range(num_requests)]
                arr = np.array(user_vals)

            # Pre-convert to target dtype (handles object arrays, None values, etc.)
            if arr.dtype == np.object_:
                fill_val = 0 if target_dtype == np.int64 else 0.0
                arr = np.array([x if x is not None else fill_val for x in arr.flat]).reshape(arr.shape)
            if arr.dtype != target_dtype:
                try:
                    arr = arr.astype(target_dtype)
                except (ValueError, TypeError):
                    arr = arr.astype(np.int64) if target_dtype == np.int64 else arr.astype(np.float32)

            # Ensure contiguous for torch.from_numpy
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            t = torch.from_numpy(arr)
            if device.type == 'cuda':
                t = t.pin_memory()
            request_user_features[feat_name] = t

    # --- Step 3: Pre-convert item features (29M rows, from DataFrame to numpy arrays) ---
    # Convert DataFrame columns to pre-typed numpy arrays ONCE
    item_feature_arrays = {}  # {col: np.ndarray with correct dtype}
    item_feature_names_in_model = []  # track which columns are in feature_map

    for col in item_feature_cols:
        if col not in feature_map.features:
            continue
        item_feature_names_in_model.append(col)
        target_dtype = feature_target_dtype.get(col, np.float32)
        col_type, _ = col_type_info[col]
        col_values = item_features_lookup[col].values

        if col_type == 'sequence':
            try:
                arr = np.stack(col_values)
            except ValueError:
                arr = np.array([np.array(x) for x in col_values])
        elif col_type == 'scalar':
            # 'scalar' = object array that's not a sequence.
            # Old code: col_values.astype(float32) first, then tensor conv to target dtype.
            # Must preserve this float32 intermediate step for numerical equivalence.
            if not isinstance(col_values, np.ndarray):
                col_values = np.array(col_values)
            if col_values.dtype == np.object_:
                col_values = np.array([x if x is not None else 0.0 for x in col_values.flat]).reshape(col_values.shape)
            try:
                arr = col_values.astype(np.float32)  # float32 first (matching old behavior)
            except (ValueError, TypeError):
                try:
                    arr = col_values.astype(np.int64)
                except (ValueError, TypeError):
                    arr = np.array([int(x) if x is not None else 0 for x in col_values], dtype=np.int64)
        else:  # 'direct'
            arr = col_values

        # Final conversion to target dtype
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        if arr.dtype == np.object_:
            fill_val = 0.0
            arr = np.array([x if x is not None else fill_val for x in arr.flat]).reshape(arr.shape)
        if arr.dtype != target_dtype:
            try:
                arr = arr.astype(target_dtype)
            except (ValueError, TypeError):
                arr = arr.astype(np.int64) if target_dtype == np.int64 else arr.astype(np.float32)

        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        t = torch.from_numpy(arr)
        if device.type == 'cuda':
            t = t.pin_memory()
        item_feature_arrays[col] = t

    # Pre-convert item_id if needed
    if item_id_col not in item_feature_arrays and item_id_col in feature_map.features:
        target_dtype = feature_target_dtype.get(item_id_col, np.int64)
        item_id_arr = valid_item_ids.astype(target_dtype) if valid_item_ids.dtype != target_dtype else valid_item_ids
        if not item_id_arr.flags['C_CONTIGUOUS']:
            item_id_arr = np.ascontiguousarray(item_id_arr)
        t = torch.from_numpy(item_id_arr)
        if device.type == 'cuda':
            t = t.pin_memory()
        item_feature_arrays[item_id_col] = t
        item_feature_names_in_model.append(item_id_col)

    # Pre-convert user_id
    if request_user_ids is not None and 'user_id' in feature_map.features:
        target_dtype = feature_target_dtype.get('user_id', np.int64)
        if request_user_ids.dtype != target_dtype:
            try:
                request_user_ids = request_user_ids.astype(target_dtype)
            except (ValueError, TypeError):
                pass
        if not request_user_ids.flags['C_CONTIGUOUS']:
            request_user_ids = np.ascontiguousarray(request_user_ids)
        t_user_ids = torch.from_numpy(request_user_ids)
        if device.type == 'cuda':
            t_user_ids = t_user_ids.pin_memory()
        request_user_ids = t_user_ids

    # Determine which user features are in the model's feature_map
    user_feature_names_in_model = [f for f in user_feature_names if f in feature_map.features]

    precompute_time = time.time() - t_precompute_start
    logger.info(f"Pre-converted {len(request_user_features)} user + {len(item_feature_arrays)} item feature arrays "
                f"to tensor-ready dtypes in {precompute_time:.2f}s")
                
    # Fine-grained timing for bottleneck analysis
    timing_stats = {
        'user_feature_prep': 0.0,
        'item_feature_extract': 0.0,
        'tensor_conversion': 0.0,
        'model_forward': 0.0,
    }

    # Pre-move user features to device, since they only span ~num_requests rows (very small)
    request_user_features_device = {}
    for feat_name in user_feature_names_in_model:
        request_user_features_device[feat_name] = request_user_features[feat_name].to(device)

    request_user_ids_device = None
    if request_user_ids is not None and 'user_id' in feature_map.features:
        request_user_ids_device = request_user_ids.to(device)

    # Pre-convert request_idx_for_valid to tensor for indexing user features
    request_idx_for_valid_tensor = torch.from_numpy(request_idx_for_valid).long()

    # Evaluation caching for teacher logits
    eval_teacher_logits = None
    cache_filled = False
    cache_key = None
    if (use_teacher_for_cloud_score or use_teacher_for_residual) and cloud_teacher_model is not None:
        stage_name = kwargs.get('stage_name', 'unknown_stage')
        cache_key = f"{stage_name}_eval_logits"
        if not hasattr(cloud_teacher_model, '_eval_logits_cache'):
            cloud_teacher_model._eval_logits_cache = {}
            
        if cache_key in cloud_teacher_model._eval_logits_cache and len(cloud_teacher_model._eval_logits_cache[cache_key]) == num_valid:
            eval_teacher_logits = cloud_teacher_model._eval_logits_cache[cache_key]
            cache_filled = True
            logger.info(f"Fast Evaluation: Loaded {num_valid} teacher logits from cache for {stage_name}")
        else:
            eval_teacher_logits = torch.zeros(num_valid, dtype=torch.float32, device='cpu')

    inference_all_start = time.time()
    num_batches = 0

    for chunk_start in range(0, num_valid, inference_batch_size):
        chunk_end = min(chunk_start + inference_batch_size, num_valid)
        num_batches += 1

        # ===== TIMING: User feature preparation (numpy advanced indexing) =====
        t_user_start = time.time()
        chunk_req_indices_cpu = request_idx_for_valid_tensor[chunk_start:chunk_end]
        timing_stats['user_feature_prep'] += time.time() - t_user_start

        # ===== TIMING: Tensor conversion (NOW JUST slice + from_numpy + to_device) =====
        t_tensor_start = time.time()
        tensor_batch = {}

        chunk_req_indices_device = chunk_req_indices_cpu.to(device, non_blocking=True)

        # User features: GPU advanced indexing (blazing fast, ~TB/s)
        for feat_name in user_feature_names_in_model:
            tensor_batch[feat_name] = request_user_features_device[feat_name][chunk_req_indices_device]

        # User ID
        if request_user_ids_device is not None:
            tensor_batch['user_id'] = request_user_ids_device[chunk_req_indices_device]

        # Item features: tensor slicing (zero copy view)
        for col in item_feature_names_in_model:
            chunk_tensor = item_feature_arrays[col][chunk_start:chunk_end]
            tensor_batch[col] = chunk_tensor.to(device, non_blocking=True)

        # Cloud score injection: run teacher model per chunk or load from cache
        teacher_logits_chunk = None
        if use_teacher_for_cloud_score or use_teacher_for_residual:
            if cache_filled:
                teacher_out_logit = eval_teacher_logits[chunk_start:chunk_end].to(device, non_blocking=True)
            else:
                # Build teacher batch: unmap compact IDs back to original IDs
                teacher_batch = dict(tensor_batch)
                if cloud_teacher_unmap_tensors:
                    for feat, unmap_tensor in cloud_teacher_unmap_tensors.items():
                        if feat in teacher_batch and teacher_batch[feat] is not None:
                            feat_tensor = teacher_batch[feat]
                            if isinstance(feat_tensor, torch.Tensor):
                                max_valid_idx = unmap_tensor.size(0) - 1
                                safe_indices = torch.clamp(feat_tensor.long(), min=0, max=max_valid_idx)
                                teacher_batch[feat] = unmap_tensor[safe_indices].to(feat_tensor.dtype)

                with torch.no_grad():
                    teacher_out = cloud_teacher_model.forward(teacher_batch)
                    teacher_out_logit = teacher_out['logit'].detach().squeeze(-1)
                
                eval_teacher_logits[chunk_start:chunk_end] = teacher_out_logit.cpu()

            if use_teacher_for_cloud_score:
                # Inject mode: add teacher logits as cloud_score feature
                cloud_feature_scale = kwargs.get('cloud_feature_scale', kwargs.get('cloud_score_scale', 1.0))
                if cloud_feature_scale != 1.0:
                    tensor_batch['cloud_score'] = teacher_out_logit / cloud_feature_scale
                    if num_batches == 1 and not cache_filled:
                        logger.info(f"[Inject Mode] Applied cloud_feature_scale={cloud_feature_scale}. "
                                    f"First batch cloud_score mean: {tensor_batch['cloud_score'].mean().item():.4f}")
                else:
                    tensor_batch['cloud_score'] = teacher_out_logit
            
            if use_teacher_for_residual:
                # Residual inject or Hybrid inject: save teacher logits for post-forward addition
                teacher_logits_chunk = teacher_out_logit

        timing_stats['tensor_conversion'] += time.time() - t_tensor_start

        # ===== TIMING: Model forward pass =====
        t_forward_start = time.time()
        with torch.no_grad():
            if use_fp16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_dict = model(tensor_batch)
            else:
                pred_dict = model(tensor_batch)

        # ===== TIMING: Result transfer to CPU =====

        if 'logit' in pred_dict:
            chunk_scores = pred_dict['logit'].detach().cpu().numpy().flatten()
        else:
            chunk_scores = pred_dict['y_pred'].detach().cpu().numpy().flatten()

        all_student_scores[chunk_start:chunk_end] = chunk_scores
        
        # Residual inject: add scaled teacher logits to student scores
        if use_teacher_for_residual and teacher_logits_chunk is not None:
            cloud_residual_scale = kwargs.get('cloud_residual_scale', kwargs.get('cloud_score_scale', 1.0))
            teacher_scores = teacher_logits_chunk.cpu().numpy().flatten()
            residual_chunk = residual_weight * (teacher_scores / cloud_residual_scale)
            chunk_scores = chunk_scores + residual_chunk
            all_residual_scores[chunk_start:chunk_end] = residual_chunk
            if num_batches == 1:
                logger.info(f"[Residual Inject] First batch: student_mean={all_student_scores[chunk_start:chunk_end].mean():.4f}, "
                            f"teacher_mean={teacher_scores.mean():.4f}, "
                            f"residual_mean={residual_chunk.mean():.4f}")
        else:
            all_residual_scores[chunk_start:chunk_end] = 0.0

        timing_stats['model_forward'] += time.time() - t_forward_start

        all_scores[chunk_start:chunk_end] = chunk_scores

        # Clean up to free memory
        del tensor_batch
        del pred_dict
        del chunk_scores

    total_inference_time = time.time() - inference_all_start

    # Log detailed timing breakdown
    logger.info(f"===== Inference Timing Breakdown ({num_batches} batches, batch_size={inference_batch_size}) =====")
    logger.info(f"  Pre-conversion:       {precompute_time:8.2f}s (one-time, outside loop)")
    logger.info(
        f"  User feature prep:    {timing_stats['user_feature_prep']:8.2f}s ({100 * timing_stats['user_feature_prep'] / total_inference_time:5.1f}%)")
    logger.info(
        f"  Tensor conversion:    {timing_stats['tensor_conversion']:8.2f}s ({100 * timing_stats['tensor_conversion'] / total_inference_time:5.1f}%)")
    logger.info(
        f"  Model forward:        {timing_stats['model_forward']:8.2f}s ({100 * timing_stats['model_forward'] / total_inference_time:5.1f}%)")
    logger.info(f"  Loop total:           {total_inference_time:8.2f}s")
    logger.info(f"  Overall total:        {precompute_time + total_inference_time:8.2f}s")
    logger.info("Finish model inference for all valid candidates.")

    if eval_teacher_logits is not None and not cache_filled:
        cloud_teacher_model._eval_logits_cache[cache_key] = eval_teacher_logits
        logger.info(f"Saved {num_valid} teacher logits to cache for {stage_name}")
    # ========== Phase 5: Scatter results back to requests (OPTIMIZED - DataFrame output) ==========
    # Create mapping from valid indices back to original global indices
    global_to_valid_idx = np.full(total_candidates, -1, dtype=np.int64)
    global_to_valid_idx[valid_global_indices] = np.arange(num_valid)
    num_queries = 0
    valid_queries = 0
    valid_queries_at_k = 0
    if compute_metrics:
        total_metrics = {f'{m}@{k}': 0.0 for k in metrics_k for m in ['Recall', 'nDCG', 'Diversity']}
        total_metrics['MRR'] = 0.0
        total_metrics['gAUC'] = 0.0
        total_metrics[f'gAUC@{top_k}'] = 0.0
        total_metrics[f'MRR@{top_k}'] = 0.0
        if evaluate_pool_diversity:
            total_metrics['Candidate_Diversity'] = 0.0
        # Accumulators for global (non-grouped) AUC
        all_query_scores_list = []  # list of per-query score arrays
        all_query_labels_list = []  # list of per-query label arrays
        # Accumulators for global AUC restricted to top-K items per user
        all_topk_scores_list = []
        all_topk_labels_list = []

    # Build per-request ranking candidate sets if ranking_candidates_df is provided
    # This is used to restrict Recall@K/nDCG@K to a preranking-filtered subset,
    # while still computing AUC/gAUC on the full pool (fair cross-stage comparison).
    ranking_item_sets = None
    if ranking_candidates_df is not None and compute_metrics:
        # Build a dict: request_id -> set of item_ids in the filtered subset
        ranking_item_sets = (
            ranking_candidates_df.groupby('request_id')['item_id']
            .apply(set)
            .to_dict()
        )
        logger.info(f"Using ranking_candidates_df with {len(ranking_item_sets)} requests "
                    f"for Recall@K/nDCG@K ({len(ranking_candidates_df)} filtered candidates)")

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
            # Pre-sort by descending score with RANDOM tie-breaking:
            # 1. Randomly permute indices so tied scores get random ordering
            # 2. Stable sort by descending score preserves the random order within ties
            random_perm = np.random.permutation(len(req_scores))
            sorted_order = random_perm[np.argsort(-req_scores[random_perm], kind='stable')]
            sorted_scores = req_scores[sorted_order]
            sorted_labels = req_labels[sorted_order]
            sorted_item_ids = req_item_ids[sorted_order]

            # ------------------------------------------------------------------
            # If ranking_candidates_df is provided, use it to restrict the
            # Recall@K / nDCG@K sorted list to the preranking-filtered items.
            # AUC / gAUC still use the full sorted_scores / sorted_labels.
            # ------------------------------------------------------------------
            if ranking_item_sets is not None and req_id in ranking_item_sets:
                # Mask: which positions in the full sorted list are in the filtered set
                rank_mask = np.isin(sorted_item_ids, list(ranking_item_sets[req_id]))
                recall_sorted_scores = sorted_scores[rank_mask]
                recall_sorted_labels = sorted_labels[rank_mask]
            else:
                recall_sorted_scores = sorted_scores
                recall_sorted_labels = sorted_labels

            # Accumulate raw scores/labels for global AUC (always full pool)
            all_query_scores_list.append(req_scores)
            all_query_labels_list.append(req_labels)
            # Accumulate top-K scores/labels for AUC@K (from full sorted list)
            k = min(top_k, len(sorted_scores))
            topk_labels_subset = sorted_labels[:k]
            if np.sum(topk_labels_subset) > 0:
                valid_queries_at_k += 1
                all_topk_scores_list.append(sorted_scores[:k])
                all_topk_labels_list.append(topk_labels_subset)

            # Extract item features for this query (for Diversity@K computation)
            # item_features_lookup rows are indexed same as item_feat_diversity_arr;
            # req_valid_idx maps query candidates into that global valid array.
            req_item_feat_diversity = item_feat_diversity_arr[req_valid_idx]  # (num_cands, num_feats)

            # When ranking_item_sets restricts Recall/nDCG to a filtered subset,
            # pass the full-pool sorted arrays for gAUC/MRR computation.
            fp_scores_arg = sorted_scores if ranking_item_sets is not None else None
            fp_labels_arg = sorted_labels if ranking_item_sets is not None else None
            query_metrics = compute_ranking_metrics(
                recall_sorted_scores, recall_sorted_labels, metrics_k,
                pre_sorted=True, top_k_for_metrics=top_k,
                item_features_arr=req_item_feat_diversity,
                sorted_order=sorted_order,
                full_pool_scores=fp_scores_arg,
                full_pool_labels=fp_labels_arg,
            )
            # Detailed debugging for first 3 queries
            if num_queries < 3:
                pos_scores = req_scores[req_labels == 1]
                neg_scores = req_scores[req_labels == 0]
                
                req_student = all_student_scores[req_valid_idx]
                req_residual = all_residual_scores[req_valid_idx]
                pos_student = req_student[req_labels == 1]
                pos_residual = req_residual[req_labels == 1]
                
                # Order in the sorted list
                pos_rank_index = np.where(sorted_labels == 1)[0][0] + 1
                logger.info(f"[DEBUG] Query {req_id}: {num_candidates} cands ({num_positive} pos, {num_negative} neg)")
                logger.info(f"[DEBUG]   Positive score: {pos_scores[0]:.6f} (Student logit: {pos_student[0]:.6f}, Residual: {pos_residual[0]:.6f})")
                logger.info(
                    f"[DEBUG]   Negative scores: min={neg_scores.min():.6f}, max={neg_scores.max():.6f}, mean={neg_scores.mean():.6f}")
                logger.info(f"[DEBUG]   Positive rank (In sorted list): {pos_rank_index} (1=best)")
                logger.info(
                    f"[DEBUG]   # negatives >= pos: {np.sum(neg_scores >= pos_scores[0])}, # negatives < pos: {np.sum(neg_scores < pos_scores[0])}")
            # Log individual query metrics for first 3 queries
            if num_queries < 3 and query_metrics:
                logger.info(
                    f"[DEBUG]   Query metrics: MRR={query_metrics.get('MRR', 0):.4f}, gAUC={query_metrics.get('gAUC', 0):.4f}")

            if query_metrics:
                for metric_name, value in query_metrics.items():
                    total_metrics[metric_name] += value
            
            # Compute Candidate_Diversity (diversity of ALL candidates for this query, regardless of K)
            if evaluate_pool_diversity and req_item_feat_diversity is not None and len(req_item_feat_diversity) > 1:
                feat_len = len(req_item_feat_diversity)
                sim_matrix = np.zeros((feat_len, feat_len), dtype=np.float64)
                for f in range(req_item_feat_diversity.shape[1]):
                    col = req_item_feat_diversity[:, f].reshape(-1, 1)
                    sim_matrix += (col == col.T).astype(np.float64)
                sim_matrix /= req_item_feat_diversity.shape[1]
                avg_sim = (sim_matrix.sum() - np.trace(sim_matrix)) / (feat_len * (feat_len - 1))
                total_metrics['Candidate_Diversity'] += float(1.0 - avg_sim)

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
            gauc_metrics = {'gAUC', 'MRR'}
            gauc_metrics_at_k = {f'gAUC@{top_k}', f'MRR@{top_k}'}
            logger.info(f"Valid queries with positive labels: {valid_queries}/{num_queries} ({100 * valid_queries / num_queries:.1f}%)")
            for metric_name in total_metrics:
                if metric_name in gauc_metrics:
                    # gAUC / MRR averaged only over queries that have at least one positive
                    metrics[metric_name] = total_metrics[metric_name] / valid_queries if valid_queries > 0 else 0.0
                elif metric_name in gauc_metrics_at_k:
                    metrics[metric_name] = total_metrics[metric_name] / valid_queries_at_k if valid_queries_at_k > 0 else 0.0
                elif metric_name == 'Candidate_Diversity':
                    metrics[metric_name] = total_metrics[metric_name] / valid_queries if valid_queries > 0 else 0.0
                else:
                    metrics[metric_name] = total_metrics[metric_name] / num_queries

            # ===== Global (non-grouped) AUC over all scores =====
            if all_query_scores_list:
                try:
                    global_scores = np.concatenate(all_query_scores_list)
                    global_labels = np.concatenate(all_query_labels_list)
                    if global_labels.sum() > 0 and global_labels.sum() < len(global_labels):
                        auc = binary_auroc(
                            torch.as_tensor(global_scores, dtype=torch.float32),
                            torch.as_tensor(global_labels, dtype=torch.int64)
                        )
                        metrics['AUC'] = float(auc.item())
                    else:
                        metrics['AUC'] = 0.0
                except Exception as e:
                    logger.warning(f"Failed to compute global AUC: {e}")
            # ===== Global AUC@K restricted to top-K items per user =====
            if all_topk_scores_list:
                try:
                    topk_scores = np.concatenate(all_topk_scores_list)
                    topk_labels = np.concatenate(all_topk_labels_list)
                    if 0 < topk_labels.sum() < len(topk_labels):
                        auc_at_k = binary_auroc(
                            torch.as_tensor(topk_scores, dtype=torch.float32),
                            torch.as_tensor(topk_labels, dtype=torch.int64)
                        )
                        metrics[f'AUC@{top_k}'] = float(auc_at_k.item())
                    else:
                        metrics[f'AUC@{top_k}'] = 0.0
                except Exception as e:
                    logger.warning(f"Failed to compute global AUC@{top_k}: {e}")
        else:
            logger.warning("No valid queries with positive labels for ranking evaluation.")
        
        # Calculate Pool Diversity
        if evaluate_pool_diversity and item_feat_diversity_arr is not None and len(item_feat_diversity_arr) > 0:
            try:
                metrics['Pool_Diversity'] = compute_pool_diversity(item_feat_diversity_arr)
            except Exception as e:
                logger.warning(f"Failed to compute Pool_Diversity: {e}")
                metrics['Pool_Diversity'] = 0.0

    return output, metrics
