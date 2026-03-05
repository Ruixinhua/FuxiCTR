# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
Vocabulary Pruner

Scans training data to identify actually-used feature values and builds
compact index remappings. This allows models to use much smaller embedding
tables without reprocessing the dataset.

Usage:
    prune_info = compute_vocab_pruning(feature_map, train_data_path)
    # prune_info is attached to feature_map._vocab_prune_info
    # Then in model building, apply_vocab_pruning_to_model() replaces
    # nn.Embedding with RemappedEmbedding for prunable features.
"""

import logging
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class FeaturePruneInfo:
    """Pruning info for a single feature."""
    feature_name: str
    original_vocab_size: int
    compact_vocab_size: int  # includes padding at index 0
    # Tensor of shape [original_vocab_size]: old_idx -> new_compact_idx
    remap_table: torch.IntTensor = None
    reduction_ratio: float = 0.0


@dataclass
class VocabPruneInfo:
    """Container for all pruning decisions."""
    features: Dict[str, FeaturePruneInfo] = field(default_factory=dict)
    # Map: base_feature_name -> set of features sharing its embedding
    shared_groups: Dict[str, Set[str]] = field(default_factory=dict)


def scan_parquet_vocab(data_paths: Union[str, List[str]], feature_map) -> Dict[str, set]:
    """
    Scan one or more parquet files to collect actually-used unique values per feature.

    Scans all provided data splits (e.g. train + valid + test) and unions unique
    values so that no feature value appearing in any split is mapped to padding.

    Handles:
    - Categorical features: collect unique integer values from the column
    - Sequence features: flatten array-valued columns and collect unique values
    - Shared embeddings: union values across all features sharing the same embedding

    Args:
        data_paths: Path(s) to data parquet files. Can be a single string or
                    a list of paths (e.g. [train_path, valid_path, test_path]).
        feature_map: FuxiCTR FeatureMap instance

    Returns:
        Dict mapping feature_name -> set of unique integer values
    """
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    # Filter to paths that actually exist
    valid_paths = [p for p in data_paths if os.path.exists(p)]
    if not valid_paths:
        logger.warning(f"[VocabPruner] No valid data paths found: {data_paths}")
        return {}

    logger.info(f"[VocabPruner] Scanning {len(valid_paths)} file(s) for actual vocabulary usage...")
    for p in valid_paths:
        logger.info(f"[VocabPruner]   - {p}")

    # Identify which columns to scan (only categorical/sequence with vocab_size)
    scan_features = {}
    for name, spec in feature_map.features.items():
        if spec.get("type") in ("categorical", "sequence") and "vocab_size" in spec:
            scan_features[name] = spec

    if not scan_features:
        logger.info("[VocabPruner] No categorical/sequence features to scan.")
        return {}

    # Build shared embedding groups: base_name -> [features using that embedding]
    shared_groups = defaultdict(set)
    for name, spec in scan_features.items():
        base = spec.get("share_embedding", name)
        shared_groups[base].add(name)

    all_cols_needed = list(scan_features.keys())
    unique_values = defaultdict(set)

    # Scan each file and union the unique values
    for data_path in valid_paths:
        logger.info(f"[VocabPruner] Reading {os.path.basename(data_path)}...")
        df = pd.read_parquet(data_path)
        existing_cols = [c for c in all_cols_needed if c in df.columns]

        for col in existing_cols:
            series = df[col]
            if series.dtype == "object":
                # Sequence feature stored as lists/arrays — flatten all values
                try:
                    all_vals = np.concatenate(series.to_list())
                    unique_values[col] |= set(int(v) for v in np.unique(all_vals) if v != 0)
                except Exception as e:
                    logger.warning(f"[VocabPruner] Error scanning sequence column '{col}': {e}")
            else:
                # Categorical feature: simple unique
                vals = series.to_numpy()
                unique_values[col] |= set(int(v) for v in np.unique(vals) if v != 0)

        del df

    # Log per-feature counts
    for col in sorted(unique_values.keys()):
        logger.info(f"[VocabPruner]   {col}: {len(unique_values[col])} unique values (across all splits)")

    # Union shared embedding groups
    for base_name, group_members in shared_groups.items():
        if base_name not in scan_features:
            continue
        union_vals = set()
        for member in group_members:
            if member in unique_values:
                union_vals |= unique_values[member]
        # Assign the union to the base feature and all members
        for member in group_members:
            unique_values[member] = union_vals
        unique_values[base_name] = union_vals

    # Log final summary
    for name in sorted(unique_values.keys()):
        orig = scan_features[name].get("vocab_size", 0)
        actual = len(unique_values[name])
        logger.info(f"[VocabPruner] {name}: {orig} vocab → {actual} actual unique ({actual/max(orig,1)*100:.1f}%)")

    return dict(unique_values)


def build_vocab_mapping(
    actual_vocab: Dict[str, set],
    feature_map,
    min_vocab_size: int = 100,
    min_reduction_ratio: float = 0.3,
) -> VocabPruneInfo:
    """
    Build compact index remapping for features that benefit from pruning.

    Args:
        actual_vocab: Dict from scan_parquet_vocab()
        feature_map: FuxiCTR FeatureMap instance
        min_vocab_size: Only prune features with vocab_size > this threshold
        min_reduction_ratio: Only prune if reduction > this ratio (0.3 = 30% fewer values)

    Returns:
        VocabPruneInfo with remapping tables
    """
    prune_info = VocabPruneInfo()

    # Build shared embedding groups
    shared_groups = defaultdict(set)
    for name, spec in feature_map.features.items():
        if spec.get("type") in ("categorical", "sequence") and "vocab_size" in spec:
            base = spec.get("share_embedding", name)
            shared_groups[base].add(name)
    prune_info.shared_groups = dict(shared_groups)

    # Track which base embeddings we've already processed
    processed_bases = set()

    for name, spec in feature_map.features.items():
        if spec.get("type") not in ("categorical", "sequence"):
            continue
        if "vocab_size" not in spec:
            continue

        # Determine the base embedding feature
        base_name = spec.get("share_embedding", name)

        # Skip if already processed (shared embedding)
        if base_name in processed_bases:
            # Still record the prune info reference for this shared feature
            if base_name in prune_info.features:
                prune_info.features[name] = prune_info.features[base_name]
            continue

        orig_vocab_size = spec["vocab_size"]

        # Skip small vocabularies
        if orig_vocab_size <= min_vocab_size:
            continue

        # Get actual unique values (use base_name's values)
        used_values = actual_vocab.get(base_name, actual_vocab.get(name, None))
        if used_values is None:
            continue

        actual_count = len(used_values)
        reduction = 1.0 - (actual_count / orig_vocab_size)

        # Skip if reduction is too small
        if reduction < min_reduction_ratio:
            logger.info(
                f"[VocabPruner] Skipping {name}: reduction {reduction:.1%} < threshold {min_reduction_ratio:.1%}"
            )
            continue

        # Build remap table: old_idx -> new_compact_idx
        # Index 0 is reserved for padding (maps to 0)
        # Unknown indices (not in used_values) also map to 0
        compact_vocab_size = actual_count + 1  # +1 for padding at index 0

        remap = torch.zeros(orig_vocab_size, dtype=torch.int32)
        sorted_vals = sorted(used_values)
        for new_idx, old_idx in enumerate(sorted_vals, start=1):
            if old_idx < orig_vocab_size:
                remap[old_idx] = new_idx

        info = FeaturePruneInfo(
            feature_name=base_name,
            original_vocab_size=orig_vocab_size,
            compact_vocab_size=compact_vocab_size,
            remap_table=remap,
            reduction_ratio=reduction,
        )
        prune_info.features[base_name] = info
        processed_bases.add(base_name)

        # Also record for the current feature name if different
        if name != base_name:
            prune_info.features[name] = info

        logger.info(
            f"[VocabPruner] {base_name}: {orig_vocab_size} → {compact_vocab_size} "
            f"({reduction:.1%} reduction, saves ~{(orig_vocab_size - compact_vocab_size) * 4 / 1e6:.1f}MB per emb_dim)"
        )

    return prune_info


def prune_feature_map_vocab(feature_map, prune_info: VocabPruneInfo):
    """
    Update vocab_size in feature_map to compact sizes.

    Modifies feature_map in-place. Updates total_features accordingly.

    Args:
        feature_map: FuxiCTR FeatureMap instance
        prune_info: VocabPruneInfo from build_vocab_mapping()
    """
    total_saved = 0
    for name, spec in feature_map.features.items():
        if name in prune_info.features:
            info = prune_info.features[name]
            old_size = spec["vocab_size"]
            spec["vocab_size"] = info.compact_vocab_size
            total_saved += old_size - info.compact_vocab_size

    feature_map.total_features -= total_saved
    logger.info(
        f"[VocabPruner] Updated feature_map: total_features reduced by {total_saved:,} "
        f"(new total: {feature_map.total_features:,})"
    )


def compute_vocab_pruning(
    feature_map,
    data_paths: Union[str, List[str]],
    min_vocab_size: int = 100,
    min_reduction_ratio: float = 0.3,
    cache_dir: Optional[str] = None,
) -> VocabPruneInfo:
    """
    End-to-end vocabulary pruning: scan data, build mapping, update feature_map.

    Scans all provided data paths (train + valid + test) to ensure no feature
    value appearing in any split is lost to padding during evaluation.
    Results are cached to avoid re-scanning on subsequent runs.

    Args:
        feature_map: FuxiCTR FeatureMap instance (modified in-place)
        data_paths: Path(s) to parquet files to scan (typically
                    [train_positive, valid, test] or [train, valid, test])
        min_vocab_size: Only prune features with vocab_size > this
        min_reduction_ratio: Only prune if reduction > this ratio
        cache_dir: Directory to cache prune info (default: feature_map.data_dir)

    Returns:
        VocabPruneInfo with remapping tables
    """
    if cache_dir is None:
        cache_dir = getattr(feature_map, "data_dir", None)

    cache_path = os.path.join(cache_dir, "vocab_prune_cache.pkl") if cache_dir else None

    # Try loading from cache
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                prune_info = pickle.load(f)
            logger.info(f"[VocabPruner] Loaded cached prune info from {cache_path} "
                        f"({len(prune_info.features)} features)")
            # Apply to feature_map
            prune_feature_map_vocab(feature_map, prune_info)
            return prune_info
        except Exception as e:
            logger.warning(f"[VocabPruner] Cache load failed: {e}, re-scanning...")

    # Scan all data splits and build
    actual_vocab = scan_parquet_vocab(data_paths, feature_map)
    prune_info = build_vocab_mapping(
        actual_vocab, feature_map,
        min_vocab_size=min_vocab_size,
        min_reduction_ratio=min_reduction_ratio,
    )

    if not prune_info.features:
        logger.info("[VocabPruner] No features to prune.")
        return prune_info

    # Apply to feature_map
    prune_feature_map_vocab(feature_map, prune_info)

    # Cache
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(prune_info, f)
            logger.info(f"[VocabPruner] Saved prune cache to {cache_path}")
        except Exception as e:
            logger.warning(f"[VocabPruner] Cache save failed: {e}")

    return prune_info
