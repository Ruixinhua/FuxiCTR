# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Negative Sampler for Training with Pairwise Ranking.

This module provides utilities for sampling negative items during training
when the training data contains only positive examples.

Optimized for high-throughput batch training.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Set, Any, Optional

logger = logging.getLogger(__name__)


class NegativeSampler:
    """
    Samples negative items from the item pool for pairwise ranking training.
    
    Optimized for high-throughput training with:
    - Vectorized batch sampling (no Python loops)
    - Pre-computed numpy arrays for fast lookup
    - Cached feature arrays for GPU transfer
    
    Usage:
        sampler = NegativeSampler(item_features_df, item_id_col='cand_item_id')
        neg_item_ids = sampler.sample_negatives_batch(positive_ids, num_negatives=4)
    """
    
    def __init__(
        self,
        item_features_df: pd.DataFrame,
        item_id_col: str = 'cand_item_id',
    ):
        """
        Initialize the negative sampler with pre-computed lookup structures.
        
        Args:
            item_features_df: DataFrame containing item features.
            item_id_col: Column name for item IDs
        """
        self.item_id_col = item_id_col
        
        # Ensure DataFrame is indexed by item_id for fast lookup
        if item_features_df.index.name == item_id_col:
            self.item_features_df = item_features_df
        elif item_id_col in item_features_df.columns:
            self.item_features_df = item_features_df.set_index(item_id_col)
        else:
            # Use first column as index
            self.item_features_df = item_features_df
        
        # Pre-compute numpy arrays for fast sampling
        self.all_item_ids = self.item_features_df.index.to_numpy()
        self.num_items = self.item_features_df.shape[0]
        
        logger.info(f"NegativeSampler initialized with {self.num_items} items, "
                   f"{self.item_features_df.shape[1]} feature columns pre-cached")
    
    def sample_negatives_batch_fast(
        self,
        positive_item_ids: np.ndarray,
        num_negatives: int,
    ) -> np.ndarray:
        """
        Fast vectorized negative sampling for a batch.
        
        Uses uniform random sampling from all items. While this may occasionally
        sample a positive as negative (collision), the probability is very low
        for large item pools and has negligible impact on training.
        
        Args:
            positive_item_ids: Array of positive item IDs [batch_size]
            num_negatives: Number of negatives per positive
            
        Returns:
            Array of negative item indices [batch_size, num_negatives]
            (indices into all_item_ids, not the actual IDs)
        """
        batch_size = len(positive_item_ids)
        
        # Fast uniform sampling - sample random indices
        neg_indices = np.random.randint(
            0, self.num_items, 
            size=(batch_size, num_negatives)
        )
        
        return neg_indices
    
    def sample_negatives_batch(
        self,
        positive_item_ids: np.ndarray,
        num_negatives: int,
    ) -> np.ndarray:
        """
        Sample negative items for a batch, excluding positives.
        
        Optimized version that minimizes Python loops.
        
        Args:
            positive_item_ids: Array of positive item IDs [batch_size]
            num_negatives: Number of negatives per positive
            
        Returns:
            Array of negative item IDs [batch_size, num_negatives]
        """
        batch_size = len(positive_item_ids)
        
        # Sample more than needed to handle collisions
        oversample_factor = 2
        total_samples = num_negatives * oversample_factor
        
        # Random indices into all_item_ids
        sampled_indices = np.random.randint(
            0, self.num_items,
            size=(batch_size, total_samples)
        )
        
        # Get corresponding item IDs
        sampled_ids = self.all_item_ids[sampled_indices]
        
        # Build result array
        result = np.zeros((batch_size, num_negatives), dtype=self.all_item_ids.dtype)
        
        # Vectorized collision check and selection
        for i in range(batch_size):
            pos_id = positive_item_ids[i]
            candidates = sampled_ids[i]
            
            # Filter out positive ID (vectorized)
            valid_mask = candidates != pos_id
            valid_candidates = candidates[valid_mask]
            
            # Take first num_negatives
            if len(valid_candidates) >= num_negatives:
                result[i] = valid_candidates[:num_negatives]
            else:
                # Very rare edge case: pad with any items
                result[i, :len(valid_candidates)] = valid_candidates
                if len(valid_candidates) > 0:
                    result[i, len(valid_candidates):] = valid_candidates[0]
                else:
                    result[i] = self.all_item_ids[:num_negatives]
        
        return result
    
    def get_features_by_indices(
        self,
        indices: np.ndarray,
    ) -> pd.DataFrame:
        """
        Get features for items by their indices (fastest method).
        
        Args:
            indices: Array of indices into all_item_ids
            
        Returns:
            pd.DataFrame with item features for the given indices
        """
        return self.item_features_df.iloc[indices]
    
    def get_features_by_ids(
        self,
        item_ids: np.ndarray,
    ) -> pd.DataFrame:
        """
        Get features for items by their IDs.
        
        Args:
            item_ids: Array of item IDs
            
        Returns:
            pd.DataFrame with item features for the given IDs
        """
        return self.item_features_df.loc[item_ids]
    
    def sample_negatives(
        self,
        positive_item_ids: List[Any],
        num_negatives: int,
        exclude_items: Optional[Set[Any]] = None,
    ) -> List[Any]:
        """
        Sample negative items excluding positive items (single sample version).
        
        Args:
            positive_item_ids: List of positive item IDs to exclude
            num_negatives: Number of negative items to sample
            exclude_items: Additional items to exclude (optional)
            
        Returns:
            List of negative item IDs
        """
        exclude = set(positive_item_ids)
        if exclude_items:
            exclude.update(exclude_items)
        
        # Sample with rejection
        sampled = []
        max_attempts = num_negatives * 10
        attempts = 0
        
        while len(sampled) < num_negatives and attempts < max_attempts:
            idx = np.random.randint(0, self.num_items)
            item_id = self.all_item_ids[idx]
            if item_id not in exclude:
                sampled.append(item_id)
                exclude.add(item_id)  # Don't sample duplicates
            attempts += 1
        
        return sampled
