# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Negative Sampler for Training with Pairwise Ranking.

This module provides utilities for sampling negative items during training
when the training data contains only positive examples.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Set, Any, Dict, Optional

logger = logging.getLogger(__name__)



class NegativeSampler:
    """
    Samples negative items from the item pool for pairwise ranking training.
    
    This class is used during training when the dataset contains only positive
    examples (e.g., user clicked items). It samples random negative items
    (items the user did not interact with) for BPR-style training.
    
    Usage:
        sampler = NegativeSampler(item_features_df, item_id_col='cand_item_id')
        neg_item_ids = sampler.sample_negatives(
            positive_item_ids=[1, 2, 3],
            num_negatives=4
        )
    """
    
    def __init__(
        self,
        item_features_df: pd.DataFrame,
        item_id_col: str = 'cand_item_id',
    ):
        """
        Initialize the negative sampler.
        
        Args:
            item_features_df: DataFrame containing item features.
                              If indexed, the index is used as item IDs.
                              Otherwise, item_id_col is used.
            item_id_col: Column name for item IDs (used if df is not indexed)
        """
        self.item_id_col = item_id_col
        
        # Extract item IDs
        if item_features_df.index.name == item_id_col or item_id_col not in item_features_df.columns:
            # DataFrame is indexed by item_id
            self.item_features_df = item_features_df
            self.all_item_ids = np.array(item_features_df.index.tolist())
        else:
            # Set index from column
            self.item_features_df = item_features_df.set_index(item_id_col)
            self.all_item_ids = np.array(self.item_features_df.index.tolist())
        
        self.num_items = len(self.all_item_ids)
        self.item_id_set = set(self.all_item_ids)
        
        logger.info(f"NegativeSampler initialized with {self.num_items} items")
    
    def sample_negatives(
        self,
        positive_item_ids: List[Any],
        num_negatives: int,
        exclude_items: Optional[Set[Any]] = None,
    ) -> List[Any]:
        """
        Sample negative items excluding positive items.
        
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
        
        candidates = list(self.item_id_set - exclude)
        
        if len(candidates) < num_negatives:
            # Not enough candidates, return all available
            return candidates
        
        return list(np.random.choice(candidates, size=num_negatives, replace=False))
    
    def sample_negatives_batch(
        self,
        positive_item_ids: np.ndarray,
        num_negatives: int,
    ) -> np.ndarray:
        """
        Sample negative items for a batch of positive items.
        
        Args:
            positive_item_ids: Array of positive item IDs [batch_size]
            num_negatives: Number of negatives per positive
            
        Returns:
            Array of negative item IDs [batch_size, num_negatives]
        """
        batch_size = len(positive_item_ids)
        negatives = np.zeros((batch_size, num_negatives), dtype=positive_item_ids.dtype)
        
        for i, pos_id in enumerate(positive_item_ids):
            neg_ids = self.sample_negatives([pos_id], num_negatives)
            
            # Pad if not enough negatives
            if len(neg_ids) < num_negatives:
                neg_ids = neg_ids + [neg_ids[0]] * (num_negatives - len(neg_ids))
            
            negatives[i] = neg_ids
        
        return negatives
    
    def get_item_features(
        self,
        item_ids: List[Any],
    ) -> pd.DataFrame:
        """
        Get features for the specified item IDs.
        
        Args:
            item_ids: List of item IDs
            
        Returns:
            DataFrame with item features
        """
        return self.item_features_df.loc[item_ids]
    
    def get_item_features_as_dict(
        self,
        item_ids: List[Any],
    ) -> Dict[str, np.ndarray]:
        """
        Get features for item IDs as a dictionary of numpy arrays.
        
        Args:
            item_ids: List of item IDs
            
        Returns:
            Dictionary mapping feature names to numpy arrays
        """
        df = self.get_item_features(item_ids)
        return {col: df[col].values for col in df.columns}
