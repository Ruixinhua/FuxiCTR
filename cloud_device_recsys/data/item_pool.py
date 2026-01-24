# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Item Pool Manager

Manages the candidate item pool for retrieval evaluation.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Set
import polars as pl
import pandas as pd
import numpy as np
import csv


class ItemPool:
    """
    Manages the item pool for retrieval and candidate evaluation.
    
    The item pool contains all unique items with their features
    for use in retrieval evaluation (scoring all items).
    """
    
    def __init__(self, item_features: List[str] = None):
        """
        Initialize item pool.
        
        Args:
            item_features: List of feature names that identify/describe items
        """
        self.item_features = item_features or []
        self.items_df: Optional[pd.DataFrame] = None
        self.item_embeddings: Optional[np.ndarray] = None
        self.item_id_to_idx: Dict[Any, int] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_from_data(self,
                          data_path: str,
                          item_features: List[str] = None,
                          positive_only: bool = True,
                          label_col: str = 'label') -> pd.DataFrame:
        """
        Extract unique items from data.
        
        Args:
            data_path: Path to data (parquet file or directory)
            item_features: Features to extract for each item
            positive_only: Only extract items from positive samples
            label_col: Name of label column
            
        Returns:
            DataFrame with unique items
        """
        if item_features:
            self.item_features = item_features
        
        if not self.item_features:
            raise ValueError("item_features must be specified")
        
        self.logger.info(f"Extracting items from {data_path}")
        self.logger.info(f"Item features: {self.item_features}")
        
        # Handle directory or file path
        if os.path.isdir(data_path):
            data_path = os.path.join(data_path, "*.parquet")
        
        # Load data
        df = pl.scan_parquet(data_path)
        
        # Filter to positive samples if requested
        if positive_only:
            df = df.filter(pl.col(label_col) == 1)
        
        # Get available columns
        schema = df.collect_schema()
        available_features = [f for f in self.item_features if f in schema.names()]
        
        if not available_features:
            raise ValueError(f"None of item_features found in data: {self.item_features}")
        
        self.logger.info(f"Available item features: {available_features}")
        
        # Extract unique items
        self.items_df = df.select(available_features).unique().collect().to_pandas()
        
        # Build item ID mapping
        self._build_item_index()
        
        self.logger.info(f"Extracted {len(self.items_df)} unique items")
        
        return self.items_df
    
    def _build_item_index(self) -> None:
        """Build mapping from item ID to index"""
        if self.items_df is None:
            return
        
        # Use first feature as primary item ID
        primary_id = self.item_features[0] if self.item_features else self.items_df.columns[0]
        
        self.item_id_to_idx = {
            item_id: idx 
            for idx, item_id in enumerate(self.items_df[primary_id].values)
        }
    
    def get_item_features(self, item_ids: List[Any]) -> pd.DataFrame:
        """
        Get features for specific items.
        
        Args:
            item_ids: List of item IDs
            
        Returns:
            DataFrame with item features
        """
        if self.items_df is None:
            raise ValueError("Item pool not loaded")
        
        primary_id = self.item_features[0] if self.item_features else self.items_df.columns[0]
        
        return self.items_df[self.items_df[primary_id].isin(item_ids)]
    
    def get_all_items(self) -> pd.DataFrame:
        """Get all items in the pool"""
        return self.items_df
    
    def get_item_count(self) -> int:
        """Get number of items in pool"""
        return len(self.items_df) if self.items_df is not None else 0
    
    def set_item_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Set precomputed item embeddings.
        
        Args:
            embeddings: Item embeddings [num_items, embedding_dim]
        """
        if self.items_df is None:
            raise ValueError("Item pool not loaded")
        
        if len(embeddings) != len(self.items_df):
            raise ValueError(f"Embedding count {len(embeddings)} != item count {len(self.items_df)}")
        
        self.item_embeddings = embeddings
        self.logger.info(f"Set item embeddings: {embeddings.shape}")
    
    def get_item_embeddings(self) -> np.ndarray:
        """Get item embeddings"""
        return self.item_embeddings
    
    def save(self, path: str) -> Dict[str, str]:
        """
        Save item pool to files.
        
        Args:
            path: Base path for saving
            
        Returns:
            Dictionary of saved file paths
        """
        saved_files = {}
        
        # Save items DataFrame
        if self.items_df is not None:
            parquet_path = path if path.endswith('.parquet') else f"{path}.parquet"
            self.items_df.to_parquet(parquet_path, index=False)
            saved_files['items'] = parquet_path
            
            # Also save as CSV
            csv_path = parquet_path.replace('.parquet', '.csv')
            self.items_df.to_csv(csv_path, index=False)
            saved_files['items_csv'] = csv_path
            
            self.logger.info(f"Saved {len(self.items_df)} items to {parquet_path}")
        
        # Save embeddings if available
        if self.item_embeddings is not None:
            emb_path = path.replace('.parquet', '') + '_embeddings.npy'
            np.save(emb_path, self.item_embeddings)
            saved_files['embeddings'] = emb_path
            self.logger.info(f"Saved embeddings to {emb_path}")
        
        # Save metadata
        meta_path = path.replace('.parquet', '') + '_meta.csv'
        with open(meta_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['key', 'value'])
            writer.writerow(['num_items', len(self.items_df) if self.items_df is not None else 0])
            writer.writerow(['item_features', ','.join(self.item_features)])
            if self.item_embeddings is not None:
                writer.writerow(['embedding_dim', self.item_embeddings.shape[1]])
        saved_files['meta'] = meta_path
        
        return saved_files
    
    def load(self, path: str) -> 'ItemPool':
        """
        Load item pool from files.
        
        Args:
            path: Path to parquet file
            
        Returns:
            Self for chaining
        """
        parquet_path = path if path.endswith('.parquet') else f"{path}.parquet"
        
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Item pool not found: {parquet_path}")
        
        self.items_df = pd.read_parquet(parquet_path)
        self.item_features = list(self.items_df.columns)
        self._build_item_index()
        
        self.logger.info(f"Loaded {len(self.items_df)} items from {parquet_path}")
        
        # Load embeddings if available
        emb_path = parquet_path.replace('.parquet', '_embeddings.npy')
        if os.path.exists(emb_path):
            self.item_embeddings = np.load(emb_path)
            self.logger.info(f"Loaded embeddings: {self.item_embeddings.shape}")
        
        return self
    
    def sample_negatives(self,
                         positive_items: List[Any],
                         num_negatives: int,
                         exclude_items: Set[Any] = None) -> List[Any]:
        """
        Sample negative items (items not in positive set).
        
        Args:
            positive_items: List of positive item IDs
            num_negatives: Number of negatives to sample
            exclude_items: Additional items to exclude
            
        Returns:
            List of negative item IDs
        """
        if self.items_df is None:
            raise ValueError("Item pool not loaded")
        
        primary_id = self.item_features[0] if self.item_features else self.items_df.columns[0]
        all_items = set(self.items_df[primary_id].values)
        
        exclude = set(positive_items)
        if exclude_items:
            exclude.update(exclude_items)
        
        candidates = list(all_items - exclude)
        
        if len(candidates) < num_negatives:
            return candidates
        
        return list(np.random.choice(candidates, size=num_negatives, replace=False))
