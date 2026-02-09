# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Positive Data Utilities for Pairwise Training.

This module provides utilities for managing positive-only training data
used in pairwise ranking with negative sampling.

Key concepts:
- Pointwise training: uses full data (positive + negative samples)
- Pairwise training: uses positive-only data + dynamic negative sampling

Training mode selection is based on `num_negatives` config parameter:
- num_negatives = 0 → Pointwise (full data)
- num_negatives > 0 → Pairwise (positive-only + negative sampling)
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_positive_train_path(train_path: str) -> str:
    """
    Get the path for positive-only training data.
    
    Convention: If train data is at `path/train.parquet`, 
    positive-only data is at `path/train_positive.parquet`.
    
    Args:
        train_path: Path to the original training data file
        
    Returns:
        Path to the positive-only training data file
    """
    path = Path(train_path)
    stem = path.stem  # e.g., "train"
    suffix = path.suffix  # e.g., ".parquet"
    positive_name = f"{stem}_positive{suffix}"
    return str(path.parent / positive_name)


def ensure_positive_train_data(
    train_path: str,
    label_col: str = "label",
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Ensure positive-only training data exists, creating it if necessary.
    
    This function is idempotent - it will not recreate the file if it exists.
    
    Args:
        train_path: Path to the original training data (containing both positive and negative)
        label_col: Name of the label column (1 = positive, 0 = negative)
        logger: Optional logger instance
        
    Returns:
        Path to the positive-only training data file
        
    Raises:
        FileNotFoundError: If the original train_path doesn't exist
        ValueError: If the label column is not found
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    positive_path = get_positive_train_path(train_path)
    
    # Check if already exists
    if os.path.exists(positive_path):
        logger.info(f"Positive-only training data already exists: {positive_path}")
        return positive_path
    
    # Check source exists
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    logger.info(f"Creating positive-only training data from {train_path}...")
    
    # Load original data
    if train_path.endswith('.parquet'):
        df = pd.read_parquet(train_path)
    elif train_path.endswith('.csv'):
        df = pd.read_csv(train_path)
    else:
        raise ValueError(f"Unsupported file format: {train_path}")
    
    original_count = len(df)
    logger.info(f"Original dataset size: {original_count} samples")
    
    # Check label column exists
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Available: {list(df.columns)}")
    
    # Filter positive samples
    positive_df = df[df[label_col] == 1]
    positive_count = len(positive_df)
    
    logger.info(f"Filtered to {positive_count} positive samples ({positive_count/original_count*100:.1f}%)")
    
    # Save positive-only data
    if train_path.endswith('.parquet'):
        positive_df.to_parquet(positive_path, index=False)
    else:
        positive_df.to_csv(positive_path, index=False)
    
    logger.info(f"Saved positive-only training data to: {positive_path}")
    
    return positive_path


def get_train_path_for_mode(
    train_path: str,
    num_negatives: int,
    label_col: str = "label",
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Get the appropriate training data path based on training mode.
    
    Args:
        train_path: Path to the original training data
        num_negatives: Number of negatives per positive (0 = pointwise mode)
        label_col: Name of the label column
        logger: Optional logger instance
        
    Returns:
        Path to the training data to use:
        - Original path if num_negatives <= 0 (pointwise mode)
        - Positive-only path if num_negatives > 0 (pairwise mode)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if num_negatives <= 0:
        # Pointwise mode: use full dataset
        logger.info(f"Training mode: Pointwise ({train_path})")
        return train_path
    else:
        # Pairwise mode: use positive-only dataset
        logger.info(f"Training mode: Pairwise with {num_negatives} negatives per positive")
        return ensure_positive_train_data(train_path, label_col, logger)
