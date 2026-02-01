#!/usr/bin/env python
# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Pipeline Test Script for TaobaoOpenMCC Dataset

This script tests the complete cloud-device recommendation pipeline
using the TaobaoOpenMCC dataset.

Usage:
    python test_pipeline.py --mode preprocess  # Preprocess data
    python test_pipeline.py --mode train       # Train all stages
    python test_pipeline.py --mode evaluate    # Evaluate all stages
    python test_pipeline.py --mode full        # Run full pipeline
    python test_pipeline.py --sample_size 1000 # Use sample for quick test
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_device_recsys.config.feature_groups import FeatureGroupManager, FeatureGroup
from cloud_device_recsys.data.item_pool import ItemPool
from cloud_device_recsys.data.stage_data_builder import StageDataBuilder
from cloud_device_recsys.data.eval_data_generator import EvalDataGenerator


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('PipelineTest')


def load_config():
    """Load data configuration"""
    config_path = Path(__file__).parent / 'config' / 'data_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_feature_groups(config: dict, logger: logging.Logger):
    """Test feature group assignment"""
    logger.info("=" * 60)
    logger.info("Testing Feature Group Assignment")
    logger.info("=" * 60)
    
    fg_manager = FeatureGroupManager()
    
    # Assign features from config
    for feature in config.get('feature_cols', []):
        name = feature['name']
        fg_str = feature.get('feature_group', 'FG1')
        fg = FeatureGroup.from_string(fg_str)
        fg_manager.assign_feature(name, fg)
    
    # Print summary
    fg1 = fg_manager.get_features_by_group(FeatureGroup.FG1)
    fg2 = fg_manager.get_features_by_group(FeatureGroup.FG2)
    fg3 = fg_manager.get_features_by_group(FeatureGroup.FG3)
    
    logger.info(f"FG1 (Non-personalized): {len(fg1)} features")
    for f in sorted(fg1):
        logger.info(f"  - {f}")
    
    logger.info(f"FG2 (Cloud-personalized): {len(fg2)} features")
    for f in sorted(fg2):
        logger.info(f"  - {f}")
    
    logger.info(f"FG3 (Device-only): {len(fg3)} features")
    for f in sorted(fg3):
        logger.info(f"  - {f}")
    
    # Test feature filtering
    cloud_features = fg_manager.get_cloud_features()
    device_features = fg_manager.get_device_features()
    
    logger.info(f"\nCloud features (FG1+FG2): {len(cloud_features)}")
    logger.info(f"Device features (FG1+FG2+FG3): {len(device_features)}")
    
    # Validate cloud doesn't have FG3
    fg3_in_cloud = cloud_features & fg3
    if fg3_in_cloud:
        logger.error(f"ERROR: FG3 features in cloud: {fg3_in_cloud}")
    else:
        logger.info("✓ Cloud features correctly exclude FG3")
    
    return fg_manager


def test_item_pool(config: dict, logger: logging.Logger):
    """Test item pool loading"""
    logger.info("=" * 60)
    logger.info("Testing Item Pool")
    logger.info("=" * 60)
    
    item_pool_path = config['dataset'].get('item_pool')
    if not item_pool_path:
        logger.warning("No item pool path in config")
        return None
    
    # Load item pool
    logger.info(f"Loading item pool from: {item_pool_path}")
    
    item_features = config.get('item_pool', {}).get('item_features', [])
    logger.info(f"Item features: {item_features}")
    
    item_pool = ItemPool(item_features=item_features)
    
    # Load from CSV
    pool_df = pd.read_csv(item_pool_path)
    logger.info(f"Item pool shape: {pool_df.shape}")
    logger.info(f"Columns: {list(pool_df.columns)}")
    
    # Filter to only relevant columns
    available_features = [f for f in item_features if f in pool_df.columns]
    if available_features:
        item_pool.items_df = pool_df[available_features]
        item_pool.item_features = available_features
        item_pool._build_item_index()
    else:
        item_pool.items_df = pool_df
        item_pool.item_features = list(pool_df.columns)
        item_pool._build_item_index()
    
    logger.info(f"✓ Loaded {item_pool.get_item_count()} items")
    logger.info(f"Sample items:\n{item_pool.items_df.head()}")
    
    return item_pool


def test_data_loading(config: dict, logger: logging.Logger, sample_size: int = None):
    """Test data loading with optional sampling"""
    logger.info("=" * 60)
    logger.info("Testing Data Loading")
    logger.info("=" * 60)
    
    data_root = config['dataset']['data_root']
    
    for split in ['train', 'valid', 'test']:
        data_path = config['dataset'].get(f'{split}_data')
        if not data_path or not os.path.exists(data_path):
            logger.warning(f"No {split} data found")
            continue
        
        logger.info(f"\nLoading {split} data from: {data_path}")
        
        # Load sample
        if sample_size:
            df = pd.read_csv(data_path, nrows=sample_size)
        else:
            # Just get row count
            import subprocess
            result = subprocess.run(['wc', '-l', data_path], capture_output=True, text=True)
            line_count = int(result.stdout.split()[0])
            logger.info(f"{split} data: {line_count:,} rows")
            df = pd.read_csv(data_path, nrows=5)
        
        logger.info(f"Columns ({len(df.columns)}): {list(df.columns)}")
        logger.info(f"Sample shape: {df.shape}")
        
        # Check label distribution
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            logger.info(f"Label distribution:\n{label_counts}")
    
    return True


def test_stage_data_builder(config: dict, fg_manager: FeatureGroupManager, 
                            item_pool: ItemPool, logger: logging.Logger,
                            sample_size: int = 1000):
    """Test stage data builder with sample data"""
    logger.info("=" * 60)
    logger.info("Testing Stage Data Builder")
    logger.info("=" * 60)
    
    # Load sample data
    train_path = config['dataset'].get('train_data')
    if not train_path or not os.path.exists(train_path):
        logger.error("No training data found")
        return None
    
    logger.info(f"Loading {sample_size} sample rows...")
    df = pd.read_csv(train_path, nrows=sample_size)
    logger.info(f"Loaded sample: {df.shape}")
    
    # Create builder
    builder = StageDataBuilder(
        fg_manager=fg_manager,
        item_pool=item_pool,
        output_dir="./outputs/test_stage_data"
    )
    
    # Test filtering for each stage
    for stage in ['retrieval', 'preranking', 'reranking']:
        logger.info(f"\n--- Testing {stage} stage ---")
        
        config_stage = builder.get_stage_config(stage)
        logger.info(f"Config: top_k={config_stage.top_k}, metrics={config_stage.metrics}")
        
        filtered_df = builder.filter_features(df, stage)
        logger.info(f"Filtered columns ({len(filtered_df.columns)}): {list(filtered_df.columns)}")
        
        # Check if FG3 is only in reranking
        fg3_features = fg_manager.get_features_by_group(FeatureGroup.FG3)
        has_fg3 = bool(set(filtered_df.columns) & fg3_features)
        
        if stage == 'reranking':
            if has_fg3:
                logger.info(f"✓ Re-ranking has FG3 features as expected")
            else:
                logger.warning(f"Re-ranking missing FG3 features!")
        else:
            if has_fg3:
                logger.error(f"ERROR: {stage} has FG3 features (should not!)")
            else:
                logger.info(f"✓ {stage} correctly excludes FG3")
    
    return builder


def test_eval_metrics(logger: logging.Logger):
    """Test evaluation metrics computation"""
    logger.info("=" * 60)
    logger.info("Testing Evaluation Metrics")
    logger.info("=" * 60)
    
    eval_gen = EvalDataGenerator(output_dir="./outputs/test_eval")
    
    # Test Recall@K
    logger.info("\nTesting Recall@K:")
    retrieved = {
        'user1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'user2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    }
    ground_truth = {
        'user1': [1, 2, 3],  # 3 in top-10
        'user2': [11, 21, 22]  # 1 in top-10
    }
    
    for k in [5, 10]:
        recall = eval_gen.compute_recall_at_k(retrieved, ground_truth, k)
        hit_rate = eval_gen.compute_hit_rate_at_k(retrieved, ground_truth, k)
        logger.info(f"Recall@{k}: {recall:.4f}, HitRate@{k}: {hit_rate:.4f}")
    
    # Test nDCG@K
    logger.info("\nTesting nDCG@K:")
    predictions = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    
    for k in [5, 10]:
        ndcg = eval_gen.compute_ndcg_at_k(predictions, labels, k)
        logger.info(f"nDCG@{k}: {ndcg:.4f}")
    
    # Test AUC
    logger.info("\nTesting AUC:")
    auc = eval_gen.compute_auc(predictions, labels)
    logger.info(f"AUC: {auc:.4f}")
    
    logger.info("✓ All metrics computed successfully")
    
    return eval_gen


def run_full_test(sample_size: int = 1000):
    """Run full pipeline test"""
    output_dir = "./outputs/pipeline_test"
    logger = setup_logging(output_dir)
    
    logger.info("=" * 60)
    logger.info("Cloud-Device Recommendation Pipeline Test")
    logger.info("Dataset: TaobaoOpenMCC")
    logger.info(f"Sample size: {sample_size}")
    logger.info("=" * 60)
    
    # Load config
    config = load_config()
    
    # Test 1: Feature groups
    fg_manager = test_feature_groups(config, logger)
    
    # Test 2: Item pool
    item_pool = test_item_pool(config, logger)
    
    # Test 3: Data loading
    test_data_loading(config, logger, sample_size)
    
    # Test 4: Stage data builder
    if item_pool:
        test_stage_data_builder(config, fg_manager, item_pool, logger, sample_size)
    
    # Test 5: Evaluation metrics
    test_eval_metrics(logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Test Cloud-Device Pipeline')
    parser.add_argument('--mode', type=str, default='test',
                       choices=['test', 'preprocess', 'train', 'evaluate', 'full'],
                       help='Test mode')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='Sample size for quick testing')
    parser.add_argument('--output_dir', type=str, default='./outputs/pipeline_test',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        run_full_test(args.sample_size)
    else:
        print(f"Mode '{args.mode}' - use run_pipeline.py for training/evaluation")


if __name__ == '__main__':
    main()
