#!/usr/bin/env python
# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Main Pipeline Runner

Entry point for running the cloud-device recommendation pipeline.
Supports both full pipeline execution and individual stage runs.

Usage:
    # Full pipeline
    python run_pipeline.py --config ./config --mode full --gpu 0
    
    # Individual stages
    python run_pipeline.py --config ./config --mode retrieval --gpu 0
    python run_pipeline.py --config ./config --mode preranking --gpu 0
    python run_pipeline.py --config ./config --mode reranking --gpu 0
    
    # Training
    python run_pipeline.py --config ./config --mode train --stage retrieval --gpu 0
"""

import os
import sys
import argparse
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fuxictr.utils import load_config, set_logger, print_to_json, save_results_to_csv
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import build_dataset

from cloud_device_recsys.config.feature_groups import FeatureGroupManager, FeatureGroup
from cloud_device_recsys.pipeline.pipeline_coordinator import PipelineCoordinator
from cloud_device_recsys.pipeline.base_stage import StageType
from cloud_device_recsys.retrieval.retrieval_stage import RetrievalStage
from cloud_device_recsys.preranking.preranking_stage import PrerankingStage
from cloud_device_recsys.reranking.reranking_stage import RerankingStage


def setup_logging(output_dir: str) -> None:
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_pipeline_config(config_dir: str, pipeline_id: str) -> dict:
    """Load pipeline configuration"""
    config_path = os.path.join(config_dir, f"{pipeline_id}.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "pipeline_config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def build_feature_group_manager(config: dict) -> FeatureGroupManager:
    """Build and configure feature group manager"""
    manager = FeatureGroupManager()
    
    # Load custom assignments if provided
    if 'feature_groups' in config:
        for fg_name, features in config['feature_groups'].items():
            fg = FeatureGroup.from_string(fg_name)
            for feature in features:
                manager.assign_feature(feature, fg)
    
    return manager


def create_stages(
    feature_map: FeatureMap,
    feature_group_manager: FeatureGroupManager,
    config: dict,
    output_dir: str
) -> dict:
    """Create pipeline stages based on configuration"""
    stages = {}
    
    # Get stages config (may be at root or under 'stages' key)
    stages_config = config.get('stages', config)
    
    # Retrieval stage
    if 'retrieval' in stages_config:
        retrieval_config = stages_config['retrieval']
        stages['retrieval'] = RetrievalStage(
            feature_map=feature_map,
            feature_group_manager=feature_group_manager,
            model_params=retrieval_config.get('model_params', {}),
            output_dir=os.path.join(output_dir, 'retrieval'),
            top_k=retrieval_config.get('top_k', 1000)
        )
    
    # Pre-ranking stage
    if 'preranking' in stages_config:
        preranking_config = stages_config['preranking']
        stages['preranking'] = PrerankingStage(
            feature_map=feature_map,
            feature_group_manager=feature_group_manager,
            model_params=preranking_config.get('model_params', {}),
            output_dir=os.path.join(output_dir, 'preranking'),
            top_k=preranking_config.get('top_k', 100),
            use_diversity=preranking_config.get('use_diversity', False)
        )
    
    # Re-ranking stage
    if 'reranking' in stages_config:
        reranking_config = stages_config['reranking']
        stages['reranking'] = RerankingStage(
            feature_map=feature_map,
            feature_group_manager=feature_group_manager,
            model_params=reranking_config.get('model_params', {}),
            output_dir=os.path.join(output_dir, 'reranking'),
            top_k=reranking_config.get('top_k', 10),
            support_distillation=reranking_config.get('distillation', False)
        )
    
    return stages


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Cloud-Device Recommendation Pipeline')
    parser.add_argument('--config', type=str, default='./config',
                       help='Configuration directory')
    parser.add_argument('--pipeline_id', type=str, default='default',
                       help='Pipeline configuration ID')
    parser.add_argument('--dataset_id', type=str, default='taobao_mcc_x1',
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
    parser.add_argument('--prev_output', type=str, default=None,
                       help='Path to previous stage output (for individual stage runs)')
    parser.add_argument('--seed', type=int, default=2024,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.output_dir)
    logger = logging.getLogger('PipelineRunner')
    
    logger.info(f"Starting pipeline with mode: {args.mode}")
    logger.info(f"Configuration: {args.config}, Pipeline: {args.pipeline_id}")
    
    seed_everything(args.seed)
    
    # Load configurations
    pipeline_config = load_pipeline_config(args.config, args.pipeline_id)
    
    # Load dataset configuration
    dataset_config_path = os.path.join(args.config, 'dataset_config.yaml')
    with open(dataset_config_path, 'r') as f:
        all_dataset_config = yaml.safe_load(f)
    
    dataset_config = all_dataset_config.get(args.dataset_id, {})
    dataset_config['gpu'] = args.gpu
    
    logger.info(f"Dataset: {args.dataset_id}")
    
    # Build feature map
    # Use processed_data_root if available, otherwise data_root + dataset_id
    if 'processed_data_root' in dataset_config:
        data_dir = dataset_config['processed_data_root']
    else:
        data_dir = os.path.join(dataset_config.get('data_root', './data'), args.dataset_id)
        
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    
    if os.path.exists(feature_map_json):
        # Convert dict-format feature_map to FuxiCTR's list format if needed
        with open(feature_map_json, 'r') as f:
            fm_data = json.load(f)
        
        # Check if features is a dict (our format) vs list (FuxiCTR format)
        if isinstance(fm_data.get('features'), dict):
            logger.info("Converting dict-format feature_map to FuxiCTR list format...")
            # Convert {"name": {...}, ...} to [{"name": {...}}, ...]
            fm_data['features'] = [{k: v} for k, v in fm_data['features'].items()]
            
            # Write converted format back
            converted_json = os.path.join(data_dir, "feature_map_fuxictr.json")
            with open(converted_json, 'w') as f:
                json.dump(fm_data, f, indent=2)
            feature_map_json = converted_json
            logger.info(f"Saved converted feature_map to {converted_json}")
        
        feature_map = FeatureMap(args.dataset_id, data_dir)
        feature_map.load(feature_map_json, dataset_config)
        logger.info(f"Loaded feature map with {len(feature_map.features)} features")
    else:
        logger.warning(f"Feature map not found at {feature_map_json}")
        logger.info("Please run data preprocessing first")
        return
    
    # Build feature group manager
    fg_manager = build_feature_group_manager(pipeline_config)
    fg_manager.auto_assign_groups(feature_map)
    
    # Save feature group assignments
    fg_csv_path = os.path.join(args.output_dir, 'feature_group_assignments.csv')
    fg_manager.save_to_csv(fg_csv_path)
    logger.info(f"Saved feature group assignments to {fg_csv_path}")
    
    # Create stages
    stages = create_stages(
        feature_map=feature_map,
        feature_group_manager=fg_manager,
        config=pipeline_config,
        output_dir=args.output_dir
    )
    
    # Create pipeline coordinator
    coordinator = PipelineCoordinator(
        output_dir=args.output_dir,
        save_intermediate=True,
        save_csv=True
    )
    
    # Register stages
    stage_type_map = {
        'retrieval': StageType.RETRIEVAL,
        'preranking': StageType.PRERANKING,
        'reranking': StageType.RERANKING
    }
    
    for name, stage in stages.items():
        coordinator.register_stage(stage)
    
    # Execute based on mode
    if args.mode == 'full':
        logger.info("Running full pipeline")
        
        # Build item index for retrieval stage (required for real predictions)
        if 'retrieval' in stages:
            retrieval_stage = stages['retrieval']
            if retrieval_stage.item_embeddings is None:
                logger.info("Building item index for retrieval stage...")
                
                # Get data paths - use processed item pool if available
                item_pool_config = dataset_config.get('item_pool', {})
                item_pool_file = item_pool_config.get('file', 'cand_item_list')
                processed_data_root = dataset_config.get('processed_data_root', data_dir)
                raw_data_root = dataset_config.get('raw_data_root', data_dir)
                
                # Try processed item pool first (parquet format)
                processed_item_pool = os.path.join(processed_data_root, 'item_pool.parquet')
                raw_item_pool = os.path.join(raw_data_root, f'{item_pool_file}.csv')
                
                if os.path.exists(processed_item_pool):
                    item_pool_path = processed_item_pool
                    data_format = 'parquet'
                elif os.path.exists(raw_item_pool):
                    item_pool_path = raw_item_pool
                    data_format = 'csv'
                else:
                    item_pool_path = None
                    
                if item_pool_path:
                    logger.info(f"Loading item pool from: {item_pool_path}")
                    
                    # Build model first if not built
                    if retrieval_stage.model is None:
                        retrieval_stage.build_model()
                    
                    # For CSV files, we need to preprocess them first
                    # For now, use the processed train data to build item embeddings
                    # from unique items (simpler approach)
                    if data_format == 'csv':
                        logger.info("Using train data items for building index (CSV item pool not supported by RankDataLoader)")
                        # Use processed train data instead
                        train_path = os.path.join(processed_data_root, 'train.parquet')
                        if os.path.exists(train_path):
                            item_loader = RankDataLoader(
                                feature_map=feature_map,
                                stage='test',
                                test_data=train_path,
                                batch_size=4096,
                                shuffle=False,
                                data_format='parquet'
                            )
                            item_gen = item_loader.make_iterator()
                            retrieval_stage.build_item_index(item_gen)
                            logger.info("Item index built from train data")
                    else:
                        # Use processed item pool (parquet)
                        item_loader = RankDataLoader(
                            feature_map=feature_map,
                            stage='test',
                            test_data=item_pool_path,
                            batch_size=4096,
                            shuffle=False,
                            data_format='parquet'
                        )
                        item_gen = item_loader.make_iterator()
                        retrieval_stage.build_item_index(item_gen)
                        logger.info("Item index built successfully")
                else:
                    logger.warning(f"Item pool not found at {processed_item_pool} or {raw_item_pool}")
        
        results = coordinator.run_full_pipeline()
        logger.info("Pipeline complete!")
        for stage_name, output in results.items():
            logger.info(f"  {stage_name}: {output.get_total_candidates()} candidates")
    
    elif args.mode in ['retrieval', 'preranking', 'reranking']:
        stage_type = stage_type_map[args.mode]
        logger.info(f"Running single stage: {args.mode}")
        output = coordinator.run_stage(
            stage_type=stage_type,
            prev_output_path=args.prev_output
        )
        logger.info(f"Stage complete: {output.get_total_candidates()} candidates")
    
    elif args.mode == 'train':
        if args.stage is None:
            logger.error("--stage required for train mode")
            return
        
        stage_type = stage_type_map[args.stage]
        logger.info(f"Training stage: {args.stage}")
        
        # Build data paths from dataset config
        data_dir = dataset_config.get('processed_data_root', data_dir)
        data_format = dataset_config.get('processed_data_format', 'parquet')
        
        train_path = os.path.join(data_dir, f'train.{data_format}')
        valid_path = os.path.join(data_dir, f'valid.{data_format}')
        
        logger.info(f"Loading training data from: {train_path}")
        logger.info(f"Loading validation data from: {valid_path}")
        
        # Get stage training config
        stages_config = pipeline_config.get('stages', pipeline_config)
        stage_config = stages_config.get(args.stage, {})
        training_config = stage_config.get('training', {})
        
        batch_size = training_config.get('batch_size', 4096)
        epochs = training_config.get('epochs', 10)
        
        # Build data loaders with correct parameters
        data_loader = RankDataLoader(
            feature_map=feature_map,
            stage='train',
            train_data=train_path,
            valid_data=valid_path,
            batch_size=batch_size,
            shuffle=True,
            data_format=data_format
        )
        train_gen, valid_gen = data_loader.make_iterator()
        
        logger.info(f"Starting training with batch_size={batch_size}, epochs={epochs}")
        
        metrics = coordinator.train_stage(
            stage_type=stage_type,
            train_data=train_gen,
            valid_data=valid_gen,
            epochs=epochs
        )
        logger.info(f"Training complete: {metrics}")
        
        # For retrieval stage, build item index after training
        if args.stage == 'retrieval':
            logger.info("Building item index for retrieval...")
            
            # Load item pool data
            item_pool_config = dataset_config.get('item_pool', {})
            item_pool_file = item_pool_config.get('file', 'cand_item_list')
            raw_data_root = dataset_config.get('raw_data_root', data_dir)
            raw_format = dataset_config.get('raw_data_format', 'csv')
            item_pool_path = os.path.join(raw_data_root, f'{item_pool_file}.{raw_format}')
            
            logger.info(f"Loading item pool from: {item_pool_path}")
            
            # Create item data loader
            item_loader = RankDataLoader(
                feature_map=feature_map,
                stage='test',
                test_data=item_pool_path,
                batch_size=batch_size,
                shuffle=False,
                data_format=raw_format
            )
            item_gen = item_loader.make_iterator()
            
            # Build item index
            retrieval_stage = coordinator.get_stage(stage_type)
            if retrieval_stage is not None:
                retrieval_stage.build_item_index(item_gen)
                logger.info("Item index built successfully")
    
    elif args.mode == 'evaluate':
        if args.stage is None:
            logger.error("--stage required for evaluate mode")
            return
        
        stage_type = stage_type_map[args.stage]
        logger.info(f"Evaluating stage: {args.stage}")
        
        # Build test data path
        data_dir = dataset_config.get('processed_data_root', data_dir)
        data_format = dataset_config.get('processed_data_format', 'parquet')
        test_path = os.path.join(data_dir, f'test.{data_format}')
        
        logger.info(f"Loading test data from: {test_path}")
        
        test_loader = RankDataLoader(
            feature_map=feature_map,
            stage='test',
            test_data=test_path,
            batch_size=4096,
            shuffle=False,
            data_format=data_format
        )
        test_gen = test_loader.make_iterator()
        
        metrics = coordinator.evaluate_stage(
            stage_type=stage_type,
            test_data=test_gen
        )
        logger.info(f"Evaluation complete: {metrics}")
    
    # Print pipeline status
    status = coordinator.get_pipeline_status()
    logger.info(f"Pipeline status: {status}")


if __name__ == '__main__':
    main()
