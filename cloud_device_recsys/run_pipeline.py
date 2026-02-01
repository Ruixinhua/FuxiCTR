#!/usr/bin/env python
# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
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

"""

import os
import copy
import sys
import argparse
import logging
import json
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything

from cloud_device_recsys.config.feature_groups import FeatureGroupManager, FeatureGroup
from cloud_device_recsys.retrieval.retrieval_stage import RetrievalStage
from cloud_device_recsys.preranking.preranking_stage import PrerankingStage
from cloud_device_recsys.reranking.reranking_stage import RerankingStage
from cloud_device_recsys.utils import setup_logging, load_pipeline_config, get_data_dir


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
    output_dir: str,
    gpu: int = -1
) -> dict:
    """Create pipeline stages based on configuration"""
    stages = {}
    
    # Get stages config (maybe at root or under 'stages' key)
    stages_config = config.get('stages', config)
    
    # Retrieval stage
    if 'retrieval' in stages_config:
        retrieval_config = stages_config['retrieval']
        # Merge GPU into model params
        model_params = retrieval_config.get('model_params', {})
        model_params['gpu'] = gpu
        # Pass metrics if defined
        if 'metrics' in retrieval_config:
            model_params['metrics'] = retrieval_config['metrics']
        
        stages['retrieval'] = RetrievalStage(
            feature_map=feature_map,
            feature_group_manager=feature_group_manager,
            allowed_feature_groups=[getattr(FeatureGroup, f) for f in retrieval_config['features']],
            model_params=model_params,
            output_dir=os.path.join(output_dir, 'retrieval'),
            top_k=retrieval_config.get('top_k', 1000)
        )
    
    # Pre-ranking stage
    if 'preranking' in stages_config:
        preranking_config = stages_config['preranking']
        # Merge GPU into model params
        model_params = preranking_config.get('model_params', {})
        model_params['gpu'] = gpu
        
        stages['preranking'] = PrerankingStage(
            feature_map=feature_map,
            feature_group_manager=feature_group_manager,
            model_params=model_params,
            output_dir=os.path.join(output_dir, 'preranking'),
            top_k=preranking_config.get('top_k', 100),
            use_diversity=preranking_config.get('use_diversity', False)
        )
    
    # Re-ranking stage
    if 'reranking' in stages_config:
        reranking_config = stages_config['reranking']
        # Merge GPU into model params
        model_params = reranking_config.get('model_params', {})
        model_params['gpu'] = gpu
        
        stages['reranking'] = RerankingStage(
            feature_map=feature_map,
            feature_group_manager=feature_group_manager,
            model_params=model_params,
            output_dir=os.path.join(output_dir, 'reranking'),
            top_k=reranking_config.get('top_k', 10),
            support_distillation=reranking_config.get('distillation', False)
        )
    
    return stages


def run_retrival_stage(retrieval_stage, pipeline_config, dataset_config, fg_manager, logger=setup_logging('./outputs')):
    data_dir = get_data_dir(dataset_config, pipeline_config['dataset_id'])
    retrieval_config = pipeline_config['stages']['retrieval']
    # Use retrieval_stage.feature_map (filtered) to ensure loader only loads allowed features
    # This prevents loading FG3 features which are not in the model's feature_map
    train_fm = retrieval_stage.feature_map
    if retrieval_stage.item_embeddings is None:  # If item embeddings not built yet, then train and build
        logger.info("[Training] Building item index for retrieval stage...")

        # Get data paths - use processed item pool if available
        item_pool_config = dataset_config.get('item_pool', {})
        item_pool_file = item_pool_config.get('file', 'cand_item_list')
        processed_data_root = dataset_config.get('processed_data_root', data_dir)

        # Try processed item pool first (parquet format)
        processed_item_pool = os.path.join(processed_data_root, f'{item_pool_file}.parquet')

        # TRAIN RETRIEVAL MODEL
        logger.info("Training retrieval model...")
        data_format = dataset_config.get('processed_data_format', 'parquet')
        train_path = os.path.join(processed_data_root, f'train.{data_format}')
        valid_path = os.path.join(processed_data_root, f'valid.{data_format}')
        # Create train/valid loaders
        train_loader = RankDataLoader(
            feature_map=train_fm,
            stage='train',
            train_data=train_path,
            valid_data=valid_path,
            batch_size=retrieval_config['training'].get('batch_size', 4096),
            shuffle=True,
            data_format=data_format
        )
        train_gen, valid_gen = train_loader.make_iterator()  # Unpack generators

        # Create Item Pool Loader for Retrieval Evaluation
        logger.info(f"Loading item pool from {processed_item_pool}")

        item_feature_names = [name for name, spec in train_fm.features.items()
                              if fg_manager.feature_assignments.get(name) == FeatureGroup.FG1]
        # Create a lean feature map
        item_fm = copy.deepcopy(train_fm)
        item_fm.features = {k: v for k, v in item_fm.features.items() if k in item_feature_names}
        # IMPORTANT: Re-index the feature map so column indices start from 0 and match the number of features
        item_fm.set_column_index()

        item_loader = RankDataLoader(
            feature_map=item_fm,
            stage='train',  # Acts as feed generator
            train_data=processed_item_pool,
            batch_size=retrieval_config['training'].get('batch_size', 4096),
            shuffle=False,
            data_format=data_format
        )
        item_gen, _ = item_loader.make_iterator()
        retrieval_stage.build_model()
        # Train retrieval stage with custom loop
        retrieval_stage.train(
            train_data=train_gen,
            valid_data=valid_gen,
            item_data=item_gen,
            epochs=retrieval_config['training'].get('epochs', 10),
            patience=retrieval_config['training'].get('patience', 2),
            monitor=retrieval_config['training'].get('monitor', 'Recall@1000'),  # Monitor Recall
            mode=retrieval_config['training'].get('mode', 'max')
        )
        # After training, load the best model and build item index for valid and test evaluation
        retrieval_stage.build_item_index(item_gen)
        logger.info("Item index built successfully")
        retrieval_stage.evaluate(valid_gen)
    test_path = os.path.join(dataset_config.get('processed_data_root', data_dir), 'test.parquet')
    test_loader = RankDataLoader(
        feature_map=train_fm,
        stage='test',
        test_data=test_path,
        batch_size=retrieval_config['training'].get('batch_size', 4096),
        shuffle=False,
        data_format='parquet'
    )
    test_gen = test_loader.make_iterator()
    retrieval_stage.evaluate(test_gen)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Cloud-Device Recommendation Pipeline')
    parser.add_argument('--config', type=str, default='./config',
                       help='Configuration directory')
    parser.add_argument('--pipeline_id', type=str, default='default',
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

    dataset_id = pipeline_config['dataset_id'] if args.dataset_id is None else args.dataset_id
    pipeline_config['dataset_id'] = dataset_id
    dataset_config = all_dataset_config.get(dataset_id, {})
    logger.info(f"Dataset: {dataset_id}")
    dataset_config['gpu'] = args.gpu
    logger.info(f"Used GPU: {args.gpu if args.gpu >=0 else 'CPU'}")

    # Build feature map
    # Use processed_data_root if available, otherwise data_root + dataset_id
    data_dir = get_data_dir(dataset_config, pipeline_config['dataset_id'])
        
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
        
        feature_map = FeatureMap(dataset_id, data_dir)
        feature_map.load(feature_map_json, dataset_config)
        feature_map.dataset_config = dataset_config # Attach config for access in stages
        logger.info(f"Loaded feature map with {len(feature_map.features)} features")
    else:
        logger.warning(f"Feature map not found at {feature_map_json}")
        logger.info("Please run data preprocessing first")
        return
    
    # Build feature group manager
    fg_manager = build_feature_group_manager(pipeline_config)
    # Pass dataset_config to use explicit feature group definitions
    fg_manager.auto_assign_groups(feature_map, dataset_config)
    
    # Save feature group assignments
    fg_csv_path = os.path.join(args.output_dir, 'feature_group_assignments.csv')
    fg_manager.save_to_csv(fg_csv_path)
    logger.info(f"Saved feature group assignments to {fg_csv_path}")
    
    # Create stages
    stages = create_stages(
        feature_map=feature_map,
        feature_group_manager=fg_manager,
        config=pipeline_config,
        output_dir=args.output_dir,
        gpu=args.gpu
    )
    
    # Execute based on mode
    if args.mode == 'full':
        logger.info("Running full pipeline")
        # Build item index for retrieval stage (required for real predictions)
        if 'retrieval' in stages:
            retrieval_stage = stages['retrieval']
            run_retrival_stage(retrieval_stage, pipeline_config, dataset_config, fg_manager, logger=logger)
    elif args.mode == 'retrival':
        logger.info("Running retrieval stage only")
        if 'retrieval' not in stages:
            logger.error("Retrieval stage not configured in pipeline")
            return
        retrieval_stage = stages['retrieval']
        run_retrival_stage(retrieval_stage, pipeline_config, dataset_config, fg_manager, logger=logger)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
