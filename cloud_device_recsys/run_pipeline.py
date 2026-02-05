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
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything

from cloud_device_recsys.config.feature_groups import FeatureGroupManager, FeatureGroup
from cloud_device_recsys.pipeline import RetrievalStage, PrerankingStage, RerankingStage
from cloud_device_recsys.utils import (
    setup_logging, get_data_dir, get_data_paths,
    prepare_debug_paths, parse_pipeline_args,
    save_stage_output, load_stage_outputs_from_dir
)
from cloud_device_recsys.config.config_parser import ConfigParser

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
    output_dir,
    gpu: int = -1
) -> dict:
    """Create pipeline stages based on configuration"""
    stages = {}
    stages_config = config.get('stages', config)

    # Map stage names to their classes and special parameters
    STAGE_REGISTRY = {
        'retrieval': {
            'class': RetrievalStage,
            'model_kwargs_map': {
                'features': 'allowed_feature_groups',
                'top_k': 'top_k'
            }
        },
        'preranking': {
            'class': PrerankingStage,
            'model_kwargs_map': {
                'top_k': 'top_k',
                'use_diversity': 'use_diversity'
            }
        },
        'reranking': {
            'class': RerankingStage,
            'model_kwargs_map': {
                'top_k': 'top_k',
                'distillation': 'support_distillation' # config 'distillation' maps to class 'support_distillation'
            }
        },
    }

    for stage_name, stage_info in STAGE_REGISTRY.items():
        if stage_name in stages_config and stages_config[stage_name].get('enabled', False):
            stage_config = stages_config[stage_name]
            
            # Prepare model_params
            model_params = stage_config.get('model_params', {}).copy() # Use .copy() to avoid modifying original config
            model_params['gpu'] = gpu
            model_params['model'] = stage_config.get('model')  # Pass model name from config
            if 'metrics' in stage_config:
                model_params['metrics'] = stage_config['metrics']
            
            # Prepare stage-specific keyword arguments for the constructor
            stage_kwargs = {
                'feature_map': feature_map,
                'feature_group_manager': feature_group_manager,
                'model_params': model_params,
                'output_dir': os.path.join(output_dir, stage_name),
            }

            # Add extra parameters dynamically
            for config_key, class_param in stage_info['model_kwargs_map'].items():
                if config_key in stage_config:
                    if config_key == 'features': # Special handling for 'features' to map to enum
                        stage_kwargs[class_param] = [getattr(FeatureGroup, f) for f in stage_config['features']]
                    elif config_key == 'distillation': # Special handling for 'distillation' to extract 'enabled'
                        stage_kwargs[class_param] = stage_config[config_key].get('enabled', False)
                    else:
                        stage_kwargs[class_param] = stage_config[config_key]

            # Instantiate the stage
            stages[stage_name] = stage_info['class'](**stage_kwargs)

    return stages

def _prepare_stage_data_loaders(feature_map, stage_config: dict, paths: dict, 
                                  create_train=True, create_test=True, shuffle_train=True,
                                  create_item_loader=False, item_feature_map=None):
    """
    Create data loaders for a pipeline stage.
    
    Args:
        feature_map: FeatureMap instance
        stage_config: Stage-specific config (e.g., pipeline_config['stages']['retrieval'])
        paths: Dict from get_data_paths()
        create_train: Whether to create train/valid loader
        create_test: Whether to create test loader
        shuffle_train: Whether to shuffle training data
        create_item_loader: Whether to create item pool loader (for retrieval)
        item_feature_map: Feature map for item pool (required if create_item_loader=True)
        
    Returns:
        dict with keys: train_loader, test_loader, item_loader (based on flags)
    """
    batch_size = stage_config.get('training', stage_config).get('batch_size', 4096)
    data_format = paths['data_format']
    
    result = {}
    
    if create_train:
        train_fm = copy.deepcopy(feature_map)
        if "impression_id" in train_fm.labels:
            train_fm.labels.remove("impression_id")
        result['train_loader'] = RankDataLoader(
            feature_map=train_fm,
            stage='train',
            train_data=paths['train_path'],
            batch_size=batch_size,
            shuffle=shuffle_train,
            data_format=data_format
        )
        result['valid_loader'] = RankDataLoader(
            feature_map=feature_map,
            stage='test',
            test_data=paths['valid_path'],
            batch_size=batch_size,
            shuffle=False,
            data_format=data_format
        )
    
    if create_test:
        result['test_loader'] = RankDataLoader(
            feature_map=feature_map,
            stage='test',
            test_data=paths['test_path'],
            batch_size=batch_size,
            shuffle=False,
            data_format=data_format
        )
    
    if create_item_loader and item_feature_map is not None:
        result['item_loader'] = RankDataLoader(
            feature_map=item_feature_map,
            stage='train',  # Acts as feed generator
            train_data=paths['item_pool_path'],
            batch_size=batch_size,
            shuffle=False,
            data_format=data_format
        )
    
    return result

def prepare_shared_data_loaders(feature_map, dataset_config, pipeline_config, fg_manager, logger):
    """
    Prepare all shared data loaders for the entire pipeline (created once, reused by all stages).
    
    Args:
        feature_map: Full FeatureMap instance
        dataset_config: Dataset configuration dict
        pipeline_config: Pipeline configuration dict
        fg_manager: FeatureGroupManager instance
        logger: Logger instance
        
    Returns:
        dict with keys: paths, train_loader, test_loader, item_loader
    """
    # Get common data paths and prepare debug datasets if needed
    paths = get_data_paths(dataset_config, pipeline_config, logger)
    paths = prepare_debug_paths(paths, dataset_config, logger)
    
    # Create item feature map for item pool loader
    item_fm = _create_item_feature_map(feature_map, fg_manager, dataset_config)
    logger.info(f"Loading item pool from {paths['item_pool_path']}")
    
    # Get batch_size from retrieval config (primary stage)
    retrieval_config = pipeline_config['stages'].get('retrieval', {})
    batch_size = retrieval_config.get('training', {}).get('batch_size', 4096)
    
    # Create all loaders once
    loaders = _prepare_stage_data_loaders(
        feature_map=feature_map,
        stage_config={'training': {'batch_size': batch_size}},
        paths=paths,
        create_train=True,
        create_test=True,
        shuffle_train=True,
        create_item_loader=True,
        item_feature_map=item_fm
    )
    
    return {
        'paths': paths,
        'train_loader': loaders['train_loader'],
        'valid_loader': loaders['valid_loader'],
        'test_loader': loaders['test_loader'],
        'item_loader': loaders['item_loader'],
    }

def _create_item_feature_map(feature_map, fg_manager, dataset_config):
    """
    Create a lean feature map for item pool (FG1 features only).
    
    Args:
        feature_map: Full FeatureMap instance
        fg_manager: FeatureGroupManager instance
        dataset_config: Dataset configuration dict
        
    Returns:
        FeatureMap with only item (FG1) features
    """
    item_feature_names = [name for name, spec in feature_map.features.items()
                          if fg_manager.feature_assignments.get(name) == FeatureGroup.FG1]
    
    item_fm = copy.deepcopy(feature_map)
    item_fm.features = {k: v for k, v in item_fm.features.items() if k in item_feature_names}
    item_fm.labels = []

    item_fm.set_column_index()
    item_fm.column_index = {k: v for k, v in item_fm.column_index.items() if k in item_feature_names}
    return item_fm

def run_retrieval_stage(retrieval_stage, pipeline_config, dataset_config, fg_manager, logger=None, shared_loaders=None):
    if logger is None:
        logger = logging.getLogger('PipelineRunner')
    retrieval_config = pipeline_config['stages']['retrieval']
    metrics = {}

    # 1. Prepare data loaders (use shared if provided, otherwise create them)
    if shared_loaders is not None:
        train_loader = shared_loaders['train_loader']
        valid_loader = shared_loaders['valid_loader']
        item_loader = shared_loaders['item_loader']
        test_loader = shared_loaders['test_loader']
    else:
        # Create loaders using shared function
        loaders = prepare_shared_data_loaders(
            feature_map=retrieval_stage.feature_map,
            dataset_config=dataset_config,
            pipeline_config=pipeline_config,
            fg_manager=fg_manager,
            logger=logger
        )
        train_loader = loaders['train_loader']
        valid_loader = loaders['valid_loader']
        item_loader = loaders['item_loader']
        test_loader = loaders['test_loader']
    # 2. Train and build item index if needed
    if retrieval_stage.item_embeddings is None:
        logger.info("[Training] Building item index for retrieval stage...")
        train_gen, _ = train_loader.make_iterator()
        valid_gen = valid_loader.make_iterator()
        item_gen, _ = item_loader.make_iterator()
        retrieval_stage.build_model()
        train_metrics = retrieval_stage.train(
            train_data=train_gen,
            valid_data=valid_gen,
            item_data=item_gen,
            epochs=retrieval_config['training'].get('epochs', 10),
            patience=retrieval_config['training'].get('patience', 2),
            monitor=retrieval_config['training'].get('monitor', 'Recall@1000'),
            mode=retrieval_config['training'].get('mode', 'max')
        )
        if train_metrics:
            metrics.update({f"train_{k}": v for k, v in train_metrics.items()})

        # After training, load the best model and build item index for valid evaluation
        retrieval_stage.build_item_index(item_gen)
        logger.info("Item index built successfully after training.")

        logger.info("Evaluating on validation set after training...")
        valid_metrics = retrieval_stage.evaluate(valid_gen)
        metrics.update({f"retrieval_valid_{k}": v for k, v in valid_metrics.items()})
    else:
        logger.info("Retrieval model already trained and item index built. Skipping training.")

    # 3. Evaluate on test set
    logger.info("Evaluating on test set...")
    # Ensure item embeddings are available for test evaluation if not from training
    if retrieval_stage.item_embeddings is None:
        item_gen, _ = item_loader.make_iterator()
        retrieval_stage.build_item_index(item_gen)
        logger.info("Item index built for test evaluation.")

    # 4. Generate Candidates for Pipeline
    logger.info("Generating candidates for pipeline flow...")
    test_output, test_metrics = retrieval_stage.process(test_loader.make_iterator())
    valid_output, _ = retrieval_stage.process(valid_loader.make_iterator(), compute_metrics=False)
    metrics.update({f"retrieval_test_{k}": v for k, v in test_metrics.items()})
    return metrics, valid_output, test_output

def run_preranking_stage(preranking_stage, pipeline_config, dataset_config, logger=None, shared_loaders=None,
                         prev_output_test=None, prev_output_valid=None):
    if logger is None:
        logger = logging.getLogger('PipelineRunner')
    
    preranking_config = pipeline_config['stages']['preranking']
    metrics = {}
    
    # 1. Prepare Data Loaders (use shared if provided)
    if shared_loaders is not None:
        paths = shared_loaders['paths']
        train_gen, _ = shared_loaders['train_loader'].make_iterator()
    else:
        paths = get_data_paths(dataset_config, pipeline_config, logger)
        loaders = _prepare_stage_data_loaders(
            feature_map=preranking_stage.feature_map,
            stage_config=preranking_config,
            paths=paths
        )
        train_gen, _ = loaders['train_loader'].make_iterator()

    # Load Item Pool for Evaluation/Processing
    if os.path.exists(paths['item_pool_path']):
        preranking_stage.load_item_features(paths['item_pool_path'])
    else:
        logger.warning(f"Item pool not found at {paths['item_pool_path']}. Evaluation will lack negative features.")
    # 2. Train Preranking Model
    logger.info("[Preranking] Training model...")
    preranking_stage.build_model()
    preranking_stage.train(
        train_data=train_gen,
        valid_data=prev_output_valid,
        epochs=preranking_config['training'].get('epochs', 5),
        batch_size=preranking_config['training'].get('batch_size', 4096),
    )
    # 4. Pipeline Processing
    logger.info("[Preranking] Processing pipeline candidates...")
    test_output, test_metrics = preranking_stage.process(prev_output_test, compute_metrics=True)
    metrics.update({f"preranking_test_{k}": v for k, v in test_metrics.items()})
    valid_output, valid_metrics = preranking_stage.process(prev_output_valid, compute_metrics=True)
    metrics.update({f"preranking_valid_{k}": v for k, v in valid_metrics.items()})
    return metrics, valid_output, test_output

def run_reranking_stage(reranking_stage, pipeline_config, dataset_config, logger=None, shared_loaders=None,
                        prev_output_valid=None, prev_output_test=None):
    if logger is None:
        logger = logging.getLogger('PipelineRunner')
    
    reranking_config = pipeline_config['stages']['reranking']
    metrics = {}
    
    # 1. Prepare Data Loaders (use shared if provided)
    if shared_loaders is not None:
        paths = shared_loaders['paths']
        train_gen, _ = shared_loaders['train_loader'].make_iterator()
    else:
        paths = get_data_paths(dataset_config, pipeline_config, logger)
        loaders = _prepare_stage_data_loaders(
            feature_map=reranking_stage.feature_map,
            stage_config=reranking_config,
            paths=paths
        )
        train_gen, _ = loaders['train_loader'].make_iterator()

    # Load Item Pool for Evaluation/Processing
    if os.path.exists(paths['item_pool_path']):
        reranking_stage.load_item_features(paths['item_pool_path'])
    # 2. Train Reranking Model
    logger.info("[Reranking] Training model...")
    reranking_stage.build_model()
    reranking_stage.train(
        train_data=train_gen,
        valid_data=prev_output_valid,
        epochs=reranking_config['training'].get('epochs', 5),
        batch_size=reranking_config['training'].get('batch_size', 4096)
    )
    valid_metrics = reranking_stage.evaluate(prev_output_valid)
    metrics.update({f"reranking_valid_{k}": v for k, v in valid_metrics.items()})
    # 3. Evaluate Reranking Model (List-wise if prev_output available)
    logger.info("[Reranking] Evaluating on test set...")
    test_metrics = reranking_stage.evaluate(prev_output_test)
    logger.info(f"Test (Ranking): {test_metrics}")
    metrics.update({f"reranking_test_{k}": v for k, v in test_metrics.items()})
    return metrics

def main():
    print("DEBUG: main() started", flush=True)
    """Main entry point"""
    args = parse_pipeline_args()    
    # Construct unique output directory for this run
    run_output_base = args.output_dir # e.g. ./outputs
    logger = logging.getLogger('PipelineRunner')
    if args.experiment_id:
        logger.info(f"Experiment ID: {args.experiment_id}")
        run_output_dir = f"{run_output_base}/{args.experiment_id}"
    else:
        run_output_dir = os.path.join(run_output_base, f"{args.pipeline_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_output_dir, exist_ok=True)
    stage_output_dir = f"{run_output_dir}/stage_outputs"

    # Setup logging to the unique output directory
    setup_logging(run_output_dir)

    logger.info(f"Starting pipeline with mode: {args.mode}")
    logger.info(f"Configuration: {args.config}, Pipeline: {args.pipeline_id}")
    logger.info(f"Output Directory: {run_output_dir}")
    
    seed_everything(args.seed)
    
    # Load configurations using ConfigParser
    config_parser = ConfigParser(config_dir=args.config)
    full_config = config_parser.get_full_config(
        pipeline_path=os.path.join(args.config, f"{args.pipeline_id}.yaml") if args.pipeline_id != 'default' else None,
        dataset_path=os.path.join(args.config, 'dataset_config.yaml'),
        dataset_id=args.dataset_id
    )
    
    pipeline_config = full_config['pipeline']
    dataset_config = full_config['dataset']
    
    # Ensure dataset_id in pipeline_config is consistent
    pipeline_config['dataset_id'] = dataset_config['dataset_id'] 

    # --- Apply Retrieval Stage Overrides ---
    if 'retrieval' in pipeline_config.get('stages', {}):
        r_stage = pipeline_config['stages']['retrieval']
        r_params = r_stage.get('model_params', {})
        r_train = r_stage.get('training', {})

        # Write back
        pipeline_config['stages']['retrieval']['model_params'] = r_params
        pipeline_config['stages']['retrieval']['training'] = r_train

    # --- Save Run Configuration ---
    run_config_path = os.path.join(run_output_dir, 'run_config.yaml')
    if args.n_rows is not None:
        if 'debug' not in pipeline_config:
            pipeline_config['debug'] = {}
        pipeline_config['debug']['n_rows'] = args.n_rows
        logger.info(f"Override debug n_rows: {args.n_rows}")
        
    with open(run_config_path, 'w') as f:
        yaml.dump(full_config, f, indent=2)
    logger.info(f"Saved run configuration to {run_config_path}")
    
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

        feature_map = FeatureMap(pipeline_config['dataset_id'], data_dir)
        feature_map.load(feature_map_json, dataset_config)
        feature_map.dataset_config = dataset_config # Attach config for access in stages
        # --- Critical Fix: Ensure impression_id is loaded by DataLoaders ---
        # Add impression_id_col to labels so it gets loaded but not treated as a feature
        impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
        if impression_id_col not in feature_map.labels:
             logger.info(f"Adding '{impression_id_col}' to feature_map.labels to ensure loading.")
             feature_map.labels.append(impression_id_col)
             
        logger.info(f"Loaded feature map with {len(feature_map.features)} features")
    else:
        logger.warning(f"Feature map not found at {feature_map_json}")
        logger.info("Please run data preprocessing first")
        return

    # Build feature group manager
    fg_manager = build_feature_group_manager(pipeline_config)
    # Pass dataset_config to use explicit feature group definitions
    fg_manager.auto_assign_groups(feature_map, dataset_config)

    # Create stages
    stages = create_stages(
        feature_map=feature_map,
        feature_group_manager=fg_manager,
        config=pipeline_config,
        output_dir=run_output_dir,
        gpu=args.gpu
    )
    
    all_metrics = {}
    metrics_path = os.path.join(run_output_dir, 'metrics.json')
    # Execute based on mode
    if args.mode == 'full':
        logger.info("Running full pipeline")
        retrieval_stage = stages['retrieval']
        # Prepare shared data loaders once for all stages
        logger.info("Preparing shared data loaders...")
        shared_loaders = prepare_shared_data_loaders(
            feature_map=feature_map,
            dataset_config=dataset_config,
            pipeline_config=pipeline_config,
            fg_manager=fg_manager,
            logger=logger
        )
        r_metrics, r_valid, r_test = run_retrieval_stage(
            retrieval_stage, pipeline_config, dataset_config, fg_manager, 
            logger=logger, shared_loaders=shared_loaders
        )
        all_metrics.update(r_metrics)
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_path}")
        # Save retrieval stage outputs if requested
        if args.save_stage_outputs:
            save_stage_output(r_valid, stage_output_dir, 'retrieval_valid', logger)
            save_stage_output(r_test, stage_output_dir, 'retrieval_test', logger)
            
        preranking_stage = stages['preranking']
        p_metrics, p_valid, p_test = run_preranking_stage(
            preranking_stage, pipeline_config, dataset_config, 
            logger=logger, prev_output_valid=r_valid, prev_output_test=r_test,
            shared_loaders=shared_loaders
        )
        all_metrics.update(p_metrics)
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_path}")
        # Save preranking stage outputs if requested
        if args.save_stage_outputs:
            save_stage_output(p_valid, stage_output_dir, 'preranking_valid', logger)
            save_stage_output(p_test, stage_output_dir, 'preranking_test', logger)

        reranking_stage = stages['reranking']
        d_metrics = run_reranking_stage(
            reranking_stage, pipeline_config, dataset_config, logger=logger, shared_loaders=shared_loaders,
            prev_output_valid=p_valid, prev_output_test=p_test
        )
        all_metrics.update(d_metrics)
            
    elif args.mode == 'retrieval':
        logger.info("Running retrieval stage only")
        if 'retrieval' not in stages:
            raise RuntimeError("No retrieval stage found. Please run data preprocessing first")
        retrieval_stage = stages['retrieval']
        r_metrics, r_valid, r_test = run_retrieval_stage(
            retrieval_stage, pipeline_config, dataset_config, fg_manager, logger=logger
        )
        all_metrics.update(r_metrics)
        
        # Save stage outputs if requested
        if args.save_stage_outputs:
            stage_output_dir = os.path.join(run_output_dir, 'stage_outputs')
            save_stage_output(r_valid, stage_output_dir, 'retrieval_valid', logger)
            save_stage_output(r_test, stage_output_dir, 'retrieval_test', logger)
            
    elif args.mode == 'preranking':
        logger.info("Running preranking stage only")
        if 'preranking' not in stages:
            raise RuntimeError("No preranking stage found. Please run retrieval first")

        # Load previous stage outputs if provided (from retrieval stage)
        if args.prev_output_path:
            prev_output_valid, prev_output_test = load_stage_outputs_from_dir(
                args.prev_output_path, 'retrieval', logger
            )
        else:
            raise RuntimeError("Previous stage outputs not provided or failed to load. Preranking will run without candidate filtering from retrieval.")
        
        preranking_stage = stages['preranking']
        p_metrics, p_valid, p_test = run_preranking_stage(
            preranking_stage, pipeline_config, dataset_config, logger=logger,
            prev_output_valid=prev_output_valid, prev_output_test=prev_output_test
        )
        all_metrics.update(p_metrics)
        
        # Save stage outputs if requested
        if args.save_stage_outputs:
            save_stage_output(p_valid, stage_output_dir, 'preranking_valid', logger)
            save_stage_output(p_test, stage_output_dir, 'preranking_test', logger)
            
    elif args.mode == 'reranking':
        logger.info("Running reranking stage only")
        if 'reranking' not in stages:
            raise RuntimeError("No reranking stage found. Please run preranking first")

        # Load previous stage outputs if provided (from preranking stage)
        if args.prev_output_path:
            prev_output_valid, prev_output_test = load_stage_outputs_from_dir(
                args.prev_output_path, 'preranking', logger
            )
        else:
            raise RuntimeError("Previous stage outputs not provided or failed to load. "
                          "Reranking will run without candidate filtering from preranking.")

        reranking_stage = stages['reranking']
        d_metrics = run_reranking_stage(
            reranking_stage, pipeline_config, dataset_config, logger=logger,
            prev_output_valid=prev_output_valid, prev_output_test=prev_output_test
        )
        all_metrics.update(d_metrics)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Save final metrics
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    logger.info(f"Saved metrics to {metrics_path}")

if __name__ == '__main__':
    main()
