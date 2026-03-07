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
from cloud_device_recsys.pipeline import RetrievalStage, PrerankingStage, RerankingStage, DTCNPrerankingStage
from cloud_device_recsys.utils import (
    setup_logging, get_data_dir, get_data_paths,
    prepare_debug_paths, parse_pipeline_args,
    save_stage_output, load_stage_outputs_from_dir,
    enrich_stage_output_user_features
)
from cloud_device_recsys.data.item_pool import ensure_item_pool, ensure_full_item_pool
from cloud_device_recsys.data.positive_data import get_train_path_for_mode
from cloud_device_recsys.config.config_parser import ConfigParser
import pandas as pd

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
            'kwargs_map': {
                'features': 'allowed_feature_groups',
                'top_k': 'top_k'
            }
        },
        'preranking': {
            'class': PrerankingStage,
            'kwargs_map': {
                'features': 'allowed_feature_groups',
                'top_k': 'top_k',
                'use_diversity': 'use_diversity'
            }
        },
        'reranking': {
            'class': RerankingStage,
            'kwargs_map': {
                'features': 'allowed_feature_groups',
                'top_k': 'top_k',
                'distillation': 'support_distillation'  # config 'distillation' maps to class 'support_distillation'
            }
        },
    }

    for stage_name, stage_info in STAGE_REGISTRY.items():
        if stage_name in stages_config and stages_config[stage_name].get('enabled', False):
            stage_config = stages_config[stage_name]

            # Prepare model_params
            model_params = stage_config.get('model_params', {}).copy()  # Use .copy() to avoid modifying original config
            model_params['gpu'] = gpu
            model_params['model'] = stage_config.get('model')  # Pass model name from config
            if 'metrics' in stage_config:
                model_params['metrics'] = stage_config['metrics']

            # Prepare stage-specific keyword arguments for the constructor
            stage_kwargs = {
                'feature_map': copy.deepcopy(feature_map),
                'feature_group_manager': feature_group_manager,
                'model_params': model_params,
                'output_dir': os.path.join(output_dir, stage_name),
            }

            # Add extra parameters dynamically
            for config_key, class_param in stage_info['kwargs_map'].items():
                if config_key in stage_config:
                    if config_key == 'features':  # Special handling for 'features' to map to enum
                        stage_kwargs[class_param] = [getattr(FeatureGroup, f) for f in stage_config['features']]
                    elif config_key == 'distillation':  # Special handling for 'distillation' to extract 'enabled'
                        stage_kwargs[class_param] = stage_config[config_key].get('enabled', False)
                    else:
                        stage_kwargs[class_param] = stage_config[config_key]

            # Lazy Instantiation: Return a factory method
            # Use default arguments to capture loop variables (cls, kwargs) correctly!
            stages[stage_name] = lambda cls=stage_info['class'], kwargs=stage_kwargs: cls(**kwargs)

    return stages

def _prepare_stage_data_loaders(feature_map, stage_config: dict, paths: dict,
                                create_train=True, create_test=True, shuffle_train=True,
                                create_item_loader=False, item_feature_map=None,
                                num_negatives: int = 0, label_col: str = "label",
                                logger=None):
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
        num_negatives: Number of negatives per positive (0 = pointwise, >0 = pairwise)
        label_col: Label column name for filtering positive samples
        logger: Logger instance for positive data creation

    Returns:
        dict with keys: train_loader, test_loader, item_loader (based on flags)
    """
    batch_size = stage_config.get('training', stage_config).get('batch_size', 4096)
    data_format = paths['data_format']

    result = {}

    if create_train:
        # Select training data path based on training mode
        train_path = get_train_path_for_mode(
            train_path=paths['train_path'],
            num_negatives=num_negatives,
            label_col=label_col,
            logger=logger
        )

        # train_fm = copy.deepcopy(feature_map)
        # if "impression_id" in train_fm.labels:
        #     train_fm.labels.remove("impression_id")
        result['train_loader'] = RankDataLoader(
            feature_map=feature_map,
            stage='train',
            train_data=train_path,
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

def _create_item_feature_map(feature_map, fg_manager):
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


def run_retrieval_stage(retrieval_stage, pipeline_config, dataset_config, fg_manager, logger=None, run_test=True):
    if logger is None:
        logger = logging.getLogger('PipelineRunner')
    retrieval_config = pipeline_config['stages']['retrieval']
    metrics = {}

    # 1. Prepare data loaders for retrieval stage
    paths = get_data_paths(dataset_config, pipeline_config, logger)
    paths = prepare_debug_paths(paths, dataset_config, logger)

    # Ensure item pool exists (generate if missing)
    ensure_item_pool(
        data_paths={'item_pool_path': paths['item_pool_path'], 'test_path': paths['test_path'],
                    'valid_path': paths['valid_path']},
        dataset_config=dataset_config,
        feature_group_manager=fg_manager,
        logger=logger
    )

    # Create item feature map for item pool loader
    item_fm = _create_item_feature_map(retrieval_stage.feature_map, fg_manager)
    logger.info(f"Loading item pool from {paths['item_pool_path']}")

    # Get training mode config
    num_negatives = retrieval_config.get('model_params', {}).get('num_negatives', 0)
    label_col = dataset_config.get('label_col', {}).get('name', 'label')

    # Debug: Verify feature map size
    logger.info(f"[Retrieval] Using feature map with {len(retrieval_stage.feature_map.features)} features.")
    debug_features = sorted(list(retrieval_stage.feature_map.features.keys()))
    logger.info(f"[Retrieval] Features: {debug_features[:10]} ... (Total {len(debug_features)})")
    
    loaders = _prepare_stage_data_loaders(
        feature_map=retrieval_stage.feature_map,
        stage_config=retrieval_config,
        paths=paths,
        create_train=True,
        create_test=run_test,
        shuffle_train=True,
        create_item_loader=True,
        item_feature_map=item_fm,
        num_negatives=num_negatives,
        label_col=label_col,
        logger=logger
    )
    train_loader = loaders['train_loader']
    valid_loader = loaders['valid_loader']
    item_loader = loaders['item_loader']
    test_loader = loaders['test_loader'] if run_test else None

    # 2. Train and build item index if needed
    if retrieval_stage.item_embeddings is None:
        logger.info("[Training] Building item index for retrieval stage...")
        train_gen, _ = train_loader.make_iterator()
        valid_gen = valid_loader.make_iterator()
        item_gen, _ = item_loader.make_iterator()
        retrieval_stage.build_model()

        # Load item features for negative sampling if enabled
        # Use FULL item pool (train+valid+test) for more representative negative sampling
        item_features_df = None
        num_negatives = retrieval_stage.num_negatives if hasattr(retrieval_stage, 'num_negatives') else 0
        if num_negatives > 0:
            # Generate full item pool from train+valid+test if not exists
            full_pool_path = ensure_full_item_pool(
                data_paths=paths,
                dataset_config=dataset_config,
                feature_group_manager=fg_manager,
                logger=logger
            )
            if full_pool_path and os.path.exists(full_pool_path):
                logger.info(f"Loading FULL item pool for negative sampling from {full_pool_path}")
                item_features_df = pd.read_parquet(full_pool_path)
                logger.info(f"Loaded {len(item_features_df)} items for negative sampling (full pool)")
            else:
                logger.warning("Full item pool not found. Negative sampling will be disabled.")

        retrieval_stage.train(
            train_data=train_gen,
            valid_data=valid_gen,
            item_data=item_gen,
            item_features_df=item_features_df,
            epochs=retrieval_config['training'].get('epochs', 10),
            patience=retrieval_config['training'].get('patience', 2),
            monitor=retrieval_config['training'].get('monitor', 'Recall@1000'),
            mode=retrieval_config['training'].get('mode', 'max')
        )

        # After training, load the best model and build item index for valid evaluation
        retrieval_stage.build_item_index(item_gen)
        logger.info("Item index built successfully after training.")
    else:
        logger.info("Retrieval model already trained and item index built. Skipping training.")

    # Ensure item embeddings are available for test evaluation if not from training
    if retrieval_stage.item_embeddings is None:
        item_gen, _ = item_loader.make_iterator()
        retrieval_stage.build_item_index(item_gen)
        logger.info("Item index built for test evaluation.")
    logger.info("Evaluating on valid set and generating candidate sets")
    valid_output, valid_metrics = retrieval_stage.process(valid_loader.make_iterator(), compute_metrics=True)
    metrics.update({f"retrieval_valid_{k}": v for k, v in valid_metrics.items()})
    # 3. Evaluate on test set
    test_output = None
    if run_test:
        logger.info("Evaluating on test set and generating candidate sets")
        test_output, test_metrics = retrieval_stage.process(test_loader.make_iterator())
        metrics.update({f"retrieval_test_{k}": v for k, v in test_metrics.items()})
    else:
        logger.info("Skipping retrieval test evaluation as requested.")
    
    return metrics, valid_output, test_output

def run_preranking_stage(preranking_stage, pipeline_config, dataset_config, fg_manager=None, logger=None,
                         prev_output_test=None, prev_output_valid=None, run_test=True, **kwargs):
    if logger is None:
        logger = logging.getLogger('PipelineRunner')

    preranking_config = pipeline_config['stages']['preranking']
    metrics = {}

    # 1. Prepare Data Loaders for preranking stage
    paths = get_data_paths(dataset_config, pipeline_config, logger)
    paths = prepare_debug_paths(paths, dataset_config, logger)

    # Create data loaders for preranking stage
    num_negatives = preranking_config.get('model_params', {}).get('num_negatives', 0)
    loaders = _prepare_stage_data_loaders(
        feature_map=preranking_stage.feature_map,
        stage_config=preranking_config,
        paths=paths,
        create_train=True,
        create_test=False,
        num_negatives=num_negatives,
        label_col=dataset_config.get('label_col', {}).get('name', 'label'),
        logger=logger
    )
    train_gen, _ = loaders['train_loader'].make_iterator()
    cand_item_pool_path = ensure_item_pool(
        data_paths={'item_pool_path': paths['item_pool_path'], 'valid_path': paths['valid_path'],
                    'test_path': paths['test_path']},
        dataset_config=dataset_config,
        feature_group_manager=fg_manager,
        logger=logger
    )
    # Load item pool for training (negative sampling) vs evaluation
    if num_negatives > 0 and fg_manager is not None:
        neg_pool_mode = preranking_config.get('neg_sampling_pool', 'candidate')
        if neg_pool_mode == 'full':
            neg_pool_path = ensure_full_item_pool(
                data_paths=paths,
                dataset_config=dataset_config,
                feature_group_manager=fg_manager,
                logger=logger
            )
        else:
            neg_pool_path = cand_item_pool_path
        # Use item pool (train+valid+test) for negative sampling during training
        preranking_stage.load_item_features(neg_pool_path)
        logger.info(f"[Preranking] Negative sampling pool: '{neg_pool_mode}' → {neg_pool_path} ({len(preranking_stage.item_features_df)} items) for negative sampling")
    else:
        # Pointwise mode: load test/valid-only pool directly for evaluation
        if os.path.exists(paths['item_pool_path']):
            preranking_stage.load_item_features(paths['item_pool_path'])
        else:
            logger.warning(f"Item pool not found at {paths['item_pool_path']}. Evaluation will lack negative features.")
    # Enrich stage outputs with missing FG3 user features (backward compatibility)
    if fg_manager is not None:
        impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
        prev_output_valid = enrich_stage_output_user_features(
            prev_output_valid, paths['valid_path'], fg_manager, impression_id_col, logger
        )
        prev_output_test = enrich_stage_output_user_features(
            prev_output_test, paths['test_path'], fg_manager, impression_id_col, logger
        )
    # 2. Train Preranking Model
    logger.info("[Preranking] Training model...")
    preranking_stage.build_model()
    preranking_stage.train(
        train_data=train_gen,
        valid_data=prev_output_valid,
        epochs=preranking_config['training'].get('epochs', 5),
        batch_size=preranking_config['training'].get('batch_size', 4096),
    )

    # After training, switch to test/valid-only item pool for evaluation/processing
    if num_negatives > 0 and os.path.exists(cand_item_pool_path):
        preranking_stage.load_item_features(cand_item_pool_path)
        logger.info(f"[Preranking] Switched to eval item pool '{cand_item_pool_path}' → {cand_item_pool_path} ({len(preranking_stage.item_features_df)} items) for negative sampling")

    # 3. Pipeline Processing
    logger.info("[Preranking] Processing pipeline candidates...")
    
    test_output = None
    if run_test:
        if prev_output_test is None:
            logger.warning("[Preranking] No previous test output provided, cannot run test evaluation.")
        else:
            test_output, test_metrics = preranking_stage.process(prev_output_test, compute_metrics=True)
            metrics.update({f"preranking_test_{k}": v for k, v in test_metrics.items()})
    else:
        logger.info("Skipping preranking test evaluation as requested.")
        
    valid_output, valid_metrics = preranking_stage.process(prev_output_valid, compute_metrics=True)
    metrics.update({f"preranking_valid_{k}": v for k, v in valid_metrics.items()})
    return metrics, valid_output, test_output

def run_joint_training_stage(preranking_stage, reranking_stage, pipeline_config, dataset_config, fg_manager, logger=None,
                             run_test=True, prev_output_test=None, prev_output_valid=None):
    """Orchestrate joint preranking + reranking training.

    Delegates all training logic to CloudDeviceJointTrainingStage, which holds
    references to the stage objects and calls their existing train() methods.
    """
    if logger is None:
        logger = logging.getLogger('PipelineRunner')

    import copy
    from cloud_device_recsys.pipeline import CloudDeviceJointTrainingStage

    preranking_config = pipeline_config['stages']['preranking']
    reranking_config  = pipeline_config['stages']['reranking']
    joint_params = pipeline_config.get('joint_training', {})

    # 1. Prepare data paths
    paths = get_data_paths(dataset_config, pipeline_config, logger)
    paths = prepare_debug_paths(paths, dataset_config, logger)

    # Per-stage num_negatives: read from each stage's model_params
    # (joint_params.num_negatives acts as fallback if not set per-stage)
    _jt_nn = joint_params.get('num_negatives', None)
    pre_num_negatives = preranking_config.get('model_params', {}).get('num_negatives', _jt_nn or 0)
    re_num_negatives  = reranking_config.get('model_params', {}).get('num_negatives', _jt_nn or 0)
    logger.info(f"[Joint Training] num_negatives: preranking={pre_num_negatives}, reranking={re_num_negatives}")

    label_col = dataset_config.get('label_col', {}).get('name', 'label')

    # 1a. Preranking train loader — uses preranking feature_map + config (correct batch_size)
    pre_loader_fm = copy.deepcopy(preranking_stage.feature_map)
    pre_loaders = _prepare_stage_data_loaders(
        feature_map=pre_loader_fm,
        stage_config=preranking_config,
        paths=paths,
        create_train=True,
        create_test=False,
        num_negatives=pre_num_negatives,
        label_col=label_col,
        logger=logger,
    )
    preranking_train_loader = pre_loaders['train_loader']

    # 1b. Reranking train loader — uses reranking feature_map + config (correct batch_size)
    re_loader_fm = copy.deepcopy(reranking_stage.feature_map)
    re_loaders = _prepare_stage_data_loaders(
        feature_map=re_loader_fm,
        stage_config=reranking_config,
        paths=paths,
        create_train=True,
        create_test=False,
        num_negatives=re_num_negatives,
        label_col=label_col,
        logger=logger,
    )
    reranking_train_loader = re_loaders['train_loader']

    # 2. Ensure item pool exists
    if fg_manager is not None:
        ensure_item_pool(
            data_paths={'item_pool_path': paths['item_pool_path'],
                        'valid_path': paths['valid_path'],
                        'test_path': paths['test_path']},
            dataset_config=dataset_config,
            feature_group_manager=fg_manager,
            logger=logger,
        )
        impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
        prev_output_valid = enrich_stage_output_user_features(
            prev_output_valid, paths['valid_path'], fg_manager, impression_id_col, logger
        )
        prev_output_test = enrich_stage_output_user_features(
            prev_output_test, paths['test_path'], fg_manager, impression_id_col, logger
        )
    # 3. Build individual stage models
    preranking_stage.build_model()
    preranking_stage.best_weights_path = os.path.join(
        preranking_stage.model.model_dir, preranking_stage.model.model_id + ".model"
    )
    reranking_stage.build_model()
    reranking_stage.best_weights_path = os.path.join(
        reranking_stage.model.model_dir, reranking_stage.model.model_id + ".model"
    )

    # 4. Load item features for negative sampling
    # neg_sampling_pool: 'full' → cand_items_all.parquet (800K+ items, like standalone preranking)
    #                    'candidate' → item_pool_path (~28K, default)
    neg_pool_mode = joint_params.get('neg_sampling_pool', 'candidate')
    if neg_pool_mode == 'full' and os.path.exists(paths['full_item_pool_path']):
        neg_pool_path = paths['full_item_pool_path']
    else:
        neg_pool_path = paths['item_pool_path']
        if neg_pool_mode == 'full':
            logger.warning("[Joint Training] full_item_pool_path not found, falling back to candidate pool.")
    logger.info(f"[Joint Training] Negative sampling pool: '{neg_pool_mode}' → {neg_pool_path}")
    max_neg_nn = max(pre_num_negatives, re_num_negatives)
    if max_neg_nn > 0 and os.path.exists(neg_pool_path):
        preranking_stage.load_item_features(neg_pool_path)
        # Share item_features_df to reranking — avoids a redundant file load
        reranking_stage.item_features_df = preranking_stage.item_features_df
        logger.info("[Joint Training] Shared item_features_df from preranking to reranking stage.")
    neg_pool_path_for_params = neg_pool_path

    # 5. Merge per-stage num_negatives and pool info for the orchestrator
    merged_joint_params = dict(joint_params)
    merged_joint_params['pre_num_negatives'] = pre_num_negatives
    merged_joint_params['re_num_negatives']  = re_num_negatives
    merged_joint_params['_neg_pool_path']    = neg_pool_path_for_params

    # 6. Create orchestrator and train
    logger.info("[Joint Training] Initializing CloudDeviceJointTrainingStage...")
    joint = CloudDeviceJointTrainingStage(
        preranking_stage=preranking_stage,
        reranking_stage=reranking_stage,
        joint_params=merged_joint_params,
        preranking_config=preranking_config,
        reranking_config=reranking_config,
        preranking_train_loader=preranking_train_loader,
        reranking_train_loader=reranking_train_loader,
        fg_manager=fg_manager,
        dataset_config=dataset_config,
        paths=paths,
        logger=logger,
    )
    metrics = joint.train(prev_output_valid, prev_output_test, run_test=run_test)
    return metrics


def run_dtcn_preranking_stage(pipeline_config, dataset_config, fg_manager, feature_map,
                              logger=None, run_test=True,
                              prev_output_test=None, prev_output_valid=None, gpu=-1, **kwargs):
    """
    Run the DTCN preranking stage: dual-model (cloud + full) training with CL.

    Creates two DataLoaders:
      - Cloud DataLoader: FG1+FG2 features
      - Full DataLoader:  FG1+FG2+FG3 features
    Then creates a DTCNPrerankingStage and delegates training/evaluation.
    """
    if logger is None:
        logger = logging.getLogger('PipelineRunner')

    from cloud_device_recsys.utils import filter_feature_map

    preranking_config = pipeline_config['stages']['preranking']
    dtcn_config = pipeline_config['stages'].get('dtcn', {})
    metrics = {}

    # 1. Prepare Data Paths
    paths = get_data_paths(dataset_config, pipeline_config, logger)
    paths = prepare_debug_paths(paths, dataset_config, logger)

    num_negatives = preranking_config.get('model_params', {}).get('num_negatives', 0)
    label_col = dataset_config.get('label_col', {}).get('name', 'label')

    # 2. Build feature maps
    #    Cloud feature map: FG1 + FG2 (same as normal preranking)
    cloud_features = [FeatureGroup.from_string(f) for f in preranking_config.get('features', ['FG1', 'FG2'])]
    cloud_feature_map = filter_feature_map(
        feature_map, fg_manager, cloud_features,
        use_feature_encoder=preranking_config.get('model_params', {}).get('use_feature_encoder', False)
    )
    cloud_feature_map.default_emb_dim = preranking_config.get('model_params', {}).get('embedding_dim', 16)

    #    Full feature map: FG1 + FG2 + FG3
    full_features_list = [FeatureGroup.from_string(f) for f in dtcn_config.get('full_features', ['FG1', 'FG2', 'FG3'])]
    full_feature_map = filter_feature_map(
        feature_map, fg_manager, full_features_list,
        use_feature_encoder=dtcn_config.get('full_model_params', {}).get('use_feature_encoder',
                                            preranking_config.get('model_params', {}).get('use_feature_encoder', False))
    )
    full_feature_map.default_emb_dim = dtcn_config.get('full_model_params', {}).get(
        'embedding_dim', preranking_config.get('model_params', {}).get('embedding_dim', 16)
    )

    logger.info(f"[DTCN] Cloud feature map: {len(cloud_feature_map.features)} features")
    logger.info(f"[DTCN] Full feature map: {len(full_feature_map.features)} features")

    # 3. Create DataLoaders
    #    Cloud DataLoader (FG1+FG2)
    cloud_loaders = _prepare_stage_data_loaders(
        feature_map=cloud_feature_map,
        stage_config=preranking_config,
        paths=paths,
        create_train=True,
        create_test=False,
        num_negatives=num_negatives,
        label_col=label_col,
        logger=logger
    )

    #    Full DataLoader (FG1+FG2+FG3)
    full_loaders = _prepare_stage_data_loaders(
        feature_map=full_feature_map,
        stage_config=preranking_config,
        paths=paths,
        create_train=True,
        create_test=False,
        num_negatives=num_negatives,
        label_col=label_col,
        logger=logger
    )

    # 4. Ensure item pool exists and load it
    cand_item_pool_path = ensure_item_pool(
        data_paths={'item_pool_path': paths['item_pool_path'], 'valid_path': paths['valid_path'],
                    'test_path': paths['test_path']},
        dataset_config=dataset_config,
        feature_group_manager=fg_manager,
        logger=logger
    )

    # 5. Build model params
    model_params = preranking_config.get('model_params', {}).copy()
    model_params['gpu'] = gpu
    model_params['model'] = preranking_config.get('model', 'DINRanker')
    if 'metrics' in preranking_config:
        model_params['metrics'] = preranking_config['metrics']

    # 6. Create DTCNPrerankingStage
    output_dir = kwargs.get('output_dir', './outputs/dtcn_preranking')
    dtcn_stage = DTCNPrerankingStage(
        feature_map=cloud_feature_map,
        full_feature_map=full_feature_map,
        feature_group_manager=fg_manager,
        model_params=model_params,
        dtcn_params=dtcn_config,
        allowed_feature_groups=cloud_features,
        output_dir=os.path.join(output_dir, 'preranking'),
        top_k=preranking_config.get('top_k', 100),
    )

    # 7. Load item features for negative sampling
    if num_negatives > 0 and fg_manager is not None:
        neg_pool_mode = preranking_config.get('neg_sampling_pool', 'candidate')
        if neg_pool_mode == 'full':
            neg_pool_path = ensure_full_item_pool(
                data_paths=paths,
                dataset_config=dataset_config,
                feature_group_manager=fg_manager,
                logger=logger
            )
        else:
            neg_pool_path = cand_item_pool_path
        dtcn_stage.load_item_features(neg_pool_path)
        logger.info(f"[DTCN] Negative sampling pool: '{neg_pool_mode}' -> {neg_pool_path} ({len(dtcn_stage.item_features_df)} items)")
    else:
        if os.path.exists(paths['item_pool_path']):
            dtcn_stage.load_item_features(paths['item_pool_path'])
        else:
            logger.warning(f"Item pool not found at {paths['item_pool_path']}.")

    # 8. Enrich stage outputs with FG3 user features
    if fg_manager is not None:
        impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
        prev_output_valid = enrich_stage_output_user_features(
            prev_output_valid, paths['valid_path'], fg_manager, impression_id_col, logger
        )
        prev_output_test = enrich_stage_output_user_features(
            prev_output_test, paths['test_path'], fg_manager, impression_id_col, logger
        )

    # 9. Train DTCN models
    logger.info("[DTCN] Building models...")
    dtcn_stage.build_model()

    cloud_train_gen, _ = cloud_loaders['train_loader'].make_iterator()
    full_train_gen, _ = full_loaders['train_loader'].make_iterator()

    dtcn_stage.train(
        train_data=cloud_train_gen,
        full_train_data=full_train_gen,
        valid_data=prev_output_valid,
        epochs=preranking_config['training'].get('epochs', 10),
        batch_size=preranking_config['training'].get('batch_size', 4096),
    )

    # 10. After training, switch to candidate item pool for evaluation
    if num_negatives > 0 and os.path.exists(cand_item_pool_path):
        dtcn_stage.load_item_features(cand_item_pool_path)
        logger.info(f"[DTCN] Switched to eval item pool ({len(dtcn_stage.item_features_df)} items)")

    # 11. Evaluate and process
    logger.info("[DTCN] Processing pipeline candidates (cloud model only)...")
    valid_output, valid_metrics = dtcn_stage.process(prev_output_valid, compute_metrics=True)
    metrics.update({f"preranking_valid_{k}": v for k, v in valid_metrics.items()})

    test_output = None
    if run_test:
        if prev_output_test is None:
            logger.warning("[DTCN] No previous test output provided, cannot run test evaluation.")
        else:
            test_output, test_metrics = dtcn_stage.process(prev_output_test, compute_metrics=True)
            metrics.update({f"preranking_test_{k}": v for k, v in test_metrics.items()})
    else:
        logger.info("Skipping DTCN preranking test evaluation.")

    return metrics, valid_output, test_output


def run_reranking_stage(reranking_stage, pipeline_config, dataset_config, fg_manager=None, run_test=True, logger=None,
                        prev_output_path=None, preranking_model=None, prev_output_test=None, prev_output_valid=None,
                        retrieval_output_valid=None, retrieval_output_test=None,
                        feature_map=None):
    """Run reranking stage with optional cloud teacher and evaluation scope alignment.

    Args:
        reranking_stage: RerankingStage instance (lazily instantiated)
        pipeline_config: Full pipeline configuration dict
        dataset_config: Dataset configuration dict
        fg_manager: FeatureGroupManager instance
        run_test: Whether to evaluate on test set
        logger: Logger instance
        prev_output_path: Path to load previous stage outputs from disk
        preranking_model: Pre-ranking model for legacy cloud_score injection
        prev_output_test: Preranking test output (StageOutput)
        prev_output_valid: Preranking valid output (StageOutput)
        retrieval_output_valid: Optional retrieval valid output for eval scope alignment.
            When provided, AUC/gAUC/MRR are computed on this larger pool (~1000 candidates)
            instead of the preranking subset (~100), aligning with preranking evaluation.
        retrieval_output_test: Optional retrieval test output for eval scope alignment.
        feature_map: Base FeatureMap (needed for cloud_teacher feature map construction)
    """
    if logger is None:
        logger = logging.getLogger('PipelineRunner')
    if prev_output_path is not None:
        prev_output_valid, prev_output_test = load_stage_outputs_from_dir(prev_output_path, "preranking", logger)
    reranking_config = pipeline_config['stages']['reranking']
    metrics = {}

    # 1. Parse cloud_teacher config and build teacher feature map
    cloud_teacher_config = pipeline_config['stages'].get('cloud_teacher', {})
    cloud_teacher_feature_map = None

    if cloud_teacher_config.get('mode') and feature_map is not None and fg_manager is not None:
        from cloud_device_recsys.utils import filter_feature_map
        teacher_features_str = cloud_teacher_config.get('cloud_teacher_features', ['FG1', 'FG2'])
        teacher_feature_groups = [FeatureGroup.from_string(f) for f in teacher_features_str]
        cloud_teacher_feature_map = filter_feature_map(
            feature_map, fg_manager, teacher_feature_groups,
            use_feature_encoder=cloud_teacher_config.get('cloud_teacher_model_params', {}).get(
                'use_feature_encoder',
                reranking_config.get('model_params', {}).get('use_feature_encoder', False)
            )
        )
        cloud_teacher_feature_map.default_emb_dim = cloud_teacher_config.get(
            'cloud_teacher_model_params', {}).get(
            'embedding_dim',
            reranking_config.get('model_params', {}).get('embedding_dim', 16)
        )
        logger.info(f"[Reranking] Cloud teacher feature map: {len(cloud_teacher_feature_map.features)} features "
                     f"({teacher_features_str})")

        # Inject into reranking_stage
        reranking_stage.cloud_teacher_params = cloud_teacher_config
        reranking_stage.cloud_teacher_feature_map = cloud_teacher_feature_map
        reranking_stage.cloud_teacher_mode = cloud_teacher_config.get('mode')

        # Update cloud_score injection flag based on mode
        if reranking_stage.cloud_teacher_mode == 'inject':
            reranking_stage.use_cloud_score = True
        elif reranking_stage.cloud_teacher_mode == 'distill':
            reranking_stage.use_cloud_score = False
            reranking_stage.kd_loss_weight = cloud_teacher_config.get('kd_loss_weight', 0.1)
            reranking_stage.kd_loss_type = cloud_teacher_config.get('kd_loss_type', 'mse')
            reranking_stage.kd_temperature = cloud_teacher_config.get('kd_temperature', 1.0)

    # 2. Prepare Data Loaders for reranking stage
    paths = get_data_paths(dataset_config, pipeline_config, logger)
    paths = prepare_debug_paths(paths, dataset_config, logger)

    # Ensure item pool exists (test/valid only - for evaluation)
    if fg_manager is not None:
        ensure_item_pool(
            data_paths={'item_pool_path': paths['item_pool_path'], 'valid_path': paths['valid_path'],
                        'test_path': paths['test_path']},
            dataset_config=dataset_config,
            feature_group_manager=fg_manager,
            logger=logger
        )

    # Create data loaders for reranking stage
    # Use deepcopy of feature_map so DataLoader doesn't see cloud_score
    # (cloud_score is registered in build_model() but doesn't exist in parquet)
    import copy
    loader_feature_map = copy.deepcopy(reranking_stage.feature_map)
    num_negatives = reranking_config.get('model_params', {}).get('num_negatives', 0)
    loaders = _prepare_stage_data_loaders(
        feature_map=loader_feature_map,
        stage_config=reranking_config,
        paths=paths,
        create_train=True,
        create_test=False,
        num_negatives=num_negatives,
        label_col=dataset_config.get('label_col', {}).get('name', 'label'),
        logger=logger
    )
    train_gen, _ = loaders['train_loader'].make_iterator()

    # Load item pool for training (negative sampling) vs evaluation
    if num_negatives > 0 and fg_manager is not None:
        neg_pool_mode = reranking_config.get('neg_sampling_pool', 'candidate')
        if neg_pool_mode == 'full':
            neg_pool_path = ensure_full_item_pool(
            data_paths=paths,
            dataset_config=dataset_config,
            feature_group_manager=fg_manager,
            logger=logger
            )
        else:
            neg_pool_path = paths['item_pool_path']
        reranking_stage.load_item_features(neg_pool_path)
        logger.info(f"[Reranking] Negative sampling pool: '{neg_pool_mode}' → {neg_pool_path} ({len(reranking_stage.item_features_df)} items) for negative sampling")
    else:
        # Pointwise mode: load test/valid-only pool directly for evaluation
        if os.path.exists(paths['item_pool_path']):
            reranking_stage.load_item_features(paths['item_pool_path'])

    # Enrich stage outputs with missing FG3 user features (backward compatibility)
    if fg_manager is not None:
        impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
        prev_output_valid = enrich_stage_output_user_features(
            prev_output_valid, paths['valid_path'], fg_manager, impression_id_col, logger
        )
        prev_output_test = enrich_stage_output_user_features(
            prev_output_test, paths['test_path'], fg_manager, impression_id_col, logger
        )

    # 3. Train Reranking Model
    logger.info("[Reranking] Training model...")
    
    # Set cloud score teacher if enabled (legacy path — only if no cloud_teacher config)
    if not cloud_teacher_config.get('mode'):
        use_cloud_score = reranking_config.get('model_params', {}).get('use_cloud_score', False)
        if use_cloud_score and preranking_model is not None:
            reranking_stage.set_cloud_score_teacher(preranking_model)
        elif use_cloud_score and preranking_model is None:
            logger.warning("[Reranking] use_cloud_score=True but no preranking model is loaded. Cloud score disabled.")
    
    reranking_stage.build_model()
    reranking_stage.train(
        train_data=train_gen,
        valid_data=prev_output_valid,
        epochs=reranking_config['training'].get('epochs', 5),
        batch_size=reranking_config['training'].get('batch_size', 4096)
    )

    # After training, switch to test/valid-only item pool for evaluation/processing
    if num_negatives > 0 and os.path.exists(paths['item_pool_path']):
        reranking_stage.load_item_features(paths['item_pool_path'])
        logger.info(f"[Reranking] Switched to eval item pool ({len(reranking_stage.item_features_df)} items) for processing")

    # 4. Evaluate with optional evaluation scope alignment
    valid_metrics = reranking_stage.evaluate(
        prev_output_valid,
        retrieval_output=retrieval_output_valid,
    )
    metrics.update({f"reranking_valid_{k}": v for k, v in valid_metrics.items()})

    if run_test:
        if prev_output_test is None:
             logger.warning("[Reranking] No previous test output provided, cannot run test evaluation.")
        else:
            logger.info("[Reranking] Evaluating on test set...")
            test_metrics = reranking_stage.evaluate(
                prev_output_test,
                retrieval_output=retrieval_output_test,
            )
            logger.info(f"Test (Ranking): {test_metrics}")
            metrics.update({f"reranking_test_{k}": v for k, v in test_metrics.items()})
    else:
        logger.info("Skipping reranking test evaluation as requested.")
        
    return metrics


def main():
    print("DEBUG: main() started", flush=True)
    """Main entry point"""
    args = parse_pipeline_args()
    # Construct unique output directory for this run
    run_output_base = args.output_dir  # e.g. ./outputs
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
    logger.info(f"Used GPU: {args.gpu if args.gpu >= 0 else 'CPU'}")

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
        feature_map.dataset_config = dataset_config  # Attach config for access in stages
        # --- Critical Fix: Ensure impression_id is loaded by DataLoaders ---
        # Add impression_id_col to labels so it gets loaded but not treated as a feature
        impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
        if impression_id_col not in feature_map.labels:
            logger.info(f"Adding '{impression_id_col}' to feature_map.labels to ensure loading.")
            feature_map.labels.append(impression_id_col)

        logger.info(f"Loaded feature map with {len(feature_map.features)} features")

        # --- Vocabulary Pruning (optional) ---
        # Reduces embedding size by scanning data for actually-used feature values.
        # Scans train + valid + test to ensure no feature value is lost during evaluation.
        vocab_pruning_config = pipeline_config.get('vocab_pruning', {})
        vocab_pruning_mode = vocab_pruning_config.get('mode', 'runtime')  # 'runtime' or 'offline'
        save_fp16 = vocab_pruning_config.get('save_fp16', False)

        if vocab_pruning_config.get('enabled', False) and vocab_pruning_mode == 'offline':
            # Offline mode: use pre-mapped data from mapped/ subdirectory
            mapped_dir = os.path.join(data_dir, 'mapped')
            mapped_feature_map_json = os.path.join(mapped_dir, 'feature_map.json')
            if os.path.exists(mapped_feature_map_json):
                logger.info(f"[VocabPruner] Offline mode: using pre-mapped data from {mapped_dir}")
                # Reload feature_map from mapped directory with compact vocab sizes
                data_dir = mapped_dir
                feature_map = FeatureMap(pipeline_config['dataset_id'], mapped_dir)
                feature_map.load(mapped_feature_map_json, dataset_config)
                feature_map.dataset_config = dataset_config
                impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
                if impression_id_col not in feature_map.labels:
                    feature_map.labels.append(impression_id_col)
                # No runtime remapping needed — data is already compact
                feature_map._vocab_prune_info = None
                # Override processed_data_root so all downstream get_data_paths() use mapped dir
                dataset_config['processed_data_root'] = mapped_dir
                logger.info(f"[VocabPruner] Offline mode: loaded mapped feature_map with {len(feature_map.features)} features")
            else:
                logger.warning(f"[VocabPruner] Offline mode requested but {mapped_feature_map_json} not found. "
                               f"Run `python -m cloud_device_recsys.data.remap_vocab_data --data_dir {data_dir}` first.")
                feature_map._vocab_prune_info = None

        elif vocab_pruning_config.get('enabled', False):
            # Runtime mode: current behavior with RemappedEmbedding
            from cloud_device_recsys.data.vocab_pruner import compute_vocab_pruning
            # Collect all data paths to scan (train + valid + test)
            train_positive_path = os.path.join(data_dir, 'train_positive.parquet')
            train_full_path = os.path.join(data_dir, 'train.parquet')
            valid_path = os.path.join(data_dir, 'valid.parquet')
            test_path = os.path.join(data_dir, 'test.parquet')
            # Use train_positive if it exists (negative sampling mode), otherwise full train
            train_scan_path = train_positive_path if os.path.exists(train_positive_path) else train_full_path
            scan_paths = [train_scan_path, valid_path, test_path]
            logger.info(f"[VocabPruner] Will scan {len(scan_paths)} data splits for vocabulary usage")
            prune_info = compute_vocab_pruning(
                feature_map,
                data_paths=scan_paths,
                min_vocab_size=vocab_pruning_config.get('min_vocab_size', 100),
                min_reduction_ratio=vocab_pruning_config.get('min_reduction_ratio', 0.3),
                cache_dir=data_dir,
            )
            # Attach to feature_map for use in model building (registry.py)
            feature_map._vocab_prune_info = prune_info
            logger.info(f"[VocabPruner] Pruning complete. {len(prune_info.features)} features pruned.")
        else:
            feature_map._vocab_prune_info = None

        # Propagate FP16 saving flag to feature_map for registry.py to pick up
        feature_map._save_fp16 = save_fp16

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
        # Instantiate Retrieval Stage lazily
        retrieval_stage = stages['retrieval']()
        # Run retrieval stage with its own data loaders
        r_metrics, r_valid, r_test = run_retrieval_stage(
            retrieval_stage, pipeline_config, dataset_config, fg_manager, logger=logger,
            run_test=bool(args.run_retrieval_test)
        )
        all_metrics.update(r_metrics)
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_path}")
        # Save retrieval stage outputs if requested
        if args.save_stage_outputs:
            save_stage_output(r_valid, stage_output_dir, 'retrieval_valid', logger)
            if r_test is not None:
                save_stage_output(r_test, stage_output_dir, 'retrieval_test', logger)
            logger.info(f"Saved retrieval_valid and retrieval_test metrics to {metrics_path}")

        # Instantiate Preranking Stage lazily (after retrieval)
        preranking_stage = stages['preranking']()
        # Run preranking stage with its own data loaders
        p_metrics, p_valid, p_test = run_preranking_stage(
            preranking_stage, pipeline_config, dataset_config, fg_manager=fg_manager,
            logger=logger, prev_output_valid=r_valid, prev_output_test=r_test,
            run_test=bool(args.run_preranking_test),
        )
        all_metrics.update(p_metrics)
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_path}")
        # Save preranking stage outputs if requested
        if args.save_stage_outputs:
            save_stage_output(p_valid, stage_output_dir, 'preranking_valid', logger)
            if p_test is not None:
                save_stage_output(p_test, stage_output_dir, 'preranking_test', logger)

        # Instantiate Reranking Stage lazily (after preranking)
        reranking_stage = stages['reranking']()
        # Run reranking stage with its own data loaders
        d_metrics = run_reranking_stage(reranking_stage, pipeline_config, dataset_config, fg_manager=fg_manager,
                                        logger=logger, run_test=bool(args.run_reranking_test),
                                        prev_output_path=args.prev_output_path, preranking_model=preranking_stage.model,
                                        prev_output_valid=p_valid, prev_output_test=p_test,
                                        retrieval_output_valid=r_valid, retrieval_output_test=r_test,
                                        feature_map=feature_map)
        all_metrics.update(d_metrics)

    elif args.mode == 'retrieval':
        logger.info("Running retrieval stage only")
        if 'retrieval' not in stages:
            raise RuntimeError("No retrieval stage found. Please run data preprocessing first")
        # Instantiate Retrieval Stage lazily
        retrieval_stage = stages['retrieval']()
        r_metrics, r_valid, r_test = run_retrieval_stage(
            retrieval_stage, pipeline_config, dataset_config, fg_manager, logger=logger,
            run_test=bool(args.run_retrieval_test)
        )
        all_metrics.update(r_metrics)

        # Save stage outputs if requested
        if args.save_stage_outputs:
            stage_output_dir = os.path.join(run_output_dir, 'stage_outputs')
            save_stage_output(r_valid, stage_output_dir, 'retrieval_valid', logger)
            if r_test is not None:
                save_stage_output(r_test, stage_output_dir, 'retrieval_test', logger)

    elif args.mode == 'preranking':
        logger.info("Running preranking stage only")
        if 'preranking' not in stages:
            raise RuntimeError("No preranking stage found. Please run retrieval first")

        # Load previous stage outputs if provided (from retrieval stage)
        if args.prev_output_path:
            prev_output_valid, prev_output_test = load_stage_outputs_from_dir(
                args.prev_output_path, 'retrieval', logger, load_test=bool(args.run_preranking_test)
            )
        else:
            raise RuntimeError(
                "Previous stage outputs not provided or failed to load. Preranking will run without candidate filtering from retrieval.")

        # Instantiate Preranking Stage lazily
        preranking_stage = stages['preranking']()
        p_metrics, p_valid, p_test = run_preranking_stage(
            preranking_stage, pipeline_config, dataset_config, fg_manager=fg_manager,
            logger=logger, prev_output_valid=prev_output_valid, prev_output_test=prev_output_test,
            run_test=bool(args.run_preranking_test)
        )
        all_metrics.update(p_metrics)

        # Save stage outputs if requested
        if args.save_stage_outputs:
            save_stage_output(p_valid, stage_output_dir, 'preranking_valid', logger)
            if p_test is not None:
                save_stage_output(p_test, stage_output_dir, 'preranking_test', logger)

    elif args.mode == 'reranking':
        logger.info("Running reranking stage only")
        if 'reranking' not in stages:
            raise RuntimeError("No reranking stage found. Please run preranking first")
        # Instantiate Reranking Stage lazily
        reranking_stage = stages['reranking']()
        d_metrics = run_reranking_stage(reranking_stage, pipeline_config, dataset_config, fg_manager=fg_manager,
                                        logger=logger, run_test=bool(args.run_reranking_test),
                                        prev_output_path=args.prev_output_path,
                                        feature_map=feature_map)
        all_metrics.update(d_metrics)
        
    elif args.mode == 'joint_train':
        logger.info("Running Cloud-Device Joint Training")
        if 'preranking' not in stages or 'reranking' not in stages:
            raise RuntimeError("Both preranking and reranking stages must be defined to run joint training")
            
        if args.prev_output_path:
            prev_output_valid, prev_output_test = load_stage_outputs_from_dir(
                args.prev_output_path, 'retrieval', logger
            )
        else:
            raise RuntimeError("Previous stage outputs (from retrieval) must be provided via --prev_output_path for joint training validation.")

        preranking_stage = stages['preranking']()
        reranking_stage = stages['reranking']()
        
        j_metrics = run_joint_training_stage(
            preranking_stage, reranking_stage, pipeline_config, dataset_config, fg_manager,
            logger=logger, run_test=bool(args.run_reranking_test),
            prev_output_valid=prev_output_valid, prev_output_test=prev_output_test
        )
        all_metrics.update(j_metrics)

    elif args.mode == 'dtcn_preranking':
        logger.info("Running DTCN Cloud-Side Preranking")
        if 'preranking' not in stages:
            raise RuntimeError("No preranking stage found in config.")

        if args.prev_output_path:
            prev_output_valid, prev_output_test = load_stage_outputs_from_dir(
                args.prev_output_path, 'retrieval', logger, load_test=bool(args.run_preranking_test)
            )
        else:
            raise RuntimeError("Previous stage outputs (from retrieval) must be provided via --prev_output_path.")

        d_metrics, _, _ = run_dtcn_preranking_stage(
            pipeline_config, dataset_config, fg_manager, feature_map,
            logger=logger, run_test=bool(args.run_preranking_test),
            prev_output_valid=prev_output_valid, prev_output_test=prev_output_test,
            gpu=args.gpu, output_dir=run_output_dir,
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
