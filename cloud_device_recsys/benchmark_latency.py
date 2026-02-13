#!/usr/bin/env python
# =========================================================================
# Latency Benchmark: Cloud-Device Collaborative vs Pure-Cloud Inference
# =========================================================================
"""
Compare inference latency between:
  1. Cloud-Device Collaborative: Preranking(FG1+FG2) → Reranking(FG1+FG2+FG3)
  2. Pure-Cloud Baseline: Single model using all features (FG1+FG2+FG3)

Usage:
    python -m cloud_device_recsys.benchmark_latency \
        --config cloud_device_recsys/config \
        --pipeline_id TaobaoOpenMCC_preranking_FCN_top100 \
        --dataset_id TaobaoOpenMCC \
        --gpu 0
"""

import os
import sys
import copy
import json
import time
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fuxictr.features import FeatureMap
from cloud_device_recsys.config.feature_groups import FeatureGroupManager, FeatureGroup
from cloud_device_recsys.config.config_parser import ConfigParser
from cloud_device_recsys.models import build_model as registry_build_model
from cloud_device_recsys.utils import filter_feature_map, get_data_dir

logger = logging.getLogger("LatencyBenchmark")


# =========================================================================
# Data classes for results
# =========================================================================
@dataclass
class LatencyResult:
    """Timing results for a single benchmark run."""
    name: str
    num_features: int
    num_candidates: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    throughput: float  # candidates per second


# =========================================================================
# Synthetic data generation
# =========================================================================
def create_synthetic_batch(
    feature_map: FeatureMap,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Create a synthetic batch of data that matches the feature map specs.
    
    - categorical → random int in [0, vocab_size)
    - numeric → random float in [0, 1)
    - sequence → random int matrix [batch_size, max_len]
    """
    batch = {}
    for feat_name, feat_spec in feature_map.features.items():
        ftype = feat_spec.get("type", "categorical")
        
        if ftype == "sequence":
            max_len = feat_spec.get("max_len", 50)
            vocab_size = feat_spec.get("vocab_size", 100)
            batch[feat_name] = torch.randint(
                0, max(vocab_size, 2), (batch_size, max_len),
                dtype=torch.long, device=device
            )
        elif ftype == "numeric":
            batch[feat_name] = torch.randn(
                batch_size, dtype=torch.float32, device=device
            )
        else:  # categorical
            vocab_size = feat_spec.get("vocab_size", 100)
            batch[feat_name] = torch.randint(
                0, max(vocab_size, 2), (batch_size,),
                dtype=torch.long, device=device
            )
    
    return batch


# =========================================================================
# Model building utilities
# =========================================================================
def build_stage_model(
    feature_map: FeatureMap,
    fg_manager: FeatureGroupManager,
    allowed_groups: List[FeatureGroup],
    model_params: Dict[str, Any],
    output_dir: str,
    model_name: str = "DCNv3",
) -> Any:
    """Build a model for a specific set of feature groups."""
    filtered_fm = filter_feature_map(
        feature_map, fg_manager, allowed_groups,
        use_feature_encoder=model_params.get("use_feature_encoder", False)
    )
    filtered_fm.default_emb_dim = model_params.get("embedding_dim", 16)
    
    os.makedirs(os.path.join(output_dir, filtered_fm.dataset_id), exist_ok=True)
    
    model = registry_build_model(
        model_name=model_name,
        feature_map=filtered_fm,
        model_params=model_params,
        output_dir=output_dir,
    )
    return model, filtered_fm


# =========================================================================
# Latency measurement
# =========================================================================
def measure_latency(
    model: Any,
    feature_map: FeatureMap,
    batch_size: int,
    device: torch.device,
    warmup: int = 10,
    repeats: int = 100,
    name: str = "model",
) -> LatencyResult:
    """
    Measure model inference latency with warmup and multiple repeats.
    
    Returns:
        LatencyResult with timing statistics.
    """
    model.eval()
    
    # Create synthetic batch
    batch = create_synthetic_batch(feature_map, batch_size, device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            t_start = time.perf_counter()
            _ = model(batch)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            
            latencies.append((t_end - t_start) * 1000)  # ms
    
    latencies = np.array(latencies)
    num_features = len(feature_map.features)
    
    return LatencyResult(
        name=name,
        num_features=num_features,
        num_candidates=batch_size,
        mean_ms=float(np.mean(latencies)),
        std_ms=float(np.std(latencies)),
        min_ms=float(np.min(latencies)),
        max_ms=float(np.max(latencies)),
        p50_ms=float(np.percentile(latencies, 50)),
        p95_ms=float(np.percentile(latencies, 95)),
        throughput=float(batch_size / (np.mean(latencies) / 1000)),
    )


def measure_data_prep_latency(
    feature_map: FeatureMap,
    batch_size: int,
    device: torch.device,
    warmup: int = 10,
    repeats: int = 100,
    name: str = "data_prep",
) -> LatencyResult:
    """
    Measure data preparation latency (numpy → tensor conversion + to device).
    Simulates the tensor conversion overhead in process_and_rank_candidates.
    """
    # Create numpy arrays simulating pre-converted features
    np_arrays = {}
    for feat_name, feat_spec in feature_map.features.items():
        ftype = feat_spec.get("type", "categorical")
        if ftype == "sequence":
            max_len = feat_spec.get("max_len", 50)
            np_arrays[feat_name] = np.random.randint(0, 100, (batch_size, max_len), dtype=np.int64)
        elif ftype == "numeric":
            np_arrays[feat_name] = np.random.randn(batch_size).astype(np.float32)
        else:
            np_arrays[feat_name] = np.random.randint(0, 100, (batch_size,), dtype=np.int64)
    
    # Warmup
    for _ in range(warmup):
        batch = {}
        for feat_name, arr in np_arrays.items():
            batch[feat_name] = torch.from_numpy(arr).to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    # Timed runs
    latencies = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        t_start = time.perf_counter()
        batch = {}
        for feat_name, arr in np_arrays.items():
            batch[feat_name] = torch.from_numpy(arr).to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        
        latencies.append((t_end - t_start) * 1000)
    
    latencies = np.array(latencies)
    return LatencyResult(
        name=name,
        num_features=len(feature_map.features),
        num_candidates=batch_size,
        mean_ms=float(np.mean(latencies)),
        std_ms=float(np.std(latencies)),
        min_ms=float(np.min(latencies)),
        max_ms=float(np.max(latencies)),
        p50_ms=float(np.percentile(latencies, 50)),
        p95_ms=float(np.percentile(latencies, 95)),
        throughput=float(batch_size / (np.mean(latencies) / 1000)),
    )


# =========================================================================
# Result formatting
# =========================================================================
def print_result_table(results: List[LatencyResult], title: str = ""):
    """Print a nicely formatted table of latency results."""
    
    sep = "=" * 110
    print(f"\n{sep}")
    if title:
        print(f"  {title}")
        print(sep)
    
    header = (
        f"{'Scenario':<45} {'#Feat':>5} {'#Cand':>6} "
        f"{'Mean(ms)':>9} {'Std':>7} {'P50':>7} {'P95':>7} "
        f"{'Min':>7} {'Max':>7} {'Throughput':>12}"
    )
    print(header)
    print("-" * 110)
    
    for r in results:
        row = (
            f"{r.name:<45} {r.num_features:>5} {r.num_candidates:>6} "
            f"{r.mean_ms:>9.3f} {r.std_ms:>7.3f} {r.p50_ms:>7.3f} {r.p95_ms:>7.3f} "
            f"{r.min_ms:>7.3f} {r.max_ms:>7.3f} {r.throughput:>10.0f}/s"
        )
        print(row)
    
    print(sep)


def print_comparison(
    cloud_preranking: LatencyResult,
    device_reranking: LatencyResult,
    pure_cloud: LatencyResult,
    cloud_preranking_prep: Optional[LatencyResult] = None,
    device_reranking_prep: Optional[LatencyResult] = None,
    pure_cloud_prep: Optional[LatencyResult] = None,
):
    """Print the final latency comparison summary."""
    
    # Model inference only
    cd_total = cloud_preranking.mean_ms + device_reranking.mean_ms
    pc_total = pure_cloud.mean_ms
    
    # With data prep
    cd_total_full = cd_total
    pc_total_full = pc_total
    if cloud_preranking_prep and device_reranking_prep and pure_cloud_prep:
        cd_total_full += cloud_preranking_prep.mean_ms + device_reranking_prep.mean_ms
        pc_total_full += pure_cloud_prep.mean_ms
    
    sep = "=" * 80
    print(f"\n{sep}")
    print("  LATENCY COMPARISON SUMMARY")
    print(sep)
    
    print(f"\n  --- Model Forward Pass Only ---")
    print(f"  {'Cloud Preranking (FG1+FG2):':<50} {cloud_preranking.mean_ms:>8.3f} ms  ({cloud_preranking.num_candidates} candidates)")
    print(f"  {'Device Reranking (FG1+FG2+FG3):':<50} {device_reranking.mean_ms:>8.3f} ms  ({device_reranking.num_candidates} candidates)")
    print(f"  {'Cloud-Device Total:':<50} {cd_total:>8.3f} ms")
    print(f"  {'Pure-Cloud Full Model (FG1+FG2+FG3):':<50} {pc_total:>8.3f} ms  ({pure_cloud.num_candidates} candidates)")
    print()
    
    if pc_total > 0:
        speedup = pc_total / cd_total
        reduction = (1 - cd_total / pc_total) * 100
        print(f"  Speedup (pure-cloud / cloud-device):  {speedup:.2f}x")
        print(f"  Latency reduction:                    {reduction:+.1f}%")
    
    if cloud_preranking_prep and device_reranking_prep and pure_cloud_prep:
        print(f"\n  --- Including Data Preparation ---")
        cd_prep = cloud_preranking_prep.mean_ms + device_reranking_prep.mean_ms
        pc_prep = pure_cloud_prep.mean_ms
        print(f"  {'Cloud-Device (model + data prep):':<50} {cd_total_full:>8.3f} ms  (prep: {cd_prep:.3f} ms)")
        print(f"  {'Pure-Cloud (model + data prep):':<50} {pc_total_full:>8.3f} ms  (prep: {pc_prep:.3f} ms)")
        
        if pc_total_full > 0:
            speedup_full = pc_total_full / cd_total_full
            reduction_full = (1 - cd_total_full / pc_total_full) * 100
            print(f"\n  Speedup (with data prep):              {speedup_full:.2f}x")
            print(f"  Latency reduction (with data prep):    {reduction_full:+.1f}%")
    
    print(f"\n  Note: Cloud-Device system has LOWER latency because the cloud")
    print(f"  preranking uses fewer features (FG1+FG2) to score a large candidate")
    print(f"  set, while the device reranking uses all features on a much smaller")
    print(f"  candidate set (top-K from preranking).")
    print(sep)


# =========================================================================
# Main
# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark latency: Cloud-Device vs Pure-Cloud"
    )
    parser.add_argument("--config", type=str, default="cloud_device_recsys/config",
                        help="Config directory")
    parser.add_argument("--pipeline_id", type=str, default="TaobaoOpenMCC_preranking_FCN_top100",
                        help="Pipeline config name")
    parser.add_argument("--dataset_id", type=str, default="TaobaoOpenMCC",
                        help="Dataset ID")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU device ID (-1 for CPU)")
    parser.add_argument("--retrieval_candidates", type=int, default=1000,
                        help="Number of candidates from retrieval (input to preranking)")
    parser.add_argument("--preranking_top_k", type=int, default=100,
                        help="Top-K output from preranking (input to reranking)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Number of warmup iterations")
    parser.add_argument("--repeats", type=int, default=200,
                        help="Number of timed iterations")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name override (default: from pipeline config)")
    parser.add_argument("--preranking_embedding_dim", type=int, default=None,
                        help="Embedding dim for cloud preranking & pure-cloud (default: from config)")
    parser.add_argument("--reranking_embedding_dim", type=int, default=4,
                        help="Embedding dim for on-device reranking (default: 4, smaller model)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    
    # Device setup
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Load configurations
    config_parser = ConfigParser(config_dir=args.config)
    full_config = config_parser.get_full_config(
        pipeline_path=os.path.join(args.config, f"{args.pipeline_id}.yaml"),
        dataset_path=os.path.join(args.config, "dataset_config.yaml"),
        dataset_id=args.dataset_id,
    )
    pipeline_config = full_config["pipeline"]
    dataset_config = full_config["dataset"]
    
    # Build feature map
    data_dir = get_data_dir(dataset_config, args.dataset_id)
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    
    if not os.path.exists(feature_map_json):
        logger.error(f"Feature map not found at {feature_map_json}. Run preprocessing first.")
        return
    
    with open(feature_map_json, "r") as f:
        fm_data = json.load(f)
    if isinstance(fm_data.get("features"), dict):
        fm_data["features"] = [{k: v} for k, v in fm_data["features"].items()]
        converted_json = os.path.join(data_dir, "feature_map_fuxictr.json")
        with open(converted_json, "w") as f:
            json.dump(fm_data, f, indent=2)
        feature_map_json = converted_json
    
    feature_map = FeatureMap(args.dataset_id, data_dir)
    feature_map.load(feature_map_json, dataset_config)
    feature_map.dataset_config = dataset_config
    
    # Build feature group manager
    fg_manager = FeatureGroupManager()
    if "feature_groups" in pipeline_config:
        for fg_name, features in pipeline_config["feature_groups"].items():
            fg = FeatureGroup.from_string(fg_name)
            for feature in features:
                fg_manager.assign_feature(feature, fg)
    fg_manager.auto_assign_groups(feature_map, dataset_config)
    
    # Get model config
    preranking_config = pipeline_config.get("stages", {}).get("preranking", {})
    cloud_model_params = preranking_config.get("model_params", {}).copy()
    cloud_model_params["gpu"] = args.gpu
    model_name = args.model or preranking_config.get("model", "DCNv3")
    cloud_model_params["model"] = model_name
    
    # Override embedding dim if specified
    if args.preranking_embedding_dim is not None:
        cloud_model_params["embedding_dim"] = args.preranking_embedding_dim
    
    # Device reranking uses a smaller model (smaller embedding_dim)
    device_model_params = cloud_model_params.copy()
    device_model_params["embedding_dim"] = args.reranking_embedding_dim
    
    cloud_emb_dim = cloud_model_params.get("embedding_dim", 16)
    device_emb_dim = device_model_params["embedding_dim"]
    
    output_dir = "/tmp/benchmark_latency"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {args.dataset_id}")
    logger.info(f"Cloud embedding_dim: {cloud_emb_dim}")
    logger.info(f"Device embedding_dim: {device_emb_dim}")
    logger.info(f"Retrieval candidates: {args.retrieval_candidates}")
    logger.info(f"Preranking top-K: {args.preranking_top_k}")
    logger.info(f"Warmup: {args.warmup}, Repeats: {args.repeats}")
    
    # ===================================================================
    # Build 3 models
    # ===================================================================
    
    # 1. Cloud Preranking Model (FG1 + FG2) — embedding_dim from config
    logger.info(f"Building Cloud Preranking model (FG1 + FG2, emb_dim={cloud_emb_dim})...")
    preranking_model, preranking_fm = build_stage_model(
        feature_map=feature_map,
        fg_manager=fg_manager,
        allowed_groups=[FeatureGroup.FG1, FeatureGroup.FG2],
        model_params=cloud_model_params,
        output_dir=os.path.join(output_dir, "preranking"),
        model_name=model_name,
    )
    logger.info(f"  Features: {len(preranking_fm.features)} ({sorted(preranking_fm.features.keys())})")
    
    # 2. Device Reranking Model (FG1 + FG2 + FG3) — smaller embedding_dim for on-device
    logger.info(f"Building Device Reranking model (FG1 + FG2 + FG3, emb_dim={device_emb_dim})...")
    reranking_model, reranking_fm = build_stage_model(
        feature_map=feature_map,
        fg_manager=fg_manager,
        allowed_groups=[FeatureGroup.FG1, FeatureGroup.FG2, FeatureGroup.FG3],
        model_params=device_model_params,
        output_dir=os.path.join(output_dir, "reranking"),
        model_name=model_name,
    )
    logger.info(f"  Features: {len(reranking_fm.features)} ({sorted(reranking_fm.features.keys())})")
    
    # 3. Pure-Cloud Full Model (FG1 + FG2 + FG3) — same embedding_dim as cloud
    logger.info(f"Building Pure-Cloud Full model (FG1 + FG2 + FG3, emb_dim={cloud_emb_dim})...")
    pure_cloud_model, pure_cloud_fm = build_stage_model(
        feature_map=feature_map,
        fg_manager=fg_manager,
        allowed_groups=[FeatureGroup.FG1, FeatureGroup.FG2, FeatureGroup.FG3],
        model_params=cloud_model_params,
        output_dir=os.path.join(output_dir, "pure_cloud"),
        model_name=model_name,
    )
    logger.info(f"  Features: {len(pure_cloud_fm.features)} ({sorted(pure_cloud_fm.features.keys())})")
    
    # Print model parameter counts
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    logger.info(f"\nModel Parameter Counts:")
    logger.info(f"  Preranking (FG1+FG2):     {count_params(preranking_model):>10,}")
    logger.info(f"  Reranking (FG1+FG2+FG3):  {count_params(reranking_model):>10,}")
    logger.info(f"  Pure-Cloud (FG1+FG2+FG3): {count_params(pure_cloud_model):>10,}")
    
    # ===================================================================
    # Benchmark Model Forward Pass
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Starting Model Forward Pass Benchmark...")
    logger.info("=" * 60)
    
    # Cloud preranking: scores all retrieval candidates
    r_preranking = measure_latency(
        preranking_model, preranking_fm,
        batch_size=args.retrieval_candidates,
        device=device, warmup=args.warmup, repeats=args.repeats,
        name=f"Cloud Preranking (FG1+FG2, {args.retrieval_candidates} cands)",
    )
    
    # Device reranking: scores top-K from preranking
    r_reranking = measure_latency(
        reranking_model, reranking_fm,
        batch_size=args.preranking_top_k,
        device=device, warmup=args.warmup, repeats=args.repeats,
        name=f"Device Reranking (All feat, {args.preranking_top_k} cands)",
    )
    
    # Pure cloud: scores all retrieval candidates with full features
    r_pure_cloud = measure_latency(
        pure_cloud_model, pure_cloud_fm,
        batch_size=args.retrieval_candidates,
        device=device, warmup=args.warmup, repeats=args.repeats,
        name=f"Pure-Cloud Full (All feat, {args.retrieval_candidates} cands)",
    )
    
    # ===================================================================
    # Benchmark Data Preparation (numpy→tensor)
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Starting Data Preparation Benchmark...")
    logger.info("=" * 60)
    
    r_prep_preranking = measure_data_prep_latency(
        preranking_fm, batch_size=args.retrieval_candidates,
        device=device, warmup=args.warmup, repeats=args.repeats,
        name=f"Data Prep: Preranking ({args.retrieval_candidates} cands)",
    )
    
    r_prep_reranking = measure_data_prep_latency(
        reranking_fm, batch_size=args.preranking_top_k,
        device=device, warmup=args.warmup, repeats=args.repeats,
        name=f"Data Prep: Reranking ({args.preranking_top_k} cands)",
    )
    
    r_prep_pure_cloud = measure_data_prep_latency(
        pure_cloud_fm, batch_size=args.retrieval_candidates,
        device=device, warmup=args.warmup, repeats=args.repeats,
        name=f"Data Prep: Pure-Cloud ({args.retrieval_candidates} cands)",
    )
    
    # ===================================================================
    # Print Results
    # ===================================================================
    print_result_table(
        [r_preranking, r_reranking, r_pure_cloud],
        title="MODEL FORWARD PASS LATENCY"
    )
    
    print_result_table(
        [r_prep_preranking, r_prep_reranking, r_prep_pure_cloud],
        title="DATA PREPARATION LATENCY (numpy → tensor → device)"
    )
    
    print_comparison(
        cloud_preranking=r_preranking,
        device_reranking=r_reranking,
        pure_cloud=r_pure_cloud,
        cloud_preranking_prep=r_prep_preranking,
        device_reranking_prep=r_prep_reranking,
        pure_cloud_prep=r_prep_pure_cloud,
    )
    
    # ===================================================================
    # Feature Group Breakdown
    # ===================================================================
    sep = "=" * 80
    print(f"\n{sep}")
    print("  FEATURE GROUP BREAKDOWN")
    print(sep)
    
    fg1_feats = [f for f, g in fg_manager.feature_assignments.items() if g == FeatureGroup.FG1]
    fg2_feats = [f for f, g in fg_manager.feature_assignments.items() if g == FeatureGroup.FG2]
    fg3_feats = [f for f, g in fg_manager.feature_assignments.items() if g == FeatureGroup.FG3]
    
    print(f"  FG1 (Item features):          {len(fg1_feats)} features")
    print(f"    {sorted(fg1_feats)}")
    print(f"  FG2 (Behavior sequences):     {len(fg2_feats)} features")
    print(f"    {sorted(fg2_feats)}")
    print(f"  FG3 (Device-only user prof.): {len(fg3_feats)} features")
    print(f"    {sorted(fg3_feats)}")
    print(f"\n  Cloud uses:  FG1 + FG2 = {len(fg1_feats) + len(fg2_feats)} features")
    print(f"  Device uses: FG1 + FG2 + FG3 = {len(fg1_feats) + len(fg2_feats) + len(fg3_feats)} features")
    print(sep)


if __name__ == "__main__":
    main()
