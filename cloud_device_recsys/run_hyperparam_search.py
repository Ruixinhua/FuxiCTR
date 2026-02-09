#!/usr/bin/env python
# =========================================================================
# Hyperparameter Search Script for Cloud-Device Recommendation System
# =========================================================================
"""
Performs grid search over hyperparameters for pipeline stages.

Usage:
    # Dry run (show parameter combinations only)
    python run_hyperparam_search.py \
        --search_config ./config/search_example.yaml \
        --base_config ./config/TaobaoOpenMCC_full_DT_DIN_DIN.yaml \
        --dry_run

    # Actual search
    python run_hyperparam_search.py \
        --search_config ./config/search_example.yaml \
        --base_config ./config/TaobaoOpenMCC_full_DT_DIN_DIN.yaml \
        --output_dir ./outputs/hp_search \
        --gpu 0

    # Quick test with limited data
    python run_hyperparam_search.py \
        --search_config ./config/search_example.yaml \
        --base_config ./config/TaobaoOpenMCC_full_DT_DIN_DIN.yaml \
        --output_dir ./outputs/hp_search_test \
        --n_rows 1000 \
        --gpu 0
"""

import os
import sys
import argparse
import yaml
import json
import copy
import subprocess
import itertools
import hashlib
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter Search for Cloud-Device Recommendation Pipeline'
    )
    parser.add_argument('--search_config', type=str, required=True,
                        help='Path to search configuration YAML file')
    parser.add_argument('--base_config', type=str, required=True,
                        help='Path to base pipeline configuration YAML file')
    parser.add_argument('--config_dir', type=str, default='./config',
                        help='Configuration directory (for dataset_config.yaml)')
    parser.add_argument('--dataset_id', type=str, default=None,
                        help='Dataset ID to use (overrides base config)')
    parser.add_argument('--output_dir', type=str, default='./outputs/hp_search',
                        help='Output directory for search results')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'retrieval', 'preranking', 'reranking'],
                        help='Pipeline execution mode')
    parser.add_argument('--n_rows', type=int, default=None,
                        help='Override debug n_rows for quick testing')
    parser.add_argument('--prev_output_path', type=str, default=None,
                        help='Path to previous stage outputs (for preranking/reranking mode)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only show parameter combinations without running')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous search (skip completed experiments)')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    """Load a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str):
    """Save data to a YAML file."""
    with open(path, 'w') as f:
        yaml.dump(data, f, indent=2, allow_unicode=True)


def set_nested_value(d: dict, key_path: str, value: Any) -> dict:
    """
    Set a nested value in a dictionary using dot notation.
    
    Example:
        set_nested_value({}, 'model_params.embedding_dim', 64)
        -> {'model_params': {'embedding_dim': 64}}
    """
    keys = key_path.split('.')
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return d


def get_nested_value(d: dict, key_path: str, default=None) -> Any:
    """
    Get a nested value from a dictionary using dot notation.
    """
    keys = key_path.split('.')
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def generate_param_combinations(search_space: Dict[str, Dict[str, List]]) -> List[Dict]:
    """
    Generate all parameter combinations from search space.
    
    Args:
        search_space: {stage_name: {param_path: [value1, value2, ...]}}
        
    Returns:
        List of parameter dictionaries, each representing one combination.
        Format: [{stage_name: {param_path: value, ...}, ...}, ...]
    """
    # Flatten to list of (stage, param_path, values)
    param_specs = []
    for stage, params in search_space.items():
        for param_path, values in params.items():
            param_specs.append((stage, param_path, values))
    
    if not param_specs:
        return [{}]
    
    # Generate cartesian product
    all_values = [spec[2] for spec in param_specs]
    combinations = list(itertools.product(*all_values))
    
    # Convert to list of dicts
    result = []
    for combo in combinations:
        param_dict = {}
        for i, value in enumerate(combo):
            stage, param_path, _ = param_specs[i]
            if stage not in param_dict:
                param_dict[stage] = {}
            param_dict[stage][param_path] = value
        result.append(param_dict)
    
    return result


def apply_params_to_config(base_config: dict, params: Dict[str, Dict[str, Any]], 
                           fixed_overrides: Dict[str, Dict[str, Any]] = None) -> dict:
    """
    Apply parameter overrides to a base configuration.
    
    Args:
        base_config: Base pipeline configuration dict
        params: {stage_name: {param_path: value, ...}}
        fixed_overrides: Optional fixed overrides to apply (not part of search)
        
    Returns:
        Modified configuration dict
    """
    config = copy.deepcopy(base_config)
    
    # Apply fixed overrides first
    if fixed_overrides:
        for stage, stage_params in fixed_overrides.items():
            if stage not in config.get('stages', {}):
                continue
            for param_path, value in stage_params.items():
                set_nested_value(config['stages'][stage], param_path, value)
    
    # Apply search params
    for stage, stage_params in params.items():
        if stage not in config.get('stages', {}):
            print(f"Warning: Stage '{stage}' not found in config, skipping")
            continue
        for param_path, value in stage_params.items():
            set_nested_value(config['stages'][stage], param_path, value)
    
    return config


def generate_experiment_id(params: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate a unique experiment ID from parameters.
    
    The ID is human-readable with a short hash suffix for uniqueness.
    """
    # Build a descriptive string
    parts = []
    for stage, stage_params in sorted(params.items()):
        for param_path, value in sorted(stage_params.items()):
            # Get last part of param path for brevity
            short_name = param_path.split('.')[-1]
            # Format value
            if isinstance(value, list):
                val_str = '_'.join(str(v) for v in value)
            elif isinstance(value, float):
                val_str = f"{value:.0e}" if value < 0.01 else str(value)
            else:
                val_str = str(value)
            parts.append(f"{stage[0]}{short_name}_{val_str}")
    
    # Create base name
    base_name = '__'.join(parts) if parts else 'baseline'
    
    # Add short hash for uniqueness
    full_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    
    return f"{base_name}__{full_hash}"


def params_to_flat_dict(params: Dict[str, Dict[str, Any]], prefix: str = '') -> Dict[str, Any]:
    """
    Flatten nested params dict for CSV output.
    
    Example:
        {'retrieval': {'model_params.embedding_dim': 64}}
        -> {'retrieval.model_params.embedding_dim': 64}
    """
    result = {}
    for stage, stage_params in params.items():
        for param_path, value in stage_params.items():
            key = f"{prefix}{stage}.{param_path}" if prefix else f"{stage}.{param_path}"
            # Convert lists to strings for CSV
            if isinstance(value, list):
                result[key] = str(value)
            else:
                result[key] = value
    return result


def run_experiment(
    config_name: str,
    config_dir: str,
    experiment_id: str,
    args: argparse.Namespace,
    script_dir: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run a single experiment by invoking run_pipeline.py.
    
    Args:
        config_name: Name of the config file (without .yaml extension)
        config_dir: Directory containing the config file
        experiment_id: Unique experiment identifier
        args: Parsed command line arguments
        script_dir: Directory containing run_pipeline.py
    
    Returns:
        (success: bool, metrics: dict)
    """
    run_pipeline_path = os.path.join(script_dir, 'run_pipeline.py')
    
    # Build command
    cmd = [
        sys.executable,
        run_pipeline_path,
        '--config', config_dir,
        '--pipeline_id', config_name,  # Just the name, not .yaml extension
        '--mode', args.mode,
        '--output_dir', args.output_dir,
        '--experiment_id', experiment_id,
        '--gpu', str(args.gpu),
        '--seed', str(args.seed),
    ]
    
    if args.dataset_id:
        cmd.extend(['--dataset_id', args.dataset_id])
    
    if args.n_rows:
        cmd.extend(['--n_rows', str(args.n_rows)])
    
    if args.prev_output_path:
        cmd.extend(['--prev_output_path', args.prev_output_path])
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        subprocess.run(
            cmd,
            cwd=script_dir,
            capture_output=False,
            text=True
        )
        
        # Check for metrics file
        metrics_path = os.path.join(args.output_dir, experiment_id, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return True, metrics
        else:
            print(f"Warning: metrics.json not found at {metrics_path}")
            return False, {}
            
    except Exception as e:
        print(f"Error running experiment: {e}")
        return False, {}


def save_results(
    results: List[Dict[str, Any]], 
    output_path: str,
    all_param_keys: List[str] = None
):
    """
    Save results to CSV file.
    
    Args:
        results: List of result dicts with 'params', 'metrics', 'experiment_id', 'status'
        output_path: Path to output CSV
        all_param_keys: All parameter keys for consistent column ordering
    """
    rows = []
    for r in results:
        row = {
            'experiment_id': r['experiment_id'],
            'status': r['status'],
            'timestamp': r.get('timestamp', ''),
        }
        # Add params
        flat_params = params_to_flat_dict(r['params'])
        row.update(flat_params)
        # Add metrics
        row.update(r.get('metrics', {}))
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns: experiment_id, status, timestamp, params..., metrics...
    if len(df) > 0:
        meta_cols = ['experiment_id', 'status', 'timestamp']
        param_cols = sorted([c for c in df.columns if c.startswith(('retrieval.', 'preranking.', 'reranking.'))])
        metric_cols = sorted([c for c in df.columns if c not in meta_cols + param_cols])
        df = df[meta_cols + param_cols + metric_cols]
    
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    args = parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load configurations
    print(f"Loading search config: {args.search_config}")
    search_config = load_yaml(args.search_config)
    
    print(f"Loading base config: {args.base_config}")
    base_config = load_yaml(args.base_config)
    
    # Extract search space and fixed overrides
    search_space = search_config.get('search_space', {})
    fixed_overrides = search_config.get('fixed_overrides', {})
    
    # Generate parameter combinations
    param_combinations = generate_param_combinations(search_space)
    
    print(f"\n{'='*60}")
    print("Hyperparameter Search Configuration")
    print(f"{'='*60}")
    print(f"Total combinations: {len(param_combinations)}")
    print(f"Stages being searched: {list(search_space.keys())}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print(f"GPU: {args.gpu}")
    if args.n_rows:
        print(f"Debug n_rows: {args.n_rows}")
    print(f"{'='*60}\n")
    
    # Display all combinations
    print("Parameter combinations:")
    for i, params in enumerate(param_combinations):
        exp_id = generate_experiment_id(params)
        print(f"\n  [{i+1}/{len(param_combinations)}] {exp_id}")
        for stage, stage_params in params.items():
            for param_path, value in stage_params.items():
                print(f"      {stage}.{param_path} = {value}")
    
    if args.dry_run:
        print("\n[DRY RUN] Exiting without running experiments.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Temp configs will be saved in the original config_dir (where dataset_config.yaml exists)
    # with a unique prefix to avoid conflicts
    temp_config_prefix = 'hp_temp_'
    
    # Check for existing results (for resume)
    results_path = os.path.join(args.output_dir, 'results.csv')
    completed_experiments = set()
    existing_results = []
    
    if args.resume and os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        completed_experiments = set(existing_df[existing_df['status'] == 'success']['experiment_id'].tolist())
        existing_results = existing_df.to_dict('records')
        print(f"\nResuming: Found {len(completed_experiments)} completed experiments")
    
    # Run experiments
    results = existing_results.copy()
    
    for i, params in enumerate(param_combinations):
        experiment_id = generate_experiment_id(params)
        
        # Skip if already completed
        if experiment_id in completed_experiments:
            print(f"\n[{i+1}/{len(param_combinations)}] Skipping {experiment_id} (already completed)")
            continue
        
        print(f"\n[{i+1}/{len(param_combinations)}] Starting {experiment_id}")
        
        # Apply params to config
        modified_config = apply_params_to_config(base_config, params, fixed_overrides)
        
        # Save temp config to the original config_dir (where dataset_config.yaml is)
        # Use 'hp_temp_' prefix to identify temp configs
        config_name = f'{temp_config_prefix}{experiment_id}'
        temp_config_path = os.path.join(args.config_dir, f'{config_name}.yaml')
        save_yaml(modified_config, temp_config_path)
        
        # Run experiment
        success, metrics = run_experiment(
            config_name=config_name,
            config_dir=args.config_dir,  # Use original config_dir
            experiment_id=experiment_id,
            args=args,
            script_dir=script_dir
        )
        
        # Clean up temp config file
        try:
            os.remove(temp_config_path)
        except OSError:
            pass
        
        # Record result
        result = {
            'experiment_id': experiment_id,
            'params': params,
            'metrics': metrics,
            'status': 'success' if success else 'failed',
            'timestamp': datetime.now().isoformat(),
        }
        results.append(result)
        
        # Save intermediate results
        save_results(results, results_path)
    
    # Final summary
    print(f"\n{'='*60}")
    print("Search Complete!")
    print(f"{'='*60}")
    print(f"Total experiments: {len(param_combinations)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Results saved to: {results_path}")
    
    # Show best results if available
    if results:
        successful = [r for r in results if r['status'] == 'success' and r.get('metrics')]
        if successful:
            # Try to find a good metric to sort by
            sample_metrics = successful[0]['metrics']
            sort_metrics = [
                'reranking_test_Recall@5', 'preranking_test_Recall@100', 
                'retrieval_test_Recall@1000', 'test_auc'
            ]
            sort_by = None
            for m in sort_metrics:
                if m in sample_metrics:
                    sort_by = m
                    break
            
            if sort_by:
                best = max(successful, key=lambda x: x['metrics'].get(sort_by, 0))
                print(f"\nBest result (by {sort_by}):")
                print(f"  Experiment: {best['experiment_id']}")
                print(f"  {sort_by}: {best['metrics'].get(sort_by)}")


if __name__ == '__main__':
    main()
