#!/usr/bin/env python3
"""
Merge all individual KD tuner configs into a single experiment directory,
then (optionally) run grid_search across multiple GPUs.

Usage:
    # Dry-run: only enumerate and merge, don't run
    python prepare_merged_experiments.py --config_dir config/kd_sonic_v3 --dry_run

    # Actually run with 4 GPUs
    python prepare_merged_experiments.py --config_dir config/kd_sonic_v3 --gpu 0 1 2 3
"""

import argparse
import glob
import os
import sys
import yaml
import shutil

# Add parent paths so we can import fuxictr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fuxictr_version  # noqa: F401
from fuxictr import autotuner


def merge_configs(config_dir, output_dir):
    """
    For each YAML in config_dir, call enumerate_params to expand the tuner space,
    then merge all generated model_config.yaml and dataset_config.yaml into output_dir.
    """
    yaml_files = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    if not yaml_files:
        print(f"ERROR: No YAML files found in {config_dir}")
        sys.exit(1)

    print(f"Found {len(yaml_files)} config files in {config_dir}")

    merged_model_configs = {}
    merged_dataset_configs = {}
    temp_dirs = []

    for yaml_file in yaml_files:
        fname = os.path.basename(yaml_file)
        print(f"\n{'='*60}")
        print(f"Processing: {fname}")
        print(f"{'='*60}")

        try:
            expanded_dir, tuner_keys = autotuner.enumerate_params(yaml_file)
            temp_dirs.append(expanded_dir)
        except Exception as e:
            print(f"ERROR processing {fname}: {e}")
            continue

        # Load the expanded model_config.yaml
        model_config_path = os.path.join(expanded_dir, "model_config.yaml")
        if os.path.exists(model_config_path):
            with open(model_config_path, "r") as f:
                model_configs = yaml.load(f, Loader=yaml.FullLoader)
            if model_configs:
                # Check for key collisions
                for key in model_configs:
                    if key in merged_model_configs:
                        print(f"  WARNING: Duplicate model config key: {key}")
                merged_model_configs.update(model_configs)
                print(f"  Added {len(model_configs)} model configs")

        # Load the expanded dataset_config.yaml
        dataset_config_path = os.path.join(expanded_dir, "dataset_config.yaml")
        if os.path.exists(dataset_config_path):
            with open(dataset_config_path, "r") as f:
                dataset_configs = yaml.load(f, Loader=yaml.FullLoader)
            if dataset_configs:
                merged_dataset_configs.update(dataset_configs)
                print(f"  Added {len(dataset_configs)} dataset configs")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Write merged model_config.yaml
    model_output = os.path.join(output_dir, "model_config.yaml")
    with open(model_output, "w") as f:
        yaml.dump(merged_model_configs, f, default_flow_style=None, indent=4)
    print(f"\n{'='*60}")
    print(f"Merged model configs: {len(merged_model_configs)} experiments → {model_output}")

    # Write merged dataset_config.yaml
    dataset_output = os.path.join(output_dir, "dataset_config.yaml")
    with open(dataset_output, "w") as f:
        yaml.dump(merged_dataset_configs, f, default_flow_style=None, indent=4)
    print(f"Merged dataset configs: {len(merged_dataset_configs)} datasets → {dataset_output}")

    # Clean up temporary expanded directories
    for temp_dir in temp_dirs:
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"  Cleaned up: {temp_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple KD tuner configs and run grid_search with multi-GPU."
    )
    parser.add_argument(
        "--config_dir", type=str, required=True,
        help="Directory containing individual KD tuner YAML configs (e.g., config/kd_sonic_v3)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for merged configs. Default: {config_dir}_merged"
    )
    parser.add_argument(
        "--gpu", nargs="+", default=[0, 1, 2, 3],
        help="List of GPU device IDs for parallel grid search (default: 0 1 2 3)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only enumerate and merge configs, don't run experiments"
    )
    args = parser.parse_args()

    config_dir = os.path.abspath(args.config_dir)
    output_dir = args.output_dir or (config_dir + "_merged")
    output_dir = os.path.abspath(output_dir)

    print(f"Config source:  {config_dir}")
    print(f"Merged output:  {output_dir}")
    print(f"GPU devices:    {args.gpu}")
    print(f"Dry run:        {args.dry_run}")
    print()

    # Step 1: Merge all configs
    merged_dir = merge_configs(config_dir, output_dir)

    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN complete. Merged configs are in:")
        print(f"  {merged_dir}")
        print(f"{'='*60}")
        return

    # Step 2: Run grid search with all GPUs
    print(f"\n{'='*60}")
    print(f"Starting grid_search with GPUs: {args.gpu}")
    print(f"{'='*60}")

    # Collect tuner_params_key from all configs (union of all tuner keys)
    all_tuner_keys = set()
    yaml_files = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        if "tuner_space" in config:
            all_tuner_keys.update(config["tuner_space"].keys())

    tuner_params_key = ",".join(sorted(all_tuner_keys))
    print(f"Tuner params key: {tuner_params_key}")

    autotuner.grid_search(
        merged_dir,
        gpu_list=args.gpu,
        tunner_params_key=tuner_params_key,
    )

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
