# Cloud-Device Recommendation System

This project implements a generic pipeline for cloud-device recommendation, supporting multiple stages (Retrieval, Preranking, Reranking) and flexible configuration.

## 🚀 Usage

You can execute the pipeline either directly via Python or by submitting a job to the cluster using the provided Bash script.

### 1. Python Direct Execution (`run_pipeline.py`)

Use the `run_pipeline.py` script to run the pipeline manually. This is suitable for local development or direct server execution.

**Command Structure:**
```bash
python cloud_device_recsys/run_pipeline.py \
    --config ./cloud_device_recsys/config \
    --dataset_id <DATASET_ID> \
    --mode <MODE> \
    [OPTIONS]
```

**Key Arguments:**
- `--config`: Path to the config directory (default: `./cloud_device_recsys/config`).
- `--dataset_id`: Dataset identifier (e.g., `TaobaoOpenMCC`, `TaobaoAd`).
- `--mode`: Execution mode.
  - `full`: Run all enabled stages sequentially.
  - `retrieval`: Run only the retrieval stage.
  - `preranking`: Run only the preranking stage.
  - `reranking`: Run only the reranking stage.
- `--gpu`: GPU ID to use (`-1` for CPU, `0`, `1`, etc.).
- `--pipeline_id`: Identifier for the pipeline configuration file (default: `default`). This loads `config/<PIPELINE_ID>.yaml`.
- `--experiment_id`: Custom identifier for the experiment run. If provided, the output directory will be `outputs/<EXPERIMENT_ID>`. If not, a timestamp-based directory is created.
- `--save_stage_outputs`: **Important**. Use this flag to save intermediate stage outputs (validation/test predictions) to disk. This is required if you plan to run subsequent stages independently later.
- `--prev_output_path`: Path to a directory containing outputs from a previous stage run. Use this when running `preranking` or `reranking` in isolation.

**Examples:**

*Run the full pipeline:*
```bash
python cloud_device_recsys/run_pipeline.py --dataset_id TaobaoOpenMCC --mode full --gpu 0
```

*Run step-by-step (saving intermediate results):*
```bash
# 1. Run Retrieval and save outputs
python cloud_device_recsys/run_pipeline.py --mode retrieval --save_stage_outputs

# 2. Run Preranking (loading from retrieval outputs)
# Replace path with your actual experiment output directory
python cloud_device_recsys/run_pipeline.py --mode preranking \
    --prev_output_path ./outputs/exp_20260203_120000/stage_outputs \
    --save_stage_outputs
```

---

### 2. SLURM/HPC Execution (`sonic_bash/run_pipeline_generic.sh`)

For cluster environments (like Sonic), use the generic wrapper script. It handles `conda` environment activation, `PYTHONPATH` setup, and job scheduling.

**Command Structure:**
```bash
sbatch sonic_bash/run_pipeline_generic.sh [DATASET_ID] [EXTRA_PYTHON_ARGS...]
```

**Arguments:**
1. **`DATASET_ID`** (1st Arg): Defaults to `TaobaoOpenMCC` if not specified.
2. **`EXTRA_ARGS`** (Rest): Any additional arguments are passed directly to `run_pipeline.py`.

**Examples:**

*Standard full run on TaobaoOpenMCC:*
```bash
sbatch sonic_bash/run_pipeline_generic.sh
```

*Run on TaobaoAd dataset:*
```bash
sbatch sonic_bash/run_pipeline_generic.sh TaobaoAd
```

*Pass arbitrary Python arguments (e.g., specific mode or GPU):*
```bash
# Runs TaobaoOpenMCC in 'retrieval' mode only
sbatch sonic_bash/run_pipeline_generic.sh TaobaoOpenMCC --mode retrieval --save_stage_outputs

# Runs TaobaoAd with specific config path
sbatch sonic_bash/run_pipeline_generic.sh TaobaoAd --config ./my_custom_config
```

## 📂 Output Structure

All results are saved to the `outputs/` directory:

```text
outputs/
└── exp_YYYYMMDD_HHMMSS/          # Unique experiment run
    ├── metrics.json              # Final evaluation metrics
    ├── run_config.yaml           # Snapshot of config used
    ├── pipeline.log              # Execution logs
    ├── retrieval/                # Model checkpoints/logs for retrieval
    ├── preranking/               # Model checkpoints/logs for preranking
    └── stage_outputs/            # (Optional) Saved intermediate outputs
        ├── retrieval_valid.pkl
        ├── retrieval_test.pkl
        └── ...
```
