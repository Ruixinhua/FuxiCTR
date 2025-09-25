#!/bin/bash -l
# adjust paths and env name as needed
conda activate your_env_name
export PYTHONUNBUFFERED=1
export PYTHONPATH="/path_to/FuxiCTR:$PYTHONPATH"
cd /path_to/FuxiCTR/experiment

# Running the parameter tuner with specified config and GPUs -- specify GPU IDs as needed
python run_param_tuner.py --config config/avazu_maskmerge_050_050_PNN_hyper.yaml --gpu 0 1

exit $?