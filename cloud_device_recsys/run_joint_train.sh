# =============================================================================
# Joint Training Hyperparameter Search: PNN (Preranking) + PNN (Reranking)
# =============================================================================

export PYTHONBUFFERED=1
output_root=/scratch/dliu2/FuxiCTR/outputs
cloud_device_recsys_dir=~/FuxiCTR/cloud_device_recsys
cd ${cloud_device_recsys_dir}

version="sequential_v3"
#models=("FCN" "PNN")
#datasets=("TaobaoOpenMCC" "TaobaoAd")
models=("FCN" "PNN")
datasets=("TaobaoAd")
# --run_preranking_test 1 --run_reranking_test 1
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        python -u run_hyperparam_search.py \
            --mode joint_train \
            --gpu 0 \
            --dataset_id "${dataset}" \
            --search_config "${cloud_device_recsys_dir}/hp_config/joint_${model}_${model}.yaml" \
            --base_config "${cloud_device_recsys_dir}/config/${dataset}_joint_${model}_${model}.yaml" \
            --config_dir ${cloud_device_recsys_dir}/config \
            --prev_output_path "${output_root}/${dataset}_retrieval_DT_top1000_parquet/stage_outputs" \
            --output_dir "${output_root}/${dataset}_JointTrain/${model}_${model}_${version}/"  > "${output_root}/log/${dataset}_joint_${model}_${model}_${version}".log
    done
done
