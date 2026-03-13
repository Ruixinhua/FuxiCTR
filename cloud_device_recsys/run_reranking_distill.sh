export PYTHONBUFFERED=1
output_root=/scratch/dliu2/FuxiCTR/outputs
cloud_device_recsys_dir=~/FuxiCTR/cloud_device_recsys
cd ${cloud_device_recsys_dir}
version="_baseline_v4"
#version="_inject_v1"
#version="_distill_v1"
tag="_PNN"  # "_FCN" "_DNN" "_PNN"
#models=("FCN" "PNN" "DNN")
datasets=("TaobaoOpenMCC")
models=("FCN")
mode="reranking"
#datasets=("TaobaoAd" "TaobaoOpenMCC" )
#seeds=(2024 2026 42)
dataset="TaobaoAd"
python -u run_pipeline.py --mode save_preranking_outputs \
    --config ./config \
    --pipeline_id "${dataset}_dtcn_preranking_FCN" \
    --dataset_id "${dataset}" \
    --prev_output_path "${output_root}/${dataset}_retrieval_DT_top1000_parquet/stage_outputs" \
    --model_weights_path /scratch/dliu2/FuxiCTR/outputs/TaobaoAd/FCN_diversity_logit_final/seed_2024__pmodel_DCNv3__pdiversity_kernel_gram__pdiversity_lambda_0.1__pnum_negatives_4__789076/preranking/TaobaoAd/DCNv3_20260303_020014.model \
    --gpu 0 \
    --run_preranking_test 1 \
    --save_stage_outputs 1 \
    --output_dir "${output_root}/" \
    --experiment_id ${dataset}_${model}_preranking_diversity_outputs > "${output_root}/log/${dataset}_${model}_save_outputs".log
#dataset="TaobaoOpenMCC"
#python -u run_pipeline.py --mode save_preranking_outputs \
#    --config ./config \
#    --pipeline_id "${dataset}_dtcn_preranking_FCN" \
#    --dataset_id "${dataset}" \
#    --prev_output_path "${output_root}/${dataset}_retrieval_DT_top1000_parquet/stage_outputs" \
#    --model_weights_path /scratch/dliu2/FuxiCTR/outputs/TaobaoOpenMCC/FCN_diversity_cand_v2/seed_2026__pmodel_DCNv3__pdiversity_kernel_rbf__pdiversity_lambda_0.01__pnum_deep_cross_layers_4__pnum_heads_16__pnum_negatives_4__pnum_shallow_cross_layers_1__pneg_sampling_pool_candidate__d5f3ad/preranking/TaobaoOpenMCC/DCNv3_20260307_134911.model \
#    --gpu 0 \
#    --run_preranking_test 1 \
#    --save_stage_outputs 1 \
#    --output_dir "${output_root}/" \
#    --experiment_id ${dataset}_${model}_preranking_diversity_outputs > "${output_root}/log/${dataset}_${model}_save_outputs".log

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    python -u run_hyperparam_search.py \
      --mode ${mode} \
      --gpu 0 --run_reranking_test 1 \
      --dataset_id "${dataset}" \
      --search_config "${cloud_device_recsys_dir}/hp_config/${dataset}_${mode}${tag}_seed.yaml" \
      --base_config "${cloud_device_recsys_dir}/config/${dataset}_${mode}_cloud_teacher_${model}.yaml" \
      --config_dir ${cloud_device_recsys_dir}/config \
      --prev_output_path "${output_root}/${dataset}_${model}_preranking_diversity_outputs/stage_outputs" \
      --retrieval_output_path "${output_root}/${dataset}_retrieval_DT_top1000_parquet/stage_outputs" \
      --output_dir "${output_root}/${dataset}/Reranking/${model}${tag}${version}/" \
      > "${output_root}/log/${dataset}_${model}${tag}${version}".log
  done
done