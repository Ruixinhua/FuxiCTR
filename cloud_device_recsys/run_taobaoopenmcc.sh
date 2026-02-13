export PYTHONBUFFERED=1
output_root=outputs
cloud_device_recsys_dir=~/FuxiCTR/cloud_device_recsys
cd ~/FuxiCTR/cloud_device_recsys

python -u run_hyperparam_search.py --mode retrieval --gpu 0 --dataset_id TaobaoOpenMCC \
    --search_config ${cloud_device_recsys_dir}/hp_config/openmcc_retrieval.yaml \
    --base_config ${cloud_device_recsys_dir}/config/TaobaoOpenMCC_retrieval_baseline_top1000.yaml \
    --config_dir ${cloud_device_recsys_dir}/config \
    --output_dir ${output_root}/TaobaoOpenMCC_retrieval  > ${output_root}/TaobaoOpenMCC_retrieval.log

python run_pipeline.py --mode retrieval --gpu 0 --dataset_id TaobaoOpenMCC \
    --output_dir ${output_root} --config ${cloud_device_recsys_dir}/config \
    --pipeline_id TaobaoOpenMCC_retrieval_baseline_top1000 \
    --experiment_id TaobaoOpenMCC_retrieval_baseline_top1000_parquet_negative \
    --run_retrieval_test 1 --save_stage_outputs 1 > ${output_root}/TaobaoOpenMCC_retrieval_baseline_top1000_parquet_negative.log

python -u run_hyperparam_search.py --mode preranking --gpu 0 \
    --search_config ${cloud_device_recsys_dir}/hp_config/openmcc_preranking.yaml \
    --base_config ${cloud_device_recsys_dir}/config/TaobaoOpenMCC_preranking_DINRanker_top100.yaml \
    --config_dir ${cloud_device_recsys_dir}/config \
    --prev_output_path ${output_root}/TaobaoOpenMCC_retrieval_baseline_top1000_parquet_negative/stage_outputs \
    --output_dir ${output_root}/TaobaoOpenMCC_preranking_emb_lr  > ${output_root}/TaobaoOpenMCC_preranking_emb_lr.log
