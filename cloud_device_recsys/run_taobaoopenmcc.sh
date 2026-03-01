export PYTHONBUFFERED=1
output_root=/scratch/dliu2/FuxiCTR/outputs
cloud_device_recsys_dir=~/FuxiCTR/cloud_device_recsys
#conda init
#conda activate fuxictr
cd ${cloud_device_recsys_dir}

# save stage outputs for preranking stage
#python run_pipeline.py --mode preranking --gpu 0 --dataset_id TaobaoOpenMCC \
#    --output_dir ${output_root} --config ${cloud_device_recsys_dir}/config \
#    --pipeline_id TaobaoOpenMCC_preranking_FCN_top100 \
#    --experiment_id TaobaoOpenMCC_preranking_FCN_top100 \
#    --prev_output_path ${output_root}/TaobaoOpenMCC_retrieval_baseline_top1000_parquet_negative/stage_outputs \
#    --run_preranking_test 1 --save_stage_outputs 1 > ${output_root}/log/TaobaoOpenMCC_preranking_FCN_top100.log
#
#python run_pipeline.py --mode preranking --gpu 0 --dataset_id TaobaoOpenMCC \
#    --output_dir ${output_root} --config ${cloud_device_recsys_dir}/config \
#    --pipeline_id TaobaoOpenMCC_preranking_PNN_top100 \
#    --experiment_id TaobaoOpenMCC_preranking_PNN_top100 \
#    --prev_output_path ${output_root}/TaobaoOpenMCC_retrieval_baseline_top1000_parquet_negative/stage_outputs \
#    --run_preranking_test 1 --save_stage_outputs 1 > ${output_root}/log/TaobaoOpenMCC_preranking_PNN_top100.log

python -u run_hyperparam_search.py --mode preranking --gpu 0 --dataset_id TaobaoOpenMCC \
    --search_config ${cloud_device_recsys_dir}/hp_config/preranking_FCN.yaml \
    --base_config ${cloud_device_recsys_dir}/config/TaobaoOpenMCC_preranking_FCN_top100.yaml \
    --config_dir ${cloud_device_recsys_dir}/config \
    --prev_output_path ${output_root}/TaobaoOpenMCC_retrieval_baseline_top1000_parquet_negative/stage_outputs \
    --output_dir ${output_root}/OpenMCC_Preranking/FCN_hyper/ > ${output_root}/log/TaobaoOpenMCC_preranking_FCN_hyper.log

# Add diversity to the baseline models and run hyperparameter search for preranking stage
#python -u run_hyperparam_search.py --mode preranking --gpu 0 --dataset_id TaobaoOpenMCC \
#    --search_config ${cloud_device_recsys_dir}/hp_config/preranking_FCN_diversity.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/TaobaoOpenMCC_preranking_FCN_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoOpenMCC_retrieval_baseline_top1000_parquet_negative/stage_outputs \
#    --output_dir ${output_root}/Preranking/TaobaoOpenMCC_preranking_FCN_diversity_v9 > ${output_root}/log/TaobaoOpenMCC_preranking_FCN_diversity_v9.log


#python -u run_hyperparam_search.py --mode preranking --gpu 0 --dataset_id TaobaoOpenMCC \
#    --search_config ${cloud_device_recsys_dir}/hp_config/preranking_DIN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/preranking_DIN_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoOpenMCC_retrieval_baseline_top1000_parquet_negative/stage_outputs \
#    --output_dir ${output_root}/Preranking/TaobaoOpenMCC_preranking_DIN  > ${output_root}/log/TaobaoOpenMCC_preranking_DIN.log

# Run Reranking stage with different combinations of models for candidate generation and re-ranking, and run hyperparameter search for each combination
# --run_reranking_test 1


#python -u run_hyperparam_search.py --mode reranking --gpu 0 --dataset_id TaobaoOpenMCC \
#    --search_config ${cloud_device_recsys_dir}/hp_config/reranking_FCN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/reranking_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoOpenMCC_preranking_FCN_top100/stage_outputs \
#    --output_dir ${output_root}/Reranking/TaobaoOpenMCC_reranking_FCN_FCN  > ${output_root}/log/TaobaoOpenMCC_reranking_FCN_FCN.log
#
#python -u run_hyperparam_search.py --mode reranking --gpu 0 --dataset_id TaobaoOpenMCC \
#    --search_config ${cloud_device_recsys_dir}/hp_config/reranking_PNN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/reranking_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoOpenMCC_preranking_FCN_top100/stage_outputs \
#    --output_dir ${output_root}/Reranking/TaobaoOpenMCC_reranking_FCN_PNN  > ${output_root}/log/TaobaoOpenMCC_reranking_FCN_PNN.log

#python -u run_hyperparam_search.py --mode reranking --gpu 0 --dataset_id TaobaoOpenMCC \
#    --search_config ${cloud_device_recsys_dir}/hp_config/reranking_FCN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/reranking_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoOpenMCC_preranking_PNN_top100/stage_outputs \
#    --output_dir ${output_root}/Reranking/TaobaoOpenMCC_reranking_PNN_FCN  > ${output_root}/log/TaobaoOpenMCC_reranking_PNN_FCN.log
#
#python -u run_hyperparam_search.py --mode reranking --gpu 0 --dataset_id TaobaoOpenMCC \
#    --search_config ${cloud_device_recsys_dir}/hp_config/reranking_PNN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/reranking_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoOpenMCC_preranking_PNN_top100/stage_outputs \
#    --output_dir ${output_root}/Reranking/TaobaoOpenMCC_reranking_PNN_PNN  > ${output_root}/log/TaobaoOpenMCC_reranking_PNN_PNN.log

#python run_pipeline.py --mode preranking --gpu 0 --dataset_id TaobaoOpenMCC \
#    --output_dir ${output_root} --config ${cloud_device_recsys_dir}/config \
#    --pipeline_id TaobaoOpenMCC_preranking_DIN_top100 \
#    --experiment_id TaobaoOpenMCC_preranking_DIN_top100 \
#    --prev_output_path ${output_root}/TaobaoOpenMCC_retrieval_baseline_top1000_parquet_negative/stage_outputs \
#    --run_preranking_test 1 --save_stage_outputs 1 > ${output_root}/log/TaobaoOpenMCC_preranking_DIN_top100.log
#
#python -u run_hyperparam_search.py --mode preranking --gpu 0 --dataset_id TaobaoOpenMCC \
#    --search_config ${cloud_device_recsys_dir}/hp_config/preranking_diversity.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/TaobaoOpenMCC_preranking_DIN_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoOpenMCC_retrieval_baseline_top1000_parquet_negative/stage_outputs \
#    --output_dir ${output_root}/Preranking/TaobaoOpenMCC_preranking_DIN_diversity > ${output_root}/log/TaobaoOpenMCC_preranking_DIN_diversity.log
#
#python -u run_hyperparam_search.py --mode reranking --gpu 0 --dataset_id TaobaoOpenMCC \
#    --search_config ${cloud_device_recsys_dir}/hp_config/reranking_DIN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/reranking_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoOpenMCC_preranking_DIN_top100/stage_outputs \
#    --output_dir ${output_root}/Reranking/TaobaoOpenMCC_reranking_DIN_DIN  > ${output_root}/log/TaobaoOpenMCC_reranking_DIN_DIN.log
#
#python -u run_hyperparam_search.py --mode reranking --gpu 0 --dataset_id TaobaoOpenMCC \
#    --search_config ${cloud_device_recsys_dir}/hp_config/reranking_FCN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/reranking_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoOpenMCC_preranking_DIN_top100/stage_outputs \
#    --output_dir ${output_root}/Reranking/TaobaoOpenMCC_reranking_DIN_FCN  > ${output_root}/log/TaobaoOpenMCC_reranking_DIN_FCN.log
#
#python -u run_hyperparam_search.py --mode reranking --gpu 0 --dataset_id TaobaoOpenMCC \
#    --search_config ${cloud_device_recsys_dir}/hp_config/reranking_PNN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/reranking_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoOpenMCC_preranking_DIN_top100/stage_outputs \
#    --output_dir ${output_root}/Reranking/TaobaoOpenMCC_reranking_DIN_PNN  > ${output_root}/log/TaobaoOpenMCC_reranking_DIN_PNN.log
