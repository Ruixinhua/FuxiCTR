export PYTHONBUFFERED=1
output_root=/scratch/dliu2/FuxiCTR/outputs
cloud_device_recsys_dir=~/FuxiCTR/cloud_device_recsys
#conda init
#conda activate fuxictr
cd ${cloud_device_recsys_dir}

#python -u run_hyperparam_search.py --mode retrieval --gpu 0 \
#    --search_config ${cloud_device_recsys_dir}/hp_config/taobaoad_retrieval.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/TaobaoAd_retrieval_DT_top1000.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --output_dir ${output_root}/TaobaoAd_retrieval_v8  > ${output_root}/TaobaoAd_retrieval_v8.log

#python -u run_pipeline.py --mode retrieval --gpu 0 --dataset_id TaobaoAd \
#    --output_dir ${output_root} --config ${cloud_device_recsys_dir}/config \
#    --pipeline_id TaobaoAd_retrieval_DT_top1000 \
#    --experiment_id TaobaoAd_retrieval_DT_top1000_parquet \
#    --run_retrieval_test 1 --save_stage_outputs 1 > ${output_root}/TaobaoAd_retrieval_DT_top1000_parquet.log

#python -u run_hyperparam_search.py --mode preranking --gpu 0 --dataset_id TaobaoAd \
#    --search_config ${cloud_device_recsys_dir}/hp_config/preranking_PNN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/preranking_PNN_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoAd_retrieval_DT_top1000_parquet/stage_outputs \
#    --output_dir ${output_root}/Preranking/TaobaoAd_preranking_PNN  > ${output_root}/log/TaobaoAd_preranking_PNN.log

python -u run_hyperparam_search.py --mode preranking --gpu 0 --dataset_id TaobaoAd \
    --search_config ${cloud_device_recsys_dir}/hp_config/preranking_FCN.yaml \
    --base_config ${cloud_device_recsys_dir}/config/preranking_FCN_top100.yaml \
    --config_dir ${cloud_device_recsys_dir}/config \
    --prev_output_path ${output_root}/TaobaoAd_retrieval_DT_top1000_parquet/stage_outputs \
    --output_dir ${output_root}/TaobaoAd_Preranking/FCN_hyper/  > ${output_root}/log/TaobaoAd_preranking_FCN_hyper.log

#python -u run_hyperparam_search.py --mode preranking --gpu 0 --dataset_id TaobaoAd \
#    --search_config ${cloud_device_recsys_dir}/hp_config/preranking_DIN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/preranking_DIN_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoAd_retrieval_DT_top1000_parquet/stage_outputs \
#    --output_dir ${output_root}/Preranking/TaobaoAd_preranking_DIN  > ${output_root}/log/TaobaoAd_preranking_DIN.log

#python -u run_hyperparam_search.py --mode preranking --gpu 0 --dataset_id TaobaoAd \
#    --search_config ${cloud_device_recsys_dir}/hp_config/preranking_PNN_diversity.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/TaobaoAd_preranking_PNN_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoAd_retrieval_DT_top1000_parquet/stage_outputs \
#    --output_dir ${output_root}/Preranking/TaobaoAd_preranking_PNN_diversity_v3  > ${output_root}/log/TaobaoAd_preranking_PNN_diversity_v3.log
#
#python -u run_hyperparam_search.py --mode preranking --gpu 0 --dataset_id TaobaoAd \
#    --search_config ${cloud_device_recsys_dir}/hp_config/preranking_FCN_diversity.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/TaobaoAd_preranking_FCN_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoAd_retrieval_DT_top1000_parquet/stage_outputs \
#    --output_dir ${output_root}/Preranking/TaobaoAd_preranking_FCN_diversity_v2  > ${output_root}/log/TaobaoAd_preranking_FCN_diversity_v2.log

#python run_pipeline.py --mode preranking --gpu 0 --dataset_id TaobaoAd \
#    --output_dir ${output_root} --config ${cloud_device_recsys_dir}/config \
#    --pipeline_id TaobaoAd_preranking_PNN_top100 \
#    --experiment_id TaobaoAd_preranking_PNN_top100 \
#    --prev_output_path ${output_root}/TaobaoAd_retrieval_DT_top1000_parquet/stage_outputs \
#    --run_preranking_test 1 --save_stage_outputs 1 > ${output_root}/log/TaobaoAd_preranking_PNN_top100.log
#
#python run_pipeline.py --mode preranking --gpu 0 --dataset_id TaobaoAd \
#    --output_dir ${output_root} --config ${cloud_device_recsys_dir}/config \
#    --pipeline_id TaobaoAd_preranking_FCN_top100 \
#    --experiment_id TaobaoAd_preranking_FCN_top100 \
#    --prev_output_path ${output_root}/TaobaoAd_retrieval_DT_top1000_parquet/stage_outputs \
#    --run_preranking_test 1 --save_stage_outputs 1 > ${output_root}/log/TaobaoAd_preranking_FCN_top100.log

#python -u run_hyperparam_search.py --mode preranking --gpu 0 --dataset_id TaobaoAd \
#    --search_config ${cloud_device_recsys_dir}/hp_config/taobaoad_preranking_DIEN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/preranking_DIEN_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoAd_retrieval_DT_top1000_parquet/stage_outputs \
#    --output_dir ${output_root}/Preranking/TaobaoAd_preranking_DIEN_v5  > ${output_root}/log/TaobaoAd_preranking_DIEN_v5.log
#
#python -u run_hyperparam_search.py --mode reranking --gpu 0 --dataset_id TaobaoAd \
#    --search_config ${cloud_device_recsys_dir}/hp_config/reranking_FCN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/reranking_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoAd_preranking_FCN_top100/stage_outputs \
#    --output_dir ${output_root}/Reranking/TaobaoAd_reranking_FCN_FCN  > ${output_root}/log/TaobaoAd_reranking_FCN_FCN.log
#
#python -u run_hyperparam_search.py --mode reranking --gpu 0 --dataset_id TaobaoAd \
#    --search_config ${cloud_device_recsys_dir}/hp_config/reranking_PNN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/reranking_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoAd_preranking_FCN_top100/stage_outputs \
#    --output_dir ${output_root}/Reranking/TaobaoAd_reranking_FCN_PNN  > ${output_root}/log/TaobaoAd_reranking_FCN_PNN.log
#
#python -u run_hyperparam_search.py --mode reranking --gpu 0 --dataset_id TaobaoAd \
#    --search_config ${cloud_device_recsys_dir}/hp_config/reranking_FCN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/reranking_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoAd_preranking_PNN_top100/stage_outputs \
#    --output_dir ${output_root}/Reranking/TaobaoAd_reranking_PNN_FCN  > ${output_root}/log/TaobaoAd_reranking_PNN_FCN.log
#
#python -u run_hyperparam_search.py --mode reranking --gpu 0 --dataset_id TaobaoAd \
#    --search_config ${cloud_device_recsys_dir}/hp_config/reranking_PNN.yaml \
#    --base_config ${cloud_device_recsys_dir}/config/reranking_top100.yaml \
#    --config_dir ${cloud_device_recsys_dir}/config \
#    --prev_output_path ${output_root}/TaobaoAd_preranking_PNN_top100/stage_outputs \
#    --output_dir ${output_root}/Reranking/TaobaoAd_reranking_PNN_PNN  > ${output_root}/log/TaobaoAd_reranking_PNN_PNN.log
