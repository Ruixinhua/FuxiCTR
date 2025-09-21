# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import logging
from fuxictr.utils import load_config, set_logger, print_to_json, save_results_to_csv
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import build_dataset
from fuxictr.datasets.avazu import CustomizedFeatureProcessor
import model_zoo
import gc
import argparse
import os
from pathlib import Path


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--save_predictions', action='store_true', help='Whether to save prediction results for model ensemble')
    parser.add_argument('--predictions_dir', type=str, default='./predictions', help='Directory to save prediction results')
    parser.add_argument('--save_checkpoints', action='store_true', help='Whether to save model checkpoints')
    parser.add_argument('--tunner_params_key', type=str, default=None,
                        help='Parameters for hyper-parameter tuning, in format of key1,key2,...,')
    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    params['save_checkpoints'] = args['save_checkpoints']
    params['tunner_params_key'] = args['tunner_params_key']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    # Build feature_map and transform data
    feature_encoder = CustomizedFeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = \
        build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    
    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model
    
    # Print model structure
    logging.info("Model structure:")
    logging.info(str(model))

    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

    logging.info('****** Validation evaluation ******')
    # 检查是否为MTCL模型且启用了分塔最优保存
    if hasattr(model, 'use_tower_optimal_saving') and model.use_tower_optimal_saving:
        logging.info('Using tower optimal combination evaluation for validation')
        valid_result = model.evaluate_with_tower_optimal(valid_gen)
        
        # 同时提供标准评估结果作为对比
        logging.info('Standard evaluation for comparison:')
        valid_result_standard = model.evaluate(valid_gen, save_predictions=args['save_predictions'], save_dir=os.path.join(args['predictions_dir'], 'validation'))
        logging.info('[Standard] ' + ' - '.join([f'{k}: {v}' for k, v in valid_result_standard.items() if isinstance(v, (int, float))]))
        logging.info('[Tower Optimal] ' + ' - '.join([f'{k}: {v}' for k, v in valid_result.items() if isinstance(v, (int, float)) and k != 'optimal_epochs']))
        
        if 'optimal_epochs' in valid_result:
            logging.info('[Tower Optimal Epochs] ' + ' - '.join([f'{k}: epoch {v}' for k, v in valid_result['optimal_epochs'].items()]))
        valid_result = {
            k: float(v) for k, v in valid_result.items() if not isinstance(v, bool) and isinstance(v, (int, float))
        }
    else:
        valid_result = model.evaluate(valid_gen, save_predictions=args['save_predictions'], save_dir=os.path.join(args['predictions_dir'], 'validation'))
    
    del train_gen, valid_gen
    gc.collect()
    
    test_result = {}
    if params["test_data"]:
        logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        
        # 同样对测试集使用分塔最优评估
        if hasattr(model, 'use_tower_optimal_saving') and model.use_tower_optimal_saving:
            logging.info('Using tower optimal combination evaluation for test')
            test_result = model.evaluate_with_tower_optimal(test_gen)
            
            # 同时提供标准评估结果作为对比
            logging.info('Standard test evaluation for comparison:')
            test_result_standard = model.evaluate(test_gen, save_predictions=args['save_predictions'], save_dir=os.path.join(args['predictions_dir'], 'test'))
            logging.info('[Standard Test] ' + ' - '.join([f'{k}: {v}' for k, v in test_result_standard.items() if isinstance(v, (int, float))]))
            logging.info('[Tower Optimal Test] ' + ' - '.join([f'{k}: {v}' for k, v in test_result.items() if isinstance(v, (int, float)) and k != 'optimal_epochs']))
            
            if 'optimal_epochs' in test_result:
                logging.info('[Test Tower Optimal Epochs] ' + ' - '.join([f'{k}: epoch {v}' for k, v in test_result['optimal_epochs'].items()]))
            test_result = {
                k: float(v) for k, v in test_result.items() if not isinstance(v, bool) and isinstance(v, (int, float))
            }
        else:
            test_result = model.evaluate(test_gen, save_predictions=args['save_predictions'], save_dir=os.path.join(args['predictions_dir'], 'test'))
    # 使用抽离的保存函数
    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    save_results_to_csv(params, experiment_id, result_filename, valid_result, test_result)
