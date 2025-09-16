# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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
import sys
import logging
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.preprocess import FeatureProcessor, build_dataset


os.chdir(os.path.dirname(os.path.realpath(__file__)))

# 添加模型路径
sys.path.append('./src')
from MTCL import MTCL


def run_expid(config_dir, expid, gpu_device=-1):
    # 设置日志
    set_logger(config_dir)
    logging.info("Params: " + print_to_json(config_dir))
    
    # 加载配置
    config = load_config(config_dir, expid)
    
    # 设置设备
    if gpu_device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    
    # 获取数据集
    dataset_id = config['dataset_id']
    dataset_config = config.get('dataset_config', {})
    
    # 构建特征映射
    feature_map = FeatureMap(config)
    
    # 处理数据
    data_dir = os.path.join(config['data_root'], dataset_id)
    if config.get('preprocess', False):
        feature_processor = FeatureProcessor(**config)
        build_dataset(feature_processor, **config)
    
    # 载入数据
    train_gen, valid_gen, test_gen = None, None, None
    if config.get('train_data'):
        train_gen = feature_map.get_dataloader(stage='train', **config)
    if config.get('valid_data'):
        valid_gen = feature_map.get_dataloader(stage='valid', **config)
    if config.get('test_data'):
        test_gen = feature_map.get_dataloader(stage='test', **config)
    
    # 初始化模型
    model = MTCL(feature_map, **config)
    
    # 训练模型
    if train_gen is not None:
        model.fit(train_gen, validation_data=valid_gen, **config)
    
    # 预测和评估
    if test_gen is not None:
        model.load_weights(model.checkpoint)
        logging.info('***** 测试结果 *****')
        test_result = model.evaluate(test_gen)
        logging.info('[Metrics] ' + print_to_list(test_result))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default='./config/', help='配置目录')
    parser.add_argument('--expid', type=str, default='MTCL_DCNv3', help='实验ID')
    parser.add_argument('--gpu_device', type=int, default=-1, help='GPU设备ID')
    
    args = parser.parse_args()
    run_expid(args.config_dir, args.expid, args.gpu_device) 