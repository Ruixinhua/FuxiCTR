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

"""
DCNv2 with MultiTowerModule 示例
这个文件展示了如何通过继承原始 DCNv2 来集成 MultiTowerModule 组件
遵循开闭原则，最大程度保持与原始 DCNv2 的一致性
"""

import torch
import logging
from fuxictr.pytorch.layers import MultiTowerModule
from .DCNv2 import DCNv2


class DCNv2WithMultiTower(DCNv2):
    """
    DCNv2 with MultiTowerModule support
    
    继承自原始 DCNv2 类，通过扩展的方式支持域感知结构，
    保持与原始 DCNv2 的完全兼容性。
    """
    
    def __init__(self, 
                 feature_map,
                 learning_rate=1e-3,
                 # 域感知结构相关参数（与 DCNv3 保持一致）
                 use_domain_aware_structure=False,
                 tower_hidden_units_list=None,
                 tower_activation='ReLU',
                 tower_l2_reg_list=None,
                 tower_dropout_list=None,
                 use_bn_tower=True,
                 scene_name='scene_id',
                 scene_num_shift=1,
                 use_scene_id_mapping=False,
                 mapping_feature_name=None,
                 mapping_feature_type=None,
                 feature2id_dict=None,
                 default_value=None,
                 feature_map_dict=None,
                 **kwargs):
        
        # 存储域感知结构相关参数
        self.use_domain_aware_structure = use_domain_aware_structure
        multi_tower_params = {
            'tower_hidden_units_list': tower_hidden_units_list,
            'tower_activation': tower_activation,
            'tower_l2_reg_list': tower_l2_reg_list,
            'tower_dropout_list': tower_dropout_list,
            'use_bn_tower': use_bn_tower,
            'scene_name': scene_name,
            'scene_num_shift': scene_num_shift,
            'use_scene_id_mapping': use_scene_id_mapping,
            'mapping_feature_name': mapping_feature_name,
            'mapping_feature_type': mapping_feature_type,
            'feature2id_dict': feature2id_dict,
            'default_value': default_value,
            'feature_map_dict': feature_map_dict
        }
        
        # 调用父类初始化
        super(DCNv2WithMultiTower, self).__init__(feature_map, learning_rate=learning_rate, **kwargs)
        
        # 如果使用域感知结构，初始化多塔模块（必须在父类初始化后进行，确保设备正确）
        if use_domain_aware_structure:
            self._init_multi_tower_structure(multi_tower_params, kwargs)
            # 确保MultiTowerModule在正确的设备上
            self.multi_tower_module.to(self.device)
            self.reset_parameters()
            self.model_to_device()
        logging.info(f"DCNv2WithMultiTower initialized. Use domain-aware structure: {use_domain_aware_structure}")
    
    def _init_multi_tower_structure(self, multi_tower_params, model_kwargs):
        """初始化多塔结构，仿照 DCNv3 的实现"""
        tower_hidden_units_list = multi_tower_params['tower_hidden_units_list']
        if tower_hidden_units_list is None:
            raise ValueError("`tower_hidden_units_list` must be specified when using multi-tower structure.")
        
        # 计算 DCNv2 输出维度，根据不同的 model_structure 计算
        tower_input_dim = self._calculate_tower_input_dim(model_kwargs)
        
        # 使用 MultiTowerModule 替换原有的多塔实现
        self.multi_tower_module = MultiTowerModule(
            input_dim=tower_input_dim,
            tower_hidden_units_list=tower_hidden_units_list,
            tower_activation=multi_tower_params['tower_activation'],
            tower_l2_reg_list=multi_tower_params['tower_l2_reg_list'],
            tower_dropout_list=multi_tower_params['tower_dropout_list'],
            use_bn_tower=multi_tower_params['use_bn_tower'],
            scene_name=multi_tower_params['scene_name'],
            scene_num_shift=multi_tower_params['scene_num_shift'],
            use_scene_id_mapping=multi_tower_params['use_scene_id_mapping'],
            mapping_feature_name=multi_tower_params['mapping_feature_name'],
            mapping_feature_type=multi_tower_params['mapping_feature_type'],
            feature2id_dict=multi_tower_params['feature2id_dict'],
            default_value=multi_tower_params['default_value'],
            feature_map_dict=multi_tower_params['feature_map_dict']
        )
    
    def _calculate_tower_input_dim(self, model_kwargs):
        """计算多塔模块的输入维度，根据 DCNv2 的不同结构"""
        input_dim = self.feature_map.sum_emb_out_dim()
        stacked_dnn_hidden_units = model_kwargs.get('stacked_dnn_hidden_units', [])
        parallel_dnn_hidden_units = model_kwargs.get('parallel_dnn_hidden_units', [])
        model_structure = model_kwargs.get('model_structure', 'parallel')
        
        if model_structure == "crossnet_only":
            # 只有 crossnet，输出维度等于输入维度
            return input_dim
        elif model_structure == "stacked":
            # crossnet + stacked_dnn，输出维度为 stacked_dnn 的最后一层
            return stacked_dnn_hidden_units[-1]
        elif model_structure == "parallel":
            # crossnet + parallel_dnn 并行，输出维度为两者相加
            return input_dim + parallel_dnn_hidden_units[-1]
        elif model_structure == "stacked_parallel":
            # crossnet + stacked_dnn + parallel_dnn，输出维度为两个 dnn 的相加
            return stacked_dnn_hidden_units[-1] + parallel_dnn_hidden_units[-1]
        else:
            raise ValueError(f"Unsupported model_structure: {model_structure}")
    
    def forward(self, inputs):
        """重写 forward 方法以支持域感知结构"""
        if not self.use_domain_aware_structure:
            # 不使用域感知结构时，直接调用父类方法，保持完全一致
            return super().forward(inputs)
        
        # 使用域感知结构时，需要自己处理前向传播
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        
        # 通过 DCNv2 处理，但不应用最终的 fc 层
        dcnv2_output = self._get_dcnv2_features(feature_emb)
        
        # 使用域感知结构生成最终输出
        y_pred = self._generate_multi_tower_output(X, dcnv2_output)
        
        return_dict = {"y_pred": y_pred}
        return return_dict
    
    def _get_dcnv2_features(self, feature_emb):
        """获取 DCNv2 的特征输出（不包含最终的 fc 层）"""
        # 通过 crossnet 处理
        cross_out = self.crossnet(feature_emb)
        
        # 根据不同的 model_structure 组合特征
        if self.model_structure == "crossnet_only":
            final_out = cross_out
        elif self.model_structure == "stacked":
            final_out = self.stacked_dnn(cross_out)
        elif self.model_structure == "parallel":
            dnn_out = self.parallel_dnn(feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        elif self.model_structure == "stacked_parallel":
            final_out = torch.cat([self.stacked_dnn(cross_out), self.parallel_dnn(feature_emb)], dim=-1)
        else:
            raise ValueError(f"Unsupported model_structure: {self.model_structure}")
        
        return final_out
    
    def _generate_multi_tower_output(self, X_features, net_output):
        """生成域感知结构输出，仿照 DCNv3 的实现"""
        # 使用 MultiTowerModule 处理域感知逻辑
        final_logits = self.multi_tower_module(net_output, X_features)
        # 应用输出激活函数
        output = self.output_activation(final_logits)
        return output 