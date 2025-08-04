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
MaskNet with MultiTowerModule 示例
这个文件展示了如何通过继承原始 MaskNet 来集成 MultiTowerModule 组件
遵循开闭原则，最大程度保持与原始 MaskNet 的一致性
"""

import torch
import logging
from fuxictr.pytorch.layers import MultiTowerModule
from .MaskNet import MaskNet


class MaskNetWithMultiTower(MaskNet):
    """
    MaskNet with MultiTowerModule support
    
    继承自原始 MaskNet 类，通过扩展的方式支持域感知结构，
    保持与原始 MaskNet 的完全兼容性。
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
        super(MaskNetWithMultiTower, self).__init__(feature_map, **kwargs)
        # 提取并移除域感知结构相关参数，避免传递给父类
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
        
        # 存储参数用于域感知结构初始化
        if use_domain_aware_structure:
            self._masknet_params = {
                'input_dim': feature_map.num_fields * kwargs.get('embedding_dim', 16),
                'hidden_units': kwargs.get('dnn_hidden_units', [256, 128, 64])
            }
        # 如果使用域感知结构，初始化多塔模块（必须在父类初始化后进行，确保设备正确）
        if use_domain_aware_structure:
            # 传递额外参数给多塔初始化方法
            self._init_multi_tower_structure(multi_tower_params, kwargs)
            # 确保MultiTowerModule在正确的设备上
            self.multi_tower_module.to(self.device)
        logging.info(f"MaskNetWithMultiTower initialized. Use domain-aware structure: {use_domain_aware_structure}")
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
    
    def _init_multi_tower_structure(self, multi_tower_params, model_kwargs):
        """初始化多塔结构，仿照 DCNv3 的实现"""
        tower_hidden_units_list = multi_tower_params['tower_hidden_units_list']
        if tower_hidden_units_list is None:
            raise ValueError("`tower_hidden_units_list` must be specified when using multi-tower structure.")
        
        # 计算 MaskNet 输出维度，根据不同类型的MaskNet计算
        dnn_hidden_units = self._masknet_params['hidden_units']
        masknet_input_dim = self._masknet_params['input_dim']
        
        # 根据mask_net的类型计算正确的输入维度
        if hasattr(self.mask_net, 'hidden_units'):
            # SerialMaskNet - 输出维度是最后一个隐藏层的维度
            if dnn_hidden_units:
                tower_input_dim = dnn_hidden_units[-1]
            else:
                tower_input_dim = masknet_input_dim
        elif hasattr(self.mask_net, 'num_blocks'):
            # ParallelMaskNet - 输出维度是 block_dim * num_blocks
            # 从ParallelMaskNet的构造可知，每个MaskBlock的输出维度是block_dim
            # 我们需要从MaskNet的初始化参数中获取block_dim
            block_dim = model_kwargs.get('parallel_block_dim', 64)  # 默认值64
            tower_input_dim = block_dim * self.mask_net.num_blocks
        else:
            # 默认情况
            if dnn_hidden_units:
                tower_input_dim = dnn_hidden_units[-1]
            else:
                tower_input_dim = masknet_input_dim
        
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
    
    def forward(self, inputs):
        """重写 forward 方法以支持域感知结构"""
        if not self.use_domain_aware_structure:
            # 不使用域感知结构时，直接调用父类方法，保持完全一致
            return super().forward(inputs)
        
        # 使用域感知结构时，需要自己处理前向传播
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        
        # 应用层归一化，与父类保持一致
        if self.emb_norm is not None:
            feat_list = feature_emb.chunk(self.num_fields, dim=1)
            V_hidden = torch.cat([self.emb_norm[i](feat) for i, feat in enumerate(feat_list)], dim=1)
        else:
            V_hidden = feature_emb
        
        # 通过 MaskNet 处理，但不应用最终的输出层
        masknet_output = self._get_masknet_features(feature_emb.flatten(start_dim=1), V_hidden.flatten(start_dim=1))
        
        # 使用域感知结构生成最终输出
        y_pred = self._generate_multi_tower_output(X, masknet_output)
        
        return_dict = {"y_pred": y_pred}
        return return_dict
    
    def _get_masknet_features(self, V_emb, V_hidden):
        """获取 MaskNet 的特征输出（不包含最终输出层）"""
        # 检查 mask_net 的类型并相应处理
        if hasattr(self.mask_net, 'hidden_units'):
            # SerialMaskNet - 执行所有MaskBlock但跳过最终的fc层
            v_out = V_hidden
            for idx in range(len(self.mask_net.hidden_units) - 1):
                v_out = self.mask_net.mask_blocks[idx](V_emb, v_out)
            # 返回MaskBlock的输出，不经过fc层
            return v_out
        elif hasattr(self.mask_net, 'num_blocks'):
            # ParallelMaskNet - 返回mask_blocks的连接输出，跳过dnn层
            block_out = []
            for i in range(self.mask_net.num_blocks):
                block_out.append(self.mask_net.mask_blocks[i](V_emb, V_hidden))
            concat_out = torch.cat(block_out, dim=-1)
            # 直接返回concat_out，避免进入dnn（因为dnn会降维到1）
            return concat_out
        else:
            raise ValueError(f"Unsupported mask_net type: {type(self.mask_net)}")
    
    def _generate_multi_tower_output(self, X_features, net_output):
        """生成域感知结构输出，仿照 DCNv3 的实现"""
        # 使用 MultiTowerModule 处理域感知逻辑
        final_logits = self.multi_tower_module(net_output, X_features)
        # 应用输出激活函数
        output = self.output_activation(final_logits)
        return output


