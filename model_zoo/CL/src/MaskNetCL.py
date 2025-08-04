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
MaskNetCL: MaskNet with Contrastive Learning

基于MaskNetWithMultiTower模型实现的对比学习版本，支持：
1. 个性化特征掩码
2. 特征对齐损失  
3. 字段均匀性损失
4. 距离损失
5. SerialMaskNet/ParallelMaskNet结构
6. 🎯 多塔(MT)支持（通过继承MaskNetWithMultiTower实现）
"""

import torch
import torch.nn as nn
import logging

# 导入MaskNetWithMultiTower模型
from ...MaskNet.src.MaskNet import MaskNet
from .base import ContrastiveLearningBase
from fuxictr.pytorch.layers import MultiTowerModule


class MaskNetCL(MaskNet, ContrastiveLearningBase):
    """
    MaskNet with Contrastive Learning
    
    继承自MaskNetWithMultiTower，集成对比学习功能
    支持SerialMaskNet和ParallelMaskNet两种结构
    🎯 通过父类支持多塔(MT)结构
    """
    
    def __init__(self, 
                 feature_map,
                 learning_rate=1e-3,
                 # CL相关参数
                 cl_config=None,
                 # MT相关参数（继承自MaskNetWithMultiTower）
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
        
        # 初始化ContrastiveLearningBase
        ContrastiveLearningBase.__init__(self, cl_config=cl_config, **kwargs)
        MaskNet.__init__(self, feature_map=feature_map, **kwargs)
        # 初始化MaskNetWithMultiTower（包含所有MT相关功能）
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

    def _forward_with_logits(self, X):
        """
        一次性计算logits和y_pred，避免重复计算
        
        Args:
            X: 输入特征字典
            
        Returns:
            tuple: (logits, y_pred)
        """
        feature_emb = self.embedding_layer(X)
        
        # 应用嵌入层归一化
        if self.emb_norm is not None:
            feat_list = feature_emb.chunk(self.num_fields, dim=1)
            V_hidden = torch.cat([self.emb_norm[i](feat) for i, feat in enumerate(feat_list)], dim=1)
        else:
            V_hidden = feature_emb
        
        V_emb_flat = feature_emb.flatten(start_dim=1)
        V_hidden_flat = V_hidden.flatten(start_dim=1)
        
        # 获取 MaskNet 的输出（已包含激活函数）
        y_pred = self.mask_net(V_emb_flat, V_hidden_flat)
        
        # 为了获取logits，我们需要手动重新计算不包含激活函数的输出
        if hasattr(self.mask_net, 'hidden_units'):
            # SerialMaskNet
            v_out = V_hidden_flat
            for idx in range(len(self.mask_net.hidden_units) - 1):
                v_out = self.mask_net.mask_blocks[idx](V_emb_flat, v_out)
            
            # 获取最后一个Linear层的输出作为logits（不应用激活函数）
            if hasattr(self.mask_net, 'fc') and self.mask_net.fc is not None:
                # 遍历fc的所有层，找到最后一个Linear层
                logits = v_out
                for layer in self.mask_net.fc:
                    if isinstance(layer, nn.Linear):
                        logits = layer(logits)
                        break  # 第一个Linear层就是输出层
            else:
                logits = v_out
                
        elif hasattr(self.mask_net, 'num_blocks'):
            # ParallelMaskNet
            block_out = []
            for i in range(self.mask_net.num_blocks):
                block_out.append(self.mask_net.mask_blocks[i](V_emb_flat, V_hidden_flat))
            concat_out = torch.cat(block_out, dim=-1)
            
            # 获取dnn中最后一个Linear层的输出作为logits
            if hasattr(self.mask_net, 'dnn'):
                # 手动应用dnn的所有层，直到最后一个Linear层（但不包括激活函数）
                logits = concat_out
                for layer in self.mask_net.dnn.mlp:
                    if isinstance(layer, nn.Linear):
                        logits = layer(logits)
                        # 如果这是输出层（输出维度为1），则停止
                        if logits.shape[-1] == 1:
                            break
                    elif not isinstance(layer, nn.Sigmoid):
                        # 应用除了Sigmoid之外的所有层（如ReLU、Dropout等）
                        logits = layer(logits)
            else:
                logits = concat_out
        else:
            # 其他情况，假设没有激活函数
            logits = y_pred
        
        return logits, y_pred

    def _forward_with_logits_mt(self, X):
        """
        MT模式下一次性计算logits和y_pred，避免重复计算
        
        Args:
            X: 输入特征字典
            
        Returns:
            tuple: (logits, y_pred)
        """
        feature_emb = self.embedding_layer(X)
        
        # 应用层归一化（与父类保持一致）
        if self.emb_norm is not None:
            feat_list = feature_emb.chunk(self.num_fields, dim=1)
            V_hidden = torch.cat([self.emb_norm[i](feat) for i, feat in enumerate(feat_list)], dim=1)
        else:
            V_hidden = feature_emb
        
        # 通过MaskNet处理，但不应用最终的输出层
        masknet_features = self._get_masknet_features(feature_emb.flatten(start_dim=1), V_hidden.flatten(start_dim=1))
        
        # 通过多塔结构生成 logits（MultiTowerModule 输出未激活的 logits）
        final_logits = self.multi_tower_module(masknet_features, X)
        
        # 应用输出激活函数得到 y_pred
        y_pred = self.output_activation(final_logits)
        
        return final_logits, y_pred

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

    def forward(self, inputs):
        """
        前向传播，集成对比学习功能，支持MT
        避免重复计算，直接从中间结果获取logits
        
        Args:
            inputs: 包含特征和标签的字典
            
        Returns:
            dict: 包含预测结果和中间结果的字典
        """
        X = self.get_inputs(inputs)
        
        if not self.use_domain_aware_structure:
            # 非MT模式：一次性计算logits和y_pred，避免重复计算
            logits, y_pred = self._forward_with_logits(X)
            return_dict = {"y_pred": y_pred, "logits": logits}
        else:
            # MT模式：一次性计算logits和y_pred，避免调用父类forward导致的重复计算
            logits, y_pred = self._forward_with_logits_mt(X)
            return_dict = {"y_pred": y_pred, "logits": logits}
        
        # 获取组信息（基于is_personalization特征）
        if self.training and 'is_personalization' in inputs:
            group_ids = self.get_group_ids(inputs)
            if group_ids is not None:
                return_dict["group_ids"] = group_ids
        
        # 如果启用CL，从已计算的embedding中提取特征嵌入，避免重复计算
        if self.training and (self.feature_alignment_loss_weight > 0 or
                             self.field_uniformity_loss_weight > 0):
            # 复用已经计算的embedding，避免重新计算
            feature_emb = self.embedding_layer(X)
            feature_embeddings = self._extract_feature_embeddings_from_tensor(feature_emb, X)
            return_dict["feature_embeddings"] = feature_embeddings
        
        # 如果启用个性化掩码，生成对比视图，复用已计算的logits
        if self.training and self.use_cl_mask:
            # h1使用已计算的logits，避免重复计算
            h1_logits = logits
            
            # h2只需要计算非个性化视图的logits
            h2_logits = self._compute_non_personalized_logits(X)
            
            if h1_logits is not None and h2_logits is not None:
                return_dict["h1_logits"] = h1_logits
                return_dict["h2_logits"] = h2_logits
        
        return return_dict
    
    def _extract_feature_embeddings_from_tensor(self, feature_emb, X):
        """
        从已计算的特征嵌入张量中提取各个特征的嵌入，避免重复计算
        
        Args:
            feature_emb: 已计算的特征嵌入张量 [batch_size, num_fields, embedding_dim]
            X: 输入特征字典
            
        Returns:
            dict: {feature_name: embedding_tensor}
        """
        feature_embeddings = {}
        feature_names = list(X.keys())
        
        # 将特征嵌入按字段分割
        if feature_emb.dim() == 3:  # [batch_size, num_fields, embedding_dim]
            feat_list = feature_emb.chunk(len(feature_names), dim=1)
            for i, feature_name in enumerate(feature_names):
                if i < len(feat_list):
                    feature_embeddings[feature_name] = feat_list[i].squeeze(1)
        else:
            # 如果是展平的，需要重新计算各个特征的嵌入
            feature_embeddings = self.get_feature_embeddings(self.embedding_layer, X)
            
        return feature_embeddings
    
    def _compute_non_personalized_logits(self, X):
        """
        计算非个性化视图的logits
        
        Args:
            X: 输入特征字典
            
        Returns:
            torch.Tensor: 非个性化视图的logits
        """
        if not self.use_cl_mask or self.mask_type != 'Personalisation':
            return None
        
        # 生成非个性化视图
        non_personalized_X = {}
        for feature_name, feature_value in X.items():
            if feature_name in self.personalization_feature_list:
                # 使用置零策略（可以根据需要扩展其他策略）
                non_personalized_X[feature_name] = torch.zeros_like(feature_value)
            else:
                non_personalized_X[feature_name] = feature_value
        
        # 计算非个性化视图的logits
        if not self.use_domain_aware_structure:
            # 非MT模式
            h2_logits, _ = self._forward_with_logits(non_personalized_X)
        else:
            # MT模式
            h2_logits, _ = self._forward_with_logits_mt(non_personalized_X)
        
        return h2_logits
    
    def add_loss(self, return_dict, y_true):
        """
        计算包含对比学习的总损失
        
        Args:
            return_dict: forward方法的返回字典
            y_true: 真实标签
            
        Returns:
            torch.Tensor: 总损失
        """
        # 基础MaskNet损失
        base_loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        
        # 如果不在训练模式或没有启用CL，直接返回基础损失
        if not self.training or not self.use_cl_loss:
            return base_loss
        
        # 提取CL相关的中间结果
        feature_embeddings = return_dict.get("feature_embeddings", None)
        h1_logits = return_dict.get("h1_logits", None)
        h2_logits = return_dict.get("h2_logits", None)
        group_ids = return_dict.get("group_ids", None)
        
        # 计算完整的CL损失
        total_loss = self.compute_cl_loss(
            base_loss=base_loss,
            feature_embeddings=feature_embeddings,
            h1_logits=h1_logits,
            h2_logits=h2_logits,
            labels=y_true,
            group_ids=group_ids
        )
        
        return total_loss 