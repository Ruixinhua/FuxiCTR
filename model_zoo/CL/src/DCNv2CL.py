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
DCNv2CL: DCNv2 with Contrastive Learning

基于DCNv2WithMultiTower模型实现的对比学习版本，支持：
1. 个性化特征掩码
2. 特征对齐损失  
3. 字段均匀性损失
4. 距离损失
5. 多种网络结构 (parallel, stacked, crossnet_only)
6. 🎯 多塔(MT)支持（通过继承DCNv2WithMultiTower实现）
"""

import torch
import logging

# 导入DCNv2WithMultiTower模型
from ...DCNv2.src.DCNv2MT import DCNv2WithMultiTower
from .base import ContrastiveLearningBase


class DCNv2CL(DCNv2WithMultiTower, ContrastiveLearningBase):
    """
    DCNv2 with Contrastive Learning
    
    继承自DCNv2WithMultiTower，集成对比学习功能
    支持多种网络结构模式、对比学习和域感知结构的联合训练
    🎯 通过父类支持多塔(MT)结构
    """
    
    def __init__(self, 
                 feature_map, 
                 model_id="DCNv2CL", 
                 gpu=-1,
                 model_structure="parallel",
                 use_low_rank_mixture=False,
                 low_rank=32,
                 num_experts=4,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 stacked_dnn_hidden_units=[], 
                 parallel_dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None,
                 # CL相关参数
                 cl_config=None,
                 # MT相关参数（继承自DCNv2WithMultiTower）
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
        
        # 初始化DCNv2WithMultiTower（包含所有MT相关功能）
        DCNv2WithMultiTower.__init__(self,
                                   feature_map=feature_map,
                                   model_id=model_id,
                                   gpu=gpu,
                                   learning_rate=learning_rate,
                                   model_structure=model_structure,
                                   use_low_rank_mixture=use_low_rank_mixture,
                                   low_rank=low_rank,
                                   num_experts=num_experts,
                                   embedding_dim=embedding_dim,
                                   stacked_dnn_hidden_units=stacked_dnn_hidden_units,
                                   parallel_dnn_hidden_units=parallel_dnn_hidden_units,
                                   dnn_activations=dnn_activations,
                                   num_cross_layers=num_cross_layers,
                                   net_dropout=net_dropout,
                                   batch_norm=batch_norm,
                                   embedding_regularizer=embedding_regularizer,
                                   net_regularizer=net_regularizer,
                                   # MT相关参数
                                   use_domain_aware_structure=use_domain_aware_structure,
                                   tower_hidden_units_list=tower_hidden_units_list,
                                   tower_activation=tower_activation,
                                   tower_l2_reg_list=tower_l2_reg_list,
                                   tower_dropout_list=tower_dropout_list,
                                   use_bn_tower=use_bn_tower,
                                   scene_name=scene_name,
                                   scene_num_shift=scene_num_shift,
                                   use_scene_id_mapping=use_scene_id_mapping,
                                   mapping_feature_name=mapping_feature_name,
                                   mapping_feature_type=mapping_feature_type,
                                   feature2id_dict=feature2id_dict,
                                   default_value=default_value,
                                   feature_map_dict=feature_map_dict,
                                   **kwargs)
        
        logging.info(f"DCNv2CL模型初始化完成。CL配置: {self.cl_config}")
        logging.info(f"使用网络结构: {model_structure}")
        logging.info(f"使用域感知结构(MT): {self.use_domain_aware_structure}")
    
    def _get_dcnv2_logits(self, X):
        """
        获取DCNv2模型的logits（不应用激活函数）
        适配DCNv2WithMultiTower的结构
        
        Args:
            X: 输入特征字典
            
        Returns:
            torch.Tensor: 原始logits
        """
        if not self.use_domain_aware_structure:
            # 非MT模式：获取原始DCNv2的logits
            feature_emb = self.embedding_layer(X, flatten_emb=True)
            cross_out = self.crossnet(feature_emb)
            
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
                raise ValueError(f"Unsupported model structure: {self.model_structure}")
            
            # 获取最终层的logits（不应用激活函数）
            logits = self.fc(final_out)
            
            return logits
        else:
            # MT模式：利用父类的多塔结构获取logits
            feature_emb = self.embedding_layer(X, flatten_emb=True)
            
            # 通过DCNv2处理，获取特征表示（使用父类方法）
            dcnv2_features = self._get_dcnv2_features(feature_emb)
            
            # 通过多塔模块获取原始logits（不应用激活函数）
            logits = self.multi_tower_module(dcnv2_features, X)
            
            return logits
    
    def _get_base_model_logits(self, X, base_model=None, **kwargs):
        """
        实现ContrastiveLearningBase要求的抽象方法
        
        Args:
            X: 输入特征字典
            base_model: 基础模型（在这里忽略，直接使用self）
            **kwargs: 其他参数
            
        Returns:
            torch.Tensor: logits
        """
        return self._get_dcnv2_logits(X)
    
    def forward(self, inputs):
        """
        前向传播，集成对比学习功能，支持MT
        利用父类DCNv2WithMultiTower的forward逻辑
        
        Args:
            inputs: 包含特征和标签的字典
            
        Returns:
            dict: 包含预测结果和中间结果的字典
        """
        X = self.get_inputs(inputs)
        
        if not self.use_domain_aware_structure:
            # 非MT模式：使用父类的基础forward逻辑
            parent_result = super(DCNv2WithMultiTower, self).forward(inputs)
        else:
            # MT模式：使用父类的完整forward逻辑
            parent_result = super().forward(inputs)
        
        return_dict = {"y_pred": parent_result["y_pred"], "logits": self._get_dcnv2_logits(X)}
        
        # 🎯 对比学习相关处理（无论是否使用多塔都支持）
        # 获取组信息（基于is_personalization特征）
        if self.training:
            group_ids = self.get_group_ids(inputs)
            if group_ids is not None:
                return_dict["group_ids"] = group_ids
        
        # 如果启用CL，计算额外的CL组件
        if self.training and (self.feature_alignment_loss_weight > 0 or 
                             self.field_uniformity_loss_weight > 0):
            # 获取各个特征的嵌入用于计算CL损失
            feature_embeddings = self.get_feature_embeddings(self.embedding_layer, X)
            return_dict["feature_embeddings"] = feature_embeddings
        
        # 如果启用个性化掩码，生成对比视图
        # if self.training and self.use_cl_mask:
        #     h1_logits, h2_logits = self.apply_personalization_mask(X, base_model=self, **{})
        #     if h1_logits is not None and h2_logits is not None:
        #         return_dict["h1_logits"] = h1_logits
        #         return_dict["h2_logits"] = h2_logits
        
        return return_dict
    
    def add_loss(self, return_dict, y_true):
        """
        计算包含对比学习的总损失
        
        Args:
            return_dict: forward方法的返回字典
            y_true: 真实标签
            
        Returns:
            torch.Tensor: 总损失
        """
        # 基础DCNv2损失
        base_loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        
        # 如果不在训练模式或没有启用任何CL损失，直接返回基础损失
        if not self.training or not self.use_cl_loss:
            return base_loss
        
        # 提取CL相关的中间结果
        feature_embeddings = return_dict.get("feature_embeddings", None)
        h1_logits = return_dict.get("h1_logits", None)
        h2_logits = return_dict.get("h2_logits", None)
        
        # 🚀 获取组信息（基于is_personalization特征）
        # is_personalization=1: 个性化用户，is_personalization=0或2: 非个性化用户
        group_ids = return_dict.get("group_ids", None)
        
        # 计算完整的CL损失（使用改进的版本）
        total_loss = self.compute_cl_loss(
            base_loss=base_loss,
            feature_embeddings=feature_embeddings,
            h1_logits=h1_logits,  # 个性化视图（教师）
            h2_logits=h2_logits,  # 非个性化视图（学生）
            labels=y_true,
            group_ids=group_ids  # 组标识
        )
        
        return total_loss 