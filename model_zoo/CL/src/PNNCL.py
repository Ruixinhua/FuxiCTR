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
PNNCL: PNN with Contrastive Learning

基于PNN模型实现的对比学习版本，支持：
1. 个性化特征掩码
2. 特征对齐损失  
3. 字段均匀性损失
4. 距离损失
"""

import torch
import torch.nn.functional as F
import logging
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, InnerProductInteraction, MultiTowerModule
from .base import ContrastiveLearningBase


class PNNCL(BaseModel, ContrastiveLearningBase):
    """
    PNN with Contrastive Learning
    
    结合PNN的基础架构和对比学习的增强功能
    """
    
    def __init__(self, 
                 feature_map, 
                 model_id="PNNCL", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0,
                 batch_norm=False,
                 product_type="inner",
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 # CL相关参数
                 cl_config=None,
                 # MT相关参数
                 use_domain_aware_structure=False,
                 **kwargs):
        
        # 🔧 将所有参数传递给ContrastiveLearningBase，支持autotuner扁平化参数
        ContrastiveLearningBase.__init__(self, cl_config=cl_config, **kwargs)
        
        # 初始化BaseModel
        BaseModel.__init__(self, 
                          feature_map, 
                          model_id=model_id, 
                          gpu=gpu, 
                          embedding_regularizer=embedding_regularizer, 
                          net_regularizer=net_regularizer,
                          **kwargs)
        
        # 🎯 MT相关配置
        self.use_domain_aware_structure = use_domain_aware_structure
        if self.use_domain_aware_structure:
            # 存储MT配置供后续使用
            self._mt_params = {
                'input_dim': feature_map.num_fields * embedding_dim,
                'hidden_units': hidden_units
            }
        
        # PNN模型结构
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        
        if product_type != "inner":
            raise NotImplementedError("product_type={} has not been implemented.".format(product_type))
            
        self.remove_feature = kwargs.get("remove_feature")
        if self.remove_feature is not None:
            num_fields = feature_map.num_fields - 1
        else:
            num_fields = feature_map.num_fields
            
        self.inner_product_layer = InnerProductInteraction(num_fields, output="inner_product")
        input_dim = int(num_fields * (num_fields - 1) / 2) + num_fields * embedding_dim
        
        self.dnn = MLP_Block(input_dim=input_dim,
                            output_dim=1, 
                            hidden_units=hidden_units,
                            hidden_activations=hidden_activations,
                            output_activation=self.output_activation,
                            dropout_rates=net_dropout, 
                            batch_norm=batch_norm)
        
        # 保存模型参数供CL使用
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        
        # 🎯 如果启用域感知结构(MT)，初始化多塔模块
        if self.use_domain_aware_structure:
            # 🔧 创建专门用于特征提取的DNN（不包含最终输出层）
            self.feature_dnn = MLP_Block(
                input_dim=input_dim,
                output_dim=None,  # 不包含最终输出层
                hidden_units=hidden_units,
                hidden_activations=hidden_activations,
                output_activation=None,  # 特征提取不需要输出激活
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            )
            self._init_multi_tower_structure(kwargs)
        
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
        
        logging.info(f"PNNCL模型初始化完成。CL配置: {self.cl_config}")
        logging.info(f"使用域感知结构(MT): {self.use_domain_aware_structure}")
    
    def _init_multi_tower_structure(self, model_kwargs):
        """
        初始化多塔结构
        """
        tower_hidden_units_list = model_kwargs.get('tower_hidden_units_list')
        if tower_hidden_units_list is None:
            raise ValueError("`tower_hidden_units_list` must be specified when using multi-tower structure.")
        
        # 计算PNN输出维度
        dnn_hidden_units = self._mt_params['hidden_units']
        pnn_input_dim = self._mt_params['input_dim']
        
        # PNN的输出维度是最后一个隐藏层的维度
        if dnn_hidden_units:
            tower_input_dim = dnn_hidden_units[-1]
        else:
            tower_input_dim = pnn_input_dim
        
        # 使用MultiTowerModule
        self.multi_tower_module = MultiTowerModule(
            input_dim=tower_input_dim,
            tower_hidden_units_list=tower_hidden_units_list,
            tower_activation=model_kwargs.get('tower_activation', 'ReLU'),
            tower_l2_reg_list=model_kwargs.get('tower_l2_reg_list'),
            tower_dropout_list=model_kwargs.get('tower_dropout_list'),
            use_bn_tower=model_kwargs.get('use_bn_tower', True),
            scene_name=model_kwargs.get('scene_name', 'scene_id'),
            scene_num_shift=model_kwargs.get('scene_num_shift', 1),
            use_scene_id_mapping=model_kwargs.get('use_scene_id_mapping', False),
            mapping_feature_name=model_kwargs.get('mapping_feature_name'),
            mapping_feature_type=model_kwargs.get('mapping_feature_type'),
            feature2id_dict=model_kwargs.get('feature2id_dict'),
            default_value=model_kwargs.get('default_value'),
            feature_map_dict=model_kwargs.get('feature_map_dict')
        )
        
        # 确保在正确的设备上
        self.multi_tower_module.to(self.device)
        
        logging.info(f"PNNCL多塔结构初始化完成，输入维度: {tower_input_dim}")
    
    def _get_pnn_logits(self, X):
        """
        获取PNN模型的logits（不应用激活函数）
        优化版本：消除重复代码
        
        Args:
            X: 输入特征字典
            
        Returns:
            torch.Tensor: 原始logits
        """
        # 🔧 统一的特征预处理（消除重复代码）
        if self.remove_feature is not None and self.remove_feature in X:
            X = {k: v for k, v in X.items() if k != self.remove_feature}
            
        feature_emb = self.embedding_layer(X)
        inner_products = self.inner_product_layer(feature_emb)
        dense_input = torch.cat([feature_emb.flatten(start_dim=1), inner_products], dim=1)
        
        if not self.use_domain_aware_structure:
            # 非MT模式：通过DNN获取logits
            dnn_layers = list(self.dnn.mlp.children())
            if len(dnn_layers) > 0:
                # 应用除了最后激活函数之外的所有层
                x = dense_input
                for layer in dnn_layers[:-1]:  # 排除最后的激活函数
                    x = layer(x)
                logits = x
            else:
                logits = dense_input
                
            return logits
        else:
            # MT模式：通过多塔结构获取logits
            pnn_features = self._get_pnn_features(dense_input)
            final_logits = self.multi_tower_module(pnn_features, X)
            return final_logits
    
    def _get_pnn_features(self, dense_input):
        """
        获取PNN的特征输出（不包含最终输出层）
        注：此方法只在MT模式下调用
        """
        # 多塔模式：使用专门的feature_dnn
        return self.feature_dnn(dense_input)
    
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
        return self._get_pnn_logits(X)
    
    def forward(self, inputs):
        """
        前向传播，集成对比学习功能，支持MT
        避免重复计算，优化性能
        
        Args:
            inputs: 包含特征和标签的字典
            
        Returns:
            dict: 包含预测结果和中间结果的字典
        """
        X = self.get_inputs(inputs)
        
        # 一次性计算logits和y_pred，避免重复计算
        base_logits = self._get_pnn_logits(X)
        y_pred = self.output_activation(base_logits)
        
        return_dict = {"y_pred": y_pred, "logits": base_logits}
        
        # 获取组信息（基于is_personalization特征）
        if self.training:
            group_ids = self.get_group_ids(inputs)
            if group_ids is not None:
                return_dict["group_ids"] = group_ids
        
        # 如果启用CL，从已计算的embedding中提取特征嵌入，避免重复计算
        if self.training and (self.feature_alignment_loss_weight > 0 or 
                             self.field_uniformity_loss_weight > 0):
            # 复用已经计算的embedding，避免重新计算
            if self.remove_feature is not None and self.remove_feature in X:
                X_processed = {k: v for k, v in X.items() if k != self.remove_feature}
            else:
                X_processed = X
            
            feature_emb = self.embedding_layer(X_processed)
            feature_embeddings = self._extract_feature_embeddings_from_tensor(feature_emb, X_processed)
            return_dict["feature_embeddings"] = feature_embeddings
        
        # 如果启用个性化掩码，生成对比视图，复用已计算的logits
        if self.training and self.use_cl_mask:
            # h1使用已计算的logits，避免重复计算
            h1_logits = base_logits
            
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
        return self._get_pnn_logits(non_personalized_X)
    
    def add_loss(self, return_dict, y_true):
        """
        计算包含对比学习的总损失
        
        Args:
            return_dict: forward方法的返回字典
            y_true: 真实标签
            
        Returns:
            torch.Tensor: 总损失
        """
        # 基础PNN损失
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