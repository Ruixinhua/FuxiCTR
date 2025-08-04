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
DCNv3CL: DCNv3 with Contrastive Learning

基于DCNv3模型实现的对比学习版本，支持：
1. 个性化特征掩码
2. 特征对齐损失  
3. 字段均匀性损失
4. 距离损失
5. 多塔结构 (MT)
"""

import torch
import logging


from ...DCNv3.src import DCNv3
from .base import ContrastiveLearningBase


class DCNv3CL(DCNv3, ContrastiveLearningBase):
    """
    DCNv3 with Contrastive Learning
    
    结合DCNv3的深度交叉网络架构和对比学习的增强功能
    支持多塔结构和对比学习的联合训练
    """
    
    def __init__(self,
                 feature_map,
                 model_id="DCNv3CL",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_deep_cross_layers=4,
                 num_shallow_cross_layers=4,
                 deep_net_dropout=0.1,
                 shallow_net_dropout=0.3,
                 layer_norm=True,
                 batch_norm=False,
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 use_domain_aware_structure=False,
                 # CL相关参数
                 cl_config=None,
                 **kwargs):
        
        # 初始化ContrastiveLearningBase
        ContrastiveLearningBase.__init__(self, cl_config=cl_config, **kwargs)
        
        # 初始化DCNv3
        DCNv3.__init__(self,
                      feature_map=feature_map,
                      model_id=model_id,
                      gpu=gpu,
                      learning_rate=learning_rate,
                      embedding_dim=embedding_dim,
                      num_deep_cross_layers=num_deep_cross_layers,
                      num_shallow_cross_layers=num_shallow_cross_layers,
                      deep_net_dropout=deep_net_dropout,
                      shallow_net_dropout=shallow_net_dropout,
                      layer_norm=layer_norm,
                      batch_norm=batch_norm,
                      num_heads=num_heads,
                      embedding_regularizer=embedding_regularizer,
                      net_regularizer=net_regularizer,
                      use_domain_aware_structure=use_domain_aware_structure,
                      **kwargs)
        
        logging.info(f"DCNv3CL模型初始化完成。CL配置: {self.cl_config}")
        logging.info(f"使用域感知结构(MT): {self.use_domain_aware_structure}")
    
    def _get_dcnv3_logits(self, X):
        """
        获取DCNv3模型的logits（不应用激活函数）
        
        Args:
            X: 输入特征字典
            
        Returns:
            torch.Tensor: 原始logits
        """
        feature_emb = self.embedding_layer(X)

        if self.use_domain_aware_structure:
            xld_intermediate = self.ECN(feature_emb)
            xls_intermediate = self.LCN(feature_emb)

            xld_flat = xld_intermediate.view(xld_intermediate.size(0), -1)
            xls_flat = xls_intermediate.view(xls_intermediate.size(0), -1)

            logits_xld = self._generate_domain_aware_logits_pytorch(X, xld_flat)
            logits_xls = self._generate_domain_aware_logits_pytorch(X, xls_flat)
        else:
            logits_xld = self.ECN(feature_emb).mean(dim=1)
            logits_xls = self.LCN(feature_emb).mean(dim=1)
        
        # 合并ECN和LCN的输出作为最终logits
        logits = (logits_xld + logits_xls) * 0.5
        
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
        return self._get_dcnv3_logits(X)
    
    def forward(self, inputs):
        """
        前向传播，集成对比学习功能
        
        Args:
            inputs: 包含特征和标签的字典
            
        Returns:
            dict: 包含预测结果和中间结果的字典
        """
        X = self.get_inputs(inputs)
        
        # 获取DCNv3的完整前向传播结果
        dcnv3_output = super().forward(inputs)
        
        # 从父类的实例变量中获取logits，避免重复计算
        # 父类DCNv3在forward中已经计算了self.logits_xld和self.logits_xls
        base_logits = (self.logits_xld + self.logits_xls) * 0.5
        
        # 添加logits到返回字典
        dcnv3_output["logits"] = base_logits
        
        # 获取组信息（基于is_personalization特征）
        if self.training:
            group_ids = self.get_group_ids(inputs)
            if group_ids is not None:
                dcnv3_output["group_ids"] = group_ids
        
        # 如果启用CL，从已计算的embedding中提取特征嵌入，避免重复计算
        if self.training and (self.feature_alignment_loss_weight > 0 or 
                             self.field_uniformity_loss_weight > 0):
            # 复用已经计算的embedding，避免重新计算
            feature_emb = self.embedding_layer(X)
            feature_embeddings = self._extract_feature_embeddings_from_tensor(feature_emb, X)
            dcnv3_output["feature_embeddings"] = feature_embeddings
        
        # 如果启用个性化掩码，生成对比视图，复用已计算的logits
        if self.training and self.use_cl_mask:
            # h1使用已计算的logits，避免重复计算
            h1_logits = base_logits
            
            # h2只需要计算非个性化视图的logits
            h2_logits = self._compute_non_personalized_logits(X)
            
            if h1_logits is not None and h2_logits is not None:
                dcnv3_output["h1_logits"] = h1_logits
                dcnv3_output["h2_logits"] = h2_logits
        
        return dcnv3_output
    
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
        return self._get_dcnv3_logits(non_personalized_X)
    
    def add_loss(self, return_dict, y_true):
        """
        计算包含对比学习的总损失
        
        Args:
            return_dict: forward方法的返回字典
            y_true: 真实标签
            
        Returns:
            torch.Tensor: 总损失
        """
        # 获取DCNv3的基础损失（包含深浅层损失）
        dcnv3_base_loss = super().add_loss(return_dict, y_true)
        
        # 如果不在训练模式或没有启用任何CL损失，直接返回DCNv3基础损失
        if not self.training or not self.use_cl_loss:
            return dcnv3_base_loss
        
        # 提取CL相关的中间结果
        feature_embeddings = return_dict.get("feature_embeddings", None)
        h1_logits = return_dict.get("h1_logits", None)
        h2_logits = return_dict.get("h2_logits", None)
        group_ids = return_dict.get("group_ids", None)
        
        # 🚀 使用统一的compute_cl_loss方法计算完整的CL损失
        total_loss = self.compute_cl_loss(
            base_loss=dcnv3_base_loss,
            feature_embeddings=feature_embeddings,
            h1_logits=h1_logits,
            h2_logits=h2_logits,
            labels=y_true,
            group_ids=group_ids
        )
        
        return total_loss 