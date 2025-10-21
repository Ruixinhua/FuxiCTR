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
对比学习基础组件

提供通用的对比学习方法，包括：
1. 特征掩码 (Feature Masking)
2. 特征对齐损失 (Feature Alignment Loss) 
3. 字段均匀性损失 (Field Uniformity Loss)
4. 距离损失 (Distance Loss)
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
import logging



class ContrastiveLearningBase(ABC):
    """
    对比学习基础类
    
    提供所有CL模型共享的基础功能：
    - 特征掩码生成
    - 各种CL损失计算
    - 统一的CL配置接口
    """
    
    def __init__(self, cl_config=None, **kwargs):
        """
        初始化ContrastiveLearningBase
        
        Args:
            cl_config (dict): CL配置参数 (传统方式)
            **kwargs: 支持从顶级参数读取CL配置 (autotuner兼容)
        """
        # 🔧 支持autotuner的扁平化参数结构
        # 如果kwargs中包含CL参数，优先使用；否则使用cl_config字典
        self.cl_config = cl_config or {}
        
        # 🎯 参数读取优先级：kwargs > cl_config > default
        def get_param(key, default_value):
            return kwargs.get(key, self.cl_config.get(key, default_value))
        
        self.personalization_feature_list = get_param("personalization_feature_list", [])

        # 🔧 向后兼容性
        self.use_personalisation = True if len(self.personalization_feature_list) else False

        if len(self.personalization_feature_list) == 0 and self.use_personalisation:
            logging.warning("personalization_feature_list为空，但use_personalisation=True")

        
        self.mask_type = get_param('mask_type', 'Personalisation')
        self.use_cl_mask = get_param('use_cl_mask', False)
        self.keep_prob = get_param('keep_prob', 1.0)
        
        # 🔧 损失权重参数
        self.base_loss_weight = get_param('base_loss_weight', 1.0)
        self.feature_alignment_loss_weight = get_param('feature_alignment_loss_weight', 0.0)
        self.field_uniformity_loss_weight = get_param('field_uniformity_loss_weight', 0.0)
        self.distance_loss_weight = get_param('distance_loss_weight', 0.0)
        
        # 🔧 内存优化参数
        self.max_pairs_for_alignment = get_param('max_pairs_for_alignment', 50000)
        self.chunk_size_for_alignment = get_param('chunk_size_for_alignment', 256)
        
        # 🚀 新增的CL损失类型参数
        self.knowledge_distillation_loss_weight = get_param('knowledge_distillation_loss_weight', 0.0)
        self.group_aware_loss_weight = get_param('group_aware_loss_weight', 0.0)
        self.mask_strategy = get_param('mask_strategy', 'zero')  # 'zero', 'noise', 'dropout'
        self.mask_noise_std = get_param('mask_noise_std', 0.1)
        self.mask_dropout_rate = get_param('mask_dropout_rate', 0.3)
        self.temperature = get_param('temperature', 4.0)  # 知识蒸馏温度参数

        self.use_cl_loss = (self.feature_alignment_loss_weight > 0 or
                            self.field_uniformity_loss_weight > 0 or
                            self.distance_loss_weight > 0 or
                            self.knowledge_distillation_loss_weight > 0 or
                            self.group_aware_loss_weight > 0)
        logging.info(f"Use CL Loss: {self.use_cl_loss}")
        # 损失缓存
        self.feature_alignment_loss = None
        self.field_uniformity_loss = None
        self.distance_loss = None
        self.knowledge_distillation_loss = None
        self.group_aware_loss = None
        
        if self.use_personalisation:
            self._setup_personalization()
            
    def _setup_personalization(self):
        """设置个性化相关参数"""
        if not self.personalization_feature_list:
            logging.warning("personalization_feature_list为空，但use_personalisation=True")

    def get_feature_embeddings(self, embedding_layer, X, feature_names=None):
        """
        获取单个特征的嵌入表示
        
        Args:
            embedding_layer: 嵌入层
            X: 输入特征字典
            feature_names: 特征名称列表，如果为None则使用所有特征
            
        Returns:
            dict: {feature_name: embedding_tensor}
        """
        feature_embeddings = {}
        
        if feature_names is None:
            feature_names = list(X.keys())
            
        for feature in feature_names:
            if feature in X:
                # 为单个特征创建输入
                single_feature_input = {feature: X[feature]}
                # 获取该特征的嵌入
                feature_emb = embedding_layer(single_feature_input)
                feature_embeddings[feature] = feature_emb
                
        return feature_embeddings
    
    def sum_unique_pairwise_distances(self, tensor):
        """
        计算张量中所有唯一成对元素的L2距离之和 (内存优化版本)
        
        Args:
            tensor: 输入张量 [batch_size, feature_dim]
            
        Returns:
            tuple: (sum_distances, n_pairs)
        """
        batch_size = tensor.size(0)
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype), \
                   torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
        
        # 内存优化参数
        max_pairs = getattr(self, 'max_pairs_for_alignment', 50000)  # 最大成对数量
        chunk_size = getattr(self, 'chunk_size_for_alignment', 256)   # 分块大小
        
        # 计算总的成对数量
        total_pairs = batch_size * (batch_size - 1) // 2
        
        # 如果成对数量超过阈值，使用采样策略
        if total_pairs > max_pairs:
            return self._compute_sampled_pairwise_distances(tensor, max_pairs)
        
        # 如果batch_size较大，使用分块计算
        if batch_size > chunk_size:
            return self._compute_chunked_pairwise_distances(tensor, chunk_size)
        
        # 原始计算方法（适用于小batch）
        return self._compute_full_pairwise_distances(tensor)
    
    def _compute_full_pairwise_distances(self, tensor):
        """原始的完整成对距离计算"""
        batch_size = tensor.size(0)
        n_pairs = batch_size * (batch_size - 1) // 2
        
        # 创建上三角掩码
        i, j = torch.meshgrid(torch.arange(batch_size, device=tensor.device), 
                             torch.arange(batch_size, device=tensor.device), indexing='ij')
        mask = i < j
        
        # 获取唯一对
        elements_i = tensor[i[mask]]
        elements_j = tensor[j[mask]]
        
        # 计算L2距离
        distances = torch.norm(elements_i - elements_j, dim=-1)
        sum_distances = torch.sum(distances)
        
        return sum_distances, torch.tensor(n_pairs, device=tensor.device, dtype=tensor.dtype)
    
    def _compute_sampled_pairwise_distances(self, tensor, max_pairs):
        """采样策略的成对距离计算"""
        batch_size = tensor.size(0)
        
        # 随机采样成对索引
        all_pairs = []
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                all_pairs.append((i, j))
        
        # 随机选择max_pairs个成对
        import random
        if len(all_pairs) > max_pairs:
            sampled_pairs = random.sample(all_pairs, max_pairs)
        else:
            sampled_pairs = all_pairs
        
        # 计算采样成对的距离
        sum_distances = 0.0
        for i, j in sampled_pairs:
            distance = torch.norm(tensor[i] - tensor[j])
            sum_distances += distance
        
        return sum_distances, torch.tensor(len(sampled_pairs), device=tensor.device, dtype=tensor.dtype)
    
    def _compute_chunked_pairwise_distances(self, tensor, chunk_size):
        """分块计算的成对距离计算"""
        batch_size = tensor.size(0)
        total_sum = 0.0
        total_pairs = 0
        
        # 分块处理
        for start_i in range(0, batch_size, chunk_size):
            end_i = min(start_i + chunk_size, batch_size)
            chunk_i = tensor[start_i:end_i]
            
            for start_j in range(start_i, batch_size, chunk_size):
                end_j = min(start_j + chunk_size, batch_size)
                chunk_j = tensor[start_j:end_j]
                
                # 计算chunk间的距离
                chunk_sum, chunk_pairs = self._compute_chunk_distances(
                    chunk_i, chunk_j, start_i, start_j, end_i, end_j
                )
                total_sum += chunk_sum
                total_pairs += chunk_pairs
        
        return total_sum, torch.tensor(total_pairs, device=tensor.device, dtype=tensor.dtype)
    
    def _compute_chunk_distances(self, chunk_i, chunk_j, start_i, start_j, end_i, end_j):
        """计算两个chunk之间的距离"""
        sum_distances = 0.0
        pairs_count = 0
        
        for i, emb_i in enumerate(chunk_i):
            global_i = start_i + i
            j_start = max(0, start_j - start_i) if start_i == start_j else 0
            j_start = max(j_start, i + 1) if start_i == start_j else j_start
            
            for j in range(j_start, len(chunk_j)):
                global_j = start_j + j
                if global_i < global_j:  # 只计算上三角
                    distance = torch.norm(emb_i - chunk_j[j])
                    sum_distances += distance
                    pairs_count += 1
        
        return sum_distances, pairs_count
    
    def compute_feature_alignment_loss(self, feature_embeddings):
        """
        计算特征对齐损失
        
        Args:
            feature_embeddings: {feature_name: embedding_tensor}
            
        Returns:
            torch.Tensor: 特征对齐损失
        """
        total_distance = 0.0
        total_pairs = 0.0
        
        for feature_name, feature_emb in feature_embeddings.items():
            # feature_emb shape: [batch_size, embedding_dim]
            if feature_emb.dim() > 2:
                feature_emb = feature_emb.view(feature_emb.size(0), -1)
                
            sum_distances, n_pairs = self.sum_unique_pairwise_distances(feature_emb)
            total_distance += sum_distances
            total_pairs += n_pairs
        
        # 避免除零
        if total_pairs > 0:
            feature_alignment_loss = total_distance / total_pairs
        else:
            feature_alignment_loss = torch.tensor(0.0, device=feature_emb.device)
            
        return feature_alignment_loss
    
    def compute_field_uniformity_loss(self, feature_embeddings):
        """
        计算字段均匀性损失 (修复版本)
        
        通过最小化不同特征间的余弦相似度来促进特征多样性
        
        Args:
            feature_embeddings: {feature_name: embedding_tensor}
            
        Returns:
            torch.Tensor: 字段均匀性损失
        """
        if not feature_embeddings or len(feature_embeddings) < 2:
            return torch.tensor(0.0, dtype=torch.float32)
        
        # 标准化特征向量
        normalized_features = {}
        for feature_name, feature_emb in feature_embeddings.items():
            # feature_emb shape: [batch_size, embedding_dim]
            if feature_emb.dim() > 2:
                feature_emb = feature_emb.view(feature_emb.size(0), -1)
            normalized_features[feature_name] = F.normalize(feature_emb, p=2, dim=-1)
        
        # 计算两两之间的余弦相似度
        feature_cos_sim_list = []
        feature_names = list(normalized_features.keys())
        
        for i, feature_i in enumerate(feature_names):
            for j, feature_j in enumerate(feature_names):
                if i < j:  # 只计算上三角，避免重复
                    # 🔧 修复：按样本计算余弦相似度，然后取batch平均
                    cos_sim_per_sample = torch.sum(
                        normalized_features[feature_i] * normalized_features[feature_j], 
                        dim=-1  # 沿特征维度求和，保持batch维度
                    )
                    # 取batch平均的绝对值（我们希望相似度尽可能小）
                    avg_cos_sim = torch.mean(torch.abs(cos_sim_per_sample))
                    feature_cos_sim_list.append(avg_cos_sim)
        
        if feature_cos_sim_list:
            # 字段均匀性损失：特征间相似度的平均值
            field_uniformity_loss = torch.mean(torch.stack(feature_cos_sim_list))
        else:
            field_uniformity_loss = torch.tensor(0.0, dtype=torch.float32)
            
        return field_uniformity_loss

    def compute_distance_loss(self, h1_logits, h2_logits, labels):
        """
        计算距离损失 (修复版本)
        
        使用对比学习的思想：相同标签的h1和h2应该相近，不同标签的应该远离
        
        Args:
            h1_logits: 第一个视图的logits
            h2_logits: 第二个视图的logits  
            labels: 真实标签
            
        Returns:
            torch.Tensor: 距离损失
        """
        if h1_logits is None or h2_logits is None:
            return torch.tensor(0.0, dtype=torch.float32)
        
        # 🔧 修复：使用简单的MSE损失鼓励两个视图的一致性
        # 对比学习的核心思想：相同样本的不同视图应该产生相似的预测
        distance_loss = F.mse_loss(h1_logits, h2_logits, reduction='mean')
        
        return distance_loss
    
    def compute_knowledge_distillation_loss(self, h1_logits, h2_logits, labels):
        """
        计算知识蒸馏损失 (核心改进)
        
        让非个性化视图(h2)从个性化视图(h1)中学习软标签知识
        这是专门为提升非个性化用户性能设计的损失
        
        Args:
            h1_logits: 个性化视图的logits (教师)
            h2_logits: 非个性化视图的logits (学生)  
            labels: 真实标签
            
        Returns:
            torch.Tensor: 知识蒸馏损失
        """
        if h1_logits is None or h2_logits is None:
            return torch.tensor(0.0, dtype=torch.float32)
        
        # 🔧 修复：处理二分类情况，将logits转换为概率
        if h1_logits.shape[-1] == 1:
            # 二分类：使用sigmoid转换为概率
            eps = 1e-7  # 增大epsilon值
            teacher_probs = torch.clamp(torch.sigmoid(h1_logits.squeeze() / self.temperature), eps, 1-eps)
            student_probs = torch.clamp(torch.sigmoid(h2_logits.squeeze() / self.temperature), eps, 1-eps)
            
            # 构造完整的概率分布 [p_negative, p_positive]
            teacher_probs_full = torch.stack([1 - teacher_probs, teacher_probs], dim=-1)
            student_log_probs_full = torch.stack([
                torch.log(1 - student_probs + 1e-8), 
                torch.log(student_probs + 1e-8)
            ], dim=-1)
            
            # KL散度损失
            kd_loss = F.kl_div(student_log_probs_full, teacher_probs_full, reduction='batchmean')
        else:
            # 多分类：使用原有逻辑
            teacher_probs = F.softmax(h1_logits / self.temperature, dim=-1)
            student_log_probs = F.log_softmax(h2_logits / self.temperature, dim=-1)
            kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        # 温度平方缩放（标准KD做法）
        kd_loss = kd_loss * (self.temperature ** 2)
        
        return kd_loss
    
    def compute_group_aware_loss(self, h1_logits, h2_logits, labels, group_ids=None):
        """
        计算组感知损失 (针对非个性化用户的专门优化)
        
        专门优化非个性化用户(group_2.0)的预测性能
        
        Args:
            h1_logits: 个性化视图的logits
            h2_logits: 非个性化视图的logits
            labels: 真实标签
            group_ids: 组标识 (1.0=个性化用户, 2.0=非个性化用户)
            
        Returns:
            torch.Tensor: 组感知损失
        """
        if h1_logits is None or h2_logits is None:
            return torch.tensor(0.0, dtype=torch.float32)

        # 如果没有组信息，假设所有样本都需要优化非个性化性能
        if group_ids is None:
            # 🔧 修复维度不匹配：确保logits和labels维度一致
            # 对所有样本使用非个性化视图的BCE损失
            if h2_logits.dim() > 1 and h2_logits.shape[-1] == 1:
                h2_logits_flat = h2_logits.squeeze(-1)  # [batch_size, 1] -> [batch_size]
            else:
                h2_logits_flat = h2_logits
                
            if labels.dim() > 1 and labels.shape[-1] == 1:
                labels_flat = labels.squeeze(-1)  # [batch_size, 1] -> [batch_size]
            else:
                labels_flat = labels
                
            group_loss = F.binary_cross_entropy_with_logits(h2_logits_flat, labels_flat.float(), reduction='mean')
        else:
            # 只对非个性化用户(group_2.0)优化
            non_personalized_mask = (group_ids == 2.0)
            num_non_personalized = non_personalized_mask.sum().item()

            if num_non_personalized > 0:
                non_pers_h2_logits = h2_logits[non_personalized_mask]
                non_pers_labels = labels[non_personalized_mask]
                
                # 🔧 修复维度不匹配：确保logits和labels维度一致
                if non_pers_h2_logits.dim() > 1 and non_pers_h2_logits.shape[-1] == 1:
                    non_pers_h2_logits_flat = non_pers_h2_logits.squeeze(-1)
                else:
                    non_pers_h2_logits_flat = non_pers_h2_logits
                    
                if non_pers_labels.dim() > 1 and non_pers_labels.shape[-1] == 1:
                    non_pers_labels_flat = non_pers_labels.squeeze(-1)
                else:
                    non_pers_labels_flat = non_pers_labels
                
                group_loss = F.binary_cross_entropy_with_logits(
                    non_pers_h2_logits_flat, 
                    non_pers_labels_flat.float(), 
                    reduction='mean'
                )
            else:
                group_loss = torch.tensor(0.0, dtype=torch.float32)

        return group_loss
    
    def compute_cl_loss(self, base_loss, feature_embeddings=None, h1_logits=None, h2_logits=None, labels=None, group_ids=None):
        """
        计算完整的对比学习损失 (改进版本)
        
        新增针对非个性化用户的专门优化策略
        
        Args:
            base_loss: 基础损失
            feature_embeddings: 特征嵌入字典 
            h1_logits: 第一个视图logits (个性化视图 - 教师)
            h2_logits: 第二个视图logits (非个性化视图 - 学生)
            labels: 真实标签
            group_ids: 组标识 (1.0=个性化用户, 2.0=非个性化用户)
            
        Returns:
            torch.Tensor: 总损失
        """
        total_loss = self.base_loss_weight * base_loss

        # 🔧 添加数值稳定性检查
        if torch.isnan(base_loss) or torch.isinf(base_loss):
            logging.warning(f"基础损失异常: {base_loss}")
            base_loss = torch.tensor(0.0, dtype=base_loss.dtype, device=base_loss.device)
        
        # 🎯 新增：知识蒸馏损失 (核心改进)
        if self.knowledge_distillation_loss_weight > 0 and h1_logits is not None and h2_logits is not None:
            # 在知识蒸馏损失计算前检查logits是否包含NaN或Inf
            if torch.isnan(h1_logits).any() or torch.isinf(h1_logits).any():
                logging.warning(f"h1_logits包含NaN或Inf，跳过知识蒸馏损失计算。")
                self.knowledge_distillation_loss = torch.tensor(0.0, dtype=torch.float32, device=h1_logits.device)
            else:
                self.knowledge_distillation_loss = self.compute_knowledge_distillation_loss(h1_logits, h2_logits, labels)
                # 数值检查
                if torch.isnan(self.knowledge_distillation_loss) or torch.isinf(self.knowledge_distillation_loss):
                    logging.warning(f"知识蒸馏损失异常: {self.knowledge_distillation_loss}")
                    self.knowledge_distillation_loss = torch.tensor(0.0, dtype=self.knowledge_distillation_loss.dtype, device=self.knowledge_distillation_loss.device)
            weighted_kd_loss = self.knowledge_distillation_loss_weight * self.knowledge_distillation_loss
            total_loss += weighted_kd_loss

        # 🎯 新增：组感知损失 (专门优化非个性化用户)
        if self.group_aware_loss_weight > 0 and h1_logits is not None and h2_logits is not None:
            self.group_aware_loss = self.compute_group_aware_loss(h1_logits, h2_logits, labels, group_ids)
            # 数值检查
            if torch.isnan(self.group_aware_loss) or torch.isinf(self.group_aware_loss):
                logging.warning(f"组感知损失异常: {self.group_aware_loss}")
                self.group_aware_loss = torch.tensor(0.0, dtype=self.group_aware_loss.dtype, device=self.group_aware_loss.device)
            weighted_group_loss = self.group_aware_loss_weight * self.group_aware_loss
            total_loss += weighted_group_loss

        # 特征对齐损失
        if self.feature_alignment_loss_weight > 0 and feature_embeddings is not None:
            self.feature_alignment_loss = self.compute_feature_alignment_loss(feature_embeddings)
            # 数值检查
            if torch.isnan(self.feature_alignment_loss) or torch.isinf(self.feature_alignment_loss):
                logging.warning(f"特征对齐损失异常: {self.feature_alignment_loss}")
                self.feature_alignment_loss = torch.tensor(0.0, dtype=self.feature_alignment_loss.dtype, device=self.feature_alignment_loss.device)
            weighted_fa_loss = self.feature_alignment_loss_weight * self.feature_alignment_loss
            total_loss += weighted_fa_loss

        # 字段均匀性损失  
        if self.field_uniformity_loss_weight > 0 and feature_embeddings is not None:
            self.field_uniformity_loss = self.compute_field_uniformity_loss(feature_embeddings)
            # 数值检查
            if torch.isnan(self.field_uniformity_loss) or torch.isinf(self.field_uniformity_loss):
                logging.warning(f"字段均匀性损失异常: {self.field_uniformity_loss}")
                self.field_uniformity_loss = torch.tensor(0.0, dtype=self.field_uniformity_loss.dtype, device=self.field_uniformity_loss.device)
            weighted_fu_loss = self.field_uniformity_loss_weight * self.field_uniformity_loss
            total_loss += weighted_fu_loss

        # 距离损失 (原有的，现在作为备选)
        if self.distance_loss_weight > 0 and h1_logits is not None and h2_logits is not None:
            self.distance_loss = self.compute_distance_loss(h1_logits, h2_logits, labels)
            # 数值检查
            if torch.isnan(self.distance_loss) or torch.isinf(self.distance_loss):
                logging.warning(f"距离损失异常: {self.distance_loss}")
                self.distance_loss = torch.tensor(0.0, dtype=self.distance_loss.dtype, device=self.distance_loss.device)
            weighted_dist_loss = self.distance_loss_weight * self.distance_loss
            total_loss += weighted_dist_loss

        # 🔧 最终数值检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logging.error(f"总损失异常: {total_loss}, 基础损失: {base_loss}")
            total_loss = base_loss  # 回退到基础损失
        
        return total_loss
    
    def get_group_ids(self, inputs):
        """
        从inputs中提取组标识信息
        
        根据is_personalization特征区分个性化/非个性化用户：
        - is_personalization=1: 个性化用户 (返回1.0)
        - is_personalization=0或2: 非个性化用户 (返回2.0)
        
        Args:
            inputs: 模型输入字典，包含所有特征
            
        Returns:
            torch.Tensor or None: 组标识张量，1.0表示个性化用户，2.0表示非个性化用户
        """
        try:
            if 'is_personalization' in inputs:
                personalization_flag = inputs['is_personalization']
                
                # 确保是张量格式
                if not isinstance(personalization_flag, torch.Tensor):
                    personalization_flag = torch.tensor(personalization_flag)
                
                # 转换为组标识：1->1.0 (个性化), 0或2->2.0 (非个性化)
                group_ids = torch.where(
                    personalization_flag == 1.0,
                    torch.tensor(1.0, dtype=torch.float32, device=personalization_flag.device),  # 个性化用户
                    torch.tensor(2.0, dtype=torch.float32, device=personalization_flag.device)   # 非个性化用户
                )
                
                return group_ids
            else:
                return None
                
        except Exception as e:
            logging.warning(f"提取组标识时出错: {e}")
            return None
