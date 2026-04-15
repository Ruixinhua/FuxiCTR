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

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from fuxictr.pytorch.layers import FeatureEmbedding, InnerProductInteraction, MLP_Block


class BaseModelAdapter(nn.Module, ABC):
    """
    基础模型适配器抽象类，用于从不同的基础模型中提取hidden states
    现在集成对比学习功能
    """

    def __init__(self,
                 feature_map,
                 use_personalisation=False,
                 personalization_feature_list=None,
                 mask_type='Personalisation',
                 use_cl_mask=False,
                 embedding_dim=10,
                 **kwargs):
        super(BaseModelAdapter, self).__init__()

        self.feature_map = feature_map
        self.embedding_dim = embedding_dim

        # 个性化参数
        self.use_personalisation = use_personalisation
        self.personalization_feature_list = personalization_feature_list or []
        self.mask_type = mask_type
        self.use_cl_mask = use_cl_mask

        if self.use_personalisation:
            # 构建非个性化特征列表
            self.non_personalization_feature_list = []
            for field in feature_map.features:
                if field not in self.personalization_feature_list:
                    self.non_personalization_feature_list.append(field)
            logging.info(f"Personalization features: {self.personalization_feature_list}")
            logging.info(f"Non-personalization features: {self.non_personalization_feature_list}")

    @abstractmethod
    def get_hidden_states(self, inputs):
        """
        从基础模型中提取hidden states

        Args:
            inputs: 模型输入

        Returns:
            hidden_states: 提取的hidden states (batch_size, hidden_dim)
        """
        pass

    @abstractmethod
    def get_hidden_dim(self):
        """
        获取hidden states的维度

        Returns:
            hidden_dim: hidden states的维度
        """
        pass

    def get_model_output(self, inputs):
        """
        获取基础模型的完整输出（用于SingleTower模式）
        默认实现：从get_model_return_dict中提取y_pred

        Args:
            inputs: 模型输入

        Returns:
            model_output: 基础模型的原始输出 (batch_size, 1)
        """
        return_dict = self.get_model_return_dict(inputs)
        return return_dict["y_pred"]

    @abstractmethod
    def get_model_return_dict(self, inputs):
        """
        获取基础模型的完整返回字典（用于SingleTower模式的完整输出）

        Args:
            inputs: 模型输入

        Returns:
            return_dict: 包含基础模型所有输出的字典，至少包含 'y_pred'
        """
        pass

    def has_custom_loss(self):
        """
        指示基础模型是否有自定义的损失计算逻辑

        Returns:
            bool: 如果基础模型有自定义损失计算则返回True，否则返回False
        """
        return False

    def compute_custom_loss(self, return_dict, y_true, loss_fn):
        """
        计算基础模型的自定义损失（如果有的话）

        Args:
            return_dict: 基础模型的输出字典
            y_true: 真实标签
            loss_fn: 损失函数

        Returns:
            custom_loss: 自定义损失值
        """
        # 默认实现：使用标准的预测损失
        y_pred = return_dict["y_pred"]
        return loss_fn(y_pred, y_true, reduction='mean')

    # 对比学习相关方法（通用实现）
    def get_feature_embeddings(self, inputs):
        """获取每个特征的单独嵌入（通用实现）"""
        feature_embeddings = {}
        X = inputs

        for field_name in X:
            # 为每个特征单独计算嵌入
            single_field_dict = {field_name: X[field_name]}
            try:
                # 调用子类实现的特定方法
                single_field_emb = self._extract_single_field_embedding(field_name, single_field_dict)
                feature_embeddings[field_name] = single_field_emb
            except Exception as e:
                logging.warning(f"Failed to extract embedding for field {field_name}: {e}")
                continue

        return feature_embeddings

    def sum_unique_pairwise_distances(self, tensor):
        """计算同一特征下所有样本嵌入之间的成对L2距离之和"""
        batch_size = tensor.size(0)

        # 成对数量 = m*(m-1)/2
        n_pairs = torch.tensor(batch_size * (batch_size - 1) / 2,
                               dtype=tensor.dtype, device=tensor.device)

        # 只有一个样本时返回0
        if batch_size == 1:
            return torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device), \
                torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device)

        # 创建上三角掩码
        indices = torch.triu_indices(batch_size, batch_size, offset=1, device=tensor.device)

        # 获取成对元素
        elements_i = tensor[indices[0]]  # [n_pairs, embedding_dim]
        elements_j = tensor[indices[1]]  # [n_pairs, embedding_dim]

        # 计算L2距离
        distances = torch.norm(elements_i - elements_j, p=2, dim=-1)

        # 求和
        sum_distances = torch.sum(distances)

        return sum_distances, n_pairs

    def get_feature_alignment_loss(self, feature_embeddings):
        """计算特征对齐损失 - 使同一特征的不同样本嵌入更相似（通用实现）"""
        total_distance = 0.0
        total_pairs = 0.0

        for field_name, field_emb in feature_embeddings.items():
            # 处理不同的嵌入形状（通用处理）
            if field_emb.dim() == 3:
                field_emb_flat = field_emb.view(field_emb.size(0), -1)  # [batch_size, flattened_dim]
            elif field_emb.dim() == 2:
                field_emb_flat = field_emb  # [batch_size, embedding_dim]
            else:
                logging.warning(f"Unexpected embedding shape for field {field_name}: {field_emb.shape}")
                continue

            sum_distances, n_pairs = self.sum_unique_pairwise_distances(field_emb_flat)
            total_distance += sum_distances
            total_pairs += n_pairs

        # 避免除零
        if total_pairs > 0:
            feature_alignment_loss = total_distance / total_pairs
        else:
            feature_alignment_loss = torch.tensor(0.0, device=total_distance.device if hasattr(total_distance,
                                                                                               'device') else 'cpu')

        return feature_alignment_loss

    def get_field_uniformity_loss(self, feature_embeddings):
        """计算字段一致性损失 - 控制不同特征间的余弦相似度分布（通用实现）"""
        # 归一化特征嵌入
        normalized_embeddings = {}
        for field_name, field_emb in feature_embeddings.items():
            # 处理不同的嵌入形状（通用处理）
            if field_emb.dim() == 3:
                field_emb_flat = field_emb.view(field_emb.size(0), -1)
            elif field_emb.dim() == 2:
                field_emb_flat = field_emb
            else:
                logging.warning(f"Unexpected embedding shape for field {field_name}: {field_emb.shape}")
                continue

            # 对每个样本的嵌入进行L2归一化
            normalized_emb = F.normalize(field_emb_flat, p=2, dim=-1)
            # 对批次维度求平均得到特征级别的表示
            normalized_embeddings[field_name] = torch.mean(normalized_emb, dim=0, keepdim=True)

        # 计算特征间余弦相似度
        field_names = list(normalized_embeddings.keys())
        total_cos_sim = 0.0
        pair_count = 0

        for i, field_i in enumerate(field_names):
            for j, field_j in enumerate(field_names):
                if i != j:
                    cos_sim = torch.sum(normalized_embeddings[field_i] * normalized_embeddings[field_j])
                    total_cos_sim += cos_sim
                    pair_count += 1

        if pair_count > 0:
            field_uniformity_loss = total_cos_sim / pair_count
        else:
            field_uniformity_loss = torch.tensor(0.0, device=list(normalized_embeddings.values())[0].device)

        return field_uniformity_loss

    def get_contrastive_outputs(self, inputs):
        """获取对比学习的输出（通用实现）"""
        cl_outputs = {}

        # h1: 完整特征的输出
        h1_hidden_states = self.get_hidden_states(inputs)
        cl_outputs['h1_hidden_states'] = h1_hidden_states

        # h2: 非个性化特征的输出（如果启用个性化掩码）
        if (self.use_cl_mask and self.mask_type == 'Personalisation' and
                self.use_personalisation and len(self.non_personalization_feature_list) > 0):

            # 获取完整特征的嵌入并创建掩码
            if hasattr(self, 'embedding_layer'):
                try:
                    full_feature_emb = self.embedding_layer(inputs)
                    masked_feature_emb = self._create_embedding_mask(full_feature_emb, inputs)

                    # 使用子类实现的方法重新计算hidden states
                    h2_hidden_states = self._recompute_masked_hidden_states(masked_feature_emb, inputs)

                    cl_outputs['h2_hidden_states'] = h2_hidden_states
                    cl_outputs['has_contrastive'] = True

                except Exception as e:
                    logging.warning(f"Failed to create contrastive outputs: {e}")
                    cl_outputs['has_contrastive'] = False
            else:
                cl_outputs['has_contrastive'] = False
        else:
            cl_outputs['has_contrastive'] = False

        return cl_outputs

    # 抽象方法供子类实现特定的对比学习逻辑
    @abstractmethod
    def _extract_single_field_embedding(self, field_name, single_field_dict):
        """提取单个特征的嵌入（子类实现）"""
        pass

    @abstractmethod
    def _create_embedding_mask(self, full_feature_emb, X):
        """创建个性化特征掩码（子类实现）"""
        pass

    @abstractmethod
    def _recompute_masked_hidden_states(self, masked_feature_emb, inputs):
        """使用掩码后的嵌入重新计算hidden states（子类实现）"""
        pass


class DCNv3Adapter(BaseModelAdapter):
    """
    DCNv3模型适配器，从DCNv3中提取hidden states，并集成对比学习功能。
    内部使用统一的 DCNv3Backbone 实现核心计算。
    """

    def __init__(self,
                 feature_map,
                 embedding_dim=10,
                 num_deep_cross_layers=4,
                 num_shallow_cross_layers=4,
                 deep_net_dropout=0.1,
                 shallow_net_dropout=0.1,
                 layer_norm=True,
                 batch_norm=True,
                 num_heads=8,
                 output_activation=None,
                 **kwargs):
        super(DCNv3Adapter, self).__init__(feature_map, embedding_dim=embedding_dim, **kwargs)

        from fuxictr.pytorch.backbone import DCNv3Backbone

        self._backbone = DCNv3Backbone(
            feature_map,
            embedding_dim=embedding_dim,
            num_deep_cross_layers=num_deep_cross_layers,
            num_shallow_cross_layers=num_shallow_cross_layers,
            deep_net_dropout=deep_net_dropout,
            shallow_net_dropout=shallow_net_dropout,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            num_heads=num_heads,
            output_activation=output_activation,
        )

        # Alias for CL methods that access embedding_layer directly
        self.embedding_layer = self._backbone.embedding_layer
        cross_input_dim = feature_map.num_fields * embedding_dim
        self.hidden_dim = num_heads * cross_input_dim

    def get_hidden_states(self, inputs):
        """
        获取ECN和LCN的分离intermediate输出，用于多塔处理

        Returns:
            tuple: (xld_flat, xls_flat) - ECN和LCN的扁平化输出
        """
        return self._backbone.forward_features(inputs)

    def get_hidden_dim(self):
        return self.hidden_dim

    def get_model_return_dict(self, inputs):
        return self._backbone.get_model_return_dict(inputs)

    def has_custom_loss(self):
        return self._backbone.has_custom_loss()

    def compute_custom_loss(self, return_dict, y_true, loss_fn):
        return self._backbone.compute_custom_loss(return_dict, y_true, loss_fn)

    # 对比学习相关方法实现
    def _extract_single_field_embedding(self, field_name, single_field_dict):
        """DCNv3特定的特征嵌入提取"""
        # 使用底层的FeatureEmbedding
        if hasattr(self.embedding_layer, 'embedding_layer'):
            single_field_emb = self.embedding_layer.embedding_layer(single_field_dict)
        else:
            single_field_emb = self.embedding_layer(single_field_dict)

        return single_field_emb

    def _create_embedding_mask(self, full_feature_emb, X):
        """DCNv3特定的嵌入掩码创建"""
        if full_feature_emb.dim() == 3:
            embedding_dim_per_head = self.embedding_dim
        else:
            embedding_dim_per_head = self.embedding_dim

        # 获取特征顺序
        feature_names = list(X.keys())

        # 创建掩码
        mask = torch.ones_like(full_feature_emb)

        # 将个性化特征对应的位置置零
        for idx, field_name in enumerate(feature_names):
            if field_name in self.personalization_feature_list:
                # 计算这个特征在嵌入中的位置范围
                start_idx = idx * embedding_dim_per_head
                end_idx = (idx + 1) * embedding_dim_per_head

                if full_feature_emb.dim() == 3:
                    mask[:, :, start_idx:end_idx] = 0.0
                else:
                    mask[:, start_idx:end_idx] = 0.0

        # 应用掩码
        masked_feature_emb = full_feature_emb * mask
        return masked_feature_emb

    def _recompute_masked_hidden_states(self, masked_feature_emb, inputs):
        """DCNv3特定的hidden states重计算"""
        # 对于DCNv3，直接使用masked embedding作为h2_hidden_states（简化处理）
        if masked_feature_emb.dim() == 3:
            h2_hidden_states = masked_feature_emb.view(masked_feature_emb.size(0), -1)
        else:
            h2_hidden_states = masked_feature_emb

        return h2_hidden_states


class PNNAdapter(BaseModelAdapter):
    """
    PNN模型适配器，从PNN中提取hidden states，并集成对比学习功能。
    内部使用统一的 PNNBackbone 实现核心计算。
    """

    def __init__(self, feature_map, embedding_dim=10, product_type="inner", **kwargs):
        super(PNNAdapter, self).__init__(feature_map, embedding_dim=embedding_dim, **kwargs)

        from fuxictr.pytorch.backbone import PNNBackbone

        self._backbone = PNNBackbone(
            feature_map,
            embedding_dim=embedding_dim,
            product_type=product_type,
            hidden_units=kwargs.get("hidden_units", []),
            hidden_activations=kwargs.get("hidden_activations", "ReLU"),
            net_dropout=kwargs.get("dropout_rates", 0.0),
            batch_norm=kwargs.get("batch_norm", False),
            output_activation=kwargs.get("output_activation"),
        )

        # Alias for CL methods
        self.embedding_layer = self._backbone.embedding_layer
        self.inner_product_layer = self._backbone.inner_product_layer
        self.hidden_dim = self._backbone._feature_input_dim

    def get_hidden_states(self, inputs):
        """提取hidden states（返回拼接后的hidden_states，不是tuple）"""
        return self._backbone.forward_features(inputs)

    def get_hidden_dim(self):
        return self.hidden_dim

    def get_model_return_dict(self, inputs):
        return self._backbone.get_model_return_dict(inputs)

    def has_custom_loss(self):
        return self._backbone.has_custom_loss()

    # 对比学习相关方法实现
    def _extract_single_field_embedding(self, field_name, single_field_dict):
        """PNN特定的特征嵌入提取"""
        # PNN使用FeatureEmbedding，结构相对简单
        # 直接调用embedding_layer
        single_field_emb = self.embedding_layer(single_field_dict)
        # single_field_emb shape: [batch_size, num_fields, embedding_dim]
        # 对于单个字段，num_fields = 1
        if single_field_emb.dim() == 3 and single_field_emb.size(1) == 1:
            single_field_emb = single_field_emb.squeeze(1)  # [batch_size, embedding_dim]

        return single_field_emb

    def _create_embedding_mask(self, full_feature_emb, X):
        """PNN特定的嵌入掩码创建"""
        # PNN的embedding通常是[batch_size, num_fields, embedding_dim]
        if full_feature_emb.dim() == 3:
            batch_size, num_fields, embedding_dim = full_feature_emb.shape
        elif full_feature_emb.dim() == 2:
            # 如果是2D，说明已经被flatten了
            batch_size, total_dim = full_feature_emb.shape
            num_fields = len(X)
            embedding_dim = total_dim // num_fields
            # 重新reshape为3D方便处理
            full_feature_emb = full_feature_emb.view(batch_size, num_fields, embedding_dim)
        else:
            raise ValueError(f"Unexpected embedding dimension: {full_feature_emb.dim()}")

        # 获取特征顺序
        feature_names = list(X.keys())

        # 创建掩码
        mask = torch.ones_like(full_feature_emb)

        # 将个性化特征对应的位置置零
        for idx, field_name in enumerate(feature_names):
            if field_name in self.personalization_feature_list and idx < num_fields:
                # 将这个特征的嵌入位置置零
                mask[:, idx, :] = 0.0

        # 应用掩码
        masked_feature_emb = full_feature_emb * mask

        # 如果原来是2D，需要flatten回去
        if len(full_feature_emb.shape) != len(mask.shape):
            masked_feature_emb = masked_feature_emb.view(batch_size, -1)

        return masked_feature_emb

    def _recompute_masked_hidden_states(self, masked_feature_emb, inputs):
        """PNN特定的hidden states重计算"""
        # 使用掩码后的嵌入重新计算hidden states
        # 对于PNN，需要重新计算inner product interaction
        if hasattr(self, 'inner_product_layer'):
            # 确保masked_feature_emb的形状正确
            if masked_feature_emb.dim() == 2:
                # 重新reshape为3D: [batch_size, num_fields, embedding_dim]
                num_fields = len(inputs)
                embedding_dim = masked_feature_emb.size(-1) // num_fields
                masked_feature_emb_3d = masked_feature_emb.view(
                    masked_feature_emb.size(0), num_fields, embedding_dim)
            else:
                masked_feature_emb_3d = masked_feature_emb

            # 重新计算inner product
            masked_inner_products = self.inner_product_layer(masked_feature_emb_3d)

            # 拼接masked embedding和inner products作为h2_hidden_states
            masked_feature_emb_flat = masked_feature_emb_3d.flatten(start_dim=1)
            h2_hidden_states = torch.cat([masked_feature_emb_flat, masked_inner_products], dim=1)

        else:
            # 如果没有inner_product_layer，直接使用masked embedding
            h2_hidden_states = masked_feature_emb.view(masked_feature_emb.size(0), -1)

        return h2_hidden_states


class FinalNetAdapter(BaseModelAdapter):
    """
    FinalNet模型适配器，从FinalNet中提取hidden states，并集成对比学习功能。
    内部使用统一的 FinalNetBackbone 实现核心计算。
    """

    def __init__(self, 
                 feature_map,
                 embedding_dim=10,
                 block_type="2B",
                 batch_norm=True,
                 use_feature_gating=False,
                 block1_hidden_units=[800],
                 block1_hidden_activations="ReLU",
                 block1_dropout=0.2,
                 block2_hidden_units=[800, 800],
                 block2_hidden_activations=None,
                 block2_dropout=0.3,
                 residual_type="concat",
                 output_activation=None,
                 **kwargs):
        super(FinalNetAdapter, self).__init__(feature_map, embedding_dim=embedding_dim, **kwargs)

        from fuxictr.pytorch.backbone import FinalNetBackbone

        self._backbone = FinalNetBackbone(
            feature_map,
            embedding_dim=embedding_dim,
            block_type=block_type,
            batch_norm=batch_norm,
            use_feature_gating=use_feature_gating,
            block1_hidden_units=block1_hidden_units,
            block1_hidden_activations=block1_hidden_activations,
            block1_dropout=block1_dropout,
            block2_hidden_units=block2_hidden_units,
            block2_hidden_activations=block2_hidden_activations,
            block2_dropout=block2_dropout,
            residual_type=residual_type,
            output_activation=output_activation,
        )

        # Alias for CL methods
        self.embedding_layer = self._backbone.embedding_layer
        self.block_type = block_type
        self.use_feature_gating = use_feature_gating
        self.hidden_dim = self._backbone.latent_dim
        if use_feature_gating:
            self.feature_gating = self._backbone.feature_gating
        self.block1 = self._backbone.block1

    def get_hidden_states(self, inputs):
        """提取hidden states（返回block1的输出）"""
        return self._backbone.forward_features(inputs)

    def get_hidden_dim(self):
        return self.hidden_dim

    def get_model_return_dict(self, inputs):
        return self._backbone.get_model_return_dict(inputs)

    def has_custom_loss(self):
        return self._backbone.has_custom_loss()

    def compute_custom_loss(self, return_dict, y_true, loss_fn):
        return self._backbone.compute_custom_loss(return_dict, y_true, loss_fn)

    # 对比学习相关方法实现
    def _extract_single_field_embedding(self, field_name, single_field_dict):
        """FinalNet特定的特征嵌入提取"""
        # FinalNet使用FeatureEmbedding，结构相对简单
        single_field_emb = self.embedding_layer(single_field_dict)
        # single_field_emb shape: [batch_size, num_fields, embedding_dim]
        # 对于单个字段，num_fields = 1
        if single_field_emb.dim() == 3 and single_field_emb.size(1) == 1:
            single_field_emb = single_field_emb.squeeze(1)  # [batch_size, embedding_dim]
        
        return single_field_emb

    def _create_embedding_mask(self, full_feature_emb, X):
        """FinalNet特定的嵌入掩码创建"""
        # FinalNet的embedding通常是[batch_size, num_fields, embedding_dim]
        if full_feature_emb.dim() == 3:
            batch_size, num_fields, embedding_dim = full_feature_emb.shape
        elif full_feature_emb.dim() == 2:
            # 如果是2D，说明已经被flatten了
            batch_size, total_dim = full_feature_emb.shape
            num_fields = len(X)
            embedding_dim = total_dim // num_fields
            # 重新reshape为3D方便处理
            full_feature_emb = full_feature_emb.view(batch_size, num_fields, embedding_dim)
        else:
            raise ValueError(f"Unexpected embedding dimension: {full_feature_emb.dim()}")

        # 获取特征顺序
        feature_names = list(X.keys())

        # 创建掩码
        mask = torch.ones_like(full_feature_emb)

        # 将个性化特征对应的位置置零
        for idx, field_name in enumerate(feature_names):
            if field_name in self.personalization_feature_list and idx < num_fields:
                # 将这个特征的嵌入位置置零
                mask[:, idx, :] = 0.0

        # 应用掩码
        masked_feature_emb = full_feature_emb * mask

        # 如果原来是2D，需要flatten回去
        if len(full_feature_emb.shape) != len(mask.shape):
            masked_feature_emb = masked_feature_emb.view(batch_size, -1)

        return masked_feature_emb

    def _recompute_masked_hidden_states(self, masked_feature_emb, inputs):
        """FinalNet特定的hidden states重计算"""
        # 使用掩码后的嵌入重新计算hidden states
        # 对于FinalNet，需要重新通过block1
        if masked_feature_emb.dim() == 2:
            # 重新reshape为3D: [batch_size, num_fields, embedding_dim]
            num_fields = len(inputs)
            embedding_dim = masked_feature_emb.size(-1) // num_fields
            masked_feature_emb_3d = masked_feature_emb.view(
                masked_feature_emb.size(0), num_fields, embedding_dim)
        else:
            masked_feature_emb_3d = masked_feature_emb

        # 如果使用feature_gating，需要重新应用
        if self.use_feature_gating:
            masked_feature_emb_3d = self.feature_gating(masked_feature_emb_3d)

        # 重新通过block1计算hidden states
        h2_hidden_states = self.block1(masked_feature_emb_3d.flatten(start_dim=1))

        return h2_hidden_states

