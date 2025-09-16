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
import os
from model_zoo.MTCL.src.DualTowerModel import DualTowerModel
from model_zoo.CL.src.base import ContrastiveLearningBase


class DualTowerCL(DualTowerModel, ContrastiveLearningBase):
    """
    双塔对比学习模型：集成对比学习策略的双塔架构
    
    核心思想：
    1. 通过对比学习让非个性化模型学习个性化模型的预测结果
    2. 保持个性化模型的推荐性能不受影响
    3. 支持并行训练和串行训练两种模式
    
    训练模式：
    - 并行模式 (parallel): 同时训练两个塔，添加CL损失进行知识传递
    - 串行模式 (sequential): 先训练个性化塔，再固定个性化塔训练非个性化塔
    """
    
    def __init__(self,
                 feature_map,
                 model_id="DualTowerCL",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 # 双塔基础配置（继承自DualTowerModel）
                 personalized_model_type="PNN",
                 personalized_model_params=None,
                 non_personalized_model_type="DCNv3",
                 non_personalized_model_params=None,
                 personalization_feature_list=None,
                 personalization_field="is_personalization",
                 personalized_loss_weight=1.0,
                 non_personalized_loss_weight=1.0,
                 personalized_model_use_all_data=False,
                 non_personalized_model_use_all_data=True,
                 # 对比学习核心配置
                 training_mode="parallel",  # "parallel" 或 "sequential"
                 cl_loss_weight=0.1,  # CL总损失权重
                 knowledge_distillation_loss_weight=1.0,  # 知识蒸馏损失权重
                 group_aware_loss_weight=0.5,  # 组感知损失权重
                 feature_alignment_loss_weight=0.0,  # 特征对齐损失权重
                 field_uniformity_loss_weight=0.0,  # 字段均匀性损失权重
                 distance_loss_weight=0.0,  # 距离损失权重（备用）
                 temperature=4.0,  # 知识蒸馏温度参数
                 # 串行训练配置
                 personalized_model_path=None,  # 预训练个性化模型路径
                 freeze_personalized_in_sequential=True,  # 串行模式下是否冻结个性化模型
                 sequential_warmup_epochs=2,  # 串行模式预热轮数
                 # 特征级CL配置
                 use_feature_level_cl=False,  # 是否启用特征级对比学习
                 personalization_feature_list_for_cl=None,  # 用于CL的个性化特征列表
                 # 其他配置
                 use_tower_specific_monitoring=True,
                 personalized_monitor_metric="AUC_group_1.0",
                 non_personalized_monitor_metric="AUC_group_2.0",
                 tower_patience=3,
                 save_tower_models=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        
        # 初始化对比学习基础类
        ContrastiveLearningBase.__init__(
            self,
            personalization_feature_list=personalization_feature_list,
            knowledge_distillation_loss_weight=knowledge_distillation_loss_weight,
            group_aware_loss_weight=group_aware_loss_weight,
            feature_alignment_loss_weight=feature_alignment_loss_weight,
            field_uniformity_loss_weight=field_uniformity_loss_weight,
            distance_loss_weight=distance_loss_weight,
            temperature=temperature,
            **kwargs
        )
        
        # 初始化双塔模型
        DualTowerModel.__init__(
            self,
            feature_map=feature_map,
            model_id=model_id,
            gpu=gpu,
            learning_rate=learning_rate,
            embedding_dim=embedding_dim,
            personalized_model_type=personalized_model_type,
            personalized_model_params=personalized_model_params,
            non_personalized_model_type=non_personalized_model_type,
            non_personalized_model_params=non_personalized_model_params,
            personalization_feature_list=personalization_feature_list,
            personalization_field=personalization_field,
            personalized_loss_weight=personalized_loss_weight,
            non_personalized_loss_weight=non_personalized_loss_weight,
            personalized_model_use_all_data=personalized_model_use_all_data,
            non_personalized_model_use_all_data=non_personalized_model_use_all_data,
            use_contrastive_learning=False,  # 禁用原来的简单CL，使用新的CL策略
            use_tower_specific_monitoring=use_tower_specific_monitoring,
            personalized_monitor_metric=personalized_monitor_metric,
            non_personalized_monitor_metric=non_personalized_monitor_metric,
            tower_patience=tower_patience,
            save_tower_models=save_tower_models,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs
        )
        
        # CL特有配置
        self.training_mode = training_mode
        self.cl_loss_weight = cl_loss_weight
        self.personalized_model_path = personalized_model_path
        self.freeze_personalized_in_sequential = freeze_personalized_in_sequential
        self.sequential_warmup_epochs = sequential_warmup_epochs
        self.use_feature_level_cl = use_feature_level_cl
        self.personalization_feature_list_for_cl = personalization_feature_list_for_cl or personalization_feature_list
        
        # 训练状态管理
        self.is_sequential_phase_2 = False  # 是否处于串行训练第二阶段
        self.current_training_epoch = 0
        
        # 如果是串行模式且提供了预训练模型，加载它
        if self.training_mode == "sequential" and self.personalized_model_path:
            self._load_pretrained_personalized_model()
        
        logging.info(f"DualTowerCL initialized:")
        logging.info(f"  - Training mode: {training_mode}")
        logging.info(f"  - CL loss weight: {cl_loss_weight}")
        logging.info(f"  - Knowledge distillation weight: {knowledge_distillation_loss_weight}")
        logging.info(f"  - Group aware loss weight: {group_aware_loss_weight}")
        logging.info(f"  - Temperature: {temperature}")
        logging.info(f"  - Use feature level CL: {use_feature_level_cl}")
        if training_mode == "sequential":
            logging.info(f"  - Freeze personalized in sequential: {freeze_personalized_in_sequential}")
            logging.info(f"  - Sequential warmup epochs: {sequential_warmup_epochs}")
    
    def _load_pretrained_personalized_model(self):
        """
        加载预训练的个性化模型
        """
        if not self.personalized_model_path or not os.path.exists(self.personalized_model_path):
            logging.warning(f"Pretrained personalized model not found: {self.personalized_model_path}")
            return
        
        try:
            # 加载预训练模型
            checkpoint = torch.load(self.personalized_model_path, map_location=self.device, weights_only=False)
            
            if "model_state_dict" in checkpoint:
                # 如果是tower模型格式
                self.personalized_model.load_state_dict(checkpoint["model_state_dict"])
                logging.info(f"Loaded pretrained personalized tower model from: {self.personalized_model_path}")
                if "best_metric" in checkpoint:
                    logging.info(f"  - Pretrained model metric: {checkpoint['best_metric']:.6f}")
            else:
                # 如果是完整模型格式，尝试提取个性化部分
                self.personalized_model.load_state_dict(checkpoint, strict=False)
                logging.info(f"Loaded pretrained personalized model (partial) from: {self.personalized_model_path}")
            
            # 如果需要冻结个性化模型
            if self.freeze_personalized_in_sequential:
                self._freeze_personalized_model()
                
        except Exception as e:
            logging.error(f"Failed to load pretrained personalized model: {e}")
    
    def _freeze_personalized_model(self):
        """
        冻结个性化模型参数
        """
        for param in self.personalized_model.parameters():
            param.requires_grad = False
        logging.info("Personalized model parameters frozen")
    
    def _unfreeze_personalized_model(self):
        """
        解冻个性化模型参数
        """
        for param in self.personalized_model.parameters():
            param.requires_grad = True
        logging.info("Personalized model parameters unfrozen")
    
    def forward(self, inputs):
        """
        前向传播 - 扩展支持特征级对比学习
        
        Args:
            inputs: 输入特征字典
            
        Returns:
            return_dict: 包含预测结果和CL相关信息的字典
        """
        # 缓存原始输入用于CL损失计算
        self._current_inputs = inputs
        
        # 调用父类的前向传播
        return_dict = super().forward(inputs)
        
        # 如果启用特征级对比学习，获取特征嵌入
        if self.use_feature_level_cl and self.training:
            X = self.get_inputs(inputs)
            full_features, non_personalized_features = self.feature_separator.separate_features(X)
            
            # 获取个性化特征的嵌入（用于特征对齐和均匀性损失）
            if hasattr(self.personalized_model, 'embedding_layer'):
                personalized_feature_embeddings = self.get_feature_embeddings(
                    self.personalized_model.embedding_layer,
                    full_features,
                    # self.personalization_feature_list_for_cl
                )
                return_dict["personalized_feature_embeddings"] = personalized_feature_embeddings
            
            if hasattr(self.non_personalized_model, 'embedding_layer'):
                non_personalized_feature_embeddings = self.get_feature_embeddings(
                    self.non_personalized_model.embedding_layer,
                    non_personalized_features,
                )
                return_dict["non_personalized_feature_embeddings"] = non_personalized_feature_embeddings
        
        return return_dict
    
    def add_loss(self, return_dict, y_true):
        """
        计算包含对比学习的组合损失函数
        
        Args:
            return_dict: 模型输出字典
            y_true: 真实标签
            
        Returns:
            total_loss: 计算的总损失值
        """
        # 1. 获取基础双塔损失
        base_dual_tower_loss = super().add_loss(return_dict, y_true)
        
        # 2. 根据训练模式决定是否添加CL损失
        if not self._should_apply_cl_loss():
            return base_dual_tower_loss
        
        # 3. 准备CL损失计算所需的数据
        personalized_pred = return_dict["personalized_pred"]
        non_personalized_pred = return_dict["non_personalized_pred"]
        
        # 获取组标识 - 从缓存的输入中获取
        group_ids = None
        if hasattr(self, '_current_inputs') and self._current_inputs is not None:
            group_ids = self.get_group_ids(self._current_inputs)
        
        # 4. 计算对比学习损失
        cl_loss = self.compute_cl_loss(
            base_loss=torch.tensor(0.0, device=y_true.device),  # 基础损失已经在上面计算了
            feature_embeddings=return_dict.get("personalized_feature_embeddings"),
            h1_logits=personalized_pred,  # 个性化视图（教师）
            h2_logits=non_personalized_pred,  # 非个性化视图（学生）
            labels=y_true,
            group_ids=group_ids
        )
        cl_loss = self.compute_cl_loss(  # 仅计算非个性化模型对应的 filed uniformity 和 feature alignment loss
            base_loss=cl_loss,
            feature_embeddings=return_dict.get("non_personalized_feature_embeddings"),
            labels=y_true,
            group_ids=group_ids
        )
        
        # 5. 组合总损失
        total_loss = base_dual_tower_loss + self.cl_loss_weight * cl_loss
        
        # 6. 记录详细的损失信息
        # self._log_cl_loss_details(base_dual_tower_loss, cl_loss, total_loss)
        
        return total_loss
    
    def _should_apply_cl_loss(self):
        """
        判断当前是否应该应用CL损失
        """
        if not self.use_cl_loss:
            return False
        
        if self.training_mode == "parallel":
            # 并行模式：总是应用CL损失
            return True
        elif self.training_mode == "sequential":
            # 串行模式：只在第二阶段应用CL损失
            return self.is_sequential_phase_2
        else:
            return False
    
    def _log_cl_loss_details(self, base_loss, cl_loss, total_loss):
        """
        记录详细的CL损失信息
        """
        # 记录基本的损失信息（每个epoch都记录）
        if self.current_training_epoch <= 5 or self.current_training_epoch % 5 == 0:  # 前5个epoch和每5个epoch记录一次
            logging.info(f"[CL Loss] Epoch {self.current_training_epoch} - Mode: {self.training_mode}")
            logging.info(f"  - Base dual tower loss: {base_loss.item():.6f}")
            logging.info(f"  - CL loss (raw): {cl_loss.item():.6f}")
            logging.info(f"  - CL loss (weighted, w={self.cl_loss_weight}): {(self.cl_loss_weight * cl_loss).item():.6f}")
            logging.info(f"  - Total loss: {total_loss.item():.6f}")
            
            # 记录各个CL组件的损失
            if hasattr(self, 'knowledge_distillation_loss') and self.knowledge_distillation_loss is not None:
                logging.info(f"    * Knowledge distillation: {self.knowledge_distillation_loss.item():.6f}")
            if hasattr(self, 'group_aware_loss') and self.group_aware_loss is not None:
                logging.info(f"    * Group aware: {self.group_aware_loss.item():.6f}")
            if hasattr(self, 'feature_alignment_loss') and self.feature_alignment_loss is not None:
                logging.info(f"    * Feature alignment: {self.feature_alignment_loss.item():.6f}")
            if hasattr(self, 'field_uniformity_loss') and self.field_uniformity_loss is not None:
                logging.info(f"    * Field uniformity: {self.field_uniformity_loss.item():.6f}")
            
            # 记录训练模式状态
            if self.training_mode == "sequential":
                logging.info(f"  - Sequential phase 2 active: {self.is_sequential_phase_2}")
                if self.is_sequential_phase_2 and self.freeze_personalized_in_sequential:
                    logging.info(f"  - Personalized model frozen: {not next(self.personalized_model.parameters()).requires_grad}")
            
            logging.info(f"  - CL should apply: {self._should_apply_cl_loss()}")
            logging.info("=" * 40)
    
    def switch_to_sequential_phase_2(self):
        """
        切换到串行训练的第二阶段
        
        在这个阶段：
        1. 冻结个性化模型（如果配置要求）
        2. 专注于训练非个性化模型
        3. 启用CL损失让非个性化模型学习个性化模型
        """
        if self.training_mode != "sequential":
            logging.warning("switch_to_sequential_phase_2() called but training_mode is not 'sequential'")
            return
        
        self.is_sequential_phase_2 = True
        
        if self.freeze_personalized_in_sequential:
            self._freeze_personalized_model()
        
        logging.info("=== Switched to Sequential Training Phase 2 ===")
        logging.info("  - Personalized model frozen: {}".format(self.freeze_personalized_in_sequential))
        logging.info("  - CL losses enabled for non-personalized model training")
        logging.info("  - Non-personalized model will learn from personalized model predictions")
    
    def switch_to_sequential_phase_1(self):
        """
        切换到串行训练的第一阶段（或重置到并行模式）
        """
        self.is_sequential_phase_2 = False
        
        if self.training_mode == "sequential":
            # 解冻个性化模型以便训练
            self._unfreeze_personalized_model()
            logging.info("=== Switched to Sequential Training Phase 1 ===")
            logging.info("  - Training personalized model only")
            logging.info("  - CL losses disabled")
        else:
            logging.info("=== Reset to Parallel Training Mode ===")
    
    def train_epoch(self, data_generator):
        """
        重写训练epoch方法，支持串行训练模式
        """
        # 更新当前训练epoch
        self.current_training_epoch = self._epoch_index + 1
        
        # 串行模式的阶段切换逻辑
        if self.training_mode == "sequential":
            logging.info(f"[Sequential Mode] Epoch {self.current_training_epoch}: Phase 2 active = {self.is_sequential_phase_2}, Warmup epochs = {self.sequential_warmup_epochs}")
            
            # 检查是否需要切换到第二阶段
            if not self.is_sequential_phase_2 and self.current_training_epoch >= self.sequential_warmup_epochs:
                # 预热期结束，切换到第二阶段
                self.switch_to_sequential_phase_2()
                logging.info(f"=== Epoch {self.current_training_epoch}: Switched to Sequential Phase 2 ===")
            elif self.is_sequential_phase_2:
                logging.info(f"[Sequential Mode] Epoch {self.current_training_epoch}: Running in Phase 2 (CL active)")
            else:
                logging.info(f"[Sequential Mode] Epoch {self.current_training_epoch}: Running in Phase 1 (warmup, CL inactive)")
        
        # 调用父类的训练方法
        super().train_epoch(data_generator)

    def get_cl_training_summary(self):
        """
        获取CL训练状态摘要
        
        Returns:
            dict: CL训练状态信息
        """
        summary = {
            "training_mode": self.training_mode,
            "current_epoch": self.current_training_epoch,
            "cl_loss_weight": self.cl_loss_weight,
            "use_cl_loss": self.use_cl_loss,
            "cl_components": {
                "knowledge_distillation_weight": self.knowledge_distillation_loss_weight,
                "group_aware_weight": self.group_aware_loss_weight,
                "feature_alignment_weight": self.feature_alignment_loss_weight,
                "field_uniformity_weight": self.field_uniformity_loss_weight,
                "temperature": self.temperature
            }
        }
        
        if self.training_mode == "sequential":
            summary["sequential_info"] = {
                "is_phase_2": self.is_sequential_phase_2,
                "warmup_epochs": self.sequential_warmup_epochs,
                "personalized_frozen": self.freeze_personalized_in_sequential and self.is_sequential_phase_2,
                "pretrained_model_path": self.personalized_model_path
            }
        
        if self.use_feature_level_cl:
            summary["feature_level_cl"] = {
                "enabled": True,
                "personalization_features_for_cl": self.personalization_feature_list_for_cl
            }
        
        return summary
    
    def fit(self, data_generator, epochs=1, validation_data=None, **kwargs):
        """
        重写训练方法，添加CL训练状态日志
        """
        # 输出CL训练配置摘要
        cl_summary = self.get_cl_training_summary()
        logging.info("=== DualTowerCL Training Configuration ===")
        logging.info(f"Training mode: {cl_summary['training_mode']}")
        logging.info(f"CL loss weight: {cl_summary['cl_loss_weight']}")
        logging.info(f"Knowledge distillation weight: {cl_summary['cl_components']['knowledge_distillation_weight']}")
        logging.info(f"Group aware loss weight: {cl_summary['cl_components']['group_aware_weight']}")
        logging.info(f"Temperature: {cl_summary['cl_components']['temperature']}")
        
        if self.training_mode == "sequential":
            logging.info(f"Sequential warmup epochs: {cl_summary['sequential_info']['warmup_epochs']}")
            logging.info(f"Freeze personalized in phase 2: {self.freeze_personalized_in_sequential}")
        
        logging.info("=" * 50)
        
        # 调用父类训练方法
        super().fit(data_generator, epochs, validation_data, **kwargs)
        
        # 训练完成后输出CL训练总结
        final_summary = self.get_cl_training_summary()
        logging.info("=== DualTowerCL Training Summary ===")
        logging.info(f"Total epochs trained: {final_summary['current_epoch']}")
        if self.training_mode == "sequential":
            logging.info(f"Sequential phase 2 activated: {final_summary['sequential_info']['is_phase_2']}")
        logging.info("=" * 50)
