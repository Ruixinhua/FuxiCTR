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
import os
import logging
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import FeatureSeparator


class DualTowerRouter:
    """
    双塔路由器：基于is_personalization标签进行硬路由
    """
    
    def __init__(self, personalization_field="is_personalization"):
        """
        初始化路由器
        
        Args:
            personalization_field: 个性化标识字段名
        """
        self.personalization_field = personalization_field
        logging.info(f"DualTowerRouter initialized with personalization field: {personalization_field}")
    
    def get_user_masks(self, inputs):
        """
        获取用户类型掩码
        
        Args:
            inputs: 输入特征字典
            
        Returns:
            tuple: (personalized_mask, non_personalized_mask)
        """
        if self.personalization_field not in inputs:
            # 如果没有个性化标识字段，默认全部为非个性化用户
            batch_size = list(inputs.values())[0].size(0)
            device = list(inputs.values())[0].device
            personalized_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            non_personalized_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            logging.warning(f"Personalization field '{self.personalization_field}' not found, treating all as non-personalized")
        else:
            personalization_flag = inputs[self.personalization_field]
            # 1 代表个性化用户，其他值代表非个性化用户
            personalized_mask = (personalization_flag == 1)
            non_personalized_mask = (personalization_flag != 1)
        
        num_personalized = torch.sum(personalized_mask).item()
        num_non_personalized = torch.sum(non_personalized_mask).item()
        
        logging.debug(f"Batch routing: {num_personalized} personalized users, {num_non_personalized} non-personalized users")
        
        return personalized_mask, non_personalized_mask
    
    def route_predictions(self, personalized_pred, non_personalized_pred, personalized_mask, non_personalized_mask):
        """
        根据用户类型路由预测结果
        
        Args:
            personalized_pred: 个性化塔预测结果
            non_personalized_pred: 非个性化塔预测结果
            personalized_mask: 个性化用户掩码
            non_personalized_mask: 非个性化用户掩码
            
        Returns:
            final_pred: 路由后的最终预测结果
        """

        # 初始化最终预测结果
        final_pred = torch.zeros_like(personalized_pred)
        
        # 个性化用户使用个性化塔的预测
        if torch.sum(personalized_mask) > 0:
            final_pred[personalized_mask] += personalized_pred[personalized_mask]
        
        # 非个性化用户使用非个性化塔的预测
        if torch.sum(non_personalized_mask) > 0:
            final_pred[non_personalized_mask] += non_personalized_pred[non_personalized_mask]
        
        return final_pred


class DualTowerModel(BaseModel):
    """
    双塔模型：两个完全独立的模型分别处理个性化和非个性化用户
    
    训练阶段：
    1. 特征处理：
       - 个性化塔：使用全部特征
       - 非个性化塔：使用全部特征，但个性化数据的个性化特征被mask
    2. 并行训练：两个模型独立训练，使用各自的特征集和数据源
    3. 损失计算：分别计算两个模型的损失，可选择性添加对比学习损失
    
    推理阶段：
    1. 用户类型识别：通过"is_personalization"标签确定用户类型
    2. 特征准备：为每个塔准备对应的特征集（非个性化塔mask个性化特征）
    3. 模型推理：两个模型分别在对应数据上进行推理
    4. 结果路由：根据路由策略选择对应模型的输出结果
    """
    
    def __init__(self,
                 feature_map,
                 model_id="DualTowerModel",
                 gpu=-1,
                 learning_rate=1e-3,
                 # 个性化塔配置
                 personalized_model_type="PNN",
                 personalized_model_params=None,
                 # 非个性化塔配置
                 non_personalized_model_type="DCNv3",
                 non_personalized_model_params=None,
                 # 特征分离配置
                 personalization_feature_list=None,
                 personalization_field="is_personalization",
                 use_mask_for_all: bool = False,
                 # 损失权重配置
                 personalized_loss_weight=1.0,
                 non_personalized_loss_weight=1.0,
                 # 训练数据配置
                 personalized_model_use_all_data=False,  # 个性化模型是否使用全部数据（默认只用个性化数据）
                 non_personalized_model_use_all_data=True,  # 非个性化模型是否使用全部数据（默认使用全部数据）
                 # 对比学习配置
                 # 分塔监控配置
                 use_tower_specific_monitoring=True,  # 是否启用分塔监控
                 personalized_monitor_metric="AUC_group_1.0",  # 个性化塔监控指标
                 non_personalized_monitor_metric="AUC_group_2.0",  # 非个性化塔监控指标
                 tower_patience=3,  # 分塔早停耐心值
                 save_tower_models=True,  # 是否分别保存塔模型
                 # 其他配置
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        
        super(DualTowerModel, self).__init__(feature_map,
                                           model_id=model_id,
                                           gpu=gpu,
                                           embedding_regularizer=embedding_regularizer,
                                           net_regularizer=net_regularizer,
                                           **kwargs)
        
        # 保存配置参数
        self.personalized_model_type = personalized_model_type
        self.non_personalized_model_type = non_personalized_model_type
        self.personalized_model_params = personalized_model_params or {}
        self.non_personalized_model_params = non_personalized_model_params or {}
        self.personalization_feature_list = personalization_feature_list or []
        self.personalization_field = personalization_field
        
        # 损失权重配置
        self.personalized_loss_weight = personalized_loss_weight
        self.non_personalized_loss_weight = non_personalized_loss_weight
        
        # 训练数据配置
        self.personalized_model_use_all_data = personalized_model_use_all_data
        self.non_personalized_model_use_all_data = non_personalized_model_use_all_data

        # 分塔监控配置
        self.use_tower_specific_monitoring = use_tower_specific_monitoring
        self.personalized_monitor_metric = personalized_monitor_metric
        self.non_personalized_monitor_metric = non_personalized_monitor_metric
        self.tower_patience = tower_patience
        self.save_tower_models = save_tower_models
        
        # 初始化特征分离器
        self.personalization_feature_list = [f for f in self.personalization_feature_list if f in self.feature_map.features]
        if len(self.personalization_feature_list) > 0:
            logging.info(f'Personalization features: {self.personalization_feature_list}')
        self.feature_separator = FeatureSeparator(self.personalization_feature_list, self.feature_map)
        
        # 初始化路由器
        self.router = DualTowerRouter(self.personalization_field)
        self.use_mask_for_all = use_mask_for_all
        
        # 初始化两个独立的模型
        self._init_personalized_model()
        self._init_non_personalized_model()
        
        # 初始化分塔监控
        if self.use_tower_specific_monitoring:
            self._init_tower_monitoring()
        
        # 编译模型
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
        
        logging.info(f"DualTowerModel initialized:")
        logging.info(f"  - Personalized tower: {personalized_model_type}")
        logging.info(f"  - Non-personalized tower: {non_personalized_model_type}")
        logging.info(f"  - Personalization features: {self.personalization_feature_list}")
        logging.info(f"  - Personalized model use all data: {personalized_model_use_all_data}")
        logging.info(f"  - Non-personalized model use all data: {non_personalized_model_use_all_data}")

    def _init_model(self, model_type, model_params):
        if model_type == "PNN":
            from model_zoo.DTCN.src import PNNAdapter
            model = PNNAdapter(
                feature_map=self.feature_map,
                output_mode="SingleTower",
                **model_params
            )
        elif model_type == "DCNv3":
            from model_zoo.DTCN.src import DCNv3Adapter
            model = DCNv3Adapter(
                feature_map=self.feature_map,
                output_mode="SingleTower",
                **model_params
            )
        elif model_type == "FinalNet":
            from model_zoo.DTCN.src import FinalNetAdapter
            model = FinalNetAdapter(
                feature_map=self.feature_map,
                output_mode="SingleTower",
                **model_params
            )
        else:
            raise NotImplementedError(f"Model type '{model_type}' not implemented.")
        return model

    def _init_personalized_model(self):
        """
        初始化个性化模型（使用全部特征）
        """
        # 合并参数
        personalized_params = {
            "embedding_dim": self.personalized_model_params.get("embedding_dim", 10),
            "output_activation": self.output_activation,
            **self.personalized_model_params
        }
        self.personalized_model = self._init_model(self.personalized_model_type, personalized_params)
        logging.info(f"Personalized model initialized: {self.personalized_model_type}")
    
    def _init_non_personalized_model(self):
        """
        初始化非个性化模型（使用全部特征，但训练时会mask个性化特征）
        """
        # 合并参数
        non_personalized_params = {
            "embedding_dim": self.non_personalized_model_params.get("embedding_dim", 10),
            "output_activation": self.output_activation,
            **self.non_personalized_model_params
        }
        self.non_personalized_model = self._init_model(self.non_personalized_model_type, non_personalized_params)
        
        logging.info(f"Non-personalized model initialized: {self.non_personalized_model_type} (using full feature set with masking)")
    
    def forward(self, inputs):
        """
        前向传播
        
        Args:
            inputs: 输入特征字典
            
        Returns:
            return_dict: 包含预测结果的字典
        """
        # 获取输入特征
        X = self.get_inputs(inputs)
        
        # 1. 获取用户类型掩码
        personalized_mask, non_personalized_mask = self.router.get_user_masks(X)
        if not self.use_mask_for_all:
            # set 1 for both personalized and non-personalized users
            personalized_mask = torch.ones_like(personalized_mask, device=personalized_mask.device)
            non_personalized_mask = torch.ones_like(non_personalized_mask, device=non_personalized_mask.device)
        
        # 2. 特征分离（新逻辑：非个性化模型使用mask后的全特征）
        personalized_features, non_personalized_features = self.feature_separator.separate_features(X, personalized_mask)
        
        # 3. 两个模型分别进行推理
        # 个性化模型使用全部特征
        personalized_return_dict = self.personalized_model.get_model_return_dict(personalized_features)
        personalized_pred = personalized_return_dict["y_pred"]
        
        # 非个性化模型使用mask后的全特征
        non_personalized_return_dict = self.non_personalized_model.get_model_return_dict(non_personalized_features)
        non_personalized_pred = non_personalized_return_dict["y_pred"]
        
        # 4. 路由预测结果
        final_pred = self.router.route_predictions(
            personalized_pred, non_personalized_pred,
            personalized_mask, non_personalized_mask
        )
        
        # 构建返回字典
        return_dict = {
            "y_pred": final_pred,
            "personalized_pred": personalized_pred,
            "non_personalized_pred": non_personalized_pred,
            "personalized_features": personalized_features,
            "non_personalized_features": non_personalized_features,
            "personalized_mask": personalized_mask,
            "non_personalized_mask": non_personalized_mask
        }
        
        # 如果个性化模型有额外输出（如DCNv3的y_d, y_s，FinalNet的y1, y2），也包含进来
        if "y_d" in personalized_return_dict:
            return_dict["personalized_y_d"] = personalized_return_dict["y_d"]
            return_dict["personalized_y_s"] = personalized_return_dict["y_s"]
        if "y1" in personalized_return_dict:
            return_dict["personalized_y1"] = personalized_return_dict["y1"]
            return_dict["personalized_y2"] = personalized_return_dict["y2"]
        
        if "y_d" in non_personalized_return_dict:
            return_dict["non_personalized_y_d"] = non_personalized_return_dict["y_d"]
            return_dict["non_personalized_y_s"] = non_personalized_return_dict["y_s"]
        if "y1" in non_personalized_return_dict:
            return_dict["non_personalized_y1"] = non_personalized_return_dict["y1"]
            return_dict["non_personalized_y2"] = non_personalized_return_dict["y2"]
        
        return return_dict
    
    def add_loss(self, return_dict, y_true):
        """
        计算组合损失函数
        
        新的训练流程：
        - 个性化模型：默认只用个性化用户数据，使用全部特征（可配置使用全部数据）
        - 非个性化模型：默认使用全部数据，使用全特征但个性化数据的个性化特征被mask（可配置）
        
        Args:
            return_dict: 模型输出字典
            y_true: 真实标签
            
        Returns:
            total_loss: 计算的总损失值
        """
        personalized_pred = return_dict["personalized_pred"]
        non_personalized_pred = return_dict["non_personalized_pred"]
        personalized_mask = return_dict["personalized_mask"]
        non_personalized_mask = return_dict["non_personalized_mask"]
        
        total_loss = torch.tensor(0.0, device=y_true.device)
        
        # 1. 个性化模型损失
        # 根据配置决定使用哪些样本训练个性化模型
        if self.personalized_model_use_all_data:
            # 使用全部数据训练个性化模型
            personalized_training_mask = torch.ones_like(personalized_mask, dtype=torch.bool)
        else:
            # 默认：只使用个性化用户数据训练个性化模型
            personalized_training_mask = personalized_mask
        
        personalized_training_count = torch.sum(personalized_training_mask).item()
        if personalized_training_count > 0:
            personalized_y_true = y_true[personalized_training_mask]
            personalized_y_pred = personalized_pred[personalized_training_mask]
            
            # 如果个性化模型有自定义损失（如DCNv3、FinalNet）
            if hasattr(self.personalized_model, 'has_custom_loss') and self.personalized_model.has_custom_loss():
                # 构建个性化模型的return_dict
                personalized_return_dict = {"y_pred": personalized_y_pred}
                # DCNv3的输出
                if "personalized_y_d" in return_dict:
                    personalized_return_dict["y_d"] = return_dict["personalized_y_d"][personalized_training_mask]
                    personalized_return_dict["y_s"] = return_dict["personalized_y_s"][personalized_training_mask]
                # FinalNet的输出
                if "personalized_y1" in return_dict:
                    personalized_return_dict["y1"] = return_dict["personalized_y1"][personalized_training_mask]
                    personalized_return_dict["y2"] = return_dict["personalized_y2"][personalized_training_mask]
                
                personalized_loss = self.personalized_model.compute_custom_loss(
                    personalized_return_dict, personalized_y_true, self.loss_fn
                )
            else:
                personalized_loss = self.loss_fn(personalized_y_pred, personalized_y_true, reduction='mean')
            
            total_loss += self.personalized_loss_weight * personalized_loss
            
            data_usage = "all data" if self.personalized_model_use_all_data else "personalized data only"
            logging.debug(f"Personalized loss: {personalized_loss.item():.6f} (samples: {personalized_training_count}, using: {data_usage})")
        
        # 2. 非个性化模型损失
        # 根据配置决定使用哪些样本训练非个性化模型
        if self.non_personalized_model_use_all_data:
            # 默认：使用全部数据训练非个性化模型
            non_personalized_training_mask = torch.ones_like(non_personalized_mask, dtype=torch.bool)
        else:
            # 只使用非个性化用户数据训练非个性化模型
            non_personalized_training_mask = non_personalized_mask
        
        non_personalized_training_count = torch.sum(non_personalized_training_mask).item()
        if non_personalized_training_count > 0:
            non_personalized_y_true = y_true[non_personalized_training_mask]
            non_personalized_y_pred = non_personalized_pred[non_personalized_training_mask]
            
            # 如果非个性化模型有自定义损失（如DCNv3、FinalNet）
            if hasattr(self.non_personalized_model, 'has_custom_loss') and self.non_personalized_model.has_custom_loss():
                # 构建非个性化模型的return_dict
                non_personalized_return_dict = {"y_pred": non_personalized_y_pred}
                # DCNv3的输出
                if "non_personalized_y_d" in return_dict:
                    non_personalized_return_dict["y_d"] = return_dict["non_personalized_y_d"][non_personalized_training_mask]
                    non_personalized_return_dict["y_s"] = return_dict["non_personalized_y_s"][non_personalized_training_mask]
                # FinalNet的输出
                if "non_personalized_y1" in return_dict:
                    non_personalized_return_dict["y1"] = return_dict["non_personalized_y1"][non_personalized_training_mask]
                    non_personalized_return_dict["y2"] = return_dict["non_personalized_y2"][non_personalized_training_mask]
                
                non_personalized_loss = self.non_personalized_model.compute_custom_loss(
                    non_personalized_return_dict, non_personalized_y_true, self.loss_fn
                )
            else:
                non_personalized_loss = self.loss_fn(non_personalized_y_pred, non_personalized_y_true, reduction='mean')
            
            total_loss += self.non_personalized_loss_weight * non_personalized_loss
            
            data_usage = "all data" if self.non_personalized_model_use_all_data else "non-personalized data only"
            logging.debug(f"Non-personalized loss: {non_personalized_loss.item():.6f} (samples: {non_personalized_training_count}, using: {data_usage})")
        total_loss +=  self.regularization_loss()
        return total_loss

    def _init_tower_monitoring(self):
        """
        初始化分塔监控系统
        """
        import numpy as np
        
        # 分塔监控状态
        self.tower_monitoring = {
            "personalized": {
                "best_metric": -np.inf if "AUC" in self.personalized_monitor_metric else np.inf,
                "best_epoch": 0,
                "patience_count": 0,
                "model_path": self.checkpoint,
                "is_better": lambda current, best: current > best if "AUC" in self.personalized_monitor_metric else current < best
            },
            "non_personalized": {
                "best_metric": -np.inf if "AUC" in self.non_personalized_monitor_metric else np.inf,
                "best_epoch": 0,
                "patience_count": 0,
                "model_path": self.checkpoint,
                "is_better": lambda current, best: current > best if "AUC" in self.non_personalized_monitor_metric else current < best
            }
        }
        
        logging.info(f"Tower monitoring initialized:")
        logging.info(f"  - Personalized tower metric: {self.personalized_monitor_metric}")
        logging.info(f"  - Non-personalized tower metric: {self.non_personalized_monitor_metric}")
        logging.info(f"  - Tower patience: {self.tower_patience}")
        logging.info(f"  - Save tower models: {self.save_tower_models}")
    
    def update_tower_monitoring(self, eval_metrics, current_epoch):
        """
        更新分塔监控状态
        
        Args:
            eval_metrics: 评估指标字典
            current_epoch: 当前epoch
            
        Returns:
            dict: 更新信息
        """
        if not self.use_tower_specific_monitoring:
            return {}
        
        update_info = {}
        
        # 更新个性化塔监控
        if self.personalized_monitor_metric in eval_metrics:
            current_metric = eval_metrics[self.personalized_monitor_metric]
            tower_info = self.tower_monitoring["personalized"]
            
            if tower_info["is_better"](current_metric, tower_info["best_metric"]):
                # 找到更好的个性化塔模型
                tower_info["best_metric"] = current_metric
                tower_info["best_epoch"] = current_epoch
                tower_info["patience_count"] = 0
                
                if self.save_tower_models:
                    # 保存个性化塔最佳模型
                    tower_model_path = f"{self.checkpoint}_personalized_best.model"
                    self._save_personalized_model(tower_model_path)
                    tower_info["model_path"] = tower_model_path
                
                update_info["personalized"] = {
                    "metric": self.personalized_monitor_metric,
                    "value": current_metric,
                    "epoch": current_epoch,
                    "improved": True
                }
                
                logging.info(f"New best personalized tower: {self.personalized_monitor_metric}={current_metric:.6f} at epoch {current_epoch}")
            else:
                tower_info["patience_count"] += 1
                update_info["personalized"] = {
                    "metric": self.personalized_monitor_metric,
                    "value": current_metric,
                    "epoch": current_epoch,
                    "improved": False,
                    "patience": tower_info["patience_count"]
                }
        
        # 更新非个性化塔监控
        if self.non_personalized_monitor_metric in eval_metrics:
            current_metric = eval_metrics[self.non_personalized_monitor_metric]
            tower_info = self.tower_monitoring["non_personalized"]
            
            if tower_info["is_better"](current_metric, tower_info["best_metric"]):
                # 找到更好的非个性化塔模型
                tower_info["best_metric"] = current_metric
                tower_info["best_epoch"] = current_epoch
                tower_info["patience_count"] = 0
                
                if self.save_tower_models:
                    # 保存非个性化塔最佳模型
                    tower_model_path = f"{self.checkpoint}_non_personalized_best.model"
                    self._save_non_personalized_model(tower_model_path)
                    tower_info["model_path"] = tower_model_path
                
                update_info["non_personalized"] = {
                    "metric": self.non_personalized_monitor_metric,
                    "value": current_metric,
                    "epoch": current_epoch,
                    "improved": True
                }
                
                logging.info(f"New best non-personalized tower: {self.non_personalized_monitor_metric}={current_metric:.6f} at epoch {current_epoch}")
            else:
                tower_info["patience_count"] += 1
                update_info["non_personalized"] = {
                    "metric": self.non_personalized_monitor_metric,
                    "value": current_metric,
                    "epoch": current_epoch,
                    "improved": False,
                    "patience": tower_info["patience_count"]
                }
        
        return update_info
    
    def should_early_stop_towers(self):
        """
        检查是否应该基于分塔监控进行早停
        
        Returns:
            bool: 是否应该早停
        """
        if not self.use_tower_specific_monitoring:
            return False
        
        # 检查两个塔是否都达到了耐心值
        personalized_patience_exceeded = (
            self.tower_monitoring["personalized"]["patience_count"] >= self.tower_patience
        )
        non_personalized_patience_exceeded = (
            self.tower_monitoring["non_personalized"]["patience_count"] >= self.tower_patience
        )
        
        # 可以选择不同的早停策略：
        # 1. 任一塔达到耐心值就停止
        # 2. 两个塔都达到耐心值才停止
        # 这里采用策略2（两个塔都需要达到耐心值）
        should_stop = personalized_patience_exceeded and non_personalized_patience_exceeded
        
        if should_stop:
            logging.info("Early stopping triggered by tower-specific monitoring:")
            logging.info(f"  - Personalized tower patience: {self.tower_monitoring['personalized']['patience_count']}/{self.tower_patience}")
            logging.info(f"  - Non-personalized tower patience: {self.tower_monitoring['non_personalized']['patience_count']}/{self.tower_patience}")
        
        return should_stop
    
    def _save_personalized_model(self, model_path):
        """
        保存个性化塔的最佳模型
        
        Args:
            model_path: 模型保存路径
        """
        try:
            # 保存模型权重和优化器状态，确保可以正确恢复训练
            personalized_state = {
                "model_state_dict": self.personalized_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),  # 保存optimizer状态
                "model_type": self.personalized_model_type,
                "model_params": self.personalized_model_params,
                "best_metric": self.tower_monitoring["personalized"]["best_metric"],
                "best_epoch": self.tower_monitoring["personalized"]["best_epoch"]
            }
            
            torch.save(personalized_state, model_path)
            logging.debug(f"Personalized tower model saved to: {model_path}")
            
        except Exception as e:
            logging.error(f"Failed to save personalized tower model: {e}")
    
    def _save_non_personalized_model(self, model_path):
        """
        保存非个性化塔的最佳模型
        
        Args:
            model_path: 模型保存路径
        """
        try:
            # 保存模型权重和优化器状态，确保可以正确恢复训练
            non_personalized_state = {
                "model_state_dict": self.non_personalized_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),  # 保存optimizer状态
                "model_type": self.non_personalized_model_type,
                "model_params": self.non_personalized_model_params,
                "best_metric": self.tower_monitoring["non_personalized"]["best_metric"],
                "best_epoch": self.tower_monitoring["non_personalized"]["best_epoch"]
            }
            
            torch.save(non_personalized_state, model_path)
            logging.debug(f"Non-personalized tower model saved to: {model_path}")
            
        except Exception as e:
            logging.error(f"Failed to save non-personalized tower model: {e}")
    
    def get_tower_monitoring_summary(self):
        """
        获取分塔监控摘要信息
        
        Returns:
            dict: 监控摘要
        """
        if not self.use_tower_specific_monitoring:
            return {}
        
        summary = {
            "personalized_tower": {
                "best_metric": self.personalized_monitor_metric,
                "best_value": self.tower_monitoring["personalized"]["best_metric"],
                "best_epoch": self.tower_monitoring["personalized"]["best_epoch"],
                "model_path": self.tower_monitoring["personalized"]["model_path"]
            },
            "non_personalized_tower": {
                "best_metric": self.non_personalized_monitor_metric,
                "best_value": self.tower_monitoring["non_personalized"]["best_metric"],
                "best_epoch": self.tower_monitoring["non_personalized"]["best_epoch"],
                "model_path": self.tower_monitoring["non_personalized"]["model_path"]
            }
        }
        
        return summary
    
    def _load_tower_optimal_models(self, load_optimizer=False):
        """
        加载分塔最佳模型权重，组合成最优的双塔模型
        
        Args:
            load_optimizer: 是否加载optimizer状态（默认False，因为训练已完成）
        """

        personalized_loaded = False
        non_personalized_loaded = False
        
        # 加载个性化塔最佳模型
        personalized_path = self.tower_monitoring["personalized"]["model_path"]
        if personalized_path and os.path.exists(personalized_path):
            try:
                # 使用weights_only=False来兼容包含自定义对象的模型文件
                personalized_state = torch.load(personalized_path, map_location=self.device, weights_only=False)
                self.personalized_model.load_state_dict(personalized_state["model_state_dict"])
                
                # 可选：加载optimizer状态（用于继续训练）
                if load_optimizer and "optimizer_state_dict" in personalized_state:
                    try:
                        self.optimizer.load_state_dict(personalized_state["optimizer_state_dict"])
                        logging.info(f"  - Optimizer state loaded from personalized tower checkpoint")
                    except Exception as e:
                        logging.warning(f"  - Failed to load optimizer state: {e}")
                
                personalized_loaded = True
                logging.info(f"Loaded personalized tower best model from: {personalized_path}")
                logging.info(f"  - Best {self.personalized_monitor_metric}: {self.tower_monitoring['personalized']['best_metric']:.6f} at epoch {self.tower_monitoring['personalized']['best_epoch']}")
            except Exception as e:
                logging.error(f"Failed to load personalized tower model: {e}")

        # 加载非个性化塔最佳模型
        non_personalized_path = self.tower_monitoring["non_personalized"]["model_path"]
        if non_personalized_path and os.path.exists(non_personalized_path):
            try:
                # 使用weights_only=False来兼容包含自定义对象的模型文件
                non_personalized_state = torch.load(non_personalized_path, map_location=self.device, weights_only=False)
                self.non_personalized_model.load_state_dict(non_personalized_state["model_state_dict"])
                
                # 可选：加载optimizer状态（用于继续训练）
                # 注意：如果两个塔都加载optimizer，以非个性化塔的为准（后加载的会覆盖）
                if load_optimizer and "optimizer_state_dict" in non_personalized_state:
                    try:
                        self.optimizer.load_state_dict(non_personalized_state["optimizer_state_dict"])
                        logging.info(f"  - Optimizer state loaded from non-personalized tower checkpoint")
                    except Exception as e:
                        logging.warning(f"  - Failed to load optimizer state: {e}")
                
                non_personalized_loaded = True
                logging.info(f"Loaded non-personalized tower best model from: {non_personalized_path}")
                logging.info(f"  - Best {self.non_personalized_monitor_metric}: {self.tower_monitoring['non_personalized']['best_metric']:.6f} at epoch {self.tower_monitoring['non_personalized']['best_epoch']}")
            except Exception as e:
                logging.error(f"Failed to load non-personalized tower model: {e}")
        # 检查加载状态
        if personalized_loaded and non_personalized_loaded:
            logging.info("✅ Successfully loaded optimal tower combination:")
            logging.info(f"  - Personalized tower: epoch {self.tower_monitoring['personalized']['best_epoch']} ({self.personalized_monitor_metric}={self.tower_monitoring['personalized']['best_metric']:.6f})")
            logging.info(f"  - Non-personalized tower: epoch {self.tower_monitoring['non_personalized']['best_epoch']} ({self.non_personalized_monitor_metric}={self.tower_monitoring['non_personalized']['best_metric']:.6f})")
        elif personalized_loaded or non_personalized_loaded:
            logging.warning("⚠️  Partially loaded tower models:")
            if personalized_loaded:
                logging.warning("  - Personalized tower: loaded successfully")
            else:
                logging.warning("  - Personalized tower: failed to load, using current weights")
            if non_personalized_loaded:
                logging.warning("  - Non-personalized tower: loaded successfully")
            else:
                logging.warning("  - Non-personalized tower: failed to load, using current weights")
        else:
            logging.warning("❌ Failed to load any tower models, falling back to original model")
            logging.info("Load best model: {}".format(self.checkpoint))
            self.load_weights(self.checkpoint)
        loaded_path = [self.checkpoint, personalized_path, non_personalized_path]
        if not self._save_checkpoints:
            for model_path in loaded_path:
                if os.path.exists(model_path):
                    logging.info("Remove checkpoints: {}".format(model_path))
                    os.remove(model_path)

    def fit(self, data_generator, epochs=1, validation_data=None, **kwargs):
        """
        重写训练方法，集成分塔监控
        """
        import numpy as np
        
        # 调用父类的训练准备
        self.valid_gen = validation_data
        self._max_gradient_norm = kwargs.get('max_gradient_norm', 10.)
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch
        
        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        
        for epoch in range(epochs):
            self._epoch_index = epoch
            
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        
        logging.info("Training finished.")
        
        # 输出分塔监控摘要
        if self.use_tower_specific_monitoring:
            tower_summary = self.get_tower_monitoring_summary()
            logging.info("Tower monitoring summary:")
            for tower_name, info in tower_summary.items():
                logging.info(f"  {tower_name}:")
                logging.info(f"    Best {info['best_metric']}: {info['best_value']:.6f} at epoch {info['best_epoch']}")
                if info['model_path']:
                    logging.info(f"    Model saved to: {info['model_path']}")
        
        # 加载最佳模型
        if self.use_tower_specific_monitoring and self.save_tower_models:
            # 使用分塔最佳模型组合
            self._load_tower_optimal_models()
        else:
            # 使用原来的逻辑
            logging.info("Load best model: {}".format(self.checkpoint))
            self.load_weights(self.checkpoint)
            if not self._save_checkpoints:
                logging.info("Remove checkpoints: {}".format(self.checkpoint))
                os.remove(self.checkpoint)
    
    def eval_step(self):
        """
        重写评估步骤，集成分塔监控
        """
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.checkpoint_and_earlystop(val_logs)
        # 更新分塔监控
        if self.use_tower_specific_monitoring:
            self.update_tower_monitoring(val_logs, self._epoch_index + 1)
            self._stop_training = self.should_early_stop_towers()
            # 检查分塔早停
            if self._stop_training:
                logging.info("Early stopping triggered by tower-specific monitoring")
                return

        # 原来的早停逻辑
        self.train()
