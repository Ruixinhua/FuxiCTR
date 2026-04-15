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
VanillaKD: Standard Two-Stage Knowledge Distillation for CTR/CVR Prediction

Two-stage pipeline:
  Stage 1: Train a teacher model using ALL features (personalized + non-personalized).
           If teacher_model_path is provided, this stage is skipped.
           Otherwise, the teacher is trained automatically within fit().
  Stage 2: Freeze the teacher, train a student model using ONLY non-personalized features,
           with an auxiliary KL-divergence loss that aligns the student's predictions
           to the teacher's soft targets (Hinton et al., 2015).

Data setup:
  - is_personalization == 1 (group_id=1): rows with ALL features available (teacher data)
  - is_personalization != 1 (group_id=2): rows with only NP features (student data)

Routing:
  - Training: teacher trains on all data, student trains on all data (personalized features
    masked for group_id=1 rows), KD loss applied only on group_id=1 rows
  - Inference: group_id=1 -> teacher prediction, group_id=2 -> student prediction

Supported backbones: PNN, FinalNet, DCNv3 (FCN).

Reference:
  Hinton et al., "Distilling the Knowledge in Neural Networks", NeurIPS Workshop 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import FeatureSeparator, get_optimizer
from fuxictr.pytorch.backbone import build_backbone


class VanillaKD(BaseModel):

    def __init__(self,
                 feature_map,
                 model_id="VanillaKD",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 # KD-specific parameters
                 teacher_model_path=None,
                 teacher_pretrain_epochs=100,
                 kd_temperature=4.0,
                 kd_loss_weight=1.0,
                 base_loss_weight=1.0,
                 kd_loss_type="kl",
                 # Feature separation
                 personalization_feature_list=None,
                 personalization_field="is_personalization",
                 # Backbone config
                 backbone_type="PNN",
                 # PNN params
                 hidden_units=[400, 400, 400],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 product_type="inner",
                 # FinalNet params
                 block_type="2B",
                 use_feature_gating=False,
                 block1_hidden_units=[64, 64, 64],
                 block1_hidden_activations=None,
                 block1_dropout=0,
                 block2_hidden_units=[64, 64, 64],
                 block2_hidden_activations=None,
                 block2_dropout=0,
                 residual_type="concat",
                 # DCNv3 params
                 num_deep_cross_layers=4,
                 num_shallow_cross_layers=4,
                 deep_net_dropout=0.1,
                 shallow_net_dropout=0.3,
                 layer_norm=True,
                 num_heads=1,
                 # Standard params
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):

        super(VanillaKD, self).__init__(feature_map,
                                        model_id=model_id,
                                        gpu=gpu,
                                        embedding_regularizer=embedding_regularizer,
                                        net_regularizer=net_regularizer,
                                        **kwargs)

        self.kd_temperature = kd_temperature
        self.kd_loss_weight = kd_loss_weight
        self.base_loss_weight = base_loss_weight
        self.kd_loss_type = kd_loss_type
        self.teacher_model_path = teacher_model_path
        self.personalization_feature_list = personalization_feature_list or []
        self.personalization_field = personalization_field
        self._learning_rate = learning_rate
        self._optimizer_name = kwargs["optimizer"]
        self.teacher_pretrain_epochs = teacher_pretrain_epochs

        self.personalization_feature_list = [
            f for f in self.personalization_feature_list
            if f in self.feature_map.features
        ]
        self.feature_separator = FeatureSeparator(
            self.personalization_feature_list, self.feature_map
        )

        # Collect backbone kwargs
        backbone_kwargs = dict(
            embedding_dim=embedding_dim,
            hidden_units=hidden_units, hidden_activations=hidden_activations,
            net_dropout=net_dropout, batch_norm=batch_norm, product_type=product_type,
            block_type=block_type, use_feature_gating=use_feature_gating,
            block1_hidden_units=block1_hidden_units,
            block1_hidden_activations=block1_hidden_activations,
            block1_dropout=block1_dropout,
            block2_hidden_units=block2_hidden_units,
            block2_hidden_activations=block2_hidden_activations,
            block2_dropout=block2_dropout,
            residual_type=residual_type,
            num_deep_cross_layers=num_deep_cross_layers,
            num_shallow_cross_layers=num_shallow_cross_layers,
            deep_net_dropout=deep_net_dropout, shallow_net_dropout=shallow_net_dropout,
            layer_norm=layer_norm, num_heads=num_heads,
        )

        self.teacher_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)
        self.student_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)

        self._teacher_pretrained = False
        if self.teacher_model_path and os.path.exists(self.teacher_model_path):
            self._freeze_teacher()
            self._teacher_pretrained = True

        self._training_phase = "student" if self._teacher_pretrained else "teacher"

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        logging.info(f"VanillaKD initialized: backbone={backbone_type}, "
                     f"T={kd_temperature}, kd_weight={kd_loss_weight}, "
                     f"kd_type={kd_loss_type}")

    def _get_personalized_mask(self, X):
        if self.personalization_field in X:
            flag = X[self.personalization_field]
            return (flag == 1), (flag != 1)
        batch_size = list(X.values())[0].size(0)
        device = list(X.values())[0].device
        return (torch.zeros(batch_size, dtype=torch.bool, device=device),
                torch.ones(batch_size, dtype=torch.bool, device=device))

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        personalized_mask, non_personalized_mask = self._get_personalized_mask(X)

        if self._training_phase == "teacher":
            logit = self.teacher_backbone(X)
            y_pred = self.output_activation(logit)
            return {"y_pred": y_pred, "logit": logit,
                    "personalized_mask": personalized_mask,
                    "non_personalized_mask": non_personalized_mask}

        _, student_X = self.feature_separator.separate_features(X, personalized_mask)
        student_logit = self.student_backbone(student_X)
        student_y_pred = self.output_activation(student_logit)

        with torch.no_grad():
            teacher_logit = self.teacher_backbone(X)
            teacher_y_pred = self.output_activation(teacher_logit)

        final_pred = torch.zeros_like(student_y_pred)
        if personalized_mask.any():
            final_pred[personalized_mask] = teacher_y_pred[personalized_mask]
        if non_personalized_mask.any():
            final_pred[non_personalized_mask] = student_y_pred[non_personalized_mask]

        return {"y_pred": final_pred,
                "student_y_pred": student_y_pred, "student_logit": student_logit,
                "teacher_y_pred": teacher_y_pred.detach(),
                "teacher_logit": teacher_logit.detach(),
                "personalized_mask": personalized_mask,
                "non_personalized_mask": non_personalized_mask}

    def add_loss(self, return_dict, y_true):
        personalized_mask = return_dict["personalized_mask"]
        if self._training_phase == "teacher":
            return self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')

        base_loss = self.loss_fn(return_dict["student_y_pred"], y_true, reduction='mean')
        total_loss = self.base_loss_weight * base_loss

        if "teacher_logit" in return_dict and personalized_mask.any():
            kd_loss = self._compute_kd_loss(
                return_dict["student_logit"][personalized_mask],
                return_dict["teacher_logit"][personalized_mask])
            total_loss = total_loss + self.kd_loss_weight * kd_loss
        return total_loss

    def _freeze_teacher(self):
        for name, param in self.named_parameters():
            if name.startswith("teacher_"):
                param.requires_grad = False

    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        if self._teacher_pretrained:
            self._training_phase = "student"
            super().fit(data_generator, epochs=epochs,
                        validation_data=validation_data,
                        max_gradient_norm=max_gradient_norm, **kwargs)
            return

        # Phase 1: Train teacher
        teacher_epochs = self.teacher_pretrain_epochs or epochs
        logging.info("=" * 60)
        logging.info(f"=== Phase 1: Training teacher ({teacher_epochs} epochs) ===")
        logging.info("=" * 60)
        self._training_phase = "teacher"
        self.optimizer = get_optimizer(self._optimizer_name,
                                      [p for n, p in self.named_parameters()
                                       if n.startswith("teacher_")],
                                      self._learning_rate)
        teacher_checkpoint = self.checkpoint + ".teacher"
        orig_checkpoint = self.checkpoint
        self.checkpoint = teacher_checkpoint
        super().fit(data_generator, epochs=teacher_epochs,
                    validation_data=validation_data,
                    max_gradient_norm=max_gradient_norm, **kwargs)
        self.checkpoint = orig_checkpoint
        if os.path.exists(teacher_checkpoint):
            self.load_weights(teacher_checkpoint)
            os.remove(teacher_checkpoint)
        self._freeze_teacher()
        self._teacher_pretrained = True

        # Phase 2: Train student with KD
        logging.info("=" * 60)
        logging.info(f"=== Phase 2: Training student ({epochs} epochs) with KD ===")
        logging.info("=" * 60)
        self._training_phase = "student"
        self.optimizer = get_optimizer(self._optimizer_name,
                                      [p for n, p in self.named_parameters()
                                       if p.requires_grad],
                                      self._learning_rate)
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._stop_training = False
        super().fit(data_generator, epochs=epochs,
                    validation_data=validation_data,
                    max_gradient_norm=max_gradient_norm, **kwargs)

    def _compute_kd_loss(self, student_logit, teacher_logit):
        T = self.kd_temperature
        if self.kd_loss_type == "kl":
            s_prob = torch.sigmoid(student_logit / T).clamp(1e-7, 1 - 1e-7)
            t_prob = torch.sigmoid(teacher_logit / T).clamp(1e-7, 1 - 1e-7)
            kd_loss = (t_prob * torch.log(t_prob / s_prob) +
                       (1 - t_prob) * torch.log((1 - t_prob) / (1 - s_prob)))
            return kd_loss.mean() * (T * T)
        elif self.kd_loss_type == "mse":
            return F.mse_loss(student_logit, teacher_logit)
        elif self.kd_loss_type == "cosine":
            return 1 - F.cosine_similarity(student_logit, teacher_logit, dim=-1).mean()
        else:
            raise ValueError(f"Unknown kd_loss_type: {self.kd_loss_type}")
