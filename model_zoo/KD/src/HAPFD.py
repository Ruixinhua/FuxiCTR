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
HA-PFD: Hardness-Aware Privileged Features Distillation with Latent Alignment

Extends PFD with two key innovations:
  1. Focal-style distillation loss: adaptively adjusts the weight of each instance
     based on its "hardness" — poorly predicted instances where the teacher's guidance
     is most needed receive higher distillation weight.
  2. Latent-level distillation: aligns intermediate representations between teacher
     and student via a straightforward layer alignment approach, facilitating the
     student's representation learning.

Architecture:
  - Teacher backbone: uses ALL features (personalized + non-personalized)
  - Student backbone: uses ONLY non-personalized features (masked)
  - Both trained jointly (online distillation, same as PFD)
  - Projection heads for latent alignment when teacher/student dims differ

Loss:
  L = alpha * BCE(student, label)
    + alpha_t * BCE(teacher, label)         [on group_id=1 only]
    + beta * focal_KD(student, teacher)     [on group_id=1 only]
    + gamma * latent_alignment(student_h, teacher_h) [on group_id=1 only]

Supported backbones: PNN, FinalNet, DCNv3 (FCN).

Reference:
  Guo et al., "Hardness-aware Privileged Features Distillation with Latent Alignment
  for CVR Prediction", KDD 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import FeatureSeparator
from .backbone import build_backbone


class HAPFD(BaseModel):

    def __init__(self,
                 feature_map,
                 model_id="HAPFD",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 # KD parameters
                 kd_temperature=4.0,
                 kd_loss_weight=1.0,
                 base_loss_weight=1.0,
                 teacher_loss_weight=1.0,
                 kd_loss_type="kl",
                 # HA-PFD specific: focal KD
                 focal_gamma=2.0,
                 # HA-PFD specific: latent alignment
                 latent_loss_weight=1.0,
                 latent_loss_type="mse",  # "mse" or "cosine"
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

        super(HAPFD, self).__init__(
            feature_map, model_id=model_id, gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer, **kwargs)

        self.kd_temperature = kd_temperature
        self.kd_loss_weight = kd_loss_weight
        self.base_loss_weight = base_loss_weight
        self.teacher_loss_weight = teacher_loss_weight
        self.kd_loss_type = kd_loss_type
        self.focal_gamma = focal_gamma
        self.latent_loss_weight = latent_loss_weight
        self.latent_loss_type = latent_loss_type
        self.personalization_feature_list = personalization_feature_list or []
        self.personalization_field = personalization_field

        self.personalization_feature_list = [
            f for f in self.personalization_feature_list
            if f in self.feature_map.features
        ]
        self.feature_separator = FeatureSeparator(
            self.personalization_feature_list, self.feature_map
        )

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

        # Projection heads for latent alignment (identity if dims match)
        teacher_dim = self.teacher_backbone.latent_dim
        student_dim = self.student_backbone.latent_dim
        if teacher_dim != student_dim:
            self.student_projector = nn.Linear(student_dim, teacher_dim)
        else:
            self.student_projector = nn.Identity()

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        logging.info(f"HA-PFD initialized: backbone={backbone_type}, "
                     f"T={kd_temperature}, kd_weight={kd_loss_weight}, "
                     f"focal_gamma={focal_gamma}, latent_weight={latent_loss_weight}, "
                     f"latent_type={latent_loss_type}")

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

        _, student_X = self.feature_separator.separate_features(X, personalized_mask)

        if self.training:
            # Need latent features for alignment loss
            student_logit, student_latent = self.student_backbone.forward_with_latent(student_X)
            teacher_logit, teacher_latent = self.teacher_backbone.forward_with_latent(X)
        else:
            student_logit = self.student_backbone(student_X)
            teacher_logit = self.teacher_backbone(X)
            student_latent = teacher_latent = None

        student_y_pred = self.output_activation(student_logit)
        teacher_y_pred = self.output_activation(teacher_logit)

        # Route: group_id=1 -> teacher, group_id=2 -> student
        final_pred = torch.zeros_like(student_y_pred)
        if personalized_mask.any():
            final_pred[personalized_mask] = teacher_y_pred[personalized_mask]
        if non_personalized_mask.any():
            final_pred[non_personalized_mask] = student_y_pred[non_personalized_mask]

        return_dict = {
            "y_pred": final_pred,
            "student_y_pred": student_y_pred,
            "student_logit": student_logit,
            "personalized_mask": personalized_mask,
            "non_personalized_mask": non_personalized_mask,
        }
        if self.training:
            return_dict["teacher_y_pred"] = teacher_y_pred
            return_dict["teacher_logit"] = teacher_logit
            return_dict["student_latent"] = student_latent
            return_dict["teacher_latent"] = teacher_latent
        return return_dict

    def add_loss(self, return_dict, y_true):
        personalized_mask = return_dict["personalized_mask"]

        # Student BCE on all data
        student_base_loss = self.loss_fn(return_dict["student_y_pred"], y_true, reduction='mean')
        total_loss = self.base_loss_weight * student_base_loss

        if "teacher_y_pred" in return_dict and personalized_mask.any():
            # Teacher BCE on group_id=1
            teacher_base_loss = self.loss_fn(
                return_dict["teacher_y_pred"][personalized_mask],
                y_true[personalized_mask], reduction='mean')
            total_loss = total_loss + self.teacher_loss_weight * teacher_base_loss

            # Focal KD loss on group_id=1
            focal_kd_loss = self._compute_focal_kd_loss(
                return_dict["student_logit"][personalized_mask],
                return_dict["teacher_logit"][personalized_mask],
                y_true[personalized_mask])
            total_loss = total_loss + self.kd_loss_weight * focal_kd_loss

            # Latent alignment loss on group_id=1
            if (return_dict.get("student_latent") is not None and
                    return_dict.get("teacher_latent") is not None):
                latent_loss = self._compute_latent_loss(
                    return_dict["student_latent"][personalized_mask],
                    return_dict["teacher_latent"][personalized_mask])
                total_loss = total_loss + self.latent_loss_weight * latent_loss

        return total_loss

    def _compute_focal_kd_loss(self, student_logit, teacher_logit, y_true):
        """Focal-style KD loss: weight each instance by hardness.

        Hardness is measured as how poorly the student predicts — instances where
        the student's prediction is far from the true label get higher weight.

        focal_weight_i = (1 - p_correct_i)^gamma
        where p_correct_i = student_pred if y=1, else (1 - student_pred)
        """
        T = self.kd_temperature

        # Compute per-instance KD loss
        if self.kd_loss_type == "kl":
            s_prob = torch.sigmoid(student_logit / T).clamp(1e-7, 1 - 1e-7)
            t_prob = torch.sigmoid(teacher_logit / T).clamp(1e-7, 1 - 1e-7)
            per_instance_kd = (t_prob * torch.log(t_prob / s_prob) +
                               (1 - t_prob) * torch.log((1 - t_prob) / (1 - s_prob)))
            per_instance_kd = per_instance_kd * (T * T)
        elif self.kd_loss_type == "mse":
            per_instance_kd = (student_logit - teacher_logit) ** 2
        else:
            raise ValueError(f"Focal KD requires 'kl' or 'mse', got {self.kd_loss_type}")

        # Compute focal weights based on student's prediction hardness
        with torch.no_grad():
            student_pred = torch.sigmoid(student_logit).clamp(1e-7, 1 - 1e-7)
            # p_correct = p if y=1, (1-p) if y=0
            p_correct = student_pred * y_true + (1 - student_pred) * (1 - y_true)
            focal_weight = (1 - p_correct) ** self.focal_gamma

        # squeeze to handle shape mismatch ([B, 1] vs [B, 1])
        focal_weight = focal_weight.view_as(per_instance_kd)
        weighted_kd = focal_weight * per_instance_kd
        return weighted_kd.mean()

    def _compute_latent_loss(self, student_latent, teacher_latent):
        """Compute latent alignment loss between intermediate representations."""
        # Project student latent to teacher's space if needed
        student_proj = self.student_projector(student_latent)
        # Detach teacher to avoid teacher being pulled toward student's representation
        teacher_target = teacher_latent.detach()

        if self.latent_loss_type == "mse":
            return F.mse_loss(student_proj, teacher_target)
        elif self.latent_loss_type == "cosine":
            return 1 - F.cosine_similarity(student_proj, teacher_target, dim=-1).mean()
        else:
            raise ValueError(f"Unknown latent_loss_type: {self.latent_loss_type}")
