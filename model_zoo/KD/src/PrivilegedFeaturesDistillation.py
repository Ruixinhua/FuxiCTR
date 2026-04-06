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
PrivilegedFeaturesDistillation (PFD): Online Knowledge Distillation with Privileged Features

Single-stage joint training pipeline:
  - Teacher: uses ALL features (personalized + non-personalized), trained end-to-end
  - Student: uses ONLY non-personalized features (personalized features masked to padding)
  - Both networks are trained simultaneously. The teacher's soft predictions serve as
    an online distillation signal for the student.

This is inspired by the Privileged Features Distillation approach used in industrial
advertising systems, where certain features (e.g., user identity) are available during
training but not at inference time for privacy-constrained traffic.

Key differences from VanillaKD:
  - No pre-trained teacher required; teacher and student are trained jointly.
  - The teacher continues to improve during training, providing increasingly
    informative soft targets.
  - Shared embedding layer between teacher and student for efficiency.

Reference:
  Xu et al., "Privileged Features Distillation at Taobao Recommendations", KDD 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, InnerProductInteraction
from fuxictr.pytorch.torch_utils import FeatureSeparator


class PrivilegedFeaturesDistillation(BaseModel):
    """
    Privileged Features Distillation for CTR/CVR under personalized-feature constraints.

    Architecture:
      - Shared embedding layer for teacher and student.
      - Teacher network: full PNN backbone using ALL features.
      - Student network: PNN backbone using only non-personalized features (masked).

    Training (single stage, joint):
      Loss = α * BCE(teacher_pred, label)
           + α * BCE(student_pred, label)
           + β * KD_loss(student_logit, teacher_logit, T)

    Inference:
      Only the student network is used.
    """

    def __init__(self,
                 feature_map,
                 model_id="PrivilegedFeaturesDistillation",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 hidden_units=[400, 400, 400],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 product_type="inner",
                 # PFD-specific parameters
                 kd_temperature=4.0,
                 kd_loss_weight=1.0,
                 base_loss_weight=1.0,
                 teacher_loss_weight=1.0,
                 kd_loss_type="kl",  # "kl", "mse", "cosine"
                 share_embedding=True,
                 # Teacher-specific architecture (optional overrides)
                 teacher_hidden_units=None,
                 teacher_net_dropout=None,
                 # Feature separation
                 personalization_feature_list=None,
                 personalization_field="is_personalization",
                 # Standard params
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):

        super(PrivilegedFeaturesDistillation, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs
        )

        self.kd_temperature = kd_temperature
        self.kd_loss_weight = kd_loss_weight
        self.base_loss_weight = base_loss_weight
        self.teacher_loss_weight = teacher_loss_weight
        self.kd_loss_type = kd_loss_type
        self.share_embedding = share_embedding
        self.personalization_feature_list = personalization_feature_list or []
        self.personalization_field = personalization_field

        # Use teacher overrides if specified, otherwise match student
        teacher_hidden_units = teacher_hidden_units or hidden_units
        teacher_net_dropout = teacher_net_dropout if teacher_net_dropout is not None else net_dropout

        # Filter personalization features to those in feature_map
        self.personalization_feature_list = [
            f for f in self.personalization_feature_list
            if f in self.feature_map.features
        ]

        # Initialize feature separator
        self.feature_separator = FeatureSeparator(
            self.personalization_feature_list, self.feature_map
        )

        # === Shared embedding layer ===
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        if product_type != "inner":
            raise NotImplementedError(f"product_type={product_type} not implemented.")

        # === Student network (NP features only) ===
        self.student_interaction = InnerProductInteraction(self.num_fields, output="inner_product")
        student_input_dim = (int(self.num_fields * (self.num_fields - 1) / 2)
                             + self.num_fields * embedding_dim)
        self.student_dnn = MLP_Block(
            input_dim=student_input_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activations=hidden_activations,
            output_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm
        )

        # === Teacher network (all features) ===
        if share_embedding:
            # Reuse the same embedding layer
            self.teacher_embedding = self.embedding_layer
        else:
            self.teacher_embedding = FeatureEmbedding(feature_map, embedding_dim)

        self.teacher_interaction = InnerProductInteraction(self.num_fields, output="inner_product")
        teacher_input_dim = (int(self.num_fields * (self.num_fields - 1) / 2)
                             + self.num_fields * embedding_dim)
        self.teacher_dnn = MLP_Block(
            input_dim=teacher_input_dim,
            output_dim=1,
            hidden_units=teacher_hidden_units,
            hidden_activations=hidden_activations,
            output_activation=None,
            dropout_rates=teacher_net_dropout,
            batch_norm=batch_norm
        )

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        logging.info(f"PrivilegedFeaturesDistillation initialized:")
        logging.info(f"  - Share embedding: {share_embedding}")
        logging.info(f"  - KD temperature: {kd_temperature}")
        logging.info(f"  - KD loss weight: {kd_loss_weight}")
        logging.info(f"  - Teacher loss weight: {teacher_loss_weight}")
        logging.info(f"  - KD loss type: {kd_loss_type}")
        logging.info(f"  - Personalization features masked: {self.personalization_feature_list}")
        logging.info(f"  - Student hidden_units: {hidden_units}")
        logging.info(f"  - Teacher hidden_units: {teacher_hidden_units}")

    def _forward_teacher(self, X):
        """Teacher forward: uses all features."""
        feature_emb = self.teacher_embedding(X)
        inner_products = self.teacher_interaction(feature_emb)
        dense_input = torch.cat([feature_emb.flatten(start_dim=1), inner_products], dim=1)
        logit = self.teacher_dnn(dense_input)
        y_pred = self.output_activation(logit)
        return y_pred, logit

    def _forward_student(self, X):
        """Student forward: personalized features are masked."""
        feature_emb = self.embedding_layer(X)
        inner_products = self.student_interaction(feature_emb)
        dense_input = torch.cat([feature_emb.flatten(start_dim=1), inner_products], dim=1)
        logit = self.student_dnn(dense_input)
        y_pred = self.output_activation(logit)
        return y_pred, logit

    def forward(self, inputs):
        X = self.get_inputs(inputs)

        # Mask personalized features for the student
        batch_size = list(X.values())[0].size(0)
        all_personalized_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        _, student_X = self.feature_separator.separate_features(X, all_personalized_mask)

        # Student forward (masked features)
        student_y_pred, student_logit = self._forward_student(student_X)

        return_dict = {
            "y_pred": student_y_pred,  # student prediction is the primary output
            "student_logit": student_logit,
        }

        # Teacher forward (all features) — trained jointly
        if self.training:
            teacher_y_pred, teacher_logit = self._forward_teacher(X)
            return_dict["teacher_y_pred"] = teacher_y_pred
            return_dict["teacher_logit"] = teacher_logit
        # At inference, only student is used (y_pred already set)

        return return_dict

    def add_loss(self, return_dict, y_true):
        """Compute combined loss: teacher BCE + student BCE + KD loss."""
        student_y_pred = return_dict["y_pred"]

        # Student base loss
        student_base_loss = self.loss_fn(student_y_pred, y_true, reduction='mean')
        total_loss = self.base_loss_weight * student_base_loss

        if "teacher_y_pred" in return_dict:
            teacher_y_pred = return_dict["teacher_y_pred"]
            student_logit = return_dict["student_logit"]
            teacher_logit = return_dict["teacher_logit"]

            # Teacher base loss (teacher also learns from labels)
            teacher_base_loss = self.loss_fn(teacher_y_pred, y_true, reduction='mean')
            total_loss = total_loss + self.teacher_loss_weight * teacher_base_loss

            # KD loss (student learns from teacher's soft predictions)
            kd_loss = self._compute_kd_loss(student_logit, teacher_logit)
            total_loss = total_loss + self.kd_loss_weight * kd_loss

            logging.debug(
                f"PFD loss: student_bce={student_base_loss.item():.6f}, "
                f"teacher_bce={teacher_base_loss.item():.6f}, "
                f"kd={kd_loss.item():.6f}"
            )

        return total_loss

    def _compute_kd_loss(self, student_logit, teacher_logit):
        """Compute knowledge distillation loss."""
        T = self.kd_temperature

        if self.kd_loss_type == "kl":
            # KL-divergence with temperature (Hinton et al., 2015)
            # Binary classification: sigmoid-based KL
            student_prob = torch.sigmoid(student_logit / T)
            teacher_prob = torch.sigmoid(teacher_logit / T)

            eps = 1e-7
            teacher_prob = teacher_prob.clamp(eps, 1 - eps)
            student_prob = student_prob.clamp(eps, 1 - eps)

            kd_loss = teacher_prob * torch.log(teacher_prob / student_prob) + \
                      (1 - teacher_prob) * torch.log((1 - teacher_prob) / (1 - student_prob))
            # Note: teacher_logit is NOT detached here — the teacher also receives
            # gradient from KD loss, which is the key difference from VanillaKD.
            # The teacher is encouraged to produce more informative soft targets.
            kd_loss = kd_loss.mean() * (T * T)

        elif self.kd_loss_type == "mse":
            kd_loss = F.mse_loss(student_logit, teacher_logit)

        elif self.kd_loss_type == "cosine":
            kd_loss = 1 - F.cosine_similarity(student_logit, teacher_logit, dim=-1).mean()

        else:
            raise ValueError(f"Unknown kd_loss_type: {self.kd_loss_type}")

        return kd_loss
