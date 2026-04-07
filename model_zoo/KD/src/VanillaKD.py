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
  - Inference: group_id=1 → teacher prediction, group_id=2 → student prediction

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
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, InnerProductInteraction
from fuxictr.pytorch.torch_utils import FeatureSeparator, get_optimizer


class VanillaKD(BaseModel):
    """
    Vanilla Knowledge Distillation for CTR/CVR under personalized-feature constraints.

    Architecture:
      - Teacher: a full-feature backbone (PNN / FCN / FINAL) pre-trained with all features.
      - Student: the same backbone architecture but receiving only non-personalized features
                 (personalized features are masked to padding values).

    Training:
      Loss = α * BCE(student_pred, label) + β * KD_loss(student_logit, teacher_logit, T)
      where KD_loss = KL(σ(teacher_logit/T) || σ(student_logit/T)) * T²

    Inference:
      Only the student is used. The teacher is discarded.
    """

    def __init__(self,
                 feature_map,
                 model_id="VanillaKD",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 hidden_units=[400, 400, 400],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 product_type="inner",
                 # KD-specific parameters
                 teacher_model_path=None,
                 teacher_pretrain_epochs=100,  # auto-train teacher if no checkpoint
                 kd_temperature=4.0,
                 kd_loss_weight=1.0,
                 base_loss_weight=1.0,
                 kd_loss_type="kl",  # "kl", "mse", "cosine"
                 # Feature separation
                 personalization_feature_list=None,
                 personalization_field="is_personalization",
                 # Backbone type (teacher and student share the same architecture)
                 backbone_type="PNN",
                 backbone_params=None,
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
        self.backbone_type = backbone_type
        self.backbone_params = backbone_params or {}
        self._learning_rate = learning_rate
        self._optimizer_name = kwargs["optimizer"]
        self._loss_name = kwargs["loss"]

        # Teacher pretrain epochs: default to same as total epochs if not specified
        self.teacher_pretrain_epochs = teacher_pretrain_epochs

        # Filter personalization features to only those in feature_map
        self.personalization_feature_list = [
            f for f in self.personalization_feature_list
            if f in self.feature_map.features
        ]

        # Initialize feature separator for masking personalized features
        self.feature_separator = FeatureSeparator(
            self.personalization_feature_list, self.feature_map
        )

        # Build student network (uses all feature slots but personalized ones are masked)
        self._build_student(embedding_dim, hidden_units, hidden_activations,
                            net_dropout, batch_norm, product_type)

        # Build teacher network (same architecture, uses all features)
        self._build_teacher(embedding_dim, hidden_units, hidden_activations,
                            net_dropout, batch_norm, product_type)

        # Determine training mode
        self._teacher_pretrained = False
        if self.teacher_model_path and os.path.exists(self.teacher_model_path):
            self._load_and_freeze_teacher(self.teacher_model_path)
            self._teacher_pretrained = True

        # _training_phase: "teacher" during Phase 1, "student" during Phase 2
        self._training_phase = "student" if self._teacher_pretrained else "teacher"

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        # Re-freeze teacher after reset_parameters if loaded from checkpoint
        if self._teacher_pretrained:
            self._load_and_freeze_teacher(self.teacher_model_path)
        self.model_to_device()

        logging.info(f"VanillaKD initialized:")
        logging.info(f"  - Backbone: {backbone_type}")
        logging.info(f"  - Teacher model: {teacher_model_path}")
        logging.info(f"  - Teacher pretrained: {self._teacher_pretrained}")
        if not self._teacher_pretrained:
            logging.info(f"  - Teacher will be auto-trained in Phase 1 "
                         f"(pretrain_epochs={teacher_pretrain_epochs})")
        logging.info(f"  - KD temperature: {kd_temperature}")
        logging.info(f"  - KD loss weight: {kd_loss_weight}")
        logging.info(f"  - KD loss type: {kd_loss_type}")
        logging.info(f"  - Personalization features masked: {self.personalization_feature_list}")

    def _build_pnn(self, prefix, embedding_dim, hidden_units, hidden_activations,
                   net_dropout, batch_norm, product_type):
        """Build a PNN backbone, returning (embedding_layer, inner_product_layer, dnn)."""
        embedding_layer = FeatureEmbedding(self.feature_map, embedding_dim)
        if product_type != "inner":
            raise NotImplementedError(f"product_type={product_type} not implemented.")
        inner_product_layer = InnerProductInteraction(self.num_fields, output="inner_product")
        input_dim = int(self.num_fields * (self.num_fields - 1) / 2) + self.num_fields * embedding_dim
        dnn = MLP_Block(input_dim=input_dim,
                        output_dim=1,
                        hidden_units=hidden_units,
                        hidden_activations=hidden_activations,
                        output_activation=None,
                        dropout_rates=net_dropout,
                        batch_norm=batch_norm)
        return embedding_layer, inner_product_layer, dnn

    def _build_student(self, embedding_dim, hidden_units, hidden_activations,
                       net_dropout, batch_norm, product_type):
        """Build the student network."""
        self.student_embedding, self.student_interaction, self.student_dnn = \
            self._build_pnn("student", embedding_dim, hidden_units, hidden_activations,
                            net_dropout, batch_norm, product_type)

    def _build_teacher(self, embedding_dim, hidden_units, hidden_activations,
                       net_dropout, batch_norm, product_type):
        """Build the teacher network (same architecture as student)."""
        self.teacher_embedding, self.teacher_interaction, self.teacher_dnn = \
            self._build_pnn("teacher", embedding_dim, hidden_units, hidden_activations,
                            net_dropout, batch_norm, product_type)

    def _load_and_freeze_teacher(self, model_path):
        """Load pre-trained teacher weights and freeze all teacher parameters."""
        logging.info(f"Loading teacher model from: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")

        # Map teacher checkpoint keys to our teacher sub-modules
        teacher_state = {}
        for key, value in checkpoint.items():
            if key.startswith("embedding_layer."):
                new_key = key.replace("embedding_layer.", "teacher_embedding.", 1)
                teacher_state[new_key] = value
            elif key.startswith("inner_product_layer."):
                new_key = key.replace("inner_product_layer.", "teacher_interaction.", 1)
                teacher_state[new_key] = value
            elif key.startswith("dnn."):
                new_key = key.replace("dnn.", "teacher_dnn.", 1)
                teacher_state[new_key] = value

        # Load with strict=False to allow partial loading
        missing, unexpected = self.load_state_dict(teacher_state, strict=False)
        loaded_count = len(teacher_state) - len(unexpected)
        logging.info(f"Teacher weights loaded: {loaded_count} params loaded, "
                     f"{len(missing)} missing, {len(unexpected)} unexpected")

        # Freeze teacher parameters
        for name, param in self.named_parameters():
            if name.startswith("teacher_"):
                param.requires_grad = False

        teacher_params = sum(1 for n, _ in self.named_parameters() if n.startswith("teacher_"))
        logging.info(f"Teacher frozen: {teacher_params} parameter groups")

    def _forward_backbone(self, embedding_layer, interaction_layer, dnn, X):
        """Forward pass through a PNN backbone, returning (y_pred, logit)."""
        feature_emb = embedding_layer(X)
        inner_products = interaction_layer(feature_emb)
        dense_input = torch.cat([feature_emb.flatten(start_dim=1), inner_products], dim=1)
        logit = dnn(dense_input)
        y_pred = self.output_activation(logit)
        return y_pred, logit

    def _get_personalized_mask(self, X):
        """Get personalized/non-personalized masks from is_personalization field.
        
        Returns:
            (personalized_mask, non_personalized_mask): bool tensors [batch_size]
            personalized_mask=True for group_id=1 rows (full features available)
        """
        if self.personalization_field in X:
            flag = X[self.personalization_field]
            personalized_mask = (flag == 1)
            non_personalized_mask = (flag != 1)
        else:
            batch_size = list(X.values())[0].size(0)
            device = list(X.values())[0].device
            personalized_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            non_personalized_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            logging.warning(f"'{self.personalization_field}' not found in inputs, "
                            f"treating all as non-personalized")
        return personalized_mask, non_personalized_mask

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        personalized_mask, non_personalized_mask = self._get_personalized_mask(X)

        if self._training_phase == "teacher":
            # Phase 1: Teacher training — use all features, standard PNN forward
            teacher_y_pred, teacher_logit = self._forward_backbone(
                self.teacher_embedding, self.teacher_interaction, self.teacher_dnn, X
            )
            return {
                "y_pred": teacher_y_pred,
                "logit": teacher_logit,
                "personalized_mask": personalized_mask,
                "non_personalized_mask": non_personalized_mask,
            }

        # Phase 2 / inference: both teacher and student forward
        # Student: mask personalization features for group_id=1 rows
        _, student_X = self.feature_separator.separate_features(X, personalized_mask)
        student_y_pred, student_logit = self._forward_backbone(
            self.student_embedding, self.student_interaction, self.student_dnn, student_X
        )

        # Teacher forward (frozen, all features)
        with torch.no_grad():
            teacher_y_pred, teacher_logit = self._forward_backbone(
                self.teacher_embedding, self.teacher_interaction, self.teacher_dnn, X
            )

        # Route predictions: group_id=1 → teacher, group_id=2 → student
        final_pred = torch.zeros_like(student_y_pred)
        if personalized_mask.any():
            final_pred[personalized_mask] = teacher_y_pred[personalized_mask]
        if non_personalized_mask.any():
            final_pred[non_personalized_mask] = student_y_pred[non_personalized_mask]

        return_dict = {
            "y_pred": final_pred,
            "student_y_pred": student_y_pred,
            "student_logit": student_logit,
            "teacher_y_pred": teacher_y_pred.detach(),
            "teacher_logit": teacher_logit.detach(),
            "personalized_mask": personalized_mask,
            "non_personalized_mask": non_personalized_mask,
        }

        return return_dict

    def add_loss(self, return_dict, y_true):
        personalized_mask = return_dict["personalized_mask"]

        if self._training_phase == "teacher":
            # Phase 1: standard BCE loss for teacher on all data
            return self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')

        # Phase 2: student BCE on ALL data + KD loss on group_id=1 only
        student_y_pred = return_dict["student_y_pred"]
        base_loss = self.loss_fn(student_y_pred, y_true, reduction='mean')
        total_loss = self.base_loss_weight * base_loss

        # KD loss only on personalized rows (where teacher has privileged features)
        if "teacher_logit" in return_dict and personalized_mask.any():
            student_logit = return_dict["student_logit"][personalized_mask]
            teacher_logit = return_dict["teacher_logit"][personalized_mask]
            kd_loss = self._compute_kd_loss(student_logit, teacher_logit)
            total_loss = total_loss + self.kd_loss_weight * kd_loss

            logging.debug(f"VanillaKD loss: base={base_loss.item():.6f}, "
                          f"kd={kd_loss.item():.6f}, "
                          f"kd_samples={personalized_mask.sum().item()}")

        return total_loss

    def _freeze_teacher(self):
        """Freeze all teacher parameters."""
        for name, param in self.named_parameters():
            if name.startswith("teacher_"):
                param.requires_grad = False
        teacher_params = sum(1 for n, _ in self.named_parameters() if n.startswith("teacher_"))
        logging.info(f"Teacher frozen: {teacher_params} parameter groups")

    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        """Two-phase training: auto-train teacher if no checkpoint, then train student."""

        if self._teacher_pretrained:
            # Teacher already loaded from checkpoint — go directly to Phase 2
            logging.info("=== Teacher loaded from checkpoint, skipping Phase 1 ===")
            self._training_phase = "student"
            super().fit(data_generator, epochs=epochs,
                        validation_data=validation_data,
                        max_gradient_norm=max_gradient_norm, **kwargs)
            return

        # ============================================================
        # Phase 1: Train teacher with all features
        # ============================================================
        teacher_epochs = self.teacher_pretrain_epochs or epochs
        logging.info("=" * 60)
        logging.info(f"=== Phase 1: Training teacher for {teacher_epochs} epochs ===")
        logging.info("=" * 60)

        self._training_phase = "teacher"
        # Only optimize teacher parameters
        self.optimizer = get_optimizer(self._optimizer_name,
                                      [p for n, p in self.named_parameters()
                                       if n.startswith("teacher_")],
                                      self._learning_rate)

        # Save teacher checkpoint separately
        teacher_checkpoint = self.checkpoint + ".teacher"
        orig_checkpoint = self.checkpoint
        self.checkpoint = teacher_checkpoint

        super().fit(data_generator, epochs=teacher_epochs,
                    validation_data=validation_data,
                    max_gradient_norm=max_gradient_norm, **kwargs)

        # Restore checkpoint path
        self.checkpoint = orig_checkpoint

        # Load best teacher weights and freeze
        if os.path.exists(teacher_checkpoint):
            logging.info(f"Loading best teacher weights from: {teacher_checkpoint}")
            self.load_weights(teacher_checkpoint)
            os.remove(teacher_checkpoint)
        self._freeze_teacher()
        self._teacher_pretrained = True

        # ============================================================
        # Phase 2: Train student with KD loss
        # ============================================================
        logging.info("=" * 60)
        logging.info(f"=== Phase 2: Training student for {epochs} epochs with KD ===")
        logging.info("=" * 60)

        self._training_phase = "student"
        # Re-create optimizer for student parameters only
        self.optimizer = get_optimizer(self._optimizer_name,
                                      [p for n, p in self.named_parameters()
                                       if p.requires_grad],
                                      self._learning_rate)

        # Reset early stopping state for Phase 2
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._stop_training = False

        super().fit(data_generator, epochs=epochs,
                    validation_data=validation_data,
                    max_gradient_norm=max_gradient_norm, **kwargs)

    def _compute_kd_loss(self, student_logit, teacher_logit):
        """Compute knowledge distillation loss between student and teacher logits."""
        T = self.kd_temperature

        if self.kd_loss_type == "kl":
            # KL-divergence with temperature scaling (Hinton et al., 2015)
            # For binary classification: use sigmoid to get probabilities
            student_prob = torch.sigmoid(student_logit / T)
            teacher_prob = torch.sigmoid(teacher_logit / T)

            # Binary KL divergence: KL(teacher || student)
            # = teacher * log(teacher/student) + (1-teacher) * log((1-teacher)/(1-student))
            eps = 1e-7
            teacher_prob = teacher_prob.clamp(eps, 1 - eps)
            student_prob = student_prob.clamp(eps, 1 - eps)

            kd_loss = teacher_prob * torch.log(teacher_prob / student_prob) + \
                      (1 - teacher_prob) * torch.log((1 - teacher_prob) / (1 - student_prob))
            kd_loss = kd_loss.mean() * (T * T)

        elif self.kd_loss_type == "mse":
            # MSE on logits
            kd_loss = F.mse_loss(student_logit, teacher_logit)

        elif self.kd_loss_type == "cosine":
            # Cosine similarity loss
            kd_loss = 1 - F.cosine_similarity(student_logit, teacher_logit, dim=-1).mean()

        else:
            raise ValueError(f"Unknown kd_loss_type: {self.kd_loss_type}")

        return kd_loss
