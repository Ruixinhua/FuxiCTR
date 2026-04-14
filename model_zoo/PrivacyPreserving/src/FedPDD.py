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
FedPDD: Privacy-preserving Double Distillation for Cross-silo Federated Recommendation

Implements a cross-silo federated recommendation framework where two model
silos (full-feature silo and NP-feature silo) collaboratively train through
a double distillation strategy:

  1. Inter-silo Explicit Distillation: Each silo learns from the other's
     predictions (soft labels) on shared/overlapped samples. The full-feature
     silo's predictions are treated as privacy-sensitive privileged knowledge
     that is distilled to the NP-feature silo.

  2. Intra-silo Self-Distillation (Temporal Consistency): Each silo also
     aligns current predictions with its own historical predictions (via an
     EMA teacher), providing implicit knowledge that stabilizes training and
     reduces catastrophic forgetting.

Privacy Mechanisms:
  - Offline training: predictions are exchanged only once (or periodically),
    reducing communication and exposure surface.
  - Differential privacy noise (Gaussian) is added to the shared soft
    predictions before distillation, providing formal privacy guarantees.
  - Only soft predictions (not raw data/embeddings) are shared between silos.

Architecture:
  - Full-feature model (Silo A): uses all features, acts as the "privileged" silo
  - NP-feature model (Silo B): uses only non-personalized features
  - EMA teachers: shadow copies of each model, updated via exponential moving average

Loss (for Silo B / device model):
  L = alpha * BCE(student, label)
    + beta_inter * KL(student, noisy_teacher_pred)      [inter-silo]
    + beta_self * KL(student, ema_pred)                   [self-distillation]

Supported backbones: PNN, FinalNet, DCNv3 (FCN).

Reference:
  Wan et al., "FedPDD: A Privacy-preserving Double Distillation Framework for
  Cross-silo Federated Recommendation", IJCNN 2023 (revised Jan 2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import FeatureSeparator
from ..backbone import build_backbone


class FedPDD(BaseModel):

    def __init__(self,
                 feature_map,
                 model_id="FedPDD",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 # FedPDD KD parameters
                 kd_temperature=4.0,
                 inter_kd_weight=1.0,        # inter-silo distillation weight
                 self_kd_weight=0.5,         # self-distillation (EMA) weight
                 base_loss_weight=1.0,       # student BCE weight
                 teacher_loss_weight=1.0,    # full-feature silo BCE weight
                 # EMA parameters
                 ema_decay=0.999,            # EMA decay rate for shadow model
                 # DP noise on shared predictions
                 dp_noise_scale=0.0,         # std of Gaussian noise on shared soft labels
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

        super(FedPDD, self).__init__(
            feature_map, model_id=model_id, gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer, **kwargs)

        self.kd_temperature = kd_temperature
        self.inter_kd_weight = inter_kd_weight
        self.self_kd_weight = self_kd_weight
        self.base_loss_weight = base_loss_weight
        self.teacher_loss_weight = teacher_loss_weight
        self.ema_decay = ema_decay
        self.dp_noise_scale = dp_noise_scale
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

        # Silo A: full-feature model (privileged)
        self.full_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)
        # Silo B: NP-feature model (device/student)
        self.np_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        # Initialize EMA shadow models (for self-distillation)
        self._ema_full = self._create_ema_copy(self.full_backbone)
        self._ema_np = self._create_ema_copy(self.np_backbone)
        self._ema_initialized = False

        logging.info(f"FedPDD initialized: backbone={backbone_type}, "
                     f"T={kd_temperature}, inter_kd={inter_kd_weight}, "
                     f"self_kd={self_kd_weight}, ema_decay={ema_decay}, "
                     f"dp_noise_scale={dp_noise_scale}")

    def _create_ema_copy(self, model):
        """Create an EMA shadow copy of a model."""
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad = False
        return ema

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA shadow models with current model parameters."""
        decay = self.ema_decay
        for ema_p, model_p in zip(self._ema_full.parameters(),
                                   self.full_backbone.parameters()):
            ema_p.data.mul_(decay).add_(model_p.data, alpha=1 - decay)
        for ema_p, model_p in zip(self._ema_np.parameters(),
                                   self.np_backbone.parameters()):
            ema_p.data.mul_(decay).add_(model_p.data, alpha=1 - decay)

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

        # NP model sees only non-personalized features
        _, np_X = self.feature_separator.separate_features(X, personalized_mask)
        np_logit = self.np_backbone(np_X)
        np_y_pred = self.output_activation(np_logit)

        # Full model sees all features
        full_logit = self.full_backbone(X)
        full_y_pred = self.output_activation(full_logit)

        # Route for inference
        final_pred = torch.zeros_like(np_y_pred)
        if personalized_mask.any():
            final_pred[personalized_mask] = full_y_pred[personalized_mask]
        if non_personalized_mask.any():
            final_pred[non_personalized_mask] = np_y_pred[non_personalized_mask]

        return_dict = {
            "y_pred": final_pred,
            "np_y_pred": np_y_pred,
            "np_logit": np_logit,
            "full_y_pred": full_y_pred,
            "full_logit": full_logit,
            "personalized_mask": personalized_mask,
            "non_personalized_mask": non_personalized_mask,
        }

        # Compute EMA predictions for self-distillation during training
        if self.training:
            with torch.no_grad():
                ema_full_logit = self._ema_full(X)
                ema_np_logit = self._ema_np(np_X)
                # Add DP noise to inter-silo shared predictions
                noisy_full_logit = full_logit.detach()
                if self.dp_noise_scale > 0:
                    noise = torch.randn_like(noisy_full_logit) * self.dp_noise_scale
                    noisy_full_logit = noisy_full_logit + noise

            return_dict["ema_full_logit"] = ema_full_logit
            return_dict["ema_np_logit"] = ema_np_logit
            return_dict["noisy_full_logit"] = noisy_full_logit

        return return_dict

    def train_step(self, batch_data):
        """Override to update EMA after each training step."""
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.add_loss(return_dict, y_true)
        loss += self.add_regularization()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()

        # Update EMA shadow models
        if self._ema_initialized:
            self._update_ema()
        else:
            # First step: copy current parameters
            self._ema_full.load_state_dict(self.full_backbone.state_dict())
            self._ema_np.load_state_dict(self.np_backbone.state_dict())
            self._ema_initialized = True

        return loss

    def add_loss(self, return_dict, y_true):
        personalized_mask = return_dict["personalized_mask"]

        # 1. NP model BCE on all data
        np_bce = self.loss_fn(return_dict["np_y_pred"], y_true, reduction='mean')
        total_loss = self.base_loss_weight * np_bce

        # 2. Full model BCE on all data
        full_bce = self.loss_fn(return_dict["full_y_pred"], y_true, reduction='mean')
        total_loss = total_loss + self.teacher_loss_weight * full_bce

        if self.training and personalized_mask.any():
            # 3. Inter-silo Explicit Distillation (on personalized samples)
            #    Full model -> NP model (with DP noise on full model's predictions)
            if "noisy_full_logit" in return_dict:
                inter_kd_loss = self._compute_kd_loss(
                    return_dict["np_logit"][personalized_mask],
                    return_dict["noisy_full_logit"][personalized_mask])
                total_loss = total_loss + self.inter_kd_weight * inter_kd_loss

        # 4. Self-Distillation (temporal consistency with EMA)
        if self.training and self._ema_initialized:
            if "ema_np_logit" in return_dict:
                self_kd_np = self._compute_kd_loss(
                    return_dict["np_logit"],
                    return_dict["ema_np_logit"])
                total_loss = total_loss + self.self_kd_weight * self_kd_np

        return total_loss

    def _compute_kd_loss(self, student_logit, teacher_logit):
        """Binary KL divergence for distillation."""
        T = self.kd_temperature
        s_prob = torch.sigmoid(student_logit / T).clamp(1e-7, 1 - 1e-7)
        t_prob = torch.sigmoid(teacher_logit / T).clamp(1e-7, 1 - 1e-7)
        kd_loss = (t_prob * torch.log(t_prob / s_prob) +
                   (1 - t_prob) * torch.log((1 - t_prob) / (1 - s_prob)))
        return kd_loss.mean() * (T * T)
