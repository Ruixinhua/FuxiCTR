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
DeepMutualLearning (DML): Online Mutual Distillation for Privacy-Preserving CTR/CVR

Trains two peer networks simultaneously, each teaching the other via KL-divergence
on their output distributions. Unlike traditional KD, there is no pre-trained teacher;
both networks learn collaboratively from scratch.

In the privacy-preserving setting:
  - Network 1 (full model): Uses ALL features (personalized + non-personalized)
  - Network 2 (NP model): Uses ONLY non-personalized features (personalized features masked)
  - Both networks are trained jointly with mutual distillation loss
  - At inference: group_id=1 → full model, group_id=2 → NP model

This differs from VanillaKD (two-stage, offline teacher) and PFD (online but
only teacher→student) in that distillation is bidirectional: the NP model's
predictions also influence the full model through KL-divergence.

Supported backbones: PNN, FinalNet, DCNv3 (FCN).

Reference:
  Zhang et al., "Deep Mutual Learning", CVPR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import FeatureSeparator
from ..backbone import build_backbone


class DeepMutualLearning(BaseModel):

    def __init__(self,
                 feature_map,
                 model_id="DeepMutualLearning",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 # DML-specific parameters
                 kd_temperature=4.0,
                 mutual_loss_weight=1.0,
                 base_loss_weight=1.0,
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

        super(DeepMutualLearning, self).__init__(
            feature_map, model_id=model_id, gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer, **kwargs)

        self.kd_temperature = kd_temperature
        self.mutual_loss_weight = mutual_loss_weight
        self.base_loss_weight = base_loss_weight
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

        # Two peer networks
        self.full_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)
        self.np_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        logging.info(f"DML initialized: backbone={backbone_type}, "
                     f"T={kd_temperature}, mutual_weight={mutual_loss_weight}")

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

        # NP model: mask personalized features
        _, np_X = self.feature_separator.separate_features(X, personalized_mask)
        np_logit = self.np_backbone(np_X)
        np_y_pred = self.output_activation(np_logit)

        # Full model: uses all features
        full_logit = self.full_backbone(X)
        full_y_pred = self.output_activation(full_logit)

        # Route: group_id=1 -> full model, group_id=2 -> NP model
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
        return return_dict

    def add_loss(self, return_dict, y_true):
        personalized_mask = return_dict["personalized_mask"]

        # BCE loss for both networks on all data
        full_bce = self.loss_fn(return_dict["full_y_pred"], y_true, reduction='mean')
        np_bce = self.loss_fn(return_dict["np_y_pred"], y_true, reduction='mean')

        total_loss = self.base_loss_weight * (full_bce + np_bce)

        # Mutual distillation: bidirectional KL on personalized data
        if personalized_mask.any():
            full_logit_p = return_dict["full_logit"][personalized_mask]
            np_logit_p = return_dict["np_logit"][personalized_mask]

            # Full → NP: NP learns from full model
            kd_full_to_np = self._compute_kl_loss(np_logit_p, full_logit_p.detach())
            # NP → Full: Full model also learns from NP model (mutual)
            kd_np_to_full = self._compute_kl_loss(full_logit_p, np_logit_p.detach())

            total_loss = total_loss + self.mutual_loss_weight * (kd_full_to_np + kd_np_to_full)

        return total_loss

    def _compute_kl_loss(self, student_logit, teacher_logit):
        """Symmetric KL divergence for binary classification with temperature scaling."""
        T = self.kd_temperature
        s_prob = torch.sigmoid(student_logit / T).clamp(1e-7, 1 - 1e-7)
        t_prob = torch.sigmoid(teacher_logit / T).clamp(1e-7, 1 - 1e-7)
        kl_loss = (t_prob * torch.log(t_prob / s_prob) +
                   (1 - t_prob) * torch.log((1 - t_prob) / (1 - s_prob)))
        return kl_loss.mean() * (T * T)
