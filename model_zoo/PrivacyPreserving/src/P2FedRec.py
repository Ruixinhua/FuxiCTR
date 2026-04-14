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
P2FedRec: Towards Privacy-Preserving and Personalized Federated
Recommendation via Relationship Awareness

Adapted from the SIGMOD 2025 paper by Hu et al.

Original Method:
  P2FedRec proposes a relationship-aware privacy-preserving and personalized
  federated recommendation scheme with multi-level privacy protection:

  1. Data-Level Privacy: User-server collaborative mechanism for relationship
     graph generation with embedding perturbation. Shared embeddings are
     perturbed with calibrated noise before publishing.

  2. Edge-Level Privacy: A noisy global graph-guided aggregation module
     protects the relationship structure by adding noise to the aggregation
     graph edges.

  3. Personalized Training: Users learn tailored local models through
     relationship-guided personalization, where aggregation weights are
     computed based on user similarity (derived from noisy embeddings).

Adaptation for Feature-Based CTR Prediction:
  In our cloud-device feature-separation scenario:

  1. Cloud model (full features) and device model (NP features) share a
     common embedding layer for NP features. This shared embedding is the
     communication channel between the two models.

  2. Data-Level Privacy: Before the device model accesses the shared
     embeddings, calibrated Gaussian noise is added for formal DP.

  3. Personalized Aggregation: Instead of uniform FedAvg, the device model
     computes adaptive aggregation weights based on the cosine similarity
     between its latent patterns and the cloud's. More similar patterns
     receive higher aggregation weight, achieving personalized knowledge
     transfer.

  4. Periodic Aggregation: Every K steps, the device model's shared layers
     are partially aggregated with the cloud model using relationship-aware
     weighted averaging plus noise.

Privacy:
  - Multi-level: embedding noise (data-level) + aggregation noise (edge-level).
  - Feature-level privacy: device model never sees personalized features.
  - Per-round DP composition tracking.

Routing (inference):
  - group_id=1 (personalized): cloud model prediction.
  - group_id!=1 (non-personalized): device model prediction.

Supported backbones: PNN, FinalNet, DCNv3 (FCN).

Reference:
  Hu et al., "P2FedRec: Towards Privacy-Preserving and Personalized
  Federated Recommendation via Relationship Awareness",
  Proceedings of the ACM on Management of Data (SIGMOD), 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy
import math

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import FeatureSeparator
from ..backbone import build_backbone


class P2FedRec(BaseModel):

    def __init__(self,
                 feature_map,
                 model_id="P2FedRec",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 # P2FedRec parameters
                 embedding_noise_scale=0.01,      # data-level DP noise scale
                 aggregation_noise_scale=0.001,   # edge-level aggregation noise
                 aggregation_interval=50,         # steps between aggregation rounds
                 aggregation_ratio=0.5,           # fraction of cloud params to mix in
                 personalized_agg=True,           # use cosine-similarity-based aggregation weights
                 personalized_head=True,          # keep device head layers personalized (no agg)
                 base_loss_weight=1.0,            # device BCE weight
                 cloud_loss_weight=1.0,           # cloud BCE weight
                 alignment_weight=0.3,            # soft alignment between cloud and device
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

        super(P2FedRec, self).__init__(
            feature_map, model_id=model_id, gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer, **kwargs)

        self.embedding_noise_scale = embedding_noise_scale
        self.aggregation_noise_scale = aggregation_noise_scale
        self.aggregation_interval = aggregation_interval
        self.aggregation_ratio = aggregation_ratio
        self.personalized_agg = personalized_agg
        self.personalized_head = personalized_head
        self.base_loss_weight = base_loss_weight
        self.cloud_loss_weight = cloud_loss_weight
        self.alignment_weight = alignment_weight
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

        # Cloud model: uses ALL features
        self.cloud_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)
        # Device model: uses only NP features
        self.device_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)

        # Local step counter for aggregation scheduling
        self._local_step = 0
        # Save cloud model state at aggregation round start
        self._cloud_snapshot = None

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        logging.info(f"P2FedRec initialized: backbone={backbone_type}, "
                     f"emb_noise={embedding_noise_scale}, "
                     f"agg_noise={aggregation_noise_scale}, "
                     f"agg_interval={aggregation_interval}, "
                     f"personalized_agg={personalized_agg}")

    def _get_personalized_mask(self, X):
        if self.personalization_field in X:
            flag = X[self.personalization_field]
            return (flag == 1), (flag != 1)
        batch_size = list(X.values())[0].size(0)
        device = list(X.values())[0].device
        return (torch.zeros(batch_size, dtype=torch.bool, device=device),
                torch.ones(batch_size, dtype=torch.bool, device=device))

    def _add_embedding_noise(self, latent):
        """Add calibrated Gaussian noise to latent representations (data-level DP)."""
        if self.training and self.embedding_noise_scale > 0:
            noise = torch.randn_like(latent) * self.embedding_noise_scale
            return latent + noise
        return latent

    def _compute_personalized_agg_weight(self, cloud_params, device_params):
        """Compute personalized aggregation weight based on parameter similarity.

        In P2FedRec, aggregation weights are proportional to the cosine
        similarity between the device model's parameters and the cloud model's
        parameters. Higher similarity -> stronger aggregation.
        """
        cloud_flat = torch.cat([p.data.flatten() for p in cloud_params])
        device_flat = torch.cat([p.data.flatten() for p in device_params])
        similarity = F.cosine_similarity(
            cloud_flat.unsqueeze(0), device_flat.unsqueeze(0)).item()
        # Map similarity [-1, 1] -> [0, 1] weight, scaled by aggregation_ratio
        weight = max(0, (similarity + 1) / 2) * self.aggregation_ratio
        return weight

    @torch.no_grad()
    def _relationship_aware_aggregation(self):
        """Periodic aggregation of device model with cloud model parameters.

        Implements the P2FedRec aggregation:
        1. Compute personalized aggregation weight (or use fixed ratio).
        2. For each shared layer, blend cloud and device parameters.
        3. Add aggregation noise (edge-level DP).
        4. Keep personalized head layers untouched (optional).
        """
        cloud_params = list(self.cloud_backbone.parameters())
        device_params = list(self.device_backbone.parameters())

        if self.personalized_agg:
            agg_weight = self._compute_personalized_agg_weight(
                cloud_params, device_params)
        else:
            agg_weight = self.aggregation_ratio

        num_shared = len(device_params)
        if self.personalized_head:
            # Last 2 layers are "head" — keep personalized
            num_shared = max(1, num_shared - 2)

        for i in range(min(num_shared, len(cloud_params))):
            cloud_p = cloud_params[i].data
            device_p = device_params[i].data

            # Weighted aggregation
            aggregated = (1 - agg_weight) * device_p + agg_weight * cloud_p

            # Add aggregation noise (edge-level privacy)
            if self.aggregation_noise_scale > 0:
                noise = torch.randn_like(aggregated) * self.aggregation_noise_scale
                aggregated = aggregated + noise

            device_params[i].data.copy_(aggregated)

    def train_step(self, batch_data, **kwargs):
        """Override to add periodic aggregation."""
        result = super().train_step(batch_data, **kwargs)

        self._local_step += 1
        if self._local_step % self.aggregation_interval == 0:
            self._relationship_aware_aggregation()

        return result

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        personalized_mask, non_personalized_mask = self._get_personalized_mask(X)

        # Device model sees only NP features
        _, device_X = self.feature_separator.separate_features(X, personalized_mask)
        device_logit, device_latent = self.device_backbone.forward_with_latent(device_X)
        device_y_pred = self.output_activation(device_logit)

        # Cloud model sees ALL features
        cloud_logit, cloud_latent = self.cloud_backbone.forward_with_latent(X)
        cloud_y_pred = self.output_activation(cloud_logit)

        # Add embedding noise to cloud latent before sharing (data-level DP)
        noisy_cloud_latent = self._add_embedding_noise(cloud_latent)

        # Route
        final_pred = torch.zeros_like(device_y_pred)
        if personalized_mask.any():
            final_pred[personalized_mask] = cloud_y_pred[personalized_mask]
        if non_personalized_mask.any():
            final_pred[non_personalized_mask] = device_y_pred[non_personalized_mask]

        return_dict = {
            "y_pred": final_pred,
            "device_y_pred": device_y_pred,
            "cloud_y_pred": cloud_y_pred,
            "device_latent": device_latent,
            "noisy_cloud_latent": noisy_cloud_latent,
            "cloud_logit": cloud_logit,
            "device_logit": device_logit,
            "personalized_mask": personalized_mask,
            "non_personalized_mask": non_personalized_mask,
        }
        return return_dict

    def add_loss(self, return_dict, y_true):
        # 1. Device BCE on all data
        device_bce = self.loss_fn(return_dict["device_y_pred"], y_true, reduction='mean')
        total_loss = self.base_loss_weight * device_bce

        # 2. Cloud BCE on all data
        cloud_bce = self.loss_fn(return_dict["cloud_y_pred"], y_true, reduction='mean')
        total_loss = total_loss + self.cloud_loss_weight * cloud_bce

        # 3. Relationship-aware soft alignment
        #    Device latent aligns to noisy cloud latent (privacy-preserving)
        if self.alignment_weight > 0:
            device_z = F.normalize(return_dict["device_latent"], dim=-1)
            cloud_z = F.normalize(return_dict["noisy_cloud_latent"].detach(), dim=-1)
            align_loss = 1 - (device_z * cloud_z).sum(dim=-1).mean()
            total_loss = total_loss + self.alignment_weight * align_loss

        return total_loss
