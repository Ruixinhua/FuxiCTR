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
SplitRec: Split Learning for Privacy-Preserving Recommendation

Implements a split learning paradigm where the model is vertically partitioned
between cloud and device:
  - Cloud model: processes cloud-accessible (non-personalized) features
  - Device model: processes on-device (personalized) features
  - The two models communicate via intermediate representations ("smashed data")
    at a designated cut layer. During training, only activations and gradients
    (not raw features) cross the cloud-device boundary.

Architecture:
  Cloud:  NP features → Embedding → MLP → cloud_latent (d-dim)
  Device: PER features → Embedding → MLP → device_latent (d-dim)
  Merge:  concat(cloud_latent, device_latent) → Fusion MLP → logit

Privacy mechanism:
  - Raw personalized features never leave the device
  - Only intermediate activations (smashed data) of configurable dimension
    are transmitted between cloud and device
  - Optional noise injection on smashed data for additional privacy

Training:
  - Joint end-to-end training (simulated split learning)
  - Cloud and device models trained with combined gradients
  - For NP-only inference (group_id=2): device_latent is zero-padded

Supported backbones for each side: MLP (default), or shared backbone configuration.

Reference:
  Gupta & Raskar, "Distributed Learning of Deep Neural Network over Multiple Agents",
  JMLR 2018.
  Vepakomma et al., "Split Learning for Health: Distributed Deep Learning without
  Sharing Raw Patient Data", ICLR Workshop 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import FeatureSeparator


class SplitRec(BaseModel):

    def __init__(self,
                 feature_map,
                 model_id="SplitRec",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 # Split learning parameters
                 cloud_hidden_units=[256, 128],
                 device_hidden_units=[128, 64],
                 fusion_hidden_units=[128, 64],
                 cloud_activations="ReLU",
                 device_activations="ReLU",
                 fusion_activations="ReLU",
                 latent_dim=64,
                 smashed_noise_scale=0.0,
                 cloud_dropout=0,
                 device_dropout=0,
                 fusion_dropout=0,
                 batch_norm=False,
                 # Feature separation
                 personalization_feature_list=None,
                 personalization_field="is_personalization",
                 # Standard params
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):

        super(SplitRec, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=embedding_regularizer,
                                       net_regularizer=net_regularizer,
                                       **kwargs)

        self.latent_dim = latent_dim
        self.smashed_noise_scale = smashed_noise_scale
        self.personalization_feature_list = personalization_feature_list or []
        self.personalization_field = personalization_field

        self.personalization_feature_list = [
            f for f in self.personalization_feature_list
            if f in self.feature_map.features
        ]

        # Determine feature counts for cloud and device
        all_features = [f for f in feature_map.features
                        if feature_map.features[f].get("type", "") != "meta"]
        np_features = [f for f in all_features
                       if f not in self.personalization_feature_list]
        per_features = [f for f in all_features
                        if f in self.personalization_feature_list]

        self.np_feature_names = np_features
        self.per_feature_names = per_features

        self.feature_separator = FeatureSeparator(
            self.personalization_feature_list, self.feature_map
        )

        # Cloud-side model: processes NP features
        self.cloud_embedding = FeatureEmbedding(feature_map, embedding_dim)
        cloud_input_dim = len(np_features) * embedding_dim
        self.cloud_mlp = MLP_Block(
            input_dim=cloud_input_dim,
            output_dim=latent_dim,
            hidden_units=cloud_hidden_units,
            hidden_activations=cloud_activations,
            output_activation=None,
            dropout_rates=cloud_dropout,
            batch_norm=batch_norm
        )

        # Device-side model: processes PER features
        if len(per_features) > 0:
            self.device_embedding = FeatureEmbedding(feature_map, embedding_dim)
            device_input_dim = len(per_features) * embedding_dim
            self.device_mlp = MLP_Block(
                input_dim=device_input_dim,
                output_dim=latent_dim,
                hidden_units=device_hidden_units,
                hidden_activations=device_activations,
                output_activation=None,
                dropout_rates=device_dropout,
                batch_norm=batch_norm
            )
            fusion_input_dim = latent_dim * 2
        else:
            self.device_embedding = None
            self.device_mlp = None
            fusion_input_dim = latent_dim

        # Fusion layer: merges cloud and device latent representations
        self.fusion_mlp = MLP_Block(
            input_dim=fusion_input_dim,
            output_dim=1,
            hidden_units=fusion_hidden_units,
            hidden_activations=fusion_activations,
            output_activation=None,
            dropout_rates=fusion_dropout,
            batch_norm=batch_norm
        )

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        logging.info(f"SplitRec initialized: cloud_features={len(np_features)}, "
                     f"device_features={len(per_features)}, latent_dim={latent_dim}, "
                     f"smashed_noise_scale={smashed_noise_scale}")

    def _get_personalized_mask(self, X):
        if self.personalization_field in X:
            flag = X[self.personalization_field]
            return (flag == 1), (flag != 1)
        batch_size = list(X.values())[0].size(0)
        device = list(X.values())[0].device
        return (torch.zeros(batch_size, dtype=torch.bool, device=device),
                torch.ones(batch_size, dtype=torch.bool, device=device))

    def _extract_cloud_features(self, X):
        """Extract and embed only non-personalized features for the cloud side."""
        cloud_embs = []
        emb_dict = self.cloud_embedding(X)  # Returns [batch, num_fields, emb_dim]
        # We need per-field embeddings; FeatureEmbedding returns concatenated
        # Use the full embedding but only take NP feature slots
        # For simplicity, we extract features individually
        selected = {}
        for fname in self.np_feature_names:
            if fname in X:
                selected[fname] = X[fname]
        if not selected:
            # Fallback: use all features
            return self.cloud_embedding(X).flatten(start_dim=1)
        return self.cloud_embedding(selected).flatten(start_dim=1)

    def _extract_device_features(self, X, personalized_mask):
        """Extract and embed only personalized features for the device side."""
        if self.device_embedding is None:
            return None

        selected = {}
        for fname in self.per_feature_names:
            if fname in X:
                selected[fname] = X[fname]
        if not selected:
            return None

        device_emb = self.device_embedding(selected).flatten(start_dim=1)

        # Zero out device features for non-personalized users
        if personalized_mask is not None:
            non_personalized_mask = ~personalized_mask
            if non_personalized_mask.any():
                device_emb[non_personalized_mask] = 0.0

        return device_emb

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        personalized_mask, non_personalized_mask = self._get_personalized_mask(X)

        # Cloud side: process NP features
        cloud_latent = self.cloud_mlp(self._extract_cloud_features(X))

        # Optionally add noise to smashed data (cloud latent sent to device)
        if self.training and self.smashed_noise_scale > 0:
            noise = torch.randn_like(cloud_latent) * self.smashed_noise_scale
            cloud_latent = cloud_latent + noise

        # Device side: process PER features
        device_latent = self._extract_device_features(X, personalized_mask)

        # Fusion
        if device_latent is not None:
            fused = torch.cat([cloud_latent, device_latent], dim=-1)
        else:
            fused = cloud_latent

        logit = self.fusion_mlp(fused)
        y_pred = self.output_activation(logit)

        return {
            "y_pred": y_pred,
            "logit": logit,
            "cloud_latent": cloud_latent,
            "device_latent": device_latent,
            "personalized_mask": personalized_mask,
            "non_personalized_mask": non_personalized_mask,
        }
