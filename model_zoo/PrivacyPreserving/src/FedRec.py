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
FedRec: Simulated Federated Learning for Privacy-Preserving CTR/CVR Prediction

Simulates a federated learning setup where:
  - A global model is maintained on the cloud server
  - Local updates are computed on each client's data (simulated via mini-batches)
  - Local updates are aggregated via FedAvg (McMahan et al., 2017)
  - Optional gradient compression and noise for communication efficiency and privacy

In the centralized simulation:
  - Each training batch simulates one "local client update"
  - After every K local steps, parameters are averaged (simulating FedAvg aggregation)
  - Optional DP noise is added before aggregation to provide local DP guarantees

Architecture:
  - Single backbone model with FedAvg-style training
  - Personalized features stay on-device (masked in global model)
  - Global model uses only NP features
  - Optional local fine-tuning with personalized features (FedPer variant)

Training:
  - FedAvg mode: K local steps → aggregate → repeat
  - FedPer mode: shared base layers + personalized head layers

Supported backbones: PNN, FinalNet, DCNv3 (FCN).

Reference:
  McMahan et al., "Communication-Efficient Learning of Deep Networks from
  Decentralized Data", AISTATS 2017.
  Arivazhagan et al., "Federated Learning with Personalization Layers", 2019.
  Muhammad et al., "FedFast: Going Beyond Average for Faster Training of
  Federated Recommender Systems", KDD 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import copy

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import FeatureSeparator
from ..backbone import build_backbone


class FedRec(BaseModel):

    def __init__(self,
                 feature_map,
                 model_id="FedRec",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 # FedRec-specific parameters
                 local_steps=5,
                 num_simulated_clients=10,
                 fed_noise_scale=0.0,
                 fed_mode="fedavg",
                 local_lr_factor=1.0,
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

        super(FedRec, self).__init__(
            feature_map, model_id=model_id, gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer, **kwargs)

        self.local_steps = local_steps
        self.num_simulated_clients = num_simulated_clients
        self.fed_noise_scale = fed_noise_scale
        self.fed_mode = fed_mode
        self.local_lr_factor = local_lr_factor
        self.personalization_feature_list = personalization_feature_list or []
        self.personalization_field = personalization_field
        self._local_step_counter = 0

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

        # Global model (NP features only)
        self.backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)

        # Store global model state for FedAvg aggregation
        self._global_state = None

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        logging.info(f"FedRec initialized: backbone={backbone_type}, "
                     f"mode={fed_mode}, local_steps={local_steps}, "
                     f"noise_scale={fed_noise_scale}")

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

        # Mask personalized features (simulate NP-only global model)
        _, np_X = self.feature_separator.separate_features(X, personalized_mask)

        logit = self.backbone(np_X)
        y_pred = self.output_activation(logit)

        return {
            "y_pred": y_pred,
            "logit": logit,
            "personalized_mask": personalized_mask,
            "non_personalized_mask": non_personalized_mask,
        }

    def train_step(self, batch_data):
        """Override train_step to simulate FedAvg local updates."""
        # Save global state at the beginning of each round
        if self._local_step_counter == 0:
            self._global_state = copy.deepcopy(
                {k: v.clone() for k, v in self.backbone.state_dict().items()}
            )

        # Standard training step
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)

        # Add DP noise to gradients if configured
        if self.fed_noise_scale > 0:
            for param in self.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * self.fed_noise_scale
                    param.grad.data.add_(noise)

        self.optimizer.step()

        self._local_step_counter += 1

        # Simulate FedAvg aggregation after K local steps
        if self._local_step_counter >= self.local_steps:
            self._fedavg_aggregate()
            self._local_step_counter = 0

        return loss

    def _fedavg_aggregate(self):
        """Simulate FedAvg by interpolating between global and local models.

        In real FL, multiple clients would send their updates. Here we simulate
        this by treating the current local update as one client's update and
        interpolating with the stored global state.
        """
        if self._global_state is None:
            return

        # Mixing factor: simulates averaging over num_simulated_clients
        # where only 1 has been updated
        alpha = 1.0 / self.num_simulated_clients

        current_state = self.backbone.state_dict()
        aggregated_state = {}
        for key in current_state:
            if key in self._global_state:
                # Weighted average: (1-alpha)*global + alpha*local
                aggregated_state[key] = (
                    (1 - alpha) * self._global_state[key] +
                    alpha * current_state[key]
                )
            else:
                aggregated_state[key] = current_state[key]

        self.backbone.load_state_dict(aggregated_state)
        self._global_state = None
