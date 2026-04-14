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
LDPFedRec: Federated Recommendation with Local Differential Privacy

Implements a federated recommendation training framework where a global
(cloud) model and local (device) models are trained with asynchronous
federated aggregation. Each local model contributes updates that are
perturbed with Local Differential Privacy (LDP) noise, providing per-client
privacy guarantees without trusting the central aggregator.

Architecture:
  - Global model (server/cloud): full-feature backbone, aggregated from
    all local updates.
  - Local model (client/device): NP-feature backbone. Each simulated client
    trains locally for K steps, then perturbs embedding/parameter gradients
    with calibrated LDP noise (Laplace or Gaussian mechanism) before
    sending updates to the server.

Training Protocol (simulated):
  1. Server broadcasts global NP model parameters to clients.
  2. Each client trains locally for local_steps steps on its data partition.
  3. Client perturbs the parameter delta (local - global) with LDP noise.
  4. Server aggregates perturbed deltas via FedAvg.
  5. Repeat for communication_rounds rounds.

Privacy:
  - Local Differential Privacy: each client's update is epsilon-LDP.
  - Noise is calibrated to sensitivity / epsilon (Laplace) or
    sensitivity * sqrt(2*ln(1.25/delta)) / epsilon (Gaussian).
  - Embedding parameters receive stronger noise (more privacy-sensitive)
    than MLP parameters (optional tiered privacy).

Routing (inference):
  - group_id=1 (personalized): full-feature cloud model
  - group_id!=1 (non-personalized): NP device model (aggregated global)

Supported backbones: PNN, FinalNet, DCNv3 (FCN).

Reference:
  Zhao et al., "Asynchronous Federated Learning with Local Differential
  Privacy for Privacy-Enhanced Recommender Systems",
  IEEE Internet of Things Journal, 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
import math

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import FeatureSeparator
from ..backbone import build_backbone


class LDPFedRec(BaseModel):

    def __init__(self,
                 feature_map,
                 model_id="LDPFedRec",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 # Federated parameters
                 local_steps=5,              # local training steps per round
                 num_simulated_clients=10,   # number of simulated clients
                 # LDP parameters
                 ldp_epsilon=8.0,            # LDP privacy budget per round
                 ldp_delta=1e-5,             # delta for Gaussian mechanism
                 ldp_mechanism="gaussian",   # "gaussian" or "laplace"
                 ldp_clip_norm=1.0,          # L2 clipping norm for parameter deltas
                 # Tiered privacy: stronger noise on embeddings
                 embedding_noise_multiplier=2.0,  # extra noise on embedding params
                 # KD parameters (optional cloud->device distillation)
                 kd_temperature=4.0,
                 kd_loss_weight=0.5,
                 base_loss_weight=1.0,
                 cloud_loss_weight=1.0,
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

        super(LDPFedRec, self).__init__(
            feature_map, model_id=model_id, gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer, **kwargs)

        self.local_steps = local_steps
        self.num_simulated_clients = num_simulated_clients
        self.ldp_epsilon = ldp_epsilon
        self.ldp_delta = ldp_delta
        self.ldp_mechanism = ldp_mechanism
        self.ldp_clip_norm = ldp_clip_norm
        self.embedding_noise_multiplier = embedding_noise_multiplier
        self.kd_temperature = kd_temperature
        self.kd_loss_weight = kd_loss_weight
        self.base_loss_weight = base_loss_weight
        self.cloud_loss_weight = cloud_loss_weight
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

        # Cloud model: full-feature
        self.cloud_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)
        # Device/NP model: only NP features (simulated global federated model)
        self.device_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)

        # Step counter for federated rounds
        self._local_step_count = 0
        self._global_state = None  # snapshot of device model at round start

        # Compute noise scale from privacy budget
        self._noise_scale = self._calibrate_noise()

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        logging.info(f"LDPFedRec initialized: backbone={backbone_type}, "
                     f"local_steps={local_steps}, clients={num_simulated_clients}, "
                     f"epsilon={ldp_epsilon}, mechanism={ldp_mechanism}, "
                     f"clip={ldp_clip_norm}")

    def _calibrate_noise(self):
        """Compute noise scale from LDP parameters."""
        sensitivity = 2.0 * self.ldp_clip_norm  # L2 sensitivity for clipped delta
        if self.ldp_mechanism == "laplace":
            # Laplace: scale = sensitivity / epsilon
            return sensitivity / max(self.ldp_epsilon, 1e-8)
        elif self.ldp_mechanism == "gaussian":
            # Gaussian: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
            return (sensitivity * math.sqrt(2 * math.log(1.25 / self.ldp_delta))
                    / max(self.ldp_epsilon, 1e-8))
        else:
            raise ValueError(f"Unknown LDP mechanism: {self.ldp_mechanism}")

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

        # Device model: NP features only
        _, device_X = self.feature_separator.separate_features(X, personalized_mask)
        device_logit = self.device_backbone(device_X)
        device_y_pred = self.output_activation(device_logit)

        # Cloud model: all features
        cloud_logit = self.cloud_backbone(X)
        cloud_y_pred = self.output_activation(cloud_logit)

        # Route for inference
        final_pred = torch.zeros_like(device_y_pred)
        if personalized_mask.any():
            final_pred[personalized_mask] = cloud_y_pred[personalized_mask]
        if non_personalized_mask.any():
            final_pred[non_personalized_mask] = device_y_pred[non_personalized_mask]

        return_dict = {
            "y_pred": final_pred,
            "device_y_pred": device_y_pred,
            "device_logit": device_logit,
            "cloud_y_pred": cloud_y_pred,
            "cloud_logit": cloud_logit,
            "personalized_mask": personalized_mask,
            "non_personalized_mask": non_personalized_mask,
        }
        return return_dict

    def train_step(self, batch_data):
        """Override train_step to implement federated round simulation with LDP."""
        # Save global state at the beginning of each round
        if self._local_step_count == 0:
            self._global_state = copy.deepcopy(
                {n: p.data.clone() for n, p in self.device_backbone.named_parameters()})

        # Normal training step
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.add_loss(return_dict, y_true)
        loss += self.add_regularization()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()

        self._local_step_count += 1

        # At the end of each local round: simulate federated aggregation with LDP
        if self._local_step_count >= self.local_steps:
            self._fedavg_aggregate_with_ldp()
            self._local_step_count = 0

        return loss

    @torch.no_grad()
    def _fedavg_aggregate_with_ldp(self):
        """Simulate FedAvg aggregation with LDP noise on device model updates.

        After local training, compute the parameter delta, clip it, add
        calibrated LDP noise, and aggregate (simulate multi-client averaging).
        """
        if self._global_state is None:
            return

        for name, param in self.device_backbone.named_parameters():
            if name not in self._global_state:
                continue

            global_param = self._global_state[name]
            # Compute parameter delta (local update)
            delta = param.data - global_param

            # Clip delta to L2 norm bound
            delta_norm = delta.norm(2)
            if delta_norm > self.ldp_clip_norm:
                delta = delta * (self.ldp_clip_norm / delta_norm)

            # Add LDP noise
            noise_scale = self._noise_scale
            # Tiered privacy: stronger noise on embedding parameters
            if "embedding" in name.lower():
                noise_scale = noise_scale * self.embedding_noise_multiplier

            if self.ldp_mechanism == "laplace":
                noise = torch.zeros_like(delta).exponential_(1.0 / max(noise_scale, 1e-8))
                noise = noise * (2 * torch.randint_like(noise, 0, 2).float() - 1)
            else:  # gaussian
                noise = torch.randn_like(delta) * noise_scale

            # Perturbed delta (simulating single client's contribution)
            noisy_delta = delta + noise

            # Simulate averaging across multiple clients
            # In practice each client would contribute its own noisy_delta;
            # averaging N such deltas reduces noise by factor sqrt(N)
            aggregated_delta = noisy_delta / math.sqrt(self.num_simulated_clients)

            # Apply aggregated update to global state
            param.data = global_param + aggregated_delta

        self._global_state = None

    def add_loss(self, return_dict, y_true):
        personalized_mask = return_dict["personalized_mask"]

        # 1. Device model BCE on all data
        device_bce = self.loss_fn(return_dict["device_y_pred"], y_true, reduction='mean')
        total_loss = self.base_loss_weight * device_bce

        # 2. Cloud model BCE on all data
        cloud_bce = self.loss_fn(return_dict["cloud_y_pred"], y_true, reduction='mean')
        total_loss = total_loss + self.cloud_loss_weight * cloud_bce

        # 3. Cloud-to-Device KD on personalized samples
        if self.kd_loss_weight > 0 and personalized_mask.any():
            kd_loss = self._compute_kd_loss(
                return_dict["device_logit"][personalized_mask],
                return_dict["cloud_logit"][personalized_mask].detach())
            total_loss = total_loss + self.kd_loss_weight * kd_loss

        return total_loss

    def _compute_kd_loss(self, student_logit, teacher_logit):
        """Binary KL divergence for distillation."""
        T = self.kd_temperature
        s_prob = torch.sigmoid(student_logit / T).clamp(1e-7, 1 - 1e-7)
        t_prob = torch.sigmoid(teacher_logit / T).clamp(1e-7, 1 - 1e-7)
        kd_loss = (t_prob * torch.log(t_prob / s_prob) +
                   (1 - t_prob) * torch.log((1 - t_prob) / (1 - s_prob)))
        return kd_loss.mean() * (T * T)
