# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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
DTDN: Dual-Tower Distillation Networks for Robust Ads Recommendation
      under Personalized-Feature Constraints

Architecture (two towers):
  - Tower A (teacher): full features, serves PER users at inference
  - Tower B (student): NP features only (DTDN) or full features (DT-only),
    serves NP users at inference

Training: **only on PER data (group_id=1)**
  - Tower A: BCE(y_A, y) on PER data with full features
  - Tower B: BCE(y_B, y) on PER data with NP features only
  - Distance KD: BCE(|y_A - y_B|, y) on PER data — teacher signal from A to B

Operating Modes:
  - DT-only (β=0): Tower B sees full features, no KD
  - DTDN   (β>0): Tower B sees NP features only, with distance KD from Tower A

Inference: PER → Tower A, NP → Tower B

Loss:
  L_total = α_A * L_A + α_B * L_B + β * L_dis + L_reg
"""

import torch
import torch.nn.functional as F
import logging
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import FeatureSeparator


class DTDN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DTDN",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 # Tower backbone types
                 tower_type="PNN",        # backbone for Tower A (teacher)
                 tower_b_type=None,       # backbone for Tower B (student); defaults to tower_type
                 # Embedding sharing
                 share_embedding=True,
                 # Feature separation
                 personalization_feature_list=None,
                 personalization_field="is_personalization",
                 # Loss weights
                 distance_loss_weight=100.0,  # β: distance KD loss weight
                 tower_a_loss_weight=1.0,     # α_A: Tower A BCE weight
                 tower_b_loss_weight=1.0,     # α_B: Tower B BCE weight
                 # ── Shared backbone defaults ──
                 # PNN backbone params
                 hidden_units=[400, 400, 400],
                 hidden_activations="relu",
                 net_dropout=0,
                 batch_norm=False,
                 product_type="inner",
                 # FinalNet backbone params
                 block_type="2B",
                 use_feature_gating=False,
                 block1_hidden_units=[800],
                 block1_hidden_activations="ReLU",
                 block1_dropout=0.2,
                 block2_hidden_units=[800, 800],
                 block2_hidden_activations="ReLU",
                 block2_dropout=0.3,
                 residual_type="concat",
                 # DCNv3 backbone params
                 num_deep_cross_layers=3,
                 num_shallow_cross_layers=4,
                 deep_net_dropout=0.1,
                 shallow_net_dropout=0.1,
                 layer_norm=True,
                 num_heads=8,
                 # Per-tower overrides
                 tower_b_config=None,     # e.g. {"block1_hidden_units": [400]}
                 # Regularization
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):

        super(DTDN, self).__init__(feature_map,
                                   model_id=model_id,
                                   gpu=gpu,
                                   embedding_regularizer=embedding_regularizer,
                                   net_regularizer=net_regularizer,
                                   **kwargs)

        self.tower_type = tower_type
        self.tower_b_type = tower_b_type or tower_type
        self.share_embedding = share_embedding
        self.distance_loss_weight = distance_loss_weight
        self.tower_a_loss_weight = tower_a_loss_weight
        self.tower_b_loss_weight = tower_b_loss_weight
        self.personalization_field = personalization_field

        # Filter personalization features to those present in feature_map
        self.personalization_feature_list = [
            f for f in (personalization_feature_list or [])
            if f in self.feature_map.features
        ]
        if self.personalization_feature_list:
            logging.info(f"Personalization features ({len(self.personalization_feature_list)}): "
                         f"{self.personalization_feature_list}")

        # Feature separator: masks personalized features for Tower B in DTDN mode
        self.feature_separator = FeatureSeparator(
            self.personalization_feature_list, self.feature_map
        )

        # Backbone default params
        backbone_defaults = dict(
            embedding_dim=embedding_dim,
            output_activation=self.output_activation,
            # PNN
            product_type=product_type,
            hidden_units=hidden_units,
            hidden_activations=hidden_activations,
            net_dropout=net_dropout,
            batch_norm=batch_norm,
            # FinalNet
            block_type=block_type,
            use_feature_gating=use_feature_gating,
            block1_hidden_units=block1_hidden_units,
            block1_hidden_activations=block1_hidden_activations,
            block1_dropout=block1_dropout,
            block2_hidden_units=block2_hidden_units,
            block2_hidden_activations=block2_hidden_activations,
            block2_dropout=block2_dropout,
            residual_type=residual_type,
            # DCNv3
            num_deep_cross_layers=num_deep_cross_layers,
            num_shallow_cross_layers=num_shallow_cross_layers,
            deep_net_dropout=deep_net_dropout,
            shallow_net_dropout=shallow_net_dropout,
            layer_norm=layer_norm,
            num_heads=num_heads,
        )

        tower_a_kwargs = backbone_defaults
        tower_b_kwargs = {**backbone_defaults, **(tower_b_config or {})}

        # Two towers only
        self.tower_a = self._build_tower(self.tower_type, tower_a_kwargs)
        self.tower_b = self._build_tower(self.tower_b_type, tower_b_kwargs)

        # Embedding sharing
        if self.share_embedding:
            self.tower_b.embedding_layer = self.tower_a.embedding_layer
            logging.info("Tower B shares embedding layer with Tower A")

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        logging.info(f"DTDN initialized: tower_a={tower_type}, tower_b={self.tower_b_type}, "
                     f"share_emb={share_embedding}, β={distance_loss_weight}")

    def _build_tower(self, tower_type, kwargs):
        from fuxictr.pytorch.backbone import build_backbone
        return build_backbone(tower_type, self.feature_map, **kwargs)

    def _get_personalization_mask(self, X):
        """Return boolean mask: True for personalized samples."""
        if self.personalization_field in X:
            return (X[self.personalization_field] == 1).squeeze()
        batch_size = next(iter(X.values())).size(0)
        device = next(iter(X.values())).device
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    # ─────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        per_mask = self._get_personalization_mask(X)  # [B], True = personalized

        # Feature preparation
        full_features, masked_features = self.feature_separator.separate_features(
            X, per_mask
        )

        # ── Tower A: always uses full features ──
        ta_dict = self.tower_a.get_model_return_dict(full_features)
        y_ta = ta_dict["y_pred"]  # [B, 1]

        # ── Tower B: NP features in DTDN mode, full features in DT-only mode ──
        if self.distance_loss_weight > 0:
            tb_dict = self.tower_b.get_model_return_dict(masked_features)
        else:
            tb_dict = self.tower_b.get_model_return_dict(full_features)
        y_tb = tb_dict["y_pred"]  # [B, 1]

        # ── Inference routing: PER → Tower A, NP → Tower B ──
        pm = per_mask.float()
        if y_ta.dim() == 2:
            pm = pm.unsqueeze(-1)
        y_pred = torch.where(pm > 0.5, y_ta, y_tb)

        return {
            "y_pred": y_pred,       # routed prediction for evaluation
            "y_ta": y_ta,           # Tower A prediction (teacher)
            "y_tb": y_tb,           # Tower B prediction (student)
            "per_mask": per_mask,   # personalization mask
            "ta_dict": ta_dict,     # full Tower A output (for custom loss)
            "tb_dict": tb_dict,     # full Tower B output (for custom loss)
        }

    # ─────────────────────────────────────────────────────────────
    # Loss — trained only on PER data (group_id=1)
    # ─────────────────────────────────────────────────────────────

    def add_loss(self, return_dict, y_true):
        per_mask = return_dict["per_mask"]
        per_count = per_mask.sum().item()

        total_loss = torch.tensor(0.0, device=y_true.device)

        if per_count == 0:
            # No PER data in this batch — skip training
            total_loss += self.regularization_loss()
            return total_loss

        y_true_per = y_true[per_mask]

        # ═══════ Tower A BCE on PER data ═══════
        ta_dict = return_dict["ta_dict"]
        if self.tower_a.has_custom_loss():
            masked = {k: (v[per_mask] if isinstance(v, torch.Tensor) else v)
                      for k, v in ta_dict.items()}
            ta_loss = self.tower_a.compute_custom_loss(
                masked, y_true_per, self.loss_fn)
        else:
            ta_loss = self.loss_fn(
                ta_dict["y_pred"][per_mask], y_true_per, reduction='mean')
        total_loss = total_loss + self.tower_a_loss_weight * ta_loss

        # ═══════ Tower B BCE on PER data (NP features only in DTDN mode) ═══════
        tb_dict = return_dict["tb_dict"]
        if self.tower_b.has_custom_loss():
            masked = {k: (v[per_mask] if isinstance(v, torch.Tensor) else v)
                      for k, v in tb_dict.items()}
            tb_loss = self.tower_b.compute_custom_loss(
                masked, y_true_per, self.loss_fn)
        else:
            tb_loss = self.loss_fn(
                tb_dict["y_pred"][per_mask], y_true_per, reduction='mean')
        total_loss = total_loss + self.tower_b_loss_weight * tb_loss

        # ═══════ Distance KD loss on PER data ═══════
        if self.distance_loss_weight > 0:
            y_ta_per = return_dict["y_ta"][per_mask]
            y_tb_per = return_dict["y_tb"][per_mask]

            # |y_A - y_B|: how far student is from teacher
            d_dis = torch.abs(y_ta_per - y_tb_per)
            d_dis = torch.clamp(d_dis, 1e-7, 1.0 - 1e-7)

            # BCE(distance, y_true): distance should be small when y=0,
            # large when y=1 (teacher-student agreement matters more on
            # positive samples)
            l_dis = F.binary_cross_entropy(
                d_dis.view(-1), y_true_per.view(-1).float(), reduction='mean')

            total_loss = total_loss + self.distance_loss_weight * l_dis

        total_loss += self.regularization_loss()
        return total_loss
