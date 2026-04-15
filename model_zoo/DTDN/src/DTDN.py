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
      under Personalized-Feature Constraints (SIGIR 2026)

Architecture:
  1. Foundational Dual-Tower (Tower A + Tower B):
     - Two parallel FI towers with independent parameters
     - Both process all data with all features
     - Output masking: Tower A for personalized samples, Tower B for non-personalized
     - Foundation prediction: y_F = y_A + y_B  (Eq. 3)
  2. KD-Guided Non-Personalized Pathway:
     - Separate FI + PL processing only non-personalized features
     - Distance-based KD loss: BCE(|y_F - y_NP|, y_true) on NP samples  (Eq. 5-6)

Training:
  L_total = L_base + β * L_dis + L_reg   (Eq. 7)
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
                 tower_type="PNN",        # backbone for Tower A (personalized)
                 tower_b_type=None,       # backbone for Tower B; defaults to tower_type
                 np_tower_type=None,      # backbone for NP pathway; defaults to tower_type
                 # Parameter sharing between Tower A and Tower B
                 share_tower_params=False, # if True, Tower B = Tower A (shared weights)
                 # Feature separation
                 personalization_feature_list=None,
                 personalization_field="is_personalization",
                 # Loss weights
                 distance_loss_weight=100.0,  # β in paper (optimal range ~80-150)
                 tower_a_loss_weight=1.0,
                 tower_b_loss_weight=1.0,
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
        self.np_tower_type = np_tower_type or tower_type
        self.share_tower_params = share_tower_params
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

        # Feature separator: masks personalized features for NP pathway
        self.feature_separator = FeatureSeparator(
            self.personalization_feature_list, self.feature_map
        )

        # Backbone params passed to all tower builders
        backbone_kwargs = dict(
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

        # Create towers
        self.tower_a = self._build_tower(self.tower_type, backbone_kwargs)
        if self.share_tower_params:
            # Paper Sec 3.1.2: towers "can be configured to either share
            # parameters or operate as independent models"
            self.tower_b = self.tower_a
            logging.info("Tower A and B share parameters")
        else:
            self.tower_b = self._build_tower(self.tower_b_type, backbone_kwargs)
        self.np_pathway = self._build_tower(self.np_tower_type, backbone_kwargs)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        logging.info(f"DTDN initialized: tower_a={tower_type}, tower_b={self.tower_b_type}"
                     f"{'(shared)' if share_tower_params else ''}, "
                     f"NP={self.np_tower_type}, β={distance_loss_weight}")

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
        #   full_features:   inputs unchanged (personalized features intact)
        #   masked_features: personalized features zeroed for personalized users
        #                    (NP users already have default/padding values)
        full_features, masked_features = self.feature_separator.separate_features(
            X, per_mask
        )

        # ── Tower A & B: both process full_features ──
        ta_dict = self.tower_a.get_model_return_dict(full_features)
        tb_dict = self.tower_b.get_model_return_dict(full_features)
        y_ta = ta_dict["y_pred"]  # [B, 1]
        y_tb = tb_dict["y_pred"]  # [B, 1]

        # Output masking (Eq. 1-3)
        pm = per_mask.float()
        if y_ta.dim() == 2:
            pm = pm.unsqueeze(-1)
        y_a = y_ta * pm              # personalized predictions only
        y_b = y_tb * (1.0 - pm)      # non-personalized predictions only
        y_f = y_a + y_b              # foundation prediction

        # ── NP Pathway: uses masked_features ──
        np_dict = self.np_pathway.get_model_return_dict(masked_features)
        y_np = np_dict["y_pred"]     # [B, 1]

        # Evaluation prediction: use foundation prediction y_f
        # Paper Sec 4.4: "the auxiliary pipeline is thus not required at
        # inference, and its benefits are absorbed into the trained
        # non-personalized tower."  Tower B handles NP users.
        y_pred = y_f

        return {
            "y_pred": y_pred,       # for evaluation (= y_f)
            "y_f": y_f,             # foundation prediction (for loss)
            "y_np": y_np,           # NP pathway prediction (for loss)
            "y_ta": y_ta,           # Tower A raw prediction
            "y_tb": y_tb,           # Tower B raw prediction
            "per_mask": per_mask,   # personalization mask
            "ta_dict": ta_dict,     # full Tower A output (for custom loss)
            "tb_dict": tb_dict,     # full Tower B output (for custom loss)
            "np_dict": np_dict,     # full NP output (for custom loss)
        }

    # ─────────────────────────────────────────────────────────────
    # Loss
    # ─────────────────────────────────────────────────────────────

    def add_loss(self, return_dict, y_true):
        per_mask = return_dict["per_mask"]
        np_mask = ~per_mask
        per_count = per_mask.sum().item()
        np_count = np_mask.sum().item()

        total_loss = torch.tensor(0.0, device=y_true.device)
        total_count = per_count + np_count

        # ═══════ Base Loss (Eq. 4): L_base = (1/B) Σ ℓ(y_F, y) ═══════
        # Because y_F = y_ta for PER samples and y_tb for NP samples (output
        # masking), we compute each tower's loss on its subset and weight by
        # the subset proportion so the combined loss equals the paper's
        # mean-over-batch formulation.

        # Tower A loss — personalized samples only
        if per_count > 0:
            ta_dict = return_dict["ta_dict"]
            if self.tower_a.has_custom_loss():
                masked = {k: (v[per_mask] if isinstance(v, torch.Tensor) else v)
                          for k, v in ta_dict.items()}
                ta_loss = self.tower_a.compute_custom_loss(
                    masked, y_true[per_mask], self.loss_fn)
            else:
                ta_loss = self.loss_fn(
                    ta_dict["y_pred"][per_mask], y_true[per_mask], reduction='mean')
            total_loss = total_loss + self.tower_a_loss_weight * ta_loss * (per_count / total_count)

        # Tower B loss — non-personalized samples only
        if np_count > 0:
            tb_dict = return_dict["tb_dict"]
            if self.tower_b.has_custom_loss():
                masked = {k: (v[np_mask] if isinstance(v, torch.Tensor) else v)
                          for k, v in tb_dict.items()}
                tb_loss = self.tower_b.compute_custom_loss(
                    masked, y_true[np_mask], self.loss_fn)
            else:
                tb_loss = self.loss_fn(
                    tb_dict["y_pred"][np_mask], y_true[np_mask], reduction='mean')
            total_loss = total_loss + self.tower_b_loss_weight * tb_loss * (np_count / total_count)

        # ═══════ Distance Loss (Eq. 5-6): KD on NP samples ═══════
        if np_count > 0 and self.distance_loss_weight > 0:
            y_f_np = return_dict["y_f"][np_mask]     # foundation pred for NP
            y_np = return_dict["y_np"][np_mask]       # NP pathway pred for NP
            y_true_np = y_true[np_mask]

            # Per-sample L1 distance (Eq. 5)
            d_dis = torch.abs(y_f_np - y_np)
            d_dis = torch.clamp(d_dis, 1e-7, 1.0 - 1e-7)

            # BCE on distance (Eq. 6)
            l_dis = F.binary_cross_entropy(
                d_dis.view(-1), y_true_np.view(-1).float(), reduction='mean')

            total_loss = total_loss + self.distance_loss_weight * l_dis

        total_loss += self.regularization_loss()
        return total_loss
