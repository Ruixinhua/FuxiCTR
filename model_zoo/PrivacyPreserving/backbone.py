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
Backbone factory for KD models. Supports PNN, FinalNet, and DCNv3 (FCN).
Each backbone wraps the core computation: embedding → interaction → logit.
"""

import torch
import torch.nn as nn

from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, InnerProductInteraction
from fuxictr.pytorch.torch_utils import get_activation


class PNNBackbone(nn.Module):
    """PNN backbone: embedding → inner product → MLP → logit."""

    def __init__(self, feature_map, embedding_dim=10, hidden_units=[400, 400, 400],
                 hidden_activations="ReLU", net_dropout=0, batch_norm=False,
                 product_type="inner", **kwargs):
        super(PNNBackbone, self).__init__()
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        num_fields = feature_map.num_fields
        if product_type != "inner":
            raise NotImplementedError(f"product_type={product_type} not implemented.")
        self.inner_product_layer = InnerProductInteraction(num_fields, output="inner_product")
        input_dim = int(num_fields * (num_fields - 1) / 2) + num_fields * embedding_dim
        self.dnn = MLP_Block(input_dim=input_dim, output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        # For latent alignment: the last hidden layer dim
        self.latent_dim = hidden_units[-1] if hidden_units else input_dim

    def forward(self, X):
        feature_emb = self.embedding_layer(X)
        inner_products = self.inner_product_layer(feature_emb)
        dense_input = torch.cat([feature_emb.flatten(start_dim=1), inner_products], dim=1)
        logit = self.dnn(dense_input)
        return logit

    def forward_with_latent(self, X):
        """Return (logit, latent_features) for latent alignment."""
        feature_emb = self.embedding_layer(X)
        inner_products = self.inner_product_layer(feature_emb)
        dense_input = torch.cat([feature_emb.flatten(start_dim=1), inner_products], dim=1)
        # Run through all but the last layer to get latent
        layers = list(self.dnn.mlp)
        x = dense_input
        # The last layer is Linear(hidden[-1], 1) — everything before it is the latent
        for layer in layers[:-1]:
            x = layer(x)
        last_hidden = x
        logit = layers[-1](x)
        return logit, last_hidden


class FinalNetBackbone(nn.Module):
    """FinalNet backbone: embedding → FinalBlock(s) → logit."""

    def __init__(self, feature_map, embedding_dim=10, block_type="2B",
                 batch_norm=True, use_feature_gating=False,
                 block1_hidden_units=[64, 64, 64], block1_hidden_activations=None,
                 block1_dropout=0, block2_hidden_units=[64, 64, 64],
                 block2_hidden_activations=None, block2_dropout=0,
                 residual_type="concat", **kwargs):
        super(FinalNetBackbone, self).__init__()
        assert block_type in ["1B", "2B"], f"block_type={block_type} not supported."
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        num_fields = feature_map.num_fields
        self.use_feature_gating = use_feature_gating
        self.block_type = block_type

        if use_feature_gating:
            self.feature_gating = _FeatureGating(num_fields, gate_residual="concat")
            gate_out_dim = embedding_dim * num_fields * 2
        else:
            gate_out_dim = embedding_dim * num_fields

        self.block1 = _FinalBlock(input_dim=gate_out_dim,
                                  hidden_units=block1_hidden_units,
                                  hidden_activations=block1_hidden_activations,
                                  dropout_rates=block1_dropout,
                                  batch_norm=batch_norm,
                                  residual_type=residual_type)
        self.fc1 = nn.Linear(block1_hidden_units[-1], 1)

        if block_type == "2B":
            self.block2 = _FinalBlock(input_dim=embedding_dim * num_fields,
                                      hidden_units=block2_hidden_units,
                                      hidden_activations=block2_hidden_activations,
                                      dropout_rates=block2_dropout,
                                      batch_norm=batch_norm,
                                      residual_type=residual_type)
            self.fc2 = nn.Linear(block2_hidden_units[-1], 1)

        self.latent_dim = block1_hidden_units[-1]

    def forward(self, X):
        feature_emb = self.embedding_layer(X)
        if self.block_type == "1B":
            logit = self._forward1(feature_emb)
        else:
            y1 = self._forward1(feature_emb)
            y2 = self._forward2(feature_emb)
            logit = 0.5 * (y1 + y2)
        return logit

    def forward_with_latent(self, X):
        feature_emb = self.embedding_layer(X)
        block1_out = self.block1(self._gate_input(feature_emb))
        latent = block1_out
        y1 = self.fc1(block1_out)
        if self.block_type == "2B":
            block2_out = self.block2(feature_emb.flatten(start_dim=1))
            y2 = self.fc2(block2_out)
            logit = 0.5 * (y1 + y2)
        else:
            logit = y1
        return logit, latent

    def _gate_input(self, feature_emb):
        if self.use_feature_gating:
            return self.feature_gating(feature_emb).flatten(start_dim=1)
        return feature_emb.flatten(start_dim=1)

    def _forward1(self, feature_emb):
        x = self._gate_input(feature_emb)
        block1_out = self.block1(x)
        return self.fc1(block1_out)

    def _forward2(self, feature_emb):
        block2_out = self.block2(feature_emb.flatten(start_dim=1))
        return self.fc2(block2_out)


class DCNv3Backbone(nn.Module):
    """DCNv3 (FCN) backbone: MultiHeadEmbedding → ECN + LCN → logit."""

    def __init__(self, feature_map, embedding_dim=10,
                 num_deep_cross_layers=4, num_shallow_cross_layers=4,
                 deep_net_dropout=0.1, shallow_net_dropout=0.3,
                 layer_norm=True, batch_norm=False, num_heads=1, **kwargs):
        super(DCNv3Backbone, self).__init__()
        num_fields = feature_map.num_fields
        self.num_heads = num_heads

        self.embedding_layer = _MultiHeadFeatureEmbedding(
            feature_map, embedding_dim * num_heads, num_heads)

        cross_input_dim = num_fields * embedding_dim

        self.ECN = _ExponentialCrossNetwork(
            input_dim=cross_input_dim, num_cross_layers=num_deep_cross_layers,
            net_dropout=deep_net_dropout, layer_norm=layer_norm,
            batch_norm=batch_norm, num_heads=num_heads)
        self.LCN = _LinearCrossNetwork(
            input_dim=cross_input_dim, num_cross_layers=num_shallow_cross_layers,
            net_dropout=shallow_net_dropout, layer_norm=layer_norm,
            batch_norm=batch_norm, num_heads=num_heads)

        self.latent_dim = cross_input_dim

    def forward(self, X):
        feature_emb = self.embedding_layer(X)
        logit_d = self.ECN(feature_emb).mean(dim=1)
        logit_s = self.LCN(feature_emb).mean(dim=1)
        logit = (logit_d + logit_s) * 0.5
        return logit

    def forward_with_latent(self, X):
        feature_emb = self.embedding_layer(X)
        # Get intermediate features before final linear
        ecn_latent = self.ECN.forward_latent(feature_emb)  # B × H × D
        latent = ecn_latent.mean(dim=1)  # B × D
        logit = self.forward(X)
        return logit, latent


# ===== Backbone Factory =====

BACKBONE_REGISTRY = {
    "PNN": PNNBackbone,
    "FinalNet": FinalNetBackbone,
    "DCNv3": DCNv3Backbone,
    "FCN": DCNv3Backbone,
}


def build_backbone(backbone_type, feature_map, **kwargs):
    """Build a backbone network by type name.

    Args:
        backbone_type: One of "PNN", "FinalNet", "DCNv3", "FCN"
        feature_map: FuxiCTR FeatureMap object
        **kwargs: Backbone-specific parameters

    Returns:
        backbone (nn.Module) with .forward(X) → logit and
        .forward_with_latent(X) → (logit, latent)
    """
    if backbone_type not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone_type: {backbone_type}. "
                         f"Supported: {list(BACKBONE_REGISTRY.keys())}")
    cls = BACKBONE_REGISTRY[backbone_type]
    return cls(feature_map, **kwargs)


# ===== Internal components (FinalNet blocks, DCNv3 layers) =====
# These are self-contained copies to avoid cross-model_zoo imports.

class _FeatureGating(nn.Module):
    def __init__(self, num_fields, gate_residual="concat"):
        super(_FeatureGating, self).__init__()
        self.linear = nn.Linear(num_fields, num_fields)
        self.gate_residual = gate_residual

    def forward(self, feature_emb):
        gates = self.linear(feature_emb.transpose(1, 2)).transpose(1, 2)
        if self.gate_residual == "concat":
            return torch.cat([feature_emb, feature_emb * gates], dim=1)
        return feature_emb + feature_emb * gates


class _FactorizedInteraction(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, residual_type="sum"):
        super(_FactorizedInteraction, self).__init__()
        self.residual_type = residual_type
        if residual_type == "sum":
            output_dim = output_dim * 2
        else:
            assert output_dim % 2 == 0
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        h = self.linear(x)
        h2, h1 = torch.chunk(h, chunks=2, dim=-1)
        if self.residual_type == "concat":
            return torch.cat([h2, h1 * h2], dim=-1)
        return h2 + h1 * h2


class _FinalBlock(nn.Module):
    def __init__(self, input_dim, hidden_units=[], hidden_activations=None,
                 dropout_rates=[], batch_norm=True, residual_type="sum"):
        super(_FinalBlock, self).__init__()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.layer = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.activation = nn.ModuleList()
        dims = [input_dim] + hidden_units
        for idx in range(len(hidden_units)):
            self.layer.append(_FactorizedInteraction(dims[idx], dims[idx + 1],
                                                     residual_type=residual_type))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(dims[idx + 1]))
            if idx < len(dropout_rates) and dropout_rates[idx] > 0:
                self.dropout.append(nn.Dropout(dropout_rates[idx]))
            else:
                self.dropout.append(None)
            self.activation.append(get_activation(hidden_activations[idx]))

    def forward(self, X):
        x = X
        for i in range(len(self.layer)):
            x = self.layer[i](x)
            if i < len(self.norm):
                x = self.norm[i](x)
            if self.activation[i] is not None:
                x = self.activation[i](x)
            if self.dropout[i] is not None:
                x = self.dropout[i](x)
        return x


class _MultiHeadFeatureEmbedding(nn.Module):
    def __init__(self, feature_map, embedding_dim, num_heads=2):
        super(_MultiHeadFeatureEmbedding, self).__init__()
        self.num_heads = num_heads
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

    def forward(self, X):
        feature_emb = self.embedding_layer(X)
        multihead_feature_emb = torch.tensor_split(feature_emb, self.num_heads, dim=-1)
        multihead_feature_emb = torch.stack(multihead_feature_emb, dim=1)
        multihead_feature_emb1, multihead_feature_emb2 = torch.tensor_split(
            multihead_feature_emb, 2, dim=-1)
        multihead_feature_emb1 = multihead_feature_emb1.flatten(start_dim=2)
        multihead_feature_emb2 = multihead_feature_emb2.flatten(start_dim=2)
        return torch.cat([multihead_feature_emb1, multihead_feature_emb2], dim=-1)


class _ExponentialCrossNetwork(nn.Module):
    def __init__(self, input_dim, num_cross_layers=3, layer_norm=True,
                 batch_norm=False, net_dropout=0.1, num_heads=1):
        super(_ExponentialCrossNetwork, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        for i in range(num_cross_layers):
            self.w.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            self.b.append(nn.Parameter(torch.zeros((input_dim,))))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim // 2))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
        self.masker = nn.ReLU()
        self.dfc = nn.Linear(input_dim, 1)

    def forward(self, x):
        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)
            H = torch.cat([H, H * mask], dim=-1)
            x = x * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.dfc(x)

    def forward_latent(self, x):
        """Return features before final linear layer."""
        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)
            H = torch.cat([H, H * mask], dim=-1)
            x = x * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return x


class _LinearCrossNetwork(nn.Module):
    def __init__(self, input_dim, num_cross_layers=3, layer_norm=True,
                 batch_norm=True, net_dropout=0.1, num_heads=1):
        super(_LinearCrossNetwork, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        for i in range(num_cross_layers):
            self.w.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            self.b.append(nn.Parameter(torch.zeros((input_dim,))))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim // 2))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
        self.masker = nn.ReLU()
        self.sfc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x0 = x
        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)
            H = torch.cat([H, H * mask], dim=-1)
            x = x0 * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.sfc(x)
