# =========================================================================
# Copyright (C) 2024 salmon@github
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

import torch
import logging
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding


class DCNv3(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DCNv3",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_deep_cross_layers=4,
                 num_shallow_cross_layers=4,
                 deep_net_dropout=0.1,
                 shallow_net_dropout=0.3,
                 layer_norm=True,
                 batch_norm=False,
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 use_domain_aware_structure=False,
                 **kwargs):
        super(DCNv3, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.hparams = kwargs
        self.use_domain_aware_structure = use_domain_aware_structure
        self.num_heads = num_heads

        self.embedding_layer = MultiHeadFeatureEmbedding(feature_map, embedding_dim * num_heads, num_heads)
        
        cross_input_dim = feature_map.num_fields * embedding_dim

        self.ECN = ExponentialCrossNetwork(input_dim=cross_input_dim,
                                           num_cross_layers=num_deep_cross_layers,
                                           net_dropout=deep_net_dropout,
                                           layer_norm=layer_norm,
                                           batch_norm=batch_norm,
                                           num_heads=num_heads,
                                           output_intermediate_features=self.use_domain_aware_structure)
        self.LCN = LinearCrossNetwork(input_dim=cross_input_dim,
                                      num_cross_layers=num_shallow_cross_layers,
                                      net_dropout=shallow_net_dropout,
                                      layer_norm=layer_norm,
                                      batch_norm=batch_norm,
                                      num_heads=num_heads,
                                      output_intermediate_features=self.use_domain_aware_structure)

        if self.use_domain_aware_structure:
            tower_input_dim = num_heads * cross_input_dim
            self._init_domain_aware_structure_params_pytorch(tower_input_dim)
            self.logits_xld = None
            self.logits_xls = None

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def _init_domain_aware_structure_params_pytorch(self, tower_input_dim):
        self.tower_hidden_units_list = self.hparams.get("tower_hidden_units_list")
        if self.tower_hidden_units_list is None:
            raise ValueError("`tower_hidden_units_list` must be specified for domain-aware structure.")
        self.scene_num = len(self.tower_hidden_units_list)
        if self.scene_num <= 0:
            raise ValueError("`tower_hidden_units_list` cannot be empty.")

        self.tower_activation = self.hparams.get("tower_activation", "relu")
        self.tower_l2_reg_list = self.hparams.get("tower_l2_reg_list", [0.0] * self.scene_num)
        self.tower_dropout_list = self.hparams.get("tower_dropout_list", [0.0] * self.scene_num)
        self.use_bn_tower = self.hparams.get("use_bn_tower", True)
        
        self.tower_dnns = nn.ModuleList()
        self.tower_output_layers = nn.ModuleList()

        for i in range(self.scene_num):
            self.tower_dnns.append(DNN(inputs_dim=tower_input_dim,
                                       hidden_units=self.tower_hidden_units_list[i],
                                       activation=self.tower_activation,
                                       l2_reg=self.tower_l2_reg_list[i],
                                       dropout_rates=self.tower_dropout_list[i],
                                       batch_norm=self.use_bn_tower,
                                       use_bias=True))
            dnn_output_dim = self.tower_hidden_units_list[i][-1] if self.tower_hidden_units_list[i] else tower_input_dim
            self.tower_output_layers.append(nn.Linear(dnn_output_dim, 1))

        self.scene_name = self.hparams.get("scene_name", "scene_id")
        self.scene_num_shift = self.hparams.get("scene_num_shift", 1)
        self.use_scene_id_mapping = self.hparams.get("use_scene_id_mapping", False)
        if self.use_scene_id_mapping:
            self.mapping_feature_name = self.hparams.get("mapping_feature_name")
            if not self.mapping_feature_name:
                raise ValueError("`mapping_feature_name` required for scene_id mapping.")
            self.mapping_feature_type = self.hparams.get("mapping_feature_type")
            self.feature2id_dict = self.hparams.get("feature2id_dict")
            if not self.feature2id_dict:
                raise ValueError("`feature2id_dict` required for scene_id mapping.")
            self.default_value = self.hparams.get("default_value")
            if self.default_value is None or not (1 <= self.default_value <= self.scene_num):
                raise ValueError(f"`default_value` ({self.default_value}) must be a valid 1-based scene_id (1 to {self.scene_num}).")
            if self.mapping_feature_type == "sparse":
                self.feature_map_dict = self.hparams.get("feature_map_dict")
                if not self.feature_map_dict:
                     raise ValueError("`feature_map_dict` required for sparse scene_id mapping.")
        logging.info(f"Domain-aware structure initialized with {self.scene_num} towers for PyTorch DCNv3.")


    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        if self.use_domain_aware_structure:
            xld_intermediate = self.ECN(feature_emb)
            xls_intermediate = self.LCN(feature_emb)

            xld_flat = xld_intermediate.view(xld_intermediate.size(0), -1)
            xls_flat = xls_intermediate.view(xls_intermediate.size(0), -1)

            self.logits_xld = self._generate_domain_aware_logits_pytorch(X, xld_flat)
            self.logits_xls = self._generate_domain_aware_logits_pytorch(X, xls_flat)
        else:
            self.logits_xld = self.ECN(feature_emb).mean(dim=1)
            self.logits_xls = self.LCN(feature_emb).mean(dim=1)
        
        logit = (self.logits_xld + self.logits_xls) * 0.5
        y_pred = self.output_activation(logit)
        
        return_dict = {"y_pred": y_pred,
                       "y_d": self.output_activation(self.logits_xld),
                       "y_s": self.output_activation(self.logits_xls)}
        return return_dict

    def _generate_domain_aware_logits_pytorch(self, X_features, net_output):
        scene_id_0_indexed = self._scene_id_mapping_pytorch(X_features)

        tower_logits_list = []
        for i in range(self.scene_num):
            tower_dnn_out = self.tower_dnns[i](net_output)
            tower_logit = self.tower_output_layers[i](tower_dnn_out)
            tower_logits_list.append(tower_logit)
        
        scene_tower_output_concat = torch.cat(tower_logits_list, dim=1)
        final_logits = self._logits_routing_pytorch(scene_tower_output_concat, scene_id_0_indexed)
        return final_logits

    def _scene_id_mapping_pytorch(self, X_features):
        if self.use_scene_id_mapping:
            feature_values = X_features.get(self.mapping_feature_name)
            if feature_values is None:
                raise ValueError(f"Mapping feature '{self.mapping_feature_name}' not found in input features.")
            
            if feature_values.ndim > 1:
                 feature_values = feature_values.squeeze(-1)

            default_scene_id_0_indexed = torch.tensor(self.default_value - self.scene_num_shift, 
                                                      device=feature_values.device, dtype=torch.long)
            scene_ids = torch.full_like(feature_values, default_scene_id_0_indexed, dtype=torch.long)

            for feat_val_str, scene_id_1_based in self.feature2id_dict.items():
                target_scene_id_0_indexed = torch.tensor(scene_id_1_based - self.scene_num_shift, 
                                                         device=feature_values.device, dtype=torch.long)
                
                current_feat_val_for_comp = None
                if self.mapping_feature_type == 'sparse':
                    mapped_int_val = self.feature_map_dict.get(feat_val_str)
                    if mapped_int_val is None:
                        logging.warning(f"Feature value \'{feat_val_str}\' not in feature_map_dict. Skipping for scene_id mapping.")
                        continue
                    current_feat_val_for_comp = torch.tensor(mapped_int_val, device=feature_values.device, dtype=feature_values.dtype)
                else:
                    try:
                        current_feat_val_for_comp = torch.tensor(int(feat_val_str), device=feature_values.device, dtype=feature_values.dtype) \
                                                if feature_values.dtype != torch.float32 else \
                                                torch.tensor(float(feat_val_str), device=feature_values.device, dtype=feature_values.dtype)
                    except ValueError:
                        logging.warning(f"Cannot convert feature value \'{feat_val_str}\' to numeric type for comparison. Skipping.")
                        continue
                
                scene_ids = torch.where(feature_values == current_feat_val_for_comp, 
                                        target_scene_id_0_indexed, 
                                        scene_ids)
            return scene_ids
        else:
            scene_id_tensor = X_features.get(self.scene_name)
            if scene_id_tensor is None:
                 scene_id_tensor = X_features.get('scene_id')
                 if scene_id_tensor is None:
                      raise ValueError(f"Scene feature '{self.scene_name}' (and fallback 'scene_id') not found in input features.")

            if scene_id_tensor.ndim > 1:
                scene_id_tensor = scene_id_tensor.squeeze(-1)
            scene_id_tensor = scene_id_tensor.long()
            return scene_id_tensor - self.scene_num_shift

    def _logits_routing_pytorch(self, scene_tower_output_concat, scene_id_0_indexed):
        num_towers = scene_tower_output_concat.size(1)
        scene_select = F.one_hot(scene_id_0_indexed, num_classes=num_towers)
        
        scene_select = scene_select.to(scene_tower_output_concat.dtype)
        
        final_logits = torch.sum(scene_tower_output_concat * scene_select, dim=-1, keepdim=True)
        return final_logits

    def add_loss(self, return_dict, y_true):
        y_pred = return_dict["y_pred"]
        y_d = return_dict["y_d"]
        y_s = return_dict["y_s"]
        
        loss = self.loss_fn(y_pred, y_true, reduction='mean')
        loss_d = self.loss_fn(y_d, y_true, reduction='mean')
        loss_s = self.loss_fn(y_s, y_true, reduction='mean')
        
        weight_d = loss_d - loss
        weight_s = loss_s - loss
        
        weight_d = torch.where(weight_d > 0, weight_d, torch.zeros_like(weight_d))
        weight_s = torch.where(weight_s > 0, weight_s, torch.zeros_like(weight_s))
        
        total_loss = loss + loss_d * weight_d + loss_s * weight_s
        return total_loss

class MultiHeadFeatureEmbedding(nn.Module):
    def __init__(self, feature_map, embedding_dim, num_heads=2):
        super(MultiHeadFeatureEmbedding, self).__init__()
        self.num_heads = num_heads
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

    def forward(self, X):  # H = num_heads
        feature_emb = self.embedding_layer(X)  # B × F × D
        multihead_feature_emb = torch.tensor_split(feature_emb, self.num_heads, dim=-1)
        multihead_feature_emb = torch.stack(multihead_feature_emb, dim=1)  # B × H × F × D/H
        multihead_feature_emb1, multihead_feature_emb2 = torch.tensor_split(multihead_feature_emb, 2,
                                                                            dim=-1)  # B × H × F × D/2H
        multihead_feature_emb1, multihead_feature_emb2 = multihead_feature_emb1.flatten(start_dim=2), \
                                                         multihead_feature_emb2.flatten(
                                                             start_dim=2)  # B × H × FD/2H; B × H × FD/2H
        multihead_feature_emb = torch.cat([multihead_feature_emb1, multihead_feature_emb2], dim=-1)
        return multihead_feature_emb  # B × H × FD/H


class ExponentialCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 layer_norm=True,
                 batch_norm=False,
                 net_dropout=0.1,
                 num_heads=1,
                 output_intermediate_features=False):
        super(ExponentialCrossNetwork, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.output_intermediate_features = output_intermediate_features
        self.intermediate_output_dim = input_dim

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
        
        if self.output_intermediate_features:
            return x
        
        logit = self.dfc(x)
        return logit


class LinearCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 layer_norm=True,
                 batch_norm=True,
                 net_dropout=0.1,
                 num_heads=1,
                 output_intermediate_features=False):
        super(LinearCrossNetwork, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.output_intermediate_features = output_intermediate_features
        self.intermediate_output_dim = input_dim

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
        
        if self.output_intermediate_features:
            return x

        logit = self.sfc(x)
        return logit


class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rates=0,
                 batch_norm=False, use_bias=True, seed=1024, output_layer=False):
        super(DNN, self).__init__()
        if not isinstance(hidden_units, list):
            raise ValueError("hidden_units must be a list")

        self.dropout_rates = dropout_rates
        self.batch_norm = batch_norm
        self.output_layer = output_layer

        hidden_layers = []
        input_size = inputs_dim
        for i, layer_size in enumerate(hidden_units):
            hidden_layers.append(nn.Linear(input_size, layer_size, bias=use_bias))
            if self.batch_norm:
                hidden_layers.append(nn.BatchNorm1d(layer_size))
            
            # Activation
            if activation == "relu":
                hidden_layers.append(nn.ReLU())
            elif activation == "sigmoid":
                hidden_layers.append(nn.Sigmoid())
            elif activation == "tanh":
                hidden_layers.append(nn.Tanh())
            # Add other activations as needed, e.g., PReLU, Dice
            else: # Default or if activation is an nn.Module instance
                try:
                    if isinstance(activation, str):
                         act_fn = getattr(nn, activation, None)()
                         if act_fn is not None:
                              hidden_layers.append(act_fn)
                         else: 
                              raise ValueError(f"Activation {activation} not found in nn module and not a known string.")
                    elif isinstance(activation, nn.Module):
                         hidden_layers.append(activation)
                    else:
                         raise ValueError(f"Activation {activation} not supported.")
                except Exception as e:
                    raise ValueError(f"Error creating activation {activation}: {e}")


            if isinstance(self.dropout_rates, list):
                if i < len(self.dropout_rates) and self.dropout_rates[i] > 0:
                    hidden_layers.append(nn.Dropout(self.dropout_rates[i]))
            elif self.dropout_rates > 0:
                 hidden_layers.append(nn.Dropout(self.dropout_rates))
            input_size = layer_size
        
        self.dnn = nn.Sequential(*hidden_layers)

        if self.output_layer: # Optional final linear layer if DNN itself is the output
            self.fc = nn.Linear(input_size, 1)

        # weight initialization - can be added if specific init is needed
        # for name, param in self.dnn.named_parameters():
        #     if 'weight' in name:
        #         nn.init.xavier_normal_(param)

    def forward(self, x):
        x = self.dnn(x)
        if self.output_layer:
            x = self.fc(x)
        return x

