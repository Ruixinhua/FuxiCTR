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
DTDN: Dual-Tower Distillation Networks

Reimplementation of DualTowerModel + DualTowerCL using unified backbone,
faithfully replicating old DTCN training logic.

Architecture:
  - Tower A (personalized/teacher): processes full features
  - Tower B (non-personalized/student): processes features with PER features masked

Matching old DualTowerModel + DualTowerCL behavior:
  use_mask_for_all=False (default):
    - Feature masking: PER features zeroed for ALL samples in Tower B input
    - Training: both towers on ALL data (mask=all-True overrides use_all_data)
    - KD losses computed on ALL data

  use_mask_for_all=True:
    - Feature masking: PER features zeroed only for PER samples in Tower B
    - Training: tower_a_use_all_data / tower_b_use_all_data control data subsets

  Routing (ALWAYS proper, regardless of use_mask_for_all):
    - PER samples → Tower A prediction
    - NP samples → Tower B prediction
    NOTE: Old code used y_pred = y_ta + y_tb (sum of sigmoids in [0,2]),
    which broke logloss computation. Fixed to proper per-group routing.

KD losses (matching old CL module, controlled by kd_loss_weight):
  - distance_loss_weight:  MSE(y_ta, y_tb)
  - knowledge_distillation_loss_weight: KL(teacher || student) with temperature
  - group_aware_loss_weight: BCE(y_tb, y_true) on NP samples

Loss:
  L = α_A * L_A + α_B * L_B + kd_w * (dist_w * MSE + kd_w * KL + ga_w * BCE) + L_reg
"""

import os
import numpy as np
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
                 tower_type="PNN",
                 tower_b_type=None,       # defaults to tower_type
                 # Embedding sharing
                 share_embedding=True,
                 # Feature separation
                 personalization_feature_list=None,
                 personalization_field="is_personalization",
                 # Mask mode (matches old DualTowerModel.use_mask_for_all)
                 use_mask_for_all=False,
                 # Training data config (effective when use_mask_for_all=True)
                 tower_a_use_all_data=False,   # Tower A on PER data only (default)
                 tower_b_use_all_data=True,    # Tower B on ALL data (default)
                 # Loss weights
                 tower_a_loss_weight=1.0,
                 tower_b_loss_weight=1.0,
                 # KD config (replaces old CL module)
                 kd_loss_weight=1.0,                       # = old cl_loss_weight
                 knowledge_distillation_loss_weight=0.0,   # KL div component
                 group_aware_loss_weight=0.0,              # group-aware BCE
                 distance_loss_weight=0.0,                 # MSE distance
                 temperature=4.0,                          # KD temperature
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
                 tower_b_config=None,
                 # Tower-specific monitoring
                 use_tower_specific_monitoring=True,
                 personalized_monitor_metric="AUC_group_1.0",
                 non_personalized_monitor_metric="AUC_group_2.0",
                 tower_patience=3,
                 save_tower_models=True,
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
        self.personalization_field = personalization_field

        # Mask/routing mode
        self.use_mask_for_all = use_mask_for_all
        self.tower_a_use_all_data = tower_a_use_all_data
        self.tower_b_use_all_data = tower_b_use_all_data

        # Loss weights
        self.tower_a_loss_weight = tower_a_loss_weight
        self.tower_b_loss_weight = tower_b_loss_weight

        # KD config
        self.kd_loss_weight = kd_loss_weight
        self.knowledge_distillation_loss_weight = knowledge_distillation_loss_weight
        self.group_aware_loss_weight = group_aware_loss_weight
        self.distance_loss_weight = distance_loss_weight
        self.temperature = temperature
        self.use_kd = (kd_loss_weight > 0 and
                       (distance_loss_weight > 0 or
                        knowledge_distillation_loss_weight > 0 or
                        group_aware_loss_weight > 0))

        # Filter personalization features to those present in feature_map
        self.personalization_feature_list = [
            f for f in (personalization_feature_list or [])
            if f in self.feature_map.features
        ]
        if self.personalization_feature_list:
            logging.info(f"Personalization features ({len(self.personalization_feature_list)}): "
                         f"{self.personalization_feature_list}")

        # Feature separator
        self.feature_separator = FeatureSeparator(
            self.personalization_feature_list, self.feature_map
        )

        # Backbone params
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

        # Tower-specific monitoring config
        self.use_tower_specific_monitoring = use_tower_specific_monitoring
        self.personalized_monitor_metric = personalized_monitor_metric
        self.non_personalized_monitor_metric = non_personalized_monitor_metric
        self.tower_patience = tower_patience
        self.save_tower_models = save_tower_models

        # Build towers
        self.tower_a = self._build_tower(self.tower_type, tower_a_kwargs)
        self.tower_b = self._build_tower(self.tower_b_type, tower_b_kwargs)

        # Embedding sharing
        if self.share_embedding:
            self.tower_b.embedding_layer = self.tower_a.embedding_layer
            logging.info("Tower B shares embedding layer with Tower A")

        # Init tower monitoring
        if self.use_tower_specific_monitoring:
            self._init_tower_monitoring()

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        logging.info(f"DTDN initialized: tower_a={tower_type}, tower_b={self.tower_b_type}, "
                     f"share_emb={share_embedding}, use_mask_for_all={use_mask_for_all}")
        logging.info(f"  KD config: kd_w={kd_loss_weight}, dist_w={distance_loss_weight}, "
                     f"kd_kl_w={knowledge_distillation_loss_weight}, ga_w={group_aware_loss_weight}, "
                     f"T={temperature}")
        logging.info(f"  Training: tower_a_use_all={tower_a_use_all_data}, "
                     f"tower_b_use_all={tower_b_use_all_data}")

    def _build_tower(self, tower_type, kwargs):
        from fuxictr.pytorch.backbone import build_backbone
        return build_backbone(tower_type, self.feature_map, **kwargs)

    def _get_personalization_mask(self, X):
        """Return boolean mask: True for personalized samples (group_id=1)."""
        if self.personalization_field in X:
            return (X[self.personalization_field] == 1).squeeze()
        batch_size = next(iter(X.values())).size(0)
        device = next(iter(X.values())).device
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    # ─────────────────────────────────────────────────────────────
    # Forward — matches old DualTowerModel.forward()
    # ─────────────────────────────────────────────────────────────

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        per_mask = self._get_personalization_mask(X)  # real group mask

        # Feature separation mask (matches old use_mask_for_all logic)
        if not self.use_mask_for_all:
            # Old default: mask PER features for ALL samples in Tower B
            feat_mask = torch.ones_like(per_mask)
        else:
            # Proper mode: only mask PER features for PER samples
            feat_mask = per_mask

        full_features, masked_features = self.feature_separator.separate_features(
            X, feat_mask
        )

        # Tower A: full features
        ta_dict = self.tower_a.get_model_return_dict(full_features)
        y_ta = ta_dict["y_pred"]

        # Tower B: masked features (PER features zeroed)
        tb_dict = self.tower_b.get_model_return_dict(masked_features)
        y_tb = tb_dict["y_pred"]

        # Routing: ALWAYS use proper per-group routing
        # PER samples → Tower A, NP samples → Tower B
        # NOTE: Old code did y_pred = y_ta + y_tb when use_mask_for_all=False,
        # which produced values in [0,2] (sum of two sigmoid outputs) and broke
        # logloss computation. Proper routing fixes this.
        pm = per_mask.float()
        if y_ta.dim() == 2:
            pm = pm.unsqueeze(-1)
        y_pred = torch.where(pm > 0.5, y_ta, y_tb)

        return {
            "y_pred": y_pred,
            "y_ta": y_ta,
            "y_tb": y_tb,
            "per_mask": per_mask,   # always the REAL group mask
            "ta_dict": ta_dict,
            "tb_dict": tb_dict,
        }

    # ─────────────────────────────────────────────────────────────
    # Loss — matches old DualTowerModel.add_loss() + DualTowerCL CL losses
    # ─────────────────────────────────────────────────────────────

    def _compute_tower_loss(self, tower, tower_dict, y_true, mask):
        """Compute BCE (or custom) loss for a tower on masked samples."""
        count = mask.sum().item()
        if count == 0:
            return torch.tensor(0.0, device=y_true.device)

        y_true_m = y_true[mask]
        if tower.has_custom_loss():
            masked_dict = {k: (v[mask] if isinstance(v, torch.Tensor) else v)
                           for k, v in tower_dict.items()}
            return tower.compute_custom_loss(masked_dict, y_true_m, self.loss_fn)
        else:
            return self.loss_fn(tower_dict["y_pred"][mask], y_true_m, reduction='mean')

    def _compute_kd_loss(self, y_ta, y_tb, y_true, per_mask):
        """Compute KD losses matching old ContrastiveLearningBase.compute_cl_loss()."""
        kd_total = torch.tensor(0.0, device=y_true.device)

        # Distance loss: MSE(y_ta, y_tb) on all data
        if self.distance_loss_weight > 0:
            dist_loss = F.mse_loss(y_ta, y_tb, reduction='mean')
            kd_total = kd_total + self.distance_loss_weight * dist_loss

        # Knowledge distillation: KL(teacher || student) with temperature
        # Matches old compute_knowledge_distillation_loss exactly
        if self.knowledge_distillation_loss_weight > 0:
            T = self.temperature
            eps = 1e-7
            teacher_probs = torch.clamp(torch.sigmoid(y_ta.squeeze() / T), eps, 1 - eps)
            student_probs = torch.clamp(torch.sigmoid(y_tb.squeeze() / T), eps, 1 - eps)

            teacher_full = torch.stack([1 - teacher_probs, teacher_probs], dim=-1)
            student_log_full = torch.stack([
                torch.log(1 - student_probs + 1e-8),
                torch.log(student_probs + 1e-8)
            ], dim=-1)

            kd_loss = F.kl_div(student_log_full, teacher_full,
                               reduction='batchmean') * (T ** 2)
            kd_total = kd_total + self.knowledge_distillation_loss_weight * kd_loss

        # Group-aware loss: BCE for Tower B on NP samples
        # Matches old compute_group_aware_loss with group_ids
        if self.group_aware_loss_weight > 0:
            np_mask = ~per_mask
            if np_mask.any():
                ga_loss = F.binary_cross_entropy_with_logits(
                    y_tb[np_mask].squeeze(-1),
                    y_true[np_mask].squeeze(-1).float(),
                    reduction='mean')
                kd_total = kd_total + self.group_aware_loss_weight * ga_loss

        return kd_total

    def add_loss(self, return_dict, y_true):
        per_mask = return_dict["per_mask"]  # real group mask
        ta_dict = return_dict["ta_dict"]
        tb_dict = return_dict["tb_dict"]

        total_loss = torch.tensor(0.0, device=y_true.device)

        # ═══════ Determine training masks ═══════
        # Matches old DualTowerModel behavior:
        # When use_mask_for_all=False, personalized_mask=all-True in add_loss,
        # so tower_a_use_all_data=False + all-True mask → ALL data
        if not self.use_mask_for_all:
            # Both towers train on ALL data (old default)
            ta_train_mask = torch.ones(per_mask.size(0), dtype=torch.bool,
                                       device=per_mask.device)
            tb_train_mask = torch.ones_like(ta_train_mask)
        else:
            # Proper mode: respect use_all_data settings
            if self.tower_a_use_all_data:
                ta_train_mask = torch.ones(per_mask.size(0), dtype=torch.bool,
                                           device=per_mask.device)
            else:
                ta_train_mask = per_mask
            if self.tower_b_use_all_data:
                tb_train_mask = torch.ones(per_mask.size(0), dtype=torch.bool,
                                           device=per_mask.device)
            else:
                tb_train_mask = ~per_mask

        # ═══════ Tower A loss ═══════
        ta_loss = self._compute_tower_loss(self.tower_a, ta_dict, y_true, ta_train_mask)
        total_loss = total_loss + self.tower_a_loss_weight * ta_loss

        # ═══════ Tower B loss ═══════
        tb_loss = self._compute_tower_loss(self.tower_b, tb_dict, y_true, tb_train_mask)
        total_loss = total_loss + self.tower_b_loss_weight * tb_loss

        # ═══════ KD losses (on ALL data, matching old CL loss) ═══════
        if self.use_kd:
            kd_loss = self._compute_kd_loss(
                return_dict["y_ta"], return_dict["y_tb"],
                y_true, per_mask)
            total_loss = total_loss + self.kd_loss_weight * kd_loss

        total_loss += self.regularization_loss()
        return total_loss

    # ─────────────────────────────────────────────────────────────
    # Tower-specific monitoring (from old DualTowerModel)
    # ─────────────────────────────────────────────────────────────

    def _init_tower_monitoring(self):
        self.tower_monitoring = {
            "tower_a": {
                "best_metric": -np.inf if "AUC" in self.personalized_monitor_metric else np.inf,
                "best_epoch": 0,
                "patience_count": 0,
                "model_path": self.checkpoint,
                "is_better": (lambda c, b: c > b) if "AUC" in self.personalized_monitor_metric else (lambda c, b: c < b),
            },
            "tower_b": {
                "best_metric": -np.inf if "AUC" in self.non_personalized_monitor_metric else np.inf,
                "best_epoch": 0,
                "patience_count": 0,
                "model_path": self.checkpoint,
                "is_better": (lambda c, b: c > b) if "AUC" in self.non_personalized_monitor_metric else (lambda c, b: c < b),
            },
        }
        logging.info(f"Tower monitoring: tower_a={self.personalized_monitor_metric}, "
                     f"tower_b={self.non_personalized_monitor_metric}, patience={self.tower_patience}")

    def update_tower_monitoring(self, eval_metrics, current_epoch):
        if not self.use_tower_specific_monitoring:
            return
        for tower_key, metric_name in [("tower_a", self.personalized_monitor_metric),
                                        ("tower_b", self.non_personalized_monitor_metric)]:
            if metric_name not in eval_metrics:
                continue
            val = eval_metrics[metric_name]
            info = self.tower_monitoring[tower_key]
            if info["is_better"](val, info["best_metric"]):
                info["best_metric"] = val
                info["best_epoch"] = current_epoch
                info["patience_count"] = 0
                if self.save_tower_models:
                    path = f"{self.checkpoint}_{tower_key}_best.model"
                    self._save_tower_model(tower_key, path)
                    info["model_path"] = path
                logging.info(f"New best {tower_key}: {metric_name}={val:.6f} at epoch {current_epoch}")
            else:
                info["patience_count"] += 1

    def should_early_stop_towers(self):
        if not self.use_tower_specific_monitoring:
            return False
        a_exceeded = self.tower_monitoring["tower_a"]["patience_count"] >= self.tower_patience
        b_exceeded = self.tower_monitoring["tower_b"]["patience_count"] >= self.tower_patience
        should_stop = a_exceeded and b_exceeded
        if should_stop:
            logging.info(f"Tower early stop: tower_a patience "
                         f"{self.tower_monitoring['tower_a']['patience_count']}/{self.tower_patience}, "
                         f"tower_b patience "
                         f"{self.tower_monitoring['tower_b']['patience_count']}/{self.tower_patience}")
        return should_stop

    def _save_tower_model(self, tower_key, model_path):
        tower = self.tower_a if tower_key == "tower_a" else self.tower_b
        try:
            state = {
                "model_state_dict": tower.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_metric": self.tower_monitoring[tower_key]["best_metric"],
                "best_epoch": self.tower_monitoring[tower_key]["best_epoch"],
            }
            torch.save(state, model_path)
            logging.debug(f"{tower_key} model saved to: {model_path}")
        except Exception as e:
            logging.error(f"Failed to save {tower_key} model: {e}")

    def _load_tower_optimal_models(self):
        loaded = []
        for tower_key in ["tower_a", "tower_b"]:
            tower = self.tower_a if tower_key == "tower_a" else self.tower_b
            path = self.tower_monitoring[tower_key]["model_path"]
            if path and os.path.exists(path):
                try:
                    state = torch.load(path, map_location=self.device, weights_only=False)
                    tower.load_state_dict(state["model_state_dict"])
                    loaded.append(tower_key)
                    metric_name = self.personalized_monitor_metric if tower_key == "tower_a" else self.non_personalized_monitor_metric
                    logging.info(f"Loaded {tower_key} best model: {metric_name}="
                                 f"{self.tower_monitoring[tower_key]['best_metric']:.6f} "
                                 f"at epoch {self.tower_monitoring[tower_key]['best_epoch']}")
                except Exception as e:
                    logging.error(f"Failed to load {tower_key} model: {e}")

        if len(loaded) < 2:
            logging.warning(f"Only loaded {loaded}, falling back to global checkpoint")
            if not loaded:
                self.load_weights(self.checkpoint)

        # Clean up checkpoints if configured
        if not self._save_checkpoints:
            for path in [self.checkpoint] + [self.tower_monitoring[k]["model_path"] for k in ["tower_a", "tower_b"]]:
                if path and os.path.exists(path):
                    logging.info(f"Remove checkpoint: {path}")
                    os.remove(path)

    def get_tower_monitoring_summary(self):
        if not self.use_tower_specific_monitoring:
            return {}
        return {
            "tower_a": {
                "metric": self.personalized_monitor_metric,
                "best_value": self.tower_monitoring["tower_a"]["best_metric"],
                "best_epoch": self.tower_monitoring["tower_a"]["best_epoch"],
            },
            "tower_b": {
                "metric": self.non_personalized_monitor_metric,
                "best_value": self.tower_monitoring["tower_b"]["best_metric"],
                "best_epoch": self.tower_monitoring["tower_b"]["best_epoch"],
            },
        }

    # ─────────────────────────────────────────────────────────────
    # Training loop overrides for tower monitoring
    # ─────────────────────────────────────────────────────────────

    def fit(self, data_generator, epochs=1, validation_data=None, **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = kwargs.get('max_gradient_norm', 10.)
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")

        for epoch in range(epochs):
            self._epoch_index = epoch
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))

        logging.info("Training finished.")

        # Tower monitoring summary
        if self.use_tower_specific_monitoring:
            summary = self.get_tower_monitoring_summary()
            for name, info in summary.items():
                logging.info(f"  {name}: best {info['metric']}={info['best_value']:.6f} "
                             f"at epoch {info['best_epoch']}")

        # Load best models
        if self.use_tower_specific_monitoring and self.save_tower_models:
            self._load_tower_optimal_models()
        else:
            logging.info("Load best model: {}".format(self.checkpoint))
            self.load_weights(self.checkpoint)
            if not self._save_checkpoints:
                if os.path.exists(self.checkpoint):
                    logging.info("Remove checkpoint: {}".format(self.checkpoint))
                    os.remove(self.checkpoint)

    def eval_step(self):
        logging.info('Evaluation @epoch {} - batch {}: '.format(
            self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.checkpoint_and_earlystop(val_logs)
        # Tower monitoring update
        if self.use_tower_specific_monitoring:
            self.update_tower_monitoring(val_logs, self._epoch_index + 1)
            if self.should_early_stop_towers():
                self._stop_training = True
                logging.info("Early stopping triggered by tower-specific monitoring")
                return
        self.train()
