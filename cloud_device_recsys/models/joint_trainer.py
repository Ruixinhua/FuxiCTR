# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
CloudDeviceJointTrainer — nn.Module for simultaneous joint training.

This class handles the low-level forward pass and loss computation when
preranking and reranking models are trained simultaneously (sharing a
single optimizer). Pipeline-level orchestration (sequential training,
evaluation, checkpointing) lives in pipeline/joint_training_stage.py.
"""

import logging
import torch
from fuxictr.pytorch.models import BaseModel
from model_zoo.CL.src.base import ContrastiveLearningBase
from cloud_device_recsys.models.losses import bpr_loss, margin_ranking_loss, softmax_cross_entropy_loss
from fuxictr.pytorch.torch_utils import get_loss


class CloudDeviceJointTrainer(BaseModel, ContrastiveLearningBase):
    """
    Joint Trainer for simultaneous training of Cloud (Preranking) and
    Device (Reranking) models.

    Supports:
    - Simultaneous joint training with a shared optimizer
    - Pointwise (cross-entropy) and Pairwise (BPR / Margin / Softmax) losses
    - Optional Contrastive Learning (preranking aligns with reranking)

    For sequential training or pipeline orchestration, use
    ``pipeline.CloudDeviceJointTrainingStage`` instead.
    """

    def __init__(self,
                 preranking_model,
                 reranking_model,
                 gpu=-1,
                 learning_rate=1e-3,
                 use_contrastive_learning=False,
                 cl_loss_weight=1.0,
                 knowledge_distillation_loss_weight=1.0,
                 group_aware_loss_weight=0.5,
                 feature_alignment_loss_weight=0.0,
                 field_uniformity_loss_weight=0.0,
                 distance_loss_weight=0.0,
                 temperature=4.0,
                 num_negatives=0,
                 loss_type='bpr',
                 margin=1.0,
                 **kwargs):
        # Provide defaults for missing BaseModel kwargs
        kwargs.setdefault('id', 'CloudDeviceJointTrainer')
        kwargs.setdefault('verbose', preranking_model._verbose if hasattr(preranking_model, '_verbose') else 1)
        kwargs.setdefault('model_root', './outputs/joint_model')
        kwargs.setdefault('metrics', getattr(reranking_model, 'validation_metrics', ['AUC']))

        # Initialize BaseModel with reranking feature map (contains FG1, FG2, FG3)
        BaseModel.__init__(self, reranking_model.feature_map, gpu=gpu, **kwargs)
        # Initialize ContrastiveLearningBase
        self.use_contrastive_learning = use_contrastive_learning
        ContrastiveLearningBase.__init__(
            self,
            knowledge_distillation_loss_weight=knowledge_distillation_loss_weight,
            group_aware_loss_weight=group_aware_loss_weight,
            feature_alignment_loss_weight=feature_alignment_loss_weight,
            field_uniformity_loss_weight=field_uniformity_loss_weight,
            distance_loss_weight=distance_loss_weight,
            temperature=temperature,
            **kwargs
        )

        self.cl_loss_weight = cl_loss_weight
        self.preranking_model = preranking_model
        self.reranking_model = reranking_model

        # Loss configs
        self.num_negatives = num_negatives
        self.loss_type = loss_type
        self.margin = margin

        # Negative sampler — injected after model build via set_negative_sampler().
        # Use object.__setattr__ to bypass torch.nn.Module.__setattr__ which would
        # intercept None assignments and break later attribute access.
        object.__setattr__(self, 'negative_sampler', None)
        object.__setattr__(self, '_item_id_col', 'cand_item_id')

        self.compile(kwargs.get("optimizer", "adam"), kwargs.get("loss", "binary_crossentropy"), learning_rate)

    def set_negative_sampler(self, sampler, item_id_col: str = 'cand_item_id'):
        """Inject a NegativeSampler so train_epoch can build pairwise batches on the fly."""
        object.__setattr__(self, 'negative_sampler', sampler)
        object.__setattr__(self, '_item_id_col', item_id_col)
        logging.info(f"[JointTrainer] Negative sampler set: {self.num_negatives} negatives/positive, loss={self.loss_type}")

    def compile(self, optimizer, loss, lr):
        """Override to create a shared optimizer across both models."""
        self.optimizer_name = optimizer
        self.learning_rate = lr
        self.loss_fn = get_loss(loss)
        self.optimizer = self._get_optimizer(optimizer, lr)

    def _get_optimizer(self, optimizer, lr):
        params = list(self.preranking_model.parameters()) + list(self.reranking_model.parameters())
        if optimizer.lower() == 'adam':
            return torch.optim.Adam(params, lr=lr)
        elif optimizer.lower() == 'rmsprop':
            return torch.optim.RMSprop(params, lr=lr)
        elif optimizer.lower() == 'sgd':
            return torch.optim.SGD(params, lr=lr)
        else:
            raise NotImplementedError(f"Optimizer {optimizer} not implemented.")

    # ------------------------------------------------------------------
    # Forward / Loss
    # ------------------------------------------------------------------

    def forward(self, inputs, re_inputs=None):
        """Forward pass through both preranking and reranking models.

        Args:
            inputs: Batch dict for the **preranking** model (FG1/FG2 features).
            re_inputs: Batch dict for the **reranking** model (FG1/FG2/FG3 + optional
                cloud_score).  When ``None`` the same ``inputs`` dict is used for both
                models — this is the legacy behaviour valid only when both models share
                an identical feature set.
        """
        target_device = next(self.preranking_model.parameters()).device
        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        if re_inputs is None:
            re_inputs = inputs
        else:
            re_inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                         for k, v in re_inputs.items()}

        self._current_inputs = inputs

        pre_output = self.preranking_model(inputs)

        # Inject cloud_score into re_inputs if the reranking feature_map expects it.
        # We run the preranking model on the reranking batch's shared features (FG1/FG2)
        # to produce a cloud_score that matches re_inputs' batch size.
        if 'cloud_score' in self.reranking_model.feature_map.features and \
                'cloud_score' not in re_inputs:
            pre_known = set(self.preranking_model.feature_map.features.keys())
            re_pre_inputs = {k: v for k, v in re_inputs.items() if k in pre_known}
            with torch.no_grad():
                re_pre_out = self.preranking_model(re_pre_inputs)
                pre_logit = torch.logit(re_pre_out['y_pred'].detach().squeeze(-1), eps=1e-7)
                mean, std = pre_logit.mean(), pre_logit.std() + 1e-8
                re_inputs['cloud_score'] = (pre_logit - mean) / std

        re_output  = self.reranking_model(re_inputs)

        return_dict = {
            "pre_pred": pre_output.get("y_pred"),
            "re_pred":  re_output.get("y_pred"),
        }

        # Extract embeddings for CL if needed.
        if self.use_contrastive_learning and self.training:
            if hasattr(self.preranking_model, 'embedding_layer'):
                pre_known = set(self.preranking_model.feature_map.features.keys())
                return_dict["pre_emb"] = self.get_feature_embeddings(
                    self.preranking_model.embedding_layer,
                    {k: v for k, v in inputs.items() if k in pre_known}
                )
            if hasattr(self.reranking_model, 'embedding_layer'):
                re_known = set(self.reranking_model.feature_map.features.keys())
                return_dict["re_emb"] = self.get_feature_embeddings(
                    self.reranking_model.embedding_layer,
                    {k: v for k, v in re_inputs.items() if k in re_known}
                )

        return return_dict

    def add_loss(self, return_dict, y_true):
        """Joint loss: preranking task loss + reranking task loss + optional CL loss."""
        target_device = return_dict["pre_pred"].device
        y_true = y_true.to(target_device)

        pre_loss = self.loss_fn(return_dict["pre_pred"], y_true, reduction='mean')
        re_loss  = self.loss_fn(return_dict["re_pred"],  y_true, reduction='mean')
        base_dual_tower_loss = pre_loss + re_loss

        if not self.use_contrastive_learning:
            return base_dual_tower_loss

        group_ids = self.get_group_ids(self._current_inputs) if hasattr(self, '_current_inputs') else None
        cl_loss = self.compute_cl_loss(
            base_loss=torch.tensor(0.0, device=y_true.device),
            feature_embeddings=return_dict.get("re_emb"),
            h1_logits=return_dict["re_pred"],   # Teacher
            h2_logits=return_dict["pre_pred"],  # Student
            labels=y_true,
            group_ids=group_ids
        )
        return base_dual_tower_loss + self.cl_loss_weight * cl_loss

    def regularization_loss(self):
        pre_reg = self.preranking_model.regularization_loss() if hasattr(self.preranking_model, 'regularization_loss') else 0
        re_reg  = self.reranking_model.regularization_loss()  if hasattr(self.reranking_model,  'regularization_loss') else 0
        return pre_reg + re_reg

    # ------------------------------------------------------------------
    # Training loop (simultaneous)
    # ------------------------------------------------------------------

    def train_epoch(self, data_generator, re_data_generator=None):
        """One simultaneous training epoch: both models update every batch.

        Args:
            data_generator: Iterator for **preranking** model batches.
            re_data_generator: Optional iterator for **reranking** model batches.
                When provided each step draws one preranking batch and one
                (independent) reranking batch and feeds them to the respective
                models.  When ``None`` the same preranking batch is used for
                both models — valid only if both models share the same feature set.
        """
        self._batch_index = 0
        train_loss = 0
        self.train()

        use_negatives = (self.num_negatives > 0
                         and self.negative_sampler is not None
                         and self._item_id_col is not None)

        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            from tqdm import tqdm
            import sys
            batch_iterator = tqdm(data_generator, disable=True, file=sys.stdout)

        re_iter = iter(re_data_generator) if re_data_generator is not None else None

        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index

            # Obtain a reranking batch (cycle if shorter than preranking loader)
            re_batch_data = None
            if re_iter is not None:
                try:
                    re_batch_data = next(re_iter)
                except StopIteration:
                    re_iter = iter(re_data_generator)
                    re_batch_data = next(re_iter)

            if use_negatives:
                loss = self._train_step_with_negatives_online(batch_data, re_batch_data)
            else:
                loss = self.train_step(batch_data, re_batch_data)
            train_loss += loss.item()

        return train_loss / (self._batch_index + 1)

    def train_step(self, batch_data, re_batch_data=None):
        """Pointwise cross-entropy simultaneous training step."""
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data, re_batch_data)
        y_true = self.get_labels(batch_data).to(self.device)
        loss = self.add_loss(return_dict, y_true) + self.regularization_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.preranking_model.parameters()) + list(self.reranking_model.parameters()),
            getattr(self, '_max_gradient_norm', 10.0)
        )
        self.optimizer.step()
        return loss

    def _train_step_with_negatives_online(self, batch_data, re_batch_data=None):
        """Online pairwise training step with on-the-fly negative sampling.

        Builds a negative batch for the *preranking* model from ``batch_data``.
        If ``re_batch_data`` is provided it is used as the reranking positive
        batch and a corresponding reranking negative batch is constructed from
        it; otherwise the same preranking batch/neg_batch are reused for the
        reranking model (legacy, only valid when feature sets are identical).
        """
        import numpy as np
        batch_dict = dict(batch_data)
        item_id_col = self._item_id_col
        model_device = next(self.preranking_model.parameters()).device

        # ---- Preranking negatives (built from preranking batch) ----
        pos_item_ids = batch_dict[item_id_col].cpu().numpy()
        neg_item_ids = self.negative_sampler.sample_negatives_batch(pos_item_ids, self.num_negatives)
        neg_ids_flat = neg_item_ids.reshape(-1)
        neg_features_df = self.negative_sampler.get_features_by_ids(neg_ids_flat)

        neg_batch_dict = {}
        for key, val in batch_dict.items():
            if key == item_id_col:
                neg_batch_dict[key] = torch.tensor(neg_ids_flat, device=model_device)
            elif key in neg_features_df.columns:
                col_vals = neg_features_df[key].to_numpy(copy=False)
                if not np.isscalar(col_vals[0]):
                    col_vals = np.vstack(col_vals)
                neg_batch_dict[key] = torch.tensor(col_vals, device=model_device)
            else:
                if hasattr(val, 'to'):
                    val = val.to(model_device)
                    neg_batch_dict[key] = val.repeat_interleave(self.num_negatives, dim=0)
                else:
                    neg_batch_dict[key] = val

        # ---- Reranking negatives (built from re_batch_data if available) ----
        if re_batch_data is not None:
            re_batch_dict = dict(re_batch_data)
            re_pos_item_ids = re_batch_dict[item_id_col].cpu().numpy()
            re_neg_item_ids = self.negative_sampler.sample_negatives_batch(
                re_pos_item_ids, self.num_negatives
            )
            re_neg_ids_flat = re_neg_item_ids.reshape(-1)
            re_neg_features_df = self.negative_sampler.get_features_by_ids(re_neg_ids_flat)

            re_neg_batch_dict = {}
            for key, val in re_batch_dict.items():
                if key == item_id_col:
                    re_neg_batch_dict[key] = torch.tensor(re_neg_ids_flat, device=model_device)
                elif key in re_neg_features_df.columns:
                    col_vals = re_neg_features_df[key].to_numpy(copy=False)
                    if not np.isscalar(col_vals[0]):
                        col_vals = np.vstack(col_vals)
                    re_neg_batch_dict[key] = torch.tensor(col_vals, device=model_device)
                else:
                    if hasattr(val, 'to'):
                        val = val.to(model_device)
                        re_neg_batch_dict[key] = val.repeat_interleave(self.num_negatives, dim=0)
                    else:
                        re_neg_batch_dict[key] = val
        else:
            # Legacy: reuse preranking batches for reranking model
            re_batch_dict = batch_dict
            re_neg_batch_dict = neg_batch_dict

        return self._train_step_with_negatives(
            batch_dict, neg_batch_dict, re_batch_dict, re_neg_batch_dict
        )

    def _train_step_with_negatives(self, pos_batch_data, neg_batch_data,
                                   re_pos_batch_data=None, re_neg_batch_data=None):
        """Pairwise (BPR / Margin / Softmax) simultaneous training step.

        Args:
            pos_batch_data: Positive batch for preranking model.
            neg_batch_data: Negative batch for preranking model.
            re_pos_batch_data: Positive batch for reranking model (may differ in feature set).
                               Falls back to ``pos_batch_data`` when ``None``.
            re_neg_batch_data: Negative batch for reranking model.
                               Falls back to ``neg_batch_data`` when ``None``.
        """
        if re_pos_batch_data is None:
            re_pos_batch_data = pos_batch_data
        if re_neg_batch_data is None:
            re_neg_batch_data = neg_batch_data

        batch_size    = next(iter(pos_batch_data.values())).size(0)
        re_batch_size = next(iter(re_pos_batch_data.values())).size(0)

        pos_output = self.forward(pos_batch_data, re_pos_batch_data)
        neg_output = self.forward(neg_batch_data, re_neg_batch_data)

        pre_pos = pos_output["pre_pred"]
        re_pos  = pos_output["re_pred"]
        pre_neg = neg_output["pre_pred"].view(batch_size, self.num_negatives)
        re_neg  = neg_output["re_pred"].view(re_batch_size, self.num_negatives)

        if self.loss_type == 'bpr':
            pre_loss = bpr_loss(pre_pos, pre_neg)
            re_loss  = bpr_loss(re_pos,  re_neg)
        elif self.loss_type == 'margin':
            pre_loss = margin_ranking_loss(pre_pos, pre_neg, margin=self.margin)
            re_loss  = margin_ranking_loss(re_pos,  re_neg,  margin=self.margin)
        elif self.loss_type == 'softmax':
            pre_loss = softmax_cross_entropy_loss(pre_pos, pre_neg)
            re_loss  = softmax_cross_entropy_loss(re_pos,  re_neg)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        base_loss = pre_loss + re_loss

        if not self.use_contrastive_learning:
            loss = base_loss + self.regularization_loss()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.preranking_model.parameters()) + list(self.reranking_model.parameters()),
                getattr(self, '_max_gradient_norm', 10.0)
            )
            self.optimizer.step()
            return loss

        gpu_device = pre_pos.device
        y_true = self.get_labels(pos_batch_data).to(gpu_device)
        group_ids = self.get_group_ids(pos_batch_data)
        cl_loss = self.compute_cl_loss(
            base_loss=torch.tensor(0.0, device=gpu_device),
            feature_embeddings=pos_output.get("re_emb"),
            h1_logits=pos_output["re_pred"],
            h2_logits=pos_output["pre_pred"],
            labels=y_true,
            group_ids=group_ids
        )
        total_loss = base_loss + self.cl_loss_weight * cl_loss + self.regularization_loss()
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.preranking_model.parameters()) + list(self.reranking_model.parameters()),
            getattr(self, '_max_gradient_norm', 10.0)
        )
        self.optimizer.step()
        return total_loss

    def save_weights(self, prerank_path, rerank_path):
        """Save both models' weights to their respective stage paths."""
        self.preranking_model.save_weights(prerank_path)
        self.reranking_model.save_weights(rerank_path)
