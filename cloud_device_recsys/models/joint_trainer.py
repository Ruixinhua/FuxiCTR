# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

import logging
import torch
from fuxictr.pytorch.models import BaseModel
from model_zoo.CL.src.base import ContrastiveLearningBase
from cloud_device_recsys.models.losses import bpr_loss, margin_ranking_loss, softmax_cross_entropy_loss
from fuxictr.pytorch.torch_utils import get_loss


class CloudDeviceJointTrainer(BaseModel, ContrastiveLearningBase):
    """
    Joint Trainer for Cloud (Preranking) and Device (Reranking) Models.
    
    Supports:
    - Joint Training (without CL) -> Training both models simultaneously.
    - Joint Training with Contrastive Learning (CL) -> Cloud model mimics Device model.
    - Pointwise and Pairwise (BPR/Margin) negative sampling training.
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

        # Loss Configs
        self.num_negatives = num_negatives
        self.loss_type = loss_type
        self.margin = margin

        # Negative sampling (injected after model build via set_negative_sampler)
        # Use object.__setattr__ to bypass torch.nn.Module.__setattr__,
        # which would intercept None assignments and break later attribute access.
        object.__setattr__(self, 'negative_sampler', None)
        object.__setattr__(self, '_item_id_col', 'cand_item_id')

        # Compiler
        self.compile(kwargs.get("optimizer", "adam"), kwargs.get("loss", "binary_crossentropy"), learning_rate)

    def set_negative_sampler(self, sampler, item_id_col: str = 'cand_item_id'):
        """Inject a NegativeSampler so train_epoch can build pairwise batches on the fly."""
        object.__setattr__(self, 'negative_sampler', sampler)
        object.__setattr__(self, '_item_id_col', item_id_col)
        logging.info(f"[JointTrainer] Negative sampler set: {self.num_negatives} negatives/positive, loss={self.loss_type}")


    def compile(self, optimizer, loss, lr):
        """Override to pass parameters from both models to the unified optimizer."""
        self.optimizer_name = optimizer
        self.learning_rate = lr
        self.loss_fn = get_loss(loss)
        self.optimizer = self._get_optimizer(optimizer, lr)

    def _get_optimizer(self, optimizer, lr):
        if optimizer.lower() == 'adam':
            # Gather parameters from BOTH models
            params = list(self.preranking_model.parameters()) + list(self.reranking_model.parameters())
            return torch.optim.Adam(params, lr=lr)
        elif optimizer.lower() == 'rmsprop':
            params = list(self.preranking_model.parameters()) + list(self.reranking_model.parameters())
            return torch.optim.RMSprop(params, lr=lr)
        elif optimizer.lower() == 'sgd':
            params = list(self.preranking_model.parameters()) + list(self.reranking_model.parameters())
            return torch.optim.SGD(params, lr=lr)
        else:
            raise NotImplementedError(f"Optimizer {optimizer} not implemented.")

    def forward(self, inputs):
        """
        Forward pass through both preranking (Cloud) and reranking (Device) models.
        """
        # Move all input tensors to the model device.
        # Data loaders yield CPU tensors; sub-models are on GPU.
        target_device = next(self.preranking_model.parameters()).device
        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        self._current_inputs = inputs

        pre_output = self.preranking_model(inputs)
        re_output = self.reranking_model(inputs)

        return_dict = {
            "pre_pred": pre_output.get("y_pred"),
            "re_pred": re_output.get("y_pred"),
        }

        # Extract features for CL if needed.
        # IMPORTANT: filter inputs to each model's own feature set.
        # The batch uses the reranking feature map (all FG1+FG2+FG3 features), but the
        # preranking embedding layer only knows FG1+FG2 — passing FG3 keys causes a KeyError.
        if self.use_contrastive_learning and self.training:
            if hasattr(self.preranking_model, 'embedding_layer'):
                pre_known = set(self.preranking_model.feature_map.features.keys())
                pre_inputs = {k: v for k, v in inputs.items() if k in pre_known}
                return_dict["pre_emb"] = self.get_feature_embeddings(
                    self.preranking_model.embedding_layer, pre_inputs
                )
            if hasattr(self.reranking_model, 'embedding_layer'):
                re_known = set(self.reranking_model.feature_map.features.keys())
                re_inputs = {k: v for k, v in inputs.items() if k in re_known}
                return_dict["re_emb"] = self.get_feature_embeddings(
                    self.reranking_model.embedding_layer, re_inputs
                )

        return return_dict


    def add_loss(self, return_dict, y_true):
        """
        Compute joint loss: Preranking Task Loss + Reranking Task Loss + (Optional) CL Loss
        """
        target_device = return_dict["pre_pred"].device
        y_true = y_true.to(target_device)

        pre_loss = self.loss_fn(return_dict["pre_pred"], y_true, reduction='mean')
        re_loss = self.loss_fn(return_dict["re_pred"], y_true, reduction='mean')

        base_dual_tower_loss = pre_loss + re_loss

        if not self.use_contrastive_learning:
            return base_dual_tower_loss

        group_ids = self.get_group_ids(self._current_inputs) if hasattr(self, '_current_inputs') else None

        # CL: Make Preranking align with Reranking.
        # A single compute_cl_loss call accumulates all sub-losses:
        #   - KD + group-aware: use re_pred vs pre_pred logits
        #   - Feature alignment / field uniformity: use re_emb (reranking feature embeddings)
        # We pass base_loss=0 to get a pure CL loss, then scale and add.
        cl_loss = self.compute_cl_loss(
            base_loss=torch.tensor(0.0, device=y_true.device),
            feature_embeddings=return_dict.get("re_emb"),  # reranking embeddings for alignment/uniformity
            h1_logits=return_dict["re_pred"],              # Teacher logits
            h2_logits=return_dict["pre_pred"],             # Student logits
            labels=y_true,
            group_ids=group_ids
        )

        total_loss = base_dual_tower_loss + self.cl_loss_weight * cl_loss

        return total_loss
    
    def regularization_loss(self):
        pre_reg = self.preranking_model.regularization_loss() if hasattr(self.preranking_model, 'regularization_loss') else 0
        re_reg = self.reranking_model.regularization_loss() if hasattr(self.reranking_model, 'regularization_loss') else 0
        return pre_reg + re_reg

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        self.train()

        use_negatives = (self.num_negatives > 0
                         and self.negative_sampler is not None
                         and self._item_id_col is not None)

        # Determine iterator type
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            from tqdm import tqdm
            import sys
            batch_iterator = tqdm(data_generator, disable=True, file=sys.stdout)


        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            if use_negatives:
                loss = self._train_step_with_negatives_online(batch_data)
            else:
                loss = self.train_step(batch_data)
            train_loss += loss.item()

        return train_loss / (self._batch_index + 1)

    def train_step(self, batch_data):
        """Pointwise cross-entropy training step."""
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data).to(self.device)
        loss = self.add_loss(return_dict, y_true) + self.regularization_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.preranking_model.parameters()) + list(self.reranking_model.parameters()),
            getattr(self, '_max_gradient_norm', 10.0)
        )
        self.optimizer.step()
        return loss

    def _train_step_with_negatives_online(self, batch_data):
        """
        Online pairwise training step: samples negatives from self.negative_sampler,
        builds neg_batch_dict, then delegates to _train_step_with_negatives.
        Mirrors PrerankingStage._train_step_with_negatives.
        """
        import numpy as np
        batch_dict = {k: v for k, v in batch_data.items()}
        item_id_col = self._item_id_col

        # Sample negatives
        pos_item_ids = batch_dict[item_id_col].cpu().numpy()
        neg_item_ids = self.negative_sampler.sample_negatives_batch(pos_item_ids, self.num_negatives)
        neg_ids_flat = neg_item_ids.reshape(-1)  # [B * num_neg]

        # Get negative item features from sampler
        neg_features_df = self.negative_sampler.get_features_by_ids(neg_ids_flat)

        # Build negative batch dict: replace item features, repeat user features.
        # Note: forward() handles device placement, so we can leave tensors on CPU here
        # and they'll be moved to GPU inside forward(). We keep dtype consistency though.
        model_device = next(self.preranking_model.parameters()).device
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
                # Repeat user features: [B, ...] -> [B * num_neg, ...]
                if hasattr(val, 'to'):
                    val = val.to(model_device)
                    neg_batch_dict[key] = val.repeat_interleave(self.num_negatives, dim=0)
                else:
                    neg_batch_dict[key] = val

        return self._train_step_with_negatives(batch_dict, neg_batch_dict)

    def _train_step_with_negatives(self, pos_batch_data, neg_batch_data):
        """
        Train using pair-wise / negative sampling.
        We do forward pass through both models for both pos and neg batches.
        """
        batch_size = pos_batch_data[list(pos_batch_data.keys())[0]].size(0)
        
        # Positive forward
        pos_output = self.forward(pos_batch_data)
        pre_pos_scores = pos_output["pre_pred"] 
        re_pos_scores = pos_output["re_pred"] 
        
        # Negative forward
        neg_output = self.forward(neg_batch_data)
        pre_neg_scores_flat = neg_output["pre_pred"]
        re_neg_scores_flat = neg_output["re_pred"]
        
        pre_neg_scores = pre_neg_scores_flat.view(batch_size, self.num_negatives)
        re_neg_scores = re_neg_scores_flat.view(batch_size, self.num_negatives)
        
        # Base BPR loss
        if self.loss_type == 'bpr':
            pre_loss = bpr_loss(pre_pos_scores, pre_neg_scores)
            re_loss = bpr_loss(re_pos_scores, re_neg_scores)
        elif self.loss_type == 'margin':
            pre_loss = margin_ranking_loss(pre_pos_scores, pre_neg_scores, margin=self.margin)
            re_loss = margin_ranking_loss(re_pos_scores, re_neg_scores, margin=self.margin)
        elif self.loss_type == 'softmax':
            pre_loss = softmax_cross_entropy_loss(pre_pos_scores, pre_neg_scores)
            re_loss = softmax_cross_entropy_loss(re_pos_scores, re_neg_scores)
        else:
             raise ValueError(f"Unknown loss_type: {self.loss_type}")
             
        base_dual_tower_loss = pre_loss + re_loss

        if not self.use_contrastive_learning:
             loss = base_dual_tower_loss + self.regularization_loss()
             self.optimizer.zero_grad()
             loss.backward()
             torch.nn.utils.clip_grad_norm_(list(self.preranking_model.parameters()) + list(self.reranking_model.parameters()), getattr(self, '_max_gradient_norm', 10.0))
             self.optimizer.step()
             return loss
             
        # Add CL Loss using positive samples only
        # y_true: derive device from pre_pos_scores (already on GPU from forward())
        gpu_device = pre_pos_scores.device
        y_true = self.get_labels(pos_batch_data).to(gpu_device)
        group_ids = self.get_group_ids(pos_batch_data)

        cl_loss = self.compute_cl_loss(
            base_loss=torch.tensor(0.0, device=gpu_device),
            feature_embeddings=pos_output.get("re_emb"),  # reranking embeddings for alignment/uniformity
            h1_logits=pos_output["re_pred"],               # Teacher logits
            h2_logits=pos_output["pre_pred"],              # Student logits
            labels=y_true,
            group_ids=group_ids
        )
        
        total_loss = base_dual_tower_loss + self.cl_loss_weight * cl_loss + self.regularization_loss()
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.preranking_model.parameters()) + list(self.reranking_model.parameters()), getattr(self, '_max_gradient_norm', 10.0))
        self.optimizer.step()
        
        return total_loss
        
    def save_weights(self, prerank_path, rerank_path):
        """Save the updated weights to their respective stage paths."""
        self.preranking_model.save_weights(prerank_path)
        self.reranking_model.save_weights(rerank_path)
