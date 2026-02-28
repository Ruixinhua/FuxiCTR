# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
Joint Training Stage — Pipeline-level orchestrator.

Holds references to PrerankingStage and RerankingStage and coordinates
their training by calling their existing train() methods.

No training logic is duplicated here; all low-level training is delegated
to the stage objects themselves.
"""

import os
import logging
from typing import Optional, Dict, Any

from .stage_output import StageOutput
from ..utils import enrich_stage_output_user_features


class CloudDeviceJointTrainingStage:
    """
    Pipeline-level orchestrator for joint preranking + reranking training.

    Supports two training modes controlled by `training_mode`:

    - ``"simultaneous"``: Both models are trained jointly each epoch via a
      shared ``CloudDeviceJointTrainer`` (nn.Module) that computes a unified
      loss (optionally with contrastive learning).

    - ``"sequential"``: Preranking is trained first (Phase 1) by calling
      ``preranking_stage.train()``, then reranking is trained (Phase 2)
      by calling ``reranking_stage.train()``.  No training logic is
      duplicated — the stages' existing loops are reused directly.

    In both modes, after each phase / epoch, ``_validate_and_checkpoint``
    is called to evaluate both stages on the validation set using the
    fair-AUC approach (reranking AUC computed on 1000-candidate pool,
    Recall@K on preranking-filtered top-100).
    """

    def __init__(
        self,
        preranking_stage,
        reranking_stage,
        joint_params: Dict[str, Any],
        preranking_config: Dict[str, Any],
        reranking_config: Dict[str, Any],
        preranking_train_loader,
        reranking_train_loader,
        fg_manager,
        dataset_config: Dict[str, Any],
        paths: Dict[str, str],
        logger: Optional[logging.Logger] = None,
    ):
        self.preranking_stage = preranking_stage
        self.reranking_stage = reranking_stage
        self.joint_params = joint_params
        self.preranking_config = preranking_config
        self.reranking_config = reranking_config
        self.preranking_train_loader = preranking_train_loader
        self.reranking_train_loader = reranking_train_loader
        self.fg_manager = fg_manager
        self.dataset_config = dataset_config
        self.paths = paths
        self.logger = logger or logging.getLogger(__name__)

        # Training mode
        self.training_mode = joint_params.get('training_mode', 'simultaneous')
        self.freeze_preranking = joint_params.get('freeze_preranking', False)
        self.epochs = joint_params.get('epochs', 10)
        # Epoch counts: read from each stage's model_params, fall back to joint epochs
        self.preranking_epochs = preranking_config.get('model_params', {}).get('epochs', self.epochs)
        self.reranking_epochs  = reranking_config.get('model_params', {}).get('epochs', self.epochs)

        # Monitors
        self.impression_id_col = dataset_config.get('impression_id_col', 'impression_id')

        # Lazy-initialised joint trainer (simultaneous mode only)
        self._joint_trainer = None

        # Negative sampling pool path (may differ from eval item_pool_path)
        self._neg_pool_path = joint_params.get('_neg_pool_path', paths.get('item_pool_path', ''))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, prev_output_valid, prev_output_test=None, run_test=True):
        """Orchestrate joint training and return a dict of final validation + test metrics."""
        metrics = {}
        if self.training_mode == 'sequential':
            final_pre_v, final_re_v = self._train_sequential(prev_output_valid)
        else:
            final_pre_v, final_re_v = self._train_simultaneous(prev_output_valid)

        # Include final validation metrics in output
        if final_pre_v:
            metrics.update({f"prerank_valid_{k}": v for k, v in final_pre_v.items()})
        if final_re_v:
            metrics.update({f"rerank_valid_{k}": v for k, v in final_re_v.items()})

        # Final test evaluation (both modes)
        if run_test and prev_output_test is not None:
            self.logger.info("[JointTraining/Test] Loading best weights...")
            self.preranking_stage.model.load_weights(self.preranking_stage.best_weights_path)
            self.reranking_stage.model.load_weights(self.reranking_stage.best_weights_path)
            # self._reload_item_features()
            self.logger.info("[JointTraining/Test] Preranking step...")
            prev_output_test = self._enrich(prev_output_test, self.paths['test_path'])
            retrieval_enriched_test = self._enrich(prev_output_test, self.paths['test_path'])
            prerank_out_test, prerank_test_metrics = self.preranking_stage.process(
                prev_output_test, compute_metrics=True, load_best=False
            )
            prerank_out_test_enriched = self._enrich(prev_output_test, self.paths['test_path'])
            self.logger.info("[JointTraining/Test] Reranking step (fair AUC on 1000 candidates)...")
            rerank_test_metrics = self.reranking_stage.evaluate(
                retrieval_enriched_test, preranking_output=prerank_out_test_enriched
            )
            metrics.update({f"prerank_test_{k}": v for k, v in prerank_test_metrics.items()})
            metrics.update({f"rerank_test_{k}": v for k, v in rerank_test_metrics.items()})
        elif run_test:
            self.logger.warning("[JointTraining/Test] No prev_output_test provided. Skipping test.")

        return metrics

    # ------------------------------------------------------------------
    # Training modes
    # ------------------------------------------------------------------

    def _train_sequential(self, prev_output_valid):
        """
        Phase 1: train preranking via PrerankingStage.train() with its own loader.
        Phase 2: train reranking via RerankingStage.train() with its own loader.

        Training kwargs (patience, learning_rate, batch_size, etc.) are read
        from preranking_config['training'] and reranking_config['training']
        respectively, which are populated by the HP search runner via the
        normal pipeline config pathway.
        """
        _explicit = {'epochs', 'train_data', 'valid_data'}

        # Gather per-stage training kwargs from stage config `training` block
        preranking_train_cfg = dict(self.preranking_config.get('training', {}))
        reranking_train_cfg  = dict(self.reranking_config.get('training', {}))
        # Remove any keys that are passed explicitly to avoid duplicate-kwarg errors
        preranking_train_cfg = {k: v for k, v in preranking_train_cfg.items() if k not in _explicit}
        reranking_train_cfg  = {k: v for k, v in reranking_train_cfg.items()  if k not in _explicit}
        best_pre, best_re = -float('inf'), -float('inf')
        # prev_output_valid = self._enrich(prev_output_valid, self.paths['valid_path'])  # enrich once and reuse for both phases
        # ---- Phase 1: Preranking ----
        self.logger.info(
            f"[Sequential] Phase 1: Training Preranking for up to {self.preranking_epochs} epochs..."
        )
        self.preranking_stage.train(
            train_data=self.preranking_train_loader.make_iterator()[0],
            valid_data=prev_output_valid,
            epochs=self.preranking_epochs,
            **preranking_train_cfg,
        )
        self.logger.info("[Sequential] Phase 1 complete. Evaluating preranking + reranking baseline...")
        best_pre, _, pre1_metrics, re_baseline_metrics = self._validate_and_checkpoint(
            "Phase1/Final", best_pre, -float('inf'), prev_output_valid,
            save_reranking=False  # Reranking not yet trained — log as baseline only
        )
        # Optionally freeze preranking params
        if self.freeze_preranking:
            for p in self.preranking_stage.model.parameters():
                p.requires_grad = False
            self.logger.info("[Sequential] Preranking parameters FROZEN for Phase 2.")

        # ---- Phase 2: Reranking ----
        if self.reranking_stage.use_cloud_score:
            self.reranking_stage.cloud_score_teacher = self.preranking_stage.model
            self.preranking_stage.model.eval()   # teacher in inference mode during Phase 2
            self.logger.info(
                "[Sequential] Phase 2: cloud_score_teacher set to preranking model "
                "(dynamic cloud_score injection enabled)."
            )

        # Generate preranking top-100 output as the validation feed for reranking
        self.logger.info("[Sequential] Generating preranking output for reranking validation...")
        prerank_valid_for_re, _ = self.preranking_stage.process(prev_output_valid, compute_metrics=False, load_best=True)
        prerank_valid_for_re = self._enrich(prerank_valid_for_re, self.paths['valid_path'])

        self.logger.info(
            f"[Sequential] Phase 2: Training Reranking for up to {self.reranking_epochs} epochs "
            f"(preranking frozen={self.freeze_preranking})..."
        )
        self.reranking_stage.train(
            train_data=self.reranking_train_loader.make_iterator()[0],
            valid_data=prerank_valid_for_re,
            epochs=self.reranking_epochs,
            **reranking_train_cfg,
        )
        self.logger.info("[Sequential] Phase 2 complete. Running validation...")
        # Reset best_re so Phase 2's trained model always saves at least once
        _, best_re, final_pre_v, final_re_v = self._validate_and_checkpoint(
            "Phase2/Final", best_pre, -float('inf'), prev_output_valid
        )
        return final_pre_v, final_re_v

    def _train_simultaneous(self, prev_output_valid):
        """
        Both models trained jointly each epoch via CloudDeviceJointTrainer.

        Each model receives batches from its own loader (preranking_train_loader for
        the preranking model, reranking_train_loader for the reranking model) so that
        the feature dimensionalities match each model's feature_map exactly.
        cloud_score is injected into the reranking batch automatically inside
        CloudDeviceJointTrainer.forward() when use_cloud_score is enabled.
        """
        from ..models.joint_trainer import CloudDeviceJointTrainer
        from ..data.negative_sampler import NegativeSampler

        joint_params = self.joint_params
        num_negatives = joint_params.get('pre_num_negatives', joint_params.get('num_negatives', 0))
        cl_weights = joint_params.get('cl_weights', {})

        self._joint_trainer = CloudDeviceJointTrainer(
            preranking_model=self.preranking_stage.model,
            reranking_model=self.reranking_stage.model,
            gpu=self.preranking_stage.model_params.get('gpu', -1),
            learning_rate=joint_params.get('learning_rate', 1e-3),
            use_contrastive_learning=joint_params.get('use_contrastive_learning', False),
            num_negatives=num_negatives,
            loss_type=joint_params.get('loss_type', 'bpr'),
            margin=joint_params.get('margin', 1.0),
            **cl_weights,
        )

        # Setup negative sampler
        if num_negatives > 0 and self.preranking_stage.item_features_df is not None:
            item_id_col = getattr(
                self.preranking_stage.feature_map, 'dataset_config', {}
            ).get('item_id_col', 'cand_item_id')
            neg_sampler = NegativeSampler(self.preranking_stage.item_features_df, item_id_col=item_id_col)
            self._joint_trainer.set_negative_sampler(neg_sampler, item_id_col=item_id_col)
        elif num_negatives > 0:
            self.logger.warning(
                "[Simultaneous] num_negatives > 0 but item pool not loaded. Falling back to pointwise."
            )

        prerank_monitor = self.preranking_stage.model_params.get('monitor', 'Recall@100')
        rerank_monitor = self.reranking_stage.model_params.get('monitor', 'Recall@1')
        self.logger.info(
            f"[Simultaneous] Starting {self.epochs} epochs. "
            f"Preranking monitor: {prerank_monitor} | Reranking monitor: {rerank_monitor}"
        )
        best_pre, best_re = -float('inf'), -float('inf')
        final_pre_v, final_re_v = {}, {}
        for epoch in range(self.epochs):
            self.logger.info(f"--- Epoch {epoch + 1}/{self.epochs} ---")
            # Each model receives batches from its own loader so that the
            # feature dimensionalities match the model's feature_map exactly.
            pre_iter = self.preranking_train_loader.make_iterator()[0]
            re_iter  = self.reranking_train_loader.make_iterator()[0]
            self._joint_trainer.train_epoch(pre_iter, re_data_generator=re_iter)
            best_pre, best_re, final_pre_v, final_re_v = self._validate_and_checkpoint(
                f"Epoch {epoch + 1}/{self.epochs}", best_pre, best_re, prev_output_valid
            )
        return final_pre_v, final_re_v


    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_and_checkpoint(self, label, best_pre, best_re, prev_output_valid,
                                  save_preranking: bool = True,
                                  save_reranking: bool = True):
        """Evaluate both stages, optionally save best weights.

        Args:
            save_preranking: If False, preranking metrics are logged as baseline
                             but weights are NOT saved.
            save_reranking:  If False, reranking metrics are logged as baseline
                             but weights are NOT saved. Use for Phase1/Final
                             where reranking is not yet trained.

        Returns:
            (best_pre, best_re, prerank_v_metrics, rerank_v_metrics)
        """
        prerank_monitor = self.preranking_stage.model_params.get('monitor', 'Recall@100')
        # retrieval_enriched_v = self._enrich(prev_output_valid, self.paths['valid_path'])
        prerank_out_v, prerank_v_metrics = self.preranking_stage.process(
            prev_output_valid, compute_metrics=True, load_best_model=False
        )
        self.logger.info(f"[{label}/Preranking] Valid: {prerank_v_metrics}")
        prerank_out_v_enriched = self._enrich(prerank_out_v, self.paths['valid_path'])
        rerank_monitor = self.reranking_stage.model_params.get('monitor', 'Recall@1')
        # Fair AUC: score reranking on 1000-candidate pool; Recall on preranking top-100
        rerank_v_metrics = self.reranking_stage.evaluate(
            prev_output_valid, preranking_output=prerank_out_v_enriched
        )
        self.logger.info(f"[{label}/Reranking] Valid: {rerank_v_metrics}")

        cur_pre = prerank_v_metrics.get(prerank_monitor, 0)
        cur_re = rerank_v_metrics.get(rerank_monitor, 0)

        if save_preranking and cur_pre > best_pre:
            best_pre = cur_pre
            self.logger.info(f"[Preranking] New best {prerank_monitor}={cur_pre:.4f}. Saving...")
            self.preranking_stage.model.save_weights(self.preranking_stage.best_weights_path)
        elif not save_preranking:
            self.logger.info(
                f"[Preranking] Baseline {prerank_monitor}={cur_pre:.4f} (weights not saved)."
            )
        if save_reranking and cur_re > best_re:
            best_re = cur_re
            self.logger.info(f"[Reranking] New best {rerank_monitor}={cur_re:.4f}. Saving...")
            self.reranking_stage.model.save_weights(self.reranking_stage.best_weights_path)
        elif not save_reranking:
            self.logger.info(
                f"[Reranking] Baseline {rerank_monitor}={cur_re:.4f} (weights not saved)."
            )
        return best_pre, best_re, prerank_v_metrics, rerank_v_metrics

    def _enrich(self, stage_output: StageOutput, split_path: str) -> StageOutput:
        """Add FG3 user features to a stage output (required by reranking model)."""
        if self.fg_manager is not None:
            return enrich_stage_output_user_features(
                stage_output, split_path, self.fg_manager, self.impression_id_col, self.logger
            )
        return stage_output
